// Recurrent layers: Elman RNN (tanh), GRU and LSTM over a token sequence.
//
// The input is a sequence of n tokens (Shape {n, 1, in}, token t's features
// at slots t*B_in + j, see transformer.h); the output is every hidden state
// (Shape {n, 1, hidden}, token stride B) or only the last one (Shape
// {hidden, 1, 1}, compact). The weights follow torch.nn.RNN / GRU / LSTM:
// w_ih [G*hidden][in], w_hh [G*hidden][hidden], b_ih, b_hh [G*hidden] with
// G = 1 (rnn), 3 (gru: gates r, z, n) or 4 (lstm: gates i, f, g, o).
//
// Encrypted evaluation runs the recurrence on all tokens at once
// (Jacobi iteration of the triangular system h_t = f(x_t, h_{t-1})): the
// state ciphertext H holds every token's hidden state, and one step
//
//   H <- cell(U + W_hh shift(H))         (U = W_ih x + b, computed once)
//
// updates them all, where shift moves token t-1's state into token t's slots
// (one linear transform with the 2*hidden-1 diagonals of a token-shifted
// W_hh per gate). After k steps the states of tokens 0..k-1 are exact; the
// others are the states of the same RNN started from a zero state k tokens
// earlier, which are ordinary RNN states (they stay inside the activation
// intervals like the exact ones). n steps give the exact result, the same
// sequential depth as the step-by-step recurrence, but every operation
// works on all tokens in SIMD and nothing depends on the sequence position.
//
// The gates live in G regions of R slots (R = n*B rounded up to a power of
// two): the linear transforms write gate g of token t to slot g*R + t*B + j,
// one Chebyshev series evaluates the sigmoid of every gate at once, and a
// rotation by g*R brings gate g onto the state's slots. The tanh of the LSTM
// candidate g is 2 sigmoid(2x) - 1, so it shares the gates' series (its
// pre-activation must lie in [-gate_a / 2, gate_a / 2]). Products with a
// masked copy keep the state's other slots zero, so nothing outside the
// series' intervals ever reaches a series.
//
//   rnn:  h = tanh(U + W h)                                   1 + pd levels
//   gru:  r, z = sigmoid(U_rz + W_rz h'), n = tanh(U_n + r (W_hn h' + b_hn)),
//         h = n + z (h' - n)                                 2 pd + 4 levels
//   lstm: i, f, o = sigmoid(U + W h'), g = 2 sigmoid(2 .) - 1, c = f c' + i g,
//         h = o tanh(c)                                      2 pd + 4 levels
//
// with h', c' = the previous token's state (a rotation by one token stride)
//
// per step (pd = poly_depth(degree)). The series intervals are folded into
// the weights (no level): the gates' sigmoid on [-gate_a, gate_a], the tanh
// of the RNN (gate_a) and of the GRU candidate / LSTM cell (cell_a). The
// LSTM cell state is kept divided by cell_a, so it is within [-1, 1].
//
// n steps of 5-12 levels do not fit one bootstrapping stretch: when the
// network is bootstrapped (Network::place_bootstraps) the layer refreshes
// its ciphertexts itself whenever the next operations would not fit the
// context's stretch budget. All refreshed values are within [-1, 1] by
// construction (states, gates, series arguments) except the input and its
// projection, whose bounds come from calibrate(). At 128-bit security
// (5-6 levels per stretch at ring 2^16) that is one bootstrap per RNN step
// and three per GRU / LSTM step.
//
// forward_plain() runs the same algorithm on cleartext slot vectors (same
// approximations and iterations), so it matches encrypted inference up to
// the CKKS error.
//
// Training: the Trainer keeps the three slot maps (slot_maps()) as
// encrypted linear nodes and backpropagates through time with the same
// all-tokens Jacobi iteration on the adjoint recurrence (see train.cpp).
#ifndef FHEML_RECURRENT_H_
#define FHEML_RECURRENT_H_

#include <array>
#include <string>
#include <vector>

#include "fheml/layers.h"

namespace fheml {

class Recurrent : public Layer {
 public:
  enum class Cell { kRNN, kGRU, kLSTM };
  static Cell parse_cell(const std::string& name);
  static std::string cell_name(Cell c);
  static int gates(Cell c) { return c == Cell::kRNN ? 1 : c == Cell::kGRU ? 3 : 4; }

  // gate_a, degree: the sigmoid series of the gates on [-gate_a, gate_a]
  // (rnn: the tanh series); cell_a, cell_degree: the tanh series of the GRU
  // candidate / LSTM cell on [-cell_a, cell_a] (unused by rnn). last: output
  // only the final hidden state.
  Recurrent(Cell cell, int n, int in, int hidden, std::vector<double> w_ih, std::vector<double> w_hh,
            std::vector<double> b_ih, std::vector<double> b_hh, double gate_a = 16.0, uint32_t degree = 13,
            double cell_a = 8.0, uint32_t cell_degree = 13, bool last = false);
  std::string type() const override { return "recurrent"; }
  std::string describe() const override;
  Shape configure(const Shape& in) override;
  Layout configure_layout(const Layout& in, Packing packing) override;
  void compile(uint32_t slots) override;
  std::vector<double> forward_plain(const std::vector<double>& x) const override;
  Ctxt forward(const FHEContext& ctx, const Ctxt& x) const override;
  uint32_t depth() const override;
  std::vector<int32_t> rotation_indices() const override;
  bool composes_rotations() const override { return true; }
  size_t min_slots() const override;
  void save(std::ostream& os) const override;

  // Self-bootstrapping (see the top of the file).
  uint32_t level_after(uint32_t d, uint32_t budget) const override;
  uint32_t level_peak(uint32_t d, uint32_t budget) const override;
  bool can_self_bootstrap() const override { return true; }
  void set_self_bootstrapping(bool on) override { self_ = on; }
  bool self_bootstrapping() const override { return self_; }
  uint32_t internal_bootstraps(uint32_t d, uint32_t budget) const override;
  void calibrate(const std::vector<std::vector<double>>& inputs, double headroom) override;

  // Scale folding for Network::place_bootstraps(): an input arriving
  // multiplied by c (w_ih takes 1/c), and the output multiplied by f (the
  // last step's series or products take it).
  void absorb_input_scale(double c);
  void scale_output(double f);

  // Largest |argument| of the gate series and of the cell / candidate series
  // over the inputs and every step (both must stay within 1: the series
  // diverge outside their intervals).
  std::array<double, 2> series_ranges(const std::vector<std::vector<double>>& inputs) const;

  // Training (torch layout): w_ih, w_hh, b_ih, b_hh. The Trainer
  // backpropagates through time on the same slot layout (see train.cpp).
  std::vector<double> params() const override;
  void set_params(const std::vector<double>& p) override;
  bool trainable() const override { return true; }

  // The slot maps the layer evaluates, as entries of params() scaled into
  // the series' units: the input projection of every gate but the GRU
  // candidate's (0), the GRU candidate's input projection (1) and the
  // recurrent map from the previous token's state (2). A bias slot may hold
  // the sum of two parameters (b_ih + b_hh), both at the same scale.
  struct MapEntry {
    size_t row, col;
    int param;
    double scale;
  };
  struct MapBias {
    size_t row;
    int param, pair;  // pair: the second parameter summed into the slot, -1 = none
    double scale;
  };
  struct SlotMap {
    std::vector<MapEntry> w;
    std::vector<MapBias> b;
  };
  std::array<SlotMap, 3> slot_maps() const;
  size_t stride() const { return stride_; }
  size_t region() const { return region_; }
  uint32_t plain_slots() const { return plain_slots_; }
  const std::vector<double>& state_mask() const { return mask_; }
  const std::vector<double>& gate_mask() const { return mask_in_; }
  double hn_scale() const { return hn_scale_; }
  const std::vector<double>& gate_coeffs() const { return gate_c_; }
  const std::vector<double>& cell_coeffs() const { return cell_c_; }

  Cell cell() const { return cell_; }
  int tokens() const { return n_; }
  int input_dim() const { return in_dim_; }
  int hidden() const { return h_; }
  bool last() const { return last_; }
  double gate_a() const { return gate_a_; }
  double cell_a() const { return cell_a_; }
  uint32_t degree() const { return degree_; }
  uint32_t cell_degree() const { return cell_degree_; }
  double out_scale() const { return out_scale_; }
  // Value bounds used when refreshing (0 = not calibrated; see Site).
  const std::vector<double>& bounds() const { return bounds_; }
  void set_bounds(std::vector<double> b);

  // Refresh sites: the input, its projections, the state, the gates'
  // pre-activations, the gates, the LSTM cell state / GRU candidate
  // argument, the candidate and the cell's tanh.
  enum Site { kX, kU, kUN, kH, kPre, kS, kC, kTN, kN, kTC, kSites };

 private:
  template <class Ops>
  typename Ops::V run(Ops& ops, typename Ops::V x) const;
  void build();  // series coefficients, scaled weights and slot vectors
  void build_transforms(uint32_t slots, LinearTransform* x, LinearTransform* xn, LinearTransform* h);
  double series_scale(size_t gate, bool recurrent) const;  // pre-activation -> series units
  struct LevelRun {
    bool fits = false;
    uint32_t out = 0, peak = 0, boots = 0;
  };
  LevelRun plan(uint32_t d, uint32_t budget, bool self) const;
  std::vector<double> zmask_scaled() const {
    std::vector<double> m(mask_);
    for (double& v : m) v *= out_scale_;
    return m;
  }

  Cell cell_;
  int n_, in_dim_, h_;
  std::vector<double> w_ih_, w_hh_, b_ih_, b_hh_;
  double gate_a_, cell_a_;
  uint32_t degree_, cell_degree_;
  bool last_;
  double out_scale_ = 1.0;
  bool self_ = false;
  std::vector<double> bounds_ = std::vector<double>(kSites, 0.0);

  // Layout: token stride B, region size R, slots of the cleartext reference.
  size_t stride_ = 0, region_ = 0;
  uint32_t plain_slots_ = 0, slots_ = 0;
  // Series in t in [-1, 1]: gates (sigmoid, or tanh for rnn) and cell / candidate tanh.
  std::vector<double> gate_c_, cell_c_;
  // Scaled weights: the slot matrices of the input projection (all gates but
  // the GRU candidate), the GRU candidate's input projection and the
  // recurrent map, with their biases per slot.
  std::vector<double> bias_x_, bias_xn_, bias_h_;
  std::vector<double> mask_;      // 1 on the state's slots
  std::vector<double> mask_in_;   // LSTM: i gate mask / cell_a; GRU: r gate mask / hn scale
  std::vector<double> rh_b_;      // GRU first step: r mask times b_hn (scaled)
  double hn_scale_ = 1.0;         // GRU: W_hn h + b_hn is divided by this in the gates' ciphertext
  LinearTransform plain_x_, plain_xn_, plain_h_;
  LinearTransform lt_x_, lt_xn_, lt_h_;
};

}  // namespace fheml

#endif  // FHEML_RECURRENT_H_
