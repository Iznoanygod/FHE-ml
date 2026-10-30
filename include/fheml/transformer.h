// Transformer building blocks.
//
// A sequence of n tokens with d features is the Shape {n, 1, d}. Feature j of
// token t lives in slot t*B + j, where the token stride B >= d is fixed by the
// network input (pad every token of the input to the widest layer, e.g. 64).
// TokenDense keeps the stride whenever its output fits, so kron(I_n, W) has
// only in + out - 1 diagonals instead of ~n times more.
//
// Attention is fully homomorphic (no client round trip). Softmax is replaced
// by a trainable quadratic score function, A[i][j] = f(q_i . k_j) with
// f(s) = c0 + c1 s + c2 s^2 (normalisation by n and 1/sqrt(d) are folded into
// the coefficients / W_q at training time), in the spirit of polynomial
// attention (Zimerman et al. ICML 2024; Rho et al. ICLR 2025). LayerNorm
// exists (see LayerNorm below, about 10 levels for the inverse square root)
// but the example models do without it.
//
// Encrypted QK^T and AV use the diagonal method over token offsets s:
// for each s, K and V are rotated by s*d (a copy appended at [n*d, 2n*d)
// makes the rotation cyclic over tokens), Q (*) K_s is summed inside each
// d-block with log2(d) rotations, the masked score polynomial is applied,
// the result is broadcast back over the block and multiplied with V_s.
// Cost: about n * (2 log2(d) rotations + 3 ct-ct products); depth 6 for the
// whole attention block including the four projections.
//
// Multi-head attention (H heads of width dh = d/H) uses the same layout: head
// h of token t is the sub-block [t*B + h*dh, t*B + (h+1)*dh), which is what
// the concatenated Q/K/V projections produce anyway. The window sums and the
// broadcasts just shrink from d to dh slots (log2(dh) rotations), the score
// masks pick slot t*B + h*dh for every head, and the head concatenation
// before W_o is free. So H heads cost less than one head of width d: no extra
// ciphertexts, rotations or levels. Each head may have its own score
// polynomial, at no cost.
#ifndef FHEML_TRANSFORMER_H_
#define FHEML_TRANSFORMER_H_

#include <memory>
#include <vector>

#include "fheml/layers.h"

namespace fheml {

// Same dense map applied to every token: {n, 1, in} -> {n, 1, out}.
class TokenDense : public LinearLayer {
 public:
  // weights: [out][in]; bias: empty, [out] (shared by all tokens) or
  // [n][out] (per token, e.g. a positional embedding folded in).
  TokenDense(int n, int in, int out, std::vector<double> weights, std::vector<double> bias);
  std::string type() const override { return "tokendense"; }
  std::string describe() const override;
  Shape configure(const Shape& in) override;
  Layout configure_layout(const Layout& in, Packing packing) override;
  SparseMatrix matrix() const override;
  std::vector<double> bias() const override;
  std::vector<ParamRef> weight_refs() const override;
  std::vector<int> bias_refs() const override;
  std::vector<double> params() const override;
  void set_params(const std::vector<double>& p) override;
  void save(std::ostream& os) const override;

 private:
  int n_, in_dim_, out_dim_;
  std::vector<double> w_, b_;
};

// Mean over tokens: {n, 1, d} -> {d, 1, 1}.
class TokenMean : public LinearLayer {
 public:
  TokenMean(int n, int d);
  std::string type() const override { return "tokenmean"; }
  std::string describe() const override { return type() + LinearLayer::describe(); }
  Shape configure(const Shape& in) override;
  SparseMatrix matrix() const override;
  void save(std::ostream& os) const override;

 private:
  int n_, d_;
};

// Per-token layer normalisation: y = gamma * (x - mean) / sqrt(var + eps) +
// beta over the d features of each token (shape {n, 1, d}, d a power of
// two). Mean and E[x^2] are token-block sums (log2(d) rotations, one masked
// product, log2(d) rotations to broadcast); the inverse square root of the
// variance is a Chebyshev series of 1 / sqrt(v + eps) on [var_min, var_max]
// refined by `newton` Newton steps r <- r (3 - v r^2) / 2 (two levels each,
// the error squares). Degree 15 with one step is exact to ~1e-9 when
// var_max / var_min <= 10; a 100x range wants degree 31. The variance range
// must be calibrated like a Chebyshev activation interval: outside it the
// series diverges. forward_plain() evaluates the same approximation.
// Depth: 2 (E[x^2]) + poly_depth(degree) + 1 + 2 newton + 1.
//
// Self-bootstrapping (set_self_bootstrapping, enabled by
// Network::place_bootstraps when the layer does not fit a stretch): the
// variance is mapped onto t in [-1, 1] (the series' own interval map, done
// by hand) and bootstrapped, so the series costs poly_depth(degree) from a
// fresh ciphertext; 1/sqrt is refreshed again before a Newton step or the
// final product when the budget runs out (Newton then also removes the
// bootstrapping error), and the input itself when it arrives too deep. The
// centred input is never bootstrapped: it waits at its level for the final
// product. 1-3 bootstraps; the input bound comes from calibrate().
class LayerNorm : public Layer {
 public:
  LayerNorm(int n, int d, std::vector<double> gamma, std::vector<double> beta, double eps, double var_min,
            double var_max, uint32_t degree = 15, uint32_t newton = 1);
  std::string type() const override { return "layernorm"; }
  std::string describe() const override;
  Shape configure(const Shape& in) override;
  Layout configure_layout(const Layout& in, Packing packing) override;
  void compile(uint32_t slots) override;
  std::vector<double> forward_plain(const std::vector<double>& x) const override;
  Ctxt forward(const FHEContext& ctx, const Ctxt& x) const override;
  uint32_t depth() const override;
  std::vector<int32_t> rotation_indices() const override;
  void save(std::ostream& os) const override;
  // Scale folding for Network::place_bootstraps(): the input arriving
  // multiplied by c leaves the output unchanged once eps and the variance
  // range are multiplied by c^2; the output is scaled by f through gamma
  // and beta.
  void absorb_input_scale(double c);
  void scale_output(double f);
  // 1 / sqrt(v + eps) as the layer approximates it.
  double inv_sqrt(double v) const;
  // The pieces of forward() for the trainer: x - mean and r = 1/sqrt(var +
  // eps) (broadcast over each token); forward() is gamma (*) centred (*) r +
  // beta. r sits inv_sqrt_depth() levels below x.
  void normalize(const FHEContext& ctx, const Ctxt& x, Ctxt& centred, Ctxt& r) const;
  void normalize_plain(const std::vector<double>& x, std::vector<double>& centred, std::vector<double>& r) const;
  uint32_t inv_sqrt_depth() const { return 2 + poly_depth(degree_) + 1 + 2 * newton_; }
  // Mean over each token's d features, in every slot of the token (one level).
  Ctxt block_mean(const FHEContext& ctx, const Ctxt& c) const;
  // Training: gamma followed by beta.
  std::vector<double> params() const override;
  void set_params(const std::vector<double>& p) override;
  bool trainable() const override { return true; }
  // Self-bootstrapping (see the top of the class).
  uint32_t level_after(uint32_t d, uint32_t budget) const override;
  uint32_t level_peak(uint32_t d, uint32_t budget) const override;
  bool can_self_bootstrap() const override { return true; }
  void set_self_bootstrapping(bool on) override { self_ = on; }
  bool self_bootstrapping() const override { return self_; }
  uint32_t internal_bootstraps(uint32_t d, uint32_t budget) const override;
  void calibrate(const std::vector<std::vector<double>>& inputs, double headroom) override;
  // Bound on |x| used when the input itself has to be refreshed (0 = unknown).
  double input_bound() const { return x_bound_; }
  void set_input_bound(double b) { x_bound_ = b; }
  int tokens() const { return n_; }
  int dim() const { return d_; }
  size_t stride() const { return stride_; }
  double eps() const { return eps_; }
  double var_min() const { return var_min_; }
  double var_max() const { return var_max_; }
  uint32_t degree() const { return degree_; }
  uint32_t newton() const { return newton_; }
  const std::vector<double>& gamma() const { return gamma_; }
  const std::vector<double>& beta() const { return beta_; }

 private:
  int n_, d_;
  std::vector<double> gamma_, beta_;
  double eps_, var_min_, var_max_;
  uint32_t degree_, newton_;
  std::vector<double> coeffs_;  // Chebyshev series on [var_min, var_max]
  size_t stride_ = 0;           // token stride B
  std::vector<double> mask_, gamma_slots_, beta_slots_, pad_var_;  // per slot
  std::vector<double> token_mask_;  // 1 on the tokens' d features
  bool self_ = false;
  double x_bound_ = 0.0;
  struct Plan {
    bool fits = false, plain = true, refresh_x = false;
    std::vector<char> refresh_r;  // before Newton step i; [newton]: before the final product
    uint32_t out = 0, peak = 0, boots = 0;
  };
  Plan plan(uint32_t d, uint32_t budget) const;
  // Upper bound of the (scaled) series output, for refreshing it.
  double inv_sqrt_bound() const;
};

// y = body(x) + x.
class Residual : public Layer {
 public:
  explicit Residual(std::vector<std::shared_ptr<Layer>> body);
  std::string type() const override { return "residual"; }
  std::string describe() const override;
  Shape configure(const Shape& in) override;
  Layout configure_layout(const Layout& in, Packing packing) override;
  void compile(uint32_t slots) override;
  std::vector<double> forward_plain(const std::vector<double>& x) const override;
  Ctxt forward(const FHEContext& ctx, const Ctxt& x) const override;
  uint32_t depth() const override;
  std::vector<int32_t> rotation_indices() const override;
  size_t min_slots() const override;
  void encrypt_weights(const FHEContext& ctx, bool drop_plaintext) override;
  void save(std::ostream& os) const override;
  // Bootstrapped networks: the body is walked from the block's input level
  // (Bootstrap layers in it restart the count); the sum sits at the higher
  // of the two levels.
  uint32_t level_after(uint32_t d, uint32_t budget) const override;
  uint32_t level_peak(uint32_t d, uint32_t budget) const override;
  void set_bootstrapped(bool on) override;
  bool self_bootstrapping() const override;
  uint32_t internal_bootstraps(uint32_t d, uint32_t budget) const override;
  const std::vector<std::shared_ptr<Layer>>& body() const { return body_; }

 private:
  std::vector<std::shared_ptr<Layer>> body_;
};

// Softmax attention scores instead of the quadratic score function (see
// Attention::set_softmax). exp(s) is a Chebyshev series of degree
// `exp_degree` on the score interval; the row sums are inverted with a
// Chebyshev series of 1/x of degree `inv_degree` on the sum interval,
// refined by `newton` steps y <- y (2 - D y) (two levels each).
//
// subtract_max: the maximum score of each row is found with the
// scheme-switching comparison (FHEContext::compare, a tournament over the n
// score ciphertexts, n - 1 comparisons) and subtracted first, the usual
// numerically stable softmax. The exponentials then live on
// [lo - hi, 0] and the row sums on [1, n], so no calibration of the sum is
// needed and nothing can overflow; the price is the comparisons (minutes
// each at 128-bit security) and about 12 + log2(n) extra levels, which puts
// the block beyond one bootstrapping budget at ring 2^16.
// Without it the scores must lie in [lo, hi] and the sums in
// [sum_lo, sum_hi] (0, 0 = n e^lo .. n e^hi; calibrate for a tighter range,
// 1/x is hard to approximate over a wide one).
struct SoftmaxOptions {
  double lo = -4.0, hi = 4.0;
  uint32_t exp_degree = 15;
  uint32_t inv_degree = 15;
  uint32_t newton = 1;
  bool subtract_max = false;
  double sum_lo = 0.0, sum_hi = 0.0;
};

// Self-attention with a quadratic score function (see top) or softmax (see
// SoftmaxOptions); `heads` heads of width d / heads, as sub-blocks of each
// token.
//
// Self-bootstrapping softmax (set_self_bootstrapping, enabled by
// Network::place_bootstraps when the block does not fit a stretch; not with
// subtract_max): the n score ciphertexts are mapped onto [-1, 1] by the
// head mask product (exp's interval map, done by hand), then packed 2*dh to
// a ciphertext (offset s moves to slot t*B + h*dh + s, or into the free
// upper half [n*B, 2n*B)) and bootstrapped together: ceil(n / (2 dh))
// bootstraps instead of n. exp runs once per packed ciphertext, scaled so
// that the row sums land in 1/x's interval directly, and is refreshed
// (values within [-1, 1], no scale level) when the budget requires it;
// unpacking is a rotation plus the head mask product that was needed
// anyway. The row sums are bootstrapped before the 1/x series when it would
// not fit, 1/x before a Newton step or the final product, and the input and
// the unnormalised output when they arrive too deep (bounds from
// calibrate()). Typically 3-5 bootstraps per block.
class Attention : public Layer {
 public:
  // w*: [d][d] as [out][in] (head h reads/writes rows [h*dh, (h+1)*dh) of
  // Wq, Wk, Wv and columns of Wo, like torch's MultiheadAttention); b*: empty
  // or [d]; score = {c0, c1, c2} shared by all heads, or 3*heads values with
  // one polynomial per head.
  Attention(int n, int d, std::vector<double> wq, std::vector<double> wk, std::vector<double> wv,
            std::vector<double> wo, std::vector<double> bq, std::vector<double> bk, std::vector<double> bv,
            std::vector<double> bo, std::vector<double> score, int heads = 1);
  std::string type() const override { return "attention"; }
  std::string describe() const override;
  Shape configure(const Shape& in) override;
  Layout configure_layout(const Layout& in, Packing packing) override;
  void compile(uint32_t slots) override;
  std::vector<double> forward_plain(const std::vector<double>& x) const override;
  Ctxt forward(const FHEContext& ctx, const Ctxt& x) const override;
  uint32_t depth() const override;
  std::vector<int32_t> rotation_indices() const override;
  bool uses_scheme_switching() const override { return softmax_ && sm_.subtract_max; }
  void encrypt_weights(const FHEContext& ctx, bool drop_plaintext) override;
  void save(std::ostream& os) const override;
  // The cyclic copy of K and V needs 2*n*d slots.
  size_t min_slots() const override;
  int heads() const { return heads_; }
  // Projection i (0 q, 1 k, 2 v, 3 o): weights [d][d] and bias (may be empty).
  const std::vector<double>& weights(int i) const { return w_[i]; }
  const std::vector<double>& bias(int i) const { return b_[i]; }
  // Switches the scores to softmax (compile() again afterwards).
  void set_softmax(const SoftmaxOptions& o);
  bool softmax() const { return softmax_; }
  const SoftmaxOptions& softmax_options() const { return sm_; }
  // Intervals the softmax approximations use: exp on [exp_lo, exp_hi], 1/x
  // on [sum_lo, sum_hi].
  double exp_lo() const { return softmax_ && sm_.subtract_max ? sm_.lo - sm_.hi : sm_.lo; }
  double exp_hi() const { return softmax_ && sm_.subtract_max ? 0.0 : sm_.hi; }
  double sum_lo() const;
  double sum_hi() const;
  // Training: the four projections' params() (q, k, v, o) concatenated.
  std::vector<double> params() const override;
  void set_params(const std::vector<double>& p) override;
  bool trainable() const override { return true; }
  // The trainer's view of the layer (after compile()).
  const std::shared_ptr<TokenDense>& projection(int i) const { return proj_[i]; }
  int tokens() const { return n_; }
  int dim() const { return d_; }
  int head_width() const { return dh_; }
  int stride() const { return stride_; }
  // Per-slot score coefficients (quadratic), the head mask and the row-sum
  // padding, the exp and 1/x series (softmax).
  const std::vector<double>& score_mask(int k) const { return k == 0 ? mask_c0_ : k == 1 ? mask_c1_ : mask_c2_; }
  const std::vector<double>& head_mask() const { return mask_head_; }
  const std::vector<double>& pad_sum() const { return pad_sum_; }
  const std::vector<double>& exp_coeffs() const { return exp_coeffs_; }
  const std::vector<double>& inv_coeffs() const { return inv_coeffs_; }
  // Score polynomial coefficient k of head h (quadratic mode).
  double score_coeff(int h, int k) const { return score_[static_cast<size_t>(3 * h + k)]; }
  // 1 / x as the softmax approximates it (series plus Newton steps).
  double inv_plain(double x) const;
  // Self-bootstrapping (see the top of the class).
  uint32_t level_after(uint32_t d, uint32_t budget) const override;
  uint32_t level_peak(uint32_t d, uint32_t budget) const override;
  bool can_self_bootstrap() const override { return softmax_ && !sm_.subtract_max; }
  void set_self_bootstrapping(bool on) override;
  bool self_bootstrapping() const override { return self_; }
  uint32_t internal_bootstraps(uint32_t d, uint32_t budget) const override;
  void calibrate(const std::vector<std::vector<double>>& inputs, double headroom) override;
  // Bounds on |x| and on the unnormalised output sum_s a_s v_s (0 = unknown).
  double input_bound() const { return x_bound_; }
  double output_bound() const { return o_bound_; }
  void set_bounds(double x, double o) {
    x_bound_ = x;
    o_bound_ = o;
  }

 private:
  int n_, d_, heads_, dh_;
  int stride_ = 0;                   // token stride B of the input layout
  std::vector<double> w_[4], b_[4];  // q, k, v, o
  std::vector<double> score_;  // [heads][3]
  std::shared_ptr<TokenDense> proj_[4];
  uint32_t slots_ = 0;
  std::vector<double> mask_c2_, mask_c1_, mask_c0_;
  // Softmax mode.
  bool softmax_ = false;
  SoftmaxOptions sm_;
  std::vector<double> exp_coeffs_, inv_coeffs_;
  std::vector<double> mask_head_, pad_sum_;  // 1 at head slots; sum midpoint off the sub-blocks
  // Self-bootstrapping.
  bool self_ = false;
  double x_bound_ = 0.0, o_bound_ = 0.0;
  std::vector<double> token_mask_;  // 1 on the tokens' d features
  struct Plan {
    bool fits = false, plain = true, refresh_x = false, refresh_e = false, refresh_sum = false, refresh_o = false;
    std::vector<char> refresh_y;  // before Newton step i; [newton]: before the final product
    uint32_t out = 0, peak = 0, boots = 0;
  };
  Plan plan(uint32_t d, uint32_t budget) const;
  Plan plan_with(uint32_t d, uint32_t budget, bool refresh_e, bool refresh_sum) const;
  int packed_groups() const { return (n_ + 2 * dh_ - 1) / (2 * dh_); }
  // Slot shift of offset s inside its packed ciphertext.
  int32_t pack_shift(int s) const;
  // Scaling of the self-bootstrapping softmax: the row sums times alpha
  // land in [-1 - beta, 1 - beta]; exp is divided by exp_div so it can be
  // bootstrapped without a scale level.
  double sum_alpha() const { return 2.0 / (sum_hi() - sum_lo()); }
  double sum_beta() const { return -(sum_hi() + sum_lo()) / (sum_hi() - sum_lo()); }
  double exp_div() const;
  Ctxt forward_self(const FHEContext& ctx, const Ctxt& x, const Plan& p) const;
};

}  // namespace fheml

#endif  // FHEML_TRANSFORMER_H_
