// Network layers. Logically, activations are flattened channel-major (C, H, W),
// the same order as torch.flatten. In the ciphertext, element (c, y, x) lives
// in slot c*cs + y*ys + x*xs (a strided Layout). Strided convolutions and
// pooling keep their outputs "gapped" at the input positions instead of
// compacting them, which keeps their diagonals constant (k*k*(Ci+Co-1) of
// them); the next linear layer absorbs the gaps into its matrix. Every linear
// layer is lowered to a LinearTransform.
#ifndef FHEML_LAYERS_H_
#define FHEML_LAYERS_H_

#include <algorithm>
#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "fheml/context.h"
#include "fheml/linear.h"

namespace fheml {

struct Shape {
  int c = 1, h = 1, w = 1;
  size_t size() const { return static_cast<size_t>(c) * h * w; }
  std::string str() const;
};

struct Layout {
  size_t cs = 0, ys = 0, xs = 0;
  static Layout compact(const Shape& s) { return Layout{static_cast<size_t>(s.h) * s.w, static_cast<size_t>(s.w), 1}; }
  size_t slot(size_t c, size_t y, size_t x) const { return c * cs + y * ys + x * xs; }
  size_t slot(const Shape& s, size_t logical) const;
  // Number of slots needed: highest slot used + 1.
  size_t span(const Shape& s) const { return slot(s.c - 1, s.h - 1, s.w - 1) + 1; }
  bool is_compact(const Shape& s) const { return cs == static_cast<size_t>(s.h) * s.w && ys == static_cast<size_t>(s.w) && xs == 1; }
};

// Multiplicative depth OpenFHE uses for a polynomial of the given degree
// (Paterson-Stockmeyer for EvalPoly / EvalChebyshevSeries).
uint32_t poly_depth(uint32_t degree);

class Layer {
 public:
  virtual ~Layer() = default;
  virtual std::string type() const = 0;
  virtual std::string describe() const { return type(); }

  // Fixes the input shape, returns the output shape.
  virtual Shape configure(const Shape& in) = 0;
  // Picks the slot layout of the output given the input's (after configure).
  // gapped = allow strided outputs (see the top of this file).
  virtual Layout configure_layout(const Layout& in, bool gapped) {
    (void)gapped;
    return in_layout_ = out_layout_ = in;
  }
  // Lowers the layer for a given slot count (after configure_layout).
  virtual void compile(uint32_t slots) { (void)slots; }

  virtual std::vector<double> forward_plain(const std::vector<double>& x) const = 0;
  virtual Ctxt forward(const FHEContext& ctx, const Ctxt& x) const = 0;

  virtual uint32_t depth() const = 0;
  virtual std::vector<int32_t> rotation_indices() const { return {}; }
  // Slots this layer needs (after configure_layout).
  virtual size_t min_slots() const { return std::max(in_layout_.span(in_), out_layout_.span(out_)); }
  // Encrypted-weights mode (see Network::encrypt_weights). No-op for layers
  // without trained weights.
  virtual void encrypt_weights(const FHEContext& ctx, bool drop_plaintext) {
    (void)ctx;
    (void)drop_plaintext;
  }
  virtual bool is_linear() const { return false; }
  virtual bool is_bootstrap() const { return false; }

  // One-line header followed by the parameters (see network.h for the format).
  virtual void save(std::ostream& os) const = 0;

  const Shape& in_shape() const { return in_; }
  const Shape& out_shape() const { return out_; }
  const Layout& in_layout() const { return in_layout_; }
  const Layout& out_layout() const { return out_layout_; }

 protected:
  Shape in_, out_;
  Layout in_layout_, out_layout_;
};

// y = M x + b. Subclasses only describe M and b.
class LinearLayer : public Layer {
 public:
  bool is_linear() const override { return true; }
  uint32_t depth() const override { return 1; }
  Layout configure_layout(const Layout& in, bool gapped) override;
  void compile(uint32_t slots) override;
  std::vector<double> forward_plain(const std::vector<double>& x) const override;
  Ctxt forward(const FHEContext& ctx, const Ctxt& x) const override;
  std::vector<int32_t> rotation_indices() const override { return lt_.rotation_indices(); }
  std::string describe() const override;

  // Logical matrix (out_shape().size() x in_shape().size()) and bias.
  virtual SparseMatrix matrix() const = 0;
  virtual std::vector<double> bias() const { return {}; }
  // Spatial stride applied to the input layout when outputs stay gapped;
  // 0 = this layer always produces a compact output.
  virtual int gap_stride() const { return 0; }

  const LinearTransform& transform() const { return lt_; }
  // Encrypts this layer's weight diagonals and bias. Must follow one forward()
  // with clear weights so the input/output levels are known (see
  // Network::encrypt_weights).
  void encrypt_weights(const FHEContext& ctx, bool drop_plaintext) override;
  bool weights_encrypted() const { return lt_.weights_encrypted(); }
  // Forbid the folded (hybrid) lowering, which leaves partial sums in the
  // slots past the output. Needed when the output must be clean (attention).
  void set_allow_fold(bool on) { allow_fold_ = on; }
  // Overrides the output layout chosen by configure_layout (any linear map can
  // write its outputs anywhere). Used by Residual to line up y with x.
  void set_out_layout(const Layout& l) { out_layout_ = l; }
  void set_cache_plaintexts(bool on) {
    cache_ = on;
    lt_.set_cache_plaintexts(on);
  }

 private:
  LinearTransform lt_;
  SparseMatrix m_;
  std::vector<double> b_;       // logical
  std::vector<double> b_slots_; // bias placed in output slots
  Ctxt enc_bias_;
  mutable uint32_t last_out_level_ = 0;
  bool cache_ = false;
  bool allow_fold_ = true;
};

class Conv2d : public LinearLayer {
 public:
  // weights: [out_c][in_c][kh][kw] flattened; bias: [out_c] (may be empty).
  Conv2d(int in_c, int out_c, int kh, int kw, int stride, int pad, std::vector<double> weights,
         std::vector<double> bias);
  std::string type() const override { return "conv2d"; }
  std::string describe() const override;
  Shape configure(const Shape& in) override;
  SparseMatrix matrix() const override;
  std::vector<double> bias() const override;
  int gap_stride() const override { return stride_; }
  void save(std::ostream& os) const override;

 private:
  int in_c_, out_c_, kh_, kw_, stride_, pad_;
  std::vector<double> w_, b_;
};

class Dense : public LinearLayer {
 public:
  // weights: [out][in] flattened; bias: [out] (may be empty).
  Dense(int in, int out, std::vector<double> weights, std::vector<double> bias);
  std::string type() const override { return "dense"; }
  std::string describe() const override;
  Shape configure(const Shape& in) override;
  SparseMatrix matrix() const override;
  std::vector<double> bias() const override;
  void save(std::ostream& os) const override;

 private:
  int n_in_, n_out_;
  std::vector<double> w_, b_;
};

class AvgPool2d : public LinearLayer {
 public:
  AvgPool2d(int k, int stride);
  std::string type() const override { return "avgpool2d"; }
  Shape configure(const Shape& in) override;
  SparseMatrix matrix() const override;
  int gap_stride() const override { return stride_; }
  void save(std::ostream& os) const override;

 private:
  int k_, stride_;
};

// Per-channel affine map (inference-time batch norm): y = scale[c] * x + shift[c].
class ChannelAffine : public LinearLayer {
 public:
  ChannelAffine(std::vector<double> scale, std::vector<double> shift);
  std::string type() const override { return "affine"; }
  Shape configure(const Shape& in) override;
  SparseMatrix matrix() const override;
  std::vector<double> bias() const override;
  int gap_stride() const override { return 1; }
  void save(std::ostream& os) const override;

 private:
  std::vector<double> scale_, shift_;
};

// Product of consecutive linear layers, produced by Network::fuse_linear().
class FusedLinear : public LinearLayer {
 public:
  FusedLinear(Shape in, Shape out, SparseMatrix m, std::vector<double> b, std::string what);
  std::string type() const override { return "fused"; }
  std::string describe() const override;
  Shape configure(const Shape& in) override;
  SparseMatrix matrix() const override { return m_; }
  std::vector<double> bias() const override { return b_; }
  void save(std::ostream& os) const override;

 private:
  SparseMatrix m_;
  std::vector<double> b_;
  std::string what_;
};

// x -> x^2 (CryptoNets). One level.
class Square : public Layer {
 public:
  std::string type() const override { return "square"; }
  Shape configure(const Shape& in) override { return in_ = out_ = in; }
  std::vector<double> forward_plain(const std::vector<double>& x) const override;
  Ctxt forward(const FHEContext& ctx, const Ctxt& x) const override;
  uint32_t depth() const override { return 1; }
  void save(std::ostream& os) const override;
};

// x -> sum_i coeffs[i] x^i (power basis), e.g. trained polynomial activations.
class PolyActivation : public Layer {
 public:
  explicit PolyActivation(std::vector<double> coeffs);
  std::string type() const override { return "poly"; }
  std::string describe() const override;
  Shape configure(const Shape& in) override { return in_ = out_ = in; }
  std::vector<double> forward_plain(const std::vector<double>& x) const override;
  Ctxt forward(const FHEContext& ctx, const Ctxt& x) const override;
  uint32_t depth() const override;
  void save(std::ostream& os) const override;

 private:
  std::vector<double> c_;
};

// Chebyshev approximation of a named function (relu, silu, gelu, sigmoid,
// tanh) on [a, b] with the given degree. Inputs outside [a, b] diverge, so the
// range must cover the (calibrated) pre-activation values.
class ChebyshevActivation : public Layer {
 public:
  ChebyshevActivation(std::string fn, double a, double b, uint32_t degree);
  std::string type() const override { return "chebyshev"; }
  std::string describe() const override;
  Shape configure(const Shape& in) override { return in_ = out_ = in; }
  std::vector<double> forward_plain(const std::vector<double>& x) const override;
  Ctxt forward(const FHEContext& ctx, const Ctxt& x) const override;
  // OpenFHE maps [a, b] onto [-1, 1] first, which costs one extra level.
  uint32_t depth() const override { return poly_depth(degree_) + (a_ == -1.0 && b_ == 1.0 ? 0 : 1); }
  void save(std::ostream& os) const override;

  static std::function<double(double)> function(const std::string& name);

 private:
  std::string fn_name_;
  std::function<double(double)> fn_;
  double a_, b_;
  uint32_t degree_;
  std::vector<double> coeffs_;
};

// Refreshes the ciphertext's levels with CKKS bootstrapping. Slot values must
// lie roughly in [-1, 1]; scale the preceding linear layer accordingly.
class Bootstrap : public Layer {
 public:
  std::string type() const override { return "bootstrap"; }
  Shape configure(const Shape& in) override { return in_ = out_ = in; }
  std::vector<double> forward_plain(const std::vector<double>& x) const override { return x; }
  Ctxt forward(const FHEContext& ctx, const Ctxt& x) const override { return ctx.bootstrap(x); }
  uint32_t depth() const override { return 0; }
  bool is_bootstrap() const override { return true; }
  void save(std::ostream& os) const override;
};

}  // namespace fheml

#endif  // FHEML_LAYERS_H_
