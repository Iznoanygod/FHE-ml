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
// attention (Zimerman et al. ICML 2024; Rho et al. ICLR 2025). LayerNorm is
// left out: the toy models train fine without it, and an encrypted inverse
// square root would cost ~10 levels.
//
// Encrypted QK^T and AV use the diagonal method over token offsets s:
// for each s, K and V are rotated by s*d (a copy appended at [n*d, 2n*d)
// makes the rotation cyclic over tokens), Q (*) K_s is summed inside each
// d-block with log2(d) rotations, the masked score polynomial is applied,
// the result is broadcast back over the block and multiplied with V_s.
// Cost: about n * (2 log2(d) rotations + 3 ct-ct products); depth 6 for the
// whole attention block including the four projections.
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
  Layout configure_layout(const Layout& in, bool gapped) override;
  SparseMatrix matrix() const override;
  std::vector<double> bias() const override;
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

// y = body(x) + x.
class Residual : public Layer {
 public:
  explicit Residual(std::vector<std::shared_ptr<Layer>> body);
  std::string type() const override { return "residual"; }
  std::string describe() const override;
  Shape configure(const Shape& in) override;
  Layout configure_layout(const Layout& in, bool gapped) override;
  void compile(uint32_t slots) override;
  std::vector<double> forward_plain(const std::vector<double>& x) const override;
  Ctxt forward(const FHEContext& ctx, const Ctxt& x) const override;
  uint32_t depth() const override;
  std::vector<int32_t> rotation_indices() const override;
  size_t min_slots() const override;
  void encrypt_weights(const FHEContext& ctx, bool drop_plaintext) override;
  void save(std::ostream& os) const override;
  const std::vector<std::shared_ptr<Layer>>& body() const { return body_; }

 private:
  std::vector<std::shared_ptr<Layer>> body_;
};

// Single-head self-attention with a quadratic score function (see top).
class Attention : public Layer {
 public:
  // w*: [d][d] as [out][in]; b*: empty or [d]; score = {c0, c1, c2}.
  Attention(int n, int d, std::vector<double> wq, std::vector<double> wk, std::vector<double> wv,
            std::vector<double> wo, std::vector<double> bq, std::vector<double> bk, std::vector<double> bv,
            std::vector<double> bo, std::vector<double> score);
  std::string type() const override { return "attention"; }
  std::string describe() const override;
  Shape configure(const Shape& in) override;
  Layout configure_layout(const Layout& in, bool gapped) override;
  void compile(uint32_t slots) override;
  std::vector<double> forward_plain(const std::vector<double>& x) const override;
  Ctxt forward(const FHEContext& ctx, const Ctxt& x) const override;
  uint32_t depth() const override { return 6; }
  std::vector<int32_t> rotation_indices() const override;
  void encrypt_weights(const FHEContext& ctx, bool drop_plaintext) override;
  void save(std::ostream& os) const override;
  // The cyclic copy of K and V needs 2*n*d slots.
  size_t min_slots() const override;

 private:
  int n_, d_;
  int stride_ = 0;                   // token stride B of the input layout
  std::vector<double> w_[4], b_[4];  // q, k, v, o
  std::vector<double> score_;
  std::shared_ptr<TokenDense> proj_[4];
  uint32_t slots_ = 0;
  std::vector<double> mask_c2_, mask_c1_, mask_c0_;
};

}  // namespace fheml

#endif  // FHEML_TRANSFORMER_H_
