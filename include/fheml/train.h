// Encrypted training: stochastic gradient descent with backpropagation, run
// entirely on ciphertexts (inputs, targets, weights, activations, gradients).
//
// Every linear layer is kept as its generalized diagonals, each one an
// encrypted slot vector D_d (no baby-step/giant-step pre-rotation, so the
// diagonals can be updated in place):
//
//   forward   y      = sum_d D_d (*) rot_d(x) + B
//   backward  dL/dx  = sum_d rot_{-d}(D_d (*) g)
//   gradient  dL/dD_d = g (*) rot_d(x),  dL/dB = g
//
// A weight that occupies a single slot (dense layers) gets exactly that
// slot's gradient. A shared weight (a conv kernel tap, a per-channel or
// per-token bias, a token-shared dense weight) gets the sum over all its slots:
// a masked rotate-and-sum, which also broadcasts the sum back to every slot.
// Activations backpropagate through their derivative polynomial, evaluated
// under encryption. The loss is 0.5 * |y - t|^2 against an encrypted target
// (so dL/dy = y - t), or softmax cross-entropy (see Loss). Gradients are
// summed over a batch and applied in one update, after which the weights are
// bootstrapped so the next step starts
// fresh; without bootstrapping only as many updates fit as the depth allows.
//
// Layer norms train gamma and beta (shared over the tokens like a conv tap)
// and backpropagate through the normalisation with the forward's 1/sqrt;
// attention trains its four projections and backpropagates through the
// scores (quadratic or softmax) with the forward's rotated keys and values.
//
// The layers before the first trainable layer (`frozen`) only run forward,
// with their ordinary encrypted evaluation, so fine-tuning the last layers
// of any network is the same API.
//
// This is slow: every weight diagonal is a ciphertext, and each update
// bootstraps all of them. It exists to show the full pipeline works.
#ifndef FHEML_TRAIN_H_
#define FHEML_TRAIN_H_

#include <memory>
#include <string>
#include <vector>

#include "fheml/network.h"
#include "fheml/transformer.h"

namespace fheml {

// Training loss against an encrypted target t.
//
// kMSE: 0.5 |y - t|^2, dL/dy = y - t (no levels).
//
// kCrossEntropy: -sum_i t_i log softmax(y)_i for a probability-vector
// (normally one-hot) target, dL/dy = softmax(y) - t. The softmax runs under
// encryption with the approximations of softmax attention (SoftmaxOptions):
// exp is a Chebyshev series on [lo, hi] (the range of the logits), masked to
// the output slots, summed over all slots (a rotate-and-sum, so every slot
// holds the sum), inverted with a Chebyshev series of 1/x on
// [sum_lo, sum_hi] (0, 0 = k e^lo .. k e^hi for k outputs) plus `newton`
// Newton steps. Levels: pd(exp_degree) + 2 + pd(inv_degree) + 1 + 2 newton
// + 1, plus one when the last layer is not linear (its padding slots are
// masked first).
//
// subtract_max: the largest logit is found with scheme-switching comparisons
// (FHEContext::compare; a max tree over log2 of the output span, so 4
// comparisons for 10 outputs) and subtracted first; then exp lives on
// [lo - hi, 0] and the sum on [1, k], so no calibration is needed. Costs about
// compare_depth() + log2(span) + 2 levels on top and the comparisons' time;
// the context then needs scheme switching (context_options sets it).
//
// The loss value itself is not computed under encryption: the client
// decrypts the returned logits and evaluates it (cross_entropy_plain).
struct Loss {
  enum Kind { kMSE, kCrossEntropy };
  Kind kind = kMSE;
  SoftmaxOptions softmax;  // kCrossEntropy only

  static Loss mse() { return Loss{}; }
  static Loss cross_entropy(const SoftmaxOptions& o = SoftmaxOptions{}) {
    Loss l;
    l.kind = kCrossEntropy;
    l.softmax = o;
    return l;
  }
};

// The softmax Trainer computes under encryption for cross-entropy (same
// polynomials), and the exact -sum t log softmax(y).
std::vector<double> softmax_plain(const Loss& loss, const std::vector<double>& y);
double cross_entropy_plain(const std::vector<double>& y, const std::vector<double>& t);

class Trainer {
 public:
  // `net` must be compiled and outlive the trainer. `frozen` lists top-level
  // layer indices whose weights stay fixed. From the first trainable layer on
  // the network may contain linear layers, square / poly / chebyshev
  // activations, layernorm, attention, bootstrap and residual layers.
  explicit Trainer(Network& net, std::vector<size_t> frozen = {}, Loss loss = Loss{});
  ~Trainer();
  Trainer(const Trainer&) = delete;
  Trainer& operator=(const Trainer&) = delete;

  // Levels one training step (frozen prefix, forward, backward and update)
  // consumes, counted from the weights' level.
  uint32_t step_depth() const;
  // Rotation keys to generate. ladder = false: one key per rotation the step
  // uses (every diagonal offset and its negative, the rotate-and-sum steps),
  // which is heavy at 128-bit security. ladder = true: only the hoisted
  // rotations of the frozen prefix; everything else is composed from the
  // context's rotation-key ladder (set ContextOptions::rotation_window).
  std::vector<int32_t> rotation_indices(bool ladder = false) const;
  // CKKS parameters for training. Without bootstrapping, `updates` SGD
  // updates fit before the weights run out of levels.
  // With rotation_window > 0 the context uses a rotation-key ladder (see
  // rotation_indices(true)).
  ContextOptions context_options(bool secure = true, bool bootstrapping = true, uint32_t updates = 1,
                                 uint32_t rotation_window = 0) const;

  // Encrypts the network's current weights under ctx's public key.
  void encrypt_weights(const FHEContext& ctx);
  Ctxt encrypt_input(const FHEContext& ctx, const std::vector<double>& x) const;
  Ctxt encrypt_target(const FHEContext& ctx, const std::vector<double>& t) const;

  // Forward and backward pass for one sample; adds its gradient to the
  // batch. Returns the (encrypted) network output.
  Ctxt accumulate(const FHEContext& ctx, const Ctxt& x, const Ctxt& target);
  // w -= lr / n * (sum of the n accumulated gradients), then bootstraps the
  // weights when the context supports it.
  void update(const FHEContext& ctx, double lr);
  // Inference with the current encrypted weights.
  Ctxt forward(const FHEContext& ctx, const Ctxt& x);
  std::vector<double> decrypt_output(const FHEContext& ctx, const Ctxt& y) const;

  // Secret-key holder only: decrypts the trained weights into the network's
  // layers and recompiles the network.
  void decrypt_weights(const FHEContext& ctx);

  size_t weight_ciphertexts() const;
  size_t first_trained_layer() const { return first_; }
  // Highest level of any weight ciphertext (for tests and planning).
  uint32_t weight_level() const;
  std::string summary() const;
  const Loss& loss() const { return loss_; }

  struct Node;
  struct CrossEntropy;

 private:
  Network& net_;
  size_t first_ = 0;
  uint32_t slots_ = 0;
  std::vector<std::unique_ptr<Node>> nodes_;
  size_t batch_ = 0;
  uint32_t base_level_ = 0;  // level weights and inputs are encrypted at
  bool encrypted_ = false;
  Loss loss_;
  std::unique_ptr<CrossEntropy> ce_;

  // dL/dy for the network output y.
  Ctxt loss_gradient(const FHEContext& ctx, const Ctxt& y, const Ctxt& target) const;
};

// Plaintext reference of Trainer: one SGD update on a batch with the same
// loss, the same polynomials and the same weight sharing. Updates the
// network's layers in place (compile() again before encrypted inference).
void train_plain(Network& net, const std::vector<std::vector<double>>& xs, const std::vector<std::vector<double>>& ts,
                 double lr, const std::vector<size_t>& frozen = {}, const Loss& loss = Loss{});

}  // namespace fheml

#endif  // FHEML_TRAIN_H_
