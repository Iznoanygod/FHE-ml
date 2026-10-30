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
// (so dL/dy = y - t). Gradients are summed over a batch and applied in one
// update, after which the weights are bootstrapped so the next step starts
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

namespace fheml {

class Trainer {
 public:
  // `net` must be compiled and outlive the trainer. `frozen` lists top-level
  // layer indices whose weights stay fixed. From the first trainable layer on
  // the network may contain linear layers, square / poly / chebyshev
  // activations, layernorm, attention, bootstrap and residual layers.
  explicit Trainer(Network& net, std::vector<size_t> frozen = {});
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

  struct Node;

 private:
  Network& net_;
  size_t first_ = 0;
  uint32_t slots_ = 0;
  std::vector<std::unique_ptr<Node>> nodes_;
  size_t batch_ = 0;
  uint32_t base_level_ = 0;  // level weights and inputs are encrypted at
  bool encrypted_ = false;
};

// Plaintext reference of Trainer: one SGD update on a batch with the same
// loss, the same polynomials and the same weight sharing. Updates the
// network's layers in place (compile() again before encrypted inference).
void train_plain(Network& net, const std::vector<std::vector<double>>& xs, const std::vector<std::vector<double>>& ts,
                 double lr, const std::vector<size_t>& frozen = {});

}  // namespace fheml

#endif  // FHEML_TRAIN_H_
