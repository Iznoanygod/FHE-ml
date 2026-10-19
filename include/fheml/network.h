// Sequential network: plaintext reference inference, encrypted inference and
// CKKS parameter planning.
//
// Model file format (text, whitespace separated; written by tools/train_mnist.py):
//
//   fheml 1
//   input C H W
//   conv2d IN_C OUT_C KH KW STRIDE PAD HAS_BIAS   followed by weights [, bias]
//   dense IN OUT HAS_BIAS                         followed by weights [, bias]
//   avgpool2d K STRIDE
//   affine C                                      followed by C scales, C shifts
//   square
//   poly N                                        followed by N coefficients c0..c{N-1}
//   chebyshev FN A B DEGREE                       FN in relu|silu|gelu|sigmoid|tanh
//   bootstrap
//   end
#ifndef FHEML_NETWORK_H_
#define FHEML_NETWORK_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "fheml/context.h"
#include "fheml/layers.h"

namespace fheml {

class Network {
 public:
  Network() = default;
  explicit Network(Shape input) : input_(input) {}

  static Network load(const std::string& path);
  void save(const std::string& path) const;

  Network& add(std::unique_ptr<Layer> layer);
  template <typename L, typename... Args>
  Network& emplace(Args&&... args) {
    return add(std::unique_ptr<Layer>(new L(std::forward<Args>(args)...)));
  }

  // Propagates shapes. Called automatically by add().
  Shape configure();
  // Merges runs of adjacent linear layers into one matrix each: saves one
  // level per merged layer at the cost of more diagonals.
  void fuse_linear();
  // Lowers every linear layer to a BSGS plan for `slots` (0 = required_slots()).
  void compile(uint32_t slots = 0);

  // Smallest power-of-two slot count that holds every activation.
  uint32_t required_slots() const;
  // Levels consumed by the whole network / by the longest stretch between bootstraps.
  uint32_t depth() const;
  uint32_t depth_between_bootstraps() const;
  bool uses_bootstrapping() const;
  std::vector<int32_t> rotation_indices() const;

  // Level to encrypt inputs at: with bootstrapping the first segment only
  // needs depth_before_first_bootstrap() levels, and OpenFHE only refreshes
  // ciphertexts that have fewer levels left than a bootstrap provides.
  uint32_t input_level(const FHEContext& ctx) const;
  uint32_t depth_before_first_bootstrap() const;

  // Suggested CKKS parameters for the compiled network.
  ContextOptions context_options(bool secure = true) const;

  std::vector<double> forward_plain(const std::vector<double>& x) const;

  // Optional per-layer hook: (layer index, layer, output ciphertext, seconds).
  using LayerHook = std::function<void(size_t, const Layer&, const Ctxt&, double)>;
  Ctxt forward(const FHEContext& ctx, const Ctxt& x, const LayerHook& hook = nullptr) const;

  const Shape& input_shape() const { return input_; }
  Shape output_shape() const { return layers_.empty() ? input_ : layers_.back()->out_shape(); }
  const std::vector<std::unique_ptr<Layer>>& layers() const { return layers_; }
  uint32_t slots() const { return slots_; }
  std::string summary() const;

 private:
  Shape input_;
  std::vector<std::unique_ptr<Layer>> layers_;
  uint32_t slots_ = 0;
};

}  // namespace fheml

#endif  // FHEML_NETWORK_H_
