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
//   maxpool2d K STRIDE RANGE                     exact max via scheme switching, inputs in [-RANGE, RANGE]
//   affine C                                      followed by C scales, C shifts
//   square
//   poly N                                        followed by N coefficients c0..c{N-1}
//   chebyshev FN A B DEGREE [scale IN OUT]        FN in relu|silu|gelu|sigmoid|tanh; computes OUT*FN(IN*x)
//   scale C                                       followed by C per-channel factors (fixed, not trained)
//   bootstrap
//   tokendense N IN OUT NB                        followed by weights [OUT][IN], NB bias values (0, OUT or N*OUT)
//   tokenmean N D
//   attention N D HAS_BIAS                        followed by Wq Wk Wv Wo [, bq bk bv bo], 3 score coefficients
//   residual K                                    followed by K layers (y = body(x) + x)
//   end
#ifndef FHEML_NETWORK_H_
#define FHEML_NETWORK_H_

#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "fheml/context.h"
#include "fheml/layers.h"

namespace fheml {

// Parses one layer whose keyword `tok` was just read.
std::shared_ptr<Layer> read_layer(const std::string& tok, std::istream& is);

class Network {
 public:
  Network() = default;
  explicit Network(Shape input) : input_(input) {}

  static Network load(const std::string& path);
  void save(const std::string& path) const;

  Network& add(std::shared_ptr<Layer> layer);
  template <typename L, typename... Args>
  Network& emplace(Args&&... args) {
    return add(std::make_shared<L>(std::forward<Args>(args)...));
  }

  // Propagates shapes and slot layouts. Called automatically by add().
  Shape configure();
  // Strided conv/pool outputs stay gapped (default, fewer diagonals) or are
  // compacted (fewer slots). Takes effect at the next configure()/compile().
  void set_gapped_layouts(bool on) { gapped_ = on; }
  // Keep encoded weight diagonals between inferences: much faster repeated
  // inference, at one plaintext (ring_dim * towers * 8 bytes) per diagonal.
  void set_cache_plaintexts(bool on);

  // Encrypted-weights mode: encrypts the weights and biases of every linear
  // layer under ctx's public key (activation polynomials stay public). Runs
  // one inference on an encrypted zero input first to learn the level each
  // layer works at, so every weight ciphertext is encrypted with no more
  // towers than needed. drop_plaintext frees the clear diagonals of the
  // compiled plans. Linear products become ciphertext x ciphertext.
  void encrypt_weights(const FHEContext& ctx, bool drop_plaintext = true);
  // Merges runs of adjacent linear layers into one matrix each: saves one
  // level per merged layer at the cost of more diagonals.
  void fuse_linear();
  // Lowers every linear layer to a BSGS plan for `slots` (0 = required_slots()).
  void compile(uint32_t slots = 0);

  // Automatic bootstrap placement: inserts Bootstrap layers so that no
  // stretch of the network consumes more than `budget` levels, as late as
  // possible (fewest bootstraps). Each bootstrap must see values in about
  // [-1, 1], so the values entering it are divided by headroom * max|x| over
  // the calibration inputs (plaintext runs), and multiplied back afterwards.
  // The two scales are folded into neighbouring layers for free (weights and
  // biases of linear layers, polynomial coefficients, Chebyshev intervals,
  // through squares and bias-free linear layers); where that is impossible
  // (residual, attention) a fixed Scale layer is inserted, which costs one
  // level. Without calibration inputs the values are assumed to be in range
  // already. Existing Bootstrap layers are kept. Returns the number of
  // bootstraps inserted; the network must be compiled again.
  size_t place_bootstraps(uint32_t budget, const std::vector<std::vector<double>>& calibration = {},
                          double headroom = 1.5);
  // Folds the interval map of every Chebyshev activation (one level) into the
  // preceding linear layer's weights and bias: the activation then evaluates
  // on [-1, 1] directly. Activations preceded by a layer that cannot take an
  // affine map (no bias, pooling, residual) are left alone unless their
  // interval is symmetric. Returns the number of activations folded.
  size_t fold_activation_intervals();
  // Largest per-stretch budget whose bootstrapping context (scale bits and
  // level budget of `base`) still has a ring dimension of at most `max_ring`.
  static uint32_t max_bootstrap_budget(ContextOptions base, uint32_t max_ring = 1u << 16);

  // Smallest power-of-two slot count that holds every activation.
  uint32_t required_slots() const;
  // Levels consumed by the whole network / by the longest stretch between bootstraps.
  uint32_t depth() const;
  uint32_t depth_between_bootstraps() const;
  bool uses_bootstrapping() const;
  // True when a layer compares through CKKS <-> FHEW scheme switching
  // (MaxPool2d); context_options() then enables it.
  bool uses_scheme_switching() const;
  // Rotation keys to generate. ladder = true (the context has a rotation-key
  // ladder, ContextOptions::rotation_window): only the rotations of layers
  // that cannot compose them (attention); linear layers compose theirs.
  std::vector<int32_t> rotation_indices(bool ladder = false) const;

  // Level to encrypt inputs at: with bootstrapping the first segment only
  // needs depth_before_first_bootstrap() levels, and OpenFHE only refreshes
  // ciphertexts that have fewer levels left than a bootstrap provides.
  uint32_t input_level(const FHEContext& ctx) const;
  uint32_t depth_before_first_bootstrap() const;

  // Suggested CKKS parameters for the compiled network.
  ContextOptions context_options(bool secure = true) const;

  std::vector<double> forward_plain(const std::vector<double>& x) const;

  // Decrypts the network output and maps it back to logical order.
  std::vector<double> decrypt_output(const FHEContext& ctx, const Ctxt& y) const;
  // Same for the output of layer i (e.g. from a LayerHook).
  std::vector<double> decrypt_layer(const FHEContext& ctx, const Ctxt& y, size_t i) const;

  // Optional per-layer hook: (layer index, layer, output ciphertext, seconds).
  using LayerHook = std::function<void(size_t, const Layer&, const Ctxt&, double)>;
  Ctxt forward(const FHEContext& ctx, const Ctxt& x, const LayerHook& hook = nullptr) const;

  const Shape& input_shape() const { return input_; }
  Shape output_shape() const { return layers_.empty() ? input_ : layers_.back()->out_shape(); }
  const std::vector<std::shared_ptr<Layer>>& layers() const { return layers_; }
  uint32_t slots() const { return slots_; }
  std::string summary() const;

 private:
  Shape input_;
  std::vector<std::shared_ptr<Layer>> layers_;
  uint32_t slots_ = 0;
  bool gapped_ = true;
};

}  // namespace fheml

#endif  // FHEML_NETWORK_H_
