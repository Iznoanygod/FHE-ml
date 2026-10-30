// Thin wrapper over an OpenFHE (or FIDESlib, see backend.h) CKKS crypto
// context: parameter selection, key generation and encode/encrypt/decrypt
// helpers.
#ifndef FHEML_CONTEXT_H_
#define FHEML_CONTEXT_H_

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "fheml/backend.h"

namespace fheml {

struct ContextOptions {
  // Levels the network consumes between bootstraps (or in total when
  // bootstrapping is off).
  uint32_t mult_depth = 1;
  // Number of CKKS slots used; must be a power of two.
  uint32_t slots = 1024;
  uint32_t scale_mod_bits = 40;
  uint32_t first_mod_bits = 60;
  // 0 = smallest ring dimension that fits `slots` and the security level.
  uint32_t ring_dim = 0;
  // false = HEStd_NotSet. Only for tests and experiments.
  bool secure = true;
  bool bootstrapping = false;
  std::vector<uint32_t> level_budget = {3, 3};
  // 2 = OpenFHE's iterative bootstrapping: the residual error of a first
  // bootstrap is bootstrapped again, which roughly doubles the bits of
  // precision (about 10 -> 20 at 50-bit scaling) for twice the time and one
  // more level (counted in bootstrap_depth()). bootstrap_precision is the
  // precision in bits of a single bootstrap; 0 measures it in keygen().
  uint32_t bootstrap_iterations = 1;
  uint32_t bootstrap_precision = 0;
  // Rotation keys. 0: keygen() makes one key per rotation index it is given.
  // w > 0: keygen() also makes a ladder of keys for +-j * 2^(w*i), 1 <= j <=
  // 2^(w-1), and rotate() composes every other rotation from them, so the key
  // set no longer grows with the network (2 log2(slots) keys for w = 1). A
  // rotation then costs up to ceil(log2(slots) / w) key switches.
  uint32_t rotation_window = 0;
  // HYBRID key switching digits (OpenFHE numLargeDigits; 0 = OpenFHE's
  // default). More digits: smaller auxiliary modulus, so a deeper network
  // fits a given ring, but larger keys and slower key switching.
  uint32_t key_switch_digits = 0;
  // Run on the GPU when this build has FIDESlib and a CUDA device is visible;
  // otherwise (or when false) everything runs on the CPU.
  bool use_gpu = true;
  // CUDA devices to use (several = FIDESlib multi-GPU, needs NCCL).
  std::vector<int> gpu_devices = {0};
  // CKKS <-> FHEW scheme switching (OpenFHE SCHEMESWITCH), used by
  // FHEContext::compare() and the MaxPool2d layer. keygen() then also builds
  // the FHEW context, its bootstrapping keys and the switching keys. The FHEW
  // parameter set is STD128 when secure, TOY otherwise. Not available on the
  // FIDESlib (GPU) backend.
  bool scheme_switching = false;
  // log2 of the FHEW ciphertext modulus for the comparison (OpenFHE's
  // "large precision" switching). Also sets the comparison resolution: a
  // difference of two values is resolved to about 2^(2 - logq) of the range.
  uint32_t fhew_log_q = 25;
};

// How FHEContext::rotate_many evaluates a set of rotations of one
// ciphertext: a tree of single-key rotations rooted at the input (node 0).
// Every node with children is decomposed once and all its children are
// rotated from that decomposition (hoisting), so composed rotations that
// share a prefix of ladder steps also share its key switches.
struct RotationPlan {
  std::vector<int> parent;      // parent node; -1 for the input
  std::vector<int32_t> step;    // rotation key from the parent (0 for the input)
  std::vector<int32_t> offset;  // total rotation from the input, in [0, slots)
  std::vector<uint32_t> depth;  // key switches from the input
  std::vector<size_t> out;      // node holding each requested rotation

  // Rotations performed (tree edges).
  size_t key_switches() const { return parent.empty() ? 0 : parent.size() - 1; }
  // Digit decompositions performed (nodes with children).
  size_t decompositions() const;
  // Longest chain of key switches.
  uint32_t max_depth() const;
};

class FHEContext {
 public:
  explicit FHEContext(const ContextOptions& opts);
  // Ring dimension FHEContext(opts) would pick, without building keys or
  // bootstrapping precomputations. A ciphertext has ring_dim / 2 slots at no
  // extra cost per operation.
  static uint32_t ring_dim_for(const ContextOptions& opts);

  // Generates the key pair, relinearization keys, the given rotation keys and
  // (when enabled) bootstrapping keys.
  void keygen(const std::vector<int32_t>& rotations);

  Ptxt encode(const std::vector<double>& v, uint32_t level = 0, uint32_t scale_deg = 1) const;
  // level > 0 encrypts with fewer towers (cheaper), leaving
  // total_depth() - level levels for computation.
  Ctxt encrypt(const std::vector<double>& v, uint32_t level = 0) const;
  std::vector<double> decrypt(const Ctxt& c, size_t n) const;

  Ctxt bootstrap(const Ctxt& c) const;
  // Bootstraps values bounded by |x| <= bound (any bound): they are divided
  // by a power of two s >= bound first, which costs one level, and
  // multiplied back by s afterwards with additions, which costs none. So c
  // needs two levels left; the result sits at a bootstrap's output level.
  // A non-empty mask (per slot) is applied in the same product, clearing
  // slots whose values are not bounded.
  Ctxt refresh(const Ctxt& c, double bound, const std::vector<double>& mask = {}) const;
  // Levels c has consumed (after its pending rescale) and has left.
  static uint32_t level(const Ctxt& c);
  uint32_t levels_left(const Ctxt& c) const;
  // Levels between bootstraps: a bootstrap's output can consume
  // stretch_budget() levels and still be bootstrapped (OpenFHE needs one
  // level left at the input). stretch_level(c) counts c's levels from a
  // bootstrap's output level. Layers that bootstrap internally (LayerNorm,
  // softmax attention) plan with these.
  uint32_t stretch_budget() const { return opts_.bootstrapping ? opts_.mult_depth - 1 : 0; }
  uint32_t stretch_level(const Ctxt& c) const;

  // Encrypted comparison through FHEW (needs ContextOptions::scheme_switching):
  // slot i of the result is 1 where a_i < b_i and 0 otherwise, for values
  // with |a_i|, |b_i| <= range (larger values wrap around modulo the FHEW
  // plaintext space and compare wrongly). Switches a - b to FHEW, evaluates
  // the sign there with one FHEW bootstrap per slot and switches back. The
  // result is a fresh CKKS ciphertext at level compare_depth() whatever the
  // inputs' levels, which only need two levels left. count > 0 compares
  // only the first `count` slots, rounded up to a power of two (one FHEW
  // bootstrap each; the other slots of the result are 0).
  Ctxt compare(const Ctxt& a, const Ctxt& b, double range = 1.0, size_t count = 0) const;
  // Sign test [x > 0] per slot for |x| <= range: compare(0, x) with the
  // FHEW scaling set for a difference of at most range (not 2 range), so
  // small values are resolved twice as finely as by compare().
  Ctxt positive(const Ctxt& x, double range = 1.0) const;
  // Levels of a compare() result, i.e. what OpenFHE's FHEW-to-CKKS decoding
  // consumes (constant for the parameter sets used here).
  static uint32_t compare_depth() { return 12; }
  bool scheme_switching() const { return opts_.scheme_switching; }

  // Rotates by k slots (any k): with the key for k directly, otherwise
  // composed from the ladder keys (see ContextOptions::rotation_window).
  Ctxt rotate(const Ctxt& c, int32_t k) const;
  // rotate(c, k) for every k in ks, evaluated as plan_rotations(ks): with a
  // key per index all rotations share one hoisted decomposition; composed
  // rotations share their common ladder steps, start from the already
  // rotated copy nearest to them, and every copy that is rotated further is
  // decomposed once for all its successors.
  std::vector<Ctxt> rotate_many(const Ctxt& c, const std::vector<int32_t>& ks) const;
  // The rotation tree rotate_many(c, ks) evaluates (cached per index list).
  std::shared_ptr<const RotationPlan> plan_rotations(const std::vector<int32_t>& ks) const;
  bool has_rotation_key(int32_t k) const;
  // Key switches rotate(c, k) performs (0 for k = 0).
  uint32_t rotation_cost(int32_t k) const;
  // The ladder's rotation indices for the given slot count and window.
  static std::vector<int32_t> ladder_rotations(uint32_t slots, uint32_t window);
  // Rotation (automorphism) keys held, including bootstrapping keys, and
  // their size in memory.
  size_t rotation_key_count() const;
  size_t rotation_key_bytes() const;

  const CC& cc() const { return cc_; }
  uint32_t slots() const { return opts_.slots; }
  uint32_t ring_dim() const { return cc_->GetRingDimension(); }
  uint32_t total_depth() const { return total_depth_; }
  uint32_t bootstrap_depth() const { return boot_depth_; }
  const ContextOptions& options() const { return opts_; }
  // True when ciphertext operations run on the GPU.
  bool on_gpu() const { return on_gpu_; }
  // Whether callers may evaluate independent ciphertext operations from
  // several threads. FIDESlib's GPU context is not thread-safe.
  bool parallel() const { return !on_gpu_; }
  std::string describe() const;

 private:
  ContextOptions opts_;
  CC cc_;
  fhe::KeyPair<fhe::DCRTPoly> keys_;
  std::vector<char> has_key_;  // per normalised rotation index
  bool on_gpu_ = false;
  // Comparison scaling last precomputed by compare(); the switching matrix
  // plaintexts depend on the range.
  mutable double compare_range_ = 0.0;
  uint32_t compare_plaintext_mod_ = 0;
  uint32_t total_depth_ = 0;
  uint32_t boot_depth_ = 0;
  uint32_t boot_precision_ = 0;  // bits of one bootstrap (iterative bootstrapping)
  struct PlanCache {
    std::mutex mu;
    std::map<std::vector<int32_t>, std::shared_ptr<const RotationPlan>> plans;
  };
  std::shared_ptr<PlanCache> plans_ = std::make_shared<PlanCache>();
};

// Normalises a rotation index to [0, slots).
inline int32_t normalize_rotation(int64_t r, uint32_t slots) {
  int64_t s = static_cast<int64_t>(slots);
  return static_cast<int32_t>(((r % s) + s) % s);
}

}  // namespace fheml

#endif  // FHEML_CONTEXT_H_
