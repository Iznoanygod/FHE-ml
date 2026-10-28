// Thin wrapper over an OpenFHE CKKS crypto context: parameter selection,
// key generation and encode/encrypt/decrypt helpers.
#ifndef FHEML_CONTEXT_H_
#define FHEML_CONTEXT_H_

#include <cstdint>
#include <string>
#include <vector>

#include "openfhe.h"

namespace fheml {

using CC = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
using Ctxt = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
using Ptxt = lbcrypto::Plaintext;

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

  // Rotates by k slots (any k): with the key for k directly, otherwise
  // composed from the ladder keys (see ContextOptions::rotation_window).
  Ctxt rotate(const Ctxt& c, int32_t k) const;
  // rotate(c, k) for every k in ks. With a key per index the rotations share
  // one hoisted decomposition; composed rotations reuse each other (each
  // starts from the already rotated copy nearest to it).
  std::vector<Ctxt> rotate_many(const Ctxt& c, const std::vector<int32_t>& ks) const;
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
  std::string describe() const;

 private:
  ContextOptions opts_;
  CC cc_;
  lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys_;
  std::vector<char> has_key_;  // per normalised rotation index
  uint32_t total_depth_ = 0;
  uint32_t boot_depth_ = 0;
};

// Normalises a rotation index to [0, slots).
inline int32_t normalize_rotation(int64_t r, uint32_t slots) {
  int64_t s = static_cast<int64_t>(slots);
  return static_cast<int32_t>(((r % s) + s) % s);
}

}  // namespace fheml

#endif  // FHEML_CONTEXT_H_
