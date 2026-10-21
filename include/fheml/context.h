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
