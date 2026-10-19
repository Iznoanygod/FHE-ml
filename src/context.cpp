#include "fheml/context.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fheml {

using namespace lbcrypto;

namespace {

bool is_pow2(uint32_t x) { return x && !(x & (x - 1)); }

}  // namespace

FHEContext::FHEContext(const ContextOptions& opts) : opts_(opts) {
  if (!is_pow2(opts_.slots)) throw std::invalid_argument("slots must be a power of two");

  SecretKeyDist dist = UNIFORM_TERNARY;
  boot_depth_ = opts_.bootstrapping ? FHECKKSRNS::GetBootstrapDepth(opts_.level_budget, dist) : 0;
  total_depth_ = opts_.mult_depth + boot_depth_;

  CCParams<CryptoContextCKKSRNS> params;
  params.SetMultiplicativeDepth(total_depth_);
  params.SetScalingModSize(opts_.scale_mod_bits);
  params.SetFirstModSize(opts_.first_mod_bits);
  params.SetScalingTechnique(FLEXIBLEAUTO);
  params.SetKeySwitchTechnique(HYBRID);
  params.SetSecretKeyDist(dist);
  params.SetBatchSize(opts_.slots);
  params.SetSecurityLevel(opts_.secure ? HEStd_128_classic : HEStd_NotSet);

  // OpenFHE picks the ring dimension from the security level and modulus
  // size; it has to be at least 2 * slots, so grow it until both hold.
  uint32_t ring = opts_.ring_dim ? opts_.ring_dim : 2 * opts_.slots;
  std::string last_error;
  for (; ring <= (1u << 17); ring *= 2) {
    params.SetRingDim(ring);
    try {
      cc_ = GenCryptoContext(params);
      break;
    } catch (const std::exception& e) {
      last_error = e.what();
      if (opts_.ring_dim) break;
    }
  }
  if (!cc_) throw std::runtime_error("could not create CKKS context: " + last_error);

  cc_->Enable(PKE);
  cc_->Enable(KEYSWITCH);
  cc_->Enable(LEVELEDSHE);
  cc_->Enable(ADVANCEDSHE);
  if (opts_.bootstrapping) {
    cc_->Enable(FHE);
    cc_->EvalBootstrapSetup(opts_.level_budget, {0, 0}, opts_.slots);
  }
}

void FHEContext::keygen(const std::vector<int32_t>& rotations) {
  keys_ = cc_->KeyGen();
  cc_->EvalMultKeyGen(keys_.secretKey);
  std::set<int32_t> uniq;
  for (int32_t r : rotations) {
    int32_t n = normalize_rotation(r, opts_.slots);
    if (n != 0) uniq.insert(n);
  }
  if (!uniq.empty()) cc_->EvalRotateKeyGen(keys_.secretKey, std::vector<int32_t>(uniq.begin(), uniq.end()));
  if (opts_.bootstrapping) cc_->EvalBootstrapKeyGen(keys_.secretKey, opts_.slots);
}

Ptxt FHEContext::encode(const std::vector<double>& v, uint32_t level, uint32_t scale_deg) const {
  return cc_->MakeCKKSPackedPlaintext(v, scale_deg, level, nullptr, opts_.slots);
}

Ctxt FHEContext::encrypt(const std::vector<double>& v, uint32_t level) const {
  if (v.size() > opts_.slots) throw std::invalid_argument("vector larger than slot count");
  return cc_->Encrypt(keys_.publicKey, encode(v, level));
}

std::vector<double> FHEContext::decrypt(const Ctxt& c, size_t n) const {
  Ptxt p;
  cc_->Decrypt(keys_.secretKey, c, &p);
  p->SetLength(std::min<size_t>(n, opts_.slots));
  return p->GetRealPackedValue();
}

Ctxt FHEContext::bootstrap(const Ctxt& c) const {
  if (!opts_.bootstrapping) throw std::logic_error("bootstrapping is not enabled in this context");
  return cc_->EvalBootstrap(c);
}

std::string FHEContext::describe() const {
  std::ostringstream os;
  os << "CKKS ring_dim=" << cc_->GetRingDimension() << " slots=" << opts_.slots
     << " depth=" << total_depth_ << " (network " << opts_.mult_depth << ", bootstrap " << boot_depth_ << ")"
     << " scale_bits=" << opts_.scale_mod_bits << " security=" << (opts_.secure ? "128-bit" : "NOT SET (insecure)");
  return os.str();
}

}  // namespace fheml
