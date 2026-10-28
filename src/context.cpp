#include "fheml/context.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fheml {

namespace {

bool is_pow2(uint32_t x) { return x && !(x & (x - 1)); }

}  // namespace

FHEContext::FHEContext(const ContextOptions& opts) : opts_(opts) {
  if (!is_pow2(opts_.slots)) throw std::invalid_argument("slots must be a power of two");

  boot_depth_ =
      opts_.bootstrapping ? lbcrypto::FHECKKSRNS::GetBootstrapDepth(opts_.level_budget, lbcrypto::UNIFORM_TERNARY) : 0;
  total_depth_ = opts_.mult_depth + boot_depth_;

  fhe::CCParams<fhe::CryptoContextCKKSRNS> params;
  params.SetMultiplicativeDepth(total_depth_);
  params.SetScalingModSize(opts_.scale_mod_bits);
  params.SetFirstModSize(opts_.first_mod_bits);
  params.SetScalingTechnique(fhe::FLEXIBLEAUTO);
  params.SetKeySwitchTechnique(fhe::HYBRID);
  params.SetSecretKeyDist(fhe::UNIFORM_TERNARY);
  params.SetBatchSize(opts_.slots);
  params.SetSecurityLevel(opts_.secure ? fhe::HEStd_128_classic : fhe::HEStd_NotSet);
#ifdef FHEML_WITH_FIDESLIB
  // No devices = FIDESlib runs every operation on its OpenFHE context.
  on_gpu_ = opts_.use_gpu && !opts_.gpu_devices.empty() && gpu_device_count() > 0;
  if (on_gpu_) params.SetDevices(std::vector<int>(opts_.gpu_devices));
#endif

  // OpenFHE picks the ring dimension from the security level and modulus
  // size; it has to be at least 2 * slots, so grow it until both hold.
  uint32_t ring = opts_.ring_dim ? opts_.ring_dim : 2 * opts_.slots;
  // Bootstrapping fails on tiny rings even when security is off.
  if (!opts_.ring_dim && opts_.bootstrapping) ring = std::max(ring, 4096u);
  std::string last_error;
  for (; ring <= (1u << 17); ring *= 2) {
    params.SetRingDim(ring);
    try {
      cc_ = fhe::GenCryptoContext(params);
      break;
    } catch (const std::exception& e) {
      last_error = e.what();
      if (opts_.ring_dim) break;
    }
  }
  if (!cc_) throw std::runtime_error("could not create CKKS context: " + last_error);

  cc_->Enable(fhe::PKE);
  cc_->Enable(fhe::KEYSWITCH);
  cc_->Enable(fhe::LEVELEDSHE);
  cc_->Enable(fhe::ADVANCEDSHE);
  if (opts_.bootstrapping) {
    cc_->Enable(fhe::FHE);
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
#ifdef FHEML_WITH_FIDESLIB
  // Uploads the context, relinearization, rotation and bootstrapping keys.
  if (on_gpu_) cc_->LoadContext(keys_.publicKey);
#endif
}

Ptxt FHEContext::encode(const std::vector<double>& v, uint32_t level, uint32_t scale_deg) const {
  return cc_->MakeCKKSPackedPlaintext(v, scale_deg, level, nullptr, opts_.slots);
}

Ctxt FHEContext::encrypt(const std::vector<double>& v, uint32_t level) const {
  if (v.size() > opts_.slots) throw std::invalid_argument("vector larger than slot count");
  Ptxt p = encode(v, level);
  return cc_->Encrypt(keys_.publicKey, p);
}

std::vector<double> FHEContext::decrypt(const Ctxt& c, size_t n) const {
  Ptxt p;
  Ctxt ct = c;  // FIDESlib's Decrypt takes a mutable handle
  cc_->Decrypt(keys_.secretKey, ct, &p);
  p->SetLength(std::min<size_t>(n, opts_.slots));
  return p->GetRealPackedValue();
}

uint32_t FHEContext::ring_dim_for(const ContextOptions& opts) {
  ContextOptions o = opts;
  if (o.bootstrapping) {
    o.mult_depth += lbcrypto::FHECKKSRNS::GetBootstrapDepth(o.level_budget, lbcrypto::UNIFORM_TERNARY);
    o.bootstrapping = false;
    if (!o.ring_dim) o.slots = std::max(o.slots, 2048u);  // the constructor's 4096 floor
  }
  return FHEContext(o).ring_dim();
}

Ctxt FHEContext::bootstrap(const Ctxt& c) const {
  if (!opts_.bootstrapping) throw std::logic_error("bootstrapping is not enabled in this context");
  return cc_->EvalBootstrap(c);
}

std::string FHEContext::describe() const {
  std::ostringstream os;
  os << "CKKS ring_dim=" << cc_->GetRingDimension() << " slots=" << opts_.slots
     << " depth=" << total_depth_ << " (network " << opts_.mult_depth << ", bootstrap " << boot_depth_ << ")"
     << " scale_bits=" << opts_.scale_mod_bits << " security=" << (opts_.secure ? "128-bit" : "NOT SET (insecure)")
     << " device=" << (on_gpu_ ? "gpu" : "cpu");
  return os.str();
}

}  // namespace fheml
