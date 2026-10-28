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
  // EvalBootstrap requires first_mod - scale_mod <= its correction factor (7
  // at large rings), so the first modulus is capped when bootstrapping. The
  // gap is also the headroom of the final level: output values must stay
  // below 2^(first_mod - scale_mod) in magnitude.
  if (opts_.bootstrapping) opts_.first_mod_bits = std::min<uint32_t>({opts_.first_mod_bits, opts_.scale_mod_bits + 7, 60});
  params.SetMultiplicativeDepth(total_depth_);
  params.SetScalingModSize(opts_.scale_mod_bits);
  params.SetFirstModSize(opts_.first_mod_bits);
  params.SetScalingTechnique(fhe::FLEXIBLEAUTO);
  params.SetKeySwitchTechnique(fhe::HYBRID);
  params.SetSecretKeyDist(fhe::UNIFORM_TERNARY);
  params.SetBatchSize(opts_.slots);
  if (opts_.key_switch_digits) params.SetNumLargeDigits(opts_.key_switch_digits);
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

std::vector<int32_t> FHEContext::ladder_rotations(uint32_t slots, uint32_t window) {
  std::set<int32_t> r;
  if (!window) return {};
  const int64_t radix = int64_t(1) << window;
  for (int64_t step = 1; step < slots; step *= radix)
    for (int64_t j = 1; j <= radix / 2; ++j) {
      int32_t k = normalize_rotation(j * step, slots);
      if (k) r.insert(k);
      k = normalize_rotation(-j * step, slots);
      if (k) r.insert(k);
    }
  return std::vector<int32_t>(r.begin(), r.end());
}

void FHEContext::keygen(const std::vector<int32_t>& rotations) {
  keys_ = cc_->KeyGen();
  cc_->EvalMultKeyGen(keys_.secretKey);
  std::set<int32_t> uniq;
  for (int32_t r : rotations) {
    int32_t n = normalize_rotation(r, opts_.slots);
    if (n != 0) uniq.insert(n);
  }
  for (int32_t r : ladder_rotations(opts_.slots, opts_.rotation_window)) uniq.insert(r);
  has_key_.assign(opts_.slots, 0);
  for (int32_t r : uniq) has_key_[static_cast<size_t>(r)] = 1;
  if (!uniq.empty()) cc_->EvalRotateKeyGen(keys_.secretKey, std::vector<int32_t>(uniq.begin(), uniq.end()));
  if (opts_.bootstrapping) cc_->EvalBootstrapKeyGen(keys_.secretKey, opts_.slots);
#ifdef FHEML_WITH_FIDESLIB
  // Uploads the context, relinearization, rotation and bootstrapping keys.
  if (on_gpu_) cc_->LoadContext(keys_.publicKey);
#endif
}

bool FHEContext::has_rotation_key(int32_t k) const {
  const int32_t n = normalize_rotation(k, opts_.slots);
  return n == 0 || (!has_key_.empty() && has_key_[static_cast<size_t>(n)]);
}

namespace {

// Balanced radix-2^w digits of k (mod slots): k = sum_i d_i 2^(w i) with
// |d_i| <= 2^(w-1), each digit a single ladder key.
std::vector<int32_t> ladder_steps(int32_t k, uint32_t slots, uint32_t window) {
  std::vector<int32_t> steps;
  const int64_t radix = int64_t(1) << window, half = int64_t(slots) / 2;
  int64_t rem = normalize_rotation(k, slots);
  if (rem > half) rem -= slots;
  for (int64_t step = 1; rem != 0; step *= radix) {
    int64_t d = ((rem % radix) + radix) % radix;
    if (d > radix / 2 || (d == radix / 2 && rem < 0)) d -= radix;
    if (d) steps.push_back(normalize_rotation(d * step, slots));
    rem = (rem - d) / radix;
  }
  return steps;
}

}  // namespace

uint32_t FHEContext::rotation_cost(int32_t k) const {
  if (has_rotation_key(k)) return normalize_rotation(k, opts_.slots) ? 1 : 0;
  if (!opts_.rotation_window) return 0;
  return static_cast<uint32_t>(ladder_steps(k, opts_.slots, opts_.rotation_window).size());
}

Ctxt FHEContext::rotate(const Ctxt& c, int32_t k) const {
  const int32_t n = normalize_rotation(k, opts_.slots);
  if (n == 0) return c;
  if (has_rotation_key(n)) return cc_->EvalRotate(c, n);
  if (!opts_.rotation_window)
    throw std::logic_error("no rotation key for " + std::to_string(n) + " and no rotation key ladder");
  Ctxt out = c;
  for (int32_t s : ladder_steps(n, opts_.slots, opts_.rotation_window)) out = cc_->EvalRotate(out, s);
  return out;
}

// With a key per index the rotations share one hoisted decomposition.
// Otherwise (rotation-key ladder) each rotation is composed from ladder keys,
// and to save key switches it starts from the already rotated copy closest to
// it: conv offsets such as 28, 29, 30 differ by one step. Planned first, then
// evaluated in waves of independent rotations.
std::vector<Ctxt> FHEContext::rotate_many(const Ctxt& x, const std::vector<int32_t>& offs) const {
  const size_t n = offs.size();
  std::vector<Ctxt> out(n);
  if (std::all_of(offs.begin(), offs.end(), [&](int32_t k) { return has_rotation_key(k); })) {
    std::vector<int32_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = normalize_rotation(offs[i], opts_.slots);
    return backend::rotate_hoisted(cc_, x, idx, parallel());
  }
  // Plan: cheapest offsets first, each from the planned offset nearest to it.
  std::vector<size_t> order(n);
  for (size_t i = 0; i < n; ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return rotation_cost(offs[a]) < rotation_cost(offs[b]); });
  std::vector<int> source(n, -1), wave(n, 0);  // source -1 = x itself
  std::vector<size_t> planned;
  for (size_t i : order) {
    uint32_t best = rotation_cost(offs[i]);
    for (size_t j : planned) {
      uint32_t c = rotation_cost(offs[i] - offs[j]);
      if (c < best || (c == best && source[i] >= 0 && wave[j] < wave[static_cast<size_t>(source[i])])) {
        best = c;
        source[i] = static_cast<int>(j);
      }
    }
    wave[i] = source[i] < 0 ? 1 : wave[static_cast<size_t>(source[i])] + 1;
    planned.push_back(i);
  }
  const int waves = n ? *std::max_element(wave.begin(), wave.end()) : 0;
  for (int w = 1; w <= waves; ++w) {
    std::vector<size_t> now;
    for (size_t i = 0; i < n; ++i)
      if (wave[i] == w) now.push_back(i);
#pragma omp parallel for schedule(dynamic) if (parallel())
    for (size_t t = 0; t < now.size(); ++t) {
      const size_t i = now[t];
      const Ctxt& from = source[i] < 0 ? x : out[static_cast<size_t>(source[i])];
      const int32_t by = source[i] < 0 ? offs[i] : offs[i] - offs[static_cast<size_t>(source[i])];
      out[i] = rotate(from, by);
    }
  }
  return out;
}

// The rotation-key map is OpenFHE's static registry. FIDESlib keeps its keys
// on the device, so a GPU build counts the keys keygen() asked for (without
// the bootstrapping keys) and reports their size as 0.
size_t FHEContext::rotation_key_count() const {
#ifdef FHEML_WITH_FIDESLIB
  return static_cast<size_t>(std::count(has_key_.begin(), has_key_.end(), 1));
#else
  if (!keys_.secretKey) return 0;
  return lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::GetEvalAutomorphismKeyMap(keys_.secretKey->GetKeyTag()).size();
#endif
}

size_t FHEContext::rotation_key_bytes() const {
#ifdef FHEML_WITH_FIDESLIB
  return 0;
#else
  if (!keys_.secretKey) return 0;
  size_t bytes = 0;
  for (const auto& kv :
       lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::GetEvalAutomorphismKeyMap(keys_.secretKey->GetKeyTag())) {
    auto rk = std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(kv.second);
    if (!rk) continue;
    for (const auto* vec : {&rk->GetAVector(), &rk->GetBVector()})
      for (const auto& poly : *vec) bytes += poly.GetNumOfElements() * poly.GetRingDimension() * sizeof(uint64_t);
  }
  return bytes;
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
  if (opts_.rotation_window) os << " rotation_window=" << opts_.rotation_window;
  if (keys_.secretKey)
    os << " rotation_keys=" << rotation_key_count() << " (" << rotation_key_bytes() / (1 << 20) << " MB)";
  return os.str();
}

}  // namespace fheml
