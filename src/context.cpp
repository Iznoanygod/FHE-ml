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
  // Scheme switching encodes FHEW-sized vectors (n = 32 at TOY parameters)
  // into the CKKS slots: tiny insecure rings are too small for it.
  if (!opts_.ring_dim && opts_.scheme_switching) ring = std::max(ring, 2048u);
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
  if (opts_.scheme_switching) {
#ifdef FHEML_WITH_FIDESLIB
    throw std::invalid_argument("scheme switching is not available on the FIDESlib backend");
#else
    cc_->Enable(fhe::SCHEMESWITCH);
#endif
  }
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
  {
    std::lock_guard<std::mutex> lock(plans_->mu);
    plans_->plans.clear();
  }
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
#ifndef FHEML_WITH_FIDESLIB
  if (opts_.scheme_switching) {
    // One FHEW context per CKKS context; its secret key is a function of
    // the CKKS key pair only through the switching keys generated here.
    lbcrypto::SchSwchParams sp;
    sp.SetSecurityLevelCKKS(opts_.secure ? lbcrypto::HEStd_128_classic : lbcrypto::HEStd_NotSet);
    sp.SetSecurityLevelFHEW(opts_.secure ? lbcrypto::STD128 : lbcrypto::TOY);
    sp.SetCtxtModSizeFHEWLargePrec(opts_.fhew_log_q);
    sp.SetNumSlotsCKKS(opts_.slots);
    sp.SetNumValues(opts_.slots);
    lbcrypto::LWEPrivateKey lwe_sk = cc_->EvalSchemeSwitchingSetup(sp);
    cc_->EvalSchemeSwitchingKeyGen(keys_, lwe_sk);
    // Largest FHEW plaintext modulus the sign bootstrap decodes correctly.
    const uint32_t beta = cc_->GetBinCCForSchemeSwitch()->GetBeta().ConvertToInt();
    compare_plaintext_mod_ = (1u << opts_.fhew_log_q) / (2 * beta);
    compare_range_ = 0.0;
  }
#endif
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

size_t RotationPlan::decompositions() const {
  std::vector<char> has_child(parent.size(), 0);
  for (int p : parent)
    if (p >= 0) has_child[static_cast<size_t>(p)] = 1;
  return static_cast<size_t>(std::count(has_child.begin(), has_child.end(), 1));
}

uint32_t RotationPlan::max_depth() const { return depth.empty() ? 0 : *std::max_element(depth.begin(), depth.end()); }

// Greedy: offsets in order of their own cost, each reached from the tree
// node (input, earlier output or intermediate copy) it is cheapest to reach
// from, counting new rotations and new decompositions; ties go to the
// shallower result. A composed path takes its largest ladder steps first, so
// offsets with the same coarse part (conv rows, attention tokens) share the
// intermediate copies and fan out from them.
std::shared_ptr<const RotationPlan> FHEContext::plan_rotations(const std::vector<int32_t>& ks) const {
  {
    std::lock_guard<std::mutex> lock(plans_->mu);
    auto it = plans_->plans.find(ks);
    if (it != plans_->plans.end()) return it->second;
  }
  const int32_t slots = static_cast<int32_t>(opts_.slots);
  auto plan = std::make_shared<RotationPlan>();
  RotationPlan& p = *plan;
  std::vector<int> node_at(opts_.slots, -1);
  std::vector<char> has_child;
  auto add = [&](int parent, int32_t step, int32_t off) {
    const int id = static_cast<int>(p.parent.size());
    p.parent.push_back(parent);
    p.step.push_back(step);
    p.offset.push_back(off);
    p.depth.push_back(parent < 0 ? 0 : p.depth[static_cast<size_t>(parent)] + 1);
    has_child.push_back(0);
    if (parent >= 0) has_child[static_cast<size_t>(parent)] = 1;
    node_at[static_cast<size_t>(off)] = id;
    return id;
  };
  add(-1, 0, 0);
  // rotation_cost, with rotations no key reaches priced out.
  auto cost = [&](int32_t k) -> uint32_t {
    if (!has_rotation_key(k) && !opts_.rotation_window) return 1u << 20;
    return rotation_cost(k);
  };

  std::vector<size_t> order(ks.size());
  std::vector<uint32_t> own(ks.size());
  for (size_t i = 0; i < ks.size(); ++i) {
    order[i] = i;
    own[i] = cost(ks[i]);
  }
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return own[a] < own[b]; });
  p.out.assign(ks.size(), 0);
  // Single-key steps from node j to offset k: the direct key, or the ladder
  // digits, largest first.
  auto path = [&](size_t j, int32_t k) {
    const int32_t delta = normalize_rotation(int64_t(k) - p.offset[j], opts_.slots);
    std::vector<int32_t> steps;
    if (has_rotation_key(delta)) {
      steps.push_back(delta);
    } else {
      steps = ladder_steps(delta, opts_.slots, opts_.rotation_window);
      auto mag = [&](int32_t r) { return std::min(r, slots - r); };
      std::stable_sort(steps.begin(), steps.end(), [&](int32_t x, int32_t y) { return mag(x) > mag(y); });
    }
    return steps;
  };
  // Follows path(j, k) from the last copy along it already in the tree.
  // Returns the cost of the new rotations and new decompositions (nodes
  // getting their first child), weighted 2 : 1 as measured with OpenFHE's
  // HYBRID key switching (decomposition about a third of a key switch). With
  // `build`, adds the missing nodes.
  auto walk = [&](size_t j, int32_t k, bool build) {
    const std::vector<int32_t> steps = path(j, k);
    std::vector<int32_t> offs(steps.size() + 1, p.offset[j]);
    size_t t = 0;
    int start = static_cast<int>(j);
    for (size_t s = 0; s < steps.size(); ++s) {
      offs[s + 1] = normalize_rotation(int64_t(offs[s]) + steps[s], opts_.slots);
      const int at = node_at[static_cast<size_t>(offs[s + 1])];
      if (at >= 0) {
        t = s + 1;
        start = at;
      }
    }
    const size_t len = steps.size() - t;
    if (!len) return 0u;
    const uint32_t c = static_cast<uint32_t>(3 * len - 1 + (has_child[static_cast<size_t>(start)] ? 0 : 1));
    if (build)
      for (size_t s = t; s < steps.size(); ++s) start = add(start, steps[s], offs[s + 1]);
    return c;
  };
  for (size_t i : order) {
    const int32_t k = normalize_rotation(ks[i], opts_.slots);
    if (node_at[static_cast<size_t>(k)] < 0) {
      size_t from = 0;
      uint32_t best = walk(0, k, false), best_depth = cost(k);
      for (size_t j = 1; j < p.parent.size(); ++j) {
        if (cost(k - p.offset[j]) >= (1u << 20)) continue;
        const uint32_t c = walk(j, k, false);
        const uint32_t d = p.depth[j] + cost(k - p.offset[j]);
        if (c < best || (c == best && d < best_depth)) {
          from = j;
          best = c;
          best_depth = d;
        }
      }
      walk(from, k, true);
    }
    p.out[i] = static_cast<size_t>(node_at[static_cast<size_t>(k)]);
  }

  std::lock_guard<std::mutex> lock(plans_->mu);
  return plans_->plans.emplace(ks, plan).first->second;
}

// Evaluated level by level: every node of a level with children is
// decomposed once (hoisting) and all its children rotated from it, all the
// level's rotations in one parallel loop. Intermediate copies are released
// once their children exist.
std::vector<Ctxt> FHEContext::rotate_many(const Ctxt& x, const std::vector<int32_t>& offs) const {
  const size_t n = offs.size();
  if (std::all_of(offs.begin(), offs.end(), [&](int32_t k) { return has_rotation_key(k); })) {
    std::vector<int32_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = normalize_rotation(offs[i], opts_.slots);
    return backend::rotate_hoisted(cc_, x, idx, parallel());
  }
  const std::shared_ptr<const RotationPlan> plan = plan_rotations(offs);
  const RotationPlan& p = *plan;
  const size_t nodes = p.parent.size();
  std::vector<char> is_out(nodes, 0);
  for (size_t o : p.out) is_out[o] = 1;
  std::vector<std::vector<size_t>> children(nodes);
  for (size_t v = 1; v < nodes; ++v) children[static_cast<size_t>(p.parent[v])].push_back(v);
  std::vector<Ctxt> ct(nodes);
  ct[0] = x;
  for (uint32_t d = 0; d < p.max_depth(); ++d) {
    std::vector<size_t> src;
    std::vector<Ctxt> xs;
    std::vector<std::vector<int32_t>> steps;
    for (size_t v = 0; v < nodes; ++v)
      if (p.depth[v] == d && !children[v].empty()) {
        src.push_back(v);
        xs.push_back(ct[v]);
        steps.emplace_back();
        for (size_t c : children[v]) steps.back().push_back(p.step[c]);
      }
    const std::vector<std::vector<Ctxt>> rot = backend::rotate_hoisted(cc_, xs, steps, parallel());
    for (size_t s = 0; s < src.size(); ++s) {
      for (size_t c = 0; c < children[src[s]].size(); ++c) ct[children[src[s]][c]] = rot[s][c];
      if (src[s] && !is_out[src[s]]) ct[src[s]] = nullptr;
    }
  }
  std::vector<Ctxt> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = ct[p.out[i]];
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

Ctxt FHEContext::compare(const Ctxt& a, const Ctxt& b, double range) const {
  if (!opts_.scheme_switching) throw std::logic_error("scheme switching is not enabled in this context");
  if (!(range > 0.0)) throw std::invalid_argument("compare: range must be positive");
#ifdef FHEML_WITH_FIDESLIB
  (void)a;
  (void)b;
  throw std::logic_error("scheme switching is not available on the FIDESlib backend");
#else
  // a - b is scaled so that the full range (-2 range, 2 range) maps into the
  // FHEW plaintext space (-p/2, p/2) with a margin; the sign is read off the
  // half of the plaintext space the message lands in.
  if (range != compare_range_) {
    const double scale_sign = compare_plaintext_mod_ / (4.0 * range * 1.25);
    cc_->EvalCompareSwitchPrecompute(compare_plaintext_mod_, scale_sign);
    compare_range_ = range;
  }
  // OpenFHE's slots-to-coefficients transform keeps three towers of the
  // (rescaled) input: the inputs need two levels left.
  for (const Ctxt& c : {a, b}) {
    const uint32_t used = c->GetLevel() + (c->GetNoiseScaleDeg() > 1 ? 1 : 0);
    if (used + 2 > total_depth_)
      throw std::logic_error("compare: input has " + std::to_string(total_depth_ - std::min(used, total_depth_)) +
                             " levels left, needs 2");
  }
  return cc_->EvalCompareSchemeSwitching(a, b, opts_.slots, opts_.slots);
#endif
}

Ctxt FHEContext::positive(const Ctxt& x, double range) const {
  if (!(range > 0.0)) throw std::invalid_argument("positive: range must be positive");
  return compare(cc_->EvalSub(x, x), x, range / 2.0);
}

std::string FHEContext::describe() const {
  std::ostringstream os;
  os << "CKKS ring_dim=" << cc_->GetRingDimension() << " slots=" << opts_.slots
     << " depth=" << total_depth_ << " (network " << opts_.mult_depth << ", bootstrap " << boot_depth_ << ")"
     << " scale_bits=" << opts_.scale_mod_bits << " security=" << (opts_.secure ? "128-bit" : "NOT SET (insecure)")
     << " device=" << (on_gpu_ ? "gpu" : "cpu");
  if (opts_.rotation_window) os << " rotation_window=" << opts_.rotation_window;
  if (opts_.scheme_switching) os << " scheme_switching=fhew(logq " << opts_.fhew_log_q << ")";
  if (keys_.secretKey)
    os << " rotation_keys=" << rotation_key_count() << " (" << rotation_key_bytes() / (1 << 20) << " MB)";
  return os.str();
}

}  // namespace fheml
