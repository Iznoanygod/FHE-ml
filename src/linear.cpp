#include "fheml/linear.h"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>

namespace fheml {

// ---------------------------------------------------------------- SparseMatrix

size_t SparseMatrix::nnz() const {
  size_t n = 0;
  for (const auto& r : entries_) n += r.size();
  return n;
}

void SparseMatrix::add(size_t r, size_t c, double v) {
  if (r >= rows_ || c >= cols_) throw std::out_of_range("SparseMatrix::add index out of range");
  if (v == 0.0) return;
  auto& row = entries_[r];
  for (auto& e : row) {
    if (e.first == c) {
      e.second += v;
      return;
    }
  }
  row.emplace_back(c, v);
}

std::vector<double> SparseMatrix::multiply(const std::vector<double>& x) const {
  if (x.size() < cols_) throw std::invalid_argument("SparseMatrix::multiply: input too short");
  std::vector<double> y(rows_, 0.0);
  for (size_t r = 0; r < rows_; ++r)
    for (const auto& e : entries_[r]) y[r] += e.second * x[e.first];
  return y;
}

SparseMatrix SparseMatrix::multiply(const SparseMatrix& other) const {
  if (cols_ != other.rows_) throw std::invalid_argument("SparseMatrix::multiply: shape mismatch");
  SparseMatrix out(rows_, other.cols_);
  std::vector<double> acc(other.cols_, 0.0);
  std::vector<size_t> touched;
  for (size_t r = 0; r < rows_; ++r) {
    for (const auto& a : entries_[r]) {
      for (const auto& b : other.entries_[a.first]) {
        if (acc[b.first] == 0.0) touched.push_back(b.first);
        acc[b.first] += a.second * b.second;
      }
    }
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    for (size_t c : touched) {
      if (acc[c] != 0.0) out.entries_[r].emplace_back(c, acc[c]);
      acc[c] = 0.0;
    }
    touched.clear();
  }
  return out;
}

SparseMatrix SparseMatrix::identity(size_t n) {
  SparseMatrix m(n, n);
  for (size_t i = 0; i < n; ++i) m.add(i, i, 1.0);
  return m;
}

// ------------------------------------------------------------- LinearTransform

namespace {

std::vector<double> rotate_vec(const std::vector<double>& v, int64_t k) {
  const int64_t n = static_cast<int64_t>(v.size());
  std::vector<double> out(v.size());
  for (int64_t i = 0; i < n; ++i) out[i] = v[(((i + k) % n) + n) % n];
  return out;
}

// Picks the baby step B minimising (#distinct baby steps) + (#distinct giant
// steps) over the given set of diagonal offsets.
uint32_t choose_baby_step(const std::vector<uint32_t>& diags, uint32_t slots) {
  if (diags.size() <= 2) return slots;  // every diagonal becomes its own baby step
  std::vector<uint32_t> seen_b(slots, 0), seen_g(slots, 0);
  uint32_t best = 1;
  size_t best_cost = std::numeric_limits<size_t>::max();
  for (uint32_t b = 1; b <= slots; ++b) {
    size_t nb = 0, ng = 0;
    for (uint32_t d : diags) {
      uint32_t lo = d % b, hi = d - lo;
      if (lo && seen_b[lo] != b) { seen_b[lo] = b; ++nb; }
      if (hi && seen_g[hi % slots] != b) { seen_g[hi % slots] = b; ++ng; }
    }
    // Baby-step rotations are hoisted (cheaper), so weight them a bit lower.
    size_t cost = 2 * ng + nb;
    if (cost < best_cost) {
      best_cost = cost;
      best = b;
    }
  }
  return best;
}

}  // namespace

SparseMatrix LinearTransform::fold_rows(const SparseMatrix& m, uint32_t slots, uint32_t block) {
  if (m.rows() > block || slots % block) throw std::invalid_argument("fold_rows: bad block size");
  SparseMatrix out(slots, m.cols());
  for (size_t r = 0; r < m.rows(); ++r)
    for (const auto& e : m.row(r)) out.add(r + (e.first / block) * block, e.first, e.second);
  return out;
}

size_t LinearTransform::count_diagonals(const SparseMatrix& m, uint32_t slots) {
  std::vector<char> seen(slots, 0);
  size_t n = 0;
  for (size_t r = 0; r < m.rows(); ++r)
    for (const auto& e : m.row(r)) {
      int32_t d = normalize_rotation(static_cast<int64_t>(e.first) - static_cast<int64_t>(r), slots);
      if (!seen[d]) { seen[d] = 1; ++n; }
    }
  return n;
}

LinearTransform::LinearTransform(const SparseMatrix& m, uint32_t slots, uint32_t baby_step, uint32_t fold_block)
    : slots_(slots), fold_(fold_block) {
  if (fold_ && (slots % fold_ || fold_ >= slots)) throw std::invalid_argument("LinearTransform: bad fold block");
  if (m.rows() > slots || m.cols() > slots)
    throw std::invalid_argument("LinearTransform: matrix larger than slot count");

  // Collect generalized diagonals: d = (c - r) mod S.
  std::map<uint32_t, std::vector<double>> diags;
  for (size_t r = 0; r < m.rows(); ++r) {
    for (const auto& e : m.row(r)) {
      uint32_t d = static_cast<uint32_t>(normalize_rotation(static_cast<int64_t>(e.first) - static_cast<int64_t>(r), slots));
      auto& diag = diags[d];
      if (diag.empty()) diag.assign(slots, 0.0);
      diag[r] += e.second;
    }
  }
  num_diagonals_ = diags.size();

  std::vector<uint32_t> offsets;
  for (const auto& kv : diags) offsets.push_back(kv.first);
  baby_step_ = baby_step ? baby_step : choose_baby_step(offsets, slots);

  std::map<uint32_t, int32_t> baby_index;
  std::map<uint32_t, Giant> giants;
  for (auto& kv : diags) {
    uint32_t d = kv.first;
    uint32_t b = d % baby_step_;
    uint32_t g = (d - b) % slots;
    auto it = baby_index.find(b);
    if (it == baby_index.end()) {
      it = baby_index.emplace(b, static_cast<int32_t>(babies_.size())).first;
      babies_.push_back(static_cast<int32_t>(b));
    }
    Giant& gi = giants[g];
    gi.step = static_cast<int32_t>(g);
    gi.terms.push_back(Term{it->second, rotate_vec(kv.second, -static_cast<int64_t>(g))});
  }
  for (auto& kv : giants) giants_.push_back(std::move(kv.second));
}

std::vector<int32_t> LinearTransform::rotation_indices() const {
  std::vector<int32_t> out;
  for (int32_t b : babies_)
    if (b) out.push_back(b);
  for (const auto& g : giants_)
    if (g.step) out.push_back(g.step);
  if (fold_)
    for (uint32_t s = fold_; s < slots_; s *= 2) out.push_back(static_cast<int32_t>(s));
  return out;
}

std::vector<double> LinearTransform::eval_slots(const std::vector<double>& x_in) const {
  for (const auto& g : giants_)
    for (const auto& t : g.terms)
      if (t.diag.empty()) throw std::logic_error("eval_slots: plaintext diagonals were dropped");
  std::vector<double> x(x_in);
  x.resize(slots_, 0.0);
  std::vector<std::vector<double>> baby(babies_.size());
  for (size_t i = 0; i < babies_.size(); ++i) baby[i] = rotate_vec(x, babies_[i]);
  std::vector<double> y(slots_, 0.0);
  for (const auto& g : giants_) {
    std::vector<double> inner(slots_, 0.0);
    for (const auto& t : g.terms)
      for (uint32_t i = 0; i < slots_; ++i) inner[i] += t.diag[i] * baby[t.baby][i];
    inner = rotate_vec(inner, g.step);
    for (uint32_t i = 0; i < slots_; ++i) y[i] += inner[i];
  }
  if (fold_)
    for (uint32_t s = fold_; s < slots_; s *= 2) {
      auto r = rotate_vec(y, s);
      for (uint32_t i = 0; i < slots_; ++i) y[i] += r[i];
    }
  return y;
}

void LinearTransform::encrypt_weights(const FHEContext& ctx, uint32_t level, bool drop_plaintext) {
  if (ctx.slots() != slots_) throw std::invalid_argument("LinearTransform: slot count mismatch with context");
  if (!enc_.empty()) throw std::logic_error("LinearTransform: weights already encrypted");
  std::vector<std::vector<Ctxt>> enc(giants_.size());
  for (size_t gi = 0; gi < giants_.size(); ++gi) enc[gi].resize(giants_[gi].terms.size());
#pragma omp parallel for schedule(dynamic)
  for (size_t gi = 0; gi < giants_.size(); ++gi)
    for (size_t ti = 0; ti < giants_[gi].terms.size(); ++ti) enc[gi][ti] = ctx.encrypt(giants_[gi].terms[ti].diag, level);
  enc_ = std::move(enc);
  if (drop_plaintext)
    for (auto& g : giants_)
      for (auto& t : g.terms) std::vector<double>().swap(t.diag);
  cache_->by_level.clear();
}

Ctxt LinearTransform::eval(const FHEContext& ctx, const Ctxt& x) const {
  const CC& cc = ctx.cc();
  if (ctx.slots() != slots_) throw std::invalid_argument("LinearTransform: slot count mismatch with context");
  if (giants_.empty()) {
    // Zero matrix.
    return cc->EvalMult(x, 0.0);
  }

  // Baby steps: one digit decomposition shared by all rotations (hoisting),
  // or composed from the context's rotation-key ladder.
  const std::vector<Ctxt> baby = ctx.rotate_many(x, babies_);

  // Plaintext diagonals are encoded at the level the products will live at.
  const uint32_t level = x->GetLevel() + (x->GetNoiseScaleDeg() > 1 ? 1 : 0);
  last_level_ = level;
  const bool encrypted = !enc_.empty();
  const std::vector<std::vector<Ptxt>>* cached = nullptr;
  if (cache_plaintexts_ && !encrypted) {
    std::lock_guard<std::mutex> lock(cache_->mu);
    auto it = cache_->by_level.find(level);
    if (it == cache_->by_level.end()) {
      std::vector<std::vector<Ptxt>> enc(giants_.size());
      for (size_t gi = 0; gi < giants_.size(); ++gi)
        for (const auto& t : giants_[gi].terms) enc[gi].push_back(ctx.encode(t.diag, level));
      it = cache_->by_level.emplace(level, std::move(enc)).first;
    }
    cached = &it->second;
  }

  std::vector<Ctxt> partial(giants_.size());
#pragma omp parallel for schedule(dynamic)
  for (size_t gi = 0; gi < giants_.size(); ++gi) {
    const Giant& g = giants_[gi];
    Ctxt inner;
    for (size_t ti = 0; ti < g.terms.size(); ++ti) {
      const Term& t = g.terms[ti];
      Ctxt prod;
      if (encrypted) {
        // Relinearize once per giant step instead of once per product.
        prod = cc->EvalMultNoRelin(baby[t.baby], enc_[gi][ti]);
      } else {
        Ptxt p = cached ? (*cached)[gi][ti] : ctx.encode(t.diag, level);
        prod = cc->EvalMult(baby[t.baby], p);
      }
      if (!inner)
        inner = prod;
      else
        cc->EvalAddInPlace(inner, prod);
    }
    if (encrypted) cc->RelinearizeInPlace(inner);
    if (g.step) inner = ctx.rotate(inner, g.step);
    partial[gi] = inner;
  }

  Ctxt y = partial[0];
  for (size_t gi = 1; gi < partial.size(); ++gi) cc->EvalAddInPlace(y, partial[gi]);
  if (fold_)
    for (uint32_t s = fold_; s < slots_; s *= 2) cc->EvalAddInPlace(y, ctx.rotate(y, static_cast<int32_t>(s)));
  return y;
}

}  // namespace fheml
