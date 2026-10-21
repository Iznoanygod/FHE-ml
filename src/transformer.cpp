#include "fheml/transformer.h"

#include <iomanip>
#include <limits>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fheml {

namespace {

void write_vec(std::ostream& os, const std::vector<double>& v) {
  os << std::setprecision(std::numeric_limits<double>::max_digits10);
  for (size_t i = 0; i < v.size(); ++i) os << v[i] << ((i + 1) % 16 == 0 || i + 1 == v.size() ? "\n" : " ");
}

bool is_pow2(int x) { return x > 0 && !(x & (x - 1)); }

}  // namespace

// ------------------------------------------------------------------ TokenDense

TokenDense::TokenDense(int n, int in, int out, std::vector<double> weights, std::vector<double> bias)
    : n_(n), in_dim_(in), out_dim_(out), w_(std::move(weights)), b_(std::move(bias)) {
  if (w_.size() != static_cast<size_t>(in) * out) throw std::invalid_argument("tokendense: bad weight count");
  if (!b_.empty() && b_.size() != static_cast<size_t>(out) && b_.size() != static_cast<size_t>(n) * out)
    throw std::invalid_argument("tokendense: bias must have OUT or N*OUT values");
}

std::string TokenDense::describe() const {
  std::ostringstream os;
  os << "tokendense " << n_ << "x(" << in_dim_ << "->" << out_dim_ << ")" << LinearLayer::describe();
  return os.str();
}

Shape TokenDense::configure(const Shape& in) {
  if (in.size() != static_cast<size_t>(n_) * in_dim_)
    throw std::invalid_argument("tokendense: input size mismatch (" + in.str() + ")");
  in_ = Shape{n_, 1, in_dim_};
  out_ = Shape{n_, 1, out_dim_};
  return out_;
}

Layout TokenDense::configure_layout(const Layout& in, bool gapped) {
  in_layout_ = in;
  // Keep the input's token stride when the output tokens fit in it.
  if (gapped && in.xs == 1 && in.cs >= static_cast<size_t>(out_dim_))
    out_layout_ = Layout{in.cs, static_cast<size_t>(out_dim_), 1};
  else
    out_layout_ = Layout::compact(out_);
  return out_layout_;
}

SparseMatrix TokenDense::matrix() const {
  SparseMatrix m(static_cast<size_t>(n_) * out_dim_, static_cast<size_t>(n_) * in_dim_);
  for (int t = 0; t < n_; ++t)
    for (int o = 0; o < out_dim_; ++o)
      for (int i = 0; i < in_dim_; ++i)
        m.add(static_cast<size_t>(t) * out_dim_ + o, static_cast<size_t>(t) * in_dim_ + i,
              w_[static_cast<size_t>(o) * in_dim_ + i]);
  return m;
}

std::vector<double> TokenDense::bias() const {
  if (b_.empty() || b_.size() == static_cast<size_t>(n_) * out_dim_) return b_;
  std::vector<double> b(static_cast<size_t>(n_) * out_dim_);
  for (size_t i = 0; i < b.size(); ++i) b[i] = b_[i % out_dim_];
  return b;
}

std::vector<LinearLayer::ParamRef> TokenDense::weight_refs() const {
  std::vector<ParamRef> refs;
  for (int t = 0; t < n_; ++t)
    for (int o = 0; o < out_dim_; ++o)
      for (int i = 0; i < in_dim_; ++i)
        refs.push_back({static_cast<size_t>(t) * out_dim_ + o, static_cast<size_t>(t) * in_dim_ + i, o * in_dim_ + i});
  return refs;
}

std::vector<int> TokenDense::bias_refs() const {
  std::vector<int> r;
  if (b_.empty()) return r;
  const bool per_token = b_.size() == static_cast<size_t>(n_) * out_dim_;
  for (size_t i = 0; i < static_cast<size_t>(n_) * out_dim_; ++i)
    r.push_back(static_cast<int>(w_.size() + (per_token ? i : i % out_dim_)));
  return r;
}

std::vector<double> TokenDense::params() const {
  std::vector<double> p(w_);
  p.insert(p.end(), b_.begin(), b_.end());
  return p;
}

void TokenDense::set_params(const std::vector<double>& p) {
  if (p.size() != w_.size() + b_.size()) throw std::invalid_argument("tokendense: wrong parameter count");
  std::copy(p.begin(), p.begin() + static_cast<std::ptrdiff_t>(w_.size()), w_.begin());
  std::copy(p.begin() + static_cast<std::ptrdiff_t>(w_.size()), p.end(), b_.begin());
  invalidate();
}

void TokenDense::save(std::ostream& os) const {
  os << "tokendense " << n_ << " " << in_dim_ << " " << out_dim_ << " " << b_.size() << "\n";
  write_vec(os, w_);
  write_vec(os, b_);
}

// ------------------------------------------------------------------- TokenMean

TokenMean::TokenMean(int n, int d) : n_(n), d_(d) {}

Shape TokenMean::configure(const Shape& in) {
  if (in.size() != static_cast<size_t>(n_) * d_) throw std::invalid_argument("tokenmean: input size mismatch");
  in_ = Shape{n_, 1, d_};
  out_ = Shape{d_, 1, 1};
  return out_;
}

SparseMatrix TokenMean::matrix() const {
  SparseMatrix m(d_, static_cast<size_t>(n_) * d_);
  for (int t = 0; t < n_; ++t)
    for (int j = 0; j < d_; ++j) m.add(j, static_cast<size_t>(t) * d_ + j, 1.0 / n_);
  return m;
}

void TokenMean::save(std::ostream& os) const { os << "tokenmean " << n_ << " " << d_ << "\n"; }

// -------------------------------------------------------------------- Residual

Residual::Residual(std::vector<std::shared_ptr<Layer>> body) : body_(std::move(body)) {
  if (body_.empty()) throw std::invalid_argument("residual: empty body");
}

std::string Residual::describe() const {
  std::ostringstream os;
  os << "residual {";
  for (size_t i = 0; i < body_.size(); ++i) os << (i ? "; " : " ") << body_[i]->describe();
  os << " }";
  return os.str();
}

Shape Residual::configure(const Shape& in) {
  Shape s = in;
  for (auto& l : body_) s = l->configure(s);
  if (s.size() != in.size()) throw std::invalid_argument("residual: body changes the size " + in.str() + " -> " + s.str());
  in_ = out_ = in;
  return out_;
}

Layout Residual::configure_layout(const Layout& in, bool gapped) {
  Layout l = in;
  for (auto& b : body_) l = b->configure_layout(l, gapped);
  bool same = true;
  for (size_t i = 0; i < in_.size() && same; ++i) same = l.slot(body_.back()->out_shape(), i) == in.slot(in_, i);
  if (!same) {
    auto* last = dynamic_cast<LinearLayer*>(body_.back().get());
    if (!last) throw std::invalid_argument("residual: body output layout differs from its input layout");
    last->set_out_layout(in);
  }
  in_layout_ = out_layout_ = in;
  return in;
}

void Residual::compile(uint32_t slots) {
  for (auto& l : body_) l->compile(slots);
}

std::vector<double> Residual::forward_plain(const std::vector<double>& x) const {
  std::vector<double> y = x;
  for (const auto& l : body_) y = l->forward_plain(y);
  for (size_t i = 0; i < y.size(); ++i) y[i] += x[i];
  return y;
}

Ctxt Residual::forward(const FHEContext& ctx, const Ctxt& x) const {
  Ctxt y = x;
  for (const auto& l : body_) y = l->forward(ctx, y);
  // FLEXIBLEAUTO brings x down to y's level before adding.
  return ctx.cc()->EvalAdd(y, x);
}

uint32_t Residual::depth() const {
  uint32_t d = 0;
  for (const auto& l : body_) d += l->depth();
  return d;
}

std::vector<int32_t> Residual::rotation_indices() const {
  std::set<int32_t> s;
  for (const auto& l : body_)
    for (int32_t r : l->rotation_indices()) s.insert(r);
  return std::vector<int32_t>(s.begin(), s.end());
}

size_t Residual::min_slots() const {
  size_t n = Layer::min_slots();
  for (const auto& l : body_) n = std::max(n, l->min_slots());
  return n;
}

void Residual::encrypt_weights(const FHEContext& ctx, bool drop_plaintext) {
  for (auto& l : body_) l->encrypt_weights(ctx, drop_plaintext);
}

void Residual::save(std::ostream& os) const {
  os << "residual " << body_.size() << "\n";
  for (const auto& l : body_) l->save(os);
}

// ------------------------------------------------------------------- Attention

Attention::Attention(int n, int d, std::vector<double> wq, std::vector<double> wk, std::vector<double> wv,
                     std::vector<double> wo, std::vector<double> bq, std::vector<double> bk, std::vector<double> bv,
                     std::vector<double> bo, std::vector<double> score)
    : n_(n), d_(d), score_(std::move(score)) {
  if (!is_pow2(d)) throw std::invalid_argument("attention: d must be a power of two");
  if (score_.size() != 3) throw std::invalid_argument("attention: score polynomial needs 3 coefficients");
  std::vector<double>* ws[4] = {&wq, &wk, &wv, &wo};
  std::vector<double>* bs[4] = {&bq, &bk, &bv, &bo};
  for (int i = 0; i < 4; ++i) {
    w_[i] = std::move(*ws[i]);
    b_[i] = std::move(*bs[i]);
    proj_[i] = std::make_shared<TokenDense>(n, d, d, w_[i], b_[i]);
    proj_[i]->set_allow_fold(false);  // K and V must be zero past n*d (see top)
  }
}

std::string Attention::describe() const {
  std::ostringstream os;
  os << "attention n=" << n_ << " d=" << d_ << " score " << score_[0] << " + " << score_[1] << " s + " << score_[2]
     << " s^2";
  return os.str();
}

Shape Attention::configure(const Shape& in) {
  if (in.size() != static_cast<size_t>(n_) * d_) throw std::invalid_argument("attention: input size mismatch");
  in_ = out_ = Shape{n_, 1, d_};
  for (auto& p : proj_) p->configure(in_);
  return out_;
}

Layout Attention::configure_layout(const Layout& in, bool gapped) {
  (void)gapped;
  if (in.xs != 1 || in.cs < static_cast<size_t>(d_))
    throw std::invalid_argument("attention: tokens must be contiguous with stride >= d");
  stride_ = static_cast<int>(in.cs);
  for (auto& p : proj_)
    if (p->configure_layout(in, true).cs != in.cs) throw std::logic_error("attention: projection changed the layout");
  in_layout_ = out_layout_ = in;
  return in;
}

size_t Attention::min_slots() const { return 2u * n_ * stride_; }

void Attention::compile(uint32_t slots) {
  if (slots < min_slots()) throw std::invalid_argument("attention: needs at least 2*n*d slots");
  slots_ = slots;
  for (auto& p : proj_) p->compile(slots);
  // Score polynomial coefficients, masked to the first slot of each token.
  mask_c0_.assign(slots, 0.0);
  mask_c1_.assign(slots, 0.0);
  mask_c2_.assign(slots, 0.0);
  for (int t = 0; t < n_; ++t) {
    mask_c0_[t * stride_] = score_[0];
    mask_c1_[t * stride_] = score_[1];
    mask_c2_[t * stride_] = score_[2];
  }
}

std::vector<double> Attention::forward_plain(const std::vector<double>& x) const {
  std::vector<double> q = proj_[0]->forward_plain(x), k = proj_[1]->forward_plain(x), v = proj_[2]->forward_plain(x);
  std::vector<double> out(static_cast<size_t>(n_) * d_, 0.0);
  for (int i = 0; i < n_; ++i)
    for (int j = 0; j < n_; ++j) {
      double s = 0.0;
      for (int c = 0; c < d_; ++c) s += q[i * d_ + c] * k[j * d_ + c];
      double a = score_[0] + score_[1] * s + score_[2] * s * s;
      for (int c = 0; c < d_; ++c) out[i * d_ + c] += a * v[j * d_ + c];
    }
  return proj_[3]->forward_plain(out);
}

Ctxt Attention::forward(const FHEContext& ctx, const Ctxt& x) const {
  const CC& cc = ctx.cc();
  if (ctx.slots() != slots_) throw std::invalid_argument("attention: slot count mismatch with context");
  const int32_t nd = n_ * stride_;
  // Keys exist for indices normalised to [0, slots) (see FHEContext::keygen).
  auto rot = [&](const Ctxt& c, int32_t r) { return cc->EvalRotate(c, normalize_rotation(r, slots_)); };
  Ctxt q = proj_[0]->forward(ctx, x);
  Ctxt k = proj_[1]->forward(ctx, x);
  Ctxt v = proj_[2]->forward(ctx, x);
  // Cyclic copies: slots [n*B, 2n*B) repeat [0, n*B), so rotating by s*B maps
  // token i to token (i + s) mod n for every i < n.
  cc->EvalAddInPlace(k, rot(k, -nd));
  cc->EvalAddInPlace(v, rot(v, -nd));
  auto kd = cc->EvalFastRotationPrecompute(k);
  auto vd = cc->EvalFastRotationPrecompute(v);
  const uint32_t m = cc->GetCyclotomicOrder();

  std::vector<Ctxt> partial(n_);
#pragma omp parallel for schedule(dynamic)
  for (int s = 0; s < n_; ++s) {
    Ctxt ks = s ? cc->EvalFastRotation(k, static_cast<uint32_t>(s * stride_), m, kd) : k;
    Ctxt vs = s ? cc->EvalFastRotation(v, static_cast<uint32_t>(s * stride_), m, vd) : v;
    // score[i] = q_i . k_{i+s}, accumulated into slot i*B (slots i*B+d ..
    // i*B+B-1 are zero, so summing a d-wide window is enough).
    Ctxt p = cc->EvalMult(q, ks);
    for (int t = 1; t < d_; t *= 2) cc->EvalAddInPlace(p, rot(p, t));
    // a = mask * (c2 p^2 + c1 p + c0): 2 levels, zero outside slots i*d.
    const uint32_t lvl = p->GetLevel() + (p->GetNoiseScaleDeg() > 1 ? 1 : 0);
    Ctxt a = cc->EvalMult(p, ctx.encode(mask_c2_, lvl));
    Ptxt c1 = ctx.encode(mask_c1_, a->GetLevel(), a->GetNoiseScaleDeg());
    cc->EvalAddInPlace(a, c1);
    a = cc->EvalMult(p, a);
    Ptxt c0 = ctx.encode(mask_c0_, a->GetLevel(), a->GetNoiseScaleDeg());
    cc->EvalAddInPlace(a, c0);
    // Broadcast a[i*d] over token i's block, then weight v_{i+s}.
    for (int t = 1; t < d_; t *= 2) cc->EvalAddInPlace(a, rot(a, -t));
    partial[s] = cc->EvalMult(a, vs);
  }
  Ctxt o = partial[0];
  for (int s = 1; s < n_; ++s) cc->EvalAddInPlace(o, partial[s]);
  return proj_[3]->forward(ctx, o);
}

std::vector<int32_t> Attention::rotation_indices() const {
  std::set<int32_t> r;
  for (const auto& p : proj_)
    for (int32_t i : p->rotation_indices()) r.insert(i);
  r.insert(-n_ * stride_);
  for (int s = 1; s < n_; ++s) r.insert(s * stride_);
  for (int t = 1; t < d_; t *= 2) {
    r.insert(t);
    r.insert(-t);
  }
  return std::vector<int32_t>(r.begin(), r.end());
}

void Attention::encrypt_weights(const FHEContext& ctx, bool drop_plaintext) {
  for (auto& p : proj_) p->encrypt_weights(ctx, drop_plaintext);
}

void Attention::save(std::ostream& os) const {
  bool has_bias = !b_[0].empty();
  os << "attention " << n_ << " " << d_ << " " << (has_bias ? 1 : 0) << "\n";
  for (int i = 0; i < 4; ++i) write_vec(os, w_[i]);
  if (has_bias)
    for (int i = 0; i < 4; ++i) write_vec(os, b_[i].empty() ? std::vector<double>(d_, 0.0) : b_[i]);
  write_vec(os, score_);
}

}  // namespace fheml
