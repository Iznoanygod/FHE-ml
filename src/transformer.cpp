#include "fheml/transformer.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <set>
#include <string>
#include <sstream>
#include <stdexcept>

#include "math/chebyshev.h"

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

// -------------------------------------------------------------------- LayerNorm

LayerNorm::LayerNorm(int n, int d, std::vector<double> gamma, std::vector<double> beta, double eps, double var_min,
                     double var_max, uint32_t degree, uint32_t newton)
    : n_(n), d_(d), gamma_(std::move(gamma)), beta_(std::move(beta)), eps_(eps), var_min_(var_min),
      var_max_(var_max), degree_(degree), newton_(newton) {
  if (n < 1 || d < 2 || (d & (d - 1))) throw std::invalid_argument("layernorm: d must be a power of two, at least 2");
  if (gamma_.size() != static_cast<size_t>(d) || beta_.size() != static_cast<size_t>(d))
    throw std::invalid_argument("layernorm: gamma and beta must have d entries");
  if (!(eps >= 0.0) || !(var_min >= 0.0) || !(var_min < var_max))
    throw std::invalid_argument("layernorm: need 0 <= var_min < var_max and eps >= 0");
  if (degree < 1) throw std::invalid_argument("layernorm: degree must be at least 1");
  coeffs_ = lbcrypto::EvalChebyshevCoefficients([this](double v) { return 1.0 / std::sqrt(v + eps_); }, var_min_,
                                                var_max_, degree_);
}

std::string LayerNorm::describe() const {
  std::ostringstream os;
  os << type() << " " << n_ << "x" << d_ << " eps " << eps_ << " var [" << var_min_ << ", " << var_max_
     << "] degree " << degree_ << " newton " << newton_ << (self_ ? " (self-bootstrapping)" : "");
  return os.str();
}

Shape LayerNorm::configure(const Shape& in) {
  if (in.size() != static_cast<size_t>(n_) * d_) throw std::invalid_argument("layernorm: input size mismatch");
  in_ = out_ = Shape{n_, 1, d_};
  return out_;
}

Layout LayerNorm::configure_layout(const Layout& in, bool gapped) {
  (void)gapped;
  if (in.xs != 1 || in.ys != static_cast<size_t>(d_) && n_ > 1)
    throw std::invalid_argument("layernorm: tokens must be contiguous in the slots");
  in_layout_ = out_layout_ = in;
  stride_ = n_ > 1 ? in.cs : static_cast<size_t>(d_);
  return out_layout_;
}

void LayerNorm::compile(uint32_t slots) {
  mask_.assign(slots, 0.0);
  gamma_slots_.assign(slots, 0.0);
  beta_slots_.assign(slots, 0.0);
  // Slots outside the tokens see a variance of 0, where a Chebyshev series
  // for [var_min, var_max] can be astronomically wrong (and then spoils the
  // decryption of every slot), so they get a variance inside the interval.
  pad_var_.assign(slots, 0.5 * (var_min_ + var_max_));
  token_mask_.assign(slots, 0.0);
  for (int t = 0; t < n_; ++t) {
    mask_[t * stride_] = 1.0 / d_;
    for (int j = 0; j < d_; ++j) pad_var_[t * stride_ + j] = 0.0;
    for (int j = 0; j < d_; ++j) token_mask_[t * stride_ + j] = 1.0;
    for (int j = 0; j < d_; ++j) {
      gamma_slots_[t * stride_ + j] = gamma_[static_cast<size_t>(j)];
      beta_slots_[t * stride_ + j] = beta_[static_cast<size_t>(j)];
    }
  }
}

double LayerNorm::inv_sqrt(double v) const {
  double r = lbcrypto::EvalChebyshevFunctionPtxt([this](double u) { return 1.0 / std::sqrt(u + eps_); }, {v},
                                                 var_min_, var_max_, degree_)[0];
  for (uint32_t i = 0; i < newton_; ++i) r = 1.5 * r - 0.5 * (v + eps_) * r * r * r;
  return r;
}

std::vector<double> LayerNorm::forward_plain(const std::vector<double>& x) const {
  std::vector<double> centred, r, y(x.size());
  normalize_plain(x, centred, r);
  for (int t = 0; t < n_; ++t)
    for (int j = 0; j < d_; ++j) {
      const size_t i = static_cast<size_t>(t) * d_ + j;
      y[i] = gamma_[static_cast<size_t>(j)] * centred[i] * r[static_cast<size_t>(t)] + beta_[static_cast<size_t>(j)];
    }
  return y;
}

// Token block sums: after the forward rotations slot t*B holds the sum of
// the token's d features (slots t*B+d .. t*B+B-1 are zero, so nothing of the
// next token leaks in); the masked product scales by 1/d and clears every
// other slot; the backward rotations spread it over the token's d slots.
Ctxt LayerNorm::block_mean(const FHEContext& ctx, const Ctxt& in) const {
  const CC& cc = ctx.cc();
  if (mask_.size() != ctx.slots()) throw std::invalid_argument("layernorm: slot count mismatch with context");
  // (EvalAddInPlace on a copy of the handle would modify the caller's
  // ciphertext.)
  Ctxt s = cc->EvalAdd(in, ctx.rotate(in, 1));
  for (int t = 2; t < d_; t *= 2) cc->EvalAddInPlace(s, ctx.rotate(s, t));
  const uint32_t lvl = s->GetLevel() + (s->GetNoiseScaleDeg() > 1 ? 1 : 0);
  Ptxt mask = ctx.encode(mask_, lvl);  // FIDESlib takes plaintexts by mutable reference
  Ctxt m = cc->EvalMult(s, mask);
  for (int t = 1; t < d_; t *= 2) cc->EvalAddInPlace(m, ctx.rotate(m, -t));
  return m;
}

void LayerNorm::normalize(const FHEContext& ctx, const Ctxt& x, Ctxt& centred, Ctxt& r) const {
  const CC& cc = ctx.cc();
  Ctxt mean = block_mean(ctx, x);
  Ctxt var = cc->EvalSub(block_mean(ctx, cc->EvalMult(x, x)), cc->EvalMult(mean, mean));
  Ptxt pad = ctx.encode(pad_var_, var->GetLevel(), var->GetNoiseScaleDeg());
  cc->EvalAddInPlace(var, pad);
  // Newton steps r <- 1.5 r - 0.5 v r^3 without scalar products on r (each
  // would cost a level): the series is scaled to s = k r with k = 1.5^newton,
  // and s <- s - (c v) s^3 with c = 0.5 / (1.5 k^2) divides k by 1.5 per step
  // (the constant rides on v, which has levels to spare), ending at k = 1.
  double k = std::pow(1.5, static_cast<double>(newton_));
  std::vector<double> c = coeffs_;  // FIDESlib takes the coefficients by mutable reference
  for (double& ci : c) ci *= k;
  r = cc->EvalChebyshevSeries(var, c, var_min_, var_max_);
  if (newton_) {
    Ctxt v = eps_ != 0.0 ? cc->EvalAdd(var, eps_) : var;
    for (uint32_t i = 0; i < newton_; ++i) {
      Ctxt vc = cc->EvalMult(v, 0.5 / (1.5 * k * k));
      Ctxt p = cc->EvalMult(cc->EvalMult(vc, r), cc->EvalMult(r, r));
      r = cc->EvalSub(r, p);
      k /= 1.5;
    }
  }
  centred = cc->EvalSub(x, mean);
}

// Self-bootstrapping plan, in stretch levels (see Layer::level_after). The
// plain evaluation when it fits; otherwise: the input refreshed first when
// the variance would arrive too deep (t at d + 3: x^2, the 1/d mask, the map
// onto [-1, 1]); t bootstrapped; the series from level 0; each Newton step
// two levels above max(r, v) (v comes from t with one scalar product), r
// refreshed (its scale is folded into the series, the Newton constants and
// gamma, so no level) before a step or the final product that would
// overflow; output at max(centred * gamma, r) + 1.
LayerNorm::Plan LayerNorm::plan(uint32_t d, uint32_t budget) const {
  Plan p;
  const uint32_t plain = d + depth();
  if (!self_ || plain <= budget) {
    p.fits = plain <= budget;
    p.out = p.peak = plain;
    return p;
  }
  p.plain = false;
  const uint32_t pd = poly_depth(degree_);
  const bool can_refresh_r = var_min_ + eps_ > 0.0;
  uint32_t dx = d;
  if (dx + 3 > budget) {
    if (dx + 1 > budget || 3 > budget || !(x_bound_ > 0.0)) return p;
    p.refresh_x = true;
    p.peak = dx + 1;
    dx = 0;
    ++p.boots;
  }
  p.peak = std::max(p.peak, dx + 3);
  ++p.boots;  // t
  if (pd > budget) return p;
  uint32_t dr = pd;
  p.peak = std::max(p.peak, dr);
  p.refresh_r.assign(newton_ + 1, 0);
  auto refresh_r = [&](size_t at) {
    if (!can_refresh_r || dr > budget) return false;
    p.refresh_r[at] = 1;
    dr = 0;
    ++p.boots;
    return true;
  };
  for (uint32_t i = 0; i < newton_; ++i) {
    if (std::max(dr, 1u) + 2 > budget && !refresh_r(i)) return p;
    dr = std::max(dr, 1u) + 2;
    if (dr > budget) return p;
    p.peak = std::max(p.peak, dr);
  }
  if (std::max(dx + 2, dr) + 1 > budget && !refresh_r(newton_)) return p;
  p.out = std::max(dx + 2, dr) + 1;
  if (p.out > budget) return p;
  p.peak = std::max(p.peak, p.out);
  p.fits = true;
  return p;
}

uint32_t LayerNorm::level_after(uint32_t d, uint32_t budget) const {
  Plan p = plan(d, budget);
  return p.fits ? p.out : kNoFit;
}

uint32_t LayerNorm::level_peak(uint32_t d, uint32_t budget) const {
  Plan p = plan(d, budget);
  return p.fits ? p.peak : kNoFit;
}

uint32_t LayerNorm::internal_bootstraps(uint32_t d, uint32_t budget) const {
  Plan p = plan(d, budget);
  return p.fits && !p.plain ? p.boots : 0;
}

void LayerNorm::calibrate(const std::vector<std::vector<double>>& inputs, double headroom) {
  double m = 0.0;
  for (const auto& x : inputs)
    for (double v : x) m = std::max(m, std::fabs(v));
  x_bound_ = headroom * m;
}

// The series overshoots 1/sqrt(v + eps) by a little near the ends of the
// interval; Newton steps only bring it down.
double LayerNorm::inv_sqrt_bound() const { return 1.1 / std::sqrt(var_min_ + eps_); }

Ctxt LayerNorm::forward(const FHEContext& ctx, const Ctxt& x) const {
  const CC& cc = ctx.cc();
  const Plan p = self_ && ctx.stretch_budget() ? plan(ctx.stretch_level(x), ctx.stretch_budget()) : Plan{};
  if (self_ && ctx.stretch_budget() && !p.fits)
    throw std::logic_error("layernorm: does not fit a stretch of " + std::to_string(ctx.stretch_budget()) +
                           " levels from level " + std::to_string(ctx.stretch_level(x)));
  if (!p.plain) {
    const Ctxt xi = p.refresh_x ? ctx.refresh(x, x_bound_, token_mask_) : x;
    Ctxt mean = block_mean(ctx, xi);
    Ctxt var = cc->EvalSub(block_mean(ctx, cc->EvalMult(xi, xi)), cc->EvalMult(mean, mean));
    Ptxt pad = ctx.encode(pad_var_, var->GetLevel(), var->GetNoiseScaleDeg());
    cc->EvalAddInPlace(var, pad);
    // t = (2 var - var_min - var_max) / (var_max - var_min), in [-1, 1].
    const double w = var_max_ - var_min_;
    Ctxt t = cc->EvalAdd(cc->EvalMult(var, 2.0 / w), -(var_min_ + var_max_) / w);
    t = ctx.bootstrap(t);
    // u = k r / S stays within [0, 1] for every k <= k0 (S = k0 * the
    // largest 1/sqrt), so it can be bootstrapped as it is.
    double k = std::pow(1.5, static_cast<double>(newton_));
    const double big = k * inv_sqrt_bound();
    std::vector<double> c = coeffs_;
    for (double& ci : c) ci *= k / big;
    Ctxt r = cc->EvalChebyshevSeries(t, c, -1.0, 1.0);
    for (uint32_t i = 0; i < newton_; ++i) {
      if (p.refresh_r[i]) r = ctx.bootstrap(r);
      // u <- u - (c S^2) (v + eps) u^3, c (v + eps) = c (w/2 t + (var_min +
      // var_max)/2 + eps), see normalize().
      const double ci = 0.5 / (1.5 * k * k) * big * big;
      Ctxt vc = cc->EvalAdd(cc->EvalMult(t, ci * w / 2), ci * (0.5 * (var_min_ + var_max_) + eps_));
      Ctxt q = cc->EvalMult(cc->EvalMult(vc, r), cc->EvalMult(r, r));
      r = cc->EvalSub(r, q);
      k /= 1.5;
    }
    if (p.refresh_r[newton_]) r = ctx.bootstrap(r);
    Ctxt centred = cc->EvalSub(xi, mean);
    std::vector<double> gs(gamma_slots_);
    for (double& g : gs) g *= big;
    Ptxt gamma = ctx.encode(gs, FHEContext::level(centred));
    Ctxt y = cc->EvalMult(cc->EvalMult(centred, gamma), r);
    Ptxt beta = ctx.encode(beta_slots_, y->GetLevel(), y->GetNoiseScaleDeg());
    return cc->EvalAdd(y, beta);
  }
  Ctxt centred, r;
  normalize(ctx, x, centred, r);
  const uint32_t lvl = centred->GetLevel() + (centred->GetNoiseScaleDeg() > 1 ? 1 : 0);
  Ptxt gamma = ctx.encode(gamma_slots_, lvl);
  Ctxt y = cc->EvalMult(cc->EvalMult(centred, gamma), r);
  Ptxt beta = ctx.encode(beta_slots_, y->GetLevel(), y->GetNoiseScaleDeg());
  return cc->EvalAdd(y, beta);
}

void LayerNorm::normalize_plain(const std::vector<double>& x, std::vector<double>& centred,
                                std::vector<double>& r) const {
  centred.assign(x.size(), 0.0);
  r.assign(static_cast<size_t>(n_), 0.0);
  for (int t = 0; t < n_; ++t) {
    const double* xt = &x[static_cast<size_t>(t) * d_];
    double mean = 0.0, ex2 = 0.0;
    for (int j = 0; j < d_; ++j) {
      mean += xt[j];
      ex2 += xt[j] * xt[j];
    }
    mean /= d_;
    ex2 /= d_;
    r[static_cast<size_t>(t)] = inv_sqrt(ex2 - mean * mean);
    for (int j = 0; j < d_; ++j) centred[static_cast<size_t>(t) * d_ + j] = xt[j] - mean;
  }
}

std::vector<double> LayerNorm::params() const {
  std::vector<double> p(gamma_);
  p.insert(p.end(), beta_.begin(), beta_.end());
  return p;
}

void LayerNorm::set_params(const std::vector<double>& p) {
  if (p.size() != 2u * d_) throw std::invalid_argument("layernorm: wrong parameter count");
  std::copy(p.begin(), p.begin() + d_, gamma_.begin());
  std::copy(p.begin() + d_, p.end(), beta_.begin());
  if (!mask_.empty()) compile(static_cast<uint32_t>(mask_.size()));
}

uint32_t LayerNorm::depth() const { return 2 + poly_depth(degree_) + 1 + 2 * newton_ + 1; }

std::vector<int32_t> LayerNorm::rotation_indices() const {
  std::vector<int32_t> r;
  for (int t = 1; t < d_; t *= 2) {
    r.push_back(t);
    r.push_back(-t);
  }
  return r;
}

void LayerNorm::absorb_input_scale(double c) {
  if (c == 0.0) throw std::invalid_argument("layernorm: zero input scale");
  eps_ *= c * c;
  var_min_ *= c * c;
  var_max_ *= c * c;
  coeffs_ = lbcrypto::EvalChebyshevCoefficients([this](double v) { return 1.0 / std::sqrt(v + eps_); }, var_min_,
                                                var_max_, degree_);
}

void LayerNorm::scale_output(double f) {
  for (double& g : gamma_) g *= f;
  for (double& b : beta_) b *= f;
  if (!mask_.empty()) compile(static_cast<uint32_t>(mask_.size()));
}

void LayerNorm::save(std::ostream& os) const {
  os << std::setprecision(std::numeric_limits<double>::max_digits10) << "layernorm " << n_ << " " << d_ << " " << eps_
     << " " << var_min_ << " " << var_max_ << " " << degree_ << " " << newton_;
  if (self_) os << " selfboot " << x_bound_;
  os << "\n";
  write_vec(os, gamma_);
  write_vec(os, beta_);
}

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

uint32_t Residual::level_after(uint32_t d, uint32_t budget) const {
  uint32_t b = d;
  for (const auto& l : body_) {
    b = l->is_bootstrap() ? 0 : l->level_after(b, budget);
    if (b == kNoFit) return kNoFit;
  }
  return std::max(d, b);
}

uint32_t Residual::level_peak(uint32_t d, uint32_t budget) const {
  uint32_t b = d, peak = d;
  for (const auto& l : body_) {
    if (l->is_bootstrap()) {
      b = 0;
      continue;
    }
    const uint32_t p = l->level_peak(b, budget);
    b = l->level_after(b, budget);
    if (b == kNoFit || p == kNoFit) return kNoFit;
    peak = std::max(peak, p);
  }
  return std::max(peak, std::max(d, b));
}

void Residual::set_bootstrapped(bool on) {
  for (auto& l : body_) l->set_bootstrapped(on);
}

bool Residual::self_bootstrapping() const {
  return std::any_of(body_.begin(), body_.end(),
                     [](const std::shared_ptr<Layer>& l) { return l->is_bootstrap() || l->self_bootstrapping(); });
}

uint32_t Residual::internal_bootstraps(uint32_t d, uint32_t budget) const {
  uint32_t b = d, n = 0;
  for (const auto& l : body_) {
    if (l->is_bootstrap()) {
      ++n;
      b = 0;
      continue;
    }
    n += l->internal_bootstraps(b, budget);
    b = l->level_after(b, budget);
    if (b == kNoFit) return n;
  }
  return n;
}

void Residual::save(std::ostream& os) const {
  os << "residual " << body_.size() << "\n";
  for (const auto& l : body_) l->save(os);
}

// ------------------------------------------------------------------- Attention

Attention::Attention(int n, int d, std::vector<double> wq, std::vector<double> wk, std::vector<double> wv,
                     std::vector<double> wo, std::vector<double> bq, std::vector<double> bk, std::vector<double> bv,
                     std::vector<double> bo, std::vector<double> score, int heads)
    : n_(n), d_(d), heads_(heads), dh_(heads > 0 ? d / heads : 0), score_(std::move(score)) {
  if (heads < 1 || d % heads != 0) throw std::invalid_argument("attention: heads must divide d");
  if (!is_pow2(dh_)) throw std::invalid_argument("attention: the head width d/heads must be a power of two");
  if (score_.size() == 3 && heads > 1) {
    std::vector<double> shared = score_;
    score_.clear();
    for (int h = 0; h < heads; ++h) score_.insert(score_.end(), shared.begin(), shared.end());
  }
  if (score_.size() != 3u * heads)
    throw std::invalid_argument("attention: score polynomial needs 3 coefficients, or 3 per head");
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
  os << "attention n=" << n_ << " d=" << d_;
  if (heads_ > 1) os << " heads=" << heads_;
  if (softmax_) {
    os << " softmax exp[" << exp_lo() << ", " << exp_hi() << "] deg " << sm_.exp_degree << " 1/x[" << sum_lo() << ", "
       << sum_hi() << "] deg " << sm_.inv_degree << " newton " << sm_.newton
       << (sm_.subtract_max ? " max subtracted (scheme switching)" : "") << (self_ ? " (self-bootstrapping)" : "");
    return os.str();
  }
  for (int h = 0; h < heads_; ++h) {
    os << (h == 0 ? " score " : "; ") << score_[3 * h] << " + " << score_[3 * h + 1] << " s + " << score_[3 * h + 2]
       << " s^2";
    if (h == 3 && heads_ > 4) {
      os << "; ...";
      break;
    }
  }
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
  // Score polynomial coefficients, masked to the first slot of each head of
  // each token (slot t*B + h*dh holds head h's q.k sum after the window sum).
  mask_c0_.assign(slots, 0.0);
  mask_c1_.assign(slots, 0.0);
  mask_c2_.assign(slots, 0.0);
  for (int t = 0; t < n_; ++t)
    for (int h = 0; h < heads_; ++h) {
      mask_c0_[t * stride_ + h * dh_] = score_[3 * h];
      mask_c1_[t * stride_ + h * dh_] = score_[3 * h + 1];
      mask_c2_[t * stride_ + h * dh_] = score_[3 * h + 2];
    }
  token_mask_.assign(slots, 0.0);
  for (int t = 0; t < n_; ++t)
    for (int c = 0; c < d_; ++c) token_mask_[t * stride_ + c] = 1.0;
  if (softmax_) {
    mask_head_.assign(slots, 0.0);
    // Slots outside the head sub-blocks hold a row sum of 0, where the 1/x
    // series would blow up (and spoil every slot's decryption).
    pad_sum_.assign(slots, 0.5 * (sum_lo() + sum_hi()));
    for (int t = 0; t < n_; ++t)
      for (int h = 0; h < heads_; ++h) {
        mask_head_[t * stride_ + h * dh_] = 1.0;
        for (int c = 0; c < dh_; ++c) pad_sum_[t * stride_ + h * dh_ + c] = 0.0;
      }
    exp_coeffs_ = lbcrypto::EvalChebyshevCoefficients([](double x) { return std::exp(x); }, exp_lo(), exp_hi(),
                                                      sm_.exp_degree);
    inv_coeffs_ = lbcrypto::EvalChebyshevCoefficients([](double x) { return 1.0 / x; }, sum_lo(), sum_hi(),
                                                      sm_.inv_degree);
  }
}

void Attention::set_self_bootstrapping(bool on) {
  if (on && !can_self_bootstrap())
    throw std::logic_error("attention: only softmax attention without subtract_max can bootstrap internally");
  self_ = on;
}

double Attention::exp_div() const { return std::max(1.0, 1.1 * sum_alpha() * std::exp(exp_hi())); }

int32_t Attention::pack_shift(int s) const {
  const int r = s % (2 * dh_);
  return r < dh_ ? r : n_ * stride_ + (r - dh_);
}

// See the comment above the class and LayerNorm::plan. Levels: t_s at
// d + 3 (projections, q.k, the mask product), exp from the bootstrapped
// pack, a_s one level above exp (unpacking mask), o = sum a_s v_s one above
// max(a, v), the 1/x series on the row sums, Newton steps two levels above
// max(y, a), then o * y and W_o. y is within [0, 1] and refreshes without a
// scale level.
Attention::Plan Attention::plan_with(uint32_t d, uint32_t budget, bool refresh_e, bool refresh_sum) const {
  Plan p;
  p.plain = false;
  p.refresh_e = refresh_e;
  const uint32_t pe = poly_depth(sm_.exp_degree), pi = poly_depth(sm_.inv_degree);
  const uint32_t groups = static_cast<uint32_t>(packed_groups());
  uint32_t dx = d;
  if (dx + 3 > budget) {
    if (dx + 1 > budget || 3 > budget || !(x_bound_ > 0.0)) return p;
    p.refresh_x = true;
    p.peak = dx + 1;
    dx = 0;
    ++p.boots;
  }
  p.peak = std::max(p.peak, dx + 3);
  p.boots += groups;
  if (pe > budget) return p;
  p.peak = std::max(p.peak, pe);
  uint32_t de = pe;
  if (refresh_e) {
    p.boots += groups;
    de = 0;
  }
  const uint32_t da = de + 1;
  const uint32_t dv = dx + 1;
  uint32_t dout = std::max(da, dv) + 1;
  if (dout > budget) return p;
  uint32_t dsum = da;
  p.peak = std::max(p.peak, dout);
  if (refresh_sum || dsum + pi > budget) {
    p.refresh_sum = true;
    dsum = 0;
    ++p.boots;
  }
  uint32_t dy = dsum + pi;
  if (dy > budget) return p;
  p.peak = std::max(p.peak, dy);
  p.refresh_y.assign(sm_.newton + 1, 0);
  auto refresh_y = [&](size_t at) {
    p.refresh_y[at] = 1;
    dy = 0;
    ++p.boots;
    return true;
  };
  for (uint32_t i = 0; i < sm_.newton; ++i) {
    if (std::max(dy, da) + 2 > budget) refresh_y(i);
    dy = std::max(dy, da) + 2;
    if (dy > budget) return p;
    p.peak = std::max(p.peak, dy);
  }
  if (dy + 2 > budget) refresh_y(sm_.newton);
  if (dout + 2 > budget) {
    if (dout + 1 > budget || !(o_bound_ > 0.0)) return p;
    p.peak = std::max(p.peak, dout + 1);
    p.refresh_o = true;
    dout = 0;
    ++p.boots;
  }
  p.out = std::max(dout, dy) + 2;
  if (p.out > budget) return p;
  p.peak = std::max(p.peak, p.out);
  p.fits = true;
  return p;
}

Attention::Plan Attention::plan(uint32_t d, uint32_t budget) const {
  const uint32_t plain = d + depth();
  if (!self_ || plain <= budget) {
    Plan p;
    p.fits = plain <= budget;
    p.out = p.peak = plain;
    return p;
  }
  // Refreshing exp or the row sums early can save later refreshes: try all.
  Plan best;
  for (int e = 0; e < 2; ++e)
    for (int r = 0; r < 2; ++r) {
      Plan p = plan_with(d, budget, e != 0, r != 0);
      if (p.fits && (!best.fits || p.boots < best.boots || (p.boots == best.boots && p.out < best.out))) best = p;
    }
  if (!best.fits) best.plain = false;
  return best;
}

uint32_t Attention::level_after(uint32_t d, uint32_t budget) const {
  Plan p = plan(d, budget);
  return p.fits ? p.out : kNoFit;
}

uint32_t Attention::level_peak(uint32_t d, uint32_t budget) const {
  Plan p = plan(d, budget);
  return p.fits ? p.peak : kNoFit;
}

uint32_t Attention::internal_bootstraps(uint32_t d, uint32_t budget) const {
  Plan p = plan(d, budget);
  return p.fits && !p.plain ? p.boots : 0;
}

void Attention::calibrate(const std::vector<std::vector<double>>& inputs, double headroom) {
  double mx = 0.0, mo = 0.0;
  for (const auto& x : inputs) {
    for (double v : x) mx = std::max(mx, std::fabs(v));
    if (!softmax_) continue;
    // The unnormalised output sum_j a_j v_j, as forward_plain computes it.
    std::vector<double> q = proj_[0]->forward_plain(x), k = proj_[1]->forward_plain(x),
                        v = proj_[2]->forward_plain(x);
    for (int h = 0; h < heads_; ++h)
      for (int i = 0; i < n_; ++i) {
        std::vector<double> sc(n_);
        for (int j = 0; j < n_; ++j) {
          double acc = 0.0;
          for (int c = h * dh_; c < (h + 1) * dh_; ++c) acc += q[i * d_ + c] * k[j * d_ + c];
          sc[j] = acc;
        }
        std::vector<double> a = lbcrypto::EvalChebyshevFunctionPtxt([](double t) { return std::exp(t); }, sc, exp_lo(),
                                                                   exp_hi(), sm_.exp_degree);
        for (int c = h * dh_; c < (h + 1) * dh_; ++c) {
          double o = 0.0;
          for (int j = 0; j < n_; ++j) o += a[j] * v[j * d_ + c];
          mo = std::max(mo, std::fabs(o));
        }
      }
  }
  x_bound_ = headroom * mx;
  o_bound_ = headroom * mo * (softmax_ ? 1.1 / sum_lo() : 1.0);
}

void Attention::set_softmax(const SoftmaxOptions& o) {
  if (!(o.lo < o.hi)) throw std::invalid_argument("attention softmax: need lo < hi");
  if (o.exp_degree < 1 || o.inv_degree < 1) throw std::invalid_argument("attention softmax: degrees must be >= 1");
  if ((o.sum_lo != 0.0 || o.sum_hi != 0.0) && !(0.0 < o.sum_lo && o.sum_lo < o.sum_hi))
    throw std::invalid_argument("attention softmax: need 0 < sum_lo < sum_hi");
  softmax_ = true;
  sm_ = o;
}

double Attention::sum_lo() const {
  if (sm_.sum_lo != 0.0) return sm_.sum_lo;
  return sm_.subtract_max ? 1.0 : n_ * std::exp(sm_.lo);
}

double Attention::sum_hi() const {
  if (sm_.sum_hi != 0.0) return sm_.sum_hi;
  return sm_.subtract_max ? static_cast<double>(n_) : n_ * std::exp(sm_.hi);
}

double Attention::inv_plain(double x) const {
  double y = lbcrypto::EvalChebyshevFunctionPtxt([](double u) { return 1.0 / u; }, {x}, sum_lo(), sum_hi(),
                                                 sm_.inv_degree)[0];
  for (uint32_t i = 0; i < sm_.newton; ++i) y = y * (2.0 - x * y);
  return y;
}

// Polynomial score: 6 levels. Softmax: projections and q.k (2), exp series
// (+1 for its interval), the head mask, a.v, the 1/x series (+1) and its
// Newton steps, o / sum, W_o. With the maximum subtracted, the subtraction
// happens at the level of the comparison results (compare_depth() plus one
// per tournament round), whatever the input level.
uint32_t Attention::depth() const {
  if (!softmax_) return 6;
  uint32_t start = 2;
  if (sm_.subtract_max) {
    uint32_t rounds = 0;
    for (int m = n_; m > 1; m = (m + 1) / 2) ++rounds;
    start = std::max(start, FHEContext::compare_depth() + rounds);
  }
  return start + poly_depth(sm_.exp_degree) + 1 + 1 + 1 + poly_depth(sm_.inv_degree) + 1 + 2 * sm_.newton + 1 + 1;
}

std::vector<double> Attention::forward_plain(const std::vector<double>& x) const {
  std::vector<double> q = proj_[0]->forward_plain(x), k = proj_[1]->forward_plain(x), v = proj_[2]->forward_plain(x);
  std::vector<double> out(static_cast<size_t>(n_) * d_, 0.0);
  for (int h = 0; h < heads_; ++h)
    for (int i = 0; i < n_; ++i) {
      std::vector<double> sc(n_);
      for (int j = 0; j < n_; ++j) {
        double s = 0.0;
        for (int c = h * dh_; c < (h + 1) * dh_; ++c) s += q[i * d_ + c] * k[j * d_ + c];
        sc[j] = s;
      }
      std::vector<double> a(n_);
      double inv = 1.0;
      if (softmax_) {
        // The same approximations the encrypted path evaluates.
        if (sm_.subtract_max) {
          double m = *std::max_element(sc.begin(), sc.end());
          for (double& x : sc) x -= m;
        }
        a = lbcrypto::EvalChebyshevFunctionPtxt([](double x) { return std::exp(x); }, sc, exp_lo(), exp_hi(),
                                                sm_.exp_degree);
        double sum = 0.0;
        for (double e : a) sum += e;
        inv = inv_plain(sum);
      } else {
        for (int j = 0; j < n_; ++j)
          a[j] = score_[3 * h] + score_[3 * h + 1] * sc[j] + score_[3 * h + 2] * sc[j] * sc[j];
      }
      for (int j = 0; j < n_; ++j)
        for (int c = h * dh_; c < (h + 1) * dh_; ++c) out[i * d_ + c] += a[j] * v[j * d_ + c] * inv;
    }
  return proj_[3]->forward_plain(out);
}

// The self-bootstrapping softmax (see the comment above the class).
Ctxt Attention::forward_self(const FHEContext& ctx, const Ctxt& x, const Plan& pl) const {
  const CC& cc = ctx.cc();
  const int32_t nd = n_ * stride_;
  auto rot = [&](const Ctxt& c, int32_t r) {
    const int32_t k = normalize_rotation(r, slots_);
    return k ? cc->EvalRotate(c, k) : c;
  };
  auto encode_at = [&](const std::vector<double>& v, const Ctxt& c) {
    return ctx.encode(v, c->GetLevel(), c->GetNoiseScaleDeg());
  };
  auto scaled = [](const std::vector<double>& v, double f) {
    std::vector<double> o(v);
    for (double& e : o) e *= f;
    return o;
  };
  const Ctxt xi = pl.refresh_x ? ctx.refresh(x, x_bound_, token_mask_) : x;
  Ctxt q = proj_[0]->forward(ctx, xi);
  Ctxt k = proj_[1]->forward(ctx, xi);
  Ctxt v = proj_[2]->forward(ctx, xi);
  cc->EvalAddInPlace(k, rot(k, -nd));
  cc->EvalAddInPlace(v, rot(v, -nd));
  std::vector<int32_t> offs(n_);
  for (int s = 0; s < n_; ++s) offs[s] = s * stride_;
  const std::vector<Ctxt> kr = backend::rotate_hoisted(cc, k, offs, ctx.parallel());
  const std::vector<Ctxt> vr = backend::rotate_hoisted(cc, v, offs, ctx.parallel());

  // Scores, mapped onto t in [-1, 1] at the head slots (0 elsewhere).
  const double alpha = 2.0 / (exp_hi() - exp_lo()), beta = -(exp_hi() + exp_lo()) / (exp_hi() - exp_lo());
  const std::vector<double> mask_t = scaled(mask_head_, alpha), add_t = scaled(mask_head_, beta);
  std::vector<Ctxt> t(n_);
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
  for (int s = 0; s < n_; ++s) {
    Ctxt p = cc->EvalMult(q, kr[s]);
    for (int w = 1; w < dh_; w *= 2) cc->EvalAddInPlace(p, rot(p, w));
    Ptxt m = ctx.encode(mask_t, FHEContext::level(p));
    t[s] = cc->EvalMult(p, m);
    Ptxt a = encode_at(add_t, t[s]);
    cc->EvalAddInPlace(t[s], a);
  }
  // Pack, bootstrap, exp (scaled for the row sums), optionally refresh.
  const int groups = packed_groups();
  std::vector<Ctxt> e(groups);
  std::vector<double> ce = scaled(exp_coeffs_, sum_alpha() / exp_div());
  for (int g = 0; g < groups; ++g) {
    Ctxt packed;
    for (int s = g * 2 * dh_; s < std::min(n_, (g + 1) * 2 * dh_); ++s) {
      Ctxt r = rot(t[s], -pack_shift(s));
      packed = packed ? cc->EvalAdd(packed, r) : r;
    }
    std::vector<double> c = ce;
    e[g] = cc->EvalChebyshevSeries(ctx.bootstrap(packed), c, -1.0, 1.0);
    if (pl.refresh_e) e[g] = ctx.bootstrap(e[g]);
  }
  // Unpack a_s twice from the same rotation: times alpha (row sums for the
  // 1/x series' t) and times kappa = 1.1 / sum_lo (row sums for Newton and
  // the output, so that y ~ 1 / (kappa D) stays within [0, 1]); broadcast;
  // weight v_s.
  const double kappa = 1.1 / sum_lo();
  const std::vector<double> mask_a = scaled(mask_head_, exp_div()), mask_k = scaled(mask_head_, exp_div() * kappa / sum_alpha());
  std::vector<Ctxt> partial(n_), weights(n_), weights_k(n_);
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
  for (int s = 0; s < n_; ++s) {
    Ctxt u = rot(e[s / (2 * dh_)], pack_shift(s));
    Ptxt m = ctx.encode(mask_a, FHEContext::level(u));
    Ptxt mk = ctx.encode(mask_k, FHEContext::level(u));
    Ctxt a = cc->EvalMult(u, m), ak = cc->EvalMult(u, mk);
    for (int w = 1; w < dh_; w *= 2) {
      cc->EvalAddInPlace(a, rot(a, -w));
      cc->EvalAddInPlace(ak, rot(ak, -w));
    }
    partial[s] = cc->EvalMult(ak, vr[s]);
    weights[s] = a;
    weights_k[s] = ak;
  }
  Ctxt o = partial[0], sum = weights[0], sum_k = weights_k[0];
  for (int s = 1; s < n_; ++s) {
    cc->EvalAddInPlace(o, partial[s]);
    cc->EvalAddInPlace(sum, weights[s]);
    cc->EvalAddInPlace(sum_k, weights_k[s]);
  }
  // t_D = alpha D + beta on the features, 0 elsewhere; kappa D gets the
  // midpoint off the features so Newton converges there too.
  const std::vector<double> add_d = scaled(token_mask_, sum_beta());
  Ptxt pb = encode_at(add_d, sum);
  Ctxt td = cc->EvalAdd(sum, pb);
  if (pl.refresh_sum) td = ctx.bootstrap(td);
  std::vector<double> pad_k(token_mask_.size());
  for (size_t i = 0; i < pad_k.size(); ++i) pad_k[i] = token_mask_[i] ? 0.0 : kappa * 0.5 * (sum_lo() + sum_hi());
  Ptxt pk = encode_at(pad_k, sum_k);
  const Ctxt dk = cc->EvalAdd(sum_k, pk);
  std::vector<double> ci = scaled(inv_coeffs_, 1.0 / kappa);
  Ctxt y = cc->EvalChebyshevSeries(td, ci, -1.0, 1.0);
  for (uint32_t i = 0; i < sm_.newton; ++i) {
    if (pl.refresh_y[i]) y = ctx.bootstrap(y);
    Ctxt tt = cc->EvalSub(2.0, cc->EvalMult(dk, y));
    y = cc->EvalMult(y, tt);
  }
  if (pl.refresh_y[sm_.newton]) y = ctx.bootstrap(y);
  if (pl.refresh_o) o = ctx.refresh(o, o_bound_, token_mask_);
  return proj_[3]->forward(ctx, cc->EvalMult(o, y));
}

Ctxt Attention::forward(const FHEContext& ctx, const Ctxt& x) const {
  const CC& cc = ctx.cc();
  if (ctx.slots() != slots_) throw std::invalid_argument("attention: slot count mismatch with context");
  if (self_ && ctx.stretch_budget()) {
    const Plan p = plan(ctx.stretch_level(x), ctx.stretch_budget());
    if (!p.fits)
      throw std::logic_error("attention: does not fit a stretch of " + std::to_string(ctx.stretch_budget()) +
                             " levels from level " + std::to_string(ctx.stretch_level(x)));
    if (!p.plain) return forward_self(ctx, x, p);
  }
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
  std::vector<int32_t> offs(n_);
  for (int s = 0; s < n_; ++s) offs[s] = s * stride_;
  const std::vector<Ctxt> kr = backend::rotate_hoisted(cc, k, offs, ctx.parallel());
  const std::vector<Ctxt> vr = backend::rotate_hoisted(cc, v, offs, ctx.parallel());

  // score[i][h] = q_i . k_{i+s} over head h, accumulated into slot
  // i*B + h*dh by summing a dh-wide window (slots i*B+d .. i*B+B-1 are
  // zero, so the last head's window never reads the next token).
  std::vector<Ctxt> p(n_);
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
  for (int s = 0; s < n_; ++s) {
    p[s] = cc->EvalMult(q, kr[s]);
    for (int t = 1; t < dh_; t *= 2) cc->EvalAddInPlace(p[s], rot(p[s], t));
  }
  if (softmax_ && sm_.subtract_max) {
    // Row maximum over the n offsets: a tournament of scheme-switching
    // comparisons, max(a, b) = a + [a < b] (b - a). Scores (and the window
    // partial sums in the other slots) must stay within the compare range.
    const double range = 2.0 * std::max(std::fabs(sm_.lo), std::fabs(sm_.hi));
    std::vector<Ctxt> vals = p;
    while (vals.size() > 1) {
      std::vector<Ctxt> next((vals.size() + 1) / 2);
      for (size_t i = 0; i + 1 < vals.size(); i += 2) {
        Ctxt sel = ctx.compare(vals[i], vals[i + 1], range);
        next[i / 2] = cc->EvalAdd(vals[i], cc->EvalMult(sel, cc->EvalSub(vals[i + 1], vals[i])));
      }
      if (vals.size() % 2) next.back() = vals.back();
      vals.swap(next);
    }
    for (int s = 0; s < n_; ++s) p[s] = cc->EvalSub(p[s], vals[0]);
  }

  std::vector<Ctxt> partial(n_), weights(n_);
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
  for (int s = 0; s < n_; ++s) {
    Ctxt a;
    if (softmax_) {
      // a = mask * exp(p): the series everywhere, then the head slots kept.
      std::vector<double> c = exp_coeffs_;
      Ctxt e = cc->EvalChebyshevSeries(p[s], c, exp_lo(), exp_hi());
      const uint32_t lvl = e->GetLevel() + (e->GetNoiseScaleDeg() > 1 ? 1 : 0);
      Ptxt mask = ctx.encode(mask_head_, lvl);
      a = cc->EvalMult(e, mask);
    } else {
      // a = mask * (c2 p^2 + c1 p + c0): 2 levels, zero outside slots i*B+h*dh.
      const uint32_t lvl = p[s]->GetLevel() + (p[s]->GetNoiseScaleDeg() > 1 ? 1 : 0);
      Ptxt c2 = ctx.encode(mask_c2_, lvl);
      a = cc->EvalMult(p[s], c2);
      Ptxt c1 = ctx.encode(mask_c1_, a->GetLevel(), a->GetNoiseScaleDeg());
      cc->EvalAddInPlace(a, c1);
      a = cc->EvalMult(p[s], a);
      Ptxt c0 = ctx.encode(mask_c0_, a->GetLevel(), a->GetNoiseScaleDeg());
      cc->EvalAddInPlace(a, c0);
    }
    // Broadcast a[i*B+h*dh] over head h's sub-block (the dh-wide window
    // behind each slot holds at most one head slot), then weight v_{i+s}.
    for (int t = 1; t < dh_; t *= 2) cc->EvalAddInPlace(a, rot(a, -t));
    partial[s] = cc->EvalMult(a, vr[s]);
    weights[s] = a;
  }
  Ctxt o = partial[0];
  for (int s = 1; s < n_; ++s) cc->EvalAddInPlace(o, partial[s]);
  if (softmax_) {
    // Divide by the row sums: 1/x series plus Newton steps y <- y (2 - D y).
    Ctxt sum = weights[0];
    for (int s = 1; s < n_; ++s) cc->EvalAddInPlace(sum, weights[s]);
    Ptxt pad = ctx.encode(pad_sum_, sum->GetLevel(), sum->GetNoiseScaleDeg());
    cc->EvalAddInPlace(sum, pad);
    std::vector<double> c = inv_coeffs_;
    Ctxt y = cc->EvalChebyshevSeries(sum, c, sum_lo(), sum_hi());
    for (uint32_t i = 0; i < sm_.newton; ++i) {
      Ctxt t = cc->EvalSub(2.0, cc->EvalMult(sum, y));
      y = cc->EvalMult(y, t);
    }
    o = cc->EvalMult(o, y);
  }
  return proj_[3]->forward(ctx, o);
}

std::vector<double> Attention::params() const {
  std::vector<double> p;
  for (const auto& pr : proj_) {
    std::vector<double> q = pr->params();
    p.insert(p.end(), q.begin(), q.end());
  }
  return p;
}

void Attention::set_params(const std::vector<double>& p) {
  size_t at = 0;
  for (int i = 0; i < 4; ++i) {
    const size_t k = w_[i].size() + b_[i].size();
    if (at + k > p.size()) throw std::invalid_argument("attention: wrong parameter count");
    std::vector<double> q(p.begin() + static_cast<std::ptrdiff_t>(at), p.begin() + static_cast<std::ptrdiff_t>(at + k));
    std::copy(q.begin(), q.begin() + static_cast<std::ptrdiff_t>(w_[i].size()), w_[i].begin());
    std::copy(q.begin() + static_cast<std::ptrdiff_t>(w_[i].size()), q.end(), b_[i].begin());
    proj_[i]->set_params(q);
    at += k;
  }
  if (at != p.size()) throw std::invalid_argument("attention: wrong parameter count");
}

std::vector<int32_t> Attention::rotation_indices() const {
  std::set<int32_t> r;
  for (const auto& p : proj_)
    for (int32_t i : p->rotation_indices()) r.insert(i);
  r.insert(-n_ * stride_);
  for (int s = 1; s < n_; ++s) r.insert(s * stride_);
  for (int t = 1; t < dh_; t *= 2) {
    r.insert(t);
    r.insert(-t);
  }
  if (self_)
    for (int s = 1; s < n_; ++s) {
      r.insert(pack_shift(s));
      r.insert(-pack_shift(s));
    }
  r.erase(0);
  return std::vector<int32_t>(r.begin(), r.end());
}

void Attention::encrypt_weights(const FHEContext& ctx, bool drop_plaintext) {
  for (auto& p : proj_) p->encrypt_weights(ctx, drop_plaintext);
}

void Attention::save(std::ostream& os) const {
  bool has_bias = !b_[0].empty();
  os << "attention " << n_ << " " << d_ << " " << (has_bias ? 1 : 0);
  if (heads_ > 1) os << " heads " << heads_;
  if (softmax_)
    os << std::setprecision(std::numeric_limits<double>::max_digits10) << " softmax " << sm_.lo << " " << sm_.hi << " "
       << sm_.exp_degree << " " << sm_.inv_degree << " " << sm_.newton << " " << (sm_.subtract_max ? 1 : 0) << " "
       << sm_.sum_lo << " " << sm_.sum_hi;
  if (self_) os << " selfboot " << x_bound_ << " " << o_bound_;
  os << "\n";
  for (int i = 0; i < 4; ++i) write_vec(os, w_[i]);
  if (has_bias)
    for (int i = 0; i < 4; ++i) write_vec(os, b_[i].empty() ? std::vector<double>(d_, 0.0) : b_[i]);
  write_vec(os, score_);
}

}  // namespace fheml
