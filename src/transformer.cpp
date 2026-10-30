#include "fheml/transformer.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <set>
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
     << "] degree " << degree_ << " newton " << newton_;
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
  for (int t = 0; t < n_; ++t) {
    mask_[t * stride_] = 1.0 / d_;
    for (int j = 0; j < d_; ++j) pad_var_[t * stride_ + j] = 0.0;
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
  std::vector<double> y(x.size());
  for (int t = 0; t < n_; ++t) {
    const double* xt = &x[static_cast<size_t>(t) * d_];
    double mean = 0.0, ex2 = 0.0;
    for (int j = 0; j < d_; ++j) {
      mean += xt[j];
      ex2 += xt[j] * xt[j];
    }
    mean /= d_;
    ex2 /= d_;
    const double r = inv_sqrt(ex2 - mean * mean);
    for (int j = 0; j < d_; ++j)
      y[static_cast<size_t>(t) * d_ + j] = gamma_[static_cast<size_t>(j)] * (xt[j] - mean) * r + beta_[static_cast<size_t>(j)];
  }
  return y;
}

// Token block sums: after the forward rotations slot t*B holds the sum of
// the token's d features (slots t*B+d .. t*B+B-1 are zero, so nothing of the
// next token leaks in); the masked product scales by 1/d and clears every
// other slot; the backward rotations spread it over the token's d slots.
Ctxt LayerNorm::forward(const FHEContext& ctx, const Ctxt& x) const {
  const CC& cc = ctx.cc();
  if (mask_.size() != ctx.slots()) throw std::invalid_argument("layernorm: slot count mismatch with context");
  auto block_mean = [&](const Ctxt& in) {
    // (EvalAddInPlace on a copy of the handle would modify the caller's
    // ciphertext.)
    Ctxt s = cc->EvalAdd(in, ctx.rotate(in, 1));
    for (int t = 2; t < d_; t *= 2) cc->EvalAddInPlace(s, ctx.rotate(s, t));
    const uint32_t lvl = s->GetLevel() + (s->GetNoiseScaleDeg() > 1 ? 1 : 0);
    Ptxt mask = ctx.encode(mask_, lvl);  // FIDESlib takes plaintexts by mutable reference
    Ctxt m = cc->EvalMult(s, mask);
    for (int t = 1; t < d_; t *= 2) cc->EvalAddInPlace(m, ctx.rotate(m, -t));
    return m;
  };
  Ctxt mean = block_mean(x);
  Ctxt var = cc->EvalSub(block_mean(cc->EvalMult(x, x)), cc->EvalMult(mean, mean));
  Ptxt pad = ctx.encode(pad_var_, var->GetLevel(), var->GetNoiseScaleDeg());
  cc->EvalAddInPlace(var, pad);
  // Newton steps r <- 1.5 r - 0.5 v r^3 without scalar products on r (each
  // would cost a level): the series is scaled to s = k r with k = 1.5^newton,
  // and s <- s - (c v) s^3 with c = 0.5 / (1.5 k^2) divides k by 1.5 per step
  // (the constant rides on v, which has levels to spare), ending at k = 1.
  double k = std::pow(1.5, static_cast<double>(newton_));
  std::vector<double> c = coeffs_;  // FIDESlib takes the coefficients by mutable reference
  for (double& ci : c) ci *= k;
  Ctxt r = cc->EvalChebyshevSeries(var, c, var_min_, var_max_);
  if (newton_) {
    Ctxt v = eps_ != 0.0 ? cc->EvalAdd(var, eps_) : var;
    for (uint32_t i = 0; i < newton_; ++i) {
      Ctxt vc = cc->EvalMult(v, 0.5 / (1.5 * k * k));
      Ctxt p = cc->EvalMult(cc->EvalMult(vc, r), cc->EvalMult(r, r));
      r = cc->EvalSub(r, p);
      k /= 1.5;
    }
  }
  Ctxt centred = cc->EvalSub(x, mean);
  const uint32_t lvl = centred->GetLevel() + (centred->GetNoiseScaleDeg() > 1 ? 1 : 0);
  Ptxt gamma = ctx.encode(gamma_slots_, lvl);
  Ctxt y = cc->EvalMult(cc->EvalMult(centred, gamma), r);
  Ptxt beta = ctx.encode(beta_slots_, y->GetLevel(), y->GetNoiseScaleDeg());
  return cc->EvalAdd(y, beta);
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
     << " " << var_min_ << " " << var_max_ << " " << degree_ << " " << newton_ << "\n";
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
       << (sm_.subtract_max ? " max subtracted (scheme switching)" : "");
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
  os << "\n";
  for (int i = 0; i < 4; ++i) write_vec(os, w_[i]);
  if (has_bias)
    for (int i = 0; i < 4; ++i) write_vec(os, b_[i].empty() ? std::vector<double>(d_, 0.0) : b_[i]);
  write_vec(os, score_);
}

}  // namespace fheml
