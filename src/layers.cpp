#include "fheml/layers.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>

#include "math/chebyshev.h"

namespace fheml {

std::string Shape::str() const {
  std::ostringstream os;
  os << c << "x" << h << "x" << w;
  return os.str();
}

size_t Layout::slot(const Shape& s, size_t logical) const {
  const size_t hw = static_cast<size_t>(s.h) * s.w;
  return slot(logical / hw, (logical % hw) / s.w, logical % s.w);
}

uint32_t poly_depth(uint32_t degree) {
  // Levels consumed by OpenFHE v1.4 EvalPoly / EvalChebyshevSeries on [-1, 1]
  // (Paterson-Stockmeyer), measured; tests/test_fheml.cpp re-checks them.
  static const uint32_t bounds[] = {1, 2, 5, 13, 27, 59, 119, 247, 495, 1007, 2031};
  static const uint32_t depths[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  for (size_t i = 0; i < sizeof(bounds) / sizeof(bounds[0]); ++i)
    if (degree <= bounds[i]) return depths[i];
  throw std::invalid_argument("polynomial degree too large");
}

namespace {

void write_vec(std::ostream& os, const std::vector<double>& v) {
  os << std::setprecision(std::numeric_limits<double>::max_digits10);
  for (size_t i = 0; i < v.size(); ++i) os << v[i] << ((i + 1) % 16 == 0 || i + 1 == v.size() ? "\n" : " ");
}

// params() = weights then biases.
std::vector<double> concat(const std::vector<double>& w, const std::vector<double>& b) {
  std::vector<double> p(w);
  p.insert(p.end(), b.begin(), b.end());
  return p;
}

void split(const std::vector<double>& p, std::vector<double>& w, std::vector<double>& b, const std::string& what) {
  if (p.size() != w.size() + b.size()) throw std::invalid_argument(what + ": wrong parameter count");
  std::copy(p.begin(), p.begin() + static_cast<std::ptrdiff_t>(w.size()), w.begin());
  std::copy(p.begin() + static_cast<std::ptrdiff_t>(w.size()), p.end(), b.begin());
}

}  // namespace

// ----------------------------------------------------------------- LinearLayer

void LinearLayer::set_params(const std::vector<double>& p) {
  if (!p.empty()) throw std::logic_error(type() + ": has no trainable parameters");
}

Layout LinearLayer::configure_layout(const Layout& in, bool gapped) {
  in_layout_ = in;
  const size_t s = static_cast<size_t>(gap_stride());
  out_layout_ = Layout::compact(out_);
  if (gapped && s) {
    // Gapped output: same channel stride as the input, spatial strides scaled,
    // as long as no two outputs land in the same slot.
    Layout cand{in.cs, in.ys * s, in.xs * s};
    std::vector<char> used(cand.span(out_), 0);
    bool ok = true;
    for (size_t i = 0; i < out_.size() && ok; ++i) {
      size_t sl = cand.slot(out_, i);
      ok = !used[sl];
      used[sl] = 1;
    }
    if (ok) out_layout_ = cand;
  }
  return out_layout_;
}

void LinearLayer::compile(uint32_t slots) {
  m_ = matrix();
  b_ = bias();
  if (m_.rows() != out_.size() || m_.cols() != in_.size())
    throw std::logic_error(type() + ": matrix shape does not match layer shapes");

  // Logical matrix -> slot matrix: columns move to the input layout, rows to
  // the output layout (only rows are compact when the output is compact).
  const bool compact_out = out_layout_.is_compact(out_);
  SparseMatrix ms(compact_out ? out_.size() : slots, slots);
  for (size_t r = 0; r < m_.rows(); ++r) {
    size_t rs = compact_out ? r : out_layout_.slot(out_, r);
    for (const auto& e : m_.row(r)) ms.add(rs, in_layout_.slot(in_, e.first), e.second);
  }
  b_slots_.clear();
  if (!b_.empty()) {
    b_slots_.assign(out_layout_.span(out_), 0.0);
    for (size_t r = 0; r < b_.size(); ++r) b_slots_[out_layout_.slot(out_, r)] = b_[r];
  }

  // Few outputs: the folded (hybrid) form has at most 2R-1 diagonals instead
  // of ~#inputs. Pick whichever needs fewer plaintext products + rotations.
  uint32_t fold = 0;
  if (compact_out && allow_fold_) {
    uint32_t block = 1;
    while (block < out_.size()) block <<= 1;
    if (block < slots) {
      SparseMatrix folded = LinearTransform::fold_rows(ms, slots, block);
      size_t d_plain = LinearTransform::count_diagonals(ms, slots);
      size_t d_fold = LinearTransform::count_diagonals(folded, slots);
      auto cost = [](size_t diags, size_t extra) { return static_cast<double>(diags) + 4.0 * (2.0 * std::sqrt(static_cast<double>(diags)) + extra); };
      size_t log_steps = 0;
      for (uint32_t s = block; s < slots; s *= 2) ++log_steps;
      if (cost(d_fold, log_steps) < cost(d_plain, 0)) {
        fold = block;
        ms = std::move(folded);
      }
    }
  }
  lt_ = LinearTransform(ms, slots, 0, fold);
  lt_.set_cache_plaintexts(cache_);
}

std::vector<double> LinearLayer::forward_plain(const std::vector<double>& x) const {
  SparseMatrix m = m_.rows() ? m_ : matrix();
  std::vector<double> y = m.multiply(x);
  std::vector<double> b = m_.rows() ? b_ : bias();
  for (size_t i = 0; i < b.size(); ++i) y[i] += b[i];
  return y;
}

Ctxt LinearLayer::forward(const FHEContext& ctx, const Ctxt& x) const {
  Ctxt y = lt_.eval(ctx, x);
  last_out_level_ = y->GetLevel() + (y->GetNoiseScaleDeg() > 1 ? 1 : 0);
  if (enc_bias_) {
    ctx.cc()->EvalAddInPlace(y, enc_bias_);
  } else if (!b_slots_.empty()) {
    Ptxt pb = ctx.encode(b_slots_, y->GetLevel(), y->GetNoiseScaleDeg());
    ctx.cc()->EvalAddInPlace(y, pb);
  }
  return y;
}

void LinearLayer::encrypt_weights(const FHEContext& ctx, bool drop_plaintext) {
  lt_.encrypt_weights(ctx, lt_.input_level(), drop_plaintext);
  // The sum of products is rescaled before the bias is added, so the bias
  // lives one level below the products.
  if (!b_slots_.empty()) enc_bias_ = ctx.encrypt(b_slots_, last_out_level_);
}

std::string LinearLayer::describe() const {
  std::ostringstream os;
  os << " [" << in_.str() << " -> " << out_.str() << ", " << lt_.num_diagonals() << " diagonals, "
     << lt_.num_baby_steps() << " baby x " << lt_.num_giant_steps() << " giant steps";
  if (lt_.fold_block()) os << ", folded R=" << lt_.fold_block();
  if (lt_.weights_encrypted()) os << ", encrypted weights";
  if (!out_layout_.is_compact(out_)) os << ", gapped out";
  os << "]";
  return os.str();
}

// ---------------------------------------------------------------------- Conv2d

Conv2d::Conv2d(int in_c, int out_c, int kh, int kw, int stride, int pad, std::vector<double> weights,
               std::vector<double> bias)
    : in_c_(in_c), out_c_(out_c), kh_(kh), kw_(kw), stride_(stride), pad_(pad), w_(std::move(weights)),
      b_(std::move(bias)) {
  if (w_.size() != static_cast<size_t>(out_c) * in_c * kh * kw) throw std::invalid_argument("conv2d: bad weight count");
  if (!b_.empty() && b_.size() != static_cast<size_t>(out_c)) throw std::invalid_argument("conv2d: bad bias count");
  if (stride < 1) throw std::invalid_argument("conv2d: stride must be >= 1");
}

std::string Conv2d::describe() const {
  std::ostringstream os;
  os << "conv2d " << in_c_ << "->" << out_c_ << " k=" << kh_ << "x" << kw_ << " s=" << stride_ << " p=" << pad_
     << LinearLayer::describe();
  return os.str();
}

Shape Conv2d::configure(const Shape& in) {
  if (in.c != in_c_) throw std::invalid_argument("conv2d: input channels mismatch (" + in.str() + ")");
  in_ = in;
  out_.c = out_c_;
  out_.h = (in.h + 2 * pad_ - kh_) / stride_ + 1;
  out_.w = (in.w + 2 * pad_ - kw_) / stride_ + 1;
  if (out_.h <= 0 || out_.w <= 0) throw std::invalid_argument("conv2d: empty output");
  return out_;
}

SparseMatrix Conv2d::matrix() const {
  SparseMatrix m(out_.size(), in_.size());
  for (int o = 0; o < out_c_; ++o)
    for (int y = 0; y < out_.h; ++y)
      for (int x = 0; x < out_.w; ++x) {
        size_t row = (static_cast<size_t>(o) * out_.h + y) * out_.w + x;
        for (int i = 0; i < in_c_; ++i)
          for (int dy = 0; dy < kh_; ++dy)
            for (int dx = 0; dx < kw_; ++dx) {
              int iy = y * stride_ - pad_ + dy, ix = x * stride_ - pad_ + dx;
              if (iy < 0 || iy >= in_.h || ix < 0 || ix >= in_.w) continue;  // zero padding
              double wv = w_[((static_cast<size_t>(o) * in_c_ + i) * kh_ + dy) * kw_ + dx];
              m.add(row, (static_cast<size_t>(i) * in_.h + iy) * in_.w + ix, wv);
            }
      }
  return m;
}

std::vector<double> Conv2d::bias() const {
  if (b_.empty()) return {};
  std::vector<double> b(out_.size());
  for (size_t i = 0; i < b.size(); ++i) b[i] = b_[i / (static_cast<size_t>(out_.h) * out_.w)];
  return b;
}

std::vector<LinearLayer::ParamRef> Conv2d::weight_refs() const {
  std::vector<ParamRef> refs;
  for (int o = 0; o < out_c_; ++o)
    for (int y = 0; y < out_.h; ++y)
      for (int x = 0; x < out_.w; ++x) {
        size_t row = (static_cast<size_t>(o) * out_.h + y) * out_.w + x;
        for (int i = 0; i < in_c_; ++i)
          for (int dy = 0; dy < kh_; ++dy)
            for (int dx = 0; dx < kw_; ++dx) {
              int iy = y * stride_ - pad_ + dy, ix = x * stride_ - pad_ + dx;
              if (iy < 0 || iy >= in_.h || ix < 0 || ix >= in_.w) continue;
              refs.push_back({row, (static_cast<size_t>(i) * in_.h + iy) * in_.w + ix,
                              ((o * in_c_ + i) * kh_ + dy) * kw_ + dx});
            }
      }
  return refs;
}

std::vector<int> Conv2d::bias_refs() const {
  if (b_.empty()) return {};
  std::vector<int> r(out_.size());
  for (size_t i = 0; i < r.size(); ++i)
    r[i] = static_cast<int>(w_.size() + i / (static_cast<size_t>(out_.h) * out_.w));
  return r;
}

std::vector<double> Conv2d::params() const { return concat(w_, b_); }

void Conv2d::set_params(const std::vector<double>& p) {
  split(p, w_, b_, "conv2d");
  invalidate();
}

void Conv2d::save(std::ostream& os) const {
  os << "conv2d " << in_c_ << " " << out_c_ << " " << kh_ << " " << kw_ << " " << stride_ << " " << pad_ << " "
     << (b_.empty() ? 0 : 1) << "\n";
  write_vec(os, w_);
  write_vec(os, b_);
}

// ----------------------------------------------------------------------- Dense

Dense::Dense(int in, int out, std::vector<double> weights, std::vector<double> bias)
    : n_in_(in), n_out_(out), w_(std::move(weights)), b_(std::move(bias)) {
  if (w_.size() != static_cast<size_t>(in) * out) throw std::invalid_argument("dense: bad weight count");
  if (!b_.empty() && b_.size() != static_cast<size_t>(out)) throw std::invalid_argument("dense: bad bias count");
}

std::string Dense::describe() const {
  std::ostringstream os;
  os << "dense " << n_in_ << "->" << n_out_ << LinearLayer::describe();
  return os.str();
}

Shape Dense::configure(const Shape& in) {
  if (in.size() != static_cast<size_t>(n_in_)) throw std::invalid_argument("dense: input size mismatch (" + in.str() + ")");
  in_ = in;
  out_ = Shape{n_out_, 1, 1};
  return out_;
}

SparseMatrix Dense::matrix() const {
  SparseMatrix m(n_out_, n_in_);
  for (int r = 0; r < n_out_; ++r)
    for (int c = 0; c < n_in_; ++c) m.add(r, c, w_[static_cast<size_t>(r) * n_in_ + c]);
  return m;
}

std::vector<double> Dense::bias() const { return b_; }

std::vector<LinearLayer::ParamRef> Dense::weight_refs() const {
  std::vector<ParamRef> refs;
  for (int r = 0; r < n_out_; ++r)
    for (int c = 0; c < n_in_; ++c) refs.push_back({static_cast<size_t>(r), static_cast<size_t>(c), r * n_in_ + c});
  return refs;
}

std::vector<int> Dense::bias_refs() const {
  std::vector<int> r;
  for (size_t i = 0; i < b_.size(); ++i) r.push_back(static_cast<int>(w_.size() + i));
  return r;
}

std::vector<double> Dense::params() const { return concat(w_, b_); }

void Dense::set_params(const std::vector<double>& p) {
  split(p, w_, b_, "dense");
  invalidate();
}

void Dense::save(std::ostream& os) const {
  os << "dense " << n_in_ << " " << n_out_ << " " << (b_.empty() ? 0 : 1) << "\n";
  write_vec(os, w_);
  write_vec(os, b_);
}

// ------------------------------------------------------------------- AvgPool2d

AvgPool2d::AvgPool2d(int k, int stride) : k_(k), stride_(stride) {
  if (k < 1 || stride < 1) throw std::invalid_argument("avgpool2d: bad parameters");
}

Shape AvgPool2d::configure(const Shape& in) {
  in_ = in;
  out_ = Shape{in.c, (in.h - k_) / stride_ + 1, (in.w - k_) / stride_ + 1};
  return out_;
}

SparseMatrix AvgPool2d::matrix() const {
  SparseMatrix m(out_.size(), in_.size());
  const double inv = 1.0 / (k_ * k_);
  for (int c = 0; c < out_.c; ++c)
    for (int y = 0; y < out_.h; ++y)
      for (int x = 0; x < out_.w; ++x)
        for (int dy = 0; dy < k_; ++dy)
          for (int dx = 0; dx < k_; ++dx)
            m.add((static_cast<size_t>(c) * out_.h + y) * out_.w + x,
                  (static_cast<size_t>(c) * in_.h + y * stride_ + dy) * in_.w + x * stride_ + dx, inv);
  return m;
}

void AvgPool2d::save(std::ostream& os) const { os << "avgpool2d " << k_ << " " << stride_ << "\n"; }

// --------------------------------------------------------------- ChannelAffine

ChannelAffine::ChannelAffine(std::vector<double> scale, std::vector<double> shift)
    : scale_(std::move(scale)), shift_(std::move(shift)) {
  if (scale_.size() != shift_.size()) throw std::invalid_argument("affine: scale/shift size mismatch");
}

Shape ChannelAffine::configure(const Shape& in) {
  if (static_cast<size_t>(in.c) != scale_.size()) throw std::invalid_argument("affine: channel mismatch");
  return in_ = out_ = in;
}

SparseMatrix ChannelAffine::matrix() const {
  SparseMatrix m(in_.size(), in_.size());
  const size_t hw = static_cast<size_t>(in_.h) * in_.w;
  for (size_t i = 0; i < in_.size(); ++i) m.add(i, i, scale_[i / hw]);
  return m;
}

std::vector<double> ChannelAffine::bias() const {
  std::vector<double> b(in_.size());
  const size_t hw = static_cast<size_t>(in_.h) * in_.w;
  for (size_t i = 0; i < b.size(); ++i) b[i] = shift_[i / hw];
  return b;
}

std::vector<LinearLayer::ParamRef> ChannelAffine::weight_refs() const {
  std::vector<ParamRef> refs;
  const size_t hw = static_cast<size_t>(in_.h) * in_.w;
  for (size_t i = 0; i < in_.size(); ++i) refs.push_back({i, i, static_cast<int>(i / hw)});
  return refs;
}

std::vector<int> ChannelAffine::bias_refs() const {
  std::vector<int> r(in_.size());
  const size_t hw = static_cast<size_t>(in_.h) * in_.w;
  for (size_t i = 0; i < r.size(); ++i) r[i] = static_cast<int>(scale_.size() + i / hw);
  return r;
}

std::vector<double> ChannelAffine::params() const { return concat(scale_, shift_); }

void ChannelAffine::set_params(const std::vector<double>& p) {
  split(p, scale_, shift_, "affine");
  invalidate();
}

void ChannelAffine::save(std::ostream& os) const {
  os << "affine " << scale_.size() << "\n";
  write_vec(os, scale_);
  write_vec(os, shift_);
}

// ----------------------------------------------------------------- FusedLinear

FusedLinear::FusedLinear(Shape in, Shape out, SparseMatrix m, std::vector<double> b, std::string what)
    : m_(std::move(m)), b_(std::move(b)), what_(std::move(what)) {
  in_ = in;
  out_ = out;
}

std::string FusedLinear::describe() const { return "fused(" + what_ + ")" + LinearLayer::describe(); }

Shape FusedLinear::configure(const Shape& in) {
  if (in.size() != in_.size()) throw std::invalid_argument("fused: input size mismatch");
  in_ = in;
  return out_;
}

void FusedLinear::save(std::ostream&) const {
  throw std::logic_error("fused layers are a runtime optimisation and are not serialised; save before fusing");
}

// ---------------------------------------------------------------- Activations

std::vector<double> Square::forward_plain(const std::vector<double>& x) const {
  std::vector<double> y(x);
  for (double& v : y) v *= v;
  return y;
}

Ctxt Square::forward(const FHEContext& ctx, const Ctxt& x) const { return ctx.cc()->EvalSquare(x); }

void Square::save(std::ostream& os) const { os << "square\n"; }

PolyActivation::PolyActivation(std::vector<double> coeffs) : c_(std::move(coeffs)) {
  while (c_.size() > 1 && c_.back() == 0.0) c_.pop_back();
  if (c_.empty()) throw std::invalid_argument("poly: no coefficients");
}

std::string PolyActivation::describe() const {
  std::ostringstream os;
  os << "poly degree " << c_.size() - 1 << " [";
  for (size_t i = 0; i < c_.size(); ++i) os << (i ? ", " : "") << c_[i];
  os << "]";
  return os.str();
}

std::vector<double> PolyActivation::forward_plain(const std::vector<double>& x) const {
  std::vector<double> y(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    double acc = 0.0;
    for (size_t k = c_.size(); k-- > 0;) acc = acc * x[i] + c_[k];
    y[i] = acc;
  }
  return y;
}

uint32_t PolyActivation::depth() const {
  const size_t deg = c_.size() - 1;
  if (deg == 2 && c_[2] == 1.0) return 1;
  return poly_depth(static_cast<uint32_t>(deg));
}

Ctxt PolyActivation::forward(const FHEContext& ctx, const Ctxt& x) const {
  const CC& cc = ctx.cc();
  const size_t deg = c_.size() - 1;
  if (deg == 0) return cc->EvalAdd(cc->EvalMult(x, 0.0), c_[0]);
  if (deg == 1) return cc->EvalAdd(cc->EvalMult(x, c_[1]), c_[0]);
  if (deg == 2) {
    // c2 x^2 + c1 x + c0 = x * (c2 x + c1) + c0: two levels (one if c2 == 1).
    Ctxt t = c_[2] == 1.0 ? x : cc->EvalMult(x, c_[2]);
    t = cc->EvalAdd(t, c_[1]);
    return cc->EvalAdd(cc->EvalMult(x, t), c_[0]);
  }
  return backend::eval_poly(cc, x, c_);
}

void PolyActivation::save(std::ostream& os) const {
  os << "poly " << c_.size() << "\n";
  write_vec(os, c_);
}

std::function<double(double)> ChebyshevActivation::function(const std::string& name) {
  if (name == "relu") return [](double x) { return x > 0 ? x : 0.0; };
  if (name == "sigmoid") return [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
  if (name == "tanh") return [](double x) { return std::tanh(x); };
  if (name == "silu") return [](double x) { return x / (1.0 + std::exp(-x)); };
  if (name == "gelu") return [](double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); };
  throw std::invalid_argument("unknown activation function: " + name);
}

ChebyshevActivation::ChebyshevActivation(std::string fn, double a, double b, uint32_t degree)
    : fn_name_(std::move(fn)), fn_(function(fn_name_)), a_(a), b_(b), degree_(degree) {
  if (!(a < b)) throw std::invalid_argument("chebyshev: need a < b");
  coeffs_ = lbcrypto::EvalChebyshevCoefficients(fn_, a_, b_, degree_);
}

std::string ChebyshevActivation::describe() const {
  std::ostringstream os;
  os << "chebyshev " << fn_name_ << " on [" << a_ << ", " << b_ << "] degree " << degree_;
  return os.str();
}

std::vector<double> ChebyshevActivation::forward_plain(const std::vector<double>& x) const {
  // The reference uses the same approximation the encrypted path evaluates,
  // so plain-vs-encrypted comparisons isolate the CKKS error.
  return lbcrypto::EvalChebyshevFunctionPtxt(fn_, x, a_, b_, degree_);
}

Ctxt ChebyshevActivation::forward(const FHEContext& ctx, const Ctxt& x) const {
  std::vector<double> c = coeffs_;  // FIDESlib takes the coefficients by mutable reference
  return ctx.cc()->EvalChebyshevSeries(x, c, a_, b_);
}

void ChebyshevActivation::save(std::ostream& os) const {
  os << std::setprecision(std::numeric_limits<double>::max_digits10) << "chebyshev " << fn_name_ << " " << a_ << " "
     << b_ << " " << degree_ << "\n";
}

void Bootstrap::save(std::ostream& os) const { os << "bootstrap\n"; }

}  // namespace fheml
