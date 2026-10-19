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

}  // namespace

// ----------------------------------------------------------------- LinearLayer

void LinearLayer::compile(uint32_t slots) {
  m_ = matrix();
  b_ = bias();
  if (m_.rows() != out_.size() || m_.cols() != in_.size())
    throw std::logic_error(type() + ": matrix shape does not match layer shapes");
  lt_ = LinearTransform(m_, slots);
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
  if (!b_.empty()) {
    Ptxt pb = ctx.encode(b_, y->GetLevel(), y->GetNoiseScaleDeg());
    ctx.cc()->EvalAddInPlace(y, pb);
  }
  return y;
}

std::string LinearLayer::describe() const {
  std::ostringstream os;
  os << " [" << in_.str() << " -> " << out_.str() << ", " << lt_.num_diagonals() << " diagonals, "
     << lt_.num_baby_steps() << " baby x " << lt_.num_giant_steps() << " giant steps]";
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
  return cc->EvalPoly(x, c_);
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
  return ctx.cc()->EvalChebyshevSeries(x, coeffs_, a_, b_);
}

void ChebyshevActivation::save(std::ostream& os) const {
  os << std::setprecision(std::numeric_limits<double>::max_digits10) << "chebyshev " << fn_name_ << " " << a_ << " "
     << b_ << " " << degree_ << "\n";
}

void Bootstrap::save(std::ostream& os) const { os << "bootstrap\n"; }

}  // namespace fheml
