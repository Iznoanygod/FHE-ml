#include "fheml/recurrent.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>

#include "math/chebyshev.h"

namespace fheml {

namespace {

void write_vec(std::ostream& os, const std::vector<double>& v) {
  os << std::setprecision(std::numeric_limits<double>::max_digits10);
  for (size_t i = 0; i < v.size(); ++i) os << v[i] << ((i + 1) % 16 == 0 || i + 1 == v.size() ? "\n" : " ");
}

size_t pow2(size_t n) {
  size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

// c0/2 + sum_k c_k T_k(t) (OpenFHE's convention), Clenshaw.
double cheb(const std::vector<double>& c, double t) {
  double b1 = 0.0, b2 = 0.0;
  for (size_t k = c.size(); k-- > 1;) {
    const double b0 = c[k] + 2.0 * t * b1 - b2;
    b2 = b1;
    b1 = b0;
  }
  return 0.5 * c[0] + t * b1 - b2;
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// ---------------------------------------------------------------- the three executions of Recurrent::run

// Level bookkeeping (planning): a value is its stretch level.
struct LevelOps {
  struct V {
    uint32_t l = 0;
  };
  uint32_t budget;
  bool fitting;
  const std::vector<double>& bounds;
  uint32_t peak = 0, boots = 0;
  bool fail = false;

  V at(uint32_t l) {
    peak = std::max(peak, l);
    if (l > budget) fail = true;
    return V{l};
  }
  V lt(int, const V& x) { return at(x.l + 1); }
  V add(const V& a, const V& b) { return at(std::max(a.l, b.l)); }
  V sub(const V& a, const V& b) { return add(a, b); }
  V neg(const V& a) { return a; }
  V mul(const V& a, const V& b) { return at(std::max(a.l, b.l) + 1); }
  V mulp(const V& a, const std::vector<double>&) { return at(a.l + 1); }
  V addp(const V& a, const std::vector<double>&) { return a; }
  V scal(const V& a, double) { return at(a.l + 1); }
  V series(const V& a, const std::vector<double>& c, int) { return at(a.l + poly_depth(static_cast<uint32_t>(c.size() - 1))); }
  V rot(const V& a, int64_t) { return a; }
  V twice_minus_one(const V& a) { return a; }
  uint32_t level(const V& a) const { return a.l; }
  void record(const V&, int) {}
  V refresh(const V& a, int site) {
    // Bounds above 1 need a scaling product before the bootstrap.
    const bool scaled = bounds[static_cast<size_t>(site)] > 1.0;
    if (a.l + (scaled ? 1 : 0) > budget) fail = true;
    if (scaled) peak = std::max(peak, a.l + 1);
    ++boots;
    return V{0};
  }
};

// Cleartext slot vectors: the reference and the calibration.
struct SlotOps {
  using V = std::vector<double>;
  uint32_t slots;
  const LinearTransform* lts[3];
  const std::vector<double>* biases[3];
  uint32_t budget = 0;
  std::vector<double> rec = std::vector<double>(Recurrent::kSites, 0.0);
  double series_max[2] = {0.0, 0.0};
  bool fitting = false;

  static double at(const std::vector<double>& v, size_t i) { return i < v.size() ? v[i] : 0.0; }
  V lt(int which, const V& x) {
    V y = lts[which]->eval_slots(x);
    const auto& b = *biases[which];
    for (size_t i = 0; i < b.size(); ++i) y[i] += b[i];
    return y;
  }
  V add(const V& a, const V& b) {
    V y(a);
    for (size_t i = 0; i < y.size(); ++i) y[i] += b[i];
    return y;
  }
  V sub(const V& a, const V& b) {
    V y(a);
    for (size_t i = 0; i < y.size(); ++i) y[i] -= b[i];
    return y;
  }
  V neg(const V& a) {
    V y(a);
    for (double& v : y) v = -v;
    return y;
  }
  V mul(const V& a, const V& b) {
    V y(a);
    for (size_t i = 0; i < y.size(); ++i) y[i] *= b[i];
    return y;
  }
  V mulp(const V& a, const std::vector<double>& p) {
    V y(a);
    for (size_t i = 0; i < y.size(); ++i) y[i] *= at(p, i);
    return y;
  }
  V addp(const V& a, const std::vector<double>& p) {
    V y(a);
    for (size_t i = 0; i < y.size(); ++i) y[i] += at(p, i);
    return y;
  }
  V scal(const V& a, double f) {
    V y(a);
    for (double& v : y) v *= f;
    return y;
  }
  V series(const V& a, const std::vector<double>& c, int tag) {
    V y(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      series_max[tag] = std::max(series_max[tag], std::fabs(a[i]));
      y[i] = cheb(c, a[i]);
    }
    return y;
  }
  V rot(const V& a, int64_t k) {
    const int64_t n = static_cast<int64_t>(a.size());
    V y(a.size());
    for (int64_t i = 0; i < n; ++i) y[static_cast<size_t>(i)] = a[static_cast<size_t>((((i + k) % n) + n) % n)];
    return y;
  }
  V twice_minus_one(const V& a) {
    V y(a);
    for (double& v : y) v = 2.0 * v - 1.0;
    return y;
  }
  uint32_t level(const V&) const { return 0; }
  void record(const V& a, int site) {
    for (double v : a) rec[static_cast<size_t>(site)] = std::max(rec[static_cast<size_t>(site)], std::fabs(v));
  }
  V refresh(const V& a, int) { return a; }
};

// Ciphertexts.
struct CtxtOps {
  using V = Ctxt;
  const FHEContext& ctx;
  const LinearTransform* lts[3];
  const std::vector<double>* biases[3];
  const std::vector<double>& bounds;
  uint32_t budget;
  bool fitting;

  const CC& cc() const { return ctx.cc(); }
  V lt(int which, const V& x) {
    V y = lts[which]->eval(ctx, x);
    const auto& b = *biases[which];
    if (!b.empty()) {
      Ptxt p = ctx.encode(b, y->GetLevel(), y->GetNoiseScaleDeg());
      cc()->EvalAddInPlace(y, p);
    }
    return y;
  }
  V add(const V& a, const V& b) { return cc()->EvalAdd(a, b); }
  V sub(const V& a, const V& b) { return cc()->EvalSub(a, b); }
  V neg(const V& a) { return cc()->EvalNegate(a); }
  V mul(const V& a, const V& b) { return cc()->EvalMult(a, b); }
  V mulp(const V& a, const std::vector<double>& p) {
    Ptxt q = ctx.encode(p, FHEContext::level(a));
    return cc()->EvalMult(a, q);
  }
  V addp(const V& a, const std::vector<double>& p) {
    Ptxt q = ctx.encode(p, a->GetLevel(), a->GetNoiseScaleDeg());
    return cc()->EvalAdd(a, q);
  }
  V scal(const V& a, double f) { return cc()->EvalMult(a, f); }
  V series(const V& a, const std::vector<double>& c, int) {
    std::vector<double> cc_ = c;  // FIDESlib takes the coefficients by mutable reference
    return cc()->EvalChebyshevSeries(a, cc_, -1.0, 1.0);
  }
  V rot(const V& a, int64_t k) {
    const int64_t s = static_cast<int64_t>(ctx.slots());
    return ctx.rotate(a, static_cast<int32_t>(((k % s) + s) % s));
  }
  V twice_minus_one(const V& a) { return cc()->EvalSub(cc()->EvalAdd(a, a), 1.0); }
  uint32_t level(const V& a) const { return ctx.stretch_level(a); }
  void record(const V&, int) {}
  V refresh(const V& a, int site) {
    const double b = bounds[static_cast<size_t>(site)];
    return b > 1.0 ? ctx.refresh(a, b) : ctx.bootstrap(a);
  }
};

}  // namespace

// ------------------------------------------------------------------ Recurrent

Recurrent::Cell Recurrent::parse_cell(const std::string& name) {
  if (name == "rnn") return Cell::kRNN;
  if (name == "gru") return Cell::kGRU;
  if (name == "lstm") return Cell::kLSTM;
  throw std::invalid_argument("recurrent: unknown cell '" + name + "' (rnn, gru or lstm)");
}

std::string Recurrent::cell_name(Cell c) { return c == Cell::kRNN ? "rnn" : c == Cell::kGRU ? "gru" : "lstm"; }

Recurrent::Recurrent(Cell cell, int n, int in, int hidden, std::vector<double> w_ih, std::vector<double> w_hh,
                     std::vector<double> b_ih, std::vector<double> b_hh, double gate_a, uint32_t degree, double cell_a,
                     uint32_t cell_degree, bool last)
    : cell_(cell), n_(n), in_dim_(in), h_(hidden), w_ih_(std::move(w_ih)), w_hh_(std::move(w_hh)),
      b_ih_(std::move(b_ih)), b_hh_(std::move(b_hh)), gate_a_(gate_a), cell_a_(cell_a), degree_(degree),
      cell_degree_(cell_degree), last_(last) {
  const size_t g = static_cast<size_t>(gates(cell_)) * hidden;
  if (n < 1 || in < 1 || hidden < 1) throw std::invalid_argument("recurrent: tokens, inputs and hidden must be >= 1");
  if (w_ih_.size() != g * in) throw std::invalid_argument("recurrent: w_ih must have gates*hidden*inputs values");
  if (w_hh_.size() != g * hidden) throw std::invalid_argument("recurrent: w_hh must have gates*hidden*hidden values");
  if (b_ih_.empty()) b_ih_.assign(g, 0.0);
  if (b_hh_.empty()) b_hh_.assign(g, 0.0);
  if (b_ih_.size() != g || b_hh_.size() != g) throw std::invalid_argument("recurrent: biases must have gates*hidden values");
  if (!(gate_a_ > 0.0) || !(cell_a_ > 0.0)) throw std::invalid_argument("recurrent: intervals must be positive");
  if (degree_ < 1 || cell_degree_ < 1) throw std::invalid_argument("recurrent: degrees must be >= 1");
}

std::string Recurrent::describe() const {
  std::ostringstream os;
  os << "recurrent " << cell_name(cell_) << " " << n_ << "x(" << in_dim_ << "->" << h_ << ")"
     << (last_ ? " last state" : " all states") << ", " << (cell_ == Cell::kRNN ? "tanh" : "sigmoid") << " on [-"
     << gate_a_ << ", " << gate_a_ << "] degree " << degree_;
  if (cell_ != Cell::kRNN) os << ", tanh on [-" << cell_a_ << ", " << cell_a_ << "] degree " << cell_degree_;
  if (out_scale_ != 1.0) os << ", x " << out_scale_;
  if (region_)
    os << " [" << n_ << " steps, stride " << stride_ << ", " << (cell_ == Cell::kRNN ? 1 : cell_ == Cell::kGRU ? 3 : 4)
       << " x " << region_ << " slots";
  if (slots_)
    os << ", " << lt_x_.num_diagonals() + lt_xn_.num_diagonals() << " + " << lt_h_.num_diagonals()
       << " diagonals per step";
  if (region_) os << "]";
  if (self_) os << ", bootstraps internally";
  return os.str();
}

Shape Recurrent::configure(const Shape& in) {
  if (in.size() != static_cast<size_t>(n_) * in_dim_)
    throw std::invalid_argument("recurrent: input size mismatch (" + in.str() + ")");
  in_ = Shape{n_, 1, in_dim_};
  out_ = last_ ? Shape{h_, 1, 1} : Shape{n_, 1, h_};
  return out_;
}

Layout Recurrent::configure_layout(const Layout& in, Packing packing) {
  (void)packing;
  in_layout_ = in;
  // Keep the input's token stride when the states fit in it (cheapest input
  // projection); otherwise a compact one.
  stride_ = (in.m == 1 && in.xs == 1 && in.cs >= static_cast<size_t>(h_)) ? in.cs : static_cast<size_t>(h_);
  region_ = pow2(static_cast<size_t>(n_) * stride_);
  const Layout states{stride_, static_cast<size_t>(h_), 1};
  out_layout_ = last_ ? Layout::compact(out_) : states;
  plain_slots_ = static_cast<uint32_t>(pow2(std::max<size_t>(in.span(in_), min_slots())));
  build();
  return out_layout_;
}

size_t Recurrent::min_slots() const {
  const size_t regions = cell_ == Cell::kRNN ? 1 : cell_ == Cell::kGRU ? 3 : 4;
  return std::max(in_layout_.span(in_), regions * region_);
}

// Slot matrices and series. Rows of the gate ciphertext: gate g of token t
// at g*R + t*B + j, scaled into its series' t in [-1, 1].
void Recurrent::build() {
  if (!region_) return;
  const size_t R = region_, B = stride_, H = static_cast<size_t>(h_);
  const size_t regions = cell_ == Cell::kRNN ? 1 : cell_ == Cell::kGRU ? 3 : 4;
  if (cell_ == Cell::kRNN) {
    gate_c_ = lbcrypto::EvalChebyshevCoefficients([a = gate_a_](double t) { return std::tanh(a * t); }, -1.0, 1.0,
                                                  degree_);
  } else {
    gate_c_ = lbcrypto::EvalChebyshevCoefficients([a = gate_a_](double t) { return sigmoid(a * t); }, -1.0, 1.0,
                                                  degree_);
    cell_c_ = lbcrypto::EvalChebyshevCoefficients([a = cell_a_](double t) { return std::tanh(a * t); }, -1.0, 1.0,
                                                  cell_degree_);
  }
  // GRU: W_hn h + b_hn shares the gates' ciphertext (and their sigmoid
  // series, whose output there is unused): divide it by a bound from the
  // weights (|h| <= 1) so it stays inside [-1, 1].
  hn_scale_ = 1.0;
  if (cell_ == Cell::kGRU) {
    for (size_t j = 0; j < H; ++j) {
      double s = std::fabs(b_hh_[2 * H + j]);
      for (size_t k = 0; k < H; ++k) s += std::fabs(w_hh_[(2 * H + j) * H + k]);
      hn_scale_ = std::max(hn_scale_, s / cell_a_);
    }
  }
  mask_.assign(regions * R, 0.0);
  for (size_t t = 0; t < static_cast<size_t>(n_); ++t)
    for (size_t j = 0; j < H; ++j) mask_[t * B + j] = 1.0;
  mask_in_.clear();
  if (cell_ == Cell::kLSTM) {
    mask_in_ = mask_;
    for (double& m : mask_in_) m /= cell_a_;
  } else if (cell_ == Cell::kGRU) {
    mask_in_ = mask_;
    for (double& m : mask_in_) m *= hn_scale_;
  }
  if (plain_slots_ < min_slots()) plain_slots_ = static_cast<uint32_t>(pow2(min_slots()));
  build_transforms(plain_slots_, &plain_x_, &plain_xn_, &plain_h_);
  if (slots_) build_transforms(slots_, &lt_x_, &lt_xn_, &lt_h_);
}

void Recurrent::build_transforms(uint32_t slots, LinearTransform* ltx, LinearTransform* ltxn, LinearTransform* lth) {
  const size_t R = region_, B = stride_, H = static_cast<size_t>(h_), I = static_cast<size_t>(in_dim_);
  const size_t N = static_cast<size_t>(n_);
  const size_t regions = cell_ == Cell::kRNN ? 1 : cell_ == Cell::kGRU ? 3 : 4;
  // Per gate: its region and the factor into t units (input side, recurrent side).
  auto scale = [&](size_t g, bool recurrent) {
    if (cell_ == Cell::kRNN) return 1.0 / gate_a_;
    if (cell_ == Cell::kLSTM) return (g == 2 ? 2.0 : 1.0) / gate_a_;
    if (g < 2) return 1.0 / gate_a_;
    return recurrent ? 1.0 / (cell_a_ * hn_scale_) : 1.0 / cell_a_;
  };
  const size_t xgates = cell_ == Cell::kGRU ? 2 : regions;
  SparseMatrix mx(slots, slots), mxn(slots, slots), mh(slots, slots);
  auto& bx = bias_x_;
  auto& bxn = bias_xn_;
  auto& bh = bias_h_;
  bx.assign(regions * R, 0.0);
  bxn.clear();
  bh.clear();
  for (size_t g = 0; g < xgates; ++g) {
    const double s = scale(g, false);
    for (size_t t = 0; t < N; ++t)
      for (size_t j = 0; j < H; ++j) {
        const size_t row = g * R + t * B + j;
        for (size_t i = 0; i < I; ++i)
          mx.add(row, in_layout_.slot(in_, t * I + i), w_ih_[(g * H + j) * I + i] * s);
        bx[row] = (b_ih_[g * H + j] + b_hh_[g * H + j]) * s;
      }
  }
  if (cell_ == Cell::kGRU) {
    bxn.assign(R, 0.0);
    bh.assign(regions * R, 0.0);
    const double s = scale(2, false), sh = scale(2, true);
    for (size_t t = 0; t < N; ++t)
      for (size_t j = 0; j < H; ++j) {
        for (size_t i = 0; i < I; ++i)
          mxn.add(t * B + j, in_layout_.slot(in_, t * I + i), w_ih_[(2 * H + j) * I + i] * s);
        bxn[t * B + j] = b_ih_[2 * H + j] * s;
        bh[2 * R + t * B + j] = b_hh_[2 * H + j] * sh;
      }
  }
  for (size_t g = 0; g < regions; ++g) {
    const double s = scale(g, true);
    for (size_t t = 1; t < N; ++t)
      for (size_t j = 0; j < H; ++j)
        for (size_t k = 0; k < H; ++k) mh.add(g * R + t * B + j, (t - 1) * B + k, w_hh_[(g * H + j) * H + k] * s);
  }
  *ltx = LinearTransform(mx, slots);
  *ltxn = cell_ == Cell::kGRU ? LinearTransform(mxn, slots) : LinearTransform();
  // A single token has no recurrence: an empty transform is never evaluated.
  *lth = LinearTransform(mh, slots);
}

void Recurrent::compile(uint32_t slots) {
  if (slots < min_slots()) throw std::invalid_argument("recurrent: slot count too small");
  slots_ = slots;
  build_transforms(slots_, &lt_x_, &lt_xn_, &lt_h_);
}

// The algorithm, shared by encrypted inference, the cleartext reference and
// level planning (see the top of recurrent.h). fit(v, need, site): refresh v
// first when the next `need` levels would not fit the stretch budget.
template <class Ops>
typename Ops::V Recurrent::run(Ops& ops, typename Ops::V x) const {
  using V = typename Ops::V;
  const uint32_t pd = poly_depth(degree_);
  const uint32_t pdc = cell_ == Cell::kRNN ? 0 : poly_depth(cell_degree_);
  const int64_t R = static_cast<int64_t>(region_);
  auto fit = [&](V& v, uint32_t need, int site) {
    ops.record(v, site);
    if (ops.fitting && ops.level(v) + need > ops.budget) v = ops.refresh(v, site);
  };
  fit(x, 1, kX);
  V u = ops.lt(0, x);
  fit(u, pd, kU);
  V un;
  if (cell_ == Cell::kGRU) {
    un = ops.lt(1, x);
    fit(un, pdc, kUN);
  }
  V h, c;
  for (int k = 0; k < n_; ++k) {
    const bool first = k == 0, final = k + 1 == n_;
    V pre;
    if (first) {
      pre = bias_h_.empty() ? u : ops.addp(u, bias_h_);
    } else {
      fit(h, 1 + pd, kH);
      pre = ops.add(u, ops.lt(2, h));
    }
    fit(pre, pd, kPre);
    if (cell_ == Cell::kRNN) {
      std::vector<double> gc = gate_c_;
      if (final && out_scale_ != 1.0)
        for (double& v : gc) v *= out_scale_;
      h = ops.series(pre, gc, 0);
      continue;
    }
    V s = ops.series(pre, gate_c_, 0);
    fit(s, 2, kS);
    // The previous token's state, shift(v) = token t-1's values in token
    // t's slots, is a rotation by -B; the masked products keep everything
    // outside the states' slots zero.
    const int64_t shift = -static_cast<int64_t>(stride_);
    if (cell_ == Cell::kLSTM) {
      V ig = ops.mul(ops.mulp(s, mask_in_), ops.twice_minus_one(ops.rot(s, 2 * R)));
      if (first) {
        c = ig;
      } else {
        fit(c, 1, kC);
        c = ops.add(ops.mul(ops.mulp(ops.rot(s, R), mask_), ops.rot(c, shift)), ig);
      }
      fit(c, pdc + 1, kC);
      std::vector<double> cc = cell_c_;
      if (final && out_scale_ != 1.0)
        for (double& v : cc) v *= out_scale_;
      V tc = ops.series(c, cc, 1);
      fit(tc, 1, kTC);
      h = ops.mul(ops.rot(s, 3 * R), tc);
    } else {  // GRU
      V tn = ops.add(un, ops.mul(ops.mulp(s, mask_in_), ops.rot(pre, 2 * R)));
      fit(tn, pdc, kTN);
      V n = ops.series(tn, cell_c_, 1);
      fit(n, 1, kN);
      V d = first ? ops.neg(n) : ops.sub(ops.rot(h, shift), n);
      const bool scaled = final && out_scale_ != 1.0;
      V z = ops.mulp(ops.rot(s, R), scaled ? zmask_scaled() : mask_);
      h = ops.add(scaled ? ops.scal(n, out_scale_) : n, ops.mul(z, d));
    }
  }
  if (last_ && n_ > 1) h = ops.rot(h, static_cast<int64_t>((n_ - 1) * stride_));
  return h;
}

std::vector<double> Recurrent::forward_plain(const std::vector<double>& x) const {
  if (x.size() != in_.size()) throw std::invalid_argument("recurrent: input size mismatch");
  SlotOps ops{plain_slots_, {&plain_x_, &plain_xn_, &plain_h_}, {&bias_x_, &bias_xn_, &bias_h_}};
  std::vector<double> xs(plain_slots_, 0.0);
  for (size_t i = 0; i < x.size(); ++i) xs[in_layout_.slot(in_, i)] = x[i];
  std::vector<double> h = run(ops, xs);
  std::vector<double> y(out_.size());
  for (size_t i = 0; i < y.size(); ++i) y[i] = h[out_layout_.slot(out_, i)];
  return y;
}

Ctxt Recurrent::forward(const FHEContext& ctx, const Ctxt& x) const {
  if (!slots_) throw std::logic_error("recurrent: compile() first");
  const bool fitting = self_ && ctx.stretch_budget();
  CtxtOps ops{ctx, {&lt_x_, &lt_xn_, &lt_h_}, {&bias_x_, &bias_xn_, &bias_h_}, bounds_, ctx.stretch_budget(), fitting};
  return run(ops, x);
}

Recurrent::LevelRun Recurrent::plan(uint32_t d, uint32_t budget, bool self) const {
  LevelOps ops{budget, self, bounds_};
  LevelOps::V x{d};
  ops.peak = d;
  LevelOps::V y = run(ops, x);
  LevelRun r;
  r.fits = !ops.fail && d <= budget;
  r.out = y.l;
  r.peak = ops.peak;
  r.boots = ops.boots;
  return r;
}

uint32_t Recurrent::depth() const { return plan(0, 1u << 30, false).out; }

uint32_t Recurrent::level_after(uint32_t d, uint32_t budget) const {
  if (!self_) return Layer::level_after(d, budget);
  LevelRun r = plan(d, budget, true);
  return r.fits ? r.out : kNoFit;
}

uint32_t Recurrent::level_peak(uint32_t d, uint32_t budget) const {
  if (!self_) return Layer::level_peak(d, budget);
  LevelRun r = plan(d, budget, true);
  return r.fits ? r.peak : kNoFit;
}

uint32_t Recurrent::internal_bootstraps(uint32_t d, uint32_t budget) const {
  if (!self_) return 0;
  LevelRun r = plan(d, budget, true);
  return r.fits ? r.boots : 0;
}

void Recurrent::calibrate(const std::vector<std::vector<double>>& inputs, double headroom) {
  std::vector<double> m(kSites, 0.0);
  for (const auto& x : inputs) {
    SlotOps ops{plain_slots_, {&plain_x_, &plain_xn_, &plain_h_}, {&bias_x_, &bias_xn_, &bias_h_}};
    std::vector<double> xs(plain_slots_, 0.0);
    for (size_t i = 0; i < x.size() && i < in_.size(); ++i) xs[in_layout_.slot(in_, i)] = x[i];
    run(ops, xs);
    for (size_t s = 0; s < m.size(); ++s) m[s] = std::max(m[s], ops.rec[s]);
  }
  // Only the input and its projections are refreshed with a scale; the
  // other sites lie within [-1, 1] by construction.
  for (int s : {kX, kU, kUN}) bounds_[static_cast<size_t>(s)] = headroom * m[static_cast<size_t>(s)];
}

void Recurrent::set_bounds(std::vector<double> b) {
  if (b.size() != kSites) throw std::invalid_argument("recurrent: need one bound per refresh site");
  bounds_ = std::move(b);
}

std::array<double, 2> Recurrent::series_ranges(const std::vector<std::vector<double>>& inputs) const {
  std::array<double, 2> r{0.0, 0.0};
  for (const auto& x : inputs) {
    if (x.size() != in_.size()) throw std::invalid_argument("recurrent: input size mismatch");
    SlotOps ops{plain_slots_, {&plain_x_, &plain_xn_, &plain_h_}, {&bias_x_, &bias_xn_, &bias_h_}};
    std::vector<double> xs(plain_slots_, 0.0);
    for (size_t i = 0; i < x.size(); ++i) xs[in_layout_.slot(in_, i)] = x[i];
    run(ops, xs);
    r[0] = std::max(r[0], ops.series_max[0]);
    r[1] = std::max(r[1], ops.series_max[1]);
  }
  return r;
}

std::vector<int32_t> Recurrent::rotation_indices() const {
  std::vector<int32_t> r = lt_x_.rotation_indices();
  for (const auto* lt : {&lt_xn_, &lt_h_}) {
    auto v = lt->rotation_indices();
    r.insert(r.end(), v.begin(), v.end());
  }
  const int64_t s = slots_;
  auto add = [&](int64_t k) {
    if (s) r.push_back(static_cast<int32_t>(((k % s) + s) % s));
  };
  for (int g = 1; g < (cell_ == Cell::kRNN ? 1 : 4); ++g)
    if (!(cell_ == Cell::kGRU && g == 3)) add(static_cast<int64_t>(g * region_));
  if (last_ && n_ > 1) add(static_cast<int64_t>((n_ - 1) * stride_));
  if (cell_ != Cell::kRNN && n_ > 1) add(-static_cast<int64_t>(stride_));
  std::sort(r.begin(), r.end());
  r.erase(std::unique(r.begin(), r.end()), r.end());
  r.erase(std::remove(r.begin(), r.end(), 0), r.end());
  return r;
}

void Recurrent::absorb_input_scale(double c) {
  if (c == 0.0) throw std::invalid_argument("recurrent: zero input scale");
  for (double& w : w_ih_) w /= c;
  build();
}

void Recurrent::scale_output(double f) { out_scale_ *= f; }

std::vector<double> Recurrent::params() const {
  std::vector<double> p(w_ih_);
  p.insert(p.end(), w_hh_.begin(), w_hh_.end());
  p.insert(p.end(), b_ih_.begin(), b_ih_.end());
  p.insert(p.end(), b_hh_.begin(), b_hh_.end());
  return p;
}

void Recurrent::set_params(const std::vector<double>& p) {
  const size_t a = w_ih_.size(), b = w_hh_.size(), g = b_ih_.size();
  if (p.size() != a + b + 2 * g) throw std::invalid_argument("recurrent: wrong parameter count");
  std::copy(p.begin(), p.begin() + static_cast<long>(a), w_ih_.begin());
  std::copy(p.begin() + static_cast<long>(a), p.begin() + static_cast<long>(a + b), w_hh_.begin());
  std::copy(p.begin() + static_cast<long>(a + b), p.begin() + static_cast<long>(a + b + g), b_ih_.begin());
  std::copy(p.begin() + static_cast<long>(a + b + g), p.end(), b_hh_.begin());
  build();
}

void Recurrent::save(std::ostream& os) const {
  os << std::setprecision(std::numeric_limits<double>::max_digits10) << "recurrent " << cell_name(cell_) << " " << n_
     << " " << in_dim_ << " " << h_ << " " << gate_a_ << " " << degree_ << " " << cell_a_ << " " << cell_degree_ << " "
     << (last_ ? 1 : 0) << " " << out_scale_ << " " << (self_ ? 1 : 0);
  for (double b : bounds_) os << " " << b;
  os << "\n";
  write_vec(os, w_ih_);
  write_vec(os, w_hh_);
  write_vec(os, b_ih_);
  write_vec(os, b_hh_);
}

}  // namespace fheml
