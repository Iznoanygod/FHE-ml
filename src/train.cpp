#include "fheml/train.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include "fheml/transformer.h"
#include "math/chebyshev.h"

namespace fheml {

namespace {

// Level of c after its pending rescale (FLEXIBLEAUTO rescales lazily).
uint32_t lvl(const Ctxt& c) { return c->GetLevel() + (c->GetNoiseScaleDeg() > 1 ? 1 : 0); }

Ctxt mul_plain(const FHEContext& ctx, const Ctxt& c, const std::vector<double>& v) {
  Ptxt p = ctx.encode(v, lvl(c));
  return ctx.cc()->EvalMult(c, p);
}

Ctxt add_plain(const FHEContext& ctx, const Ctxt& c, const std::vector<double>& v) {
  Ptxt p = ctx.encode(v, c->GetLevel(), c->GetNoiseScaleDeg());
  return ctx.cc()->EvalAdd(c, p);
}

Ctxt sum(const FHEContext& ctx, std::vector<Ctxt> v) {
  // Pairwise, so the additions stay balanced.
  for (size_t step = 1; step < v.size(); step *= 2)
    for (size_t i = 0; i + step < v.size(); i += 2 * step) ctx.cc()->EvalAddInPlace(v[i], v[i + step]);
  return v.at(0);
}

std::vector<double> scaled(const std::vector<double>& mask, double s) {
  std::vector<double> v(mask);
  for (double& x : v) x *= s;
  return v;
}

// acc += c. Not in place: the first c is shared, not copied, and may still
// be in use (a batch's gradients).
void add_to(const FHEContext& ctx, Ctxt& acc, const Ctxt& c) {
  acc = acc ? ctx.cc()->EvalAdd(acc, c) : c;
}

}  // namespace

// Bootstrapping inside a step (Trainer::set_level_budget). Levels are counted
// from `base`, the level weights and inputs are encrypted at; fit(c, d) makes
// sure c has d levels left within the budget, bootstrapping it when not. The
// passes call fit on every operand before an operation that consumes levels,
// and never inside a parallel loop.
struct Trainer::Refresh {
  const FHEContext* ctx = nullptr;
  int base = 0;
  uint32_t budget = 0;  // 0: never
  uint32_t log_range = 0;
  size_t count = 0;

  int level(const Ctxt& c) const { return static_cast<int>(lvl(c)) - base; }

  void fit(Ctxt& c, uint32_t depth) {
    if (!budget || !c || level(c) + static_cast<int>(depth) <= static_cast<int>(budget)) return;
    if (depth > budget)
      throw std::invalid_argument("train: an operation needs " + std::to_string(depth) +
                                  " levels, more than the level budget " + std::to_string(budget));
    const CC& cc = ctx->cc();
    // Into [-1, 1] for the bootstrap (one level), back with doublings (none).
    Ctxt t = log_range ? cc->EvalMult(c, std::ldexp(1.0, -static_cast<int>(log_range))) : c;
    t = ctx->bootstrap(t);
    for (uint32_t i = 0; i < log_range; ++i) t = cc->EvalAdd(t, t);
    c = t;
    ++count;
  }

  void fit_all(std::vector<Ctxt>& v, uint32_t depth) {
    for (Ctxt& c : v) fit(c, depth);
  }
};

namespace {

using Refresh = Trainer::Refresh;

// Levels of a Chebyshev series of this degree on [a, b].
uint32_t chebyshev_depth(uint32_t degree, double a, double b) {
  return poly_depth(degree) + (a == -1.0 && b == 1.0 ? 0 : 1);
}

// d/dx of sum_k c_k x^k.
std::vector<double> poly_derivative(const std::vector<double>& c) {
  std::vector<double> d;
  for (size_t k = 1; k < c.size(); ++k) d.push_back(static_cast<double>(k) * c[k]);
  if (d.empty()) d.push_back(0.0);
  return d;
}

// d/dx of the Chebyshev series c0/2 + sum c_k T_k(t), t = (2x - a - b)/(b - a),
// in the same convention (Numerical Recipes chder, times dt/dx).
std::vector<double> chebyshev_derivative(const std::vector<double>& c, double a, double b) {
  const size_t n = c.size() - 1;  // degree
  if (n == 0) return {0.0};
  std::vector<double> d(n + 1, 0.0);  // d[n] stays 0 (makes the recurrence uniform)
  for (size_t j = n; j >= 1; --j) d[j - 1] = (j + 1 <= n ? d[j + 1] : 0.0) + 2.0 * static_cast<double>(j) * c[j];
  d.pop_back();
  for (double& v : d) v *= 2.0 / (b - a);
  return d;
}

double chebyshev_eval(const std::vector<double>& c, double a, double b, double x) {
  const double t = (2.0 * x - a - b) / (b - a);
  double t0 = 1.0, t1 = t, y = c[0] / 2.0 + (c.size() > 1 ? c[1] * t : 0.0);
  for (size_t k = 2; k < c.size(); ++k) {
    double t2 = 2.0 * t * t1 - t0;
    t0 = t1;
    t1 = t2;
    y += c[k] * t2;
  }
  return y;
}

double poly_eval(const std::vector<double>& c, double x) {
  double acc = 0.0;
  for (size_t k = c.size(); k-- > 0;) acc = acc * x + c[k];
  return acc;
}

// Derivative of an activation layer at z, as the same polynomial the
// encrypted backward pass evaluates.
std::vector<double> activation_derivative_plain(const Layer& l, const std::vector<double>& z) {
  std::vector<double> d(z.size());
  if (dynamic_cast<const Square*>(&l)) {
    for (size_t i = 0; i < z.size(); ++i) d[i] = 2.0 * z[i];
  } else if (auto* p = dynamic_cast<const PolyActivation*>(&l)) {
    auto dc = poly_derivative(p->coeffs());
    for (size_t i = 0; i < z.size(); ++i) d[i] = poly_eval(dc, z[i]);
  } else if (auto* c = dynamic_cast<const ChebyshevActivation*>(&l)) {
    auto dc = chebyshev_derivative(c->coeffs(), c->a(), c->b());
    for (size_t i = 0; i < z.size(); ++i) d[i] = chebyshev_eval(dc, c->a(), c->b(), z[i]);
  } else if (auto* r = dynamic_cast<const ExactReLU*>(&l)) {
    for (size_t i = 0; i < z.size(); ++i) d[i] = z[i] > 0.0 ? r->out_scale() : 0.0;
  } else {
    throw std::logic_error("no derivative for " + l.type());
  }
  return d;
}

bool contains(const std::vector<size_t>& v, size_t x) { return std::find(v.begin(), v.end(), x) != v.end(); }

bool has_trainable(const Layer& l) {
  if (l.trainable()) return true;
  if (auto* r = dynamic_cast<const Residual*>(&l)) {
    for (const auto& b : r->body())
      if (has_trainable(*b)) return true;
  }
  return false;
}

size_t first_trainable(const Network& net, const std::vector<size_t>& frozen) {
  const auto& layers = net.layers();
  for (size_t i = 0; i < layers.size(); ++i)
    if (!contains(frozen, i) && has_trainable(*layers[i])) return i;
  throw std::invalid_argument("train: the network has no trainable (unfrozen) layer");
}

}  // namespace

// ----------------------------------------------------------------------- Node

struct Trainer::Node {
  enum Kind { kLinear, kSquare, kPoly, kCheb, kBoot, kResidual, kNorm, kAttn, kMax, kRelu, kBatchNorm, kDropout } kind = kLinear;
  Layer* layer = nullptr;
  bool trainable = false;

  // kLinear: y = sum_i D_i (*) rot_{offs[i]}(x) + B
  // kNorm: y = D_0 (*) (x - mean) (*) r + B with gamma in D_0 and beta in B
  // (the diagonal/bias machinery below handles their sharing over the tokens).
  LinearLayer* lin = nullptr;
  LayerNorm* norm = nullptr;
  std::vector<int32_t> offs;
  std::vector<std::vector<double>> diag;  // initial values (and the constants of a frozen layer)
  std::vector<double> bias;               // in output slots; empty = no bias
  // Trainable: which parameter owns each slot (-1 = none), per diagonal and for the bias.
  std::vector<std::vector<int>> owner;
  std::vector<int> bias_owner;
  // Update plan: masks of unshared weights, and one group per shared weight.
  std::vector<std::vector<double>> unshared;  // per diagonal (empty = none)
  std::vector<double> bias_unshared;
  struct Group {
    std::vector<std::pair<int, std::vector<double>>> masks;  // (diagonal, -1 = bias), 0/1 mask
  };
  std::vector<Group> groups;
  // Every weight ciphertext is zero outside slots [0, window): bootstrapping
  // packs slots / window of them into one ciphertext.
  uint32_t window = 0;
  // Encrypted state.
  std::vector<Ctxt> D, gD;
  Ctxt B, gB;
  Ctxt in;  // forward input of the current sample

  // Activations: derivative polynomial (power basis or Chebyshev series).
  std::vector<double> dcoef;
  double a = -1.0, b = 1.0;

  // kNorm: x - mean and 1/sqrt(var + eps) of the current sample.
  Ctxt centred, r;

  // kAttn: body holds the four projection nodes (q, k, v, o); the rest is
  // the current sample's attention internals (see attention_forward).
  Attention* att = nullptr;
  // Evaluated token offsets (Attention::offsets(): those pairing some query
  // with a key it may attend to) and, per offset, the head mask and the
  // quadratic score coefficients with the masked-out pairs zeroed.
  std::vector<int> aoffs;
  std::vector<std::vector<double>> ahead, ac2, ac1, ac0;
  std::vector<Ctxt> kr, vr;  // rotated (cyclic) keys and values per token offset
  std::vector<Ctxt> p, aw, A;  // scores, head-slot weights, weights broadcast over the head
  Ctxt q, o, y;               // queries, sum_s A_s (*) v_s before the softmax division, 1 / row sum

  // kMax: per tournament round (see MaxPool2d), the comparison bit [a < b]
  // and its complement, both masked to the slots the round writes (the
  // output slots in the last round; the backward pass routes the gradient
  // with them).
  MaxPool2d* pool = nullptr;
  std::vector<std::vector<double>> round_mask;
  std::vector<Ctxt> pick_b, pick_a;

  // kRelu: the forward's step out_scale [x > 0] (the derivative of
  // out_scale relu(x)), so the backward pass needs no comparison.
  ExactReLU* relu = nullptr;
  Ctxt step;

  // kBatchNorm: gamma in D_0 and beta in B like kNorm. The batch statistics
  // couple the samples (see forward_batch): per batch size, the linear map
  // that averages each channel over the batch's summed ciphertext; the
  // shared 1/sqrt(var + eps) of the batch (r) and the running statistics,
  // broadcast over each channel's slots and encrypted like weights.
  BatchNorm* bn = nullptr;
  std::map<size_t, std::shared_ptr<LinearMap>> channel_mean;
  std::vector<double> var_pad;  // 1/sqrt's interval midpoint outside the layout
  std::vector<std::vector<uint32_t>> channel_slots;
  Ctxt rm, rv;

  // Per-sample state of the forward pass, kept between forward and backward
  // while a batch is processed layer by layer (forward_batch): the node's
  // own fields are swapped in and out (attention: with its projections).
  // kDropout: the layer (its mask generator); the sample's mask in slots.
  Dropout* drop = nullptr;
  std::vector<double> dmask;

  struct Sample {
    std::vector<double> dmask;
    Ctxt in, centred, r, q, o, y, step;
    std::vector<Ctxt> kr, vr, p, aw, A, pick_b, pick_a;
    std::vector<Sample> body;
  };
  std::vector<Sample> saved;

  std::vector<std::unique_ptr<Node>> body;  // kResidual, kAttn
};

// Softmax cross-entropy (see Loss): the exp and 1/x series, and for the
// encrypted pass the output-slot masks.
struct Trainer::CrossEntropy {
  SoftmaxOptions o;
  size_t k = 0;  // outputs
  double exp_lo = 0.0, exp_hi = 0.0, sum_lo = 0.0, sum_hi = 0.0;
  std::vector<double> exp_c, inv_c;

  // Encrypted pass only (set_layout). mask: 1 at the output slots; pad: added
  // to the masked logits so that every slot is inside the exp interval;
  // first: 1 at slot 0; window: power of two covering the output slots (the
  // max tree); mask_input: the logits' padding slots may be nonzero.
  bool mask_input = false;
  int window = 1;
  std::vector<double> mask, pad, first;

  CrossEntropy(const SoftmaxOptions& opts, size_t outputs) : o(opts), k(outputs) {
    if (!(o.lo < o.hi)) throw std::invalid_argument("cross-entropy softmax: need lo < hi");
    if (o.exp_degree < 1 || o.inv_degree < 1) throw std::invalid_argument("cross-entropy softmax: degrees must be >= 1");
    if ((o.sum_lo != 0.0 || o.sum_hi != 0.0) && !(0.0 < o.sum_lo && o.sum_lo < o.sum_hi))
      throw std::invalid_argument("cross-entropy softmax: need 0 < sum_lo < sum_hi");
    const double n = static_cast<double>(k);
    exp_lo = o.subtract_max ? o.lo - o.hi : o.lo;
    exp_hi = o.subtract_max ? 0.0 : o.hi;
    sum_lo = o.sum_lo != 0.0 ? o.sum_lo : (o.subtract_max ? 1.0 : n * std::exp(o.lo));
    sum_hi = o.sum_hi != 0.0 ? o.sum_hi : (o.subtract_max ? n : n * std::exp(o.hi));
    if (o.subtract_max && sum_lo == sum_hi) sum_hi = sum_lo + 1.0;  // one output: the sum is exactly 1
    exp_c = lbcrypto::EvalChebyshevCoefficients([](double x) { return std::exp(x); }, exp_lo, exp_hi, o.exp_degree);
    inv_c = lbcrypto::EvalChebyshevCoefficients([](double x) { return 1.0 / x; }, sum_lo, sum_hi, o.inv_degree);
  }

  void set_layout(const Layer& last, uint32_t slots, bool linear_last) {
    const Shape& out = last.out_shape();
    const Layout& ol = last.out_layout();
    mask_input = !linear_last;
    window = 1;
    while (static_cast<size_t>(window) < ol.span(out)) window *= 2;
    mask.assign(slots, 0.0);
    first.assign(slots, 0.0);
    first[0] = 1.0;
    for (size_t i = 0; i < k; ++i) mask[ol.slot(out, i)] = 1.0;
    // Without the maximum: the interval midpoint. With it: lo inside the max
    // window (never above a logit), and outside it a value whose difference
    // to the (zero) broadcast maximum is inside [lo - hi, 0].
    pad.assign(slots, 0.0);
    for (uint32_t s = 0; s < slots; ++s)
      if (mask[s] == 0.0)
        pad[s] = !o.subtract_max ? 0.5 * (o.lo + o.hi) : (s < static_cast<uint32_t>(window) ? o.lo : 0.5 * (o.lo - o.hi));
  }

  double inv(double x) const {
    double y = lbcrypto::EvalChebyshevFunctionPtxt([](double u) { return 1.0 / u; }, {x}, sum_lo, sum_hi, o.inv_degree)[0];
    for (uint32_t i = 0; i < o.newton; ++i) y = y * (2.0 - x * y);
    return y;
  }

  std::vector<double> softmax(std::vector<double> z) const {
    if (o.subtract_max) {
      const double m = *std::max_element(z.begin(), z.end());
      for (double& v : z) v -= m;
    }
    std::vector<double> e =
        lbcrypto::EvalChebyshevFunctionPtxt([](double x) { return std::exp(x); }, z, exp_lo, exp_hi, o.exp_degree);
    double S = 0.0;
    for (double v : e) S += v;
    const double r = inv(S);
    for (double& v : e) v *= r;
    return e;
  }
};

std::vector<double> softmax_plain(const Loss& loss, const std::vector<double>& y) {
  return Trainer::CrossEntropy(loss.softmax, y.size()).softmax(y);
}

double cross_entropy_plain(const std::vector<double>& y, const std::vector<double>& t) {
  if (y.size() != t.size() || y.empty()) throw std::invalid_argument("cross_entropy_plain: size mismatch");
  const double m = *std::max_element(y.begin(), y.end());
  double S = 0.0;
  for (double v : y) S += std::exp(v - m);
  const double lse = m + std::log(S);
  double l = 0.0;
  for (size_t i = 0; i < y.size(); ++i) l += t[i] * (lse - y[i]);
  return l;
}

namespace {

using Node = Trainer::Node;
using CrossEntropy = Trainer::CrossEntropy;

std::unique_ptr<Node> build(Layer* l, bool train, uint32_t slots) {
  auto n = std::make_unique<Node>();
  n->layer = l;
  if (auto* r = dynamic_cast<Residual*>(l)) {
    n->kind = Node::kResidual;
    for (const auto& b : r->body()) n->body.push_back(build(b.get(), train, slots));
    return n;
  }
  if (dynamic_cast<Square*>(l)) {
    n->kind = Node::kSquare;
    return n;
  }
  if (auto* p = dynamic_cast<PolyActivation*>(l)) {
    n->kind = Node::kPoly;
    n->dcoef = poly_derivative(p->coeffs());
    while (n->dcoef.size() > 1 && n->dcoef.back() == 0.0) n->dcoef.pop_back();
    return n;
  }
  if (auto* c = dynamic_cast<ChebyshevActivation*>(l)) {
    n->kind = Node::kCheb;
    n->a = c->a();
    n->b = c->b();
    n->dcoef = chebyshev_derivative(c->coeffs(), c->a(), c->b());
    return n;
  }
  if (dynamic_cast<Bootstrap*>(l)) {
    n->kind = Node::kBoot;
    return n;
  }
  if (auto* dr = dynamic_cast<Dropout*>(l)) {
    n->kind = Node::kDropout;
    n->drop = dr;
    return n;
  }
  if (auto* er = dynamic_cast<ExactReLU*>(l)) {
    n->kind = Node::kRelu;
    n->relu = er;
    return n;
  }
  if (auto* mp = dynamic_cast<MaxPool2d*>(l)) {
    n->kind = Node::kMax;
    n->pool = mp;
    for (uint32_t h = mp->blocks() / 2; h >= 1; h /= 2) {
      std::vector<double> m(slots, 0.0);
      if (h > 1)
        std::fill(m.begin(), m.begin() + static_cast<long>(h) * mp->block(), 1.0);
      else
        for (size_t i = 0; i < mp->out_shape().size(); ++i) m[mp->out_layout().slot(mp->out_shape(), i)] = 1.0;
      n->round_mask.push_back(std::move(m));
    }
    return n;
  }
  if (auto* at = dynamic_cast<Attention*>(l)) {
    n->kind = Node::kAttn;
    n->att = at;
    n->aoffs = at->offsets();
    for (size_t k = 0; k < n->aoffs.size(); ++k) {
      const int ki = static_cast<int>(k);
      if (at->softmax()) {
        n->ahead.push_back(at->head_mask(ki));
      } else {
        n->ac2.push_back(at->masked(at->score_mask(2), ki));
        n->ac1.push_back(at->masked(at->score_mask(1), ki));
        n->ac0.push_back(at->masked(at->score_mask(0), ki));
      }
    }
    n->trainable = train;
    for (int i = 0; i < 4; ++i) n->body.push_back(build(at->projection(i).get(), train, slots));
    return n;
  }
  if (auto* ln = dynamic_cast<LayerNorm*>(l)) {
    n->kind = Node::kNorm;
    n->norm = ln;
    n->trainable = train;
    const Shape& out = ln->out_shape();
    const Layout& ol = ln->out_layout();
    n->window = 1;
    while (n->window < ol.span(out)) n->window *= 2;
    const int d = ln->dim(), tokens = ln->tokens();
    // gamma and beta: one [d] set per token segment (one without segments).
    const size_t na = ln->gamma().size(), nb = na;
    n->offs = {0};
    n->diag.emplace_back(slots, 0.0);
    n->bias.assign(slots, 0.0);
    n->unshared.assign(1, {});
    if (n->trainable) {
      n->owner.emplace_back(slots, -1);
      n->bias_owner.assign(slots, -1);
    }
    std::vector<std::vector<double>> masks(na, std::vector<double>(slots, 0.0));
    std::vector<int> users(na, 0);  // tokens sharing each gamma/beta element
    for (int t = 0; t < tokens; ++t)
      for (int j = 0; j < d; ++j) {
        const size_t sl = ol.slot(out, static_cast<size_t>(t) * d + j), a = ln->affine_index(t, j);
        n->diag[0][sl] = ln->gamma()[a];
        n->bias[sl] = ln->beta()[a];
        if (n->trainable) {
          n->owner[0][sl] = static_cast<int>(a);
          n->bias_owner[sl] = static_cast<int>(nb + a);
        }
        masks[a][sl] = 1.0;
        ++users[a];
      }
    if (n->trainable) {
      // gamma and beta elements used by one token are updated in place; the
      // others (shared by the tokens of a segment) through a masked sum.
      for (size_t a = 0; a < na; ++a) {
        if (users[a] == 1) {
          if (n->unshared[0].empty()) {
            n->unshared[0].assign(slots, 0.0);
            n->bias_unshared.assign(slots, 0.0);
          }
          for (uint32_t sl = 0; sl < slots; ++sl)
            if (masks[a][sl] != 0.0) n->unshared[0][sl] = n->bias_unshared[sl] = 1.0;
        } else if (users[a] > 1) {
          Node::Group gg, gb;
          gg.masks.emplace_back(0, masks[a]);
          gb.masks.emplace_back(-1, masks[a]);
          n->groups.push_back(std::move(gg));
          n->groups.push_back(std::move(gb));
        }
      }
    }
    return n;
  }
  if (auto* bn = dynamic_cast<BatchNorm*>(l)) {
    n->kind = Node::kBatchNorm;
    n->bn = bn;
    n->trainable = train;
    const Shape& out = bn->out_shape();
    const Layout& ol = bn->out_layout();
    n->window = 1;
    while (n->window < ol.span(out)) n->window *= 2;
    const size_t C = bn->channels(), hw = bn->positions();
    n->offs = {0};
    n->diag.emplace_back(slots, 0.0);
    n->bias.assign(slots, 0.0);
    n->unshared.assign(1, {});
    n->var_pad.assign(slots, 0.5 * (bn->var_min() + bn->var_max()));
    n->channel_slots.assign(C, {});
    if (n->trainable) {
      n->owner.emplace_back(slots, -1);
      n->bias_owner.assign(slots, -1);
    }
    std::vector<std::vector<double>> masks(C, std::vector<double>(slots, 0.0));
    for (size_t i = 0; i < out.size(); ++i) {
      const size_t c = i / hw;
      const uint32_t sl = static_cast<uint32_t>(ol.slot(out, i));
      n->diag[0][sl] = bn->gamma()[c];
      n->bias[sl] = bn->beta()[c];
      n->var_pad[sl] = 0.0;
      n->channel_slots[c].push_back(sl);
      masks[c][sl] = 1.0;
      if (n->trainable) {
        n->owner[0][sl] = static_cast<int>(c);
        n->bias_owner[sl] = static_cast<int>(C + c);
      }
    }
    if (n->trainable) {
      if (hw == 1) {
        n->unshared[0].assign(slots, 0.0);
        n->bias_unshared.assign(slots, 0.0);
        for (const auto& cs : n->channel_slots)
          for (uint32_t sl : cs) n->unshared[0][sl] = n->bias_unshared[sl] = 1.0;
      } else {
        for (size_t c = 0; c < C; ++c) {  // gamma_c and beta_c are shared by the channel's positions
          Node::Group gg, gb;
          gg.masks.emplace_back(0, masks[c]);
          gb.masks.emplace_back(-1, masks[c]);
          n->groups.push_back(std::move(gg));
          n->groups.push_back(std::move(gb));
        }
      }
    }
    return n;
  }
  auto* lin = dynamic_cast<LinearLayer*>(l);
  if (!lin)
    throw std::invalid_argument("train: cannot backpropagate through a " + l->type() +
                                " layer; only layers before the first trainable one may be of this type");

  n->kind = Node::kLinear;
  n->lin = lin;
  n->trainable = train && lin->trainable();
  const Shape& in = lin->in_shape();
  const Shape& out = lin->out_shape();
  const Layout& il = lin->in_layout();
  const Layout& ol = lin->out_layout();

  n->window = 1;
  while (n->window < ol.span(out)) n->window *= 2;
  std::map<int32_t, size_t> index;
  auto diag_of = [&](size_t rs, size_t cs) {
    int32_t d = normalize_rotation(static_cast<int64_t>(cs) - static_cast<int64_t>(rs), slots);
    auto it = index.find(d);
    if (it != index.end()) return it->second;
    index[d] = n->offs.size();
    n->offs.push_back(d);
    n->diag.emplace_back(slots, 0.0);
    if (n->trainable) n->owner.emplace_back(slots, -1);
    return n->offs.size() - 1;
  };

  if (n->trainable) {
    const std::vector<double> p = lin->params();
    const auto refs = lin->weight_refs();
    const auto brefs = lin->bias_refs();
    std::vector<int> count(p.size(), 0);
    for (const auto& r : refs) ++count.at(static_cast<size_t>(r.param));
    for (int r : brefs) ++count.at(static_cast<size_t>(r));
    // Shared parameter -> its slots, grouped by diagonal.
    std::map<int, std::map<int, std::vector<double>>> shared;
    for (const auto& r : refs) {
      size_t rs = ol.slot(out, r.row), cs = il.slot(in, r.col);
      size_t i = diag_of(rs, cs);
      if (n->owner[i][rs] != -1) throw std::logic_error("train: two weights share one slot");
      n->owner[i][rs] = r.param;
      n->diag[i][rs] = p[static_cast<size_t>(r.param)];
    }
    n->unshared.assign(n->offs.size(), {});
    for (size_t i = 0; i < n->offs.size(); ++i)
      for (uint32_t s = 0; s < slots; ++s) {
        int q = n->owner[i][s];
        if (q < 0) continue;
        if (count[static_cast<size_t>(q)] == 1) {
          if (n->unshared[i].empty()) n->unshared[i].assign(slots, 0.0);
          n->unshared[i][s] = 1.0;
        } else {
          auto& m = shared[q][static_cast<int>(i)];
          if (m.empty()) m.assign(slots, 0.0);
          m[s] = 1.0;
        }
      }
    if (!brefs.empty()) {
      n->bias.assign(slots, 0.0);
      n->bias_owner.assign(slots, -1);
      for (size_t r = 0; r < brefs.size(); ++r) {
        size_t rs = ol.slot(out, r);
        n->bias_owner[rs] = brefs[r];
        n->bias[rs] = p[static_cast<size_t>(brefs[r])];
        if (count[static_cast<size_t>(brefs[r])] == 1) {
          if (n->bias_unshared.empty()) n->bias_unshared.assign(slots, 0.0);
          n->bias_unshared[rs] = 1.0;
        } else {
          auto& m = shared[brefs[r]][-1];
          if (m.empty()) m.assign(slots, 0.0);
          m[rs] = 1.0;
        }
      }
    }
    for (auto& kv : shared) {
      Node::Group g;
      for (auto& dm : kv.second) g.masks.emplace_back(dm.first, std::move(dm.second));
      n->groups.push_back(std::move(g));
    }
  } else {
    const SparseMatrix m = lin->matrix();
    for (size_t r = 0; r < m.rows(); ++r) {
      size_t rs = ol.slot(out, r);
      for (const auto& e : m.row(r)) n->diag[diag_of(rs, il.slot(in, e.first))][rs] += e.second;
    }
    const std::vector<double> b = lin->bias();
    if (!b.empty()) {
      n->bias.assign(slots, 0.0);
      for (size_t r = 0; r < b.size(); ++r) n->bias[ol.slot(out, r)] = b[r];
    }
  }
  return n;
}

// ----------------------------------------------------------- level planning

// Mirrors the encrypted step with integer levels. w = level of the weights.
struct Sim {
  uint32_t w;
  uint32_t max = 0;
  uint32_t new_w = 0;
  bool eval = false;  // inference with the trained weights (batch norms use their running statistics)
  uint32_t see(uint32_t l) {
    max = std::max(max, l);
    return l;
  }
};

struct SimTrace {
  std::vector<uint32_t> in;  // input level of each node
  std::vector<SimTrace> body;
  // Batch norms: levels of x_hat and r (training) for the backward pass.
  std::vector<std::pair<uint32_t, uint32_t>> bn;
};

// Levels of the attention internals of forward (see attention_forward), from
// the input level z.
struct AttnLevels {
  uint32_t q, p, a, osum, y, o, out;
};

AttnLevels attention_levels(Sim& s, const Node& n, uint32_t z) {
  const Attention& at = *n.att;
  AttnLevels L{};
  L.q = s.see(std::max(z, n.body[0]->trainable ? s.w : z) + 1);
  L.p = s.see(L.q + 1);
  if (at.softmax()) {
    const SoftmaxOptions& o = at.softmax_options();
    if (o.subtract_max) {
      uint32_t rounds = 0;
      for (int m = at.tokens(); m > 1; m = (m + 1) / 2) ++rounds;
      L.p = s.see(std::max(L.p, FHEContext::compare_depth() + rounds));
    }
    L.a = s.see(L.p + poly_depth(o.exp_degree) + 1 + 1);
    L.osum = s.see(L.a + 1);
    L.y = s.see(L.a + poly_depth(o.inv_degree) + 1 + 2 * o.newton);
    L.o = s.see(std::max(L.osum, L.y) + 1);
  } else {
    L.a = s.see(L.p + 2);
    L.osum = L.o = s.see(L.a + 1);
    L.y = 0;
  }
  L.out = s.see(std::max(L.o, n.body[3]->trainable ? s.w : L.o) + 1);
  return L;
}

// MaxPool2d forward from input level x: the gather, then every round
// compares (inputs two levels above the bottom) and selects with the fresh
// comparison bit; the masked bits sit one level below it.
uint32_t sim_maxpool_forward(Sim& s, const Node& n, uint32_t x) {
  const uint32_t c = FHEContext::compare_depth();
  x = s.see(x + 1);
  if (n.pool->rounds()) s.see(c + 1);
  for (uint32_t r = 0; r < n.pool->rounds(); ++r) {
    s.see(x + 2);
    x = s.see(std::max(x, c) + 1);
  }
  return x;
}

// Level of the stored step out_scale [x > 0] of an exact ReLU (fresh from
// the comparison at compare_depth(), one more with an output scale).
uint32_t relu_step_level(Sim& s, const Node& n) {
  const uint32_t c = s.see(FHEContext::compare_depth());
  return n.relu->out_scale() == 1.0 ? c : s.see(c + 1);
}

uint32_t sim_linear_backward(Sim& s, const Node& n, uint32_t g, uint32_t z) {
  if (n.trainable) {
    uint32_t grad = s.see(std::max(g, z) + 1);
    uint32_t upd = s.see(grad + (n.groups.empty() ? 1 : 2));
    s.new_w = std::max(s.new_w, std::max(s.w, upd));
  }
  return s.see(std::max(g, n.trainable ? s.w : g) + 1);
}

// Batch norm forward from input level x: the batch statistics in training
// (s.eval = false; with the running statistics' update), else the running
// statistics (at the weights' level).
uint32_t sim_batchnorm_forward(Sim& s, const Node& n, uint32_t x, std::pair<uint32_t, uint32_t>& keep) {
  const uint32_t aff = n.trainable ? s.w : 0;
  if (s.eval) {
    const uint32_t r = s.see(s.w + n.bn->inv_sqrt_depth());
    const uint32_t h = s.see(std::max({x, s.w, r}) + 1);
    return s.see(std::max(h, aff) + 1);
  }
  const uint32_t mu = s.see(x + 1), d = std::max(x, mu);
  const uint32_t var = s.see(s.see(d + 1) + 1);
  const uint32_t r = s.see(var + n.bn->inv_sqrt_depth());
  const uint32_t xh = s.see(std::max(d, r) + 1);
  keep = {xh, r};
  s.new_w = std::max(s.new_w, s.see(std::max({s.w, mu, var}) + 1));  // running statistics
  return s.see(std::max(xh, aff) + 1);
}

uint32_t sim_forward(Sim& s, const std::vector<std::unique_ptr<Node>>& nodes, uint32_t x, SimTrace& tr) {
  tr.in.assign(nodes.size(), 0);
  tr.body.assign(nodes.size(), {});
  tr.bn.assign(nodes.size(), {0, 0});
  for (size_t i = 0; i < nodes.size(); ++i) {
    const Node& n = *nodes[i];
    tr.in[i] = x;
    switch (n.kind) {
      case Node::kLinear: x = s.see(std::max(x, n.trainable ? s.w : x) + 1); break;
      case Node::kNorm: {
        const uint32_t rl = s.see(x + n.norm->inv_sqrt_depth());
        const uint32_t cg = s.see(std::max(x, n.trainable ? s.w : x) + 1);
        x = s.see(std::max(cg, rl) + 1);
        break;
      }
      case Node::kAttn: x = attention_levels(s, n, x).out; break;
      case Node::kMax: x = sim_maxpool_forward(s, n, x); break;
      case Node::kRelu:
        s.see(x + 2);  // the comparison needs two levels left
        x = s.see(std::max(x, relu_step_level(s, n)) + 1);
        break;
      case Node::kBatchNorm: x = sim_batchnorm_forward(s, n, x, tr.bn[i]); break;
      case Node::kDropout: x = s.eval ? x : s.see(x + 1); break;
      case Node::kSquare:
      case Node::kPoly:
      case Node::kCheb: x = s.see(x + n.layer->depth()); break;
      case Node::kBoot: x = std::min(x, 0u); break;  // levels are relative to a fresh bootstrap
      case Node::kResidual: x = std::max(sim_forward(s, n.body, x, tr.body[i]), x); break;
    }
  }
  return x;
}

uint32_t derivative_depth(const Node& n) {
  if (n.kind == Node::kSquare) return 0;
  if (n.dcoef.size() == 1) return 0;
  if (n.kind == Node::kPoly) return PolyActivation(n.dcoef).depth();
  return poly_depth(static_cast<uint32_t>(n.dcoef.size() - 1)) + (n.a == -1.0 && n.b == 1.0 ? 0 : 1);
}

uint32_t sim_backward(Sim& s, const std::vector<std::unique_ptr<Node>>& nodes, const SimTrace& tr, uint32_t g) {
  for (size_t k = nodes.size(); k-- > 0;) {
    const Node& n = *nodes[k];
    const uint32_t z = tr.in[k];
    switch (n.kind) {
      case Node::kLinear: g = sim_linear_backward(s, n, g, z); break;
      case Node::kNorm: {
        const uint32_t rl = z + n.norm->inv_sqrt_depth();
        const uint32_t xh = s.see(rl + 1);  // x_hat = centred (*) r
        const uint32_t gg = s.see(std::max(g, n.trainable ? s.w : g) + 1);
        if (n.trainable) {
          uint32_t grad = s.see(std::max(g, xh) + 1);
          uint32_t upd = s.see(grad + (n.groups.empty() ? 1 : 2));
          s.new_w = std::max(s.new_w, std::max(s.w, upd));
        }
        const uint32_t m2 = s.see(std::max(gg, xh) + 2);  // block mean of gg (*) x_hat
        const uint32_t inner = s.see(m2 + 1);            // gg - m1 - x_hat (*) m2
        g = s.see(std::max(inner, rl) + 1);
        break;
      }
      case Node::kAttn: {
        const AttnLevels L = attention_levels(s, n, z);
        const uint32_t g3 = sim_linear_backward(s, *n.body[3], g, L.o);
        uint32_t ldo = g3, dp;
        if (n.att->softmax()) {
          ldo = s.see(std::max(g3, L.y) + 1);
          const uint32_t c = s.see(std::max(s.see(std::max(g3, L.osum) + 1), L.y + 1) + 1);
          const uint32_t dA = s.see(std::max(s.see(std::max(ldo, L.q) + 1), c));
          dp = s.see(std::max(dA, L.a) + 1);
        } else {
          const uint32_t dA = s.see(std::max(g3, L.q) + 1);
          dp = s.see(std::max(dA, L.p + 1) + 1);
        }
        const uint32_t dq = s.see(std::max(dp, L.q) + 1);
        const uint32_t dv = s.see(std::max(ldo, L.a) + 1);
        uint32_t gx = sim_linear_backward(s, *n.body[0], dq, z);
        gx = std::max(gx, sim_linear_backward(s, *n.body[1], dq, z));
        gx = std::max(gx, sim_linear_backward(s, *n.body[2], dv, z));
        g = gx;
        break;
      }
      case Node::kSquare:
      case Node::kPoly:
      case Node::kCheb:
        if ((n.kind == Node::kSquare) || n.dcoef.size() > 1)
          g = s.see(std::max(g, s.see(z + derivative_depth(n))) + 1);
        else
          g = s.see(g + 1);
        break;
      case Node::kRelu: g = s.see(std::max(g, relu_step_level(s, n)) + 1); break;
      case Node::kDropout: g = s.see(g + 1); break;
      case Node::kBatchNorm: {
        const auto [xh, r] = tr.bn[k];
        const uint32_t gg = s.see(std::max(g, n.trainable ? s.w : g) + 1);
        if (n.trainable) {
          uint32_t grad = s.see(std::max(g, xh) + 1);
          uint32_t upd = s.see(grad + (n.groups.empty() ? 1 : 2));
          s.new_w = std::max(s.new_w, std::max(s.w, upd));
        }
        const uint32_t m1 = s.see(gg + 1);
        const uint32_t m2 = s.see(s.see(std::max(gg, xh) + 1) + 1);
        const uint32_t inner = s.see(std::max({gg, m1, s.see(std::max(xh, m2) + 1)}));
        g = s.see(std::max(inner, r) + 1);
        break;
      }
      case Node::kMax:
        for (uint32_t r = 0; r < n.pool->rounds(); ++r) g = s.see(std::max(g, FHEContext::compare_depth() + 1) + 1);
        g = s.see(g + 1);  // scatter
        break;
      case Node::kBoot: g = std::min(g, 0u); break;
      case Node::kResidual: g = std::max(sim_backward(s, n.body, tr.body[k], g), g); break;
    }
  }
  return g;
}

// Level of dL/dy from the output level y (see Trainer::loss_gradient).
uint32_t sim_loss(Sim& s, const CrossEntropy* ce, uint32_t y) {
  if (!ce) return y;
  const SoftmaxOptions& o = ce->o;
  uint32_t z;
  if (o.subtract_max) {
    const uint32_t m0 = s.see(y + 1);
    uint32_t m = m0;
    for (int t = 1; t < ce->window; t *= 2) m = s.see(std::max(FHEContext::compare_depth(), m) + 1);
    m = s.see(m + 1);  // keep slot 0
    z = std::max(m0, m);
  } else {
    z = s.see(y + (ce->mask_input ? 1 : 0));
  }
  const uint32_t e = s.see(z + poly_depth(o.exp_degree) + 1 + 1);
  const uint32_t r = s.see(e + poly_depth(o.inv_degree) + 1 + 2 * o.newton);
  return s.see(std::max(e, r) + 1);
}

LinearMap& channel_mean(Node& n, size_t count, uint32_t slots);

void collect_rotations(const std::vector<std::unique_ptr<Node>>& nodes, uint32_t slots, std::set<int32_t>& r,
                       bool& allsum) {
  auto add = [&](int64_t k) {
    int32_t m = normalize_rotation(k, slots);
    if (m) r.insert(m);
  };
  for (const auto& n : nodes) {
    if (n->kind == Node::kResidual || n->kind == Node::kAttn) collect_rotations(n->body, slots, r, allsum);
    if (n->kind == Node::kNorm)
      for (int32_t k : n->norm->rotation_indices()) add(k);
    if (n->kind == Node::kBatchNorm)
      for (int32_t k : channel_mean(const_cast<Node&>(*n), 1, slots).rotation_indices()) add(k);
    if (n->kind == Node::kMax) {
      for (int32_t k : n->pool->gather().rotation_indices()) add(k);
      for (int32_t k : n->pool->scatter().rotation_indices()) add(k);
      for (uint32_t h = n->pool->blocks() / 2; h >= 1; h /= 2) {
        add(static_cast<int64_t>(h) * n->pool->block());
        add(-static_cast<int64_t>(h) * n->pool->block());
      }
    }
    if (n->kind == Node::kAttn) {
      const Attention& at = *n->att;
      const int64_t nd = static_cast<int64_t>(at.tokens()) * at.stride();
      add(nd);
      add(-nd);
      for (int s = 1; s < at.tokens(); ++s) {
        add(static_cast<int64_t>(s) * at.stride());
        add(-static_cast<int64_t>(s) * at.stride());
      }
      for (int t = 1; t < at.head_width(); t *= 2) {
        add(t);
        add(-t);
      }
    }
    if (n->kind != Node::kLinear && n->kind != Node::kNorm) continue;
    for (int32_t d : n->offs) {
      if (d) r.insert(d);
      int32_t m = normalize_rotation(-static_cast<int64_t>(d), slots);
      if (m) r.insert(m);
    }
    if (!n->groups.empty()) allsum = true;
    if (n->trainable)
      for (uint32_t k = n->window; k < slots; k *= 2) {
        r.insert(static_cast<int32_t>(k));
        r.insert(normalize_rotation(-static_cast<int64_t>(k), slots));
      }
  }
}

// Every node with weight ciphertexts (linear layers, layer norms, the
// attention projections), depth first.
template <typename F>
void for_each_weighted(std::vector<std::unique_ptr<Node>>& nodes, F f) {
  for (auto& n : nodes) {
    if (n->kind == Node::kResidual || n->kind == Node::kAttn) for_each_weighted(n->body, f);
    if (n->kind == Node::kLinear || n->kind == Node::kNorm || n->kind == Node::kBatchNorm) f(*n);
  }
}

template <typename F>
void for_each_weighted(const std::vector<std::unique_ptr<Node>>& nodes, F f) {
  for (const auto& n : nodes) {
    if (n->kind == Node::kResidual || n->kind == Node::kAttn) for_each_weighted(n->body, f);
    if (n->kind == Node::kLinear || n->kind == Node::kNorm || n->kind == Node::kBatchNorm) f(*n);
  }
}

template <typename F>
void for_each_attention(std::vector<std::unique_ptr<Node>>& nodes, F f) {
  for (auto& n : nodes) {
    if (n->kind == Node::kResidual) for_each_attention(n->body, f);
    if (n->kind == Node::kAttn) f(*n);
  }
}

// --------------------------------------------------------- encrypted passes

Ctxt linear_forward(const FHEContext& ctx, Refresh& rf, Node& n, Ctxt x) {
  const CC& cc = ctx.cc();
  rf.fit(x, 1);
  n.in = x;
  const std::vector<Ctxt> rot = ctx.rotate_many(x, n.offs);
  std::vector<Ctxt> terms(n.offs.size());
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
  for (size_t i = 0; i < n.offs.size(); ++i)
    terms[i] = n.trainable ? cc->EvalMult(n.D[i], rot[i]) : mul_plain(ctx, rot[i], n.diag[i]);
  Ctxt y = sum(ctx, std::move(terms));
  if (n.trainable && n.B) {
    cc->EvalAddInPlace(y, n.B);
  } else if (!n.bias.empty()) {
    y = add_plain(ctx, y, n.bias);
  }
  return y;
}

// dL/dD_i = g (*) rot_i(x) summed over the batch; returns dL/dx when asked.
Ctxt linear_backward(const FHEContext& ctx, Refresh& rf, Node& n, Ctxt g, bool need) {
  const CC& cc = ctx.cc();
  rf.fit(g, 1);
  if (n.trainable) {
    rf.fit(n.in, 1);
    const std::vector<Ctxt> rot = ctx.rotate_many(n.in, n.offs);
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
    for (size_t i = 0; i < n.offs.size(); ++i) add_to(ctx, n.gD[i], cc->EvalMult(g, rot[i]));
    if (n.B) add_to(ctx, n.gB, g);
  }
  n.in = nullptr;
  if (!need) return nullptr;
  std::vector<Ctxt> terms(n.offs.size());
  if (!n.trainable) {
    // rot_{-o}(g * diag) = rot_{-o}(g) * rot_{-o}(diag): the rotations all
    // act on g, so they share one hoisted decomposition (or ladder tree),
    // and the constant diagonals are rotated in the clear.
    std::vector<int32_t> neg(n.offs.size());
    for (size_t i = 0; i < n.offs.size(); ++i) neg[i] = -n.offs[i];
    const std::vector<Ctxt> rot = ctx.rotate_many(g, neg);
    const size_t slots = ctx.slots();
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
    for (size_t i = 0; i < n.offs.size(); ++i) {
      std::vector<double> d(slots);
      const size_t o = static_cast<size_t>(normalize_rotation(n.offs[i], ctx.slots()));
      for (size_t j = 0; j < slots; ++j) d[j] = n.diag[i][(j + slots - o) % slots];
      terms[i] = mul_plain(ctx, rot[i], d);
    }
    return sum(ctx, std::move(terms));
  }
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
  for (size_t i = 0; i < n.offs.size(); ++i) terms[i] = ctx.rotate(cc->EvalMult(n.D[i], g), -n.offs[i]);
  return sum(ctx, std::move(terms));
}

// Sum of the slots c .. c+width-1 into slot c (the token or head window),
// and its adjoint: slot c spread over c .. c+width-1. Neither touches the
// caller's ciphertext.
Ctxt window_sum(const FHEContext& ctx, const Ctxt& c, int width) {
  if (width < 2) return c;
  Ctxt s = ctx.cc()->EvalAdd(c, ctx.rotate(c, 1));
  for (int t = 2; t < width; t *= 2) ctx.cc()->EvalAddInPlace(s, ctx.rotate(s, t));
  return s;
}

Ctxt window_broadcast(const FHEContext& ctx, const Ctxt& c, int width) {
  if (width < 2) return c;
  Ctxt s = ctx.cc()->EvalAdd(c, ctx.rotate(c, -1));
  for (int t = 2; t < width; t *= 2) ctx.cc()->EvalAddInPlace(s, ctx.rotate(s, -t));
  return s;
}

// Attention::forward with the projections as trainable nodes and every
// rotation through the context (so a rotation-key ladder works); keeps what
// the backward pass needs. With a level budget the scores, exponentials, the
// row sum and its inverse are refreshed between the steps as needed.
Ctxt attention_forward(const FHEContext& ctx, Refresh& rf, Node& n, Ctxt x) {
  const CC& cc = ctx.cc();
  const Attention& at = *n.att;
  const int tokens = static_cast<int>(n.aoffs.size()), dh = at.head_width(), B = at.stride();
  const int32_t nd = at.tokens() * B;
  rf.fit(x, 1);  // once for the three projections
  n.q = linear_forward(ctx, rf, *n.body[0], x);
  Ctxt k = linear_forward(ctx, rf, *n.body[1], x);
  Ctxt v = linear_forward(ctx, rf, *n.body[2], x);
  rf.fit(n.q, 1);
  rf.fit(k, 1);
  rf.fit(v, 1);
  // Cyclic copies in [n*B, 2n*B) so rotating by s*B maps token i to (i + s) mod n.
  k = cc->EvalAdd(k, ctx.rotate(k, -nd));
  v = cc->EvalAdd(v, ctx.rotate(v, -nd));
  std::vector<int32_t> offs(static_cast<size_t>(tokens));
  for (int s = 0; s < tokens; ++s) offs[static_cast<size_t>(s)] = n.aoffs[static_cast<size_t>(s)] * B;
  n.kr = ctx.rotate_many(k, offs);
  n.vr = ctx.rotate_many(v, offs);
  // score[i][h] = q_i . k_{i+s} over head h, in slot i*B + h*dh.
  n.p.assign(static_cast<size_t>(tokens), nullptr);
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
  for (int s = 0; s < tokens; ++s) n.p[static_cast<size_t>(s)] = window_sum(ctx, cc->EvalMult(n.q, n.kr[static_cast<size_t>(s)]), dh);
  const bool softmax = at.softmax();
  if (softmax && at.softmax_options().subtract_max) {
    const SoftmaxOptions& o = at.softmax_options();
    const double range = 2.0 * std::max(std::fabs(o.lo), std::fabs(o.hi));
    std::vector<Ctxt> vals = n.p;
    while (vals.size() > 1) {
      std::vector<Ctxt> next((vals.size() + 1) / 2);
      for (size_t i = 0; i + 1 < vals.size(); i += 2) {
        // compare() needs two levels left, the selection product one.
        rf.fit(vals[i], 1);
        rf.fit(vals[i + 1], 1);
        Ctxt sel = ctx.compare(vals[i], vals[i + 1], range);
        next[i / 2] = cc->EvalAdd(vals[i], cc->EvalMult(sel, cc->EvalSub(vals[i + 1], vals[i])));
      }
      if (vals.size() % 2) next.back() = vals.back();
      vals.swap(next);
    }
    for (auto& p : n.p) p = cc->EvalSub(p, vals[0]);
  }
  n.aw.assign(static_cast<size_t>(tokens), nullptr);
  n.A.assign(static_cast<size_t>(tokens), nullptr);
  if (softmax) {
    const SoftmaxOptions& o = at.softmax_options();
    rf.fit_all(n.p, chebyshev_depth(o.exp_degree, at.exp_lo(), at.exp_hi()));
    std::vector<Ctxt> e(static_cast<size_t>(tokens));
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
    for (int s = 0; s < tokens; ++s) {
      std::vector<double> c = at.exp_coeffs();
      e[static_cast<size_t>(s)] = cc->EvalChebyshevSeries(n.p[static_cast<size_t>(s)], c, at.exp_lo(), at.exp_hi());
    }
    rf.fit_all(e, 1);
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
    for (int s = 0; s < tokens; ++s)
      n.aw[static_cast<size_t>(s)] = mul_plain(ctx, e[static_cast<size_t>(s)], n.ahead[static_cast<size_t>(s)]);
  } else {
    rf.fit_all(n.p, 2);
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
    for (int s = 0; s < tokens; ++s) {
      const size_t i = static_cast<size_t>(s);
      Ctxt a = add_plain(ctx, mul_plain(ctx, n.p[i], n.ac2[i]), n.ac1[i]);
      n.aw[i] = add_plain(ctx, cc->EvalMult(n.p[i], a), n.ac0[i]);
    }
  }
  rf.fit_all(n.aw, 1);
  rf.fit_all(n.vr, 1);
  std::vector<Ctxt> partial(static_cast<size_t>(tokens));
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
  for (int s = 0; s < tokens; ++s) {
    const size_t i = static_cast<size_t>(s);
    n.A[i] = window_broadcast(ctx, n.aw[i], dh);
    partial[i] = cc->EvalMult(n.A[i], n.vr[i]);
  }
  Ctxt o = sum(ctx, std::move(partial));
  if (softmax) {
    const SoftmaxOptions& sm = at.softmax_options();
    Ctxt S = tokens > 1 ? cc->EvalAdd(n.A[0], n.A[1]) : n.A[0];
    for (int s = 2; s < tokens; ++s) cc->EvalAddInPlace(S, n.A[static_cast<size_t>(s)]);
    S = add_plain(ctx, S, at.pad_sum());
    rf.fit(S, chebyshev_depth(sm.inv_degree, at.sum_lo(), at.sum_hi()));
    std::vector<double> c = at.inv_coeffs();
    Ctxt y = cc->EvalChebyshevSeries(S, c, at.sum_lo(), at.sum_hi());
    for (uint32_t i = 0; i < sm.newton; ++i) {
      rf.fit(S, 2);
      rf.fit(y, 2);
      y = cc->EvalMult(y, cc->EvalSub(2.0, cc->EvalMult(S, y)));
    }
    rf.fit(y, 1);
    rf.fit(o, 1);
    n.y = y;
    n.o = o;
    o = cc->EvalMult(o, y);
  }
  return linear_forward(ctx, rf, *n.body[3], o);
}

// The gradient of attention_forward. The softmax and 1/x are differentiated
// as the functions they approximate (exp' = exp, (1/S)' = -1/S^2), so the
// gradient matches the exact softmax up to the approximation error.
Ctxt attention_backward(const FHEContext& ctx, Refresh& rf, Node& n, const Ctxt& g, bool need) {
  const CC& cc = ctx.cc();
  const Attention& at = *n.att;
  const int tokens = static_cast<int>(n.aoffs.size()), dh = at.head_width(), B = at.stride();
  const int32_t nd = at.tokens() * B;
  const bool softmax = at.softmax();
  Ctxt go = linear_backward(ctx, rf, *n.body[3], g, true);
  Ctxt dO = go, c;
  if (softmax) {
    // out = o (*) y, y = 1 / S, S = sum_s A_s: dL/dA_s = dO (*) v_s - go (*) o (*) y^2.
    rf.fit(go, 2);
    rf.fit(n.o, 2);
    rf.fit(n.y, 2);
    dO = cc->EvalMult(go, n.y);
    c = cc->EvalMult(cc->EvalMult(go, n.o), cc->EvalMult(n.y, n.y));
  }
  rf.fit(dO, 1);
  rf.fit_all(n.vr, 1);
  const size_t T = static_cast<size_t>(tokens);
  std::vector<Ctxt> da(T), P(T), dqs(T), dks(T), dvs(T);
  // A_s broadcasts a_s over the head: da_s = window sum of dA_s.
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
  for (int s = 0; s < tokens; ++s) {
    const size_t i = static_cast<size_t>(s);
    Ctxt dA = cc->EvalMult(dO, n.vr[i]);
    if (c) dA = cc->EvalSub(dA, c);
    da[i] = window_sum(ctx, dA, dh);
  }
  // Then through the (masked) score function to the q.k sums.
  rf.fit_all(da, 1);
  if (softmax)
    rf.fit_all(n.aw, 1);
  else
    rf.fit_all(n.p, 2);
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
  for (int s = 0; s < tokens; ++s) {
    const size_t i = static_cast<size_t>(s);
    Ctxt dp;
    if (softmax) {
      dp = cc->EvalMult(da[i], n.aw[i]);
    } else {
      Ctxt slope = add_plain(ctx, mul_plain(ctx, n.p[i], scaled(n.ac2[i], 2.0)), n.ac1[i]);
      dp = cc->EvalMult(da[i], slope);
    }
    P[i] = window_broadcast(ctx, dp, dh);  // the q.k sums spread over the head
  }
  rf.fit_all(P, 1);
  rf.fit(n.q, 1);
  rf.fit_all(n.kr, 1);
  rf.fit_all(n.A, 1);
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
  for (int s = 0; s < tokens; ++s) {
    const size_t i = static_cast<size_t>(s);
    dqs[i] = cc->EvalMult(P[i], n.kr[i]);
    const int32_t back = -n.aoffs[i] * B;
    dks[i] = ctx.rotate(cc->EvalMult(P[i], n.q), back);
    dvs[i] = ctx.rotate(cc->EvalMult(dO, n.A[i]), back);
  }
  Ctxt dq = sum(ctx, std::move(dqs));
  // Fold the cyclic copies [n*B, 2n*B) back onto the tokens.
  Ctxt dk = sum(ctx, std::move(dks));
  dk = cc->EvalAdd(dk, ctx.rotate(dk, nd));
  Ctxt dv = sum(ctx, std::move(dvs));
  dv = cc->EvalAdd(dv, ctx.rotate(dv, nd));
  Ctxt gx = linear_backward(ctx, rf, *n.body[0], dq, need);
  Ctxt gk = linear_backward(ctx, rf, *n.body[1], dk, need);
  Ctxt gv = linear_backward(ctx, rf, *n.body[2], dv, need);
  n.kr.clear();
  n.vr.clear();
  n.p.clear();
  n.aw.clear();
  n.A.clear();
  n.q = n.o = n.y = nullptr;
  if (!need) return nullptr;
  cc->EvalAddInPlace(gx, gk);
  cc->EvalAddInPlace(gx, gv);
  return gx;
}

// MaxPool2d::forward, keeping the comparison bits: max(a, b) = a + [a < b] (b - a).
Ctxt maxpool_forward(const FHEContext& ctx, Refresh& rf, Node& n, Ctxt x) {
  if (!ctx.scheme_switching())
    throw std::logic_error("maxpool2d needs a context with ContextOptions::scheme_switching");
  const CC& cc = ctx.cc();
  const MaxPool2d& mp = *n.pool;
  rf.fit(x, 1);
  Ctxt a = mp.gather().eval(ctx, x);
  n.pick_b.clear();
  n.pick_a.clear();
  size_t r = 0;
  for (uint32_t h = mp.blocks() / 2; h >= 1; h /= 2, ++r) {
    const size_t m = static_cast<size_t>(h) * mp.block();
    rf.fit(a, 1);  // compare() needs two levels left, the selection product one
    Ctxt b = ctx.rotate(a, static_cast<int32_t>(m));
    Ctxt sel = ctx.compare(a, b, mp.range(), m);  // [a < b]
    a = cc->EvalAdd(a, cc->EvalMult(sel, cc->EvalSub(b, a)));
    // Masked to the slots this round writes: a gradient in any other slot
    // (an output layer's g = y - t there) must not leak into the inputs.
    n.pick_b.push_back(mul_plain(ctx, sel, n.round_mask[r]));
    n.pick_a.push_back(add_plain(ctx, cc->EvalNegate(n.pick_b.back()), n.round_mask[r]));
  }
  return a;
}

// Routes g down the tournament (d max / da = 1 - [a < b], d max / db = [a < b],
// b being a rotated by the round's half) to the tap blocks, then scatters the
// taps' gradients back to their input slots (overlapping windows add up).
Ctxt maxpool_backward(const FHEContext& ctx, Refresh& rf, Node& n, Ctxt g, bool need) {
  const CC& cc = ctx.cc();
  const MaxPool2d& mp = *n.pool;
  if (need) {
    uint32_t h = 1;
    for (size_t r = n.pick_b.size(); r-- > 0; h *= 2) {
      rf.fit(g, 1);
      rf.fit(n.pick_a[r], 1);
      rf.fit(n.pick_b[r], 1);
      Ctxt gb = ctx.rotate(cc->EvalMult(g, n.pick_b[r]), -static_cast<int32_t>(static_cast<size_t>(h) * mp.block()));
      g = cc->EvalAdd(cc->EvalMult(g, n.pick_a[r]), gb);
    }
    rf.fit(g, 1);
    g = mp.scatter().eval(ctx, g);
  }
  n.pick_b.clear();
  n.pick_a.clear();
  return need ? g : nullptr;
}

Ctxt forward_nodes(const FHEContext& ctx, Refresh& rf, std::vector<std::unique_ptr<Node>>& nodes, Ctxt x);

// The channel-mean map of a batch of `count` samples: averages each channel
// over its positions, applied to the sum of the batch's ciphertexts (one
// level, zero outside the layout).
LinearMap& channel_mean(Node& n, size_t count, uint32_t slots) {
  auto& m = n.channel_mean[count];
  if (m) return *m;
  const BatchNorm& bn = *n.bn;
  const Shape& s = bn.in_shape();
  const size_t hw = bn.positions();
  const double w = 1.0 / static_cast<double>(hw * count);
  std::vector<size_t> rows, cols;
  std::vector<double> vals;
  for (size_t c = 0; c < bn.channels(); ++c)
    for (size_t i = 0; i < hw; ++i)
      for (size_t j = 0; j < hw; ++j) {
        rows.push_back(c * hw + i);
        cols.push_back(c * hw + j);
        vals.push_back(w);
      }
  m = std::make_shared<LinearMap>(s, s, rows, cols, vals, std::vector<double>{}, "channelmean");
  m->configure(s);
  m->configure_layout(bn.in_layout(), Packing::kCompact);
  m->set_out_layout(bn.in_layout());
  m->set_allow_fold(false);
  m->compile(slots);
  return *m;
}

// 1 / sqrt(v + eps) as BatchNorm::inv_sqrt computes it (series scaled so the
// Newton steps need no scalar product on r, as in LayerNorm::normalize).
Ctxt bn_inv_sqrt(const FHEContext& ctx, Refresh& rf, const Node& n, Ctxt v) {
  const CC& cc = ctx.cc();
  const BatchNorm& bn = *n.bn;
  v = add_plain(ctx, v, n.var_pad);
  rf.fit(v, bn.inv_sqrt_depth());
  double k = std::pow(1.5, static_cast<double>(bn.newton()));
  std::vector<double> c = bn.inv_sqrt_coeffs();
  for (double& ci : c) ci *= k;
  Ctxt r = cc->EvalChebyshevSeries(v, c, bn.var_min(), bn.var_max());
  if (bn.newton()) {
    Ctxt ve = cc->EvalAdd(v, bn.eps());
    for (uint32_t i = 0; i < bn.newton(); ++i) {
      Ctxt vc = cc->EvalMult(ve, 0.5 / (1.5 * k * k));
      Ctxt p = cc->EvalMult(cc->EvalMult(vc, r), cc->EvalMult(r, r));
      r = cc->EvalSub(r, p);
      k /= 1.5;
    }
  }
  return r;
}

// y = gamma (*) h + beta.
Ctxt bn_affine(const FHEContext& ctx, Refresh& rf, Node& n, Ctxt h) {
  rf.fit(h, 1);
  Ctxt y = n.trainable ? ctx.cc()->EvalMult(h, n.D[0]) : mul_plain(ctx, h, n.diag[0]);
  if (n.trainable)
    ctx.cc()->EvalAddInPlace(y, n.B);
  else
    y = add_plain(ctx, y, n.bias);
  return y;
}

// One sample through one node (not a residual block's body; attention with
// its projections). A batch norm uses its running statistics here
// (inference); forward_batch normalises with the batch's.
Ctxt forward_node(const FHEContext& ctx, Refresh& rf, Node& n, Ctxt x) {
  switch (n.kind) {
    case Node::kLinear: return linear_forward(ctx, rf, n, x);
    case Node::kNorm: {
      rf.fit(x, n.norm->inv_sqrt_depth());
      n.norm->normalize(ctx, x, n.centred, n.r);
      rf.fit(n.centred, 1);
      Ctxt cg = n.trainable ? ctx.cc()->EvalMult(n.centred, n.D[0]) : mul_plain(ctx, n.centred, n.diag[0]);
      rf.fit(cg, 1);
      rf.fit(n.r, 1);
      x = ctx.cc()->EvalMult(cg, n.r);
      if (n.trainable)
        ctx.cc()->EvalAddInPlace(x, n.B);
      else
        x = add_plain(ctx, x, n.bias);
      return x;
    }
    case Node::kAttn: return attention_forward(ctx, rf, n, x);
    case Node::kMax: return maxpool_forward(ctx, rf, n, x);
    case Node::kRelu: {
      if (!ctx.scheme_switching())
        throw std::logic_error("exactrelu needs a context with ContextOptions::scheme_switching");
      rf.fit(x, 1);  // compare() needs two levels left, the product one
      const double sc = n.relu->out_scale();
      Ctxt sel = ctx.positive(x, n.relu->range());
      n.step = sc == 1.0 ? sel : ctx.cc()->EvalMult(sel, sc);
      return ctx.cc()->EvalMult(x, n.step);
    }
    case Node::kBatchNorm: {
      Ctxt r = bn_inv_sqrt(ctx, rf, n, n.rv);
      Ctxt c = ctx.cc()->EvalSub(x, n.rm);
      rf.fit(c, 1);
      rf.fit(r, 1);
      return bn_affine(ctx, rf, n, ctx.cc()->EvalMult(c, r));
    }
    case Node::kResidual: {
      n.in = x;
      Ctxt y = forward_nodes(ctx, rf, n.body, x);
      return ctx.cc()->EvalAdd(y, x);
    }
    default:
      rf.fit(x, n.layer->depth());
      n.in = x;
      return n.layer->forward(ctx, x);
  }
}

Ctxt forward_nodes(const FHEContext& ctx, Refresh& rf, std::vector<std::unique_ptr<Node>>& nodes, Ctxt x) {
  for (auto& np : nodes) x = forward_node(ctx, rf, *np, x);
  return x;
}

// dL/dx of one sample through one node (not a residual block or a batch
// norm, see backward_batch); null when need is false.
Ctxt backward_node(const FHEContext& ctx, Refresh& rf, Node& n, Ctxt g, bool need) {
  const CC& cc = ctx.cc();
  switch (n.kind) {
    case Node::kLinear: return linear_backward(ctx, rf, n, g, need);
    case Node::kNorm: {
      // y = gamma (*) x_hat + beta, x_hat = centred (*) r:
      // dgamma = g (*) x_hat, dbeta = g (summed over the tokens at update),
      // dx = r (*) (gg - mean(gg) - x_hat (*) mean(gg (*) x_hat)), gg = g (*) gamma.
      rf.fit(n.centred, 1);
      rf.fit(n.r, 1);
      Ctxt xhat = cc->EvalMult(n.centred, n.r);
      rf.fit(g, 1);
      rf.fit(xhat, 1);
      if (n.trainable) {
        add_to(ctx, n.gD[0], cc->EvalMult(g, xhat));
        add_to(ctx, n.gB, g);
      }
      if (need) {
        Ctxt gg = n.trainable ? cc->EvalMult(g, n.D[0]) : mul_plain(ctx, g, n.diag[0]);
        rf.fit(gg, 1);
        Ctxt m1 = n.norm->block_mean(ctx, gg);
        Ctxt t = cc->EvalMult(gg, xhat);
        rf.fit(t, 1);
        Ctxt m2 = n.norm->block_mean(ctx, t);
        rf.fit(m2, 1);
        Ctxt inner = cc->EvalSub(cc->EvalSub(gg, m1), cc->EvalMult(xhat, m2));
        rf.fit(inner, 1);
        rf.fit(n.r, 1);
        g = cc->EvalMult(inner, n.r);
      }
      n.centred = n.r = nullptr;
      return need ? g : nullptr;
    }
    case Node::kAttn: return attention_backward(ctx, rf, n, g, need);
    case Node::kMax: return maxpool_backward(ctx, rf, n, g, need);
    case Node::kRelu:
      if (need) {
        rf.fit(g, 1);
        rf.fit(n.step, 1);
        g = cc->EvalMult(g, n.step);
      }
      n.step = nullptr;
      return need ? g : nullptr;
    case Node::kSquare:
    case Node::kPoly:
    case Node::kCheb:
      if (need) {
        rf.fit(g, 1);
        if (n.kind == Node::kSquare) {
          rf.fit(n.in, 1);
          g = cc->EvalMult(g, cc->EvalAdd(n.in, n.in));
        } else if (n.dcoef.size() == 1) {
          g = cc->EvalMult(g, n.kind == Node::kCheb ? n.dcoef[0] / 2.0 : n.dcoef[0]);
        } else {
          rf.fit(n.in, derivative_depth(n));
          Ctxt f = n.kind == Node::kPoly ? PolyActivation(n.dcoef).forward(ctx, n.in)
                                         : cc->EvalChebyshevSeries(n.in, n.dcoef, n.a, n.b);
          rf.fit(f, 1);
          g = cc->EvalMult(g, f);
        }
      }
      n.in = nullptr;
      return need ? g : nullptr;
    case Node::kBoot:
      return need ? ctx.bootstrap(g) : nullptr;
    default:
      throw std::logic_error("backward_node: " + n.layer->type() + " goes through backward_batch");
  }
}

// Swaps the node's per-sample forward state with s (attention: and its
// projections'). Residual bodies are walked by forward_batch itself.
void swap_state(Node& n, Node::Sample& s) {
  std::swap(n.in, s.in);
  std::swap(n.centred, s.centred);
  std::swap(n.r, s.r);
  std::swap(n.q, s.q);
  std::swap(n.o, s.o);
  std::swap(n.y, s.y);
  std::swap(n.step, s.step);
  n.dmask.swap(s.dmask);
  n.kr.swap(s.kr);
  n.vr.swap(s.vr);
  n.p.swap(s.p);
  n.aw.swap(s.aw);
  n.A.swap(s.A);
  n.pick_b.swap(s.pick_b);
  n.pick_a.swap(s.pick_a);
  if (n.kind == Node::kAttn) {
    s.body.resize(n.body.size());
    for (size_t i = 0; i < n.body.size(); ++i) swap_state(*n.body[i], s.body[i]);
  }
}

Ctxt sum_of(const FHEContext& ctx, const std::vector<Ctxt>& v) {
  Ctxt t = v[0];
  for (size_t i = 1; i < v.size(); ++i) t = ctx.cc()->EvalAdd(t, v[i]);
  return t;
}

// Batch norm in training mode: the batch's mean and biased variance per
// channel, x_hat = (x - mean) / sqrt(var + eps) kept per sample, and the
// running statistics moved toward the batch's (unbiased variance, as torch).
void batchnorm_forward(const FHEContext& ctx, Refresh& rf, Node& n, std::vector<Ctxt>& xs) {
  const CC& cc = ctx.cc();
  const BatchNorm& bn = *n.bn;
  const size_t B = xs.size();
  LinearMap& mean_map = channel_mean(n, B, ctx.slots());
  Ctxt S = sum_of(ctx, xs);
  rf.fit(S, 1);
  Ctxt mu = mean_map.forward(ctx, S);
  std::vector<Ctxt> d(B), sq(B);
  for (size_t b = 0; b < B; ++b) {
    d[b] = cc->EvalSub(xs[b], mu);
    rf.fit(d[b], 1);
    sq[b] = cc->EvalMult(d[b], d[b]);
  }
  Ctxt Q = sum_of(ctx, sq);
  rf.fit(Q, 1);
  Ctxt var = mean_map.forward(ctx, Q);
  n.r = bn_inv_sqrt(ctx, rf, n, var);
  n.saved.assign(B, {});
  for (size_t b = 0; b < B; ++b) {
    rf.fit(d[b], 1);
    rf.fit(n.r, 1);
    n.saved[b].centred = cc->EvalMult(d[b], n.r);  // x_hat
    xs[b] = bn_affine(ctx, rf, n, n.saved[b].centred);
  }
  const double m = bn.momentum(), count = static_cast<double>(B * bn.positions());
  const double unbias = count > 1.0 ? count / (count - 1.0) : 1.0;
  rf.fit(n.rm, 1);
  rf.fit(n.rv, 1);
  rf.fit(mu, 1);
  rf.fit(var, 1);
  n.rm = cc->EvalAdd(cc->EvalMult(n.rm, 1.0 - m), cc->EvalMult(mu, m));
  n.rv = cc->EvalAdd(cc->EvalMult(n.rv, 1.0 - m), cc->EvalMult(var, m * unbias));
}

// dgamma = sum g (*) x_hat, dbeta = sum g (over the batch; over the channel
// at update); dx = r (*) (gg - mean(gg) - x_hat (*) mean(gg (*) x_hat)),
// gg = g (*) gamma, the means over the batch and the channel's positions.
void batchnorm_backward(const FHEContext& ctx, Refresh& rf, Node& n, std::vector<Ctxt>& gs, bool need) {
  const CC& cc = ctx.cc();
  const size_t B = gs.size();
  for (size_t b = 0; b < B; ++b) {
    rf.fit(gs[b], 1);
    rf.fit(n.saved[b].centred, 1);
    if (n.trainable) {
      add_to(ctx, n.gD[0], cc->EvalMult(gs[b], n.saved[b].centred));
      add_to(ctx, n.gB, gs[b]);
    }
  }
  if (need) {
    LinearMap& mean_map = channel_mean(n, B, ctx.slots());
    std::vector<Ctxt> gg(B), t(B);
    for (size_t b = 0; b < B; ++b) {
      gg[b] = n.trainable ? cc->EvalMult(gs[b], n.D[0]) : mul_plain(ctx, gs[b], n.diag[0]);
      rf.fit(gg[b], 1);
      t[b] = cc->EvalMult(gg[b], n.saved[b].centred);
    }
    Ctxt G = sum_of(ctx, gg), T = sum_of(ctx, t);
    rf.fit(G, 1);
    rf.fit(T, 1);
    Ctxt m1 = mean_map.forward(ctx, G), m2 = mean_map.forward(ctx, T);
    rf.fit(m2, 1);
    for (size_t b = 0; b < B; ++b) {
      rf.fit(n.saved[b].centred, 1);
      Ctxt inner = cc->EvalSub(cc->EvalSub(gg[b], m1), cc->EvalMult(n.saved[b].centred, m2));
      rf.fit(inner, 1);
      rf.fit(n.r, 1);
      gs[b] = cc->EvalMult(inner, n.r);
    }
  }
  n.saved.clear();
  n.r = nullptr;
}

// A batch through the nodes, layer by layer: every node runs all samples
// before the next one (so a batch norm sees the whole batch), keeping each
// sample's state for the backward pass.
void forward_batch(const FHEContext& ctx, Refresh& rf, std::vector<std::unique_ptr<Node>>& nodes,
                   std::vector<Ctxt>& xs) {
  for (auto& np : nodes) {
    Node& n = *np;
    if (n.kind == Node::kResidual) {
      std::vector<Ctxt> skip = xs;
      forward_batch(ctx, rf, n.body, xs);
      for (size_t b = 0; b < xs.size(); ++b) xs[b] = ctx.cc()->EvalAdd(xs[b], skip[b]);
    } else if (n.kind == Node::kBatchNorm) {
      batchnorm_forward(ctx, rf, n, xs);
    } else if (n.kind == Node::kDropout) {
      // A fresh mask per sample, scaled by 1 / (1 - p), placed in the slots.
      const Shape& sh = n.drop->in_shape();
      const Layout& lay = n.drop->in_layout();
      n.saved.assign(xs.size(), {});
      for (size_t b = 0; b < xs.size(); ++b) {
        const std::vector<double> m = n.drop->draw();
        std::vector<double>& slots = n.saved[b].dmask;
        slots.assign(ctx.slots(), 0.0);
        for (size_t i = 0; i < m.size(); ++i) slots[lay.slot(sh, i)] = m[i];
        rf.fit(xs[b], 1);
        xs[b] = mul_plain(ctx, xs[b], slots);
      }
    } else {
      n.saved.assign(xs.size(), {});
      for (size_t b = 0; b < xs.size(); ++b) {
        swap_state(n, n.saved[b]);
        xs[b] = forward_node(ctx, rf, n, xs[b]);
        swap_state(n, n.saved[b]);
      }
    }
  }
}

// The batch's gradients back through the nodes (dL/dx of every sample when
// need_input_grad).
void backward_batch(const FHEContext& ctx, Refresh& rf, std::vector<std::unique_ptr<Node>>& nodes,
                    std::vector<Ctxt>& gs, bool need_input_grad) {
  for (size_t k = nodes.size(); k-- > 0;) {
    Node& n = *nodes[k];
    const bool need = k > 0 || need_input_grad;
    if (n.kind == Node::kResidual) {
      std::vector<Ctxt> gb = gs;
      backward_batch(ctx, rf, n.body, gb, need);
      if (need)
        for (size_t b = 0; b < gs.size(); ++b) gs[b] = ctx.cc()->EvalAdd(gb[b], gs[b]);
    } else if (n.kind == Node::kBatchNorm) {
      batchnorm_backward(ctx, rf, n, gs, need);
    } else if (n.kind == Node::kDropout) {
      if (need)
        for (size_t b = 0; b < gs.size(); ++b) {
          rf.fit(gs[b], 1);
          gs[b] = mul_plain(ctx, gs[b], n.saved[b].dmask);
        }
      n.saved.clear();
    } else {
      for (size_t b = 0; b < gs.size(); ++b) {
        swap_state(n, n.saved[b]);
        gs[b] = backward_node(ctx, rf, n, gs[b], need);
        swap_state(n, n.saved[b]);
      }
      n.saved.clear();
    }
    if (!need) return;
  }
}

// Sum of all slots, in every slot.
Ctxt allsum(const FHEContext& ctx, Ctxt c) {
  for (uint32_t k = 1; k < ctx.slots(); k *= 2) ctx.cc()->EvalAddInPlace(c, ctx.rotate(c, static_cast<int32_t>(k)));
  return c;
}

void update_linear(const FHEContext& ctx, Refresh& rf, Node& n, double step) {
  const CC& cc = ctx.cc();
  // Masking takes one level, a shared weight's masked sum and broadcast two.
  rf.fit_all(n.gD, n.groups.empty() ? 1 : 2);
  rf.fit(n.gB, n.groups.empty() ? 1 : 2);
  std::vector<Ctxt> delta(n.offs.size());
  Ctxt delta_b;
  // Unshared weights: their own slot's gradient.
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
  for (size_t i = 0; i < n.offs.size(); ++i)
    if (!n.unshared[i].empty()) delta[i] = mul_plain(ctx, n.gD[i], scaled(n.unshared[i], step));
  if (!n.bias_unshared.empty()) delta_b = mul_plain(ctx, n.gB, scaled(n.bias_unshared, step));
  // Shared weights: masked sum over their slots, broadcast, masked again.
  std::vector<std::vector<Ctxt>> contrib(n.groups.size());
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
  for (size_t gi = 0; gi < n.groups.size(); ++gi) {
    const auto& g = n.groups[gi];
    std::vector<Ctxt> parts;
    for (const auto& dm : g.masks) parts.push_back(mul_plain(ctx, dm.first < 0 ? n.gB : n.gD[static_cast<size_t>(dm.first)], dm.second));
    Ctxt total = allsum(ctx, sum(ctx, std::move(parts)));
    for (const auto& dm : g.masks) contrib[gi].push_back(mul_plain(ctx, total, scaled(dm.second, step)));
  }
  for (size_t gi = 0; gi < n.groups.size(); ++gi)
    for (size_t j = 0; j < n.groups[gi].masks.size(); ++j) {
      int d = n.groups[gi].masks[j].first;
      add_to(ctx, d < 0 ? delta_b : delta[static_cast<size_t>(d)], contrib[gi][j]);
    }
  for (size_t i = 0; i < n.offs.size(); ++i) {
    if (delta[i]) cc->EvalAddInPlace(n.D[i], delta[i]);
    n.gD[i] = nullptr;
  }
  if (delta_b) cc->EvalAddInPlace(n.B, delta_b);
  n.gB = nullptr;
}

// Bootstraps weight ciphertexts that are zero outside [0, window), packing
// slots / window of them into one bootstrap: a rotate-and-add tree packs them
// side by side, the inverse tree and a mask unpack them (one level).
void bootstrap_packed(const FHEContext& ctx, std::vector<Ctxt*> cts, uint32_t window) {
  const CC& cc = ctx.cc();
  const uint32_t k = ctx.slots() / window;
  if (k < 2) {
    for (Ctxt* c : cts) *c = ctx.bootstrap(*c);
    return;
  }
  std::vector<double> mask(ctx.slots(), 0.0);
  std::fill(mask.begin(), mask.begin() + window, 1.0);
  for (size_t start = 0; start < cts.size(); start += k) {
    const size_t m = std::min<size_t>(k, cts.size() - start);
    size_t width = 1;
    while (width < m) width *= 2;
    std::vector<Ctxt> t(width);
    for (size_t j = 0; j < m; ++j) t[j] = *cts[start + j];
    for (size_t step = 1; step < width; step *= 2)
      for (size_t i = 0; i + step < width; i += 2 * step)
        if (t[i + step]) {
          Ctxt r = ctx.rotate(t[i + step], -static_cast<int32_t>(step * window));
          t[i] = t[i] ? cc->EvalAdd(t[i], r) : r;
        }
    t[0] = ctx.bootstrap(t[0]);
    for (size_t step = width / 2; step >= 1; step /= 2)
      for (size_t i = 0; i + step < width; i += 2 * step)
        if (i + step < m) t[i + step] = ctx.rotate(t[i], static_cast<int32_t>(step * window));
#pragma omp parallel for if (ctx.parallel())
    for (size_t j = 0; j < m; ++j) *cts[start + j] = mul_plain(ctx, t[j], mask);
  }
}

// --------------------------------------------------------- plaintext reference

struct PlainTrace {
  std::vector<std::vector<std::vector<double>>> in;  // [layer][sample]
  std::vector<PlainTrace> body;                      // residual bodies; batch norms keep x_hat and r here
  std::vector<std::vector<double>> xhat;
  std::vector<double> r;
};

using Grads = std::map<Layer*, std::vector<double>>;

// Backward through a linear layer; adds its parameter gradient when trained.
std::vector<double> plain_linear_backward(LinearLayer* lin, bool train, const std::vector<double>& x,
                                          const std::vector<double>& g, Grads& grads) {
  std::vector<double> gx(x.size(), 0.0);
  if (train && lin->trainable()) {
    const std::vector<double> p = lin->params();
    auto& gr = grads[lin];
    if (gr.empty()) gr.assign(p.size(), 0.0);
    for (const auto& e : lin->weight_refs()) {
      gx[e.col] += p[static_cast<size_t>(e.param)] * g[e.row];
      gr[static_cast<size_t>(e.param)] += g[e.row] * x[e.col];
    }
    const auto br = lin->bias_refs();
    for (size_t i = 0; i < br.size(); ++i) gr[static_cast<size_t>(br[i])] += g[i];
  } else {
    const SparseMatrix m = lin->matrix();
    for (size_t r = 0; r < m.rows(); ++r)
      for (const auto& e : m.row(r)) gx[e.first] += e.second * g[r];
  }
  return gx;
}

std::vector<double> plain_layernorm_backward(LayerNorm* ln, bool train, const std::vector<double>& x,
                                             const std::vector<double>& g, Grads& grads) {
  std::vector<double> centred, r;
  ln->normalize_plain(x, centred, r);
  const int d = ln->dim();
  const std::vector<double>& gamma = ln->gamma();
  std::vector<double>* gr = nullptr;
  const size_t na = gamma.size();  // [G][d]: one set per token segment
  if (train) {
    gr = &grads[ln];
    if (gr->empty()) gr->assign(2 * na, 0.0);
  }
  std::vector<double> gx(x.size(), 0.0), xh(static_cast<size_t>(d)), gg(static_cast<size_t>(d));
  for (int t = 0; t < ln->tokens(); ++t) {
    const size_t base = static_cast<size_t>(t) * d;
    double m1 = 0.0, m2 = 0.0;
    for (int j = 0; j < d; ++j) {
      xh[j] = centred[base + j] * r[static_cast<size_t>(t)];
      const size_t a = ln->affine_index(t, j);
      gg[j] = g[base + j] * gamma[a];
      m1 += gg[j];
      m2 += gg[j] * xh[j];
      if (gr) {
        (*gr)[a] += g[base + j] * xh[j];
        (*gr)[na + a] += g[base + j];
      }
    }
    m1 /= d;
    m2 /= d;
    for (int j = 0; j < d; ++j) gx[base + j] = r[static_cast<size_t>(t)] * (gg[j] - m1 - xh[j] * m2);
  }
  return gx;
}

// Same derivatives as attention_backward (exp and 1/x differentiated as the
// exact functions).
std::vector<double> plain_attention_backward(Attention* at, bool train, const std::vector<double>& x,
                                             const std::vector<double>& g, Grads& grads) {
  const int n = at->tokens(), d = at->dim(), H = at->heads(), dh = at->head_width();
  TokenDense* pr[4];
  for (int i = 0; i < 4; ++i) pr[i] = at->projection(i).get();
  const std::vector<double> q = pr[0]->forward_plain(x), k = pr[1]->forward_plain(x), v = pr[2]->forward_plain(x);
  const bool softmax = at->softmax();
  const SoftmaxOptions& sm = at->softmax_options();
  // Forward again, keeping the weights.
  const size_t nd = static_cast<size_t>(n) * d;
  std::vector<double> o(nd, 0.0), sc(static_cast<size_t>(H) * n * n), e(sc.size()), a(sc.size()), y(static_cast<size_t>(H) * n, 1.0);
  for (int h = 0; h < H; ++h)
    for (int i = 0; i < n; ++i) {
      double* s = &sc[(static_cast<size_t>(h) * n + i) * n];
      double* es = &e[(static_cast<size_t>(h) * n + i) * n];
      double* as = &a[(static_cast<size_t>(h) * n + i) * n];
      for (int j = 0; j < n; ++j) {
        s[j] = 0.0;
        for (int c = h * dh; c < (h + 1) * dh; ++c) s[j] += q[static_cast<size_t>(i) * d + c] * k[static_cast<size_t>(j) * d + c];
      }
      if (softmax) {
        std::vector<double> sv(s, s + n);
        if (sm.subtract_max) {
          const double m = *std::max_element(sv.begin(), sv.end());
          for (double& z : sv) z -= m;
        }
        std::vector<double> ev = lbcrypto::EvalChebyshevFunctionPtxt([](double z) { return std::exp(z); }, sv,
                                                                     at->exp_lo(), at->exp_hi(), sm.exp_degree);
        double S = 0.0;
        for (int j = 0; j < n; ++j) S += (es[j] = ev[static_cast<size_t>(j)] * at->allowed(i, j));
        y[static_cast<size_t>(h) * n + i] = S == 0.0 && at->empty_row(i) ? 0.0 : at->inv_plain(S);
        for (int j = 0; j < n; ++j) as[j] = es[j] * y[static_cast<size_t>(h) * n + i];
      } else {
        const double c0 = at->score_coeff(h, 0), c1 = at->score_coeff(h, 1), c2 = at->score_coeff(h, 2);
        for (int j = 0; j < n; ++j) as[j] = at->allowed(i, j) * (c0 + c1 * s[j] + c2 * s[j] * s[j]);
      }
      for (int j = 0; j < n; ++j)
        for (int c = h * dh; c < (h + 1) * dh; ++c) o[static_cast<size_t>(i) * d + c] += as[j] * v[static_cast<size_t>(j) * d + c];
    }
  const std::vector<double> go = plain_linear_backward(pr[3], train, o, g, grads);
  std::vector<double> dq(nd, 0.0), dk(nd, 0.0), dv(nd, 0.0), da(static_cast<size_t>(n)), ds(static_cast<size_t>(n));
  for (int h = 0; h < H; ++h)
    for (int i = 0; i < n; ++i) {
      const double* s = &sc[(static_cast<size_t>(h) * n + i) * n];
      const double* es = &e[(static_cast<size_t>(h) * n + i) * n];
      const double* as = &a[(static_cast<size_t>(h) * n + i) * n];
      for (int j = 0; j < n; ++j) {
        da[j] = 0.0;
        for (int c = h * dh; c < (h + 1) * dh; ++c) {
          da[j] += go[static_cast<size_t>(i) * d + c] * v[static_cast<size_t>(j) * d + c];
          dv[static_cast<size_t>(j) * d + c] += as[j] * go[static_cast<size_t>(i) * d + c];
        }
      }
      if (softmax) {
        const double yi = y[static_cast<size_t>(h) * n + i];
        double dy = 0.0;
        for (int j = 0; j < n; ++j) dy += da[j] * es[j];
        const double dS = -yi * yi * dy;
        for (int j = 0; j < n; ++j) ds[j] = (da[j] * yi + dS) * es[j];
      } else {
        const double c1 = at->score_coeff(h, 1), c2 = at->score_coeff(h, 2);
        for (int j = 0; j < n; ++j) ds[j] = at->allowed(i, j) * da[j] * (c1 + 2.0 * c2 * s[j]);
      }
      for (int j = 0; j < n; ++j)
        for (int c = h * dh; c < (h + 1) * dh; ++c) {
          dq[static_cast<size_t>(i) * d + c] += ds[j] * k[static_cast<size_t>(j) * d + c];
          dk[static_cast<size_t>(j) * d + c] += ds[j] * q[static_cast<size_t>(i) * d + c];
        }
    }
  std::vector<double> gx = plain_linear_backward(pr[0], train, x, dq, grads);
  const std::vector<double> gk = plain_linear_backward(pr[1], train, x, dk, grads);
  const std::vector<double> gv = plain_linear_backward(pr[2], train, x, dv, grads);
  for (size_t i = 0; i < gx.size(); ++i) gx[i] += gk[i] + gv[i];
  return gx;
}

// The gradient goes to the winner of the same tournament the encrypted layer
// runs (tap blocks padded with tap 0; a tie keeps the lower block, like
// [a < b] = 0), times the layer's gain.
std::vector<double> plain_maxpool_backward(const MaxPool2d* mp, const std::vector<double>& x,
                                           const std::vector<double>& g) {
  const Shape& in = mp->in_shape();
  const Shape& out = mp->out_shape();
  const int k = mp->k(), st = mp->stride();
  std::vector<double> gx(x.size(), 0.0);
  for (int c = 0; c < out.c; ++c)
    for (int oy = 0; oy < out.h; ++oy)
      for (int ox = 0; ox < out.w; ++ox) {
        std::vector<size_t> vals;
        for (uint32_t j = 0; j < mp->blocks(); ++j) {
          const int tap = j < static_cast<uint32_t>(k * k) ? static_cast<int>(j) : 0;
          vals.push_back((static_cast<size_t>(c) * in.h + oy * st + tap / k) * in.w + ox * st + tap % k);
        }
        for (size_t h = vals.size() / 2; h >= 1; h /= 2)
          for (size_t i = 0; i < h; ++i)
            if (x[vals[i]] < x[vals[i + h]]) vals[i] = vals[i + h];
        gx[vals[0]] += mp->gain() * g[(static_cast<size_t>(c) * out.h + oy) * out.w + ox];
      }
  return gx;
}

// Attention keeps copies of its projections' weights: refresh them after the
// projections were updated.
void sync_attention(const std::vector<std::shared_ptr<Layer>>& layers) {
  for (const auto& l : layers) {
    if (auto* r = dynamic_cast<Residual*>(l.get())) sync_attention(r->body());
    if (auto* at = dynamic_cast<Attention*>(l.get())) at->set_params(at->params());
  }
}

// Batch norm in training mode on a batch, as the trainer computes it: batch
// mean and biased variance per channel, the layer's 1/sqrt approximation;
// keeps x_hat and r for the backward pass and moves the running statistics.
std::vector<std::vector<double>> plain_batchnorm_forward(BatchNorm* bn, const std::vector<std::vector<double>>& xs,
                                                         PlainTrace& tr, bool update_running) {
  const size_t C = bn->channels(), hw = bn->positions(), B = xs.size();
  const double count = static_cast<double>(B * hw);
  std::vector<double> mean(C, 0.0), var(C, 0.0);
  for (const auto& x : xs)
    for (size_t i = 0; i < x.size(); ++i) mean[i / hw] += x[i] / count;
  for (const auto& x : xs)
    for (size_t i = 0; i < x.size(); ++i) var[i / hw] += (x[i] - mean[i / hw]) * (x[i] - mean[i / hw]) / count;
  tr.r.assign(C, 0.0);
  for (size_t c = 0; c < C; ++c) tr.r[c] = bn->inv_sqrt(var[c]);
  tr.xhat.assign(B, {});
  std::vector<std::vector<double>> ys(B);
  for (size_t b = 0; b < B; ++b) {
    tr.xhat[b].resize(xs[b].size());
    ys[b].resize(xs[b].size());
    for (size_t i = 0; i < xs[b].size(); ++i) {
      const size_t c = i / hw;
      tr.xhat[b][i] = (xs[b][i] - mean[c]) * tr.r[c];
      ys[b][i] = bn->gamma()[c] * tr.xhat[b][i] + bn->beta()[c];
    }
  }
  if (update_running) {
    const double m = bn->momentum(), unbias = count > 1.0 ? count / (count - 1.0) : 1.0;
    std::vector<double> rm = bn->running_mean(), rv = bn->running_var();
    for (size_t c = 0; c < C; ++c) {
      rm[c] = (1.0 - m) * rm[c] + m * mean[c];
      rv[c] = (1.0 - m) * rv[c] + m * unbias * var[c];
    }
    bn->set_running(rm, rv);
  }
  return ys;
}

void plain_batchnorm_backward(BatchNorm* bn, bool train, const PlainTrace& tr, std::vector<std::vector<double>>& gs,
                              Grads& grads) {
  const size_t C = bn->channels(), hw = bn->positions(), B = gs.size();
  const double count = static_cast<double>(B * hw);
  if (train) {
    auto& gr = grads[bn];
    if (gr.empty()) gr.assign(2 * C, 0.0);
    for (size_t b = 0; b < B; ++b)
      for (size_t i = 0; i < gs[b].size(); ++i) {
        gr[i / hw] += gs[b][i] * tr.xhat[b][i];
        gr[C + i / hw] += gs[b][i];
      }
  }
  std::vector<double> m1(C, 0.0), m2(C, 0.0);
  for (size_t b = 0; b < B; ++b)
    for (size_t i = 0; i < gs[b].size(); ++i) {
      const double gg = gs[b][i] * bn->gamma()[i / hw];
      m1[i / hw] += gg / count;
      m2[i / hw] += gg * tr.xhat[b][i] / count;
    }
  for (size_t b = 0; b < B; ++b)
    for (size_t i = 0; i < gs[b].size(); ++i) {
      const size_t c = i / hw;
      gs[b][i] = tr.r[c] * (gs[b][i] * bn->gamma()[c] - m1[c] - tr.xhat[b][i] * m2[c]);
    }
}

// A batch layer by layer (batch norms see the whole batch); tr.in[i][b] is
// sample b's input of layer i.
std::vector<std::vector<double>> plain_forward(const std::vector<std::shared_ptr<Layer>>& layers, size_t from,
                                               std::vector<std::vector<double>> xs, PlainTrace& tr,
                                               bool update_running) {
  tr.in.assign(layers.size(), {});
  tr.body.assign(layers.size(), {});
  for (size_t i = from; i < layers.size(); ++i) {
    tr.in[i] = xs;
    if (auto* r = dynamic_cast<Residual*>(layers[i].get())) {
      std::vector<std::vector<double>> ys = plain_forward(r->body(), 0, xs, tr.body[i], update_running);
      for (size_t b = 0; b < xs.size(); ++b)
        for (size_t k = 0; k < xs[b].size(); ++k) xs[b][k] += ys[b][k];
    } else if (auto* bn = dynamic_cast<BatchNorm*>(layers[i].get())) {
      xs = plain_batchnorm_forward(bn, xs, tr.body[i], update_running);
    } else if (auto* dr = dynamic_cast<Dropout*>(layers[i].get())) {
      tr.body[i].xhat.assign(xs.size(), {});  // the masks
      for (size_t b = 0; b < xs.size(); ++b) {
        tr.body[i].xhat[b] = dr->draw();
        for (size_t k = 0; k < xs[b].size(); ++k) xs[b][k] *= tr.body[i].xhat[b][k];
      }
    } else {
      for (auto& x : xs) x = layers[i]->forward_plain(x);
    }
  }
  return xs;
}

void plain_backward(const std::vector<std::shared_ptr<Layer>>& layers, size_t from, const std::vector<char>& train,
                    const PlainTrace& tr, std::vector<std::vector<double>>& gs, Grads& grads) {
  for (size_t k = layers.size(); k-- > from;) {
    Layer* l = layers[k].get();
    if (auto* r = dynamic_cast<Residual*>(l)) {
      std::vector<char> t(r->body().size(), train[k]);
      std::vector<std::vector<double>> gb = gs;
      plain_backward(r->body(), 0, t, tr.body[k], gb, grads);
      for (size_t b = 0; b < gs.size(); ++b)
        for (size_t i = 0; i < gs[b].size(); ++i) gs[b][i] += gb[b][i];
      continue;
    }
    if (auto* bn = dynamic_cast<BatchNorm*>(l)) {
      plain_batchnorm_backward(bn, train[k], tr.body[k], gs, grads);
      continue;
    }
    if (dynamic_cast<Dropout*>(l)) {
      for (size_t b = 0; b < gs.size(); ++b)
        for (size_t i = 0; i < gs[b].size(); ++i) gs[b][i] *= tr.body[k].xhat[b][i];
      continue;
    }
    for (size_t b = 0; b < gs.size(); ++b) {
      const std::vector<double>& x = tr.in[k][b];
      std::vector<double>& g = gs[b];
      if (auto* lin = dynamic_cast<LinearLayer*>(l)) {
        g = plain_linear_backward(lin, train[k], x, g, grads);
      } else if (auto* ln = dynamic_cast<LayerNorm*>(l)) {
        g = plain_layernorm_backward(ln, train[k], x, g, grads);
      } else if (auto* at = dynamic_cast<Attention*>(l)) {
        g = plain_attention_backward(at, train[k], x, g, grads);
      } else if (auto* mp = dynamic_cast<MaxPool2d*>(l)) {
        g = plain_maxpool_backward(mp, x, g);
      } else if (dynamic_cast<Bootstrap*>(l)) {
        // identity
      } else {
        std::vector<double> d = activation_derivative_plain(*l, x);
        for (size_t i = 0; i < g.size(); ++i) g[i] *= d[i];
      }
    }
  }
}

}  // namespace

// -------------------------------------------------------------------- Trainer

Trainer::Trainer(Network& net, std::vector<size_t> frozen, Loss loss) : net_(net), loss_(loss) {
  if (!net.slots()) throw std::logic_error("train: compile() the network first");
  slots_ = net.slots();
  first_ = first_trainable(net, frozen);
  const auto& layers = net.layers();
  for (size_t i = first_; i < layers.size(); ++i) nodes_.push_back(build(layers[i].get(), !contains(frozen, i), slots_));
  if (loss_.kind == Loss::kCrossEntropy) {
    const Layer& last = *layers.back();
    ce_ = std::make_unique<CrossEntropy>(loss_.softmax, last.out_shape().size());
    ce_->set_layout(last, slots_, nodes_.back()->kind == Node::kLinear);
  }
}

Trainer::~Trainer() = default;

uint32_t Trainer::step_depth() const {
  const uint32_t d = unbounded_step_depth();
  return budget_ ? std::min(d, budget_) : d;
}

uint32_t Trainer::unbounded_step_depth() const {
  // Relative to the weights' level; the frozen prefix runs first.
  uint32_t prefix = 0;
  const auto& layers = net_.layers();
  for (size_t i = 0; i < first_; ++i) prefix = layers[i]->is_bootstrap() ? 0 : prefix + layers[i]->depth();
  Sim s{0};
  SimTrace tr;
  uint32_t y = sim_forward(s, nodes_, prefix, tr);
  sim_backward(s, nodes_, tr, sim_loss(s, ce_.get(), y));
  return std::max(s.max, s.new_w);
}

uint32_t Trainer::min_level_budget() const {
  // The deepest indivisible operation; any product of fitted operands takes
  // one level, a shared-weight update or the softmax backward two.
  uint32_t m = 2;
  const auto& layers = net_.layers();
  for (size_t i = 0; i < first_; ++i) m = std::max(m, layers[i]->depth());
  std::function<void(const std::vector<std::unique_ptr<Node>>&)> walk = [&](const auto& nodes) {
    for (const auto& np : nodes) {
      const Node& n = *np;
      switch (n.kind) {
        case Node::kSquare:
        case Node::kPoly:
        case Node::kCheb: m = std::max({m, n.layer->depth(), derivative_depth(n)}); break;
        case Node::kNorm: m = std::max(m, n.norm->inv_sqrt_depth()); break;
        case Node::kBatchNorm: m = std::max(m, n.bn->inv_sqrt_depth()); break;
        case Node::kAttn:
          if (n.att->softmax()) {
            const SoftmaxOptions& o = n.att->softmax_options();
            m = std::max({m, chebyshev_depth(o.exp_degree, n.att->exp_lo(), n.att->exp_hi()),
                          chebyshev_depth(o.inv_degree, n.att->sum_lo(), n.att->sum_hi())});
          }
          break;
        case Node::kResidual: walk(n.body); break;
        default: break;
      }
    }
  };
  walk(nodes_);
  if (ce_)
    m = std::max({m, chebyshev_depth(ce_->o.exp_degree, ce_->exp_lo, ce_->exp_hi),
                  chebyshev_depth(ce_->o.inv_degree, ce_->sum_lo, ce_->sum_hi)});
  return m;
}

void Trainer::set_level_budget(uint32_t levels, double range) {
  if (!(range >= 1.0)) throw std::invalid_argument("train: the bootstrap range must be at least 1");
  if (levels && levels < min_level_budget())
    throw std::invalid_argument("train: a level budget of " + std::to_string(levels) + " is below the " +
                                std::to_string(min_level_budget()) + " levels the deepest single operation needs");
  budget_ = levels;
  log_range_ = 0;
  while (std::ldexp(1.0, static_cast<int>(log_range_)) < range) ++log_range_;
}

std::vector<int32_t> Trainer::rotation_indices(bool ladder) const {
  std::set<int32_t> r;
  const auto& layers = net_.layers();
  for (size_t i = 0; i < first_; ++i)
    for (int32_t k : layers[i]->rotation_indices()) r.insert(k);
  if (ladder) return std::vector<int32_t>(r.begin(), r.end());
  bool all = false;
  collect_rotations(nodes_, slots_, r, all);
  if (ce_) {
    all = true;  // the softmax sum; the max tree uses its first steps
    if (ce_->o.subtract_max)
      for (int t = 1; t < ce_->window; t *= 2) r.insert(normalize_rotation(-static_cast<int64_t>(t), slots_));
  }
  if (all)
    for (uint32_t k = 1; k < slots_; k *= 2) r.insert(static_cast<int32_t>(k));
  return std::vector<int32_t>(r.begin(), r.end());
}

ContextOptions Trainer::context_options(bool secure, bool bootstrapping, uint32_t updates,
                                        uint32_t rotation_window) const {
  ContextOptions o;
  o.slots = slots_;
  o.secure = secure;
  o.bootstrapping = bootstrapping;
  o.rotation_window = rotation_window;
  o.scheme_switching = net_.uses_scheme_switching() || (ce_ && ce_->o.subtract_max);
  if (bootstrapping) {
    // +1: weights restart one level below a fresh bootstrap (unpacking mask);
    // +1: EvalBootstrap needs one level left; +1 when activations and
    // gradients are bootstrapped with a range: the division before it.
    const bool inside = budget_ && unbounded_step_depth() > budget_;
    o.mult_depth = step_depth() + 2 + (inside && log_range_ ? 1 : 0);
    // Bootstrapping precision: about 1e-2 at 50 bits on full-slot toy rings,
    // 1e-10 at 50 bits on ring 2^16, where 55 bits would not fit 24 levels.
    o.scale_mod_bits = secure ? 50 : 55;
    o.first_mod_bits = std::min<uint32_t>(o.scale_mod_bits + 7, 60);  // 60 is OpenFHE's native prime limit
  } else {
    // Each update leaves the weights deeper; the next step starts there.
    uint32_t prefix = 0;
    const auto& layers = net_.layers();
    for (size_t i = 0; i < first_; ++i) prefix += layers[i]->depth();
    uint32_t w = 0, top = 0;
    for (uint32_t u = 0; u < std::max(updates, 1u); ++u) {
      Sim s{w};
      SimTrace tr;
      uint32_t y = sim_forward(s, nodes_, std::max(prefix, 0u), tr);
      sim_backward(s, nodes_, tr, sim_loss(s, ce_.get(), y));
      top = std::max({top, s.max, s.new_w});
      w = s.new_w;
    }
    // One more forward pass to use the trained weights. s.max, not just the
    // output level: a max pool's comparisons need two levels below their
    // inputs.
    Sim s{w};
    s.eval = true;
    SimTrace tr;
    sim_forward(s, nodes_, prefix, tr);
    top = std::max(top, s.max);
    o.mult_depth = top;
  }
  return o;
}

void Trainer::encrypt_weights(const FHEContext& ctx) {
  if (ctx.slots() != slots_) throw std::invalid_argument("train: context slot count differs from the network");
  base_level_ = ctx.options().bootstrapping ? ctx.bootstrap_depth() + 1 : 0;
  for_each_weighted(nodes_, [&](Node& n) {
    if (n.kind == Node::kBatchNorm) {
      // The running statistics, broadcast over each channel's slots.
      std::vector<double> m(slots_, 0.0), v(slots_, 0.0);
      for (size_t c = 0; c < n.channel_slots.size(); ++c)
        for (uint32_t sl : n.channel_slots[c]) {
          m[sl] = n.bn->running_mean()[c];
          v[sl] = n.bn->running_var()[c];
        }
      n.rm = ctx.encrypt(m, base_level_);
      n.rv = ctx.encrypt(v, base_level_);
    }
    if (!n.trainable) return;
    n.D.assign(n.offs.size(), nullptr);
    n.gD.assign(n.offs.size(), nullptr);
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
    for (size_t i = 0; i < n.offs.size(); ++i) n.D[i] = ctx.encrypt(n.diag[i], base_level_);
    if (!n.bias.empty()) n.B = ctx.encrypt(n.bias, base_level_);
  });
  encrypted_ = true;
  batch_ = 0;
}

Ctxt Trainer::encrypt_input(const FHEContext& ctx, const std::vector<double>& x) const {
  return ctx.encrypt(x, ctx.options().bootstrapping ? ctx.bootstrap_depth() + 1 : 0);
}

Ctxt Trainer::encrypt_target(const FHEContext& ctx, const std::vector<double>& t) const {
  const Layer& last = *net_.layers().back();
  if (t.size() != last.out_shape().size()) throw std::invalid_argument("train: target size mismatch");
  std::vector<double> v(slots_, 0.0);
  for (size_t i = 0; i < t.size(); ++i) v[last.out_layout().slot(last.out_shape(), i)] = t[i];
  return ctx.encrypt(v, ctx.options().bootstrapping ? ctx.bootstrap_depth() + 1 : 0);
}

namespace {

Refresh make_refresh(const FHEContext& ctx, uint32_t base, uint32_t budget, uint32_t log_range) {
  Refresh rf;
  rf.ctx = &ctx;
  rf.base = static_cast<int>(base);
  rf.budget = ctx.options().bootstrapping ? budget : 0;
  rf.log_range = log_range;
  return rf;
}

Ctxt run_forward(const FHEContext& ctx, Refresh& rf, const Network& net, size_t first,
                 std::vector<std::unique_ptr<Node>>& nodes, Ctxt c) {
  const auto& layers = net.layers();
  for (size_t i = 0; i < first; ++i) {
    if (!layers[i]->is_bootstrap()) rf.fit(c, layers[i]->depth());
    c = layers[i]->forward(ctx, c);
  }
  return forward_nodes(ctx, rf, nodes, c);
}

}  // namespace

Ctxt Trainer::forward(const FHEContext& ctx, const Ctxt& x) {
  if (!encrypted_) throw std::logic_error("train: encrypt_weights() first");
  Refresh rf = make_refresh(ctx, base_level_, budget_, log_range_);
  Ctxt y = run_forward(ctx, rf, net_, first_, nodes_, x);
  sample_bootstraps_ = rf.count;
  return y;
}

Ctxt Trainer::accumulate(const FHEContext& ctx, const Ctxt& x, const Ctxt& target) {
  return accumulate_batch(ctx, {x}, {target})[0];
}

std::vector<Ctxt> Trainer::accumulate_batch(const FHEContext& ctx, const std::vector<Ctxt>& xs,
                                            const std::vector<Ctxt>& targets) {
  if (!encrypted_) throw std::logic_error("train: encrypt_weights() first");
  if (xs.empty() || xs.size() != targets.size())
    throw std::invalid_argument("train: need matching, non-empty batches of inputs and targets");
  Refresh rf = make_refresh(ctx, base_level_, budget_, log_range_);
  const auto& layers = net_.layers();
  std::vector<Ctxt> ys = xs;
  for (Ctxt& c : ys)
    for (size_t i = 0; i < first_; ++i) {
      if (!layers[i]->is_bootstrap()) rf.fit(c, layers[i]->depth());
      c = layers[i]->forward(ctx, c);
    }
  forward_batch(ctx, rf, nodes_, ys);
  std::vector<Ctxt> gs(ys.size());
  for (size_t b = 0; b < ys.size(); ++b) gs[b] = loss_gradient(ctx, rf, ys[b], targets[b]);
  backward_batch(ctx, rf, nodes_, gs, false);
  sample_bootstraps_ = rf.count;
  batch_ += xs.size();
  return ys;
}

Ctxt Trainer::loss_gradient(const FHEContext& ctx, Refresh& rf, Ctxt y, const Ctxt& target) const {
  const CC& cc = ctx.cc();
  if (!ce_) return cc->EvalSub(y, target);  // d(0.5 |y - t|^2)/dy
  const CrossEntropy& ce = *ce_;
  Ctxt z;
  if (ce.o.subtract_max) {
    // Max tree over the window: after it slot 0 holds the largest logit (the
    // padding holds lo); keep slot 0 and spread it back over the window.
    rf.fit(y, 1);
    z = add_plain(ctx, mul_plain(ctx, y, ce.mask), ce.pad);
    const double range = 2.0 * std::max(std::fabs(ce.o.lo), std::fabs(ce.o.hi));
    Ctxt m = z;
    for (int t = 1; t < ce.window; t *= 2) {
      rf.fit(m, 1);  // compare() needs two levels left, the selection product one
      Ctxt r = ctx.rotate(m, t);
      Ctxt sel = ctx.compare(m, r, range);  // 1 where m < r
      m = cc->EvalAdd(m, cc->EvalMult(sel, cc->EvalSub(r, m)));
    }
    rf.fit(m, 1);
    z = cc->EvalSub(z, window_broadcast(ctx, mul_plain(ctx, m, ce.first), ce.window));
  } else {
    if (ce.mask_input) rf.fit(y, 1);
    z = add_plain(ctx, ce.mask_input ? mul_plain(ctx, y, ce.mask) : y, ce.pad);
  }
  std::vector<double> c = ce.exp_c;
  rf.fit(z, chebyshev_depth(ce.o.exp_degree, ce.exp_lo, ce.exp_hi));
  Ctxt ez = cc->EvalChebyshevSeries(z, c, ce.exp_lo, ce.exp_hi);
  rf.fit(ez, 1);
  Ctxt e = mul_plain(ctx, ez, ce.mask);
  // Sum of every slot, in every slot (not allsum(): that adds in place).
  Ctxt S = e;
  for (uint32_t k = 1; k < ctx.slots(); k *= 2) S = cc->EvalAdd(S, ctx.rotate(S, static_cast<int32_t>(k)));
  c = ce.inv_c;
  rf.fit(S, chebyshev_depth(ce.o.inv_degree, ce.sum_lo, ce.sum_hi));
  Ctxt r = cc->EvalChebyshevSeries(S, c, ce.sum_lo, ce.sum_hi);
  for (uint32_t i = 0; i < ce.o.newton; ++i) {
    rf.fit(S, 2);
    rf.fit(r, 2);
    r = cc->EvalMult(r, cc->EvalSub(2.0, cc->EvalMult(S, r)));
  }
  rf.fit(e, 1);
  rf.fit(r, 1);
  return cc->EvalSub(cc->EvalMult(e, r), target);  // softmax(y) - t
}

void Trainer::update(const FHEContext& ctx, double lr) {
  if (!batch_) throw std::logic_error("train: no accumulated gradients");
  const double step = -lr / static_cast<double>(batch_);
  Refresh rf = make_refresh(ctx, base_level_, budget_, log_range_);
  for_each_weighted(nodes_, [&](Node& n) {
    if (n.trainable) update_linear(ctx, rf, n, step);
  });
  if (ctx.options().bootstrapping) {
    std::map<uint32_t, std::vector<Ctxt*>> by_window;
    for_each_weighted(nodes_, [&](Node& n) {
      if (n.kind == Node::kBatchNorm) {
        by_window[n.window].push_back(&n.rm);
        by_window[n.window].push_back(&n.rv);
      }
      if (!n.trainable) return;
      for (auto& d : n.D) by_window[n.window].push_back(&d);
      if (n.B) by_window[n.window].push_back(&n.B);
    });
    for (auto& kv : by_window) bootstrap_packed(ctx, kv.second, kv.first);
  }
  batch_ = 0;
}

std::vector<double> Trainer::decrypt_output(const FHEContext& ctx, const Ctxt& y) const {
  return net_.decrypt_output(ctx, y);
}

void Trainer::decrypt_weights(const FHEContext& ctx) {
  if (!encrypted_) throw std::logic_error("train: encrypt_weights() first");
  for_each_weighted(nodes_, [&](Node& n) {
    if (n.kind == Node::kBatchNorm) {
      const std::vector<double> m = ctx.decrypt(n.rm, slots_), v = ctx.decrypt(n.rv, slots_);
      std::vector<double> mean(n.channel_slots.size(), 0.0), var(mean.size(), 0.0);
      for (size_t c = 0; c < mean.size(); ++c) {
        for (uint32_t sl : n.channel_slots[c]) {
          mean[c] += m[sl];
          var[c] += v[sl];
        }
        mean[c] /= static_cast<double>(n.channel_slots[c].size());
        var[c] /= static_cast<double>(n.channel_slots[c].size());
      }
      n.bn->set_running(mean, var);
    }
    if (!n.trainable) return;
    std::vector<double> p = n.layer->params(), total(p.size(), 0.0), count(p.size(), 0.0);
    auto take = [&](const Ctxt& c, const std::vector<int>& owner) {
      std::vector<double> v = ctx.decrypt(c, slots_);
      for (uint32_t s = 0; s < slots_; ++s)
        if (owner[s] >= 0) {
          total[static_cast<size_t>(owner[s])] += v[s];
          count[static_cast<size_t>(owner[s])] += 1.0;
        }
    };
    for (size_t i = 0; i < n.offs.size(); ++i) take(n.D[i], n.owner[i]);
    if (n.B) take(n.B, n.bias_owner);
    // Shared weights hold the same value in every slot (up to CKKS noise).
    for (size_t k = 0; k < p.size(); ++k)
      if (count[k] > 0) p[k] = total[k] / count[k];
    n.layer->set_params(p);
  });
  for_each_attention(nodes_, [](Node& n) { n.att->set_params(n.att->params()); });
  net_.compile(slots_);
}

bool Trainer::batch_coupled() const {
  bool any = false;
  for_each_weighted(nodes_, [&](const Node& n) { any = any || n.kind == Node::kBatchNorm; });
  return any;
}

size_t Trainer::weight_ciphertexts() const {
  size_t k = 0;
  for_each_weighted(nodes_, [&](const Node& n) {
    if (n.trainable) k += n.offs.size() + (n.bias.empty() ? 0 : 1);
  });
  return k;
}

uint32_t Trainer::weight_level() const {
  uint32_t l = 0;
  for_each_weighted(nodes_, [&](const Node& n) {
    for (const auto& d : n.D) l = std::max(l, lvl(d));
    if (n.B) l = std::max(l, lvl(n.B));
    if (n.rm) l = std::max({l, lvl(n.rm), lvl(n.rv)});
  });
  return l;
}

std::string Trainer::summary() const {
  std::ostringstream os;
  const auto& layers = net_.layers();
  os << "training from layer " << first_ << " (" << layers[first_]->type() << ")";
  if (first_) os << ", layers 0.." << first_ - 1 << " run forward only";
  os << "\n";
  for_each_weighted(nodes_, [&](const Node& n) {
    os << "  " << n.layer->type() << ": " << n.offs.size() << (n.kind == Node::kNorm || n.kind == Node::kBatchNorm ? " gamma, " : " diagonals, ");
    if (n.trainable)
      os << n.groups.size() << " shared weights";
    else
      os << "frozen";
    os << "\n";
  });
  if (ce_)
    os << "loss: softmax cross-entropy (exp degree " << ce_->o.exp_degree << " on [" << ce_->exp_lo << ", " << ce_->exp_hi
       << "], 1/x degree " << ce_->o.inv_degree << " on [" << ce_->sum_lo << ", " << ce_->sum_hi << "], " << ce_->o.newton
       << " newton" << (ce_->o.subtract_max ? ", max subtracted" : "") << ")\n";
  else
    os << "loss: mean squared error\n";
  os << "weight ciphertexts " << weight_ciphertexts() << ", step depth " << step_depth();
  if (budget_ && unbounded_step_depth() > budget_)
    os << " (" << unbounded_step_depth() << " without bootstrapping activations and gradients, range "
       << bootstrap_range() << ")";
  return os.str();
}

// ------------------------------------------------------------- train_plain

std::vector<std::vector<double>> forward_plain_batch(const Network& net, const std::vector<std::vector<double>>& xs) {
  PlainTrace tr;
  return plain_forward(net.layers(), 0, xs, tr, false);
}

void train_plain(Network& net, const std::vector<std::vector<double>>& xs, const std::vector<std::vector<double>>& ts,
                 double lr, const std::vector<size_t>& frozen, const Loss& loss) {
  if (xs.size() != ts.size() || xs.empty()) throw std::invalid_argument("train_plain: need matching, non-empty batches");
  const auto& layers = net.layers();
  const size_t first = first_trainable(net, frozen);
  std::vector<char> train(layers.size());
  for (size_t i = 0; i < layers.size(); ++i) train[i] = !contains(frozen, i);
  Grads grads;
  std::unique_ptr<CrossEntropy> ce;
  if (loss.kind == Loss::kCrossEntropy) ce = std::make_unique<CrossEntropy>(loss.softmax, layers.back()->out_shape().size());
  std::vector<std::vector<double>> xb = xs;
  for (auto& x : xb)
    for (size_t i = 0; i < first; ++i) x = layers[i]->forward_plain(x);
  PlainTrace tr;
  std::vector<std::vector<double>> ys = plain_forward(layers, first, xb, tr, true);
  for (size_t s = 0; s < ys.size(); ++s) {
    std::vector<double>& y = ys[s];
    if (y.size() != ts[s].size()) throw std::invalid_argument("train_plain: target size mismatch");
    if (ce) y = ce->softmax(y);  // d(cross-entropy)/dy = softmax(y) - t
    for (size_t i = 0; i < y.size(); ++i) y[i] -= ts[s][i];
  }
  plain_backward(layers, first, train, tr, ys, grads);
  const double step = lr / static_cast<double>(xs.size());
  for (auto& kv : grads) {
    std::vector<double> p = kv.first->params();
    for (size_t i = 0; i < p.size(); ++i) p[i] -= step * kv.second[i];
    kv.first->set_params(p);
  }
  sync_attention(layers);
}

}  // namespace fheml
