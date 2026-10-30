#include "fheml/train.h"

#include <algorithm>
#include <cmath>
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

void add_to(const FHEContext& ctx, Ctxt& acc, const Ctxt& c) {
  if (acc)
    ctx.cc()->EvalAddInPlace(acc, c);
  else
    acc = c;
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
  enum Kind { kLinear, kSquare, kPoly, kCheb, kBoot, kResidual, kNorm, kAttn, kMax } kind = kLinear;
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
  std::vector<Ctxt> kr, vr;  // rotated (cyclic) keys and values per token offset
  std::vector<Ctxt> p, aw, A;  // scores, head-slot weights, weights broadcast over the head
  Ctxt q, o, y;               // queries, sum_s A_s (*) v_s before the softmax division, 1 / row sum

  // kMax: per tournament round, the number of values entering it and, per
  // pair, the comparison bit [a < b] and its complement, both masked to the
  // output slots (the backward pass routes the gradient with them).
  MaxPool2d* pool = nullptr;
  std::vector<double> out_mask;
  std::vector<size_t> round_sizes;
  std::vector<std::vector<Ctxt>> pick_b, pick_a;

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
  if (auto* mp = dynamic_cast<MaxPool2d*>(l)) {
    n->kind = Node::kMax;
    n->pool = mp;
    n->out_mask.assign(slots, 0.0);
    for (size_t i = 0; i < mp->out_shape().size(); ++i) n->out_mask[mp->out_layout().slot(mp->out_shape(), i)] = 1.0;
    return n;
  }
  if (auto* at = dynamic_cast<Attention*>(l)) {
    n->kind = Node::kAttn;
    n->att = at;
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
    n->offs = {0};
    n->diag.emplace_back(slots, 0.0);
    n->bias.assign(slots, 0.0);
    n->unshared.assign(1, {});
    if (n->trainable) {
      n->owner.emplace_back(slots, -1);
      n->bias_owner.assign(slots, -1);
    }
    std::vector<std::vector<double>> masks(static_cast<size_t>(d), std::vector<double>(slots, 0.0));
    for (int t = 0; t < tokens; ++t)
      for (int j = 0; j < d; ++j) {
        const size_t sl = ol.slot(out, static_cast<size_t>(t) * d + j);
        n->diag[0][sl] = ln->gamma()[static_cast<size_t>(j)];
        n->bias[sl] = ln->beta()[static_cast<size_t>(j)];
        if (n->trainable) {
          n->owner[0][sl] = j;
          n->bias_owner[sl] = d + j;
        }
        masks[static_cast<size_t>(j)][sl] = 1.0;
      }
    if (n->trainable) {
      if (tokens == 1) {
        n->unshared[0].assign(slots, 0.0);
        n->bias_unshared.assign(slots, 0.0);
        for (int j = 0; j < d; ++j)
          for (uint32_t sl = 0; sl < slots; ++sl)
            if (masks[static_cast<size_t>(j)][sl] != 0.0) n->unshared[0][sl] = n->bias_unshared[sl] = 1.0;
      } else {
        // gamma_j and beta_j are shared by every token.
        for (int j = 0; j < d; ++j) {
          Node::Group gg, gb;
          gg.masks.emplace_back(0, masks[static_cast<size_t>(j)]);
          gb.masks.emplace_back(-1, masks[static_cast<size_t>(j)]);
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
  uint32_t see(uint32_t l) {
    max = std::max(max, l);
    return l;
  }
};

struct SimTrace {
  std::vector<uint32_t> in;  // input level of each node
  std::vector<SimTrace> body;
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

// MaxPool2d forward from input level x: every round compares (inputs two
// levels above the bottom) and selects with the fresh comparison bit; the
// masked bits sit one level below it.
uint32_t sim_maxpool_forward(Sim& s, const Node& n, uint32_t x) {
  const uint32_t c = FHEContext::compare_depth();
  if (n.pool->rounds()) s.see(c + 1);
  for (uint32_t r = 0; r < n.pool->rounds(); ++r) {
    s.see(x + 2);
    x = s.see(std::max(x, c) + 1);
  }
  return x;
}

uint32_t sim_linear_backward(Sim& s, const Node& n, uint32_t g, uint32_t z) {
  if (n.trainable) {
    uint32_t grad = s.see(std::max(g, z) + 1);
    uint32_t upd = s.see(grad + (n.groups.empty() ? 1 : 2));
    s.new_w = std::max(s.new_w, std::max(s.w, upd));
  }
  return s.see(std::max(g, n.trainable ? s.w : g) + 1);
}

uint32_t sim_forward(Sim& s, const std::vector<std::unique_ptr<Node>>& nodes, uint32_t x, SimTrace& tr) {
  tr.in.assign(nodes.size(), 0);
  tr.body.assign(nodes.size(), {});
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
      case Node::kMax:
        for (uint32_t r = 0; r < n.pool->rounds(); ++r) g = s.see(std::max(g, FHEContext::compare_depth() + 1) + 1);
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
    if (n->kind == Node::kMax)
      for (int32_t k : n->pool->offsets()) {
        add(k);
        add(-static_cast<int64_t>(k));
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
    if (n->kind == Node::kLinear || n->kind == Node::kNorm) f(*n);
  }
}

template <typename F>
void for_each_weighted(const std::vector<std::unique_ptr<Node>>& nodes, F f) {
  for (const auto& n : nodes) {
    if (n->kind == Node::kResidual || n->kind == Node::kAttn) for_each_weighted(n->body, f);
    if (n->kind == Node::kLinear || n->kind == Node::kNorm) f(*n);
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

Ctxt linear_forward(const FHEContext& ctx, Node& n, const Ctxt& x) {
  const CC& cc = ctx.cc();
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
Ctxt linear_backward(const FHEContext& ctx, Node& n, const Ctxt& g, bool need) {
  const CC& cc = ctx.cc();
  if (n.trainable) {
    const std::vector<Ctxt> rot = ctx.rotate_many(n.in, n.offs);
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
    for (size_t i = 0; i < n.offs.size(); ++i) add_to(ctx, n.gD[i], cc->EvalMult(g, rot[i]));
    if (n.B) add_to(ctx, n.gB, g);
  }
  n.in = nullptr;
  if (!need) return nullptr;
  std::vector<Ctxt> terms(n.offs.size());
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
  for (size_t i = 0; i < n.offs.size(); ++i) {
    Ctxt t = n.trainable ? cc->EvalMult(n.D[i], g) : mul_plain(ctx, g, n.diag[i]);
    terms[i] = ctx.rotate(t, -n.offs[i]);
  }
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
// the backward pass needs.
Ctxt attention_forward(const FHEContext& ctx, Node& n, const Ctxt& x) {
  const CC& cc = ctx.cc();
  const Attention& at = *n.att;
  const int tokens = at.tokens(), dh = at.head_width(), B = at.stride();
  const int32_t nd = tokens * B;
  n.q = linear_forward(ctx, *n.body[0], x);
  Ctxt k = linear_forward(ctx, *n.body[1], x);
  Ctxt v = linear_forward(ctx, *n.body[2], x);
  // Cyclic copies in [n*B, 2n*B) so rotating by s*B maps token i to (i + s) mod n.
  k = cc->EvalAdd(k, ctx.rotate(k, -nd));
  v = cc->EvalAdd(v, ctx.rotate(v, -nd));
  std::vector<int32_t> offs(static_cast<size_t>(tokens));
  for (int s = 0; s < tokens; ++s) offs[static_cast<size_t>(s)] = s * B;
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
  std::vector<Ctxt> partial(static_cast<size_t>(tokens));
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
  for (int s = 0; s < tokens; ++s) {
    const size_t i = static_cast<size_t>(s);
    Ctxt a;
    if (softmax) {
      std::vector<double> c = at.exp_coeffs();
      a = mul_plain(ctx, cc->EvalChebyshevSeries(n.p[i], c, at.exp_lo(), at.exp_hi()), at.head_mask());
    } else {
      a = add_plain(ctx, mul_plain(ctx, n.p[i], at.score_mask(2)), at.score_mask(1));
      a = add_plain(ctx, cc->EvalMult(n.p[i], a), at.score_mask(0));
    }
    n.aw[i] = a;
    n.A[i] = window_broadcast(ctx, a, dh);
    partial[i] = cc->EvalMult(n.A[i], n.vr[i]);
  }
  Ctxt o = sum(ctx, std::move(partial));
  if (softmax) {
    const SoftmaxOptions& sm = at.softmax_options();
    Ctxt S = tokens > 1 ? cc->EvalAdd(n.A[0], n.A[1]) : n.A[0];
    for (int s = 2; s < tokens; ++s) cc->EvalAddInPlace(S, n.A[static_cast<size_t>(s)]);
    S = add_plain(ctx, S, at.pad_sum());
    std::vector<double> c = at.inv_coeffs();
    Ctxt y = cc->EvalChebyshevSeries(S, c, at.sum_lo(), at.sum_hi());
    for (uint32_t i = 0; i < sm.newton; ++i) y = cc->EvalMult(y, cc->EvalSub(2.0, cc->EvalMult(S, y)));
    n.y = y;
    n.o = o;
    o = cc->EvalMult(o, y);
  }
  return linear_forward(ctx, *n.body[3], o);
}

// The gradient of attention_forward. The softmax and 1/x are differentiated
// as the functions they approximate (exp' = exp, (1/S)' = -1/S^2), so the
// gradient matches the exact softmax up to the approximation error.
Ctxt attention_backward(const FHEContext& ctx, Node& n, const Ctxt& g, bool need) {
  const CC& cc = ctx.cc();
  const Attention& at = *n.att;
  const int tokens = at.tokens(), dh = at.head_width(), B = at.stride();
  const int32_t nd = tokens * B;
  const bool softmax = at.softmax();
  Ctxt go = linear_backward(ctx, *n.body[3], g, true);
  Ctxt dO = go, c;
  if (softmax) {
    // out = o (*) y, y = 1 / S, S = sum_s A_s: dL/dA_s = dO (*) v_s - go (*) o (*) y^2.
    dO = cc->EvalMult(go, n.y);
    c = cc->EvalMult(cc->EvalMult(go, n.o), cc->EvalMult(n.y, n.y));
  }
  std::vector<Ctxt> P(static_cast<size_t>(tokens)), dqs(static_cast<size_t>(tokens)), dks(static_cast<size_t>(tokens)),
      dvs(static_cast<size_t>(tokens));
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
  for (int s = 0; s < tokens; ++s) {
    const size_t i = static_cast<size_t>(s);
    Ctxt dA = cc->EvalMult(dO, n.vr[i]);
    if (c) dA = cc->EvalSub(dA, c);
    // A_s broadcasts a_s over the head: da_s = window sum of dA_s; then through
    // the (masked) score function to the q.k sums.
    Ctxt da = window_sum(ctx, dA, dh);
    Ctxt dp;
    if (softmax) {
      dp = cc->EvalMult(da, n.aw[i]);
    } else {
      Ctxt slope = add_plain(ctx, mul_plain(ctx, n.p[i], scaled(at.score_mask(2), 2.0)), at.score_mask(1));
      dp = cc->EvalMult(da, slope);
    }
    P[i] = window_broadcast(ctx, dp, dh);  // the q.k sums spread over the head
    dqs[i] = cc->EvalMult(P[i], n.kr[i]);
    dks[i] = ctx.rotate(cc->EvalMult(P[i], n.q), -s * B);
    dvs[i] = ctx.rotate(cc->EvalMult(dO, n.A[i]), -s * B);
  }
  Ctxt dq = sum(ctx, std::move(dqs));
  // Fold the cyclic copies [n*B, 2n*B) back onto the tokens.
  Ctxt dk = sum(ctx, std::move(dks));
  dk = cc->EvalAdd(dk, ctx.rotate(dk, nd));
  Ctxt dv = sum(ctx, std::move(dvs));
  dv = cc->EvalAdd(dv, ctx.rotate(dv, nd));
  Ctxt gx = linear_backward(ctx, *n.body[0], dq, need);
  Ctxt gk = linear_backward(ctx, *n.body[1], dk, need);
  Ctxt gv = linear_backward(ctx, *n.body[2], dv, need);
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
Ctxt maxpool_forward(const FHEContext& ctx, Node& n, const Ctxt& x) {
  if (!ctx.scheme_switching())
    throw std::logic_error("maxpool2d needs a context with ContextOptions::scheme_switching");
  const CC& cc = ctx.cc();
  std::vector<Ctxt> vals = ctx.rotate_many(x, n.pool->offsets());
  n.round_sizes.clear();
  n.pick_b.clear();
  n.pick_a.clear();
  while (vals.size() > 1) {
    const size_t pairs = vals.size() / 2;
    std::vector<Ctxt> next((vals.size() + 1) / 2), pb(pairs), pa(pairs);
    for (size_t i = 0; i < pairs; ++i) {
      const Ctxt& a = vals[2 * i];
      const Ctxt& b = vals[2 * i + 1];
      Ctxt sel = ctx.compare(a, b, n.pool->range());  // [a < b]
      next[i] = cc->EvalAdd(a, cc->EvalMult(sel, cc->EvalSub(b, a)));
      // Masked to the output slots: a gradient in any other slot (an output
      // layer's g = y - t there) must not leak into the inputs.
      pb[i] = mul_plain(ctx, sel, n.out_mask);
      pa[i] = add_plain(ctx, cc->EvalNegate(pb[i]), n.out_mask);
    }
    if (vals.size() % 2) next.back() = vals.back();
    n.round_sizes.push_back(vals.size());
    n.pick_b.push_back(std::move(pb));
    n.pick_a.push_back(std::move(pa));
    vals.swap(next);
  }
  return vals[0];
}

// Routes g down the tournament (d max / da = 1 - [a < b], d max / db = [a < b])
// to the window taps, then rotates each tap's gradient back to its input slot.
// Overlapping windows (stride < k) add up.
Ctxt maxpool_backward(const FHEContext& ctx, Node& n, const Ctxt& g, bool need) {
  const CC& cc = ctx.cc();
  std::vector<Ctxt> gs;
  if (need) {
    gs.push_back(g);
    for (size_t r = n.round_sizes.size(); r-- > 0;) {
      const size_t m = n.round_sizes[r];
      std::vector<Ctxt> prev(m);
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
      for (size_t i = 0; i < m / 2; ++i) {
        prev[2 * i] = cc->EvalMult(gs[i], n.pick_a[r][i]);
        prev[2 * i + 1] = cc->EvalMult(gs[i], n.pick_b[r][i]);
      }
      if (m % 2) prev.back() = gs.back();
      gs.swap(prev);
    }
  }
  n.round_sizes.clear();
  n.pick_b.clear();
  n.pick_a.clear();
  if (!need) return nullptr;
  const std::vector<int32_t> offs = n.pool->offsets();
  std::vector<Ctxt> terms(offs.size());
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
  for (size_t j = 0; j < offs.size(); ++j) terms[j] = ctx.rotate(gs[j], -offs[j]);
  return sum(ctx, std::move(terms));
}

Ctxt forward_nodes(const FHEContext& ctx, std::vector<std::unique_ptr<Node>>& nodes, Ctxt x) {
  for (auto& np : nodes) {
    Node& n = *np;
    switch (n.kind) {
      case Node::kLinear: x = linear_forward(ctx, n, x); break;
      case Node::kNorm: {
        n.norm->normalize(ctx, x, n.centred, n.r);
        Ctxt cg = n.trainable ? ctx.cc()->EvalMult(n.centred, n.D[0]) : mul_plain(ctx, n.centred, n.diag[0]);
        x = ctx.cc()->EvalMult(cg, n.r);
        if (n.trainable)
          ctx.cc()->EvalAddInPlace(x, n.B);
        else
          x = add_plain(ctx, x, n.bias);
        break;
      }
      case Node::kAttn: x = attention_forward(ctx, n, x); break;
      case Node::kMax: x = maxpool_forward(ctx, n, x); break;
      case Node::kResidual: {
        n.in = x;
        Ctxt y = forward_nodes(ctx, n.body, x);
        x = ctx.cc()->EvalAdd(y, x);
        break;
      }
      default:
        n.in = x;
        x = n.layer->forward(ctx, x);
        break;
    }
  }
  return x;
}

// Returns dL/dx (null when need_input_grad is false).
Ctxt backward_nodes(const FHEContext& ctx, std::vector<std::unique_ptr<Node>>& nodes, Ctxt g, bool need_input_grad) {
  const CC& cc = ctx.cc();
  for (size_t k = nodes.size(); k-- > 0;) {
    Node& n = *nodes[k];
    const bool need = k > 0 || need_input_grad;
    switch (n.kind) {
      case Node::kLinear: g = linear_backward(ctx, n, g, need); break;
      case Node::kNorm: {
        // y = gamma (*) x_hat + beta, x_hat = centred (*) r:
        // dgamma = g (*) x_hat, dbeta = g (summed over the tokens at update),
        // dx = r (*) (gg - mean(gg) - x_hat (*) mean(gg (*) x_hat)), gg = g (*) gamma.
        Ctxt xhat = cc->EvalMult(n.centred, n.r);
        if (n.trainable) {
          add_to(ctx, n.gD[0], cc->EvalMult(g, xhat));
          add_to(ctx, n.gB, g);
        }
        if (need) {
          Ctxt gg = n.trainable ? cc->EvalMult(g, n.D[0]) : mul_plain(ctx, g, n.diag[0]);
          Ctxt m1 = n.norm->block_mean(ctx, gg);
          Ctxt m2 = n.norm->block_mean(ctx, cc->EvalMult(gg, xhat));
          Ctxt inner = cc->EvalSub(cc->EvalSub(gg, m1), cc->EvalMult(xhat, m2));
          g = cc->EvalMult(inner, n.r);
        }
        n.centred = n.r = nullptr;
        break;
      }
      case Node::kAttn: g = attention_backward(ctx, n, g, need); break;
      case Node::kMax: g = maxpool_backward(ctx, n, g, need); break;
      case Node::kSquare:
      case Node::kPoly:
      case Node::kCheb:
        if (need) {
          if (n.kind == Node::kSquare) {
            g = cc->EvalMult(g, cc->EvalAdd(n.in, n.in));
          } else if (n.dcoef.size() == 1) {
            g = cc->EvalMult(g, n.kind == Node::kCheb ? n.dcoef[0] / 2.0 : n.dcoef[0]);
          } else {
            Ctxt f = n.kind == Node::kPoly ? PolyActivation(n.dcoef).forward(ctx, n.in)
                                           : cc->EvalChebyshevSeries(n.in, n.dcoef, n.a, n.b);
            g = cc->EvalMult(g, f);
          }
        }
        n.in = nullptr;
        break;
      case Node::kBoot:
        if (need) g = ctx.bootstrap(g);
        break;
      case Node::kResidual: {
        Ctxt gb = backward_nodes(ctx, n.body, g, need);
        if (need) g = cc->EvalAdd(gb, g);
        n.in = nullptr;
        break;
      }
    }
    if (!need) return nullptr;
  }
  return g;
}

// Sum of all slots, in every slot.
Ctxt allsum(const FHEContext& ctx, Ctxt c) {
  for (uint32_t k = 1; k < ctx.slots(); k *= 2) ctx.cc()->EvalAddInPlace(c, ctx.rotate(c, static_cast<int32_t>(k)));
  return c;
}

void update_linear(const FHEContext& ctx, Node& n, double step) {
  const CC& cc = ctx.cc();
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
  std::vector<std::vector<double>> in;
  std::vector<PlainTrace> body;
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
  if (train) {
    gr = &grads[ln];
    if (gr->empty()) gr->assign(2u * d, 0.0);
  }
  std::vector<double> gx(x.size(), 0.0), xh(static_cast<size_t>(d)), gg(static_cast<size_t>(d));
  for (int t = 0; t < ln->tokens(); ++t) {
    const size_t base = static_cast<size_t>(t) * d;
    double m1 = 0.0, m2 = 0.0;
    for (int j = 0; j < d; ++j) {
      xh[j] = centred[base + j] * r[static_cast<size_t>(t)];
      gg[j] = g[base + j] * gamma[static_cast<size_t>(j)];
      m1 += gg[j];
      m2 += gg[j] * xh[j];
      if (gr) {
        (*gr)[static_cast<size_t>(j)] += g[base + j] * xh[j];
        (*gr)[static_cast<size_t>(d + j)] += g[base + j];
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
        for (int j = 0; j < n; ++j) S += (es[j] = ev[static_cast<size_t>(j)]);
        y[static_cast<size_t>(h) * n + i] = at->inv_plain(S);
        for (int j = 0; j < n; ++j) as[j] = es[j] * y[static_cast<size_t>(h) * n + i];
      } else {
        const double c0 = at->score_coeff(h, 0), c1 = at->score_coeff(h, 1), c2 = at->score_coeff(h, 2);
        for (int j = 0; j < n; ++j) as[j] = c0 + c1 * s[j] + c2 * s[j] * s[j];
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
        for (int j = 0; j < n; ++j) ds[j] = da[j] * (c1 + 2.0 * c2 * s[j]);
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
// runs (a tie keeps the first tap, like [a < b] = 0).
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
        for (int dy = 0; dy < k; ++dy)
          for (int dx = 0; dx < k; ++dx)
            vals.push_back((static_cast<size_t>(c) * in.h + oy * st + dy) * in.w + ox * st + dx);
        while (vals.size() > 1) {
          std::vector<size_t> next((vals.size() + 1) / 2);
          for (size_t i = 0; i + 1 < vals.size(); i += 2)
            next[i / 2] = x[vals[i]] < x[vals[i + 1]] ? vals[i + 1] : vals[i];
          if (vals.size() % 2) next.back() = vals.back();
          vals.swap(next);
        }
        gx[vals[0]] += g[(static_cast<size_t>(c) * out.h + oy) * out.w + ox];
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

std::vector<double> plain_forward(const std::vector<std::shared_ptr<Layer>>& layers, size_t from, std::vector<double> x,
                                  PlainTrace& tr) {
  tr.in.assign(layers.size(), {});
  tr.body.assign(layers.size(), {});
  for (size_t i = from; i < layers.size(); ++i) {
    tr.in[i] = x;
    if (auto* r = dynamic_cast<Residual*>(layers[i].get())) {
      std::vector<double> y = plain_forward(r->body(), 0, x, tr.body[i]);
      for (size_t k = 0; k < y.size(); ++k) y[k] += x[k];
      x = std::move(y);
    } else {
      x = layers[i]->forward_plain(x);
    }
  }
  return x;
}

std::vector<double> plain_backward(const std::vector<std::shared_ptr<Layer>>& layers, size_t from,
                                   const std::vector<char>& train, const PlainTrace& tr, std::vector<double> g,
                                   Grads& grads) {
  for (size_t k = layers.size(); k-- > from;) {
    Layer* l = layers[k].get();
    const std::vector<double>& x = tr.in[k];
    if (auto* r = dynamic_cast<Residual*>(l)) {
      std::vector<char> t(r->body().size(), train[k]);
      std::vector<double> gb = plain_backward(r->body(), 0, t, tr.body[k], g, grads);
      for (size_t i = 0; i < g.size(); ++i) g[i] += gb[i];
    } else if (auto* lin = dynamic_cast<LinearLayer*>(l)) {
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
  return g;
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
    // +1: EvalBootstrap needs one level left.
    o.mult_depth = step_depth() + 2;
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

Ctxt Trainer::forward(const FHEContext& ctx, const Ctxt& x) {
  if (!encrypted_) throw std::logic_error("train: encrypt_weights() first");
  const auto& layers = net_.layers();
  Ctxt c = x;
  for (size_t i = 0; i < first_; ++i) c = layers[i]->forward(ctx, c);
  return forward_nodes(ctx, nodes_, c);
}

Ctxt Trainer::accumulate(const FHEContext& ctx, const Ctxt& x, const Ctxt& target) {
  Ctxt y = forward(ctx, x);
  backward_nodes(ctx, nodes_, loss_gradient(ctx, y, target), false);
  ++batch_;
  return y;
}

Ctxt Trainer::loss_gradient(const FHEContext& ctx, const Ctxt& y, const Ctxt& target) const {
  const CC& cc = ctx.cc();
  if (!ce_) return cc->EvalSub(y, target);  // d(0.5 |y - t|^2)/dy
  const CrossEntropy& ce = *ce_;
  Ctxt z;
  if (ce.o.subtract_max) {
    // Max tree over the window: after it slot 0 holds the largest logit (the
    // padding holds lo); keep slot 0 and spread it back over the window.
    z = add_plain(ctx, mul_plain(ctx, y, ce.mask), ce.pad);
    const double range = 2.0 * std::max(std::fabs(ce.o.lo), std::fabs(ce.o.hi));
    Ctxt m = z;
    for (int t = 1; t < ce.window; t *= 2) {
      Ctxt r = ctx.rotate(m, t);
      Ctxt sel = ctx.compare(m, r, range);  // 1 where m < r
      m = cc->EvalAdd(m, cc->EvalMult(sel, cc->EvalSub(r, m)));
    }
    z = cc->EvalSub(z, window_broadcast(ctx, mul_plain(ctx, m, ce.first), ce.window));
  } else {
    z = add_plain(ctx, ce.mask_input ? mul_plain(ctx, y, ce.mask) : y, ce.pad);
  }
  std::vector<double> c = ce.exp_c;
  Ctxt e = mul_plain(ctx, cc->EvalChebyshevSeries(z, c, ce.exp_lo, ce.exp_hi), ce.mask);
  // Sum of every slot, in every slot (not allsum(): that adds in place).
  Ctxt S = e;
  for (uint32_t k = 1; k < ctx.slots(); k *= 2) S = cc->EvalAdd(S, ctx.rotate(S, static_cast<int32_t>(k)));
  c = ce.inv_c;
  Ctxt r = cc->EvalChebyshevSeries(S, c, ce.sum_lo, ce.sum_hi);
  for (uint32_t i = 0; i < ce.o.newton; ++i) r = cc->EvalMult(r, cc->EvalSub(2.0, cc->EvalMult(S, r)));
  return cc->EvalSub(cc->EvalMult(e, r), target);  // softmax(y) - t
}

void Trainer::update(const FHEContext& ctx, double lr) {
  if (!batch_) throw std::logic_error("train: no accumulated gradients");
  const double step = -lr / static_cast<double>(batch_);
  for_each_weighted(nodes_, [&](Node& n) {
    if (n.trainable) update_linear(ctx, n, step);
  });
  if (ctx.options().bootstrapping) {
    std::map<uint32_t, std::vector<Ctxt*>> by_window;
    for_each_weighted(nodes_, [&](Node& n) {
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
    os << "  " << n.layer->type() << ": " << n.offs.size() << (n.kind == Node::kNorm ? " gamma, " : " diagonals, ");
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
  return os.str();
}

// ------------------------------------------------------------- train_plain

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
  for (size_t s = 0; s < xs.size(); ++s) {
    std::vector<double> x = xs[s];
    for (size_t i = 0; i < first; ++i) x = layers[i]->forward_plain(x);
    PlainTrace tr;
    std::vector<double> y = plain_forward(layers, first, x, tr);
    if (y.size() != ts[s].size()) throw std::invalid_argument("train_plain: target size mismatch");
    if (ce) y = ce->softmax(y);  // d(cross-entropy)/dy = softmax(y) - t
    for (size_t i = 0; i < y.size(); ++i) y[i] -= ts[s][i];
    plain_backward(layers, first, train, tr, y, grads);
  }
  const double step = lr / static_cast<double>(xs.size());
  for (auto& kv : grads) {
    std::vector<double> p = kv.first->params();
    for (size_t i = 0; i < p.size(); ++i) p[i] -= step * kv.second[i];
    kv.first->set_params(p);
  }
  sync_attention(layers);
}

}  // namespace fheml
