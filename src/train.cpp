#include "fheml/train.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include "fheml/transformer.h"

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
  if (auto* lin = dynamic_cast<const LinearLayer*>(&l)) return lin->trainable();
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
  enum Kind { kLinear, kSquare, kPoly, kCheb, kBoot, kResidual } kind = kLinear;
  Layer* layer = nullptr;
  bool trainable = false;

  // kLinear: y = sum_i D_i (*) rot_{offs[i]}(x) + B
  LinearLayer* lin = nullptr;
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

  std::vector<std::unique_ptr<Node>> body;  // kResidual
};

namespace {

using Node = Trainer::Node;

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

uint32_t sim_forward(Sim& s, const std::vector<std::unique_ptr<Node>>& nodes, uint32_t x, SimTrace& tr) {
  tr.in.assign(nodes.size(), 0);
  tr.body.assign(nodes.size(), {});
  for (size_t i = 0; i < nodes.size(); ++i) {
    const Node& n = *nodes[i];
    tr.in[i] = x;
    switch (n.kind) {
      case Node::kLinear: x = s.see(std::max(x, n.trainable ? s.w : x) + 1); break;
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
      case Node::kLinear:
        if (n.trainable) {
          uint32_t grad = s.see(std::max(g, z) + 1);
          uint32_t upd = s.see(grad + (n.groups.empty() ? 1 : 2));
          s.new_w = std::max(s.new_w, std::max(s.w, upd));
        }
        g = s.see(std::max(g, n.trainable ? s.w : g) + 1);
        break;
      case Node::kSquare:
      case Node::kPoly:
      case Node::kCheb:
        if ((n.kind == Node::kSquare) || n.dcoef.size() > 1)
          g = s.see(std::max(g, s.see(z + derivative_depth(n))) + 1);
        else
          g = s.see(g + 1);
        break;
      case Node::kBoot: g = std::min(g, 0u); break;
      case Node::kResidual: g = std::max(sim_backward(s, n.body, tr.body[k], g), g); break;
    }
  }
  return g;
}

void collect_rotations(const std::vector<std::unique_ptr<Node>>& nodes, uint32_t slots, std::set<int32_t>& r,
                       bool& allsum) {
  for (const auto& n : nodes) {
    if (n->kind == Node::kResidual) collect_rotations(n->body, slots, r, allsum);
    if (n->kind != Node::kLinear) continue;
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

template <typename F>
void for_each_linear(std::vector<std::unique_ptr<Node>>& nodes, F f) {
  for (auto& n : nodes) {
    if (n->kind == Node::kResidual) for_each_linear(n->body, f);
    if (n->kind == Node::kLinear) f(*n);
  }
}

template <typename F>
void for_each_linear(const std::vector<std::unique_ptr<Node>>& nodes, F f) {
  for (const auto& n : nodes) {
    if (n->kind == Node::kResidual) for_each_linear(n->body, f);
    if (n->kind == Node::kLinear) f(*n);
  }
}

// --------------------------------------------------------- encrypted passes

Ctxt linear_forward(const FHEContext& ctx, Node& n, const Ctxt& x) {
  const CC& cc = ctx.cc();
  n.in = x;
  const std::vector<Ctxt> rot = backend::rotate_hoisted(cc, x, n.offs, ctx.parallel());
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

Ctxt forward_nodes(const FHEContext& ctx, std::vector<std::unique_ptr<Node>>& nodes, Ctxt x) {
  for (auto& np : nodes) {
    Node& n = *np;
    switch (n.kind) {
      case Node::kLinear: x = linear_forward(ctx, n, x); break;
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
      case Node::kLinear: {
        if (n.trainable) {
          // dL/dD_i = g (*) rot_i(x), summed over the batch.
          const std::vector<Ctxt> rot = backend::rotate_hoisted(cc, n.in, n.offs, ctx.parallel());
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
          for (size_t i = 0; i < n.offs.size(); ++i) add_to(ctx, n.gD[i], cc->EvalMult(g, rot[i]));
          if (n.B) add_to(ctx, n.gB, g);
        }
        if (need) {
          std::vector<Ctxt> terms(n.offs.size());
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
          for (size_t i = 0; i < n.offs.size(); ++i) {
            Ctxt t = n.trainable ? cc->EvalMult(n.D[i], g) : mul_plain(ctx, g, n.diag[i]);
            terms[i] = n.offs[i] ? cc->EvalRotate(t, normalize_rotation(-static_cast<int64_t>(n.offs[i]), ctx.slots()))
                                 : t;
          }
          g = sum(ctx, std::move(terms));
        }
        n.in = nullptr;
        break;
      }
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

std::vector<double> scaled(const std::vector<double>& mask, double s) {
  std::vector<double> v(mask);
  for (double& x : v) x *= s;
  return v;
}

// Sum of all slots, in every slot.
Ctxt allsum(const FHEContext& ctx, Ctxt c) {
  for (uint32_t k = 1; k < ctx.slots(); k *= 2) ctx.cc()->EvalAddInPlace(c, ctx.cc()->EvalRotate(c, static_cast<int32_t>(k)));
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
          Ctxt r = cc->EvalRotate(t[i + step], normalize_rotation(-static_cast<int64_t>(step * window), ctx.slots()));
          t[i] = t[i] ? cc->EvalAdd(t[i], r) : r;
        }
    t[0] = ctx.bootstrap(t[0]);
    for (size_t step = width / 2; step >= 1; step /= 2)
      for (size_t i = 0; i + step < width; i += 2 * step)
        if (i + step < m) t[i + step] = cc->EvalRotate(t[i], static_cast<int32_t>(step * window));
#pragma omp parallel for if (ctx.parallel())
    for (size_t j = 0; j < m; ++j) *cts[start + j] = mul_plain(ctx, t[j], mask);
  }
}

// --------------------------------------------------------- plaintext reference

struct PlainTrace {
  std::vector<std::vector<double>> in;
  std::vector<PlainTrace> body;
};

using Grads = std::map<LinearLayer*, std::vector<double>>;

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
      std::vector<double> gx(x.size(), 0.0);
      if (train[k] && lin->trainable()) {
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
      g = std::move(gx);
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

Trainer::Trainer(Network& net, std::vector<size_t> frozen) : net_(net) {
  if (!net.slots()) throw std::logic_error("train: compile() the network first");
  slots_ = net.slots();
  first_ = first_trainable(net, frozen);
  const auto& layers = net.layers();
  for (size_t i = first_; i < layers.size(); ++i) nodes_.push_back(build(layers[i].get(), !contains(frozen, i), slots_));
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
  sim_backward(s, nodes_, tr, y);
  return std::max(s.max, s.new_w);
}

std::vector<int32_t> Trainer::rotation_indices() const {
  std::set<int32_t> r;
  const auto& layers = net_.layers();
  for (size_t i = 0; i < first_; ++i)
    for (int32_t k : layers[i]->rotation_indices()) r.insert(k);
  bool all = false;
  collect_rotations(nodes_, slots_, r, all);
  if (all)
    for (uint32_t k = 1; k < slots_; k *= 2) r.insert(static_cast<int32_t>(k));
  return std::vector<int32_t>(r.begin(), r.end());
}

ContextOptions Trainer::context_options(bool secure, bool bootstrapping, uint32_t updates) const {
  ContextOptions o;
  o.slots = slots_;
  o.secure = secure;
  o.bootstrapping = bootstrapping;
  if (bootstrapping) {
    // +1: weights restart one level below a fresh bootstrap (unpacking mask);
    // +1: EvalBootstrap needs one level left.
    o.mult_depth = step_depth() + 2;
    o.scale_mod_bits = 55;  // bootstrapping precision (about 1e-2 at 50 bits with full slots)
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
      sim_backward(s, nodes_, tr, std::max(y, 0u));
      top = std::max({top, s.max, s.new_w});
      w = s.new_w;
    }
    // One more forward pass to use the trained weights.
    Sim s{w};
    SimTrace tr;
    top = std::max(top, sim_forward(s, nodes_, prefix, tr));
    o.mult_depth = top;
  }
  return o;
}

void Trainer::encrypt_weights(const FHEContext& ctx) {
  if (ctx.slots() != slots_) throw std::invalid_argument("train: context slot count differs from the network");
  base_level_ = ctx.options().bootstrapping ? ctx.bootstrap_depth() + 1 : 0;
  for_each_linear(nodes_, [&](Node& n) {
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
  Ctxt g = ctx.cc()->EvalSub(y, target);  // d(0.5 |y - t|^2)/dy
  backward_nodes(ctx, nodes_, g, false);
  ++batch_;
  return y;
}

void Trainer::update(const FHEContext& ctx, double lr) {
  if (!batch_) throw std::logic_error("train: no accumulated gradients");
  const double step = -lr / static_cast<double>(batch_);
  for_each_linear(nodes_, [&](Node& n) {
    if (n.trainable) update_linear(ctx, n, step);
  });
  if (ctx.options().bootstrapping) {
    std::map<uint32_t, std::vector<Ctxt*>> by_window;
    for_each_linear(nodes_, [&](Node& n) {
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
  for_each_linear(nodes_, [&](Node& n) {
    if (!n.trainable) return;
    std::vector<double> p = n.lin->params(), total(p.size(), 0.0), count(p.size(), 0.0);
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
    n.lin->set_params(p);
  });
  net_.compile(slots_);
}

size_t Trainer::weight_ciphertexts() const {
  size_t k = 0;
  for_each_linear(nodes_, [&](const Node& n) {
    if (n.trainable) k += n.offs.size() + (n.bias.empty() ? 0 : 1);
  });
  return k;
}

uint32_t Trainer::weight_level() const {
  uint32_t l = 0;
  for_each_linear(nodes_, [&](const Node& n) {
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
  for_each_linear(nodes_, [&](const Node& n) {
    os << "  " << n.lin->type() << ": " << n.offs.size() << " diagonals, ";
    if (n.trainable)
      os << n.groups.size() << " shared weights";
    else
      os << "frozen";
    os << "\n";
  });
  os << "weight ciphertexts " << weight_ciphertexts() << ", step depth " << step_depth();
  return os.str();
}

// ------------------------------------------------------------- train_plain

void train_plain(Network& net, const std::vector<std::vector<double>>& xs, const std::vector<std::vector<double>>& ts,
                 double lr, const std::vector<size_t>& frozen) {
  if (xs.size() != ts.size() || xs.empty()) throw std::invalid_argument("train_plain: need matching, non-empty batches");
  const auto& layers = net.layers();
  const size_t first = first_trainable(net, frozen);
  std::vector<char> train(layers.size());
  for (size_t i = 0; i < layers.size(); ++i) train[i] = !contains(frozen, i);
  Grads grads;
  for (size_t s = 0; s < xs.size(); ++s) {
    std::vector<double> x = xs[s];
    for (size_t i = 0; i < first; ++i) x = layers[i]->forward_plain(x);
    PlainTrace tr;
    std::vector<double> y = plain_forward(layers, first, x, tr);
    if (y.size() != ts[s].size()) throw std::invalid_argument("train_plain: target size mismatch");
    for (size_t i = 0; i < y.size(); ++i) y[i] -= ts[s][i];
    plain_backward(layers, first, train, tr, y, grads);
  }
  const double step = lr / static_cast<double>(xs.size());
  for (auto& kv : grads) {
    std::vector<double> p = kv.first->params();
    for (size_t i = 0; i < p.size(); ++i) p[i] -= step * kv.second[i];
    kv.first->set_params(p);
  }
}

}  // namespace fheml
