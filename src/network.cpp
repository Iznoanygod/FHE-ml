#include "fheml/network.h"

#include "fheml/transformer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fheml {

namespace {

std::vector<double> read_vec(std::istream& is, size_t n, const std::string& what) {
  std::vector<double> v(n);
  for (size_t i = 0; i < n; ++i)
    if (!(is >> v[i])) throw std::runtime_error("model file: truncated " + what);
  return v;
}

uint32_t next_pow2(size_t n) {
  uint32_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

}  // namespace

std::shared_ptr<Layer> read_layer(const std::string& tok, std::istream& is) {
  std::shared_ptr<Layer> layer;
  if (tok == "conv2d") {
    int ic, oc, kh, kw, s, p, hb;
    is >> ic >> oc >> kh >> kw >> s >> p >> hb;
    auto w = read_vec(is, static_cast<size_t>(oc) * ic * kh * kw, "conv2d weights");
    auto b = hb ? read_vec(is, oc, "conv2d bias") : std::vector<double>{};
    layer = std::make_shared<Conv2d>(ic, oc, kh, kw, s, p, std::move(w), std::move(b));
  } else if (tok == "dense") {
    int i, o, hb;
    is >> i >> o >> hb;
    auto w = read_vec(is, static_cast<size_t>(i) * o, "dense weights");
    auto b = hb ? read_vec(is, o, "dense bias") : std::vector<double>{};
    layer = std::make_shared<Dense>(i, o, std::move(w), std::move(b));
  } else if (tok == "avgpool2d") {
    int k, s;
    is >> k >> s;
    layer = std::make_shared<AvgPool2d>(k, s);
  } else if (tok == "maxpool2d") {
    int k, s;
    double range;
    is >> k >> s >> range;
    layer = std::make_shared<MaxPool2d>(k, s, range);
  } else if (tok == "affine") {
    size_t c;
    is >> c;
    auto sc = read_vec(is, c, "affine scale");
    auto sh = read_vec(is, c, "affine shift");
    layer = std::make_shared<ChannelAffine>(std::move(sc), std::move(sh));
  } else if (tok == "square") {
    layer = std::make_shared<Square>();
  } else if (tok == "poly") {
    size_t n;
    is >> n;
    layer = std::make_shared<PolyActivation>(read_vec(is, n, "poly coefficients"));
  } else if (tok == "chebyshev") {
    std::string fn;
    double a, b;
    uint32_t d;
    is >> fn >> a >> b >> d;
    double ea = a, eb = b, out = 1.0;
    // Optional "map EA EB OUT"; otherwise the next token starts a new layer.
    std::string next;
    auto pos = is.tellg();
    if (is >> next && next == "map") {
      is >> ea >> eb >> out;
    } else {
      is.clear();
      is.seekg(pos);
    }
    layer = std::make_shared<ChebyshevActivation>(fn, a, b, d, ea, eb, out);
  } else if (tok == "scale") {
    size_t c;
    is >> c;
    layer = std::make_shared<Scale>(read_vec(is, c, "scale factors"));
  } else if (tok == "bootstrap") {
    layer = std::make_shared<Bootstrap>();
  } else if (tok == "tokendense") {
    int n, i, o;
    size_t nb;
    is >> n >> i >> o >> nb;
    auto w = read_vec(is, static_cast<size_t>(i) * o, "tokendense weights");
    auto b = read_vec(is, nb, "tokendense bias");
    layer = std::make_shared<TokenDense>(n, i, o, std::move(w), std::move(b));
  } else if (tok == "tokenmean") {
    int n, d;
    is >> n >> d;
    layer = std::make_shared<TokenMean>(n, d);
  } else if (tok == "attention") {
    int n, d, hb, heads = 1;
    is >> n >> d >> hb;
    // Optional "heads H" (multi-head); single-head files have none.
    std::string next;
    auto pos = is.tellg();
    if (is >> next && next == "heads") {
      is >> heads;
    } else {
      is.clear();
      is.seekg(pos);
    }
    if (heads < 1) throw std::runtime_error("model file: bad attention head count");
    // Optional "softmax LO HI EXP_DEG INV_DEG NEWTON SUBTRACT_MAX SUM_LO SUM_HI".
    bool softmax = false;
    SoftmaxOptions sm;
    pos = is.tellg();
    if (is >> next && next == "softmax") {
      int subtract_max;
      is >> sm.lo >> sm.hi >> sm.exp_degree >> sm.inv_degree >> sm.newton >> subtract_max >> sm.sum_lo >> sm.sum_hi;
      sm.subtract_max = subtract_max != 0;
      softmax = true;
    } else {
      is.clear();
      is.seekg(pos);
    }
    std::vector<double> w[4], b[4];
    for (auto& wi : w) wi = read_vec(is, static_cast<size_t>(d) * d, "attention weights");
    if (hb)
      for (auto& bi : b) bi = read_vec(is, d, "attention bias");
    auto score = read_vec(is, 3u * heads, "attention score polynomial");
    auto att = std::make_shared<Attention>(n, d, w[0], w[1], w[2], w[3], b[0], b[1], b[2], b[3], score, heads);
    if (softmax) att->set_softmax(sm);
    layer = att;
  } else if (tok == "layernorm") {
    int n, d;
    double eps, vmin, vmax;
    uint32_t degree, newton;
    is >> n >> d >> eps >> vmin >> vmax >> degree >> newton;
    auto g = read_vec(is, d, "layernorm gamma");
    auto b = read_vec(is, d, "layernorm beta");
    layer = std::make_shared<LayerNorm>(n, d, std::move(g), std::move(b), eps, vmin, vmax, degree, newton);
  } else if (tok == "residual") {
    size_t k;
    is >> k;
    std::vector<std::shared_ptr<Layer>> body;
    std::string t;
    for (size_t i = 0; i < k && is >> t; ++i) body.push_back(read_layer(t, is));
    if (body.size() != k) throw std::runtime_error("model file: truncated residual block");
    layer = std::make_shared<Residual>(std::move(body));
  } else {
    throw std::runtime_error("model file: unknown layer '" + tok + "'");
  }
  if (!is) throw std::runtime_error("model file: bad parameters for layer '" + tok + "'");
  return layer;
}

Network Network::load(const std::string& path) {
  std::ifstream is(path);
  if (!is) throw std::runtime_error("cannot open model file " + path);
  std::string tok;
  int version = 0;
  if (!(is >> tok >> version) || tok != "fheml" || version != 1) throw std::runtime_error("not an fheml v1 model file");
  Shape in;
  if (!(is >> tok >> in.c >> in.h >> in.w) || tok != "input") throw std::runtime_error("model file: missing input");
  Network net(in);
  while (is >> tok && tok != "end") net.add(read_layer(tok, is));
  return net;
}

void Network::save(const std::string& path) const {
  std::ofstream os(path);
  if (!os) throw std::runtime_error("cannot write model file " + path);
  os << "fheml 1\ninput " << input_.c << " " << input_.h << " " << input_.w << "\n";
  for (const auto& l : layers_) l->save(os);
  os << "end\n";
}

Network& Network::add(std::shared_ptr<Layer> layer) {
  Shape in = output_shape();
  Layout lin = layers_.empty() ? Layout::compact(input_) : layers_.back()->out_layout();
  layer->configure(in);
  layer->configure_layout(lin, gapped_);
  layers_.push_back(std::move(layer));
  slots_ = 0;
  return *this;
}

Shape Network::configure() {
  Shape s = input_;
  Layout lay = Layout::compact(input_);
  for (auto& l : layers_) {
    s = l->configure(s);
    lay = l->configure_layout(lay, gapped_);
  }
  slots_ = 0;
  return s;
}

void Network::fuse_linear() {
  std::vector<std::shared_ptr<Layer>> out;
  for (size_t i = 0; i < layers_.size();) {
    if (!layers_[i]->is_linear() || i + 1 >= layers_.size() || !layers_[i + 1]->is_linear()) {
      out.push_back(std::move(layers_[i++]));
      continue;
    }
    // y = M_k (... (M_1 x + b_1) ...) + b_k
    auto* first = static_cast<LinearLayer*>(layers_[i].get());
    Shape in = first->in_shape();
    SparseMatrix m = first->matrix();
    std::vector<double> b = first->bias();
    b.resize(m.rows(), 0.0);
    std::string what = first->type();
    Shape outs = first->out_shape();
    size_t j = i + 1;
    for (; j < layers_.size() && layers_[j]->is_linear(); ++j) {
      auto* l = static_cast<LinearLayer*>(layers_[j].get());
      SparseMatrix mj = l->matrix();
      std::vector<double> bj = l->bias();
      std::vector<double> nb = mj.multiply(b);
      for (size_t k = 0; k < bj.size(); ++k) nb[k] += bj[k];
      m = mj.multiply(m);
      b = std::move(nb);
      what += "+" + l->type();
      outs = l->out_shape();
    }
    out.emplace_back(new FusedLinear(in, outs, std::move(m), std::move(b), what));
    i = j;
  }
  layers_ = std::move(out);
  configure();
}

void Network::set_cache_plaintexts(bool on) {
  for (auto& l : layers_)
    if (l->is_linear()) static_cast<LinearLayer*>(l.get())->set_cache_plaintexts(on);
}

void Network::encrypt_weights(const FHEContext& ctx, bool drop_plaintext) {
  if (!slots_) throw std::logic_error("compile() the network first");
  forward(ctx, ctx.encrypt(std::vector<double>(input_.size(), 0.0), input_level(ctx)));
  for (auto& l : layers_) l->encrypt_weights(ctx, drop_plaintext);
}

// ---------------------------------------------------- bootstrap placement

namespace {

// How a scale on a layer's signal can move through it (see place_bootstraps).
enum class Fold {
  kAbsorb,  // the layer can take a scale into its parameters for free
  kPass,    // linear without bias: the scale passes through unchanged
  kSquare,  // the scale passes through squared
  kBlock    // neither: a Scale layer is needed
};

Fold fold_kind(const Layer& l) {
  if (dynamic_cast<const Square*>(&l)) return Fold::kSquare;
  if (dynamic_cast<const PolyActivation*>(&l) || dynamic_cast<const ChebyshevActivation*>(&l) ||
      dynamic_cast<const Scale*>(&l) || dynamic_cast<const FusedLinear*>(&l) || dynamic_cast<const LayerNorm*>(&l))
    return Fold::kAbsorb;
  if (l.is_bootstrap()) return Fold::kPass;
  if (auto* lin = dynamic_cast<const LinearLayer*>(&l)) {
    if (lin->trainable()) return Fold::kAbsorb;
    return lin->bias().empty() ? Fold::kPass : Fold::kBlock;
  }
  return Fold::kBlock;
}

// Can the output of layers[i] be scaled for free (folding back through
// passing layers until one absorbs)? boot_before marks planned bootstraps,
// which stop the fold like existing ones.
bool can_scale_output(const std::vector<std::shared_ptr<Layer>>& layers, size_t i, const std::vector<char>& boot_before) {
  for (size_t k = i + 1; k-- > 0;) {
    switch (fold_kind(*layers[k])) {
      case Fold::kAbsorb: return true;
      case Fold::kPass:
      case Fold::kSquare:
        if (layers[k]->is_bootstrap() || boot_before[k]) return false;
        continue;
      case Fold::kBlock: return false;
    }
  }
  return false;
}

// Multiplies what the layer outputs by f (f > 0 when squares are involved).
void scale_output(std::shared_ptr<Layer>& l, double f) {
  if (auto* p = dynamic_cast<PolyActivation*>(l.get())) return p->rescale(1.0, f);
  if (auto* c = dynamic_cast<ChebyshevActivation*>(l.get())) return c->rescale(1.0, f);
  if (auto* sc = dynamic_cast<Scale*>(l.get())) return sc->multiply(f);
  if (auto* fu = dynamic_cast<FusedLinear*>(l.get())) return fu->scale(f, f);
  if (auto* ln = dynamic_cast<LayerNorm*>(l.get())) return ln->scale_output(f);
  auto* lin = dynamic_cast<LinearLayer*>(l.get());
  if (!lin || !lin->trainable()) throw std::logic_error("place_bootstraps: cannot scale " + l->type());
  std::vector<double> p = lin->params();
  for (double& x : p) x *= f;
  lin->set_params(p);
}

void scale_output_chain(std::vector<std::shared_ptr<Layer>>& layers, double f) {
  for (size_t k = layers.size(); k-- > 0;) {
    switch (fold_kind(*layers[k])) {
      case Fold::kAbsorb: return scale_output(layers[k], f);
      case Fold::kSquare: f = std::sqrt(f); continue;
      case Fold::kPass:
        if (layers[k]->is_bootstrap()) throw std::logic_error("place_bootstraps: fold reached a bootstrap");
        continue;
      case Fold::kBlock: throw std::logic_error("place_bootstraps: fold blocked by " + layers[k]->type());
    }
  }
  throw std::logic_error("place_bootstraps: nothing to fold into");
}

// The layer's input arrives multiplied by c; changes the layer so that its
// output is unchanged. Only for absorbing layers.
void absorb_input(std::shared_ptr<Layer>& l, double c) {
  if (auto* p = dynamic_cast<PolyActivation*>(l.get())) return p->rescale(1.0 / c, 1.0);
  if (auto* ch = dynamic_cast<ChebyshevActivation*>(l.get())) return ch->rescale(1.0 / c, 1.0);
  if (auto* sc = dynamic_cast<Scale*>(l.get())) return sc->multiply(1.0 / c);
  if (auto* fu = dynamic_cast<FusedLinear*>(l.get())) return fu->scale(1.0 / c, 1.0);
  if (auto* ln = dynamic_cast<LayerNorm*>(l.get())) return ln->absorb_input_scale(c);
  auto* lin = dynamic_cast<LinearLayer*>(l.get());
  if (!lin || !lin->trainable()) throw std::logic_error("place_bootstraps: cannot rescale " + l->type());
  std::vector<double> p = lin->params();
  std::vector<char> is_bias(p.size(), 0);
  for (int b : lin->bias_refs()) is_bias[static_cast<size_t>(b)] = 1;
  for (size_t i = 0; i < p.size(); ++i)
    if (!is_bias[i]) p[i] /= c;
  lin->set_params(p);
}

std::shared_ptr<Layer> make_scale(const Shape& s, double f) {
  return std::make_shared<Scale>(std::vector<double>(static_cast<size_t>(s.c), f));
}

}  // namespace

size_t Network::place_bootstraps(uint32_t budget, const std::vector<std::vector<double>>& calibration,
                                 double headroom) {
  if (budget == 0) throw std::invalid_argument("place_bootstraps: budget must be at least one level");
  if (!(headroom > 0)) throw std::invalid_argument("place_bootstraps: headroom must be positive");
  configure();
  const size_t n = layers_.size();
  for (const auto& l : layers_)
    if (l->depth() > budget)
      throw std::invalid_argument("place_bootstraps: layer " + l->describe() + " alone needs " +
                                  std::to_string(l->depth()) + " levels, more than the budget");

  // Plaintext ranges: range[i] = max |output of layer i - 1| (range[0]: input).
  std::vector<double> range(n + 1, 0.0);
  for (const auto& x0 : calibration) {
    std::vector<double> x = x0;
    for (size_t i = 0; i <= n; ++i) {
      for (double v : x) range[i] = std::max(range[i], std::fabs(v));
      if (i < n) x = layers_[i]->forward_plain(x);
    }
  }

  // Plan: walk the layers with the running depth d, tracking whether a
  // bootstrap's scale is still being carried (not yet absorbed). A bootstrap
  // goes before layer i when i would overflow the budget. If the scale
  // entering it cannot be folded back, a Scale layer (one level) precedes
  // it; when that does not fit either, the bootstrap moves one layer earlier.
  std::vector<char> boot_before(n + 1, 0), pre_scale(n + 1, 0);
  std::vector<uint32_t> d_at(n + 1, 0);
  std::vector<char> carry_at(n + 1, 0);
  std::vector<Fold> kind(n);
  for (size_t i = 0; i < n; ++i) kind[i] = fold_kind(*layers_[i]);
  uint32_t d = 0;
  bool carrying = false;
  size_t i = 0;
  size_t forced = n + 1;  // a position the back-off decided on
  while (i <= n) {
    d_at[i] = d;
    carry_at[i] = carrying;
    const uint32_t dep = i < n ? layers_[i]->depth() : 0;
    const Fold k = i < n ? kind[i] : Fold::kBlock;  // the output must come out unscaled
    const bool is_boot = i < n && layers_[i]->is_bootstrap();
    uint32_t extra = (carrying && k == Fold::kBlock) ? 1 : 0;  // Scale layer restoring the carry
    const bool overflow = !is_boot && d + extra + dep > budget && (i < n || carrying);
    if (overflow || forced == i) {
      forced = n + 1;
      const bool fold = i > 0 && can_scale_output(layers_, i - 1, boot_before);
      const uint32_t pre = fold ? 0 : 1;
      if (d + pre > budget) {
        // Back off: the latest earlier position in this stretch where the
        // bootstrap (and its Scale layer, if any) still fits.
        size_t j = i;
        for (size_t p = i; p-- > 0 && j == i;) {
          if (layers_[p]->is_bootstrap() || boot_before[p]) break;  // start of the stretch
          const uint32_t pre_p = (p > 0 && can_scale_output(layers_, p - 1, boot_before)) ? 0 : 1;
          if (p > 0 && d_at[p] + pre_p <= budget) j = p;
        }
        if (j == i)
          throw std::invalid_argument("place_bootstraps: the layers up to " + layers_[std::min(i, n - 1)]->describe() +
                                      " do not fit in " + std::to_string(budget) + " levels with their scaling");
        std::fill(boot_before.begin() + static_cast<long>(j), boot_before.end(), 0);
        std::fill(pre_scale.begin() + static_cast<long>(j), pre_scale.end(), 0);
        d = d_at[j];
        carrying = carry_at[j];
        i = j;
        forced = j;
        continue;
      }
      boot_before[i] = 1;
      pre_scale[i] = !fold;
      d = 0;
      carrying = true;
      extra = (k == Fold::kBlock) ? 1 : 0;
    }
    if (is_boot) {
      d = 0;
    } else {
      d += extra + dep;
      if (k == Fold::kAbsorb || k == Fold::kBlock) carrying = false;
    }
    ++i;
  }

  // Apply the plan.
  std::vector<std::shared_ptr<Layer>> out;
  double carry = 1.0;  // factor currently on the signal relative to the calibration run
  size_t inserted = 0;
  for (size_t j = 0; j <= n; ++j) {
    const Shape shape = j == 0 ? input_ : layers_[j - 1]->out_shape();
    if (boot_before[j]) {
      // Values already within [-1, 1] need no scaling.
      const double m = range[j] * std::fabs(carry);
      const double s = m > 1.0 ? headroom * m : 1.0;
      if (pre_scale[j])
        out.push_back(make_scale(shape, 1.0 / s));
      else
        scale_output_chain(out, 1.0 / s);
      out.push_back(std::make_shared<Bootstrap>());
      carry /= s;
      ++inserted;
    }
    if (j == n) {
      if (carry != 1.0) out.push_back(make_scale(shape, 1.0 / carry));
      break;
    }
    std::shared_ptr<Layer> l = layers_[j];
    if (carry != 1.0) {
      switch (kind[j]) {
        case Fold::kAbsorb:
          absorb_input(l, carry);
          carry = 1.0;
          break;
        case Fold::kSquare: carry *= carry; break;
        case Fold::kPass: break;
        case Fold::kBlock:
          out.push_back(make_scale(shape, 1.0 / carry));
          carry = 1.0;
          break;
      }
    }
    out.push_back(l);
  }
  layers_ = std::move(out);
  configure();
  return inserted;
}

namespace {

size_t fold_intervals(std::vector<std::shared_ptr<Layer>>& layers) {
  size_t folded = 0;
  for (size_t i = 0; i < layers.size(); ++i) {
    if (auto* r = dynamic_cast<Residual*>(layers[i].get())) {
      // The body is a separate chain; its first layer's input is the block input.
      std::vector<std::shared_ptr<Layer>> body = r->body();
      size_t f = fold_intervals(body);
      if (f) {
        layers[i] = std::make_shared<Residual>(std::move(body));
        folded += f;
      }
      continue;
    }
    auto* c = dynamic_cast<ChebyshevActivation*>(layers[i].get());
    if (!c || (c->a() == -1.0 && c->b() == 1.0) || i == 0) continue;
    // t = alpha x + beta maps [a, b] onto [-1, 1].
    const double alpha = 2.0 / (c->b() - c->a()), beta = -(c->a() + c->b()) / (c->b() - c->a());
    auto* lin = dynamic_cast<LinearLayer*>(layers[i - 1].get());
    if (lin && lin->trainable() && !lin->bias_refs().empty()) {
      std::vector<double> p = lin->params();
      std::vector<char> is_bias(p.size(), 0);
      for (int b : lin->bias_refs()) is_bias[static_cast<size_t>(b)] = 1;
      for (size_t k = 0; k < p.size(); ++k) p[k] = p[k] * alpha + (is_bias[k] ? beta : 0.0);
      lin->set_params(p);
    } else if (beta == 0.0) {
      std::vector<std::shared_ptr<Layer>> prefix(layers.begin(), layers.begin() + static_cast<long>(i));
      if (!can_scale_output(prefix, i - 1, std::vector<char>(i, 0))) continue;
      scale_output_chain(prefix, alpha);
      std::copy(prefix.begin(), prefix.end(), layers.begin());
    } else {
      continue;
    }
    c->set_evaluation_interval(-1.0, 1.0);
    ++folded;
  }
  return folded;
}

}  // namespace

size_t Network::fold_activation_intervals() {
  size_t n = fold_intervals(layers_);
  configure();
  return n;
}

uint32_t Network::max_bootstrap_budget(ContextOptions base, uint32_t max_ring) {
  base.bootstrapping = true;
  base.ring_dim = 0;
  base.slots = std::min(base.slots, max_ring / 2);
  uint32_t best = 0;
  for (uint32_t b = 1; b < 64; ++b) {
    base.mult_depth = b + 1;  // as context_options() sizes it
    if (FHEContext::ring_dim_for(base) > max_ring) break;
    best = b;
  }
  return best;
}

uint32_t Network::required_slots() const {
  size_t n = input_.size();
  for (const auto& l : layers_) n = std::max(n, l->min_slots());
  return next_pow2(n);
}

void Network::compile(uint32_t slots) {
  configure();
  slots_ = slots ? slots : required_slots();
  if (slots_ < required_slots()) throw std::invalid_argument("slot count too small for this network");
  for (auto& l : layers_) l->compile(slots_);
}

uint32_t Network::depth() const {
  uint32_t d = 0;
  for (const auto& l : layers_) d += l->depth();
  return d;
}

uint32_t Network::depth_between_bootstraps() const {
  uint32_t best = 0, cur = 0;
  for (const auto& l : layers_) {
    if (l->is_bootstrap()) {
      best = std::max(best, cur);
      cur = 0;
    } else {
      cur += l->depth();
    }
  }
  return std::max(best, cur);
}

uint32_t Network::depth_before_first_bootstrap() const {
  uint32_t d = 0;
  for (const auto& l : layers_) {
    if (l->is_bootstrap()) break;
    d += l->depth();
  }
  return d;
}

uint32_t Network::input_level(const FHEContext& ctx) const {
  if (!uses_bootstrapping()) return 0;
  // One spare level: EvalBootstrap first rescales a pending product.
  const uint32_t need = depth_before_first_bootstrap() + 1;
  return ctx.total_depth() > need ? ctx.total_depth() - need : 0;
}

bool Network::uses_scheme_switching() const {
  std::function<bool(const std::vector<std::shared_ptr<Layer>>&)> any = [&](const auto& layers) {
    for (const auto& l : layers) {
      if (l->uses_scheme_switching()) return true;
      if (auto* r = dynamic_cast<const Residual*>(l.get()); r && any(r->body())) return true;
    }
    return false;
  };
  return any(layers_);
}

bool Network::uses_bootstrapping() const {
  return std::any_of(layers_.begin(), layers_.end(), [](const std::shared_ptr<Layer>& l) { return l->is_bootstrap(); });
}

namespace {

void collect_rotations(const std::vector<std::shared_ptr<Layer>>& layers, bool ladder, std::set<int32_t>& s) {
  for (const auto& l : layers) {
    if (auto* r = dynamic_cast<const Residual*>(l.get())) {
      collect_rotations(r->body(), ladder, s);
      continue;
    }
    if (ladder && l->is_linear()) continue;
    for (int32_t k : l->rotation_indices()) s.insert(k);
  }
}

}  // namespace

std::vector<int32_t> Network::rotation_indices(bool ladder) const {
  if (!slots_) throw std::logic_error("compile() the network first");
  std::set<int32_t> s;
  collect_rotations(layers_, ladder, s);
  return std::vector<int32_t>(s.begin(), s.end());
}

ContextOptions Network::context_options(bool secure) const {
  if (!slots_) throw std::logic_error("compile() the network first");
  ContextOptions o;
  o.slots = slots_;
  o.secure = secure;
  o.bootstrapping = uses_bootstrapping();
  o.scheme_switching = uses_scheme_switching();
  // +1: EvalBootstrap needs one level left when its input arrives.
  o.mult_depth = o.bootstrapping ? depth_between_bootstraps() + 1 : depth();
  if (o.bootstrapping) {
    // Bootstrapping precision needs a larger scaling factor; the first
    // modulus must stay within OpenFHE's correction factor of it.
    o.scale_mod_bits = 50;
    o.first_mod_bits = 57;  // outputs must stay below 2^7 in magnitude
  }
  return o;
}

std::vector<double> Network::forward_plain(const std::vector<double>& x) const {
  if (x.size() != input_.size()) throw std::invalid_argument("input size does not match network input");
  std::vector<double> v = x;
  for (const auto& l : layers_) v = l->forward_plain(v);
  return v;
}

Ctxt Network::forward(const FHEContext& ctx, const Ctxt& x, const LayerHook& hook) const {
  if (!slots_) throw std::logic_error("compile() the network first");
  if (ctx.slots() != slots_) throw std::invalid_argument("context slot count differs from compiled network");
  Ctxt c = x;
  for (size_t i = 0; i < layers_.size(); ++i) {
    auto t0 = std::chrono::steady_clock::now();
    c = layers_[i]->forward(ctx, c);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (hook) hook(i, *layers_[i], c, secs);
  }
  return c;
}

std::vector<double> Network::decrypt_layer(const FHEContext& ctx, const Ctxt& y, size_t i) const {
  const Layer& l = *layers_.at(i);
  auto v = ctx.decrypt(y, l.out_layout().span(l.out_shape()));
  std::vector<double> out(l.out_shape().size());
  for (size_t k = 0; k < out.size(); ++k) out[k] = v[l.out_layout().slot(l.out_shape(), k)];
  return out;
}

std::vector<double> Network::decrypt_output(const FHEContext& ctx, const Ctxt& y) const {
  if (layers_.empty()) return ctx.decrypt(y, input_.size());
  return decrypt_layer(ctx, y, layers_.size() - 1);
}

std::string Network::summary() const {
  std::ostringstream os;
  os << "input " << input_.str() << "\n";
  for (size_t i = 0; i < layers_.size(); ++i)
    os << "  [" << i << "] " << layers_[i]->describe() << "  -> " << layers_[i]->out_shape().str() << "  (depth "
       << layers_[i]->depth() << ")\n";
  os << "total depth " << depth();
  if (uses_bootstrapping()) os << ", max between bootstraps " << depth_between_bootstraps();
  if (slots_) os << ", slots " << slots_;
  return os.str();
}

}  // namespace fheml
