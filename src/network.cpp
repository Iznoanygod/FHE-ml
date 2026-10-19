#include "fheml/network.h"

#include "fheml/transformer.h"

#include <algorithm>
#include <chrono>
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
    layer = std::make_shared<ChebyshevActivation>(fn, a, b, d);
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
    int n, d, hb;
    is >> n >> d >> hb;
    std::vector<double> w[4], b[4];
    for (auto& wi : w) wi = read_vec(is, static_cast<size_t>(d) * d, "attention weights");
    if (hb)
      for (auto& bi : b) bi = read_vec(is, d, "attention bias");
    auto score = read_vec(is, 3, "attention score polynomial");
    layer = std::make_shared<Attention>(n, d, w[0], w[1], w[2], w[3], b[0], b[1], b[2], b[3], score);
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

bool Network::uses_bootstrapping() const {
  return std::any_of(layers_.begin(), layers_.end(), [](const std::shared_ptr<Layer>& l) { return l->is_bootstrap(); });
}

std::vector<int32_t> Network::rotation_indices() const {
  if (!slots_) throw std::logic_error("compile() the network first");
  std::set<int32_t> s;
  for (const auto& l : layers_)
    for (int32_t r : l->rotation_indices()) s.insert(r);
  return std::vector<int32_t>(s.begin(), s.end());
}

ContextOptions Network::context_options(bool secure) const {
  if (!slots_) throw std::logic_error("compile() the network first");
  ContextOptions o;
  o.slots = slots_;
  o.secure = secure;
  o.bootstrapping = uses_bootstrapping();
  o.mult_depth = o.bootstrapping ? depth_between_bootstraps() : depth();
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
