// Self-contained tests (no framework). Run: ./test_fheml [--quick]
//
// Cleartext tests check that the BSGS plans compute exactly M x; encrypted
// tests compare every layer type against the plaintext reference under small,
// INSECURE parameters (HEStd_NotSet) so they run quickly.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "fheml/fheml.h"

using namespace fheml;

namespace {

int g_failures = 0;
std::mt19937 g_rng(1234);

std::vector<double> rand_vec(size_t n, double lo = -1.0, double hi = 1.0) {
  std::uniform_real_distribution<double> d(lo, hi);
  std::vector<double> v(n);
  for (double& x : v) x = d(g_rng);
  return v;
}

double max_err(const std::vector<double>& a, const std::vector<double>& b, size_t n) {
  double e = 0;
  for (size_t i = 0; i < n; ++i) e = std::max(e, std::fabs(a[i] - b[i]));
  return e;
}

void check(bool ok, const std::string& name, const std::string& detail = "") {
  std::printf("[%s] %s %s\n", ok ? " OK " : "FAIL", name.c_str(), detail.c_str());
  if (!ok) ++g_failures;
}

void check_err(const std::string& name, double err, double tol) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "(max err %.2e, tol %.0e)", err, tol);
  check(err <= tol, name, buf);
}

// Runs a single-layer network through the plan simulator and checks M x.
void check_plan(const std::string& name, Network& net) {
  net.compile();
  auto* l = static_cast<const LinearLayer*>(net.layers()[0].get());
  auto x = rand_vec(net.input_shape().size());
  auto ref = l->matrix().multiply(x);
  auto slots = l->transform().eval_slots(x);
  std::vector<double> got(ref.size());
  for (size_t i = 0; i < got.size(); ++i) got[i] = slots[l->out_layout().slot(l->out_shape(), i)];
  check_err("plan " + name + " " + l->describe(), max_err(ref, got, ref.size()), 1e-9);
  if (!l->transform().fold_block()) {
    // Unused slots must stay zero (activations of garbage slots could
    // otherwise blow up). Folded layers leave bounded partial sums there.
    std::vector<char> used(slots.size(), 0);
    for (size_t i = 0; i < got.size(); ++i) used[l->out_layout().slot(l->out_shape(), i)] = 1;
    double tail = 0;
    for (size_t i = 0; i < slots.size(); ++i)
      if (!used[i]) tail = std::max(tail, std::fabs(slots[i]));
    check(tail < 1e-12, "plan " + name + " unused slots zero");
  }
}

void cleartext_tests() {
  {
    Network n(Shape{1, 8, 8});
    n.emplace<Conv2d>(1, 3, 3, 3, 1, 1, rand_vec(27), rand_vec(3));
    check_plan("conv same", n);
  }
  {
    Network n(Shape{3, 9, 7});
    n.emplace<Conv2d>(3, 4, 3, 3, 2, 1, rand_vec(108), std::vector<double>{});
    check_plan("conv stride2", n);
  }
  {
    Network n(Shape{2, 10, 10});
    n.emplace<Conv2d>(2, 2, 5, 5, 1, 0, rand_vec(100), std::vector<double>{});
    check_plan("conv valid 5x5", n);
  }
  {
    Network n(Shape{50, 1, 1});
    n.emplace<Dense>(50, 13, rand_vec(650), rand_vec(13));
    check_plan("dense", n);
  }
  {
    Network n(Shape{600, 1, 1});
    n.emplace<Dense>(600, 10, rand_vec(6000), rand_vec(10));
    check_plan("dense folded", n);
  }
  {
    // Strided conv, gapped output, then a dense layer reading the gaps.
    Network n(Shape{1, 12, 12});
    // (no biases: eval_slots covers the matrix part only)
    n.emplace<Conv2d>(1, 3, 3, 3, 2, 1, rand_vec(27), std::vector<double>{});
    n.emplace<Dense>(108, 7, rand_vec(756), std::vector<double>{});
    n.compile();
    check(!n.layers()[0]->out_layout().is_compact(n.layers()[0]->out_shape()), "strided conv output is gapped");
    auto x = rand_vec(144);
    std::vector<double> slots(n.slots(), 0.0);
    for (size_t i = 0; i < x.size(); ++i) slots[i] = x[i];
    for (const auto& l : n.layers()) slots = static_cast<const LinearLayer*>(l.get())->transform().eval_slots(slots);
    check_err("plan gapped conv -> dense", max_err(n.forward_plain(x), slots, 7), 1e-9);
  }
  {
    Network n(Shape{3, 8, 8});
    n.emplace<AvgPool2d>(2, 2);
    check_plan("avgpool", n);
  }
  {
    // Fusion: conv -> avgpool -> affine == one matrix.
    Network n(Shape{2, 8, 8});
    n.emplace<Conv2d>(2, 3, 3, 3, 1, 1, rand_vec(54), rand_vec(3));
    n.emplace<AvgPool2d>(2, 2);
    n.emplace<ChannelAffine>(rand_vec(3), rand_vec(3));
    auto x = rand_vec(128);
    auto ref = n.forward_plain(x);
    n.fuse_linear();
    check(n.layers().size() == 1, "fuse collapses 3 linear layers");
    n.compile();
    check_err("fused forward", max_err(ref, n.forward_plain(x), ref.size()), 1e-12);
  }
  {
    // Save / load round trip.
    Network n(Shape{1, 6, 6});
    n.emplace<Conv2d>(1, 2, 3, 3, 1, 0, rand_vec(18), rand_vec(2));
    n.emplace<PolyActivation>(std::vector<double>{0.1, 0.5, 0.25});
    n.emplace<MaxPool2d>(2, 1, 3.0);
    n.emplace<Dense>(18, 4, rand_vec(72), rand_vec(4));
    n.emplace<ChebyshevActivation>("relu", -4, 4, 15);
    const char* path = "test_model.fheml";
    n.save(path);
    Network m = Network::load(path);
    std::remove(path);
    auto x = rand_vec(36);
    check_err("save/load round trip", max_err(n.forward_plain(x), m.forward_plain(x), 4), 1e-12);
  }
  {
    // Max pooling in the clear against a direct computation.
    Network n(Shape{2, 5, 5});
    n.emplace<MaxPool2d>(3, 2);
    auto x = rand_vec(50);
    auto y = n.forward_plain(x);
    double err = 0;
    for (int c = 0; c < 2; ++c)
      for (int oy = 0; oy < 2; ++oy)
        for (int ox = 0; ox < 2; ++ox) {
          double m = -1e9;
          for (int dy = 0; dy < 3; ++dy)
            for (int dx = 0; dx < 3; ++dx) m = std::max(m, x[(c * 5 + oy * 2 + dy) * 5 + ox * 2 + dx]);
          err = std::max(err, std::fabs(m - y[(c * 2 + oy) * 2 + ox]));
        }
    check_err("maxpool plain 3x3 stride 2", err, 0.0);
    check(n.output_shape().size() == 8 && n.depth() == FHEContext::compare_depth() + 5, "maxpool shape and depth",
          "(depth " + std::to_string(n.depth()) + ")");
    n.compile();
    check(n.uses_scheme_switching() && n.context_options(false).scheme_switching, "maxpool enables scheme switching");
  }
}

FHEContext make_ctx(uint32_t depth, uint32_t slots, const std::vector<int32_t>& rot, bool boot = false,
                    bool switching = false) {
  ContextOptions o;
  o.scheme_switching = switching;
  o.mult_depth = depth;
  o.slots = slots;
  o.secure = false;
  o.ring_dim = boot ? 4096 : 2 * slots < 2048 ? 2048 : 2 * slots;
  o.bootstrapping = boot;
  o.scale_mod_bits = boot ? 50 : 40;
  FHEContext ctx(o);
  ctx.keygen(rot);
  return ctx;
}

// Encrypts x, runs the network, compares with the plaintext reference and the
// number of levels actually consumed against Network::depth().
void check_encrypted(const std::string& name, Network& net, double tol, double input_range = 1.0,
                     bool encrypted_weights = false) {
  net.compile();
  auto opts = net.context_options(false);
  FHEContext ctx =
      make_ctx(opts.mult_depth, opts.slots, net.rotation_indices(), opts.bootstrapping, opts.scheme_switching);
  if (encrypted_weights) net.encrypt_weights(ctx, /*drop_plaintext=*/true);  // proves no clear weight is used
  auto x = rand_vec(net.input_shape().size(), -input_range, input_range);
  Ctxt c = ctx.encrypt(x, net.input_level(ctx));
  uint32_t lvl0 = c->GetLevel();
  Ctxt y = net.forward(ctx, c);
  auto got = net.decrypt_output(ctx, y);
  auto ref = net.forward_plain(x);
  check_err("encrypted " + name, max_err(ref, got, ref.size()), tol);
  if (!opts.bootstrapping) {
    // FLEXIBLEAUTO rescales lazily: a pending rescale counts as one level.
    uint32_t used = y->GetLevel() - lvl0 + (y->GetNoiseScaleDeg() > 1 ? 1 : 0);
    check(used <= net.depth(), "depth " + name,
          "(used " + std::to_string(used) + ", planned " + std::to_string(net.depth()) + ")");
  }
}

void measure_poly_depth() {
  // Confirms poly_depth() matches what OpenFHE consumes.
  for (uint32_t deg : {2u, 3u, 5u, 7u, 13u, 27u, 59u, 119u}) {
    Network n(Shape{16, 1, 1});
    n.emplace<ChebyshevActivation>("silu", -4, 4, deg);
    check_encrypted("chebyshev silu degree " + std::to_string(deg), n, 1e-3, 4.0);
  }
}

// The GPU backend (FIDESlib) has no EvalPoly and evaluates power-basis
// polynomials as Chebyshev series on [-1, 1]. Check that rewrite on any build:
// same values (also off [-1, 1]) and no more levels than poly_depth().
void backend_tests() {
  std::vector<std::vector<double>> polys = {{0.1, 0.5, 0.25, -0.05},
                                            {0.0, 1.0, 0.0, -0.1, 0.0, 0.01},
                                            {0.3, -0.2, 0.1, 0.05, -0.02, 0.01, 0.004, -0.001, 0.0005, 0.0, 0.0, 0.0, 0.0, 1e-5}};
  auto power = [](const std::vector<double>& c, double x) {
    double acc = 0.0;
    for (size_t k = c.size(); k-- > 0;) acc = acc * x + c[k];
    return acc;
  };
  for (const auto& c : polys) {
    std::vector<double> t = backend::power_to_chebyshev(c);
    double err = 0.0;
    for (double x : {-2.0, -1.0, -0.3, 0.0, 0.7, 1.5}) {
      double t0 = 1.0, t1 = x, acc = t[0] / 2 + (t.size() > 1 ? t[1] * x : 0.0);
      for (size_t k = 2; k < t.size(); ++k) {
        double tk = 2 * x * t1 - t0;
        acc += t[k] * tk;
        t0 = t1;
        t1 = tk;
      }
      err = std::max(err, std::fabs(acc - power(c, x)));
    }
    check_err("power->chebyshev degree " + std::to_string(c.size() - 1), err, 1e-12);

    const uint32_t depth = poly_depth(static_cast<uint32_t>(c.size() - 1));
    FHEContext ctx = make_ctx(depth, 16, {});
    auto x = rand_vec(16, -1.5, 1.5);
    Ctxt in = ctx.encrypt(x);
    Ctxt y = ctx.cc()->EvalChebyshevSeries(in, t, -1.0, 1.0);
    auto got = ctx.decrypt(y, 16);
    std::vector<double> ref(16);
    for (size_t i = 0; i < 16; ++i) ref[i] = power(c, x[i]);
    const std::string name = "chebyshev-basis poly degree " + std::to_string(c.size() - 1);
    check_err(name, max_err(ref, got, 16), 1e-4);
    uint32_t used = y->GetLevel() - in->GetLevel() + (y->GetNoiseScaleDeg() > 1 ? 1 : 0);
    check(used <= depth, "depth " + name, "(used " + std::to_string(used) + ", planned " + std::to_string(depth) + ")");
  }
}

// A one-block transformer: token embedding, attention and MLP residuals,
// mean pooling and a classifier head.
Network tiny_transformer(int n, int in, int d) {
  auto attn = std::make_shared<Attention>(n, d, rand_vec(d * d, -0.3, 0.3), rand_vec(d * d, -0.3, 0.3),
                                          rand_vec(d * d, -0.3, 0.3), rand_vec(d * d, -0.3, 0.3), rand_vec(d, -0.1, 0.1),
                                          rand_vec(d, -0.1, 0.1), rand_vec(d, -0.1, 0.1), rand_vec(d, -0.1, 0.1),
                                          std::vector<double>{0.5 / n, 0.5 / n, 0.125 / n});
  std::vector<std::shared_ptr<Layer>> mlp = {
      std::make_shared<TokenDense>(n, d, 2 * d, rand_vec(2 * d * d, -0.3, 0.3), rand_vec(2 * d, -0.1, 0.1)),
      std::make_shared<PolyActivation>(std::vector<double>{0.0, 0.5, 0.25}),
      std::make_shared<TokenDense>(n, 2 * d, d, rand_vec(2 * d * d, -0.3, 0.3), rand_vec(d, -0.1, 0.1))};
  Network net(Shape{n, 1, in});
  net.emplace<TokenDense>(n, in, d, rand_vec(in * d, -0.4, 0.4), rand_vec(n * d, -0.1, 0.1))
      .add(std::make_shared<Residual>(std::vector<std::shared_ptr<Layer>>{attn}))
      .add(std::make_shared<Residual>(mlp))
      .emplace<TokenMean>(n, d)
      .emplace<Dense>(d, 3, rand_vec(3 * d, -0.5, 0.5), rand_vec(3, -0.1, 0.1));
  return net;
}

// CKKS <-> FHEW scheme switching: the raw comparison and max pooling built
// on it (TOY FHEW parameters, one FHEW bootstrap per slot and comparison).
void scheme_switching_tests(bool quick) {
  if (gpu_backend_built()) {
    // FIDESlib has no scheme switching: the context must say so.
    bool threw = false;
    try {
      make_ctx(FHEContext::compare_depth() + 1, 16, {}, false, true);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    check(threw, "scheme switching is rejected on the FIDESlib backend");
    std::printf("  skipping scheme-switching tests (FIDESlib backend)\n");
    return;
  }
  {
    FHEContext ctx = make_ctx(FHEContext::compare_depth() + 1, 16, {}, false, true);
    std::printf("  %s\n", ctx.describe().c_str());
    auto a = rand_vec(16, -2.0, 2.0), b = rand_vec(16, -2.0, 2.0);
    a[0] = b[0] = 0.5;      // tie: either answer is fine
    a[1] = 1.0, b[1] = 1.01;  // close
    a[2] = -1.99, b[2] = 1.99;  // extremes of the range
    Ctxt ca = ctx.encrypt(a), cb = ctx.encrypt(b);
    Ctxt sel = ctx.compare(ca, cb, 2.0);
    auto got = ctx.decrypt(sel, 16);
    double err = 0;
    for (size_t i = 1; i < 16; ++i) err = std::max(err, std::fabs(got[i] - (a[i] < b[i] ? 1.0 : 0.0)));
    check_err("compare a < b", err, 1e-3);
    check(got[0] > -1e-3 && got[0] < 1 + 1e-3, "compare tie is 0 or 1", "(" + std::to_string(got[0]) + ")");
    uint32_t used = sel->GetLevel() + (sel->GetNoiseScaleDeg() > 1 ? 1 : 0);
    check(used == FHEContext::compare_depth(), "compare depth",
          "(used " + std::to_string(used) + ", compare_depth " + std::to_string(FHEContext::compare_depth()) + ")");
    // Inputs deep in the chain still compare: the result is fresh.
    Ctxt deep = ctx.encrypt(a, FHEContext::compare_depth() - 1);
    Ctxt sel2 = ctx.compare(deep, cb, 2.0);
    got = ctx.decrypt(sel2, 16);
    err = 0;
    for (size_t i = 1; i < 16; ++i) err = std::max(err, std::fabs(got[i] - (a[i] < b[i] ? 1.0 : 0.0)));
    check_err("compare from a deep input", err, 1e-3);
    check(sel2->GetLevel() == sel->GetLevel(), "compare result level does not depend on the input level");
  }
  {
    Network n(Shape{2, 4, 4});
    n.emplace<MaxPool2d>(2, 2);
    check_encrypted("maxpool 2x2", n, 1e-3);
  }
  {
    // 3x3 window: 8 comparisons over 4 rounds (one odd carry per round).
    Network n(Shape{1, 5, 5});
    n.emplace<MaxPool2d>(3, 2, 2.0);
    check_encrypted("maxpool 3x3 stride 2", n, 1e-3, 2.0);
  }
  if (quick) return;
  {
    // conv -> maxpool (gapped output) -> relu -> dense: the pooled layout is
    // absorbed by the dense layer, the activation sees bounded garbage slots.
    Network n(Shape{1, 6, 6});
    n.emplace<Conv2d>(1, 2, 3, 3, 1, 1, rand_vec(18, -0.3, 0.3), rand_vec(2, -0.1, 0.1));
    n.emplace<MaxPool2d>(2, 2, 2.0);
    n.emplace<ChebyshevActivation>("relu", -2, 2, 15);
    n.emplace<Dense>(18, 4, rand_vec(72, -0.5, 0.5), rand_vec(4));
    check_encrypted("conv maxpool relu dense", n, 1e-2);
  }
}

void transformer_tests() {
  {
    Network n(Shape{5, 1, 6});
    n.emplace<TokenDense>(5, 6, 8, rand_vec(48), rand_vec(40));
    check_plan("tokendense", n);
    Network m(Shape{5, 1, 8});
    m.emplace<TokenMean>(5, 8);
    check_plan("tokenmean", m);
  }
  {
    Network net = tiny_transformer(4, 5, 8);
    const char* path = "test_transformer.fheml";
    net.save(path);
    Network back = Network::load(path);
    std::remove(path);
    auto x = rand_vec(20);
    check_err("transformer save/load round trip", max_err(net.forward_plain(x), back.forward_plain(x), 3), 1e-12);
  }
  {
    // LayerNorm in the clear against the exact formula, save/load, and the
    // scale folds place_bootstraps relies on.
    const int n = 3, d = 8;
    auto gamma = rand_vec(d, 0.5, 1.5), beta = rand_vec(d, -0.2, 0.2);
    Network net(Shape{n, 1, d});
    net.emplace<LayerNorm>(n, d, gamma, beta, 1e-5, 0.05, 1.2, 31, 1);
    auto x = rand_vec(n * d);
    auto y = net.forward_plain(x);
    std::vector<double> ref(x.size());
    for (int t = 0; t < n; ++t) {
      double mean = 0, var = 0;
      for (int j = 0; j < d; ++j) mean += x[t * d + j] / d;
      for (int j = 0; j < d; ++j) var += (x[t * d + j] - mean) * (x[t * d + j] - mean) / d;
      for (int j = 0; j < d; ++j) ref[t * d + j] = gamma[j] * (x[t * d + j] - mean) / std::sqrt(var + 1e-5) + beta[j];
    }
    check_err("layernorm plain vs exact", max_err(ref, y, ref.size()), 1e-6);
    const char* path = "test_layernorm.fheml";
    net.save(path);
    Network back = Network::load(path);
    std::remove(path);
    check_err("layernorm save/load round trip", max_err(y, back.forward_plain(x), y.size()), 1e-12);
    auto* ln = dynamic_cast<LayerNorm*>(net.layers()[0].get());
    ln->absorb_input_scale(0.25);
    std::vector<double> xs(x);
    for (double& v : xs) v *= 0.25;
    check_err("layernorm absorbs an input scale", max_err(y, net.forward_plain(xs), y.size()), 1e-9);
    ln->scale_output(3.0);
    for (double& v : y) v *= 3.0;
    check_err("layernorm scales its output", max_err(y, net.forward_plain(xs), y.size()), 1e-9);
  }
  {
    // Attention alone; n = 5 is not a power of two on purpose.
    const int n = 5, d = 8;
    Network net(Shape{n, 1, d});
    net.emplace<Attention>(n, d, rand_vec(d * d, -0.4, 0.4), rand_vec(d * d, -0.4, 0.4), rand_vec(d * d, -0.4, 0.4),
                           rand_vec(d * d, -0.4, 0.4), rand_vec(d), rand_vec(d), rand_vec(d), rand_vec(d),
                           std::vector<double>{0.2, 0.1, 0.05});
    check_encrypted("attention n=5 d=8", net, 1e-4);
  }
  {
    // Multi-head attention: 4 heads of width 2, one score polynomial per
    // head. In plaintext it is the sum of the single-head attentions over
    // each head's slice of Wq/Wk/Wv and Wo's columns.
    const int n = 5, d = 8, heads = 4, dh = d / heads;
    std::vector<double> w[4] = {rand_vec(d * d, -0.4, 0.4), rand_vec(d * d, -0.4, 0.4), rand_vec(d * d, -0.4, 0.4),
                                rand_vec(d * d, -0.4, 0.4)};
    std::vector<double> b[4] = {rand_vec(d), rand_vec(d), rand_vec(d), rand_vec(d)};
    std::vector<double> score = rand_vec(3 * heads, 0.0, 0.3);
    Attention mha(n, d, w[0], w[1], w[2], w[3], b[0], b[1], b[2], b[3], score, heads);
    mha.configure(Shape{n, 1, d});
    auto x = rand_vec(n * d);
    std::vector<double> y = mha.forward_plain(x), sum(n * d, 0.0);
    for (int h = 0; h < heads; ++h) {
      std::vector<double> wh[4], bh[4];
      for (int i = 0; i < 4; ++i) {
        wh[i].assign(d * d, 0.0);
        bh[i].assign(d, 0.0);
        for (int o = 0; o < d; ++o)
          for (int c = 0; c < d; ++c) {
            bool keep = i < 3 ? (o / dh == h) : (c / dh == h);
            if (keep) wh[i][o * d + c] = w[i][o * d + c];
          }
        if (i < 3)
          for (int o = h * dh; o < (h + 1) * dh; ++o) bh[i][o] = b[i][o];
      }
      if (h == 0) bh[3] = b[3];
      Attention one(n, d, wh[0], wh[1], wh[2], wh[3], bh[0], bh[1], bh[2], bh[3],
                    std::vector<double>(score.begin() + 3 * h, score.begin() + 3 * h + 3));
      one.configure(Shape{n, 1, d});
      std::vector<double> yh = one.forward_plain(x);
      for (size_t i = 0; i < sum.size(); ++i) sum[i] += yh[i];
    }
    check_err("multi-head attention = sum of per-head attentions", max_err(y, sum, n * d), 1e-12);

    Network net(Shape{n, 1, d});
    net.emplace<Attention>(n, d, w[0], w[1], w[2], w[3], b[0], b[1], b[2], b[3], score, heads);
    const char* path = "test_mha.fheml";
    net.save(path);
    Network back = Network::load(path);
    std::remove(path);
    check_err("multi-head attention save/load round trip",
              max_err(net.forward_plain(x), back.forward_plain(x), n * d), 1e-12);
    check(back.layers()[0]->describe().find("heads=4") != std::string::npos, "loaded attention keeps its heads");
    check_encrypted("multi-head attention n=5 d=8 heads=4", net, 1e-4);
    // Shared score (3 coefficients) with heads > 1, and a padded token stride.
    Network net2(Shape{n, 1, 16});
    net2.emplace<TokenDense>(n, 16, d, rand_vec(16 * d, -0.3, 0.3), rand_vec(d, -0.1, 0.1))
        .emplace<Attention>(n, d, w[0], w[1], w[2], w[3], b[0], b[1], b[2], b[3],
                            std::vector<double>{0.2, 0.1, 0.05}, 2);
    check_encrypted("2-head attention with stride 16", net2, 1e-4);
    bool threw = false;
    try {
      Attention bad(n, d, w[0], w[1], w[2], w[3], b[0], b[1], b[2], b[3], score, 3);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    check(threw, "attention rejects a head count that does not divide d");
  }
  {
    Network net = tiny_transformer(4, 5, 8);
    check_encrypted("tiny transformer", net, 1e-3);
    Network net2 = tiny_transformer(4, 5, 8);
    check_encrypted("enc-weights tiny transformer", net2, 1e-3, 1.0, true);
    // Input tokens padded to the MLP width: every layer keeps token stride 16.
    Network net3 = tiny_transformer(4, 16, 8);
    net3.compile();
    check(net3.layers()[1]->in_layout().cs == 16, "transformer keeps the padded token stride");
    check_encrypted("tiny transformer, strided tokens", net3, 1e-3);
  }
  {
    // The variance interval is calibrated like a Chebyshev interval: here
    // from the per-token variance of d uniform(-1, 1) values (about 1/3).
    const int n = 3, d = 8;
    Network a(Shape{n, 1, d});
    a.emplace<LayerNorm>(n, d, rand_vec(d, 0.5, 1.5), rand_vec(d, -0.2, 0.2), 1e-5, 0.05, 1.2, 31, 1);
    check_encrypted("layernorm", a, 1e-3);
    // Padded token stride: tokendense keeps the 16-slot stride, the norm
    // must not read the padding. The tokendense outputs have a smaller,
    // input-dependent variance; the test checks what it will be.
    Network b(Shape{n, 1, 16});
    b.emplace<TokenDense>(n, 16, d, rand_vec(16 * d, -0.3, 0.3), rand_vec(d, -0.1, 0.1));
    {
      double lo = 1e9, hi = 0;
      for (int probe = 0; probe < 20; ++probe) {
        auto h = b.forward_plain(rand_vec(n * 16));
        for (int t = 0; t < n; ++t) {
          double mean = 0, var = 0;
          for (int j = 0; j < d; ++j) mean += h[t * d + j] / d;
          for (int j = 0; j < d; ++j) var += (h[t * d + j] - mean) * (h[t * d + j] - mean) / d;
          lo = std::min(lo, var);
          hi = std::max(hi, var);
        }
      }
      b.emplace<LayerNorm>(n, d, rand_vec(d, 0.5, 1.5), rand_vec(d, -0.2, 0.2), 1e-5, 0.25 * lo, 4.0 * hi, 31, 1);
    }
    check_encrypted("layernorm, strided tokens", b, 1e-3);
    // Pre-norm residual MLP block.
    std::vector<std::shared_ptr<Layer>> body = {
        std::make_shared<LayerNorm>(n, d, rand_vec(d, 0.5, 1.5), rand_vec(d, -0.2, 0.2), 1e-5, 0.05, 1.2, 31, 1),
        std::make_shared<TokenDense>(n, d, 2 * d, rand_vec(2 * d * d, -0.3, 0.3), rand_vec(2 * d, -0.1, 0.1)),
        std::make_shared<PolyActivation>(std::vector<double>{0.0, 0.5, 0.25}),
        std::make_shared<TokenDense>(n, 2 * d, d, rand_vec(2 * d * d, -0.3, 0.3), rand_vec(d, -0.1, 0.1))};
    Network c(Shape{n, 1, d});
    c.add(std::make_shared<Residual>(body));
    check_encrypted("pre-norm residual block", c, 1e-3);
  }
}

void encrypted_tests(bool quick) {
  {
    Network n(Shape{2, 8, 8});
    n.emplace<Conv2d>(2, 3, 3, 3, 1, 1, rand_vec(54), rand_vec(3));
    check_encrypted("conv2d 3x3 same", n, 1e-4);
  }
  {
    Network n(Shape{1, 12, 12});
    n.emplace<Conv2d>(1, 4, 5, 5, 2, 0, rand_vec(100), rand_vec(4));
    check_encrypted("conv2d 5x5 stride 2", n, 1e-4);
  }
  {
    Network n(Shape{40, 1, 1});
    n.emplace<Dense>(40, 10, rand_vec(400), rand_vec(10));
    check_encrypted("dense", n, 1e-4);
  }
  {
    Network n(Shape{2, 8, 8});
    n.emplace<AvgPool2d>(2, 2);
    check_encrypted("avgpool", n, 1e-4);
  }
  {
    Network n(Shape{16, 1, 1});
    n.emplace<Square>();
    check_encrypted("square", n, 1e-4);
  }
  {
    Network n(Shape{16, 1, 1});
    n.emplace<PolyActivation>(std::vector<double>{0.1, 0.5, 0.25});
    check_encrypted("poly deg 2", n, 1e-4);
  }
  {
    Network n(Shape{16, 1, 1});
    n.emplace<PolyActivation>(std::vector<double>{0.0, 1.0, 0.0, -0.1, 0.0, 0.01});
    check_encrypted("poly deg 5", n, 1e-4);
  }
  {
    Network n(Shape{16, 1, 1});
    n.emplace<ChebyshevActivation>("relu", -3, 3, 31);
    check_encrypted("chebyshev relu", n, 1e-3, 3.0);
  }
  if (!quick) measure_poly_depth();
  {
    // Small CNN end to end.
    Network n(Shape{1, 10, 10});
    n.emplace<Conv2d>(1, 3, 3, 3, 1, 1, rand_vec(27, -0.5, 0.5), rand_vec(3, -0.1, 0.1));
    n.emplace<PolyActivation>(std::vector<double>{0.0, 0.5, 0.25});
    n.emplace<AvgPool2d>(2, 2);
    n.emplace<Conv2d>(3, 4, 3, 3, 2, 1, rand_vec(108, -0.3, 0.3), std::vector<double>{});
    n.emplace<Square>();
    n.emplace<Dense>(36, 5, rand_vec(180, -0.3, 0.3), rand_vec(5));
    check_encrypted("small cnn", n, 1e-3);
    n.fuse_linear();  // conv+avgpool... only adjacent linear layers merge
    check_encrypted("small cnn fused", n, 1e-3);
  }
  {
    // Encrypted weights: same networks, ciphertext x ciphertext products.
    Network a(Shape{2, 8, 8});
    a.emplace<Conv2d>(2, 3, 3, 3, 2, 1, rand_vec(54), rand_vec(3));
    check_encrypted("enc-weights conv2d stride 2", a, 1e-4, 1.0, true);
    Network b(Shape{300, 1, 1});
    b.emplace<Dense>(300, 10, rand_vec(3000, -0.2, 0.2), rand_vec(10));
    check_encrypted("enc-weights dense folded", b, 1e-4, 1.0, true);
    Network c(Shape{1, 10, 10});
    c.emplace<Conv2d>(1, 3, 3, 3, 2, 1, rand_vec(27, -0.5, 0.5), rand_vec(3, -0.1, 0.1));
    c.emplace<PolyActivation>(std::vector<double>{0.0, 0.5, 0.25});
    c.emplace<Dense>(75, 6, rand_vec(450, -0.3, 0.3), rand_vec(6));
    c.emplace<Square>();
    c.emplace<Dense>(6, 3, rand_vec(18), rand_vec(3));
    check_encrypted("enc-weights small cnn", c, 1e-3, 1.0, true);
  }
  if (!quick) {
    // Bootstrapping in the middle of a network (insecure toy ring).
    Network n(Shape{1, 8, 8});
    n.emplace<Conv2d>(1, 2, 3, 3, 1, 1, rand_vec(18, -0.2, 0.2), rand_vec(2, -0.1, 0.1));
    n.emplace<Square>();
    n.emplace<Bootstrap>();
    n.emplace<Dense>(128, 8, rand_vec(1024, -0.1, 0.1), rand_vec(8, -0.1, 0.1));
    n.emplace<ChebyshevActivation>("relu", -2, 2, 15);
    check_encrypted("bootstrap mid-network", n, 1e-3);
    check_encrypted("enc-weights bootstrap mid-network", n, 1e-3, 1.0, true);
  }
}

// Rotations composed from a ladder of keys (ContextOptions::rotation_window).
void ladder_tests() {
  for (uint32_t window : {1u, 3u}) {
    ContextOptions o;
    o.mult_depth = 1;
    o.slots = 512;
    o.secure = false;
    o.ring_dim = 2048;
    o.rotation_window = window;
    FHEContext ctx(o);
    ctx.keygen({});  // no direct keys: everything is composed
    const size_t expect = FHEContext::ladder_rotations(o.slots, window).size();
    check(ctx.rotation_key_count() == expect,
          "ladder window " + std::to_string(window) + " has " + std::to_string(expect) + " keys",
          "(" + std::to_string(ctx.rotation_key_bytes() / 1024) + " KB)");
    auto x = rand_vec(o.slots);
    Ctxt c = ctx.encrypt(x);
    double err = 0;
    uint32_t cost = 0;
    for (int32_t k : {1, -1, 3, 7, 29, -28, 100, 255, -255, 256, 257, 511, 300}) {
      auto got = ctx.decrypt(ctx.rotate(c, k), o.slots);
      for (uint32_t i = 0; i < o.slots; ++i) err = std::max(err, std::fabs(got[i] - x[(i + k + 4 * o.slots) % o.slots]));
      cost = std::max(cost, ctx.rotation_cost(k));
    }
    check_err("ladder rotations window " + std::to_string(window), err, 1e-6);
    check(cost <= (9 + window - 1) / window, "ladder window " + std::to_string(window) + " cost bound",
          "(max " + std::to_string(cost) + " key switches)");
  }
}

// Automatic bootstrap placement and interval folding.
void placement_tests(bool quick) {
  // A CNN whose activations grow well beyond [-1, 1]: conv, square, pool,
  // dense, chebyshev, dense, poly, dense (depth 1+1+1+1+(3+1)+1+2+1 = 12).
  const std::vector<double> w1 = rand_vec(18, -0.5, 0.5), b1 = rand_vec(2, -0.2, 0.2), w2 = rand_vec(384, -0.5, 0.5),
                            b2 = rand_vec(12, -0.5, 0.5), w3 = rand_vec(96, -0.6, 0.6), b3 = rand_vec(8, -0.3, 0.3),
                            w4 = rand_vec(24, -0.5, 0.5), b4 = rand_vec(3);
  auto make = [&] {
    Network n(Shape{1, 8, 8});
    n.emplace<Conv2d>(1, 2, 3, 3, 1, 1, w1, b1)
        .emplace<Square>()
        .emplace<AvgPool2d>(2, 2)
        .emplace<Dense>(32, 12, w2, b2)
        .emplace<ChebyshevActivation>("relu", -6.0, 6.0, 7)
        .emplace<Dense>(12, 8, w3, b3)
        .emplace<PolyActivation>(std::vector<double>{0.1, 0.5, 0.25})
        .emplace<Dense>(8, 3, w4, b4);
    return n;
  };
  std::vector<std::vector<double>> calib;
  for (int i = 0; i < 8; ++i) calib.push_back(rand_vec(64, -2.0, 2.0));
  Network ref = make();
  ref.compile();
  const uint32_t depth0 = ref.depth();
  double big = 0;
  for (const auto& x : calib)
    for (double v : ref.forward_plain(x)) big = std::max(big, std::fabs(v));
  check(big > 1.0, "placement test network has large activations", "(max output " + std::to_string(big) + ")");

  for (uint32_t budget : {5u, 6u, 8u}) {
    Network n = make();
    size_t k = n.place_bootstraps(budget, calib);
    n.compile();
    std::string tag = "budget " + std::to_string(budget);
    check(k > 0 && n.uses_bootstrapping(), "place_bootstraps " + tag + " inserted " + std::to_string(k));
    check(n.depth_between_bootstraps() <= budget, "place_bootstraps " + tag + " respects the budget",
          "(" + std::to_string(n.depth_between_bootstraps()) + ")");
    // Scaling is folded exactly: the plaintext function is unchanged.
    double err = 0, range = 0;
    for (const auto& x : calib) {
      err = std::max(err, max_err(ref.forward_plain(x), n.forward_plain(x), 3));
      // Every bootstrap sees values within [-1, 1] (calibration inputs).
      std::vector<double> v = x;
      for (const auto& l : n.layers()) {
        if (l->is_bootstrap())
          for (double y : v) range = std::max(range, std::fabs(y));
        v = l->forward_plain(v);
      }
    }
    check_err("place_bootstraps " + tag + " keeps the function", err, 1e-9);
    check(range <= 1.0, "place_bootstraps " + tag + " bootstrap inputs within [-1, 1]", "(max " + std::to_string(range) + ")");
    // Round-trips through the model file (scaled chebyshev, scale layers).
    n.save("/tmp/fheml_placement_test.fheml");
    Network back = Network::load("/tmp/fheml_placement_test.fheml");
    back.compile();
    check_err("place_bootstraps " + tag + " save/load", max_err(ref.forward_plain(calib[0]), back.forward_plain(calib[0]), 3), 1e-9);
    std::printf("  %s\n", n.summary().c_str());
  }
  {
    // One fewer level: the chebyshev interval map goes into the dense layer.
    Network n = make();
    size_t f = n.fold_activation_intervals();
    n.compile();
    check(f == 1 && n.depth() == depth0 - 1, "fold_activation_intervals saves a level",
          "(folded " + std::to_string(f) + ", depth " + std::to_string(n.depth()) + " vs " + std::to_string(depth0) + ")");
    check_err("fold_activation_intervals keeps the function", max_err(ref.forward_plain(calib[1]), n.forward_plain(calib[1]), 3),
              1e-9);
    if (!quick) {
      // Then bootstraps every 6 levels; encrypted on an insecure toy ring,
      // with a composed-rotation ladder, inputs in the calibrated range.
      size_t k = n.place_bootstraps(6, calib);
      n.compile();
      check(k == 2, "place_bootstraps after folding", "(" + std::to_string(k) + " bootstraps)");
      auto opts = n.context_options(false);
      opts.ring_dim = 4096;
      opts.rotation_window = 1;
      FHEContext ctx(opts);
      ctx.keygen(n.rotation_indices(true));
      auto x = rand_vec(64, -2.0, 2.0);
      Ctxt c = ctx.encrypt(x, n.input_level(ctx));
      auto got = n.decrypt_output(ctx, n.forward(ctx, c));
      check_err("encrypted auto-bootstrapped cnn (ladder keys)", max_err(ref.forward_plain(x), got, 3), 2e-2);
      std::printf("  %s\n", ctx.describe().c_str());
    }
  }
  {
    // Residual blocks cannot absorb a scale: Scale layers are inserted, and
    // they must be budgeted.
    const int t = 3, d = 4;
    std::vector<std::vector<double>> w;
    for (int i = 0; i < 4; ++i) {
      w.push_back(rand_vec(32, -0.8, 0.8));
      w.push_back(rand_vec(i % 2 ? d : 8, -0.5, 0.5));
    }
    const std::vector<double> w0 = rand_vec(20, -0.9, 0.9), b0 = rand_vec(t * d, -0.5, 0.5), wd = rand_vec(8, -0.5, 0.5),
                              bd = rand_vec(2);
    auto make_res = [&] {
      Network n(Shape{t, 1, 5});
      n.emplace<TokenDense>(t, 5, d, w0, b0);
      for (int blk = 0; blk < 2; ++blk)
        n.add(std::make_shared<Residual>(std::vector<std::shared_ptr<Layer>>{
            std::make_shared<TokenDense>(t, d, 8, w[4 * blk], w[4 * blk + 1]), std::make_shared<Square>(),
            std::make_shared<TokenDense>(t, 8, d, w[4 * blk + 2], w[4 * blk + 3])}));
      n.emplace<TokenMean>(t, d).emplace<Dense>(d, 2, wd, bd);
      return n;
    };
    std::vector<std::vector<double>> cal;
    for (int i = 0; i < 6; ++i) cal.push_back(rand_vec(15, -2.0, 2.0));
    Network n = make_res(), r2 = make_res();
    // Budget 4 cannot hold a residual block (3) plus the Scale layers around
    // it; the planner must say so instead of looping or overflowing.
    bool threw = false;
    try {
      Network n4 = make_res();
      n4.place_bootstraps(4, cal);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    check(threw, "place_bootstraps rejects a budget the scale layers cannot fit");
    size_t k = n.place_bootstraps(5, cal);
    n.compile();
    size_t scales = 0;
    for (const auto& l : n.layers()) scales += l->type() == "scale";
    check(k >= 1 && scales >= 1, "place_bootstraps residual inserts scale layers",
          "(" + std::to_string(k) + " bootstraps, " + std::to_string(scales) + " scale layers)");
    check(n.depth_between_bootstraps() <= 5, "place_bootstraps residual respects the budget");
    check_err("place_bootstraps residual keeps the function", max_err(r2.forward_plain(cal[0]), n.forward_plain(cal[0]), 2), 1e-9);
    std::printf("  %s\n", n.summary().c_str());
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
  cleartext_tests();
  ladder_tests();
  placement_tests(quick);
  backend_tests();
  encrypted_tests(quick);
  scheme_switching_tests(quick);
  transformer_tests();
  std::printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "PASSED", g_failures, g_failures == 1 ? "" : "s");
  return g_failures ? 1 : 0;
}
