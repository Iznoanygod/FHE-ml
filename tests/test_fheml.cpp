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
    n.emplace<Dense>(32, 4, rand_vec(128), rand_vec(4));
    n.emplace<ChebyshevActivation>("relu", -4, 4, 15);
    const char* path = "test_model.fheml";
    n.save(path);
    Network m = Network::load(path);
    std::remove(path);
    auto x = rand_vec(36);
    check_err("save/load round trip", max_err(n.forward_plain(x), m.forward_plain(x), 4), 1e-12);
  }
}

FHEContext make_ctx(uint32_t depth, uint32_t slots, const std::vector<int32_t>& rot, bool boot = false) {
  ContextOptions o;
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
  FHEContext ctx = make_ctx(opts.mult_depth, opts.slots, net.rotation_indices(), opts.bootstrapping);
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
    // Attention alone; n = 5 is not a power of two on purpose.
    const int n = 5, d = 8;
    Network net(Shape{n, 1, d});
    net.emplace<Attention>(n, d, rand_vec(d * d, -0.4, 0.4), rand_vec(d * d, -0.4, 0.4), rand_vec(d * d, -0.4, 0.4),
                           rand_vec(d * d, -0.4, 0.4), rand_vec(d), rand_vec(d), rand_vec(d), rand_vec(d),
                           std::vector<double>{0.2, 0.1, 0.05});
    check_encrypted("attention n=5 d=8", net, 1e-4);
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

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
  cleartext_tests();
  backend_tests();
  encrypted_tests(quick);
  transformer_tests();
  std::printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "PASSED", g_failures, g_failures == 1 ? "" : "s");
  return g_failures ? 1 : 0;
}
