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
void check_encrypted(const std::string& name, Network& net, double tol, double input_range = 1.0) {
  net.compile();
  auto opts = net.context_options(false);
  FHEContext ctx = make_ctx(opts.mult_depth, opts.slots, net.rotation_indices(), opts.bootstrapping);
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
  if (!quick) {
    // Bootstrapping in the middle of a network (insecure toy ring).
    Network n(Shape{1, 8, 8});
    n.emplace<Conv2d>(1, 2, 3, 3, 1, 1, rand_vec(18, -0.2, 0.2), rand_vec(2, -0.1, 0.1));
    n.emplace<Square>();
    n.emplace<Bootstrap>();
    n.emplace<Dense>(128, 8, rand_vec(1024, -0.1, 0.1), rand_vec(8, -0.1, 0.1));
    n.emplace<ChebyshevActivation>("relu", -2, 2, 15);
    check_encrypted("bootstrap mid-network", n, 1e-3);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
  cleartext_tests();
  encrypted_tests(quick);
  std::printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "PASSED", g_failures, g_failures == 1 ? "" : "s");
  return g_failures ? 1 : 0;
}
