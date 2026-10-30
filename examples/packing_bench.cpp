// Compares the three packings of strided conv/pool outputs (gapped, compact,
// multiplexed) on the same networks: slots needed, diagonals and rotations
// of every linear layer and, with --encrypted, the time of each linear layer
// under encryption at the ring dimension its packing needs.
//
//   packing_bench [--encrypted] [--net NAME]
//
// Networks: mnist_cnn and mnist_deepcnn (models/), and "cifar", a
// ResNet-20-shaped plain conv stack on 3x32x32 (16/32/64 channels, two
// stride-2 stages, global average pooling; random weights, no skips).
// Encrypted timings use insecure toy parameters (ring = 2 * slots) with only
// the layer's own rotation keys, one layer at a time, so they compare the
// packings rather than give deployable numbers.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "fheml/fheml.h"

using namespace fheml;

namespace {

std::mt19937 g_rng(11);

std::vector<double> rand_vec(size_t n, double scale) {
  std::uniform_real_distribution<double> u(-scale, scale);
  std::vector<double> v(n);
  for (double& x : v) x = u(g_rng);
  return v;
}

void conv(Network& n, int ci, int co, int stride) {
  n.emplace<Conv2d>(ci, co, 3, 3, stride, 1, rand_vec(static_cast<size_t>(ci) * co * 9, 1.0 / std::sqrt(9.0 * ci)),
                    rand_vec(co, 0.1));
  n.emplace<Square>();
}

Network cifar() {
  Network n(Shape{3, 32, 32});
  conv(n, 3, 16, 1);
  conv(n, 16, 16, 1);
  conv(n, 16, 32, 2);
  conv(n, 32, 32, 1);
  conv(n, 32, 64, 2);
  conv(n, 64, 64, 1);
  n.emplace<AvgPool2d>(8, 8);
  n.emplace<Dense>(64, 10, rand_vec(640, 0.1), rand_vec(10, 0.1));
  return n;
}

Network make(const std::string& name) {
  if (name == "cifar") return cifar();
  for (const char* dir : {"models/", "../models/"}) {
    try {
      return Network::load(std::string(dir) + name + ".fheml");
    } catch (const std::exception&) {
    }
  }
  throw std::runtime_error("cannot find model " + name);
}

const char* packing_name(Packing p) {
  return p == Packing::kGapped ? "gapped" : p == Packing::kCompact ? "compact" : "multiplexed";
}

double seconds_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// Runs one compiled linear layer on an encrypted random input laid out like
// its real input; returns seconds and the max error against the clear plan.
double time_layer(const LinearLayer& l, uint32_t slots, double* err) {
  ContextOptions o;
  o.secure = false;
  o.slots = slots;
  o.ring_dim = std::max<uint32_t>(2048, 2 * slots);
  o.mult_depth = 2;
  o.scale_mod_bits = 40;
  FHEContext ctx(o);
  ctx.keygen(l.rotation_indices());
  std::vector<double> x(slots, 0.0);
  auto xin = rand_vec(l.in_shape().size(), 1.0);
  for (size_t i = 0; i < xin.size(); ++i) x[l.in_layout().slot(l.in_shape(), i)] = xin[i];
  Ctxt c = ctx.encrypt(x);
  auto t = std::chrono::steady_clock::now();
  Ctxt y = l.forward(ctx, c);
  double s = seconds_since(t);
  auto got = ctx.decrypt(y, slots);
  auto ref = l.forward_plain(xin);
  *err = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    size_t sl = l.out_layout().is_compact(l.out_shape()) ? i : l.out_layout().slot(l.out_shape(), i);
    *err = std::max(*err, std::fabs(got[sl] - ref[i]));
  }
  return s;
}

void bench(const std::string& name, bool encrypted) {
  std::printf("== %s\n", name.c_str());
  std::printf("%-12s %6s %6s %8s %8s %8s %9s\n", "packing", "slots", "ring", "diags", "rots", "keys", encrypted ? "lin. s" : "");
  for (Packing p : {Packing::kGapped, Packing::kCompact, Packing::kMultiplexed}) {
    Network n = make(name);
    n.set_packing(p);
    n.compile();
    size_t diags = 0, rots = 0;
    double secs = 0, worst = 0;
    std::vector<std::string> rows;
    for (const auto& l : n.layers()) {
      if (!l->is_linear()) continue;
      const auto* ll = static_cast<const LinearLayer*>(l.get());
      const auto& t = ll->transform();
      size_t r = t.rotation_indices().size();
      diags += t.num_diagonals();
      rots += r;
      char buf[256];
      std::snprintf(buf, sizeof(buf), "    %-60s rots %3zu", ll->describe().c_str(), r);
      std::string row = buf;
      if (encrypted) {
        double err = 0;
        double s = time_layer(*ll, n.slots(), &err);
        secs += s;
        worst = std::max(worst, err);
        std::snprintf(buf, sizeof(buf), "  %7.2fs err %.1e", s, err);
        row += buf;
      }
      rows.push_back(row);
    }
    std::set<int32_t> keys;
    for (int32_t k : n.rotation_indices()) keys.insert(k);
    char tail[64] = "";
    if (encrypted) std::snprintf(tail, sizeof(tail), "%9.2f", secs);
    std::printf("%-12s %6u %6u %8zu %8zu %8zu %s\n", packing_name(p), n.slots(), std::max<uint32_t>(2048, 2 * n.slots()),
                diags, rots, keys.size(), tail);
    for (const auto& r : rows) std::printf("%s\n", r.c_str());
    if (encrypted && worst > 1e-3) std::printf("    WARNING: max error %.2e\n", worst);
  }
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  bool encrypted = false;
  std::vector<std::string> nets = {"mnist_cnn", "mnist_deepcnn", "cifar"};
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--encrypted") == 0) encrypted = true;
    else if (std::strcmp(argv[i], "--net") == 0 && i + 1 < argc) nets = {argv[++i]};
  }
  for (const auto& n : nets) bench(n, encrypted);
  return 0;
}
