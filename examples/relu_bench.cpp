// ReLU under CKKS: Chebyshev approximations against the exact
// scheme-switching ReLU (ExactReLU), both measured against max(x, 0).
//
//   relu_bench [--slots N] [--range R] [--insecure] [--degrees 15,31,63] [--reps K]
//
// Every variant gets its own context with just the levels it needs, and the
// same random inputs in [-R, R] (plus a few near 0 and at the ends).
// Reports setup + keygen time, latency per call (one ciphertext, all slots),
// levels consumed, the max error against the true ReLU and the process's
// peak memory so far (so run the expensive variant last, as here).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <sys/resource.h>

#include "fheml/fheml.h"

using namespace fheml;

namespace {

// Peak resident set of the process so far, in GB (Linux reports KB).
double peak_rss_gb() {
  struct rusage u;
  getrusage(RUSAGE_SELF, &u);
  return u.ru_maxrss / 1048576.0;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

uint32_t used_levels(const Ctxt& c) { return c->GetLevel() + (c->GetNoiseScaleDeg() > 1 ? 1 : 0); }

void run(const std::string& name, Layer& layer, ContextOptions o, const std::vector<double>& x, int reps) {
  o.mult_depth = layer.depth();
  o.scheme_switching = layer.uses_scheme_switching();
  auto t0 = std::chrono::steady_clock::now();
  FHEContext ctx(o);
  ctx.keygen({});
  const double t_setup = seconds_since(t0);
  Ctxt c = ctx.encrypt(x);
  double t = 0, err = 0, err_far = 0;
  uint32_t used = 0;
  for (int r = 0; r < reps; ++r) {
    t0 = std::chrono::steady_clock::now();
    Ctxt y = layer.forward(ctx, c);
    t += seconds_since(t0);
    used = used_levels(y);
    auto got = ctx.decrypt(y, x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      const double e = std::fabs(got[i] - std::max(x[i], 0.0));
      err = std::max(err, e);
      if (std::fabs(x[i]) > 0.1) err_far = std::max(err_far, e);
    }
  }
  std::printf("%-22s ring 2^%-2d  setup+keygen %8.2fs  per call %9.3fs  levels %2u  max err %.2e (|x|>0.1: %.2e)  peak RSS %.1f GB\n",
              name.c_str(), static_cast<int>(std::log2(ctx.ring_dim())), t_setup, t / reps, used, err, err_far,
              peak_rss_gb());
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  uint32_t slots = 256;
  double range = 4.0;
  bool secure = true;
  int reps = 1;
  std::vector<uint32_t> degrees = {15, 31, 63};
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--slots" && i + 1 < argc) slots = std::strtoul(argv[++i], nullptr, 10);
    else if (a == "--range" && i + 1 < argc) range = std::strtod(argv[++i], nullptr);
    else if (a == "--insecure") secure = false;
    else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
    else if (a == "--degrees" && i + 1 < argc) {
      degrees.clear();
      std::stringstream ss(argv[++i]);
      for (std::string d; std::getline(ss, d, ',');) degrees.push_back(std::strtoul(d.c_str(), nullptr, 10));
    } else {
      std::fprintf(stderr, "usage: %s [--slots N] [--range R] [--insecure] [--degrees 15,31,63] [--reps K]\n", argv[0]);
      return 2;
    }
  }
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-range, range);
  std::vector<double> x(slots);
  for (double& v : x) v = u(rng);
  const double near[] = {0.0, 1e-3, -1e-3, 1e-2, -1e-2, 0.999 * range, -0.999 * range};
  for (size_t i = 0; i < std::min<size_t>(slots, 7); ++i) x[i] = near[i];

  ContextOptions o;
  o.slots = slots;
  o.secure = secure;
  std::printf("ReLU on %u slots, inputs in [-%g, %g], %s\n", slots, range, range,
              secure ? "128-bit security" : "INSECURE toy parameters");
  for (uint32_t d : degrees) {
    ChebyshevActivation cheb("relu", -range, range, d);
    run("chebyshev degree " + std::to_string(d), cheb, o, x, reps);
  }
  ExactReLU exact(range);
  run("exactrelu (FHEW sign)", exact, o, x, reps);
  return 0;
}
