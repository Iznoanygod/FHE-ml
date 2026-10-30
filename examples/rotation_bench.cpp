// Benchmarks FHEContext::rotate_many on the rotation sets real layers ask for
// (the baby steps of the conv_demo CNN's linear layers, attention's token
// offsets), with one key per index and with rotation-key ladders. Three ways
// of evaluating each set:
//   independent  every rotation on its own (rotate(x, k) per k)
//   nearest      the previous rotate_many: each composed rotation starts
//                from the nearest earlier result, no hoisting
//   hoisted      rotate_many: a rotation tree with shared ladder steps, each
//                node decomposed once for all its children
// and checks all three decrypt to the same rotations.
//
//   rotation_bench [--insecure] [--ring N] [--depth D] [--reps R]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "fheml/fheml.h"

using namespace fheml;

namespace {

double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// rotate_many before the rotation tree: with a key per index one hoisted
// decomposition; otherwise cheapest offsets first, each composed from the
// already rotated copy nearest to it, the independent rotations of a wave in
// parallel. Returns the key switches it performs through `ks`.
std::vector<Ctxt> rotate_nearest(const FHEContext& ctx, const Ctxt& x, const std::vector<int32_t>& offs,
                                 size_t* ks) {
  const size_t n = offs.size();
  if (std::all_of(offs.begin(), offs.end(), [&](int32_t k) { return ctx.has_rotation_key(k); })) {
    std::vector<int32_t> idx(n);
    *ks = 0;
    for (size_t i = 0; i < n; ++i) {
      idx[i] = normalize_rotation(offs[i], ctx.slots());
      *ks += idx[i] != 0;
    }
    return backend::rotate_hoisted(ctx.cc(), x, idx, ctx.parallel());
  }
  std::vector<Ctxt> out(n);
  std::vector<size_t> order(n);
  for (size_t i = 0; i < n; ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return ctx.rotation_cost(offs[a]) < ctx.rotation_cost(offs[b]); });
  std::vector<int> source(n, -1), wave(n, 0);
  std::vector<size_t> planned;
  *ks = 0;
  for (size_t i : order) {
    uint32_t best = ctx.rotation_cost(offs[i]);
    for (size_t j : planned) {
      uint32_t c = ctx.rotation_cost(offs[i] - offs[j]);
      if (c < best || (c == best && source[i] >= 0 && wave[j] < wave[static_cast<size_t>(source[i])])) {
        best = c;
        source[i] = static_cast<int>(j);
      }
    }
    *ks += best;
    wave[i] = source[i] < 0 ? 1 : wave[static_cast<size_t>(source[i])] + 1;
    planned.push_back(i);
  }
  const int waves = n ? *std::max_element(wave.begin(), wave.end()) : 0;
  for (int w = 1; w <= waves; ++w) {
    std::vector<size_t> cur;
    for (size_t i = 0; i < n; ++i)
      if (wave[i] == w) cur.push_back(i);
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
    for (size_t t = 0; t < cur.size(); ++t) {
      const size_t i = cur[t];
      const Ctxt& from = source[i] < 0 ? x : out[static_cast<size_t>(source[i])];
      const int32_t by = source[i] < 0 ? offs[i] : offs[i] - offs[static_cast<size_t>(source[i])];
      out[i] = ctx.rotate(from, by);
    }
  }
  return out;
}

struct Set {
  std::string name;
  std::vector<int32_t> offs;
};

}  // namespace

int main(int argc, char** argv) {
  bool secure = true;
  uint32_t ring = 0, depth = 8, reps = 1;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--insecure") secure = false;
    else if (a == "--ring" && i + 1 < argc) ring = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    else if (a == "--depth" && i + 1 < argc) depth = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    else if (a == "--reps" && i + 1 < argc) reps = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    else {
      std::fprintf(stderr, "usage: %s [--insecure] [--ring N] [--depth D] [--reps R]\n", argv[0]);
      return 1;
    }
  }

  // The conv_demo CNN (3x16x16 input): its linear layers' baby steps.
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  auto rand_vec = [&](size_t n, double scale) {
    std::vector<double> v(n);
    for (double& x : v) x = scale * u(rng);
    return v;
  };
  Network net(Shape{3, 16, 16});
  net.emplace<Conv2d>(3, 8, 3, 3, 1, 1, rand_vec(8 * 3 * 9, 0.2), rand_vec(8, 0.1))
      .emplace<AvgPool2d>(2, 2)
      .emplace<ChebyshevActivation>("relu", -2.0, 2.0, 27)
      .emplace<Conv2d>(8, 8, 3, 3, 2, 1, rand_vec(8 * 8 * 9, 0.1), rand_vec(8, 0.1))
      .emplace<Dense>(8 * 4 * 4, 10, rand_vec(10 * 128, 0.1), rand_vec(10, 0.1));
  net.compile();
  std::vector<Set> sets;
  for (const auto& l : net.layers())
    if (l->is_linear()) {
      const auto* ll = static_cast<const LinearLayer*>(l.get());
      sets.push_back({l->describe(), ll->transform().baby_steps()});
    }
  // Attention over 16 tokens with a 64-slot token stride (rotations of K
  // and V by s * stride).
  Set att{"attention 16 tokens, stride 64", {}};
  for (int s = 0; s < 16; ++s) att.offs.push_back(s * 64);
  sets.push_back(att);

  const uint32_t slots = net.context_options(secure).slots;
  std::printf("rotation sets (slots=%u):\n", slots);
  for (const auto& s : sets) std::printf("  %-60s %zu rotations\n", s.name.c_str(), s.offs.size());

  for (uint32_t window : {0u, 1u, 2u, 3u}) {
    ContextOptions o;
    o.slots = slots;
    o.mult_depth = depth;
    o.scale_mod_bits = 50;
    o.secure = secure;
    o.ring_dim = ring;
    o.rotation_window = window;
    FHEContext ctx(o);
    std::vector<int32_t> keys;
    if (!window)
      for (const auto& s : sets) keys.insert(keys.end(), s.offs.begin(), s.offs.end());
    double t0 = now();
    ctx.keygen(keys);
    std::printf("\n%s\nkeygen %.1f s\n", ctx.describe().c_str(), now() - t0);
    std::vector<double> x(slots);
    for (double& v : x) v = u(rng);
    Ctxt c = ctx.encrypt(x);
    {
      // One key switch split into its two halves (sequential calls).
      const CC& cc = ctx.cc();
      const int32_t k = FHEContext::ladder_rotations(slots, 1)[0];
      const uint32_t m = cc->GetCyclotomicOrder();
      if (window || ctx.has_rotation_key(k)) {
        const int n = 10;
        double t_rot = 0, t_dec = 0, t_fast = 0;
        for (int r = 0; r < n; ++r) {
          t0 = now();
          Ctxt a = cc->EvalRotate(c, k);
          t_rot += now() - t0;
          t0 = now();
          auto digits = cc->EvalFastRotationPrecompute(c);
          t_dec += now() - t0;
          t0 = now();
          Ctxt b = cc->EvalFastRotation(c, k, m, digits);
          t_fast += now() - t0;
        }
        std::printf("rotation %.1f ms = decomposition %.1f ms + hoisted rotation %.1f ms\n", 1e3 * t_rot / n,
                    1e3 * t_dec / n, 1e3 * t_fast / n);
      }
    }
    std::printf("  %-34s %22s %22s %30s\n", "", "independent", "nearest", "hoisted tree");
    std::printf("  %-34s %10s %11s %10s %11s %10s %8s %11s\n", "set", "ks", "time", "ks", "time", "ks", "decomp",
                "time");
    for (const auto& s : sets) {
      size_t ks_ind = 0, ks_near = 0;
      for (int32_t k : s.offs) ks_ind += ctx.rotation_cost(k);
      auto plan = ctx.plan_rotations(s.offs);
      double t_ind = 0, t_near = 0, t_hoist = 0, err = 0;
      for (uint32_t r = 0; r < reps; ++r) {
        t0 = now();
        std::vector<Ctxt> a(s.offs.size());
#pragma omp parallel for schedule(dynamic) if (ctx.parallel())
        for (size_t i = 0; i < s.offs.size(); ++i) a[i] = ctx.rotate(c, s.offs[i]);
        t_ind += now() - t0;
        t0 = now();
        std::vector<Ctxt> b = rotate_nearest(ctx, c, s.offs, &ks_near);
        t_near += now() - t0;
        t0 = now();
        std::vector<Ctxt> h = ctx.rotate_many(c, s.offs);
        t_hoist += now() - t0;
        if (r == 0)
          for (size_t i = 0; i < s.offs.size(); ++i) {
            auto want = ctx.decrypt(a[i], slots);
            for (const Ctxt* got : {&b[i], &h[i]}) {
              auto g = ctx.decrypt(*got, slots);
              for (uint32_t j = 0; j < slots; ++j) {
                const double ref = x[static_cast<size_t>(normalize_rotation(int64_t(j) + s.offs[i], slots))];
                err = std::max({err, std::fabs(g[j] - ref), std::fabs(want[j] - ref)});
              }
            }
          }
      }
      std::string name = s.name.substr(0, 34);
      std::printf("  %-34s %10zu %9.2f s %10zu %9.2f s %10zu %8zu %9.2f s   err %.1e\n", name.c_str(), ks_ind,
                  t_ind / reps, ks_near, t_near / reps, plan->key_switches(), plan->decompositions(), t_hoist / reps,
                  err);
    }
  }
  return 0;
}
