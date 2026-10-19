// Builds a small CNN in code (no model file) and runs it encrypted:
//   conv 3x3 -> avgpool -> Chebyshev ReLU -> strided conv -> dense
// Shows the API and the BSGS plan each linear layer compiles to.
//
//   conv_demo [--insecure]
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "fheml/fheml.h"

using namespace fheml;

int main(int argc, char** argv) {
  bool secure = !(argc > 1 && std::strcmp(argv[1], "--insecure") == 0);
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  auto rand_vec = [&](size_t n, double scale) {
    std::vector<double> v(n);
    for (double& x : v) x = scale * u(rng);
    return v;
  };

  Network net(Shape{3, 16, 16});
  net.emplace<Conv2d>(3, 8, 3, 3, /*stride=*/1, /*pad=*/1, rand_vec(8 * 3 * 9, 0.2), rand_vec(8, 0.1))
      .emplace<AvgPool2d>(2, 2)
      .emplace<ChebyshevActivation>("relu", -2.0, 2.0, 27)
      .emplace<Conv2d>(8, 8, 3, 3, /*stride=*/2, /*pad=*/1, rand_vec(8 * 8 * 9, 0.1), rand_vec(8, 0.1))
      .emplace<Dense>(8 * 4 * 4, 10, rand_vec(10 * 128, 0.1), rand_vec(10, 0.1));
  net.compile();
  std::printf("%s\n\n", net.summary().c_str());

  FHEContext ctx(net.context_options(secure));
  ctx.keygen(net.rotation_indices());
  std::printf("%s\n\n", ctx.describe().c_str());

  std::vector<double> image = rand_vec(net.input_shape().size(), 1.0);
  Ctxt y = net.forward(ctx, ctx.encrypt(image), [](size_t i, const Layer& l, const Ctxt&, double s) {
    std::printf("  layer %zu %-10s %.3fs\n", i, l.type().c_str(), s);
  });

  auto got = net.decrypt_output(ctx, y);
  auto ref = net.forward_plain(image);
  double err = 0.0;
  for (size_t i = 0; i < 10; ++i) err = std::max(err, std::fabs(got[i] - ref[i]));
  std::printf("\nmax |encrypted - plaintext| = %.2e\n", err);
  return err < 1e-2 ? 0 : 1;
}
