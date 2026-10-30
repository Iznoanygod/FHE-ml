// Encrypted MNIST inference with a CNN trained by tools/train_mnist.py.
//
//   mnist_cnn [--model F] [--samples F] [--encrypted N] [--insecure] [--fuse] [--cache]
//             [--encrypt-weights] [--packing gapped|compact|multiplexed] [--exact-relu]
//
//   --encrypt-weights  encrypt the conv/dense weights and biases too
//                      (ciphertext x ciphertext products)
//   --fuse   merge adjacent linear layers (none in the default model)
//   --cache  keep encoded weights between samples (faster after the first)
//   --exact-relu  replace Chebyshev ReLUs by the exact scheme-switching ReLU
//                 (one FHEW bootstrap per slot each: slow)
//   --packing  layout of strided conv outputs (default gapped; see Packing)
//
// Runs the plaintext reference on every sample, then N samples under CKKS,
// reporting per-layer timings, agreement with the reference and logit error.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fheml/fheml.h"

using namespace fheml;

namespace {

struct Sample {
  int label;
  std::vector<double> pixels;
};

std::vector<Sample> load_samples(const std::string& path) {
  std::ifstream is(path);
  if (!is) throw std::runtime_error("cannot open " + path);
  size_t n, dim;
  is >> n >> dim;
  std::vector<Sample> out(n);
  for (auto& s : out) {
    is >> s.label;
    s.pixels.resize(dim);
    for (double& v : s.pixels) is >> v;
  }
  if (!is) throw std::runtime_error("truncated samples file " + path);
  return out;
}

size_t argmax(const std::vector<double>& v) { return std::max_element(v.begin(), v.end()) - v.begin(); }

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path = "models/mnist_cnn.fheml", samples_path = "models/mnist_test_samples.txt";
  size_t n_enc = 5;
  bool secure = true, fuse = false, cache = false, enc_weights = false, exact_relu = false;
  Packing packing = Packing::kGapped;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model" && i + 1 < argc) model_path = argv[++i];
    else if (a == "--samples" && i + 1 < argc) samples_path = argv[++i];
    else if (a == "--encrypted" && i + 1 < argc) n_enc = std::strtoul(argv[++i], nullptr, 10);
    else if (a == "--insecure") secure = false;
    else if (a == "--fuse") fuse = true;
    else if (a == "--cache") cache = true;
    else if (a == "--encrypt-weights") enc_weights = true;
    else if (a == "--exact-relu") exact_relu = true;
    else if (a == "--packing" && i + 1 < argc) {
      std::string p = argv[++i];
      packing = p == "compact" ? Packing::kCompact : p == "multiplexed" ? Packing::kMultiplexed : Packing::kGapped;
    } else {
      std::fprintf(stderr, "usage: %s [--model F] [--samples F] [--encrypted N] [--insecure] [--fuse] [--cache] [--encrypt-weights] [--packing P] [--exact-relu]\n", argv[0]);
      return 2;
    }
  }

  Network net = Network::load(model_path);
  auto samples = load_samples(samples_path);
  net.set_packing(packing);
  if (fuse) net.fuse_linear();
  if (exact_relu) std::printf("exact relu: replaced %zu Chebyshev relus\n", net.use_exact_relu());
  net.compile();
  net.set_cache_plaintexts(cache);
  std::printf("%s\n\n", net.summary().c_str());

  size_t correct = 0;
  std::vector<std::vector<double>> ref(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    ref[i] = net.forward_plain(samples[i].pixels);
    correct += argmax(ref[i]) == static_cast<size_t>(samples[i].label);
  }
  std::printf("plaintext accuracy: %zu/%zu = %.2f%%\n\n", correct, samples.size(), 100.0 * correct / samples.size());

  n_enc = std::min(n_enc, samples.size());
  if (!n_enc) return 0;

  auto t0 = std::chrono::steady_clock::now();
  ContextOptions opts = net.context_options(secure);
  FHEContext ctx(opts);
  ctx.keygen(net.rotation_indices());
  std::printf("%s\n", ctx.describe().c_str());
  std::printf("rotation keys: %zu, setup + keygen %.2fs\n", net.rotation_indices().size(), seconds_since(t0));
  if (enc_weights) {
    t0 = std::chrono::steady_clock::now();
    net.encrypt_weights(ctx);
    size_t n_ct = 0;
    for (const auto& l : net.layers())
      if (l->is_linear()) n_ct += static_cast<const LinearLayer*>(l.get())->transform().num_diagonals() + 1;
    std::printf("encrypted weights: %zu ciphertexts in %.2fs\n", n_ct, seconds_since(t0));
  }
  std::printf("\n");

  size_t agree = 0, enc_correct = 0;
  double worst = 0.0, total_time = 0.0;
  for (size_t i = 0; i < n_enc; ++i) {
    t0 = std::chrono::steady_clock::now();
    Ctxt c = ctx.encrypt(samples[i].pixels, net.input_level(ctx));
    double t_enc = seconds_since(t0);

    t0 = std::chrono::steady_clock::now();
    Network::LayerHook hook = nullptr;
    if (i == 0)
      hook = [](size_t idx, const Layer& l, const Ctxt& out, double secs) {
        std::printf("  layer %zu %-10s %7.3fs  level %u\n", idx, l.type().c_str(), secs,
                    static_cast<unsigned>(out->GetLevel()));
      };
    Ctxt y = net.forward(ctx, c, hook);
    double t_inf = seconds_since(t0);
    total_time += t_inf;

    auto logits = net.decrypt_output(ctx, y);
    double err = 0.0;
    for (size_t k = 0; k < 10; ++k) err = std::max(err, std::fabs(logits[k] - ref[i][k]));
    worst = std::max(worst, err);
    size_t pred = argmax(logits);
    agree += pred == argmax(ref[i]);
    enc_correct += pred == static_cast<size_t>(samples[i].label);
    std::printf("sample %zu: label %d, encrypted prediction %zu, plaintext %zu, max logit err %.2e, "
                "encrypt %.3fs, inference %.2fs\n",
                i, samples[i].label, pred, argmax(ref[i]), err, t_enc, t_inf);
  }
  std::printf("\nencrypted: %zu/%zu correct, %zu/%zu agree with plaintext, worst logit err %.2e, "
              "mean inference %.2fs\n",
              enc_correct, n_enc, agree, n_enc, worst, total_time / n_enc);
  return agree == n_enc ? 0 : 1;
}
