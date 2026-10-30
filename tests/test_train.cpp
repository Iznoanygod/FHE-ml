// Encrypted training tests (no framework, INSECURE toy parameters).
//
// 1. train_plain's gradients against finite differences, for every layer type
//    that can be trained or backpropagated through.
// 2. Encrypted SGD updates against train_plain on the same batch: the
//    decrypted weights must match the plaintext-trained ones.
// 3. The same with bootstrapped weights over several updates.
// 4. Deep networks under a level budget: activations and gradients are
//    bootstrapped inside the step (a softmax transformer block, softmax with
//    the scheme-switching row maximum).
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "fheml/fheml.h"

using namespace fheml;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& name, const std::string& detail = "") {
  std::printf("[%s] %s %s\n", ok ? " OK " : "FAIL", name.c_str(), detail.c_str());
  if (!ok) ++g_failures;
}

void check_err(const std::string& name, double err, double tol) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "(max err %.2e, tol %.0e)", err, tol);
  check(err <= tol, name, buf);
}

struct Rng {
  std::mt19937 g;
  explicit Rng(unsigned seed) : g(seed) {}
  std::vector<double> vec(size_t n, double s) {
    std::uniform_real_distribution<double> d(-s, s);
    std::vector<double> v(n);
    for (double& x : v) x = d(g);
    return v;
  }
};

using Layers = std::vector<std::shared_ptr<Layer>>;
using Builder = std::function<Network()>;

// Every trainable layer, depth first.
void trainable_layers(const Layers& ls, std::vector<Layer*>& out) {
  for (const auto& l : ls) {
    if (auto* r = dynamic_cast<Residual*>(l.get())) trainable_layers(r->body(), out);
    if (l->trainable()) out.push_back(l.get());
  }
}

std::vector<double> all_params(const Network& net) {
  std::vector<Layer*> ls;
  trainable_layers(net.layers(), ls);
  std::vector<double> p;
  for (auto* l : ls) {
    auto q = l->params();
    p.insert(p.end(), q.begin(), q.end());
  }
  return p;
}

double max_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double e = 0;
  for (size_t i = 0; i < a.size(); ++i) e = std::max(e, std::fabs(a[i] - b[i]));
  return e;
}

// ------------------------------------------------------------------ networks

Network mlp(unsigned seed) {
  Rng r(seed);
  Network n(Shape{12, 1, 1});
  n.emplace<Dense>(12, 6, r.vec(72, 0.4), r.vec(6, 0.1))
      .emplace<PolyActivation>(std::vector<double>{0.0, 0.5, 0.25})
      .emplace<Dense>(6, 3, r.vec(18, 0.4), r.vec(3, 0.1));
  return n;
}

// Shared weights everywhere: conv taps and biases, per-channel affine; a
// frozen average pool and a square in between.
Network cnn(unsigned seed) {
  Rng r(seed);
  Network n(Shape{2, 6, 6});
  n.emplace<Conv2d>(2, 3, 3, 3, 1, 1, r.vec(54, 0.3), r.vec(3, 0.1))
      .emplace<ChannelAffine>(std::vector<double>{1.0, 0.8, 1.2}, std::vector<double>{0.0, 0.1, -0.1})
      .emplace<Square>()
      .emplace<AvgPool2d>(2, 2)
      .emplace<Dense>(27, 2, r.vec(54, 0.3), r.vec(2, 0.1));
  return n;
}

Network strided_cnn(unsigned seed) {
  Rng r(seed);
  Network n(Shape{1, 8, 8});
  n.emplace<Conv2d>(1, 2, 3, 3, 2, 1, r.vec(18, 0.4), r.vec(2, 0.1))
      .emplace<PolyActivation>(std::vector<double>{0.0, 0.5, 0.25})
      .emplace<Dense>(32, 2, r.vec(64, 0.3), std::vector<double>{});
  return n;
}

// Multiplexed packing: the stride-2 conv interleaves its 4 channels into the
// gaps (m = 2), the stride-1 conv keeps that layout.
Network multiplexed_cnn(unsigned seed) {
  Rng r(seed);
  Network n(Shape{1, 8, 8});
  n.set_packing(Packing::kMultiplexed);
  n.emplace<Conv2d>(1, 4, 3, 3, 2, 1, r.vec(36, 0.4), r.vec(4, 0.1))
      .emplace<PolyActivation>(std::vector<double>{0.0, 0.5, 0.25})
      .emplace<Conv2d>(4, 2, 3, 3, 1, 1, r.vec(72, 0.3), r.vec(2, 0.1))
      .emplace<Dense>(32, 2, r.vec(64, 0.3), std::vector<double>{});
  return n;
}

// conv -> exact max pool (scheme switching) -> poly -> dense. Small conv
// weights keep the pooled values inside the comparison range.
Network pool_cnn(unsigned seed) {
  Rng r(seed);
  Network n(Shape{1, 4, 4});
  n.emplace<Conv2d>(1, 2, 3, 3, 1, 1, r.vec(18, 0.2), r.vec(2, 0.1))
      .emplace<MaxPool2d>(2, 2, 2.0)
      .emplace<PolyActivation>(std::vector<double>{0.0, 0.5, 0.25})
      .emplace<Dense>(8, 2, r.vec(16, 0.4), r.vec(2, 0.1));
  return n;
}

// Overlapping 3x3 windows (stride 1, odd carries in the tournament) as the
// output layer: the loss gradient is nonzero in the gap slots too and the
// backward pass must keep it out of the conv.
Network pool_output(unsigned seed) {
  Rng r(seed);
  Network n(Shape{1, 4, 4});
  n.emplace<Conv2d>(1, 1, 3, 3, 1, 1, r.vec(9, 0.2), r.vec(1, 0.1)).emplace<MaxPool2d>(3, 1, 2.0);
  return n;
}

// Token layers with shared and per-token biases, a Chebyshev activation and a
// residual connection.
Network tokens(unsigned seed) {
  Rng r(seed);
  const int t = 3, d = 4;
  Layers mlp = {std::make_shared<TokenDense>(t, d, 8, r.vec(32, 0.4), r.vec(8, 0.1)),
                std::make_shared<ChebyshevActivation>("silu", -3.0, 3.0, 7),
                std::make_shared<TokenDense>(t, 8, d, r.vec(32, 0.3), r.vec(d, 0.1))};
  Network n(Shape{t, 1, 5});
  n.emplace<TokenDense>(t, 5, d, r.vec(20, 0.4), r.vec(t * d, 0.1))
      .add(std::make_shared<Residual>(mlp))
      .emplace<TokenMean>(t, d)
      .emplace<Dense>(d, 2, r.vec(8, 0.4), r.vec(2, 0.1));
  return n;
}

// LayerNorm over 4 features per token (variance range calibrated on random
// inputs; eps keeps 1/sqrt(v + eps) smooth down to v = 0), trained gamma and
// beta.
std::shared_ptr<LayerNorm> layernorm(int t, int d, Rng& r, const Network& prefix) {
  std::vector<double> gamma(d), beta = r.vec(d, 0.2);
  for (double& g : gamma) g = 1.0 + r.vec(1, 0.5)[0];
  double var_max = 0.0;
  Rng cal(55);
  for (int i = 0; i < 32; ++i) {
    std::vector<double> x = prefix.forward_plain(cal.vec(prefix.input_shape().size(), 1.0));
    for (int tok = 0; tok < t; ++tok) {
      double m = 0.0, m2 = 0.0;
      for (int j = 0; j < d; ++j) {
        m += x[tok * d + j];
        m2 += x[tok * d + j] * x[tok * d + j];
      }
      var_max = std::max(var_max, m2 / d - (m / d) * (m / d));
    }
  }
  return std::make_shared<LayerNorm>(t, d, gamma, beta, 0.05, 0.0, 2.0 * var_max, 31, 1);
}

Network norm_tokens(unsigned seed) {
  Rng r(seed);
  const int t = 3, d = 4;
  const std::vector<double> w = r.vec(20, 0.4), b = r.vec(d, 0.1);
  Network prefix(Shape{t, 1, 5});
  prefix.emplace<TokenDense>(t, 5, d, w, b);
  prefix.compile();
  Network n(Shape{t, 1, 5});
  n.emplace<TokenDense>(t, 5, d, w, b)
      .add(layernorm(t, d, r, prefix))
      .emplace<TokenMean>(t, d)
      .emplace<Dense>(d, 2, r.vec(8, 0.4), r.vec(2, 0.1));
  return n;
}

std::shared_ptr<Attention> attention(int t, int d, Rng& r, bool softmax, bool subtract_max = false) {
  auto a = std::make_shared<Attention>(t, d, r.vec(d * d, 0.2), r.vec(d * d, 0.2), r.vec(d * d, 0.2), r.vec(d * d, 0.2),
                                       r.vec(d, 0.1), r.vec(d, 0.1), r.vec(d, 0.1), r.vec(d, 0.1),
                                       std::vector<double>{4.0 / (8 * t), 4.0 / (8 * t), 1.0 / (8 * t)}, 2);
  if (softmax) {
    SoftmaxOptions o;
    o.lo = -1.5;
    o.hi = 1.5;
    o.exp_degree = 15;
    o.inv_degree = 31;
    o.newton = 2;
    o.subtract_max = subtract_max;
    a->set_softmax(o);
  }
  return a;
}

// Two-head attention (quadratic or softmax scores) between token layers.
Network attention_tokens(unsigned seed, bool softmax, bool subtract_max = false) {
  Rng r(seed);
  const int t = 4, d = 8;
  Network n(Shape{t, 1, 5});
  n.emplace<TokenDense>(t, 5, d, r.vec(40, 0.4), r.vec(d, 0.1))
      .add(attention(t, d, r, softmax, subtract_max))
      .emplace<TokenMean>(t, d)
      .emplace<Dense>(d, 2, r.vec(16, 0.4), r.vec(2, 0.1));
  return n;
}

// A pre-norm transformer block: LayerNorm + softmax attention, LayerNorm +
// MLP, each with a residual connection.
Network transformer_block(unsigned seed) {
  Rng r(seed);
  const int t = 4, d = 8;
  const std::vector<double> w = r.vec(40, 0.4), b = r.vec(d, 0.1);
  Network prefix(Shape{t, 1, 5});
  prefix.emplace<TokenDense>(t, 5, d, w, b);
  prefix.compile();
  Layers attn = {layernorm(t, d, r, prefix), attention(t, d, r, true)};
  Layers mlp = {layernorm(t, d, r, prefix), std::make_shared<TokenDense>(t, d, 16, r.vec(128, 0.3), r.vec(16, 0.1)),
                std::make_shared<ChebyshevActivation>("silu", -3.0, 3.0, 7),
                std::make_shared<TokenDense>(t, 16, d, r.vec(128, 0.2), r.vec(d, 0.1))};
  Network n(Shape{t, 1, 5});
  n.emplace<TokenDense>(t, 5, d, w, b)
      .add(std::make_shared<Residual>(attn))
      .add(std::make_shared<Residual>(mlp))
      .emplace<TokenMean>(t, d)
      .emplace<Dense>(d, 2, r.vec(16, 0.4), r.vec(2, 0.1));
  return n;
}

struct Batch {
  std::vector<std::vector<double>> xs, ts;
};

// Cross-entropy batches get one-hot targets.
Batch batch(const Network& n, size_t count, unsigned seed, bool onehot = false) {
  Rng r(seed);
  Batch b;
  const size_t k = n.output_shape().size();
  for (size_t i = 0; i < count; ++i) {
    b.xs.push_back(r.vec(n.input_shape().size(), 1.0));
    b.ts.push_back(r.vec(k, 1.0));
    if (onehot) {
      std::vector<double>& t = b.ts.back();
      const size_t c = static_cast<size_t>((t[0] + 1.0) * 0.5 * static_cast<double>(k)) % k;
      std::fill(t.begin(), t.end(), 0.0);
      t[c] = 1.0;
    }
  }
  return b;
}

// --------------------------------------------------------- gradient checks

// Cross-entropy with the exact softmax: train_plain's gradient uses the
// polynomial one, so the check also bounds the approximation error.
double loss(const Network& net, const Batch& b, const Loss& kind) {
  double l = 0;
  for (size_t s = 0; s < b.xs.size(); ++s) {
    auto y = net.forward_plain(b.xs[s]);
    if (kind.kind == Loss::kCrossEntropy) {
      l += cross_entropy_plain(y, b.ts[s]);
      continue;
    }
    for (size_t i = 0; i < y.size(); ++i) l += 0.5 * (y[i] - b.ts[s][i]) * (y[i] - b.ts[s][i]);
  }
  return l / static_cast<double>(b.xs.size());
}

void gradient_check(const std::string& name, const Builder& make, const std::vector<size_t>& frozen = {},
                    const Loss& kind = Loss{}) {
  Network net = make();
  net.compile();
  Batch b = batch(net, 3, 7, kind.kind == Loss::kCrossEntropy);
  // One SGD step with lr = 1 moves the parameters by -grad.
  std::vector<double> p0 = all_params(net);
  train_plain(net, b.xs, b.ts, 1.0, frozen, kind);
  std::vector<double> p1 = all_params(net);

  Network ref = make();
  ref.compile();
  double err = 0, scale = 0;
  size_t k = 0;
  for (size_t top = 0; top < ref.layers().size(); ++top) {
    std::vector<Layer*> ls;
    trainable_layers({ref.layers()[top]}, ls);
    const bool is_frozen = std::find(frozen.begin(), frozen.end(), top) != frozen.end();
    for (auto* l : ls) {
      std::vector<double> p = l->params();
      for (size_t i = 0; i < p.size(); ++i, ++k) {
        const double eps = 1e-5, v = p[i];
        p[i] = v + eps;
        l->set_params(p);
        double up = loss(ref, b, kind);
        p[i] = v - eps;
        l->set_params(p);
        double down = loss(ref, b, kind);
        p[i] = v;
        l->set_params(p);
        double num = is_frozen ? 0.0 : (up - down) / (2 * eps);
        double got = p0[k] - p1[k];
        err = std::max(err, std::fabs(num - got));
        scale = std::max(scale, std::fabs(num));
      }
    }
  }
  check_err("gradient " + name, err / std::max(scale, 1e-3), 1e-6);
}

// ------------------------------------------------------- encrypted training

FHEContext training_context(const Trainer& tr, bool boot, uint32_t updates, uint32_t window) {
  ContextOptions o = tr.context_options(false, boot, updates, window);
  o.ring_dim = boot ? 4096 : std::max(2048u, 2 * o.slots);
  FHEContext ctx(o);
  ctx.keygen(tr.rotation_indices(window > 0));
  return ctx;
}

// budget > 0: bootstrap activations and gradients inside the step with that
// level budget and bootstrap range.
void encrypted_check(const std::string& name, const Builder& make, size_t updates, bool boot, double tol,
                     const std::vector<size_t>& frozen = {}, double lr = 0.5, uint32_t window = 0,
                     const Loss& kind = Loss{}, uint32_t budget = 0, double range = 1.0) {
  const bool onehot = kind.kind == Loss::kCrossEntropy;
  Network plain = make();
  plain.compile();
  Network enc = make();
  // Bootstrapping: use all 2048 slots of the ring so weights are packed
  // (only the slots needed with a level budget: sparse bootstraps are more
  // precise and faster, and activations are bootstrapped many times).
  enc.compile(boot && !budget ? 2048 : 0);
  Trainer tr(enc, frozen, kind);
  if (budget) {
    tr.set_level_budget(budget, range);
    check(tr.unbounded_step_depth() > budget && tr.step_depth() <= budget, name + " step fits the level budget",
          "(" + std::to_string(tr.unbounded_step_depth()) + " levels unbounded, budget " + std::to_string(budget) +
              ", min " + std::to_string(tr.min_level_budget()) + ")");
  }
  FHEContext ctx = training_context(tr, boot, static_cast<uint32_t>(updates), window);
  std::printf("  %s\n  %s\n", tr.summary().c_str(), ctx.describe().c_str());
  if (window) {
    const size_t direct = tr.rotation_indices(false).size() + (boot ? 0 : 0);
    check(ctx.rotation_key_count() < direct || boot, name + " ladder uses fewer keys",
          "(" + std::to_string(ctx.rotation_key_count()) + " vs " + std::to_string(direct) + " direct)");
  }
  tr.encrypt_weights(ctx);

  std::vector<double> before = all_params(plain);
  for (size_t u = 0; u < updates; ++u) {
    Batch b = batch(plain, 2, 100 + static_cast<unsigned>(u), onehot);
    train_plain(plain, b.xs, b.ts, lr, frozen, kind);
    size_t boots = 0;
    for (size_t s = 0; s < b.xs.size(); ++s) {
      tr.accumulate(ctx, tr.encrypt_input(ctx, b.xs[s]), tr.encrypt_target(ctx, b.ts[s]));
      boots += tr.sample_bootstraps();
    }
    tr.update(ctx, lr);
    if (budget)
      check(boots > 0, name + " bootstrapped activations and gradients",
            "(" + std::to_string(boots) + " bootstraps for " + std::to_string(b.xs.size()) + " samples)");
    if (!boot)
      check(tr.weight_level() <= ctx.total_depth(), name + " weights within the planned depth",
            "(level " + std::to_string(tr.weight_level()) + " of " + std::to_string(ctx.total_depth()) + ")");
  }
  // The trained encrypted weights also run inference.
  Batch probe = batch(plain, 1, 999);
  auto y = tr.decrypt_output(ctx, tr.forward(ctx, tr.encrypt_input(ctx, probe.xs[0])));

  tr.decrypt_weights(ctx);
  std::vector<double> want = all_params(plain), got = all_params(enc);
  check(max_diff(before, want) > 10 * tol, name + " training changed the weights");
  check_err("encrypted training " + name, max_diff(want, got), tol);
  check_err("inference with trained encrypted weights " + name, max_diff(y, plain.forward_plain(probe.xs[0])), 10 * tol);
  if (!frozen.empty()) {
    // Frozen layers keep their weights.
    Network orig = make();
    Layer* a = orig.layers()[frozen[0]].get();
    Layer* c = enc.layers()[frozen[0]].get();
    check(max_diff(a->params(), c->params()) == 0.0, name + " frozen layer unchanged");
  }
}

// Softmax cross-entropy: logits of the test networks stay within [-2, 2].
Loss cross_entropy(bool subtract_max) {
  SoftmaxOptions o;
  o.lo = -2.0;
  o.hi = 2.0;
  o.exp_degree = 15;
  o.inv_degree = 31;
  o.newton = subtract_max ? 1 : 2;
  o.subtract_max = subtract_max;
  return Loss::cross_entropy(o);
}

// A ten-class head (gapped by a strided conv in front), for the max tree
// over a wider window than the three-output mlp.
Network classifier(unsigned seed) {
  Rng r(seed);
  Network n(Shape{1, 6, 6});
  n.emplace<Conv2d>(1, 2, 3, 3, 2, 1, r.vec(18, 0.4), r.vec(2, 0.1))
      .emplace<Square>()
      .emplace<Dense>(18, 10, r.vec(180, 0.3), r.vec(10, 0.1));
  return n;
}

void cross_entropy_tests(bool quick) {
  // The polynomial softmax against the exact one.
  for (int sm = 0; sm < 2; ++sm) {
    Rng r(31);
    double err = 0.0;
    for (int i = 0; i < 20; ++i) {
      std::vector<double> y = r.vec(10, 2.0), p = softmax_plain(cross_entropy(sm != 0), y);
      double m = *std::max_element(y.begin(), y.end()), S = 0.0;
      for (double v : y) S += std::exp(v - m);
      for (size_t j = 0; j < y.size(); ++j) err = std::max(err, std::fabs(p[j] - std::exp(y[j] - m) / S));
    }
    check_err(std::string("polynomial softmax vs exact") + (sm ? ", max subtracted" : ""), err, 1e-6);
  }
  gradient_check("cross-entropy, mlp", [] { return mlp(1); }, {}, cross_entropy(false));
  gradient_check("cross-entropy, ten classes", [] { return classifier(13); }, {}, cross_entropy(false));
  gradient_check("cross-entropy, max subtracted", [] { return classifier(13); }, {}, cross_entropy(true));
  gradient_check("cross-entropy, softmax attention", [] { return attention_tokens(11, true); }, {}, cross_entropy(false));

  encrypted_check("cross-entropy mlp", [] { return mlp(1); }, 1, false, 1e-4, {}, 0.5, 0, cross_entropy(false));
  encrypted_check("cross-entropy ten classes", [] { return classifier(13); }, 1, false, 1e-4, {}, 0.5, 0,
                  cross_entropy(false));
  encrypted_check("cross-entropy mlp, ladder window 1", [] { return mlp(5); }, 1, false, 1e-4, {0}, 0.5, 1,
                  cross_entropy(false));
  if (!quick)
    encrypted_check("cross-entropy mlp, bootstrapped weights", [] { return mlp(6); }, 2, true, 1e-3, {}, 0.3, 0,
                    cross_entropy(false));
  // The loss gradient under a level budget: the softmax of the loss is
  // bootstrapped between its exp and 1/x series like the layers.
  if (!quick) {
    Network probe = mlp(6);
    probe.compile();
    const uint32_t budget = Trainer(probe, {}, cross_entropy(false)).min_level_budget() + 1;
    encrypted_check("cross-entropy mlp, level budget", [] { return mlp(6); }, 1, true, 1e-3, {}, 0.3, 0,
                    cross_entropy(false), budget, 32.0);
  }
  // Scheme-switching comparisons (CPU OpenFHE only).
  if (!quick && !gpu_backend_built())
    encrypted_check("cross-entropy ten classes, max subtracted", [] { return classifier(13); }, 1, false, 1e-3, {}, 0.5,
                    0, cross_entropy(true));
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;

  gradient_check("mlp", [] { return mlp(1); });
  gradient_check("cnn (shared weights, affine, square, avgpool)", [] { return cnn(2); });
  gradient_check("strided cnn", [] { return strided_cnn(3); });
  {
    Network n = multiplexed_cnn(13);
    n.compile();
    check(n.layers()[0]->out_layout().m == 2 && n.layers()[2]->out_layout().m == 2, "multiplexed cnn layouts (m = 2)");
  }
  gradient_check("multiplexed cnn", [] { return multiplexed_cnn(13); });
  gradient_check("tokens (residual, chebyshev, per-token bias)", [] { return tokens(4); });
  gradient_check("mlp, first layer frozen", [] { return mlp(5); }, {0});
  gradient_check("layernorm (gamma, beta, backward through the normalisation)", [] { return norm_tokens(9); });
  gradient_check("quadratic attention (two heads)", [] { return attention_tokens(10, false); });
  gradient_check("softmax attention (two heads)", [] { return attention_tokens(11, true); });
  gradient_check("softmax attention, frozen", [] { return attention_tokens(11, true); }, {1});
  gradient_check("pre-norm transformer block", [] { return transformer_block(12); });
  gradient_check("max pool 2x2", [] { return pool_cnn(13); });
  gradient_check("max pool 3x3 stride 1, output layer", [] { return pool_output(14); });

  encrypted_check("mlp", [] { return mlp(1); }, 1, false, 1e-4);
  encrypted_check("mlp, two updates", [] { return mlp(1); }, 2, false, 1e-4);
  encrypted_check("cnn", [] { return cnn(2); }, 1, false, 1e-4);
  encrypted_check("strided cnn", [] { return strided_cnn(3); }, 1, false, 1e-4);
  encrypted_check("multiplexed cnn", [] { return multiplexed_cnn(13); }, 1, false, 1e-4);
  encrypted_check("tokens", [] { return tokens(4); }, 1, false, 1e-4);
  encrypted_check("mlp, fine-tune last layer", [] { return mlp(5); }, 1, false, 1e-4, {0});
  encrypted_check("layernorm", [] { return norm_tokens(9); }, 1, false, 1e-4);
  encrypted_check("quadratic attention", [] { return attention_tokens(10, false); }, 1, false, 1e-4);
  encrypted_check("softmax attention", [] { return attention_tokens(11, true); }, 1, false, 1e-4);
  // Scheme-switching comparisons in the forward pass; the backward pass
  // reuses their bits.
  encrypted_check("max pool 2x2", [] { return pool_cnn(13); }, 1, false, 1e-3);
  if (!quick) encrypted_check("max pool 3x3 stride 1, output layer", [] { return pool_output(14); }, 1, false, 1e-3);
  if (!quick) encrypted_check("softmax attention, frozen (forward only)", [] { return attention_tokens(11, true); }, 1, false, 1e-4, {1});
  // Rotation-key ladder: the same training with composed rotations.
  encrypted_check("cnn, ladder window 1", [] { return cnn(2); }, 1, false, 1e-4, {}, 0.5, 1);
  encrypted_check("mlp, fine-tune last layer, ladder window 3", [] { return mlp(5); }, 1, false, 1e-4, {0}, 0.5, 3);
  // Training through an automatically placed bootstrap (budget 2 splits the
  // mlp after its activation).
  if (!quick)
    encrypted_check(
        "mlp with auto-placed bootstrap, ladder",
        [] {
          Network n = mlp(8);
          std::vector<std::vector<double>> cal;
          Rng r(99);
          for (int i = 0; i < 4; ++i) cal.push_back(r.vec(12, 1.0));
          n.place_bootstraps(2, cal);
          return n;
        },
        2, true, 1e-3, {}, 0.3, 1);
  if (!quick) encrypted_check("mlp, bootstrapped weights", [] { return mlp(6); }, 3, true, 1e-3, {}, 0.3);
  if (!quick) encrypted_check("cnn, bootstrapped weights", [] { return cnn(7); }, 2, true, 1e-3, {}, 0.3);
  // Bootstrapping inside the step: activations and gradients are refreshed
  // whenever an operation would go deeper than the level budget.
  encrypted_check("cnn, level budget 2", [] { return cnn(7); }, 2, true, 1e-3, {}, 0.3, 0, Loss{}, 2, 16.0);
  if (!quick) {
    Network probe = transformer_block(13);
    probe.compile();
    const uint32_t min_budget = Trainer(probe).min_level_budget();
    encrypted_check("pre-norm transformer block, level budget", [] { return transformer_block(13); }, 1, true, 1e-3,
                    {}, 0.3, 0, Loss{}, min_budget + 1, 32.0);
    Network probe2 = attention_tokens(14, true, true);
    probe2.compile();
    encrypted_check("softmax attention with row maximum, level budget", [] { return attention_tokens(14, true, true); },
                    1, true, 1e-3, {}, 0.3, 0, Loss{}, Trainer(probe2).min_level_budget() + 1, 8.0);
  }

  cross_entropy_tests(quick);

  std::printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "PASSED", g_failures, g_failures == 1 ? "" : "s");
  return g_failures ? 1 : 0;
}
