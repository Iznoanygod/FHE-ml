# FHE-ml

Encrypted neural-network inference (CNNs included) on top of
[OpenFHE](https://github.com/openfheorg/openfhe-development)'s CKKS scheme.
The server never sees the input, the intermediate activations or the result.

This is a rewrite of the 2022 capstone prototype, which ran a 784-200-10 MLP on
PALISADE and needed the client to decrypt and apply the sigmoid between layers.
That code is kept in [`legacy/`](legacy/) for reference.

## What it does

* **Every linear layer is a matrix over the ciphertext slots.** Convolutions
  (any kernel size, stride, padding, channel count), dense layers, average
  pooling and per-channel affine (inference batch-norm) are lowered to a sparse
  matrix. That matrix is evaluated with the Halevi–Shoup diagonal method,
  using baby-step/giant-step and hoisted rotations. Each linear layer costs one
  level.
* **Strided layers keep their outputs "gapped".** The outputs stay where the
  input pixels were instead of being compacted. That keeps a strided
  convolution at `k*k*(Ci+Co-1)` diagonals; the next layer absorbs the gaps.
  With `Packing::kMultiplexed` the gaps hold other channels instead
  (multiplexed packing, Lee et al. ICML'22): compact slot counts with few
  diagonals.
* **Dense heads use the Gazelle hybrid method.** Partial sums are folded into
  at most `2R-1` diagonals, plus `log2(S/R)` rotate-and-sum steps.
  * The MNIST `845->64` layer needs 126 diagonals instead of 908.
* **Activations run entirely under encryption.**
  * `square` (CryptoNets).
  * Trained polynomials `poly`.
  * Chebyshev approximations of `relu` / `silu` / `gelu` / `sigmoid` / `tanh`
    on a chosen interval and degree, via `EvalChebyshevSeries`.
  * An exact ReLU (`ExactReLU`) whose sign is computed in FHEW through
    scheme switching, as an option next to the Chebyshev ReLU (see
    [Exact ReLU](#exact-relu-via-scheme-switching)).
* **Small transformers.**
  * `TokenDense` is a per-token linear layer.
  * `Attention` is fully homomorphic self-attention: ct×ct QKᵀ and AV, with
    a trainable quadratic in place of softmax, or real softmax
    (`set_softmax`: polynomial exp and 1/x, optionally with the row maximum
    subtracted through scheme switching). `heads=H` gives multi-head
    attention with the heads as sub-blocks of each token, which costs fewer
    rotations than one wide head and no extra levels.
  * `LayerNorm` normalises each token with a Chebyshev + Newton inverse
    square root on a calibrated variance interval (10–12 levels).
  * `Residual` and `TokenMean` complete the block.
  * See [docs/DESIGN.md](docs/DESIGN.md#transformers).
* **A Python package**, `import fheml`, wraps the same API with numpy in and
  numpy out.
* **A PyTorch exporter.** Train with `torch.nn` layers plus the HE-friendly
  modules in `fheml.nn` (polynomial activations, polynomial attention), then
  `fheml.from_torch(model, input_shape)` gives the encrypted network.
* **Encrypted training.** `Trainer` runs SGD with backpropagation entirely on
  ciphertexts: encrypted inputs, labels, weights, activations and gradients.
  The encrypted weights are bootstrapped after every update. Freezing the
  first layers turns the same API into encrypted fine-tuning. See
  [Encrypted training](#encrypted-training).
* **Deep networks.** `Bootstrap` layers refresh ciphertexts, and
  `Network::place_bootstraps()` inserts them automatically: it keeps every
  stretch within a level budget and scales the values entering each
  bootstrap into [-1, 1] from calibration data, folding the scales into the
  neighbouring layers. `fold_activation_intervals()` moves each Chebyshev
  activation's interval map into the preceding linear layer, one level less
  per activation. See [Deep networks](#deep-networks).
* **Small rotation-key sets.** With `ContextOptions::rotation_window`, any
  rotation is composed from a ladder of `2 log2(slots)` keys instead of one
  key per rotation index. At ring 2^16 a key is 100 MB, which is what makes
  encrypted training and bootstrapped inference fit in memory at 128-bit
  security.
* **Optional encrypted weights.** `Network::encrypt_weights(ctx)` encrypts
  every conv/dense weight diagonal and bias, so the linear layers multiply
  ciphertext × ciphertext.
  * Activation polynomials stay public.
  * The weights are encrypted under the same key as the input, as in the 2022
    prototype. Whoever holds the secret key can therefore decrypt the model.
* **Parameter planning.** The compiled network reports its depth, slot count
  and the exact rotation keys it needs. `Network::context_options()` turns that
  into a CKKS context, at 128-bit security by default.
* **A plaintext reference** runs the same layers on doubles, so every
  encrypted result can be checked against it.

See [docs/RESEARCH.md](docs/RESEARCH.md) for the literature survey behind
these choices, and [docs/DESIGN.md](docs/DESIGN.md) for the design and the
roadmap.

## Results

Example MNIST CNN (CryptoNets/LoLa-style), trained with
`tools/train_mnist.py`:

```
conv2d 1->5 k=5 s=2 p=1   125 diagonals (gapped out)        depth 1
poly 0.5x + 0.25x^2                                         depth 2
dense 845->64             126 diagonals (folded R=64)       depth 1
poly 0.5x + 0.25x^2                                         depth 2
dense 64->10               25 diagonals (folded R=16)       depth 1
```

* **Accuracy:** 98.3% in plaintext on the 10k MNIST test set.
* **Parameters:** CKKS at 128-bit security, ring 2^15, 4096 slots, depth 7.
* **Speed:** about 3 s per encrypted image on 4 cores.
* **Agreement:** the encrypted logits match plaintext to about 1e-6, and the
  predicted class is the same.
* **With `--encrypt-weights`:**
  * The 279 weight and bias ciphertexts take about 6 s to encrypt.
  * Inference takes about 3.6 s per image.
  * Peak memory is about 2 GB.
  * The logits match plaintext to about 1e-4.

Tiny vision transformer on MNIST, trained with `tools/train_transformer.py`.
Each image is 16 tokens of 7×7 patches, with d = 32, one attention block and
one MLP block:

* **Accuracy:** 95.6% in plaintext on the 10k test set.
* **Parameters:** 128-bit security, ring 2^15, depth 12.
* **Speed:** about 15 s per encrypted image. With encrypted weights it is also
  about 15 s, after a one-time 27 s to encrypt them.
* **Agreement:** predictions match plaintext.

## Building

You need CMake ≥ 3.16, a C++17 compiler and an OpenFHE install (tested with
v1.4.2 and v1.6.1). For the optional GPU build, see
[GPU acceleration](#gpu-acceleration-fideslib-optional).

```sh
# OpenFHE (once)
git clone --branch v1.4.2 https://github.com/openfheorg/openfhe-development.git
cmake -S openfhe-development -B openfhe-development/build -DCMAKE_INSTALL_PREFIX=$HOME/openfhe \
      -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
cmake --build openfhe-development/build -j && cmake --install openfhe-development/build

# this library
cmake -S . -B build -DCMAKE_PREFIX_PATH=$HOME/openfhe
cmake --build build -j
./build/test_fheml            # plaintext plan checks + encrypted layer tests (toy parameters)
./build/mnist_cnn --encrypted 5
./build/conv_demo
```

### GPU acceleration (FIDESlib, optional)

`-DFHEML_GPU=ON` builds fheml on [FIDESlib](https://github.com/CAPS-UMU/FIDESlib)
2.2, an open-source CUDA CKKS library that wraps OpenFHE. Its API mirrors
OpenFHE's, so the library code is the same. GPU results are bit-identical to
OpenFHE on the CPU. A GPU build still runs on machines without a GPU: the
context then falls back to the OpenFHE CPU code inside FIDESlib.

* **Requirements:** CUDA 12 or 13, an NVIDIA GPU from `sm_70` up, CMake ≥ 3.25
  and GCC ≥ 11.
* **OpenFHE version:** FIDESlib needs a *patched* OpenFHE **1.6.1** with
  static libraries, built without `-march=native`. The CPU-only build keeps
  working with OpenFHE 1.4.2 or 1.6.1.
* **What runs on the GPU:** add/mult/rotate, hoisted rotations (the BSGS baby
  steps and attention), `EvalChebyshevSeries` and bootstrapping. Key
  generation, encryption and decryption stay on the CPU.
* **Differences on the GPU:**
  * FIDESlib's GPU context isn't thread-safe, so fheml's OpenMP loops run
    serially there. The GPU parallelises inside each operation instead.
  * FIDESlib has no `EvalMultNoRelin` or `EvalPoly`. Encrypted weights
    relinearize every product, and `PolyActivation` is rewritten as a
    Chebyshev series on [-1, 1], with the same depth.

```sh
# Patched OpenFHE 1.6.1 + FIDESlib (once)
git clone https://github.com/CAPS-UMU/FIDESlib.git && cd FIDESlib
git clone --branch v1.6.1 https://github.com/openfheorg/openfhe-development.git openfhe-1.6.1
git -C openfhe-1.6.1 apply ../deps/fideslib-1.6.1-compile.patch
cmake -S openfhe-1.6.1 -B openfhe-1.6.1/build -DCMAKE_INSTALL_PREFIX=$HOME/openfhe-fides \
      -DBUILD_STATIC=ON -DWITH_NATIVEOPT=OFF -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
cmake --build openfhe-1.6.1/build -j && cmake --install openfhe-1.6.1/build
# FIDESLIB_ARCH=auto builds for the visible GPUs; CUDA_PATH defaults to /usr/local/cuda
cmake -S . -B build -DOPENFHE_INSTALL_PREFIX=$HOME/openfhe-fides -DFIDESLIB_INSTALL_PREFIX=$HOME/fideslib \
      -DFIDESLIB_ARCH=auto -DFIDESLIB_COMPILE_TESTS=OFF -DFIDESLIB_COMPILE_BENCHMARKS=OFF
cmake --build build -j && cmake --install build && cd ..

# fheml on the GPU
cmake -S . -B build-gpu -DFHEML_GPU=ON -DCMAKE_PREFIX_PATH="$HOME/openfhe-fides;$HOME/fideslib"
cmake --build build-gpu -j
./build-gpu/test_fheml        # the context lines print device=gpu
```

`ContextOptions::use_gpu` (default `true`) and `gpu_devices` (default `{0}`)
choose the device. `FHEContext::on_gpu()` reports what was picked. From
Python, use `fheml.make_context(net, use_gpu=False)`, `ctx.on_gpu` and
`fheml.gpu_device_count()`.

To retrain the models (numpy only; MNIST is downloaded into `data/`):

```sh
python3 tools/train_mnist.py --out models/mnist_cnn.fheml --samples models/mnist_test_samples.txt
python3 tools/train_transformer.py --out models/mnist_transformer.fheml
```

## Python

The Python module is built automatically when pybind11 is installed
(`pip install pybind11`). It ends up in `build/python/fheml`.

```sh
export PYTHONPATH=$PWD/build/python
python3 examples/python/mnist_cnn.py --n 3
python3 examples/python/mnist_transformer.py --n 3 [--encrypt-weights]
```

```python
import numpy as np, fheml

n, d = 16, 32
net = fheml.Network(fheml.Shape(n, 1, d))
net.add(fheml.Residual([fheml.Attention(n, d, wq, wk, wv, wo, score=[0.03, 0.03, 0.008])]))
net.add(fheml.Residual([fheml.TokenDense(n, d, 64, w1, b1), fheml.Poly([0, 0.5, 0.25]),
                        fheml.TokenDense(n, 64, d, w2, b2)]))
net.add(fheml.TokenMean(n, d)).add(fheml.Dense(d, 10, wh, bh))
net.compile()
ctx = fheml.make_context(net)              # 128-bit CKKS + exactly the keys net needs
logits = fheml.infer(net, ctx, x)          # x: (n, d) numpy array; encrypt -> run -> decrypt
assert np.allclose(logits, fheml.plain(net, x), atol=1e-3)
```

Weight arguments can be numpy arrays of any shape. They are flattened in C order, so dense weights are `[out][in]`. `tests/test_python.py` is a complete runnable version of this snippet.

## PyTorch

Build the model from `nn.Sequential`, the supported `torch.nn` layers and
`fheml.nn`, train it as usual, and export it. torch is only needed for this
(`pip install torch`).

```python
import torch, fheml
from torch import nn
from fheml import nn as hnn

model = nn.Sequential(
    nn.Conv2d(1, 5, 5, stride=2, padding=1), nn.BatchNorm2d(5), hnn.Poly([0, 0.5, 0.25]),
    nn.Flatten(), nn.Linear(845, 64), hnn.Poly([0, 0.5, 0.25]), nn.Linear(64, 10))
# ... train ...
net = fheml.from_torch(model, (1, 28, 28))   # also checks fheml == torch on a random input
net.compile()
ctx = fheml.make_context(net)
logits = fheml.infer(net, ctx, image)        # image: (1, 28, 28) numpy array
```

| torch | fheml |
| --- | --- |
| `nn.Conv2d` (square kernel, zero padding, groups=1) | `conv2d` |
| `nn.Linear` | `dense` on vectors; `tokendense` on `(tokens, dim)` inputs |
| `nn.AvgPool2d`, `nn.AdaptiveAvgPool2d(1)` | `avgpool2d` |
| `hnn.MaxPool2d(k, stride, range=R)` (square window, no padding) | `maxpool2d` (exact, scheme switching) |
| `nn.BatchNorm2d` / `nn.BatchNorm1d` | folded into the preceding conv/linear, else `affine` |
| `nn.Flatten`, `nn.Dropout`, `nn.Identity` | nothing |
| `hnn.Poly`, `hnn.Square`, `hnn.Chebyshev` | `poly`, `square`, `chebyshev` |
| `hnn.Attention(dim, tokens, heads=H)`, `hnn.Residual`, `hnn.TokenMean`, `hnn.PosEmbedding` | `attention` (single or multi-head), `residual`, `tokenmean`, per-token bias |
| `hnn.Attention(..., softmax={...})` | `attention` with softmax scores (`SoftmaxOptions`) |
| `hnn.LayerNorm(dim, eps, var_min, var_max, degree, newton)` | `layernorm` |
| `hnn.Bootstrap` | `bootstrap` |

`nn.LayerNorm` and `nn.Softmax` have no exact encrypted form and are
rejected with a message naming the replacement (`hnn.LayerNorm`,
`hnn.Attention(softmax=...)`). `nn.ReLU` is rejected too unless
`from_torch(..., exact_relu=RANGE)` is given: it then becomes the exact
scheme-switching `ExactReLU` (see below; `hnn.Chebyshev("relu", ...)` modules
too); otherwise use `hnn.Chebyshev`. A plain `nn.MaxPool2d` is
rejected too because the encrypted comparison needs an input bound:
`hnn.MaxPool2d(k, stride, range=R)` is the same torch layer with that bound
and exports to the exact `MaxPool2d` (see below), at 15+ levels and a FHEW
bootstrap per slot and comparison. `hnn.Chebyshev`, `hnn.LayerNorm` and the
softmax of `hnn.Attention` evaluate the same polynomial approximations in
torch that run encrypted, so training sees what inference computes (and
pushes the model to keep its variances and scores inside the intervals). For token inputs the exporter pads
each token to the widest per-token layer, which makes those layers much cheaper
under encryption; `fheml.infer` pads inputs to match.

`examples/python/train_torch_mnist.py` trains, exports and runs both MNIST
models encrypted. With 128-bit security:

* **CNN:** 98.4% after 3 epochs, about 3 s per encrypted image.
* **ViT:** 94.8% after 15 epochs, about 16 s per encrypted image.
* **Agreement:** the encrypted logits match torch to about 1e-6.

```sh
PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch cnn --n 3
PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch vit --out models/vit_torch.fheml
```

## Max pooling via scheme switching

Max pooling is exact, not approximated: `MaxPool2d(k, stride, range)` compares
window entries through OpenFHE's CKKS ↔ FHEW scheme switching
(`cc->Enable(SCHEMESWITCH)`, `EvalCompareSchemeSwitching`). Each tap of the
window is rotated onto the window's top-left slot, then the taps are reduced
in a tournament: `max(a, b) = a + [a < b] (b - a)`, where `[a < b]` is
computed by switching `a - b` to FHEW, evaluating the sign there with one
FHEW bootstrap per slot and switching back. The layer needs a context built
with `ContextOptions::scheme_switching` (`Network::context_options()` sets it
when the network contains one); `keygen()` then also generates the FHEW
context, its bootstrapping keys and the switching keys. `fheml.MaxPool2d`
and `ctx.compare(a, b, range)` expose the same thing in Python.

```cpp
Network net(Shape{2, 4, 4});
net.emplace<MaxPool2d>(2, 2, /*range=*/1.0);   // inputs must lie in [-range, range]
net.compile();
FHEContext ctx(net.context_options(/*secure=*/true));
ctx.keygen(net.rotation_indices());
```

What it costs (measured, OpenFHE 1.4.2, 4 cores):

| | TOY FHEW, ring 2048, 16 slots | STD128 FHEW, ring 2^16, 256 slots |
|---|---|---|
| key generation | 2 s | 99 s |
| one comparison (all slots) | 1.8 s | 470–600 s |
| peak memory | < 1 GB | 11.4 GB |
| error of the max | ~1e-6 | ~2e-3 |

A `k × k` pool runs `k² − 1` comparisons. Each comparison result is a fresh
CKKS ciphertext at `FHEContext::compare_depth()` = 12 levels whatever the
inputs' levels, so the pool output sits at `12 + ceil(log2 k²)` levels and the
layer reserves one more (a comparison needs its inputs two levels above the
bottom of the chain): a 2×2 pool is a 15-level layer, a 3×3 pool 17. With
bootstrapping budgets of 20–26 levels at ring 2^16 it takes about one
bootstrap per pool, which is why the example models do not use it.
`hnn.MaxPool2d` exports it from PyTorch, and `Trainer` backpropagates through
it with the forward's comparison bits (see [Encrypted training](#encrypted-training)). The comparison resolves differences down to about
`4·range / 2^(fhew_log_q − 1)`; a smaller difference is decided either way,
which moves the maximum by at most that much. Values outside `[-range, range]`
wrap around in the FHEW plaintext space and compare wrongly, so calibrate
`range` like a Chebyshev interval.

The output stays gapped at the input positions like a strided convolution;
the slots between outputs hold maxima of other windows (bounded by `range`),
which the next linear layer ignores. `place_bootstraps` cannot fold a scale
through the layer yet and inserts a `scale` layer when it has to; the FIDESlib
(GPU) backend does not support scheme switching.

## Exact ReLU via scheme switching

`ExactReLU(range)` computes `max(x, 0)` without a polynomial: the sign
`[x > 0]` comes from the same CKKS → FHEW → CKKS comparison as max pooling
(`FHEContext::positive(x, range)`, i.e. `compare(0, x)` with the FHEW scaling
set for a difference of at most `range`), and the layer returns
`x · [x > 0]`. It is an option, not the default: the Chebyshev ReLU stays in
the models because it is two to three orders of magnitude faster (table
below).

```cpp
net.emplace<ExactReLU>(/*range=*/4.0);   // inputs must lie in [-range, range]
net.use_exact_relu();                     // or: swap every Chebyshev relu for one (keeps the interval)
```

```python
net = fheml.from_torch(model, shape, exact_relu=4.0)   # nn.ReLU / hnn.Chebyshev("relu") -> ExactReLU
net.use_exact_relu()                                    # on an existing fheml.Network
```

`./build/relu_bench [--slots N] [--range R] [--insecure]` measures both against
the true ReLU on the same inputs in `[-4, 4]` (each variant in its own context
with just the levels it needs; 4 cores, OpenFHE 1.4.2):

| 128-bit, ring 2^15, 256 slots | levels | per call | setup + keys | max error vs ReLU | error for \|x\| > 0.1 |
|---|---|---|---|---|---|
| Chebyshev degree 15 | 6 | 0.40 s | 0.3 s | 1.3e-1 | 7.8e-2 |
| Chebyshev degree 31 | 7 | 0.66 s | 0.3 s | 6.3e-2 | 2.0e-2 |
| Chebyshev degree 63 | 8 | 1.05 s | 0.4 s | 3.1e-2 | 4.8e-3 |
| `ExactReLU` (STD128 FHEW) | 13 (1 past a bootstrap) | 412 s | 80 s, 7.3 GB peak | 4.4e-3 | 4.4e-3 |

At toy parameters (`--insecure`) the exact ReLU takes about 1 s at 16 slots
and 13 s at 256 slots with an error of about 1e-5; the time grows linearly
with the slot count (one FHEW bootstrap per slot), the Chebyshev time does
not.

Error is the maximum over all slots against `max(x, 0)`; the Chebyshev
error peaks at the kink and shrinks only linearly with the degree, while the
exact ReLU is limited by CKKS noise everywhere except within the FHEW
resolution (about `2·range / 2^(fhew_log_q − 1)`) of zero, where either branch
is off by at most `|x|`.

Depth: the sign is a fresh ciphertext at `compare_depth()` = 12 levels
whatever the input's level, so from a fresh input the layer is 13 levels
deep. Past a CKKS bootstrap, whose own depth (18–20) is larger than 12, `x`
is already the deeper operand and the layer costs **one level**:
`place_bootstraps` / `fheml.auto_bootstrap` mark every `ExactReLU` that way
(`after_bootstrap`, saved in the model file) and the planner budgets 1 level
for it, against 5–6 for a degree-15 Chebyshev ReLU. That makes deep ReLU
networks need fewer CKKS bootstraps, but each ReLU still costs a FHEW
bootstrap per slot, which dominates the time.

Accuracy: a model trained with the Chebyshev ReLU loses accuracy when
switched (`mnist_cnn --model models/mnist_deepcnn.fheml --exact-relu` on 14×14
inputs, 200 test images: 98.0% → 95.5% in plaintext) because it learned the polynomial; train
with `nn.ReLU` and export with `exact_relu=` instead
(`train_torch_mnist.py --arch deepcnn --exact-relu`). Trained that way the
deep CNN reaches 97.5% (97.6% with the Chebyshev ReLU) and, with
`--bootstrap`, plans to 9 levels with a single CKKS bootstrap instead of 25
levels and 8 bootstraps; at toy parameters an encrypted image takes about
55 s (all from the four ReLUs over 256 slots) and its logits match torch to
about 8e-3. At 128-bit security the four ReLUs alone would take about
4 × 412 s ≈ 28 minutes per image, against seconds for the Chebyshev model. The trainer has no
backward pass for it yet; the FIDESlib backend has no scheme switching.

## Multiplexed packing

A gapped stride-2 output needs `C·H·W` slots where a compact one needs a
quarter of that, and each further stride-2 stage multiplies the waste by 4. A
ResNet-20-shaped CIFAR network (16/32/64 channels, two stride-2 stages) needs
65536 slots gapped, so ring 2^17. Compacting fixes the slots but turns a
strided conv into a dense mess of diagonals (15858 for `16->32 s=2`).
`Packing::kMultiplexed` interleaves `m²` channels into the gaps of a stride-`m`
output, as in Lee et al.'s multiplexed parallel convolutions: same slots as
compact, offsets still constant per (channel pair, tap). Any linear layer can
write any layout, so the BSGS plans need nothing new, and a conv after max
pooling re-packs the pool's gapped output for free. Each strided layer takes
the cheapest collision-free layout (multiplexed, gapped or compact), weighting
the BSGS cost by the slots it needs. Details in `docs/DESIGN.md`.

```cpp
net.set_packing(Packing::kMultiplexed);   // before compile(); default kGapped
```
```python
net.packing = "multiplexed"               # "gapped" | "compact" | "multiplexed"
```

Measured (OpenFHE 1.4.2, 4 cores; `examples/packing_bench`, `mnist_cnn --packing`):

| | gapped | compact | multiplexed |
|---|---|---|---|
| MNIST CNN, 128-bit (ring 2^15), per image | 2.41 s | 4.98 s | **1.80 s** |
| MNIST CNN, slots / rotation keys | 4096 / 55 | 1024 / 75 | 2048 / **40** |
| CIFAR stack, slots (ring) | 65536 (2^17) | 16384 (2^15) | **16384 (2^15)** |
| CIFAR stack, diagonals / rotation keys | 2657 / 136 | 26066 / 405 | 3822 / 146 |
| CIFAR stack, all linear layers encrypted | 28.0 s | 38.6 s | **8.6 s** |

The CIFAR row times each linear layer on its own (toy parameters, ring = 2 ×
slots, depth 2, only that layer's keys), so it compares the packings rather
than giving an end-to-end number; activations and bootstraps also get 4×
cheaper at ring 2^15 than at 2^17. The MNIST rows are full encrypted
inference at 128-bit security (3 images, all correct, logit error < 3e-6).
For the deep MNIST CNN (`mnist_deepcnn`) multiplexing keeps compact's 256
slots with 202 diagonals instead of 470 and 33 rotation keys instead of 49.
Multiplexed layouts also go through encrypted training (`test_train`).

Not done: residual blocks on images (the CIFAR stack has no skips), and the
layout is chosen greedily per layer, not searched over the whole network.

## LayerNorm and softmax

Both are polynomial, with intervals to calibrate like a Chebyshev
activation's (see `docs/DESIGN.md#softmax-and-normalisation`):

```cpp
net.emplace<LayerNorm>(n, d, gamma, beta, /*eps=*/1e-5, /*var_min=*/0.05, /*var_max=*/1.2,
                       /*degree=*/31, /*newton=*/1);          // 12 levels
SoftmaxOptions sm;                                            // scores q.k in [lo, hi]
sm.lo = -2; sm.hi = 2; sm.exp_degree = 15; sm.inv_degree = 31; sm.newton = 2;
attention->set_softmax(sm);                                   // 23 levels instead of 6
sm.subtract_max = true;                                       // exact row max via scheme switching: 33 levels for n = 4
```

`LayerNorm` computes the token mean and variance with rotations and inverts
`sqrt(var + eps)` with a Chebyshev series refined by Newton steps (two
levels each). Softmax attention evaluates a Chebyshev `exp` on every score
ciphertext and divides by the row sum with a Chebyshev `1/x` plus Newton
steps. With `subtract_max` the row maximum is first found with a tournament
of the scheme-switching comparisons above and subtracted, so the
exponentials are bounded by 1 and the sums lie in `[1, n]` with nothing to
calibrate; it costs n − 1 comparisons (each a FHEW bootstrap per slot) and
about 12 + log2(n) extra levels, which is beyond one bootstrapping budget at
ring 2^16. Both train under encryption (see below) and export from
`fheml.nn.LayerNorm` / `fheml.nn.Attention(softmax=...)`; the example models
do not use them yet.

## Encrypted training

`Trainer` trains a network on encrypted data. The client encrypts inputs and
one-hot targets; the server runs forward, backward and SGD updates on
ciphertexts and never sees the data, the labels, the gradients or the weights.
Only the secret-key holder can decrypt the trained model.

```python
net.compile()
trainer, ctx = fheml.make_trainer(net, frozen=[], secure=True)  # bootstraps the weights after each update
trainer.encrypt_weights(ctx)
for xs, ts in batches:
    fheml.train_step(trainer, ctx, xs, ts, lr=1.0)       # encrypt, accumulate gradients, update
trainer.decrypt_weights(ctx)                             # key holder: trained weights back into net
```

* **What trains:** conv2d, dense, tokendense and affine weights and biases,
  layernorm's gamma and beta, and the four projections of attention
  (quadratic or softmax scores, any number of heads), with shared weights
  handled exactly. Gradients flow through square, poly, chebyshev, average
  and max pooling, token mean, layernorm, attention, residual and bootstrap
  layers. Max pooling routes each window's gradient to the tap that won,
  multiplying by the comparison bits its forward pass already computed, so
  the backward pass adds no scheme switching (`ceil(log2 k²)` levels).
  The softmax and the layer norm's inverse square root are differentiated as
  the functions they approximate (`exp' = exp`, `(1/S)' = -1/S²`), so their
  gradients match the exact ones up to the approximation error.
* **What does not:** layers in a frozen prefix only run forward.
* **Fine-tuning:** `frozen=[...]` keeps layers fixed. Everything before the
  first trainable layer runs as ordinary encrypted inference.
* **Loss:** squared error against the encrypted target (default), or softmax
  cross-entropy against an encrypted one-hot target:
  `make_trainer(net, loss=fheml.cross_entropy(lo=-4, hi=4))`. The gradient
  `softmax(y) - t` is computed under encryption with the attention softmax's
  polynomials: exp as a Chebyshev series on the logit range `[lo, hi]`, a
  rotate-and-sum for the denominator, 1/x as a Chebyshev series plus Newton
  steps. With `subtract_max=True` the largest logit is found with
  scheme-switching comparisons first (log2 of the output span of them, 4 for
  10 classes), so exp lives on `[lo - hi, 0]`, the sum on `[1, k]` and nothing
  needs calibrating. The loss value itself stays with the client, who
  decrypts the logits anyway (`fheml.cross_entropy_plain`). Cross-entropy
  costs levels: the gradient sits 17 levels below the logits for degree
  15 / 31 with two Newton steps (step depth 27 instead of 8 for the test MLP; 20 with a `bootstrap` layer
  right before the loss, 17 with that and degree 7 / 7 on a calibrated
  `[sum_lo, sum_hi]`). That is fine with toy parameters, but more than the
  24 levels of ring 2^16 at 128-bit security without bootstrapping (28 for
  one update of the MLP at degree 7 / 7) and far more than the ~6 a
  bootstrap leaves there, so 128-bit cross-entropy training needs ring 2^17
  or bootstraps inside the softmax (not done yet).
* **Check:** `fheml.train_plain` runs the same SGD in the clear.
  `tests/test_train.cpp` shows the decrypted weights match it to about 1e-8
  without bootstrapping and 3e-4 with it, and checks the plaintext gradients
  against finite differences.

It is slow. Every weight diagonal is a ciphertext, every sample costs about
three ciphertext products and three rotations per diagonal, and each update
bootstraps all the weights. Bootstrapping packs many weight ciphertexts into
one, so with toy parameters the MNIST demo below takes about 15 s per update
of 16 images. `examples/python/train_encrypted_mnist.py` trains a
49→16→10 network on downsampled MNIST with encrypted images, labels and
weights, next to the same training in the clear:

```sh
PYTHONPATH=build/python python3 examples/python/train_encrypted_mnist.py --insecure            # from scratch
PYTHONPATH=build/python python3 examples/python/train_encrypted_mnist.py --insecure --fine-tune
```

With toy parameters, on 1000 test images:

| Run | Encrypted updates | Test accuracy before → after | Encrypted vs. clear weights |
| --- | --- | --- | --- |
| From scratch, all layers | 40 × 16 images, about 15 s each | 16.1% → 52.8% | match to 1.4e-3, same accuracy |
| Fine-tune last layer (after 300 clear pre-training steps) | 10 × 16 images, about 5 s each | 82.7% after | match to 7.8e-4, same accuracy |

The accuracy numbers only reflect how few updates were run, not the
encryption: the plaintext run with the same updates gets the same result.

### Training at 128-bit security

At 128-bit security the training context needs ring 2^16, where every
rotation key is about 100 MB. One key per weight diagonal and its negative
(105 keys, 10 GB for this small network) plus the bootstrapping keys did not
fit in 16 GB. `fheml.make_trainer(net, secure=True)` therefore uses a
**rotation-key ladder** (`rotation_window=1`): keys for ±2^i only, with every
other rotation composed from them (balanced binary digits, one key switch
per non-zero digit; see [DESIGN.md](docs/DESIGN.md#rotation-key-ladder)).
The bootstrapping keys depend on the slot count, so `slots=256` keeps them
small too (16 weight ciphertexts are packed per bootstrap instead of 2048).

Bootstrapping the weights needs 30 levels (10 for the step, 20 for the
bootstrap), which is ring 2^17 at 128-bit; without it, three updates fit
ring 2^16. Measured on 4 cores, 16 GB:

```sh
PYTHONPATH=build/python python3 examples/python/train_encrypted_mnist.py --no-bootstrap --updates 3 --batch 4 --test 200
```

| | Keys | Memory |
| --- | --- | --- |
| One key per rotation (63 rotations in the step) | 63 × 111 MB | about 7 GB of keys (computed from the measured key size) |
| Ladder, `rotation_window=1` | 11 keys | 1.2 GB of keys, 10 GB process peak with the ciphertexts |

The three updates of 4 images took 338 s, 244 s and 178 s (the weights sit
at deeper levels after each update, so the ciphertexts shrink); the
decrypted weights matched the plaintext SGD run to 3e-8. Composed rotations
cost 1-3 key switches each instead of one hoisted one, so this is slower
per sample than the toy run, but it is the first run of the trainer at
128-bit security. (These timings predate the hoisted rotation tree below.)

### Hoisted composed rotations

With a ladder, `rotate_many` evaluates a rotation tree: offsets that share
ladder steps share the rotated copies, and every copy rotated further is
decomposed once for all its successors (see
[DESIGN.md](docs/DESIGN.md#rotation-key-ladder)). `./build/rotation_bench`
compares it with rotating each offset on its own and with the previous
`rotate_many` (nearest earlier copy, no hoisting), on the baby steps of the
`conv_demo` CNN and a 16-token attention. 128-bit, ring 2^15, depth 8, 4
cores, mean of 3 (ks = key switches, dec = decompositions):

| `rotation_window=1` | independent | nearest copy | hoisted tree |
| --- | --- | --- | --- |
| conv 3x3, 9 offsets | 32 ks, 1.15 s | 10 ks, 0.58 s | 10 ks + 6 dec, 0.52 s |
| strided conv 3x3, 9 offsets | 28 ks, 0.97 s | 9 ks, 0.49 s | 9 ks + 5 dec, 0.49 s |
| dense head, 7 offsets | 9 ks, 0.29 s | 6 ks, 0.25 s | 6 ks + 3 dec, 0.24 s |
| attention, 16 token offsets | 32 ks, 0.89 s | 15 ks, 0.46 s | 15 ks + 7 dec, 0.39 s |

A key switch took 56-63 ms, of which the decomposition was 20-22 ms, so a
tree saves at most about a third over the nearest-copy plan; it saves 0-20%
of wall time here (with direct keys all three are one hoisted fan-out, 1
decomposition). Two encrypted training updates (4 images, `rotation_window=1`,
insecure ring 2^14) went from 48 s / 36 s to 46 s / 32 s, and fine-tuning
the last layer (whose frozen backward is now hoisted too) from 7.0 s / 5.9 s
to 5.9 s / 5.3 s.

## Deep networks

A CKKS ring holds a fixed number of levels (24 at ring 2^16 with 50-bit
scaling and 128-bit security), and every linear layer and activation consumes
some. Deeper networks need bootstrapping, which itself costs 18-20 levels and
needs its input within [-1, 1]. `fheml.auto_bootstrap(net, calibration)`
handles both:

1. `fold_activation_intervals()`: the affine map from a Chebyshev
   activation's interval `[a, b]` to `[-1, 1]` moves into the preceding
   linear layer's weights and bias, saving one level per activation.
2. `place_bootstraps(budget, calibration)`: bootstraps go in as late as
   possible so that no stretch exceeds `budget` levels (the most that keeps
   the ring at 2^16: 3 with level budget {3,3} and 5 with {2,2} at 50-bit
   scaling, two more with `key_switch_digits=4`). The calibration inputs
   are run in the clear, the
   values entering each bootstrap are divided by `1.5 × max|x|`, and that
   scale is folded into the neighbouring layers for free (weights,
   polynomial coefficients, Chebyshev intervals, through squares and
   pooling); only a residual block or attention needs a fixed `scale` layer,
   which costs a level the planner budgets.

`examples/python/train_torch_mnist.py --arch deepcnn --bootstrap` trains a
CNN with three conv blocks, a 32-unit dense block and Chebyshev ReLUs (30
levels, 25 after folding; 97.6% on MNIST at 14x14) in PyTorch and runs it
encrypted with the ladder keys:

```sh
PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch deepcnn --bootstrap --window 1 --level-budget 3 --digits 4 --n 3
```

Measured at 128-bit security on 4 cores (ring 2^16, 256 slots, level budget
{3,3}, `key_switch_digits=4` for 26 levels, 50-bit scaling):

| | |
| --- | --- |
| Bootstraps placed | 8, at most 5 levels between them (budget 5) |
| Rotation keys | 47 (6.2 GB): 32 bootstrapping keys plus the 15-key ladder, no per-layer keys |
| Process peak | 9.9 GB (context, keys and setup 28 s) |
| Time per image | 147 s, of which 8 bootstraps take 134 s (16-17 s each) |
| Accuracy | both test images classified like the plaintext model; logits within 0.28 of plaintext (logit range 8) |

Each bootstrap adds about 1e-2 of error to values in [-1, 1] at these
parameters (50-bit scaling is the largest that fits; a 55-bit first modulus
instead of 57 changes nothing), so eight of them leave the logits 3% off.
That is the precision budget for deeper networks at ring 2^16; a larger
ring buys 55-bit scaling and far better bootstraps at twice the memory.

The same bootstraps work inside encrypted training: the trainer bootstraps
the activation on the way forward and the gradient on the way back.

## Using the library

```cpp
#include "fheml/fheml.h"
using namespace fheml;

Network net(Shape{3, 32, 32});
net.emplace<Conv2d>(3, 16, 3, 3, /*stride=*/1, /*pad=*/1, weights, bias)
   .emplace<ChebyshevActivation>("relu", -4.0, 4.0, 59)
   .emplace<AvgPool2d>(2, 2)
   .emplace<Bootstrap>()
   .emplace<Dense>(16 * 16 * 16, 10, w2, b2);
net.compile();                                   // BSGS plans, slots, rotations

FHEContext ctx(net.context_options());           // 128-bit CKKS sized for the network
ctx.keygen(net.rotation_indices());

// optional: net.encrypt_weights(ctx);         // ciphertext weights from here on
Ctxt y = net.forward(ctx, ctx.encrypt(image, net.input_level(ctx)));
std::vector<double> logits = net.decrypt_output(ctx, y);
```

`Network::load` / `save` use a small text format that is documented in
[`include/fheml/network.h`](include/fheml/network.h).
`Network::fuse_linear()` merges adjacent linear layers, saving one level each.
`ctx.rotate(c, k)` rotates by any `k`, composing it from the ladder keys when
`ContextOptions::rotation_window` is set (then pass
`net.rotation_indices(true)` to `keygen`, which lists only the rotations that
still need their own key).

## Layout

| Path | Contents |
| --- | --- |
| `include/fheml/`, `src/` | the library: backend selection (`backend.h`: OpenFHE or FIDESlib), context, linear transforms, layers, network, encrypted training (`train.h`) |
| `examples/` | `mnist_cnn` (encrypted MNIST), `conv_demo` (network built in code), `packing_bench` (gapped vs compact vs multiplexed packing) and `rotation_bench` (hoisted rotation trees vs. plain composed rotations) |
| `tests/` | `test_fheml`: BSGS plans vs. exact products, encrypted vs. plaintext per layer, depth accounting, scheme-switching comparison and max pooling; `test_train`: encrypted training vs. plaintext SGD and finite differences; `test_python.py`, `test_torch.py`, `test_train.py` |
| `python/` | pybind11 bindings and the `fheml` Python package (`fheml.nn` + `fheml.from_torch`: PyTorch export) |
| `examples/python/` | encrypted MNIST CNN and transformer from Python; `train_torch_mnist.py` trains in torch and runs encrypted; `train_encrypted_mnist.py` trains on encrypted data |
| `tools/train_mnist.py`, `tools/train_transformer.py` | HE-friendly CNN / transformer training and export (numpy) |
| `models/` | trained example models and 200 test images |
| `legacy/` | the original PALISADE capstone code |

The tests and `--insecure` runs use `HEStd_NotSet` toy parameters, which are
fast but not secure. Everything else defaults to 128-bit security.

2022 Capstone Team 49 (original prototype).
