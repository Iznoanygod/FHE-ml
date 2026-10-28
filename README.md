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
* **Dense heads use the Gazelle hybrid method.** Partial sums are folded into
  at most `2R-1` diagonals, plus `log2(S/R)` rotate-and-sum steps.
  * The MNIST `845->64` layer needs 126 diagonals instead of 908.
* **Activations run entirely under encryption.**
  * `square` (CryptoNets).
  * Trained polynomials `poly`.
  * Chebyshev approximations of `relu` / `silu` / `gelu` / `sigmoid` / `tanh`
    on a chosen interval and degree, via `EvalChebyshevSeries`.
* **Small transformers.**
  * `TokenDense` is a per-token linear layer.
  * `Attention` is fully homomorphic self-attention: ct×ct QKᵀ and AV, with
    a trainable quadratic in place of softmax.
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
* **Bootstrapping layers** let you build deep networks.
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
| `nn.BatchNorm2d` / `nn.BatchNorm1d` | folded into the preceding conv/linear, else `affine` |
| `nn.Flatten`, `nn.Dropout`, `nn.Identity` | nothing |
| `hnn.Poly`, `hnn.Square`, `hnn.Chebyshev` | `poly`, `square`, `chebyshev` |
| `hnn.Attention`, `hnn.Residual`, `hnn.TokenMean`, `hnn.PosEmbedding` | `attention`, `residual`, `tokenmean`, per-token bias |
| `hnn.Bootstrap` | `bootstrap` |

`nn.ReLU`, softmax, LayerNorm and max pooling have no exact encrypted form
and are rejected with a message naming the replacement. `hnn.Chebyshev`
evaluates the same polynomial approximation in torch that runs encrypted,
so training sees what inference computes. For token inputs the exporter pads
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
  with shared weights handled exactly. Gradients flow through square, poly,
  chebyshev, average pooling, token mean, residual and bootstrap layers.
* **What does not:** attention layers can't be trained or backpropagated
  through yet. They can sit in a frozen prefix, which only runs forward.
* **Fine-tuning:** `frozen=[...]` keeps layers fixed. Everything before the
  first trainable layer runs as ordinary encrypted inference.
* **Loss:** squared error against the encrypted target. Softmax
  cross-entropy would need an encrypted softmax.
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
At 128-bit security the same training needs ring 2^16 and one rotation key
per weight diagonal (about 100 MB each), which does not fit in 16 GB yet (see
[DESIGN.md](docs/DESIGN.md#encrypted-training)).

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

## Layout

| Path | Contents |
| --- | --- |
| `include/fheml/`, `src/` | the library: backend selection (`backend.h`: OpenFHE or FIDESlib), context, linear transforms, layers, network, encrypted training (`train.h`) |
| `examples/` | `mnist_cnn` (encrypted MNIST) and `conv_demo` (network built in code) |
| `tests/` | `test_fheml`: BSGS plans vs. exact products, encrypted vs. plaintext per layer, depth accounting; `test_train`: encrypted training vs. plaintext SGD and finite differences; `test_python.py`, `test_torch.py`, `test_train.py` |
| `python/` | pybind11 bindings and the `fheml` Python package (`fheml.nn` + `fheml.from_torch`: PyTorch export) |
| `examples/python/` | encrypted MNIST CNN and transformer from Python; `train_torch_mnist.py` trains in torch and runs encrypted; `train_encrypted_mnist.py` trains on encrypted data |
| `tools/train_mnist.py`, `tools/train_transformer.py` | HE-friendly CNN / transformer training and export (numpy) |
| `models/` | trained example models and 200 test images |
| `legacy/` | the original PALISADE capstone code |

The tests and `--insecure` runs use `HEStd_NotSet` toy parameters, which are
fast but not secure. Everything else defaults to 128-bit security.

2022 Capstone Team 49 (original prototype).
