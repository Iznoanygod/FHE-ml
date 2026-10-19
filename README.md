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
v1.4.2).

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
| `include/fheml/`, `src/` | the library: context, linear transforms, layers, network |
| `examples/` | `mnist_cnn` (encrypted MNIST) and `conv_demo` (network built in code) |
| `tests/` | `test_fheml`: BSGS plans vs. exact products, encrypted vs. plaintext per layer, depth accounting |
| `python/` | pybind11 bindings and the `fheml` Python package |
| `examples/python/` | encrypted MNIST CNN and transformer from Python |
| `tools/train_mnist.py`, `tools/train_transformer.py` | HE-friendly CNN / transformer training and export (numpy) |
| `models/` | trained example models and 200 test images |
| `legacy/` | the original PALISADE capstone code |

The tests and `--insecure` runs use `HEStd_NotSet` toy parameters, which are
fast but not secure. Everything else defaults to 128-bit security.

2022 Capstone Team 49 (original prototype).
