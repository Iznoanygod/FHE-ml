# Design and roadmap

## Goals

The goal is CNN inference where only the client holds the secret key. The
server evaluates the whole network, including activations, with no
decrypt/re-encrypt round trips. The 2022 prototype needed those round trips
for its sigmoid.

## Data layout

- A tensor of shape (C, H, W) lives in **one ciphertext**. Element (c, y, x)
  is in slot `c*cs + y*ys + x*xs` (`Layout`).
- The input is compact: `cs = H*W`, `ys = W`, `xs = 1`, the same order as
  `torch.flatten`.
- Each layer picks its output layout:
  - Elementwise layers keep the layout they are given.
  - Convolution and average pooling with stride `s` output
    `{cs_in, s*ys_in, s*xs_in}`. Their results stay on the grid of the input
    positions they were computed at ("gapped").
  - Dense and fused layers output compact vectors.
- `Network::required_slots()` is the next power of two that covers every
  layout's span.

**Why gapped outputs.** In a compact layout, a strided convolution's matrix has
a different offset `col - row` at every output pixel. The MNIST conv needed
1024 diagonals. A gapped layout keeps offsets constant per (channel pair,
kernel tap): 125 diagonals for the same layer. It needs more slots
(Co·H·W instead of Co·H/s·W/s). That costs nothing until it forces a larger
ring.

## Linear layers → one BSGS transform

`LinearLayer::compile` works in three steps:

1. **Build the slot matrix.** It takes the layer's logical matrix (rows = output
   elements, columns = input elements) and moves rows and columns to their
   slots.
2. **Choose the evaluation method.** Two candidates:
   - **Diagonal method:** `y = Σ_d diag_d ⊙ rot_d(x)`. The `d` are split into
     baby and giant steps, with the split chosen by search to minimise
     rotations. All baby steps share one hoisted precomputation, and the
     diagonals are pre-rotated by `-g` in the clear.
   - **Folded/hybrid method:** used when the output is compact with `n` rows,
     where `R = next_pow2(n)`. Output `r` accumulates partial sums in slots
     `r + jR`, with `j = col_slot / R`. All diagonals then fall in `(-R, R)`,
     and `log2(S/R)` rotate-and-sum steps finish the job.

   Whichever has the lower estimated cost is used.
3. **Add the bias** as a plaintext placed in the output slots.

Each linear layer costs one level. `Network::fuse_linear()` multiplies runs of
adjacent linear layers (conv → pool → affine) into one matrix. That saves a
level per merged layer, at the cost of more diagonals.

`LinearTransform::eval_slots` runs the same schedule on cleartext vectors. The
tests use it to check every plan against the exact product without any
cryptography.

## Activations

| Layer | OpenFHE call | Levels |
| --- | --- | --- |
| `square` | `EvalSquare` | 1 |
| `poly` degree 2 | `x·(c2·x + c1) + c0` | 2 (1 if c2 = 1) |
| `poly` degree d > 2 | `EvalPoly` | `poly_depth(d)` |
| `chebyshev f a b d` | `EvalChebyshevSeries` | `poly_depth(d)` (+1 unless [a,b] = [-1,1]) |

The plaintext reference evaluates the *same* approximation
(`EvalChebyshevFunctionPtxt`). Encrypted-vs-plaintext error therefore measures
only the CKKS noise. The approximation error is a training-time concern.

## Parameters and bootstrapping

- `Network::context_options()` sets:
  - `mult_depth`: the total depth, or the longest stretch between bootstraps
    when the network bootstraps.
  - `slots`.
  - 128-bit security.
- `FHEContext` uses `FLEXIBLEAUTO` rescaling and `HYBRID` key switching, and
  takes the smallest ring dimension that is secure and has at least 2·slots.
- `keygen()` generates exactly the rotation keys the compiled plans list.
- **Bootstrapping.** A `Bootstrap` layer calls `EvalBootstrap`.
  - Inputs are encrypted at `Network::input_level()`, so the first segment
    gets exactly the levels it needs and the first bootstrap really refreshes.
    OpenFHE skips a bootstrap whose input has more levels left than it would
    produce.
  - Slot values must be roughly within [-1, 1]. Scale the preceding linear
    layer's weights for that.

## Status

All of the following are tested under encryption:
- Convolutions with stride, padding and multiple channels.
- Dense layers, average pooling and channel affine layers.
- Fused linear layers.
- The square, polynomial and Chebyshev activations.
- Bootstrapping in the middle of a network.
- The trained MNIST CNN at 128-bit security: about 3 s per image, with
  predictions identical to plaintext.

## Roadmap (suggested order)

1. **Client/server split.**
   - Serialize the context, public, relinearization and rotation keys and
     ciphertexts (OpenFHE `Serial::`) so the server never holds the secret
     key.
   - Rebuild the legacy socket client/server on top of that.
2. **Encrypted weights.** The legacy README promised encrypted model weights.
   - The same BSGS schedule works with ciphertext diagonals: ct×ct
     multiplication plus relinearization.
   - Add a `LinearTransform` mode that stores encrypted diagonals.
3. **Multiplexed packing (Lee et al. ICML'22, Orion).** After stride-2 layers,
   re-pack gapped outputs densely so deep networks (ResNet-20 on CIFAR-10) fit
   in fewer slots and need fewer bootstraps.
4. **Better activations.**
   - Composite minimax sign/ReLU, built by composing `EvalChebyshevSeries`
     calls.
   - Fold the Chebyshev interval map into the preceding linear layer, which
     saves one level per activation.
   - Training-time range calibration.
5. **Automatic bootstrap placement** (Orion-style shortest path over levels),
   and per-segment slot counts.
6. **Batch-norm and residual connections.**
   - Fold BN into the convolutions at load time.
   - Add a residual `Add` layer that aligns levels and layouts.
7. **Training/export from PyTorch.** An exporter for `torch.nn.Sequential`
   models with polynomial activations, next to the numpy trainer.
8. **Performance.**
   - Reduce `scale_mod_bits` / depth so the MNIST net fits ring 2^14.
   - Use `EvalFastRotationExt` to keep giant-step sums in the extended basis.
   - Benchmark threading.
