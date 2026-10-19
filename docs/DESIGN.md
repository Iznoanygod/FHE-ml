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

## Transformers

### Layout

A sequence of n tokens with d features is the Shape `{n, 1, d}`. Feature j of
token t sits in slot `t*B + j`.

- The token stride B ≥ d is set by the input: pad every input token to the
  widest layer, which is 64 in the MNIST model.
- `TokenDense` keeps that stride whenever its output fits. `kron(I_n, W)`
  then has only `in + out - 1` diagonals. Without the padding, the MNIST
  model's 32→64 layer had 575 diagonals instead of 95, and inference took
  26 s instead of 15 s.
- `Residual` forces its last linear layer to write in the residual's layout.

### Attention

There is one ciphertext in and one out, using the diagonal-over-offsets method
(NEXUS / Powerformer style). For each token offset s in 0..n-1:

1. Rotate K and V by `s*B`. Both were first extended with a cyclic copy at
   `[n*B, 2n*B)`, so the rotation maps token i to token (i+s) mod n, and the
   rotations are hoisted.
2. Multiply Q ⊙ K_s, then sum each token's d-window with log₂ d rotations.
3. Apply the masked score polynomial `mask · (c0 + c1 p + c2 p²)`, with the
   mask folded into the coefficients. This costs 2 levels.
4. Broadcast back over the token with log₂ d rotations.
5. Multiply by V_s.

- **Cost:** about n·(2 log₂ d + 2) rotations and 3n ct×ct products.
- **Depth:** 6 for the block, including the four projections.

### Softmax and normalisation

- Softmax is replaced by the fixed quadratic `A = (S+2)²/(8n)`. It is
  non-negative, with a public normaliser 1/n, as in PowerSoftmax and
  Powerformer's plaintext normaliser. 1/√d is learned into W_q.
- LayerNorm is omitted. The toy model trains fine without it.

### Next steps

- Multiple heads, which fit as sub-blocks of each token.
- Normalised power attention, using a Chebyshev 1/x on the rotation-free sum
  of the per-offset score ciphertexts.
- LayerNorm via a Chebyshev 1/√x on a training-calibrated range.
- Bootstrapping between blocks for deeper models.

## Encrypted weights

`Network::encrypt_weights(ctx)` switches every linear layer to ciphertext
weights.

**How the weights are encrypted.**
- It first runs one inference on an encrypted zero vector. That records the
  level at which each layer's products and bias addition happen.
- Each pre-rotated diagonal is then encrypted at its layer's level, so it
  carries no more towers than needed. The bias is encrypted one level lower,
  because the sum of products is rescaled before the bias is added.

**How evaluation changes.**
- Products use `EvalMultNoRelin`.
- Relinearization runs once per giant step, not once per product.
- Depth is unchanged.

**Scope of the protection.**
- Polynomial activations and average pooling are public functions, not
  trained weights, so they stay in the clear.
- The weights are encrypted under the evaluation key, as in the 2022
  prototype. This hides them from the evaluating server but not from the key
  holder.
- Hiding the model from the client and the input from the model owner
  requires threshold or multi-party CKKS. OpenFHE supports it, and it is on
  the roadmap.

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
- The same CNN with encrypted weights: about 3.6 s per image.
- A one-block MNIST vision transformer (95.6%): about 15 s per image.
- Python bindings for all of the above.
- PyTorch export (`fheml.from_torch`): torch-trained CNN and ViT run
  encrypted with logits matching torch to about 1e-6.

## Roadmap (suggested order)

1. **Client/server split.**
   - Serialize the context, public, relinearization and rotation keys and
     ciphertexts (OpenFHE `Serial::`) so the server never holds the secret
     key.
   - Rebuild the legacy socket client/server on top of that.
2. **Model privacy against the key holder.** Encrypted weights exist, but
   they share the client's key.
   - Use OpenFHE threshold CKKS so that neither party can decrypt the other's
     data alone.
   - Serialize the encrypted weights so the server never sees them in the
     clear.
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
7. **More PyTorch export.** `fheml.from_torch` covers `nn.Sequential` models
   (see the README). Next: tracing arbitrary `forward` code with `torch.fx`,
   multi-head attention, and a range penalty during training so Chebyshev
   activations stay inside their interval.
8. **Performance.**
   - Reduce `scale_mod_bits` / depth so the MNIST net fits ring 2^14.
   - Use `EvalFastRotationExt` to keep giant-step sums in the extended basis.
   - Benchmark threading.
