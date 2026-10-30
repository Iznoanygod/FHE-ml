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

#### Multiple heads

`Attention(..., heads=H)` splits each token's d features into H heads of
width dh = d/H (a power of two), laid out as the sub-blocks
`[t·B + h·dh, t·B + (h+1)·dh)` of token t. That is exactly what the
concatenated Q/K/V projections `kron(I_n, W)` already produce, and the head
concatenation before W_o is the same slot order, so no split or concat
rotations are needed at all. Only three things change in the single-head
schedule:

1. The window sum after Q ⊙ K_s runs over dh slots (log₂ dh rotations), which
   puts head h's score q_i·k_{i+s} in slot `i·B + h·dh`.
2. The score mask selects slot `i·B + h·dh` for every head. Since the mask
   carries the coefficients, each head can have its own polynomial for free.
3. The broadcast runs over dh slots. The window behind each slot holds at
   most one masked head slot, so head h's score only reaches its own
   sub-block (the zero padding past d keeps the last head from reading the
   next token).

So H heads cost n·(2 log₂(d/H) + 2) rotations instead of n·(2 log₂ d + 2):
fewer than one wide head, with the same 3n products, the same depth 6 and
fewer rotation keys. The model file writes `attention N D HAS_BIAS heads H`
with 3H score coefficients; single-head files are unchanged.

### Softmax and normalisation

- By default softmax is replaced by the fixed quadratic `A = (S+2)²/(8n)`.
  It is non-negative, with a public normaliser 1/n, as in PowerSoftmax and
  Powerformer's plaintext normaliser. 1/√d is learned into W_q. The example
  models use this (depth 6 per attention block).
- `Attention::set_softmax(SoftmaxOptions)` gives real softmax scores:
  `exp` is a Chebyshev series on the score interval `[lo, hi]`, evaluated on
  each per-offset score ciphertext; the row sum `D = Σ_s a_s` is free (the
  broadcast weights are added) and inverted with a Chebyshev series of 1/x
  on `[sum_lo, sum_hi]` plus Newton steps `y ← y (2 − D y)` (two levels
  each); the attention output is multiplied by the inverse once. Slots
  outside the head sub-blocks see a sum of 0, so a plaintext fills them with
  a value inside the interval before the series (a Chebyshev series evaluated
  far outside its interval is astronomically large and spoils the
  decryption of every slot). Depth 7 + poly_depth(exp) + poly_depth(1/x)
  + 2·newton: 17 with degree 15, 15, one step; 23 with a degree-31 1/x and
  two steps. The plaintext reference evaluates the same approximations.
- `subtract_max` is the numerically stable softmax: the row maximum is
  found with a tournament of scheme-switching comparisons over the n score
  ciphertexts (`FHEContext::compare`, n − 1 comparisons) and subtracted
  before the exponential. Then the exponentials live on `[lo − hi, 0]` and
  the sums on `[1, n]`, so no calibration of the sum is needed and nothing
  can overflow. The comparison results arrive at 12 levels whatever the
  input level, so the block costs about 12 + log2(n) + 17 levels (33 for
  the test's n = 4) and needs a deeper ring or a bootstrap it cannot place
  itself; and each comparison is a FHEW bootstrap per slot (minutes at
  128-bit security). It is the exact method; the calibrated-range softmax
  is the practical one.
- `LayerNorm` normalises each token over its d features: the mean and
  E[x²] are token-block sums (log2(d) rotations, one masked product by 1/d,
  log2(d) rotations to broadcast), the variance is E[x²] − mean², and
  1/√(var + ε) is a Chebyshev series on a calibrated variance interval
  refined by Newton steps `r ← 1.5 r − 0.5 v r³`. The 1.5 is folded into the
  series scale and the 0.5 rides on the variance, so a step costs two
  levels and no scalar products. Degree 15 with one step is exact to ~1e-9
  when var_max / var_min ≤ 10; degree 31 covers a 100× range. Depth
  2 + poly_depth(degree) + 1 + 2·newton + 1 (10 for degree 15, 12 for
  degree 31). The layer absorbs an input scale (eps and the interval scale
  by c²) and emits an output scale (gamma, beta), so `place_bootstraps`
  folds through it. Padding slots get a variance inside the interval for the
  same reason as above.

### Next steps

- Bootstrapping inside the max-subtracted softmax block.
- Bootstrapping between blocks for deeper models.
- An example model that uses them (both train under encryption and export
  from `fheml.nn.LayerNorm` / `fheml.nn.Attention(softmax=...)`).

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

## Encrypted training

`Trainer` (`include/fheml/train.h`) runs SGD with backpropagation on
ciphertexts only.

**Weights as diagonals.** Each linear layer is kept as its generalized
diagonals `D_d`, one ciphertext each, without the BSGS pre-rotation, so they
can be updated in place:

| Pass | Computation |
| --- | --- |
| forward | `y = Σ_d D_d ⊙ rot_d(x) + B` (rotations of x hoisted) |
| backward | `∂L/∂x = Σ_d rot_{-d}(D_d ⊙ g)` |
| gradient | `∂L/∂D_d = g ⊙ rot_d(x)`, `∂L/∂B = g` |

**Shared weights.** Layers say which parameter owns each matrix entry
(`LinearLayer::weight_refs`).
- A weight owning one slot (dense) gets that slot's gradient: the update is
  `D_d -= lr/n · mask ⊙ G_d`.
- A weight owning many slots (a conv tap, a per-channel bias, a weight shared
  by every token) needs the sum over its slots. A masked rotate-and-sum over
  all slots computes it and leaves it in every slot, and a second mask puts it
  back in place. That costs `log2(slots)` rotations per shared weight.
- Gradients are summed over the batch first, so this happens once per update.

**Activations** backpropagate through their derivative, evaluated under
encryption:
- `square`: `2z`.
- `poly`: the derivative polynomial.
- `chebyshev`: the derivative Chebyshev series (Numerical Recipes `chder`,
  same `c0/2` convention as OpenFHE).

The loss is `0.5 |y - t|²`, so `g = y - t` with an encrypted target `t`. No
masking is needed: unused slots of `y` only meet zero rows of the matrices,
and the update masks keep gradients out of unused weight slots.

With `Loss::cross_entropy` the loss is `-Σ t log softmax(y)` against a one-hot
`t`, so `g = softmax(y) - t` (`Trainer::loss_gradient`):

1. `z = y + pad`, where `pad` puts the interval midpoint in every non-output
   slot (they hold 0 after a linear last layer; otherwise `y` is masked
   first, one level), so the exp series never sees a value off its interval.
2. `e = mask (*) exp~(z)`: Chebyshev series on `[lo, hi]` (+1 level for the
   interval map), masked to the output slots (+1).
3. `S = Σ_k rot_k(e)` over log2(slots) power-of-two rotations: the sum in
   every slot, so the 1/x series has no padding problem.
4. `r = inv~(S)`: Chebyshev series of 1/x on `[sum_lo, sum_hi]` plus Newton
   steps `r <- r (2 - S r)` (2 levels each); `g = e (*) r - t`.

With `subtract_max` the logits are first masked and padded with `lo` (never
above a logit); a max tree `m <- m + [m < rot_t(m)] (rot_t(m) - m)` over
`t = 1, 2, ..., window/2` (the comparison through `FHEContext::compare`, fresh
at `compare_depth()`) leaves the maximum in slot 0, which is masked and
broadcast back over the window and subtracted. Slots outside the window get
`(lo - hi)/2`, so every slot lands in `[lo - hi, 0]`. `train_plain` runs the
same polynomials (`softmax_plain`); `tests/test_train.cpp` checks the gradient
against finite differences of the exact cross-entropy and the encrypted
update against `train_plain`.

**Levels.** `Trainer::step_depth()` simulates the levels of a whole step
(forward, backward, update) and `context_options()` sizes the context from
that.
- Without bootstrapping, each update leaves the weights deeper, so only a
  planned number of updates fit.
- With bootstrapping, the weights are refreshed after every update.
- Weights are zero outside a window of `w` slots (their output rows), so
  `slots / w` of them are packed side by side with a rotate-and-add tree,
  bootstrapped once, and unpacked with the inverse tree and a mask.
- `fheml.make_trainer` uses every slot of the ring for this (free per
  operation): the MNIST demo went from 67 s to 8 s of bootstrapping per
  update.
- Bootstrapping uses 55-bit scaling: at 50 bits, full-slot bootstrapping
  error is about 1e-2.

**Cost and limits.**
- **Rotation keys:** by default one per diagonal offset, plus its negative.
  That is fine for toy parameters but heavy at 128-bit security, where each
  key at ring 2^16 is about 100 MB (the MNIST training network needs 105
  of them, 10 GB). With a rotation-key ladder
  (`ContextOptions::rotation_window`, see below) the trainer composes every
  rotation from 2·log2(slots) keys instead; `Trainer::rotation_indices(true)`
  then lists only the hoisted rotations of the frozen prefix.
- **Attention:** no backward pass yet.

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
    produce. `context_options()` adds one level per stretch, which
    `EvalBootstrap` needs when its input arrives.
  - Slot values must be roughly within [-1, 1]. `Network::place_bootstraps`
    scales them (below); for hand-placed bootstraps scale the preceding
    linear layer's weights.
  - With 50-bit scaling the first modulus is capped at 57 bits: OpenFHE's
    bootstrapping correction factor allows at most 7 bits between them at
    large rings. Those 7 bits are also the headroom of the last level, so
    network outputs must stay below 128 in magnitude. 128-bit ring 2^16
    holds 24 levels at 50 bits (26 with `key_switch_digits = 4`, whose keys
    are 25% larger), 20 at 55.
  - Bootstrapping keys are rotation keys too and depend on the slot count:
    at ring 2^16 with level budget {3,3}, 45 keys for 256 slots, 75 for 1024,
    96 for 4096 (4.7, 7.6 and 9.7 GB), measured here. A bootstrap takes
    16-22 s on 4 cores at any of these slot counts. Inside a network it
    adds about 1e-2 of error to values in [-1, 1] at 50-bit scaling (the
    deep CNN example: 8 bootstraps, logits 3% off plaintext, classes
    unchanged); 55 bits would be far more precise but leaves only 20
    levels at ring 2^16.

### Rotation-key ladder

`ContextOptions::rotation_window = w` makes `keygen()` generate keys for
`±j·2^(w·i)`, `1 <= j <= 2^(w-1)` (for `w = 1`: all `±2^i`, i.e.
`2·log2(slots) - 1` keys) besides the indices it is given, and
`FHEContext::rotate(c, k)` composes any other rotation from them: `k` is
written in balanced base `2^w`, one key switch per non-zero digit (at most
`ceil(log2(slots) / w)`; signed digits make `-1` as cheap as `+1`).
`FHEContext::rotate_many(x, ks)` rotates one ciphertext by many offsets: with
direct keys it hoists them; otherwise it orders the offsets by cost and
derives each from the already rotated copy nearest to it, so the conv
offsets `28, 29, 30` cost one key switch each after the first.

- `Trainer` and `LinearTransform::eval` rotate through the context, so both
  training and inference work with a ladder; `rotation_indices(true)` on a
  `Trainer` or `Network` lists the rotations that still need their own key
  (attention, the trainer's frozen prefix).
- A composed rotation cannot be hoisted, so each costs 1-3 full key
  switches instead of one hoisted one. The trade is memory for time; at ring
  2^16 the memory is what decides whether a run fits at all.

### Automatic bootstrap placement

`Network::place_bootstraps(budget, calibration, headroom)` inserts
`Bootstrap` layers so that no stretch of the network consumes more than
`budget` levels, as late as possible (fewest bootstraps). The ring then
depends on `budget + bootstrap depth` only, not on the network's depth.

- **Range scaling.** A bootstrap needs its input in about [-1, 1]. The
  calibration inputs are run in the clear and the values entering each
  bootstrap are divided by `headroom · max|x|` (when that maximum exceeds 1)
  and multiplied back afterwards. Both scales are folded into neighbouring
  layers at no cost: weights and biases of a linear layer, `poly`
  coefficients, the evaluation interval of a `chebyshev` layer, through
  `square` (as a square root) and through bias-free linear layers (pooling,
  token mean). Where no layer can take it (a residual block, attention, the
  network output) a fixed `scale` layer is inserted, which costs one level
  that the planner budgets; the planner moves a bootstrap earlier when that
  level does not fit, and reports a budget that cannot hold a layer and its
  scale layers instead of overflowing.
- **Interval folding.** `Network::fold_activation_intervals()` moves the
  affine map `[a, b] -> [-1, 1]` of a Chebyshev activation into the
  preceding linear layer's weights and bias, so `EvalChebyshevSeries` runs
  on [-1, 1] and the activation costs one level less. Run it before placing
  bootstraps. `chebyshev` layers therefore carry an evaluation interval
  (`map EA EB OUT` in the model file) separate from the approximation
  interval.
- **Budget.** `Network::max_bootstrap_budget(options, max_ring)` gives the
  largest stretch that keeps the bootstrapping context at `max_ring`. With
  uniform ternary secrets the bootstrap itself costs 20 levels (level budget
  {3,3}) or 18 ({2,2}) plus the spare level `EvalBootstrap` wants at its
  input; at ring 2^16, 128-bit and 50-bit scaling (24 levels) that leaves 3
  or 5 levels per stretch, and 5 or 7 with `key_switch_digits = 4` (26
  levels). A degree-15 Chebyshev ReLU with its interval folded costs 5, so
  a conv + ReLU block fits a stretch of 6; the deep CNN example gets by with
  5 because the conv's level is spent before the bootstrap that precedes
  each ReLU. 45-bit scaling would give more levels, but at 256 slots its
  bootstrap error was 7e-2 here (first modulus 52), unusable, so 50 bits is
  the floor.
  `fheml.auto_bootstrap(net, calibration)` does all three steps in Python.
- Training through placed bootstraps works: the trainer bootstraps the
  activation forward and the gradient backward (`tests/test_train.cpp`).

## GPU backend (FIDESlib)

`include/fheml/backend.h` picks the CKKS API at compile time: `lbcrypto`
(OpenFHE, default) or `fideslib` (`-DFHEML_GPU=ON`). FIDESlib 2.2 mirrors
OpenFHE's `CryptoContext` API and keeps an OpenFHE 1.6.1 context inside, so
`CC`, `Ctxt` and `Ptxt` are the only types that change. GPU ciphertexts are
bit-identical to the CPU ones.

- **Device choice.** `FHEContext` passes `gpu_devices` to FIDESlib only when
  `use_gpu` is set and CUDA reports a device. With no devices, every FIDESlib
  call runs on its OpenFHE context: that is the CPU fallback.
  `keygen()` uploads the keys and bootstrapping precomputations with
  `LoadContext`. Fresh ciphertexts are then uploaded at encryption.
- **Hoisting.** FIDESlib's single-index `EvalFastRotation` doesn't hoist on
  the GPU. `backend::rotate_hoisted` therefore uses the multi-index form for
  BSGS baby steps, attention token shifts and training rotations.
- **Threads.** The GPU context isn't thread-safe: its handle registry has no
  lock. Every OpenMP loop over ciphertexts has an `if (ctx.parallel())` clause,
  which is false on the GPU.
- **Missing operations.** FIDESlib has no `EvalMultNoRelin`/`Relinearize`. The
  encrypted-weights BSGS relinearizes each product instead of each giant step.
  It also has no `EvalPoly`. `backend::eval_poly` rewrites the power basis into
  Chebyshev coefficients and calls `EvalChebyshevSeries` on [-1, 1]. On that
  interval no affine map is applied, so the result holds for every x and the
  depth matches `poly_depth()`. `test_fheml` checks this on every build.

## Status

All of the following are tested under encryption:
- Convolutions with stride, padding and multiple channels.
- Dense layers, average pooling and channel affine layers.
- Exact max pooling through CKKS ↔ FHEW scheme switching (TOY FHEW
  parameters in the tests; the STD128 numbers are in the README).
- LayerNorm (compact and padded token strides, inside a pre-norm residual
  block) and softmax attention, with and without the scheme-switching
  maximum subtraction.
- Fused linear layers.
- The square, polynomial and Chebyshev activations.
- Bootstrapping in the middle of a network, placed by hand or by
  `place_bootstraps` (with scale folding and interval folding).
- Rotations composed from a key ladder, for inference and training.
- The trained MNIST CNN at 128-bit security: about 3 s per image, with
  predictions identical to plaintext.
- The same CNN with encrypted weights: about 3.6 s per image.
- A one-block MNIST vision transformer (95.6%): about 15 s per image.
- Python bindings for all of the above.
- Encrypted training (`Trainer`): SGD updates on encrypted data and weights
  match plaintext SGD, with and without bootstrapping; this includes
  layer norm (gamma, beta) and attention (quadratic or softmax scores),
  whose backward passes reuse the forward's rotations and masks.
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
   - Training-time range calibration.
5. **Bootstrap placement**: per-segment slot counts, and a cost model that
   weighs an earlier bootstrap against the scale layers around residual
   blocks (the planner is greedy-latest with back-off).
6. **Batch-norm and residual connections.**
   - Fold BN into the convolutions at load time.
   - Add a residual `Add` layer that aligns levels and layouts.
7. **More PyTorch export.** `fheml.from_torch` covers `nn.Sequential` models
   (see the README). Next: tracing arbitrary `forward` code with `torch.fx`,
   and a range penalty during training so Chebyshev activations stay inside
   their interval.
8. **Encrypted training.** An encrypted softmax for cross-entropy. Hoisting for composed rotations (one
   decomposition per ladder digit).
9. **Performance.**
   - Benchmark the FIDESlib GPU backend on a real GPU (see below), and batch
     the remaining single-ciphertext calls (giant steps, rotate-and-sum) for it.
   - Reduce `scale_mod_bits` / depth so the MNIST net fits ring 2^14.
   - Use `EvalFastRotationExt` to keep giant-step sums in the extended basis.
   - Benchmark threading.
