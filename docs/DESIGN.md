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

### Multiplexed packing

`Network::set_packing(Packing::kMultiplexed)` (Python: `net.packing =
"multiplexed"`) fills the gaps with other channels, as in the multiplexed
parallel convolutions of Lee et al. (ICML'22). A layout carries a
multiplexing factor `m` and the fine grid `(fy, fx)` of the original image;
`m²` channels share one block of `cs` slots, and channel `c` sits at
sub-offset `(i, j) = ((c mod m²) / m, c mod m)`:

    slot = (c / m²)·cs + i·fy + j·fx + y·ys + x·xs

After a stride-2 conv on a compact `H×W` input, `ys = 2W`, `xs = 2`, `m = 2`,
`(fy, fx) = (W, 1)`: the four channels of a block interleave like the pixels
of a 2×2 window, and a second stride-2 stage makes `m = 4`. The offset
between an input and an output element still depends only on (input
channel, output channel, kernel tap), so a strided conv keeps few diagonals,
while the activation needs `C·H·W/m²` slots, like a compact one.

There is no new evaluation algorithm: a layout is only a map from elements to
slots, and every linear layer's slot matrix is built from its input and output
layouts, so the BSGS diagonal method handles any layout. The masks of Lee et
al. are the zeros of the diagonals. That also makes re-packing free: a
stride-1 conv after max pooling (whose output stays gapped) writes its output
multiplexed.

**Choosing a layout per layer.** Multiplexing is not always cheapest. A stride-1
conv on a compact input already has constant offsets (`(Ci+Co-1)·k²`
diagonals), while the interleaved sub-offsets of a multiplexed one give up to
`(Ci/m²+Co/m²-1)·((k+1)m-1)²`. With `kMultiplexed`, each strided (or
re-packing) linear layer considers the multiplexed, gapped and compact output
layouts that are collision-free, and takes the one with the lowest estimated
cost `(diagonals + 8·sqrt(diagonals)) × slots`, where slots is the power of
two that holds both the input and that output. Weighting by slots stops a
gapped layout from winning on diagonals when it would double the ring.
Odd sizes (7×7 → 4×4 with padding) can't interleave without collisions and
fall back to gapped or compact. `kGapped` stays the default; `kCompact` is
`set_gapped_layouts(false)`.

Measured numbers (`packing_bench`, `mnist_cnn --packing`) are in the README section
"Multiplexed packing".

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

The convolution family is written this way too. `Conv2d` covers channel
groups (a depthwise convolution is `groups = channels`), dilation and
separate vertical and horizontal padding; a 1-D convolution over a `(C, L)`
signal is the same layer on the `(C, 1, L)` shape with `kh = 1`.
`ConvTranspose2d` is the transpose of a strided convolution's matrix, and
`Upsample` (nearest, nearest-exact, bilinear, or linear on signals) is a
fixed matrix with at most four entries per row. None of them needs special
evaluation code: their cost is set by their distinct slot offsets, so a
depthwise 3×3 has 9 diagonals per channel block where a dense one has
`9 (Ci + Co - 1)`. Shared taps train like any other conv weight (a masked
rotate-and-sum of the gradient), which gives U-Net- and MobileNet-style
encoders and decoders the same encrypted training as plain CNNs.

Because a layer is nothing more than its logical matrix and bias,
`LinearMap` takes them directly: a fixed sparse matrix between any two
shapes, plus a bias per output element. It covers the rearrangements that no
convolution expresses, so that post-processing stays encrypted. Examples are
pixel shuffle (a permutation from `(C r², H, W)` to `(C, rH, rW)`), gathering
each pixel's neighbourhood into channels, and decoders that add each grid
cell's own position. It is lowered like any other linear layer: one level,
plus a rotation per distinct slot offset. Its torch twin is
`fheml.nn.LinearMap`, which stores (row, col, value) triples.

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
| `exactrelu R s` | `EvalCompareSchemeSwitching(0, x)` (FHEW sign), `x · sign` | 13; 1 past a bootstrap |

The plaintext reference evaluates the *same* approximation
(`EvalChebyshevFunctionPtxt`). Encrypted-vs-plaintext error therefore measures
only the CKKS noise. The approximation error is a training-time concern.

`exactrelu` has no approximation: its reference is `max(x, 0)`. Its sign
comes back from FHEW as a fresh ciphertext 12 levels deep regardless of the
input, so its cost in levels depends on where it sits: 13 from a fresh
input, 1 when the input is already deeper than 12, which always holds past a
CKKS bootstrap (18–20 levels). The additive depth model expresses this with a
per-layer flag (`after_bootstrap`) that `place_bootstraps` sets, and the
layer checks at run time that its input really is that deep.

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

#### Masks, segments and encoder-decoders

Every attention pattern of a transformer is one schedule with a mask.
`Attention(..., mask=M)` takes an n×n 0/1 matrix, `M[i][j] = 1` where query
i may attend to key j. Offset s pairs query i with key (i + s) mod n, so the
mask enters the per-offset masks that the schedule multiplies by anyway: the
score coefficients (quadratic scores) or the head mask after `exp`
(softmax). A masked-out pair gets weight 0 at no extra level or rotation,
and an offset that pairs nothing is skipped, with its rotations and
products. A query that attends to nothing (all its row masked) gets the
middle of the `1/x` interval as its row sum, like the slots outside the
heads, and outputs 0. With internal bootstrapping, masked scores are mapped
as if they were 0 before the packed bootstrap, so they stay in range.

An encoder-decoder runs as one sequence of S source and T target tokens.
Token segments (`segments=`, a segment id per token) give the per-token
layers one weight set per segment: `TokenDense` (embeddings, MLPs, the
head), `LayerNorm` (gamma, beta) and the four attention projections.
Different weights on different tokens cost nothing: `kron(I_n, W)` becomes
block-diagonal with other blocks, and its diagonals stay at the same
offsets. A block that belongs to one side has zero weights on the other
side's tokens, so its residual passes them through. Then:

* encoder self-attention: Q, K, V, O on the source, mask source × source
  (2S − 1 of the n offsets);
* decoder self-attention: Q, K, V, O on the target, causal mask
  `S ≤ j ≤ i`;
* cross-attention: Q and O on the target, K and V on the source, mask
  target × source;
* the head: a `TokenDense` acting on the target tokens.

Every layer keeps the depth it has without a mask (6 for quadratic scores),
and the planner places bootstraps as usual. The price of one sequence is
that every layer touches all n tokens: an encoder block computes with
zeros on the T target tokens.

Generation stays encrypted from the letters to the last token. Greedy
decoding unrolls into one network of T − 1 residual steps. Step t runs the
model, and a `Select` (a 0/1 slot matrix, one level) picks target position
t's logits. `ArgMax` turns them into an encrypted one-hot by a tournament
of scheme-switching comparisons. A second `Select` writes the one-hot into
the input slots of position t + 1, and the residual adds it to the
sequence. That position held zeros, which is also how training pads the
target, so every position stays inside the series intervals. The causal
mask means step t + 1 sees exactly the decoded prefix. Every input runs
every step, so neither the tokens nor the output length leak, and the
client decrypts only the final sequence. In a bootstrapped network,
`ArgMax` costs the stretch only its gather. The comparison bits come back
fresh from the scheme switch, above a bootstrap's output level, and the
running maxima are refreshed (`FHEContext::refresh`) whenever the next
round's comparison would run out of levels, so tournaments of any length
fit a stretch of two levels or more. `fheml.nn.SegmentLinear`, `segments=` on `fheml.nn.LayerNorm` and
`fheml.nn.Attention` (with `query_segments` and `key_segments`) and
`fheml.nn.encoder_decoder_masks(S, T)` are the torch twins. The model file
writes `segments s_0 … s_{n-1}` after a layer's sizes and
`mask m_00 … m_{n-1,n-1}` after the heads; an attention bias flag of 2 marks
per-token biases.

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

### Bootstrapping inside LayerNorm and softmax attention

A softmax attention block costs 17–23 levels and LayerNorm 10–12, far more
than the 3–7 levels a ring-2^16 bootstrapping context leaves per stretch.
So both layers bootstrap internally (`can_self_bootstrap()`), switched on by
`place_bootstraps` when the layer does not fit the budget B:

- **Plan.** `LayerNorm::plan(d, B)` / `Attention::plan(d, B)` walk the
  layer's steps from input stretch level d and put a refresh before each
  step that would leave the stretch; `level_after`, `level_peak` and
  `internal_bootstraps` report the result to the planner. The attention
  plan tries refreshing the exponentials and the row sum or not (four
  combinations) and keeps the cheapest that fits.
- **No scale levels.** A refresh is a bare `EvalBootstrap`, which needs
  its input in [-1, 1]. Instead of a scaling product (one level) before
  each one, the scales are folded into what precedes it: the Chebyshev
  coefficients of exp and 1/√v / 1/x are divided by a bound, the Newton
  constants absorb the square of it, gamma and the output masks undo it.
  `FHEContext::refresh(c, bound)` (one power-of-two product, bootstrap,
  doubling back) is used only where a calibrated value bound is needed: the
  layer input x and the attention output o, bounded by `calibrate()` with
  the planner's headroom.
- **Attention scores.** The n score ciphertexts are packed two head-dims
  per ciphertext (the upper half of the token block holds the second one),
  bootstrapped together and unpacked with two masks, so a refresh of the
  scores costs ⌈n / 2d_h⌉ bootstraps instead of n.
- **Residual blocks.** `place_bootstraps` plans a residual body that does
  not fit recursively from its input level (`ChainPlan`), so bootstraps go
  inside the block when that is cheaper than in front of it.

`max subtract` softmax cannot bootstrap itself (its comparisons need the
whole block in one stretch). The model file records `selfboot` with the
calibrated bounds and the planner's `budget`, so a saved model keeps its
plan. Measured in `tests/test_fheml.cpp`: a two-block transformer
(LayerNorm, 2-head softmax attention, LayerNorm, MLP) at B = 5 runs with 12
bootstraps inside layers and decrypts to within 1e-4 of the plaintext.

## Recurrent layers

`Recurrent` (src/recurrent.cpp) evaluates an Elman RNN, a GRU or an LSTM
over n tokens laid out like a transformer's (token t at slots t·B + j).

- **All steps at once.** The recurrence h_t = f(x_t, h_{t−1}) is a
  triangular system, solved by Jacobi iteration: the state ciphertext holds
  every token's state, and one step computes f for all tokens from the
  previous iterate, shifted by one token. The input projection
  U = W_ih x + b is computed once; per step the recurrent map is one linear
  transform from token t−1's slots to token t's (a token-shifted W_hh with
  2h − 1 diagonals per gate), and the previous cell state (LSTM) or hidden
  state (GRU) is a rotation by −B. After k steps tokens 0..k−1 are exact;
  the others hold the state of the same RNN started from zero k tokens
  earlier, an ordinary RNN state that stays inside the series' intervals. n
  steps give the exact result with the same depth as a step-by-step
  evaluation, and no operation depends on the position.
- **Packed gates.** The gate pre-activations go to G regions of R slots
  (R = n·B rounded up to a power of two; G = 1, 3, 4), so one Chebyshev
  series evaluates every sigmoid of a step, and a rotation by g·R brings gate
  g onto the state slots. The LSTM's tanh candidate is 2 σ(2x) − 1 on the
  same series. The GRU's W_hn h + b_hn (multiplied by r before its tanh)
  shares the ciphertext, divided by a weight bound so its unused sigmoid
  stays in range. Masked products (the i, f, r and z gates times a
  plaintext mask) keep every slot outside the states at zero, which is what
  keeps the series' arguments inside their intervals everywhere.
- **Free interval maps.** The series' intervals are folded into the weights
  (t = x / a), and the LSTM cell state is kept divided by its tanh interval,
  so every series runs on [−1, 1] at poly_depth(degree) levels. Per step:
  1 + pd for the RNN, 2 pd + 4 for GRU and LSTM (5 and 12 at degree 13).
- **Bootstrapping inside.** One shared description of the step
  (`Recurrent::run`) is executed three ways: on ciphertexts, on cleartext
  slot vectors (the plaintext reference and the calibration, which
  therefore see the junk-free slots and every intermediate iterate) and on
  level counters (the plan for `level_after`, `level_peak`,
  `internal_bootstraps`). Before each group of operations it refreshes the
  operand whose next levels would leave the stretch: the state before the
  recurrent map, the gate pre-activations before the series, the gates
  before their products, the cell state before its tanh. These are in
  [−1, 1] by construction, so a refresh is a bare bootstrap; only the input
  and its projection carry calibrated bounds (`FHEContext::refresh`). With a
  5-level stretch the RNN needs one bootstrap per step, GRU and LSTM three.
  `place_bootstraps` folds scales through the layer: an input scale into
  W_ih, an output scale into the last step's series or masks.

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

`Trainer` (`include/fheml/train.h`) runs backpropagation and an optimizer
(SGD with momentum and weight decay, or Adam / AdamW) on ciphertexts only.

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
- A weight owning one slot (dense) gets that slot's gradient: the plain SGD
  update is `D_d -= lr/n · mask ⊙ G_d`.
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

**Max pooling** reuses the forward's comparisons. Each round of the
tournament computes `max(a, b) = a + s (b - a)` with `s = [a < b]`, so
`∂max/∂a = 1 - s` and `∂max/∂b = s`. The forward keeps `s ⊙ m` and
`m - s ⊙ m`, where `m` masks the pool's output slots. The backward walks the
tournament from the root, multiplying the gradient by these, and rotates
each tap's gradient back by `-offset` to its input slot (overlapping windows
add up). That is `ceil(log2 k²)` levels and no scheme switching. The mask
matters when the pool is the output layer: `g = y - t` is nonzero in the
gap slots there and must not reach the layer below. `train_plain` runs the
same tournament in the clear (a tie keeps the first tap, as `[a < b] = 0`
does).

**Exact ReLU** keeps the comparison bit of its forward pass,
`s = out_scale · [x > 0]`, and the backward pass is `g ⊙ s`: one level, no
scheme switching. **Masked and segmented attention** backpropagate over the
offsets the forward evaluated (`Attention::offsets()`), with the same score
and head masks, so a causal decoder or an encoder-decoder trains like plain
self-attention.

**Batch norm** couples the samples of a batch, so a network with one trains
the batch layer by layer (`Trainer::accumulate_batch`): every node runs all
samples before the next node does, and each sample's forward state is kept
for the backward pass. For a batch of `B` the batch norm computes

| Value | Computation | Levels |
| --- | --- | --- |
| mean | `μ = M(Σ_b x_b)` | 1 |
| variance | `σ² = M(Σ_b (x_b - μ)²)` | 2 |
| `r ≈ 1/√(σ² + ε)` | Chebyshev series on `[var_min, var_max]` + Newton | as `layernorm` |
| output | `y_b = γ ⊙ (x_b - μ) ⊙ r + β` | 2 |

where `M` is a fixed linear map: the mean over a channel's positions,
replicated to them, divided by `B` (one level, cached per batch size). The
backward pass is the usual one, `∂L/∂x_b = r ⊙ (g̃_b - M(Σ g̃) - x̂_b ⊙
M(Σ g̃ ⊙ x̂))` with `g̃ = γ ⊙ g`, so it needs two more applications of `M`.
The running mean and variance are ciphertexts updated with torch's
momentum rule (unbiased variance) and are bootstrapped with the weights.
Inference uses them as a channel affine map: `r` from the running variance,
`y = γ ⊙ (x - m) ⊙ r + β`.

**Dropout** draws a fresh mask per sample at each step from the layer's
seeded generator and multiplies it in as a plaintext (one level). Kept units
are scaled by `1/(1 - p)` (inverted dropout, as torch), so the expected
activation equals the inference one and the layer is the identity at
inference. The backward pass multiplies the gradient by the same mask.

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

**Optimizers.** The optimizer sees, per weight ciphertext, the masked mean
gradient `P` above: every slot holds its parameter's gradient (a shared
weight's broadcast sum), zero elsewhere. Every public constant of the update
(learning rate, batch size, momentum, Adam's bias corrections) rides on the
plaintext masks that produce `P`, so several scaled copies of `P` come out of
the same masked sum at no extra level. The optimizer state is one ciphertext
per weight ciphertext with the same zero pattern, so it is bootstrapped in
the same packed bootstraps as the weights.

| Optimizer | Encrypted update | Levels after `P` |
| --- | --- | --- |
| SGD | `D += P[-lr/n]` | 0 |
| weight decay `wd` (L2) | `+ D ⊙ (-lr wd)`; decoupled (SGDW/AdamW) the same product outside `g'` | 1 (on `D`) |
| momentum `mu` | `b = P[1/n] + wd D + mu b`, `D += P[-lr/n] - lr wd D - lr mu b_old` | 1 (on `D`, `b`) |
| Nesterov | `D += P[-lr(1+mu)/n] - lr(1+mu) wd D - lr mu² b_old` | 1 |
| Adam | `m̂ = P[(1-b1)/(c1 n)] + b1 (c1'/c1) m̂`, `v̂ = P[√((1-b2)/c2)/n]² + b2 (c2'/c2) v̂`, `D += (m̂ · -lr) ⊙ rsqrt(v̂)` | 1 + pd(deg) + 1 + 2 newton + 1 |

`c1 = 1 - b1^t`, `c2 = 1 - b2^t` and their values `c1'`, `c2'` at `t - 1`
are public. Adam stores the bias-corrected moments `m̂ = m/c1` and `v̂ =
v/c2`, which follow the same recurrences with these constants: the raw `v`
is `c2` times smaller (1e-3 at the first step for `b2 = 0.999`), so its
bootstraps, whose error is absolute, would lose most of its digits.
`rsqrt(u) ≈ 1/sqrt(u + eps)` is the layer norm's inverse square root: a
Chebyshev series on `[0, v_max]` and Newton steps `r <- 1.5 r - 0.5 (u +
eps) r³`, which take `v̂ + eps` from a scalar product on `v̂` (it has levels
to spare). As in the layer norm, the series is scaled by
`1.5^newton` so that the Newton steps need no scalar product on the root.
`eps` sits inside the root (torch adds it outside) and sets how steep the
series is: `v_max / eps = 100` with degree 31 and two Newton steps is
accurate to 3e-12, `1000` needs degree 63. Under a level budget the series,
the Newton steps and the final product are bootstrapped like any other
operation; `min_level_budget()` includes the series. `train_plain` runs the
same update with an `OptimizerState`, and `tests/test_train.cpp` checks it
against the textbook definitions (exact square root) and the encrypted
update against it over several steps, with bootstrapped weights and state
and under a level budget.

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
- **Level budget** (`Trainer::set_level_budget(K, range)`): for steps
  deeper than one bootstrapping context holds, every ciphertext stays within
  K levels of the weights. The passes call `fit(c, d)` on each operand before
  an operation that consumes d levels; it bootstraps c when its level plus d
  would pass K. Decisions use the ciphertexts' real levels, so nothing has to
  be planned per layer: activations, gradients, the forward values kept for
  the backward pass (bootstrapped in place, so later uses see the fresh copy)
  and the intermediate values of attention and layer norm are refreshed
  exactly where needed, including between the comparison tournament, the exp
  series and the 1/x series of the max-subtracted softmax. Fits run between
  the parallel loops, never inside them. Indivisible operations (a
  polynomial, the layer norm's `normalize`, a frozen layer) must fit K
  (`min_level_budget()`). Values are divided by `range` (a power of two, one
  level) before each of these bootstraps and doubled back after it with
  additions (no level), so the context needs K + 3 levels plus the
  bootstrap's. The weights are still refreshed once per update, packed.

**Cost and limits.**
- **Rotation keys:** by default one per diagonal offset, plus its negative.
  That is fine for toy parameters but heavy at 128-bit security, where each
  key at ring 2^16 is about 100 MB (the MNIST training network needs 105
  of them, 10 GB). With a rotation-key ladder
  (`ContextOptions::rotation_window`, see below) the trainer composes every
  rotation from 2·log2(slots) keys instead; `Trainer::rotation_indices(true)`
  then lists only the hoisted rotations of the frozen prefix.

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
direct keys it hoists them (one digit decomposition, then one key-switch
inner product per offset). With a ladder it evaluates a **hoisted rotation
tree** (`FHEContext::plan_rotations`): every node is a rotated copy of `x`,
every edge one ladder key, and every node with children is decomposed once
for all of them. The planner adds the offsets cheapest first, each from the
tree node (input, earlier offset or intermediate copy) that is cheapest to
extend, counting a new key switch as 2 and a new decomposition as 1
(measured: the decomposition is about a third of an OpenFHE HYBRID key
switch, 19 of 55 ms at ring 2^15). Composed paths take their largest digit
first, so offsets with a common coarse part (conv rows `±W`, attention
tokens `s·B`) share the copy at that part and fan out from it. The tree is
evaluated level by level, each level's decompositions and rotations as one
parallel loop (one batched FIDESlib call per node on the GPU); plans are
cached per offset list.

Every place that rotates one ciphertext by several offsets goes through
`rotate_many`: BSGS baby steps (conv, pooling, dense), max pooling taps,
attention's key/value token rotations (which used per-index keys directly
before), and the trainer's linear forward/backward. A frozen layer's
backward `Σ_i rot_{-o_i}(g ⊙ diag_i)` is now evaluated as
`Σ_i rot_{-o_i}(g) ⊙ rot_{-o_i}(diag_i)`, so its rotations all act on `g`
and are hoisted, with the constant diagonals rotated in the clear. What
stays unhoisted are chains where each rotation acts on the previous result
(rotate-and-sum in LayerNorm, attention windows and folded dense heads,
BSGS giant steps), which hoisting cannot share. `examples/rotation_bench`
measures all of this; see the README.

- `Trainer` and `LinearTransform::eval` rotate through the context, so both
  training and inference work with a ladder; `rotation_indices(true)` on a
  `Trainer` or `Network` lists the rotations that still need their own key
  (attention, the trainer's frozen prefix).
- A composed rotation still costs 1-3 key switches instead of one hoisted
  one; the tree shares the decompositions and common prefixes but not the
  digits themselves. The trade is memory for time; at ring 2^16 the memory
  is what decides whether a run fits at all.

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
  token mean, max pooling's gather), attention (W_q, W_k, W_v take an
  input scale, W_o an output scale) and residual blocks whose body starts
  and ends with such a layer (skip and body scale alike). Where no layer
  can take it (the network output, a scale carried into a residual block) a
  fixed `scale` layer is inserted, which costs one level that the planner
  budgets; the planner moves a bootstrap earlier when that
  level does not fit, and reports a budget that cannot hold a layer and its
  scale layers instead of overflowing.
- **Interval folding.** `Network::fold_activation_intervals()` moves the
  affine map `[a, b] -> [-1, 1]` of a Chebyshev activation into the
  preceding linear layer's weights and bias, so `EvalChebyshevSeries` runs
  on [-1, 1] and the activation costs one level less. Run it before placing
  bootstraps. `chebyshev` layers therefore carry an evaluation interval
  (`map EA EB OUT` in the model file) separate from the approximation
  interval.
- **Inside layers.** Layers that cannot fit a stretch bootstrap themselves
  (see "Bootstrapping inside LayerNorm and softmax attention"), and residual
  bodies are planned recursively. Max pooling absorbs a scale in its gather.
- **Last layer.** No bootstrap follows the network's last layer, so it may
  end one level past the budget: the spare level `EvalBootstrap` needs at
  its input is free there.
- **Precision.** A bootstrap's error is absolute: about 1e-3 at 50-bit
  scaling and first modulus 57 (toy ring 4096, {3,3}), 5e-3 at ring 2^14.
  Layers that bootstrap internally divide values by their interval bounds
  first, so a wide interval multiplies that error. Two remedies:
  `ContextOptions::bootstrap_iterations = 2` (OpenFHE's iterative
  bootstrapping: the residual error of a first bootstrap is bootstrapped
  again, 1.2e-3 -> 4e-6 measured at ring 4096, for twice the time and one
  more level), and narrow intervals, which `fheml.nn.range_loss()` lets a
  model learn to stay inside while training.
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
- Convolutions with stride, padding and multiple channels, with gapped,
  compact or multiplexed outputs (also through encrypted training).
- Dense layers, average pooling and channel affine layers.
- Exact max pooling through CKKS ↔ FHEW scheme switching (TOY FHEW
  parameters in the tests; the STD128 numbers are in the README).
- LayerNorm (compact and padded token strides, inside a pre-norm residual
  block) and softmax attention, with and without the scheme-switching
  maximum subtraction.
- Fused linear layers.
- Recurrent layers (Elman RNN, GRU, LSTM, all states or the last one), with
  bootstrapping inside the recurrence, against torch.
- The square, polynomial and Chebyshev activations, and the exact
  scheme-switching ReLU (checked against `max(x, 0)`, also past bootstraps).
- Bootstrapping in the middle of a network, placed by hand or by
  `place_bootstraps` (with scale folding and interval folding).
- Rotations composed from a key ladder, for inference and training.
- The trained MNIST CNN at 128-bit security: about 3 s per image, with
  predictions identical to plaintext.
- The same CNN with encrypted weights: about 3.6 s per image.
- A one-block MNIST vision transformer (95.6%): about 15 s per image.
- Python bindings for all of the above.
- Encrypted training (`Trainer`): SGD (momentum, Nesterov, weight decay)
  and Adam / AdamW updates on encrypted data, weights and optimizer state
  match the plaintext optimizers, with and without bootstrapping; this
  includes layer norm (gamma, beta), batch norm (batch statistics and running
  statistics), dropout, attention (quadratic or softmax scores, masked and
  segmented), whose backward passes reuse the forward's rotations and masks,
  and exact max pooling and ReLU, whose backward passes reuse the forward's
  comparison bits.
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
3. **Multiplexed packing**: done (`Packing::kMultiplexed`, see above). Next:
   residual blocks on images (a residual `Add` that lines up multiplexed
   layouts), and a network-level layout search instead of the per-layer
   greedy choice.
4. **Better activations.**
   - Composite minimax sign/ReLU, built by composing `EvalChebyshevSeries`
     calls.
   - Training-time range calibration.
5. **Bootstrap placement**: per-segment slot counts, and a cost model that
   weighs an earlier bootstrap against the scale layers around residual
   blocks (the planner is greedy-latest with back-off).
6. **Residual connections on images.** Add a residual `Add` layer that
   aligns levels and layouts.
7. **More PyTorch export.** `fheml.from_torch` covers `nn.Sequential` models
   (see the README). Next: tracing arbitrary `forward` code with `torch.fx`,
   and a range penalty during training so Chebyshev activations stay inside
   their interval.
8. **Encrypted training.** An encrypted softmax for cross-entropy.
9. **Performance.**
   - Benchmark the FIDESlib GPU backend on a real GPU (see below), and batch
     the remaining single-ciphertext calls (giant steps, rotate-and-sum) for it.
   - Reduce `scale_mod_bits` / depth so the MNIST net fits ring 2^14.
   - Use `EvalFastRotationExt` to keep giant-step sums in the extended basis.
   - Benchmark threading.
