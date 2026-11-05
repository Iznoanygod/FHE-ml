"""HE-friendly PyTorch modules that ``fheml.from_torch`` knows how to export.

Train a model built from ``torch.nn.Sequential``, the supported ``torch.nn``
layers (Conv2d, Linear, AvgPool2d, AdaptiveAvgPool2d(1), BatchNorm, Flatten,
Dropout, Identity) and the modules below (``MaxPool2d`` is torch's with an
input bound), then export it::

    import torch, fheml
    from fheml import nn as hnn

    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 5, 5, stride=2, padding=1), hnn.Poly([0, 0.5, 0.25]),
        torch.nn.Flatten(), torch.nn.Linear(845, 10))
    ...train...
    net = fheml.from_torch(model, (1, 28, 28))

Every module here computes exactly what the encrypted layer computes, so the
model you train is the model that runs under encryption.
"""
import math

import numpy as np
import torch
from torch import nn

__all__ = ["Poly", "Square", "Chebyshev", "Attention", "LayerNorm", "MaxPool2d", "Residual", "TokenMean",
           "PosEmbedding", "Bootstrap", "SegmentLinear", "chebyshev_coefficients", "range_loss", "segment_ids",
           "causal_mask", "encoder_decoder_masks"]


class Poly(nn.Module):
    """c0 + c1 x + c2 x^2 + ...  (``trainable=True`` learns the coefficients)."""

    def __init__(self, coeffs, trainable=False):
        super().__init__()
        c = torch.as_tensor(coeffs, dtype=torch.get_default_dtype())
        if trainable:
            self.coeffs = nn.Parameter(c)
        else:
            self.register_buffer("coeffs", c)

    def forward(self, x):
        y = torch.zeros_like(x)
        for c in reversed(self.coeffs):  # Horner
            y = y * x + c
        return y

    def extra_repr(self):
        return f"coeffs={[round(float(c), 4) for c in self.coeffs]}"


class Square(nn.Module):
    """x^2 (CryptoNets)."""

    def forward(self, x):
        return x * x


_FUNCTIONS = {
    "relu": lambda x: np.maximum(x, 0.0),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "tanh": np.tanh,
    "silu": lambda x: x / (1.0 + np.exp(-x)),
    "gelu": lambda x: 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0))),
}


def chebyshev_coefficients(fn, a, b, degree):
    """Chebyshev interpolant of ``fn`` on [a, b] (same nodes as OpenFHE).

    ``fn`` is a name (relu, sigmoid, tanh, silu, gelu) or a numpy callable.
    """
    f = _FUNCTIONS[fn] if isinstance(fn, str) else fn
    return np.polynomial.chebyshev.chebinterpolate(lambda t: f((b - a) / 2 * t + (a + b) / 2), degree)


def _chebyshev_eval(c, a, b, x):
    """Clenshaw evaluation of the series ``c`` on [a, b] at ``x`` (differentiable)."""
    t = (2 * x - (a + b)) / (b - a)
    b1, b2 = torch.zeros_like(x), torch.zeros_like(x)
    for ck in reversed(c[1:]):
        b1, b2 = 2 * t * b1 - b2 + ck, b1
    return t * b1 - b2 + c[0]


def _excess(x, lo, hi):
    """Mean squared distance of x outside [lo, hi] (0 inside)."""
    return (torch.relu(x - hi) + torch.relu(lo - x)).pow(2).mean()


def range_loss(model):
    """Sum of how far the last training forward pushed Chebyshev activation
    inputs, LayerNorm variances and softmax scores / row sums outside the intervals their encrypted series
    cover (mean squared excess). Add ``weight * range_loss(model)`` to the
    training loss so the model learns to stay inside them: the series
    diverge outside, and encrypted inference cannot clamp."""
    terms = [m._excess for m in model.modules() if getattr(m, "_excess", None) is not None]
    return sum(terms) if terms else torch.zeros(())


def _series_buffer(fn, a, b, degree):
    return torch.as_tensor(chebyshev_coefficients(fn, a, b, degree), dtype=torch.get_default_dtype())


class Chebyshev(nn.Module):
    """Degree-``degree`` Chebyshev approximation of relu|sigmoid|tanh|silu|gelu on [a, b].

    The forward pass evaluates the polynomial itself, not the exact function,
    so training sees what encrypted inference computes. Inputs outside [a, b]
    blow up quickly, as they do under encryption: while training, the series
    sees its input clamped to [a, b] and records how far the input went
    outside, which ``range_loss`` turns into a penalty that pulls the
    pre-activations back inside. Check the ranges on data before encrypting.
    """

    def __init__(self, fn, a, b, degree):
        super().__init__()
        if fn not in _FUNCTIONS:
            raise ValueError(f"unknown function {fn!r}; expected one of {sorted(_FUNCTIONS)}")
        self.fn, self.a, self.b, self.degree = fn, float(a), float(b), int(degree)
        self.register_buffer("c", torch.as_tensor(chebyshev_coefficients(fn, a, b, degree),
                                                  dtype=torch.get_default_dtype()))

    def forward(self, x):
        if self.training:
            self._excess = _excess(x, self.a, self.b)
            x = x.clamp(self.a, self.b)
        return _chebyshev_eval(self.c, self.a, self.b, x)

    def extra_repr(self):
        return f"{self.fn}, a={self.a}, b={self.b}, degree={self.degree}"


def segment_ids(segments):
    """Token segment ids from segment lengths, e.g. ``(3, 2) -> [0, 0, 0, 1, 1]``.

    A sequence may consist of segments (the source and the target of an
    encoder-decoder): the per-token modules then hold one weight set per
    segment, which costs nothing extra under encryption (see
    ``fheml.TokenDense``).
    """
    if segments is None:
        return None
    lengths = [int(v) for v in segments]
    if not lengths or min(lengths) < 1:
        raise ValueError("segments: give the segment lengths, each at least 1")
    return [g for g, n in enumerate(lengths) for _ in range(n)]


def _segment_ids(segments, tokens=None):
    ids = segment_ids(segments)
    if ids is None:
        return None
    if tokens is not None and len(ids) != tokens:
        raise ValueError(f"segments cover {len(ids)} tokens, the layer has {tokens}")
    return torch.as_tensor(ids, dtype=torch.long)


def causal_mask(n, start=0):
    """(n, n) mask, 1 where query i may attend to key j: tokens start.. attend
    to themselves and earlier tokens from start on; the others to nothing."""
    m = torch.zeros(n, n)
    for i in range(start, n):
        m[i, start:i + 1] = 1
    return m


def encoder_decoder_masks(src, tgt):
    """The three attention masks of an encoder-decoder over one sequence of
    ``src`` source tokens followed by ``tgt`` target tokens: ``encoder``
    (source x source), ``causal`` (target x earlier target) and ``cross``
    (target queries x source keys)."""
    n = src + tgt
    enc = torch.zeros(n, n)
    enc[:src, :src] = 1
    cross = torch.zeros(n, n)
    cross[src:, :src] = 1
    return dict(encoder=enc, causal=causal_mask(n, src), cross=cross)


class SegmentLinear(nn.Module):
    """Per-token linear map with one weight set per token segment.

    Input (batch, tokens, in). Token t uses ``weight[g]``, ``bias[g]`` with
    ``g = segment_ids(segments)[t]``. ``active`` lists the segments the map
    acts on (default all); the others get fixed zero weights and biases, so
    inside a ``Residual`` their tokens pass through unchanged (an encoder
    block acting on the source segment only, say). Exports to
    ``fheml.TokenDense(..., segments=...)``.
    """

    def __init__(self, in_features, out_features, segments, active=None, bias=True):
        super().__init__()
        seg = _segment_ids(segments)
        groups = int(seg.max()) + 1
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("seg", seg)
        on = torch.zeros(groups)
        on[list(range(groups)) if active is None else list(active)] = 1
        self.register_buffer("on", on)
        bound = 1 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(groups, out_features, in_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(groups, out_features).uniform_(-bound, bound)) if bias else None

    def weights(self):
        """(groups, out, in), inactive segments zeroed."""
        return self.weight * self.on.view(-1, 1, 1)

    def token_bias(self):
        """(tokens, out) or None, inactive segments zeroed."""
        if self.bias is None:
            return None
        return (self.bias * self.on.view(-1, 1))[self.seg]

    def forward(self, x):
        w = self.weights()[self.seg]  # (tokens, out, in)
        y = torch.einsum("bti,toi->bto", x, w)
        b = self.token_bias()
        return y if b is None else y + b

    def extra_repr(self):
        active = [g for g in range(len(self.on)) if self.on[g]]
        return (f"in={self.in_features}, out={self.out_features}, segments={len(self.on)}, "
                f"active={active}")


class LayerNorm(nn.Module):
    """Per-token layer norm over the last dimension, as ``fheml.LayerNorm`` computes it.

    ``y = gamma * (x - mean) * r + beta`` with the biased variance and
    ``r = 1/sqrt(var + eps)`` evaluated as a degree-``degree`` Chebyshev series
    on ``[var_min, var_max]`` refined by ``newton`` Newton steps, exactly like
    the encrypted layer. Keep the per-token variance inside the interval (the
    series diverges outside it); ``eps`` of 1e-2..1e-1 keeps the function
    smooth down to ``var_min = 0``. ``dim`` must be a power of two.
    ``segments`` (see ``segment_ids``) gives every token segment its own
    gamma and beta, shape (segments, dim).
    """

    def __init__(self, dim, eps=1e-2, var_min=0.0, var_max=1.0, degree=15, newton=1, elementwise_affine=True,
                 segments=None):
        super().__init__()
        if dim < 2 or dim & (dim - 1):
            raise ValueError("dim must be a power of two, at least 2")
        if not (0 <= var_min < var_max) or eps < 0:
            raise ValueError("need 0 <= var_min < var_max and eps >= 0")
        self.dim, self.eps, self.var_min, self.var_max = int(dim), float(eps), float(var_min), float(var_max)
        self.degree, self.newton = int(degree), int(newton)
        seg = _segment_ids(segments)
        self.register_buffer("seg", seg)
        groups = 1 if seg is None else int(seg.max()) + 1
        g, b = torch.ones(dim), torch.zeros(dim)
        if seg is not None:
            g, b = torch.ones(groups, dim), torch.zeros(groups, dim)
        if elementwise_affine:
            self.gamma, self.beta = nn.Parameter(g), nn.Parameter(b)
        else:
            self.register_buffer("gamma", g)
            self.register_buffer("beta", b)
        self.register_buffer("c", _series_buffer(lambda v: 1.0 / np.sqrt(v + self.eps), var_min, var_max, degree))

    def inv_sqrt(self, var):
        """1/sqrt(var + eps) as the layer approximates it. While training, the
        series sees the variance clamped to its interval (a stray value would
        otherwise blow it up); check the ranges on data before encrypting."""
        if self.training:
            self._excess = _excess(var, self.var_min, self.var_max)
        v = var.clamp(self.var_min, self.var_max) if self.training else var
        r = _chebyshev_eval(self.c, self.var_min, self.var_max, v)
        for _ in range(self.newton):
            r = 1.5 * r - 0.5 * (var + self.eps) * r * r * r
        return r

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x * x).mean(-1, keepdim=True) - mean * mean
        g, b = (self.gamma, self.beta) if self.seg is None else (self.gamma[self.seg], self.beta[self.seg])
        return g * (x - mean) * self.inv_sqrt(var) + b

    def extra_repr(self):
        return (f"dim={self.dim}, eps={self.eps}, var=[{self.var_min}, {self.var_max}], degree={self.degree}, "
                f"newton={self.newton}")


class Attention(nn.Module):
    """Self-attention with a polynomial score instead of softmax.

    Input (batch, tokens, dim). ``A = c0 + c1 S + c2 S^2`` with ``S = Q K^T``
    (no 1/sqrt(d): it is learned into W_q). The default score is
    ``(S + 2)^2 / (8 n)``: non-negative and about 1/n at S = 0.

    ``heads`` > 1 splits the dim features into heads of width ``dim / heads``
    (a power of two) exactly like ``nn.MultiheadAttention``: head h uses rows
    ``[h*dh, (h+1)*dh)`` of q/k/v and the matching columns of o. ``score`` is
    a shared ``(c0, c1, c2)`` or one triple per head, shape ``(heads, 3)``.
    Under encryption extra heads are free (fewer rotations than one wide head).

    ``softmax`` replaces the quadratic score by the polynomial softmax of
    ``fheml.SoftmaxOptions``: ``True`` for the defaults or a dict with any of
    ``lo, hi`` (score interval of the exp series), ``exp_degree``,
    ``inv_degree, newton`` (1/x series on the row-sum interval and its Newton
    steps), ``subtract_max`` (subtract the row maximum first; exact here,
    scheme switching under encryption), ``sum_lo, sum_hi`` (row-sum interval,
    0 = derived from ``lo, hi``). The forward pass evaluates the same series,
    so train with the scores inside ``[lo, hi]``.

    ``mask`` (tokens x tokens, 1 where query i may attend to key j) and
    ``segments`` make the other attention patterns of a transformer: causal
    self-attention, and with segments (see ``segment_ids``) the attention of
    an encoder-decoder over one sequence of source and target tokens. The
    projections then hold one weight set per segment: q and o act on
    ``query_segments``, k and v on ``key_segments`` (default: all), the other
    segments get zeros. Cross-attention is ``query_segments=[1],
    key_segments=[0]`` with ``encoder_decoder_masks(...)["cross"]``. Masked
    pairs get no weight; a query with no allowed key outputs 0 (plus o's
    bias). Under encryption, token offsets that pair nothing are skipped.
    """

    SOFTMAX_DEFAULTS = dict(lo=-4.0, hi=4.0, exp_degree=15, inv_degree=15, newton=1, subtract_max=False,
                            sum_lo=0.0, sum_hi=0.0)

    def __init__(self, dim, tokens, score=None, bias=True, heads=1, softmax=None, mask=None, segments=None,
                 query_segments=None, key_segments=None):
        super().__init__()
        if heads < 1 or dim % heads:
            raise ValueError("heads must divide dim")
        if score is None:
            score = (4.0 / (8 * tokens), 4.0 / (8 * tokens), 1.0 / (8 * tokens))
        score = torch.as_tensor(score, dtype=torch.get_default_dtype())
        if score.shape == (3,):
            score = score.expand(heads, 3).clone()
        if score.shape != (heads, 3):
            raise ValueError("score must be (c0, c1, c2) or (heads, 3)")
        self.tokens, self.heads = tokens, heads
        if segments is None:
            if query_segments is not None or key_segments is not None:
                raise ValueError("query_segments / key_segments need segments")
            self.q, self.k, self.v, self.o = (nn.Linear(dim, dim, bias=bias) for _ in range(4))
        else:
            _segment_ids(segments, tokens)
            self.q, self.o = (SegmentLinear(dim, dim, segments, query_segments, bias) for _ in range(2))
            self.k, self.v = (SegmentLinear(dim, dim, segments, key_segments, bias) for _ in range(2))
        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.get_default_dtype())
            if mask.shape != (tokens, tokens) or not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("mask must be a (tokens, tokens) tensor of zeros and ones")
            if not mask.any():
                raise ValueError("mask allows no (query, key) pair")
        self.register_buffer("mask", mask)
        self.register_buffer("score", score)
        self.softmax = None
        if softmax:
            o = dict(self.SOFTMAX_DEFAULTS)
            extra = set(softmax) - set(o) if isinstance(softmax, dict) else set()
            if extra:
                raise ValueError(f"unknown softmax options {sorted(extra)}")
            if isinstance(softmax, dict):
                o.update(softmax)
            if not o["lo"] < o["hi"]:
                raise ValueError("softmax: need lo < hi")
            if o["subtract_max"] and mask is not None:
                raise ValueError("softmax: subtract_max does not support masks")
            if (o["sum_lo"] or o["sum_hi"]) and not 0 < o["sum_lo"] < o["sum_hi"]:
                raise ValueError("softmax: need 0 < sum_lo < sum_hi")
            self.softmax = o
            self.register_buffer("exp_c", _series_buffer(np.exp, self.exp_lo, self.exp_hi, o["exp_degree"]))
            self.register_buffer("inv_c", _series_buffer(lambda u: 1.0 / u, self.sum_lo, self.sum_hi, o["inv_degree"]))

    # Intervals of the softmax series, as fheml.Attention derives them.
    @property
    def exp_lo(self):
        o = self.softmax
        return o["lo"] - o["hi"] if o["subtract_max"] else o["lo"]

    @property
    def exp_hi(self):
        return 0.0 if self.softmax["subtract_max"] else self.softmax["hi"]

    @property
    def sum_lo(self):
        o = self.softmax
        return o["sum_lo"] or (1.0 if o["subtract_max"] else self.tokens * math.exp(o["lo"]))

    @property
    def sum_hi(self):
        o = self.softmax
        return o["sum_hi"] or (float(self.tokens) if o["subtract_max"] else self.tokens * math.exp(o["hi"]))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = (f(x).reshape(b, n, self.heads, d // self.heads).transpose(1, 2) for f in (self.q, self.k, self.v))
        s = q @ k.transpose(-1, -2)  # (b, heads, n, n)
        m = self.mask
        if self.softmax is None:
            c = self.score.view(1, self.heads, 1, 1, 3)
            a = c[..., 0] + c[..., 1] * s + c[..., 2] * s * s
            if m is not None:
                a = a * m
        else:
            if self.softmax["subtract_max"]:
                s = s - s.max(-1, keepdim=True).values
            # While training the series see their arguments clamped to their
            # intervals (outside them they diverge); check the ranges on data
            # before encrypting (fheml.Network.forward_plain does not clamp).
            if self.training:
                excess = _excess(s, self.exp_lo, self.exp_hi)
                s = s.clamp(self.exp_lo, self.exp_hi)
            e = _chebyshev_eval(self.exp_c, self.exp_lo, self.exp_hi, s)
            if m is not None:
                e = e * m
            total = e.sum(-1, keepdim=True)
            # Rows that attend to nothing (masked) sum to 0 and output 0;
            # under encryption their sum is padded into the 1/x interval.
            live = torch.ones(()) if m is None else (m.sum(-1, keepdim=True) > 0).to(total.dtype)
            pad = (1 - live) * 0.5 * (self.sum_lo + self.sum_hi)
            if self.training:
                self._excess = excess + _excess(total + pad, self.sum_lo, self.sum_hi)
            t = (total + pad).clamp(self.sum_lo, self.sum_hi) if self.training else total + pad
            y = _chebyshev_eval(self.inv_c, self.sum_lo, self.sum_hi, t)
            for _ in range(self.softmax["newton"]):
                y = y * (2 - (total + pad) * y)
            a = e * y * live
        return self.o((a @ v).transpose(1, 2).reshape(b, n, d))

    def extra_repr(self):
        r = f"tokens={self.tokens}, heads={self.heads}"
        if self.mask is not None:
            r += ", masked"
        if self.softmax:
            r += ", softmax " + ", ".join(f"{k}={v}" for k, v in self.softmax.items())
        return r


class Residual(nn.Module):
    """y = body(x) + x."""

    def __init__(self, *body):
        super().__init__()
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return x + self.body(x)


class TokenMean(nn.Module):
    """Mean over tokens: (batch, tokens, dim) -> (batch, dim)."""

    def forward(self, x):
        return x.mean(-2)


class PosEmbedding(nn.Module):
    """Learned per-token additive embedding; exported into the preceding token Linear's bias."""

    def __init__(self, tokens, dim):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(tokens, dim))

    def forward(self, x):
        return x + self.pos


class MaxPool2d(nn.MaxPool2d):
    """Exact max pooling, exported to ``fheml.MaxPool2d`` (CKKS <-> FHEW scheme switching).

    Computes ``torch.nn.MaxPool2d`` (square windows, no padding); ``range``
    bounds the inputs, ``|x| <= range``, as the encrypted comparison wraps
    around outside it: calibrate it like a Chebyshev interval. A k x k pool
    costs ceil(log2 k^2) comparison rounds (one FHEW bootstrap per window
    and pair) and 12 + ceil(log2 k^2) + 1 levels. Plain ``nn.MaxPool2d``
    exports too with ``from_torch(..., maxpool_range=R)``. ``fheml.Trainer`` trains through it,
    routing each window's gradient to the tap that won, like torch.
    """

    def __init__(self, kernel_size, stride=None, range=1.0):
        super().__init__(kernel_size, stride)
        if not range > 0:
            raise ValueError("MaxPool2d: range must be positive")
        self.range = float(range)

    def extra_repr(self):
        return super().extra_repr() + f", range={self.range}"


class Bootstrap(nn.Module):
    """Identity in torch; refreshes the ciphertext's levels when encrypted.

    Keep values roughly within [-1, 1] at this point.
    """

    def forward(self, x):
        return x
