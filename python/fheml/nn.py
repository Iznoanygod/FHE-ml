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
           "PosEmbedding", "Bootstrap", "chebyshev_coefficients"]


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


def _series_buffer(fn, a, b, degree):
    return torch.as_tensor(chebyshev_coefficients(fn, a, b, degree), dtype=torch.get_default_dtype())


class Chebyshev(nn.Module):
    """Degree-``degree`` Chebyshev approximation of relu|sigmoid|tanh|silu|gelu on [a, b].

    The forward pass evaluates the polynomial itself, not the exact function,
    so training sees what encrypted inference computes. Inputs outside [a, b]
    blow up quickly, as they do under encryption: keep activations in range.
    """

    def __init__(self, fn, a, b, degree):
        super().__init__()
        if fn not in _FUNCTIONS:
            raise ValueError(f"unknown function {fn!r}; expected one of {sorted(_FUNCTIONS)}")
        self.fn, self.a, self.b, self.degree = fn, float(a), float(b), int(degree)
        self.register_buffer("c", torch.as_tensor(chebyshev_coefficients(fn, a, b, degree),
                                                  dtype=torch.get_default_dtype()))

    def forward(self, x):
        return _chebyshev_eval(self.c, self.a, self.b, x)

    def extra_repr(self):
        return f"{self.fn}, a={self.a}, b={self.b}, degree={self.degree}"


class LayerNorm(nn.Module):
    """Per-token layer norm over the last dimension, as ``fheml.LayerNorm`` computes it.

    ``y = gamma * (x - mean) * r + beta`` with the biased variance and
    ``r = 1/sqrt(var + eps)`` evaluated as a degree-``degree`` Chebyshev series
    on ``[var_min, var_max]`` refined by ``newton`` Newton steps, exactly like
    the encrypted layer. Keep the per-token variance inside the interval (the
    series diverges outside it); ``eps`` of 1e-2..1e-1 keeps the function
    smooth down to ``var_min = 0``. ``dim`` must be a power of two.
    """

    def __init__(self, dim, eps=1e-2, var_min=0.0, var_max=1.0, degree=15, newton=1, elementwise_affine=True):
        super().__init__()
        if dim < 2 or dim & (dim - 1):
            raise ValueError("dim must be a power of two, at least 2")
        if not (0 <= var_min < var_max) or eps < 0:
            raise ValueError("need 0 <= var_min < var_max and eps >= 0")
        self.dim, self.eps, self.var_min, self.var_max = int(dim), float(eps), float(var_min), float(var_max)
        self.degree, self.newton = int(degree), int(newton)
        g, b = torch.ones(dim), torch.zeros(dim)
        if elementwise_affine:
            self.gamma, self.beta = nn.Parameter(g), nn.Parameter(b)
        else:
            self.register_buffer("gamma", g)
            self.register_buffer("beta", b)
        self.register_buffer("c", _series_buffer(lambda v: 1.0 / np.sqrt(v + self.eps), var_min, var_max, degree))

    def inv_sqrt(self, var):
        """1/sqrt(var + eps) as the layer approximates it."""
        r = _chebyshev_eval(self.c, self.var_min, self.var_max, var)
        for _ in range(self.newton):
            r = 1.5 * r - 0.5 * (var + self.eps) * r * r * r
        return r

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x * x).mean(-1, keepdim=True) - mean * mean
        return self.gamma * (x - mean) * self.inv_sqrt(var) + self.beta

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
    """

    SOFTMAX_DEFAULTS = dict(lo=-4.0, hi=4.0, exp_degree=15, inv_degree=15, newton=1, subtract_max=False,
                            sum_lo=0.0, sum_hi=0.0)

    def __init__(self, dim, tokens, score=None, bias=True, heads=1, softmax=None):
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
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim, bias=bias) for _ in range(4))
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
        if self.softmax is None:
            c = self.score.view(1, self.heads, 1, 1, 3)
            a = c[..., 0] + c[..., 1] * s + c[..., 2] * s * s
        else:
            if self.softmax["subtract_max"]:
                s = s - s.max(-1, keepdim=True).values
            e = _chebyshev_eval(self.exp_c, self.exp_lo, self.exp_hi, s)
            total = e.sum(-1, keepdim=True)
            y = _chebyshev_eval(self.inv_c, self.sum_lo, self.sum_hi, total)
            for _ in range(self.softmax["newton"]):
                y = y * (2 - total * y)
            a = e * y
        return self.o((a @ v).transpose(1, 2).reshape(b, n, d))

    def extra_repr(self):
        r = f"tokens={self.tokens}, heads={self.heads}"
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
    costs k^2 - 1 comparisons (one FHEW bootstrap per slot each) and
    12 + ceil(log2 k^2) + 1 levels. ``fheml.Trainer`` trains through it,
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
