"""HE-friendly PyTorch modules that ``fheml.from_torch`` knows how to export.

Train a model built from ``torch.nn.Sequential``, the supported ``torch.nn``
layers (Conv2d, Linear, AvgPool2d, AdaptiveAvgPool2d(1), BatchNorm, Flatten,
Dropout, Identity) and the modules below, then export it::

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

__all__ = ["Poly", "Square", "Chebyshev", "Attention", "Residual", "TokenMean", "PosEmbedding", "Bootstrap",
           "chebyshev_coefficients"]


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
    """Chebyshev interpolant of ``fn`` on [a, b] (same nodes as OpenFHE)."""
    f = _FUNCTIONS[fn]
    return np.polynomial.chebyshev.chebinterpolate(lambda t: f((b - a) / 2 * t + (a + b) / 2), degree)


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
        t = (2 * x - (self.a + self.b)) / (self.b - self.a)
        b1, b2 = torch.zeros_like(x), torch.zeros_like(x)
        for c in reversed(self.c[1:]):  # Clenshaw
            b1, b2 = 2 * t * b1 - b2 + c, b1
        return t * b1 - b2 + self.c[0]

    def extra_repr(self):
        return f"{self.fn}, a={self.a}, b={self.b}, degree={self.degree}"


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
    """

    def __init__(self, dim, tokens, score=None, bias=True, heads=1):
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

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = (f(x).reshape(b, n, self.heads, d // self.heads).transpose(1, 2) for f in (self.q, self.k, self.v))
        s = q @ k.transpose(-1, -2)  # (b, heads, n, n)
        c = self.score.view(1, self.heads, 1, 1, 3)
        a = c[..., 0] + c[..., 1] * s + c[..., 2] * s * s
        return self.o((a @ v).transpose(1, 2).reshape(b, n, d))

    def extra_repr(self):
        return f"tokens={self.tokens}, heads={self.heads}"


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


class Bootstrap(nn.Module):
    """Identity in torch; refreshes the ciphertext's levels when encrypted.

    Keep values roughly within [-1, 1] at this point.
    """

    def forward(self, x):
        return x
