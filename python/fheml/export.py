"""PyTorch -> fheml exporter.

``from_torch(model, input_shape)`` turns a trained ``torch.nn.Sequential``
(see ``fheml.nn`` for the HE-friendly modules) into an ``fheml.Network``.
``input_shape`` excludes the batch dimension:

* ``(C, H, W)`` image: Conv2d, AvgPool2d, AdaptiveAvgPool2d(1), BatchNorm2d;
  ``nn.Flatten`` then ``nn.Linear`` for the head.
* ``(n, d)`` token sequence: ``nn.Linear`` acts per token, plus Attention
  (quadratic or softmax scores), LayerNorm, Residual, PosEmbedding, TokenMean.
* ``(k,)`` vector: Linear, BatchNorm1d.

Activations must be ``fheml.nn`` polynomials (Poly, Square, Chebyshev):
``nn.ReLU`` and friends have no exact encrypted equivalent.
"""
import numpy as np
import torch
from torch import nn

from . import _fheml
from . import nn as hnn

__all__ = ["from_torch", "save_torch"]


def _np(t):
    return None if t is None else t.detach().cpu().double().numpy()


def _flat(a):
    return [] if a is None else np.ascontiguousarray(a, dtype=np.float64).ravel().tolist()


def _pair(v, what):
    if isinstance(v, (tuple, list)):
        if len(set(v)) != 1:
            raise ValueError(f"{what}: only square {what.split()[-1]}s are supported, got {tuple(v)}")
        v = v[0]
    return int(v)


class _Exporter:
    """Walks the module tree tracking the activation shape and emits a list of
    layer specs (plain tuples), so later passes can fold or pad weights."""

    def __init__(self, maxpool_range=None):
        self.max_token_width = 0
        self.maxpool_range = maxpool_range

    def run(self, modules, state):
        out = []
        for m in modules:
            state = self.module(m, state, out)
        return out, state

    def module(self, m, state, out):
        kind = state[0]
        name = type(m).__name__
        if isinstance(m, nn.Sequential):
            specs, state = self.run(m, state)
            out.extend(specs)
            return state
        if isinstance(m, (nn.Identity, nn.Dropout, hnn.Bootstrap)):
            if isinstance(m, hnn.Bootstrap):
                out.append(("bootstrap",))
            return state
        if isinstance(m, nn.Flatten):
            if m.start_dim != 1 or m.end_dim != -1:
                raise ValueError("nn.Flatten: only start_dim=1, end_dim=-1 is supported")
            return ("vector", int(np.prod(state[1:])))

        if isinstance(m, nn.Conv2d):
            self.need(kind, "image", name)
            c, h, w = state[1:]
            if m.groups != 1 or _pair(m.dilation, "Conv2d dilation") != 1 or m.padding_mode != "zeros":
                raise ValueError("nn.Conv2d: groups, dilation and padding modes other than the defaults are unsupported")
            if isinstance(m.padding, str):
                raise ValueError("nn.Conv2d: give padding as a number")
            k = _pair(m.kernel_size, "Conv2d kernel size")
            s, p = _pair(m.stride, "Conv2d stride"), _pair(m.padding, "Conv2d padding")
            out.append(("conv2d", m.in_channels, m.out_channels, k, k, s, p, _np(m.weight), _np(m.bias)))
            return ("image", m.out_channels, (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1)
        if isinstance(m, nn.AvgPool2d):
            self.need(kind, "image", name)
            if _pair(m.padding, "AvgPool2d padding") != 0 or m.ceil_mode:
                raise ValueError("nn.AvgPool2d: padding and ceil_mode are unsupported")
            if m.divisor_override:
                raise ValueError("nn.AvgPool2d: divisor_override is unsupported")
            k = _pair(m.kernel_size, "AvgPool2d kernel size")
            s = _pair(m.stride if m.stride is not None else k, "AvgPool2d stride")
            c, h, w = state[1:]
            out.append(("avgpool2d", k, s))
            return ("image", c, (h - k) // s + 1, (w - k) // s + 1)
        if isinstance(m, nn.AdaptiveAvgPool2d):
            self.need(kind, "image", name)
            c, h, w = state[1:]
            if _pair(m.output_size, "AdaptiveAvgPool2d output size") != 1 or h != w:
                raise ValueError("nn.AdaptiveAvgPool2d: only output_size=1 on square inputs is supported")
            out.append(("avgpool2d", h, h))
            return ("image", c, 1, 1)
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if isinstance(m, nn.BatchNorm2d):
                self.need(kind, "image", name)
            elif kind != "vector":
                raise ValueError("nn.BatchNorm1d is supported on vectors only (after Flatten/Linear/TokenMean)")
            if m.running_mean is None:
                raise ValueError(f"{name}: needs track_running_stats=True")
            scale = 1.0 / np.sqrt(_np(m.running_var) + m.eps)
            shift = -_np(m.running_mean) * scale
            if m.affine:
                scale, shift = scale * _np(m.weight), shift * _np(m.weight) + _np(m.bias)
            out.append(("affine", scale, shift))
            return state

        if isinstance(m, nn.Linear):
            if kind == "image":
                raise ValueError("nn.Linear on an image: add nn.Flatten() before it")
            if kind == "tokens":
                n, d = state[1:]
                self.need_width(d, m.in_features, name)
                self.max_token_width = max(self.max_token_width, m.in_features, m.out_features)
                out.append(("tokendense", n, m.in_features, m.out_features, _np(m.weight), _np(m.bias)))
                return ("tokens", n, m.out_features)
            self.need_width(state[1], m.in_features, name)
            out.append(("dense", m.in_features, m.out_features, _np(m.weight), _np(m.bias)))
            return ("vector", m.out_features)

        if isinstance(m, hnn.Square):
            out.append(("square",))
            return state
        if isinstance(m, hnn.Poly):
            out.append(("poly", _np(m.coeffs)))
            return state
        if isinstance(m, hnn.Chebyshev):
            out.append(("chebyshev", m.fn, m.a, m.b, m.degree))
            return state

        if isinstance(m, hnn.Attention):
            self.need(kind, "tokens", name)
            n, d = state[1:]
            self.need_width(d, m.q.in_features, name)
            ws = [_np(l.weight) for l in (m.q, m.k, m.v, m.o)]
            bs = [_np(l.bias) for l in (m.q, m.k, m.v, m.o)]
            out.append(("attention", n, d, ws, bs, _np(m.score), m.heads, m.softmax))
            return state
        if isinstance(m, hnn.LayerNorm):
            self.need(kind, "tokens", name)
            n, d = state[1:]
            self.need_width(d, m.dim, name)
            out.append(("layernorm", n, d, _np(m.gamma), _np(m.beta), m.eps, m.var_min, m.var_max, m.degree, m.newton))
            return state
        if isinstance(m, hnn.Residual):
            body, after = self.run(m.body, state)
            if after != state:
                raise ValueError(f"Residual: body changes the shape {state[1:]} -> {after[1:]}")
            out.append(("residual", body))
            return state
        if isinstance(m, hnn.PosEmbedding):
            self.need(kind, "tokens", name)
            pos = _np(m.pos)
            if pos.shape != tuple(state[1:]):
                raise ValueError(f"PosEmbedding: shape {pos.shape} does not match tokens {state[1:]}")
            out.append(("posembed", pos))
            return state
        if isinstance(m, hnn.TokenMean):
            self.need(kind, "tokens", name)
            n, d = state[1:]
            out.append(("tokenmean", n, d))
            return ("vector", d)

        if isinstance(m, (nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU)):
            raise ValueError(f"nn.{name} has no exact encrypted form: use fheml.nn.Chebyshev"
                             f"('{name.lower()}', a, b, degree) or a trained fheml.nn.Poly")
        if isinstance(m, nn.MaxPool2d):
            self.need(kind, "image", name)
            rng = m.range if isinstance(m, hnn.MaxPool2d) else self.maxpool_range
            if rng is None:
                raise ValueError("nn.MaxPool2d: pass from_torch(..., maxpool_range=R) with R >= max |input|, or use "
                                 "fheml.nn.MaxPool2d(k, stride, range) (exact max via CKKS<->FHEW scheme switching)")
            if _pair(m.padding, "MaxPool2d padding") != 0 or _pair(m.dilation, "MaxPool2d dilation") != 1 \
                    or m.ceil_mode or m.return_indices:
                raise ValueError("nn.MaxPool2d: padding, dilation, ceil_mode and return_indices are unsupported")
            k = _pair(m.kernel_size, "MaxPool2d kernel size")
            s = _pair(m.stride if m.stride is not None else k, "MaxPool2d stride")
            c, h, w = state[1:]
            out.append(("maxpool2d", k, s, float(rng)))
            return ("image", c, (h - k) // s + 1, (w - k) // s + 1)
        if isinstance(m, nn.LayerNorm):
            raise ValueError("nn.LayerNorm has no exact encrypted form: use fheml.nn.LayerNorm(dim, eps, var_min, "
                             "var_max, degree) on token sequences (a polynomial 1/sqrt on a calibrated variance range)")
        if isinstance(m, nn.Softmax):
            raise ValueError("nn.Softmax has no exact encrypted form: fheml.nn.Attention(dim, tokens, softmax={...}) "
                             "evaluates a polynomial softmax inside attention")
        if isinstance(m, nn.MultiheadAttention):
            raise ValueError(f"nn.{name} is not supported under encryption (see docs/DESIGN.md#transformers)"
                             " (fheml.nn.Attention(dim, tokens, heads=h) is the polynomial multi-head form)")
        raise ValueError(f"unsupported module {name}: build the model from nn.Sequential, the supported "
                         "torch.nn layers and fheml.nn modules")

    @staticmethod
    def need(kind, want, name):
        if kind != want:
            raise ValueError(f"{name} expects a {want} input here, got a {kind}")

    @staticmethod
    def need_width(have, want, name):
        if have != want:
            raise ValueError(f"{name}: expects {want} input features, the previous layer gives {have}")


def _fold(specs):
    """Folds batch-norm into the preceding Conv2d/Linear (keeps the conv's
    cheap gapped layout, unlike Network.fuse_linear) and PosEmbedding into the
    preceding token Linear's per-token bias."""
    out = []
    for s in specs:
        if s[0] == "residual":
            s = ("residual", _fold(s[1]))
        if s[0] == "affine" and out and out[-1][0] in ("conv2d", "dense"):
            prev, (scale, shift) = out[-1], s[1:]
            w, b = prev[-2], prev[-1]
            w = w * scale.reshape((-1,) + (1,) * (w.ndim - 1))
            b = shift if b is None else b * scale + shift
            out[-1] = prev[:-2] + (w, b)
            continue
        if s[0] == "posembed":
            if out and out[-1][0] == "tokendense":
                _, n, i, o, w, b = out[-1]
                out[-1] = ("tokendense", n, i, o, w, (0 if b is None else b) + s[1])
                continue
            n, d = s[1].shape  # no Linear to fold into: identity map with the embedding as bias
            s = ("tokendense", n, d, d, np.eye(d), s[1])
        out.append(s)
    return out


def _build(s):
    k = s[0]
    if k == "conv2d":
        return _fheml.Conv2d(*s[1:7], _flat(s[7]), _flat(s[8]))
    if k == "avgpool2d":
        return _fheml.AvgPool2d(*s[1:])
    if k == "maxpool2d":
        return _fheml.MaxPool2d(*s[1:])
    if k == "affine":
        return _fheml.ChannelAffine(_flat(s[1]), _flat(s[2]))
    if k == "dense":
        return _fheml.Dense(s[1], s[2], _flat(s[3]), _flat(s[4]))
    if k == "tokendense":
        return _fheml.TokenDense(s[1], s[2], s[3], _flat(s[4]), _flat(s[5]))
    if k == "tokenmean":
        return _fheml.TokenMean(s[1], s[2])
    if k == "square":
        return _fheml.Square()
    if k == "poly":
        return _fheml.Poly(_flat(s[1]))
    if k == "chebyshev":
        return _fheml.Chebyshev(*s[1:])
    if k == "bootstrap":
        return _fheml.Bootstrap()
    if k == "attention":
        _, n, d, ws, bs, score, heads, softmax = s
        att = _fheml.Attention(n, d, *map(_flat, ws), *map(_flat, bs), score=_flat(score), heads=heads)
        if softmax:
            o = _fheml.SoftmaxOptions()
            for key, v in softmax.items():
                setattr(o, key, v)
            att.set_softmax(o)
        return att
    if k == "layernorm":
        _, n, d, gamma, beta, eps, var_min, var_max, degree, newton = s
        return _fheml.LayerNorm(n, d, _flat(gamma), _flat(beta), eps, var_min, var_max, degree, newton)
    if k == "residual":
        return _fheml.Residual([_build(b) for b in s[1]])
    raise AssertionError(k)


def from_torch(model, input_shape, token_width="auto", check=True, maxpool_range=None):
    """Converts a trained torch model to an ``fheml.Network`` (not yet compiled).

    token_width: for token inputs, zero-pad each input token to this width
      ("auto" = the widest per-token layer, None = no padding). A wider token
      stride makes per-token layers much cheaper under encryption; the
      helpers ``fheml.infer``/``encrypt``/``plain`` pad inputs automatically.
    check: compare the fheml plaintext reference with the torch model on a
      random input and raise if they disagree.
    maxpool_range: input bound R of every plain nn.MaxPool2d (exported as the
      exact scheme-switching max pool; fheml.nn.MaxPool2d carries its own).
    """
    shape = tuple(int(v) for v in input_shape)
    if len(shape) == 3:
        state, net_shape = ("image",) + shape, _fheml.Shape(*shape)
    elif len(shape) == 2:
        state, net_shape = ("tokens",) + shape, _fheml.Shape(shape[0], 1, shape[1])
    elif len(shape) == 1:
        state, net_shape = ("vector",) + shape, _fheml.Shape(shape[0], 1, 1)
    else:
        raise ValueError("input_shape must be (C, H, W), (tokens, dim) or (features,)")

    ex = _Exporter(maxpool_range)
    modules = model if isinstance(model, nn.Sequential) else [model]
    with torch.no_grad():
        specs, _ = ex.run(modules, state)
    specs = _fold(specs)

    if state[0] == "tokens" and token_width is not None:
        n, d = shape
        width = ex.max_token_width if token_width == "auto" else int(token_width)
        if width > d and specs and specs[0][0] == "tokendense":
            _, n, i, o, w, b = specs[0]
            wp = np.zeros((o, width))
            wp[:, :i] = w
            specs[0] = ("tokendense", n, width, o, wp, b)
            net_shape = _fheml.Shape(n, 1, width)

    net = _fheml.Network(net_shape)
    for s in specs:
        net.add(_build(s))

    if check:
        _check(model, net, shape)
    return net


def save_torch(model, input_shape, path, **kwargs):
    """``from_torch`` followed by ``Network.save(path)``; returns the network."""
    net = from_torch(model, input_shape, **kwargs)
    net.save(path)
    return net


def _check(model, net, shape, tol=1e-6):
    from . import plain

    was_training = model.training
    model.eval()
    try:
        x = np.random.default_rng(0).uniform(-1, 1, shape)
        with torch.no_grad():
            p = next(iter(model.parameters()), None)
            dtype = p.dtype if p is not None else torch.get_default_dtype()
            ref = model(torch.as_tensor(x[None], dtype=dtype)).double().numpy().ravel()
        got = plain(net, x)
    finally:
        model.train(was_training)
    err = np.abs(got - ref).max() / max(1.0, np.abs(ref).max())
    # float32 models only agree to ~1e-6 relative.
    if err > (tol if dtype == torch.float64 else 1e-4):
        raise RuntimeError(f"export check failed: fheml plaintext differs from torch by {err:.2e} "
                           "(relative). Please report this model.")
