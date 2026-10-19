"""Encrypted neural-network inference on OpenFHE CKKS.

Thin numpy-friendly layer over the C++ library (module ``fheml._fheml``).
Layer constructors accept numpy arrays of any shape for weights (flattened in
C order, i.e. [out][in] for dense maps)::

    import numpy as np, fheml
    net = fheml.Network.load("models/mnist_cnn.fheml")
    net.compile()
    ctx = fheml.make_context(net)               # 128-bit CKKS sized for net
    y = fheml.infer(net, ctx, image)            # encrypt -> forward -> decrypt
"""
import numpy as np

from . import _fheml
from ._fheml import (  # noqa: F401
    Bootstrap,
    Ciphertext,
    Context,
    ContextOptions,
    Layer,
    LinearLayer,
    Network,
    Residual,
    Shape,
    Square,
)


def _arg(v):
    # numpy arrays of any shape -> flat C-order lists (weights are [out][in]).
    return np.ascontiguousarray(v, dtype=np.float64).ravel().tolist() if isinstance(v, np.ndarray) else v


def _accepting_numpy(cls):
    class Wrapped(cls):
        __doc__ = cls.__doc__

        def __init__(self, *args, **kwargs):
            super().__init__(*map(_arg, args), **{k: _arg(v) for k, v in kwargs.items()})

    Wrapped.__name__ = Wrapped.__qualname__ = cls.__name__
    return Wrapped


Attention = _accepting_numpy(_fheml.Attention)
AvgPool2d = _fheml.AvgPool2d
ChannelAffine = _accepting_numpy(_fheml.ChannelAffine)
Chebyshev = _fheml.Chebyshev
Conv2d = _accepting_numpy(_fheml.Conv2d)
Dense = _accepting_numpy(_fheml.Dense)
Poly = _accepting_numpy(_fheml.Poly)
TokenDense = _accepting_numpy(_fheml.TokenDense)
TokenMean = _fheml.TokenMean


def _flat(a):
    return np.ascontiguousarray(a, dtype=np.float64).ravel().tolist()


def make_context(net, secure=True, **overrides):
    """Create a CKKS context with keys for a compiled network.

    ``overrides`` set ContextOptions fields, e.g. scale_mod_bits=45.
    """
    opts = net.context_options(secure)
    for k, v in overrides.items():
        setattr(opts, k, v)
    ctx = Context(opts)
    ctx.keygen(net.rotation_indices())
    return ctx


def encrypt(net, ctx, x):
    """Encrypt an input array (any shape, flattened C-order) for ``net``."""
    return ctx.encrypt(_flat(x), net.input_level(ctx))


def decrypt(net, ctx, y):
    """Decrypt a network output to a numpy array."""
    return np.asarray(net.decrypt_output(ctx, y))


def infer(net, ctx, x):
    """Encrypt ``x``, run ``net`` homomorphically and decrypt the result."""
    return decrypt(net, ctx, net.forward(ctx, encrypt(net, ctx, x)))


def plain(net, x):
    """Plaintext reference forward pass (same approximations as encrypted)."""
    return np.asarray(net.forward_plain(_flat(x)))
