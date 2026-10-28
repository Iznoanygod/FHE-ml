"""Encrypted neural-network inference on OpenFHE CKKS.

Thin numpy-friendly layer over the C++ library (module ``fheml._fheml``).
Layer constructors accept numpy arrays of any shape for weights (flattened in
C order, i.e. [out][in] for dense maps)::

    import numpy as np, fheml
    net = fheml.Network.load("models/mnist_cnn.fheml")
    net.compile()
    ctx = fheml.make_context(net)               # 128-bit CKKS sized for net
    y = fheml.infer(net, ctx, image)            # encrypt -> forward -> decrypt

Models trained in PyTorch (with the HE-friendly modules in ``fheml.nn``)
convert with ``fheml.from_torch(model, input_shape)``.
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
    Trainer,
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


def _prepare(net, x):
    # Token inputs narrower than the network's token stride (from_torch pads
    # the first per-token layer) are zero-padded to it.
    x = np.asarray(x, dtype=np.float64)
    s = net.input_shape
    if x.size != s.size() and x.ndim >= 1 and x.shape[-1] < s.w and x.size // x.shape[-1] == s.c * s.h:
        x = np.concatenate([x.reshape(-1, x.shape[-1]), np.zeros((s.c * s.h, s.w - x.shape[-1]))], axis=1)
    return _flat(x)


def __getattr__(name):
    # The torch exporter is imported on first use so fheml does not need torch.
    if name in ("from_torch", "save_torch"):
        from . import export
        return getattr(export, name)
    raise AttributeError(f"module 'fheml' has no attribute {name!r}")


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
    return ctx.encrypt(_prepare(net, x), net.input_level(ctx))


def decrypt(net, ctx, y):
    """Decrypt a network output to a numpy array."""
    return np.asarray(net.decrypt_output(ctx, y))


def infer(net, ctx, x):
    """Encrypt ``x``, run ``net`` homomorphically and decrypt the result."""
    return decrypt(net, ctx, net.forward(ctx, encrypt(net, ctx, x)))


def make_training_context(trainer, secure=True, bootstrapping=True, updates=1, **overrides):
    """CKKS context with every key encrypted training needs.

    Without bootstrapping, ``updates`` SGD updates fit before the weights run
    out of levels. ``overrides`` set ContextOptions fields.
    """
    opts = trainer.context_options(secure, bootstrapping, updates)
    for k, v in overrides.items():
        setattr(opts, k, v)
    ctx = Context(opts)
    ctx.keygen(trainer.rotation_indices())
    return ctx


def make_trainer(net, frozen=(), secure=True, bootstrapping=True, updates=1, full_slots=True, **overrides):
    """Encrypted-training setup: returns ``(trainer, ctx)``.

    full_slots recompiles ``net`` to use every slot of the ring (free per
    operation), so one bootstrap refreshes many weight ciphertexts at once.
    """
    def options(tr):
        opts = tr.context_options(secure, bootstrapping, updates)
        for k, v in overrides.items():
            setattr(opts, k, v)
        return opts

    trainer = Trainer(net, list(frozen))
    opts = options(trainer)
    if full_slots:
        full = Context.ring_dim_for(opts) // 2
        if full > net.slots:
            net.compile(full)
            trainer = Trainer(net, list(frozen))
            opts = options(trainer)
    ctx = Context(opts)
    ctx.keygen(trainer.rotation_indices())
    return trainer, ctx


def train_step(trainer, ctx, xs, targets, lr):
    """One encrypted SGD update on a batch of (numpy) samples.

    The client encrypts each input and target; the server accumulates
    gradients and updates the encrypted weights. Returns the decrypted
    outputs (for monitoring; only the key holder can do this).
    """
    outs = []
    for x, t in zip(xs, targets):
        y = trainer.accumulate(ctx, trainer.encrypt_input(ctx, _flat(x)), trainer.encrypt_target(ctx, _flat(t)))
        outs.append(trainer.decrypt_output(ctx, y))
    trainer.update(ctx, lr)
    return np.asarray(outs)


def train_plain(net, xs, targets, lr, frozen=()):
    """Plaintext reference of ``Trainer``: one SGD update on a batch."""
    _fheml.train_plain(net, [_flat(x) for x in xs], [_flat(t) for t in targets], lr, list(frozen))


def plain(net, x):
    """Plaintext reference forward pass (same approximations as encrypted)."""
    return np.asarray(net.forward_plain(_prepare(net, x)))
