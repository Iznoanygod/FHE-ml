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
    Loss,
    Network,
    Residual,
    Scale,
    Shape,
    Square,
    Trainer,
    cross_entropy_plain,
    gpu_backend_built,
    gpu_device_count,
)


def _arg(v):
    # numpy arrays of any shape, and nested lists such as a per-head score
    # [[c0, c1, c2], ...], -> flat C-order lists (weights are [out][in]).
    nested = isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (list, tuple, np.ndarray))
    return np.ascontiguousarray(v, dtype=np.float64).ravel().tolist() if isinstance(v, np.ndarray) or nested else v


def _accepting_numpy(cls):
    class Wrapped(cls):
        __doc__ = cls.__doc__

        def __init__(self, *args, **kwargs):
            super().__init__(*map(_arg, args), **{k: _arg(v) for k, v in kwargs.items()})

    Wrapped.__name__ = Wrapped.__qualname__ = cls.__name__
    return Wrapped


Attention = _accepting_numpy(_fheml.Attention)
AvgPool2d = _fheml.AvgPool2d
MaxPool2d = _fheml.MaxPool2d
ChannelAffine = _accepting_numpy(_fheml.ChannelAffine)
Chebyshev = _fheml.Chebyshev
Conv2d = _accepting_numpy(_fheml.Conv2d)
Dense = _accepting_numpy(_fheml.Dense)
Poly = _accepting_numpy(_fheml.Poly)
TokenDense = _accepting_numpy(_fheml.TokenDense)
TokenMean = _fheml.TokenMean
LayerNorm = _accepting_numpy(_fheml.LayerNorm)
SoftmaxOptions = _fheml.SoftmaxOptions


def cross_entropy(**softmax):
    """Softmax cross-entropy training loss (targets one-hot).

    Keyword arguments set SoftmaxOptions fields: ``lo``/``hi`` bound the
    logits (exp is approximated there), ``exp_degree``, ``inv_degree``,
    ``newton``, ``sum_lo``/``sum_hi`` (the 1/x interval; default
    k e^lo .. k e^hi), ``subtract_max`` (largest logit subtracted through
    scheme-switching comparisons; then exp lives on [lo - hi, 0]).
    """
    o = SoftmaxOptions()
    for k, v in softmax.items():
        setattr(o, k, v)
    return Loss.cross_entropy(o)


def softmax(loss, y):
    """The softmax a cross-entropy ``Trainer`` evaluates under encryption."""
    return np.asarray(_fheml.softmax_plain(loss, _flat(y)))


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


def make_context(net, secure=True, rotation_window=0, **overrides):
    """Create a CKKS context with keys for a compiled network.

    ``rotation_window`` > 0 generates a small ladder of rotation keys and
    composes the linear layers' rotations from it (slower, far less key
    memory at ring 2^16). ``overrides`` set ContextOptions fields, e.g.
    scale_mod_bits=45, or use_gpu=False to stay on the CPU in a GPU
    (FIDESlib) build.
    """
    opts = net.context_options(secure)
    opts.rotation_window = rotation_window
    for k, v in overrides.items():
        setattr(opts, k, v)
    ctx = Context(opts)
    ctx.keygen(net.rotation_indices(rotation_window > 0))
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


def make_training_context(trainer, secure=True, bootstrapping=True, updates=1, rotation_window=None, **overrides):
    """CKKS context with every key encrypted training needs.

    Without bootstrapping, ``updates`` SGD updates fit before the weights run
    out of levels. ``rotation_window`` > 0 composes the training rotations
    from a small ladder of keys instead of one key per weight diagonal
    (default: 1 when secure, since each key is about 100 MB at ring 2^16).
    ``overrides`` set ContextOptions fields.
    """
    if rotation_window is None:
        rotation_window = 1 if secure else 0
    opts = trainer.context_options(secure, bootstrapping, updates, rotation_window)
    for k, v in overrides.items():
        setattr(opts, k, v)
    ctx = Context(opts)
    ctx.keygen(trainer.rotation_indices(rotation_window > 0))
    return ctx


def make_trainer(net, frozen=(), secure=True, bootstrapping=True, updates=1, full_slots=True, slots=None,
                 rotation_window=None, loss=None, level_budget=None, bootstrap_range=1.0, **overrides):
    """Encrypted-training setup: returns ``(trainer, ctx)``.

    full_slots recompiles ``net`` to use every slot of the ring (free per
    operation), so one bootstrap refreshes many weight ciphertexts at once;
    ``slots`` caps that (bootstrapping keys grow with the slot count: at ring
    2^16 about 45 keys for 256 slots, 96 for 4096). See
    make_training_context for rotation_window. ``loss``: a ``Loss``
    (default mean squared error; see ``cross_entropy``).

    level_budget bootstraps activations and gradients inside each step so
    that no ciphertext goes deeper than that many levels (deep networks:
    the step otherwise needs its whole depth between two bootstraps);
    "min" picks Trainer.min_level_budget. bootstrap_range bounds the slot
    values at those bootstraps (see Trainer.set_level_budget).
    """
    loss = loss or Loss.mse()
    if rotation_window is None:
        rotation_window = 1 if secure else 0

    def new_trainer():
        t = Trainer(net, list(frozen), loss)
        if level_budget and bootstrapping:
            t.set_level_budget(t.min_level_budget if level_budget == "min" else level_budget, bootstrap_range)
        return t

    def options(tr):
        opts = tr.context_options(secure, bootstrapping, updates, rotation_window)
        for k, v in overrides.items():
            setattr(opts, k, v)
        return opts

    trainer = new_trainer()
    opts = options(trainer)
    if full_slots or slots:
        want = slots or Context.ring_dim_for(opts) // 2
        if want != net.slots and want >= net.required_slots():
            net.compile(want)
            trainer = new_trainer()
            opts = options(trainer)
    ctx = Context(opts)
    ctx.keygen(trainer.rotation_indices(rotation_window > 0))
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


def train_plain(net, xs, targets, lr, frozen=(), loss=None):
    """Plaintext reference of ``Trainer``: one SGD update on a batch."""
    _fheml.train_plain(net, [_flat(x) for x in xs], [_flat(t) for t in targets], lr, list(frozen),
                       loss or Loss.mse())


def auto_bootstrap(net, calibration, budget=None, level_budget=(2, 2), max_ring=1 << 16, headroom=1.5,
                   fold_intervals=True, **context_overrides):
    """Make a deep network fit: fold Chebyshev interval maps into the preceding
    layers (one level each), then insert bootstraps so that no stretch
    exceeds ``budget`` levels (default: the most that keeps the 128-bit
    bootstrapping context at ring ``max_ring``; with ``level_budget`` (2, 2)
    and 50-bit scaling that is 6 levels at ring 2^16). ``calibration`` are
    plain inputs used to scale the values entering each bootstrap into
    [-1, 1]. Recompiles ``net`` and returns ``(bootstraps, budget)``. Pass the
    same ``level_budget`` (and any ``context_overrides``, e.g.
    key_switch_digits) to make_context.
    """
    if fold_intervals:
        net.fold_activation_intervals()
    net.compile()
    if budget is None:
        opts = net.context_options(True)
        opts.bootstrapping = True
        opts.scale_mod_bits, opts.first_mod_bits = 50, 57
        opts.level_budget = list(level_budget)
        for k, v in context_overrides.items():
            setattr(opts, k, v)
        budget = Network.max_bootstrap_budget(opts, max_ring)
    n = net.place_bootstraps(budget, [_prepare(net, x) for x in calibration], headroom)
    net.compile()
    return n, budget


def plain(net, x):
    """Plaintext reference forward pass (same approximations as encrypted)."""
    return np.asarray(net.forward_plain(_prepare(net, x)))
