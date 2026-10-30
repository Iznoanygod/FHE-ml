"""Encrypted training from Python: one encrypted SGD update must match
fheml.train_plain on the same batch (toy parameters).

    PYTHONPATH=build/python python3 tests/test_train.py
"""
import numpy as np

import fheml

rng = np.random.default_rng(0)


def build():
    r = np.random.default_rng(1)
    net = fheml.Network(fheml.Shape(10))
    net.add(fheml.Dense(10, 6, r.uniform(-0.4, 0.4, (6, 10)), r.uniform(-0.1, 0.1, 6)))
    net.add(fheml.Poly([0, 0.5, 0.25]))
    net.add(fheml.Dense(6, 3, r.uniform(-0.4, 0.4, (3, 6)), r.uniform(-0.1, 0.1, 3)))
    net.compile()
    return net


def params(net):
    return np.concatenate([l.params() for l in net.layers if l.trainable])


xs, ts = rng.uniform(-1, 1, (4, 10)), rng.uniform(-1, 1, (4, 3))
for frozen in ([], [0]):
    enc, ref = build(), build()
    trainer, ctx = fheml.make_trainer(enc, frozen, secure=False, bootstrapping=False)
    print(trainer.summary())
    trainer.encrypt_weights(ctx)
    fheml.train_step(trainer, ctx, xs, ts, lr=0.5)
    fheml.train_plain(ref, xs, ts, lr=0.5, frozen=frozen)
    trainer.decrypt_weights(ctx)
    err = np.abs(params(enc) - params(ref)).max()
    print(f"frozen={frozen}: max weight error {err:.1e}")
    assert err < 1e-4
    assert np.abs(params(ref) - params(build())).max() > 1e-2  # the update did something


# Layer norm and (quadratic) attention train too.
def build_tokens():
    r = np.random.default_rng(2)
    n, d = 3, 4
    net = fheml.Network(fheml.Shape(n, 1, 5))
    net.add(fheml.TokenDense(n, 5, d, r.uniform(-0.4, 0.4, (d, 5)), r.uniform(-0.1, 0.1, d)))
    net.add(fheml.LayerNorm(n, d, r.uniform(0.5, 1.5, d), r.uniform(-0.2, 0.2, d), eps=0.05, var_min=0.0,
                            var_max=1.0, degree=31, newton=1))
    net.add(fheml.Attention(n, d, *(r.uniform(-0.3, 0.3, (d, d)) for _ in range(4)),
                            *(r.uniform(-0.1, 0.1, d) for _ in range(4)), score=[0.1, 0.1, 0.03], heads=2))
    net.add(fheml.TokenMean(n, d))
    net.add(fheml.Dense(d, 2, r.uniform(-0.4, 0.4, (2, d)), r.uniform(-0.1, 0.1, 2)))
    net.compile()
    return net


xs, ts = rng.uniform(-1, 1, (2, 3, 5)), rng.uniform(-1, 1, (2, 2))
enc, ref = build_tokens(), build_tokens()
trainer, ctx = fheml.make_trainer(enc, [], secure=False, bootstrapping=False)
print(trainer.summary())
trainer.encrypt_weights(ctx)
fheml.train_step(trainer, ctx, xs, ts, lr=0.5)
fheml.train_plain(ref, xs, ts, lr=0.5)
trainer.decrypt_weights(ctx)
err = np.abs(params(enc) - params(ref)).max()
print(f"layernorm + attention: max weight error {err:.1e}")
assert err < 1e-4
assert np.abs(params(ref) - params(build_tokens())).max() > 1e-3
print("PASSED")
