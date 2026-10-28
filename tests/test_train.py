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
    return np.concatenate([l.params() for l in net.layers if isinstance(l, fheml.LinearLayer)])


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
print("PASSED")
