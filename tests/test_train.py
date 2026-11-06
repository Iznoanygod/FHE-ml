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
# Softmax cross-entropy against one-hot targets.
ce = fheml.cross_entropy(lo=-2.0, hi=2.0, exp_degree=15, inv_degree=31, newton=2)
y = rng.uniform(-2, 2, 3)
p = fheml.softmax(ce, y)
assert np.abs(p - np.exp(y) / np.exp(y).sum()).max() < 1e-8
xs, ts = rng.uniform(-1, 1, (2, 10)), np.eye(3)[[0, 2]]
enc, ref = build(), build()
trainer, ctx = fheml.make_trainer(enc, [], secure=False, bootstrapping=False, loss=ce)
print(trainer.summary())
trainer.encrypt_weights(ctx)
outs = fheml.train_step(trainer, ctx, xs, ts, lr=0.5)
print("cross-entropy before the update:", np.mean([fheml.cross_entropy_plain(o, t) for o, t in zip(outs, ts)]))
fheml.train_plain(ref, xs, ts, lr=0.5, loss=ce)
trainer.decrypt_weights(ctx)
err = np.abs(params(enc) - params(ref)).max()
print(f"cross-entropy: max weight error {err:.1e}")
assert err < 1e-4
assert np.abs(params(ref) - params(build())).max() > 1e-2

# Bootstrapping activations and gradients inside the step (level budget).
xs, ts = rng.uniform(-1, 1, (2, 10)), rng.uniform(-1, 1, (2, 3))
enc, ref = build(), build()
trainer, ctx = fheml.make_trainer(enc, [], secure=False, full_slots=False, level_budget="min", bootstrap_range=4)
print(trainer.summary())
assert trainer.step_depth == trainer.level_budget < trainer.unbounded_step_depth
try:
    trainer.set_level_budget(1)
    raise AssertionError("a budget below min_level_budget must be rejected")
except ValueError:
    pass
trainer.encrypt_weights(ctx)
fheml.train_step(trainer, ctx, xs, ts, lr=0.3)
assert trainer.sample_bootstraps > 0
fheml.train_plain(ref, xs, ts, lr=0.3)
trainer.decrypt_weights(ctx)
err = np.abs(params(enc) - params(ref)).max()
print(f"level budget {trainer.level_budget}: {trainer.sample_bootstraps} bootstraps per sample, "
      f"max weight error {err:.1e}")
assert err < 1e-3

# Optimizers: Nesterov momentum with weight decay, then AdamW, over two
# updates each (the encrypted state carries over).
for opt, lr in ((fheml.sgd(momentum=0.9, weight_decay=0.05, nesterov=True), 0.3),
                (fheml.adamw(weight_decay=0.1, betas=(0.9, 0.99), eps=0.04, v_max=4.0), 0.05)):
    enc, ref = build(), build()
    trainer, ctx = fheml.make_trainer(enc, [], secure=False, bootstrapping=False, updates=2, optimizer=opt)
    print(trainer.summary())
    trainer.encrypt_weights(ctx)
    state = fheml.OptimizerState()
    for step in range(2):
        xs, ts = rng.uniform(-1, 1, (2, 10)), rng.uniform(-1, 1, (2, 3))
        fheml.train_step(trainer, ctx, xs, ts, lr=lr)
        fheml.train_plain(ref, xs, ts, lr=lr, optimizer=opt, state=state)
    assert trainer.steps == state.steps == 2
    trainer.decrypt_weights(ctx)
    err = np.abs(params(enc) - params(ref)).max()
    print(f"{opt}: max weight error {err:.1e} after 2 updates")
    assert err < 1e-4
    assert np.abs(params(ref) - params(build())).max() > 1e-2
# A torch CNN with BatchNorm2d/BatchNorm1d (kept as trainable batch norms)
# and Dropout: the encrypted batched step matches train_plain, running
# statistics included.
try:
    import torch
    from torch import nn
    from fheml import nn as hnn
except ImportError:
    torch = None
if torch is not None:
    torch.manual_seed(3)

    def build_bn():
        torch.manual_seed(3)
        m = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.BatchNorm2d(2), hnn.Poly([0, 0.5, 0.25]),
                          nn.Dropout(0.25), nn.Flatten(), nn.Linear(32, 4), nn.BatchNorm1d(4), hnn.Square(),
                          nn.Linear(4, 2)).double()
        net = fheml.from_torch(m, (1, 4, 4), batchnorm_range=(0.0, 3.0))
        net.compile()
        return net

    enc, ref = build_bn(), build_bn()
    assert [l.type for l in enc.layers].count("batchnorm") == 2 and any(l.type == "dropout" for l in enc.layers)
    xb, tb = rng.uniform(-1, 1, (3, 1, 4, 4)), rng.uniform(-1, 1, (3, 2))
    trainer, ctx = fheml.make_trainer(enc, secure=False, bootstrapping=False)
    assert trainer.batch_coupled
    trainer.encrypt_weights(ctx)
    fheml.train_step(trainer, ctx, xb, tb, lr=0.5)
    fheml.train_plain(ref, xb, tb, lr=0.5)
    trainer.decrypt_weights(ctx)
    err = np.abs(params(enc) - params(ref)).max()
    bns = [(a, b) for a, b in zip(enc.layers, ref.layers) if a.type == "batchnorm"]
    run_err = max(np.abs(np.subtract(a.running_mean, b.running_mean)).max() +
                  np.abs(np.subtract(a.running_var, b.running_var)).max() for a, b in bns)
    print(f"batch norm + dropout: max weight error {err:.1e}, running statistics {run_err:.1e}")
    assert err < 1e-4 and run_err < 1e-4

    # Recurrent layers train by backpropagation through time: torch's
    # autograd through the same series (fheml.nn.GRU / LSTM) gives the
    # reference SGD step, which train_plain and the encrypted step match.
    for cls in (hnn.RNN, hnn.GRU, hnn.LSTM):
        torch.manual_seed(4)
        T, I, H = 3, 2, 3
        model = nn.Sequential(cls(I, H, last=True), nn.Linear(H, 2)).double()
        enc, ref = (fheml.from_torch(model, (T, I)) for _ in range(2))
        for n in (enc, ref):
            n.compile()
        xb, tb = rng.uniform(-1, 1, (2, T, I)), rng.uniform(-1, 1, (2, 2))
        opt = torch.optim.SGD(model.parameters(), lr=0.5)
        loss = 0.5 * ((model(torch.as_tensor(xb)) - torch.as_tensor(tb)) ** 2).sum() / len(xb)
        loss.backward()
        opt.step()
        rec = model[0]
        wih = np.zeros((rec.weight_ih.shape[0], ref.input_shape.w))  # the exporter pads tokens to the stride
        wih[:, :I] = rec.weight_ih.detach().numpy()
        want = np.concatenate([wih.ravel(), *(p.detach().numpy().ravel() for p in (rec.weight_hh, rec.bias_ih, rec.bias_hh))])
        fheml.train_plain(ref, [fheml._prepare(ref, x) for x in xb], tb, lr=0.5)
        torch_err = np.abs(np.asarray(ref.layers[0].params()) - want).max()
        trainer, ctx = fheml.make_trainer(enc, secure=False, bootstrapping=False)
        trainer.encrypt_weights(ctx)
        fheml.train_step(trainer, ctx, [fheml._prepare(enc, x) for x in xb], tb, lr=0.5)
        trainer.decrypt_weights(ctx)
        err = np.abs(params(enc) - params(ref)).max()
        print(f"{cls.__name__:4s} BPTT: train_plain vs torch autograd {torch_err:.1e}, encrypted vs train_plain {err:.1e}")
        assert torch_err < 1e-9 and err < 1e-4
print("PASSED")
