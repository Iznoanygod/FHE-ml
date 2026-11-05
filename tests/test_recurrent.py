"""Recurrent layers (fheml.nn.RNN / GRU / LSTM -> fheml.Recurrent): the
plaintext reference against torch, the model file round trip, and encrypted
inference with bootstrapping inside the recurrence (toy parameters).

    PYTHONPATH=build/python python3 tests/test_recurrent.py [--quick]
"""
import os
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn

import fheml
from fheml import nn as hnn

torch.manual_seed(0)
rng = np.random.default_rng(0)
quick = "--quick" in sys.argv
T, I, H = 5, 3, 4


def model_for(cls, last):
    head = [nn.Linear(H, 2)] if last else [hnn.TokenMean(), nn.Linear(H, 2)]
    return nn.Sequential(nn.Linear(I, I), cls(I, H, last=last), *head).double().eval()


def ref(model, x):
    with torch.no_grad():
        return model(torch.as_tensor(x[None])).numpy().ravel()


# Plaintext reference (Jacobi iteration over all tokens, slot by slot) ==
# torch's sequential recurrence with the same Chebyshev series.
for cls in (hnn.RNN, hnn.GRU, hnn.LSTM):
    for last in (False, True):
        model = model_for(cls, last)
        net = fheml.from_torch(model, (T, I))  # raises if plaintext != torch
        x = rng.uniform(-1, 1, (T, I))
        err = np.abs(fheml.plain(net, x) - ref(model, x)).max()
        assert err < 1e-8, (cls.__name__, last, err)
        rec = net.layers[1]
        assert rec.type == "recurrent" and rec.cell == cls.__name__.lower() and rec.last == last
        g, c = rec.series_ranges([np.asarray(net.layers[0].forward_plain(fheml._prepare(net, x)))])
        assert 0 < g < 1 and (c == 0 if cls is hnn.RNN else 0 < c < 1)
        # model file round trip
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.fheml")
            net.save(path)
            again = fheml.Network.load(path)
        assert np.abs(fheml.plain(again, x) - fheml.plain(net, x)).max() < 1e-12
        print(f"{cls.__name__:4s} {'last' if last else 'seq '}: plaintext vs torch {err:.1e}, depth {net.depth}")

# Weights of a trained torch.nn.LSTM load into the HE-friendly twin.
lstm = nn.LSTM(I, H, batch_first=True)
twin = hnn.LSTM(I, H).load_torch(lstm)
assert torch.equal(twin.weight_hh, lstm.weight_hh_l0) and torch.equal(twin.bias_ih, lstm.bias_ih_l0)
with torch.no_grad():
    x = torch.rand(2, T, I) - 0.5
    gap = (twin(x) - lstm(x)[0]).abs().max().item()
assert gap < 0.1, gap  # degree-13 series on [-16, 16] / [-8, 8] vs the exact sigmoid/tanh
try:
    fheml.from_torch(nn.Sequential(lstm), (T, I))
    raise AssertionError("nn.LSTM must be rejected")
except ValueError:
    pass

# range_loss sees the recurrent series while training.
twin.train()
twin(100 * torch.ones(1, T, I))
assert hnn.range_loss(twin) > 0
twin.eval()

# A short RNN fits without bootstrapping.
model = model_for(hnn.RNN, True)
model[1] = hnn.RNN(I, H, last=True, degree=5).double()
net = fheml.from_torch(model, (2, I))
net.compile()
ctx = fheml.make_context(net, secure=False)
x = rng.uniform(-1, 1, (2, I))
err = np.abs(fheml.infer(net, ctx, x) - ref(model, x)).max()
print(f"rnn 2 tokens, no bootstrapping: depth {net.depth}, encrypted vs torch {err:.1e}")
assert err < 1e-4

# Encrypted, bootstrapping inside the recurrence (a 5-level stretch, as at
# 128-bit security with --digits 4, on a toy ring).
cells = (hnn.LSTM,) if quick else (hnn.RNN, hnn.GRU, hnn.LSTM)
for cls in cells:
    model = model_for(cls, cls is not hnn.GRU)
    net = fheml.from_torch(model, (T, I))
    calib = rng.uniform(-1, 1, (8, T, I))
    fheml.auto_bootstrap(net, calib, budget=5)
    rec = net.layers[1]
    assert net.uses_bootstrapping and net.internal_bootstraps > 0
    ctx = fheml.make_context(net, secure=False, ring_dim=4096)
    x = rng.uniform(-1, 1, (T, I))
    t = time.time()
    y = fheml.infer(net, ctx, x)
    err = np.abs(y - ref(model, x)).max()
    print(f"{cls.__name__:4s} encrypted: {net.internal_bootstraps} bootstraps, at most "
          f"{net.depth_between_bootstraps} levels between them, err {err:.1e}, {time.time() - t:.1f}s")
    assert err < 5e-3, (cls.__name__, err)

print("recurrent tests PASSED")
