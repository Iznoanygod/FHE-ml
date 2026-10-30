"""PyTorch exporter test: exports small torch models and checks the fheml
plaintext reference and encrypted inference against torch (toy parameters).

    PYTHONPATH=build/python python3 tests/test_torch.py
"""
import numpy as np
import torch
from torch import nn

import fheml
from fheml import nn as hnn

torch.manual_seed(0)
rng = np.random.default_rng(0)


def run(name, model, shape, tol=1e-3):
    model = model.double().eval()
    with torch.no_grad():
        for m in model.modules():  # non-trivial batch-norm statistics
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.running_mean.uniform_(-0.2, 0.2)
                m.running_var.uniform_(0.5, 1.5)
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.2, 0.2)
    net = fheml.from_torch(model, shape)  # also checks plaintext vs torch
    net.compile()
    print(f"--- {name}\n{net.summary()}")
    ctx = fheml.make_context(net, secure=False)
    x = rng.uniform(-1, 1, shape)
    with torch.no_grad():
        ref = model(torch.as_tensor(x[None])).numpy().ravel()
    enc = fheml.infer(net, ctx, x)
    err = np.abs(enc - ref).max()
    print(f"torch {np.round(ref, 4)}\nfheml {np.round(enc, 4)}\nmax err {err:.1e}")
    assert err < tol, name
    return net


cnn = nn.Sequential(
    nn.Conv2d(2, 4, 3, stride=2, padding=1), nn.BatchNorm2d(4), hnn.Poly([0, 0.5, 0.25]),
    nn.Conv2d(4, 4, 3, padding=1, bias=False), nn.AvgPool2d(2), hnn.Chebyshev("silu", -3, 3, 13),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 6), nn.BatchNorm1d(6), hnn.Square(), nn.Dropout(0.1),
    nn.Linear(6, 3))
run("cnn", cnn, (2, 8, 8))

# Exact max pooling (scheme switching): small conv weights keep the pooled
# values inside range.
pooled = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), hnn.MaxPool2d(2, range=2.0), hnn.Square(), nn.Flatten(),
                       nn.Linear(8, 2))
with torch.no_grad():
    pooled[0].weight.mul_(0.5)
run("max pool", pooled, (1, 4, 4))

n, d, h = 4, 8, 16
vit = nn.Sequential(
    nn.Linear(6, d), hnn.PosEmbedding(n, d),
    hnn.Residual(hnn.Attention(d, n)),
    hnn.Residual(hnn.Attention(d, n, heads=2, score=[[0.1, 0.1, 0.03], [0.1, 0.05, 0.02]])),
    hnn.Residual(nn.Linear(d, h), hnn.Poly([0, 0.5, 0.25], trainable=True), nn.Linear(h, d)),
    hnn.TokenMean(), nn.Linear(d, 3))
net = run("transformer", vit, (n, 6))
assert net.input_shape.w == h, "tokens are padded to the widest per-token layer"

# Pre-norm block with softmax attention: hnn.LayerNorm and the softmax of
# hnn.Attention evaluate the library's polynomial approximations.
sm = dict(lo=-1.5, hi=1.5, exp_degree=15, inv_degree=31, newton=2)
prenorm = nn.Sequential(
    nn.Linear(6, d),
    hnn.Residual(hnn.LayerNorm(d, eps=0.05, var_min=0.0, var_max=1.0, degree=31), hnn.Attention(d, n, heads=2, softmax=sm)),
    hnn.TokenMean(), nn.Linear(d, 3))
with torch.no_grad():
    for lin in (prenorm[1].body[1].q, prenorm[1].body[1].k):
        lin.weight.mul_(0.5)  # scores inside [lo, hi]
    prenorm[1].body[0].gamma.uniform_(0.5, 1.5)
    prenorm[1].body[0].beta.uniform_(-0.2, 0.2)
run("pre-norm softmax block", prenorm, (n, 6))
for bad in (dict(lo=1, hi=0), dict(bogus=1), dict(sum_lo=2, sum_hi=1)):
    try:
        hnn.Attention(d, n, softmax=bad)
        raise AssertionError("expected an error")
    except ValueError:
        pass

# hnn.Attention with heads > 1 is torch's multi-head split: same result as
# running each head's slice of q/k/v through a single-head module.
mha = hnn.Attention(d, n, heads=2, score=[[0.1, 0.1, 0.03], [0.1, 0.05, 0.02]]).double()
x = torch.rand(2, n, d, dtype=torch.float64)
with torch.no_grad():
    cat = torch.zeros(2, n, d, dtype=torch.float64)
    for hd in range(2):
        sl = slice(hd * d // 2, (hd + 1) * d // 2)
        one = hnn.Attention(d, n, score=mha.score[hd].tolist()).double()
        for f, g in ((one.q, mha.q), (one.k, mha.k), (one.v, mha.v)):
            f.weight.zero_(); f.bias.zero_()
            f.weight[sl] = g.weight[sl]; f.bias[sl] = g.bias[sl]
        one.o.weight.zero_(); one.o.bias.zero_()
        one.o.weight[:, sl] = mha.o.weight[:, sl]
        cat += one(x) - one.o.bias
    cat += mha.o.bias
    assert torch.allclose(mha(x), cat, atol=1e-12), "multi-head attention is the per-head split"
for bad in ({"heads": 3}, {"heads": 2, "score": [[1, 2, 3]]}):
    try:
        hnn.Attention(d, n, **bad)
        raise AssertionError("expected an error")
    except ValueError:
        pass

# Errors name the offending module.
for bad, msg in ((nn.Sequential(nn.Linear(4, 4), nn.ReLU()), "Chebyshev"),
                 (nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4)), "fheml.nn.LayerNorm"),
                 (nn.Sequential(nn.Linear(4, 4), nn.Softmax(-1)), "softmax="),
                 (nn.Sequential(nn.Conv2d(1, 1, 3), nn.MaxPool2d(2)), "fheml.nn.MaxPool2d"),
                 (nn.Sequential(nn.Conv2d(1, 1, 3), nn.Linear(6, 2)), "Flatten")):
    try:
        fheml.from_torch(bad, (4,) if isinstance(bad[0], nn.Linear) else (1, 8, 8))
        raise AssertionError("expected an error")
    except ValueError as e:
        assert msg in str(e), e

# A short training run: the exported model reproduces the trained one.
x = torch.rand(256, 1, 8, 8, dtype=torch.float64)
y = (x[:, 0, :4].mean((1, 2)) > x[:, 0, 4:].mean((1, 2))).long()
model = nn.Sequential(nn.Conv2d(1, 3, 3, stride=2), hnn.Poly([0, 0.5, 0.25]), nn.Flatten(), nn.Linear(27, 2)).double()
opt = torch.optim.Adam(model.parameters(), 0.05)
for _ in range(100):
    opt.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
acc = (model(x).argmax(1) == y).double().mean().item()
print(f"trained toy model: loss {loss.item():.3f}, accuracy {acc:.2f}")
run("trained", model, (1, 8, 8))
print("PASSED")
