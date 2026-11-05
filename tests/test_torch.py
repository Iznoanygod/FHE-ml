"""PyTorch exporter test: exports small torch models and checks the fheml
plaintext reference and encrypted inference against torch (toy parameters).

    PYTHONPATH=build/python python3 tests/test_torch.py
"""
import os
import tempfile

import numpy as np
import torch
from torch import nn

import fheml
from fheml import nn as hnn

torch.manual_seed(0)
rng = np.random.default_rng(0)


def run(name, model, shape, tol=1e-3, **export):
    model = model.double().eval()
    with torch.no_grad():
        for m in model.modules():  # non-trivial batch-norm statistics
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.running_mean.uniform_(-0.2, 0.2)
                m.running_var.uniform_(0.5, 1.5)
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.2, 0.2)
    net = fheml.from_torch(model, shape, **export)  # also checks plaintext vs torch
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

# Encrypted argmax (scheme switching): a one-hot of the logits. 6 classes
# pad to 8 blocks (the extra ones hold -range and never win).
clf = nn.Sequential(nn.Linear(5, 6), hnn.ArgMax(range=4.0))
net = run("argmax", clf, (5,))
assert net.layers[-1].type == "argmax"
# Per token and per group of 3 features (the pixel classes of a patch token),
# in the token-strided layout of the previous layer.
seg = nn.Sequential(nn.Linear(4, 8), hnn.PosEmbedding(3, 8), hnn.Residual(hnn.Attention(8, 3)), nn.Linear(8, 6),
                    hnn.ArgMax(range=4.0, classes=3))
net = run("token argmax", seg, (3, 4))
path = os.path.join(tempfile.mkdtemp(), "argmax.fheml")
net.save(path)
x = rng.uniform(-1, 1, (3, 4))
assert np.array_equal(fheml.plain(fheml.Network.load(path), x), fheml.plain(net, x)), "argmax save/load"

# Comparisons on fewer switched slots than the ciphertext holds
# (ContextOptions.compare_slots): one fold for 8 slots, chunks of 8 for 32.
o = fheml.ContextOptions()
o.mult_depth, o.slots, o.secure, o.scheme_switching, o.compare_slots = 16, 64, False, True, 8
cctx = fheml.Context(o)
cctx.keygen([8, 16, 24, 32, 40, 48, 56, -8 % 64, -16 % 64, -24 % 64])
a, b = rng.uniform(-1, 1, 64), rng.uniform(-1, 1, 64)
for count in (8, 32):
    r = np.array(cctx.decrypt(cctx.compare(cctx.encrypt(list(a)), cctx.encrypt(list(b)), 1.0, count), 64))
    assert np.abs(r[:count] - (a < b)[:count]).max() < 1e-3, count
    if count > 8:
        assert np.abs(r[count:]).max() < 1e-3, "chunked comparison leaves zeros beyond count"
print("--- compare_slots: folded and chunked comparisons OK")

# Per pixel over the channels of an image (CNN segmentation logits), on the
# gapped layout a strided convolution leaves.
pix = nn.Sequential(nn.Conv2d(1, 3, 3, stride=2, padding=1), hnn.ArgMax(range=4.0, dim=1))
run("pixel argmax", pix, (1, 4, 4))

# Plain nn.ReLU exports as the exact scheme-switching ReLU: the encrypted
# result matches torch's true ReLU, not a polynomial.
relu_mlp = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 3))
net = run("exact relu", relu_mlp, (6,), exact_relu=4.0)
assert [l.type for l in net.layers] == ["dense", "exactrelu", "dense"]
try:
    fheml.from_torch(relu_mlp, (6,))
    raise AssertionError("nn.ReLU without exact_relu must be rejected")
except ValueError as e:
    assert "exact_relu" in str(e)

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

# Encoder-decoder over one sequence of 3 source and 2 target tokens:
# segment-wise embedding and head, encoder self-attention, causal decoder
# self-attention with softmax, cross-attention, segment-wise layer norm.
S, T, d = 3, 2, 8
seg = (S, T)
masks = hnn.encoder_decoder_masks(S, T)
sm = dict(lo=-3.0, hi=3.0, exp_degree=15, inv_degree=15, newton=2, sum_lo=0.05, sum_hi=45.0)
encdec = nn.Sequential(
    hnn.SegmentLinear(6, d, seg), hnn.PosEmbedding(S + T, d),
    hnn.Residual(hnn.Attention(d, S + T, heads=2, mask=masks["encoder"], segments=seg, query_segments=[0],
                               key_segments=[0])),
    hnn.Residual(hnn.SegmentLinear(d, 16, seg, active=[0]), hnn.Poly([0, 0.5, 0.25]),
                 hnn.SegmentLinear(16, d, seg, active=[0])),
    hnn.Residual(hnn.LayerNorm(d, eps=0.1, var_max=4.0, degree=31, segments=seg),
                 hnn.Attention(d, S + T, heads=2, softmax=sm, mask=masks["causal"], segments=seg,
                               query_segments=[1], key_segments=[1])),
    hnn.Residual(hnn.Attention(d, S + T, mask=masks["cross"], segments=seg, query_segments=[1], key_segments=[0])),
    hnn.SegmentLinear(d, 4, seg, active=[1]))
with torch.no_grad():
    for p in encdec.parameters():
        p.mul_(0.5)
net = run("encoder-decoder", encdec, (S + T, 6))
xs = torch.rand(1, S + T, 6, dtype=torch.float64)
with torch.no_grad():
    y0 = encdec(xs)
    xs2 = xs.clone()
    xs2[0, -1] += 1
    y1 = encdec(xs2)
assert torch.allclose(y0[0, :-1], y1[0, :-1]) and not torch.allclose(y0[0, -1], y1[0, -1]), "causality"
assert torch.all(y0[0, :S] == 0), "head acts on the target segment only"

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

# Chebyshev activations: training clamps to [a, b] and the range loss
# penalises inputs outside it; eval mode evaluates the bare series.
act = hnn.Chebyshev("sigmoid", -2, 2, 7)
xs = torch.tensor([-3.0, 0.0, 1.0, 4.0], requires_grad=True)
model = nn.Sequential(act).train()
y_train = model(xs)
penalty = hnn.range_loss(model)
assert abs(penalty.item() - (1 + 4) / 4) < 1e-6, penalty
assert torch.allclose(y_train[0], act.eval()(torch.tensor(-2.0))), "clamped while training"
penalty.backward()
assert xs.grad[0] < 0 and xs.grad[3] > 0 and xs.grad[1] == 0, "penalty pulls the inputs back inside"
assert not torch.allclose(act(torch.tensor(4.0)), act(torch.tensor(2.0))), "no clamp in eval mode"
print("PASSED")
