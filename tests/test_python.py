"""Python binding smoke test: builds a tiny transformer from numpy arrays and
checks encrypted inference against the plaintext reference (toy parameters).

    PYTHONPATH=build/python python3 tests/test_python.py
"""
import numpy as np

import fheml

rng = np.random.default_rng(0)
n, d, h = 4, 8, 16


def w(o, i, s=0.3):
    return rng.uniform(-s, s, (o, i))  # 2-D arrays are flattened by fheml


net = fheml.Network(fheml.Shape(n, 1, d))
net.add(fheml.Residual([fheml.Attention(n, d, w(d, d), w(d, d), w(d, d), w(d, d), score=[0.1, 0.1, 0.03])]))
# Two heads of width 4 with a score polynomial per head.
net.add(fheml.Residual([fheml.Attention(n, d, w(d, d), w(d, d), w(d, d), w(d, d),
                                        score=[[0.1, 0.1, 0.03], [0.1, 0.05, 0.02]], heads=2)]))
net.add(fheml.Residual([fheml.TokenDense(n, d, h, w(h, d), w(1, h, 0.1)), fheml.Poly([0, 0.5, 0.25]),
                        fheml.TokenDense(n, h, d, w(d, h), w(1, d, 0.1))]))
net.add(fheml.TokenMean(n, d)).add(fheml.Dense(d, 3, w(3, d), w(1, 3, 0.1)))
net.compile()
print(net.summary())
assert "heads=2" in net.layers[1].describe()

ctx = fheml.make_context(net, secure=False)
x = rng.uniform(-1, 1, (n, d))
enc, ref = fheml.infer(net, ctx, x), fheml.plain(net, x)
err = np.abs(enc - ref).max()
print("encrypted", enc, "\nplaintext", ref, f"\nmax err {err:.1e}")
assert err < 1e-3

net.encrypt_weights(ctx)
err = np.abs(fheml.infer(net, ctx, x) - ref).max()
print(f"encrypted weights: max err {err:.1e}")
assert err < 1e-3

# Multiplexed packing: the stride-2 conv interleaves 8 channels into the gaps.
cnn = fheml.Network(fheml.Shape(2, 8, 8))
cnn.packing = "multiplexed"
cnn.add(fheml.Conv2d(2, 8, 3, 3, 2, 1, rng.uniform(-0.3, 0.3, 144), rng.uniform(-0.1, 0.1, 8)))
cnn.add(fheml.Square()).add(fheml.Dense(128, 3, w(3, 128, 0.1), w(1, 3, 0.1)))
cnn.compile()
print(cnn.summary())
assert cnn.packing == "multiplexed" and "multiplexed out m=2" in cnn.layers[0].describe() and cnn.slots == 128
ctx = fheml.make_context(cnn, secure=False)
x = rng.uniform(-1, 1, (2, 8, 8))
err = np.abs(fheml.infer(cnn, ctx, x) - fheml.plain(cnn, x)).max()
print(f"multiplexed cnn: max err {err:.1e}")
assert err < 1e-3
# Select: token 2's first 5 features into token 3 (picking one token out of a
# sequence and writing it into another, as the encrypted G2P decoder does).
sel = fheml.Network(fheml.Shape(n, 1, d))
sel.add(fheml.Select(fheml.Shape(n, 1, d), fheml.Shape(1, 1, 5), [2 * d + j for j in range(5)]))
put = [-1] * (n * d)
for j in range(5):
    put[3 * d + j] = j
sel.add(fheml.Select(fheml.Shape(1, 1, 5), fheml.Shape(n, 1, d), put))
sel.compile()
xs = rng.normal(size=(n, d))
want = np.zeros((n, d))
want[3, :5] = xs[2, :5]
assert np.allclose(fheml.plain(sel, xs).reshape(n, d), want)
sctx = fheml.make_context(sel, secure=False)
err = np.abs(fheml.infer(sel, sctx, xs).reshape(n, d) - want).max()
assert err < 1e-6, err
print(f"select: max err {err:.1e}")
print("PASSED")
