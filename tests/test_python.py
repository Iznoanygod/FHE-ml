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
net.add(fheml.Residual([fheml.TokenDense(n, d, h, w(h, d), w(1, h, 0.1)), fheml.Poly([0, 0.5, 0.25]),
                        fheml.TokenDense(n, h, d, w(d, h), w(1, d, 0.1))]))
net.add(fheml.TokenMean(n, d)).add(fheml.Dense(d, 3, w(3, d), w(1, 3, 0.1)))
net.compile()
print(net.summary())

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
print("PASSED")
