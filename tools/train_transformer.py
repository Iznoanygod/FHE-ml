#!/usr/bin/env python3
"""Train a tiny HE-friendly vision transformer on MNIST with numpy and export
it for fheml.

Each 28x28 image is cut into 16 patches of 7x7 (16 tokens of 49 pixels,
zero-padded to 64 in the exported model's input).

    tokendense 49 (64) -> 32 (+ learned positional embedding)
    residual { attention, quadratic scores A = c0 + c1 S + c2 S^2 }
    residual { tokendense 32 -> 64, poly 0.5x + 0.25x^2, tokendense 64 -> 32 }
    tokenmean, dense 32 -> 10

There is no softmax or LayerNorm: attention weights come from a fixed
non-negative quadratic of the raw scores, normalised by the number of tokens
(polynomial attention, cf. Zimerman et al. ICML 2024, PowerSoftmax). These
replacements keep the whole model at multiplicative depth 13 (12 after fusing
the head), with no bootstrapping.

    python3 tools/train_transformer.py --out models/mnist_transformer.fheml
"""
import argparse
import os

import numpy as np

from train_mnist import fmt, load_mnist

N_TOK, PATCH, D, HID = 16, 49, 32, 64
# Exported input tokens are zero-padded to the widest layer so every layer
# keeps the same token stride in the ciphertext (far fewer diagonals).
TOKEN_STRIDE = 64
POLY = (0.0, 0.5, 0.25)
# A = (S + 2)^2 / (8 n) = c0 + c1 S + c2 S^2: non-negative, ~1/n at S = 0.
SCORE = (4.0 / (8 * N_TOK), 4.0 / (8 * N_TOK), 1.0 / (8 * N_TOK))


def patchify(x):
    """(B, 1, 28, 28) -> (B, 16, 49): row-major 4x4 grid of 7x7 patches."""
    b = len(x)
    return x.reshape(b, 4, 7, 4, 7).transpose(0, 1, 3, 2, 4).reshape(b, N_TOK, PATCH)


def poly(x, c):
    return c[0] + c[1] * x + c[2] * x * x


def dpoly(x, c):
    return c[1] + 2 * c[2] * x


class Model:
    def __init__(self, rng):
        def lin(o, i, gain=1.0):
            return rng.normal(0, gain / np.sqrt(i), (o, i))

        self.p = {
            "E": lin(D, PATCH), "P": np.zeros((N_TOK, D)),
            "Wq": lin(D, D, 0.5), "bq": np.zeros(D), "Wk": lin(D, D, 0.5), "bk": np.zeros(D),
            "Wv": lin(D, D), "bv": np.zeros(D), "Wo": lin(D, D, 0.5), "bo": np.zeros(D),
            "W1": lin(HID, D), "b1": np.zeros(HID), "W2": lin(D, HID, 0.5), "b2": np.zeros(D),
            "Wh": lin(10, D), "bh": np.zeros(10),
        }

    def forward(self, x0):
        p = self.p
        h = x0 @ p["E"].T + p["P"]
        q, k, v = h @ p["Wq"].T + p["bq"], h @ p["Wk"].T + p["bk"], h @ p["Wv"].T + p["bv"]
        s = np.einsum("bid,bjd->bij", q, k)
        a = poly(s, SCORE)
        o = np.einsum("bij,bjd->bid", a, v)
        h1 = h + o @ p["Wo"].T + p["bo"]
        u = h1 @ p["W1"].T + p["b1"]
        g = poly(u, POLY)
        h2 = h1 + g @ p["W2"].T + p["b2"]
        pooled = h2.mean(1)
        self.cache = (x0, h, q, k, v, s, a, o, h1, u, g, pooled)
        return pooled @ p["Wh"].T + p["bh"]

    def backward(self, dlog):
        p = self.p
        x0, h, q, k, v, s, a, o, h1, u, g, pooled = self.cache
        gr = {}
        gr["Wh"], gr["bh"] = dlog.T @ pooled, dlog.sum(0)
        dh2 = np.repeat((dlog @ p["Wh"])[:, None, :] / N_TOK, N_TOK, axis=1)
        gr["W2"], gr["b2"] = np.einsum("bto,bti->oi", dh2, g), dh2.sum((0, 1))
        du = (dh2 @ p["W2"]) * dpoly(u, POLY)
        gr["W1"], gr["b1"] = np.einsum("bto,bti->oi", du, h1), du.sum((0, 1))
        dh1 = dh2 + du @ p["W1"]
        gr["Wo"], gr["bo"] = np.einsum("bto,bti->oi", dh1, o), dh1.sum((0, 1))
        do = dh1 @ p["Wo"]
        da = np.einsum("bid,bjd->bij", do, v)
        dv = np.einsum("bij,bid->bjd", a, do)
        ds = da * dpoly(s, SCORE)
        dq = np.einsum("bij,bjd->bid", ds, k)
        dk = np.einsum("bij,bid->bjd", ds, q)
        dh = dh1.copy()
        for name, dz in (("q", dq), ("k", dk), ("v", dv)):
            gr["W" + name], gr["b" + name] = np.einsum("bto,bti->oi", dz, h), dz.sum((0, 1))
            dh += dz @ p["W" + name]
        gr["E"], gr["P"] = np.einsum("bto,bti->oi", dh, x0), dh.sum(0)
        return gr


def xent(logits, y):
    z = logits - logits.max(1, keepdims=True)
    pr = np.exp(z)
    pr /= pr.sum(1, keepdims=True)
    loss = -np.log(pr[np.arange(len(y)), y] + 1e-12).mean()
    pr[np.arange(len(y)), y] -= 1
    return loss, pr / len(y)


def grad_check(rng):
    m = Model(rng)
    x, y = rng.normal(size=(3, N_TOK, PATCH)) * 0.3, np.array([1, 2, 3])
    _, dl = xent(m.forward(x), y)
    g = m.backward(dl)
    worst = 0.0
    for name in m.p:
        idx = tuple(rng.integers(0, s) for s in m.p[name].shape)
        old = m.p[name][idx]
        m.p[name][idx] = old + 1e-5
        lp, _ = xent(m.forward(x), y)
        m.p[name][idx] = old - 1e-5
        lm, _ = xent(m.forward(x), y)
        m.p[name][idx] = old
        num = (lp - lm) / 2e-5
        worst = max(worst, abs(num - g[name][idx]) / max(1e-6, abs(num) + abs(g[name][idx])))
    print(f"gradient check: worst relative error {worst:.1e}")
    assert worst < 1e-4


def accuracy(m, x, y, bs=1000):
    return np.mean(np.concatenate([m.forward(x[i:i + bs]).argmax(1) == y[i:i + bs] for i in range(0, len(x), bs)]))


def export(m, path):
    p = m.p
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        e = np.zeros((D, TOKEN_STRIDE))
        e[:, :PATCH] = p["E"]
        f.write(f"fheml 1\ninput {N_TOK} 1 {TOKEN_STRIDE}\n")
        f.write(f"tokendense {N_TOK} {TOKEN_STRIDE} {D} {N_TOK * D}\n{fmt(e)}\n{fmt(p['P'])}\n")
        f.write(f"residual 1\nattention {N_TOK} {D} 1\n")
        for w in ("Wq", "Wk", "Wv", "Wo"):
            f.write(fmt(p[w]) + "\n")
        for b in ("bq", "bk", "bv", "bo"):
            f.write(fmt(p[b]) + "\n")
        f.write(fmt(SCORE) + "\n")
        f.write(f"residual 3\ntokendense {N_TOK} {D} {HID} {HID}\n{fmt(p['W1'])}\n{fmt(p['b1'])}\n")
        f.write(f"poly 3\n{fmt(POLY)}\n")
        f.write(f"tokendense {N_TOK} {HID} {D} {D}\n{fmt(p['W2'])}\n{fmt(p['b2'])}\n")
        f.write(f"tokenmean {N_TOK} {D}\ndense {D} 10 1\n{fmt(p['Wh'])}\n{fmt(p['bh'])}\nend\n")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models/mnist_transformer.fheml")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    grad_check(rng)

    data = load_mnist(args.data)
    xtr, ytr = patchify(data["train_x"]), data["train_y"]
    xte, yte = patchify(data["test_x"]), data["test_y"]
    m = Model(rng)
    mom = {k: np.zeros_like(v) for k, v in m.p.items()}
    vel = {k: np.zeros_like(v) for k, v in m.p.items()}
    t, lr = 0, args.lr
    for ep in range(args.epochs):
        perm = rng.permutation(len(xtr))
        tot = 0.0
        for i in range(0, len(xtr), args.batch):
            b = perm[i:i + args.batch]
            loss, dl = xent(m.forward(xtr[b]), ytr[b])
            g = m.backward(dl)
            t += 1
            for k in m.p:
                gk = np.clip(g[k], -1, 1)
                mom[k] = 0.9 * mom[k] + 0.1 * gk
                vel[k] = 0.999 * vel[k] + 0.001 * gk * gk
                m.p[k] -= lr * (mom[k] / (1 - 0.9 ** t)) / (np.sqrt(vel[k] / (1 - 0.999 ** t)) + 1e-8)
            tot += loss * len(b)
        lr *= 0.75
        print(f"epoch {ep + 1}: loss {tot / len(xtr):.4f}  test acc {accuracy(m, xte, yte):.4f}")
    export(m, args.out)


if __name__ == "__main__":
    main()
