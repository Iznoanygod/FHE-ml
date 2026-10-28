#!/usr/bin/env python3
"""Train an HE-friendly CNN on MNIST with numpy and export it for fheml.

The architecture follows CryptoNets/LoLa, with the square activation replaced
by the quadratic f(x) = 0.5 x + 0.25 x^2 (the 2nd-order Taylor expansion of
SiLU around 0), which trains more stably than x^2 and costs 2 CKKS levels:

    conv 1->5, 5x5, stride 2, pad 1   (1x28x28 -> 5x13x13)
    poly 0.5x + 0.25x^2
    dense 845 -> 64
    poly 0.5x + 0.25x^2
    dense 64 -> 10

Usage:
    python3 tools/train_mnist.py --data data --out models/mnist_cnn.fheml \
        --samples models/mnist_test_samples.txt
Only numpy is required. MNIST is downloaded into --data if missing.
"""
import argparse
import gzip
import os
import struct
import urllib.request

import numpy as np

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
POLY = (0.0, 0.5, 0.25)  # c0 + c1 x + c2 x^2


def load_idx(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic == 2051:
            rows, cols = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), np.uint8).reshape(n, rows, cols)
        return np.frombuffer(f.read(), np.uint8)


def load_mnist(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    out = {}
    for key, name in FILES.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            print("downloading", name)
            urllib.request.urlretrieve(MIRROR + name, path)
        out[key] = load_idx(path)
    for k in ("train_x", "test_x"):
        out[k] = out[k].astype(np.float64)[:, None, :, :] / 255.0
    return out


def poly(x):
    return POLY[0] + POLY[1] * x + POLY[2] * x * x


def dpoly(x):
    return POLY[1] + 2 * POLY[2] * x


class Model:
    def __init__(self, rng):
        self.k, self.s, self.p, self.c = 5, 2, 1, 5
        self.oh = (28 + 2 * self.p - self.k) // self.s + 1  # 13
        n_conv = self.c * self.oh * self.oh
        self.params = {
            "cw": rng.normal(0, np.sqrt(2.0 / 25), (self.c, 1, self.k, self.k)),
            "cb": np.zeros(self.c),
            "w1": rng.normal(0, np.sqrt(1.0 / n_conv), (64, n_conv)),
            "b1": np.zeros(64),
            "w2": rng.normal(0, np.sqrt(1.0 / 64), (10, 64)),
            "b2": np.zeros(10),
        }
        # Patch gather indices for im2col on the padded 30x30 input.
        idx = []
        for y in range(self.oh):
            for x in range(self.oh):
                for dy in range(self.k):
                    for dx in range(self.k):
                        idx.append((y * self.s + dy) * (28 + 2 * self.p) + x * self.s + dx)
        self.patch_idx = np.array(idx).reshape(self.oh * self.oh, self.k * self.k)

    def im2col(self, x):
        xp = np.pad(x[:, 0], ((0, 0), (self.p, self.p), (self.p, self.p))).reshape(len(x), -1)
        return xp[:, self.patch_idx]  # (B, 169, 25)

    def forward(self, x, cache=False):
        P = self.params
        cols = self.im2col(x)
        z0 = cols @ P["cw"].reshape(self.c, -1).T + P["cb"]  # (B, 169, 5)
        z0 = z0.transpose(0, 2, 1).reshape(len(x), -1)       # channel-major flatten (C, H, W)
        a0 = poly(z0)
        z1 = a0 @ P["w1"].T + P["b1"]
        a1 = poly(z1)
        z2 = a1 @ P["w2"].T + P["b2"]
        if cache:
            self.cache = (cols, z0, a0, z1, a1)
        return z2

    def backward(self, dz2):
        P, (cols, z0, a0, z1, a1) = self.params, self.cache
        g = {}
        g["w2"] = dz2.T @ a1
        g["b2"] = dz2.sum(0)
        dz1 = (dz2 @ P["w2"]) * dpoly(z1)
        g["w1"] = dz1.T @ a0
        g["b1"] = dz1.sum(0)
        dz0 = (dz1 @ P["w1"]) * dpoly(z0)
        dz0 = dz0.reshape(len(dz0), self.c, -1).transpose(0, 2, 1)  # (B, 169, 5)
        g["cw"] = np.einsum("bpc,bpk->ck", dz0, cols).reshape(P["cw"].shape)
        g["cb"] = dz0.sum((0, 1))
        return g


def softmax_xent(logits, y):
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    p[np.arange(len(y)), y] -= 1
    return loss, p / len(y)


def accuracy(model, x, y, bs=1000):
    correct = 0
    for i in range(0, len(x), bs):
        correct += (model.forward(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum()
    return correct / len(x)


def train(model, data, epochs, bs, lr, rng):
    m = {k: np.zeros_like(v) for k, v in model.params.items()}
    v = {k: np.zeros_like(v) for k, v in model.params.items()}
    t = 0
    x, y = data["train_x"], data["train_y"]
    for ep in range(epochs):
        perm = rng.permutation(len(x))
        tot = 0.0
        for i in range(0, len(x), bs):
            b = perm[i:i + bs]
            loss, dz = softmax_xent(model.forward(x[b], cache=True), y[b])
            g = model.backward(dz)
            t += 1
            for k in model.params:
                gk = np.clip(g[k], -1, 1)
                m[k] = 0.9 * m[k] + 0.1 * gk
                v[k] = 0.999 * v[k] + 0.001 * gk * gk
                mh, vh = m[k] / (1 - 0.9 ** t), v[k] / (1 - 0.999 ** t)
                model.params[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
            tot += loss * len(b)
        lr *= 0.7
        print(f"epoch {ep + 1}: loss {tot / len(x):.4f}  test acc {accuracy(model, data['test_x'], data['test_y']):.4f}")


def fmt(a):
    a = np.asarray(a).ravel()
    return "\n".join(" ".join(repr(float(v)) for v in a[i:i + 16]) for i in range(0, len(a), 16))


def export(model, path):
    P = model.params
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("fheml 1\ninput 1 28 28\n")
        f.write(f"conv2d 1 {model.c} {model.k} {model.k} {model.s} {model.p} 1\n{fmt(P['cw'])}\n{fmt(P['cb'])}\n")
        f.write(f"poly 3\n{fmt(POLY)}\n")
        f.write(f"dense {P['w1'].shape[1]} 64 1\n{fmt(P['w1'])}\n{fmt(P['b1'])}\n")
        f.write(f"poly 3\n{fmt(POLY)}\n")
        f.write(f"dense 64 10 1\n{fmt(P['w2'])}\n{fmt(P['b2'])}\n")
        f.write("end\n")
    print("wrote", path)


def export_samples(data, path, n):
    with open(path, "w") as f:
        f.write(f"{n} 784\n")
        for i in range(n):
            px = data["test_x"][i].ravel()
            f.write(str(int(data["test_y"][i])) + " " + " ".join(f"{v:.6g}" for v in px) + "\n")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models/mnist_cnn.fheml")
    ap.add_argument("--samples", default="models/mnist_test_samples.txt")
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    data = load_mnist(args.data)
    model = Model(rng)
    train(model, data, args.epochs, args.batch, args.lr, rng)
    export(model, args.out)
    if args.samples:
        export_samples(data, args.samples, args.num_samples)


if __name__ == "__main__":
    main()
