"""Encrypted MNIST inference from Python with the CNN in models/.

    PYTHONPATH=build/python python3 examples/python/mnist_cnn.py [--n 3] [--insecure]
"""
import argparse
import time

import numpy as np

import fheml


def load_samples(path):
    with open(path) as f:
        n, dim = map(int, f.readline().split())
        rows = [np.array(line.split(), dtype=np.float64) for line in f]
    return np.array([int(r[0]) for r in rows]), np.stack([r[1:] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/mnist_cnn.fheml")
    ap.add_argument("--samples", default="models/mnist_test_samples.txt")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--insecure", action="store_true", help="toy parameters (fast, NOT secure)")
    args = ap.parse_args()

    labels, images = load_samples(args.samples)
    net = fheml.Network.load(args.model)
    net.compile()
    print(net.summary())

    t = time.time()
    ctx = fheml.make_context(net, secure=not args.insecure)
    print(ctx.describe(), f"(setup {time.time() - t:.1f}s)")

    for i in range(args.n):
        t = time.time()
        enc = fheml.infer(net, ctx, images[i])
        ref = fheml.plain(net, images[i])
        print(f"sample {i}: label {labels[i]}  encrypted {enc.argmax()}  plaintext {ref.argmax()}  "
              f"max err {np.abs(enc - ref).max():.1e}  {time.time() - t:.2f}s")


if __name__ == "__main__":
    main()
