"""Encrypted inference with the tiny MNIST vision transformer in models/.

The client cuts the image into 16 patches of 7x7 (in the clear, before
encrypting); the server runs embedding, attention, MLP and classifier head
homomorphically.

    PYTHONPATH=build/python python3 examples/python/mnist_transformer.py [--n 3] [--insecure]
"""
import argparse
import time

import numpy as np

import fheml


def load_samples(path):
    with open(path) as f:
        f.readline()
        rows = [np.array(line.split(), dtype=np.float64) for line in f]
    return np.array([int(r[0]) for r in rows]), np.stack([r[1:] for r in rows])


def patchify(img, width):
    """16 patches of 7x7, each zero-padded to the model's token width (64)."""
    p = img.reshape(4, 7, 4, 7).transpose(0, 2, 1, 3).reshape(16, 49)
    return np.pad(p, ((0, 0), (0, width - 49)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/mnist_transformer.fheml")
    ap.add_argument("--samples", default="models/mnist_test_samples.txt")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--encrypt-weights", action="store_true")
    ap.add_argument("--insecure", action="store_true", help="toy parameters (fast, NOT secure)")
    args = ap.parse_args()

    labels, images = load_samples(args.samples)
    net = fheml.Network.load(args.model)
    net.fuse_linear()  # token mean + classifier -> one matrix, saves a level
    net.compile()
    print(net.summary())
    width = net.input_shape.w
    acc = np.mean([fheml.plain(net, patchify(im, width)).argmax() == lab for im, lab in zip(images, labels)])
    print(f"plaintext accuracy on {len(labels)} samples: {acc:.1%}")

    t = time.time()
    ctx = fheml.make_context(net, secure=not args.insecure)
    print(ctx.describe(), f"(setup {time.time() - t:.1f}s)")
    if args.encrypt_weights:
        t = time.time()
        net.encrypt_weights(ctx)
        print(f"weights encrypted in {time.time() - t:.1f}s")

    for i in range(args.n):
        x = patchify(images[i], width)
        t = time.time()
        enc = fheml.infer(net, ctx, x)
        dt = time.time() - t
        ref = fheml.plain(net, x)
        print(f"sample {i}: label {labels[i]}  encrypted {enc.argmax()}  plaintext {ref.argmax()}  "
              f"max err {np.abs(enc - ref).max():.1e}  {dt:.2f}s")


if __name__ == "__main__":
    main()
