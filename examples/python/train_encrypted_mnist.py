"""Train a small MNIST network entirely under encryption.

The client encrypts each (downsampled) image and its one-hot label; the
server runs forward, backward and SGD updates on ciphertexts, with encrypted
weights that are bootstrapped after every update. The same updates run in the
clear alongside (fheml.train_plain) so the two can be compared; at the end
the key holder decrypts the trained weights.

    PYTHONPATH=build/python python3 examples/python/train_encrypted_mnist.py --insecure [--updates 20]
    ... --fine-tune    # pre-train in the clear, then train only the last layer encrypted
    ... --slots 256    # 128-bit security: ring 2^16, rotation-key ladder, small bootstraps

This is slow (every weight diagonal is a ciphertext and is bootstrapped after
every update); it demonstrates that the whole pipeline works. At 128-bit
security the rotations are composed from a ladder of 2 log2(slots) keys
instead of one key per weight diagonal (about 100 MB each at ring 2^16).
"""
import argparse
import os
import sys
import time

import numpy as np

import fheml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
from train_mnist import load_mnist  # noqa: E402

HIDDEN = 16
POLY = [0.0, 0.5, 0.25]


def downsample(x):
    """(B, 1, 28, 28) -> (B, 49): 4x4 average pooling, done by the client."""
    return x.reshape(-1, 7, 4, 7, 4).mean((2, 4)).reshape(-1, 49)


def preprocess(x, mean, std):
    """Client-side: downsample and centre (public statistics), values in about [-1.4, 1.4]."""
    return (downsample(x) - mean) / (3 * std)


def build(rng):
    net = fheml.Network(fheml.Shape(49))
    net.add(fheml.Dense(49, HIDDEN, rng.normal(0, 1 / 7, (HIDDEN, 49)), np.zeros(HIDDEN)))
    net.add(fheml.Poly(POLY))
    net.add(fheml.Dense(HIDDEN, 10, rng.normal(0, 0.5 / 4, (10, HIDDEN)), np.zeros(10)))
    net.compile()
    return net


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def accuracy(net, x, y):
    return np.mean([fheml.plain(net, xi).argmax() == yi for xi, yi in zip(x, y)])


def weights(net):
    return np.concatenate([l.params() for l in net.layers if isinstance(l, fheml.LinearLayer)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--fine-tune", action="store_true", help="pre-train in the clear, train only the last layer")
    ap.add_argument("--insecure", action="store_true", help="toy parameters (fast, NOT secure)")
    ap.add_argument("--test", type=int, default=1000, help="test images for the accuracy check")
    ap.add_argument("--slots", type=int, help="CKKS slots (default: all of the ring; fewer = fewer bootstrapping keys)")
    ap.add_argument("--window", type=int, help="rotation-key ladder window (default: 1 when secure, 0 = key per rotation)")
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="size the context for --updates updates instead of bootstrapping the weights")
    args = ap.parse_args()

    data = load_mnist(args.data)
    small = downsample(data["train_x"])
    mean, std = small.mean(0), small.std()
    xtr, ytr = preprocess(data["train_x"], mean, std), data["train_y"]
    xte, yte = preprocess(data["test_x"][: args.test], mean, std), data["test_y"][: args.test]
    onehot = np.eye(10)

    # Two identical networks: one trained encrypted, one in the clear.
    enc_net, ref_net = build(np.random.default_rng(0)), build(np.random.default_rng(0))
    frozen = []
    if args.fine_tune:
        rng = np.random.default_rng(1)
        for _ in range(300):  # quick pre-training in the clear
            idx = rng.integers(0, len(xtr), 32)
            for n in (enc_net, ref_net):
                fheml.train_plain(n, xtr[idx], onehot[ytr[idx]], 1.0)
        frozen = [0, 1]
        for n in (enc_net, ref_net):
            n.compile()
    print(f"initial test accuracy {accuracy(ref_net, xte, yte):.1%}")

    t = time.time()
    trainer, ctx = fheml.make_trainer(enc_net, frozen, secure=not args.insecure, slots=args.slots,
                                      rotation_window=args.window, bootstrapping=not args.no_bootstrap,
                                      updates=args.updates, full_slots=not args.no_bootstrap)
    print(trainer.summary())
    print(ctx.describe(), f"(keys {time.time() - t:.0f}s, {len(trainer.rotation_indices())} rotations in the step, "
          f"process RSS {rss_mb():.0f} MB)")
    t = time.time()
    trainer.encrypt_weights(ctx)
    print(f"{trainer.weight_ciphertexts} weight ciphertexts encrypted in {time.time() - t:.1f}s")

    rng = np.random.default_rng(2)
    for u in range(args.updates):
        idx = rng.integers(0, len(xtr), args.batch)
        t = time.time()
        out = fheml.train_step(trainer, ctx, xtr[idx], onehot[ytr[idx]], args.lr)
        dt = time.time() - t
        fheml.train_plain(ref_net, xtr[idx], onehot[ytr[idx]], args.lr, frozen)
        loss = 0.5 * np.mean(np.sum((out - onehot[ytr[idx]]) ** 2, axis=1))
        print(f"update {u + 1}: batch loss {loss:.3f}, {dt:.0f}s ({dt / args.batch:.1f}s per sample incl. the update; "
              f"RSS {rss_mb():.0f} MB)", flush=True)

    trainer.decrypt_weights(ctx)  # the key holder recovers the trained model
    diff = np.abs(weights(enc_net) - weights(ref_net)).max()
    print(f"max |encrypted-trained - plaintext-trained weight| = {diff:.1e} (largest weight {np.abs(weights(ref_net)).max():.2f})")
    print(f"test accuracy: encrypted-trained {accuracy(enc_net, xte, yte):.1%}, "
          f"plaintext-trained {accuracy(ref_net, xte, yte):.1%}")


if __name__ == "__main__":
    main()
