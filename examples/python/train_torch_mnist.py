"""Train an HE-friendly MNIST model in PyTorch, export it to fheml and run it
encrypted.

    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch cnn [--n 3] [--insecure]
    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch vit

The model is ordinary torch code: torch.nn layers plus the polynomial
activation / attention modules in fheml.nn, which compute exactly what the
encrypted layers compute. MNIST is downloaded into data/.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch import nn

import fheml
from fheml import nn as hnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
from train_mnist import load_mnist  # noqa: E402

N_TOK, PATCH, D, HID = 16, 49, 32, 64


def cnn():
    # CryptoNets/LoLa shape. from_torch folds BatchNorm into the convolution,
    # so it costs nothing under encryption.
    return nn.Sequential(
        nn.Conv2d(1, 5, 5, stride=2, padding=1), nn.BatchNorm2d(5), hnn.Poly([0, 0.5, 0.25]),
        nn.Flatten(), nn.Linear(845, 64), hnn.Poly([0, 0.5, 0.25]),
        nn.Linear(64, 10))


def vit():
    # Tokens are 7x7 patches; softmax -> quadratic score, no LayerNorm.
    return nn.Sequential(
        nn.Linear(PATCH, D), hnn.PosEmbedding(N_TOK, D),
        hnn.Residual(hnn.Attention(D, N_TOK)),
        hnn.Residual(nn.Linear(D, HID), hnn.Poly([0, 0.5, 0.25]), nn.Linear(HID, D)),
        hnn.TokenMean(), nn.Linear(D, 10))


def patchify(x):
    """(B, 1, 28, 28) -> (B, 16, 49): row-major 4x4 grid of 7x7 patches."""
    return x.reshape(-1, 4, 7, 4, 7).transpose(0, 1, 3, 2, 4).reshape(-1, N_TOK, PATCH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("cnn", "vit"), default="cnn")
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, help="default: 3 (cnn), 15 (vit)")
    ap.add_argument("--out", help="also save the exported model here (.fheml)")
    ap.add_argument("--n", type=int, default=3, help="images to classify encrypted")
    ap.add_argument("--insecure", action="store_true", help="toy parameters (fast, NOT secure)")
    args = ap.parse_args()

    torch.manual_seed(0)
    data = load_mnist(args.data)
    prep = (lambda x: x) if args.arch == "cnn" else patchify
    xtr, ytr = torch.tensor(prep(data["train_x"]), dtype=torch.float32), torch.tensor(data["train_y"]).long()
    xte, yte = torch.tensor(prep(data["test_x"]), dtype=torch.float32), torch.tensor(data["test_y"]).long()

    model = cnn() if args.arch == "cnn" else vit()
    epochs = args.epochs or (3 if args.arch == "cnn" else 15)
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * ((len(xtr) + 127) // 128))
    for epoch in range(epochs):
        model.train()
        t = time.time()
        for idx in torch.randperm(len(xtr)).split(128):
            opt.zero_grad()
            nn.functional.cross_entropy(model(xtr[idx]), ytr[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # polynomial nets can spike early
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(xte).argmax(1) == yte).float().mean().item()
        print(f"epoch {epoch + 1}: test accuracy {acc:.1%} ({time.time() - t:.0f}s)")

    shape = (1, 28, 28) if args.arch == "cnn" else (N_TOK, PATCH)
    net = fheml.from_torch(model, shape)  # checks fheml plaintext == torch
    if args.out:
        net.save(args.out)
        print("wrote", args.out)
    if args.arch == "vit":
        net.fuse_linear()  # token mean + head -> one matrix, saves a level
    net.compile()
    print(net.summary())

    t = time.time()
    ctx = fheml.make_context(net, secure=not args.insecure)
    print(ctx.describe(), f"(setup {time.time() - t:.1f}s)")
    model.double()
    x = xte.double().numpy()
    for i in range(args.n):
        t = time.time()
        enc = fheml.infer(net, ctx, x[i])
        dt = time.time() - t
        with torch.no_grad():
            ref = model(xte[i:i + 1].double()).numpy().ravel()
        print(f"image {i}: label {int(yte[i])}, encrypted -> {enc.argmax()}, torch -> {ref.argmax()}, "
              f"max |enc - torch| {np.abs(enc - ref).max():.1e}, {dt:.1f}s")


if __name__ == "__main__":
    main()
