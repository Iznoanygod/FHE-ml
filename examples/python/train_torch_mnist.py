"""Train an HE-friendly MNIST model in PyTorch, export it to fheml and run it
encrypted.

    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch cnn [--n 3] [--insecure]
    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch vit
    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch deepcnn --bootstrap

The model is ordinary torch code: torch.nn layers plus the polynomial
activation / attention modules in fheml.nn, which compute exactly what the
encrypted layers compute. MNIST is downloaded into data/.

deepcnn is a four-block CNN with Chebyshev ReLUs whose 25 levels exceed what
a 128-bit ring 2^16 holds. --bootstrap folds the activation intervals into
the convolutions (one level each) and lets fheml.auto_bootstrap() insert
bootstraps, scaled from the training images, so it runs at ring 2^16.
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


def deepcnn():
    # Deeper than ring 2^16 allows without bootstrapping: 4 x (linear + relu
    # approximation on [-R, R], degree 15: 1 + 5 levels) + head = 25 levels
    # once the intervals are folded. The ReLU intervals must cover the
    # pre-activations (a Chebyshev series diverges outside them);
    # check_ranges() reports how well they do. The client pools the image to
    # 14x14 first, so every activation fits in 256 slots: at ring 2^16 the
    # bootstrapping key set grows with the slot count (45-69 keys at 256).
    relu = lambda r: hnn.Chebyshev("relu", -r, r, 15)  # noqa: E731
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, stride=2, padding=1), nn.BatchNorm2d(4), relu(8),
        nn.Conv2d(4, 4, 3, stride=1, padding=1), nn.BatchNorm2d(4), relu(8),
        nn.Conv2d(4, 4, 3, stride=1, padding=1), nn.BatchNorm2d(4), relu(8),
        nn.Flatten(), nn.Linear(196, 32), nn.BatchNorm1d(32), relu(8),
        nn.Linear(32, 10))


def pool2(x):
    """(B, 1, 28, 28) -> (B, 1, 14, 14): 2x2 average pooling, done by the client."""
    return x.reshape(-1, 1, 14, 2, 14, 2).mean((3, 5))


def check_ranges(net, xs):
    """Largest |input| of every chebyshev layer over xs, against its interval."""
    xs = [np.asarray(x, dtype=np.float64).ravel() for x in xs]
    for i, layer in enumerate(net.layers):
        if layer.type == "chebyshev":
            peak = max(np.abs(x).max() for x in xs)
            print(f"  layer {i} {layer.describe()}: max |input| {peak:.2f} over {len(xs)} images")
        xs = [np.asarray(layer.forward_plain(x)) for x in xs]


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
    ap.add_argument("--arch", choices=("cnn", "vit", "deepcnn"), default="cnn")
    ap.add_argument("--bootstrap", action="store_true",
                    help="fold activation intervals and insert bootstraps automatically (ring 2^16)")
    ap.add_argument("--window", type=int, default=0, help="rotation-key ladder window (0 = one key per rotation)")
    ap.add_argument("--calibration", type=int, default=64, help="training images for bootstrap range scaling")
    ap.add_argument("--ring", type=int, help="ring dimension override (insecure runs; bootstrapping precision grows with it)")
    ap.add_argument("--digits", type=int, default=0, help="HYBRID key-switching digits (0 = OpenFHE default; 4 fits "
                    "two more levels in ring 2^16 for 25%% larger keys)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, help="default: 3 (cnn), 15 (vit)")
    ap.add_argument("--out", help="also save the exported model here (.fheml)")
    ap.add_argument("--n", type=int, default=3, help="images to classify encrypted")
    ap.add_argument("--insecure", action="store_true", help="toy parameters (fast, NOT secure)")
    args = ap.parse_args()

    torch.manual_seed(0)
    data = load_mnist(args.data)
    prep = {"cnn": lambda x: x, "vit": patchify, "deepcnn": pool2}[args.arch]
    xtr, ytr = torch.tensor(prep(data["train_x"]), dtype=torch.float32), torch.tensor(data["train_y"]).long()
    xte, yte = torch.tensor(prep(data["test_x"]), dtype=torch.float32), torch.tensor(data["test_y"]).long()

    model = {"cnn": cnn, "vit": vit, "deepcnn": deepcnn}[args.arch]()
    epochs = args.epochs or (3 if args.arch == "cnn" else 15 if args.arch == "vit" else 4)
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

    shape = {"cnn": (1, 28, 28), "vit": (N_TOK, PATCH), "deepcnn": (1, 14, 14)}[args.arch]
    net = fheml.from_torch(model, shape)  # checks fheml plaintext == torch
    if args.out:
        net.save(args.out)
        print("wrote", args.out)
    if args.arch == "vit":
        net.fuse_linear()  # token mean + head -> one matrix, saves a level
    level_budget = [3, 3]
    if args.arch == "deepcnn":
        net.set_gapped_layouts(False)  # compact activations: fewer slots, so fewer bootstrapping keys
    if args.bootstrap:
        level_budget = [2, 2]  # bootstrap depth 18 instead of 20: 5 levels per stretch at ring 2^16 (6 with --digits 4)
        calib = xtr[:args.calibration].double().numpy()
        net.compile()
        check_ranges(net, calib)
        depth = net.depth
        boots, budget = fheml.auto_bootstrap(net, calib, level_budget=level_budget, max_ring=1 << 16,
                                             key_switch_digits=args.digits)
        print(f"auto_bootstrap: depth {depth} -> {net.depth}, {boots} bootstraps, at most {budget} levels between them")
    net.compile()
    print(net.summary())

    t = time.time()
    overrides = {"level_budget": level_budget, "key_switch_digits": args.digits}
    if args.ring or (args.insecure and net.uses_bootstrapping):
        overrides["ring_dim"] = args.ring or 4096
    ctx = fheml.make_context(net, secure=not args.insecure, rotation_window=args.window, **overrides)
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
