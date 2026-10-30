"""Train an HE-friendly MNIST model in PyTorch, export it to fheml and run it
encrypted.

    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch cnn [--n 3] [--insecure]
    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch vit
    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch deepcnn --bootstrap
    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch deepcnn --bootstrap --exact-relu

The model is ordinary torch code: torch.nn layers plus the polynomial
activation / attention modules in fheml.nn, which compute exactly what the
encrypted layers compute. MNIST is downloaded into data/.

deepcnn is a four-block CNN with Chebyshev ReLUs whose 25 levels exceed what
a 128-bit ring 2^16 holds. --bootstrap folds the activation intervals into
the convolutions (one level each) and lets fheml.auto_bootstrap() insert
bootstraps, scaled from the training images, so it runs at ring 2^16.
--exact-relu trains deepcnn with plain nn.ReLU and exports it as
fheml.ExactReLU (sign through CKKS<->FHEW scheme switching): no approximation
error and one level per ReLU past a bootstrap, but one FHEW bootstrap per
slot per ReLU, so encrypted inference is far slower.

poolcnn is a small CNN with exact max pooling (CKKS <-> FHEW scheme
switching; one FHEW bootstrap per compared slot, so it keeps the pooled
feature map in few slots). lnvit / lnvit16 are vision transformers with
pre-norm LayerNorm, multi-head softmax attention and an MLP block, on 4
tokens (14x14 image, 7x7 patches) or 16 tokens (28x28): about 60 levels, so
--bootstrap is needed at 128-bit security; layer norm and softmax attention
then bootstrap internally (see Network.place_bootstraps).

    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch poolcnn
    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch lnvit --bootstrap --window 1 --level-budget 3 --digits 4
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


def deepcnn(exact_relu=False):
    # Deeper than ring 2^16 allows without bootstrapping: 4 x (linear + relu
    # approximation on [-R, R], degree 15: 1 + 5 levels) + head = 25 levels
    # once the intervals are folded. The ReLU intervals must cover the
    # pre-activations (a Chebyshev series diverges outside them);
    # check_ranges() reports how well they do. The client pools the image to
    # 14x14 first, so every activation fits in 256 slots: at ring 2^16 the
    # bootstrapping key set grows with the slot count (45-69 keys at 256).
    relu = (lambda r: nn.ReLU()) if exact_relu else (lambda r: hnn.Chebyshev("relu", -r, r, 15))  # noqa: E731
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, stride=2, padding=1), nn.BatchNorm2d(4), relu(8),
        nn.Conv2d(4, 4, 3, stride=1, padding=1), nn.BatchNorm2d(4), relu(8),
        nn.Conv2d(4, 4, 3, stride=1, padding=1), nn.BatchNorm2d(4), relu(8),
        nn.Flatten(), nn.Linear(196, 32), nn.BatchNorm1d(32), relu(8),
        nn.Linear(32, 10))


def poolcnn(channels=4, stride=2):
    # conv -> exact 2x2 max pool -> square-ish polynomial -> dense head. The
    # pool gathers its four taps into blocks of M slots (M = the pooled size
    # rounded up to a power of two) and compares 2M then M slots, one FHEW
    # bootstrap each: with the default 4 channels at stride 2 (4x7x7 ->
    # 4x3x3, M = 64) that is 192, and the network fits 256 slots, the most
    # whose 128-bit scheme-switching and rotation keys fit in 14 GB.
    # poolcnn-s1 (stride 1, 4x14x14 -> 4x7x7, M = 256) is more accurate but
    # needs 1024 slots. The pool costs 15 levels (12 for the FHEW-to-CKKS
    # decoding of each comparison), 22 in all: it fits ring 2^16 without
    # bootstrapping. The pool's inputs must stay in [-POOL_RANGE, POOL_RANGE]
    # (the comparison wraps around beyond it).
    side = 14 // stride // 2
    return nn.Sequential(
        nn.Conv2d(1, channels, 3, stride=stride, padding=1), nn.BatchNorm2d(channels),
        hnn.MaxPool2d(2, 2, range=POOL_RANGE), hnn.Poly([0, 0.5, 0.25]),
        nn.Flatten(), nn.Linear(channels * side * side, 32), nn.BatchNorm1d(32), hnn.Poly([0, 0.5, 0.25]),
        nn.Linear(32, 10))


POOL_RANGE = 8.0
RANGE_WEIGHT = 1.0
# Softmax and layer-norm intervals of the transformers. The training forward
# clamps to them; check_softmax_ranges() reports how the test data fits.
# Narrow intervals matter under encryption: values are scaled by the
# interval bounds before each bootstrap inside the layers, so a wide
# interval spends the bootstrap's ~10-20 bits of precision on headroom. The
# row-sum interval [0.5, 100] is far inside the worst case [n e^lo, n e^hi]
# and the range loss keeps the sums there.
SOFTMAX = dict(lo=-4.0, hi=4.0, exp_degree=7, inv_degree=7, newton=2, sum_lo=0.5, sum_hi=100.0)
LN = dict(eps=0.05, var_min=0.0, var_max=4.0, degree=7, newton=1)


def lnvit(tokens):
    # Pre-norm transformer block: LayerNorm + 2-head softmax attention, then
    # LayerNorm + MLP; mean over tokens and a linear head.
    d, hid = (16, 32) if tokens == 4 else (32, 64)
    return nn.Sequential(
        nn.Linear(PATCH, d), hnn.PosEmbedding(tokens, d),
        hnn.Residual(hnn.LayerNorm(d, **LN), hnn.Attention(d, tokens, heads=2 if tokens == 4 else 4, softmax=SOFTMAX)),
        hnn.Residual(hnn.LayerNorm(d, **LN), nn.Linear(d, hid), hnn.Poly([0, 0.5, 0.25]), nn.Linear(hid, d)),
        hnn.TokenMean(), nn.Linear(d, 10))


def check_softmax_ranges(model, x):
    """Per-token variance at each LayerNorm, attention scores and row sums,
    against the intervals their series are built for."""
    stats = []

    def ln_hook(m, inp, _):
        v = inp[0].var(-1, unbiased=False)
        stats.append(f"layernorm: variance [{v.min():.3f}, {v.max():.3f}] vs [{m.var_min}, {m.var_max}]")

    def att_hook(m, inp, _):
        z = inp[0]
        b, n, d = z.shape
        q, k = (f(z).reshape(b, n, m.heads, d // m.heads).transpose(1, 2) for f in (m.q, m.k))
        s = q @ k.transpose(-1, -2)
        tot = torch.exp(s.clamp(m.exp_lo, m.exp_hi)).sum(-1)
        stats.append(f"attention: scores [{s.min():.2f}, {s.max():.2f}] vs [{m.exp_lo}, {m.exp_hi}], row sums "
                     f"[{tot.min():.2f}, {tot.max():.2f}] vs [{m.sum_lo:.2f}, {m.sum_hi:.2f}]")

    hooks = [m.register_forward_hook(ln_hook) for m in model.modules() if isinstance(m, hnn.LayerNorm)]
    hooks += [m.register_forward_hook(att_hook) for m in model.modules() if isinstance(m, hnn.Attention)]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    for line in stats:
        print("  " + line)


def pool2(x):
    """(B, 1, 28, 28) -> (B, 1, 14, 14): 2x2 average pooling, done by the client."""
    return x.reshape(-1, 1, 14, 2, 14, 2).mean((3, 5))


def check_ranges(net, xs):
    """Largest |input| of every chebyshev layer over xs, against its interval."""
    xs = [np.asarray(x, dtype=np.float64).ravel() for x in xs]
    for i, layer in enumerate(net.layers):
        if layer.type in ("chebyshev", "exactrelu", "maxpool2d"):
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


def patchify14(x):
    """(B, 1, 28, 28) -> 14x14 -> (B, 4, 49): 2x2 grid of 7x7 patches."""
    return pool2(x).reshape(-1, 2, 7, 2, 7).transpose(0, 1, 3, 2, 4).reshape(-1, 4, PATCH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("cnn", "vit", "deepcnn", "poolcnn", "poolcnn-s1", "lnvit", "lnvit16"), default="cnn")
    ap.add_argument("--profile", action="store_true", help="print each layer's time for the first encrypted image")
    ap.add_argument("--json", help="append the run's numbers to this JSON-lines file")
    ap.add_argument("--bootstrap", action="store_true",
                    help="fold activation intervals and insert bootstraps automatically (ring 2^16)")
    ap.add_argument("--window", type=int, default=0, help="rotation-key ladder window (0 = one key per rotation)")
    ap.add_argument("--calibration", type=int, default=64, help="training images for bootstrap range scaling")
    ap.add_argument("--ring", type=int, help="ring dimension override (insecure runs; bootstrapping precision grows with it)")
    ap.add_argument("--digits", type=int, default=0, help="HYBRID key-switching digits (0 = OpenFHE default; 4 fits "
                    "two more levels in ring 2^16 for 25%% larger keys)")
    ap.add_argument("--level-budget", type=int, default=2, help="bootstrap level budget {b,b}: 2 costs 18 levels and 69 "
                    "bootstrapping keys at 256 slots, 3 costs 20 levels and 45 keys (use 3 with --digits 4)")
    ap.add_argument("--exact-relu", type=float, nargs="?", const=8.0, default=None, metavar="RANGE",
                    help="deepcnn: train with nn.ReLU and run it as the exact scheme-switching ReLU on "
                    "[-RANGE, RANGE] (default 8)")
    ap.add_argument("--packing", choices=("gapped", "compact", "multiplexed"),
                    help="layout of strided conv outputs (default: compact for deepcnn, gapped otherwise)")
    ap.add_argument("--boot-iterations", type=int, default=1, choices=(1, 2),
                    help="2 = iterative bootstrapping: about twice the bits of precision for twice the time")
    ap.add_argument("--boot-precision", type=int, default=0,
                    help="bits of precision of one bootstrap for --boot-iterations 2 (default: measured at keygen)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, help="default: 3 (cnn), 15 (vit)")
    ap.add_argument("--out", help="also save the exported model here (.fheml)")
    ap.add_argument("--n", type=int, default=3, help="images to classify encrypted")
    ap.add_argument("--insecure", action="store_true", help="toy parameters (fast, NOT secure)")
    args = ap.parse_args()

    torch.manual_seed(0)
    data = load_mnist(args.data)
    prep = {"cnn": lambda x: x, "vit": patchify, "deepcnn": pool2, "poolcnn": pool2, "poolcnn-s1": pool2, "lnvit": patchify14,
            "lnvit16": patchify}[args.arch]
    xtr, ytr = torch.tensor(prep(data["train_x"]), dtype=torch.float32), torch.tensor(data["train_y"]).long()
    xte, yte = torch.tensor(prep(data["test_x"]), dtype=torch.float32), torch.tensor(data["test_y"]).long()

    if args.exact_relu and args.arch != "deepcnn":
        ap.error("--exact-relu is for --arch deepcnn")
    model = {"cnn": cnn, "vit": vit, "deepcnn": lambda: deepcnn(bool(args.exact_relu)), "poolcnn": poolcnn,
             "poolcnn-s1": lambda: poolcnn(4, 1), "lnvit": lambda: lnvit(4), "lnvit16": lambda: lnvit(16)}[args.arch]()
    epochs = args.epochs or {"cnn": 3, "vit": 15, "deepcnn": 4, "poolcnn": 4, "poolcnn-s1": 4, "lnvit": 10,
                             "lnvit16": 10}[args.arch]
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * ((len(xtr) + 127) // 128))
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        t = time.time()
        for idx in torch.randperm(len(xtr)).split(128):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xtr[idx]), ytr[idx])
            (loss + RANGE_WEIGHT * hnn.range_loss(model)).backward()  # keep series arguments in their intervals
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # polynomial nets can spike early
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(xte).argmax(1) == yte).float().mean().item()
        print(f"epoch {epoch + 1}: test accuracy {acc:.1%} ({time.time() - t:.0f}s)")

    train_time = time.time() - t0
    shape = {"cnn": (1, 28, 28), "vit": (N_TOK, PATCH), "deepcnn": (1, 14, 14), "poolcnn": (1, 14, 14), "poolcnn-s1": (1, 14, 14),
             "lnvit": (4, PATCH), "lnvit16": (N_TOK, PATCH)}[args.arch]
    if args.arch.startswith("lnvit"):
        check_softmax_ranges(model, xte[:2000])
    net = fheml.from_torch(model, shape, exact_relu=args.exact_relu)  # checks fheml plaintext == torch
    if args.out:
        net.save(args.out)
        print("wrote", args.out)
    if args.arch in ("vit", "lnvit", "lnvit16"):
        net.fuse_linear()  # token mean + head -> one matrix, saves a level
    if args.arch.startswith("poolcnn"):
        net.compile()
        check_ranges(net, xtr[:args.calibration].double().numpy())
    # Accuracy of the network as encrypted inference computes it (the same
    # polynomial approximations, no clamping), in the clear.
    n_plain = min(len(xte), 2000)
    xp = xte[:n_plain].double().numpy()
    plain_acc = float(np.mean([fheml.plain(net, xp[i]).argmax() == int(yte[i]) for i in range(n_plain)]))
    with torch.no_grad():
        torch_head = (model(xte[:n_plain]).argmax(1) == yte[:n_plain]).float().mean().item()
    print(f"fheml plaintext accuracy {plain_acc:.1%} on {n_plain} test images (torch: {torch_head:.1%} on them, "
          f"{acc:.1%} on all)")
    level_budget = [3, 3]
    if args.packing:
        net.packing = args.packing
    elif args.arch in ("deepcnn", "poolcnn", "poolcnn-s1"):
        # Compact activations: fewer slots, so fewer bootstrapping keys, and
        # max pooling compares only the slots up to its last input.
        net.set_gapped_layouts(False)
    if args.bootstrap:
        # Levels per stretch at ring 2^16 and 50 bits: 5 with {2,2}, 3 with {3,3}, two more with --digits 4.
        level_budget = [args.level_budget, args.level_budget]
        calib = xtr[:args.calibration].double().numpy()
        net.compile()
        check_ranges(net, calib)
        depth = net.depth
        boots, budget = fheml.auto_bootstrap(net, calib, level_budget=level_budget, max_ring=1 << 16,
                                             key_switch_digits=args.digits,
                                             bootstrap_iterations=args.boot_iterations)
        print(f"auto_bootstrap: depth {depth} -> {net.depth}, {boots} bootstraps, at most {budget} levels between them")
    net.compile()
    print(net.summary())

    t = time.time()
    overrides = {"level_budget": level_budget, "key_switch_digits": args.digits,
                 "bootstrap_iterations": args.boot_iterations, "bootstrap_precision": args.boot_precision}
    if args.ring or (args.insecure and net.uses_bootstrapping):
        overrides["ring_dim"] = args.ring or 4096
    ctx = fheml.make_context(net, secure=not args.insecure, rotation_window=args.window, **overrides)
    print(ctx.describe(), f"(setup {time.time() - t:.1f}s)")
    setup = time.time() - t
    model.double()
    model.eval()
    x = xte.double().numpy()
    times, errs, agree = [], [], 0
    for i in range(args.n):
        t = time.time()
        if args.profile and i == 0:
            layer_times = []
            y = net.forward(ctx, fheml.encrypt(net, ctx, x[i]),
                            hook=lambda k, what, c, secs: layer_times.append((k, what, secs)))
            enc = fheml.decrypt(net, ctx, y)
            for k, what, secs in layer_times:
                print(f"  layer {k} ({secs:.1f}s): {what[:110]}")
        else:
            enc = fheml.infer(net, ctx, x[i])
        dt = time.time() - t
        with torch.no_grad():
            ref = model(xte[i:i + 1].double()).numpy().ravel()
        times.append(dt)
        errs.append(float(np.abs(enc - ref).max()))
        agree += int(enc.argmax() == ref.argmax())
        print(f"image {i}: label {int(yte[i])}, encrypted -> {enc.argmax()}, torch -> {ref.argmax()}, "
              f"max |enc - torch| {errs[-1]:.1e}, {dt:.1f}s")
    if args.json:
        import json
        import resource
        rec = dict(arch=args.arch, secure=not args.insecure, bootstrap=args.bootstrap, torch_accuracy=acc,
                   plain_accuracy=plain_acc, torch_accuracy_plain_images=torch_head, plain_images=n_plain, train_seconds=train_time, depth=net.depth,
                   uses_bootstrapping=net.uses_bootstrapping, internal_bootstraps=net.internal_bootstraps,
                   bootstrap_layers=sum(1 for layer in net.layers if layer.type == "bootstrap"),
                   context=ctx.describe(), setup_seconds=setup, image_seconds=times, max_abs_err=errs,
                   argmax_agree=agree, images=args.n,
                   peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
        with open(args.json, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
