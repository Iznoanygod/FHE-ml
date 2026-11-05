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
pre-norm LayerNorm, multi-head softmax attention and a GELU MLP block, on 4
tokens (14x14 image, 7x7 patches) or 16 tokens (28x28): about 60 levels, so
--bootstrap is needed at 128-bit security; layer norm and softmax attention
then bootstrap internally (see Network.place_bootstraps).

    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch poolcnn
    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch lnvit --bootstrap --window 1 --level-budget 3 --digits 4
"""
import argparse
import os
import sys

import torch
from torch import nn

import fheml
from fheml import nn as hnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
import fhe_eval  # noqa: E402
from datasets import load_mnist  # noqa: E402
from fhe_eval import check_ranges, check_softmax_ranges  # noqa: E402,F401

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
# Softmax and layer-norm intervals of the transformers. The training forward
# clamps to them; check_softmax_ranges() reports how the test data fits.
# Narrow intervals matter under encryption: values are scaled by the
# interval bounds before each bootstrap inside the layers, so a wide
# interval spends the bootstrap's ~10-20 bits of precision on headroom. The
# row-sum interval [0.5, 100] is far inside the worst case [n e^lo, n e^hi]
# and the range loss keeps the sums there.
SOFTMAX = dict(lo=-4.0, hi=4.0, exp_degree=7, inv_degree=7, newton=2, sum_lo=0.5, sum_hi=100.0)
LN = dict(eps=0.05, var_min=0.0, var_max=4.0, degree=7, newton=1)
# The transformers' MLPs use GELU, as real ones do: a Chebyshev series on
# [-GELU_RANGE, GELU_RANGE] (5 levels at degree 15). Training clamps the
# pre-activations to the interval and the range loss keeps them inside.
GELU_RANGE, GELU_DEGREE = 6.0, 15


def gelu():
    return hnn.Chebyshev("gelu", -GELU_RANGE, GELU_RANGE, GELU_DEGREE)


def lnvit(tokens):
    # Pre-norm transformer block: LayerNorm + 2-head softmax attention, then
    # LayerNorm + GELU MLP; mean over tokens and a linear head.
    d, hid = (16, 32) if tokens == 4 else (32, 64)
    return nn.Sequential(
        nn.Linear(PATCH, d), hnn.PosEmbedding(tokens, d),
        hnn.Residual(hnn.LayerNorm(d, **LN), hnn.Attention(d, tokens, heads=2 if tokens == 4 else 4, softmax=SOFTMAX)),
        hnn.Residual(hnn.LayerNorm(d, **LN), nn.Linear(d, hid), gelu(), nn.Linear(hid, d)),
        hnn.TokenMean(), nn.Linear(d, 10))


def pool2(x):
    """(B, 1, 28, 28) -> (B, 1, 14, 14): 2x2 average pooling, done by the client."""
    return x.reshape(-1, 1, 14, 2, 14, 2).mean((3, 5))


def vit():
    # Tokens are 7x7 patches; softmax -> quadratic score, no LayerNorm; GELU MLP.
    return nn.Sequential(
        nn.Linear(PATCH, D), hnn.PosEmbedding(N_TOK, D),
        hnn.Residual(hnn.Attention(D, N_TOK)),
        hnn.Residual(nn.Linear(D, HID), gelu(), nn.Linear(HID, D)),
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
    ap.add_argument("--exact-relu", type=float, nargs="?", const=8.0, default=None, metavar="RANGE",
                    help="deepcnn: train with nn.ReLU and run it as the exact scheme-switching ReLU on "
                    "[-RANGE, RANGE] (default 8)")
    ap.add_argument("--logits", action="store_true",
                    help="vit / lnvit: return the logits instead of the encrypted argmax (one-hot class)")
    fhe_eval.add_args(ap)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
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
    # The transformers' GELU series diverge outside [-6, 6]: a stronger range
    # penalty keeps their inputs there, so no test image's logits blow up
    # (which would also cost the encrypted argmax its resolution).
    range_weight = 10.0 if args.arch in ("vit", "lnvit", "lnvit16") else fhe_eval.RANGE_WEIGHT
    acc, train_time = fhe_eval.train(model, xtr, ytr, xte, yte, epochs, range_weight=range_weight)

    shape = {"cnn": (1, 28, 28), "vit": (N_TOK, PATCH), "deepcnn": (1, 14, 14), "poolcnn": (1, 14, 14), "poolcnn-s1": (1, 14, 14),
             "lnvit": (4, PATCH), "lnvit16": (N_TOK, PATCH)}[args.arch]
    if args.arch.startswith("lnvit"):
        check_softmax_ranges(model, xte[:2000])
    if args.arch in ("vit", "lnvit", "lnvit16") and not args.logits:
        # The prediction runs encrypted too: an exact argmax (scheme
        # switching) returns the one-hot of the class. Its comparisons need a
        # bound on the logits, taken from the training images.
        bound = fhe_eval.argmax_range(model, xtr[:2000])
        model = nn.Sequential(*model, hnn.ArgMax(bound))
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
    plain_acc, n_plain = fhe_eval.plain_score(net, xte.double().numpy(), yte)
    torch_head = fhe_eval.accuracy(fhe_eval.predict(model, xte[:n_plain]), yte[:n_plain])
    print(f"fheml plaintext accuracy {plain_acc:.1%} on {n_plain} test images (torch: {torch_head:.1%} on them, "
          f"{acc:.1%} on all)")
    if not args.packing and args.arch in ("deepcnn", "poolcnn", "poolcnn-s1"):
        # Compact activations: fewer slots, so fewer bootstrapping keys, and
        # max pooling compares only the slots up to its last input.
        net.set_gapped_layouts(False)
    level_budget = fhe_eval.prepare(net, args, xtr[:args.calibration].double().numpy())
    ctx, setup = fhe_eval.context(net, args, level_budget)
    run = fhe_eval.encrypted_run(
        net, ctx, model, xte.double().numpy(), args,
        describe=lambda i, enc, ref: f"label {int(yte[i])}, encrypted -> {enc.argmax()}, torch -> {ref.argmax()}")
    fhe_eval.record(args, net, ctx, setup, run, arch=args.arch, torch_accuracy=acc, plain_accuracy=plain_acc,
                    torch_accuracy_plain_images=torch_head, plain_images=n_plain, train_seconds=train_time)


if __name__ == "__main__":
    main()
