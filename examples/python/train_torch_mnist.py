"""Train an HE-friendly MNIST model in PyTorch, export it to fheml and run it
encrypted.

    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch cnn [--n 3] [--insecure]
    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch mlp
    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch vit
    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch deepcnn --bootstrap
    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch deepcnn --bootstrap --exact-relu

The MNIST baselines run on the full 28x28 image, fully encrypted, with
Chebyshev ReLU or sigmoid activations: mlp is the 784-200-10 sigmoid network
of the original prototype, mlp-relu the same with ReLU, mlp-deep /
mlp-deep-relu are 784-512-256-128-10 versions, and cnn is a
CryptoNets-shaped CNN with ReLUs. The -poly variants (mlp-poly,
mlp-deep-poly, cnn-poly) use the quadratic activation of CryptoNets for
comparison. mnist_mlp_cnn.py runs them side by side.

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

poolcnn is a small ReLU CNN with exact max pooling (CKKS <-> FHEW scheme
switching; one FHEW bootstrap per compared slot, so it keeps the pooled
feature map in few slots); at 30 levels it needs --bootstrap.
poolcnn-poly is the quadratic version, 22 levels without bootstrapping. lnvit / lnvit16 are vision transformers with
pre-norm LayerNorm, multi-head softmax attention and a GELU MLP block, on 4
tokens (14x14 image, 7x7 patches) or 16 tokens (28x28): about 60 levels, so
--bootstrap is needed at 128-bit security; layer norm and softmax attention
then bootstrap internally (see Network.place_bootstraps).

    PYTHONPATH=build/python python3 examples/python/train_torch_mnist.py --arch poolcnn --bootstrap --insecure
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
HIDDEN, SIGMOID_RANGE, RELU_RANGE = 200, 16.0, 8.0
DEEP_HIDDEN = (512, 256, 128)
MLP_LR = 1e-2


def activation(act):
    """The hidden activation of the MNIST MLPs and CNNs, as encrypted
    inference computes it: relu (degree-15 Chebyshev on [-8, 8]) or sigmoid
    (degree 15 on [-16, 16]), 6 levels each once the interval is folded into
    the preceding layer; poly is the CryptoNets quadratic 0.5x + 0.25x^2
    (2 levels), kept as the cheap comparison point."""
    if act == "poly":
        return hnn.Poly([0, 0.5, 0.25])
    r = SIGMOID_RANGE if act == "sigmoid" else RELU_RANGE
    return hnn.Chebyshev(act, -r, r, 15)


def mlp(act="sigmoid", hidden=(HIDDEN,)):
    # Fully connected network, 784-200-10 by default: the network of the 2022
    # prototype (legacy/). There the client decrypted the hidden layer,
    # applied the sigmoid in the clear and re-encrypted it; here every
    # activation is a Chebyshev series and the whole network stays
    # encrypted. BatchNorm, folded into the preceding layer on export, and
    # the range loss keep the pre-activations inside the series' interval (a
    # few test values still land just outside; the accuracies count them).
    # The sigmoid saturates, so the sigmoid MLPs train with lr 1e-2 (MLP_LR)
    # instead of 2e-3. The prototype's output sigmoid is dropped: it is
    # monotone, so the argmax is unchanged. hidden= gives the hidden widths;
    # DEEP_HIDDEN is the deeper 784-512-256-128-10 network.
    layers, width = [nn.Flatten()], 784
    for h in hidden:
        layers += [nn.Linear(width, h), nn.BatchNorm1d(h), activation(act)]
        width = h
    return nn.Sequential(*layers, nn.Linear(width, 10))


def cnn(act="relu"):
    # CryptoNets/LoLa shape: 5x5 stride-2 convolution with 5 channels, a
    # 845-64 dense layer and a 64-10 head. from_torch folds BatchNorm into
    # the preceding layer, so it costs nothing under encryption. act="poly"
    # is the original CryptoNets network with quadratic activations.
    return nn.Sequential(
        nn.Conv2d(1, 5, 5, stride=2, padding=1), nn.BatchNorm2d(5), activation(act),
        nn.Flatten(), nn.Linear(845, 64), nn.BatchNorm1d(64), activation(act),
        nn.Linear(64, 10))


# The MNIST baselines on full 28x28 images: builder, epochs, learning rate.
BASELINES = {
    "mlp": (lambda: mlp("sigmoid"), 5, MLP_LR),
    "mlp-relu": (lambda: mlp("relu"), 5, 2e-3),
    "mlp-poly": (lambda: mlp("poly"), 5, 2e-3),
    "mlp-deep": (lambda: mlp("sigmoid", DEEP_HIDDEN), 5, MLP_LR),
    "mlp-deep-relu": (lambda: mlp("relu", DEEP_HIDDEN), 5, 2e-3),
    "mlp-deep-poly": (lambda: mlp("poly", DEEP_HIDDEN), 5, 2e-3),
    "cnn": (lambda: cnn("relu"), 3, 2e-3),
    "cnn-poly": (lambda: cnn("poly"), 3, 2e-3),
}


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


def poolcnn(channels=4, stride=2, act="relu"):
    # conv -> ReLU -> exact 2x2 max pool -> dense -> ReLU -> dense, the usual
    # order of a small CNN. The pool gathers its four taps into blocks of M
    # slots (M = the pooled size rounded up to a power of two) and compares
    # 2M then M slots, one FHEW bootstrap each: with the default 4 channels
    # at stride 2 (4x7x7 -> 4x3x3, M = 64) that is 192, and the network fits
    # 256 slots. poolcnn-s1 (stride 1, 4x14x14 -> 4x7x7, M = 256) is more
    # accurate but needs 1024 slots. The pool costs 15 levels (12 for the
    # FHEW-to-CKKS decoding of each comparison) and each Chebyshev ReLU 6,
    # 30 in all, so the ReLU models need --bootstrap. The pool's inputs must
    # stay in [-POOL_RANGE, POOL_RANGE] (the comparison wraps around beyond
    # it). act="poly" (poolcnn-poly) puts the quadratic after the pool
    # instead: 22 levels, which fit ring 2^16 without bootstrapping.
    side = 14 // stride // 2
    conv = [nn.Conv2d(1, channels, 3, stride=stride, padding=1), nn.BatchNorm2d(channels)]
    pool = hnn.MaxPool2d(2, 2, range=POOL_RANGE)
    head = [nn.Flatten(), nn.Linear(channels * side * side, 32), nn.BatchNorm1d(32), activation(act),
            nn.Linear(32, 10)]
    if act == "poly":
        return nn.Sequential(*conv, pool, activation(act), *head)
    return nn.Sequential(*conv, activation(act), pool, *head)


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
    ap.add_argument("--arch", default="cnn", choices=(*BASELINES, "vit", "deepcnn", "poolcnn", "poolcnn-s1",
                                                      "poolcnn-poly", "lnvit", "lnvit16"))
    ap.add_argument("--exact-relu", type=float, nargs="?", const=8.0, default=None, metavar="RANGE",
                    help="deepcnn: train with nn.ReLU and run it as the exact scheme-switching ReLU on "
                    "[-RANGE, RANGE] (default 8)")
    ap.add_argument("--logits", action="store_true",
                    help="vit / lnvit: return the logits instead of the encrypted argmax (one-hot class)")
    fhe_eval.add_args(ap)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data = load_mnist(args.data)
    baseline = args.arch in BASELINES
    prep = (lambda x: x) if baseline else {"vit": patchify, "deepcnn": pool2, "poolcnn": pool2, "poolcnn-s1": pool2, "poolcnn-poly": pool2,
                                           "lnvit": patchify14, "lnvit16": patchify}[args.arch]
    xtr, ytr = torch.tensor(prep(data["train_x"]), dtype=torch.float32), torch.tensor(data["train_y"]).long()
    xte, yte = torch.tensor(prep(data["test_x"]), dtype=torch.float32), torch.tensor(data["test_y"]).long()

    if args.exact_relu and args.arch != "deepcnn":
        ap.error("--exact-relu is for --arch deepcnn")
    if baseline:
        build, epochs, lr = BASELINES[args.arch]
        model = build()
    else:
        model = {"vit": vit, "deepcnn": lambda: deepcnn(bool(args.exact_relu)), "poolcnn": poolcnn,
                 "poolcnn-s1": lambda: poolcnn(4, 1), "poolcnn-poly": lambda: poolcnn(act="poly"),
                 "lnvit": lambda: lnvit(4),
                 "lnvit16": lambda: lnvit(16)}[args.arch]()
        epochs = {"vit": 15, "deepcnn": 4, "poolcnn": 4, "poolcnn-s1": 4, "poolcnn-poly": 4, "lnvit": 10, "lnvit16": 10}[args.arch]
        lr = 2e-3
    acc, train_time = fhe_eval.train(model, xtr, ytr, xte, yte, args.epochs or epochs, lr=lr)

    shape = (1, 28, 28) if baseline else {"vit": (N_TOK, PATCH), "deepcnn": (1, 14, 14), "poolcnn": (1, 14, 14),
                                          "poolcnn-s1": (1, 14, 14), "poolcnn-poly": (1, 14, 14), "lnvit": (4, PATCH),
                                          "lnvit16": (N_TOK, PATCH)}[args.arch]
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
    if baseline or args.arch.startswith("poolcnn"):
        net.compile()
        check_ranges(net, xtr[:args.calibration].double().numpy())
    # Accuracy of the network as encrypted inference computes it (the same
    # polynomial approximations, no clamping), in the clear.
    plain_acc, n_plain = fhe_eval.plain_score(net, xte.double().numpy(), yte)
    torch_head = fhe_eval.accuracy(fhe_eval.predict(model, xte[:n_plain]), yte[:n_plain])
    print(f"fheml plaintext accuracy {plain_acc:.1%} on {n_plain} test images (torch: {torch_head:.1%} on them, "
          f"{acc:.1%} on all)")
    if not args.packing and (args.arch == "deepcnn" or args.arch.startswith("poolcnn")):
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
