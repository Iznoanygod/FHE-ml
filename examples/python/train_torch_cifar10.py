"""CIFAR-10 CNNs: train in PyTorch, export to fheml and classify encrypted
32x32 colour images.

    PYTHONPATH=build/python python3 examples/python/train_torch_cifar10.py --arch cnn --window 1
    PYTHONPATH=build/python python3 examples/python/train_torch_cifar10.py --arch deepcnn --bootstrap \
        --level-budget 3 --digits 4 --insecure --ring 8192 --scale-bits 59
    PYTHONPATH=build/python python3 examples/python/train_torch_cifar10.py --arch poolcnn --bootstrap \
        --level-budget 3 --digits 4 --insecure --ring 8192 --scale-bits 59

The client sends the whole normalised 3x32x32 image (3072 values, 4096
slots); every layer runs on the server. The three models share one shape,
two stride-2 stages and a pooled head, and differ in what they exercise:

cnn       the quadratic-activation baseline (CryptoNets style), for
          comparison: conv 3->16 s2, conv 16->32 s2, conv 32->32, each with BatchNorm
          (folded into the convolution) and x/2 + x^2/4, 2x2 average
          pooling and a 512->10 head. 11 levels: runs at 128-bit security
          (ring 2^15) without bootstrapping.
deepcnn   six convolutions (16, 16, 32, 32, 64, 64 channels; three stride-2)
          with degree-7 Chebyshev ReLUs on [-8, 8], global average pooling
          and a 64->10 head. 38 levels: auto_bootstrap() folds the ReLU
          intervals into the convolutions and places one bootstrap per
          convolution block.
poolcnn   deepcnn's first four convolution blocks, then an exact 2x2
          MaxPool2d (CKKS <-> FHEW scheme switching) on the 32x8x8 map and
          a 512->10 head: convolutions, multiplexed packing, exact max
          pooling and bootstrapping in one network. The pool compares 1024
          then 512 slots, one FHEW bootstrap each, so it is the slow part.

The bootstrapped models run on toy parameters: 4096-slot bootstrapping
keys at ring 2^16 take more than 14 GB, and a 4096-slot bootstrap needs
59-bit scaling to keep 12 bits of precision (about 3 at 50 bits, see the
README).
--weights caches the trained torch model between runs.

Strided convolutions use multiplexed packing (each layer takes the cheapest
of multiplexed, gapped and compact; --packing overrides), so the 16x16 and
8x8 maps keep the 4096 input slots instead of spreading out by the stride.
Training uses random crops and horizontal flips; CIFAR-10 downloads into
data/ (fast.ai's mirror).
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch import nn

import fheml
from fheml import nn as hnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
import fhe_eval  # noqa: E402
from datasets import CIFAR10_CLASSES, load_cifar10  # noqa: E402

MEAN = np.array([0.4914, 0.4822, 0.4465], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.2470, 0.2435, 0.2616], np.float32).reshape(1, 3, 1, 1)
RELU = 8.0       # Chebyshev ReLU interval [-RELU, RELU]; BatchNorm and the range loss keep inputs inside
POOL_RANGE = 8.0  # |inputs| of the exact max pool (the comparison wraps around beyond it)


def sq():
    # x/2 + x^2/4: two levels.
    return hnn.Poly([0, 0.5, 0.25])


def relu():
    # Degree 7: four levels, three once its interval folds into the
    # convolution, so a convolution and its ReLU fit one bootstrap stretch.
    return hnn.Chebyshev("relu", -RELU, RELU, 7)


def conv(cin, cout, stride, act):
    return [nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.BatchNorm2d(cout), act()]


def cnn():
    return nn.Sequential(
        *conv(3, 16, 2, sq), *conv(16, 32, 2, sq), *conv(32, 32, 1, sq),
        nn.AvgPool2d(2), nn.Flatten(), nn.Linear(32 * 4 * 4, 10))


def deepcnn():
    return nn.Sequential(
        *conv(3, 16, 2, relu), *conv(16, 16, 1, relu),
        *conv(16, 32, 2, relu), *conv(32, 32, 1, relu),
        *conv(32, 64, 2, relu), *conv(64, 64, 1, relu),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))


def poolcnn():
    return nn.Sequential(
        *conv(3, 16, 2, relu), *conv(16, 16, 1, relu),
        *conv(16, 32, 2, relu), *conv(32, 32, 1, relu),
        hnn.MaxPool2d(2, 2, range=POOL_RANGE),
        nn.Flatten(), nn.Linear(32 * 4 * 4, 10))


# builder, epochs, learning rate
MODELS = {"cnn": (cnn, 30, 4e-3), "deepcnn": (deepcnn, 40, 2e-3), "poolcnn": (poolcnn, 40, 2e-3)}


def augment(x):
    """Random 32x32 crops of the 4-pixel zero-padded batch and horizontal flips."""
    n = len(x)
    pad = nn.functional.pad(x, (4, 4, 4, 4))
    dy, dx = torch.randint(0, 9, (2,))
    out = pad[:, :, dy:dy + 32, dx:dx + 32]
    flip = torch.rand(n) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(-1), out)


def normalise(x):
    return (x.astype(np.float32) / 255.0 - MEAN) / STD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=tuple(MODELS), default="cnn")
    ap.add_argument("--weights", help="torch weights: loaded if the file exists, else saved after training")
    fhe_eval.add_args(ap)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data = load_cifar10(args.data)
    xtr, ytr = torch.tensor(normalise(data["train_x"])), torch.tensor(data["train_y"]).long()
    # The cached test set is sorted by class: shuffle it once (fixed order) so
    # the plaintext check and the encrypted inputs cover every class.
    order = np.random.default_rng(0).permutation(len(data["test_y"]))
    xte, yte = torch.tensor(normalise(data["test_x"][order])), torch.tensor(data["test_y"][order]).long()

    build, epochs, lr = MODELS[args.arch]
    model = build()
    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights))
        acc, train_time = fhe_eval.accuracy(fhe_eval.predict(model, xte), yte), 0.0
        print(f"loaded {args.weights}: test accuracy {acc:.4f}")
    else:
        acc, train_time = fhe_eval.train(model, xtr, ytr, xte, yte, args.epochs or epochs, lr=lr,
                                         weight_decay=5e-4, augment=augment)
        if args.weights:
            torch.save(model.state_dict(), args.weights)

    net = fheml.from_torch(model, (3, 32, 32))  # checks fheml plaintext == torch
    if args.out:
        net.save(args.out)
        print("wrote", args.out)
    net.compile()
    fhe_eval.check_ranges(net, xtr[:args.calibration].double().numpy())
    # The accuracy of the network as encrypted inference computes it (same
    # polynomials, no clamping), in the clear.
    plain_acc, n_plain = fhe_eval.plain_score(net, xte.double().numpy(), yte)
    torch_head = fhe_eval.accuracy(fhe_eval.predict(model, xte[:n_plain]), yte[:n_plain])
    print(f"fheml plaintext accuracy {plain_acc:.1%} on {n_plain} test images (torch: {torch_head:.1%} on them, "
          f"{acc:.1%} on all)")

    level_budget = fhe_eval.prepare(net, args, xtr[:args.calibration].double().numpy(), packing="multiplexed")
    ctx, setup = fhe_eval.context(net, args, level_budget)
    run = fhe_eval.encrypted_run(
        net, ctx, model, xte.double().numpy(), args,
        describe=lambda i, enc, ref: f"{CIFAR10_CLASSES[int(yte[i])]}, encrypted -> "
        f"{CIFAR10_CLASSES[enc.argmax()]}, torch -> {CIFAR10_CLASSES[ref.argmax()]}")
    fhe_eval.record(args, net, ctx, setup, run, arch=args.arch, dataset="cifar10", torch_accuracy=acc,
                    plain_accuracy=plain_acc, torch_accuracy_plain_images=torch_head, plain_images=n_plain,
                    train_seconds=train_time)


if __name__ == "__main__":
    main()
