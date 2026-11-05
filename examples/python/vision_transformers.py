"""Vision transformers beyond MNIST: CIFAR-10 classification (ViT) and Oxford
Pet semantic segmentation, each with a single-head and a multi-head variant,
trained in PyTorch, exported to fheml and run encrypted end to end: from the
encrypted patches to an encrypted one-hot prediction, with no decryption in
between.

    F="--bootstrap --level-budget 3 --digits 4 --scale-bits 59 --window 1 --insecure --ring 16384"
    PYTHONPATH=build/python python3 examples/python/vision_transformers.py --task cifar --heads 1 $F
    PYTHONPATH=build/python python3 examples/python/vision_transformers.py --task cifar --heads 4 $F
    PYTHONPATH=build/python python3 examples/python/vision_transformers.py --task pets --heads 1 $F
    PYTHONPATH=build/python python3 examples/python/vision_transformers.py --task pets --heads 4 $F

The client cuts the image into non-overlapping patches before encrypting
(a reshape: every token is one flattened patch) and reads the decrypted
one-hot. Everything else runs on ciphertexts: the patch embedding, the
transformer blocks, the head and the argmax.

cifar   32x32 RGB images, 8x8 grid of 4x4 patches (64 tokens of 48 values).
        ViT: patch embedding + learned positions, transformer blocks, final
        layer norm, mean over tokens, linear classifier, argmax over the 10
        classes. Metric: top-1 accuracy.

pets    Oxford-IIIT Pet trimaps (pet / background / border) at 32x32, the
        same 64 patch tokens. Segmentation transformer with a per-patch
        linear decoder (as in SETR-naive and Segmenter's linear decoder):
        every output token holds the class logits of the 4x4 pixels of its
        patch, so no upsampling layer is needed, and the argmax runs per
        pixel over its 3 classes (1024 pixels, 3072 encrypted comparisons).
        Metrics: mean IoU over the three classes and pixel accuracy.

Blocks (--block):

vit     (default) a standard pre-norm transformer block: LayerNorm, softmax
        attention, LayerNorm, MLP with GELU. LayerNorm's 1/sqrt, softmax's
        exp and 1/x and GELU are Chebyshev series; the model trains with
        their arguments clamped to (and penalised outside) their intervals
        and runs on intervals 1.5x wider (widen()), so test images that
        stray a little past the training ranges stay accurate. About 70
        levels: --bootstrap places bootstraps between layers and layer norm
        and softmax refresh inside. d = 32: at d = 64 the keys of the
        16384-slot ciphertexts exceed 14 GB even at toy parameters.
basic   the "very basic" transformer: attention with the quadratic score
        A = c0 + c1 S + c2 S^2 instead of softmax, no layer norm and a
        square MLP activation; d = 64. With --logits it is 12 levels and
        runs at 128-bit security on ring 2^15 without bootstrapping.

--heads 1 is single-head attention; --heads H splits the model dimension
into H heads (sub-blocks of each token's slots, so under encryption extra
heads cost no split/concat rotations and need fewer rotation keys).
--logits returns the logits instead of the encrypted argmax.
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
from datasets import PET_MASK_CLASSES, load_cifar10, load_pets  # noqa: E402

SIZE, PATCH = 32, 4
GRID = SIZE // PATCH            # 8x8 patches
TOKENS = GRID * GRID            # 64 tokens
FEATURES = 3 * PATCH * PATCH    # 48 values per patch
CLASSES = {"cifar": 10, "pets": len(PET_MASK_CLASSES)}
SQUARE = [0.0, 0.5, 0.25]       # 0.5x + 0.25x^2: one level

# Intervals of the polynomial softmax, layer norm and GELU (--block vit), as
# in the MNIST lnvit example: narrow intervals keep the internal bootstraps
# precise; the range loss keeps training inside them.
SOFTMAX = dict(lo=-4.0, hi=4.0, exp_degree=7, inv_degree=7, newton=2, sum_lo=0.5, sum_hi=100.0)
LN = dict(eps=0.05, var_min=0.0, var_max=4.0, degree=7, newton=1)
GELU_RANGE, GELU_DEGREE = 6.0, 15


def patchify(x):
    """(B, 3, 32, 32) -> (B, 64, 48): row-major 8x8 grid of 4x4 patches, each
    flattened as (channel, row, column)."""
    b = len(x)
    return x.reshape(b, 3, GRID, PATCH, GRID, PATCH).transpose(0, 2, 4, 1, 3, 5).reshape(b, TOKENS, FEATURES)


def unpatchify_logits(y, classes):
    """Per-patch decoder output (B, 64, 16 * classes) -> pixel logits
    (B, classes, 32, 32). Token feature (r * 4 + c) * classes + k is class k
    of pixel (r, c) of the patch."""
    y = torch.as_tensor(y).reshape(-1, GRID, GRID, PATCH, PATCH, classes)
    return y.permute(0, 5, 1, 3, 2, 4).reshape(-1, classes, SIZE, SIZE)


def block(d, hid, heads, kind):
    """One transformer block as two residual branches (attention, MLP)."""
    if kind == "basic":
        return [hnn.Residual(hnn.Attention(d, TOKENS, heads=heads)),
                hnn.Residual(nn.Linear(d, hid), hnn.Poly(SQUARE), nn.Linear(hid, d))]
    return [hnn.Residual(hnn.LayerNorm(d, **LN), hnn.Attention(d, TOKENS, heads=heads, softmax=SOFTMAX)),
            hnn.Residual(hnn.LayerNorm(d, **LN), nn.Linear(d, hid), hnn.Chebyshev("gelu", -GELU_RANGE, GELU_RANGE,
                                                                                  GELU_DEGREE), nn.Linear(hid, d))]


def widen(model, factor=1.5, degree=15):
    """Training margin: the model trains with its series' arguments clamped
    to (and penalised outside) the intervals above, then the series are
    rebuilt on intervals `factor` times wider, at a higher degree, for
    inference. A few test images push attention scores or layer-norm
    variances a little past the training intervals; inside the wider ones
    the series still follow exp, 1/x, 1/sqrt and GELU instead of diverging.
    Returns the model (changed in place)."""
    for m in model.modules():
        if isinstance(m, hnn.Attention) and m.softmax:
            o = m.softmax
            o.update(lo=o["lo"] * factor, hi=o["hi"] * factor, sum_lo=o["sum_lo"] / factor,
                     sum_hi=o["sum_hi"] * factor, exp_degree=degree, inv_degree=degree)
            m.exp_c = hnn._series_buffer(np.exp, m.exp_lo, m.exp_hi, degree)
            m.inv_c = hnn._series_buffer(lambda u: 1.0 / u, m.sum_lo, m.sum_hi, degree)
        elif isinstance(m, hnn.LayerNorm):
            m.var_max, m.degree = m.var_max * factor, degree
            m.c = hnn._series_buffer(lambda v: 1.0 / np.sqrt(v + m.eps), m.var_min, m.var_max, degree)
        elif isinstance(m, hnn.Chebyshev):
            m.a, m.b, m.degree = m.a * factor, m.b * factor, degree
            m.c = hnn._series_buffer(hnn._FUNCTIONS[m.fn], m.a, m.b, degree)
    return model


def model(task, d, hid, heads, depth, kind):
    layers = [nn.Linear(FEATURES, d), hnn.PosEmbedding(TOKENS, d)]
    for _ in range(depth):
        layers += block(d, hid, heads, kind)
    if kind == "vit":  # final layer norm before the head, as in ViT
        layers += [hnn.LayerNorm(d, **LN)]
    if task == "cifar":
        layers += [hnn.TokenMean(), nn.Linear(d, CLASSES["cifar"])]
    else:  # per-patch linear decoder: the logits of every pixel of the patch
        layers += [nn.Linear(d, PATCH * PATCH * CLASSES["pets"])]
    return nn.Sequential(*layers)


# ------------------------------------------------------------------ data

def test_order(n):
    """A fixed shuffle of the test set: the loaders return it sorted by class
    (CIFAR-10) or breed (Pets), and the plaintext and encrypted checks use its
    first images."""
    return np.random.default_rng(1234).permutation(n)


def cifar_data(args):
    data = load_cifar10(args.data)
    mean = data["train_x"].reshape(-1, 3, SIZE * SIZE).mean((0, 2)) / 255
    std = data["train_x"].reshape(-1, 3, SIZE * SIZE).std((0, 2)) / 255

    def prep(x):
        x = (x.astype(np.float32) / 255 - mean[:, None, None]) / std[:, None, None]
        return torch.tensor(patchify(x), dtype=torch.float32)

    te = test_order(len(data["test_y"]))
    return (prep(data["train_x"]), torch.tensor(data["train_y"]),
            prep(data["test_x"][te]), torch.tensor(data["test_y"][te]))


def cifar_augment(xb):
    """Random horizontal flips and shifts by up to one patch row/column of
    pixels (4), on the patch tokens: rebuild the images, shift, re-cut."""
    b = len(xb)
    img = xb.reshape(b, GRID, GRID, 3, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5).reshape(b, 3, SIZE, SIZE)
    flip = torch.rand(b) < 0.5
    img = torch.where(flip[:, None, None, None], img.flip(-1), img)
    pad = nn.functional.pad(img, (4, 4, 4, 4))
    dx, dy = torch.randint(0, 9, (2,))
    img = pad[:, :, dy:dy + SIZE, dx:dx + SIZE]
    return img.reshape(b, 3, GRID, PATCH, GRID, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(b, TOKENS, FEATURES)


def pets_data(args):
    data = load_pets(args.data, size=SIZE)
    mean = data["train_x"].reshape(-1, 3, SIZE * SIZE).mean((0, 2)) / 255
    std = data["train_x"].reshape(-1, 3, SIZE * SIZE).std((0, 2)) / 255

    def prep(x):
        x = (x.astype(np.float32) / 255 - mean[:, None, None]) / std[:, None, None]
        return torch.tensor(patchify(x), dtype=torch.float32)

    te = test_order(len(data["test_mask"]))
    return (prep(data["train_x"]), torch.tensor(data["train_mask"]).long(),
            prep(data["test_x"][te]), torch.tensor(data["test_mask"][te]).long())


def pets_augment_pairs(xb, yb):
    """Random horizontal flips of images and their masks together."""
    b = len(xb)
    img = xb.reshape(b, GRID, GRID, 3, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5).reshape(b, 3, SIZE, SIZE)
    flip = torch.rand(b) < 0.5
    img = torch.where(flip[:, None, None, None], img.flip(-1), img)
    yb = torch.where(flip[:, None, None], yb.flip(-1), yb)
    return img.reshape(b, 3, GRID, PATCH, GRID, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(b, TOKENS, FEATURES), yb


def seg_loss(out, masks):
    return nn.functional.cross_entropy(unpatchify_logits(out, CLASSES["pets"]), masks)


def seg_pred(out):
    return unpatchify_logits(out, CLASSES["pets"]).argmax(1)


def seg_miou(out, masks):
    """Mean IoU over pet / background / border of the per-pixel argmax."""
    return fhe_eval.mean_iou(seg_pred(out).numpy(), np.asarray(masks), CLASSES["pets"])[0]


seg_miou.__name__ = "mean_iou"


# ------------------------------------------------------------------ main

DEFAULTS = {
    # (task, block): (d, hidden, blocks, epochs). The vit blocks are half as
    # wide: at d = 64 the bootstrapping, attention and switching keys of a
    # 16384-slot ciphertext go over 14 GB even at toy parameters; d = 32
    # (8192 slots) fits.
    ("cifar", "vit"): (32, 64, 1, 30),
    ("pets", "vit"): (32, 64, 1, 40),
    ("cifar", "basic"): (64, 128, 1, 30),
    ("pets", "basic"): (64, 128, 1, 40),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", choices=("cifar", "pets"), default="cifar")
    ap.add_argument("--heads", type=int, default=1, help="attention heads (1 = single-head)")
    ap.add_argument("--block", choices=("vit", "basic"), default="vit",
                    help="vit: pre-norm LayerNorm, softmax attention, GELU MLP (needs --bootstrap); basic: "
                    "quadratic attention and square MLP, no layer norm")
    ap.add_argument("--dim", type=int, help="model dimension (power of two)")
    ap.add_argument("--hidden", type=int, help="MLP hidden width")
    ap.add_argument("--blocks", type=int, help="transformer blocks")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weights", help="torch weights: loaded from this file if it exists (no training), else "
                    "saved there after training")
    ap.add_argument("--logits", action="store_true",
                    help="return the logits instead of the encrypted argmax (one-hot class / per-pixel label)")
    fhe_eval.add_args(ap)
    args = ap.parse_args()
    d, hid, depth, epochs = DEFAULTS[args.task, args.block]
    d, hid, depth = args.dim or d, args.hidden or hid, args.blocks or depth
    epochs = args.epochs or epochs

    torch.manual_seed(args.seed)
    seg = args.task == "pets"
    # The polynomial softmax / layer norm / GELU diverge outside their
    # intervals: penalise excursions hard while training.
    range_weight = 10.0 if args.block == "vit" else fhe_eval.RANGE_WEIGHT
    xtr, ytr, xte, yte = (pets_data if seg else cifar_data)(args)
    print(f"{args.task}: {len(xtr)} train / {len(xte)} test, {TOKENS} tokens of {FEATURES}")
    net_model = model(args.task, d, hid, args.heads, depth, args.block)
    print(net_model)

    metric = seg_miou if seg else fhe_eval.accuracy
    if args.weights and os.path.exists(args.weights):
        if args.block == "vit":
            widen(net_model)  # the saved weights are the widened model's
        net_model.load_state_dict(torch.load(args.weights))
        train_time = 0.0
        score = metric(fhe_eval.predict(net_model, xte), yte)
        print(f"loaded {args.weights}: test score {score:.4f}")
    else:
        if seg:
            score, train_time = fhe_eval.train(net_model, xtr, ytr, xte, yte, epochs, loss_fn=seg_loss,
                                               metric=seg_miou, lr=args.lr, batch=64,
                                               augment_pairs=pets_augment_pairs, range_weight=range_weight)
        else:
            score, train_time = fhe_eval.train(net_model, xtr, ytr, xte, yte, epochs, lr=args.lr, batch=128,
                                               augment=cifar_augment, range_weight=range_weight)
        if args.block == "vit":
            widen(net_model)
            score = metric(fhe_eval.predict(net_model, xte), yte)
            print(f"intervals widened 1.5x for inference: test score {score:.4f}")
        if args.weights:
            torch.save(net_model.state_dict(), args.weights)
    if args.block == "vit":
        fhe_eval.check_softmax_ranges(net_model, xtr[:2000])
    trained = net_model
    if not args.logits:
        # The prediction itself runs encrypted: an exact argmax through
        # scheme switching, over the 10 classes (CIFAR) or the 3 classes of
        # every pixel (Pets), returning a one-hot. Its comparisons need a
        # bound on the logits, taken from the training images.
        bound = fhe_eval.argmax_range(net_model, xtr[:2000])
        net_model = nn.Sequential(*net_model, hnn.ArgMax(bound, classes=CLASSES["pets"] if seg else None))
    net = fheml.from_torch(net_model, (TOKENS, FEATURES))  # checks fheml plaintext == torch
    if args.out:
        net.save(args.out)
        print("wrote", args.out)
    if not seg:
        net.fuse_linear()  # token mean + classifier -> one matrix, saves a level

    # The metric of the network as encrypted inference computes it, in the clear.
    n_plain = 500 if seg else 2000
    plain, n_plain = fhe_eval.plain_score(net, xte.double().numpy(), yte, metric=metric, n=n_plain)
    torch_sub = metric(fhe_eval.predict(net_model, xte[:n_plain]), yte[:n_plain])
    name = "mean IoU" if seg else "accuracy"
    print(f"fheml plaintext {name} {plain:.4f} on {n_plain} test images (torch: {torch_sub:.4f} on them, "
          f"{score:.4f} on all {len(xte)})")
    extra = {}
    if seg:
        out = fhe_eval.predict(trained, xte)
        extra["torch_pixel_accuracy"] = fhe_eval.pixel_accuracy(seg_pred(out).numpy(), yte.numpy())
        ious = fhe_eval.mean_iou(seg_pred(out).numpy(), yte.numpy(), CLASSES["pets"])[1]
        print("torch per-class IoU: " + ", ".join(f"{c} {v:.3f}" for c, v in zip(PET_MASK_CLASSES, ious))
              + f"; pixel accuracy {extra['torch_pixel_accuracy']:.4f}")

    level_budget = fhe_eval.prepare(net, args, xtr[:args.calibration].double().numpy())
    ctx, setup = fhe_eval.context(net, args, level_budget)

    def describe(i, enc, ref):
        if not seg:
            return f"label {int(yte[i])}, encrypted -> {enc.argmax()}, torch -> {ref.argmax()}"
        pe, pr = seg_pred(enc[None]).numpy(), seg_pred(ref[None]).numpy()
        agree = float((pe == pr).mean())
        return (f"pixel accuracy encrypted {fhe_eval.pixel_accuracy(pe, yte[i:i + 1].numpy()):.3f} "
                f"(torch {fhe_eval.pixel_accuracy(pr, yte[i:i + 1].numpy()):.3f}), pixels agreeing {agree:.4f}")

    run = fhe_eval.encrypted_run(net, ctx, net_model, xte.double().numpy(), args, describe=describe)
    if seg:
        del run["argmax_agree"]  # a whole-map argmax means nothing here; pixels_agree replaces it
        outs = np.stack(run["outputs"])
        pe = seg_pred(outs).numpy()
        pr = seg_pred(fhe_eval.predict(net_model, xte[:len(outs)].double())).numpy()
        truth = yte[:len(outs)].numpy()
        extra.update(encrypted_mean_iou=fhe_eval.mean_iou(pe, truth, CLASSES["pets"])[0],
                     encrypted_pixel_accuracy=fhe_eval.pixel_accuracy(pe, truth),
                     torch_mean_iou_same_images=fhe_eval.mean_iou(pr, truth, CLASSES["pets"])[0],
                     pixels_agree=float((pe == pr).mean()))
        print(f"encrypted on {len(outs)} images: mean IoU {extra['encrypted_mean_iou']:.4f} (torch on them "
              f"{extra['torch_mean_iou_same_images']:.4f}), pixel accuracy {extra['encrypted_pixel_accuracy']:.4f}, "
              f"pixels agreeing with torch {extra['pixels_agree']:.4f}")
    else:
        print(f"encrypted: {run['argmax_agree']}/{run['images']} predictions agree with torch")
    extra["argmax"] = not args.logits
    fhe_eval.record(args, net, ctx, setup, run, task=args.task, heads=args.heads, block=args.block, dim=d,
                    hidden=hid, blocks=depth, torch_score=score, plain_score=plain, plain_images=n_plain,
                    torch_score_plain_images=torch_sub, train_seconds=train_time, **extra)


if __name__ == "__main__":
    main()
