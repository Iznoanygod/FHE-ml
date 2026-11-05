"""Object detection and semantic segmentation on COCO 2017, encrypted end to
end: the server runs the network and all of its post-processing on the
ciphertext, and the client only decrypts the finished answer.

    PYTHONPATH=build/python python3 examples/python/coco_cnn.py --task detect --classes person \
        --insecure --bootstrap --window 1 --digits 4 --scale-bits 59 --n 2
    PYTHONPATH=build/python python3 examples/python/coco_cnn.py --task segment --classes person \
        --insecure --bootstrap --window 1 --digits 4 --scale-bits 59 --n 2
    PYTHONPATH=build/python python3 examples/python/coco_cnn.py --task detect --n 0   # train and score only

The images are squashed to 48x48 RGB. A fully convolutional backbone (five
3x3 convolutions with BatchNorm and ReLU, three of them stride 2) turns the
image into a 6x6 grid of feature vectors and a 1x1 convolution reads each
cell's prediction out of its vector. ReLU is a Chebyshev series of degree
31 on [-20, 20] that the torch model trains with, so encrypted inference
computes what was trained; --activation quad swaps in the CryptoNets-style
0.5x + 0.25x^2 for comparison. The network is about 60 levels deep, so it
runs with --bootstrap: the series' intervals fold into the convolutions and
bootstraps are placed automatically (never a decryption).

* detect: a single-shot grid detector. Every cell predicts its box (centre
  inside the cell, width and height as fractions of the image) and K class
  scores, all through one Chebyshev sigmoid (per-class sigmoid scores, as in
  YOLOv3, so no softmax is needed); the cell holding a box's centre is
  responsible for it. Non-maximum suppression is a 3x3 local maximum per
  class (as in CenterNet): a LinearMap gathers every score's 3x3
  neighbourhood (itself first, zero padding) into 9 channels, the encrypted
  ArgMax (scheme-switching comparisons) marks with c = 1 the scores that
  are their neighbourhood's maximum, and a residual adds c - 1 to each
  score. A second LinearMap decodes the boxes, adding each cell's own
  position: x0 = (gx + sx) / 6 - sw / 2 and so on. A ReLU (Chebyshev, degree
  31 on [-1.5, 1.5]) then zeroes the suppressed scores (s + c - 1 < 0 when
  c = 0) and clips the box corners at 0. The decrypted (4 + K, 6, 6) map
  holds a box per cell and K scores that are 0 unless the cell is a
  detection.
* segment: every cell predicts the class logits of its own 4x4 block of a
  24x24 label map, (K + 1) * 16 channels; a LinearMap rearranges them into
  the (K + 1, 24, 24) map (pixel shuffle, a permutation), so the model is a
  linear decoder over the grid and needs no upsampling layer; the encrypted
  per-pixel ArgMax over the channels turns the logits into the one-hot label
  map that the client decrypts.

Sizes are set by memory: 48x48 keeps every activation within 8192 slots,
so the toy context is ring 2^14 (64x64 needs 16384 slots, whose bootstrapping
keys alone go over 14 GB). The comparisons cost one FHEW bootstrap per
compared slot, thousands per image. 128-bit parameters do not fit this
machine. --tail none stops after the sigmoid scores / class logits.

The classes default to six frequent COCO categories with large instances
(person, car, bus, train, cat, dog); --classes all uses the 80. Training uses
the first --train-images images of train2017 (fetched one by one from the
archive), evaluation the 5000 images of val2017. At this size half of
COCO's boxes of these classes are smaller than a grid cell (8 pixels), so
detection is trained and scored on the objects whose sqrt(area) is at least
--min-side of the image (default one cell); smaller objects and crowd
regions are ignore regions: a detection on them is neither a hit nor a false
positive. It reports COCO-style AP at IoU 0.5 and averaged over 0.5:0.95,
plus AP50 on COCO's large objects (area > 96^2 in the original image).
Segmentation reports mean IoU over background + K classes and pixel accuracy
of the 24x24 label maps (ground truth subsampled to 24x24).
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import fheml
from fheml import nn as hnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
import fhe_eval  # noqa: E402
from datasets import load_coco  # noqa: E402

SIZE, GRID = 48, 6  # input side, detection grid (cells of 8 pixels)
LOAD = 64  # the cached COCO size, resized to SIZE
MASK = 24  # resolution of the predicted label maps
CELL = MASK // GRID
SIG = 8.0  # the sigmoid's Chebyshev interval [-SIG, SIG] (range_loss keeps the logits inside)
SIG_DEGREE = 31
# The tail's final ReLU on [-1.5, 1.5]: degree 31 is within 0.024 of relu
# (0.005 beyond |x| > 0.05), fits one bootstrapping stretch, and needs no
# scheme switching (an exact ReLU would make the comparison keys cover every
# slot).
TAIL_RELU_DEGREE = 31
RANGE_WEIGHT = 10.0
HEADROOM = 0.6
DEFAULT_CLASSES = ("person", "car", "bus", "train", "cat", "dog")
MEAN = np.array([0.471, 0.448, 0.408]).reshape(1, 3, 1, 1)
STD = np.array([0.274, 0.270, 0.284]).reshape(1, 3, 1, 1)
IOUS = np.round(np.arange(0.5, 0.96, 0.05), 2)


# The backbone ReLU's Chebyshev interval and degree. Batch-normalized
# pre-activations of a few images reach 10-20 even with the range penalty,
# and a series evaluated outside its interval diverges (no clamping under
# encryption), so the interval is wide. Degree 31 is within 0.3 of relu (at 0);
# training sees the polynomial itself, so the network adapts to it.
RELU = 20.0
RELU_DEGREE = 31


class Series(hnn.Chebyshev):
    """A Chebyshev activation whose training range penalty starts at
    HEADROOM times its interval: the training images then stay well
    inside it, and the occasional test image that goes further still lands
    where the series is accurate."""

    def forward(self, x):
        y = super().forward(x)
        if self.training:
            self._excess = hnn._excess(x, HEADROOM * self.a, HEADROOM * self.b)
        return y


def backbone(out_channels, width=12, activation="relu"):
    if activation == "relu":
        def act():
            return Series("relu", -RELU, RELU, RELU_DEGREE)
    else:  # the CryptoNets-style quadratic, for comparison
        def act():
            return hnn.Poly([0, 0.5, 0.25])
    w = width
    return nn.Sequential(
        nn.Conv2d(3, w, 3, stride=2, padding=1), nn.BatchNorm2d(w), act(),                # 24x24
        nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), nn.BatchNorm2d(2 * w), act(),        # 12x12
        nn.Conv2d(2 * w, 2 * w, 3, stride=1, padding=1), nn.BatchNorm2d(2 * w), act(),
        nn.Conv2d(2 * w, 4 * w, 3, stride=2, padding=1), nn.BatchNorm2d(4 * w), act(),    # 6x6
        nn.Conv2d(4 * w, 4 * w, 3, stride=1, padding=1), nn.BatchNorm2d(4 * w), act(),
        nn.Conv2d(4 * w, out_channels, 1))


# ------------------------------------------------------------------- data

class Split:
    """One COCO split restricted to the chosen classes: normalized images,
    per-image boxes / labels / original areas / crowd flags, and the
    semantic label map (0 background, 1 + class). Detection is trained and
    scored on the non-crowd boxes at least min_side of the image in size
    (square root of the box area); smaller and crowd boxes are ignored, as
    COCO ignores crowd regions."""

    def __init__(self, d, keep, min_side):
        remap = np.full(80, -1, np.int64)
        remap[keep] = np.arange(len(keep))
        x = torch.tensor((d["x"] / 255.0 - MEAN) / STD, dtype=torch.float32)
        self.x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)
        n = len(self.x)
        lab = remap[d["box_label"]]
        sel = lab >= 0
        empty = lambda shape, t: [np.zeros(shape, t) for _ in range(n)]  # noqa: E731
        self.boxes, self.labels, self.areas, self.crowd = (empty((0, 4), np.float32), empty(0, np.int64),
                                                           empty(0, np.float32), empty(0, bool))
        order = np.argsort(d["box_image"][sel], kind="stable")
        img, box, lab_, area, crowd = (a[sel][order] for a in (d["box_image"], d["box"], lab, d["box_area"],
                                                               d["box_crowd"].astype(bool)))
        for i, s, c in zip(*np.unique(img, return_index=True, return_counts=True)):
            e = s + c
            self.boxes[i], self.labels[i], self.areas[i], self.crowd[i] = box[s:e], lab_[s:e], area[s:e], crowd[s:e]
        self.min_side = min_side
        self.scored = [~cr & (np.sqrt(np.prod(b[:, 2:] - b[:, :2], 1)) >= min_side)
                       for b, cr in zip(self.boxes, self.crowd)]
        lut = np.zeros(81, np.int64)
        lut[1 + np.asarray(keep)] = 1 + np.arange(len(keep))
        mask = torch.tensor(lut[np.asarray(d["mask"], np.int64)])
        self.mask = F.interpolate(mask[:, None].float(), size=(SIZE, SIZE), mode="nearest")[:, 0].long()
        self.has = np.array([m.any() for m in self.scored])

    def truth(self, i, c, large=False):
        """(scored boxes, ignored boxes) of class c in image i; large keeps
        only COCO's large objects (area > 96^2 in the original image)."""
        m = self.labels[i] == c
        keep = m & self.scored[i] & ((self.areas[i] > 96 ** 2) if large else True)
        return self.boxes[i][keep], self.boxes[i][m & ~keep]

    def grid_targets(self):
        """(N, 6, GRID, GRID): objectness, centre offset in the cell (x, y),
        width, height (fractions of the image), class. A cell holding several
        centres keeps the largest box."""
        t = np.zeros((len(self.x), 6, GRID, GRID), np.float32)
        best = np.zeros((len(self.x), GRID, GRID), np.float32)
        for i, (bs, ls, sc) in enumerate(zip(self.boxes, self.labels, self.scored)):
            for (x0, y0, x1, y1), c in zip(bs[sc], ls[sc]):
                cx, cy = (x0 + x1) / 2 * GRID, (y0 + y1) / 2 * GRID
                gx, gy = min(int(cx), GRID - 1), min(int(cy), GRID - 1)
                a = (x1 - x0) * (y1 - y0)
                if a > best[i, gy, gx]:
                    best[i, gy, gx] = a
                    t[i, :, gy, gx] = [1, cx - gx, cy - gy, x1 - x0, y1 - y0, c]
        return torch.tensor(t)


def flip_detect(x, t):
    """Horizontal flip of images and grid targets."""
    t = t.flip(-1).clone()
    t[:, 1] = torch.where(t[:, 0] > 0, 1 - t[:, 1], t[:, 1])
    return x.flip(-1), t


# -------------------------------------------------------------- detection

def detect_loss(p, t, classes):
    """p: the sigmoid outputs (B, 4 + K, G, G); t: grid_targets."""
    pos = t[:, 0] > 0
    onehot = F.one_hot(t[:, 5].long(), classes).permute(0, 3, 1, 2).float() * pos[:, None]
    q = p[:, 4:].clamp(1e-4, 1 - 1e-4)
    # Per-class binary cross-entropy, positives weighted up (about 1 in 60 cells holds a box).
    loss = -(4 * onehot * q.log() + (1 - onehot) * (1 - q).log()).mean() * classes
    if pos.any():
        o, g = p.permute(0, 2, 3, 1)[pos], t.permute(0, 2, 3, 1)[pos]
        loss = loss + 5 * F.mse_loss(o[:, :4], g[:, 1:5])
    return loss


def shuffle_map(channels, r, grid):
    """LinearMap of pixel shuffle: (channels * r^2, grid, grid) -> (channels, grid * r, grid * r)."""
    n = channels * r * r * grid * grid
    src = F.pixel_shuffle(torch.arange(n, dtype=torch.float64).reshape(1, channels * r * r, grid, grid), r)
    return hnn.LinearMap((torch.arange(n), src.flatten().long(), torch.ones(n)), (channels, grid * r, grid * r),
                         name="pixelshuffle", in_size=n)


class PixelShuffle(nn.Module):
    """F.pixel_shuffle while training; exported as its LinearMap."""

    def __init__(self, channels, r, grid):
        super().__init__()
        self.map = shuffle_map(channels, r, grid)
        self.r = r

    def forward(self, x):
        return F.pixel_shuffle(x, self.r)


def nms_tail(classes, range_):
    """Encrypted local-maximum suppression and box decoding on the sigmoid
    outputs (4 + K, G, G) -> (4 + K, G, G): corners, then scores that stay
    only where they are the maximum of their 3x3 neighbourhood. Ties go to
    the centre, which comes first in the gather."""
    k, g = classes, GRID
    cells = g * g
    # Gather: 9 channels (centre, then the 8 neighbours) of a (K * G, G) image.
    rows, cols = [], []
    offsets = [(0, 0)] + [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    for j, (dy, dx) in enumerate(offsets):
        for c in range(k):
            for y in range(g):
                for x in range(g):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < g and 0 <= xx < g:
                        rows.append(j * k * cells + c * cells + y * g + x)
                        cols.append((4 + c) * cells + yy * g + xx)
    n_in = (4 + k) * cells
    gather = hnn.LinearMap((rows, cols, np.ones(len(rows))), (9, k * g, g), name="nms-gather", in_size=n_in)
    # Back: the centre's one-hot bit c, minus 1, onto each score.
    m = 9 * k * cells
    back = hnn.LinearMap((np.arange(4 * cells, n_in), np.arange(k * cells), np.ones(k * cells)), (4 + k, g, g),
                         bias=np.r_[np.zeros(4 * cells), -np.ones(k * cells)], name="nms-keep", in_size=m)
    # Decode: x0 = (gx + sx)/G - sw/2, y0, x1, y1; scores pass through.
    rows, cols, vals, bias = [], [], [], np.zeros(n_in)
    gy, gx = np.divmod(np.arange(cells), g)
    for corner, (off, size, sign) in enumerate(((0, 2, -1), (1, 3, -1), (0, 2, 1), (1, 3, 1))):
        for i in range(cells):
            rows += [corner * cells + i, corner * cells + i]
            cols += [off * cells + i, size * cells + i]
            vals += [1 / g, sign / 2]
            bias[corner * cells + i] = (gx[i] if off == 0 else gy[i]) / g
    rows += list(range(4 * cells, n_in))
    cols += list(range(4 * cells, n_in))
    vals += [1.0] * (k * cells)
    decode = hnn.LinearMap((rows, cols, vals), (4 + k, g, g), bias=bias, name="box-decode", in_size=n_in)
    return nn.Sequential(hnn.Residual(gather, hnn.ArgMax(range_, dim=1), back), decode,
                         hnn.Chebyshev("relu", -1.5, 1.5, TAIL_RELU_DEGREE))


def detections(out, classes, floor=0.03):
    """Decrypted (4 + K) * G * G outputs -> per class (D, 5) arrays (x0, y0,
    x1, y1, score): the cells whose score survived suppression, i.e. above
    the final ReLU's approximation error (0.024 at 0). Reading only, no
    computation."""
    o = np.asarray(out, np.float64).reshape(4 + classes, GRID * GRID)
    boxes = o[:4].T
    return [np.concatenate([boxes[o[4 + c] > floor], o[4 + c][o[4 + c] > floor, None]], 1) for c in range(classes)]


def detection_scores(outputs, split, idx, names, ious=(0.5,), large=False):
    """Per-class AP over images idx at each IoU; returns (mAP per IoU,
    per-class AP50, large-object AP50)."""
    k = len(names)
    dets = [detections(o, k) for o in np.asarray(outputs)]
    ap = np.full((len(ious), k), np.nan)
    ap_large = np.full(k, np.nan)
    for c in range(k):
        d = [dd[c] for dd in dets]
        g, ign = zip(*(split.truth(i, c) for i in idx))
        for j, t in enumerate(ious):
            ap[j, c], _ = fhe_eval.average_precision(d, g, t, ignore=ign)
        if large:
            g, ign = zip(*(split.truth(i, c, large=True) for i in idx))
            ap_large[c], _ = fhe_eval.average_precision(d, g, 0.5, ignore=ign)
    return np.nanmean(ap, 1), ap[0], np.nanmean(ap_large) if large else float("nan")


# ----------------------------------------------------------- segmentation

def activation_ranges(model, x):
    """Largest |input| of every Chebyshev activation over x, against its interval."""
    peaks = {}

    def hook(m, inp, _):
        a = inp[0].abs()
        peaks[m] = max(peaks.get(m, 0.0), float(a.max()) if torch.isfinite(a).all() else float("inf"))

    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, hnn.Chebyshev)]
    fhe_eval.predict(model, x)
    for h in hooks:
        h.remove()
    for m, p in peaks.items():
        print(f"  {m.fn} on [{m.a:g}, {m.b:g}]: max |input| {p:.2f} over {len(x)} val images"
              + ("  OUTSIDE" if p > m.b else ""))


def labels(out, classes):
    """(B, (K + 1) * MASK^2) one-hot or logit maps -> (B, MASK, MASK) labels."""
    return torch.as_tensor(np.asarray(out)).reshape(-1, classes + 1, MASK, MASK).argmax(1).numpy()


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=("detect", "segment"), default="detect")
    ap.add_argument("--tail", choices=("encrypted", "none"), default="encrypted",
                    help="encrypted: NMS / argmax under encryption (default); none: stop at the sigmoid "
                    "scores / class logits (polynomial only, runs at 128-bit)")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                    help="comma-separated COCO category names, or 'all' (default: %(default)s)")
    ap.add_argument("--train-images", type=int, default=40000,
                    help="first N images of train2017 (those without a chosen class are dropped)")
    ap.add_argument("--min-side", type=float, default=1 / GRID,
                    help="detection: smallest object scored, as sqrt(box area) / image side (default one grid "
                    "cell, 8 of 48 pixels); smaller ones are ignored")
    ap.add_argument("--activation", choices=("relu", "quad"), default="relu",
                    help="relu: Chebyshev ReLU of degree 31 on [-20, 20] (needs --bootstrap); quad: 0.5x + 0.25x^2 "
                    "(shallow, no bootstrapping; comparison)")
    ap.add_argument("--width", type=int, default=12, help="channels of the first convolution")
    ap.add_argument("--checkpoint", help="torch weights: loaded if the file exists (no training), else saved after "
                    "training")
    ap.add_argument("--eval-images", type=int, default=5000, help="val2017 images to score (plaintext)")
    fhe_eval.add_args(ap, n=2)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    val = load_coco(args.data, "val", LOAD, masks=True)
    all_names = list(val["names"])
    names = all_names if args.classes == "all" else args.classes.split(",")
    unknown = [n for n in names if n not in all_names]
    if unknown:
        ap.error(f"unknown COCO categories {unknown}")
    keep = [all_names.index(n) for n in names]
    k = len(keep)
    tr = Split(load_coco(args.data, "train", LOAD, max_images=args.train_images, masks=True), keep, args.min_side)
    te = Split(val, keep, args.min_side)
    # Training images with at least one scored box (detection) or labelled
    # pixel (segmentation) of the chosen classes; all val images (the model
    # has to stay quiet on the others).
    sel = np.nonzero(tr.has if args.task == "detect" else tr.mask.flatten(1).any(1).numpy())[0]
    te_idx = np.arange(min(args.eval_images, len(te.x)))
    print(f"{k} classes, {len(sel)} training images, {len(te_idx)} val images, "
          f"{sum(te.scored[i].sum() for i in te_idx)} scored val boxes "
          f"(of {sum(len(te.boxes[i]) for i in te_idx)})")

    if args.task == "detect":
        # Backbone, then one sigmoid over all 4 + K channels.
        net_t = nn.Sequential(backbone(4 + k, args.width, args.activation),
                              Series("sigmoid", -SIG, SIG, SIG_DEGREE))
        t = tr.grid_targets()[sel]
        xf, tf = flip_detect(tr.x[sel], t)
        xtr, ytr = torch.cat([tr.x[sel], xf]), torch.cat([t, tf])
        del xf, tf, t

        def loss_fn(p, t):
            return detect_loss(p, t, k)

        # While training, the suppression and decoding run in torch (exactly
        # what the encrypted tail computes) on the sigmoid outputs.
        tail = nms_tail(k, 1.5)

        def metric(out, idx, tailed=False):
            out = torch.as_tensor(out)
            if not tailed:
                with torch.no_grad():
                    out = tail(out.reshape(-1, 4 + k, GRID, GRID))
            return detection_scores(out, te, np.asarray(idx), names)[0][0]
        metric.__name__ = "mAP50"
    else:
        net_t = nn.Sequential(backbone((k + 1) * CELL * CELL, args.width, args.activation),
                              PixelShuffle(k + 1, CELL, GRID))
        sub = SIZE // MASK
        xtr = torch.cat([tr.x[sel], tr.x[sel].flip(-1)])
        ytr = torch.cat([tr.mask[sel, ::sub, ::sub], tr.mask[sel, ::sub, ::sub].flip(-1)])
        weight = torch.ones(k + 1)
        weight[0] = 0.5  # background covers most pixels

        def loss_fn(out, y):
            return F.cross_entropy(out, y, weight=weight)

        def metric(out, idx):
            return fhe_eval.mean_iou(labels(out, k), te.mask[np.asarray(idx), ::sub, ::sub].numpy(), k + 1)[0]
        metric.__name__ = "mean_IoU"

    xte, yte = te.x[te_idx], torch.as_tensor(te_idx)
    epochs = args.epochs or 25
    if args.checkpoint and os.path.exists(args.checkpoint):
        net_t.load_state_dict(torch.load(args.checkpoint))
        train_time = float("nan")
        print("loaded", args.checkpoint)
    else:
        # Per-epoch progress on the first 1000 val images; the full scores
        # follow. A strong range penalty: the Chebyshev series diverge
        # outside their intervals, and encrypted inference cannot clamp.
        _, train_time = fhe_eval.train(net_t, xtr, ytr, xte[:1000], yte[:1000], epochs, loss_fn=loss_fn,
                                       metric=metric, lr=3e-3, batch=128, range_weight=RANGE_WEIGHT)
        if args.checkpoint:
            torch.save(net_t.state_dict(), args.checkpoint)
    activation_ranges(net_t, xte)
    calib = xtr[:args.calibration].double().numpy()
    del xtr, ytr, tr  # the training set (GBs) is not needed past here

    # The exported model: the trained network, then (default) the encrypted tail.
    out = fhe_eval.predict(net_t, xte)
    if args.task == "detect":
        print(f"sigmoid outputs on val: [{out.min():.3f}, {out.max():.3f}]")
        full = nn.Sequential(net_t, tail) if args.tail == "encrypted" else net_t
        maps, ap50, large = detection_scores(fhe_eval.predict(nn.Sequential(net_t, tail), xte), te, te_idx, names,
                                             IOUS, large=True)
        print(f"torch: mAP50 {maps[0]:.3f}, mAP50:95 {maps.mean():.3f}, large-object mAP50 {large:.3f}")
        print("  AP50 per class: " + ", ".join(f"{n} {a:.3f}" for n, a in zip(names, ap50)))
        scores = dict(map50=maps[0], map50_95=maps.mean(), map50_large=large,
                      ap50={n: a for n, a in zip(names, ap50)})
    else:
        bound = float(out.abs().max())
        rng = float(np.ceil(1.25 * bound))
        print(f"class logits on val within +-{bound:.1f}: ArgMax range {rng}")
        full = nn.Sequential(net_t, hnn.ArgMax(rng, dim=1)) if args.tail == "encrypted" else net_t
        gt = te.mask[te_idx, ::SIZE // MASK, ::SIZE // MASK].numpy()
        pred = labels(out, k)
        miou, ious = fhe_eval.mean_iou(pred, gt, k + 1)
        pacc = fhe_eval.pixel_accuracy(pred, gt)
        print(f"torch: mean IoU {miou:.3f}, pixel accuracy {pacc:.3f}")
        print("  IoU per class: " + ", ".join(f"{n} {a:.3f}" for n, a in zip(["background"] + names, ious)))
        scores = dict(mean_iou=miou, pixel_accuracy=pacc, iou={n: a for n, a in zip(["background"] + names, ious)})

    full.eval()
    net = fheml.from_torch(full, (3, SIZE, SIZE))  # checks fheml plaintext == torch
    if args.out:
        net.save(args.out)
        print("wrote", args.out)
    # The encrypted computation in the clear, on the first val images.
    n_plain = min(200 if args.tail == "none" else 20, len(te_idx))
    tailed = args.task == "detect" and args.tail == "encrypted"
    full_metric = (lambda out, idx: metric(out, idx, tailed=True)) if tailed else metric  # noqa: E731
    plain, n_plain = fhe_eval.plain_score(net, xte.double().numpy(), yte, metric=full_metric, n=n_plain)
    torch_n = full_metric(fhe_eval.predict(full.double(), xte[:n_plain].double()), te_idx[:n_plain])
    print(f"fheml plaintext {metric.__name__} {plain:.3f} on {n_plain} val images (torch {torch_n:.3f} on them)")

    if args.n == 0:
        return
    level_budget = fhe_eval.prepare(net, args, calib, packing="multiplexed")
    ctx, setup = fhe_eval.context(net, args, level_budget)

    def describe(i, enc, ref):
        if args.task == "detect":
            if args.tail == "none":
                enc, ref = (tail(torch.as_tensor(v).reshape(1, 4 + k, GRID, GRID)).numpy() for v in (enc, ref))
            de, dr = detections(enc, k), detections(ref, k)
            top = sorted(((d[4], names[c]) for c, dd in enumerate(de) for d in dd), reverse=True)[:3]
            same = all(len(a) == len(b) and (len(a) == 0 or np.abs(a - b).max() < 1e-2) for a, b in zip(de, dr))
            top = ", ".join(f"{n} {s:.2f}" for s, n in top) or "none"
            return (f"{te.scored[i].sum()} scored boxes; {sum(len(d) for d in de)} encrypted detections, top: {top}; "
                    f"{'same as' if same else 'DIFFERENT from'} torch")
        pe, pr = labels(enc[None], k), labels(ref[None], k)
        miou = fhe_eval.mean_iou(pe, te.mask[i:i + 1, ::SIZE // MASK, ::SIZE // MASK].numpy(), k + 1)[0]
        return f"labels agreeing with torch {(pe == pr).mean():.4f}, image mean IoU {miou:.3f}"

    run = fhe_eval.encrypted_run(net, ctx, full, xte.double().numpy(), args, describe=describe)
    fhe_eval.record(args, net, ctx, setup, run, task=args.task, tail=args.tail, classes=names, width=args.width,
                    train_images=len(sel), train_seconds=train_time, plain_metric=plain, plain_images=n_plain,
                    **scores)


if __name__ == "__main__":
    main()
