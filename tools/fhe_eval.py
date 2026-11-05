"""Shared harness of the fheml examples: train a torch model, export it, check
the plaintext fheml model, place bootstraps, build a context and run test
inputs encrypted, then report and record the numbers.

An example script defines its model and data and calls these in order:

    ap = argparse.ArgumentParser(); fhe_eval.add_args(ap); args = ap.parse_args()
    fhe_eval.train(model, xtr, ytr, xte, yte, epochs=..., metric=fhe_eval.accuracy)
    net = fheml.from_torch(model, shape)              # checks fheml plaintext == torch
    fhe_eval.plain_score(net, xte, yte, ...)          # the encrypted computation, in the clear
    level_budget = fhe_eval.prepare(net, args, calib) # packing, bootstraps, compile
    ctx, setup = fhe_eval.context(net, args, level_budget)
    run = fhe_eval.encrypted_run(net, ctx, model, xte, args)
    fhe_eval.record(args, arch=..., **run)

``add_args`` gives every example the same encryption flags (--insecure,
--bootstrap, --window, --digits, --level-budget, --ring, --boot-iterations,
--boot-precision, --scale-bits, --packing, --n, --profile, --json). The metrics below
cover classification, segmentation, detection and sequence tasks.
"""
import json
import resource
import time

import numpy as np
import torch
from torch import nn

import fheml
from fheml import nn as hnn

RANGE_WEIGHT = 1.0


def add_args(ap, n=3):
    """The encryption options shared by the examples."""
    ap.add_argument("--data", default="data", help="dataset directory (downloads go here)")
    ap.add_argument("--epochs", type=int, help="training epochs (default: per model)")
    ap.add_argument("--out", help="also save the exported model here (.fheml)")
    ap.add_argument("--n", type=int, default=n, help="test inputs to run encrypted")
    ap.add_argument("--insecure", action="store_true", help="toy parameters (fast, NOT secure)")
    ap.add_argument("--profile", action="store_true", help="print each layer's time for the first encrypted input")
    ap.add_argument("--json", help="append the run's numbers to this JSON-lines file")
    ap.add_argument("--bootstrap", action="store_true",
                    help="fold activation intervals and insert bootstraps automatically (ring 2^16)")
    ap.add_argument("--window", type=int, default=0, help="rotation-key ladder window (0 = one key per rotation)")
    ap.add_argument("--calibration", type=int, default=64, help="training inputs for bootstrap range scaling")
    ap.add_argument("--ring", type=int, help="ring dimension override (insecure runs; bootstrapping precision grows with it)")
    ap.add_argument("--digits", type=int, default=0, help="HYBRID key-switching digits (0 = OpenFHE default; 4 fits "
                    "two more levels in ring 2^16 for 25%% larger keys)")
    ap.add_argument("--level-budget", type=int, default=2, help="bootstrap level budget {b,b}: 2 costs 18 levels and 69 "
                    "bootstrapping keys at 256 slots, 3 costs 20 levels and 45 keys (use 3 with --digits 4)")
    ap.add_argument("--packing", choices=("gapped", "compact", "multiplexed"),
                    help="layout of strided conv outputs (default: per model)")
    ap.add_argument("--boot-iterations", type=int, default=1, choices=(1, 2),
                    help="2 = iterative bootstrapping: about twice the bits of precision for twice the time")
    ap.add_argument("--boot-precision", type=int, default=0,
                    help="bits of precision of one bootstrap for --boot-iterations 2 (default: measured at keygen)")
    ap.add_argument("--scale-bits", type=int, default=0,
                    help="CKKS scaling modulus bits (0 = 50). Toy runs of bootstrapped networks: 59 makes a "
                    "4096-slot bootstrap about 2^-14 accurate instead of 2^-3 (128-bit ring 2^16 fits 50)")
    ap.add_argument("--seed", type=int, default=0)
    return ap


# --------------------------------------------------------------- training

def batches(n, size, shuffle=True):
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    return idx.split(size)


def predict(model, x, batch=512):
    """Model outputs on x (no grad, eval mode), concatenated over batches."""
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i]) for i in batches(len(x), batch, shuffle=False)])


def argmax_range(model, x, quantile=0.999, headroom=1.25):
    """Comparison range for an encrypted ArgMax after model: the largest
    logit magnitude of all but the top 0.1% of inputs, with 25% headroom.
    The excluded inputs are ones whose polynomial activations left their
    intervals; their logits run to thousands and are already wrong, and a
    range that covered them would cost the comparisons of every other input
    their precision (FHEW compares (a - b) / range to a fixed resolution)."""
    logits = predict(model, x).abs().reshape(len(x), -1).amax(1)
    bound = headroom * float(torch.quantile(logits.double(), quantile))
    print(f"argmax range {bound:.1f} (largest training logit: median {logits.median():.1f}, "
          f"{quantile:.1%} below {bound / headroom:.1f}, max {logits.max():.1f})")
    return bound


def train(model, xtr, ytr, xte, yte, epochs, loss_fn=None, metric=None, lr=2e-3, batch=128,
          range_weight=RANGE_WEIGHT, clip=1.0, weight_decay=0.0, augment=None, augment_pairs=None, log=print):
    """Adam with a one-cycle schedule. loss_fn(outputs, targets) defaults to
    cross-entropy; metric(outputs, targets) -> float is printed after every
    epoch on the test set. fheml.nn.range_loss keeps polynomial arguments
    (layer norm variances, softmax scores and sums) inside the intervals the
    encrypted series cover; gradient clipping tames polynomial networks
    early on. augment(x) -> x is applied to each training batch;
    augment_pairs(x, y) -> (x, y) instead transforms inputs and targets
    together (flipping an image and its segmentation mask). Returns
    (final metric, training seconds)."""
    loss_fn = loss_fn or nn.functional.cross_entropy
    metric = metric or accuracy
    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    steps = epochs * ((len(xtr) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    t0, score = time.time(), float("nan")
    for epoch in range(epochs):
        model.train()
        t = time.time()
        for idx in batches(len(xtr), batch):
            xb, yb = xtr[idx], ytr[idx]
            if augment is not None:
                xb = augment(xb)
            if augment_pairs is not None:
                xb, yb = augment_pairs(xb, yb)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            (loss + range_weight * hnn.range_loss(model)).backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            sched.step()
        score = metric(predict(model, xte), yte)
        log(f"epoch {epoch + 1}: loss {loss.item():.3f}, test {_metric_name(metric)} {score:.4f} ({time.time() - t:.0f}s)")
    return score, time.time() - t0


def _metric_name(metric):
    return getattr(metric, "__name__", "metric").replace("_", " ")


# ------------------------------------------------------------------ metrics

def accuracy(outputs, targets):
    """Top-1 accuracy of logits (N, K) against labels (N,)."""
    return float((torch.as_tensor(outputs).argmax(-1) == torch.as_tensor(targets)).float().mean())


def confusion(pred, target, classes, ignore=None):
    """(classes, classes) confusion counts of integer label maps (any shape)."""
    pred, target = np.asarray(pred).ravel(), np.asarray(target).ravel()
    keep = target != ignore if ignore is not None else slice(None)
    return np.bincount(target[keep] * classes + pred[keep], minlength=classes * classes).reshape(classes, classes)


def mean_iou(pred, target, classes, ignore=None):
    """Mean intersection over union over the classes present (segmentation),
    and the per-class IoUs."""
    c = confusion(pred, target, classes, ignore).astype(np.float64)
    inter, union = np.diag(c), c.sum(0) + c.sum(1) - np.diag(c)
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou)), iou


def pixel_accuracy(pred, target, ignore=None):
    pred, target = np.asarray(pred).ravel(), np.asarray(target).ravel()
    keep = target != ignore if ignore is not None else np.ones_like(target, bool)
    return float((pred[keep] == target[keep]).mean())


def box_iou(a, b):
    """IoU of boxes (..., 4) as (x0, y0, x1, y1), broadcast: a (N, 1, 4) vs b (1, M, 4) -> (N, M)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    lt, rb = np.maximum(a[..., :2], b[..., :2]), np.minimum(a[..., 2:], b[..., 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), -1)
    area = lambda x: np.prod(np.clip(x[..., 2:] - x[..., :2], 0, None), -1)  # noqa: E731
    return inter / np.maximum(area(a) + area(b) - inter, 1e-12)


def average_precision(dets, gts, iou=0.5):
    """VOC/COCO-style AP of one class at one IoU threshold.

    dets: list over images of (K, 5) arrays (x0, y0, x1, y1, score); gts:
    list over images of (M, 4) boxes. Greedy matching by score, all-point
    interpolated precision (COCO uses 101 recall points; the difference is
    small). Returns (AP, number of ground-truth boxes)."""
    scores, tp = [], []
    total = sum(len(g) for g in gts)
    for d, g in zip(dets, gts):
        d = np.asarray(d, np.float64).reshape(-1, 5)
        g = np.asarray(g, np.float64).reshape(-1, 4)
        used = np.zeros(len(g), bool)
        for k in np.argsort(-d[:, 4]):
            scores.append(d[k, 4])
            if len(g):
                o = box_iou(d[k, :4], g)
                o[used] = -1
                j = int(o.argmax())
                if o[j] >= iou:
                    used[j] = True
                    tp.append(1)
                    continue
            tp.append(0)
    if total == 0:
        return float("nan"), 0
    order = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.asarray(tp, np.float64)[order]
    ctp = np.cumsum(tp)
    recall = ctp / total
    precision = ctp / np.arange(1, len(tp) + 1)
    # precision envelope, integrated over recall
    prec = np.concatenate([[0.0], precision, [0.0]])
    rec = np.concatenate([[0.0], recall, [recall[-1] if len(recall) else 0.0]])
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:])), total


def sequence_accuracy(pred, target, pad=0):
    """(token accuracy, whole-sequence accuracy) of integer sequences (N, T),
    ignoring positions where the target is pad."""
    pred, target = np.asarray(pred), np.asarray(target)
    keep = target != pad
    tok = float((pred[keep] == target[keep]).mean())
    seq = float(np.mean([np.all(p[k] == t[k]) for p, t, k in zip(pred, target, keep)]))
    return tok, seq


# ------------------------------------------------------------- fheml side

def plain_score(net, x, y, metric=None, n=2000, outputs=False):
    """The metric of the exported network as encrypted inference computes it
    (same polynomial approximations, no clamping), in the clear, on the first
    n test inputs. Returns (score, n) or (score, n, outputs)."""
    metric = metric or accuracy
    n = min(len(x), n)
    xs = np.asarray(x[:n], np.float64)
    out = np.stack([np.asarray(fheml.plain(net, xs[i])) for i in range(n)])
    score = metric(torch.as_tensor(out), y[:n])
    return (score, n, out) if outputs else (score, n)


def check_ranges(net, xs):
    """Largest |input| of every Chebyshev / exact-relu / max-pool layer over
    xs, against its interval (the series diverge outside it; the comparison
    wraps around)."""
    xs = [fheml._prepare(net, x) for x in xs]  # token inputs padded to the token stride
    for i, layer in enumerate(net.layers):
        if layer.type in ("chebyshev", "exactrelu", "maxpool2d"):
            peak = max(np.abs(x).max() for x in xs)
            print(f"  layer {i} {layer.describe()}: max |input| {peak:.2f} over {len(xs)} inputs")
        xs = [np.asarray(layer.forward_plain(x)) for x in xs]


def check_softmax_ranges(model, x):
    """Per-token variance at each LayerNorm, attention scores and row sums,
    against the intervals their series are built for."""
    stats = []

    def ln_hook(m, inp, _):
        v = inp[0].var(-1, unbiased=False)
        stats.append(f"layernorm: variance [{v.min():.3f}, {v.max():.3f}] vs [{m.var_min}, {m.var_max}]")

    def att_hook(m, inp, _):
        if m.softmax is None:
            return
        z = inp[0]
        b, n, d = z.shape
        q, k = (f(z).reshape(b, n, m.heads, d // m.heads).transpose(1, 2) for f in (m.q, m.k))
        s = q @ k.transpose(-1, -2)
        e = torch.exp(s.clamp(m.exp_lo, m.exp_hi))
        if m.mask is not None:
            e = e * m.mask
            live = m.mask.sum(-1) > 0
            tot = e.sum(-1)[..., live]
        else:
            tot = e.sum(-1)
        stats.append(f"attention: scores [{s.min():.2f}, {s.max():.2f}] vs [{m.exp_lo}, {m.exp_hi}], row sums "
                     f"[{tot.min():.2f}, {tot.max():.2f}] vs [{m.sum_lo:.2f}, {m.sum_hi:.2f}]")

    hooks = [m.register_forward_hook(ln_hook) for m in model.modules() if isinstance(m, hnn.LayerNorm)]
    hooks += [m.register_forward_hook(att_hook) for m in model.modules() if isinstance(m, hnn.Attention)]
    model.eval()
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    for line in stats:
        print("  " + line)


def prepare(net, args, calib, packing=None):
    """Packing, then (with --bootstrap) folded intervals and automatic
    bootstraps scaled from the calibration inputs, then compile. Returns the
    bootstrap level budget for context()."""
    level_budget = [3, 3]
    if args.packing or packing:
        net.packing = args.packing or packing
    calib = np.asarray(calib, np.float64)
    if args.bootstrap:
        # Levels per stretch at ring 2^16 and 50 bits: 5 with {2,2}, 3 with {3,3}, two more with --digits 4.
        level_budget = [args.level_budget, args.level_budget]
        net.compile()
        check_ranges(net, calib)
        depth = net.depth
        boots, budget = fheml.auto_bootstrap(net, calib, level_budget=level_budget, max_ring=1 << 16,
                                             key_switch_digits=args.digits,
                                             bootstrap_iterations=args.boot_iterations)
        print(f"auto_bootstrap: depth {depth} -> {net.depth}, {boots} bootstraps, at most {budget} levels between them")
    net.compile()
    print(net.summary())
    return level_budget


def context(net, args, level_budget):
    """The CKKS context for the compiled network: 128-bit unless --insecure.
    Returns (context, setup seconds)."""
    t = time.time()
    overrides = {"level_budget": level_budget, "key_switch_digits": args.digits,
                 "bootstrap_iterations": args.boot_iterations, "bootstrap_precision": args.boot_precision}
    if args.scale_bits:
        overrides["scale_mod_bits"] = args.scale_bits
        overrides["first_mod_bits"] = min(60, args.scale_bits + 7)
    if args.ring or (args.insecure and net.uses_bootstrapping):
        overrides["ring_dim"] = args.ring or max(4096, 2 * net.slots)  # toy ring: the smallest that bootstraps
    ctx = fheml.make_context(net, secure=not args.insecure, rotation_window=args.window, **overrides)
    setup = time.time() - t
    print(ctx.describe(), f"(setup {setup:.1f}s)")
    return ctx, setup


def encrypted_run(net, ctx, model, x, args, describe=None):
    """Runs the first args.n inputs of x encrypted and compares every output
    with the torch model's (float64). describe(i, enc, ref) -> str adds a
    task-specific note per input (default: the argmax of both). Returns a
    dict of per-input seconds, max |enc - torch|, how many argmaxes agree and
    the decrypted outputs."""
    model = model.double().eval()
    xs = torch.as_tensor(np.asarray(x[:args.n]), dtype=torch.float64)
    times, errs, agree, outs = [], [], 0, []
    for i in range(len(xs)):
        xi = xs[i].numpy()
        t = time.time()
        if args.profile and i == 0:
            layer_times = []
            y = net.forward(ctx, fheml.encrypt(net, ctx, xi),
                            hook=lambda k, what, c, secs: layer_times.append((k, what, secs)))
            enc = fheml.decrypt(net, ctx, y)
            for k, what, secs in layer_times:
                print(f"  layer {k} ({secs:.1f}s): {what[:110]}")
        else:
            enc = fheml.infer(net, ctx, xi)
        dt = time.time() - t
        with torch.no_grad():
            ref = model(xs[i:i + 1]).numpy().ravel()
        times.append(dt)
        errs.append(float(np.abs(enc - ref).max()))
        agree += int(enc.argmax() == ref.argmax())
        outs.append(enc)
        note = describe(i, enc, ref) if describe else f"encrypted -> {enc.argmax()}, torch -> {ref.argmax()}"
        print(f"input {i}: {note}, max |enc - torch| {errs[-1]:.1e}, {dt:.1f}s")
    return dict(image_seconds=times, max_abs_err=errs, argmax_agree=agree, images=len(xs), outputs=outs)


def record(args, net, ctx, setup, run, **fields):
    """Appends one JSON line with the run's numbers to args.json (if set)."""
    if not args.json:
        return
    rec = dict(secure=not args.insecure, bootstrap=args.bootstrap, depth=net.depth,
               uses_bootstrapping=net.uses_bootstrapping, internal_bootstraps=net.internal_bootstraps,
               bootstrap_layers=sum(1 for layer in net.layers if layer.type == "bootstrap"),
               context=ctx.describe(), setup_seconds=setup,
               peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    rec.update({k: v for k, v in run.items() if k != "outputs"})
    rec.update(fields)
    with open(args.json, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
