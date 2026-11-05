"""MNIST baselines: a fully connected feedforward network (MLP) next to a
convolutional network (CNN), each trained in PyTorch, exported to fheml and
run encrypted, with one comparison table at the end.

    PYTHONPATH=build/python python3 examples/python/mnist_mlp_cnn.py                    # all three, 128-bit
    PYTHONPATH=build/python python3 examples/python/mnist_mlp_cnn.py --arch mlp --n 10
    PYTHONPATH=build/python python3 examples/python/mnist_mlp_cnn.py --insecure --json runs.jsonl

The models are train_torch_mnist.py's: mlp (the 784-200-10 sigmoid network
of the original 2022 prototype, fully encrypted with a Chebyshev sigmoid),
mlp-poly (the same with a quadratic activation) and cnn (the CryptoNets/LoLa
CNN).

Every model runs on the full 28x28 image with no bootstrapping; at 128-bit
security mlp and cnn need ring 2^15 and mlp-poly fits ring 2^14. The table reports test
accuracy in torch, accuracy of the exported network as encrypted inference
computes it (in the clear, on 2000 test images), the multiplicative depth,
the context, seconds per encrypted image and the largest gap between an
encrypted output and the torch output.
"""
import argparse
import os
import sys

import torch

import fheml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
import fhe_eval  # noqa: E402
from datasets import load_mnist  # noqa: E402
from train_torch_mnist import MLP_LR, cnn, mlp  # noqa: E402

# builder, epochs, learning rate
MODELS = {"mlp": (mlp, 5, MLP_LR), "mlp-poly": (lambda: mlp("poly"), 5, 2e-3), "cnn": (cnn, 3, 2e-3)}


def run(arch, args, data):
    print(f"\n=== {arch} ===")
    torch.manual_seed(args.seed)
    xtr, ytr, xte, yte = data
    build, epochs, lr = MODELS[arch]
    model = build()
    acc, train_time = fhe_eval.train(model, xtr, ytr, xte, yte, args.epochs or epochs, lr=lr)
    params = sum(p.numel() for p in model.parameters())

    net = fheml.from_torch(model, (1, 28, 28))  # checks fheml plaintext == torch
    if args.out:
        path = f"{args.out}.{arch}.fheml"
        net.save(path)
        print("wrote", path)
    net.compile()
    fhe_eval.check_ranges(net, xtr[:args.calibration].double().numpy())
    plain_acc, n_plain = fhe_eval.plain_score(net, xte.double().numpy(), yte)
    torch_head = fhe_eval.accuracy(fhe_eval.predict(model, xte[:n_plain]), yte[:n_plain])
    print(f"fheml plaintext accuracy {plain_acc:.1%} on {n_plain} test images (torch: {torch_head:.1%} on them, "
          f"{acc:.1%} on all)")

    level_budget = fhe_eval.prepare(net, args, xtr[:args.calibration].double().numpy())
    ctx, setup = fhe_eval.context(net, args, level_budget)
    out = fhe_eval.encrypted_run(
        net, ctx, model, xte.double().numpy(), args,
        describe=lambda i, enc, ref: f"label {int(yte[i])}, encrypted -> {enc.argmax()}, torch -> {ref.argmax()}")
    fhe_eval.record(args, net, ctx, setup, out, arch=arch, params=params, torch_accuracy=acc,
                    plain_accuracy=plain_acc, torch_accuracy_plain_images=torch_head, plain_images=n_plain,
                    train_seconds=train_time)
    times = out["image_seconds"]
    return dict(arch=arch, params=params, torch=acc, plain=plain_acc, depth=net.depth, context=ctx.describe(),
                setup=setup, seconds=sum(times[1:] or times) / len(times[1:] or times),
                err=max(out["max_abs_err"]), agree=f"{out['argmax_agree']}/{out['images']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=(*MODELS, "all"), default="all")
    fhe_eval.add_args(ap)
    args = ap.parse_args()

    raw = load_mnist(args.data)
    data = (torch.tensor(raw["train_x"], dtype=torch.float32), torch.tensor(raw["train_y"]).long(),
            torch.tensor(raw["test_x"], dtype=torch.float32), torch.tensor(raw["test_y"]).long())
    archs = list(MODELS) if args.arch == "all" else [args.arch]
    rows = [run(a, args, data) for a in archs]

    print(f"\nMNIST, {'insecure toy parameters' if args.insecure else '128-bit security'}, "
          f"{args.n} encrypted test images per model (time excludes the first image when n > 1);\n"
          "torch = accuracy on all 10k test images, fheml = the encrypted computation in the clear on the first 2000")
    print(f"{'model':<9} {'params':>7} {'torch':>7} {'fheml':>7} {'depth':>5} {'s/image':>8} {'max err':>8} "
          f"{'agree':>6}  context")
    for r in rows:
        print(f"{r['arch']:<9} {r['params']:>7} {r['torch']:>7.2%} {r['plain']:>7.2%} {r['depth']:>5} "
              f"{r['seconds']:>8.2f} {r['err']:>8.1e} {r['agree']:>6}  {r['context']}")


if __name__ == "__main__":
    main()
