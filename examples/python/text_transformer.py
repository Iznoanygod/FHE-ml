"""Encoder-only transformers for text classification, trained in PyTorch,
exported to fheml and run encrypted: a basic single-head model and its
multi-head counterpart, on AG News (4 news topics) or SST-2 (sentiment).

    PYTHONPATH=build/python python3 examples/python/text_transformer.py --task agnews --heads 1 --bootstrap --insecure
    PYTHONPATH=build/python python3 examples/python/text_transformer.py --task agnews --heads 4 --bootstrap --insecure
    PYTHONPATH=build/python python3 examples/python/text_transformer.py --task sst2 --heads 4 --bootstrap --insecure

The client tokenizes its text, looks the words up in the embedding table
(released with the model, like the tokenizer) and encrypts the (tokens, dim)
matrix of word vectors. The server runs everything else homomorphically:

    token Linear dim -> dim (+ learned positional embedding)
    L x pre-norm encoder block:
        residual { LayerNorm, softmax self-attention with H heads }
        residual { LayerNorm, Linear dim -> 2 dim, 0.5x + 0.25x^2, Linear 2 dim -> dim }
    LayerNorm, mean over tokens, Linear dim -> classes

--heads 1 is the basic transformer: one attention head of width dim. --heads
4 splits the same dim into four heads of width dim / 4 (the parameter count
does not change); under encryption the heads cost fewer rotations than one
wide head. --attention quadratic swaps softmax for the polynomial score
A = c0 + c1 S + c2 S^2 (depth 6 instead of ~23), --no-norm drops the
LayerNorms. Softmax and LayerNorm are tens of levels deep, so --bootstrap is
needed: fheml.auto_bootstrap places bootstraps between layers and the layers
bootstrap internally (see Network.place_bootstraps).
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
from datasets import AGNEWS_CLASSES, SST2_CLASSES, load_agnews, load_sst2  # noqa: E402

TASKS = {
    # tokens, vocabulary, classes, default epochs
    "agnews": dict(load=load_agnews, tokens=16, vocab=20000, classes=AGNEWS_CLASSES, epochs=4),
    "sst2": dict(load=load_sst2, tokens=32, vocab=12000, classes=SST2_CLASSES, epochs=8),
}
# Series intervals (see train_torch_mnist.py): the training forward clamps to
# them, range_loss keeps activations inside, and check_softmax_ranges()
# reports how the test data fits before encrypting.
SOFTMAX = dict(lo=-4.0, hi=4.0, exp_degree=7, inv_degree=7, newton=2, sum_lo=0.5, sum_hi=100.0)
LN = dict(eps=0.05, var_min=0.0, var_max=8.0, degree=7, newton=1)
# The residual stream grows through the blocks: the final norm sees larger variances.
LN_FINAL = dict(LN, var_max=16.0)


class Embed(nn.Module):
    """Client side: word ids -> word vectors (the encrypted input)."""

    def __init__(self, vocab, dim):
        super().__init__()
        self.table = nn.Embedding(vocab, dim)
        nn.init.normal_(self.table.weight, std=0.5)

    def forward(self, ids):
        return self.table(ids)


def encoder(tokens, dim, classes, heads, layers=1, attention="softmax", norm=True):
    """The server side, on (batch, tokens, dim) word vectors."""
    ln = (lambda o=LN: [hnn.LayerNorm(dim, **o)]) if norm else (lambda o=None: [])  # noqa: E731
    att = dict(softmax=SOFTMAX) if attention == "softmax" else {}
    blocks = []
    for _ in range(layers):
        blocks += [hnn.Residual(*ln(), hnn.Attention(dim, tokens, heads=heads, **att)),
                   hnn.Residual(*ln(), nn.Linear(dim, 2 * dim), hnn.Poly([0, 0.5, 0.25]), nn.Linear(2 * dim, dim))]
    return nn.Sequential(nn.Linear(dim, dim), hnn.PosEmbedding(tokens, dim), *blocks, *ln(LN_FINAL), hnn.TokenMean(),
                         nn.Linear(dim, len(classes)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=sorted(TASKS), default="agnews")
    ap.add_argument("--heads", type=int, default=1, help="attention heads (1 = basic transformer)")
    ap.add_argument("--dim", type=int, default=32, help="model width (power of two)")
    ap.add_argument("--layers", type=int, default=1, help="encoder blocks")
    ap.add_argument("--attention", choices=("softmax", "quadratic"), default="softmax")
    ap.add_argument("--no-norm", dest="norm", action="store_false", help="no LayerNorm")
    ap.add_argument("--train-size", type=int, help="use only this many training texts")
    fhe_eval.add_args(ap)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    task = TASKS[args.task]
    data = task["load"](args.data, task["tokens"], task["vocab"])
    xtr, ytr = torch.tensor(data["train_x"]), torch.tensor(data["train_y"])
    xte, yte = torch.tensor(data["test_x"]), torch.tensor(data["test_y"])
    if args.train_size:
        xtr, ytr = xtr[:args.train_size], ytr[:args.train_size]
    classes, tokens = task["classes"], task["tokens"]

    embed = Embed(len(data["vocab"]), args.dim)
    model = encoder(tokens, args.dim, classes, args.heads, args.layers, args.attention, args.norm)
    full = nn.Sequential(embed, model)
    acc, train_time = fhe_eval.train(full, xtr, ytr, xte, yte, args.epochs or task["epochs"], weight_decay=1e-5)

    # From here on the encrypted input is the word vectors.
    with torch.no_grad():
        vte = embed(xte).double().numpy()
        vcal = embed(xtr[:args.calibration]).double().numpy()
    model = model.float()
    fhe_eval.check_softmax_ranges(model, torch.tensor(vte[:2000], dtype=torch.float32))
    net = fheml.from_torch(model, (tokens, args.dim))  # checks fheml plaintext == torch
    if args.out:
        net.save(args.out)
        np.save(os.path.splitext(args.out)[0] + "_embedding.npy", embed.table.weight.detach().numpy())
        print("wrote", args.out)
    net.fuse_linear()  # token mean + head -> one matrix, saves a level
    plain_acc, n_plain = fhe_eval.plain_score(net, vte, yte)
    torch_sub = fhe_eval.accuracy(fhe_eval.predict(model, torch.tensor(vte[:n_plain], dtype=torch.float32)),
                                  yte[:n_plain])
    print(f"fheml plaintext accuracy {plain_acc:.1%} on {n_plain} test texts (torch: {torch_sub:.1%} on them, "
          f"{acc:.1%} on all {len(yte)})")

    level_budget = fhe_eval.prepare(net, args, vcal)
    ctx, setup = fhe_eval.context(net, args, level_budget)
    vocab = data["vocab"]

    def describe(i, enc, ref):
        words = " ".join(w for w in vocab[data["test_x"][i]] if w != "<pad>")
        return (f'"{words[:60]}" label {classes[int(yte[i])]}, encrypted -> {classes[enc.argmax()]}, '
                f"torch -> {classes[ref.argmax()]}")

    run = fhe_eval.encrypted_run(net, ctx, model, vte, args, describe=describe)
    fhe_eval.record(args, net, ctx, setup, run, task=args.task, heads=args.heads, dim=args.dim, layers=args.layers,
                    attention=args.attention, norm=args.norm, tokens=tokens, torch_accuracy=acc,
                    plain_accuracy=plain_acc, torch_accuracy_plain_texts=torch_sub, plain_texts=n_plain,
                    train_seconds=train_time)


if __name__ == "__main__":
    main()
