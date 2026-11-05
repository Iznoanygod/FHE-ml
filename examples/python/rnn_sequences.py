"""Recurrent networks (Elman RNN, GRU, LSTM) on sequence tasks, trained in
PyTorch, exported to fheml and run encrypted end to end:

    smnist  sequential MNIST: the image is read row by row, 28 steps of 28 pixels
    sst2    SST-2 sentence sentiment, 32 words
    agnews  AG News topic classification, 16 words

    PYTHONPATH=build/python python3 examples/python/rnn_sequences.py --task smnist --cell lstm --bootstrap --insecure
    PYTHONPATH=build/python python3 examples/python/rnn_sequences.py --task sst2 --cell gru --bootstrap --insecure
    PYTHONPATH=build/python python3 examples/python/rnn_sequences.py --task agnews --cell rnn --bootstrap

The model is one recurrent layer with the real activations of its cell
(sigmoid gates, tanh, as Chebyshev series on calibrated intervals; see
fheml.nn.RNN / GRU / LSTM) followed by a linear classifier on the last
hidden state:

    smnist: rows (28 x 28)       -> RNN/GRU/LSTM(28 -> hidden), last state -> Linear(hidden, 10)
    text:   word vectors (n x d) -> RNN/GRU/LSTM(d -> hidden), last state -> Linear(hidden, classes)

For text the client tokenizes and looks the words up in the embedding table
(released with the model, like the tokenizer) and encrypts the word
vectors; for MNIST it encrypts the pixels. Everything else runs
homomorphically: the recurrence over all time steps, its sigmoids and tanhs,
and the classifier. The client only decrypts the logits.

The recurrence is n time steps deep (5 levels per step for the RNN, 12 for
GRU and LSTM), so --bootstrap is needed: the recurrent layer bootstraps its
own state whenever the next step would not fit (fheml.Recurrent).
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
from datasets import AGNEWS_CLASSES, SST2_CLASSES, load_agnews, load_mnist, load_sst2  # noqa: E402

RANGE_WEIGHT = 20.0
CELLS = {"rnn": hnn.RNN, "gru": hnn.GRU, "lstm": hnn.LSTM}
TASKS = {
    # tokens, (vocabulary), classes, default epochs
    "smnist": dict(tokens=28, classes=[str(d) for d in range(10)], epochs=5),
    "sst2": dict(load=load_sst2, tokens=32, vocab=12000, classes=SST2_CLASSES, epochs=8),
    "agnews": dict(load=load_agnews, tokens=16, vocab=20000, classes=AGNEWS_CLASSES, epochs=4),
}


class Embed(nn.Module):
    """Client side: word ids -> word vectors (the encrypted input)."""

    def __init__(self, vocab, dim):
        super().__init__()
        self.table = nn.Embedding(vocab, dim)
        nn.init.normal_(self.table.weight, std=0.5)

    def forward(self, ids):
        return self.table(ids)


def recurrent(cell, inputs, hidden, classes):
    """The server side, on (batch, tokens, inputs)."""
    return nn.Sequential(CELLS[cell](inputs, hidden, last=True), nn.Linear(hidden, len(classes)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=sorted(TASKS), default="smnist")
    ap.add_argument("--cell", choices=sorted(CELLS), default="lstm")
    ap.add_argument("--hidden", type=int, default=32, help="hidden state size")
    ap.add_argument("--dim", type=int, default=32, help="word vector size (text tasks)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--scale-bits", type=int, help="CKKS scaling bits (default 59 for bootstrapped toy runs)")
    ap.add_argument("--threads", type=int, help="torch threads for training")
    ap.add_argument("--train-size", type=int, help="use only this many training sequences")
    fhe_eval.add_args(ap)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    task = TASKS[args.task]
    classes, tokens = task["classes"], task["tokens"]
    if args.task == "smnist":
        data = load_mnist(args.data)
        xtr = torch.tensor(data["train_x"][:, 0], dtype=torch.float32)  # (N, 28 rows, 28 pixels)
        xte = torch.tensor(data["test_x"][:, 0], dtype=torch.float32)
        ytr, yte = torch.tensor(data["train_y"], dtype=torch.long), torch.tensor(data["test_y"], dtype=torch.long)
        inputs = 28
    else:
        data = task["load"](args.data, tokens, task["vocab"])
        xtr, ytr = torch.tensor(data["train_x"]), torch.tensor(data["train_y"])
        xte, yte = torch.tensor(data["test_x"]), torch.tensor(data["test_y"])
        inputs = args.dim
    if args.train_size:
        xtr, ytr = xtr[:args.train_size], ytr[:args.train_size]

    model = recurrent(args.cell, inputs, args.hidden, classes)
    embed = Embed(len(data["vocab"]), args.dim) if args.task != "smnist" else nn.Identity()
    full = nn.Sequential(embed, model)
    # A strong range penalty: the series run on every step and diverge
    # outside their intervals, and encrypted inference cannot clamp.
    acc, train_time = fhe_eval.train(full, xtr, ytr, xte, yte, args.epochs or task["epochs"], lr=args.lr,
                                     weight_decay=1e-5, range_weight=RANGE_WEIGHT)

    # From here on the encrypted input is the pixel rows or the word vectors.
    with torch.no_grad():
        vte = embed(xte).double().numpy()
        vcal = embed(xtr[:args.calibration]).double().numpy()
    net = fheml.from_torch(model.float(), (tokens, inputs))  # checks fheml plaintext == torch
    if args.out:
        net.save(args.out)
        if args.task != "smnist":
            np.save(os.path.splitext(args.out)[0] + "_embedding.npy", embed.table.weight.detach().numpy())
        print("wrote", args.out)
    print("series ranges on the test set (encrypted steps included; must stay <= 1):")
    fhe_eval.check_ranges(net, vte[:200])
    plain_acc, n_plain = fhe_eval.plain_score(net, vte, yte, n=1000)
    torch_sub = fhe_eval.accuracy(fhe_eval.predict(model, torch.tensor(vte[:n_plain], dtype=torch.float32)),
                                  yte[:n_plain])
    print(f"fheml plaintext accuracy {plain_acc:.1%} on {n_plain} test sequences (torch: {torch_sub:.1%} on them, "
          f"{acc:.1%} on all {len(yte)})")

    if args.bootstrap and args.budget is None and args.insecure:
        # Toy rings are not bound by the 128-bit modulus limit: one bootstrap
        # per two RNN steps (10 levels) or per GRU/LSTM step (12) instead of
        # one or three per step at 5 levels.
        args.budget = 10 if args.cell == "rnn" else 12
    level_budget = fhe_eval.prepare(net, args, vcal)
    if args.bootstrap and args.insecure and not args.scale_bits:
        # At 4096 slots a 50-bit scale leaves bootstrapping only a few bits of
        # precision; 59 bits keeps about 12 (toy rings have room for it).
        args.scale_bits = 59
    ctx, setup = fhe_eval.context(net, args, level_budget)

    def describe(i, enc, ref):
        what = f"digit {int(yte[i])}" if args.task == "smnist" else \
            '"' + " ".join(w for w in data["vocab"][data["test_x"][i]] if w != "<pad>")[:50] + f'" label {classes[int(yte[i])]}'
        return f"{what}, encrypted -> {classes[enc.argmax()]}, torch -> {classes[ref.argmax()]}"

    run = fhe_eval.encrypted_run(net, ctx, model, vte, args, describe=describe)
    fhe_eval.record(args, net, ctx, setup, run, task=args.task, cell=args.cell, hidden=args.hidden, tokens=tokens,
                    torch_accuracy=acc, plain_accuracy=plain_acc, torch_accuracy_plain_texts=torch_sub,
                    plain_texts=n_plain, train_seconds=train_time)


if __name__ == "__main__":
    main()
