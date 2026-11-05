"""Unit tests of the example harness metrics (tools/fhe_eval.py), no data needed.

    PYTHONPATH=build/python python3 tests/test_eval.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import fhe_eval as E  # noqa: E402
import datasets as D  # noqa: E402

# accuracy
assert np.isclose(E.accuracy(np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]), np.array([1, 0, 0])), 2 / 3)

# segmentation: IoU of class 0 = 1/3, class 1 = 2/4
pred = np.array([[0, 0], [1, 1], [1, 1]])
target = np.array([[0, 1], [0, 1], [1, 1]])
miou, per = E.mean_iou(pred, target, 2)
assert np.allclose(per, [1 / 3, 3 / 5]) and np.isclose(miou, (1 / 3 + 3 / 5) / 2), per
assert E.pixel_accuracy(pred, target) == 4 / 6
assert E.mean_iou(pred, target, 3)[1][2] != E.mean_iou(pred, target, 3)[1][2]  # absent class -> NaN

# boxes
assert np.isclose(E.box_iou([0, 0, 2, 2], [1, 1, 3, 3]), 1 / 7)
assert E.box_iou(np.zeros((3, 1, 4)) + [0, 0, 1, 1], np.zeros((1, 5, 4)) + [0, 0, 1, 1]).shape == (3, 5)

# average precision: one perfect detection, one false positive with lower
# score, one missed box -> recall 1/2 at precision 1, AP 0.5
dets = [np.array([[0, 0, 1, 1, 0.9], [5, 5, 6, 6, 0.1]]), np.zeros((0, 5))]
gts = [np.array([[0, 0, 1, 1]]), np.array([[2, 2, 3, 3]])]
ap, n = E.average_precision(dets, gts)
assert n == 2 and np.isclose(ap, 0.5), ap
# a duplicate detection of the same box counts as a false positive
ap2, _ = E.average_precision([np.array([[0, 0, 1, 1, 0.9], [0, 0, 1, 1, 0.8]])], [np.array([[0, 0, 1, 1]])])
assert np.isclose(ap2, 1.0), ap2
ap3, _ = E.average_precision([np.array([[0, 0, 1, 1, 0.5], [3, 3, 4, 4, 0.9]])], [np.array([[0, 0, 1, 1]])])
assert np.isclose(ap3, 0.5), ap3

# sequences: pad (0) positions are ignored
tok, seq = E.sequence_accuracy(np.array([[3, 4, 2, 9], [3, 5, 2, 0]]), np.array([[3, 4, 2, 0], [3, 4, 2, 0]]))
assert np.isclose(tok, 5 / 6) and seq == 0.5, (tok, seq)

# text: tokenizer and the shared vocabulary / padding of the text loaders
words = ["wall", "st", ".", "bears", "'", "it's", "3", ",", "000", "fine", "!"]
assert D.tokenize("Wall St. Bears' it's 3,000\\fine!") == words
enc = D._encode_text([(["a", "b", "a"], 1), (["b", "c"], 0)], [(["a", "z", "b", "c", "a"], 2)], 3, 4)
assert list(enc["vocab"]) == ["<pad>", "<unk>", "a", "b"]
assert enc["train_x"].tolist() == [[2, 3, 2], [3, 1, 0]] and enc["test_x"].tolist() == [[2, 1, 3]]
assert enc["train_len"].tolist() == [3, 2] and enc["test_y"].tolist() == [2]

# check_ranges pads token inputs to the network's token stride like fheml.plain
import torch  # noqa: E402
from torch import nn  # noqa: E402

import fheml  # noqa: E402
from fheml import nn as hnn  # noqa: E402

torch.manual_seed(0)
tok = nn.Sequential(nn.Linear(6, 8), hnn.Chebyshev("relu", -4, 4, 7), nn.Linear(8, 16), hnn.TokenMean(),
                    nn.Linear(16, 2))
net = fheml.from_torch(tok, (4, 6))
net.compile()
E.check_ranges(net, np.random.default_rng(0).normal(size=(2, 4, 6)))
print("PASSED")
