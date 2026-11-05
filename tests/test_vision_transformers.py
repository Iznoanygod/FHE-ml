"""Checks of examples/python/vision_transformers.py that need no dataset:
patch layout, the per-patch decoder's pixel order, paired flips, and that
both tasks export (single- and multi-head, both blocks, widened series,
argmax) with fheml plaintext == torch.

    PYTHONPATH=build/python python3 tests/test_vision_transformers.py
"""
import os
import sys

import numpy as np
import torch

import fheml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "python"))
import vision_transformers as vt  # noqa: E402

rng = np.random.default_rng(0)

# patchify: token t = (row, col) of the 8x8 grid holds that 4x4 patch, (channel, y, x) order
img = rng.normal(size=(2, 3, 32, 32))
tok = vt.patchify(img)
assert tok.shape == (2, 64, 48)
r, c = 3, 5
assert np.array_equal(tok[1, r * 8 + c].reshape(3, 4, 4), img[1, :, 4 * r:4 * r + 4, 4 * c:4 * c + 4])

# the decoder's pixel order: logits that put the class of each pixel at its own
# feature index come back as the right pixel map
labels = rng.integers(0, 3, size=(2, 32, 32))
onehot = np.eye(3)[labels]                                           # (2, 32, 32, 3)
out = onehot.reshape(2, 8, 4, 8, 4, 3).transpose(0, 1, 3, 2, 4, 5).reshape(2, 64, 48)
assert np.array_equal(vt.seg_pred(torch.as_tensor(out)).numpy(), labels)

# flips move images and masks together
x = torch.as_tensor(vt.patchify(img), dtype=torch.float32)
y = torch.as_tensor(labels)
torch.manual_seed(1)
xa, ya = vt.pets_augment_pairs(x, y)
for i in range(2):
    flipped = not torch.equal(ya[i], y[i])
    want = vt.patchify(img[i:i + 1, :, :, ::-1].copy() if flipped else img[i:i + 1])[0]
    assert np.allclose(xa[i].numpy(), want, atol=1e-6)

# both tasks, one and four heads, both block kinds export and match torch (from_torch checks)
for task in ("cifar", "pets"):
    for heads in (1, 4):
        for kind in ("vit", "basic"):
            torch.manual_seed(0)
            m = vt.model(task, 16, 32, heads, 1, kind)
            if kind == "vit":
                vt.widen(m)  # series rebuilt on wider intervals still export exactly
            m = torch.nn.Sequential(*m, fheml.nn.ArgMax(8.0, classes=3 if task == "pets" else None)).double().eval()
            net = fheml.from_torch(m, (vt.TOKENS, vt.FEATURES))
            want = 10 if task == "cifar" else 64 * 48
            got = fheml.plain(net, rng.uniform(-1, 1, (vt.TOKENS, vt.FEATURES)))
            assert got.size == want, (task, heads, kind, got.size)
            groups = got.reshape(-1, 10 if task == "cifar" else 3)
            assert np.all(groups.sum(1) == 1) and set(np.unique(groups)) <= {0.0, 1.0}, "one-hot per group"
print("vision transformer example checks passed")
