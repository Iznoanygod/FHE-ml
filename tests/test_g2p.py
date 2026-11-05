"""Checks of the encoder-decoder example (examples/python/g2p_transformer.py)
without data or training: export of both variants, causality of the decoder
in the fheml plaintext network, and the client's greedy decoding loop.

    PYTHONPATH=build/python python3 tests/test_g2p.py
"""
import os
import sys

import numpy as np
import torch

import fheml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "python"))
import g2p_transformer as G  # noqa: E402

torch.manual_seed(0)
rng = np.random.default_rng(0)
src = rng.integers(1, 27, size=(1, G.S))

for heads in (1, 4):
    model = G.g2p(heads).eval()
    net = fheml.from_torch(model, (G.N, G.V))  # raises if fheml plaintext != torch
    net.compile()
    # The logits at target position t depend on the letters and on the
    # prefix up to t only: changing later prefix tokens leaves them unchanged.
    a = np.concatenate([[G.BOS], rng.integers(3, G.V, size=G.T - 1)])
    b = a.copy()
    b[4:] = rng.integers(3, G.V, size=G.T - 4)
    ya = fheml.plain(net, G.inputs(src, a[None])[0]).reshape(G.N, G.V)
    yb = fheml.plain(net, G.inputs(src, b[None])[0]).reshape(G.N, G.V)
    assert np.allclose(ya[G.S:G.S + 4], yb[G.S:G.S + 4], atol=1e-9), "decoder sees later tokens"
    assert not np.allclose(ya[G.S + 4:], yb[G.S + 4:]), "decoder ignores its prefix"
    # Source tokens get no logits (the head acts on the target only).
    assert np.abs(ya[:G.S]).max() == 0
    # Changing a letter changes the target logits (cross-attention reads the encoder).
    src2 = src.copy()
    src2[0, 0] = src[0, 0] % 26 + 1
    yc = fheml.plain(net, G.inputs(src2, a[None])[0]).reshape(G.N, G.V)
    assert not np.allclose(ya[G.S:], yc[G.S:])
    # The source and target embeddings are separate weight sets.
    assert {layer.type for layer in net.layers} >= {"tokendense", "residual"}

# Greedy decoding: a stub server that always predicts phoneme 5, 6, ...,
# then </s> at position 3. The client feeds its own choices back, stops at
# </s> and pads after it.
seen = []


def stub(x):
    prefix = x[:, G.S:].argmax(-1)
    seen.append(prefix.copy())
    logits = np.zeros((len(x), G.N, G.V))
    for t in range(G.T):
        logits[:, G.S + t, G.EOS if t == 3 else 5 + t] = 1
    logits[:, :, G.PAD] = logits[:, :, G.BOS] = 9  # the client never picks these
    return logits


pred, passes = G.greedy(stub, src)
assert passes == 4, passes
assert pred.tolist() == [[5, 6, 7, G.EOS, 0, 0, 0, 0]], pred
assert seen[3][0].tolist() == [G.BOS, 5, 6, 7, 0, 0, 0, 0], seen[3]
tok, word = __import__("fhe_eval").sequence_accuracy(pred, np.array([[5, 6, 7, G.EOS, 0, 0, 0, 0]]))
assert tok == 1 and word == 1
print("PASSED")
