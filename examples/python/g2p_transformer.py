"""Encrypted grapheme-to-phoneme conversion with an encoder-decoder transformer.

The task is the CMU pronouncing dictionary: spell a word, get its phonemes
("knight" -> N AY T). The model is a one-layer encoder-decoder transformer
trained in PyTorch and run encrypted by fheml:

    PYTHONPATH=build/python python3 examples/python/g2p_transformer.py --arch basic [--n 3] [--insecure]
    PYTHONPATH=build/python python3 examples/python/g2p_transformer.py --arch multihead

``basic`` has single-head attention, ``multihead`` has 4 heads of width 8.
Everything else is the same.

The source (8 letters) and the target (<s> plus 7 phonemes) form one
encrypted sequence of 16 tokens. Token segments give the source and the
target tokens their own weights, and attention masks give each attention
its pattern (see the README section on encoder-decoders):

    embed         letters and phonemes (one-hot) -> d, per segment, plus positions
    self-attn     encoder (source x source) and causal decoder (target x earlier
                  target) attention, as ONE masked attention layer: neither
                  reads the other's tokens, so they run side by side
    encoder FFN   source tokens only
    cross-attn    target queries x source keys
    decoder FFN   target tokens only
    head          target tokens -> phoneme logits

The client decodes greedily. It encrypts the letters and the target prefix,
the server returns encrypted next-phoneme logits for every target position,
and the client decrypts them, appends the phoneme at the current position
and sends the sequence again, until </s>. The server never sees a letter or a
phoneme. The causal mask makes the logits at position t depend only on the
prefix up to t, so one pass per output phoneme suffices.

The script trains with teacher forcing, decodes the test words greedily in
torch and with the fheml plaintext network (the encrypted computation in the
clear), and then decodes --n test words encrypted. It reports phoneme and
whole-word accuracy (word accuracy = 1 - WER) and compares every encrypted
pass with torch.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch import nn

import fheml
from fheml import nn as hnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
import fhe_eval  # noqa: E402
from datasets import load_cmudict  # noqa: E402

S, T = 8, 8   # source letters; target tokens = <s> + 7 phonemes (the last input never feeds a prediction)
N = S + T
V = 42        # one-hot width: 42 phoneme ids (pad, <s>, </s>, 39 phonemes); letters use 27 of them
PAD, BOS, EOS = 0, 1, 2
SEG = (S, T)


def g2p(heads, d=32, hid=64):
    m = hnn.encoder_decoder_masks(S, T)
    act = lambda: hnn.Poly([0, 0.5, 0.25])  # noqa: E731
    return nn.Sequential(
        hnn.SegmentLinear(V, d, SEG), hnn.PosEmbedding(N, d),
        # encoder self-attention + causal decoder self-attention, one layer
        hnn.Residual(hnn.Attention(d, N, heads=heads, mask=m["encoder"] + m["causal"], segments=SEG)),
        hnn.Residual(hnn.SegmentLinear(d, hid, SEG, active=[0]), act(),
                     hnn.SegmentLinear(hid, d, SEG, active=[0])),                     # encoder FFN
        hnn.Residual(hnn.Attention(d, N, heads=heads, mask=m["cross"], segments=SEG,
                                   query_segments=[1], key_segments=[0])),            # cross-attention
        hnn.Residual(hnn.SegmentLinear(d, hid, SEG, active=[1]), act(),
                     hnn.SegmentLinear(hid, d, SEG, active=[1])),                     # decoder FFN
        hnn.SegmentLinear(d, V, SEG, active=[1]))                                     # phoneme logits


def inputs(src, prefix):
    """One-hot model inputs (B, N, V): the letters, then the decoder input
    prefix (B, T) of phoneme ids (pad beyond the prefix)."""
    x = np.zeros((len(src), N, V), np.float32)
    b = np.arange(len(src))[:, None]
    x[b, np.arange(S), src] = 1
    x[b, S + np.arange(T), prefix] = 1
    return x


def teacher_inputs(src, tgt):
    """Decoder input <s> + tgt[:-1] (teacher forcing)."""
    return inputs(src, np.concatenate([np.full((len(tgt), 1), BOS), tgt[:, :-1]], 1))


def g2p_loss(out, tgt):
    return nn.functional.cross_entropy(out[:, S:].reshape(-1, V), tgt.reshape(-1), ignore_index=PAD)


def phoneme_accuracy(out, tgt):
    """Teacher-forced next-phoneme accuracy over the non-pad target positions."""
    pred = torch.as_tensor(out)[:, S:].argmax(-1)
    tgt = torch.as_tensor(tgt)
    keep = tgt != PAD
    return float((pred[keep] == tgt[keep]).float().mean())


def pick(logits):
    """The client's choice from (…, V) target logits: never pad or <s>."""
    logits = np.array(logits, np.float64)
    logits[..., [PAD, BOS]] = -np.inf
    return logits.argmax(-1)


def greedy(step, src, on_pass=None):
    """Greedy decoding of a batch of words. step(x) -> (B, N, V) logits for
    one-hot inputs x (B, N, V). Returns phoneme ids (B, T) with </s> and pad
    after it, like the dataset's targets, and the number of passes."""
    B = len(src)
    prefix = np.zeros((B, T), np.int64)
    prefix[:, 0] = BOS
    pred = np.zeros((B, T), np.int64)
    done = np.zeros(B, bool)
    for t in range(T):
        x = inputs(src, prefix)
        logits = step(x)
        if on_pass:
            on_pass(t, x, logits)
        p = pick(logits[:, S + t])
        pred[:, t] = np.where(done, PAD, p)
        done |= p == EOS
        if done.all() or t + 1 == T:
            return pred, t + 1
        prefix[:, t + 1] = np.where(done, PAD, p)
    return pred, T


def torch_step(model):
    def step(x):
        with torch.no_grad():
            return model(torch.as_tensor(x, dtype=next(model.parameters()).dtype)).double().numpy()
    return step


def spell(ids, phonemes):
    return " ".join(str(phonemes[i]) for i in ids if i not in (PAD, BOS, EOS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=("basic", "multihead"), default="basic",
                    help="single-head (basic) or 4-head attention")
    ap.add_argument("--plain-words", type=int, default=500, help="test words decoded with the fheml plaintext network")
    fhe_eval.add_args(ap)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data = load_cmudict(args.data, max_letters=S, max_phonemes=T - 1)
    phonemes = data["phonemes"]
    src_tr, tgt_tr = data["train_src"], data["train_tgt"]
    # The test words come sorted; shuffle them once (fixed seed) so the
    # first k, which run in fheml and encrypted, are a fair sample.
    order = np.random.default_rng(0).permutation(len(data["test_src"]))
    src_te, tgt_te, words = (data[k][order] for k in ("test_src", "test_tgt", "test_words"))
    assert len(phonemes) == V, f"expected {V} phoneme ids, got {len(phonemes)}"
    xtr, xte = torch.tensor(teacher_inputs(src_tr, tgt_tr)), torch.tensor(teacher_inputs(src_te, tgt_te))
    ytr, yte = torch.tensor(tgt_tr), torch.tensor(tgt_te)
    print(f"CMUdict: {len(xtr)} training and {len(xte)} test words")

    model = g2p(1 if args.arch == "basic" else 4)
    tf_acc, train_time = fhe_eval.train(model, xtr, ytr, xte, yte, args.epochs or 30, loss_fn=g2p_loss,
                                        metric=phoneme_accuracy, lr=3e-3, batch=256)

    # Greedy decoding of the whole test set in torch.
    model.eval()
    pred, _ = greedy(torch_step(model), src_te)
    torch_tok, torch_word = fhe_eval.sequence_accuracy(pred, tgt_te)
    print(f"torch greedy decoding, {len(src_te)} test words: phoneme accuracy {torch_tok:.1%}, "
          f"word accuracy {torch_word:.1%}")

    net = fheml.from_torch(model, (N, V))  # checks fheml plaintext == torch
    if args.out:
        net.save(args.out)
        print("wrote", args.out)
    net.compile()

    # The encrypted computation in the clear, greedy-decoded like the client does.
    k = min(args.plain_words, len(src_te))
    plain_step = lambda x: np.stack([fheml.plain(net, xi).reshape(N, V) for xi in x])  # noqa: E731
    plain_pred, _ = greedy(plain_step, src_te[:k])
    plain_tok, plain_word = fhe_eval.sequence_accuracy(plain_pred, tgt_te[:k])
    ref_tok, ref_word = fhe_eval.sequence_accuracy(pred[:k], tgt_te[:k])
    same = float(np.mean(np.all(plain_pred == pred[:k], 1)))
    print(f"fheml plaintext greedy decoding, first {k} test words: phoneme accuracy {plain_tok:.1%}, word accuracy "
          f"{plain_word:.1%} (torch on them: {ref_tok:.1%}, {ref_word:.1%}; same output for {same:.1%})")

    if args.n <= 0:
        return
    level_budget = fhe_eval.prepare(net, args, xtr[:args.calibration].double().numpy())
    ctx, setup = fhe_eval.context(net, args, level_budget)

    # Encrypted greedy decoding, one word at a time: one encrypted pass per phoneme.
    model = model.double()
    ref_step = torch_step(model)
    times, errs, enc_pred = [], [], []
    for i in range(min(args.n, len(src_te))):
        word_t = time.time()

        def enc_step(x):
            t = time.time()
            y = fheml.infer(net, ctx, x[0]).reshape(1, N, V)
            times.append(time.time() - t)
            return y

        def compare(t, x, logits):
            ref = ref_step(x)[:, S:]
            errs.append(float(np.abs(logits[:, S:] - ref).max()))

        p, passes = greedy(enc_step, src_te[i:i + 1], on_pass=compare)
        enc_pred.append(p[0])
        print(f"word {i} '{words[i]}': encrypted -> {spell(p[0], phonemes)} | torch -> {spell(pred[i], phonemes)} | "
              f"truth {spell(tgt_te[i], phonemes)}; {passes} passes, {time.time() - word_t:.1f}s, "
              f"max |enc - torch| {max(errs[-passes:]):.1e}")
    enc_pred = np.asarray(enc_pred).reshape(-1, T)
    n = len(enc_pred)
    enc_tok, enc_word = fhe_eval.sequence_accuracy(enc_pred, tgt_te[:n])
    agree = int(np.sum(np.all(enc_pred == pred[:n], 1)))
    print(f"encrypted: phoneme accuracy {enc_tok:.1%}, word accuracy {enc_word:.1%} on {n} words; "
          f"{agree}/{n} decoded exactly like torch; {np.mean(times):.1f}s per pass, "
          f"max |enc - torch| {max(errs):.1e}")

    run = dict(image_seconds=times, max_abs_err=errs, argmax_agree=agree, images=n)
    fhe_eval.record(args, net, ctx, setup, run, arch=f"g2p-{args.arch}", teacher_forced_accuracy=tf_acc,
                    torch_phoneme_accuracy=torch_tok, torch_word_accuracy=torch_word, plain_phoneme_accuracy=plain_tok,
                    plain_word_accuracy=plain_word, plain_words=k, plain_same_as_torch=same,
                    encrypted_phoneme_accuracy=enc_tok, encrypted_word_accuracy=enc_word,
                    train_seconds=train_time)


if __name__ == "__main__":
    main()
