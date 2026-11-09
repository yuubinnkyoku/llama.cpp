#!/usr/bin/env python3
"""Phase 50 - per token negative log likelihood from two decode-probe logit dumps.

    python p50_nll.py <ref.f32> <test.f32> <tokens> [steps] [vocab]

Each dump holds `steps` rows of `vocab` float32 logits.  Row s predicts the
token that decode-probe actually forced at step s, so the replay token list is
the target sequence.  Because both configurations are scored on the same forced
tokens, the difference per step is paired and much tighter than an absolute
perplexity over one short chunk.

Rows are compared on the full vocabulary in float64.  Also reports the argmax
per step, which is what greedy decoding would emit from that row.
"""
import sys
import numpy as np

STEPS = 4
VOCAB = 248320


def log_softmax(x):
    m = np.max(x)
    z = x - m
    return z - np.log(np.sum(np.exp(z)))


def main():
    ref_path, tst_path = sys.argv[1], sys.argv[2]
    tokens = [int(t) for t in sys.argv[3].split()]
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else STEPS
    vocab = int(sys.argv[5]) if len(sys.argv) > 5 else VOCAB
    if len(tokens) < steps:
        raise SystemExit(f"{len(tokens)} tokens but {steps} steps")

    ref = np.fromfile(ref_path, dtype="<f4").reshape(steps, vocab)
    tst = np.fromfile(tst_path, dtype="<f4").reshape(steps, vocab)

    print(f"{'step':>4s} {'target':>8s} {'nll_ref':>10s} {'nll_tst':>10s} {'d_nll':>10s} {'p_ref':>9s} {'p_tst':>9s} {'argmax_ref':>11s} {'argmax_tst':>11s}")
    diffs = []
    for s in range(steps):
        r = ref[s].astype(np.float64)
        t = tst[s].astype(np.float64)
        lr, lt = log_softmax(r), log_softmax(t)
        tgt = tokens[s]
        nr, nt = -lr[tgt], -lt[tgt]
        diffs.append(nt - nr)
        print(f"{s:4d} {tgt:8d} {nr:10.6f} {nt:10.6f} {nt - nr:+10.6f} "
              f"{np.exp(lr[tgt]):9.6f} {np.exp(lt[tgt]):9.6f} {int(np.argmax(r)):11d} {int(np.argmax(t)):11d}")

    d = np.array(diffs, dtype=np.float64)
    print()
    print(f"mean nll ref   {np.mean([-log_softmax(ref[s].astype(np.float64))[tokens[s]] for s in range(steps)]):.6f}")
    print(f"mean nll test  {np.mean([-log_softmax(tst[s].astype(np.float64))[tokens[s]] for s in range(steps)]):.6f}")
    print(f"mean d_nll     {d.mean():+.6f}")
    if steps > 1:
        se = d.std(ddof=1) / np.sqrt(steps)
        print(f"se(d_nll)      {se:.6f}  t={d.mean() / se:+.2f}  dof={steps - 1}")
    print(f"perplexity ref {np.exp(np.mean([-log_softmax(ref[s].astype(np.float64))[tokens[s]] for s in range(steps)])):.4f}")
    print(f"perplexity tst {np.exp(np.mean([-log_softmax(tst[s].astype(np.float64))[tokens[s]] for s in range(steps)])):.4f}")
    same = sum(1 for s in range(steps) if int(np.argmax(ref[s])) == int(np.argmax(tst[s])))
    print(f"argmax same    {same}/{steps}")
    # greedy path: would both configs have accepted the forced token?
    acc_r = sum(1 for s in range(steps) if int(np.argmax(ref[s])) == tokens[s])
    acc_t = sum(1 for s in range(steps) if int(np.argmax(tst[s])) == tokens[s])
    print(f"greedy==forced ref {acc_r}/{steps} test {acc_t}/{steps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
