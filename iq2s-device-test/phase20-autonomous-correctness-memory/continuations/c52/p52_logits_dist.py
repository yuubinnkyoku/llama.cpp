#!/usr/bin/env python3
"""Phase 52 - logits distance statistics between two decode-probe dumps.

    python p52_logits_dist.py <ref.f32> <test.f32> <steps> [vocab]

p50_nll.py reports the NLL and the argmax; this reports how far the two logit
vectors are from each other, which is the other half of the picture: a small
NLL difference with a large relative distance means the distributions differ
without changing the ranking.

Per step it prints the relative L2 distance (||a-b|| / ||a||), the maximum
absolute difference, the total variation distance and the KL divergence of the
two softmax distributions. The summary gives the min/median/max of each.
"""
import sys
import numpy as np

STEPS = 4
VOCAB = 248320


def softmax(x):
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def main():
    ref_path, tst_path = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else STEPS
    vocab = int(sys.argv[4]) if len(sys.argv) > 4 else VOCAB

    ref = np.fromfile(ref_path, dtype="<f4").reshape(steps, vocab)
    tst = np.fromfile(tst_path, dtype="<f4").reshape(steps, vocab)

    rel_l2, max_abs, tv, kl = [], [], [], []
    print(f"{'step':>4s} {'rel_l2%':>9s} {'max_abs':>9s} {'tv%':>8s} {'kl':>11s}")
    for s in range(steps):
        a = ref[s].astype(np.float64)
        b = tst[s].astype(np.float64)
        d = a - b
        r = np.linalg.norm(d) / np.linalg.norm(a) * 100.0
        m = np.max(np.abs(d))
        pa, pb = softmax(a), softmax(b)
        t = 0.5 * np.sum(np.abs(pa - pb)) * 100.0
        # KL(pa || pb), guarded against zeros
        mask = pa > 0
        k = float(np.sum(pa[mask] * np.log(pa[mask] / np.maximum(pb[mask], 1e-300))))
        rel_l2.append(r)
        max_abs.append(m)
        tv.append(t)
        kl.append(k)
        print(f"{s:4d} {r:9.3f} {m:9.4f} {t:8.3f} {k:11.3e}")

    def summary(name, v, fmt):
        v = np.array(v, dtype=np.float64)
        print(f"{name:10s} min {fmt.format(v.min())}  median {fmt.format(np.median(v))}  max {fmt.format(v.max())}")

    print()
    summary("rel_l2%", rel_l2, "{:.3f}")
    summary("max_abs", max_abs, "{:.4f}")
    summary("tv%", tv, "{:.3f}")
    summary("kl", kl, "{:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
