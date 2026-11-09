#!/usr/bin/env python3
"""Phase 49.5 - compare 27B logits between configurations.

    python p495_logits_cmp.py <ref.f32> <test.f32> [steps] [vocab]

Both files hold `steps` rows of `vocab` float32 logits written by decode-probe.
The reference is treated as the truth. Prints one block of metrics.
"""
import sys
import numpy as np

STEPS = 4
VOCAB = 248320


def load(path, steps, vocab):
    a = np.fromfile(path, dtype="<f4")
    if a.size != steps * vocab:
        raise SystemExit(f"{path}: {a.size} floats, expected {steps * vocab}")
    return a.reshape(steps, vocab)


def metrics(ref, tst, step):
    r = ref[step].astype(np.float64)
    t = tst[step].astype(np.float64)
    d = t - r

    fin_r = np.isfinite(r)
    fin_t = np.isfinite(t)
    nonfinite = int((~fin_r).sum() + (~fin_t).sum())

    absd = np.abs(d)
    denom = float(np.linalg.norm(r))
    rel_l2 = float(np.linalg.norm(d) / denom) if denom > 0 else float("nan")
    rms = float(np.sqrt(np.mean(d * d)))

    # argmax agreement over the full vocabulary
    am_r = int(np.argmax(r))
    am_t = int(np.argmax(t))

    # rank changes for the reference top 10
    top_r = np.argsort(-r)[:10]
    top_t = np.argsort(-t)[:10]
    rank_t = {int(v): i for i, v in enumerate(top_t)}
    ranks = [rank_t.get(int(v), -1) for v in top_r]
    top1_moved = int(top_r[0] != top_t[0])
    top10_overlap = len(set(int(v) for v in top_r) & set(int(v) for v in top_t))

    # softmax over the full vocabulary
    pr = np.exp(r - r.max())
    pr /= pr.sum()
    pt = np.exp(t - t.max())
    pt /= pt.sum()
    tv = 0.5 * float(np.abs(pr - pt).sum())          # total variation distance
    kl = float(np.sum(pr * np.log(np.maximum(pr, 1e-300) / np.maximum(pt, 1e-300))))

    return {
        "step": step,
        "max_abs": float(absd.max()),
        "mean_abs": float(absd.mean()),
        "rel_l2": rel_l2,
        "rms": rms,
        "nonfinite": nonfinite,
        "argmax_ref": am_r,
        "argmax_test": am_t,
        "argmax_same": int(am_r == am_t),
        "top1_moved": top1_moved,
        "top10_overlap": top10_overlap,
        "top10_ranks": ranks,
        "tv": tv,
        "kl": kl,
        "max_logit": float(r.max()),
        "ref_top1_prob": float(pr[am_r]),
        "test_top1_prob": float(pt[am_r]),
    }


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    ref_path, tst_path = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else STEPS
    vocab = int(sys.argv[4]) if len(sys.argv) > 4 else VOCAB

    ref = load(ref_path, steps, vocab)
    tst = load(tst_path, steps, vocab)

    print(f"REF  {ref_path}")
    print(f"TEST {tst_path}")

    same = bool(np.array_equal(ref.view(np.uint32), tst.view(np.uint32)))
    print(f"bitexact {int(same)}")

    worst = {"rel_l2": 0.0, "max_abs": 0.0, "tv": 0.0}
    for s in range(steps):
        m = metrics(ref, tst, s)
        print(
            "step {step} max_abs {max_abs:.6g} mean_abs {mean_abs:.6g} rel_l2 {rel_l2:.6g} "
            "rms {rms:.6g} nonfinite {nonfinite} argmax {argmax_ref}->{argmax_test} same {argmax_same} "
            "top1_moved {top1_moved} top10_overlap {top10_overlap} ranks {top10_ranks} "
            "tv {tv:.6g} kl {kl:.6g} max_logit {max_logit:.6g} p_top1 {ref_top1_prob:.6g}->{test_top1_prob:.6g}".format(**m)
        )
        worst["rel_l2"] = max(worst["rel_l2"], m["rel_l2"])
        worst["max_abs"] = max(worst["max_abs"], m["max_abs"])
        worst["tv"] = max(worst["tv"], m["tv"])

    print(
        "worst rel_l2 {rel_l2:.6g} max_abs {max_abs:.6g} tv {tv:.6g}".format(**worst)
    )


if __name__ == "__main__":
    main()
