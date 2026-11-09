#!/usr/bin/env python3
"""Phase 50 - re-derive the NGL comparison with a stall robust statistic.

Every run in this environment contains one to three multi second stalls.  The
median therefore moves with the stall count: two NGL=16 runs of the same
configuration differ by 28% in med_all, but their minimum step agrees to 0.3%
(615.1 vs 617.3).  The per run minimum is the stall free floor, so use the
median of the per run minima across runs as the estimator.

    python p50_ngl_robust.py

Each entry is the `min` field printed by P50_STAT (or by p50_run.sh), taken from
runs that report P50_CAP_STABLE YES at cap0 2112000 / cap7 1862400, 8 steps,
noreplay, the default override, and skel C or D.
"""
import statistics as st

RUNS = {
    0: {
        "c3a": 668.681459, "cpu2": 678.479323, "f1": 729.573020,
        "g1": 692.967188, "g3": 681.518385, "p6": 681.142240,
    },
    16: {
        "c1b": 607.944635, "c2b": 605.027083, "c3b": 606.603906,
        "htp2": 626.907656, "p2": 615.095781, "p5": 617.250938,
        "th7p2": 615.030469, "base1": 497.576875, "base2": 619.861094,
    },
    24: {
        "c2c": 601.008490, "c3c": 593.300208, "c4a": 600.531458,
        "c4c": 605.005781, "g2": 591.123124, "p3": 622.209270,
    },
    28: {"h2": 627.127604},
}


def main():
    print("per run minimum step (stall free floor), ms")
    for ngl in sorted(RUNS):
        v = sorted(RUNS[ngl].values())
        print(f"  NGL={ngl:2d} n={len(v):2d} {[round(x,1) for x in v]}")
        print(f"          min {v[0]:.1f}  median {st.median(v):.1f}  mean {st.mean(v):.1f}  max {v[-1]:.1f}")

    print()
    print("estimator = median of the per run minima (robust to one odd run)")
    base = st.median(list(RUNS[0].values()))
    print(f"{'NGL':>4s} {'median_min':>11s} {'vs NGL=0':>10s}")
    for ngl in sorted(RUNS):
        m = st.median(list(RUNS[ngl].values()))
        print(f"{ngl:4d} {m:11.1f} {(m / base - 1) * 100:+9.1f}%")

    print()
    a = st.median(list(RUNS[16].values()))
    b = st.median(list(RUNS[24].values()))
    print(f"NGL=24 vs NGL=16: {(b / a - 1) * 100:+.1f}% (median of minima)")
    # range overlap tells you whether the ordering is resolvable
    r16 = (min(RUNS[16].values()), max(RUNS[16].values()))
    r24 = (min(RUNS[24].values()), max(RUNS[24].values()))
    print(f"NGL=16 min range {r16[0]:.1f} .. {r16[1]:.1f}")
    print(f"NGL=24 min range {r24[0]:.1f} .. {r24[1]:.1f}")
    print(f"overlap: {'yes' if not (r24[0] > r16[1] or r16[0] > r24[1]) else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
