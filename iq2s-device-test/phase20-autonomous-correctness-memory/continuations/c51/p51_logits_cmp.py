#!/usr/bin/env python3
"""Phase 51 - compare two logits files row by row.

The decode probe writes one f32 block per step. The vocabulary is the last
dimension, so a step is a contiguous run of floats. This prints, per step, the
maximum absolute difference, the mean absolute difference, how many values are
bit identical, and whether the argmax agrees.

Usage: python p51_logits_cmp.py <a.f32> <b.f32> [vocab]
"""
import struct
import sys
import pathlib

VOCAB = 248320  # Qwen3.8 27B vocabulary; overridable on the command line


def cmp_step(a, b, label):
    n = len(a)
    ndiff = 0
    dmax = 0.0
    dsum = 0.0
    nonfinite = 0
    for x, y in zip(a, b):
        if x != x or y != y or x in (float("inf"), float("-inf")):
            nonfinite += 1
        if x != y:
            ndiff += 1
            d = abs(x - y)
            dmax = max(dmax, d)
            dsum += d
    ia = max(range(n), key=lambda i: a[i])
    ib = max(range(n), key=lambda i: b[i])
    top_a = sorted(range(n), key=lambda i: a[i], reverse=True)[:10]
    top_b = sorted(range(n), key=lambda i: b[i], reverse=True)[:10]
    overlap = len(set(top_a) & set(top_b))
    print(f"{label}: n {n} differ {ndiff} ({100.0*ndiff/n:.2f}%) max |d| {dmax:.6g} "
          f"mean |d| {dsum/max(ndiff,1):.3g} nonfinite {nonfinite} "
          f"argmax {ia}/{ib} {'same' if ia == ib else 'DIFF'} top10 overlap {overlap}")


def main():
    a = pathlib.Path(sys.argv[1]).read_bytes()
    b = pathlib.Path(sys.argv[2]).read_bytes()
    vocab = int(sys.argv[3]) if len(sys.argv) > 3 else VOCAB
    if len(a) != len(b):
        print(f"size mismatch {len(a)} vs {len(b)}")
    fa = struct.unpack("<%df" % (len(a) // 4), a)
    fb = struct.unpack("<%df" % (len(b) // 4), b)
    steps = len(fa) // vocab
    print(f"{len(fa)} floats, {steps} steps of {vocab}")
    for s in range(min(steps, len(fb) // vocab)):
        cmp_step(fa[s * vocab:(s + 1) * vocab], fb[s * vocab:(s + 1) * vocab], f"step {s}")


if __name__ == "__main__":
    main()
