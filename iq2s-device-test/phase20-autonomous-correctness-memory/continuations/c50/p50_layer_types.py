#!/usr/bin/env python3
"""Phase 50 - MUL_MAT weight bytes per 16-layer block and per quantized type.

    python p50_layer_types.py <sched-debug-dump.txt>

Reads a GGML_SCHED_DEBUG=2 node dump and prints the logical weight size of every
MUL_MAT tensor, grouped by layer block and guessed type.  This is the graph's
logical size, not DRAM traffic: the same bytes are read once per token by
whichever backend holds the tensor.

The header row is the guessed type, the last column the block total in MiB.
Layer -1 means a tensor outside blk.N (the output layer).
"""
import re
import sys
from collections import defaultdict

NODE_RE = re.compile(r"^node #\s*(\d+) \(\s*([A-Z0-9_]+)\):\s*\S+\s*\(\s*([0-9.]+)([KMGT]?)\)[^:]*:(.*)$")
PART_RE = re.compile(r"(\S+)\s*\(\s*([0-9.]+)([KMGT]?)\)")
MULT = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}

# bytes per 256 element block
BLOCKS = {
    "IQ2_S": 84, "IQ2_XS": 74, "IQ2_XXS": 66, "IQ3_S": 110, "IQ3_XXS": 98,
    "IQ4_XS": 136, "Q2_K": 84, "Q3_K": 110, "Q4_K": 144, "BF16": 512, "F32": 1024,
}


def guess(wb, k, n):
    if k <= 0 or n <= 0:
        return "?"
    per_elem = wb / float(k * n)
    best = None
    for name, blk in BLOCKS.items():
        want = blk / 256.0
        d = abs(per_elem - want) / want
        if best is None or d < best[0]:
            best = (d, name)
    return best[1] if best[0] < 0.06 else "?"


def main():
    seen = {}
    for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
        m = NODE_RE.match(line)
        if not m or m.group(2) != "MUL_MAT":
            continue
        srcs = [(p[0], float(p[1]) * MULT[p[2]]) for p in PART_RE.findall(m.group(5))]
        if len(srcs) < 2:
            continue
        wname, wbytes = srcs[0]
        out = float(m.group(3)) * MULT[m.group(4)]
        k, n = srcs[1][1] / 4.0, out / 4.0
        lm = re.match(r"blk\.(\d+)\.", wname)
        # k and n come from the activation and the output of the same node
        seen.setdefault(wname, (wbytes, guess(wbytes, k, n), int(lm.group(1)) if lm else -1))

    tab = defaultdict(float)
    for wname, (wbytes, wtype, layer) in seen.items():
        blk = "other" if layer < 0 else f"{layer // 16 * 16:02d}-{layer // 16 * 16 + 15}"
        tab[(blk, wtype)] += wbytes

    types = sorted({t for (_, t) in tab})
    blocks = sorted({b for (b, _) in tab})
    print(f"{'block':7s} " + " ".join(f"{t:>8s}" for t in types) + f"{'total':>10s}")
    for b in blocks:
        row = [tab.get((b, t), 0.0) for t in types]
        print(f"{b:7s} " + " ".join(f"{v / 1048576.0:8.1f}" for v in row) + f"{sum(row) / 1048576.0:10.1f}")
    tot = [sum(tab.get((b, t), 0.0) for b in blocks) for t in types]
    print(f"{'TOTAL':7s} " + " ".join(f"{v / 1048576.0:8.1f}" for v in tot) + f"{sum(tot) / 1048576.0:10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
