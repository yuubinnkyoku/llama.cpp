#!/usr/bin/env python3
"""Phase 49.5 - classify the MUL_MAT nodes of a sched debug dump.

    python p495_mulmat_classes.py <stderr.txt> [graph_index]

Each line of GGML_SCHED_DEBUG=2 holds the node and its sources with sizes:
    node # 15 (MUL_MAT): node_15 ( 40K) [CPU] use=1,c=1: blk.0.attn_qkv.weigh ( 21M) [CPU] attn_norm-0 ( 20K)
For a mul_mat the first source is the weight, the second the activations and the
node is the output, so K = act_bytes / 4 and N = out_bytes / 4. The weight bytes
per element then identify the quantized type.
"""
import re
import sys
from collections import defaultdict

NODE_RE = re.compile(r"^node #\s*(\d+) \(\s*([A-Z0-9_]+)\):\s*\S+\s*\(\s*([0-9.]+)([KMGT]?)\)[^:]*:(.*)$")
PART_RE = re.compile(r"(\S+)\s*\(\s*([0-9.]+)([KMGT]?)\)")
MULT = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}

# bytes per 256 element block, only the types the 27B uses
BLOCKS = {
    "IQ2_S": 84, "IQ2_XS": 74, "IQ2_XXS": 66, "IQ2_M": 116,
    "IQ3_S": 110, "IQ3_XXS": 98, "IQ3_M": 134,
    "IQ4_XS": 136, "IQ4_NL": 18, "Q4_K": 144, "Q5_K": 176, "Q6_K": 210,
    "Q8_0": 34, "F16": 512, "BF16": 512, "F32": 1024,
}


def guess_type(weight_bytes, k, n):
    if k <= 0 or n <= 0:
        return "?"
    per_elem = weight_bytes / float(k * n)
    best = None
    for name, blk in BLOCKS.items():
        want = blk / 256.0
        d = abs(per_elem - want) / want
        if best is None or d < best[0]:
            best = (d, name)
    return best[1] if best[0] < 0.06 else f"?{per_elem:.3f}B"


def graphs_of(path):
    graphs, cur = [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = NODE_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        if idx == 0 and cur:
            graphs.append(cur)
            cur = []
        parts = PART_RE.findall(m.group(5))
        srcs = [(p[0], float(p[1]) * MULT[p[2]]) for p in parts]
        cur.append((m.group(2), float(m.group(3)) * MULT[m.group(4)], srcs))
    if cur:
        graphs.append(cur)
    return graphs


def main():
    path = sys.argv[1]
    want = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    graphs = graphs_of(path)
    if want < 0:
        want = max(range(len(graphs)), key=lambda i: len(graphs[i]))
    g = graphs[want]
    print(f"graphs {len(graphs)} nodes {[len(x) for x in graphs]} using {want}")

    per = defaultdict(lambda: [0, 0.0, 0.0])  # count, weight bytes, out bytes
    total_w = 0.0
    for op, out_bytes, srcs in g:
        if not srcs:
            continue
        wname, wbytes = srcs[0]
        if op != "MUL_MAT":
            # elementwise, norm and gather ops move about as much as their output, the
            # full src0 of a GET_ROWS is the shared table and would be counted once per node
            total_w += out_bytes
            per[f"{op} (out)"] [0] += 1
            per[f"{op} (out)"][1] += out_bytes
            per[f"{op} (out)"][2] += out_bytes
            continue
        act = srcs[1][1] if len(srcs) > 1 else 0.0
        k = act / 4.0
        n = out_bytes / 4.0
        t = guess_type(wbytes, k, n)
        key = f"MUL_MAT {t} K{int(k)} N{int(n)}"
        per[key][0] += 1
        per[key][1] += wbytes
        per[key][2] += out_bytes

    print(f"{'class':28s} {'count':>6s} {'weight_MiB':>11s} {'share%':>7s}")
    tot = sum(v[1] for v in per.values())
    for k, v in sorted(per.items(), key=lambda kv: -kv[1][1]):
        print(f"{k:28s} {v[0]:6d} {v[1] / 1048576.0:11.1f} {100.0 * v[1] / tot:7.2f}")
    print(f"{'TOTAL src0 read':28s} {'':6s} {tot / 1048576.0:11.1f} {100.0:7.2f}")


if __name__ == "__main__":
    main()
