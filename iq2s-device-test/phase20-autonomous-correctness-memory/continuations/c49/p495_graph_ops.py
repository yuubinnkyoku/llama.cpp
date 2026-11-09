#!/usr/bin/env python3
"""Phase 49.5 - per op counts and bytes of each graph in a sched debug dump.

    python p495_graph_ops.py <stderr.txt> [graph_index]

GGML_SCHED_DEBUG=2 prints one line per node:
    node #  0 (  GET_ROWS): name (  20K) [  CPU ] use=2,c=1: src ...
Graphs are separated by the "node #  0" index restarting. Prints the op table of
the requested graph (default: the widest decode graph) and the node count.
"""
import re
import sys
from collections import defaultdict

NODE_RE = re.compile(r"^node #\s*(\d+) \(\s*([A-Z0-9_]+)\):\s*(\S+)\s*\(\s*([0-9.]+)([KMGT]?)\)")


def size_bytes(num, unit):
    mult = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}[unit]
    return float(num) * mult


def main():
    path = sys.argv[1]
    want = int(sys.argv[2]) if len(sys.argv) > 2 else -1

    graphs = []
    cur = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = NODE_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        if idx == 0 and cur:
            graphs.append(cur)
            cur = []
        cur.append((m.group(2), m.group(3), size_bytes(m.group(4), m.group(5))))
    if cur:
        graphs.append(cur)

    print(f"graphs {len(graphs)} nodes {[len(g) for g in graphs]}")

    if want < 0:
        want = max(range(len(graphs)), key=lambda i: len(graphs[i]))

    g = graphs[want]
    print(f"graph {want} nodes {len(g)}")
    nbytes = sum(x[2] for x in g)

    per = defaultdict(lambda: [0, 0.0])
    for op, name, nb in g:
        per[op][0] += 1
        per[op][1] += nb

    print(f"{'op':16s} {'count':>6s} {'MiB':>10s} {'share%':>7s}")
    for op, (c, nb) in sorted(per.items(), key=lambda kv: -kv[1][1]):
        print(f"{op:16s} {c:6d} {nb / 1048576.0:10.1f} {100.0 * nb / nbytes:7.2f}")
    print(f"{'TOTAL':16s} {len(g):6d} {nbytes / 1048576.0:10.1f} {100.0:7.2f}")

    # the largest single tensors, these dominate a memory bound decode
    print()
    print("largest written tensors")
    for op, name, nb in sorted(g, key=lambda x: -x[2])[:12]:
        print(f"  {nb / 1048576.0:9.2f} MiB  {op:12s} {name}")


if __name__ == "__main__":
    main()
