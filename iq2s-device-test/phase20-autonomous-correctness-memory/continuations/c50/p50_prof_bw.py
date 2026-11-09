#!/usr/bin/env python3
"""Phase 50 - per-type MUL_MAT effective bandwidth from a GGML_HEXAGON_PROFILE dump.

    python p50_prof_bw.py <profile.txt> [label]

One dump line looks like

  P48_I1 profile-op MUL_MAT|blk.56.attn_qkv.weight x attn_norm-56 -> node_3375|5120:10240 x 5120:5 -> 10240:5|iq3_xxs x f32 -> f32|98:1960 x 4:20480 -> 4:40960|hvx-tiled vtcm 1925120|usec 4512 cycles 9530158 start ... mhz 2112.2

Fields split on '|': name, shapes, types, byte pairs, kernel+budget, time.
The weight is the first source, so ne0 is K and ne1 is N. Bandwidth is the
logical weight bytes (blocks x block size) over the measured DSP usec.
This is weight traffic, not DRAM traffic.
"""
import re
import sys
from collections import defaultdict

# bytes per 256 element block, same table as p495_mulmat_classes.py
BLOCKS = {
    "IQ2_S": 84, "IQ2_XS": 74, "IQ2_XXS": 66, "IQ2_M": 116,
    "IQ3_S": 110, "IQ3_XXS": 98, "IQ3_M": 134,
    "IQ4_XS": 136, "IQ4_NL": 18, "Q2_K": 84, "Q3_K": 110, "Q4_K": 144, "Q5_K": 176, "Q6_K": 210,
    "Q8_0": 34, "F16": 512, "BF16": 512, "F32": 1024,
}

USEC_RE = re.compile(r"\busec (\d+)")
SHAPE_RE = re.compile(r"^(\d+):(\d+) x (\d+):(\d+) -> (\d+):(\d+)$")


def parse(path, want_m):
    rows = []  # (wtype, m, n, k, usec)
    skipped = 0
    for line in open(path, encoding="utf-8", errors="replace"):
        if "profile-op MUL_MAT|" not in line:
            continue
        parts = line.split("|")
        if len(parts) < 7:
            skipped += 1
            continue
        types = parts[3].split(" x ")
        wtype = types[0].strip().upper()
        m = SHAPE_RE.match(parts[2].strip())
        u = USEC_RE.search(parts[-1])
        if not m or not u:
            skipped += 1
            continue
        k, n = int(m.group(1)), int(m.group(2))
        bm = int(m.group(4))  # activation ne1 is the batch size
        if want_m and bm != want_m:
            continue
        rows.append((wtype, bm, n, k, int(u.group(1))))
    return rows, skipped


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else ""
    want_m = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    rows, skipped = parse(path, want_m)
    if not rows:
        print(f"no MUL_MAT profile lines with m={want_m} in {path}")
        return 1

    per = defaultdict(lambda: [0, 0.0, 0, 0.0])  # count, bytes, usec, maccs
    total_bytes = 0.0
    total_usec = 0
    total_mac = 0.0
    unknown = set()
    for wtype, m_, n, k, usec in rows:
        blk = BLOCKS.get(wtype)
        if blk is None:
            unknown.add(wtype)
            continue
        nbytes = (k * n / 256.0) * blk
        mac = float(k) * n * m_
        e = per[wtype]
        e[0] += 1
        e[1] += nbytes
        e[2] += usec
        e[3] += mac
        total_bytes += nbytes
        total_usec += usec
        total_mac += mac

    print(f"{'type':10s} {'ops':>5s} {'MiB':>9s} {'usec':>9s} {'GB/s':>8s} {'GMAC/s':>8s} {'share%':>7s}")
    for t, v in sorted(per.items(), key=lambda kv: -kv[1][1]):
        gbs = v[1] / v[2] / 1000.0 if v[2] else 0.0
        gmac = v[3] / v[2] / 1000.0 if v[2] else 0.0
        print(f"{t:10s} {v[0]:5d} {v[1] / 1048576.0:9.1f} {v[2]:9d} {gbs:8.2f} {gmac:8.2f} {100.0 * v[1] / total_bytes:7.2f}")
    if total_usec:
        print(f"{'TOTAL':10s} {len(rows):5d} {total_bytes / 1048576.0:9.1f} {total_usec:9d} "
              f"{total_bytes / total_usec / 1000.0:8.2f} {total_mac / total_usec / 1000.0:8.2f} {100.0:7.2f}")
    if unknown:
        print(f"unknown types skipped: {sorted(unknown)}")
    if skipped:
        print(f"unparsed MUL_MAT lines: {skipped}")
    if label:
        print(f"LABEL {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
