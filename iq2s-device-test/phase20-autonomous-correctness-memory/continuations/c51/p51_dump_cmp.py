#!/usr/bin/env python3
"""Phase 51 - compare a GGML_HEXAGON_SPLIT_DUMP file with its reference dump.

Both files are a sequence of records, one per split op per decode step:

  "C46D" | step u32 | count u32 | count * f32

The split run writes the rows its two shares produced (the output was poisoned
with -12345 first, so an unwritten row shows up). The reference run writes the
full tensor from the DSP alone. Rows [0, r) come from the CPU share, so they may
round differently; rows [r, m) come from the DSP share and must match the full
DSP result bit for bit.

Usage: python p51_dump_cmp.py <dump.bin> <dump.ref>
"""
import struct
import sys
import pathlib


def read_records(path):
    raw = pathlib.Path(path).read_bytes()
    out = []
    off = 0
    while off + 12 <= len(raw):
        tag = raw[off:off + 4]
        if tag != b"C46D":
            raise SystemExit(f"{path}: bad tag {tag!r} at {off}")
        step, cnt = struct.unpack_from("<II", raw, off + 4)
        nbytes = cnt * 4
        body = raw[off + 12:off + 12 + nbytes]
        if len(body) != nbytes:
            raise SystemExit(f"{path}: truncated record at {off}")
        out.append((step, cnt, struct.unpack("<%df" % cnt, body)))
        off += 12 + nbytes
    return out, off, len(raw)


def main():
    a, used, total = read_records(sys.argv[1])
    b, used2, total2 = read_records(sys.argv[2])
    print(f"dump {len(a)} records {total} B, ref {len(b)} records {total2} B")
    print("record i pairs the same op of the same step; the step field is one shared")
    print("counter, so the dump gets even values and the ref odd ones")
    if len(a) != len(b):
        print("RECORD COUNT MISMATCH")
    bad = 0
    prefix_rows = []
    tail_diff = 0
    sentinel_total = 0
    nonfinite_total = 0
    for i, ((sa, ca, va), (sb, cb, vb)) in enumerate(zip(a, b)):
        if ca != cb:
            print(f"rec {i}: shape mismatch dump {ca} ref {cb}")
            bad += 1
            continue
        sentinel = sum(1 for v in va if v == -12345.0)
        nonfinite = sum(1 for v in va if v != v or v in (float("inf"), float("-inf")))
        sentinel_total += sentinel
        nonfinite_total += nonfinite
        diff = [j for j in range(ca) if va[j] != vb[j]]
        if not diff:
            prefix_rows.append(0)
            continue
        first, last = diff[0], diff[-1]
        if (last - first + 1) != len(diff) or first != 0:
            print(f"rec {i}: {len(diff)} rows differ in [{first},{last}], not a leading run")
            bad += 1
            continue
        prefix_rows.append(len(diff))
        for j in range(len(diff), ca):
            if va[j] != vb[j]:
                tail_diff += 1
        if sentinel or nonfinite:
            print(f"rec {i}: sentinel {sentinel} nonfinite {nonfinite}")
            bad += 1
    print()
    if prefix_rows:
        print(f"records with a leading CPU-share difference: {len(prefix_rows)},"
              f" row counts {sorted(set(prefix_rows))}")
    print(f"rows differing outside the CPU share: {tail_diff}")
    print(f"sentinel rows total {sentinel_total}, nonfinite values total {nonfinite_total}")
    print("PASS" if bad == 0 and tail_diff == 0 else f"FAIL bad {bad} tail {tail_diff}")


if __name__ == "__main__":
    main()
