#!/usr/bin/env python3
"""Phase 51 - re-read the Phase 47 row split ABBA blocks.

The Phase 47 device logs (pulled from /data/local/tmp/c46-27b/) contain ABBA
blocks: each block runs one "off" mode (SPLIT=0) and one "async" mode
(SPLIT=1, SPLIT_CPU_FRAC=0.9) run at ngl 9. This script turns those logs into
pairs and keeps only the pairs whose CPU cap did not move between the two runs,
because the cap decides the CPU/DSP balance and a pair across a cap change is
not a comparison.

Usage: python p51_p47_pairs.py <dir-with-extracted-logs>
"""
import re
import sys
import pathlib
import statistics

RUN = re.compile(
    r"^P47_RUN idx (\d+) mode (\w+) ngl (\d+) rows default (.*?) rc (\d+)"
    r".*?cap_after (\d+) (\d+)")
START = re.compile(
    r"^P47_RUN idx (\d+) mode (\w+) ngl (\d+) rows default (.*?) cap0 (\d+) cap7 (\d+)")
RESULT = re.compile(r"^P47_OUT RESULT forwards (\d+) decode_ms ([\d.]+)")


def parse(path):
    runs = []
    cur = None
    for line in path.read_text(errors="replace").splitlines():
        m = START.match(line)
        if m:
            cur = {"file": path.name, "idx": int(m.group(1)), "mode": m.group(2),
                   "ngl": int(m.group(3)), "frac": m.group(4).strip(),
                   "cap0": int(m.group(5)), "cap_after": None, "rc": None,
                   "ms": None, "steps": None}
            runs.append(cur)
            continue
        m = RUN.match(line)
        if m and cur is not None:
            cur["rc"] = int(m.group(5))
            cur["cap_after"] = int(m.group(6))
            continue
        m = RESULT.match(line)
        if m and cur is not None:
            cur["steps"] = int(m.group(1))
            cur["ms"] = float(m.group(2))
    return runs


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "air")
    runs = []
    for p in sorted(root.rglob("*.log")):
        runs.extend(parse(p))
    # the p47-*.out aggregates repeat the same runs that the per block logs carry

    by_file = {}
    for r in runs:
        by_file.setdefault(r["file"], []).append(r)

    print(f"parsed {len(runs)} run records in {len(by_file)} files")
    stable, refused, incomplete, moved = [], 0, 0, 0
    for f, rs in sorted(by_file.items()):
        rs = sorted(rs, key=lambda r: r["idx"])
        for a, b in zip(rs[0::2], rs[1::2]):
            if a["rc"] != 0 or b["rc"] != 0:
                refused += 1
                continue
            if a["ms"] is None or b["ms"] is None:
                incomplete += 1
                continue
            if a["cap_after"] != b["cap0"]:
                moved += 1
                continue
            if a["cap0"] != a["cap_after"] or b["cap0"] != b["cap_after"]:
                moved += 1
                continue
            if a["mode"] == b["mode"]:
                moved += 1
                continue
            off = a if a["mode"] == "off" else b
            asy = b if a["mode"] == "off" else a
            stable.append({"file": f, "off": off, "async": asy,
                           "cap": a["cap_after"]})

    print(f"valid pairs {len(stable)}, cap moved between the two runs {moved}, "
          f"failed/refused {refused}, no RESULT line {incomplete}")
    print()

    def tier(cap):
        return "fast" if cap == 3628800 else ("capped" if cap == 2112000 else str(cap))

    for t in ("fast", "capped"):
        sel = [s for s in stable if tier(s["cap"]) == t]
        if not sel:
            continue
        print(f"== {t} cap ({len(sel)} valid pairs) ==")
        print(f"{'block':22s} {'off ms':>9s} {'async ms':>9s} {'async/off':>10s} {'steps':>6s}")
        ratios = []
        for s in sel:
            r = s["async"]["ms"] / s["off"]["ms"]
            ratios.append(r)
            print(f"{s['file']:22s} {s['off']['ms']:9.2f} {s['async']['ms']:9.2f}"
                  f" {r:10.4f} {s['async']['steps']:6d}")
        med = statistics.median(ratios)
        print(f"median async/off = {med:.4f} -> "
              f"{'async faster by' if med < 1 else 'async slower by'} "
              f"{abs(1 - med) * 100:.1f}%")
        print()


if __name__ == "__main__":
    main()
