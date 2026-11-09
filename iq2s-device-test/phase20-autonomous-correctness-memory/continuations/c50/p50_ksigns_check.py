#!/usr/bin/env python3
"""Phase 50 - check that ksigns64 is the even-parity expansion of a 7-bit index.

IQ2_XS / IQ2_XXS fetch the sign mask with two vgather per group. If the table is
pure structure, the mask can be built with arithmetic instead, like the IQ3_XXS
sign mask already is.

Claim under test: byte j of ksigns64[i] is 0xff for j in 0..6 iff bit j of i is
set, and byte 7 is 0xff iff the parity of i is odd.
"""
import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "ggml/src/ggml-common.h"


def load():
    text = open(SRC, encoding="utf-8", errors="replace").read()
    m = re.search(r"GGML_TABLE_BEGIN\(uint64_t, ksigns64, 128\)(.*?)GGML_TABLE_END", text, re.S)
    if not m:
        raise SystemExit("ksigns64 not found")
    vals = [int(t, 16) for t in re.findall(r"0x([0-9a-fA-F]+)", m.group(1))]
    if len(vals) != 128:
        raise SystemExit(f"expected 128 entries, got {len(vals)}")
    return vals


def predicted(i):
    mask = 0
    for j in range(7):
        if i >> j & 1:
            mask |= 0xFF << (8 * j)
    if bin(i).count("1") & 1:
        mask |= 0xFF << (8 * 7)
    return mask


def main():
    vals = load()
    bad = []
    for i, v in enumerate(vals):
        if v != predicted(i):
            bad.append((i, v, predicted(i)))
    print(f"entries {len(vals)} mismatches {len(bad)}")
    for i, got, want in bad[:8]:
        print(f"  i={i} got 0x{got:016x} want 0x{want:016x}")
    # also confirm the table never has a partially set byte
    partial = [i for i, v in enumerate(vals) if any((v >> (8 * j)) & 0xFF not in (0, 0xFF) for j in range(8))]
    print(f"partial bytes {len(partial)}")
    print("STRUCTURE_OK" if not bad and not partial else "STRUCTURE_MISMATCH")
    return 0 if not bad and not partial else 1


if __name__ == "__main__":
    sys.exit(main())
