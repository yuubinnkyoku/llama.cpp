import io
import re
import sys
import statistics

# Phase 54 - analyse the ABBA A/B of the iq3_xxs shared decode kernel.
#
# The device rebooted before this run, so it started cool and the clock cap
# drifted downward through the whole session instead of sitting at the floor.
# A pair whose cap moved between its two arms is therefore not a controlled
# comparison and is reported but excluded from the ratio.
#
# Two metrics are reported per pair:
#   prefill128_ms  absolute, cap sensitive
#   prefill/decode ratio  both measured inside the same run, so a cap that is
#                         constant within the run cancels out

path = sys.argv[1] if len(sys.argv) > 1 else "p54-ab.log"

runs = []
with io.open(path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if not line.startswith("P54_RUN "):
            continue
        d = {}
        for m in re.finditer(r"(\w+) (\S+)", line[len("P54_RUN "):]):
            d[m.group(1)] = m.group(2)
        runs.append(d)

if not runs:
    print("no P54_RUN lines found")
    raise SystemExit(1)

print("runs: %d" % len(runs))
print()
hdr = "%-4s %-4s %-4s %-4s %-9s %-9s %-11s %-11s %-10s %-8s" % (
    "pair", "ord", "arm", "skel", "cap_start", "cap_end", "prefill128", "prefill256", "decode", "pf/dec")
print(hdr)
print("-" * len(hdr))

pairs = {}
for r in runs:
    pf128 = r.get("prefill128_ms", "NA")
    dec = r.get("decode_ms", "NA")
    ratio = "NA"
    try:
        ratio = "%.1f" % (float(pf128) / float(dec))
    except (ValueError, ZeroDivisionError):
        pass
    print("%-4s %-4s %-4s %-4s %-9s %-9s %-11s %-11s %-10s %-8s" % (
        r.get("pair"), r.get("order"), r.get("arm"), r.get("skel"),
        r.get("cap0_start"), r.get("cap0_end"),
        pf128, r.get("prefill256_ms", "NA"), dec, ratio))
    pairs.setdefault(r.get("pair"), {})[r.get("arm")] = r

print()
print("per pair, new/old (lower is better for time, higher for ratio):")
print("%-4s %-22s %-22s %-10s %-10s %-8s" % (
    "pair", "cap old", "cap new", "pf128 n/o", "pf/dec n/o", "verdict"))
print("-" * 84)

usable = []
for p in sorted(pairs, key=lambda x: int(x)):
    a = pairs[p]
    if "old" not in a or "new" not in a:
        print("%-4s incomplete" % p)
        continue
    o, n = a["old"], a["new"]
    cap_o = "%s->%s" % (o.get("cap0_start"), o.get("cap0_end"))
    cap_n = "%s->%s" % (n.get("cap0_start"), n.get("cap0_end"))
    try:
        pf_o = float(o["prefill128_ms"]); pf_n = float(n["prefill128_ms"])
        r_o = pf_o / float(o["decode_ms"]); r_n = pf_n / float(n["decode_ms"])
    except (KeyError, ValueError, ZeroDivisionError):
        print("%-4s unparsable" % p)
        continue
    same_cap = (o.get("cap0_start") == n.get("cap0_start") == o.get("cap0_end") == n.get("cap0_end"))
    verdict = "same cap" if same_cap else "cap differs"
    print("%-4s %-22s %-22s %-10.3f %-10.3f %-8s" % (
        p, cap_o, cap_n, pf_n / pf_o, r_n / r_o, verdict))
    if same_cap:
        usable.append((p, pf_n / pf_o, r_n / r_o))

print()
if usable:
    pfs = [u[1] for u in usable]
    rts = [u[2] for u in usable]
    print("usable pairs (identical cap in both arms): %d" % len(usable))
    print("  prefill128 new/old  median %.3f  range %.3f..%.3f" % (
        statistics.median(pfs), min(pfs), max(pfs)))
    print("  pf/dec     new/old  median %.3f  range %.3f..%.3f" % (
        statistics.median(rts), min(rts), max(rts)))
    print("  (below 1.0 means the new kernel is faster)")
else:
    print("no pair had an identical cap in both arms; the cap drifted through the")
    print("whole session because the device had just rebooted and was still cool.")
    print("The pf/dec column is still informative because both numbers come from")
    print("the same run, but it is not a substitute for a controlled pair.")
