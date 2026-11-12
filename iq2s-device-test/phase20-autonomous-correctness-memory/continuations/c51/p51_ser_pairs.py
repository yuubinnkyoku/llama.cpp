#!/usr/bin/env python3
"""Phase 51 - turn the SER lines of p51_series.sh into pairs and ratios.

Runs are paired with their neighbour: off then async, or async then off. That works
for one block of four and for several two run calls with a cooldown in between. A pair counts only when both runs exited 0, both report a
minimum step, and both spent the run in one and the same cap tier; a run whose cap
samples cover two tiers is dropped, because the CPU frequency decides the balance
between the host share and the DSP share.

Usage: python p51_ser_pairs.py <log-with-SER-lines> [--tier 2112000_1862400]
"""
import re
import sys
import statistics
import pathlib

RUN = re.compile(r"^SER (?P<tag>\S+) mode (?P<mode>\w+) frac (?P<frac>\S+) exit (?P<exit>\S+)"
                 r" min_ms (?P<min>[\d.]+) swap (?P<swap>[\d.]+) caps (?P<caps>.*?) steps")
LOGITS = re.compile(r"^SER (?P<tag>\S+) logits (?P<h>[0-9a-f]{64})")
CAPS = re.compile(r"\b(\d+)_(\d+) x(\d+)")


def main():
    path = sys.argv[1]
    want_tier = None
    if "--tier" in sys.argv:
        want_tier = sys.argv[sys.argv.index("--tier") + 1]

    runs = {}
    order = []
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        m = RUN.match(line)
        if m:
            tag = m["tag"]
            caps = CAPS.findall(m["caps"])
            runs.setdefault(tag, {})["caps"] = caps
            runs[tag].update(mode=m["mode"], frac=m["frac"], exit=m["exit"],
                             min_ms=float(m["min"]), swap=int(m["swap"]))
            if tag not in order:
                order.append(tag)
            continue
        m = LOGITS.match(line)
        if m and m["tag"] in runs:
            runs[m["tag"]]["logits"] = m["h"]

    if not runs:
        print("no SER lines")
        return

    def tier_of(tag):
        caps = runs[tag].get("caps") or []
        caps = [c for c in caps if int(c[2]) > 0]
        if not caps:
            return None
        biggest = max(caps, key=lambda c: int(c[2]))
        total = sum(int(c[2]) for c in caps)
        # one stray sample at the end is allowed, two tiers with real weight are not
        if int(biggest[2]) * 10 < total * 9:
            return None
        return f"{biggest[0]}_{biggest[1]}"

    print(f"{'tag':16s} {'mode':6s} {'exit':4s} {'min_ms':>8s} {'swap_pg':>8s} tier")
    for tag in order:
        r = runs[tag]
        print(f"{tag:16s} {r['mode']:6s} {r['exit']:4s} {r['min_ms']:8.1f} {r['swap']:8d}"
              f" {tier_of(tag)}")

    # pair each run with the one next to it: off then async, or async then off
    valid = []
    dropped = 0
    pairs = []
    i = 0
    while i + 1 < len(order):
        a, b = order[i], order[i + 1]
        if {runs[a]["mode"], runs[b]["mode"]} == {"off", "async"}:
            pairs.append((a, b) if runs[a]["mode"] == "off" else (b, a))
            i += 2
        else:
            i += 1
    for off_tag, async_tag in pairs:
        if True:
            o, a = runs[off_tag], runs[async_tag]
            if o["exit"] != "0" or a["exit"] != "0" or o["min_ms"] <= 0 or a["min_ms"] <= 0:
                dropped += 1
                continue
            to, ta = tier_of(off_tag), tier_of(async_tag)
            if to is None or to != ta:
                dropped += 1
                continue
            if want_tier and to != want_tier:
                dropped += 1
                continue
            valid.append((off_tag, async_tag, to, o["min_ms"], a["min_ms"]))

    print()
    if not valid:
        print(f"no valid pair ({dropped} dropped)")
        return

    print(f"{'pair':34s} {'tier':20s} {'off ms':>8s} {'async ms':>8s} {'async/off':>9s}")
    ratios = []
    for off_tag, async_tag, tier, om, am in valid:
        r = am / om
        ratios.append(r)
        print(f"{off_tag + ' / ' + async_tag:34s} {tier:20s} {om:8.1f} {am:8.1f} {r:9.4f}")
    med = statistics.median(ratios)
    print()
    print(f"{len(valid)} valid pairs, {dropped} dropped")
    print(f"median async/off = {med:.4f} -> "
          f"{'async faster' if med < 1 else 'async slower'} by {abs(1 - med) * 100:.1f}%")

    async_hashes = {runs[t[1]].get("logits") for t in valid}
    print(f"async logits hashes: {sorted(h for h in async_hashes if h)}")


if __name__ == "__main__":
    main()
