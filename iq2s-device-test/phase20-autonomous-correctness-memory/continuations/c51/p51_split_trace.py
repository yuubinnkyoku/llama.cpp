#!/usr/bin/env python3
"""Phase 51 - read GGML_HEXAGON_SPLIT_TRACE lines and report the split cost.

Each line is one split MUL_MAT of one decode step:

  ggml-hex: split <name> <mode> t_submit_begin .. t_wait_end submit_us .. cpu_us .. wait_us .. op_us ..

submit_us  host time to enqueue the DSP share (returns before the DSP finishes)
cpu_us     host time of the CPU share; it runs while the DSP share is running
wait_us    host time blocked on the DSP reply after the CPU share returned
op_us      wall time of the whole op from the first submit to the last wait

wait_us == 0 means the DSP share finished before the CPU share, so the DSP work
was fully hidden. wait_us > 0 means the DSP share is the critical path.

Usage: python p51_split_trace.py <stderr.txt> [<stderr.txt> ...]
"""
import re
import sys
import pathlib
import statistics

LINE = re.compile(
    r"split (?P<name>\S+) (?P<mode>nosync|seq|async) t_submit_begin (?P<ts0>\d+)"
    r" t_submit_return (?P<ts1>\d+) t_cpu_begin (?P<tc0>\d+) t_cpu_end (?P<tc1>\d+)"
    r" t_wait_begin (?P<tw0>\d+) t_wait_end (?P<tw1>\d+) submit_us (?P<sub>\d+)"
    r" cpu_us (?P<cpu>\d+) wait_us (?P<wait>\d+) op_us (?P<op>\d+)")


def main():
    rows = []
    for arg in sys.argv[1:]:
        p = pathlib.Path(arg)
        if not p.exists():
            print(f"missing {p}")
            continue
        for line in p.read_text(errors="replace").splitlines():
            m = LINE.search(line)
            if m:
                rows.append({k: (int(v) if k not in ("name", "mode") else v)
                             for k, v in m.groupdict().items()})
    if not rows:
        print("no split trace lines")
        return

    names = sorted({r["name"] for r in rows})
    steps = len(rows) // len(names) if names else 0
    print(f"{len(rows)} split ops, {len(names)} distinct tensors, about {steps} steps")
    print("a stall can only add time, so the minimum over the steps is the estimator")
    print()
    print(f"{'tensor':34s} {'n':>3s} {'submit':>7s} {'cpu':>8s} {'wait':>8s} {'op':>8s} {'hidden':>7s}")
    tot_cpu = tot_wait = tot_op = 0
    for n in names:
        sel = [r for r in rows if r["name"] == n]
        cpu = min(r["cpu"] for r in sel)
        wait = min(r["wait"] for r in sel)
        op = min(r["op"] for r in sel)
        sub = min(r["sub"] for r in sel)
        hidden = 100.0 * cpu / (cpu + wait) if cpu + wait else 0.0
        print(f"{n:34s} {len(sel):3d} {sub:7.0f} {cpu:8.0f} {wait:8.0f} {op:8.0f} {hidden:6.1f}%")
        tot_cpu += cpu
        tot_wait += wait
        tot_op += op
    print()
    print(f"sum of minima over {len(names)} tensors, per decode step: cpu {tot_cpu:.0f} us, "
          f"wait {tot_wait:.0f} us, op {tot_op:.0f} us")
    n_wait0 = sum(1 for r in rows if r["wait"] == 0)
    print(f"ops with wait_us == 0: {n_wait0}/{len(rows)}")


if __name__ == "__main__":
    main()
