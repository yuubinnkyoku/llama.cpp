#!/usr/bin/env python3
"""Phase 51 - how much a per tensor CPU frac would recover for the row split.

Two inputs from the same model, NGL and step count:
  trace   stderr of a run with GGML_HEXAGON_SPLIT_TRACE=1, split enabled
  profile stderr of a run with GGML_HEXAGON_PROFILE=1, split disabled

The trace gives cpu_us, the host time of the CPU share at the frac the run used.
cpu_us = frac * C, so C, the full CPU time of that tensor, follows. The profile
gives D, the full DSP time of the same tensor. With both, one op costs
max(f*C, (1-f)*D), and the f that minimises it is D/(C+D).

Usage: python p51_frac_model.py <trace.stderr> <profile.stderr> <frac>
"""
import re
import sys
import pathlib
import statistics

TRACE = re.compile(r"split (?P<name>\S+) async t_submit_begin \d+ t_submit_return \d+"
                   r" t_cpu_begin \d+ t_cpu_end \d+ t_wait_begin \d+ t_wait_end \d+"
                   r" submit_us (?P<sub>\d+) cpu_us (?P<cpu>\d+) wait_us (?P<wait>\d+)"
                   r" op_us (?P<op>\d+)")
PROF = re.compile(r"profile-op MUL_MAT\|(?P<name>\S+)\.weight x \S+ -> \S+"
                  r"\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|usec (?P<usec>\d+)")


def main():
    trace_path, prof_path, frac = sys.argv[1], sys.argv[2], float(sys.argv[3])
    cpu = {}
    op = {}
    for line in pathlib.Path(trace_path).read_text(errors="replace").splitlines():
        m = TRACE.search(line)
        if m:
            name = m["name"]
            if name.endswith(".weight"):
                name = name[:-len(".weight")]
            cpu.setdefault(name, []).append(int(m["cpu"]))
            op.setdefault(name, []).append(int(m["op"]))

    dsp = {}
    steps = {}
    for line in pathlib.Path(prof_path).read_text(errors="replace").splitlines():
        m = PROF.search(line)
        if not m or " x " not in line:
            continue
        if not re.search(r" x \d+:1 -> ", line):
            continue  # keep decode steps only
        dsp[m["name"]] = dsp.get(m["name"], 0) + int(m["usec"])
        steps[m["name"]] = steps.get(m["name"], 0) + 1

    print(f"{'tensor':28s} {'C us':>8s} {'D us':>8s} {'ideal f':>8s}"
          f" {'now us':>8s} {'ideal us':>9s} {'gain us':>8s}")
    tot_now = tot_ideal = 0.0
    rows = []
    for name in cpu:
        if name not in dsp or steps[name] == 0:
            continue
        D = dsp[name] / steps[name]
        C = statistics.median(cpu[name]) / frac
        now = statistics.median(op[name])
        f = D / (C + D) if C + D else 0.5
        f = min(max(f, 0.02), 0.98)
        ideal = max(f * C, (1.0 - f) * D)
        rows.append((name, C, D, f, now, ideal))
        tot_now += now
        tot_ideal += ideal
        print(f"{name:28s} {C:8.0f} {D:8.0f} {f:8.3f} {now:8.0f} {ideal:9.0f} {now - ideal:8.0f}")
    print()
    print(f"sum now {tot_now:.0f} us/step, sum ideal {tot_ideal:.0f} us/step, "
          f"recoverable {tot_now - tot_ideal:.0f} us/step")
    mix = sum(r[2] for r in rows) / (sum(r[1] for r in rows) + sum(r[2] for r in rows))
    print(f"aggregate DSP side share D/(C+D) of the whole list = {1-mix:.3f} "
          f"=> one global frac would be about {mix:.3f}")


if __name__ == "__main__":
    main()
