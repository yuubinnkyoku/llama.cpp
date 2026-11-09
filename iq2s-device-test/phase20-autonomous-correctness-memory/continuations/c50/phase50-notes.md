# Phase 50 measurement notes (running)

Working record while the gates run. The final report is `phase50-report.md`.

Branch `codex/c50-perf`, worktree `llama.cpp-c50`, based on `048e79b50`
(`codex/overnight-autonomy-phase50-checked`). HEAD after the first commit:
`f348c6100` (IQ2 gather batching).

## Skels under test

| id | sha256 | content |
|---|---|---|
| A | `a87330ad0e1739dbf7cf23c58b6935bfbc1a4b21c0d75735e608ef9e53795d49` | Phase 49.5 build (baseline) |
| B | `c0bec5e5b713895c735d1daf6eb09232837843db1ab5b5ec5aec8fe6dd0c9de7` | A + IQ2_S batched gather |
| C | `10483e5b07d50286be5f9ceb70dac9cc8c7da5280bfcf58688044dc26aecc55c` | B + IQ2_XS/XXS batched gather |

Only `libggml-htp-v81.so` changes between A, B and C. The host libraries are
unchanged, so every A/B/C comparison swaps exactly one file.

## Gate 0 - baseline regression

BF16 (`p49_bf16_test`, 2026-11-10 01:19):

```
P49BF16 exact patterns 65536 checked 65536 bad 0
P49BF16 perm bad 0
P49BF16 dot m1 K 5120 N 48 M 1 max_abs 1.5102612e-05 rel_l2 2.19e-07 over 0 nonfinite 0 ok
P49BF16 dot m2 K 5120 N 48 M 2 max_abs 1.34725684e-05 rel_l2 2.32e-07 over 0 nonfinite 0 ok
P49BF16 dot m3 K 5120 N 48 M 3 max_abs 2.3910756e-05 rel_l2 2.3e-07 over 0 nonfinite 0 ok
P49BF16 dot tailK K 5152 N 48 M 1 rel_l2 2.9e-07 over 0 nonfinite 0 ok
P49BF16 dot small K 96 N 33 M 2 rel_l2 1.47e-07 over 0 nonfinite 0 ok
P49BF16 done fail 0
```

Same values as the Phase 49.5 record. Type and shape gates pass, no non-finite.

Scheduler (`p495_sched_timing_test`):

- `mix 300 10`: every split line reports `calls 10`, `sync_us 0 copy_us 0`,
  `compute_us == total`, nodes 1 and 3 alternate. No negative and no
  `compute > total`.
- `async 15 20000`: the MOCK split reports `compute_us 49..73` for a 20 ms job,
  the consumer reports `sync_us` near 0, and `ggml-sched-wait backend MOCK
  avg_us 20984..21426`. The asynchronous submit is not counted as compute.
  `max_err 1.19e-07`.
- `GGML_SCHED_SPLIT_TIMING=0`: zero `ggml-sched-` lines, the old path.

IQ3 / 27B logits: 4 step replay, `25358 11 198 760`.

| skel | NGL | logits sha256 |
|---|---:|---|
| A | 9 | `5585d4edc47ee7505c558aeb1b929a7eb7089e238d1b2d81a3ffcfa4958bd480` |
| B | 9 | `5585d4ed...` (identical) |
| C | 9 | `5585d4ed...` (identical) |

The Phase 49 baseline for NGL=9 is `5585d4ed...`, so the IQ2 rewrite is
bit exact and Phase 49's IQ3 result is reproduced.

## Gate 2 - IQ2 gather batching

Method: `GGML_HEXAGON_PROFILE=1`, 4 step replay, NGL=9, fast cap. Per type the
logical weight bytes are summed over the M=1 `MUL_MAT` ops and divided by the
summed DSP `usec`. `p50_prof_bw.py` reproduces the Phase 49.5 table from
`../c49/p495-prof9.txt` exactly (IQ3_S 28.10, IQ3_XXS 21.99, IQ2_XS 7.81,
IQ4_XS 61.95, Q2_K 43.12, Q4_K 59.68, BF16 33.51, total 35.21 GB/s /
76.68 GMAC/s), which validates the instrument before it is used on new runs.

| type | A GB/s | B GB/s | C GB/s | C vs A |
|---|---:|---:|---:|---:|
| IQ2_S | 10.08 | 13.95 | 13.95 | +38.4% |
| IQ2_XS | 7.80 | 7.79 | 9.77 | +25.3% |
| IQ3_S | 28.14 | 27.90 | 28.13 | -0.0% |
| IQ3_XXS | 22.01 | 21.87 | 22.00 | -0.0% |
| IQ4_XS | 66.17 | 64.62 | 63.45 | -4.1% |
| Q2_K | 44.80 | 44.72 | 43.89 | -2.0% |
| Q4_K | 62.35 | 61.21 | 60.66 | -2.7% |
| BF16 | 34.80 | 34.59 | 34.29 | -1.5% |
| TOTAL | 34.69 | 34.93 | 35.27 | +1.7% |

DSP time for the changed types:

| type | A usec | C usec | change |
|---|---:|---:|---:|
| IQ2_S | 15026 | 10855 | -27.8% |
| IQ2_XS | 13209 | 10551 | -20.1% |

IQ3_S and IQ3_XXS are untouched by the change and land within 0.1%, which is
the internal control: the difference is not a global shift in the run. IQ2_S
holds at 13.95 in B and C because the family change does not touch it, a second
control.

IQ2_XXS does not appear on the DSP at NGL=9 (it is on the CPU layers there), so
its bandwidth is still unmeasured.

## Gate 3 - NGL

Overrides: `override-base.txt` for NGL=0, `override-l57.txt` (which forces
`blk.57.ffn_up.weight` to HTP) for NGL>0.

Valid same-cap series, tier `cap0 2112000 / cap7 1862400`, 8 steps, noreplay,
median of step 1..7:

| tag | NGL | median ms | min | max | state |
|---|---:|---:|---:|---:|---|
| c3a | 0 | 714.97 | 668.68 | 845.76 | same |
| c1b | 16 | 667.20 | 607.94 | 681.78 | same |
| c2b | 16 | 621.07 | 605.03 | 3230.44 | same |
| c3b | 16 | 634.96 | 606.60 | 676.57 | same |
| c2c | 24 | 619.69 | 601.01 | 817.07 | same |
| c3c | 24 | 614.68 | 593.30 | 644.45 | same |
| c4a | 24 | 705.76 | 600.53 | 1298.61 | same |
| c4c | 24 | 659.81 | 605.01 | 797.68 | same |

Medians of the valid runs: NGL=0 714.97 (n=1), NGL=16 634.96 (n=3),
NGL=24 639.74 (n=4).

- NGL=16 vs NGL=0: -11.2%
- NGL=24 vs NGL=0: -10.5%
- NGL=24 vs NGL=16: +0.8% (inside the spread, not a gain)

NGL=32 stopped with `exit 6` (the swap-out guard) at the same cap, so it is not
viable and was not retried.

## Environment facts learned this session

- `cpuinfo_max_freq` is 3628800 (policy0, cores 0-5) and 4608000 (policy6,
  cores 6-7). There are only two cpufreq policies, not the three the Phase 49
  notes imply (`policy0`/`policy4`/`policy7` are per-cpu symlinks onto the same
  two policies).
- The vendor cap moves in steps: 3628800, 3513600, 3187200, 2361600, 2112000,
  and 883200 in Dozing. It can change inside one run.
- The cap is **not** applied by the Android thermal framework: every cooling
  device reports `cur_state 0`, `dumpsys thermalservice` says `Thermal Status: 0`,
  and the hardware trip points (105 C) are far away. It is a vendor power
  policy.
- Cooling the device does not release the cap. 5 minutes idle took `tz0` from
  35.3 C to 31.5 C and the cap stayed at 2112000/1862400. A screen off/on cycle
  and `svc power stayon true` did not release it either.
- `tz0` reaches 96-98 C inside an 8 step run at the fast cap, so the boost
  window is short lived.
- `run50.sh` is `run49.sh` with one line changed: the expected skel hash comes
  from `WANT_SKEL` and defaults to the Phase 49 value. `run_split50.sh` is
  `run_split49.sh` calling `run50.sh`. `run49.sh` is untouched and
  `run49.sh.p49` is kept beside it.
- `test-backend-ops perf` is not usable on this backend: one
  `MUL_MAT(iq2_s, m=17408, n=1, k=5120)` reports 121638 us/run, which is the
  per-run weight upload, not the kernel (the same op is ~500-1500 us in a real
  run).
