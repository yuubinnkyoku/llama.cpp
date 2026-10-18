# Phase 16 - 07 27B performance projection

## Historical baseline (Phase 2G)
- Decode: 9642.94 ms/token (~0.104 tok/s)
- IQ2_S DSP kernels: 7259.961 ms (61 invocations, 99.3% of DSP kernel time)
- Other DSP: ~49.445 ms
- Wall - DSP total: ~2370.5 ms
- OPBATCH submits: 302

## Projection (conditional, NOT measured)
Using historical IQ2_S cumulative time replaced by measured gather speedup. Synthetic small-shape speedup (4-7x) must NOT be applied unconditionally to all 61 invocations.

| IQ2_S speedup | 27B ms/token | relative |
|---|---:|---:|
| 1x (no change) | 9643 | 1.00x |
| 2x | ~6050 | 1.60x |
| 4x | ~4240 | 2.29x |
| 7x | ~3460 | 2.80x |
| 9x | ~3230 | 3.00x |
| 12x | ~3020 | 3.20x |

These are conditional trial calculations, not current measurements.

## Residual after IQ2_S speedup
- Host residual ~2370 ms (submit/RPC/CPU fallback/attention/other) becomes dominant once IQ2_S is fast.
- Next bottleneck likely host orchestration, not IQ2_S.

## Status
PHASE16_27B_SPEEDUP_POTENTIAL_CONFIRMED: NOT YET (correctness + A/B unmeasured).
PHASE16_27B_SPEEDUP_INCONCLUSIVE: YES (until measured).
