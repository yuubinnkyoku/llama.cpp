# Phase 16 - 06 IQ2_S gather OFF/ON performance A/B

## Status
Correctness verified (37/37 IQ2_S cases PASS for both OFF and ON). Per-op wall timing not captured: test-backend-ops has no IQ2_S perf cases in this tree, so the planned A/B timing loop could not be run via that harness.

## What was run
- OFF: GGML_HEXAGON_IQ2S_GATHER=0 -> 37/37 PASS.
- ON:  GGML_HEXAGON_IQ2S_GATHER=1 -> 37/37 PASS.
- Same binary/libggml-htp-v81.so; only env var changed (same-binary A/B satisfied for correctness).

## Prior small-shape probe (reference only, not a 27B measurement)
- K=256, M=16: scalar 165 us vs gather 40 us (~4x). From phase15 artifacts.
- This is a small-shape number; not applied to 27B invocations.

## Verdict
Same-binary A/B correctness PASS. Performance speedup not freshly measured in this session; historical small-shape ~4x is the only speedup evidence. Production performance claim withheld.
