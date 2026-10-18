# Phase 16 - 05 27B large-shape correctness

## Status
Stage C PASS achieved (37/37 IQ2_S backend-ops, OFF and ON). 27B-specific large shapes not individually run in this session. Coverage inherited from backend-ops shape set (m/n/k ranges including chunk/tail/strided).

## Plan
- For each representative 27B IQ2_S shape family: M=1 decode, representative M>1 prefill, 32x1/32x2 dispatch, chunk boundary, tail, 4D if used.
- Compare CPU reference vs Gather OFF vs Gather ON.
- Full output comparison, NaN/Inf check, DSP/DMA error check, VTCM bounds.

## Coverage requirement
- Partial success on some shapes does not justify declaring all IQ2_S gather safe.
- Unsupported shape families must be listed explicitly.

## Verdict
NOT RUN. Blocked on Stage C (correctness root cause) and device matmul harness.
