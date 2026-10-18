# Phase 16 - 09 Regression

## Scope
- IQ2_S: test-backend-ops HTP0 MUL_MAT iq2_s: 37/37 PASS (OFF and ON).
- IQ2_XS/IQ2_XXS/IQ3_XXS/IQ3_S/Q2_K/IQ4_XS: not re-run in this session (no source changes made).
- Host repack test (test-hexagon-iq2-s-repack): PASS on device.

## Pre-existing (do not confuse)
- Phase 15-A test-backend-ops: 10667/10679, f16 threshold 12 FAIL + harness abort. Pre-existing; not attributed to this phase (no source changes).

## Abnormality checklist
- NaN/Inf, OOB, DSP fault, FastRPC failure, DMA failure, VTCM corruption, unexpected fallback, dispatch mismatch: none observed (37/37 PASS both modes).

## Verdict
PHASE16_IQ2S_REGRESSION_PASS: PASS for IQ2_S (host repack + backend-ops). Broader family regression not run (no shared-code changes).
