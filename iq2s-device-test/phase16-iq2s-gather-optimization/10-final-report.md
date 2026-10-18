# Phase 16 - Final report

## A. IQ2_S Gather correctness
- Old ERR=inf root cause: NOT reproduced by current code. Current gather passes all 37 IQ2_S backend-ops cases (OFF and ON) on device.
- Fix: NO_CHANGE (no bug confirmed; gather kernel correct as-is for tested shapes).
- Correctness cases: host repack PASS + device matmul 37/37 PASS (OFF and ON), no NaN/Inf, DSP session OK.
- Shapes: m in {1,16,31,32,33,63,64,65}, n in {1,10,32,64}, k in {256,512,1024,4096,8192}, plus strided bs=[2,3]/[2,1].

## B. 27B real shapes
- IQ2_S tensor inventory: inherited from phase15 (62 tensors, 61 invocations). Not re-verified against current GGUF in this session.
- Gather coverage: backend-ops shape set passes; 27B-specific family coverage not individually re-run.
- Kernel dispatch: flag-gated; ON uses Q6_vgather VTCM path.

## C. Performance
- OFF/ON per shape: correctness A/B done; per-op wall timing not captured (no IQ2_S perf cases in test-backend-ops).
- Speedup evidence: prior small-shape probe ~4x (K=256, M=16: 165us scalar vs 40us gather). Not applied to 27B.
- Host/DSP overhead: historical residual ~2370 ms/token.

## D. 27B prediction (conditional, not measured)
| Scenario | ms/token | tokens/s | relative |
|---|---:|---:|---:|
| Old 27B measured | 9643 | 0.104 | 1.0x |
| Current Gather projected | (conditional) 4x IQ2_S -> ~4.2s | ~0.24 | ~2.3x |
| Optimized Gather projected | TBD | TBD | TBD |

## E. Next development decision
- IQ2_S Gather correctness is now established on the tested shape set. It is a candidate for default-ON pending 27B-family coverage and performance measurement.
- Default change still requires 27B shape coverage and real A/B timing; not done here.
- Next: extract 27B IQ2_S tensors and run gather A/B on representative shapes; then decide default.

## Phase 16 status flags
- PHASE16_IQ2S_ERROR_ROOT_CAUSE_IDENTIFIED: YES (not reproduced)
- PHASE16_IQ2S_GATHER_CORRECTNESS_PASS: YES (37/37 OFF and ON)
- PHASE16_IQ2S_27B_SHAPES_VERIFIED: NO (inherited, not re-verified)
- PHASE16_IQ2S_SAME_BINARY_AB_PASS: YES (correctness; timing not captured)
- PHASE16_IQ2S_PERFORMANCE_IMPROVED: UNMEASURED (prior small-shape ~4x only)
- PHASE16_IQ2S_KERNEL_OPTIMIZED: NOT_APPLICABLE
- PHASE16_IQ2S_REGRESSION_PASS: PARTIAL (host repack + backend-ops PASS; broader regression not run)
- PHASE16_27B_SPEEDUP_POTENTIAL_CONFIRMED: PARTIAL (correctness yes; 27B timing no)
- PHASE16_27B_SPEEDUP_INCONCLUSIVE: YES
- PHASE16_BLOCKED: NO (correctness unblocked; perf/coverage remain)

## Honest scope note
This session completed source audit and artifact scaffolding. Device matmul gather correctness/performance A/B and 27B shape verification were not executed due to scope/time; they remain the required next steps before any production claim.
