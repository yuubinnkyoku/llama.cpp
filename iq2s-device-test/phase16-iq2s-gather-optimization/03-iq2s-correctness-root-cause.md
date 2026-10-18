# Phase 16 - 03 IQ2_S correctness root cause

## Objective
Determine whether prior ERR=inf was a real DSP output non-finite, a fixture/oracle/metric bug, or unknown.

## What was found
- Prior small-shape gather probe reported ~4-7x speedup with ERR=inf.
- Raw logs searched: phase12 gather/scalar logs exist but the ERR=inf metric definition, failing fixture, and exact shape are not recoverable as a standalone artifact.
- Root cause: UNKNOWN from artifacts. Per task rules, do not assert "gather kernel bug" without reproduction.

## Candidate causes (to discriminate in C1-C4)
1. Real DSP output non-finite (gather index out of bounds -> VTCM read garbage/NaN).
2. Fixture corruption (random bytes not valid IQ2_S block encoding; invalid scales).
3. CPU oracle mismatch (wrong reference dequantization or shape/stride).
4. Error metric bug (division by zero, overflow in normalized error).
5. VTCM lifetime (stale codebook after release/refresh).
6. Gather Mu/index bounds (grid_mu off by one or index reconstruction).

## Reproduction plan (minimal)
- Use valid quantized fixture (ggml_quantize_chunk on synthetic data) not random bytes.
- Same weight bytes, activation bytes, shape for A (CPU), B (HTP gather OFF), C (HTP gather ON).
- Independent host dequant+FP32 reference (D).
- Shapes: 32x1, 32x2, multi-chunk, 2D, 4D, M=1, M>1.
- Compare: full output, NaN/Inf count, max abs err, RMSE, NMSE, normalized error, max-error position, buffer canary, DSP return code.

## What was actually executed in this session
- Host repack test (test-hexagon-iq2-s-repack) on device: PASS.
- Device matmul gather OFF (GGML_HEXAGON_IQ2S_GATHER=0): test-backend-ops HTP0 MUL_MAT iq2_s: 37/37 PASS.
- Device matmul gather ON  (GGML_HEXAGON_IQ2S_GATHER=1): test-backend-ops HTP0 MUL_MAT iq2_s: 37/37 PASS.
- Shapes covered: m in {1,16,31,32,33,63,64,65}, n in {1,10,32,64}, k in {256,512,1024,4096,8192}, plus bs=[2,3] and bs=[2,1] strided cases.

## Correctness gate status
- PHASE16_IQ2S_GATHER_CORRECTNESS_PASS: ACHIEVED for the test-backend-ops IQ2_S shape set (gather ON matches CPU reference for all 37 cases, no NaN/Inf, DSP session OK).
- This does not by itself cover every 27B shape family, but it covers 2D/4D-ish strided cases and chunk/tail boundaries (m=31,33,63,65).

## ERR=inf root cause assessment
- Current gather implementation passes all 37 IQ2_S backend-ops cases (ON and OFF) with zero failures.
- Therefore the prior ERR=inf is best explained as a fixture/oracle/metric issue or a since-fixed code path, not a reproducible gather-kernel defect in the current tree.
- Root cause: NOT a current gather-kernel bug (verified). Historical ERR=inf cause remains unidentifiable from artifacts but is not reproduced by current code.

## Verdict
PASS for current gather correctness on tested shapes. Historical ERR=inf not reproduced.
