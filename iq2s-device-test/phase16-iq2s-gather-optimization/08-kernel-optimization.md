# Phase 16 - 08 Kernel optimization

## Status
NOT PERFORMED. Requires Stage G (A/B) showing large real effect and correctness PASS.

## Candidate optimization points (if gather ON is correct and fast)
1. Gather index decode
2. Codebook VTCM lookup
3. 32x1/32x2 kernel selection
4. Weight decode reuse
5. Activation reuse
6. Loop unrolling
7. VTCM staging
8. DMA wait
9. Large-K tiling
10. Register spill
11. Worker scheduling

## Principle
- One change at a time.
- Save pre-change binary and measurements.
- Re-run host correctness, device correctness, numerical error, kernel speed, full wall, VTCM, DMA, DSP stability.
- Adopt only if correctness gate passes, regression passes, dispatch confirmed, DSP/DMA no errors, speedup exceeds noise.

## Verdict
PHASE16_IQ2S_KERNEL_OPTIMIZED: NOT_APPLICABLE (no optimization attempted).
