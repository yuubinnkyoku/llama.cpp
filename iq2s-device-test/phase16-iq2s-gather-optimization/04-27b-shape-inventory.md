# Phase 16 - 04 27B IQ2_S shape inventory

## Source of truth
- GGUF: Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf (9,259,510,912 bytes)
- SHA256 (recorded): 16c9802111aa9ef3acde465188d6d601f8db128ee3d828ad983a5caca4135ecb
- Inventory derived from phase15-final-feasibility: 62 IQ2_S weight tensors in 27B model.
- Historical profile (Phase 2G): 61 IQ2_S MATMUL invocations, 7259.961 ms total, 99.3% of DSP kernel time.

## Tensor count vs invocation count
- 62 weight tensors vs 61 runtime MUL_MAT invocations: difference explained by embedding/output or shared weights not invoked per token, and by tensors that are not on the decode MUL_MAT path (e.g. non-MUL_MAT consumers). Exact mapping not re-derived in this session; flagged as gap.

## Shape families (from prior inventory)
- Attention Q/K/V/O projections (IQ2_S where enrolled)
- FFN projections (gate/up/down)
- Large-K and Large-N variants

## Gather eligibility
- IQ2_S gather eligible when shape is repacked 2D/4D compatible (K multiple of 256, N rows).
- Unsupported: K not multiple of 256, invalid strides, non-repacked buffers.

## 27B full load prohibited
- This phase does not load the full 27B model. Only tensor metadata/shape inventory used.
- Real-weight extraction (read-only GGUF tensor slice) not performed in this session; documented as next step.

## Deliverable CSV
- iq2s-27b-shapes.csv: to be populated with per-tensor K,N,M,eligibility,dispatch. NOT populated in this session (metadata extraction not run).

## Status
PHASE16_IQ2S_27B_SHAPES_VERIFIED: NOT ACHIEVED (inventory inherited from phase15, not re-verified against current GGUF).
