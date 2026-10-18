# Phase 16 - 02 Gather contract and dispatch

## Control flow
- Env GGML_HEXAGON_IQ2S_GATHER=0 (default): opt_iq2s_gather=0, HTP_OPFLAGS_IQ2S_GATHER not set.
- Env =1: flag set on op context; VTCM codebook copy ensured; gather path active.

## What changes between OFF and ON
| Item | OFF | ON |
|---|---|---|
| Codebook source | .rodata iq2s_grid (scalar) | VTCM ctx->iq2s_grid (gather) |
| IQ2 family (XS/XXS) codebook | .rodata iq2xs_grid/iq2xxs_grid | VTCM ctx->iq2xs_grid/iq2xxs_grid |
| IQ2 family signs | .rodata ksigns64 | VTCM ctx->iq2f_signs |
| Scratch | NULL (scalar path) | per-thread VTCM slot (IQ2S_SCRATCH_STRIDE=512 bytes) |
| Unpack function | iq2_s_unpack_group_8k (scalar loop) | iq2_s_unpack_group_8k_gather (Q6_vgather) |

## Dispatch conditions (ggml-hexagon.cpp ~6878)
- IQ2_XS HTP enrollment requires opt_iq2s_gather (disabled when gather off for IQ2_XS/XXS in that guard).
- IQ2_S itself is enrolled regardless; gather flag only switches unpack path.

## Quantization semantics (IQ2_S)
- Block layout: QK_K=256 values per block; 8 sub-blocks of 32? Actually 4 groups of 8 magnitudes per 32-value chunk; see iq2-s-repack.
- Scale: d * (nibble + 0.5) * 0.25.
- Codebook: 1024 entries x 8 int8 magnitudes {8,25,43}; index = idx | ((qh >> (2*l)) & 3) << 8.
- Sign: 4-bit sign mask per 8 magnitudes; iq2_s_apply_sign4 branchless.
- Accumulation: fp32 (HVX vrmpy + scale).

## HVX gather path (hvx-mm-kernels-tiled.h 622-670)
- Q6_vgather_ARMVw loads codebook magnitudes and sign LUT from VTCM into per-thread scratch.
- grid_mu = IQ2S_GRID_BYTES-1 (byte offset of last valid byte).
- lut_mu = IQ2S_LUT_BYTES-1.
- Known risk surface: gather Mu is byte offset of last valid byte (not region size); index bounds must stay within grid.

## VTCM layout
- Reserve: HTP_IQ2S_GRID_VTCM_RESERVE at top of ctx->vtcm (size shrunk).
- IQ2_S codebook + LUT first; then per-thread scratch (128-aligned, 512B/thread).
- IQ2 family tables after scratch; IQ3_XXS/IQ3_S further.
- All regions 128-byte aligned; static asserts enforce fit.

## Correctness-relevant invariants
- gather index must be < 1024 (grid_index).
- grid_mu covers whole codebook region.
- scratch per-thread, no cross-thread aliasing.
- VTCM contents lost on release -> refresh copies bytes back.

## Performance A/B requirement
- Same binary, same libs; only env var changes.
- Record binary/skel SHA256, shape, activation, thread count, MM_SELECT, etc.

## Conclusion
- Gather ON is a real, flag-gated code path with distinct kernel identity (Q6_vgather vs scalar .rodata).
- OFF is NOT a different quantization; same math, different lookup mechanism.
- Correctness of gather path must be established per shape (Stage C/F).
