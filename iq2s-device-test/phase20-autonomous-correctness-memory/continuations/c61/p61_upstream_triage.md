# Phase 61 - triage of the 61 upstream hexagon commits

Range: `HEAD..upstream/master` restricted to `ggml/src/ggml-hexagon/`.
Upstream tip at analysis time: `c622f623f` (2026-11-21 14:25).
Fork tip: `842dbb917` (Phase 60 end).

**This Phase does not merge anything.** The classification below is input for Phase 62,
which will do the intake on a dedicated branch with a regression gate.

## 0. Shape of the problem

| metric | value |
| --- | --- |
| upstream hexagon commits not in the fork | **61** |
| fork-only hexagon commits | **67** |
| hexagon files the fork changed | **23** |
| upstream commits touching `ggml-hexagon.cpp` | **32** |
| upstream commits touching `htp/matmul-ops.c` | **13** |
| upstream commits touching the two CMakeLists | **13** |

The fork and upstream have diverged in both directions. This is not a fast-forward; it is a
merge with real conflicts.

## 1. Classification

### welcome - the fork has no equivalent and the value is high

| commit | PR | what | why welcome |
| --- | --- | --- | --- |
| `ded522f83` | #31921 | SSM_SCAN op | the 27B is a qwen35 (SSM hybrid); scan on the HTP is a direct win |
| `522021325` | #30995 | conv2d support | SSM conv path |
| `669204ee2` | #31814 | conv_2d optimizations + conv_3d | follows the above |
| `31288c9c6` | #30731 | FA improvements for long context and SWA | decode quality at long ctx |
| `78a8ae14b` | #31055 | KV-head team split for multicore HMX FA | FA throughput |
| `fe8841a78` | #30218 | multistream FA | multi-seq |
| `ceefd16da` | #30581 | correct HMX FA sinks | correctness |
| `c6c0ec6d5` | #32243 | fix fa no-mask tail | correctness, 1 file |
| `ec0bfac17` | #30423 | configurable HTP DCVS/HMX power mode | thermal control on a phone |
| `76a67730f` | #31781 | automatic libhtp loading without legacy env vars | removes the env plumbing the fork relies on |
| `7aca0d2ea` | #31221 | GGML_HEXAGON_VTCM_SIZE | tuning knob |
| `846639535` | #30854 | snapdragon-toolchain v0.8 + hexagon v85 | toolchain currency |
| `cf6463ace` | #30952 | fix 64-bit FastRPC handle truncation | **already partially applied** in the fork |
| `67107ac34` | #31744 | set_tensor_n + repacking robustness | **already partially applied** in the fork |
| `b684b498b` | #30655 | fix VTCM size query with layout out-params | correctness, 1 file |
| `a91eab67d` | #31442 | fix DMA transfers with strides > 24 bits | correctness, 2 files |
| `097f5b533` | #30133 | do not assume aligned read/write with fused bias | correctness, 1 file |
| `9c2e0e491` | #30126 | enable alloc_buffer_n | 1 file |
| `06cad0b9e` | #30121 | Q6_K weight dequant speedup | the 27B has q6_K tensors |
| `e937b17a2` | #30035 | optimize q2_k bpw and activation | q2_K is in the model |
| `70815103c` | #30104 | improved GELU accuracy | quality |
| `27c7a769d` | #30435 | vectorize softplus with HVX | quality/perf, 1 file |
| `6b9c0cfcb` | #31279 | faster TOP_K for the backend sampler | decode |
| `0ae0e1976` | #30346 | SGN op | op coverage |
| `36e36e63a` | #31006 | CEIL op | op coverage |
| `50569eb87` | #29995 | pool op support | op coverage |
| `8a199ad31` | #31331 | UPSCALE op | op coverage |
| `b13a93a3d` | #30305 | IM2COL_3D | op coverage |
| `d4d82d67f` | #30189 | fix IM2COL patch-embed DMA ring overflow | correctness, 1 file |
| `2e0d86beb` | #31050 | IM2COL block-based HVX parallelization | perf |
| `da6c30ade` | #31277 | DDR bw in transposed CONCAT | perf |
| `8bb3cf0d2` | #30341 | CPY/CONCAT round 3 | perf |
| `c479922ac` | #30067 | CPY/CONCAT/CONT/DUP DMA/HVX overhaul | perf |
| `4413a1924` | #30245 | scalar DMA broadcasting for binary ops | perf |
| `51fe42c5d` | #31446 | tiled hvx permute | perf |
| `ca89968a2` | #31822 | fuse post-norm residual into RMS_NORM_MUL | perf |
| `5e03bdd87` | #29971 | ssm-conv updates | SSM |
| `9daa0aa22` | #32144 | buffer allocations and mappings (mmap stacking) | memory |
| `dca216ad6` | #32228 | TIMESTEP_EMBEDDING HVX kernel | op coverage |
| `7071ba378` | #31552 | TOP_K_MASK op | op coverage |
| `20a51f24a` | #30953 | DSV4_HC_PRE/POST ops | op coverage |
| `bee23322f` | #31289 | LIGHTNING_INDEXER support | op coverage |
| `92d6c8a5c` | #32011 | host build with clang-cl on Windows ARM64 | build, 1 file |
| `494d0f47b` | #31896 | MUL_MAT src1 precision tests and gating | test infra, 1 file |
| `d6be06557` | #31851 | follow reference precision of ggml_prec_set_acc | correctness, 1 file |
| `1fa7f1c48` | #30206 | YaRN truncate=false | unrelated to hexagon, 2 files |
| `1b6935b22` | #29510 | flash_attn_ext_rows | unrelated to hexagon, 1 file |
| `418f905a4` | #30881 | batched tensor copy API | unrelated to hexagon, 1 file |
| `701fd25b3` | #26631 | MUL_MAT_ID -1 index skips computation | 1 file |

### defer - needs a foundation change first

| commit | PR | what | why defer |
| --- | --- | --- | --- |
| `b797b9ca8` | #30635 | **tensor repacking rework** (tiled + flat quantized and fp) | 21 files; the fork's whole IQ2/IQ3 repack layer is built on the old scheme |
| `643b0cdac` | #30843 | **Op Fusion refactoring** + BINARY_NX | 70 files; the fork's split path hooks into the op dispatch |
| `e9a2a44fe` | #31857 | n_hvx and n_threads cleanup | 82 files; the fork's `P44_NTHREADS` and split thread logic assume the old model |
| `f0fb5984b` | #31071 | generic dims collapse/reshape | 8 files; interacts with the fork's split shape handling |
| `846639535` | #30854 | toolchain v0.8 | listed as welcome but it is a build-system change; do it with the toolchain bump, not alone |

### skip - the fork already has its own version, or the target is not used

| commit | PR | what | why skip |
| --- | --- | --- | --- |
| `61dbf4922` | #31954 | IQ2_XXS + NVFP4 | **the fork's IQ2_XXS HTP kernel is its own work**; taking upstream's would replace it. NVFP4 is not used by the 27B |
| `0efe8e4c6` | #30901 | neon-vectorized threaded repack | the fork has its own repack path (`iq2-family-repack.h`, `iq3-*-repack.h`) |
| `aa5e0092f` | #30115 | tiled Q4_K/Q6_K GET_ROWS | the fork's GET_ROWS handling differs; check before taking |
| `848bbda90` | #30627 | hadamard MUL_MAT + GET_ROW for Q1_0/Q2_0 | Q1_0/Q2_0 are not in the 27B |
| `797de63b2` | #30122 | Q1_0 and Q2_0 support | not in the 27B |
| `2a4daf0e0` | #32094 | Q5_0 and Q5_1 | not in the 27B |
| `caa8191df` | #31169 | MUL_MAT_ID row skipping | the fork's MoE path is its own |
| `8345f3339` | #29974 | matmul and FA scalability updates | superseded by later commits in the list |

## 2. Conflict prediction by area

The brief named three areas. Measured against the actual file lists:

| area | upstream commits touching it | fork commits touching it | risk |
| --- | --- | --- | --- |
| `ggml-hexagon.cpp` | 32 | many (split, FFN offload, IQ gathers) | **high** |
| `htp/CMakeLists.txt` + `CMakeLists.txt` | 13 | several (skel build, IQ kernels) | **high** |
| `ggml-cpu/tiled` | 0 | 4 (Phase 60) | **none** - upstream has no tiled changes in this range |

**The tiled work is safe.** No upstream commit in this range touches `ggml-cpu/tiled/`, so the
Phase 60 NEON kernel will not conflict.

## 3. Recommended intake order for Phase 62

Grouped so each step is independently testable:

1. **Correctness-only, 1-2 files each** (lowest risk, do first):
   `c6c0ec6d5`, `b684b498b`, `a91eab67d`, `097f5b533`, `9c2e0e491`, `d4d82d67f`, `cf6463ace`
2. **Op coverage, self-contained**: `0ae0e1976`, `36e36e63a`, `50569eb87`, `8a199ad31`,
   `b13a93a3d`, `7071ba378`, `20a51f24a`, `bee23322f`, `dca216ad6`
3. **Quant kernels the 27B actually uses**: `06cad0b9e` (Q6_K), `e937b17a2` (q2_K)
4. **SSM path** (the 27B is qwen35): `ded522f83`, `522021325`, `669204ee2`, `5e03bdd87`
5. **FA improvements**: `31288c9c6`, `78a8ae14b`, `fe8841a78`, `ceefd16da`
6. **Perf, non-structural**: `da6c30ade`, `8bb3cf0d2`, `c479922ac`, `4413a1924`, `51fe42c5d`,
   `ca89968a2`, `2e0d86beb`, `27c7a769d`, `6b9c0cfcb`
7. **Env/toolchain**: `76a67730f`, `7aca0d2ea`, `ec0bfac17`, `846639535`, `92d6c8a5c`
8. **Memory**: `9daa0aa22`
9. **Foundation changes** (separate branch, full regression): `b797b9ca8`, `643b0cdac`,
   `e9a2a44fe`, `f0fb5984b`

Steps 1-3 are the ones with the best value/risk ratio and should be the first Phase 62 target.

## 4. What is NOT decided here

- Whether to take upstream's IQ2_XXS (#31954) and drop the fork's. That is a strategy call:
  the fork's kernel is tuned for this device, upstream's is general. **Not decided.**
- Whether the foundation changes (repack rework, op fusion) are worth the regression risk at
  all. They unlock most of the perf commits, so the answer is probably yes, but it is a
  multi-phase job.
- The regression hash will change once any numerical upstream change lands. Phase 62 must
  re-baseline `e9da92f8` / `44cd99ce` and move the quality gate to NLL/argmax.
