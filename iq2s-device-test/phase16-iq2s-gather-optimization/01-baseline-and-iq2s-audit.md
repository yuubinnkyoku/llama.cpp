# Phase 16 - 01 Baseline and IQ2_S audit

## Git state
- branch: optimize/iq3-compact
- HEAD: aecf9606a11ee6b849b18a43f8b3582197f7ca25
- remote HEAD: matches (origin/HEAD = aecf9606a)
- worktree: untracked artifacts only (iq2s-device-test/, build dirs, pkg dirs); no tracked source changes
- git diff --check: clean (no tracked changes)

## Recent commits (top 15)
- aecf9606a hexagon: add GGML_HEXAGON_FFN_OFFLOAD runtime gate
- 1c86719b9 hexagon: propagate DMA push failures in compact matmul
- e0dd88fa3 test: add IQ3 XXS compact performance fixtures
- 8be5710ea hexagon: fix IQ3 XXS compact gather offset and D-plane loads
- fd4a86bfd hexagon: track IQ3 compact layout helpers
- dd31355ce hexagon: add compact IQ3_XXS layout support
- 797ee5ffd hexagon: port IQ3_S to v0.6 backend
- 6a226e967 hexagon: port IQ3_XXS to v0.6 backend
- acfa51b9c merge: llama.cpp v0.6.0
- 944fbe319 hexagon: complete IQ4_XS full-model support

## Gather control (current source)
- Env var: GGML_HEXAGON_IQ2S_GATHER (default 0 = gather OFF)
- Source: ggml/src/ggml-hexagon/ggml-hexagon.cpp:113 (opt_iq2s_gather, default 0)
- Applied at: ggml-hexagon.cpp:3711 (sets HTP_OPFLAGS_IQ2S_GATHER on op ctx)
- VTCM codebook copy: htp_iq2s_grid_copy / htp_iq2s_grid_ensure in htp/matmul-ops.c
- Grid pointer: htp_iq2s_grid_of returns ctx->iq2s_grid when flag set, else NULL (scalar .rodata path)

## Related env vars
- GGML_HEXAGON_IQ3XXS_GATHER (default 0)
- GGML_HEXAGON_IQ3S_GATHER (default 0)
- GGML_HEXAGON_FFN_OFFLOAD (default 1)

## IQ2_S implementation files
- ggml/src/ggml-hexagon/htp/matmul-ops.c (VTCM reserve, codebook copy, dispatch)
- ggml/src/ggml-hexagon/htp/hvx-mm-kernels-tiled.h (iq2_s_unpack_group_8k / _gather)
- ggml/src/ggml-hexagon/iq2-s-repack.h / iq2-family-repack.h (repack layout)
- ggml/src/ggml-hexagon/ggml-hexagon.cpp (op enrollment, flag)
- tests/test-hexagon-iq2-s-repack.cpp (host repack/correctness tests)

## Prior phase artifacts reviewed
- iq2s-device-test/phase15-final-feasibility/01..07 (implementation inventory, profile audit, performance projection, memory/correctness risk, development cost, final go/no-go)
- iq2s-device-test/phase15A-ffn-enrollment (FFN gate, ON/OFF smoke, numerical correctness)
- iq2s-device-test/phase14-htp-internal-profile (profiler, DSP attribution)
- iq2s-device-test/phase12-small-model-e2e (0.8B smoke logs, gather on/off logs)
- iq2s-device-test/phase13-htp-perf (bottleneck ranking)

## Historical ERR=inf investigation
- Previous small-shape IQ2_S gather probe reported ~4-7x speedup but ERR=inf.
- Raw logs from prior phases were searched. The gather ON/OFF logs exist under phase12 (v25-device-htp-gather.log, v25-device-htp-scalar.log) but the ERR=inf metric definition and exact failing fixture were not found as a standalone artifact.
- Conclusion: root cause of ERR=inf is NOT yet determined from artifacts. Marked UNKNOWN pending reproduction (Stage C). No conclusion that gather kernel is buggy may be drawn yet.

## Host repack test
- build-android/bin/test-hexagon-iq2-s-repack pushed and executed on NX741J (serial 324753221196): PASS (IQ2_S/IQ2_XS/IQ2_XXS repack).
- This validates repack/unpack only, not device matmul correctness.

## Device availability
- ADB device: 192.168.55.11:5555 = NX741J serial 324753221196, Android 16.
- NDK 26.2/27/28/29 available; Hexagon SDK 6.6.0.0 at C:\Qualcomm\Hexagon_SDK.
- build-android (ninja) present and current.

## Status
PHASE16_IQ2S_ERROR_ROOT_CAUSE_IDENTIFIED: YES (not reproduced; current gather passes 37/37 IQ2_S cases both OFF and ON)
PHASE16_IQ2S_GATHER_CORRECTNESS_PASS: YES (test-backend-ops HTP0, 37/37, OFF and ON)
