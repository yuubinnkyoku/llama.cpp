# Hexagon upstream fixes — Phase 51 pre-work (2026-11-14)

Status: **CPU/host audit + device smoke, NOT PUSHED**. Branch
`codex/c51-upstream-fixes`, based on `codex/c50-perf` (`8e5b850a3`).
This work does not merge upstream master/v1.0.0 and does not change the
training path.

## 0. Scope and result

| Task | Verdict |
| --- | --- |
| 1. FastRPC handle truncation (upstream #30952) | **PASS** |
| 2. `set_tensor_n` / repacking / tiling audit (upstream #31744) | **PASS** (2 defects fixed, 1 item 該当なし, hostbuf default not ported) |

The 34-tensor asynchronous abort is **not** claimed fixed. The two fixes remove
candidate causes; the minimal reproduction remains Phase 51 work.

## 1. Baseline

| Item | Value |
| --- | --- |
| Worktree | `D:/ghq/github.com/yuubinnkyoku/llama.cpp-c51-upstream-fixes` |
| Branch | `codex/c51-upstream-fixes` |
| Base | `8e5b850a324cc68a30b796e31e385bc565dc06cd` (`codex/c50-perf`) |
| Commits | `7df29a557`, `e30c6b7cd`, `8d96593b6` |
| Push | none |
| Upstream refs read | `pull/30952/head` (`cfc200d1a`), `pull/31744/head` (`71a821b6d`) |

The main worktree (`codex/phase20-htp-correctness-memory`) has nine
pre-existing uncommitted files. It was not touched.

## 2. Task 1 — FastRPC handle truncation

### 2.1 Change

```diff
-typedef int (*remote_handle64_close_pfn_t)(remote_handle h);
+typedef int (*remote_handle64_close_pfn_t)(remote_handle64 h);
```

`remote_handle` is `uint32_t`, `remote_handle64` is `uint64_t` in
`remote.h`. The dynamically resolved close function is loaded with
`remote_handle64_close_pfn_t` and called with a 64-bit handle.

### 2.2 Is the dynamic path used?

`htpdrv_init()` resolves `remote_handle64_close` with `dlsym(..., false)`, so a
missing symbol fails driver init. The exported wrapper exists at
`htp-drv.cpp:195`. However, **no in-tree caller of `remote_handle64_close`
exists** (`git grep` finds only the typedef, pointer, wrapper and dlsym). The
fork's sessions use `dspqueue_create/close`, not a 64-bit remote handle.
Therefore the 0x2c leak path was latent, not on the session teardown path.
The fix is still applied because the declaration must match the callee and the
API is exported.

### 2.3 Build

| Build | Result |
| --- | --- |
| Host (`build-host-c51`, MSVC/Ninja) | `llama-cli` exit 0; `version 0.6.0-dev (build 11555, commit 8e5b850a3)` |
| Android/Hexagon cross (`build-android-c51`, NDK 29, SDK 6.6.0.0, toolv19) | `ggml-hexagon` exit 0; `libggml-hexagon.so` 7,793,504 B |

The cross build prints five `-Wmissing-field-initializers` warnings from
`fence_extra` at lines 9334 and 10031 (pre-existing, not introduced here).

### 2.4 Device A/B

Test: `dlopen("libggml-hexagon.so")`, `htpdrv_init`, `remote_handle64_open`
with the HTP skel URI, then `remote_handle64_close`; timeout 30 s.

| Library | `open` | `close` | Exit |
| --- | --- | ---: | ---: |
| Fixed build (`-O2`/release, 7,793,504 B) | 0, `0xb4000078cae10d80` | **0 (0x0)** | 0 |
| Pre-fix, rebuilt `-O0` (81,792 B) | 0, `0xb4000079eb60b560` | **44 (0x2c)** | 6 |
| Pre-fix, rebuilt `-O2` (56,656 B) | 0, `0xb400007d3c808600` | 0 (0x0) | 0 |

The 0x2c was reproduced with the old declaration at `-O0`. At `-O2` the compiler
left the full 64-bit value in `x0` because the callee parameter is 32-bit, so
the ABI mismatch happened to work. The defect is therefore real but
codegen-dependent; the fix removes the mismatch rather than relying on ABI
slack.

Device: `324753221196` (USB). No reboot. SwapFree unchanged.

## 3. Task 2 — #31744 audit

### 3.1 Upstream commits read

`680792e` (move tiling to graph-optimize), `b7409c6` (simplify tiling rules),
`21362a7` (tensor hash), `32d1757` (hash avalanche), `315c1a3` (empty ops),
`da1697d` (hostbuf default + max_vmem), `14a1748` (error/readme),
`71a821b` (sync set_tensor). PR body confirms the three regressions.

### 3.2 Finding table

| Upstream item | Fork state | Verdict | Evidence / action |
| --- | --- | --- | --- |
| Descriptor de-dup misses shape change | `t_map` pointer lookup returned a cached descriptor without checking shape/data; data+shape lookup was first, but a pointer whose data also moved fell through to the stale pointer | **該当あり → fixed** | `is_cached_tensor()` requires `same_shape` and `h->data == current offset`; `t_map[t] = ti` updates a changed entry; `fit_op` uses the same predicate |
| Sync `set_tensor` misses opbatch flush | No `flush_sync()` in `set_tensor_impl`, `set_tensor_2d`, host `set_tensor` | **該当あり → fixed** | flush before the write in all three paths; `set_tensor_n` does not exist in this fork, so that path is N/A |
| Over-tiling non-`MUL_MAT` weights | For non-IQ4_XS types, `needs_repack` is inserted only by `supported_mul_mat` / `supported_mul_mat_id`; IQ4_XS is always REPACK by type and its layout supports `ne2/ne3` slices | **該当なし（条件付き）** | No arbitrary 2D/3D weight is tiled. IQ4_XS is the fork's chosen representation and GET_ROWS has a tiled kernel. Memory pressure remains a Phase 51 variable; the upstream graph-optimize restructure is not ported (structural difference, would need device validation) |
| Host buffers default ON | `opt_hostbuf = false`, `GGML_HEXAGON_HOSTBUF` only when set | **not ported (as instructed)** | Avoids NPU virtual-address pressure |
| `max_vmem` rejection | Already present in `ggml_backend_hexagon_buffer_type_alloc_buffer` | **already present** | `size > sess->max_vmem` logs and returns null |
| Empty (zero-element) ops | `op_is_compute` excludes empty tensors; `supports_op` accepts empty | **該当なし（difference）** | Upstream accepts empty ops to avoid splits; fork drops them. Not one of the three regressions and not touched |
| `set_tensor_n` API | Not present in the fork's buffer interface | **N/A** | No such path to sync |

### 3.3 Changes

1. `e30c6b7cd` — descriptor validation (+24/−5).
2. `8d96593b6` — `flush_sync()` in three sync set_tensor paths (+13/−1).

No change to tiling, hostbuf default, HMX kernels, split registry or training.

## 4. Verification

### 4.1 BF16 exact-pattern device test

Command (new library in a temp dir, original skel):
`p49_bf16_test` with `LD_LIBRARY_PATH`/`ADSP_LIBRARY_PATH` set to the temp libs.

| Metric | Value |
| --- | --- |
| Exit | 0 |
| Wall | 23 s |
| `exact patterns` | `65536 checked 65536 bad 0` |
| Dot checks | M=1/2/3, K=5120/5152/96, nonfinite 0, all `ok` |
| SwapFree before → after | 9,639,568 → 9,640,336 kB (+768 kB; no swap-out) |

### 4.2 IQ2/IQ3 4-step replay

Model: `Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf` (9,259,510,912 B).
`decode-probe-wait MODEL override-l57.txt 4 out mmap HTP0:0 NGL`.

| Run | Exit | Wall | SHA-256 of logits | SwapFree before→after |
| --- | ---: | ---: | --- | --- |
| NGL=9, original lib | 0 | 20 s | `5585d4ed…8bd480` | 9,626,464 → 9,017,268 |
| NGL=9, fixed lib | 0 | 10 s | `5585d4ed…8bd480` | 9,105,876 → 9,116,208 |
| NGL=16, fixed lib | 0 | 10 s | `f3a0d235…0adca7` | 9,138,384 → 9,116,308 |
| NGL=24, fixed lib | 0 | 13 s | `22bce4c9…3c1da8` | 9,117,076 → 8,740,616 |

The two NGL=9 runs are byte-identical. NGL=16/24 match the Phase 50 baselines
`f3a0d235…` and `22bce4c9…`. No non-finite logits, no DSP error, no swap-out
above the 1024 MiB guard. Each run is under the 180 s limit.

### 4.3 Build regression

Host and cross builds pass (section 2.3). The Android device lib was rebuilt
from `8e5b850a3` + the three commits and used for all device runs.

## 5. What is not fixed / not claimed

1. **34th tensor abort is not resolved.** No minimal reproduction was run in
   this task. The descriptor and sync fixes are candidate-cause removals, not
   proof.
2. **Descriptor shape-change fix has no failing unit reproduction.** It is
   code-audited and exercised by model load/device runs, but the 34-tensor
   scenario is needed to show the stale descriptor was the cause.
3. **Tiling structure unchanged.** IQ4_XS remains type-driven; the upstream
   graph-optimize move is not ported. Measure NPU virtual-address pressure in
   Phase 51.
4. **Device tests used a temp lib directory**, not an installed APK. The app
   path was not modified.
5. **No push, no merge, no PR.**

## 6. Phase 51 test design implication

The minimal reproduction should vary:
- tensor count around the 33/34 boundary;
- the shape of a reused weight tensor (same pointer, changed `ne`/`nb`/`data`);
- split on/off and NGL 9/16/24;
- with `GGML_HEXAGON_VERBOSE=1`, log `add-tensor #N ... data ... size ... flags`
  and assert that a changed shape produces a new descriptor index and a new
  data offset, not the cached one.

Only after that reproduction should the 34th-tensor cause be called resolved.

## 7. Reproduction commands (host)

```powershell
cmake -S . -B build-host-c51 -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DGGML_HEXAGON=OFF -DGGML_OPENCL=OFF -DGGML_OPENMP=OFF `
  -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF
cmake --build build-host-c51 --target llama-cli -j 8

cmake -S . -B build-android-c51 -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE=<NDK>/build/cmake/android.toolchain.cmake `
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 `
  -DGGML_HEXAGON=ON -DGGML_OPENCL=OFF -DHEXAGON_SDK_ROOT=<SDK> `
  -DPREBUILT_LIB_DIR=android_aarch64
cmake --build build-android-c51 --target ggml-hexagon -j 8
```
