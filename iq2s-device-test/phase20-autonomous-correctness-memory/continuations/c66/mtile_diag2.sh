#!/bin/bash
# Phase 67 diagnostic: tiled vs non-tiled ID path on a 6-layer down placement.
# 5 token prompt -> 40 rows; forced m_chunk=32 gives 2 chunks (32+8).
set -u
export MSYS_NO_PATHCONV=1
A=adb
B=/data/local/tmp/c61-inkling2
D=D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/mtile-diag
$A push "$D/libggml-htp-v81.so" $B/diag-htp.so >/dev/null
$A push "$D/libggml-hexagon.so" $B/diag-hex.so >/dev/null
$A shell "rm -rf $B/variant-diag; cp -a $B/libs $B/variant-diag; cp $B/diag-htp.so $B/variant-diag/libggml-htp-v81.so; cp $B/diag-hex.so $B/variant-diag/libggml-hexagon.so; sha256sum $B/variant-diag/libggml-htp-v81.so $B/variant-diag/libggml-hexagon.so"
echo "== mem"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
run() { # tag mchunk
  echo "== run $1 m_chunk=${2:-0}"
  if [ -n "${2:-}" ]; then
    MC="GGML_HEXAGON_ID_MCHUNK=$2"
  else
    MC=""
  fi
  $A shell "LIBDIR=$B/variant-diag $MC NB=32 NUB=8 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-down-6.txt sh $B/p62_probe.sh c67d_$1 1 48 HTP0:0" >/dev/null 2>&1
  echo "rc=$?"
  $A shell "grep -a -E 'TOKEN step 0|RESULT|PREFILL' $B/runs/c67d_$1/stdout.txt; grep -a 'model buffer size' $B/runs/c67d_$1/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c67d_$1/manifest.txt; ls -l $B/runs/c67d_$1/logits.f32 2>/dev/null | awk '{print \$5}'"
}
run a
run b 32
$A shell "sha256sum $B/runs/c67d_a/logits.f32 $B/runs/c67d_b/logits.f32 2>/dev/null"
echo "== done"