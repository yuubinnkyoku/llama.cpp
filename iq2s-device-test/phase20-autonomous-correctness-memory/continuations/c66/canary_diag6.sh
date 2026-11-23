#!/bin/bash
# Phase 68 canary diagnostic at a 6-layer down placement (low memory).
set -u
export MSYS_NO_PATHCONV=1
A=adb
B=/data/local/tmp/c61-inkling2
D=D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/mtile-diag
$A push "$D/libggml-htp-v81-canary.so" $B/diag-htp-canary.so >/dev/null
$A push "$D/libggml-hexagon-probecap.so" $B/diag-hex-probecap.so >/dev/null
$A shell "rm -rf $B/variant-diag; cp -a $B/libs $B/variant-diag; cp $B/diag-htp-canary.so $B/variant-diag/libggml-htp-v81.so; cp $B/diag-hex-probecap.so $B/variant-diag/libggml-hexagon.so; sha256sum $B/variant-diag/libggml-htp-v81.so $B/variant-diag/libggml-hexagon.so"
echo "== mem"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
run() { # tag pt mchunk
  echo "== run $1 PT=$2 m_chunk=$3"
  $A shell "LIBDIR=$B/variant-diag GGML_HEXAGON_ID_PROBE_TOKENS=32 GGML_HEXAGON_ID_MCHUNK=$3 PT=$2 NB=32 NUB=32 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-down-6.txt sh $B/p62_probe.sh c68c_$1 1 48 HTP0:0" >/dev/null 2>&1
  echo "rc=$?"
  $A shell "grep -a -E 'TOKEN step 0|PREFILL' $B/runs/c68c_$1/stdout.txt; grep -a 'model buffer size' $B/runs/c68c_$1/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c68c_$1/manifest.txt; sha256sum $B/runs/c68c_$1/logits.f32 2>/dev/null; tail -3 $B/runs/c68c_$1/stderr.txt"
}
run a 5 32
run b 17 128
echo "== done"