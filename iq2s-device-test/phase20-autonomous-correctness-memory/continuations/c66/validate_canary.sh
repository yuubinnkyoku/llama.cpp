#!/bin/bash
# Phase 69: canary validation, regression, and prefill 512.
set -u
export MSYS_NO_PATHCONV=1
A=adb
B=/data/local/tmp/c61-inkling2
D=D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/mtile-diag
$A push "$D/libggml-htp-v81-canary5.so" $B/diag-htp-canary5.so >/dev/null
$A shell "rm -rf $B/variant-diag $B/variant-safehost; cp -a $B/libs $B/variant-diag; cp -a $B/libs $B/variant-safehost; cp $B/diag-htp-canary5.so $B/variant-diag/libggml-htp-v81.so; cp $B/diag-htp-canary5.so $B/variant-safehost/libggml-htp-v81.so; cp $B/diag-hex.so $B/variant-diag/libggml-hexagon.so; sha256sum $B/variant-diag/libggml-htp-v81.so $B/variant-safehost/libggml-htp-v81.so $B/variant-safehost/libggml-hexagon.so"
echo "== mem"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
echo "== replay hash with diagnostic skel and safe host"
$A shell "LIBDIR=$B/variant-safehost REPLAY=$B/replay4.txt NB=32 NUB=8 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-inkling.txt sh $B/p62_probe.sh c69hash 4 48 HTP0:0" >/dev/null 2>&1
$A shell "sha256sum $B/runs/c69hash/logits.f32; grep -a 'model buffer size' $B/runs/c69hash/stderr.txt | grep -a HTP; grep -a 'TOKEN step 0' $B/runs/c69hash/stdout.txt"
run() { # tag pt mchunk
  echo "== run $1 PT=$2 m_chunk=$3"
  $A shell "LIBDIR=$B/variant-diag GGML_HEXAGON_ID_MCHUNK=$3 PT=$2 NB=32 NUB=32 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-down-24.txt sh $B/p62_probe.sh c69v_$1 1 48 HTP0:0" >/dev/null 2>&1
  echo "rc=$?"
  $A shell "grep -a -E 'TOKEN step 0' $B/runs/c69v_$1/stdout.txt; grep -a swap_total $B/runs/c69v_$1/manifest.txt; sha256sum $B/runs/c69v_$1/logits.f32 2>/dev/null; grep -a -c 'dsp-error' $B/runs/c69v_$1/stderr.txt"
}
run a 5 0
run b 5 32
run c 17 0
run d 17 128
run e 32 128
echo "== prefill 512 (warmup was run e)"
$A shell "LIBDIR=$B/variant-diag PT=512 TWO_PHASE=1 NOLOGITS=1 NB=32 NUB=32 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-down-24.txt sh $B/p62_probe.sh c69p512 1 48 HTP0:0" >/dev/null 2>&1
echo "rc=$?"
$A shell "grep -a -E 'PREFILL|TOKEN step 0|RESULT' $B/runs/c69p512/stdout.txt; grep -a 'model buffer size' $B/runs/c69p512/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c69p512/manifest.txt; grep -a -c 'dsp-error' $B/runs/c69p512/stderr.txt; grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
echo "== done"