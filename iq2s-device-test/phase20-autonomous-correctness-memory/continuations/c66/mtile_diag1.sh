#!/bin/bash
# Phase 67 task 2: diagnostic tiled run, forced small m_chunk, 5-token prompt.
set -u
export MSYS_NO_PATHCONV=1
A=adb
B=/data/local/tmp/c61-inkling2
D=D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/mtile-diag
$A push "$D/libggml-htp-v81.so" $B/diag-htp.so
$A push "$D/libggml-hexagon.so" $B/diag-hex.so
$A shell "rm -rf $B/variant-diag; cp -a $B/libs $B/variant-diag; cp $B/diag-htp.so $B/variant-diag/libggml-htp-v81.so; cp $B/diag-hex.so $B/variant-diag/libggml-hexagon.so; sha256sum $B/variant-diag/libggml-htp-v81.so $B/variant-diag/libggml-hexagon.so"
echo "== mem"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
echo "== diag run (m_chunk=32, 5 token prompt)"
$A shell "GGML_HEXAGON_ID_MCHUNK=32 LIBDIR=$B/variant-diag NB=32 NUB=8 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-inkling.txt sh $B/p62_probe.sh c67diag 1 48 HTP0:0" >/dev/null 2>&1
echo "rc=$?"
$A shell "grep -a -E 'TOKEN step 0|RESULT|PREFILL' $B/runs/c67diag/stdout.txt; grep -a 'model buffer size' $B/runs/c67diag/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c67diag/manifest.txt; tail -5 $B/runs/c67diag/stderr.txt"
echo "== done"