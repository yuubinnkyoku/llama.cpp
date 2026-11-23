#!/bin/bash
set -u
export MSYS_NO_PATHCONV=1
A=adb
B=/data/local/tmp/c61-inkling2
D=D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/mtile-diag
$A push "$D/libggml-htp-v81-canary2.so" $B/diag-htp-canary2.so >/dev/null
$A shell "rm -rf $B/variant-diag; cp -a $B/libs $B/variant-diag; cp $B/diag-htp-canary2.so $B/variant-diag/libggml-htp-v81.so; cp $B/diag-hex.so $B/variant-diag/libggml-hexagon.so; sha256sum $B/variant-diag/libggml-htp-v81.so"
echo "== mem"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
$A shell "LIBDIR=$B/variant-diag GGML_HEXAGON_ID_MCHUNK=32 PT=5 NB=32 NUB=32 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-down-24.txt sh $B/p62_probe.sh c68d_5 1 48 HTP0:0" >/dev/null 2>&1
echo "rc=$?"
$A shell "grep -a -E 'dsp-error|dsp-rsp' $B/runs/c68d_5/stderr.txt | tail -3; grep -a -E 'TOKEN step 0|PREFILL' $B/runs/c68d_5/stdout.txt; grep -a 'model buffer size' $B/runs/c68d_5/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c68d_5/manifest.txt"
echo "== done"