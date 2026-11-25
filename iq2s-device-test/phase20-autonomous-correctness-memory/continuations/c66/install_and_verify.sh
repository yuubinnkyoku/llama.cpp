#!/bin/bash
# Phase 72: install the adopted NUB=32 / M tiling libs as the standard and verify.
set -u
export MSYS_NO_PATHCONV=1
if [ -z "${ANDROID_SERIAL:-}" ]; then
  export ANDROID_SERIAL=$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')
fi
A=adb
B=/data/local/tmp/c61-inkling2
D=D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66
$A push "$D/c72-libs/libggml-hexagon.so" $B/new-hex.so >/dev/null
$A push "$D/c72-libs/libggml-htp-v81.so" $B/new-htp.so >/dev/null
$A shell "rm -rf $B/libs-prev; cp -a $B/libs $B/libs-prev; cp $B/new-hex.so $B/libs/libggml-hexagon.so; cp $B/new-htp.so $B/libs/libggml-htp-v81.so; sha256sum $B/libs/libggml-hexagon.so $B/libs/libggml-htp-v81.so $B/libs/libllama.so $B/libs/decode-probe"
echo "== replay NUB=32 with the new standard"
$A shell "REPLAY=$B/replay4.txt NB=32 NUB=32 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-inkling.txt sh $B/p62_probe.sh c72v_hash 4 48 HTP0:0" >/dev/null 2>&1
$A shell "sha256sum $B/runs/c72v_hash/logits.f32; grep -a 'model buffer size' $B/runs/c72v_hash/stderr.txt | grep -a HTP; grep -a 'TOKEN step 0' $B/runs/c72v_hash/stdout.txt; grep -a swap_total $B/runs/c72v_hash/manifest.txt"
echo "== decode 32 NUB=32 with the new standard"
$A shell "PT=5 NB=32 NUB=32 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-inkling.txt sh $B/p62_probe.sh c72v_dec 32 48 HTP0:0" >/dev/null 2>&1
$A shell "grep -a -E 'PREFILL|RESULT forwards' $B/runs/c72v_dec/stdout.txt; grep -a 'model buffer size' $B/runs/c72v_dec/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c72v_dec/manifest.txt"
echo "== done"