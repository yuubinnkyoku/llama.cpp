#!/bin/bash
# Phase 72: confirm the frozen hash at the adoption config (NUB=32, auto m_chunk).
set -u
export MSYS_NO_PATHCONV=1
if [ -z "${ANDROID_SERIAL:-}" ]; then
  export ANDROID_SERIAL=$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')
fi
A=adb
B=/data/local/tmp/c61-inkling2
$A shell "sha256sum $B/variant-diag/libggml-htp-v81.so $B/variant-diag/libggml-hexagon.so"
$A shell "LIBDIR=$B/variant-diag REPLAY=$B/replay4.txt NB=32 NUB=32 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-inkling.txt sh $B/p62_probe.sh c72hash 4 48 HTP0:0" >/dev/null 2>&1
echo "rc=$?"
$A shell "sha256sum $B/runs/c72hash/logits.f32 2>/dev/null; grep -a 'model buffer size' $B/runs/c72hash/stderr.txt | grep -a HTP; grep -a 'TOKEN step 0' $B/runs/c72hash/stdout.txt; grep -a swap_total $B/runs/c72hash/manifest.txt"
echo "== done"