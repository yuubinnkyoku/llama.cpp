#!/bin/bash
# Phase 69 regression: canary4, tiled vs non-tiled, same prompt, bit compare.
set -u
export MSYS_NO_PATHCONV=1
A=adb
B=/data/local/tmp/c61-inkling2
echo "== mem"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
run() { # tag pt mchunk
  echo "== run $1 PT=$2 m_chunk=$3"
  $A shell "LIBDIR=$B/variant-diag GGML_HEXAGON_ID_MCHUNK=$3 PT=$2 NB=32 NUB=32 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-down-24.txt sh $B/p62_probe.sh c69r_$1 1 48 HTP0:0" >/dev/null 2>&1
  echo "rc=$?"
  $A shell "grep -a -E 'PROMPT_IDS|TOKEN step 0' $B/runs/c69r_$1/stdout.txt; grep -a 'model buffer size' $B/runs/c69r_$1/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c69r_$1/manifest.txt; sha256sum $B/runs/c69r_$1/logits.f32 2>/dev/null; grep -a -c 'dsp-error' $B/runs/c69r_$1/stderr.txt"
}
run a 5 0
run b 5 32
run c 17 0
run d 17 128
echo "== done"