#!/bin/bash
# Phase 69 task 4: prefill 512 and decode A/B, NUB=8 non-tiled vs NUB=32 tiled.
set -u
export MSYS_NO_PATHCONV=1
A=adb
B=/data/local/tmp/c61-inkling2
echo "== mem"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
run() { # tag nub pt steps mchunk
  echo "== run $1 NUB=$2 PT=$3 steps=$4 m_chunk=$5"
  $A shell "LIBDIR=$B/variant-diag GGML_HEXAGON_ID_MCHUNK=$5 PT=$3 TWO_PHASE=1 NOLOGITS=1 NB=32 NUB=$2 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-inkling.txt sh $B/p62_probe.sh c69ab_$1 $4 48 HTP0:0" >/dev/null 2>&1 || true
  $A shell "grep -a -E 'PREFILL|RESULT forwards' $B/runs/c69ab_$1/stdout.txt; grep -a 'model buffer size' $B/runs/c69ab_$1/stderr.txt | grep -a HTP; grep -a 'splits =' $B/runs/c69ab_$1/stderr.txt | head -1; grep -a swap_total $B/runs/c69ab_$1/manifest.txt"
}
run p8 8 512 1 0
run p32 32 512 1 128
echo "== warmup decode"
$A shell "LIBDIR=$B/variant-diag PT=5 NB=32 NUB=8 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-inkling.txt sh $B/p62_probe.sh c69ab_w 8 48 HTP0:0" >/dev/null 2>&1
run d8 8 5 32 0
run d32 32 5 32 128
echo "== done"