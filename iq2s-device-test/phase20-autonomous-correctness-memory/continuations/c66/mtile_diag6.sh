#!/bin/bash
# Phase 67 diagnostic 5: tiled prefill ladder, 128 tokens (4 ubatches of 32).
set -u
export MSYS_NO_PATHCONV=1
A=adb
B=/data/local/tmp/c61-inkling2
echo "== mem"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
$A shell "LIBDIR=$B/variant-diag PT=128 NOLOGITS=1 NB=32 NUB=32 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-down-24.txt sh $B/p62_probe.sh c67h_128 1 48 HTP0:0" >/dev/null 2>&1
echo "rc=$?"
$A shell "grep -a -E 'TOKEN step 0|PREFILL' $B/runs/c67h_128/stdout.txt; grep -a 'model buffer size' $B/runs/c67h_128/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c67h_128/manifest.txt; pidof decode-probe"
echo "== done"