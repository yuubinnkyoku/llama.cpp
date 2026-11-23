#!/bin/bash
# Phase 67 diagnostic 2: non-tiled vs tiled ID path, same 24-layer down config.
# The page cache is warm after diagnostic 1, so both runs should stay in budget.
set -u
export MSYS_NO_PATHCONV=1
A=adb
B=/data/local/tmp/c61-inkling2
echo "== mem"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
run() { # tag mchunk
  echo "== run $1 m_chunk=${2:-0}"
  if [ -n "${2:-}" ]; then MC="GGML_HEXAGON_ID_MCHUNK=$2"; else MC=""; fi
  $A shell "LIBDIR=$B/variant-diag $MC NB=32 NUB=8 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-down-24.txt sh $B/p62_probe.sh c67e_$1 1 48 HTP0:0" >/dev/null 2>&1
  echo "rc=$?"
  $A shell "grep -a -E 'TOKEN step 0' $B/runs/c67e_$1/stdout.txt; grep -a 'model buffer size' $B/runs/c67e_$1/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c67e_$1/manifest.txt; ls -l $B/runs/c67e_$1/logits.f32 2>/dev/null | awk '{print \$5}'"
}
run a
run b 32
echo "== hashes"
$A shell "sha256sum $B/runs/c67e_a/logits.f32 $B/runs/c67e_b/logits.f32 2>/dev/null"
echo "== done"