#!/bin/bash
# Phase 67 diagnostic 3: 17-token prompt (136 rows), m_chunk=128 vs non-tiled.
# Exercises the 128-row chunk path that crashed prefill 512, at 1/15 the scale.
set -u
export MSYS_NO_PATHCONV=1
A=adb
B=/data/local/tmp/c61-inkling2
echo "== mem"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
run() { # tag mchunk
  echo "== run $1 m_chunk=${2:-0}"
  if [ -n "${2:-}" ]; then MC="GGML_HEXAGON_ID_MCHUNK=$2"; else MC=""; fi
  $A shell "LIBDIR=$B/variant-diag $MC PT=17 NB=32 NUB=32 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-down-24.txt sh $B/p62_probe.sh c67f_$1 1 48 HTP0:0" >/dev/null 2>&1
  echo "rc=$?"
  $A shell "grep -a -E 'TOKEN step 0|PREFILL' $B/runs/c67f_$1/stdout.txt; grep -a 'model buffer size' $B/runs/c67f_$1/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c67f_$1/manifest.txt; ls -l $B/runs/c67f_$1/logits.f32 2>/dev/null | awk '{print \$5}'"
}
run a
run b 128
echo "== hashes"
$A shell "sha256sum $B/runs/c67f_a/logits.f32 $B/runs/c67f_b/logits.f32 2>/dev/null"
echo "== done"