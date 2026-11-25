#!/bin/bash
# Phase 70: adoption A/B, prefill 512 and decode 32, NUB=8 non-tiled vs NUB=32 tiled.
set -u
export MSYS_NO_PATHCONV=1
if [ -z "${ANDROID_SERIAL:-}" ]; then
  export ANDROID_SERIAL=$(adb devices | awk 'NR>1 && $2=="device" && $1 ~ /_adb-tls-connect/ {print $1; exit}')
  [ -z "${ANDROID_SERIAL:-}" ] && export ANDROID_SERIAL=$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')
fi
A=adb
B=/data/local/tmp/c61-inkling2
echo "== mem start"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
prefill() { # tag nub mchunk
  echo "== prefill run $1 NUB=$2 m_chunk=$3"
  $A shell "LIBDIR=$B/variant-diag GGML_HEXAGON_ID_MCHUNK=$3 PT=512 TWO_PHASE=1 NOLOGITS=1 NB=32 NUB=$2 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-inkling.txt sh $B/p62_probe.sh c70_$1 1 48 HTP0:0" >/dev/null 2>&1
  echo "rc=$?"
  $A shell "grep -a -E 'PREFILL|RESULT forwards' $B/runs/c70_$1/stdout.txt; grep -a 'model buffer size' $B/runs/c70_$1/stderr.txt | grep -a HTP; grep -a 'splits =' $B/runs/c70_$1/stderr.txt | head -1; grep -a swap_total $B/runs/c70_$1/manifest.txt"
}
decode() { # tag nub
  echo "== decode run $1 NUB=$2"
  $A shell "LIBDIR=$B/variant-diag NB=32 NUB=$2 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-inkling.txt sh $B/p62_probe.sh c70_$1 32 48 HTP0:0" >/dev/null 2>&1
  echo "rc=$?"
  $A shell "grep -a -E 'PREFILL|RESULT forwards' $B/runs/c70_$1/stdout.txt; grep -a 'model buffer size' $B/runs/c70_$1/stderr.txt | grep -a HTP; grep -a 'splits =' $B/runs/c70_$1/stderr.txt | head -1; grep -a swap_total $B/runs/c70_$1/manifest.txt"
}
prefill p8 8 0
prefill p32 32 128
echo "== warmup decode"
$A shell "LIBDIR=$B/variant-diag PT=5 NB=32 NUB=8 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-inkling.txt sh $B/p62_probe.sh c70_w 8 48 HTP0:0" >/dev/null 2>&1
decode d8 8
decode d32 32
echo "== done"