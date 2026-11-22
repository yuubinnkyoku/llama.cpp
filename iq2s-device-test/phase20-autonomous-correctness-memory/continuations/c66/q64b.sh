#!/bin/bash
# Phase 66 task 2 v2: 64-step quality gate with per-config warmup.
set -u
export MSYS_NO_PATHCONV=1
A="adb -s 324753221196"
B=/data/local/tmp/c61-inkling2
OUTW=D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/q64
mkdir -p "$OUTW"
echo "== mem start"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"

run() { # $1 tag $2 override $3 steps
  echo "== run $1 ($2, $3 steps)"
  $A shell "NB=32 NUB=8 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=$2 sh $B/p62_probe.sh c66w_$1 $3 48 HTP0:0" >/dev/null 2>&1
  $A shell "grep -a -E 'RESULT forwards|TOKEN step 0 ' $B/runs/c66w_$1/stdout.txt; grep -a 'model buffer size' $B/runs/c66w_$1/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c66w_$1/manifest.txt"
}

run warm_cpu override-down-cpu.txt 8
run cpu64 override-down-cpu.txt 64
run warm_htp override-inkling.txt 8
run htp64 override-inkling.txt 64

$A pull $B/runs/c66w_cpu64/logits.f32 "$OUTW/cpu64.f32"
$A pull $B/runs/c66w_htp64/logits.f32 "$OUTW/htp64.f32"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
echo "== done"