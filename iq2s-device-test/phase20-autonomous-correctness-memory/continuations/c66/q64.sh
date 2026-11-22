#!/bin/bash
# Phase 66 task 2: 64-step quality gate (CPU reference vs default HTP).
set -u
export MSYS_NO_PATHCONV=1
SER=324753221196
A="adb -s $SER"
B=/data/local/tmp/c61-inkling2
OUTW=D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/q64
mkdir -p "$OUTW"
echo "== mem start"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
$A shell "am kill-all >/dev/null 2>&1; sleep 2; am kill-all >/dev/null 2>&1; sync; sleep 3; grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
echo "== cpu64"
$A shell "NB=32 NUB=8 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-down-cpu.txt sh $B/p62_probe.sh c66q64_cpu 64 48 HTP0:0" >/dev/null 2>&1
$A shell "grep -a -E 'RESULT forwards|TOKEN step 0 ' $B/runs/c66q64_cpu/stdout.txt; grep -a 'model buffer size' $B/runs/c66q64_cpu/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c66q64_cpu/manifest.txt"
echo "== htp64"
$A shell "NB=32 NUB=8 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-inkling.txt sh $B/p62_probe.sh c66q64_htp 64 48 HTP0:0" >/dev/null 2>&1
$A shell "grep -a -E 'RESULT forwards|TOKEN step 0 ' $B/runs/c66q64_htp/stdout.txt; grep -a 'model buffer size' $B/runs/c66q64_htp/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c66q64_htp/manifest.txt"
$A pull $B/runs/c66q64_cpu/logits.f32 "$OUTW/cpu64.f32"
$A pull $B/runs/c66q64_htp/logits.f32 "$OUTW/htp64.f32"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo; grep -a 'model buffer size' $B/runs/c66q64_htp/stderr.txt | grep -a HTP"
echo "== done"
