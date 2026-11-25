#!/bin/bash
# Phase 72b: 64-step quality gate under the adopted NUB=32 configuration.
set -u
export MSYS_NO_PATHCONV=1
if [ -z "${ANDROID_SERIAL:-}" ]; then
  export ANDROID_SERIAL=$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')
fi
A=adb
B=/data/local/tmp/c61-inkling2
echo "== mem start"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
run() { # tag override steps
  echo "== run $1 $2 $3 steps"
  $A shell "NB=32 NUB=32 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=$2 sh $B/p62_probe.sh c72q_$1 $3 48 HTP0:0" >/dev/null 2>&1
  echo "rc=$?"
  $A shell "grep -a -E 'RESULT forwards|TOKEN step 0' $B/runs/c72q_$1/stdout.txt; grep -a 'model buffer size' $B/runs/c72q_$1/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c72q_$1/manifest.txt"
}
run wcpu override-down-cpu.txt 8
run cpu override-down-cpu.txt 64
run whtp override-inkling.txt 8
run htp override-inkling.txt 64
D=D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/q64n32
mkdir -p "$D"
$A pull $B/runs/c72q_cpu/logits.f32 "$D/cpu64.f32" >/dev/null
$A pull $B/runs/c72q_htp/logits.f32 "$D/htp64.f32" >/dev/null
python D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/gate.py "$D/cpu64.f32" "$D/htp64.f32"
echo "== done"