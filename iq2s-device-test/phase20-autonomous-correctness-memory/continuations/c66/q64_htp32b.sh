#!/bin/bash
# Phase 72b: HTP 64-step at NUB=32, no cleanup (the gate check supplies the window).
set -u
export MSYS_NO_PATHCONV=1
if [ -z "${ANDROID_SERIAL:-}" ]; then
  export ANDROID_SERIAL=$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')
fi
A=adb
B=/data/local/tmp/c61-inkling2
echo "== mem start"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
echo "== warmup htp 8"
$A shell "NB=32 NUB=32 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-inkling.txt sh $B/p62_probe.sh c72s_w 8 48 HTP0:0" >/dev/null 2>&1
$A shell "grep -a -E 'RESULT forwards' $B/runs/c72s_w/stdout.txt; grep -a swap_total $B/runs/c72s_w/manifest.txt"
echo "== htp 64"
$A shell "NB=32 NUB=32 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-inkling.txt sh $B/p62_probe.sh c72s_h 64 48 HTP0:0" >/dev/null 2>&1
echo "rc=$?"
$A shell "grep -a -E 'RESULT forwards|TOKEN step 0' $B/runs/c72s_h/stdout.txt; grep -a 'model buffer size' $B/runs/c72s_h/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c72s_h/manifest.txt"
D=D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/q64n32
$A pull $B/runs/c72s_h/logits.f32 "$D/htp64c.f32" >/dev/null 2>&1
python D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/gate.py "$D/cpu64.f32" "$D/htp64c.f32" 2>&1 | head -8
echo "== done"