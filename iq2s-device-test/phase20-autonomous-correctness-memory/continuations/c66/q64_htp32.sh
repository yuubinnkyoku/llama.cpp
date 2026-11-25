#!/bin/bash
# Phase 72b retry: cleanup then HTP 64-step at NUB=32 in one gate.
set -u
export MSYS_NO_PATHCONV=1
if [ -z "${ANDROID_SERIAL:-}" ]; then
  export ANDROID_SERIAL=$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')
fi
A=adb
B=/data/local/tmp/c61-inkling2
FG=$($A shell "dumpsys activity activities 2>/dev/null | sed -n 's/.*topResumedActivity=ActivityRecord{[^ ]* [^ ]* \([^\/]*\)\/.*/\1/p' | head -1" | tr -d '\r')
echo "FG=$FG"
for p in net.waterfox.android.release dev.shiggy.cord com.google.android.googlequicksearchbox com.kazumaproject.markdownhelperkeyboard.npu com.kazumaproject.markdownhelperkeyboard com.android.vending jp.nicovideo.android com.facebook.aura com.android.chrome app.morphe.android.youtube com.google.android.apps.photos org.mozilla.fenix com.google.android.apps.messaging jp.naver.line.android org.torproject.vpn com.twitter.android.piko; do
  [ -n "$FG" ] && [ "$p" = "$FG" ] && continue
  $A shell "am force-stop $p" >/dev/null 2>&1
done
$A shell "am kill-all >/dev/null 2>&1; sleep 2; am kill-all >/dev/null 2>&1; sync; sleep 5"
MEM=$($A shell "awk '/MemAvailable/{print \$2}' /proc/meminfo" | tr -d '\r')
echo "MEM_AVAILABLE=$MEM"
if [ -z "$MEM" ] || [ "$MEM" -lt 11534336 ]; then echo LOW; exit 76; fi
echo "== warmup htp 8"
$A shell "NB=32 NUB=32 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-inkling.txt sh $B/p62_probe.sh c72r_w 8 48 HTP0:0" >/dev/null 2>&1
$A shell "grep -a -E 'RESULT forwards' $B/runs/c72r_w/stdout.txt; grep -a swap_total $B/runs/c72r_w/manifest.txt"
echo "== htp 64"
$A shell "NB=32 NUB=32 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-inkling.txt sh $B/p62_probe.sh c72r_h 64 48 HTP0:0" >/dev/null 2>&1
echo "rc=$?"
$A shell "grep -a -E 'RESULT forwards|TOKEN step 0' $B/runs/c72r_h/stdout.txt; grep -a 'model buffer size' $B/runs/c72r_h/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c72r_h/manifest.txt"
D=D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/q64n32
$A pull $B/runs/c72r_h/logits.f32 "$D/htp64b.f32" >/dev/null 2>&1
python D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/gate.py "$D/cpu64.f32" "$D/htp64b.f32" 2>&1 | head -6
echo "== done"