#!/bin/bash
# Phase 72: cleanup then run the A/B in one gate, no inter-gate memory gap.
set -u
export MSYS_NO_PATHCONV=1
if [ -z "${ANDROID_SERIAL:-}" ]; then
  export ANDROID_SERIAL=$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')
fi
A=adb
FG=$($A shell "dumpsys activity activities 2>/dev/null | sed -n 's/.*topResumedActivity=ActivityRecord{[^ ]* [^ ]* \([^\/]*\)\/.*/\1/p' | head -1" | tr -d '\r')
echo "FG=$FG"
for p in net.waterfox.android.release dev.shiggy.cord com.google.android.googlequicksearchbox com.kazumaproject.markdownhelperkeyboard.npu com.kazumaproject.markdownhelperkeyboard com.android.vending jp.nicovideo.android com.facebook.aura com.android.chrome app.morphe.android.youtube com.google.android.apps.photos org.mozilla.fenix com.google.android.apps.messaging jp.naver.line.android org.torproject.vpn com.twitter.android.piko; do
  [ -n "$FG" ] && [ "$p" = "$FG" ] && continue
  $A shell "am force-stop $p" >/dev/null 2>&1
done
$A shell "am kill-all >/dev/null 2>&1; sleep 2; am kill-all >/dev/null 2>&1; sync; sleep 5"
MEM=$($A shell "awk '/MemAvailable/{print \$2}' /proc/meminfo" | tr -d '\r')
echo "MEM_AVAILABLE=$MEM"
if [ -z "$MEM" ] || [ "$MEM" -lt 11534336 ]; then
  echo "LOW: need 11534336"
  exit 76
fi
echo "== window open, start A/B"
bash /d/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/ab70.sh