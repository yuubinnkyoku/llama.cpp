#!/bin/bash
# Gated cleanup: free memory keeping the foreground app.
set -u
export MSYS_NO_PATHCONV=1
A=adb
FG=$($A shell "dumpsys activity activities 2>/dev/null | sed -n 's/.*topResumedActivity=ActivityRecord{[^ ]* [^ ]* \([^\/]*\)\/.*/\1/p' | head -1" | tr -d '\r')
echo "FG=$FG"
for p in net.waterfox.android.release dev.shiggy.cord com.google.android.googlequicksearchbox com.kazumaproject.markdownhelperkeyboard.npu com.kazumaproject.markdownhelperkeyboard com.android.vending jp.nicovideo.android com.facebook.aura com.android.chrome app.morphe.android.youtube com.google.android.apps.photos org.mozilla.fenix com.google.android.apps.messaging jp.naver.line.android org.torproject.vpn; do
  [ -n "$FG" ] && [ "$p" = "$FG" ] && continue
  $A shell "am force-stop $p" >/dev/null 2>&1
done
$A shell "am kill-all >/dev/null 2>&1; sleep 2; am kill-all >/dev/null 2>&1; sync; sleep 4; grep -E '^(MemAvailable|SwapFree):' /proc/meminfo; uptime; df -h /data | tail -1"