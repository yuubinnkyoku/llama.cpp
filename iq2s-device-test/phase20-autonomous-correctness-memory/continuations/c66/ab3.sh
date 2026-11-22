#!/bin/bash
# Phase 66 A/B v3: 8 alternating runs for 4 samples per variant.
set -u
export MSYS_NO_PATHCONV=1
A="adb -s 324753221196"
B=/data/local/tmp/c61-inkling2
FG=$($A shell "dumpsys activity activities 2>/dev/null | sed -n 's/.*topResumedActivity=ActivityRecord{[^ ]* [^ ]* \([^\/]*\)\/.*/\1/p' | head -1" | tr -d '\r')
echo "FG=$FG"
for p in net.waterfox.android.release dev.shiggy.cord com.google.android.googlequicksearchbox com.kazumaproject.markdownhelperkeyboard.npu com.kazumaproject.markdownhelperkeyboard com.android.vending jp.nicovideo.android com.facebook.aura com.android.chrome app.morphe.android.youtube com.google.android.apps.photos org.mozilla.fenix com.google.android.apps.messaging jp.naver.line.android org.torproject.vpn; do
  [ -n "$FG" ] && [ "$p" = "$FG" ] && continue
  $A shell "am force-stop $p" >/dev/null 2>&1
done
$A shell "am kill-all >/dev/null 2>&1; sleep 2; am kill-all >/dev/null 2>&1; sync; sleep 4; grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
echo "== variant hashes"
$A shell "sha256sum $B/variant-base/libggml-cpu.so $B/variant-dot/libggml-cpu.so"
run() {
  echo "== run $1 $(basename $2) $3 steps"
  $A shell "LIBDIR=$2 NB=32 NUB=8 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-down-cpu.txt sh $B/p62_probe.sh c66v3_$1 $3 48 HTP0:0" >/dev/null 2>&1
  $A shell "grep -a -E 'RESULT forwards|TOKEN step 0 ' $B/runs/c66v3_$1/stdout.txt; grep -a swap_total $B/runs/c66v3_$1/manifest.txt"
}
run warm $B/variant-base 8
for seq in a1 b1 b2 a2 a3 b3 b4 a4; do
  case $seq in
    a*) LIB=$B/variant-base ;;
    b*) LIB=$B/variant-dot ;;
  esac
  run $seq $LIB 16
done
echo "== done"