#!/bin/bash
# Phase 66 A/B v2: baseline (AC7CC94C) vs dotprod+i8mm (A3B7D4E7).
# One gate invocation: cleanup, install variants, warmup, A/B/B/A 16 steps.
set -u
export MSYS_NO_PATHCONV=1
SER=324753221196
A="adb -s $SER"
B=/data/local/tmp/c61-inkling2
SRC=D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66

FG=$($A shell "dumpsys activity activities 2>/dev/null | sed -n 's/.*topResumedActivity=ActivityRecord{[^ ]* [^ ]* \([^\/]*\)\/.*/\1/p' | head -1" | tr -d '\r')
echo "FG=$FG"
for p in net.waterfox.android.release dev.shiggy.cord com.google.android.googlequicksearchbox com.kazumaproject.markdownhelperkeyboard.npu com.kazumaproject.markdownhelperkeyboard com.android.vending jp.nicovideo.android com.facebook.aura com.android.chrome app.morphe.android.youtube com.google.android.apps.photos org.mozilla.fenix com.google.android.apps.messaging jp.naver.line.android org.torproject.vpn; do
  [ -n "$FG" ] && [ "$p" = "$FG" ] && continue
  $A shell "am force-stop $p" >/dev/null 2>&1
done
$A shell "am kill-all >/dev/null 2>&1; sleep 2; am kill-all >/dev/null 2>&1; sync; sleep 4; grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"

$A push "$SRC/p62_probe.sh" $B/p62_probe.sh
$A push "$SRC/libggml-cpu-base.so" $B/libggml-cpu-base.so
$A push "$SRC/libggml-cpu-dot.so" $B/libggml-cpu-dot.so
$A shell "chmod 755 $B/p62_probe.sh; rm -rf $B/variant-base $B/variant-dot; cp -a $B/libs $B/variant-base; cp -a $B/libs $B/variant-dot; cp $B/libggml-cpu-base.so $B/variant-base/libggml-cpu.so; cp $B/libggml-cpu-dot.so $B/variant-dot/libggml-cpu.so"
echo "== variant hashes (expect base ac7cc94c..., dot a3b7d4e7...)"
H=$($A shell "sha256sum $B/variant-base/libggml-cpu.so $B/variant-dot/libggml-cpu.so" | tr -d '\r')
echo "$H"
case "$H" in
  *ac7cc94c*ac7cc94c*) echo "ABORT: both variants are the baseline"; exit 3 ;;
esac
echo "== mem before"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"

run() { # $1 tag, $2 libdir, $3 steps
  echo "== run $1 $(basename $2)"
  $A shell "LIBDIR=$2 NB=32 NUB=8 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-down-cpu.txt sh $B/p62_probe.sh c66v2_$1 $3 48 HTP0:0" >/dev/null 2>&1
  $A shell "grep -a -E 'RESULT forwards|TOKEN step 0 ' $B/runs/c66v2_$1/stdout.txt; grep -a 'model buffer size' $B/runs/c66v2_$1/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c66v2_$1/manifest.txt"
}

run warm $B/variant-base 8
run a1 $B/variant-base 16
run b1 $B/variant-dot 16
run b2 $B/variant-dot 16
run a2 $B/variant-base 16
echo "== done"
