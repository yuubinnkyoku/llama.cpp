#!/bin/bash
# Phase 66 A/B: CPU backend variants in one session.
# baseline (AC7CC94C) vs dotprod+i8mm (A3B7D4E7), alternating A/B/B/A,
# decode 16 steps each with the down experts on the CPU (light footprint).
set -u
SER=324753221196
A="adb -s $SER"
B=/data/local/tmp/c61-inkling2
SRC=/d/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66
$A push "$SRC/p62_probe.sh" $B/p62_probe.sh >/dev/null
$A push "$SRC/libggml-cpu-base.so" $B/libggml-cpu-base.so >/dev/null
$A push "$SRC/libggml-cpu-dot.so" $B/libggml-cpu-dot.so >/dev/null
$A shell "chmod 755 $B/p62_probe.sh; rm -rf $B/variant-base $B/variant-dot; cp -a $B/libs $B/variant-base; cp -a $B/libs $B/variant-dot; cp $B/libggml-cpu-base.so $B/variant-base/libggml-cpu.so; cp $B/libggml-cpu-dot.so $B/variant-dot/libggml-cpu.so; sha256sum $B/variant-base/libggml-cpu.so $B/variant-dot/libggml-cpu.so"
echo "== mem before"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
for tag in a1 b1 b2 a2; do
  case $tag in
    a1|a2) LIB=$B/variant-base ;;
    b1|b2) LIB=$B/variant-dot ;;
  esac
  echo "== run $tag lib $(basename $LIB)"
  $A shell "LIBDIR=$LIB NB=32 NUB=8 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-down-cpu.txt sh $B/p62_probe.sh c66_$tag 16 48 HTP0:0" >/dev/null 2>&1
  $A shell "grep -a -E 'RESULT forwards|TOKEN step 0 ' $B/runs/c66_$tag/stdout.txt; grep -a 'model buffer size' $B/runs/c66_$tag/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c66_$tag/manifest.txt; grep -a -E '^(MemAvailable|SwapFree):' /proc/meminfo"
done
echo "== done"
