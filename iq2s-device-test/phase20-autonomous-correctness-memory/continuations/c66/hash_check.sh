#!/bin/bash
# Phase 66: confirm the frozen baseline hash 65dd1c56 after the c66 work.
set -u
export MSYS_NO_PATHCONV=1
A="adb -s 324753221196"
B=/data/local/tmp/c61-inkling2
for tag in h1 h2; do
  $A shell "REPLAY=$B/replay4.txt NB=32 NUB=8 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-inkling.txt sh $B/p62_probe.sh c66hash_$tag 4 48 HTP0:0" >/dev/null 2>&1
  echo "== $tag"
  $A shell "sha256sum $B/runs/c66hash_$tag/logits.f32; grep -a 'model buffer size' $B/runs/c66hash_$tag/stderr.txt | grep -a HTP; grep -a 'TOKEN step 0' $B/runs/c66hash_$tag/stdout.txt; grep -a swap_total $B/runs/c66hash_$tag/manifest.txt"
done
echo "== done"