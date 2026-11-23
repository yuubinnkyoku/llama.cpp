#!/bin/bash
set -u
export MSYS_NO_PATHCONV=1
A=adb
B=/data/local/tmp/c61-inkling2
for r in c68c_a c68c_b; do
  echo "== $r"
  $A shell "grep -a -E 'dsp-error|dsp-rsp|canary|abort|failed|abort' $B/runs/$r/stderr.txt | tail -8"
done