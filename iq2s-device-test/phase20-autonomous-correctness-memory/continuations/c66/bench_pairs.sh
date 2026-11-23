#!/bin/bash
# Phase 67 task 1: paired microbenchmark A/B (base, dot) x 8 pairs.
set -u
export MSYS_NO_PATHCONV=1
A=adb
B=/data/local/tmp/c61-inkling2
for i in 1 2 3 4 5 6 7 8; do
  for v in base dot; do
    G=$($A shell "export LD_LIBRARY_PATH=$B/variant-$v ADSP_LIBRARY_PATH=$B/libs; cd $B; ./test-backend-ops perf -b CPU -o MUL_MAT -p 'type_a=iq2_xxs.*n=8,' 2>&1 | grep 'MUL_MAT(type_a=iq2_xxs' | awk '{print \$(NF-1)}'" | tr -d '\r')
    echo "PAIR $i $v $G"
  done
done
echo "== done"