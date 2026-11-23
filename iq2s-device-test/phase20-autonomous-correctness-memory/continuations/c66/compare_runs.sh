#!/bin/bash
set -u
export MSYS_NO_PATHCONV=1
A=adb
B=/data/local/tmp/c61-inkling2
echo "== c69_base stdout head"
$A shell "head -20 $B/runs/c69_base/stdout.txt"
echo "== c67e_a stdout head"
$A shell "head -20 $B/runs/c67e_a/stdout.txt 2>/dev/null"
echo "== c69_base tail"
$A shell "tail -6 $B/runs/c69_base/stdout.txt"