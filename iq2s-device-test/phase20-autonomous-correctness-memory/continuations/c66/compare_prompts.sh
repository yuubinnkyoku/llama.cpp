#!/bin/bash
set -u
export MSYS_NO_PATHCONV=1
A=adb
B=/data/local/tmp/c61-inkling2
echo "== probe file"
$A shell "ls -l $B/libs/decode-probe; sha256sum $B/libs/decode-probe"
for r in c65hash_h1 c66hash_h1 c66q64_cpu c67e_a c69_base; do
  echo "== $r"
  $A shell "grep -a PROMPT_IDS $B/runs/$r/stdout.txt 2>/dev/null"
done