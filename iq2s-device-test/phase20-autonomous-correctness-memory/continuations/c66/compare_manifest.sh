#!/bin/bash
set -u
export MSYS_NO_PATHCONV=1
A=adb
B=/data/local/tmp/c61-inkling2
echo "== c67e_a manifest"
$A shell "cat $B/runs/c67e_a/manifest.txt 2>/dev/null | head -20"
echo "== c69_base manifest"
$A shell "cat $B/runs/c69_base/manifest.txt 2>/dev/null | head -20"
echo "== model"
$A shell "ls -l $B/models/Inkling-2-Mini-AD-IQ2_XXS.gguf"