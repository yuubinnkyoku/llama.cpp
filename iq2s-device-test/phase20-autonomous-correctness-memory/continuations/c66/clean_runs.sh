#!/bin/bash
set -u
export MSYS_NO_PATHCONV=1
A=adb
B=/data/local/tmp/c61-inkling2
echo "== before"
$A shell "df -h /data | tail -1; du -sh $B/runs 2>/dev/null"
echo "== delete old logits > 5 MiB (keep manifests/rc)"
$A shell "find $B/runs -name 'logits.f32' -size +5M -delete 2>/dev/null; find $B/runs -name 'logits.f32.*.telemetry' -size +5M -delete 2>/dev/null"
echo "== after"
$A shell "df -h /data | tail -1; du -sh $B/runs 2>/dev/null"