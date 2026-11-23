#!/bin/bash
set -u
export MSYS_NO_PATHCONV=1
A=adb
B=/data/local/tmp/c61-inkling2
$A shell "df -h /data | tail -1; du -sh $B/runs 2>/dev/null; grep -E '^(MemAvailable|SwapFree):' /proc/meminfo; uptime; pidof decode-probe"