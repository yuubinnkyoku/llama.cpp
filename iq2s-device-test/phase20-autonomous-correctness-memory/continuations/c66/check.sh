#!/bin/bash
set -u
export MSYS_NO_PATHCONV=1
A="adb -s 324753221196"
B=/data/local/tmp/c61-inkling2
$A shell "pidof decode-probe; echo ---; tail -2 $B/runs/c66q64_cpu/stdout.txt; echo ---; tail -2 $B/runs/c66q64_htp/stdout.txt; echo ---; cat $B/runs/c66q64_cpu/guard.txt $B/runs/c66q64_htp/guard.txt 2>/dev/null; echo ---; grep -E '^(MemAvailable|SwapFree):' /proc/meminfo; uptime; df -h /data | tail -1"