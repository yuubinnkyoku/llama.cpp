#!/system/bin/sh
# Phase 49.5 - one measured 27B run with the screen held awake.
#   sh p495_run.sh <steps> <noreplay|replay> <ngl> <tag> [sched_debug_level]
set -u
BASE=/data/local/tmp/c46-27b
STEPS=${1:-8}
TOK=${2:-noreplay}
NGL=${3:-16}
TAG=${4:-run}
SD=${5:-1}
if [ "$NGL" = "0" ]; then OV=$BASE/libs/override-base.txt; else OV=$BASE/libs/override-l57.txt; fi
export NGL OV REPLAY_MODE=$TOK
export ALLOW_CAPPED=1 SCHED_DEBUG=$SD SPLIT_TIMING=1 EARLY_REL=1 TRACE=0 TIMELINE=0
export PROFILE_ARG=${PROFILE_ARG:-0}
cap0() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
cap7() { cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq 2>/dev/null; }
mem() { awk '/^(MemAvailable|AnonPages):/{printf "%s=%s ", $1, $2}' /proc/meminfo; }
( i=0; while [ $i -lt 40 ]; do input keyevent KEYCODE_WAKEUP >/dev/null 2>&1; sleep 20; i=$((i+1)); done ) &
KP=$!
input keyevent KEYCODE_WAKEUP >/dev/null 2>&1
for p in com.google.android.gms com.openai.chatgpt de.mm20.launcher2.nightly com.android.vending com.google.android.apps.wellbeing com.kazumaproject.markdownhelperkeyboard; do
    am force-stop $p >/dev/null 2>&1
done
am kill-all >/dev/null 2>&1
sleep 2
echo "P495 tag $TAG ngl $NGL steps $STEPS tok $TOK sd $SD cap0 $(cap0) cap7 $(cap7) $(mem) load $(cut -d' ' -f1 /proc/loadavg) t $(cut -d' ' -f1 /proc/uptime)"
sh $BASE/approve.sh "p495-$TAG" >/dev/null 2>&1
sh $BASE/run_split49.sh off "$STEPS" > $BASE/p495-$TAG.out 2>&1
RC=$?
kill $KP 2>/dev/null
R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
echo "P495_END tag $TAG rc $RC exit $(cat $R/exit.txt 2>/dev/null) cap0 $(cap0) cap7 $(cap7) dir $R"
grep -E "DECODE step|THERMAL_STATE|RESIDENCY_RANGES tag pre |RESULT" "$R/stdout.txt" 2>/dev/null | sed 's/^/P495_OUT /'
grep -E "ggml-sched-(split|wait)" "$R/stderr.txt" 2>/dev/null | sed 's/^/P495_SPLIT /'
echo "NODE_LINES $(grep -c '^node #' $R/stderr.txt 2>/dev/null)"
exit 0