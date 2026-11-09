#!/system/bin/sh
# Phase 50 - one measured 27B run with the screen held awake.
#
#   sh p50_run.sh <steps> <noreplay|replay> <ngl> <tag> [sched_debug_level]
#
# Wraps run50.sh (run49.sh with WANT from WANT_SKEL) and records the conditions
# Gate 1 needs: cap before and after, a cap sample every 2 s during the run,
# power state, temperatures, memory and swap-out deltas.
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

cap() { cat /sys/devices/system/cpu/cpu$1/cpufreq/scaling_max_freq 2>/dev/null; }
maxcap() { cat /sys/devices/system/cpu/cpu$1/cpufreq/cpuinfo_max_freq 2>/dev/null; }
cur() { cat /sys/devices/system/cpu/cpu$1/cpufreq/scaling_cur_freq 2>/dev/null; }
memval() { awk -v k="$1" '$1==k":"{print $2}' /proc/meminfo; }
pswpout() { awk '/^pswpout /{print $2}' /proc/vmstat; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }
batt() { dumpsys battery 2>/dev/null | awk -F': ' '/AC powered:/{a=$2} /USB powered:/{u=$2} /^  temperature:/{t=$2} /^  level:/{l=$2} END{printf "ac=%s usb=%s temp=%s level=%s", a, u, t, l}'; }

SW0=$(pswpout)
SAMPLES=$BASE/p50-$TAG.caps
: > $SAMPLES

# hold the screen awake: stayon needs external power, the keyevent works either way
( i=0; while [ $i -lt 40 ]; do input keyevent KEYCODE_WAKEUP >/dev/null 2>&1; sleep 20; i=$((i+1)); done ) &
KP=$!
# sample the cap so a mid-run change cannot go unnoticed
( i=0; while [ $i -lt 200 ]; do echo "$(cut -d' ' -f1 /proc/uptime) $(cap 0) $(cap 7) $(cur 7) $(tz0)" >> $SAMPLES; sleep 2; i=$((i+1)); done ) &
SP=$!

input keyevent KEYCODE_WAKEUP >/dev/null 2>&1
# Phase 49.5 stopped the first list. The rest are user apps that hold anon pages
# and push the run over the 1024 MiB swap-out guard. None of them is a system
# service; they restart when the user opens them. systemui, system and the
# thermal/ZRAM stacks are never touched.
for p in com.google.android.gms com.openai.chatgpt de.mm20.launcher2.nightly com.android.vending com.google.android.apps.wellbeing com.kazumaproject.markdownhelperkeyboard com.twitter.android.piko net.waterfox.android.release org.mozilla.fenix com.nikkei.newspaper com.github.android app.alextran.immich app.morphe.android.youtube com.google.android.apps.photos com.google.android.apps.walletnfcrel dev.shiggy.cord com.google.android.googlequicksearchbox; do
    am force-stop $p >/dev/null 2>&1
done
am kill-all >/dev/null 2>&1
sleep 2
am kill-all >/dev/null 2>&1
sync >/dev/null 2>&1
sleep 3
am kill-all >/dev/null 2>&1
sleep 1

echo "P50 tag $TAG ngl $NGL steps $STEPS tok $TOK sd $SD prof $PROFILE_ARG"
echo "P50_START cap0 $(cap 0) cap7 $(cap 7) max0 $(maxcap 0) max7 $(maxcap 7)"
echo "P50_POWER $(batt) screen_timeout $(settings get system screen_off_timeout 2>/dev/null) bright $(settings get system screen_brightness 2>/dev/null)"
echo "P50_MEM MemAvailable $(memval MemAvailable) SwapFree $(memval SwapFree) AnonPages $(memval AnonPages) pswpout $SW0"
echo "P50_TEMP tz0 $(tz0) load $(cut -d' ' -f1-3 /proc/loadavg) uptime $(cut -d' ' -f1 /proc/uptime)"
echo "P50_BG $(dumpsys meminfo 2>/dev/null | sed -n '5,9p' | tr '\n' ' ')"
sh $BASE/approve.sh "p50-$TAG" >/dev/null 2>&1
WANT_SKEL=${WANT_SKEL:-} sh $BASE/run_split50.sh off "$STEPS" > $BASE/p50-$TAG.out 2>&1
RC=$?
kill $KP 2>/dev/null
kill $SP 2>/dev/null
R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
SW1=$(pswpout)
echo "P50_END tag $TAG rc $RC exit $(cat $R/exit.txt 2>/dev/null) cap0 $(cap 0) cap7 $(cap 7) dir $R"
echo "P50_MEM_END MemAvailable $(memval MemAvailable) SwapFree $(memval SwapFree) pswpout $SW1 delta_pages $((SW1 - SW0))"
echo "P50_TEMP_END tz0 $(tz0) $(batt)"
echo "P50_CAP_STABLE $(awk 'NR==1{a=$2;b=$3} $2!=a||$3!=b{bad=1} END{print (bad?"NO":"YES")}' $SAMPLES) samples $(wc -l < $SAMPLES)"
echo "P50_CAP_RANGE cap0_min $(awk 'NR==1{m=$2} $2<m{m=$2} END{print m}' $SAMPLES) cap0_max $(awk 'NR==1{m=$2} $2>m{m=$2} END{print m}' $SAMPLES) cap7_min $(awk 'NR==1{m=$3} $3<m{m=$3} END{print m}' $SAMPLES)"
grep -E "DECODE step|THERMAL_STATE|RESIDENCY_RANGES tag pre |RESULT" "$R/stdout.txt" 2>/dev/null | sed 's/^/P50_OUT /'
grep -E "ggml-sched-(split|wait)" "$R/stderr.txt" 2>/dev/null | sed 's/^/P50_SPLIT /'
echo "SKEL $(sha256sum $BASE/libs/libggml-htp-v81.so | cut -d' ' -f1)"
echo "NODE_LINES $(grep -c '^node #' $R/stderr.txt 2>/dev/null)"
exit 0
