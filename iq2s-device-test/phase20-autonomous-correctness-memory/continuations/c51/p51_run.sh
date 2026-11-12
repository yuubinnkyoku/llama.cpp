#!/system/bin/sh
# Phase 51 - one measured 27B run with the screen held awake.
#
#   P51_MODE=<off|seq|async> sh p51_run.sh <steps> <noreplay|replay> <ngl> <tag> [sched_debug_level]
#
# Same envelope as p50_run.sh (force-stop list, cap sampling, memory record),
# with one change: the split mode is a parameter instead of a fixed "off".
# SPLIT_WEIGHTS in the environment selects the split list, so a run can split
# split33.txt or split34.txt without editing anything.
set -u
BASE=/data/local/tmp/c46-27b
STEPS=${1:-8}
TOK=${2:-noreplay}
NGL=${3:-16}
TAG=${4:-run}
SD=${5:-1}
MODE=${P51_MODE:-off}
if [ -n "${OV_FILE:-}" ]; then OV=$OV_FILE; elif [ "$NGL" = "0" ]; then OV=$BASE/libs/override-base.txt; else OV=$BASE/libs/override-l57.txt; fi
export NGL OV REPLAY_MODE=$TOK
export ALLOW_CAPPED=1 SCHED_DEBUG=$SD SPLIT_TIMING=1 EARLY_REL=1 TRACE=${TRACE:-0} TIMELINE=${TIMELINE:-0}
export PROFILE_ARG=${PROFILE_ARG:-0}

cap() { cat /sys/devices/system/cpu/cpu$1/cpufreq/scaling_max_freq 2>/dev/null; }
maxcap() { cat /sys/devices/system/cpu/cpu$1/cpufreq/cpuinfo_max_freq 2>/dev/null; }
cur() { cat /sys/devices/system/cpu/cpu$1/cpufreq/scaling_cur_freq 2>/dev/null; }
memval() { awk -v k="$1" '$1==k":"{print $2}' /proc/meminfo; }
pswpout() { awk '/^pswpout /{print $2}' /proc/vmstat; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }
batt() { timeout 20 dumpsys battery 2>/dev/null | awk -F': ' '/AC powered:/{a=$2} /USB powered:/{u=$2} /^  temperature:/{t=$2} /^  level:/{l=$2} END{printf "ac=%s usb=%s temp=%s level=%s", a, u, t, l}'; }

SW0=$(pswpout)
SAMPLES=$BASE/p51-$TAG.caps
: > $SAMPLES

# hold the screen awake: stayon needs external power, the keyevent works either way
( i=0; while [ $i -lt 40 ]; do input keyevent KEYCODE_WAKEUP >/dev/null 2>&1; sleep 20; i=$((i+1)); done ) &
KP=$!
# sample the cap so a mid-run change cannot go unnoticed
( i=0; while [ $i -lt 200 ]; do echo "$(cut -d' ' -f1 /proc/uptime) $(cap 0) $(cap 7) $(cur 7) $(tz0)" >> $SAMPLES; sleep 2; i=$((i+1)); done ) &
SP=$!

input keyevent KEYCODE_WAKEUP >/dev/null 2>&1
# app.morphe.android.youtube is out of this list: the owner asked to keep it alive
# until 24:00 on 2026-11-12. The list only frees memory for a run, so leaving one
# app in it costs room and nothing else.
for p in com.google.android.gms com.openai.chatgpt de.mm20.launcher2.nightly com.android.vending com.google.android.apps.wellbeing com.kazumaproject.markdownhelperkeyboard com.twitter.android.piko net.waterfox.android.release org.mozilla.fenix com.nikkei.newspaper com.github.android app.alextran.immich com.google.android.apps.photos com.google.android.apps.walletnfcrel dev.shiggy.cord com.google.android.googlequicksearchbox; do
    am force-stop $p >/dev/null 2>&1
done
am kill-all >/dev/null 2>&1
sleep 2
am kill-all >/dev/null 2>&1
sync >/dev/null 2>&1
sleep 3
am kill-all >/dev/null 2>&1
sleep 1

echo "P51 tag $TAG mode $MODE ngl $NGL steps $STEPS tok $TOK sd $SD prof $PROFILE_ARG"
echo "P51_START cap0 $(cap 0) cap7 $(cap 7) max0 $(maxcap 0) max7 $(maxcap 7)"
echo "P51_POWER $(batt) screen_timeout $(settings get system screen_off_timeout 2>/dev/null)"
echo "P51_MEM MemAvailable $(memval MemAvailable) SwapFree $(memval SwapFree) AnonPages $(memval AnonPages) pswpout $SW0"
echo "P51_TEMP tz0 $(tz0) load $(cut -d' ' -f1-3 /proc/loadavg) uptime $(cut -d' ' -f1 /proc/uptime)"
sh $BASE/approve.sh "p51-$TAG" >/dev/null 2>&1
WANT_SKEL=${WANT_SKEL:-} sh $BASE/run_split50.sh "$MODE" "$STEPS" ${P51_DUMP:-} > $BASE/p51-$TAG.out 2>&1
RC=$?
kill $KP 2>/dev/null
kill $SP 2>/dev/null
R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
SW1=$(pswpout)
echo "P51_END tag $TAG mode $MODE rc $RC exit $(cat $R/exit.txt 2>/dev/null) cap0 $(cap 0) cap7 $(cap 7) dir $R"
echo "P51_MEM_END MemAvailable $(memval MemAvailable) SwapFree $(memval SwapFree) pswpout $SW1 delta_pages $((SW1 - SW0))"
echo "P51_TEMP_END tz0 $(tz0) $(batt)"
echo "P51_CAP_STABLE $(awk 'NR==1{a=$2;b=$3} $2!=a||$3!=b{bad=1} END{print (bad?"NO":"YES")}' $SAMPLES) samples $(wc -l < $SAMPLES)"
echo "P51_CAP_RANGE cap0_min $(awk 'NR==1{m=$2} $2<m{m=$2} END{print m}' $SAMPLES) cap0_max $(awk 'NR==1{m=$2} $2>m{m=$2} END{print m}' $SAMPLES)"
echo "P51_REPRO $(cat $BASE/p51-$TAG.out 2>/dev/null | grep -c 'REFUSING')"
grep -E "DECODE step|THERMAL_STATE|RESULT" "$R/stdout.txt" 2>/dev/null | sed 's/^/P51_OUT /'
grep -E "^SPLIT_ENV|^SPLIT_RUN|^SPLIT_WEIGHTS" $BASE/p51-$TAG.out 2>/dev/null | sed 's/^/P51_RAW /'
grep -E "split weight blk.63.attn_output|split .* not taken|split .* rejected|dspqueue_read|split verify" "$R/stderr.txt" 2>/dev/null | sed 's/^/P51_SPLIT /'
grep -c "split weight" "$R/stderr.txt" 2>/dev/null | sed 's/^/P51_SPLIT_COUNT /'
echo "SKEL $(sha256sum $BASE/libs/libggml-htp-v81.so | cut -d' ' -f1)"
exit 0
