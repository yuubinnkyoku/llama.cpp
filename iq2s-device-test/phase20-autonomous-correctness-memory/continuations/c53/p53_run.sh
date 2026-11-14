#!/system/bin/sh
# Phase 53 - one measured 27B run.
#
#   sh p53_run.sh <steps> <noreplay|replay> <ngl> <tag> [sched_debug_level]
#
# Phase 50's p50_run.sh with two changes: it drives run_split53.sh (run50.sh
# with RUN_TIMEOUT instead of a hard 180 s) and it records the Phase 53 probe
# environment, so a run that used a long prompt, several segments or sampling
# says so in its own header instead of only in the caller's shell history.
# The guard, the residency prewarm and the approval file are unchanged.
set -u
BASE=/data/local/tmp/c46-27b
STEPS=${1:-8}
TOK=${2:-noreplay}
NGL=${3:-16}
TAG=${4:-run}
SD=${5:-1}
if [ -n "${OV_FILE:-}" ]; then OV=$OV_FILE; elif [ "$NGL" = "0" ]; then OV=$BASE/libs/override-base.txt; else OV=$BASE/libs/override-l57.txt; fi
export NGL OV REPLAY_MODE=$TOK
export ALLOW_CAPPED=1 SCHED_DEBUG=$SD SPLIT_TIMING=1 EARLY_REL=1 TRACE=0 TIMELINE=0
export PROFILE_ARG=${PROFILE_ARG:-0}

cap() { cat /sys/devices/system/cpu/cpu$1/cpufreq/scaling_max_freq 2>/dev/null; }
maxcap() { cat /sys/devices/system/cpu/cpu$1/cpufreq/cpuinfo_max_freq 2>/dev/null; }
cur() { cat /sys/devices/system/cpu/cpu$1/cpufreq/scaling_cur_freq 2>/dev/null; }
memval() { awk -v k="$1" '$1==k":"{print $2}' /proc/meminfo; }
pswpout() { awk '/^pswpout /{print $2}' /proc/vmstat; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }
batt() { timeout 20 dumpsys battery 2>/dev/null | awk -F': ' '/AC powered:/{a=$2} /USB powered:/{u=$2} /^  temperature:/{t=$2} /^  level:/{l=$2} END{printf "ac=%s usb=%s temp=%s level=%s", a, u, t, l}'; }

SW0=$(pswpout)
SAMPLES=$BASE/p53-$TAG.caps
: > $SAMPLES

( i=0; while [ $i -lt ${WAKE_TICKS:-120} ]; do input keyevent KEYCODE_WAKEUP >/dev/null 2>&1; sleep 20; i=$((i+1)); done ) &
KP=$!
( i=0; while [ $i -lt ${CAP_SAMPLES:-200} ]; do echo "$(cut -d' ' -f1 /proc/uptime) $(cap 0) $(cap 7) $(cur 7) $(tz0)" >> $SAMPLES; sleep 2; i=$((i+1)); done ) &
SP=$!

input keyevent KEYCODE_WAKEUP >/dev/null 2>&1
# TVer is in this list from Phase 53 on: the owner authorised stopping it.
for p in jp.hamitv.hamiand1 com.google.android.gms com.openai.chatgpt de.mm20.launcher2.nightly com.android.vending com.google.android.apps.wellbeing com.kazumaproject.markdownhelperkeyboard com.twitter.android.piko net.waterfox.android.release org.mozilla.fenix com.nikkei.newspaper com.github.android app.alextran.immich app.morphe.android.youtube com.google.android.apps.photos com.google.android.apps.walletnfcrel dev.shiggy.cord com.google.android.googlequicksearchbox; do
    am force-stop $p >/dev/null 2>&1
done
am kill-all >/dev/null 2>&1
sleep 2
am kill-all >/dev/null 2>&1
sync >/dev/null 2>&1
sleep 3
am kill-all >/dev/null 2>&1
sleep 1

echo "P53 tag $TAG ngl $NGL steps $STEPS tok $TOK sd $SD prof $PROFILE_ARG"
echo "P53_PROBE steps_max ${P53_STEPS_MAX:-64} n_ctx ${P53_N_CTX:-128} n_batch ${P53_N_BATCH:-32} n_ubatch ${P53_N_UBATCH:-32}"
echo "P53_PROBE segments ${P53_SEGMENTS:-1} prompt_tokens ${P53_PROMPT_TOKENS:-0} prompt_file ${P53_PROMPT_FILE:-none} temp ${P53_TEMP:-0} seed ${P53_SEED:-0} no_logits ${P53_NO_LOGITS:-0}"
echo "P53_PROBE run_timeout ${RUN_TIMEOUT:-180} guard_pages ${GUARD_PAGES:-262144} ov $OV"
echo "P53_START cap0 $(cap 0) cap7 $(cap 7) max0 $(maxcap 0) max7 $(maxcap 7)"
echo "P53_POWER $(batt) screen_timeout $(settings get system screen_off_timeout 2>/dev/null) bright $(settings get system screen_brightness 2>/dev/null)"
echo "P53_MEM MemAvailable $(memval MemAvailable) SwapFree $(memval SwapFree) AnonPages $(memval AnonPages) pswpout $SW0"
echo "P53_TEMP tz0 $(tz0) load $(cut -d' ' -f1-3 /proc/loadavg) uptime $(cut -d' ' -f1 /proc/uptime)"
sh $BASE/approve.sh "p53-$TAG" >/dev/null 2>&1
WANT_SKEL=${WANT_SKEL:-} sh $BASE/run_split53.sh off "$STEPS" > $BASE/p53-$TAG.out 2>&1
RC=$?
kill $KP 2>/dev/null
kill $SP 2>/dev/null
R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
SW1=$(pswpout)
echo "P53_END tag $TAG rc $RC exit $(cat $R/exit.txt 2>/dev/null) cap0 $(cap 0) cap7 $(cap 7) dir $R"
echo "P53_MEM_END MemAvailable $(memval MemAvailable) SwapFree $(memval SwapFree) pswpout $SW1 delta_pages $((SW1 - SW0))"
echo "P53_TEMP_END tz0 $(tz0) $(batt)"
echo "P53_CAP_STABLE $(awk 'NR==1{a=$2;b=$3} $2!=a||$3!=b{bad=1} END{print (bad?"NO":"YES")}' $SAMPLES) samples $(wc -l < $SAMPLES)"
echo "P53_CAP_RANGE cap0_min $(awk 'NR==1{m=$2} $2<m{m=$2} END{print m}' $SAMPLES) cap0_max $(awk 'NR==1{m=$2} $2>m{m=$2} END{print m}' $SAMPLES) cap7_min $(awk 'NR==1{m=$3} $3<m{m=$3} END{print m}' $SAMPLES)"
grep -E "CTX |PROMPT_LEN|PROMPT_SHORT|NCTX_AUTO|NBATCH_AUTO|PREFILL|SWEEP|TTFT_MS|SAMPLER|SEGMENTS|GENERATED|NODE_LINES" "$R/stdout.txt" 2>/dev/null | sed 's/^/P53_OUT /'
grep -E "^RESULT|^GENERATION" "$R/stdout.txt" 2>/dev/null | sed 's/^/P53_SUM /'
grep -c "^DECODE step" "$R/stdout.txt" 2>/dev/null | sed 's/^/P53_DECODE_LINES /'
cat $R/wall.txt 2>/dev/null | sed 's/^/P53_WALL /'
echo "SKEL $(sha256sum $BASE/libs/libggml-htp-v81.so | cut -d' ' -f1)"
echo "PROBE $(sha256sum $BASE/libs/decode-probe-wait | cut -d' ' -f1)"
exit 0
