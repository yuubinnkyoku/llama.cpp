#!/system/bin/sh
# Phase 59 - sample the probe's CPU ticks while it runs.
#
#   sh p59_ticks.sh <tag> [ngl] [threads]
#
# Starts one run, then samples /proc/<pid>/stat (utime + stime) and the per
# thread ticks every 2 s until the run ends. A process using N cores advances
# its ticks at N x wall, so the ratio says whether the CPU side is parallel.
set -u
BASE=/data/local/tmp/c46-27b
TAG=${1:-ticks}
NGL=${2:-0}
NTH=${3:-7}
TIMEOUT=${TIMEOUT:-900}
SKEL=288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d
HZ=$(getconf CLK_TCK 2>/dev/null || echo 100)

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

echo "P59_TICKS start tag $TAG ngl $NGL threads $NTH hz $HZ cap0 $(cap) tz0 $(tz0)"

setsid sh -c "env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
    WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 P44_NTHREADS=$NTH \
    P53_PROMPT_TOKENS=128 P53_NO_LOGITS=1 \
    sh $BASE/p53_run.sh 4 noreplay $NGL $TAG 0 > $BASE/p59-$TAG.out 2>&1" > /dev/null 2>&1 < /dev/null &

# find the probe pid by name (the runner's childpid file is written late)
P=""
i=0
while [ $i -lt 120 ]; do
    P=$(ps -A -o PID,NAME | awk '/decode-probe/{print $1; exit}')
    [ -n "$P" ] && break
    sleep 1
    i=$((i + 1))
done
echo "P59_TICKS pid ${P:-none} after $i s"
[ -z "${P:-}" ] && { echo "P59_TICKS no probe"; exit 2; }

# wait for prefill to start
i=0
while [ $i -lt 120 ]; do
    R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    grep -q "PHASE prefill_begin" $R/stdout.txt 2>/dev/null && break
    sleep 1
    i=$((i + 1))
done
echo "P59_TICKS prefill_begin seen after $i s, run $R"

T0=$(awk '{print $14 + $15}' /proc/$P/stat 2>/dev/null)
W0=$(cut -d' ' -f1 /proc/uptime)
echo "P59_TICKS threads $(ls /proc/$P/task 2>/dev/null | wc -l)"
for t in /proc/$P/task/*; do
    echo "P59_TICKS t0 $(cat $t/comm 2>/dev/null) $(awk '{print $14 + $15}' $t/stat 2>/dev/null)"
done

S=0
while [ $S -lt 60 ]; do
    sleep 2
    S=$((S + 1))
    [ -d /proc/$P ] || break
    grep -q "PHASE prefill_end" $R/stdout.txt 2>/dev/null && break
done
T1=$(awk '{print $14 + $15}' /proc/$P/stat 2>/dev/null)
W1=$(cut -d' ' -f1 /proc/uptime)
echo "P59_TICKS window wall_s $(awk -v a=$W0 -v b=$W1 'BEGIN{printf "%.1f", b-a}') ticks $((T1 - T0)) cores $(awk -v t=$((T1 - T0)) -v h=$HZ -v a=$W0 -v b=$W1 'BEGIN{printf "%.2f", (t/h)/(b-a)}')"
for t in /proc/$P/task/*; do
    echo "P59_TICKS t1 $(cat $t/comm 2>/dev/null) $(awk '{print $14 + $15}' $t/stat 2>/dev/null)"
done

i=0
while [ $i -lt 600 ]; do
    [ -f "$R/exit.txt" ] && break
    sleep 2
    i=$((i + 1))
done
grep -E "P53_OUT PREFILL|P53_SUM RESULT forwards|P53_END" $BASE/p59-$TAG.out 2>/dev/null | sed 's/^/P59_TICKS /'
echo "P59_TICKS done cap0 $(cap) tz0 $(tz0)"
