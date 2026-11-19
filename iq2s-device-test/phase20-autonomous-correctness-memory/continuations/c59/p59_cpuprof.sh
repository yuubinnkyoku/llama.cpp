#!/system/bin/sh
# Phase 59 - where does the CPU-only prefill time go?
#
#   sh p59_cpuprof.sh <tag> [ngl] [threads]
#
# Starts one probe run, waits for the child pid, then samples /proc/<pid>/stat
# (utime + stime, in clock ticks) and the per-thread ticks while prefill runs.
# A process that uses N cores advances its ticks at N x wall time, so the ratio
# says whether the CPU side is actually parallel. Also records the per-thread
# comm names so the ggml worker threads can be told apart from the rest.
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
TAG=${1:-cpuprof}
NGL=${2:-0}
NTH=${3:-7}
TIMEOUT=${TIMEOUT:-900}
SKEL=288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }
HZ=$(getconf CLK_TCK 2>/dev/null || echo 100)

echo "P59_CPUPROF start tag $TAG ngl $NGL threads $NTH hz $HZ cap0 $(cap) tz0 $(tz0)"

# remember the newest run dir so the wait below only accepts a NEW one
PREV=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)

setsid sh -c "env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
    WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 P44_NTHREADS=$NTH \
    P53_PROMPT_TOKENS=128 P53_NO_LOGITS=1 \
    sh $BASE/p53_run.sh 4 noreplay $NGL $TAG 0 > $BASE/p59-$TAG.out 2>&1" > /dev/null 2>&1 < /dev/null &

# wait for a NEW run directory and the child pid
i=0
R=""
while [ $i -lt 300 ]; do
    R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    if [ -n "$R" ] && [ "$R" != "$PREV" ] && [ -f "$R/childpid" ] && grep -q "PHASE prefill_begin" $R/stdout.txt 2>/dev/null; then
        break
    fi
    sleep 1
    i=$((i + 1))
done
C=$(cat $R/childpid 2>/dev/null)
echo "P59_CPUPROF run $R child ${C:-none} waited $i s"

if [ -z "${C:-}" ] || [ ! -d /proc/$C ]; then
    echo "P59_CPUPROF no child, abort"
    exit 2
fi

read_t() { awk '{print $14 + $15}' /proc/$C/stat 2>/dev/null; }
T0=$(read_t)
W0=$(cut -d' ' -f1 /proc/uptime)
echo "P59_CPUPROF threads $(ls /proc/$C/task | wc -l)"
for t in /proc/$C/task/*; do
    echo "P59_CPUPROF thread $(cat $t/comm 2>/dev/null) $(awk '{print $14 + $15}' $t/stat 2>/dev/null)"
done

# sample while prefill runs
S=0
while [ $S -lt 40 ]; do
    sleep 2
    S=$((S + 1))
    grep -q "PHASE prefill_end" $R/stdout.txt 2>/dev/null && break
done
T1=$(read_t)
W1=$(cut -d' ' -f1 /proc/uptime)
echo "P59_CPUPROF window wall_s $(awk -v a=$W0 -v b=$W1 'BEGIN{printf "%.1f", b-a}') ticks $((T1 - T0)) cores $(awk -v t=$((T1 - T0)) -v h=$HZ -v a=$W0 -v b=$W1 'BEGIN{printf "%.2f", (t/h)/(b-a)}')"

# per thread at the end of the window
for t in /proc/$C/task/*; do
    echo "P59_CPUPROF endthread $(cat $t/comm 2>/dev/null) $(awk '{print $14 + $15}' $t/stat 2>/dev/null)"
done

# wait for the run to finish
i=0
while [ $i -lt 600 ]; do
    [ -f "$R/exit.txt" ] && break
    sleep 2
    i=$((i + 1))
done
grep -E "P53_OUT PREFILL|P53_SUM RESULT forwards|P53_END" $BASE/p59-$TAG.out 2>/dev/null | sed 's/^/P59_CPUPROF /'
echo "P59_CPUPROF done cap0 $(cap) tz0 $(tz0)"
