#!/system/bin/sh
# Phase 59 - CPU thread count sweep, done correctly.
#
#   sh p59_threads.sh "<thread list>" <tag> [ngl]
#
# run53.sh takes the thread count as its fifth argument, which run_split53.sh
# fills from $THREADS. Setting P44_NTHREADS in the caller does nothing (phase 59
# first attempt made that mistake and every run used 7). This script sets
# THREADS, and reads the value back from the run manifest to prove it took.
set -u
BASE=/data/local/tmp/c46-27b
LIST=${1:-"6 7 8"}
TAG=${2:-thr}
NGL=${3:-0}
TIMEOUT=${TIMEOUT:-900}
SKEL=${SKEL:-288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d}

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

echo "P59_THR start list $LIST ngl $NGL cap0 $(cap) tz0 $(tz0)"
for T in $LIST; do
    TAG2=$TAG-t$T
    C0=$(cap); Z0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 THREADS=$T \
        P53_PROMPT_TOKENS=128 P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay $NGL $TAG2 0 > $BASE/p59-$TAG2.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); Z1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p59-$TAG2.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    USED=$(grep -oE "P44_NTHREADS=[0-9]+" $R/manifest.txt 2>/dev/null | head -1)
    PF=$(grep -oE "PREFILL seg 0 tokens 128 ms [0-9.]+" $BASE/p59-$TAG2.out | awk '{print $7}')
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p59-$TAG2.out | awk '{print $5}')
    RB=$(awk '/^READ_BYTES/{v=$2} END{print v}' $R/events.txt 2>/dev/null)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    echo "P59_THRRUN tag $TAG2 threads $T used $USED rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) prefill128_ms ${PF:-NA} decode_ms ${DEC:-NA} read_bytes ${RB:-NA} cap0 $C0-$C1 tz0 $Z0-$Z1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P59_THR KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
done
echo "P59_THR_DONE cap0 $(cap) tz0 $(tz0)"
