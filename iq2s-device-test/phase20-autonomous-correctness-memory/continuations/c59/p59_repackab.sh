#!/system/bin/sh
# Phase 59 - CPU repack A/B. The harness has always passed P19_NO_EXTRA_BUFTS=1,
# which turns off the cpu backend's repacked weight buffer type. NOEXTRA=0 lets
# it through. Only the types the repack table knows (q4_0, q8_0, q2_K, q4_K,
# q5_K, q6_K, iq4_nl, mxfp4) are affected, so the model's iq2/iq3 weights are
# untouched and the logits hash must not change.
#
#   sh p59_repackab.sh <pairs> <tag> [ngl] [steps]
set -u
BASE=/data/local/tmp/c46-27b
PAIRS=${1:-2}
TAG=${2:-repackab}
NGL=${3:-0}
STEPS=${4:-4}
TIMEOUT=${TIMEOUT:-900}
SKEL=76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

run_arm() {
    ARM=$1
    PAIR=$2
    ORDER=$3
    if [ "$ARM" = "off" ]; then NE=1; else NE=0; fi
    T=$TAG-p$PAIR-$ORDER-$ARM
    C0=$(cap); Z0=$(tz0); S0=$(date +%s)
    env NOEXTRA=$NE WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
        P53_PROMPT_TOKENS=128 P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh $STEPS noreplay $NGL $T 0 > $BASE/p59-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); Z1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p59-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    PF=$(grep -oE "PREFILL seg 0 tokens 128 ms [0-9.]+" $BASE/p59-$T.out | awk '{print $7}')
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p59-$T.out | awk '{print $5}')
    RP=$(grep -oE "CPU_REPACK model buffer size = *[0-9.]+ MiB" $R/stderr.txt 2>/dev/null | awk '{print $6}')
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    echo "P59_REPACKRUN tag $T pair $PAIR order $ORDER arm $ARM noextra $NE rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) prefill128_ms ${PF:-NA} decode_ms ${DEC:-NA} repack_mib ${RP:-0} cap0 $C0-$C1 tz0 $Z0-$Z1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P59_REPACK KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

echo "P59_REPACK start ngl $NGL pairs $PAIRS cap0 $(cap) tz0 $(tz0)"
P=1
while [ $P -le $PAIRS ]; do
    if [ $((P % 2)) -eq 1 ]; then
        run_arm off $P 1
        run_arm on  $P 2
    else
        run_arm on  $P 1
        run_arm off $P 2
    fi
    P=$((P + 1))
done
echo "P59_REPACK_DONE cap0 $(cap) tz0 $(tz0)"
