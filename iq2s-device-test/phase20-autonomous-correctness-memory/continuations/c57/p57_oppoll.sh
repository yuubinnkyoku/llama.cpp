#!/system/bin/sh
# Phase 57 - is the per batch host wait a wakeup latency or real DSP work?
#
#   sh p57_oppoll.sh <tag> [ngl] [pairs]
#
# GGML_HEXAGON_OPPOLL=0 makes the host block in dspqueue_read (the known good
# condition), =1 makes it poll with a zero timeout. If the fixed cost is a
# wakeup latency the polling arm should be much faster; if it is real work the
# two arms match. run_split53.sh forces OPPOLL=0, so this calls run53.sh
# directly with the same environment it would have set.
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
TAG=${1:-oppoll}
NGL=${2:-44}
PAIRS=${3:-2}
STEPS=${4:-4}
TIMEOUT=${TIMEOUT:-600}
OV=${OV:-$L/override-l57.txt}
SKEL=$(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

run_arm() {
    ARM=$1
    PAIR=$2
    ORDER=$3
    T=$TAG-p$PAIR-$ORDER-$ARM
    C0=$(cap); T0=$(tz0); S0=$(date +%s)
    sh $BASE/approve.sh "p57-$T" >/dev/null 2>&1
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        OPPOLL=$ARM NGL=$NGL OV=$OV ALLOW_CAPPED=1 \
        GGML_HEXAGON_SPLIT=0 GGML_HEXAGON_SPLIT_SEQ=0 GGML_HEXAGON_DMA64=0 \
        EARLY_REL=1 TIMELINE=0 TRACE=0 SPLIT_TIMING=1 SCHED_DEBUG=0 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 PROFILE_ARG=0 \
        P53_PROMPT_TOKENS=128 P53_NO_LOGITS=1 \
        sh $BASE/run53.sh run $STEPS noreplay 0 7 > $BASE/p57-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); T1=$(tz0)
    R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    PF=$(grep -oE "PREFILL seg 0 tokens 128 ms [0-9.]+" $R/stdout.txt | awk '{print $7}')
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $R/stdout.txt | awk '{print $5}')
    echo "P57_OPPOLLRUN tag $T arm $ARM ngl $NGL rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) prefill128_ms ${PF:-NA} decode_ms ${DEC:-NA} cap0 $C0-$C1 tz0 $T0-$T1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P57_OPPOLL KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

echo "P57_OPPOLL start skel $SKEL ngl $NGL pairs $PAIRS cap0 $(cap) tz0 $(tz0)"
P=1
while [ $P -le $PAIRS ]; do
    if [ $((P % 2)) -eq 1 ]; then
        run_arm 0 $P 1
        run_arm 1 $P 2
    else
        run_arm 1 $P 1
        run_arm 0 $P 2
    fi
    P=$((P + 1))
done
echo "P57_OPPOLL_DONE skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
