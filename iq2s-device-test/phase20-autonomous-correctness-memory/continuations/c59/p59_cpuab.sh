#!/system/bin/sh
# Phase 59 - CPU backend A/B: the c44 baseline against the dotprod+i8mm build.
#
#   sh p59_cpuab.sh <pairs> <tag> [ngl] [steps]
#
# ABBA order, cap and temperature recorded at both ends of every run, sha256 of
# the CPU backend checked after every copy. The CPU backend is the only variable.
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
PAIRS=${1:-2}
TAG=${2:-cpuab}
NGL=${3:-0}
STEPS=${4:-4}
TIMEOUT=${TIMEOUT:-900}

OLD=$L/libggml-cpu.so.c44bak
NEW=$L/libggml-cpu.so.c59dot
OLD_SHA=0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9
NEW_SHA=93ad71fe3de7b1eb30bd8f8d196b702800304ebc8d90b3a72fddef592ace1eb2
SKEL=288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

[ -f "$OLD" ] || { echo "P59_CPUAB REFUSING: missing $OLD"; exit 2; }
[ -f "$NEW" ] || { echo "P59_CPUAB REFUSING: missing $NEW"; exit 2; }

run_arm() {
    ARM=$1
    PAIR=$2
    ORDER=$3
    if [ "$ARM" = "old" ]; then
        cp "$OLD" $L/libggml-cpu.so
        WANT=$OLD_SHA
    else
        cp "$NEW" $L/libggml-cpu.so
        WANT=$NEW_SHA
    fi
    chmod 644 $L/libggml-cpu.so
    GOT=$(sha256sum $L/libggml-cpu.so | cut -d' ' -f1)
    [ "$GOT" = "$WANT" ] || { echo "P59_CPUAB REFUSING cpu $GOT != $WANT"; return 2; }

    T=$TAG-p$PAIR-$ORDER-$ARM
    C0=$(cap); T0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
        P53_PROMPT_TOKENS=128 P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh $STEPS noreplay $NGL $T 0 > $BASE/p59-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); T1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p59-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    PF=$(grep -oE "PREFILL seg 0 tokens 128 ms [0-9.]+" $BASE/p59-$T.out | awk '{print $7}')
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p59-$T.out | awk '{print $5}')
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    echo "P59_CPUABRUN tag $T pair $PAIR order $ORDER arm $ARM cpu $(echo $GOT | cut -c1-8) rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) prefill128_ms ${PF:-NA} decode_ms ${DEC:-NA} cap0 $C0-$C1 tz0 $T0-$T1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P59_CPUAB KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

echo "P59_CPUAB start ngl $NGL pairs $PAIRS cap0 $(cap) tz0 $(tz0)"
P=1
while [ $P -le $PAIRS ]; do
    if [ $((P % 2)) -eq 1 ]; then
        run_arm old $P 1
        run_arm new $P 2
    else
        run_arm new $P 1
        run_arm old $P 2
    fi
    P=$((P + 1))
done

cp "$OLD" $L/libggml-cpu.so
chmod 644 $L/libggml-cpu.so
echo "P59_CPUAB_DONE cpu_restored $(sha256sum $L/libggml-cpu.so | cut -d' ' -f1)"
