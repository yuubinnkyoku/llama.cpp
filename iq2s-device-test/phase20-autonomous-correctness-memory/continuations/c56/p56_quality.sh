#!/system/bin/sh
# Phase 56 - quality runs: 32 step replay with logits, for NLL/argmax against the
# CPU reference (44cd99ce...). The replay file holds 32 tokens while this runs
# and is restored to the standard 4 tokens at the end.
#
#   sh p56_quality.sh <tag> "<ngl list>"
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
TAG=${1:-q}
LIST=${2:-"44 44 48"}
TIMEOUT=${TIMEOUT:-600}
OV=${OV:-$L/override-l57.txt}
MEM_MIN=${MEM_MIN_KB:-4000000}
SKEL=$(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

cp $L/replay32.txt $L/replay.txt || exit 2
echo "P56_Q start skel $SKEL ov $OV ngl $LIST timeout $TIMEOUT replay $(wc -w < $L/replay.txt) cap0 $(cap) tz0 $(tz0)"

I=0
for N in $LIST; do
    I=$((I + 1))
    T=$TAG-n$N-r$I
    C0=$(cap); T0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=$MEM_MIN OV_FILE=$OV \
        WARM_SKIP_TOUCH=${WARM_SKIP_TOUCH:-1} WARM_RES_MIN_PCT=${WARM_RES_MIN_PCT:-0} \
        sh $BASE/p53_run.sh 32 replay $N $T 0 > $BASE/p56-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); T1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p56-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    H=$(sha256sum $R/logits.f32 2>/dev/null | cut -d' ' -f1)
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p56-$T.out | awk '{print $5}')
    echo "P56_QRUN tag $T ngl $N rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) decode_ms ${DEC:-NA} logits_sha ${H:-NA} cap0 $C0-$C1 tz0 $T0-$T1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P56_KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
done

cp $BASE/replay4.bak $L/replay.txt
echo "P56_Q replay_restored $(cat $L/replay.txt) words $(wc -w < $L/replay.txt)"
echo "P56_Q_DONE skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
