#!/system/bin/sh
# Phase 57 - 64 step quality at the default placement, against cpu only.
#
#   sh p57_q64.sh <tag>
#
# Uses replay64.txt (64 forced tokens). The probe's step limit is 64 by default.
# NGL=0 keeps override-base.txt (the cpu reference), NGL=44 the phase 56 file.
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
TAG=${1:-q64}
TIMEOUT=${TIMEOUT:-1800}
SKEL=288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

ACT=$(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)
[ "$ACT" = "$SKEL" ] || { echo "P57_Q64 REFUSING skel $ACT"; exit 2; }

cp $L/replay64.txt $L/replay.txt || exit 2
echo "P57_Q64 start skel $ACT replay $(wc -w < $L/replay.txt) cap0 $(cap) tz0 $(tz0)"

for N in 0 44; do
    T=$TAG-n$N
    if [ "$N" = "0" ]; then OVN=""; else OVN="OV_FILE=$L/override-l57.txt"; fi
    C0=$(cap); T0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 $OVN \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
        sh $BASE/p53_run.sh 64 replay $N $T 0 > $BASE/p57-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); T1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p57-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    H=$(sha256sum $R/logits.f32 2>/dev/null | cut -d' ' -f1)
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p57-$T.out | awk '{print $5}')
    echo "P57_Q64RUN tag $T ngl $N rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) decode_ms ${DEC:-NA} logits_sha ${H:-NA} cap0 $C0-$C1 tz0 $T0-$T1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P57_Q64 KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
done

cp $BASE/replay4.bak $L/replay.txt
echo "P57_Q64 replay_restored $(cat $L/replay.txt) words $(wc -w < $L/replay.txt)"
echo "P57_Q64_DONE skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
