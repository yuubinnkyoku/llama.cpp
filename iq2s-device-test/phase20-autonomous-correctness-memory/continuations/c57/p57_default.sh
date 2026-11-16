#!/system/bin/sh
# Phase 57 - goal 1: the default placement is NGL=44. Two 32 step runs with the
# ngl argument left empty, so the runner's default is what is under test. The
# expected logits hash is the phase 56 ngl 44 hash e9da92f8.
#
#   sh p57_default.sh <tag>
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
TAG=${1:-def}
TIMEOUT=${TIMEOUT:-1800}
SKEL=288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d
WANT=e9da92f8d7280750f5c62a8bf3489fdc7332fc30e1774d708f1df7d65c0ec6c1

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

ACT=$(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)
[ "$ACT" = "$SKEL" ] || { echo "P57_DEF REFUSING skel $ACT"; exit 2; }

cp $L/replay32.txt $L/replay.txt || exit 2
echo "P57_DEF start skel $ACT replay $(wc -w < $L/replay.txt) cap0 $(cap) tz0 $(tz0)"

I=0
for T2 in r1 r2; do
    I=$((I + 1))
    T=$TAG-$T2
    C0=$(cap); T0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        sh $BASE/p53_run.sh 32 replay "" $T 0 > $BASE/p57-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); T1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p57-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    H=$(sha256sum $R/logits.f32 2>/dev/null | cut -d' ' -f1)
    if [ "$H" = "$WANT" ]; then V=MATCH; else V=MISMATCH; fi
    NGLUSED=$(grep -oE "P53 tag [^ ]+ ngl [0-9]+" $BASE/p57-$T.out | awk '{print $5}')
    WARM=$(grep -oE "warm_skip [0-9]+ warm_min [0-9]+" $BASE/p57-$T.out | head -1)
    echo "P57_DEFRUN tag $T ngl_used ${NGLUSED:-NA} $WARM rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) logits_sha ${H:-NA} want $WANT $V cap0 $C0-$C1 tz0 $T0-$T1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P57_DEF KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
done

cp $BASE/replay4.bak $L/replay.txt
echo "P57_DEF replay_restored $(cat $L/replay.txt) words $(wc -w < $L/replay.txt)"
echo "P57_DEF_DONE skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
