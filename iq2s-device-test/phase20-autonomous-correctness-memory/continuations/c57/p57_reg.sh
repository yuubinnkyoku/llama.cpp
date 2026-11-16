#!/system/bin/sh
# Phase 57 - goal 0 regression gate: the two known 32 step hashes.
#
#   sh p57_reg.sh <tag>
#
# NGL=16 must give b21ca4bf, NGL=0 must give 44cd99ce. The replay file holds 32
# tokens while this runs and is restored to the standard 4 at the end. NGL=0
# keeps p53_run.sh's own override file (override-base.txt).
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
TAG=${1:-reg}
TIMEOUT=${TIMEOUT:-1800}
SKEL=288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d
WANT16=b21ca4bf1da8a2ef3bc48f717ee79f13acc17ced0783141211fc7178f0c2e160
WANT0=44cd99ce376d5ce627ebbd4dc1c1d6cb90e83ea7822371f13a4874f921262295

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

ACT=$(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)
[ "$ACT" = "$SKEL" ] || { echo "P57_REG REFUSING skel $ACT"; exit 2; }

cp $L/replay32.txt $L/replay.txt || exit 2
echo "P57_REG start skel $ACT replay $(wc -w < $L/replay.txt) cap0 $(cap) tz0 $(tz0)"

for N in 16 0; do
    T=$TAG-n$N
    if [ "$N" = "0" ]; then OVN=""; else OVN="OV_FILE=$L/override-l57.txt"; fi
    C0=$(cap); T0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 $OVN \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
        sh $BASE/p53_run.sh 32 replay $N $T 0 > $BASE/p57-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); T1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p57-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    H=$(sha256sum $R/logits.f32 2>/dev/null | cut -d' ' -f1)
    if [ "$N" = "16" ]; then WANT=$WANT16; else WANT=$WANT0; fi
    if [ "$H" = "$WANT" ]; then V=MATCH; else V=MISMATCH; fi
    echo "P57_REGRUN tag $T ngl $N rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) logits_sha ${H:-NA} want $WANT $V cap0 $C0-$C1 tz0 $T0-$T1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P57_REG KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
done

cp $BASE/replay4.bak $L/replay.txt
echo "P57_REG replay_restored $(cat $L/replay.txt) words $(wc -w < $L/replay.txt)"
echo "P57_REG_DONE skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
