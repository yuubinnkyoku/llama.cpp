#!/system/bin/sh
# Phase 55 - ABBA A/B of the combined iq3 kernels (both iq3_xxs and iq3_s).
#
#   sh p55_ab.sh <pairs> <tag>
#
# Old skel 288e6712 (phase 50 D) vs new skel e311bf14 (iq3_xxs shared decode and
# iq3_s shared gathers). Same shape as the phase 54 short form: one 128 token
# prefill plus a short decode, ABBA order, cap recorded at both ends of every
# run so a pair whose clock moved can be dropped instead of averaged.
#
# The foreground app is never stopped: the list is filtered against the current
# topResumedActivity first.
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
PAIRS=${1:-6}
TAG=${2:-ab}

OLD=$L/libggml-htp-v81.so.p50D.bak2
NEW=$L/libggml-htp-v81.so.expF
OLD_SHA=288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d
NEW_SHA=e311bf148b58444ea42a45626663bfb43d749d44128d5026547e37f60595f1e7

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

[ -f "$OLD" ] || { echo "REFUSING: missing $OLD"; exit 2; }
[ -f "$NEW" ] || { echo "REFUSING: missing $NEW"; exit 2; }

echo "P55_SETTLE start cap0 $(cap) tz0 $(tz0)"

run_arm() {
    ARM=$1
    PAIR=$2
    ORDER=$3
    if [ "$ARM" = "old" ]; then
        cp "$OLD" $L/libggml-htp-v81.so
        WANT=$OLD_SHA
    else
        cp "$NEW" $L/libggml-htp-v81.so
        WANT=$NEW_SHA
    fi
    chmod 644 $L/libggml-htp-v81.so
    GOT=$(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)
    [ "$GOT" = "$WANT" ] || { echo "REFUSING: skel $GOT != $WANT"; return 2; }

    T=$TAG-p$PAIR-$ORDER-$ARM
    C0=$(cap)
    T0=$(tz0)
    S0=$(date +%s)
    WANT_SKEL=$WANT GUARD_PAGES=2097152 RUN_TIMEOUT=1800 \
        P53_PROMPT_TOKENS=128 P53_PROMPT_LIST=128 P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay 16 $T 0 > $BASE/p55-$T.out 2>&1
    RC=$?
    S1=$(date +%s)
    C1=$(cap)
    T1=$(tz0)
    R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    PF=$(grep -oE "PREFILL_SWEEP len 128 ms [0-9.]+" $BASE/p55-$T.out | awk '{print $5}')
    DEC=$(grep -oE "RESULT forwards 4 decode_ms [0-9.]+" $BASE/p55-$T.out | awk '{print $5}')
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    echo "P55_RUN tag $T pair $PAIR order $ORDER arm $ARM skel $(echo $GOT | cut -c1-8) rc $RC exit $EXIT cap0_start $C0 cap0_end $C1 tz0_start $T0 tz0_end $T1 wall_s $((S1 - S0)) prefill128_ms ${PF:-NA} decode_ms ${DEC:-NA} dir $R"
}

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

cp "$OLD" $L/libggml-htp-v81.so
chmod 644 $L/libggml-htp-v81.so
echo "P55_DONE skel_restored $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
