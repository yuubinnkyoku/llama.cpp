#!/system/bin/sh
# Phase 54 - ABBA A/B of the iq3_xxs shared decode kernel, short form.
#
#   sh p54_ab2.sh <pairs> <tag>
#
# Same idea as p54_ab.sh but each run measures only the 128 token prefill plus
# the decode, which halves the wall time per run and therefore buys twice as
# many pairs for the same session. More pairs is what actually beats the run to
# run spread; a longer single run does not.
#
# The device rebooted before this session, so it starts cool and the clock cap
# walks downward instead of sitting at the floor. The script records the cap at
# the start and end of every run so pairs whose cap moved can be dropped in the
# analysis rather than averaged in.
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
PAIRS=${1:-4}
TAG=${2:-ab2}

OLD=$L/libggml-htp-v81.so.p50D.bak2
NEW=$L/libggml-htp-v81.so.newk
OLD_SHA=288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d
NEW_SHA=ab407f84c27a0e062cc14138e6af978cd8b94f52dbab29b7fecc56b0328d36ff

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

[ -f "$OLD" ] || { echo "REFUSING: missing $OLD"; exit 2; }
[ -f "$NEW" ] || { echo "REFUSING: missing $NEW"; exit 2; }

echo "P54_SETTLE start cap0 $(cap) tz0 $(tz0)"

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
    WANT_SKEL=$WANT GUARD_PAGES=524288 RUN_TIMEOUT=1800 \
        P53_PROMPT_TOKENS=128 P53_PROMPT_LIST=128 P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay 16 $T 0 > $BASE/p54-$T.out 2>&1
    RC=$?
    S1=$(date +%s)
    C1=$(cap)
    T1=$(tz0)
    R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    PF=$(grep -oE "PREFILL_SWEEP len 128 ms [0-9.]+" $BASE/p54-$T.out | awk '{print $5}')
    DEC=$(grep -oE "RESULT forwards 4 decode_ms [0-9.]+" $BASE/p54-$T.out | awk '{print $5}')
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    echo "P54_RUN tag $T pair $PAIR order $ORDER arm $ARM skel $(echo $GOT | cut -c1-8) rc $RC exit $EXIT cap0_start $C0 cap0_end $C1 tz0_start $T0 tz0_end $T1 wall_s $((S1 - S0)) prefill128_ms ${PF:-NA} decode_ms ${DEC:-NA} dir $R"
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
echo "P54_DONE skel_restored $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
