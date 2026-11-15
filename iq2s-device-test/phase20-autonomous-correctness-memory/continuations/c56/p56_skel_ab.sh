#!/system/bin/sh
# Phase 56 - skel A/B at a high ngl, where the HTP owns about half of the
# prefill wall instead of a sixth. Old skel 288e6712 (phase 50 D) against the
# phase 55 kernels e311bf14 (iq3_xxs shared decode + iq3_s shared gathers).
#
#   sh p56_skel_ab.sh <pairs> <tag> [ngl]
#
# ABBA order, cap and temperature recorded at both ends of every run, sha256 of
# the skel checked after every copy. The foreground app is never stopped.
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
PAIRS=${1:-2}
TAG=${2:-ab}
NGL=${3:-44}
STEPS=${4:-4}
TIMEOUT=${TIMEOUT:-600}
OV=${OV:-$L/override-l57.txt}

OLD=$L/libggml-htp-v81.so.p50D.bak2
NEW=$L/libggml-htp-v81.so.expF
OLD_SHA=288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d
NEW_SHA=e311bf148b58444ea42a45626663bfb43d749d44128d5026547e37f60595f1e7

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

[ -f "$OLD" ] || { echo "REFUSING: missing $OLD"; exit 2; }
[ -f "$NEW" ] || { echo "REFUSING: missing $NEW"; exit 2; }

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
    [ "$GOT" = "$WANT" ] || { echo "P56_AB REFUSING skel $GOT != $WANT"; return 2; }

    T=$TAG-p$PAIR-$ORDER-$ARM
    C0=$(cap); T0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$WANT GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 OV_FILE=$OV \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 PROFILE_ARG=1 \
        P53_PROMPT_TOKENS=128 P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh $STEPS noreplay $NGL $T 0 > $BASE/p56-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); T1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p56-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    PF=$(grep -oE "PREFILL seg 0 tokens 128 ms [0-9.]+" $BASE/p56-$T.out | awk '{print $7}')
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p56-$T.out | awk '{print $5}')
    HTP=$(awk '/profile-op OPBATCH/{for(i=1;i<=NF;i++) if($i ~ /usec$/){s+=$(i+1); break}} END{printf "%d", s}' $R/stderr.txt 2>/dev/null)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    echo "P56_AB tag $T pair $PAIR order $ORDER arm $ARM skel $(echo $GOT | cut -c1-8) rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) prefill128_ms ${PF:-NA} decode_ms ${DEC:-NA} htp_usec ${HTP:-NA} cap0 $C0-$C1 tz0 $T0-$T1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P56_AB KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
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
echo "P56_AB_DONE skel_restored $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
