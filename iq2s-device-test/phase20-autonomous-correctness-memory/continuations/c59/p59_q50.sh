#!/system/bin/sh
# Phase 59 - goal 2: quality at NGL=50 with the c58x4 skeleton.
#
#   sh p59_q50.sh <tag> [runs]
#
# 32 step replay with logits, NGL=50, c58x4. The CPU reference (ngl 0) is the
# known hash 44cd99ce from phase 56/57, so it is not re-measured here; the
# comparison is done on the host with p50_nll.py against the stored reference.
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
TAG=${1:-q50}
RUNS=${2:-5}
TIMEOUT=${TIMEOUT:-1800}
SKEL=76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef
THREADS=${THREADS:-8}

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

ACT=$(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)
[ "$ACT" = "$SKEL" ] || { echo "P59_Q50 REFUSING skel $ACT"; exit 2; }

cp $L/replay32.txt $L/replay.txt || exit 2
echo "P59_Q50 start skel $ACT threads $THREADS runs $RUNS cap0 $(cap) tz0 $(tz0)"

I=0
while [ $I -lt $RUNS ]; do
    I=$((I + 1))
    T=$TAG-r$I
    C0=$(cap); Z0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 THREADS=$THREADS \
        sh $BASE/p53_run.sh 32 replay ${NGL:-50} $T 0 > $BASE/p59-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); Z1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p59-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    H=$(sha256sum $R/logits.f32 2>/dev/null | cut -d' ' -f1)
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p59-$T.out | awk '{print $5}')
    SW=$(grep -oE "delta_pages [0-9-]+" $BASE/p59-$T.out | awk '{print $2}')
    echo "P59_Q50RUN tag $T rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) decode_ms ${DEC:-NA} logits_sha ${H:-NA} swap_delta_pages ${SW:-NA} cap0 $C0-$C1 tz0 $Z0-$Z1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P59_Q50 KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
done

cp $BASE/replay4.bak $L/replay.txt
echo "P59_Q50 replay_restored $(cat $L/replay.txt) words $(wc -w < $L/replay.txt)"
echo "P59_Q50_DONE skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
