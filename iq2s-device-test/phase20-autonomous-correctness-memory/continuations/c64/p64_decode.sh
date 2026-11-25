#!/system/bin/sh
# Phase 64 - decode spread, 12 runs, NGL 44 against NGL 50, interleaved.
#
# Phase 61 did 6 runs each and got medians 886.5 and 838.6 ms, a 5.7% gap with
# an IQR near 100 ms on both. That is too few to separate a real effect from
# noise, so this doubles it to 12 and interleaves to cancel thermal drift.
#
# Tiled is off by default in run53.sh now, which is right here: decode is one
# token per step and never takes the tiled path either way.
#
#   sh p64_decode.sh <tag> <runs_per_ngl>
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
TAG=${1:-dec}
RUNS=${2:-12}
TIMEOUT=${TIMEOUT:-1800}

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

one() {
    T=$1; NGL=$2
    C0=$(cap); Z0=$(tz0)
    env GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
        P53_PROMPT_TOKENS=128 P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay $NGL $T 0 > $BASE/p64d-$T.out 2>&1
    RC=$?
    C1=$(cap); Z1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p64d-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p64d-$T.out | awk '{print $5}')
    SW=$(grep -oE "delta_pages [0-9-]+" $BASE/p64d-$T.out | awk '{print $2}')
    echo "P64D_RUN tag $T ngl $NGL rc $RC exit ${EXIT:-NA} decode_ms ${DEC:-NA} swap_delta_pages ${SW:-NA} cap0 $C0-$C1 tz0 $Z0-$Z1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

echo "P64D start tag $TAG runs $RUNS uptime $(cut -d' ' -f1 /proc/uptime) cap0 $(cap) tz0 $(tz0)"
cp $L/replay32.txt $L/replay.txt

i=1
while [ $i -le $RUNS ]; do
    if [ $((i % 2)) -eq 1 ]; then
        one $TAG-$i-44 44
        one $TAG-$i-50 50
    else
        one $TAG-$i-50 50
        one $TAG-$i-44 44
    fi
    i=$((i + 1))
done

cp $BASE/replay4.bak $L/replay.txt
echo "P64D replay_restored $(cat $L/replay.txt)"
echo "P64D stray $(ps -A -o PID,NAME | awk '/decode-probe/{printf "%s ", $1}')"
echo "P64D_DONE cpu $(sha256sum $L/libggml-cpu.so | cut -d' ' -f1) skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
