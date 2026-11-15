#!/system/bin/sh
# Phase 56 - fast load scan over ngl values.
#
#   sh p56_load.sh "<ngl list>" <tag>
#
# One 1-token prompt and one decode step per ngl: the point is only whether the
# model load completes (the FastRPC stall shows up during load) and how long it
# takes. Wall measurements use p56_ngl.sh with the 128 token prompt.
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
NGL_LIST=${1:-"16 20 24 28 32 40 48 56 64"}
TAG=${2:-load}
TIMEOUT=${TIMEOUT:-180}
OV=${OV:-$L/override-l57.txt}
SKEL=$(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

echo "P56_LOAD start skel $SKEL ov $OV ngl $NGL_LIST timeout $TIMEOUT cap0 $(cap) tz0 $(tz0)"
for N in $NGL_LIST; do
    T=$TAG-n$N
    C0=$(cap); T0=$(tz0); S0=$(date +%s)
    WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT OV_FILE=$OV \
        P53_PROMPT_TOKENS=1 P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 1 noreplay $N $T 0 > $BASE/p56-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); T1=$(tz0)
    R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    LOAD=$(grep -oE "load_ms [0-9.]+" $R/stdout.txt 2>/dev/null | head -1 | awk '{print $2}')
    OFF=$(grep -oE "offloading [0-9]+ repeating layers to GPU" $R/stderr.txt 2>/dev/null | head -1 | awk '{print $2}')
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    ERR=$(grep -oE "ggml-hex: (error|failed)[^\"]*" $R/stderr.txt 2>/dev/null | head -1)
    echo "P56_LOADRUN tag $T ngl $N rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) load_ms ${LOAD:-NA} off ${OFF:-NA} cap0 $C0-$C1 tz0 $T0-$T1 err ${ERR:-none} dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P56_KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
done
echo "P56_STRAY $(ps -A -o PID,NAME | awk '/decode-probe/{printf "%s ", $1}')"
echo "P56_LOAD_DONE skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
