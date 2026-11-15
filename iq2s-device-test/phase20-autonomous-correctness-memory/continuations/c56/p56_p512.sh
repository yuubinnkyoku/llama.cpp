#!/system/bin/sh
# Phase 56 - 512 token prefill at two placements. CPU only (ngl 0) against the
# best measured placement (ngl 44). Same skel, same prompt text, 4 decode steps.
#
#   sh p56_p512.sh <tag>
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
TAG=${1:-p512}
TIMEOUT=${TIMEOUT:-1200}
OV=${OV:-$L/override-l57.txt}
SKEL=$(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

echo "P56_P512 start skel $SKEL ov $OV timeout $TIMEOUT cap0 $(cap) tz0 $(tz0)"
for N in 44 0; do
    T=$TAG-n$N
    if [ "$N" = "0" ]; then OVN=""; else OVN="OV_FILE=$OV"; fi
    C0=$(cap); T0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 $OVN PROFILE_ARG=1 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 P53_PROMPT_TOKENS=512 P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay $N $T 0 > $BASE/p56-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); T1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p56-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    PF=$(grep -oE "PREFILL seg 0 tokens 512 ms [0-9.]+" $BASE/p56-$T.out | awk '{print $7}')
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p56-$T.out | awk '{print $5}')
    HTP=$(awk '/profile-op OPBATCH/{for(i=1;i<=NF;i++) if($i ~ /usec$/){s+=$(i+1); break}} END{printf "%d", s}' $R/stderr.txt 2>/dev/null)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    echo "P56_P512RUN tag $T ngl $N rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) prefill512_ms ${PF:-NA} decode_ms ${DEC:-NA} htp_usec ${HTP:-NA} cap0 $C0-$C1 tz0 $T0-$T1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P56_P512 KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
done
echo "P56_P512_DONE skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
