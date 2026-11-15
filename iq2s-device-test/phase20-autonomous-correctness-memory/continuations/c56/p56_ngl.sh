#!/system/bin/sh
# Phase 56 - measured run per ngl: one 128 token prefill plus a short decode.
#
#   sh p56_ngl.sh "<ngl list>" <tag> [steps]
#
# The clock and the temperature are sampled at both ends of every run, so a run
# whose vendor cap moved is visible and can be dropped instead of averaged.
# PROFILE=1 turns on the DSP profile (GGML_HEXAGON_PROFILE) so the run also
# reports the HTP time, which lets the wall be split into HTP and CPU.
#
# ngl 0 keeps p53_run.sh's own override file (override-base.txt, the CPU
# reference); every other value uses $OV.
#
# The foreground app is never stopped: p53_run.sh filters its list against the
# current topResumedActivity.
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
NGL_LIST=${1:-"16 24 32"}
TAG=${2:-ngl}
STEPS=${3:-4}
TIMEOUT=${TIMEOUT:-400}
OV=${OV:-$L/override-l57.txt}
PROFILE=${PROFILE:-0}
NL=${P53_NO_LOGITS:-1}
MEM_MIN=${MEM_MIN_KB:-4000000}

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }
SKEL=$(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)

echo "P56_SWEEP start skel $SKEL ov $OV ngl $NGL_LIST steps $STEPS timeout $TIMEOUT profile $PROFILE mem_min $MEM_MIN cap0 $(cap) tz0 $(tz0) warm_min ${WARM_RES_MIN_PCT:-90}"

for N in $NGL_LIST; do
    T=$TAG-n$N
    if [ "$N" = "0" ]; then OVN=""; else OVN="OV_FILE=$OV"; fi
    C0=$(cap); T0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=$MEM_MIN $OVN PROFILE_ARG=$PROFILE \
        P53_PROMPT_TOKENS=128 P53_NO_LOGITS=$NL \
        sh $BASE/p53_run.sh $STEPS noreplay $N $T 0 > $BASE/p56-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); T1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p56-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    PF=$(grep -oE "PREFILL seg 0 tokens 128 ms [0-9.]+" $BASE/p56-$T.out | awk '{print $7}')
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p56-$T.out | awk '{print $5}')
    LOAD=$(grep -oE "load_ms [0-9.]+" $R/stdout.txt 2>/dev/null | head -1 | awk '{print $2}')
    OFF=$(grep -oE "offloading [0-9]+ repeating layers to GPU" $R/stderr.txt 2>/dev/null | head -1 | awk '{print $2}')
    HTP=$(awk '/profile-op OPBATCH/{for(i=1;i<=NF;i++) if($i ~ /usec$/){s+=$(i+1); break}} END{printf "%d", s}' $R/stderr.txt 2>/dev/null)
    BATCHES=$(awk '/profile-op OPBATCH/{for(i=1;i<=NF;i++) if($i ~ /usec$/){printf "%s,", $(i+1); break}}' $R/stderr.txt 2>/dev/null)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    echo "P56_RUN tag $T ngl $N skel $(echo $SKEL | cut -c1-8) rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) load_ms ${LOAD:-NA} prefill128_ms ${PF:-NA} decode_ms ${DEC:-NA} htp_usec ${HTP:-NA} cap0 $C0-$C1 tz0 $T0-$T1 off ${OFF:-NA} dir $R"
    echo "P56_BATCH tag $T batches ${BATCHES:-NA}"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P56_KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
done

echo "P56_STRAY $(ps -A -o PID,NAME | awk '/decode-probe/{printf "%s ", $1}')"
echo "P56_DONE skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
