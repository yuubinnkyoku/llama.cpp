#!/system/bin/sh
# Phase 65 target 2 - NGL sweep after dotprod made the CPU floor about 2.3x faster.
#
# Phase 61 concluded 44 was best when a CPU layer cost what it did then. The CPU
# floor moved, so the crossover between CPU-resident and HTP-resident layers may
# have moved with it.
#
# Fixes the two failures from the batch ABBA (report 4.1 and 4.6):
#   - the run log is deleted first, so a stale file cannot be read
#   - the log must exist and be newer than a marker written just before the run,
#     otherwise the run is rejected instead of reporting an old number
#   - a prefill value that was already seen is flagged as a duplicate
#
#   sh p65_ngl.sh <tag> <rounds>
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
TAG=${1:-ngl}
ROUNDS=${2:-2}
NGL_LIST=${NGL_LIST:-"32 36 40 44 48"}
TIMEOUT=${TIMEOUT:-1800}

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

SEEN=""

pf() {  # $1 tag, $2 ngl, $3 tokens
    T=$1; NGL=$2; NT=$3
    LOG=$BASE/p53-$T.out
    rm -f $LOG
    touch $BASE/.mark-$T
    C0=$(cap); Z0=$(tz0)
    env WANT_SKEL=$(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1) \
        GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
        P53_PROMPT_TOKENS=$NT P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay $NGL $T 0 > /dev/null 2>&1
    RC=$?
    C1=$(cap); Z1=$(tz0)
    if [ "$RC" != "0" ]; then echo "P65N_ABORT $T rc $RC"; return 3; fi
    # the log must be newer than the marker, else we would read a previous run
    [ -f $LOG ] || { echo "P65N_ABORT $T no log"; return 3; }
    if [ $BASE/.mark-$T -nt $LOG ]; then echo "P65N_ABORT $T log older than marker"; return 3; fi
    R=$(grep -oE "SPLIT_RUN rc [0-9]+ dir .*" $LOG 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || { echo "P65N_ABORT $T no run dir in log"; return 3; }
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    PFL=$(grep -oE "PREFILL seg 0 tokens [0-9]+ ms [0-9.]+" $R/stdout.txt 2>/dev/null | head -1 | awk '{print $7}')
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $R/stdout.txt 2>/dev/null | head -1 | awk '{print $5}')
    [ -n "$PFL" ] || { echo "P65N_ABORT $T no prefill in $R"; return 3; }
    DUP=""
    case " $SEEN " in *" $PFL "*) DUP=" DUP!" ;; esac
    SEEN="$SEEN $PFL"
    echo "P65N_RUN $T ngl $NGL nt $NT exit ${EXIT:-NA} prefill_ms $PFL decode_ms ${DEC:-NA} cap0 $C0-$C1 tz0 $Z0-$Z1$DUP"
}

echo "P65N start tag $TAG rounds $ROUNDS uptime $(cut -d' ' -f1 /proc/uptime) cap0 $(cap) tz0 $(tz0)"
cp $L/replay32.txt $L/replay.txt

i=1
while [ $i -le $ROUNDS ]; do
    if [ $((i % 2)) -eq 1 ]; then ORDER=$NGL_LIST; else ORDER=$(echo $NGL_LIST | tr ' ' '\n' | tac | tr '\n' ' '); fi
    for N in $ORDER; do
        pf "$TAG-r$i-n$N" $N 128
        [ $? -eq 0 ] || exit 3
    done
    i=$((i + 1))
done

cp $BASE/libs/replay.txt.std $L/replay.txt
echo "P65N_DONE cap $(cap) tz $(tz0)"
