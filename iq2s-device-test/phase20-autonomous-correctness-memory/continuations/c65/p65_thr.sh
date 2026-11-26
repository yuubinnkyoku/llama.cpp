#!/system/bin/sh
# Phase 65 target 2.4 - thread count spot check after dotprod and tiled changed.
#
# The thread count was unreachable from the measured path: run53.sh takes it as
# the fifth positional argument with no env override, and p53_run.sh passes only
# two arguments through run_split53.sh. run53.sh now reads
# NTHREADS=${NTHREADS:-${5:-6}}, and the default was verified to still produce
# 8259e4b0 before this sweep, so the measurement base is unchanged.
#
# Carries the safety rules from p65_abba2.sh: log deleted first and checked
# against a marker, duplicate prefill flagged, any failure aborts.
#
#   sh p65_thr.sh <tag> <rounds>
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
TAG=${1:-thr}
ROUNDS=${2:-2}
THR_LIST=${THR_LIST:-"4 6 8"}
TOKENS=${TOKENS:-128}
TIMEOUT=${TIMEOUT:-1800}

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

SEEN=""

one() {  # $1 tag, $2 threads
    T=$1; NTH=$2
    LOG=$BASE/p53-$T.out
    rm -f $LOG
    touch $BASE/.t-$T
    C0=$(cap); Z0=$(tz0)
    env NTHREADS=$NTH WANT_SKEL=$(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1) \
        GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
        P53_PROMPT_TOKENS=$TOKENS P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay 44 $T 0 > /dev/null 2>&1
    RC=$?
    C1=$(cap); Z1=$(tz0)
    if [ "$RC" != "0" ]; then echo "P65T_ABORT $T rc $RC"; return 3; fi
    [ -f $LOG ] || { echo "P65T_ABORT $T no log"; return 3; }
    if [ $BASE/.t-$T -nt $LOG ]; then echo "P65T_ABORT $T log older than marker"; return 3; fi
    R=$(grep -oE "SPLIT_RUN rc [0-9]+ dir .*" $LOG 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || { echo "P65T_ABORT $T no run dir"; return 3; }
    PFL=$(grep -oE "PREFILL seg 0 tokens [0-9]+ ms [0-9.]+" $R/stdout.txt 2>/dev/null | head -1 | awk '{print $7}')
    [ -n "$PFL" ] || { echo "P65T_ABORT $T no prefill"; return 3; }
    GOTTHR=$(grep -oE "threads [0-9]+" $LOG | head -1 | awk '{print $2}')
    DUP=""
    case " $SEEN " in *" $PFL "*) DUP=" DUP!" ;; esac
    SEEN="$SEEN $PFL"
    echo "P65T_RUN $T thr $NTH applied ${GOTTHR:-NA} prefill_ms $PFL cap0 $C0-$C1 tz0 $Z0-$Z1$DUP"
}

echo "P65T start tag $TAG rounds $ROUNDS threads [$THR_LIST] tokens $TOKENS uptime $(cut -d' ' -f1 /proc/uptime) cap0 $(cap) tz0 $(tz0)"
cp $L/replay32.txt $L/replay.txt

i=1
while [ $i -le $ROUNDS ]; do
    if [ $((i % 2)) -eq 1 ]; then ORDER=$THR_LIST; else ORDER=$(echo $THR_LIST | tr ' ' '\n' | tac | tr '\n' ' '); fi
    for N in $ORDER; do
        one "$TAG-r$i-t$N" $N
        [ $? -eq 0 ] || exit 3
    done
    i=$((i + 1))
done

cp $BASE/replay4.bak $L/replay.txt
echo "P65T_DONE cap $(cap) tz $(tz0)"
