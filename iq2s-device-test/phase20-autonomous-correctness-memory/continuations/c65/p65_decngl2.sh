#!/system/bin/sh
# Phase 65 - decode against ngl, using the Phase 64 method.
#
# The first attempt at this (p65_decngl.sh) omitted P53_PROMPT_TOKENS=128, so the
# probe ran on a much shorter prefix and gave 54% average within-ngl spread, far
# too noisy to compare anything. Phase 64 measured 44 against 50 with the 128
# token prefix over 12 runs each and got an IQR of 94 ms, so the prefix is what
# makes the probe usable.
#
# This runs ngl 20 against 44 with that prefix, alternating, and flags repeated
# decode values.
#
#   sh p65_decngl2.sh <tag> [runs per ngl]
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
TAG=${1:-d2}
RUNS=${2:-6}
NGL_LIST=${NGL_LIST:-"20 44"}
SKEL=cdc628a9849b9f211c50bd7f2dca4c54e846b24034afaf189c2a1d2621c8abe2

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

SEEN=""

one() {  # $1 ngl, $2 index
    NGL=$1; IDX=$2
    T="$TAG-n$NGL-$IDX"
    LOG=$BASE/p53-$T.out
    rm -f $LOG
    touch $BASE/.d2-$T
    C0=$(cap); Z0=$(tz0)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=1800 MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
        P53_PROMPT_TOKENS=128 P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay $NGL "$T" 0 > /dev/null 2>&1
    RC=$?
    C1=$(cap); Z1=$(tz0)
    if [ "$RC" != "0" ]; then echo "P65E_END ngl $NGL idx $IDX rc $RC"; return 0; fi
    if [ ! -f $LOG ]; then echo "P65E_END ngl $NGL idx $IDX no log"; return 0; fi
    if [ $BASE/.d2-$T -nt $LOG ]; then echo "P65E_END ngl $NGL idx $IDX stale"; return 0; fi
    R=$(grep -oE "SPLIT_RUN rc [0-9]+ dir .*" $LOG 2>/dev/null | tail -1 | sed 's/.*dir //')
    if [ -z "$R" ] || [ ! -d "$R" ]; then echo "P65E_END ngl $NGL idx $IDX no dir"; return 0; fi
    DEC=$(grep -oE "decode_ms [0-9.]+" $R/stdout.txt 2>/dev/null | tail -1 | sed 's/decode_ms //')
    if [ -z "$DEC" ]; then echo "P65E_END ngl $NGL idx $IDX no decode_ms"; return 0; fi
    SW=$(grep -oE "pswpout [0-9]+" $R/stdout.txt 2>/dev/null | tail -1 | sed 's/pswpout //')
    DUP=""
    case " $SEEN " in *" $DEC "*) DUP=" DUP!" ;; esac
    SEEN="$SEEN $DEC"
    echo "P65E_RUN ngl $NGL idx $IDX decode_ms $DEC cap0 $C0-$C1 tz0 $Z0-$Z1 pswpout ${SW:-NA}$DUP"
}

echo "P65E start tag $TAG runs $RUNS ngl [$NGL_LIST] tokens 128 uptime $(cut -d' ' -f1 /proc/uptime) cap0 $(cap) tz0 $(tz0)"
cp $L/replay32.txt $L/replay.txt

i=1
while [ $i -le $RUNS ]; do
    if [ $((i % 2)) -eq 1 ]; then ORDER=$NGL_LIST; else ORDER=$(echo $NGL_LIST | tr ' ' '\n' | tac | tr '\n' ' '); fi
    for N in $ORDER; do one $N $i; done
    i=$((i + 1))
done

cp $L/replay.txt.std $L/replay.txt
if [ "$(cat $L/replay.txt)" = "25358 11 198 760" ]; then
    echo "P65E replay_restore OK"
else
    echo "P65E REPLAY_RESTORE_FAILED $(cat $L/replay.txt)"
fi
