#!/system/bin/sh
# Phase 65 - decode against ngl. Phase 64 only compared 44 with 50 and found no
# difference, but 20 was never tried, and the first look shows 44 slower than 20.
#
# Phase 64's decode runs used a 128 token prefix; these use the plain probe
# sequence so the numbers here are for comparing ngl against each other, not for
# comparing against the 926 ms figure.
#
# cap is recorded around every run and reported, and a repeated decode_ms is
# flagged, because the duplicate read problem bit twice already.
#
#   sh p65_decngl.sh <tag> [rounds]
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
TAG=${1:-dn}
ROUNDS=${2:-2}
NGL_LIST=${NGL_LIST:-"12 20 28 36 44"}
SKEL=cdc628a9849b9f211c50bd7f2dca4c54e846b24034afaf189c2a1d2621c8abe2

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

SEEN=""

one() {  # $1 ngl
    NGL=$1
    T="$TAG-n$NGL"
    LOG=$BASE/p53-$T.out
    rm -f $LOG
    touch $BASE/.d-$T
    C0=$(cap); Z0=$(tz0)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=1800 MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
        sh $BASE/p53_run.sh 4 noreplay $NGL "$T" 0 > /dev/null 2>&1
    RC=$?
    C1=$(cap); Z1=$(tz0)
    if [ "$RC" != "0" ]; then echo "P65N2_END ngl $NGL rc $RC"; return 0; fi
    if [ ! -f $LOG ]; then echo "P65N2_END ngl $NGL no log"; return 0; fi
    if [ $BASE/.d-$T -nt $LOG ]; then echo "P65N2_END ngl $NGL stale"; return 0; fi
    R=$(grep -oE "SPLIT_RUN rc [0-9]+ dir .*" $LOG 2>/dev/null | tail -1 | sed 's/.*dir //')
    if [ -z "$R" ] || [ ! -d "$R" ]; then echo "P65N2_END ngl $NGL no dir"; return 0; fi
    DEC=$(grep -oE "decode_ms [0-9.]+" $R/stdout.txt 2>/dev/null | tail -1 | sed 's/decode_ms //')
    if [ -z "$DEC" ]; then echo "P65N2_END ngl $NGL no decode_ms"; return 0; fi
    NS=$(grep -c "ggml-sched-split" $R/stderr.txt 2>/dev/null)
    DUP=""
    case " $SEEN " in *" $DEC "*) DUP=" DUP!" ;; esac
    SEEN="$SEEN $DEC"
    echo "P65N2_RUN ngl $NGL decode_ms $DEC splits ${NS:-0} cap0 $C0-$C1 tz0 $Z0-$Z1$DUP"
}

echo "P65N2 start tag $TAG rounds $ROUNDS ngl [$NGL_LIST] uptime $(cut -d' ' -f1 /proc/uptime) cap0 $(cap) tz0 $(tz0)"
cp $L/replay32.txt $L/replay.txt

i=1
while [ $i -le $ROUNDS ]; do
    if [ $((i % 2)) -eq 1 ]; then ORDER=$NGL_LIST; else ORDER=$(echo $NGL_LIST | tr ' ' '\n' | tac | tr '\n' ' '); fi
    for N in $ORDER; do one $N; done
    i=$((i + 1))
done

cp $L/replay.txt.std $L/replay.txt
if [ "$(cat $L/replay.txt)" = "25358 11 198 760" ]; then
    echo "P65N2 replay_restore OK"
else
    echo "P65N2 REPLAY_RESTORE_FAILED $(cat $L/replay.txt)"
fi
