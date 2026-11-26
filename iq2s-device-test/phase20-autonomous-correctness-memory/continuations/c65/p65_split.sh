#!/system/bin/sh
# Phase 65 - where does decode time go, and does raising ngl move it?
#
# The split timing lines are written to stderr.txt, not stdout.txt, which is why
# the first attempt saw none. For each ngl this runs the decode probe and sums
# compute_us per backend.
#
# Caveat carried from the report: the summed compute exceeds wall time because
# warmup splits are included, so only the CPU:HTP ratio is meaningful, never the
# absolute totals.
#
#   sh p65_split.sh <tag> [ngl list]
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
TAG=${1:-split}
NGL_LIST=${2:-"44 20"}
SKEL=cdc628a9849b9f211c50bd7f2dca4c54e846b24034afaf189c2a1d2621c8abe2

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

one() {  # $1 ngl
    NGL=$1
    T="$TAG-n$NGL"
    LOG=$BASE/p53-$T.out
    rm -f $LOG
    touch $BASE/.s-$T
    C0=$(cap); Z0=$(tz0)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=1800 MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
        sh $BASE/p53_run.sh 4 noreplay $NGL "$T" 0 > /dev/null 2>&1
    RC=$?
    C1=$(cap); Z1=$(tz0)
    if [ "$RC" != "0" ]; then echo "P65S_END ngl $NGL rc $RC"; return 0; fi
    if [ ! -f $LOG ]; then echo "P65S_END ngl $NGL no log"; return 0; fi
    if [ $BASE/.s-$T -nt $LOG ]; then echo "P65S_END ngl $NGL stale"; return 0; fi
    R=$(grep -oE "SPLIT_RUN rc [0-9]+ dir .*" $LOG 2>/dev/null | tail -1 | sed 's/.*dir //')
    if [ -z "$R" ] || [ ! -d "$R" ]; then echo "P65S_END ngl $NGL no dir"; return 0; fi
    DEC=$(grep -oE "decode_ms [0-9.]+" $R/stdout.txt 2>/dev/null | tail -1 | sed 's/decode_ms //')
    NS=$(grep -c "ggml-sched-split" $R/stderr.txt 2>/dev/null)
    SUM=$(grep "ggml-sched-split" $R/stderr.txt 2>/dev/null | awk '
        { b=""; n=0; c=0;
          for (i=1;i<=NF;i++) { if ($i=="backend") b=$(i+1); if ($i=="nodes") n=$(i+1); if ($i=="compute_us") c=$(i+1) }
          if (b=="CPU") { cpuc+=c; cpun+=n; cpuk++ } else { htpc+=c; htpn+=n; htpk++ } }
        END { printf "cpu_splits %d cpu_nodes %d cpu_us %d htp_splits %d htp_nodes %d htp_us %d cpu_share_pct %.1f",
                      cpuk, cpun, cpuc, htpk, htpn, htpc, (cpuc+htpc)>0 ? 100*cpuc/(cpuc+htpc) : -1 }')
    echo "P65S_END ngl $NGL decode_ms ${DEC:-NA} nsplits ${NS:-0} $SUM cap0 $C0-$C1 tz0 $Z0-$Z1"
}

echo "P65S start tag $TAG ngl [$NGL_LIST] uptime $(cut -d' ' -f1 /proc/uptime) cap0 $(cap) tz0 $(tz0)"
cp $L/replay32.txt $L/replay.txt
for N in $NGL_LIST; do one $N; done
cp $L/replay.txt.std $L/replay.txt
if [ "$(cat $L/replay.txt)" = "25358 11 198 760" ]; then
    echo "P65S replay_restore OK"
else
    echo "P65S REPLAY_RESTORE_FAILED $(cat $L/replay.txt)"
fi
