#!/system/bin/sh
# Phase 66 target 1 - decode against ngl, three arms, twelve runs each.
#
# This is the measurement Phase 65 left open: ngl 0 beat 44 by 1.47x by median
# but Welch t was 1.51 at n=6, so it was not established.
#
# Rules, all from p66_common.sh:
#   - run dir from the probe's SPLIT_RUN line, never a newest-directory fallback
#   - cap before and after every run, reported so only matched caps are compared
#   - exact repeats flagged, since a repeat means a stale read
#   - replay restored and compared against the expected value at the end
#   - any run failure aborts instead of reporting the previous value
#
# Decode uses P53_PROMPT_TOKENS=128. Four forwards alone gave 54% spread in
# Phase 65 and is unusable for comparison.
#
#   sh p66_decngl.sh <tag> [runs per arm]
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
. $BASE/p66_common.sh

TAG=${1:-d}
RUNS=${2:-12}
NGL_LIST=${NGL_LIST:-"0 20 44"}
SKEL=cdc628a9849b9f211c50bd7f2dca4c54e846b24034afaf189c2a1d2621c8abe2

one() {  # $1 ngl, $2 index
    NGL=$1; IDX=$2
    T="$TAG-n$NGL-$IDX"
    LOG=$BASE/p53-$T.out
    rm -f $LOG
    touch $BASE/.p66-$T
    C0=$(p66_cap); Z0=$(p66_tz)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=1800 MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
        P53_PROMPT_TOKENS=128 P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay $NGL "$T" 0 > /dev/null 2>&1
    RC=$?
    C1=$(p66_cap); Z1=$(p66_tz)
    if [ "$RC" != "0" ]; then echo "P66D_ABORT ngl $NGL idx $IDX rc $RC"; return 1; fi
    [ -f $LOG ] || { echo "P66D_ABORT ngl $NGL idx $IDX no log"; return 1; }
    if [ $BASE/.p66-$T -nt $LOG ]; then echo "P66D_ABORT ngl $NGL idx $IDX stale log"; return 1; fi
    R=$(p66_rundir $LOG)
    [ -n "$R" ] && [ -d "$R" ] || { echo "P66D_ABORT ngl $NGL idx $IDX no run dir"; return 1; }
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $R/stdout.txt 2>/dev/null | head -1 | awk '{print $5}')
    [ -n "$DEC" ] || { echo "P66D_ABORT ngl $NGL idx $IDX no decode_ms"; return 1; }
    D=""
    p66_dup "$DEC" >/dev/null && D=" DUP!"
    echo "P66D_RUN ngl $NGL idx $IDX decode_ms $DEC cap0 $C0-$C1 tz0 $Z0-$Z1$D"
    return 0
}

echo "P66D start tag $TAG runs $RUNS ngl [$NGL_LIST] tokens 128 uptime $(cut -d' ' -f1 /proc/uptime) cap0 $(p66_cap) tz0 $(p66_tz)"
p66_replay_set replay32.txt || exit 2

i=1
while [ $i -le $RUNS ]; do
    if [ $((i % 2)) -eq 1 ]; then ORDER=$NGL_LIST; else ORDER=$(echo $NGL_LIST | tr ' ' '\n' | tac | tr '\n' ' '); fi
    for N in $ORDER; do
        one $N $i || exit 1
    done
    i=$((i + 1))
done

p66_replay_restore || exit 1
echo "P66D_DONE cap $(p66_cap) tz $(p66_tz)"
