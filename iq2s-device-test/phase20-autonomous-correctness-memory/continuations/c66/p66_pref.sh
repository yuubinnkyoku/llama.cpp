#!/system/bin/sh
# Phase 66 - prefill at ngl 44 and ngl 0, for the preset table.
#
# Uses p66_common.sh for the rules: run dir from SPLIT_RUN only, cap recorded
# around each run, repeats flagged, replay restored and compared.
#
#   sh p66_pref.sh <tag> [rounds]
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
. $BASE/p66_common.sh

TAG=${1:-p}
ROUNDS=${2:-2}
NGL_LIST=${NGL_LIST:-"44 0"}
SKEL=cdc628a9849b9f211c50bd7f2dca4c54e846b24034afaf189c2a1d2621c8abe2

one() {  # $1 ngl, $2 index
    NGL=$1; IDX=$2
    T="$TAG-n$NGL-$IDX"
    LOG=$BASE/p53-$T.out
    rm -f $LOG
    touch $BASE/.p66p-$T
    C0=$(p66_cap); Z0=$(p66_tz)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=1800 MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
        P53_PROMPT_TOKENS=128 P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay $NGL "$T" 0 > /dev/null 2>&1
    RC=$?
    C1=$(p66_cap); Z1=$(p66_tz)
    if [ "$RC" != "0" ]; then echo "P66P_ABORT ngl $NGL idx $IDX rc $RC"; return 1; fi
    [ -f $LOG ] || { echo "P66P_ABORT ngl $NGL idx $IDX no log"; return 1; }
    if [ $BASE/.p66p-$T -nt $LOG ]; then echo "P66P_ABORT ngl $NGL idx $IDX stale log"; return 1; fi
    R=$(p66_rundir $LOG)
    [ -n "$R" ] && [ -d "$R" ] || { echo "P66P_ABORT ngl $NGL idx $IDX no run dir"; return 1; }
    PF=$(grep -oE "PREFILL seg 0 tokens 128 ms [0-9.]+" $R/stdout.txt 2>/dev/null | head -1 | awk '{print $7}')
    [ -n "$PF" ] || { echo "P66P_ABORT ngl $NGL idx $IDX no prefill"; return 1; }
    D=""
    p66_dup "$PF" >/dev/null && D=" DUP!"
    echo "P66P_RUN ngl $NGL idx $IDX prefill_ms $PF cap0 $C0-$C1 tz0 $Z0-$Z1$D"
    return 0
}

echo "P66P start tag $TAG rounds $ROUNDS ngl [$NGL_LIST] tokens 128 cap0 $(p66_cap) tz0 $(p66_tz)"
p66_replay_set replay32.txt || exit 2
i=1
while [ $i -le $ROUNDS ]; do
    if [ $((i % 2)) -eq 1 ]; then ORDER=$NGL_LIST; else ORDER=$(echo $NGL_LIST | tr ' ' '\n' | tac | tr '\n' ' '); fi
    for N in $ORDER; do one $N $i || exit 1; done
    i=$((i + 1))
done
p66_replay_restore || exit 1
echo "P66P_DONE cap $(p66_cap) tz $(p66_tz)"
