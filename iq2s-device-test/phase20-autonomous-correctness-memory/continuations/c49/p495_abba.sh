#!/system/bin/sh
# Phase 49.5 / Phase 50 - interleaved comparison series at the cap that is
# currently active.
#
#   sh p495_abba.sh <steps> <ngl>:<tag> ...
#   sh p495_abba.sh 8 0:a1 16:b1 0:a2 16:b2
#
# Every run goes through p495_run.sh, so it stops the background apps, keeps the
# screen awake and records the cap before and after. A run whose cap changed is
# marked CAP_CHANGED and must be dropped: the tier is not the same one.
set -u

BASE=/data/local/tmp/c46-27b
STEPS=${1:-8}
[ $# -gt 0 ] && shift
LOG=$BASE/p495-abba.log
TOK=${REPLAY_MODE:-noreplay}

: > $LOG
log() { echo "$*" >> $LOG; echo "$*"; }
cap0() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
cap7() { cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq 2>/dev/null; }

for spec in "$@"; do
    NGL=${spec%%:*}
    TAG=${spec#*:}
    C0=$(cap0)
    sh $BASE/p495_run.sh "$STEPS" "$TOK" "$NGL" "$TAG" > $BASE/p495-$TAG.log 2>&1
    RC=$?
    C1=$(cap0)
    R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    if [ "$C0" = "$C1" ]; then
        STATE=same
    else
        STATE=CAP_CHANGED
    fi
    log "P495_RUN tag $TAG ngl $NGL steps $STEPS tok $TOK rc $RC exit $(cat $R/exit.txt 2>/dev/null) cap $C0 $(cap7) -> $C1 state $STATE"
    for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
        V=$(grep "DECODE step $s " "$R/stdout.txt" 2>/dev/null | awk '{print $5}')
        [ -n "$V" ] && log "P495_STEP tag $TAG step $s ms $V"
    done
    grep -E "ggml-sched-(split|wait)" "$R/stderr.txt" 2>/dev/null | sed "s/^/P495_SPLIT tag $TAG /" >> $LOG
done
log "P495_DONE"
