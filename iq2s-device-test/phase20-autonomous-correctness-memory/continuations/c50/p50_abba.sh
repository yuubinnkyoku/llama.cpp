#!/system/bin/sh
# Phase 50 - interleaved comparison series at the cap that is currently active.
#
#   sh p50_abba.sh <steps> <ngl>:<tag> ...
#   sh p50_abba.sh 8 0:a1 16:b1 0:a2 16:b2
#
# Every run goes through p50_run.sh, so it stops the background apps, holds the
# screen awake, and records the cap before, during and after. A run whose cap
# changed is marked CAP_CHANGED and must be dropped: the tier is not the same
# one. Medians over step 1..n and over step 0..n are printed per tag.
set -u

BASE=/data/local/tmp/c46-27b
STEPS=${1:-8}
[ $# -gt 0 ] && shift
LOG=$BASE/p50-abba.log
TOK=${REPLAY_MODE:-noreplay}

: > $LOG
log() { echo "$*" >> $LOG; echo "$*"; }
cap0() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
cap7() { cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq 2>/dev/null; }

median() { awk '{print $1}' | sort -n | awk '{a[NR]=$1} END{if(NR==0){print "-"}else if(NR%2){print a[(NR+1)/2]}else{printf "%.2f", (a[NR/2]+a[NR/2+1])/2}}'; }

tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

# P50_COOL seconds of idle before every run. The vendor cap latches downward
# when the SoC heats, so a cold start is the only way to measure in the fast
# band. 0 keeps the back to back behaviour of Phase 49.5.
COOL=${P50_COOL:-0}

LAST=""
for spec in "$@"; do
    NGL=${spec%%:*}
    TAG=${spec#*:}
    if [ "$COOL" -gt 0 ]; then
        log "P50_COOL tag $TAG start_s $COOL cap0 $(cap0) cap7 $(cap7) tz0 $(tz0)"
        sleep "$COOL"
        log "P50_COOL tag $TAG woke cap0 $(cap0) cap7 $(cap7) tz0 $(tz0)"
    fi
    C0=$(cap0)
    sh $BASE/p50_run.sh "$STEPS" "$TOK" "$NGL" "$TAG" > $BASE/p50-$TAG.log 2>&1
    RC=$?
    C1=$(cap0)
    STABLE=$(grep "^P50_CAP_STABLE" $BASE/p50-$TAG.log 2>/dev/null | awk '{print $2}')
    R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    if [ "$C0" = "$C1" ] && [ "$STABLE" = "YES" ]; then
        STATE=same
    else
        STATE=CAP_CHANGED
    fi
    log "P50_RUN tag $TAG ngl $NGL steps $STEPS tok $TOK rc $RC exit $(cat $R/exit.txt 2>/dev/null) cap $C0 $(cap7) -> $C1 mid_stable $STABLE state $STATE"
    : > $BASE/p50-$TAG.steps
    for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
        V=$(grep "DECODE step $s " "$R/stdout.txt" 2>/dev/null | awk '{print $5}')
        if [ -n "$V" ]; then
            echo "$s $V" >> $BASE/p50-$TAG.steps
            log "P50_STEP tag $TAG step $s ms $V"
        fi
    done
    ALLM=$(awk '{print $2}' $BASE/p50-$TAG.steps | median)
    T1M=$(awk '$1>=1{print $2}' $BASE/p50-$TAG.steps | median)
    MIN=$(awk 'NR==1||$2<m{m=$2} END{print m}' $BASE/p50-$TAG.steps)
    MAX=$(awk 'NR==1||$2>m{m=$2} END{print m}' $BASE/p50-$TAG.steps)
    log "P50_STAT tag $TAG ngl $NGL med_all $ALLM med_step1 $T1M min $MIN max $MAX state $STATE"
    grep -E "P50_MEM_END|P50_TEMP_END|P50_POWER|SKEL " $BASE/p50-$TAG.log 2>/dev/null | sed "s/^/P50_META tag $TAG /" >> $LOG
    NODES=$(grep "^NODE_LINES" $BASE/p50-$TAG.log 2>/dev/null)
    log "P50_NODES tag $TAG $NODES"
    LAST=$TAG
done
log "P50_DONE"
