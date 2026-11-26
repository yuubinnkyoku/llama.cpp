#!/system/bin/sh
# Phase 65 - does more than one HTP session help the 27B?
#
# Our runs use GGML_HEXAGON_DEVICES=HTP0:0, so mdev_count is 1. That makes
# #31055's KV-head team split unreachable (n_teams = gcd(n_kv_heads, 1) = 1), so
# before porting it the question is whether a second session helps at all.
# PrismML's own hexagon preset runs four sessions on their model.
#
# Each case runs the same prefill and reports the prefill time, the device count
# the backend actually found, and cap so the numbers can be compared honestly.
#
#   sh p65_mdev.sh <tag> [rounds]
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
TAG=${1:-mdev}
ROUNDS=${2:-2}
SKEL=cdc628a9849b9f211c50bd7f2dca4c54e846b24034afaf189c2a1d2621c8abe2
TOKENS=${TOKENS:-128}

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

one() {  # $1 label, $2 device string
    LAB=$1; DEV=$2
    T="$TAG-$LAB"
    LOG=$BASE/p53-$T.out
    rm -f $LOG
    touch $BASE/.m-$T
    C0=$(cap); Z0=$(tz0)
    GDEV="$DEV" env GGML_HEXAGON_DEVICES="$DEV" WANT_SKEL=$SKEL \
        GUARD_PAGES=2097152 RUN_TIMEOUT=1800 MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
        P53_PROMPT_TOKENS=$TOKENS P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay 44 "$T" 0 > /dev/null 2>&1
    RC=$?
    C1=$(cap); Z1=$(tz0)
    if [ "$RC" != "0" ]; then echo "P65M_END $LAB rc $RC (no data)"; return 0; fi
    if [ ! -f $LOG ]; then echo "P65M_END $LAB no log"; return 0; fi
    if [ $BASE/.m-$T -nt $LOG ]; then echo "P65M_END $LAB stale log"; return 0; fi
    R=$(grep -oE "SPLIT_RUN rc [0-9]+ dir .*" $LOG 2>/dev/null | tail -1 | sed 's/.*dir //')
    if [ -z "$R" ] || [ ! -d "$R" ]; then echo "P65M_END $LAB no run dir"; return 0; fi
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    ND=$(grep -cE "ggml-hex: device [0-9]+:" $R/stderr.txt 2>/dev/null)
    PFL=$(grep -oE "PREFILL seg 0 tokens [0-9]+ ms [0-9.]+" $R/stdout.txt 2>/dev/null | head -1 | sed 's/.* ms //')
    ERR=$(grep -oE "failed to open session|HTP_STATUS[A-Z_]*" $R/stderr.txt 2>/dev/null | head -1)
    echo "P65M_END $LAB dev $DEV exit ${EXIT:-NA} ndev ${ND:-0} prefill_ms ${PFL:-NA} cap0 $C0-$C1 tz0 $Z0-$Z1 ${ERR:-}"
}

echo "P65M start tag $TAG rounds $ROUNDS tokens $TOKENS uptime $(cut -d' ' -f1 /proc/uptime) cap0 $(cap) tz0 $(tz0)"
cp $L/replay32.txt $L/replay.txt

i=1
while [ $i -le $ROUNDS ]; do
    if [ $((i % 2)) -eq 1 ]; then
        one "r$i-one" "HTP0:0"
        one "r$i-two" "HTP0:0,HTP0:1"
    else
        one "r$i-two" "HTP0:0,HTP0:1"
        one "r$i-one" "HTP0:0"
    fi
    i=$((i + 1))
done

cp $L/replay.txt.std $L/replay.txt
if [ "$(cat $L/replay.txt)" = "25358 11 198 760" ]; then
    echo "P65M replay_restore OK"
else
    echo "P65M REPLAY_RESTORE_FAILED $(cat $L/replay.txt)"
fi
