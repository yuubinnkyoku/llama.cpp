#!/system/bin/sh
# Phase 43 post load warm. Two touch passes at most. Gate: 90 percent resident.
# Usage: sh warmup-c43.sh <run dir>
#
# Phase 56: the gate is now tunable, because the whole-file residency stops
# being a meaningful gate once a large part of the weights lives in the HTP
# buffer: those source pages are released on purpose (P20_EARLY_SOURCE_RELEASE),
# so the whole-file percentage cannot reach 90 no matter how warm the CPU part
# is. WARM_RES_MIN_PCT sets the gate (default 90, the Phase 43 value) and
# WARM_SKIP_TOUCH=1 skips the touch passes for a fast load-only scan. Both are
# recorded by the caller.
set -u
RUN=${1:-/data/local/tmp/c43-27b/runs/manual}
L=/data/local/tmp/c43-27b/libs
M=/data/local/tmp/llama.cpp/models/Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf
RR=$L/gguf-residency
RG=$L/ranges-ngl23.txt
LOG=$RUN/warmup-touch.log
MIN=${WARM_RES_MIN_PCT:-90}
SKIP_TOUCH=${WARM_SKIP_TOUCH:-0}
: > "$LOG"

snap() {
    echo "== $1 $(cut -d' ' -f1 /proc/uptime)" >> "$LOG"
    awk '/^(MemFree|MemAvailable|Cached|SwapFree|AnonPages):/{print}' /proc/meminfo >> "$LOG"
    awk '/^(pswpout|pgmajfault) /{print}' /proc/vmstat >> "$LOG"
}
res() {
    $RR $M $RG "$1" 2>/dev/null | awk -v t="$1" '/^RESIDENCY_RANGES/{printf "RES %s %s/%s (%d%%)\n", t, $9, $7, int($9*100/$7)}' | tee -a "$LOG"
}

START_SWAP=$(awk '/^pswpout /{print $2}' /proc/vmstat)
snap before
res before

FINAL=0
TOTAL=0
if [ "$SKIP_TOUCH" = "1" ]; then
    echo "WARM_SKIP touch min $MIN"
else
for pass in 1 2; do
    T0=$(cut -d' ' -f1 /proc/uptime)
    $RR $M $RG touch --touch >> "$LOG" 2>&1
    T1=$(cut -d' ' -f1 /proc/uptime)
    OUT=$(res "pass$pass")
    echo "$OUT"
    FINAL=$(echo "$OUT" | awk '{print $4}' | tr -d '()%' )
    TOTAL=$(echo "$OUT" | awk -F'[ /]+' '{print $4}')
    W=$(awk -v a=$T0 -v b=$T1 'BEGIN{printf "%.2f", b-a}')
    echo "WARM_PASS $pass wall_s $W"
    [ -z "$FINAL" ] && FINAL=0
    [ "$FINAL" -ge "$MIN" ] && break
    NOW_SWAP=$(awk '/^pswpout /{print $2}' /proc/vmstat)
    SWAPPED=$((NOW_SWAP - START_SWAP))
    echo "WARM_SWAPOUT pages $SWAPPED"
    [ "$SWAPPED" -gt 100000 ] && { echo "WARM_STOP swapout pages $SWAPPED"; break; }
done
fi

snap after
OUT=$(res after)
echo "$OUT"
FINAL=$(echo "$OUT" | awk '{print $4}' | tr -d '()%')
[ -z "$FINAL" ] && FINAL=0

if [ "$FINAL" -lt "$MIN" ]; then
    echo "WARM_FAIL resident_pages pct $FINAL min $MIN"
    exit 5
fi
echo "WARM_DONE pct $FINAL min $MIN"
exit 0
