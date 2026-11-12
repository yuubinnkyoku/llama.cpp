#!/system/bin/sh
# Phase 51 - finish the 27B comparison in one command.
#
#   sh p51_finish.sh [ngl] [pairs]
#
# Runs four ABBA pairs of SPLIT=0 against the fixed async row split at one NGL,
# then prints one line per run with the exit code, the smallest decode step, the
# cap samples and the logits hash, and finally the pair ratios.
#
# Refuses unless the device can take it: the skel must be Phase 50 D, no probe may
# be running, and memory must clear the same gates run50.sh uses. It stops and
# prints WAIT if the cap moves inside a run, because such a pair is not a pair.
set -u
BASE=/data/local/tmp/c46-27b
SKEL=288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d
NGL=${1:-9}
PAIRS=${2:-4}

refuse() { echo "REFUSING: $*"; exit 2; }

[ -d $BASE ] || refuse "missing $BASE"
ACT=$(sha256sum $BASE/libs/libggml-htp-v81.so | cut -d' ' -f1)
[ "$ACT" = "$SKEL" ] || refuse "skel $ACT is not Phase 50 D"
ps -A -o NAME | grep -q decode-probe && refuse "a decode-probe is already running"
# run50.sh checks MemAvailable and SwapFree itself, after the force-stop list,
# so this script does not pre-check them: a loaded phone can still pass.
echo "PRE OK skel $ACT ngl $NGL pairs $PAIRS (memory is checked by run50.sh after the force stop)"

i=0
while [ "$i" -lt "$PAIRS" ]; do
    i=$((i + 1))
    if [ $((i % 2)) -eq 1 ]; then ORDER="off:0 async:0.25"; else ORDER="async:0.25 off:0"; fi
    SPLIT_WEIGHTS=$(cat $BASE/libs/split33.txt) WANT_SKEL=$SKEL sh $BASE/p51_series.sh "$NGL" 4 replay $ORDER
done

echo "FINISH_DONE"
echo "read the SER lines above: every pair is one off run and one async run next to it"
echo "a pair counts only when both runs have exit 0, the same cap samples, and a min_ms"
echo "async runs must all share one logits hash (the fixed path is deterministic)"
