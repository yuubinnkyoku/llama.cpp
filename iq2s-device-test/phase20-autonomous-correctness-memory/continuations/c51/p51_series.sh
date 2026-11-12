#!/system/bin/sh
# Phase 51 - run a series of split conditions and print one line per run.
#
#   sh p51_series.sh <ngl> <steps> <tok> <mode:frac> [<mode:frac> ...]
#
# mode is off, seq or async; frac is the CPU share for the split. The split
# weight list comes from SPLIT_WEIGHTS in the environment. The caller chooses a
# palindromic order so the drift of a warming device cancels.
set -u
BASE=/data/local/tmp/c46-27b
NGL=${1:-9}
STEPS=${2:-4}
TOK=${3:-replay}
shift 3

i=0
for spec in "$@"; do
    i=$((i + 1))
    MODE=${spec%%:*}
    FRAC=${spec##*:}
    TAG=$(printf "%s%d-%s-%s" "${SER_PFX:-s}" "$i" "$MODE" "$FRAC")
    export GGML_HEXAGON_SPLIT_CPU_FRAC=$FRAC
    P51_MODE=$MODE sh $BASE/p51_run.sh "$STEPS" "$TOK" "$NGL" "$TAG" 1 > $BASE/p51-series-$TAG.log 2>&1
    R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    MIN=$(awk '/^P51_OUT DECODE step/{ if (m == "" || $6 + 0 < m) m = $6 + 0 } END { print (m == "" ? "none" : m) }' $BASE/p51-series-$TAG.log)
    STEPS_MS=$(awk '/^P51_OUT DECODE step/{printf "%s,", $6}' $BASE/p51-series-$TAG.log)
    SWAP=$(awk '/^P51_MEM_END/{print $9}' $BASE/p51-series-$TAG.log)
    SHA=$(sha256sum $R/logits.f32 2>/dev/null | cut -d' ' -f1)
    CAPS=$(awk 'NR==1{a=$2"_"$3; next} {c[$2"_"$3]++} END{printf "first %s", a; for (k in c) printf " | %s x%d", k, c[k]}' $BASE/p51-$TAG.caps 2>/dev/null)
    echo "SER $TAG mode $MODE frac $FRAC exit ${EXIT:-none} min_ms $MIN swap $SWAP caps $CAPS steps $STEPS_MS"
    echo "SER $TAG logits ${SHA:-none}"
    echo "SER $TAG dir $R"
done
echo SERIES_DONE
