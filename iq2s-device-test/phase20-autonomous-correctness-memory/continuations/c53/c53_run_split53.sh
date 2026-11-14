#!/system/bin/sh
# Phase 46 - single tensor row split runs for the 27B.
#
#   sh run_split.sh <off|seq|async> <steps> [dump.bin]
#
# off   : split disabled, the reference path (GGML_HEXAGON_SPLIT=0)
# seq   : split enabled, the host share runs before the DSP share
# async : split enabled, the DSP share is submitted first, then the host share
#
# Only one tensor is ever split: blk.57.ffn_up.weight (IQ3_S) by default. The environment is exported, and the
# runner passes its own environment through `env`, so no runner edit is needed
# for these variables. P20_EARLY_SOURCE_RELEASE is the one variable the runner
# sets explicitly; it reads it from EARLY_REL (default 1, unchanged behaviour).
set -u

BASE=/data/local/tmp/c46-27b
L=$BASE/libs

MODE=${1:-off}
STEPS=${2:-2}
DUMP=${3:-}
# replay pins the generated tokens to /data/local/tmp/c46-27b/libs/replay.txt and
# the probe then requires steps == the token count there (4). noreplay takes any
# step count and samples freely; use it for the short correctness passes.
TOKENS=${REPLAY_MODE:-noreplay}

SPLIT=0
SEQ=0
case "$MODE" in
    off)   SPLIT=0; SEQ=0 ;;
    seq)   SPLIT=1; SEQ=1 ;;
    async) SPLIT=1; SEQ=0 ;;
    *) echo "REFUSING: unknown mode '$MODE'"; exit 2 ;;
esac

export GGML_HEXAGON_SPLIT=$SPLIT
export GGML_HEXAGON_SPLIT_SEQ=$SEQ
export GGML_HEXAGON_SPLIT_WEIGHTS=${SPLIT_WEIGHTS:-blk.57.ffn_up.weight}
export GGML_HEXAGON_SPLIT_TRACE=${TRACE:-0}
export GGML_HEXAGON_SPLIT_THREADS=${SPLIT_THREADS:-6}
export GGML_HEXAGON_DMA64=0
export GGML_HEXAGON_TIMELINE=${TIMELINE:-0}
export OPPOLL=0
export EARLY_REL=${EARLY_REL:-0}
export OV=${OV:-$L/override-base.txt}

if [ -n "$DUMP" ]; then
    rm -f "$BASE/$DUMP" "$BASE/$DUMP.ref"
    export GGML_HEXAGON_SPLIT_DUMP="$BASE/$DUMP"
    # the split op writes the full tensor reference of the same step here
    export GGML_HEXAGON_SPLIT_DUMP_REF="$BASE/$DUMP.ref"
else
    unset GGML_HEXAGON_SPLIT_DUMP
    unset GGML_HEXAGON_SPLIT_DUMP_REF
fi

echo "SPLIT_RUN mode $MODE steps $STEPS tokens $TOKENS dump ${DUMP:-none} dump_path ${GGML_HEXAGON_SPLIT_DUMP:-none}"
echo "SPLIT_ENV split $GGML_HEXAGON_SPLIT seq $GGML_HEXAGON_SPLIT_SEQ trace $GGML_HEXAGON_SPLIT_TRACE threads $GGML_HEXAGON_SPLIT_THREADS"
echo "SPLIT_ENV dma64 $GGML_HEXAGON_DMA64 oppoll $OPPOLL early_release $EARLY_REL timeline $GGML_HEXAGON_TIMELINE"
echo "SPLIT_OV $OV"

sh $BASE/run53.sh run "$STEPS" "$TOKENS" "${PROFILE_ARG:-0}" "${THREADS:-7}"
RC=$?

NEW=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
if [ -n "$NEW" ]; then
    {
        echo "SPLIT_MODE $MODE"
        echo "SPLIT_STEPS $STEPS"
        echo "SPLIT_TOKENS $TOKENS"
        echo "SPLIT_DUMP ${DUMP:-none}"
        echo "SPLIT_WEIGHTS $GGML_HEXAGON_SPLIT_WEIGHTS"
        echo "SPLIT_TRACE $GGML_HEXAGON_SPLIT_TRACE"
        echo "SPLIT_THREADS $GGML_HEXAGON_SPLIT_THREADS"
        echo "DMA64 $GGML_HEXAGON_DMA64"
        echo "OPPOLL $OPPOLL"
        echo "EARLY_RELEASE $EARLY_REL"
        echo "OVERRIDE $OV"
    } > "$NEW/c46-env.txt" 2>/dev/null
fi
echo "SPLIT_RUN rc $RC dir $NEW"
exit $RC
