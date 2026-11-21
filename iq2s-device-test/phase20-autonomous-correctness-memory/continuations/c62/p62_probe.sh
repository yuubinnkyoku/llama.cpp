#!/system/bin/sh
# Phase 62 guarded probe for Inkling 2 Mini on the HTP.
#
# The 1 GiB swap rule stays for every other model. For this model only the
# owner allowed 1.5 GiB, paired with a MemAvailable abort line, because the
# 9.7 GiB mmap model makes the CPU-only control cross 1 GiB without the HTP.
#
# TWO_PHASE=1 splits the run at the probe's post-prefill halt. The swap guard
# then covers the decode window only. Both deltas are recorded, so a run is
# never reported as passing on a part of its life.
set -u
BASE=/data/local/tmp/c61-inkling2
L=$BASE/libs
M=$BASE/models/Inkling-2-Mini-AD-IQ2_XXS.gguf
ID=${1:-p1}
STEPS=${2:-32}
NGL=${3:-48}
DEV=${4:-HTP0:0}
SWAP_LIMIT=${SWAP_LIMIT:-393216}   # 1.5 GiB in 4 KiB pages, this model only
MEM_LIMIT=${MEM_LIMIT:-2800000}    # MemAvailable abort line in kB
RUN_TIMEOUT=${RUN_TIMEOUT:-1800}
R=$BASE/runs/$ID
mkdir -p "$R"

export LD_LIBRARY_PATH=$L
export ADSP_LIBRARY_PATH=$L
export GGML_HEXAGON_DEVICES=HTP0:0
export GGML_HEXAGON_NHVX=8
export GGML_HEXAGON_OPPOLL=0
export GGML_HEXAGON_MM_SELECT=1
export GGML_HEXAGON_IQ2S_GATHER=1
export GGML_HEXAGON_IQ3XXS_GATHER=1
export GGML_HEXAGON_IQ3S_GATHER=1
export GGML_HEXAGON_FFN_OFFLOAD=1
export GGML_HEXAGON_OPBATCH=${OPBATCH:-1280}
export GGML_HEXAGON_OPQUEUE=${OPQUEUE:-32}
export GGML_HEXAGON_DMA64=${DMA64:-1}
export GGML_HEXAGON_FA_SELECT=${FA_SELECT:-2}
export GGML_HEXAGON_VERBOSE=${VERBOSE:-0}
export GGML_HEXAGON_2D_ONLY=${HEX2D:-0}
export GGML_HEXAGON_SMALL_MM_CPU=${SMALLMM:-0}
if [ -n "${OPFILTER:-}" ]; then export GGML_HEXAGON_OPFILTER="$OPFILTER"; fi
if [ -f "$BASE/opfilter.txt" ]; then export GGML_HEXAGON_OPFILTER="$(cat "$BASE/opfilter.txt")"; fi
export P20_EARLY_SOURCE_RELEASE=${PRELEASE:-1}
export P19_NO_EXTRA_BUFTS=1
export P53_STEPS_MAX=64
export P53_N_CTX=${CTX:-512}
export P53_N_BATCH=${NB:-32}
export P53_N_UBATCH=${NUB:-32}
if [ -n "${PT:-}" ]; then export P53_PROMPT_TOKENS=$PT; fi
if [ -n "${PLIST:-}" ]; then export P53_PROMPT_LIST=$PLIST; fi
if [ -n "${NOLOGITS:-}" ]; then export P53_NO_LOGITS=1; fi
if [ -n "${TEMP:-}" ]; then export P53_TEMP=$TEMP; fi
if [ -n "${PROMPT_FILE:-}" ]; then export P53_PROMPT_FILE=$PROMPT_FILE; fi
if [ -n "${REPLAY:-}" ]; then export P19_REPLAY_TOKENS=$REPLAY; fi
if [ "${TWO_PHASE:-0}" = "1" ]; then
    export P27_RESUME2_FILE=$R/resume2.txt
    rm -f "$R/resume2.txt"
fi

{
    echo "ID $ID STEPS $STEPS NGL $NGL DEV $DEV SWAP_LIMIT $SWAP_LIMIT MEM_LIMIT $MEM_LIMIT TWO_PHASE ${TWO_PHASE:-0}"
    date
    sha256sum $L/libggml-hexagon.so $L/libllama.so $L/decode-probe $L/libggml-htp-v81.so
    grep -E '^(MemAvailable|SwapFree|MemFree):' /proc/meminfo
    grep -E '^pswpout ' /proc/vmstat
} > "$R/manifest.txt"
START_SWAP=$(awk '/^pswpout /{print $2}' /proc/vmstat)
echo "$START_SWAP" > "$R/start_swap.txt"

if [ "${TWO_PHASE:-0}" = "1" ]; then
    timeout -k 10 -s TERM $RUN_TIMEOUT \
        $L/decode-probe $M $BASE/${OVERRIDE:-override-inkling.txt} $STEPS $R/logits.f32 mmap $DEV $NGL \
        > "$R/stdout.txt" 2> "$R/stderr.txt" &
    PID=$!
else
    timeout -k 10 -s TERM $RUN_TIMEOUT /data/local/tmp/c15-native-swap-guard-v1/guard-v2 /proc/vmstat $SWAP_LIMIT 20 \
        $L/decode-probe $M $BASE/${OVERRIDE:-override-inkling.txt} $STEPS $R/logits.f32 mmap $DEV $NGL \
        > "$R/stdout.txt" 2> "$R/stderr.txt" &
    PID=$!
fi

# Stop the run if MemAvailable falls below the abort line.
(
    while kill -0 $PID 2>/dev/null; do
        AV=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
        if [ -n "$AV" ] && [ "$AV" -lt "$MEM_LIMIT" ]; then
            echo "MEM_GUARD $AV" >> "$R/guard.txt"
            kill -TERM $PID 2>/dev/null
            sleep 2
            kill -KILL $PID 2>/dev/null
            break
        fi
        sleep 1
    done
) &
MON=$!

# Memory trace, one line every 10 s, so a 32-step run has a timeline.
(
    while kill -0 $PID 2>/dev/null; do
        echo "$(date +%s) $(awk '/MemAvailable/{print $2}' /proc/meminfo) $(awk '/SwapFree/{print $2}' /proc/meminfo) $(awk '/^pswpout /{print $2}' /proc/vmstat)" >> "$R/memlog.txt"
        sleep 10
    done
) &
MEM=$!

if [ "${TWO_PHASE:-0}" = "1" ]; then
    # Wait for the post-prefill halt, then cover decode with the swap guard.
    n=0
    while kill -0 $PID 2>/dev/null; do
        if grep -q "pause2_begin" "$R/stdout.txt" 2>/dev/null; then break; fi
        n=$((n + 1))
        [ $n -gt 1200 ] && break
        sleep 0.5
    done
    DECODE_SWAP=$(awk '/^pswpout /{print $2}' /proc/vmstat)
    echo "$DECODE_SWAP" > "$R/decode_swap.txt"
    (
        while kill -0 $PID 2>/dev/null; do
            cur=$(awk '/^pswpout /{print $2}' /proc/vmstat)
            delta=$((cur - DECODE_SWAP))
            if [ $delta -gt $SWAP_LIMIT ]; then
                echo "DECODE_SWAP_GUARD $delta" >> "$R/guard.txt"
                kill -TERM $PID 2>/dev/null
                sleep 2
                kill -KILL $PID 2>/dev/null
                break
            fi
            sleep 1
        done
    ) &
    SW=$!
    echo "RESUME2_OK" > "$R/resume2.txt"
    wait $PID
    RC=$?
    kill $SW 2>/dev/null
    wait $SW 2>/dev/null
else
    wait $PID
    RC=$?
fi
kill $MON 2>/dev/null
kill $MEM 2>/dev/null
wait $MON 2>/dev/null
wait $MEM 2>/dev/null

END_SWAP=$(awk '/^pswpout /{print $2}' /proc/vmstat)
echo "RC $RC" > "$R/rc.txt"
{
    echo "swap_start $START_SWAP swap_end $END_SWAP swap_total_pages $((END_SWAP - START_SWAP))"
    if [ -f "$R/decode_swap.txt" ]; then
        DEC=$(cat "$R/decode_swap.txt")
        echo "swap_decode_base $DEC swap_decode_pages $((END_SWAP - DEC))"
    fi
    echo "--- after"
    date
    grep -E '^(MemAvailable|SwapFree|MemFree):' /proc/meminfo
    grep -E '^pswpout ' /proc/vmstat
} >> "$R/manifest.txt"
echo "$(date +%s) $(awk '/MemAvailable/{print $2}' /proc/meminfo) $(awk '/SwapFree/{print $2}' /proc/meminfo) $(awk '/^pswpout /{print $2}' /proc/vmstat)" >> "$R/memlog.txt"
exit $RC
