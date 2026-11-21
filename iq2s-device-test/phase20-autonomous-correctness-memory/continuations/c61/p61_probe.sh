#!/system/bin/sh
set -u
BASE=/data/local/tmp/c61-inkling2
L=$BASE/libs
M=$BASE/models/Inkling-2-Mini-AD-IQ2_XXS.gguf
ID=${1:-p1}
STEPS=${2:-32}
NGL=${3:-48}
DEV=${4:-HTP0:0}
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
export GGML_HEXAGON_VERBOSE=${VERBOSE:-0}
export GGML_HEXAGON_2D_ONLY=${HEX2D:-0}
export GGML_HEXAGON_SMALL_MM_CPU=${SMALLMM:-0}
if [ -n "${OPFILTER:-}" ]; then export GGML_HEXAGON_OPFILTER="$OPFILTER"; fi
if [ -f "$BASE/opfilter.txt" ]; then export GGML_HEXAGON_OPFILTER="$(cat "$BASE/opfilter.txt")"; fi
export P20_EARLY_SOURCE_RELEASE=1
export P19_NO_EXTRA_BUFTS=1
export P53_STEPS_MAX=64
export P53_N_CTX=${CTX:-512}
export P53_N_BATCH=${NB:-32}
export P53_N_UBATCH=${NUB:-32}

{
    echo "ID $ID STEPS $STEPS NGL $NGL"
    date
    sha256sum $L/libggml-hexagon.so $L/libllama.so $L/decode-probe $L/libggml-htp-v81.so
    grep -E '^(MemAvailable|SwapFree|MemFree):' /proc/meminfo
    grep -E '^pswpout ' /proc/vmstat
} > "$R/manifest.txt"

timeout -k 10 -s TERM 600 /data/local/tmp/c15-native-swap-guard-v1/guard-v2 /proc/vmstat 262144 20 \
    $L/decode-probe $M $BASE/override-inkling.txt $STEPS $R/logits.f32 mmap $DEV $NGL \
    > "$R/stdout.txt" 2> "$R/stderr.txt" &
PID=$!

# Stop the run if MemAvailable falls below 3 GB. The swap guard is the second line.
(
    while kill -0 $PID 2>/dev/null; do
        AV=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
        if [ -n "$AV" ] && [ "$AV" -lt 3000000 ]; then
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

wait $PID
RC=$?
kill $MON 2>/dev/null
wait $MON 2>/dev/null

echo "RC $RC" > "$R/rc.txt"
{
    grep -E '^(MemAvailable|SwapFree|MemFree):' /proc/meminfo
    grep -E '^pswpout ' /proc/vmstat
} >> "$R/manifest.txt"
exit $RC
