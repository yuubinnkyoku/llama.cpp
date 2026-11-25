#!/system/bin/sh
# Phase 64 - direct measurement that tiled engages in llama-cli.
#
# Phase 63 only showed "it ran with no tiled env set". This measures the thing
# itself: the same prompt and the same binary, twice, once with the tiled
# default on and once with GGML_CPU_TILED_MM=0. If tiled runs, the prefill has
# to be faster by roughly the margin Phase 60/61 measured.
#
# A long prompt keeps the prefill batch above the rows >= 8 floor.
#
#   sh p64_tiled.sh <tiled_on|tiled_off> [tag]
set -u
BASE=/data/local/tmp/c46-27b
C=$BASE/cli63
M=/data/local/tmp/llama.cpp/models/Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf
MODE=${1:-tiled_on}
TAG=${2:-t64}

# a prompt long enough that prefill runs in batches of 8 or more
PROMPT="The city of Paris has been the capital of France for many centuries and its history is long and complicated. It grew from a small settlement on the banks of the Seine into one of the largest cities in Europe. The river divides the city into two parts and many famous buildings stand on both sides of it. People from all over the world visit the city every year to see its museums and its art. The weather there is mild for most of the year and the food is well known. Write one sentence about this city."

case "$MODE" in
    tiled_on)  EXTRA="" ;;
    tiled_off) EXTRA="GGML_CPU_TILED_MM=0" ;;
    *) echo "P64T REFUSING unknown mode $MODE"; exit 2 ;;
esac

OUT=$BASE/p64t-$TAG-$MODE.log
echo "P64T start mode $MODE uptime $(cut -d' ' -f1 /proc/uptime) cap0 $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq) tz0 $(cat /sys/class/thermal/thermal_zone0/temp)"
echo "P64T mem $(grep MemAvailable /proc/meminfo | awk '{print $2}') swapfree $(grep SwapFree /proc/meminfo | awk '{print $2}')"
echo "P64T env GGML_CPU_TILED_MM=${GGML_CPU_TILED_MM:-unset} extra ${EXTRA:-none}"

S_START=$(date +%s%N)
env LD_LIBRARY_PATH=$C ADSP_LIBRARY_PATH=$C \
    GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_NHVX=8 GGML_HEXAGON_OPPOLL=0 \
    GGML_HEXAGON_MM_SELECT=1 GGML_HEXAGON_IQ2S_GATHER=1 GGML_HEXAGON_IQ3XXS_GATHER=1 \
    GGML_HEXAGON_IQ3S_GATHER=1 GGML_HEXAGON_FFN_OFFLOAD=1 GGML_HEXAGON_PROFILE=0 \
    GGML_HEXAGON_OPBATCH=1280 P20_EARLY_SOURCE_RELEASE=1 P44_NTHREADS=8 \
    $EXTRA \
    timeout -k 5 -s TERM 900 \
    $C/llama-cli -m $M -ngl 44 -c 1024 -n 1 -t 8 --temp 0 \
        -p "$PROMPT" --no-warmup -st > $OUT 2>&1
RC=$?
S_END=$(date +%s%N)
WALL_MS=$(( (S_END - S_START) / 1000000 ))
echo "P64T rc $RC wall_ms $WALL_MS"
echo "P64T rate $(grep -oE 'Prompt: [0-9.]+ t/s' $OUT | head -1)"
echo "P64T cap_end $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq) tz_end $(cat /sys/class/thermal/thermal_zone0/temp)"
echo "P64T mem_after $(grep MemAvailable /proc/meminfo | awk '{print $2}') swapfree $(grep SwapFree /proc/meminfo | awk '{print $2}')"
echo "P64T stray $(ps -A -o PID,NAME | awk '/llama-cli/{printf "%s ", $1}')"
