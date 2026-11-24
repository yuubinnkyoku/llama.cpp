#!/system/bin/sh
# Phase 63 - run llama-cli once with a real prompt, no tiled env set.
#
# The point is to confirm two things at once:
#   1. the practical preset (chat template, EOG, repetition penalty) produces a
#      sensible answer
#   2. tiled engages without any env, because the ARM default is now on
#
# The prompt is long enough (>= 8 tokens) that prefill takes the tiled path.
set -u
BASE=/data/local/tmp/c46-27b
C=$BASE/cli63
M=/data/local/tmp/llama.cpp/models/Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf
OUT=$BASE/p63-cli.log

echo "P63C start uptime $(cut -d' ' -f1 /proc/uptime) cap0 $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq) tz0 $(cat /sys/class/thermal/thermal_zone0/temp)"
echo "P63C mem $(grep MemAvailable /proc/meminfo | awk '{print $2}') swapfree $(grep SwapFree /proc/meminfo | awk '{print $2}')"
echo "P63C libs $(sha256sum $C/libggml-cpu.so | cut -d' ' -f1) $(sha256sum $C/libggml-htp-v81.so | cut -d' ' -f1)"
echo "P63C tiled_env GGML_CPU_TILED_MM=${GGML_CPU_TILED_MM:-unset}"

# no GGML_CPU_TILED_MM: the build default decides
env LD_LIBRARY_PATH=$C ADSP_LIBRARY_PATH=$C \
    GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_NHVX=8 GGML_HEXAGON_OPPOLL=0 \
    GGML_HEXAGON_MM_SELECT=1 GGML_HEXAGON_IQ2S_GATHER=1 GGML_HEXAGON_IQ3XXS_GATHER=1 \
    GGML_HEXAGON_IQ3S_GATHER=1 GGML_HEXAGON_FFN_OFFLOAD=1 GGML_HEXAGON_PROFILE=0 \
    GGML_HEXAGON_OPBATCH=1280 P20_EARLY_SOURCE_RELEASE=1 P44_NTHREADS=8 \
    timeout -k 5 -s TERM 900 \
    $C/llama-cli -m $M -ngl 44 -c 512 -n 48 -t 8 --temp 0 \
        -p "What is the capital of France? Answer in one short sentence." \
        --no-warmup -st > $OUT 2>&1
RC=$?
echo "P63C rc $RC"
echo "P63C tail_out $(tail -20 $OUT 2>/dev/null | tr '\n' '|')"
echo "P63C perf $(grep -E 'prompt eval time|eval time' $OUT 2>/dev/null | tr '\n' '|')"
echo "P63C splits $(grep -oE 'graph: nodes = [0-9]+, splits = [0-9]+' $OUT 2>/dev/null | tail -1)"
echo "P63C mem_after $(grep MemAvailable /proc/meminfo | awk '{print $2}') swapfree $(grep SwapFree /proc/meminfo | awk '{print $2}')"
echo "P63C stray $(ps -A -o PID,NAME | awk '/llama-cli|decode-probe/{printf "%s ", $1}')"
