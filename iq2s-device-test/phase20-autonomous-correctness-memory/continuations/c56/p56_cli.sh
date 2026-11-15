#!/system/bin/sh
# Phase 56 - llama-cli at two placements. Phase 55 recorded that llama-cli
# stalls at -ngl 3 while the probe runs at -ngl 16, so this asks whether the
# practical entry point also works at the placement the probe likes.
#
#   sh p56_cli.sh "<ngl list>"
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/cli56
M=/data/local/tmp/llama.cpp/models/Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf
LIST=${1:-"3 44"}
TMO=${TMO:-600}

echo "P56_CLI start cli $(sha256sum $L/llama-cli | cut -d' ' -f1) skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1) ngl $LIST timeout $TMO"
for N in $LIST; do
    T=cli-n$N
    S0=$(date +%s)
    timeout -k 5 -s TERM $TMO env LD_LIBRARY_PATH=$L ADSP_LIBRARY_PATH=$L \
        GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_NHVX=8 GGML_HEXAGON_OPPOLL=0 \
        GGML_HEXAGON_MM_SELECT=1 GGML_HEXAGON_IQ2S_GATHER=1 GGML_HEXAGON_IQ3XXS_GATHER=1 \
        GGML_HEXAGON_IQ3S_GATHER=1 GGML_HEXAGON_FFN_OFFLOAD=1 GGML_HEXAGON_PROFILE=0 \
        GGML_HEXAGON_OPBATCH=1280 P20_EARLY_SOURCE_RELEASE=1 P19_NO_EXTRA_BUFTS=1 P44_NTHREADS=6 \
        $L/llama-cli -m $M -ngl $N -lm mmap -c 128 -b 32 -t 6 -n 4 --no-repack --no-warmup \
        -p "The capital of Japan is" > $BASE/p56-$T.out 2>&1
    RC=$?
    S1=$(date +%s)
    echo "P56_CLI ngl $N rc $RC wall_s $((S1 - S0))"
    grep -E "offloading|assigned to device HTP|Prompt:|Generation:|Exiting|error|failed" $BASE/p56-$T.out 2>/dev/null | tail -6 | sed 's/^/P56_CLI_OUT /'
    for p in $(ps -A -o PID,NAME | awk '/llama-cli/{print $1}'); do
        echo "P56_CLI KILL $p"
        kill -9 $p 2>/dev/null
    done
    sleep 2
done
echo "P56_CLI_DONE stray $(ps -A -o PID,NAME | awk '/llama-cli/{printf "%s ", $1}')"
