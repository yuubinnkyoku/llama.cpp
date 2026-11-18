#!/system/bin/sh
# Phase 58 - long running decode curve on llama-server.
#
#   sh p58_curve.sh <tag> [duration_s] [interval_s]
#
# Keeps one llama-server (ngl 44, the default placement) alive and sends one
# greedy completion request per interval, so the model is never reloaded and the
# only variable is how long the session has been running. Every sample records
# the request timings, the vendor cap, the temperature, the memory and the swap
# counters. The screen is woken every sample because the vendor drops the cpu
# cap to 883 MHz when it sleeps (measured: 883200 asleep, 1785600 awake).
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/cli56
M=/data/local/tmp/llama.cpp/models/Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf
TAG=${1:-curve}
DURATION=${2:-9000}
INTERVAL=${3:-60}
PORT=${PORT:-18082}
LOG=$BASE/p58-$TAG.log
SRVLOG=$BASE/p58-$TAG.srv
OUT=$BASE/p58-$TAG.samples

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }
memv() { awk -v k="$1" '$1==k":"{print $2}' /proc/meminfo; }
swpo() { awk '/^pswpout /{print $2}' /proc/vmstat; }

echo "P58_CURVE start tag $TAG duration $DURATION interval $INTERVAL port $PORT cap0 $(cap) tz0 $(tz0) uptime $(cut -d' ' -f1 /proc/uptime)" > $LOG
echo "# uptime_s prefill_ms pred_n pred_ms pred_per_tok_ms cap0 tz0 memavail_kb swapfree_kb pswpout" > $OUT

SRVPID=""
if curl -s --max-time 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
    echo "P58_CURVE server already listening on $PORT" >> $LOG
else
    setsid sh -c "exec env LD_LIBRARY_PATH=$L ADSP_LIBRARY_PATH=$L \
        GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_NHVX=8 GGML_HEXAGON_OPPOLL=0 \
        GGML_HEXAGON_MM_SELECT=1 GGML_HEXAGON_IQ2S_GATHER=1 GGML_HEXAGON_IQ3XXS_GATHER=1 \
        GGML_HEXAGON_IQ3S_GATHER=1 GGML_HEXAGON_FFN_OFFLOAD=1 GGML_HEXAGON_OPBATCH=1280 \
        P20_EARLY_SOURCE_RELEASE=1 P19_NO_EXTRA_BUFTS=1 P44_NTHREADS=6 \
        timeout -k 5 -s TERM $((DURATION + 1800)) $L/llama-server -m $M -ngl 44 -c 512 -t 6 \
        --no-repack --no-warmup --port $PORT" > $SRVLOG 2>&1 < /dev/null &
    echo "P58_CURVE starting server, log $SRVLOG" >> $LOG
    i=0
    while [ $i -lt 240 ]; do
        curl -s --max-time 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
        sleep 2
        i=$((i + 1))
    done
    echo "P58_CURVE server ready after $((i * 2)) s" >> $LOG
fi

T0=$(date +%s)
TEND=$((T0 + DURATION))
N=0
while [ "$(date +%s)" -lt "$TEND" ]; do
    input keyevent KEYCODE_WAKEUP >/dev/null 2>&1
    N=$((N + 1))
    S0=$(date +%s)
    R=$(curl -s --max-time 900 http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
        -d '{"prompt":"The history of the city of Kyoto begins","n_predict":48,"temperature":0,"cache_prompt":false}' 2>/dev/null)
    RC=$?
    S1=$(date +%s)
    PM=$(echo "$R" | grep -o '"prompt_ms":[0-9.]*' | head -1 | cut -d: -f2)
    PN=$(echo "$R" | grep -o '"predicted_n":[0-9]*' | head -1 | cut -d: -f2)
    PD=$(echo "$R" | grep -o '"predicted_ms":[0-9.]*' | head -1 | cut -d: -f2)
    PT=$(echo "$R" | grep -o '"predicted_per_token_ms":[0-9.]*' | head -1 | cut -d: -f2)
    echo "P58_SAMPLE n $N t $(cut -d' ' -f1 /proc/uptime) rc $RC wall_s $((S1 - S0)) prompt_ms ${PM:-NA} pred_n ${PN:-NA} pred_ms ${PD:-NA} per_tok_ms ${PT:-NA} cap0 $(cap) tz0 $(tz0) memavail $(memv MemAvailable) swapfree $(memv SwapFree) pswpout $(swpo)" >> $LOG
    echo "$(cut -d' ' -f1 /proc/uptime) ${PM:-0} ${PN:-0} ${PD:-0} ${PT:-0} $(cap) $(tz0) $(memv MemAvailable) $(memv SwapFree) $(swpo)" >> $OUT
    # keep the cadence: the request itself is the work, sleep the remainder
    NOW=$(date +%s)
    SLEEP=$((INTERVAL - (NOW - S0)))
    [ $SLEEP -gt 0 ] && sleep $SLEEP
done

echo "P58_CURVE done samples $N uptime $(cut -d' ' -f1 /proc/uptime) cap0 $(cap) tz0 $(tz0)" >> $LOG
for p in $(ps -A -o PID,NAME | awk '/llama-server/{print $1}'); do
    echo "P58_CURVE stop server $p" >> $LOG
    kill -9 $p 2>/dev/null
done
echo "P58_CURVE end" >> $LOG
