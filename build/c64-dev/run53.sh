#!/system/bin/sh
# Phase 44 - NGL23 27B decode runner.
#
# Same safety envelope and warm-up discipline as the Phase 43 runner. The two
# changes are that PROFILE and the CPU thread count are explicit arguments, so
# one A/B changes exactly one variable, and every run records both.
#
# Usage (device):
#   sh run-27b-ngl23-c44.sh preflight
#   sh run-27b-ngl23-c44.sh run <steps> <replay|noreplay> <profile> <threads>
set -u

BASE=/data/local/tmp/c46-27b
RUNS=$BASE/runs
L=$BASE/libs
AP=$BASE/APPROVED
M=/data/local/tmp/llama.cpp/models/Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf
G=/data/local/tmp/c15-native-swap-guard-v1/guard-v2
# WANT_SKEL names the skel this run must use. The default is the Phase 49 build.
WANT=${WANT_SKEL:-312ad7b7ab304762f7c347e7909ee5d569541b15994073cfc11f55d0edf38114}
APPROVAL_TTL_S=${APPROVAL_TTL_S:-900}
MEM_MIN_KB=${MEM_MIN_KB:-9500000}
SWAP_MIN_KB=8000000
RES_MIN_PCT=95
RES_POST_MIN_PCT=90

RUN_TIMEOUT=${RUN_TIMEOUT:-180}
GUARD_PAGES=${GUARD_PAGES:-262144}
WAIT_LOOP=${WAIT_LOOP:-600}
MODE=${1:-run}
STEPS=${2:-4}
case "$STEPS" in ''*[!0-9]*) echo "REFUSING: STEPS must be a number"; exit 2 ;; esac
REPLAY_MODE=${3:-replay}
PROFILE=${4:-2}
case "$PROFILE" in ''*[!0-9]*) echo "REFUSING: PROFILE must be a number"; exit 2 ;; esac
NTHREADS=${5:-6}
case "$NTHREADS" in ''*[!0-9]*) echo "REFUSING: NTHREADS must be a number"; exit 2 ;; esac
if [ "$REPLAY_MODE" = "noreplay" ]; then
    REPLAY_ENV=""
else
    REPLAY_ENV="P19_REPLAY_TOKENS=$L/replay.txt"
fi
OV=${OV:-$L/override.txt}
# Phase 59: the repack buffer type is off by default (P19_NO_EXTRA_BUFTS=1).
# NOEXTRA=0 lets the cpu backend use its repacked weights.
NOEXTRA=${NOEXTRA:-1}
if [ "$NOEXTRA" = "1" ]; then NOEXTRA="P19_NO_EXTRA_BUFTS=1"; else NOEXTRA=""; fi
refuse() { echo "REFUSING: $*"; exit 2; }

res_pages() {  # $1 = substring of the tag, prints "<resident> <total> <pct>"
    $L/gguf-residency $M $L/ranges-ngl23.txt "$1" 2>/dev/null |
        awk '/^RESIDENCY_RANGES/{r=$9; t=$7} END{if (t>0) printf "%s %s %s\n", r, t, int(r*100/t); else print "0 1 0"}'
}

# ---- 1. identity ---------------------------------------------------------
[ -d $L ] || refuse "missing $L"
ACT=$(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)
[ "$ACT" = "$WANT" ] || refuse "skel sha256 $ACT != $WANT"
[ -f $M ] || refuse "model missing"
M_STAT=$(stat -c '%s %Y' $M)
case "$M_STAT" in "9259510912 "* ) ;; *) refuse "unexpected model size $M_STAT" ;; esac
ps -A -o NAME | grep -q decode-probe && refuse "a decode-probe is already running"

# ---- 2. memory -----------------------------------------------------------
MEM_AVAIL=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
SWAP_FREE=$(awk '/^SwapFree:/{print $2}' /proc/meminfo)
[ "$MEM_AVAIL" -ge $MEM_MIN_KB ] || refuse "MemAvailable ${MEM_AVAIL} kB < ${MEM_MIN_KB} kB"
[ "$SWAP_FREE" -ge $SWAP_MIN_KB ] || refuse "SwapFree ${SWAP_FREE} kB < ${SWAP_MIN_KB} kB"

# ---- 3. thermal gate -----------------------------------------------------
# Phase 43 showed 633 ms only while the vendor cap is released. The cap is
# recorded, never modified.
CAP0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null)
CAP7=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq 2>/dev/null)
CAP0_MAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null)
CAP7_MAX=$(cat /sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_max_freq 2>/dev/null)
THERMAL=fast
[ "$CAP0" = "$CAP0_MAX" ] && [ "$CAP7" = "$CAP7_MAX" ] || THERMAL=capped
echo "THERMAL_STATE $THERMAL cap0 $CAP0/$CAP0_MAX cap7 $CAP7/$CAP7_MAX"
# A capped run is allowed only when the caller says so. The cap is recorded in
# the manifest and never touched, so a capped run stays a separate condition.
if [ "$MODE" = "run" ] && [ "$THERMAL" != "fast" ] && [ "${ALLOW_CAPPED:-0}" != "1" ]; then
    refuse "CPU frequency is capped (cap0 $CAP0/$CAP0_MAX cap7 $CAP7/$CAP7_MAX); cool down first or set ALLOW_CAPPED=1"
fi

# ---- 4. fresh two pass prewarm ------------------------------------------
set -- $(res_pages pre0)
R0=$1; T=$2; P0=$3
echo "RESIDENCY pre0 ${R0}/${T} (${P0}%)"
if [ "$MODE" = "run" ] || [ "$P0" -lt $RES_MIN_PCT ]; then
    for pass in 1 2; do
        echo "PREWARM pass $pass start $(cut -d' ' -f1 /proc/uptime)"
        $L/gguf-residency $M $L/ranges-ngl23.txt touch --touch 2>&1 | tail -1
        set -- $(res_pages "prewarm$pass")
        echo "RESIDENCY prewarm$pass ${1}/${2} (${3}%)"
    done
else
    echo "PREWARM skipped, already ${P0}%"
fi
set -- $(res_pages pre)
R1=$1; T=$2; P1=$3
echo "RESIDENCY pre ${R1}/${T} (${P1}%)"
[ "$P1" -ge $RES_MIN_PCT ] || refuse "prewarm only reached ${P1}% (< ${RES_MIN_PCT}%)"

echo "PREFLIGHT_OK skel $ACT mem ${MEM_AVAIL} swap ${SWAP_FREE} resident ${R1}/${T} (${P1}%)"
[ "$MODE" = "preflight" ] && exit 0

# ---- 5. one shot approval (consumed only after the checks above) --------
[ -f $AP ] || refuse "no approval file. Create $AP with one line: '<unix epoch seconds> <note>'"
read -r AP_TS AP_NOTE < $AP || refuse "approval file unreadable"
case "$AP_TS" in ''|*[!0-9]*) refuse "approval file has no epoch seconds" ;; esac
NOW=$(date +%s)
AGE=$((NOW - AP_TS))
[ $AGE -ge 0 ] || refuse "approval timestamp is in the future"
[ $AGE -le $APPROVAL_TTL_S ] || refuse "approval expired ${AGE}s ago (limit ${APPROVAL_TTL_S}s)"
rm -f $AP || refuse "cannot consume the approval file"
echo "APPROVAL ok age_s $AGE note $AP_NOTE"

# ---- 6. run --------------------------------------------------------------
RUNID=$(date +%Y%m%d-%H%M%S)-$$
R=$RUNS/$RUNID
[ -e $R ] && refuse "$R already exists"
mkdir -p $R || refuse "cannot create $R"

{
    echo "RUNID $RUNID"
    echo "SKEL $ACT"
    echo "MODEL $M_STAT"
    echo "MEM_AVAIL_KB $MEM_AVAIL"
    echo "SWAP_FREE_KB $SWAP_FREE"
    echo "RESIDENT_PREWARM $R1/$T"
    echo "THERMAL_STATE $THERMAL CAP0 $CAP0 CAP7 $CAP7"
    echo "ENV GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_NHVX=8 GGML_HEXAGON_OPPOLL=${OPPOLL:-1}"
    echo "ENV GGML_HEXAGON_MM_SELECT=1 GGML_HEXAGON_IQ2S_GATHER=1 GGML_HEXAGON_IQ3XXS_GATHER=1"
    echo "ENV GGML_HEXAGON_IQ3S_GATHER=1 GGML_HEXAGON_FFN_OFFLOAD=1 GGML_HEXAGON_PROFILE=$PROFILE"
    echo "ENV GGML_HEXAGON_OPBATCH=1280 PAGEOUT=none STEPS=$STEPS TIMEOUT=${RUN_TIMEOUT:-180} GUARD_PAGES=$GUARD_PAGES GUARD_SWAPOUT_MIB=$((GUARD_PAGES / 256))"
    echo "ENV P44_NTHREADS=$NTHREADS GGML_HEXAGON_TIMELINE=${TIMELINE:-0} GGML_SCHED_DEBUG=${SCHED_DEBUG:-0} GGML_SCHED_SPLIT_TIMING=${SPLIT_TIMING:-0}"
    echo "OVERRIDE_FILE $OV"
    echo "OVERRIDE_SHA $(sha256sum $OV | cut -d' ' -f1)"
    echo "REPLAY_MODE $REPLAY_MODE"
    if [ "$REPLAY_MODE" = "noreplay" ]; then echo "REPLAY none"; else echo "REPLAY $(cat $L/replay.txt)"; fi
    sha256sum $L/libggml-hexagon.so $L/libllama.so $L/decode-probe-wait $L/gguf-residency
} > $R/manifest.txt

snap() {
    echo "== $1 $(cut -d' ' -f1 /proc/uptime)"
    grep -E '^(MemFree|MemAvailable|Cached|SwapFree|AnonPages):' /proc/meminfo
    grep -E '^(pswpout|pswpin|pgmajfault) ' /proc/vmstat
    C=$(cat $R/childpid 2>/dev/null)
    if [ -n "$C" ] && [ -r "/proc/$C/io" ]; then
        echo "READ_BYTES $(awk '/^read_bytes:/{print $2}' /proc/$C/io 2>/dev/null)"
        echo "CPU_TICKS $(awk '{print $14, $15}' /proc/$C/stat 2>/dev/null)"
        echo "CPU_FREQ $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null) $(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq 2>/dev/null) $(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq 2>/dev/null)"
    fi
}
res() { echo "$1 $(res_pages "$1")" >> $R/res-log.txt; $L/gguf-residency $M $L/ranges-ngl23.txt "$1" 2>/dev/null; }
cpustat() {
    : > $R/time_in_state_$1.txt
    for p in /sys/devices/system/cpu/cpufreq/policy*/stats/time_in_state; do
        echo "== $p" >> $R/time_in_state_$1.txt
        cat $p >> $R/time_in_state_$1.txt 2>/dev/null
    done
    C=$(cat $R/childpid 2>/dev/null)
    if [ -n "$C" ] && [ -r "/proc/$C/status" ]; then
        grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/$C/status >> $R/time_in_state_$1.txt 2>/dev/null
        echo "NICE $(awk '{print $19}' /proc/$C/stat 2>/dev/null)" >> $R/time_in_state_$1.txt
    fi
}
cpustat pre

snap pre
res pre

# The watcher snapshots /proc and calls gguf-residency on every stdout change.
# That call costs about 0.9 s, so with 7 model threads it saturates the last
# core and starves the DSP driver.  SKIP_WATCH=1 turns it off.
if [ "${SKIP_WATCH:-0}" = "1" ]; then
    : > $R/events.txt
    WATCH=""
else
(
    last=0
    while [ ! -f $R/stop ]; do
        cur=$(wc -c < $R/stdout.txt 2>/dev/null || echo 0)
        if [ "$cur" != "$last" ]; then
            nl=$(tail -c +$((last+1)) $R/stdout.txt 2>/dev/null)
            last=$cur
            echo "== EVENT $(cut -d' ' -f1 /proc/uptime)"
            echo "STDOUT $nl"
            snap ev
            res ev
        fi
        sleep 0.2
    done
) > $R/events.txt 2>&1 &
WATCH=$!
fi

T0=$(cut -d' ' -f1 /proc/uptime)
env LD_LIBRARY_PATH=$L ADSP_LIBRARY_PATH=$L \
    GGML_CPU_TILED_MM=${GGML_CPU_TILED_MM:-0} GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_NHVX=8 GGML_HEXAGON_OPPOLL=${OPPOLL:-1} \
    GGML_HEXAGON_MM_SELECT=1 GGML_HEXAGON_IQ2S_GATHER=1 GGML_HEXAGON_IQ3XXS_GATHER=1 \
    GGML_HEXAGON_IQ3S_GATHER=1 GGML_HEXAGON_FFN_OFFLOAD=1 GGML_HEXAGON_PROFILE=$PROFILE \
    GGML_HEXAGON_TIMELINE=${TIMELINE:-0} GGML_SCHED_DEBUG=${SCHED_DEBUG:-0} GGML_SCHED_SPLIT_TIMING=${SPLIT_TIMING:-0} \
    GGML_HEXAGON_OPBATCH=1280 P20_EARLY_SOURCE_RELEASE=${EARLY_REL:-1} P20_EARLY_SOURCE_TRACE=1 \
    P20_STREAM_HEAD_ROWS=0 P44_NTHREADS=$NTHREADS \
    $REPLAY_ENV $NOEXTRA P20_GUARD_CHILD_PID_FILE=$R/childpid \
    P27_RESUME_FILE=$R/resume \
    timeout -k 5 -s TERM $RUN_TIMEOUT $G /proc/vmstat $GUARD_PAGES 20 \
    $L/decode-probe-wait $M $OV $STEPS $R/logits.f32 mmap HTP0:0 ${NGL:-23} \
    > $R/stdout.txt 2> $R/stderr.txt &
GPID=$!

WAITED=0
while [ $WAITED -lt $WAIT_LOOP ]; do
    grep -q "PHASE warm_wait_begin" $R/stdout.txt 2>/dev/null && break
    kill -0 $GPID 2>/dev/null || break
    sleep 0.2
    WAITED=$((WAITED + 1))
done
echo "HANDSHAKE warm_wait_seen $WAITED ($(cut -d' ' -f1 /proc/uptime))" > $R/handshake.txt
snap load_end
res load_end
cpustat load_end

W0=$(cut -d' ' -f1 /proc/uptime)
sh $L/warmup-c43.sh $R > $R/warmup.log 2>&1
WRC=$?
W1=$(cut -d' ' -f1 /proc/uptime)
awk -v a=$W0 -v b=$W1 'BEGIN{printf "WARMUP wall_s %.2f\n", b-a}' >> $R/warmup.log
tail -3 $R/warmup.log >> $R/handshake.txt
snap after_warm
res after_warm

if [ $WRC -eq 0 ]; then
    echo "WARM_DONE $(date +%s)" > $R/resume
    echo "RELEASE ok $(cut -d' ' -f1 /proc/uptime)" >> $R/handshake.txt
else
    echo "WARM_FAIL rc $WRC" > $R/resume
    echo "RELEASE fail $(cut -d' ' -f1 /proc/uptime)" >> $R/handshake.txt
fi

wait $GPID
echo $? > $R/exit.txt
T1=$(cut -d' ' -f1 /proc/uptime)
awk -v a=$T0 -v b=$T1 'BEGIN{printf "RUN wall_s %.2f\n", b-a}' > $R/wall.txt

touch $R/stop
sleep 1
kill $WATCH 2>/dev/null
cpustat post
snap post
res post
sha256sum $L/libggml-htp-v81.so | sed 's#^#SKEL_POST #' >> $R/manifest.txt
echo "DONE $RUNID exit=$(cat $R/exit.txt) wall=$(cat $R/wall.txt)"
