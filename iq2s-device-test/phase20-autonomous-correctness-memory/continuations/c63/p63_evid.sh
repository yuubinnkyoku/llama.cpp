#!/system/bin/sh
# Phase 63 - evidence test: run the 27B on a pristine upstream build.
#
# The point is to measure what stock llama.cpp can do with this model, with no
# fork code in the path. The upstream set lives in libs/up63/ and is swapped in
# for the duration of the run, then the stock pair is restored.
#
#   sh p63_evid.sh <tag> <ngl> [prompt_tokens]
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
U=$L/up63
SKEL_UP=41bfebd9a687adb453c0cf219410e3f0e3a304950e3f95abbfb687d54fe2f401
CPU_UP=e655bf5ca3ac53b4d18548dfa648c0ce4431b049251c018eb6b02fa75f04ca93
BASE_UP=4671a16bd39ce103e2f70c9ff5a3b71b91dece8954b092feb680ec5cf22a4301
HEX_UP=f40c30eb5382882fc588c9076d8a452e6a6eef25c13cdbd1aaeee08894b60f2c
CPU_STOCK=0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9
BASE_STOCK=d950aa96587fe4bceb27cb1d88f27a3ccbc2d19f951d6a4f89b95b2f28893a27
SKEL_STOCK=76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef
HEX_STOCK=9c9bfc4874a00a2611caff34ad7b85ce56cf46d2787278f44552d228f7214a72

TAG=${1:-evid}
NGL=${2:-44}
NT=${3:-128}

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

put() {  # $1 = src, $2 = dest name, $3 = expected sha
    cp "$1" "$L/$2" || return 2
    chmod 644 "$L/$2"
    GOT=$(sha256sum "$L/$2" | cut -d' ' -f1)
    [ "$GOT" = "$3" ] || { echo "P63E REFUSING $2 $GOT != $3"; return 2; }
    echo "P63E_LIB $2 $GOT"
}

echo "P63E start tag $TAG ngl $NGL nt $NT uptime $(cut -d' ' -f1 /proc/uptime) cap0 $(cap) tz0 $(tz0)"
echo "P63E mem_before $(grep MemAvailable /proc/meminfo | awk '{print $2}') swapfree $(grep SwapFree /proc/meminfo | awk '{print $2}')"

# swap in the pristine upstream set
put $U/libggml-base.so    libggml-base.so    $BASE_UP || exit 2
put $U/libggml-cpu.so     libggml-cpu.so     $CPU_UP  || exit 2
put $U/libggml-hexagon.so libggml-hexagon.so $HEX_UP  || exit 2
put $U/libggml-htp-v81.so libggml-htp-v81.so $SKEL_UP || exit 2

cp $L/replay32.txt $L/replay.txt

C0=$(cap); Z0=$(tz0); S0=$(date +%s)
env WANT_SKEL=$SKEL_UP GUARD_PAGES=2097152 RUN_TIMEOUT=1800 MEM_MIN_KB=4000000 \
    WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
    P53_PROMPT_TOKENS=$NT P53_NO_LOGITS=1 \
    sh $BASE/p53_run.sh 4 noreplay $NGL $TAG 0 > $BASE/p63e-$TAG.out 2>&1
RC=$?
S1=$(date +%s); C1=$(cap); Z1=$(tz0)

R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p63e-$TAG.out 2>/dev/null | awk '{print $NF}')
[ -n "$R" ] || R=$(ls -td $BASE/runs/*/ | head -1)
EXIT=$(cat $R/exit.txt 2>/dev/null)
PF=$(grep -oE "PREFILL seg 0 tokens [0-9]+ ms [0-9.]+" $BASE/p63e-$TAG.out | awk '{print $7}')
DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p63e-$TAG.out | awk '{print $5}')
SW=$(grep -oE "delta_pages [0-9-]+" $BASE/p63e-$TAG.out | awk '{print $2}')
echo "P63E_RUN tag $TAG ngl $NGL nt $NT rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) prefill_ms ${PF:-NA} decode_ms ${DEC:-NA} swap_delta_pages ${SW:-NA} cap0 $C0-$C1 tz0 $Z0-$Z1 dir $R"

# what the backend said about the graph: split count and per-backend node counts
echo "P63E_SPLITS $(grep -oE 'graph: nodes = [0-9]+, splits = [0-9]+' $R/stderr.txt 2>/dev/null | tail -1)"
echo "P63E_SCHED $(grep -oE 'ggml-sched-split id [0-9]+ backend [A-Za-z0-9:]+ nodes [0-9]+' $R/stderr.txt 2>/dev/null | tr '\n' '|')"
echo "P63E_ERRTAIL $(tail -6 $R/stderr.txt 2>/dev/null | tr '\n' '|')"
echo "P63E_OUTTAIL $(tail -6 $R/stdout.txt 2>/dev/null | tr '\n' '|')"

if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
    for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
        echo "P63E KILL orphan $p"
        kill -9 $p 2>/dev/null
    done
    sleep 2
fi

# restore the stock pair
put $L/libggml-base.so.orig libggml-base.so $BASE_STOCK || exit 2
put $L/libggml-cpu.so.c44bak libggml-cpu.so  $CPU_STOCK  || exit 2
put $L/libggml-hexagon.so.c62g2 libggml-hexagon.so $HEX_STOCK || exit 2
put $L/libggml-htp-v81.so.c58x4 libggml-htp-v81.so $SKEL_STOCK || exit 2

cp $BASE/replay4.bak $L/replay.txt
echo "P63E mem_after $(grep MemAvailable /proc/meminfo | awk '{print $2}') swapfree $(grep SwapFree /proc/meminfo | awk '{print $2}')"
echo "P63E replay_restored $(cat $L/replay.txt)"
echo "P63E stray $(ps -A -o PID,NAME | awk '/decode-probe/{printf "%s ", $1}')"
echo "P63E_DONE cpu $(sha256sum $L/libggml-cpu.so | cut -d' ' -f1) base $(sha256sum $L/libggml-base.so | cut -d' ' -f1) skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
