#!/system/bin/sh
# Phase 61 - end-to-end effect of the tiled NEON kernel, and the NGL re-sweep.
#
#   sh p61_e2e.sh <stage> [tag]
#
#   stage = unit   : the tiled matmul unit test (no model, seconds)
#   stage = gate   : regression gate on the current standard skel (c58x4)
#   stage = e2e    : end-to-end A/B at NGL=44, ABBA, prefill 128 and 512
#   stage = ng    : NGL sweep with tiled on, prefill 128
#   stage = cpu    : CPU only prefill (NGL=0) quality check
#   stage = dec    : decode spread, NGL 44 vs 50, interleaved
#
# The two cpu backends must already be on the device:
#   libs/libggml-cpu.so.c44bak   0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9
#   libs/libggml-cpu.so.c60neon2 318508e89c15c0191ad8e2335e6b0da4034db6493dd7bd1906bee236be1b77b9
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
STAGE=${1:-gate}
TAG=${2:-p61}
TIMEOUT=${TIMEOUT:-1800}
SKEL=76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef
CPU_BASE=0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9
CPU_NEON=318508e89c15c0191ad8e2335e6b0da4034db6493dd7bd1906bee236be1b77b9

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

set_cpu() {
    cp "$1" $L/libggml-cpu.so || return 2
    chmod 644 $L/libggml-cpu.so
    GOT=$(sha256sum $L/libggml-cpu.so | cut -d' ' -f1)
    [ "$GOT" = "$2" ] || { echo "P61 REFUSING cpu $GOT != $2"; return 2; }
    echo "P61_CPU $GOT"
}

# one 32 step replay run, prints the logits hash
run32() {
    T=$1; NGL=$2; EXTRA=$3
    C0=$(cap); Z0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 $EXTRA \
        sh $BASE/p53_run.sh 32 replay $NGL $T 0 > $BASE/p61-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); Z1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p61-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    H=$(sha256sum $R/logits.f32 2>/dev/null | cut -d' ' -f1)
    SW=$(grep -oE "delta_pages [0-9-]+" $BASE/p61-$T.out | awk '{print $2}')
    echo "P61_RUN tag $T ngl $NGL rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) logits_sha ${H:-NA} swap_delta_pages ${SW:-NA} cap0 $C0-$C1 tz0 $Z0-$Z1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P61 KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

# one prefill + 4 decode. NT = prompt tokens (128 or 512)
runpf() {
    T=$1; NGL=$2; EXTRA=$3; NT=$4
    C0=$(cap); Z0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 $EXTRA \
        P53_PROMPT_TOKENS=$NT P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay $NGL $T 0 > $BASE/p61-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); Z1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p61-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    PF=$(grep -oE "PREFILL seg 0 tokens [0-9]+ ms [0-9.]+" $BASE/p61-$T.out | awk '{print $7}')
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p61-$T.out | awk '{print $5}')
    SW=$(grep -oE "delta_pages [0-9-]+" $BASE/p61-$T.out | awk '{print $2}')
    # wall breakdown: the CPU split time is the host side, the rest is HTP
    CPUUS=$(grep -oE "ggml-sched-split id 0 backend CPU nodes [0-9]+ avg_us [0-9]+" $R/stderr.txt 2>/dev/null | awk '{print $9}' | sort -rn | head -1)
    echo "P61_PF tag $T ngl $NGL nt $NT rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) prefill_ms ${PF:-NA} decode_ms ${DEC:-NA} cpu_split_us ${CPUUS:-NA} swap_delta_pages ${SW:-NA} cap0 $C0-$C1 tz0 $Z0-$Z1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P61 KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

echo "P61 start stage $STAGE tag $TAG skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1) cap0 $(cap) tz0 $(tz0) uptime $(cut -d' ' -f1 /proc/uptime)"

case "$STAGE" in
unit)
    U=$BASE/c60unit
    if [ ! -x "$U/test-tiled-mulmat" ]; then
        echo "P61_UNIT missing $U/test-tiled-mulmat"
        exit 2
    fi
    echo "P61_UNIT start cap0 $(cap) tz0 $(tz0)"
    LD_LIBRARY_PATH=$U ADSP_LIBRARY_PATH=$U "$U/test-tiled-mulmat" > $BASE/p61-unit.log 2>&1
    RC=$?
    echo "P61_UNIT rc $RC pass $(grep -c PASS $BASE/p61-unit.log) fail $(grep -c FAIL $BASE/p61-unit.log)"
    grep -E 'FAIL' $BASE/p61-unit.log | head -10
    LD_LIBRARY_PATH=$U ADSP_LIBRARY_PATH=$U "$U/test-tiled-mulmat" --fuzz > $BASE/p61-unit-fuzz.log 2>&1
    RC2=$?
    echo "P61_UNIT fuzz rc $RC2 pass $(grep -c PASS $BASE/p61-unit-fuzz.log) fail $(grep -c FAIL $BASE/p61-unit-fuzz.log)"
    grep -E 'FAIL' $BASE/p61-unit-fuzz.log | head -10
    ;;
gate)
    set_cpu $L/libggml-cpu.so.c44bak $CPU_BASE || exit 2
    cp $L/replay32.txt $L/replay.txt
    run32 $TAG-gate44 44 ""
    run32 $TAG-gate0  0  ""
    # the NEON build with tiled off must reproduce the same hash: the kernel swap
    # must not touch the replay path
    set_cpu $L/libggml-cpu.so.c60neon2 $CPU_NEON || exit 2
    run32 $TAG-gate44neon 44 "GGML_CPU_TILED_MM=0"
    cp $BASE/replay4.bak $L/replay.txt
    set_cpu $L/libggml-cpu.so.c44bak $CPU_BASE
    echo "P61 replay_restored $(cat $L/replay.txt)"
    ;;
e2e)
    cp $L/replay32.txt $L/replay.txt
    for P in 1 2 3; do
        if [ $((P % 2)) -eq 1 ]; then
            set_cpu $L/libggml-cpu.so.c44bak $CPU_BASE || exit 2
            runpf $TAG-p$P-1-base 44 "" 128
            runpf $TAG-p$P-1b-base512 44 "" 512
            set_cpu $L/libggml-cpu.so.c60neon2 $CPU_NEON || exit 2
            runpf $TAG-p$P-2-neon 44 "GGML_CPU_TILED_MM=1" 128
            runpf $TAG-p$P-2b-neon512 44 "GGML_CPU_TILED_MM=1" 512
        else
            set_cpu $L/libggml-cpu.so.c60neon2 $CPU_NEON || exit 2
            runpf $TAG-p$P-1-neon 44 "GGML_CPU_TILED_MM=1" 128
            runpf $TAG-p$P-1b-neon512 44 "GGML_CPU_TILED_MM=1" 512
            set_cpu $L/libggml-cpu.so.c44bak $CPU_BASE || exit 2
            runpf $TAG-p$P-2-base 44 "" 128
            runpf $TAG-p$P-2b-base512 44 "" 512
        fi
    done
    cp $BASE/replay4.bak $L/replay.txt
    set_cpu $L/libggml-cpu.so.c44bak $CPU_BASE
    echo "P61 replay_restored $(cat $L/replay.txt)"
    ;;
ng)
    set_cpu $L/libggml-cpu.so.c60neon2 $CPU_NEON || exit 2
    cp $L/replay32.txt $L/replay.txt
    for N in 16 32 40 44 48 50 56; do
        runpf $TAG-ngl$N $N "GGML_CPU_TILED_MM=1" 128
    done
    cp $BASE/replay4.bak $L/replay.txt
    set_cpu $L/libggml-cpu.so.c44bak $CPU_BASE
    echo "P61 replay_restored $(cat $L/replay.txt)"
    ;;
cpu)
    set_cpu $L/libggml-cpu.so.c60neon2 $CPU_NEON || exit 2
    cp $L/replay32.txt $L/replay.txt
    # natural path: n_ubatch 32 gives rows >= 8, so tiled engages without FORCE
    run32 $TAG-cpu-tiled 0 "GGML_CPU_TILED_MM=1"
    run32 $TAG-cpu-off   0 "GGML_CPU_TILED_MM=0"
    cp $BASE/replay4.bak $L/replay.txt
    set_cpu $L/libggml-cpu.so.c44bak $CPU_BASE
    echo "P61 replay_restored $(cat $L/replay.txt)"
    ;;
dec)
    set_cpu $L/libggml-cpu.so.c44bak $CPU_BASE || exit 2
    cp $L/replay32.txt $L/replay.txt
    for P in 1 2 3 4 5 6; do
        runpf $TAG-dec44-$P 44 "" 128
        runpf $TAG-dec50-$P 50 "" 128
    done
    cp $BASE/replay4.bak $L/replay.txt
    echo "P61 replay_restored $(cat $L/replay.txt)"
    ;;
*)
    echo "P61 REFUSING unknown stage $STAGE"
    exit 2
    ;;
esac

echo "P61 stray $(ps -A -o PID,NAME | awk '/decode-probe/{printf "%s ", $1}')"
echo "P61_DONE cpu $(sha256sum $L/libggml-cpu.so | cut -d' ' -f1) skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
