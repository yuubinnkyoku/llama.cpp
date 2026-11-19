#!/system/bin/sh
# Phase 60 - run this when the device is back. Everything here was prepared on the
# host while the device was unreachable; none of it has been executed yet.
#
#   sh p60_tiled.sh <stage> [tag]
#
#   stage = gate   : regression gate on the current standard skel (c58x4)
#   stage = a      : Stage A, scalar microtile, GGML_CPU_TILED_MM=1
#   stage = b      : Stage B, NEON microtile, GGML_CPU_TILED_MM=1
#   stage = ab     : speed A/B, baseline cpu vs NEON+tiled, ABBA
#
# The two new cpu backends must already be on the device:
#   libs/libggml-cpu.so.c60scalar  eba4af55457b2ad19f3199237f4974229dfa2fa7ac99e6906138d4b15659e16a
#   libs/libggml-cpu.so.c60neon    c5499c18431c2d236713408d33e6b38de1a676bb214e85fea0c1334cec7f2db5
# and the baseline:
#   libs/libggml-cpu.so.c44bak     0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
STAGE=${1:-gate}
TAG=${2:-p60}
TIMEOUT=${TIMEOUT:-1800}
SKEL=76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef
CPU_BASE=0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9
CPU_SCALAR=eba4af55457b2ad19f3199237f4974229dfa2fa7ac99e6906138d4b15659e16a
CPU_NEON=c5499c18431c2d236713408d33e6b38de1a676bb214e85fea0c1334cec7f2db5

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

set_cpu() {
    cp "$1" $L/libggml-cpu.so || return 2
    chmod 644 $L/libggml-cpu.so
    GOT=$(sha256sum $L/libggml-cpu.so | cut -d' ' -f1)
    [ "$GOT" = "$2" ] || { echo "P60 REFUSING cpu $GOT != $2"; return 2; }
    echo "P60_CPU $GOT"
}

# one 32 step replay run, prints the logits hash
run32() {
    T=$1; NGL=$2; EXTRA=$3
    C0=$(cap); Z0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 $EXTRA \
        sh $BASE/p53_run.sh 32 replay $NGL $T 0 > $BASE/p60-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); Z1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p60-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    H=$(sha256sum $R/logits.f32 2>/dev/null | cut -d' ' -f1)
    SW=$(grep -oE "delta_pages [0-9-]+" $BASE/p60-$T.out | awk '{print $2}')
    echo "P60_RUN tag $T ngl $NGL rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) logits_sha ${H:-NA} swap_delta_pages ${SW:-NA} cap0 $C0-$C1 tz0 $Z0-$Z1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P60 KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

# one 128 token prefill + 4 decode, prints prefill and decode
run128() {
    T=$1; NGL=$2; EXTRA=$3
    C0=$(cap); Z0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 $EXTRA \
        P53_PROMPT_TOKENS=128 P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay $NGL $T 0 > $BASE/p60-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); Z1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p60-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    PF=$(grep -oE "PREFILL seg 0 tokens 128 ms [0-9.]+" $BASE/p60-$T.out | awk '{print $7}')
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p60-$T.out | awk '{print $5}')
    SW=$(grep -oE "delta_pages [0-9-]+" $BASE/p60-$T.out | awk '{print $2}')
    echo "P60_RUN128 tag $T ngl $NGL rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) prefill128_ms ${PF:-NA} decode_ms ${DEC:-NA} swap_delta_pages ${SW:-NA} cap0 $C0-$C1 tz0 $Z0-$Z1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P60 KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

echo "P60 start stage $STAGE tag $TAG skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1) cap0 $(cap) tz0 $(tz0) uptime $(cut -d' ' -f1 /proc/uptime)"

case "$STAGE" in
gate)
    set_cpu $L/libggml-cpu.so.c44bak $CPU_BASE || exit 2
    cp $L/replay32.txt $L/replay.txt
    run32 $TAG-gate44 44 ""
    run32 $TAG-gate0  0  ""
    cp $BASE/replay4.bak $L/replay.txt
    echo "P60 replay_restored $(cat $L/replay.txt)"
    ;;
a|b)
    if [ "$STAGE" = "a" ]; then set_cpu $L/libggml-cpu.so.c60scalar $CPU_SCALAR || exit 2
    else                        set_cpu $L/libggml-cpu.so.c60neon   $CPU_NEON   || exit 2; fi
    cp $L/replay32.txt $L/replay.txt
    # tiled on, cpu only: this is the correctness checkpoint
    run32 $TAG-$STAGE-tiled 0 "GGML_CPU_TILED_MM=1"
    # tiled off must reproduce the known cpu hash, proving the path was taken
    run32 $TAG-$STAGE-off   0 "GGML_CPU_TILED_MM=0"
    cp $BASE/replay4.bak $L/replay.txt
    echo "P60 replay_restored $(cat $L/replay.txt)"
    echo "P60 NOTE tiled-on hash is expected to differ from 44cd99ce; judge it with p50_nll.py"
    ;;
ab)
    cp $L/replay32.txt $L/replay.txt
    for P in 1 2; do
        if [ $((P % 2)) -eq 1 ]; then
            set_cpu $L/libggml-cpu.so.c44bak $CPU_BASE || exit 2
            run128 $TAG-p$P-1-base 0 ""
            set_cpu $L/libggml-cpu.so.c60neon $CPU_NEON || exit 2
            run128 $TAG-p$P-2-neon 0 "GGML_CPU_TILED_MM=1"
        else
            set_cpu $L/libggml-cpu.so.c60neon $CPU_NEON || exit 2
            run128 $TAG-p$P-1-neon 0 "GGML_CPU_TILED_MM=1"
            set_cpu $L/libggml-cpu.so.c44bak $CPU_BASE || exit 2
            run128 $TAG-p$P-2-base 0 ""
        fi
    done
    cp $BASE/replay4.bak $L/replay.txt
    set_cpu $L/libggml-cpu.so.c44bak $CPU_BASE
    echo "P60 replay_restored $(cat $L/replay.txt)"
    ;;
*)
    echo "P60 REFUSING unknown stage $STAGE"
    exit 2
    ;;
esac

echo "P60 stray $(ps -A -o PID,NAME | awk '/decode-probe/{printf "%s ", $1}')"
echo "P60_DONE cpu $(sha256sum $L/libggml-cpu.so | cut -d' ' -f1) skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
