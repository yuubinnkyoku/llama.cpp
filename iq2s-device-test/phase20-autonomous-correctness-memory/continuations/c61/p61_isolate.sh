#!/system/bin/sh
# Phase 61 - isolate why the NEON build with tiled off does not reproduce e9da92f8.
#
# Two candidate causes:
#   (a) the dotprod/i8mm arch flags change the stock vec_dot code
#   (b) the tiled gate change in tiled.cpp alters the stock path
#
# The scalar build (armv8-a, no dotprod) has the same gate change as the NEON
# build, so:
#   scalar + tiled off == e9da92f8  -> the gate change is innocent, dotprod is the cause
#   scalar + tiled off != e9da92f8  -> the gate change itself moved the hash
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
SKEL=76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef
CPU_BASE=0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9
CPU_SCALAR=238b09835974b10a45b014289993cc6bec49adda82de4dae77201f69e5173f63
CPU_NEON=318508e89c15c0191ad8e2335e6b0da4034db6493dd7bd1906bee236be1b77b9

set_cpu() {
    cp "$1" $L/libggml-cpu.so || return 2
    chmod 644 $L/libggml-cpu.so
    GOT=$(sha256sum $L/libggml-cpu.so | cut -d' ' -f1)
    [ "$GOT" = "$2" ] || { echo "P61I REFUSING cpu $GOT != $2"; return 2; }
    echo "P61I_CPU $GOT"
}

one() {
    T=$1; NGL=$2; EXTRA=$3
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=1800 MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 $EXTRA \
        sh $BASE/p53_run.sh 32 replay $NGL $T 0 > $BASE/p61i-$T.out 2>&1
    RC=$?
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p61i-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    H=$(sha256sum $R/logits.f32 2>/dev/null | cut -d' ' -f1)
    SW=$(grep -oE "delta_pages [0-9-]+" $BASE/p61i-$T.out | awk '{print $2}')
    echo "P61I_RUN tag $T ngl $NGL rc $RC exit ${EXIT:-NA} logits_sha ${H:-NA} swap_delta_pages ${SW:-NA} dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

cp $L/replay32.txt $L/replay.txt

# 1. scalar build (armv8-a, no dotprod) with tiled off
set_cpu $L/libggml-cpu.so.c60scalar $CPU_SCALAR || exit 2
one scalar-off 44 "GGML_CPU_TILED_MM=0"

# 2. scalar build with tiled on but not forced: rows < 8 so tiled must not engage
set_cpu $L/libggml-cpu.so.c60scalar $CPU_SCALAR || exit 2
one scalar-on 44 "GGML_CPU_TILED_MM=1"

# 3. NEON build with tiled off, NGL=0 as well (the other known hash)
set_cpu $L/libggml-cpu.so.c60neon2 $CPU_NEON || exit 2
one neon-off-ngl0 0 "GGML_CPU_TILED_MM=0"

cp $BASE/replay4.bak $L/replay.txt
set_cpu $L/libggml-cpu.so.c44bak $CPU_BASE
echo "P61I replay_restored $(cat $L/replay.txt)"
echo "P61I stray $(ps -A -o PID,NAME | awk '/decode-probe/{printf "%s ", $1}')"
