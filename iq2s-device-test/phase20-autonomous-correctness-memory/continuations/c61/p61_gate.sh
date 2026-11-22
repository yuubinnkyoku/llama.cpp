#!/system/bin/sh
# Phase 61 - is the ARM gate in tiled.cpp what moves the replay hash?
#
# The baseline 0a3db9c5 has ggml_compute_forward_mul_mat_tiled compiled to
# "mov w0, wzr; ret" (always false), so tiled never runs and ggml_tiled_wdata_size
# returns 0. That zero is reserved from params->wdata, so with the gate present
# every downstream buffer in the work area shifts by 64 + nth*512KB. This build
# has the gate reverted to the pre-Phase-60 form, so wdata_size is 0 again.
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
SKEL=76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef
CPU_BASE=0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9
CPU_NOGATE=5012719033b95fedf8fce2031b23522b1c9649354e4dbb08125e998b72dbd4f7

set_cpu() {
    cp "$1" $L/libggml-cpu.so || return 2
    chmod 644 $L/libggml-cpu.so
    GOT=$(sha256sum $L/libggml-cpu.so | cut -d' ' -f1)
    [ "$GOT" = "$2" ] || { echo "P61G REFUSING cpu $GOT != $2"; return 2; }
    echo "P61G_CPU $GOT"
}

one() {
    T=$1; NGL=$2; EXTRA=$3
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=1800 MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 $EXTRA \
        sh $BASE/p53_run.sh 32 replay $NGL $T 0 > $BASE/p61g2-$T.out 2>&1
    RC=$?
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p61g2-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    H=$(sha256sum $R/logits.f32 2>/dev/null | cut -d' ' -f1)
    SW=$(grep -oE "delta_pages [0-9-]+" $BASE/p61g2-$T.out | awk '{print $2}')
    echo "P61G_RUN tag $T ngl $NGL rc $RC exit ${EXIT:-NA} logits_sha ${H:-NA} swap_delta_pages ${SW:-NA} dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

cp $L/replay32.txt $L/replay.txt

# gate reverted: must reproduce e9da92f8 if the wdata reservation was the cause
set_cpu $L/libggml-cpu.so.c61nogate $CPU_NOGATE || exit 2
one nogate-44 44 ""
one nogate-0  0  ""

cp $BASE/replay4.bak $L/replay.txt
set_cpu $L/libggml-cpu.so.c44bak $CPU_BASE
echo "P61G replay_restored $(cat $L/replay.txt)"
echo "P61G stray $(ps -A -o PID,NAME | awk '/decode-probe/{printf "%s ", $1}')"
