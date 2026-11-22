#!/system/bin/sh
# Phase 61 - goal 2: does turning tiled on by default keep the replay hash?
#
# The brief says: if the replay hash changes when tiled is on by default, the
# default-on change must be withdrawn. This measures exactly that.
#
# The build here has the ARM default flipped to true in ggml_tiled_matmul_enabled.
# It is a GGML_OPENMP=OFF build so it matches the device's libggml-base.so.
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
SKEL=76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef
CPU_BASE=0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9
CPU_DEFON=ddb07e639fbe2a976a64f8ae6d0e7c2065fe156f182df50b93e0f6ff696dabbf

set_cpu() {
    cp "$1" $L/libggml-cpu.so || return 2
    chmod 644 $L/libggml-cpu.so
    GOT=$(sha256sum $L/libggml-cpu.so | cut -d' ' -f1)
    [ "$GOT" = "$2" ] || { echo "P61D REFUSING cpu $GOT != $2"; return 2; }
    echo "P61D_CPU $GOT"
}

one() {
    T=$1; NGL=$2; EXTRA=$3
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=1800 MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 $EXTRA \
        sh $BASE/p53_run.sh 32 replay $NGL $T 0 > $BASE/p61d2-$T.out 2>&1
    RC=$?
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p61d2-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    H=$(sha256sum $R/logits.f32 2>/dev/null | cut -d' ' -f1)
    SW=$(grep -oE "delta_pages [0-9-]+" $BASE/p61d2-$T.out | awk '{print $2}')
    echo "P61D_RUN tag $T ngl $NGL rc $RC exit ${EXIT:-NA} logits_sha ${H:-NA} swap_delta_pages ${SW:-NA} dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

cp $L/replay32.txt $L/replay.txt
set_cpu $L/libggml-cpu.so.c61defon $CPU_DEFON || exit 2

# no env at all: the build's own default decides
one defon-44 44 ""
one defon-0  0  ""
# and with the env explicitly off, to show the default is what changed
one defon-off-44 44 "GGML_CPU_TILED_MM=0"

cp $BASE/replay4.bak $L/replay.txt
set_cpu $L/libggml-cpu.so.c44bak $CPU_BASE
echo "P61D replay_restored $(cat $L/replay.txt)"
echo "P61D stray $(ps -A -o PID,NAME | awk '/decode-probe/{printf "%s ", $1}')"
