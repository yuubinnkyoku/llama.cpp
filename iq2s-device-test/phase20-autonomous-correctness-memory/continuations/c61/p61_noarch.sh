#!/system/bin/sh
# Phase 61 - does the no-march build reproduce the baseline hash?
#
# The baseline 0a3db9c5 was built from build/p53-android, which has an EMPTY
# GGML_CPU_ARM_ARCH, so no -march flag is passed at all. The c60/c61 builds all
# pass -march=armv8-a or armv8.6-a+dotprod+i8mm. This checks whether that is
# what moves the replay hash away from e9da92f8.
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
SKEL=76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef
CPU_BASE=0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9
CPU_NOARCH=526338779e2c50b33c99d51e82d974f193135716b9a1363c21bacba8e1c124f2

set_cpu() {
    cp "$1" $L/libggml-cpu.so || return 2
    chmod 644 $L/libggml-cpu.so
    GOT=$(sha256sum $L/libggml-cpu.so | cut -d' ' -f1)
    [ "$GOT" = "$2" ] || { echo "P61N REFUSING cpu $GOT != $2"; return 2; }
    echo "P61N_CPU $GOT"
}

one() {
    T=$1; NGL=$2; EXTRA=$3
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=1800 MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 $EXTRA \
        sh $BASE/p53_run.sh 32 replay $NGL $T 0 > $BASE/p61n-$T.out 2>&1
    RC=$?
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p61n-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    H=$(sha256sum $R/logits.f32 2>/dev/null | cut -d' ' -f1)
    SW=$(grep -oE "delta_pages [0-9-]+" $BASE/p61n-$T.out | awk '{print $2}')
    echo "P61N_RUN tag $T ngl $NGL rc $RC exit ${EXIT:-NA} logits_sha ${H:-NA} swap_delta_pages ${SW:-NA} dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

cp $L/replay32.txt $L/replay.txt

# no-march build, tiled off: must reproduce e9da92f8 if -march was the cause
set_cpu $L/libggml-cpu.so.c61noarch $CPU_NOARCH || exit 2
one noarch-off 44 "GGML_CPU_TILED_MM=0"

# and with tiled on but not forced (rows < 8, so tiled must not engage)
set_cpu $L/libggml-cpu.so.c61noarch $CPU_NOARCH || exit 2
one noarch-on 44 "GGML_CPU_TILED_MM=1"

cp $BASE/replay4.bak $L/replay.txt
set_cpu $L/libggml-cpu.so.c44bak $CPU_BASE
echo "P61N replay_restored $(cat $L/replay.txt)"
echo "P61N stray $(ps -A -o PID,NAME | awk '/decode-probe/{printf "%s ", $1}')"
