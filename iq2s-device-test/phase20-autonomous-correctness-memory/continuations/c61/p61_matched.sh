#!/system/bin/sh
# Phase 61 - test with a fully matched library set.
#
# The device's libggml-base.so (d950aa96, 1384 exported symbols) is from an older
# source tree than the current one (1536 symbols). Every libggml-cpu.so built
# from the current tree therefore runs against a base library it was not built
# with. This stage swaps in a matched base+cpu pair and checks whether the
# replay hash returns to e9da92f8.
#
# run53.sh sets LD_LIBRARY_PATH=$L itself, so the swap has to be done in place.
# The device's original base is backed up first and restored at the end.
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
M=$L/c61m
SKEL=76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef
CPU_BASE=0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9
BASE_ORIG=d950aa96587fe4bceb27cb1d88f27a3ccbc2d19f951d6a4f89b95b2f28893a27
CPU_MNEON=17afd925dc67d5d4f9b17f69661666517a0b84d8a586974de070ae50751c1ed8
BASE_MNEON=a410309aead3a6c0713aa86ffbdcc313ef943fb3439fbc151c89e4069b36c91a

set_lib() {  # $1 = source, $2 = dest name, $3 = expected sha
    cp "$1" "$L/$2" || return 2
    chmod 644 "$L/$2"
    GOT=$(sha256sum "$L/$2" | cut -d' ' -f1)
    [ "$GOT" = "$3" ] || { echo "P61M REFUSING $2 $GOT != $3"; return 2; }
    echo "P61M_LIB $2 $GOT"
}

one() {
    T=$1; NGL=$2; EXTRA=$3
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=1800 MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 $EXTRA \
        sh $BASE/p53_run.sh 32 replay $NGL $T 0 > $BASE/p61m-$T.out 2>&1
    RC=$?
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p61m-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    H=$(sha256sum $R/logits.f32 2>/dev/null | cut -d' ' -f1)
    SW=$(grep -oE "delta_pages [0-9-]+" $BASE/p61m-$T.out | awk '{print $2}')
    echo "P61M_RUN tag $T ngl $NGL rc $RC exit ${EXIT:-NA} logits_sha ${H:-NA} swap_delta_pages ${SW:-NA} dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

# back up the device's original base once
[ -f $L/libggml-base.so.orig ] || cp $L/libggml-base.so $L/libggml-base.so.orig

cp $L/replay32.txt $L/replay.txt

# 1. matched base+cpu, tiled off: does the hash return to e9da92f8?
set_lib $M/libggml-base.so libggml-base.so $BASE_MNEON || exit 2
set_lib $M/libggml-cpu.so  libggml-cpu.so  $CPU_MNEON  || exit 2
one mneon-off 44 "GGML_CPU_TILED_MM=0"

# 2. matched base+cpu, tiled on but not forced (rows < 8, tiled must not engage)
one mneon-on 44 "GGML_CPU_TILED_MM=1"

# restore the device's original libraries
set_lib $L/libggml-base.so.orig libggml-base.so $BASE_ORIG || exit 2
set_lib $L/libggml-cpu.so.c44bak libggml-cpu.so $CPU_BASE || exit 2

cp $BASE/replay4.bak $L/replay.txt
echo "P61M replay_restored $(cat $L/replay.txt)"
echo "P61M stray $(ps -A -o PID,NAME | awk '/decode-probe/{printf "%s ", $1}')"
echo "P61M_DONE cpu $(sha256sum $L/libggml-cpu.so | cut -d' ' -f1) base $(sha256sum $L/libggml-base.so | cut -d' ' -f1)"
