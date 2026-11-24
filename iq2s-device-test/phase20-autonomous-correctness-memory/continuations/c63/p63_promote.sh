#!/system/bin/sh
# Phase 63 - promote the integrated build to the standard set.
#
# Before: cpu 0a3db9c5, base d950aa96, hexagon 9c9bfc48, skel 76dd89e3
# After:  cpu 1cc02638 (tiled default on), base d950aa96, hexagon 9c9bfc48, skel 312ad7b7
#
# The base library stays: the integrated CPU was built against it and it has no
# libomp dependency, which is what the device needs.
#
#   sh p63_promote.sh <before|after|verify>
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
CPU_NEW=1cc02638355148a78e0423036169f9ae8f480c44d5eeebf0cc9e3ed3506c9150
SKEL_NEW=312ad7b7ab304762f7c347e7909ee5d569541b15994073cfc11f55d0edf38114
HEX_NEW=9c9bfc4874a00a2611caff34ad7b85ce56cf46d2787278f44552d228f7214a72
BASE_STD=d950aa96587fe4bceb27cb1d88f27a3ccbc2d19f951d6a4f89b95b2f28893a27
CPU_OLD=0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9
SKEL_OLD=76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef

show() {
    echo "P63P_STATE cpu $(sha256sum $L/libggml-cpu.so | cut -d' ' -f1)"
    echo "P63P_STATE base $(sha256sum $L/libggml-base.so | cut -d' ' -f1)"
    echo "P63P_STATE hexagon $(sha256sum $L/libggml-hexagon.so | cut -d' ' -f1)"
    echo "P63P_STATE skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
    echo "P63P_STATE replay $(cat $L/replay.txt)"
}

put() {
    cp "$1" "$L/$2" || return 2
    chmod 644 "$L/$2"
    GOT=$(sha256sum "$L/$2" | cut -d' ' -f1)
    [ "$GOT" = "$3" ] || { echo "P63P REFUSING $2 $GOT != $3"; return 2; }
    echo "P63P_LIB $2 $GOT"
}

case "${1:-verify}" in
before)
    echo "P63P before"
    show
    ;;
after)
    echo "P63P promoting"
    # keep the outgoing pair under explicit names so a rollback is one cp
    [ -f $L/libggml-cpu.so.c63prev ] || cp $L/libggml-cpu.so $L/libggml-cpu.so.c63prev
    [ -f $L/libggml-htp-v81.so.c63prev ] || cp $L/libggml-htp-v81.so $L/libggml-htp-v81.so.c63prev
    put $L/libggml-cpu.so.c63defon libggml-cpu.so $CPU_NEW || exit 2
    put $L/libggml-htp-v81.so.c62q6 libggml-htp-v81.so $SKEL_NEW || exit 2
    put $L/libggml-hexagon.so.c62g2 libggml-hexagon.so $HEX_NEW || exit 2
    put $L/libggml-base.so.orig libggml-base.so $BASE_STD || exit 2
    echo "P63P after"
    show
    ;;
verify)
    echo "P63P verify"
    show
    CPU=$(sha256sum $L/libggml-cpu.so | cut -d' ' -f1)
    SKEL=$(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)
    [ "$CPU" = "$CPU_NEW" ] && echo "P63P_OK cpu integrated" || echo "P63P_NOTE cpu $CPU"
    [ "$SKEL" = "$SKEL_NEW" ] && echo "P63P_OK skel integrated" || echo "P63P_NOTE skel $SKEL"
    ;;
*)
    echo "P63P REFUSING unknown action $1"
    exit 2
    ;;
esac
