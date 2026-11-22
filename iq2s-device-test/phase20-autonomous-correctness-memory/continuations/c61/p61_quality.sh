#!/system/bin/sh
# Phase 61 - goal 2: real-batch quality, tiled on against tiled off.
#
# The 32-step replay is one token per step, so tiled never engages there and the
# hash is identical either way. A 128-token prefill does engage it (n_ubatch 32
# gives rows >= 8), so the logits of that prefill are the real-batch evidence.
#
# Both runs use the same cpu binary (c60neon2) so the only difference is the env
# switch, which keeps the wdata layout identical and makes the comparison clean.
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
SKEL=76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef
CPU_BASE=0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9
CPU_NEON=318508e89c15c0191ad8e2335e6b0da4034db6493dd7bd1906bee236be1b77b9

set_cpu() {
    cp "$1" $L/libggml-cpu.so || return 2
    chmod 644 $L/libggml-cpu.so
    GOT=$(sha256sum $L/libggml-cpu.so | cut -d' ' -f1)
    [ "$GOT" = "$2" ] || { echo "P61Q REFUSING cpu $GOT != $2"; return 2; }
    echo "P61Q_CPU $GOT"
}

# prefill 128 with logits kept, so the output can be compared
runq() {
    T=$1; EXTRA=$2
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=1800 MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 $EXTRA \
        P53_PROMPT_TOKENS=128 \
        sh $BASE/p53_run.sh 4 noreplay 0 $T 0 > $BASE/p61q-$T.out 2>&1
    RC=$?
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p61q-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    H=$(sha256sum $R/logits.f32 2>/dev/null | cut -d' ' -f1)
    PF=$(grep -oE "PREFILL seg 0 tokens [0-9]+ ms [0-9.]+" $BASE/p61q-$T.out | awk '{print $7}')
    echo "P61Q_RUN tag $T rc $RC exit ${EXIT:-NA} prefill_ms ${PF:-NA} logits_sha ${H:-NA} dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

cp $L/replay32.txt $L/replay.txt
set_cpu $L/libggml-cpu.so.c60neon2 $CPU_NEON || exit 2

# tiled off first, then on, then off again: a repeat of the off run proves the
# measurement is stable, so a difference in the on run is the kernel and not noise
runq q-off1 "GGML_CPU_TILED_MM=0"
runq q-on1  "GGML_CPU_TILED_MM=1"
runq q-off2 "GGML_CPU_TILED_MM=0"

cp $BASE/replay4.bak $L/replay.txt
set_cpu $L/libggml-cpu.so.c44bak $CPU_BASE
echo "P61Q replay_restored $(cat $L/replay.txt)"
echo "P61Q stray $(ps -A -o PID,NAME | awk '/decode-probe/{printf "%s ", $1}')"
