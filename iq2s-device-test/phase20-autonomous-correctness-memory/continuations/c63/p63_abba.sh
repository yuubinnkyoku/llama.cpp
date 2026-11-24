#!/system/bin/sh
# Phase 63 - ABBA of the integrated build against the pre-integration build.
#
#   A = pre-integration  cpu c62pre (17afd925) + stock hexagon + stock skel
#   B = integrated       cpu c62g1  (cc32ed77) + hexagon c62g2 (9c9bfc48) + skel c62q6 (312ad7b7)
#
# Both use the device's own libggml-base.so (d950aa96), which is what the
# integrated CPU was built against.
#
#   sh p63_abba.sh <tag> <rounds>
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
CPU_A=f4414f5ff4633e292db098749ae89b891a553328a346ba67de38f938a8ab44b3
CPU_B=cc32ed7733223fa4d832c80e0e2a9feba137055d66189dc1ac6f90a516f3f2b3
HEX_B=9c9bfc4874a00a2611caff34ad7b85ce56cf46d2787278f44552d228f7214a72
SKEL_B=312ad7b7ab304762f7c347e7909ee5d569541b15994073cfc11f55d0edf38114
HEX_STOCK=9c9bfc4874a00a2611caff34ad7b85ce56cf46d2787278f44552d228f7214a72
SKEL_STOCK=76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef
CPU_STOCK=0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9

TAG=${1:-abba}
ROUNDS=${2:-3}
TIMEOUT=${TIMEOUT:-1800}

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

put() {
    cp "$1" "$L/$2" || return 2
    chmod 644 "$L/$2"
    GOT=$(sha256sum "$L/$2" | cut -d' ' -f1)
    [ "$GOT" = "$3" ] || { echo "P63A REFUSING $2 $GOT != $3"; return 2; }
}

set_A() {
    put $L/libggml-cpu.so.c63fairA libggml-cpu.so $CPU_A || return 2
    put $L/libggml-hexagon.so.c62g2 libggml-hexagon.so $HEX_STOCK || return 2
    put $L/libggml-htp-v81.so.c58x4 libggml-htp-v81.so $SKEL_STOCK || return 2
    echo "P63A_ARM A cpu $CPU_A"
}

set_B() {
    put $L/libggml-cpu.so.c62g1 libggml-cpu.so $CPU_B || return 2
    put $L/libggml-hexagon.so.c62g2 libggml-hexagon.so $HEX_B || return 2
    put $L/libggml-htp-v81.so.c62q6 libggml-htp-v81.so $SKEL_B || return 2
    echo "P63A_ARM B cpu $CPU_B skel $SKEL_B"
}

pf() {
    T=$1; NGL=$2; NT=$3; SKEL=$4
    C0=$(cap); Z0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
        P53_PROMPT_TOKENS=$NT P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay $NGL $T 0 > $BASE/p63a-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); Z1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p63a-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    PFMS=$(grep -oE "PREFILL seg 0 tokens [0-9]+ ms [0-9.]+" $BASE/p63a-$T.out | awk '{print $7}')
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p63a-$T.out | awk '{print $5}')
    SW=$(grep -oE "delta_pages [0-9-]+" $BASE/p63a-$T.out | awk '{print $2}')
    CPUUS=$(grep -oE "ggml-sched-split id 0 backend CPU nodes [0-9]+ avg_us [0-9]+" $R/stderr.txt 2>/dev/null | awk '{print $9}' | sort -rn | head -1)
    SPLITS=$(grep -oE "graph: nodes = [0-9]+, splits = [0-9]+" $R/stderr.txt 2>/dev/null | tail -1 | grep -oE "splits = [0-9]+" | awk '{print $3}')
    echo "P63A_RUN tag $T ngl $NGL nt $NT rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) prefill_ms ${PFMS:-NA} decode_ms ${DEC:-NA} cpu_split_us ${CPUUS:-NA} splits ${SPLITS:-NA} swap_delta_pages ${SW:-NA} cap0 $C0-$C1 tz0 $Z0-$Z1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P63A KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

echo "P63A start tag $TAG rounds $ROUNDS uptime $(cut -d' ' -f1 /proc/uptime) cap0 $(cap) tz0 $(tz0)"
cp $L/replay32.txt $L/replay.txt

for P in $(seq 1 $ROUNDS); do
    if [ $((P % 2)) -eq 1 ]; then
        set_A || exit 2
        pf $TAG-p$P-1-A128 44 128 $SKEL_STOCK
        pf $TAG-p$P-1b-A512 44 512 $SKEL_STOCK
        set_B || exit 2
        pf $TAG-p$P-2-B128 44 128 $SKEL_B
        pf $TAG-p$P-2b-B512 44 512 $SKEL_B
    else
        set_B || exit 2
        pf $TAG-p$P-1-B128 44 128 $SKEL_B
        pf $TAG-p$P-1b-B512 44 512 $SKEL_B
        set_A || exit 2
        pf $TAG-p$P-2-A128 44 128 $SKEL_STOCK
        pf $TAG-p$P-2b-A512 44 512 $SKEL_STOCK
    fi
done

# restore the stock pair
put $L/libggml-cpu.so.c44bak libggml-cpu.so $CPU_STOCK || exit 2
put $L/libggml-hexagon.so.c62g2 libggml-hexagon.so $HEX_STOCK || exit 2
put $L/libggml-htp-v81.so.c58x4 libggml-htp-v81.so $SKEL_STOCK || exit 2
cp $BASE/replay4.bak $L/replay.txt
echo "P63A replay_restored $(cat $L/replay.txt)"
echo "P63A stray $(ps -A -o PID,NAME | awk '/decode-probe/{printf "%s ", $1}')"
echo "P63A_DONE cpu $(sha256sum $L/libggml-cpu.so | cut -d' ' -f1) skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
