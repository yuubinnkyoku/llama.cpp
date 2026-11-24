#!/system/bin/sh
# Phase 63 - isolate the ABBA regression. B is 10-14% slower than A; find which half.
#
#   A = cpu c62pre (17afd925) + skel c58x4 (76dd89e3)   <- pre-integration
#   B = cpu c62g1  (cc32ed77) + skel c62q6 (312ad7b7)   <- integrated
#
# Two mixed arms split the difference:
#   C = cpu c62pre + skel c62q6   (skel only)
#   D = cpu c62g1  + skel c58x4   (cpu only)
#
#   sh p63_iso.sh <tag> <rounds>
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
CPU_A=17afd925dc67d5d4f9b17f69661666517a0b84d8a586974de070ae50751c1ed8
CPU_B=cc32ed7733223fa4d832c80e0e2a9feba137055d66189dc1ac6f90a516f3f2b3
SKEL_A=76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef
SKEL_B=312ad7b7ab304762f7c347e7909ee5d569541b15994073cfc11f55d0edf38114
CPU_STOCK=0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9

TAG=${1:-iso}
ROUNDS=${2:-2}
TIMEOUT=${TIMEOUT:-1800}

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

put() {
    cp "$1" "$L/$2" || return 2
    chmod 644 "$L/$2"
    GOT=$(sha256sum "$L/$2" | cut -d' ' -f1)
    [ "$GOT" = "$3" ] || { echo "P63I REFUSING $2 $GOT != $3"; return 2; }
}

arm() {  # $1 = cpu file, $2 = cpu sha, $3 = skel file, $4 = skel sha, $5 = label
    put "$L/$1" libggml-cpu.so "$2" || return 2
    put "$L/$3" libggml-htp-v81.so "$4" || return 2
    echo "P63I_ARM $5 cpu $2 skel $4"
}

pf() {
    T=$1; NT=$2; SKEL=$3
    C0=$(cap); Z0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
        P53_PROMPT_TOKENS=$NT P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay 44 $T 0 > $BASE/p63i-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); Z1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p63i-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    PFMS=$(grep -oE "PREFILL seg 0 tokens [0-9]+ ms [0-9.]+" $BASE/p63i-$T.out | awk '{print $7}')
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p63i-$T.out | awk '{print $5}')
    SW=$(grep -oE "delta_pages [0-9-]+" $BASE/p63i-$T.out | awk '{print $2}')
    CPUUS=$(grep -oE "ggml-sched-split id 0 backend CPU nodes [0-9]+ avg_us [0-9]+" $R/stderr.txt 2>/dev/null | awk '{print $9}' | sort -rn | head -1)
    echo "P63I_RUN tag $T nt $NT rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) prefill_ms ${PFMS:-NA} decode_ms ${DEC:-NA} cpu_split_us ${CPUUS:-NA} swap_delta_pages ${SW:-NA} cap0 $C0-$C1 tz0 $Z0-$Z1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

echo "P63I start tag $TAG rounds $ROUNDS uptime $(cut -d' ' -f1 /proc/uptime) cap0 $(cap) tz0 $(tz0)"
cp $L/replay32.txt $L/replay.txt

for P in $(seq 1 $ROUNDS); do
    arm libggml-cpu.so.c62pre $CPU_A libggml-htp-v81.so.c58x4 $SKEL_A A || exit 2
    pf $TAG-p$P-A 128 $SKEL_A
    arm libggml-cpu.so.c62pre $CPU_A libggml-htp-v81.so.c62q6 $SKEL_B C || exit 2
    pf $TAG-p$P-C 128 $SKEL_B
    arm libggml-cpu.so.c62g1 $CPU_B libggml-htp-v81.so.c58x4 $SKEL_A D || exit 2
    pf $TAG-p$P-D 128 $SKEL_A
    arm libggml-cpu.so.c62g1 $CPU_B libggml-htp-v81.so.c62q6 $SKEL_B B || exit 2
    pf $TAG-p$P-B 128 $SKEL_B
done

put $L/libggml-cpu.so.c44bak libggml-cpu.so $CPU_STOCK || exit 2
put $L/libggml-htp-v81.so.c58x4 libggml-htp-v81.so $SKEL_A || exit 2
cp $BASE/replay4.bak $L/replay.txt
echo "P63I replay_restored $(cat $L/replay.txt)"
echo "P63I stray $(ps -A -o PID,NAME | awk '/decode-probe/{printf "%s ", $1}')"
echo "P63I_DONE cpu $(sha256sum $L/libggml-cpu.so | cut -d' ' -f1) skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
