#!/system/bin/sh
# Phase 62 - upstream integration (welcome group) measurements.
#
#   sh p62_run.sh unit  [tag]                       # tiled mulmat unit test
#   sh p62_run.sh base  [tag]                       # regression gate on the stock binaries
#   sh p62_run.sh qual  <tag> <cpu-sha> <ngl> <env> # 32-step with logits, for NLL/argmax
#   sh p62_run.sh pf    <tag> <cpu-sha> <ngl> <nt> <env>  # prefill spot, no logits
#   sh p62_run.sh evid  <tag> <cpu-sha> <ngl> <env> # evidence test: load and run, keep everything
#
# <env> is a single word, use - for none, or quote a "K=V K=V" string.
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
SKEL=${P62_SKEL:-$(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)}
CPU_BASE=0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9
TIMEOUT=${TIMEOUT:-1800}

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

# swap the cpu backend. $1 = file name under libs/, $2 = expected sha (optional).
# An empty $1 keeps whatever is installed; an empty $2 skips the sha check.
set_cpu() {
    [ -n "${1:-}" ] || { echo "P62_CPU unchanged $(sha256sum $L/libggml-cpu.so | cut -d' ' -f1)"; return 0; }
    # copying a file onto itself is a no-op, not an error
    if [ "$L/$1" = "$L/libggml-cpu.so" ]; then
        echo "P62_CPU in place $(sha256sum $L/libggml-cpu.so | cut -d' ' -f1)"
        return 0
    fi
    cp "$L/$1" $L/libggml-cpu.so || return 2
    chmod 644 $L/libggml-cpu.so
    GOT=$(sha256sum $L/libggml-cpu.so | cut -d' ' -f1)
    if [ -n "${2:-}" ] && [ "$GOT" != "$2" ]; then
        echo "P62 REFUSING cpu $GOT != $2"
        return 2
    fi
    echo "P62_CPU $GOT"
}

# 32-step replay with logits kept, so NLL/argmax can be computed
qual() {
    T=$1; NGL=$2; EXTRA=$3
    C0=$(cap); Z0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 $EXTRA \
        sh $BASE/p53_run.sh 32 replay $NGL $T 0 > $BASE/p62-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); Z1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p62-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    H=$(sha256sum $R/logits.f32 2>/dev/null | cut -d' ' -f1)
    SW=$(grep -oE "delta_pages [0-9-]+" $BASE/p62-$T.out | awk '{print $2}')
    CPUUS=$(grep -oE "ggml-sched-split id 0 backend CPU nodes [0-9]+ avg_us [0-9]+" $R/stderr.txt 2>/dev/null | awk '{print $9}' | sort -rn | head -1)
    echo "P62_QUAL tag $T ngl $NGL rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) logits_sha ${H:-NA} cpu_split_us ${CPUUS:-NA} swap_delta_pages ${SW:-NA} cap0 $C0-$C1 tz0 $Z0-$Z1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P62 KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

# prefill + 4 decode, no logits
pf() {
    T=$1; NGL=$2; NT=$3; EXTRA=$4
    C0=$(cap); Z0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 $EXTRA \
        P53_PROMPT_TOKENS=$NT P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay $NGL $T 0 > $BASE/p62-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); Z1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p62-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    PFMS=$(grep -oE "PREFILL seg 0 tokens [0-9]+ ms [0-9.]+" $BASE/p62-$T.out | awk '{print $7}')
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p62-$T.out | awk '{print $5}')
    SW=$(grep -oE "delta_pages [0-9-]+" $BASE/p62-$T.out | awk '{print $2}')
    CPUUS=$(grep -oE "ggml-sched-split id 0 backend CPU nodes [0-9]+ avg_us [0-9]+" $R/stderr.txt 2>/dev/null | awk '{print $9}' | sort -rn | head -1)
    echo "P62_PF tag $T ngl $NGL nt $NT rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) prefill_ms ${PFMS:-NA} decode_ms ${DEC:-NA} cpu_split_us ${CPUUS:-NA} swap_delta_pages ${SW:-NA} cap0 $C0-$C1 tz0 $Z0-$Z1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P62 KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

# evidence test: try to load and run, keep whatever happens
evid() {
    T=$1; NGL=$2; EXTRA=$3
    C0=$(cap); Z0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 $EXTRA \
        P53_PROMPT_TOKENS=128 P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay $NGL $T 0 > $BASE/p62-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); Z1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p62-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    PFMS=$(grep -oE "PREFILL seg 0 tokens [0-9]+ ms [0-9.]+" $BASE/p62-$T.out | awk '{print $7}')
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p62-$T.out | awk '{print $5}')
    echo "P62_EVID tag $T ngl $NGL rc $RC exit ${EXIT:-NA} wall_s $((S1 - S0)) prefill_ms ${PFMS:-NA} decode_ms ${DEC:-NA} cap0 $C0-$C1 tz0 $Z0-$Z1 dir $R"
    echo "P62_EVID srctail $(tail -5 $R/stdout.txt 2>/dev/null | tr '\n' '|')"
    echo "P62_EVID errtail $(tail -5 $R/stderr.txt 2>/dev/null | tr '\n' '|')"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P62 KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

STAGE=${1:-unit}
TAG=${2:-p62}

echo "P62 start stage $STAGE tag $TAG skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1) cpu $(sha256sum $L/libggml-cpu.so | cut -d' ' -f1) base $(sha256sum $L/libggml-base.so | cut -d' ' -f1) cap0 $(cap) tz0 $(tz0) uptime $(cut -d' ' -f1 /proc/uptime)"

case "$STAGE" in
unit)
    U=$BASE/c60unit
    [ -x "$U/test-tiled-mulmat" ] || { echo "P62_UNIT missing"; exit 2; }
    LD_LIBRARY_PATH=$U ADSP_LIBRARY_PATH=$U "$U/test-tiled-mulmat" > $BASE/p62-unit.log 2>&1
    RC=$?
    echo "P62_UNIT rc $RC pass $(grep -c PASS $BASE/p62-unit.log) fail $(grep -c FAIL $BASE/p62-unit.log)"
    grep -E 'FAIL' $BASE/p62-unit.log | head -10
    LD_LIBRARY_PATH=$U ADSP_LIBRARY_PATH=$U "$U/test-tiled-mulmat" --fuzz > $BASE/p62-unit-fuzz.log 2>&1
    RC2=$?
    echo "P62_UNIT fuzz rc $RC2 pass $(grep -c PASS $BASE/p62-unit-fuzz.log) fail $(grep -c FAIL $BASE/p62-unit-fuzz.log)"
    grep -E 'FAIL' $BASE/p62-unit-fuzz.log | head -10
    ;;
base)
    # the stock pair: 0a3db9c5 cpu with the device's own base d950aa96
    set_cpu libggml-cpu.so.c44bak $CPU_BASE || exit 2
    cp $L/replay32.txt $L/replay.txt
    qual $TAG-base44 44 ""
    qual $TAG-base0  0  ""
    cp $BASE/replay4.bak $L/replay.txt
    echo "P62 replay_restored $(cat $L/replay.txt)"
    ;;
qual)
    CSHA=${3:?cpu sha}; NGL=${4:-44}; EXTRA=${5:-}
    [ "$EXTRA" = "-" ] && EXTRA=""
    set_cpu "$CSHA" "" || exit 2
    cp $L/replay32.txt $L/replay.txt
    qual $TAG $NGL "$EXTRA"
    cp $BASE/replay4.bak $L/replay.txt
    ;;
pf)
    CSHA=${3:?cpu sha}; NGL=${4:-44}; NT=${5:-128}; EXTRA=${6:-}
    [ "$EXTRA" = "-" ] && EXTRA=""
    set_cpu "$CSHA" "" || exit 2
    cp $L/replay32.txt $L/replay.txt
    pf $TAG $NGL $NT "$EXTRA"
    cp $BASE/replay4.bak $L/replay.txt
    ;;
evid)
    CSHA=${3:-}; NGL=${4:-44}; EXTRA=${5:-}
    [ "$EXTRA" = "-" ] && EXTRA=""
    [ -n "$CSHA" ] && { set_cpu "$CSHA" "" || exit 2; }
    cp $L/replay32.txt $L/replay.txt
    evid $TAG $NGL "$EXTRA"
    cp $BASE/replay4.bak $L/replay.txt
    ;;
*)
    echo "P62 REFUSING unknown stage $STAGE"
    exit 2
    ;;
esac

echo "P62 stray $(ps -A -o PID,NAME | awk '/decode-probe/{printf "%s ", $1}')"
echo "P62_DONE cpu $(sha256sum $L/libggml-cpu.so | cut -d' ' -f1) base $(sha256sum $L/libggml-base.so | cut -d' ' -f1) skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
