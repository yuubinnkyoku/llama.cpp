#!/system/bin/sh
# Phase 65 target 1 - A/B/A of the two batches Phase 64 could not time.
#
# A  = current standard      hexagon 9c9bfc48  skel 312ad7b7
# B1 = q2_K/ssm batch        hexagon 0b94938e  skel 76b20839   (hash 085e13b5, quality PASS)
# B2 = FA batch              hexagon c9dadbde  skel cdc628a9   (hash 8259e4b0, quality PASS)
#
# CPU is abecaf60 on every arm, so only the HTP pair changes.
# cap is read before and after each run. Phase 64 showed cap swings move prefill by
# nearly 2x, so runs are only comparable within the same cap; the caller sorts that out.
#
#   sh p65_abba.sh <tag> <rounds>
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
CPU=abecaf609042e0921b1dcd651a5982e3882ed0dbbc673026bf9bec3cd36d8c51
TAG=${1:-abba}
ROUNDS=${2:-2}
TIMEOUT=${TIMEOUT:-1800}

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

# arm <hexagon suffix> <skel suffix> <label>
arm() {
    HEXL=$1; SKEL=$2; LAB=$3
    cp $L/libggml-hexagon.so.$HEXL $L/libggml-hexagon.so || return 2
    cp $L/libggml-htp-v81.so.$SKEL $L/libggml-htp-v81.so || return 2
    chmod 644 $L/libggml-hexagon.so $L/libggml-htp-v81.so
    G1=$(sha256sum $L/libggml-hexagon.so | cut -d' ' -f1)
    G2=$(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)
    G3=$(sha256sum $L/libggml-cpu.so | cut -d' ' -f1)
    [ "$G3" = "$CPU" ] || { echo "P65A REFUSING cpu $G3 != $CPU"; return 2; }
    CUR_SKEL=$G2
    echo "P65A_ARM $LAB hex ${G1:0:8} skel ${G2:0:8}"
}

pf() {  # $1 tag, $2 tokens
    T=$1; NT=$2
    C0=$(cap); Z0=$(tz0)
    env GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
        WANT_SKEL=$CUR_SKEL P53_PROMPT_TOKENS=$NT P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay 44 $T 0 > /dev/null 2>&1
    RC=$?
    C1=$(cap); Z1=$(tz0)
    R=$(grep -oE "SPLIT_RUN rc [0-9]+ dir .*" $BASE/p53-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    PFL=$(grep -oE "PREFILL seg 0 tokens [0-9]+ ms [0-9.]+" $R/stdout.txt 2>/dev/null | head -1 | awk '{print $7}')
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $R/stdout.txt 2>/dev/null | head -1 | awk '{print $5}')
    if [ "$RC" != "0" ]; then echo "P65A_ABORT run $T failed rc $RC (see p53-$T.out)"; return 3; fi
    echo "P65A_RUN $T nt $NT rc $RC exit ${EXIT:-NA} prefill_ms ${PFL:-NA} decode_ms ${DEC:-NA} cap0 $C0-$C1 tz0 $Z0-$Z1"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

echo "P65A start tag $TAG rounds $ROUNDS uptime $(cut -d' ' -f1 /proc/uptime) cap0 $(cap) tz0 $(tz0)"
cp $L/replay32.txt $L/replay.txt

i=1
while [ $i -le $ROUNDS ]; do
    # rotate the arm order each round so drift hits all three equally
    case $((i % 3)) in
        1) ORDER="A B1 B2" ;;
        2) ORDER="B1 B2 A" ;;
        *) ORDER="B2 A B1" ;;
    esac
    for LAB in $ORDER; do
        case $LAB in
            A)  arm c62g2 c62q6 A  || exit 2 ;;
            B1) arm c64q2k c64q2k B1 || exit 2 ;;
            B2) arm c64fa2 c64fa2 B2 || exit 2 ;;
        esac
        pf "$TAG-r$i-$LAB-128" 128
        pf "$TAG-r$i-$LAB-512" 512
    done
    i=$((i + 1))
done

cp $BASE/replay4.bak $L/replay.txt
echo "P65A replay_restored $(cat $L/replay.txt)"
echo "P65A_DONE cpu $(sha256sum $L/libggml-cpu.so | cut -d' ' -f1) hex $(sha256sum $L/libggml-hexagon.so | cut -d' ' -f1) skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
