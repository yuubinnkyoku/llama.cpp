#!/system/bin/sh
# Phase 64 - speed ABBA, Phase 63 standard against the foundation build.
#
# Only the HTP side differs. The CPU backend is the same file in both arms, so
# the comparison isolates what the hexagon and skel changes did.
#
#   A = cpu 1cc02638 + hexagon 9c9bfc48 + skel 312ad7b7   (Phase 63 standard)
#   B = cpu 1cc02638 + hexagon 0b94938e + skel 76b20839   (foundation batch)
#
#   sh p64_abba.sh <tag> <rounds>
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
CPU=1cc02638355148a78e0423036169f9ae8f480c44d5eeebf0cc9e3ed3506c9150
HEX_A=9c9bfc4874a00a2611caff34ad7b85ce56cf46d2787278f44552d228f7214a72
SKEL_A=312ad7b7ab304762f7c347e7909ee5d569541b15994073cfc11f55d0edf38114
HEX_B=0b94938e93107fbae8668740ad6431bb9c008cdbbed27cf4f54840597e350eac
SKEL_B=76b20839d30974e9f025e1049d089b489d28f5bc8b8d2d4c05ca73831dbdda3c

TAG=${1:-abba}
ROUNDS=${2:-2}
TIMEOUT=${TIMEOUT:-1800}

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

put() {
    cp "$1" "$L/$2" || return 2
    chmod 644 "$L/$2"
    GOT=$(sha256sum "$L/$2" | cut -d' ' -f1)
    [ "$GOT" = "$3" ] || { echo "P64A REFUSING $2 $GOT != $3"; return 2; }
}

arm() {  # $1 hex file, $2 hex sha, $3 skel file, $4 skel sha, $5 label
    CPUG=$(sha256sum $L/libggml-cpu.so | cut -d' ' -f1)
    [ "$CPUG" = "$CPU" ] || { echo "P64A REFUSING cpu $CPUG != $CPU"; return 2; }
    put "$L/$1" libggml-hexagon.so "$2" || return 2
    put "$L/$3" libggml-htp-v81.so "$4" || return 2
    echo "P64A_ARM $5 hex $2 skel $4"
}

pf() {
    T=$1; NT=$2; SKEL=$3
    C0=$(cap); Z0=$(tz0); S0=$(date +%s)
    env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
        P53_PROMPT_TOKENS=$NT P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay 44 $T 0 > $BASE/p64a-$T.out 2>&1
    RC=$?
    S1=$(date +%s); C1=$(cap); Z1=$(tz0)
    R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p64a-$T.out 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || R=$(ls -td $BASE/runs/*/ | head -1)
    EXIT=$(cat $R/exit.txt 2>/dev/null)
    PFMS=$(grep -oE "PREFILL seg 0 tokens [0-9]+ ms [0-9.]+" $BASE/p64a-$T.out | awk '{print $7}')
    DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p64a-$T.out | awk '{print $5}')
    SW=$(grep -oE "delta_pages [0-9-]+" $BASE/p64a-$T.out | awk '{print $2}')
    CPUUS=$(grep -oE "ggml-sched-split id 0 backend CPU nodes [0-9]+ avg_us [0-9]+" $R/stderr.txt 2>/dev/null | awk '{print $9}' | sort -rn | head -1)
    SPL=$(grep -oE "splits = [0-9]+" $R/stderr.txt 2>/dev/null | tail -1 | awk '{print $3}')
    echo "P64A_RUN tag $T nt $NT rc $RC exit ${EXIT:-NA} prefill_ms ${PFMS:-NA} decode_ms ${DEC:-NA} cpu_split_us ${CPUUS:-NA} splits ${SPL:-NA} swap_delta_pages ${SW:-NA} cap0 $C0-$C1 tz0 $Z0-$Z1 dir $R"
    if [ "$RC" != "0" ] || [ "${EXIT:-x}" != "0" ]; then
        for p in $(ps -A -o PID,NAME | awk '/decode-probe/{print $1}'); do
            echo "P64A KILL orphan $p"
            kill -9 $p 2>/dev/null
        done
        sleep 2
    fi
}

echo "P64A start tag $TAG rounds $ROUNDS uptime $(cut -d' ' -f1 /proc/uptime) cap0 $(cap) tz0 $(tz0)"
cp $L/replay32.txt $L/replay.txt

for P in $(seq 1 $ROUNDS); do
    if [ $((P % 2)) -eq 1 ]; then
        arm libggml-hexagon.so.c62g2 $HEX_A libggml-htp-v81.so.c62q6 $SKEL_A A || exit 2
        pf $TAG-p$P-A128 128 $SKEL_A
        pf $TAG-p$P-A512 512 $SKEL_A
        arm libggml-hexagon.so.c64q2k $HEX_B libggml-htp-v81.so.c64q2k $SKEL_B B || exit 2
        pf $TAG-p$P-B128 128 $SKEL_B
        pf $TAG-p$P-B512 512 $SKEL_B
    else
        arm libggml-hexagon.so.c64q2k $HEX_B libggml-htp-v81.so.c64q2k $SKEL_B B || exit 2
        pf $TAG-p$P-B128 128 $SKEL_B
        pf $TAG-p$P-B512 512 $SKEL_B
        arm libggml-hexagon.so.c62g2 $HEX_A libggml-htp-v81.so.c62q6 $SKEL_A A || exit 2
        pf $TAG-p$P-A128 128 $SKEL_A
        pf $TAG-p$P-A512 512 $SKEL_A
    fi
done

# restore the Phase 63 standard
put $L/libggml-hexagon.so.c62g2 libggml-hexagon.so $HEX_A || exit 2
put $L/libggml-htp-v81.so.c62q6 libggml-htp-v81.so $SKEL_A || exit 2
cp $BASE/replay4.bak $L/replay.txt
echo "P64A replay_restored $(cat $L/replay.txt)"
echo "P64A stray $(ps -A -o PID,NAME | awk '/decode-probe/{printf "%s ", $1}')"
echo "P64A_DONE cpu $(sha256sum $L/libggml-cpu.so | cut -d' ' -f1) skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
