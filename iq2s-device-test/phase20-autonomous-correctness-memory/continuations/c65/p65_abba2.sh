#!/system/bin/sh
# Phase 65 target 1, second attempt - decide B1 against B2.
#
# A  = standard      hexagon 9c9bfc48  skel 312ad7b7
# B1 = q2_K/ssm      hexagon 0b94938e  skel 76b20839
# B2 = FA            hexagon c9dadbde  skel cdc628a9
#
# 512 tokens only: that is where B1 and B2 traded places across caps, and a
# longer prefill gives more HTP work so the signal is larger. 128 was already
# measured and never separated them.
#
# Carries the safety rules that the first attempt lacked (report 4.1 and 4.6):
#   - WANT_SKEL is passed per arm, else run53.sh refuses every B arm with rc 2
#   - the run log is removed first and must be newer than a marker, so a stale
#     directory cannot be read
#   - a repeated prefill value is flagged DUP
#   - any failure aborts the whole sweep instead of reporting old numbers
#
#   sh p65_abba2.sh <tag> <rounds>
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
CPU=abecaf609042e0921b1dcd651a5982e3882ed0dbbc673026bf9bec3cd36d8c51
TAG=${1:-dec}
ROUNDS=${2:-3}
TOKENS=${TOKENS:-512}
TIMEOUT=${TIMEOUT:-1800}

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

SEEN=""
CUR_SKEL=""

arm() {  # $1 hexagon file suffix, $2 skel file suffix, $3 label
    cp $L/libggml-hexagon.so.$1 $L/libggml-hexagon.so || return 2
    cp $L/libggml-htp-v81.so.$2 $L/libggml-htp-v81.so || return 2
    chmod 644 $L/libggml-hexagon.so $L/libggml-htp-v81.so
    G1=$(sha256sum $L/libggml-hexagon.so | cut -d' ' -f1)
    G2=$(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)
    G3=$(sha256sum $L/libggml-cpu.so | cut -d' ' -f1)
    [ "$G3" = "$CPU" ] || { echo "P65D REFUSING cpu ${G3:0:8}"; return 2; }
    CUR_SKEL=$G2
    echo "P65D_ARM $3 hex ${G1:0:8} skel ${G2:0:8}"
}

one() {  # $1 tag
    T=$1
    LOG=$BASE/p53-$T.out
    rm -f $LOG
    touch $BASE/.m-$T
    C0=$(cap); Z0=$(tz0)
    env WANT_SKEL=$CUR_SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 \
        WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 \
        P53_PROMPT_TOKENS=$TOKENS P53_NO_LOGITS=1 \
        sh $BASE/p53_run.sh 4 noreplay 44 $T 0 > /dev/null 2>&1
    RC=$?
    C1=$(cap); Z1=$(tz0)
    if [ "$RC" != "0" ]; then echo "P65D_ABORT $T rc $RC"; return 3; fi
    [ -f $LOG ] || { echo "P65D_ABORT $T no log"; return 3; }
    if [ $BASE/.m-$T -nt $LOG ]; then echo "P65D_ABORT $T log older than marker"; return 3; fi
    R=$(grep -oE "SPLIT_RUN rc [0-9]+ dir .*" $LOG 2>/dev/null | awk '{print $NF}')
    [ -n "$R" ] || { echo "P65D_ABORT $T no run dir"; return 3; }
    PFL=$(grep -oE "PREFILL seg 0 tokens [0-9]+ ms [0-9.]+" $R/stdout.txt 2>/dev/null | head -1 | awk '{print $7}')
    [ -n "$PFL" ] || { echo "P65D_ABORT $T no prefill"; return 3; }
    SW=$(grep -oE "pswpout [0-9]+" $R/stdout.txt 2>/dev/null | tail -1 | awk '{print $2}')
    DUP=""
    case " $SEEN " in *" $PFL "*) DUP=" DUP!" ;; esac
    SEEN="$SEEN $PFL"
    echo "P65D_RUN $T nt $TOKENS prefill_ms $PFL cap0 $C0-$C1 tz0 $Z0-$Z1 pswpout ${SW:-NA}$DUP"
}

echo "P65D start tag $TAG rounds $ROUNDS tokens $TOKENS uptime $(cut -d' ' -f1 /proc/uptime) cap0 $(cap) tz0 $(tz0)"
cp $L/replay32.txt $L/replay.txt

i=1
while [ $i -le $ROUNDS ]; do
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
        one "$TAG-r$i-$LAB"
        [ $? -eq 0 ] || exit 3
    done
    i=$((i + 1))
done

cp $BASE/replay4.bak $L/replay.txt
echo "P65D_DONE cpu $(sha256sum $L/libggml-cpu.so | cut -d' ' -f1) hex $(sha256sum $L/libggml-hexagon.so | cut -d' ' -f1) skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
