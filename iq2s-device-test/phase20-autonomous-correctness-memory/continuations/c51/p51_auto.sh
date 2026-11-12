#!/system/bin/sh
# Phase 51 - wait for the phone to be left alone, then run the comparison.
#
#   sh p51_auto.sh [ngl] [pairs] [minutes]
#
# Polls the resumed activity every 30 s. When the launcher has been in front for
# two minutes in a row (four polls), it runs p51_finish.sh, which checks the skel
# and the running probe and then lets run50.sh gate memory after its force-stop
# list. Nothing is forced off before that: the phone stays usable while waiting.
set -u
BASE=/data/local/tmp/c46-27b
NGL=${1:-9}
PAIRS=${2:-2}
MINUTES=${3:-60}

idle=0
i=0
while [ "$i" -lt $((MINUTES * 2)) ]; do
    fg=$(dumpsys activity activities 2>/dev/null | grep -m1 topResumedActivity)
    case "$fg" in
        *launcher*|*Launcher*) idle=$((idle + 1)) ;;
        *)                     idle=0 ;;
    esac
    if [ "$idle" -ge 4 ]; then
        echo "IDLE $(date +%H:%M:%S) after $((i * 30))s"
        break
    fi
    sleep 30
    i=$((i + 1))
done

if [ "$idle" -lt 4 ]; then
    echo "NOT_IDLE after $((MINUTES)) minutes, nothing run"
    exit 0
fi

echo "START $(date +%H:%M:%S) ngl $NGL pairs $PAIRS"
sh $BASE/p51_finish.sh "$NGL" "$PAIRS"
echo "AUTO_DONE $(date +%H:%M:%S)"
