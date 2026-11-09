#!/system/bin/sh
# Phase 50 - 64 token paired NLL: same forced tokens under CPU only and NGL=16.
#
#   sh p50_nll64.sh <skel_sha>
#
# Runs detached so an adb hiccup cannot kill it.  Restores the standard 4 token
# replay file at the end either way.
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
SKEL=${1:-288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d}

[ -f "$L/replay64.txt" ] || { echo "REFUSING: missing $L/replay64.txt"; exit 2; }
[ -f "$L/replay.txt.std" ] || { echo "REFUSING: missing $L/replay.txt.std"; exit 2; }
cp "$L/replay.txt.std" "$L/replay.txt.std.keep"

cp "$L/replay64.txt" "$L/replay.txt"
echo "NLL64 tokens $(wc -w < $L/replay.txt) start $(cut -d' ' -f1 /proc/uptime)"
WANT_SKEL=$SKEL PROFILE_ARG=0 sh $BASE/p50_run.sh 64 replay 0 nll64cpu 1 > $BASE/p50-nll64cpu.log 2>&1
echo "NLL64 cpu done $(grep -oE 'exit [0-9]+' $BASE/p50-nll64cpu.log | head -1)"
WANT_SKEL=$SKEL PROFILE_ARG=0 sh $BASE/p50_run.sh 64 replay 16 nll64htp 1 > $BASE/p50-nll64htp.log 2>&1
echo "NLL64 htp done $(grep -oE 'exit [0-9]+' $BASE/p50-nll64htp.log | head -1)"

cp "$L/replay.txt.std" "$L/replay.txt"
rm -f "$L/replay.txt.std.keep"
echo "NLL64 replay restored $(cat $L/replay.txt)"
for t in nll64cpu nll64htp; do
    R=$(grep -oE "runs/[0-9]+-[0-9]+-[0-9]+" $BASE/p50-$t.log 2>/dev/null | head -1)
    echo "NLL64 $t rundir $R"
    [ -n "$R" ] && sha256sum "$BASE/$R/logits.f32" 2>/dev/null
done
echo "NLL64 DONE"
