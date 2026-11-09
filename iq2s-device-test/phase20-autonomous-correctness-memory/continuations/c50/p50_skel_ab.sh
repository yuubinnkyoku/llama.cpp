#!/system/bin/sh
# Phase 50 - A/B two DSP skels back to back at one NGL.
#
#   sh p50_skel_ab.sh <steps> <ngl> <pairs> <tag> [noreplay|replay]
#
# Swaps only libggml-htp-v81.so between the Phase 49 baseline (A) and the
# Phase 50 build (C).  The host libraries are identical, so this isolates the
# DSP kernel.  Every run goes through p50_run.sh and therefore records the cap
# before, during and after.
#
# With replay the token sequence is pinned, so step s of A and step s of C do
# the same work and the pair can be compared directly.  steps must then be 4.
set -u

BASE=/data/local/tmp/c46-27b
STEPS=${1:-8}
NGL=${2:-16}
PAIRS=${3:-2}
PFX=${4:-sab}
TOK=${5:-noreplay}
A_SHA=a87330ad0e1739dbf7cf23c58b6935bfbc1a4b21c0d75735e608ef9e53795d49
C_SHA=10483e5b07d50286be5f9ceb70dac9cc8c7da5280bfcf58688044dc26aecc55c
A_SRC=/data/local/tmp/c50/skel-A-p49.so
C_SRC=$BASE/libs/skelC.so

[ -f "$A_SRC" ] || { echo "REFUSING: missing $A_SRC"; exit 2; }
[ -f "$C_SRC" ] || { echo "REFUSING: missing $C_SRC"; exit 2; }
if [ "$TOK" = "replay" ] && [ "$STEPS" != "4" ]; then
    echo "REFUSING: replay needs steps 4"; exit 2
fi

i=1
while [ "$i" -le "$PAIRS" ]; do
    cp "$A_SRC" $BASE/libs/libggml-htp-v81.so
    WANT_SKEL=$A_SHA PROFILE_ARG=0 sh $BASE/p50_run.sh "$STEPS" "$TOK" "$NGL" "$PFX-A$i" 1 > $BASE/p50-"$PFX"-A"$i".log 2>&1
    cp "$C_SRC" $BASE/libs/libggml-htp-v81.so
    WANT_SKEL=$C_SHA PROFILE_ARG=0 sh $BASE/p50_run.sh "$STEPS" "$TOK" "$NGL" "$PFX-C$i" 1 > $BASE/p50-"$PFX"-C"$i".log 2>&1
    i=$((i + 1))
done
# leave the Phase 50 build installed
cp "$C_SRC" $BASE/libs/libggml-htp-v81.so
echo "SKEL_RESTORED $(sha256sum $BASE/libs/libggml-htp-v81.so | cut -d' ' -f1)"
for f in $BASE/p50-"$PFX"-A*.log $BASE/p50-"$PFX"-C*.log; do
    grep -h -E "^P50_RUN|^P50_STAT" "$f" 2>/dev/null
done
