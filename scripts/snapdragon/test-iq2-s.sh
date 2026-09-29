#!/usr/bin/env bash
set -euo pipefail

# Minimal IQ2_S Hexagon device validation.
#
# n=1  exercises the decode/matvec HVX path.
# n=10 is HMX-eligible and exercises the prefill path on v81.
#
# Run from the llama.cpp repository root after building the Snapdragon preset.
# An adb-visible Snapdragon device is required.

PARAMS='^type_a=iq2_s,type_b=f32,m=16,n=(1|10),k=256,bs=\[1,1\],nr=\[1,1\]'

echo '== IQ2_S support check =='
./scripts/snapdragon/run.py --target adb --devices HTP0:0 -- \
    test-backend-ops support -b HTP0:0 -o MUL_MAT -p "$PARAMS"

echo '== IQ2_S correctness: HVX decode + HMX prefill =='
./scripts/snapdragon/run.py --target adb --devices HTP0:0 -- \
    test-backend-ops test -b HTP0:0 -o MUL_MAT -p "$PARAMS"

echo '== IQ2_S profile smoke test =='
./scripts/snapdragon/run.py --target adb --devices HTP0:0 --hex-profile 1 -- \
    test-backend-ops perf -b HTP0:0 -o MUL_MAT -p "$PARAMS"
