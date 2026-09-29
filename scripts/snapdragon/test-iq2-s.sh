#!/usr/bin/env bash
set -euo pipefail

# IQ2_S Hexagon device validation.
#
# TARGET may include an adb serial, for example:
#   TARGET=adb:192.168.55.17:5555 ./scripts/snapdragon/test-iq2-s.sh
#
# The test matrix covers:
#   - base decode (HVX) and prefill (HMX)
#   - multiple IQ2_S 256-value super-blocks along K
#   - 32-row tile boundaries
#   - ne2/ne3 batched slices

TARGET="${TARGET:-adb}"
DEVICES="${DEVICES:-HTP0:0}"

BASE_COMMON=(./scripts/snapdragon/run.py --target "$TARGET" --devices "$DEVICES")

BASE_DECODE='^type_a=iq2_s,type_b=f32,m=16,n=1,k=256,bs=\[1,1\],nr=\[1,1\]'
BASE_PREFILL='^type_a=iq2_s,type_b=f32,m=16,n=10,k=256,bs=\[1,1\],nr=\[1,1\]'

K_DECODE='^type_a=iq2_s,type_b=f32,m=16,n=1,k=(512|1024|4096|8192),bs=\[1,1\],nr=\[1,1\]'
K_PREFILL='^type_a=iq2_s,type_b=f32,m=16,n=10,k=(512|1024|4096|8192),bs=\[1,1\],nr=\[1,1\]'

ROW_DECODE='^type_a=iq2_s,type_b=f32,m=(31|32|33|63|64|65),n=1,k=1024,bs=\[1,1\],nr=\[1,1\]'
ROW_PREFILL='^type_a=iq2_s,type_b=f32,m=(31|32|33|63|64|65),n=10,k=1024,bs=\[1,1\],nr=\[1,1\]'

BATCH_DECODE='^type_a=iq2_s,type_b=f32,m=33,n=1,k=1024,bs=\[2,3\],nr=\[1,1\]'
BATCH_PREFILL='^type_a=iq2_s,type_b=f32,m=33,n=10,k=1024,bs=\[2,3\],nr=\[1,1\]'
LARGE_HMX='^type_a=iq2_s,type_b=f32,m=64,n=32,k=4096,bs=\[2,1\],nr=\[1,1\]'

echo '== IQ2_S base support =='
"${BASE_COMMON[@]}" --     test-backend-ops support -b "$DEVICES" -o MUL_MAT     -p 'type_a=iq2_s,type_b=f32,m=16,n=(1|10),k=256'

echo '== IQ2_S base HVX decode =='
"${BASE_COMMON[@]}" --hex-mm-select 1 --hex-profile 1 --     test-backend-ops test -b "$DEVICES" -o MUL_MAT -p "$BASE_DECODE"

echo '== IQ2_S base HMX prefill =='
"${BASE_COMMON[@]}" --hex-mm-select 2 --hex-profile 1 --     test-backend-ops test -b "$DEVICES" -o MUL_MAT -p "$BASE_PREFILL"

echo '== IQ2_S larger-K HVX decode =='
"${BASE_COMMON[@]}" --hex-mm-select 1 --     test-backend-ops test -b "$DEVICES" -o MUL_MAT -p "$K_DECODE"

echo '== IQ2_S larger-K HMX prefill =='
"${BASE_COMMON[@]}" --hex-mm-select 2 --     test-backend-ops test -b "$DEVICES" -o MUL_MAT -p "$K_PREFILL"

echo '== IQ2_S row-boundary HVX decode =='
"${BASE_COMMON[@]}" --hex-mm-select 1 --     test-backend-ops test -b "$DEVICES" -o MUL_MAT -p "$ROW_DECODE"

echo '== IQ2_S row-boundary HMX prefill =='
"${BASE_COMMON[@]}" --hex-mm-select 2 --     test-backend-ops test -b "$DEVICES" -o MUL_MAT -p "$ROW_PREFILL"

echo '== IQ2_S batched HVX decode =='
"${BASE_COMMON[@]}" --hex-mm-select 1 --     test-backend-ops test -b "$DEVICES" -o MUL_MAT -p "$BATCH_DECODE"

echo '== IQ2_S batched HMX prefill =='
"${BASE_COMMON[@]}" --hex-mm-select 2 --     test-backend-ops test -b "$DEVICES" -o MUL_MAT -p "$BATCH_PREFILL"

echo '== IQ2_S larger batched HMX shape =='
"${BASE_COMMON[@]}" --hex-mm-select 2 --hex-profile 1 --     test-backend-ops test -b "$DEVICES" -o MUL_MAT -p "$LARGE_HMX"
