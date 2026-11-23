#!/bin/bash
# Phase 67 diagnostic 4: 32-token prompt (256 rows) tiled HTP vs CPU reference.
set -u
export MSYS_NO_PATHCONV=1
A=adb
B=/data/local/tmp/c61-inkling2
echo "== mem"
$A shell "grep -E '^(MemAvailable|SwapFree):' /proc/meminfo"
echo "== tiled HTP (m_chunk=128)"
$A shell "LIBDIR=$B/variant-diag GGML_HEXAGON_ID_MCHUNK=128 PT=32 NB=32 NUB=32 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-down-24.txt sh $B/p62_probe.sh c67g_t 1 48 HTP0:0" >/dev/null 2>&1
echo "rc=$?"
$A shell "grep -a -E 'TOKEN step 0|PREFILL' $B/runs/c67g_t/stdout.txt; grep -a 'model buffer size' $B/runs/c67g_t/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c67g_t/manifest.txt; ls -l $B/runs/c67g_t/logits.f32 2>/dev/null | awk '{print \$5}'"
echo "== CPU reference"
$A shell "LIBDIR=$B/libs PT=32 NB=32 NUB=32 FA_SELECT=1 DMA64=0 SMALLMM=0 OVERRIDE=override-down-cpu.txt sh $B/p62_probe.sh c67g_c 1 48 HTP0:0" >/dev/null 2>&1
echo "rc=$?"
$A shell "grep -a -E 'TOKEN step 0|PREFILL' $B/runs/c67g_c/stdout.txt; grep -a 'model buffer size' $B/runs/c67g_c/stderr.txt | grep -a HTP; grep -a swap_total $B/runs/c67g_c/manifest.txt"
$A pull $B/runs/c67g_t/logits.f32 D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/q64/diag_t.f32 >/dev/null
$A pull $B/runs/c67g_c/logits.f32 D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/q64/diag_c.f32 >/dev/null
python D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/gate.py D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/q64/diag_c.f32 D:/ghq/github.com/yuubinnkyoku/llama.cpp-c61/iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c66/q64/diag_t.f32
echo "== done"