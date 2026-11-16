#!/system/bin/sh
# Phase 57 - measure the batch boundary cache maintenance directly.
#
#   sh p57_flush.sh <tag> [ngl] [steps]
#
# Runs with GGML_HEXAGON_PROFILE=3 (HTP_PROF_TRACE) so the DSP records the
# HTP_TRACE_EVT_L2FLUSH spans that already bracket qurt_mem_cache_clean in
# htp/main.c and the range flushes in htp-tensor.c. The host dumps them as
# "trace-evt L2FLUSH: thread T info I start|stop CYCLES"; this script pairs the
# spans per thread and reports count, mean and max in cycles and microseconds
# at the DSP clock the profile also reports.
set -u
BASE=/data/local/tmp/c46-27b
L=$BASE/libs
TAG=${1:-flush}
NGL=${2:-44}
STEPS=${3:-4}
TIMEOUT=${TIMEOUT:-600}
OV=${OV:-$L/override-l57.txt}
SKEL=$(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)

cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

echo "P57_FLUSH start skel $SKEL ngl $NGL steps $STEPS cap0 $(cap) tz0 $(tz0)"
env WANT_SKEL=$SKEL GUARD_PAGES=2097152 RUN_TIMEOUT=$TIMEOUT MEM_MIN_KB=4000000 OV_FILE=$OV \
    PROFILE_ARG=3 P53_PROMPT_TOKENS=128 P53_NO_LOGITS=1 \
    sh $BASE/p53_run.sh $STEPS noreplay $NGL $TAG 0 > $BASE/p57-$TAG.out 2>&1
RC=$?
R=$(grep -oE "P53_END .* dir [^ ]+" $BASE/p57-$TAG.out 2>/dev/null | awk '{print $NF}')
[ -n "$R" ] || R=$(ls -td $BASE/runs/*/ 2>/dev/null | head -1)
EXIT=$(cat $R/exit.txt 2>/dev/null)
PF=$(grep -oE "PREFILL seg 0 tokens 128 ms [0-9.]+" $BASE/p57-$TAG.out | awk '{print $7}')
DEC=$(grep -oE "RESULT forwards [0-9]+ decode_ms [0-9.]+" $BASE/p57-$TAG.out | awk '{print $5}')
MZH=$(grep -oE "profile-op OPBATCH.*mhz [0-9.]+" $R/stderr.txt 2>/dev/null | head -1 | awk '{print $NF}')

grep "trace-evt L2FLUSH" $R/stderr.txt > $R/l2flush.txt 2>/dev/null
N=$(wc -l < $R/l2flush.txt)
echo "P57_FLUSH run tag $TAG ngl $NGL rc $RC exit ${EXIT:-NA} prefill128_ms ${PF:-NA} decode_ms ${DEC:-NA} mhz ${MZH:-NA} l2flush_events $N dir $R"

awk -v mhz="${MZH:-1700}" '
/start/ { for (i = 1; i <= NF; i++) if ($i == "thread") { t = $(i+1); break }
          for (i = 1; i <= NF; i++) if ($i == "start") { st[t] = $(i+1); break }
          next }
/stop/  { for (i = 1; i <= NF; i++) if ($i == "thread") { t = $(i+1); break }
          for (i = 1; i <= NF; i++) if ($i == "stop") { sp = $(i+1); break }
          if (!(t in st)) next
          d = sp - st[t]; if (d < 0) d += 4294967296
          n++; sum += d; if (d > mx) mx = d; if (mn == 0 || d < mn) mn = d
          delete st[t]; next }
END { if (n == 0) { print "P57_FLUSHSPAN none"; exit }
      printf "P57_FLUSHSPAN n %d mean_cyc %d max_cyc %d min_cyc %d mean_us %.1f max_us %.1f min_us %.1f\n",
             n, sum/n, mx, mn, (sum/n)/mhz, mx/mhz, mn/mhz }' $R/l2flush.txt

echo "P57_FLUSH_DONE skel $(sha256sum $L/libggml-htp-v81.so | cut -d' ' -f1)"
