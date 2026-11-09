#!/system/bin/sh
# Phase 50 - catch the fast CPU cap and measure while it lasts.
#
#   sh p50_fastcap.sh wait <max_wait_s> [steps] [skel_sha]
#   sh p50_fastcap.sh now  [steps] [skel_sha]
#
# The vendor cap sits at 2112000 / 1862400 for hours and only occasionally
# returns to the hardware maximum (3628800 / 4608000), for about ten minutes at
# a time.  `wait` polls until cap0 and cap7 both equal cpuinfo_max_freq, then
# immediately runs the palindromic series 0,16,24,24,16,0 through p50_abba.sh.
# The palindrome is needed because a run later in a series comes out around 5%
# faster at the same configuration (measured: 741.9 then 700.3 at NGL=0).
#
# Nothing here changes the cap, the thermal stack or any service; it only waits
# and observes.
set -u

BASE=/data/local/tmp/c46-27b
MODE=${1:-now}
WAIT_MAX=${2:-7200}
STEPS=${3:-8}
SKEL=${4:-288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d}

MAX0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null)
MAX7=$(cat /sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_max_freq 2>/dev/null)
cap0() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
cap7() { cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq 2>/dev/null; }
tz0() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

case "$STEPS" in ''*[!0-9]*) echo "REFUSING: steps must be a number"; exit 2 ;; esac
[ -n "$MAX0" ] && [ -n "$MAX7" ] || { echo "REFUSING: cannot read cpuinfo_max_freq"; exit 2; }

is_fast() { [ "$(cap0)" = "$MAX0" ] && [ "$(cap7)" = "$MAX7" ]; }

if [ "$MODE" = "wait" ]; then
    echo "WAIT start cap0 $(cap0)/$MAX0 cap7 $(cap7)/$MAX7 tz $(tz0) max ${WAIT_MAX}s"
    waited=0
    LOW=900000   # the Dozing step observed at 883200; below this the device is idle
    while [ "$waited" -lt "$WAIT_MAX" ]; do
        if is_fast; then
            echo "WAIT fast_at ${waited}s cap0 $(cap0) cap7 $(cap7) tz $(tz0)"
            break
        fi
        # The fast cap has only been seen right after waking a device that had
        # settled at the Dozing level, so wake only in that state and re-check.
        if [ "$(cap0)" -lt "$LOW" ] 2>/dev/null; then
            echo "WAIT dozing_at ${waited}s cap0 $(cap0) waking"
            input keyevent KEYCODE_WAKEUP >/dev/null 2>&1
            sleep 3
            input keyevent KEYCODE_MENU >/dev/null 2>&1
            sleep 2
            if is_fast; then
                echo "WAIT fast_after_wake ${waited}s cap0 $(cap0) cap7 $(cap7) tz $(tz0)"
                break
            fi
            echo "WAIT wake_no_fast cap0 $(cap0)/$MAX0 cap7 $(cap7)/$MAX7"
        fi
        sleep 10
        waited=$((waited + 10))
    done
    if ! is_fast; then
        echo "WAIT timeout ${waited}s cap0 $(cap0)/$MAX0 cap7 $(cap7)/$MAX7 tz $(tz0)"
        exit 3
    fi
fi

if ! is_fast; then
    echo "REFUSING: not at the fast cap. cap0 $(cap0)/$MAX0 cap7 $(cap7)/$MAX7."
    echo "Use the 'wait' mode, or measure at the throttled tier instead."
    exit 2
fi

# hold the screen awake for the duration; the same keepalive as p50_run.sh
( i=0; while [ $i -lt 60 ]; do input keyevent KEYCODE_WAKEUP >/dev/null 2>&1; sleep 20; i=$((i+1)); done ) &
KP=$!

echo "FASTCAP begin cap0 $(cap0) cap7 $(cap7) tz $(tz0) steps $STEPS skel $SKEL"
WANT_SKEL=$SKEL PROFILE_ARG=0 sh $BASE/p50_abba.sh "$STEPS" 0:f0a 16:f16a 24:f24a 24:f24b 16:f16b 0:f0b > $BASE/p50-fastcap.out 2>&1
kill $KP 2>/dev/null
echo "FASTCAP end cap0 $(cap0) cap7 $(cap7) tz $(tz0)"
grep -E "^P50_RUN|^P50_STAT" $BASE/p50-fastcap.out
