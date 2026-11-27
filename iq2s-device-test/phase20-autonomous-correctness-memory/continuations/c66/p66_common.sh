#!/system/bin/sh
# Phase 66 - shared measurement helpers.
#
# Sourced by the sweep scripts, not run directly. Every rule that Phase 64 and
# Phase 65 learned the hard way lives here once instead of being retyped per
# script, because retyping is how it got forgotten.
#
#   . /data/local/tmp/c46-27b/p66_common.sh
#
# What it guarantees:
#   - the run directory comes from the probe's own SPLIT_RUN line. There is no
#     "newest directory" fallback: that fallback silently reported a previous
#     run's numbers three times across Phase 65.
#   - cap is read before and after, and the caller compares only matched caps.
#   - a prefill or decode value already seen is flagged, since an exact repeat
#     means a stale read rather than a measurement.
#   - the replay file is restored from the real file and then compared with the
#     expected value. Phase 65 had the restore fail silently twice.
#   - a failed run aborts the sweep instead of reporting the last good value.
#
# Requires: BASE, L, and the caller's tag conventions.

P66_REPLAY_EXPECT="25358 11 198 760"
P66_SEEN=""

p66_cap() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null; }
p66_tz()  { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; }

# Restore the standard replay and prove it. Returns non-zero if the value is wrong.
p66_replay_restore() {
    cp "$L/replay.txt.std" "$L/replay.txt" 2>/dev/null || true
    got=$(cat "$L/replay.txt" 2>/dev/null)
    if [ "$got" = "$P66_REPLAY_EXPECT" ]; then
        echo "P66 replay_restore OK $got"
        return 0
    fi
    echo "P66 REPLAY_RESTORE_FAILED got [$got] want [$P66_REPLAY_EXPECT]"
    return 1
}

# Install a replay file for a measured run and check it took.
# $1 = file name under libs (e.g. replay32.txt)
p66_replay_set() {
    cp "$L/$1" "$L/replay.txt" 2>/dev/null || { echo "P66 SET_FAILED $1"; return 1; }
    got=$(cat "$L/replay.txt" 2>/dev/null)
    want=$(cat "$L/$1" 2>/dev/null)
    [ "$got" = "$want" ] && return 0
    echo "P66 SET_MISMATCH $1 got [$got] want [$want]"
    return 1
}

# Run dir from the probe's own stdout line. No fallback.
# $1 = p53 log path
p66_rundir() {
    sed -n 's/.*SPLIT_RUN rc [0-9][0-9]* dir \(.*\)$/\1/p' "$1" 2>/dev/null | tail -1
}

# Returns 0 and echoes the value, or returns 1 with a reason on stderr.
# $1 = log, $2 = run dir, $3 = grep pattern, $4 = awk field, $5 = label
p66_field() {
    dir=$(p66_rundir "$1")
    [ -n "$dir" ] || { echo "P66_ABORT $5 no SPLIT_RUN line in $1"; return 1; }
    [ -d "$dir" ] || { echo "P66_ABORT $5 dir missing $dir"; return 1; }
    v=$(grep -oE "$3" "$dir/stdout.txt" 2>/dev/null | tail -1 | awk -v f="$4" '{print $f}')
    [ -n "$v" ] || { echo "P66_ABORT $5 field missing in $dir/stdout.txt"; return 1; }
    echo "$v"
    return 0
}

# Flag an exact repeat. $1 = value, $2 = label
p66_dup() {
    case " $P66_SEEN " in
        *" $1 "*) echo "DUP!"; return 0 ;;
    esac
    P66_SEEN="$P66_SEEN $1"
    return 1
}
