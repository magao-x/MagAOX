#!/usr/bin/env bash
# Capture an operator-run finite, main-coordinated zero-auxiliary trial.
# Only synco's disabled step count and finite-trial enable switch are changed.
set -u
set -o pipefail

usage()
{
    cat <<'HELP'
Usage: capture-zero-aux-test.sh RATE [SECONDS]

RATE is effective total cadence, 4000, 6000 or 8000 Hz, with the main loop at 2000 Hz.
SECONDS is the active-trial duration, default 120, integer 1 through 120.
The app must already be running with zeroTest.enabled=true and enable=Off.
The script sets two/three/four steps while disabled, then starts the finite trial.
It never changes the main loop, camera, dmcomb, CPU placement or app config.

Requires camwfs at 2 kHz/synchronization On, RTC latency On, a configured
exclusive all-zero auxiliary, and the reader-release signal. No sudo required.
HELP
}

if [[ ${1:-} == --help || ${1:-} == -h ]]
then
    usage
    exit 0
fi
if (( $# < 1 || $# > 2 ))
then
    usage >&2
    exit 2
fi
zero_rate=$1
zero_seconds=${2:-120}
if [[ $zero_rate != 4000 && $zero_rate != 6000 && $zero_rate != 8000 ]]
then
    printf 'RATE must be effective total cadence 4000, 6000 or 8000\n' >&2
    exit 2
fi
if [[ ! $zero_seconds =~ ^[0-9]{1,3}$ ]] || (( 10#$zero_seconds < 1 || 10#$zero_seconds > 120 ))
then
    printf 'SECONDS must be an integer from 1 through 120\n' >&2
    exit 2
fi
zero_seconds=$((10#$zero_seconds))
zero_steps=$((zero_rate / 2000))
for zero_command in timeout logdump getINDI setINDI stdbuf git date sleep awk
do
    if ! command -v "$zero_command" >/dev/null
    then
        printf 'Required command not found: %s\n' "$zero_command" >&2
        exit 2
    fi
done

zero_capture_dir="$HOME/tmp/zero-aux-${zero_rate}Hz-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$zero_capture_dir" || exit 1
zero_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
git -C "$zero_script_dir/../../.." rev-parse HEAD > "$zero_capture_dir/git.txt" || exit 1
printf 'effectiveRate=%s\nsteps=%s\nactiveSeconds=%s\nmainRate=2000\n' \
    "$zero_rate" "$zero_steps" "$zero_seconds" > "$zero_capture_dir/test.txt"
zero_collectors=()
zero_devices=()
zero_started=0
zero_cleaned=0
zero_capture_failed=0

status()
{
    printf '%s\n' "$*" >> "$zero_capture_dir/status.txt"
}

setup_failure()
{
    local zero_actual=${2:-<missing>}
    local zero_message
    printf -v zero_message 'Setup check failed: %s actual="%s"; expected="%s"' "$1" "$zero_actual" "$3"
    printf '%s\n' "$zero_message" >&2
    status "$zero_message"
    zero_capture_failed=1
}

read_property()
{
    local zero_key=$1
    local zero_file=$2
    awk -F= -v key="$zero_key" '$1 == key { value = $2 } END { print value }' "$zero_file"
}

snapshot()
{
    local zero_phase=$1
    local zero_device=$2
    local zero_status=0
    timeout -k 2s 8s getINDI "$zero_device" \
        > "$zero_capture_dir/$zero_device-indi-$zero_phase.txt" \
        2> "$zero_capture_dir/$zero_device-indi-$zero_phase.err" || zero_status=$?
    status "$zero_device $zero_phase status=$zero_status"
    if (( zero_status != 0 ))
    then
        zero_capture_failed=1
        if [[ $zero_phase == before ]]
        then
            setup_failure "getINDI $zero_device baseline status" "$zero_status" 0
        fi
    fi
}

set_property()
{
    local zero_status=0
    timeout -k 2s 5s setINDI "$@" >> "$zero_capture_dir/control.txt" 2>&1 || zero_status=$?
    status "setINDI $* status=$zero_status"
    return "$zero_status"
}

cleanup()
{
    if (( zero_cleaned ))
    then
        return
    fi
    zero_cleaned=1
    if (( zero_started ))
    then
        if ! set_property -s dmtweeter-synco.enable.toggle=Off
        then
            zero_capture_failed=1
        fi
        zero_started=0
    fi
    local zero_index zero_pid zero_status
    for zero_pid in "${zero_collectors[@]}"
    do
        kill -TERM "$zero_pid" 2>/dev/null || true
    done
    for zero_index in "${!zero_collectors[@]}"
    do
        zero_pid=${zero_collectors[$zero_index]}
        zero_status=0
        wait "$zero_pid" || zero_status=$?
        status "collector ${zero_devices[$zero_index]} requestedStop=1 status=$zero_status"
        case $zero_status in
            0|124|130|143) ;;
            *) zero_capture_failed=1 ;;
        esac
    done
    zero_collectors=()
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Start all telemetry readers before taking the baseline or requesting any write.
for zero_device in dmtweeter-synco dmtweeter camwfs
do
    timeout -k 5s -s INT "$((zero_seconds + 180))s" \
        stdbuf -oL -eL logdump --dir=/opt/MagAOX/telem --ext=.bintel -f "$zero_device" \
        > "$zero_capture_dir/$zero_device.txt" 2>&1 &
    zero_collectors+=("$!")
    zero_devices+=("$zero_device")
done
for zero_device in dmtweeter-synco dmtweeter camwfs holoop sysMonRTC.set_latency.toggle
do
    snapshot before "$zero_device"
done
for zero_index in "${!zero_collectors[@]}"
do
    zero_pid=${zero_collectors[$zero_index]}
    if ! kill -0 "$zero_pid" 2>/dev/null
    then
        setup_failure "telemetry collector ${zero_devices[$zero_index]} (pid $zero_pid)" 'not running' running
    fi
done

zero_synco_before="$zero_capture_dir/dmtweeter-synco-indi-before.txt"
zero_camera_before="$zero_capture_dir/camwfs-indi-before.txt"
zero_latency_before="$zero_capture_dir/sysMonRTC.set_latency.toggle-indi-before.txt"
zero_enabled=$(read_property dmtweeter-synco.enable.toggle "$zero_synco_before")
zero_mode=$(read_property dmtweeter-synco.zero_test.enabled "$zero_synco_before")
zero_ideal=$(read_property dmtweeter-synco.zero_test.idealPublisher "$zero_synco_before")
zero_limit=$(read_property dmtweeter-synco.zero_test.durationSec "$zero_synco_before")
zero_camera_fps=$(read_property camwfs.fps.current "$zero_camera_before")
zero_camera_sync=$(read_property camwfs.synchro.toggle "$zero_camera_before")
zero_latency=$(read_property sysMonRTC.set_latency.toggle "$zero_latency_before")
[[ $zero_enabled == Off ]] || setup_failure dmtweeter-synco.enable.toggle "$zero_enabled" Off
[[ $zero_mode == 1 ]] || setup_failure dmtweeter-synco.zero_test.enabled "$zero_mode" 1
[[ $zero_ideal == 1 ]] || setup_failure dmtweeter-synco.zero_test.idealPublisher "$zero_ideal" 1
[[ $zero_camera_sync == On ]] || setup_failure camwfs.synchro.toggle "$zero_camera_sync" On
[[ $zero_latency == On ]] || setup_failure sysMonRTC.set_latency.toggle "$zero_latency" On
if ! awk -v fps="$zero_camera_fps" \
     'BEGIN { exit !(fps ~ /^[0-9]+([.][0-9]+)?$/ && fps >= 1999 && fps <= 2001) }'
then
    setup_failure camwfs.fps.current "$zero_camera_fps" 'numeric FPS from 1999 through 2001'
fi
if ! awk -v limit="$zero_limit" -v duration="$zero_seconds" \
     'BEGIN { exit !(limit ~ /^[0-9]+([.][0-9]+)?$/ && limit >= duration && limit <= 120) }'
then
    setup_failure dmtweeter-synco.zero_test.durationSec "$zero_limit" \
        "numeric seconds from requested $zero_seconds through 120"
fi
if (( zero_capture_failed ))
then
    printf 'Setup check failed; no enable request sent. Inspect before snapshots and status.txt.\n' >&2
    cleanup
    printf '%s\n' "$zero_capture_dir"
    exit 1
fi

if ! set_property -n "dmtweeter-synco.settings.steps=$zero_steps"
then
    cleanup
    printf '%s\n' "$zero_capture_dir"
    exit 1
fi
sleep 1
snapshot configured dmtweeter-synco
zero_synco_configured="$zero_capture_dir/dmtweeter-synco-indi-configured.txt"
if (( zero_capture_failed )) ||
   [[ $(read_property dmtweeter-synco.enable.toggle "$zero_synco_configured") != Off ||
      $(read_property dmtweeter-synco.settings.steps "$zero_synco_configured") != "$zero_steps" ]]
then
    printf 'Disabled step-count readback failed; no enable request sent.\n' >&2
    cleanup
    printf '%s\n' "$zero_capture_dir"
    exit 1
fi

# Mark ownership before requesting On, since a timeout can leave the request applied.
zero_started=1
zero_result=0
if ! set_property -s dmtweeter-synco.enable.toggle=On
then
    zero_result=1
else
    zero_warm_deadline=$((SECONDS + 30))
    zero_active_deadline=0
    zero_seen_progress=0
    zero_seen_active=0
    while true
    do
        zero_poll_status=0
        timeout -k 2s 8s getINDI dmtweeter-synco.zero_test \
            > "$zero_capture_dir/zero-test-current.txt" \
            2>> "$zero_capture_dir/zero-test-poll.err" || zero_poll_status=$?
        status "zero_test poll status=$zero_poll_status"
        cat "$zero_capture_dir/zero-test-current.txt" >> "$zero_capture_dir/zero-test-history.txt"
        if (( zero_poll_status != 0 ))
        then
            zero_result=1
            break
        fi
        zero_current="$zero_capture_dir/zero-test-current.txt"
        zero_current_active=$(read_property dmtweeter-synco.zero_test.active "$zero_current")
        zero_current_warming=$(read_property dmtweeter-synco.zero_test.warming "$zero_current")
        if [[ $zero_current_active == 1 || $zero_current_warming == 1 ]]
        then
            zero_seen_progress=1
        fi
        if [[ $zero_current_active == 1 ]]
        then
            zero_seen_active=1
        fi
        # An old finished readback can survive until the next INDI refresh after On.
        if (( zero_seen_progress )) &&
           [[ $(read_property dmtweeter-synco.zero_test.finished "$zero_current") == 1 ]]
        then
            zero_finished_elapsed=$(read_property dmtweeter-synco.zero_test.elapsedSec "$zero_current")
            if (( !zero_seen_active ))
            then
                status 'trial finished without reaching active qualification'
                zero_result=1
            elif ! awk -v elapsed="$zero_finished_elapsed" -v duration="$zero_seconds" \
                     'BEGIN { exit !(elapsed ~ /^[0-9]+([.][0-9]+)?$/ && elapsed >= duration) }'
            then
                status 'trial finished before the requested active duration'
                zero_result=1
            fi
            break
        fi
        if [[ $zero_current_active == 1 ]]
        then
            if (( zero_active_deadline == 0 ))
            then
                zero_active_deadline=$((SECONDS + zero_seconds))
            fi
            zero_active_elapsed=$(read_property dmtweeter-synco.zero_test.elapsedSec "$zero_current")
            if (( SECONDS >= zero_active_deadline )) ||
               awk -v elapsed="$zero_active_elapsed" -v duration="$zero_seconds" \
                   'BEGIN { exit !(elapsed ~ /^[0-9]+([.][0-9]+)?$/ && elapsed >= duration) }'
            then
                break
            fi
        elif (( zero_active_deadline != 0 || SECONDS >= zero_warm_deadline ))
        then
            status 'trial stopped early or failed to become active within 30 seconds'
            zero_result=1
            break
        fi
        sleep 1
    done
fi
if ! set_property -s dmtweeter-synco.enable.toggle=Off
then
    zero_result=1
else
    zero_started=0
fi
for zero_device in dmtweeter-synco dmtweeter camwfs holoop sysMonRTC.set_latency.toggle
do
    snapshot after "$zero_device"
done
cleanup
printf '%s\n' "$zero_capture_dir"
if (( zero_result != 0 ))
then
    exit "$zero_result"
fi
exit "$zero_capture_failed"
