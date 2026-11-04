#!/usr/bin/env bash
# Capture an operator-run finite, main-coordinated zero-auxiliary trial.
# Only synco's disabled steps/duration and finite-trial enable switch are changed.
set -u
set -o pipefail

usage()
{
    cat <<'HELP'
Usage: capture-zero-aux-test.sh RATE [SECONDS [ADVANCE_US [WAKE_AHEAD_US [COMMIT_LEAD_US]]]]

RATE is effective total cadence, 4000, 6000 or 8000 Hz, with the main loop at 2000 Hz.
SECONDS is the active-trial duration, default 120, integer 1 through 120.
ADVANCE_US must be 0: downstream dmcomb compensation is not used for counter targets.
WAKE_AHEAD_US is the expected startup validation lead, default 0, from 0 through 50 us.
COMMIT_LEAD_US is the expected startup final guard/API lead, default 0, from 0 through 10 us.
The app must report schema 8 or newer, targetEndpoint=1 (cnt0), and matching startup leads.
The app must already be running with zeroTest.enabled=true and enable=Off.
The script sets steps and the app-enforced duration while disabled, then starts the trial.
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
if (( $# < 1 || $# > 5 ))
then
    usage >&2
    exit 2
fi
zero_rate=$1
zero_seconds=${2:-120}
zero_advance=${3:-0}
zero_wake_ahead=${4:-0}
zero_commit_lead=${5:-0}
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
if ! awk -v advance="$zero_advance" \
     'BEGIN { exit !(advance ~ /^[0-9]+([.][0-9]+)?$/ && advance == 0) }'
then
    printf 'ADVANCE_US must be 0 for the cnt0 publication target\n' >&2
    exit 2
fi
if ! awk -v lead="$zero_commit_lead" \
     'BEGIN { exit !(lead ~ /^[0-9]+([.][0-9]+)?$/ && lead >= 0 && lead <= 10) }'
then
    printf 'COMMIT_LEAD_US must be numeric microseconds from 0 through 10\n' >&2
    exit 2
fi
if ! awk -v ahead="$zero_wake_ahead" \
     'BEGIN { exit !(ahead ~ /^[0-9]+([.][0-9]+)?$/ && ahead >= 0 && ahead <= 50) }'
then
    printf 'WAKE_AHEAD_US must be numeric microseconds from 0 through 50\n' >&2
    exit 2
fi

zero_capture_dir="$HOME/tmp/zero-aux-${zero_rate}Hz-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$zero_capture_dir" || exit 1
zero_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
git -C "$zero_script_dir/../../.." rev-parse HEAD > "$zero_capture_dir/git.txt" || exit 1
printf 'effectiveRate=%s\nsteps=%s\nactiveSeconds=%s\nmainRate=2000\ntargetEndpoint=cnt0\npublicationAdvanceUs=%s\nwakeAheadUs=%s\ncommitLeadUs=%s\n' \
    "$zero_rate" "$zero_steps" "$zero_seconds" "$zero_advance" "$zero_wake_ahead" "$zero_commit_lead" \
    > "$zero_capture_dir/test.txt"
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

check_phase_preset()
{
    local zero_file=$1
    local zero_phase=$2
    local zero_phase_schema zero_actual_advance zero_actual_late zero_actual_endpoint
    zero_phase_schema=$(read_property dmtweeter-synco.zero_test.labelVersion "$zero_file")
    zero_actual_endpoint=$(read_property dmtweeter-synco.zero_test.targetEndpoint "$zero_file")
    if ! awk -v schema="$zero_phase_schema" -v endpoint="$zero_actual_endpoint" \
         'BEGIN { exit !(schema ~ /^[0-9]+$/ && schema >= 8 &&
                         endpoint ~ /^[0-9]+([.][0-9]+)?$/ && endpoint == 1) }'
    then
        setup_failure "$zero_phase counter-target endpoint" \
            "schema=$zero_phase_schema targetEndpoint=$zero_actual_endpoint" \
            'schema>=8 and targetEndpoint=1 (cnt0)'
    fi
    zero_actual_advance=$(read_property dmtweeter-synco.zero_test.publicationAdvanceUs "$zero_file")
    zero_actual_late=$(read_property dmtweeter-synco.zero_test.lateUs "$zero_file")
    if ! awk -v schema="$zero_phase_schema" -v actual="$zero_actual_advance" \
             -v late="$zero_actual_late" -v expected="$zero_advance" \
         'BEGIN { number="^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$";
                  exit !(schema ~ /^[0-9]+$/ && schema >= 8 && expected == 0 &&
                         actual ~ number && actual == 0 && late ~ number && late >= 0 && late <= 50) }'
    then
        setup_failure "$zero_phase publication-advance preset" \
            "schema=$zero_phase_schema advanceUs=$zero_actual_advance lateUs=$zero_actual_late" \
            'advanceUs=0; schema>=8 and 0<=lateUs<=50'
    fi
    local zero_actual_wake
    zero_actual_wake=$(read_property dmtweeter-synco.zero_test.wakeAheadUs "$zero_file")
    if ! awk -v schema="$zero_phase_schema" -v actual="$zero_actual_wake" -v expected="$zero_wake_ahead" \
         'BEGIN { number="^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$";
                  exit !(schema ~ /^[0-9]+$/ && schema >= 8 &&
                         actual ~ number && actual == expected && actual >= 0 && actual <= 50) }'
    then
        setup_failure "$zero_phase wake-ahead preset" \
            "schema=$zero_phase_schema wakeAheadUs=$zero_actual_wake" \
            "wakeAheadUs=$zero_wake_ahead; schema>=8"
    fi
    local zero_actual_commit
    zero_actual_commit=$(read_property dmtweeter-synco.zero_test.commitLeadUs "$zero_file")
    if ! awk -v schema="$zero_phase_schema" -v actual="$zero_actual_commit" -v expected="$zero_commit_lead" \
         'BEGIN { number="^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$";
                  exit !(schema ~ /^[0-9]+$/ && schema >= 8 && actual ~ number &&
                         actual == expected && actual >= 0 && actual <= 10) }'
    then
        setup_failure "$zero_phase commit-lead preset" \
            "schema=$zero_phase_schema commitLeadUs=$zero_actual_commit" \
            "commitLeadUs=$zero_commit_lead; schema>=8"
    fi
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
zero_schema=$(read_property dmtweeter-synco.zero_test.labelVersion "$zero_synco_before")
zero_snapshot_valid=$(read_property dmtweeter-synco.zero_test.snapshotValid "$zero_synco_before")
zero_snapshot_age=$(read_property dmtweeter-synco.zero_test.snapshotAgeSec "$zero_synco_before")
zero_snapshot_sequence=$(read_property dmtweeter-synco.zero_test.snapshotSequence "$zero_synco_before")
zero_drops_before=$(read_property dmtweeter-synco.zero_test.eventsDropped "$zero_synco_before")
zero_log_failed_before=$(read_property dmtweeter-synco.logic_timing.eventsLogFailed "$zero_synco_before")
zero_limit=$(read_property dmtweeter-synco.zero_test.durationLimitSec "$zero_synco_before")
zero_camera_fps=$(read_property camwfs.fps.current "$zero_camera_before")
zero_camera_sync=$(read_property camwfs.synchro.toggle "$zero_camera_before")
zero_latency=$(read_property sysMonRTC.set_latency.toggle "$zero_latency_before")
[[ $zero_enabled == Off ]] || setup_failure dmtweeter-synco.enable.toggle "$zero_enabled" Off
[[ $zero_mode == 1 ]] || setup_failure dmtweeter-synco.zero_test.enabled "$zero_mode" 1
[[ $zero_ideal == 1 ]] || setup_failure dmtweeter-synco.zero_test.idealPublisher "$zero_ideal" 1
check_phase_preset "$zero_synco_before" before
if ! awk -v schema="$zero_schema" -v valid="$zero_snapshot_valid" -v age="$zero_snapshot_age" \
         -v seq="$zero_snapshot_sequence" -v drops="$zero_drops_before" -v failed="$zero_log_failed_before" \
     'BEGIN { number="^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$";
              exit !(schema ~ /^[0-9]+$/ && schema >= 8 && valid == 1 &&
                     age ~ number && age >= 0 && age <= 2 && seq ~ /^[0-9]+$/ && seq > 0 &&
                     drops ~ /^[0-9]+$/ && failed ~ /^[0-9]+$/) }'
then
    setup_failure 'zero_test progress snapshot' \
        "schema=$zero_schema valid=$zero_snapshot_valid age=$zero_snapshot_age sequence=$zero_snapshot_sequence" \
        'schema>=8, valid=1, age<=2 seconds and a positive sequence'
fi
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
    setup_failure dmtweeter-synco.zero_test.durationLimitSec "$zero_limit" \
        "numeric seconds from requested $zero_seconds through 120"
fi
if (( zero_capture_failed ))
then
    printf 'Setup check failed; no enable request sent. Inspect before snapshots and status.txt.\n' >&2
    cleanup
    printf '%s\n' "$zero_capture_dir"
    exit 1
fi

if ! set_property -n "dmtweeter-synco.settings.steps=$zero_steps" ||
   ! set_property -n "dmtweeter-synco.settings.durationSec=$zero_seconds"
then
    cleanup
    printf '%s\n' "$zero_capture_dir"
    exit 1
fi
sleep 1
snapshot configured dmtweeter-synco
zero_synco_configured="$zero_capture_dir/dmtweeter-synco-indi-configured.txt"
check_phase_preset "$zero_synco_configured" configured
zero_trial_before=$(read_property dmtweeter-synco.zero_test.trial "$zero_synco_configured")
zero_configured_duration=$(read_property dmtweeter-synco.zero_test.durationSec "$zero_synco_configured")
zero_configured_sequence=$(read_property dmtweeter-synco.zero_test.snapshotSequence "$zero_synco_configured")
if (( zero_capture_failed )) ||
   [[ $(read_property dmtweeter-synco.enable.toggle "$zero_synco_configured") != Off ||
      $(read_property dmtweeter-synco.settings.steps "$zero_synco_configured") != "$zero_steps" ]] ||
   ! awk -v trial="$zero_trial_before" -v selected="$zero_configured_duration" -v expected="$zero_seconds" \
         -v seq="$zero_configured_sequence" -v before="$zero_snapshot_sequence" \
     'BEGIN { exit !(trial ~ /^[0-9]+$/ && selected ~ /^[0-9]+([.][0-9]+)?$/ && selected == expected &&
                      seq ~ /^[0-9]+$/ && seq >= before) }'
then
    printf 'Disabled step-count/duration readback failed; no enable request sent.\n' >&2
    cleanup
    printf '%s\n' "$zero_capture_dir"
    exit 1
fi

# Exact elements let getINDI exit on one complete vector instead of its wildcard timeout.
zero_poll_fields=()
for zero_field in active warming finished requested trial mainObserved durationSec elapsedSec \
                  snapshotSequence snapshotValid snapshotAgeSec eventsDropped
do
    zero_poll_fields+=("dmtweeter-synco.zero_test.$zero_field")
done
zero_expected_trial=$((zero_trial_before + 1))
# Mark ownership before requesting On, since a timeout can leave the request applied.
zero_started=1
zero_result=0
if ! set_property -s dmtweeter-synco.enable.toggle=On
then
    zero_result=1
else
    zero_warm_deadline=$((SECONDS + 30))
    zero_active_deadline=0
    zero_seen_trial=0
    zero_last_sequence=$zero_configured_sequence
    zero_sequence_deadline=$((SECONDS + 5))
    while true
    do
        zero_poll_status=0
        timeout -k 2s 8s getINDI "${zero_poll_fields[@]}" \
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
        zero_current_sequence=$(read_property dmtweeter-synco.zero_test.snapshotSequence "$zero_current")
        zero_current_valid=$(read_property dmtweeter-synco.zero_test.snapshotValid "$zero_current")
        zero_current_age=$(read_property dmtweeter-synco.zero_test.snapshotAgeSec "$zero_current")
        zero_current_drops=$(read_property dmtweeter-synco.zero_test.eventsDropped "$zero_current")
        if ! awk -v seq="$zero_current_sequence" -v valid="$zero_current_valid" -v age="$zero_current_age" \
             'BEGIN { exit !(seq ~ /^[0-9]+$/ && seq > 0 && valid == 1 &&
                              age ~ /^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$/ && age >= 0 && age <= 2) }'
        then
            status 'progress snapshot missing, invalid or older than 2 seconds'
            zero_result=1
            break
        fi
        if [[ -n $zero_last_sequence ]] &&
           awk -v old="$zero_last_sequence" -v current="$zero_current_sequence" 'BEGIN { exit !(current < old) }'
        then
            status 'progress snapshot sequence regressed; app identity changed'
            zero_result=1
            break
        fi
        if [[ $zero_current_sequence != "$zero_last_sequence" ]]
        then
            zero_last_sequence=$zero_current_sequence
            zero_sequence_deadline=$((SECONDS + 5))
        elif (( SECONDS >= zero_sequence_deadline ))
        then
            status 'progress snapshot sequence did not advance within 5 seconds'
            zero_result=1
            break
        fi
        if ! awk -v before="$zero_drops_before" -v current="$zero_current_drops" \
             'BEGIN { exit !(current ~ /^[0-9]+$/ && current <= before) }'
        then
            status 'new zero-event queue loss; capture is incomplete'
            zero_result=1
            break
        fi
        zero_current_trial=$(read_property dmtweeter-synco.zero_test.trial "$zero_current")
        if ! [[ $zero_current_trial =~ ^[0-9]+$ ]]
        then
            status 'missing trial identity'
            zero_result=1
            break
        fi
        # Old finished/active readbacks can survive On; only the exact new generation counts.
        if [[ $zero_current_trial == "$zero_trial_before" ]] && (( !zero_seen_trial ))
        then
            if (( SECONDS >= zero_warm_deadline ))
            then
                status 'new trial failed to appear within 30 seconds'
                zero_result=1
                break
            fi
            sleep 1
            continue
        fi
        if [[ $zero_current_trial != "$zero_expected_trial" ]]
        then
            status 'unexpected trial identity; capture is incomplete'
            zero_result=1
            break
        fi
        zero_seen_trial=1
        zero_current_duration=$(read_property dmtweeter-synco.zero_test.durationSec "$zero_current")
        zero_current_elapsed=$(read_property dmtweeter-synco.zero_test.elapsedSec "$zero_current")
        if ! awk -v selected="$zero_current_duration" -v expected="$zero_seconds" -v elapsed="$zero_current_elapsed" \
             'BEGIN { number="^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$";
                      exit !(selected ~ number && selected == expected && elapsed ~ number && elapsed >= 0) }'
        then
            status 'app duration differs from the requested trial'
            zero_result=1
            break
        fi
        zero_current_active=$(read_property dmtweeter-synco.zero_test.active "$zero_current")
        zero_current_warming=$(read_property dmtweeter-synco.zero_test.warming "$zero_current")
        if [[ $(read_property dmtweeter-synco.zero_test.finished "$zero_current") == 1 ]]
        then
            zero_current_mains=$(read_property dmtweeter-synco.zero_test.mainObserved "$zero_current")
            if ! [[ $zero_current_mains =~ ^[0-9]+$ ]] || (( zero_current_mains == 0 ))
            then
                status 'trial finished without reaching active qualification'
                zero_result=1
            elif [[ $zero_current_active != 0 || $zero_current_warming != 0 ||
                    $(read_property dmtweeter-synco.zero_test.requested "$zero_current") != 0 ]]
            then
                status 'finished trial has inconsistent active/requested state'
                zero_result=1
            elif ! awk -v elapsed="$zero_current_elapsed" -v duration="$zero_seconds" \
                     'BEGIN { exit !(elapsed ~ /^[0-9]+([.][0-9]+)?$/ && elapsed >= duration) }'
            then
                status 'trial finished before the requested active duration'
                zero_result=1
            elif awk -v elapsed="$zero_current_elapsed" -v duration="$zero_seconds" \
                     'BEGIN { exit !(elapsed > duration+5) }'
            then
                status 'trial finished beyond the app duration plus reporting grace'
                zero_result=1
            fi
            break
        fi
        if [[ $zero_current_active == 1 ]]
        then
            if (( zero_active_deadline == 0 ))
            then
                zero_remaining=$(awk -v elapsed="$zero_current_elapsed" -v duration="$zero_seconds" \
                    'BEGIN { left=duration-elapsed; print (left>0 ? int(left)+1 : 0) }')
                zero_active_deadline=$((SECONDS + zero_remaining + 5))
            fi
            if (( SECONDS >= zero_active_deadline )) ||
               awk -v elapsed="$zero_current_elapsed" -v duration="$zero_seconds" \
                   'BEGIN { exit !(elapsed ~ /^[0-9]+([.][0-9]+)?$/ && elapsed > duration+5) }'
            then
                status 'app did not report completion within the duration plus reporting grace'
                zero_result=1
                break
            fi
        elif [[ $zero_current_warming != 1 ]] ||
             (( zero_active_deadline != 0 || SECONDS >= zero_warm_deadline ))
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
zero_synco_after="$zero_capture_dir/dmtweeter-synco-indi-after.txt"
zero_drops_after=$(read_property dmtweeter-synco.zero_test.eventsDropped "$zero_synco_after")
zero_log_failed_after=$(read_property dmtweeter-synco.logic_timing.eventsLogFailed "$zero_synco_after")
if ! awk -v before="$zero_drops_before" -v current="$zero_drops_after" \
         -v logBefore="$zero_log_failed_before" -v logCurrent="$zero_log_failed_after" \
     'BEGIN { exit !(current ~ /^[0-9]+$/ && current == before &&
                      logCurrent ~ /^[0-9]+$/ && logCurrent == logBefore) }'
then
    status 'final zero-event transport loss or missing quality readback; capture is incomplete'
    zero_result=1
fi
cleanup
printf '%s\n' "$zero_capture_dir"
if (( zero_result != 0 ))
then
    exit "$zero_result"
fi
exit "$zero_capture_failed"
