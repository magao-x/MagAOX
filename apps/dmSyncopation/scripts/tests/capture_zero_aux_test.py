#!/usr/bin/env python3
r"""\file capture_zero_aux_test.py
\brief Offline coordinated-trial capture tests with private fake INDI/services.
"""

import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "capture-zero-aux-test.sh"


class CaptureZeroAuxTest(unittest.TestCase):
    """Exercise control ownership and capture cleanup without an INDI connection."""

    def setUp(self):
        """Install fake services that can only change this fixture's files."""
        self.temp = tempfile.TemporaryDirectory(prefix="synco-zero-capture-")
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.env = dict(os.environ)
        self.env.update(
            HOME=str(self.root),
            PATH=f"{self.bin}:/usr/bin:/bin",
            ZERO_TEST_ROOT=str(self.root),
        )
        (self.root / "enable").write_text("Off")
        (self.root / "steps").write_text("2")
        (self.root / "duration").write_text("120")
        self.stub("timeout", 'while [[ $1 == -* ]]; do shift 2; done\nshift\nexec "$@"')
        self.stub("git", "printf 'offline-test-commit\\n'")
        self.stub("sleep", "/bin/sleep 0.02")
        self.stub(
            "logdump",
            'printf "%s\\n" "$$" >> "$ZERO_TEST_ROOT/collector-pids"\n'
            'if [[ ${ZERO_TEST_COLLECTOR_STATUS:-0} != 0 ]]; then\n'
            '    exit "$ZERO_TEST_COLLECTOR_STATUS"\nfi\n'
            'trap "exit 0" TERM INT\nprintf "fake telemetry\\n"\n'
            'while true; do /bin/sleep 0.02; done',
        )
        self.stub(
            "getINDI",
            r'''printf "%s\n" "$*" >> "$ZERO_TEST_ROOT/indi-calls"
/bin/sleep 0.02
[[ ${ZERO_TEST_INDI_STATUS:-0} != 0 ]] && exit "$ZERO_TEST_INDI_STATUS"
enable=$(cat "$ZERO_TEST_ROOT/enable")
steps=$(cat "$ZERO_TEST_ROOT/steps")
duration=$(cat "$ZERO_TEST_ROOT/duration")
advance=${ZERO_TEST_ADVANCE-0}
late=${ZERO_TEST_LATE-5}
wake_ahead=${ZERO_TEST_WAKE_AHEAD-0}
commit_lead=${ZERO_TEST_COMMIT_LEAD-0}
endpoint=${ZERO_TEST_TARGET_ENDPOINT-1}
schema=${ZERO_TEST_SCHEMA-9}
if [[ -f $ZERO_TEST_ROOT/set-calls ]]; then
    advance=${ZERO_TEST_CONFIGURED_ADVANCE-$advance}
    late=${ZERO_TEST_CONFIGURED_LATE-$late}
    wake_ahead=${ZERO_TEST_CONFIGURED_WAKE_AHEAD-$wake_ahead}
    commit_lead=${ZERO_TEST_CONFIGURED_COMMIT_LEAD-$commit_lead}
    endpoint=${ZERO_TEST_CONFIGURED_TARGET_ENDPOINT-$endpoint}
    schema=${ZERO_TEST_CONFIGURED_SCHEMA-$schema}
fi
case $1 in
    dmtweeter-synco)
        printf "dmtweeter-synco.enable.toggle=%s\n" "$enable"
        printf "dmtweeter-synco.settings.steps=%s\n" "$steps"
        printf "dmtweeter-synco.settings.durationSec=%s\n" "$duration"
        printf "dmtweeter-synco.zero_test.enabled=%s\n" "${ZERO_TEST_MODE-1}"
        printf "dmtweeter-synco.zero_test.idealPublisher=%s\n" "${ZERO_TEST_IDEAL-1}"
        printf "dmtweeter-synco.zero_test.labelVersion=%s\n" "$schema"
        printf "dmtweeter-synco.zero_test.publicationAdvanceUs=%s\n" "$advance"
        printf "dmtweeter-synco.zero_test.lateUs=%s\n" "$late"
        printf "dmtweeter-synco.zero_test.wakeAheadUs=%s\n" "$wake_ahead"
        printf "dmtweeter-synco.zero_test.commitLeadUs=%s\n" "$commit_lead"
        printf "dmtweeter-synco.zero_test.targetEndpoint=%s\n" "$endpoint"
        printf "dmtweeter-synco.zero_test.snapshotValid=1\n"
        printf "dmtweeter-synco.zero_test.snapshotAgeSec=0\n"
        printf "dmtweeter-synco.zero_test.snapshotSequence=%s\n" "${ZERO_TEST_BASELINE_SEQUENCE-1}"
        printf "dmtweeter-synco.zero_test.eventsDropped=0\n"
        printf "dmtweeter-synco.logic_timing.eventsLogFailed=%s\n" \
            "$(if [[ -f $ZERO_TEST_ROOT/polls ]]; then printf %s "${ZERO_TEST_LOG_FAILED-0}"; else printf 0; fi)"
        printf "dmtweeter-synco.zero_test.durationSec=%s\n" "$duration"
        printf "dmtweeter-synco.zero_test.durationLimitSec=%s\n" "${ZERO_TEST_LIMIT-120}"
        printf "dmtweeter-synco.zero_test.trial=%s\n" "${ZERO_TEST_BASELINE_TRIAL-0}"
        ;;
    dmtweeter-synco.zero_test.active)
        [[ ${ZERO_TEST_POLL_STATUS:-0} != 0 ]] && exit "$ZERO_TEST_POLL_STATUS"
        count=0
        [[ -f $ZERO_TEST_ROOT/polls ]] && count=$(cat "$ZERO_TEST_ROOT/polls")
        count=$((count+1))
        printf "%s" "$count" > "$ZERO_TEST_ROOT/polls"
        sequence=$((count+1))
        trial=${ZERO_TEST_CURRENT_TRIAL-$(( ${ZERO_TEST_BASELINE_TRIAL:-0} + 1 ))}
        if [[ ${ZERO_TEST_FROZEN_SNAPSHOT:-0} == 1 ]]; then sequence=2; fi
        if [[ ${ZERO_TEST_STALE_FINISHED:-0} == 1 ]]; then
            if ((count==1)); then
                count=2
                trial=${ZERO_TEST_BASELINE_TRIAL:-0}
            else
                count=$((count-1))
            fi
        fi
        if [[ ${ZERO_TEST_HOLD_ACTIVE:-0} == 1 || ${ZERO_TEST_FROZEN_SNAPSHOT:-0} == 1 ]]; then count=1; fi
        if [[ ${ZERO_TEST_FINISH_FIRST:-0} == 1 ]]; then count=2; fi
        active=$((count==1))
        warming=0
        if [[ ${ZERO_TEST_WARM_TIMEOUT:-0} == 1 ]]; then
            active=0
            warming=$((count==1))
        fi
        elapsed=${ZERO_TEST_ELAPSED:-0}
        if ((count>=2)); then elapsed=${ZERO_TEST_FINISHED_ELAPSED-$duration}; fi
        if [[ ${ZERO_TEST_WARM_TIMEOUT:-0} == 1 ]]; then elapsed=0; fi
        mains=2000
        if [[ ${ZERO_TEST_WARM_TIMEOUT:-0} == 1 ]]; then mains=0; fi
        printf "dmtweeter-synco.zero_test.active=%s\n" "$active"
        printf "dmtweeter-synco.zero_test.requested=%s\n" "$((count<2))"
        printf "dmtweeter-synco.zero_test.trial=%s\n" "$trial"
        printf "dmtweeter-synco.zero_test.mainObserved=%s\n" "$mains"
        printf "dmtweeter-synco.zero_test.durationSec=%s\n" "${ZERO_TEST_CURRENT_DURATION-$duration}"
        printf "dmtweeter-synco.zero_test.snapshotSequence=%s\n" "$sequence"
        printf "dmtweeter-synco.zero_test.snapshotValid=%s\n" "${ZERO_TEST_SNAPSHOT_VALID-1}"
        printf "dmtweeter-synco.zero_test.snapshotAgeSec=%s\n" "${ZERO_TEST_SNAPSHOT_AGE-0}"
        printf "dmtweeter-synco.zero_test.eventsDropped=%s\n" "${ZERO_TEST_EVENT_DROPS-0}"
        printf "dmtweeter-synco.zero_test.warming=%s\n" "$warming"
        printf "dmtweeter-synco.zero_test.elapsedSec=%s\n" "$elapsed"
        printf "dmtweeter-synco.zero_test.finished=%s\n" "$((count>=2))"
        ;;
    camwfs)
        printf "camwfs.fps.current=%s\ncamwfs.synchro.toggle=%s\n" \
            "${ZERO_TEST_FPS-2000}" "${ZERO_TEST_SYNC-On}"
        ;;
    sysMonRTC.set_latency.toggle)
        printf "sysMonRTC.set_latency.toggle=%s\n" "${ZERO_TEST_LATENCY-On}"
        ;;
    *) printf "fake.current=1\n" ;;
esac''',
        )
        self.stub(
            "setINDI",
            'printf "%s\\n" "$*" >> "$ZERO_TEST_ROOT/set-calls"\n'
            'case $2 in\n'
            'dmtweeter-synco.settings.steps=*)\n'
            ' [[ $(cat "$ZERO_TEST_ROOT/enable") == Off ]] || exit 9\n'
            ' [[ ${ZERO_TEST_IGNORE_STEPS:-0} == "${2##*=}" ]] && exit 0\n'
            ' printf "%s" "${2##*=}" > "$ZERO_TEST_ROOT/steps";;\n'
            'dmtweeter-synco.settings.durationSec=*)\n'
            ' [[ $(cat "$ZERO_TEST_ROOT/enable") == Off ]] || exit 9\n'
            ' [[ ${ZERO_TEST_IGNORE_DURATION:-0} == 1 ]] && exit 0\n'
            ' printf "%s" "${2##*=}" > "$ZERO_TEST_ROOT/duration";;\n'
            'dmtweeter-synco.enable.toggle=On)\n'
            ' printf On > "$ZERO_TEST_ROOT/enable"\n'
            ' exit "${ZERO_TEST_ON_STATUS:-0}";;\n'
            'dmtweeter-synco.enable.toggle=Off) printf Off > "$ZERO_TEST_ROOT/enable";;\n'
            '*) exit 10;;\nesac',
        )

    def tearDown(self):
        """Remove only private mock data."""
        self.temp.cleanup()

    def stub(self, name, body):
        """Create an executable fake service in the private PATH."""
        path = self.bin / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)

    def run_capture(self, *args, interrupt=False):
        """Bound each offline test and kill its private session on a hang."""
        process = subprocess.Popen(
            ["bash", str(SCRIPT), *args],
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            if interrupt:
                deadline = time.monotonic() + 5
                while not (self.root / "polls").exists() and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertTrue((self.root / "polls").exists())
                process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            self.fail("Mock capture did not finish or stop its owned collectors")
        pids = self.root / "collector-pids"
        if pids.exists():
            for line in pids.read_text().splitlines():
                with self.assertRaises(ProcessLookupError):
                    os.kill(int(line), 0)
        return process.returncode, stdout, stderr

    def set_calls(self):
        """Return requested mock mutations, if any."""
        path = self.root / "set-calls"
        return path.read_text().splitlines() if path.exists() else []

    def capture_dir(self):
        """Return this fixture's single result directory."""
        directories = list((self.root / "tmp").glob("zero-aux-*"))
        self.assertEqual(len(directories), 1)
        return directories[0]

    def test_four_khz_requests_one_auxiliary_per_main(self):
        """A finite 4 kHz total test changes only disabled steps and its own enable."""
        code, _, stderr = self.run_capture("4000", "1")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(
            self.set_calls(),
            ["-n dmtweeter-synco.settings.steps=2", "-n dmtweeter-synco.settings.durationSec=1",
             "-s dmtweeter-synco.enable.toggle=On",
             "-s dmtweeter-synco.enable.toggle=Off"],
        )
        directory = self.capture_dir()
        self.assertIn("effectiveRate=4000", (directory / "test.txt").read_text())
        self.assertTrue((directory / "dmtweeter-synco.txt").exists())
        self.assertTrue((directory / "zero-test-history.txt").exists())
        self.assertEqual((self.root / "enable").read_text(), "Off")

    def test_six_khz_requests_two_auxiliaries_per_main(self):
        """The higher target selects three total steps, not 6 kHz extra writes."""
        code, _, stderr = self.run_capture("6000", "1")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(self.set_calls()[0], "-n dmtweeter-synco.settings.steps=3")
        self.assertIn("steps=3", (self.capture_dir() / "test.txt").read_text())

    def test_eight_khz_requests_three_auxiliaries_per_main(self):
        """An 8 kHz total trial selects four steps while preserving the 2 kHz main."""
        code, _, stderr = self.run_capture("8000", "10")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(
            self.set_calls(),
            ["-n dmtweeter-synco.settings.steps=4", "-n dmtweeter-synco.settings.durationSec=10",
             "-s dmtweeter-synco.enable.toggle=On",
             "-s dmtweeter-synco.enable.toggle=Off"],
        )
        metadata = (self.capture_dir() / "test.txt").read_text()
        self.assertIn("steps=4", metadata)
        self.assertIn("effectiveRate=8000", metadata)
        self.assertIn("activeSeconds=10", metadata)
        self.assertIn("mainRate=2000", metadata)

    def test_eight_khz_requires_four_step_readback(self):
        """An old app rejecting four steps is never enabled by a successful client request."""
        self.env["ZERO_TEST_IGNORE_STEPS"] = "4"
        code, _, stderr = self.run_capture("8000", "10")
        self.assertEqual(code, 1)
        self.assertEqual(self.set_calls(), ["-n dmtweeter-synco.settings.steps=4",
                                           "-n dmtweeter-synco.settings.durationSec=10"])
        self.assertEqual((self.root / "enable").read_text(), "Off")
        self.assertIn("Disabled step-count/duration readback failed", stderr)

    def test_counter_target_trial_records_current_endpoint(self):
        """The current endpoint is cnt0 publication with no downstream phase advance."""
        code, _, stderr = self.run_capture("8000", "10", "0.0")
        self.assertEqual(code, 0, stderr)
        metadata = (self.capture_dir() / "test.txt").read_text()
        self.assertIn("targetEndpoint=cnt0", metadata)
        self.assertIn("publicationAdvanceUs=0.0", metadata)
        self.assertIn("commitLeadUs=0", metadata)

    def test_counter_target_rejects_all_previous_endpoint_schemas(self):
        """Older schemas lack current clock-rejection evidence and recovery accounting."""
        for schema in ["5", "6", "7", "8"]:
            with self.subTest(schema=schema):
                self.env["ZERO_TEST_SCHEMA"] = schema
                code, _, stderr = self.run_capture("8000", "10")
                self.assertEqual(code, 1)
                self.assertEqual(self.set_calls(), [])
                self.assertIn("before counter-target endpoint", stderr)

    def test_schema_is_rechecked_before_enable(self):
        """A stale schema-8 configured snapshot cannot bypass the recovery capability gate."""
        self.env["ZERO_TEST_CONFIGURED_SCHEMA"] = "8"
        code, _, stderr = self.run_capture("8000", "10")
        self.assertEqual(code, 1)
        self.assertEqual(
            self.set_calls(),
            ["-n dmtweeter-synco.settings.steps=4", "-n dmtweeter-synco.settings.durationSec=10"],
        )
        self.assertEqual((self.root / "enable").read_text(), "Off")
        self.assertIn("configured counter-target endpoint", stderr)
        self.assertIn("schema>=9", stderr)

    def test_counter_target_requires_explicit_endpoint_kind(self):
        """Schema capability alone cannot identify the target's meaning."""
        for endpoint in ["0", "2", "", "nan", "inf", "cnt0"]:
            with self.subTest(endpoint=endpoint):
                self.env["ZERO_TEST_TARGET_ENDPOINT"] = endpoint
                code, _, stderr = self.run_capture("8000", "10")
                self.assertEqual(code, 1)
                self.assertEqual(self.set_calls(), [])
                self.assertIn("before counter-target endpoint", stderr)

    def test_commit_lead_checks_numeric_startup_preset(self):
        """A measured small API lead is verified without changing startup configuration."""
        self.env.update(ZERO_TEST_COMMIT_LEAD="5e-1", ZERO_TEST_LATE="2", ZERO_TEST_WAKE_AHEAD="10")
        code, _, stderr = self.run_capture("8000", "10", "0", "10", "0.50")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(
            self.set_calls(),
            ["-n dmtweeter-synco.settings.steps=4", "-n dmtweeter-synco.settings.durationSec=10",
             "-s dmtweeter-synco.enable.toggle=On", "-s dmtweeter-synco.enable.toggle=Off"],
        )
        metadata = (self.capture_dir() / "test.txt").read_text()
        self.assertIn("commitLeadUs=0.50", metadata)
        self.assertIn("targetEndpoint=cnt0", metadata)

    def test_commit_lead_rejects_wrong_missing_or_nonfinite_readback(self):
        """The expected API lead must match a finite bounded startup value."""
        self.env["ZERO_TEST_WAKE_AHEAD"] = "10"
        for lead in ["0", "0.4", "", "nan", "inf", "-1", "11"]:
            with self.subTest(lead=lead):
                self.env["ZERO_TEST_COMMIT_LEAD"] = lead
                code, _, stderr = self.run_capture("8000", "10", "0", "10", "0.5")
                self.assertEqual(code, 1)
                self.assertEqual(self.set_calls(), [])
                self.assertIn("before commit-lead preset", stderr)

    def test_omitted_commit_lead_cannot_silently_select_nonzero_setting(self):
        """An omitted fifth argument explicitly expects an uncalibrated zero API lead."""
        self.env["ZERO_TEST_COMMIT_LEAD"] = "0.5"
        code, _, stderr = self.run_capture("8000", "10")
        self.assertEqual(code, 1)
        self.assertEqual(self.set_calls(), [])
        self.assertIn("commitLeadUs=0", stderr)

    def test_commit_lead_is_rechecked_before_enable(self):
        """A changed API-lead readback after disabled settings prevents enabling."""
        self.env.update(ZERO_TEST_COMMIT_LEAD="0.5", ZERO_TEST_CONFIGURED_COMMIT_LEAD="0.4")
        code, _, stderr = self.run_capture("8000", "10", "0", "0", "0.5")
        self.assertEqual(code, 1)
        self.assertNotIn("-s dmtweeter-synco.enable.toggle=On", self.set_calls())
        self.assertIn("configured commit-lead preset", stderr)

    def test_counter_endpoint_is_rechecked_before_enable(self):
        """The configured snapshot must retain the counter-publication endpoint."""
        self.env["ZERO_TEST_CONFIGURED_TARGET_ENDPOINT"] = "0"
        code, _, stderr = self.run_capture("8000", "10")
        self.assertEqual(code, 1)
        self.assertNotIn("-s dmtweeter-synco.enable.toggle=On", self.set_calls())
        self.assertIn("configured counter-target endpoint", stderr)

    def test_early_validation_trial_checks_numeric_startup_preset(self):
        """A counter-target trial verifies a 10 us validation lead numerically."""
        self.env.update(ZERO_TEST_LATE="2", ZERO_TEST_WAKE_AHEAD="10")
        code, _, stderr = self.run_capture("8000", "10", "0", "10.0")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(
            self.set_calls(),
            ["-n dmtweeter-synco.settings.steps=4", "-n dmtweeter-synco.settings.durationSec=10",
             "-s dmtweeter-synco.enable.toggle=On", "-s dmtweeter-synco.enable.toggle=Off"],
        )
        metadata = (self.capture_dir() / "test.txt").read_text()
        self.assertIn("publicationAdvanceUs=0", metadata)
        self.assertIn("wakeAheadUs=10.0", metadata)

    def test_early_validation_rejects_wrong_or_missing_preset(self):
        """The current capability still requires its exact finite validation lead."""
        for value in ["0", "9", "", "nan", "inf", "-1", "51"]:
            with self.subTest(value=value):
                self.env["ZERO_TEST_WAKE_AHEAD"] = value
                code, _, stderr = self.run_capture("8000", "10", "0", "10")
                self.assertEqual(code, 1)
                self.assertEqual(self.set_calls(), [])
                self.assertIn("before wake-ahead preset", stderr)

    def test_omitted_validation_lead_cannot_silently_select_new_policy(self):
        """The fourth argument must explicitly select a configured nonzero validation lead."""
        self.env["ZERO_TEST_WAKE_AHEAD"] = "10"
        code, _, stderr = self.run_capture("8000", "10", "0")
        self.assertEqual(code, 1)
        self.assertEqual(self.set_calls(), [])
        self.assertIn("wakeAheadUs=0", stderr)

    def test_new_schema_preserves_zero_validation_lead(self):
        """The new publication grid can retain the zero-lead waiting policy."""
        code, _, stderr = self.run_capture("8000", "10")
        self.assertEqual(code, 0, stderr)
        self.assertIn("wakeAheadUs=0", (self.capture_dir() / "test.txt").read_text())

    def test_validation_lead_is_rechecked_before_enable(self):
        """The configured snapshot cannot silently lose the expected validation lead."""
        self.env.update(ZERO_TEST_WAKE_AHEAD="10", ZERO_TEST_CONFIGURED_WAKE_AHEAD="0")
        code, _, stderr = self.run_capture("8000", "10", "0", "10")
        self.assertEqual(code, 1)
        self.assertNotIn("-s dmtweeter-synco.enable.toggle=On", self.set_calls())
        self.assertIn("configured wake-ahead preset", stderr)

    def test_nonzero_downstream_advance_readback_never_enables(self):
        """The counter-target app must not retain the former downstream compensation."""
        for value in ["5", "", "nan", "inf", "-1"]:
            with self.subTest(value=value):
                self.env["ZERO_TEST_ADVANCE"] = value
                code, _, stderr = self.run_capture("8000", "10")
                self.assertEqual(code, 1)
                self.assertEqual(self.set_calls(), [])
                self.assertIn("before publication-advance preset", stderr)

    def test_admission_grace_requires_finite_bounded_readback(self):
        """The zero advance does not remove the independent admission grace checks."""
        for value in ["-1", "51", "", "nan", "inf"]:
            with self.subTest(value=value):
                self.env["ZERO_TEST_LATE"] = value
                code, _, stderr = self.run_capture("8000", "10")
                self.assertEqual(code, 1)
                self.assertEqual(self.set_calls(), [])
                self.assertIn("before publication-advance preset", stderr)

    def test_preset_is_checked_again_before_enable(self):
        """A changed startup advance cannot reintroduce the wrong endpoint policy."""
        self.env["ZERO_TEST_CONFIGURED_ADVANCE"] = "5"
        code, _, stderr = self.run_capture("8000", "10")
        self.assertEqual(code, 1)
        self.assertNotIn("-s dmtweeter-synco.enable.toggle=On", self.set_calls())
        self.assertIn("configured publication-advance preset", stderr)

    def test_progress_snapshot_capability_is_required(self):
        """The earlier mutex-dependent status implementation cannot start this capture."""
        self.env["ZERO_TEST_SCHEMA"] = "3"
        code, _, stderr = self.run_capture("8000", "10")
        self.assertEqual(code, 1)
        self.assertEqual(self.set_calls(), [])
        self.assertIn("progress snapshot", stderr)

    def test_invalid_or_aged_progress_cancels_owned_trial(self):
        """Freshness failures are reported instead of waiting through a false warmup."""
        for key, value in [("ZERO_TEST_SNAPSHOT_VALID", "0"), ("ZERO_TEST_SNAPSHOT_AGE", "3")]:
            with self.subTest(key=key):
                self.env[key] = value
                code, _, _ = self.run_capture("8000", "10")
                self.assertEqual(code, 1)
                self.assertEqual(self.set_calls()[-1], "-s dmtweeter-synco.enable.toggle=Off")
                del self.env[key]

    def test_frozen_progress_sequence_cancels_owned_trial(self):
        """A cached active flag and cached small age cannot hide a frozen status publisher."""
        self.env["ZERO_TEST_FROZEN_SNAPSHOT"] = "1"
        code, _, _ = self.run_capture("8000", "120")
        self.assertEqual(code, 1)
        self.assertEqual(self.set_calls()[-1], "-s dmtweeter-synco.enable.toggle=Off")
        self.assertIn("sequence did not advance", (self.capture_dir() / "status.txt").read_text())

    def test_new_event_loss_marks_capture_incomplete(self):
        """Intentional sampling is allowed, while a new queue overflow aborts qualification."""
        self.env["ZERO_TEST_EVENT_DROPS"] = "1"
        code, _, _ = self.run_capture("8000", "10")
        self.assertEqual(code, 1)
        self.assertEqual(self.set_calls()[-1], "-s dmtweeter-synco.enable.toggle=Off")
        self.assertIn("new zero-event queue loss", (self.capture_dir() / "status.txt").read_text())

    def test_preexisting_enabled_run_is_not_cancelled(self):
        """Refuse a run owned by another operator without sending any mutation."""
        (self.root / "enable").write_text("On")
        code, _, stderr = self.run_capture("4000", "1")
        self.assertEqual(code, 1)
        self.assertEqual(self.set_calls(), [])
        self.assertEqual((self.root / "enable").read_text(), "On")
        self.assertIn('dmtweeter-synco.enable.toggle actual="On"; expected="Off"', stderr)

    def test_wrong_camera_or_mode_never_enables(self):
        """Startup defaults or a missing test binary cannot start a trial."""
        for key, value in [("ZERO_TEST_FPS", "3622"), ("ZERO_TEST_MODE", "0")]:
            with self.subTest(key=key):
                self.env[key] = value
                code, _, _ = self.run_capture("4000", "1")
                self.assertEqual(code, 1)
                self.assertEqual(self.set_calls(), [])
                del self.env[key]

    def test_baseline_reader_failure_never_enables(self):
        """A failed telemetry reader prevents the write request."""
        self.env["ZERO_TEST_COLLECTOR_STATUS"] = "255"
        code, _, stderr = self.run_capture("4000", "1")
        self.assertEqual(code, 1)
        self.assertEqual(self.set_calls(), [])
        self.assertIn("status=255", (self.capture_dir() / "status.txt").read_text())
        self.assertIn("telemetry collector dmtweeter-synco (pid ", stderr)
        self.assertIn('actual="not running"; expected="running"', stderr)

    def test_ideal_publisher_capability_is_required(self):
        """An old or non-ideal publisher cannot silently repeat the previous trial."""
        for value in ["0", ""]:
            with self.subTest(value=value):
                self.env["ZERO_TEST_IDEAL"] = value
                code, _, stderr = self.run_capture("4000", "1")
                self.assertEqual(code, 1)
                self.assertEqual(self.set_calls(), [])
                self.assertIn("dmtweeter-synco.zero_test.idealPublisher", stderr)

    def test_setup_reports_every_mismatched_gate(self):
        """Print all actual/expected property mismatches without requesting writes."""
        (self.root / "enable").write_text("On")
        self.env.update(
            ZERO_TEST_MODE="0", ZERO_TEST_LIMIT="0", ZERO_TEST_FPS="3622",
            ZERO_TEST_SYNC="Off", ZERO_TEST_LATENCY="Off",
        )
        code, _, stderr = self.run_capture("4000", "60")
        self.assertEqual(code, 1)
        self.assertEqual(self.set_calls(), [])
        messages = [
            'dmtweeter-synco.enable.toggle actual="On"; expected="Off"',
            'dmtweeter-synco.zero_test.enabled actual="0"; expected="1"',
            'camwfs.synchro.toggle actual="Off"; expected="On"',
            'sysMonRTC.set_latency.toggle actual="Off"; expected="On"',
            'camwfs.fps.current actual="3622"; expected="numeric FPS from 1999 through 2001"',
            'dmtweeter-synco.zero_test.durationLimitSec actual="0"; '
            'expected="numeric seconds from requested 60 through 120"',
        ]
        saved = (self.capture_dir() / "status.txt").read_text()
        for message in messages:
            with self.subTest(message=message):
                self.assertIn(message, stderr)
                self.assertIn(message, saved)

    def test_setup_reports_missing_numeric_values(self):
        """Missing numeric readbacks name both failed gates explicitly."""
        self.env.update(ZERO_TEST_LIMIT="", ZERO_TEST_FPS="")
        code, _, stderr = self.run_capture("6000", "60")
        self.assertEqual(code, 1)
        self.assertEqual(self.set_calls(), [])
        self.assertIn('camwfs.fps.current actual="<missing>"', stderr)
        self.assertIn('dmtweeter-synco.zero_test.durationLimitSec actual="<missing>"', stderr)

    def test_setup_reports_duration_bounds(self):
        """Preserve the requested-duration and 120-second upper-limit gates."""
        for value in ["30", "121", "nan"]:
            with self.subTest(value=value):
                self.env["ZERO_TEST_LIMIT"] = value
                code, _, stderr = self.run_capture("4000", "60")
                self.assertEqual(code, 1)
                self.assertEqual(self.set_calls(), [])
                self.assertIn(
                    f'dmtweeter-synco.zero_test.durationLimitSec actual="{value}"; '
                    'expected="numeric seconds from requested 60 through 120"',
                    stderr,
                )

    def test_setup_reports_baseline_command_status(self):
        """A failed baseline command names the device and actual nonzero status."""
        self.env["ZERO_TEST_INDI_STATUS"] = "124"
        code, _, stderr = self.run_capture("4000", "1")
        self.assertEqual(code, 1)
        self.assertEqual(self.set_calls(), [])
        self.assertIn(
            'getINDI dmtweeter-synco baseline status actual="124"; expected="0"',
            stderr,
        )
        self.assertIn('dmtweeter-synco.enable.toggle actual="<missing>"; expected="Off"', stderr)

    def test_failed_enable_still_requests_off(self):
        """An uncertain enable result is cancelled because it may have applied."""
        self.env["ZERO_TEST_ON_STATUS"] = "124"
        code, _, _ = self.run_capture("4000", "1")
        self.assertEqual(code, 1)
        self.assertEqual(self.set_calls()[-1], "-s dmtweeter-synco.enable.toggle=Off")
        self.assertEqual((self.root / "enable").read_text(), "Off")

    def test_poll_failure_cancels_owned_trial(self):
        """Read failure after On stops only the trial started by this script."""
        self.env["ZERO_TEST_POLL_STATUS"] = "124"
        code, _, _ = self.run_capture("4000", "1")
        self.assertEqual(code, 1)
        self.assertEqual((self.root / "enable").read_text(), "Off")
        self.assertEqual(self.set_calls()[-1], "-s dmtweeter-synco.enable.toggle=Off")

    def test_operator_interrupt_cancels_only_owned_trial(self):
        """An interrupted active capture requests Off and joins its collectors."""
        self.env["ZERO_TEST_HOLD_ACTIVE"] = "1"
        code, _, _ = self.run_capture("4000", "120", interrupt=True)
        self.assertEqual(code, 143)
        self.assertEqual((self.root / "enable").read_text(), "Off")
        self.assertEqual(self.set_calls()[-1], "-s dmtweeter-synco.enable.toggle=Off")

    def test_requested_shorter_duration_uses_app_elapsed(self):
        """Elapsed reaching the target cannot substitute for the app finishing its trial."""
        self.env["ZERO_TEST_ELAPSED"] = "1"
        code, _, stderr = self.run_capture("6000", "1")
        self.assertEqual(code, 0, stderr)
        self.assertEqual((self.root / "polls").read_text(), "2")
        self.assertEqual((self.root / "duration").read_text(), "1")
        self.assertEqual((self.root / "enable").read_text(), "Off")

    def test_previous_finished_readback_does_not_end_new_trial(self):
        """Require new warming/active progress before accepting finished after On."""
        self.env["ZERO_TEST_STALE_FINISHED"] = "1"
        code, _, stderr = self.run_capture("6000", "1")
        self.assertEqual(code, 0, stderr)
        self.assertEqual((self.root / "polls").read_text(), "3")
        self.assertEqual(self.set_calls()[-1], "-s dmtweeter-synco.enable.toggle=Off")

    def test_warmup_timeout_is_not_a_completed_trial(self):
        """A finished arm attempt without an active phase reports incomplete capture."""
        self.env["ZERO_TEST_WARM_TIMEOUT"] = "1"
        code, _, _ = self.run_capture("4000", "120")
        self.assertEqual(code, 1)
        self.assertEqual((self.root / "enable").read_text(), "Off")
        self.assertIn(
            "trial finished without reaching active qualification",
            (self.capture_dir() / "status.txt").read_text(),
        )

    def test_premature_finish_reports_incomplete_duration(self):
        """An active trial ending before the requested interval is not successful."""
        self.env["ZERO_TEST_FINISHED_ELAPSED"] = "2"
        code, _, _ = self.run_capture("6000", "60")
        self.assertEqual(code, 1)
        self.assertEqual((self.root / "enable").read_text(), "Off")
        self.assertIn(
            "trial finished before the requested active duration",
            (self.capture_dir() / "status.txt").read_text(),
        )

    def test_app_restart_before_first_poll_cancels_owned_trial(self):
        """Compare the first poll with the baseline identity as well as later polls."""
        self.env["ZERO_TEST_BASELINE_SEQUENCE"] = "100"
        code, _, _ = self.run_capture("8000", "10")
        self.assertNotEqual(code, 0)
        self.assertIn("sequence regressed", (self.capture_dir() / "status.txt").read_text())
        self.assertEqual((self.root / "enable").read_text(), "Off")

    def test_logger_enqueue_failure_invalidates_capture(self):
        """Final transport quality includes popped rows lost after queue draining."""
        self.env["ZERO_TEST_LOG_FAILED"] = "1"
        code, _, _ = self.run_capture("8000", "1")
        self.assertNotEqual(code, 0)
        self.assertIn("final zero-event transport loss", (self.capture_dir() / "status.txt").read_text())
        self.assertEqual((self.root / "enable").read_text(), "Off")

    def test_old_duration_capability_never_enables(self):
        """Schema 4 cannot promise an app-enforced selected short interval."""
        self.env["ZERO_TEST_SCHEMA"] = "4"
        code, _, _ = self.run_capture("8000", "10")
        self.assertEqual(code, 1)
        self.assertEqual(self.set_calls(), [])

    def test_ignored_duration_setting_never_enables(self):
        """A successful client send still requires the effective app duration readback."""
        self.env["ZERO_TEST_IGNORE_DURATION"] = "1"
        code, _, stderr = self.run_capture("8000", "10")
        self.assertEqual(code, 1)
        self.assertNotIn("-s dmtweeter-synco.enable.toggle=On", self.set_calls())
        self.assertIn("duration readback failed", stderr)

    def test_longer_selection_after_short_run_uses_startup_ceiling(self):
        """A previous one-second selection does not lower the immutable maximum."""
        (self.root / "duration").write_text("1")
        code, _, stderr = self.run_capture("8000", "120")
        self.assertEqual(code, 0, stderr)
        self.assertEqual((self.root / "duration").read_text(), "120")

    def test_completed_trial_can_arrive_before_first_active_poll(self):
        """A new finished identity with commands and full elapsed qualifies a short run."""
        self.env["ZERO_TEST_FINISH_FIRST"] = "1"
        code, _, stderr = self.run_capture("8000", "1")
        self.assertEqual(code, 0, stderr)
        self.assertEqual((self.root / "polls").read_text(), "1")

    def test_wrong_trial_identity_cancels_owned_run(self):
        """A different generation cannot be accepted as this script's measurement."""
        self.env["ZERO_TEST_CURRENT_TRIAL"] = "2"
        code, _, _ = self.run_capture("8000", "10")
        self.assertEqual(code, 1)
        self.assertIn("unexpected trial identity", (self.capture_dir() / "status.txt").read_text())

    def test_mismatched_active_duration_cancels_owned_run(self):
        """Readbacks must keep the selected duration frozen throughout the new trial."""
        self.env["ZERO_TEST_CURRENT_DURATION"] = "120"
        code, _, _ = self.run_capture("8000", "10")
        self.assertEqual(code, 1)
        self.assertIn("duration differs", (self.capture_dir() / "status.txt").read_text())

    def test_active_overrun_is_incomplete_not_success(self):
        """A controller that fails its own bound is cancelled after reporting grace."""
        self.env.update(ZERO_TEST_HOLD_ACTIVE="1", ZERO_TEST_ELAPSED="7")
        code, _, _ = self.run_capture("8000", "1")
        self.assertEqual(code, 1)
        self.assertIn("did not report completion", (self.capture_dir() / "status.txt").read_text())

    def test_finished_overrun_is_incomplete(self):
        """A finished report cannot excuse an app that ran far past its selected bound."""
        self.env.update(ZERO_TEST_FINISH_FIRST="1", ZERO_TEST_FINISHED_ELAPSED="120")
        code, _, _ = self.run_capture("8000", "10")
        self.assertEqual(code, 1)
        self.assertIn("finished beyond", (self.capture_dir() / "status.txt").read_text())

    def test_poll_uses_exact_element_specs(self):
        """Polling returns on a complete vector rather than the wildcard timeout."""
        code, _, stderr = self.run_capture("8000", "1")
        self.assertEqual(code, 0, stderr)
        polls = [line.split() for line in (self.root / "indi-calls").read_text().splitlines()
                 if line.startswith("dmtweeter-synco.zero_test.")]
        self.assertTrue(polls)
        for fields in polls:
            self.assertIn("dmtweeter-synco.zero_test.trial", fields)
            self.assertIn("dmtweeter-synco.zero_test.finished", fields)
            self.assertTrue(all(field.count(".") == 2 and "*" not in field for field in fields))

    def test_invalid_inputs_and_help_do_not_contact_services(self):
        """Reject additive/free-run arguments and invalid finite durations."""
        for args in [("--with-main", "4000"), ("5000",), ("10000",), ("4000", "0"),
                     ("4000", "121"), ("4000", "999999999999999999999"),
                     ("8000", "10", "-1"), ("8000", "10", "51"),
                     ("8000", "10", "nan"), ("8000", "10", "1;echo bad"),
                     ("8000", "10", "5"), ("8000", "10", "0", "extra"),
                     ("8000", "10", "0", "-1"), ("8000", "10", "0", "51"),
                     ("8000", "10", "0", "nan"), ("8000", "10", "0", "10", "extra"),
                     ("8000", "10", "0", "10", "-1"), ("8000", "10", "0", "10", "11"),
                     ("8000", "10", "0", "10", "nan"), ("8000", "10", "0", "10", "0", "extra")]:
            with self.subTest(args=args):
                code, _, _ = self.run_capture(*args)
                self.assertEqual(code, 2)
        code, stdout, _ = self.run_capture("--help")
        self.assertEqual(code, 0)
        self.assertIn("effective total cadence", stdout)
        self.assertFalse((self.root / "indi-calls").exists())


if __name__ == "__main__":
    unittest.main()
