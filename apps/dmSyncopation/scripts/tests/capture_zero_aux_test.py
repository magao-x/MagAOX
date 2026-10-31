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
            'printf "%s\\n" "$*" >> "$ZERO_TEST_ROOT/indi-calls"\n'
            '/bin/sleep 0.02\n'
            '[[ ${ZERO_TEST_INDI_STATUS:-0} != 0 ]] && exit "$ZERO_TEST_INDI_STATUS"\n'
            'enable=$(cat "$ZERO_TEST_ROOT/enable")\nsteps=$(cat "$ZERO_TEST_ROOT/steps")\n'
            'case $1 in\n'
            'dmtweeter-synco)\n'
            ' printf "dmtweeter-synco.enable.toggle=%s\\n" "$enable"\n'
            ' printf "dmtweeter-synco.settings.steps=%s\\n" "$steps"\n'
            ' printf "dmtweeter-synco.zero_test.enabled=%s\\n" "${ZERO_TEST_MODE:-1}"\n'
            ' printf "dmtweeter-synco.zero_test.durationSec=120\\n";;\n'
            'dmtweeter-synco.zero_test)\n'
            ' [[ ${ZERO_TEST_POLL_STATUS:-0} != 0 ]] && exit "$ZERO_TEST_POLL_STATUS"\n'
            ' count=0\n[[ -f $ZERO_TEST_ROOT/polls ]] && count=$(cat "$ZERO_TEST_ROOT/polls")\n'
            ' count=$((count+1))\nprintf "%s" "$count" > "$ZERO_TEST_ROOT/polls"\n'
            ' if [[ ${ZERO_TEST_STALE_FINISHED:-0} == 1 ]]; then\n'
            '   if ((count==1)); then count=2; else count=$((count-1)); fi\nfi\n'
            ' if [[ ${ZERO_TEST_HOLD_ACTIVE:-0} == 1 ]]; then count=1; fi\n'
            ' active=$((count==1))\nwarming=0\n'
            ' if [[ ${ZERO_TEST_WARM_TIMEOUT:-0} == 1 ]]; then\n'
            '   active=0\nwarming=$((count==1))\nfi\n'
            ' elapsed=${ZERO_TEST_ELAPSED:-0}\n'
            ' if ((count>=2)); then elapsed=${ZERO_TEST_FINISHED_ELAPSED:-120}; fi\n'
            ' if [[ ${ZERO_TEST_WARM_TIMEOUT:-0} == 1 ]]; then elapsed=0; fi\n'
            ' printf "dmtweeter-synco.zero_test.active=%s\\n" "$active"\n'
            ' printf "dmtweeter-synco.zero_test.warming=%s\\n" "$warming"\n'
            ' printf "dmtweeter-synco.zero_test.elapsedSec=%s\\n" "$elapsed"\n'
            ' printf "dmtweeter-synco.zero_test.finished=%s\\n" "$((count>=2))";;\n'
            'camwfs)\n'
            ' printf "camwfs.fps.current=%s\\ncamwfs.synchro.toggle=On\\n" "${ZERO_TEST_FPS:-2000}";;\n'
            'sysMonRTC.set_latency.toggle) printf "sysMonRTC.set_latency.toggle=On\\n";;\n'
            '*) printf "fake.current=1\\n";;\nesac',
        )
        self.stub(
            "setINDI",
            'printf "%s\\n" "$*" >> "$ZERO_TEST_ROOT/set-calls"\n'
            'case $2 in\n'
            'dmtweeter-synco.settings.steps=*)\n'
            ' [[ $(cat "$ZERO_TEST_ROOT/enable") == Off ]] || exit 9\n'
            ' printf "%s" "${2##*=}" > "$ZERO_TEST_ROOT/steps";;\n'
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
            ["-n dmtweeter-synco.settings.steps=2", "-s dmtweeter-synco.enable.toggle=On",
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

    def test_preexisting_enabled_run_is_not_cancelled(self):
        """Refuse a run owned by another operator without sending any mutation."""
        (self.root / "enable").write_text("On")
        code, _, _ = self.run_capture("4000", "1")
        self.assertEqual(code, 1)
        self.assertEqual(self.set_calls(), [])
        self.assertEqual((self.root / "enable").read_text(), "On")

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
        code, _, _ = self.run_capture("4000", "1")
        self.assertEqual(code, 1)
        self.assertEqual(self.set_calls(), [])
        self.assertIn("status=255", (self.capture_dir() / "status.txt").read_text())

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
        """The operator's shorter duration uses monotonic active time from the app."""
        self.env["ZERO_TEST_HOLD_ACTIVE"] = "1"
        self.env["ZERO_TEST_ELAPSED"] = "1"
        code, _, stderr = self.run_capture("6000", "1")
        self.assertEqual(code, 0, stderr)
        self.assertEqual((self.root / "polls").read_text(), "1")
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

    def test_invalid_inputs_and_help_do_not_contact_services(self):
        """Reject additive/free-run arguments and invalid finite durations."""
        for args in [("--with-main", "4000"), ("5000",), ("4000", "0"),
                     ("4000", "121"), ("4000", "999999999999999999999")]:
            with self.subTest(args=args):
                code, _, _ = self.run_capture(*args)
                self.assertEqual(code, 2)
        code, stdout, _ = self.run_capture("--help")
        self.assertEqual(code, 0)
        self.assertIn("effective total cadence", stdout)
        self.assertFalse((self.root / "indi-calls").exists())


if __name__ == "__main__":
    unittest.main()
