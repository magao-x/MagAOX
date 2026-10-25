#!/usr/bin/env python3
## \file capture_camera_timing_test.py
# \brief Offline ownership and failure regressions for the operator capture helper.
# \ingroup ocam2KCtrl_unit_test

"""Exercise the capture lifecycle using ordinary temporary files, never tracefs."""

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock


script = Path(__file__).resolve().parents[1] / "capture-camera-timing.py"
spec = importlib.util.spec_from_file_location("camera_trace", script)
camera_trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_trace)


class FakeCapture(camera_trace.TraceCapture):
    """Materialize tracefs controls as temporary files and remove only those files."""

    def create_instance(self):
        """Copy a fake instance without accessing a kernel tracing filesystem."""
        shutil.copytree(self.tracefs / "template", self.instance)
        self.created = True

    def remove_instance(self):
        """Remove only the fixture-owned fake controls and instance."""
        shutil.rmtree(self.instance)
        self.created = False


class TraceCaptureTest(unittest.TestCase):
    """Check setup/teardown boundaries and preservation under injected failures."""

    def setUp(self):
        """Create isolated fake controls and unrelated traces to protect."""
        self.temporary = tempfile.TemporaryDirectory(prefix="camera-trace-fake-")
        self.root = Path(self.temporary.name)
        self.tracefs = self.root / "tracefs"
        self.output = self.root / "output"
        self.output.mkdir()
        (self.tracefs / "instances" / "existing").mkdir(parents=True)
        (self.tracefs / "trace").write_text("global trace must survive\n")
        (self.tracefs / "tracing_on").write_text("1\n")
        (self.tracefs / "trace_options").write_text("unrelated global options\n")
        (self.tracefs / "instances" / "existing" / "trace").write_text("other user's trace\n")
        template = self.tracefs / "template"
        template.mkdir()
        for name, value in {
                "trace_clock": "local [mono] mono_raw\n", "tracing_on": "0\n",
                "tracing_cpumask": "ffff\n", "buffer_size_kb": "64\n",
                "buffer_total_size_kb": "8192\n", "trace_options": "nooptionchanges\n",
                "set_event_pid": "", "set_event_notrace_pid": "", "trace": ""}.items():
            (template / name).write_text(value)
        for cpu in (45, 61):
            directory = template / "per_cpu" / f"cpu{cpu}"
            directory.mkdir(parents=True)
            (directory / "buffer_size_kb").write_text("64\n")
            (directory / "stats").write_text("entries: 100\noverrun: 0\ncommit overrun: 0\ndropped events: 0\n")
        events = ("sched/sched_switch", "sched/sched_wakeup", "sched/sched_waking",
                  "irq/irq_handler_entry", "irq/irq_handler_exit", "irq/softirq_entry",
                  "irq/softirq_exit", "irq/softirq_raise", "irq_vectors/local_timer_entry",
                  "irq_vectors/local_timer_exit", "timer/hrtimer_expire_entry")
        for event in events:
            directory = template / "events" / event
            directory.mkdir(parents=True)
            (directory / "enable").write_text("0\n")
            (directory / "format").write_text(f"format for {event}\n")
        (template / "events" / "enable").write_text("0\n")

    def tearDown(self):
        """Remove the ordinary temporary fixture directory."""
        self.temporary.cleanup()

    def assert_other_traces_unchanged(self):
        """Global controls and other instance data must retain their exact values."""
        self.assertEqual((self.tracefs / "trace").read_text(), "global trace must survive\n")
        self.assertEqual((self.tracefs / "tracing_on").read_text(), "1\n")
        self.assertEqual((self.tracefs / "trace_options").read_text(), "unrelated global options\n")
        self.assertEqual((self.tracefs / "instances" / "existing" / "trace").read_text(),
                         "other user's trace\n")

    def test_stopped_trace_saved_and_only_owned_instance_removed(self):
        """Save a complete capture while retaining unrelated traces and options."""
        capture = FakeCapture(self.tracefs, self.output, [45, 61], extra_events=True)
        capture.setup()
        self.assertEqual((capture.instance / "tracing_cpumask").read_text(), "20002000,00000000\n")
        self.assertIn("timer/hrtimer_expire_entry", capture.events)
        self.assertIn("sched/sched_waking", capture.events)
        self.assertIn("irq_vectors/local_timer_entry", capture.events)
        self.assertEqual((capture.instance / "set_event_pid").read_text(), "\n")
        self.assertEqual((capture.instance / "trace_options").read_text(), "nooptionchanges\n")
        capture.write("tracing_on", 1)
        (capture.instance / "trace").write_text("retained camera trace\n")
        self.assertEqual(capture.close(), [])
        self.assertFalse(capture.instance.exists())
        self.assertEqual((self.output / "trace.txt").read_text(), "retained camera trace\n")
        self.assertIn("overrun: 0", (self.output / "cpu45-stats.txt").read_text())
        self.assert_other_traces_unchanged()

    def test_optional_apic_events_may_be_absent_and_are_reported(self):
        """Expose optional vector coverage without requiring architecture-specific events."""
        shutil.rmtree(self.tracefs / "template" / "events" / "irq_vectors")
        capture = FakeCapture(self.tracefs, self.output, [45])
        capture.setup()
        self.assertEqual(capture.event_groups["irq_vectors"],
                         {"available": False, "enabled": False})
        self.assertEqual(capture.close(), [])
        self.assert_other_traces_unchanged()

    def test_setup_failure_stops_and_removes_partial_private_instance(self):
        """Clean a partially configured instance when a required event is absent."""
        (self.tracefs / "template" / "events" / "irq" / "irq_handler_exit" / "enable").unlink()
        capture = FakeCapture(self.tracefs, self.output, [45])
        with self.assertRaisesRegex(RuntimeError, "Required kernel event unavailable"):
            capture.setup()
        self.assertTrue(capture.created)
        self.assertEqual(capture.close(), [])
        self.assertFalse(capture.instance.exists())
        self.assert_other_traces_unchanged()

    def test_unreadable_trace_is_stopped_and_preserved(self):
        """Keep the stopped private instance if its trace cannot be saved."""
        capture = FakeCapture(self.tracefs, self.output, [45])
        capture.setup()
        capture.write("tracing_on", 1)
        (capture.instance / "trace").unlink()
        errors = capture.close()
        self.assertTrue(any("preserving" in error for error in errors))
        self.assertTrue(capture.instance.exists())
        self.assertEqual((capture.instance / "tracing_on").read_text(), "0\n")
        self.assertEqual((capture.instance / "events" / "enable").read_text(), "0\n")
        self.assert_other_traces_unchanged()

    def test_failed_stop_or_event_disable_preserves_owned_instance(self):
        """Preserve an owned instance after either control write fails."""
        for target in ("tracing_on", "events/enable"):
            with self.subTest(control=target):
                capture = FakeCapture(self.tracefs, self.output, [45])
                capture.setup()
                capture.write("tracing_on", 1)
                original_write = capture.write

                def fail_control(relative, value):
                    """Fail the selected control while forwarding all other fixture writes."""
                    if relative == target:
                        raise PermissionError("Injected control write failure")
                    original_write(relative, value)

                with mock.patch.object(capture, "write", side_effect=fail_control):
                    errors = capture.close()
                self.assertTrue(any("Preserving" in error for error in errors))
                self.assertTrue(capture.created)
                self.assertTrue(capture.instance.exists())
                self.assertTrue((self.output / "trace.txt").exists())
                self.assert_other_traces_unchanged()

    def test_missing_monotonic_clock_has_no_silent_substitute(self):
        """Reject clock substitution and clean only the incomplete private capture."""
        (self.tracefs / "template" / "trace_clock").write_text("[local] mono_raw\n")
        capture = FakeCapture(self.tracefs, self.output, [45])
        with self.assertRaisesRegex(RuntimeError, "lacks the mono trace clock"):
            capture.setup()
        self.assertEqual(capture.close(), [])
        self.assert_other_traces_unchanged()

    def test_cpu_mask_and_input_validation(self):
        """Encode CPU masks and reject malformed CPU lists before tracing."""
        self.assertEqual(camera_trace.cpu_list("0,3-5,45"), [0, 3, 4, 5, 45])
        self.assertEqual(camera_trace.cpu_mask([0, 31, 32]), "00000001,80000001")
        for invalid in ("-1", "4-2", "1,,2", "all", "65536"):
            with self.assertRaises(ValueError):
                camera_trace.cpu_list(invalid)
        stamps = camera_trace.anchor()
        self.assertLessEqual(stamps["monotonic_before_ns"], stamps["monotonic_after_ns"])

    def run_main(self, options=(), sleep=None, metadata=None, fail_stop=False, fail_enable=False):
        """Run the CLI with fake proc/sys paths, tracefs controls and bounded mocked waits."""
        output = self.root / "main-output"
        status = "Tgid:\t123\nCpus_allowed_list:\t45\n"
        status_path = self.root / "camera-status"
        status_path.write_text(status)
        cpu_path = self.root / "cpu45"
        cpu_path.mkdir(exist_ok=True)
        captured = []
        activity = []
        original_path = Path

        def fake_path(value):
            """Redirect every absolute proc/sys path used by main to its fixture counterpart."""
            path = original_path(value)
            if str(path) == "/proc/123/status":
                return status_path
            if str(path) in ("/sys/kernel/tracing", "/sys/kernel/debug/tracing"):
                return self.tracefs
            if str(path) == "/sys/devices/system/cpu/cpu45":
                return cpu_path
            if str(path).startswith(("/proc/", "/sys/")):
                self.fail(f"Unexpected instrument diagnostic path: {path}")
            return path

        def fake_capture(*arguments):
            """Track ownership and inject control failures after setup has completed."""
            capture = FakeCapture(*arguments)
            capture.activity = activity
            original_write = capture.write

            def write(relative, value):
                """Forward fixture controls, optionally failing the timed start or stop."""
                started = any(call == ("tracing_on", 1) for call in capture.control_calls)
                capture.control_calls.append((relative, value))
                activity.append(("control", relative, value))
                if relative == "tracing_on" and ((value == 1 and fail_enable) or
                                                 (value == 0 and started and fail_stop)):
                    raise PermissionError("Injected capture control failure")
                original_write(relative, value)

            capture.control_calls = []
            capture.write = write
            captured.append(capture)
            return capture

        def fake_sleep(duration):
            """Delegate the wait without blocking or touching actual tracefs."""
            activity.append(("sleep", duration))
            if sleep is not None:
                sleep(duration, captured[0])

        def fake_metadata(tid, cpus):
            """Return deterministic identity metadata or a requested fault."""
            activity.append(("metadata",))
            if metadata is not None:
                return metadata(tid, cpus)
            return {"tid": tid, "cpus": cpus, "thread_status": status}

        def fake_print(*arguments, **keywords):
            """Record terminal output so measured-interval regressions are visible."""
            activity.append(("print", *arguments))

        arguments = [str(script), "--tid", "123", "--output", str(output), *options]
        with mock.patch("sys.argv", arguments), mock.patch.object(camera_trace, "Path", fake_path), \
                mock.patch.object(camera_trace.os, "geteuid", return_value=0), \
                mock.patch.object(camera_trace, "TraceCapture", side_effect=fake_capture), \
                mock.patch.object(camera_trace, "metadata", side_effect=fake_metadata), \
                mock.patch.object(camera_trace.time, "sleep", side_effect=fake_sleep), \
                mock.patch.object(camera_trace.signal, "signal"), \
                mock.patch("builtins.print", side_effect=fake_print):
            result = camera_trace.main()

        return result, captured, json.loads((output / "metadata.json").read_text())

    def test_main_defaults_settle_before_five_second_capture(self):
        """Separate metadata/setup from a disabled settle and bounded private capture."""
        waits = []

        def check_wait(duration, capture):
            """Require tracing off during settle and on only during the measured wait."""
            waits.append(duration)
            self.assertEqual((capture.instance / "tracing_on").read_text(),
                             "0\n" if len(waits) == 1 else "1\n")

        result, captures, details = self.run_main(sleep=check_wait)
        self.assertEqual(result, 0)
        self.assertEqual(waits, [5, 5])
        self.assertTrue(details["capture_started"])
        self.assertTrue(details["capture_completed"])
        self.assertEqual(details["settle_requested_s"], 5)
        self.assertEqual(details["duration_requested_s"], 5)
        self.assertFalse(captures[0].instance.exists())
        phases = details["phase_anchors"]
        previous_end = 0
        for phase in ("metadata_before", "setup", "settle", "capture_enable",
                      "capture_disable", "metadata_after"):
            self.assertLessEqual(previous_end, phases[phase]["begin"]["monotonic_before_ns"])
            self.assertLessEqual(phases[phase]["begin"]["monotonic_after_ns"],
                                 phases[phase]["end"]["monotonic_before_ns"])
            previous_end = phases[phase]["end"]["monotonic_after_ns"]
        self.assertEqual(details["capture_begin"], phases["capture_enable"]["begin"])
        self.assertEqual(details["capture_end"], phases["capture_disable"]["begin"])
        self.assert_other_traces_unchanged()

    def test_main_interrupted_settle_never_enables_capture(self):
        """Save interrupted settling as a partial preparation with no capture boundaries."""
        def interrupt_wait(duration, capture):
            """Deliver Ctrl-C while tracing remains disabled."""
            self.assertEqual((capture.instance / "tracing_on").read_text(), "0\n")
            raise KeyboardInterrupt

        result, captures, details = self.run_main(sleep=interrupt_wait)
        self.assertEqual(result, 130)
        self.assertEqual(details["interrupted_phase"], "settle")
        self.assertFalse(details["capture_started"])
        self.assertFalse(details["capture_completed"])
        self.assertNotIn("capture_begin", details)
        self.assertNotIn("capture_end", details)
        self.assertNotIn("end", details["phase_anchors"]["settle"])
        self.assertNotIn(("tracing_on", 1), captures[0].control_calls)
        self.assertFalse(captures[0].instance.exists())
        self.assert_other_traces_unchanged()

    def test_main_interrupted_capture_stops_and_saves_partial_interval(self):
        """Retain actual toggle brackets when the measured sleep is interrupted."""
        def interrupt_capture(duration, capture):
            """Deliver Ctrl-C only after tracing has started."""
            if (capture.instance / "tracing_on").read_text() == "1\n":
                raise KeyboardInterrupt

        result, captures, details = self.run_main(("--settle-seconds", "0", "--seconds", "2"),
                                                 sleep=interrupt_capture)
        self.assertEqual(result, 130)
        self.assertEqual(details["interrupted_phase"], "capture")
        self.assertTrue(details["capture_started"])
        self.assertFalse(details["capture_completed"])
        self.assertIn("end", details["phase_anchors"]["capture_disable"])
        self.assertLess(details["capture_begin"]["monotonic_before_ns"],
                        details["capture_end"]["monotonic_before_ns"])
        self.assertFalse(captures[0].instance.exists())
        self.assert_other_traces_unchanged()

    def test_main_setup_failure_has_no_fabricated_capture(self):
        """Record an incomplete setup and clean its private controls without starting."""
        (self.tracefs / "template" / "events" / "irq" / "irq_handler_exit" / "enable").unlink()
        result, captures, details = self.run_main()
        self.assertEqual(result, 1)
        self.assertIn("Required kernel event unavailable", details["error"])
        self.assertNotIn("end", details["phase_anchors"]["setup"])
        self.assertNotIn("settle", details["phase_anchors"])
        self.assertNotIn("capture_begin", details)
        self.assertNotIn("capture_end", details)
        self.assertFalse(details["capture_started"])
        self.assertFalse(details["capture_completed"])
        self.assertFalse(captures[0].instance.exists())
        self.assert_other_traces_unchanged()

    def test_main_interrupted_metadata_retains_partial_phase(self):
        """Protect initial metadata with the same interruption cleanup as setup."""
        calls = []

        def interrupted_metadata(tid, cpus):
            """Interrupt the first diagnostic snapshot and allow the cleanup snapshot."""
            calls.append(tid)
            if len(calls) == 1:
                raise KeyboardInterrupt
            return {"thread_status": "Cpus_allowed_list:\t45\n"}

        result, captures, details = self.run_main(metadata=interrupted_metadata)
        self.assertEqual(result, 130)
        self.assertEqual(details["interrupted_phase"], "metadata_before")
        self.assertNotIn("end", details["phase_anchors"]["metadata_before"])
        self.assertNotIn("capture_begin", details)
        self.assertNotIn("capture_end", details)
        self.assertFalse(captures[0].created)
        self.assert_other_traces_unchanged()

    def test_main_failed_start_retains_attempt_and_cleanup_stop(self):
        """Keep a failed enable attempt distinct from a confirmed capture."""
        result, captures, details = self.run_main(fail_enable=True)
        self.assertEqual(result, 1)
        self.assertFalse(details["capture_started"])
        self.assertFalse(details["capture_completed"])
        self.assertNotIn("end", details["phase_anchors"]["capture_enable"])
        self.assertIn("end", details["phase_anchors"]["capture_disable"])
        self.assertIn("capture_begin", details)
        self.assertIn("capture_end", details)
        self.assertFalse(captures[0].instance.exists())
        self.assert_other_traces_unchanged()

    def test_main_failed_stop_preserves_unconfirmed_capture(self):
        """Do not label the capture complete when its final stop write fails."""
        result, captures, details = self.run_main(fail_stop=True)
        self.assertEqual(result, 1)
        self.assertTrue(details["capture_started"])
        self.assertFalse(details["capture_completed"])
        self.assertNotIn("end", details["phase_anchors"]["capture_disable"])
        self.assertTrue(details["cleanup_errors"])
        self.assertTrue(captures[0].instance.exists())
        self.assert_other_traces_unchanged()

    def test_capture_interval_has_no_diagnostic_or_output_io(self):
        """Keep filesystem reads/writes, console output and metadata outside timed tracing."""
        running = False
        read_text = Path.read_text
        write_text = Path.write_text

        def checked_read(path, *arguments, **keywords):
            """Reject every read while fake trace recording is enabled."""
            self.assertFalse(running, f"Read during measured capture: {path}")
            return read_text(path, *arguments, **keywords)

        def checked_write(path, data, *arguments, **keywords):
            """Allow only the essential start/stop control writes during recording."""
            nonlocal running
            if running:
                self.assertEqual(path.name, "tracing_on")
                self.assertEqual(data, "0\n")
            result = write_text(path, data, *arguments, **keywords)
            if path.name == "tracing_on" and path.parent.name.startswith("ocam-timing-"):
                running = data == "1\n"
            return result

        with mock.patch.object(Path, "read_text", checked_read), \
                mock.patch.object(Path, "write_text", checked_write):
            result, captures, details = self.run_main()
        self.assertEqual(result, 0)
        activity = captures[0].activity
        start = activity.index(("control", "tracing_on", 1))
        self.assertEqual(activity[start + 1:start + 3], [("sleep", 5), ("control", "tracing_on", 0)])
        self.assertTrue(details["capture_completed"])

    def test_capture_wait_error_reports_only_after_stop(self):
        """Stop a failed measured wait before reporting the error to the terminal."""
        def fail_measured_wait(duration, capture):
            """Fail only the second wait, after the private trace has been enabled."""
            if (capture.instance / "tracing_on").read_text() == "1\n":
                raise OSError("Injected measured wait failure")

        result, captures, details = self.run_main(sleep=fail_measured_wait)
        self.assertEqual(result, 1)
        self.assertTrue(details["capture_started"])
        self.assertFalse(details["capture_completed"])
        activity = captures[0].activity
        start = activity.index(("control", "tracing_on", 1))
        self.assertEqual(activity[start + 1:start + 3], [("sleep", 5), ("control", "tracing_on", 0)])
        error_print = next(index for index, event in enumerate(activity)
                           if event[0] == "print" and "Capture failed" in event[1])
        self.assertGreater(error_print, start + 2)
        self.assertFalse(captures[0].instance.exists())
        self.assert_other_traces_unchanged()

    def test_nonfinite_or_out_of_range_waits_are_rejected_before_io(self):
        """Reject unbounded waits before diagnostics, output creation or tracefs access."""
        for option, values in (("--seconds", ("nan", "inf", "-inf", "0", "31")),
                               ("--settle-seconds", ("nan", "inf", "-inf", "-1", "61"))):
            for value in values:
                with self.subTest(option=option, value=value):
                    arguments = [str(script), "--tid", "123", "--output", str(self.output),
                                 f"{option}={value}"]
                    with mock.patch("sys.argv", arguments), \
                            mock.patch.object(camera_trace.os, "geteuid", return_value=0), \
                            mock.patch.object(camera_trace, "read_if_present") as read, \
                            mock.patch.object(camera_trace, "TraceCapture") as capture, \
                            mock.patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
                        camera_trace.main()
                    self.assertEqual(raised.exception.code, 2)
                    read.assert_not_called()
                    capture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
