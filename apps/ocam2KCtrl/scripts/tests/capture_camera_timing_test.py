#!/usr/bin/env python3
## \file capture_camera_timing_test.py
# \brief Offline ownership and failure regressions for the operator capture helper.
# \ingroup ocam2KCtrl_unit_test

"""Exercise the capture lifecycle using ordinary temporary files, never tracefs."""

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
