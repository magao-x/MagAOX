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
from types import SimpleNamespace
import unittest
from unittest import mock


script = Path(__file__).resolve().parents[1] / "capture-camera-timing.py"
spec = importlib.util.spec_from_file_location("camera_trace", script)
camera_trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_trace)
edt_probes = camera_trace.edt_probe_helper()


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

    def probe_fixture(self):
        """Add unrelated global probe state and active-profile-required private options."""
        (self.tracefs / "kprobe_events").write_text("p:unrelated/other edt:other\n")
        (self.tracefs / "kprobe_profile").write_text("other 9 0\n")
        for root in (self.tracefs, self.tracefs / "instances" / "existing"):
            event = root / "events" / "unrelated" / "other"
            event.mkdir(parents=True)
            (event / "enable").write_text("1\n")
            (event / "format").write_text("unrelated format\n")
        (self.tracefs / "template" / "trace_options").write_text("context-info\nirq-info\n")

    def probe_helper(self, faults, owners, activity):
        """Exercise the real probe owner against a fake dynamic-event kernel interface."""
        tracefs = self.tracefs

        class FakeProbeCheck(edt_probes.ProbeCheck):
            """Model dynamic event registration without opening tracefs."""

            def __init__(owner, *arguments, **keywords):
                """Keep the exact production owner state and record its test instance."""
                super().__init__(*arguments, **keywords)
                owner.registrations = 0
                owner.removals = 0
                owners.append(owner)

            def report(owner, message):
                """Retain reporting order without writing the terminal."""
                activity.append(("probe_report", message))

            def write_command(owner, command):
                """Materialize per-instance controls and reconcile exact definition removals."""
                activity.append(("probe_command", command))
                fields = command.split()
                name = fields[0].split(":", 1)[1]
                if any(len(part) >= 64 for part in name.split("/")):
                    raise OSError("Event/group name exceeds Linux trace probe limit")
                lines = (tracefs / "kprobe_events").read_text().splitlines()
                roots = [tracefs] + list((tracefs / "instances").glob("*"))
                if fields[0].startswith("-:"):
                    owner.removals += 1
                    if owner.removals == faults.get("remove"):
                        raise PermissionError("Injected probe removal failure")
                    lines = [line for line in lines if line.split()[0].split(":", 1)[1] != name]
                    for root in roots:
                        directory = root / "events" / name
                        if directory.exists():
                            shutil.rmtree(directory)
                            if not any(directory.parent.iterdir()):
                                directory.parent.rmdir()
                else:
                    owner.registrations += 1
                    lines.append(command)
                    for root in roots:
                        directory = root / "events" / name
                        directory.mkdir(parents=True)
                        (directory / "enable").write_text("0\n")
                        (directory / "format").write_text(f"format for {name}\n")
                (tracefs / "kprobe_events").write_text("\n".join(lines) + "\n")
                own_events = [line.split()[0].split(":", 1)[1].split("/", 1)[1]
                              for line in lines if owner.group + "/" in line]
                (tracefs / "kprobe_profile").write_text("other 9 0\n" +
                    "".join(f"{event} 0 0\n" for event in own_events))
                if owner.registrations == faults.get("register") and not fields[0].startswith("-:"):
                    raise OSError("Injected registration failure after kernel accepted definition")

        def preflight(root):
            """Require preflight before any output or private-instance mutation."""
            activity.append(("preflight",))
            self.assertFalse((self.root / "main-output").exists())
            self.assertEqual(list((root / "instances").iterdir()), [root / "instances" / "existing"])
            if faults.get("preflight"):
                raise RuntimeError("Injected EDT identity mismatch")

        def identity():
            """Record the loaded identity recheck after tracing has stopped."""
            activity.append(("identity_after",))
            if faults.get("identity_after"):
                raise RuntimeError("Injected final EDT identity mismatch")

        return SimpleNamespace(ProbeCheck=FakeProbeCheck, preflight=preflight, check_identity=identity,
                               KERNEL=edt_probes.KERNEL, VERSION=edt_probes.VERSION,
                               SRCVERSION=edt_probes.SRCVERSION, BUILD_ID=edt_probes.BUILD_ID)

    def run_main(self, options=(), sleep=None, metadata=None, fail_stop=False, fail_enable=False,
                 edt_faults=None):
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
        edt_faults = edt_faults or {}
        owners = []
        probe_controls = {"enable": 0, "disable": 0}
        if "--edt-wrappers" in options:
            self.probe_fixture()
        helper = self.probe_helper(edt_faults, owners, activity)

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
                if relative.startswith("events/ocam_edt_capture_") and relative.endswith("/enable"):
                    action = "enable" if value == 1 else "disable"
                    probe_controls[action] += 1
                    if edt_faults.get(action) == probe_controls[action]:
                        # Apply first to test cleanup after an uncertain control write.
                        original_write(relative, value)
                        raise PermissionError(f"Injected EDT {action} failure")
                original_write(relative, value)

            if edt_faults.get("save"):
                def fail_save():
                    """Fail output extraction without bypassing probe cleanup."""
                    raise OSError("Injected trace save failure")
                capture.save = fail_save
            capture.probe_owners = owners
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
                mock.patch.object(camera_trace, "edt_probe_helper", return_value=helper), \
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

    def assert_unrelated_probes_unchanged(self):
        """Retain another user's global definitions and enable state in every instance."""
        self.assertIn("p:unrelated/other edt:other\n", (self.tracefs / "kprobe_events").read_text())
        for root in (self.tracefs, self.tracefs / "instances" / "existing"):
            self.assertEqual((root / "events/unrelated/other/enable").read_text(), "1\n")
        self.assert_other_traces_unchanged()

    def test_active_edt_capture_saves_typed_definitions_and_cleans_only_owners(self):
        """Use eight accepted wrapper probes with larger buffers and exact saved metadata."""
        result, captures, details = self.run_main(("--edt-wrappers", "--seconds", "1"))
        self.assertEqual(result, 0)
        capture = captures[0]
        probe = details["edt_wrappers"]
        self.assertEqual(len(probe["definitions"]), 8)
        self.assertEqual(len(probe["formats"]), 8)
        self.assertTrue(probe["identity_after_stop_verified"])
        self.assertFalse(probe["return_pool_misses_verified"])
        self.assertTrue(probe["window_includes_settle"])
        self.assertEqual(probe["remaining_definitions"], [])
        self.assertTrue(all("r64:" in definition or "=%di:x64" in definition
                            for definition in probe["definitions"].values()))
        self.assertTrue(probe["profile_before"]["raw"].startswith("other 9 0\n"))
        self.assertEqual(len(probe["profile_before"]["owned_rows"]), 8)
        self.assertEqual(len(probe["profile_after"]["owned_rows"]), 8)
        self.assertEqual(probe["verified_identity"]["build_id"], edt_probes.BUILD_ID)
        for phase in ("activation", "deactivation"):
            self.assertLessEqual(probe[phase]["begin"]["monotonic_after_ns"],
                                 probe[phase]["end"]["monotonic_before_ns"])
        self.assertIn(("per_cpu/cpu45/buffer_size_kb", 32768), capture.control_calls)
        for name in probe["definitions"]:
            self.assertIn((f"events/{name}/enable", 1), capture.control_calls)
            self.assertIn((f"events/{name}/enable", 0), capture.control_calls)
            self.assertTrue((self.root / "main-output/event-formats" /
                             f"{name.replace('/', '-')}.txt").exists())
        self.assertEqual(capture.probe_owners[0].removals, 8)
        self.assertFalse(capture.instance.exists())
        self.assertEqual((self.tracefs / "kprobe_events").read_text(), "p:unrelated/other edt:other\n")
        self.assert_unrelated_probes_unchanged()

    def test_active_partial_registration_is_reconciled_after_write_failure(self):
        """Remove even the failed registration whose definition was accepted before an error."""
        result, captures, details = self.run_main(("--edt-wrappers",), edt_faults={"register": 3})
        self.assertEqual(result, 1)
        self.assertIn("registration failure", details["error"])
        self.assertEqual(len(details["edt_wrappers"]["definitions"]), 3)
        self.assertEqual(len(details["edt_wrappers"]["formats"]), 3)
        self.assertEqual(details["edt_wrappers"]["remaining_definitions"], [])
        self.assertFalse(captures[0].instance.exists())
        self.assertFalse(details["capture_started"])
        self.assert_unrelated_probes_unchanged()

    def test_active_partial_enable_is_disabled_and_reconciled(self):
        """Disable all definitions when a private enable write succeeds but reports failure."""
        result, captures, details = self.run_main(("--edt-wrappers",), edt_faults={"enable": 3})
        self.assertEqual(result, 1)
        self.assertNotIn("end", details["edt_wrappers"]["activation"])
        self.assertIn("end", details["edt_wrappers"]["deactivation"])
        self.assertEqual(details["edt_wrappers"]["remaining_definitions"], [])
        self.assertEqual(captures[0].probe_owners[0].removals, 8)
        self.assertFalse(captures[0].instance.exists())
        self.assert_unrelated_probes_unchanged()

    def test_active_failed_save_preserves_buffer_and_disabled_definitions(self):
        """Keep the stopped trace and its dynamic-event decoders after output failure."""
        result, captures, details = self.run_main(("--edt-wrappers",), edt_faults={"save": True})
        self.assertEqual(result, 1)
        self.assertTrue(captures[0].instance.exists())
        self.assertEqual((captures[0].instance / "tracing_on").read_text(), "0\n")
        self.assertEqual(len(details["edt_wrappers"]["remaining_definitions"]), 8)
        self.assertEqual(len(details["edt_wrappers"]["formats"]), 8)
        self.assertEqual(captures[0].probe_owners[0].removals, 0)
        for name in details["edt_wrappers"]["remaining_definitions"]:
            self.assertEqual((captures[0].instance / "events" / name / "enable").read_text(), "0\n")
        self.assert_unrelated_probes_unchanged()

    def test_active_unconfirmed_disable_retains_definitions_and_instance(self):
        """Never remove global probes after one private disable cannot be confirmed."""
        result, captures, details = self.run_main(("--edt-wrappers",), edt_faults={"disable": 2})
        self.assertEqual(result, 1)
        self.assertTrue(captures[0].instance.exists())
        self.assertEqual(len(details["edt_wrappers"]["remaining_definitions"]), 8)
        self.assertNotIn("end", details["edt_wrappers"]["deactivation"])
        self.assertEqual(captures[0].probe_owners[0].removals, 0)
        self.assertTrue(any("Retained EDT definitions" in error for error in details["cleanup_errors"]))
        self.assert_unrelated_probes_unchanged()

    def test_active_removal_failure_retains_exact_remaining_name(self):
        """Continue independent cleanup and expose the one unresolved definition."""
        result, captures, details = self.run_main(("--edt-wrappers",), edt_faults={"remove": 2})
        self.assertEqual(result, 1)
        self.assertTrue(captures[0].instance.exists())
        self.assertEqual(len(details["edt_wrappers"]["remaining_definitions"]), 1)
        self.assertEqual(captures[0].probe_owners[0].removals, 8)
        self.assert_unrelated_probes_unchanged()

    def test_active_final_identity_failure_is_reported_after_safe_cleanup(self):
        """Report module changes after stopping and still reconcile disabled definitions."""
        result, captures, details = self.run_main(("--edt-wrappers",),
                                                 edt_faults={"identity_after": True})
        self.assertEqual(result, 1)
        self.assertFalse(details["edt_wrappers"]["identity_after_stop_verified"])
        self.assertEqual(details["edt_wrappers"]["remaining_definitions"], [])
        self.assertTrue(captures[0].instance.exists())
        self.assert_unrelated_probes_unchanged()

    def test_active_preflight_failure_precedes_all_output_and_trace_mutations(self):
        """Refuse an unmatched module without creating output or a private instance."""
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
            self.run_main(("--edt-wrappers",), edt_faults={"preflight": True})
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse((self.root / "main-output").exists())
        self.assertEqual(list((self.tracefs / "instances").iterdir()),
                         [self.tracefs / "instances/existing"])
        self.assert_unrelated_probes_unchanged()

    def test_active_capture_interval_has_no_diagnostic_or_output_io(self):
        """Keep EDT profiles, formats, identity checks and cleanup outside active recording."""
        running = False
        read_text = Path.read_text
        write_text = Path.write_text

        def checked_read(path, *arguments, **keywords):
            """Reject all helper metadata reads while the private buffer records."""
            self.assertFalse(running, f"Read during active capture: {path}")
            return read_text(path, *arguments, **keywords)

        def checked_write(path, data, *arguments, **keywords):
            """Allow only the private trace stop after recording starts."""
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
            result, captures, details = self.run_main(("--edt-wrappers", "--seconds", "1"))
        self.assertEqual(result, 0)
        self.assertTrue(details["capture_completed"])
        activity = captures[0].activity
        start = activity.index(("control", "tracing_on", 1))
        self.assertEqual(activity[start + 1:start + 3], [("sleep", 1), ("control", "tracing_on", 0)])
        self.assert_unrelated_probes_unchanged()

    def test_active_unconfirmed_stop_keeps_decoder_definitions(self):
        """Preserve dynamic types and the private instance when stopping cannot be confirmed."""
        result, captures, details = self.run_main(("--edt-wrappers",), fail_stop=True)
        self.assertEqual(result, 1)
        self.assertTrue(captures[0].instance.exists())
        self.assertEqual(len(details["edt_wrappers"]["remaining_definitions"]), 8)
        self.assertEqual(captures[0].probe_owners[0].removals, 0)
        self.assertFalse(details["capture_completed"])
        self.assert_unrelated_probes_unchanged()

    def test_active_interruption_cleans_partial_probe_registration(self):
        """Reconcile an interrupted entry registration even before activation begins."""
        helper = self.probe_helper({}, [], [])
        capture = FakeCapture(self.tracefs, self.output, [45], edt_helper=helper)
        self.probe_fixture()
        original_register = capture.probes.register

        def interrupt_registration(*arguments, **keywords):
            """Deliver the signal after the kernel has accepted an owned definition."""
            original_register(*arguments, **keywords)
            raise KeyboardInterrupt

        with mock.patch.object(capture.probes, "register", side_effect=interrupt_registration):
            with self.assertRaises(KeyboardInterrupt):
                capture.setup()
        self.assertEqual(capture.close(), [])
        self.assertEqual(len(capture.edt["formats"]), 1)
        self.assertFalse(capture.instance.exists())
        self.assertEqual(capture.edt["remaining_definitions"], [])
        self.assert_unrelated_probes_unchanged()

    def test_active_missing_context_flags_refuses_registration(self):
        """Require existing context/IRQ flags without changing any trace options."""
        self.probe_fixture()
        (self.tracefs / "template/trace_options").write_text("context-info\nnoirq-info\n")
        helper = self.probe_helper({}, [], [])
        capture = FakeCapture(self.tracefs, self.output, [45], edt_helper=helper)
        with self.assertRaisesRegex(RuntimeError, "requires existing"):
            capture.setup()
        self.assertEqual(capture.probes.attempted, [])
        self.assertEqual((capture.instance / "trace_options").read_text(), "context-info\nnoirq-info\n")
        self.assertEqual(capture.close(), [])
        self.assert_unrelated_probes_unchanged()

    def test_active_changed_definition_is_preserved_without_enable_write(self):
        """Refuse controls and removal when another actor changes an attempted definition."""
        self.probe_fixture()
        helper = self.probe_helper({}, [], [])
        capture = FakeCapture(self.tracefs, self.output, [45], edt_helper=helper)
        capture.setup()
        name, definition = capture.probes.attempted[0]
        control = capture.instance / "events" / name / "enable"
        contents = (self.tracefs / "kprobe_events").read_text().replace(definition, definition + " other=%si:x64")
        (self.tracefs / "kprobe_events").write_text(contents)
        errors = capture.close()
        self.assertTrue(any("Definition changed" in error for error in errors))
        self.assertEqual(control.read_text(), "1\n")
        self.assertEqual(len(capture.edt["remaining_definitions"]), 8)
        self.assertEqual(capture.probes.removals, 0)
        self.assertTrue(capture.instance.exists())
        self.assert_unrelated_probes_unchanged()

    def test_active_duration_is_rejected_before_diagnostics(self):
        """Reject EDT traces longer than five seconds before loading helpers or reading proc."""
        arguments = [str(script), "--tid", "123", "--output", str(self.output),
                     "--edt-wrappers", "--seconds", "5.001"]
        with mock.patch("sys.argv", arguments), \
                mock.patch.object(camera_trace.os, "geteuid", return_value=0), \
                mock.patch.object(camera_trace, "read_if_present") as read, \
                mock.patch.object(camera_trace, "edt_probe_helper") as helper, \
                mock.patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
            camera_trace.main()
        self.assertEqual(raised.exception.code, 2)
        read.assert_not_called()
        helper.assert_not_called()

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
