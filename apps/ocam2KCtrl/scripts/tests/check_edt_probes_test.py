#!/usr/bin/env python3
## \file check_edt_probes_test.py
# \brief Offline ownership, identity, and failure checks for disabled EDT probes.
# \ingroup ocam2KCtrl_unit_test

"""Use ordinary temporary files and mocked writes, never a kernel tracing mount."""

import contextlib
import errno
import importlib.util
import io
import os
from pathlib import Path
import shutil
import signal
import struct
import tempfile
import unittest
from unittest import mock


script = Path(__file__).resolve().parents[1] / "check-edt-probes.py"
spec = importlib.util.spec_from_file_location("edt_probes", script)
edt_probes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(edt_probes)


class FakeKernel:
    """Emulate registration side effects inside an ordinary temporary directory."""

    def __init__(self, root):
        """Create controls containing an unrelated definition and private instance."""
        self.root = root
        self.definitions = {"other/keep": "p:other/keep unrelated_function"}
        self.commands = []
        self.reject = set()
        self.short = set()
        self.fail_after_apply = set()
        self.enabled = set()
        self.after_write = None
        (root / "instances/existing").mkdir(parents=True)
        (root / "trace").write_text("other trace data\n")
        (root / "tracing_on").write_text("1\n")
        self.save()

    def save(self):
        """Render the fixture's current definitions as a kernel readback would."""
        (self.root / "kprobe_events").write_text("\n".join(self.definitions.values()) + "\n")

    def write(self, descriptor, data):
        """Apply one command, optionally injecting rejection, short count, or failure."""
        command = data.decode("ascii").strip()
        self.commands.append(command)
        if command in self.reject:
            raise OSError(errno.EINVAL, "Injected rejection")
        head = command.split()[0]
        name = head.split(":", 1)[1]
        if command.startswith("-:"):
            for path in [self.root] + list((self.root / "instances").glob("*")):
                if (path / "events" / name / "enable").read_text().strip() != "0":
                    raise OSError(errno.EBUSY, "Enabled elsewhere")
            self.definitions.pop(name, None)
            for path in [self.root] + list((self.root / "instances").glob("*")):
                shutil.rmtree(path / "events" / name)
                directory = path / "events" / name.split("/")[0]
                if not any(directory.iterdir()):
                    directory.rmdir()
        else:
            if name in self.definitions:
                raise OSError(errno.EEXIST, "Existing fixture event")
            self.definitions[name] = command
            for path in [self.root] + list((self.root / "instances").glob("*")):
                event = path / "events" / name
                event.mkdir(parents=True)
                value = "1\n" if name in self.enabled else "0\n"
                (event / "enable").write_text(value)
        self.save()
        if self.after_write:
            self.after_write(command)
        if command in self.fail_after_apply:
            raise OSError(errno.EIO, "Failure after application")
        return len(data) - int(command in self.short)


class ProbeCheckTest(unittest.TestCase):
    """Verify append-only mutation and exact event ownership across failure paths."""

    def setUp(self):
        """Prepare unprivileged ordinary files and intercept only command writes."""
        self.temporary = tempfile.TemporaryDirectory(prefix="edt-probes-fake-")
        self.root = Path(self.temporary.name)
        self.kernel = FakeKernel(self.root)
        self.check = edt_probes.ProbeCheck(self.root, "ocam_edt_check_test")
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.open_mock = mock.patch.object(edt_probes.os, "open", wraps=os.open).start()
        mock.patch.object(edt_probes.os, "write", side_effect=self.kernel.write).start()

    def tearDown(self):
        """Protect unrelated controls, then discard the temporary fake filesystem."""
        self.assertEqual((self.root / "trace").read_text(), "other trace data\n")
        self.assertEqual((self.root / "tracing_on").read_text(), "1\n")
        self.assertEqual(self.kernel.definitions["other/keep"], "p:other/keep unrelated_function")
        command_opens = [call for call in self.open_mock.call_args_list
                         if call.args[1] & os.O_WRONLY]
        self.assertEqual(len(command_opens), len(self.kernel.commands))
        for call in command_opens:
            self.assertEqual(call.args[0], self.root / "kprobe_events")
            self.assertEqual(call.args[1], os.O_WRONLY | os.O_APPEND)
        mock.patch.stopall()
        self.stdout.__exit__(None, None, None)
        self.temporary.cleanup()

    def definition(self, function="edt_test_intr", kind="p"):
        """Return the expected event name and exact module-qualified registration."""
        name = f"{self.check.group}/{kind}_{function}"
        probe = "p" if kind == "p" else "r1"
        return name, f"{probe}:{name} edt:{function}"

    def test_all_twenty_disabled_probes_and_reverse_cleanup(self):
        """Register every candidate with deterministic return reservation and no enables."""
        self.check.guard_group()
        for function in edt_probes.FUNCTIONS:
            for kind in ("p", "r"):
                self.check.register(function, kind)
        self.assertEqual(len(self.check.attempted), 20)
        self.assertTrue(all(" edt:" in command for command in self.kernel.commands))
        self.assertEqual(sum(command.startswith("r1:") for command in self.kernel.commands), 10)
        expected = [f"-:{name} {definition.split()[1]}"
                    for name, definition in reversed(self.check.attempted)]
        self.check.cleanup()
        self.assertEqual(self.kernel.commands[20:], expected)
        self.assertEqual(self.kernel.definitions, {"other/keep": "p:other/keep unrelated_function"})
        self.assertEqual(self.check.errors, [])

    def test_existing_group_or_event_is_never_adopted(self):
        """Refuse definitions and empty event directories belonging to another actor."""
        name, definition = self.definition()
        self.kernel.definitions[name] = definition
        self.kernel.save()
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            self.check.guard_group()
        with self.assertRaisesRegex(RuntimeError, "existing event"):
            self.check.register("edt_test_intr", "p")
        self.check.cleanup()
        self.assertIn(name, self.kernel.definitions)
        self.assertEqual(self.kernel.commands, [])
        del self.kernel.definitions[name]
        self.kernel.save()
        (self.root / "events" / self.check.group).mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            self.check.guard_group()

    def test_failed_registration_cleans_prior_success(self):
        """A rejected second command does not prevent prior definition removal."""
        self.check.register("edt_test_intr", "p")
        _, rejected = self.definition(kind="r")
        self.kernel.reject.add(rejected)
        with self.assertRaises(OSError):
            self.check.register("edt_test_intr", "r")
        self.check.cleanup()
        self.assertEqual(list(self.kernel.definitions), ["other/keep"])
        self.assertEqual(self.check.errors, [])

    def test_short_or_failed_write_after_application_is_reconciled(self):
        """Remove an exact owned definition even when its write result was uncertain."""
        for mode in (self.kernel.short, self.kernel.fail_after_apply):
            with self.subTest(mode=mode):
                name, definition = self.definition()
                mode.add(definition)
                with self.assertRaises(OSError):
                    self.check.register("edt_test_intr", "p")
                self.assertIn(name, self.kernel.definitions)
                self.check.cleanup()
                self.assertNotIn(name, self.kernel.definitions)
                self.assertEqual(self.check.errors, [])
                mode.clear()
                self.check.attempted.clear()

    def test_changed_definition_is_preserved_and_named(self):
        """Do not delete a collided or replaced definition with the same event name."""
        self.check.register("edt_test_intr", "p")
        name, _ = self.definition()
        self.kernel.definitions[name] = f"p:{name} other:symbol"
        self.kernel.save()
        self.check.cleanup()
        self.assertIn(name, self.kernel.definitions)
        self.assertIn(name, self.check.errors[0])
        self.assertFalse(any(command.startswith(f"-:{name} ")
                             for command in self.kernel.commands))

    def test_cleanup_failure_does_not_stop_other_removals(self):
        """Report the exact retained event while removing every other owned event."""
        self.check.register("edt_test_intr", "p")
        self.check.register("edt_test_intr", "r")
        name, _ = self.definition(kind="r")
        self.kernel.reject.add(f"-:{name} edt:edt_test_intr")
        self.check.cleanup()
        self.assertIn(name, self.kernel.definitions)
        self.assertEqual(len(self.kernel.definitions), 2)
        self.assertIn(name, self.check.errors[0])

    def test_external_instance_enable_is_reported_without_disable(self):
        """Detect an external consumer without changing its enable state."""
        name, _ = self.definition()
        self.kernel.enabled.add(name)
        self.kernel.after_write = lambda command: (
            self.root / "events" / name / "enable").write_text("0\n")
        with self.assertRaisesRegex(RuntimeError, "enabled"):
            self.check.register("edt_test_intr", "p")
        self.check.cleanup()
        self.assertIn(name, self.kernel.definitions)
        self.assertIn(name, self.check.errors[0])
        self.assertEqual((self.root / "events" / name / "enable").read_text(), "0\n")
        self.assertEqual((self.root / "instances/existing/events" / name / "enable").read_text(), "1\n")

    def test_output_failure_cannot_interrupt_cleanup(self):
        """A closed output pipe produces failure status while every owned probe is removed."""
        with mock.patch.object(edt_probes, "preflight"), \
                mock.patch.object(edt_probes, "check_identity"), \
                mock.patch.object(edt_probes, "ProbeCheck", return_value=self.check), \
                mock.patch.object(edt_probes.signal, "signal"), \
                mock.patch("builtins.print", side_effect=BrokenPipeError("Closed pipe")):
            result = edt_probes.main(["--tracefs", str(self.root)])
        self.assertEqual(result, 1)
        self.assertEqual(len(self.check.attempted), 20)
        self.assertEqual(list(self.kernel.definitions), ["other/keep"])

    def test_short_cleanup_write_is_reported_even_if_removal_applied(self):
        """Keep cleanup errors visible despite successful uncertain removal side effects."""
        self.check.register("edt_test_intr", "p")
        name, _ = self.definition()
        self.kernel.short.add(f"-:{name} edt:edt_test_intr")
        self.check.cleanup()
        self.assertNotIn(name, self.kernel.definitions)
        self.assertIn(name, self.check.errors[0])

    def test_signal_during_successful_write_still_cleans_and_returns_failure(self):
        """A signal between registration and bookkeeping cannot orphan a probe."""
        self.kernel.after_write = lambda command: self.check.interrupt(signal.SIGTERM, None)
        with mock.patch.object(edt_probes, "preflight"), \
                mock.patch.object(edt_probes, "check_identity"), \
                mock.patch.object(edt_probes, "ProbeCheck", return_value=self.check), \
                mock.patch.object(edt_probes.signal, "signal"):
            result = edt_probes.main(["--tracefs", str(self.root)])
        self.assertEqual(result, 1)
        self.assertEqual(len(self.check.attempted), 1)
        self.assertEqual(list(self.kernel.definitions), ["other/keep"])

    def test_post_registration_identity_failure_cleans_and_returns_failure(self):
        """A module identity change after registration causes failure and cleanup."""
        with mock.patch.object(edt_probes, "preflight"), \
                mock.patch.object(edt_probes, "check_identity", side_effect=RuntimeError("Changed EDT")), \
                mock.patch.object(edt_probes, "ProbeCheck", return_value=self.check), \
                mock.patch.object(edt_probes.signal, "signal"):
            result = edt_probes.main(["--tracefs", str(self.root)])
        self.assertEqual(result, 1)
        self.assertEqual(len(self.check.attempted), 20)
        self.assertEqual(list(self.kernel.definitions), ["other/keep"])

    def test_preflight_failure_makes_no_mutations(self):
        """Reject missing root, mounts, or identity before any registration write."""
        with mock.patch.object(edt_probes, "preflight", side_effect=RuntimeError("Wrong build")), \
                mock.patch.object(edt_probes, "ProbeCheck", return_value=self.check):
            self.assertEqual(edt_probes.main(["--tracefs", str(self.root)]), 1)
        self.assertEqual(self.kernel.commands, [])

    def test_actual_preflight_requires_root_and_tracefs_mount(self):
        """Require the tracefs filesystem rather than an arbitrary writable directory."""
        with mock.patch.object(edt_probes.os, "geteuid", return_value=1000):
            with self.assertRaisesRegex(RuntimeError, "root"):
                edt_probes.preflight(self.root)
        read_text = Path.read_text
        filesystem = "tracefs"

        def read_mountinfo(path, *args, **kwargs):
            """Supply only synthetic mount metadata while retaining normal fixture reads."""
            if path == Path("/proc/self/mountinfo"):
                return f"30 1 0:1 / {self.root} rw - {filesystem} tracefs rw\n"
            return read_text(path, *args, **kwargs)

        with mock.patch.object(edt_probes.os, "geteuid", return_value=0), \
                mock.patch.object(edt_probes, "check_identity") as identity, \
                mock.patch.object(Path, "read_text", read_mountinfo):
            edt_probes.preflight(self.root)
            identity.assert_called_once_with()
            filesystem = "ext4"
            with self.assertRaisesRegex(RuntimeError, "mounted tracefs"):
                edt_probes.preflight(self.root)
        self.assertEqual(self.kernel.commands, [])

    def test_explicit_zero_entry_offset_is_accepted(self):
        """Accept the kernel's equivalent explicit zero-offset entry rendering."""
        def render_offset(command):
            """Add the kernel's optional zero suffix to a newly created entry."""
            if command.startswith("p:"):
                name = command.split()[0].split(":", 1)[1]
                self.kernel.definitions[name] = command + "+0x0"
                self.kernel.save()
        self.kernel.after_write = render_offset
        self.check.register("edt_test_intr", "p")
        self.check.cleanup()
        self.assertEqual(self.check.errors, [])
        self.assertEqual(list(self.kernel.definitions), ["other/keep"])


class IdentityTest(unittest.TestCase):
    """Decode real GNU note structure and reject incomplete or mismatched identity."""

    def note(self, descriptor):
        """Construct an aligned little-endian GNU ELF note for the given descriptor."""
        return struct.pack("<III", 4, len(descriptor), 3) + b"GNU\0" + descriptor

    def test_build_id_note_and_malformed_rejection(self):
        """Extract the verified ID and reject missing or truncated descriptors."""
        note = self.note(bytes.fromhex(edt_probes.BUILD_ID))
        self.assertEqual(edt_probes.build_id(note), edt_probes.BUILD_ID)
        for invalid in (b"", note[:-1], struct.pack("<III", 4, 20, 3)):
            with self.subTest(note=invalid), self.assertRaises(ValueError):
                edt_probes.build_id(invalid)

    def test_loaded_identity_rejects_every_mismatch(self):
        """Require kernel release, module version, source version, and GNU build ID."""
        with tempfile.TemporaryDirectory(prefix="edt-identity-fake-") as temporary:
            module = Path(temporary)
            (module / "notes").mkdir()
            expected = {"version": edt_probes.VERSION, "srcversion": edt_probes.SRCVERSION}
            for name, value in expected.items():
                (module / name).write_text(value + "\n")
            (module / "notes/.note.gnu.build-id").write_bytes(
                self.note(bytes.fromhex(edt_probes.BUILD_ID)))
            with mock.patch.object(edt_probes.platform, "release", return_value=edt_probes.KERNEL):
                edt_probes.check_identity(module)
                for name, value in expected.items():
                    (module / name).write_text("different\n")
                    with self.assertRaisesRegex(RuntimeError, "differs"):
                        edt_probes.check_identity(module)
                    (module / name).write_text(value)
                (module / "notes/.note.gnu.build-id").write_bytes(self.note(bytes(20)))
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    edt_probes.check_identity(module)
            with mock.patch.object(edt_probes.platform, "release", return_value="different"):
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    edt_probes.check_identity(module)


if __name__ == "__main__":
    unittest.main()
