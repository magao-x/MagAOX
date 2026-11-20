#!/usr/bin/env python3
## \file remove_cpusets_test.py
## \brief Verify RTC cpuset cleanup with temporary files and mocked removals.

import contextlib
import importlib.machinery
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "remove_cpusets"
LOADER = importlib.machinery.SourceFileLoader("remove_cpusets", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
cleanup = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(cleanup)


class CleanupTest(unittest.TestCase):
    """Use an ordinary temporary tree; never read or modify actual cgroups."""

    def setUp(self):
        """Create fake control files, including occupied root/system groups."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.mountpoint = Path(self.temporary.name)
        self.group("", "1\n")
        self.group("system", "2\n")
        self.mounts = [(self.mountpoint, "/", "cgroup", {"rw", "cpuset"})]

    def group(self, name, tasks=""):
        """Create the named fake cpuset with the supplied task-list contents."""
        directory = self.mountpoint / name
        directory.mkdir(parents=True, exist_ok=True)

        for control, content in (("tasks", tasks), ("cpuset.cpus", "33\n"), ("cpuset.mems", "0\n")):
            (directory / control).write_text(content)

        return directory

    def run_main(self, arguments, remove=None):
        """Run the CLI against fake mount metadata and intercept every removal."""
        output = io.StringIO()

        with mock.patch.object(cleanup, "configured_mountpoint", return_value=self.mountpoint), \
             mock.patch.object(cleanup, "mount_records", return_value=self.mounts), \
             mock.patch.object(cleanup.os, "geteuid", return_value=0), \
             mock.patch.object(cleanup.sys, "argv", [str(SCRIPT), *arguments]), \
             mock.patch.object(Path, "rmdir", autospec=True, side_effect=remove) as removals, \
             contextlib.redirect_stdout(output):
            cleanup.main()

        return output.getvalue(), removals

    def test_dry_run_preserves_root_system_and_empty_descendants(self):
        """Preview child-before-parent removals without touching any directory."""
        parent = self.group("camwfs")
        child = self.group("camwfs/helper")
        output, removals = self.run_main([])
        removals.assert_not_called()
        self.assertLess(output.index(f"Would remove {child}"), output.index(f"Would remove {parent}\n"))
        self.assertEqual((self.mountpoint / "tasks").read_text(), "1\n")
        self.assertEqual((self.mountpoint / "system/tasks").read_text(), "2\n")
        self.assertTrue(child.is_dir())

    def test_busy_child_and_parent_refuse_before_removing_anything(self):
        """Both thread lists are reported, and even other empty sets stay intact."""
        self.group("camwfs", "999999999\n")
        self.group("camwfs/helper", "999999998\n")
        self.group("spare35")

        def unexpected_remove(directory):
            """Fail if an occupied-tree preflight permits even one removal."""
            self.fail(f"Attempted removal during a failed preflight: {directory}")

        with self.assertRaisesRegex(RuntimeError, "999999998.*\n.*999999999"):
            self.run_main(["--apply"], unexpected_remove)

    def test_unknown_top_level_group_refuses_entire_cleanup(self):
        """An unrelated cgroup must never be silently removed or migrated."""
        self.group("camwfs")
        self.group("unrelated")

        with self.assertRaisesRegex(RuntimeError, "Unknown cpuset.*unrelated"):
            self.run_main(["--apply"])

    def test_apply_only_calls_rmdir_deepest_first(self):
        """Model kernel removal of virtual controls while preserving root/system."""
        parent = self.group("camwfs")
        child = self.group("camwfs/helper")
        actual_rmdir = Path.rmdir

        def remove(directory):
            """Emulate virtual control-file disappearance on an empty fake group."""
            self.assertNotIn(directory, [self.mountpoint, self.mountpoint / "system"])

            for control in directory.iterdir():
                self.assertTrue(control.is_file())
                control.unlink()

            actual_rmdir(directory)

        output, removals = self.run_main(["--apply"], remove)
        self.assertEqual(removals.call_args_list, [mock.call(child), mock.call(parent)])
        self.assertIn("Removed 2 cpuset directories", output)
        self.assertTrue((self.mountpoint / "system").is_dir())
        self.assertEqual((self.mountpoint / "system/tasks").read_text(), "2\n")

    def test_mount_must_be_complete_cgroup_v1_cpuset(self):
        """Reject ordinary filesystems, cgroup v2, other controllers and subtrees."""
        for root, filesystem, options in (("/", "tmpfs", {"cpuset"}),
                                           ("/", "cgroup2", {"cpuset"}),
                                           ("/", "cgroup", {"cpu"}),
                                           ("/container", "cgroup", {"cpuset"})):
            with self.subTest(filesystem=filesystem, root=root, options=options):
                with self.assertRaisesRegex(RuntimeError, "not a complete"):
                    cleanup.removal_plan(self.mountpoint, {"camwfs"},
                                         [(self.mountpoint, root, filesystem, options)])

    def test_new_tasks_during_apply_stop_further_removal(self):
        """Recheck a parent after its child is removed to catch a restarted thread."""
        parent = self.group("camwfs")
        child = self.group("camwfs/helper")
        removed = []

        def remove(directory):
            """Model a thread entering the parent after the child removal."""
            removed.append(directory)
            (parent / "tasks").write_text("999999999\n")

        with self.assertRaisesRegex(RuntimeError, "Occupied cpusets"):
            self.run_main(["--apply"], remove)

        self.assertEqual(removed, [child])

    def test_apply_requires_root_before_inspecting_cgroups(self):
        """Reject an unprivileged mutation request without accessing any mount."""
        with mock.patch.object(cleanup.os, "geteuid", return_value=1000), \
             mock.patch.object(cleanup.sys, "argv", [str(SCRIPT), "--apply"]), \
             mock.patch.object(cleanup, "configured_mountpoint") as mountpoint:
            with self.assertRaisesRegex(RuntimeError, "requires root"):
                cleanup.main()

            mountpoint.assert_not_called()

    def test_missing_control_file_refuses_removal(self):
        """Do not treat an arbitrary directory as an empty cpuset."""
        parent = self.group("camwfs")
        (parent / "tasks").unlink()

        with self.assertRaisesRegex(RuntimeError, "Missing or unsafe cpuset control"):
            self.run_main(["--apply"])

    def test_nested_mount_and_symlink_are_rejected(self):
        """Never traverse an unrelated mount or follow a link outside the tree."""
        parent = self.group("camwfs")

        with self.assertRaisesRegex(RuntimeError, "nested mount"):
            cleanup.removal_plan(self.mountpoint, {"camwfs"},
                                 [*self.mounts, (parent, "/", "tmpfs", {"rw"})])

        (parent / "link").symlink_to(self.mountpoint / "system", target_is_directory=True)

        with self.assertRaisesRegex(RuntimeError, "symlink"):
            cleanup.removal_plan(self.mountpoint, {"camwfs"}, self.mounts)

    def test_allowlist_combines_legacy_and_sibling_setup_names(self):
        """Read quoted/unquoted declarations and fail closed on computed names."""
        setup = self.mountpoint / "make_cpusets"
        setup.write_text('mkdir -p $cpusetMount/system\nmkdir -p "$cpusetMount/new-name"\n')
        names = cleanup.allowed_names(setup)
        self.assertIn("camacq", names)
        self.assertIn("new-name", names)
        self.assertNotIn("system", names)
        setup.write_text('mkdir -p $cpusetMount/system\nmkdir -p "$cpusetMount/$name"\n')

        with self.assertRaisesRegex(RuntimeError, "Unrecognized"):
            cleanup.allowed_names(setup)


if __name__ == "__main__":
    unittest.main()
