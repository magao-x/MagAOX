r"""\file native_linux_build_test.py
\brief Offline build-orchestration fault injection; no compiler, native app or Linux build is executed.
\ingroup loPredCtrlAcc_unit_test
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import native_linux_build as build


class NativeBuildTest(unittest.TestCase):
    """Verify host refusal, immutable input snapshots, output confinement and independent failure reporting."""

    def setUp(self):
        """Create a test-owned source fixture and a distinct, initially absent build directory."""
        self.temporary = tempfile.TemporaryDirectory(prefix="ddspc-build-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.source = self.base / "repo"
        self.source.mkdir()
        (self.source / "sample.cpp").write_text("original source\n")
        self.output = self.base / "build"
        self.commands = []
        self.fail_profile = None
        self.omit_binary = False
        self.addCleanup(patch.stopall)
        patch.object(build, "ROOT", self.source).start()
        patch.object(build.platform, "system", return_value="Linux").start()
        patch.object(build.platform, "platform", return_value="simulated Linux: no native build").start()
        patch.object(build.socket, "gethostname", return_value="offline-development").start()
        patch.object(build.shutil, "which", side_effect=lambda name: "/fake/" + name).start()
        patch.object(build.subprocess, "check_output", side_effect=self.read_metadata).start()
        patch.object(build, "run", side_effect=self.build_command).start()
        patch.object(sys, "argv", ["native_linux_build.py", "--build-root", str(self.output)]).start()

    def read_metadata(self, arguments, **kwargs):
        """Provide explicit fake Git/compiler metadata; this test is not native compilation evidence."""
        if arguments[0] == "git":
            return b"sample.cpp\0" if arguments[1] == "ls-files" else b"test-head\n"
        self.assertEqual(arguments[1:], ["--version"])
        return b"FAKE compiler for orchestration tests only\n"

    def build_command(self, arguments, directory, environment, log):
        """Reject all non-build invocations and simulate metadata, successful links or injected build failures."""
        self.commands.append(arguments)
        self.assertEqual(environment["GIT_OPTIONAL_LOCKS"], "0")
        if arguments[0] == "/fake/gengithead.sh":
            Path(arguments[2]).write_text('#define MAGAOX_CURRENT_SHA1 "test-head"\n')
            return
        self.assertEqual(arguments[0], "make")  # No native executable, install, launch or remote command is allowed.
        self.assertNotIn("install", arguments)
        self.assertNotIn("apps/loPredCtrl", arguments)
        self.assertTrue(directory.is_relative_to(self.output))
        self.assertEqual((directory / "sample.cpp").read_text(), "original source\n")
        if directory.name == "legacy":
            # Simulate a source edit while the first build is running: the second must use the same frozen bytes.
            (self.source / "sample.cpp").write_text("changed during build\n")
        if "apps/loPredCtrlAcc" in arguments:
            if directory.name == self.fail_profile:
                raise subprocess.CalledProcessError(2, arguments)
            if not self.omit_binary:
                binary = directory / "apps/loPredCtrlAcc/loPredCtrlAcc"
                binary.parent.mkdir(parents=True)
                binary.write_bytes(b"TEST FIXTURE ONLY: not an executable")

    def invoke(self):
        """Suppress fixture-only progress output while retaining real exceptions and assertions."""
        with contextlib.redirect_stdout(io.StringIO()):
            build.main()

    def test_identical_snapshots_and_separate_flags(self):
        """Both profiles must consume identical frozen bytes and use their declared optimization flags."""
        self.invoke()
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["files"]["sample.cpp"], hashlib.sha256(b"original source\n").hexdigest())
        for profile, flag in (("legacy", "-ffast-math"), ("shadow", "-fno-fast-math")):
            result = manifest["profiles"][profile]
            self.assertTrue(result["linked"])
            self.assertIn("OPTIMIZE=" + flag, result["make_variables"])
            self.assertEqual((self.output / profile / "sample.cpp").read_text(), "original source\n")
        self.assertTrue(any("DDSPC_SHADOW_BUILD=yes" in command for command in self.commands))

    def test_failure_does_not_skip_other_profile(self):
        """A failed Legacy build must remain failed while the independent shadow attempt still runs."""
        self.fail_profile = "legacy"
        with self.assertRaises(SystemExit) as error:
            self.invoke()
        self.assertEqual(error.exception.code, 1)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertFalse(manifest["profiles"]["legacy"]["linked"])
        self.assertEqual(manifest["profiles"]["legacy"]["status"], "failed")
        self.assertTrue(manifest["profiles"]["shadow"]["linked"])

    def test_successful_make_without_binary_is_failure(self):
        """A masked make failure or missing link artifact must never produce a successful manifest."""
        self.omit_binary = True
        with self.assertRaises(SystemExit):
            self.invoke()
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertTrue(all(not result["linked"] for result in manifest["profiles"].values()))

    def test_refuses_instrument_and_non_linux_hosts(self):
        """Host refusal must precede any staging, metadata generation or build command."""
        for system, host in (("Darwin", "offline-development"), ("Linux", "EXAO-test")):
            with self.subTest(system=system, host=host), patch.object(build.platform, "system", return_value=system), \
                    patch.object(build.socket, "gethostname", return_value=host), self.assertRaises(SystemExit):
                self.invoke()
        self.assertEqual(self.commands, [])
        self.assertFalse(self.output.exists())

    def test_refuses_output_inside_source(self):
        """An output path within the source tree must be rejected before creating any files."""
        with patch.object(sys, "argv", ["native_linux_build.py", "--build-root", str(self.source / "output")]), \
                self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse((self.source / "output").exists())
        self.assertEqual(self.commands, [])

    def test_rejects_source_symlink(self):
        """Directory symlinks must not be silently omitted or followed outside the declared source snapshot."""
        (self.source / "linked").symlink_to(self.base, target_is_directory=True)
        with self.assertRaises(RuntimeError):
            build.stage_sources(["linked"], self.base / "snapshot")

    def test_metadata_wrapper_is_frozen_and_confined(self):
        """The real wrapper only copies frozen metadata and rejects an output that escapes its stage."""
        stage = self.base / "stage"
        stage.mkdir()
        version = self.base / "version.h"
        version.write_text("frozen metadata\n")
        wrapper = build.write_wrapper(stage, version) / "gengithead.sh"
        # Execute only this generated pure-file-copy Python wrapper; no fake or real application is invoked.
        output = stage / "version.h"
        subprocess.run([sys.executable, str(wrapper), str(stage), str(output), "MAGAOX"], check=True)
        self.assertEqual(output.read_text(), "frozen metadata\n")
        escaped = self.base / "escaped.h"
        result = subprocess.run([sys.executable, str(wrapper), str(stage), str(escaped), "MAGAOX"], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(escaped.exists())


if __name__ == "__main__":
    unittest.main()
