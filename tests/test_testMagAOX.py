"""Offline regression tests for persistent test logs and failure propagation."""

import csv
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class TestRunnerTests(unittest.TestCase):
    """Exercise the real shell scripts using disposable test commands."""

    def setUp(self):
        """Create a fixture tree and stub the generated logger test commands."""
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.tests = self.root / "tests"
        self.tests.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        source = Path(__file__).resolve().parent
        shutil.copy2(source / "testMagAOX.bash", self.tests)
        shutil.copy2(source / "Makefile", self.tests)
        (self.tests / "coverage").mkdir()
        shutil.copy2(source / "coverage/update_coverage", self.tests / "coverage")
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}")
        self.write_command(self.bin / "make", """
printf 'logger stdout: %s\\n' "$*"
printf 'logger stderr\\n' >&2
if [[ "$*" == *do_catch2_test* ]]; then exit "${LOGGER_CATCH_EXIT:-0}"; fi
exit "${LOGGER_ENTROPY_EXIT:-0}"
""")

    def write_command(self, path, body):
        """Write an executable Bash fixture without touching repository files."""
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)

    def run_script(self, *arguments, input_text=""):
        """Run from outside tests/ and capture the terminal output too."""
        return subprocess.run(["/bin/bash", *arguments], cwd=self.root, env=self.env,
                              input=input_text, text=True, capture_output=True, timeout=15)

    def summary(self):
        """Read the latest run's per-command results."""
        with (self.tests / "results/latest/summary.tsv").open() as stream:
            return list(csv.DictReader(stream, delimiter="\t"))

    def test_both_streams_and_crlf_list(self):
        """Save all output, accept CRLF, and process a final unterminated line."""
        self.write_command(self.tests / "pass", "printf 'test stdout\\n'; printf 'test stderr\\n' >&2\n")
        listing = b"# comment\r\n\r\n./pass"
        (self.tests / "tests.list").write_bytes(listing)
        result = self.run_script("tests/testMagAOX.bash")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        log = (self.tests / "results/latest/run.log").read_text()
        for message in ("test stdout", "test stderr", "logger stdout", "logger stderr"):
            self.assertIn(message, log)
            self.assertIn(message, result.stdout)
        self.assertEqual([row["result"] for row in self.summary()], ["PASS"] * 3)
        self.assertEqual((self.tests / "tests.list").read_bytes(), listing)

    def test_failures_crashes_and_missing_executables_continue(self):
        """Record raw exit codes and keep running after ordinary failures and aborts."""
        self.write_command(self.tests / "fail", "exit 7\n")
        self.write_command(self.tests / "abort", 'ulimit -c 0\nkill -ABRT "$$"\n')
        self.write_command(self.tests / "pass", "exit 0\n")
        (self.tests / "tests.list").write_text("./fail\n./abort\n./missing\n./pass\n")
        self.env["LOGGER_CATCH_EXIT"] = "2"
        result = self.run_script("-e", "tests/testMagAOX.bash")
        self.assertEqual(result.returncode, 1)
        self.assertEqual([int(row["exit_code"]) for row in self.summary()], [7, 134, 127, 0, 2, 0])
        self.assertIn("Summary: 2 passed, 4 failed", result.stdout)

    def test_stdin_does_not_consume_the_test_list(self):
        """A test reading stdin must not silently skip subsequent tests."""
        self.write_command(self.tests / "reader", 'read -r value\nprintf "read: %s\\n" "$value"\n')
        self.write_command(self.tests / "pass", "exit 0\n")
        (self.tests / "tests.list").write_text("./reader\n./pass\n")
        result = self.run_script("tests/testMagAOX.bash", input_text="user input\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("read: user input", result.stdout)
        self.assertEqual(len(self.summary()), 4)

    def test_previous_run_survives_and_latest_advances(self):
        """Keep completed logs even when consecutive runs share a timestamp."""
        (self.tests / "tests.list").write_text("")
        self.assertEqual(self.run_script("tests/testMagAOX.bash").returncode, 0)
        first = (self.tests / "results/latest").resolve()
        first_log = (first / "run.log").read_text()
        self.env["LOGGER_ENTROPY_EXIT"] = "3"
        self.assertEqual(self.run_script("tests/testMagAOX.bash").returncode, 1)
        self.assertNotEqual(first, (self.tests / "results/latest").resolve())
        self.assertEqual((first / "run.log").read_text(), first_log)
        self.assertEqual(self.summary()[-1]["exit_code"], "3")

    def test_log_write_failure_is_not_success(self):
        """Fail the run if tee cannot save a test's output."""
        (self.tests / "tests.list").write_text("")
        self.env["TEE_COUNT"] = str(self.root / "tee-count")
        self.write_command(self.bin / "tee", """
count=0
[[ ! -f "$TEE_COUNT" ]] || read -r count < "$TEE_COUNT"
count=$((count + 1))
printf '%s\\n' "$count" > "$TEE_COUNT"
(( count != 3 )) || exit 1
exec /usr/bin/tee "$@"
""")
        result = self.run_script("tests/testMagAOX.bash")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Could not save complete test output", result.stderr)

    def test_make_test_uses_logging_runner(self):
        """The Makefile entrypoint creates saved results and propagates failures."""
        (self.tests / "tests.list").write_text("")
        self.env["LOGGER_CATCH_EXIT"] = "2"
        result = subprocess.run(["/usr/bin/make", "-C", str(self.tests), "test"], env=self.env,
                                text=True, capture_output=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([row["result"] for row in self.summary()], ["FAIL", "PASS"])

    def test_coverage_is_generated_after_failure(self):
        """Generate HTML after failed tests while preserving their exit status."""
        trace = self.root / "coverage-calls"
        self.env["TRACE"] = str(trace)
        self.write_command(self.bin / "bash", 'printf "tests\\n" >> "$TRACE"\nexit 3\n')
        for command in ("lcov", "genhtml"):
            self.write_command(self.bin / command, f'printf "{command} %s\\n" "$*" >> "$TRACE"\n')
        result = self.run_script("tests/coverage/update_coverage")
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        calls = trace.read_text()
        self.assertIn("tests\n", calls)
        self.assertIn("--capture", calls)
        self.assertIn("genhtml coverage_filtered.info", calls)


if __name__ == "__main__":
    unittest.main()
