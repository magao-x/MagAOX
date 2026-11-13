r"""\file offline_cli_test.py
\brief Exercise only the standalone numerical CLI, argument rejection and private artifact ownership.
"""
import csv
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class OfflineCliTest(unittest.TestCase):
    """No native app/INDI/stream command is permitted by this fixture."""

    def setUp(self):
        """Require an explicitly selected standalone binary, never an application name."""
        self.binary = Path(os.environ["DDSPC_OFFLINE_CLI"]).resolve()
        self.assertEqual(self.binary.name, "adc_ddspc_sim")

    def run_cli(self, *args):
        """Invoke virtual-time numerical work with a bounded test timeout."""
        return subprocess.run([str(self.binary), *map(str, args)], text=True, capture_output=True, timeout=60)

    def test_help_and_invalid_arguments(self):
        """Reject invalid or ambiguous configuration before exporting any artifacts."""
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Offline ADC-DDSPC", result.stdout)
        for args in [("--future", "0"), ("--unknown", "1"), ("--seed", "-1"),
                     ("--qrd", "2"), ("--recovery-frames", "0"), ("--max-step", "0"),
                     ("--uncorrelated-adc", "2"), ("--latency-ns", "-1"), ("--adc-drift-ppm", "10000"), ("--train", "20x"), ("--train", "200", "--train", "300")]:
            result = self.run_cli(*args)
            self.assertNotEqual(result.returncode, 0, args)

    def test_export_is_explicit_and_refuses_existing_results(self):
        """New private CSVs are readable, and a second run cannot overwrite their bytes."""
        with tempfile.TemporaryDirectory(prefix="ddspc-offline-cli-") as temporary:
            output = Path(temporary)/"report"
            result = self.run_cli("--train", 160, "--evaluate", 256, "--coupling", .3, "--output", output)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = list(csv.DictReader(line for line in result.stdout.splitlines() if not line.startswith("#")))
            self.assertEqual([row["model"] for row in rows], ["open", "standard", "assisted"])
            self.assertEqual(rows[0]["conditional_rmse"], "NA")
            for row in rows:
                self.assertGreaterEqual(float(row["variance"]), 0)
                self.assertEqual(int(row["gate_violations"]), 0)
                self.assertEqual(int(row["adc_samples"]), 4*(160+256))
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            self.assertEqual(set(before), {"metrics.csv", "residuals.csv", "psd.csv", "learning.csv", "decisions.csv", "transitions.csv"})
            learning = list(csv.DictReader(io.StringIO(before["learning.csv"].decode())))
            self.assertEqual({row["model"] for row in learning}, {"standard", "assisted"})
            self.assertGreater(max(int(row["updates"]) for row in learning), 100)
            self.assertEqual(len(list(csv.DictReader(io.StringIO(before["residuals.csv"].decode())))), 256)
            decisions = list(csv.DictReader(io.StringIO(before["decisions.csv"].decode())))
            self.assertEqual(len(decisions), 512)
            for decision in decisions:
                self.assertLessEqual(float(decision["step"]), .2)
                if decision["effective"] == "assisted":
                    self.assertEqual(decision["features_valid"], "1")
                    self.assertEqual(decision["assisted_ready"], "1")
                    self.assertLessEqual(int(decision["available_ns"]), int(decision["decision_ns"]))
            transitions = list(csv.DictReader(io.StringIO(before["transitions.csv"].decode())))
            self.assertGreater(len(transitions), 0)
            result = self.run_cli("--output", output)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
