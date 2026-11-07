r"""\file test_validate_closed_loop.py
\brief Genuine corrected-WFS feedback, frozen calibration, two-frame plant timing and controlled-disable tests.
\ingroup accelFFCtrl_unit_test
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from inspect_reduced_telemetry import sha256
from validate_closed_loop import ORIGIN, analyze_run, prepare_calibration


@unittest.skipUnless(os.environ.get("ACCELFFCTRL_CLOSED_LOOP"), "build closed-loop-tools for actual numerical feedback")
class ClosedLoopTest(unittest.TestCase):
    """Use the actual simulator/estimator/shadow and existing AO integrator; no runtime or stream dependency."""

    @classmethod
    def setUpClass(cls):
        """Prepare a deterministic short simulation and all fixed control cases once per test process."""
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.executable = Path(os.environ["ACCELFFCTRL_CLOSED_LOOP"])
        cls.metadata = prepare_calibration(cls.executable, cls.root, 12, "dual", 42)
        subprocess.run([str(cls.executable), "run", str(cls.root)], check=True)
        _, cls.report = analyze_run(cls.root, cls.metadata)
        cls.base = pd.read_csv(cls.root / "wfs_only.csv")
        cls.active = pd.read_csv(cls.root / "wfs_harmonic.csv")

    @classmethod
    def tearDownClass(cls):
        """Remove only owned local numerical fixtures."""
        cls.directory.cleanup()

    def copy_fixture(self, root):
        """Copy immutable test forcing/settings/calibration for isolated fault injection."""
        for name in ("source.txt", "source.bin", "calibration.txt"):
            shutil.copyfile(self.root / name, root / name)

    def test_feedback_reacts_after_plant_and_measurement_delay(self):
        """FF changes later WFS, which changes subsequent AO commands; feedback cannot react before that measurement exists."""
        first = int(np.flatnonzero(self.active.ff_enabled.to_numpy())[0])
        fields = ["residual_tip", "residual_tilt"]
        pd.testing.assert_frame_equal(
            self.active.loc[: first + 1, fields], self.base.loc[: first + 1, fields], check_exact=True
        )
        pd.testing.assert_frame_equal(
            self.active.loc[: first + 2, ["ao0", "ao1"]], self.base.loc[: first + 2, ["ao0", "ao1"]], check_exact=True
        )
        self.assertGreater(
            np.max(np.abs(self.active[["ao0", "ao1"]].to_numpy() - self.base[["ao0", "ao1"]].to_numpy())), 0.01
        )
        self.assertTrue(self.report["audits"]["wfs_harmonic"]["causal_feedback"])

    def test_source_adc_identity_and_frozen_training(self):
        """Exactly two ADC codes retain the simulator encoding; all calibration targets precede enabling control."""
        source = np.fromfile(self.root / "source.bin", dtype=np.float64).reshape(-1, 4)
        self.assertEqual(len(source), 12 * 8000)
        np.testing.assert_array_equal(source[:, 2:], np.floor(source[:, 2:]))
        self.assertTrue(np.all((source[:, 2:] >= 0) & (source[:, 2:] <= 4095)))
        self.assertLess(self.metadata["last_label_ns"], self.metadata["fit_end_ns"])
        ready = self.active.ready == 1
        np.testing.assert_array_equal(self.active.loc[ready, "sample_ns"], self.active.loc[ready, "query_ns"] - 125000)
        self.assertEqual(sha256(self.root / "source.bin"), self.metadata["source_sha256"])

    def test_future_forcing_cannot_change_earlier_commands(self):
        """A future disturbance/ADC mutation leaves the complete preceding closed-loop trajectory exactly unchanged."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.copy_fixture(root)
            source = np.fromfile(root / "source.bin", dtype=np.float64).reshape(-1, 4)
            source[8 * 8000 :, :2] += 1e-8
            source[8 * 8000 :, 2:] = 2050
            source.tofile(root / "source.bin")
            before = sha256(root / "calibration.txt")
            subprocess.run([str(self.executable), "run", str(root)], check=True)
            after = pd.read_csv(root / "wfs_harmonic.csv")
            prefix = self.active.query_ns < ORIGIN + 8_000_000_000
            pd.testing.assert_frame_equal(self.active.loc[prefix], after.loc[prefix], check_exact=True)
            self.assertEqual(before, sha256(root / "calibration.txt"))

    def test_disable_flushes_ff_and_restores_feedback_baseline(self):
        """FF off issues zeros and naturally converges to WFS-only without resetting AO state; zero gain has exact parity."""
        off = pd.read_csv(self.root / "ff_off.csv")
        selected = off.query_ns >= ORIGIN + 9_000_000_000
        np.testing.assert_array_equal(off.loc[selected, ["ff0", "ff1"]], np.zeros((selected.sum(), 2)))
        self.assertIsNotNone(self.report["ff_disable_recovery_s"])
        self.assertLess(self.report["ff_disable_recovery_s"], 1)
        self.assertLess(self.report["ff_disable_post1s_maximum_error_sd"], 1e-8)
        self.assertTrue(self.report["disabled_shadow_exact_parity"])

    def test_saturation_does_not_hide_linear_instability(self):
        """The declared high-gain stress case is linearly unstable even when hard bounds keep plant outputs finite."""
        self.assertLess(self.report["ao_pole_radius"]["0.1"], 1)
        self.assertLess(self.report["ao_pole_radius"]["0.3"], 1)
        self.assertGreater(self.report["ao_pole_radius"]["0.8"], 1)
        stress = self.report["audits"]["high_gain08"]
        self.assertGreater(stress["test_saturation_percent"], 0)
        self.assertGreater(stress["test_maximum_ao"], 3)
        self.assertLessEqual(stress["maximum_total"], 3 * (1 + 1e-12))

    def test_real_telemetry_calibration_cannot_be_loaded(self):
        """Reject the incompatible count-normalized real-data basis instead of silently applying it to simulator radians."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.copy_fixture(root)
            path = root / "calibration.txt"
            path.write_text(path.read_text().replace("accelFFCtrl-sim-phase-v1", "accelFFCtrl-count-phase-v1"))
            result = subprocess.run([str(self.executable), "run", str(root)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("incompatible simulation calibration", result.stderr)


if __name__ == "__main__":
    unittest.main()
