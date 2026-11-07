r"""\file test_validate_phase_calibration.py
\brief Earlier-only phase fitting and actual-C++ calibrated timing, inhibition, sign, clipping and legacy parity.
\ingroup accelFFCtrl_unit_test
"""

import copy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from qualify_telemetry import replay
from validate_phase_calibration import fit_calibration, shadow_checks
from validate_real_harmonic import HORIZON, normalized_manifest, phase_diagnostic_features, run_cpp


class PhaseFitTest(unittest.TestCase):
    """Check the small calibration solve and chronological selection independently of the estimator."""

    def test_training_only_gain_and_phase(self):
        """Recover independent band gains/phases and prove future labels cannot alter an earlier calibration."""
        time = np.arange(8000) / 2000
        x = np.column_stack(
            [
                np.cos(2 * np.pi * 28.5 * time),
                np.sin(2 * np.pi * 28.5 * time),
                np.cos(2 * np.pi * 56 * time),
                np.sin(2 * np.pi * 56 * time),
            ]
        )
        coefficient = np.array([[2, -3], [4, 1], [-2, 0.5], [3, -4]])
        truth = x @ coefficient
        fit = np.arange(len(x)) < 4000
        learned = fit_calibration(x, truth, fit)
        np.testing.assert_allclose(learned, coefficient, atol=1e-12)
        changed = truth.copy()
        changed[~fit] = 1e9
        np.testing.assert_array_equal(fit_calibration(x, changed, fit), learned)
        np.testing.assert_allclose(x[~fit] @ learned, truth[~fit], atol=1e-12)
        with self.assertRaises(ValueError):
            fit_calibration(np.zeros_like(x), truth, fit)
        with self.assertRaises(ValueError):
            fit_calibration(x, truth, np.zeros(len(x), dtype=bool))
        invalid = x.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_calibration(invalid, truth, fit)


@unittest.skipUnless(os.environ.get("ACCELFFCTRL_TELEMETRY_REPLAY"), "build telemetry-tools for the C++ phase path")
class PhaseReplayTest(unittest.TestCase):
    """Exercise calibration only in the file executable, retaining actual estimator and shadow integrity checks."""

    def setUp(self):
        """Prepare an owned raw-ADC fixture with later unused samples and exact acquisition/query stamps."""
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.executable = Path(os.environ["ACCELFFCTRL_TELEMETRY_REPLAY"])
        self.epoch = 1700000000000000001
        n = np.arange(5000)
        time = n / 2000
        acquired = self.epoch + n * 500000
        self.adc = pd.DataFrame(
            dict(
                acquired_ns=acquired,
                written_ns=acquired,
                receipt_ns=acquired,
                counter=n + 1,
                generation=1,
                adc_x=np.rint(670 + 20 * np.cos(2 * np.pi * 28.5 * time)).astype(np.uint16),
                adc_y=np.rint(680 + 15 * np.cos(2 * np.pi * 56 * time + 0.3)).astype(np.uint16),
            )
        )
        self.query = self.epoch + 100000 + n[:4000] * 500000
        queries = self.root / "queries.csv"
        pd.DataFrame(dict(query_ns=self.query)).to_csv(queries, index=False)
        self.manifest = normalized_manifest(queries, [28.5, 56])
        self.coefficient = np.array([[80, -100], [120, 50], [-400, 300], [250, -500]])

    def test_cpp_phase_shadow_and_causality(self):
        """The real C++ path agrees with exported quadratures, preserves raw fits, clips coherently and excludes future ADC."""
        raw = run_cpp(self.executable, self.manifest, self.adc, self.epoch, self.root)
        x = phase_diagnostic_features(raw, [28.5, 56])
        forecast = x @ self.coefficient
        self.manifest["phase_calibration"] = self.coefficient.tolist()
        sigma = np.array([0.02, 0.03])
        self.manifest["shadow"].update(modal_map=np.diag(1 / sigma).tolist(), bounds=[0.1, 0.1])
        disabled = run_cpp(self.executable, self.manifest, self.adc, self.epoch, self.root)
        shadow_checks(disabled, forecast, sigma, gain=0, bounds=(0.1, 0.1))
        self.manifest["shadow"]["gain"] = 1
        active = run_cpp(self.executable, self.manifest, self.adc, self.epoch, self.root)
        audit = shadow_checks(active, forecast, sigma, bounds=(0.1, 0.1))
        self.assertGreater(audit["clipped"], 0)
        self.assertEqual(audit["invalid_nonzero"], 0)
        columns = [c for c in raw.columns if c.startswith(("px", "py", "ax_", "ay_"))]
        pd.testing.assert_frame_equal(raw[columns], active[columns], check_exact=True)
        changed = self.adc.copy()
        changed.loc[changed.acquired_ns > self.query[-1], ["adc_x", "adc_y"]] = 100
        future = run_cpp(self.executable, self.manifest, changed, self.epoch, self.root)
        pd.testing.assert_frame_equal(active, future, check_exact=True)
        legacy = os.environ.get("ACCELFFCTRL_TELEMETRY_REPLAY_LEGACY")
        if legacy:
            old = run_cpp(
                Path(legacy),
                normalized_manifest(Path(self.manifest["query"]["file"]), [28.5, 56]),
                self.adc,
                self.epoch,
                self.root,
            )
            pd.testing.assert_frame_equal(raw, old, check_exact=True)

    def test_integrity_faults_and_stale_queries(self):
        """Warmup, rollback/restart/invalid samples and stale queries cannot turn into held calibrated commands."""
        self.manifest["phase_calibration"] = self.coefficient.tolist()
        self.manifest["shadow"]["gain"] = 1
        for fault in ("restart", "rollback", "invalid_adc"):
            adc = self.adc.copy()
            if fault == "restart":
                adc.loc[2500:, "generation"] = 2
            elif fault == "rollback":
                adc.loc[2500:, "counter"] -= 2500
            else:
                adc.loc[2500, "adc_x"] = 5000
            d = run_cpp(self.executable, self.manifest, adc, self.epoch, self.root)
            self.assertTrue(np.all(d.loc[d.valid == 0, ["mode0", "mode1"]].to_numpy() == 0))
            self.assertFalse(bool(d.valid.iloc[0]))
            self.assertFalse(bool(d.valid.iloc[2600]))
            self.assertTrue(bool(d.valid.iloc[-1]))
            self.assertGreater(int(d.resets.iloc[-1]), 1)
        queries = np.r_[self.query, int(self.adc.acquired_ns.iloc[-1]) + 3000000]
        pd.DataFrame(dict(query_ns=queries)).to_csv(self.manifest["query"]["file"], index=False)
        stale = run_cpp(self.executable, self.manifest, self.adc, self.epoch, self.root)
        self.assertEqual(int(stale.valid.iloc[-1]), 0)
        np.testing.assert_array_equal(stale.iloc[-1][["mode0", "mode1"]], [0, 0])

    def test_calibration_file_rejection(self):
        """Reject nonfinite coefficients, mismatched carrier/horizon/axis and falsely physical input calibration."""
        manifest = copy.deepcopy(self.manifest)
        manifest["phase_calibration"] = self.coefficient.tolist()
        work = self.root / "work"
        replay(self.executable, manifest, self.adc, self.epoch, work, HORIZON)
        calibration = work / "phase-calibration.txt"
        original = calibration.read_text()
        for bad in (
            original.replace("1000000", "2000000", 1),
            original.replace("28.5 0", "29.5 0", 1),
            original.replace("28.5 0", "28.5 1", 1),
            original + " extra",
        ):
            calibration.write_text(bad)
            result = subprocess.run(
                [
                    str(self.executable),
                    str(work / f"parameters_{HORIZON}.txt"),
                    str(work / "normalized_adc.csv"),
                    str(work / "queries.csv"),
                    str(work / "rejected.csv"),
                    str(calibration),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("phase calibration", result.stderr)
        manifest["phase_calibration"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            replay(self.executable, manifest, self.adc, self.epoch, work, HORIZON)
        manifest["phase_calibration"] = self.coefficient.tolist()
        manifest["adc"]["volts_per_bit"] = [0.0083799, 0.0083799]
        with self.assertRaises(subprocess.CalledProcessError):
            run_cpp(self.executable, manifest, self.adc, self.epoch, self.root)


if __name__ == "__main__":
    unittest.main()
