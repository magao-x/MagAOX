r"""\file test_validate_shadow.py
\brief Independent shadow timing, absolute semantics, clipping and pupil-calibration scoring tests.
\ingroup accelFFCtrl_unit_test
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from validate_shadow import audit_log, fit_pupil


class ShadowAnalysisTest(unittest.TestCase):
    """Audit numeric conventions without treating fixture registration as measured instrument calibration."""

    def row(self):
        """Use exact times/counters above float64's integer boundary to detect accidental rounding."""
        now = 1_700_000_000_000_000_007
        return dict(record=1, epoch=0, input_revision=0, calibration_id=1, evidence=0,
                    query_ns=now, sample_ns=now-125_000, prediction_target_ns=now+1_000_000,
                    requested_target_ns=now+1_000_000, source_counter=2**53+3, source_generation=1,
                    state=2, fault=0, prediction_valid=1, scale=1, theta_x_rad=1, theta_y_rad=-2,
                    dm_x_rad=1, dm_y_rad=-2, mode0=-.5, mode1=1)

    def audit(self, row):
        """Exercise the actual integer-preserving CSV parser with a fully explicit synthetic identity mapping."""
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"shadow.csv"
            pd.DataFrame([row]).to_csv(path, index=False)
            result = audit_log(path, np.eye(2), .5, [2, 2], 1_000_000)
            data = pd.read_csv(path, dtype={"source_counter": "uint64", "query_ns": "int64"})
            self.assertEqual(int(data.source_counter.iloc[0]), 2**53+3)
            self.assertEqual(int(data.query_ns.iloc[0]), row["query_ns"])
            return result

    def test_absolute_and_time(self):
        """A complete opposing command has the expected gain and unchanged exact target time."""
        self.assertEqual(self.audit(self.row())["active"], 1)

    def test_invalid_and_stale(self):
        """Held nonzero invalid output, stale readiness and shifted targets are audit failures."""
        for key, value in (("state", 1), ("sample_ns", self.row()["query_ns"]-2_000_001),
                           ("prediction_target_ns", self.row()["prediction_target_ns"]+1)):
            row = self.row(); row[key] = value
            with self.assertRaises(AssertionError):
                self.audit(row)
        row = self.row(); row.update(state=1, prediction_valid=0, mode0=0, mode1=0)
        self.assertEqual(self.audit(row)["inhibited"], 1)

    def test_clipping(self):
        """One common scalar bounds both modes without changing proposed direction."""
        row = self.row()
        row.update(state=3, scale=.5, mode0=-.25, mode1=.5)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"shadow.csv"
            pd.DataFrame([row]).to_csv(path, index=False)
            self.assertEqual(audit_log(path, np.eye(2), .5, [.25, 2], 1_000_000)["limited"], 1)

    def test_pupil_scale_order_and_mask(self):
        """Independent synthetic plane fits retain reflection, modal sign/order and zero-weight exclusions."""
        xy = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=float)
        modes = xy[:, ::-1].copy(); modes[:, 0] *= -2
        weights = np.ones(4)
        mapping, residual = fit_pupil(modes, weights, xy)
        np.testing.assert_allclose(mapping, [[0, -250000], [500000, 0]], atol=1e-9)
        self.assertLess(max(residual), 1e-14)
        weights[0] = -1
        with self.assertRaises(ValueError):
            fit_pupil(modes, weights, xy)
        with self.assertRaises(ValueError):
            fit_pupil(np.zeros_like(modes), np.ones(4), xy)


if __name__ == "__main__":
    unittest.main()
