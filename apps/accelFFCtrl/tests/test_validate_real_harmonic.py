r"""\file test_validate_real_harmonic.py
\brief Numerical mapping, exact target labels and existing-C++ normalized shadow sign/unit/causality checks.
\ingroup accelFFCtrl_unit_test
"""

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from validate_real_harmonic import (
    fit_map,
    normalized_manifest,
    phase_diagnostic_features,
    run_cpp,
    target_labels,
    HORIZON,
)


class RealHarmonicScoringTest(unittest.TestCase):
    """Check the experiment's calibration arithmetic without inventing a physical sensor/DM scale."""

    def test_origin_preserving_mappings(self):
        """A real map preserves correction sign/order and zero input; constrained alternatives remain explicit."""
        x = np.random.default_rng(14).normal(size=(500, 2))
        for kind, m in (
            ("direct", np.diag([-2.0, 3.0])),
            ("swapped", np.array([[0.0, -2.0], [3.0, 0.0]])),
            ("full", np.array([[-2.0, 0.5], [1.0, 3.0]])),
        ):
            y = x @ m.T
            result = fit_map(x, y, kind)
            np.testing.assert_allclose(result, m, rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(np.zeros((1, 2)) @ result.T, np.zeros((1, 2)))
        with self.assertRaises(ValueError):
            fit_map(x, x, "implicit")
        with self.assertRaises(ValueError):
            fit_map(np.zeros_like(x), x, "direct")

    def test_future_labels_and_partition_edges(self):
        """Only labels interpolate WFS; integer brackets prevent a training target from using a held-out endpoint."""
        epoch = 1700000000000000001
        wt = epoch + np.arange(10, dtype=np.int64) * 500000
        wfs = np.column_stack((np.arange(10), -np.arange(10))).astype(float)
        target = epoch + np.array([750000, 1250000], dtype=np.int64)
        y, right = target_labels(wfs, wt, target, epoch)
        np.testing.assert_allclose(y, [[1.5, -1.5], [2.5, -2.5]])
        np.testing.assert_array_equal(right, epoch + np.array([1000000, 1500000]))
        keep = right < epoch + 1500000
        self.assertEqual(keep.tolist(), [True, False])
        changed = wfs.copy()
        changed[wt >= epoch + 1500000] = 1e6
        y2, _ = target_labels(changed, wt, target, epoch)
        np.testing.assert_array_equal(y[keep], y2[keep])
        shifted, _ = target_labels(wfs, wt, target, epoch, center_offset_ns=250000)
        np.testing.assert_allclose(shifted, [[1, -1], [2, -2]])

    def test_quadrature_diagnostic_does_not_hide_static_map_limit(self):
        """Real coincident axes cannot supply a quarter-cycle response; existing quadratures can diagnose missing phase calibration."""
        t = np.arange(8000) / 2000
        x = np.column_stack([np.cos(2 * np.pi * 28.5 * t), 2 * np.cos(2 * np.pi * 28.5 * t)])
        y = np.column_stack([np.sin(2 * np.pi * 28.5 * t), -np.sin(2 * np.pi * 28.5 * t)])
        p = x @ fit_map(x, y, "full").T
        self.assertGreater(np.sqrt(np.mean((p - y) ** 2)), 0.70)
        d = pd.DataFrame(
            dict(ax_re0=x[:, 0], ax_im0=y[:, 0], ay_re1=np.cos(2 * np.pi * 56 * t), ay_im1=np.sin(2 * np.pi * 56 * t))
        )
        q = phase_diagnostic_features(d, [28.5, 56])
        coefficient = np.linalg.lstsq(q[:4000], y[:4000], rcond=None)[0]
        np.testing.assert_allclose(q[4000:] @ coefficient, y[4000:], atol=1e-12)


@unittest.skipUnless(os.environ.get("ACCELFFCTRL_TELEMETRY_REPLAY"), "build the existing telemetry-tools replay")
class ExistingHarmonicReplayTest(unittest.TestCase):
    """Use the unchanged C++ estimator and shadow implementation, with explicit count-unit fixtures."""

    def test_units_shadow_and_future_exclusion(self):
        """Normalization is a linear rescaling; C++ sign/clipping/target time and no-future ingestion hold."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            epoch = 1700000000000000001
            n = np.arange(2000)
            t = n / 2000
            a = np.rint(660 + 20 * np.cos(2 * np.pi * 28.5 * t)).astype(np.uint16)
            b = np.rint(670 + 15 * np.sin(2 * np.pi * 56 * t)).astype(np.uint16)
            acquired = epoch + n * 500000
            adc = pd.DataFrame(
                dict(
                    acquired_ns=acquired,
                    written_ns=acquired,
                    receipt_ns=acquired,
                    counter=n + 1,
                    generation=1,
                    adc_x=a,
                    adc_y=b,
                )
            )
            queries = root / "queries.csv"
            q = epoch + 100000 + n[:1200] * 500000
            pd.DataFrame(dict(query_ns=q)).to_csv(queries, index=False)
            manifest = normalized_manifest(queries, [28.5, 56])
            executable = Path(os.environ["ACCELFFCTRL_TELEMETRY_REPLAY"])
            zero = run_cpp(executable, manifest, adc, epoch, root)
            self.assertTrue(np.all(zero[["mode0", "mode1"]] == 0))
            changed = adc.copy()
            changed.loc[acquired > q[-1], ["adc_x", "adc_y"]] = 100
            future = run_cpp(executable, manifest, changed, epoch, root)
            np.testing.assert_array_equal(zero[["px0", "py0", "px1", "py1"]], future[["px0", "py0", "px1", "py1"]])
            mapping = np.array([[4.0, -1.0], [0.5, 2.0]])
            manifest["shadow"].update(modal_map=mapping.tolist(), gain=0.5, bounds=[1e-4, 1e-4])
            active = run_cpp(executable, manifest, adc, epoch, root)
            valid = active.valid.astype(bool)
            pointing = active[["theta_x", "theta_y"]].to_numpy()
            intended = -0.5 * pointing @ mapping.T
            scale = np.minimum(
                1, np.min(np.divide(1e-4, np.abs(intended), out=np.ones_like(intended), where=intended != 0), axis=1)
            )
            expected = intended * scale[:, None]
            expected[~valid] = 0
            np.testing.assert_allclose(active[["mode0", "mode1"]], expected, rtol=1e-11, atol=1e-14)
            self.assertTrue(np.any(active.state == 3))
            np.testing.assert_array_equal(active.target_ns - active.query_ns, np.full(len(active), HORIZON))
            # Same kernel with its numerical simulator defaults differs by the declared scalar, not waveform/phase.
            manifest["adc"].update(zero=[2048, 2048], volts_per_bit=[0.0083799] * 2, sensitivity=[1.02] * 2)
            manifest["estimator"]["lever_m"] = 10
            native = run_cpp(executable, manifest, adc, epoch, root)
            factor = 0.0083799 / 1.02 / 10
            np.testing.assert_allclose(
                native.loc[valid, ["px0", "py0", "px1", "py1"]] / factor,
                zero.loc[valid, ["px0", "py0", "px1", "py1"]],
                rtol=1e-7,
                atol=1e-10,
            )


if __name__ == "__main__":
    unittest.main()
