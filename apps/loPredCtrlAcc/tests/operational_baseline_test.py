r"""\file operational_baseline_test.py
\brief Verify fifteen-frame native histories, operational law/export, and timing/stability algebra offline.
"""
import os
from pathlib import Path
import unittest

import numpy as np

from operational_baseline import parameters, pole_radius
from prediction_information import states, forecast
from simulation_demo import APP, aosim_methods, bridge, simulate, source
from vibration_report import pair


class OperationalBaselineTest(unittest.TestCase):
    """Use only the isolated operational-profile numerical bridge; no native app or DM stream."""

    @classmethod
    def setUpClass(cls):
        """Open the explicit test profile compiled from the provided values."""
        compiler = os.environ.get("OPERATIONAL_CXX", "c++")
        cls.lib = bridge(APP/"build"/f"operational-{compiler}/libddspc_sim.so")
        cls.methods = aosim_methods(Path(os.environ.get("MAGAOX_REPO", Path.home()/"source/MagAOX")))

    def test_fixture_has_exact_provided_operational_parameters(self):
        """Configuration values are interpreted explicitly instead of reusing the short-history numeric preset."""
        p = parameters()
        self.assertEqual((p["history"], p["future"], p["num_modes"]), (15, 3, 2))
        self.assertEqual((p["gain"], p["copy_gain"], p["regularization"], p["gamma"], p["covariance"]), (.3, 1., 100., 1., 1e-5))
        self.assertTrue(p["qrd"]); self.assertTrue(p["own_shmim"])

    def test_native_fifteen_step_gain_and_prediction_use_actual_causal_state(self):
        """Actual native proposals match the exported scaled law plus feedback on fifteen WFS/fourteen command records."""
        raw, pointing, units = source(self.lib, 1201, 100, seed=101)
        for mode, features in ((3, 64), (4, 74)):
            model = {}
            data = simulate(self.lib, self.methods, raw, pointing, units, train=1200, mode=mode,
                            evaluation_probe=0, calibration_model=model)
            self.assertEqual(int(model["history"]), 15)
            self.assertEqual(model["theta"].shape, (6, features))
            self.assertEqual(model["version"], 1200-17)
            self.assertTrue(data["valid"][-1])
            x = states(data, pair(data, "ax", "ay"), mode, np.array([1200]), wfs_history=15)
            np.testing.assert_allclose(x @ model["gain"][-2:].T, pair(data, "request_x", "request_y")[-1:], rtol=1e-5, atol=2e-6)
            theta = model["theta"].astype(float); c = theta[:, -6:]
            scaled = np.linalg.solve(c.T@c+100*np.max(c.T@c)*np.eye(6), -c.T@theta[:, :-6])
            scaled[-2:, :2] += .3*np.eye(2)
            np.testing.assert_allclose(model["gain"], scaled, rtol=2e-5, atol=3e-6)
            predicted, _, state = forecast(data, pair(data, "ax", "ay"), model, mode, np.array([1100]))
            self.assertEqual(state.shape[1], features-6)
            np.testing.assert_allclose(predicted, state @ model["theta"][:, :-6].T, rtol=0, atol=0)

    def test_pole_radius_includes_the_verified_two_frame_delay(self):
        """Pure positive .3 feedback on an absolute two-frame plant has radius sqrt(.3), without invented sign reversal."""
        gain = np.zeros((6, 58))
        gain[-2:, :2] = .3*np.eye(2)
        self.assertAlmostEqual(pole_radius(dict(history=15, gain=gain)), np.sqrt(.3), places=12)

    def test_unready_diagnostic_copy_does_not_authorize_corrections(self):
        """Explicitly scoring an untrained fit leaves the native readiness and command gates unchanged."""
        raw, pointing, units = source(self.lib, 35, 100, seed=307)
        with self.assertRaisesRegex(RuntimeError, "No ready calibration snapshot"):
            simulate(self.lib, self.methods, raw, pointing, units, train=30, mode=3, calibration_model={})
        model = {}
        data = simulate(self.lib, self.methods, raw, pointing, units, train=30, mode=3,
                        calibration_model=model, allow_unready_diagnostic=True)
        self.assertFalse(model["calibration_ready"])
        self.assertFalse(data["valid"].any())
        np.testing.assert_array_equal(pair(data, "request_x", "request_y")[30:],
                                      np.repeat(pair(data, "applied_x", "applied_y")[29:30], 5, axis=0))


if __name__ == "__main__":
    unittest.main()
