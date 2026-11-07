r"""\file prediction_information_test.py
\brief Verify diagnostic snapshot, forecast-origin and measured cross-phase contracts without changing any controller.
"""
import ctypes as ct
import os
from pathlib import Path
import unittest

import numpy as np

from prediction_information import forecast, peak_coherence, states
from simulation_demo import APP, aosim_methods, bridge, simulate, source
from vibration_report import pair


class PredictionInformationTest(unittest.TestCase):
    """Use actual frozen-fit libraries for export checks and independent analytic controls for scoring."""

    @classmethod
    def setUpClass(cls):
        """Open only the numerical test library, never an application or stream."""
        cls.lib = bridge(Path(os.environ.get("DDSPC_SIM_FIXED_LIB", APP/"build/simulation-c++-train2400/libddspc_sim.so")))
        cls.methods = aosim_methods(Path(os.environ.get("MAGAOX_REPO", Path.home()/"source/MagAOX")))

    def test_unready_and_wrong_size_export_do_not_copy_coefficients(self):
        """An absent fit or incompatible caller buffer must not be mistaken for a ready coherent snapshot."""
        lib = self.lib
        handle = lib.ddspc_sim_controller_create(1, np.array([.001, .002, 2048, .0083799, 1.02]))
        self.assertTrue(handle)
        try:
            theta, gain, version = np.full(6*38, 123, np.float32), np.full(6*32, 456, np.float32), ct.c_uint64(99)
            for size in (theta.size, theta.size-1):
                self.assertEqual(lib.ddspc_sim_copy_model(handle, theta, size, gain, gain.size, ct.byref(version)), -1)
                self.assertTrue((theta == 123).all())
                self.assertTrue((gain == 456).all())
                self.assertEqual(version.value, 99)
        finally:
            lib.ddspc_sim_controller_destroy(handle)

    def test_exported_fixed_gain_reproduces_actual_native_commands(self):
        """Copied canonical matrices retain actual native reference-law, axis and newest-first conventions."""
        raw, pointing, units = source(self.lib, 2800, 100)
        for mode in (3, 4):
            model = {}
            data = simulate(self.lib, self.methods, raw, pointing, units, mode=mode, evaluation_probe=0,
                            calibration_model=model)
            origins = np.flatnonzero((data["frame"] >= 2400) & (data["valid"] != 0))
            self.assertGreater(len(origins), 300)
            np.testing.assert_array_equal(data["version"][origins], model["version"])
            acceleration = pair(data, "ax", "ay")
            predicted_command = states(data, acceleration, mode, origins) @ model["gain"][-2:].T
            np.testing.assert_allclose(predicted_command, pair(data, "request_x", "request_y")[origins],
                                       rtol=1e-5, atol=2e-6)
            c, known = model["theta"][:, -6:].astype(float), model["theta"][:, :-6].astype(float)
            expected = np.linalg.solve(c.T @ c + .02*np.eye(6), -c.T @ known)
            np.testing.assert_allclose(model["gain"], expected, rtol=1e-5, atol=2e-6)

    def test_forecast_target_removes_only_actual_future_command_effects(self):
        """A two-frame analytic plant leaves u[k-1] at horizon one; unknown u[k:] never enters prediction state."""
        n = 40
        data = np.zeros(n, dtype=[(name, float) for name in ("yx", "yy", "applied_x", "applied_y")])
        commands = np.column_stack((np.arange(n), -2*np.arange(n))).astype(float)
        disturbance = np.tile([3., -5.], (n, 1))
        for axis, key in enumerate(("applied_x", "applied_y")):
            data[key] = commands[:, axis]
        for axis, key in enumerate(("yx", "yy")):
            data[key] = disturbance[:, axis]
            data[key][2:] += commands[:-2, axis]
        theta = np.zeros((6, 28), np.float32)
        for start in (0, 2, 4):
            theta[start:start+2, :2] = np.eye(2)
            theta[start:start+2, 14:16] = -np.eye(2)  # Remove the already delayed u[k-2] in y[k].
        theta[4:6, 12:14] = np.eye(2)  # Known u[k-1] still affects y[k+1].
        origins = np.arange(6, n-3)
        prediction, target, original_state = forecast(data, np.zeros((n, 2)), dict(theta=theta), 3, origins)
        np.testing.assert_array_equal(prediction, target)
        np.testing.assert_array_equal(target[:, 2:4], disturbance[origins+2])
        changed = data.copy()
        # At one origin, a later command/WFS can change scoring targets but never the prediction inputs.
        changed["applied_x"][20:] += 100
        np.testing.assert_array_equal(states(changed, np.zeros((n, 2)), 3, np.array([20])), original_state[origins == 20])

    def test_adc_to_wfs_cross_phase_sign_with_quantized_two_line_control(self):
        """Independent encoded sinusoids verify coherence and WFS-minus-ADC phase, including the negative sign."""
        t = np.arange(16384)/2000
        adc = np.empty((len(t), 2))
        wfs = np.empty_like(adc)
        for axis in range(2):
            phase = .3*axis
            adc[:, axis] = np.round(2048-100*(np.sin(2*np.pi*28*t+phase)+.4*np.sin(2*np.pi*56*t+phase)))-2048
            wfs[:, axis] = np.sin(2*np.pi*28*(t-.001)+phase)+.1*np.sin(2*np.pi*56*(t-.001)+phase)
        rows = peak_coherence(adc, wfs)
        for row in rows:
            if row["adc_channel"] != row["wfs_axis"]:
                continue
            self.assertGreater(row["coherence"], .999)
            self.assertAlmostEqual(row["wfs_minus_adc_phase_deg"], 180-360*row["line_hz"]*.001, delta=.2)


if __name__ == "__main__":
    unittest.main()
