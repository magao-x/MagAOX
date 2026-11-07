r"""\file calibration_generalization_test.py
\brief Verify shortened native ADC histories, independent standard fits and validation-only configuration selection.
"""
import os
import unittest

import numpy as np

from calibration_generalization import candidates, identity, select
from prediction_information import states
from simulation_demo import APP, aosim_methods, bridge, simulate, source
from vibration_report import pair


class CalibrationGeneralizationTest(unittest.TestCase):
    """Use the real experimental libraries; no stream, alternative learner or application is created."""

    @classmethod
    def setUpClass(cls):
        """Open only the three existing-prior libraries differing in sensor history."""
        from pathlib import Path
        compiler = os.environ.get("GENERALIZATION_CXX", "c++")
        cls.libraries = {history: bridge(APP/"build"/f"generalization-endaligned-{compiler}-d4_h{history}_p001"/"libddspc_sim.so")
                         for history in (1, 3, 5)}
        cls.methods = aosim_methods(Path(os.environ.get("MAGAOX_REPO", Path.home()/"source/MagAOX")))
        cls.raw, cls.pointing, cls.units = source(cls.libraries[5], 1001, 100)

    def test_short_histories_match_native_command_and_reference_gain(self):
        """Each exported matrix has exactly the selected past sensor samples; u_k still uses six WFS records."""
        for history in (1, 3):
            model = {}
            # Capture after999 and check query1000 before its next update; this test does not pause training.
            data = simulate(self.libraries[history], self.methods, self.raw, self.pointing, self.units,
                            train=1000, mode=4, evaluation_probe=0, calibration_model=model)
            self.assertEqual(model["theta"].shape, (6, 28+2*history))
            self.assertTrue(data["valid"][-1])
            state = states(data, pair(data, "ax", "ay"), 4, np.array([1000]), accel_history=history)
            command = state @ model["gain"][-2:].T
            np.testing.assert_allclose(command, pair(data, "request_x", "request_y")[-1:], rtol=1e-5, atol=2e-6)
            c = model["theta"][:, -6:].astype(float)
            expected = np.linalg.solve(c.T @ c + .02*np.eye(6), -c.T @ model["theta"][:, :-6])
            np.testing.assert_allclose(model["gain"], expected, rtol=1e-5, atol=2e-6)

    def test_standard_is_bit_exact_across_sensor_history_builds(self):
        """Changing the independent assisted model's dimension must not change standard coefficients or commands."""
        reference = None
        for history in (5, 3, 1):
            model = {}
            data = simulate(self.libraries[history], self.methods, self.raw, self.pointing, self.units,
                            train=1000, mode=3, evaluation_probe=0, calibration_model=model)
            self.assertEqual(model["theta"].shape, (6, 28))
            if reference is None:
                reference = data, model
            else:
                np.testing.assert_array_equal(data, reference[0])
                np.testing.assert_array_equal(model["theta"], reference[1]["theta"])
                np.testing.assert_array_equal(model["gain"], reference[1]["gain"])

    def test_pooled_validation_selection_uses_both_sensor_levels_and_separate_baselines(self):
        """A reference-noise penalty can outweigh a high-SNR gain; baselines cannot become assisted variants."""
        validation = {scale: {identity(c): dict(prediction_h2_rms=5.) for c in candidates()} for scale in (1, 100)}
        best, noisy = "d4_h1_p001_assisted", "d4_h3_p001_assisted"
        for scale in validation:
            validation[scale][best]["prediction_h2_rms"] = 2.
            validation[scale]["d2_h5_p001_standard"]["prediction_h2_rms"] = 1.
            validation[scale]["d4_h5_p001_assisted"]["prediction_h2_rms"] = 3.
        validation[100][noisy]["prediction_h2_rms"] = .1
        validation[1][noisy]["prediction_h2_rms"] = 4.
        choice = select(validation)["choices"]
        self.assertEqual(choice, dict(standard="d2_h5_p001_standard", existing_assisted="d4_h5_p001_assisted",
                                      selected_assisted=best))


if __name__ == "__main__":
    unittest.main()
