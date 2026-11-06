r"""\file test_validate_predictions.py
\brief Independent checks of the offline scoring conventions, masks and timestamp oracle.
\ingroup accelFFCtrl_unit_test
"""

import unittest

import numpy as np

from validate_predictions import BASE, PERIOD, OMEGA, availability, metrics, spectra, timing_us, truth_indices, weighted_fit


class PredictionAnalysisTest(unittest.TestCase):
    """Analytic references prevent a shifted truth grid or amplitude convention passing unnoticed."""

    def test_metrics_and_phase(self):
        """A known gain, phase lead and bias give exact amplitude, phase, correlation and RMSE."""
        t = np.arange(16000) / 2000
        truth = .002 * np.cos(OMEGA * t)
        predicted = .003 * np.cos(OMEGA * t + .2) + .0004
        result = metrics(predicted, truth, t)
        self.assertAlmostEqual(result["coherent_peak_truth"], .002)
        self.assertAlmostEqual(result["amplitude_error_pct"], 50)
        self.assertAlmostEqual(result["phase_error_deg"], np.degrees(.2))
        self.assertAlmostEqual(result["bias"], .0004)
        self.assertAlmostEqual(result["correlation"], np.cos(.2))
        expected = np.sqrt((.002**2 + .003**2 - 2 * .002 * .003 * np.cos(.2)) / 2 + .0004**2)
        self.assertAlmostEqual(result["rmse"], expected)
        self.assertAlmostEqual(result["nrmse"], expected / (.002 / np.sqrt(2)))
        transfer, _ = spectra(predicted, truth)
        self.assertAlmostEqual(transfer["spectral_frequency_hz"], 28)
        self.assertAlmostEqual(transfer["spectral_gain_28"], 1.5)
        self.assertAlmostEqual(transfer["spectral_phase_28_deg"], np.degrees(.2))

    def test_target_time_and_units(self):
        """Sample age belongs in the target index; half-grid stamps are never rounded to truth."""
        query = BASE + PERIOD
        target = query + 1_000_000
        self.assertEqual(truth_indices([target], 20)[0], 9)  # Age + horizon, not sample + horizon (index 8).
        data = np.zeros((1, 21))
        data[0, [1, 2, 3, 4]] = [query, BASE, target, 1]
        age, effective = timing_us(data)
        self.assertEqual(age[0], 125)
        self.assertEqual(effective[0], 1125)
        for invalid in (target + 1, target + .5, BASE - PERIOD, BASE + 20 * PERIOD):
            with self.assertRaises(ValueError):
                truth_indices([invalid], 20)
        self.assertAlmostEqual((2049 - 2048) * .0083799 / 1.02, .008215588235294117)
        acceleration = .002 * np.cos(OMEGA * np.arange(16000) / 2000)
        theta = -acceleration / (OMEGA**2 * 10)
        self.assertAlmostEqual(np.max(theta), .002 / (OMEGA**2 * 10))

    def test_availability_and_censored_recovery(self):
        """Missing scheduled slots reduce availability; repeated gaps censor incomplete warmup."""
        data = np.zeros((8, 21))
        data[:, 0] = data[:, 1] = BASE + np.arange(8) * 100_000_000
        data[:, 4] = [1, 1, 0, 0, 1, 0, 0, 0]
        data[3, 0] = -1  # An extra late query does not add a nominal deadline.
        values, recovered = availability(data, [BASE + 200_000_000, BASE + 500_000_000], settle=0)
        self.assertAlmostEqual(values["availability_pct"], 300 / 7)
        self.assertEqual(values["recovered_events"], 1)
        self.assertEqual(values["censored_events"], 1)
        self.assertEqual(values["recovery_mean_ms"], 200)
        np.testing.assert_array_equal(recovered, [True, False])

    def test_batch_window_oracle(self):
        """Batch weights agree with independent recursive moments, including one missing acquisition."""
        for skip in (False, True):
            times = np.arange(2000) / 8000
            if skip:
                times = np.delete(times, 1100)
            acceleration = np.column_stack((.002 * np.cos(OMEGA * times + .31) + .001,
                                             .003 * np.cos(OMEGA * times - .74) - .002))
            # Deterministic unrelated component exposes errors in the window, unlike an exactly represented tone.
            acceleration += .004 * np.cos(2 * np.pi * 120 * times)[:, None]
            gram, moments = np.zeros((2, 3, 3)), np.zeros((2, 3, 2))
            previous = -1 / 8000
            for t, a in zip(times, acceleration):
                beta = -np.expm1(-2 * (t-previous) / .1)
                u = np.array([1, np.cos(OMEGA*t), np.sin(OMEGA*t)])
                gram[0] = (1-beta)*gram[0] + beta*np.outer(u, u)
                moments[0] = (1-beta)*moments[0] + beta*np.outer(u, a)
                gram[1] = (1-beta)*gram[1] + beta*gram[0]
                moments[1] = (1-beta)*moments[1] + beta*moments[0]
                previous = t
            target = times[-1] + .001125
            coef = np.linalg.solve(gram[1], moments[1])
            expected = coef[1] * np.cos(OMEGA*target) + coef[2] * np.sin(OMEGA*target)
            predicted, condition, residual = weighted_fit(times, acceleration, target)
            np.testing.assert_allclose(predicted, expected, rtol=1e-12, atol=1e-16)
            self.assertGreater(condition, 1e-8)
            self.assertLess(residual, 1e-10)


if __name__ == "__main__":
    unittest.main()
