r"""\file test_validate_gaps.py
\brief Independent rate-grid, masked phase and gap-window uncertainty checks.
\ingroup accelFFCtrl_unit_test
"""
import unittest

import numpy as np

from validate_gaps import fit_quality
from validate_predictions import BASE, OMEGA, metrics, truth_indices


class GapAnalysisTest(unittest.TestCase):
    """Verify the comparison retains actual time and uses the correct acquisition rate."""

    def test_four_khz_grid(self):
        """A 250-us-old sample and query+1ms target select five 4-kHz samples ahead."""
        self.assertEqual(truth_indices([BASE+1_250_000], 20, period=250_000)[0], 5)
        with self.assertRaises(ValueError):
            truth_indices([BASE+1_125_000], 20, period=250_000)

    def test_masked_phase(self):
        """Known amplitude/phase survive an irregular mask when actual target timestamps are used."""
        times = np.arange(20000)/2000
        mask = (np.arange(len(times)) % 1700) > 600
        t = times[mask]
        result = metrics(.003*np.cos(OMEGA*t+.2), .002*np.cos(OMEGA*t), t)
        self.assertAlmostEqual(result["amplitude_error_pct"], 50)
        self.assertAlmostEqual(result["phase_error_deg"], np.degrees(.2))

    def test_gap_uncertainty_oracle(self):
        """Exact tones retain amplitude and phase on both rates despite one or several omitted records."""
        for rate in (4000,8000):
            t = np.arange(rate)/rate
            for missing in (1,3,7):
                indices = np.delete(np.arange(rate), np.arange(rate-100,rate-100+missing))
                actual = t[indices]
                acceleration = np.column_stack((.002*np.cos(OMEGA*actual+.31)+.001,
                                                 .003*np.cos(OMEGA*actual-.74)-.001))
                target = actual[-1]+.001+1/rate
                quality = fit_quality(actual, acceleration, target, rate)
                expected = [.002*np.cos(OMEGA*target+.31), .003*np.cos(OMEGA*target-.74)]
                np.testing.assert_allclose(quality["prediction"],expected,atol=1e-14)
                self.assertGreater(quality["rcond"],.1)
                self.assertLess(quality["residual"],1e-10)
                self.assertLess(max(quality["weighted_rms"]),1e-14)
                self.assertTrue(np.all(np.isfinite(quality["phase_se"])))
                self.assertGreater(min(quality["amplitude_se"]),0)


if __name__ == "__main__":
    unittest.main()
