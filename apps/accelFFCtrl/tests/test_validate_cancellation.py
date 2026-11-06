r"""\file test_validate_cancellation.py
\brief Independent sinusoidal sign, gain, phasor and PSD-suppression scoring tests.
\ingroup accelFFCtrl_unit_test
"""
import unittest

import numpy as np

from validate_cancellation import coherent, present, spectral_suppression


class CancellationScoringTest(unittest.TestCase):
    """Compare the scoring conventions with exact sinusoidal cancellation."""

    def test_gain_and_wrong_sign(self):
        """Opposing gain leaves (1-g) amplitude; reversed correction leaves (1+g)."""
        t = np.arange(64000)/8000
        for frequency in (28, 56):
            truth = .003*np.cos(2*np.pi*frequency*t+.31)
            self.assertAlmostEqual(abs(coherent(truth, t, frequency)), .003)
            self.assertAlmostEqual(np.angle(coherent(truth, t, frequency)), .31)
            for gain in (0, .25, .5, 1):
                for sign in (-1, 1):
                    residual = (1+sign*gain)*truth
                    expected = abs(1+sign*gain)
                    self.assertAlmostEqual(abs(coherent(residual, t, frequency)/coherent(truth, t, frequency)), expected)
                    score, _ = spectral_suppression(truth, residual, frequency)
                    if expected:
                        self.assertAlmostEqual(score["band_suppression_db"], -20*np.log10(expected))
                    else:
                        self.assertTrue(np.isinf(score["band_suppression_db"]))

    def test_phase_mismatch(self):
        """Latency phase error follows the complex residual 1-g*exp(i*omega*dt)."""
        t = np.arange(64000)/8000
        for frequency in (28, 56):
            phase = 2*np.pi*frequency*t+.31
            error = 2*np.pi*frequency*.002
            truth = np.cos(phase)
            residual = truth-.8*np.cos(phase+error)
            actual = coherent(residual, t, frequency)/coherent(truth, t, frequency)
            self.assertAlmostEqual(abs(actual-(1-.8*np.exp(1j*error))), 0)

    def test_mode_labels(self):
        """Absent carrier bins must not be reported as suppression of an injected mode."""
        self.assertTrue(present("ar_0", 28))
        self.assertFalse(present("ar_0", 56))
        self.assertTrue(present("analytic_1", 56))
        self.assertFalse(present("analytic_1", 28))
        self.assertTrue(present("ar_2_noise1", 28))
        self.assertTrue(present("ar_2_noise1", 56))


if __name__ == "__main__":
    unittest.main()
