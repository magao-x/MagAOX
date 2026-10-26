r"""\file test_validate_psd.py
\brief Offline regression checks for AR covariance, Welch overlays, and RMS units.
\ingroup accelSim_unit_test
"""

import unittest

import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import get_window

import validate_psd as validation


class SpectrumTests(unittest.TestCase):
    """Check the analysis mathematics independently of the C++ acquisition runtime."""

    def test_white_noise(self):
        """One-sided density and the Hann mean-removal corrections have known white-noise limits."""
        variance, fps, count = 2.0, 128.0, 64
        covariance = np.zeros((count, 2))
        covariance[0] = variance
        power = validation.expected_welch(covariance, fps)
        np.testing.assert_allclose(power[2:-1], 2 * variance / fps, rtol=1e-12)
        np.testing.assert_allclose(power[0], variance / (3 * fps), rtol=1e-12)
        np.testing.assert_allclose(power[1], 2 * variance / fps * 5/6, rtol=1e-12)
        np.testing.assert_allclose(power[-1], variance / fps, rtol=1e-12)

    def test_dense_covariance_oracle(self):
        """FFT theory matches direct mean-projected Toeplitz quadratic forms, for odd and even segments."""
        for count in (63, 64):
            for r in (0.9, 0.9999999):
                fps, omega = 6000.0, 2*np.pi*11/6000
                lag = np.arange(count)
                correlation = r**lag * (np.cos(lag*omega) +
                                       (1-r*r)/(1+r*r)/np.tan(omega)*np.sin(lag*omega))
                window = get_window("hann", count)
                matrix = toeplitz(correlation)
                expected = []
                for k in range(count//2 + 1):
                    weights = window * np.exp(-2j*np.pi*k*lag/count)
                    weights -= np.mean(weights)
                    value = np.einsum("i,ij,j->", weights.conj(), matrix, weights).real / (fps * np.sum(window**2))
                    if k > 0 and not (count % 2 == 0 and k == count//2):
                        value *= 2
                    expected.append(value)
                actual = validation.expected_welch(correlation[:, None], fps)[:, 0]
                np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-15)

    def test_physical_units_and_coupling(self):
        """Mode RMS, lag-one covariance, mas conversion, per-axis coupling, and sensor noise remain distinct."""
        modes = np.array([(28, .002, .99999, .05, .25)],
                         dtype=[(key, float) for key in ("frequency", "accel_rms", "r", "coupling_x", "coupling_y")])
        covariance = validation.model_covariances(modes, 6000, 10, 1e-8, 64)
        pointing_variance = (.002 / ((2*np.pi*28)**2 * 10 * validation.ARCSEC_TO_RAD) * 1000)**2
        np.testing.assert_allclose(covariance["pointing"][0], pointing_variance)
        np.testing.assert_allclose(covariance["wfs"][0], pointing_variance * np.array([.05, .25])**2)
        np.testing.assert_allclose(covariance["accel"][0], .002**2 + 1e-8*6000/2)
        rho = 2*.99999*np.cos(2*np.pi*28/6000)/(1+.99999**2)
        np.testing.assert_allclose(covariance["pointing"][1], pointing_variance*rho)
        np.testing.assert_allclose(covariance["accel"][1], .002**2*rho)

    def test_rms_retains_mean(self):
        """Reported RMS is sqrt(mean(x²)), not the standard deviation of a mean-subtracted record."""
        np.testing.assert_allclose(validation.rms(np.array([[3., 4.], [3., 4.]])), [3, 4])


if __name__ == "__main__":
    unittest.main()
