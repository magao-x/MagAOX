r"""\file independent_realizations_test.py
\brief Verify actual source-seed independence, backward-compatible pairing and per-trial statistical/spectral metrics.
"""
import math
import os
from pathlib import Path
import unittest

import numpy as np

from independent_realizations import ROLES, SEEDS, broadband_rms, comparisons, initial_phases, summarize
from simulation_demo import APP, aosim_methods, bridge, simulate, source
from vibration_report import pair, spectra


class IndependentRealizationsTest(unittest.TestCase):
    """All native calls use the existing headless numerical library, never a live application or stream."""

    @classmethod
    def setUpClass(cls):
        """Use the actual four-second QR bridge selected only by compiler, without changing its settings."""
        compiler = os.environ.get("GENERALIZATION_CXX", "c++")
        cls.lib = bridge(APP/"build"/f"generalization-endaligned-{compiler}-d4_h5_p001/libddspc_sim.so")
        cls.methods = aosim_methods(Path(os.environ.get("MAGAOX_REPO", Path.home()/"source/MagAOX")))

    def test_actual_seed_replays_and_changes_both_waveform_and_initial_phases(self):
        """A seed changes actual noisy ADC/pointing together, with fixed physical normalization and repeatable phases."""
        a, p, units = source(self.lib, 600, 100, seed=101)
        b, q, other_units = source(self.lib, 600, 100, seed=101)
        c, r, changed_units = source(self.lib, 600, 100, seed=211)
        np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(p, q)
        np.testing.assert_array_equal(units, other_units); np.testing.assert_array_equal(units, changed_units)
        self.assertFalse(np.array_equal(a, c))
        self.assertFalse(np.array_equal(p, r))
        self.assertNotEqual(initial_phases(p), initial_phases(r))
        np.testing.assert_array_equal(source(self.lib, 40, 1)[0], source(self.lib, 40, 1, seed=42)[0])

    def test_seed_defaults_preserve_complete_traces_and_noise_never_changes_disturbance(self):
        """Explicit legacy seeds are bit-identical; changing probe/noise affects measurements but not the source channel."""
        raw, pointing, units = source(self.lib, 600, 100)
        implicit = simulate(self.lib, self.methods, raw, pointing, units, train=400, mode=3)
        explicit = simulate(self.lib, self.methods, raw, pointing, units, train=400, mode=3,
                            calibration_seed=100, wfs_seed=303)
        changed = simulate(self.lib, self.methods, raw, pointing, units, train=400, mode=3,
                           calibration_seed=111, wfs_seed=222)
        np.testing.assert_array_equal(implicit, explicit)
        self.assertFalse(np.array_equal(implicit["yx"][:400], changed["yx"][:400]))
        np.testing.assert_array_equal(pair(implicit, "disturbance_x", "disturbance_y"),
                                      pair(changed, "disturbance_x", "disturbance_y"))

    def test_broadband_metric_integrates_noise_band_and_excludes_the_two_lines(self):
        """Independent analytic250Hz power must survive while28/56Hz line power is excluded from the metric."""
        t = np.arange(16384)/2000
        values = .1*np.sin(2*np.pi*28*t)+.1*np.sin(2*np.pi*56*t)+.2*np.sin(2*np.pi*250*t)
        f, power = spectra(np.column_stack((values, values)))
        self.assertAlmostEqual(broadband_rms(f, power), .2/math.sqrt(2), delta=.001)

    def test_bootstrap_uses_complete_trial_effects_and_reports_sample_spread(self):
        """Known signed effects verify mean/median/sample-SD, repeatable uncertainty and finite one-dimensional input."""
        values = [10, 20, -10, 0, 30, -20, 10, 0]
        result = summarize(values)
        self.assertEqual(result["n"], 8)
        self.assertEqual(result["mean"], 5)
        self.assertEqual(result["median"], 5)
        self.assertAlmostEqual(result["sample_std"], math.sqrt(1800/7))
        self.assertEqual(result["wins"], 4); self.assertEqual(result["ties"], 2)
        self.assertEqual(result, summarize(values))
        self.assertLess(result["mean_ci95_percentile"][0], 5)
        self.assertGreater(result["mean_ci95_percentile"][1], 5)
        self.assertEqual(summarize([10]*8)["mean_ci95_percentile"], [10, 10])
        for invalid in ([1], [1, np.nan], np.ones((8, 2))):
            with self.assertRaises(ValueError):
                summarize(invalid)

    def test_comparisons_pair_within_trial_and_do_not_inflate_trial_count(self):
        """Scaled baselines yield the same percentage; peak losses remain losses and missing trials are not discarded."""
        rows = []
        for scale in (1, 100):
            for index, seed in enumerate(SEEDS):
                for role in ROLES:
                    multiplier = dict(no_control=2., standard=1., assisted_h5=.9, assisted_h1=1.1)[role]
                    value = (index+1)*multiplier*scale
                    rows.append(dict(scale=scale, seed=seed, controller=role,
                                     **{metric: value for metric in ("residual_rms", "tip_rms", "tilt_rms", "prediction_h2_rms", "broadband_rms")},
                                     suppression_28_db=3. if role=="assisted_h5" else 0.,
                                     suppression_56_db=-2. if role=="assisted_h5" else 0.))
        paired, aggregate = comparisons(rows)
        self.assertEqual(len(paired), 32)
        for scale in (1, 100):
            self.assertEqual(aggregate[scale]["assisted_h5"]["residual_rms_improvement_percent"]["n"], 8)
            self.assertAlmostEqual(aggregate[scale]["assisted_h5"]["residual_rms_improvement_percent"]["mean"], 10)
            self.assertAlmostEqual(aggregate[scale]["assisted_h1"]["residual_rms_improvement_percent"]["mean"], -10)
            self.assertEqual(aggregate[scale]["assisted_h5"]["line_56_advantage_db"]["mean"], -2.)
        with self.assertRaises(KeyError):
            comparisons(rows[:-1])


if __name__ == "__main__":
    unittest.main()
