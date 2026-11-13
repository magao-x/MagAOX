r"""\file handoff_config_test.py
\brief Check that manual examples remain output-inhibited with explicit input calibration and source boundaries.
"""
import configparser
from pathlib import Path
import unittest


class HandoffConfigTest(unittest.TestCase):
    """Validate distributed manual fixtures; no application, stream, network or device is opened."""

    def test_example_is_explicitly_non_actuating_and_uses_the_supplied_profile(self):
        """Retain the supplied numeric values while refusing to imply verified input names or enable the FLOAT reader."""
        path = Path(__file__).resolve().parents[1]/"examples/loPredCtrlAcc-shadow.conf.example"
        config = configparser.ConfigParser()
        config.read(path)
        p = config["parameters"]
        self.assertTrue(p.getboolean("shadow"))
        self.assertFalse(p.getboolean("is_integrating"))
        self.assertFalse(p.getboolean("own_shmim"))
        self.assertFalse(p.getboolean("accel_enabled"))
        self.assertFalse(p.getboolean("legacy_copy_gain_compat"))
        self.assertTrue(p.getboolean("qrd"))
        self.assertEqual([p.getfloat(k) for k in ("gain", "copy_gain", "regularization", "gamma", "covariance")],
                         [.3, 1., 100., 1., 1e-5])
        self.assertEqual([p.getint(k) for k in ("history", "future", "num_modes", "accel_channels")], [15, 3, 2, 2])
        self.assertEqual(config["shmimMonitor"]["shmimName"], "REPLACE_WITH_VERIFIED_MODAL_WFS_STREAM")
        self.assertEqual(set(config.sections()), {"parameters", "shmimMonitor"})

    def test_adc_example_keeps_output_inhibited_and_uses_explicit_simulator_calibration(self):
        """The new opt-in reader must retain shadow gates and a separate explicit unsigned ADC profile."""
        path = Path(__file__).resolve().parents[1]/"examples/loPredCtrlAcc-adc-shadow.conf.example"
        config = configparser.ConfigParser()
        config.read(path)
        p = config["parameters"]
        self.assertTrue(p.getboolean("shadow"))
        self.assertFalse(p.getboolean("shadow_pipeline"))
        for key in ("is_integrating", "own_shmim", "accel_enabled"):
            self.assertFalse(p.getboolean(key))
        self.assertEqual(config["shmimMonitor"]["shmimName"], "")
        self.assertEqual(set(config.sections()), {"parameters", "shmimMonitor", "adc"})
        adc = config["adc"]
        self.assertTrue(adc.getboolean("enabled"))
        self.assertEqual(adc["shmimName"], "acc000")
        self.assertEqual(adc["profile"], "accelSim12")
        self.assertEqual([adc.getint(k) for k in ("min_code", "max_code", "capacity")], [0, 4095, 4096])
        for axis in (0, 1):
            self.assertEqual(adc.getfloat(f"zero{axis}"), 2048.)
            self.assertEqual(adc.getfloat(f"volts_per_count{axis}"), .0083799)
            self.assertEqual(adc.getfloat(f"sensitivity{axis}"), 1.02)
            self.assertEqual(adc.getfloat(f"polarity{axis}"), 1.)


if __name__ == "__main__":
    unittest.main()
