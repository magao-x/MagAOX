r"""\file handoff_config_test.py
\brief Check that the manual example stays an explicit output-inhibited placeholder, never a live-training recipe.
"""
import configparser
from pathlib import Path
import unittest


class HandoffConfigTest(unittest.TestCase):
    """Validate the distributed manual fixture; no application, stream, network or device is opened."""

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


if __name__ == "__main__":
    unittest.main()
