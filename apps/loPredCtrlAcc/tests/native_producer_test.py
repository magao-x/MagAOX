r"""\file native_producer_test.py
\brief Offline source/build contracts for the unchanged syncopation base and shadow-only integration.
\ingroup loPredCtrlAcc_unit_test
"""
import hashlib
from pathlib import Path
import re
import unittest

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parents[1]


class NativeProducerTests(unittest.TestCase):
    def test_existing_producer_is_reused_without_algorithm_changes(self):
        expected = {
            "libMagAOX/app/dev/dmSyncopationProducer.hpp": "2657e3cfc7a743046b648435f625010d7e355e3dfb8052479aa0c6e4e71fcf54",
            "libMagAOX/app/dmSyncopationTiming.hpp": "d6ca188b584e1576587120b0cb38b797c3d721cbdcb467656a214327ad10ebbb",
        }
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)

    def test_shadow_application_never_polls_or_publishes_auxiliary_transport(self):
        source = (APP / "loPredCtrlAcc.hpp").read_text()
        self.assertIn("protected dev::dmSyncopationProducer<loPredCtrlAcc>", source)
        for macro in ("SETUP_CONFIG", "LOAD_CONFIG", "APP_SHUTDOWN"):
            self.assertIn("DMSYNCOPATIONPRODUCER_" + macro, source)
        self.assertNotRegex(source, r"dmSyncopationProducerT::syncopation(Poll|Publish)\s*\(")
        self.assertIn('"dm10disp10_aux"', source)
        self.assertIn("override DDSPC_SHADOW_ONLY = yes", (APP / "Makefile").read_text())

    def test_schedule_observer_has_no_producer_or_semaphore_operations(self):
        source = (APP / "native_observer.cpp").read_text()
        reader = source[source.index("class ReadOnlySchedule"):source.index("/// Capture actual local monotonic")]
        self.assertIn("synco::readRecord", reader)
        for forbidden in ("openRecord", "createIm", "UpdateIm", "sempost", "semflush", "getsemwaitindex", "flock"):
            self.assertNotIn(forbidden, reader)

    def test_default_configuration_never_opts_into_training(self):
        import configparser
        config = configparser.ConfigParser()
        config.read_string("[application]\n" + (APP / "loPredCtrlAcc.conf").read_text())
        self.assertNotIn("learn", config["parameters"])
        self.assertNotIn("predict", config["parameters"])
        self.assertEqual(config["dm"]["channelName"], "dm10disp10_aux")
        self.assertNotIn("native_inputs", config)


if __name__ == "__main__":
    unittest.main()
