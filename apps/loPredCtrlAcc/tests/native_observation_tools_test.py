r"""\file native_observation_tools_test.py
\brief Offline private configuration/artifact validation; never invokes INDI or opens a native stream.
\ingroup loPredCtrlAcc_unit_test
"""
import configparser
import csv
import json
import struct
from pathlib import Path
import tempfile
import unittest

from verify_native_observation import verify
from inspect_native_commands import inspect


class NativeObservationToolsTest(unittest.TestCase):
    """Test private handoff settings and reject incomplete/corrupt observation evidence."""

    def test_full_command_journal_retains_all_values_and_rejects_truncation(self):
        """Decode complete physical-coordinate requests without promoting them to applied evidence."""
        with tempfile.TemporaryDirectory(prefix="ddspc-command-smoke-") as temporary:
            path = Path(temporary) / "dm_commands.bin"
            header = struct.pack("<6Q4q7Q", 4, 1, 2, 3, 4, 123, -1, 100, 200, 210,
                                 2, 2, 1, 9, 0, 17, 19)
            path.write_bytes(b"DDDM0001" + header + struct.pack("<4f", .1, .2, .3, .4))
            result = inspect(path)
            self.assertEqual(result["records"], 1)
            self.assertEqual(result["latest"]["counter"], 123)
            self.assertEqual(result["latest"]["acquired_raw_ns"], -1)
            self.assertAlmostEqual(result["latest"]["first_values"][3], .4)
            self.assertFalse(result["physical_application_qualified"])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                inspect(path)
            self.assertTrue(inspect(path, allow_partial=True)["incomplete_tail"])

    def test_private_config_preserves_adc_calibration_and_inhibits_control(self):
        """Compare the entire ADC section with the previously validated ADC-only example."""
        examples = Path(__file__).resolve().parents[1] / "examples"
        old, new = configparser.ConfigParser(), configparser.ConfigParser()
        old.read(examples / "loPredCtrlAcc-adc-shadow.conf.example")
        new.read(examples / "loPredCtrlAcc-native-observe.conf.example")
        self.assertEqual(dict(old["adc"]), dict(new["adc"]))
        self.assertTrue(new.getboolean("native_shadow", "enabled"))
        self.assertTrue(new.getboolean("parameters", "shadow"))
        for option in ("shadow_pipeline", "own_shmim", "accel_enabled", "is_integrating"):
            self.assertFalse(new.getboolean("parameters", option))
        self.assertEqual(new["shmimMonitor"]["shmimName"], "")
        self.assertEqual(new["native_shadow"]["dm_stream"], "dm01disp")
        self.assertEqual(new["native_shadow"]["completion_stream"], "")
        self.assertEqual(new["native_shadow"]["sdk_stream"], "")
        self.assertNotIn("framegrabber", [name.lower() for name in new.sections()])
        live = configparser.ConfigParser()
        live.read(examples / "loPredCtrlAcc-live-inputs.conf.example")
        self.assertEqual(dict(old["adc"]), dict(live["adc"]))
        self.assertTrue(live.getboolean("native_shadow", "record_dm_commands"))
        self.assertFalse(live.getboolean("parameters", "shadow_pipeline"))
        self.assertNotIn("framegrabber", [name.lower() for name in live.sections()])

    def test_artifacts_report_limitations_without_authorizing_training(self):
        """A transport pass must leave provenance unqualified; loss, corrupt counters and unsafe flags fail."""
        with tempfile.TemporaryDirectory(prefix="ddspc-observation-tools-") as temporary:
            directory = Path(temporary)
            sources = []
            for role in ("adc", "modal", "camera", "dm", "completion", "sdk"):
                s = dict.fromkeys(("counter_gaps", "restarts", "protocol_restarts", "clock_unknown",
                                   "clock_mismatches", "invalid", "protocol_invalid", "protocol_incomplete",
                                   "sequence_gaps", "association_failures", "metadata_associations", "busy", "races",
                                   "incompatible"), 0)
                s.update(role=role, samples=2 if len(sources) < 4 else 0)
                sources.append(s)
            sources[1]["association_failures"] = 2
            summary = dict(schema=1, observation_only=True, training_enabled=False, exposure_qualified=False,
                           physical_application_qualified=False, io_error=False, started_ns=1, stopped_ns=125001,
                           queue_drops=0, written=8, adc_missing=0, sources=sources,
                           adc_capture=dict(dropped=0, invalid=0, timing_gaps=0, restarts=0))
            fields = ("source", "event", "device", "file_inode", "generation", "counter", "acquired_raw_ns",
                      "received_mono_ns", "available_mono_ns")
            with (directory / "observations.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                for slot in range(4):
                    for n in range(2):
                        writer.writerow(dict(zip(fields, (slot, 0, 1, 2, 1, n+1, 1000000000+n*125000, 10, 11))))
            def check():
                """Evaluate a newly serialized test-owned summary against immutable synthetic CSV evidence."""
                (directory / "summary.json").write_text(json.dumps(summary))
                return verify(directory, minimum_seconds=0)
            result = check()
            self.assertTrue(result["transport_pass"])
            self.assertFalse(result["training_allowed"])
            self.assertIn("modal.association_failures=2", result["limitations"])
            summary["adc_capture"]["dropped"] = 1
            self.assertFalse(check()["transport_pass"])
            summary["adc_capture"]["dropped"] = 0
            summary["training_enabled"] = True
            self.assertFalse(check()["transport_pass"])
            summary["training_enabled"] = False
            summary["written"] = 9
            self.assertFalse(check()["transport_pass"])
            summary["written"] = 8
            summary.update(dm_commands_enabled=True, dm_commands_written=2,
                           dm_command_drops=0, dm_command_oversize=0)
            payload = b"DDDM0001"
            for counter in (1, 2):
                payload += struct.pack("<6Q4q7Q", 1, 1, 2, 3, 1, counter, 100, 110, 120, 130,
                                       1, 1, 1, 9, 0, 0, 0) + struct.pack("<f", .1)
            (directory / "dm_commands.bin").write_bytes(payload)
            self.assertTrue(check()["transport_pass"])
            summary["dm_command_drops"] = 1
            self.assertFalse(check()["transport_pass"])
            summary["dm_command_drops"] = 0
            summary["dm_commands_written"] = 1
            self.assertFalse(check()["transport_pass"])


if __name__ == "__main__":
    unittest.main()
