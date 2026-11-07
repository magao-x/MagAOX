r"""\file test_inspect_reduced_telemetry.py
\brief Exact archive labels, file-local PSD units and read-only inventory tests using synthetic temporary arrays.
\ingroup accelFFCtrl_unit_test
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from inspect_reduced_telemetry import filename_ns, file_spectrum, inspect, inspect_stream, sha256, utc


class ReducedInspectionTest(unittest.TestCase):
    """Keep missing timing explicitly unqualified while testing array and spectral bookkeeping."""

    def setUp(self):
        """Prepare small synthetic file chunks, never instrument data or stream names."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root/"dataset"; self.data.mkdir()
        (self.data/"metadata.json").write_text(json.dumps(dict(acc_files=2, wfs_files=2, wfs_modes=dict(tip=0, tilt=1))))
        for stream, prefix, channels, dtype in (("acc000", "acc000", 8, "uint16"), ("wfs_tiptilt", "aol1_modevalWFS", 2, "float32")):
            folder = self.data/stream; folder.mkdir()
            manifest = []
            for index in range(2):
                stamp = 1744705800000000001+index*1024000000
                suffix = utc(stamp).replace("-", "").replace(":", "").replace(".", "").replace("T", "").replace("Z", "")
                source = f"{prefix}_{suffix}.xrif"
                name = Path(source).with_suffix(".npy").name
                t = np.arange(2048)/2000
                signal = (-1)**index*5*np.cos(2*np.pi*28*t)
                a = np.tile(signal[:, None], (1, channels))
                a = np.rint(660+a).astype(dtype) if dtype == "uint16" else a.astype(dtype)
                np.save(folder/name, a, allow_pickle=False)
                manifest.append(dict(source_file=source, output_file=name, shape=list(a.shape), dtype=dtype))
            (folder/"manifest.json").write_text(json.dumps(manifest))

    def test_exact_filename_label(self):
        """Nine encoded nanosecond digits survive parsing, independent of their unverified physical semantics."""
        self.assertEqual(filename_ns("acc000_20250415083000000000001.xrif"), 1744705800000000001)
        self.assertEqual(utc(1744705800000000001), "2025-04-15T08:30:00.000000001Z")
        for name in ("no_timestamp.npy", "acc000_20250415083000001.npy", "acc000_20250230083000000000000.xrif"):
            with self.assertRaises(ValueError): filename_ns(name)

    def test_sample_spectrum_units(self):
        """Peak and integrated power survive sample-to-conditional-Hz conversion without creating timestamps."""
        n = 4000; t = np.arange(n)/2000
        a = 660+3*np.cos(2*np.pi*28*t)+2*np.sin(2*np.pi*56*t)
        f, p = file_spectrum(np.column_stack((a, a)))
        self.assertAlmostEqual(f[np.argmax(p[:, 0])], 28/2000)
        self.assertAlmostEqual(np.sum(p[:, 0])/n, (3**2+2**2)/2)
        self.assertAlmostEqual(np.sum(p[:, 0]/2000)*(2000/n), (3**2+2**2)/2)
        _, opposite = file_spectrum(np.column_stack((1320-a, 1320-a)))
        np.testing.assert_allclose(opposite, p, rtol=1e-12, atol=1e-20)
        with self.assertRaises(ValueError): file_spectrum(np.array([[np.nan]]))

    def test_readonly_inventory_stays_unqualified(self):
        """All bytes remain identical; no frame timestamps, physical transfer or replay qualification is asserted."""
        before = {str(p): sha256(p) for p in self.data.rglob("*") if p.is_file()}
        report = inspect(self.data, self.root/"report", 2000, plots=False)
        self.assertEqual(report["streams"]["acc000"]["frames"], 4096)
        self.assertEqual(report["streams"]["acc000"]["channels"], 8)
        self.assertEqual(report["streams"]["acc000"]["median_bulk_record_density_hz"], 2000)
        self.assertFalse(report["physical_timing_qualified"])
        self.assertFalse(report["cross_sensor_analysis_performed"])
        self.assertFalse(report["prediction_replay_performed"])
        self.assertNotIn("conditional_last_frame_utc", report["streams"]["acc000"])
        self.assertEqual(before, {str(p): sha256(p) for p in self.data.rglob("*") if p.is_file()})
        with self.assertRaises(ValueError): inspect(self.data, self.data/"output", 2000, plots=False)

    def test_manifest_mismatch_and_order(self):
        """Unlisted arrays, duplicate chunks, reversed ordering and wrong shapes are integrity failures."""
        path = self.data/"acc000"/"manifest.json"
        original = json.loads(path.read_text())
        for mutate in (lambda m: list(reversed(m)), lambda m: [m[0], m[0]], lambda m: m[:1]):
            path.write_text(json.dumps(mutate(original)))
            with self.assertRaises(ValueError): inspect_stream(path.parent, "acc000", 2000)
        original[0]["shape"] = [2047, 8]
        path.write_text(json.dumps(original))
        with self.assertRaises(ValueError): inspect_stream(path.parent, "acc000", 2000)


if __name__ == "__main__":
    unittest.main()
