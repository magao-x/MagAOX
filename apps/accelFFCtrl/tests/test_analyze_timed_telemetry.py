r"""\file test_analyze_timed_telemetry.py
\brief Original timing association, epoch precision, recorded-clock spectra and causal held-out prediction tests.
\ingroup accelFFCtrl_unit_test
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze_timed_telemetry import cadence, features, predict_record, spectral_grid, spectral_report, timed_stream
from inspect_reduced_telemetry import utc


class TimedTelemetryTest(unittest.TestCase):
    """Synthetic tests qualify analysis mechanics, not the real dataset's physical clock/exposure semantics."""

    def test_original_timing_association(self):
        """All original rows and exact file starts must correspond; malformed times/counters fail closed."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "acc000").mkdir()
            epoch = 1700000000000000001
            n = 128
            acquired = epoch + np.arange(n, dtype=np.int64) * 500000
            timing = np.column_stack(
                (
                    np.arange(1, n + 1),
                    acquired // 1000000000,
                    acquired % 1000000000,
                    acquired // 1000000000,
                    acquired % 1000000000,
                )
            ).astype(np.uint64)
            entries = []
            for start in (0, 64):
                suffix = (
                    utc(acquired[start])
                    .replace("-", "")
                    .replace(":", "")
                    .replace("T", "")
                    .replace(".", "")
                    .replace("Z", "")
                )
                source = f"acc000_{suffix}.xrif"
                name = Path(source).with_suffix(".npy").name
                np.save(root / "acc000" / name, np.full((64, 8), 660, dtype=np.uint16), allow_pickle=False)
                entries.append(dict(source_file=source, output_file=name, shape=[64, 8], dtype="uint16"))
            (root / "acc000" / "manifest.json").write_text(json.dumps(entries))
            path = root / "acc000_timings.npy"
            np.save(path, timing, allow_pickle=False)
            data, t, w, b = timed_stream(root, "acc000")
            self.assertEqual(int(t[0]), epoch)
            self.assertEqual(data.shape, (128, 8))
            self.assertEqual(len(b), 2)
            np.testing.assert_array_equal(t, w)
            for fault in ("boundary", "counter_gap", "nsec", "dtype", "length"):
                bad = timing.copy()
                if fault == "boundary":
                    bad[64, 2] += 1
                if fault == "counter_gap":
                    bad[60:, 0] += 1
                if fault == "nsec":
                    bad[0, 2] = 1000000000
                if fault == "dtype":
                    bad = bad.astype(float)
                if fault == "length":
                    bad = bad[:-1]
                np.save(path, bad, allow_pickle=False)
                with self.assertRaises(ValueError):
                    timed_stream(root, "acc000")

    def test_asymmetric_cadence(self):
        """Short median intervals can coexist with a 500-us mean and compensating negative serial correlation."""
        intervals = np.tile([490000, 490000, 520000], 4000)
        t = 1700000000000000001 + np.r_[0, np.cumsum(intervals)]
        report = cadence(t, t)
        self.assertAlmostEqual(report["interval_mean_ns"], 500000)
        self.assertEqual(report["interval_quantiles_ns"]["0.5"], 490000)
        self.assertAlmostEqual(report["average_rate_hz"], 2000)
        self.assertFalse(report["publication_latency_identified"])
        self.assertLess(report["lag1_interval_correlation"], 0)
        self.assertTrue(report["write_equals_acquisition"])

    def test_feature_causality_and_epoch_translation(self):
        """Future records cannot affect a query; integer epoch shifts preserve the exact admissible sample set."""
        n = 100
        t = 1700000000000000001 + np.arange(n, dtype=np.int64) * 500000
        at = t - 100000
        w = np.column_stack((np.arange(n), 2 * np.arange(n))).astype(float)
        a = np.column_stack((np.arange(n), 3 * np.arange(n), -np.arange(n))).astype(float)
        indices = np.array([20])
        x = features(a, at, w, t, indices, history=4, guard_ns=1000000)
        np.testing.assert_array_equal(x[0, :8], w[[20, 19, 18, 17]].ravel())
        np.testing.assert_array_equal(x[0, 8:], a[[18, 17, 16, 15]].ravel())
        a[at > t[20]] = 1e20
        w[t > t[20]] = -1e20
        np.testing.assert_array_equal(x, features(a, at, w, t, indices, history=4, guard_ns=1000000))
        np.testing.assert_array_equal(x, features(a, at + 37, w, t + 37, indices, history=4, guard_ns=1000000))
        with self.assertRaises(ValueError):
            features(a, at, w, t, np.array([2]), history=4)

    def test_chronological_predictor(self):
        """A known sensor-driven future innovation improves held-out ARX; late targets cannot tune ridge strengths."""
        rng = np.random.default_rng(130)
        n = 7000
        at = 1700000000000000001 + np.arange(n, dtype=np.int64) * 500000
        a = rng.normal(size=(n, 3))
        w = np.zeros((n, 2))
        for i in range(2, n):
            w[i] = 0.15 * w[i - 1] + np.array([a[i - 2, 0], a[i - 2, 1]]) + rng.normal(0, 0.01, 2)
        start, stop = int(at[64]), int(at[-1])
        scores, traces, alpha, split = predict_record(
            a, at - 100000, w, at, start, stop, steps=(2,), history=4, guards=(0,)
        )
        for axis in range(2):
            rows = scores[scores.axis == axis].set_index("model")
            self.assertLess(rows.loc["wfs_accel", "rmse"], 0.03 * rows.loc["wfs_only", "rmse"])
        changed = w.copy()
        changed[at >= split["test_start_ns"]] += rng.normal(0, 20, changed[at >= split["test_start_ns"]].shape)
        _, _, again, _ = predict_record(a, at - 100000, changed, at, start, stop, steps=(2,), history=4, guards=(0,))
        self.assertTrue(alpha.equals(again))
        self.assertTrue(np.all(at[traces[2, "wfs_only"][0]] >= split["test_start_ns"]))
        self.assertLess(split["last_refit_target_ns"], split["first_test_query_ns"])

    def test_irregular_clock_alignment_phase(self):
        """Actual acquisition-tag interpolation recovers a known phase; an arbitrary clock shift changes apparent phase."""
        rng = np.random.default_rng(13)
        epoch = 1700000000000000001
        n = 202000
        base = np.arange(n, dtype=np.int64) * 500000
        at = epoch + base + rng.integers(-10000, 10001, n)
        wt = epoch + base + rng.integers(-2000, 2001, n)

        def signal_at(stamps, delay=0):
            """Return two synthetic sinusoids evaluated from integer-differenced times."""
            t = (stamps - epoch) * 1e-9 - delay
            return np.cos(2 * np.pi * 28.75 * t) + 0.6 * np.sin(2 * np.pi * 55.75 * t)

        a = np.column_stack([signal_at(at)] * 3)
        w = np.column_stack([0.7 * signal_at(wt, 0.001)] * 2)
        start, stop = epoch + 10000000, epoch + 100010000000
        grid, x, y = spectral_grid(a, at, w, wt, start, stop, epoch)
        rows, _ = spectral_report(grid, x, y, start, stop)
        for frequency in (28.75, 55.75):
            row = rows[
                (rows.interval == "full")
                & (rows.accel_channel == 0)
                & (rows.wfs_axis == 0)
                & (rows.frequency_hz == frequency)
            ].iloc[0]
            self.assertGreater(row.coherence, 0.999)
            self.assertAlmostEqual(row.recorded_clock_phase_deg, -360 * frequency * 0.001, delta=0.05)
        _, _, shifted = spectral_grid(a, at, w, wt + 1000000, start, stop, epoch)
        changed, _ = spectral_report(grid, x, shifted, start, stop)
        row = changed[
            (changed.interval == "full")
            & (changed.accel_channel == 0)
            & (changed.wfs_axis == 0)
            & (changed.frequency_hz == 28.75)
        ].iloc[0]
        self.assertAlmostEqual(row.recorded_clock_phase_deg, -360 * 28.75 * 0.002, delta=0.05)


if __name__ == "__main__":
    unittest.main()
