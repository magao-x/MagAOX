r"""\file test_analyze_timestamps.py
\brief Integer-clock, cadence, gap-eligibility and prediction-timing audit regression tests.
\ingroup accelFFCtrl_unit_test
"""
import csv
from pathlib import Path
import tempfile
import unittest

from analyze_timestamps import FRAME_FIELDS, analyze, frame_analysis, query_analysis, read_csv, wfs_alignment


def frames(rate=8000, indices=(0, 1, 3), offset=0):
    """Create exact epoch stamps, deliberately above the float64 integer-precision boundary."""
    epoch = 1_700_000_000_000_000_007
    period = 1_000_000_000//rate
    result = []
    for index in indices:
        acquired = epoch+index*period+offset
        mono = 1_000_000_000+index*period+offset
        result.append(dict(zip(FRAME_FIELDS, (1, index+1, acquired, acquired+25_000,
                                              mono+50_000, acquired+50_000, mono+50_000, 10, 1, 1))))
    return result


class TimestampAnalysisTest(unittest.TestCase):
    """Audit prerequisites conservatively, without pretending to reconstruct harmonic maturity."""

    def test_exact_epoch_and_csv(self):
        """Single nanoseconds survive CSV parsing and clock translation without float epochs."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"frames.csv"
            rows = frames()
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FRAME_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(read_csv(path, FRAME_FIELDS), rows)
            report = analyze(path, Path(directory)/"report", 8000)
            self.assertEqual(report["timing_eligible_gaps"], 1)
            self.assertEqual(report["sample_age_ns"]["max"], 50_000)
            self.assertTrue((Path(directory)/"report"/"summary.json").exists())
            path.write_text(path.read_text().replace(str(rows[0]["acquired_ns"]), "1.7e18"))
            with self.assertRaises(ValueError):
                read_csv(path, FRAME_FIELDS)

    def test_guard_boundary_at_both_rates(self):
        """The exact .05P edge passes; one ns beyond resets timing eligibility only."""
        for rate in (4000, 8000):
            period = 1_000_000_000//rate
            for jitter, expected in ((0, 1), (period//20, 1), (period//20+1, 0), (-period//10, 0)):
                rows = frames(rate)
                rows[-1]["acquired_ns"] += jitter
                report, _ = frame_analysis(rows, rate)
                self.assertEqual(report["observed_missing_lower_bound"], 1)
                self.assertEqual(report["timing_eligible_gaps"], expected)
                self.assertEqual(report["timing_fault_records"], 0)

    def test_cluster_and_generation(self):
        """Several missing frames, rollback and producer restart never become a single retained gap."""
        rows = frames(indices=(0, 1, 5))
        report, _ = frame_analysis(rows, 8000)
        self.assertEqual(report["observed_missing_lower_bound"], 3)
        self.assertEqual(report["timing_eligible_gaps"], 0)
        rows[-1]["counter"] = 1
        report, _ = frame_analysis(rows, 8000)
        self.assertEqual(report["counter_rollbacks"], 1)
        self.assertEqual(report["timing_eligible_gaps"], 0)
        rows[-1]["generation"] = 2
        report, _ = frame_analysis(rows, 8000)
        self.assertEqual(report["generation_changes"], 1)
        self.assertEqual(report["counter_rollbacks"], 0)

    def test_integrity_and_stale(self):
        """Bad clock brackets, invalid ADC/time, stale priors and rejected records cannot qualify a gap."""
        changes = ({"clock_uncertainty_ns": -1}, {"acquired_ns": 0}, {"written_ns": 0},
                   {"clock_real_ns": 1}, {"adc_x": 4096, "adc_y": 2048}, {"depth": 8},
                   {"acquired_sec": 1, "acquired_nsec": 1_000_000_000},
                   {"acquired_ns": 2**63})
        for change in changes:
            rows = frames()
            rows[-1].update(change)
            report, _ = frame_analysis(rows, 8000)
            self.assertEqual(report["timing_eligible_gaps"], 0)
        rows = frames()
        for key in ("receipt_mono_ns", "clock_real_ns", "clock_mono_ns"):
            rows[-1][key] += 1_800_000  # New age is fresh, preceding acquisition is stale.
        report, _ = frame_analysis(rows, 8000)
        self.assertEqual(report["timing_fault_records"], 0)
        self.assertEqual(report["timing_eligible_gaps"], 0)
        rows = frames()
        rows[1]["accepted"] = 0
        self.assertEqual(frame_analysis(rows, 8000)[0]["timing_eligible_gaps"], 0)

    def test_period_drift(self):
        """A centered counter/time slope measures per-publication drift despite missing observations."""
        rows = frames(indices=(0, 1, 3, 10))
        for row in rows:
            row["acquired_ns"] += (row["counter"]-1)*25
        report, _ = frame_analysis(rows, 8000)
        self.assertAlmostEqual(report["segment_period_drift_ppm"][0], 200)
        self.assertEqual(report["cadence_error_per_publication_ns"]["max"], 25)

    def test_query_horizon_and_stale(self):
        """Target must equal query+horizon, with measurement age included once and stale readiness flagged."""
        row = {"query_ns": 1_000_000_000, "sample_ns": 999_875_000,
               "target_ns": 1_001_000_000, "ready": 1}
        report = query_analysis([row])
        self.assertEqual(report["effective_horizon_ns"]["max"], 1_125_000)
        self.assertEqual(report["invalid_ready_queries"], 0)
        for key, value in (("target_ns", 1_001_125_000), ("sample_ns", 997_000_000),
                           ("sample_ns", 1_000_000_001)):
            bad = dict(row, **{key: value})
            self.assertEqual(query_analysis([bad])["invalid_ready_queries"], 1)
        with self.assertRaises(ValueError):
            query_analysis([dict(row, dropped=2), dict(row, dropped=1)])

    def test_observed_wfs_alignment(self):
        """Observer WFS receipt alignment is reported separately from acquisition phase."""
        wfs = frames(indices=(0,))
        _, audited_wfs = frame_analysis(wfs, 2000)
        _, audited_acc = frame_analysis(frames(indices=(1,)), 8000)
        report = wfs_alignment(audited_acc, list(zip(wfs, audited_wfs)))
        self.assertEqual(report["acc_since_observed_wfs_receipt_ns"]["mean"], 75_000)
        self.assertEqual(report["acc_since_that_wfs_acquisition_ns"]["mean"], 125_000)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name, rows in (("acc", frames(indices=(1,))), ("wfs", wfs)):
                with (directory/f"{name}.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FRAME_FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
            report = analyze(directory/"acc.csv", directory/"report", 8000, wfs=directory/"wfs.csv")
            self.assertEqual(report["wfs_timing"]["records"], 1)
            self.assertEqual(report["wfs_alignment"]["acc_since_observed_wfs_receipt_ns"]["mean"], 75_000)


if __name__ == "__main__":
    unittest.main()
