r"""\file qualified_shadow_report_test.py
\brief Check synthetic/live separation and exact negative-control target identity using private ordinary artifacts.
\ingroup loPredCtrlAcc_unit_test
"""
import json
from pathlib import Path
import tempfile
import unittest

from qualified_shadow_report import inspect, independence


class QualifiedReportTest(unittest.TestCase):
    """Use minimal ordinary files, never a process, native stream, INDI or device."""

    def fixture(self, directory):
        """Write a complete explicitly synthetic one-horizon summary with paired two-axis targets."""
        summary = dict(evidence="synthetic_requested", live_qualified=False, deadline_qualified=False,
                       adc_samples=400, frames=100, adc_invalid=0, adc_dropped=0, score_drops=0,
                       standard_ready=True, assisted_ready=True, adc_rate_hz=8000, future=1)
        (directory / "summary.json").write_text(json.dumps(summary))
        (directory / "compute.csv").write_text("frame,whole_frame_us\n" + "".join(f"{i},10\n" for i in range(100)))
        (directory / "metrics.csv").write_text("horizon,channel,count,standard_rmse,assisted_rmse,welch_windows\n"
                                               "1,0,60,1,0.5,0\n1,1,60,2,1,0\n")
        (directory / "psd.csv").write_text("horizon,channel,frequency_hz,wfs_psd,standard_error_psd,assisted_error_psd\n")
        (directory / "coefficients.csv").write_text("mode,row,column,value\n0,0,0,0.4\n")
        (directory / "predictions.csv").write_text("mode,origin,target,horizon,channel,measurement\n0,10,11,1,0,0.5\n")
        return summary

    def test_explicit_synthetic_results_have_no_deadline_or_physical_claim(self):
        """MSE uses squared RMSE on matched targets; absent PSD windows remain unavailable."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            self.fixture(path)
            result = inspect(path)
            self.assertEqual(result["metrics"][0]["mse_improvement_percent"], 75)
            self.assertIsNone(result["metrics"][0]["band_56_improvement_percent"])
            self.assertEqual(result["summary"]["compute_us"]["maximum"], 10)

    def test_unknown_or_live_qualification_is_rejected(self):
        """No reported readiness, schema flag or file label can promote synthetic evidence to hardware proof."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            original = self.fixture(path)
            for key, value in (("live_qualified", True), ("deadline_qualified", True), ("evidence", "verified_applied"),
                               ("adc_dropped", 1), ("score_drops", 1), ("assisted_ready", False)):
                with self.subTest(key=key):
                    (path / "summary.json").write_text(json.dumps(dict(original, **{key: value})))
                    with self.assertRaises(ValueError):
                        inspect(path)

    def test_negative_control_requires_identical_standard_fit_and_wfs(self):
        """Changing a target or standard coefficient is a failed matched comparison, not an ADC effect."""
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a", Path(d) / "b"
            a.mkdir()
            b.mkdir()
            self.fixture(a)
            self.fixture(b)
            independence(a, b)
            (b / "predictions.csv").write_text((b / "predictions.csv").read_text().replace("0.5", "0.6"))
            with self.assertRaises(ValueError):
                independence(a, b)
            self.fixture(b)
            (b / "coefficients.csv").write_text((b / "coefficients.csv").read_text().replace("0.4", "0.6"))
            with self.assertRaises(ValueError):
                independence(a, b)


if __name__ == "__main__":
    unittest.main()
