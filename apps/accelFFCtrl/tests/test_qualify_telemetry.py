r"""\file test_qualify_telemetry.py
\brief Deterministic spectral, timestamp, resampling, causal-replay and inhibition qualification.
\ingroup accelFFCtrl_unit_test
"""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from qualify_telemetry import (ADC_FIELDS, align_runs, analyze, decode, integer_csv, map_clock,
                               replay, score_prediction, select_horizon, spectra, synthetic_fixture, timing)


class SpectralQualificationTest(unittest.TestCase):
    """Independent analytic transfers, exact clocks and deliberately uncorrelated records."""

    def test_phase_delay_and_psd(self):
        """Conj(X)*Y gives a negative phase for a known delayed response, and PSD integrates to variance."""
        rng = np.random.default_rng(12)
        fs = 2000
        t = np.arange(80000)/fs
        x = rng.normal(size=(len(t), 2))
        y = -2*np.roll(x, 4, axis=0)
        result = spectra([(t[100:], x[100:], y[100:])], fs)
        self.assertGreater(result["segments"], 8)
        for frequency in (28, 56):
            i = np.argmin(abs(result["f"]-frequency))
            h = result["h"][i, 0]
            self.assertAlmostEqual(abs(h), 2, delta=.01)
            self.assertAlmostEqual(-np.angle(h/-1)/(2*np.pi*frequency), .002, delta=2e-5)
            self.assertGreater(result["coherence"][i, 0], .999)
        self.assertAlmostEqual(np.trapezoid(result["pxx"][:, 0], result["f"]), np.var(x[:, 0]), delta=.03)

    def test_known_sinusoidal_transfer(self):
        """Peak amplitude, phase and angular inverse have the same signs as the production estimator."""
        t = np.arange(80000)/2000
        x = np.column_stack([np.cos(2*np.pi*28*t+.3), np.cos(2*np.pi*56*t-.7)])
        y = np.column_stack([3*np.cos(2*np.pi*28*t+.3-.2), .4*np.cos(2*np.pi*56*t-.7+.5)])
        result = spectra([(t, x, y)], 2000)
        for axis, frequency, gain, phase in ((0, 28, 3, -.2), (1, 56, .4, .5)):
            index = np.argmin(abs(result["f"]-frequency))
            self.assertAlmostEqual(abs(result["h"][index, axis]), gain, places=10)
            self.assertAlmostEqual(np.angle(result["h"][index, axis]), phase, places=10)

    def test_low_coherence_rejected(self):
        """Independent sensor/WFS noise cannot qualify a transfer even when individual spectra are finite."""
        rng = np.random.default_rng(433)
        t = np.arange(120000)/2000
        result = spectra([(t, rng.normal(size=(len(t), 2)), rng.normal(size=(len(t), 2)))], 2000)
        for frequency in (28, 56):
            index = np.argmin(abs(result["f"]-frequency))
            self.assertFalse(np.any(result["accepted"][index]))
        with self.assertRaises(ValueError):
            spectra([(t[:3000], np.ones((3000, 2)), np.ones((3000, 2)))], 2000)

    def test_pipeline_refuses_unqualified_transfer_and_clocks(self):
        """A complete unrelated WFS file yields no calibrated forecast; unknown clocks fail before phase inference."""
        with tempfile.TemporaryDirectory() as root:
            path = synthetic_fixture(Path(root)/"data", 40)
            wfs_path = path.parent/"wfs.csv"
            wfs = pd.read_csv(wfs_path)
            wfs[["tip", "tilt"]] = np.random.default_rng(918).normal(0, 1e-7, (len(wfs), 2))
            wfs.to_csv(wfs_path, index=False)
            output = Path(root)/"report"
            report = analyze(path, output, Path("/not/an/executable"), plots=False)
            self.assertIn("prediction_inhibited", report)
            self.assertTrue(all(not b["transfer_accepted"] and b["magnitude"] is None for b in report["bands"]))
            self.assertFalse((output/"prediction_metrics.csv").exists())
            meta = json.loads(path.read_text()); meta["wfs"]["timing_verified"] = False
            path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError): analyze(path, output, plots=False)

    def test_antialias_and_fractional_alignment(self):
        """8-to-2-kHz conversion suppresses an alias; fractional exposure centers preserve 28-Hz phase."""
        epoch = 1700000000000000000
        t = np.arange(48000)/8000
        wt = np.arange(12000)/2000+.000123
        x = np.column_stack([np.cos(2*np.pi*28*t)+np.cos(2*np.pi*1500*t)]*2)
        a = pd.DataFrame(dict(acquired_ns=epoch+np.rint(t*1e9).astype(np.int64), counter=np.arange(1, len(t)+1), generation=1))
        w = pd.DataFrame(dict(acquired_ns=epoch+np.rint(wt*1e9).astype(np.int64), counter=np.arange(1, len(wt)+1), generation=1,
                              tip=np.cos(2*np.pi*28*wt), tilt=np.cos(2*np.pi*28*wt)))
        blocks = align_runs(a, w, x, 8000, 2000, epoch)
        grid, down, _ = blocks[0]
        self.assertLess(np.std(down[:, 0]-np.cos(2*np.pi*28*grid)), .001)
        a = a.drop(index=24000)
        blocks = align_runs(a, w, x, 8000, 2000, epoch)
        self.assertFalse(any(b[0][0] < 3 < b[0][-1] for b in blocks))

    def test_exact_timestamps_and_clock_registration(self):
        """Epoch ns beyond 2^53 remain exact; clock corrections are supplied independently of the signals."""
        epoch = 1700000000000000001
        frame = pd.DataFrame(dict(acquired_ns=[epoch, epoch+125000], written_ns=[epoch+10, epoch+125010],
                                  receipt_ns=[epoch+20, epoch+125020], counter=[1, 3], generation=[1, 1], adc_x=[2048, 2050], adc_y=[2047, 2048]))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"a.csv"; frame.to_csv(path, index=False)
            data = integer_csv(path, ADC_FIELDS)
            self.assertEqual(int(data.acquired_ns.iloc[0]), epoch)
            path.write_text(path.read_text().replace(str(epoch), str(epoch)+".0"))
            with self.assertRaises(ValueError): integer_csv(path, ADC_FIELDS)
        report, _ = timing(frame, 8000)
        self.assertEqual(report["missing"], 1)
        clock = dict(timing_verified=True, timestamp_reference="acquisition", clock_offset_ns=700000,
                     clock_anchor_ns=epoch, clock_drift_ppm=10, clock_uncertainty_ns=1)
        corrected = map_clock(frame, clock)
        self.assertEqual(int(corrected.acquired_ns.iloc[0]), epoch+700000)
        clock["timing_verified"] = False
        with self.assertRaises(ValueError): map_clock(frame, clock)
        for field, value in (("acquired_ns", 0), ("written_ns", epoch-1), ("receipt_ns", epoch-1), ("counter", 0)):
            bad = frame.copy(); bad.loc[0, field] = value
            with self.assertRaises(ValueError): timing(bad, 8000)

    def test_adc_units_and_horizon_choice(self):
        """ADC calibration has SI acceleration units; the selector uses held-out RMSE rather than a preferred delay."""
        data = pd.DataFrame(dict(adc_x=[2049], adc_y=[2047]))
        values = decode(data, dict(zero=[2048]*2, volts_per_bit=[.0083799]*2, sensitivity=[1.02]*2))
        np.testing.assert_allclose(values, [[.0083799/1.02, -.0083799/1.02]])
        self.assertAlmostEqual(-values[0, 0]/((2*np.pi*28)**2*10), -2.654361924322414e-8)
        table = pd.DataFrame(dict(horizon_ns=[0, 0, 1750000, 1750000, 5000000, 5000000], nrmse=[.1, .11, .02, .03, .2, .21]))
        self.assertEqual(select_horizon(table), 1750000)

    def test_horizon_scoring_uses_target_not_query(self):
        """An intentionally mistargeted analytic forecast selects its true 1.75-ms timestamp; WFS holes are excluded."""
        epoch = 1700000000000000000
        t = np.arange(80000)/2000
        q = 1_000_000_000+np.arange(len(t), dtype=np.int64)*500000
        truth = np.cos(2*np.pi*28*t)
        wfs = pd.DataFrame(dict(acquired_ns=epoch+q-1_000_000_000, counter=np.arange(1, len(t)+1), generation=1, tip=truth, tilt=truth))
        phase = 2*np.pi*28*(t+.00175)
        metrics = []
        for horizon in (0, 1000000, 1750000, 3000000):
            d = pd.DataFrame(dict(query_ns=q, target_ns=q+horizon, valid=1, ax_re0=np.cos(phase), ax_im0=np.sin(phase),
                                  ay_re0=np.cos(phase), ay_im0=np.sin(phase)))
            rows, _ = score_prediction(d, wfs, epoch, np.ones((1, 2), dtype=complex), [28], 1, 39)
            metrics.extend([dict(row, horizon_ns=horizon) for row in rows])
            if horizon == 1750000:
                with_gap = wfs.drop(index=np.arange(40000, 40200))
                fewer, _ = score_prediction(d, with_gap, epoch, np.ones((1, 2), dtype=complex), [28], 1, 39)
                self.assertLess(fewer[0]["count"], rows[0]["count"]-190)
        self.assertEqual(select_horizon(pd.DataFrame(metrics)), 1750000)


@unittest.skipUnless(os.environ.get("ACCELFFCTRL_TELEMETRY_REPLAY"), "build/run make telemetry-test for native numerical replay")
class ProductionFileReplayTest(unittest.TestCase):
    """Exercise the compiled production predictor and Phase 11 shadow with local files only."""

    def setUp(self):
        """Prepare a temporary analytic input; it never touches a stream or application singleton."""
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.exe = Path(os.environ["ACCELFFCTRL_TELEMETRY_REPLAY"])
        epoch = 1700000000000000000
        n = np.arange(8000); t = n/8000
        adc = np.rint(2048+20*np.cos(2*np.pi*28*t+.3)).astype(int)
        self.acc = pd.DataFrame(dict(acquired_ns=epoch+n*125000, written_ns=epoch+n*125000+20000,
                                      receipt_ns=epoch+n*125000+50000, counter=n+1, generation=1, adc_x=adc, adc_y=adc))
        self.origin = epoch
        self.meta = dict(accel=dict(rate_hz=8000), estimator=dict(memory_s=.1, lever_m=10, rotation_deg=0, frequencies_hz=[28]),
                         adc=dict(zero=[2048]*2, volts_per_bit=[.0083799]*2, sensitivity=[1.02]*2),
                         query=dict(rate_hz=2000, start_offset_ns=100000),
                         shadow=dict(gain=0, bounds=[1, 1], calibration_id=12, evidence=0,
                                     coordinates=np.eye(2).tolist(), modal_map=(1e6*np.eye(2)).tolist(), provenance="synthetic"))

    def test_default_zero_and_exact_horizon(self):
        """Mature estimates preserve target times and signed analytic acceleration quadratures at zero command gain."""
        d = replay(self.exe, self.meta, self.acc, self.origin, self.root, 1750000)
        self.assertGreater(d.valid.mean(), .69)
        self.assertTrue(np.all(d[["mode0", "mode1"]] == 0))
        v = d.valid.astype(bool)
        phase = 2*np.pi*28*(d.target_ns[v]-1e9)*1e-9+.3
        expected = 20*.0083799/1.02*np.cos(phase)
        self.assertLess(np.std(d.ax_re0[v]-expected)/np.std(expected), .02)
        np.testing.assert_allclose(d.px0[v], -d.ax_re0[v]/((2*np.pi*28)**2*10), rtol=1e-12)

    def test_drop_restart_stale_and_clipping(self):
        """Single lost frames retain default full warmup; new sources, stale input and bad data inhibit instead of holding."""
        self.meta["shadow"].update(gain=1, bounds=[1e-4, 1e-4])
        good = replay(self.exe, self.meta, self.acc, self.origin, self.root, 1000000)
        self.assertTrue(np.any(good.state == 3))
        for fault in ("drop", "restart", "invalid_adc", "invalid_time", "stale"):
            a = self.acc.copy()
            if fault == "drop": a = a.drop(index=4000)
            if fault == "restart": a.loc[4000:, "generation"] = 2
            if fault == "invalid_adc": a.loc[4000:4100, "adc_x"] = 4096
            if fault == "invalid_time": a.loc[4000:4100, "acquired_ns"] = 0
            if fault == "stale": a.loc[4000:4100, "receipt_ns"] += 3_000_000; a = a.sort_values("receipt_ns", kind="stable")
            d = replay(self.exe, self.meta, a, self.origin, self.root, 1000000)
            # A delayed record does not invalidate an earlier still-fresh acquisition immediately.
            start = 1_504_000_000 if fault == "stale" else 1_501_000_000
            lost = (d.query_ns >= start) & (d.query_ns < 1_790_000_000)
            self.assertTrue(np.all(d.loc[lost, "valid"] == 0), fault)
            self.assertTrue(np.all(d.loc[lost, ["mode0", "mode1"]] == 0), fault)
            self.assertGreater(d.resets.iloc[-1], 1)
            if fault == "drop": self.assertEqual(d.dropped.iloc[-1], 1)

    def test_synthetic_response_is_explicit(self):
        """The optional existing numerical plant has zero response at zero gain and rejects a claimed measured model."""
        self.meta["synthetic_response"] = dict(evidence="synthetic", provenance="unit fixture only", period_ns=500000,
                                                delay_ns=1000000, bandwidth_hz=300, mapping=(1e-6*np.eye(2)).tolist())
        d = replay(self.exe, self.meta, self.acc, self.origin, self.root, 1000000)
        self.assertTrue(np.all(d[["synthetic_effect_x", "synthetic_effect_y"]] == 0))
        self.meta["synthetic_response"]["evidence"] = "measured"
        with self.assertRaises(ValueError): replay(self.exe, self.meta, self.acc, self.origin, self.root, 1000000)


if __name__ == "__main__":
    unittest.main()
