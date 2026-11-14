r"""\file offline_eof_test.py
\brief Synthetic native-format loader, provisional association, purged EOF and negative-control regressions.
\ingroup loPredCtrlAcc_unit_test
"""
from copy import deepcopy
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy import signal
from threadpoolctl import threadpool_limits

from offline_eof import (Samples, association_rows, contiguous_runs, experiment, features, fit_eof,
                         independent_sensor, load_recording, predict, provisional_associations,
                         segmented_psd, split_origins, target_values)
from native_timing_audit_test import FIELDS as TIMING_FIELDS, record, write_fixture

CALIBRATION = .0083799/1.02
BASE = 1000000


def synthetic_records(frames=8000, coupled=True, seed=813):
    """Create a causal ten-frame sensor-to-WFS delay with seeded innovations and 28.5/56-Hz lines; no native streams."""
    rng = np.random.default_rng(seed)
    t = np.arange(frames+5)/2000
    adc = signal.lfilter([1], [1, -.25], rng.normal(size=(frames+5, 2)), axis=0)
    adc += .3*np.column_stack((np.sin(2*np.pi*28.5*t), np.sin(2*np.pi*56*t)))
    raw = np.clip(2048+np.rint(adc/CALIBRATION), 1, 4094).astype(int)
    adc = (raw-2048)*CALIBRATION
    innovations = rng.normal(scale=.15 if coupled else 1, size=(frames, 2))
    if coupled:
        innovations[10:] += adc[:frames-10]@np.array([[.9, -.1], [.15, .7]])
    wfs = signal.lfilter([1], [1, -.4 if coupled else -.7], innovations, axis=0)
    rows = []
    for n in range(frames*4+18):
        stamp = BASE+n*125000
        r = record(0, n+1, stamp, stamp+10000, stamp+20000)
        r.update(flags=0, width=2, height=1, depth=128, datatype=3, elements=2,
                 value0=adc[n//4, 0], value1=adc[n//4, 1], raw0=int(raw[n//4, 0]), raw1=int(raw[n//4, 1]))
        rows.append(r)
    for k in range(frames):
        stamp = BASE+k*500000
        camera = record(2, k+1, stamp+300000, stamp+430000, stamp+450000)
        camera.update(flags=2, width=120, height=120, depth=1, datatype=3, elements=14400,
                      value0=0, value1=0, raw0=0, raw1=0)
        modal = record(1, k+1, stamp+470000, stamp+470000, stamp+490000)
        # Deliberately unknown and different raw clock domain; the analysis must never subtract it from ADC time.
        modal['acquired_raw_ns'] += 37000000000
        modal['published_raw_ns'] += 37000000000
        modal.update(flags=2 | 64, width=2, height=1, depth=1, datatype=9, elements=2,
                     value0=wfs[k, 0], value1=wfs[k, 1], raw0=0, raw1=0)
        rows.extend((camera, modal))
    return rows


def write_measurements(directory, rows):
    """Reuse the journal fixture writer and add the real modal selection/unknown clock declarations."""
    write_fixture(directory, [{key: row[key] for key in TIMING_FIELDS} for row in rows])
    with (directory/'observations.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row['available_mono_ns']))
    path = directory/'summary.json'
    summary = json.loads(path.read_text())
    summary['modal_indices'] = [0, 1]
    for slot in (1, 2):
        summary['sources'][slot]['acquisition_clock'] = 'unknown'
        summary['sources'][slot]['publication_clock'] = 'unknown'
    path.write_text(json.dumps(summary))


def as_sources(rows):
    """Build actual loader-column objects for fast numerical tests using the same processing method."""
    sources = {name: Samples() for name in ('adc', 'modal', 'camera')}
    for row in rows:
        sources[('adc', 'modal', 'camera')[row['source']]].append_sample(row, ('adc', 'modal', 'camera')[row['source']])
    return sources


class OfflineEofTest(unittest.TestCase):
    """Validate ordinary-file analysis and mathematical information recovery without instrument dependencies."""

    @classmethod
    def setUpClass(cls):
        """Bound numerical concurrency so deterministic small tests do not depend on BLAS thread heuristics."""
        cls.threads = threadpool_limits(limits=1)
        cls.rows = synthetic_records(frames=2400)
        cls.sources = as_sources(cls.rows)

    @classmethod
    def tearDownClass(cls):
        """Restore the test process's original BLAS limits."""
        cls.threads.restore_original_limits()

    def test_native_loader_preserves_values_raw_clocks_and_uint64(self):
        """Read actual native measurement columns without confusing TAI-like raw epochs with observer time."""
        rows = deepcopy(self.rows)
        for row in rows:
            row['counter'] += 2**53+7
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_measurements(directory, rows)
            sources, report = load_recording(directory, minimum_seconds=0)
        self.assertTrue(report['transport']['transport_pass'])
        self.assertEqual(sources['adc'].counter[0], 2**53+8)
        self.assertEqual(sources['adc'].raw[0][0], rows[0]['raw0'])
        self.assertEqual(sources['adc'].values[0][0], rows[0]['value0'])
        self.assertEqual(report['sources']['modal']['acquisition_clock'], 'unknown')
        self.assertEqual(report['sources']['adc']['valid_samples'], 9618)
        self.assertAlmostEqual(report['sources']['adc']['observer_rate_hz'], 8000)
        self.assertGreater(sources['modal'].acquired[0]-sources['adc'].acquired[0], 37000000000)

    def test_loader_rejects_timing_only_fixture_and_wrong_modal_selection(self):
        """Missing measurements or impossible selected indices cannot silently become zero-valued inputs."""
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_measurements(directory, self.rows)
            summary = json.loads((directory/'summary.json').read_text())
            summary['modal_indices'] = [0, 99]
            (directory/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'extent'):
                load_recording(directory, 0)
            summary['modal_indices'] = [0, 1]
            (directory/'summary.json').write_text(json.dumps(summary))
            (directory/'observations.csv').write_text('source,event\n0,0\n')
            with self.assertRaisesRegex(ValueError, 'columns'):
                load_recording(directory, 0)

    def test_invalid_samples_dimensions_datatypes_and_recovery(self):
        """Bad records remain explicit holes, and later valid data can resume without imputation."""
        rows = deepcopy(self.rows)
        rows[44]['value0'] = float('nan')
        rows[48]['datatype'] = 9
        rows[52]['width'] = 3
        sources = as_sources(rows)
        self.assertFalse(sources['adc'].valid[44])
        self.assertFalse(sources['adc'].valid[48])
        self.assertFalse(sources['adc'].valid[52])
        self.assertTrue(sources['adc'].valid[56])
        a = provisional_associations(sources)
        self.assertGreater(a['report']['status_counts']['invalid_sample'], 0)
        self.assertTrue(a['valid'][-1])

    def test_receipt_regression_rejected_not_sorted(self):
        """Sorting a damaged source into a convenient order is prohibited."""
        rows = deepcopy(self.rows)
        rows[20]['available_mono_ns'] = rows[19]['available_mono_ns']-1
        with self.assertRaisesRegex(ValueError, 'regressed'):
            as_sources(rows)

    def test_missing_producer_times_do_not_fabricate_or_erase_valid_values(self):
        """A missing modal atime blocks clock qualification, but does not erase a finite observer-timed measurement."""
        rows = deepcopy(self.rows)
        for row in rows:
            if row['source'] == 1:
                row['acquired_raw_ns'] = 0
                row['flags'] |= 1
        sources = as_sources(rows)
        self.assertEqual(sum(sources['modal'].valid), 2400)
        self.assertEqual(sources['modal'].timestamp_limitations['nonpositive_raw_timestamp'], 2400)
        a = provisional_associations(sources)
        splits, _ = split_origins(sources['modal'], a, 8, (1, 8), (16,))
        self.assertGreater(len(splits['test']), 32)
        self.assertEqual(sources['modal'].acquired[100], 0)

    def test_four_to_one_counter_pairing_and_effective_raw_centroid(self):
        """Expected groups contain four original ADC identities and preserve integer epoch precision."""
        a = provisional_associations(self.sources)
        k = 40
        self.assertEqual(a['end'][k]-a['start'][k], 3)
        self.assertEqual(a['end'][k], 4*k-1)
        self.assertEqual(a['acquired_centroid_raw_ns'][k], self.sources['adc'].acquired[4*k-4]+187500)
        self.assertTrue(a['report']['counter_stride_four_is_imposed'])
        self.assertFalse(a['report']['physical_synchronization_qualified'])

    def test_missing_counter_preserves_anchor_and_masks_window(self):
        """A removed ADC publication is not renumbered into a new four-sample group."""
        rows = [r for r in self.rows if not (r['source'] == 0 and r['counter'] == 100)]
        a = provisional_associations(as_sources(rows))
        self.assertEqual(a['report']['status_counts']['missing_counter'], 1)
        self.assertEqual(len(a['report']['anchors']), 1)
        self.assertTrue(a['valid'][26])

    def test_late_and_stale_samples_are_not_admitted(self):
        """An early acquisition timestamp does not excuse observer delivery after the analysis cutoff."""
        sources = deepcopy(self.sources)
        # Freeze an early anchor, then deliver a contiguous group late while retaining monotonic receipt order.
        for i in range(80, len(sources['adc'].available)):
            sources['adc'].available[i] += 1500000
        a = provisional_associations(sources, max_age_us=5000)
        self.assertGreater(a['report']['status_counts']['late_delivery'], 0)
        stale = provisional_associations(self.sources, lag_frames=4, max_age_us=1500)
        self.assertGreater(stale['report']['status_counts']['stale_receipt'], 0)
        with self.assertRaises(ValueError):
            provisional_associations(self.sources, max_age_us=float('nan'))
        for row in association_rows(self.sources, provisional_associations(self.sources)):
            if row['status'] == 'valid':
                self.assertLessEqual(row['adc_last_available_mono_ns'], row['modal_available_mono_ns'])

    def test_unknown_raw_domains_do_not_change_provisional_pairing(self):
        """Raw acquisition/publication epoch offsets cannot secretly drive the receipt-only association."""
        sources = deepcopy(self.sources)
        for role in ('modal', 'camera'):
            for column in ('acquired', 'published'):
                for i in range(len(getattr(sources[role], column))):
                    getattr(sources[role], column)[i] += 123000000000
        original, changed = provisional_associations(self.sources), provisional_associations(sources)
        np.testing.assert_array_equal(original['end'], changed['end'])
        np.testing.assert_array_equal(original['values'], changed['values'])

    def test_jitter_and_batched_delivery_keep_only_past_complete_groups(self):
        """Burst delivery and jitter affect provisional grouping without granting a hardware timing certificate."""
        rows = deepcopy(self.rows)
        rng = np.random.default_rng(22)
        for row in rows:
            if row['source'] == 0:
                delta = (-row['available_mono_ns']) % 250000
                row['received_mono_ns'] += delta
                row['available_mono_ns'] += delta
            elif row['source'] == 1:
                delta = int(rng.integers(-10000, 10000))
                row['received_mono_ns'] += delta
                row['available_mono_ns'] += delta
        sources = as_sources(rows)
        a = provisional_associations(sources)
        self.assertGreater(a['report']['valid_fraction'], .99)
        for k in np.flatnonzero(a['valid']):
            self.assertLessEqual(sources['adc'].available[a['end'][k]], sources['modal'].available[k])

    def test_source_restarts_and_wfs_loss_do_not_bridge_histories(self):
        """Generation changes and missing WFS counters purge dependent windows while allowing later recovery."""
        rows = deepcopy(self.rows)
        for row in rows:
            threshold = 1201 if row['source'] else 4801
            if row['counter'] >= threshold:
                row['generation'] += 1
                row['counter'] -= threshold-1
        rows = [r for r in rows if not (r['source'] == 1 and r['generation'] == 1 and r['counter'] == 100)]
        sources = as_sources(rows)
        a = provisional_associations(sources)
        splits, _ = split_origins(sources['modal'], a, 8, (1, 8), (16,))
        runs = contiguous_runs(sources['modal'])
        for origins in splits.values():
            np.testing.assert_array_equal(runs[origins-24], runs[origins+8])
        self.assertGreater(len(a['report']['anchors']), 1)

    def test_purged_splits_features_and_targets_never_overlap(self):
        """Every feature and future label lies wholly inside its assigned chronological partition."""
        a = provisional_associations(self.sources)
        splits, report = split_origins(self.sources['modal'], a, 16, (1, 2, 4, 8), (32,))
        for name, origins in splits.items():
            lo, hi = report[name]['raw_bounds']
            self.assertGreaterEqual(origins[0]-48, lo)
            self.assertLess(origins[-1]+8, hi)
        self.assertLess(splits['train'][-1]+8, splits['validation'][0]-48)
        data = np.column_stack((np.arange(2400), -np.arange(2400)))
        x = features(data, data*10, np.array([100]), 3, 'combined', lag=2)
        np.testing.assert_array_equal(x, [[100, -100, 99, -99, 98, -98, 980, -980, 970, -970, 960, -960]])

    def test_future_adc_missingness_does_not_select_current_origin(self):
        """Eligibility can depend on future WFS labels, but never on future sensor validity or features."""
        a = provisional_associations(self.sources)
        splits, _ = split_origins(self.sources['modal'], a, 8, (8,), ())
        origin = int(splits['test'][20])
        changed = deepcopy(a)
        changed['valid'][origin+1] = False
        after, _ = split_origins(self.sources['modal'], changed, 8, (8,), ())
        self.assertIn(origin, after['test'])

    def test_correlated_sensor_recovers_known_heldout_improvement(self):
        """Causally leading sensor innovations improve both axes at all four requested horizons."""
        sources = as_sources(synthetic_records(frames=6000))
        r = experiment(sources, history=16, control_lags=(32,), repeats=2)
        gains = [row['mse_improvement_vs_wfs'] for row in r['metrics'] if row['model'] == 'combined']
        self.assertEqual(len(gains), 8)
        self.assertGreater(min(gains), .70)
        for key in ('native_training_allowed', 'dm_output_allowed', 'wfs_deadline_qualified', 'synco_deadline_qualified'):
            self.assertFalse(r['report'][key])
        self.assertFalse(r['report']['native_ddspc_cpp_exercised'])

    def test_independent_sensor_has_no_large_reproducible_gain(self):
        """The seeded uncoupled negative control cannot masquerade as the strongly predictive positive control."""
        sources = as_sources(synthetic_records(frames=6000, coupled=False, seed=917))
        r = experiment(sources, history=8, control_lags=(32,), repeats=2)
        gains = [row['mse_improvement_vs_wfs'] for row in r['metrics'] if row['model'] == 'combined']
        self.assertLess(max(gains), .05)
        self.assertEqual(len(r['report']['negative_control']), 8)

    def test_test_targets_cannot_change_fit_scaling_or_validation_selection(self):
        """Changing held-out future labels cannot alter the fitted model, training means or chosen ridge."""
        a = provisional_associations(self.sources)
        splits, report = split_origins(self.sources['modal'], a, 8, (1, 8), (16,))
        wfs = np.column_stack(self.sources['modal'].values)
        before = fit_eof(wfs, a['values'], splits, 8, (1, 8), 'combined')
        changed = wfs.copy()
        changed[report['test']['raw_bounds'][0]:] *= 10000
        after = fit_eof(changed, a['values'], splits, 8, (1, 8), 'combined')
        for key in ('mean', 'scale', 'target_mean', 'coefficient'):
            np.testing.assert_array_equal(before[key], after[key])
        self.assertEqual(before['diagnostics']['chosen_ridge_fraction'], after['diagnostics']['chosen_ridge_fraction'])

    def test_standard_fit_is_unchanged_by_adc_values(self):
        """On the same common rows, accelerometer values never enter the WFS-only fit."""
        a = provisional_associations(self.sources)
        splits, _ = split_origins(self.sources['modal'], a, 8, (1, 8), (16,))
        wfs = np.column_stack(self.sources['modal'].values)
        first = fit_eof(wfs, a['values'], splits, 8, (1, 8), 'wfs')
        second = fit_eof(wfs, np.zeros_like(a['values']), splits, 8, (1, 8), 'wfs')
        np.testing.assert_array_equal(first['coefficient'], second['coefficient'])

    def test_frozen_combined_forecast_consumes_adc_standard_does_not(self):
        """Perturb only held-out sensor features: the combined forecast changes and WFS-only remains identical."""
        a = provisional_associations(self.sources)
        splits, _ = split_origins(self.sources['modal'], a, 8, (1, 8), (16,))
        wfs = np.column_stack(self.sources['modal'].values)
        for kind in ('wfs', 'combined'):
            model = fit_eof(wfs, a['values'], splits, 8, (1, 8), kind)
            original = predict(model, wfs, a['values'], splits['test'], 8)
            changed = predict(model, wfs, np.zeros_like(a['values']), splits['test'], 8)
            if kind == 'wfs':
                np.testing.assert_array_equal(original, changed)
                self.assertEqual(model['diagnostics']['extra_weight_norm'], 0)
            else:
                self.assertGreater(np.linalg.norm(original-changed), 1)
                self.assertGreater(model['diagnostics']['extra_weight_norm'], 0)

    def test_independent_controls_use_training_statistics_only(self):
        """A changed test ADC realization cannot alter the independently generated control."""
        values = np.random.default_rng(90).normal(size=(1000, 2))
        train = np.arange(500)
        first = independent_sensor(values, train, 12)
        values[500:] += 1e6
        np.testing.assert_array_equal(first, independent_sensor(values, train, 12))

    def test_rank_deficient_and_constant_features_remain_finite(self):
        """EOF truncation and positive ridge recover finite forecasts with duplicated or zero input columns."""
        a = provisional_associations(self.sources)
        splits, _ = split_origins(self.sources['modal'], a, 8, (1, 8), (16,))
        wfs = np.column_stack(self.sources['modal'].values)
        model = fit_eof(wfs, np.zeros_like(wfs), splits, 8, (1, 8), 'adc')
        self.assertEqual(model['diagnostics']['retained_eof_rank'], 0)
        self.assertTrue(np.isfinite(predict(model, wfs, np.zeros_like(wfs), splits['test'], 8)).all())

    def test_covariance_eof_matches_direct_svd_ridge_solution(self):
        """Independently compare the batch covariance fit with the thin-SVD ridge formula on the same training matrix."""
        a = provisional_associations(self.sources)
        splits, _ = split_origins(self.sources['modal'], a, 8, (1, 8), (16,))
        wfs = np.column_stack(self.sources['modal'].values)
        model = fit_eof(wfs, a['values'], splits, 8, (1, 8), 'combined', ridge_grid=(.01,))
        x = (features(wfs, a['values'], splits['train'], 8, 'combined')-model['mean'])/model['scale']
        y = target_values(wfs, splits['train'], (1, 8))-model['target_mean']
        u, singular, vt = np.linalg.svd(x, full_matrices=False)
        expected = vt.T@((singular/(singular**2+.01*singular[0]**2))[:, None]*(u.T@y))
        np.testing.assert_allclose(model['coefficient'], expected, rtol=1e-9, atol=1e-10)

    def test_welch_preserves_variance_and_never_joins_gaps(self):
        """A known sinusoid has half-unit power, and separated short spans remain separate Welch windows."""
        t = np.arange(8192)/2000
        x = np.sin(2*np.pi*56*t)[:, None]
        f, p, _ = segmented_psd(x, np.arange(len(x)))
        self.assertAlmostEqual(float(p.sum()*(f[1]-f[0])), .5, delta=.001)
        _, _, stats = segmented_psd(x[:128], np.r_[np.arange(64), np.arange(1000, 1064)], nperseg=64)
        self.assertEqual(stats['welch_windows'], 2)

    def test_cli_artifacts_plots_readonly_and_output_collision(self):
        """The public CLI produces reproducible offline tables/plots and never overwrites inputs or an existing run."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'input'
            source.mkdir()
            write_measurements(source, self.rows)
            before = hashlib.sha256((source/'observations.csv').read_bytes()).hexdigest()
            command = [sys.executable, str(Path(__file__).with_name('offline_eof.py')), str(source),
                       '--output', str(root/'output'), '--data-kind', 'synthetic-correlated', '--minimum-seconds', '0',
                       '--history', '8', '--control-lags', '16', '--control-repeats', '1', '--nperseg', '128']
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(root/'mpl'))
            result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root/'output/report.json').read_text())
            self.assertTrue(report['pipeline_execution_pass'])
            self.assertFalse(report['dm_output_allowed'])
            self.assertEqual(before, hashlib.sha256((source/'observations.csv').read_bytes()).hexdigest())
            for name in ('forecast_error_psd.png', 'negative_controls.png', 'input_diagnostics.png', 'prediction_rmse.png'):
                self.assertEqual((root/'output'/name).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            self.assertNotEqual(subprocess.run(command, capture_output=True, env=env).returncode, 0)
            command[command.index('--output')+1] = str(source/'new-analysis')
            self.assertNotEqual(subprocess.run(command, capture_output=True, env=env).returncode, 0)
            self.assertFalse((source/'new-analysis').exists())
            command[command.index('--output')+1] = str(root/'insufficient')
            command[command.index('--history')+1] = '2000'
            self.assertNotEqual(subprocess.run(command, capture_output=True, env=env).returncode, 0)
            failure = json.loads((root/'insufficient/failure.json').read_text())
            self.assertFalse(failure['pipeline_execution_pass'])
            self.assertTrue((root/'insufficient/association_quality.json').exists())

    def test_analysis_imports_no_runtime_transport_or_native_controller(self):
        """Audit the tool's imports; ordinary-file numerical code grants no control authority."""
        tree = ast.parse(Path(__file__).with_name('offline_eof.py').read_text())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(item.name.split('.')[0] for item in node.names)
            if isinstance(node, ast.ImportFrom):
                names.add((node.module or '').split('.')[0])
        self.assertFalse(names & {'ctypes', 'socket', 'subprocess', 'ImageStreamIO', 'pyMilk', 'simulation_demo'})


if __name__ == '__main__':
    unittest.main()
