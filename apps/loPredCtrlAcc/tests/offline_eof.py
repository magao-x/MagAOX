r"""\file offline_eof.py
\brief Ordinary-file native-journal EOF prediction and uncoupled-sensor controls; no runtime transport.
\ingroup loPredCtrlAcc_unit_test
"""
import argparse
from array import array
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import time

import numpy as np
import scipy
from scipy import linalg, signal
from threadpoolctl import threadpool_info, threadpool_limits

from native_timing_audit import ROLES, Series, cadence, distribution
from verify_native_observation import verify

FS = 2000
STATUSES = ("valid", "no_adc", "warmup", "missing_counter", "late_delivery", "stale_receipt", "invalid_sample")
FIELDS = ("source", "event", "flags", "device", "file_inode", "image_inode", "generation", "segment",
          "counter", "counter1", "has_counter1", "acquired_raw_ns", "published_raw_ns", "received_mono_ns",
          "available_mono_ns", "width", "height", "depth", "datatype", "elements", "value0", "value1", "raw0", "raw1")


class Samples(Series):
    """Extend the existing integer timing columns with owned recorded values and validity; no ADC recalibration."""

    def __init__(self):
        """Allocate compact journal columns and reason counters."""
        super().__init__()
        self.values, self.raw = [array("d"), array("d")], [array("q"), array("q")]
        self.valid = array("B")
        self.rejections = Counter()
        self.timestamp_limitations = Counter()

    def append_sample(self, record, role):
        """Preserve original timing/identity and mark malformed payloads without filling or deleting rows."""
        reasons = []
        width, height, depth, dtype, elements = (int(record[k]) for k in
                                                ("width", "height", "depth", "datatype", "elements"))
        if min(width, height, depth) <= 0 or elements != width*height:
            reasons.append("dimensions")
        if role == "adc" and (width != 2 or height != 1 or dtype != 3):
            reasons.append("adc_layout_not_uint16_pair")
        if role == "modal" and (height != 1 or width < 2 or dtype != 9):
            reasons.append("modal_layout_not_float_vector")
        if role == "camera" and dtype not in (1, 3):
            reasons.append("camera_datatype")
        values = [float(record[f"value{j}"]) for j in range(2)]
        raw = [int(record[f"raw{j}"]) for j in range(2)]
        if not np.isfinite(values).all():
            reasons.append("nonfinite_values")
        if any(v < 0 or v > 65535 for v in raw):
            reasons.append("raw_not_uint16")
        if int(record["flags"]) & (8 | 16):
            reasons.append("invalid_native_payload_flags")
        if int(record["counter"]) <= 0:
            reasons.append("invalid_counter")
        # This experiment uses observer availability, so absent/unqualified producer clocks do not invent or erase payloads.
        if min(int(record[k]) for k in ("acquired_raw_ns", "published_raw_ns")) <= 0:
            self.timestamp_limitations["nonpositive_raw_timestamp"] += 1
        for flag, name in ((1, "native_source_time_flag"), (4, "native_clock_mismatch_flag"), (2, "native_unknown_clock_flag")):
            if int(record["flags"]) & flag:
                self.timestamp_limitations[name] += 1
        received, available = (int(record[k]) for k in ("received_mono_ns", "available_mono_ns"))
        if received < 0 or available < received:
            reasons.append("invalid_observer_times")
        if self.available and available < self.available[-1]:
            raise ValueError(f"{role}: observer availability regressed; refusing to sort/relabel records")
        super().append(record)
        if len(self.counter) > 1 and self.epoch[-1] == self.epoch[-2] and self.counter[-1] <= self.counter[-2]:
            reasons.append("nonincreasing_counter")
        for axis in range(2):
            self.values[axis].append(values[axis])
            self.raw[axis].append(raw[axis])
        self.valid.append(not reasons)
        self.rejections.update(reasons)


def ordinary_file(path):
    """Require an existing regular artifact, never a FIFO, device or runtime shared-memory path."""
    path = Path(path).resolve(strict=True)
    if any(path == root or root in path.parents for root in map(Path, ("/milk/shm", "/dev/shm", "/proc", "/sys"))):
        raise ValueError("Select copied ordinary observation artifacts, not runtime streams")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Not an ordinary file: {path}")
    return path


def digest(path):
    """Hash immutable inputs in bounded chunks for reproducible run binding."""
    result = hashlib.sha256()
    with ordinary_file(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def load_recording(directory, minimum_seconds=59.5):
    """Load selected native columns, preserving raw clocks and uint64 counters; reuse the scoped transport verifier."""
    directory = Path(directory).resolve(strict=True)
    summary = json.loads(ordinary_file(directory/"summary.json").read_text())
    if summary.get("schema") != 1 or summary.get("observation_only") is not True:
        raise ValueError("Only schema-1 completed observation journals are supported")
    for field in ("training_enabled", "exposure_qualified", "physical_application_qualified"):
        if summary.get(field) is not False:
            raise ValueError(f"Unexpected observation qualification: {field}")
    indices = summary.get("modal_indices")
    if indices is None or len(indices) != 2 or indices[0] == indices[1] or min(indices) < 0:
        raise ValueError("Summary must identify the two recorded modal indices; they cannot be reconstructed afterward")
    sources = {role: Samples() for role in ROLES[:3]}
    rows = 0
    with ordinary_file(directory/"observations.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS) <= set(reader.fieldnames or []):
            raise ValueError("Missing native value/geometry/timing columns; timing-only fixtures are not measurement data")
        for row in reader:
            rows += 1
            slot = int(row["source"])
            if not 0 <= slot < len(ROLES):
                raise ValueError("Unknown source role")
            if int(row["event"]) == 0 and slot < 3:
                if slot == 1 and max(indices) >= int(row["elements"]):
                    raise ValueError("Recorded modal indices exceed the modal stream extent")
                sources[ROLES[slot]].append_sample(row, ROLES[slot])
    transport = verify(directory, minimum_seconds=minimum_seconds, profile="adc-wfs")
    diagnostics = {"rows": rows, "modal_indices": indices, "transport": transport, "sources": {}}
    for role, source in sources.items():
        if len(source.counter) < 2:
            raise ValueError(f"{role}: insufficient samples")
        elapsed = source.available[-1]-source.available[0]
        diagnostics["sources"][role] = dict(cadence(source), valid_samples=sum(source.valid),
            invalid_reasons=dict(source.rejections), timestamp_limitations=dict(source.timestamp_limitations),
            observer_rate_hz=(len(source.counter)-1)*1e9/elapsed if elapsed > 0 else None,
            acquisition_clock=summary["sources"][ROLES.index(role)].get("acquisition_clock", "unknown"),
            publication_clock=summary["sources"][ROLES.index(role)].get("publication_clock", "unknown"))
        good = np.column_stack(source.values)[np.asarray(source.valid, dtype=bool)]
        diagnostics["sources"][role]["recorded_value_mean"] = good.mean(axis=0).tolist() if len(good) else None
        diagnostics["sources"][role]["recorded_value_std"] = good.std(axis=0).tolist() if len(good) else None
        if role == "adc":
            diagnostics["sources"][role]["raw_min"] = [min(column) for column in source.raw]
            diagnostics["sources"][role]["raw_max"] = [max(column) for column in source.raw]
    return sources, diagnostics


def provisional_associations(sources, lag_frames=1, max_age_us=1500):
    """Freeze 4:1 counter anchors from past observer availability; this is neither exposure nor physical synchronization."""
    if lag_frames < 0 or not np.isfinite(max_age_us) or max_age_us <= 0:
        raise ValueError("Invalid provisional lag/receipt-age bound")
    adc, modal, camera = (sources[role] for role in ("adc", "modal", "camera"))
    deadlines = np.asarray(modal.available)
    arrival = np.asarray(adc.available)
    latest = np.searchsorted(arrival, deadlines, side="right")-1
    camera_index = np.searchsorted(camera.available, deadlines, side="right")-1
    previous = np.r_[deadlines[0]-500000, deadlines[:-1]]
    receipt_counts = np.searchsorted(arrival, deadlines, side="right")-np.searchsorted(arrival, previous, side="right")
    lookup = {(epoch, counter): i for i, (epoch, counter) in enumerate(zip(adc.epoch, adc.counter))}
    values = np.full((len(deadlines), 2), np.nan)
    status = np.ones(len(deadlines), dtype=np.int8)
    start, end, group_epoch = (np.full(len(deadlines), -1, dtype=np.int64) for _ in range(3))
    acquired_centroid = np.full(len(deadlines), -1, dtype=np.int64)
    receipt_age = np.full(len(deadlines), np.nan)
    anchors = {}
    recorded = np.column_stack(adc.values)
    for k, last in enumerate(latest):
        if last < 0:
            continue
        key = int(adc.epoch[last]), int(modal.epoch[k])
        if key not in anchors:
            anchors[key] = int(adc.counter[last]), int(modal.counter[k])
        a0, k0 = anchors[key]
        target = a0+4*(int(modal.counter[k])-k0-lag_frames)
        if target < 4:
            status[k] = 2
            continue
        indexes = [lookup.get((key[0], n)) for n in range(target-3, target+1)]
        if any(i is None for i in indexes):
            status[k] = 3
            continue
        start[k], end[k], group_epoch[k] = indexes[0], indexes[-1], key[0]
        if not all(adc.valid[i] for i in indexes):
            status[k] = 6
            continue
        if any(adc.available[i] > deadlines[k] for i in indexes):
            status[k] = 4
            continue
        receipt_age[k] = (int(deadlines[k])-max(adc.available[i] for i in indexes))/1000
        if receipt_age[k] > max_age_us:
            status[k] = 5
            continue
        values[k] = recorded[indexes].mean(axis=0)
        # Center in the source's original clock; never convert epoch ns through floating point.
        if all(adc.acquired[i] > 0 for i in indexes):
            base = adc.acquired[indexes[0]]
            acquired_centroid[k] = base+sum(adc.acquired[i]-base for i in indexes)//4
        status[k] = 0
    timing = np.column_stack(((deadlines-previous)/1000-500, receipt_age))
    report = {"reference": "modal_observer_availability_only", "physical_synchronization_qualified": False,
              "counter_stride_four_is_imposed": True, "adc_lag_frames": lag_frames,
              "maximum_receipt_age_us": max_age_us, "status_counts": {name: int(sum(status == i)) for i, name in enumerate(STATUSES)},
              "valid_fraction": float(np.mean(status == 0)),
              "receipt_window_counts": {str(n): int(count) for n, count in Counter(map(int, receipt_counts)).items()},
              "valid_receipt_age": distribution((receipt_age[status == 0]*1000).tolist()),
              "anchors": [{"adc_epoch": a, "modal_epoch": m, "adc_counter": n, "modal_counter": k}
                          for (a, m), (n, k) in anchors.items()],
              "camera_receipt_candidates": int(np.sum(camera_index >= 0)), "modal_camera_origin_qualified": False}
    return dict(values=values, valid=status == 0, status=status, start=start, end=end, epoch=group_epoch,
                acquired_centroid_raw_ns=acquired_centroid, age_us=receipt_age, timing=timing,
                camera_index=camera_index, receipt_counts=receipt_counts, report=report)


def contiguous_runs(source):
    """Assign analysis runs without joining invalid rows, source restarts, counter gaps or long receipt gaps."""
    c, e, t, valid = map(np.asarray, (source.counter, source.epoch, source.available, source.valid))
    breaks = np.ones(len(c), dtype=bool)
    breaks[1:] = ((e[1:] != e[:-1]) | (c[1:] != c[:-1]+np.uint64(1)) | ~valid[1:].astype(bool) |
                  ~valid[:-1].astype(bool) | (np.diff(t) > 1000000))
    return np.cumsum(breaks)


def split_origins(modal, association, history, horizons, control_lags=(1000, 2000), fractions=(.6, .2)):
    """Purge feature/target footprints at chronological split boundaries; missing future ADC never selects an origin."""
    if history < 1 or not horizons or min(horizons) < 1 or len(set(horizons)) != len(horizons):
        raise ValueError("History and distinct positive horizons are required")
    if len(fractions) != 2 or min(fractions) <= 0 or sum(fractions) >= 1 or any(lag < 1 for lag in control_lags):
        raise ValueError("Invalid split fractions or delayed-control lags")
    n, future = len(modal.counter), max(horizons)
    past = history+max((0, *control_lags)) # The timing control also reads the preceding receipt interval.
    bounds = [0, int(n*fractions[0]), int(n*sum(fractions)), n]
    runs, usable = contiguous_runs(modal), np.asarray(modal.valid, dtype=bool)
    bad = np.r_[0, np.cumsum(~association["valid"])]
    result, diagnostics = {}, {}
    for name, lo, hi in zip(("train", "validation", "test"), bounds[:-1], bounds[1:]):
        origin = np.arange(lo+past, hi-future, dtype=np.int64)
        if not len(origin):
            raise ValueError(f"{name}: too short for the purged history/horizons/control lags")
        keep = usable[origin-past] & usable[origin+future] & (runs[origin-past] == runs[origin+future])
        for lag in (0, *control_lags):
            first, last = origin-lag-history+1, origin-lag
            keep &= (bad[last+1] == bad[first]) & (association["epoch"][first] == association["epoch"][last])
        selected = origin[keep]
        if len(selected) < 32:
            raise ValueError(f"{name}: fewer than 32 common complete origins; inspect dimensions, continuity and association diagnostics")
        result[name] = selected
        diagnostics[name] = dict(raw_bounds=[lo, hi], candidates=len(origin), accepted=len(selected),
                                 rejected=int(sum(~keep)), first_feature_row=int(selected[0]-past),
                                 last_target_row=int(selected[-1]+future))
    return result, diagnostics


def features(wfs, extra, origins, history, kind, lag=0):
    """Build newest-first histories at or before each origin; no future target or command is an input."""
    indexes = origins[:, None]-np.arange(history)
    parts = []
    if kind != "adc":
        parts.append(wfs[indexes].reshape(len(origins), -1))
    if kind != "wfs":
        parts.append(extra[indexes-lag].reshape(len(origins), -1))
    return np.concatenate(parts, axis=1)


def target_values(wfs, origins, horizons):
    """Keep future WFS labels separate from every predictor feature builder."""
    return np.concatenate([wfs[origins+h] for h in horizons], axis=1)


def batches(indexes, size):
    """Yield bounded origin batches without changing chronological order."""
    for start in range(0, len(indexes), size):
        yield indexes[start:start+size]


def fit_eof(wfs, extra, splits, history, horizons, kind, ridge_grid=(1e-6, 1e-4, .01, 1), lag=0, batch_size=4096):
    """Fit training-only EOF covariance modes; select ridge per horizon/axis on validation, then freeze."""
    started = time.perf_counter()
    if not ridge_grid or any(not np.isfinite(a) or a <= 0 for a in ridge_grid) or batch_size < 1:
        raise ValueError("Use finite positive ridge fractions and a bounded batch size")
    train, validation = splits["train"], splits["validation"]
    xsum, ysum = None, np.zeros(2*len(horizons))
    for origin in batches(train, batch_size):
        x = features(wfs, extra, origin, history, kind, lag)
        xsum = x.sum(axis=0) if xsum is None else xsum+x.sum(axis=0)
        ysum += target_values(wfs, origin, horizons).sum(axis=0)
    xmean, ymean = xsum/len(train), ysum/len(train)
    covariance, cross = np.zeros((len(xmean), len(xmean))), np.zeros((len(xmean), len(ymean)))
    for origin in batches(train, batch_size):
        x = features(wfs, extra, origin, history, kind, lag)-xmean
        y = target_values(wfs, origin, horizons)-ymean
        covariance += x.T@x
        cross += x.T@y
    covariance /= len(train)
    scale = np.sqrt(np.maximum(0, covariance.diagonal()))
    scale[scale < 1e-12] = 1
    covariance /= scale[:, None]*scale[None, :]
    cross /= len(train)*scale[:, None]
    eigenvalues, vectors = linalg.eigh(covariance, check_finite=True)
    largest = max(0., float(eigenvalues[-1]))
    if eigenvalues[0] < -1e-9*max(1, largest):
        raise ValueError("Training covariance has significant negative eigenvalues")
    keep = eigenvalues > max(1e-12, np.finfo(float).eps*len(xmean))*largest
    eigenvalues, vectors = eigenvalues[keep], vectors[:, keep]
    coefficients = [(vectors@((vectors.T@cross)/(eigenvalues[:, None]+alpha*largest)))
                    for alpha in ridge_grid]
    fitted_seconds = time.perf_counter()-started
    started = time.perf_counter()
    losses = np.zeros((len(ridge_grid), len(ymean)))
    for origin in batches(validation, batch_size):
        x = (features(wfs, extra, origin, history, kind, lag)-xmean)/scale
        truth = target_values(wfs, origin, horizons)
        for j, coefficient in enumerate(coefficients):
            losses[j] += np.square(x@coefficient+ymean-truth).sum(axis=0)
    selected = losses.argmin(axis=0)
    coefficient = np.column_stack([coefficients[j][:, column] for column, j in enumerate(selected)])
    wfs_columns = 0 if kind == "adc" else 2*history
    diagnostics = dict(features=len(xmean), train_rows=len(train), validation_rows=len(validation),
                       retained_eof_rank=int(sum(keep)), retained_singular_condition=float(np.sqrt(eigenvalues[-1]/eigenvalues[0])) if len(eigenvalues) else None,
                       chosen_ridge_fraction=[float(ridge_grid[j]) for j in selected],
                       validation_mse=(losses[selected, np.arange(len(ymean))]/len(validation)).tolist(),
                       coefficient_norm=float(np.linalg.norm(coefficient)), fit_seconds=fitted_seconds,
                       wfs_weight_norm=float(np.linalg.norm(coefficient[:wfs_columns])),
                       extra_weight_norm=float(np.linalg.norm(coefficient[wfs_columns:])),
                       validation_selection_seconds=time.perf_counter()-started,
                       solver="training standardized covariance eigendecomposition; ridge with alpha=lambda_fraction*largest_eigenvalue")
    return dict(mean=xmean, scale=scale, target_mean=ymean, coefficient=coefficient, kind=kind, lag=lag,
                diagnostics=diagnostics)


def predict(model, wfs, extra, origins, history, batch_size=4096):
    """Apply frozen EOF weights in batches; this wall time is not real-time deadline qualification."""
    return np.concatenate([((features(wfs, extra, origin, history, model["kind"], model["lag"])-model["mean"])/model["scale"])@
                           model["coefficient"]+model["target_mean"] for origin in batches(origins, batch_size)])


def independent_sensor(values, train, seed):
    """Generate independent colored Gaussian controls using only training ADC mean, variance and lag-one correlation."""
    rng = np.random.default_rng(seed)
    sample = values[train]
    center, scale = sample.mean(axis=0), sample.std(axis=0)
    consecutive = np.diff(train) == 1
    left, right = sample[:-1][consecutive]-center, sample[1:][consecutive]-center
    rho = np.clip(np.sum(left*right, axis=0)/np.maximum(np.sum(left*left, axis=0), 1e-30), -.98, .98)
    noise = rng.normal(size=values.shape)
    for axis in range(2):
        noise[:, axis] = signal.lfilter([np.sqrt(1-rho[axis]**2)], [1, -rho[axis]], noise[:, axis])
    return noise*scale+center


def segmented_psd(values, origins, nperseg=2048):
    """Average Hann Welch windows only inside contiguous test runs; never concatenate across missing samples."""
    chunks = np.split(np.arange(len(origins)), np.flatnonzero(np.diff(origins) != 1)+1)
    size = min(nperseg, max(map(len, chunks)))
    if size < 32:
        return None, None, {"welch_windows": 0, "reason": "no contiguous test span of 32 samples"}
    overlap, total, power, frequency = size//2, 0, None, None
    for chunk in chunks:
        if len(chunk) < size:
            continue
        frequency, current = signal.welch(values[chunk], fs=FS, nperseg=size, noverlap=overlap,
                                          window="hann", detrend="constant", axis=0)
        count = 1+(len(chunk)-size)//(size-overlap)
        power = current*count if power is None else power+current*count
        total += count
    return frequency, power/total, dict(welch_windows=total, nperseg=size, nominal_fs_hz=FS,
                                       reference="modal frame lattice, not qualified exposure timestamps")


def evaluate(truth, predictions, origins, horizons, nperseg=2048):
    """Score per-axis forecast errors on exactly the same held-out targets; report descriptive negative controls."""
    metrics, psds, blocks = [], {}, []
    frequency, measured, spectral = segmented_psd(truth, origins, nperseg)
    if frequency is not None:
        psds["measured_wfs"] = measured
    reference = np.square(predictions["wfs"]-truth)
    for name, prediction in predictions.items():
        error = prediction-truth
        _, psd, _ = segmented_psd(error, origins, nperseg)
        if psd is not None:
            psds[name] = psd
        for h, horizon in enumerate(horizons):
            for axis, label in enumerate(("tip", "tilt")):
                column = 2*h+axis
                mse, baseline = float(np.mean(error[:, column]**2)), float(np.mean(reference[:, column]))
                row = dict(model=name, horizon_frames=horizon, axis=label, samples=len(origins), mse=mse,
                           rmse=float(np.sqrt(mse)), error_variance=float(np.var(error[:, column])),
                           error_bias=float(np.mean(error[:, column])),
                           mse_improvement_vs_wfs=1-mse/baseline if baseline else None)
                for low, high, tag in ((27, 30, "28_29"), (54, 58, "56")):
                    band = (frequency >= low) & (frequency <= high) if frequency is not None else None
                    row[f"error_bandpower_{tag}"] = float(psd[band, column].sum()*(frequency[1]-frequency[0])) if band is not None and band.any() else None
                metrics.append(row)
                for block, indexes in enumerate(np.array_split(np.arange(len(origins)), 4)):
                    base = float(reference[indexes, column].mean())
                    blocks.append(dict(model=name, horizon_frames=horizon, axis=label, block=block,
                                       samples=len(indexes), mse_improvement_vs_wfs=1-float(np.mean(error[indexes, column]**2))/base if base else None))
    negative = []
    for row in [r for r in metrics if r["model"] == "combined"]:
        controls = [r["mse_improvement_vs_wfs"] for r in metrics if r["model"].startswith("independent_") and
                    r["horizon_frames"] == row["horizon_frames"] and r["axis"] == row["axis"]]
        gain = row["mse_improvement_vs_wfs"]
        negative.append(dict(horizon_frames=row["horizon_frames"], axis=row["axis"], observed_mse_improvement=gain,
                             independent_controls=controls,
                             descriptive_exceedance_fraction_not_pvalue=(1+sum(v >= gain for v in controls))/(1+len(controls)) if controls and gain is not None else None))
    return metrics, blocks, negative, frequency, psds, spectral


def experiment(sources, history=100, horizons=(1, 2, 4, 8), lag_frames=1, max_age_us=1500,
               control_lags=(1000, 2000), repeats=5, seed=101, ridge_grid=(1e-6, 1e-4, .01, 1),
               fractions=(.6, .2), batch_size=4096, nperseg=2048, association=None):
    """Run a frozen, purged offline EOF experiment; never construct qualified native DDSPC inputs or commands."""
    if repeats < 1:
        raise ValueError("At least one independent-sensor negative control is required")
    started = time.perf_counter()
    if association is None:
        association = provisional_associations(sources, lag_frames, max_age_us)
    splits, split_report = split_origins(sources["modal"], association, history, horizons, control_lags, fractions)
    wfs = np.column_stack(sources["modal"].values)
    adc = association["values"]
    specs = [("wfs", "wfs", adc, 0), ("adc", "adc", adc, 0), ("combined", "combined", adc, 0),
             ("wfs_timing", "combined", association["timing"], 0)]
    specs.extend((f"delayed_{lag}", "combined", adc, lag) for lag in control_lags)
    # Controls are generated one at a time to keep memory independent of the repetition count.
    specs.extend((f"independent_{j}", "combined", None, 0) for j in range(repeats))
    truth = target_values(wfs, splits["test"], horizons)
    predictions = {"persistence": np.tile(wfs[splits["test"]], (1, len(horizons)))}
    models = {}
    for j, (name, kind, extra, lag) in enumerate(specs):
        if extra is None:
            extra = independent_sensor(adc, splits["train"], seed+j)
        model = fit_eof(wfs, extra, splits, history, horizons, kind, ridge_grid, lag, batch_size)
        begin = time.perf_counter()
        predictions[name] = predict(model, wfs, extra, splits["test"], history, batch_size)
        model["diagnostics"]["test_prediction_seconds"] = time.perf_counter()-begin
        if not np.isfinite(predictions[name]).all():
            raise ValueError(f"{name}: nonfinite prediction")
        models[name] = model
    metrics, blocks, negative, frequency, psds, spectral = evaluate(truth, predictions, splits["test"], horizons, nperseg)
    report = {"history_frames": history, "history_ms_at_nominal_rate": history/2,
              "horizons_frames": list(horizons), "split_fractions": list(fractions), "splits": split_report,
              "ridge_grid": list(ridge_grid), "control_lags_frames": list(control_lags), "seed": seed,
              "association": association["report"], "models": {name: m["diagnostics"] for name, m in models.items()},
              "negative_control": negative, "spectral": spectral,
              "offline_fit_and_evaluation_seconds": time.perf_counter()-started,
              "pipeline_execution_pass": True, "native_ddspc_cpp_exercised": False,
              "physical_control_performance_demonstrated": False,
              "native_training_allowed": False, "dm_output_allowed": False,
              "exposure_qualified": False, "physical_synchronization_qualified": False,
              "wfs_deadline_qualified": False, "synco_deadline_qualified": False,
              "negative_control_interpretation": "No physical accelSim-to-WFS coupling in the operator-described recording. Positive gains are exploratory; inspect independent, delayed, timing controls and chronological blocks. Exceedance fractions are descriptive, not p-values or hardware/control improvement."}
    return dict(report=report, association=association, splits=splits, truth=truth, predictions=predictions,
                models=models, metrics=metrics, blocks=blocks, frequency=frequency, psds=psds)


def write_csv(path, rows):
    """Create one new ordinary result table, refusing an existing file."""
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def association_rows(sources, association):
    """Yield raw timestamp/counter evidence and provisional pair identities without making a clock conversion."""
    adc, modal, camera = (sources[role] for role in ("adc", "modal", "camera"))
    for k in range(len(modal.counter)):
        row = dict(modal_row=k, status=STATUSES[association["status"][k]], modal_epoch=modal.epoch[k])
        for name, source, index in (("modal", modal, k), ("adc_first", adc, association["start"][k]),
                                    ("adc_last", adc, association["end"][k]), ("camera_receipt_candidate", camera, association["camera_index"][k])):
            for label, column in (("counter", "counter"), ("acquired_raw_ns", "acquired"), ("published_raw_ns", "published"),
                                  ("received_mono_ns", "received"), ("available_mono_ns", "available")):
                row[name+"_"+label] = int(getattr(source, column)[index]) if index >= 0 else ""
        row["adc_acquired_centroid_raw_ns"] = int(association["acquired_centroid_raw_ns"][k])
        row["adc_receipt_age_us"] = float(association["age_us"][k]) if np.isfinite(association["age_us"][k]) else ""
        row["adc_in_receipt_window"] = int(association["receipt_counts"][k])
        for axis in range(2):
            row[f"modal_value{axis}"] = modal.values[axis][k] if np.isfinite(modal.values[axis][k]) else ""
            row[f"adc_mean{axis}"] = float(association["values"][k, axis]) if association["valid"][k] else ""
            index = association["end"][k]
            row[f"adc_last_raw{axis}"] = adc.raw[axis][index] if index >= 0 else ""
        yield row


def figures(output, sources, result):
    """Write static scientific plots in a private output directory using a noninteractive backend."""
    os.environ["MPLCONFIGDIR"] = str(output/"matplotlib-cache")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    horizons = result["report"]["horizons_frames"]
    selected = ("wfs", "adc", "combined", "persistence")
    fig, axes = plt.subplots(2, len(horizons), figsize=(4*len(horizons), 7), squeeze=False)
    f = result["frequency"]
    for h, horizon in enumerate(horizons):
        for axis, label in enumerate(("Tip", "Tilt")):
            ax = axes[axis, h]
            if f is not None:
                for name in ("measured_wfs", *selected):
                    ax.semilogy(f, np.maximum(result["psds"][name][:, 2*h+axis], 1e-30), label=name)
            ax.set(xlim=(5, 150), title=f"{label}: horizon {horizon}", xlabel="Frequency [nominal Hz]",
                   ylabel="PSD [modal unit²/Hz]" if h == 0 else "")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Held-out WFS and forecast-error PSDs; offline, no DM corrections")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output/"forecast_error_psd.png", dpi=150)
    fig.savefig(output/"forecast_error_psd.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, label in zip(axes, ("tip", "tilt")):
        for name in selected:
            rows = [r for r in result["metrics"] if r["axis"] == label and r["model"] == name]
            ax.plot([r["horizon_frames"] for r in rows], [r["rmse"] for r in rows], "o-", label=name)
        ax.set(title=label, xlabel="Prediction horizon [WFS frames]", ylabel="Held-out RMSE [recorded modal units]")
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output/"prediction_rmse.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, label in zip(axes, ("tip", "tilt")):
        for name in result["models"]:
            if name in ("wfs", "adc"):
                continue
            rows = [r for r in result["metrics"] if r["axis"] == label and r["model"] == name]
            ax.plot([r["horizon_frames"] for r in rows], [100*r["mse_improvement_vs_wfs"] if r["mse_improvement_vs_wfs"] is not None else np.nan for r in rows],
                    "o-", alpha=.4 if name.startswith("independent") else 1, label=name)
        ax.axhline(0, color="black", lw=.5)
        ax.set(title=label+": sensor-information controls", xlabel="Horizon [frames]", ylabel="MSE improvement vs WFS-only [%]")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output/"negative_controls.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7))
    for ax, role in zip(axes[:2], ("adc", "modal")):
        source = sources[role]
        elapsed = (np.asarray(source.available)-source.available[0])/1e9
        keep = elapsed <= .25
        ax.plot(elapsed[keep], np.column_stack(source.values)[keep])
        ax.set(xlabel="Observer availability since source first receipt [s]", ylabel=role+" recorded values")
    a = result["association"]
    axes[2].hist(a["receipt_counts"], bins=np.arange(max(a["receipt_counts"])+2)-.5)
    axes[2].set(xlabel="ADC records per modal receipt interval (not exposure window)", ylabel="Intervals")
    fig.tight_layout()
    fig.savefig(output/"input_diagnostics.png", dpi=150)
    plt.close(fig)


def main():
    """Read completed local files and write new offline artifacts; no native executable, subprocess or network access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-kind", required=True, choices=("native-uncoupled", "synthetic-correlated", "synthetic-independent"))
    parser.add_argument("--history", type=int, default=100)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--control-lags", type=int, nargs="*", default=[1000, 2000])
    parser.add_argument("--control-repeats", type=int, default=5)
    parser.add_argument("--adc-lag-frames", type=int, default=1)
    parser.add_argument("--max-receipt-age-us", type=float, default=1500)
    parser.add_argument("--ridge-grid", type=float, nargs="+", default=[1e-6, 1e-4, .01, 1])
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--minimum-seconds", type=float, default=59.5)
    parser.add_argument("--nperseg", type=int, default=2048)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    owned_output = False
    try:
        started = time.perf_counter()
        directory, output = args.directory.resolve(strict=True), args.output.resolve()
        if directory == output or directory in output.parents or output in directory.parents:
            raise ValueError("Output must be new and outside the input artifact tree")
        if any(output == root or root in output.parents for root in map(Path, ("/opt/MagAOX", "/milk/shm", "/dev/shm", "/proc", "/sys"))):
            raise ValueError("Output must be a private analysis directory")
        if output.exists() or args.blas_threads < 1 or args.nperseg < 32 or not np.isfinite(args.minimum_seconds) or args.minimum_seconds < 0:
            raise ValueError("Use a new output directory, positive thread count and nperseg >=32")
        hashes = {name: digest(directory/name) for name in ("observations.csv", "summary.json")}
        loaded = time.perf_counter()
        sources, quality = load_recording(directory, args.minimum_seconds)
        loading_seconds = time.perf_counter()-loaded
        association = provisional_associations(sources, args.adc_lag_frames, args.max_receipt_age_us)
        output.mkdir(parents=True, exist_ok=False)
        owned_output = True
        (output/"input_quality.json").write_text(json.dumps(quality, indent=2, allow_nan=False)+"\n")
        (output/"association_quality.json").write_text(json.dumps(association["report"], indent=2, allow_nan=False)+"\n")
        rows = association_rows(sources, association)
        first = next(rows)
        with (output/"associations.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(first))
            writer.writeheader()
            writer.writerow(first)
            writer.writerows(rows)
        with threadpool_limits(limits=args.blas_threads):
            blas_libraries = threadpool_info()
            result = experiment(sources, args.history, tuple(args.horizons), args.adc_lag_frames, args.max_receipt_age_us,
                                tuple(args.control_lags), args.control_repeats, args.seed, tuple(args.ridge_grid),
                                batch_size=args.batch_size, nperseg=args.nperseg, association=association)
        if any(digest(directory/name) != value for name, value in hashes.items()):
            raise ValueError("Input artifacts changed during analysis")
        write_csv(output/"metrics.csv", result["metrics"])
        write_csv(output/"chronological_blocks.csv", result["blocks"])
        if result["frequency"] is not None:
            write_csv(output/"psd.csv", [dict(frequency_hz=float(f), **{
                f"{name}_h{h}_{axis}": float(psd[i, 2*j+k]) for name, psd in result["psds"].items()
                for j, h in enumerate(args.horizons) for k, axis in enumerate(("tip", "tilt"))})
                for i, f in enumerate(result["frequency"])])
        origins = result["splits"]["test"]
        modal = sources["modal"]
        targets = origins[:, None]+np.asarray(args.horizons)
        np.savez_compressed(output/"predictions.npz", origin_rows=origins, origin_counter=np.asarray(modal.counter)[origins],
                            target_counter=np.asarray(modal.counter)[targets], origin_available_mono_ns=np.asarray(modal.available)[origins],
                            target_available_mono_ns=np.asarray(modal.available)[targets],
                            truth=result["truth"], horizons=np.asarray(args.horizons), **result["predictions"])
        arrays = {f"{name}_{key}": model[key] for name, model in result["models"].items()
                  for key in ("mean", "scale", "target_mean", "coefficient")}
        np.savez_compressed(output/"models.npz", **arrays)
        if not args.no_plots:
            figures(output, sources, result)
        report = dict(result["report"], data_kind=args.data_kind, input_sha256=hashes, input_unchanged=True,
                      recording_input_checks_pass=quality["transport"]["transport_pass"] and
                          all(not value["invalid_reasons"] for value in quality["sources"].values()),
                      input_quality=quality, loading_and_transport_verification_seconds=loading_seconds,
                      total_offline_wall_seconds=time.perf_counter()-started,
                      software=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                    platform=platform.platform(), blas_threads=args.blas_threads, blas_libraries=blas_libraries),
                      source_sha256={name: digest(Path(__file__).with_name(name)) for name in
                                     ("offline_eof.py", "native_timing_audit.py", "verify_native_observation.py")},
                      axis_labels="tip/tilt denote recorded columns value0/value1 per experiment request; actual modal indices are retained, physical basis/units unqualified",
                      psd_interpretation="Held-out forecast-error PSD, not physical closed-loop residual after DM correction")
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps(dict(output=str(output), data_kind=args.data_kind, transport_pass=quality["transport"]["transport_pass"],
                              association_valid_fraction=report["association"]["valid_fraction"], splits=report["splits"],
                              native_training_allowed=False, dm_output_allowed=False), indent=2))
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        if owned_output:
            (output/"failure.json").write_text(json.dumps(dict(pipeline_execution_pass=False, error=str(error),
                native_training_allowed=False, dm_output_allowed=False), indent=2)+"\n")
        raise SystemExit(f"Offline EOF analysis failed: {error}") from error


if __name__ == "__main__":
    main()
