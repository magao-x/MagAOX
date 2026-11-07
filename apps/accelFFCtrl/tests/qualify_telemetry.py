#!/usr/bin/env python3
r"""\file qualify_telemetry.py
\brief Offline timestamp-qualified spectra, held-out causal prediction and host-only shadow replay.
\ingroup accelFFCtrl_unit_test

CSV inputs are exported acquisitions, never shared-memory names. Synthetic fixtures
and every calibration/clock assumption travel with a mandatory JSON manifest.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy import signal, stats

from validate_cancellation import coherent

ADC_FIELDS = ("acquired_ns", "written_ns", "receipt_ns", "counter", "generation", "adc_x", "adc_y")
WFS_FIELDS = ("acquired_ns", "written_ns", "counter", "generation", "tip", "tilt")
HORIZONS = (0, 500000, 1000000, 1750000, 3000000, 5000000, 10000000)


def integer_csv(path, fields):
    """Read exact signed integer clocks and unsigned source IDs; refuse missing/nonintegral fields."""
    dtype = {k: ("uint64" if k in ("counter", "generation") else "int64")
             for k in fields if k not in ("tip", "tilt")}
    data = pd.read_csv(path, dtype={k: "string" for k in dtype})
    if data.empty or not set(fields).issubset(data):
        raise ValueError(f"missing required fields in {path}")
    if data[list(fields)].isna().any().any():
        raise ValueError("missing telemetry value")
    for key, kind in dtype.items():
        if not data[key].str.fullmatch(r"[0-9]+").all():
            raise ValueError(f"nonintegral/missing timestamp or identity: {key}")
        data[key] = data[key].astype(kind)
    return data


def timing(data, rate):
    """Reject invalid acquisition/publication times; report counter-based loss and cadence without filling gaps."""
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("invalid declared rate")
    a, w = data.acquired_ns.to_numpy(), data.written_ns.to_numpy()
    c, g = data.counter.to_numpy(), data.generation.to_numpy()
    if np.any(a <= 0) or np.any(w < a) or np.any(c == 0) or np.any(g == 0):
        raise ValueError("invalid acquisition/publication/source identity")
    if np.any(np.diff(a) <= 0):
        raise ValueError("nonmonotonic acquisition: split clock steps/restarts into separate datasets")
    if "receipt_ns" in data and (np.any(data.receipt_ns < w) or np.any(np.diff(data.receipt_ns) < 0)):
        raise ValueError("invalid receipt ordering")
    same = g[1:] == g[:-1]
    # Convert counters before subtraction: unsigned wrap is not a large positive gap.
    dc = np.asarray([int(b)-int(a) for a, b in zip(c[:-1], c[1:])], dtype=np.int64)
    if np.any(same & (dc <= 0)):
        raise ValueError("counter duplicate/rollback: split source runs before optical calibration")
    dt = np.diff(a)
    good = same & (dc > 0)
    errors = dt[good]/dc[good]-1e9/rate
    drift = (np.median(dt[good]/dc[good])*rate/1e9-1)*1e6 if np.any(good) else None
    return dict(records=len(data), missing=int(np.sum(np.maximum(dc[good]-1, 0))),
                generation_changes=int(np.sum(~same)), effective_rate_hz=float(1e9/np.median(dt[good]/dc[good])),
                median_period_drift_ppm=drift, cadence_rms_ns=float(np.sqrt(np.mean(errors**2))),
                cadence_p99_abs_ns=float(np.quantile(np.abs(errors), .99)),
                publication_age_median_ns=float(np.median(w-a))), errors


def map_clock(data, meta):
    """Apply independently supplied clock registration in integer ns, never a signal-derived best phase shift."""
    if meta.get("timing_verified") is not True or meta.get("timestamp_reference") not in ("acquisition", "exposure_center"):
        raise ValueError("unverified acquisition clock/reference; publication/arrival time is insufficient")
    offset, anchor = int(meta["clock_offset_ns"]), int(meta["clock_anchor_ns"])
    drift = float(meta["clock_drift_ppm"])
    uncertainty = float(meta["clock_uncertainty_ns"])
    if not np.isfinite(drift) or not np.isfinite(uncertainty) or uncertainty < 0:
        raise ValueError("invalid clock registration")
    result = data.copy()
    for field in ("acquired_ns", "written_ns", "receipt_ns"):
        if field in result:
            original = result[field].to_numpy()
            delta = original-anchor
            correction = np.rint(delta.astype(float)*drift*1e-6).astype(np.int64)
            # Epoch subtraction happens in integers, before conversion to seconds.
            mapped = [int(t)+offset+int(d) for t, d in zip(original, correction)]
            if any(t <= 0 or t > 2**63-1 for t in mapped):
                raise ValueError("clock mapping overflow")
            result[field] = np.asarray(mapped, dtype=np.int64)
    return result


def contiguous(data, rate, jitter_fraction=.1):
    """Return contiguous source/cadence runs; interpolation never bridges a missing publication or restart."""
    a, c, g = data.acquired_ns.to_numpy(), data.counter.to_numpy(), data.generation.to_numpy()
    cuts = [0]
    for i in range(1, len(data)):
        if (g[i] != g[i-1] or int(c[i])-int(c[i-1]) != 1 or
                abs(int(a[i])-int(a[i-1])-1e9/rate) > jitter_fraction*1e9/rate):
            cuts.append(i)
    return [data.iloc[a:b] for a, b in zip(cuts, cuts[1:]+[len(data)]) if b-a >= 4]


def align_runs(acc, wfs, acceleration, acc_rate, wfs_rate, origin):
    """Resample each contiguous run with polyphase antialias filtering, then register exposure centers.

    Interpolation of bounded timestamp jitter and symmetric filtering are for
    spectral scoring only. The C++ predictor receives untouched acquisition rows.
    """
    if wfs_rate > acc_rate:
        raise ValueError("analysis requires WFS rate <= ADC rate")
    ratio = Fraction(wfs_rate/acc_rate).limit_denominator(10000)
    if abs(float(ratio)-wfs_rate/acc_rate) > 1e-12:
        raise ValueError("unsupported rate ratio")
    blocks = []
    for ar in contiguous(acc, acc_rate):
        at = (ar.acquired_ns.to_numpy()-origin)*1e-9
        # Interpolate the acquisition clock onto its native analysis grid first.
        native = at[0]+np.arange(int((at[-1]-at[0])*acc_rate)+1)/acc_rate
        raw = acceleration[ar.index]
        regular = np.column_stack([np.interp(native, at, raw[:, axis]) for axis in range(2)])
        down = signal.resample_poly(regular, ratio.numerator, ratio.denominator, axis=0)
        down_t = native[0]+np.arange(len(down))/wfs_rate
        for wr in contiguous(wfs, wfs_rate):
            wt = (wr.acquired_ns.to_numpy()-origin)*1e-9
            lo, hi = max(at[0], wt[0])+.05, min(at[-1], wt[-1])-.05
            if hi-lo < 2:
                continue
            grid = np.arange(np.ceil(lo*wfs_rate), np.floor(hi*wfs_rate)+1)/wfs_rate
            x = np.column_stack([np.interp(grid, down_t, down[:, axis]) for axis in range(2)])
            y = np.column_stack([np.interp(grid, wt, wr[["tip", "tilt"]].to_numpy()[:, axis]) for axis in range(2)])
            blocks.append((grid, x, y))
    if not blocks:
        raise ValueError("no simultaneous contiguous two-second interval")
    return blocks


def spectra(blocks, rate, seconds=2, threshold=.8):
    """Nonoverlapping Hann Welch blocks: Pxy=conj(X)*Y, H=Pxy/Pxx; coherence gates transfer estimates."""
    n = int(round(seconds*rate))
    if n < 16 or not 0 < threshold < 1:
        raise ValueError("invalid spectral controls")
    pxx, pyy, pxy = [], [], []
    for _, x, y in blocks:
        for start in range(0, len(x)-n+1, n):
            a, b = x[start:start+n], y[start:start+n]
            f, xx = signal.welch(a, fs=rate, nperseg=n, noverlap=0, axis=0)
            _, yy = signal.welch(b, fs=rate, nperseg=n, noverlap=0, axis=0)
            _, xy = signal.csd(a, b, fs=rate, nperseg=n, noverlap=0, axis=0)
            pxx.append(xx); pyy.append(yy); pxy.append(xy)
    k = len(pxx)
    if k < 8:
        raise ValueError("need at least eight independent nonoverlapping analysis blocks")
    xx, yy, xy = [np.mean(v, axis=0) for v in (pxx, pyy, pxy)]
    coherence = np.clip(np.abs(xy)**2/np.maximum(xx*yy, np.finfo(float).tiny), 0, 1)
    accepted = coherence >= max(threshold, 1-.05**(1/(k-1)))
    h = xy/np.maximum(xx, np.finfo(float).tiny)
    phase_se = np.sqrt((1-coherence)/np.maximum(2*k*coherence, np.finfo(float).tiny))
    return dict(f=f, pxx=xx, pyy=yy, pxy=xy, coherence=coherence, h=h, accepted=accepted,
                phase_ci95_rad=1.96*phase_se, segments=k,
                psd_ci95_factors=[2*k/stats.chi2.ppf(.975, 2*k), 2*k/stats.chi2.ppf(.025, 2*k)])


def decode(acc, calibration):
    """Decode raw acc000 pairs using declared channel order/polarity and V/(m/s²), without a g factor."""
    raw = acc[["adc_x", "adc_y"]].to_numpy()
    if np.any(raw < 0) or np.any(raw > 4095):
        raise ValueError("ADC outside 12-bit range")
    # This replay format deliberately uses the production default channel order and signs.
    zero, vpb, sensitivity = [np.asarray(calibration[k], dtype=float) for k in ("zero", "volts_per_bit", "sensitivity")]
    if any(v.shape != (2,) or not np.all(np.isfinite(v)) for v in (zero, vpb, sensitivity)) or np.any(vpb <= 0) or np.any(sensitivity <= 0):
        raise ValueError("invalid per-axis ADC calibration")
    result = (raw-zero)*vpb/sensitivity
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite decoded acceleration")
    return result


def synthetic_fixture(directory, seconds=60):
    """Generate explicit analytic vibration/ADC and delayed WFS fixtures; no real measurement is fabricated."""
    if seconds < 40:
        raise ValueError("fixture needs >=40 s for training/test spectral blocks")
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12342)
    epoch = 1_700_000_000_000_000_000
    freq = np.array([28., 56.])
    amps = np.array([[2e-7, 8e-8], [1.4e-7, 1e-7]])
    phases = np.array([[.3, -.6], [-.8, .9]])
    def components(t):
        """Return analytic per-axis/band telescope pointing in radians."""
        return amps[None, :, :]*np.cos(2*np.pi*t[:, None, None]*freq+phases)
    t = np.arange(int(seconds*8000))/8000
    acceleration = np.sum(-components(t)*(2*np.pi*freq)**2*10, axis=2)
    acceleration += rng.normal(0, .002, acceleration.shape)
    adc = np.rint(2048+acceleration*1.02/.0083799).astype(int)
    acq = epoch+np.arange(len(t), dtype=np.int64)*125000
    pd.DataFrame(dict(acquired_ns=acq, written_ns=acq+20000, receipt_ns=acq+50000,
                      counter=np.arange(1, len(t)+1), generation=1, adc_x=adc[:, 0], adc_y=adc[:, 1])).to_csv(root/"accel.csv", index=False)
    tw = np.arange(int(seconds*2000))/2000
    wfs = np.sum(components(tw-.001), axis=2)*np.array([1.2, .8])
    wfs += rng.normal(0, 2e-9, wfs.shape)
    wt = epoch+np.arange(len(tw), dtype=np.int64)*500000
    # Different tagged clock, independently known +700 us correction to common.
    pd.DataFrame(dict(acquired_ns=wt-700000, written_ns=wt-600000, counter=np.arange(1, len(tw)+1),
                      generation=1, tip=wfs[:, 0], tilt=wfs[:, 1])).to_csv(root/"wfs.csv", index=False)
    clock = dict(timing_verified=True, clock_anchor_ns=epoch, clock_offset_ns=0, clock_drift_ppm=0,
                 clock_uncertainty_ns=0, timestamp_reference="acquisition", common_clock_id="synthetic_common",
                 clock_registration_provenance="known analytic fixture clocks; no replacement timestamp")
    manifest = dict(schema=1, evidence="synthetic", provenance="analytic 28/56-Hz telescope pointing; seeds 12342; not instrument telemetry",
                    accel=dict(file="accel.csv", rate_hz=8000, channels=["raw acc000[0]", "raw acc000[1]"], **clock),
                    wfs=dict(file="wfs.csv", rate_hz=2000, units="rad", channels=["analytic tip", "analytic tilt"],
                             measurement_kind="synthetic_disturbance", loop_state_provenance="analytic fixture, no real AO loop",
                             **dict(clock, clock_offset_ns=700000, timestamp_reference="exposure_center")),
                    adc=dict(zero=[2048, 2048], volts_per_bit=[.0083799]*2, sensitivity=[1.02]*2,
                             sensitivity_units="V/(m/s^2)", provenance="existing simulator numerical calibration"),
                    estimator=dict(memory_s=.1, lever_m=10, rotation_deg=0, frequencies_hz=[28, 56]),
                    query=dict(rate_hz=2000, start_offset_ns=100000, provenance="synthetic receipt/query schedule; not live availability"),
                    shadow=dict(coordinates=[[1, 0], [0, 1]], modal_map=[[1e6, 0], [0, 1e6]], gain=0,
                                bounds=[1, 1], calibration_id=12, evidence=0, provenance="explicit synthetic coefficient/rad map; no hardware meaning"),
                    reference=dict(accel_wfs_sign=[-1, -1], wfs_delay_s=.001, wfs_gains=[1.2, .8]),
                    limits="paired-channel transfer only; no identification of coupled two-input mapping; WFS center is instantaneous synthetic exposure")
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return root/"manifest.json"


def replay(executable, manifest, acc, origin, output, horizon):
    """Replay only available acquisitions; an optional earlier-fit quadrature map stays inside the file-only C++ tool."""
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    relative = acc.copy()
    for field in ("acquired_ns", "written_ns", "receipt_ns"):
        relative[field] = relative[field]-origin+1_000_000_000
    relative[list(ADC_FIELDS)].to_csv(root/"normalized_adc.csv", index=False)
    query = manifest["query"]
    if "file" in query:
        times = integer_csv(query["file"], ("query_ns",)).query_ns.to_numpy()-origin+1_000_000_000
    else:
        step = int(round(1e9/query["rate_hz"]))
        if abs(step*query["rate_hz"]-1e9) > 1:
            raise ValueError("nonintegral query period")
        times = np.arange(1_000_000_000+query["start_offset_ns"], int(relative.receipt_ns.iloc[-1]), step, dtype=np.int64)
    pd.DataFrame(dict(query_ns=times)).to_csv(root/"queries.csv", index=False)
    e, c, s = manifest["estimator"], manifest["adc"], manifest["shadow"]
    if not s.get("provenance") or s["evidence"] not in (0, 1):
        raise ValueError("explicit synthetic/archived-pupil-only shadow mapping provenance required")
    values = [manifest["accel"]["rate_hz"], e["memory_s"], e["lever_m"], e["rotation_deg"],
              *c["zero"], *c["volts_per_bit"], *c["sensitivity"], horizon, s["gain"], *s["bounds"],
              *np.asarray(s["coordinates"]).ravel(), *np.asarray(s["modal_map"]).ravel(),
              s["calibration_id"], s["evidence"], len(e["frequencies_hz"]), *e["frequencies_hz"]]
    plant = manifest.get("synthetic_response")
    values.append(int(plant is not None))
    if plant:
        if plant.get("evidence") != "synthetic" or not plant.get("provenance"):
            raise ValueError("response model must explicitly be synthetic; no measured DM response is available")
        values += [plant["period_ns"], plant["delay_ns"], plant["bandwidth_hz"], *np.asarray(plant["mapping"]).ravel()]
    params = root/f"parameters_{horizon}.txt"; params.write_text(" ".join(map(str, values))+"\n")
    path = root/f"predictions_{horizon}.csv"
    command = [str(executable), str(params), str(root / "normalized_adc.csv"), str(root / "queries.csv"), str(path)]
    if "phase_calibration" in manifest:
        coefficients = np.asarray(manifest["phase_calibration"], dtype=float)
        if coefficients.shape != (4, 2) or not np.all(np.isfinite(coefficients)) or len(e["frequencies_hz"]) != 2:
            raise ValueError("expected finite paired-band 4x2 phase calibration")
        phase = root / "phase-calibration.txt"
        lines = ["accelFFCtrl-count-phase-v1", str(horizon)]
        for band, frequency in enumerate(e["frequencies_hz"]):
            lines.append(" ".join(map(str, [frequency, band, *coefficients[2 * band : 2 * band + 2].T.ravel()])))
        phase.write_text("\n".join(lines) + "\n")
        command.append(str(phase))
    subprocess.run(command, check=True)
    d = pd.read_csv(path, dtype={k: "int64" for k in ("query_ns", "sample_ns", "target_ns")})
    valid = d.valid.to_numpy().astype(bool)
    if not np.all(np.isfinite(d.to_numpy())) or not np.all(d.valid.isin([0, 1])):
        raise ValueError("invalid/nonfinite replay diagnostics")
    if np.any(d.target_ns-d.query_ns != horizon):
        raise ValueError("replay target changed configured horizon")
    age = d.loc[valid, "query_ns"]-d.loc[valid, "sample_ns"]
    if np.any(age < 0) or np.any(age > 2_000_000) or np.any(d.loc[valid, "sample_ns"] <= 0) or np.any(d.loc[valid, "target_ns"]-d.loc[valid, "sample_ns"] > 50_000_000):
        raise ValueError("stale/over-horizon prediction")
    if np.any(d.loc[~valid | (d.state == 0), ["mode0", "mode1"]].to_numpy() != 0):
        raise ValueError("nonzero inhibited/disabled shadow")
    if np.any(np.abs(d[["mode0", "mode1"]].to_numpy()) > np.asarray(s["bounds"])*(1+1e-12)):
        raise ValueError("shadow bound violation")
    return d


def score_prediction(data, wfs, origin, transfer, frequencies, start, end, offset_ns=0):
    """Score held-out queries at their actual target, with fixed training-only paired-channel transfer."""
    target = (data.target_ns.to_numpy()-1_000_000_000)*1e-9
    wt = (wfs.acquired_ns.to_numpy()-origin)*1e-9
    prediction = np.zeros((len(data), 2))
    for band, _ in enumerate(frequencies):
        for axis, name in enumerate(("ax", "ay")):
            z = data[f"{name}_re{band}"].to_numpy()+1j*data[f"{name}_im{band}"].to_numpy()
            prediction[:, axis] += np.real(transfer[band, axis]*z)
    query = (data.query_ns.to_numpy()-1_000_000_000)*1e-9
    # Equal query interval across horizons; optical scoring never bridges missing WFS frames.
    keep = data.valid.to_numpy().astype(bool) & (query >= start) & (query <= end-max(HORIZONS)*1e-9)
    times = target+offset_ns*1e-9
    index = np.searchsorted(wt, times, side="right")
    inside = (index > 0) & (index < len(wt))
    left, right = np.clip(index-1, 0, len(wt)-1), np.clip(index, 0, len(wt)-1)
    keep &= inside & (wfs.generation.to_numpy()[left] == wfs.generation.to_numpy()[right])
    keep &= (wfs.counter.to_numpy()[right]-wfs.counter.to_numpy()[left] == 1)
    keep &= (wt[right]-wt[left] <= 1.1*np.median(np.diff(wt)))
    truth = np.column_stack([np.interp(times, wt, wfs[k]) for k in ("tip", "tilt")])
    if np.sum(keep) < 100:
        raise ValueError("insufficient held-out predictions")
    rows = []
    # Never concatenate unavailable periods when computing prediction/WFS coherence.
    indices = np.flatnonzero(keep)
    chunks = np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1)
    query_rate = 1/np.median(np.diff(query))
    try:
        if np.max(np.abs(np.diff(query)-1/query_rate)) > 1e-9:
            raise ValueError("irregular prediction queries require a separate spectral registration")
        cs = spectra([(target[c], prediction[c], truth[c]) for c in chunks if len(c)], query_rate)
    except ValueError:
        cs = None  # Insufficient continuous ready coverage is not zero coherence.
    for axis in range(2):
        a, b, t = prediction[keep, axis], truth[keep, axis], target[keep]
        error = a-b
        row = dict(axis=axis, count=len(a), rmse=float(np.sqrt(np.mean(error**2))),
                   nrmse=float(np.sqrt(np.mean(error**2)/np.mean(b**2))), bias=float(np.mean(error)),
                   correlation=float(np.corrcoef(a, b)[0, 1]))
        for frequency in frequencies:
            ratio = coherent(a, t, frequency)/coherent(b, t, frequency)
            row[f"amplitude_error_pct_{frequency:g}"] = float(100*(abs(ratio)-1))
            row[f"phase_error_deg_{frequency:g}"] = float(np.angle(ratio, deg=True))
            row[f"component_nrmse_{frequency:g}"] = float(abs(ratio-1))
            row[f"coherence_{frequency:g}"] = float(cs["coherence"][np.argmin(abs(cs["f"]-frequency)), axis]) if cs else None
        rows.append(row)
    return rows, (target, prediction, truth, keep)


def select_horizon(table):
    """Select a tested held-out grid minimum; it is a forecast score, not an inferred DM latency."""
    grouped = table.groupby("horizon_ns").nrmse.mean()
    if not np.all(np.isfinite(grouped)):
        raise ValueError("nonfinite horizon scores")
    return int(grouped.idxmin())


def analyze(manifest_path, output, executable=None, plots=True, synthetic_counterfactual=False):
    """Validate provenance/timing, estimate passive pairwise transfers, then independently score causal replay."""
    path = Path(manifest_path); meta = json.loads(path.read_text()); root = Path(output); root.mkdir(parents=True, exist_ok=True)
    if meta.get("schema") != 1 or meta.get("evidence") not in ("synthetic", "recorded") or not meta.get("provenance"):
        raise ValueError("explicit dataset evidence/provenance required")
    if (meta["adc"].get("sensitivity_units") != "V/(m/s^2)" or not meta["adc"].get("provenance") or
            not meta["wfs"].get("units") or not meta["wfs"].get("loop_state_provenance") or
            meta["wfs"].get("measurement_kind") not in ("synthetic_disturbance", "closed_loop_residual", "pseudo_open_loop", "open_loop") or
            any(len(meta[k].get("channels", [])) != 2 for k in ("accel", "wfs"))):
        raise ValueError("explicit ADC units, channel mappings and WFS units/AO state required")
    if meta["wfs"]["timestamp_reference"] != "exposure_center":
        raise ValueError("WFS timestamp must be independently registered to its exposure center")
    if (not meta["accel"].get("common_clock_id") or meta["accel"].get("common_clock_id") != meta["wfs"].get("common_clock_id") or
            not all(meta[k].get("clock_registration_provenance") for k in ("accel", "wfs"))):
        raise ValueError("independent registration to a named common clock required")
    if "file" in meta["query"]:
        meta["query"]["file"] = str((path.parent/meta["query"]["file"]).resolve())
    acc_path, wfs_path = [path.parent/meta[k]["file"] for k in ("accel", "wfs")]
    acc = map_clock(integer_csv(acc_path, ADC_FIELDS), meta["accel"])
    wfs = map_clock(integer_csv(wfs_path, WFS_FIELDS), meta["wfs"])
    if not np.all(np.isfinite(wfs[["tip", "tilt"]])):
        raise ValueError("nonfinite WFS measurements")
    at, aj = timing(acc, meta["accel"]["rate_hz"]); wt, wj = timing(wfs, meta["wfs"]["rate_hz"])
    origin = int(min(acc.acquired_ns.iloc[0], wfs.acquired_ns.iloc[0]))
    acceleration = decode(acc, meta["adc"])
    blocks = align_runs(acc, wfs, acceleration, meta["accel"]["rate_hz"], meta["wfs"]["rate_hz"], origin)
    spec = spectra(blocks, meta["wfs"]["rate_hz"])
    cross = spectra([(t, np.repeat(x[:, :1], 2, axis=1), np.repeat(x[:, 1:], 2, axis=1)) for t, x, _ in blocks], meta["wfs"]["rate_hz"])
    # Passive calibration uses only the first half; test targets stay in the second half.
    split = .5*min((int(acc.acquired_ns.iloc[-1])-origin)*1e-9, (int(wfs.acquired_ns.iloc[-1])-origin)*1e-9)
    training = [(t[t < split], x[t < split], y[t < split]) for t, x, y in blocks]
    train = spectra(training, meta["wfs"]["rate_hz"])
    frequencies = meta["estimator"]["frequencies_hz"]
    transfer = np.empty((len(frequencies), 2), dtype=complex)
    bands = []
    for band, frequency in enumerate(frequencies):
        index = int(np.argmin(np.abs(spec["f"]-frequency)))
        if abs(spec["f"][index]-frequency) > 1e-9:
            raise ValueError("requested band is not a resolved analysis bin")
        for axis in range(2):
            accepted = bool(train["accepted"][index, axis])
            transfer[band, axis] = train["h"][index, axis] if accepted else complex(np.nan)
            h = spec["h"][index, axis]
            sign = meta.get("reference", {}).get("accel_wfs_sign", [None, None])[axis]
            delay = -np.angle(h/sign)/(2*np.pi*frequency) if sign in (-1, 1) and spec["accepted"][index, axis] else None
            bands.append(dict(frequency_hz=frequency, axis=axis, coherence=float(spec["coherence"][index, axis]),
                              input_cross_coherence=float(cross["coherence"][index, 0]),
                              normalized_two_input_rcond=float((1-np.sqrt(cross["coherence"][index, 0]))/(1+np.sqrt(cross["coherence"][index, 0]))),
                              transfer_accepted=bool(spec["accepted"][index, axis]), training_accepted=accepted,
                              magnitude=float(abs(h)) if spec["accepted"][index, axis] else None,
                              phase_deg=float(np.angle(h, deg=True)) if spec["accepted"][index, axis] else None,
                              phase_ci95_deg=float(np.degrees(spec["phase_ci95_rad"][index, axis])) if spec["accepted"][index, axis] else None,
                              magnitude_relative_ci95=float(spec["phase_ci95_rad"][index, axis]) if spec["accepted"][index, axis] else None,
                              phase_equivalent_delay_s=delay,
                              delay_ci95_s=float(spec["phase_ci95_rad"][index, axis]/(2*np.pi*frequency)) if delay is not None else None,
                              delay_ambiguity_period_s=1/frequency))
    pd.DataFrame(bands).to_csv(root/"bands.csv", index=False)
    for axis in range(2):
        pd.DataFrame(dict(frequency_hz=spec["f"], acceleration_psd=spec["pxx"][:, axis], wfs_psd=spec["pyy"][:, axis],
                          csd_real=spec["pxy"][:, axis].real, csd_imag=spec["pxy"][:, axis].imag,
                          coherence=spec["coherence"][:, axis], transfer_real=np.where(spec["accepted"][:, axis], spec["h"][:, axis].real, np.nan),
                          transfer_imag=np.where(spec["accepted"][:, axis], spec["h"][:, axis].imag, np.nan))).to_csv(root/f"spectrum_{axis}.csv", index=False)
    report = dict(evidence=meta["evidence"], provenance=meta["provenance"], hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, acc_path, wfs_path)},
                  accel_timing=at, wfs_timing=wt, bands=bands, spectral_segments=spec["segments"],
                  psd_ci95_factors=spec["psd_ci95_factors"], training_end_s=split,
                  uncertainty="Conditional stationary Gaussian/nonoverlapping Welch interval approximations, not coverage guarantees for deterministic lines. ADC quantization, resampling, timing and calibration systematic bias are excluded.",
                  clock_phase_uncertainty_deg={str(f):360*f*(meta["accel"]["clock_uncertainty_ns"]+meta["wfs"]["clock_uncertainty_ns"])*1e-9 for f in frequencies},
                  scope="Passive paired-channel transfer, not a coupled two-input calibration, DM sign/response or measured suppression.",
                  query_provenance=meta["query"]["provenance"], physical_horizon_recommendation=None)
    report["wfs_measurement_kind"] = meta["wfs"]["measurement_kind"]
    report["adc_rail_values"] = int(np.sum((acc[["adc_x", "adc_y"]].to_numpy() == 0) | (acc[["adc_x", "adc_y"]].to_numpy() == 4095)))
    if "file" in meta["query"]:
        query_path = Path(meta["query"]["file"])
        report["hashes"]["queries:"+query_path.name] = hashlib.sha256(query_path.read_bytes()).hexdigest()
    report["normalized_origin_common_ns"] = origin
    report["replay_origin_ns"] = 1_000_000_000
    traces = None; replay_data = None
    if executable:
        metrics, offsets = [], []
        if not np.all(np.isfinite(transfer)):
            report["prediction_inhibited"] = "training coherence insufficient; no reliable transfer or prediction score inferred"
        else:
            end = min((int(wfs.acquired_ns.iloc[-1])-origin)*1e-9-.02, 2*split-.02)
            for horizon in HORIZONS:
                d = replay(executable, meta, acc, origin, root/"replay", horizon)
                rows, trace = score_prediction(d, wfs, origin, transfer, frequencies, split+1, end)
                for row in rows:
                    row.update(horizon_ns=horizon, availability_pct=float(100*d.valid.mean()),
                               dropped=int(d.dropped.iloc[-1]), resets=int(d.resets.iloc[-1]),
                               clipped=int(np.sum(d.state == 3)), inhibited=int(np.sum(d.state == 1)))
                metrics.extend(rows)
                if horizon == 1750000:
                    traces, replay_data = trace, d
                    for offset in (-1000000, -500000, -100000, 0, 100000, 500000, 1000000):
                        changed, _ = score_prediction(d, wfs, origin, transfer, frequencies, split+1, end, offset)
                        offsets.extend([dict(row, offset_ns=offset) for row in changed])
            table = pd.DataFrame(metrics); table.to_csv(root/"prediction_metrics.csv", index=False)
            pd.DataFrame(offsets).to_csv(root/"offset_sensitivity.csv", index=False)
            report["best_tested_forecast_horizon_ns"] = select_horizon(table)
            report["horizon_selection_limit"] = "Held-out grid minimum only; target times change with horizon. No DM response was measured; 1.75 ms is not physically qualified."
            jitter_rows = []
            if meta["evidence"] == "synthetic":
                # Known synthetic tag error only: values/receipt/query/training transfer stay identical.
                for jitter in (2000, 10000):
                    changed = acc.copy()
                    rng = np.random.default_rng(12000+jitter)
                    changed.acquired_ns += rng.integers(-jitter, jitter+1, len(changed))
                    d = replay(executable, meta, changed, origin, root/f"jitter_{jitter}", 1750000)
                    rows, _ = score_prediction(d, wfs, origin, transfer, frequencies, split+1, end)
                    jitter_rows.extend([dict(row, bounded_tag_jitter_ns=jitter, availability_pct=float(100*d.valid.mean()),
                                             dropped=int(d.dropped.iloc[-1]), resets=int(d.resets.iloc[-1])) for row in rows])
                pd.DataFrame(jitter_rows).to_csv(root/"jitter_sensitivity.csv", index=False)
            report["replay_timing"] = dict(sample_age_median_ns=float(np.median(replay_data.query_ns[replay_data.valid == 1]-replay_data.sample_ns[replay_data.valid == 1])),
                                            effective_horizon_median_ns=float(np.median(replay_data.target_ns[replay_data.valid == 1]-replay_data.sample_ns[replay_data.valid == 1])),
                                            default_gap_policy=False, shadow_gain=meta["shadow"]["gain"])
            if synthetic_counterfactual:
                if meta["evidence"] != "synthetic":
                    raise ValueError("this built-in counterfactual is restricted to the analytic synthetic fixture")
                cf = json.loads(json.dumps(meta))
                cf["shadow"]["gain"] = .5
                cf["synthetic_response"] = dict(evidence="synthetic", provenance="chosen first-order 2-kHz synthetic WFS response; no hardware measurement",
                                                  period_ns=500000, delay_ns=1000000, bandwidth_hz=300,
                                                  mapping=np.diag(np.asarray(meta["reference"]["wfs_gains"])*1e-6).tolist())
                d = replay(executable, cf, acc, origin, root/"counterfactual", 1750000)
                (root/"counterfactual"/"manifest.json").write_text(json.dumps(cf, indent=2)+"\n")
                qt = (d.query_ns.to_numpy()-1_000_000_000)*1e-9
                optical_time = (wfs.acquired_ns.to_numpy()-origin)*1e-9
                use = (qt >= split+1) & (qt <= end)
                truth = np.column_stack([np.interp(qt, optical_time, wfs[k]) for k in ("tip", "tilt")])
                residual = truth+d[["synthetic_effect_x", "synthetic_effect_y"]].to_numpy()
                result = []
                for axis in range(2):
                    before, after = [np.sqrt(np.mean(v[use, axis]**2)) for v in (truth, residual)]
                    result.append(dict(axis=axis, before_rms=float(before), after_rms=float(after),
                                       variance_reduction_pct=float(100*(1-(after/before)**2))))
                report["synthetic_counterfactual"] = dict(evidence="SIMULATED suppression only", gain=.5, horizon_ns=1750000, axes=result)
    (root/"manifest_copy.json").write_text(json.dumps(meta, indent=2)+"\n")
    (root/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if plots:
        make_plots(root, spec, bands, meta, aj, wj, traces, replay_data)
    return report


def make_plots(root, spec, bands, meta, aj, wj, traces, replay_data):
    """Save scientific figures with evidence, units, masked transfer and explicit offline timing labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    for axis in range(2):
        f, h, ok = spec["f"], spec["h"][:, axis], spec["accepted"][:, axis]
        axes[0, axis].loglog(f[1:], spec["pxx"][1:, axis], color="tab:blue")
        axes[0, axis].set_ylabel("Acceleration PSD [(m/s²)²/Hz]", color="tab:blue")
        optical = axes[0, axis].twinx()
        optical.loglog(f[1:], spec["pyy"][1:, axis], color="tab:orange")
        optical.set_ylabel(f'WFS PSD [{meta["wfs"]["units"]}²/Hz]', color="tab:orange")
        axes[1, axis].plot(f, spec["coherence"][:, axis]); axes[1, axis].axhline(.8, color="red", ls=":")
        axes[1, axis].set_ylabel("Magnitude-squared coherence")
        axes[2, axis].semilogy(f, np.where(ok, abs(h), np.nan)); axes[2, axis].set_ylabel(f'H [{meta["wfs"]["units"]}/(m/s²)]')
        axes[3, axis].plot(f, np.where(ok, np.angle(h, deg=True), np.nan)); axes[3, axis].set_ylabel("Transfer phase [deg]")
        for band in bands:
            if band["axis"] == axis and band["transfer_accepted"]:
                axes[2, axis].errorbar(band["frequency_hz"], band["magnitude"],
                                       yerr=band["magnitude"]*band["magnitude_relative_ci95"], fmt="o", ms=3)
                axes[3, axis].errorbar(band["frequency_hz"], band["phase_deg"], yerr=band["phase_ci95_deg"], fmt="o", ms=3)
        axes[3, axis].set(xlabel="Frequency [Hz]", xlim=(10, 100))
        axes[0, axis].set_title(f"Paired axis {axis}; {meta['evidence']}")
    fig.tight_layout(); fig.savefig(root/"spectra.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for values, label in ((aj, "ADC"), (wj, "WFS")):
        axes[0].hist(values, bins=60, alpha=.5, label=label)
    axes[0].set(xlabel="Per-counter cadence error [ns]", ylabel="Count"); axes[0].legend()
    axes[1].bar(("ADC", "WFS"), [meta[k]["clock_offset_ns"]*1e-6 for k in ("accel", "wfs")])
    axes[1].set_ylabel("Declared clock correction [ms]; not measured from vibration")
    fig.tight_layout(); fig.savefig(root/"timing.png"); plt.close(fig)
    if traces is None:
        return
    table = pd.read_csv(root/"prediction_metrics.csv")
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for axis, rows in table.groupby("axis"):
        axes[0].plot(rows.horizon_ns*1e-6, rows.nrmse*100, "o-", label=f"axis {axis}")
        axes[1].plot(rows.horizon_ns*1e-6, rows.availability_pct, "o-")
    axes[0].set_ylabel("Held-out NRMSE [%]"); axes[0].legend()
    axes[1].set(xlabel="Causal horizon beyond query [ms]", ylabel="Replay availability [%]")
    fig.suptitle(f"{meta['evidence']}; fixed first-half passive transfer; no DM optimum")
    fig.tight_layout(); fig.savefig(root/"horizons.png"); plt.close(fig)
    t, predicted, truth, valid = traces; first = t[valid][0]; use = valid & (t < first+.3)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for axis in range(2):
        axes[axis].plot(t[use], truth[use, axis], label="WFS interpolated at target")
        axes[axis].plot(t[use], predicted[use, axis], label="Causal calibrated prediction", alpha=.7)
        axes[axis].set_ylabel(meta["wfs"]["units"]); axes[axis].legend()
    axes[2].plot(t[use], (predicted-truth)[use]); axes[2].set(xlabel="Target time [s]", ylabel="Residual")
    fig.tight_layout(); fig.savefig(root/"prediction.png"); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot((replay_data.query_ns-1e9)*1e-9, replay_data[["mode0", "mode1"]])
    axes[0].set(xlabel="Query time [s]", ylabel="Shadow coefficients", title="Host-only proposals; configured gain defaults zero")
    v = replay_data.valid.astype(bool)
    axes[1].hist((replay_data.query_ns[v]-replay_data.sample_ns[v])*1e-3, bins=50)
    axes[1].set(xlabel="Sample age at ready query [us]", ylabel="Count")
    fig.tight_layout(); fig.savefig(root/"shadow.png"); plt.close(fig)
    offset = pd.read_csv(root/"offset_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, rows in offset.groupby("axis"):
        axes[0].plot(rows.offset_ns*1e-6, rows.nrmse*100, "o-", label=f"axis {axis}")
    axes[0].set(xlabel="Deliberately incorrect WFS timestamp offset [ms]", ylabel="NRMSE [%]"); axes[0].legend()
    for jitter in (2000, 10000):
        path = root/f"jitter_{jitter}"/"normalized_adc.csv"
        if path.exists():
            stamps = pd.read_csv(path, usecols=["acquired_ns"], dtype="int64").acquired_ns.to_numpy()
            axes[1].hist((np.diff(stamps)-1e9/meta["accel"]["rate_hz"])*1e-3, bins=50, alpha=.5, label=f"bounded {jitter/1000:g}-us tag error")
    axes[1].set(xlabel="Injected acquisition cadence error [us]", ylabel="Count"); axes[1].legend()
    fig.tight_layout(); fig.savefig(root/"offset_jitter.png"); plt.close(fig)
    path = root/"counterfactual"/"predictions_1750000.csv"
    if path.exists():
        cf = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot((cf.query_ns-1e9)*1e-9, cf[["mode0", "mode1"]], lw=.4)
        ax.set(xlabel="Query time [s]", ylabel="Synthetic absolute coefficients", title="Explicit counterfactual gain 0.5; host memory only")
        fig.tight_layout(); fig.savefig(root/"counterfactual"/"commands.png"); plt.close(fig)


def main():
    """Operate on local files only; optionally generate explicitly synthetic validation input."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--synthetic", type=float, metavar="SECONDS")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--synthetic-counterfactual", action="store_true")
    args = parser.parse_args()
    manifest = synthetic_fixture(args.output/"fixture", args.synthetic) if args.synthetic else args.manifest
    report = analyze(manifest, args.output, args.replay, not args.no_plots, args.synthetic_counterfactual)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
