#!/usr/bin/env python3
r"""\file inspect_reduced_telemetry.py
\brief Read-only reduced-array inventory and sample-domain spectra without fabricated frame timestamps.
\ingroup accelFFCtrl_unit_test

The supplied display rate labels a conditional frequency axis only. This tool
does not compute cross-sensor coherence, phase, physical calibration or replay.
"""
import argparse
import calendar
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import signal


def filename_ns(name):
    """Read exact UTC nanoseconds encoded in an XRIF/NPY name, without assigning frame-time semantics."""
    match = re.fullmatch(r"(.+)_([0-9]{14})([0-9]{9})\.(xrif|npy)", name)
    if not match:
        raise ValueError(f"invalid archive filename: {name}")
    date = datetime.strptime(match[2], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return calendar.timegm(date.utctimetuple())*1_000_000_000+int(match[3])


def utc(ns):
    """Format an integer encoded archive timestamp without epoch float rounding."""
    seconds, nanos = divmod(int(ns), 1_000_000_000)
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")+f".{nanos:09d}Z"


def sha256(path):
    """Hash one existing local file opened read-only."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_spectrum(values):
    """Compute a complete-file Hann periodogram in cycles/recorded-frame; never bridge archive boundaries."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 16 or not np.all(np.isfinite(values)):
        raise ValueError("invalid array for file-local spectrum")
    return signal.periodogram(values, fs=1., window="hann", detrend="constant", scaling="density", axis=0)


def inspect_stream(directory, stream, display_rate):
    """Check every manifest/array and return descriptive statistics and conditional sample-index spectra."""
    directory = Path(directory)
    entries = json.loads((directory/"manifest.json").read_text())
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("need an ordered multi-file manifest")
    names = [entry["output_file"] for entry in entries]
    sources = [entry["source_file"] for entry in entries]
    if len(set(names)) != len(names) or len(set(sources)) != len(sources):
        raise ValueError("duplicate manifest file")
    if set(names) != {p.name for p in directory.glob("*.npy")}:
        raise ValueError("manifest/NPY inventory mismatch")
    stamps = [filename_ns(name) for name in sources]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("nonmonotonic manifest filename order")
    rows, arrays, psds, unique = [], [], [], None
    shape, dtype = None, None
    for index, entry in enumerate(entries):
        name = entry["output_file"]
        if Path(name).name != name or name != Path(entry["source_file"]).with_suffix(".npy").name:
            raise ValueError("unsafe/inconsistent output filename")
        path = directory/name
        a = np.load(path, mmap_mode="r", allow_pickle=False)
        if a.ndim != 2 or list(a.shape) != entry["shape"] or str(a.dtype) != entry["dtype"]:
            raise ValueError("array/manifest shape or dtype mismatch")
        if shape is None:
            shape, dtype = a.shape, a.dtype
            if stream == "acc000" and dtype != np.uint16:
                raise ValueError("accelerometer array must contain UINT16 codes")
            if stream == "wfs_tiptilt" and (dtype != np.float32 or a.shape[1] != 2):
                raise ValueError("WFS array must contain two FLOAT32 modes")
            unique = [set() for _ in range(shape[1])] if dtype.kind == "u" else None
        if a.shape != shape or a.dtype != dtype:
            raise ValueError("changing chunk geometry: inspect segments separately")
        frequency, power = file_spectrum(a)
        psds.append(power)
        values = np.asarray(a, dtype=float)
        arrays.append(values)
        if unique is not None:
            for channel in range(shape[1]):
                unique[channel].update(np.unique(a[:, channel]).tolist())
        rounded = entry.get("filename_timestamp_utc")
        error = float(Fraction.from_float(float(rounded))*1_000_000_000-stamps[index]) if rounded is not None else None
        rows.append(dict(stream=stream, file_index=index, source_file=entry["source_file"], output_file=name,
                         frames=len(a), channels=a.shape[1], dtype=str(a.dtype), filename_ns=stamps[index],
                         filename_utc=utc(stamps[index]), manifest_float_error_ns=error, sha256=sha256(path)))
    values = np.concatenate(arrays)
    power = np.mean(psds, axis=0)
    stats = []
    peaks = []
    conditional_f = frequency*display_rate
    for channel in range(shape[1]):
        x = values[:, channel]
        stats.append(dict(stream=stream, channel=channel, label=("tip" if channel == 0 else "tilt") if stream == "wfs_tiptilt" else f"raw_{channel}",
                          frames=len(x), minimum=float(np.min(x)), maximum=float(np.max(x)),
                          mean=float(np.mean(x)), fluctuation_rms=float(np.std(x)), zero_fraction=float(np.mean(x == 0)),
                          unique_codes=len(unique[channel]) if unique else None))
        for name, band, background in (("28_29", (27, 30), (20, 40)), ("56", (54, 58), (47, 67))):
            bins = np.flatnonzero((conditional_f >= band[0]) & (conditional_f <= band[1]))
            side = (conditional_f >= background[0]) & (conditional_f <= background[1]) & ~((conditional_f >= band[0]-2) & (conditional_f <= band[1]+2))
            if not len(bins) or not np.any(side):
                raise ValueError("display rate does not resolve requested search bands")
            peak = bins[np.argmax(power[bins, channel])]
            floor = np.median(power[side, channel])
            contrast = float(10*np.log10(power[peak, channel]/floor)) if floor > 0 and power[peak, channel] > 0 else None
            peaks.append(dict(stream=stream, channel=channel, band=name, peak_cycles_per_frame=float(frequency[peak]),
                              conditional_peak_hz=float(conditional_f[peak]), conditional_bin_hz=float(display_rate/shape[0]),
                              peak_power_per_cycles_per_frame=float(power[peak, channel]),
                              conditional_peak_psd_per_hz=float(power[peak, channel]/display_rate), local_contrast_db=contrast))
    intervals = np.diff(np.asarray(stamps, dtype=np.int64))
    densities = shape[0]*1e9/intervals.astype(float)
    span = stamps[-1]-stamps[0]
    summary = dict(files=len(entries), frames=len(values), channels=shape[1], dtype=str(dtype),
                   shape_per_file=list(shape), manifest_sorted_lexically=names == sorted(names),
                   first_filename_utc=utc(stamps[0]), last_filename_utc=utc(stamps[-1]),
                   first_filename_ns=stamps[0], last_filename_ns=stamps[-1], filename_span_s=span*1e-9,
                   filename_interval_median_s=float(np.median(intervals)*1e-9),
                   filename_interval_min_s=float(np.min(intervals)*1e-9), filename_interval_max_s=float(np.max(intervals)*1e-9),
                   median_bulk_record_density_hz=float(np.median(densities)),
                   aggregate_bulk_record_density_hz=(len(entries)-1)*shape[0]*1e9/span,
                   conditional_duration_s=len(values)/display_rate,
                   fractional_rate_and_per_frame_timing_verified=False)
    return summary, rows, stats, peaks, frequency, power


def inspect(dataset, output, display_rate, plots=True):
    """Inspect local data read-only, hash before/after and write diagnostics exclusively outside the dataset."""
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output == dataset or dataset in output.parents:
        raise ValueError("output must be outside the original dataset")
    if not np.isfinite(display_rate) or display_rate <= 0:
        raise ValueError("positive explicit display rate required")
    paths = sorted(p for p in dataset.rglob("*") if p.is_file())
    before = {str(p.relative_to(dataset)): sha256(p) for p in paths}
    metadata = json.loads((dataset/"metadata.json").read_text())
    if metadata.get("wfs_modes") != dict(tip=0, tilt=1):
        raise ValueError("explicit tip=0/tilt=1 reduced-array mapping required")
    output.mkdir(parents=True, exist_ok=True)
    report = dict(dataset=str(dataset), evidence="real telescope recording, user-provided XRIF-derived NPY reduction",
                  metadata=metadata, display_rate_hz=display_rate, physical_timing_qualified=False,
                  cross_sensor_analysis_performed=False, prediction_replay_performed=False,
                  limitation="Filename-only timing. PSDs are sample-domain; Hz axes assume the explicit display rate. No per-frame acquisition timestamps generated.", streams={})
    inventory, stats, peaks, spectra = [], [], [], {}
    for stream in ("acc000", "wfs_tiptilt"):
        summary, rows, st, pk, f, p = inspect_stream(dataset/stream, stream, display_rate)
        expected = metadata.get("acc_files" if stream == "acc000" else "wfs_files")
        if expected is not None and expected != summary["files"]:
            raise ValueError("metadata file count mismatch")
        report["streams"][stream] = summary
        inventory.extend(rows); stats.extend(st); peaks.extend(pk); spectra[stream] = (f, p)
        columns = dict(cycles_per_recorded_frame=f, conditional_frequency_hz=f*display_rate)
        for channel in range(p.shape[1]):
            columns[f"channel{channel}_power_per_cycles_per_frame"] = p[:, channel]
            columns[f"channel{channel}_conditional_psd_per_hz"] = p[:, channel]/display_rate
        pd.DataFrame(columns).to_csv(output/f"{stream}_psd.csv", index=False)
    pd.DataFrame(inventory).to_csv(output/"inventory.csv", index=False)
    pd.DataFrame(stats).to_csv(output/"channel_statistics.csv", index=False)
    pd.DataFrame(peaks).to_csv(output/"conditional_peaks.csv", index=False)
    if plots:
        make_plots(output, spectra, pd.DataFrame(stats), pd.DataFrame(inventory), display_rate)
    after = {str(p.relative_to(dataset)): sha256(p) for p in sorted(dataset.rglob("*")) if p.is_file()}
    if before != after:
        raise ValueError("dataset changed during inspection")
    report["input_files_verified_unchanged"] = len(before)
    (output/"input_sha256.json").write_text(json.dumps(before, indent=2)+"\n")
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def make_plots(output, spectra, stats, inventory, rate):
    """Save conditional-frequency spectra and chunk cadence; draw no synchronized cross-stream timeline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for stream, (f, p) in spectra.items():
        channels = p.shape[1]
        fig, axes = plt.subplots(channels, 1, figsize=(11, max(6, 2*channels)), sharex=True, squeeze=False)
        for channel in range(channels):
            ax = axes[channel, 0]
            ax.semilogy(f*rate, p[:, channel]/rate)
            for lo, hi in ((27, 30), (54, 58)):
                ax.axvspan(lo, hi, color="orange", alpha=.2)
            ax.set(xlim=(5, 180), ylabel=f'{"tip" if channel == 0 else "tilt"} modal²/Hz' if stream == "wfs_tiptilt" else f"ch{channel} counts²/Hz")
        axes[-1, 0].set_xlabel(f"Frequency [Hz ONLY IF {rate:g} recorded frames/s; acquisition timing unverified]")
        fig.suptitle(f"REAL {stream}; file-local Hann spectra; no joining archive boundaries")
        fig.tight_layout(); fig.savefig(output/f"{stream}_spectra.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    a = stats[stats.stream == "acc000"]
    axes[0].bar(a.channel, a.fluctuation_rms); axes[0].set(xlabel="Raw accelerometer channel", ylabel="Raw fluctuation RMS [counts]")
    for stream, group in inventory.groupby("stream"):
        intervals = np.diff(group.filename_ns.to_numpy(dtype=np.int64))*1e-9
        expected = group.frames.iloc[0]/rate
        axes[1].plot(np.arange(len(intervals)), (intervals-expected)*1e6, label=stream)
    axes[1].set(xlabel="Archive interval index (separate streams)", ylabel="Start-label spacing minus N/display-rate [us]")
    axes[1].legend(); fig.suptitle("Archive-start differences are NOT per-frame jitter")
    fig.tight_layout(); fig.savefig(output/"channels_and_archive_cadence.png"); plt.close(fig)


def main():
    """Inspect a local reduced bundle; the display rate is explicitly assumed, never a timestamp reconstruction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--display-rate-hz", required=True, type=float)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(inspect(args.dataset, args.output, args.display_rate_hz, not args.no_plots), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
