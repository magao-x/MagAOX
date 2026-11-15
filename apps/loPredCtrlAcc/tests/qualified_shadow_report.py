r"""\file qualified_shadow_report.py
\brief Reproduce and summarize actual native-DDSPC synthetic replay artifacts using ordinary files only.
\ingroup loPredCtrlAcc_unit_test
"""
import argparse
import csv
import json
import hashlib
import math
from pathlib import Path
import statistics
import subprocess
import socket


def rows(path):
    """Read an ordinary CSV artifact, never shared-memory or a process interface."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def inspect(directory):
    """Reject physical/deadline claims and retain matched target metrics from one explicitly synthetic run."""
    summary = json.loads((directory / "summary.json").read_text())
    if (summary["evidence"] != "synthetic_requested" or summary["live_qualified"] is not False
            or summary["deadline_qualified"] is not False):
        raise ValueError("Synthetic qualification must never become live or deadline qualification")
    if (summary["adc_samples"] != 4 * summary["frames"] or summary["adc_invalid"]
            or summary["adc_dropped"] or summary["score_drops"]
            or not summary["standard_ready"] or not summary["assisted_ready"]
            or abs(summary["adc_rate_hz"] - 8000) > 1):
        raise ValueError("Incomplete replay, loss or unready model")
    times = sorted(float(row["whole_frame_us"]) for row in rows(directory / "compute.csv"))
    if len(times) != summary["frames"] or any(not math.isfinite(t) or t < 0 for t in times):
        raise ValueError("Invalid compute measurements")
    summary["compute_us"] = dict(mean=statistics.mean(times), median=statistics.median(times),
                                 p99=times[math.ceil(.99 * len(times)) - 1], maximum=times[-1])
    timing = rows(directory / "compute.csv")
    if timing and "numerical_us" in timing[0]:
        for key in ("numerical_us", "max_job_us"):
            values = sorted(float(row[key]) for row in timing)
            summary[key] = dict(mean=statistics.mean(values), p99=values[math.ceil(.99*len(values))-1], maximum=values[-1])
    spectra = {}
    for row in rows(directory / "psd.csv"):
        key = (int(row["horizon"]), int(row["channel"]))
        entry = tuple(float(row[k]) for k in ("frequency_hz", "wfs_psd", "standard_error_psd", "assisted_error_psd"))
        if any(not math.isfinite(v) or v < 0 for v in entry):
            raise ValueError("Invalid prediction-error PSD")
        spectra.setdefault(key, []).append(entry)
    result = []
    for row in rows(directory / "metrics.csv"):
        h, channel = int(row["horizon"]), int(row["channel"])
        standard, assisted = float(row["standard_rmse"]), float(row["assisted_rmse"])
        if (not 1 <= h <= summary["future"] or channel not in (0, 1) or int(row["count"]) < 1
                or not all(math.isfinite(v) and v >= 0 for v in (standard, assisted))):
            raise ValueError("Invalid matched prediction metric")
        item = dict(horizon=h, channel=channel, samples=int(row["count"]), standard_rmse=standard,
                    assisted_rmse=assisted, mse_improvement_percent=100*(1-(assisted/standard)**2) if standard else None,
                    welch_windows=int(row["welch_windows"]))
        spectrum = spectra.get((h, channel), [])
        for center in (28.5, 56):
            band = [entry for entry in spectrum if center-2 <= entry[0] <= center+2]
            item[f"band_{center:g}_improvement_percent"] = None
            if band and sum(entry[2] for entry in band) > 0:
                item[f"band_{center:g}_improvement_percent"] = 100*(1-sum(entry[3] for entry in band)/sum(entry[2] for entry in band))
        result.append(item)
    if len(result) != 2 * summary["future"]:
        raise ValueError("Missing per-axis/per-horizon matched results")
    return dict(summary=summary, metrics=result)


def independence(coherent, unrelated):
    """Require bit-identical standard coefficients and measured targets when only ADC information changes."""
    def standard(directory):
        return [row for row in rows(directory / "coefficients.csv") if row["mode"] == "0"]
    if standard(coherent) != standard(unrelated):
        raise ValueError("Standard fit changed under an ADC-only negative control")
    def targets(directory):
        return {(row["origin"], row["target"], row["horizon"], row["channel"]): row["measurement"]
                for row in rows(directory / "predictions.csv") if row["mode"] == "0"}
    if targets(coherent) != targets(unrelated):
        raise ValueError("Negative control changed WFS targets or standard forecast coverage")


def main():
    """Run only an explicit offline binary, or inspect existing ordinary artifacts; never install or start services."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New report directory")
    parser.add_argument("--binary", type=Path, help="Explicit synthetic offline CLI; omitted when inspecting --runs")
    parser.add_argument("--runs", type=Path, nargs="*", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[73, 91, 107])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--frames", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=1500)
    args = parser.parse_args()
    if args.binary is not None and args.runs:
        parser.error("Choose --binary or --runs")
    if args.binary and socket.gethostname().startswith("exao"):
        parser.error("Run numerical replay on a personal/development host, not an observing instrument host")
    args.output.mkdir(exist_ok=False)
    directories = args.runs
    if args.binary:
        directories = []
        for seed in args.seeds:
            for h in args.horizons:
                for learner in ("rls", "qr"):
                    pair = []
                    for adc in ("coherent", "unrelated"):
                        directory = args.output / f"seed{seed}_f{h}_{learner}_{adc}"
                        subprocess.run([str(args.binary.resolve()), "--output", str(directory), "--seed", str(seed),
                                        "--future", str(h), "--frames", str(args.frames), "--warmup", str(args.warmup),
                                        "--learner", learner, "--adc", adc], check=True)
                        pair.append(directory)
                        directories.append(directory)
                    independence(*pair)
    if not directories:
        parser.error("Provide --binary or --runs")
    reports = [inspect(directory) for directory in directories]
    (args.output / "report.json").write_text(json.dumps(dict(live_qualified=False, deadline_qualified=False,
                                                           binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest() if args.binary else None,
                                                           interpretation="Origin-only held-last-input forecast errors; conditional errors are auxiliary, no closed-loop suppression",
                                                           runs=reports), indent=2) + "\n")
    text = ["# Synthetic native-DDSPC qualification", "",
            "Actual C++ ADC/RLS/QR/multirate code. No physical or deadline qualification.", "",
            "Errors use identical targets and horizons after the specified chronological warm-up.",
            "Primary forecasts use only origin state and the last confirmed input. Actual future commands enter auxiliary conditional scoring only.", "",
            "| Seed | F | Learner | ADC | h | Axis | Standard RMSE | Assisted RMSE | MSE improvement % | 28.5-Hz band % | 56-Hz band % |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for report in reports:
        s = report["summary"]
        for m in report["metrics"]:
            display = lambda v: "unavailable" if v is None else f"{v:.4g}"
            text.append(f"| {s['seed']} | {s['future']} | {s['learner']} | {s['adc']} | {m['horizon']} | "
                        f"{('tip', 'tilt')[m['channel']]} | {m['standard_rmse']:.6g} | {m['assisted_rmse']:.6g} | "
                        f"{display(m['mse_improvement_percent'])} | {display(m['band_28.5_improvement_percent'])} | "
                        f"{display(m['band_56_improvement_percent'])} |")
    text += ["", "Positive synthetic information is not telescope performance. Unrelated ADC may hurt through finite-sample fitting.",
             "No hypothesis test or physical vibration suppression is inferred from these descriptive results.", ""]
    (args.output / "README.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
