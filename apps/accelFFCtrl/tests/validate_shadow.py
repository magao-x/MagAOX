#!/usr/bin/env python3
r"""\file validate_shadow.py
\brief Offline shadow audit, read-only archived pupil calibration and synthetic AO interaction analysis.
\ingroup accelFFCtrl_unit_test

No application startup, stream attachment, physical timing inference or instrument access.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from validate_predictions import RAD_TO_MAS

INTEGER_FIELDS = ("record", "epoch", "input_revision", "calibration_id", "evidence", "query_ns", "sample_ns",
                  "prediction_target_ns", "requested_target_ns", "source_counter", "source_generation", "state", "fault", "prediction_valid")


def audit_log(path, mapping, gain, bounds, horizon, max_age=2_000_000, max_prediction=50_000_000):
    """Audit exact integer times/identities and absolute modal values; numerical validity is not optical approval."""
    dtype = {key: "int64" if key.endswith("_ns") else "uint64" for key in INTEGER_FIELDS}
    data = pd.read_csv(path, dtype=dtype)
    if data.empty or not set(INTEGER_FIELDS).issubset(data.columns):
        raise ValueError("incomplete shadow log")
    np.testing.assert_array_equal(data.record.to_numpy(), np.arange(1, len(data)+1, dtype=np.uint64))
    values = data[["scale", "theta_x_rad", "theta_y_rad", "dm_x_rad", "dm_y_rad", "mode0", "mode1"]].to_numpy()
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite shadow diagnostic")
    np.testing.assert_array_equal(data.requested_target_ns-data.query_ns, np.full(len(data), horizon))
    assert np.all(data.state.isin([0, 1, 2, 3]))
    assert np.all(data.prediction_valid.isin([0, 1]))
    assert np.all(data.loc[data.state == 1, "prediction_valid"] == 0)
    active = data.state.isin([2, 3]).to_numpy()
    zero = ~active
    np.testing.assert_array_equal(data.loc[zero, ["mode0", "mode1"]], np.zeros((np.sum(zero), 2)))
    if np.any(active):
        current = data[active]
        age = current.query_ns-current.sample_ns
        assert np.all((age >= 0) & (age <= max_age))
        assert np.all(current.prediction_target_ns == current.requested_target_ns)
        assert np.all(current.requested_target_ns-current.sample_ns <= max_prediction)
        assert np.all(current.fault == 0)
        assert np.all(current.prediction_valid == 1)
        assert np.all(current.source_counter > 0) and np.all(current.source_generation > 0)
        proposed = -gain*current[["dm_x_rad", "dm_y_rad"]].to_numpy() @ np.asarray(mapping).T
        ratios = np.divide(np.asarray(bounds)[None, :], np.abs(proposed), out=np.ones_like(proposed), where=proposed != 0)
        scale = np.minimum(1, np.min(ratios, axis=1))
        np.testing.assert_allclose(current.scale, scale, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(current[["mode0", "mode1"]], proposed*scale[:, None], rtol=1e-12, atol=1e-16)
        assert np.all(np.abs(current[["mode0", "mode1"]].to_numpy()) <= np.asarray(bounds)*(1+1e-12))
        for _, segment in current.groupby("epoch"):
            assert segment.source_generation.nunique() == 1
            assert np.all(segment.source_counter.to_numpy()[1:] >= segment.source_counter.to_numpy()[:-1])
    return dict(records=len(data), active=int(np.sum(active)), inhibited=int(np.sum(data.state == 1)),
                disabled=int(np.sum(data.state == 0)), limited=int(np.sum(data.state == 3)),
                numerically_valid=int(np.sum(data.prediction_valid == 1)), epochs=data.epoch.nunique(), evidence=data.evidence.unique().tolist())


def audit_directory(root):
    """Verify synthetic study provenance, explicit identity coordinates and shadow records for each selected replay."""
    root = Path(root)
    reports = []
    for path in sorted(root.glob("*.csv")):
        text = path.with_suffix(".txt").read_text()
        header, matrix = text.split("K=\n", 1)
        meta = dict(line.split("=", 1) for line in header.splitlines())
        if meta["fixture"] != "synthetic_pupil8" or meta["registration"] != "explicit_identity_synthetic_only":
            raise ValueError("unexpected or unqualified study provenance")
        mapping = np.loadtxt(io.StringIO(matrix))
        result = audit_log(path, mapping, float(meta["gain"]), list(map(float, meta["bounds"].split(","))),
                           int(meta["horizon_ns"]), int(meta["max_age_ns"]), int(meta["max_prediction_ns"]))
        data = pd.read_csv(path, dtype={key: "int64" if key.endswith("_ns") else "uint64" for key in INTEGER_FIELDS})
        np.testing.assert_array_equal(data[["theta_x_rad", "theta_y_rad"]], data[["dm_x_rad", "dm_y_rad"]].to_numpy())
        result["run"] = path.stem
        reports.append(result)
    if not reports:
        raise ValueError("no shadow log fixtures")
    return reports


def fit_pupil(modes, mask, coordinates):
    """Independent SVD pixel fit of B*K=.5e6*[x,y], retaining supplied normalization and all mode pixels."""
    b, w, xy = map(np.asarray, (modes, mask, coordinates))
    if b.ndim != 2 or b.shape[1] != 2 or w.shape != (len(b),) or xy.shape != b.shape:
        raise ValueError("invalid calibration geometry")
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("invalid calibration samples/weights")
    support = w > 0
    if not np.all(np.isfinite(xy[support])):
        raise ValueError("invalid pupil coordinates")
    weight = np.sqrt(w[support]/np.max(w))[:, None]
    design, target = weight*b[support], weight*.5e6*xy[support]
    if np.linalg.matrix_rank(design) != 2 or np.linalg.matrix_rank(target) != 2:
        raise ValueError("degenerate calibration")
    mapping = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = np.linalg.norm(design@mapping-target, axis=0)/np.linalg.norm(target, axis=0)
    return mapping, residual


def archived_calibration(directory, output, projector=None):
    """Read archived FITS only; use the previously owner-defined DM-pupil geometry, without WFS/sensor registration."""
    from astropy.io import fits
    directory, output = Path(directory), Path(output)
    paths = [directory/"CMmodesDM.00.fits", directory/"dmmask.fits"]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with fits.open(paths[0], mode="readonly", memmap=False) as f:
        modes = np.asarray(f[0].data, dtype=float)
    with fits.open(paths[1], mode="readonly", memmap=False) as f:
        mask = np.asarray(f[0].data, dtype=float)
    if modes.shape != (2, 50, 50) or mask.shape != (50, 50):
        raise ValueError("expected archived two-plane 50x50 TT basis and mask")
    row, column = np.indices((50, 50))
    coordinates = np.column_stack(((column.ravel()-24.5)*6.5/48, (row.ravel()-24.5)*6.5/48))
    b, w = modes.reshape(2, -1).T, mask.ravel()
    mapping, residual = fit_pupil(b, w, coordinates)
    if max(residual) > 1e-6:
        raise ValueError("archived numerical pupil fit exceeds explicit analysis tolerance")
    if projector:
        old = np.loadtxt(projector, delimiter=",")
        np.testing.assert_allclose(mapping, old[:, :2].T, rtol=1e-12, atol=1e-6)
    output.mkdir(parents=True, exist_ok=True)
    angles = np.asarray([[1, 0], [-1, 0], [0, 1], [0, -1], [2, -3], [3, 4]], dtype=float)/RAD_TO_MAS
    commands = -angles @ mapping.T
    with (output/"calibration_fixture.txt").open("w") as stream:
        np.savetxt(stream, mapping, fmt="%.17g")
        stream.write(f"{len(angles)}\n")
        np.savetxt(stream, np.column_stack((angles, commands)), fmt="%.17g")
    pd.DataFrame(np.column_stack((angles*RAD_TO_MAS, commands)), columns=("pupil_x_mas", "pupil_y_mas", "correction_mode0", "correction_mode1")).to_csv(output/"archived_examples.csv", index=False)
    report = dict(qualification="archived DM-pupil basis only; accelerometer/WFS registration, sign and dynamics UNMEASURED",
                  hashes=hashes, coordinate_source="prior owner guidance: increasing column/row, center24.5, D6.5m/48 pitches",
                  mapping_coefficients_per_rad=mapping.tolist(), residual=residual.tolist(),
                  masked_rms=np.sqrt(np.sum(w[:, None]*b*b, axis=0)/np.sum(w)).tolist(),
                  projector_crosscheck=bool(projector))
    for p in paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[p.name]
    (output/"archived_calibration.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def analyze_ao(directory, plots=True):
    """Score a prescribed synthetic AO model; saturation-bounded trajectories are not proof of closed-loop stability."""
    root = Path(directory)
    table = pd.read_csv(root/"summary.csv", dtype={"trajectory_hash": "uint64"})
    before = table[table["case"] == "ao_only"].trajectory_hash.iloc[0]
    after = table[table["case"] == "shadow_only"].trajectory_hash.iloc[0]
    assert before == after
    data = {name: pd.read_csv(root/f"{name}.csv") for name in table["case"].unique()}
    baseline, shadow = data["ao_only"], data["shadow_only"]
    np.testing.assert_array_equal(baseline[["residual_x", "residual_y", "ao0", "ao1", "total0", "total1"]],
                                  shadow[["residual_x", "residual_y", "ao0", "ao1", "total0", "total1"]])
    assert np.max(np.abs(shadow[["shadow0", "shadow1"]].to_numpy())) > 0
    np.testing.assert_array_equal(baseline[["shadow0", "shadow1"]], np.zeros((len(baseline), 2)))
    assert np.all(table.total_peak <= table.limit*(1+1e-12))
    table["before_rms_mas"] = table.before_rms_rad*RAD_TO_MAS
    table["after_rms_mas"] = table.after_rms_rad*RAD_TO_MAS
    table.to_csv(root/"metrics.csv", index=False)
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 1, figsize=(11, 8))
        for name in ("ao_only", "ao_ff", "duplicate_ff", "extra_latency", "high_ao_gain"):
            d = data[name]
            subset = (d.time_s >= 4) & (d.time_s < 4.3)
            axes[0].plot(d.time_s[subset], d.residual_x[subset]*RAD_TO_MAS, label=name)
            scored = d.time_s >= 4
            f, psd = signal.welch(d.residual_x[scored].to_numpy(), fs=2000, nperseg=min(16000, np.sum(scored)))
            axes[1].semilogy(f, psd*RAD_TO_MAS**2, label=name)
        axes[0].set(xlabel="Time [s]", ylabel="Synthetic WFS X residual [mas]")
        axes[1].set(xlim=(10, 250), xlabel="Frequency [Hz]", ylabel="Sampled 2-kHz PSD [mas²/Hz]")
        axes[0].legend(ncol=3)
        figure.suptitle("SYNTHETIC AO INTERACTIONS — no measured MagAO-X loop parameters")
        figure.tight_layout(); figure.savefig(root/"interactions.png"); plt.close(figure)
    return table


def main():
    """Run selected offline audits; all output is ordinary local diagnostic files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-directory", type=Path)
    parser.add_argument("--ao-directory", type=Path)
    parser.add_argument("--archived-pupil-directory", type=Path)
    parser.add_argument("--projector-columns", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    if args.shadow_directory:
        report["shadow"] = audit_directory(args.shadow_directory)
    if args.archived_pupil_directory:
        report["archived_pupil"] = archived_calibration(args.archived_pupil_directory, args.output, args.projector_columns)
    if args.ao_directory:
        report["ao_cases"] = len(analyze_ao(args.ao_directory, not args.no_plots))//2
    if not report:
        parser.error("select at least one offline audit")
    (args.output/"shadow_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
