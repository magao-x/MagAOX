r"""\file simulation_diagnose.py
\brief Focused adaptation/channel-01 audit of the existing simulation; all truth and batch solves are post-hoc.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.linalg import solve
from simulation_demo import APP, TESTS, aosim_methods, bridge, simulate
from vibration_report import pair, FS
from matplotlib import pyplot as plt


def trace(path):
    """Read complete unfiltered traces, preserving held and invalid frames."""
    with path.open() as f:
        names = f.readline().strip().split(",")
        return np.loadtxt(f, delimiter=",", dtype=[(n, float) for n in names])


def batch_reference(b, mode):
    """Reconstruct identical actual regressors and compare QR to independently accumulated double weighted LS."""
    y = pair(b, "yx", "yy").astype(np.float32).astype(float)
    u = pair(b, "applied_x", "applied_y").astype(np.float32).astype(float)
    a = pair(b, "ax", "ay").astype(np.float32).astype(float)
    p = 28 + (10 if mode == 4 else 0)
    gamma = float(np.float32(.9995))
    info, rhs = np.eye(p) * float(np.float32(.001)), np.zeros((6, p))
    expected = np.zeros((6, 6))
    expected[:2, 2:4] = expected[2:4, 4:6] = np.eye(2)
    rows = []
    for i in range(8, len(b) - 1):
        origin = i - 3
        phi = np.concatenate([y[origin-j] for j in range(6)] + [u[origin-j-1] for j in range(5)] +
                             ([a[origin-j] for j in range(5)] if mode == 4 else []) +
                             [u[origin+2-j] for j in range(3)])
        target = np.concatenate([y[i-j] for j in range(3)])
        info = gamma * info + np.outer(phi, phi)
        rhs = gamma * rhs + np.outer(target, phi)
        if i + 1 in (2399, 5400, 10399):
            weights = solve(info, rhs.T, assume_a="pos").T
            scales = np.sqrt(np.diag(info))
            rows.append(dict(frame=i+1, native_control_error=float(b["control_error"][i+1]),
                             batch_control_error=float(np.linalg.norm(weights[:, -6:] - expected)),
                             normalized_design_condition=float(np.sqrt(np.linalg.cond(info/scales[:, None]/scales[None, :]))),
                             rank=int(b["rank"][i+1]), fit_rmse=float(b["fit_rmse"][i+1])))
    return rows


def summarize(b):
    """Check actual consumed inputs and optical delay, and retain early/late performance and clipping."""
    request, consumed = pair(b, "request_x", "request_y"), pair(b, "applied_x", "applied_y")
    dm = pair(b, "dm_x", "dm_y")
    y = pair(b, "yx", "yy")
    return dict(first_scored_second_rms=float(np.sqrt(np.mean(y[3400:5400]**2))),
                last_second_rms=float(np.sqrt(np.mean(y[-2000:]**2))),
                request_consumed_max_error=float(np.max(np.abs(request-consumed))),
                two_frame_response_max_error=float(np.max(np.abs(dm[2:]-consumed[:-2]))),
                peak_command=float(np.max(np.abs(request[2400:]))),
                limit_fraction=float(np.mean(np.any(np.abs(request[2400:]) >= 4-1e-6, axis=1))),
                first_version=int(b["version"][2400]), last_version=int(b["version"][-1]),
                last_evidence_sequence=int(b["model_sequence"][-1]))


def rms_plot(records, output):
    """Compare complete adaptive and calibrated-fit RMS time histories rather than selecting a favorable segment."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for axis, scale in zip(axes, (1, 100)):
        for policy, mode, label, color, style in (("adaptive", 0, "No DDSPC", "#555555", "-"),
                ("adaptive", 3, "Adaptive standard", "#0072b2", "--"),
                ("adaptive", 4, "Adaptive assisted 0/1", "#009e73", "--"),
                ("calibrated", 3, "Calibrated standard", "#0072b2", "-"),
                ("calibrated", 4, "Calibrated assisted 0/1", "#009e73", "-")):
            a = records[policy, scale]
            b = a[(a["controller"] == mode) & (a["frame"] >= 2400)]
            bins = np.arange(0, len(b)-99, 100)
            y = pair(b, "yx", "yy")
            axis.plot((b["frame"][bins+50]-2400)/FS,
                      [np.sqrt(np.mean(y[i:i+100]**2)) for i in bins], color=color, linestyle=style, label=label)
        axis.set(title=f"Source amplitude {scale}×; actual ADC channels 0 and 1", ylabel="50-ms residual RMS (normalized)")
        axis.grid(alpha=.25)
    axes[0].legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Time from end of calibration (s)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / ("rms_vs_time."+ext), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    """Optionally reproduce both existing training settings and write the bounded likely-cause investigation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--magaox-repo", type=Path, default=Path.home()/"source/MagAOX")
    parser.add_argument("--output", type=Path, default=TESTS/"results/channel01_validation")
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.run:
        for policy, options in (("adaptive", []), ("calibrated", ["--training-end", "2400"])):
            subprocess.run([sys.executable, str(TESTS/"simulation_demo.py"), "--magaox-repo", str(args.magaox_repo),
                            "--cxx", args.cxx, "--accel-channels", "0", "1", "--output", str(out/policy)] + options, check=True)
    records, findings = {}, {}
    for policy, suffix in (("adaptive", ""), ("calibrated", "-train2400")):
        build = APP/"build"/("simulation-"+Path(args.cxx).name+suffix)
        for scale in (1, 100):
            a = trace(build/f"trace_scale_{scale}.csv"); records[policy, scale] = a
            findings[f"{policy}_{scale}"] = {str(mode): summarize(a[a["controller"] == mode]) for mode in (0, 3, 4)}
            if policy == "adaptive":
                for mode in (3, 4):
                    findings[f"{policy}_{scale}"][str(mode)]["weighted_ls"] = batch_reference(a[a["controller"]==mode], mode)
    # One excitation contrast, using the unchanged existing probe mechanism only in this simulation.
    build = APP/"build"/("simulation-"+Path(args.cxx).name)
    source = np.load(build/"source_scale_100.npz")
    lib, methods = bridge(build/"libddspc_sim.so"), aosim_methods(args.magaox_repo.resolve())
    findings["independent_probe_control"] = {}
    for mode in (3, 4):
        b = simulate(lib, methods, source["adc"], source["pointing"], source["units"], mode=mode, evaluation_probe=.3)
        findings["independent_probe_control"][str(mode)] = {str(k): float(b["control_error"][k]) for k in (2399,5400,10399)}
    findings["calibration_weight_after_8000_updates"] = float(np.float32(.9995))**8000
    rms_plot(records, out)
    (out/"diagnosis.json").write_text(json.dumps(findings, indent=2)+"\n")
    rows = []
    for policy in ("adaptive", "calibrated"):
        with (out/policy/"metrics.csv").open() as f:
            for row in csv.DictReader(f):
                row["training_policy"] = policy; rows.append(row)
    with (out/"comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print("Wrote complete RMS histories, independent weighted-LS checks, and retained adaptive/calibrated results:", out)


if __name__ == "__main__":
    main()
