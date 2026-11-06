r"""\file phase11_report.py
\brief Quantify offline estimator/recovery candidates, retain failures, verify references, and render scientific evidence.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ddspc-phase11-matplotlib")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vibration_report import metrics, pair, spectra

PERIOD_MS = .5
PRIMARY = ("baseline", "symmetric", "recover", "symmetric_recover")


def digest(path):
    """Hash a finite artifact in bounded chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def read_config(root):
    """Return complete declared configurations; no case is inferred from its measured performance."""
    with (root / "config.csv").open() as f:
        return list(csv.DictReader(f))


def read_trace(path):
    """Parse declared numerical columns directly, retaining every NaN and diagnostic row."""
    with path.open() as src:
        names = src.readline().strip().split(",")
        return np.loadtxt(src, delimiter=",", dtype=[(name, np.float64) for name in names])


def prefix_equal(old, new):
    """Require every original numerical/text payload byte unchanged, ignoring only newly appended diagnostics."""
    rows = 0
    with old.open("rb") as a, new.open("rb") as b:
        header = a.readline().rstrip(b"\r\n")
        new_header = b.readline().rstrip(b"\r\n")
        assert new_header.startswith(header + b",") or new_header == header
        extra = new_header.count(b",") - header.count(b",")
        while True:
            x, y = a.readline(), b.readline()
            if not x or not y:
                assert not x and not y, (old, "different lengths")
                break
            y = y.rstrip(b"\r\n")
            if extra:
                y = y.rsplit(b",", extra)[0]
            assert x.rstrip(b"\r\n") == y, (old.name, rows)
            rows += 1
    return rows


def recovery_metrics(data, cfg, model):
    """Measure whole trajectories and stable post-fault return without scoring a probe as a model correction."""
    b = data[data["controller"] == model]
    train = int(cfg["train"])
    p = (int(cfg["future"]) + 2 * int(cfg["history"]) - 1) * 2
    if model == 4:
        p += 2 * int(cfg["accel_history"])
    finish = -1
    for key, length in (("reset_frame", 0), ("missing_frame", int(cfg["missing_frames"])),
                        ("stall_frame", int(cfg["stall_frames"])), ("freeze_frame", int(cfg.get("freeze_frames", 0)))):
        if int(cfg.get(key, -1)) >= 0:
            finish = max(finish, int(cfg[key]) + length)
    event = finish >= 0
    run, recovered = 0, None
    for i in range(max(train, finish), len(b)):
        accepted = b["correction_accepted"][i] != 0
        tracking = b["recovery"][i] == 0
        run = run + 1 if accepted and tracking else 0
        if run >= p:
            recovered = i
            break
    evaluated = b[train:]
    return dict(correction_availability=float(evaluated["correction_accepted"].mean()),
                probe_frames=int(np.sum(evaluated["action"] == 4)),
                neutral_frames=int(np.sum(evaluated["action"] == 3)),
                event=event, fault_end_frame=finish,
                recovery_ms=(recovered - finish + 1) * PERIOD_MS if event and recovered is not None else float("nan"),
                recovery_failure=int(event and recovered is None),
                tail_rms=float(np.sqrt(np.mean(pair(b[-1000:], "yx", "yy") ** 2))),
                confidence_bound_max=float(b["confidence_bound"].max()))


def collect(root, compiler, out):
    """Score every complete strategy/case and retain data only for the required audit plots."""
    rows, selected, manifest = [], {}, {}
    for directory in sorted(root.iterdir()):
        if not (directory / "config.csv").exists():
            continue
        strategy = directory.name
        for cfg in read_config(directory):
            case = cfg["scenario"]
            path = directory / (case + ".csv")
            a = read_trace(path)
            assert len(a) == 5 * (int(cfg["train"]) + int(cfg["evaluate"]))
            assert not a["invalid_changes"].any()
            assert np.isfinite(pair(a, "request_x", "request_y")).all()
            assert np.max(np.abs(pair(a, "request_x", "request_y"))) <= float(cfg["limit"])
            for row in metrics(a, cfg):
                row.update(strategy=strategy, compiler=compiler)
                row.update(recovery_metrics(a, cfg, row["controller"]))
                row["amplifies_no_control"] = int(row["variance_reduction"] < 0)
                rows.append(row)
            if case in ("broadband", "broadband_qr", "reset_qr", "missing_qr", "overflow_qr", "excitation_loss_qr"):
                if strategy in ("baseline", "symmetric", "symmetric_recover"):
                    selected[strategy, case] = (a, cfg)
            manifest[str(path.relative_to(root))] = digest(path)
        print(f"scored {compiler} {strategy}", flush=True)
    with (out / f"phase11-metrics-{compiler}.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows, selected, manifest


def save(fig, out, name):
    """Create standalone raster/vector scientific artifacts, with no instrument or interactive dependencies."""
    fig.savefig(out / (name + ".png"), dpi=140, bbox_inches="tight")
    fig.savefig(out / (name + ".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def figures(selected, diagnostics, out):
    """Expose estimator divergence, suppression and complete recovery timelines rather than masking held periods."""
    plt.rcParams.update({"font.size": 10})
    f, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for strategy, case, label, color in (("baseline", "broadband", "Original RLS", "#d55e00"),
          ("symmetric", "broadband", "Symmetric RLS", "#0072b2"),
          ("baseline", "broadband_qr", "Reference QR", "#009e73")):
        data, cfg = selected[strategy, case]
        b = data[data["controller"] == 4]; t = (b["frame"] - int(cfg["train"])) / 2000
        starts = np.arange(0, len(b) - 199, 200)
        axes[0].semilogy(t[starts], [np.sqrt(np.mean(pair(b[i:i + 200], "yx", "yy") ** 2)) for i in starts],
                         color=color, label=label)
        valid = b["valid"] != 0
        axes[1].semilogy(t[valid], b["theta_norm"][valid], color=color)
    axes[0].set(ylabel="0.1-s WFS residual RMS (synthetic units)", title="Offline assisted broadband: numerical estimator comparison")
    axes[1].set(xlabel="Time from calibration end (s)", ylabel="Coefficient norm (valid snapshots)")
    for ax in axes: ax.axvline(0, color="black", alpha=.4); ax.grid(alpha=.25)
    axes[0].legend(); save(f, out, "phase11-estimation")

    p = diagnostics / "phase11-replay-broadband-m4-gbase-pbase.csv"
    d = np.genfromtxt(p, delimiter=",", names=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for key, label, color in (("p_min", "Original float", "#d55e00"), ("sym_min", "Symmetric float", "#0072b2"),
                              ("double_min", "Double reference", "#009e73")):
        axes[0].plot(d["frame"] * .0005, d[key], color=color, label=label)
    axes[0].set_yscale("symlog", linthresh=1e-5)
    axes[0].set(ylabel="Minimum covariance eigenvalue", title="Identical recorded regressors: covariance and weighted-LS consistency")
    axes[0].axhline(0, color="black", alpha=.4); axes[0].legend()
    for key, label in (("theta_error_native", "Original"), ("theta_error_symmetric", "Symmetric"),
                       ("theta_error_qr", "QR"), ("theta_error_double", "Double")):
        axes[1].semilogy(d["frame"] * .0005, d[key], label=label)
    axes[1].set_ylabel("Coefficient error vs weighted LS"); axes[1].legend(ncol=4)
    axes[2].semilogy(d["frame"] * .0005, d["condition_info"], color="#555555")
    axes[2].set(xlabel="Virtual time (s)", ylabel="Information condition number")
    for ax in axes: ax.axvline(1.2, color="black", alpha=.4); ax.grid(alpha=.25)
    save(fig, out, "phase11-covariance")

    fig, ax = plt.subplots(figsize=(10, 4))
    for strategy, case, label, color in (("baseline", "broadband", "Original RLS", "#d55e00"),
          ("symmetric", "broadband", "Symmetric RLS", "#0072b2"),
          ("baseline", "broadband_qr", "Reference QR", "#009e73")):
        data, cfg = selected[strategy, case]
        b = data[(data["controller"] == 4) & (data["frame"] >= int(cfg["train"]) + 1000)]
        hz, psd = spectra(pair(b, "yx", "yy")); ax.semilogy(hz, psd, label=label, color=color)
    ax.set(xlim=(5, 300), xlabel="Frequency (Hz)", ylabel="Residual PSD (synthetic units²/Hz)", title="Offline assisted broadband: all measured residuals retained")
    ax.legend(); ax.grid(alpha=.25); save(fig, out, "phase11-psd")

    for case in ("reset_qr", "missing_qr", "overflow_qr", "excitation_loss_qr"):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for strategy, label, color in (("baseline", "Phase 10 hold", "#d55e00"),
                                        ("symmetric_recover", "Offline recovery", "#0072b2")):
            data, cfg = selected[strategy, case]; b = data[data["controller"] == 4]
            event = max(int(cfg[k]) for k in ("reset_frame", "missing_frame", "stall_frame", "freeze_frame"))
            indices = np.arange(max(event - 100, 0), min(event + 600, len(b) - 50), 10)
            axes[0].plot((indices - event) * .5, [np.sqrt(np.mean(pair(b[max(0, i - 49):i + 1], "yx", "yy") ** 2)) for i in indices],
                          color=color, label=label)
            area = (b["frame"] >= event - 100) & (b["frame"] < event + 600)
            axes[1].plot((b["frame"][area] - event) * .5, b["request_x"][area], color=color)
            axes[2].plot((b["frame"][indices] - event) * .5,
                          [b[max(0, i - 49):i + 1]["correction_accepted"].mean() for i in indices], color=color)
        axes[0].set(ylabel="Trailing 25-ms WFS residual RMS (SU)", title="Offline recovery: " + case)
        axes[1].set_ylabel("Tip request (synthetic DU)")
        axes[2].set(xlabel="Time from injected fault start (ms)", ylabel="Accepted correction fraction", ylim=(-.05, 1.05))
        axes[0].legend()
        for ax in axes: ax.axvline(0, color="black", alpha=.3); ax.grid(alpha=.25)
        save(fig, out, "phase11-recovery-" + case.removesuffix("_qr"))


def main():
    """Score complete data, verify exact references and emit transparent acceptance and source manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clang", required=True, type=Path)
    parser.add_argument("--gcc", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    rows, selected, hashes = collect(args.clang, "clang", out)
    gcc, _, gcc_hashes = collect(args.gcc, "gcc", out)
    checks = {}
    for compiler, root in (("clang", args.clang), ("gcc", args.gcc)):
        before = args.build / ("phase11-before-" + compiler)
        matched = sum(prefix_equal(before / (c["scenario"] + ".csv"), root / "baseline" / (c["scenario"] + ".csv"))
                      for c in read_config(before))
        qr = sum(prefix_equal(root / "baseline" / (c["scenario"] + ".csv"), root / "symmetric" / (c["scenario"] + ".csv"))
                 for c in read_config(root / "baseline") if int(c["qrd"]))
        checks[compiler] = dict(original_rows_byte_exact=matched, unchanged_qr_rows_byte_exact=qr)
    acceptance = dict(exact_references=checks, comparisons=[])
    for compiler, table in (("clang", rows), ("gcc", gcc)):
        baseline = {(r["scenario"], r["controller"]): r for r in table if r["strategy"] == "baseline"}
        for r in table:
            if r["controller"] not in (3, 4) or r["strategy"] == "baseline": continue
            b = baseline[r["scenario"], r["controller"]]
            acceptance["comparisons"].append(dict(compiler=compiler, strategy=r["strategy"], scenario=r["scenario"],
                model=r["controller"], rms_difference=r["rms"]-b["rms"],
                baseline_success=b["qrd"] == 1 and b["correction_availability"] >= .95,
                recovery_failure=r["recovery_failure"], baseline_recovery_failure=b["recovery_failure"]))
    (out / "phase11-acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
    configurations = {compiler: {p.name: read_config(p) for p in root.iterdir() if (p / "config.csv").exists()}
                      for compiler, root in (("clang", args.clang), ("gcc", args.gcc))}
    (out / "phase11-config.json").write_text(json.dumps(configurations, indent=2) + "\n")
    figures(selected, args.build, out)
    repo = Path(__file__).resolve().parents[3]
    files = list(Path(__file__).parent.glob("*stability*")) + [Path(__file__), Path(__file__).parent / "run_phase11.py"]
    files += [Path(__file__).parent / x for x in ("vibration.cpp", "vibration.hpp", "vibration_main.cpp", "rls_diagnostics.cpp", "fast_math_probe.cpp", "Makefile.offline", "Makefile.vibration")]
    files = [p for p in files if p.is_file()]
    manifest = dict(head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip(),
                    host=platform.node(), platform=platform.platform(), python=platform.python_version(),
                    numpy=np.__version__, matplotlib=matplotlib.__version__,
                    sources={str(p.relative_to(repo)): digest(p) for p in files},
                    executions={compiler: {p.parent.name: json.loads(p.read_text()) for p in root.glob("*/execution.json")}
                                for compiler, root in (("clang", args.clang), ("gcc", args.gcc))},
                    clang_traces=hashes, gcc_traces=gcc_hashes)
    (out / "phase11-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(rows)} Clang / {len(gcc)} GCC rows, exact-reference checks and seven PNG/PDF plots")


if __name__ == "__main__":
    main()
