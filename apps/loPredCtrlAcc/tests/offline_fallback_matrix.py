r"""\file offline_fallback_matrix.py
\brief Reproducible private offline comparison matrix; invokes only the standalone numerical executable.
"""
import argparse
import csv
from pathlib import Path
import subprocess


def scenarios():
    """Keep physical forcing, WFS noise, probing, limits and delays common within each policy comparison."""
    cases = [("normal", [])]
    for duration in (1, 5, 25, 100):
        cases.append((f"drop_{duration}ms", ["--drop-start", 9600, "--drop-count", duration*8]))
    cases.extend([
        ("intermittent", ["--drop-start", 9600, "--drop-count", 8, "--drop-period", 400]),
        ("uncorrelated", ["--uncorrelated-adc", 1]),
        ("coupling_0", ["--coupling", 0]),
        ("coupling_0.3", ["--coupling", .3]),
        ("coupling_3", ["--coupling", 3]),
        ("lambda_0.01", ["--lambda", .01]),
        ("lambda_1", ["--lambda", 1]),
        ("horizon_1", ["--future", 1]),
        ("horizon_6", ["--future", 6]),
        ("jitter", ["--jitter-ns", 10000]),
        ("latency", ["--latency-ns", 100000, "--latency-jitter-ns", 80000]),
        ("late_with_lag", ["--latency-ns", 200000, "--latency-jitter-ns", 100000,
                           "--window-lag-ns", 375000, "--feature-max-age-ns", 750000]),
        ("stale_delivery", ["--latency-ns", 2000000]),
        ("drift_minus200", ["--adc-drift-ppm", -200]),
        ("drift_plus200", ["--adc-drift-ppm", 200]),
        ("combined_timing", ["--jitter-ns", 10000, "--latency-ns", 100000,
                             "--latency-jitter-ns", 80000, "--adc-drift-ppm", 200]),
        ("wfs_loss", ["--wfs-invalid-start", 2400, "--wfs-invalid-count", 50]),
        ("seed722", ["--seed", 722]),
        ("seed723", ["--seed", 723]),
    ])
    return cases


def main():
    """Write new per-scenario artifacts plus one comparison CSV, refusing any existing output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    if binary.name != "adc_ddspc_sim":
        parser.error("select the standalone adc_ddspc_sim numerical executable")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, options in scenarios():
        command = [str(binary), *map(str, options), "--output", str(args.output/name)]
        result = subprocess.run(command, text=True, capture_output=True, timeout=120, check=True)
        metrics = list(csv.DictReader(line for line in result.stdout.splitlines() if not line.startswith("#")))
        for row in metrics:
            if int(row["numerical_fault_frames"]) or int(row["gate_violations"]):
                raise RuntimeError(f"{name}: invalid controller outcome: {row}")
            rows.append({"scenario": name, **row})
        print(name, "standard variance", metrics[1]["variance"], "assisted variance", metrics[2]["variance"], flush=True)
    with (args.output/"comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
