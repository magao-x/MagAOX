r"""\file run_phase11.py
\brief Run the offline Phase 11 matrix and declared recovery ablations; never launch a MagAO-X application.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ABLATION_CASES = ("broadband", "mixed", "reset_qr", "missing_qr", "overflow_qr", "stale_transient_qr",
                  "excitation_loss_qr", "wrong_sign_qr", "command_limit_qr", "publication_6_qr")


def main():
    """Execute complete primary matrices and explicitly limited neutral/guard ablations with exact configurations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strategies", nargs="+", default=["baseline", "symmetric", "recover", "symmetric_recover", "neutral", "guard"])
    args = parser.parse_args()
    if args.binary.name != "ddspc_vibration":
        parser.error("Use only the separately built offline ddspc_vibration executable")
    for strategy in args.strategies:
        destination = args.output / strategy
        destination.mkdir(parents=True, exist_ok=True)
        cases = ABLATION_CASES if strategy in ("neutral", "guard") else ("all",)
        rows, commands = [], []
        with (destination / "execution.log").open("w") as log:
            for case in cases:
                command = [str(args.binary.resolve()), str(destination.resolve()), case, strategy]
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                commands.append(command)
                with (destination / "config.csv").open() as src:
                    rows.extend(csv.DictReader(src))
                print(f"completed {strategy}: {case}", flush=True)
        with (destination / "config.csv").open("w") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        manifest = dict(binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(), commands=commands,
                        cases=len(rows), ablation_only=strategy in ("neutral", "guard"))
        (destination / "execution.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
