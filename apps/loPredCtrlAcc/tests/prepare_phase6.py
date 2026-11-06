r"""\file prepare_phase6.py
\brief Materialize immutable Phase 6 numerical sources for independent optimization regression tests.
"""
import argparse
from pathlib import Path
import subprocess

REFERENCE = "352446f83737573023870096d2a5388859afcd3e"
ROOT = Path(__file__).resolve().parents[3]
NAMES = ("ar_controller", "recursive_least_squares", "qrd_rls", "utils", "shadow_controller")


def main():
    """Read pinned local Git objects without a checkout, network access or source modifications."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        for suffix in (".cpp", ".hpp"):
            filename = name + suffix
            source = f"{REFERENCE}:apps/loPredCtrlAcc/{filename}"
            data = subprocess.check_output(["git", "show", source], cwd=ROOT)
            target = args.output / filename
            if not target.exists() or target.read_bytes() != data:
                target.write_bytes(data)
    (args.output / "snapshot.stamp").write_text(REFERENCE + "\n")
    print(f"Phase 6 numerical oracle: {REFERENCE}")


if __name__ == "__main__":
    main()
