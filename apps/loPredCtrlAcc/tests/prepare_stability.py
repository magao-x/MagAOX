r"""\file prepare_stability.py
\brief Instantiate a pinned offline symmetric-covariance RLS candidate without editing production numerical sources.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

REFERENCE = "2fcf61a108c8f3d9c3d237a053f4e0634f0f4e6f"
ROOT = Path(__file__).resolve().parents[3]
NAMES = ("ar_controller", "recursive_least_squares", "qrd_rls", "utils", "shadow_controller")


def main():
    """Copy local committed objects and apply exactly one declared invariant-restoration operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    files = [name + ext for name in NAMES for ext in (".hpp", ".cpp")]
    files += ["tests/async_shadow.hpp", "tests/async_shadow.cpp", "training_profile.hpp"]
    hashes = {}
    for name in files:
        data = subprocess.check_output(["git", "show", f"{REFERENCE}:apps/loPredCtrlAcc/{name}"], cwd=ROOT)
        if name == "recursive_least_squares.cpp":
            needle = b"    inverse_covariance -= covarianceUpdate;"
            assert data.count(needle) == 1
            data = data.replace(needle, needle + b"\n    // Offline candidate: restore the exact symmetric covariance invariant.\n"
                b"    inverse_covariance = (realT(.5) * (inverse_covariance + inverse_covariance.transpose())).eval();")
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != data:
            target.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(dict(reference=REFERENCE, sources=hashes,
        transformation="After causal covariance subtraction only: P=(P+P.transpose())/2"), indent=2) + "\n")
    (args.output / "snapshot.stamp").write_text(REFERENCE + "\n")
    loop = subprocess.check_output(["git", "show", f"{REFERENCE}:apps/loPredCtrlAcc/tests/vibration.cpp"], cwd=ROOT)
    (args.output / "phase10_loop.cpp").write_bytes(loop)


if __name__ == "__main__":
    main()
