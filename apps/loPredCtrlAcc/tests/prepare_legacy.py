r"""\file prepare_legacy.py
\brief Extract an immutable pre-accelerometer reference without checking out or editing source.
"""

import argparse
from pathlib import Path
import subprocess


REFERENCE = "f59d68be7311fb2d3138ef2a4ce269d091ee49e7"
TREE = "d37d0a7e416204b4ab844fb8caf1ba83e0ebff4a"
ROOT = Path(__file__).resolve().parents[3]


def git(*args):
    """Read local Git objects; never fetch, switch branches, or change the index."""
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def write_changed(path, data):
    """Write generated fixture files only when their bytes differ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


def main():
    """Verify the restored directory and optionally generate stream-free compile fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--demo-dir", type=Path)
    parser.add_argument("--demo-name", action="append", choices=("testRLS", "testPredCtrl"))
    args = parser.parse_args()
    for revision in (REFERENCE, "HEAD"):
        actual = git("rev-parse", f"{revision}:apps/loPredCtrl").decode().strip()
        if actual != TREE:
            raise RuntimeError(f"{revision} loPredCtrl tree changed: {actual}; review compatibility before updating the pin")
    paths = git("ls-tree", "-r", "--name-only", REFERENCE, "--", "apps/loPredCtrl").decode().splitlines()
    for name in paths:
        original = git("show", f"{REFERENCE}:{name}")
        current = (ROOT / name).read_bytes()
        if current != original:
            raise RuntimeError(f"Working-tree source differs from the original: {name}")
        if args.output:
            write_changed(args.output / "original" / name, original)
            write_changed(args.output / "restored" / name, current)
    if args.output:
        # Historical app includes use repo-relative paths. These guards permit only the existing offline doubles.
        stub = b'''/** \\file libMagAOX.hpp
 * \\brief Generated historical-test include guard; native application/hardware libraries are forbidden.
 */
#ifndef libMagAOX_hpp
#error Include app_test_support.hpp before historical app headers
#endif
'''
        version = b'''/** \\file magaox_git_version.h
 * \\brief Generated historical-test version guard; real metadata is supplied by the offline fixture.
 */
#ifndef MAGAOX_CURRENT_SHA1
#error Historical app tests require offline version metadata
#endif
'''
        for name in ("original", "restored"):
            write_changed(args.output / name / "libMagAOX/libMagAOX.hpp", stub)
            write_changed(args.output / name / "magaox_git_version.h", version)
        write_changed(args.output / "snapshot.stamp", (REFERENCE + "\n" + TREE + "\n").encode())
    print(f"Legacy reference {REFERENCE}: {len(paths)} files match restored loPredCtrl byte-for-byte (tree {TREE})")
    if args.demo_dir:
        for implementation in ("original", "current"):
            for name in args.demo_name or ("testRLS", "testPredCtrl"):
                executable = (args.demo_dir / f"demo_{implementation}_{name}").resolve()
                work = args.demo_dir / "demo-output" / f"{implementation}_{name}"
                (work / "tests").mkdir(parents=True, exist_ok=True)
                with (work / "output.log").open("w") as output:
                    subprocess.run([str(executable)], cwd=work, stdout=output, stderr=subprocess.STDOUT, check=True)
                print(f"Offline demo {implementation}/{name}: exit 0; log {work / 'output.log'}")


if __name__ == "__main__":
    main()
