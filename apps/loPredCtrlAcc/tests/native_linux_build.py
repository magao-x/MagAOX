r"""\file native_linux_build.py
\brief Build and link real loPredCtrlAcc in fresh source snapshots; never run or install the application.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def run(arguments, directory, environment, log):
    """Record and execute a build-only command, propagating its failure."""
    log.write((json.dumps(arguments) + "\n").encode())
    log.flush()
    subprocess.run(arguments, cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    """Guard the host, stage current source bytes, and build separate Legacy and strict-shadow profiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", required=True, type=Path, help="New output directory; must not exist")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if platform.system() != "Linux" or socket.gethostname().lower().startswith("exao"):
        raise SystemExit("Requires a non-instrument Linux development environment; nothing was built")
    if not 1 <= args.jobs <= 64:
        raise SystemExit("--jobs must be in [1,64]")
    compiler = shutil.which(args.cxx)
    generator = shutil.which("gengithead.sh")
    if not compiler or not generator:
        raise SystemExit("Provision the compiler and MagAO-X/mxlib build dependencies first")
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT, env=environment
    ).decode().split("\0")
    destination = args.build_root.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "host": socket.gethostname(), "platform": platform.platform(),
        "compiler": subprocess.check_output([compiler, "--version"]).decode(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "source": str(ROOT), "files": {}, "profiles": {},
    }
    for profile in ("legacy", "shadow"):
        stage = destination / profile
        stage.mkdir()
        for name in sorted(set(paths)):
            if not name:
                continue
            source = ROOT / name
            if source.is_dir() or not source.exists():
                continue  # Deleted working files remain deleted; Git submodules require separate provisioning.
            if source.is_symlink():
                raise RuntimeError(f"Review source symlink before staging: {name}")
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            manifest["files"][name] = hashlib.sha256(target.read_bytes()).hexdigest()
        # Preserve real source metadata while keeping all generated headers inside the disposable stage.
        # The original generator only reads Git; optional Git index refresh locks are disabled.
        tools = stage / "build-tools"
        tools.mkdir()
        wrapper = tools / "gengithead.sh"
        wrapper.write_text(
            "#!/usr/bin/env python3\nimport os, pathlib, subprocess, sys\n"
            f"root = pathlib.Path({str(ROOT)!r})\nstage = pathlib.Path({str(stage)!r})\n"
            "output = pathlib.Path(sys.argv[2]).resolve()\n"
            "if not output.is_relative_to(stage): raise SystemExit('Generated header escaped build stage')\n"
            f"subprocess.run([{generator!r}, str(root), str(output), sys.argv[3]], cwd=root, check=True)\n"
        )
        wrapper.chmod(0o755)
        build_environment = dict(environment, PATH=str(tools) + os.pathsep + environment["PATH"])
        optimization = "-fno-fast-math" if profile == "shadow" else "-ffast-math"
        common = [f"CXX={compiler}", "EDT=false", "NEED_CUDA=no", f"OPTIMIZE={optimization}"]
        manifest["profiles"][profile] = {"make_variables": common, "linked": False}
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with (destination / f"{profile}.log").open("wb") as log:
            for directory in ("INDI/liblilxml", "INDI/libcommon"):
                run(["make", "-C", directory, f"-j{args.jobs}", *common, "all"], stage, build_environment, log)
            run(["make", f"-j{args.jobs}", *common, "libs_all"], stage, build_environment, log)
            shadow = ["DDSPC_SHADOW_BUILD=yes"] if profile == "shadow" else []
            # The existing app makefile's phony Git-header generation is not an object dependency;
            # keep that final make serial so it cannot race the compile reading the generated header.
            run(["make", "-C", "apps/loPredCtrlAcc", "-j1", *common, *shadow, "all"],
                stage, build_environment, log)
        binary = stage / "apps/loPredCtrlAcc/loPredCtrlAcc"
        if not binary.is_file():
            raise RuntimeError("Native link did not create loPredCtrlAcc")
        manifest["profiles"][profile]["linked"] = True
        manifest["profiles"][profile]["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Linked {profile}: {binary}; application was not executed or installed")


if __name__ == "__main__":
    main()
