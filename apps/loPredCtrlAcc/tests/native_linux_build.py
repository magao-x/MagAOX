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


def stage_sources(paths, snapshot):
    """Freeze working source bytes once, rejecting symlinks/submodules rather than silently omitting them."""
    hashes = {}
    snapshot.mkdir()
    for name in sorted(set(paths)):
        if not name:
            continue
        source = ROOT / name
        if source.is_symlink() or source.is_dir():
            raise RuntimeError(f"Source symlinks and Git submodules need an explicit staging rule: {name}")
        if not source.exists():
            continue  # Preserve working-tree deletions.
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    return hashes


def write_wrapper(stage, version):
    """Serve frozen real Git metadata, confining all generated headers to this disposable stage."""
    stage, version = stage.resolve(), version.resolve()
    tools = stage / "build-tools"
    tools.mkdir()
    wrapper = tools / "gengithead.sh"
    wrapper.write_text(
        "#!/usr/bin/env python3\nimport pathlib, shutil, sys\n"
        f"stage = pathlib.Path({str(stage)!r})\nversion = pathlib.Path({str(version)!r})\n"
        "output = pathlib.Path(sys.argv[2]).resolve()\n"
        "if sys.argv[3] != 'MAGAOX': raise SystemExit('Unsupported Git metadata prefix')\n"
        "if not output.is_relative_to(stage): raise SystemExit('Generated header escaped build stage')\n"
        "shutil.copyfile(version, output)\n"
    )
    wrapper.chmod(0o755)
    return tools


def main():
    """Guard the host, freeze sources, and link requested private profiles without executing applications."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", required=True, type=Path, help="New output directory; must not exist")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--observer-only", action="store_true", help="Build only the guarded native observation executable")
    args = parser.parse_args()
    if platform.system() != "Linux" or socket.gethostname().lower().startswith("exao"):
        raise SystemExit("Requires a non-instrument Linux development environment; nothing was built")
    if not 1 <= args.jobs <= 64:
        raise SystemExit("--jobs must be in [1,64]")
    destination = args.build_root.resolve()
    if destination == ROOT or destination.is_relative_to(ROOT):
        raise SystemExit("--build-root must be outside the source tree")
    compiler = shutil.which(args.cxx)
    generator = shutil.which("gengithead.sh")
    if not compiler or not generator:
        raise SystemExit("Provision the compiler and MagAO-X/mxlib build dependencies first")
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT, env=environment
    ).decode().split("\0")
    destination.mkdir(parents=True, exist_ok=False)
    snapshot = destination / "source"
    hashes = stage_sources(paths, snapshot)
    manifest = {
        "host": socket.gethostname(), "platform": platform.platform(),
        "compiler": subprocess.check_output([compiler, "--version"]).decode(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "source": str(ROOT), "files": hashes, "profiles": {},
    }
    version = destination / "source_git_version.h"
    with (destination / "prepare.log").open("wb") as log:
        run([generator, str(ROOT), str(version), "MAGAOX"], ROOT, environment, log)
    manifest["git_header_sha256"] = hashlib.sha256(version.read_bytes()).hexdigest()
    failed = False
    for profile in (("observer",) if args.observer_only else ("legacy", "shadow")):
        stage = destination / profile
        shutil.copytree(snapshot, stage)
        tools = write_wrapper(stage, version)
        build_environment = dict(environment, PATH=str(tools) + os.pathsep + environment["PATH"])
        guarded = profile != "legacy"
        optimization = "-fno-fast-math" if guarded else "-ffast-math"
        common = [f"CXX={compiler}", "CXXSTD=-std=c++20", "EDT=false", "NEED_CUDA=no", f"OPTIMIZE={optimization}"]
        result = {"make_variables": common, "linked": False, "status": "building"}
        manifest["profiles"][profile] = result
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        try:
            with (destination / f"{profile}.log").open("wb") as log:
                for directory in ("INDI/liblilxml", "INDI/libcommon"):
                    run(["make", "-C", directory, f"-j{args.jobs}", *common, "all"], stage, build_environment, log)
                run(["make", f"-j{args.jobs}", *common, "libs_all"], stage, build_environment, log)
                shadow = ["DDSPC_SHADOW_BUILD=yes", "DDSPC_SHADOW_ONLY=yes"] if guarded else []
                target = "native-observe" if profile == "observer" else "all"
                # Serialize phony Git-header generation and the object compilation that reads it.
                run(["make", "-C", "apps/loPredCtrlAcc", "-j1", *common, *shadow, target],
                    stage, build_environment, log)
            binary = stage / "apps/loPredCtrlAcc" / ("loPredCtrlAcc_native_observe" if profile == "observer" else "loPredCtrlAcc")
            if not binary.is_file():
                raise RuntimeError("Native link did not create the requested private executable")
            result.update(linked=True, status="linked", shadow_only=guarded,
                          binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
            print(f"Linked {profile}: {binary}; application was not executed or installed")
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            failed = True
            result.update(status="failed", error=str(error))
            print(f"Build failed for {profile}; see {destination / (profile + '.log')}")
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
