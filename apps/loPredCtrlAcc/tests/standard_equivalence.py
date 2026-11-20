#!/usr/bin/env python3
"""T1: compare standard-mode DM output and decision timing with a baseline commit.

Exports the baseline tree with ``git archive``, compiles tests/standard_trace_test.cpp against the baseline and
the working tree with the same compiler and flags, runs both, and requires the rendered slot values to be
bit-for-bit identical. Decision timing is compared alternately over several runs.

Run from anywhere: python3 apps/loPredCtrlAcc/tests/standard_equivalence.py [--base 1ff631bb]
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys

APP = pathlib.Path(__file__).resolve().parents[1]
ROOT = APP.parents[1]
# Sources common to both trees; optional files are added when present in a tree.
SOURCES = [
    "../../tests/testMain.cpp", "ar_controller.cpp", "recursive_least_squares.cpp", "qrd_rls.cpp", "utils.cpp",
    "adc_features.cpp", "shadow_policy.cpp", "shadow_pipeline.cpp", "shadow_controller.cpp", "causal_input.cpp",
    "adc_input.cpp", "adc_reader.cpp", "native_observation.cpp", "native_observer.cpp", "phase_shadow.cpp",
    "native_pipeline.cpp", "synco_shadow.cpp", "synchronized_inputs.cpp", "multirate_shadow.cpp",
    "online_shadow.cpp", "tests/adc_transport_fixture.cpp", "../../INDI/libcommon/IndiProperty.cpp",
    "../../INDI/libcommon/IndiElement.cpp", "../../INDI/libcommon/TimeStamp.cpp", "tests/shadow_replay.cpp",
    "tests/offline_adc.cpp", "tests/qualified_shadow_replay.cpp", "tests/operational_platform.cpp",
    "tests/standard_trace_test.cpp",
]
OPTIONAL = ["multirate_ddspc.cpp"]


def pkg(*args):
    return subprocess.run(["pkg-config", *args], check=True, capture_output=True, text=True).stdout.strip()


def build(app: pathlib.Path, build_dir: pathlib.Path, cxx: str) -> pathlib.Path:
    sources = [str((app / s).resolve()) for s in SOURCES + [o for o in OPTIONAL if (app / o).exists()]]
    libdir = pkg("--variable=libdir", "mxlib")
    subprocess.run(
        ["make", "-s", "-j8", "-f", str(app / "tests/Makefile.offline"), f"BUILD_DIR={build_dir}", f"CXX={cxx}",
         "DDSPC_MINIMAL_APP_TESTS=yes", "CXXFLAGS=-O2 -g -std=c++20 -Wall -Wextra -fno-fast-math -DDDSPC_OPERATIONAL_BUILD",
         f"LDFLAGS=-Wl,-rpath,{libdir}", f"LDLIBS=-pthread {pkg('--libs-only-L', '--libs-only-l', 'mxlib')}",
         f"SOURCES={' '.join(sources)}", str(build_dir / "ddspc_offline_test")],
        cwd=app, check=True)
    return build_dir / "ddspc_offline_test"


def run(binary: pathlib.Path, trace: pathlib.Path) -> dict:
    env = dict(os.environ, DDSPC_STANDARD_TRACE=str(trace))
    subprocess.run([str(binary), "[standard-trace]"], env=env, check=True, capture_output=True)
    timing = {}
    for line in pathlib.Path(str(trace) + ".timing").read_text().splitlines():
        name, *values = line.split()
        timing[name] = [float(v) for v in values]  # mean, p50, p99, max [us]
    return timing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="1ff631bb")
    parser.add_argument("--work", default=str(APP / "build/standard-equivalence"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()
    work = pathlib.Path(args.work).resolve()
    base_root = work / "base"
    if base_root.exists():
        shutil.rmtree(base_root)
    base_root.mkdir(parents=True)
    archive = subprocess.Popen(["git", "archive", args.base], cwd=ROOT, stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(base_root)], stdin=archive.stdout, check=True)
    if archive.wait():
        raise RuntimeError(f"git archive {args.base} failed")
    base_app = base_root / "apps/loPredCtrlAcc"
    shutil.copy2(APP / "tests/standard_trace_test.cpp", base_app / "tests/standard_trace_test.cpp")
    binaries = {"base": build(base_app, work / "build-base", args.cxx),
                "current": build(APP, work / "build-current", args.cxx)}
    traces, timings = {}, {name: [] for name in binaries}
    for run_index in range(args.runs):
        for name, binary in binaries.items():
            trace = work / f"{name}.trace"
            timings[name].append(run(binary, trace))
            data = trace.read_bytes()
            if traces.setdefault(name, data) != data:
                print(f"FAIL: {name} trace is not reproducible between runs")
                return 1
    if traces["base"] != traces["current"]:
        a, b = traces["base"], traces["current"]
        first = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
        print(f"FAIL: standard-mode output differs at float {first // 4} (sizes {len(a)} and {len(b)} bytes)")
        return 1
    print(f"T1 PASS: {len(traces['base']) // 4} rendered values bit-for-bit identical to {args.base}")
    print(f"{'series':24s} {'base mean/p99 [us]':>20s} {'current mean/p99 [us]':>22s}")
    slower = False
    for series in timings["base"][0]:
        # Minimum over runs of each statistic suppresses unrelated host load.
        base = [min(t[series][i] for t in timings["base"]) for i in range(4)]
        current = [min(t[series][i] for t in timings["current"]) for i in range(4)]
        print(f"{series:24s} {base[0]:9.1f} / {base[2]:8.1f} {current[0]:11.1f} / {current[2]:8.1f}")
        slower |= current[0] > 1.1 * base[0] + 2
    if slower:
        print("FAIL: standard-mode decision time increased by more than 10% + 2 us")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
