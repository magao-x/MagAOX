"""Stage-by-stage import probe for WindsoCC realtime dependencies."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback
from typing import List, Tuple

STAGED_IMPORTS: List[Tuple[str, Tuple[str, ...]]] = [
    ("numpy", ("numpy",)),
    ("astropy.io.fits", ("astropy.io.fits",)),
    ("yaml", ("yaml",)),
    ("scipy", ("scipy.ndimage", "scipy.signal")),
    ("matplotlib", ("matplotlib.pyplot",)),
    ("scikit-image", ("skimage.feature", "skimage.measure")),
    ("pandas", ("pandas",)),
    ("polars", ("polars",)),
    ("sep", ("sep",)),
    ("windsocc.distill", ("windsocc.distill",)),
    ("windsocc.measure", ("windsocc.measure",)),
    ("windsocc.reduce", ("windsocc.reduce",)),
    ("windsocc.xcorr", ("windsocc.xcorr",)),
    ("windsocc.realtime", ("windsocc.realtime",)),
]

ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "MPLBACKEND",
    "PYTHONPATH",
)


def _print_environment() -> None:
    """Print runtime details that affect import-time behavior."""
    print("WindsoCC import probe environment", flush=True)
    print(f"python_executable={sys.executable}", flush=True)
    print(f"python_version={sys.version}", flush=True)
    print(f"sys_path0={sys.path[0] if sys.path else ''}", flush=True)
    for key in ENV_KEYS:
        print(f"{key}={os.environ.get(key, '')}", flush=True)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the import probe."""
    parser = argparse.ArgumentParser(
        description="Stage-by-stage WindsoCC import probe for debugging embedded import crashes."
    )
    parser.add_argument(
        "--direct-module",
        default="",
        help="Skip staged imports and import only this module directly.",
    )
    parser.add_argument(
        "--start-at",
        default="",
        help="Start at the named stage label (for example: polars, sep, windsocc.measure).",
    )
    parser.add_argument(
        "--stop-after",
        default="",
        help="Stop after the named stage label succeeds.",
    )
    parser.add_argument(
        "--no-env",
        action="store_true",
        help="Do not print environment details before importing.",
    )
    return parser.parse_args()


def _selected_stages(start_at: str, stop_after: str) -> List[Tuple[str, Tuple[str, ...]]]:
    """Return the requested contiguous stage subset."""
    stages = STAGED_IMPORTS

    if start_at:
        labels = [label for label, _ in stages]
        if start_at not in labels:
            raise ValueError(f"Unknown --start-at stage: {start_at}")
        stages = stages[labels.index(start_at) :]

    if stop_after:
        trimmed: List[Tuple[str, Tuple[str, ...]]] = []
        found = False
        for stage in stages:
            trimmed.append(stage)
            if stage[0] == stop_after:
                found = True
                break
        if not found:
            raise ValueError(f"Unknown --stop-after stage in selected range: {stop_after}")
        stages = trimmed

    return stages


def _run_stage(index: int, label: str, modules: tuple[str, ...]) -> None:
    """Import each module in one stage and print breadcrumbs."""
    module_list = ", ".join(modules)
    print(f"[{index:02d}] START {label}: {module_list}", flush=True)
    for module_name in modules:
        print(f"[{index:02d}]   import {module_name}", flush=True)
        importlib.import_module(module_name)
    print(f"[{index:02d}] OK {label}", flush=True)


def _run_direct(module_name: str) -> int:
    """Import a single module without staged grouping."""
    print(f"[00] START direct: {module_name}", flush=True)
    importlib.import_module(module_name)
    print(f"[00] OK direct: {module_name}", flush=True)
    return 0


def main() -> int:
    """Run the configured staged import probe."""
    args = _parse_args()

    if not args.no_env:
        _print_environment()

    try:
        if args.direct_module:
            return _run_direct(args.direct_module)

        stages = _selected_stages(args.start_at, args.stop_after)
        for index, (label, modules) in enumerate(stages, start=1):
            _run_stage(index, label, modules)
        print("Import probe completed successfully.", flush=True)
        return 0
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"Import probe failed: {exc!r}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
