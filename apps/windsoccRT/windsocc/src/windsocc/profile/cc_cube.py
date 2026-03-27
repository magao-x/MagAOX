"""
Profile ``process_single_cc_cube`` (see ``windsocc.measure``) with cProfile.

Write a ``.prof`` file for Snakeviz::

    uv sync --extra profile
    uv run ws_profile_cc_cube --data_dir /path/to/camwfs/run
    snakeviz measure_results/profile_process_single_cc_cube.prof

Or pass ``--output`` for a custom path.
"""

from __future__ import annotations

import argparse
import cProfile
import logging
import os
import pstats
import sys
from typing import Any, Dict, Tuple

import numpy as np
from astropy.io import fits

from windsocc.core.track_wind import process_single_cc_cube
from windsocc.io.dir_handling import allocate_measure_dirs
from windsocc.io.fits_handling import load_mf_response_cubes
from windsocc.measure import parse_config_file


def _params_for_process_single_cc(
    config_params: Dict[str, Any],
    cube_data: np.ndarray,
) -> Tuple[Any, ...]:
    """Mirror ``process_mf_response_cubes`` defaults for one cube (first-iteration behavior)."""
    sep_thresh = config_params.get("SEP_THRESH", None)
    sep_minarea = config_params.get("SEP_MINAREA", None)
    inner_radius = config_params.get("INNER_RADIUS", None)
    outer_radius = config_params.get("OUTER_RADIUS", None)
    frame_binning = config_params.get("GROUP_SIZE", None)
    image_center = config_params.get("IMAGE_CENTER", None)
    diam_pupils = config_params.get("DIAM_PRIMARY", None)
    mirror_diam = config_params.get("D_MIRROR", None)
    if mirror_diam is None or diam_pupils is None:
        mirror_diam = 6.5
        diam_pupils = 60
    meters_per_pixel = mirror_diam / diam_pupils

    if frame_binning is None:
        frame_binning = 8
    time_per_frame = (1 / 2000.0) * frame_binning

    if sep_thresh is None or sep_minarea is None:
        sep_thresh = 3.0
        sep_minarea = 5

    if inner_radius is None or outer_radius is None:
        inner_radius = 18
        outer_radius = 30

    if image_center is None:
        image_center = (
            round(cube_data.shape[1] + 1) / 2,
            round(cube_data.shape[2] + 1) / 2,
        )

    return (
        image_center,
        meters_per_pixel,
        sep_thresh,
        sep_minarea,
        inner_radius,
        outer_radius,
        time_per_frame,
    )


def run_cprofile_process_single_cc(
    cube_data: np.ndarray,
    config_params: Dict[str, Any],
    prof_path: str,
) -> None:
    """Profile only ``process_single_cc_cube`` and write binary stats for Snakeviz."""
    args = _params_for_process_single_cc(config_params, cube_data)
    prof = cProfile.Profile()
    prof.enable()
    try:
        process_single_cc_cube(cube_data, *args)
    finally:
        prof.disable()
        prof.dump_stats(prof_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="cProfile ``process_single_cc_cube``; open the .prof with snakeviz."
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help="Directory containing ws_config.yaml (same as ws_measure).",
    )
    p.add_argument(
        "--cube_index",
        type=int,
        default=0,
        help="Which MF response cube to load (default: 0).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the .prof file (default: <measure_dir>/profile_process_single_cc_cube.prof).",
    )
    p.add_argument(
        "--stats-sort",
        type=str,
        default="cumulative",
        choices=("cumulative", "tottime", "calls", "filename"),
        help="Sort key for the printed pstats summary (Snakeviz uses the raw .prof).",
    )
    p.add_argument(
        "--stats-lines",
        type=int,
        default=40,
        help="Number of lines to print from pstats after profiling.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    path_yaml = os.path.join(args.data_dir, "ws_config.yaml")
    if not os.path.isfile(path_yaml):
        logging.error("Config not found: %s", path_yaml)
        sys.exit(1)

    basedir = os.path.dirname(path_yaml)
    config_params = parse_config_file(path_yaml)

    dirs = allocate_measure_dirs(basedir=basedir, params_yaml=config_params)
    mf_response_cubes_loc = os.path.join(dirs["distill_directory"], "mf_response_cubes")
    _fnames, mf_paths = load_mf_response_cubes(mf_response_cubes_loc)

    if not mf_paths:
        logging.warning("No MF response cubes under %s; nothing to profile.", mf_response_cubes_loc)
        sys.exit(1)

    if args.cube_index < 0 or args.cube_index >= len(mf_paths):
        logging.error("cube_index %s out of range (0..%s).", args.cube_index, len(mf_paths) - 1)
        sys.exit(1)

    cube_path = mf_paths[args.cube_index]
    logging.info("Profiling process_single_cc_cube on %s", os.path.basename(cube_path))

    cube_data = fits.getdata(cube_path)

    if args.output:
        prof_path = os.path.abspath(args.output)
    else:
        out_dir = dirs["measure_directory"]
        os.makedirs(out_dir, exist_ok=True)
        prof_path = os.path.join(out_dir, "profile_process_single_cc_cube.prof")

    run_cprofile_process_single_cc(cube_data, config_params, prof_path)

    print(f"Wrote cProfile stats: {prof_path}")
    print(f"View with: snakeviz {prof_path}")

    stats = pstats.Stats(prof_path)
    stats.sort_stats(args.stats_sort)
    stats.print_stats(args.stats_lines)


if __name__ == "__main__":
    main()
