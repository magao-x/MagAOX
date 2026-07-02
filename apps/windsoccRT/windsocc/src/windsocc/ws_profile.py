#!/usr/bin/env python3
from __future__ import annotations
"""
ws_profile - Measure wind speed and direction using matched-filter response cubes.

Usage (from camwfs data root)::

    uv run ws_profile

Wind-layer HDBSCAN clustering and plots: ``uv run ws_stats`` (``windsocc.stats``).

Profiling::

    uv run ws_profile --profile
    snakeviz <output_dir>/ws_profile_profile.prof
"""

import cProfile
import os
import argparse
import logging
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["POLARS_MAX_THREADS"] = "1"

from windsocc.io.config_handling import parse_config_file
from windsocc.io.dir_handling import allocate_measure_dirs
from windsocc.io.fits_handling import load_mf_response_cubes
from windsocc.core.profile import (
    _slice_mf_response_cubes_dict,
    process_mf_response_cubes,
)

def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    # Load the configuration file
    path_yaml = os.path.join(args.data_dir, "ws_config.yaml")
    basedir = os.path.dirname(path_yaml)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(basedir, "measure_results")
    else:
        output_dir = os.path.join(basedir, args.output_dir)
    if not os.path.exists(output_dir):
        logging.info(
            f"Output directory {output_dir} does not exist. \
            Creating it...")
        os.makedirs(output_dir, exist_ok=True)
    config_params = parse_config_file(path_yaml)

    if args.profile:
        profile_path = args.profile_output
        if profile_path is None:
            profile_path = os.path.join(output_dir, "ws_measure_profile.prof")
        profile_path = os.path.abspath(profile_path)
        prof = cProfile.Profile()
        prof.enable()
        try:
            run_profile_stage(
                basedir=basedir,
                config_params=config_params,
                make_movie=config_params.get("MAKE_MOVIE", False),
            )
        finally:
            prof.disable()
            _profile_dir = os.path.dirname(profile_path)
            if _profile_dir:
                os.makedirs(_profile_dir, exist_ok=True)
            prof.dump_stats(profile_path)
        logging.info(
            "cProfile stats written to %s — view with: snakeviz %s",
            profile_path,
            profile_path,
        )
        logging.info(
            "Install snakeviz if needed: pip install 'windsocc[profile]' or uv sync --extra profile"
        )
    else:
        run_profile_stage(
            basedir=basedir,
            config_params=config_params,
            make_movie=config_params.get("MAKE_MOVIE", False),
        )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure wind layers using matched-filter response cubes from ws_distill."
    )
    parser.add_argument(
        "-d","--data_dir", type=str, default=".",
        help="Path to the directory containing the camwfs sub-directories."
    )
    # parser.add_argument(
    #     "--config", type=str, required=True,
    #     help="Location of the configuration file for observation."
    # )
    parser.add_argument(
        "--output_dir", type=str,
        default=None,help="Path to the output directory (default: data_dir/measure_results)"
        )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile for the profile stage; write binary stats for use with snakeviz.",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default=None,
        metavar="PATH",
        help="Destination .prof file (default: <output_dir>/ws_profile_profile.prof).",
    )

    args = parser.parse_args()
    
    return args

def run_profile_stage(
    basedir: str,
    config_params: dict | None = None,
    make_movie: bool | None = None,
    parity_flip_needed: bool | None = None,
) -> dict:
    """Run the profile stage and return generated output paths."""
    if config_params is None:
        config_params = parse_config_file(os.path.join(basedir, "ws_config.yaml"))

    if make_movie is None:
        make_movie = config_params.get("MAKE_MOVIE", False)

    dirs = allocate_measure_dirs(basedir=basedir, params_yaml=config_params)
    response_cubes_loc = os.path.join(dirs["distill_directory"])
    noise_maps_loc = os.path.join(dirs["distill_directory"], "noise_maps")
    loaded_mf_response_cubes = load_mf_response_cubes(response_cubes_loc)
    hp_mf_response_cube_paths = loaded_mf_response_cubes["unsharped_mf_response_cube_paths"]

    if len(hp_mf_response_cube_paths) == 0:
        logging.warning("No MF response cubes found; nothing to process.")
        return {
            "dirs": dirs,
            "mf_response_cube_paths": [],
            "json_paths": [],
            "movie_paths": [],
            "sources_all": [],
            "wind_summaries": [],
            "rejected_all": [],
            "model_rejected_all": [],
        }
    limit_cubes = config_params.get("LIMIT_CUBES", None)
    if limit_cubes is not None:
        logging.info(f"Limiting the number of cubes to process to {limit_cubes}")
        loaded_mf_response_cubes = _slice_mf_response_cubes_dict(
            loaded_mf_response_cubes, limit_cubes
        )
        hp_mf_response_cube_paths = loaded_mf_response_cubes["unsharped_mf_response_cube_paths"]
    parallelized = config_params.get("PARALLELIZED", False)
    if parallelized and len(hp_mf_response_cube_paths) > 1:
        png_only_movies = True
        sources_all = []
        wind_summaries = []
        rejected_all = []
        model_rejected_all = []
        logging.info("Parallelizing the profile process...")
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = {
                executor.submit(
                    process_mf_response_cubes,
                    {
                        "unsharped_mf_response_cube_paths": [hp_path],
                        "hp_cube_fnames": [hp_fname],
                        "og_cc_cube_paths": [og_path],
                        "og_cube_fnames": [og_fname],
                    },
                    noise_maps_dir=noise_maps_loc,
                    movie_output_dir=dirs["movies_dir"],
                    config_params=config_params,
                    roi_masks_dir=dirs["roi_masks_dir"],
                    wind_data_dir=dirs["wind_data_dir"],
                    make_movie=make_movie,
                    png_only_movies=png_only_movies,
                    rejected_dir=dirs["rejected_directory"],
                    parity_flip_needed=parity_flip_needed,
                ): hp_path
                for hp_path, hp_fname, og_path, og_fname in zip(
                    loaded_mf_response_cubes["unsharped_mf_response_cube_paths"],
                    loaded_mf_response_cubes["hp_cube_fnames"],
                    loaded_mf_response_cubes["og_cc_cube_paths"],
                    loaded_mf_response_cubes["og_cube_fnames"],
                )
            }
            for future in as_completed(futures):
                sources_i, wind_summaries_i, rejected_i, model_rejected_i = future.result()
                sources_all.extend(sources_i)
                wind_summaries.extend(wind_summaries_i)
                rejected_all.extend(rejected_i)
                model_rejected_all.extend(model_rejected_i)
    else:
        png_only_movies = False
        sources_all, wind_summaries, rejected_all, model_rejected_all = process_mf_response_cubes(
            loaded_mf_response_cubes,
            noise_maps_dir=noise_maps_loc,
            movie_output_dir=dirs["movies_dir"],
            config_params=config_params,
            roi_masks_dir=dirs["roi_masks_dir"],
            wind_data_dir=dirs["wind_data_dir"],
            make_movie=make_movie,
            png_only_movies=png_only_movies,
            rejected_dir=dirs["rejected_directory"],
            parity_flip_needed=parity_flip_needed,
        )

    json_paths = []
    movie_paths = []
    for cube_path in hp_mf_response_cube_paths:
        cube_stem = os.path.splitext(os.path.basename(cube_path))[0]
        json_paths.append(
            os.path.join(dirs["wind_data_dir"], f"{cube_stem}_wind_attributes.json")
        )
        if make_movie:
            movie_paths.extend(
                [
                    os.path.join(dirs["movies_dir"], f"{cube_stem}_extrapolated.mp4"),
                    os.path.join(dirs["movies_dir"], f"{cube_stem}_sep_results.mp4"),
                ]
            )

    return {
        "dirs": dirs,
        "mf_response_cube_paths": hp_mf_response_cube_paths,
        "json_paths": json_paths,
        "movie_paths": movie_paths,
        "sources_all": sources_all,
        "wind_summaries": wind_summaries,
        "rejected_all": rejected_all,
        "model_rejected_all": model_rejected_all,
    }


if __name__ == '__main__':
    main()
