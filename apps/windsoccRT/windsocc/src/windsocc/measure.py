#!/usr/bin/env python3
from __future__ import annotations
"""
measure.py - Measure wind speed and direction using
matched-filter response cubes from ws_distill.

Usage (must be run from the root dir of the camwfs data):
    uv run ws_measure

Profiling (cProfile + Snakeviz):

    uv run ws_measure --profile
    uv sync --extra profile   # optional: pip install snakeviz for the viewer CLI
    snakeviz <output_dir>/ws_measure_profile.prof

From camwfs experiment:

Sine pattern travelling E --> W (270 deg) on DM:
camwfs: 297.5 deg propagation direction
camsci1: 242.5 (SW) or 62.5 (NE) degree sparkle orientation


I watched the turbulence pattern on the DM on-sky and saw that a CCW shift in
the wind direction also induced a CCW shift in the WDH on camsci.
This means that we expect a Fourier mode with a NW propagation direction
on the DM to show as approx. E-W on camsci.
So -- Sine pattern travelling SE --> NW (300 deg) on DM...
we expect 272.5 degree WDH orientation on camsci.

I think I meant CW shift above... ^
It *is* a negative rotation (for N up E left) so maybe that's what I meant
by CCW shift, oops

** So, for the measured wind direction on camwfs data, we need to subtract
55 degrees from it to match the direction of the WDH on camsci. **


UPDATE 06/28/2025
Realized there is a parity change in the wind PA vector after transit,
that's why the PA wasn't lining up super well for some images.

The PA offset for the 2023A dataset is 28 degrees, not 55 degrees as
tested a few weeks ago and stated above, not sure why. This same 28 deg
is mentioned in the C++ sparkle clock code so this has been measured before.

This code now takes a yaml config file instead of CL arguments to consolidate
the various parameters that we need to keep track of for each dataset or obs.
Things like the PA offset, time of transit, how to mask the pupils, etc.

All of the measure code has been converted to polars by GPT Codex 5.4
for performance reasons. I've so far observed a x2 speedup JKK 03/21/2026

Per-track wind JSON (e.g. ``*_wind_attributes.json``, ``*_rejected_sources.json``, and
``*_model_rejected.json``)
includes ``raw_direction`` (degrees in the camwfs cube frame) and ``corrected_direction``
(after ``PA_OFFSET``, with parity handled by negating that offset when a parity flip applies).
``direction`` is set equal to ``corrected_direction`` for backward compatibility.

TODO apply parity flip when HA is positive

"""

import cProfile
import os
import sys
import argparse
import logging
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

import yaml
import pandas as pd
import polars as pl

from windsocc.io.dir_handling import allocate_measure_dirs
from windsocc.io.fits_handling import load_mf_response_cubes, load_collapsed_unsharp_response_maps
from windsocc.core.track_wind import process_single_cc_cube
from windsocc.visualization.plot_measure_results import make_source_detection_movie

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
        help="Enable cProfile for the measure stage; write binary stats for use with snakeviz.",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default=None,
        metavar="PATH",
        help="Destination .prof file (default: <output_dir>/ws_measure_profile.prof).",
    )

    args = parser.parse_args()
    
    return args


def parse_config_file(config_file: str) -> dict:
    """Parse the configuration file."""
    with open(config_file, 'r') as yaml_file:
        config_params = yaml.safe_load(yaml_file)
    return config_params


def process_mf_response_cube_paths(mf_response_cube_paths: list) -> tuple[list, list]:
    """Process the matched-filter response cube paths."""
    fname_prefixes = [os.path.basename(path).split("_")[0] for path in mf_response_cube_paths]
    times_from_fnames = [extract_time_from_fname(path) for path in mf_response_cube_paths]
    return fname_prefixes, times_from_fnames


def filter_sources_by_angle(
    sources: np.ndarray,
    center_x: float,
    center_y: float,
    min_sep_deg: float = 65.0,
    max_sep_deg: float = 115.0,
    circular_ratio: float = 0.6,
) -> np.ndarray:
    if sources.size == 0:
        return sources
    axis_ratio = np.zeros_like(sources["a"], dtype=float)
    valid = sources["a"] > 0
    axis_ratio[valid] = sources["b"][valid] / sources["a"][valid]
    is_circular = axis_ratio >= circular_ratio
    dx = sources["x"] - center_x
    dy = sources["y"] - center_y
    pos_angle = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    theta_deg = (np.degrees(sources["theta"]) + 360.0) % 180.0
    pos_mod = pos_angle % 180.0
    delta = np.abs(theta_deg - pos_mod)
    delta = np.minimum(delta, 180.0 - delta)
    angle_ok = (delta >= min_sep_deg) & (delta <= max_sep_deg)
    mask = is_circular | (~is_circular & angle_ok)
    return sources[mask]



def _apply_direction_corrections_to_rows(rows: list[dict], pa_offset_deg: float) -> None:
    """Set ``raw_direction``, ``corrected_direction``, and ``direction`` on each row.

    ``direction`` values coming from summarization are the mean angle in the camwfs
    cube frame (deg, ``[0, 360)``). ``pa_offset_deg`` is the value from config after
    parity bookkeeping (sign flip when a parity flip applies), matching the previous
    single-field behavior for ``direction``.
    """
    if not rows:
        return
    for row in rows:
        if row is None or "direction" not in row:
            continue
        try:
            raw = float(row["direction"])
        except (TypeError, ValueError):
            continue
        raw_wrapped = raw % 360.0
        row["raw_direction"] = raw_wrapped
        row["corrected_direction"] = (raw_wrapped + pa_offset_deg) % 360.0
        row["direction"] = row["corrected_direction"]


def summarize_wind_tracks(cube_sources: object) -> dict:
    if isinstance(cube_sources, pd.DataFrame):
        cube_sources = pl.DataFrame(cube_sources.to_dict("records"))
    if not isinstance(cube_sources, pl.DataFrame) or cube_sources.is_empty():
        return {"tracks": []}

    required_cols = {
        "track_id",
        "direction",
        "velocity_m_per_s",
        "matches",
        "frames",
        "flux",
        "flux_err",
        "source_area",
    }
    if not required_cols.issubset(set(cube_sources.columns)):
        return {"tracks": []}

    tracked = cube_sources.with_columns(
        [
            pl.col("track_id").cast(pl.Int64, strict=False).alias("track_id"),
            pl.col("direction").cast(pl.Float64, strict=False).alias("direction"),
            pl.col("velocity_m_per_s").cast(pl.Float64, strict=False).alias("velocity_m_per_s"),
            pl.col("matches").cast(pl.Float64, strict=False).alias("matches"),
            pl.col("frames").cast(pl.Int64, strict=False).alias("frames"),
            pl.col("flux").cast(pl.Float64, strict=False).alias("flux"),
            pl.col("flux_err").cast(pl.Float64, strict=False).alias("flux_err"),
            pl.col("source_area").cast(pl.Float64, strict=False).alias("source_area"),
        ]
    ).drop_nulls(
        subset=["track_id", "direction", "velocity_m_per_s", "matches", "frames"]
    )
    if tracked.is_empty():
        return {"tracks": []}

    summarized = (
        tracked.sort(["track_id", "frames"])
        .group_by("track_id")
        .agg(
            [
                pl.col("direction").mean().alias("direction"),
                pl.col("velocity_m_per_s").mean().alias("velocity_m_per_s"),
                pl.col("matches").max().alias("matches"),
                pl.col("flux").mean().alias("flux"),
                pl.col("flux_err").mean().alias("flux_err"),
                pl.col("source_area").mean().alias("source_area"),
            ]
        )
        .sort("track_id")
    )
    tracks = [
        {
            "track_id": int(row["track_id"]),
            "direction": float(row["direction"]),
            "velocity_m_per_s": float(row["velocity_m_per_s"]),
            "matches": int(row["matches"]),
            "flux": float(row["flux"]),
            "flux_err": float(row["flux_err"]),
            "source_area": float(row["source_area"]),
        }
        for row in summarized.iter_rows(named=True)
    ]
    return {"tracks": tracks}

def mean_inferred_origin_by_track_id(
    cube_sources: object,
    *,
    track_ids: set[int] | None = None,
) -> dict[int, float]:
    """Compute per-track mean inferred origin (traceback distance)."""
    if isinstance(cube_sources, pd.DataFrame):
        cube_sources = pl.DataFrame(cube_sources.to_dict("records"))
    if not isinstance(cube_sources, pl.DataFrame) or cube_sources.is_empty():
        return {}
    if "track_id" not in cube_sources.columns or "inferred_origin" not in cube_sources.columns:
        return {}

    working = cube_sources.with_columns(
        track_id_num=pl.col("track_id").cast(pl.Int64, strict=False),
        inferred_origin_num=pl.col("inferred_origin").cast(pl.Float64, strict=False),
    ).drop_nulls(subset=["track_id_num", "inferred_origin_num"])

    if track_ids is not None:
        working = working.filter(pl.col("track_id_num").is_in(list(track_ids)))

    if working.is_empty():
        return {}

    grouped = working.group_by("track_id_num").agg(
        pl.col("inferred_origin_num").mean().alias("inferred_origin_mean")
    )
    return {int(r["track_id_num"]): float(r["inferred_origin_mean"]) for r in grouped.iter_rows(named=True)}

def summarize_rejected_tracks(
    rejected_sources: object,
    *,
    pa_offset_deg: float = 0.0,
) -> dict:
    """Aggregate per-track rejected rows for JSON (``*_rejected_sources.json``).

    Applies the same ``raw_direction`` / ``corrected_direction`` / ``direction``
    convention as accepted tracks (see ``_apply_direction_corrections_to_rows``).
    """
    if isinstance(rejected_sources, pd.DataFrame):
        rejected_sources = pl.DataFrame(rejected_sources.to_dict("records"))
    if not isinstance(rejected_sources, pl.DataFrame) or rejected_sources.is_empty():
        return {"rejected": []}
    required_cols = {
        "track_id", "reject_reason",
        "frames", "direction",
        "velocity_m_per_s", "matches"}
    if not required_cols.issubset(set(rejected_sources.columns)):
        return {"rejected": []}
    tracked = rejected_sources.with_columns(
        pl.col("track_id").cast(pl.Int64, strict=False).alias("track_id"),
        pl.col("reject_reason").cast(pl.String, strict=False).alias("reject_reason"),
        pl.col("frames").cast(pl.Int64, strict=False).alias("frames"),
        pl.col("direction").cast(pl.Float64, strict=False).alias("direction"),
        pl.col("velocity_m_per_s").cast(pl.Float64, strict=False).alias("velocity_m_per_s"),
    ).drop_nulls(subset=["track_id", "reject_reason", "frames", "direction", "velocity_m_per_s"])
    if tracked.is_empty():
        return {"rejected": []}
    matches_sorted = tracked.with_columns(
        pl.col("matches").cast(pl.Int64, strict=False).fill_null(0).alias("matches")
    ).sort(["track_id", "frames", "matches"])
    summarized = (
        matches_sorted
        .group_by("track_id")
        .agg(
            pl.col("reject_reason").last().alias("reject_reason"),
            pl.col("frames").last().alias("frames"),
            pl.col("direction").mean().alias("direction"),
            pl.col("velocity_m_per_s").mean().alias("velocity_m_per_s"),
            pl.col("matches").max().alias("matches"),
        )
        .sort("matches", descending=True)
        )
    rejected_tracks = [
        {
            "track_id": int(row["track_id"]),
            "reject_reason": row["reject_reason"],
            "frames": int(row["frames"]),
            "direction": float(row["direction"]),
            "velocity_m_per_s": float(row["velocity_m_per_s"]),
            "matches": int(row["matches"]),
        }
        for row in summarized.iter_rows(named=True)
    ]
    _apply_direction_corrections_to_rows(rejected_tracks, pa_offset_deg)
    return {"rejected": rejected_tracks}


def summarize_model_rejected_tracks(model_rejected: object) -> dict:
    """Summarize whole-cube model rejections from ``_keep_track_ids_by_model()``."""
    if isinstance(model_rejected, pd.DataFrame):
        model_rejected = pl.DataFrame(model_rejected.to_dict("records"))
    if not isinstance(model_rejected, pl.DataFrame) or model_rejected.is_empty():
        return {"model_rejected": []}
    required_cols = {
        "track_id",
        "reject_reason",
        "frames",
        "direction",
        "velocity_m_per_s",
        "matches",
        "inferred_origin",
    }
    if not required_cols.issubset(set(model_rejected.columns)):
        return {"model_rejected": []}
    summarized = (
        model_rejected.with_columns(
            pl.col("track_id").cast(pl.Int64, strict=False).alias("track_id"),
            pl.col("reject_reason").cast(pl.String, strict=False).alias("reject_reason"),
            pl.col("frames").cast(pl.Int64, strict=False).alias("frames"),
            pl.col("direction").cast(pl.Float64, strict=False).alias("direction"),
            pl.col("velocity_m_per_s").cast(pl.Float64, strict=False).alias("velocity_m_per_s"),
            pl.col("matches").cast(pl.Int64, strict=False).alias("matches"),
            pl.col("inferred_origin").cast(pl.Float64, strict=False).alias("inferred_origin"),
        )
        .sort("matches", descending=True)
    )
    rows = [
        {
            "track_id": int(row["track_id"]),
            "reject_reason": row["reject_reason"],
            "frames": int(row["frames"]),
            "direction": float(row["direction"]),
            "velocity_m_per_s": float(row["velocity_m_per_s"]),
            "matches": int(row["matches"]),
            "inferred_origin": float(row["inferred_origin"]),
        }
        for row in summarized.iter_rows(named=True)
    ]
    return {"model_rejected": rows}


def build_measure_runtime_params(config_params: dict, cube_data: np.ndarray) -> tuple:
    """Resolve runtime parameters for ``process_single_cc_cube``."""
    sep_thresh = config_params.get("SEP_THRESH", None)
    sep_minarea = config_params.get("SEP_MINAREA", None)
    inner_radius = config_params.get("INNER_RADIUS", None)
    outer_radius = config_params.get("OUTER_RADIUS", None)
    frame_binning = config_params.get("GROUP_SIZE", None)
    image_center = config_params.get("IMAGE_CENTER", None)
    diam_pupils = config_params.get("DIAM_PRIMARY", config_params.get("DIAM_PUPILS"))
    mirror_diam = config_params.get("D_MIRROR", None)
    time_per_frame = config_params.get("TIME_PER_FRAME", None)
    if mirror_diam is None or diam_pupils is None:
        logging.warning(
            "D_MIRROR and DIAM_PRIMARY/DIAM_PUPILS not set in the config file. \
            Using default value of 6.5 and 60 respectively. \
            Please set them in the config file for better results."
        )
        mirror_diam = 6.5
        diam_pupils = 60
    else:
        logging.info(
            "Using D_MIRROR: %s and pupil diameter: %s for pupil diameter.",
            mirror_diam,
            diam_pupils,
        )
    meters_per_pixel = mirror_diam / diam_pupils

    if frame_binning is None:
        logging.warning(
            "GROUP_SIZE not set in the config file. \
            Using default value of 8. \
            Please set it in the config file for better results."
        )
        frame_binning = 8
    else:
        logging.info("Using GROUP_SIZE: %s for frame binning.", frame_binning)

    if sep_thresh is None or sep_minarea is None:
        logging.warning(
            "SEP_THRESH and SEP_MINAREA not set in the config file. \
            Using default values of 3.0 and 5 respectively. \
            Please set them in the config file for better results."
        )
        sep_thresh = 3.0
        sep_minarea = 5
    else:
        logging.info(
            "Using SEP_THRESH: %s and SEP_MINAREA: %s for source extraction.",
            sep_thresh,
            sep_minarea,
        )
    if inner_radius is None or outer_radius is None:
        logging.warning(
            "INNER_RADIUS and OUTER_RADIUS not set in the config file. \
            Using default values of 18 and 30 respectively. \
            Please set them in the config file for better results."
        )
        inner_radius = 18
        outer_radius = 30
    else:
        logging.info(
            "Using INNER_RADIUS: %s and OUTER_RADIUS: %s for source extraction.",
            inner_radius,
            outer_radius,
        )
    if image_center is None:
        image_center = (
            round(cube_data.shape[1] + 1) / 2,
            round(cube_data.shape[2] + 1) / 2,
        )
        logging.info(
            "IMAGE_CENTER not set in the config file. Using image center from cube data: %s",
            image_center,
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




def process_mf_response_cubes(
    mf_response_cube_paths: list,
    mf_response_cube_fnames: list,
    movie_output_dir: str,
    config_params: dict,
    roi_masks_dir: str,
    wind_data_dir: str,
    make_movie: bool = False,
    rejected_dir: str = None,
    parity_flip_needed: bool | None = None,
):
    """Define the tripwire region then process the matched-filter response cubes.

    Parameters:
    -----------
    mf_response_cube_paths : list
        List of paths to the matched-filter response cubes.
    config_params : dict
        Dictionary of configuration parameters.

    Returns:
    --------
    sources_all : list
        List of vetted per-frame tracked detections for each cube.
        Per-cube summary rows are written to JSON from the separate
        summary table returned by ``process_single_cc_cube()``.
    rejected_all : list
        List of tracker-side rejected detections for each cube. These are
        kept separate from the vetted movie table and summary table.
    model_rejected_all : list
        Per-cube tables of tracks rejected by whole-cube model filtering
        (``_keep_track_ids_by_model``), with reasons.
    --------
    """
    wind_peaks_all = []
    wind_summaries_all = []
    wind_rejected_all = []
    model_rejected_all = []
    # Feed the cubes into sep to collect the sources
    for cube_name, cube_path in zip(mf_response_cube_fnames, mf_response_cube_paths):
        logging.info("Processing cube: %s", cube_name)
        if parity_flip_needed is None:
            # Determine if a parity flip is needed
            current_time = extract_time_from_fname(cube_name)
            if are_we_past_transit(current_time, config_params.get("TIME_TRANSIT", None)):
                parity_flip_needed = True
                logging.info("Parity flip needed for cube: %s", cube_name)
            else:
                parity_flip_needed = False
                logging.info("No parity flip needed for cube: %s", cube_name)
        else:
            if parity_flip_needed:
                logging.info("Parity flip needed for cube: %s", cube_name)
            else:
                logging.info("No parity flip needed for cube: %s", cube_name)
        cube_data = fits.getdata(cube_path)
        (
            image_center,
            meters_per_pixel,
            sep_thresh,
            sep_minarea,
            inner_radius,
            outer_radius,
            time_per_frame,
        ) = build_measure_runtime_params(config_params, cube_data)
        (
            wind_vetted_cube,
            wind_summary_cube,
            mask_cube,
            wind_rejected_cube,
            wind_track_history,
            wind_model_rejected_cube,
        ) = process_single_cc_cube(
            cube_data,
            image_center,
            meters_per_pixel,
            sep_thresh,
            sep_minarea,
            inner_radius,
            outer_radius,
            time_per_frame,
        )
        cube_stem = os.path.splitext(os.path.basename(cube_path))[0]
        mask_save_path = os.path.join(roi_masks_dir, f"{cube_stem}_masking.fits")
        fits.writeto(mask_save_path, mask_cube.astype(np.float32), overwrite=True)

        inferred_origin_means_vetted = mean_inferred_origin_by_track_id(wind_vetted_cube)
        wind_summary = summarize_wind_tracks(wind_summary_cube)
        for track in wind_summary.get("tracks", []):
            track["inferred_origin"] = inferred_origin_means_vetted.get(track["track_id"])
        pa_offset_deg = float(config_params.get("PA_OFFSET", 0) or 0)
        if parity_flip_needed:
            pa_offset_deg = -pa_offset_deg
        _apply_direction_corrections_to_rows(wind_summary.get("tracks", []), pa_offset_deg)
        wind_summaries_all.append(wind_summary)
        wind_save_path = os.path.join(wind_data_dir, f"{cube_stem}_wind_attributes.json")
        with open(wind_save_path, "w", encoding="utf-8") as f:
            json.dump(wind_summary, f, indent=2)

        wind_peaks_all.append(wind_vetted_cube)
        wind_rejected_all.append(wind_rejected_cube)
        model_rejected_all.append(wind_model_rejected_cube)
        rejected_summary = summarize_rejected_tracks(
            wind_rejected_cube,
            pa_offset_deg=pa_offset_deg,
        )
        rejected_track_ids = {int(t["track_id"]) for t in rejected_summary.get("rejected", []) if t.get("track_id") is not None}
        inferred_origin_means_rejected = mean_inferred_origin_by_track_id(
            wind_track_history,
            track_ids=rejected_track_ids if rejected_track_ids else None,
        )
        for track in rejected_summary.get("rejected", []):
            track["inferred_origin"] = inferred_origin_means_rejected.get(track["track_id"])
        model_rejected_summary = summarize_model_rejected_tracks(wind_model_rejected_cube)
        _apply_direction_corrections_to_rows(model_rejected_summary.get("model_rejected", []), pa_offset_deg)
        # Save the rejected sources to a JSON file
        rejected_save_path = os.path.join(rejected_dir, f"{cube_stem}_rejected_sources.json")
        with open(rejected_save_path, "w", encoding="utf-8") as f:
            json.dump(rejected_summary, f, indent=2)
        model_rejected_save_path = os.path.join(rejected_dir, f"{cube_stem}_model_rejected.json")
        with open(model_rejected_save_path, "w", encoding="utf-8") as f:
            json.dump(model_rejected_summary, f, indent=2)
        if make_movie:
            make_source_detection_movie(
                mf_response_cube_path=cube_path,
                mf_response_cube_fname=cube_name,
                sources_all=wind_peaks_all,
                output_dir=movie_output_dir,
                fps=5,
                cmap="Blues_r"
            )
    return wind_peaks_all, wind_summaries_all, wind_rejected_all, model_rejected_all


def run_measure_stage(
    basedir: str,
    config_params: dict | None = None,
    make_movie: bool | None = None,
    parity_flip_needed: bool | None = None,
) -> dict:
    """Run the measure stage and return generated output paths."""
    if config_params is None:
        config_params = parse_config_file(os.path.join(basedir, "ws_config.yaml"))

    if make_movie is None:
        make_movie = config_params.get("MAKE_MOVIE", False)

    dirs = allocate_measure_dirs(basedir=basedir, params_yaml=config_params)
    mf_response_cubes_loc = os.path.join(dirs["distill_directory"], "mf_response_cubes")
    mf_response_cube_fnames, mf_response_cube_paths = load_mf_response_cubes(mf_response_cubes_loc)
    _collapsed_unsharp_names, _collapsed_unsharp_paths = load_collapsed_unsharp_response_maps(
        mf_response_cubes_loc
    )
    if len(mf_response_cube_paths) == 0:
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
        mf_response_cube_paths = mf_response_cube_paths[:limit_cubes]
        mf_response_cube_fnames = mf_response_cube_fnames[:limit_cubes]

    sources_all, wind_summaries, rejected_all, model_rejected_all = process_mf_response_cubes(
        mf_response_cube_paths,
        mf_response_cube_fnames,
        movie_output_dir=dirs["movies_dir"],
        config_params=config_params,
        roi_masks_dir=dirs["roi_masks_dir"],
        wind_data_dir=dirs["wind_data_dir"],
        make_movie=make_movie,
        rejected_dir=dirs["rejected_directory"],
        parity_flip_needed=parity_flip_needed,
    )

    json_paths = []
    movie_paths = []
    for cube_path in mf_response_cube_paths:
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
        "mf_response_cube_paths": mf_response_cube_paths,
        "json_paths": json_paths,
        "movie_paths": movie_paths,
        "sources_all": sources_all,
        "wind_summaries": wind_summaries,
        "rejected_all": rejected_all,
        "model_rejected_all": model_rejected_all,
    }



def extract_time_from_fname(fname):
    if fname.endswith(".fits"):
        fname = fname.split(".")[0]
    fname_array = fname.split("_")
    time = fname_array[1]
    return time

def are_we_past_transit(current_time, transit_time):
    #Prepare the formatting for each timestamp
    # Ex. transit_time 2023-03-10T06:09:11.342216344Z
    # dt_transit = pd.to_datetime(transit_time)
    s_transit = pl.Series(name="transit_time", values=[transit_time])
    dt_transit = s_transit.str.to_datetime(
        "%Y-%m-%dT%H:%M:%S%.9fZ",time_zone="UTC")
    # Ex. current_time 20230310054802555791000
    s_current = pl.Series(name="current_time", values=[current_time[:-3]])
    dt_current = s_current.str.to_datetime(
        "%Y%m%d%H%M%S%f",time_zone="UTC"
    )
    past_transit  = dt_current > dt_transit


    return past_transit.item()



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
            run_measure_stage(
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
        run_measure_stage(
            basedir=basedir,
            config_params=config_params,
            make_movie=config_params.get("MAKE_MOVIE", False),
        )

if __name__ == '__main__':
    main()
