#!/usr/bin/env python3
"""
Render whole-cube distill noise maps into MP4 movies.

This script consumes:
  distill_results/noise_maps/*_error_map_wholecube.fits
  distill_results/noise_maps/*_error_map_unsharp_wholecube.fits
and writes:
  wind_visuals/wholecube_error_map.mp4
  wind_visuals/wholecube_error_map_unsharp.mp4
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
from typing import Iterable

import imageio
import numpy as np
from astropy.io import fits
from matplotlib import cm


def _extract_sort_key(path: str) -> tuple[str, str]:
    """Sort maps by cam timestamp if present, then by basename."""
    base = os.path.basename(path)
    m = re.search(r"(camwfs_\d+)", base)
    stamp = m.group(1) if m else ""
    return stamp, base


def _list_maps(noise_maps_dir: str, unsharp: bool) -> list[str]:
    if unsharp:
        pattern = os.path.join(noise_maps_dir, "*_error_map_unsharp_wholecube.fits")
    else:
        pattern = os.path.join(noise_maps_dir, "*_error_map_wholecube.fits")
    paths = sorted(glob.glob(pattern), key=_extract_sort_key)
    return paths


def _compute_sequence_limits(paths: Iterable[str]) -> tuple[float, float]:
    """Compute robust global display limits for one sequence."""
    finite_values = []
    for p in paths:
        arr = np.asarray(fits.getdata(p), dtype=np.float32)
        if arr.ndim != 2:
            continue
        mask = np.isfinite(arr)
        if np.any(mask):
            finite_values.append(arr[mask].ravel())
    if not finite_values:
        return 0.0, 1.0
    vals = np.concatenate(finite_values, axis=0)
    vmin, vmax = np.percentile(vals, [1.0, 99.0])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 0.0, 1.0
    return float(vmin), float(vmax)


def _frame_to_rgb(frame: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    norm = np.zeros_like(frame, dtype=np.float32)
    if vmax > vmin:
        norm = (frame - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    rgba = cm.viridis(norm)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb


def _write_movie(map_paths: list[str], out_path: str, fps: float) -> None:
    if not map_paths:
        logging.warning("No maps found for %s; skipping.", out_path)
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vmin, vmax = _compute_sequence_limits(map_paths)
    logging.info(
        "Encoding %d frames at %.3g fps -> %s",
        len(map_paths),
        fps,
        out_path,
    )
    with imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        format="FFMPEG",
    ) as writer:
        for p in map_paths:
            frame = fits.getdata(p)
            if frame.ndim != 2:
                logging.warning("Skipping non-2D map %s with shape %s", p, np.shape(frame))
                continue
            writer.append_data(_frame_to_rgb(frame, vmin, vmax))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render whole-cube distill error maps into MP4 movies in <data_dir>/wind_visuals."
        )
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default=".",
        help="Path to data directory containing distill_results (default: .).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Movie frame rate (default: 30).",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)

    distill_results = os.path.join(data_dir, "distill_results")
    if not os.path.isdir(distill_results):
        logging.warning(
            "distill_results/ not found at %s. ws_distill needs to be run first or the path needs to be adjusted.",
            distill_results,
        )
        return 1

    noise_maps_dir = os.path.join(distill_results, "noise_maps")
    if not os.path.isdir(noise_maps_dir):
        logging.warning(
            "noise_maps/ not found at %s. ws_distill needs to be run first or the path needs to be adjusted.",
            noise_maps_dir,
        )
        return 1

    out_dir = os.path.join(data_dir, "wind_visuals")
    os.makedirs(out_dir, exist_ok=True)

    regular_maps = _list_maps(noise_maps_dir, unsharp=False)
    unsharp_maps = _list_maps(noise_maps_dir, unsharp=True)
    if not regular_maps and not unsharp_maps:
        logging.warning(
            "No whole-cube error maps found in %s. Run ws_distill first or adjust --data-dir.",
            noise_maps_dir,
        )
        return 1

    _write_movie(
        regular_maps,
        os.path.join(out_dir, "wholecube_error_map.mp4"),
        fps=args.fps,
    )
    _write_movie(
        unsharp_maps,
        os.path.join(out_dir, "wholecube_error_map_unsharp.mp4"),
        fps=args.fps,
    )

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

