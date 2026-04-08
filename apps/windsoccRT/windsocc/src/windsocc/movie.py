#!/usr/bin/env python3
"""
Stitch per-frame PNG sequences (from parallel ``ws_measure``) into MP4 files.

Reads ``measure_results/pngs/<movie_stem>/frame_*.png`` and writes
``measure_results/movies/<movie_stem>.mp4`` using imageio/ffmpeg.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import imageio
import yaml

_FRAME_RE = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)


def _parse_config(data_dir: str) -> dict:
    path = os.path.join(os.path.abspath(data_dir), "ws_config.yaml")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_fps(config: dict, cli_fps: float | None) -> float:
    if cli_fps is not None:
        return float(cli_fps)
    for key in ("MOVIE_FPS", "FPS"):
        v = config.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 30.0


def resolve_parallelized(
    config: dict,
    cli_parallel: bool,
    cli_serial: bool,
) -> bool:
    if cli_serial:
        return False
    if cli_parallel:
        return True
    return bool(config.get("PARALLELIZED", False))


def list_frame_pngs(png_dir: Path) -> list[Path]:
    """Return ``frame_NNNN.png`` paths sorted by numeric suffix."""
    frames: list[tuple[int, Path]] = []
    for p in png_dir.iterdir():
        if not p.is_file():
            continue
        m = _FRAME_RE.match(p.name)
        if m:
            frames.append((int(m.group(1)), p))
    frames.sort(key=lambda t: t[0])
    return [p for _, p in frames]


def stitch_png_dir_to_mp4(
    png_dir: str,
    out_mp4: str,
    fps: float,
    *,
    force: bool = False,
) -> str | None:
    """
    Write one MP4 from ordered PNG frames.

    Returns output path on success, ``None`` if skipped (no frames or existing file).
    """
    png_path = Path(png_dir)
    out_path = Path(out_mp4)
    frames = list_frame_pngs(png_path)
    if not frames:
        logging.warning("No frame_*.png files in %s; skipping.", png_dir)
        return None
    if out_path.is_file() and not force:
        logging.warning("Output exists (use --force to overwrite): %s", out_mp4)
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(
        "Encoding %d frames at %.3g fps -> %s",
        len(frames),
        fps,
        out_mp4,
    )
    with imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
    ) as writer:
        for fp in frames:
            frame = imageio.imread(str(fp))
            writer.append_data(frame)
    return str(out_path)


def discover_stitch_jobs(pngs_root: Path, movies_dir: Path) -> list[tuple[str, str]]:
    """Pairs ``(png_subdir, out_mp4)`` for each immediate child directory of ``pngs``."""
    if not pngs_root.is_dir():
        logging.error("PNG root does not exist: %s", pngs_root)
        return []
    jobs: list[tuple[str, str]] = []
    for entry in sorted(pngs_root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        stem = entry.name
        out_mp4 = str(movies_dir / f"{stem}.mp4")
        jobs.append((str(entry), out_mp4))
    return jobs


def run_stitch_serial(
    jobs: list[tuple[str, str]],
    fps: float,
    force: bool,
) -> list[str | None]:
    results: list[str | None] = []
    for png_dir, out_mp4 in jobs:
        results.append(stitch_png_dir_to_mp4(png_dir, out_mp4, fps, force=force))
    return results


def _stitch_job(args: tuple[str, str, float, bool]) -> str | None:
    """Picklable worker for :class:`ProcessPoolExecutor`."""
    png_dir, out_mp4, fps, force = args
    return stitch_png_dir_to_mp4(png_dir, out_mp4, fps, force=force)


def run_stitch_parallel(
    jobs: list[tuple[str, str]],
    fps: float,
    force: bool,
    max_workers: int | None,
) -> list[str | None]:
    if max_workers is None:
        max_workers = cpu_count()
    worker_args = [(png_dir, out_mp4, fps, force) for png_dir, out_mp4 in jobs]

    results_map: dict[int, str | None] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_stitch_job, a): i for i, a in enumerate(worker_args)}
        for fut in as_completed(futures):
            idx = futures[fut]
            results_map[idx] = fut.result()
    return [results_map[i] for i in range(len(jobs))]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stitch measure_results/pngs/*/frame_*.png into measure_results/movies/*.mp4."
    )
    p.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default=".",
        help="Directory containing ws_config.yaml and measure_results (default: .).",
    )
    p.add_argument(
        "--measure_results",
        type=str,
        default=None,
        help="Explicit path to measure_results (default: <data_dir>/measure_results).",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output frame rate (overrides ws_config MOVIE_FPS / FPS; default 30 if unset).",
    )
    p.add_argument(
        "--parallelized",
        action="store_true",
        help="Use process pool when multiple PNG directories exist (overrides config).",
    )
    p.add_argument(
        "--no-parallelized",
        action="store_true",
        help="Force serial encoding (overrides config and --parallelized).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing MP4 files.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max worker processes when parallelized (default: cpu_count()).",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    config = _parse_config(data_dir)

    if args.measure_results:
        measure_results = os.path.abspath(args.measure_results)
    else:
        measure_results = os.path.join(data_dir, "measure_results")

    pngs_root = Path(measure_results) / "pngs"
    movies_dir = Path(measure_results) / "movies"
    movies_dir.mkdir(parents=True, exist_ok=True)

    fps = resolve_fps(config, args.fps)
    parallelized = resolve_parallelized(
        config,
        cli_parallel=args.parallelized,
        cli_serial=args.no_parallelized,
    )

    jobs = discover_stitch_jobs(pngs_root, movies_dir)
    if not jobs:
        logging.warning("No PNG subdirectories under %s.", pngs_root)
        return 1

    logging.info(
        "measure_results=%s  fps=%.3g  parallelized=%s  jobs=%d",
        measure_results,
        fps,
        parallelized,
        len(jobs),
    )

    if parallelized and len(jobs) > 1:
        run_stitch_parallel(jobs, fps, args.force, args.max_workers)
    else:
        run_stitch_serial(jobs, fps, args.force)

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
