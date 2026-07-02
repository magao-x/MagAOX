"""
ws_reduce - WindsoCC Data Reduction Pipeline

Performs dark subtraction, optional reference subtraction, pupil cropping, and
writes reduced quadrant cubes for downstream xcorr.
"""

from __future__ import annotations

import os
import argparse
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from windsocc.io.config_handling import parse_config_file, resolve_config_path
from windsocc.io.fits_handling import write_fits_cube
from windsocc.preprocessing.reference_camwfs import create_reference
from windsocc.core.reduce import (
    QUADRANTS,
    DEFAULT_CREATE_REFERENCE,
    DEFAULT_GROUP_SIZE,
    DEFAULT_PUPIL_MASK_RADIUS,
    DEFAULT_REMAKE_REFERENCE,
    DEFAULT_START_FRAME,
    DEFAULT_STEP_FRAME,
    create_reference_from_cubes,
    find_subdirectories,
    load_dark_file,
    load_reference_products,
    process_cube,
    process_file,
    process_subdirectory,
)

def resolve_bool_setting(cli_value, config_params, preferred_key, fallback_key=None, default=False):
    """Resolve a boolean setting from CLI first, then config, then default."""
    if cli_value is not None:
        return cli_value

    if preferred_key in config_params:
        return bool(config_params[preferred_key])

    if fallback_key is not None and fallback_key in config_params:
        return bool(config_params[fallback_key])

    return default

def save_reduced_quadrant_cubes(
    run_dir: str,
    reduced_quadrants: dict,
    group_suffix: str,
    reduce_dir_name: str = "reduce_results",
    *,
    max_frames: int | None = None,
    name_suffix: str = "",
) -> list[str]:
    """Write each quadrant's reduced cube under ``run_dir/reduce_dir_name``.

    Filenames follow ``{group_suffix}_{quadrant}_reduced{name_suffix}.fits``, consistent with
    the batch ``group_suffix`` used elsewhere in the realtime pipeline.

    Parameters
    ----------
    run_dir:
        Realtime run root (e.g. ``camwfs_<timestamp>/``).
    reduced_quadrants:
        Mapping quadrant name -> array with shape ``(n_frames, height, width)``.
    group_suffix:
        Stem shared with other products (e.g. ``camwfs_<timestamp>_00000``).
    reduce_dir_name:
        Subdirectory of ``run_dir``, or an absolute path; default ``reduce_results`` matches
        ``REDUCE_DIR`` in config (same convention as ``XCORR_DIR`` / ``DISTILL_DIR``).
    max_frames:
        If set, only the first ``max_frames`` frames along axis 0 are written (e.g. one raw
        cube's worth when ``INSPECT_REDUCTION`` is enabled).
    name_suffix:
        Inserted before ``.fits`` (e.g. ``_inspect`` for debug products).

    Returns
    -------
    list of str
        Paths written, in quadrant order ``ul``, ``ur``, ``ll``, ``lr``.
    """
    out_dir = (
        reduce_dir_name
        if os.path.isabs(reduce_dir_name)
        else os.path.join(run_dir, reduce_dir_name)
    )
    os.makedirs(out_dir, exist_ok=True)
    paths: list[str] = []
    for quadrant in QUADRANTS:
        cube = reduced_quadrants.get(quadrant)
        if cube is None:
            logging.warning("Missing reduced cube for quadrant %s; skipping save.", quadrant)
            continue
        arr = np.asarray(cube, dtype=np.float32)
        if max_frames is not None:
            arr = arr[: int(max_frames)]
        fname = f"{group_suffix}_{quadrant}_reduced{name_suffix}.fits"
        fpath = os.path.join(out_dir, fname)
        write_fits_cube(fpath, arr)
        paths.append(fpath)
    if paths:
        logging.info(
            "Saved %d reduced quadrant cube(s) under %s",
            len(paths),
            out_dir,
        )
    return paths

def get_pupil_geometry(config_params):
    """Extract per-quadrant pupil centers and the circular mask radius."""
    center_keys = {
        "ul": "UL_CENTER",
        "ur": "UR_CENTER",
        "ll": "LL_CENTER",
        "lr": "LR_CENTER",
    }
    pupil_centers = {}
    for quadrant, config_key in center_keys.items():
        if config_key not in config_params:
            raise KeyError(f"Missing required config key: {config_key}")
        pupil_centers[quadrant] = normalize_center(config_params[config_key], config_key)

    mask_diam = config_params.get("DIAM_PUPILS", DEFAULT_PUPIL_MASK_RADIUS * 2)
    mask_radius = mask_diam // 2

    return pupil_centers, int(round(float(mask_radius)))

def normalize_center(center_value, key_name):
    """Validate and normalize a `[x, y]` pupil center from YAML."""
    if not isinstance(center_value, (list, tuple)) or len(center_value) != 2:
        raise ValueError(f"{key_name} must be a list like [x, y]")

    return (int(center_value[0]), int(center_value[1]))

def process_dataset(
    data_dir,
    top_level_dir,
    nstack,
    pupil_centers,
    pupil_mask_radius,
    start_frame=0,
    step_frame=1,
    group_size=8,
    create_reference_flag=True,
    remake_ref=False,
    skip_dark=False,
    subtract_reference=True,
    apply_tukey_window=False,
    tukey_alpha=0.5,
):
    """
    Process all FITS files in the raw data directory:
      - Loads the dark file (from top-level or subdirectory).
      - Loads or creates the reference image.
      - Processes each file and saves the reduced data.
    
    Parameters:
        data_dir: Directory containing FITS files to process
        top_level_dir: Top-level directory for resolving shared dark files
        nstack: Number of files to use for reference creation
        start_frame: Starting frame index
        step_frame: Frame skip step
        group_size: Number of frames to average
        remake_ref: Force recreation of reference
        skip_dark: Skip dark subtraction entirely
        subtract_reference: If False, skip reference subtraction; still normalize by noise.
    """
    subdir_name = os.path.basename(data_dir)
    logging.info("=" * 60)
    logging.info("Processing subdirectory: %s", subdir_name)
    logging.info("Full path: %s", data_dir)
    logging.info("=" * 60)
    
    ref_dir = os.path.join(data_dir, "references")
    processed_dir = os.path.join(data_dir, "reduced")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    quadrants = QUADRANTS
    
    # Load the dark image (assumed to be a FITS file of shape (120,120))
    if skip_dark:
        logging.info("Skipping dark subtraction")
        dark = None
    else:
        dark = load_dark_file(data_dir, top_level_dir)
    
    # Define the path for the reference image.
    reference_file = os.path.join(ref_dir, 'reference.fits')
    noise_file = os.path.join(ref_dir, 'noise.fits')
    
    # Process each FITS file in the input directory.
    fits_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.fits')])
    if not fits_files:
        logging.warning("[%s] No FITS files found in %s", subdir_name, data_dir)
        return False
    
    if nstack == None:
        nstack = len(fits_files)

    existing_reference_pair = os.path.exists(reference_file) and os.path.exists(noise_file)

    if not create_reference_flag:
        if not existing_reference_pair:
            logging.error(
                "[%s] CREATE_REFERENCE is false, but %s and %s do not both exist",
                subdir_name,
                reference_file,
                noise_file,
            )
            return False
        logging.info("[%s] Loading existing reference products from %s", subdir_name, ref_dir)
        reference, noise = load_reference_products(reference_file, noise_file)
    elif existing_reference_pair and not remake_ref:
        logging.info("[%s] Loading existing reference products from %s", subdir_name, ref_dir)
        reference, noise = load_reference_products(reference_file, noise_file)
    else:
        logging.info("[%s] Creating reference image from raw data in %s", subdir_name, data_dir)
        reference, noise = create_reference(
            data_dir,
            nstack,
            reference_file,
            noise_file,
            dark,
            remake_ref=True,
        )
    # # test no noise normalization
    # noise = np.ones_like(reference)
    if not subtract_reference:
        logging.info(
            "[%s] SUBTRACT_REFERENCE is false: noise normalization only (no reference subtraction).",
            subdir_name,
        )
    # make the pupil quadrant directories
    # now go through each of the FITS files and process sequentially
    for f in fits_files:
        input_path = os.path.join(data_dir, f)
        processed_cube_dict = process_file(
            input_path,
            reference,
            noise,
            quadrants,
            pupil_centers,
            pupil_mask_radius,
            start_frame,
            step_frame,
            group_size,
            dark,
            subtract_reference=subtract_reference,
            apply_tukey_window=apply_tukey_window,
            tukey_alpha=tukey_alpha,
        )
        for pup in quadrants:
            processed_quadrant = processed_cube_dict[pup]
            saveprocessed_to = f"{processed_dir}/{pup}"
            os.makedirs(saveprocessed_to, exist_ok=True)
            output_path = os.path.join(saveprocessed_to, f)
            write_fits_cube(output_path, processed_quadrant)
    
    logging.info("[%s] Completed processing", subdir_name)
    return True

def process_batch_in_memory(
    frames,
    data_dir,
    top_level_dir,
    nstack,
    pupil_centers,
    pupil_mask_radius,
    frames_per_cube,
    start_frame=0,
    step_frame=1,
    group_size=8,
    create_reference_flag=True,
    remake_ref=False,
    skip_dark=False,
    subtract_reference=True,
    inspect_reduction=False,
    tukey_alpha=0.0,
    apply_tukey_window=False,
):
    """Reduce a realtime batch in memory and return concatenated quadrant cubes.

    When ``inspect_reduction`` is true, writes ``reference.fits`` and ``noise.fits`` under
    ``data_dir/references/`` (same layout as batch file reduction) for debugging.
    """
    if frames.ndim != 3:
        raise ValueError(f"Expected batch frames with shape (n_frames, y, x), got {frames.shape}")

    raw_cube_count = frames.shape[0] // frames_per_cube
    if raw_cube_count == 0:
        raise ValueError(f"Need at least {frames_per_cube} frames to form one raw cube.")

    usable_frames = raw_cube_count * frames_per_cube
    dropped_frames = int(frames.shape[0] - usable_frames)
    raw_cubes = frames[:usable_frames].reshape(raw_cube_count, frames_per_cube, *frames.shape[1:])

    if skip_dark:
        logging.info("Skipping dark subtraction")
        dark = None
    else:
        dark = load_dark_file(data_dir, top_level_dir)

    reference_dir = os.path.join(data_dir, "references")
    reference_file = os.path.join(reference_dir, "reference.fits")
    noise_file = os.path.join(reference_dir, "noise.fits")
    existing_reference_pair = os.path.exists(reference_file) and os.path.exists(noise_file)

    if nstack is None:
        nstack = raw_cube_count

    if not create_reference_flag:
        if not existing_reference_pair:
            raise FileNotFoundError(
                f"CREATE_REFERENCE is false, but {reference_file} and {noise_file} do not both exist."
            )
        reference, noise = load_reference_products(reference_file, noise_file)
    elif existing_reference_pair and not remake_ref:
        logging.info("Loading existing reference products from %s", reference_dir)
        reference, noise = load_reference_products(reference_file, noise_file)
    else:
        reference, noise = create_reference_from_cubes(raw_cubes, nstack, dark_img=dark)

    if not subtract_reference:
        logging.info("SUBTRACT_REFERENCE is false: applying noise normalization only (no reference subtraction).")

    reduced_quadrants = {}
    reduced_frames_per_cube = None
    if tukey_alpha > 0.0:
        apply_tukey_window = True
        logging.info(f"Applying Tukey window with alpha = {tukey_alpha}")
    reduced_cube_dicts = [
        process_cube(
            cube,
            reference,
            noise,
            QUADRANTS,
            pupil_centers,
            pupil_mask_radius,
            start_frame,
            step_frame,
            group_size,
            dark,
            subtract_reference=subtract_reference,
            apply_tukey_window=apply_tukey_window,
            tukey_alpha=tukey_alpha,
        )
        for cube in raw_cubes
    ]
    for quadrant in QUADRANTS:
        reduced_cubes = [cube_dict[quadrant] for cube_dict in reduced_cube_dicts]
        if reduced_frames_per_cube is None:
            reduced_frames_per_cube = reduced_cubes[0].shape[0]
        reduced_quadrants[quadrant] = np.concatenate(reduced_cubes, axis=0)

    if inspect_reduction:
        os.makedirs(reference_dir, exist_ok=True)
        write_fits_cube(
            reference_file,
            np.asarray(reference, dtype=np.float32),
        )
        write_fits_cube(
            noise_file,
            np.asarray(noise, dtype=np.float32),
        )
        logging.info(
            "INSPECT_REDUCTION: wrote reference and noise to %s",
            reference_dir,
        )

    return {
        "reduced_quadrants": reduced_quadrants,
        "raw_cube_count": raw_cube_count,
        "reduced_frames_per_cube": int(reduced_frames_per_cube),
        "dropped_frames": dropped_frames,
        "reference": reference,
        "noise": noise,
    }

def get_reduce_settings(args, config_params):
    """Merge CLI compatibility flags with config-first reduction settings."""
    return {
        "nstack": args.nstack if args.nstack is not None else config_params.get("NSTACK"),
        "start_frame": args.start_frame if args.start_frame is not None else config_params.get("START_FRAME", DEFAULT_START_FRAME),
        "step_frame": args.step_frame if args.step_frame is not None else config_params.get("STEP_FRAME", DEFAULT_STEP_FRAME),
        "group_size": args.group_size if args.group_size is not None else config_params.get("GROUP_SIZE", DEFAULT_GROUP_SIZE),
        "workers": args.workers if args.workers is not None else config_params.get("WORKERS"),
        "create_reference": resolve_bool_setting(
            args.create_reference,
            config_params,
            "CREATE_REFERENCE",
            default=DEFAULT_CREATE_REFERENCE,
        ),
        "remake_reference": resolve_bool_setting(
            args.remake_ref,
            config_params,
            "REMAKE_REFERENCE",
            fallback_key="REMAKE_REF",
            default=DEFAULT_REMAKE_REFERENCE,
        ),
        "skip_dark": resolve_bool_setting(
            args.skip_dark,
            config_params,
            "SKIP_DARK",
            fallback_key="SUBTRACT_DARK",
            default=False,
        ),
        "subtract_reference": resolve_bool_setting(
            getattr(args, "subtract_reference", None),
            config_params,
            "SUBTRACT_REFERENCE",
            default=True,
        ),
        "apply_tukey_window": bool(config_params.get("APPLY_TUKEY_WINDOW", False)),
        "tukey_alpha": float(config_params.get("TUKEY_ALPHA", 0.5)),
    }

def main():
    parser = argparse.ArgumentParser(description="WindsoCC: camwfs data reduction")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=".",
        help="Directory containing ws_config.yaml or the path to ws_config.yaml itself.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional explicit path to ws_config.yaml. Overrides positional path resolution.",
    )
    parser.add_argument('--nstack', type=int, default=None,
                        help="N files in the dir to use for creating the ref image. Default: ALL")
    parser.add_argument('--start-frame', type=int, required=False, default=None,
                        help="Start the dataset on this frame; to remove sparkle bkg.")
    parser.add_argument('--step-frame', type=int, required=False, default=None,
                        help="Skip this number of frames to form the dataset; to remove sparkle bkg.")
    parser.add_argument('--group-size', type=int, required=False, default=None,
                        help="Number of frames to average; to remove sparkle bkg.")
    parser.add_argument('--remake-ref', dest='remake_ref', action='store_true', default=None,
                        help="Force recreation of the reference image even if one exists")
    parser.add_argument('--no-remake-ref', dest='remake_ref', action='store_false',
                        help="Reuse existing reference files when available.")
    parser.add_argument('--create-reference', dest='create_reference', action='store_true', default=None,
                        help="Create or refresh the reference products if needed.")
    parser.add_argument('--no-create-reference', dest='create_reference', action='store_false',
                        help="Require existing reference and noise files instead of creating them.")
    parser.add_argument('--skip-dark', dest='skip_dark', action="store_true", default=None,
                        help="Skip the dark frame subtraction.")
    parser.add_argument('--no-skip-dark', dest='skip_dark', action='store_false',
                        help="Apply dark subtraction when dark files are available.")
    ref_sub = parser.add_mutually_exclusive_group()
    ref_sub.add_argument(
        '--subtract-reference',
        dest='subtract_reference',
        action='store_true',
        default=None,
        help="Subtract mean reference before dividing by noise (overrides SUBTRACT_REFERENCE in config).",
    )
    ref_sub.add_argument(
        '--no-subtract-reference',
        dest='subtract_reference',
        action='store_false',
        default=None,
        help="Noise normalization only; do not subtract reference (overrides config).",
    )
    parser.add_argument('--workers', type=int, default=None,
                        help="Number of parallel workers (default: number of CPU cores)")
    
    # Parse arguments first
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    top_level_dir, config_path = resolve_config_path(args.data_dir, args.config)
    config_params = {}
    if os.path.exists(config_path):
        logging.info("Using config file: %s", config_path)
        try:
            config_params = parse_config_file(config_path)
        except Exception as e:
            logging.error("Error reading config file: %s", str(e))
            return
    else:
        logging.warning("No config file found at %s. Falling back to CLI/default values.", config_path)

    try:
        pupil_centers, pupil_mask_radius = get_pupil_geometry(config_params) if config_params else (
            {
                "ul": (30, 90),
                "ur": (90, 90),
                "ll": (30, 30),
                "lr": (90, 30),
            },
            DEFAULT_PUPIL_MASK_RADIUS,
        )
    except (KeyError, ValueError) as exc:
        logging.error("Invalid pupil geometry in config: %s", str(exc))
        return

    settings = get_reduce_settings(args, config_params)
    if "SUBTRACT_DARK" in config_params and "SKIP_DARK" not in config_params and args.skip_dark is None:
        logging.info("Using legacy SUBTRACT_DARK as a fallback for SKIP_DARK to preserve existing behavior.")
    
    # Find subdirectories to process
    subdirs = find_subdirectories(top_level_dir)
    
    if not subdirs:
        logging.error("No subdirectories with FITS files found in %s", top_level_dir)
        return
    
    logging.info("Found %d subdirectory(ies) to process:", len(subdirs))
    for subdir in subdirs:
        logging.info("  - %s", os.path.basename(subdir))
    
    # Determine number of workers
    if settings["workers"] is None:
        n_workers = cpu_count()
    else:
        n_workers = min(int(settings["workers"]), len(subdirs))  # Don't use more workers than subdirs
    
    logging.info("Using %d worker(s) for parallel processing", n_workers)
    
    # Process subdirectories in parallel
    results = []
    if len(subdirs) == 1 or n_workers == 1:
        # Single subdirectory or single worker - process sequentially
        logging.info("Processing sequentially")
        for subdir in subdirs:
            result = process_subdirectory(
                subdir,
                top_level_dir,
                settings["nstack"],
                pupil_centers,
                pupil_mask_radius,
                int(settings["start_frame"]),
                int(settings["step_frame"]),
                int(settings["group_size"]),
                settings["create_reference"],
                settings["remake_reference"],
                settings["skip_dark"],
                settings["subtract_reference"],
                settings["apply_tukey_window"],
                settings["tukey_alpha"],
            )
            results.append(result)
    else:
        # Multiple subdirectories - process in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    process_subdirectory,
                    subdir,
                    top_level_dir,
                    settings["nstack"],
                    pupil_centers,
                    pupil_mask_radius,
                    int(settings["start_frame"]),
                    int(settings["step_frame"]),
                    int(settings["group_size"]),
                    settings["create_reference"],
                    settings["remake_reference"],
                    settings["skip_dark"],
                    settings["subtract_reference"],
                    settings["apply_tukey_window"],
                    settings["tukey_alpha"],
                ): subdir
                for subdir in subdirs
            }
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    
    # Summary
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    logging.info("=" * 60)
    logging.info("Processing complete:")
    logging.info("  Successful: %d", successful)
    logging.info("  Failed: %d", failed)
    logging.info("=" * 60)
    
    if failed > 0:
        logging.warning("Failed subdirectories:")
        for subdir_path, success in results:
            if not success:
                logging.warning("  - %s", os.path.basename(subdir_path))


if __name__ == '__main__':
    main()
