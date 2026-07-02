"""Cross-correlation stage for reduced camwfs pupil cubes."""

import os
import argparse
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from windsocc.io.config_handling import parse_config_file
from windsocc.core.reduce import QUADRANTS
from windsocc.core.xcorr import (
    compute_quadrant_xcorr_from_series,
    find_quadrant_directories,
    process_quadrant_directory,
    resolve_xcorr_settings,
)

def parse_center(center_str):
    """Parse a comma-separated string (e.g., '60,60') into a tuple of two integers."""
    try:
        parts = center_str.split(',')
        if len(parts) != 2:
            raise ValueError
        return (int(parts[0]), int(parts[1]))
    except Exception:
        raise argparse.ArgumentTypeError("Center must be in the format 'x,y' with two integers.")

def main():
    parser = argparse.ArgumentParser(description="Compute cross-correlation maps for all pupil positions.")

    parser.add_argument(
        "data_dir",
        nargs="?",
        default=".",
        help="Directory containing ws_config.yaml or the path to ws_config.yaml itself.",
    )
    parser.add_argument('--min-delay', type=int, default=None,
                        help="Minimum delay (in frames) for cross-correlation. Can be set via config file.")
    parser.add_argument('--max-delay', type=int, default=None,
                        help="Maximum delay (in frames) for cross-correlation. Can be set via config file.")
    parser.add_argument('--delay-step', type=int, default=None,
                        help="Step size for delay values (in frames). Default: 1 or from config.")
    parser.add_argument(
        '--segment-cubes', type=int, default=None,
        help="Number of cubes per segment; overrides SEGMENT_LENGTH in config (default: 22).",
    )
    parser.add_argument('--overlap', type=float, default=None,
                        help="Fractional overlap between segments. Default: 0.5 or from config.")
    parser.add_argument('--fft-pad-shape', type=parse_fft_pad_shape, default=None,
                        help="FFT padding shape as 'height,width' (e.g., '239,239'). Default: auto-calculated.")
    parser.add_argument('--workers', type=int, default=None,
                        help="Number of parallel workers (default: number of CPU cores)")
    
    # Parse arguments first
    args = parser.parse_args()
    # Resolve absolute path for top-level directory
    top_level_dir = os.path.abspath(args.data_dir)

    # Create output directory in the working data subdirectory
    output_base_dir = os.path.join(top_level_dir, "xcorr_results")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Set up logging and save the output to a log file
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info(f"Output directory: {output_base_dir}")
    # # Create logger
    # logger = logging.getLogger('ws_xcorr')
    # logger.setLevel(logging.DEBUG)

    # # Create file handler
    # file_handler = logging.FileHandler(f'{output_base_dir}/xcorr_{current_time}.log')
    # file_handler.setLevel(logging.DEBUG)

    # # Create formatter
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler.setFormatter(formatter)

    # # Add handler to logger
    # logger.addHandler(file_handler)
    
    # Check for config file
    config_path = os.path.join(top_level_dir, 'ws_config.yaml')
    config_params = {}
    # if no config file, that means the pipeline is being run out of order; throw error

    if os.path.exists(config_path):
        logging.info(f"Loading config file: {config_path}")
        try:
            config_params = parse_config_file(config_path)
            logging.info("Config file loaded successfully!")
        except Exception as e:
            logging.warning(f"Error reading config file: {str(e)}. \
                \nUsing defaults and CLI arguments.", str(e))
    else:
        logging.error(f"No config file found at {config_path}.")
        logging.error("Pipeline is being run out of order.  \
        \nPlease start with ws_partition to generate a config file.")
        return
    
    # Set parameters from config or defaults, CLI overrides
    min_delay = args.min_delay if args.min_delay is not None else config_params.get('MIN_DELAY')
    max_delay = args.max_delay if args.max_delay is not None else config_params.get('MAX_DELAY')
    delay_step = args.delay_step if args.delay_step is not None else config_params.get('DELAY_STEP', 1)
    segment_cubes = args.segment_cubes if args.segment_cubes is not None else config_params.get('SEGMENT_LENGTH', 22)
    raw_frames_per_cube = int(config_params.get('RAW_FRAMES_PER_CUBE', 512))
    loop_speed_hz = float(config_params.get('LOOP_SPEED', 2000.0))
    diam_pupils = config_params.get("DIAM_PUPILS")
    overlap = args.overlap if args.overlap is not None else config_params.get('OVERLAP', 0.5)
    fft_pad_shape = args.fft_pad_shape if args.fft_pad_shape is not None else config_params.get('FFT_PAD_SHAPE', None)
    workers = args.workers if args.workers is not None else config_params.get('WORKERS')

    # Validate required parameters
    if min_delay is None:
        logging.error("min-delay must be provided either via --min-delay argument or MIN_DELAY in config file")
        return
    if max_delay is None:
        logging.error("max-delay must be provided either via --max-delay argument or MAX_DELAY in config file")
        return
    
    # logging.info("Cross-correlation parameters:")
    # logging.info(f"  min_delay: {min_delay}")
    # logging.info(f"  max_delay: {max_delay}")
    # logging.info(f"  delay_step: {delay_step}")
    # logging.info(f"  segment_cubes: {segment_cubes}")
    # logging.info(f"  overlap: {overlap}")
    # logging.info(f"  fft_pad_shape: {fft_pad_shape}")
    
    # Find all quadrant directories
    quadrant_dirs = find_quadrant_directories(top_level_dir)

    
    if not quadrant_dirs:
        logging.error("No quadrant directories found; please run ws_reduce first.")
        logging.error(f"Expected structure: {top_level_dir}/reduced/{quadrant}/ with FITS files")
        return
    
    logging.info(f"Found {len(quadrant_dirs)} quadrant directory(ies) to process")
    logging.info(f"... or {int(len(quadrant_dirs) / 4)} subdirectory(ies) in total.")

    
    # Determine number of workers
    if workers is None:
        n_workers = cpu_count()
    else:
        n_workers = min(workers, len(quadrant_dirs))  # Don't use more workers than quadrants
    
    
    # Process each quadrant directory in parallel
    results = []
    overall_starttime = datetime.now()
    
    if len(quadrant_dirs) == 1 or n_workers == 1:
        # Single quadrant or single worker - process sequentially
        logging.info("Processing sequentially...!")
        for subdir_path, quadrant, quadrant_path in quadrant_dirs:
            subdir_name = os.path.basename(subdir_path)
            success = process_quadrant_directory(
                quadrant_path, quadrant, subdir_name, min_delay, max_delay,
                delay_step, segment_cubes, overlap, output_base_dir, fft_pad_shape,
                diam_pupils, raw_frames_per_cube, loop_speed_hz,
            )
            results.append((subdir_name, quadrant, success))
    else:
        logging.info(f"Using {n_workers} worker(s) for parallel processing")
        # Multiple quadrants - process in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_quadrant_directory, quadrant_path, quadrant,
                                      os.path.basename(subdir_path), min_delay, max_delay,
                                      delay_step, segment_cubes, overlap, output_base_dir, fft_pad_shape,
                                      diam_pupils, raw_frames_per_cube, loop_speed_hz): (subdir_path, quadrant)
                      for subdir_path, quadrant, quadrant_path in quadrant_dirs}
            
            for future in as_completed(futures):
                result = future.result()
                subdir_path, quadrant = futures[future]
                subdir_name = os.path.basename(subdir_path)
                results.append((subdir_name, quadrant, result))
    
    # Summary
    successful = sum(1 for _, _, success in results if success)
    failed = len(results) - successful
    
    logging.info(":" * 80)
    logging.info("Processing complete:")
    logging.info(f"  Successful: {successful}")
    logging.info(f"  Failed: {failed}")
    logging.info(f"  Total time: {datetime.now() - overall_starttime}")
    logging.info(":" * 80)
    
    if failed > 0:
        logging.warning("Failed quadrant directories:")
        for subdir_name, quadrant, success in results:
            if not success:
                logging.warning(f"  - {subdir_name} / {quadrant}")

def parse_fft_pad_shape(shape_str):
    """Parse a comma-separated string (e.g., '239,239') into a tuple of two integers for FFT padding."""
    try:
        parts = shape_str.split(',')
        if len(parts) != 2:
            raise ValueError
        return (int(parts[0]), int(parts[1]))
    except Exception:
        raise argparse.ArgumentTypeError("FFT pad shape must be in the format 'height,width' with two integers.")

def run_xcorr_stage_in_memory(run_dir, reduced_products, config_params=None, overrides=None):
    """Run xcorr for one realtime batch using in-memory reduced quadrant cubes."""
    if config_params is None:
        config_path = os.path.join(run_dir, "ws_config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config file found at {config_path}.")
        config_params = parse_config_file(config_path)

    settings = resolve_xcorr_settings(config_params, overrides=overrides)
    run_name = os.path.basename(run_dir)
    quadrant_results = {}
    results = []
    for quadrant in QUADRANTS:
        time_series = reduced_products["reduced_quadrants"].get(quadrant)
        if time_series is None:
            results.append((quadrant, False))
            continue

        result = compute_quadrant_xcorr_from_series(
            time_series,
            reduced_products["reduced_frames_per_cube"],
            quadrant,
            run_name,
            settings["min_delay"],
            settings["max_delay"],
            settings["delay_step"],
            settings["segment_cubes"],
            settings["overlap"],
            settings["fft_pad_shape"],
            raw_frames_per_cube=settings["raw_frames_per_cube"],
            loop_speed_hz=settings["loop_speed_hz"],
        )
        results.append((quadrant, result["success"]))
        if result["success"]:
            quadrant_results[quadrant] = result

    return {
        "results": results,
        "settings": settings,
        "quadrant_results": quadrant_results,
        "group_suffix": reduced_products["group_suffix"],
    }


if __name__ == '__main__':
    main()
