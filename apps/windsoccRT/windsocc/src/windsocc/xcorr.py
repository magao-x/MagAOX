'''
From camwfs experiment:

Sine pattern travelling E --> W (270 deg) on DM:
camwfs: 297.5 deg propagation direction
camsci1: 242.5 (SW) or 62.5 (NE) degree sparkle orientation
'''


import os
import argparse
import logging
import numpy as np
from astropy.io import fits
from datetime import datetime
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import sys
from windsocc.analysis.cross_correlation import load_reduced_series, compute_all_delays_welch_optimized
from windsocc.analysis.cross_correlation import compute_aperture_bias
from windsocc.preprocessing.radial import radial_profile
# from windsocc.visualization.make_wind_movie import save_cube_as_movie

QUADRANTS = ("ul", "ur", "ll", "lr")

def parse_center(center_str):
    """Parse a comma-separated string (e.g., '60,60') into a tuple of two integers."""
    try:
        parts = center_str.split(',')
        if len(parts) != 2:
            raise ValueError
        return (int(parts[0]), int(parts[1]))
    except Exception:
        raise argparse.ArgumentTypeError("Center must be in the format 'x,y' with two integers.")

def parse_fft_pad_shape(shape_str):
    """Parse a comma-separated string (e.g., '239,239') into a tuple of two integers for FFT padding."""
    try:
        parts = shape_str.split(',')
        if len(parts) != 2:
            raise ValueError
        return (int(parts[0]), int(parts[1]))
    except Exception:
        raise argparse.ArgumentTypeError("FFT pad shape must be in the format 'height,width' with two integers.")


def mask_cc_cube_center(cube, radius, mask_value=np.nan):
    """
    Masks the central region of each frame in a 3D image cube using a circular mask.

    Parameters
    ----------
    cube : np.ndarray
        3D array with shape (n_frames, height, width) representing the image cube.
    radius : float or int
        Radius in pixels for the circular mask, centered on the spatial center of each frame.
    mask_value : float, optional
        Value to assign within the masked region (default: np.nan). Using np.nan
        is useful because many visualization routines ignore NaNs when scaling colors.
    
    Returns
    -------
    masked_cube : np.ndarray
        A new 3D array where the central circular region in each frame has been replaced with mask_value.
    """
    # Copy the cube to avoid modifying the original data.
    masked_cube = cube.copy()
    n_frames, rows, cols = cube.shape
    center_row, center_col = rows // 2, cols // 2

    # Create a 2D boolean mask for the circular region.
    y, x = np.ogrid[:rows, :cols]
    mask = (x - center_col)**2 + (y - center_row)**2 <= radius**2

    # Apply the mask to each frame using broadcasting.
    # This sets the pixel values in the masked region to mask_value for all frames.
    masked_cube[:, mask] = mask_value

    return masked_cube


def find_quadrant_directories(top_level_dir):
    """
    Find all subdirectories containing reduced/{quadrant}/ directories.
    Filters out special directories like 'darks', 'references', 'results_cubes'.
    
    Returns:
        List of tuples: (subdir_path, quadrant, quadrant_dir_path)
    """
    quadrant_dirs = []
    quadrants = ['ul', 'ur', 'll', 'lr']
    special_dirs = {'darks', 'references', 'results_cubes', 'reduced'}
    
    if not os.path.isdir(top_level_dir):
        logging.error(f"Top-level directory does not exist: {top_level_dir}")
        return quadrant_dirs
    
    # Check each subdirectory
    for item in os.listdir(top_level_dir):
        item_path = os.path.join(top_level_dir, item)
        
        # Skip if not a directory or is a special directory
        if not os.path.isdir(item_path) or item in special_dirs:
            continue
        
        # Check for reduced/{quadrant}/ directories
        reduced_path = os.path.join(item_path, 'reduced')
        if os.path.exists(reduced_path):
            for quadrant in quadrants:
                quadrant_path = os.path.join(reduced_path, quadrant)
                if os.path.isdir(quadrant_path):
                    # Check if directory contains FITS files
                    fits_files = [f for f in os.listdir(quadrant_path) 
                                if f.endswith('.fits') and os.path.isfile(os.path.join(quadrant_path, f))]
                    if fits_files:
                        quadrant_dirs.append((item_path, quadrant, quadrant_path))
    
    return sorted(quadrant_dirs)


def process_quadrant_directory(quadrant_dir_path, quadrant, subdir_name, min_delay, max_delay, 
                               delay_step, segment_cubes, overlap, output_base_dir, fft_pad_shape):
    """
    Process a single quadrant directory:
      - Loads all FITS files from the quadrant directory
      - Computes cross-correlation maps for specified delays
      - Generates output filename automatically
      - Saves the cross-correlation cube
    
    Parameters:
        quadrant_dir_path: Full path to the quadrant directory (e.g., .../reduced/ll/)
        quadrant: Quadrant code ('ul', 'ur', 'll', 'lr')
        subdir_name: Name of the subdirectory (for logging)
        min_delay: Minimum delay in frames
        max_delay: Maximum delay in frames
        delay_step: Step size for delay values
        segment_cubes: Number of cubes per segment
        overlap: Fractional overlap between segments
        output_base_dir: Base directory for output files
    
    Returns:
        Success boolean
    """
    try:
        logging.info(":" * 80)
        logging.info(f"Processing: {subdir_name} / {quadrant}")
        logging.info(f"Path: {quadrant_dir_path}")
        logging.info(":" * 80)
        
        # Get first FITS file for filename generation
        fits_files = sorted([f for f in os.listdir(quadrant_dir_path) 
                           if f.endswith('.fits') and os.path.isfile(os.path.join(quadrant_dir_path, f))])
        
        if not fits_files:
            logging.warning(f"[{subdir_name}/{quadrant}] No FITS files found in {quadrant_dir_path}")
            return False

        first_fits_basename = os.path.splitext(fits_files[0])[0]
        
        logging.info(f"[{subdir_name}/{quadrant}] Loading reduced series from {quadrant_dir_path}")
        time_series, n_per_cube, skipped_cubes = load_reduced_series(quadrant_dir_path)
        logging.info(f"[{subdir_name}/{quadrant}] Loaded reduced series of size {len(time_series)}")
        logging.info(f"[{subdir_name}/{quadrant}] Skipped {skipped_cubes} cubes due to not having insufficient frames.")
        
        result = compute_quadrant_xcorr_from_series(
            time_series,
            n_per_cube,
            quadrant,
            subdir_name,
            min_delay,
            max_delay,
            delay_step,
            segment_cubes,
            overlap,
            fft_pad_shape,
        )
        if not result["success"]:
            return False

        # disabled radial profile sub; to be put in ws_distill 01/12/2026 JKK
        # for i in range(len(delays)):
        #     rprofile = radial_profile(cc_cube[i])
        #     cc_cube[i] -= rprofile

        # diabled CC peak masking; to be put in ws_distill 01/12/2026 JKK
        # cc_rprofsub_masked = mask_cc_cube_center(cc_cube, radius=3, mask_value=0.)

        # Generate output filename
        output_filename = f"{quadrant}_{first_fits_basename}_{min_delay}min{max_delay}max{delay_step}delay.fits"
        bias_filename = f"{quadrant}_{first_fits_basename}_bias.fits"
        bias_dir = f"{output_base_dir}/biases"
        os.makedirs(bias_dir,exist_ok=True)
        bias_path = os.path.join(bias_dir, bias_filename)

        output_path = os.path.join(output_base_dir, output_filename)

        save_xcorr_products(result["cc_cube"], result["bias"], result["header"], output_path, bias_path)
        # logging.info(f"[{subdir_name}/{quadrant}] Saved cross-correlation maps cube to {output_path}")
        # logging.info(f"[{subdir_name}/{quadrant}] Saved bias cube to {bias_path}")
        elapsed_time = datetime.now() - result["starttime"]
        logging.info(f"[{subdir_name}/{quadrant}] Processing took {elapsed_time}")
        
        return True
        
    except Exception as e:
        logging.error(f"[{subdir_name}/{quadrant}] Error processing quadrant directory: {str(e)}")
        return False


def compute_quadrant_xcorr_from_series(time_series, n_per_cube, quadrant, subdir_name, min_delay,
                                       max_delay, delay_step, segment_cubes, overlap, fft_pad_shape):
    """Compute xcorr products for one quadrant time series without persisting them."""
    try:
        # logging.info(f"[{subdir_name}/{quadrant}] Computing aperture bias")
        time_series_median = np.median(time_series, axis=0)
        bias = compute_aperture_bias(time_series_median, fft_pad_shape=fft_pad_shape)
        delays = list(range(min_delay, max_delay + 1, delay_step))
        starttime = datetime.now()

        # logging.info(f"[{subdir_name}/{quadrant}] Computing cross-correlation maps for all delays (optimized FFT pre-computation)")
        # logging.info(f"[{subdir_name}/{quadrant}] Processing {len(delays)} delays: {min_delay} to {max_delay} (step {delay_step})")

        cc_maps = compute_all_delays_welch_optimized(
            time_series,
            n_per_cube,
            delays,
            segment_cube_count=segment_cubes,
            overlap_fraction=overlap,
            fft_pad_shape=fft_pad_shape,
        )

        cc_cube = np.array(cc_maps)
        cc_cube -= bias
        static_pattern = np.median(cc_cube, axis=0)
        cc_cube -= static_pattern

        header = fits.Header()
        delays_str = ",".join(str((512 / n_per_cube) * 1 / 2000 * d) for d in delays)
        header["DELAYARR"] = (delays_str, "Comma-separated delays for each frame")
        return {
            "success": True,
            "cc_cube": cc_cube,
            "bias": bias,
            "header": header,
            "delays": delays,
            "starttime": starttime,
        }
    except Exception as e:
        logging.error(f"[{subdir_name}/{quadrant}] Error computing xcorr products: {str(e)}")
        return {"success": False, "error": str(e)}


def save_xcorr_products(cc_cube, bias, header, output_path, bias_path):
    """Persist xcorr products to FITS for the file-based pipeline."""
    primary_hdu = fits.PrimaryHDU(cc_cube)
    if header is not None:
        primary_hdu.header.extend(header, update=True)
    primary_hdu.writeto(output_path, overwrite=True)
    fits.writeto(bias_path, bias, overwrite=True)


def parse_config_file(config_path):
    """Load the xcorr-stage config file."""
    with open(config_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file) or {}


def resolve_xcorr_settings(config_params, overrides=None):
    """Resolve xcorr settings from config with optional overrides."""
    overrides = overrides or {}
    min_delay = overrides.get("min_delay", config_params.get("MIN_DELAY"))
    max_delay = overrides.get("max_delay", config_params.get("MAX_DELAY"))
    delay_step = overrides.get("delay_step", config_params.get("DELAY_STEP", 1))
    segment_cubes = overrides.get(
        "segment_cubes",
        config_params.get("SEGMENT_CUBES", config_params.get("SEGMENT_LENGTH", 22)),
    )
    overlap = overrides.get("overlap", config_params.get("OVERLAP", 0.5))
    fft_pad_shape = overrides.get("fft_pad_shape", config_params.get("FFT_PAD_SHAPE", None))
    workers = overrides.get("workers", config_params.get("WORKERS"))
    if min_delay is None:
        raise ValueError(
            "min-delay must be provided either via override or MIN_DELAY in config file"
        )
    if max_delay is None:
        raise ValueError(
            "max-delay must be provided either via override or MAX_DELAY in config file"
        )
    return {
        "min_delay": int(min_delay),
        "max_delay": int(max_delay),
        "delay_step": int(delay_step),
        "segment_cubes": max(1, int(segment_cubes)),
        "overlap": float(overlap),
        "fft_pad_shape": fft_pad_shape,
        "workers": workers,
    }


def get_realtime_quadrant_directories(run_dir):
    """Return the four realtime reduced quadrant directories under one run root."""
    reduced_path = os.path.join(run_dir, "reduced")
    quadrant_dirs = []
    run_name = os.path.basename(run_dir)
    for quadrant in QUADRANTS:
        quadrant_path = os.path.join(reduced_path, quadrant)
        if not os.path.isdir(quadrant_path):
            continue
        fits_files = [
            f
            for f in os.listdir(quadrant_path)
            if f.endswith(".fits") and os.path.isfile(os.path.join(quadrant_path, f))
        ]
        if fits_files:
            quadrant_dirs.append((run_name, quadrant, quadrant_path))
    return quadrant_dirs


def run_xcorr_stage(run_dir, config_params=None, overrides=None):
    """Run xcorr for a single realtime batch directory."""
    if config_params is None:
        config_path = os.path.join(run_dir, "ws_config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config file found at {config_path}.")
        config_params = parse_config_file(config_path)

    settings = resolve_xcorr_settings(config_params, overrides=overrides)
    output_dir_name = config_params.get("XCORR_DIR", "xcorr_results")
    output_base_dir = (
        output_dir_name
        if os.path.isabs(output_dir_name)
        else os.path.join(run_dir, output_dir_name)
    )
    os.makedirs(output_base_dir, exist_ok=True)

    quadrant_dirs = get_realtime_quadrant_directories(run_dir)
    if not quadrant_dirs:
        raise FileNotFoundError(
            f"No realtime quadrant directories found under {os.path.join(run_dir, 'reduced')}."
        )

    workers = settings["workers"]
    if workers is None:
        n_workers = min(cpu_count(), len(quadrant_dirs))
    else:
        n_workers = min(int(workers), len(quadrant_dirs))

    results = []
    if len(quadrant_dirs) == 1 or n_workers <= 1:
        for subdir_name, quadrant, quadrant_dir_path in quadrant_dirs:
            results.append(
                (
                    quadrant_dir_path,
                    process_quadrant_directory(
                        quadrant_dir_path,
                        quadrant,
                        subdir_name,
                        settings["min_delay"],
                        settings["max_delay"],
                        settings["delay_step"],
                        settings["segment_cubes"],
                        settings["overlap"],
                        output_base_dir,
                        settings["fft_pad_shape"],
                    ),
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    process_quadrant_directory,
                    quadrant_dir_path,
                    quadrant,
                    subdir_name,
                    settings["min_delay"],
                    settings["max_delay"],
                    settings["delay_step"],
                    settings["segment_cubes"],
                    settings["overlap"],
                    output_base_dir,
                    settings["fft_pad_shape"],
                ): quadrant_dir_path
                for subdir_name, quadrant, quadrant_dir_path in quadrant_dirs
            }
            for future in as_completed(futures):
                quadrant_dir_path = futures[future]
                results.append((quadrant_dir_path, future.result()))

    return {
        "output_base_dir": output_base_dir,
        "quadrant_dirs": quadrant_dirs,
        "results": results,
        "settings": settings,
    }


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


def main():
    parser = argparse.ArgumentParser(description="Compute cross-correlation maps for all pupil positions.")
    parser.add_argument('-d', '--data-dir', type=str, default=".",
                        help="Top-level directory containing subdirectories with reduced FITS cubes")
    parser.add_argument('--min-delay', type=int, default=None,
                        help="Minimum delay (in frames) for cross-correlation. Can be set via config file.")
    parser.add_argument('--max-delay', type=int, default=None,
                        help="Maximum delay (in frames) for cross-correlation. Can be set via config file.")
    parser.add_argument('--delay-step', type=int, default=None,
                        help="Step size for delay values (in frames). Default: 1 or from config.")
    parser.add_argument('--segment-cubes', type=int, default=None,
                        help="Number of cubes to use per segment. Default: 22 or from config.")
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
            with open(config_path, 'r') as yaml_file:
                config_params = yaml.safe_load(yaml_file) or {}
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
    segment_cubes = args.segment_cubes if args.segment_cubes is not None else config_params.get('SEGMENT_CUBES', 22)
    overlap = args.overlap if args.overlap is not None else config_params.get('OVERLAP', 0.5)
    fft_pad_shape = args.fft_pad_shape if args.fft_pad_shape is not None else config_params.get('FFT_PAD_SHAPE', None)

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
    if args.workers is None:
        n_workers = cpu_count()
    else:
        n_workers = min(args.workers, len(quadrant_dirs))  # Don't use more workers than quadrants
    
    
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
                delay_step, segment_cubes, overlap, output_base_dir, fft_pad_shape
            )
            results.append((subdir_name, quadrant, success))
    else:
        logging.info(f"Using {n_workers} worker(s) for parallel processing")
        # Multiple quadrants - process in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_quadrant_directory, quadrant_path, quadrant,
                                      os.path.basename(subdir_path), min_delay, max_delay,
                                      delay_step, segment_cubes, overlap, output_base_dir, fft_pad_shape): (subdir_path, quadrant)
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

if __name__ == '__main__':
    main()