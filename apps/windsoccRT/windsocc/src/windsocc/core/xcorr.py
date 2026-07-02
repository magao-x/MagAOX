"""Xcorr-stage helpers for cross-correlation analysis."""
import os
import logging
import numpy as np
from astropy.io import fits
from datetime import datetime

from windsocc.analysis.cross_correlation import (
    load_reduced_series,
    compute_all_delays_welch_optimized,
    compute_aperture_bias,
)
from windsocc.core.reduce import QUADRANTS

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

def compute_quadrant_xcorr_from_series(
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
    *,
    raw_frames_per_cube=512,
    loop_speed_hz=2000.0,
):
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
        # Subtracting the bias didn't seem to help much JKK 04/27/2026
        # cc_cube -= bias
        static_pattern = np.median(cc_cube, axis=0)
        # this replaces the bias subtraction
        cc_cube -= static_pattern
        # #debug view the static pattern
        # import matplotlib.pyplot as plt
        # plt.imshow(static_pattern, cmap='gray')
        # plt.colorbar()
        # plt.show()
        # import sys
        # sys.exit()

        header = fits.Header()
        delays_str = ",".join(
            str((raw_frames_per_cube / n_per_cube) * (1.0 / loop_speed_hz) * d)
            for d in delays
        )
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
                    fits_files = [
                        f for f in os.listdir(quadrant_path) 
                                if f.endswith('.fits') \
                                     and f.startswith('camwfs_') \
                                     and os.path.isfile(os.path.join(quadrant_path, f))]
                    if fits_files:
                        quadrant_dirs.append((item_path, quadrant, quadrant_path))
    
    return sorted(quadrant_dirs)

def save_xcorr_products(cc_cube, bias, header, output_path, bias_path):
    """Persist xcorr products to FITS for the file-based pipeline."""
    primary_hdu = fits.PrimaryHDU(cc_cube)
    if header is not None:
        primary_hdu.header.extend(header, update=True)
    primary_hdu.writeto(output_path, overwrite=True)
    fits.writeto(bias_path, bias, overwrite=True)

def process_quadrant_directory(
    quadrant_dir_path, quadrant, subdir_name, min_delay, max_delay,
    delay_step, segment_cubes, overlap, output_base_dir, fft_pad_shape,
    diam_pupils, raw_frames_per_cube, loop_speed_hz,
):
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
        logging.info(f"Processing: {subdir_name}/{quadrant}")
        logging.info(f"Path: {quadrant_dir_path}")
        logging.info(":" * 80)
        
        # Get first FITS file for filename generation
        fits_files = sorted([f for f in os.listdir(quadrant_dir_path) 
                           if f.endswith('.fits') and \
                            os.path.isfile(os.path.join(quadrant_dir_path, f)) and \
                            f.startswith('camwfs_')])
        if len(fits_files) == 0:
            logging.warning(f"[{subdir_name}/{quadrant}] No FITS files found in {quadrant_dir_path}")
            return False

        first_fits_basename = os.path.splitext(fits_files[0])[0]
        
        logging.info(f"[{subdir_name}/{quadrant}] Loading reduced series from {quadrant_dir_path}")
        time_series, n_per_cube, skipped_cubes = load_reduced_series(quadrant_dir_path, diam_pupils)
        logging.info(f"[{subdir_name}/{quadrant}] Loaded reduced series of size {len(time_series)}")
        if skipped_cubes > 0:
            logging.info(f"[{subdir_name}/{quadrant}] Skipped {skipped_cubes} cubes due to either \
                not having sufficient frames or not having the expected frame size.")
        
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
            raw_frames_per_cube=raw_frames_per_cube,
            loop_speed_hz=loop_speed_hz,
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

def resolve_xcorr_settings(config_params, overrides=None):
    """Resolve xcorr settings from config with optional overrides."""
    overrides = overrides or {}
    min_delay = overrides.get("min_delay", config_params.get("MIN_DELAY"))
    max_delay = overrides.get("max_delay", config_params.get("MAX_DELAY"))
    delay_step = overrides.get("delay_step", config_params.get("DELAY_STEP", 1))
    segment_cubes = overrides.get(
        "segment_cubes",
        config_params.get("SEGMENT_LENGTH", 22),
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
    raw_frames_per_cube = int(config_params.get("RAW_FRAMES_PER_CUBE", 512))
    loop_speed_hz = float(config_params.get("LOOP_SPEED", 2000.0))
    return {
        "min_delay": int(min_delay),
        "max_delay": int(max_delay),
        "delay_step": int(delay_step),
        "segment_cubes": max(1, int(segment_cubes)),
        "overlap": float(overlap),
        "fft_pad_shape": fft_pad_shape,
        "workers": workers,
        "raw_frames_per_cube": raw_frames_per_cube,
        "loop_speed_hz": loop_speed_hz,
    }

