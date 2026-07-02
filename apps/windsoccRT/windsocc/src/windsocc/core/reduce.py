"""Reduce-stage helpers for camwfs data reduction."""
from __future__ import annotations

import os
import glob
import logging
import numpy as np
from astropy.io import fits
from skimage.measure import block_reduce

from windsocc.io.fits_handling import read_fits_cube, write_fits_cube
from windsocc.preprocessing.reference_camwfs import create_reference
from windsocc.preprocessing.crop_pupil_camwfs import crop_quadrant

DEFAULT_START_FRAME = 0


DEFAULT_STEP_FRAME = 1


DEFAULT_GROUP_SIZE = 8


DEFAULT_CREATE_REFERENCE = True


DEFAULT_REMAKE_REFERENCE = False


DEFAULT_PUPIL_MASK_RADIUS = 28


QUADRANTS = ("ul", "ur", "ll", "lr")


def find_subdirectories(top_level_dir):
    """
    Find all subdirectories containing FITS files.
    Filters out special directories like 'darks', 'references', 'reduced'.
    
    Returns:
        List of subdirectory paths containing FITS files.
    """
    subdirs = []
    special_dirs = {'darks', 'references', 'reduced'}
    
    if not os.path.isdir(top_level_dir):
        logging.error("Top-level directory does not exist: %s", top_level_dir)
        return subdirs
    
    # Check if top-level directory itself has FITS files
    fits_in_top = [f for f in os.listdir(top_level_dir) 
                   if f.endswith('.fits') and os.path.isfile(os.path.join(top_level_dir, f))]
    
    if fits_in_top:
        # If top-level has FITS files, check if there are also subdirectories
        all_items = os.listdir(top_level_dir)
        subdirs_found = [item for item in all_items 
                        if os.path.isdir(os.path.join(top_level_dir, item)) 
                        and item not in special_dirs]
        
        if subdirs_found:
            # Process subdirectories
            for item in subdirs_found:
                subdir_path = os.path.join(top_level_dir, item)
                fits_files = [f for f in os.listdir(subdir_path) 
                            if f.endswith('.fits') and os.path.isfile(os.path.join(subdir_path, f))]
                if fits_files:
                    subdirs.append(subdir_path)
        else:
            # No subdirectories, top-level itself is the data directory
            subdirs.append(top_level_dir)
    else:
        # No FITS files in top-level, look for subdirectories
        for item in os.listdir(top_level_dir):
            item_path = os.path.join(top_level_dir, item)
            if os.path.isdir(item_path) and item not in special_dirs and item.startswith("camwfs_"):
                fits_files = [f for f in os.listdir(item_path) 
                            if f.endswith('.fits') and os.path.isfile(os.path.join(item_path, f))]
                if fits_files:
                    subdirs.append(item_path)
    
    return sorted(subdirs)

def _tukey_window_1d(length, alpha):
    """Return a 1D Tukey window of given length and shape parameter alpha in [0, 1]."""
    if length <= 0:
        raise ValueError("Tukey window length must be positive.")

    if alpha <= 0:
        return np.ones(length, dtype=np.float32)

    if alpha >= 1:
        return np.hanning(length).astype(np.float32)

    n = length - 1
    w = np.ones(length, dtype=np.float32)
    edge = int(alpha * n / 2.0)
    if edge == 0:
        return w

    for i in range(edge):
        w[i] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * i / (alpha * n) - 1.0)))
        w[n - i] = w[i]

    return w

def process_subdirectory(
    subdir_path,
    top_level_dir,
    nstack,
    pupil_centers,
    pupil_mask_radius,
    start_frame,
    step_frame,
    group_size,
    create_reference_flag,
    remake_ref,
    skip_dark,
    subtract_reference=True,
    apply_tukey_window=False,
    tukey_alpha=0.5,
):
    """
    Wrapper function to process a single subdirectory.
    This is designed to be called in parallel.
    
    Returns:
        Tuple of (subdir_path, success_boolean)
    """
    try:
        from windsocc.ws_reduce import process_dataset
        success = process_dataset(
            subdir_path,
            top_level_dir,
            nstack,
            pupil_centers,
            pupil_mask_radius,
            start_frame,
            step_frame,
            group_size,
            create_reference_flag,
            remake_ref,
            skip_dark,
            subtract_reference=subtract_reference,
            apply_tukey_window=apply_tukey_window,
            tukey_alpha=tukey_alpha,
        )
        return (subdir_path, success)
    except Exception as e:
        logging.error("[%s] Error processing subdirectory: %s", 
                     os.path.basename(subdir_path), str(e))
        return (subdir_path, False)

def create_reference_from_cubes(cubes, nstack, dark_img=None):
    """Create reference and noise images directly from in-memory raw cubes."""
    if cubes.ndim != 4:
        raise ValueError(f"Expected raw cubes with shape (n_cubes, n_frames, y, x), got {cubes.shape}")

    if cubes.shape[0] == 0:
        raise ValueError("At least one raw cube is required to create an in-memory reference.")

    cubes_to_use = cubes[: min(int(nstack), cubes.shape[0])]
    if dark_img is not None:
        cubes_ds = cubes_to_use - dark_img
    else:
        cubes_ds = cubes_to_use

    mean_frames = np.mean(cubes_ds, axis=1)
    std_frames = np.std(cubes_ds, axis=1)
    reference = np.mean(mean_frames, axis=0)
    noise = np.mean(std_frames, axis=0) + 1
    return reference, noise

def _tukey_window_2d(shape, alpha):
    """Return a 2D Tukey window with the given (ny, nx) shape."""
    if len(shape) != 2:
        raise ValueError(f"Tukey window requires a 2D shape, got {shape!r}")
    ny, nx = shape
    wy = _tukey_window_1d(ny, alpha)
    wx = _tukey_window_1d(nx, alpha)
    return np.outer(wy, wx).astype(np.float32)

def process_cube(
    cube,
    reference,
    noise,
    quadrants,
    pupil_centers,
    pupil_mask_radius,
    start_frame,
    num_frames_skip,
    group_size,
    dark=None,
    subtract_reference=True,
    apply_tukey_window=False,
    tukey_alpha=0.5,
):
    """Process one in-memory raw cube and return the reduced cropped cube."""
    if dark is not None:
        cube_ds = cube - dark
    else:
        cube_ds = cube
    if subtract_reference:
        cube_rs = cube_ds - reference
    else:
        cube_rs = cube_ds
    noise_mean = np.mean(noise)
    noise_safe = noise.copy()
    noise_safe[noise_safe == 0] = noise_mean
    cube_norm = cube_rs / noise_safe
    cube_reduced = block_reduce(cube_norm, block_size=(group_size, 1, 1), func=np.mean)
    cube_no_spark = cube_reduced[start_frame::num_frames_skip]

    ul_cropped_frames = []
    ur_cropped_frames = []
    ll_cropped_frames = []
    lr_cropped_frames = []
    tukey_kernel = None
    for each, frame in enumerate(cube_no_spark):
        # need to grab the total irradiance from all 4 pupils
        quadrant_thumbnails = []
        ul_this_frame = np.zeros((pupil_mask_radius*2, pupil_mask_radius*2))
        ur_this_frame = np.zeros((pupil_mask_radius*2, pupil_mask_radius*2))
        ll_this_frame = np.zeros((pupil_mask_radius*2, pupil_mask_radius*2))
        lr_this_frame = np.zeros((pupil_mask_radius*2, pupil_mask_radius*2))
        for q in quadrants:
            cropped = crop_quadrant(
                frame, q, pupil_centers, pupil_mask_radius)
            if apply_tukey_window:
                if tukey_kernel is None:
                    tukey_kernel = _tukey_window_2d(cropped.shape, tukey_alpha)
                # # debug: view the kernel and the windowed image
                # plt.imshow(tukey_kernel, cmap='viridis')
                # plt.show()
                # plt.imshow(cropped*tukey_kernel, cmap='viridis')
                # plt.show()
                # exit()
                cropped = cropped * tukey_kernel
            quadrant_thumbnails.append(cropped)
            if q == "ul":
                assert cropped.shape == ul_this_frame.shape
                ul_this_frame += cropped
            elif q == "ur":
                assert cropped.shape == ur_this_frame.shape
                ur_this_frame += cropped
            elif q == "ll":
                assert cropped.shape == ll_this_frame.shape
                ll_this_frame += cropped
            elif q == "lr":
                assert cropped.shape == lr_this_frame.shape
                lr_this_frame += cropped
            else:
                raise ValueError(f"Invalid quadrant name {q}?")
        # total_irradiance: np.ndarray = np.sum(quadrant_thumbnails, axis=0)
        # total_irradiance_safe = total_irradiance.copy()
        # total_irradiance_safe[total_irradiance_safe == 0] = np.nan
        
        # # # normalize all pupil thumbnails by total irradiance
        
        # ul_this_frame /= total_irradiance_safe
        # ur_this_frame /= total_irradiance_safe
        # ll_this_frame /= total_irradiance_safe
        # lr_this_frame /= total_irradiance_safe

        # then append the norm frames to the main frames list
        ul_cropped_frames.append(ul_this_frame)
        ur_cropped_frames.append(ur_this_frame)
        ll_cropped_frames.append(ll_this_frame)
        lr_cropped_frames.append(lr_this_frame)
        
    reduced_pupil_cubes = {
        "ul": np.nan_to_num(np.asarray(ul_cropped_frames)),
        "ur": np.nan_to_num(np.asarray(ur_cropped_frames)),
        "ll": np.nan_to_num(np.asarray(ll_cropped_frames)),
        "lr": np.nan_to_num(np.asarray(lr_cropped_frames)),
    }
    return reduced_pupil_cubes

def process_file(
    filepath,
    reference,
    noise,
    quadrants,
    pupil_centers,
    pupil_mask_radius,
    start_frame,
    num_frames_skip,
    group_size,
    dark=None,
    subtract_reference=True,
    apply_tukey_window=False,
    tukey_alpha=0.5,
):
    """
    Process a single FITS cube file:
      - Reads the cube (shape: (512, 120, 120)).
      - Applies dark subtraction by subtracting the dark image (of shape (120,120))
        from each frame.
      - Optionally subtracts the reference image from each dark-subtracted frame.
      - Divides by the noise map (normalization).
      - Crops each frame to extract the specified pupil quadrant.

    Returns:
      The processed cube where each frame has been cropped to the desired quadrant.
    """
    # Read the data cube
    cube = read_fits_cube(filepath)

    return process_cube(
        cube,
        reference,
        noise,
        quadrants,
        pupil_centers,
        pupil_mask_radius,
        start_frame,
        num_frames_skip,
        group_size,
        dark,
        subtract_reference=subtract_reference,
        apply_tukey_window=apply_tukey_window,
        tukey_alpha=tukey_alpha,
    )

def load_dark_file(data_dir, top_level_dir):
    """
    Load dark file(s) from multiple possible locations.
    Priority order:
    1. top_level_dir/darks/ (shared location)
    2. data_dir/darks/ (subdirectory-specific)
    3. data_dir/../darks/ (parent directory)
    
    Returns:
        dark array or None if not found
    """
    # Priority 1: top-level directory
    darks_dir = os.path.join(top_level_dir, "darks")
    if os.path.exists(darks_dir):
        all_darks = glob.glob(f"{darks_dir}/*.fits")
        if all_darks:
            logging.info("Loading dark file(s) from top-level directory: %s", darks_dir)
            if len(all_darks) > 1:
                darks_data = []
                for d in all_darks:
                    with fits.open(d) as hdul:
                        dark_data = hdul[0].data
                    darks_data.append(dark_data)
                return np.mean(darks_data, axis=0)
            else:
                with fits.open(all_darks[0]) as hdul:
                    return hdul[0].data
    
    # Priority 2: subdirectory-specific
    darks_dir = os.path.join(data_dir, "darks")
    if os.path.exists(darks_dir):
        all_darks = glob.glob(f"{darks_dir}/*.fits")
        if all_darks:
            logging.info("Loading dark file(s) from subdirectory: %s", darks_dir)
            if len(all_darks) > 1:
                darks_data = []
                for d in all_darks:
                    with fits.open(d) as hdul:
                        dark_data = hdul[0].data
                    darks_data.append(dark_data)
                return np.mean(darks_data, axis=0)
            else:
                with fits.open(all_darks[0]) as hdul:
                    return hdul[0].data
    
    # Priority 3: parent directory
    parent_darks_dir = os.path.join(data_dir, "..", "darks")
    parent_darks_dir = os.path.normpath(parent_darks_dir)
    if os.path.exists(parent_darks_dir):
        all_darks = glob.glob(f"{parent_darks_dir}/*.fits")
        if all_darks:
            logging.info("Loading dark file(s) from parent directory: %s", parent_darks_dir)
            if len(all_darks) > 1:
                darks_data = []
                for d in all_darks:
                    with fits.open(d) as hdul:
                        dark_data = hdul[0].data
                    darks_data.append(dark_data)
                return np.mean(darks_data, axis=0)
            else:
                with fits.open(all_darks[0]) as hdul:
                    return hdul[0].data
    
    logging.info("No dark files found in any of the checked locations")
    logging.info("Skipping dark subtraction step...")
    return None

def load_reference_products(reference_file, noise_file):
    """Load a previously computed reference/noise pair."""
    reference = fits.getdata(reference_file)
    noise = fits.getdata(noise_file)
    return reference, noise

