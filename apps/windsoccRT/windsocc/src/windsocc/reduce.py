"""
ws_reduce.py - WindsoCC Data Reduction Pipeline

This script performs the initial data reduction steps:
  1. Dark subtraction using a provided dark FITS file (calibration data taken 
     with a closed shutter but nonzero exposure time). The dark file is assumed 
     to have shape (120, 120).
  2. Reference subtraction: either load an existing reference or create one
     from a mean stack of a specified number of raw FITS files.
  3. Cropping each frame in the data cube to extract a single pupil (one quadrant)
     of the pyramid wavefront sensor. The data cube is assumed to have shape 
     (512, 120, 120) where each of the 512 frames is processed.
  
The processed data cube is then saved to a designated output directory.

From camwfs experiment:

Sine pattern travelling E --> W (270 deg) on DM:
camwfs: 297.5 deg propagation direction
camsci1: 242.5 (SW) or 62.5 (NE) degree sparkle orientation

"""

import os
import glob
import argparse
import logging
import numpy as np
from astropy.io import fits
from skimage.measure import block_reduce
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
# Import helper functions from your modules (ensure these exist in src/)
from windsocc.preprocessing.reference_camwfs import create_reference
from windsocc.preprocessing.crop_pupil_camwfs import crop_quadrant

DEFAULT_START_FRAME = 0
DEFAULT_STEP_FRAME = 1
DEFAULT_GROUP_SIZE = 8
DEFAULT_CREATE_REFERENCE = True
DEFAULT_REMAKE_REFERENCE = False
DEFAULT_PUPIL_MASK_RADIUS = 28
QUADRANTS = ("ul", "ur", "ll", "lr")

# --- I/O Helper Functions ---
def read_fits_cube(filepath):
    """Read a FITS cube using astropy.io.fits."""
    with fits.open(filepath) as hdul:
        data = hdul[0].data
    return data

def write_fits_cube(filepath, data):
    """Write data to a FITS file."""
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(filepath, overwrite=True)

# --- Processing Functions ---
def parse_config_file(config_path):
    """Load the pipeline configuration file."""
    with open(config_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file) or {}


def resolve_config_path(path_arg, explicit_config=None):
    """
    Resolve the working directory and config file path.

    The command accepts either a directory containing `ws_config.yaml`, or a
    path to the YAML file itself.
    """
    config_candidate = explicit_config if explicit_config is not None else path_arg
    resolved_path = os.path.abspath(config_candidate)

    if os.path.isdir(resolved_path):
        return resolved_path, os.path.join(resolved_path, "ws_config.yaml")

    return os.path.dirname(resolved_path), resolved_path


def normalize_center(center_value, key_name):
    """Validate and normalize a `[x, y]` pupil center from YAML."""
    if not isinstance(center_value, (list, tuple)) or len(center_value) != 2:
        raise ValueError(f"{key_name} must be a list like [x, y]")

    return (int(center_value[0]), int(center_value[1]))


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

    mask_radius = config_params.get("PUPIL_MASK_RADIUS", DEFAULT_PUPIL_MASK_RADIUS)

    return pupil_centers, int(round(float(mask_radius)))


def resolve_bool_setting(cli_value, config_params, preferred_key, fallback_key=None, default=False):
    """Resolve a boolean setting from CLI first, then config, then default."""
    if cli_value is not None:
        return cli_value

    if preferred_key in config_params:
        return bool(config_params[preferred_key])

    if fallback_key is not None and fallback_key in config_params:
        return bool(config_params[fallback_key])

    return default


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
    }


def load_reference_products(reference_file, noise_file):
    """Load a previously computed reference/noise pair."""
    reference = fits.getdata(reference_file)
    noise = fits.getdata(noise_file)
    return reference, noise


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


def process_cube(cube, reference, noise, quadrant, pupil_centers, pupil_mask_radius,
                 start_frame, num_frames_skip, group_size, dark=None):
    """Process one in-memory raw cube and return the reduced cropped cube."""
    if dark is not None:
        cube_ds = cube - dark
    else:
        cube_ds = cube
    cube_rs = cube_ds - reference
    cube_norm = cube_rs / noise
    cube_reduced = block_reduce(cube_norm, block_size=(group_size, 1, 1), func=np.mean)
    cube_no_spark = cube_reduced[start_frame::num_frames_skip]

    cropped_frames = []
    for frame in cube_no_spark:
        cropped = crop_quadrant(frame, quadrant, pupil_centers, pupil_mask_radius)
        cropped_frames.append(cropped)

    return np.array(cropped_frames)


def process_file(filepath, reference, noise, quadrant, pupil_centers, pupil_mask_radius,
                 start_frame, num_frames_skip, group_size, dark=None):
    """
    Process a single FITS cube file:
      - Reads the cube (shape: (512, 120, 120)).
      - Applies dark subtraction by subtracting the dark image (of shape (120,120))
        from each frame.
      - Subtracts the reference image from each dark-subtracted frame.
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
        quadrant,
        pupil_centers,
        pupil_mask_radius,
        start_frame,
        num_frames_skip,
        group_size,
        dark,
    )

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
            if os.path.isdir(item_path) and item not in special_dirs:
                fits_files = [f for f in os.listdir(item_path) 
                            if f.endswith('.fits') and os.path.isfile(os.path.join(item_path, f))]
                if fits_files:
                    subdirs.append(item_path)
    
    return sorted(subdirs)

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

def process_dataset(data_dir, top_level_dir, nstack, pupil_centers, pupil_mask_radius,
                    start_frame=0, step_frame=1, group_size=8, create_reference_flag=True,
                    remake_ref=False, skip_dark=False):
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
    
    for pup in quadrants:
        saveprocessed_to = f"{processed_dir}/{pup}"
        os.makedirs(saveprocessed_to, exist_ok=True)
        
        for f in fits_files:
            input_path = os.path.join(data_dir, f)
            processed_cube = process_file(
                input_path,
                reference,
                noise,
                pup,
                pupil_centers,
                pupil_mask_radius,
                start_frame,
                step_frame,
                group_size,
                dark,
            )
            output_path = os.path.join(saveprocessed_to, f)
            write_fits_cube(output_path, processed_cube)
    
    logging.info("[%s] Completed processing", subdir_name)
    return True


def process_batch_in_memory(frames, data_dir, top_level_dir, nstack, pupil_centers, pupil_mask_radius,
                            frames_per_cube, start_frame=0, step_frame=1, group_size=8,
                            create_reference_flag=True, remake_ref=False, skip_dark=False):
    """Reduce a realtime batch in memory and return concatenated quadrant cubes."""
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

    reduced_quadrants = {}
    reduced_frames_per_cube = None
    for quadrant in QUADRANTS:
        reduced_cubes = [
            process_cube(
                cube,
                reference,
                noise,
                quadrant,
                pupil_centers,
                pupil_mask_radius,
                start_frame,
                step_frame,
                group_size,
                dark,
            )
            for cube in raw_cubes
        ]
        if reduced_frames_per_cube is None:
            reduced_frames_per_cube = reduced_cubes[0].shape[0]
        reduced_quadrants[quadrant] = np.concatenate(reduced_cubes, axis=0)

    return {
        "reduced_quadrants": reduced_quadrants,
        "raw_cube_count": raw_cube_count,
        "reduced_frames_per_cube": int(reduced_frames_per_cube),
        "dropped_frames": dropped_frames,
        "reference": reference,
        "noise": noise,
    }

def process_subdirectory(subdir_path, top_level_dir, nstack, pupil_centers, pupil_mask_radius,
                        start_frame, step_frame, group_size, create_reference_flag,
                        remake_ref, skip_dark):
    """
    Wrapper function to process a single subdirectory.
    This is designed to be called in parallel.
    
    Returns:
        Tuple of (subdir_path, success_boolean)
    """
    try:
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
        )
        return (subdir_path, success)
    except Exception as e:
        logging.error("[%s] Error processing subdirectory: %s", 
                     os.path.basename(subdir_path), str(e))
        return (subdir_path, False)

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
