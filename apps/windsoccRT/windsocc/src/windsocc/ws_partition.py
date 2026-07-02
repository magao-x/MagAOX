# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""
Organize camwfs FITS cubes into time-based subdirectories (default 10 s spans).

Run from a directory of camwfs images. Subdirectory timespans support downstream
wind lookup and WDH reconstruction per observation interval.
"""

#!/usr/bin/env python3
import os
import argparse

from windsocc.core.partition import (
    check_directory_validity,
    organize_fits_files,
    undo_organize,
)

def generate_config_file(directory):
    """
    Generate a config file in the given directory.
    """
    config_file = os.path.join(directory, 'ws_config.yaml')
    config_content = """# When running the pipeline, overwrite existing results? (default: false)
CLOBBER: true

# Step 1: REDUCE
FILE_PREFIX: "camwfs_"
REDUCE_DIR: "reduce_results" # directory for optional saved reduced quadrant FITS cubes (default: "reduce_results")
INSPECT_REDUCTION: false # when true, save reference/noise and first reduced pupil products (for debugging)
SUBTRACT_DARK: False
SKIP_DARK: false
NSTACK: null  # use all raw FITS files when building the reference
CREATE_REFERENCE: true
REMAKE_REFERENCE: false
SUBTRACT_REFERENCE: true # when false, skip subtracting the mean reference; still divide by noise (per-frame std map)
GROUP_SIZE: 8 # number of frames to average to remove the sparkle pattern
LOOP_SPEED: 2000 # Hz; WFS frame rate. With GROUP_SIZE, measure uses TIME_PER_FRAME = GROUP_SIZE / LOOP_SPEED.
WORKERS: null  # `null` to default to cpu_count()
DIAM_PUPILS: 58  # pixels
UL_CENTER: [30, 90]
UR_CENTER: [90, 90]
LL_CENTER: [30, 30]
LR_CENTER: [90, 30]
APPLY_TUKEY_WINDOW: true  # apply a Tukey window to cropped pupil thumbnails before cross-correlation
TUKEY_ALPHA: 0.0          # Tukey window shape parameter in [0, 1]

# Step 2: XCORR
START_FRAME: 0 # start the dataset on this frame; sparkle pattern zero point?
STEP_FRAME: 1 # skip this number of frames to form the dataset; to remove sparkle bkg.
MIN_DELAY: 0 # minimum delay to use for cross-correlation
MAX_DELAY: 512 # maximum delay to use for cross-correlation
DELAY_STEP: 1 # step size for delay values
SEGMENT_LENGTH: 11 # number of cubes to use per segment during cross-correlation
RAW_FRAMES_PER_CUBE: 512 # raw frames per WFS cube; used for DELAYARR time conversion in xcorr
FFT_PAD_SHAPE: [128, 128]
XCORR_DIR: "xcorr_results" # directory to save the cross-correlation results to (default: "xcorr_results")

# Step 3: DISTILL
DISTILL_DIR: "distill_results" # directory to save the distilled cubes to (default: "combined")
# Define template for the matched filter
TEMPLATE_SIZE: 65 #size of central square of the first frame of the CC cube
HIGH_PASS_FWHM: 5.0 # FWHM of the high-pass filter in pixels

# Step 4: MEASURE
MEASURE_DIR: "measure_results" # directory to save the measure results to (default: "measure_results")
# Section for writing the wind lookup table
TIME_TRANSIT: "2023-03-10T06:09:11.342216344Z"  # parity change in wind PA
WRITE_CSV: results_iz.csv
LIMIT_CUBES: null # limit the number of cubes to process (for debugging; default is null)
PARALLELIZED: true # parallelize the measure process (default is false)
INNER_RADIUS: 12 #pixels; tripwire region inner radius
OUTER_RADIUS: 24 #pixels; tripwire region outer radius
IMAGE_CENTER: [65, 65]
N_WEDGES: 72
D_MIRROR: 6.5
DEBUG: false
PA_OFFSET: 28  # degrees
SEP_THRESH: 3.0 #s/n threshold for sep to detect sources
SEP_MINAREA: 3.0 #minimum area in pixels for sep to detect sources
# TIME_PER_FRAME is computed as GROUP_SIZE / LOOP_SPEED when LOOP_SPEED is set (see Step 1).

# Step 5: VISUALS
VISUALS_DIR: "visuals_results" # directory to save the visuals results to (default: "visuals_results")
MAKE_MOVIE: true
MOVIE_FPS: 30


"""
    with open(config_file, 'w') as f:
        f.write(config_content)
    return config_file


def main():
    parser = argparse.ArgumentParser(
        description="Organize FITS files into subdirectories based on time intervals or undo the organization."
    )
    parser.add_argument(
        "-d","--directory",
        type=str,
        default=".",
        help="Path to the directory containing the camwfs files or subdirs to be organized."
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=10,
        help="Time interval in seconds to group the files (default: 10)"
    )
    parser.add_argument(
        "-n", "--number",
        type=int,
        default=None,
        help="Number of subdirs to create; overrides time interval."
    )
    parser.add_argument(
        "--undo",
        action="store_true",
        help="Undo the organization by moving files from subdirectories up one level and removing them."
    )
    args = parser.parse_args()
    if args.directory == ".":
        args.directory = os.getcwd()
    else:
        args.directory = os.path.abspath(args.directory)

    if args.undo:
        undo_organize(args.directory)

    if not check_directory_validity(args.directory):
        print(f"Directory {args.directory} is not valid. Please check the directory and try again.")
        return
    else:
        if not os.path.exists(os.path.join(args.directory, 'ws_config.yaml')):
            print(f"No config file found in {args.directory}. Creating one...")
            generate_config_file(args.directory)
        else:
            print(f"Using config file: {os.path.join(args.directory, 'ws_config.yaml')}")
        organize_fits_files(args.directory, args.interval, args.number)


if __name__ == '__main__':
    main()
