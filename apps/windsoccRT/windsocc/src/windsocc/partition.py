# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

'''
This should be run on a directory of camwfs images. It is likely to contain
an extremely large number of FITS cubes. The HPC team won't like it if we
have a single folder with this many files, probably. So, we need to organize
the cubes into smaller subdirectories.

We should make subdirectories correspond to an actual span of wall time.
That way, we can analyze each subdir and measure what the wind was doing
in this range of time. This information can then be used to populate a
lookup table for this observation which can be queried down the line to
reconstruct the WDH for a given image.

This subdirectory timespan should be small, on the order of seconds
(instead of minutes, which is what this script was originally written to do).
The default timespan is 10 seconds.

MIT License

Copyright (c) 2026 Jay Kueny

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

#!/usr/bin/env python3
import os
import re
import shutil
from datetime import datetime, timedelta
import argparse
import yaml

def check_existence_config_file(directory):
    """
    Check if a config file exists in the given directory.
    """
    config_file = os.path.join(directory, 'ws_config.yaml')
    if os.path.exists(config_file):
        return True
    if not os.path.exists(config_file):
        print(f"No config file found in {directory}. Creating one...")
        generate_config_file(directory)
        return False

def check_directory_validity(directory, max_entries=1000):
    """
    Check if the directory contains camwfs FITS files without fully listing it.
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return False
    if not os.path.isdir(directory):
        print(f"Path {directory} is not a directory.")
        return False
    if max_entries is None or max_entries <= 0:
        max_entries = float("inf")

    seen_any = False
    found_fits = []
    with os.scandir(directory) as entries:
        for idx, entry in enumerate(entries):
            seen_any = True
            if idx >= max_entries:
                break
            name = entry.name.lower()
            if entry.is_file() and name.endswith(".fits") and name.startswith("camwfs_"):
                found_fits.append(entry.name)
        if len(found_fits) > 10 and seen_any:
            return True

    if not seen_any:
        print(f"Directory {directory} is empty.")
        return False

    if max_entries == float("inf"):
        print(f"Directory {directory} does not contain any camwfs FITS files!")
    else:
        print(
            f"Directory {directory} does not contain a significant number of camwfs FITS files "
            f"in the first {max_entries} entries. Is this a valid camwfs directory?"
        )
    return False

def generate_config_file(directory):
    """
    Generate a config file in the given directory.
    """
    config_file = os.path.join(directory, 'ws_config.yaml')
    config_content = """# Step 1: REDUCE
FILE_PREFIX: "camwfs_"
SUBTRACT_DARK: False
SKIP_DARK: false
NSTACK: null  # use all raw FITS files when building the reference
CREATE_REFERENCE: true
REMAKE_REFERENCE: false
WORKERS: null  # default to cpu_count()
DIAM_PUPILS: 60  # pixels
UL_CENTER: [30, 90]
UR_CENTER: [90, 90]
LL_CENTER: [30, 30]
LR_CENTER: [90, 30]

# Step 2: XCORR
START_FRAME: 0 # start the dataset on this frame; sparkle pattern zero point?
STEP_FRAME: 1 # skip this number of frames to form the dataset; to remove sparkle bkg.
GROUP_SIZE: 8 # number of frames to average to remove the sparkle pattern
MIN_DELAY: 0 # minimum delay to use for cross-correlation
MAX_DELAY: 128 # maximum delay to use for cross-correlation
DELAY_STEP: 1 # step size for delay values
SEGMENT_LENGTH: 11 # number of cubes to use per segment during cross-correlation
FFT_PAD_SHAPE: [128, 128]

# Step 3: DISTILL
DISTILL_DIR: "distill_results" # directory to save the distilled cubes to (default: "combined")

# Step 4: MEASURE
# Section for writing the wind lookup table
TIME_TRANSIT: "2023-03-10T06:09:11.342216344Z"  # parity change in wind PA
WRITE_CSV: results_iz.csv
INNER_RADIUS: 5
OUTER_RADIUS: 28
N_WEDGES: 72
D_MIRROR: 6.5
DEBUG: false
PA_OFFSET: 28  # degrees
"""
    with open(config_file, 'w') as f:
        f.write(config_content)
    return config_file

def split_into_groups(data, n):
    """
    Split data (a list) into n groups as evenly as possible.
    """
    groups = []
    total = len(data)
    quotient, remainder = divmod(total, n)
    start = 0
    for i in range(n):
        group_size = quotient + (1 if i < remainder else 0)
        groups.append(data[start:start+group_size])
        start += group_size
    return groups


def organize_fits_files(directory, interval_seconds, number_groups=None):
    # List all FITS files in the given directory
    files = [f for f in os.listdir(directory)
             if (f.lower().endswith('.fits') and \
             os.path.isfile(os.path.join(directory, f)) and \
             f.lower().startswith('camwfs'))]
    
    # Regular expression to capture the timestamp digits after the underscore.
    pattern = re.compile(r'^[^_]+_(\d{14,})')
    
    file_data = []
    for f in files:
        match = pattern.match(f)
        if match:
            timestamp_str = match.group(1)
            # Extract main timestamp and fractional seconds
            if len(timestamp_str) >= 16:
                main_str = timestamp_str[:14]
                frac_str = timestamp_str[14:16]
            else:
                main_str = timestamp_str
                frac_str = "00"
            # Build a datetime string: add "0000" to convert fractional seconds into microseconds
            dt_str = main_str + frac_str + "0000"
            try:
                dt = datetime.strptime(dt_str, '%Y%m%d%H%M%S%f')
                file_data.append((f, dt))
            except ValueError:
                print(f"Skipping file {f}: Invalid timestamp format after processing.")
        else:
            print(f"Skipping file {f}: Does not match expected naming convention.")
    
    # Sort files by their timestamp in ascending order
    file_data.sort(key=lambda x: x[1]) #list of tuples (filename, timestamp)
    
    # Group files either by fixed number or by time interval.
    if number_groups is not None:
        if number_groups > len(file_data):
            print("Warning: Number of groups requested is greater than the number of files."
                  " Each group will have at most one file.")
        groups = split_into_groups(file_data, number_groups)
    else:
        groups = []
        current_group = []
        group_start_time = None
        for filename, dt in file_data:
            if not current_group:
                current_group.append((filename, dt))
                group_start_time = dt
            else:
                if (dt - group_start_time) < timedelta(seconds=interval_seconds):
                    current_group.append((filename, dt))
                else:
                    groups.append(current_group)
                    current_group = [(filename, dt)]
                    group_start_time = dt
        if current_group:
            groups.append(current_group)
    
    # Create subdirectories for each group, naming each one after the middle file
    for group in groups:
        if not group:
            continue
        mid_index = len(group) // 2
        mid_file_name = group[mid_index][0]
        subdir_name, _ = os.path.splitext(mid_file_name)
        subdir = os.path.join(directory, subdir_name)
        os.makedirs(subdir, exist_ok=True)
        for f, dt in group:
            src = os.path.join(directory, f)
            dst = os.path.join(subdir, f)
            shutil.move(src, dst)


def undo_organize(directory):
    """
    Undo the organization by:
      - Processing only subdirectories whose names match the pattern "camwfs_*".
      - Within each such subdirectory, recursively deleting any folders named
        "darks", "reduced", or "references".
        - archival darks should be in another folder in the data directory.
          - e.g., in /HR4796_rg_smlyot_20230312_13/camwfs/darks/
      - Moving all remaining files from the subdirectory up to the main directory.
      - Deleting the (now empty) subdirectory.
    """
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and item.startswith("camwfs_"):
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)
                if os.path.isdir(sub_item_path) and sub_item in ("darks", "reduced", "references"):
                    shutil.rmtree(sub_item_path)
                    print(f"Removed special subdirectory: {sub_item_path}")
            for f in os.listdir(item_path):
                f_path = os.path.join(item_path, f)
                if os.path.isfile(f_path):
                    shutil.move(f_path, directory)
            try:
                os.rmdir(item_path)
                print(f"Removed directory: {item_path}")
            except OSError as e:
                print(f"Could not remove {item_path}: {e}")


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

    if not check_directory_validity(args.directory):
        print(f"Directory {args.directory} is not valid. Please check the directory and try again.")
        return
    if args.undo:
        undo_organize(args.directory)
    else:
        if not os.path.exists(os.path.join(args.directory, 'ws_config.yaml')):
            print(f"No config file found in {args.directory}. Creating one...")
            generate_config_file(args.directory)
        else:
            print(f"Using config file: {os.path.join(args.directory, 'ws_config.yaml')}")
        organize_fits_files(args.directory, args.interval, args.number)

if __name__ == '__main__':
    main()
