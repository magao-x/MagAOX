"""Partition-stage helpers for organizing camwfs FITS cubes into time-based subdirectories."""
import os
import re
import shutil
from datetime import datetime, timedelta

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
                    dst = os.path.join(directory, f)
                    # Prefer os.replace over shutil.move: move fails if dst already exists
                    # (e.g. .DS_Store in both camwfs/ and camwfs_*/).
                    os.replace(f_path, dst)
            try:
                os.rmdir(item_path)
                print(f"Removed directory: {item_path}")
            except OSError as e:
                print(f"Could not remove {item_path}: {e}")

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

def check_existence_config_file(directory, config_creator=None):
    """
    Check if a config file exists in the given directory.

  When ``config_creator`` is provided and the file is missing, it is called to
  create ``ws_config.yaml`` before returning ``False``.
    """
    config_file = os.path.join(directory, 'ws_config.yaml')
    if os.path.exists(config_file):
        return True
    print(f"No config file found in {directory}. Creating one...")
    if config_creator is not None:
        config_creator(directory)
    return False

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

