from astropy.io import fits
import numpy as np
import os
import glob
import logging
import re
from datetime import datetime

_COMPACT_TIMESTAMP_RE = re.compile(r"(?<!\d)(\d{8}\d{6}\d{0,9})(?!\d)")
_REALTIME_TIMESTAMP_RE = re.compile(r"(?<!\d)(\d{8}T\d{6}\d{0,6})(?!\d)")

def read_fits_cube(filepath):
    """Read a FITS cube using astropy.io.fits."""
    with fits.open(filepath) as hdul:
        data = hdul[0].data
    return data

def write_fits_cube(filepath, data):
    """Write data to a FITS file."""
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(filepath, overwrite=True)

def convert_time_to_datetime(time):
    if "T" in time:
        date_part, rest = time.split("T", 1)
        time = f"{date_part}{rest}"
    # Python datetime supports up to microseconds (6 digits); trim if needed.
    if len(time) > 20:
        time = time[:20]
    try:
        return datetime.strptime(time, "%Y%m%d%H%M%S%f")
    except:
        raise ValueError(f"Time {time} does not follow expected format; cannot extract time from time.")

def extract_time_from_fname(fname):
    if fname.endswith(".fits"):
        fname = fname.split(".")[0]
    realtime_match = _REALTIME_TIMESTAMP_RE.search(fname)
    if realtime_match:
        return realtime_match.group(1)
    compact_matches = _COMPACT_TIMESTAMP_RE.findall(fname)
    if compact_matches:
        timestamp = compact_matches[-1]
        if len(timestamp) != 23:
            logging.warning(
                "Filename %s does not contain a legacy 23-digit timestamp; "
                "using extracted timestamp %s.",
                fname,
                timestamp,
            )
        return timestamp
    fname_array = fname.split("_")
    if fname_array[1].isdigit():
        #ex. timestamp 20230313071832943473000
        if len(fname_array[1]) != 23:
            logging.warning(f"Filename {fname} does not follow expected format; attempting to extract time from filename anyway.")
            for pc in fname_array:
                if pc.isdigit() and len(pc) == 23:
                    return pc
            raise ValueError(f"Filename {fname} does not follow expected format; cannot extract time from filename.")
        else:
            return fname_array[1]
    else:
        logging.warning(f"Filename {fname} does not follow expected format; attempting to extract time from filename anyway.")
        for pc in fname_array:
            if pc.isdigit() and len(pc) == 23:
                return pc
        raise ValueError(f"Filename {fname} does not follow expected format; cannot extract time from filename.")

def collapse_cube(input_cube):
    # Read the FITS cube
    with fits.open(input_cube) as hdul:
        data = hdul[0].data#[15:30]
    cube_length = data.shape
    data[data < 0] = 0
    # Sum along the time axis (assumed axis=0)
    stacked_image = np.sum(data, axis=0) / cube_length[0] #average
    # variance_image = np.std(data, axis=0)**2
    median_image = np.median(data, axis=0)
    return stacked_image, median_image


def save_fits(image, output_path, crop_radius):
    center = ((image.shape[0] // 2), (image.shape[1] // 2))
    full_crop_size = crop_radius * 2
    crop_start = center[0] - crop_radius
    crop_end = crop_start + full_crop_size
    cropped_image = image[crop_start: crop_end,
                          crop_start:crop_end]
    hdu = fits.PrimaryHDU(cropped_image)
    hdu.writeto(output_path, overwrite=True)
    # print(f"Saved stacked image as FITS to {output_path}")

def mask_outside_radius(data, radius, center=None, fill_value=0):
    """
    Sets all pixels outside a given radius to `fill_value`.

    Parameters
    ----------
    fits_in : str
        Path to input FITS file.
    fits_out : str
        Path to output FITS file.
    radius : float
        Radius (in pixels) around `center` inside which pixels are left unchanged.
    center : tuple of float, optional
        (x0, y0) coordinates of the center.  If None, uses image center.
    fill_value : scalar, optional
        Value to assign to pixels with distance > radius.
    """

    ny, nx = data.shape
    if center is None:
        x0, y0 = nx/2, ny/2
    else:
        x0, y0 = center

    # 2) Build coordinate grids and distance map
    y = np.arange(ny)
    x = np.arange(nx)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X - x0)**2 + (Y - y0)**2)

    # 3) Apply mask
    data[R > radius] = fill_value

    return data

def load_mf_response_cubes(response_cubes_loc: str) -> dict:
    """Load the matched-filter response cubes."""
    if not os.path.exists(response_cubes_loc):
        raise FileNotFoundError(
            f"The matched-filter response cubes directory {response_cubes_loc} does not exist. \
            Is the pipeline being run out of order? \
            Please run ws_distill first.")
    unsharped_mf_response_cube_paths = glob.glob(
        os.path.abspath(os.path.join(
            response_cubes_loc,
            "mf_response_cubes",
            "*_response_unsharp.fits")))
    cc_response_cube_paths = glob.glob(
        os.path.abspath(os.path.join(
            response_cubes_loc,
            "camwfs*00.fits")))
    assert len(unsharped_mf_response_cube_paths) == len(cc_response_cube_paths), "Number of unsharp and sharp response cubes must match"
    hp_cube_fnames = [os.path.basename(path) for path in unsharped_mf_response_cube_paths]
    cc_cube_fnames = [os.path.basename(path) for path in cc_response_cube_paths]
    out_dict = {
        "hp_cube_fnames": hp_cube_fnames,
        "og_cube_fnames": cc_cube_fnames,
        "unsharped_mf_response_cube_paths": unsharped_mf_response_cube_paths,
        "og_cc_cube_paths": cc_response_cube_paths
    }
    return out_dict


def load_collapsed_unsharp_response_maps(mf_response_cubes_loc: str) -> tuple[list, list]:
    """Load unsharp, mean-collapsed MF response FITS products from distill."""
    if not os.path.exists(mf_response_cubes_loc):
        raise FileNotFoundError(
            f"The matched-filter response cubes directory {mf_response_cubes_loc} does not exist. \
            Is the pipeline being run out of order? \
            Please run ws_distill first."
        )
    collapsed_paths = glob.glob(
        os.path.abspath(
            os.path.join(mf_response_cubes_loc, "*_mf_response_unsharp_mean_collapsed.fits")
        )
    )
    collapsed_names = [os.path.basename(path) for path in collapsed_paths]
    return collapsed_names, collapsed_paths

def load_and_average(file_list):
    """
    Load a list of FITS cubes and average them.
    
    Parameters:
    file_list (list): List of FITS file paths. They should all have the same dimensions.
    
    Returns:
    tuple: (averaged_cube, header)
    """
    cubes = []
    header = None
    for file_path in file_list:
        with fits.open(file_path) as hdul:
            data = hdul[0].data  # assume the cube is in the primary HDU
            cubes.append(data)
            # Use header from the first file (adjust if needed)
            if header is None:
                header = hdul[0].header

    # Convert the list of arrays to a single array and average across the new axis (i.e. from the set of 4 cubes)
    cubes_array = np.array(cubes)  # shape should be (4, 251, 120, 120)
    averaged_cube = np.mean(cubes_array, axis=0)  # resulting shape: (251, 120, 120)
    
    return averaged_cube, header

def write_cube(output_path, cube, header):
    """Write out a FITS cube with the provided header."""
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    fits.writeto(output_path, cube, header=header, overwrite=True)
    logging.info(f"Wrote file to: {output_path}")