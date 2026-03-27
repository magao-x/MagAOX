from astropy.io import fits
import numpy as np
import os
import glob

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

def load_mf_response_cubes(mf_response_cubes_loc: str) -> tuple[list, list]:
    """Load the matched-filter response cubes."""
    if not os.path.exists(mf_response_cubes_loc):
        raise FileNotFoundError(
            f"The matched-filter response cubes directory {mf_response_cubes_loc} does not exist. \
            Is the pipeline being run out of order? \
            Please run ws_distill first.")
    mf_response_cube_paths = glob.glob(os.path.abspath(os.path.join(mf_response_cubes_loc, "*_response_unsharp.fits")))
    cube_fnames = [os.path.basename(path) for path in mf_response_cube_paths]
    return cube_fnames, mf_response_cube_paths


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