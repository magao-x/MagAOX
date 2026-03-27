#!/usr/bin/env python3
"""
reference_camwfs.py - Create a master reference image for CAMWFS

This script computes a master reference image by averaging FITS cubes in batches.
Each FITS cube is assumed to have shape (512, 120, 120). For each cube, a mean frame
is computed by averaging over the 512 frames (axis=0), resulting in a 120×120 image.
Because the number of files to be averaged can be very large, the averaging is done
in batches (default batch size is 20 cubes at a time). The final master reference
image is saved to a specified output path. If the reference image already exists and a
remake isn't requested, the script simply loads and returns the existing reference.

Usage:
    python reference_camwfs.py --data-dir /path/to/data \
                                 --nstack 100 \
                                 --output-ref /path/to/output/reference.fits
"""

import os
import argparse
import logging
import numpy as np
from astropy.io import fits

def create_reference(data_dir, nstack, output_ref, output_noise, dark_img=None, batch_size=20, remake_ref=False):
    """
    Create a master reference image by averaging FITS cubes from a directory in batches.

    Parameters
    ----------
    data_dir : str
        Directory containing the FITS cubes to be stacked.
    nstack : int
        Number of FITS cubes to use for the mean stack.
    output_ref : str
        File path where the master reference image (FITS file) will be saved.
    batch_size : int, optional
        Number of FITS cubes to average at a time (default is 20).
    remake_ref : bool, optional
        If True, force recalculation of the reference image even if it exists.

    Returns
    -------
    ref : ndarray
        The computed master reference image (shape (120, 120)).
    """
    # If the reference image already exists and we are not forcing a remake, load and return it.
    if not remake_ref and os.path.exists(output_ref):
        logging.info("Reference image already exists at %s. Loading...", output_ref)
        ref = fits.getdata(output_ref)
        noise = fits.getdata(output_noise)
        return ref, noise

    # Get a sorted list of FITS files in data_dir.
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.fits')])
    if len(all_files) == 0:
        logging.error("No FITS files found in %s", data_dir)
        return None

    # Use only the first nstack files (or all available files if there are fewer than nstack).
    files_to_use = all_files[:min(nstack, len(all_files))]
    logging.info("Using %d files for reference creation.", len(files_to_use))

    batch_means = []  # To store the mean image from each batch
    batch_noises = []

    # Process files in batches.
    for i in range(0, len(files_to_use), batch_size):
        batch_files = files_to_use[i:i+batch_size]
        logging.info("Processing batch %d to %d", i+1, i+len(batch_files))
        batch_stack = []  # To store the mean frame for each cube in this batch
        batch_std = []

        for file in batch_files:
            with fits.open(file) as hdul:
                cube = hdul[0].data  # cube should have shape (512, 120, 120)
            # Compute the mean over the 512 frames (axis=0) to get a 120×120 image.
            if dark_img is not None:
                cube_ds = cube - dark_img # subtract the dark
            else:
                cube_ds = cube
            mean_frame = np.mean(cube_ds, axis=0)
            std_frame = np.std(cube_ds, axis=0)
            batch_stack.append(mean_frame)
            batch_std.append(std_frame)

        batch_stack = np.array(batch_stack)
        batch_std = np.array(batch_std)
        # Compute the mean of the batch (averaging the 120×120 images).
        batch_mean = np.mean(batch_stack, axis=0)
        batch_noise = np.mean(batch_std, axis=0)
        batch_means.append(batch_mean)
        batch_noises.append(batch_noise)

    # Average the batch means to obtain the final master reference image.
    batch_means = np.array(batch_means)
    batch_noises = np.array(batch_noises)
    ref = np.mean(batch_means, axis=0)
    noise_ref = np.mean(batch_noises, axis=0) + 1 # just in case of zeros

    # Save the reference image to the specified output file.
    fits.writeto(output_ref, ref, overwrite=True)
    
    fits.writeto(output_noise, noise_ref, overwrite=True)
    logging.info("Saved references to %s", output_ref)

    return ref, noise_ref

def main():
    parser = argparse.ArgumentParser(description="Create a master reference image for CAMWFS.")
    parser.add_argument('--data-dir', type=str, required=True,
                        help="Directory containing the FITS cubes to be stacked.")
    parser.add_argument('--nstack', type=int, required=True,
                        help="Number of FITS cubes to use for the mean stack.")
    parser.add_argument('--output-ref', type=str, required=True,
                        help="File path to save the master reference image (FITS file).")
    parser.add_argument('--batch-size', type=int, default=20,
                        help="Number of FITS cubes to average at a time (default: 20).")
    parser.add_argument('--remake-ref', action='store_true',
                        help="Force recalculation of the reference image even if it exists.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ref = create_reference(args.data_dir, args.nstack, args.output_ref, args.batch_size, args.remake_ref)
    if ref is not None:
        logging.info("Reference image creation completed successfully.")
    else:
        logging.error("Reference image creation failed.")

if __name__ == '__main__':
    main()
