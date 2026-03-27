#!/usr/bin/env python3
import os
import argparse
from astropy.io import fits
import numpy as np

def subtract_psf_from_directory(psf_path, img_dir, subdir_name="med_psf_subbed"):
    """
    Subtract the PSF image at psf_path from each FITS in img_dir.
    Outputs go into img_dir/subdir_name/ with the same filenames.
    """
    # Load PSF data (assumes primary HDU image)
    with fits.open(psf_path) as hdul_psf:
        psf_data = hdul_psf[0].data.astype(float)

    # Prepare output directory
    out_dir = os.path.join(img_dir, subdir_name)
    os.makedirs(out_dir, exist_ok=True)

    # Loop over FITS files in img_dir
    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(".fits"):
            continue
        src_path = os.path.join(img_dir, fname)
        # Skip the PSF file itself if it's in the same folder
        if os.path.abspath(src_path) == os.path.abspath(psf_path):
            continue

        # Read science image
        with fits.open(src_path) as hdul_in:
            sci_data = hdul_in[0].data.astype(float)
            sci_hdr  = hdul_in[0].header

        # Subtract
        sub_data = sci_data - psf_data

        # Write out
        out_hdu = fits.PrimaryHDU(data=sub_data, header=sci_hdr)
        out_path = os.path.join(out_dir, fname)
        out_hdu.writeto(out_path, overwrite=True)
        print(f"Written PSF-subtracted image: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Subtract a single PSF FITS from all FITS in a directory."
    )
    parser.add_argument(
        "psf",
        help="Path to the PSF FITS file."
    )
    parser.add_argument(
        "directory",
        help="Directory containing FITS images to process."
    )
    parser.add_argument(
        "--subdir", "-s",
        default="med_psf_subbed",
        help="Name of the output subdirectory (default: med_psf_subbed)."
    )
    args = parser.parse_args()

    subtract_psf_from_directory(args.psf, args.directory, args.subdir)

if __name__ == "__main__":
    main()
