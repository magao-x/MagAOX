#!/usr/bin/env python3
import os
import glob
import argparse
from astropy.io import fits

def process_file(file_path, chunk_size, output_dir):
    """Process a single FITS file by splitting its cube along the first axis."""
    try:
        with fits.open(file_path) as hdul:
            data = hdul[0].data  # Assumes cube is stored in the primary HDU
            header = hdul[0].header
    except Exception as e:
        print(f"Error opening {file_path}: {e}")
        return

    # Verify that the first dimension can be evenly divided by the chunk size
    n_slices = data.shape[0]
    if n_slices % chunk_size != 0:
        print(f"Skipping {file_path}: number of slices ({n_slices}) is not a multiple of the chunk size ({chunk_size}).")
        return

    n_chunks = n_slices // chunk_size
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk_data = data[start:end, :, :]

        # Update the header for the new cube size if needed
        header['NAXIS3'] = chunk_size

        # Create a new Primary HDU for the chunk and write to a file
        hdu = fits.PrimaryHDU(data=chunk_data, header=header)
        output_file = os.path.join(output_dir, f"{base_name}_chunk_{i:02d}.fits")
        try:
            hdu.writeto(output_file, overwrite=True)
            print(f"Wrote {output_file}")
        except Exception as e:
            print(f"Error writing {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Split all FITS cubes in a directory into smaller cubes.")
    parser.add_argument("directory", help="Directory containing the FITS files to process.")
    parser.add_argument("--chunk", type=int, default=512,
                        help="Chunk size along the first axis (default: 512).")
    parser.add_argument("--output", default=None,
                        help="Output directory for the split FITS files. Defaults to the input directory.")

    args = parser.parse_args()

    input_dir = args.directory
    chunk_size = args.chunk
    output_dir = args.output if args.output else input_dir

    if not os.path.isdir(input_dir):
        print(f"The directory {input_dir} does not exist or is not accessible.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process all FITS files in the directory
    fits_files = glob.glob(os.path.join(input_dir, "*.fits"))
    if not fits_files:
        print("No FITS files found in the specified directory.")
        return

    for file_path in fits_files:
        print(f"Processing {file_path}...")
        process_file(file_path, chunk_size, output_dir)

if __name__ == "__main__":
    main()
