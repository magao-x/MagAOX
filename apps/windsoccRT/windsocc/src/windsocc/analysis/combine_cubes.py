import os
from astropy.io import fits
import numpy as np

def average_fits_cubes(input_folder, output_filename):
    """
    Average all FITS image cubes in a given folder and save the resulting averaged cube.
    
    Parameters
    ----------
    input_folder : str
        Path to the folder containing the FITS cubes.
    output_filename : str
        Path (including filename) where the averaged FITS cube will be saved.
    
    Raises
    ------
    ValueError
        If no FITS files are found in the folder or if cube shapes are not consistent.
    """
    # List all files in the folder ending with '.fits'
    fits_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
                  if f.lower().endswith('.fits')]
    
    if not fits_files:
        raise ValueError("No FITS files found in the provided folder.")
    
    cubes = []
    
    # Use the first file to get the expected shape.
    with fits.open(fits_files[0]) as hdul:
        cube_shape = hdul[0].data.shape
    
    # Loop over each FITS file, check for shape consistency, and store the cube.
    for file in fits_files:
        with fits.open(file) as hdul:
            data = hdul[0].data
            if data.shape != cube_shape:
                raise ValueError(f"Cube shape mismatch: File {file} has shape {data.shape}, expected {cube_shape}.")
            cubes.append(data)
    
    # Convert list to a NumPy array. Its shape will be (n_cubes, ...cube_shape...)
    cubes_array = np.array(cubes)
    
    # Average along the first axis (over the different cubes)
    averaged_cube = np.mean(cubes_array, axis=0)
    
    # Save the averaged cube to a new FITS file.
    hdu = fits.PrimaryHDU(averaged_cube)
    hdu.writeto(output_filename, overwrite=True)
    print(f"Averaged cube saved to {output_filename}.")

# Example usage:
if __name__ == '__main__':
    input_folder = "/Users/jkueny/projects/windsocc/results_cubes/results_hpc_iz"
    output_file = "/Users/jkueny/projects/windsocc/results_cubes/averaged_cube.fits"
    average_fits_cubes(input_folder, output_file)
