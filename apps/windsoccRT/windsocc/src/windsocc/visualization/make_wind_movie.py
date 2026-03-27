#!/usr/bin/env python3
"""
visualize_wind.py - Visualize Wind Direction via Cross-Correlation Maps

This script:
  - Loads a continuous time series of reduced FITS cubes (each of shape (512, 120, 120))
    from a specified directory.
  - For a range of delays, computes the master cross-correlation maps using Welch’s method.
    For each pair, the cross-correlation is computed between the circular subaperture (patch)
    extracted from the frame at time t and the full image at time t+delay, with the patch's
    self-correlation (with its own full frame) subtracted out.
  - Negative values in each map are zeroed and the map is normalized.
  - An animated movie is created to visualize how the master cross-correlation map changes with delay.
  - A FITS cube containing all the cross-correlation maps (one per delay) is saved.

Usage:
    python visualize_wind.py --reduced-dir /path/to/reduced_cubes \
                             --center 60,60 \
                             --radius 20 \
                             --min-delay 80 \
                             --max-delay 120 \
                             --delay-step 5 \
                             --output-movie output.mp4 \
                             --output-cube cc_maps_cube.fits \
                             [--segment-cubes 22] [--overlap 0.5] [--fps 2]
"""

import os
import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from astropy.io import fits
from datetime import datetime

# Add the src/ directory to sys.path so we can import our analysis functions.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_path)

from analysis.cross_correlation import load_reduced_series, compute_master_cc_map_welch_shared_fft
from preprocessing.radial import radial_profile

def parse_center(center_str):
    """Parse a comma-separated string (e.g., '60,60') into a tuple of two integers."""
    try:
        parts = center_str.split(',')
        if len(parts) != 2:
            raise ValueError
        return (int(parts[0]), int(parts[1]))
    except Exception:
        raise argparse.ArgumentTypeError("Center must be in the format 'x,y' with two integers.")

import numpy as np

import numpy as np

def mask_cc_cube_center(cube, radius, mask_value=np.nan):
    """
    Masks the central region of each frame in a 3D image cube using a circular mask.

    Parameters
    ----------
    cube : np.ndarray
        3D array with shape (n_frames, height, width) representing the image cube.
    radius : float or int
        Radius in pixels for the circular mask, centered on the spatial center of each frame.
    mask_value : float, optional
        Value to assign within the masked region (default: np.nan). Using np.nan
        is useful because many visualization routines ignore NaNs when scaling colors.
    
    Returns
    -------
    masked_cube : np.ndarray
        A new 3D array where the central circular region in each frame has been replaced with mask_value.
    """
    # Copy the cube to avoid modifying the original data.
    masked_cube = cube.copy()
    n_frames, rows, cols = cube.shape
    center_row, center_col = rows // 2, cols // 2

    # Create a 2D boolean mask for the circular region.
    y, x = np.ogrid[:rows, :cols]
    mask = (x - center_col)**2 + (y - center_row)**2 <= radius**2

    # Apply the mask to each frame using broadcasting.
    # This sets the pixel values in the masked region to mask_value for all frames.
    masked_cube[:, mask] = mask_value

    return masked_cube

def save_cube_as_movie(image_cube, delay_array, save_to, fps):
    # Set up the figure for animation.
    vmin = np.min(image_cube[-1]) * 0.75
    vmax = np.max(image_cube[-1]) * 0.75
    fig, ax = plt.subplots()
    im = ax.imshow(image_cube[0], cmap='inferno', origin="lower",
                   vmin=vmin, vmax=vmax)
    ax.set_title(f"Delay: {delay_array[0]} frames ({delay_array[0] * 1/2000} s)")
    plt.colorbar(im, ax=ax)

    def update(frame):
        im.set_data(image_cube[frame])
        ax.set_title(f"Delay: {delay_array[frame]} frames ({delay_array[frame] * 1/2000} s)")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(delay_array),
                                  blit=True, interval=1000/fps)

    logging.info("Saving animation to %s.mp4", save_to)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='windsock'), bitrate=1800)
    ani.save(f"results_movies/{save_to}.mp4", writer=writer)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize Wind Direction via Cross-Correlation Maps")
    parser.add_argument('--reduced-dir', type=str, required=True,
                        help="Directory containing reduced FITS cubes (each of shape (512,120,120)).")
    parser.add_argument('--min-delay', type=int, required=True,
                        help="Minimum delay (in frames) for cross-correlation.")
    parser.add_argument('--max-delay', type=int, required=True,
                        help="Maximum delay (in frames) for cross-correlation.")
    parser.add_argument('--delay-step', type=int, default=1,
                        help="Step size for delay values (in frames).")
    parser.add_argument('--output-name', type=str, required=True,
                        help="Name to the output movie and fits cube files (e.g., test_run).")
    parser.add_argument('--fps', type=int, default=2,
                        help="Frames per second for the output movie.")
    parser.add_argument('--segment-cubes', type=int, default=22,
                        help="Number of cubes to use per segment (default: 22).")
    parser.add_argument('--overlap', type=float, default=0.5,
                        help="Fractional overlap between segments (default: 0.5).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logging.info("Loading reduced series from %s", args.reduced_dir)
    time_series, n_per_cube, skipped_cubes = load_reduced_series(args.reduced_dir)
    logging.info("Loaded reduced series of size %s", len(time_series))
    logging.info("Skipped %s cubes due to not having insufficient frames.", skipped_cubes)
    
    os.makedirs("results_movies", exist_ok=True)
    os.makedirs("results_cubes", exist_ok=True)
    delays = list(range(args.min_delay, args.max_delay + 1, args.delay_step))
    cc_maps = []
    starttime = datetime.now()
    static_pattern = np.median(time_series, axis=0)
    time_series -= static_pattern
    for d in delays:
        logging.info("Computing master cross-correlation map for delay %d", d)
        cc_map = compute_master_cc_map_welch_shared_fft(time_series, n_per_cube, d,
                                             segment_cube_count=args.segment_cubes,
                                             overlap_fraction=args.overlap)
        cc_maps.append(cc_map)
    
    # Save the cross-correlation maps as a FITS cube.
    cc_cube = np.array(cc_maps)
    static_pattern = np.median(cc_cube, axis=0)
    cc_cube -= static_pattern

    for i in range(len(delays)):
        rprofile = radial_profile(cc_cube[i])
        cc_cube -= rprofile

    cc_rprofsub_masked = mask_cc_cube_center(cc_cube, radius=3, mask_value=0.)

    # Assign header information to each frame in the cube
    primary_hdu = fits.PrimaryHDU(cc_rprofsub_masked)
    # Convert the delays list into a comma-separated string
    #TODO support different loop speeds
    delays_str = ",".join(str((512 / n_per_cube) * 1/2000 * d) for d in delays) #in seconds
    
    # Add the delays string to the header with a descriptive comment
    primary_hdu.header['DELAYARR'] = (delays_str, 'Comma-separated delays for each frame')
    
    primary_hdu.writeto(f"results_cubes/{args.output_name}.fits", overwrite=True)
    logging.info("Saved cross-correlation maps cube to %s.fits", args.output_name)
    
    
    logging.info("This run took %s", (datetime.now() - starttime))

if __name__ == '__main__':
    main()
