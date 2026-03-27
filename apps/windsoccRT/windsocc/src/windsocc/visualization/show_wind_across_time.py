import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.animation import FFMpegWriter

def fits_to_movie_matplotlib(
    directory,
    output_path='movie.mp4',
    fps=10,
    vmin=None,
    vmax=None,
    cmap='viridis',
    sort_key=None
):
    """
    Converts FITS files in a directory into a movie using matplotlib (no imageio).

    Parameters:
    - directory (str): Path to folder with .fits files.
    - output_path (str): Output video path.
    - fps (int): Frames per second.
    - vmin, vmax (float or None): Color scale range. If None, computed globally.
    - cmap (str): Colormap name.
    - sort_key (callable): Optional function to sort filenames.
    """
    fits_files = sorted(glob.glob(os.path.join(directory, '*.fits')), key=sort_key)
    if not fits_files:
        raise FileNotFoundError("No FITS files found in directory.")

    # Preload data and compute global color limits
    frames = []
    names = []
    if vmin is None or vmax is None:
        all_data_values = []
    for file in fits_files:
        fname = os.path.basename(file)
        data = fits.getdata(file)
        frames.append(data)
        names.append(fname)
        if vmin is None or vmax is None:
            all_data_values.append(data)

    if vmin is None or vmax is None:
        all_values = np.concatenate([d.ravel() for d in all_data_values])
        if vmin is None:
            vmin = np.nanpercentile(all_values, 1)
        if vmax is None:
            vmax = np.nanpercentile(all_values, 99)

    # Set up the figure and writer
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation')
    title = ax.set_title("")

    save_to = os.path.join(directory, output_path)

    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, save_to, dpi=300):
        for i, frame in enumerate(frames):
            im.set_data(frame)
            # title.set_text(f"Stack {i + 1}")
            title.set_text(names[i])
            writer.grab_frame()

    plt.close(fig)
    print(f"Saved movie to: {save_to}")

fits_to_movie_matplotlib(
    directory='/Users/jkueny/projects/HR4796a_lco2023a_magao-x_20230309_10/camwfs/diagnostics/stacks',
    # directory='/Users/jkueny/projects/HR4796_rg_smlyot_20230312_13/camwfs/diagnostics/stacks',
    output_path='wind_direction_through_time.mp4',
    fps=30,
    vmin=None,   # Or use 0, 1000 for manual range
    vmax=None,
    cmap='bone'
)