import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os

def prep_polar_coords(image_shape, image_center = None, x_roi_off = 0, y_roi_off = 0):
    """Prepare polar coordinates for wedging operations.

    Args:
        image_shape (tuple): dimensions of given image using numpy's .shape
        image_center (tuple, optional): Center of image pixel coords.
        Defaults to None.
        x_roi_off (int, optional): Do you want the region of interest offset
        from the center in the x-direction? Defaults to no.
        y_roi_off (int, optional): Do you want the region of interest offset
        from the center in the y-direction? Defaults to no.
    """
    y, x = np.indices(image_shape)
    
    if image_center is None:
        cx, cy = (image_shape[1] / 2 - 0.5), (image_shape[0] / 2 - 0.5)
    else:
        cx, cy = image_center
    # Calculate polar coordinates relative to the center
    # x increases to the right (west),  y increases down (south)
    # we want dx positive ⇒ west (+), dx negative ⇒ east (–)
    #      dy positive ⇒ north,   dy negative ⇒ south
    dx =  x - cx
    dy =  cy - y
    r  = np.hypot(dx, dy)
    theta = np.arctan2(dx, dy)
    # now theta ∈ [–π, +π], 0 at north, +CW, –CCW

    return r, theta

def wedge_photometry(image, inner_radius, outer_radius, num_wedges,
                     xroioff=0, yroioff=0,
                     debug=False):
    """
    Sums pixel counts in annular wedges centered on the center of the image.

    Parameters:
    - image: 2D numpy array representing the image.
    - inner_radius: Inner radius of the annular region.
    - outer_radius: Outer radius of the annular region.
    - num_wedges: Number of wedges to divide the annular region into.

    Returns:
    - counts: Array of summed pixel counts for each wedge.
    """

    # Calculate polar coordinates relative to the center
    r, theta = prep_polar_coords(image.shape)

    # Create mask for annular region
    annular_mask = (r >= inner_radius) & (r < outer_radius)
    region_mask = np.zeros_like(image, dtype=bool)

    # Bin pixels into wedges
    # divide the circle into num_wedges bins from –π to +π
    wedge_edges   = np.linspace(-np.pi, np.pi, num_wedges+1)
    # wedge_edges   = np.linspace(0, 2*np.pi, num_wedges+1)
    wedge_centers = (wedge_edges[:-1] + wedge_edges[1:]) / 2.0


    counts = np.zeros(num_wedges)
    std_devs = np.zeros(num_wedges)
    for i in range(num_wedges):
        a_min, a_max = wedge_edges[i], wedge_edges[i+1]
        mask = (theta >= a_min) & (theta < a_max)

        combined = annular_mask & mask
        counts[i] = image[combined].sum()
        if debug:
            fig, ax = plt.subplots()
            ax.imshow(image,
                      origin='lower',
                      )
            # overlay the wedge region
            # use mask as alpha
            ax.imshow(mask,
                      origin='lower',
                      alpha=0.3,
                      )
            ax.set_title(
                f"Wedge {i+1}/{num_wedges}\n"
                f"Angle [{np.degrees(a_min):.1f}°, {np.degrees(a_max):.1f}°]"
            )
            plt.show()

    return counts, std_devs, wedge_centers, region_mask

def wedge_find_peak(image, angle, wedge_width, r_start, r_end, ap_width, sn_threshold,
                    r, theta, center=None):
    """
    Isolate an annular wedge region from a 2D image and determine the radial bin 
    (within a given annulus) that has the highest summed signal meeting a specified
    signal-to-noise (S/N) threshold.
    
    The coordinate system is defined such that:
      - 0° is along the positive x-axis,
      - angles increase counter-clockwise.
    
    
    Parameters
    ----------
    image : np.ndarray
        2D array (e.g., a cross-correlation map).
    angle : float
        Central angle (in degrees) of the wedge (0° is right, increasing CCW).
    wedge_width : float
        Total angular width (in degrees) of the wedge.
    r_start : float
        Starting radius (in pixels) of the annular region.
    r_end : float
        Ending radius (in pixels) of the annular region.
    bin_width : float
        Width (in pixels) of the radial bins.
    sn_threshold : float
        Minimum S/N ratio required for a bin to be considered.
    center : tuple of float, optional
        (x, y) coordinates of the image center. If None, defaults to the center of the image.
    
    Returns
    -------
    best_r : float
        The radial distance (bin center) corresponding to the bin with the highest summed 
        signal that meets the S/N threshold. If no bin meets the threshold, returns np.nan.
    """


    # print(f"print(theta) -> {theta}")
    # Define the wedge based on the provided central angle and wedge width.
    half_wedge = wedge_width / 2.0
    ang_min = (angle - half_wedge) % 360
    ang_max = (angle + half_wedge) % 360
    if ang_min < ang_max:
        wedge_mask = (theta >= ang_min) & (theta <= ang_max)
    else:
        wedge_mask = (theta >= ang_min) | (theta <= ang_max)
    
    # Define the annulus mask.
    annulus_mask = (r >= r_start) & (r < r_end)
    # The ROI mask is the intersection of the annulus and the wedge.
    roi_mask = annulus_mask & wedge_mask

    # Prepare arrays to store the scanned S/N values and corresponding aperture center positions.
    scan_positions = []
    sn_values = []
    sums = []
    rad_dists = range(int(r_start), int(r_end - ap_width) + 1)
    
    # For each radial aperture starting position (integer pixels) from r_start to r_end - aperture_width.
    for r_ap in rad_dists:
        # Define current aperture: pixels with radial distance in [r_ap, r_ap + aperture_width).
        current_ap_mask = (r >= r_ap) & (r < r_ap + ap_width)
        # We restrict to the annulus of interest.
        current_ap_mask = current_ap_mask & annulus_mask
        # The ROI is the intersection of the current aperture and the wedge.
        current_roi_mask = current_ap_mask & wedge_mask
        
        # Compute the signal: sum of pixel values in the ROI.
        if np.any(current_roi_mask):
            signal = np.mean(image[current_roi_mask])
        else:
            signal = 0
        
        # For noise estimation, use the pixels in the current aperture that are outside the wedge.
        current_nonroi_mask = current_ap_mask & (~wedge_mask)
        if np.any(current_nonroi_mask):
            noise = np.std(image[current_nonroi_mask])
        else:
            noise = 1e-6  # Avoid division by zero.
        
        sn = signal / noise
        # plt.imshow(image*current_roi_mask, origin="lower")
        # plt.title(sn)
        # plt.show()
        scan_positions.append(r_ap + ap_width / 2.0)  # use aperture center as the effective radius.
        sn_values.append(sn)
        sums.append(signal)
    # plt.imshow(image, origin="lower")
    # plt.colorbar()
    # # plt.title(sn)
    # plt.show()
    
    scan_positions = np.array(scan_positions)
    sn_values = np.array(sn_values)
    
    # Determine the best aperture based on maximum S/N that meets the threshold.
    valid = sn_values >= sn_threshold
    if not np.any(valid):
        best_r = np.nan
        best_sum = np.nan
        best_sn = np.nan
    else:
        best_index = np.argmax(sn_values * valid)  # multiply by valid mask to ignore non-valid bins.
        best_r = scan_positions[best_index]
        best_sum = sums[best_index]
        best_sn = sn_values[best_index]

    # # Full S/N map is image divided by noisemap.
    # full_sn_map = np.full(image.shape, np.nan)
    # # Only update pixels within the overall annulus.
    # indices = np.where(annulus_mask)
    # for y_idx, x_idx in zip(*indices):
    #     rp = r[y_idx, x_idx]
    #     # Only assign if rp is in the scanned range.
    #     if rp >= r_start and rp <= r_end - ap_width:
    #         # Find the scanned aperture center closest to rp.
    #         idx = np.argmin(np.abs(scan_positions - rp))
    #         full_sn_map[y_idx, x_idx] = sn_values[idx]
    # # Also produce the ROI S/N map (only keep S/N values where wedge_mask is True).
    # # roi_sn_map = np.where(wedge_mask & annulus_mask, full_sn_map, np.nan)


    return best_r, best_sum, best_sn

def fig_wedge_photometry(wedge_counts, wedge_errs, save_to, figsize=(8,8)):
    fig, ax = plt.subplots(1,1, figsize=figsize, subplot_kw={"polar": True})
    assert len(wedge_counts) == len(wedge_errs)
    nwedges = len(wedge_counts)
    theta = np.linspace(0, 2 * np.pi, nwedges, endpoint=False)
    width = 2 * np.pi / nwedges
    ax.set_theta_zero_location('N')
    ax.tick_params(labelsize=20,labelcolor='b',pad=12)
    barcolor = "orange"
    bars = ax.bar(
            theta,
            wedge_counts,
            width=width,
            # bottom=inner_radius,
            # yerr=wedgeErr,
            color=barcolor,
            alpha=1.0,
            edgecolor='black',
            align='edge',
            label="Counts"
        )
    for bar, std in zip(bars, wedge_errs):
        # Plot the 3-sigma errorbars for each wedge region
        bar_center = bar.get_x() + bar.get_width() / 2
        bar_height = bar.get_height()
        ax.errorbar(
            bar_center, bar_height, yerr=std*3, fmt='--', color="black", ecolor="black", capsize=3
        )

    plt.tight_layout()
    plt.savefig(f"{save_to}.png")
    # plt.show()