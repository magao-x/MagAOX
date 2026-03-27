import numpy as np

def _get_radial_indices(image, center=None):
    # Generate grid of (x, y) coordinates
    y, x = np.indices(image.shape)
    
    # If no center is provided, use the center of the image.
    if center is None:
        center = (x.max() / 2.0, y.max() / 2.0)
    
    # Compute the radial distance of each pixel from the center.
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Convert distances to integers (each integer corresponds to a radial bin)
    r_int = r.astype(int)

    return r_int



def radial_profile(image, center=None):
    """
    Compute the radial profile of a 2D image.

    Parameters
    ----------
    image : np.ndarray
        2D numpy array representing the image.
    center : tuple of float, optional
        (x, y) coordinates of the center. If None, the center of the image is used.

    Returns
    -------
    radial_prof : np.ndarray
        1D array containing the average pixel value for each integer radius.
    r : np.ndarray
        1D array of radius values corresponding to the bins in radial_prof.
    """
    r_int = _get_radial_indices(image, center=center)
    # Pre-allocate array for median values. The maximum radius is the maximum integer value in r_int.
    max_r = r_int.max() + 1
    radial_prof = np.empty(max_r, dtype=image.dtype)
    
    for i in range(max_r):
        mask = (r_int == i)
        if np.any(mask):
            radial_prof[i] = np.median(image[mask])
        else:
            radial_prof[i] = np.nan  # or you can choose to set to zero
    
    # Build a 2D radial profile map by indexing into the radial profile.
    radial_map = radial_prof[r_int]
    
    return radial_map

def radial_scaling(image, center=None):
    r_int = _get_radial_indices(image, center=center)
    max_r = r_int.max() + 1
    r_distances = np.empty(max_r, dtype=image.dtype)
    r_distances_squared = np.empty(max_r, dtype=image.dtype)

    for i in range(max_r):
        r_distances[i] = i
        r_distances_squared[i] = i*i


    # Now make it into a 2D map of radial distances
    rad_dist_map = r_distances[r_int]
    rad_dist_squared_map = r_distances_squared[r_int]

    return rad_dist_map, rad_dist_squared_map