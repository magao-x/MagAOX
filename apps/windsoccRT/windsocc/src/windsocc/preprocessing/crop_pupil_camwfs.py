import numpy as np

DEFAULT_PUPIL_CENTERS = {
    "ul": (30, 90),
    "ur": (90, 90),
    "ll": (30, 30),
    "lr": (90, 30),
}
DEFAULT_PUPIL_MASK_RADIUS = 28

def circular_mask(im2masksize, centerpt, outrad):
    blank = np.zeros(im2masksize)
    xcenter, ycenter = centerpt
    xrange = np.arange(im2masksize[0])[None,:] - xcenter
    yrange = np.arange(im2masksize[1])[:,None] - ycenter
    rho2d = np.sqrt(xrange**2 + yrange**2)
    blank[np.where(rho2d <= outrad)] = 1
    mask_circle_ap = blank
    return mask_circle_ap

def get_square_thumbnail_from_pupil(maskedpupilimage):
    """
    Extract a square thumbnail from a 2D image based on the diameter of the circular region of interest.

    Parameters:
        image (np.ndarray): 2D NumPy array representing the image. The circular region of interest
                           should be non-zero, and the rest should be zero.

    Returns:
        np.ndarray: Square thumbnail containing the circular region.
    """
    # Get the non-zero indices of the circular region
    y_indices, x_indices = np.nonzero(maskedpupilimage)

    # Find the bounding box of the circular region
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # Calculate the diameter of the circle
    diameter = max(x_max - x_min, y_max - y_min)

    # Determine the center of the circle
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    # Calculate the half side length of the square
    half_side = diameter // 2

    # Define the square's bounds, ensuring they stay within image boundaries
    x_start = max(cx - half_side, 0)
    x_end = min(cx + half_side, maskedpupilimage.shape[1])
    y_start = max(cy - half_side, 0)
    y_end = min(cy + half_side, maskedpupilimage.shape[0])

    # Extract the square thumbnail
    thumbnail = maskedpupilimage[y_start:y_end, x_start:x_end]

    return thumbnail

def crop_quadrant(image2crop, quadrant_pupil, pupil_centers=None, pupil_mask_radius=DEFAULT_PUPIL_MASK_RADIUS):
    framesize = image2crop.shape
    quadrant_key = quadrant_pupil.lower()
    centers = pupil_centers or DEFAULT_PUPIL_CENTERS
    if quadrant_key not in centers:
        raise ValueError(f"Unknown pupil quadrant: {quadrant_pupil}")

    pupil_center = tuple(centers[quadrant_key])
    pupil_mask = circular_mask(framesize, centerpt=pupil_center, outrad=pupil_mask_radius)
    cropped_camwfs = image2crop * pupil_mask
    isolated_pupil = get_square_thumbnail_from_pupil(cropped_camwfs)

    return isolated_pupil


