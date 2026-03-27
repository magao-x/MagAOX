import numpy as np

def make_annular_mask(image_shape, inner_radius, outer_radius):
    """Make an annular boolean mask.
    
    To work with sep, True values are masked, False values are unmasked.
    """
    mask = np.ones(image_shape)
    y, x = np.ogrid[:image_shape[0], :image_shape[1]]
    center_y, center_x = image_shape[0] // 2, image_shape[1] // 2
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask[(r > inner_radius) & (r < outer_radius)] = 0
    return mask