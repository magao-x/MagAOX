#!/usr/bin/env python
from hcipy import *
import numpy as np
    
def measure_center_position(image, threshold=1, mask_diameter=20):
    center_mask = make_circular_aperture(mask_diameter)(image.grid)
    mask = center_mask * ((image / np.std(image)) < threshold)   
    
    xc = np.sum(mask * image.grid.x) / np.sum(mask)
    yc = np.sum(mask * image.grid.y) / np.sum(mask)
    return np.array([xc, yc])