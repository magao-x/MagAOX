#!/usr/bin/env python
from hcipy import *
import numpy as np
from skimage import feature

def measure_center_position(image, threshold=1, mask_diameter=20):
    center_mask = make_circular_aperture(mask_diameter)(image.grid)
    mask = center_mask * ((image / np.nanstd(image)) < threshold)

    if np.sum(mask) > 0:
        xc = np.sum(mask * image.grid.x) / np.sum(mask)
        yc = np.sum(mask * image.grid.y) / np.sum(mask)
        return np.array([xc, yc])
    else:
        return np.array([0.0, 0.0])

def knife_edge_dist(image, theta=-0.47, mask_diameter=200, threshold=0.5):
    '''Measure the distance from the knife edge focal plane mask to the center of an image.


        Parameters
        ----------
        image : array of floats 
            Default: None.
        theta: integer
            The knife edge mask angle with respect to the image's x-axis. Default: 0.
            mask_diameter: The diameter of the circular mask applied to the image for edge detection. Default: 200.
            threshold: The sigma used to mask pixels along the circular mask edge. Default: 0.5.

        Returns
        -------
        np.array()
            A NumPy array containing the knife edge distance in X and Y with respect to the image center.
        '''
    THETA  = np.deg2rad(theta) # Angle measured from x-axis
    
    grid = image.grid

    center_mask = make_circular_aperture(mask_diameter)(grid)
    mask = ((image / np.std(image)) < threshold)

    # Apply circular aperture mask to thresholded field
    masked_im = (center_mask*mask)

    # Detect edge and convert edge arr back to Field obj
    edge = feature.canny(masked_im.shaped, sigma=4) # Widened Gaussian filter for edge detection on noisier images

    edge_field = Field(edge.ravel(), image.grid)

    # Get x, y edge coords from edge Field obj
    x_edge = image.grid.x[edge_field>0]
    y_edge = image.grid.y[edge_field>0]

    # Project x, y coords onto x and y axes
    d_normal = x_edge * np.sin(THETA) + y_edge * np.cos(THETA) 
    d_parallel = x_edge * np.sin(THETA + np.pi/2) + y_edge * np.cos(THETA + np.pi/2) 
    d = np.hypot(d_normal, d_parallel)
    
    # Identify minimum absolute distance from the origin
    min_dist_indx = np.argmin(abs(d))

    return np.array([0.0, y_edge[min_dist_indx]])