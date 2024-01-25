#!/usr/bin/env python
from hcipy import *
import numpy as np
    
def measure_center_position(image, threshold=1, mask_diameter=20):
	center_mask = make_circular_aperture(mask_diameter)(image.grid)
	mask = center_mask * ((image / np.nanstd(image)) < threshold)

	if np.sum(mask) > 0:
		xc = np.sum(mask * image.grid.x) / np.sum(mask)
		yc = np.sum(mask * image.grid.y) / np.sum(mask)
		return np.array([xc, yc])
	else:
		return np.array([0.0, 0.0])

def knife_edge_dist(self, image):
	THETA  = -25. * (np.pi/180) # Angle measured from x-axis
	
	# Read in field
	im_field = util.read_field(image)
	grid = im_field.grid

	# Threshold the data
	# threshold = 1
	threshold = 0.5
	mask_diameter = 200
	center_mask = make_circular_aperture(mask_diameter)(im_field.grid)
	mask = ((im_field / np.std(im_field)) < threshold)

	# Apply circular aperture mask to thresholded field
	masked_im = (center_mask*mask)

	# Convert masked field back to 2D array
	masked_im = masked_im.shaped
	masked_im_arr = np.array(masked_im)
	masked_im_arr = np.flip(masked_im_arr, axis=0)

	# Detect edge and convert edge arr back to Field obj
	edge = feature.canny(masked_im_arr, sigma=4) # Widened Gaussian filter for edge detection on noisier images

	edge_field = Field(edge.ravel(), im_field.grid)

	# Get x, y edge coords from edge Field obj
	x_edge = im_field.grid.x[edge_field>0]
	y_edge = im_field.grid.y[edge_field>0]

	# Project x, y coords onto x and y axes
	d = x_edge * np.sin(THETA) + y_edge * np.cos(THETA) 

	# Identify minimum absolute distance from the origin
	min_dist_indx = abs(np.argmin(d))

	dmin = d[min_dist_indx]
	dmin = abs(dmin) 

	return dmin