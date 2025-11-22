import sys
import logging
import xconf
from magaox.indi.device import XDevice, BaseConfig

from hcipy import *
import numpy as np
import time
from matplotlib import pyplot as plt

from magaox.camera import XCam
from magaox.deformable_mirror import XDeformableMirror

from skimage import feature

import purepyindi2 as indi

from pupil_alignment_utils import *
from focal_alignment_utils_josh import *
from scipy import stats


log = logging.getLogger(__name__)

# TODO: add support for auto aligning the other FPMs (future project)
@xconf.config
class FpmCtrlConfig(BaseConfig):
    """Python INDI device for auto aligning the knife edge mask.

    """
    fpm : str = xconf.field(default="knife edge", help="FPM to auto align")
    # TODO: Consider renaming this b/c it could be confused with FPM diameter
    maskdiameter : float = xconf.field(default=20.0, help="diameter of pixel mask for edge detection")
    threshold : float = xconf.field(default=3.0, help="set pix vals below threshold to 0")
    sigma : float = xconf.field(default=4.0, help="width of Gaussian kernel in high pass filter")

class fpmCtrl(XDevice):
    config : FpmCtrlConfig

    def loop(self):
        log.info("Looping")

    def knife_edge_dist(self):
        grid = image.grid

        center_mask = make_circular_aperture(self.maskdiameter)(grid)
        mask = ((image / np.std(image)) < self.threshold)

        # Apply circular aperture mask to thresholded field
        masked_im = (center_mask*mask)

        # Detect edge and convert edge arr back to Field obj
        edge = feature.canny(masked_im.shaped, sigma=self.sigma) # Widened Gaussian filter for edge detection on noisier images
        edge_field = Field(edge.ravel(), image.grid)

        
        # Get x, y edge coords from edge Field obj
        # x_edge = image.grid.x[edge_field>0]
        y_edge = image.grid.y[edge_field>0]

        # if plot:
        #     plt.plot(0, y_edge, 'C1o')
        #     # plt.plot(0, 0, 'C1o')

        # imshow_field(edge_field)
        # print(y_edge)

        # debugging
        d_normal = y_edge
        # Identify minimum absolute distance from the origin
        min_dist_indx = abs(np.argmin(d_normal))

        if plot_edge:
            imshow_field(edge_field)
            plt.plot(0, y_edge[min_dist_indx], 'C1o')

        # if verbose:
        #     print(f'FPM Offset Dist: {y_edge[min_dist_indx]}')

        return np.array([0.0, y_edge[min_dist_indx]])
    
        # practice commanding fpm to specific offset pos

    # TODO: modify this such that it works in the app format
    def knife_edge_align(self):
        fwfpm_offset = 0
        fwfpm_cmd = 0
        edge_dist_list = []
        # delta_list = []
        # delta_cnt = 0
        lower_bound = 4.5
        upper_bound = 6
        stagnation_count = 0
        patience = 1
        tol = 1e-3
        threshold = 3
        cutoff = 1e-4

        fwfpm_current = client['fwfpm.filter.current']
        client['fwfpm.filter.target'] = np.clip(fwfpm_current + fwfpm_cmd + fwfpm_offset, lower_bound, upper_bound)
        time.sleep(5.0)

        Nitrs = 25

        # TODO: Ti
        # gain = 5e-3
        gain = 5e-3

        # get edge dist reference pos
        edge_dist_ref = knife_edge_dist(im_ref, mask_diameter=10, threshold=threshold)
        delta_list = [edge_dist_ref[1]]


        for k in range(Nitrs):
            # fwfpm_current = client['fwfpm.filter.current']
            # print(f'Current FPM Position: {fwfpm_current}')
            print(f'Iteration: {k}')
            client['fwfpm.filter.target'] = np.clip(fwfpm_current + fwfpm_cmd + fwfpm_offset, lower_bound, upper_bound)
            time.sleep(2)
            
            im = cam.grab_stack(2)
            im = Field(im.ravel(), cam.grid)
            # norm_im = np.sum(im * norm_phot_aperture, axis=-1, keepdims=True)

            # im /= norm_im
            
            # measure shift from reference
            edge_dist = knife_edge_dist(im, mask_diameter=10, threshold=threshold)
            # delta_list = [edge_dist_ref[1]]
            # delta_list.append(delta)

            edge_dist_list.append(edge_dist[1])
            # delta_list.append(edge_dist_ref[1])
            # delta = 0
            # delta_list.append(delta)
            # print(shift)

            # TODO: Check to see if algo is getting stuck in local minimum
            if k > 0:
                # delta = np.abs(edge_dist_list[k] - edge_dist_list[k-1])
                delta = np.abs(edge_dist_ref[1] - edge_dist_list[k])
                delta_list.append(delta)
                # delta_cnt+=1

                # TODO: check if this works...
                if k > 1:
                    if np.abs((delta_list[k-1] < delta_list[k])):
                        gain *= -0.5 # decrease the stepsize whenever passing the 0 point
                        print('GO BACK')
                        

                # attempt to exit local minima by executing a larger move
                # # TODO: maybe remove this...
                #     if k > 2:
                delta_delta = np.abs(delta_list[k-1] - delta_list[k])
                if (delta_delta <= tol) and (delta_delta >= cutoff):
                    stagnation_count+=1
                    if stagnation_count >= patience:
                        print(f" FWFPM is likely stuck...trying again")
                        continue
                else:
                    stagnation_count = 0
                    
                fwfpm_cmd = fwfpm_cmd + gain
                print(f'FPM Offset Dist: {delta}')

                print(f'Sending Command: {fwfpm_cmd}')
                if delta <= cutoff:
                    break
                
            # iteratively converge on correct FPM pos