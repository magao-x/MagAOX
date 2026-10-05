import sys
import logging
from enum import Enum
import time
import numpy as np

import xconf

from magaox.indi.device import XDevice, BaseConfig
from magaox.camera import XCam
from magaox.constants import StateCodes

from purepyindi2 import device, properties, constants
from purepyindi2.messages import DefNumber, DefSwitch, DefLight, DefText

import hcipy as hp
from scipy.optimize import minimize

class AdcFitter:
    def __init__(self):
        pass

    def window_field(data,center,width,height):
        '''crop a smaller field out of a big field. data must be an hcipy field.'''
        indx = data.grid.closest_to(center)
        y_ind, x_ind = np.unravel_index(indx, data.shaped.shape)
        cutout = data.shaped[(y_ind-height//2):(y_ind + height//2), (x_ind-width//2):(x_ind+width//2)]
        sub_grid = make_pupil_grid([width, height], [width * data.grid.delta[0], height * data.grid.delta[1]])
        return Field(cutout.ravel(), sub_grid)

    def crop_image(image,extent=100,mask_diam=0.5E-6): 
        '''cuts out a centered PSF with the central core masked'''
        img_normalized = image/np.max(image)

        img_subtracted = img_normalized >0.1
        center_of_intensity = np.array([sum(img_subtracted*img_subtracted.grid.x)/sum(img_subtracted),sum(img_subtracted*img_subtracted.grid.y)/sum(img_subtracted)])
        mask_ap = make_circular_aperture(mask_diam,center_of_intensity)
        mask = mask_ap(img_subtracted.grid)
        mask = abs(mask - 1)
        image = mask * image

        image = window_field(image,[center_of_intensity[0],center_of_intensity[1]],extent,extent)
        image = Field([x if x>0 else 0 for x in image],image.grid)

        return image

    def speckle_cutout(img,speckle_number,angle,f=10,window_size=30):
        extent=1E-6
        separation = f #note that this version is slightly different than the _testing notebook version
        #because it assumes the grid is in units of lam/D instead of pixel angular size!

        speckle_coords = np.array([
            [0,           separation],
            [separation,  0],
            [0,          -separation],
            [-separation, 0]
        ])
        speckle_center = speckle_coords[speckle_number]
        rect = make_rotated_aperture(make_rectangular_aperture(size=(extent,extent), center=speckle_center), np.deg2rad(-angle))(img.grid)
        speckle_img = rect * img.copy()

        max_pixel = speckle_img.grid[np.argmax(speckle_img)]
        new_img = window_field(img,[max_pixel[0],max_pixel[1]],window_size,window_size)
        return np.array(new_img.shaped)

    def moment_angle(
        crop: np.ndarray,
        fwhm_px: float = 6.0,
        max_iter: int = 5,
        tol_deg: float = 0.05,
    ) -> float:

        # 0. Physical Positivity Enforcer (Fix for pre-subtracted negative pixels)
        crop_pos = np.maximum(crop.astype(np.float64), 0.0)
        ny, nx = crop_pos.shape
        y_grid, x_grid = np.mgrid[0:ny, 0:nx]

        # 1. Estimate Residual Noise Floor on positive-floored signal
        border_pixels = np.concatenate(
            [
                crop_pos[0, :],
                crop_pos[-1, :],
                crop_pos[:, 0],
                crop_pos[:, -1],
            ]
        )

        # Estimate noise level using robust MAD on non-zero background
        bg_level = np.median(border_pixels)
        mad = np.median(np.abs(border_pixels - bg_level))
        sigma_noise = 1.4826 * mad if mad > 0 else 1

        # 2. Subtract residual local background and re-enforce non-negativity
        img_sub = np.maximum(crop_pos - bg_level, 0.0)

        # 2. Smooth to find initial 2D Peak & Estimate Peak SNR
        sigma_smooth = fwhm_px / 2.355
        smoothed = ndimage.gaussian_filter(np.maximum(img_sub, 0.0), sigma=sigma_smooth)
        peak_y, peak_x = np.unravel_index(np.argmax(smoothed), smoothed.shape)
        peak_val = img_sub[peak_y, peak_x]

        snr_est = peak_val / sigma_noise if sigma_noise > 0 else 100.0

        # 3. Dynamic SNR Adjustments
        if snr_est < 15.0:
            # Low SNR regime (5-15): Aggressive noise cut, tight window
            snr_thresh_factor = 2.5
            min_win_scale = 0.8
            max_win_scale = 1.2
        elif snr_est < 50.0:
            # Mid SNR regime (15-50): Balanced settings
            snr_thresh_factor = 1.5
            min_win_scale = 0.8
            max_win_scale = 1.8
        else:
            # High SNR regime (>50): Relaxed threshold, wide window for accuracy
            snr_thresh_factor = 1.0
            min_win_scale = 1.0
            max_win_scale = 2.5

        # Threshold image based on estimated SNR
        img_clean = np.maximum(
            img_sub - (snr_thresh_factor * sigma_noise), 0.0
        )

        # Initial state
        x_bar, y_bar = float(peak_x), float(peak_y)
        angle = 0.0
        sigma_u = fwhm_px * 0.5
        sigma_v = fwhm_px * 0.5

        # 4. Iterative Moment Computation
        for iteration in range(max_iter):
            prev_angle = angle

            dx = x_grid - x_bar
            dy = y_grid - y_bar

            # Check centroid drift: Don't let iteration drift > 1 FWHM from smoothed peak
            dist_from_peak = np.sqrt((x_bar - peak_x) ** 2 + (y_bar - peak_y) ** 2)
            if dist_from_peak > fwhm_px:
                x_bar, y_bar = float(peak_x), float(peak_y)
                dx = x_grid - x_bar
                dy = y_grid - y_bar

            # Rotate to principal axis frame
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a

            # Soft Elliptical Gaussian Window
            window = np.exp(-0.5 * ((u / sigma_u) ** 2 + (v / sigma_v) ** 2))
            weights = img_clean * window
            total_weight = np.sum(weights)

            if total_weight == 0:
                # Fallback for extreme noise: use raw peak-centered unweighted moments
                weights = np.maximum(img_clean, 0.0)
                total_weight = np.sum(weights)
                if total_weight == 0:
                    return 0.0

            # Refine Centroid
            x_bar = np.sum(x_grid * weights) / total_weight
            y_bar = np.sum(y_grid * weights) / total_weight

            # Compute 2nd Central Moments
            dx_c = x_grid - x_bar
            dy_c = y_grid - y_bar

            mu20 = np.sum((dx_c**2) * weights)
            mu02 = np.sum((dy_c**2) * weights)
            mu11 = np.sum((dx_c * dy_c) * weights)

            # Unbiased Angle Calculation
            angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

            # Calculate shape variances
            var_u = 0.5 * (
                mu20 + mu02 + np.sqrt((mu20 - mu02) ** 2 + 4 * (mu11**2))
            )
            var_v = 0.5 * (
                mu20 + mu02 - np.sqrt((mu20 - mu02) ** 2 + 4 * (mu11**2))
            )

            # Adaptively update window dimensions with SNR-based bounds
            raw_sigma_u = np.sqrt(max(var_u / total_weight, 0.25))
            raw_sigma_v = np.sqrt(max(var_v / total_weight, 0.25))

            sigma_u = np.clip(
                raw_sigma_u, min_win_scale * fwhm_px, max_win_scale * fwhm_px
            )
            sigma_v = np.clip(
                raw_sigma_v, min_win_scale * fwhm_px, max_win_scale * fwhm_px
            )

            # Check convergence
            if (
                np.abs(np.degrees(angle - prev_angle)) < tol_deg
                and iteration > 0
            ):
                break

        return float(np.degrees(angle))


@xconf.config
class CameraConfig:
    """
    """
    shmim : str = xconf.field(help="Name of the camera device (specifically, the associated shmim, if different)")
    dark_shmim : str = xconf.field(help="Name of the dark frame shmim associated with this camera device")

@xconf.config
class AdcCtrlConfig(BaseConfig):
    """ Active ADC control
    """
    camera : CameraConfig = xconf.field(help="Camera to use")
    sleep_interval_sec : float = xconf.field(default=0.25, help="Sleep interval between loop() calls")

class States(Enum):
    IDLE = 0
    CLOSED_LOOP = 1
    ONESHOT = 2
    MEASURE_ONLY = 3

class adcCtrl(XDevice):
    config: AdcCtrlConfig

    def setup(self):
        self.log.debug(f"I was configured! See? {self.config=}")

        fsm = properties.TextVector(name='fsm')
        fsm.add_element(DefText(name='state', _value=StateCodes.INITIALIZED.name))
        self.add_property(fsm)

        sv = properties.SwitchVector(
            name='state',
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        sv.add_element(DefSwitch(name="idle", _value=constants.SwitchState.ON))
        sv.add_element(DefSwitch(name="adcLoop", _value=constants.SwitchState.OFF))
        sv.add_element(DefSwitch(name="oneshot", _value=constants.SwitchState.OFF))
        sv.add_element(DefSwitch(name="measure-only", _value=constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_state)

        nv = properties.NumberVector(name='n_avg')
        nv.add_element(DefNumber(
            name='current', label='Number of frames', format='%i',
            min=1, max=150, step=1, _value=1
        ))
        nv.add_element(DefNumber(
            name='target', label='Number of frames', format='%i',
            min=1, max=150, step=1, _value=1
        ))
        self.add_property(nv, callback=self.handle_n_avg)

        nv = properties.NumberVector(name='no_measurements')
        nv.add_element(DefNumber(
            name='number', label='number', format='%i',
            min=1, max=100.00, step=1, _value=1
        ))
        self.add_property(nv, callback=self.handle_no_measurements)

        nv = properties.NumberVector(name='gain')
        nv.add_element(DefNumber(
            name='current', label='ADC Loop Gain', format='%.2f',
            min=0.00, max=1.00, step=0.01, _value=0.10
        ))
        nv.add_element(DefNumber(
            name='target', label='ADC Loop Gain', format='%.2f',
            min=0.00, max=1.00, step=0.01, _value=0.10
        ))
        self.add_property(nv, callback=self.handle_gain)

        nv = properties.NumberVector(name='offset')
        nv.add_element(DefNumber(
            name='current', label='offset', format='%.2f',
            min=-45, max=45, step=0.01, _value=0.0
        ))
        nv.add_element(DefNumber(
            name='target', label='offset', format='%.2f',
            min=-45, max=45, step=0.01, _value=0.0
        ))
        self.add_property(nv, callback=self.handle_offset)

        nv = properties.NumberVector(name='ctrl_mtx')
        nv.add_element(DefNumber( #first element
            name='m00', label='m00', format='%.4f',
            min=-10.00, max=10.00, step=0.0001, _value=0.21178766
        ))
        nv.add_element(DefNumber(
            name='m01', label='m01', format='%.4f',
            min=-10.00, max=10.00, step=0.0001, _value=0.19275196
        ))
        self.add_property(nv, callback=self.handle_ctrl_mtx)

        sv = properties.SwitchVector(
            name='labmode',
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        sv.add_element(DefSwitch(name="toggle", _value=constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_labmode)

        sv = properties.SwitchVector(
            name='reset_deltaADCs',
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        sv.add_element(DefSwitch(name="request", _value=constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_reset)

        self.client.get_properties('adctrack')
        self.client.get_properties('fwsci1')

        self.log.info("Found camera: {:s}".format(self.config.camera.shmim))
        self.camera = XCam(
            self.config.camera.shmim,
            pixel_size=6.0/21.0,
            use_hcipy=False,
            indi_client=self.client
        )

        self._state = States.IDLE

        #self._loop_counter = 0
        self._n_avg = 1
        self._gain = 0.5
        self._command = 0
        self._control_mtx = np.array([0.21178766, 0.19275196])
        self._crop_extent = 400
        self.delta_1 = 0
        self.delta_2 = 0
        self._offset = 0
        self._mask_diam = 50
        self._lab = False
        self._no_measurements = 1

        if self.client['adctrack.deltaADC1.current'] != 0:
            self.set_command(0,0)
            self.send_command()

        if self.client['fwsci1.filterName.i'] == constants.SwitchState.ON:
            self._center_wavelength = 762E-9
        elif self.client['fwsci1.filterName.z'] == constants.SwitchState.ON:
            self._center_wavelength = 908E-9
        else:
            self._center_wavelength = 656E-9

        self.ADC = AdcFitter(wavelength=self._center_wavelength)
        self.log.debug(f'initial normalized wavelength value: {self.ADC.normalized_wavelength}')
        #self.update_wavelength()
        self.ADC.set_control_mtx(self._control_mtx)

        self.properties['fsm']['state'] = StateCodes.READY.name
        self.update_property(self.properties['fsm'])

    def handle_state(self, existing_property, new_message):
        target_list = ['idle', 'adcLoop', 'oneshot','measure-only']
        for key in target_list:
            if existing_property[key] == constants.SwitchState.ON:
                current_state = key

        if current_state not in new_message:

            for key in target_list:
                existing_property[key] = constants.SwitchState.OFF
                if key in new_message:
                    existing_property[key] = new_message[key]

                    if key == 'idle':
                        self._state = States.IDLE
                        self.properties['fsm']['state'] = StateCodes.READY.name
                        self.log.debug('State changed to idle')
                    elif key == 'adcLoop':
                        self._state = States.CLOSED_LOOP
                        self.properties['fsm']['state'] = StateCodes.OPERATING.name
                        self.log.debug('State changed to closed-loop')
                    elif key == 'oneshot':
                        self._state = States.ONESHOT
                        self.properties['fsm']['state'] = StateCodes.OPERATING.name
                        self.log.debug('State changed to oneshot')
                    elif key == 'measure-only':
                        self._state = States.MEASURE_ONLY
                        self.properties['fsm']['state'] = StateCodes.OPERATING.name
                        self.log.debug('State changed to measure-only')

            self.update_property(existing_property)
            self.update_property(self.properties['fsm'])

    def handle_labmode(self,existing_property, new_message):
        if 'toggle' in new_message and new_message['toggle'] is constants.SwitchState.ON:
            self.log.debug('changing to lab mode')
            existing_property['toggle'] = constants.SwitchState.ON
            self._lab = True
        else:
            self.log.debug('changing to onsky mode')
            existing_property['toggle'] = constants.SwitchState.OFF
            self._lab = False

        self.update_property(existing_property)

    def handle_reset(self,existing_property, new_message):
        if 'request' in new_message and new_message['request'] is constants.SwitchState.ON:
            self.log.debug('resetting deltaADC properties')
            existing_property['request'] = constants.SwitchState.OFF
            self.set_command(0,0)
            self.send_command()
            self._command = 0

        self.update_property(existing_property)

    def handle_n_avg(self, existing_property, new_message):
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
            self._n_avg = int(new_message['target'])
            self.log.debug(f'now averaging over {self._n_avg} frames')
        self.update_property(existing_property)

    def handle_no_measurements(self, existing_property, new_message):
        if 'number' in new_message and new_message['number'] != existing_property['number']:
            existing_property['number'] = new_message['number']
            self._no_measurements = int(new_message['number'])
            self.log.debug(f'now averaging {self._no_measurements} measurements before sending command')
        self.update_property(existing_property)

    def handle_gain(self, existing_property, new_message):
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
            self._gain = float(new_message['target'])
            self.log.debug(f'loop gain changed to {self._gain}')
        self.update_property(existing_property)

    def handle_offset(self, existing_property, new_message):
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
            self._offset = float(new_message['target'])
            self.log.debug(f'offset changed to {self._offset}')
            self.send_command()
        self.update_property(existing_property)

    def handle_ctrl_mtx(self, existing_property, new_message):
        old_matrix = self._control_mtx
        if 'm00' in new_message and new_message['m00'] != existing_property['m00']:
            existing_property['m00'] = new_message['m00']
            self._control_mtx[0] = float(new_message['m00'])

        if 'm01' in new_message and new_message['m01'] != existing_property['m01']:
            existing_property['m01'] = float(new_message['m01'])
            self._control_mtx[1] = new_message['m01']

        self.log.debug(f'control matrix changed to {self._control_mtx}')
        self.update_property(existing_property)

    def transition_to_idle(self):
        self.properties['state']['oneshot'] = constants.SwitchState.OFF
        self.properties['state']['adcLoop'] = constants.SwitchState.OFF
        self.properties['state']['measure-only'] = constants.SwitchState.OFF
        self.properties['state']['idle'] = constants.SwitchState.ON
        self.update_property(self.properties['state'])
        self._state = States.IDLE

    def set_command(self, d1, d2):
        self.delta_1 = d1
        self.delta_2 = d2

    def add_command(self, d1,d2):
        self.delta_1 += d1
        self.delta_2 += d2

    def send_command(self):
        self.client['adctrack.deltaADC1.target'] = self.delta_1 + self.delta_2 + self._offset
        self.client['adctrack.deltaADC2.target'] = self.delta_1 - self.delta_2 + self._offset

        do_check = True
        tolerance = 0.05
        while do_check:

            current_1 = self.client['adctrack.deltaADC1.current']
            current_2 = self.client['adctrack.deltaADC2.current']

            if abs(current_1 - self.delta_1 - self.delta_2 - self._offset) < tolerance and abs(current_2 - self.delta_1 + self.delta_2 - self._offset) < tolerance:
                do_check = False

            time.sleep(0.05)

    def loop(self):
        if self._state == States.CLOSED_LOOP:
                pass 
                # if np.abs(error*self._gain) < 0.7: #setting a threshold so the prisms don't do anything crazy     
                #     self.add_command(error * self._gain,0)
                #     self.send_command()
                #     self.log.info(f'delta command: {error * self._gain}')
                #     self.log.info(f'total command: {self.delta_1}')
                # else: self.log.info(f'ADC command {error} exceeds acceptable threshold and was not sent')
        
        elif self._state == States.ONESHOT:
            pass

        elif self._state == States.MEASURE_ONLY:
            measurements = []
            error = 0

            for i in range(self._no_measurements):
                    #grab images
                    img = self.camera.grab_stack(self._n_avg)

                    #make into hcipy fields with correct l/D dimensions
                    img = np.pad(img,pad_width=50, mode='constant', constant_values=0)
                    dim = np.sqrt(img.size)
                    extent = dim * 6/21
                    pgrid = make_pupil_grid(dim,extent)
                    img = Field(img.ravel(),pgrid)
                    img -= np.median(img) 

                    #crop so that PSF is in the center, do not mask
                    img = self.ADC.crop_image(img, self._crop_extent, mask_diam=0)
                    
                    #radial profile subtract 
                    binc, profile, std_profile, ncount = radial_profile(img,5) 
                    r_coordinates = img.grid.as_('polar').r 
                    radial_map = np.interp(r_coordinates, binc, profile) 
                    img_subtracted = img - radial_map

                    #TODO: make mask diameter dynamic based on sparkle separation
                    img = self.ADC.crop_image(img_subtracted,extent=self._crop_extent,mask_diam=30) 
                    
                    #background subtraction and set negatives to zero
                    bg = np.median(img)
                    img -= bg
                    img[img <0] = 0

                    #measure angles
                    angles = np.zeros(4)
                    for i in range(4):
                        speckle_img = speckle_cutout(cropped,i,angle,f,window_size=20,search_extent=20)
                        angles[i] = np.abs(moment_angle(speckle_img))

                    self.log.debug(f'measured speckle angles: {angles}')

                    #calculate command
                    # command = np.squeeze(self.ADC.calculate_command(pairs))
                    # self.log.debug(f'single error command: {-command}')
                    # measurements.append(command)
                
                error = -np.nanmean(measurements)
                self.log.debug(f'mean predicted command across {self._no_measurements} measurements: {-error} (measured, not sent)')



# Used to make the pyproject.toml just a little simpler,
# with fewer repetitions of the app name:
main = adcCtrl.console_app
