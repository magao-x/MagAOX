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
    def __init__(self,wavelength=656E-9,bandwidth=100E-9,grating_angle=-28,grating_freq=47):
        self.wavelength = wavelength
        self.bandwidth = bandwidth
        self.grating_angle = grating_angle
        self.grating_freq = grating_freq
        self.normalized_wavelength = wavelength / 656E-9 #normalizing the wavelengths to the ha values
        self.normalized_bandwidth = bandwidth / 656E-9 
        self.maxiter = 6
        self.control_mtx = np.matrix([[0,0],])
        self.current_speckle = None

    def set_measurement(self, data):
        self.data = data

    def make_gaussian(self, mu_x, mu_y, sigma_x,sigma_y, orientation):
        def func(grid):
            new_grid = grid.shifted([-mu_x, -mu_y]).rotated(orientation)
            x = new_grid.x / sigma_x
            y = new_grid.y / sigma_y
            r2 = x**2 + y**2
            return hp.Field(np.exp(-0.5 * r2), grid)
        return func

    def satellite_spot(self, amplitude, mu_x, mu_y, sigma_x, sigma_y, orientation, background):
        def func(grid):
            return hp.Field(amplitude * self.make_gaussian(mu_x, mu_y, sigma_x, sigma_y, orientation)(grid) + background, grid) 
        return func

    def cost(self, theta):
        fit = self.satellite_spot(*theta)(self.data.grid)
        j = np.sum( (self.data - fit)**2)

        aspect_ratio = np.abs(theta[3] / theta[4])

        ##boundary condition for the aspect ratio
        if self.current_speckle == 0 or self.current_speckle == 2:
            if aspect_ratio  >= 1:
                j+= 1E3 * aspect_ratio **2
        elif self.current_speckle ==1 or self.current_speckle == 3:
            if aspect_ratio <= 1:
                j+= 1E3 * 1/aspect_ratio**2

        return j
    
    def fit(self,theta_est):
        fitting = minimize(self.cost,theta_est,options={'maxiter':self.maxiter})
        return fitting
    
    def estimate_centroid(self):
        M00 = np.sum(self.data)
        M10 = np.sum(self.data * self.data.grid.x)
        M01 = np.sum(self.data * self.data.grid.y)

        centroid = [M10/M00,M01/M00]
        return centroid
    
    def estimate_angle(self):
        M00 = np.sum(self.data)
        M10 = np.sum(self.data * self.data.grid.x)
        M01 = np.sum(self.data * self.data.grid.y)
        
        M20 = np.sum(self.data * self.data.grid.x**2)
        M02 = np.sum(self.data * self.data.grid.y**2)
        M11 = np.sum(self.data * self.data.grid.y * self.data.grid.x)

        mu10 = M10 / M00
        mu01 = M01 / M00
        mu20 = M20 / M00 - mu10**2
        mu02 = M02 / M00 - mu01**2
        mu11 = M11 / M00 - mu10 * mu01
        angle = (1/2 * np.arctan2(2 * mu11, mu20 - mu02))

        return angle

    def find_speckle(self, image,speckle_number):
        '''speckles are indexed from the top right going counter clockwise'''
        grating_freq = self.grating_freq
        grating_angle = self.grating_angle
        corners = np.array([[0, grating_freq * self.normalized_wavelength],[grating_freq * self.normalized_wavelength,0],[0, -grating_freq * self.normalized_wavelength],[-grating_freq * self.normalized_wavelength,0]])
        sizes = np.array([[8,20],[20,8],[8,20],[20,8]])
        #sizes = np.array([[16,25],[25,16],[16,25],[25,16]])

        rect = hp.make_rotated_aperture(hp.make_rectangular_aperture(size=sizes[speckle_number], center=corners[speckle_number]), np.deg2rad(-grating_angle))(image.grid)
        speckle_img = rect * image
        return speckle_img   

    def set_psf(self,psf):
        self.psf = psf  

    def find_speckle_angles2(self):

        speckle_angles = np.zeros(4)
        sig_x = [0.8,3.5,0.8,3.5]
        sig_y = [3.5,0.8,3.5,0.8]

        for i in range(4):
            img = self.find_speckle(self.psf,i)
            self.current_speckle = i
            self.set_measurement(img)

            sigma_x = sig_x[i]
            sigma_y = sig_y[i]
            #orientation = np.radians(28)

            orientation = self.estimate_angle()
            if self.current_speckle == 0 or self.current_speckle ==2:
                orientation = np.pi/2 - orientation
            elif self.current_speckle == 1 or self.current_speckle == 3:
                orientation = -orientation

            amplitude = self.data.max()
            centroid = self.estimate_centroid()
            mu_x = centroid[0]
            mu_y = centroid[1]
            background = 0

            theta_est = np.array([amplitude, mu_x, mu_y, sigma_x, sigma_y, orientation, background])
            fit = self.fit(theta_est) 
            speckle_angles[i] = np.degrees(fit.x[5])

        self.current_speckle = None
        speckle_angles = np.array(speckle_angles).T

        return speckle_angles
    
    def speckle_pairs(self, speckle_angles):
        pair02 = speckle_angles[0] - speckle_angles[2]
        pair13 = speckle_angles[1] - speckle_angles[3]
        return np.array([pair02,pair13])

    def calculate_command(self,speckle_angles):
        predicted_disp = self.control_mtx * np.matrix(speckle_angles).T
        predicted_disp = np.array(predicted_disp)
        return -predicted_disp

    def clear(self):
        self.psf = None
        self.data = None

    def set_control_mtx(self,matrix):
        self.control_mtx = matrix

    '''this is stuff specifically for working with the real calibration datacubes'''
    def window_field(self,data, center, width, height):
        indx = data.grid.closest_to(center)
        y_ind, x_ind = np.unravel_index(indx, data.shaped.shape)
        cutout = data.shaped[(y_ind-height//2):(y_ind + height//2), (x_ind-width//2):(x_ind+width//2)]
        sub_grid = hp.make_pupil_grid([width, height], [width * data.grid.delta[0], height * data.grid.delta[1]])
        return hp.Field(cutout.ravel(), sub_grid)
    
    def crop_image(self, image,extent=400,mask_diam=60): 
        #cutout a centered PSF
        img = image/image.max()

        img_subtracted = img >0.1
        center_of_intensity = np.array([sum(img_subtracted*img_subtracted.grid.x)/sum(img_subtracted),sum(img_subtracted*img_subtracted.grid.y)/sum(img_subtracted)])
        mask_ap = hp.make_circular_aperture(mask_diam,center_of_intensity)
        mask = mask_ap(img_subtracted.grid)
        mask = abs(mask - 1)
        masked_img = mask * img

        img = masked_img
        img = self.window_field(img,[center_of_intensity[0],center_of_intensity[1]],extent,extent)
        img /= img.max()
        
        bk = np.median(img)
        img -= bk
        img = hp.Field([x if x>0 else 0 for x in img],img.grid)

        return img

    def filter_image(self,img,low_freq = 0.01,high_freq=1):

        ff = hp.FourierFilter(img.grid, hp.make_circular_aperture(2 * np.pi * low_freq))
        filtered_img= np.real(ff.forward(img + 0j))
        img = img - filtered_img

        ff2 = hp.FourierFilter(img.grid, hp.make_circular_aperture(2 * np.pi * high_freq))
        filtered_img = np.real(ff2.forward(img + 0j))
        img = filtered_img
        filtered_subtracted = img 
        
        return filtered_subtracted


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
    CALIB = 3

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
        sv.add_element(DefSwitch(name="calibrate", _value=constants.SwitchState.OFF))
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
            name='knife_edge',
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        sv.add_element(DefSwitch(name="toggle", _value=constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_knife_edge) 

        sv = properties.SwitchVector(
            name='knife_edge_findzero',
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        sv.add_element(DefSwitch(name="request", _value=constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_knife_edge_findzero) 

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
            use_hcipy=True,
            indi_client=self.client
        )
        
        self._state = States.IDLE

        #self._loop_counter = 0
        self._n_avg = 1
        self._gain = 0.5
        self._command = 0
        self._control_mtx = np.array([0.21178766, 0.19275196])
        self._extent = 400
        self.delta_1 = 0
        self.delta_2 = 0
        self._offset = 0
        self._lab = False
        self._knife_edge = False
        self._knife_edge_zero1 = 26.78175714
        self._knife_edge_zero2 = 26.544759645

        if self.client['adctrack.deltaADC1.current'] != 0:
            self.set_command(0,0)
            self.send_command()

        self.update_wavelength()
        self.ADC = AdcFitter(wavelength=self._center_wavelength)
        self.ADC.set_control_mtx(self._control_mtx)

        self.properties['fsm']['state'] = StateCodes.READY.name
        self.update_property(self.properties['fsm'])

    def handle_state(self, existing_property, new_message):        
        target_list = ['idle', 'adcLoop', 'oneshot','calibrate']
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
                        #self._command = 0
                        self.log.debug('State changed to idle')                    
                    elif key == 'adcLoop':
                        self._state = States.CLOSED_LOOP
                        self.update_wavelength()
                        self.properties['fsm']['state'] = StateCodes.OPERATING.name
                        self.log.debug('State changed to closed-loop')
                    elif key == 'oneshot':
                        self._state = States.ONESHOT
                        self.update_wavelength()
                        self.properties['fsm']['state'] = StateCodes.OPERATING.name
                        self.log.debug('State changed to oneshot')
                    elif key == 'calibrate':
                        self._state = States.CALIB
                        self.update_wavelength()
                        self.properties['fsm']['state'] = StateCodes.OPERATING.name
                        self.log.debug('State changed to calibration')

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


    def handle_knife_edge(self,existing_property, new_message):
        if 'toggle' in new_message and new_message['toggle'] is constants.SwitchState.ON:
            self.log.debug('changing knife edge mode')
            existing_property['toggle'] = constants.SwitchState.ON
            self._knife_edge = True
        else:
            self.log.debug('exiting knife edge mode')
            existing_property['toggle'] = constants.SwitchState.OFF
            self._knife_edge = False

        self.update_property(existing_property)

    def handle_knife_edge_findzero(self,existing_property, new_message):
        if 'request' in new_message and new_message['request'] is constants.SwitchState.ON:
            self.log.debug('finding convergence point for use with knife edge ')
            existing_property['request'] = constants.SwitchState.OFF
            
            img = self.camera.grab_stack(self._n_avg)
            transpose = img.shaped.T 
            img = transpose.ravel()

            if self._lab == False:
                img = self.ADC.filter_image(img)

            angles = self.ADC.find_speckle_angles2()
            self._knife_edge_zero1 = angles[1]
            self._knife_edge_zero2 = angles[2]

        self.log.debug(f'zero points for bottom speckles changed to {self._knife_edge_zero1} and {self._knife_edge_zero2}')
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
        
    def update_wavelength(self):
        if self.client['fwsci1.filterName.i'] == constants.SwitchState.ON:
            self._center_wavelength = 762E-9
            self._extent = 400
        elif self.client['fwsci1.filterName.z'] == constants.SwitchState.ON:
            self._center_wavelength = 908E-9
            self._extent = 480
        else: 
            self._center_wavelength = 656E-9
            self._extent = 400
        self.log.debug(f'using center wavelength {self._center_wavelength*1E9} nm')

    def transition_to_idle(self):
        #self._command = 0
        self.properties['state']['oneshot'] = constants.SwitchState.OFF
        self.properties['state']['adcLoop'] = constants.SwitchState.OFF
        self.properties['state']['calibrate'] = constants.SwitchState.OFF
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
            img = self.camera.grab_stack(self._n_avg)
            transpose = img.shaped.T 
            img = transpose.ravel()

            if self._lab == False:
                img = self.ADC.filter_image(img)

            img = self.ADC.crop_image(img,extent=self._extent)
            self.ADC.set_psf(img)
            
            if self._knife_edge:
                angles = self.ADC.find_speckle_angles2()
                bottom_speckle_angles = np.array([angles[1] - self._knife_edge_zero1 ,angles[2] - self._knife_edge_zero2])
                self.log.debug(f'angle offsets: {angles}')
                error = self.ADC.calculate_command(bottom_speckle_angles)
            else:
                angles = self.ADC.find_speckle_angles2()
                pair_angles = self.ADC.speckle_pairs(angles)
                self.log.debug(f'angle offsets: {angles}')
                error = self.ADC.calculate_command(pair_angles)

            self.log.debug(f'measured error: {error}')
            #self._command = np.squeeze(self._command + -self._gain * error)
            
            if np.abs(self._command) < 10: #setting a threshold so the prisms don't do anything crazy     
                self.add_command(error * self._gain,0)
                self.send_command()
                self.log.debug(f'ADC command sent: {self._command}')
            else: self.log.info(f'ADC command {self._command} exceeds acceptable threshold and was not sent')

        elif self._state == States.ONESHOT:
            img = self.camera.grab_stack(self._n_avg)
            transpose = img.shaped.T 
            img = transpose.ravel()

            if self._lab == False:
                img = self.ADC.filter_image(img)

            img = self.ADC.crop_image(img,extent=self._extent)
            self.ADC.set_psf(img)
            #center_of_intensity = np.array([sum(img*img.grid.x)/sum(img),sum(img*img.grid.y)/sum(img)])
            #self.log.info(f'center of intensity: {center_of_intensity}')
            
            if self._knife_edge:
                angles = self.ADC.find_speckle_angles2()
                bottom_speckle_angles = np.array([angles[1],angles[2]])
                self.log.debug(f'angle offsets: {angles}')
                self._command = np.squeeze(self.ADC.calculate_command(bottom_speckle_angles))
            else:
                angles = self.ADC.find_speckle_angles2()
                pair_angles = self.ADC.speckle_pairs(angles)
                self.log.debug(f'angle offsets: {angles}')
                self._command = np.squeeze(self.ADC.calculate_command(pair_angles))

            self.log.info(f'One-shot ADC correction calculated a command of: {self._command}')

            if np.abs(self._command) < 5: #setting a threshold so the prisms don't do anything crazy     
                self.add_command(self._command,0)
                self.send_command()
                self.log.debug(f'ADC command sent: {self._command}')
            else: 
                self.log.info(f'ADC command {self._command} exceeds acceptable threshold and was not sent')            

            self.log.info('transitioning to idle')
            self.transition_to_idle()
            self._command = 0
            self.log.info('successfully transitioned to idle')

        elif self._state == States.CALIB:
            sweep_angles = np.linspace(-3,3,26)
            diff_pointing_pairs = np.zeros((len(sweep_angles),2)) 

            for i, orientation in enumerate(sweep_angles):
                self.log.debug(f'Step {i:d}')
                self.set_command(orientation, 0)
                self.send_command()

                img = self.camera.grab_stack(self._n_avg)
                transpose = img.shaped.T 
                img = transpose.ravel()

                if self._lab == False:
                    img = self.ADC.filter_image(img)

                img = self.ADC.crop_image(img,extent=self._extent)
                self.ADC.set_psf(img)
                
                angles = self.ADC.find_speckle_angles2()
                pointing_pair = self.ADC.speckle_pairs(angles)
                diff_pointing_pairs[i,] = pointing_pair

            self.set_command(0,0)
            self.send_command()

            a1 = np.zeros(2)
            b1 = np.zeros(2)

            for j in range(2):
                b1[j] , a1[j] = np.polyfit(sweep_angles,diff_pointing_pairs[:,j],deg=1)

            response = np.matrix([b1])
            self.log.debug(f'response matrix: {response}')
            
            if np.isnan(np.sum(response)):
                self.log.info(f'calibration failed, measured response is NaN')
                self.transition_to_idle()
            else:
                new_control_mtx = np.linalg.pinv(response)

                self._control_mtx = new_control_mtx.T
                self.ADC.set_control_mtx(self._control_mtx)
                self.log.info(f'calibration updated control matrix to: {self._control_mtx}')

                self.properties['ctrl_mtx']['m00'] = self._control_mtx[0,0]
                self.properties['ctrl_mtx']['m01'] = self._control_mtx[0,1]
                self.update_property(self.properties['ctrl_mtx'])
                
                self.transition_to_idle()






