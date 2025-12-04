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

from hcipy import *
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy import ndimage

class AdcFitter2:
    def __init__(self,wavelength=656E-9,bandwidth=100E-9,grating_angle=28,grating_freq=47,ncpc = False,snr_threshold=1.6,log=False,speckle_window=30):
            self.wavelength = wavelength
            self.bandwidth = bandwidth
            self.grating_angle = grating_angle
            self.grating_freq = grating_freq
            self.ncpc = ncpc 
            self.snr_threshold = snr_threshold
            self.normalized_wavelength = wavelength / 656E-9
            self.normalized_bandwidth = bandwidth / 656E-9
            self.control_matrix = np.array([0,0])
            self.log = log
            self.speckle_window = speckle_window

    def gauss(self,x,mu,sigma2):
        '''standard gaussian function'''
        return np.exp(-(x-mu)**2 / (2 * sigma2) )

    def isolate_gaussian(self,data):
        '''crop off the noisy tails of the gaussian'''
        new_data = data.copy()

        mid = np.argmax(new_data)
        first_derivative = np.gradient(new_data) 

        valley_indices = np.where((first_derivative[:-1] < 0) & (first_derivative[1:] > 0))[0] + 1

        left_valley = valley_indices[valley_indices < mid][-1] if any(valley_indices < mid) else None
        right_valley = valley_indices[valley_indices > mid][0] if any(valley_indices > mid) else None

        if left_valley is not None:
            new_data[0:left_valley] = 0

        if right_valley is not None:
            new_data[right_valley:len(new_data)] = 0

        return new_data

    def mu(self,data):
        '''crop off the noisy tails and find the mean of a single gaussian slice'''
        isolated = self.isolate_gaussian(data)
        x = np.arange(len(data))
        popt = curve_fit(self.gauss,x,isolated,p0=[np.argmax(data),2])

        mu = popt[0][0]
        sigma = popt[0][1]
        fwhm = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(sigma)
        return mu, sigma, fwhm
    
    def window_field(self,data,center,width,height):
        '''crop a smaller field out of a big field. data must be an hcipy field.'''
        indx = data.grid.closest_to(center)
        y_ind, x_ind = np.unravel_index(indx, data.shaped.shape)
        cutout = data.shaped[(y_ind-height//2):(y_ind + height//2), (x_ind-width//2):(x_ind+width//2)]
        sub_grid = make_pupil_grid([width, height], [width * data.grid.delta[0], height * data.grid.delta[1]])
        return Field(cutout.ravel(), sub_grid)
        

    def slice_speckle_angle(self,img,speckle_number,print_updates=False,positions=False):
        '''calculate the angle of a single speckle using the gaussian slicing method. image must be an hcipy field.
        returns the slope of the designated speckle in degrees.'''
    
        #locate the general area of the speckle
        extent=20
        ncpc_freq = 29

        if self.ncpc == False: #if we're not using the ncpc speckles, proceed as normal
            speckle_coords = np.array([[0, self.grating_freq * self.normalized_wavelength],[self.grating_freq * self.normalized_wavelength,0],[0, -self.grating_freq * self.normalized_wavelength],[-self.grating_freq * self.normalized_wavelength,0]])
            speckle_center = speckle_coords[speckle_number]
            rect = make_rotated_aperture(make_rectangular_aperture(size=(extent,extent), center=speckle_center), np.deg2rad(-self.grating_angle))(img.grid)
            #self.log.debug(f'speckle {speckle_number} is approximately centered at {speckle_center}')
        else: #if we are, there's a different rotation angle for speckles 1&3 and 2&4
            speckle_coords = np.array([[ncpc_freq * self.normalized_wavelength,0],[self.grating_freq * self.normalized_wavelength,0],[-ncpc_freq * self.normalized_wavelength,0],[-self.grating_freq * self.normalized_wavelength,0]])
            speckle_center = speckle_coords[speckle_number]
            if speckle_number==0 or speckle_number==2:
                rect = make_rectangular_aperture(size=(extent,extent), center=speckle_center)(img.grid)
            else:
                rect = make_rotated_aperture(make_rectangular_aperture(size=(extent,extent), center=speckle_center), np.deg2rad(-self.grating_angle))(img.grid)

        speckle_img = rect * img
        max_pixel = speckle_img.grid[np.argmax(speckle_img)]
        window_size=self.speckle_window
        new_img = self.window_field(img,[max_pixel[0],max_pixel[1]],window_size,window_size)

        #self.log.debug(f'calculated max pixel for speckle {speckle_number}: {max_pixel}')

        #new_img = self.window_field(img,[center_of_intensity[0],center_of_intensity[1]],window_size,window_size)
        
        shaped = new_img.shaped
        mus = np.zeros(shaped.shape[0])

        for i in range(shaped.shape[0]):
            if self.ncpc==True:
                sliced = shaped[:,i] 
            elif speckle_number % 2 != 0: #columns vs rows depending on which speckle it is
                sliced = shaped[:,i] 
            else: 
                sliced = shaped[i,:]

            zscore = (np.max(sliced) - np.mean(sliced))/np.std(sliced)

            if zscore > self.snr_threshold:
                s = self.isolate_gaussian(sliced)
                s /= s.max()
                try:
                    m, std, fwhm = self.mu(s)
                except:
                    self.log.debug(f'fitting error. could not fit speckle {speckle_number}')
                    return np.nan
                
                mus[i] = m
            else: mus[i] = -1

        #take the second derivative so we can isolate the nonlinear region
        deriv2 = np.gradient(np.gradient(mus))
        nonlin_region = np.atleast_1d(np.squeeze(np.where(np.abs(deriv2) >= 0.4)))

        if len(nonlin_region)==0:
            self.log.debug('edges of speckle outside speckle window, or snr threshold must be increased')

        #start from the middle of the mu vector and move outward to find the bounds of the linear region
        midpoint = int(np.floor(window_size / 2))

        if any(nonlin_region < midpoint):
            leftbound = nonlin_region[nonlin_region < midpoint][-1]
        elif deriv2[0] <= 0.4:
            leftbound = 0
        else: leftbound = None

        if any(nonlin_region > midpoint):
            rightbound = nonlin_region[nonlin_region > midpoint][0]
        elif deriv2[-1] <= 0.4:
            rightbound = 50
            #print('speckle falls off right (top) edge')
        else: rightbound = None        

        #find the centroid for the vector version
        if leftbound != None and rightbound != None:
            lin_region_center = (leftbound+rightbound)//2 #this is the center WRT the new window. to do this properly you need it in terms of the old window.
            corresponding_mu = mus[lin_region_center]
            relative_center = np.array([lin_region_center - self.speckle_window//2,corresponding_mu - self.speckle_window//2])
            if speckle_number % 2 != 0: #columns vs rows depending on which speckle it is
                centroid = np.array([max_pixel[0] + relative_center[0]*speckle_img.grid.delta[0],max_pixel[1] + relative_center[1] * speckle_img.grid.delta[1]])
            else:
                relative_center = np.flip(relative_center) 
                centroid = np.array([max_pixel[0] + relative_center[0]*speckle_img.grid.delta[0],max_pixel[1] + relative_center[1] * speckle_img.grid.delta[1]])

        #do a linear regression on the mus in the linear region. if there is no clear linear region, print an error if print_updates is enabled.
        if leftbound is not None and rightbound is not None:
            linear_region = mus[leftbound:rightbound]
            lin_x = np.arange(len(linear_region))

            m , b = np.polyfit(lin_x,linear_region,deg=1)
            angle = np.arctan(m)
            if print_updates ==True:
                self.log.debug(f'slope: {m:.2f}\ncorresponding angle: {angle:.2f} (rad) or {np.degrees(angle):.2f}°')

            if positions:
                return np.degrees(angle),centroid
            else:
                return np.degrees(angle) #returns the slope of the individual speckle in degrees.
        else: 
            if print_updates == True:
                self.log.debug(f'unable to fit speckle {speckle_number}')
            return np.nan

    def all_speckle_angles(self,img):
        '''returns a numpy vector containing the four speckle angles found in the image.
        need a way to incorporate the snr threshold??
        '''
        angles = np.zeros(4)
        for i in range(4):
            angles[i] = self.slice_speckle_angle(img,i)
        return angles

    def hpf(data,sigma):
        return Field((data.shaped - ndimage.gaussian_filter(data.shaped,sigma)).ravel(),data.grid)
    
    def crop_image(self, image,extent,mask_diam=60): 
        '''cuts out a centered PSF with the central core masked'''
        bk = np.median(image)
        image -= bk
        img = image/np.max(image)

        img_subtracted = img >0.03
        center_of_intensity = np.array([sum(img_subtracted*img_subtracted.grid.x)/sum(img_subtracted),sum(img_subtracted*img_subtracted.grid.y)/sum(img_subtracted)])
        mask_ap = make_circular_aperture(mask_diam,center_of_intensity)
        mask = mask_ap(img.grid)
        mask = abs(mask - 1)
        masked_img = mask * img

        img = masked_img
        img = self.window_field(img,[center_of_intensity[0],center_of_intensity[1]],extent,extent)


        img /= np.max(img)
        
        mask2 = make_circular_aperture(mask_diam,np.array([0,0]))(img.grid)
        mask2 = abs(mask2-1)
        img = mask2 * img
        img = Field([x if x>0 else 0 for x in img],img.grid)

        return img
    
    def filter_image(self,img,low_freq = 0.01,high_freq=1):

        ff = FourierFilter(img.grid, make_circular_aperture(2 * np.pi * low_freq))
        filtered_img= np.real(ff.forward(img + 0j))
        img = img - filtered_img

        ff2 = FourierFilter(img.grid, make_circular_aperture(2 * np.pi * high_freq))
        filtered_img = np.real(ff2.forward(img + 0j))
        img = filtered_img

        # binc, profile, std_profile, ncount = radial_profile(img,.25)
        # r_coordinates = img.grid.as_('polar').r
        # radial_map = np.interp(r_coordinates, binc, profile)

        filtered_subtracted = img #- radial_map
        
        return filtered_subtracted

    def speckle_pairs(self,speckle_angles):
        '''calculate the pair offset angles for each pair of speckles'''
        #when one of the speckle angles is nan, it just results in a command that is nan. we filter that out using the nanmean.
        diff13 = speckle_angles[0] - speckle_angles[2]
        diff24 = speckle_angles[1] - speckle_angles[3]
        pairs = np.array([diff13,diff24])
        return pairs

    def calculate_command(self,speckle_pairs):
        '''calculate the command that would be sent if the gain were one. this is a command, not a measurement, hence the sign.'''
        #again, what to do when the pairs are nan?
        predicted_disp = self.control_matrix * np.matrix(speckle_pairs).T
        predicted_disp = np.array(predicted_disp)
        return -predicted_disp

    #dispersion vector estimator
    def est_mag_dir(self,psf,logged=False):
        slopes = np.zeros(4)
        intercepts = np.zeros(4)
        points = []
        lines = np.zeros((4,3))

        for i in range(4):
            try:
                angle,point = self.slice_speckle_angle(psf,i,positions=True)
            except:
                print(f'fitting not successful for speckle {i}')
                return np.nan
            if i == 0 or i ==2:
                angle = 90 - angle

            slope = np.tan(np.radians(angle))
            slopes[i] = slope
            points.append(point)
            intercepts[i] = -slope * point[0] + point[1]

        for j in range(4):
            lines[j,0] = slopes[j]
            lines[j,1] = -1
            lines[j,2] = points[j][1] - slopes[j]*points[j][0]

        A = lines[:,0]
        B = lines[:,1]
        C = lines[:,2]
        norm = np.sqrt(A**2 + B**2)

        A /= norm
        B /= norm
        C /= norm

        M = np.stack((A.T,B.T),axis=1)
        C = -C

        (x, y), *_ = np.linalg.lstsq(M, C)
        if printed:
            print(f'least-squares solution: ({x},{y})')
        magnitude_guess = np.sqrt(x**2+y**2) 

        #added logic for quadrants
        if x > 0 and y > 0: #i
            orientation_guess = np.degrees(np.atan(y/x))
        elif x < 0 and y >0: #ii
            orientation_guess = 180+np.degrees(np.atan(y/x))
        elif x < 0 and y < 0: #iii
            orientation_guess = -np.degrees(np.atan(y/x))
        else: #iv
            orientation_guess = -180-np.degrees(np.atan(y/x))

        if logged:  
            self.log.debug(f'Estimated dispersion direction {orientation_guess}°')
            self.log.debug(f'Estimated dispersion magnitude {magnitude_guess}')
            
        return magnitude_guess,orientation_guess


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
        nv.add_element(DefNumber( 
            name='m00', label='m00', format='%.4f',
            min=-10.00, max=10.00, step=0.0001, _value=0.08902178
        ))
        nv.add_element(DefNumber( 
            name='m01', label='m01', format='%.4f',
            min=-10.00, max=10.00, step=0.0001, _value=-0.1929974
        ))
        self.add_property(nv, callback=self.handle_ctrl_mtx) 

        nv = properties.NumberVector(name='filterParams')
        nv.add_element(DefNumber( 
            name='hpf', label='hpf', format='%i',
            min=0, max=500, step=1, _value=8
        ))
        nv.add_element(DefNumber( 
            name='lpf', label='lpf', format='%i',
            min=0, max=10.00, step=1, _value=3
        ))
        self.add_property(nv, callback=self.handle_filter) 

        sv = properties.SwitchVector(
            name='labmode',
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        sv.add_element(DefSwitch(name="toggle", _value=constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_labmode) 

        sv = properties.SwitchVector(
            name='vectorize',
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        sv.add_element(DefSwitch(name="toggle", _value=constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_vectorize) 

        sv = properties.SwitchVector(
            name='knife_edge',
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        sv.add_element(DefSwitch(name="toggle", _value=constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_knife_edge) 

        sv = properties.SwitchVector(
            name='reset_deltaADCs',
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        sv.add_element(DefSwitch(name="request", _value=constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_reset) 

        self.client.get_properties('adctrack')
        self.client.get_properties('fwsci1')
        self.client.get_properties('fwfpm')

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
        self._gain = 0.3
        self._command = 0
        self._control_mtx = np.array([ 0.08902178, -0.1929974]) 
        self._extent = 400
        self.delta_1 = 0
        self.delta_2 = 0
        self._offset = 0
        self._mask_diam = 45
        self._lab = False
        self._knife_edge = False
        self._knife_edge_zero1 = 26.78175714 #need to re-calibrate these values
        self._knife_edge_zero2 = 26.544759645
        self._no_measurements = 1
        self._ke_top = True #orientation of the knife mask
        self._vectorize = False #defaults to not calculating dispersion orientation
        self._hpf_sigma = 8
        self._lpf_sigma = 3

        if self.client['adctrack.deltaADC1.current'] != 0:
            self.set_command(0,0)
            self.send_command()

        if self.client['fwsci1.filterName.i'] == constants.SwitchState.ON:
            self._center_wavelength = 762E-9
            self._extent = 512
        elif self.client['fwsci1.filterName.z'] == constants.SwitchState.ON:
            self._center_wavelength = 908E-9
            self._extent = 512
        else: 
            self._center_wavelength = 656E-9
            self._extent = 500

        self.ADC = AdcFitter2(wavelength=self._center_wavelength,log=self.log)
        self.log.debug(f'initial normalized wavelength value: {self.ADC.normalized_wavelength}')
        self.ADC.control_matrix = self._control_mtx

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
                        #self.update_wavelength()
                        self.properties['fsm']['state'] = StateCodes.OPERATING.name
                        self.log.debug('State changed to closed-loop')
                    elif key == 'oneshot':
                        self._state = States.ONESHOT
                        #self.update_wavelength()
                        self.properties['fsm']['state'] = StateCodes.OPERATING.name
                        self.log.debug('State changed to oneshot')
                    elif key == 'calibrate':
                        self._state = States.CALIB
                        #self.update_wavelength()
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

    def handle_vectorize(self,existing_property, new_message):
        if 'toggle' in new_message and new_message['toggle'] is constants.SwitchState.ON:
            self.log.debug('vector mode on')
            existing_property['toggle'] = constants.SwitchState.ON
            self._vectorize = True
        else:
            self.log.debug('vector mode off')
            existing_property['toggle'] = constants.SwitchState.OFF
            self._vectorize = False

        self.update_property(existing_property)

    def handle_knife_edge(self,existing_property, new_message):
        if 'toggle' in new_message and new_message['toggle'] is constants.SwitchState.ON:
            self.log.debug('changing into knife edge mode')
            existing_property['toggle'] = constants.SwitchState.ON
            self._knife_edge = True
            if self.client['fwfpm.filterName.knifemask'] == constants.SwitchState.ON:
                self._ke_top = False
            elif self.client['fwfpm.filterName.knifemaskZ'] == constants.SwitchState.ON:
                self._ke_top = True
        else:
            self.log.debug('exiting knife edge mode')
            existing_property['toggle'] = constants.SwitchState.OFF
            self._knife_edge = False

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
        
        self.ADC.control_matrix = self._control_mtx
        self.log.debug(f'control matrix changed to {self._control_mtx}')
        self.update_property(existing_property)


    def handle_filter(self, existing_property, new_message):
        old_matrix = self._control_mtx
        if 'hpf' in new_message and new_message['hpf'] != existing_property['hpf']:
            existing_property['hpf'] = new_message['hpf']
            self._hpf_sigma = new_message['hpf']

        if 'lpf' in new_message and new_message['lpf'] != existing_property['lpf']:
            existing_property['lpf'] = new_message['lpf']
            self._lpf_sigma = new_message['lpf']
        
        self.log.debug(f'filtering parameters changed to (high, low) = ({self._hpf_sigma},{self._lpf_sigma})')
        self.update_property(existing_property)
        
    def update_wavelength(self):
        if self.client['fwsci1.filterName.i'] == constants.SwitchState.ON:
            self._center_wavelength = 762E-9
            self._extent = 480
        elif self.client['fwsci1.filterName.z'] == constants.SwitchState.ON:
            self._center_wavelength = 908E-9
            self._extent = 512
            self.log.debug('filter in zprime')
        else: 
            self._center_wavelength = 656E-9
            self._extent = 480
        
        self.ADC.wavelength = self._center_wavelength
        self.ADC.normalized_wavelength = self.ADC.wavelength / 6565E-9
        self.log.debug(f'using center wavelength {self._center_wavelength*1E9} nm, ADC instance sees {self.ADC.wavelength} & {self.ADC.normalized_wavelength} normalized')

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

    def loop(self): #new loop function for gaussian fitter
            if self._state == States.CLOSED_LOOP:
                measurements = []
                error = 0
                
                for i in range(self._no_measurements):
                    img = self.camera.grab_stack(self._n_avg)
                    img = np.pad(img,pad_width=50, mode='constant', constant_values=0)
                    dim = np.sqrt(img.size)
                    extent = dim * 6/21
                    pgrid = make_pupil_grid(dim,extent)
                    img = Field(img.ravel(),pgrid)
                    img -= np.median(img) ############## not sure if this will help or hurt. gotta test it
                    img = self.ADC.crop_image(img,extent=self._extent,mask_diam=self._mask_diam)
                    img = self.ADC.filter_image(img)
                    
                    if self._knife_edge:
                        zps = np.array([ 25.47224616, -25.9758844,   25.82760786, -25.5740744 ]) #zero points for each speckle
                        if self._ke_top:
                            ctrl_mtx_top = np.matrix([[0.27260458, 0.69552136]])
                            s1 = self.ADC.slice_speckle_angle(img,0) - zps[0]
                            s4 = self.ADC.slice_speckle_angle(img,3) - zps[3]
                            command = -np.squeeze(ctrl_mtx_top @ np.array([s1,s4]))
                        else:
                            ctrl_mtx_bot = np.matrix([[-0.68087296, -0.36395656]])
                            s2 = self.ADC.slice_speckle_angle(img,1) - zps[1]
                            s3 = self.ADC.slice_speckle_angle(img,2) - zps[2]
                            command = -np.squeeze(ctrl_mtx_bot @ np.array([s2,s3]))
                    else:
                        angles = self.ADC.all_speckle_angles(img)
                        pairs = self.ADC.speckle_pairs(angles)
                        command = np.squeeze(self.ADC.calculate_command(pairs))

                        self.log.debug(f'measured speckle angles: {angles}')

                    self.log.debug(f'single error measurement: {-command}')
                    measurements.append(command)
                
                error = -np.nanmean(measurements)
                self.log.debug(f'mean error across {self._no_measurements} measurements: {-error}')

                if np.abs(error*self._gain) < 0.7: #setting a threshold so the prisms don't do anything crazy     
                    self.add_command(error * self._gain,0)
                    self.send_command()
                    self.log.info(f'delta command: {error * self._gain}')
                    self.log.info(f'total command: {self.delta_1}')
                else: self.log.info(f'ADC command {error} exceeds acceptable threshold and was not sent')

            elif self._state == States.ONESHOT:
                measurements = []
                for i in range(self._no_measurements):
                    img = self.camera.grab_stack(self._n_avg)
                    img = np.pad(img,pad_width=50, mode='constant', constant_values=0)
                    self.log.debug(f'extent: {self._extent}')
                    dim = np.sqrt(img.size)
                    self.log.debug(f'camera ROI square with dim {dim} pixels')
                    extent = dim * 6/21
                    pgrid = make_pupil_grid(dim,extent)
                    img = Field(img.ravel(),pgrid)
                    img -= np.median(img)

                    write_field(img,'/tmp/adcDebug.fits')

                    ################### FAKE CAMERA IMAGE ######################
                    #img = read_field('/data/users/twitchell/full_img.fits')
                    #self.log.debug('note that a real picture is not being taken! a loaded image is being used')

                    img = self.hpf(img,20)
                    img = self.ADC.crop_image(img,extent=self._extent,mask_diam=self._mask_diam)
                    #img = self.ADC.filter_image(img)

                    #if we want to find the orientation as well
                    if self._vectorize:
                        mag,ang = self.ADC.est_mag_dir(img)
                        self.log.debug(f'estimated dispersion direction {ang}°')
                        #### then calculate the way you'd rotate the adcs, averaged over the number of measurements specified in indi

                    ## if we're in knife edge mode
                    if self._knife_edge:
                        zps = np.array([ 26.06322496, -24.34992527,  25.44309035, -26.44816027]) #zero points for each speckle
                        if self._ke_top:
                            ctrl_mtx_top = np.matrix([[0.23973406, 0.41542301]])
                            s1 = self.ADC.slice_speckle_angle(img,0) - zps[0]
                            s4 = self.ADC.slice_speckle_angle(img,3) - zps[3]
                            command = -np.squeeze(ctrl_mtx_top @ np.array([s1,s4]))
                        else:
                            ctrl_mtx_bot = np.matrix([[-0.39625451, -0.19269755]])
                            s2 = self.ADC.slice_speckle_angle(img,1) - zps[1]
                            s3 = self.ADC.slice_speckle_angle(img,2) - zps[2]
                            command = -np.squeeze(ctrl_mtx_bot @ np.array([s2,s3]))

                    else:
                        angles = self.ADC.all_speckle_angles(img)
                        pairs = self.ADC.speckle_pairs(angles)
                        command = np.squeeze(self.ADC.calculate_command(pairs))

                        self.log.debug(f'measured speckle angles: {angles}')

                    self.log.debug(f'single error measurement: {-command}')
                    measurements.append(command)

                error = np.nanmean(measurements)

                self.log.info(f'mean error across {self._no_measurements} measurements: {error} (command calculated but not sent)')          
                self.log.info('transitioning to idle')
                self.transition_to_idle()
                self.log.info('successfully transitioned to idle')

            elif self._state == States.CALIB:
                sweep_angles = np.linspace(-3,3,26)
                diff_pointing_pairs = np.zeros((len(sweep_angles),2)) 

                if self._knife_edge == False:
                    self.log.debug(f'calibrating in regular mode')
                    for i, orientation in enumerate(sweep_angles):
                        self.log.debug(f'Step {i:d}')
                        self.set_command(orientation, 0)
                        self.send_command()

                        img = self.camera.grab_stack(self._n_avg)
                        transpose = Field(img.shaped.T.ravel(),img.grid)
                        img = transpose

                        img = self.ADC.crop_image(img,extent=self._extent,mask_diam=self._mask_diam)
                        img = self.ADC.filter_image(img)
                        
                        angles = self.ADC.all_speckle_angles()
                        pointing_pair = self.ADC.speckle_pairs(angles)
                        diff_pointing_pairs[i,] = pointing_pair
                else:
                    self.log.debug(f'calibrating in knife-edge mode')
                    pass 

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
                self.ADC.control_matrix = self._control_mtx
                self.log.info(f'calibration updated control matrix to: {self._control_mtx}')

                self.properties['ctrl_mtx']['m00'] = self._control_mtx[0,0]
                self.properties['ctrl_mtx']['m01'] = self._control_mtx[0,1]
                self.update_property(self.properties['ctrl_mtx'])
                
                self.transition_to_idle()





