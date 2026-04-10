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

import os
import datetime
import hcipy as hp
from astropy.io import fits
from scipy.optimize import minimize

from camtipSR.utils import *

@xconf.config
class CameraConfig:
    """
    """
    shmim : str = xconf.field(help="Name of the camera device (specifically, the associated shmim, if different)")
    dark_shmim : str = xconf.field(help="Name of the dark frame shmim associated with this camera device")
    #TODO: Does camtip have a dark frame?

@xconf.config
class camtipSRConfig(BaseConfig):
    """ Configure  """
    # if want default need to be a cam config obj
    camera : CameraConfig = xconf.field(help="Camera to use")

class States(Enum):
    IDLE = 0
    CLOSED_LOOP = 1
    ONESHOT = 2

class camtipSR(XDevice):
    config: camtipSRConfig

    def setup(self):
        # Finite State Machine (FSM) current state (I think)
        fsm = properties.TextVector(name='fsm')
        fsm.add_element(DefText(name='state', _value=StateCodes.INITIALIZED.name))
        self.add_property(fsm)

        # init INDI PROPERTY: loop state
        sv = properties.SwitchVector(
            name='state',
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        sv.add_element(DefSwitch(name="idle", _value=constants.SwitchState.ON))
        sv.add_element(DefSwitch(name="SRLoop", _value=constants.SwitchState.OFF))
        sv.add_element(DefSwitch(name="oneshot", _value=constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_state)

        # init INDI PROPERTY: n_avg
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

        # init INDI PROPERTY: n_avg
        nv = properties.NumberVector(name='n_jitter')
        nv.add_element(DefNumber(
            name='current', label='Number of frames', format='%i',
            min=1, max=150, step=1, _value=1
        ))
        nv.add_element(DefNumber(
            name='target', label='Number of frames', format='%i',
            min=1, max=150, step=1, _value=1
        ))
        self.add_property(nv, callback=self.handle_n_jitter)

        # init INDI PROPERTY: SR_est gaussian width method
        nv = properties.NumberVector(name='SR_est')
        nv.add_element(DefNumber(
            name='current', label='SR_est', format='%i',
            min=0, max=1, step=0.01, _value=1
        ))
        self.add_property(nv)

        # init INDI PROPERTY: SR_EE encircled energy method
        nv = properties.NumberVector(name='SR_EE')
        nv.add_element(DefNumber(
            name='current', label='SR_EE', format='%i',
            min=0, max=1, step=0.01, _value=1
        ))
        self.add_property(nv)

        # init INDI PROPERTY: jitter RMS x
        nv = properties.NumberVector(name='jitter_RMS')
        nv.add_element(DefNumber(
            name='current', label='jitter_RMS', format='%i',
            min=0, max=100, step=0.1, _value=0.0
        ))
        self.add_property(nv)

        # init INDI PROPERTY: jitter RMS x
        nv = properties.NumberVector(name='jitter_x')
        nv.add_element(DefNumber(
            name='current', label='jitter_x', format='%i',
            min=-20, max=20, step=0.1, _value=0.0
        ))
        self.add_property(nv)

         # init INDI PROPERTY: jitter rms y
        nv = properties.NumberVector(name='jitter_y')
        nv.add_element(DefNumber(
            name='current', label='jitter_y', format='%i',
            min=-20, max=20, step=0.1, _value=0.0
        ))
        self.add_property(nv)

        # get INDI PROPERTIES

        #TODO: Get camwfs to not crash - hiding for now
        # Modulator for determining lab calibration file
        #self.client.get_properties_and_wait(['modwfs'])
        self.client.get_properties(['modwfs'])
        time.sleep(10.0) # this takes a long time to load
        self.modRadius = self.client['modwfs.modRadius.current']
        self.log.info(f"Mod radius is {self.modRadius}")

        # find the camera
        self.log.info("Found camera: {:s}".format(self.config.camera.shmim))
        self.camera = XCam(
            self.config.camera.shmim,
            pixel_size=6.0/21.0, #TODO: I need to redo that plate scale methinks
            use_hcipy=True,
            indi_client=self.client,
        )

        # Starting values for states
        self._state = States.IDLE
        self._dark = False
        self._n_avg = 1
        self._n_jitter = 100
        self._SR_est = 0.0
        self._SR_est_list = []
        self._SR_EE = 0.0
        self._SR_EE_list = []
        self._x0_list = []
        self._y0_list = []
        self._jitter_RMS = 0.0
        self._jitter_x = 0.0
        self._jitter_y = 0.0
        self.data_directory = '/opt/MagAOX/rawimages/camtipSR/'
        self.lab_directory = '/opt/MagAOX/calib/camtipSR/'
        self.dark_directory = '/opt/MagAOX/calib/camtip-dark/'
        # SETUP: for the camtip fitter
        self.camFit = camtipFitter()
        # SETTING UP LAB
        # TODO: pick the correct lab file, rn it's on vibes
        if self.modRadius == 3:
            self.labf = 'lab_2000_3ld_ND2.fits'
        elif self.modRadius == 2:
            self.labf = 'lab_3000_2ld_ND2.fits'
        else:
            self.log.exception(" Not a calibrated mod radius, applying a bogus lab file!")
            self.labf = 'lab_2000_3ld_ND2.fits'
        self.camFit.setup_lab(self.lab_directory, self.labf)
        #lab_fit = [6.776e+00, 3.333e+01, 2.870e+00, -4.846e-01, 4.571e+00]
        #self.camFit.set_lab(lab_fit)
        self.properties['fsm']['state'] = StateCodes.READY.name
        self.update_property(self.properties['fsm'])

    def handle_state(self, existing_property, new_message):
        target_list = ['idle', 'SRLoop', 'oneshot']
        current_state = None
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
                        self._command = 0
                        self.log.debug('State changed to idle')
                    elif key == 'SRLoop':
                        self._state = States.CLOSED_LOOP
                        #self.modRadius = self.client['modwfs.modRadius.current'] # TODO: uncomment when modwfs is correct
                        self.properties['fsm']['state'] = StateCodes.OPERATING.name
                        self.log.debug('State changed to continuous')
                    elif key == 'oneshot':
                        self._state = States.ONESHOT
                        #self.modRadius = self.client['modwfs.modRadius.current'] # TODO: uncomment when modwfs is correct
                        self.properties['fsm']['state'] = StateCodes.OPERATING.name
                        self.log.debug('State changed to oneshot')

            self.update_property(existing_property)
            self.update_property(self.properties['fsm'])

    def handle_n_avg(self, existing_property, new_message):
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
            self._n_avg = int(new_message['target'])
        self.update_property(existing_property)

    def handle_n_jitter(self, existing_property, new_message):
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
            self._n_jitter = int(new_message['target'])
        self.update_property(existing_property)

    def transition_to_idle(self):
        self._command = 0
        self.log.info(f"Transitioning IDLE")
        self.properties['state']['oneshot'] = constants.SwitchState.OFF
        self.properties['state']['SRLoop'] = constants.SwitchState.OFF
        self.properties['state']['idle'] = constants.SwitchState.ON
        self.update_property(self.properties['state'])
        self._state = States.IDLE

    def fit_SR_gauss(self, img):
        # Set the data in the camtipFitter
        self.camFit.set_data(img) # background subtracted here
        #self.log.info(f"Image has been set. ")
        # Fit the data
        self.camFit.fit_data()
        #self.log.info(f"Image has been fit.")
        # Calculate the SR
        SR_est_new = self.camFit.calc_SR()
        self._SR_est_list.append(SR_est_new)
        #self.log.info(f"SR estimate: {self._SR_est}.")
        idx = np.min([len(self._SR_est_list), self._n_avg])
        self._SR_est = np.mean(self._SR_est_list[-idx:])
        # Set the SR
        self.properties['SR_est']['current'] = self._SR_est
        self.update_property(self.properties['SR_est'])
        #self.log.info(f"SR  has been set.")
        return

    def fit_SR_EE(self, img):
        # Set the data in the camtipFitter
        self.camFit.set_data(img) # background subtracted here
        #self.log.info(f"Image has been set. ")
        # Calculate the SR
        SR_EE_new = self.camFit.calc_SR_EE()
        self._SR_EE_list.append(SR_EE_new)
        #self.log.info(f"SR EE estimate: {self._SR_EE}.")
        idx = np.min([len(self._SR_EE_list), self._n_avg])
        self._SR_EE = np.mean(self._SR_EE_list[-idx:])
        # Set the SR
        self.properties['SR_EE']['current'] = self._SR_EE
        self.update_property(self.properties['SR_EE'])
        #self.log.info(f"SR EE has been set.")
        return

    def calc_jitter_RMS(self):
        # this needs to be run AFTER SR_EE
        self._x0_list.append(self.camFit.x0)
        self._y0_list.append(self.camFit.y0)
        # need to filter to make sure only the proper amount of the list is used in a calculation
        idx = np.min([len(self._x0_list), self._n_jitter])
        x0_all = self._x0_list[-idx:]
        y0_all = self._y0_list[-idx:]
        # x and Y are single pixels
        self._jitter_x = (x0_all[-1] - np.mean(x0_all)) * self.camFit.px_scale
        self._jitter_y = (y0_all[-1] - np.mean(y0_all)) * self.camFit.px_scale
        mean_sqr = np.mean(((x0_all - np.mean(x0_all))**2 + (y0_all - np.mean(y0_all))**2))
        self._jitter_RSM = np.sqrt(mean_sqr)
        # set the new jitter values
        self.properties['jitter_RMS']['current'] = self._jitter_RSM
        self.update_property(self.properties['jitter_RMS'])
        self.properties['jitter_x']['current'] = self._jitter_x
        self.update_property(self.properties['jitter_x'])
        self.properties['jitter_y']['current'] = self._jitter_y
        self.update_property(self.properties['jitter_y'])
        return

    def save_ex_img(self, img, name='testframe'):
        #TODO: need to make this directory if it doesn't already exist?
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%M%S")
        self.last_image_filename = f"camtipSR_{name}_{timestamp}.fits"
        outpath = f"{self.data_directory}/{self.last_image_filename}"
        #self.log.info(f"Saving to {outpath}")
        try:
            hdu = fits.PrimaryHDU(data=img)
            hdul = fits.HDUList([hdu])
            hdul.writeto(outpath, overwrite=True)
            self.log.info(f" File has been saved to {outpath}")
        except Exception:
            self.log.exception(f" Unable to save frame!")
        return

    def grab_img(self):
        # start of any loop will look for files
        try:
            img_t = self.camera.grab(subtract_dark=True)
            self._dark = True
        except RuntimeError:
            if self._dark == True:
                self.log.info("Error finidng files, likely the dark.")
            img_t = self.camera.grab(subtract_dark=False)
            self._dark = False
        img = img_t.shaped # vs. how it would be in a jupyter notebook
        # enforce cropping
        if img.shape == (512, 672):
            #self.log.info(f"Image size is {img.shape}, cropping...")
            cx, cy =  244, 414
            m = 64
            img = img[cx-m:cx+m, cy-m:cy+m]
        if img.shape != (128, 128):
            #self.log.error(f"Image size is {img.shape}, expected (128, 128), set proper ROI")
            self.transition_to_idle()
            return #TODO: need to figure out how to break the loop here
        return img

    def grab_stack(self, n):
        try:
            img_t = self.camera.grab_stack(n, subtract_dark=True)
            self._dark = True
        except RuntimeError as e:
            if self._dark == True:
                self.log.warning("Error finding files, likely the dark.")
            img_t = self.camera.grab_stack(n, subtract_dark=False)
            self._dark = False
        img = img_t.shaped # vs. how it would be in a jupyter notebook
        # enforce cropping
        if img.shape == (512, 672):
            #self.log.info(f"Image size is {img.shape}, cropping...")
            cx, cy =  244, 414
            m = 64
            img = img[cx-m:cx+m, cy-m:cy+m]
        if img.shape != (128, 128):
            #self.log.error(f"Image size is {img.shape}, expected (128, 128), set proper ROI")
            self.transition_to_idle()
            return #TODO: need to figure out how to break the loop here
        return img

    def loop(self):

        if self._state == States.CLOSED_LOOP:
            img = self.grab_img()
            ## CALL FIT FUNCTION
            self.fit_SR_gauss(img)
            self.fit_SR_EE(img)
            self.calc_jitter_RMS()
            # will then continue to loop

        elif self._state == States.ONESHOT:
            img = self.grab_img()
            ## CALL FIT FUNCTION
            self.fit_SR_gauss(img)
            self.fit_SR_EE(img)
            self.calc_jitter_RMS()
            # check to see the image
            self.save_ex_img(self.camFit.data_bg_sub)
            # will now exit out
            self.transition_to_idle()
            return
