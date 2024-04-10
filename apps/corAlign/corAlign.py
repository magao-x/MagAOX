import sys
import logging
from enum import Enum
import time
import numpy as np

import xconf

from magaox.indi.device import XDevice, BaseConfig
from magaox.camera import XCam
from magaox.constants import StateCodes

from hcipy import util, make_circular_aperture, Field

from skimage import feature


from purepyindi2 import device, properties, constants
from purepyindi2.messages import DefNumber, DefSwitch, DefLight, DefText

from utils import *

@xconf.config
class CameraConfig:
    """
    """
    shmim : str = xconf.field(help="Name of the camera device (specifically, the associated shmim, if different)")
    dark_shmim : str = xconf.field(help="Name of the dark frame shmim associated with this camera device")

@xconf.config
class CorAlignConfig(BaseConfig):
    """Automatic coronagraph alignment assistant
    """
    camera : CameraConfig = xconf.field(help="Camera to use")
    sleep_interval_sec : float = xconf.field(default=0.25, help="Sleep interval between loop() calls")

class States(Enum):
    IDLE = 0
    CLOSED_LOOP = 1
    PSF = 2

class corAlign(XDevice):
    config : CorAlignConfig
    
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
        sv.add_element(DefSwitch(name="fpm", _value=constants.SwitchState.OFF))
        sv.add_element(DefSwitch(name="psf", _value=constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_state)

        nv = properties.NumberVector(name='measurement')
        nv.add_element(DefNumber(
            name='counter', label='Loop counter', format='%i',
            min=0, max=2**32-1, step=1, _value=0
        ))
        nv.add_element(DefNumber(
            name='x', label='X shift', format='%3.3f',
            min=-150.0, max=150.0, step=0.01, _value=0
        ))
        nv.add_element(DefNumber(
            name='y', label='Y shift', format='%3.3f',
            min=-150.0, max=150.0, step=0.01, _value=0
        ))
        self.add_property(nv)

        nv = properties.NumberVector(name='ref_x')
        nv.add_element(DefNumber(
            name='current', label='X ref', format='%3.3f',
            min=-150.0, max=150.0, step=0.01, _value=0
        ))
        nv.add_element(DefNumber(
            name='target', label='X ref', format='%3.3f',
            min=-150.0, max=150.0, step=0.01, _value=0
        ))
        self.add_property(nv, callback=self.handle_ref_x)

        nv = properties.NumberVector(name='ref_y')
        nv.add_element(DefNumber(
            name='current', label='Y ref', format='%3.3f',
            min=-150.0, max=150.0, step=0.01, _value=0
        ))
        nv.add_element(DefNumber(
            name='target', label='Y ref', format='%3.3f',
            min=-150.0, max=150.0, step=0.01, _value=0
        ))
        self.add_property(nv, callback=self.handle_ref_y)

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

        self.client.get_properties('fwfpm')
        #self.client.register_callback(func, device_name='fwfpm')

        self.log.info("Found camera: {:s}".format(self.config.camera.shmim))
        self.camera = XCam(self.config.camera.shmim, pixel_size=6.0/21.0, use_hcipy=True)
        self._state = States.IDLE

        self._loop_counter = 0
        self._ref_x = 0
        self._ref_y = 0
        self._n_avg = 1
        self.properties['fsm']['state'] = StateCodes.READY.name
        self.update_property(self.properties['fsm'])

    def handle_n_avg(self, existing_property, new_message):
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
            self._n_avg = int(new_message['target'])
        self.update_property(existing_property)

    def handle_state(self, existing_property, new_message):           
        target_list = ['idle', 'psf', 'fpm']
        for key in target_list:
            if existing_property[key] == constants.SwitchState.ON:
                current_state = key
        
        if current_state not in new_message:

            for key in target_list:
                existing_property[key] = constants.SwitchState.OFF  # Turn everything off
                if key in new_message:
                    existing_property[key] = new_message[key]

                    if key == 'idle':
                        self._state = States.IDLE
                        self.properties['fsm']['state'] = StateCodes.READY.name                    
                    elif key == 'psf':
                        self._state = States.PSF
                        self.properties['fsm']['state'] = StateCodes.OPERATING.name
                    elif key == 'fpm':
                        self._state = States.CLOSED_LOOP
                        self.properties['fsm']['state'] = StateCodes.OPERATING.name

            self.update_property(existing_property)
            self.update_property(self.properties['fsm'])

    def handle_closed_loop(self, existing_property, new_message):
        if 'toggle' in new_message and new_message['toggle'] is constants.SwitchState.ON:
            print("switch to closed-loop")
            existing_property['toggle'] = constants.SwitchState.ON
            self._state = States.CLOSED_LOOP
            self.properties['fsm']['state'] = StateCodes.OPERATING.name

        if 'toggle' in new_message and new_message['toggle'] is constants.SwitchState.OFF:
            print("switch to IDLE")
            existing_property['toggle'] = constants.SwitchState.OFF
            self._state = States.IDLE
            self.properties['fsm']['state'] = StateCodes.READY.name

        self.update_property(existing_property)
        self.update_property(self.properties['fsm'])

    def handle_ref_x(self, existing_property, new_message):
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
            self._ref_x = new_message['target']
        self.update_property(existing_property)

    def handle_ref_y(self, existing_property, new_message):
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
            self._ref_y = new_message['target']
        self.update_property(existing_property)

    def measure_fpm_mask_shift(self, image):
        if self.client['fwfpm.filterName.lyotsm'] == constants.SwitchState.ON:
            return measure_center_position(image)
        elif self.client['fwfpm.filterName.lyotlg'] == constants.SwitchState.ON:
            print("Is Lyot large mask")
            return measure_center_position(image)
        elif self.client['fwfpm.filterName.knifemask'] == constants.SwitchState.ON:
            return knife_edge_dist(image, theta=0, mask_diameter=200, threshold=0.5)
        elif self.client['fwfpm.filterName.spare'] == constants.SwitchState.ON:
            return np.array([0.0, 0.0]) # Not implemented yet.
        else:
            # Do nothing if in an unspecified position.
            return np.array([0.0, 0.0])
    
    def measure_psf_position(self, image):
        return np.array([0.0, 0.0])

    def transition_to_idle(self):
        self.properties['state']['psf'] = constants.SwitchState.OFF
        self.properties['state']['fpm'] = constants.SwitchState.OFF
        self.properties['state']['idle'] = constants.SwitchState.ON
        self.update_property(self.properties['state'])
        self._state = States.IDLE            

    def loop(self):
        if self._state == States.CLOSED_LOOP:
            image = self.camera.grab_stack(self._n_avg)
            center_error = self.measure_fpm_mask_shift(image)
            print(center_error)
            self._loop_counter += 1

            # Send new values to indi server.
            self.properties['measurement']['x'] = center_error[0]
            self.properties['measurement']['y'] = center_error[1]
            self.properties['measurement']['counter'] = self._loop_counter
            self.update_property(self.properties['measurement'])

        elif self._state == States.PSF:
            image = self.camera.grab_stack(self._n_avg)
            center_reference = self.measure_psf_position(image)

            # Send new values to indi server.
            self.properties['ref_x']['current'] = center_reference[0]
            self.properties['ref_x']['target'] = center_reference[0]
            self.update_property(self.properties['ref_x'])

            self.properties['ref_y']['current'] = center_reference[1]
            self.properties['ref_y']['target'] = center_reference[1]
            self.update_property(self.properties['ref_y'])

            self.transition_to_idle()
