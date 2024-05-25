import sys
import logging
from enum import Enum
import time
import numpy as np

import xconf

from magaox.indi.device import XDevice, BaseConfig
from magaox.camera import XCam
from magaox.deformable_mirror import XDeformableMirror
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

@xconf.config
class FileConfig:
    """
    """
    reference_pupil_name : str = xconf.field(help="File path to the pupil reference image.")
    reference_act_name : str = xconf.field(help="File path to the actuator reference image.")
    reference_probe_name : str = xconf.field(help="File path to the reference probe pattern.")

@xconf.config
class PupilCorAlignConfig(BaseConfig):
    """Automatic coronagraph alignment assistant
    """
    camera : CameraConfig = xconf.field(help="Camera to use")
    files : FileConfig = xconf.field(help="files to use")
    sleep_interval_sec : float = xconf.field(default=0.25, help="Sleep interval between loop() calls")

class States(Enum):
    IDLE = 0
    CLOSED_LOOP = 1
    REFERENCE = 2

class PupilCorAlign(XDevice):
    config : PupilCorAlignConfig
    
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
        sv.add_element(DefSwitch(name="closedloop", _value=constants.SwitchState.OFF))
        sv.add_element(DefSwitch(name="reference", _value=constants.SwitchState.OFF))
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

        self.log.info("Found camera: {:s}".format(self.config.camera.shmim))
        
        self.camera = XCam(self.config.camera.shmim, pixel_size=1.0, use_hcipy=True)
        
        self.ncpc_act_grid = make_pupil_grid(34, 34/30.0 * np.array([1.0, np.sqrt(2)]))
        self.ncpc_dm = XDeformableMirror(dm='dmncpc', channel=8)

        self._state = States.IDLE

        self._loop_counter = 0
        self._ref_x = 0
        self._ref_y = 0
        self._n_avg = 1
        self.properties['fsm']['state'] = StateCodes.READY.name
        self.update_property(self.properties['fsm'])

        self.load_reference_files()

    def load_reference_files(self):
        #self.config.files.shmim
        self._probe_pattern = 0
        self._reference_probe_image = 0
        self._reference_pupil_image =0

    def handle_n_avg(self, existing_property, new_message):
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
            self._n_avg = int(new_message['target'])
        self.update_property(existing_property)

    def handle_state(self, existing_property, new_message):           
        target_list = ['idle', 'closedloop', 'reference']
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
                    elif key == 'closedloop':
                        self._state = States.PSF
                        self.properties['fsm']['state'] = StateCodes.OPERATING.name
                    elif key == 'reference':
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

    def measure_pupil_stop_shift(self, image):
        valid_magaox_state = True

        if valid_magaox_state:
            return np.array([0.0, 0.0])
        else:
            # Do nothing if in an unspecified position.
            return np.array([0.0, 0.0])
    
    def measure_psf_position(self, image):
        return np.array([0.0, 0.0])

    def transition_to_idle(self):
        self.properties['state']['closedloop'] = constants.SwitchState.OFF
        self.properties['state']['reference'] = constants.SwitchState.OFF
        self.properties['state']['idle'] = constants.SwitchState.ON
        self.update_property(self.properties['state'])
        self._state = States.IDLE            

    def loop(self):
        if self._state == States.CLOSED_LOOP:
            image = self.camera.grab_stack(self._n_avg)
            center_error = self.measure_pupil_stop_shift(image)
            print(center_error)
            self._loop_counter += 1

            # Send new values to indi server.
            self.properties['measurement']['x'] = center_error[0]
            self.properties['measurement']['y'] = center_error[1]
            self.properties['measurement']['counter'] = self._loop_counter
            self.update_property(self.properties['measurement'])

        elif self._state == States.REFERENCE:
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
