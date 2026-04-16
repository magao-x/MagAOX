import logging
from enum import Enum
import time
import numpy as np
import hcipy as hp

import xconf

from magaox.indi.device import XDevice, BaseConfig
from magaox.camera import XCam
from magaox.deformable_mirror import XDeformableMirror
from magaox.constants import StateCodes
from magaox.state_manager import XStateMachine

from purepyindi2 import device, properties, constants
from purepyindi2.messages import DefNumber, DefSwitch, DefLight, DefText

from .utils import *

@xconf.config
class CameraConfig:
    """
    """
    shmim : str = xconf.field(help="Name of the camera device (specifically, the associated shmim, if different).")
    dark_shmim : str = xconf.field(help="Name of the dark frame shmim associated with this camera device.")
    stagepos : float = xconf.field(help="Stage position for pupil control.")

@xconf.config
class CalibrationConfig:
    """
    """
    path : str = xconf.field(help="Path to the calibration files for the pupil alignment.")
    superuser : bool = xconf.field(default=False, help="Whether to enable super user options for calibration.")
    fwpupil_modes : list[str] = xconf.field(help="List of calibrated fwpupil masks.")
    fwlyot_modes : list[str] = xconf.field(help="List of calibrated fwlyot masks.")

@xconf.config
class DMConfig:
    """
    """
    shmim : str = xconf.field(help="Name of the DM device (specifically, the associate shmim, if different).")
    channel : int = xconf.field(help="The DM channel number.")

@xconf.config
class pupilCorAlignConfig(BaseConfig):
    """Automatic coronagraph alignment assistant
    """
    camera : CameraConfig = xconf.field(help="Camera to use")
    dm : DMConfig = xconf.field(help="DM to use")
    calibration : CalibrationConfig = xconf.field(help='Calibration parameters')

class States(Enum):
    IDLE = 0
    FWPUPIL = 1
    FWLYOT = 2
    CENTROID = 3

class RefStates(Enum):
    IDLE = 0
    FWPUPILREF = 1
    FWLYOTREF = 2
    CENTROIDREF = 3

class pupilCorAlign(XDevice):
    config : pupilCorAlignConfig

    def setup(self):
        self.log.debug(f"I was configured! See? {self.config=}")

        self.log.info("Found camera: {:s}".format(self.config.camera.shmim))
        self.camera = XCam(self.config.camera.shmim, use_hcipy=True)

        self.ncpc_act_grid = hp.make_pupil_grid(34, 34/30.0 * np.array([1.0, np.sqrt(2)]))
        self.ncpc_dm = XDeformableMirror(dm=self.config.dm.shmim, channel=self.config.dm.channel)

        self.client.get_properties('fwscind')
        self.client.get_properties('fwpupil')
        self.client.get_properties('fwfpm')
        self.client.get_properties('fwpupil')
        self.client.get_properties('fwlyot')
        self.client.get_properties('camsci1')
        self.client.get_properties('camsci1-dark')
        self.client.get_properties('stagesci1')
        self.client.get_properties('picomotors')
        self.client.get_properties('ttmperi')

        fsm = properties.TextVector(name='fsm')
        fsm.add_element(DefText(name='state', _value=StateCodes.INITIALIZED.name))
        self.add_property(fsm)

        self._state_names = ['idle', 'fwpupil', 'fwlyot', 'centroid']
        self._state_callbacks = [None, self.handle_fwpupil, self.handle_fwlyot, self.handle_centroid]
        self._state_machine = XStateMachine(self, self._state_names, States, self._state_callbacks)

        if self.config.calibration.superuser:
            print(self.config.calibration.superuser)
            #self._reference_state_names = ['idle', 'fwpupilRef', 'fwlyotRef', 'centroidRef']
            #self._reference_state_callbacks = [None, self.handle_fwpupil_ref, self.handle_fwlyot_ref, self.handle_centroid_ref]
            #self._reference_state_machine = XStateMachine(self, self._reference_state_names, RefStates, self._reference_state_callbacks, 'reference')

        self.pupil_reference = hp.read_field(self.config.calibration.path + "reference_pupil_image.fits")
        self.xcorr_pupil = XCorrShift(self.pupil_reference, 1001, 101, filter_size=1)

        self.actuator_reference = hp.read_field(self.config.calibration.path + "actuator_reference_image.fits")
        self.xcorr_actuators = XCorrShift(self.actuator_reference, 1001, 101, filter_size=1)
        self.xcorr_actuators_global = XCorrShift(self.actuator_reference, 1001, 950, filter_size=1)
        self.global_reference_shift = np.array([0., 0.])

        self.fwpupil_references = [hp.read_field(self.config.calibration.path + "reference_{:s}_image.fits".format(mode.replace('-', '_'))) for mode in self.config.calibration.fwpupil_modes]
        self.xcorr_fwpupil = [XCorrShift(im, 1001, 101, filter_size=1) for im in self.fwpupil_references]

        self.fwlyot_references = [hp.read_field(self.config.calibration.path + "reference_{:s}_image.fits".format(mode.replace('-', '_'))) for mode in self.config.calibration.fwlyot_modes]
        self.xcorr_fwlyot = [XCorrShift(im, 1001, 101, filter_size=1) for im in self.fwlyot_references]

        # Load all the calibration files
        self.actuators_shift = np.array([0.0, 0.0])
        self.actuator_probe_pattern = make_ncpc_alignment_poke_pattern()

        self.pupil_targets = ['ttmperi.axis1_voltage', 'ttmperi.axis2_voltage']
        self.fwpupil_targets = ['picomotors.picopupil_pos', 'fwpupil.filter']
        self.fwlyot_targets = ['picomotors.picolyot_pos', 'fwlyot.filter']

        ttmperi_response_matrix = np.loadtxt(self.config.calibration.path + 'ttmperi_pupil_response_matrix.txt')
        fwpupil_response_matrix = np.loadtxt(self.config.calibration.path + 'fwpupil_picopupil_bumpmask_response_matrix.txt')[:,::-1]
        fwlyot_response_matrix = np.loadtxt(self.config.calibration.path + 'fwlyot_picolyot_lyot_response_matrix.txt')[:,::-1]

        #
        #current_position_picolyot = self.client['picomotors.picolyot_pos.current']
        #current_position_picopupil = self.client['picomotors.picolyot_pos.current']

        self.ttm_reconstruction_matrix = np.linalg.pinv(ttmperi_response_matrix)
        self.fwpupil_reconstruction_matrix = np.linalg.pinv(fwpupil_response_matrix)
        self.fwlyot_reconstruction_matrix = np.linalg.pinv(fwlyot_response_matrix)

        # These also need to be exposed to indi
        self.num_stack = 1
        self.gain = 0.25
        self.pattern_repeat = 1
        self.amp = 0.25
        self.sleep_time = 2.0

        nv = properties.NumberVector(name='nstack')
        nv.add_element(DefNumber(
            name='current', label='Number of frames', format='%i',
            min=1, max=150, step=1, _value=1
        ))
        nv.add_element(DefNumber(
            name='target', label='Number of frames', format='%i',
            min=1, max=150, step=1, _value=1
        ))
        self.add_property(nv, callback=self.handle_nstack)

        nv = properties.NumberVector(name='gain')
        nv.add_element(DefNumber(
            name='current', label='Loop Gain', format='%.2f',
            min=0.00, max=1.00, step=0.01, _value=0.10
        ))
        nv.add_element(DefNumber(
            name='target', label='Loop Gain', format='%.2f',
            min=0.00, max=1.00, step=0.01, _value=0.10
        ))
        self.add_property(nv, callback=self.handle_gain)

        # These need to be exposed to INDI
        self.fwpupil_error = np.array([0., 0.])
        self.fwlyot_error = np.array([0., 0.])

        nv = properties.NumberVector(name='fwpupil')
        nv.add_element(DefNumber( #first element
            name='dx', label='dx', format='%.4f',
            min=-50.00, max=50.00, step=0.0001, _value=0.21178766
        ))
        nv.add_element(DefNumber(
            name='dy', label='dy', format='%.4f',
            min=-50.00, max=50.00, step=0.0001, _value=0.19275196
        ))
        self.add_property(nv, callback=self.handle_fwpupil_error)

        nv = properties.NumberVector(name='fwlyot')
        nv.add_element(DefNumber( #first element
            name='dx', label='dx', format='%.4f',
            min=-50.00, max=50.00, step=0.0001, _value=0.21178766
        ))
        nv.add_element(DefNumber(
            name='dy', label='dy', format='%.4f',
            min=-50.00, max=50.00, step=0.0001, _value=0.19275196
        ))
        self.add_property(nv, callback=self.handle_fwlyot_error)

        self.taken_reference = False
        self.log.info(f'pupilCorAlign app is fully setup.')

    def stages_are_ready(self):
        '''
        '''
        if self.client['fwpupil.fsm.state'] != 'READY':
            return False
        elif self.client['fwlyot.fsm.state'] != 'READY':
            return False
        elif self.client['picomotors.fsm.state'] != 'READY':
            return False
        return True

    def control_position(self, shift, indi_targets, reconstruction_matrix):
        '''
        '''
        if self.stages_are_ready():
            current_positions = [self.client['{:s}.current'.format(indi_targets[i])] for i in [0, 1]]
            cmd = reconstruction_matrix.dot(shift)
            self.client['{:s}.target'.format(indi_targets[0])] = current_positions[0] - self.gain * cmd[0]
            self.client['{:s}.target'.format(indi_targets[1])] = current_positions[1] - self.gain * cmd[1]
            time.sleep(self.sleep_time)

    def measure_position(self, correlator):
        '''
        '''
        im = self.camera.grab_stack(self.num_stack, check_before_wait=True)
        shift = correlator.measure(im, self.actuators_shift)
        return shift

    def measure_actuator(self):
        '''
        '''
        actuator_im = 0.
        for i in range(self.pattern_repeat):
            for s in [-1, 1]:
                self.ncpc_dm.actuators += s * self.amp * self.actuator_probe_pattern
                self.ncpc_dm.send(0.05)

                self.camera.grab_stack(2, check_before_wait=True)
                actuator_im += s * self.camera.grab_stack(self.num_stack, check_before_wait=True)

                self.ncpc_dm.actuators -= s * self.amp * self.actuator_probe_pattern
        self.ncpc_dm.send()

        return actuator_im

    def get_current_state(self, indi_property):
        '''
        '''
        for name in self.client[indi_property]:
            if self.client[indi_property + '.' + name] == constants.SwitchState.ON:
                return name

    def handle_fwpupil(self):
        '''
        '''
        for mi, m in enumerate(self.config.calibration.fwpupil_modes):
            # Check if we are in the correct position
            if self.client['fwpupil.filterName.{:s}'.format(m)] == constants.SwitchState.ON:
                self.fwpupil_error = self.measure_position(self.xcorr_fwpupil[mi])
                if self.taken_reference:
                    self.control_position(self.fwpupil_error, self.fwpupil_targets, self.fwpupil_reconstruction_matrix)
                else:
                    self.log.info(f'No reference taken. Take centroid reference first.')

    def handle_fwlyot(self):
        '''
        '''
        for mi, m in enumerate(self.config.calibration.fwlyot_modes):
            # Check if we are in the correct position
            if self.client['fwlyot.filterName.{:s}'.format(m)] == constants.SwitchState.ON:
                self.fwlyot_error = self.measure_position(self.xcorr_fwlyot[mi])
                if self.taken_reference:
                    self.control_position(self.fwlyot_error, self.fwlyot_targets, self.fwlyot_reconstruction_matrix)
                else:
                    self.log.info(f'No reference taken. Take centroid reference first.')

    def handle_centroid(self):
        '''
        '''
        actuator_im = self.measure_actuator()
        self.global_reference_shift = self.xcorr_actuators_global.measure(actuator_im)
        self.actuators_shift = self.xcorr_actuators.measure(actuator_im, self.global_reference_shift)
        self.taken_reference = True
        self._state_machine.transition_to_idle()

    def handle_fwpupil_ref(self):
        '''
        '''
        im = self.camera.grab_stack(self.num_stack, check_before_wait=True)
        current_name = self.get_current_state('fwpupil.filterName')
        filename = "reference_{:s}_image.fits".format(current_name.replace('-', '_'))
        hp.write_field(im, self.config.calibration.path + filename)
        self._reference_state_machine.transition_to_idle()

    def handle_fwlyot_ref(self):
        '''
        '''
        print("entering handle fwlyot ref")
        im = self.camera.grab_stack(self.num_stack, check_before_wait=True)
        current_name = self.get_current_state('fwlyot.filterName')
        filename = "reference_{:s}_image.fits".format(current_name.replace('-', '_'))
        print(current_name)

        print("Writing file to {:s}".format(self.config.calibration.path + filename))
        hp.write_field(im, self.config.calibration.path + filename)

        print("transition back")
        self._reference_state_machine.transition_to_idle()

    def handle_centroid_ref(self):
        '''
        '''
        actuator_im = self.measure_actuator()
        filename = "actuator_reference_image.fits"
        hp.write_field(im, self.config.calibration.path + filename)
        self._reference_state_machine.transition_to_idle()

    def loop(self):
        '''
        '''
        self._state_machine.loop()
        self._reference_state_machine.loop()
        self.update_properties()

    def update_properties(self):
        '''
        '''
        self.properties['fwpupil']['dx'] = float(self.fwpupil_error[0])
        self.properties['fwpupil']['dy'] = float(self.fwpupil_error[1])
        self.update_property(self.properties['fwpupil'])

        self.properties['fwlyot']['dx'] = float(self.fwlyot_error[0])
        self.properties['fwlyot']['dy'] = float(self.fwlyot_error[1])
        self.update_property(self.properties['fwlyot'])

    def handle_nstack(self, existing_property, new_message):
        '''
        '''
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
            self.num_stack = int(new_message['target'])
            self.log.debug(f'now averaging over {self.num_stack} frames')
        self.update_property(existing_property)

    def handle_gain(self, existing_property, new_message):
        '''
        '''
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
            self.gain = float(new_message['target'])
            self.log.debug(f'new feedback gain {self.gain} ')
        self.update_property(existing_property)

    # Ask joseph for a nicer way to deal with this
    def handle_fwpupil_error(self, existing_property, new_message):
        pass

    def handle_fwlyot_error(self, existing_property, new_message):
        pass

main = pupilCorAlign.console_app
