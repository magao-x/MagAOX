import sys
import logging
from enum import Enum
import time
import numpy as np
import datetime
import os
import hcipy as hp

import xconf

from magaox.indi.device import XDevice, BaseConfig
from magaox.camera import XCam
from magaox.deformable_mirror import XDeformableMirror
from magaox.constants import StateCodes
from magaox.state_manager import XStateMachine

from purepyindi2 import device, properties, constants
from purepyindi2.messages import DefNumber, DefSwitch, DefLight, DefText


class XCorrShift():
    def __init__(self, reference_image, domain_pixels=480, domain_size=40, filter_size=None):
        self._reference_image = reference_image
        self._xgrid = hp.make_pupil_grid(domain_pixels, domain_size)

        self._fft = hp.FastFourierTransform(self._reference_image.grid)
        self._mft = hp.MatrixFourierTransform(self._xgrid, self._fft.output_grid)

        self._filter_size = filter_size
        if filter_size is not None:
            # Change this to a super gaussian filter to remove ringing.
            self._spatial_filter = hp.make_circular_aperture(self._filter_size)(self._fft.output_grid)
        else:
            self._spatial_filter = 1

        self._kernel = np.conj(self._fft.forward(self._reference_image + 0j))

    def cross_correlate(self, image):
        xcorr = np.real(self._mft.backward(self._fft.forward(image + 0j) * self._spatial_filter * self._kernel))
        return xcorr

    def measure(self, image):
        # Do a cross-correlation and find the peak pixel
        # TODO: implement sub-pixel precision with polynomial fitting
        xcorr = self.cross_correlate(image)
        indx_max = np.argmax(xcorr)
        return self._xgrid.points[indx_max]

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
    fwpupil_modes : list[str] = xconf.field(help="List of calibrated fwpupil masks.")
    fwlyot_modes : list[str] = xconf.field(help="List of calibrated fwlyot masks.")

@xconf.config
class sparkleTrackerConfig(BaseConfig):
    """Sparkle trackker config.
    """
    camera : CameraConfig = xconf.field(help="Camera to use")
    calibration : CalibrationConfig = xconf.field(help='Calibration parameters')

class States(Enum):
    IDLE = 0
    TRACK = 1
    MEASURE = 2

class sparkleTracker(XDevice):
    config : sparkleTrackerConfig

    def setup(self):
        self.log.debug(f"I was configured! See? {self.config=}")

        self.log.info("Found camera: {:s}".format(self.config.camera.shmim))
        self.camera = XCam(self.config.camera.shmim, use_hcipy=True)

        fsm = properties.TextVector(name='fsm')
        fsm.add_element(DefText(name='state', _value=StateCodes.INITIALIZED.name))
        self.add_property(fsm)

        self._state_names = ['idle', 'track', 'measure']
        self._state_callbacks = [None, self.track, self.measure]
        self._state_machine = XStateMachine(self, self._state_names, States, self._state_callbacks)

        self.log.info(f'sparkleTracker app is fully setup.')

    def calculate_shift(self):
        pass

    def track(self):
        pass

    def loop(self):
        '''
        '''
        self._state_machine.loop()
        self.update_properties()