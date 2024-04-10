import sys
import logging
from enum import Enum
import time
import numpy as np

import xconf

from magaox.indi.device import XDevice, BaseConfig
from magaox.camera import XCam
from magaox.deformable_mirror import XDeformableMirror, XFourierMirror, XPokeMirror
from magaox.constants import StateCodes

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
class efcControlConfig(BaseConfig):
    """Automatic coronagraph alignment assistant
    """
    camera : CameraConfig = xconf.field(help="Camera to use")
    sleep_interval_sec : float = xconf.field(default=0.25, help="Sleep interval between loop() calls")

class States(Enum):
    IDLE = 0

class efcControl(XDevice):
    config : efcControlConfig
    
    def setup(self):
        self.log.debug(f"I was configured! See? {self.config=}")
        fsm = properties.TextVector(name='fsm')
        fsm.add_element(DefText(name='state', _value=StateCodes.INITIALIZED.name))
        self.add_property(fsm)

        self.log.info("Found camera: {:s}".format(self.config.camera.shmim))
        self.camera = XCam(self.config.camera.shmim, use_hcipy=True)
        self._state = States.IDLE

        self.properties['fsm']['state'] = StateCodes.READY.name
        self.update_property(self.properties['fsm'])

    def loop(self):
       pass 
		# Basic EFC loop
		# Acquire sequence of images
		# Need to trigger based on start of exposure?
		# Send DM command
		# Get
