import sys
import logging
import xconf
from magaox.indi.device import XDevice, BaseConfig
from magaox.constants import StateCodes
from purepyindi2 import device, properties, constants
from purepyindi2.messages import DefNumber, DefSwitch, DefLight, DefText

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

class corAlign(XDevice):
    config : CorAlignConfig

    def setup(self):
        self.log.debug(f"I was configured! See? {self.config=}")
        fsm = properties.TextVector(name='fsm')
        fsm.add_element(DefText(name='state', _value=StateCodes.INITIALIZED.name))
        self.add_property(fsm)

    #     self.camera = XCam(self.config.camera.shmim, use_hcipy=True)

    # def loop(self):
    #     #log.debug("Looping")
    #     image = self.camera.grab()
    #     print(np.median(image))
    #     center = measure_center_position(image)