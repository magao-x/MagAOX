import sys
import logging
import xconf
from magaox.indi.device import XDevice, BaseConfig

log = logging.getLogger(__name__)

@xconf.config
class ExampleConfig(BaseConfig):
    """Example Python INDI device for MagAO-X

    Write your command-line help here, and it will be displayed when
    someone runs `pythonIndiExample -h` in the terminal (along with
    a summary of available options)
    """
    configurable_doodad_1 : str = xconf.field(default="abc", help="Configurable doodad 1")

class pythonIndiExample(XDevice):
    config : ExampleConfig

    def loop(self):
        log.info("Looping")