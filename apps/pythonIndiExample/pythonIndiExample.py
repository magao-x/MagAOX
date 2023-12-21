import sys
import logging
import xconf
from magaox.indi.device import XDevice, BaseConfig

log = logging.getLogger(__name__)

@xconf.config
class ExampleConfig(BaseConfig):
    """Example Python INDI device for MagAO-X
    """
    configurable_doodad_1 : str = xconf.field(default="abc", help="Configurable doodad 1")

class PythonIndiExample(XDevice):
    config : ExampleConfig

    def loop(self):
        log.info("Looping")