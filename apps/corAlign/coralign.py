#!/usr/bin/env python
import logging
import configparser
from purepyindi2 import device, properties, constants
from purepyindi2.messages import DefNumber, DefText, DefSwitch

from magaox.camera import XCam

class Configuration(configparser.ConfigParser):

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self._parse_config()

    def _parse_config(self):
        self.read('/opt/MagAOX/config/' + self.filename + '.conf')

    def get_param(self, skey, key, dtype):
        val = self.get(skey, key) # still a str
        vallist = val.split(',') # handle lists
        if len(vallist) == 1:
            return dtype(vallist[0])
        else:
            return [dtype(v) for v in vallist]

    def update_from_dict(self, update_dict):
        for key, value in update_dict.items():
            section, option = key.split('.')
            try:
                self.get(section, option) # needed to verify that option already exists
                self.set(section, option, value=value)
            except configparser.NoSectionError as e:
                raise RuntimeError(f'Could not find section "{section}" in config file. Double-check override option "{key}={value}."') from e
            except configparser.NoOptionError as e:
                raise RuntimeError(f'Could not find option "{option}" in "{section}" section of config file. Double-check override option "{key}={value}."') from e

log = logging.getLogger(__name__)
    
def measure_center_position(image, threshold=1, mask_diameter=20):
    center_mask = make_circular_aperture(mask_diameter)(image.grid)
    mask = center_mask * ((image / np.std(image)) < threshold)   
    
    xc = np.sum(mask * image.grid.x) / np.sum(mask)
    yc = np.sum(mask * image.grid.y) / np.sum(mask)
    return np.array([xc, yc])

class corAlign(device.Device):

	def handle_loop(self, existing_property, new_message):
		if 'request' in new_message and new_message['request'] is constants.SwitchState.ON:
			pass
		if 'cancel' in new_message and new_message['cancel'] is constants.SwitchState.ON:
			pass
		self.update_property(existing_property)  # ensure the switch turns back off at the client

    def _init_properties(self):
        fsmstate = properties.TextVector(
            name='fsm',
        )
        fsmstate.add_element(DefText(name="state", _value="READY"))
        self.add_property(fsmstate)
		
		sv = properties.SwitchVector(
				name='loop',
				rule=constants.SwitchRule.ONE_OF_MANY,
				perm=constants.PropertyPerm.READ_WRITE,
		)
		sv.add_element(DefSwitch(name="nothing", _value=constants.SwitchState.ON))
		sv.add_element(DefSwitch(name="measure", _value=constants.SwitchState.OFF))
 		self.add_property(sv, callback=self.handle_expose)

    def _load_config(self):
        self._config = Configuration(self.name)
        
        self._camera_shmim = self._config.get_param('camera', 'camera_shmim', str)
        self._dark_shmim = self._config.get_param('camera', 'dark_shmim', str)
        log.info("Got shmims {:s} and {:s}".format(self._camera_shmim, self._dark_shmim))

    def setup(self):      
        self._load_config()
        
        # Connect to hardware
        self.camera = XCam(self._camera_shmim, use_hcipy=True)
        
        # Setup properties
        self._init_properties()

        # Finialize setup
        log.debug("Set up complete")

    def loop(self):
		
		image = self.camera.grab()
        #uptime_prop = self.properties['uptime']
        #uptime_prop['uptime_sec'] += 1
        #self.update_property(uptime_prop)
        #log.debug(f"Current uptime: {uptime_prop}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    corAlign(name="coralign").main()