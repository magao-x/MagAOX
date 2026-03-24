import logging
import numpy as np

import ImageStreamIOWrap
import xconf

from purepyindi2 import device, properties, constants
from purepyindi2.messages import DefNumber, DefSwitch, DefText
from magaox.indi.device import XDevice, BaseConfig
from magaox.shmim import Image

def create_if_not_exist_shmim(name, shape, dtype=np.float32):
    """Create or verify a shared memory image (shmim) with the specified parameters.
    
    Creates a new shared memory image if it does not exist, or verifies that an existing
    image has the correct shape and data type. If an existing image has incompatible
    parameters, it is destroyed and recreated.
    
    Args:
        name (str): The name of the shared memory image to create or access.
        shape (tuple): The desired shape of the array as (height, width).
        dtype (type, optional): The numpy data type for the array. 
            Defaults to np.float32 (FLOAT). Can also be np.float64 (DOUBLE).
    
    Returns:
        None
        
    Note:
        This function closes the connection to the shmim after creation/verification.
        Other processes can open the shmim by its name independently.
    """
    img = ImageStreamIOWrap.Image()
    
    # Check if the shmim exists
    if img.open(name) == 40:
        img.create(name, np.zeros(shape, dtype=dtype))
    else:
        # the shmim opened successfully, check if it has the right shape and dtype, if not destroy and recreate
        if not (img.md.size[0] == shape[0] and img.md.size[1] == shape[1] and img.md.datatype == (ImageStreamIOWrap.ImageStreamIODataType.FLOAT if dtype == np.float32 else ImageStreamIOWrap.ImageStreamIODataType.DOUBLE)):
            img.destroy()
            img.create(name, np.zeros(shape, dtype=dtype))

    # Close the connection to the shmim.
    img.close()

@xconf.config
class aoSimConfig(BaseConfig):
    """Configuration for the aoSim application.
    """
    num_modes : int = xconf.field(default=2, help="Number of modes to simulate in the AO system.")
    lag : int = xconf.field(default=1, help="Lag in timesteps for DM command updates.")
    noise : float = xconf.field(default=0.0, help="Amplitude of noise to add to the wavefront sensor measurements.")
    frequency : float = xconf.field(default=0.5, help="Frequency in Hz at which the AO simulation loop runs.")

class aoSim(XDevice):
    """Adaptive Optics System Simulator.
    
    The aoSim application provides a simulated modal AO system for testing and development.
    """
    config : aoSimConfig

    def setup(self):
        """Initialize the AO simulator application.
        
        Sets up shared memory images for DM, WFS, and disturbance, initializes
        simulation state variables, and configures the control loop parameters.
        """

        self._nmodes = self.config.num_modes

        create_if_not_exist_shmim('aoSim_dm', [self._nmodes, 1], dtype=np.float32)
        self._dm = Image('aoSim_dm')

        create_if_not_exist_shmim('aoSim_wfs', [self._nmodes, 1], dtype=np.float32)
        self._wfs = Image('aoSim_wfs')

        create_if_not_exist_shmim('aoSim_disturbance', [self._nmodes, 1], dtype=np.float32)
        self._disturbance = Image('aoSim_disturbance')

        self._t = np.array([0], dtype=np.float32)
        self._dt = np.array([0.01], dtype=np.float32)
        self._current_disturbance = np.zeros((self._nmodes, 1), dtype=np.float32)
        self._current_dm_state = np.zeros((self._nmodes, 1), dtype=np.float32)

        self._lag = self.config.lag
        self._dm_command_history = np.zeros((self._lag + 1, self._nmodes, 1), dtype=np.float32)

        self._noise = self.config.noise
        self._frequency = self.config.frequency
        # Update the dm first so that the current shape in the shared memory is correct
        #  before the first WFS update, which relies on the current DM state.
        self.update_dm()

        sv = properties.SwitchVector(
            name='reset',
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        sv.add_element(DefSwitch(name="toggle", _value=constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_reset)

        self.log.info(f'aoSim app is fully setup.')

    def handle_reset(self, existing_property, new_message):
        if 'toggle' in new_message and new_message['toggle'] == constants.SwitchState.ON:
            self._t = np.array([0], dtype=np.float32)
            
            self._current_disturbance = np.zeros((self._nmodes, 1), dtype=np.float32)
            self._current_dm_state = np.zeros((self._nmodes, 1), dtype=np.float32)
            self._dm_command_history = np.zeros((self._lag + 1, self._nmodes, 1), dtype=np.float32)

            self._wfs.write(np.zeros((self._nmodes, 1), dtype=np.float32))
            self._disturbance.write(np.zeros((self._nmodes, 1), dtype=np.float32))
            self._dm.write(np.zeros((self._nmodes, 1), dtype=np.float32))

            self.update_dm()

            existing_property['toggle'] = constants.SwitchState.OFF
            self.update_property(existing_property)
            self.log.info('AO simulation state has been reset.')

    def update_dm(self):
        """Update the deformable mirror state from its command history.
        """
        self._current_dm_state = self._dm_command_history[0]
        self._dm_command_history = np.roll(self._dm_command_history, shift=-1, axis=0)
        self._dm_command_history[-1] = self._dm.get_data(wait=False)
        
    def update_wfs(self):
        """Update the wavefront sensor measurement in shared memory.
        """
        self.err = self._current_disturbance + self._current_dm_state 
        self.err += self._noise * np.random.randn(self._nmodes, 1).astype(np.float32)
        self._wfs.write(self.err)

    def update_disturbance(self):
        """Update the atmospheric/system disturbance for this timestep.
        """
        self._current_disturbance = 0.1 * np.sin(2 * np.pi * self._frequency * self._t) * np.ones((self._nmodes, 1), dtype=np.float32)
        self._disturbance.write(self._current_disturbance)

    def loop(self):
        """Execute one iteration of the AO simulation loop.
        """
        self.update_dm()
        self.update_disturbance()
        self.update_wfs()
        
        print('t={:.2f}, disturbance={:.3f}, dm_state={:.3f}, wfs={:.3f}'.format(self._t[0], self._current_disturbance[0,0], self._current_dm_state[0,0], self.err[0,0]))
        self._t += self._dt

main = aoSim.console_app
