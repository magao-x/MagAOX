import numpy as np
from purepyindi2 import device, properties, constants
from purepyindi2.messages import DefNumber, DefSwitch, DefLight, DefText

from magaox.constants import StateCodes

class XStateMachine:
    '''A generic MagAO-X Indi state machine manager
    '''
    def __init__(self, device, state_names, state_enum, state_callbacks, top_level_name='state'):
        self._device = device
        self._state_names = state_names
        self._state_enum = state_enum
        self._state_callbacks = state_callbacks
        self.top_level_name = top_level_name

        sv = properties.SwitchVector(
            name=top_level_name,
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        for si, state_name in enumerate(state_names):
            sv.add_element(DefSwitch(name=state_name, _value=constants.SwitchState.ON if si == 0 else constants.SwitchState.OFF))
        self._device.add_property(sv, callback=self.handle_state)

        self._state = self._state_enum.IDLE

    def transition_to_idle(self):
        for name in self._state_names:
            self._device.properties[self.top_level_name][name] = constants.SwitchState.OFF
        self._device.properties[self.top_level_name]['idle'] = constants.SwitchState.ON
        self._device.update_property(self._device.properties[self.top_level_name])
        self._state = self._state_enum.IDLE

    def handle_state(self, existing_property, new_message):
        for key in self._state_names:
            if existing_property[key] == constants.SwitchState.ON:
                current_state = key

        if current_state not in new_message:
            for key in self._state_names:
                existing_property[key] = constants.SwitchState.OFF

                if key in new_message:
                    existing_property[key] = new_message[key]

                    for ti, test_state in enumerate(self._state_names):
                        if key == test_state:
                            self._state = self._state_enum(ti)
                            if ti == 0:
                                self._device.properties['fsm']['state'] = StateCodes.READY.name
                            else:
                                self._device.properties['fsm']['state'] = StateCodes.OPERATING.name
                            self._device.log.info('State {:s} changed to {:s}'.format(self.top_level_name, test_state))

            self._device.update_property(existing_property)
            self._device.update_property(self._device.properties['fsm'])

    def loop(self):
        for si, state in enumerate(self._state_enum):
            if self._state == state:
                if self._state_callbacks[si] is not None:
                    self._state_callbacks[si]()