import os
import sys
from pathlib import Path

import numpy as np
import logging

# Ensure tests run from apps/aoSim as requested
repo_root = Path(__file__).resolve().parents[2]  # apps/aoSim/test -> apps/aoSim -> apps
sys.path.insert(0, str(repo_root / "aoSim"))

from xapp.aoSim import aoSim, aoSimConfig
from purepyindi2 import constants


class DummyImage:
    def __init__(self, data=None):
        self._data = data

    def get_data(self, wait=False):
        if self._data is None:
            raise RuntimeError("No data configured")
        return self._data

    def write(self, data):
        self._data = np.array(data, dtype=np.float32)


def make_fake_aosim():
    sim = object.__new__(aoSim)
    sim._nmodes = 2
    sim._noise = 0.0
    sim._frequency = 0.5
    sim._t = np.array([0.0], dtype=np.float32)
    sim._dt = np.array([0.01], dtype=np.float32)
    sim._current_disturbance = np.zeros((2, 1), dtype=np.float32)
    sim._current_dm_state = np.zeros((2, 1), dtype=np.float32)
    sim._lag = 1
    sim.log = logging.getLogger()
    sim._dm_command_history = np.zeros((sim._lag + 1, sim._nmodes, 1), dtype=np.float32)

    sim._dm = DummyImage(np.array([[0.5], [1.5]], dtype=np.float32))
    sim._disturbance = DummyImage()
    sim._wfs = DummyImage()
    return sim


def test_config_default_num_modes():
    cfg = aoSimConfig()
    assert cfg.num_modes == 2


def test_update_dm_uses_command_history_and_dm_data():
    sim = make_fake_aosim()
    sim._dm_command_history[0] = np.array([[0.1], [0.2]], dtype=np.float32)
    sim._dm_command_history[1] = np.array([[0.3], [0.4]], dtype=np.float32)

    sim.update_dm()

    np.testing.assert_array_equal(sim._current_dm_state, np.array([[0.1], [0.2]], dtype=np.float32))
    np.testing.assert_array_equal(sim._dm_command_history[0], np.array([[0.3], [0.4]], dtype=np.float32))
    np.testing.assert_array_equal(sim._dm_command_history[-1, :, 0], np.array([0.5, 1.5], dtype=np.float32))


def test_update_disturbance_and_wfs_propagate_values():
    sim = make_fake_aosim()
    sim._current_dm_state = np.array([[0.2], [0.3]], dtype=np.float32)
    sim._t = np.array([0.25], dtype=np.float32)

    sim.update_disturbance()
    assert sim._disturbance.get_data() is not None

    sim.update_wfs()
    np.testing.assert_array_equal(sim.err, sim._current_disturbance + sim._current_dm_state)
    np.testing.assert_array_equal(sim._wfs.get_data(), sim.err)


def test_loop_advances_time_and_updates_fields():
    sim = make_fake_aosim()
    sim._dm_command_history[0] = np.array([[0.1], [0.1]], dtype=np.float32)
    sim._dm_command_history[1] = np.array([[0.0], [0.0]], dtype=np.float32)

    t_before = sim._t.copy()
    sim.loop()

    np.testing.assert_allclose(sim._t, t_before + sim._dt)
    assert sim._wfs.get_data() is not None
    assert sim._disturbance.get_data() is not None


def test_handle_reset_resets_simulation_state():
    sim = make_fake_aosim()
    # Modify state to non-zero values
    sim._t = np.array([1.5], dtype=np.float32)
    sim._current_disturbance = np.array([[0.5], [0.3]], dtype=np.float32)
    sim._current_dm_state = np.array([[0.2], [0.4]], dtype=np.float32)
    sim._dm_command_history = np.ones((sim._lag + 1, sim._nmodes, 1), dtype=np.float32)
    
    # Mock the property and message
    existing_property = {'toggle': constants.SwitchState.OFF}
    new_message = {'toggle': constants.SwitchState.ON}
    
    # Mock update_property to do nothing
    sim.update_property = lambda prop: None
    print(sim._dm_command_history)
    sim.handle_reset(existing_property, new_message)
    
    # Check that state is reset
    np.testing.assert_array_equal(sim._t, np.array([0.0], dtype=np.float32))
    np.testing.assert_array_equal(sim._current_disturbance, np.zeros((2, 1), dtype=np.float32))
    np.testing.assert_array_equal(sim._current_dm_state, np.zeros((2, 1), dtype=np.float32))
    print(sim._dm_command_history)
    np.testing.assert_array_equal(sim._dm_command_history, np.zeros((sim._lag + 1, sim._nmodes, 1), dtype=np.float32))
    
    # Check that shmims are written with zeros
    np.testing.assert_array_equal(sim._wfs.get_data(), np.zeros((2, 1), dtype=np.float32))
    np.testing.assert_array_equal(sim._disturbance.get_data(), np.zeros((2, 1), dtype=np.float32))
    np.testing.assert_array_equal(sim._dm.get_data(), np.zeros((2, 1), dtype=np.float32))
    
    # Check that property is set back to OFF
    assert existing_property['toggle'] == constants.SwitchState.OFF
