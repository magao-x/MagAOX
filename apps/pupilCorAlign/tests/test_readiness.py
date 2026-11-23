## \file test_readiness.py
# \brief Exercise delayed INDI and camera startup through the real device loop.
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from purepyindi2 import client, constants, device, messages, transports

from xapp import pupilCorAlign as alignment


class Driver(alignment.pupilCorAlign):
    """Use the framework's device loop with an in-memory transport."""

    def __init__(self, config):
        # Bypass instrument PID, telemetry and FIFO setup.
        device.Device.__init__(self, 'pupilCorAlign', connection_class=Mock)
        self.log = Mock()
        self.telem = Mock()
        self.config = config

    def __del__(self):
        """This fixture has no telemetry file to close."""


def test_server_and_camera_start_after_driver(monkeypatch, tmp_path):
    """Discover the idle driver after seven seconds and attach its camera when available."""
    # Load the deployed settings through the same strict parser as the command-line app.
    config_file = tmp_path / 'pupilCorAlign.conf'
    config_file.write_text("""[camera]
shmim = "camsci1"
dark_shmim = "camsci1_dark"
stagepos = 54.5

[dm]
shmim = 'dmncpc'
channel = 6
offset = 0

[calibration]
path = '/opt/MagAOX/calib/coron-pupil/'
fwpupil_modes = ['bump-mask',]
fwlyot_modes = ['LyotLg1', ]
gain = 0.25
nrepeat = 4
nstack = 10
""")
    app = Driver(alignment.pupilCorAlignConfig.from_config(config_path_or_paths=str(config_file)))

    # Exercise real registration and state callbacks with simulated resources.
    outbound = client.IndiClient()
    outbound.connection = transports.IndiConnection()
    monkeypatch.setattr(outbound.connection, 'start', lambda: None)
    observer = client.IndiClient()
    observer.connection = Mock()
    observer.get_properties(app.name)
    monkeypatch.setattr(device.client, 'IndiClient', lambda: outbound)
    monkeypatch.setattr(alignment, 'XDeformableMirror', Mock())
    monkeypatch.setattr(alignment.hp, 'read_field', Mock())
    monkeypatch.setattr(alignment, 'XCorrShift', Mock())
    monkeypatch.setattr(alignment.np, 'loadtxt', lambda path: np.eye(2))
    camera = Mock()
    open_camera = Mock(side_effect=[FileNotFoundError('testcam'), camera])
    monkeypatch.setattr(alignment, 'XCam', open_camera)
    app.connection.status = constants.ConnectionStatus.CONNECTED
    elapsed = 0

    # Deliver requests over the driver's normal INDI interface.
    def select(mode):
        msg = messages.NewSwitchVector(device=app.name, name='state')
        msg.add_element(messages.OneSwitch(name=mode, _value=constants.SwitchState.ON))
        app.handle_message(msg)

    # Advance virtual time through startup, late discovery and a server restart.
    def tick(seconds):
        nonlocal elapsed
        elapsed += seconds
        if elapsed == 7:
            open_camera.assert_not_called()
            app.connection.send.reset_mock()
            app.handle_message(messages.GetProperties(device=app.name))
            for call in app.connection.send.call_args_list:
                observer.handle_message(call.args[0])
            assert observer[app.name + '.fsm.state'] == 'NOTCONNECTED'
            assert observer[app.name + '.state.idle'] == constants.SwitchState.ON
            assert app.nstack == observer[app.name + '.nstack.current'] == observer[app.name + '.nstack.target'] == 10
            assert app.gain == app.properties['gain']['current'] == app.properties['gain']['target'] == 0.25
            assert app.nrepeat == observer[app.name + '.nrepeat.current'] == observer[app.name + '.nrepeat.target'] == 4
            outbound.connection.status = constants.ConnectionStatus.CONNECTED
        elif elapsed == 8:
            assert app.properties['fsm']['state'] == 'CONNECTED'
            select('centroid')
            assert app.properties['state']['idle'] == constants.SwitchState.ON
        elif elapsed == 9:
            assert app.properties['fsm']['state'] == 'READY'
            open_camera.assert_called_with('camsci1', indi_client=outbound, use_hcipy=True, already_subscribed=True)
            assert not camera.grab_stack.called
            select('centroid')
            assert app.properties['fsm']['state'] == 'OPERATING'
            outbound.connection.status = constants.ConnectionStatus.ERROR
            outbound.connection.dispatch_callbacks(constants.TransportEvent.disconnection, outbound.status)
            assert app.properties['state']['idle'] == constants.SwitchState.ON
            outbound.connection.status = constants.ConnectionStatus.CONNECTED
        elif elapsed == 10:
            assert app.properties['fsm']['state'] == 'READY'
            select('fwpupil')
        elif elapsed == 11:
            assert app.properties['state']['idle'] == constants.SwitchState.ON
            assert open_camera.call_count == 2 and not camera.grab_stack.called
            app.connection.status = constants.ConnectionStatus.STOPPED

    monkeypatch.setattr(device, 'time', SimpleNamespace(sleep=tick))
    app.run()
    assert elapsed == 11


def test_position_relative_to_actuator_reference():
    """Verify the operational branch's relative and absolute position measurements."""
    # Supply a nonzero actuator reference so dropping the subtraction is detectable.
    app = SimpleNamespace(camera=Mock(), nstack=10, actuators_shift=np.array([3., -2.]), log=Mock())
    correlator = Mock()
    correlator.measure.side_effect = lambda image, reference: np.array([5., 4.])

    # Wheel measurements default to the delta, with absolute position available on request.
    np.testing.assert_array_equal(alignment.pupilCorAlign.measure_position(app, correlator), [2., 6.])
    np.testing.assert_array_equal(alignment.pupilCorAlign.measure_position(app, correlator, calculate_delta=False), [5., 4.])
    app.camera.grab_stack.assert_called_with(10, check_before_wait=True)
