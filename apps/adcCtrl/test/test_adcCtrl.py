"""Unit tests for the adcCtrl application.

Run from the repository root with the MagAO-X Python environment
(purepyindi2, xconf, magaox, hcipy):

    python -m pytest apps/adcCtrl/test -v

Device tests build adcCtrl with object.__new__ and inject a fake INDI client
and camera, so no INDI server, shmim or log/telemetry directories are needed.
Synthetic frames come from make_frame(), a broadband model in which the core
and spot positions scale with wavelength and the core shifts linearly across
the band to emulate residual dispersion.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

# Run from apps/adcCtrl, following apps/aoSim/test
app_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(app_root))

from xapp.adcCtrl.app import (
    AdcCtrlConfig,
    SpotGeometry,
    States,
    adcCtrl,
    compute_step,
    crop_image,
    expected_axes,
    inlier_mask,
    measure_spot_angles,
    moment_angle,
    pair_offsets,
    window_field,
    wrap90,
)
from purepyindi2 import constants
import hcipy as hp

PIXEL_SCALE = 6.0 / 21.0
ON = constants.SwitchState.ON
OFF = constants.SwitchState.OFF


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------

def elongated_spot(angle_deg, n=30, amp=500.0, sigma_long=4.0, sigma_short=1.6, center=None, noise=0.0, seed=0):
    """A 2D Gaussian crop elongated along `angle_deg` (x toward y, array coordinates)."""
    y, x = np.mgrid[0:n, 0:n].astype(float)
    x0, y0 = center if center is not None else ((n - 1) / 2, (n - 1) / 2)
    t = np.radians(angle_deg)
    u = (x - x0) * np.cos(t) + (y - y0) * np.sin(t)
    v = -(x - x0) * np.sin(t) + (y - y0) * np.cos(t)
    img = amp * np.exp(-0.5 * ((u / sigma_long) ** 2 + (v / sigma_short) ** 2))
    if noise > 0:
        img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
    return img


def golden_crop(i):
    """Deterministic elongated-spot crop with pseudo-noise (no RNG), for golden values."""
    n = 30
    y, x = np.mgrid[0:n, 0:n].astype(float)
    theta = np.radians(-80 + 23 * i)
    x0, y0 = 14.3 + 0.37 * i % 2, 15.1 - 0.29 * (i % 3)
    u = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
    v = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
    amp = [400.0, 60.0, 15.0, 1000.0, 120.0, 30.0, 8.0, 250.0][i]
    img = amp * np.exp(-0.5 * ((u / 4.0) ** 2 + (v / 1.6) ** 2))
    noise = 3.0 * np.sin(12.9898 * x + 78.233 * y + 1.7 * i) * np.cos(4.1 * x * y + i)
    return img + noise + 2.0


# moment_angle(golden_crop(i)) from adc_sims/algo_26B/adc_ctrl.py
GOLDEN_MOMENT_ANGLES = [
    -79.98673840436133,
    -57.330027045125775,
    -32.868761013164196,
    -10.998993126442297,
    12.045148385080944,
    34.004645869951894,
    53.20448748299557,
    80.92745962147545,
]


def make_frame(
    size=256,
    separation=15.0,
    angle=0.0,
    wavelength=656e-9,
    bandwidth=0.15,
    dispersion=(0.0, 0.0),
    core_amp=1000.0,
    spot_amp=60.0,
    noise=0.0,
    seed=0,
):
    """Broadband PSF core plus four satellite spots, in camera pixels.

    `separation` is in lambda/D at the observing wavelength and `angle` rotates
    the spot pattern clockwise, matching speckle_cutout. `dispersion` is the
    core shift (x, y) in reference lambda/D across the full band.
    """
    ny = nx = size
    x = (np.arange(nx) - (nx - 1) / 2) * PIXEL_SCALE
    y = (np.arange(ny) - (ny - 1) / 2) * PIXEL_SCALE
    xx, yy = np.meshgrid(x, y)
    frame = np.zeros((ny, nx))

    lam_ref = 656e-9
    fractions = np.linspace(-0.5, 0.5, 11)
    polar = np.radians(90.0 - angle - 90.0 * np.arange(4))
    for frac in fractions:
        lam = wavelength * (1 + bandwidth * frac)
        s = lam / lam_ref
        cx, cy = dispersion[0] * frac, dispersion[1] * frac
        sigma = 0.45 * s
        frame += core_amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
        for p in polar:
            sx = cx + separation * (wavelength / lam_ref) * (lam / wavelength) * np.cos(p)
            sy = cy + separation * (wavelength / lam_ref) * (lam / wavelength) * np.sin(p)
            frame += spot_amp * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * sigma**2))

    frame /= len(fractions)
    if noise > 0:
        frame += np.random.default_rng(seed).normal(0, noise, frame.shape)
    return frame


def sparkle_geometry(separation=15.0, angle=0.0, normalized_wavelength=1.0):
    return SpotGeometry(separation * normalized_wavelength, angle, 20, 20.0)


# --------------------------------------------------------------------------
# Fakes for the device
# --------------------------------------------------------------------------

class FakeClient(dict):
    """dict-backed INDI client that records writes and can emulate adctrack."""

    def __init__(self, *args, follow_adc=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.writes = []
        self.follow_adc = follow_adc

    def __setitem__(self, key, value):
        if hasattr(self, "writes"):
            self.writes.append((key, value))
        super().__setitem__(key, value)
        if getattr(self, "follow_adc", False) and key.endswith(".target") and ".deltaADC" in key:
            super().__setitem__(key.replace(".target", ".current"), value)

    def get_properties(self, device_name):
        pass

    def adc_writes(self):
        return [w for w in self.writes if w[0].startswith("adctrack.")]


class FakeCamera:
    def __init__(self, frame=None, dark=True, fail=False):
        self.frame = frame
        self._dark_exists = dark
        self.fail = fail
        self.calls = []

    def grab_stack(self, num_images, subtract_dark=True):
        self.calls.append((num_images, subtract_dark))
        if self.fail:
            raise TimeoutError("no frames")
        return self.frame


def adc_client(**kwargs):
    client = FakeClient(**kwargs)
    dict.update(client, {
        "adctrack.deltaADC1.current": 0.0,
        "adctrack.deltaADC2.current": 0.0,
        "adctrack.tracking.toggle": ON,
        "tweeterSpeck.separation.current": 15.0,
        "tweeterSpeck.angle.current": 0.0,
        "fwsci1.filterName.i": OFF,
        "fwsci1.filterName.z": OFF,
        "fwsci1.filterName.r": OFF,
    })
    return client


def make_device(client=None, camera=None, **config_overrides):
    """Build an adcCtrl without starting INDI, logging files or the camera."""
    dev = object.__new__(adcCtrl)
    dev.config = AdcCtrlConfig(**config_overrides)
    dev.log = logging.getLogger("adcCtrl-test")
    dev.client = client if client is not None else adc_client()
    dev.properties = {}
    dev.callbacks = {}

    def add_property(prop, callback=None):
        dev.properties[prop.name] = prop
        dev.callbacks[prop.name] = callback

    dev.add_property = add_property
    dev.update_property = lambda prop: None
    dev.telem_records = []
    dev.telem = lambda event, message: dev.telem_records.append((event, message))
    dev.init_state()
    dev.create_properties()
    dev.camera = camera
    return dev


def send(dev, prop_name, **elements):
    """Deliver a new-property message to a device callback."""
    dev.callbacks[prop_name](dev.properties[prop_name], dict(elements))


# --------------------------------------------------------------------------
# Algorithm
# --------------------------------------------------------------------------

def test_moment_angle_recovers_noiseless_angles():
    """moment_angle recovers the axis of noiseless elongated spots to 0.1 deg."""
    for angle in np.arange(-85, 86, 5):
        measured = moment_angle(elongated_spot(angle))
        assert abs(wrap90(measured - angle)) < 0.1, (angle, measured)


def test_moment_angle_snr_tiers():
    """moment_angle bias stays below 0.5 deg in each SNR regime."""
    for amp in [8.0, 30.0, 100.0]:
        errors = [
            wrap90(moment_angle(elongated_spot(30.0, amp=amp, noise=1.0, seed=s)) - 30.0)
            for s in range(50)
        ]
        assert abs(np.mean(errors)) < 0.5, (amp, np.mean(errors), np.std(errors))


def test_moment_angle_matches_simulation_reference():
    """moment_angle is unchanged from adc_sims/algo_26B/adc_ctrl.py."""
    for i, expected in enumerate(GOLDEN_MOMENT_ANGLES):
        assert abs(moment_angle(golden_crop(i)) - expected) < 1e-10


def test_wrap90_is_continuous_across_vertical_axis():
    """Deviations from a vertical nominal axis keep their sign across the +-90 seam."""
    nominal = expected_axes(0.0)[0]
    assert nominal == 90.0
    for delta in [-5.0, -2.0, -0.5, 0.5, 2.0, 5.0]:
        raw = moment_angle(elongated_spot(90.0 + delta))
        assert abs(raw) > 80.0
        dev = wrap90(raw - nominal)
        assert abs(dev - delta) < 0.2, (delta, raw, dev)


def test_pair_offsets():
    """pair_offsets differences opposite spots."""
    np.testing.assert_allclose(pair_offsets([1.0, 2.0, -1.0, 0.5]), [2.0, 1.5])


def test_window_field_pads_outside_image():
    """window_field returns the requested shape when the window leaves the image."""
    grid = hp.make_pupil_grid(40, 40)
    field = hp.Field(np.ones(grid.size), grid)
    cut = window_field(field, [19.5, 19.5], 20, 20)
    assert cut.shaped.shape == (20, 20)
    assert np.sum(cut) < 20 * 20


def test_crop_image_blank_frame():
    """crop_image does not fail on an all-zero frame."""
    grid = hp.make_pupil_grid(64, 64 * PIXEL_SCALE)
    out = crop_image(hp.Field(np.zeros(grid.size), grid), 40, mask_diam=5)
    assert out.shaped.shape == (40, 40)
    assert np.all(out == 0)


def test_spot_localization_and_wavelength_scaling():
    """All four spots are found for sparkles and DM spots, with wavelength scaling."""
    cases = [
        (15.0, 0.0, 656e-9, 20, 20.0, 256),
        (15.0, 28.0, 908e-9, 20, 20.0, 256),
        (47.0, 28.0, 762e-9, 50, 30.0, 480),
    ]
    for separation, angle, wavelength, window, search, size in cases:
        nwl = wavelength / 656e-9
        frame = make_frame(size=size, separation=separation, angle=angle, wavelength=wavelength)
        geometry = SpotGeometry(separation * nwl, angle, window, search)
        raw, dev = measure_spot_angles(frame, geometry, PIXEL_SCALE)
        # undispersed spots are radial
        assert np.all(np.abs(dev) < 1.0), (separation, angle, wavelength, raw, dev)


def test_missing_wavelength_scaling_misses_spots():
    """Without wavelength scaling, z-band sparkle searches land off the spots."""
    frame = make_frame(separation=15.0, angle=10.0, wavelength=908e-9)
    good = SpotGeometry(15.0 * 908 / 656, 10.0, 20, 6.0)
    bad = SpotGeometry(15.0, 10.0, 20, 6.0)
    _, dev_good = measure_spot_angles(frame, good, PIXEL_SCALE)
    _, dev_bad = measure_spot_angles(frame, bad, PIXEL_SCALE)
    assert np.all(np.abs(dev_good) < 1.0)
    assert np.max(np.abs(dev_bad)) > np.max(np.abs(dev_good))


def test_pair_offsets_linear_in_dispersion():
    """Pair offsets respond linearly and antisymmetrically to dispersion."""
    angle = 0.0
    geometry = sparkle_geometry(angle=angle)
    # spot 1 direction, perpendicular to the spot 0/2 axis
    direction = np.array([np.cos(np.radians(-angle)), np.sin(np.radians(-angle))])
    amounts = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    pair02 = []
    for amount in amounts:
        frame = make_frame(separation=15.0, angle=angle, dispersion=amount * direction)
        _, dev = measure_spot_angles(frame, geometry, PIXEL_SCALE)
        pair02.append(pair_offsets(dev)[0])
        # opposite spots rotate in opposite directions
        assert abs(dev[0] + dev[2]) < 0.5 + 0.1 * abs(dev[0])
    pair02 = np.array(pair02)
    slope, intercept = np.polyfit(amounts, pair02, 1)
    residual = pair02 - (slope * amounts + intercept)
    r2 = 1 - np.sum(residual**2) / np.sum((pair02 - pair02.mean()) ** 2)
    assert abs(slope) > 1.0
    assert r2 > 0.99, r2
    assert np.sign(pair02[0]) == -np.sign(pair02[-1])


# --------------------------------------------------------------------------
# Control math
# --------------------------------------------------------------------------

def test_compute_step_rejects_large_steps():
    """Steps are gain and sign scaled, and rejected (not clipped) at the limit."""
    assert compute_step(1.0, 0.5, 1.0, 0.7) == (0.5, True)
    assert compute_step(1.0, 0.5, -1.0, 0.7) == (-0.5, True)
    step, accepted = compute_step(2.0, 0.5, 1.0, 0.7)
    assert step == 1.0 and not accepted
    assert compute_step(np.nan, 0.5, 1.0, 0.7)[1] is False


def test_inlier_mask_rejects_outliers_and_nans():
    """inlier_mask drops NaNs and values far from the median."""
    mask = inlier_mask([1.0, 1.1, 0.9, np.nan, 50.0, 1.05])
    np.testing.assert_array_equal(mask, [True, True, True, False, False, True])
    assert not np.any(inlier_mask([np.nan, np.nan]))


# --------------------------------------------------------------------------
# Device behavior
# --------------------------------------------------------------------------

def test_state_transitions():
    """Each state switch sets the internal state, switch elements and fsm."""
    dev = make_device()
    for name, state, fsm in [
        ("adcLoop", States.CLOSED_LOOP, "OPERATING"),
        ("measure-only", States.MEASURE_ONLY, "OPERATING"),
        ("oneshot", States.ONESHOT, "OPERATING"),
        ("idle", States.IDLE, "READY"),
    ]:
        send(dev, "state", **{name: ON})
        assert dev._state == state
        assert dev.properties["state"][name] == ON
        assert dev.properties["fsm"]["state"] == fsm
    send(dev, "state", idle=OFF)
    assert dev._state == States.IDLE


def test_oneshot_sends_once_then_idles():
    """One-shot sends exactly one command and returns to idle."""
    camera = FakeCamera(make_frame(dispersion=(0.02, 0.0)))
    dev = make_device(camera=camera)
    send(dev, "state", oneshot=ON)
    dev.loop()
    assert dev._state == States.IDLE
    assert len([w for w in dev.client.adc_writes() if w[0].endswith("deltaADC1.target")]) == 1
    dev.loop()
    assert len(camera.calls) == 1


def test_measure_only_never_writes_adc():
    """Measure-only publishes measurements but never writes to adctrack."""
    camera = FakeCamera(make_frame(dispersion=(0.02, 0.0)))
    dev = make_device(camera=camera)
    send(dev, "state", **{"measure-only": ON})
    for _ in range(3):
        dev.loop()
    assert dev.client.adc_writes() == []
    assert dev.properties["measurement"]["n_valid"] == 1
    assert dev.properties["status"]["last_command"] == "measure-only"


def test_idle_does_nothing():
    """Idle never touches the camera or adctrack."""
    camera = FakeCamera(make_frame())
    dev = make_device(camera=camera)
    dev.loop()
    assert camera.calls == []
    assert dev.client.adc_writes() == []


def test_send_command_times_out():
    """send_command returns False after the timeout if the stages never arrive."""
    client = adc_client(follow_adc=False)
    dev = make_device(client=client, send_timeout_sec=0.2)
    dev.set_command(0.3, 0.0)
    start = time.monotonic()
    assert dev.send_command() is False
    assert time.monotonic() - start < 1.0


def test_send_command_tracking_off_does_not_wait():
    """With tracking off the command is written without waiting for the stages."""
    client = adc_client(follow_adc=False)
    dict.__setitem__(client, "adctrack.tracking.toggle", OFF)
    dev = make_device(client=client, send_timeout_sec=5.0)
    dev.set_command(0.3, 0.0)
    start = time.monotonic()
    assert dev.send_command() is True
    assert time.monotonic() - start < 0.5
    assert ("adctrack.deltaADC1.target", 0.3) in client.writes


def test_camera_failures_drop_closed_loop_to_idle():
    """Repeated camera failures in closed loop end in idle without raising."""
    dev = make_device(camera=FakeCamera(fail=True), max_consecutive_failures=3)
    send(dev, "state", adcLoop=ON)
    for _ in range(3):
        dev.loop()
    assert dev._state == States.IDLE
    assert dev.client.adc_writes() == []


def test_missing_external_properties_use_defaults():
    """check_indi_props falls back to defaults when other devices are absent."""
    dev = make_device(client=FakeClient())
    dev.check_indi_props()
    assert dev._normalized_wavelength == 1.0
    assert dev._sparkle_freq == 15.0
    dev.apply_pending()  # adctrack missing: nothing written, no exception
    assert dev.client.writes == []


def test_filter_sets_normalized_wavelength():
    """The selected filter on the active wheel sets the normalized wavelength."""
    client = adc_client()
    dict.__setitem__(client, "fwsci1.filterName.z", ON)
    dict.__setitem__(client, "fwsci2.filterName.i", ON)
    dev = make_device(client=client)
    dev.check_indi_props()
    assert abs(dev._normalized_wavelength - 908 / 656) < 1e-12


def test_dark_used_only_when_present():
    """Dark subtraction is requested only when a dark exists."""
    frame = make_frame()
    for dark in [True, False]:
        camera = FakeCamera(frame, dark=dark)
        dev = make_device(camera=camera)
        dev.grab_frame()
        assert camera.calls[-1][1] is dark


def test_camera_switch_only_when_idle():
    """The camera can be switched while idle, and the filter wheel follows it."""
    client = adc_client()
    dict.__setitem__(client, "fwsci2.filterName.i", ON)
    dev = make_device(client=client, camera=FakeCamera(make_frame()))
    send(dev, "state", adcLoop=ON)
    send(dev, "camera", camsci2=ON)
    assert dev._camera_name == "camsci1"
    send(dev, "state", idle=ON)
    send(dev, "camera", camsci2=ON)
    assert dev._camera_name == "camsci2"
    assert dev.camera is None
    assert dev.properties["camera"]["camsci2"] == ON
    dev.check_indi_props()
    assert abs(dev._normalized_wavelength - 762 / 656) < 1e-12


def test_default_gain():
    """The loop gain defaults to 0.1, and the gain property shows the value in use."""
    dev = make_device()
    assert dev._gain == 0.1
    assert dev.properties["gain"]["current"] == 0.1
    assert dev.properties["gain"]["target"] == 0.1


def test_ctrl_mtx_handler():
    """ctrl_mtx updates set the matching element as a float."""
    dev = make_device()
    send(dev, "ctrl_mtx", m00=1.5)
    send(dev, "ctrl_mtx", m01=-2.25)
    np.testing.assert_allclose(dev._control_mtx, [1.5, -2.25])
    assert dev.properties["ctrl_mtx"]["m01"] == -2.25


def test_state_change_aborts_batch():
    """Going idle during a batch aborts it without commanding."""
    dev = make_device()

    class SwitchingCamera(FakeCamera):
        def grab_stack(self, num_images, subtract_dark=True):
            send(dev, "state", idle=ON)
            return super().grab_stack(num_images, subtract_dark)

    dev.camera = SwitchingCamera(make_frame())
    send(dev, "state", adcLoop=ON)
    dev._no_measurements = 5
    dev.loop()
    assert len(dev.camera.calls) == 1
    assert dev.client.adc_writes() == []


def test_offset_and_reset_callbacks_do_not_block():
    """Offset and reset callbacks only queue work, which loop() applies."""
    client = adc_client(follow_adc=False)
    dev = make_device(client=client, send_timeout_sec=0.2)
    start = time.monotonic()
    send(dev, "offset", target=1.0)
    send(dev, "reset_deltaADCs", request=ON)
    assert time.monotonic() - start < 0.1
    assert client.adc_writes() == []
    dev.loop()
    assert ("adctrack.deltaADC1.target", 1.0) in client.writes
    assert dev.delta_1 == 0.0


def test_loop_sign_flips_command():
    """The loop_sign switch flips the sign of the commanded step."""
    frame = make_frame(dispersion=(0.02, 0.0))
    steps = []
    for sign in ["positive", "negative"]:
        dev = make_device(camera=FakeCamera(frame))
        send(dev, "loop_sign", **{sign: ON})
        send(dev, "state", oneshot=ON)
        dev.loop()
        steps.append(dev.delta_1)
    assert steps[0] != 0.0
    assert abs(steps[0] + steps[1]) < 1e-12
