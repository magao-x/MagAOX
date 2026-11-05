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

import io
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Run from apps/adcCtrl, following apps/aoSim/test
app_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(app_root))

import xapp.adcCtrl.app as app_module
from xapp.adcCtrl.app import (
    AdcCtrlConfig,
    KnifeEdgeParams,
    SpotGeometry,
    States,
    adcCtrl,
    compute_step,
    crop_image,
    expected_axes,
    inlier_mask,
    ke_blocked_spots,
    ke_sensitivity,
    ke_spot_order,
    ke_star_from_pair,
    locate_knife_edge_spots,
    measure_knife_edge,
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
    center=(0.0, 0.0),
):
    """Broadband PSF core plus four satellite spots, in camera pixels.

    `separation` is in lambda/D at the observing wavelength and `angle` rotates
    the spot pattern clockwise, matching speckle_cutout. `dispersion` is the
    core shift (x, y) in reference lambda/D across the full band. `center` is
    the star position (x, y) in reference lambda/D from the frame center.
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
        cx, cy = center[0] + dispersion[0] * frac, center[1] + dispersion[1] * frac
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


def frame_coords(frame):
    """Reference lambda/D coordinates (xx, yy) of a make_frame frame, origin at the frame center."""
    ny, nx = frame.shape
    x = (np.arange(nx) - (nx - 1) / 2) * PIXEL_SCALE
    y = (np.arange(ny) - (ny - 1) / 2) * PIXEL_SCALE
    return np.meshgrid(x, y)


def to_pixels(frame, point):
    """Frame pixel (x, y) of a point given in reference lambda/D from the frame center."""
    ny, nx = frame.shape
    return point[0] / PIXEL_SCALE + (nx - 1) / 2, point[1] / PIXEL_SCALE + (ny - 1) / 2


def make_ke_frame(separation=15.0, angle=17.0, center=(2.0, -3.0), edge=1.1, glint_amp=0.0, halo_amp=0.0,
                  noise=0.0, seed=0, **kwargs):
    """A make_frame frame behind a knife edge.

    Everything below a horizontal edge `edge` lambda/D above the star is
    blocked. Optional: a smooth halo, a glint line along the edge (fading
    over ~5 lambda/D to the sides of the star), and noise over the whole
    frame.
    """
    frame = make_frame(separation=separation, angle=angle, center=center, **kwargs)
    xx, yy = frame_coords(frame)
    if halo_amp:
        frame += halo_amp * np.exp(-np.hypot(xx - center[0], yy - center[1]) / 8.0)
    edge_y = center[1] + edge
    frame *= yy > edge_y
    if glint_amp:
        frame += glint_amp * np.exp(-((yy - edge_y) ** 2) / (2 * 0.3**2) - (xx - center[0]) ** 2 / (2 * 5.0**2))
    if noise > 0:
        frame += np.random.default_rng(seed).normal(0, noise, frame.shape)
    return frame


def ke_geometry(separation=15.0, angle=17.0):
    """Sparkle search geometry with the default config's window and search sizes."""
    config = AdcCtrlConfig()
    return SpotGeometry(
        separation, angle, config.sparkle_window_size, config.sparkle_search_extent, config.sparkle_search_halfwidth,
    )


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
    # XDevice.__del__ closes the telemetry file that __init__ would have opened
    dev._telem_file = io.StringIO()
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


def test_sparkle_search_ignores_lopsided_core_residual():
    """A bright residual just outside the core mask must not be picked as a sparkle.

    Reproduces the first on-instrument run (i-band, sparkles at 15 l/D, angle 0):
    with the old 20 l/D search box, spot 2's search reached ~1 l/D outside the
    core mask and locked onto the core residual instead of the sparkle.
    """
    wavelength = 762e-9
    separation = 15.0 * wavelength / 656e-9
    frame = make_frame(separation=15.0, wavelength=wavelength)
    x = (np.arange(frame.shape[1]) - (frame.shape[1] - 1) / 2) * PIXEL_SCALE
    xx, yy = np.meshgrid(x, x)
    frame += 80.0 * np.exp(-((xx - 0.5) ** 2 + (yy + 9.5) ** 2) / (2 * 0.8**2))

    config = AdcCtrlConfig()
    old = SpotGeometry(separation, 0.0, 20, 20.0)
    new = SpotGeometry(
        separation, 0.0, config.sparkle_window_size,
        config.sparkle_search_extent, config.sparkle_search_halfwidth,
    )
    _, dev_old = measure_spot_angles(frame, old, PIXEL_SCALE)
    _, dev_new = measure_spot_angles(frame, new, PIXEL_SCALE)
    assert abs(dev_old[2]) > 10.0
    assert np.all(np.abs(dev_new) < 1.0), dev_new


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
# Knife-edge algorithm
# --------------------------------------------------------------------------

def test_ke_spot_selection_geometry():
    """Spot selections are ordered b counter-clockwise of a; opposite pairs are refused."""
    assert ke_spot_order([0, 3]) == (0, 3)
    assert ke_spot_order([3, 0]) == (0, 3)
    assert ke_spot_order([0, 1]) == (1, 0)
    assert ke_spot_order([2]) == (2,)
    for bad in [[0, 2], [1, 3], [0, 1, 2], [], [4]]:
        try:
            ke_spot_order(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    # sparkles at 15-20 deg: spots 0 and 3 visible, spot 0 alone has low sensitivity
    assert ke_blocked_spots((0, 3), 17.0) == []
    assert ke_blocked_spots((1, 0), 17.0) == [1]
    assert abs(ke_sensitivity((0,), 15.0) - np.sin(np.radians(15.0))) < 1e-12
    assert abs(ke_sensitivity((0, 3), 15.0) - (np.sin(np.radians(15.0)) + np.cos(np.radians(15.0)))) < 1e-12
    # star at the right-angle corner
    star, sep = ke_star_from_pair([3.0, 10.0], [-10.0, 3.0])
    np.testing.assert_allclose(star, [0.0, 0.0], atol=1e-12)
    assert abs(sep - np.hypot(3.0, 10.0)) < 1e-12


def test_ke_star_estimate_needs_no_angle():
    """The two-spot star estimate is within 0.5 l/D for several angles and separations, with no angle given."""
    for separation, angle, spots in [(10.0, 15.0, [0, 3]), (15.0, 20.0, [0, 3]), (12.0, 28.0, [0, 3]),
                                     (15.0, -20.0, [0, 1])]:
        center = (2.0, -3.0)
        frame = make_ke_frame(separation=separation, angle=angle, center=center)
        result = measure_knife_edge(frame, ke_geometry(separation, angle=0.0), ke_spot_order(spots), PIXEL_SCALE)
        star = np.array(result.location.star)
        error = np.hypot(*(star - to_pixels(frame, center))) * PIXEL_SCALE
        assert error < 0.5, (separation, angle, error)
        assert abs(result.separation - separation) < 0.3, (separation, result.separation)
        assert abs(result.location.angle - angle) < 1.0, (angle, result.location.angle)


def test_ke_dm_spots():
    """DM spots at 47 l/D and 28 deg (i band) are found, with the star and deviations recovered."""
    config = AdcCtrlConfig()
    nwl = 762e-9 / 656e-9
    center = (3.0, -20.0)
    frame = make_ke_frame(size=480, separation=47.0, angle=28.0, wavelength=762e-9, center=center)
    geometry = SpotGeometry(
        47.0 * nwl, 28.0, config.dm_window_size, config.dm_search_extent, config.dm_search_halfwidth,
    )
    result = measure_knife_edge(frame, geometry, (0, 3), PIXEL_SCALE)
    assert np.hypot(*(np.array(result.location.star) - to_pixels(frame, center))) * PIXEL_SCALE < 0.5
    assert np.all(np.abs(result.dev[[0, 3]]) < 1.0), result.dev


def test_ke_glint_and_noise_do_not_capture_search():
    """A bright edge glint and noise in the blocked half do not capture the spot search."""
    center = (2.0, -3.0)
    frame = make_ke_frame(center=center, glint_amp=200.0, noise=1.0)
    result = measure_knife_edge(frame, ke_geometry(), ke_spot_order([0, 3]), PIXEL_SCALE)
    star = np.array(result.location.star)
    assert np.hypot(*(star - to_pixels(frame, center))) * PIXEL_SCALE < 0.5
    assert np.all(np.abs(result.dev[[0, 3]]) < 1.0), result.dev


def test_ke_bootstrap_rejects_distractors_and_wrong_separation():
    """The bootstrap finds the right pair among brighter blobs, and finds no pair at the wrong separation.

    The blob at (0, 32) is at the right spacing from spot 0 but much
    brighter, so it is rejected by the brightness ratio.
    """
    center = (2.0, -3.0)
    frame = make_ke_frame(center=center)
    xx, yy = frame_coords(frame)
    for bx, by in [(-28.0, 28.0), (22.0, 2.0), (0.0, 32.0)]:
        frame += 150.0 * np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * 0.45**2))
    geometry = ke_geometry()
    result = measure_knife_edge(frame, geometry, (0, 3), PIXEL_SCALE)
    assert np.hypot(*(np.array(result.location.star) - to_pixels(frame, center))) * PIXEL_SCALE < 0.5
    assert result.location.method == "bootstrap"

    try:
        locate_knife_edge_spots(
            make_ke_frame(center=center), (0, 3), 1.5 * 15.0 / PIXEL_SCALE, 17.0, PIXEL_SCALE, KnifeEdgeParams(),
            refine_half=10,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("a pair was found at 1.5x the separation")


def test_ke_tracking_follows_star():
    """After a lock, spots are tracked from the previous frame; a large jump falls back to the bootstrap."""
    geometry = ke_geometry()
    first = measure_knife_edge(make_ke_frame(center=(2.0, -3.0)), geometry, (0, 3), PIXEL_SCALE)
    moved = make_ke_frame(center=(2.5, -2.6))
    second = measure_knife_edge(moved, geometry, (0, 3), PIXEL_SCALE, prev=first.location)
    assert second.location.method == "tracked"
    assert np.hypot(*(np.array(second.location.star) - to_pixels(moved, (2.5, -2.6)))) * PIXEL_SCALE < 0.5
    far = make_ke_frame(center=(-8.0, 4.0))
    third = measure_knife_edge(far, geometry, (0, 3), PIXEL_SCALE, prev=first.location)
    assert third.location.method == "bootstrap"
    assert np.hypot(*(np.array(third.location.star) - to_pixels(far, (-8.0, 4.0)))) * PIXEL_SCALE < 0.5


def test_ke_deviations_one_and_two_spots():
    """Undispersed spots give deviations within 0.5 deg; rigid pattern rotation leaves two-spot m unchanged."""
    frame = make_ke_frame(angle=20.0)
    two = measure_knife_edge(frame, ke_geometry(angle=20.0), (0, 3), PIXEL_SCALE)
    assert np.all(np.abs(two.dev[[0, 3]]) < 0.5), two.dev
    assert np.all(np.isnan(two.dev[[1, 2]]))
    assert abs(two.m - (two.dev[0] - two.dev[3])) < 1e-12

    for spot in [0, 3]:
        one = measure_knife_edge(frame, ke_geometry(angle=20.0), (spot,), PIXEL_SCALE)
        assert abs(one.m) < 0.5, (spot, one.m)
        assert abs(one.m - two.dev[spot]) < 0.3

    # a 3 deg error in the configured angle (e.g. tweeterSpeck) cancels in the two-spot m only
    two_off = measure_knife_edge(frame, ke_geometry(angle=17.0), (0, 3), PIXEL_SCALE)
    one_off = measure_knife_edge(frame, ke_geometry(angle=17.0), (3,), PIXEL_SCALE)
    assert abs(two_off.m - two.m) < 0.3, (two_off.m, two.m)
    assert abs(abs(one_off.m) - 3.0) < 0.5, one_off.m


def test_ke_single_spot_seeded_by_star_hint():
    """One spot is found from a rough star hint even when a brighter blob is elsewhere."""
    center = (2.0, -3.0)
    frame = make_ke_frame(center=center)
    xx, yy = frame_coords(frame)
    frame += 150.0 * np.exp(-((xx + 25.0) ** 2 + (yy - 25.0) ** 2) / (2 * 0.45**2))
    hint = to_pixels(frame, (center[0] + 3.0, center[1] - 2.0))
    result = measure_knife_edge(frame, ke_geometry(), (3,), PIXEL_SCALE, star_hint=hint)
    assert result.location.method == "hint"
    assert abs(result.m) < 0.5
    assert np.hypot(*(np.array(result.location.star) - to_pixels(frame, center))) * PIXEL_SCALE < 0.5


def test_ke_half_plane_profile_removes_halo():
    """A half-plane halo is removed without biasing the spot angles by more than 0.5 deg."""
    clean = measure_knife_edge(make_ke_frame(), ke_geometry(), (0, 3), PIXEL_SCALE)
    halo = measure_knife_edge(make_ke_frame(halo_amp=20.0), ke_geometry(), (0, 3), PIXEL_SCALE)
    assert np.all(np.abs(halo.dev[[0, 3]] - clean.dev[[0, 3]]) < 0.5), (halo.dev, clean.dev)


def test_ke_margin_guard_keeps_clear_of_spot():
    """Spot 3 at 10 l/D, 15 deg sits 2.6 l/D above the star: the margin is capped and spot 3 is unbiased."""
    frame = make_ke_frame(separation=10.0, angle=15.0)
    geometry = ke_geometry(separation=10.0, angle=15.0)
    height = 10.0 * np.sin(np.radians(15.0))
    for edge_margin in [1.5, 5.0]:
        result = measure_knife_edge(frame, geometry, (0, 3), PIXEL_SCALE, KnifeEdgeParams(edge_margin=edge_margin))
        assert result.margin_capped
        assert abs(result.margin - (height - 1.5)) < 0.3, result.margin
        assert abs(result.dev[3]) < 0.5, result.dev
    wide = measure_knife_edge(make_ke_frame(separation=15.0, angle=20.0), ke_geometry(angle=20.0), (0, 3),
                              PIXEL_SCALE)
    assert not wide.margin_capped and wide.margin == 1.5


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


class RecordingHandler(logging.Handler):
    """Collects (level, message) pairs from a logger."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append((record.levelno, record.getMessage()))

    def count(self, level):
        return sum(1 for lvl, _ in self.records if lvl == level)


def test_startup_grace_and_camera_outage_logging():
    """Missing INDI devices are quiet during startup, then reported once per outage."""
    dev = make_device(client=FakeClient())
    dev.log = logging.getLogger("adcCtrl-test-grace")
    dev.log.setLevel(logging.DEBUG)
    handler = RecordingHandler()
    dev.log.addHandler(handler)

    def no_camera(*args, **kwargs):
        raise TimeoutError("INDI not up")

    original_xcam = app_module.XCam
    app_module.XCam = no_camera
    try:
        # during the startup grace period: nothing above DEBUG
        assert dev.in_startup_grace()
        dev.check_indi_props()
        assert not dev.open_camera()
        assert dev.properties["status"]["last_error"].startswith("could not open camera")
        assert handler.count(logging.WARNING) == 0
        assert handler.count(logging.ERROR) == 0

        # after the grace period: each missing key warns once, the outage errors once
        dev._start_time -= dev.config.startup_grace_sec + 1
        dev.check_indi_props()
        dev.check_indi_props()
        assert handler.count(logging.WARNING) == len(set(dev._warned_keys)) > 0
        assert not dev.open_camera()
        assert not dev.open_camera()
        assert handler.count(logging.ERROR) == 1

        # recovery is logged, and a later outage is reported again
        app_module.XCam = lambda *args, **kwargs: FakeCamera()
        assert dev.open_camera()
        app_module.XCam = no_camera
        assert not dev.open_camera()
        assert handler.count(logging.ERROR) == 2
    finally:
        app_module.XCam = original_xcam
        dev.log.removeHandler(handler)


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


def test_default_ctrl_mtx_and_loop_sign():
    """Defaults match the on-instrument DM-spot calibration: this matrix with negative feedback."""
    dev = make_device()
    np.testing.assert_allclose(dev._control_mtx, [-0.4156, -0.24515])
    assert dev.properties["ctrl_mtx"]["m00"] == -0.4156
    assert dev.properties["ctrl_mtx"]["m01"] == -0.24515
    assert dev._loop_sign == -1.0
    assert dev.properties["loop_sign"]["negative"] == ON
    assert dev.properties["loop_sign"]["positive"] == OFF

    dev = make_device(loop_sign="positive")
    assert dev._loop_sign == 1.0
    assert dev.properties["loop_sign"]["positive"] == ON


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


# --------------------------------------------------------------------------
# Knife-edge device behavior
# --------------------------------------------------------------------------

def ke_client(angle=17.0, **kwargs):
    """adc_client with tweeterSpeck sparkles at `angle`, as used with the knife edge."""
    client = adc_client(**kwargs)
    dict.__setitem__(client, "tweeterSpeck.angle.current", angle)
    return client


def recording_device(name, **kwargs):
    """make_device with a RecordingHandler attached to its logger; returns (device, handler)."""
    dev = make_device(**kwargs)
    dev.log = logging.getLogger(name)
    dev.log.setLevel(logging.DEBUG)
    dev.log.handlers = []
    handler = RecordingHandler()
    dev.log.addHandler(handler)
    return dev, handler


def messages(handler, level, text):
    """Messages logged at `level` containing `text`."""
    return [m for lvl, m in handler.records if lvl == level and text in m]


def test_coronagraph_switch_only_when_idle_and_keeps_delta1():
    """The coronagraph mode changes only while idle, and leaves delta_1 alone."""
    dev = make_device(client=ke_client())
    assert dev._knife_edge is False
    assert dev.properties["coronagraph"]["standard"] == ON
    dev.set_command(0.4, 0.0)
    send(dev, "state", adcLoop=ON)
    send(dev, "coronagraph", knife_edge=ON)
    assert dev._knife_edge is False
    send(dev, "state", idle=ON)
    send(dev, "coronagraph", knife_edge=ON)
    assert dev._knife_edge is True
    assert dev.properties["coronagraph"]["knife_edge"] == ON
    assert dev.properties["coronagraph"]["standard"] == OFF
    assert dev.delta_1 == 0.4
    send(dev, "coronagraph", standard=ON)
    assert dev._knife_edge is False
    assert dev.delta_1 == 0.4


def test_ke_spots_refuses_invalid_selections():
    """ke_spots accepts one spot or two adjacent spots, and refuses anything else keeping the previous selection."""
    dev = make_device(client=ke_client())
    assert dev._ke_order == (0, 3)
    assert [dev.properties["ke_spots"][f"spot{n}"] for n in range(4)] == [ON, OFF, OFF, ON]

    send(dev, "ke_spots", spot2=ON)  # 0, 2, 3
    assert dev._ke_order == (0, 3)
    assert dev.properties["ke_spots"]["spot2"] == OFF
    send(dev, "ke_spots", spot0=OFF)  # 3
    assert dev._ke_order == (3,)
    send(dev, "ke_spots", spot1=ON)  # 1, 3: opposite
    assert dev._ke_order == (3,)
    send(dev, "ke_spots", spot3=OFF)  # nothing
    assert dev._ke_order == (3,)
    send(dev, "ke_spots", spot0=ON, spot1=ON, spot2=OFF, spot3=OFF)  # full vector: 0, 1
    assert dev._ke_order == (1, 0)
    assert [dev.properties["ke_spots"][f"spot{n}"] for n in range(4)] == [ON, ON, OFF, OFF]

    try:
        make_device(ke_spots=[0, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("opposite spots accepted in the config")


def test_ke_spots_reports_but_accepts_blocked_and_low_sensitivity():
    """A blocked-half spot or a low-sensitivity selection is reported at INFO, not refused."""
    dev, handler = recording_device("adcCtrl-test-ke-spots", client=ke_client(angle=17.0))
    send(dev, "ke_spots", spot3=OFF)  # spot 0 alone: sensitivity sin(17) = 0.29
    assert dev._ke_order == (0,)
    assert len(messages(handler, logging.INFO, "low sensitivity")) == 1
    send(dev, "ke_spots", spot1=ON)  # spots 0 and 1: spot 1 is in the blocked half
    assert dev._ke_order == (1, 0)
    assert len(messages(handler, logging.INFO, "blocked half")) == 1
    assert handler.count(logging.WARNING) == 0


def test_entering_knife_edge_reports_selection():
    """Entering knife-edge mode reports a blocked or low-sensitivity selection."""
    dev, handler = recording_device("adcCtrl-test-ke-enter", client=ke_client(angle=17.0), ke_spots=[0])
    send(dev, "coronagraph", knife_edge=ON)
    assert len(messages(handler, logging.INFO, "low sensitivity")) == 1


def test_ke_gain_default_and_handler():
    """ke_gain defaults to the provisional -0.62 and can be changed through INDI."""
    dev = make_device()
    assert dev._ke_gain == -0.62
    assert dev.properties["ke_gain"]["current"] == -0.62
    send(dev, "ke_gain", target=-1.5)
    assert dev._ke_gain == -1.5
    assert dev.properties["ke_gain"]["current"] == -1.5


def test_ke_error_wraps_residual():
    """The set point is subtracted with a wrap at +-90 deg, then scaled by ke_gain."""
    residual, error = app_module.ke_error(89.0, -89.0, 2.0)
    assert abs(residual + 2.0) < 1e-12 and abs(error + 4.0) < 1e-12
    residual, error = app_module.ke_error(0.5, 0.2, -0.62)
    assert abs(residual - 0.3) < 1e-12 and abs(error + 0.186) < 1e-12


def test_ke_setpoint_manual_entry_and_expiry():
    """Entering knife-edge mode reports a config-default or expired set point; a manual entry restarts its age."""
    dev, handler = recording_device("adcCtrl-test-ke-age", client=ke_client(), ke_setpoint=0.25)
    assert dev._ke_setpoint == 0.25 and dev.properties["ke_setpoint"]["current"] == 0.25
    send(dev, "coronagraph", knife_edge=ON)
    assert len(messages(handler, logging.INFO, "config default")) == 1

    send(dev, "ke_setpoint", target=0.4)
    assert dev._ke_setpoint == 0.4
    assert dev.properties["ke_setpoint"]["current"] == 0.4
    send(dev, "coronagraph", standard=ON)
    send(dev, "coronagraph", knife_edge=ON)
    assert len(messages(handler, logging.INFO, "expired")) == 0
    assert len(messages(handler, logging.INFO, "config default")) == 1

    dev._ke_setpoint_time -= 25 * 3600
    send(dev, "coronagraph", standard=ON)
    send(dev, "coronagraph", knife_edge=ON)
    assert len(messages(handler, logging.INFO, "expired")) == 1


def test_ke_capture_sets_batch_mean():
    """A capture runs one batch from loop() and sets the set point to its mean m; it is refused when not allowed."""
    frame = make_ke_frame(dispersion=(0.0, 0.4))
    dev, handler = recording_device("adcCtrl-test-ke-capture", client=ke_client(), camera=FakeCamera(frame))
    dev._no_measurements = 3

    send(dev, "ke_capture_setpoint", request=ON)  # standard mode
    assert not dev._pending_capture
    send(dev, "coronagraph", knife_edge=ON)
    send(dev, "state", adcLoop=ON)
    send(dev, "ke_capture_setpoint", request=ON)  # closed loop
    assert not dev._pending_capture
    assert len(messages(handler, logging.WARNING, "capture refused")) == 2
    send(dev, "state", idle=ON)

    send(dev, "ke_capture_setpoint", request=ON)
    assert dev._pending_capture
    assert dev.properties["ke_capture_setpoint"]["request"] == OFF
    dev.loop()
    expected = measure_knife_edge(frame, ke_geometry(), (0, 3), PIXEL_SCALE).m
    assert abs(dev._ke_setpoint - expected) < 1e-6
    assert abs(dev._ke_setpoint) > 0.2  # dispersion makes it clearly non-zero
    assert dev.properties["ke_setpoint"]["current"] == dev._ke_setpoint
    assert dev._ke_setpoint_time is not None
    assert len(dev.camera.calls) == 3
    assert dev.client.adc_writes() == []
    assert len(messages(handler, logging.INFO, "captured knife-edge set point")) == 1
    assert dev.properties["measurement"]["ke_m"] == dev._ke_setpoint
    assert dev.properties["measurement"]["angle1"] == 0.0


class DispersionCamera(FakeCamera):
    """Camera whose frames have vertical dispersion proportional to the device's delta_1 offset.

    The dispersion is `scale * (delta_1 - delta_star)` reference lambda/D
    across the band, plus a fixed horizontal (co-rotation) term
    `horizontal`. `make` builds the frame from the dispersion.
    """

    def __init__(self, dev, make, delta_star=0.3, scale=1.0, horizontal=0.0):
        super().__init__()
        self.dev = dev
        self.make = make
        self.delta_star = delta_star
        self.scale = scale
        self.horizontal = horizontal

    def dispersion(self, delta_1):
        return (self.horizontal, self.scale * (delta_1 - self.delta_star))

    def grab_stack(self, num_images, subtract_dark=True):
        self.calls.append((num_images, subtract_dark))
        return self.make(dispersion=self.dispersion(self.dev.delta_1))


def test_ke_loop_converges_to_setpoint_not_zero():
    """Closed loop in knife-edge mode drives delta_1 to where m equals the captured set point.

    A fixed co-rotation term makes m non-zero at zero vertical dispersion,
    so converging to m = 0 would land elsewhere. ke_gain follows the
    documented rule, 2 / slope of m versus delta_1, and loop_sign stays
    negative.
    """
    dev = make_device(client=ke_client(), gain=0.3)
    camera = DispersionCamera(dev, make_ke_frame, horizontal=0.3)
    geometry = ke_geometry()

    deltas = np.array([-0.2, 0.3, 0.8])
    ms = [measure_knife_edge(make_ke_frame(dispersion=camera.dispersion(d)), geometry, (0, 3), PIXEL_SCALE).m
          for d in deltas]
    slope = np.polyfit(deltas, ms, 1)[0]
    assert abs(slope) > 1.0, slope

    dev.camera = camera
    send(dev, "ke_gain", target=2.0 / slope)
    send(dev, "coronagraph", knife_edge=ON)
    dev.set_command(camera.delta_star, 0.0)
    send(dev, "ke_capture_setpoint", request=ON)
    dev.loop()
    setpoint = dev._ke_setpoint
    zero_crossing = camera.delta_star - setpoint / slope
    assert abs(zero_crossing - camera.delta_star) > 0.1, (setpoint, slope)

    dev.set_command(camera.delta_star + 0.5, 0.0)
    send(dev, "state", adcLoop=ON)
    for _ in range(15):
        dev.loop()
    assert dev._state == States.CLOSED_LOOP
    assert abs(dev.delta_1 - camera.delta_star) < 0.05, dev.delta_1
    cycle = dev.telem_records[-1][1]
    assert cycle["mode"] == "knife_edge" and cycle["ke_spots"] == [0, 3]
    assert cycle["angles"][1] is None and abs(cycle["residual"]) < 0.2


def test_same_loop_sign_converges_in_both_modes():
    """With ctrl_mtx = 1/pair slopes and ke_gain = 2/m slope, negative loop_sign converges in both modes."""
    geometry = ke_geometry()
    std_geometry = SpotGeometry(15.0, 17.0, 20, 10.0, 5.0)
    deltas = np.array([-0.2, 0.3, 0.8])
    probe = DispersionCamera(None, None)
    pairs = np.array([pair_offsets(measure_spot_angles(make_frame(angle=17.0, dispersion=probe.dispersion(d)),
                                                       std_geometry, PIXEL_SCALE)[1]) for d in deltas])
    slopes = [np.polyfit(deltas, pairs[:, i], 1)[0] for i in range(2)]
    ms = [measure_knife_edge(make_ke_frame(dispersion=probe.dispersion(d)), geometry, (0, 3), PIXEL_SCALE).m
          for d in deltas]
    ke_slope = np.polyfit(deltas, ms, 1)[0]

    # the provisional ke_gain rule from the pair slopes (plan R2-3): s0 = S02/2, s3 = -S13/2
    provisional = 2.0 / (slopes[0] / 2 + slopes[1] / 2)
    assert abs(provisional * ke_slope / 2.0 - 1.0) < 0.1, (provisional, 2.0 / ke_slope)

    for knife_edge in [False, True]:
        dev = make_device(client=ke_client(), gain=0.3, ctrl_mtx=[1.0 / slopes[0], 1.0 / slopes[1]],
                          ke_gain=2.0 / ke_slope)
        assert dev._loop_sign == -1.0
        make = make_ke_frame if knife_edge else (lambda dispersion: make_frame(angle=17.0, dispersion=dispersion))
        dev.camera = DispersionCamera(dev, make)
        if knife_edge:
            send(dev, "coronagraph", knife_edge=ON)
        dev.set_command(0.8, 0.0)
        send(dev, "state", adcLoop=ON)
        for _ in range(15):
            dev.loop()
        assert abs(dev.delta_1 - 0.3) < 0.05, (knife_edge, dev.delta_1)
