"""Closed-loop ADC control from satellite spot pointing angles.

adcCtrl measures residual atmospheric dispersion in science camera images and
corrects it by sending counter-rotation offsets to the ADC tracker
(``adctrack.deltaADC1/2``), on top of adcTracker's model-based tracking.

Author: Katie Twitchell (twitchell@arizona.edu)

Method
------
Four satellite spots (active "sparkles" from tweeterSpeck, or passive DM
print-through spots) are elongated radially by the filter bandwidth. Residual
dispersion rotates opposite spots in opposite directions, so the difference of
their pointing angles (the pair offset) is proportional to the dispersion
(arXiv:2608.10307). For each frame the app:

1. pads the frame, places it on a lambda/D grid and median subtracts it,
2. centers the PSF, subtracts the radial profile and masks the core,
3. cuts out each spot at its expected position, scaled to the filter wavelength,
4. measures each spot's elongation axis with ``moment_angle`` (method of
   moments, verified in simulation; do not modify),
5. wraps each angle relative to the spot's nominal radial axis, so the result
   is signed and continuous across the +-90 degree seam,
6. forms pair offsets ``[d0 - d2, d1 - d3]`` and the dispersion error
   ``ctrl_mtx . pairs``.

A command averages ``n_avg`` frames per image and ``no_measurements`` images,
rejecting failed measurements and outliers. The step ``loop_sign * gain *
error`` is added to the integrator ``delta_1`` unless it exceeds
``step_limit_deg``, in which case it is rejected. Offsets are sent as
``deltaADC1 = delta_1 + delta_2 + offset`` and
``deltaADC2 = delta_1 - delta_2 + offset``; ``delta_2`` stays 0.

States
------
idle          no camera reads, no commands
adcLoop       measure and command every cycle (closed loop)
oneshot       one measure-and-command cycle, then back to idle
measure-only  measure and publish the would-be command, never send it

INDI properties
---------------
state            switch    idle / adcLoop / oneshot / measure-only
fsm              text      READY when idle, OPERATING otherwise
n_avg            number    frames averaged per image
no_measurements  number    images per command
gain             number    loop gain
offset           number    common offset added to both ADCs [deg]
ctrl_mtx         number    m00, m01: pair offsets -> dispersion error
loop_sign        switch    positive / negative feedback polarity (platform dependent)
satellite_spots  switch    sparkles / dm_spots
camera           switch    science camera (and its filter wheel); idle only
reset_deltaADCs  switch    request: zero the ADC offsets
measurement      number    (read-only) angles, deviations, pair offsets, error, n_valid, n_total
command          number    (read-only) last step, total delta1
status           text      (read-only) last command outcome and last error

Operator procedure
------------------
1. Select the camera and spot source, then run measure-only and check that
   the measurement property is stable.
2. Calibrate ctrl_mtx by hand: counter-rotate the ADCs by known offsets, fit
   the pair offset slopes, and enter their inverse as m00/m01.
3. Run oneshot to confirm the correction reduces the error. If it grows,
   flip loop_sign.
4. Run adcLoop.
"""

import time
from enum import Enum
from typing import NamedTuple, Optional

import numpy as np
import hcipy as hp
from scipy import ndimage

import xconf

from magaox.indi.device import XDevice, BaseConfig
from magaox.camera import XCam
from magaox.constants import StateCodes

from purepyindi2 import properties, constants
from purepyindi2.messages import DefNumber, DefSwitch, DefText

# Extra pixels removed from the second (masked) crop relative to the first,
# so the radial-profile edge region is excluded
SECOND_CROP_MARGIN = 25


def window_field(data, center, width, height, pad_value=0.0):
    """Cut a `width` x `height` window centered on `center` out of an hcipy Field.

    Regions of the window outside the input image are filled with `pad_value`,
    so the output always has the requested shape.
    """
    indx = data.grid.closest_to(center)
    ny_in, nx_in = data.shaped.shape
    y_ind, x_ind = np.unravel_index(indx, (ny_in, nx_in))

    x_min_ideal = x_ind - width // 2
    x_max_ideal = x_min_ideal + width
    y_min_ideal = y_ind - height // 2
    y_max_ideal = y_min_ideal + height

    x_min_valid = max(0, x_min_ideal)
    x_max_valid = min(nx_in, x_max_ideal)
    y_min_valid = max(0, y_min_ideal)
    y_max_valid = min(ny_in, y_max_ideal)

    out_x_min = x_min_valid - x_min_ideal
    out_x_max = out_x_min + (x_max_valid - x_min_valid)
    out_y_min = y_min_valid - y_min_ideal
    out_y_max = out_y_min + (y_max_valid - y_min_valid)

    cutout = np.full((height, width), pad_value, dtype=data.shaped.dtype)

    if (x_max_valid > x_min_valid) and (y_max_valid > y_min_valid):
        cutout[out_y_min:out_y_max, out_x_min:out_x_max] = data.shaped[
            y_min_valid:y_max_valid, x_min_valid:x_max_valid
        ]

    dx = data.grid.delta[0]
    dy = data.grid.delta[1]
    sub_grid = hp.make_pupil_grid([width, height], [width * dx, height * dy])

    return hp.Field(cutout.ravel(), sub_grid)


def crop_image(image, extent=100, mask_diam=0.5e-6):
    """Cut out a PSF-centered square of side `extent` pixels with the core masked.

    The center is the centroid of pixels above 10% of the peak. A circular mask
    of diameter `mask_diam` (grid units) is applied there, and negative pixels
    are floored to zero.
    """
    img_max = np.max(image)
    if img_max > 0:
        img_normalized = image / img_max
    else:
        img_normalized = image

    img_subtracted = img_normalized > 0.1
    total_sig = np.sum(img_subtracted)

    if total_sig > 0:
        center_of_intensity = np.array([
            np.sum(img_subtracted * img_subtracted.grid.x) / total_sig,
            np.sum(img_subtracted * img_subtracted.grid.y) / total_sig,
        ])
    else:
        center_of_intensity = np.array([0.0, 0.0])

    mask_ap = hp.make_circular_aperture(mask_diam, center_of_intensity)
    mask = np.abs(mask_ap(img_subtracted.grid) - 1.0)
    image = mask * image

    image = window_field(image, center_of_intensity, extent, extent)

    return np.maximum(image, 0.0)


def subtract_radial_profile(image, bin_size):
    """Subtract the azimuthally averaged profile (bins of `bin_size` grid units)."""
    binc, profile, _, _ = hp.radial_profile(image, bin_size)
    good = np.isfinite(profile)
    if not np.any(good):
        return image
    r_coordinates = image.grid.as_('polar').r
    radial_map = np.interp(r_coordinates, binc[good], profile[good])
    return image - radial_map


def speckle_cutout(img, speckle_number, angle, f=10, window_size=30, search_extent=20):
    """Cut out satellite spot `speckle_number` (0-3, counter-clockwise from top).

    The spot is searched for in a square box of side `search_extent` (grid
    units) centered `f` grid units from the image center, with the spot
    pattern rotated clockwise by `angle` degrees. Returns a `window_size`
    square array centered on the brightest pixel in that box.
    """
    extent = search_extent
    separation = f

    speckle_coords = np.array([
        [0, separation],
        [separation, 0],
        [0, -separation],
        [-separation, 0],
    ])
    speckle_center = speckle_coords[speckle_number]
    rect = hp.make_rotated_aperture(
        hp.make_rectangular_aperture(size=(extent, extent), center=speckle_center),
        np.deg2rad(-angle),
    )(img.grid)
    speckle_img = rect * img.copy()

    max_pixel = speckle_img.grid[np.argmax(speckle_img)]
    new_img = window_field(img, [max_pixel[0], max_pixel[1]], window_size, window_size)
    return np.array(new_img.shaped)


def moment_angle(
    crop: np.ndarray,
    fwhm_px: float = 6.0,
    max_iter: int = 5,
    tol_deg: float = 0.05,
) -> float:
    """Elongation axis of a single satellite spot by iterative weighted moments.

    This is the simulation-verified estimator from adc_sims/algo_26B/adc_ctrl.py
    and must not be changed.

    Parameters
    ----------
    crop : ndarray
        2D cutout containing one spot.
    fwhm_px : float
        Expected spot FWHM in pixels; sets the smoothing and window sizes.
    max_iter : int
        Maximum number of moment iterations.
    tol_deg : float
        Convergence tolerance on the angle, in degrees.

    Returns
    -------
    float
        Axis angle in degrees in (-90, 90], measured from the array x axis
        (columns) toward the y axis (rows). 0.0 if the crop has no signal.
    """

    # 0. Physical Positivity Enforcer (Fix for pre-subtracted negative pixels)
    # Floor raw input to zero immediately so negative background noise cannot act as "negative mass"
    crop_pos = np.maximum(crop.astype(np.float64), 0.0)
    ny, nx = crop_pos.shape
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]

    # 1. Estimate Residual Noise Floor on positive-floored signal
    border_pixels = np.concatenate(
        [
            crop_pos[0, :],
            crop_pos[-1, :],
            crop_pos[:, 0],
            crop_pos[:, -1],
        ]
    )

    # Estimate noise level using robust MAD on non-zero background
    bg_level = np.median(border_pixels)
    mad = np.median(np.abs(border_pixels - bg_level))
    sigma_noise = 1.4826 * mad if mad > 0 else 1

    # 2. Subtract residual local background and re-enforce non-negativity
    img_sub = np.maximum(crop_pos - bg_level, 0.0)

    # 2. Smooth to find initial 2D Peak & Estimate Peak SNR
    sigma_smooth = fwhm_px / 2.355
    smoothed = ndimage.gaussian_filter(np.maximum(img_sub, 0.0), sigma=sigma_smooth)
    peak_y, peak_x = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    peak_val = img_sub[peak_y, peak_x]

    snr_est = peak_val / sigma_noise if sigma_noise > 0 else 100.0

    # 3. Dynamic SNR Adjustments
    if snr_est < 15.0:
        # Low SNR regime (5-15): Aggressive noise cut, tight window
        snr_thresh_factor = 2.5
        min_win_scale = 0.8
        max_win_scale = 1.2
    elif snr_est < 50.0:
        # Mid SNR regime (15-50): Balanced settings
        snr_thresh_factor = 1.5
        min_win_scale = 0.8
        max_win_scale = 1.8
    else:
        # High SNR regime (>50): Relaxed threshold, wide window for accuracy
        snr_thresh_factor = 1.0
        min_win_scale = 1.0
        max_win_scale = 2.5

    # Threshold image based on estimated SNR
    img_clean = np.maximum(
        img_sub - (snr_thresh_factor * sigma_noise), 0.0
    )

    # Initial state
    x_bar, y_bar = float(peak_x), float(peak_y)
    angle = 0.0
    sigma_u = fwhm_px * 0.5
    sigma_v = fwhm_px * 0.5

    # 4. Iterative Moment Computation
    for iteration in range(max_iter):
        prev_angle = angle

        dx = x_grid - x_bar
        dy = y_grid - y_bar

        # Check centroid drift: Don't let iteration drift > 1 FWHM from smoothed peak
        dist_from_peak = np.sqrt((x_bar - peak_x) ** 2 + (y_bar - peak_y) ** 2)
        if dist_from_peak > fwhm_px:
            x_bar, y_bar = float(peak_x), float(peak_y)
            dx = x_grid - x_bar
            dy = y_grid - y_bar

        # Rotate to principal axis frame
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        u = dx * cos_a + dy * sin_a
        v = -dx * sin_a + dy * cos_a

        # Soft Elliptical Gaussian Window
        window = np.exp(-0.5 * ((u / sigma_u) ** 2 + (v / sigma_v) ** 2))
        weights = img_clean * window
        total_weight = np.sum(weights)

        if total_weight == 0:
            # Fallback for extreme noise: use raw peak-centered unweighted moments
            weights = np.maximum(img_clean, 0.0)
            total_weight = np.sum(weights)
            if total_weight == 0:
                return 0.0

        # Refine Centroid
        x_bar = np.sum(x_grid * weights) / total_weight
        y_bar = np.sum(y_grid * weights) / total_weight

        # Compute 2nd Central Moments
        dx_c = x_grid - x_bar
        dy_c = y_grid - y_bar

        mu20 = np.sum((dx_c**2) * weights)
        mu02 = np.sum((dy_c**2) * weights)
        mu11 = np.sum((dx_c * dy_c) * weights)

        # Unbiased Angle Calculation
        angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

        # Calculate shape variances
        var_u = 0.5 * (
            mu20 + mu02 + np.sqrt((mu20 - mu02) ** 2 + 4 * (mu11**2))
        )
        var_v = 0.5 * (
            mu20 + mu02 - np.sqrt((mu20 - mu02) ** 2 + 4 * (mu11**2))
        )

        # Adaptively update window dimensions with SNR-based bounds
        raw_sigma_u = np.sqrt(max(var_u / total_weight, 0.25))
        raw_sigma_v = np.sqrt(max(var_v / total_weight, 0.25))

        sigma_u = np.clip(
            raw_sigma_u, min_win_scale * fwhm_px, max_win_scale * fwhm_px
        )
        sigma_v = np.clip(
            raw_sigma_v, min_win_scale * fwhm_px, max_win_scale * fwhm_px
        )

        # Check convergence
        if (
            np.abs(np.degrees(angle - prev_angle)) < tol_deg
            and iteration > 0
        ):
            break

    return float(np.degrees(angle))


def wrap90(angle_deg):
    """Wrap an axis angle (degrees, 180-degree periodic) into [-90, 90)."""
    return (np.asarray(angle_deg, dtype=float) + 90.0) % 180.0 - 90.0


def expected_axes(grating_angle):
    """Nominal (radial) elongation axis in degrees of spots 0-3 with no dispersion."""
    return np.array([
        90.0 - grating_angle,
        -grating_angle,
        90.0 - grating_angle,
        -grating_angle,
    ])


def pair_offsets(deviations):
    """Pair offset angles [d0 - d2, d1 - d3] from per-spot axis deviations."""
    deviations = np.asarray(deviations, dtype=float)
    return np.array([deviations[0] - deviations[2], deviations[1] - deviations[3]])


class SpotGeometry(NamedTuple):
    """Where to look for the satellite spots in a frame."""
    separation: float  # spot distance from the PSF core, grid units (reference lambda/D)
    angle: float  # clockwise rotation of the spot pattern, degrees
    window_size: int  # side of the cutout handed to moment_angle, pixels
    search_extent: float  # side of the spot search box, grid units


def measure_spot_angles(frame, geometry, pixel_scale, pad=50, mask_factor=0.7, radial_bin=5 * 6.0 / 21.0):
    """Measure the elongation axis of all four satellite spots in one frame.

    Returns `(raw, deviations)`: the moment_angle results in degrees and their
    wrapped deviations from the nominal radial axes.
    """
    frame = np.asarray(frame, dtype=float)
    if frame.ndim != 2:
        raise ValueError(f"expected a 2D frame, got shape {frame.shape}")

    img = np.pad(frame, pad_width=pad, mode='constant', constant_values=0)
    ny, nx = img.shape
    grid = hp.make_pupil_grid([nx, ny], [nx * pixel_scale, ny * pixel_scale])
    field = hp.Field(img.ravel(), grid)
    field = field - np.median(field)

    crop_extent = min(nx, ny) - pad
    field = crop_image(field, crop_extent, mask_diam=0)
    field = subtract_radial_profile(field, radial_bin)
    field = crop_image(field, crop_extent - SECOND_CROP_MARGIN, mask_diam=mask_factor * geometry.separation)

    field = field - np.median(field)
    field = np.maximum(field, 0.0)

    raw = np.zeros(4)
    for n in range(4):
        cutout = speckle_cutout(
            field, n, geometry.angle, geometry.separation, geometry.window_size, geometry.search_extent
        )
        raw[n] = moment_angle(cutout)

    deviations = wrap90(raw - expected_axes(geometry.angle))
    return raw, deviations


def inlier_mask(values, k=3.0):
    """Boolean mask of finite `values` within k robust sigma (1.4826 MAD) of the median."""
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if not np.any(mask):
        return mask
    good = values[mask]
    med = np.median(good)
    mad = np.median(np.abs(good - med))
    if mad > 0:
        mask[mask] = np.abs(good - med) <= k * 1.4826 * mad
    else:
        mask[mask] = np.isclose(good, med)
    return mask


def compute_step(error, gain, sign, step_limit):
    """Return `(step, accepted)`: the gain-scaled step and whether it is within the step limit."""
    step = sign * gain * error
    accepted = bool(np.isfinite(step) and abs(step) < step_limit)
    return step, accepted


@xconf.config
class CameraConfig:
    """A science camera and the filter wheel in front of it."""
    shmim : str = xconf.field(help="Name of the camera device (and its shmim)")
    filter_wheel : str = xconf.field(help="INDI device name of the filter wheel used with this camera")


def _default_cameras():
    """Default camera table: camsci1 with fwsci1, camsci2 with fwsci2."""
    return {
        'camsci1': CameraConfig(shmim='camsci1', filter_wheel='fwsci1'),
        'camsci2': CameraConfig(shmim='camsci2', filter_wheel='fwsci2'),
    }


def _default_filter_wavelengths():
    """Default filter center wavelengths [m], keyed by filter wheel element name."""
    return {'r': 615e-9, 'i': 762e-9, 'z': 908e-9}


@xconf.config
class AdcCtrlConfig(BaseConfig):
    """Active ADC control

    Closed-loop correction of residual atmospheric dispersion from satellite
    spot pointing angles. Defaults reproduce the previous hard-coded values.
    """
    sleep_interval_sec : float = xconf.field(default=0.25, help="Sleep interval between loop() calls")
    cameras : dict[str, CameraConfig] = xconf.field(default_factory=_default_cameras, help="Selectable science cameras, keyed by INDI element name")
    default_camera : str = xconf.field(default='camsci1', help="Camera selected at startup (key of cameras)")
    adc_device : str = xconf.field(default='adctrack', help="INDI device name of the ADC tracker")
    speckle_device : str = xconf.field(default='tweeterSpeck', help="INDI device name of the active satellite spot (sparkle) generator")
    pixel_scale_lod : float = xconf.field(default=6.0 / 21.0, help="Pixel scale in lambda/D at the reference wavelength")
    reference_wavelength : float = xconf.field(default=656e-9, help="Reference wavelength [m] for the pixel scale, used when no listed filter is selected")
    filter_wavelengths : dict[str, float] = xconf.field(default_factory=_default_filter_wavelengths, help="Filter wheel element name to center wavelength [m]")
    dm_spot_separation : float = xconf.field(default=47.0, help="Passive DM spot separation in lambda/D at the observing wavelength")
    dm_spot_angle : float = xconf.field(default=28.0, help="Passive DM spot pattern rotation [deg]")
    dm_window_size : int = xconf.field(default=50, help="Passive DM spot cutout size [pixels]")
    dm_search_extent : float = xconf.field(default=30.0, help="Passive DM spot search box size [lambda/D]")
    sparkle_window_size : int = xconf.field(default=20, help="Sparkle cutout size [pixels]")
    sparkle_search_extent : float = xconf.field(default=20.0, help="Sparkle search box size [lambda/D]")
    mask_factor : float = xconf.field(default=0.7, help="Core mask diameter as a fraction of the spot separation")
    pad : int = xconf.field(default=50, help="Zero padding added around each frame [pixels]")
    radial_bin : float = xconf.field(default=5 * 6.0 / 21.0, help="Radial profile bin size [lambda/D]")
    gain : float = xconf.field(default=0.5, help="Initial loop gain")
    ctrl_mtx : list[float] = xconf.field(default_factory=lambda: [0.21178766, 0.19275196], help="Initial control matrix [m00, m01]")
    step_limit_deg : float = xconf.field(default=0.7, help="Steps with |step| at or above this [deg] are rejected")
    send_timeout_sec : float = xconf.field(default=30.0, help="Time to wait for the ADC stages to reach a commanded offset")
    send_tolerance_deg : float = xconf.field(default=0.05, help="Tolerance for a commanded offset to count as reached [deg]")
    max_consecutive_failures : int = xconf.field(default=5, help="Failed closed-loop cycles in a row before dropping to idle")
    outlier_k : float = xconf.field(default=3.0, help="Outlier rejection threshold in robust sigma")
    min_valid_fraction : float = xconf.field(default=0.5, help="Minimum fraction of valid measurements needed to act on a batch")
    camera_retry_sec : float = xconf.field(default=10.0, help="Interval between attempts to open the camera")


class States(Enum):
    """Operating states of the app."""
    IDLE = 0
    CLOSED_LOOP = 1
    ONESHOT = 2
    MEASURE_ONLY = 3


# INDI `state` switch element name -> operating state
STATE_ELEMENTS = {
    'idle': States.IDLE,
    'adcLoop': States.CLOSED_LOOP,
    'oneshot': States.ONESHOT,
    'measure-only': States.MEASURE_ONLY,
}

# INDI `loop_sign` switch element name -> multiplier applied to each step
LOOP_SIGNS = {'positive': 1.0, 'negative': -1.0}


def requested_switch(new_message, names):
    """Return the first of `names` switched ON in `new_message`, or None."""
    for name in names:
        if name in new_message and new_message[name] == constants.SwitchState.ON:
            return name
    return None


class adcCtrl(XDevice):
    """INDI device that measures residual dispersion and commands adctrack offsets."""

    config: AdcCtrlConfig

    def setup(self):
        """Create properties, subscribe to other devices and open the camera.

        Nothing here blocks on, or fails because of, a missing external device:
        the camera is retried from loop() and the initial ADC zeroing is queued.
        """
        self.init_state()
        self.create_properties()

        for device_name in self.external_devices():
            self.client.get_properties(device_name)

        self.open_camera()

        # zero the ADC offsets once adctrack is reachable
        self._pending_reset = True

        self.properties['fsm']['state'] = StateCodes.READY.name
        self.update_property(self.properties['fsm'])

    def init_state(self):
        """Initialize internal state from the configuration."""
        # Operating state, changed through the `state` switch
        self._state = States.IDLE
        # Frames averaged per image and images per command
        self._n_avg = 1
        self._no_measurements = 1
        # Loop gain and feedback polarity (+1/-1, from `loop_sign`)
        self._gain = float(self.config.gain)
        self._loop_sign = 1.0
        # Common offset added to both ADCs [deg]
        self._offset = 0.0
        # 1x2 control matrix mapping pair offsets to the dispersion error
        self._control_mtx = np.array(self.config.ctrl_mtx, dtype=float)
        # True for active sparkles, False for passive DM spots
        self._use_sparkles = True
        # Selected camera (key of config.cameras) and its XCam, None until opened
        self._camera_name = self.config.default_camera
        self.camera = None
        # monotonic time of the last camera open attempt, for retry pacing
        self._last_camera_attempt = None
        # One-time warning bookkeeping
        self._warned_no_dark = False
        self._warned_keys = set()
        # Failed cycles in a row; closed loop drops to idle at the configured limit
        self._consecutive_failures = 0
        # ADC writes requested by callbacks or setup, performed by loop()
        self._pending_reset = False
        self._pending_send = False
        # Integrated ADC offsets [deg]; delta_2 is currently always 0
        self.delta_1 = 0.0
        self.delta_2 = 0.0
        # Filter center wavelength / reference wavelength
        self._normalized_wavelength = 1.0
        # Sparkle separation [lambda/D] and pattern angle [deg] from tweeterSpeck
        self._sparkle_freq = 15.0
        self._sparkle_angle = 0.0

    def external_devices(self):
        """INDI devices this app reads from or writes to."""
        wheels = [cam.filter_wheel for cam in self.config.cameras.values()]
        return [self.config.adc_device, self.config.speckle_device] + wheels

    def create_properties(self):
        """Create all INDI properties."""
        fsm = properties.TextVector(name='fsm')
        fsm.add_element(DefText(name='state', _value=StateCodes.INITIALIZED.name))
        self.add_property(fsm)

        sv = properties.SwitchVector(
            name='state',
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        for name, state in STATE_ELEMENTS.items():
            sv.add_element(DefSwitch(name=name, _value=constants.SwitchState.ON if state == States.IDLE else constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_state)

        nv = properties.NumberVector(name='n_avg')
        nv.add_element(DefNumber(
            name='current', label='Number of frames', format='%i',
            min=1, max=150, step=1, _value=self._n_avg
        ))
        nv.add_element(DefNumber(
            name='target', label='Number of frames', format='%i',
            min=1, max=150, step=1, _value=self._n_avg
        ))
        self.add_property(nv, callback=self.handle_n_avg)

        nv = properties.NumberVector(name='no_measurements')
        nv.add_element(DefNumber(
            name='number', label='number', format='%i',
            min=1, max=100.00, step=1, _value=self._no_measurements
        ))
        self.add_property(nv, callback=self.handle_no_measurements)

        nv = properties.NumberVector(name='gain')
        nv.add_element(DefNumber(
            name='current', label='ADC Loop Gain', format='%.2f',
            min=0.00, max=1.00, step=0.01, _value=self._gain
        ))
        nv.add_element(DefNumber(
            name='target', label='ADC Loop Gain', format='%.2f',
            min=0.00, max=1.00, step=0.01, _value=self._gain
        ))
        self.add_property(nv, callback=self.handle_gain)

        nv = properties.NumberVector(name='offset')
        nv.add_element(DefNumber(
            name='current', label='offset', format='%.2f',
            min=-45, max=45, step=0.01, _value=self._offset
        ))
        nv.add_element(DefNumber(
            name='target', label='offset', format='%.2f',
            min=-45, max=45, step=0.01, _value=self._offset
        ))
        self.add_property(nv, callback=self.handle_offset)

        nv = properties.NumberVector(name='ctrl_mtx')
        nv.add_element(DefNumber(
            name='m00', label='m00', format='%.4f',
            min=-10.00, max=10.00, step=0.0001, _value=float(self._control_mtx[0])
        ))
        nv.add_element(DefNumber(
            name='m01', label='m01', format='%.4f',
            min=-10.00, max=10.00, step=0.0001, _value=float(self._control_mtx[1])
        ))
        self.add_property(nv, callback=self.handle_ctrl_mtx)

        sv = properties.SwitchVector(
            name='loop_sign',
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        sv.add_element(DefSwitch(name='positive', _value=constants.SwitchState.ON))
        sv.add_element(DefSwitch(name='negative', _value=constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_loop_sign)

        sv = properties.SwitchVector(
            name='satellite_spots',
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        sv.add_element(DefSwitch(name='sparkles', _value=constants.SwitchState.ON))
        sv.add_element(DefSwitch(name='dm_spots', _value=constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_spots)

        sv = properties.SwitchVector(
            name='camera',
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        for name in self.config.cameras:
            sv.add_element(DefSwitch(name=name, _value=constants.SwitchState.ON if name == self._camera_name else constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_camera)

        sv = properties.SwitchVector(
            name='reset_deltaADCs',
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        sv.add_element(DefSwitch(name='request', _value=constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_reset)

        nv = properties.NumberVector(name='measurement', perm=constants.PropertyPerm.READ_ONLY)
        for name in ['angle0', 'angle1', 'angle2', 'angle3', 'dev0', 'dev1', 'dev2', 'dev3', 'pair02', 'pair13', 'error']:
            nv.add_element(DefNumber(name=name, label=name, format='%.3f', min=-1e6, max=1e6, step=0, _value=0.0))
        nv.add_element(DefNumber(name='n_valid', label='n_valid', format='%i', min=0, max=1000, step=1, _value=0))
        nv.add_element(DefNumber(name='n_total', label='n_total', format='%i', min=0, max=1000, step=1, _value=0))
        self.add_property(nv)

        nv = properties.NumberVector(name='command', perm=constants.PropertyPerm.READ_ONLY)
        nv.add_element(DefNumber(name='step', label='last step', format='%.3f', min=-1e6, max=1e6, step=0, _value=0.0))
        nv.add_element(DefNumber(name='delta1', label='total delta1', format='%.3f', min=-1e6, max=1e6, step=0, _value=0.0))
        self.add_property(nv)

        tv = properties.TextVector(name='status', perm=constants.PropertyPerm.READ_ONLY)
        tv.add_element(DefText(name='last_command', _value='none'))
        tv.add_element(DefText(name='last_error', _value=''))
        self.add_property(tv)

    def ext(self, key, default=None):
        """Read an external INDI value, returning `default` (and warning once) if unavailable."""
        try:
            value = self.client[key]
        except Exception:
            value = None
        if value is None:
            if key not in self._warned_keys:
                self._warned_keys.add(key)
                self.log.warning(f'{key} is not available, using {default}')
            return default
        return value

    def open_camera(self):
        """Try to connect to the selected camera; returns True on success."""
        self._last_camera_attempt = time.monotonic()
        cam_cfg = self.config.cameras[self._camera_name]
        try:
            self.camera = XCam(
                cam_cfg.shmim,
                pixel_size=self.config.pixel_scale_lod,
                use_hcipy=False,
                indi_client=self.client,
            )
        except Exception as e:
            self.camera = None
            self.set_error(f'could not open camera {cam_cfg.shmim}: {e}')
            return False
        self._warned_no_dark = False
        self.log.info(f'Using camera {cam_cfg.shmim}')
        return True

    def check_indi_props(self):
        """Refresh the filter wavelength and sparkle geometry from other devices."""
        wheel = self.config.cameras[self._camera_name].filter_wheel
        wavelength = self.config.reference_wavelength
        for name, filter_wavelength in self.config.filter_wavelengths.items():
            if self.ext(f'{wheel}.filterName.{name}') == constants.SwitchState.ON:
                wavelength = filter_wavelength
                break
        self._normalized_wavelength = wavelength / self.config.reference_wavelength

        dev = self.config.speckle_device
        self._sparkle_freq = float(self.ext(f'{dev}.separation.current', self._sparkle_freq))
        self._sparkle_angle = float(self.ext(f'{dev}.angle.current', self._sparkle_angle))

        self.log.debug(f'normalized wavelength {self._normalized_wavelength:.3f}, sparkles {self._sparkle_freq} l/D at {self._sparkle_angle} deg')

    def spot_geometry(self):
        """Search geometry for the selected spot source at the current wavelength."""
        if self._use_sparkles:
            return SpotGeometry(
                separation=self._sparkle_freq * self._normalized_wavelength,
                angle=self._sparkle_angle,
                window_size=self.config.sparkle_window_size,
                search_extent=self.config.sparkle_search_extent,
            )
        return SpotGeometry(
            separation=self.config.dm_spot_separation * self._normalized_wavelength,
            angle=self.config.dm_spot_angle,
            window_size=self.config.dm_window_size,
            search_extent=self.config.dm_search_extent,
        )

    def set_status(self, last_command=None, last_error=None):
        """Update the read-only status property."""
        if last_command is not None:
            self.properties['status']['last_command'] = last_command
        if last_error is not None:
            self.properties['status']['last_error'] = last_error
        self.update_property(self.properties['status'])

    def set_error(self, message):
        """Log an error and show it in the status property."""
        self.log.error(message)
        self.set_status(last_error=message)

    def set_state(self, state):
        """Switch operating state, updating the state switch and fsm."""
        self._state = state
        for name, s in STATE_ELEMENTS.items():
            self.properties['state'][name] = constants.SwitchState.ON if s == state else constants.SwitchState.OFF
        self.properties['fsm']['state'] = StateCodes.READY.name if state == States.IDLE else StateCodes.OPERATING.name
        self.update_property(self.properties['state'])
        self.update_property(self.properties['fsm'])

    def transition_to_idle(self):
        """Return to the idle state."""
        self.set_state(States.IDLE)

    def handle_state(self, existing_property, new_message):
        """INDI callback for `state`; refreshes external context when leaving idle."""
        target = requested_switch(new_message, STATE_ELEMENTS)
        if target is None:
            self.update_property(existing_property)
            return
        state = STATE_ELEMENTS[target]
        if state != States.IDLE and self._state == States.IDLE:
            self.check_indi_props()
            self._consecutive_failures = 0
        self.set_state(state)
        self.log.debug(f'State changed to {target}')

    def handle_spots(self, existing_property, new_message):
        """INDI callback for `satellite_spots`: choose sparkles or DM spots."""
        target = requested_switch(new_message, ['sparkles', 'dm_spots'])
        if target is not None:
            for key in ['sparkles', 'dm_spots']:
                existing_property[key] = constants.SwitchState.ON if key == target else constants.SwitchState.OFF
            self._use_sparkles = (target == 'sparkles')
            self.check_indi_props()
            self.log.debug(f'using {target}')
        self.update_property(existing_property)

    def handle_camera(self, existing_property, new_message):
        """INDI callback for `camera`; allowed only while idle. The camera opens in loop()."""
        target = requested_switch(new_message, list(self.config.cameras))
        if target is not None and target != self._camera_name:
            if self._state != States.IDLE:
                self.log.warning('Camera can only be changed while idle')
            else:
                existing_property[self._camera_name] = constants.SwitchState.OFF
                existing_property[target] = constants.SwitchState.ON
                self._camera_name = target
                self.camera = None
                self._last_camera_attempt = None
                self.log.info(f'camera changed to {target}')
        self.update_property(existing_property)

    def handle_loop_sign(self, existing_property, new_message):
        """INDI callback for `loop_sign`: set the feedback polarity."""
        target = requested_switch(new_message, LOOP_SIGNS)
        if target is not None:
            for key in LOOP_SIGNS:
                existing_property[key] = constants.SwitchState.ON if key == target else constants.SwitchState.OFF
            self._loop_sign = LOOP_SIGNS[target]
            self.log.info(f'loop sign set to {target}')
        self.update_property(existing_property)

    def handle_reset(self, existing_property, new_message):
        """INDI callback for `reset_deltaADCs`: queue zeroing of the ADC offsets."""
        if 'request' in new_message and new_message['request'] == constants.SwitchState.ON:
            self.log.debug('resetting deltaADC properties')
            self._pending_reset = True
        existing_property['request'] = constants.SwitchState.OFF
        self.update_property(existing_property)

    def handle_n_avg(self, existing_property, new_message):
        """INDI callback for `n_avg`: frames averaged per image."""
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
            self._n_avg = int(new_message['target'])
            self.log.debug(f'now averaging over {self._n_avg} frames')
        self.update_property(existing_property)

    def handle_no_measurements(self, existing_property, new_message):
        """INDI callback for `no_measurements`: images per command."""
        if 'number' in new_message and new_message['number'] != existing_property['number']:
            existing_property['number'] = new_message['number']
            self._no_measurements = int(new_message['number'])
            self.log.debug(f'now averaging {self._no_measurements} measurements before sending command')
        self.update_property(existing_property)

    def handle_gain(self, existing_property, new_message):
        """INDI callback for `gain`."""
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
            self._gain = float(new_message['target'])
            self.log.debug(f'loop gain changed to {self._gain}')
        self.update_property(existing_property)

    def handle_offset(self, existing_property, new_message):
        """INDI callback for `offset`; the new offset is sent by loop()."""
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
            self._offset = float(new_message['target'])
            self._pending_send = True
            self.log.debug(f'offset changed to {self._offset}')
        self.update_property(existing_property)

    def handle_ctrl_mtx(self, existing_property, new_message):
        """INDI callback for `ctrl_mtx` (m00, m01)."""
        for index, key in enumerate(['m00', 'm01']):
            if key in new_message and new_message[key] != existing_property[key]:
                self._control_mtx[index] = float(new_message[key])
                existing_property[key] = self._control_mtx[index]
        self.log.debug(f'control matrix changed to {self._control_mtx}')
        self.update_property(existing_property)

    def set_command(self, d1, d2):
        """Set the integrated offsets (not sent until send_command)."""
        self.delta_1 = d1
        self.delta_2 = d2

    def add_command(self, d1, d2):
        """Add to the integrated offsets (not sent until send_command)."""
        self.delta_1 += d1
        self.delta_2 += d2

    def send_command(self):
        """Write the ADC offsets to adctrack and wait (bounded) for the stages.

        Returns True if the offsets were written and either reached or not
        expected to move (tracking off), False otherwise.
        """
        dev = self.config.adc_device
        target_1 = self.delta_1 + self.delta_2 + self._offset
        target_2 = self.delta_1 - self.delta_2 + self._offset
        try:
            self.client[f'{dev}.deltaADC1.target'] = target_1
            self.client[f'{dev}.deltaADC2.target'] = target_2
        except Exception as e:
            self.set_error(f'could not write ADC offsets: {e}')
            return False

        self.properties['command']['delta1'] = self.delta_1
        self.update_property(self.properties['command'])

        if self.ext(f'{dev}.tracking.toggle') != constants.SwitchState.ON:
            self.log.debug(f'{dev} tracking is off: deltaADC1/2 written but the stages will not move')
            return True

        tolerance = self.config.send_tolerance_deg
        deadline = time.monotonic() + self.config.send_timeout_sec
        while time.monotonic() < deadline:
            current_1 = self.ext(f'{dev}.deltaADC1.current')
            current_2 = self.ext(f'{dev}.deltaADC2.current')
            if (current_1 is not None and current_2 is not None
                    and abs(current_1 - target_1) < tolerance and abs(current_2 - target_2) < tolerance):
                return True
            time.sleep(0.05)

        self.set_error(f'ADC offsets not reached within {self.config.send_timeout_sec} s')
        return False

    def adc_available(self):
        """True if the ADC tracker's offset properties are visible."""
        dev = self.config.adc_device
        try:
            self.client[f'{dev}.deltaADC1.current']
            self.client[f'{dev}.deltaADC2.current']
        except Exception:
            return False
        return True

    def apply_pending(self):
        """Perform ADC writes requested by callbacks or at startup."""
        if not (self._pending_reset or self._pending_send):
            return
        if not self.adc_available():
            return
        if self._pending_reset:
            self.set_command(0, 0)
        self._pending_reset = False
        self._pending_send = False
        self.send_command()

    def grab_frame(self):
        """Grab and average `n_avg` frames, dark subtracted if a dark exists."""
        subtract_dark = bool(getattr(self.camera, '_dark_exists', False))
        if not subtract_dark and not self._warned_no_dark:
            self._warned_no_dark = True
            self.log.warning('No dark available, using median subtraction only')
        frame = self.camera.grab_stack(self._n_avg, subtract_dark=subtract_dark)
        frame = np.asarray(frame, dtype=float)
        if frame.ndim != 2:
            raise RuntimeError('no new frames received from camera')
        return frame

    def measure_dispersion(self):
        """Measure dispersion over `no_measurements` images.

        Returns a dict of averaged results, or None if the batch was aborted or
        had too few valid measurements.
        """
        start_state = self._state
        geometry = self.spot_geometry()
        n_total = self._no_measurements
        raws, devs, pairs, errors = [], [], [], []

        for k in range(n_total):
            if self._state != start_state:
                self.log.info('state changed, measurement batch aborted')
                return None
            try:
                frame = self.grab_frame()
                raw, dev = measure_spot_angles(
                    frame, geometry, self.config.pixel_scale_lod,
                    pad=self.config.pad, mask_factor=self.config.mask_factor, radial_bin=self.config.radial_bin,
                )
                pair = pair_offsets(dev)
                error = float(self._control_mtx @ pair)
                if not np.isfinite(error):
                    raise ValueError('non-finite dispersion measurement')
            except Exception as e:
                self.log.warning(f'measurement {k + 1}/{n_total} failed: {e}')
                continue
            self.log.debug(f'measured speckle angles: {raw}, deviations: {dev}, error: {error}')
            raws.append(raw)
            devs.append(dev)
            pairs.append(pair)
            errors.append(error)

        mask = inlier_mask(errors, self.config.outlier_k) if errors else np.zeros(0, dtype=bool)
        n_valid = int(np.sum(mask))
        if n_total == 0 or n_valid / n_total < self.config.min_valid_fraction:
            self.log.warning(f'only {n_valid}/{n_total} valid measurements')
            return None

        result = {
            'raw': np.mean(np.array(raws)[mask], axis=0),
            'dev': np.mean(np.array(devs)[mask], axis=0),
            'pairs': np.mean(np.array(pairs)[mask], axis=0),
            'error': float(np.mean(np.array(errors)[mask])),
            'n_valid': n_valid,
            'n_total': n_total,
        }
        self.publish_measurement(result)
        return result

    def publish_measurement(self, result):
        """Show a measurement result in the measurement property and telemetry."""
        prop = self.properties['measurement']
        for n in range(4):
            prop[f'angle{n}'] = float(result['raw'][n])
            prop[f'dev{n}'] = float(result['dev'][n])
        prop['pair02'] = float(result['pairs'][0])
        prop['pair13'] = float(result['pairs'][1])
        prop['error'] = result['error']
        prop['n_valid'] = result['n_valid']
        prop['n_total'] = result['n_total']
        self.update_property(prop)

    def cycle_failed(self, message):
        """Record a failed cycle; drop to idle after too many in closed loop."""
        self._consecutive_failures += 1
        self.set_status(last_command='skipped', last_error=message)
        self.log.warning(f'cycle failed ({self._consecutive_failures}): {message}')
        if (self._state == States.CLOSED_LOOP
                and self._consecutive_failures >= self.config.max_consecutive_failures):
            self.log.error(f'{self._consecutive_failures} consecutive failed cycles, going idle')
            self.transition_to_idle()

    def run_cycle(self):
        """One measure (and possibly command) cycle for the current state."""
        state = self._state
        self.check_indi_props()
        result = self.measure_dispersion()
        if self._state != state:
            return
        if result is None:
            self.cycle_failed('no valid measurement')
            return
        self._consecutive_failures = 0

        error = result['error']
        step, accepted = compute_step(error, self._gain, self._loop_sign, self.config.step_limit_deg)
        self.properties['command']['step'] = step
        self.update_property(self.properties['command'])
        self.telem('adcctrl_cycle', {
            'state': state.name,
            'angles': result['raw'].tolist(),
            'deviations': result['dev'].tolist(),
            'pairs': result['pairs'].tolist(),
            'error': error,
            'step': step,
            'accepted': accepted,
            'n_valid': result['n_valid'],
            'n_total': result['n_total'],
        })

        if state == States.MEASURE_ONLY:
            self.log.info(f'measured error {error:.4f}, step {step:.4f} (measured, not sent)')
            self.set_status(last_command='measure-only')
            return

        if not accepted:
            self.log.info(f'ADC step {step:.4f} exceeds acceptable threshold and was not sent')
            self.set_status(last_command='rejected')
            return

        self.add_command(step, 0)
        sent = self.send_command()
        self.log.info(f'delta command: {step:.4f}, total command: {self.delta_1:.4f}')
        self.set_status(last_command='sent' if sent else 'send failed')

    def loop(self):
        """Main loop body: apply queued ADC writes, then run a cycle unless idle.

        Never raises; errors are logged and counted as failed cycles. A one-shot
        always ends in idle, whether or not its cycle succeeded.
        """
        try:
            self.apply_pending()
            if self._state == States.IDLE:
                if self.camera is None:
                    self.retry_camera()
                return

            state = self._state
            try:
                if self.camera is None and not self.retry_camera():
                    self.cycle_failed('camera unavailable')
                    return
                self.run_cycle()
            finally:
                if state == States.ONESHOT and self._state == States.ONESHOT:
                    self.transition_to_idle()
        except Exception as e:
            self.log.exception('unexpected error in loop')
            self.cycle_failed(f'unexpected error: {e}')

    def retry_camera(self):
        """Re-open the camera if the retry interval has passed; returns True if open."""
        if self.camera is not None:
            return True
        if (self._last_camera_attempt is not None
                and time.monotonic() - self._last_camera_attempt < self.config.camera_retry_sec):
            return False
        return self.open_camera()


# Used to make the pyproject.toml just a little simpler,
# with fewer repetitions of the app name:
main = adcCtrl.console_app
