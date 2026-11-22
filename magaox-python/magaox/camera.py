## \file camera.py
# \brief Acquire science frames, darks, and camera metadata for MagAO-X cameras.
"""Camera acquisition and INDI controls."""

import logging
import operator
import threading
import time
import warnings
from typing import Optional, Union

import numpy as np
import purepyindi2 as indi

from .constants import DEFAULT_SHMIM_TIMEOUT_SEC
from .shmim import Image, ShapeMismatchException, ShmimNotReady, ShmimTimeout, _lock_until

log = logging.getLogger(__name__)


class XCam:
    """A Python interface for MagAO-X cameras"""

    _roi_properties: tuple[str] = (
        "roi_region_h",
        "roi_region_w",
        "roi_region_bin_x",
        "roi_region_bin_y",
    )
    _roi_state: dict[str, float]
    grid: Optional["hcipy.Field"]

    def __init__(self, shm_name, pixel_size=1, indi_client=None, use_hcipy=False):
        """Attach by stream name, optionally sharing an INDI client and scaling an HCIPy grid."""
        # Connect the INDI client and acquire the initial device properties.
        self.shm_name = shm_name
        if indi_client is None:
            indi_client = indi.client.IndiClient()
            indi_client.connect()
        self._client = indi_client
        self._client.wait_to_connect()
        self._client.get_properties_and_wait(shm_name)

        # Serialize camera reads and explicit reconnects independently of ROI notifications.
        self._camera_lock = threading.RLock()
        self._pixel_size = pixel_size
        self._use_hcipy = use_hcipy
        self._grid_shape = None
        self.grid = None
        self.shmim = None
        self.dark_shmim = None
        self.connect_camera()

        # Track ROI telemetry separately from the stream attachment.
        self._roi_state = self._get_roi_state()
        self._client.register_callback(self._handle_indi_changes, device_name=self.shm_name)

    def _get_roi_state(self):
        """Collect the current ROI when the camera provides the complete property set."""
        if not all(k in self._client[self.shm_name] for k in self._roi_properties):
            return None
        return {k: self._client[self.shm_name][k]["current"] for k in self._roi_properties}

    def _handle_indi_changes(self, message):
        """Refresh cached ROI telemetry independently of producer reconfiguration."""
        if message.name in self._roi_properties:
            self._roi_state = self._get_roi_state()

    def create_grid(self, shape=None):
        """Cache an HCIPy grid in native (width, height) order."""
        # Standalone calls use the current geometry. Grabs supply the accepted frame geometry.
        with self._camera_lock:
            if self._use_hcipy:
                from hcipy import make_pupil_grid

                shape = self.shape if shape is None else shape
                if self._grid_shape != shape:
                    self.grid = make_pupil_grid(shape, self._pixel_size * np.array(shape))
                    self._grid_shape = shape

    def connect_camera(self):
        """Attach to the named camera and discover its optional dark stream."""
        # Verify the science stream before replacing the camera's existing handles.
        with self._camera_lock:
            science = Image(self.shm_name)
            try:
                dark = Image(self.shm_name + "_dark")
            except (FileNotFoundError, ShmimNotReady):
                log.debug("Dark stream %s_dark is unavailable", self.shm_name)
                dark = None
            except BaseException:
                science.close()
                raise

            # Release old mappings after the replacement handles are ready.
            if self.shmim is not None:
                self.shmim.close()
            if self.dark_shmim is not None:
                self.dark_shmim.close()
            self.shmim = science
            self.dark_shmim = dark
            self.create_grid()

    @property
    def counter(self):
        """Frame counter of the current science stream."""
        with self._camera_lock:
            return self.shmim.counter

    @property
    def meta_data(self):
        """Snapshot the camera's cached INDI properties and elements."""
        meta = {}
        this_device = self._client[self.shm_name]
        for prop in this_device:
            meta[prop] = {}
            for elem in this_device[prop]:
                meta[prop][elem] = this_device[prop][elem]
        return meta

    @property
    def shape(self):
        """Current science dimensions in (width, height) order."""
        with self._camera_lock:
            return self.shmim.shape[:2]

    @property
    def x(self):
        """Horizontal ROI binning reported by INDI."""
        return self._roi_state["roi_region_bin_x"]

    @property
    def y(self):
        """Vertical ROI binning reported by INDI."""
        return self._roi_state["roi_region_bin_y"]

    @property
    def width(self):
        """ROI width reported by INDI."""
        return self._roi_state["roi_region_w"]

    @property
    def height(self):
        """ROI height reported by INDI."""
        return self._roi_state["roi_region_h"]

    @property
    def exposure_time(self):
        """Exposure time in seconds."""
        return self._client[self.shm_name + ".exptime.current"]

    @exposure_time.setter
    def exposure_time(self, new_exposure_time):
        """Set the exposure target, using frame rate on cameras with only that control."""
        if "exptime" not in self._client[self.shm_name]:
            self.fps = 1 / new_exposure_time
        else:
            if new_exposure_time < 0:
                raise ValueError(f"Cannot set exposure time {new_exposure_time} sec")
            self._client[self.shm_name + ".exptime.target"] = new_exposure_time

    @property
    def fps(self):
        """Read the frame-rate control through the camera's INDI properties."""
        if "fps" not in self._client[self.shm_name]:
            return 1 / self._client[self.shm_name + ".exptime.current"]
        return self._client[self.shm_name + ".exptime.current"]

    @fps.setter
    def fps(self, new_fps):
        """Set the frame-rate target or its reciprocal exposure time."""
        if "fps" not in self._client[self.shm_name]:
            self.exposure_time = 1 / new_fps
        else:
            if new_fps < 0:
                raise ValueError(f"Cannot set FPS {new_fps}")
            self._client[self.shm_name + ".fps.target"] = new_fps

    @property
    def emgain(self):
        """Current electron-multiplication gain reported by the camera."""
        if "emgain" in self._client[self.shm_name]:
            return self._client[self.shm_name + ".emgain.current"]
        else:
            raise ValueError("This camera has no emgain.")

    @emgain.setter
    def emgain(self, new_emgain):
        """Set the electron-multiplication gain target."""
        if "emgain" in self._client[self.shm_name]:
            self._client[self.shm_name + ".emgain.target"] = new_emgain
        else:
            raise ValueError("This camera has no emgain.")

    @property
    def temperature(self):
        """Current detector temperature reported by the camera."""
        if "temp_ccd" in self._client[self.shm_name]:
            return self._client[self.shm_name + ".temp_ccd.current"]
        else:
            raise ValueError("This camera has no temperature monitor.")

    @temperature.setter
    def temperature(self, new_temperature):
        """Set the detector temperature target."""
        if "temp_ccd" in self._client[self.shm_name]:
            self._client[self.shm_name + ".temp_ccd.target"] = new_temperature
        else:
            raise ValueError("This camera has no temperature monitor.")

    @property
    def shutter(self):
        """Read the camera shutter toggle."""
        return (
            True
            if self.properties["shutter"]["toggle"] == indi.SwitchState.ON
            else False
        )

    @shutter.setter
    def shutter(self, shutter_state):
        """Set the camera shutter toggle."""
        self._client[self.shm_name + ".shutter.toggle"] = (
            indi.SwitchState.ON if shutter_state else indi.SwitchState.OFF
        )

    def process(self, data, subtract_dark, camera_shape=None):
        """Apply an optional dark and express the copied pixels on their acquisition grid."""
        with self._camera_lock:
            # Snapshot the dark once so the geometry check and subtraction use the same pixels.
            if subtract_dark:
                try:
                    if self.dark_shmim is None:
                        self.dark_shmim = Image(self.shm_name + "_dark")
                    dark, dark_attachment = self.dark_shmim._snapshot()
                except (FileNotFoundError, ShmimNotReady) as exc:
                    raise RuntimeError("No dark found, but subtract_dark=False was not supplied") from exc

                # Retain the science pixels while the dark producer catches up to the new ROI.
                dark_shape = dark_attachment.shape[:2]
                if dark.shape == data.shape and (camera_shape is None or dark_shape == camera_shape):
                    # Float64 science pixels already promote real camera darks during subtraction.
                    if data.dtype != np.float64 or np.iscomplexobj(dark):
                        dark = dark.astype(float, copy=False)
                    data = data - dark
                else:
                    warnings.warn(f"Science shape {data.shape} (camera {camera_shape}) and dark shape {dark.shape} (camera {dark_shape}) differ, skipping dark subtraction")

            # Use the accepted frame dimensions even if the producer has already resized again.
            if self._use_hcipy:
                from hcipy import Field

                self.create_grid(camera_shape)
                data = Field(data.ravel(), self.grid)
            return data

    def grab(self, subtract_dark=True, timeout_sec: Optional[float] = DEFAULT_SHMIM_TIMEOUT_SEC, check_before_wait: bool = False) -> Union[np.ndarray, "hcipy.Field"]:
        """Grab one fresh frame. timeout_sec includes reconnection and may be None to wait indefinitely.

        Set subtract_dark=False to read without a dark. Set check_before_wait=True
        to check for replacement before the first semaphore wait as well as during recovery.
        """
        # Carry the geometry from the mapping that supplied the science pixels.
        deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
        with _lock_until(self._camera_lock, deadline, self.shm_name):
            pixels, attachment = self.shmim._get_frame(deadline, check_before_wait)

            # Release the native-format snapshot before allocating the calibrated result.
            pixels = pixels.astype(float, copy=False)
            return self.process(pixels, subtract_dark, attachment.shape[:2])

    def grab_stack(self, num_images, subtract_dark=True, timeout_sec: Optional[float] = DEFAULT_SHMIM_TIMEOUT_SEC, check_before_wait: bool = True) -> Union[np.ndarray, "hcipy.Field"]:
        """Average num_images fresh frames from a single attachment and optionally subtract its dark.

        A replacement restarts the sum. Each read has at most timeout_sec, and the entire
        acquisition has at most num_images * timeout_sec. None allows an unlimited wait.
        check_before_wait controls the optional identity check before each frame's first wait.
        """
        # Bound both individual reads and the time spent restarting a partial stack.
        num_images = operator.index(num_images)
        if num_images < 1:
            raise ValueError("num_images must be positive")
        deadline = None if timeout_sec is None else time.monotonic() + num_images * timeout_sec

        # Only combine frames acquired through the same attachment, including same-size restarts.
        with _lock_until(self._camera_lock, deadline, self.shm_name):
            previous = None
            count = 0
            while count < num_images:
                frame_deadline = None
                if deadline is not None:
                    now = time.monotonic()
                    if now >= deadline:
                        raise ShmimTimeout(f"Timed out collecting {num_images} frames from {self.shm_name!r}")
                    frame_deadline = min(deadline, now + timeout_sec)

                # Count the first fresh replacement frame as the start of the new sum.
                pixels, attachment = self.shmim._get_frame(frame_deadline, check_before_wait)
                if attachment is not previous:
                    total = pixels.astype(float, copy=False)
                    previous = attachment
                    count = 1
                else:
                    total += pixels
                    count += 1

            # Apply the dark once to the completed average.
            total /= num_images
            return self.process(total, subtract_dark, attachment.shape[:2])
