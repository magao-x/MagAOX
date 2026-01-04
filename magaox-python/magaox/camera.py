import logging
import time
import warnings
from typing import Optional, Union

import numpy as np
import purepyindi2 as indi

from .constants import DEFAULT_SHMIM_TIMEOUT_SEC
from .shmim import Image, ShapeMismatchException

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
        self.shm_name = shm_name
        if indi_client is None:
            indi_client = indi.client.IndiClient()
            indi_client.connect()
        self._client = indi_client
        self._client.wait_to_connect()
        self._client.get_properties_and_wait(shm_name)

        self._old_counter = 0
        self._need_reconnect = False
        self._pixel_size = pixel_size
        self._use_hcipy = use_hcipy

        self.connect_camera()
        self._roi_state = self._get_roi_state()
        self._client.register_callback(
            self._handle_indi_changes, device_name=self.shm_name
        )

    def _get_roi_state(self):
        if not all(k in self._client[self.shm_name] for k in self._roi_properties):
            return None
        return {
            k: self._client[self.shm_name][k]["current"] for k in self._roi_properties
        }

    def _handle_indi_changes(self, message):
        if message.name in self._roi_properties:
            if (
                self._roi_state is not None
                and self._roi_state[message.name] != message["current"]
            ):
                log.debug(
                    "ROI property %s was %f, now is %f. Reconnecting to the shmim...",
                    (message.name, self._roi_state[message.name], message["current"]),
                )
                self._roi_state = self._get_roi_state()
                self.connect_camera()

    def create_grid(self):
        if self._use_hcipy:
            from hcipy import make_pupil_grid

            self.grid = make_pupil_grid(
                self.shape, self._pixel_size * np.array(self.shape)
            )
        else:
            self.grid = None

    def connect_camera(self):
        self.shmim = Image(self.shm_name)

        # A dark frame might not exist. Check for existence!
        self._dark_exists = False
        try:
            self.dark_shmim = Image(self.shm_name + "_dark")
            self._dark_exists = True
        except FileNotFoundError:
            log.warning(
                "Dark shmim '{:s}' does not exist.".format(self.shm_name + "_dark")
            )
        self.create_grid()

    @property
    def counter(self):
        return self.shmim.md.cnt0

    @property
    def meta_data(self):
        meta = {}
        this_device = self._client[self.shm_name]
        for prop in this_device:
            meta[prop] = {}
            for elem in this_device[prop]:
                meta[prop][elem] = this_device[prop][elem]
        return meta

    @property
    def shape(self):
        return (self.shmim.md.size[0], self.shmim.md.size[1])

    @property
    def x(self):
        return self._roi_state["roi_region_bin_x"]

    @property
    def y(self):
        return self._roi_state["roi_region_bin_y"]

    @property
    def width(self):
        return self._roi_state["roi_region_w"]

    @property
    def height(self):
        return self._roi_state["roi_region_h"]

    @property
    def exposure_time(self):
        return self._client[self.shm_name + ".exptime.current"]

    @exposure_time.setter
    def exposure_time(self, new_exposure_time):
        if "exptime" not in self._client[self.shm_name]:
            self.fps = 1 / new_exposure_time
        else:
            if new_exposure_time < 0:
                raise ValueError(f"Cannot set exposure time {new_exposure_time} sec")
            self._client[self.shm_name + ".exptime.target"] = new_exposure_time

    @property
    def fps(self):
        if "fps" not in self._client[self.shm_name]:
            return 1 / self._client[self.shm_name + ".exptime.current"]
        return self._client[self.shm_name + ".exptime.current"]

    @fps.setter
    def fps(self, new_fps):
        if "fps" not in self._client[self.shm_name]:
            self.exposure_time = 1 / new_fps
        else:
            if new_fps < 0:
                raise ValueError(f"Cannot set FPS {new_fps}")
            self._client[self.shm_name + ".fps.target"] = new_fps

    @property
    def emgain(self):
        if "emgain" in self._client[self.shm_name]:
            return self._client[self.shm_name + ".emgain.current"]
        else:
            raise ValueError("This camera has no emgain.")

    @emgain.setter
    def emgain(self, new_emgain):
        if "emgain" in self._client[self.shm_name]:
            self._client[self.shm_name + ".emgain.target"] = new_emgain
        else:
            raise ValueError("This camera has no emgain.")

    @property
    def temperature(self):
        if "temp_ccd" in self._client[self.shm_name]:
            return self._client[self.shm_name + ".temp_ccd.current"]
        else:
            raise ValueError("This camera has no temperature monitor.")

    @temperature.setter
    def temperature(self, new_temperature):
        if "temp_ccd" in self._client[self.shm_name]:
            self._client[self.shm_name + ".temp_ccd.target"] = new_temperature
        else:
            raise ValueError("This camera has no temperature monitor.")

    @property
    def shutter(self):
        return (
            True
            if self.properties["shutter"]["toggle"] == indi.SwitchState.ON
            else False
        )

    @shutter.setter
    def shutter(self, shutter_state):
        self._client[self.shm_name + ".shutter.toggle"] = (
            indi.SwitchState.ON if shutter_state else indi.SwitchState.OFF
        )

    def process(self, data, subtract_dark):
        if subtract_dark and not self._dark_exists:
            raise RuntimeError(
                "No dark found, but subtract_dark=False was not supplied"
            )
        elif self._dark_exists and subtract_dark:
            dark_shape = np.squeeze(self.dark_shmim).shape
            if dark_shape == data.shape:
                dark = self.dark_shmim.copy().astype(float)
                arr = data - dark
            else:
                warnings.warn(
                    f"Got {data.shape=} but {dark_shape=}, skipping dark subtraction"
                )
                arr = data
        else:
            arr = data

        if self._use_hcipy:
            from hcipy import Field
            # recreate grid for each image in case of SHMIM resizing.
            self.create_grid()
            arr = Field(arr.ravel(), self.grid)
        return arr

    def grab(
        self,
        subtract_dark=True,
        timeout_sec: Optional[float] = DEFAULT_SHMIM_TIMEOUT_SEC,
        check_before_wait: bool = False,
    ) -> Union[np.ndarray, "hcipy.Field"]:
        '''Grab one frame from the camera

        Parameters
        ----------
        subtract_dark : bool (default: True)
            Whether to subtract the currently loaded dark frame from
            the average. If the dark is not available, a `RuntimeError`
            is raised.
        timeout_sec : float or None (default: True)
            How long to wait (None to block until ready)
        check_before_wait : bool (default: True)
            Whether to check the identity of the shmim before beginning
            to wait. For a very modest speedup you can set
            `check_before_wait` to `False`, and `Image.get_data` won't
            verify the inode by statting the shmim file before trying
            to read. However, you will not know if the shmim has been
            recreated (e.g. with new dimensions). Always `True` if
            `timeout_sec` is `None`.
        '''
        self._old_counter = self.counter
        data = self.shmim.get_data(wait=True, timeout_sec=timeout_sec, check_before_wait=check_before_wait).astype(float)

        if self.counter == self._old_counter:
            self._need_reconnect = True
        else:
            self._old_counter = self.counter

        data = self.process(data, subtract_dark)
        return data

    def grab_stack(
        self,
        num_images,
        subtract_dark=True,
        timeout_sec: Optional[float] = DEFAULT_SHMIM_TIMEOUT_SEC,
        check_before_wait: bool = True,
    ) -> Union[np.ndarray, "hcipy.Field"]:
        """
        Grab `num_images` updates to the shmim and average them.

        Parameters
        ----------
        num_images : int
            How many (fresh) frames to capture
        subtract_dark : bool (default: True)
            Whether to subtract the currently loaded dark frame from
            the average. If the dark is not available, a `RuntimeError`
            is raised.
        timeout_sec : float or None (default: True)
            How long to wait (None to block until ready)
        check_before_wait : bool (default: True)
            Whether to check the identity of the shmim before beginning
            to wait. For a very modest speedup you can set
            `check_before_wait` to `False`, and `Image.get_data` won't
            verify the inode by statting the shmim file before trying
            to read. However, you will not know if the shmim has been
            recreated (e.g. with new dimensions). Always `True` if
            `timeout_sec` is `None`.
        """
        stacked_image = 0
        k = 0
        for i in range(num_images):
            self._old_counter = self.counter
            image = self.shmim.get_data(wait=True, timeout_sec=timeout_sec, check_before_wait=check_before_wait).astype(float)

            if self.counter == self._old_counter:
                self._need_reconnect = True
                stacked_image = 0
                k = 0
            else:
                self._old_counter = self.counter
                stacked_image += image
                k += 1

        if k != 0:
            stacked_image = stacked_image / k

        stacked_image = self.process(stacked_image, subtract_dark)
        return stacked_image
