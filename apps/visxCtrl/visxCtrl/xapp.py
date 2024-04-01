import xconf
import numpy as np
import warnings
from typing import Optional
import datetime
from astropy.io import fits
import os
from purepyindi2 import device, properties, constants
from purepyindi2.messages import DefNumber, DefSwitch, DefText, DefLight
from magaox.indi.device import XDevice, BaseConfig
import ImageStreamIOWrap as ISIO
import logging
import time
import sys

from .qhyccd import QHYCCDSDK, QHYCCDCamera

log = logging.getLogger(__name__)

EXTERNAL_RECORDED_PROPERTIES = {
	'tcsi.catalog.object': 'OBJECT',
    'tcsi.catdata.ra': None,
    'tcsi.catdata.dec': None,
    'tcsi.catdata.epoch': None,
    'observers.current_observer.full_name': 'OBSERVER',
    'tcsi.teldata.pa': 'PARANG',
    'flipacq.position.in': None,
}

RECORDED_WHEELS = ('fwfpm', 'fwlyot')

CAMERA_CONNECT_RETRY_SEC = 5

def find_active_filter(client, fwname):
    fwelems = client[f"{fwname}.filterName"]
    if fwelems is None:
        return
    for elem in fwelems:
        if fwelems[elem] == constants.SwitchState.ON:
            return elem

@xconf.config
class VisXConfig(BaseConfig):
    full_sdk_path : str = xconf.field(default='/usr/local/lib/libqhyccd.so')
    temp_on_target_pct_diff : float = xconf.field(default=0.05, help="Absolute percent difference between temperature setpoint and currently reported value")

class VisX(XDevice):
    config : VisXConfig
    # us
    data_directory : str = "/opt/MagAOX/rawimages/camvisx"
    exposure_start_ts : float = 0
    should_cancel : bool = False
    currently_exposing : bool = False
    should_begin_exposure : bool = False
    last_image_filename : Optional[str] = None
    shmim : ISIO.Image
    frame : np.ndarray
    exposure_start_telem : Optional[dict] = None
    # them
    sdk : Optional[QHYCCDSDK] = None
    camera : Optional[QHYCCDCamera] = None
    exposure_time_sec : Optional[float] = None
    camera_gain : Optional[int] = None
    temp_target_deg_c : Optional[float] = None
    temp_current_deg_c : Optional[float] = None

    @property
    def cooling_on_target(self):
        temp_on_target = abs((self.camera.temperature - self.camera.target_temperature) / self.camera.target_temperature) < self.config.temp_on_target_pct_diff
        return temp_on_target

    def emit_telem_stdcam(self):
        w, h = 9600, 6422
        self.telem("telem_stdcam", {
            "roi": {
                "xcen": (w - 1) / 2,
                "ycen": (h - 1) / 2,
                "w": w,
                "h": h,
                "xbin": 1,
                "ybin": 1,
            },
            "exptime": self.exposure_time_sec,
            "fps": 1/self.exposure_time_sec,
            "emGain": self.camera.gain,
            "adcSpeed": -1,
            "tempCtrl": {
                "temp": self.camera.temperature,
                "setpt": self.camera.target_temperature,
                "status": True,
                "ontarget": self.cooling_on_target,
                "statusStr": "LOCKED" if self.cooling_on_target else "UNLOCKED",
            },
            "shutter": {"statusStr": None, "state": None},
            "synchro": 0,
            "vshift": -1,
            "cropMode": 0
        })

    def handle_exptime(self, existing_property, new_message):
        log.debug(f"In handle_exptime")
        if self.currently_exposing:
            self.log.debug("Ignoring exposure time change request while currently exposing")
        elif self.sdk is None:
            self.log.debug("Ignoring exposure time change while we don't have an SDK handle")
        elif not self.currently_exposing and 'target' in new_message and new_message['target'] != existing_property['current']:
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
            self.exposure_time_sec = new_message['target']
            self.camera.exposure_time = self.exposure_time_sec
            self.log.debug(f"Exposure time changed to {new_message['target']} seconds")
            self.telem('exptime', {'exptime': self.exposure_time_sec})
        self.update_property(existing_property)
    
    def handle_gain(self, existing_property, new_message):
        log.debug(f"In handle_gain")
        if self.currently_exposing:
            self.log.debug("Ignoring gain change request while currently exposing")
        elif self.sdk is None:
            self.log.debug("Ignoring gain change while we don't have an SDK handle")
        elif not self.currently_exposing and 'target' in new_message and new_message['target'] != existing_property['current']:
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
            self.camera_gain = new_message['target']
            self.camera.gain = self.camera_gain
            self.log.debug(f"Gain changed to {new_message['target']}")
            self.telem('emGain', {'emGain': self.camera_gain})
        self.update_property(existing_property)
    
    def handle_expose(self, existing_property, new_message):
        if self.sdk is None:
            self.log.debug("Ignoring request for exposure while we don't have an SDK handle")
        elif 'request' in new_message and new_message['request'] is constants.SwitchState.ON:
            self.log.debug("Exposure requested!")
            self.should_begin_exposure = True
        elif 'cancel' in new_message and new_message['cancel'] is constants.SwitchState.ON:
            self.log.debug("Exposure cancellation requested")
            self.should_cancel = True
        self.update_property(existing_property)  # ensure the switch turns back off at the client

    def handle_temp_ccd(self, existing_property, new_message):
        if self.sdk is None:
            self.log.debug("Ignoring temperature setpoint change while we don't have an SDK handle")
        elif 'target' in new_message and new_message['target'] != existing_property['current']:
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
            self.temp_target_deg_c = new_message['target']
            self.log.debug(f"CCD temperature setpoint changed to {self.temp_target_deg_c} deg C")
            self.telem('tempcontrol', {'temp_target_deg_c': self.temp_target_deg_c})
        self.update_property(existing_property)

    def _init_camera(self):
        # Load SDK
        self.sdk = QHYCCDSDK(dll_path=self.config.full_sdk_path)
        if self.sdk.number_of_cameras < 1:
            del self.sdk
            return False
        # Find camera
        self.camera = QHYCCDCamera(self.sdk, 0)
        self.exposure_time_sec = self.camera.exposure_time
        self.temp_target_deg_c = self.camera.target_temperature
        self.camera_gain = self.camera.gain
        return True

    def _init_properties(self):
        fsmstate = properties.TextVector(
            name='fsm',
        )
        fsmstate.add_element(DefText(name="state", _value="NODEVICE"))
        self.add_property(fsmstate)

        tv = properties.TextVector(
            name='last_frame',
        )
        tv.add_element(DefText(name="filename", _value=None))
        self.add_property(tv)
        sv = properties.SwitchVector(
            name='expose',
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        sv.add_element(DefSwitch(name="request", _value=constants.SwitchState.OFF))
        sv.add_element(DefSwitch(name="cancel", _value=constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_expose)

        nv = properties.NumberVector(name='exptime', perm=constants.PropertyPerm.READ_WRITE)
        nv.add_element(DefNumber(
            name='current', label='Exposure time (sec)', format='%3.1f',
            min=0, max=1_000_000, step=0.001, _value=self.exposure_time_sec
        ))
        nv.add_element(DefNumber(
            name='target', label='Requested exposure time (sec)', format='%3.1f',
            min=0, max=1_000_000, step=0.001, _value=self.exposure_time_sec
        ))
        self.add_property(nv, callback=self.handle_exptime)

        nv = properties.NumberVector(name='gain', perm=constants.PropertyPerm.READ_WRITE)
        nv.add_element(DefNumber(
            name='current', label='Gain', format='%d',
            min=0, max=100, step=1, _value=self.camera_gain
        ))
        nv.add_element(DefNumber(
            name='target', label='Requested gain', format='%d',
            min=0, max=100, step=1, _value=self.camera_gain
        ))
        self.add_property(nv, callback=self.handle_gain)

        nv = properties.NumberVector(name='temp_ccd', perm=constants.PropertyPerm.READ_WRITE)
        nv.add_element(DefNumber(
            name='current', label='Current temperature (deg C)', format='%3.3f',
            min=-100, max=100, step=0.1, _value=self.temp_target_deg_c
        ))
        nv.add_element(DefNumber(
            name='target', label='Requested temperature (deg C)', format='%3.3f',
            min=-100, max=100, step=0.1, _value=self.temp_target_deg_c
        ))
        self.add_property(nv, callback=self.handle_temp_ccd)

        nv = properties.NumberVector(name='current_exposure')
        nv.add_element(DefNumber(
            name='remaining_sec', label='Time remaining (sec)', format='%3.3f',
            min=0, max=1_000_000, step=1, _value=0.0
        ))
        nv.add_element(DefNumber(
            name='remaining_pct', label='Percentage remaining', format='%i',
            min=0, max=100, step=0.1, _value=0.0
        ))
        self.add_property(nv)

    def subscribe_to_other_devices(self):
        devices = set()
        for prop in EXTERNAL_RECORDED_PROPERTIES:
            device = prop.split('.')[0]
            devices.add(device)
            self.log.debug(f"subscribe to device: {device}")
        for fw in RECORDED_WHEELS:
            devices.add(fw)
        try:
            self.client.get_properties_and_wait(devices)
        except TimeoutError as e:
            log.warning(f"Timed out waiting to get properties from external INDI devices: {e}")

    def setup(self):
        os.makedirs(self.data_directory, exist_ok=True)
        while self.client.status is not constants.ConnectionStatus.CONNECTED:
            self.log.info(f"Connecting to INDI as a client to get {list(EXTERNAL_RECORDED_PROPERTIES.keys())} and {RECORDED_WHEELS}...")
            time.sleep(1)
        self.log.info(f"INDI client connection: {self.client.status}")
        self.subscribe_to_other_devices()

        self._init_properties()
        self.properties['fsm']['state'] = 'NOTCONNECTED'
        self.log.debug("Set FSM prop")
        self.update_property(self.properties['fsm'])
        self.log.debug("Sent FSM prop")
        self.log.info("Set up complete")

    def update_from_camera(self):
        if self.camera is None:
            return
        self.temp_target_deg_c = self.camera.target_temperature
        self.temp_current_deg_c = self.camera.temperature
        self.exposure_time_sec = self.camera.exposure_time
        self.camera_gain = self.camera.gain
        self.log.debug(f"Read from camera: target = {self.temp_target_deg_c} deg C, current = {self.temp_current_deg_c} deg C, exptime = {self.exposure_time_sec} s, gain = {self.camera_gain}")

    def refresh_properties(self):
        now = time.time()
        current = self.properties['current_exposure']
        if self.currently_exposing:
            remaining_sec = max((self.exposure_start_ts + self.exposure_time_sec) - now, 0)
            remaining_pct = 100 * remaining_sec / self.exposure_time_sec
            self.properties['fsm']['state'] = 'OPERATING'
            self.update_property(self.properties['fsm'])
        else:
            remaining_sec = 0
            remaining_pct = 0
            self.properties['fsm']['state'] = 'READY'
            self.update_property(self.properties['fsm'])
        if remaining_sec != current['remaining_sec']:
            current['remaining_sec'] = remaining_sec
            current['remaining_pct'] = remaining_pct
            self.update_property(current)
        
        self.update_from_camera()

        self.properties['temp_ccd']['current'] = self.temp_current_deg_c
        self.properties['temp_ccd']['target'] = self.temp_target_deg_c
        self.update_property(self.properties['temp_ccd'])

        self.properties['exptime']['current'] = self.exposure_time_sec
        self.properties['exptime']['target'] = self.exposure_time_sec
        self.update_property(self.properties['exptime'])

        self.properties['gain']['current'] = self.camera_gain
        self.properties['gain']['target'] = self.camera_gain
        self.update_property(self.properties['gain'])


    def maintain_temperature_control(self):
        '''User code must close the loop on temperature control'''
        if self.camera is None:
            return
        self.camera.target_temperature = self.temp_target_deg_c

    def _gather_metadata(self):
        meta = {
            'CCDTEMP': self.camera.temperature,
            'CCDSETP': self.camera.target_temperature,
            'GAIN': self.camera.gain,
            'EXPTIME': self.camera.exposure_time,
        }
        for indi_prop in EXTERNAL_RECORDED_PROPERTIES:
            if EXTERNAL_RECORDED_PROPERTIES[indi_prop] is None:
                new_kw = indi_prop.upper().replace('.', ' ')
            else:
                new_kw = EXTERNAL_RECORDED_PROPERTIES[indi_prop]
            #value = self.client.get(indi_prop)
            value = self.client[indi_prop]
            if hasattr(value, 'value'):
                value = value.value
            meta[new_kw] = value
        for fwname in RECORDED_WHEELS:
            meta[f"{fwname.upper()} PRESET NAME"] = find_active_filter(self.client, fwname)
        return meta

    def begin_exposure(self):
        self.currently_exposing = True
        self.should_begin_exposure = False
        self.exposure_start_ts = time.time()
        self.exposure_start_telem = self._gather_metadata()
        self.camera.start_exposure()
        self.log.debug("Asking camera to begin exposure")

    def finalize_exposure(self, actual_exptime_sec=None):
        img = self.camera.readout()
        # Create FITS structure
        hdul = fits.HDUList([
            fits.PrimaryHDU(img)
        ])
        # Populate headers
        meta = self._gather_metadata()
        self.log.debug(f"{meta=}")
        meta['DATE-OBS'] = datetime.datetime.fromtimestamp(self.exposure_start_ts).isoformat()
        exposure_time = self.camera.exposure_time if actual_exptime_sec is None else actual_exptime_sec
        meta['DATE-END'] = datetime.datetime.fromtimestamp(self.exposure_start_ts + exposure_time).isoformat()
        meta['DATE'] = datetime.datetime.utcnow().isoformat()
        for key in self.exposure_start_telem:
            meta[f"BEGIN {key}"] = self.exposure_start_telem[key]
        meta['INSTRUME'] = 'MagAO-X'
        meta['CAMERA'] = 'VIS-X'
        meta['TELESCOP'] = "Magellan Clay, Las Campanas Obs."
        with warnings.catch_warnings(): 
            warnings.simplefilter('ignore')
            hdul[0].header.update(meta)
        # Note if exposure was canceled
        if actual_exptime_sec is not None:
            hdul[0].header['CANCELD'] = True
        # Write to /data path
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%S")
        self.last_image_filename = f"camvisx_{timestamp}.fits"
        outpath = f"{self.data_directory}/{self.last_image_filename}"
        self.log.info(f"Saving to {outpath}")
        try:
            hdul.writeto(outpath)
        except Exception:
            self.log.exception(f"Unable to save frame!")
        self.currently_exposing = False

    def cancel_exposure(self):
        self.currently_exposing = False
        self.should_cancel = False
        self.log.debug("Asking camera to cancel exposure")
        # TODO actually cancel
        actual_exptime_sec = time.time() - self.exposure_start_ts
        self.finalize_exposure(actual_exptime_sec=actual_exptime_sec)

    def loop(self):
        if self.client.interested_properties_missing:
            self.subscribe_to_other_devices()
            self.log.debug(f"Repeating subscription because some external devices we use for headers are not showing up")

        if self.sdk is None:
            self.log.debug("Initializing camera SDK...")
            success = self._init_camera()
            if not success:
                self.log.debug("No camera found yet, retrying on next loop")
                return
            self.log.debug(f"Have camera: {self.camera}")
            self.properties['fsm']['state'] = 'CONNECTED'
            self.update_property(self.properties['fsm'])

        now = time.time()
        if self.should_cancel:
            self.cancel_exposure()
        elif not self.currently_exposing and self.should_begin_exposure:
            self.begin_exposure()
        elif self.currently_exposing and now > (self.exposure_time_sec + self.exposure_start_ts):
            self.currently_exposing = False
            self.log.debug("Exposure finished")
            self.finalize_exposure()
        self.maintain_temperature_control()
        self.refresh_properties()
