# Experimental State app / still has items to be added potentially but being used for testing camera.
## Last version date- 2025-03-03, SFR
## Separate Tool from Dcam API needed to control fan to be turned off currently (DCAM-API_Lite_for_Linux_v24.12.6898)
import numpy as np
import threading 
import subprocess
import argparse
from typing import Optional
import sys
import datetime
import logging
import xconf
import os
import time
from .dcam import *
from .dcamcon import *
from .dcamapi4 import *

import ImageStreamIOWrap as ISIO
from purepyindi2 import device, properties, constants
from purepyindi2.messages import DefNumber, DefSwitch, DefLight, DefText

## NOTE- magaox/indi/devicepy -> from purepyindi2 import Device -> cannot import name Device from purepyindi2. (Looks like thats why the nsv was setup a bit differently as well. - SFR
from magaox.indi.device import XDevice, BaseConfig
# Might remove / testing
from magpyx.utils import create_shmim, ImageStream

log = logging.getLogger(__name__)


EXTERNAL_RECORDED_PROPERTIES = {
    'tcsi.catalog.object': 'OBJECT',
    'tcsi.catdata.ra': None,
    'tcsi.catdata.dec': None,
    'tcsi.catdata.epoch': None,
    'observers.current_observer.full_name': 'OBSERVER',
    'tcsi.teldata.pa': 'PARANG',
    'flipacq.presetName.in': None,
}

CAMERA_CONNECT_RETRY_SEC = 5


@xconf.config
class HamCamConfig(BaseConfig):
    """Python INDI Device for controlling the Hamamatsu 15550-22UP camera.

    Pulled from the 'Example Python INDI device for MagAO-X
    Command-line help statements that will displayed when `pythonIndiExample -h`
    is ran in the terminal (along with a summary of available options)
    
    """

    # configurable_doodad_1 : str = xconf.field(default="abc", help="Configurable doodad 1")
    exptime : float = xconf.field(default=0.01, help='Exposure time in seconds')
    #gain : int = xconf.field(default=1, help='Camera gain')
    binning : float = xconf.field(default=1.0, help='Binning:[1]- 1x1, [2]- 2x2, [3]- 4x4')
    hpos : float = xconf.field(default=0.0, help='x center pixel of window (software)')
    vpos : float = xconf.field(default=0.0, help='y center pixel of window (software)')
    hsize : float = xconf.field(default=4096.0, help='Window width (software)')
    vsize : float = xconf.field(default=2304.0, help='Window height (software)')

    #camera_stream(dpath='/dev/video2', exptime=args.exptime, gain=args.gain,
                  #window=(args.x0, args.y0, args.width, args.height))"""

class HamCam(XDevice):
    config : HamCamConfig
    
    # Testing 
    data_dir : str = "/opt/MagAOX/rawimages/hamcam"
    cancel : bool = False
    exp_start : float = 0
    shmim : ISIO.Image
    frame : np.ndarray
    shmim_name : str = "hamcam"
    last_image : Optional[str] = None
    start_telem : Optional[dict] = None
    exptime : Optional[float] = None
    #gain : Optional[float] = None
    vpos : Optional[float] = None
    hpos : Optional[float] = None
    vsize : Optional[float] = None
    hsize : Optional[float] = None
    binning : Optional[float] = None
    height : Optional[int] = None
    width : Optional[int] = None
    cam : Optional[Dcam] = None
    temp : Optional[float] = None
    temp_target : Optional[float] = None # Testing
    temp_status : Optional[int] = None
    frame_rate : Optional[float] = None
    camstream : Optional[ImageStream] = None
    fan_status : Optional[float] = None # Testing
    protect_status : Optional[float] = None # Testing
    #lasterr = None
    th = threading.Thread()


    def emit_telem_hamcam(self):
        self.log.info(f"In emit_telem_hamcam")
        print("In emit_telem_hamcam")
        w = self.width #self.width
        h = self.height #self.height
        x = self.hpos
        y = self.vpos
        self.telem("telem_hamcam", {
            "roi": {
                "xcen": x,  # Need to check this
                "ycen": y, # Need to check this
                "w": w,
                "h": h,
                "xbin": 1, # Need to change this
                "ybin": 1, # Need to change this
            },
            "exptime": self.exptime, #(Standard Scan- 7.2us to 1800s, ultra quiet- 33.9us to 1800s)
            "frame_rate": self.frame_rate, # Need to do this (Standard Scan- 120frames/s (CoaXPress), 17.6 frames/s(USB (Same for ultra quiet scan) -> assuming full resolution)
            #"emGain": self.gain, # this isn't working right
            "adcSpeed": -1,
            "shutter": {"statusStr": None, "state": None},
            "synchro": 0,
            "vshift": -1,
            "cropMode": 0
        })
    ### Testing this breakdown instead
    def _init_camera(self):
        print("Initalizing Hamamatsu!")
        # initialize some defaults
        iDevice=0
        self.cam = Dcam(iDevice) # Temp hardcode
        #self.dcamcon = Dcamcon()
        if Dcamapi.init():
            #self.cam = Dcam(iDevice)
            # Don't need this but having it here for logs / troubleshooting
            if self.cam.dev_open():
                self.exptime = self.cam.prop_getvalue(DCAM_IDPROP.EXPOSURETIME)
                self.log.info(f"Exptime: {self.exptime}")
                #self.gain = self.cam.prop_getvalue(DCAM_IDPROP.CONTRASTGAIN) # Why is this false?
                #self.log.info(f"GAIN: {self.gain}")
                self.hsize = self.cam.prop_getvalue(DCAM_IDPROP.SUBARRAYHSIZE) 
                self.log.info(f"HSIZE: {self.hsize}")
                self.vsize = self.cam.prop_getvalue(DCAM_IDPROP.SUBARRAYVSIZE) 
                self.log.info(f"VSIZE: {self.vsize}")
                self.hpos = self.cam.prop_getvalue(DCAM_IDPROP.SUBARRAYHPOS)
                self.log.info(f"HPOS: {self.hpos}")
                self.vpos = self.cam.prop_getvalue(DCAM_IDPROP.SUBARRAYVPOS)
                self.log.info(f"VPOS: {self.vpos}")
                self.binning = self.cam.prop_getvalue(DCAM_IDPROP.BINNING)
                self.log.info(f"BINNING: {self.binning}")
                self.temp = self.cam.prop_getvalue(DCAM_IDPROP.SENSORTEMPERATURE)
                self.log.info(f"TEMP: {self.temp}" + "C")
                self.temp_target = self.cam.prop_getvalue(DCAM_IDPROP.SENSORTEMPERATURETARGET)
                self.log.info(f"TEMP TARGET: {self.temp_target}" + "C")
                #self.temp_status = self.cam.prop_getvalue(DCAM_IDPROP.SENSORTEMPERATURE_STATUS) # This one works
                self.temp_status = self.cam.prop_getvalue(DCAM_IDPROP.SENSORCOOLERSTATUS) # Testing <- this is also working
                self.log.info(f"TEMP STATUS: {self.temp_status}")
                self.fan_status = self.cam.prop_getvalue(DCAM_IDPROP.SENSORCOOLER) # Testing this / not working
                self.log.info(f"SENSORCOOLER: {self.fan_status}")
                #self.lasterr = self.cam.lasterr()
                #self.log.info(f"LAST DCAM ERROR: {self.lasterr}")
                self.frame_rate = self.cam.prop_getvalue(DCAM_IDPROP.INTERNALFRAMERATE)
                self.log.info(f"INTERNAL FRAMERATE: {self.frame_rate}")
                self.width = int(self.cam.prop_getvalue(DCAM_IDPROP.IMAGE_WIDTH))
                self.log.info(f"Width: {self.width}")
                self.height = int(self.cam.prop_getvalue(DCAM_IDPROP.IMAGE_HEIGHT))
                self.log.info(f"HEIGHT: {self.height}")

                model = self.cam.dev_getstring(DCAM_IDSTR.MODEL)
                output = 'MODEL={}'.format(model)
                cameraid = self.cam.dev_getstring(DCAM_IDSTR.CAMERAID)
                output = output + ', CAMERAID={}'.format(cameraid)
                self.log.info(output)

                # quick test of fan on/off here
                self.cam.prop_setvalue(DCAM_IDPROP.SENSORCOOLER, DCAMPROP.SENSORCOOLER.OFF)

                fanmode = self.cam.prop_getvalue(DCAM_IDPROP.SENSORCOOLER)
                self.log.info(f'Fan mode is {fanmode}???')
                #self.log.info(f'Failed with error: {self.cam.lasterr().name}')     

                # testing this being here instead
                #_init_properties()
                #self.cam.cap_start()
                self.start_stream()
                return True
            else:
                self.log.info('-NG: Dcam.dev_open() fails with error {}'.format(dcam.lasterr()))
        else:
            self.log.info('-NG: Dcamapi.init() fails with error {}'.format(Dcamapi.lasterr()))
        
        #Dcamapi.uninit()

        return False


    def _init_properties(self):
        """
        Setup / initialize indi properties to be used.
        magaox/indi/device.py -> imports properties from purepyindi2 and is within the XDevice class

        """
        self.log.info(f"Hamamatsu was configured! {self.config=}")
        fsmstate = properties.TextVector(name="fsm")
        fsmstate.add_element(DefText(name="state", _value="NODEVICE"))
        self.add_property(fsmstate)    
        
        # ------ initialize INDI properties ------

        # SwitchStates ---------------------------------------------------------------------
        
        ## Needs adjustments:
        #sv = properties.SwitchVector(
        #    name='expose',
        #    rule=constants.SwitchRule.ONE_OF_MANY,
        #    perm=constants.PropertyPerm.READ_WRITE,
        #)
        #sv.add_element(DefSwitch(name="request", _value=constants.SwitchState.OFF))
        #sv.add_element(DefSwitch(name="cancel", _value=constants.SwitchState.OFF))
        #self.add_property(sv, callback=self.)

        # Exposure Time ---------------------------------------------------------------------
        nv = properties.NumberVector(name='exptime', perm=constants.PropertyPerm.READ_WRITE)
        nv.add_element(DefNumber(
            name='current', label='Exposure time (sec)', format='%3.1f',
            min=0, max=1_000, step=1e-8, _value=self.exptime
        ))
        nv.add_element(DefNumber(
            name='target', label="Requested exposure time(sec)", format="%3.1f",
            min=0, max=1_000, step=1e-8, _value=self.exptime
        ))
        self.add_property(nv, callback=self.set_exptime)

        # Last DCAM Error -------------------------------------------------------------------
        #tv = properties.TextVector(name='error')
        #tv.add_element(DefText(
        #    name='current', label="Last Dcam Error", _value=self.lasterr
        #))
        #self.add_property(tv)

        # Gain ------------------------------------------------------------------------------
        # nv = properties.NumberVector(name='gain', perm=constants.PropertyPerm.READ_WRITE)
        # nv.add_element(DefNumber(
        #     name='current', label='Gain', format='%d',
        #     min=0, max=100, step=1, _value=self.gain
        # ))
        # nv.add_element(DefNumber(
        #     name='target', label='Requested gain', format='%d',
        #     min=0, max=100, step=1, _value=self.gain
        # ))
        # self.add_property(nv, callback=self.set_gain)

        # Binning ----------------------------------------------------------------------------
        nv = properties.NumberVector(name='binning', perm=constants.PropertyPerm.READ_WRITE)
        nv.add_element(DefNumber(
            name='current', label='Binning', format='%3.1f',
            min=1, max=3, step=1, _value=self.binning
        ))
        nv.add_element(DefNumber(
            name='target', label='Requested binning', format='%3.1f',
            min=1, max=3, step=1, _value=self.binning
        ))
        self.add_property(nv, callback=self.set_binning)

        # ROI
        ## HPosition ------------------------------------------------------------------------
        nv = properties.NumberVector(name='hpos', perm=constants.PropertyPerm.READ_WRITE)
        nv.add_element(DefNumber(
            name='current', label='hpos', format='%3.1f',
            min=0, max=4096, step=4, _value=self.hpos
        ))
        nv.add_element(DefNumber(
            name='target', label='Requested hpos', format='%3.1f',
            min=0, max=4096, step=4, _value=self.hpos
        ))
        self.add_property(nv, callback=self.set_hpos)

        ## VPosition -----------------------------------------------------------------------
        nv = properties.NumberVector(name='vpos', perm=constants.PropertyPerm.READ_WRITE)
        nv.add_element(DefNumber(
            name='current', label='vpos', format='%3.1f',
            min=0, max=2300, step=4, _value=self.vpos
        ))
        nv.add_element(DefNumber(
            name='target', label='Requested vpos', format='%3.1f',
            min=0, max=2300, step=4, _value=self.vpos
        ))
        self.add_property(nv, callback=self.set_vpos)

        ## HSize ----------------------------------------------------------------------------
        nv = properties.NumberVector(name='hsize', perm=constants.PropertyPerm.READ_WRITE)
        nv.add_element(DefNumber(
            name='current', label='hsize', format='%3.1f',
            min=4, max=4096, step=4, _value=self.hsize
        ))
        nv.add_element(DefNumber(
            name='target', label='Requested hsize', format='%3.1f',
            min=4, max=4096, step=4, _value=self.hsize
        ))
        self.add_property(nv, callback=self.set_hsize)

        ## VSize ----------------------------------------------------------------------------
        nv = properties.NumberVector(name='vsize', perm=constants.PropertyPerm.READ_WRITE)
        nv.add_element(DefNumber(
            name='current', label='vsize', format='%3.1f',
            min=4, max=2304, step=4, _value=self.vsize
        ))
        nv.add_element(DefNumber(
            name='target', label='Requested vsize', format='%3.1f',
            min=4, max=2304, step=4, _value=self.vsize
        ))
        self.add_property(nv, callback=self.set_vsize)

        ## END ROI---------------------------------------------------------------------------

        # Temperature -----------------------------------------------------------------------
        nv = properties.NumberVector(name='temp', perm=constants.PropertyPerm.READ_WRITE)
        nv.add_element(DefNumber(
            name='current', label='Current Temperature (deg C)', format='%3.1f',
            min=-50, max=100, step=0.1, _value=self.temp
        ))
        nv.add_element(DefNumber(
            name='target', label='Target Temperature (deg C)', format='%3.1f',
            min=-50, max=100, step=0.1, _value=self.temp_target
        ))
        self.add_property(nv, callback=self.set_temp)

        # Temp Status -----------------------------------------------------------------------
        # Testing changing format- previous %d -> Changing to textvector?
        nv = properties.NumberVector(name='temp_status', perm=constants.PropertyPerm.READ_WRITE)
        nv.add_element(DefNumber(
            name='current', label='Temperature Status', format='%3.1f',
            min=-5, max=5, step=1, _value=self.temp_status
        ))
        nv.add_element(DefNumber(
            name='target', label='Requested Temperature Status', format='%3.1f',
            min=-5, max=5, step=1, _value=self.temp_status
        ))
        self.add_property(nv, callback=self.set_tempstatus)

        # Internal Frame rate (s) -----------------------------------------------------------
        nv = properties.NumberVector(name="frame_rate")
        nv.add_element(DefNumber(
            name='current', label='Current Frame Rate per second', format='%3.1f',
            min=17, max=60.1 , step=0.1, _value=self.frame_rate
        ))
        self.add_property(nv)

        # Shmim Info ------------------------------------------------------------------------
        tv = properties.TextVector(name="fg_shmimName")
        tv.add_element(DefText(
            name='name', label='Shmim Name', _value=self.shmim_name,
        ))
        self.add_property(tv)   

        nv = properties.NumberVector(name="fg_framesize")
        nv.add_element(DefNumber(
            name='height', label='Frame size height', format='%d', 
            min=0, max=2304, step=1, _value=self.height,
        ))
        nv.add_element(DefNumber(
            name='width', label='Frame size width', format='%d', 
            min=0, max=4096, step=1, _value=self.width,
        ))
        self.add_property(nv)

        # Testing this here so it will update first before moving to shmim items
        #_init_camera()

        # End of properties -------------------------------------------------
        self.log.info("Set up properties complete")


    def setup(self):
        os.makedirs(self.data_dir, exist_ok=True)
        while self.client.status is not constants.ConnectionStatus.CONNECTED:
            self.log.info(f"Connecting to INDI as a client to get {list(EXTERNAL_RECORDED_PROPERTIES.keys())}")
            time.sleep(1)
        self.log.info(f"INDI client connection: {self.client.status}")
        #self.subscribe_to_other_devices()

        self._init_properties()
        self.properties['fsm']['state'] = 'NOTCONNECTED'
        self.log.info("Set FSM properties")
        self.update_property(self.properties['fsm'])
        self.log.info("Set up complete")   


    def update_from_camera(self):
        # If no camera is detected
        #print("Updating from camera")
        if self.cam is None:
            self.log.info(f"No camera detected. Cannot update from camera.")
            return
        # Otherwise get values from camera / This can be done better / temp
        self.exptime = self.cam.prop_getvalue(DCAM_IDPROP.EXPOSURETIME)
        #self.gain = self.cam.prop_getvalue(DCAM_IDPROP.CONTRASTGAIN)
        self.hsize = self.cam.prop_getvalue(DCAM_IDPROP.SUBARRAYHSIZE)
        self.vsize = self.cam.prop_getvalue(DCAM_IDPROP.SUBARRAYVSIZE)
        self.hpos = self.cam.prop_getvalue(DCAM_IDPROP.SUBARRAYHPOS)
        self.vpos = self.cam.prop_getvalue(DCAM_IDPROP.SUBARRAYVPOS)
        self.temp = self.cam.prop_getvalue(DCAM_IDPROP.SENSORTEMPERATURE)
        self.temp_target = self.cam.prop_getvalue(DCAM_IDPROP.SENSORTEMPERATURETARGET)
        #self.temp_status = self.cam.prop_getvalue(DCAM_IDPROP.SENSORTEMPERATURE_STATUS) #This one works but not editable
        self.temp_status = self.cam.prop_getvalue(DCAM_IDPROP.SENSORCOOLER) #SENSORCOOLERSTATUS
        #self.lasterr = self.cam.lasterr() # testing for getting last error for quicker view
        self.binning = self.cam.prop_getvalue(DCAM_IDPROP.BINNING)
        self.frame_rate = self.cam.prop_getvalue(DCAM_IDPROP.INTERNALFRAMERATE)
        self.width = int(self.cam.prop_getvalue(DCAM_IDPROP.IMAGE_WIDTH))
        self.height = int(self.cam.prop_getvalue(DCAM_IDPROP.IMAGE_HEIGHT))
        # gain = {self.gain}
        self.log.debug(f"Read from camera: exptime = {self.exptime}, hsize = {self.hsize}, vsize = {self.vsize}, hpos = {self.hpos}, vpos = {self.vpos}")

    def refresh_properties(self):
        
        self.update_from_camera()

        self.properties['hsize']['current'] = self.hsize
        self.properties['hsize']['target'] = self.hsize
        self.update_property(self.properties['hsize'])

        self.properties['vsize']['current'] = self.vsize
        self.properties['vsize']['target'] = self.vsize
        self.update_property(self.properties['vsize'])

        self.properties['hpos']['current'] = self.hpos
        self.properties['hpos']['target'] = self.hpos
        self.update_property(self.properties['hpos'])

        self.properties['vpos']['current'] = self.vpos
        self.properties['vpos']['target'] = self.vpos
        self.update_property(self.properties['vpos'])

        self.properties['exptime']['current'] = self.exptime
        self.properties['exptime']['target'] = self.exptime
        self.update_property(self.properties['exptime'])

        #self.properties['error']['current'] = self.lasterr # testing

        #self.properties['gain']['current'] = self.gain
        #self.properties['gain']['target'] = self.gain
        #self.update_property(self.properties['gain'])

        self.properties['binning']['current'] = self.binning
        self.properties['binning']['target'] = self.binning
        self.update_property(self.properties['binning'])

        self.properties['temp']['current'] = self.temp
        self.properties['temp']['target'] = self.temp_target
        self.update_property(self.properties['temp'])

        self.properties['frame_rate']['current'] = self.frame_rate
        self.update_property(self.properties['frame_rate'])

        self.properties['temp_status']['current'] = self.temp_status
        self.properties['temp_status']['target'] = self.temp_status
        self.update_property(self.properties['temp_status'])

        self.properties['fg_framesize']['width'] = self.width
        self.properties['fg_framesize']['height'] = self.height
        self.update_property(self.properties['fg_framesize'])

    # Testing Item
    def teardown(self):
        self.log.info('Shutting down.')
        #self.streamthread.pause()
        self.cam.cap_stop()
        self.th.join() # maybe?
        self.cam.buf_release()
        self.log.info("Released buffer")
        self.cam.dev_close()
        #self.camstream.close()
        Dcamapi.uninit()

    # Testing -> might be working now
    def start_stream(self):
        self.log.info("Starting stream for hamamatsu")   
        #self.log.info(f"Successfully started cap_start.")
        self.camera_stream()
        #else:
        #    self.log.info(f"Issues with cap_start working.")
        #    return False
    
    # Testing / not working
    def camera_stream(self):
        """
        Testing this adjustment
        """
        # DCAM_IDPROP_IMAGE_WIDTH useful here? / need to check
        #shmim_shape = ( int(self.width), int(self.height) )
        shmim_shape = ( int(self.hsize), int(self.vsize) )
        try:
             self.camstream = ImageStream(self.shmim_name, expected_shape=shmim_shape)
        except RuntimeError or ValueError:
             self.log.info(f"Failed to open shmim {self.shmim_name}. Trying to create...")
             create_shmim(self.shmim_name, shmim_shape)
             self.camstream == ImageStream(self.shmim_name)

        
        if self.cam.buf_alloc(10):
            #self.stream_thread()
            self.th = threading.Thread(target=self.stream_thread)
            self.th.start()
            # Release buffer
            #self.cam.buf_release()
            #self.log.info("Released buffer")
        else:
            self.log.info(f"Buf_alloc(10) failed")
            #self.th.join()
            self.cam.buf_release()
            self.log.info("Released buffer in camera_stream")
    
    # Testing / not working
    def stream_thread(self):
        """
        Testing
        """
        self.log.info("In stream thread")
        timeout = 10000 #(millisec)
        status = 0
        cam_status = self.cam.cap_status()
        self.log.info(cam_status) # THis is reporting '2'

        if self.cam.cap_start():
            cam_status = self.cam.cap_status()
            self.log.info(cam_status) # THis is reporting '1'
            #self.log.info(self.cam.cap_start(bSequence=True))
            self.log.info("cam.cap_start working")
            while cam_status == DCAMCAP_STATUS.BUSY:
                #if self.cam.wait_capevent_frameready(timeout):
                #data = self.cam.buf_getlastframedata()
                #data = self.cam.buf_getframe(1)
                #print(data) # this is false / having issues doing this?
                #print(self.cam.lasterr())
                #status = self.show_framedata(data, status)
                if self.cam.wait_capevent_frameready(timeout):
                    #self.log.info("In cam.wait_capevent_frameready()")
                    data = self.cam.buf_getlastframedata()
                    #print(data) # this is false / having issues doing this?
                    #if isinstance(data, bool) and not data:
                    #    print(self.cam.lasterr())
                    if data.dtype == np.uint16:
                        #rawframe = np.frombuffer(data, dtype=np.uint16).reshape(int(self.height), int(self.width))
                        rawframe = np.frombuffer(data, dtype=np.uint16).reshape(int(self.vsize), int(self.hsize))
                        self.camstream.write(rawframe)
                        #self.log.info("in camstream.writer")
                    else:
                        dcamerr = self.cam.lasterr()
                        if dcamerr.is_timeout():
                            self.log.info("===: timeout")
                        else:
                            self.log.info("Dcam.wait_event() failed with error {}".format(dcamerr))
                cam_status = self.cam.cap_status() # check if cap_status has been changed (e.g., when parameter change is requested)
                #self.log.info(cam_status) # noisy, but useful for debugging
            else:
                self.log.info("here")
                self.log.info(self.cam.lasterr())
            #self.log.info("stream_thread ended?")
        else:
            self.log.info("Issues with capturing / cap_start or stream was triggered to end.")
            self.log.info(self.cam.lasterr())

    def pause_stream(self):
        self.log.info("Pausing Stream")
        self.cam.cap_stop()
        self.th.join() # maybe?
        self.cam.buf_release()
            
        
    # Testing / not working -> I dont' think I need this?
    def show_framedata(self, data, status):
        """
        Testing this
        """
        self.log.info("Trying to show framedata")
        if data.dtype == np.uint16:
            imax = np.amax(data)
            if imax > 0:
                imul = int(65535/imax)
                data = data * imul
            return 1
        else:
            self.log.info("Issues with showing framedata")

    def check_fan(self):
        """
        Check fan status
        DCAM_IDPROP_SENSORCOOLER 
            DCAMPROP_SENSORCOOLER_OFF
        """
        if Dcamapi.init():
            #cooler = self.cam.prop_getvaluetext(DCAM_IDPROP.SENSORCOOLER, 1.0)
            self.fan_status = self.cam.prop_getvalue(DCAM_IDPROP.SENSORCOOLER)
            #cooler = dcamcon.get_propertyvalue(SENSORCOOLER)
            self.log.info(f"In check fan: {cooler}")
            if cooler == False:
                self.log.info("ISSUES WITH GRABBING SENSORCOOLER!")
            elif cooler == DCAMPROP.SENSORCOOLER.ON:
                self.log.info("Cooler is on")
                return True
            elif cooler == DCAMPROP.SENSORCOOLER.OFF:
                self.log.info("Cooler is on")
                return False
            else:
                self.log.info("other case")
        else:
            print("issues with dcamapi call in check fan")
        Dcamapi.uninit()


    def set_hpos(self, existing_property, new_message):
        """
        DCAM_IDPROP_SUBARRAYHPOS
            0 to 4096, step 4, default 0
                For DCAMPROP_SENSORMODE_AREA or PHOTONNUMBERRESOLVING
            0 to 4096, step 1, default 0
                For DCAMPROP_SENSORMODE_PROGRESS
        """
        ## Need to turn off subarray before modifying
        self.pause_stream()
        subarray = self.check_subarray()
        if subarray == True:
            self.switch_subarray()

        if 'target' in new_message and new_message['target'] != existing_property['current']:
            self.log.info(f"Setting hpos (x position)")
            if self.cam is None:
                self.log.debug('-NG: Dcamcon is not opened')
                return False
            else:# prop_setvalue(self, idprop: DCAM_IDPROP, fValue)
                print("In hpos set self.cam else statement.")
                hpos_requested = float(new_message['target'])
                self.log.info(f'Setting horitizonal postiion to {hpos_requested}')
                self.cam.prop_setvalue(DCAM_IDPROP.SUBARRAYHPOS, hpos_requested)
                hpos_actual = self.cam.prop_getvalue(DCAM_IDPROP.SUBARRAYHPOS)
                self.log.info(f'Went to an actual hpos of {hpos_actual}')
                if hpos_requested != hpos_actual:
                    self.log.info(f"Hpos request does not = hpos actual.")
                else:
                    print("In hpos set second to last else statement")
                    existing_property['current'] = new_message['target']
                    existing_property['target'] = new_message['target']
                    self.hpos = hpos_actual
                    self.update_property(existing_property)
        else:
            self.log.info(f"Target requested is equal to current set value for hpos")

        self.switch_subarray()  
        self.update_property(existing_property)
        self.start_stream()
        return True
    
    def set_hsize(self, existing_property, new_message):
        """
        DCAM_IDPROP_SUBARRAYHSIZE
            4 to 4096, step 4, default 4096
                For DCAMPROP_SENSORMODE_AREA or PHOTONNUMBERRESOLVING
            1 to 4096, step 1, default 4096
                For DCAMPROP_SENSORMODE_PROGRESS
        """

        ## Need to turn off subarray before modifying
        self.pause_stream()
        subarray = self.check_subarray()
        if subarray == True:
            self.switch_subarray()

        self.log.debug(f"Setting hsize (width)")
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            if self.cam is None:
                self.log.debug('-NG: Dcamcon is not opened')
                return False
            else:
                print("In hsize set self.cam else statement.")
                hsize_requested = float(new_message['target'])
                self.log.info(f'Setting horitizonal postiion to {hsize_requested}')
                self.cam.prop_setvalue(DCAM_IDPROP.SUBARRAYHSIZE, hsize_requested)
                hsize_actual = self.cam.prop_getvalue(DCAM_IDPROP.SUBARRAYHSIZE)
                self.log.info(f'Went to an actual hsize of {hsize_actual}')
                if hsize_requested != hsize_actual:
                    self.log.info(f"Hsize request does not = hsize actual.")
                else:
                    print("In hsize set second to last else statement")
                    existing_property['current'] = new_message['target']
                    existing_property['target'] = new_message['target']
                    self.hsize = hsize_actual
                    self.update_property(existing_property)
        else:
            self.log.info(f"Target requested is equal to current set value for hpos")

        self.switch_subarray()  
        self.update_property(existing_property)
        self.start_stream()
        return True


    def set_vpos(self, existing_property, new_message):
        """
        DCAM_IDPROP_SUBARRAYVPOS
            0 to 2300, step 4, default 0
        """
        ## Need to turn off subarray before modifying
        self.pause_stream()
        subarray = self.check_subarray()
        if subarray == True:
            self.switch_subarray()

        self.log.info(f"Setting vpos (y position)")
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            if self.cam is None:
                self.log.debug('-NG: Dcamcon is not opened')
                return False
            else:# prop_setvalue(self, idprop: DCAM_IDPROP, fValue)
                print("In vpos set self.cam else statement.")
                vpos_requested = float(new_message['target'])
                self.log.info(f'Setting horitizonal postiion to {vpos_requested}')
                self.cam.prop_setvalue(DCAM_IDPROP.SUBARRAYVPOS, vpos_requested)
                vpos_actual = self.cam.prop_getvalue(DCAM_IDPROP.SUBARRAYVPOS)
                self.log.info(f'Went to an actual vpos of {vpos_actual}')
                if vpos_requested != vpos_actual:
                    self.log.info(f"Vpos request does not = vpos actual.")
                else:
                    print("In vpos set second to last else statement")
                    existing_property['current'] = new_message['target']
                    existing_property['target'] = new_message['target']
                    self.vpos = vpos_actual
                    self.update_property(existing_property)
        else:
            self.log.info(f"Target requested is equal to current set value for hpos")

        self.switch_subarray()  
        self.update_property(existing_property)
        self.start_stream()
        return True
    
    
    def set_vsize(self, existing_property, new_message):
        """
        DCAM_IDPROP_SUBARRAYVSIZE
            4 to 2304, step 4, default 2304
        """
        ## Need to turn off subarray before modifying
        self.pause_stream()
        subarray = self.check_subarray()
        if subarray == True:
            self.switch_subarray()

        self.log.debug(f"Setting vsize (height)")
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            if self.cam is None:
                self.log.debug('-NG: Dcamcon is not opened')
                return False
            else:
                # prop_setvalue(self, idprop: DCAM_IDPROP, fValue)
                vsize_requested = float(new_message['target'])
                self.log.info(f'Setting horitizonal postiion to {vsize_requested}')
                self.cam.prop_setvalue(DCAM_IDPROP.SUBARRAYVSIZE, vsize_requested)
                vsize_actual = self.cam.prop_getvalue(DCAM_IDPROP.SUBARRAYVSIZE)
                self.log.info(f'Went to an actual hpos of {vsize_actual}')
                if vsize_requested != vsize_actual:
                    log.info(f"Height request does not = actual height.")
                else:
                    existing_property['current'] = new_message['target']
                    existing_property['target'] = new_message['target']
                    self.vsize = vsize_actual
                    self.update_property(existing_property)
        else:
            self.log.info(f"Target requested is equal to current set value for vsize")

        self.switch_subarray()  
        self.update_property(existing_property)
        self.start_stream()
        return True


    def set_binning(self, existing_property, new_message):
        """
        Available Binning Options:
            [1]- 1x1 (DCAMPROP_BINNING_1)
            [2]- 2x2 (DCAMPROP_BINNING_2)
            [4]- 4x4 (DCAMPROP_BINNING_4)
            [8]- 8x8 (DCAMPROP_BINNING_8)
            [9]- 16x16 (DCAMPROP_BINNING_16)
        """
        self.pause_stream()
        self.log.debug(f"Setting binning")
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            if self.cam is None:
                self.log.debug('-NG: Dcamcon is not opened')
                return False
            # prop_setvalue(self, idprop: DCAM_IDPROP, fValue)
            bin_requested = float(new_message['target'])
            self.log.info(f'Setting binning to {bin_requested}')                
            self.cam.prop_setvalue(DCAM_IDPROP.BINNING, bin_requested)
            bin_actual = self.cam.prop_getvalue(DCAM_IDPROP.BINNING)
            self.log.info(f'Went to a binning of {bin_actual}')
            if bin_requested != bin_actual:
                self.log.info(f"Binning request does not = actual binning.")
            else:
                existing_property['current'] = new_message['target']
                existing_property['target'] = new_message['target']
                self.binning = bin_actual
                self.update_property(existing_property)
        else:
            self.log.info(f"Target requested is equal to current set value for binning")
        self.start_stream()


    # Might remove / probably don't need to have this like this.
    def check_subarray(self):
        """
        ROI Mode Setting:
            DCAM_IDPROP_SUBARRAYMODE_ON / OFF
            Default- OFF
            Returns- bool value if subarraymode is on or not. True = ON, False = OFF
        """

        subarraymode = self.cam.prop_getvalue(DCAM_IDPROP.SUBARRAYMODE)
        self.log.info(f"In subarrarymode check: {subarraymode}")

        if subarraymode == DCAMPROP.MODE.ON:
            self.log.info(f"DCAM_IDPROP.SUBARRAYMODE is turned on")
            return True
        elif subarraymode == DCAMPROP.MODE.OFF:
            self.log.info(f"DCAM_IDPROP.SUBARRAYMODE is turned off")
            return False
        else:
            # Error happened
            self.log.info(f"Issues with detecting subarraymode")
            return False
    
    def switch_subarray(self):
        """
        ROI Mode Setting:
            DCAM_IDPROP_SUBARRAYMODE_ON / OFF
            Default- OFF
            Function- inverse / switch subarray setting based on initial value
            Return- bool value if successful
        """
        initialmode = self.check_subarray()

        if initialmode == True:
            self.cam.prop_setvalue(DCAM_IDPROP.SUBARRAYMODE, 1.0)
            newmode = self.check_subarray()
            if newmode == False:
                self.log.info(f"Successfully changed to ROI mode / Subarraymode")
                return True
            else:
                self.log.info(f"Issues with changing subarraymode off")
                return False
        elif initialmode == False:
            self.cam.prop_setvalue(DCAM_IDPROP.SUBARRAYMODE, 2.0)
            newmode = self.check_subarray()
            if newmode == True:
                self.log.info(f"Successfully changed to ROI mode / Subarraymode")
                return True
            else:
                self.log.info(f"Issues with changing subarraymode on")
                return False
        else:
            # Error happened
            self.log.info(f"Issues with detecting subarraymode")
            return False

    # Having this try to set a target temp over actually turnning off the fan since its not working....
    def set_temp(self, existing_property, new_message):
        """
        Target Temperature:
            DCAM_IDPROP_SENSORTEMPERATURETARGET
        """
        self.log.info("TRYING TO SET TARGET TEMP")

        if 'target' in new_message and new_message['target'] != existing_property['current']:
            if self.cam is None:
                self.log.debug('-NG: Dcamcon is not opened')
                return False
            # prop_setvalue(self, idprop: DCAM_IDPROP, fValue)
            temp_requested = float(new_message['target'])
            self.log.info(f'Setting TEMP to {temp_requested}')
            self.cam.prop_setvalue(DCAM_IDPROP.SENSORTEMPERATURETARGET, temp_requested)
            temp_actual = self.cam.prop_getvalue(DCAM_IDPROP.SENSORTEMPERATURETARGET)
            self.log.info(f"TEMP actually set too: {self.temp_status}")
            if temp_requested != temp_actual:
                self.log.info(f"Temp status not = to actual.")
                existing_property['target'] = new_message['target']
                self.update_property(existing_property)
            else:
                existing_property['current'] = new_message['target']
                existing_property['target'] = new_message['target']
                self.temp_status = temp_actual
                self.update_property(existing_property)

            print(temp_actual, temp_requested)

    
    
    # Not implemented yet / but should be a 'quick' add / This isn't working....false values even on the other IDS that should work for dcam and are R/W
    def set_tempstatus(self, existing_property, new_message):
        """
        Temp Mode Setting:
            DCAM_IDPROP_SENSORCOOLER (at Water Cooling only)
            __OFF, __ON, __MAX
        """
        self.log.info("Attempting to set temperature. Not Implemented yet fully.")
        
        #self.pause_stream()

        if 'target' in new_message and new_message['target'] != existing_property['current']:
            if self.cam is None:
                self.log.debug('-NG: Dcamcon is not opened')
                return False
            # prop_setvalue(self, idprop: DCAM_IDPROP, fValue)
            mode_requested = float(new_message['target'])
            self.log.info(f'Setting SENSORCOOLER to {mode_requested}')
            self.cam.prop_setvalue(DCAM_IDPROP.SENSORCOOLER, mode_requested)
            mode_actual = self.cam.prop_getvalue(DCAM_IDPROP.SENSORCOOLER)
            self.log.info(f"SENSORCOOLER actually set too: {self.temp_status}")
            if mode_requested != mode_actual:
                self.log.info(f"Sensor status not = to actual.")
            else:
                existing_property['current'] = new_message['target']
                existing_property['target'] = new_message['target']
                self.temp_status = mode_actual
                self.update_property(existing_property)

            print(mode_actual, mode_requested)
        
        #self.start_stream()


    # Not implemented yet / but should be a 'quick' add
    def get_temp(self, existing_property, new_message):
        """
        Temp Mode Setting:
            DCAM_IDPROP_SENSORCOOLER (at Water Cooling only)
            __OFF, __ON, __MAX
        """
        print("Not Implemented yet.")

        return False

    # This isn't working right / gain in general -> outputs false / something wrong
    def set_gain(self, existing_property, new_message):
        """
        Setting Gain Value:
            DCAM_IDPROP.CONTRASTGAIN
        """
        self.pause_stream()
        self.log.debug("Gain requested!")
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            if self.cam is None:
                self.log.debug('-NG: Dcamcon is not opened')
                return False
            # prop_setvalue(self, idprop: DCAM_IDPROP, fValue)
            gain_requested = float(new_message['target'])
            self.log.info(f'Setting exposure time to {gain_requested}')
            self.cam.prop_setvalue(DCAM_IDPROP.CONTRASTGAIN, gain_requested)
            gain_actual = self.cam.prop_getvalue(DCAM_IDPROP.CONTRASTGAIN)
            self.log.info(f'Went to an actual exposure time of {gain_actual}')
            if gain_requested != gain_actual:
                self.log.info(f"Exposure time request does not = gain actual.")
            else:
                existing_property['current'] = new_message['target']
                existing_property['target'] = new_message['target']
                self.gain = gain_actual
                self.update_property(existing_property)

            print(gain_actual, gain_requested)
        self.start_stream()


    def set_exptime(self, existing_property, new_message):
        """
        Set the Exposure Time in seconds
            DCAM_IDPROP.EXPOSURETIME

            0.000033949 to 1800.000015185, step 0.00000001, default 0.0082944
                DCAMPROP_SENSORMODE_AREA and DCAM_IDPROP_READOUTSPEED=1 or
                DCAMPROP_SENSORMODE_PHOTONNUMBERRESOLVING
                Depends on SUBARRAY properties
            0.0000072 to 1800.0, step 0.00000001, default 0.0082944
                DCAMPROP_SENSORMODE_AREA and DCAM_IDPROP_READOUTSPEED=2
                Depends on SUBARRAY properties
            0.0000072 to 0.0082944, step 0.00000001, default 0.0082944
                DCAMPROP_SENSORMODE_PROGRESSIVE 
                Depends on INTERNALLINESPEED 
                and INTERNAL_LINEINTERVAL, SUBARRY properties
        """
        self.pause_stream()
        self.log.debug(f"Setting exposure time")
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            if self.cam is None:
                self.log.debug('-NG: Dcamcon is not opened')
                return False
            # prop_setvalue(self, idprop: DCAM_IDPROP, fValue)
            exptime_requested = float(new_message['target'])
            self.log.info(f'Setting exposure time to {exptime_requested}')
            self.cam.prop_setvalue(DCAM_IDPROP.EXPOSURETIME, exptime_requested)
            exptime_actual = self.cam.prop_getvalue(DCAM_IDPROP.EXPOSURETIME)
            self.log.info(f'Went to an actual exposure time of {exptime_actual}')
            if exptime_requested != exptime_actual:
                self.log.info(f"Exposure time request does not = exptime actual.")
            else:
                existing_property['current'] = new_message['target']
                existing_property['target'] = new_message['target']
                self.exptime = exptime_actual
                self.update_property(existing_property)
        self.start_stream()

    def loop(self):
        if self.cam is None:
            self.log.debug("Initializing camera...")
            success = self._init_camera()
            if not success:
                self.log.debug("No camera found yet, retrying on next loop")
                #self.properties['fsm']['state'] = 'NODEVICE' # Probably don't need this
                #self.update_property(self.properties['fsm']) # Probably don't need this
                return
            self.log.debug(f"Have camera: {self.cam}")
            self.properties['fsm']['state'] = 'CONNECTED'
            self.update_property(self.properties['fsm'])
        self.refresh_properties()

    # Not 'really' being used / placeholder
    def _gather_metadata(self):
        meta = {
            #'GAIN': self.cam.prop_getvalue(DCAM_IDPROP.CONTRASTGAIN),
            'EXPTIME': self.cam.prop_getvalue(DCAM_IDPROP.EXPOSURETIME),
            'HPOS': self.cam.prop_getvalue(DCAM_IDPROP.SUBARRAYHPOS),
            'VPOS': self.cam.prop_getvalue(DCAM_IDPROP.SUBARRAYVPOS),
            'HSIZE': self.cam.prop_getvalue(DCAM_IDPROP.SUBARRAYHSIZE),
            'VSIZE': self.cam.prop_getvalue(DCAM_IDPROP.SUBARRAYVSIZE),
            'BINNING': self.cam.prop_getvalue(DCAM_IDPROP.BINNING),
        }
        return meta

# Not fully up yet
class CameraStreamThread(threading.Thread):
    '''
    Camera stream thread to enable pausing and resume while setting parameters

    Unclear if it's actually needed for exposure time and gain.

    '''    
    # stolen from https://stackoverflow.com/a/15734837

    def __init__(self, cam, shmim, height, width):
        super(CameraStreamThread, self).__init__()
        self.iterations = 0
        self.daemon = True  # Allow main to exit even if still running.
        self.paused = True  # Start out paused.
        self.state = threading.Condition()

        self.cam = cam
        self.shmim = shmim
        self.height = height
        self.width = width

    def run(self):
        self.resume()
