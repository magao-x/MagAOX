"""Module to control DCAM on the console.

This is the module that implements functions that prompt 
for user input to control DCAM on the console.
For example:
Choose the device when several devices are detected.
Set the value of specified prperty
"""

__copyright__ = 'Copyright (C) 2024 Hamamatsu Photonics K.K.'

from .dcam import *
# for control DCAM functions

# True means dcamapi_init() was called and succeed.
called_dcamapi_init = False

# array of Dcamcon instance. This is initialized at dcamcon_init()
dcamcon_list = []

class PromptRestrictMode(IntEnum):

    """Restrict flag for prompt_propvalue().

    A flag that specifies the type of data to limit the choices in prompt_propvalue().
    """

    No = 0
    ModeFilter = 1  # filter by specified list
    ClipMinimum = 2  # clip minimum value by specified value



class Dcamcon:

    """Control DCAM-API on the console.

    Functions that are interactive and easy to control DCAM-API.
    This class is used for the sample of camera function
    """

    def __init__(self):
        self.deviceindex = -1
        # Dcam instance
        self.dcam : Dcam = None
        # MODEL + CAMERAID + BUS. for OpenCV window title and camera list
        self.device_title = None
        # number of DCAM frame buffers
        self.__number_of_frames = 10

    def close(self):
        """Close Dcam.

        Call Dcam.close() and set None to self.dcam

        Returns:
            bool: result
        """
        if self.dcam is None:
            return True
        
        if not self.dcam.dev_close():
            print('-NG: Dcam.dev_close() failed with error {}'.format(self.dcam.lasterr().name))
            return False
        
        self.dcam = None
        return True
        
    def allocbuffer(self, number_of_frames):
        """Allocate buffer.

        Allocate buffer with Dcam.buf_alloc().
        If success, set value is kept by self.__number_of_frames.

        Args:
            number_of_frames (int): Value to set for Dcam.buf_alloc()
        
        Returns:
            bool: result
        """
        if self.dcam is None:
            print('-NG: Dcamcon is not opened')
            return False

        if not self.dcam.buf_alloc(number_of_frames):
            print('-NG: Dcam.buf_alloc({}) failed with error {}'.format(number_of_frames, self.dcam.lasterr().name))
            return False
        
        self.__number_of_frames = number_of_frames
        return True
    
    def releasebuffer(self):
        """Release allocated buffer.

        Release allcated buffer with Dcam.buf_release().

        Returns:
            bool: result
        """
        if self.dcam is None:
            print('-NG: Dcamcon is not opened')
            return False
        
        if not self.dcam.buf_release():
            print('-NG: Dcam.buf_release() failed with error {}'.format(self.dcam.lasterr().name))
            return False
        
        return True
    
    def startcapture(self, is_sequence = True):
        """Start capturing.

        Start capturing with Dcam.cap_start().
        If failure, if shows error message.

        Args:
            is_sequence (bool): if True, sequential capturing, otherwise snap capturing
        
        Returns:
            bool: result
        """
        if self.dcam is None:
            print('-NG: Dcamcon is not opened')
            return False
        
        if not self.dcam.cap_start(is_sequence):
            print('-NG: Dcam.cap_start() failed with error {}'.format(self.dcam.lasterr().name))
            return False
        
        return True
    
    def stopcapture(self):
        """Stop capturing.

        Stop capturing with Dcam.cap_stop().
        If failure, it shows error message

        Returns:
            bool: result
        """
        if self.dcam is None:
            print('-NG: Dcamcon is not opened')
            return False
        
        if not self.dcam.cap_stop():
            print('-NG: Dcam.cap_stop() failed with error {}'. format(self.dcam.lasterr().name))
            return False
        
        return True
    
    def is_capstaus_ready(self):
        """Check whether DCAMCAP_STATUS is READY or not.

        Call Dcam.cap_status() and check whetherthe value is READY or not

        Returns:
            bool: result 
        """
        if self.dcam is None:
            print('-NG: Dcamcon is not opened')
            return False
        
        capstatus = self.dcam.cap_status()
        if capstatus is False:
            print('-NG: Dcam.cap_status() failed with error {}'.format(self.dcam.lasterr().name))
            return False
        
        return capstatus == DCAMCAP_STATUS.READY
    
    def firetrigger(self):
        """Fire trigger.

        Fire software trigger when TRIGGERSOURCE is SOFTWARE.

        Returns:
            bool: result
        """
        if self.dcam is None:
            print('-NG: Dcamcon is not opened')
            return False
        
        if not self.dcam.cap_firetrigger():
            print('-NG: Dcam.cap_firetrigger() failed with error {}'.format(self.dcam.lasterr().name))
            return False
        
        return True
    
    def wait_capevent_frameready(self, timeout_millisec):
        """Wait capture event frameready.

        Wait for frameready event for amount of time specified by timeout_millisec.
        If frameready event happened, it returns True. Otherwise it returns DCAMERR.

        Args:
            timeout_millisec (int): timeout time for waiting frameready event in ms
        
        Returns:
            bool: True if success
            DCAMERR: Dcam.lasterr() if failure 
        """
        if self.dcam is None:
            print('-NG: Dcamcon is not opened')
            return False
        
        if not self.dcam.wait_capevent_frameready(timeout_millisec):
            return self.dcam.lasterr()
        
        return True
    
    def get_lastframedata(self):
        """Get last frame data.

        Access last frame data with Dcam.buf_getlastframedata().
        If success, it returns Numpy ndarray stored last image data.
        If failure, it returns False

        Returns:
            NumPy ndarray: NumPy ndarray stored image if success
            bool: False if failure
        """
        if self.dcam is None:
            print('-NG: Dcamcon is not opened')
            return False
        
        ret = self.dcam.buf_getlastframedata()
        if ret is False:
            print('-NG: Dcam.buf_getlastframedata() failed with error {}'.format(self.dcam.lasterr().name))
            return False
        
        return ret
    
    def get_transferinfo(self):
        """Get transfer status.

        Get the total number of images captured and the frame index of the last captured.

        Returns:
            (int, int): index of the last captured frame, number of captured frames
            bool: False if failure
        """
        if self.dcam is None:
            print('-NG: Dcamcon is not opened')
            return False
        
        captransferinfo = self.dcam.cap_transferinfo()
        if captransferinfo is False:
            print('-NG: Dcam.cap_transferinfo() failed with error {}'.format(self.dcam.lasterr().name))
            return False

        if captransferinfo.nFrameCount < 1:
            print('-NG: There are no images retrieved.')
            return False
        
        return (captransferinfo.nNewestFrameIndex, captransferinfo.nFrameCount)

    def save_rawimages(self, prefix):
        """Save acquired images as raw.

        Save acquired and retained images as raw data.
        The output file name is "{prefix} - {frameindex}.raw"
        "frameindex" starts at 1 and is numbered from the oldest image.

        Args:
            prefix (string): prefix of output filename
        
        Returns:
            bool: result
        """
        if self.dcam is None:
            print('-NG: Dcamcon is not opened')
            return False
        
        captransferinfo = self.dcam.cap_transferinfo()
        if captransferinfo is False:
            print('-NG: Dcam.cap_transferinfo() failed with error {}'.format(self.dcam.lasterr().name))
            return False
        
        if captransferinfo.nFrameCount < 1:
            print('-NG: There are no images retrieved.')
            return False
        
        if captransferinfo.nFrameCount > self.__number_of_frames:
            number_of_images = self.__number_of_frames
            start_frameindex = (captransferinfo.nNewestFrameIndex + 1) % self.__number_of_frames
        else:
            number_of_images = captransferinfo.nFrameCount
            start_frameindex = 0
        
        for i in range(0, number_of_images, 1):
            index = (start_frameindex + i) % self.__number_of_frames
            datai = self.dcam.buf_getframedata(index)
            filename = '{} - {}.raw'.format(prefix, i+1)
            datai.tofile(filename)
        
        return True
    
    def get_propertyvalue(self, propid:IntEnum, showerrmsg=True):
        """Get property value.

        Get property value with Dcam.prop_getvalue()
        if showerrmsg is True, it shows error message 
        when Dcam.prop_getvalue() return False.
        showerrmsg defaults to True.
        Set showerrmsg to False when it is meaningful that it is an error. 

        Args:
            propid (IntEnum): DCAM_IDPROP IntEnum
            showerrmsg (bool): if True, print error message.
        
        Returns:
            double: get value if success
            bool: False if failure            
        """
        if self.dcam is None:
            print('-NG: Dcamcon is not opened')
            return False
        
        propvalue = self.dcam.prop_getvalue(propid.value)
        if propvalue is False:
            if showerrmsg:
                print('-NG: Dcam.prop_getvalue({}) failed with error {}'.format(propid.name, self.dcam.lasterr().name))
            return False
        
        return propvalue
    
    def set_propertyvalue(self, propid:IntEnum, val):
        """Set property value.

        Set property value with Dcam.prop_setvalue()
        it shows error message when Dcam.prop_setvalue() return False

        Args:
            propid (IntEnum): DCAM_IDPROP IntEnum. property ID.
            val (double): set value
        
        Returns:
            bool: result
        """
        if self.dcam is None:
            print('-NG: Dcamcon is not opened')
            return False
        
        if not self.dcam.prop_setvalue(propid, val):
            print('-NG: Dcam.prop_setvalue({}, {}) failed with error {}'.format(propid.name, val, self.dcam.lasterr().name))
            return False
        
        return True
    
    def setget_propertyvalue(self, propid:IntEnum, val):
        """Set and get property value.

        Set and get property value with Dcam.prop_setgetvalue().
        If success, it returns get value. If failure, it returns False

        Args:
            propid (IntEnum): DCAM_IDPROP IntEnum. property ID
            val (double): set value
        
        Returns:
            double: get value if success.
            bool: False if failure.
        """
        if self.dcam is None:
            print('-NG: Dcamcon is not opened')
            return False
        
        res = self.dcam.prop_setgetvalue(propid, val)
        if res is False:
            print('-NG: Dcam.prop_setgetvalue({}, {}) failed with error {}'.format(propid.name, val, self.dcam.lasterr().name))
            return False
        
        return res

    def prompt_propvalue(self, propid, restrictmode=PromptRestrictMode.No, restrictval=None):
        """Set property value at the prompt.

        Set property specified by propid at the prompt
        If success, it returns set value.
        If failure, it returns False

        Args:
            propid (IntEnum): DCAM_PROP IntEnum. property ID
        
        Returns:
            None: if property is not available.
            bool: result when property is available
        """
        if self.dcam is None:
            print('-NG: Dcamcon is not opened')
            return False
        
        val = None
        propattr = self.dcam.prop_getattr(propid)
        if propattr is False:
            # error happened
            return None
        
        min = propattr.valuemin
        max = propattr.valuemax
        step = propattr.valuestep
        default = propattr.valuedefault

        if (propattr.attribute & DCAM_PROP.ATTR.EFFECTIVE and
            propattr.attribute & DCAM_PROP.ATTR.WRITABLE and
            min != max):
            proptype = propattr.attribute & DCAM_PROP.TYPE.MASK
            if proptype == DCAM_PROP.TYPE.MODE:    # change value to text
                def gettextvaluelist(propid):
                    """Return supported values.

                    Returns an array of the supported values for text.
                    
                    Args:
                        propid (int): Property ID
                    
                    Returns: value list
                    """
                    currentvalue = min
                    valuelist = []
                    if (restrictmode != PromptRestrictMode.ModeFilter or
                        currentvalue in restrictval):
                        valuelist.append(int(currentvalue))
                    while currentvalue != max:
                        currentvalue = self.dcam.prop_queryvalue(propid, currentvalue, DCAMPROP_OPTION.NEXT)
                        if (restrictmode != PromptRestrictMode.ModeFilter or
                            currentvalue in restrictval):
                            valuelist.append(int(currentvalue))
                    return valuelist
                
                valuelist = gettextvaluelist(propid)
                value_is_good = False
                while not value_is_good:
                    print()
                    prompt = '\nEnter a [value] for ' + str(self.dcam.prop_getname(propid)) + ' between:\n'
                    for textval in valuelist:
                        valuetext = self.dcam.prop_getvaluetext(propid, textval)
                        prompt += '[{}]'.format(int(textval)) + valuetext + '\n'

                    valuetext = self.dcam.prop_getvaluetext(propid, default)
                    prompt += '\n[default] ' + valuetext
                    prompt += '\n\n>'
                    try:
                        instr = input(prompt)
                        val = int(instr)
                    except ValueError:
                        val = int(default)
                        break
                    value_is_good = (val in valuelist)
            elif proptype == DCAM_PROP.TYPE.LONG:
                if (restrictmode == PromptRestrictMode.ClipMinimum and min < restrictval):
                    min = restrictval
                while True:
                    print()
                    prompt = '\nEnter a value for ' + str(self.dcam.prop_getname(propid))
                    prompt += ' between ' + str(int(min))
                    prompt += ' and ' + str(int(max))
                    prompt += ' in steps of ' + str(int(step))
                    prompt += ' [default is ' + str(int(default)) + ']'
                    prompt += '\n\n> '
                    try:
                        instr = input(prompt)
                        val = int(instr)
                    except ValueError:
                        val = int(default)
                        break

                    if (val % int(step) == 0 and
                        val >= int(min) and
                        val <= int(max)):
                        break
            elif proptype == DCAM_PROP.TYPE.REAL:
                if (restrictmode == PromptRestrictMode.ClipMinimum and min < restrictval):
                    min = restrictval
                while True:
                    print()

                    def get_units(unitid):
                        unitlist = {
                            DCAMPROP_UNIT.SECOND: 's',
                            DCAMPROP_UNIT.CELSIUS: '°C',
                            DCAMPROP_UNIT.KELVIN: 'K',
                            DCAMPROP_UNIT.METERPERSECOND: 'm/s',
                            DCAMPROP_UNIT.PERSECOND: '/s',
                            DCAMPROP_UNIT.DEGREE: '°',
                            DCAMPROP_UNIT.MICROMETER: 'µm',
                        }
                        unitstr = unitlist.get(unitid, '')
                        return unitstr
                    
                    unitname = get_units(propattr.iUnit)
                    minstr = '{:.6f}'.format(min).rstrip('0')
                    if minstr[-1] == '.':
                       minstr += '0'
                    maxstr = '{:.6f}'.format(max).rstrip('0')
                    if maxstr[-1] == '.':
                        maxstr += '0'
                    stepstr = '{:.6f}'.format(step).rstrip('0')
                    if stepstr[-1] == '.':
                        stepstr += '0'
                    defstr = '{:.6f}'.format(default).rstrip('0')
                    if defstr[-1] == '.':
                        defstr += '0'
                    prompt = '\nEnter a value for ' + str(self.dcam.prop_getname(propid))
                    prompt += ' between ' + minstr + unitname
                    prompt += ' and ' + maxstr + unitname
                    prompt += ' in steps of ' + stepstr + unitname
                    prompt += ' [default is ' + defstr + unitname + ']'
                    prompt += '\n\n> '
                    try:
                        instr = input(prompt)
                        val = float(instr)
                    except ValueError:
                        val = default
                        break
                    if (val >= min and
                        val <= max):
                        # ignore step for REAL check due to float precision possible problems.
                        # nomally the property has AUTOROUNDING
                        break
        
        if val is not None:
            val = self.set_propertyvalue(propid, val)

        return val
    
    def _prompt_longpropvalue_stack(self, propid, clipmax=False):
        """Get to set value of long property.

        Prompt to input setting value, but not set to DCAM.
        Input value is returned if success

        Args:
            propid (DCAM_IDPROP): property id
            clipmax (bool or int): clip maximum of range if clipmax is not False

        Returns:
            None: attribute does not have EFFECTIVE or(and) WRITABLE
            int: input value
            bool: False if failure. 
        """
        if self.dcam is None:
            print('-NG: Dcamcon is not opened')
            return False
        
        val = None
        propattr = self.dcam.prop_getattr(propid)
        if propattr is False:
            # error happened
            return False
        
        attribute = propattr.attribute
        proptype = attribute & DCAM_PROP.TYPE.MASK
        if proptype != DCAM_PROP.TYPE.LONG:
            # not support
            return False
        
        min = propattr.valuemin
        max = propattr.valuemax
        if clipmax is not False:
            max = clipmax
        step = propattr.valuestep
        default = propattr.valuedefault

        if (attribute & DCAM_PROP.ATTR.EFFECTIVE and
            attribute & DCAM_PROP.ATTR.WRITABLE and
            min != max):
            while True:
                print()
                prompt = '\nEnter a value for ' + str(self.dcam.prop_getname(propid))
                prompt += ' between ' + str(int(min))
                prompt += ' and ' + str(int(max))
                prompt += ' in steps of ' + str(int(step))
                prompt += ' [default is ' + str(int(default)) + ']'
                prompt += '\n\n> '
                try:
                    instr = input(prompt)
                    val = int(instr)
                except ValueError:
                    val = int(default)
                    break

                if (val % int(step) == 0 and
                    val >= int(min) and
                    val <= int(max)):
                    break
            
        return val
    
    def prompt_propvalue_subarray(self):
        """Set property value related subarray at the prompt.

        Set following properties at the prompt.
        DCAM_IDPROP.SUBARRAYMODE,
        DCAM_IDPROP.SUBARRAYHPOS, DCAM_IDPROP.SUBARRAYHSIZE,
        DCAM_IDPROP.SUBARRAYVPOS, DCAM_IDPROP.SUBARRAYVSIZE,
        Control offset and size combinations

        Returns:
            bool: result
        """
        if self.dcam is None:
            print('-NG: Dcamcon is not opened')
            return False
        
        res = self.prompt_propvalue(DCAM_IDPROP.SUBARRAYMODE)
        if res is None:
            # not available
            return True
        elif res is False:
            # error happened
            return False

        res = True
        subarraymode = self.get_propertyvalue(DCAM_IDPROP.SUBARRAYMODE)
        if subarraymode is False:
            # error happened
            return False
        elif subarraymode == DCAMPROP.MODE.OFF:
            # not need setting subarray parameters
            res = True
        else:
            # subarraymode == DCAMPROP.MODE.ON
            def prompt_offset_and_size(offsetid, sizeid):
                """Set subarray offset and size.

                Set subarray offset and size.

                Args:
                    offsetid (DCAM_IDPROP): SUBARRAYHPOS or SUBARRAYVPOS
                    sizeid (DCAM_IDPROP): SUBARRAYHSIZE or SUBARRAYVSIZE
                
                Returns:
                    None: if property is not available.
                """
                propattr_offset = self.dcam.prop_getattr(offsetid)
                if propattr_offset is False:
                    # error happen
                    return False
                
                propattr_size = self.dcam.prop_getattr(sizeid)
                if propattr_size is False:
                    # error happen
                    return False
                
                is_offset_available = (propattr_offset.attribute & DCAM_PROP.ATTR.EFFECTIVE and
                                       propattr_offset.attribute & DCAM_PROP.ATTR.WRITABLE and
                                       propattr_offset.valuemin != propattr_offset.valuemax)
                
                is_size_available = (propattr_size.attribute & DCAM_PROP.ATTR.EFFECTIVE and
                                     propattr_size.attribute & DCAM_PROP.ATTR.WRITABLE and
                                     propattr_size.valuemin != propattr_size.valuemax)
                
                if (is_offset_available and
                    is_size_available):
                    # set both offset and size
                    offsetval = self._prompt_longpropvalue_stack(offsetid)
                    if offsetval is False:
                        # error happened
                        res = False
                    elif offsetval is None:
                        # not need to set offset. prompt to set size
                        res = self.prompt_propvalue(sizeid)
                    else:
                        # offsetval is int value. temporarily suspend setting
                        # clip the maximum of size by subtracting offsetval
                        clipmax = propattr_size.valuemax - offsetval
                        sizeval = self._prompt_longpropvalue_stack(sizeid, clipmax)
                        if sizeval is False:
                            res = False
                        elif sizeval is None:
                            # not need to set size. set offset value
                            res = self.set_propertyvalue(offsetid, offsetval)
                        else:
                            # sizeval is int value. need to consider setting order
                            cursize = self.get_propertyvalue(sizeid)
                            if cursize is False:
                                res = False
                            else:
                                if sizeval < cursize:
                                    if (self.set_propertyvalue(sizeid, sizeval) and
                                        self.set_propertyvalue(offsetid, offsetval)):
                                        res = True
                                    else:
                                        res = False
                                else:
                                    if (self.set_propertyvalue(offsetid, offsetval) and
                                        self.set_propertyvalue(sizeid, sizeval)):
                                        res = True
                                    else:
                                        res = False
                elif is_offset_available:
                    # prompt to set offset only
                    res = self.prompt_propvalue(offsetid)
                elif is_size_available:
                    # prompt to set size only
                    res = self.prompt_propvalue(sizeid)
                else:
                    # nothing to configure
                    res = True
                
                return res
            
            # horizontal
            if prompt_offset_and_size(DCAM_IDPROP.SUBARRAYHPOS, DCAM_IDPROP.SUBARRAYHSIZE) is False:
                return False

            # vertical
            if prompt_offset_and_size(DCAM_IDPROP.SUBARRAYVPOS, DCAM_IDPROP.SUBARRAYVSIZE) is False:
                return False
            
            res = True
        
        return res

def dcamcon_init():
    """Initialize DCAM and make device list.

    Initialize DCAM-API and make device list.

    If Dcamapi.init() is already called, it returns True.
    If not, it calls Dcamapi.init().
    If Dcamapi.init() returns True, 

    Return:
        bool: retuslt of initialization
    """
    global called_dcamapi_init
    if called_dcamapi_init:
        # dcamapi_init() is already called
        return True
    
    print('Calling Dcamapi.init()')
    if not Dcamapi.init():
        print('-NG: Dcamapi.init() failed with error {}'.format(Dcamapi.lasterr().name))
        # should call Dcamapi.uninit to call Dcamapi.init() again if if fails
        Dcamapi.uninit()
        return False
    
    called_dcamapi_init = True

    # check number of detected devices.
    cameracount = Dcamapi.get_devicecount()
    if cameracount <= 0:
        print('-NG: Dcamapi.init() succeeded but not device is available.')
        return False
    
    # update device list
    global dcamcon_list
    dcamcon_list = []
    for icamera in range(cameracount):
        dcam = Dcam(icamera)
        device_title = '#[{}]: '.format(icamera)

        # check model string of the device
        model = dcam.dev_getstring(DCAM_IDSTR.MODEL)
        text = ''
        if model is False:
            text = 'NO MODEL'
        else:
            text = 'MODEL={}'.format(model)
        
        device_title += text

        # check cameraid string of the device
        cameraid = dcam.dev_getstring(DCAM_IDSTR.CAMERAID)
        text = ''
        if cameraid is False:
            text = ', NO CAMERAID'
        else:
            text = ', CAMERAID={}'.format(cameraid)
        
        device_title += text

        # check bus string of the device
        bus = dcam.dev_getstring(DCAM_IDSTR.BUS)
        text = ''
        if bus is False:
            text = ', NO BUS'
        else:
            text = ', BUS={}'.format(bus)
        
        device_title += text

        # create and initialize Dcamcon instance
        my = Dcamcon()
        my.iCamera = icamera
        my.device_title = device_title
        my.dcam = None

        # append to dcamcon_list
        dcamcon_list.append(my)
    
    return True

def dcamcon_uninit():
    """Clear device list and uninitialize DCAM.

    Clear device list and uninitialize DCAM-API
    """
    # close device
    global dcamcon_list
    for my in dcamcon_list:
        if my.dcam is None:
            # not opened or closed
            continue
        
        my.stopcapture()    # Stop capturing. No effect if already stopped

        if my.is_capstatus_ready():
            my.releasebuffer()

        my.close()
        my.dcam = None
    
    # clear device list
    dcamcon_list.clear()

    # uninitialize DCAM-API
    global called_dcamapi_init
    if called_dcamapi_init:
        Dcamapi.uninit()
        called_dcamapi_init = False

def dcamcon_choose_and_open():
    """Choose device.
    
    Choose DCAM device from dcamcon_list and Dcamcon instance

    return:
        Dcamcon: Dcamcon instance opened device. it returns None when open is failed.
    """
    global dcamcon_list
    devicecount = len(dcamcon_list)
    if devicecount <= 0:
        print('-NG: No device is available.')
        return None
    
    idevice = 0
    if devicecount == 1:
        print(dcamcon_list[0].device_title)
    else:
        # mean devicecount > 1

        # print device list
        devicelist = ''
        for dcamcon in dcamcon_list:
            devicelist += devicelist + dcamcon.device_title + '\n'
        
        print(devicelist)

        # choose device index
        fmt = '\n# Choose device index between 0 - {}. [default] is 0\n '
        prompt = fmt.format(devicecount - 1)
        while True:
            instr = input(prompt)
            if instr == '':
                # default index
                idevice = 0
                break
            
            try:
                idevice = int(instr)
            except ValueError:
                idevice = -1
            
            if (idevice >= 0 and
                idevice < devicecount):
                break
        
    dcam = Dcam(idevice)
    if not dcam.dev_open():
        print('-NG: Dcam.dev_open() failed with error {}'.format(dcam.lasterr().name))
        return None
    
    dcamcon_list[idevice].dcam = dcam
    return dcamcon_list[idevice]

