from enum import Enum

class StateCodes(Enum):
    FAILURE = -20      # The application has failed should be used when m_shutdown is set for an error.
    ERROR = -10        # The application has encountered an error from which it is recovering (with or without intervention)
    UNINITIALIZED = 0  # The application is unitialized the default
    INITIALIZED = 1    # The application has been initialized set just before calling appStartup().
    NODEVICE = 2       # No device exists for the application to control.
    POWEROFF = 4       # The device power is off.
    POWERON = 6        # The device power is on.
    NOTCONNECTED = 8   # The application is not connected to the device or service.
    CONNECTED = 10     # The application has connected to the device or service.
    LOGGEDIN = 15      # The application has logged into the device or service
    CONFIGURING = 20   # The application is configuring the device.
    NOTHOMED = 24      # The device has not been homed.
    HOMING = 25        # The device is homing.
    OPERATING = 30     # The device is operating other than homing.
    READY = 35         # The device is ready for operation but is not operating.
    SHUTDOWN = 10000   # The application has shutdown set just after calling appShutdown().

import pathlib
import os
import re

LOOKYLOO_DATA_ROOTS = ['/opt/MagAOX', '/srv/icc/data', '/srv/rtc/data']
QUICKLOOK_PATH = pathlib.Path('/data/obs')
LOG_PATH = pathlib.Path('/opt/MagAOX/logs/lookyloo')
HISTORY_FILENAME = ".lookyloo_succeeded"
FAILED_HISTORY_FILENAME = ".lookyloo_failed"
DEFAULT_SEPARATE = object()
DEFAULT_CUBE = object()
ALL_CAMERAS = {
    'camsci1': DEFAULT_SEPARATE,
    'camsci2': DEFAULT_SEPARATE,
    'camlowfs': DEFAULT_SEPARATE,
    'camwfs': DEFAULT_SEPARATE,
    'camtip': DEFAULT_SEPARATE,
    'camacq': DEFAULT_SEPARATE,
}
AUTO_EXPORT_CAMERAS = ['camsci1', 'camsci2']
SLEEP_FOR_TELEMS = 5
CHECK_INTERVAL_SEC = 30
LINE_BUFFERED = 1
XRIF2FITS_TIMEOUT_SEC = 120

# note: we must truncate to microsecond precision due to limitations in
# `datetime`, so this pattern works only after chopping off the last
# three characters
MODIFIED_TIME_FORMAT = "%Y%m%d%H%M%S%f"
PRETTY_MODIFIED_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
OBSERVERS_DEVICE = "observers"
LINE_FORMAT_REGEX = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\.(\d{6})(?:\d{3}) "
    r"TELM \[observer\] email: (.*) obs: (.*) (\d)"
)
FOLDER_TIMESTAMP_FORMAT = '%Y-%m-%d_%H%M%S'