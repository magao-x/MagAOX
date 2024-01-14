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