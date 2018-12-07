filterWheelCtrl
==========

[TOC]

------------------------------------------------------------------------

# NAME 

filterWheelCtrl − controls a Faulhaber MCBL 3006S based filter wheel for MagAO-X.

# SYNOPSIS 

```
filterWheelCtrl [options] 
```

`filterWheelCtrl` is normally configured with a configuration file, hence all command-line arguments are optional. But note that if the `-n name` option is not given, then a configuration file named `filterWheelCtrl.conf` must be available at the MagAO-X standard config path.

# DESCRIPTION 

`filterWheelCtrl` controls an MCBL 3006S based filter wheel for MagAO-X.  As of Dec, 2018, there are 6 such wheels in MagAO-X.  This program communicates with the MCBL 3006S controller via USB.  It monitors the position and motion status of the wheel, and via INDI accepts commands to home and change wheel position.

# C.L. OPTIONS

|Short | Long                 |    Config-File       |     Type          | Description  |
| ---  | ---                  | ---                  |   ---             | ---          |
|   -c | --config             | config               |   string          | A local config file |
|   -h | --help               |                      |   none            | Print this message and exit | 
|   -p | --loopPause          | loopPause            |   unsigned long   | The main loop pause time in ns |
|   -P | --RTPriority         | RTPriority           |   unsigned        | The real-time priority (0-99) | 
|   -L | --logDir             | logger.logDir        |   string          | The directory for log files  | 
|      | --logExt             | logger.logExt        |   string          | The extension for log files  | 
|      | --maxLogSize         | logger.maxLogSize    |   string          | The maximum size of log files | 
|      | --writePause         | logger.writePause    |   unsigned long   | The log thread pause time in ns |                                                                                                
|      | --logThreadPrio      | logger.logThreadPrio |     int           | The log thread priority   |
|   -l | --logLevel           | logger.logLevel      |     string        | The log level   | 
|  -n  | --name               | name                 |    string         | The name of the application, specifies config.
|      | --power.device       | power.device         |    string         | Device controlling power for this app's device (INDI name).
|      | --power.outlet       | power.outlet         |    string         | Outlet (or channel) on device for this app's device (INDI name).
|      | --power.element      | power.element        |    string         | INDI element name.  Default is "state", only need to specify if different.
|      | --usb.idVendor       | usb.idVendor         |    string         | USB vendor id, 4 digits
|      | --usb.idProduct      | usb.idProduct        |    string         | USB product id, 4 digits
|      | --usb.serial         | usb.serial           |    string         | USB serial number 
|      | --usb.baud           | usb.baud             |    real           | USB tty baud rate (i.e. 9600) 
|      | --timeouts.write     | timeouts.write       | int               |    The timeout for writing to the device [msec]. Default = 1000
|      | --timeouts.read      | timeouts.read        | int               |    The timeout for reading the device [msec]. Default = 1000
|      | --motor.acceleration | motor.acceleration   | real              |     The motor acceleration parameter. Default=1000. 
|      | --motor.speed        | motor.speeed         | real              |       The motor speed parameter.  Default=1000.
|      | --motor.circleSteps  | motor.circleSteps    | long              |      The number of steps in 1 revolution.
|      | --motor.homeOffset   | motor.homeOffset     | long              |     The homing offset in motor counts.
|      | --motor.powerOnHome  | motor.powerOnHome    | bool              |     If true, home at startup/power-on. Default=false.
|      | --filters.names      | filters.names        | vector<string>    | The names of the filters. 
|      | --filters.positions  | filters.positions    | vector<double>    | The positions of the filters.  If omitted or 0 then order is used.

# INDI PROPERTIES

## Read-Only INDI Properties

list them here

## Read-Write INDI Properties

list them here

# EXIT STATUS

`filterWheelCtrl` runs until killed.  Use the `logdump` utility to examine the process log for errors.


# EXAMPLES

To start the filter wheel controller for science camera 1:
```
/opt/MagAOX/bin/filterWheelCtrl -n fwsci1
```

# SEE ALSO 
