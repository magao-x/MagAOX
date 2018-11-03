trippLitePDU
==========

[TOC]

------------------------------------------------------------------------

# NAME 

trippLitePDU − The MagAOX App to control a Tripp Lite PDU

# SYNOPSIS 

```
trippLitePDU [options] 
```

`trippLitePDU` is normally configured with a configuration file, hence all command-line arguments are optional.

# DESCRIPTION 

`trippLitePDU` provides an interface to a Tripp Lite power distribution unit.  INDI properties provide the status of each outlet, as well as line power status.

# OPTIONS

   |Short | Long                   |  Config-File              |  Type         | Description  |
   | ---  | ---                    | ---                       |   ---         | ---          |
   | -c | --config                 |      config               | string        | A local config file    |
   | -h | --help                   |                           | none          | Print this message and exit    |
   | -p | --loopPause              | loopPause                 | unsigned long | The main loop pause time in ns |
   | -P | --RTPriority             | RTPriority                | unsigned      | The real-time priority (0-99)  |
   | -L | --logDir                 | logger.logDir             | string        | The directory for log files    |
   |    | --logExt                 | logger.logExt             | string        | The extension for log files    |
   |    | --maxLogSize             | logger.maxLogSize         | string        | The maximum size of log files  |
   |    | --writePause             | logger.writePause         | unsigned long | The log thread pause time in ns |
   |    |--logThreadPrio           | logger.logThreadPrio      | int           | The log thread priority        |
   | -l | --logLevel               | logger.logLevel           | string        | The log level                  |
   | -n | --name                   | name                      | string        | The name of the application, specifies config. |
   | -a |--device.address          | device.address            | string        | The device address.                   | 
   | -p |--device.port             | device.port               | string        | The device port.                     |
   | -u |--device.username         | device.username           | string        | The device login username.               |
   |    |--device.passfile         | device.passfile           | string        | The device login password file (relative to secrets dir).    |
   |    |--timeouts.write          | timeouts.write            | int           | The timeout for writing to the device [msec]. Default = 1000  |
   |    |--timeouts.read           | timeouts.read             | int           | The timeout for reading the device [msec]. Default = 2000      |
   |    |--timeouts.outletStateDel | timeouts.outletStateDelay | int           | The maximum time to wait for an outlet to change state [msec]. Default = 5000|

# EXIT STATUS

`trippLitePDU` runs until killed or a critical error occurs.


# EXAMPLES


# SEE ALSO 
