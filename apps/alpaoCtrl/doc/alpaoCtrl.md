alpaoCtrl {#page_module_alpaoCtrl}
==========

[TOC]

------------------------------------------------------------------------

# NAME 

alpaoCtrl − Control an ALPAO DM using Milk shared memory streams.

# SYNOPSIS 

```
alpaoCtrl [options] 
```

`alpaoCtrl` is normally configured with a configuration file, hence all command-line arguments are optional. But note that if the `-n name` option is not given, then a configuration file named `alpaoCtrl.conf` must be available at the MagAO-X standard config path.

# DESCRIPTION 

`alpaoCtrl` manages an ALPAO deformable mirror, providing initialization and safe shutdown (release) functions, as well flat and test command processing.  Additionally, real-time optimized response to a CACAO `dmcomb` triggered command is facilitated by a separate thread. 

# OPTIONS

|Short | Long                 |    Config-File*      |     Type          | Description  |
| ---  | ---                  | ---                  |   ---             | ---          |
|   -c | --config             | config               |   string          | A local config file |
|   -h | --help               |                      |   none            | Print this message and exit | 
|  -n  | --name               | name                 |    string         | The name of the application, specifies config.
|   -p | --loopPause          | loopPause            |   unsigned long   | The main loop pause time in ns |
|   -L | --logDir             | logger.logDir        |   string          | The directory for log files  | 
|      | --logExt             | logger.logExt        |   string          | The extension for log files  | 
|      | --maxLogSize         | logger.maxLogSize    |   string          | The maximum size of log files | 
|      | --writePause         | logger.writePause    |   unsigned long   | The log thread pause time in ns |                                                                                                
|      | --logThreadPrio      | logger.logThreadPrio |     int           | The log thread priority   |
|   -l | --logLevel           | logger.logLevel      |     string        | The log level   | 
|      | --power.device       | power.device         |    <string>       | Device controlling power for this app's  device (INDI name).                      
|      | --power.channel      | power.channel        |    <string>       | Channel on device for this app's device (INDI name).                                   
|      | --power.element      | power.element        |    <string>       | INDI element name.  Default is "state", onlyneed to specify if different.            
|      | --dm.serialNumber    | dm.serialNumber      |    <string>       | The ALPAO serial number used to find the default config directory.                
|      | --dm.calibPath       | dm.calibPath         |    <string>       | The path to calibration files, relative tothe MagAO-X calibration path.            
|      | --dm.flatPath        | dm.flatPath          |    <string>       | The path to flat files.  Default is thecalibration path.                        
|      | --dm.flat            | dm.flat              |    <string>       | The name of the flat file, a FITS filecontaining the flat command for this DM.  Must be in the directdory flatPath.      
|      | --dm.threadPrio      | dm.threadPrio        |    <int>          | The real-time priority of the dm control thread.                                  
|      | --dm.shmimName       | dm.shmimName         |    <string>       | The name of the ImageStreamIO shared memory image to monitor for DM comands. Will be used as /tmp/<shmimName>.im.shm.              
|      | --dm.shmimFlat       | dm.shmimFlat         |    <string>       | The name of the ImageStreamIO shared memoryimage to write the flat command to.  Default is shmimName with 00 apended (i.e. dm00disp -> dm00disp00).                          
|      | --dm.shmimTest       | dm.shmimTest         |    <string>       | The name of the ImageStreamIO shared memory image to write the test command to.  Defaultis shmimName with 01 apended (i.e. dm00disp -> dm00disp01).                          
|      | --dm.width           | dm.width             |    <string>       | The width of the DM in actuators.        
|      | --dm.height          | dm.height            |    <string>       | The height of the DM in actuators.       
                                                                           

`*` format in the config file column is section.option which implies the format 
```
[section]
option=value
```
in the config file.

# INDI PROPERTIES

## Read-Only INDI Properties

list them here

## Read-Write INDI Properties

list them here

# EXIT STATUS

`alpaoCtrl` runs until killed.  Use the `logdump` utility to examine the process log for errors.


# EXAMPLES

To start the ALPAP controller for the woofer deformable mirror in MagAO-X:
```
/opt/MagAOX/bin/alpaoCtrl -n dmwoofer
```

# SEE ALSO 

[Source code.](../sw_html/group__alpaoCtrl.html)
