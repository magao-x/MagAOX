
# Control of a Tripp Lite PDU

The MagAOX App to control a Tripp Lite PDU

## Build

This application can be built using the standard MagAO-X build system by issuing the command
```
$ make -f ../../Make/magAOXApp.mk t=trippLitePDU
```
in the trippLitePDU directory.

After the build is successful, do:

```
$ sudo chown root:root trippLitePDU`
$ sudo chmod +s trippLitePDU`
```

## Running

For this to work, you have to have setup the MagAO-X config and logging system (environment variables and a directory structure).  Then you type:

`$ ./trippLitePDU`
 
## Configuration

The following configurable options are accepted by trippLitePDU:

  Required arguments:
   None.
  
  Optional arguments:

   |Short | Long         |  Config-File         |     T               | Description  |
   | ---  | ---          | ---                  |   ---               | ---          |
   | -c | --config       |      config          |    <string>         | A local config file    |
   | -h | --help         |                      |     <none>          | Print this message and exit    |
   | -p | --loopPause    | loopPause            |     <unsigned long> | The main loop pause time in ns |
   | -P | --RTPriority   | RTPriority           |     <unsigned>      | The real-time priority (0-99)  |
   | -L | --logDir       | logger.logDir        |     <string>        | The directory for log files    |
   |    | --logExt       | logger.logExt        |     <string>        | The extension for log files    |
   |    | --maxLogSize   | logger.maxLogSize    |     <string>        | The maximum size of log files  |
   |    | --writePause   | logger.writePause    |     <unsigned long> | The log thread pause time in ns |
   |    |--logThreadPrio | logger.logThreadPrio |     <int>           | The log thread priority        |
   | -l | --logLevel     | logger.logLevel      |     <string>        | The log level                  |
   |    | --idVendor     | usb.idVendor         |     <<string>>      | USB vendor id, 4 digits        |
   |    | --idProduct    | usb.idProduct        |     <<string>>      | USB product id, 4 digits       |
   |    | --serial       | usb.serial           |     <<string>>      | USB serial number        |
   |    | --baud         | usb.baud             |     <real>          | USB tty baud rate (i.e. 9600) | 
   | -n | --name         | name                 |     <string>        | The name of the application, specifies config. | 
   
   
