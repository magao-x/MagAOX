xt1121DCDU  {#page_module_xt1121DCDU}
==========

[TOC]

------------------------------------------------------------------------

# Name

xt1121DCDU − The MagAOX App to control an xt1121 based DCDU

# Synopsis 

```
xt1121DCDU [options] 
```

`xt1121DCDU` is normally configured with a configuration file.  Typically the only command-line argument is `-n` to specify the configuration name.

# Description 

`xt1121DCDU` provides an interface to a xt1121-based D.C. distribution unit.  This is an Electronics-Salon D-228 relay module controlled by an Acromag xt1121 digital I/O module.   
INDI properties provide the status of each outlet.

# Options

   |Short | Long                      |  Config-File              |  Type         | Description  |
   | ---  | ---                       | ---                       |   ---         | ---          |
   | -c | --config                    |      config               | string        | A local config file    |
   | -h | --help                      |                           | none          | Print this message and exit    |
   | -p | --loopPause                 | loopPause                 | unsigned long | The main loop pause time in ns |
   | -P | --RTPriority                | RTPriority                | unsigned      | The real-time priority (0-99)  |
   | -L | --logDir                    | logger.logDir             | string        | The directory for log files    |
   |    | --logExt                    | logger.logExt             | string        | The extension for log files    |
   |    | --maxLogSize                | logger.maxLogSize         | string        | The maximum size of log files  |
   |    | --writePause                | logger.writePause         | unsigned long | The log thread pause time in ns |
   |    | --logThreadPrio             | logger.logThreadPrio      | int           | The log thread priority        |
   | -l | --logLevel                  | logger.logLevel           | string        | The log level                  |
   | -n | --name                      | name                      | string        | The name of the application, specifies config. |
   |    | --power.device              | power.device              | string        | Device controlling power for this app's device (INDI name). |
   |    | --power.channe              | power.channel             | string        | Channel on device for this app's device (INDI name). |
   |    | --power.element              | power.element             | string        | INDI element name.  Default is "state", only need to specify if different.
   |    | --device.name               | device.name               | string        | The device INDI name.                   | 
   |    | --device.channelNumbers     | device.channelNumbers     | vector<int>   | The channel numbers to use for the outlets,in order.    |

## Channel Configuration

The outlets are controlled in channels, which consist of at least one outlet.  Channels are configured as sections in the configuration file.  Any section name, say `[channel1]`, which has either a `oulet=` or `outlets=` keyword will be treated as a channel specification.

The `oulet=` or `outlets=` keyword=value pair specifies which outlet or outletss, 1-8, are controlled by this channel. Multiple outlets are specified in an comma separate list.

You can also specify the order in which the outlets are turnned on with the `onOrder` and `offOrder` keywords.  The values contain indices in the vector specified by the `outlet`/`outlets` keyword, not the outlet numbers.  So if you have `outlets=7,8` you would then have `onOrder=1,0` to turn on outlet 8 first, then outlet 7.

You can use `onDelays` and `offDelays`, to specify the delays between outlet operations in milliseconds.  The first entry is always ignored, then the second entry specifies the delay between the first and second outlet operation, etc.

An example config file section is:
```
[sue]           #this channel will be named sue
outlets=4,5     #this channel uses outlets 4 and 5
onOrder=1,0     #outlet 5 will be turned on first
offOrder=0,1    #Outlet 4 will be turned off first
onDelays=0,150  #a 150 msec delay between outlet turn on
offDelays=0,345 #a 345 msec delay between outlet turn off
```


# INDI

## Read/write Properties

Channel states are reported by the property with the channel name, and state changes can be requested by the same property.  The `target` element contains the last requested state until the `state` element, which is the current state, matches it.  Changing either `state` or `target` will request a state change in that channel.
```
<name>.<channelName>.state = On | Int | Off
<name>.<channelName>.target = On | Off | [empty]
```

## Read-only Properties


Each outlet has a property which indicates its current state.
```
<name>.outlet<N>.state = On | Int | Off
```

# Examples

This app is started for DCDU 0, which has config file `dcdu0.conf`, using
```
/opt/MagAOX/bin/xt1121DCDU -n dcdu0
```

# Troubleshooting


# SEE ALSO 

[Source code.](../sw_html/group__xt1121DCDU.html)
