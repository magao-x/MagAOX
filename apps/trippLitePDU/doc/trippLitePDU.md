trippLitePDU  {#page_module_trippLitePDU}
==========

[TOC]

------------------------------------------------------------------------

# Name

trippLitePDU − The MagAOX App to control a Tripp Lite PDU

# Synopsis 

```
trippLitePDU [options] 
```

`trippLitePDU` is normally configured with a configuration file.  Typically the only command-line argument is `-n` to specify the configuration name.

# Description 

`trippLitePDU` provides an interface to a Tripp Lite power distribution unit.  INDI properties provide the status of each outlet, as well as line power status.

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
   | -a | --device.address            | device.address            | string        | The device address.                   | 
   | -p | --device.port               | device.port               | string        | The device port.                     |
   | -u | --device.username           | device.username           | string        | The device login username.               |
   |    | --device.passfile           | device.passfile           | string        | The device login password file (relative to secrets dir).    |
   |    | --device.readTimeout        | device.readTimeout        | int           | timeout for reading from device |
   |    | --device.writeTimeout       | device.writeTimeout       | int           | timeout for writing to device |
   |    | --timeouts.outletStateDelay | timeouts.outletStateDelay | int           | The maximum time to wait for an outlet to change state [msec]. Default = 5000|
   |    | --limits.freqLowWarn        | limits.freqLowWarn        | int           |  The low-frequency warning threshold |
   |    | --limits.freqHighWarn       | limits.freqHighWarn       | int           |  The high-frequency warning threshold |
   |    | --limits.freqLowAlert       | limits.freqLowAlert       | int           |  The low-frequency alert threshold    |
   |    | --limits.freqHighAlert      | limits.freqHighAlert      | int           |  The high-frequency alert threshold    | 
   |    | --limits.freqLowEmerg       | limits.freqLowEmerg       | int           |  The low-frequency emergency threshold  | 
   |    | --limits.freqHighEmerg      | limits.freqHighEmerg      | int           |  The high-frequency emergency threshold | 
   |    | --limits.voltLowWarn        | limits.voltLowWarn        | int           |  The low-voltage warning threshold      | 
   |    | --limits.voltHighWarn       | limits.voltHighWarn       | int           |  The high-voltage warning threshold     | 
   |    | --limits.voltLowAlert       | limits.voltLowAlert       | int           |  The low-voltage alert threshold        | 
   |    | --limits.voltHighAlert      | limits.voltHighAlert      | int           |  The high-voltage alert threshold       |
   |    | --limits.voltLowEmerg       | limits.voltLowEmerg       | int           |  The low-voltage emergency threshold      |
   |    | --limits.voltHighEmerg      | limits.voltHighEmerg      | int           |  The high-voltage emergency threshold     |
   |    | --limits.currWarn           | limits.currWarn           | int           |  The high-current warning threshold       |
   |    | --limits.currAlert          | limits.currAlert          | int           |  The high-current alert threshold         |
   |    | --limits.currEmerg          | limits.currEmerg          | int           |  The high-current emergency threshold     |

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

The status property gives the reported device status.  This does not appear to be useful.
```
<name>.status.value = ?
```

The `load` property gives the line frequency, line voltage, and drawn current as reported by the device.
```
<name>.load.frequency = [Hz] 
<name>.load.voltage = [V]
<name>.load.current = [A]
```

Each outlet has a property which indicates its current state.
```
<name>.outlet<N>.state = On | Int | Off
```

# Examples

This app is started for PDU 0, which has config file `pdu0.conf`, using
```
/opt/MagAOX/bin/trippLitePDU -n pdu0
```

# Troubleshooting

## Device not responding
If the device stops responding on the CLI port.  A fix is to login via ssh, e.g.
```
$ ssh localadmin@x.x.x.x
```
You will need the current password for this device.  Once logged in, navigate to the menu choice to `Restart PowerAlert`.
```
2- System Configuration

   7- Restart PowerAlert
   
      1- Restart PowerAlert Now
```
This can take a loooooong time to reboot.

If the software reboot doesn't fix it, reset the device using the recessed reset button under the ethernet adapter.

If that doesn't work, you need to completely unplug the device to force a full reset.

# SEE ALSO 

[Source code.](../sw_html/group__trippLitePDU.html)
