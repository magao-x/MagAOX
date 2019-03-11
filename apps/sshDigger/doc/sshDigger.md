sshDigger {#page_module_sshDigger}
==========

[TOC]

------------------------------------------------------------------------

# NAME 

sshDigger − the MagAO-X SSH tunnel manager

# SYNOPSIS 

```
$ sshDigger [options] -n tunnel_name
```

The `tunnel_name` denotes the section in the configuration file(s) which contains the specification of the tunnel.  `sshDigger` is normally configured via a base configuration file, hence all other command-line arguments are optional.

# DESCRIPTION 

`sshDigger` uses the `autossh` utility to form a robust `SSH` tunnel or port forward to a remote host.  In addition, the forked `autossh` process is monitored, and if it dies a new one is created. 

The base configuration is normally located at `/opt/MagAOX/config/sshTunnels.conf`.  It should contain options applicable to all tunnels, as well as the tunnel definitions themselves.

The tunnel name must be specified with the `-n` command line option.

This app does not require that an instance specific configuration `tunnel_name.conf` be available.  If one is available matching the name given with the `-n tunnel_name` option, then any settings contained therein will override those given in the base config file.

# Tunnel Specification

Tunnels are specified by a section in the configuration files, normally the base `sshTunnels.conf` file.  The section must have the following members
```
[tunnel_name]
remoteHost=resolvable_name
localPort=X
remotePort=Y
```

Where
- `resolvable_name` is an ip address or host name.  This can include a user name `user@` at the beginning if needed.
- `X` denotes the integer local port number.
- 'Y' denotes the integer remote port number

This results in `ssh` being started with 
```
$ ssh -nNTL X:localhost:Y resolvable_name
```
by the `autossh` utility.

# OPTIONS


|Short | Long                 |    Config-File*      |     Type          | Description  |
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

TODO: add more of the config options, make -n required.


`*` format in the config file column is section.option which implies the format: 
```
[section]
option=value
```
in the config file.

# INDI PROPERTIES

This app does not use INDI

# EXIT STATUS

`sshDigger` runs until killed, and will restart `autossh`, which in turn restarts `ssh`, as long as it is running.  Use the `logdump` utility to examine the process log for errors.


# EXAMPLES

To create an SSH tunnel for `magaox_aoc_to_rtc_indi`:
```
$ /opt/MagAOX/bin/sshDigger -n magaox_aoc_to_rtc_indi
```

Which expects a configuration entry of the form:
```
[magaox_aoc_to_rtc_indi]
remoteHost=rtc 
localPort=7630
remotePort=7624
```

This then securely forwards traffic from `localhost:7630` to the INDI server on `rtc:7624`.


# SEE ALSO 

[Source code.](../sw_html/group__sshDigger.html)
