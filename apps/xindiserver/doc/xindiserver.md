xindiserver
==========

[TOC]

------------------------------------------------------------------------

# NAME 

xindiserver − wrapper for `indiserver`, integrating it into the MagAO-X ecosystem.

# SYNOPSIS 

```
xindiserver [options] 
```

`xindiserver` is normally configured with a configuration file, hence all command-line arguments are optional.

# DESCRIPTION 

`xindiserver` wraps the standard `indiserver` program in an MagAO-X interface.  This includes exposing configuration options, and capturing logs which are reformatted in the `flatlogs` binary logging system.

# OPTIONS

|Short | Long            |    Config-File        |     Type          | Description  |
| ---  | ---             | ---                   |   ---             | ---          |
|   -c | --config        |    config             |   string          | A local config file |
|   -h |--help           |                       |   none            | Print this message and exit | 
|   -p | --loopPause     |    loopPause          |   unsigned long   | The main loop pause time in ns |
|   -P |--RTPriority     |   RTPriority          |   unsigned        | The real-time priority (0-99) | 
|   -L |--logDir         |   logger.logDir       |   string          | The directory for log files  | 
|      |--logExt         |   logger.logExt       |   string          | The extension for log files  | 
|      |--maxLogSize     |   logger.maxLogSize   |   string          | The maximum size of log files | 
|      |--writePause     |   logger.writePause   |   unsigned long   | The log thread pause time in ns |                                                                                                
|      | --logThreadPrio | logger.logThreadPrio  |     int           | The log thread priority   |
|   -l | --logLevel      | logger.logLevel       |     string        | The log level   | 
|   -m |                 | indiserver.m          |     int           | indiserver kills client if it gets more  than this many MB behind, default 50 |
|  -n  |                 | indiserver.n          |     bool          | indiserver: ignore /tmp/noindi 
|  -p  |                 | indiserver.p          |     int           | indiserver: alternate IP port, default 7624                   
|  -v  |                 | indiserver.v          |     int           | indiserver: log verbosity, -v, -vv or -vvv                        
|  -x  |                 | indiserver.x          |     bool          | exit after last client disconnects -- FOR PROFILING ONLY          
|  -L  | --local         | local.drivers         |    vector string  | List of local drivers to start.                                                                                                
|  -R  | --remote        | remote.drivers        |    vector string  | List of remote drivers to start, in the form of name\@hostname without the port.  Hostname needs an entry in remote.hosts                            
|  -H  | --hosts         | remote.hosts          |    vector string  | List of remote hosts, in the form of hostname[:remote_port]:local_port.  remote_port is optional if it is the INDI default.
|  -n  | --name          | name                  |    string         | The name of the application, specifies config.
    
# DRIVER SPECIFICATIONS

To-do: describe drivers specs.

# EXIT STATUS

`xindiserver` runs until killed.


# EXAMPLES


# SEE ALSO 
