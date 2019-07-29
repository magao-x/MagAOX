<template> {#page_module_<template>}
==========

[TOC]

------------------------------------------------------------------------

# NAME 

<template> − [does something]

# SYNOPSIS 

```
<template> [options] 
```

`<template>` is normally configured with a configuration file, hence all command-line arguments are optional. But note that if the `-n name` option is not given, then a configuration file named `<template>.conf` must be available at the MagAO-X standard config path.

# DESCRIPTION 

`<template>` [does stuff in detail]

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

[and other stuff]

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

`<template>` runs until killed.  Use the `logdump` utility to examine the process log for errors.


# EXAMPLES

To start the [description of app or template] for [context]:
```
/opt/MagAOX/bin/<template> -n [contextname]
```

# SEE ALSO 

[Source code.](../sw_html/group__template.html)
