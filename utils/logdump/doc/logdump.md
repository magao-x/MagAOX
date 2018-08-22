logdump {#logdump}
==========

[TOC]

------------------------------------------------------------------------

# NAME 


logdump − prints MagAO-X binary log files in a readable format.

# SYNOPSIS 

```
logdump [options] appname
```

# DESCRIPTION


logdump reads the log file(s) for the app specified with `appname`.  Possible modes of operation are:
- Print entries from all log files present (default if no other options are given)
- Print just the N most recent files, specified with `--nfiles``
- Follow the application's logs, printing each new entry untill logdump is killed.
- In any of these cases, the minimum log level can be specified.
- In any of these cases, the output can be limited to a specific event code or set of codes.

`logdump` by default looks in the MagAO-X system log directory, normally `/opt/MagAOX/logs`.  This can be changed with thte `--dir` option.


# OPTIONS 


|Short | Long         |  Config-File         |     Type               | Description  |
| ---  | ---          | ---                  |   ---               | ---          |
| -c | --config      |  config      |              string      |  A local config file  |           
| -h | --help       |               |              none        |  Print this message and exit |
| -p | --pauseTime  |   pauseTime  |               int        |   When following, time in milliseconds to pause before checking for new entries. |
| -F  | --fileCheckInterval |  fileCheckInterval    |     int   |  When following, number of pause intervals between checks for new  files. |
|-d   | --dir      |     dir   |                    string    |    Directory to search for logs. MagAO-X default is normally used. |
| -e  | --ext    |       ext   |                    string   |     The file extension of log files. MagAO-X default is normally used. |
| -n  |--nfiles   |      nfiles  |                   int    |      Number of log files to dump.  If 0, then all matching files dumped. Default: 0, 1 if following. |
| -f  | --follow  |      follow  |                  bool     |     Follow the log, printing new entries as they appear.        |
| -L  |--level    |     level        |             int/string  |  Minimum log level to dump, either an integer or a string. -1/TELEMETRY [the default],  0/DEFAULT, 1/D1/DBG1/DEBUG2, 2/D2/DBG2/DEBUG1,3/INFO,4/WARNING,5/ERROR,6/CRITICAL,7/FATAL.  Note  that only the mininum unique string is required. |                  
| -C | --code    | code            |   int vector       |    The event code, or vector of codes, to dump.  If not specified, all codes are dumped.  See logCodes.hpp for a complete list of codes.|



# EXIT STATUS


How `logdump` exits depends on the mode operation.  If it is printing all or a set number of log files, it will exit normally after printing the last entry.  If it is following a log, it will not exit until signaled (e.g. with `ctrl-c`).

# EXAMPLES


To dump all the log entries for the application `trippLitePDU0`:

```
$ logdump trippLitePDU0
```

To dump just the last 2 log files:

```
$ logdump -n 2 trippLitePDU0
```

To show only log entries with level WARNING or higher:

```
$ logdump -n2 -L W trippLitePDU0
```

To show only specific log entries, in this case change of PDU outlet state, run:

```
$ logdump -C 12001,12002 trippLitePDU0
```

To follow the log, showing log level WARNING or higher:

```
$ logdump -L W -f trippLitePDU0
```


# SEE ALSO 

