resurrector_indi
====

[TOC]

----

# NAME

resurrector_indi - manage processes in an INDI framework.

# SYNOPSIS 
```
[export MAGAOX_ROLE=magaox-role]
[MAGAOX_ROLE=magaox-role] ./resurrector_indi [-r magaox-role] [--role=magaox-role] [-nor|--no-output-redirect] [-l|--logging] [-v|--verbose] [-h|--help]

```
- I.e. four ways to specify the role:  two using environment variable; two using command-line arguments.

# DESCRIPTION

The resurrector program ensures all processes in a MagAO-X INDI framework,
comprising one INDI server process and multiple INDI driver processes,
will start up and continue both to communicate and to run even if some
of them fail by either exiting, crashing, or locking up.

This diagram shows how it works, although with a single INDI driver:
```
     SIGUSR2
        |
        v
+------------------+                                       
|                  |                                         /path/to/config/magaox.conf
|                  | <-- /path/to/config/proclist_role.txt     |
|                  |                                           v
| resurrector_indi |                                       +-------------+---------+
|                  | <-- /path/to/fifos/drivername.hb ---- |             |         |
|                  |                                       | resurrectee |         |
|                  | ...fork(2).....                       |             |         |
+------------------+                \                      +-------------+         |
   .        ^                        \                     |                       |
 fork(2)    |                         \..................> |      Device           |
   .      /path/to/fifos/isXXX.hb                          |     Controller        |
   .        |                                              |                       |
   v        |                                              | bin/app -n drivername |
+----+-------------+                                       +------------+          |
|    | resurrectee | <-- /path/to/fifos/drivername.out --- |            |          |
|    +-------------+                                       |    INDI    |          |
|                  | --- /path/to/fifos/drivername.in ---> |   DRIVER   |          |
|    indiserver    |                                       |            |          |
|                  | <-- /path/to/fifos/indidriver.ctrl -- |            |          |
+------------------+                                       +------------+----------+
```

## Legend

* fork(2) - the fork system call for starting children processes
* indiserver - The INDI server, a child process of the resurrector
* resurrector_indi - The resurrector process
* INDI driver - A child process of the resurrector
    * INDI DRIVER or indiDriver- An indiDriver class instance
    * resurrectee - A resurrectee class instance
    * Device Controller - The MagAO-X business logic
* /path/to/fifos/isXXX.hb - A named FIFO for transmitting INDI server heartbeats (Hexbeats) to the resurrector_indi process
* /path/to/config/magaox.conf - A file with host-wide global configuration parameters, specifically "indiserver_ctrl_fifo" here
* /path/to/fifos/drivername.hb - A named FIFO for transmitting INDI driver heartbeats (Hexbeats) to the resurrector_indi process
* /path/to/fifos/drivername.in - A named FIFO for transmitting INDI protocol messages to the INDI driver process
* /path/to/fifos/drivername.out - A named FIFO for transmitting INDI protocol messages to the INDI server process
* /path/to/fifos/indiserver.ctrl - A named FIFO telling the INDI server process to start listening for specific INDI driver processes
* /path/to/config/proclist_role.txt - A configuration file with list of Device Controller names and executable binaries

## resurrector_indi, fork(2)

The primary purpose of the resurrector_indi process is to keep all of the pieces, INDI server and INDI drivers, of the MagAO-X system running:

* Starts and stops all processes; 
* Monitors Hexbeats (heartbeats) from all processes;
* Detects if any have crashed or have otherwise failed;
* Restarts any that have failed

The resurrector_indi process is designed to replace the xctrl Python script for controlling processes in the MagAO-X system.

The resurrector_indi process parses the names and executables of the INDI server and INDI drivers from the "/opt/MagAOX/config/proclist_role.txt" process list configuration file.
The resurrector_indi process forks a single INDI server, and multiple INDI driver, childen processes as parsed from the process list.
The resurrector_indi process then listens, at ~1Hz, on the "*.hb" named FIFOs for Hexbeats (cf. below) from its children processes; those children processes are also known as resurrectees.
If the resurrector_indi process receives a SIGUSR2 signal, it then re-parses the process list configuration file, kills any children processes that were previously started but are no longer in the configuration process list, starts any new children processes, and restarts any old children processes **that are not currently running**.  N.B. this means that this signal to the resurrector_indi process will **_not_** kill a locked up process; that will happen when the current time passes the last Hexbeat of the locked process.

A Hexbeat is a representation of a time at some point in the future.
The resurrector continually checks the current time against the most recent Hexbeat time received from the resurrectee class instance (Hexbeater) of an INDI server or driver process.
Normally the Hexbeat from each process will be updated at about 1Hz, so the latest Hexbeat (future) time obtained by the resurrector will always exceed the current time. 
However, if the process containing the Hexbeater fails, then the latest Hexbeat will stop changing, and eventually the current time will exceed the Hexbeat time;
at that point the resurrector will have detected a failure of the Hexbeater (resurrected), will stop that resurrectee's procees, and start a new process with the same command-line parameters.

## proclist_role.txt

The process list is the same configuration file that was formerly used by the Python script "xctrl" to manually start the INDI server and drivers.
See "EXAMPLES" below for a sample process list configuration file.

The MagAO-X process list file is where the generic INDI framework becomes a MagAO-X system.  INDI server itself is a general-purpose application that facilitates INDI protocol messaging among INDI clients (drivers), but it cares not a whit about the application-specific content of said messaging.

The INDI server must be the only process name in that process list that starts with an "is" prefix, e.g. isRTC, isVM; the suffix after the "is" prefix is usually the uppercase version of role, but can be anything.

## INDI server and drivers

The indiserver app is the hub of all inter- and intra-host INDI protocol communication over the named FIFOs "drivername.in" and "drivername.out" connected to the indiDriver class instance in each INDI driver.
The named FIFO "indiserver.ctrl" is also monitored by the indiserver app to detect when a new INDI driver starts up.
The INDI server actually comprises two running processes:  xindiserver, which is a C++ application that reads the isXXX.conf file and forks the INDI server itself; indiserver, a C application which know

An INDI driver is a child process forked by the resurrector, running an app comprising (at least) three pieces:  Device Controller; indiDriver  class instance; resurrectee class instance.

### Device Controller

The Device Controller executes the MagAO-X business logic of the INDI driver, such as communicating with a Fast Steerable Mirror or a device that measures the wavefront.

### indiDRIVER class instance

The indiDriver class instance handles the inter-process INDI protocol communication between the Device Controller and other Device Controllers.
The indiDriver class instance manages the INDI driver (app) communication with the INDI server and other INDI drivers.
Note the similarity of the phrases "INDI driver" and "indiDriver" here:
the former refers to the complete application;
the latter refers to a class instance that handles INDI protocol messages, and like the INDI server it is agnostic toward the content of those messages.

### resurrectee class instance

The resurrectee class instance handles one-way communication with the resurrector, sending a Hexbeat (heartbeat; future time) at about 1Hz.
The resurrectee class is also refer to as a Hexbeater.

## Named FIFOs

The named FIFOs must be located in the directory pointed to by the default macros at compile time, which is "/opt/MagAOX/drivers/fifos//" currently.

### drivername.hb, isXXX.hb

The INDI driver heartbeat FIFO must be named "drivername.hb" for INDI drivers; the single heartbeat FIFO that starts with "is" is for the INDI server. 
The ".hb" extension means HexBeat (heartbeat).
These named FIFOs are used by the INDI processes to send Hexbeats to the resurrector.
A Hexbeat is a representation of a time at some point in the future.

### indiserver.ctrl, magaox.conf

The INDI server control FIFO is named "indiserver.ctrl" and it must be configured both as the INDI server parameter "indiserver.f" in isXXX.conf, and as the global parameter "indiserver_ctrl_fifo" in magaox.conf.
This is a named FIFO used by any INDI driver process for transmitting a "start /path/to/fifos/drivername<LF>" message to the INDI server process, which message notifies the INDI server that a new driver is joining the INDI protocol network.

The file "magaox.conf" is a global configuration TOML file that is read by all INDI processes, both server and drivers.  See "EXAMPLES" below for a sample magaox.conf file.

### drivername.in; drivername.out

The INDI driver FIFOs must be named "drivername.in" and "drivername.out" and are used to communicate INDI protocol messages between  the INDI server process and the INDI drivers processes.

# OPTIONS 

* Role
    * the host-wide role e.g. AOC, ICC, RTC, vm
    * Process list is parsed from proclist_role.txt
    * Default source is environment variable (envvar) MAGAOX_ROLE
        * Typically defined in /etc/profile.d/magaox_role.sh
    * Command-line syntax can be used overrides default
      * -r role
      * --role=role
* Output redirect
    * Default action redirect each INDI driver device's STDERR and STDOUT outputs to a file
        * Target file is /opt/MagAOX/sys/<devicename>/outputs
        * N.B. "/opt/MagAOX" prefix can be overridden with envvar MagAOX_PATH
    * To not redirect devices' outputs:
        * --no-output-redirect
        * -nor
* Some logging
    * -l
    * --logging
* Verbose logging
    * -v
    * --verbose
* Help
    * Prints Usage to STDOUT
    * Command-line syntax
        * -h
        * --help
        * N.B. presence of -h or --help will prevent resurrector_indi from running

# EXAMPLES

## Command-line execution

    /opt/MagAOX/bin/resurrector_indi
* Run with defaults
    * Role obtained from MAGAOX_ROLE envvar
    * Process list read from /opt/MagAOX/config/proclist_role.txt

    /opt/MagAOX/bin/resurrector_indi -r vm
or
    /opt/MagAOX/bin/resurrector_indi --role=vm
* Process list will be read from /opt/MagAOX/config/proclist_vm.txt

## Configuration files (/opt/MagAOX/config/)

### Process list configuration file, /ope/MagAOX/config/proclist_vm.txt

* Here is an example of a minimal process list configuration file;
"isVM" is the INDI server.
* Lines beginning with a "#" are comments and/or ignored by any parsers.
Only lines with two whitespace-separated tokens are used.

```
########################################################################
### proclist_vm.txt
########################################################################
# Processes for configuration to exercise magaoxMaths
# Process-ID            Executable
# ==========            ==========
isVM                    xindiserver
fpga0_um_sm             CGFSMHIfpga
fpga0_um_fg             CGFSMUIfpga
########################################################################
```

### Global configuration file /opt/MagAOX/config/magaox.conf

```
########################################################################
### The presence of keyword [indiserver_ctrl_fifo] here directs the INDI
### drivers to alert the host's INDI server to their presence by writing
### [start /opt/.../{drvrname}\n] to this named INDI server control FIFO

indiserver_ctrl_fifo = /opt/MagAOX/drivers/fifos/indiserver.ctrl

### N.B. this *must* match the [indiserver.f] value in is{THISHOST}.conf
###      and it might be better to move the local [indiserver] TOML here
```

## Techniques

### Triggering resurrector_indi to restart a driver or server process

A Hexbeat is a nine-digit hexadecimal representation of the time*, terminated by a newline**.
Since each process's Hexbeat FIFO is available in the file system, it is possible to send an expired Hexbeat to resurrector_indi to trigger a synthetic Hexbeat expiration:

    % echo 000000000 >> /opt/MagAOX/drivers/fifos/drivername.hb

After receive such a Hexbeat telling it that the corresponding driver/server expired several decades ago, resurrector_indi should stop and restart that process.
<!-- the next line must end in two spaces -->
\* seconds since the Unix(tm) epoch of 1970-01-01T00:00:00  
\*\* ASCII 10 = 0x0A

## Ca. 2023-02-21 Temporary documentation for output-redirection prototype

### Build

    make EXTRACPPFLAGS=-DTEST_MAIN=main redirect_prototype

### Run/test

    while true ; do tail -fc+1 /opt/MagAOX/sys/devicename/outputs || sleep 5 ; done &

    ./redirect_prototype devicename
    <type some data, then hit return>
    <type some data again, then hit return again>
    <type some data again, then hit return again>
    ...
    ^D (Control-D, i.e. End-Of-File)

# TESTING

TBD

# SEE ALSO

indiserver

http://www.clearskyinstitute.com/INDI/INDI.pdf.
