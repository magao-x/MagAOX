resurrector_indi
====

[TOC]

----

# NAME

resurrector_indi - manage processes in an INDI framework.

# SYNOPSIS 
```
[MAGAOX_ROLE=magaox-role ./resurrector_indi [-r magaox-role] [--role=magaox-role]
```
# DESCRIPTION

A program ensure all processes in an INDI framework,
comprising one INDI server process and multiple INDI driver processes
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
   .      /path/to/fifos/isXXX.hb                         |     Controller        |
   .        |                                              |                       |
   v        |                                              | bin/run -n drivername |
+------------------+                                       +------------+          |
|                  | <-- /path/to/fifos/drivername.ctrl -- |            |          |
|                  |                                       |    INDI    |          |
|    indiserver    | --- /path/to/fifos/drivername.in ---> |   DRIVER   |          |
|                  |                                       |            |          |
|                  | <-- /path/to/fifos/indidriver.out --- |            |          |
+------------------+                                       +------------+----------+
```

## Legend

* fork(2) - the fork system call for starting children processes
** indiserver - The INDI server, a child process of the resurrector
* resurrector_indi - The resurrector process
** Device Controller - The INDI driver process, a child process of the resurrector
    * INDI DRIVER - An indiDriver class instance
    * resurrectee - An resurrectee class instance
* /path/to/fifos/isXXX.hb - A named FIFO for transmitting INDI server heartbeats (Hexbeats) to the resurrector_indi process
* /path/to/config/magaox.conf - A file with host-wide global configuration parameters, specifically "indiserver_ctrl_fifo" here
* /path/to/fifos/drivername.hb - A named FIFO for transmitting INDI driver heartbeats (Hexbeats) to the resurrector_indi process
* /path/to/fifos/drivername.in - A named FIFO for transmitting INDI protocol messages to the INDI driver process
* /path/to/fifos/drivername.out - A named FIFO for transmitting INDI protocol messages to the INDI server process
* /path/to/fifos/indiserver.ctrl - A named FIFO telling the INDI server process to start listening for specific INDI driver processes
* /path/to/config/proclist_role.txt - A configuration file with list of Device Controller names and executable binaries

## resurrector_indi, fork(2)

The resurrector_indi process parses the names and executables of the INDI server and INDI drivers from the "/opt/MagAOX/config/proclist_role.txt" process list configuration file.
The resurrector_indi process forks a single INDI server, and multiple INDI driver, childen processes as parsed from the process list.
The resurrector_indi process then listens, at ~1Hz, on the "*.hb" named FIFOs for hexbeats (heartbeats) from its children processes; those children processes are also known as resurrectees.
If the resurrector_indi process receives a SIGUSR2 signal, it then re-parses the process list configuration file, kills any children processes that were previously started but are no longer in the process list, and starts any new children processes.

## proclist_role.txt

The process list is the same configuration file that was formerly used by the Python script "xctrl" to manually start the INDI server and drivers.
See "EXAMPLES" below for a sample process list configuration file.

The INDI server must be the only process name in that process list that starts with an "is" prefix, e.g. isRTC, isVM; the suffix after the "is" prefix is usually the uppercase version of role, but can be anything.

## Named FIFOs

The named FIFOs must be located in the directory pointed to by the default macros at compile time, which is "/opt/MagAOX/config/" currently.

### drivername.hb, isXXX.hb

The INDI driver heartbeat FIFO must be named "drivername.hb" for INDI drivers; the single heartbeat FIFO that starts with "is" is for the INDI server. 
The ".hb" extension means HexBeat (heartbeat).
The INDI driver heartbeat FIFO must be named "drivername.hb" with the ".hb" extension meaning HexBeat (heartbeat).

### indiserver.ctrl, indiserver, magaox.conf

The INDI server FIFO is named "indiserver.ctrl" and it must be configured both as the INDI server parameter "indiserver.f" in isXXX.conf, and as the global parameter "indiserver_ctrl_fifo" in magaox.conf.
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

### Process list configuration file

* Here is an example of a minimal process list configuration file;
"isVM" is the INDI server.
Lines beginning with a "#" are comments and/or ignored by any parsers.
Only lines with two whitespace-separated tokens are used.

    ####################################################################
    ### proclist_vm.txt
    ####################################################################
    # Processes for configuration to exercise magaoxMaths
    # Process-ID            Executable
    # ==========            ==========
    isVM                    xindiserver
    fpga0_um_sm             CGFSMHIfpga
    fpga0_um_fg             CGFSMUIfpga
    ####################################################################

### Global configuration file magaox.conf

    #loopPause = 1000000000
    #ignore_git = 1

    ### The presence of keyword [indiserver_ctrl_fifo] here directs the INDI
    ### drivers to alert the host's INDI server to their presence by writing
    ### [start /opt/.../{drvrname}\n] to this named INDI server control FIFO
    indiserver_ctrl_fifo = /opt/MagAOX/drivers/fifos/indiserver.ctrl
    ### N.B. this *must* match the [indiserver.f] value in is{THISHOST}.conf
    ###      and it might be better to move the local [indiserver] TOML here

# TESTING

TBD

# SEE ALSO

indiserver

http://www.clearskyinstitute.com/INDI/INDI.pdf.
