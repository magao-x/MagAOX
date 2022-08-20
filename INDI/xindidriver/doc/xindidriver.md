xindidriver
==========

[TOC]

------------------------------------------------------------------------

# NAME

indidriver − create a passthrough from STDIN/STDOUT to a pair of FIFOs belonging to a device controller implementing the INDI protocol.

# SYNOPSIS 
```
symlinked-name
```
where `symlinked-name` is the a file with the name of the driver, symbolically linked to point at the `xindidriver` executable.

For testing, one can also use:
```
xindidriver name
```
where name is the driver name.

# DESCRIPTION

A program to act as the INDI driver for indiserver, but pass the STDIN and STDOUT to/from FIFOs exposed by a device controller with INDI driver functionality. This allows the device controller to integrate INDI processing, without using STDIN and STDOUT and without being subject to process control by `indiserver`.

This diagram shows how it works:
```
                                                                                     --------------------------
                                                                                     |         Device         |
                                                                                     |      Controller        |
                                                                                     |                        |
--------------                ---------------                                        --------------           |
|            | -- STDIN  -->  |             |  -- /path/to/fifos/drivername.in  -->  |    INDI    |           |
| indiserver |                | xindidriver |                                        |   DRIVER   |           |
|            | <-- STDOUT --  |             |  <-- /path/to/fifos/drivername.out --  |            |           |
--------------                ---------------                                        --------------------------
```

The name of the driver is determined from the basename of `argv[0]`, meaning the name used to invoke `xindidriver`.   The expectation is that `xindidriver` will be symlinked from a file with the driver name.  The diver name can also be passed as the sole argument to `xindidriver` for testing.

The FIFOs must be located at the path pointed to by the `XINDID_FIFODIR` macro at compile time.  The FIFOs must be named "drivername.in" and "drivername.out", which take the place of STDIN and STDOUT in the normal `indiserver` framework.

If at startup the FIFOs do not exist, the program will patiently wait for them to come into existence.  If some other error occurs, say due to permissions, the program will exit.  In this case `indiserver` should restart it automatically.

An exclusive lock is placed on the `.in` FIFO.  If this fails, it means that another instance of `xindidriver` is already running.  The instance which could not get a lock will exit.  This is necessary to prevent lost data on the FIFOs, etc.

The presence of a process listening-on or writing-to the FIFOs has no effect.

A third fifo, `drivername.ctrl` is used for signaling `xindidriver` that the controller has restarted.  Anything written to this FIFO will cause `xindidriver` to exit, and it will then be restarted by `indiserver`.  This is done to keep all snoops, etc, up to date and fresh.


# OPTIONS 

There are no options.

# EXAMPLES



# TESTING

To test xindidriver, one can perform the following steps:

1) Create a set of fifos in `SINDID_FIFODIR` (set at compile time)
```
$ mkfifo test.in   # This is the STDIN of the device controller
$ mkfifo test.out  # This is the STDOUT of the device controller
```
Make sure to set permissions appropriately.  Make another FIFO anywhere you want:
```
$ mkfifo stdinpipe # This will be the STDIN of xindidriver
```

2) Now start xindidriver taking STDIN from the `stdinpipe` created above
```
$ ./xindidriver test < stdinpipe &
$ sleep infinity > stdinpipe & # Keeps the pipe open as if STDIN is captured by, say, indiserver
```

3) Send test inputs

In a separate terminal, read from test.in
```
$ cat test.in
```

In another terminal, write to stdinpipe
```
$ echo "test input" > stdinpipe
```

You should see output from `cat test.in`

In a separate terminal, write to test.out
```
$ echo "test output" > test.out
```

You should see "test output" printed from xindidriver in the original terminal.


# SEE ALSO

indiserver

http://www.clearskyinstitute.com/INDI/INDI.pdf.
