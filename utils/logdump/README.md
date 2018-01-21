# The logdump Utility

A simple utility for dumping MagAO-X binary logs to stdout as ASCII text in a standard format

## Build

This depends on mxlib and can be compiled as a standard mxlib application:

`make -B -f $MXMAKEFILE logdump`

## Usage

Invoke in the logs directory with

`$> logdump <prefix> [n]`

Where <prefix> is replaced by the application name.

n is an optional integer > 0 setting the number of past log files to dump.  0 dumps them all in order.

 
