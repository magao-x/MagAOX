
# Control of a Tripp Lite PDU

The MagAOX App to control a Tripp Lite PDU

## Build

Since the MagAO-X build system is not done, this is a manual build.  To get full functionality, do this:

1. `make -B -f $MXMAKEFILE trippLitePDU`
2. When the above fails, copy the g++ command.
3. Add `-ludev' to the command (usually just before the last -o)
4. On CentOS 7: Dump LD_LIBRARY_CONFIG, and to each path in it add `-Wl,-rpath,' at the beginning.  Add this to the g++ command.
4. Now execute the modified g++ command.

After the build, do
`sudo chown root:root trippLitePDU'
'sudo chmod +s trippLitePDU'


## Running

For this to work, you have to have setup the MagAO-X config and logging system (environment variables and a directory structure).  Then you type:

`$> ./trippLitePDU`
 
