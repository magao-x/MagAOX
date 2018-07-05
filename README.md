[![Codacy Badge](https://api.codacy.com/project/badge/Grade/dc3d038ce7074b7bab093699d0806759)](https://www.codacy.com/app/jaredmales/MagAOX?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=magao-x/MagAOX&amp;utm_campaign=Badge_Grade)

# The MagAOX Software System

This is the software which runs the MagAOX ExAO system.  

## Dependencies

### Current:
1. mxlib (https://github.com/jaredmales/mxlib)
2. libudev (for introspective device discovery)
3. zlib1g-dev [ubuntu]

### Future
1. libhdf5 (though not for anything currently implemented, but we will)

## Building

A rudimentary build system has been implemented.

To build an app, cd the app's directory and type
```
make -f ../../Make/magAOXApp.mk t=<appname>
```
replacing `appname` with the name of the application.

Some notes:

* libMagAOX (part of this repository) is a c++ header-only library.  It is compiled into a pre-compiled-header (PCH), managed by the build system.  Any changes to libMagAOX should trigger a re-compile of the PCH, and of any apps depending on it.

* Allowing setuid for RT priority handling, access to ttys and FIFOs, etc., requires hardcoding LD_LIBRARY_PATH at build time.  This is done in the makefiles, and so should be transparent.

* Install requires root privileges.

ToDo:
- [] Implement su and asking for password as part of install process
- [] Use environment variables for path to various parts of build system, path to libMagAOX PCH, etc.
- [] Possibly use name of directory for target app name.

## System Setup

The following are the typical MagAOX system directories, which means they are #define-ed in libMagAOX/common/config.hpp:

```
/opt/MagAOX               [MagAOX system directory]
/opt/MagAOX/bin           [Contains all applications]
/opt/MagAOX/drivers
/opt/MagAOX/drivers/fifos
/opt/MagAOX/config        [Contains the configuration files for the applications]
/opt/MagAOX/logs          [Directory where logs are written by the applications] (chown :xlog, chmod g+w, chmod g+s)
/opt/MagAOX/sys           [Directory for application status files, e.g. PID lock-files]
/opt/MagAOX/secrets       [Directory containing device passwords, etc.]
```

ToDo:
- [] Script installation of these directories, with appropriate permission setting, etc.
- [] Establish appropriate environment variables to allow overriding these.

## Documentation

The code is more-or-less carefully documented with doxygen, though the doxyfile has not been created, etc.

- [] Init doxygen doc system
- [] Decide: do we use github pages, or host it on one of the snazzy magao-x domains?
