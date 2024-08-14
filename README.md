# The MagAO-X Software System

[![Status of Rocky test build](https://github.com/magao-x/MagAOX/actions/workflows/install-rocky.yml/badge.svg)](https://github.com/magao-x/MagAOX/actions/workflows/install-rocky.yml) [![Status of Ubuntu test build](https://github.com/magao-x/MagAOX/actions/workflows/install-ubuntu.yml/badge.svg)](https://github.com/magao-x/MagAOX/actions/workflows/install-ubuntu.yml) [![Status of container build](https://github.com/magao-x/MagAOX/actions/workflows/container.yml/badge.svg)](https://github.com/magao-x/MagAOX/actions/workflows/container.yml)

**Handbook:** https://magao-x.org/docs/handbook/ | **[C++ docs](https://magao-x.org/docs/api/):** [![Status of Doxygen build](https://github.com/magao-x/MagAOX/actions/workflows/build-doxygen.yml/badge.svg)](https://github.com/magao-x/MagAOX/actions/workflows/build-doxygen.yml)

This is the software which runs the MagAO-X ExAO system.

## 1 Dependencies

1. mxlib (https://github.com/jaredmales/mxlib).
   For a MagAO-X machine change the prefix to `/usr/local` in the mxlib install
2. libudev (for introspective device discovery).  Get from package manager.
3. zlib (compression for INDI). Get from package manager:
   - zlib-devel [centOS-7]
   - zlib1g-dev [ubuntu]
4. flatbuffers (https://google.github.io/flatbuffers/flatbuffers_guide_building.html)
   To build and install the flatc compiler and install the include files in /usr/local:
   ```
   $ cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
   $ make
   $ sudo make install
   ```

## 2 Software Configuration

For a standard build, nothing needs to be done here.  However, various defaults can be changed with macros.  To override at build time, the following macros can be redefined in `local/config.mk`.  For example to change the `MAGAOX_env_path` environment variable, one could add
```
CXXFLAGS += -DMAGAOX_env_path="/my/magaox/path"
```
to  `local/config.mk`.

#### 2.1 Environment Variables

These are defined in `libMagAOX/common/environment.hpp`.  They do not need to be set, but can be used to override relevant defaults.

- `MAGAOX_env_path`, the name of the environment variable holding the base path to the MagAOX system directories.  Default = "MagAOX_PATH"

- `MAGAOX_env_config`, the name of the environment variable holding the relative path to the config files.  Default = "MagAOX_CONFIG".

- `INDIS_NAMED_FIFO_DIR`, the name of the environment variable holding the absolute path to indiSERVER<->XindiDRIVER named FIFOs.  Optional, no default
  - As of 2022-05-23, this is only implemented in the **feature/named-pipes-consolidate-indiserver** branch

#### 2.2 Default Values

These are values which might be changed for different components. These are defined in `libMagAOX/common/defaults.hpp`.  They do not normally need to be changed, but these macros can be used to override relevant defaults.

`MAGAOX_default_writePause`

`MAGAOX_default_max_logSize`

`MAGAOX_default_loopPause`

#### 2.3 Config Values

These are values which should change for all components of MagAO-X.  These are defined in `libMagAOX/common/config.hpp`.  They do not normally need to be changed for a typical build, but these macros can be used to override relevant defaults.

`MAGAOX_logExt`

`MAGAOX_RT_SCHED_POLICY` is the real-time scheduling policy.  The result is `SCHED_FIFO`.

#### 2.4 Paths

These are defined in `libMagAOX/common/paths.hpp`.  They do not normally need to be changed, but these macros can be used to override the relevant defaults.

- `MAGAOX_path` The path to the MagAO-X system files This directory will have subdirectories including config, log, and sys directories.

- `MAGAOX_configRelPath` This is the subdirectory for configuration files.

- `MAGAOX_globalConfig` The filename for the global configuration file. Will be looked for in the config/ subdirectory.

- `MAGAOX_logRelPath` This is the subdirectory for logs.

- `MAGAOX_sysRelPath` This is the subdirectory for the system status files.

- `MAGAOX_secretsRelPath` This is the subdirectory for secrets.

- `MAGAOX_driverRelPath` This is the subdirectory for the INDI drivers.

- `MAGAOX_driverFIFORelPath` This is the subdirectory for the INDI driver FIFOs.

## 3 Building

Typing make in the MagAOX top-level directory will build the entire system.

Each app (subfolders of the `apps` directory) and each utilitiy (subfolders of the `utils` directory) have a Makefile.  Typing make in any of those folders should build just that component, along with any dependencies (usually including `libMagAOX`).

Note the difference between the `apps` and `utils` components, is that upon install MagAO-X Applications get setuid, whereas the utilities do not.

Some notes:

* libMagAOX (part of this repository) is a c++ header-only library.  It is compiled into a pre-compiled-header (PCH), managed by the build system.  Any changes to libMagAOX should trigger a re-compile of the PCH, and of any apps or utils depending on it.

* Allowing setuid for RT priority handling, access to ttys and FIFOs, etc., requires hardcoding LD_LIBRARY_PATH at build time.  This is done in the makefiles, and so should be transparent.

* Install requires root privileges.

### 3.1 Selective Build

It is sometimes necessary or desired to not build some parts of the system.  For instance on a machine on which not all dependencies are installed.  The make files allow the following to be turned off

#### CACAO
To prevent linking against the CACAO library, define `CACAO=false` in `local/common.mk`.  Note that this has not been tested in a long time and will probably faile completely.

#### EDT
To prevent linking against the EDT framegrabber library, define `EDT=false` in `local/common.mk`.  This will prevent building the `ocam2KCtrl` and `andorCtrl` apps.

#### PICam
To prevent linking against the Princeton Instruments PICam library, define `PICAM=false` in `local/common.mk`.  This will prevent building the `picamCtrl` app.

#### Pylon
To prevent linking against the Basler Pylon library, define `Pylon=false` in `local/common.mk`.  This will prevent building the `baslerCtrl` app.

## 4 Software Install

The software install process (from BIOS setup to building the MagAO-X software) is described in detail in [setup/README.md](setup/README.md). To the extent practicable, this is automated by the scripts in `setup/` that take over after CentOS is installed.

### Directory structure

| Directory                   | Description |
|-----------------------------|-------------|
| `/opt/MagAOX`               | MagAOX system directory |
| `/opt/MagAOX/bin`           | Contains all applications |
| `/opt/MagAOX/calib`         | [magao-x/calib](https://github.com/magao-x/calib) repo for instrument calibration files |
| `/opt/MagAOX/config`        | [magao-x/config](https://github.com/magao-x/config) repo for instrument config files |
| `/opt/MagAOX/drivers`       | Symlinks for INDI |
| `/opt/MagAOX/drivers/fifos` | FIFOs for INDI |
| `/opt/MagAOX/logs`          | Directory where logs are written by the applications |
| `/opt/MagAOX/rawimages`     | Directory where raw images are written by the applications |
| `/opt/MagAOX/secrets`       | Directory containing device passwords, etc. |
| `/opt/MagAOX/source`        | Directory containing clones of this repo, [cacao-org/cacao](https://github.com/cacao-org/cacao), [jaredmales/mxlib](https://github.com/jaredmales/mxlib)                    , [jaredmales/milkzmq](https://github.com/jaredmales/milkzmq) |
| `/opt/MagAOX/sys`           | Directory for application status files, e.g. PID lock-files |
| `/opt/MagAOX/vendor`        | Vendor software packages (anything we aren't frequent committers to ourselves) |

This directory structure is #define-ed in [libMagAOX/common/defaults.hpp](https://github.com/magao-x/MagAOX/blob/master/libMagAOX/common/paths.hpp).
System directories are created and permissions are set by [setup/steps/ensure_dirs_and_perms.sh](https://github.com/magao-x/MagAOX/blob/master/setup/steps/ensure_dirs_and_perms.sh). If permissions get out of whack, it is safe to re-run that script.


On install, symlinks are made for utility executables from `/usr/local/bin` to `/opt/MagAOX/bin`.

## 5 Documentation

The code is more-or-less carefully documented with doxygen

- [] Complete doxygen doc system
- [] Script construction of complete docs from various places (doxygen, and the markdown app docs)
- [] Create system to automatically make c.l. option table for each app
- [] Decide: do we use github pages, or host it on one of the snazzy magao-x domains?

## 6 To-Do

To-do items are listed in the above sections.  Also see the Todo page in the doxygen html.  Other misc. items below:

- [] split base INDI off into separate repo, which will be the minimum someone needs to have INDI utils for interacting with MagAO-X without installing the whole tree.
- [] create indiserver startup script which takes a list of drivers from a config file, creates symlinks to xindidriver as needed, and then starts indiserver itself.
- [] start issue tracking
- [] Brian T. Carcich, Ascending Node Technologies, starting ca. March, 2022:
  - [] make it possible to start and stop drivers and associated INDI protocol communications on-the-fly
    - eliminate xindidriver intermediate pass-through processes
  - See sub-directory 2022_ANT/ for more detail

## 7 Develop in a VM with [Vagrant](https://vagrantup.com)

To develop in the most "flight-like" configuration, a Vagrantfile is provided to set up a development VM. A quick-start guide is available [in the handbook](https://magao-x.org/docs/handbook/appendices/development_vm.html).
=======
