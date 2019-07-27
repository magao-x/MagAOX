[![Codacy Badge](https://api.codacy.com/project/badge/Grade/dc3d038ce7074b7bab093699d0806759)](https://www.codacy.com/app/jaredmales/MagAOX?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=magao-x/MagAOX&amp;utm_campaign=Badge_Grade)

# The MagAOX Software System

This is the software which runs the MagAOX ExAO system.

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
To prevent linking against the Basler Pylon library, define `Pylong=false` in `local/common.mk`.  This will prevent building the `baslerCtrl` app.

## 4 Software Install

The software install process (from BIOS setup to building the MagAO-X software) is described in detail in [setup/README.md](setup/README.md). To the extent practicable, this is automated by the scripts in `setup/` that take over after CentOS is installed.

### Directory structure

The following are the default MagAOX system directories.

| Directory                   | Description                                                                               |
|-----------------------------|-------------------------------------------------------------------------------------------|
| `/opt/MagAOX`               | MagAOX system directory                                                                   |
| `/opt/MagAOX/bin`           | Contains all applications                                                                 |
| `/opt/MagAOX/drivers`       | Contains symlinks to `xindidriver` for app drivers                                        |
| `/opt/MagAOX/drivers/fifos` | Contains the FIFOs for INDI communications                                                |
| `/opt/MagAOX/config`        | Contains the configuration files for the applications                                     |
| `/opt/MagAOX/logs`          | Directory where logs are written by the applications (chown :magaox, mode g+rws)          |
| `/opt/MagAOX/sys`           | Directory for application status files, e.g. PID lock-files                               |
| `/opt/MagAOX/secrets`       | Directory containing device passwords, etc.                                               |


 This directory structure is #define-ed in libMagAOX/common/defaults.hpp.  It is created by the script `setup/make_directories.sh` (except for `config`, which is made by cloning [magao-x/config](https://github.com/magao-x/config) into `/opt/MagAOX/config`).  Permissions are set in `setup/set_permissions.sh`. This structure is also specified in `local/config.mk`.  Changing this isn't yet very simple, but we intend for it to be possible to have parallel installations.

ToDo:
- [ ] Investigate using appropriate environment variables to allow overriding these.

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

## 7 Develop in a VM with [Vagrant](https://vagrantup.com)

The MagAOX code is intimately tied to Linux OS internals, and targets CentOS 7 for the realtime control computers. To develop in the most "flight-like" configuration, a Vagrantfile is provided to set up a development VM.

### Prerequisites:

  * VirtualBox
  * Vagrant
  * NFS

### Usage:

After cloning the MagAOX repository, `cd` into it and run `vagrant up`. Provisioning uses the `setup/vagrant_provision.sh` script, which in turn calls other scripts in `setup/` and sets permissions. Provisioning is slow (~ 10s of minutes), but only costly the first time you start the VM. Vagrant will download a virtual machine image for CentOS 7 and then set up all the dependencies required. NFS is used to sync the contents of your repository clone to the VM.

To connect to the VM, use `vagrant ssh`. The VM has a view of your copy of this repository under `/vagrant`. For example, no matter where you cloned this repository on your own (host) machine, the virtual machine will see this file at `/vagrant/README.md`. (For consistency with production, we symlink `/opt/MagAOX/source/MagAOX` to `/vagrant`.) Edits to the MagAO-X software source on your computer will be instantly reflected on the VM side, ready for you to `make` or `make install`.

The `vagrant` user you log in as will be a member of the `magaox` and `magaox-dev` groups, and should have all the necessary permissions to run the system (or, at least, the parts you can run in a VM).
