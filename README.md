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

For a standard build, nothing needs to be done here.  However, various defaults can be changed with macros.  To override at build time, the following macros can be redefined in `local/config.mk` [how?]

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

A rudimentary build system has been implemented.

To build an app, cd to the app's directory and type
```
make -f ../../Make/magAOXApp.mk t=<appname>
```
replacing `appname` with the name of the application.

To build a utility, cd to the utility's directory and type
```
make -f ../../Make/magAOXUtil.mk t=<utilname>
```
replacing `utilname` with the name of the utility.

The difference between the two Makefiles is in install, in that MagAO-X Applications get setuid, whereas the utilities do not.

Some notes:

* libMagAOX (part of this repository) is a c++ header-only library.  It is compiled into a pre-compiled-header (PCH), managed by the build system.  Any changes to libMagAOX should trigger a re-compile of the PCH, and of any apps depending on it.

* Allowing setuid for RT priority handling, access to ttys and FIFOs, etc., requires hardcoding LD_LIBRARY_PATH at build time.  This is done in the makefiles, and so should be transparent.

* Install requires root privileges.

ToDo:
- [] Use environment variables for path to various parts of build system, path to libMagAOX PCH, etc.
- [] Possibly use name of directory for target app name.

## 4 Software Install

Beginning from a CentOS 7 fresh install with network access, once you've logged in as a user in the `wheel` group:

1. Install development tools (including `git`)

   ```
   $ sudo yum groupinstall -y 'Development Tools'
   $ sudo yum install -y screen
   ```

2. Clone this repository into your home directory (not into `/opt/MagAOX`, yet)

   ```
   $ cd
   $ git clone https://github.com/magao-x/MagAOX.git
   ```

3. Download Intel MKL -- This has to be done interactively, since Intel enforces a registration requirement to download the MKL package.

    **If you're not yet registered with Intel:** Starting at https://software.intel.com/en-us/mkl, click "Free Download" and follow the prompts to create and verify an account.

    **If you've registered before:** It's well hidden, but https://registrationcenter.intel.com/en/products/ should take you right to the page with the download links. We want Intel Performance Libraries for Linux, specifically Intel Math Kernel Library.

    Copy the download link (e.g. right-click and "Copy Link Location") and switch to a terminal on a production machine to run `curl -OL <pasted-url>`. (I'd put the download link in these docs, but it's _export controlled_.)

4. Extract and install Intel MKL

   ```
   $ tar xvzf l_mkl_2018.3.222.tgz
   $ cd l_mkl_2018.3.222
   ```

   Install as root:
   ```
   $ sudo ./install.sh -s ~/MagAOX/setup/intel_mkl_silent_install.cfg
   ```

5. Switch to the MagAOX directory you cloned (in this example: `~/MagAOX/setup`) and set up users and groups.

   ```
   $ cd ~/MagAOX/setup
   $ ./setup_users_and_groups.sh
   ```

  **Note:** This creates the `xsup` user account with a default password (`extremeAO!`). Change this to the real password (`sudo passwd xsup`), or be a disappointment to @jaredmales.

   This changes the group memberships of the installing user (i.e. `$USER`, so most likely you). Before that takes effect, you will have to log out and back in. (Alternatively, you can run `newgrp magaox-dev` to start a new subshell where the new group is active, but this can get confusing.)

6. Run the provisioning script

   ```
   $ cd ~/MagAOX/setup
   $ screen  # optional: lets you detach from the build and come back later
   $ ./production_provision.sh
   ```

### Directory structure

The following are the default MagAOX system directories.

| Directory                   | Description                                                                               |
|-----------------------------|-------------------------------------------------------------------------------------------|
| `/opt/MagAOX`               | MagAOX system directory                                                                   |
| `/opt/MagAOX/bin`           | Contains all applications                                                                 |
| `/opt/MagAOX/drivers`       |                                                                                           |
| `/opt/MagAOX/drivers/fifos` |                                                                                           |
| `/opt/MagAOX/config`        | Contains the configuration files for the applications                                     |
| `/opt/MagAOX/logs`          | Directory where logs are written by the applications (chown :magaox, mode g+rws)          |
| `/opt/MagAOX/sys`           | Directory for application status files, e.g. PID lock-files                               |
| `/opt/MagAOX/secrets`       | Directory containing device passwords, etc.                                               |


 This directory structure is #define-ed in libMagAOX/common/defaults.hpp.  It is created by the script `setup/make_directories.sh` (except for `config`, which is made by cloning [magao-x/config](https://github.com/magao-x/config) into `/opt/MagAOX/config`).  Permissions are set in `setup/set_permissions.sh`. This structure is also specified in `local/config.mk`.  Changing this isn't yet very simple, but we intend for it to be possible to have parallel installations.

ToDo:
- [ ] Investigate using appropriate environment variables to allow overriding these.
- [ ] Investigate having the defines be passed in via make.  E.g. `-DMAGAOX_path=/opt/MagAOX-DEV` will override, maybe we should just inherit from `local/config.mk`

On install, symlinks are made for executables from `/usr/local/bin` to `/opt/MagAOX/bin`.

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
