The scripts in this folder automate the setup of the MagAO-X software system on new hardware (to the extent it is possible to automate). The code in this repo is tightly coupled to our choice of operating system (Linux/CentOS 7), as well as the existence of folders, users, groups, and permissions that make the whole thing work.

The main scripts that call the others are `production_provision.sh` for the physical machines that will be used with the instrument, and `vagrant_provision.sh` for VMs used in software development. For usage documentation, consult the computer setup appendix of the [MagAO-X Handbook](https://github.com/magao-x/handbook).

Each script has comments throughout. They should all be *idempotent*. (In other words: running `./install_MagAOX.sh` twice should only do a fresh install the first time, and should not clobber anything or emit an error if run again.)

Some scripts accept a `--dev` or `--prod` switch. The `production_provision.sh` script supplies `--prod` where necessary, and the `vagrant_provision.sh` script supplies `--dev` where necessary, so you shouldn't have to worry about it.