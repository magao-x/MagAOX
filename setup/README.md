The scripts in this folder automate the setup of the MagAO-X software system on new hardware (to the extent it is possible to automate). The code in this repo is tightly coupled to our choice of operating system (Linux/CentOS 7), as well as the existence of folders, users, groups, and permissions that make the whole thing work.

The main script that calls the others is `provision.sh` for both the physical machines that will be used with the instrument and the VMs used in software development. For usage documentation, consult the [computer setup appendix](https://magao-x.org/docs/handbook/appendices/computer_setup/computer_setup.html) of the [MagAO-X Handbook](https://magao-x.org/docs/handbook/).

The `provision.sh` script does different things depending on the `MAGAOX_ROLE` environment variable of the machine it's run on. (It makes `/etc/profile.d/magaox_role.sh` to set that the first time you run it.) This can be:

  * `AOC` - Adaptive optics Operator Computer
  * `RTC` - Real Time control Computer
  * `ICC` - Instrument Control Computer
  * `TIC` - Testbed Instrument Computer
  * `TOC` - Testbed Operator Computer
  * `workstation` - Any other MagAO-X workstation
  * `ci` - Continuous Integration environment (CircleCI, invoked as described in [`.circleci/config.yml`](../.circleci/config.yml) in the repo root)
  * `vm` - Virtual machine (Vagrant) environment (see [the handbook](https://magao-x.org/docs/handbook/appendices/development_vm.html))

Each script has comments throughout. They should all be *idempotent*. (In other words: running `provision.sh` twice should only do a fresh install the first time, and should not clobber anything or emit an error if run again.)


# multipass notes

```
multipass launch -n primary 22.04
multipass set local.privileged-mounts=Yes  # windows only
multipass mount ~/devel/MagAOX/ primary:/opt/MagAOX/source/MagAOX
multipass stop
multipass set local.primary.disk=20GiB
multipass set local.primary.cpus=4
multipass shell
multipass shell
ubuntu@primary:~$ cd /opt/MagAOX/source/MagAOX/setup
ubuntu@primary:/opt/MagAOX/source/MagAOX/setup$ bash -lx provision.sh
ubuntu@primary$ exit
multipass exec primary -- bash -c "echo `cat ~/.ssh/id_ed25519.pub` >> ~/.ssh/authorized_keys"
ssh -Y ubuntu@$(multipass exec primary -- hostname -I | awk '{ print $1 }' ) xeyes
```
