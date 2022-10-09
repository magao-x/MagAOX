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
multipass mount \
  -g $(id -g):1000 \
  -g $(id -g):1001 \
  -g $(id -g):1002 \
  -u $()
  ~/devel/MagAOX/:/opt/MagAOX/source/MagAOX
```

```
multipass mount ~/devel/MagAOX/ primary:/opt/MagAOX/source/MagAOX
multipass set local.primary.disk=10GiB
multipass set local.primary.cpus=4
multipass start
multipass shell
ubuntu@primary:~$ cd /opt/MagAOX/source/MagAOX/setup
ubuntu@primary:/opt/MagAOX/source/MagAOX/setup$ bash -lx provision.sh
```


UID/GID translation isn't automatic (and maybe couldn't be made so) but adding `magaox` and `magaox-dev` as the first new groups means there are predictable UIDs and GIDs.

`/etc/group`:

```
magaox:x:1001:
magaox-dev:x:1002:
```

`/etc/passwd`:

```
ubuntu:x:1000:1000:Ubuntu:/home/ubuntu:/bin/bash
```

`multipass mount --help`

```
% multipass mount --help
Usage: multipass mount [options] <source> <target> [<target> ...]
Mount a local directory inside the instance. If the instance is
not currently running, the directory will be mounted
automatically on next boot.

Options:
  -h, --help                       Displays help on commandline options
  -v, --verbose                    Increase logging verbosity. Repeat the 'v'
                                   in the short option for more detail. Maximum
                                   verbosity is obtained with 4 (or more) v's,
                                   i.e. -vvvv.
  -g, --gid-map <host>:<instance>  A mapping of group IDs for use in the mount.
                                   File and folder ownership will be mapped from
                                   <host> to <instance> inside the instance. Can
                                   be used multiple times.
  -u, --uid-map <host>:<instance>  A mapping of user IDs for use in the mount.
                                   File and folder ownership will be mapped from
                                   <host> to <instance> inside the instance. Can
                                   be used multiple times.

Arguments:
  source                           Path of the local directory to mount
  target                           Target mount points, in <name>[:<path>]
                                   format, where <name> is an instance name, and
                                   optional <path> is the mount point. If
                                   omitted, the mount point will be the same as
                                   the source's absolute path
```

## macOS

Example ids:

```
% id
uid=501(josephlong) gid=20(staff) groups=20(staff),101(access_bpf),12(everyone),61(localaccounts),79(_appserverusr),80(admin),81(_appserveradm),98(_lpadmin),701(com.apple.sharepoint.group.1),33(_appstore),100(_lpoperator),204(_developer),250(_analyticsusers),395(com.apple.access_ftp),398(com.apple.access_screensharing),399(com.apple.access_ssh),400(com.apple.access_remote_ae)
18:14:07 kestrel:~/devel/MagAOX josephlong
```

First user account created is uid=501, admin accounts are all `staff`, so it would make sense to map like this:

