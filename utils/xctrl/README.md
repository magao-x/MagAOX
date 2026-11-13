xctrl
=====

`xctrl` runs the MagAO-X apps of a MAGAOX\_ROLE as systemd services. Systemd starts, watches and restarts each app;
`xctrl` turns the proclist and the apps' configuration into systemd units and drives `systemctl`.

# Terms

| Term          | Meaning                                                                                       |
|---------------|-----------------------------------------------------------------------------------------------|
| `xctrl`       | this program: the one command for registering, starting, stopping and checking the apps        |
| sentinel      | the systemd supervision of a role's apps: the units `xctrl register` generates, their watchdog and restart policy, and the proclist record in `sys/sentinel_proclist` |
| watchdog      | systemd's check that an app is alive: the app sends `WATCHDOG=1` at least every `watchdogSec`, or systemd restarts it |
| `[systemd]`   | the configuration section, in `magaox.conf` or an app's `.conf`, holding the sentinel settings |

Each app reports to systemd with the `sd_notify` protocol (see `libMagAOX/sys/sdNotify.hpp` and
`magaox/indi/device.py`): `READY=1` once started, `WATCHDOG=1` from its main loop, and `STOPPING=1` on the way out.

`xctrl register` parses the role's proclist, registers the apps in it with systemd, and deregisters apps that are
registered but no longer in the proclist.

# Generated files

For `MAGAOX_ROLE=ICC`, `sudo xctrl register` writes:

```
/etc/systemd/system/
    magaox-icc-app@.service               template shared by all apps of the role
    magaox-icc.target                     wants the apps enabled for automatic startup
    magaox-icc.target.wants/              one link per enabled app (systemctl enable)
    magaox-icc-app@camsci1.service.d/
        override.conf                     camsci1's command and settings
/etc/magaox-icc/
    apps/camsci1.conf                     registry: every app registered, enabled or not
    environment                           optional, not generated: extra environment for all apps
/opt/MagAOX/sys/sentinel_proclist/proclist_ICC.txt.active
                                          the proclist last registered, kept up to date by add,
                                          enable, disable and delete
```

Each app runs as `/opt/MagAOX/bin/<executable> -n <name>`, as user `xsup`, with `MAGAOX_ROLE` set, and
`PYTHONUNBUFFERED=1` so that Python apps' output is not held back. Its output goes to the journal
(`xctrl peek NAME`); its binary logs are unchanged (`logdump NAME`).

Unit names use the role in lower case. App names may contain `-`; units use them as is (`%i`).

# Proclist

`/opt/MagAOX/config/proclist_<role>.txt` has one app per line, `NAME EXECUTABLE`.

| Line                     | Meaning                                                                          |
|--------------------------|----------------------------------------------------------------------------------|
| `camsci1  picamCtrl`     | an app, registered and started automatically, with a watchdog                    |
| `-camsci1  picamCtrl`    | an app, registered and started automatically, without a watchdog                 |
| `#!camsci1  picamCtrl`   | a disabled app: registered if `camsci1.conf` exists, but started only by hand     |
| `#!-camsci1  picamCtrl`  | a disabled app without a watchdog                                                |
| `#camsci1  picamCtrl`    | a comment, like any line starting with `#` but not `#!`: not an app, not registered |
| `#-camsci1  picamCtrl`   | a comment too                                                                    |

`#!` may be followed by spaces: `#! camsci1 picamCtrl` is the same disabled app. A `#!` line that is not
`NAME EXECUTABLE`, or whose app has no `.conf`, is reported by `register` and otherwise ignored. Comments are never
touched by `enable`, `disable`, `add` or `delete`.

| Prefix      | Meaning                                                                         |
|-------------|---------------------------------------------------------------------------------|
| none        | the app notifies systemd: `Type=notify`, with a watchdog                        |
| `-`         | the app does not notify systemd: `Type=simple`, no watchdog, still restarted if it exits |

On the command line, `--` lets a name start with `-`: `sudo xctrl add -- -camsci-1 someExec`.

The app run by `xindiserver` is the role's INDI server. Apps are ordered after it, so they stop before it and their
last messages get through; it is ordered after the `sshDigger` tunnels.

# Settings

The template holds the default settings. `magaox.conf` changes them for every app of the role, and an app's `.conf`
changes them for that app, both under `[systemd]`, with the same keys:

| Key                     | systemd directive        | Default  | Meaning                                                  |
|-------------------------|--------------------------|----------|----------------------------------------------------------|
| `timeoutStartSec`       | `TimeoutStartSec=`       | 120      | time allowed from start to `READY=1`                     |
| `watchdogSec`           | `WatchdogSec=`           | 10       | time allowed between `WATCHDOG=1`s, once ready; 0 disables the watchdog |
| `restart`               | `Restart=`               | always   | when systemd restarts the app, e.g. `always`, `on-failure`, `no` |
| `restartSec`            | `RestartSec=`            | 5        | wait before a restart                                    |
| `startLimitBurst`       | `StartLimitBurst=`       | 10       | starts allowed within `startLimitIntervalSec`            |
| `startLimitIntervalSec` | `StartLimitIntervalSec=` | 600      |                                                          |
| `timeoutStopSec`        | `TimeoutStopSec=`        | 30       | time allowed to stop before SIGKILL                      |
| `killSignal`            | `KillSignal=`            | SIGTERM  | signal that asks the app to stop                         |

Times are seconds, or `infinity`. The apps ignore the whole `[systemd]` section; only `xctrl` reads it. `register` reports an
unknown or misspelled key, or a bad value, as an error on the console and does not (re)register that app; `start`,
`startup` and `restart` warn and start it with its last good settings. The
`[resurrectee]` section is no longer used: `register` reports it as an error, and its `timeout = RUN,STARTUP` becomes
`watchdogSec = RUN` and `timeoutStartSec = STARTUP`.

Example, in `camsci1.conf`:

```
[systemd]
timeoutStartSec = 180
watchdogSec = 20
restart = on-failure
killSignal = SIGINT
```

`register` checks every value and every executable; an app with a problem is reported and not (re)registered, and
keeps whatever registration it had.

# Commands

Any user can run `xctrl`. It is installed setuid root, and decides from who ran it what it does for them:

- **Looking** (`status`, `peek`, `inspect`): anyone.
- **Starting and stopping** registered apps: root, `xsup` and members of group `magaox`.
- **Changing which apps are registered or enabled** (`register`, `add`, `enable`, `disable`, `delete`, `boot`): a
  privileged user, meaning root (e.g. through `sudo`) or a user who may write the role's proclist. The config
  checkout is normally group `magaox-dev` writable, so its members qualify and `xsup` does not. Anyone else gets,
  for example:

  ```
  error: cannot enable camsci1 as user xsup: enable requires permissions or sudo
  ```

  The proclist and the apps' `.conf` files are protected only by their own permissions, so they should not be
  writable by regular users. In the table below, "privileged" means root or a user who can write the proclist.

| Who                         | Command                | What it does                                                             |
|-----------------------------|------------------------|--------------------------------------------------------------------------|
| anyone                      | `status [NAME...]`     | Every registered app, one line each, `name: state (pid: N)`, colored and sorted by state: running green, manually started (running with autostart off) blue, disabled (`#!`, not running) yellow, not started cyan, starting/stopping magenta, dead red. `-v` adds a table with autostart, restarts, uptime, seconds since the last watchdog notification, watchdog timeout, CPU % now, memory and GPU memory. |
| anyone                      | `peek [NAME...]`       | State and the last `logdump` lines of the apps (`-n LINES`, default 10), with the console output of an app that is not running. |
| anyone                      | `inspect NAME`         | Follow the app's console output.                                         |
| xsup, group magaox          | `startup [NAME...]`    | Start the apps enabled for automatic startup, then the INDI server; with names, start those apps. |
| xsup, group magaox          | `start NAME...`        | Start apps, enabled or not. The sentinel supervises them either way.     |
| xsup, group magaox          | `stop NAME...`         | Stop apps. They stay stopped until started again: systemd does not restart an app that was stopped on purpose, but `startup` or a reboot with `boot on` starts enabled ones. |
| xsup, group magaox          | `shutdown NAME...\|--all` | As `stop`; `--all` stops every running app of the role, enabled or not, then the INDI server. |
| xsup, group magaox          | `restart NAME...\|--all` | Restart apps; `--all` is `shutdown --all` then `startup`.              |
| privileged                  | `register`             | Register the proclist's apps with systemd, generate their drop-ins, enable them for automatic startup. Disabled apps (`#!`) with a valid `NAME.conf` are registered but not enabled: `start NAME` runs them, `startup` does not. Apps that became disabled are stopped (`--keep-running` leaves them running). Apps no longer in the proclist are stopped and deregistered. New apps start at once if the role is up. |
| privileged                  | `add NAME EXECUTABLE`  | Add the app to the proclist, disabled (`#!NAME EXECUTABLE`), and register it, to start by hand or `enable`. It needs `NAME.conf`, and must not be in the proclist already; otherwise nothing changes. |
| privileged                  | `enable NAME...\|--all` | Autostart on: remove the app's `#!` in the proclist, or add it, and register it if needed. `--all` enables every app in the proclist and every registered app. Does not start them. |
| privileged                  | `disable NAME...\|--all` | Autostart off: disable the app with `#!` in the proclist. It stays registered, to start by hand. `--all` disables every app. Does not stop them. |
| privileged                  | `delete NAME...\|--all` | Remove the app's lines from the proclist, stop it and deregister it. `--all` empties the role's proclist of apps, keeping `#` comments, and deregisters every app. |
| privileged                  | `boot on\|off`         | Start the role's enabled apps at boot, or not. Off is the default.        |

For example, to take an app out and put it back later:

```
sudo xctrl delete maths_CO2            # out of the proclist and deregistered; maths_CO2.conf stays
sudo xctrl add maths_CO2 magAOXMaths   # back as #!maths_CO2 magAOXMaths, registered, autostart off
xctrl start maths_CO2                  # run it now
sudo xctrl enable maths_CO2            # and at every startup
```

If the proclist has been edited by hand since it was last registered, `status`, `start`, `startup` and `restart`
warn, naming the apps that are new, removed, or have a different autostart (`#!`), watchdog (`-`) or executable:

```
warning: /opt/MagAOX/config/proclist_CO4.txt changed since the last registration; new: maths_new; removed: maths_q; autostart (#!), watchdog (-) or executable changed: camsci_1 maths_CO4
warning: run sudo xctrl register to apply it
```

Comments and the order of lines do not count. `add`, `enable`, `disable` and `delete` change the registration along
with the proclist, so they cause no warning.

The role comes from `-r ROLE`, `MAGAOX_ROLE`, or `/etc/profile.d/magaox_role.sh`, since `sudo` can drop the
environment.

`start`, `startup` and `restart` first bring each app's drop-in up to date with its `.conf`: if its settings
changed, e.g. a new `[systemd] watchdogSec`, or the watchdog systemd has differs from it, the drop-in is rewritten and
systemd reloaded before the app starts. An app already running keeps its settings until it is restarted. Only a
privileged user's `start`, `startup` or `restart` does that. For anyone else, such as `xsup`, the app starts with its
registered settings, and `xctrl` warns that its `.conf` changed and how to apply it:

```
warning: maths_CO4: maths_CO4.conf changed since it was registered; it will run with the registered settings
warning: to update the registered settings in systemd, a privileged user must run the start command or update the settings via the register command
```

The generated files under `/etc/systemd/system` and `/etc/magaox-<role>` are readable by everyone (files 0644,
directories 0755, whatever the caller's umask), so any user can compare them.

Stopping and disabling are separate: `stop`/`shutdown` never change what starts automatically, and
`enable`/`disable` never start or stop anything. The proclist always says what starts automatically: `enable`,
`disable`, `add` and `delete` edit the app's lines in it, leaving the rest of the file as it is, so the next `register`
keeps their effect. Editing the proclist by hand and running `sudo xctrl register` does the same.

An app that stops sending `WATCHDOG=1` within `watchdogSec` gets SIGABRT (with a core dump) and is restarted per its
`restart` policy. `restart NAME` does not restart the INDI server: apps reconnect to it through the control FIFO.

After the start limit (`startLimitBurst` starts within `startLimitIntervalSec`) an app is left failed;
`xctrl start NAME` clears that and tries again.

# Output and logging

Messages are colored on a terminal (unless `NO_COLOR` is set): errors red, as `error: ...`, warnings yellow, as
`warning: ...`, the `systemctl` commands xctrl runs blue, as `+ systemctl ...`, and completed changes green.
Errors and warnings go to stderr, the rest to stdout.

Every command except `status`, `peek` and `inspect` is also logged, with all its messages, timestamped, in the
standard MagAO-X binary log format, so it reads like any app's log:

```
logdump xctrl
```

The log is `/opt/MagAOX/logs/xctrl/YYYY_MM_DD/xctrl_<timestamp>.binlog`: one file per day, appended to by every
run, and a new one if it passes the usual 10 MB. Each run starts with its command line and the user who ran it;
`systemctl`'s own output and each command's success or failure are logged too. xctrl, setuid root, opens the log
before it gives up its privileges, so every user's commands go to the same file, which, like its directories, is
owned by `xsup` and group `magaox` and readable by everyone. Entries are standard `text_log`s, written with the
flatlogs format and the `text_log` code and schema generated by the libMagAOX build, so `make` in `utils/xctrl`
needs libMagAOX built first.

# Testing without systemd

`--dry-run` prints the files and `systemctl` commands instead, and logs nothing; with `--top DIR`, xctrl logs under `DIR/logs`, as the calling user. `--sysroot DIR --top DIR` writes the units under
`DIR/etc` from a test MagAO-X tree, and runs only the `systemctl --root` commands that work offline (enable,
disable). Check the generated units with
`SYSTEMD_UNIT_PATH=DIR/etc/systemd/system: systemd-analyze verify magaox-<role>-app@NAME.service`.
