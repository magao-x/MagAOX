resuctrl
========

`resuctrl`, also installed as `xctrl`, runs the MagAO-X apps of a MAGAOX\_ROLE as systemd services. It replaces
`resurrector_indi` and the tmux-based `xctrl`: systemd starts, watches and restarts each app, and `resuctrl` turns the
proclist and the apps' configuration into systemd units and drives `systemctl`.

The INDI FIFOs (`.in`, `.out`, `indiserver.ctrl`) are unchanged. The resurrector's `.hb` hexbeat FIFOs are gone,
replaced by systemd's watchdog: apps send `READY=1` once started and `WATCHDOG=1` from their main loop (`sd_notify` protocol, see
`libMagAOX/sys/sdNotify.hpp` and `magaox/indi/device.py`).

# Generated files

For `MAGAOX_ROLE=ICC`, `sudo resuctrl reset` writes:

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
/opt/MagAOX/sys/resurrector_proclist/proclist_ICC.txt.active
                                          the proclist lines last applied by reset
```

Each app runs as `/opt/MagAOX/bin/<executable> -n <name>`, as user `xsup`, with `MAGAOX_ROLE` set. Its output goes
to the journal (`resuctrl peek NAME`); its binary logs are unchanged (`logdump NAME`).

Unit names use the role in lower case. App names may contain `-`; units use them as is (`%i`).

# Proclist

`/opt/MagAOX/config/proclist_<role>.txt` is read as before: `NAME EXECUTABLE` per line, `#` comments.

| Prefix      | Meaning                                                                     |
|-------------|-----------------------------------------------------------------------------|
| none, `py:` | the app notifies systemd: `Type=notify`, with a watchdog                    |
| `-`, `nhb:` | the app sends no heartbeat: `Type=simple`, no watchdog, still restarted     |

The app run by `xindiserver` is the role's INDI server. Apps are ordered after it, so they stop before it and their
last messages get through; it is ordered after the `sshDigger` tunnels.

# Settings

Defaults are in the template. They can be changed for the whole role in `magaox.conf`, or for one app in its
`.conf`, under `[systemd]`:

| Key                     | systemd directive        | Default  |
|-------------------------|--------------------------|----------|
| `restart`               | `Restart=`               | always   |
| `restartSec`            | `RestartSec=`            | 5        |
| `startLimitBurst`       | `StartLimitBurst=`       | 10       |
| `startLimitIntervalSec` | `StartLimitIntervalSec=` | 600      |
| `watchdogSec`           | `WatchdogSec=`           | 10       |
| `timeoutStartSec`       | `TimeoutStartSec=`       | 120      |
| `timeoutStopSec`        | `TimeoutStopSec=`        | 30       |
| `killSignal`            | `KillSignal=`            | SIGTERM  |

Times are seconds, or `infinity`. An app's existing `[resurrectee] timeout = RUN[,STARTUP]` still works: `RUN` sets
`WatchdogSec` (negative disables it) and `STARTUP` sets `TimeoutStartSec`, the time allowed until the app reports
`READY=1`. `[systemd]` settings override them. The apps accept these keys and ignore them.

Example, in `camsci1.conf`:

```
[resurrectee]
timeout = 20,180

[systemd]
restart = on-failure
killSignal = SIGINT
```

`reset` checks every value and every executable; an app with a problem is reported and not (re)registered, and
keeps whatever registration it had.

# Commands

`resuctrl` is also installed as `xctrl`, and its commands work as `xctrl`'s did: `startup` starts the apps, `stop`
and `shutdown` stop them, `restart` restarts them. The INDI server always starts up last.

Like `xctrl`, it is run by regular users. It is installed setuid root so that it can ask systemd to start and stop
the role's apps, but it only does that for root, `xsup` and members of group `magaox`, and only for registered apps.
Resetting the systemd registration for which apps should be active via `resuctrl reset` requires sudo. 

| Who                         | Command                | What it does                                                             |
|-----------------------------|------------------------|--------------------------------------------------------------------------|
| anyone                      | `status [NAME...]`     | One line per app, `name: state (pid: N)`, colored and sorted by state as the old resuctrl: running green, disabled yellow, not started cyan, starting/stopping magenta, dead red. `-v` adds a table with autostart, restarts, uptime, seconds since the last watchdog ping, watchdog timeout, CPU % now, memory and GPU memory. |
| anyone                      | `peek [NAME...]`       | State and the last `logdump` lines of the apps (`-n LINES`, default 10), with the console output of an app that is not running. |
| anyone                      | `inspect NAME`         | Follow the app's console output, which `xctrl` showed in its tmux session. |
| xsup, group magaox          | `startup [NAME...]`    | Start the apps enabled for automatic startup, then the INDI server; with names, start those apps. |
| xsup, group magaox          | `start NAME...`        | Start apps, enabled or not. Systemd supervises them either way.          |
| xsup, group magaox          | `stop NAME...`         | Stop apps. They stay stopped until started again: systemd does not restart an app that was stopped on purpose, but `startup` or a reboot with `boot on` starts enabled ones. |
| xsup, group magaox          | `shutdown NAME...\|--all` | As `stop`; `--all` stops every running app of the role, enabled or not, then the INDI server. |
| xsup, group magaox          | `restart NAME...\|--all` | Restart apps; `--all` is `shutdown --all` then `startup`.              |
| xsup, group magaox          | `defib NAME...`        | Same as `restart`, kept for habit.                                       |
| sudo                        | `reset`                | Register the proclist's apps, generate their drop-ins, enable them for automatic startup. Apps no longer listed are disabled and stopped (`--keep-running` leaves them running) but stay registered. New apps start at once if the role is up. Changed settings apply at an app's next start. |
| sudo                        | `enable NAME...`       | Enable apps for automatic startup.                                       |
| sudo                        | `disable NAME...`      | Disable apps for automatic startup. A running app keeps running.         |
| sudo                        | `delete NAME...`       | Stop, disable and deregister apps.                                       |
| sudo                        | `boot on\|off`         | Start the role's enabled apps at boot, or not. Off by default.           |

The role comes from `-r ROLE`, `MAGAOX_ROLE`, or `/etc/profile.d/magaox_role.sh`, since `sudo` can drop the
environment.

Stopping and disabling are separate: `stop`/`shutdown` never change what starts automatically, and
`enable`/`disable` never start or stop anything. `enable` and `disable` change systemd only: the next `reset` enables
exactly the apps in the proclist again. To change the set for good, edit the proclist and run `sudo resuctrl reset`.

`defib` used to make the resurrector restart an app by writing an expired hexbeat. Systemd's watchdog now does that
by itself: an app that stops sending `WATCHDOG=1` gets SIGABRT (with a core dump) and is restarted per its
`restart` policy. So `defib` is a plain `restart`. Unlike `xctrl`, `restart NAME` no longer restarts the INDI server
too: apps reconnect to it through the control FIFO.

After the start limit (`startLimitBurst` starts within `startLimitIntervalSec`) an app is left failed, as the
resurrector gave up after 10 restarts; `resuctrl start NAME` clears that and tries again.

# Moving from xctrl or resurrector_indi

The resurrector, its `.hb` hexbeat FIFOs and the old `xctrl` and `resuctrl` scripts are removed.

1. Stop the old processes the old way, before installing, e.g. `xctrl shutdown --all` or
   `resuctrl shutdown --all`.
2. Build and install: `resuctrl` is in `utils_to_build`; installing it also links `xctrl` to it.
3. `sudo resuctrl reset`, then `resuctrl startup`, then `resuctrl status`.
4. Optionally `sudo resuctrl boot on`.

Old `.hb` files in `/opt/MagAOX/drivers/fifos` can be deleted.

# Testing without systemd

`--dry-run` prints the files and `systemctl` commands instead. `--sysroot DIR --top DIR` writes the units under
`DIR/etc` from a test MagAO-X tree, and runs only the `systemctl --root` commands that work offline (enable,
disable). Check the generated units with
`SYSTEMD_UNIT_PATH=DIR/etc/systemd/system: systemd-analyze verify magaox-<role>-app@NAME.service`.
