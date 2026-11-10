resuctrl
========

`resuctrl`, also installed as `xctrl`, runs the MagAO-X apps of a MAGAOX\_ROLE as systemd services. It replaces both
`resurrector_indi` and the tmux-based `xctrl`: systemd starts, watches and restarts each app, and `resuctrl` turns the
proclist and the apps' configuration into systemd units and drives `systemctl`.

Each app publishes to systemd's watchdog, sending `READY=1` when started and `WATCHDOG=1` from the main loop
(`sd_notify` protocol, see `libMagAOX/sys/sdNotify.hpp` and `magaox/indi/device.py`).

`resuctrl reset` parses the server's proclist and registers apps that are present, and de-registers apps that are 
already registered with systemd that no longer exist in the proclist (for a given MAGAOX\_ROLE).   

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

`/opt/MagAOX/config/proclist_<role>.txt` is read as before: `NAME EXECUTABLE` per line.

| Line                       | Meaning                                                                       |
|----------------------------|-------------------------------------------------------------------------------|
| `camsci1  picamCtrl`       | an app, started automatically                                                 |
| `#camsci1  picamCtrl`      | a disabled app: registered only if `camsci1.conf` exists, started by hand; `enable`/`disable` uncomments/comments |
| `##camsci_old  camsci`     | only a comment: ignored by `reset`, never touched by `enable`, `disable` or `delete` |
| `## any text`, `# any text` | only a comment                                                               |

A `#` line is taken for a disabled app only if it is exactly `#NAME EXECUTABLE` and `NAME.conf` exists, so ordinary
`#` comments are safe; use `##` to keep an app's line in the file without it ever being registered.

| Prefix      | Meaning                                                                     |
|-------------|-----------------------------------------------------------------------------|
| none, `py:` | the app notifies systemd: `Type=notify`, with a watchdog                    |
| `-`, `nhb:` | the app sends no heartbeat: `Type=simple`, no watchdog, still restarted     |

The app run by `xindiserver` is the role's INDI server. Apps are ordered after it, so they stop before it and their
last messages get through; it is ordered after the `sshDigger` tunnels.

# Settings

Default systemd app settings are in a template. Any new apps registered without any overrides or specific settings wil 
inherit the template values. The template overrieds can be changed via `magaox.conf`, or one app in its
`.conf`, under the section for `[systemd]`:

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
| anyone                      | `status [NAME...]`     | Every registered app, one line each, `name: state (pid: N)`, colored and sorted by state as the old resuctrl: running green, manually started (running with autostart off) blue, disabled (commented out, not running) yellow, not started cyan, starting/stopping magenta, dead red. `-v` adds a table with autostart, restarts, uptime, seconds since the last watchdog ping, watchdog timeout, CPU % now, memory and GPU memory. |
| anyone                      | `peek [NAME...]`       | State and the last `logdump` lines of the apps (`-n LINES`, default 10), with the console output of an app that is not running. |
| anyone                      | `inspect NAME`         | Follow the app's console output, which `xctrl` showed in its tmux session. |
| xsup, group magaox          | `startup [NAME...]`    | Start the apps enabled for automatic startup, then the INDI server; with names, start those apps. |
| xsup, group magaox          | `start NAME...`        | Start apps, enabled or not. Systemd supervises them either way.          |
| xsup, group magaox          | `stop NAME...`         | Stop apps. They stay stopped until started again: systemd does not restart an app that was stopped on purpose, but `startup` or a reboot with `boot on` starts enabled ones. |
| xsup, group magaox          | `shutdown NAME...\|--all` | As `stop`; `--all` stops every running app of the role, enabled or not, then the INDI server. |
| xsup, group magaox          | `restart NAME...\|--all` | Restart apps; `--all` is `shutdown --all` then `startup`.              |
| xsup, group magaox          | `defib NAME...`        | Same as `restart`, kept for habit.                                       |
| sudo                        | `reset`                | Register the proclist's apps, generate their drop-ins, enable them for automatic startup. A commented-out line, `#NAME EXECUTABLE`, whose app has a valid `NAME.conf` is registered too but not enabled: `start NAME` runs it, `startup` does not. Apps that became commented out are disabled and stopped (`--keep-running` leaves them running). Apps no longer in the proclist at all are stopped and deregistered. New apps start at once if the role is up. |
| sudo                        | `enable NAME...`       | Autostart on: uncomment the app in the proclist, or add it, and register it if needed. Does not start it. |
| sudo                        | `disable NAME...`      | Autostart off: comment the app out in the proclist. It stays registered, to start by hand. Does not stop it. |
| sudo                        | `delete NAME...`       | Remove the app's lines from the proclist, stop it and deregister it.      |
| sudo                        | `boot on\|off`         | Automatically start the MAGAOX\_ROLE enabled apps at boot, or not. Off is the default.           |

The role comes from `-r ROLE`, `MAGAOX_ROLE`, or `/etc/profile.d/magaox_role.sh`, since `sudo` can drop the
environment.

`start`, `startup` and `restart` first bring each app's drop-in up to date with its `.conf`: if its settings
changed, e.g. a new `[resurrectee] timeout`, or the watchdog systemd has differs from it, the drop-in is rewritten and
systemd reloaded before the app starts. An app already running keeps its settings until it is restarted.

Stopping and disabling are separate: `stop`/`shutdown` never change what starts automatically, and
`enable`/`disable` never start or stop anything. The proclist always says what starts automatically: `enable`,
`disable` and `delete` edit the app's lines in it, leaving the rest of the file as it is, so the next `reset` keeps
their effect. Editing the proclist by hand and running `sudo resuctrl reset` does the same.

`defib` used to make the resurrector restart an app by writing an expired hexbeat. Systemd's watchdog now does that
by itself: an app that stops sending `WATCHDOG=1` gets SIGABRT (with a core dump) and is restarted per its
`restart` policy. So `defib` is a plain `restart`. Unlike `xctrl`, `restart NAME` no longer restarts the INDI server
too: apps reconnect to it through the control FIFO.

After the start limit (`startLimitBurst` starts within `startLimitIntervalSec`) an app is left failed, as the
resurrector gave up after 10 restarts; `resuctrl start NAME` clears that and tries again.

# Testing without systemd

`--dry-run` prints the files and `systemctl` commands instead. `--sysroot DIR --top DIR` writes the units under
`DIR/etc` from a test MagAO-X tree, and runs only the `systemctl --root` commands that work offline (enable,
disable). Check the generated units with
`SYSTEMD_UNIT_PATH=DIR/etc/systemd/system: systemd-analyze verify magaox-<role>-app@NAME.service`.
