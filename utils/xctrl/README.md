xctrl
=====

`xctrl` runs the MagAO-X apps of a MAGAOX\_ROLE as systemd services. Systemd starts, watches and restarts each app;
`xctrl` turns the proclist and the apps' configuration into systemd units and drives `systemctl`.

# Terms

| Term          | Meaning                                                                                       |
|---------------|-----------------------------------------------------------------------------------------------|
| `xctrl`       | this program: the one command for registering, starting, stopping and checking the apps        |
| sentinel      | the systemd supervision of a role's apps: the units `xctrl reset` generates, their watchdog and restart policy, and the proclist record in `sys/sentinel_proclist` |
| watchdog      | systemd's check that an app is alive: the app sends `WATCHDOG=1` at least every `watchdogSec`, or systemd restarts it |
| `[systemd]`   | the configuration section, in `magaox.conf` or an app's `.conf`, holding the sentinel settings |

Each app reports to systemd with the `sd_notify` protocol (see `libMagAOX/sys/sdNotify.hpp` and
`magaox/indi/device.py`): `READY=1` once started, `WATCHDOG=1` from its main loop, and `STOPPING=1` on the way out.

`xctrl reset` parses the role's proclist, registers the apps in it, and deregisters apps that are registered but no
longer in the proclist.

# Generated files

For `MAGAOX_ROLE=ICC`, `sudo xctrl reset` writes:

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
                                          the proclist lines last applied by reset
```

Each app runs as `/opt/MagAOX/bin/<executable> -n <name>`, as user `xsup`, with `MAGAOX_ROLE` set, and
`PYTHONUNBUFFERED=1` so that Python apps' output is not held back. Its output goes to the journal (`xctrl peek NAME`); its binary logs are unchanged (`logdump NAME`).

Unit names use the role in lower case. App names may contain `-`; units use them as is (`%i`).

# Proclist

`/opt/MagAOX/config/proclist_<role>.txt` has one app per line, `NAME EXECUTABLE`.

| Line                    | Meaning                                                                          |
|-------------------------|----------------------------------------------------------------------------------|
| `camsci1  picamCtrl`    | an app, started automatically                                                    |
| `#camsci1  picamCtrl`   | a disabled app: registered if `camsci1.conf` exists, started only by hand        |
| `# camsci1  picamCtrl`  | the same: any line starting with a single `#` is a disabled app                  |
| `## any text`           | a comment: ignored by `reset`, never touched by `enable`, `disable` or `delete`  |

A single `#` always means a disabled app, so comments need `##`. A single-`#` line that is not `NAME EXECUTABLE`,
or whose app has no `.conf`, is reported by `reset` and otherwise ignored.

| Prefix      | Meaning                                                                         |
|-------------|---------------------------------------------------------------------------------|
| none        | the app notifies systemd: `Type=notify`, with a watchdog                        |
| `-`         | the app does not notify systemd: `Type=simple`, no watchdog, still restarted    |

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

Times are seconds, or `infinity`. The apps accept these keys and ignore them; only `xctrl` reads them. The
`[resurrectee]` section is no longer used: `reset` reports it as an error, and its `timeout = RUN,STARTUP` becomes
`watchdogSec = RUN` and `timeoutStartSec = STARTUP`.

Example, in `camsci1.conf`:

```
[systemd]
timeoutStartSec = 180
watchdogSec = 20
restart = on-failure
killSignal = SIGINT
```

`reset` checks every value and every executable; an app with a problem is reported and not (re)registered, and
keeps whatever registration it had.

# Commands

Regular users run `xctrl`. It is installed setuid root so that it can ask systemd to start and stop the role's apps,
but it only does that for root, `xsup` and members of group `magaox`, and only for registered apps. Changing which
apps are registered or enabled needs sudo.

| Who                         | Command                | What it does                                                             |
|-----------------------------|------------------------|--------------------------------------------------------------------------|
| anyone                      | `status [NAME...]`     | Every registered app, one line each, `name: state (pid: N)`, colored and sorted by state: running green, manually started (running with autostart off) blue, disabled (commented out, not running) yellow, not started cyan, starting/stopping magenta, dead red. `-v` adds a table with autostart, restarts, uptime, seconds since the last watchdog notification, watchdog timeout, CPU % now, memory and GPU memory. |
| anyone                      | `peek [NAME...]`       | State and the last `logdump` lines of the apps (`-n LINES`, default 10), with the console output of an app that is not running. |
| anyone                      | `inspect NAME`         | Follow the app's console output.                                         |
| xsup, group magaox          | `startup [NAME...]`    | Start the apps enabled for automatic startup, then the INDI server; with names, start those apps. |
| xsup, group magaox          | `start NAME...`        | Start apps, enabled or not. The sentinel supervises them either way.     |
| xsup, group magaox          | `stop NAME...`         | Stop apps. They stay stopped until started again: systemd does not restart an app that was stopped on purpose, but `startup` or a reboot with `boot on` starts enabled ones. |
| xsup, group magaox          | `shutdown NAME...\|--all` | As `stop`; `--all` stops every running app of the role, enabled or not, then the INDI server. |
| xsup, group magaox          | `restart NAME...\|--all` | Restart apps; `--all` is `shutdown --all` then `startup`.              |
| sudo                        | `reset`                | Register the proclist's apps, generate their drop-ins, enable them for automatic startup. Disabled apps (single `#`) with a valid `NAME.conf` are registered but not enabled: `start NAME` runs them, `startup` does not. Apps that became disabled are stopped (`--keep-running` leaves them running). Apps no longer in the proclist are stopped and deregistered. New apps start at once if the role is up. |
| sudo                        | `enable NAME...`       | Autostart on: uncomment the app in the proclist, or add it, and register it if needed. Does not start it. |
| sudo                        | `disable NAME...`      | Autostart off: comment the app out in the proclist. It stays registered, to start by hand. Does not stop it. |
| sudo                        | `delete NAME...`       | Remove the app's lines from the proclist, stop it and deregister it.      |
| sudo                        | `boot on\|off`         | Start the role's enabled apps at boot, or not. Off is the default.        |

The role comes from `-r ROLE`, `MAGAOX_ROLE`, or `/etc/profile.d/magaox_role.sh`, since `sudo` can drop the
environment.

`start`, `startup` and `restart` first bring each app's drop-in up to date with its `.conf`: if its settings
changed, e.g. a new `[systemd] watchdogSec`, or the watchdog systemd has differs from it, the drop-in is rewritten and
systemd reloaded before the app starts. An app already running keeps its settings until it is restarted.

Stopping and disabling are separate: `stop`/`shutdown` never change what starts automatically, and
`enable`/`disable` never start or stop anything. The proclist always says what starts automatically: `enable`,
`disable` and `delete` edit the app's lines in it, leaving the rest of the file as it is, so the next `reset` keeps
their effect. Editing the proclist by hand and running `sudo xctrl reset` does the same.

An app that stops sending `WATCHDOG=1` within `watchdogSec` gets SIGABRT (with a core dump) and is restarted per its
`restart` policy. `restart NAME` does not restart the INDI server: apps reconnect to it through the control FIFO.

After the start limit (`startLimitBurst` starts within `startLimitIntervalSec`) an app is left failed;
`xctrl start NAME` clears that and tries again.

# Testing without systemd

`--dry-run` prints the files and `systemctl` commands instead. `--sysroot DIR --top DIR` writes the units under
`DIR/etc` from a test MagAO-X tree, and runs only the `systemctl --root` commands that work offline (enable,
disable). Check the generated units with
`SYSTEMD_UNIT_PATH=DIR/etc/systemd/system: systemd-analyze verify magaox-<role>-app@NAME.service`.
