xctrl
=====

`xctrl` runs the MagAO-X apps of a MAGAOX\_ROLE as systemd services. Systemd starts, watches and restarts each app;
`xctrl` turns the proclist and the apps' configuration into systemd units and drives `systemctl`.

# Terms

| Term          | Meaning                                                                                       |
|---------------|-----------------------------------------------------------------------------------------------|
| `xctrl`       | this program: the one command for syncing, starting, stopping and checking the apps           |
| sentinel      | the systemd supervision of a role's apps: the units `xctrl sync` generates, and their watchdog and restart policy |
| watchdog      | systemd's check that an app is alive: the app sends `WATCHDOG=1` at least every `watchdogSec`, or systemd restarts it |
| `[systemd]`   | the configuration section, in `magaox.conf` or an app's `.conf`, holding the sentinel settings |

Each app reports to systemd with the `sd_notify` protocol (see `libMagAOX/sys/sdNotify.hpp` and
`magaox/indi/device.py`): `READY=1` once started, `WATCHDOG=1` from its main loop, and `STOPPING=1` on the way out.

`xctrl sync` parses the role's proclist, registers new and changed apps with systemd, and unregisters apps that are
no longer in the proclist and not running. It never starts or stops an app.

# Generated files

For `MAGAOX_ROLE=ICC`, `xctrl sync` writes:

```
/etc/systemd/system/
    magaox-icc-app@.service               template shared by all apps of the role, without settings
    magaox-icc.target                     wants the apps enabled for automatic startup
    magaox-icc.target.wants/              one link per enabled app (systemctl enable)
    magaox-icc-app@camsci1.service.d/
        override.conf                     camsci1's command, and its settings: magaox.conf's
                                          [systemd] with camsci1.conf's in their place
/etc/magaox-icc/
    environment                           optional, not generated: extra environment for all apps

/etc/magaox/
    systemd-directives.txt                systemd default allowlist for directives, installed via `make install` in utils/xctrl
    systemd-directives.local.txt          allowlist xctrl uses for app `.conf` and `magaox.conf` directives. Users edit this
```

The proclist says what should be registered; the drop-ins and links are what is registered. An app is registered when
it has a drop-in: its name is the drop-in's directory, its executable is in `ExecStart=`, and `Type=simple` marks the
`-` prefix. Its link in `magaox-<role>.target.wants/` makes it start automatically. There is no other registry: the
first `sync` removes the `/etc/magaox-<role>/apps` registry and the `sys/sentinel_proclist/proclist_<ROLE>.txt.active`
record of earlier versions, and rewrites every drop-in once, as their header changed.

`sync` writes a file only if its content changed, and runs `systemctl daemon-reload` only if it wrote or removed one.

Each app runs as `/opt/MagAOX/bin/<executable> -n <name>`, as user `xsup`, with `MAGAOX_ROLE` set, and
`PYTHONUNBUFFERED=1` so that Python apps' output is not held back. Its output goes to the journal
(`xctrl peek NAME`); its binary logs are unchanged (`logdump NAME`). `xctrl inspect` shows `journald`
 output for an app and requires a user with `adm`, `systemd-journal`, or `sudo` to view.

Unit names use the role in lower case. App names may contain `-`; units use them as is (`%i`).

# Proclist

`/opt/MagAOX/config/proclist_<role>.txt` has one app per line, `NAME EXECUTABLE`.

| Line                     | Meaning                                                                          |
|--------------------------|----------------------------------------------------------------------------------|
| `camsci1  picamCtrl`     | an app, registered and started automatically, with a watchdog                    |
| `-camsci1  picamCtrl`    | an app, registered and started automatically, without a watchdog                 |
| `#!camsci1  picamCtrl`   | a disabled app: registered if `camsci1.conf` exists, but started only by hand    |
| `#!-camsci1  picamCtrl`  | a disabled app without a watchdog                                                |
| `#camsci1  picamCtrl`    | a comment, like any line starting with `#` but not `#!`                          |
| `#-camsci1  picamCtrl`   | a comment too                                                                    |

`#!` may be followed by spaces: `#! camsci1 picamCtrl` is the same disabled app. A `#!` line that is not
`NAME EXECUTABLE`, or whose app has no `.conf`, is reported by `sync` and otherwise ignored. Comments are never
touched by `enable`, `disable`, `add` or `delete`.

| Prefix      | Meaning                                                                         |
|-------------|---------------------------------------------------------------------------------|
| none        | the app notifies systemd: `Type=notify`, with a watchdog                        |
| `-`         | the app does not notify systemd: `Type=simple`, no watchdog, still restarted if it exits |

On the command line, `--` lets a name start with `-`: `xctrl add -- -camsci1 picamCtrl`.

The app run by `xindiserver` is the role's INDI server. `sshDigger` tunnels start first, then the INDI server,
then regular apps on `startup`, and on `shutdown` apps stop first, then INDI server, then ssh tunnels. 

`xctrl` calls to `systemctl` would execute: 

```
$ xctrl startup
+ systemctl start magaox-<role>-app@<sshDigger>.service 
+ systemctl start magaox-<role>-app@is<role>.service
+ systemctl start magaox-<role>-app@<app>.service
+ systemctl start magaox-<role>.target

$ xctrl shutdown --all
+ systemctl stop magaox-<role>.target
+ systemctl stop magaox-<role>-app@<app>.service
+ systemctl stop magaox-<role>-app@is<role>.service
+ systemctl stop magaox-<role>-app@<sshDigger>.service 
```

# Settings

xctrl sets no defaults of its own. `magaox.conf` sets the default `[systemd]` configuration for every app of the 
role, and an app's `.conf` overrides them for that app. Settings go into the app's drop-in, the app's value in place of 
the role's, so each directive is set once. If a directive is not present in either the `magaox.conf` or the app's 
.conf, the app would inherit systemd's own default, which is typically not what MagAOX desires, e.g.: `Restart=no`,
providing no watchdog.

`magaox.conf` should at a minimum contain these directives (set values to your needs):

```
[systemd]
Restart = always
RestartSec = 5
StartLimitBurst = 10
StartLimitIntervalSec = 600
WatchdogSec = 10
TimeoutStartSec = 120
TimeoutStopSec = 30
```

| Directive               | Meaning                                                                                    |
|-------------------------|--------------------------------------------------------------------------------------------|
| `Restart`               | when systemd restarts the app, e.g. `always`, `on-failure`, `no` (systemd's default)       |
| `RestartSec`            | wait before a restart                                                                      |
| `StartLimitBurst`       | starts allowed within `StartLimitIntervalSec`; after that the app is left failed           |
| `StartLimitIntervalSec` | the interval for `StartLimitBurst`; 0 allows unlimited restarts                            |
| `WatchdogSec`           | time allowed between `WATCHDOG=1`s, once ready; 0, systemd's default, means no watchdog    |
| `TimeoutStartSec`       | time allowed from start to `READY=1`                                                       |
| `TimeoutStopSec`        | time allowed to stop after `KillSignal` (SIGTERM by default) before SIGKILL                |

**`WatchdogSec` is needed for the watchdog.** An app that reports to systemd (`sd_notify`, no `-` prefix in the
proclist) is watched only if `WatchdogSec` is set, in `magaox.conf` or its `.conf`, and above 0. An app that does not
report to systemd must have the `-` prefix: it then gets no watchdog whatever is set. `xctrl status` shows
`no watchdog` for an app without one, and `status -v` shows `none` in its WATCHDOG column:

```
maths_s: running (pid: 4242), no watchdog
```

A key is a systemd directive in any case: `runtimeMaxSec`, `RuntimeMaxSec` and `runtimemaxsec` all set
`RuntimeMaxSec`, spelled as the host's systemd spells it. Any directive in the allow list (below) may be set, e.g.
`runtimeMaxSec`, `nice`, `cpuAffinity`, `memoryMax`, `limitMEMLOCK` or `environment`; `xctrl` writes each in the
right section (`man systemd.unit`, `systemd.service`, `systemd.exec`, `systemd.kill`, `systemd.resource-control`).
`xctrl` learns which directives exist, and in which section, from the host's systemd itself
(`systemd --dump-configuration-items`); without that list `sync` stops with an error. Values are written as given, in
any form systemd accepts, e.g. `500ms`, `2min`, `infinity` or `SIGABRT`, after systemd itself has checked them
(`systemd-analyze verify` on a scratch unit, run as the caller, once per run for all values).

## The allow list

`sync` writes the units as root, for any user who can edit the proclist and the `.conf` files. Many directives are
applied by systemd as root, e.g. `StandardOutput=file:PATH` or `BindPaths`, and would let such a user become root.
So a `[systemd]` key is used only if its directive is in the allow list, one directive per line, `#` starting a
comment. xctrl reads `/etc/magaox/systemd-directives.local.txt`, or, if it does not exist,
`/etc/magaox/systemd-directives.txt`. The file must be owned by root and writable only by root, as must its directory;
edit the `.local` copy with `sudo` to add or remove directives.

`make install` in `utils/xctrl` installs two files, root:root 0644:

| File                                        | `make install`                                                     |
|---------------------------------------------|--------------------------------------------------------------------|
| `/etc/magaox/systemd-directives.txt`        | the defaults, `utils/xctrl/systemd-directives.txt`, always overwritten; it warns if they differ from the file that was there. Do not edit it. |
| `/etc/magaox/systemd-directives.local.txt`  | a copy of the defaults, made only if it does not exist; never changed afterwards. This is the file to edit, and the one xctrl reads. |

After an update, compare the two to pick up new defaults. With neither file, `sync`, and an editor's `start` or
`startup`, stop with:

```
error: no directive allow list /etc/magaox/systemd-directives.local.txt or systemd-directives.txt: install it with sudo make install in utils/xctrl; nothing synced
```

The default allows the watchdog, start and stop timing and restart directives (`WatchdogSec`, `TimeoutStartSec`,
`RuntimeMaxSec`, `Restart`, `StartLimitBurst`, `KillSignal`, ...), the CPU (`Nice`, `CPUAffinity`,
`CPUSchedulingPolicy`, `CPUSchedulingPriority`, `NUMAPolicy`, `AllowedCPUs`, ...), memory and I/O (`MemoryMax`,
`TasksMax`, `IOWeight`, `OOMScoreAdjust`, ...), every `Limit...` (`LimitMEMLOCK`, `LimitRTPRIO`, `LimitCORE`, ...),
and `Environment`, e.g. `environment = CUDA_VISIBLE_DEVICES=1` to choose a GPU.

xctrl sets any directive the allow list has; it refuses none for what it does. Adding one is the site's decision, so
know these before adding them to `systemd-directives.local.txt`:

| Directives | Why they are dangerous |
|---|---|
| `ExecStart`, `ExecStartPre`, `ExecStartPost`, `ExecStop`, `ExecStopPost`, `ExecReload`, `ExecCondition` | run any command; a `+` prefix runs it as root, so a proclist editor could become root |
| `User`, `Group`, `DynamicUser`, `SupplementaryGroups`, `PAMName` | change who the app runs as, e.g. root |
| `AmbientCapabilities`, `CapabilityBoundingSet`, `SecureBits`, `NoNewPrivileges` | give the app privileges `xsup` does not have |
| `StandardInput`, `StandardOutput`, `StandardError` (`file:...`), `TTYPath`, `PIDFile`, `RootDirectory`, `RootImage`, `BindPaths`, `ReadWritePaths`, `LoadCredential`, `StateDirectory`, `LogsDirectory`, `PermissionsStartOnly`, `RootDirectoryStartOnly` | make systemd open, create or mount files and paths as root, e.g. overwrite any file |
| `FailureAction`, `SuccessAction`, `StartLimitAction`, `JobTimeoutAction` | act on the whole host, e.g. reboot it |
| `Type`, `NotifyAccess`, `ExecStart`, `User` | set by xctrl itself; setting them breaks the units (use the `-` proclist prefix instead of `Type`) |

The apps ignore the `[systemd]` section; only `xctrl` reads it. A key that can not be used is a warning, and the app
is registered without it, so with the role's value or systemd's default:

```
warning: [systemd] runtimeMaxSec directive for maths_CO2 not in systemd-directives allow list; ignoring directive
warning: [systemd] fooSec directive for maths_CO2 does not exist on this version of systemd; ignoring directive
warning: [systemd] WatchdogSec = ten for maths_CO2 is an invalid value (systemd: Failed to parse sec value, ignoring: ten); ignoring directive
```

Likewise a directive that may not be set, the same directive spelled twice, or a value with a control character.
`sync` also warns once about entries of the allow list that this systemd does not have. Without `systemd-analyze`,
`sync` warns that values are not checked, and systemd ignores any it can not use when it loads the units. Only a
missing executable keeps an app from being registered.

Example, in `camsci1.conf`:

```
[systemd]
timeoutStartSec = 180
watchdogSec = 20
restart = on-failure
killSignal = SIGINT
limitMEMLOCK = infinity
```

# Commands

Any user can run `xctrl`. It is installed setuid root, and decides from who ran it what it does for them:

- **Looking** (`status`, `peek`): anyone. `inspect` follows the journal as the caller, so it needs `sudo` or
  membership in group `systemd-journal` or `adm`; `peek` shows the last few lines of the binary log (trimmed 
  `logdump`).
- **Starting and stopping** registered apps (`start`, `stop`, `startup`, `shutdown`, `restart`): operators, meaning
  root, `xsup` and members of group `magaox`. `xsup` can start a disabled (`#!`) app by hand.
- **Changing what is registered or enabled** (`sync`, `add`, `enable`, `disable`, `delete`, `boot`): editors,
  meaning root (e.g. through `sudo`) or a user who may write the role's proclist. The config checkout is normally
  group `magaox-dev` writable, so its members qualify. `xsup` who is not able to write the proclist, would get
  an error running these, for example:

  ```
  error: cannot enable camsci1 as user xsup: enable needs write access to /opt/MagAOX/config/proclist_ICC.txt (group magaox-dev), or sudo
  ```

- **Changing the allow list**: root only, with `sudo`.

The proclist and the apps' `.conf` files are protected only by their own permissions, so they should not be writable
by regular users.

xctrl gives up root as soon as it starts: it runs as the caller, and reads the proclist, `magaox.conf` and the apps'
`.conf` files with the caller's permissions. It takes root back only for a moment, to open its log, to write or remove
a generated file (only the role's template, target, links and `magaox-<role>-app@NAME.service.d/override.conf`), and
to run the `systemctl` commands that change something. Files it writes are owned by root:root. `status`, `peek`,
`inspect` and `--help` give up root for good. This guards against mistakes in what xctrl parses and builds, but does
not make it safe against a memory-safety bug as root can still be taken back while it runs.

| Who                         | Command                | What it does                                                             |
|-----------------------------|------------------------|--------------------------------------------------------------------------|
| anyone                      | `status [NAME...]`     | Every registered app, one line each, `name: state (pid: N)`, colored and sorted by state: running green, manually started (running with autostart off) blue, disabled (`#!`, not running) yellow, not started cyan, starting/stopping magenta, dead red. `-v` for verbose options adds a table with autostart, restarts, uptime, seconds since the last watchdog notification, watchdog timeout, CPU % now, memory and GPU memory. `-q` for quiet hushes warnings and is a usable option with all other commands. It does not hush errors. |
| anyone                      | `peek [NAME...]`       | State and the last `logdump` lines of the apps (`-n LINES`, default 10), with the console output of an app that is not running. |
| anyone                      | `inspect NAME`         | Follow the app's console output. `journalctl` on the .service will show similar output. |
| xsup, group magaox          | `startup [NAME...]`    | Start the INDI server, then the apps enabled for automatic startup; with names, start those apps, the INDI server first. An editor's `startup` syncs first if the proclist or a `.conf` changed. |
| xsup, group magaox          | `start NAME...`        | Start apps, enabled or not. The sentinel supervises them either way. An editor's `start` syncs first, as `startup` does, so a new app can be started at once. |
| xsup, group magaox          | `stop NAME...`         | Stop apps. They stay stopped until started again: systemd does not restart an app that was stopped on purpose, but `startup` or a reboot with `boot on` starts enabled ones. |
| xsup, group magaox          | `shutdown NAME...\|--all` | As `stop`; `--all` stops every running app of the role, enabled or not, then the INDI server. |
| xsup, group magaox          | `restart NAME...\|--all` | Restart apps; `--all` is `shutdown --all` then `startup`.              |
| editors                     | `sync`                 | Bring systemd in line with the proclist and the `.conf` files: register new apps, rewrite the drop-ins of apps whose executable, `-` prefix or `[systemd]` settings changed, and set autostart from `#!`. Disabled apps (`#!`) with a `NAME.conf` are registered but not enabled: `start NAME` runs them, `startup` does not. Apps no longer in the proclist are unregistered, unless they are running: those are kept, with a warning to stop them first, but their autostart is turned off. Starts and stops nothing: an app newly disabled (`#!`) keeps running, with autostart off; a running app gets new settings when it is restarted. |
| editors                     | `add NAME EXECUTABLE`  | Add the app to the proclist, disabled (`#!NAME EXECUTABLE`), and register it, to start by hand or `enable`. It needs `NAME.conf`, and must not be in the proclist already; otherwise nothing changes. `add -- -NAME EXECUTABLE` adds an app without a watchdog. |
| editors                     | `enable NAME...\|--all` | Autostart on: remove the app's `#!` in the proclist, or add it, and register it if needed. `--all` enables every app in the proclist and every registered app. Does not start them. |
| editors                     | `disable NAME...\|--all` | Autostart off: disable the app with `#!` in the proclist. It stays registered, to start by hand. `--all` disables every app. Does not stop them. |
| editors                     | `delete NAME...\|--all` | Remove the app's lines from the proclist, stop it and unregister it. `--all` empties the role's proclist of apps, keeping `#` comments, and unregisters every app. |
| editors                     | `boot on\|off`         | Start the role's enabled apps at boot, or not. Off is the default.        |

For example, to take an app out and put it back later:

```
xctrl delete maths_CO2            # stopped, out of the proclist and unregistered; maths_CO2.conf stays
xctrl add maths_CO2 magAOXMaths   # back as #!maths_CO2 magAOXMaths, registered, autostart off
xctrl start maths_CO2             # run it now
xctrl enable maths_CO2            # and at every startup
```

If the proclist differs from what is registered, e.g. after editing it by hand, `status`, `start`, `startup` and
`restart` warn, naming the apps that are not registered, registered but no longer in the proclist, or have a
different autostart (`#!`), watchdog (`-`) or executable:

```
warning: /opt/MagAOX/config/proclist_CO4.txt differs from what is synced with systemd; not registered: maths_new; registered, no longer in the proclist: maths_old; autostart (#!), watchdog (-) or executable changed: camsci1
warning: xctrl sync applies it, by a user who can edit the proclist
```

`add`, `enable`, `disable` and `delete` change the registration along with the proclist, so they cause no warning. 
An app removed from the proclist while running stays registered until it is stopped; `status` and `sync` say so. 
`sync` also turns its autostart off, so neither `startup` nor a reboot with `boot on` starts it again, and does the
 same for a running app newly disabled (`#!`), without stopping either:

```
warning: camsci1 removed from proclist but still running, so still registered; stop it first (xctrl stop camsci1), then xctrl sync unregisters it
camsci1: autostart turned off (now removed from proclist); however the app is still running.
```

or after `disable` while running:
```
camsci1: autostart turned off (now #! disabled in proclist); however the app is still running.
```

The role comes from `-r ROLE`, `MAGAOX_ROLE`, or `/etc/profile.d/magaox_role.sh`, since `sudo` can drop the
environment.

An operator's `start`, `startup` or `restart`, e.g. by `xsup`, starts an app with its registered settings. If its
`.conf` or `magaox.conf` changed since the last sync, `xctrl` warns, as only an editor's `sync` applies them; an
editor's `start` and `startup` sync first, so the app starts with the new settings:

```
warning: maths_CO4 starts with its synced settings; WatchdogSec=20 from maths_CO4.conf is not synced yet (synced: 10); xctrl sync, by a user who can edit the proclist, applies them
```

`status` also warns about running apps whose settings differ from what `sync` would write now, directive by
directive, naming where each comes from and, for a setting that is ignored now, why. Each difference is shown once,
with every running app it affects; apps that are not running get the synced settings when they start. `status`
also shows the `[systemd]` settings of running apps that are ignored, as `sync` does:

```
warning: MemoryMax=1G from magaox.conf is not synced yet; running with the synced settings: isCO2 maths_a; xctrl sync, then xctrl restart isCO2 maths_a, applies it
warning: WatchdogSec=20 from maths_a.conf is not synced yet (synced: 10); running with the synced settings: maths_a; xctrl sync, then xctrl restart maths_a, applies it
warning: MemoryMax=1G is synced, but no longer set in magaox.conf or the app's .conf; running with the synced settings: isCO2 maths_a; xctrl sync, then xctrl restart isCO2 maths_a, applies it
warning: [systemd] MemoryMax directive for magaox.conf not in systemd-directives allow list; ignoring directive
warning: MemoryMax=1G is synced, but magaox.conf's MemoryMax = 1G is ignored now: not in systemd-directives allow list; running with the synced settings: isCO2 maths_a; xctrl sync, then xctrl restart isCO2 maths_a, applies it
warning: maths_CO4: runs with settings registered before it started (watchdog 2s, now 0.5s); xctrl restart maths_CO4 applies the new ones
```

The last is an app synced again while it kept running: systemd fixes the settings when an app starts, so only a
restart applies them.

The generated files under `/etc/systemd/system` are readable by everyone (files 0644, directories 0755, whatever the
caller's umask), so any user can compare them.

## Other roles on the same host

A host can have the apps of several roles registered, e.g. after switching `MAGAOX_ROLE`. `xctrl` works on one role at
a time, but finds the others from their drop-ins and from systemd's
`magaox-<role>-app@NAME.service` units. `startup`, `start` and `status` warn about apps of another role still running:

```
warning: 4 app(s) of role AOCTest still running: isAOCTest tcsi sysMonAOC labCool; xctrl -r AOCTest shutdown --all stops them
```

`status` also warns about another role whose target starts at boot (`xctrl -r ROLE boot on`) with apps enabled, as
they would start after a reboot alongside, and perhaps in conflict with, the current role's:

```
warning: role AOCTest starts 2 app(s) at boot, which may conflict with role AOC after a reboot; xctrl -r AOCTest boot off stops that
```

`sync` first lists every other role, as a warning if any of its apps are running, then the current one:

```
warning: role AOCTest: 5 app(s) registered, 4 running
Current role: AOC: 0 app(s) registered, 0 running
```

Stopping and disabling are separate: `stop`/`shutdown` never change what starts automatically, and
`enable`/`disable` never start or stop anything. The proclist always says what starts automatically: `enable`,
`disable`, `add` and `delete` edit the app's lines in it, leaving the rest of the file as it is, so the next `sync`
keeps their effect. Editing the proclist by hand and running `xctrl sync` does the same.

An app that stops sending `WATCHDOG=1` within `watchdogSec` gets SIGABRT (with a core dump) and is restarted per its
`restart` policy. `restart NAME` does not restart the INDI server: apps reconnect to it through the control FIFO.

After the start limit (`startLimitBurst` starts within `startLimitIntervalSec`) an app is left failed;
`xctrl start NAME` clears that and tries again.

# Tab completion

`make install` also installs `xctrl.bash-completion` as `/usr/share/bash-completion/completions/xctrl`, and
`/etc/profile.d/xctrl_completion.sh`, which loads it in every bash login shell. So completion works for every user
in a new login shell, e.g. a new ssh session or `su - USER`, whether or not their `~/.bashrc` enables the
`bash-completion` package. In a shell opened before the install, `source /etc/profile.d/xctrl_completion.sh`.
`xctrl <TAB>` lists the commands, and `xctrl start AOC<TAB>` completes, or lists, the registered apps of the role
starting with `AOC`; likewise for every command that takes app names, also after `sudo`. `enable`, `disable` and
`delete` also offer the apps in the proclist, `add` the apps with a `.conf` not yet in the proclist and then the
executables, `boot` on and off, and `-r` the roles with a proclist. The role is the one xctrl would use: `-r`, else
`MAGAOX_ROLE`, else `/etc/profile.d/magaox_role.sh`. Names come from the apps' drop-ins and the proclist, never from
systemd, so completion is instant.

# Output and logging

Messages are colored on a terminal (unless `NO_COLOR` is set): errors red, as `error: ...`, warnings yellow, as
`warning: ...`, the `systemctl` commands xctrl runs blue, as `+ systemctl ...`, and completed changes green.
Errors and warnings go to stderr, the rest to stdout. `-q` (`--quiet`) leaves warnings off the console, showing the
normal output and errors only, e.g. `xctrl status -vq`; warnings are still logged. Single-letter flags combine: `-vq`
is `-v -q`.

The journal lines that `peek` and `inspect` show are colored as `journalctl` colors them, warnings yellow and errors
red, on a terminal and unless `NO_COLOR` is set.

Every command except `status`, `peek` and `inspect` is also logged, with all its messages, timestamped, in the
standard MagAO-X binary log format, so it reads like any app's log:

```
logdump xctrl
```

The log is `/opt/MagAOX/logs/xctrl/YYYY_MM_DD/xctrl_<timestamp>.binlog`: one file per day, appended to by every
run, and a new one if it passes the usual 10 MB. Each run starts with its command line and the user who ran it;
`systemctl`'s own output and each command's success or failure are logged too. xctrl, setuid root, opens the log
as root, so every user's commands go to the same file, which, like its directories, is
owned by `xsup` and group `magaox` and readable by everyone. Entries are standard `text_log`s, written with the
flatlogs format and the `text_log` code and schema generated by the libMagAOX build, so `make` in `utils/xctrl`
needs libMagAOX built first.

# Testing without systemd

`--dry-run` prints the files and `systemctl` commands instead, and logs nothing; with `--top DIR`, xctrl logs under `DIR/logs`, as the calling user. `--sysroot DIR --top DIR` writes the units under
`DIR/etc` from a test MagAO-X tree, reads the allow list from `DIR/etc/magaox`, and runs only the `systemctl --root` commands that work offline (enable,
disable). Check the generated units with
`SYSTEMD_UNIT_PATH=DIR/etc/systemd/system: systemd-analyze verify magaox-<role>-app@NAME.service`.
