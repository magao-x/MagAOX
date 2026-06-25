## `xctrl` vs `resuctrl` — Comparison of Features & Behaviour

## Brief Description

**`xctrl`** manages processes directly by creating and destroying named mux sessions. Each MagAO-X application runs inside its own `tmux` session. `xctrl` communicates with processes by sending keystrokes into their sessions (`Ctrl-C`, `exit`, etc.).

**`resuctrl`** is an interface to a long-lived C++ daemon, `resurrector_indi`. resuctrl sends Unix signals (`SIGUSR1`, `SIGUSR2`) to the daemon, which then launches, monitors, and stops the actual processes. The daemon monitors each process via a hexbeat, a hexadecimal timestamp each process writes periodically into a named FIFO. If a process stops writing hexbeats (crash or hang), the daemon auto-restarts it. `resuctrl` cannot control any individual process unless `resurrector_indi` is already running.

---

## `xctrl`
- **Language** - Python
- Creates/destroys a named tmux session per process
- Doesn't auto-restart process on crash
- Reads `/opt/MagAOX/config/proclist_{role}.txt` directly; never modifies it
- Any operator can attach to a process with `tmux attach -t {name}`
- **Debug mode** with `XCTRL_DEBUG=1`

| Command | Detail |
|---|---|
| **`startup`** | Start named processes (or all) in new tmux sessions |
| **`start`** | Alias for `startup` but always requires at least one process name (cannot target all) |
| **`shutdown`** | Requires `--all` or specific names. Force-kills tmux session if needed |
| **`stop`** | Alias for `shutdown` but always requires at least one process name (cannot target all) |
| **`restart`** | Runs shutdown then startup. Requires `--all` or names |
| **`status`** | Five states: **running**, **not started**, **session exists / process not running**, **dead** (stale PID), **unknown** . Checks tmux session + `/opt/MagAOX/sys/{name}/pid` |
| **`peek`** | Shows status + last 10 log lines. Defaults to all processes if no names given |

---

## `resuctrl`
- **Language** - Bash
- Sends signals to `resurrector_indi` daemon; no tmux
- Auto-restarts hanging / crashed processes
- Uses a writable cached copy of the proclist (separate from the version-controlled base file). `start`/`stop` comment/uncomment entries in this file
- **Setup required** — `resuctrl reset` must be run (as privileged user) before first use

| Command | Detail |
|---|---|
| **`startup` (no args)** | Starts the `resurrector_indi` daemon (which then starts all uncommented processes). Fails if daemon is already running |
| **`startup PROCNAME` / `start PROCNAME`** | Uncomments the entry in the cached proclist, then sends `SIGUSR2` to the daemon to start that process.|
| **`shutdown --all` / `stop --all`** | Sends `SIGUSR1` to the daemon -> daemon kills all processes and exits. |
| **`shutdown PROCNAME` / `stop PROCNAME`** | Comments out the entry in the cached proclist, then sends `SIGUSR2` to daemon to stop that process. |
| **`restart --all / PROCNAME`** | Runs shutdown --all / PROCNAME, sleeps 5 s, runs startup --all / PROCNAME |
| **`status` / `lstatus`** | Five states: **running**, **disabled** (commented out in cached proclist), **no resurrector / not started**, **resurrector exists / not started**, **dead**. `lstatus` tries first to use `lsof` for more accurate FIFO detection; `status` uses `ps x` only |
| **`peek --all / PROCNAME`** | Shows status + last 10 log lines. |
| **`defib PROCNAME`** | Injects an expired hexbeat into the process's FIFO, causing the daemon to treat it as hung and restart it immediately. Force-restart without modifying the proclist |
| **`reset`** | Creates the cached proclist directory, copies the base proclist into it, creates `/opt/MagAOX/sys/{name}/` dirs and hexbeat FIFOs for all processes. **Must be run before first use or whenever proclist is updated** |
| **`ripath`** | Prints the path to the `resurrector_indi` binary |

---

## Key differences

| Command | `xctrl` | `resuctrl` |
|---|---|---|
| `start` vs `startup` | Near-identical; `start` always requires process names | Distinct: `startup` = start daemon; `start PROCNAME` = enable+start process |
| `stop` vs `shutdown` | Near-identical; `stop` always requires process names | Fully identical |
| `startup` (no args) | Acts on all processes | Act on the daemon only which then starts all processes |
| `shutdown --all` kills | All tmux sessions | The daemon (and all its processes) |
| `shutdown PROCNAME` | Doesn't modify proclist | Comments out process in caches proclist |
| `defib` | N/A | Yes |
| `reset` | N/A | Must be run by privileged user before first use or whenever proclist is updated  |
