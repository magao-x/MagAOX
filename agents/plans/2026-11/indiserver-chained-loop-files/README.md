# indiserver chained-loop simulation harness

Reproduces the findings in [../indiserver-chained-loop.md](../indiserver-chained-loop.md)
on one workstation. Everything runs on `127.0.0.1` ports 27624 (A, "AOC"),
27625 (R, "RTC") and 27626 (I, "ICC") with fake drivers only: it never touches
instrument hardware or instrument INDI tunnels.

## Files

- `build.sh`: builds `build/{dev,devfix,stefan,oldres}/INDI/INDI/indiserver` from git refs
  (set `REPO` if this directory is not inside the MagAOX checkout). `devfix` is dev plus
  `keep-name-in-generic-getProperties.patch`; `branch` is `jrmales/indiserver-getproperties-loop`.
- `fakedrv.py`: fake driver that mimics `MagAOXApp` INDI behavior. A device-wide or
  generic `getProperties` sends all defs and re-sends every snoop `getProperties`.
  A named `getProperties` sends one def. Missing snoops are retried with 1 to 60 s
  backoff. The device name is `basename(argv[0])`, through the `drv/` symlinks.
- `client.py port seconds [device]`: `getINDI`-like counter. It prints `def`/`set`
  per second and when a `getINDI` with a 2 s quiet timeout would have exited.
- `indi_traffic.py [host:]port [seconds] [device]`: read-only profiler for a live
  server (sends one `getProperties`, like `getINDI`). It prints `def`/`set`/`del`/
  `message` vectors per second with the top `def` devices each second. For t ≥ 3 s it
  prints defs per device, distinct properties and approximate full dumps, plus the
  top `def` and `set` keys. `client.py` counts member elements too.
- `cfg.json`: devices, properties and snoops (MagAO-X-like cross-server snoops).
  `cfg_nosnoop.json` has the snoops removed. `cfg_univ.json` makes `loloop` emit one
  device-less `<message>` 8 s after start.
  `cfg_empty1.json` gives `logopt` (ICC) an empty-device subscription.
  `cfg_empty2.json` also gives one to `observers` (AOC), which makes dev loop.
- `run.sh <dev|stefan> <out>`: three threaded servers with fork/exec local drivers.
  Stop with `kill.sh <out>`.
- `run_fifo.sh <dev|stefan> <out>`: three Stefan-branch servers in `-F` FIFO-driver
  mode. Stop with `kill_old.sh <out>`.
- `run_old.sh <out>`: three old resurrector servers (FIFO drivers in `<out>`).
  `TOPO=3|3nochain|2|2oneway`. Stop with `kill_old.sh <out>`.
- `run_mixed.sh <dev|stefan> <out>`: A and I run the threaded server, R runs the old
  resurrector. Stop with `kill_old.sh <out>`.
- `restartR.sh <dev|stefan> <out>`: restarts server R in a `run.sh` setup.
- `run_exit.sh <dev|stefan> <out>`: same as `run.sh`, but appends each server's exit status
  to `<out>/exits`. Use it to catch the dev startup segfault (exit 139).

## Typical use

```bash
./build.sh
O=$(mktemp -d)
./run_old.sh $O; sleep 20; python3 client.py 27624 15; ./kill_old.sh $O
O=$(mktemp -d)
./run_fifo.sh stefan $O; sleep 10; python3 client.py 27624 15; ./kill_old.sh $O
```

To reproduce the dev loop and its fix:

```bash
O=$(mktemp -d)
SIMCFG=$PWD/cfg_empty2.json ./run_exit.sh dev $O; sleep 15; python3 client.py 27624 15; ./kill.sh $O
O=$(mktemp -d)
SIMCFG=$PWD/cfg_empty2.json ./run_exit.sh devfix $O; sleep 15; python3 client.py 27624 15; ./kill.sh $O
```

Prefix a run with `SIMCFG=$PWD/cfg_nosnoop.json` or `SIMCFG=$PWD/cfg_univ.json` to
switch configurations.
