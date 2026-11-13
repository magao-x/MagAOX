# indiserver chained-server feedback loop

User request: "we have identified a bug in INDI/INDI/indiserver.c where with multiple
indiservers (3 or more) a runaway feedback loop develops. A key symptom is that
`getINDI` never stops. A fix has been developed on the
stefan_schilperoort/dev-resurrector branch. Please analyze and document findings."
Follow-ups: "the problem is happening on current dev on MagAO-X" and "maybe stefan's
branch is a red-herring".

Analysis only: no source changes. All testing used local simulated servers and fake
drivers, never instrument hardware or instrument INDI tunnels.

## Status

- **A dev loop mechanism is found and reproduced** (D1). It needs a driver that sends
  `getProperties` with a name but no device, on two or more servers that are remote
  to each other.
- **Implemented on `jrmales/indiserver-getproperties-loop`** (see "Implementation"):
  F1, F2, F3, the D2 guard, the D3 `ready` flag, and virtualPDU no longer polls.
- **The production culprit app is not yet identified.** With F2, the indiserver log
  now names it (`Driver X: ignoring snoop getProperties without a device`), and with
  F3 a misconfigured MagAOXApp refuses to start and logs the empty key.
- Stefan's branch has the same D1 behavior as dev, so it is not the dev fix. It does
  fix a dev startup segfault (D3) and the old resurrector server's separate loops.
- **2026-11-11: the branch is deployed on AOC, ICC and RTC (rebuilt and restarted),
  and `getINDI` still never stops.** One app had not been rebuilt because it was on a
  different branch. After it was rebuilt, `getINDI` exits normally. So the
  indiserver changes alone were not enough on the instrument, and the app-side
  changes are needed. The app was `accelSim` (`acc000`, a local driver on RTC),
  built from `parkertjohnson1/accelSim`. The checkout on exao2 was at `a1a8fb56`
  with a clean tree; only build products were untracked. At that commit, and at
  every other on the branch, accelSim subscribes only to non-empty devices
  (`camwfs.fps` in `acc000.conf`), has no polling, and never re-sends `def`s. Its
  `MagAOXApp` base (merge-base `cd37915d`, 2026-10-22) already has the retry
  backoff (`1a2d2503`). **The mechanism is unconfirmed.** The untracked binary may
  have predated that commit or been built from local edits, or restarting `acc000`
  may have cleared a bad runtime state. If the symptom returns, `indi_traffic.py`
  should name the device.
- **2026-11-12: root cause found.** accelSim sends `false`/`true` as number
  values, which crashes purepyindi2 clients. They reconnect every 2 s with a
  catch-all `getProperties`. See "Root cause on the instrument".
- **No routing loop is needed for this symptom.** One app that sends a named
  `getProperties` more often than every 2 s makes the source send a `def` each time,
  and that `def` reaches every all-device client on every server. That alone keeps
  `getINDI` from ever going quiet, and the indiserver changes would not help. vpdu0's 5 s poll was under that threshold alone, but it added to the def
  traffic.

## Implementation

Branch `jrmales/indiserver-getproperties-loop`, from `dev`.

- `INDI/INDI/indiserver.c`. Its legacy K&R/tab style was kept; clang-format was not
  run on it because that would rewrite the whole file.
  - **F1:** `q2Drivers` takes the property name. A generic `getProperties` rewritten
    for a remote driver keeps `name`, and a name containing XML quote or markup
    characters is not forwarded.
  - **F2:** a driver's snoop `getProperties` without a device is logged with the
    driver name and dropped instead of fanned out.
  - **D2:** messages without a device other than `getProperties` are not sent to
    remote drivers, so chained connections never register an all-device interest.
    `q2Clients` delivers a device-less message only to clients that asked for every
    device (`findClAllDevices`). A device-less message therefore stays on the server
    where it originated, and is not duplicated once per chained link.
  - **D3:** `DvrInfo::ready` is set by `startDvrThread` after it takes
    `restart_lock`. `q2Drivers` and `q2SnoopingDrivers` skip drivers that are not
    ready.
- **F3:** `MagAOXApp::registerIndiPropertySet` returns -1 and logs for an empty
  device or property name. Through `REG_INDI_SETPROP`, a misconfigured app's
  `appStartup()` now fails. Test: `MagAOXApp INDI SetProperty rejects empty names`.
- **virtualPDU:** the periodic `sendGetPropertySetList( true )` and the
  `pollInterval`/`staleTimeout` receipt-time expiry are removed. Availability is the
  latest `fsm` READY plus a valid channel state. A non-READY `fsm` invalidates the
  source's outlets until both are reported again. The old `[device]` keys are
  ignored with a warning, so the deployed `vpdu0.conf` still loads; they should be
  removed from it. Trade-off: a source that dies without publishing a state change
  is no longer detected by expiry.

### Verification

Simulation (harness `build/branch`):

| Scenario | dev | branch |
|---|---|---|
| startup crashes, `cfg_empty2`, 3-server starts | 8 of 9 | 0 of 10 |
| culprit pair `cfg_empty2`: defs / quiet exit / CPU | 138,830 / never / 282% | 54 / 2.1 s / 0% |
| device-less `<message>` from ICC, copies seen by all-device clients on AOC / RTC / ICC | ~47,000 / – / ~42,000, looping at 36-54% CPU | 0 / 0 / 1 |
| client `<getProperties name='state'/>` on AOC | every remote driver gets a device-wide request and re-sends all its snoops | each remote driver gets a named request |

Unit tests, run against a rebuilt `libMagAOX.a` (it explicitly instantiates
`MagAOXApp`, so a stale archive hides header changes):

- `MagAOXApp_test` passes except `powerOnWait() == 0`, which fails the same way on
  dev (environment).
- `virtualPDU_test` passes.
- 34 suites for apps with config-driven subscriptions: 32 pass. `mcp3208Ctrl_test`
  fails the same 6 assertions on dev, and `modalGainOpt_test` does not finish within
  15 minutes on dev either.

## Topology (from `/opt/MagAOX/config`)

- AOC (`servers=isICC`, `servers=isRTC`): all 96 ICC and 91 RTC local devices are
  remote drivers on AOC.
- RTC: 10 AOC and 7 ICC remote devices. ICC: 17 AOC, 10 RTC, 1 ACC001 remote devices.
- Routing is acyclic for every device. The only multi-hop chains are `ttmpupil` and
  `flipacq` (ICC→AOC→RTC). ICC's `gmtpicos@icc_rtc_indi` points at a device that is
  not local on RTC.

Because routing is acyclic per device, a dev loop has to come from messages that are
not device-addressed.

## D1: device-less getProperties ping-pong (dev)

Line numbers are from dev `INDI/INDI/indiserver.c`.

1. A driver sends a snoop `getProperties` with an empty device. pcf omits empty
   attributes (`IndiXmlParser::createGetProperties`), so a subscription registered
   with device `""` goes out as `<getProperties name="x" version="1.7">`.
   `MagAOXApp::registerIndiPropertySet` does not reject an empty device.
2. `driverStdoutReaderThread` calls `q2Drivers` for it (1086, "force initial
   report"). Because the device is empty, this is treated as a generic
   `getProperties` (`isggp`, 1426).
3. For every remote driver, `q2Drivers` rewrites it to
   `<getProperties version='1.7' device='X' />` (1445-1455). **The `name` is
   dropped**, so a single-property request becomes a device-wide request to every
   remote device on that server.
4. A MagAOXApp that receives a device-wide `getProperties` sends all of its defs and
   calls `sendGetPropertySetList( true )`
   ([MagAOXApp.hpp](../../../libMagAOX/app/MagAOXApp.hpp), `handleGetProperties`).
   That resets every subscription's retry backoff and re-sends every subscription
   immediately, including its own empty-device ones.
5. If an app with an empty-device subscription on server Y is a remote device on
   server X, and X also has one whose app is remote on Y, the two keep triggering
   each other with no backoff. Each lap sends full def dumps of every remote device
   on both servers to every interested client, so `getINDI` never sees 2 s without a
   def. Longer cycles through three servers work the same way.

Why "3 or more": AOC has every ICC/RTC device as a remote, so any ICC/RTC culprit is
triggered by any AOC culprit. Cross-triggering among AOC, ICC and RTC culprits only
needs each culprit to be in another server's remote list. With three
mutually-chained servers there are many such pairings; two one-way servers have none.

A single culprit does not loop. It causes a burst of full dumps on each of its
retries (1, 2, 4 … 60 s) and on every generic `getProperties` from any client.

### Simulation (dev)

Three servers modeled on AOC/RTC/ICC, with fake `MagAOXApp`-style drivers (see the
harness).

| Config | defs seen by one generic client | Quiet exit | indiserver CPU |
|---|---|---|---|
| normal subscriptions | 34 | 2.1 s | 0% |
| `logopt` (ICC) subscribes `("", "nosuch")` | 80, then 26 per retry | 5.2 s | 0% |
| plus `observers` (AOC) subscribes `("", "nosuch2")` | 138,830 in 15 s (7-8k/s) | never | 282% / 163% / 121% |
| same, culprit 2 = `holoop` (RTC, not remote on ICC) | bursts of 12-104 | intermittent | 0% |
| two-culprit case, dev + F1 | 54 | 2.1 s | 0% |

In the looping run, `logopt` received 50,768 device-wide `getProperties` in about
30 s.

### F1: keep the name when rewriting (implemented)

[keep-name-in-generic-getProperties.patch](indiserver-chained-loop-files/keep-name-in-generic-getProperties.patch)
passes `name` into `q2Drivers` and emits `device='X' name='n'` when the original
request had a name. That matches the intent of the rewrite: only stop the request
going around. Remote apps then get a named request for a property they don't have
and answer nothing, so the cascade cannot start. It applies cleanly to dev.

### F2 and F3: further hardening (implemented)

- **F2:** In `driverStdoutReaderThread`, do not fan out a driver's snoop
  `getProperties` that has no device; log it with the driver name. A snoop must
  name a device, and the log identifies culprits directly.
- **F3:** Make `MagAOXApp::registerIndiPropertySet` reject (log and fail) an empty
  device. About 60 config-driven `REG_INDI_SETPROP`/`registerIndiPropertySet` call
  sites in `apps/` and `libMagAOX/app/dev/` have no local empty check. The ones
  checked for running instances (trackers, modalPSDs, dmPokeCenter, closedLoopIndi
  input, psfFit, power management) are defaulted or guarded, but rule- and
  mapping-driven ones (`stateRuleEngine`, `virtualPDU`, `observerCtrl` stream
  writers, `indiTSAccumulator`) depend on config parsing.
- **Empty-name subscriptions** (`device="Q"`, no name) are a related hazard that F1
  does not cover. They send Q a device-wide request, so two apps that subscribe to
  each other that way ping-pong even on one server. F3 could reject an empty name
  as well.

## D2: device-less messages loop (dev and Stefan)

`q2Clients` sends any message without a device to every client (`findClDevice`:
`!dev[0]`, 1867), including chained servers, which send it on to all of theirs.
With `cfg_univ.json`, one device-less `<message>` circulates indefinitely at about
3,000/s and 55-120% CPU per server. These are not `def*`, so `getINDI` still exits.
No in-tree C++ app emits one (no `IDMessage(NULL, ...)`). Fixed as described under
"Implementation".

### Known protocol limitation

INDI routes only by device name. Messages carry no origin, server identity, or hop
count, and server chaining is not part of the protocol: a peer indiserver connects
as an ordinary client and is served by the other side as an ordinary driver. Nothing
distinguishes a forwarded message from a new one. Routing is loop-free only while
every message names a device and each device has one acyclic path. Device-less
elements break that:

- **Generic `getProperties`:** rewritten per remote device, now keeping its name (F1).
- **General `<message>` (no device):** there is no routing key, so the only safe
  choices are "local only" or "one hop, duplicated once per chained connection" (96
  copies on AOC for an ICC message). This branch chose local only: a general message
  reaches the all-device clients of the server whose driver emitted it, and GUIs on
  other servers do not see it. MagAO-X does not currently use general messages.

A proper fix would let servers recognize peer connections. The version handshake on
Stefan's branch (`getProperties device='indiserver' name='version'`) could mark a
connection as a peer, so general messages could be forwarded exactly once to each
server, with neither loops nor duplicates. Revisit this if general messages are ever
needed.

## D3: dev startup segfault (fixed here and on Stefan's branch)

Separate from the loop, but it causes reconnect storms. The dev `indiserver` crashed
at startup with `SIGSEGV`: 4 cores in about 10 three-server starts with normal
subscriptions, and 8 of 9 starts with the looping `cfg_empty2.json` (more peer
`getProperties` during startup). None crashed under gdb because of the timing
change. Every core has the same stack:

```text
chkFQ (q=0x0) fq.c:129
pushFQ (q=0x0) fq.c:72
pushMsg (dp=<remote driver, pid 0, msgq NULL>) indiserver.c:1690
q2Drivers (dev="pdu1"|"ttmpupil"|"camwfs", roottag="getProperties") indiserver.c:1462
clientReaderThread indiserver.c:929
```

`main()` accepts clients while `initDvr()` is still creating driver threads. Until a
driver's `startDvrThread` takes the write lock, `tryrdlock` succeeds, `dp->dev` is
empty (so it matches every device), and `dp->msgq` is NULL. This is most likely when
one server restarts while its peers are up. It is not the libc crash in
`doc/incidents/2026-03-03-indiserver-segfault.md`. Stefan's branch fixes it with
`DvrInfo::ready`, set after the write lock and checked in
`q2Drivers`/`q2SnoopingDrivers`; it could be ported to dev on its own.

## Root cause on the instrument: accelSim `dm_status` crashes purepyindi2 clients

2026-11-12. The deployed indiserver fixes improved things. With `acc000`
(`accelSim`) stopped, a clean restart of all three servers left `getINDI` working.
Starting `acc000` reliably made the next `getINDI` run forever. Data:
[indi_traffic_py_results_20261112_1400.txt](indiserver-chained-loop-files/indi_traffic_py_results_20261112_1400.txt).

- On AOC, after an `isAOC` restart, every device on all three servers sent a full
  def dump about every 2 s (hogopt 26 dumps in 27 s), plus named re-subscription
  defs (`camwfs.fps` 322). On RTC, `acc000` alone was dumped every 2 s.
  `acc000.fps` came three times per dump, because `acc000-cbuff` and `acc000-psds`
  re-subscribe whenever they receive a device-wide request.
- `accelSim::updateDMINDI` assigns `bool`s to the number elements
  `dm_status.ready` and `dm_status.active`. pcf formats values with
  `std::boolalpha`, so acc000 sends `<oneNumber name="ready">false</oneNumber>` in
  every def and in a `set` on every app loop.
- purepyindi2 `IndiStreamParser.parse` catches only `expat.ExpatError`. The
  `float("false")` in `OneNumber.value_from_text` raises `ValueError` out of the
  expat callback (reproduced locally against the installed purepyindi2).
- In the asyncio client (`AsyncINDIClient.run`), that exception ends the
  connection. It reconnects after `RECONNECTION_DELAY_SEC = 2` and re-sends a
  catch-all `getProperties`. Every MagAOXApp on every server then dumps and
  re-subscribes, the dumps include `acc000.dm_status`, and the cycle repeats every
  2 s. No routing loop is involved. The threaded `IndiTcpClientConnection` reader
  thread dies on the same error, leaving that client deaf instead.
- `isAOC` was initially unresponsive even to the profiler's own request. That was
  not diagnosed. Next time, get a stack with
  `gdb -p <pid> -batch -ex "thread apply all bt"`.

Fixes:

1. **pcf (implemented on `jrmales/indi-bool-numeric`):** `IndiElement` writes a
   `bool` as `1`/`0` through every setter, using the new
   `pcf::formatIndiValue`. `get<bool>()`/`getValue<bool>()` read through the new
   `pcf::parseIndiValue`, which accepts `true`/`false` as well as any number
   (nonzero is true), so values from older writers still read correctly.
   `indi::updateIfChanged`/`updatesIfChanged` compare using `formatIndiValue`, so
   they agree with the stored strings. Any app that assigns a `bool` to an
   element, including accelSim's `dm_status`, sends valid numbers once rebuilt;
   accelSim itself needs no change.
   - Compatibility: a binary built before this change that reads a `bool` with
     `get<bool>()` would read a new `1` as `false`. No C++ in this repo, `rtimv`,
     `mxqtuio` or `milk-xwcl` reads an element with `get<bool>`. Text elements
     assigned a `bool` now carry `1`/`0` instead of `true`/`false`.
2. **purepyindi2:** `parse()` should catch any exception, log the device and
   property, drop that element or property, and keep the connection. One bad value
   should never disconnect a client.

## After deployment: `getINDI` still never stops

`getINDI` exits only after 2 s without a top-level `def*` (`getINDI.c`, `alarm` reset
on `def`). Every device-wide `getProperties` that reaches a MagAOXApp makes it send all
its defs and re-send every subscription (`handleGetProperties`). Each named
subscription request then makes the source app send one def, which reaches every
all-device client on every server. Repeated device-wide requests therefore keep
`getINDI` alive whether or not there is a routing loop. Candidates the branch does
not address:

- **H1, not deployed everywhere:** the running `indiserver` is not the new binary, or
  other chained servers still run old binaries: `isROC` and `isCOC` chain all devices
  of AOC, ICC and RTC, and `isACC001` is a leaf under ICC.
- **H2, reconnect storm:** a client or chained connection more than 50 MB behind
  (`DEFMAXQSIZ`) is shut down (`Client N: ... bytes behind, shutting down`, or
  `Driver X: ... bytes behind, restarting` for a remote driver). A remote driver
  reconnects at once and sends `<getProperties device='X'/>`, so X dumps and
  re-subscribes. `isROC`/`isCOC` hold one connection per remote device, over slow
  links. The bursts make other connections fall behind, so the storm can sustain
  itself.
- **H3, crash-looping app:** each restart sends a full def dump and all
  subscriptions (`Driver X: Exit status`/`Exit signal`, `restart #N`).
- **H4, client sending generic `getProperties` repeatedly:** each one makes every app
  on all three servers dump and re-subscribe. For example, a purepyindi2 client
  sends a catch-all `getProperties` on every reconnect.
- **H6, a driver or client polling named `getProperties` faster than every 2 s:**
  each poll draws one `def`, broadcast to every all-device client. This matches the
  accelSim fix. In `indi_traffic.py` it shows up as one key with a steady `def` rate.
- **H5, no loop:** an app re-sends `def`s as normal operation, for example a
  property whose elements change. The traffic would be one or a few keys, not full
  dumps.

`indi_traffic.py` (in the harness) separates these. It counts `def` vectors per
second and per device after the reply to its own request, with an approximate
number of full dumps per device. In simulation, the branch is quiet at 2.1 s with
no defs after 3 s. Dev with `cfg_empty2.json` gives about 3,500 def vectors/s from
the looping devices and never goes quiet. Note that `client.py` counts member
elements (`defText` etc.) as well as vectors, so its def counts elsewhere in this
document are element counts.

Data requested from the instrument, from each of AOC, ICC and RTC:

1. Confirm the binary: whether `strings` of the installed `indiserver` contains
   `ignoring snoop getProperties`, the binary mtime, and the `indiserver` process
   start time. Also check whether `isROC`/`isCOC`/`isACC001` are running and which
   build they run.
2. `indi_traffic.py 7624 30`.
3. Counts of `bytes behind`, `restart #`, `now connected`, `Exit status`,
   `Exit signal` and `ignoring snoop` in the last `logdump` files of the indiserver
   app.
4. `getProperties` arriving over TCP, with source ports, from `tcpdump`, mapped to
   processes with `ss -tnp`.

## Finding the culprit on the instrument (for the user to run)

- Look for unresolved empty-device subscriptions in app logs: grep `logdump` output
  for `INDI property still unresolved after retry backoff: .` (a key that starts
  with `.` has an empty device). While the loop is running, backoff keeps resetting,
  so look at quiet periods or right after app start.
- Briefly raise one indiserver's verbosity. At `-v`, dev logs `Driver X: Loop caught,
  adding X to generic getProperties` for every rewrite, so the line rate shows the
  loop. At `-vv`/`-vvv`, `from Driver <name>: read <getProperties device='' name='...'>`
  names the culprit driver directly.
- While `getINDI` is looping, run `getINDI -t 0 | head -n 200000 | sort | uniq -c |
  sort -rn | head`. Repeated full dumps of many devices point to D1. One or two
  repeating properties point to something else, such as the March pattern of
  `pdu1.camwfs` with a vpdu property.

## Secondary: the old resurrector indiserver

The indilib-lineage `indiserver.c` on `origin/dev-resurrector` (last version
`dac13a5f`) loops much harder. Line numbers below are from that file.

- **M1:** generic `getProperties` is forwarded unchanged to one remote driver per
  host:port (`q2RDrivers`, 1781-1807). It goes around forever, and plain `getINDI`
  starts it.
- **M2:** snoop requests are pushed upstream to chained clients (`q2Servers`,
  2305-2311), and `def`/`set` from clients are forwarded to snooping drivers
  including remote ones (2045-2048). Interest lists widen on every message
  (2012-2015, 2334-2335).

In simulation it sent 1,000-240,000 defs/s and never went quiet, even with two
mutually-chained servers or no snoops. One old server in a ring of threaded servers
gave 2.5-3× normal def traffic. Stefan's `077c0eb2` replaces this server with dev's
threaded routing, so his branch fixes M1 and M2, but not D1 or D2.

## Harness

[indiserver-chained-loop-files/](indiserver-chained-loop-files/README.md):
`build.sh` builds `dev`, `devfix` (F1), `stefan` and `oldres`. The launchers cover
fork/exec, FIFO (`-F`), old-server, mixed and exit-capture setups. The configs
include `cfg_empty1.json` (one culprit) and `cfg_empty2.json` (the looping pair).

## Follow-ups

- Identify the production culprit(s) with the instrument checks above. After
  deployment, the indiserver logs and any app that refuses to start (F3) name them.
- Remove the obsolete `[device] pollInterval`/`staleTimeout` keys from `vpdu0.conf`
  in the config repository.
- Deploy every indiserver together. The D2 guard relies on peers never sending
  device-less messages to remote drivers.
- If Stefan's branch is kept, merge `dev` into it first: it is far behind dev, and a
  direct diff deletes most recent dev work.
