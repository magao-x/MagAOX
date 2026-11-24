Split elliptecCtrl into elliptecLowLevel (bus owner) and elliptecCtrl (per-mount client), modeled on zaberLowLevel / zaberCtrl.

Motivation (hardware, 2026-11-19): two ELL21 mounts (stageqwprot1 at address 1, stageqwprot2 at address 2) share one Elliptec bus behind one USB adapter (FTDI DP06TVFC). Running one elliptecCtrl per mount put two processes on the same tty; each read stole the other's bytes, producing truncated frames (e.g. `2IN15121000702000`, parsed as 7348224 pulses/rev) and moves to wrong targets. The bus must have a single owner.

Reference design: `apps/zaberLowLevel` owns the port and PDU power, configures one section per stage, and publishes shared INDI properties with one element per stage (`curr_state`, `curr_pos`, commands `tgt_pos`, `req_home`, `req_halt`). `apps/zaberCtrl` has no port or power management, subscribes to the low-level properties filtered by its `stage.stageName`, maps `curr_state` to its FSM, sends commands with `sendNewProperty`, and owns units, presets, `stdMotionStage`, `powerOnHome`/`homePreset`, and telemetry.

Design decisions:

1. Responsibility split (agreed 2026-11-19):
   - `elliptecLowLevel` (one per USB port): serial port with exclusive lock, PDU power, frame reader and per-address routing, per-address connect/retry/`in` validation (model, pulses/rev), round-robin `gp`/`gs` polling of every configured address, per-address pending operation and target, completion by position, error latching, motion timeout, velocity floor, maintenance commands (`optimize`/`save`) behind `enableMaintenance`. Publishes per stage `curr_state`, `curr_pos` (degrees, wrapped), error text; accepts per stage `tgt_pos` (degrees), `req_home`, `req_stop`, `req_rel`.
   - `elliptecCtrl` (one per mount): no port, no power management. Subscribes to the low-level properties for its stage, maps `curr_state` to its FSM, sends commands via `sendNewProperty`. Owns `stdMotionStage` presets/names, `presetTolerance`, `absDeg`/`relDeg`/`relMove`, `homeOffset`, `powerOnHome`, telemetry.
   - The protocol code already debugged in elliptecCtrl (frame reader, pending tracking, completion by position, simulated-ELL21 tests) moves into elliptecLowLevel with per-address state.

2. elliptecLowLevel configuration (agreed 2026-11-19):
   - Port discovery via `tty::usbDevice` (`[usb] idVendor, idProduct, serial, baud`), as zaberLowLevel does; PDU power via the standard `[power]` section.
   - One section per mount, named by the stage name used in the shared properties; a section is a mount if it has an `address` key (hex nibble 0-F).
   - Optional per-stage `serial`: the device serial from the `in` reply; on mismatch the stage goes to NODEVICE and is never commanded.
   - Per-stage `velPercent`, applied with `sv` at connect, still raised to the model minimum.
   - Bus-wide settings live once in the low-level config: `serial.trace`, `serial.readTimeoutMs`, `comm.maxPollMisses`, `motion.moveTimeoutSec`, `motion.optimizeTimeoutSec`, `stage.enableMaintenance`.
   - Example (qwp bus):
     ```ini
     [usb]
     idVendor=0403
     idProduct=6015
     serial=DP06TVFC
     baud=9600

     [power]
     device=pdu2
     channel=stageqwprot

     [stageqwprot1]
     address=1
     serial=12100017
     velPercent=80

     [stageqwprot2]
     address=2
     serial=12100070
     velPercent=80
     ```

3. Shared INDI interface on elliptecLowLevel, one element per stage (agreed 2026-11-19):
   | Property | Type | Direction | Meaning |
   |---|---|---|---|
   | `curr_state` | Text | low -> ctrl | `POWEROFF`, `POWERON`, `NODEVICE`, `NOTCONNECTED`, `NOTHOMED`, `HOMING`, `OPERATING`, `READY` |
   | `curr_pos` | Number | low -> ctrl | position in degrees, wrapped to [0, 360) |
   | `tgt_pos` | Number | ctrl -> low | absolute target in degrees; low level sends `ma` and tracks completion |
   | `req_home` | Switch | ctrl -> low | home this stage |
   | `req_halt` | Switch | ctrl -> low | send `st`; unsupported stop reported honestly |
   | `last_error` | Text | low -> ctrl | why the last operation failed; cleared by the next command |
   | `velocity` | Number | operator -> low | per-stage velocity percent, model minimum enforced |
   | `req_optimize`, `req_save` | Switch | operator -> low | maintenance, only with `stage.enableMaintenance` |
   - No relative-move command: elliptecCtrl converts `relMove` to an absolute `tgt_pos` (current + step, wrapped).
   - Velocity and maintenance live on elliptecLowLevel (device-level, like zaber's `knob_enable`/`led_enable`); elliptecCtrl drops its `velocity`, `optimize`, `save` properties.
   - elliptecLowLevel owns PDU power and publishes `POWEROFF` for every stage while the bus is off; elliptecCtrl disables its own power management, as zaberCtrl does.
   - `homeOffset` stays in elliptecCtrl (like zaberCtrl's `homePreset`): on HOMING -> READY it sends the offset as `tgt_pos` and stays HOMING until arrival.

4. Bus sharing and failure isolation (agreed 2026-11-19):
   - Each loop polls `gp`/`gs` for every configured address in turn. Commands from elliptecCtrl are sent from the INDI callback under `m_indiMutex`, without waiting, between polls.
   - Queries to a stage with a pending motion use a short timeout (`serial.busyTimeoutMs`, default 200 ms) because silence is expected; idle stages use `serial.readTimeoutMs` (1000 ms). Completion still requires a fully answered poll.
   - Per-stage miss budget and motion timeout. A stage that exhausts its budget goes NOTCONNECTED and is re-identified with `in` on later loops; other stages are unaffected. Only a hard port error (unplug, read/write error) or all stages failing closes and reopens the port.
   - After power-on each address is identified independently with the retry-until-`in` logic, so a slow-booting mount does not block the other.

5. Organization and delivery (agreed 2026-11-19):
   - New `apps/elliptecLowLevel/`: `elliptecDevice.hpp` (per-address identification, pending operation and target, completion by position, error latching, miss counting; no I/O, unit-testable like `zaberStage.hpp`), `elliptecLowLevel.hpp`/`.cpp` (header-only app plus main: port via `tty::usbDevice`, exclusive lock, power, round-robin polling, shared properties), tests `elliptecDevice_test.cpp` and `elliptecLowLevel_test.cpp` (simulated multi-address ELL21 bus on one PTY; the 11 existing scenarios plus two-mount scenarios).
   - `elliptecCtrl` keeps its name and becomes an INDI client of the low level (serial code removed), tested zaberCtrl-style by injecting `curr_state`/`curr_pos` and checking the commands sent.
   - Low-level instance name `elliptecqwp` (one per bus); each elliptecCtrl sets `stage.lowLevelName=elliptecqwp`.
   - Continue on `jrmales/elliptecCtrl`. Commits: (1) elliptecDevice + elliptecLowLevel with tests, (2) elliptecCtrl client rewrite with tests, (3) docs + plan, (4) formatting if needed.
   - Deployment on exao3: new `elliptecqwp.conf`; trimmed `stageqwprot1/2.conf` (drop `port`, `address`, `[power]`, `serial.*`, `velPercent`; add `stage.lowLevelName`; keep `presets`, `powerOnHome`, `homeOffset`); add `elliptecqwp` to the indiserver driver list and proclist ahead of the stages. Config files to be provided to the user.

Implementation status (2026-11-19):
- `apps/elliptecLowLevel/elliptecDevice.hpp`: per-address identification (strict 30-hex-digit `in` validation, serial check), position/status application, pending operation with target, completion by position (idle + still for two polls at the target, >= 0.5 s after the command), error latching, missed-reply accounting (tolerated during an operation until its timeout; budgeted when idle, then Lost), state string. No I/O.
- `apps/elliptecLowLevel/elliptecLowLevel.hpp`: `tty::usbDevice` lookup (skipped when no USB ids are set, as in tests), `flock` on the port, frame reader with per-address routing, per-device identify/poll each loop with `serial.busyTimeoutMs` for devices with an operation in progress, shared INDI properties, power handling via `powerOnWaitElapsed()`, and a port reopen after `s_maxQuietLoops` loops with no device identified.
- `apps/elliptecCtrl/elliptecCtrl.hpp`: rewritten as a client. Power management disabled; state from `curr_state`; commands via `tgt_pos`/`req_home`/`req_halt`; `relMove` -> absolute target; post-home offset sent on HOMING -> READY and completed only once `curr_pos` is at the offset; power-on home sent once per POWEROFF.
- Tests: `elliptecDevice_test` (unit), `elliptecLowLevel_test` (simulated multi-address bus in `tests/fakeElliptecBus.hpp`: two mounts, hardware quirks, isolation of a silent mount, dropout and re-identification, serial mismatch, boot retry and truncated `in`, device errors, motion timeout, halt/velocity/maintenance, power cycle, port lock, configuration), `elliptecCtrl_test` (client, with `ELLIPTECCTRL_TEST_NOSEND` recording commands in `m_lastCommand` instead of contacting an indiserver). The old single-process test files were removed.
- Bugs found by the new tests and fixed: text INDI values read with `get<std::string>()` stop at the first space (now `getValue()`); `REG_INDI_NEWPROP_NOCB` recreates the property, so elements must be added after registration (status/stageNamePos); a late `GS03` reply to a stop could be latched as a motion error.
- Both apps added to `apps_icc` and `all_buildable_apps` in the top-level Makefile.
- Test-build note: `tests/Makefile.one` does not track header dependencies, so a test must be rebuilt (delete its `.o`) after header-only changes.
