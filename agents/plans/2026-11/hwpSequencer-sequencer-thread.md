# hwpSequencer: single-owner sequencer thread and audit fixes

# Task Description

Date: 2026-11-18. Branch: `ml/hwpSequencer` (based on `dev` at `ccc26f4f`;
plan written at `eb5e10b7` plus the user's uncommitted `appShutdown` reorder).

## Problem Statement

An audit of `apps/hwpSequencer` against libMagAOX and similar apps
(`adcTracker`, `kTracker`, `dmPokeCenter`, `ttmModulator`) found that the
sequence is driven by three threads that coordinate only through flags. That
causes a shutdown deadlock, a stop-vs-step race that can leave the camera
trigger Off while the app reports READY, waits that never time out, and a
sequencer thread that exits silently on error.

User request: "prepare your changes in a plan under agents/plans/2026-11/ and
grill me on the plan." The user will write the code; this plan is the design
record and checklist.

## Discussion

The app synchronizes HWP moves with camera frames:

1. Turn the fxngen trigger channel Off (`fxngensync.C2outp`) and wait until it
   confirms Off.
2. Wait for the frame that was in flight to arrive in the shmim
   (`dev::shmimMonitor`, `processImage` posts `m_smSemaphore`).
3. Command `hwptrack.hwp_position.target` and wait until `current` is within
   tolerance.
4. Turn the trigger On, wait for confirmation, and on the first step turn
   `observers.obs_on` On.
5. Dwell for `timePerPos`, then go to the next position. After each pass
   through all positions, end if `lastCycle` is set or `numCycles` is reached.

Reference patterns in this repo:

- Worker thread woken by a semaphore with a stop flag: `dmPokeCenter`
  (`m_wfsSemaphore`, `m_stopMeasurement`).
- Waiting for remote INDI values with a timeout: `ttmModulator::waitValue`
  (`apps/ttmModulator/ttmModulator.hpp:504`).
- `m_indiMutex` snapshot in `appLogic` and locked updates in callbacks:
  `adcTracker` (`apps/adcTracker/adcTracker.hpp:569`, `:805`).
- Thread-alive check: `shmimMonitor::appLogic`
  (`libMagAOX/app/dev/shmimMonitor.hpp:689`).
- Sending to other devices: `sendNewProperty(prop, el, val)` (copies the
  property) and `sendNewStandardIndiToggle(device, property, onoff)`.

## Requirements

See "Requirements" under Agent Findings and Plan.

## Tests and Metrics

See "Tests" under Agent Findings and Plan.

## Scope and Caveats

- Follow [AGENTS.md](../../../AGENTS.md): header-only app (rule 21), full
  changed-file documentation pass (rule 16), braces and spacing (rule 26), INDI
  subscription rules (rule 27), `{ //mutex scope` (rule 13), separate
  functional / plan / documentation / formatting commits (rules 22–23), and the
  application unit-test Doxygen conventions (rules 19–20).
- Work only on a workstation. Agents must not run `getINDI`/`setINDI`, start
  apps, or attach to streams on instrument computers, including through a
  tunnel. All instrument validation is run by the user, who reports results
  here.
- The user writes the implementation. The agent keeps this plan current and
  may review diffs on request.
- Out of scope: changing `siglentSDG`, `observerCtrl`, `shmimMonitor`, or
  camera apps; HWP tracking behavior in `hwpTracker` beyond the registration
  fix already committed (`eb5e10b7`).

# Instructions to Agent

Analyze the above task and create a plan to implement a solution. Document
findings below under "Agent Findings and Plan". Keep this document up to date
as work proceeds. Review AGENTS.md. Do not begin implementation until the user
has reviewed the plan and answered the questions.

# Agent Findings and Plan

## Task Summary

Restructure `hwpSequencer` so one thread owns the whole sequence and every
wait is bounded, then clean up callbacks, configuration, logging, docs and
tests. Audit item numbers below refer to the review given in the session that
produced this plan.

## Findings

### F1. Three threads drive the sequence (audit 3, 6, 9)

`appLogic` advances index/cycle and may call `stopSequencing()`; INDI callbacks
call `startSequencing()`/`stopSequencing()`; the sequencer thread runs
`doHwpAction()`. `stopSequencing()` and `doHwpAction()` both send fxngen
commands, both `sem_wait` on `m_smSemaphore` and both command the HWP, with no
ordering between them. A stop during a step can leave fxngen Off after the app
returns to READY. Stepping driven by `appLogic` adds up to `loopPause` per
position.

### F2. Shutdown can hang, and the uncommitted reorder does not fully fix it (audit 2)

The working tree moves `SHMIMMONITOR_APP_SHUTDOWN` to the end of
`appShutdown()`. However, the shmimMonitor thread loops on
`derived().shutdown() == 0` (`shmimMonitor.hpp:744`, `:960`) and returns on
its own once the shutdown flag is set, which happens before `appShutdown()` is
called. So `stopSequencing()`'s untimed `sem_wait` in `appShutdown()` can still
wait forever. Any frame wait reachable during shutdown must be timed and must
check `shutdown()`.

### F3. shmimMonitor only runs in OPERATING

`shmimMonitor::m_targetState` defaults to `stateCodes::OPERATING`
(`shmimMonitor.hpp:165`). The monitor detaches when the app leaves OPERATING
and needs up to ~1 s (its `sleep(1)` poll) plus open time to attach after
entering it. Consequences:

- The app must stay OPERATING through every frame wait, including the final
  wait in cleanup. Set READY/ERROR only after the last frame wait.
- The first step's frame wait may time out because the monitor has not
  attached yet. That is harmless if a timeout is treated as "no frame in
  flight" (D4).
- While idle (READY), `processImage` is not called, so `m_smSemaphore` does
  not accumulate.

### F4. Waits without timeouts (audit 4)

`while (m_fxngenOutp)` / `while (!m_fxngenOutp)` and both `sem_wait` calls
have no timeout or shutdown check. `XWC_SEM_TIMEDWAIT_LOOP` is not a fix here:
it `continue`s on `ETIMEDOUT`, and with a deadline computed outside the loop it
spins once the deadline passes.

### F5. Thread exits silently on error (audit 5)

`doHwpAction()` failure returns from `sequencerThreadExec()`. fxngen may be
Off, `m_sequencing` stays true, and nothing checks the thread.

### F6. Dwell is not interruptible (audit 8)

`mx::sys::microSleep(m_timePerPos*1e6)` (up to 600 s). It probably uses
`sleep_for`, which resumes after `SIGUSR1` (not verified). `sem_timedwait`
always returns `EINTR` on a signal.

### F7. Frame-boundary ordering (part of audit 4)

Current order: flush, trigger Off, confirm, wait one frame. With fast
exposures, frames completing during the Off round-trip satisfy the wait while a
later frame may still be exposing. If the trigger stops between frames, no
frame comes and the wait never ends.

### F8. HWP position may never have been received (audit 10)

`m_reportedHwpPos` starts at 0, so a first move to 0° passes immediately even
if `hwptrack` has never published `hwp_position`.

### F9. Smaller items (audit 7, 11–22)

`appLogic` early return skips `SHMIMMONITOR_APP_LOGIC`/`UPDATE_INDI`;
unchecked `sendNewProperty` returns and member properties modified from
several threads; callback input errors logged as `software_error`; `std::cerr`
and `SEMVALUE` debug output; redundant `getName()` checks after
`INDI_VALIDATE_CALLBACK_PROPS`; hard-coded positions/tolerance/timeouts;
hand-built read-only properties; no telemetry; copy-paste docs; template test
file; formatting.

## Key Assumptions

- A1. Only one camera stream gates the sequence (the configured
  `shmimMonitor.shmimName`).
- A2. **Verified:** `siglentSDG` publishes `C2outp.value` only from its
  hardware readback (`siglentSDG.hpp:1488–1495`); `changeOutp()` writes the
  command but does not update the property. So confirmation reflects the
  hardware state, with latency up to one siglentSDG poll.
- A3. `hwptrack.hwp_position.current` reports the actual HWP angle in the same
  frame as `target`, with no 0/360 wrap within the configured positions.
- A4. The dwell starts when the trigger is confirmed On (current behavior),
  so per-position time is `timePerPos` plus move overhead.
- A5. Nothing in this repo outside `apps/hwpSequencer` uses its INDI property
  names (checked 2026-11-18), so renaming is free inside the repo. External
  scripts or notebooks: to confirm (D8).

## Requirements

R1. Exactly one thread (the sequencer thread) sends fxngen, HWP and `obs_on`
    commands and waits on `m_smSemaphore`.

R2. INDI callbacks only validate input and set request flags/atomics, then
    wake the sequencer thread. They never block.

R3. Every wait is bounded and checks `shutdown()`: frame waits, fxngen
    confirmation, HWP position, dwell, and the idle wait.

R4. All endings (completed cycles, stop request, error, shutdown) go through
    one cleanup routine that leaves the trigger On and `obs_on` Off.

R5. On error, the sequence ends through cleanup, the error is logged, and the
    FSM shows it (D6). The sequencer thread stays alive and can start a new
    sequence.

R6. The app stays OPERATING for the whole sequence, including cleanup's frame
    wait (F3).

R7. `appShutdown()` wakes and joins the sequencer thread (cleanup runs, with
    short timeouts) before `SHMIMMONITOR_APP_SHUTDOWN`.

R8. `appLogic` always runs `SHMIMMONITOR_APP_LOGIC` and
    `SHMIMMONITOR_UPDATE_INDI`, and checks the sequencer thread is alive.

R9. `canStart()` refuses (warning log, `sequence` switch back to Off) unless:
    `hwp_position`, `C2outp`, `C2freq` and `obs_on` have each been received at
    least once (rule 27: keep last value; only "never received" blocks);
    `C2freq > 0`; `obs_on` is Off (D12); `time_per_pos ≥ 1/C2freq` (D15). The
    trigger may be On or Off at start.

R10. The positions list is snapshotted at start. `timePerPos` and `numCycles`
     may change while sequencing and apply as described in D9.

R11. Positions, HWP tolerance, HWP/fxngen timeouts, poll interval and the
     frame-timeout margin are configurable, with current values as defaults.

R12. Behavior visible to operators is logged with `log<text_log>` at an
     appropriate priority; no `std::cerr` debug output remains.

## Design

### Threads and ownership

| Thread | Does | Never does |
| --- | --- | --- |
| Main (`appLogic`) | shmimMonitor macros, sequencer-alive check, INDI housekeeping | sequence logic, blocking waits |
| INDI callbacks | validate input; set request flags (`m_startRequested`, `m_stopRequested`, `m_abortRequested`), `m_lastCycle`, `m_timePerPos`, `m_numCycles`; post `m_wakeSemaphore` (abort also posts `m_smSemaphore`) | fxngen/HWP/obs commands, semaphore waits |
| shmimMonitor | `processImage` → `sem_post(&m_smSemaphore)` (only while OPERATING, F3) | anything else |
| Sequencer | everything in the sequence | — |

### Sequencer thread

```text
sequencerThreadExec:
    while !shutdown:
        waitWake( 1 s )                        // sem_timedwait on m_wakeSemaphore
        if !m_startRequested: continue
        clear start/stop/abort flags and last_cycle
        if !canStart(): warn, sequence switch Off; continue      // R9, D12, D15
        snapshot positions; state(OPERATING); sequence switch On
        rv = runSequence()                     // 0 normal/stop, 1 abort, -1 error
        finishSequence( rv )

runSequence:
    for cycle = 1 ..:
        for idx in positions:
            if abort: return 1
            if stop or shutdown: return 0
            if stepTo( idx, firstStep ) < 0: return abort ? 1 : -1
            publish status {pos_index, cycle, target}; INFO log      // D8, D11
            switch dwell( m_timePerPos ):      // live value, read at dwell start (D9)
                abort: return 1
                stop / shutdown: return 0
        INFO log cycle complete
        if lastCycle or (m_numCycles > 0 and cycle >= m_numCycles): return 0  // D7, D9, D18

stepTo( idx ):                                  // always completes unless abort/shutdown/error (D1)
    sendTrigger(Off); waitFor( fxngen Off, fxngenTimeout )
    flush m_smSemaphore                        // after confirmed Off (D4)
    waitFrame( 1/C2freq + margin, capped )     // 0 frame, 1 none in flight: both OK (D4, D5)
    sendHwpTarget( positions[idx] ); waitFor( |pos - target| <= tol, hwpTimeout )
    sendTrigger(On); waitFor( fxngen On, fxngenTimeout )
    if firstStep: sendObsOn(true); m_weStartedObs = true          // D12
    if obs_on went Off mid-sequence: warn once, carry on            // D12

finishSequence( rv ):                          // still OPERATING (F3, R6)
    sendTrigger(On) (+ bounded confirm)
    if rv == 0 and not shutdown: waitFrame( frame timeout )        // D3; skipped on abort/error/shutdown
    if m_weStartedObs: sendObsOn(false)        // D12
    if rv >= 0: sendHwpTarget( positions[0] ), no wait             // D2, D17 (abort included; not on error)
    sequence and last_cycle switches Off; clear flags
    rv < 0 ? state(READY, true) : state(READY)                     // D6; abort logged at NOTICE
```

The final frame wait is skipped after an error (D19).

### Helpers

- `waitFrame(timeout)`: `XWC_SEM_WAIT_TS` then a `sem_timedwait` loop that
  continues on `EINTR`; returns 0 frame, 1 timeout, shutdown or abort, -1 error.
  Checks `m_abortRequested` after every wake (D13).
- `waitFor(pred, stop, timeout_ms, poll_ms)`: poll `pred`; -1 on timeout or
  when `stop()` (`shutdown() || m_abortRequested`) is true. App-local, written
  to be liftable (D10).
- `dwell(seconds)`: `sem_timedwait` on `m_wakeSemaphore` to an absolute
  deadline; returns early only on stop, abort or shutdown; otherwise loops
  until the deadline.
- Frame timeout = 1/`C2freq` + `frame.margin`, capped at `frame.maxTimeout`
  (D5).

### Shared state after the change

- Atomics: `m_startRequested`, `m_stopRequested`, `m_abortRequested`,
  `m_lastCycle`, `m_timePerPos`, `m_numCycles`, `m_sequencing`,
  `m_fxngenOutp`, `m_haveFxngen`, `m_fxngenFreq`, `m_haveFxngenFreq`,
  `m_obsOn`, `m_haveObsOn`, `m_reportedHwpPos`, `m_haveHwpPos`.
- Thread-owned: index, cycle, positions snapshot, `m_weStartedObs`.
- Local INDI properties updated from both the callback and sequencer threads
  (`sequence`, `last_cycle`, `abort`, `time_per_pos`, `num_cycles`, `status`)
  are updated under `m_indiMutex` (`{ //mutex scope`).
- Remote property templates (`hwp_position`, `C2outp`) are built once in
  `appStartup` and only sent with the copying `sendNewProperty(prop, el, val)`;
  `obs_on` uses `sendNewStandardIndiToggle`. Subscriptions: `hwp_position`,
  `C2outp`, `C2freq`, `obs_on` (`REG_INDI_SETPROP`).

## Decisions

Filled in from the grilling session. Each entry: decision, then rationale.

- D0. **Accepted (2026-11-18):** the sequencer thread owns the whole sequence.
  Callbacks and `appLogic` only set request flags and wake it.
  `startSequencing()`/`stopSequencing()` become requests. Rationale: matches
  `dmPokeCenter`; fixes audit items 3, 5, 6, 8 and 9 together; callbacks never
  block. Rejected alternative: a mutex around the existing split, which would
  block callbacks during moves/dwells and keep `loopPause`-driven stepping.

- D1. **Accepted (2026-11-18):** a stop is honoured at step boundaries and
  interrupts the dwell immediately; a step that has started (trigger Off →
  frame → HWP move → trigger On) always completes. Only shutdown and timeouts
  end a step's waits early. Rationale: cleanup always begins from a known
  HWP/trigger state; the extra delay (~1–5 s) is acceptable.
- D2. **Accepted (2026-11-18):** on completed, stopped, aborted and shutdown
  endings, send the HWP to `positions[0]` (amended by D17; was 0°) without
  waiting. After an error, leave the HWP
  where it is for diagnosis. Shutdown counts as a normal ending; the command
  may be lost if `hwptrack` is also shutting down, which is harmless.
- D3. **Accepted (2026-11-18):** stop stays graceful. Cleanup keeps the final
  bounded frame wait (trigger On → `waitFrame` → `obs_on` Off), skipped on
  shutdown and abort. Assumed purpose: the frame from the final dwell is saved
  before observers stops (user did not contradict; not explicitly confirmed).
- D13. **Accepted (2026-11-18):** add an `abort` toggle (new INDI switch,
  `std::atomic<bool> m_abortRequested`). Abort interrupts every wait, including
  step frame waits, HWP and fxngen waits, and the dwell, then runs cleanup
  without the final frame wait. The in-flight frame may be lost or incomplete.
  Rationale: with exposures up to 600 s a graceful stop can take ~10 min.
  `waitFrame`, `waitFor` and `dwell` check `m_abortRequested` alongside
  `shutdown()`. The abort callback sets the flag, then posts both
  `m_wakeSemaphore` and `m_smSemaphore`; every wait checks the flag after it
  wakes, so a wake-up caused by abort is never mistaken for a frame. Polling
  waits (`waitFor`) see the flag within one `m_hwpWait` interval.
- D16. **Accepted (2026-11-18):** testability follows `adcTracker`: virtual
  device-facing primitives overridden by a `\cond`-hidden test subclass that
  records actions; sequence logic runs for real in tests. One threaded test
  (T6). See Tests.
- D17. **Accepted (2026-11-18):** positions and HWP settings come from config:
  `hwp.positions` (comma-separated degrees, default `0,45,22.5,67.5`),
  `hwp.tolerance` (0.5°), `hwp.timeout` (5000 ms), `hwp.wait` (100 ms).
  `loadConfigImpl` fails startup if the list is empty, contains repeated
  values (the user specifies a shorter cycle instead, e.g. `0,45`), or has a
  value outside −360..360 (`hwp_position` range), or if tolerance, timeout or
  wait ≤ 0. A single position is allowed. Normal endings return the HWP to
  `positions[0]` rather than 0° (amends D2). The list is published in a
  read-only text property `positions`.
- D18. **Accepted (2026-11-18):** `last_cycle` is kept. Switching it On while
  idle is rejected (NOTICE log, switch reset Off). During a run it ends the
  sequence at the end of the current cycle, including when switched On during
  the last position's dwell; switching it Off cancels. It is cleared at start
  and at every ending (completed, stop, abort, error, shutdown). The INFO
  message on enable gives the remaining positions and estimated time using the
  live `time_per_pos`. Redundant with setting `num_cycles` mid-run, but kept as
  a one-click operator control.
- D19. **Accepted (2026-11-18):** cleanup skips the final frame wait after an
  error, as for abort and shutdown. Rationale: after an fxngen error the wait
  would usually run to its full timeout (up to 610 s); after an HWP error the
  in-flight frame's angle is unknown; the error is reported without delay.
- D14. **Accepted (2026-11-18):** keep the `timePerPos` INDI maximum at 600 s.
- D15. **Accepted (2026-11-18):** refuse to start if `time_per_pos < 1/C2freq`
  (warning gives both values); the `time_per_pos` callback rejects a
  mid-sequence change below the trigger period with the same message. A
  trigger frequency change during a run is not checked (the frame timeout
  follows `C2freq` at every wait). The trigger may be On or Off at start: an
  Off trigger makes the first step's frame wait time out, which D4 treats as
  success. User named no other manual pre-start checks.
- D4. **Accepted (2026-11-18):** step order is trigger Off → confirm → flush
  → `waitFrame(exptime + margin)`. Timeout means no frame was in flight;
  continue and log at debug level. Rationale: frames finishing before the
  confirmed Off are discarded, so the awaited frame can only be the one in
  flight. Costs at most one extra exposure + margin per step; the current code
  hangs in that case. Repeat the fast/slow angle validation (V1). No known
  camera behavior (on-camera buffering, overlapped readout) contradicts this;
  user did not raise any.
- D5. **Accepted (2026-11-18):** frame timeout = 1 / `<fxngen>.<channel>freq.current`
  + `frame.margin` (config, default 0.5 s), capped at `frame.maxTimeout`
  (config, default 610 s). Subscribe with `REG_INDI_SETPROP`; `C2freq` joins
  R9 (no start until received; refuse start if ≤ 0). Rationale: camsci is
  always triggered by the fxngen channel when sequencing (user: never
  free-running), so the trigger period bounds the in-flight frame; siglentSDG
  publishes `C2freq.current` from hardware readback. Exposures may be up to
  600 s, typically < 30 s.
- D6. **Accepted (2026-11-18):** a sequence that ends in error finishes with
  `state(stateCodes::READY, true)`: usable again immediately, with a sticky FSM
  alert until an operator presses `fsm_clear_alert`. Starting a new sequence
  does not clear it. Abort is an operator action: log at NOTICE, plain READY,
  no alert. Rationale: the fault is downstream (`hwptrack`, fxngen, camera);
  `stateCodes::ERROR` implies the app itself is recovering.
- D7. **Accepted (2026-11-18):** `numCycles = 0` means run until stopped
  (replaces -1; default 0). INDI range `0..10800`; every value is valid, so the
  callback has no special case. End check: `numCycles > 0 && cycle >= numCycles`.
  Matches `observerCtrl`'s `obsDuration` convention.
- D8. **Accepted (2026-11-18):** snake_case names to match siblings
  (`hwp_position`, `obs_on`, `poke_amp`): `sequence`, `last_cycle`, `abort`
  (toggles); `time_per_pos`, `num_cycles` (current/target numbers); one
  read-only `status` property (`createROIndiNumber`) with elements
  `pos_index` (0-based, as in the code), `cycle` and `target` (HWP angle for
  that index), updated together after each move. Replaces `hwpPosIndex` and
  `curCycle`. User confirmed renames are acceptable.
- D9. **Accepted (2026-11-18):** `timePerPos` and `numCycles` can be changed
  while sequencing and apply to the running sequence. Both become atomics the
  thread reads at each use: `timePerPos` at the start of each dwell (a dwell
  already under way keeps its deadline), `numCycles` at each end-of-cycle
  check (a value at or below the current cycle ends the sequence there, like
  `lastCycle`). `current` and `target` both update immediately, and each change
  is logged at NOTICE so the record shows when it took effect. The positions
  list is config-only and is snapshotted at start. Noted trade-off, accepted by
  the user: changing `timePerPos` mid-cycle gives unequal exposure sets across
  positions within that cycle. Supersedes the snapshot of these two values in
  R10.
- D10. **Accepted (2026-11-18):** `waitFor` is an app-local member template
  in `hwpSequencer`. Write it so it can be lifted out unchanged: predicate,
  stop condition (`shutdown() || m_abortRequested`), timeout and poll interval
  passed in, no other member state. No common predicate-wait helper exists in
  libMagAOX (`ttmModulator::waitValue` is app-local, has no shutdown check).
  Follow-up: propose a common `waitFor` and migrate `ttmModulator` (AGENTS.md
  rule 1).
- D11. **Accepted (2026-11-18):** no new telemetry type on this branch.
  `hwpTracker` already records `telem_poltrack` (`set_angle`, `actual_angle`,
  `pos_name`, `tracking`). hwpSequencer logs each step with `text_log` at
  INFO (cycle, `pos_index`, target angle, trigger On), plus start, stop, abort,
  error and parameter changes. Follow-up if reduction needs structured data: a
  `telem_hwpseq` type via `dev::telemeter`.
- D12. **Accepted (2026-11-18):** subscribe to `<observers>.obs_on`
  (`REG_INDI_SETPROP`; joins R9). Refuse to start if it is already On (warning:
  stop observing first). Turn it On after the first step's move, and in
  cleanup turn it Off only if the sequencer turned it On (`m_weStartedObs`). If
  it goes Off mid-sequence (`obsDuration` expiry or an operator), keep
  sequencing, log a warning, and do not turn it back On. Rationale:
  `obs_on` controls all default stream writers and an observation record
  (`observerCtrl.hpp`), so the sequencer must not end an observation it did
  not start. Replaces the current forced Off at start.

## Questions and Points of Clarification

All resolved on 2026-11-18; see Decisions. Remaining unconfirmed assumption:
the purpose of the final frame wait in cleanup (D3).

## Tests

Unit tests in `apps/hwpSequencer/tests/hwpSequencer_test.cpp`, following
`apps/adcTracker/tests/adcTracker_test.cpp` (namespace
`libXWCTest::hwpSequencerTest`, `\defgroup hwpSequencer_unit_test` under
`application_unit_test` in `tests/groups.dox`, a Doxygen brief per
`TEST_CASE`, `HWPSEQUENCER_TEST_DOXYGEN_REF` links):

- T1. INDI callback validation for every new/set callback
  (`XWCTEST_INDI_CALLBACK_VALIDATION`).
- T2. Config loading: defaults and overrides for all R11 keys, including the
  positions list.
- T3. `waitFrame`: frame posted → 0; no post → 1 after the timeout; shutdown
  set → returns promptly.
- T4. `waitFor`: predicate becomes true → 0; never true → -1 after the timeout;
  shutdown → -1 promptly.
- T5. Sequence behavior through the action seams (D16): step order (D4),
  cycle termination (`num_cycles` reached, changed mid-run, 0 = forever;
  `last_cycle` mid-cycle), `time_per_pos` change at the next dwell (D9), stop
  at a step boundary and during the dwell (D1), abort skipping the final frame
  wait (D13), error ending with no HWP-to-0 and a sticky alert (D2, D6),
  `obs_on` Off only if we turned it On (D12).
- T6. One threaded test: real sequencer thread, start and stop through the
  real callbacks; callbacks return immediately; thread joins cleanly.
- T7. `canStart()` refuses for each missing value and each R9/D15 condition.

Seams (D16): the device-facing primitives are `virtual` (`sendTrigger`,
`sendHwpTarget`, `sendObsOn`, `waitFrame`, `waitFor`, `dwell`). A test
subclass `hwpSequencer_test` (hidden with `\cond`, as in
`apps/adcTracker/tests/adcTracker_test.cpp:51`) records an action list and
returns scripted results. Except T6, tests call `runSequence()`/`stepTo()`/
`finishSequence()` directly on the test thread.

Tests are built and run by the user on a Linux workstation (`sem_timedwait`
and `pthread_tryjoin_np` are not available on macOS). They are offline: no INDI
server, real streams or hardware.

User-run instrument checks (results recorded below):

- V1. Fast and slow exposures: HWP angle valid for every saved frame with the
  new flush order (repeat the earlier validation).
- V2. Stop during an HWP move, and during a long dwell: trigger ends On,
  `obs_on` Off, state READY.
- V3. Shutdown while sequencing: exits within a few seconds; trigger On.
- V4. Stop camsci mid-sequence: sequence ends by timeout through cleanup,
  error visible, trigger On.
- V5. `numCycles` and `lastCycle` endings.

## Implementation Plan

User-implemented; commit order per AGENTS.md rule 22.

The app will not run on the instrument before step 2 lands (user,
2026-11-18), so no interim safety fixes are made to the old code paths.

1. Small fixes: shmim macros before any early return (R8); callback cleanup
   using `indiTargetUpdate`, warnings instead of `software_error`, remove
   redundant name checks.
2. Restructure (R1–R10, D0–D18): `waitFrame`, `waitFor`, `dwell` helpers;
   wake semaphore and request flags (start, stop, abort); thread-owned
   `runSequence`/`stepTo`/`finishSequence`/`canStart`; virtual action seams
   (D16); new subscriptions (`C2freq`, `obs_on`); new `appShutdown` order (R7,
   absorbing the uncommitted reorder); `appLogic` alive check; INDI renames and
   new properties (`abort`, `status`, `positions`, D8, D17).
3. Config and logging (R11, R12, D17, D11): positions/tolerance/timeouts/
   frame margin and cap, validation, help-text defaults, `text_log` per step,
   remove `std::cerr`/`SEMVALUE`.
4. Tests (T1–T7) and `tests/groups.dox`.
5. Documentation pass over `hwpSequencer.hpp`, `hwpSequencer.cpp` and the
   handbook page if one exists.
6. `clang-format` as a final, separate commit (restore `{ //mutex scope`).

Update this plan's Decisions and Status as each step lands.

## Follow-up and Edge Cases

- Follow-up (D11): `telem_hwpseq` telemetry if reduction needs structured
  sequence state.
- Follow-up (D10): generalize `waitFor` into libMagAOX with tests and migrate
  `ttmModulator::waitValue`.

- `hwptrack` tracking mode (resolved from code, 2026-11-18): `hwp_position.current`
  is reported in the set-point frame (`actual - trackingOffset`,
  `hwpTracker.hpp:509`), so the tolerance check works with tracking on or off.
  With tracking on, the stage keeps moving during exposures by design. A
  tracking update during the move wait can briefly offset `current` by the
  change in offset; the 0.5° default tolerance absorbs it. The sequencer does
  not require any tracking state.
- fxngen or `hwptrack` restarting mid-sequence: the step's wait times out and
  the sequence ends with an error. No automatic retry.
- Exposure time changed mid-sequence: only matters if D5 uses a fixed config
  value.
- Two clients toggling `sequence` quickly: requests are flags, so the last
  request wins at the next check.

## Status

- 2026-11-18: plan drafted; `hwpTracker` `hwp_position` registration fixed in
  `eb5e10b7`. Grilling session resolved D0–D19. The user's uncommitted
  `appShutdown` reorder is folded into step 2 (it does not fix F2 on its own).
- 2026-11-18: implemented by the agent at the user's request (the user had
  planned to write it). Commits: step 1 `5dd5e9d0`, step 2 `2bdfae73`, step 3
  `a3b5e5bc`, step 4 `a6cb8eda`, then documentation. **Not compiled or run**:
  the workstation used for the work has no mxlib/MagAO-X build environment
  and no `clang-format`. The user builds and runs the tests on a Linux
  workstation (`make -f Makefile.one t=../apps/hwpSequencer/tests/hwpSequencer_test`
  from `tests/`), runs `clang-format` (step 6) and then the instrument
  checks V1–V5.

- 2026-11-18: user reports the app builds without errors and all
  `hwpSequencer_test` cases pass (built and run by the user with
  `tests/Makefile.one`). `clang-format` (step 6) and V1–V5 still to do.

### Implementation notes

- Seams (D16): `waitFor` is a static template and cannot be virtual, so the
  virtual waits are the specific ones: `waitTrigger(bool)`, `waitHwp(float)`,
  `waitFrame(double)`, `dwell(double)`, alongside `sendTrigger`,
  `sendHwpTarget`, `sendObsOn`.
- `appShutdown` no longer turns the trigger On unconditionally. Cleanup does it
  for every sequence ending, including shutdown; when no sequence is running
  the app leaves the trigger as it is. This changes the old behavior, where an
  idle shutdown always sent "On".
- Logging went into step 2 because the restructure replaced every `std::cerr`
  line; step 3 is config only.
- Remaining race, accepted: a stop that arrives after the thread has accepted a
  start but before it sets `m_sequencing` is cleared and the sequence runs.
  Pressing stop again stops it.
- The `last_cycle` time estimate includes the current dwell:
  (remaining positions + 1) × `time_per_pos`.
- The `gslInterpolation`, `readColumns` and ImageStreamIO includes were unused
  and are removed; the Makefile's `-lgsl` is left in place.
