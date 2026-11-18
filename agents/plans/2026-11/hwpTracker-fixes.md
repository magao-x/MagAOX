# Task Description

## Problem Statement

`hwpTracker` (device `hwptrack`) does not compile at HEAD, never sends HWP
stage commands even once compiled, ignores the `tracking.sign` configuration,
and has several robustness and documentation gaps. `hwpSequencer` now depends
on `hwptrack.hwp_position.current` to verify moves, so these defects block
polarimetric sequencing.

## Discussion

Review of `apps/hwpTracker/hwpTracker.hpp` at `ae8460dd` (branch
`ml/hwpSequencer`) found:

### Bugs

1. Commit `2b8434ec` renamed members but `appStartup()` (lines 302-307) still
   uses the undeclared `m_indiP_hwpStagePos_target` and
   `m_indiP_hwpStagePos_current`, so the app does not compile.
2. `updateHwpPos()` sends `m_indiP_hwpStagePos["target"]`, but
   `m_indiP_hwpStagePos` is never given a device, name, or `target` element.
   No move command can reach `stagepolrot`.
3. Line 307 would register a second subscription to `stagepolrot.position`,
   which `m_indiP_stagePolRot` (line 284) already covers.
4. `tracking.sign` is declared in `setupConfig()` but never loaded into
   `m_trackOffsetSign`, so Nasmyth-West operation cannot be configured.
5. The stage FSM callback copies the stage's state onto the app. TCS loss is
   not visible, and `str2CodeFast` returns -999 for unrecognized strings, which
   is then set as the app's state.

### Behaviour and robustness

- While tracking, a new stage target is sent every `updateInterval` regardless
  of the size of the change or whether the stage is still moving.
- `m_zero + m_sign*pos` can fall outside the stage range; `getHwpStatus()` does
  not treat angles modulo 90 deg (e.g. 90 deg reports `Unknown`).
- When tracking turns on/off the offset changes before the stage readback
  arrives, so `hwp_position.current` and `hwp_position_name` are briefly wrong.
- `static lastupdate` in `appLogic()` and the change-detection statics in
  `recordPolTrack()` should be members.
- `std::cerr` debug prints in `updateHwpPos()` duplicate the `text_log`.
- `getHwpStatus()` hard-codes positions (0/22.5/45/67.5) and a 0.5 deg
  tolerance; `hwpSequencer` now has configurable positions.
- `teldata` values are not validated, and there is no record of whether
  `zd`/`pa` have been received, so tracking can start from altitude 0 /
  parallactic angle 0.

### Style and documentation (AGENTS.md)

- Class brief says "The MagAO-X ADC Tracker"; `updateInterval` help says the
  default is 1 s (it is 10); `m_devName` comment says `stagehwprot`.
- Many members and methods lack `///` docs; the "Configurable Parameters"
  section mixes config with runtime state.
- Unneeded `virtual` on app-local helpers; redundant name checks after
  `INDI_VALIDATE_CALLBACK_PROPS`; unbraced `if` bodies.
- Test file has `\author` and is only a placeholder.
- Telemetry stores `m_hwpCurPos` in the field named `hwpSetPos`.

## Requirements

1. The app compiles and the stage command property targets
   `<hwp.devName>.position.target`.
2. Exactly one subscription to `<hwp.devName>.position`.
3. `tracking.sign` is loaded and used.
4. The app's state reflects its own readiness, not a raw copy of the stage
   FSM, and never takes an invalid state code.
5. Existing INDI property names and the `telem_poltrack` schema are preserved
   unless explicitly agreed otherwise.

## Tests and Metrics

- Unit tests for the tracking-offset formula, stage<->HWP angle conversion,
  readback/position-name logic, config loading, and the INDI callbacks.

## Scope and Caveats

- No instrument interaction. Any on-sky or stage verification is performed by
  the user.

# Instructions to Agent

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan

## Task Summary

Fix the four compile/command/config bugs and the state-mirroring problem in
`hwpTracker`, add robustness where the user agrees, bring the file up to
AGENTS.md documentation standards, and replace the placeholder unit test with
real coverage. Work on a new branch `ml/hwpTracker-fixes`.

## Key Assumptions

- The tracking formula
  `offset = trackSign * 0.5 * parang + altitude + pupilOffset` is physically
  correct and is not to be changed.
- `hwptrack` property names (`tracking`, `hwp_position`,
  `hwp_tracking_offset`, `hwp_position_name`, `hwp_position_actual`) are
  consumed by `hwpSequencer`, GUIs and scripts and must remain stable.
- The stage is driven via `<hwp.devName>.position.target`, read back via
  `position.current`, and reports `fsm.state`; this interface is unchanged.
- `stagepolrot` rotates freely through 360 deg with no hard stops.
- `tcsi` publishes `teldata.zd` and `teldata.pa`, with `pa` in [0, 360).

## Requirements

1. Compiles; one subscription to `<hwp.devName>.position`; the stage
   command property is fully initialized (device, name, `target`).
2. `tracking.sign` is loaded. Config is validated in `loadConfigImpl()`
   (signs are +/-1, `updateInterval` finite and > 0, `hwp.zero` and
   `tracking.pupilOffset` finite, device names non-empty); failure logs the
   reason, returns -1, and `loadConfig()` sets `m_shutdown`.
3. App state per item 2 below; never an invalid state code.
4. Stage commands are sent only when the stage is `READY`; requested moves
   queue (item 3a); periodic tracking updates skip while not `READY`.
5. Stage target wrapped into [0, 360); position names matched modulo 90.
6. Tracking refuses to start before valid `teldata`; invalid `teldata` is
   ignored; `pa` is unwrapped within a tracking session.
7. Tracking-off behaviour, INDI property names and the `telem_poltrack`
   schema are unchanged.
8. Statics in `appLogic()`/`recordPolTrack()` become members; `std::cerr`
   debug prints removed; `updateInterval` help text corrected to 10 s.

## Questions and Points of Clarification

1. **Decided (2026-11-18): branch `ml/hwpTracker-updates` from `dev`.**
   `apps/hwpTracker` is identical on `dev` and `ml/hwpSequencer`, and `dev`
   has the same compile error.
2. **Decided (2026-11-18): option B.** `hwptrack` reports its own readiness:
   `NOTCONNECTED` until both the stage and `tcsi` have published, `READY`
   when the stage is `READY`, `OPERATING` while tracking or the stage is
   moving, `ERROR` on stage error. It no longer copies the stage FSM.
2a. **Decided (2026-11-18): scope is `hwptrack` only.** A `stagepolrot`
   audit was considered and dropped. The existing stage interface
   (`<hwp.devName>.position.current/target`, `<hwp.devName>.fsm.state`) is
   kept as-is. Noted for follow-up: `elliptecCtrl` on
   `origin/jrmales/elliptecCtrl` publishes `absDeg`, not `position`, so it
   would not match this interface. Branch base is `dev`.
3. **Decided (2026-11-18): option B.** Periodic tracking updates in
   `appLogic()` are skipped when the stage FSM is not `READY`; the next
   interval retries. No deadband. Known limitation: near transit, if a move
   takes longer than `updateInterval`, tracking lags.
3a. **Decided (2026-11-18): queue requested moves (option C).** New
   `hwp_position.target` and tracking on/off update the setpoint at once.
   The stage command is sent immediately if the stage is `READY`, otherwise
   a single pending flag is set (latest request wins) and the command is
   sent on the next stage `READY`. A pending move waits through stage
   error/`NOTCONNECTED`; `hwpSequencer`'s `hwp.timeout` covers that case.
   Periodic tracking updates share the same READY check.
   Working branch: `ml/hwpTracker-updates` (from `dev` at `ccc26f4f`).
4. **Decided (2026-11-18):**
   - Stage targets are wrapped into [0, 360). `stagepolrot` rotates freely
     with no hard stops, so no shortest-path or forbidden-angle logic.
     No 90/180 deg travel-minimizing wrap (keeps physical orientation).
   - Position names match `hwp_position.current` modulo 90 deg, using the
     difference wrapped into [-45, 45).
   - `hwp_position.current` is not wrapped; it stays in the user's frame so
     `hwpSequencer`'s `fabs(current - target)` check keeps working.
5. **Decided (2026-11-18): keep named positions hard-coded (option A).**
   Q+/Q-/U+/U- at 0/45/22.5/67.5 deg with 0.5 deg tolerance become named
   constants; only the modulo-90 matching changes. `hwpSequencer`'s
   `hwp.positions` is an unnamed cycle order, so nothing to share. Fast-axis
   misalignment is handled by `hwp.zero`.
6. **Decided (2026-11-18): `teldata` validity.**
   - Tracking-on is refused (warning logged, switch reset to Off) until a
     valid `teldata` has been received.
   - Each update requires finite `zd` and `pa` and `zd` in [0, 90]. Invalid
     updates are ignored, keeping the last good values; log only on
     validity transitions.
   - No receipt-age expiry (AGENTS.md rule 27).
   - `tcsi` publishes `pa` in [0, 360).
6a. **Decided (2026-11-18): unwrap `pa` (option B).** Each new `pa` is
   shifted by a multiple of 360 to lie within +/-180 of the previous value,
   keeping `0.5*pa` continuous through transit (pa crosses 0 for targets
   transiting south of zenith at LCO). Unwrapping resets to the raw `pa` when
   tracking is enabled. Accepted trade-off: within a session the HWP may sit
   180 deg from where a fresh start at the same `pa` would put it (same
   polarization state).
7. **Resolved (2026-11-18): no telemetry schema change.** The schema field
   is `set_angle` and correctly holds the measured user-frame angle; only the
   local `static hwpSetPos` in `recordPolTrack()` was misnamed and it becomes
   a member. Adding `target_angle`/`offset` fields was considered and
   declined.

8. **Decided (2026-11-18): tracking-off behaviour unchanged.** Offset goes
   to 0 and the stage is commanded to the bare setpoint (through the queued
   move path of item 3a).
9. **Resolved (2026-11-18):** `tracking.updateInterval` default stays 10 s
   (matches kTracker/adcTracker); only the help text is wrong.
10. **Decided (2026-11-18): validate config** as listed in Requirements 2,
   following `hwpSequencer`. `hwp.zero` is not range-limited (target is
   wrapped).

## Tests

Replace the placeholder `apps/hwpTracker/tests/hwpTracker_test.cpp`,
following `hwpSequencer_test.cpp` (`XWCTEST_SETUP_INDI_*` /
`XWCTEST_INDI_*_CALLBACK` macros, a test subclass that captures stage
commands instead of sending them, `HWPTRACKER_TEST_DOXYGEN_REF` refs, one
Doxygen block per `TEST_CASE`). Register it in `tests/tests.list`.

1. Callback validation: each new/set callback accepts its own property and
   rejects others.
2. Config: every key is loaded (incl. `tracking.sign`); each invalid value in
   Requirements 2 makes `loadConfigImpl()` return -1.
3. Offset formula for both tracking signs, with pupil offset.
4. Stage target: zero/sign applied, wrapped into [0, 360) for negative and
   > 360 inputs.
5. Readback: `hwp_position_actual`, `hwp_position.current` (unwrapped),
   near-zero cleanup, and names modulo 90 (e.g. 90 -> Qplus, -22.5 ->
   Uminus, 10 -> Unknown, boundaries at tolerance).
6. Queueing: a move requested while the stage is not `READY` is not sent,
   the latest request wins, and it is sent on the stage's next `READY`;
   periodic tracking update is skipped while not `READY`.
7. `teldata`: tracking-on refused before valid data (switch back Off);
   non-finite or out-of-range values ignored, last good kept.
8. `pa` unwrapping across 359.9 -> 0.1 and 0.1 -> 359.9 keeps the offset
   continuous; resets when tracking is enabled.
9. State: `NOTCONNECTED` -> `READY` -> `OPERATING` -> `ERROR` transitions
   from stage FSM, `teldata` receipt and tracking; unknown stage state
   strings do not produce an invalid code.
10. Tracking off: offset 0 and stage commanded to bare setpoint.
11. `recordPolTrack()` emits on change and on force only.

## Implementation Plan

Branch `ml/hwpTracker-updates`. Commits:

1. Fix: load `tracking.sign`.
2. Config validation and `m_shutdown` on failure.
3. App state logic (item 2) and stage-READY gating with queued moves
   (items 3, 3a).
4. Target wrapping and modulo-90 names (items 4, 5).
5. `teldata` validation and `pa` unwrapping (items 6, 6a).
6. Statics to members, drop `std::cerr`.
7. Unit tests + `tests/tests.list`, with plan update recording build/test
   results.
8. Documentation pass over `hwpTracker.hpp`, `hwpTracker.cpp` and the test
   file (class brief, help texts, `///` on all members/methods, inline
   parameter docs, section ordering, remove `\author`, drop unneeded
   `virtual`, redundant name checks, braces).
9. clang-format (restore `{ //mutex scope` if any).

Build and run the hwpTracker test locally; no instrument commands.

## Status

- 2026-11-18: implementation started on `ml/hwpTracker-updates`.
  **Correction to the review:** the review above was made on
  `ml/hwpSequencer`, not `dev`. On `dev` (`ccc26f4f`), bugs 1-3 are already
  fixed: the app compiles, `m_indiP_hwpStagePos` is initialized, and there is
  no duplicate subscription. `ml/hwpSequencer` carries three hwpTracker
  commits on top of `dev` (`67ec6dea`, `361bcd86`, `eb5e10b7`) that
  reintroduce the undeclared `m_indiP_hwpStagePos_target/_current` lines.
  Those commits must be dropped (or this branch merged over them) before
  `ml/hwpSequencer` reaches `dev`, otherwise hwpTracker stops compiling.
  Bug 4 (`tracking.sign` not loaded) is present on `dev` and is fixed in
  commit 1.

### Implementation notes

- Commits 2-6 of the implementation plan were made as one functional commit:
  the changes all touch the same callbacks and the shared move path, so
  splitting them would not give independently useful steps. New and
  rewritten declarations were documented as written; the separate
  documentation commit covers the remaining files.
- Shared state is protected by `m_indiMutex`, following `adcTracker`. INDI
  callbacks are not serialized by the framework, and `appLogic()` runs on
  the main thread. `state()` try-locks `m_indiMutex`, so it is only called
  from `appLogic()` with the mutex released.
- All stage commands go through `requestMove()`; `sendStageTarget()` is the
  virtual test seam. The tracking offset applied is computed at send time
  and stored as `m_hwpTrackingOffset`, which the readback uses, so a queued
  move does not change the reported angle until it is actually sent.
- A periodic tracking update skipped because the stage is not `READY` is
  retried on the next `appLogic()` pass after the stage becomes `READY`,
  not a full `updateInterval` later (`m_lastUpdate` is only advanced when
  the update is sent). This reduces lag near transit.
- Readback: the measured angle is shifted by a multiple of 360 to lie
  nearest the commanded angle (`setpoint + offset`), so
  `hwp_position_actual` and `hwp_position.current` stay in the user's frame
  even though the stage target is wrapped into [0, 360).
- Stage state strings are parsed with `stateCodes::str2Code` and checked by
  round trip through `codeText`; unrecognized strings are logged and
  treated as `NOTCONNECTED`.
- The app now starts in `NOTCONNECTED` instead of `READY`.
- Internal renames (no INDI name changes): `m_indiP_hwpStagePos` ->
  `m_indiP_stageTarget`, `m_indiP_stagePolRot` -> `m_indiP_stagePos`,
  `m_indiP_stagePolRotFsm` -> `m_indiP_stageFsm`, since the stage device
  name is configurable. `getHwpStatus()`/`getHwpTrackingOffset()`/
  `updateHwpPos()` are replaced by static `positionName()`,
  `trackingOffset()`, `stageTarget()`, `nearestAngle()` and `requestMove()`.
- Unused includes (`gslInterpolation.hpp`, `readColumns.hpp`) removed;
  `mx/math/geo.hpp` added for `angleMod`.
- Known race, accepted: between sending a target and the stage publishing
  `OPERATING`, the stage still appears `READY`, so a request arriving in
  that window (one stage poll period) is sent mid-move. Tracking updates
  are 10 s apart and `hwpSequencer` waits for arrival before its next
  request, so this window is rarely hit.

### Tests

`apps/hwpTracker/tests/hwpTracker_test.cpp` (registered in
`tests/tests.list`) has 7 `TEST_CASE`s: callback validation, configuration
(defaults, overrides incl. `tracking.sign`, each invalid value, shutdown via
`loadConfig()`), angle helpers (offset, wrapped stage target, nearest angle,
names modulo 90), move queueing, tracking (teldata gating, invalid teldata,
start/stop, periodic skip while not READY, `pa` unwrapping and reset),
stage readback in the user's frame, app state, and telemetry. Non-finite and
empty-string config values are tested by setting the member before
`loadConfigImpl()`, since the config parser may not round-trip them.

**Not compiled or run**: this Mac has no mxlib/MagAO-X build environment.
The user builds and runs on a Linux workstation:
`make -f Makefile.one t=../apps/hwpTracker/tests/hwpTracker_test` from
`tests/`, and builds the app (`make` in `apps/hwpTracker`).

## Follow-up and Edge Cases

- User to verify on the instrument that tracking commands reach
  `stagepolrot`, that queued moves complete, and that `hwpSequencer` sees
  correct `hwp_position.current`.
- Tracking lags near transit if a move takes longer than `updateInterval`.
- A pending move waits indefinitely through stage error; relies on the
  caller's timeout.
- `elliptecCtrl` (on `origin/jrmales/elliptecCtrl`) publishes `absDeg`, not
  `position`; if `stagepolrot` moves to that app, `hwptrack` must be
  adapted.
- Within a tracking session the HWP may sit 180 deg from a fresh start at
  the same `pa` (unwrapping trade-off).
