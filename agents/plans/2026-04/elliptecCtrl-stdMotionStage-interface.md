The new app elliptecCtrl is being imported from a different fork of MagAO-X.  We need to make some updates to it:
- while it inherits from stdMotionStage, it does not use all of the interface and re-defines several parts of it (homing, stop, presets)
- it does not telemeter everything that it should.  some of this will get cleaned up by switching to being a full user of stdMotionStage interfaces.
- We also need to add tests.  AGENTS.md is now up to date on this branch with our latest policy developments.  You can also review the branches jrmales/tracker-crash-guards and jrmales/flowrpm-app for how we have brought the adcTracker and flowRPM apps up to standards.

Please review AGENTS.md, and then formulate a plan to address the above points.  Fill in the plan below, but do not modify this prompt.  Do not being executing until I have a chance to review the plan.

Plan:
1. Audit `elliptecCtrl` against the actual `dev::stdMotionStage` and `dev::telemeter` contracts before changing behavior.
   - Document the current divergences in `apps/elliptecCtrl/elliptecCtrl.hpp`:
     - `dev::stdMotionStage<elliptecCtrl>::appStartup()` is never called.
     - `dev::stdMotionStage<elliptecCtrl>::updateINDI()` is never called from the main loop.
     - the app re-registers its own `home`, `stop`, and preset-selection properties instead of using the base interface.
     - there are no `onPowerOff()` / `whilePowerOff()` overrides to keep the stage helper and telemetry in sync with power-management transitions.
     - `recordStage()` is only reached through forced telemeter callbacks, so stage-state changes are not recorded promptly the way other stage apps do.
   - Use `apps/hsfwCtrl/hsfwCtrl.hpp`, `apps/smc100ccCtrl/smc100ccCtrl.hpp`, and `apps/zaberCtrl/zaberCtrl.hpp` as the main comparison points for how mature stage apps use the base helper.

2. Bring the app declaration and full touched-file documentation up to current `AGENTS.md` standards while preserving the local header-only app pattern.
   - Keep `apps/elliptecCtrl/elliptecCtrl.hpp` as the primary implementation file and `apps/elliptecCtrl/elliptecCtrl.cpp` as the main-entrypoint file only, matching the MagAO-X app header-only preference.
   - Add or refresh:
     - top-of-file Doxygen blocks,
     - class/group Doxygen if missing,
     - `///` summaries for declarations,
     - inline parameter documentation on header declarations,
     - member-variable documentation for non-trivial state,
     - named sections in the `... - Data` then accessor order required by `AGENTS.md`.
   - Add a local `typedef` / alias for the telemeter base if that improves consistency with the rest of the tree, but keep behavior changes minimal and scoped.

3. Refactor `elliptecCtrl` so `stdMotionStage` owns the standard preset/home/stop interface instead of the app maintaining a parallel copy.
   - In `setupConfig()` and `loadConfig()`, keep using the base helper for preset configuration and remove the mirrored `m_userPresetNames` / `m_userPresetDeg` state if the base `m_presetNames` / `m_presetPositions` can be used directly everywhere.
   - In `appStartup()`, replace the local registration of home/stop/preset-selection controls with `dev::stdMotionStage<elliptecCtrl>::appStartup()`, while preserving Elliptec-specific properties such as:
     - `absDeg`
     - `relDeg`
     - `relMove`
     - `velocity`
     - `status`
     - `optimize`
     - `save`
     - any remaining read-only helper property that is still useful
   - Remove the bespoke `m_ipHome`, `m_ipStop`, and `m_ipStageGoto` command path once the standard base properties cover those operations.
   - Decide whether `m_ipStageNamePos` remains as a read-only convenience property or should be dropped if it is redundant with the standard preset-name interface.

4. Make the Elliptec-specific motion implementation satisfy the base-class lifecycle cleanly.
   - Keep `stop()`, `startHoming()`, `presetNumber()`, and `moveTo(float)` as the required `stdMotionStage` surface, but update them so they work with the base helper’s state bookkeeping rather than a parallel local UI path.
   - Ensure `m_preset`, `m_preset_target`, `m_moving`, and any preset-name alias tracking maintained by `stdMotionStage` are updated consistently when:
     - a preset move is commanded,
     - a direct absolute-degree move is commanded,
     - homing starts,
     - homing completes and a home offset move begins,
     - a stop request interrupts a move,
     - the controller reconnects or loses connection.
   - Pay special attention to the post-home offset path so it does not leave stale preset or homing state behind after `stdMotionStage` has initiated the home request.

5. Wire the main loop and power-management callbacks through the base helpers so INDI and telemetry update in the standard way.
   - In `appLogic()`:
     - keep the Elliptec serial poll/reconnect FSM,
     - call `dev::stdMotionStage<elliptecCtrl>::updateINDI()` once the stage state has been refreshed,
     - continue running the telemeter helper each loop using the `TELEMETER_*` macros or equivalent helper calls preferred by `AGENTS.md`.
   - Add `onPowerOff()` and `whilePowerOff()` overrides so the stage helper sees powered-off transitions and the app can continue to emit forced telemetry at the configured cadence while power is off.
   - In `appShutdown()`, add the standard telemeter shutdown call before or alongside closing the serial port.

6. Expand telemetry coverage to include both standard stage-state telemetry and the stage’s native position telemetry.
   - Keep `telem_stage` as the standard state log and let `dev::stdMotionStage<elliptecCtrl>::recordStage()` handle it.
   - Add `telem_position` support for the native degree position, modeled on `filterWheelCtrl` and `smc100ccCtrl`, with:
     - `checkRecordTimes()` including both `telem_stage()` and `telem_position()`,
     - `recordTelem( const telem_stage * )`,
     - `recordTelem( const telem_position * )`,
     - `recordPosition( bool force = false )`.
   - Call `recordPosition()` on real state changes, not just on the telemeter max-interval path, so reconnects, homing completion, and motion completion are logged promptly.
   - Keep the custom status text property if it is still useful for operators, but make sure it does not become the only way motion state is surfaced.

7. Add focused unit tests for the app and for the new standard-interface behavior.
   - Create `apps/elliptecCtrl/tests/elliptecCtrl_test.cpp`.
   - Follow the updated app-test documentation policy from `AGENTS.md`:
     - use `namespace libXWCTest { namespace elliptecCtrlTest { ... } }`,
     - define `\defgroup elliptecCtrl_unit_test` in the test file,
     - keep the file itself only in `\ingroup elliptecCtrl_files`,
     - add a brief Doxygen block for each `TEST_CASE`,
     - include `tests/testXWC.hpp` and add Doxygen-only symbol references if protected-access test seams would otherwise hide the real API under test.
   - Structure the harness like the existing stage-app tests, but cover the `elliptecCtrl`-specific regressions we care about:
     - standard `stdMotionStage` callbacks for preset, preset-name, home, and stop are registered and dispatched,
     - Elliptec-specific callbacks for absolute/relative/velocity/optimize/save still work,
     - the poll-resolution logic updates `m_moving`, `m_preset`, and `m_preset_target` correctly across move, home, offset, stop, and reconnect paths,
     - `onPowerOff()` / `whilePowerOff()` clear or preserve stage telemetry state correctly,
     - `recordPosition()` and `recordStage()` emit on change and on forced telemeter callbacks,
     - duplicate preset positions, if configured, report the intended preset-name alias through the standard helper instead of the removed custom `stageGoto` path.

8. Update the shared test-group documentation to match the current policy.
   - Add the `app_unit_test` grouping entry to `tests/groups.dox` if it is still missing on this branch.
   - Treat `AGENTS.md` as the source of truth here, since several older in-tree tests still use the pre-policy structure.

9. Finish with the repo-standard verification and commit separation.
   - Run `clang-format -i` on the touched files.
   - Build and run the targeted `elliptecCtrl` unit test, and re-run any nearby stage-helper tests that are directly affected if the base-interface integration changes expose shared behavior issues.
   - Keep the work separated into clean commits on the feature branch:
     - functional `elliptecCtrl` / telemetry refactor first,
     - documentation and test-structure updates next,
     - formatting-only cleanup as a final commit if needed.

Style note / ambiguity to resolve during implementation:
- The current tree still contains older app-test Doxygen patterns, and `tests/groups.dox` does not yet define `app_unit_test` on this branch.
- Per `AGENTS.md`, I will treat the updated policy as authoritative and bring the new `elliptecCtrl` tests into that newer structure rather than copying the older examples verbatim.

Implementation status after the first refactor stage:
- Completed the functional and telemetry refactor in `apps/elliptecCtrl/elliptecCtrl.hpp`.
- `elliptecCtrl` now calls the `stdMotionStage` lifecycle hooks in `appStartup()`, `appLogic()`, `appShutdown()`, `onPowerOff()`, and `whilePowerOff()`.
- The app now uses the standard `stdMotionStage` preset/home/stop INDI surface instead of the previous parallel local interface.
- `telem_position` support was added alongside `telem_stage`, with forced `recordStage(true)` plus `recordPosition(true)` used on software-initiated state changes to bias toward over-recording rather than missing a transition.
- The ambiguous public `moveTo(double)` overload was removed in favor of a named internal helper so the only public `moveTo(...)` entrypoint remains the required `moveTo(float)` from `stdMotionStage`.

Implementation status after the test stage:
- Added `apps/elliptecCtrl/tests/elliptecCtrl_test.cpp` using `TEST_CASE` and the current `app_unit_test` Doxygen structure from `AGENTS.md`.
- Registered the new test in `tests/tests.list` and added the shared `app_unit_test` grouping entry in `tests/groups.dox`.
- Expanded the harness to cover:
  - `elliptecCtrl` configuration loading, lifecycle, serial helpers, callback glue, motion helpers, parser/error paths, and poll-resolution state handling.
  - `stdMotionStage` default-position synthesis, integer preset formatting, startup error branches, and callback edge cases through a dedicated probe harness.
- The tests exposed a real bug in `apps/elliptecCtrl/elliptecCtrl.hpp`: `elliptecCtrl` had a local `m_moving` that shadowed `stdMotionStage::m_moving`. This was removed, and the inherited state is now initialized in the constructor.
- Latest coverage report after the test pass:
  - `apps/elliptecCtrl/elliptecCtrl.hpp`: `96.4%` line coverage, `100%` function coverage.
  - `libMagAOX/app/dev/stdMotionStage.hpp`: `99.0%` line coverage, `100%` function coverage.
- Those results satisfy the repo-standard coverage target for this work item (`100%` function coverage and at least `95%` statement/line coverage).

Hardware bring-up (ELL21) — design review, 2026-11-18:
- Symptom: on an ELL21 rotation mount, pressing `home` cycles the app between HOMING, NOTCONNECTED and NOTHOMED.
- Reference: Thorlabs ELLx OEM/Bare modules protocol manual, Issue 12 (ETN032283-D03).
- Findings from the manual:
  - `ho`/`ma`/`mr` reply with `GS<code>` while incomplete (busy or error) and with `PO<pos>` when the move completes.
  - `st` only stops continuous (`fw`/`bw` with jog step 0) motion; it does not interrupt `ma`/`ho`.
  - ELL21: 65536 pulses/rev, 360 deg travel.
  - ELL16/ELL21 minimum acceptable velocity is 50%; lower values may stall.
  - GS codes other than 00 (OK) and 09 (busy) are errors (e.g. 02 mechanical timeout, 0C out of range) and must not be treated as idle.
- Findings from the code:
  - Serial I/O is not serialized: INDI callbacks (home/preset/absDeg/etc.) issue `txrx_` from the INDI thread while `appLogic()` polls from the main thread, so the two threads can consume each other's reply frames.
  - `q_status_()`/`q_position_()` accept whatever frame arrives next and do not check address or header.
- Decisions:
  - Default `motion.velPercent` is 80.
  - Enforce a per-model minimum velocity (50% for ELL16/ELL21/ELL22, detected from the `in` reply) on both config and INDI input, logging a warning when clamped.
  - Serialize all serial I/O with `m_indiMutex`, matching `smc100ccCtrl`/`zaberCtrl`/`filterWheelCtrl`: `appLogic()` holds it around connect/poll/reconnect, and every `elliptecCtrl` INDI callback acquires it (the `stdMotionStage` callbacks already do).
  - Motion commands (`ho`, `ma`, `mr`, `om`) are issued without waiting for completion so callbacks never hold the mutex for the duration of a move; the poll loop observes `GS09`/`PO` to track progress and completion.
  - Replace the "next frame is the answer" I/O with a generic frame reader: parse address/header/payload, drop frames for other addresses, log malformed frames at debug, and dispatch by header (`GS` -> `m_gs`, `PO` -> position only, `IN` -> device info including model and pulses/rev).
  - Queries (`gs`, `gp`, `in`) write, then dispatch every received frame until the expected header arrives or the timeout expires; each poll first drains already-buffered frames non-blocking.
  - Motion completion is decided by status only: a pending move/home completes when `gs` returns `00`, provided busy was observed at least once or ~200 ms elapsed since the command. `PO` frames never signal completion because `gp` also returns `PO`.
  - Add a config flag `serial.trace` that logs every TX/RX frame at `LOG_DEBUG2`, to diagnose hardware behavior from logdump (first target: confirm the ELL21 answers `gs` with `GS09` during a home).
  - Device-reported errors (GS codes other than 00/09) are logged at `LOG_ERROR` with the manual's meaning, clear the pending operation and the moving state, set `m_homed = false` if the failed operation was a home, surface the error text in `status`, and keep the port open.
  - Only true communication failures (read timeout, write/read error) count against `m_commMaxMisses`; malformed or unexpected frames do not. Reconnect only when the budget is exhausted.
  - Drop the separate busy read timeout; use a single short read timeout (~1 s) since `gs` answers promptly mid-move.
  - Reconnect preserves `m_homed` (only power-off clears it). If the first `gs` after reconnect returns `09`, treat it as an unknown motion in progress (OPERATING) until `00`.
  - After power-on the stage stays NOTHOMED until homed; presets and `absDeg` moves remain allowed while NOTHOMED (matches `smc100ccCtrl`).
  - Implement `stage.powerOnHome`: when true, issue one home after the first connection following power-on (not after a USB reconnect). The ELL21 instrument config should set `powerOnHome=true` (follow-up in the config repo).
  - Post-home offset stays app-side but uses an absolute move: on home `GS00`, issue `ma` to `stage.homeOffset` (not `mr`). Home + offset is one operation: state remains HOMING and `m_homed` is set only when the offset move also reports `00`; an offset failure is treated as a failed home. Do not modify the device's stored home offset (`so`). `homeOffset` only sets the post-home rest position; presets and `absDeg` remain in device coordinates.
  - Single-turn 360 deg model: display/telemetry wrap position to [0, 360); `absDeg` accepts [0, 360) and maps to `ma` in [0, 65536) pulses (verify with the trace that `ma` from a multi-turn count lands correctly); `relDeg`/`relMove` may be either sign, limited to +/-360 per step; preset matching uses circular distance. Remove `stage.allowMultiturn` and the 720 deg limits.
  - Add `stage.presetTolerance` (deg, default 0.5) used for preset matching with circular distance. Reject configs where distinct preset positions are closer than 2x the tolerance (exact duplicates remain allowed as name aliases).
  - `stop()` sends `st` under the mutex without waiting and does not assume success: pending operation and moving state are left for the status poll to resolve. A `GS03` reply logs a warning and sets `status` to "Stop not supported; move will complete" (not a fault). `m_preset_target` is cleared only once the device reports idle after a stop request. Verify on hardware whether `st` interrupts `ma`/`ho` on the ELL21.
  - `optimize` (`om`) tracked as `Pending::Optimize` with status-based completion. While optimizing: soft misses do not count toward the reconnect budget until `motion.optimizeTimeoutSec` (default 600) elapses (then treated as a failure); state OPERATING with status "Optimizing motors (may take minutes)"; motion requests (preset, home, absDeg, relative) are rejected with a warning while `stop` passes through. After completion set `m_homed = false` (no automatic re-home).
  - `save` (`us`) allowed only when idle and not optimizing; logged at `LOG_NOTICE` since it writes device EEPROM.
  - New config `stage.enableMaintenance` (default false): when false, `optimize` and `save` INDI properties are read-only/rejected.
  - Tests: replace the stateless `fakeElliptecDevice` with a stateful fake ELL21 on the existing PTY (pulse position at 65536/rev, timed motion, `GS09` while busy, unsolicited `PO` on completion, `in` info string, `st` -> `GS03` by default, timed `om`) with fault injection (GS02/GS0C errors, dropped replies, stray/malformed frames, under-velocity, position slip across power-off).
  - Regression scenarios: (1) home + offset completes HOMING -> READY without NOTCONNECTED; (2) concurrent INDI-thread home vs looping `appLogic()` completes without misses; (3) late `PO` during a `gs` query neither breaks the poll nor completes early; (4) `GS02` during home -> NOTHOMED, port open, status shows error; (5) `powerOnHome` homes once after power-on, not after reconnect; (6) preset arrival within 0.5 deg including 0/360 wrap; (7) optimize exceeds miss budget without reconnect, rejects motion, ends NOTHOMED; (8) `stop` with `GS03` leaves move to complete.
  - Bug-first: write scenarios 1-3 against the current code and confirm they fail before fixing. Keep the 100% function / >=95% line coverage targets; `stdMotionStage_probe` stays.
  - Work continues on `jrmales/elliptecCtrl` (coordinated with Jared). The rebased branch must be force-pushed (`--force-with-lease`) before new commits are shared. Commit order: failing fake-device tests (scenarios 1-3); serial/threading fix (mutex, frame reader, non-blocking motion, trace); behavior changes (errors, homing/powerOnHome/offset, 360 deg model, preset tolerance, velocity floor, stop, maintenance) with their tests, split where practical; documentation + plan; formatting.
- Hardware acceptance checklist (ELL21, `serial.trace=true`; record results here):
  1. Connect: `in` identifies ELL21 with 65536 pulses/rev; app reaches NOTHOMED/READY with velocity 80.
  2. Power-on: `powerOnHome` homes once, applies offset, ends READY with no NOTCONNECTED in logs.
  3. Manual home from READY: clean HOMING -> READY; trace shows `GS09` during and `GS00`/`PO` at completion.
  4. Presets: each preset reports the correct `presetName` on arrival; record landing error vs. the 0.5 deg tolerance.
  5. Wrap: moves crossing 0/360 in both directions via `absDeg` and `relMove`; an `ma` after more than a full turn of relative moves.
  6. Stop: `stop` during a long move; record whether it stops or replies `GS03`.
  7. USB reconnect: reconnects, keeps `m_homed`, no re-home.
  8. PDU power cycle: POWEROFF -> NOTHOMED -> auto-home -> READY.
  9. Optimize (maintenance enabled): record `om` duration, no reconnect, ends NOTHOMED.
  10. Logdump `telem_stage` / `telem_position` covering the above.
- Note: the earlier "Implementation status" sections describe the first refactor and test passes; the ELL21 design review above supersedes them where they conflict (notably the fake device, timeouts, `stop()` semantics, and the 720 deg range).

Implementation status after the ELL21 design review (2026-11-18):
- Bug-first: `apps/elliptecCtrl/tests/elliptecCtrl_ell21_test.cpp` adds a stateful simulated ELL21 (`fakeEll21`) on a PTY. Scenarios 1-3 (home + offset, concurrent INDI-thread home, stray `PO` during a move) were confirmed to fail on the pre-fix code before any app change. Scenario 3 initially passed on the old code because the simulator's completion `PO` re-aligned the old one-frame-behind reader; it was tightened so the simulator reports completion only via `gs`, which reproduces the stuck state.
- `elliptecCtrl.hpp` was rewritten around the decisions above in one commit (the serial/threading and behavior changes share the new frame reader and pending-operation tracking, so splitting them was not practical):
  - `m_indiMutex` serializes all serial I/O; `stop()` takes it itself because `stdMotionStage` calls it unlocked.
  - `readFrame_()` assembles CRLF frames into `frameT {addr, header, payload}`; `query_()` drains buffered frames, sends, and dispatches non-matching frames via `handleFrame_()` until the expected header arrives.
  - Motion (`ho`, `ma`, `mr`, `om`) is fire-and-forget; `Pending {None, Move, Home, HomeOffset, Optimize}` is resolved in `resolvePending_()` from `GS` status, with error latching (`m_latchedError`) and the 200 ms not-yet-started guard.
  - `appLogic()` no longer duplicates the framework's power-off handling; it leaves the device alone when power is off or being switched off.
  - Removed config: `serial.busyReadTimeoutMs`, `stage.allowMultiturn`. Added: `serial.trace`, `stage.presetTolerance`, `stage.enableMaintenance`, `motion.optimizeTimeoutSec`. Changed defaults: `serial.readTimeoutMs` 1000, `motion.velPercent` 80. `stage.homeOffset` is now an absolute angle (wrapped into [0, 360)).
  - Additional guard not discussed in the review: motion requests are also rejected while a home is in progress (not only during optimize), so a preset press cannot abandon a home midway.
- Tests: the existing unit suite was updated for the new internals (443 assertions) and the device-model suite covers scenarios 1-8.
- Open hardware questions (resolve with the acceptance checklist and `serial.trace=true`):
  - Whether the ELL21 answers `gs`/`gp` while moving. If it is silent during a home or move, the miss budget will still force a reconnect mid-motion (only optimize suspends the budget). If the trace shows silence, extend the optimize-style suspension to Move/Home with a motion timeout.
  - Whether a motion command produces both an immediate `GS09` and a completion `PO`, or a single reply; the app handles either.
