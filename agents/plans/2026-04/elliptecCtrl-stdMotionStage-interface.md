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
- Test work is still pending. When that stage starts, use `TEST_CASE` rather than `SCENARIO`.
