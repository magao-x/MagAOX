Problem: in the stdMotionStage base class if two or more presets have the same position but different names, and the user selects one of them by name, the earliest one in the list will be selected as the preset name instead of the one selected.  We need to implement tracking the intended preset name as well as the position and handle the case where multiple presets have the same position such that the user sees their intended preset name.  Review AGENTS.md then fill in this document with a plan to implement this capability.

Plan:
1. Inspect and document the current failure path in `libMagAOX/app/dev/stdMotionStage.hpp`.
   The alias bug comes from `newCallBack_m_indiP_presetName()` translating the requested switch selection into only `m_preset_target`, while `updateINDI()` and `recordStage()` later reconstruct the displayed preset name from `derived().presetNumber()`.
   Because that reconstruction uses only the current numeric preset index, any duplicate-position presets collapse to the first matching entry and the originally requested alias is lost.

2. Add explicit preset-name tracking state to `stdMotionStage`.
   Introduce protected member data for the selected or intended preset-name index, separate from the floating-point preset position.
   Keep this state documented in the same style as the rest of the class and place it with the other stage status members.
   Initialize and clear it in the same places where `m_preset` and `m_preset_target` are initialized or reset, including the power-off path.

3. Capture alias intent when commands arrive.
   Update `newCallBack_m_indiP_presetName()` so that, in addition to setting `m_preset_target`, it records the exact preset-name entry the user selected before issuing `moveTo()`.
   Update `newCallBack_m_indiP_preset()` so direct numeric moves clear the explicit-name selection, since those moves do not identify a unique alias.
   Preserve current behavior for homing and stop requests unless the motion-state transitions require clearing stale alias intent.

4. Define when tracked alias intent remains valid.
   In `updateINDI()`, prefer the tracked preset-name selection when all of the following are true:
   the tracked index is in range, the stage is at the tracked preset position, and the current move was a preset-name-driven request or has otherwise converged to that same alias target.
   Fall back to the current numeric `presetNumber()` lookup when there is no valid tracked alias, when the stage has been moved by raw position, or when the current position no longer matches the tracked alias location.
   This keeps alias display sticky only when it is justified by actual stage state.

5. Update INDI and telemetry emission to use the tracked name consistently.
   Refactor the name-resolution logic used by `updateINDI()` and `recordStage()` into one shared helper or one consistent code path so the INDI `presetName` property and `telem_stage` logging report the same alias.
   Ensure duplicate-position presets set only the intended switch element to `On`, rather than the first matching name.
   Keep the existing busy/idle handling for `m_indiP_preset` and `m_indiP_presetName`.

6. Audit derived-stage interactions for compatibility.
   Confirm that the current `filterWheelCtrl`, `hsfwCtrl`, and `zaberCtrl` code only needs the base-class behavior change and does not need per-app alias logic.
   Pay special attention to paths that overwrite `m_preset_target` after homing or while syncing telemetry, because those transitions may need to clear or preserve the tracked alias intentionally.

7. Expand regression coverage around duplicate preset positions.
   Add or extend unit tests, most likely in `apps/zaberCtrl/tests/zaberCtrl_test.cpp`, to cover:
   selecting one of multiple preset names that share a position,
   verifying `presetName` reports the selected alias after motion completes,
   verifying a direct numeric move to the same position does not falsely claim a specific alias,
   verifying power-off or other reset paths clear stale alias state.
   If `stdMotionStage` behavior is easiest to exercise through the existing zaber test fixture, keep the new tests there rather than introducing a separate harness.

8. Finish with style and verification steps required by this repo.
   Run `clang-format` on any touched C++ files.
   Re-read touched files for Doxygen/member documentation consistency per `AGENTS.md`.
   Run the relevant stage test target if available, or otherwise document what was verified and what remains untested.
