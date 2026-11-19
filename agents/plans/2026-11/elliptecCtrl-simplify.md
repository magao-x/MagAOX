Simplify the elliptecCtrl INDI interface and normalize it with the other `stdMotionStage` apps (zaberCtrl, smc100ccCtrl, filterWheelCtrl). Follows `elliptec-lowlevel-split.md`.

Design decisions (agreed 2026-11-19):

1. Position: `absDeg` becomes `position` (`current`/`target`, degrees, `createStandardIndiNumber<float>` as in zaberCtrl). Targets outside [0, 360) wrap rather than being rejected, so `position.target = current + delta` works without a client-side modulo. `current` follows the low-level `curr_pos` and is BUSY while moving or homing. No `rawpos`: counts belong to elliptecLowLevel.
2. Relative moves: `relDeg` and `relMove` removed. No other stdMotionStage app has them; clients nudge with `position.target`.
3. Status: the `status` text property removed. Every string except the error text duplicated `fsm`. The low-level `last_error` is still followed, only to log at WARNING when an error appears and INFO when it clears (elliptecLowLevel publishes but does not log it). The live text stays at `<lowLevel>.last_error.<stage>`.
4. `stageNamePos` removed. No other stage publishes preset angles; if a GUI needs them it should be a generic stdMotionStage addition.
5. Home: `stage.homeOffset` (an elliptecCtrl-only post-home move to an angle) replaced by the standard `stage.homePreset` (0-based), which elliptecCtrl previously accepted and ignored. On READY after a home, the app requests that preset with `moveTo`, as zaberCtrl does. The two-stage homing state and the READY-before-offset re-check are gone; `m_homingState` became `bool m_homing`. During the post-home move `fsm` shows OPERATING, matching zaber. `homePreset` beyond the configured presets is a config error.
6. Telemetry unchanged (`telem_stage`, `telem_position`, as filterWheelCtrl).
7. Hard rename, no compatibility aliases: elliptecCtrl is unmerged and nothing deployed uses the old names. `apps/qwpTracker` (subscribes to `absDeg`) is updated separately by its author.

Resulting properties: `position`, `preset`, `presetName`, `home`, `stop`, `fsm`.

Verification: not compiled in this session (macOS host without the Linux build environment). Build and run `apps/elliptecCtrl/tests/elliptecCtrl_test.cpp` with `tests/Makefile.one` before merging; delete the test `.o` first since Makefile.one does not track header dependencies.
