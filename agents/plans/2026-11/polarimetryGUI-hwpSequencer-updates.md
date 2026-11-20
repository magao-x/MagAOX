# Task Description

## Problem Statement

Update `polarimetryGUI` (`gui/widgets/polarimetry/`) for the reworked `hwpSequencer`
app: add the `frames_per_pos` option, update the text of the time-per-position entry,
fix the broken stagepollin presets drop-down, and add an "abort" button while
sequencing.

## Discussion

The `hwpSequencer` app was restructured on `ml/hwpSequencer` (see
`agents/plans/2026-11/hwpSequencer-sequencer-thread.md` and
`agents/plans/2026-11/hwpSequencer-frames-per-pos.md`); that branch is merged into
`ml/hwpsequencerGUI`. The GUI was written against the old INDI interface.

## Requirements

1. Add a `frames_per_pos` entry.
2. Update the text for the time-per-position entry.
3. Fix the stagepollin presets drop-down.
4. Add an "abort" button, available while sequencing.

## Tests and Metrics

## Scope and Caveats

# Instructions to Agent

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan

Date: 2026-11-18. Branch: `ml/hwpsequencerGUI` (contains `ml/hwpSequencer` at `764b97d9`).

## Task Summary

Bring `gui/widgets/polarimetry/polarimetry.hpp` and `polarimetry.ui` in line with the
current `hwpSequencer` INDI interface, add the frames-per-position and abort controls,
and fix the stage preset combo boxes.

## Findings

### G1. The GUI uses property names that no longer exist

`hwpSequencer` renamed its properties (sequencer-thread plan, D8). Current interface
(`apps/hwpSequencer/hwpSequencer.hpp:858-900`):

| Old (GUI uses) | Current | Kind |
| --- | --- | --- |
| `sequence.toggle` | `sequence.toggle` (unchanged) | toggle |
| `lastCycle.toggle` | `last_cycle.toggle` | toggle |
| — | `abort.toggle` | toggle (new) |
| `timePerPos` | `time_per_pos` (1–600 s, `%0.1f`) | current/target number |
| — | `frames_per_pos` (0–10000, 0 = time mode) | current/target number (new) |
| `numCycles` (-1 = forever) | `num_cycles` (0–10800, **0** = until stopped) | current/target number |
| `hwpPosIndex.value`, `curCycle.value` | `status.pos_index`, `status.cycle`, `status.target`, `status.frames` | RO number |
| — | `positions.value` (comma-separated degrees) | RO text |

Consequences today: the num-cycles and time-per-pos entries never connect, the
"last cycle" button sends to a non-existent property, the cycle/position status never
updates, and the `(-1 for infinite)` hint is wrong (0 now means forever, -1 is rejected).
The position display also hard-codes `/ 4` (`polarimetry.hpp:502`); positions now come
from config (`hwp.positions`).

### G2. The stage preset combos subscribe to nothing

`ui.comboHwpLin->setup("stagepollin", "", "", ...)` (`polarimetry.hpp:152`) passes an
empty property name. `statusCombo::subscribe` only subscribes to `m_property` when it is
non-empty (`statusCombo.hpp:283`), and `handleSetProperty` matches on it, so the list is
never filled and "go" sends to property `""`. Same for `comboQwpLin` / `stageqwplin`
(`:159`). Standard stages publish presets as the `presetName` switch vector
(`stagePairStatus.hpp:103` is the in-repo example). stagepollin and stageqwplin appear
next to `zaberLowLevel` in the INDI traffic logs, so they are likely `zaberCtrl`
(`stdMotionStage`) devices with `presetName` — to be confirmed (Q3).

### G3. Abort

`abort.toggle` On is honoured only while sequencing (otherwise reset Off). The app sets
it On/BUSY on request and resets it Off in cleanup (`hwpSequencer.hpp:1715`). Abort
interrupts every wait and drops the in-flight frame (D13); stop finishes the current
step and waits for the last frame.

### G4. Live-editable parameters

`time_per_pos`, `frames_per_pos` and `num_cycles` may now be changed during a sequence
(D9, frames plan decision 12); the GUI disables their entries while sequencing
(`polarimetry.hpp:487-489`).

### G5. Style

`polarimetry.hpp` has no `\file` block, undocumented members/functions, 3-space indent,
and inline definitions below the class (consistent with other GUI widgets, header-only
with `moc` include). AGENTS.md rule 16 requires a documentation pass over the touched file.

## Key Assumptions

- A1. Device names stay `hwpsequence`, `hwptrack`, `stagepollin`, `stageqwplin`,
  `qwptrack`, `stagepolrot`.
- A2. No other client depends on the GUI; only the GUI changes. No `hwpSequencer` changes.
- A3. `statusEntry` handles current/target numbers (as for `num_cycles` today), so
  `frames_per_pos` uses `statusEntry::INT` like `entryNumCycles`.

## Requirements

R1. All hwpSequencer property names updated per G1 (subscriptions, `statusEntry` setups,
    `handleSetProperty`, `last_cycle` button).
R2. New `frames_per_pos` `statusEntry` (INT) with a hint that 0 uses time per position.
R3. Time-per-position label/hint text updated (Q1).
R4. `num_cycles` hint changed to `(0 for infinite)`.
R5. Status shows `pos_index + 1 / <number of positions>` from `positions`, cycle as now,
    and frames (Q6).
R6. Preset combos use `presetName` (stagepollin, and stageqwplin per Q3).
R7. Abort button visible and enabled only while sequencing; sends `abort.toggle` On (Q4).
R8. Documentation pass over `polarimetry.hpp` (Q8).

## Questions and Points of Clarification

All resolved on 2026-11-18; answers recorded under Decisions (D1–D8).

- Q1. Time-per-pos text: what wording? Grey it out when `frames_per_pos` > 0?
- Q2. Allow editing `time_per_pos`, `frames_per_pos`, `num_cycles` during a sequence
  (app supports it), or keep them locked?
- Q3. Is stagepollin a `presetName` stage? Fix stageqwplin the same way?
- Q4. Abort: confirmation dialog? Button placement/style? Replace or sit beside "Stop"?
- Q5. Should the stage combos stay disabled while sequencing?
- Q6. Show `status.frames` / `status.target`?
- Q7. Tests: add a `polarimetry_test` (as `gui/widgets/pupilGuide/tests`) or manual only?
- Q8. Full doc/format pass on `polarimetry.hpp` now, or a separate later commit/PR?

## Decisions

- D1 (Q2). Only `num_cycles` is editable while sequencing. `time_per_pos` and
  `frames_per_pos` stay disabled during a sequence, as now.
- D2 (Q1). `Frames per pos.` entry placed directly above `Time per pos.` [s]. Hints in
  column 1: `(0 = use time)` and `(if frames = 0)`. When `frames_per_pos.current` > 0
  the time entry and its hint are disabled.
- D3 (Q3). User confirmed stagepollin and stageqwplin both publish `presetName`. Both
  combos are set up with `presetName`, `ctrlWidget(nullptr)` kept, fixed in the same
  commit.
- D4 (Q4). `buttonAbortSequence` ("Abort") in the button row beside Stop, visible and
  enabled only while sequencing. No confirmation dialog. Red via a new
  `QPushButton[isAbortButton="true"]` rule in `magaox.qss`. Disabled while
  `abort.toggle` reads On, until the app resets it.
- D5 (Q5). Stage combos stay disabled while sequencing (unchanged).
- D6 (Q6). Position line `pos_index + 1 / N`, N parsed from `positions.value`
  (fall back to `pos_index + 1` alone if missing or unparseable); no target angle.
  Cycle line unchanged. New `Frames (last pos):` row, `status.frames / frames_per_pos`,
  shown only in frames mode (`frames_per_pos.current` > 0) while sequencing; yellow on
  mismatch. No label warning style exists in `magaox.qss`, so add
  `QLabel[isWarning="true"]` alongside the abort button rule (toggle via
  `setProperty` + `style()->polish`). `status.frames` is not reset at sequence start
  (`publishFrames` is only called from the pause, `hwpSequencer.hpp:1458`), so the GUI
  shows `---` while `cycle` = 1 and `pos_index` = 0.
- D7 (Q7). Add `gui/widgets/polarimetry/tests/polarimetry_test.cpp`/`.pro` following
  `gui/widgets/pupilGuide/tests` (T1–T4 below). Built and run by the user.
- D8 (Q8). Full documentation pass (rule 16, `pupilGuide.hpp` as the reference) and
  `clang-format` on this branch, as separate commits after the functional and test
  commits (rule 22).

## Tests

`gui/widgets/polarimetry/tests/polarimetry_test.cpp` + `.pro` (D7), offline: synthetic
`pcf::IndiProperty` objects into `handleSetProperty`, capture of outgoing commands, as in
`pupilGuide_test`. Doxygen conventions per AGENTS.md rules 19–20.

- T1. Subscriptions use the current names: `hwpsequence.{sequence, last_cycle, abort,
  time_per_pos, frames_per_pos, num_cycles, status, positions}`, and
  `stagepollin`/`stageqwplin` `presetName`.
- T2. Enable/visibility rules: only `num_cycles` editable while sequencing (D1); time
  entry and hint disabled when `frames_per_pos` > 0 (D2); Abort/Stop/Last cycle shown
  only while sequencing; Abort disabled while `abort.toggle` is On (D4); stage combos
  disabled while sequencing (D5).
- T3. Status text (D6): `2 / 4` from `positions`; `2` when `positions` is missing or
  unparseable; frames row hidden in time mode, `---` at cycle 1 position 0, `4 / 4`
  normal, warning property set on mismatch.
- T4. Commands: Abort sends `abort.toggle` On; Last cycle sends `last_cycle.toggle`;
  Start/Stop send `sequence.toggle`.

Manual checks by the user against the instrument (agents do not connect to it):

- M1. All entries connect and show current values; `num_cycles` 0 accepted.
- M2. stagepollin (and stageqwplin) combos list presets and move the stage.
- M3. Start, last cycle, stop, abort: buttons show/hide correctly, app reaches READY,
  abort button resets.
- M4. Frames mode: `frames_per_pos` = 4 runs; status updates per position.

## Implementation Plan

1. Functional commit (with this plan):
   - `polarimetry.hpp`: property renames in `subscribe`, `setup` calls and
     `handleSetProperty` (R1); `num_cycles` hint `(0 for infinite)` (R4); `presetName`
     for both combos (R6); `entryFramesPerPos` + hints and greying (R2, R3, D2);
     `num_cycles` editable while sequencing (D1); position/frames status from
     `status` and `positions` (R5, D6); `on_buttonAbortSequence_pressed` and abort
     state tracking (R7, D4).
   - `polarimetry.ui`: frames entry row above time entry, hint labels, frames status
     row, `buttonAbortSequence`.
   - `magaox.qss`: `QPushButton[isAbortButton="true"]`, `QLabel[isWarning="true"]`.
2. Tests commit: `polarimetry_test` (T1–T4), test registration as for the other GUI
   widget tests.
3. Documentation-only commit: full pass over `polarimetry.hpp` (R8, D8).
4. `clang-format` commit on touched files (restore `{ //mutex scope` if any).

Builds and tests are run by the user (no MagAO-X build environment on this machine).

## Follow-up and Edge Cases

- `positions` text could be malformed or empty; fall back to `pos_index + 1` with no
  total.
- Done: hwpSequencer now publishes `status.frames` = 0 at sequence start
  (`ml/hwpSequencer` `203cf5f4`). The GUI still shows `---` at cycle 1, position 0, since
  0 would otherwise show as a mismatch.
- `frames_per_pos` changed via `setINDI` mid-sequence still updates the GUI (greying,
  frames row) since both follow `frames_per_pos.current`.
- If `stagepollin`/`stageqwplin` are renamed in config, the GUI device names are
  hard-coded and must follow.

## Status

- 2026-11-18: plan drafted and grilled (D1–D8). Awaiting go-ahead to implement.
- 2026-11-18: step 1 (functional) implemented.
- 2026-11-18: step 2 (tests) implemented. `polarimetry_test` built by the agent on macOS
  (Homebrew Qt, `qmake`, INDI static libs built in a scratch copy of the tree) and run
  with `QT_QPA_PLATFORM=offscreen`: all 5 test cases pass (134 assertions). A mutation
  check (empty stagepollin property; `num_cycles` locked while sequencing) failed 2 cases
  as expected. The full `polarimetryGUI` app was not built (needs mxlib). M1–M4 remain
  for the user.
- 2026-11-18: step 3 (documentation) done: `\file` block, class doc, grouped member docs,
  `///` and inline parameter docs on every declaration. Removed the unused `m_appState`
  member and `<iostream>` include (no behavior change). Tests rebuilt and pass.
- 2026-11-18: step 4 (`clang-format` on `polarimetry.hpp`; `polarimetry_test.cpp` was
  already clean). Tests rebuilt and pass. Remaining: user builds `polarimetryGUI` and
  runs M1–M4 on the instrument.

### Implementation notes

- `negOneLabel` renamed `labelNumCyclesHint`; new hint labels `labelFramesHint`,
  `labelTimeHint`; new status row `labelFramesStatus`/`framesStatus`.
- `status` elements are published by the app as doubles, so the GUI reads them with
  `get<double>()` and casts.
- `countPositions` (public static) returns 0 for an empty or unparseable list, which
  selects the `pos_index + 1` fallback.
- Abort button colours: `#a32a2a` background, `#d04040` border; warning label
  `#f0c419`. No existing red/yellow convention in `magaox.qss`.
- Abort button carries a tooltip: "Stop now: the frame being exposed is discarded".
- 2026-11-18: user could not run `polarimetry_test` ("no Qt platform plugin could be
  initialized"); the test `main` now sets `QT_QPA_PLATFORM=offscreen`, as `camera_test`
  does. Branch rebased onto `ml/hwpSequencer` at `203cf5f4` (status.frames reset).

## QWP tracking display (2026-11-19)

`qwpTracker` (`qwptrack`) now works on `ml/qwpTracker`. The GUI had a placeholder QWP
tracking slider (`sliderQwpTracking`, always disabled) and a `COMP_IMR` label beside it.
Grilled with the user on 2026-11-19.

### Findings

- Q-F1. `qwptrack` publishes `tracking.toggle`, and `qwp1`/`qwp2` as current/target
  numbers (degrees, fast axis). `current` comes from the elliptec stage readback, rounded
  to 0.01 and folded to [0, 180); `target` is the tracking-law request.
- Q-F2. The panel never subscribed to `qwptrack.fsm` or `qwptrack.tracking` (the latter
  was commented out), so `m_qwptrackFsmOk`/`m_qwpTracking` never changed; `onConnect`
  forced the slider disabled and greyed the `qwptrack` device label; the QWP slider lock
  in `updateGUI` used `m_hwptrackFsmOk`.
- Q-F3. The GUI depends on the app only through INDI names, so no rebase onto
  `ml/qwpTracker` is needed.

### Decisions

- QD1. No rebase. `ml/hwpsequencerGUI` and `ml/qwpTracker` merge independently; the
  coupling is the property names `qwptrack.{fsm, tracking, qwp1, qwp2}`.
- QD2. Show `qwpN.current` (where the waveplates are), not `target`.
- QD3. Format `QWP1 %3.0f°  QWP2 %3.0f°`, padded with figure spaces (U+2007, digit width)
  since the GUI font is proportional and ordinary spaces would let the width change.
- QD4. Label visible only while `tracking` is On (as `labelHwpTracking`); `---` when
  `qwptrack` is not READY/OPERATING or on disconnect.
- QD5. The QWP slider ignores input while sequencing or when `qwptrack` is not
  READY/OPERATING (toggling QWP tracking mid-sequence would change the polarization
  state within a dataset). The label is greyed while sequencing.
- QD6. Tests extend `polarimetry_test`; this section records the plan.

### Implementation

1. Functional: subscribe to `qwptrack.{fsm, tracking, qwp1, qwp2}`; cache `qwp1`/`qwp2`
   `current`; label text and visibility; enable the slider on connect; lock slider by
   `m_qwptrackFsmOk`. Slider locking factored into `lockSlider()`.
2. Tests: subscriptions; QWP label text/visibility/greying; slider lock.
3. Documentation pass and `clang-format`, if needed.

Manual check by the user (M5): with `qwptrack` running, toggle tracking from the GUI;
angles appear and follow the stages; slider is locked during a sequence.

## QWP pupil tracking slider (2026-11-20)

User request: "we need to add another slider to this gui based off the new qwpPupilTracker in the
ml/qwp-pupil-tracker branch. The slider should toggle the qwppupil tracking toggle. when it's on, the
right side should show the ttmperi offsets applied. we don't need to add the FSM to the top, but make
sure the toggle is disabled when the qwppupil FSM isn't ready (I think thats the default)". Follow-up:
"also, allow this slider to be turned off regardless if we're sequencing or not".

### Findings

- P-F1. `qwpPupilTracker` (device `qwppupil`, `ml/qwp-pupil-tracker`, plan
  `qwp-pupil-tracker.md`) publishes `tracking.toggle` (standard toggle switch) and read-only
  `offset` with elements `dx`, `dy` (camwfs px) and `v1`, `v2` (V, `-CM * offset`, the ttmperi
  voltage relative to the baseline `v0` captured at enable). These are the "ttmperi offsets applied".
- P-F2. `qwppupil` state: NOTCONNECTED until both stages and ttmperi are usable, READY then,
  OPERATING while tracking. It refuses to enable unless READY, so a GUI lock on
  "not READY/OPERATING" matches the app. Tracking stays On (OPERATING) if an input drops out.
- P-F3. `offset` reads zero while not tracking (the reference follows the stages).
- P-F4. As with `qwptrack` (QD1), the GUI couples to the app only through INDI names, so no rebase
  onto `ml/qwp-pupil-tracker` is needed.
- P-F5. The existing sliders use `lockSlider()` (input-transparent, not disabled, since a disabled
  `toggleSlider` draws as Off). `m_qwppupilFsmOk` defaults to false, so the slider starts locked.

### Decisions

Grilled with the user on 2026-11-20.

- PD1. New `sliderQwpPupilTracking` (`qwppupil`, `tracking`, `toggle`, label "QWP Pupil Tracking")
  and `labelQwpPupilTracking` beside it, in a new row directly above the QWP Tracking row
  (user: "this should be on first"; QWP Tracking and later rows shift down by one). No `qwppupil`
  FSM display in the status grid.
- PD2. Subscribe to `qwppupil.{fsm, tracking, offset}`; cache `m_qwppupilFsmOk`,
  `m_qwppupilTracking`, `m_qwppupilV1`, `m_qwppupilV2`.
- PD3. Label shows `v1`, `v2` only, 1 decimal, as `ΔV1: +1.2 V  ΔV2: −12.4 V`. Each value is padded
  with leading figure spaces to the width of a sign and three integer digits, and uses `+` / U+2212
  so the width stays constant. Visible only while tracking; `---` when `qwppupil` is not
  READY/OPERATING or on disconnect.
- PD4. Sequencing never locks the slider or greys the label (user: "Fully free"); it can be turned
  on or off during a sequence. It is locked only when `qwppupil` is not READY/OPERATING, which is
  the default until the app reports in.
- PD4a. For consistency the user changed the QWP label (QD3) to `QWP1: %3.0f°  QWP2: %3.0f°`.
- PD5. Tests extend `polarimetry_test`: subscriptions; label text/visibility; lock until READY;
  unlocked while sequencing.

### Implementation

1. Functional: UI row; subscriptions; `qwppupil` state cache in `handleSetProperty`; slider and label
   in `updateGUI`; static `pupilOffsetText()` for the label.
2. Tests: `qwppupil` subscriptions; `pupilOffsetText` formatting (padding, sign, rounding to -0.0);
   label visibility and text; lock until READY, unlocked while sequencing.
3. `clang-format` on the touched sources.

Status 2026-11-20: implemented. `polarimetry_test` built on macOS (Homebrew Qt 6.11, scratch tree)
and run offscreen: 7 test cases, 176 assertions pass. A mutation check (re-adding the sequencing
lock) failed 2 assertions as expected. Not built: the full `polarimetryGUI` app (needs mxlib).

Manual check by the user (M6): with `qwppupil` running, the slider is locked until `qwppupil` is
READY; enabling shows `ΔV1/ΔV2` that follow the QWP moves; the slider can be toggled during a
sequence.
