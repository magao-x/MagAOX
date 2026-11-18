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
- `status.frames` is stale at the first position of a sequence (D6). A cleaner fix is
  for the app to publish `frames` = 0 at sequence start; out of scope here.
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
