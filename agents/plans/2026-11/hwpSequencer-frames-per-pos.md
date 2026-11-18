# hwpSequencer: frames_per_pos and pulse-locked trigger pauses

Branch: `ml/hwpSequencer` (rebased on `ml/hwpTracker-updates`).

## Problem

On the instrument (2026-11-18, camsci1 `exptime` = 10 s, `C2freq` = 0.09998761863 Hz,
`readout_time` = 0.226 s):

- `time_per_pos` = 11 s gave 2 frames per position, as designed (trigger On for 11 s
  at a 10 s period), not the 4 the user wanted.
- `time_per_pos` = 8 s gave 2, 2, 1, 1 frames per position, and one frame was tagged
  `HWPTRACK SET ANGLE = 44.42`, i.e. exposed while the HWP was moving from 45 deg.
  (`telem_poltrack.set_angle` is the measured angle without offset, `m_hwpCurPos`.)

## Findings

- picamCtrl synchro is rising-edge, `ReadoutPerTrigger`, frame-transfer only
  (`apps/picamCtrl/picamCtrl.hpp:2426`, `:1841`). Exposure length is the camera
  `exptime`. When synchro is on, picamCtrl sets `C2freq` to the SDK
  `FrameRateCalculation` (`:1724`), about 1/(exptime + frame shift): the camera keeps up,
  and readout overlaps the next exposure. A frame lands (`atime`) about
  exptime + readout after its pulse, i.e. after the next pulse has already fired.
- siglentSDG `changeOutp` only sends `OUTP ON|OFF`, so the trigger output is gated
  with the generator running: the first pulse after On has an arbitrary phase. A
  time-based dwell therefore gives N-1 or N frames.
- Bug in the existing pause: after the trigger is confirmed Off, `stepTo` flushes the
  frame semaphore and treats the next frame to land as the in-flight frame. If Off comes
  within `readout` of a pulse, the next frame to land is the previous one, and the HWP
  moves while the last frame is still exposing. This explains the 44.42 deg frame
  (about 2% of pauses at 0.226 s / 10 s).

## Decisions (grilled 2026-11-18)

1. Camera timing: edge trigger, exposure = `<camera>.exptime`. Subscribe to
   `<camera>.exptime.current` and `<camera>.readout_time.value`; L = exptime + readout.
2. Interface: new INDI number `frames_per_pos`, 0 = time mode (`time_per_pos`),
   N > 0 = N frames per position. Mirrors `num_cycles` "0 = no limit".
3. Pulse grid: the generator keeps its phase while the output is gated, so pulse
   times are anchor + k P, P = 1/`C2freq`. The anchor is the pulse of the most recent
   landed frame: `atime - L`, from `shmimMonitor::currAtime()` (picamCtrl stamps
   `CLOCK_REALTIME` when the SDK delivers the frame). Re-anchored at every pause, so
   drift does not accumulate.
4. Frames mode: trigger On at a midpoint between pulses (anchor + (k + 1/2) P, no
   spurious edge with the short pulse), Off N P later (also a midpoint). Pulses k+1 to
   k+N fire. Landed frames are counted as a check.
5. Shared pause (both modes, decision 13): every trigger Off is sent at a midpoint, and
   the pause waits for the frame of the last pulse before it (atime >= that pulse + L - P/2)
   instead of "the next frame". Time mode's dwell can therefore run up to one period
   longer than `time_per_pos` (the next midpoint can be up to P away; P/2 was quoted
   during grilling, which was wrong); its turn-on stays immediate. Without exptime/readout (no anchor),
   time mode falls back to the old flush-and-wait pause.
6. No anchor at sequence start: prime. If timing is known, turn the trigger On if it
   is Off, wait one frame (not saved, `obs_on` is still Off) and anchor on it. A
   frames-mode position without an anchor, or a missing last-pulse frame in frames mode,
   ends the sequence with an error. In time mode a missing frame is a warning and
   drops the anchor.
7. `frames.minPeriod` (default 1 s): frames mode is refused (start and callback) when
   the trigger period is shorter, because On/Off must land well within P/2 through
   INDI -> siglentSDG -> SDG.
8. `obs_on` for the first position is turned On after the first move and before the
   trigger, and the sequencer waits (bounded, `observers.timeout`) for it to read back On.
   Applies to both modes.
9. `frameTimeout = min(P + L + frame.margin, frame.maxTimeout)`, covering the
   in-flight, priming and cleanup waits. If L is unknown: `min(P + margin, maxTimeout)`
   (deviation from the grilled answer, which fell back to maxTimeout: a missing camera
   subscription must not make a time-mode step hang 610 s).
10. `camera.devName` config, default = `shmimMonitor.shmimName` (camsci1). camsci2 needs
    no handling: it shares camsci1's trigger and exptime, so its exposures end at the same
    times.
11. Count mismatch: log a WARNING with position, expected and actual count, publish the
    last count in `status.frames`, continue.
12. `stop`/`abort` interrupt a frames dwell immediately (as in time mode); `last_cycle` is
    the graceful end. `frames_per_pos` changes, including 0 <-> N, apply from the next
    position and are validated like a start.
13. Tests: pure static timing functions (anchor, next midpoint, landed threshold,
    timeout) tested directly; flow tested with the existing harness. No simulated clock;
    the real interleaving is verified on the instrument (V-F below).

## Implementation

- Config: `camera.devName`, `frames.minPeriod`, `observers.timeout`.
- INDI: `frames_per_pos` (new number), `status.frames`; subscriptions to camera
  `exptime` and `readout_time`.
- `processImage`: store `currAtime()` (as double) and increment a landed-frame counter
  before posting the semaphore.
- Static helpers: `pulseTime`, `nextMidpoint`, `landedThreshold`, `calcFrameTimeout`.
- `framesModeOk(why)`, used by `canStart` and the `frames_per_pos` callback.
- `pauseTrigger()`: shared pause (midpoint Off + last-pulse wait, or the fallback),
  re-anchor, count check.
- `stepTo`: pause, move, first-position `obs_on` + wait, resume (frames: wait for the
  midpoint, then On; sets the off time for the dwell).
- `runSequence`: prime; frames dwell = until the off time.
- `finishSequence`: for completed/stopped with an anchor, pause (catch the last frame
  and count), `obs_on` Off, then trigger On. Otherwise the existing cleanup.
- Virtual `waitUntil(t)`, `waitLanded(pulse)`, `waitObsOn()` for the test harness.

## Verification

Unit tests (user, Linux workstation, `tests/Makefile.one`):
`hwpSequencer_test` including new cases for the helpers, `canStart`/callback validation,
priming, `obs_on` ordering, shared pause, count warnings and cleanup.

On the instrument (user runs; agents do not touch hardware):

- V-F1: frames mode, exptime 10 s, `frames_per_pos` = 4, 4 positions, 2 cycles: every
  position has exactly 4 camsci1 and camsci2 files, no `SET ANGLE` between positions,
  no count warnings in the hwpSequencer log.
- V-F2: same with `frames_per_pos` = 1.
- V-F3: time mode, `time_per_pos` = 8 and 11: no frame tagged with an intermediate angle.
- V-F4: frames mode with a short exposure near `frames.minPeriod` (e.g. 1.5 s period):
  counts exact; otherwise record the On/Off latency from the logs.

## Status

- 2026-11-18: plan grilled and recorded; implementation started.
- 2026-11-18: implemented in `apps/hwpSequencer/hwpSequencer.hpp` with tests in
  `apps/hwpSequencer/tests/hwpSequencer_test.cpp`; `clang-format` run. Not compiled:
  the agent workstation has no mxlib/MagAO-X build environment. Next: the user builds
  and runs `hwpSequencer_test`, then V-F1 to V-F4.
- Implementation notes:
  - The pulse anchor is refreshed at every pause from the last landed frame, and cleared
    at the start and end of each sequence.
  - Priming also runs in time mode when camera timing is known, so the first pause is
    already locked to the pulses. In time mode a missing priming frame is only a warning.
  - `status.frames` is published after every position in both modes; the count warning
    is only logged in frames mode and is skipped for a position cut short by `stop`.
  - The fallback pause (no camera timing) still sets the anchor from the frame it
    catches, so switching to frames mode mid-sequence works after one time-mode position.
