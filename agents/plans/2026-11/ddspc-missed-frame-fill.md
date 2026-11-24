# DDSPC: keep predicting through a missed WFS or ACC frame

2026-11-24. Branch `parkertjohnson1/accel_ddspc`. Workstation `ASTR-LT-428.local`; no instrument access.

## Why

User direction: a dropped WFS or ACC frame must not stop prediction or zero the DM. The DM keeps its command, and the
history vectors are filled with the previous measurement (or a prediction).

Before this change:

- **WFS gap, standard:** `step()` called `invalidate()`, which resets the standard fit's status. With learning off the
  fit stays not ready until the history refills (~history + future frames), so the output loses qualification
  (`command_invalid`) on the next slot: fault, own-channel zero, prediction off.
- **WFS gap, assisted:** `MultirateDDSPC::append()` breaks the history; slots hold for ~history frames. No fault.
- **ACC gap or late frame:** `sensorHistory()` rejects the window (`counter_gap`, `no_new_sample`), so assisted output
  loses qualification (`acc_window`): fault.
- `processImage()` drops a frame the writer overwrote during the copy, which itself produces a WFS gap.

## Design

`parameters.max_fill` (default 5, range 0–20; user direction: at least 5) consecutive missed frames are filled; 0
restores the old behaviour. The copied-slot ring holds `(max(max_fill, 2) + 2)·q` slots (`ringSize()`, set in
`loadConfig()`; previously a fixed 4 frames), so every slot of a filled gap and the current interval is retained. A gap
the ring cannot hold falls back to the newest copy. 20 frames (10 ms) is the WFS `input_timeout`.

- **WFS:** in `step()`, a gap of 1..`max_fill` frames calls `fillFrame()` for each missed counter before the current
  frame. It holds the previous measurement and takes the missed frame's interval from the copied-slot ring (the q slots
  after the previous decision), then advances exactly the state a received frame advances: standard `process()` of the
  pending record and `predict_next()` (origin chain), the integrator/slot input helpers, and for assisted
  `MultirateDDSPC::append()` without an ACC window (later windows overlap it, so the sample history stays contiguous).
  It does not learn, explore, or publish an origin to the DM worker, so the DM keeps its command: standard plays the
  held (integrator) or planned (absolute) slot; assisted holds the last slot. `m_decisionOrdinal` advances by q per
  filled frame, so the current decision is not counted as a slot shift.
- **ACC:** `sensorHistory()` looks samples up by counter. A missing sample in the window holds the sample before it. A
  WFS frame with no new samples gets the window one frame after the previous anchor (anchors keep advancing by r), at
  most `max_fill` frames beyond the newest real sample. More than `max_fill·r` filled samples still rejects
  (`counter_gap`). Late real samples replace the filled values in later windows.
- **Learning:** a fill sets `m_fillHoldoff = max(history, accel_history) + future + 1` WFS frames; `applyTraining()`
  disables both learners until it expires, so no example containing a filled value is learned. ACC fills count only in
  assisted mode. Prediction, scoring and output continue.
- Diagnostics: `ddspc_performance.wfs_filled`, `acc_filled`; `acc_window` reads `ok_filled`.

Filling with the previous value rather than a prediction keeps the change small and never feeds the model its own
output; a predicted fill can replace it later if hold proves worse.

## Not changed

- Gaps longer than `max_fill`, counter resets, `wfs_stale` (`input_timeout`), non-finite measurements, upstream lock
  loss and transport faults still behave as before.

## Execution notes

- Code: `loPredCtrlAcc_operational.hpp` (`step()`, `fillFrame()`, `sensorHistory()`, `applyTraining()`, config,
  INDI). Config: `max_fill = 1` in both `loPredCtrlAcc.conf` copies. Docs: RTC_TESTING "Missed frames".
- Tests (`tests/rtc_controller_test.cpp`): the harness `advance()`/`advanceAssisted()` gained a `missed` argument (slots
  and ACC continue, no decision).
  - ACC window: a frame with no new samples is filled once and refused the second consecutive time; one missing sample
    inside the window holds the previous one; `max_fill = 0` refuses.
  - Standard (absolute and integrator): learning pauses for the expected frames after a fill and resumes; frozen, a
    missed frame every 97 keeps the fit ready and the output qualified, no slot shifts, residual ratio < 0.25;
    `max_fill = 0` reproduces the old stop.
  - Assisted (absolute and integrator): a missed frame every 23 never breaks the multirate history, and every slot of
    the following frame is predicted.
- T1 (`standard_equivalence.py`): its "edges" scenario injects a WFS gap, which is now filled. The trace turns filling off
  (like the integrator) through a `requires` check, so T1 still verifies that nothing else changed: 192000 values
  bit-for-bit identical to HEAD and to `1ff631bb`.
- Default raised from 1 to 5: the ACC test fills five consecutive empty frames and refuses the sixth; the WFS tests use
  gaps of 1–5 frames (standard residual ratio 0.092).
- Results on `ASTR-LT-428.local`: `check-rtc-controller` 30/30, `check-operational-controls` 16/16 plus the config unit
  tests, `check-operational-output` 5/5.
- Formatting: changed line ranges only; the two hand-aligned `ddspc_performance` name tables were left in their layout.
