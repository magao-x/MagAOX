# DDSPC detected dm_slots and acc_ratio

## Purpose

`parameters.dm_slots` (DM slots per WFS frame) and `parameters.acc_ratio` (ACC samples per WFS frame) were fixed startup
values, so a change of the WFS, DM or ACC rate needed a matching configuration edit and restart. The request: "the acc
ratio and dm slot should be able to just be identified by what we are running everything at".

## Design

- `0` (the new default) detects a value; `1`-`16` fixes it as before. Until a detected value is confirmed the members
  keep their old defaults (4, 4).
- **dm_slots**: the DM scheduler's steps per cycle, `steps` of a valid production record in `syncopation.timingStream`.
  appLogic opens the stream, reads one record and closes it again once per window, so no reader stays attached.
- **acc_ratio**: the ACC samples accepted by the reader (`AdcStatus::m_samples`) per WFS frame received (the WFS source
  counter `cnt0`, so frames the WFS thread misses still count), over the window, rounded. It is usable only within 0.25
  of a whole number from 1 to 16 with at least 100 frames, and when `acc_ratio x accel_history` fits the 128-sample ACC
  history. acc000 at 7990 Hz with a 2 kHz WFS gives 3.995, so 4; acc000's own `scans_per_wfs`, which can read 3
  transiently, is not used, and any ACC source works.
- **Confirmation**: windows are one second long (appLogic); two consecutive windows that agree confirm a value, so a rate
  change in progress is not applied half-way.
- **Application**: the WFS thread applies a confirmed change at the start of `processImage`, before the frame's ACC
  window is built, and only while prediction is off (`m_predicting`, under `m_modelMutex`). It sets `m_slots`, `m_ratio`
  and the ACC window length, reallocates the copied-slot ring and resets the models (their dimensions change; a new
  WFS rate also changes the standard fit's meaning). While predicting, the change waits; the assisted state shows
  `<name>_<value>_applies_when_prediction_off`.
- **Prediction gate**: a request is refused while dm_slots is not known (except the main-loop route, which writes no DM
  slots) or, in assisted mode, while acc_ratio is not known. They are known when the applied value equals the confirmed
  one, or fixed.
- **INDI**: `rates` publishes the applied `dm_slots` and `acc_ratio`, and the last window's `scheduler_steps` and
  `acc_per_wfs`; BUSY until both are known. With a detected ratio, `ddspc_performance` carries the per-phase elements for
  all 16 possible phases, so a change needs no new INDI elements.

## Limits

- Below a 500 Hz WFS with an 8 kHz ACC the ratio exceeds 16 and is not detected; assisted is then refused.
- A rate change while predicting keeps the old dimensions until prediction is turned off.

## Verification (offline, development Mac)

- `check-rtc-controller`: new test "Detected dm_slots and acc_ratio resize the controller while prediction is off"
  (ratio cases; refusal before detection; no change while predicting; resize to 2 slots and ratio 8, then 300 assisted
  frames at the new dimensions with no step mismatch; fixed values ignore targets). The rest pass except the known
  timing-flaky "mfilt clone 2k" test.
- `check-operational-controls`, `check-operational-output`, `operational_config_test.py` pass.
