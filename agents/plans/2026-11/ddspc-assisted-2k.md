# ACC-assisted DDSPC at 2 kHz (ddspc_mode accel_assisted_2k)

## Purpose

The request: "I want to be able to run accel at 2kHz and have a mode where the acc assisted send dm commands at 2kHz,
maybe this doesnt need to be a new mode but it should be able to send commands at 2kHz just like standard ddspc but
with accel in the math", and "its fine to fill the dm command slots as long as they are the same as the dm command
calculated at 2khz."

## Design

A `ddspc_mode` element, `accel_assisted_2k` (m_mode AccelAssisted with `m_assistedFrame`), reuses standard DDSPC's
per-frame path with the shadow controller's assisted fit selected. That fit already existed and was fed every frame
in standard mode (`m_pending.m_acceleration`, `m_accelValid`), but was never trained or selected:

- **Model**: `DDSPC::ShadowController`'s AccelAssisted predictor, a per-WFS-frame DDSPC whose regressor adds the ACC
  window (the newest `acc_ratio x accel_history` samples, `m_settings.m_accelHistory = 1`) to the WFS and command
  histories, with one command per frame (`m_assistedCommandSlots = 1`). It learns from the same applied input as
  standard (the frame's mean copied command, or its delta with the integrator).
- **Output**: one command per WFS frame, held on every DM slot of the frame through `m_slotPlan`, integrated once per
  frame with the main loop's gain (as standard): every slot carries the frame's 2 kHz command. The per-slot multirate
  fit, `assistedCandidate` and the `dm_slots` match it needs are not used.
- **Training/readiness**: `applyTraining` trains the per-frame assisted fit in this mode and the multirate fit only in
  `accel_assisted`; `selectedReady`, `assisted_ready`, the model blocker (`model_<shadow reason>`) and
  `assisted_qualification.model` (`2k <reason>`) follow the per-frame fit. Probes are held for the frame (both per-frame
  fits identify one command per frame; the per-slot draw applied only to the unreachable per-slot path).
- **Inputs**: as `accel_assisted`: the ACC window must be valid (`acc_window` holds, then stops after `hold_timeout`),
  at any ACC rate (`agents/plans/2026-11/ddspc-free-running-acc.md`). With a 2 kHz ACC, `acc_ratio` is 1.

The per-frame plan indexing for the assisted model previously assumed `m_slots` commands per frame
(`m_assistedCommandSlots = m_slots`); it now uses one per frame, as configured.

## Verification (offline, development Mac)

- "ACC-assisted 2 kHz predicts one command per WFS frame with the ACC in the regressor": selected through the real
  `ddspc_mode` callback, trained through the WFS path, ready; every slot of each of 2000 frames equals the frame's
  command; no multirate update or per-slot calculation; `assisted_ready=true`, `requested_mode=accel_assisted_2k`.
- "ACC-assisted 2 kHz corrects better than standard with an informative ACC": same plant and training, residual RMS
  over the last 1000 frames 0.33 (accel_assisted_2k) against 0.65 (standard). With an uninformative ACC it was 0.58.
- Standard numerics unchanged (its plan and probes are as before); the rest of `check-rtc-controller` passes except
  the known timing-flaky "mfilt clone 2k" test; `check-operational-controls` and `check-operational-output` pass.
