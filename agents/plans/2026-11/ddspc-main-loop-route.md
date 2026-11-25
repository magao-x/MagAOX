# DDSPC main-loop route (ddspc_mode main_loop)

2026-11-25. Branch `parkertjohnson1/accel_ddspc`. Workstation `ASTR-LT-428.local`; no instrument access.

## Why

On sky the standard mode (own channel `dm01disp05_aux` through synco, with the integrator) blew up the loop, and so did
Sebastiaan's restarted loPredCtrl. User direction (2026-11-25): add a third selection, toggled from the GUI, that runs
standard DDSPC at the main-loop rate, never writes `dm01disp05`, never zeroes modes 0 and 1, and sends the tip/tilt
prediction to the main loop as loPredCtrl does. Standard and ACC-assisted must not change.

loPredCtrl publishes a full modal frame each WFS frame: modes 0 and 1 are its command `new_command` (with
`is_integrating=false`), every other mode the WFS value, and cacao `mfilt` integrates it,
`DM[k] = mc*DM[k-1] - g*input[k]`. loPredCtrlAcc already republishes every WFS frame to that same stream (the main-loop
modal output) and zeroes modes 0 and 1 while DDSPC owns tip/tilt.

## Design

- `ddspc_mode` gains a third element, `main_loop`. Internally the mode stays `ShadowMode::Standard` (the standard fit,
  its learning and `learn_std`), and `m_mainLoop` selects the route. Selecting it requires the own channel's removal to be
  confirmed, as any mode change; a change while main-loop predicting stops prediction with the logged reason
  "DDSPC mode changed". A two-element request from an older GUI is still accepted.
- Decision: the standard plan's first column before the integrator, `gain*y + K-part + probe`, is the delta loPredCtrl
  sent. It replaces modes 0 and 1 of this frame's modal output; every other mode is copied. The integrator is not used:
  mfilt integrates with the main loop's own gain, multiplier and limit, and the integrator's parameters are not required
  for `predict` or `learn_std`.
- Publication order: the route publishes after `step()` (the decision uses this frame's measurement). Standard and
  assisted keep publishing before `step()`. The added latency is the standard decision time: T1 measures 13.5 us mean
  while learning, 1.4 us frozen, on the development Mac, so loPredCtrl's post-before-update ordering was not copied.
- Own channel: never requested (`m_output.request(false)`), so synco writes nothing beyond the startup zero; the slot
  plan and integrator state are held at zero. appLogic no longer turns prediction off for an unrequested output in this
  route.
- Prediction off: tip/tilt passes through unchanged (loPredCtrl sent `gain*y`; here the main loop is as if DDSPC were
  absent). `main_loop_tt` reports `predicted`, `passthrough`, or the existing `zeroed`.
- Training input: the tip/tilt actually published for the previous frame (the delta, or the passed-through measurement),
  not the own channel's copied positions. A missed WFS frame published nothing, so its input is zero; the fill holdoff
  already keeps such examples out of learning.
- A non-finite decision or numerical fault while predicting publishes the measurement instead and stops prediction:
  "DDSPC prediction stopped: standard model numerical fault; main-loop tip/tilt passed through".
- The delta's units and sign are those of the standard integrator's training input, so a fit trained in standard mode
  with the integrator is the same model; one trained with the integrator off (absolute positions) is not.

## Verification (offline, development Mac)

- `check-rtc-controller`: 35/35, including two new cases: the route publishes `gain*y` (untrained) in modes 0 and 1 and
  copies the rest, needs neither the own channel nor the integrator parameters, reports `main_loop`/`predicted`, uses
  the published delta as the applied input, and a mode change stops prediction and passes tip/tilt through on the next
  frame (this found a stale-flag bug, fixed). Closed loop through a simulated mfilt (gain 0.5): residual/disturbance
  0.221 for the main loop alone, 0.0025 with the route trained by probes and then frozen.
- `check-operational-controls` 16/16 and config tests OK.
- T1 `standard_equivalence.py --base HEAD`: 192000 standard rendered values bit-for-bit identical.

## Operator checks

- Stop loPredCtrl first: one publisher per modal stream (unchanged).
- Prediction off before selecting `main_loop`; `safety.zero_confirmed=true`.
- Sign: with `gain` and no model, the route sends `gain*y` into mfilt, exactly what passthrough sends scaled by `gain`;
  if that is stable the sign is right. A model learned in this route trains on what mfilt received.
