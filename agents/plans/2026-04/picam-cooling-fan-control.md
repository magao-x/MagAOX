## Problem Statement

Extend `apps/picamCtrl` to support fan control using the same `dev::stdCamera` fan-speed interface, logging, and telemetry path that was added for `pvcamCtrl`.

The PICam SDK reference is:

`/home/jrmales/Documents/MyPapers/Projects/MagAOX/Electronics/Cameras/Princeton/manuals/PICam 5.x Programmers Manual.pdf`

The design direction for PICam is now:

- treat the PICam cooling fan as a two-state fan with:
  - `on`
  - `off`
- reuse the existing `stdCamera` `fan_speed` machinery
- reuse the same startup/change logging pattern
- reuse the same `telem_stdcam` fan-state telemetry field

## What I Checked

- PICam 5.x Programmer's Manual
  - `PicamParameter_CoolingFanStatus`
  - `PicamParameter_DisableCoolingFan`
  - `PicamCoolingFanStatus`
- `apps/picamCtrl/picamCtrl.hpp`
  - current `c_stdCamera_fanSpeed` setting
  - existing temperature/status polling path
  - current startup write to `PicamParameter_DisableCoolingFan`
- `libMagAOX/app/dev/stdCamera.hpp`
  - current generic fan-speed config, callback, INDI, and telemetry implementation

## PICam Capability Summary

The PICam manual does not describe PVCAM-style multi-speed fan control.

What it does describe is:

- `PicamParameter_CoolingFanStatus`
  - reports the cooling fan state
- `PicamCoolingFanStatus`
  - `PicamCoolingFanStatus_On`
  - `PicamCoolingFanStatus_Off`
  - `PicamCoolingFanStatus_ForcedOn`
- `PicamParameter_DisableCoolingFan`
  - control parameter
  - described as enabling/disabling the thermoelectric cooling fan

For MagAO-X purposes, this is still close enough to fit the shared `stdCamera` fan interface if we model PICam as a two-state fan:

- `on`
- `off`

The `ForcedOn` readback state should be treated as an operational readback detail, not as a third user-commanded speed.

## Existing `picamCtrl` Behavior

Current relevant state in `apps/picamCtrl/picamCtrl.hpp`:

- `c_stdCamera_fanSpeed = false`
- `getTemps()` already polls camera thermal status in the normal app loop
- in the `CONNECTED` state, `picamCtrl` already writes:
  - `setPicamParameter( m_modelHandle, PicamParameter_DisableCoolingFan, PicamCoolingFanStatus_Off );`

So yes: the existing app already disables the cooling fan by default on connect.

That existing behavior is a good starting point for this feature, with two caveats:

- the control should be moved into the shared `stdCamera` fan-control path rather than being hardcoded
- the exact polarity/type of `PicamParameter_DisableCoolingFan` should still be checked against the PICam headers during implementation

## Evaluation

### 1. PICam can reuse the `stdCamera` fan interface if we generalize it slightly

The current `stdCamera` fan-speed implementation was written around the PVCAM names:

- `high`
- `medium`
- `low`
- `off`

However, the generic machinery itself is already broader than that:

- `stdCamera` stores the allowed values in `m_fanSpeedNames`
- the INDI property is built from `m_fanSpeedNames`
- callbacks already select among `m_fanSpeedNames`
- telemetry already stores the current fan state as a string

The one piece that is still PVCAM-specific is config validation:

- `camera.defaultFanSpeed` is currently hard-failed unless it is one of:
  - `high`
  - `medium`
  - `low`
  - `off`

To let PICam reuse the same interface cleanly, `stdCamera` should validate `camera.defaultFanSpeed` against the derived app's configured `m_fanSpeedNames` instead of against that fixed four-value PVCAM list.

With that change:

- `pvcamCtrl` can continue to use `high/medium/low/off`
- `picamCtrl` can use `on/off`
- logging and telemetry can remain shared

### 2. `ForcedOn` should not become a user-selectable fan speed

PICam reports:

- `On`
- `Off`
- `ForcedOn`

Recommended handling:

- user-selectable values remain only:
  - `on`
  - `off`
- readback should map:
  - `On` -> `on`
  - `Off` -> `off`
  - `ForcedOn` -> `on`

Additionally:

- when PICam reports `ForcedOn`, log that as an operational notice or warning
- do not expose `forced_on` as a switch choice in the writable `fan_speed` property
- some PICam models may not expose `CoolingFanStatus`; in that case, keep command support if `DisableCoolingFan` exists, but fall back to commanded-state logging/telemetry without hardware fan readback

This preserves the shared control interface while still surfacing the safety-related hardware behavior in logs.

## Recommended Design

### A. Generalize `stdCamera` fan-speed validation

Update `libMagAOX/app/dev/stdCamera.hpp` so that:

- `camera.defaultFanSpeed` is validated against `m_fanSpeedNames`
- the failure message reports the configured value and the allowed values for that camera

This is the main base-class change needed to support both PVCAM and PICam under the same interface.

### B. Enable the shared fan interface in `picamCtrl`

Update `apps/picamCtrl/picamCtrl.hpp` to:

- set `c_stdCamera_fanSpeed = true`
- populate:
  - `m_fanSpeedNames = { "on", "off" }`
  - `m_fanSpeedNameLabels = { "On", "Off" }`
- initialize:
  - `m_defaultFanSpeed`
  - `m_fanSpeedName`
  - `m_fanSpeedNameSet`

Recommended startup default:

- keep the engineering default in code as `on`
- set site-specific `off` behavior explicitly in config files where desired
- keep that decision in `camera.defaultFanSpeed` rather than hardcoding it in the connect path

### C. Add PICam fan readback

Add a helper in `picamCtrl`, analogous to the new PVCAM `getFanSpeed()` helper:

- read `PicamParameter_CoolingFanStatus`
- map it to the shared `stdCamera` string state:
  - `On` -> `on`
  - `Off` -> `off`
  - `ForcedOn` -> `on`
- if `ForcedOn` is reported, emit an operational log message

Recommended hook points:

- after connect, before applying the configured default
- during the normal steady-state polling path, alongside `getTemps()`

### D. Implement `setFanSpeed()` in `picamCtrl`

Implement a derived-app `setFanSpeed()` using the shared `stdCamera` interface.

Recommended mapping:

- `on` -> `DisableCoolingFan = false`
- `off` -> `DisableCoolingFan = true`

This polarity should be verified against the PICam headers during implementation.

Recommended behavior:

- because `DisableCoolingFan` is not currently onlineable in `picamCtrl`, treat `setFanSpeed()` like the other deferred PICam setters:
  - mark the app for reconfiguration
  - apply the actual PICam parameter in `configureAcquisition()`
- update `m_fanSpeedName` on successful apply inside the reconfiguration path
- use the same logging behavior as PVCAM:
  - if the applied state differs from prior state:
    - `fan speed changed from '...' to '...'`
  - otherwise:
    - `fan speed set to '...'`
- use `logPrio::LOG_NOTICE`
- force `recordCamera( true )`

### E. Remove the old hardcoded connect-time fan write

The existing direct call:

- `setPicamParameter( m_modelHandle, PicamParameter_DisableCoolingFan, PicamCoolingFanStatus_Off );`

should be replaced by the shared startup path:

- read current fan state
- apply configured default through `setFanSpeed()`
- log through the standard fan-speed logging path

This keeps PICam behavior aligned with PVCAM behavior.

## Concrete Implementation Steps

1. Generalize `stdCamera`
   - change `camera.defaultFanSpeed` validation from a fixed PVCAM list to membership in `m_fanSpeedNames`
   - update the config help string so it no longer claims the only allowed values are always `high`, `medium`, `low`, and `off`
   - keep hard-fail behavior for invalid configured defaults

2. Update `picamCtrl` declarations and defaults
   - enable `c_stdCamera_fanSpeed`
   - add `setFanSpeed()`
   - add a `getFanSpeed()` helper
   - set the fan names/labels to `on/off`
   - initialize current/set/default fan state consistently

3. Update `picamCtrl` startup/connect logic
   - remove the hardcoded direct disable-fan write
   - read current fan state after connect
   - queue the configured default through `setFanSpeed()`
   - apply the actual parameter in `configureAcquisition()`
   - log startup behavior

4. Update steady-state polling
   - poll the current PICam fan state in the app logic loop
   - keep `m_fanSpeedName` synchronized for INDI and telemetry
   - log if PICam reports `ForcedOn`

5. Verify behavior
   - `camera.fanSpeedControl=false` hides the fan interface
   - `camera.fanSpeedControl=true` exposes a two-state `fan_speed` property
   - `camera.defaultFanSpeed` accepts `on` or `off` for PICam
   - startup applies the configured default
   - logs use the same `LOG_NOTICE` pattern as PVCAM
   - telemetry continues to flow through `telem_stdcam`

## Verification Checklist

- confirm `PicamParameter_DisableCoolingFan` type and polarity against the PICam headers
- verify the existing hardcoded startup disable behavior is fully replaced by the shared fan path
- verify the writable `fan_speed` INDI property shows:
  - `on`
  - `off`
- verify startup always emits a fan-speed log
- verify readback updates current state correctly
- verify `ForcedOn` is logged but not offered as a selectable command
- verify `telem_stdcam` records the PICam fan state string the same way it now does for PVCAM

## Recommendation

Proceed with PICam fan support by reusing the current `stdCamera` fan interface, but first make the small `stdCamera` generalization needed so fan names are camera-defined rather than implicitly PVCAM-defined.

That gives us:

- one shared INDI control model
- one shared logging pattern
- one shared telemetry path
- PVCAM with `high/medium/low/off`
- PICam with `on/off`

## Affected Files

Expected primary touch points:

- `agents/plans/picam-cooling-fan-control.md`
- `libMagAOX/app/dev/stdCamera.hpp`
- `apps/picamCtrl/picamCtrl.hpp`

Telemetry files should not need additional changes if PICam reuses the existing `telem_stdcam` fan-state string that was added for PVCAM.
