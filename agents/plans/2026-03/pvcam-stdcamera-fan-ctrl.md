Implemented by https://github.com/magao-x/MagAOX/pull/335

## Problem Statement

Implement fan speed control in `apps/pvcamCtrl` using the PVCAM fan-speed setpoint interface described in:

`/home/jrmales/Documents/MyPapers/Projects/MagAOX/Electronics/Cameras/Teledyne/PVCAM_User_Manual/_ex__fan_speed_and_temperature.xhtml`

The requested behavior is:

- Add a new optional fan-speed interface to `dev::stdCamera`.
- Expose the control through an INDI one-of-many switch with the four PVCAM fan levels:
  - `high`
  - `medium`
  - `low`
  - `off`
- Include the fan state in `stdCamera` telemetry.
- Integrate the feature in `pvcamCtrl` with minimal behavioral change outside the new control.

## What I Checked

- `libMagAOX/app/dev/stdCamera.hpp`
  - Existing optional camera features follow a consistent pattern:
    - compile-time `c_stdCamera_*` flag in the derived app
    - config values loaded before `appStartup()`
    - `m_*` state in `stdCamera`
    - INDI property creation in `appStartup()`
    - callback dispatch in `newCallBack_stdCamera()`
    - `updateINDI()` publishing
    - `recordCamera()` telemetry packing
- `apps/pvcamCtrl/pvcamCtrl.hpp`
  - `pvcamCtrl` already uses `stdCamera` readout-speed support, which is the closest analogue for a fan-speed selection switch.
  - `setupConfig()` and `loadConfig()` happen before `STDCAMERA_APP_STARTUP`, so config is the right place to decide whether to expose the fan interface and how many speeds to advertise.
- PVCAM manual
  - `PARAM_FAN_SPEED_SETPOINT` is the relevant parameter.
  - The documented enum values are `FAN_SPEED_HIGH`, `FAN_SPEED_MEDIUM`, `FAN_SPEED_LOW`, and `FAN_SPEED_OFF`.
  - The parameter is documented as "Sets and gets the desired fan speed."
- `libMagAOX/logger/types/schemas/telem_stdcam.fbs` and `libMagAOX/logger/types/telem_stdcam.hpp`
  - `telem_stdcam` does not currently include any fan field.

## Recommended Design

### 1. Extend `dev::stdCamera` with an optional fan-speed interface

Add a new optional feature in `libMagAOX/app/dev/stdCamera.hpp` following the existing readout-speed pattern.

Recommended interface pieces:

- New compile-time flag in derived apps:
  - `static constexpr bool c_stdCamera_fanSpeed = true/false;`
- New config-backed state in `stdCamera`:
  - `bool m_fanSpeedControlEnabled{ false };`
  - `std::string m_defaultFanSpeed;`
- New `stdCamera` state:
  - `std::vector<std::string> m_fanSpeedNames;`
  - `std::vector<std::string> m_fanSpeedNameLabels;`
  - `std::string m_fanSpeedName;`
  - `std::string m_fanSpeedNameSet;`
  - `pcf::IndiProperty m_indiP_fanSpeed;`
- New derived-app interface:
  - `int setFanSpeed();`

Implementation should mirror `readoutSpeed`, but its exposure should be config-driven:

- in `setupConfig()`, add config entries only when `c_stdCamera_fanSpeed == true`
- in `loadConfig()`, load whether fan control is enabled and how many speeds should be exposed
- only create the INDI switch in `appStartup()` when config says the interface is enabled
- create a one-of-many INDI switch named something like `fan_speed`
- on callback, determine the selected element and store it in `m_fanSpeedNameSet`
- call `derived().setFanSpeed()`
- in `updateINDI()`, publish `m_fanSpeedName` through `indi::updateSelectionSwitchIfChanged()`
- in `recordCamera()`, include the current fan setting and trigger telemetry when it changes

Recommended config keys:

- `camera.fanSpeedControl`
  - bool
  - whether the app should expose fan speed control at all
- `camera.defaultFanSpeed`
  - string
  - the default fan speed to apply at startup or power-on
  - valid values should be `high`, `medium`, `low`, or `off`
  - invalid values should hard-fail config load

The exposed choices should always be the full PVCAM set:

- `high`
- `medium`
- `low`
- `off`

This keeps the exposure decision in config instead of runtime hardware probing, while avoiding per-deployment variation in the number of advertised speeds.

### 2. Add fan-speed telemetry to `telem_stdcam`

Update:

- `libMagAOX/logger/types/schemas/telem_stdcam.fbs`
- `libMagAOX/logger/types/telem_stdcam.hpp`

Recommended payload shape:

- add a single string field for the active fan setting

Rationale:

- `stdCamera` currently tracks human-readable selection names for similar controls.
- PVCAM documents `PARAM_FAN_SPEED_SETPOINT` as the desired fan speed, so a named state is the most natural fit.
- This keeps telemetry changes small and avoids tying `telem_stdcam` to PVCAM-specific integer enum values.

After changing the schema, regenerate the corresponding FlatBuffer-generated header if that is part of the normal workflow in this tree.

### 3. Implement the PVCAM mapping in `pvcamCtrl`

Update `apps/pvcamCtrl/pvcamCtrl.hpp` to:

- enable the new interface with `c_stdCamera_fanSpeed = true`
- declare and document `setFanSpeed()`
- use `stdCamera` config for enable/default behavior
- initialize the fan speed names and labels before `appStartup()` to the full four-speed PVCAM set
- initialize the current/set values in `powerOnDefaults()`
- optionally read back the current fan setpoint after connect to synchronize status, but not to decide whether the interface exists
- call `pl_set_param( m_handle, PARAM_FAN_SPEED_SETPOINT, ... )` in `setFanSpeed()`

Recommended canonical PVCAM mapping:

- `high` -> `FAN_SPEED_HIGH`
- `medium` -> `FAN_SPEED_MEDIUM`
- `low` -> `FAN_SPEED_LOW`
- `off` -> `FAN_SPEED_OFF`

Recommended hook points:

- `loadConfigImpl()`
  - rely on `stdCamera` to load `camera.fanSpeedControl` and `camera.defaultFanSpeed`
  - populate `m_fanSpeedNames` and `m_fanSpeedNameLabels` with all four fan speeds
- `powerOnDefaults()`
  - restore `m_fanSpeedName` and `m_fanSpeedNameSet` from `m_defaultFanSpeed`
- `connect()`
  - optionally read `ATTR_CURRENT` to synchronize telemetry and INDI status
- `setFanSpeed()`
  - validate the requested string
  - map it to the PVCAM enum value
  - set the parameter
  - update `m_fanSpeedName`
  - force a telemetry record

## Configuration Model

The interface should be controlled at two levels:

- compile time
  - `c_stdCamera_fanSpeed` says the app knows how to implement fan control
- config time
  - config says whether the feature should be exposed for this deployment
  - config says which of the four levels is the default

Under this model:

- `stdCamera` does not need runtime capability discovery logic for deciding whether to create the property
- `pvcamCtrl` does not need runtime checks to decide whether fan control exists
- the user is responsible for matching config to hardware
- runtime PVCAM failures should still be logged normally, but they are operational failures, not feature-discovery logic

This is closer to how the current `stdCamera` options are structured and keeps the interface deterministic across startup.

## Concrete Implementation Steps

1. Extend `stdCamera.hpp`
   - add the new compile-time flag documentation
   - add config entries for fan-control enable/default
   - add config-backed enabled/default member state
   - add fan-speed member data and INDI property
   - add creation, callback, dispatch, and update helpers
   - gate property creation and updates on both compile-time support and config enable
   - hard-fail `loadConfig()` if `camera.defaultFanSpeed` is not one of `high`, `medium`, `low`, or `off`
   - add fan-speed state to `recordCamera()`

2. Extend `telem_stdcam`
   - add a fan-speed field to the FlatBuffer schema
   - update `messageT`, formatting, and metadata accessors in `telem_stdcam.hpp`
   - regenerate generated FlatBuffer code if required

3. Update `pvcamCtrl`
   - enable the feature flag
   - document and declare `setFanSpeed()`
   - build names/labels from the fixed four-speed PVCAM set before `STDCAMERA_APP_STARTUP`
   - optionally synchronize current state in `connect()`
   - implement the PVCAM enum mapping and setter

4. Verify behavior
   - app starts with fan control hidden when config disables it
   - app starts with all four fan-speed choices when enabled
   - app starts with the configured default selected
   - selecting each exposed INDI fan level updates internal state
   - `pl_set_param(PARAM_FAN_SPEED_SETPOINT, ...)` is called with the expected PVCAM enum
   - telemetry changes when fan level changes
   - misconfigured hardware fails in a clear logged way

## Validation Notes

- The closest existing code path to model after is readout-speed selection in `stdCamera`.
- The main difference from the previous draft is that exposure is decided by config, not by runtime capability probing.
- Because this is a potential safety issue, invalid `camera.defaultFanSpeed` should be treated as a fatal configuration error, not silently corrected.
- Telemetry verification should include both:
  - forced records after a set operation
  - change-driven records during steady-state operation
- If the generated FlatBuffer headers are committed in this repo, they should be updated in the same functional change.

## Affected Files

Expected primary touch points:

- `agents/plans/pvcam-stdcamera-fan-ctrl.md`
- `libMagAOX/app/dev/stdCamera.hpp`
- `libMagAOX/logger/types/schemas/telem_stdcam.fbs`
- `libMagAOX/logger/types/telem_stdcam.hpp`
- generated FlatBuffer output for `telem_stdcam`, if tracked
- `apps/pvcamCtrl/pvcamCtrl.hpp`

## Follow-Up Questions

- None at the planning level. The remaining implementation detail is to make the hard-fail path in `stdCamera::loadConfig()` produce a clear configuration error message naming the invalid value.
