# Prompt
Create a plan in an orcaCtrl-liquidCooling.md file to add a boolean option that will switch the camera from fan cooling to liquid cooling. First, find the DCAM API parameter that turns the cooling fan off and also find the parameter that switches the camera to liquid cooling mode. Create this boolean option such that it is consistent with the stdCam cooling implementation. After writing out this plan, following conventions of agents/plans/2026-09, commit it to my GitHub branch

# Plan

## Analysis

### DCAM parameters

Searched the DCAM SDK headers used by `apps/orcaCtrl/Makefile`
(`/opt/hamamatsu_new/hamamatsu_sdk/dcamsdk4/inc/dcamprop.h` and `dcamapi4.h`) for cooler, fan, water
and liquid properties.

- **Cooling fan off:** `DCAM_IDPROP_SENSORCOOLERFAN = 0x00200350` (`R/W, mode, "SENSOR COOLER FAN"`).
  The header defines no fan-specific enum values. It is a `mode` property, so the expected values are
  the generic `DCAMPROP_MODE__OFF = 1` and `DCAMPROP_MODE__ON = 2`. These must be confirmed on the
  camera with `dcamprop_getattr` and `dcamprop_getvaluetext` (step 1).
- **Liquid cooling mode:** the SDK headers have **no dedicated liquid- or water-cooling property**. The
  only other cooler properties are:
  - `DCAM_IDPROP_SENSORCOOLER = 0x00200320` (`R/W, mode, "SENSOR COOLER"`), with values
    `DCAMPROP_SENSORCOOLER__OFF = 1`, `__ON = 2` and `__MAX = 4`. This switches the thermoelectric
    sensor cooler, not the heat-removal method.
  - `DCAM_IDPROP_SENSORCOOLERSTATUS = 0x00200340` (read-only), `DCAM_IDPROP_SENSORTEMPERATURETARGET`,
    and the `DCAM_IDPROP_SENSORTEMPERATURE*` readbacks.
- Working assumption: on the ORCA-Quest2, liquid-cooled operation means **turning the cooling fan off
  with `DCAM_IDPROP_SENSORCOOLERFAN` while the external chiller supplies coolant.** There is no separate
  mode switch. `DCAM_IDPROP_SENSORCOOLER = __MAX` is a possible camera-specific extra that has to be
  checked against the ORCA-Quest2 instruction manual and a property enumeration on the camera before
  using it.

### stdCamera cooling implementation to stay consistent with

`libMagAOX/app/dev/stdCamera.hpp` exposes fan control when the app sets `c_stdCamera_fanSpeed = true`.
orcaCtrl already sets this.

- The config keys live in the `camera` section: `camera.fanSpeedControl` (bool, default `true`, decides
  whether the INDI `fan_speed` switch is published) and `camera.defaultFanSpeed`, which must be one of
  `m_fanSpeedNames` or config loading fails.
- Behavior is set by config at load time, not by probing the hardware (see
  `agents/plans/2026-03/pvcam-stdcamera-fan-ctrl.md`). The user is responsible for matching config to
  the hardware.
- The app maps `m_fanSpeedNameSet` to the hardware in `setFanSpeed()` or the next
  `configureAcquisition()`, keeps `m_fanSpeedName` and `m_fanSpeedValid` current, and logs the startup
  and change notices through `m_fanSpeedLogPending`.
- `telem_stdcam` already records the fan state, so a forced-off fan shows up in telemetry with no schema
  change.

So the new option is a boolean config key, `camera.liquidCooling`, in the same `camera` section. It is
loaded next to the stdCamera fan keys and implemented by driving the existing stdCamera fan state. That
reuses the INDI, logging and telemetry paths rather than adding a parallel property.

### Existing orcaCtrl fan code, which the option depends on

orcaCtrl's fan code was copied from picamCtrl and doesn't yet drive the DCAM property correctly.
`camera.liquidCooling` depends on it, so these need fixing first:

1. `configureAcquisition()` (~line 1315) writes `c_enableCoolingFan = 0` and `c_disableCoolingFan = 1`
   to `DCAM_IDPROP_SENSORCOOLERFAN`. That is PICam's `DisableCoolingFan` convention. For a DCAM `mode`
   property, `0` is invalid and `1` means OFF, so asking for "on" writes an invalid value.
2. `getFanSpeed()` (~line 1002) reads `DCAM_IDPROP_SENSORCOOLER`, the sensor cooler, and compares it to
   `DCAMPROP_SENSORCOOLER__*`. It should read `DCAM_IDPROP_SENSORCOOLERFAN`.
3. `m_fanStatusSupported` is never set to `true`, so `getFanSpeed()` never runs from `appLogic()` and the
   fan state is never read back from the camera.
4. `connect()` (~line 881) logs "Cooling fan control is enabled in config but not supported" whenever
   the property isn't writable, even when `camera.fanSpeedControl` is false.

## Implementation Plan

1. **Check the DCAM properties on the camera.**
   - Write a short standalone probe (in the scratchpad, not committed) that opens the camera and prints
     `dcamprop_getattr` (writable flag, min and max) and every `dcamprop_getvaluetext` value for
     `DCAM_IDPROP_SENSORCOOLERFAN` and `DCAM_IDPROP_SENSORCOOLER`. Use `dcamprop_getnextid` to list any
     other cooler-related properties.
   - Run it only while `orcaCtrl` is stopped, because both need exclusive access to the DCAM device.
   - Record the results in this plan's Execution Notes. If a real liquid-cooling property or value turns
     up, update the design before writing code.

2. **Fix the DCAM fan mapping in orcaCtrl.** This is a functional commit.
   - Replace `c_enableCoolingFan` and `c_disableCoolingFan` with `DCAMPROP_MODE__ON` and
     `DCAMPROP_MODE__OFF`, or whatever values step 1 finds.
   - Make `getFanSpeed()` read `DCAM_IDPROP_SENSORCOOLERFAN` and map ON/OFF to `"on"`/`"off"`.
   - In `connect()`, set `m_fanStatusSupported` from the property's readable attribute, and log the
     unsupported message only when `m_fanSpeedControlEnabled` is true.

3. **Add the `camera.liquidCooling` option.** This is a functional commit.
   - New member, documented per rule 4:
     `bool m_liquidCooling{ false }; ///< True when the camera is liquid cooled, so the cooling fan is held off.`
     Put it in the "configurable parameters" group next to `m_serialNumber`.
   - `setupConfig()`: add
     `config.add( "camera.liquidCooling", "", "camera.liquidCooling", argType::Required, "camera", "liquidCooling", false, "bool", "If true the camera is liquid cooled and the cooling fan is turned off. Default is false (fan cooled)." );`
   - `loadConfigImpl()`: read it after `STDCAMERA_LOAD_CONFIG`, so it can override the stdCamera fan
     defaults:
     - When `m_liquidCooling` is true, set `m_defaultFanSpeed = "off"` and
       `m_fanSpeedControlEnabled = false`. The INDI `fan_speed` switch is then hidden, so an operator can't
       turn the fan back on.
     - If `camera.defaultFanSpeed` was set to `"on"` in the same config, log a warning that
       `camera.liquidCooling` overrides it.
   - `powerOnDefaults()` and `connect()`: when `m_liquidCooling` is true, set `m_fanSpeedNameSet = "off"`
     and `m_fanSpeedLogPending = true` whether or not `m_fanSpeedControlEnabled` is set, so every
     power-on and reconnect drives the fan off.
   - `configureAcquisition()`: change the fan block's condition from
     `m_fanSpeedControlEnabled && m_fanControlSupported` to
     `( m_fanSpeedControlEnabled || m_liquidCooling ) && m_fanControlSupported`. The existing block
     then writes `DCAM_IDPROP_SENSORCOOLERFAN = OFF` and logs "fan speed set to 'off'".
   - Fail safe: if `m_liquidCooling` is true but the camera reports the fan property isn't writable, log
     `LOG_CRITICAL` and go to `stateCodes::ERROR`. Don't run silently with the fan in an unknown state.
   - Log the cooling mode once at startup ("cooling mode: liquid (fan off)" or "cooling mode: fan"), in
     the same place as the existing fan startup notice.
   - Don't touch `DCAM_IDPROP_SENSORCOOLER` unless step 1 shows the Quest2 needs it for liquid cooling.

4. **Config and documentation.**
   - Add `liquidCooling=false` with a comment to the orcaCtrl config template, if one exists in the
     config repo. The deployment config for the liquid-cooled camera sets `liquidCooling=true`.
   - Rule 15 applies: document the new member, and update the class documentation to describe
     `camera.liquidCooling` and how it interacts with `camera.fanSpeedControl` and
     `camera.defaultFanSpeed`.

5. **Tests** (rule 20).
   - Add `apps/orcaCtrl/tests/orcaCtrl_test.cpp` under `libXWCTest::orcaCtrlTest`, with a
     `\defgroup orcaCtrl_unit_test` that `\ingroup application_unit_test` in `tests/groups.dox`.
   - Use `loadConfigImpl()` to check that `camera.liquidCooling=true` forces `m_defaultFanSpeed == "off"`
     and `m_fanSpeedControlEnabled == false`, and that the default (`false`) leaves the stdCamera fan
     config alone.
   - The DCAM calls can't run without hardware or a DCAM mock, so the tests only cover config handling.
     Add `\ref` links per rule 21 if the harness hides the real symbols from Doxygen.

6. **Verify and commit.**
   - Format with the cpptools clang-format and build with `gcc-toolset-14`.
   - On hardware, with the chiller running: start with `liquidCooling=true`, then confirm the fan reads
     back `off` in INDI and telemetry, `fan_speed` isn't published, and `SENSORCOOLERSTATUS` reaches
     READY/LOCKED.
   - Commit order (rule 19): step 2 fan-mapping fix, then the step 3 option, then tests, then the docs
     pass, then formatting if needed. Update this plan's Execution Notes in each commit.

## Open Questions / Assumptions

- **No liquid-cooling DCAM property exists** in the installed SDK headers. The plan assumes that fan off
  plus the external chiller is the whole switch, and step 1 plus the ORCA-Quest2 manual must confirm
  that. If Hamamatsu documents a camera-specific property or a `SENSORCOOLER` value such as `MAX` for
  water cooling, step 3 will write it too.
- **Safety:** turning the fan off without coolant flowing can overheat the camera. This plan relies on
  config being correct, as stdCamera does, and on the camera's own thermal protection
  (`SENSORCOOLERSTATUS__WARNING` is already logged as `LOG_ALERT` in `getTemps()`). Should
  `liquidCooling=true` also check that the chiller is running, for example through another app's INDI
  property? The plan assumes not.
- **Config file vs runtime:** the option is config-only, like `camera.fanSpeedControl`. Switching between
  air and liquid cooling is a hardware change, so it shouldn't be an INDI toggle.
- **Temperature setpoint:** `m_ccdTempSetpt = -35` is hard-coded, and the
  `SENSORTEMPERATURETARGET` write is commented out. If the Quest2's reachable sensor temperature depends
  on the cooling mode, that should be handled separately. This plan doesn't change it.

## Execution Notes

- Plan only. Nothing is implemented yet, and step 1 (the hardware check) hasn't been run.
- Found the DCAM fan property and confirmed that no liquid-cooling property exists by searching
  `dcamprop.h` and `dcamapi4.h` in `/opt/hamamatsu_new/hamamatsu_sdk/dcamsdk4/inc/`.
- Checked the stdCamera fan config pattern in `libMagAOX/app/dev/stdCamera.hpp` (`camera.fanSpeedControl`,
  `camera.defaultFanSpeed`) and the earlier plans `agents/plans/2026-03/pvcam-stdcamera-fan-ctrl.md` and
  `agents/plans/2026-04/picam-cooling-fan-control.md`.
- While reviewing, found the four fan bugs listed under "Existing orcaCtrl fan code"; step 2 fixes them.
