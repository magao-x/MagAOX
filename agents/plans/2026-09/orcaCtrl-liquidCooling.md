# Prompt
Create a plan in an orcaCtrl-liquidCooling.md file to add a boolean option that will switch the camera from fan cooling to liquid cooling. First, find the DCAM API parameter that turns the cooling fan off and also find the parameter that switches the camera to liquid cooling mode. Create this boolean option such that it is consistent with the stdCam cooling implementation. After writing out this plan, following conventions of agents/plans/2026-09, commit it to my GitHub branch

Follow-up prompt:

Update the plan so that SensorCooler is set to 'MAX' when liquid cooling is enabled

Follow-up prompt:

liquidCooling=True should also check that the chiller is running. The temperature setpoint should be getting written out to the "SENSORTEMPERATURETARGET" prop if it isn't doing so already. Also, check these changes against the relevant cooling settings in "/home/jliberman/source/MagAOX-ham/ham_doc.md" to see if there is a DCAM configurator option for changing to liquid cooling mode

Follow-up prompt:

Yes, orcaCtrl should stop acquisition when the chiller flow is lost. A koolanceCtrl-style INDI app will publish the chiller status for the camera. Commit and push the updated plan along with the Q and A

Follow-up prompt:

I am having an issue where I power on the orcaQuest 2 camera powered by the orcaCtrl app and, despite the camera being set in liquid cooling mode, the temp_ccd INDI property is not decreasing but rather is increasing and reaching 29 C. The target temp is -35C  and I cannot toggle camscc.temp_controller. I would expect the camera temperature to decrease, as I am running glycol through the camera. Please investigate the source of this reported temperature issue in the orcaCtrl app and add your findings to orcaCtrl-liquidCooling.md

Follow-up prompt:

Implement step 4 of the plan. Cooling flow is started before the camera turns on

Follow-up prompt:

Don't poll the chiller flow rate as I do not have a way of monitoring it

# Plan

## Analysis

### What the ORCA-Quest2 manual says (`ham_doc.md`)

`ham_doc.md` is the ORCA-Quest 2 (C15550-22UP / C15550-22UP01) instruction manual, Ver. 1.1.

- **The cooling method is a stored camera setting, changed with DCAM Configurator, not with a DCAM API
  property.** Section 8-1(1) says: "The default setting of cooling method is Air-cooling. Cooling mode
  can be changed by software which is called, 'DCAM Configurator'." Section 9-2 says: "After cooling
  mode was changed, the camera memorizes the last setting as the default setting for cooling." Section
  9-5 says: "Even if the camera's power supply is turned off, the state of setting is kept."
- **Air cooling:** "When the camera is turned on, the fan starts rotating and cooling is started."
- **Water cooling:** "Cooling does not start just turning on the camera. Cooling water circulation must
  be started before start operating the camera in water-cooling. A fan inside the camera does not
  rotate." Section 9-2-2 gives the startup order: turn on the chiller, check the water is circulating,
  turn on the camera, then "Turn on the cooling switch of the camera from application software". That
  last step is `DCAM_IDPROP_SENSORCOOLER`.
- **Chiller requirements** (section 8-1(6)): "Confirm the water is flowing before starting the camera
  cooling ... Keep 0.45 L/min flow rate ... Do not stop the circulating water cooler while the camera is
  working." Hamamatsu recommends 25 °C water (8-1(3)), and warns about condensation in warm, humid
  conditions (Graph 8-1).
- **Cooling temperatures** (section 13 specifications): forced-air cooled -20 °C; water cooled -20 °C
  (25 °C water); **water cooled (Max cooling) -35 °C** (20 °C water, typical). The overview says -35 °C
  is reached "when switched to Max cooling". So MAX is a water-cooling feature, and the current
  hard-coded -35 °C setpoint can only be reached in liquid-cooled MAX mode.
- **Protection:** if the camera overheats, the thermal protection circuit sounds a buzzer, lights the
  red LED and cuts power to the Peltier element (section 9-1(3)).
- The manual never mentions a user-adjustable temperature setpoint. It lists only the fixed values above.

### DCAM Configurator on this machine: `dcamcfgc`

The Linux DCAM install includes a command-line Configurator,
`/opt/hamamatsu_new/hamamatsu_api/tools/x86_64/dcamcfgc` ("DCAM Configurator ver 24.12.6898"). Its
strings show it supports the `C15550-22UP`. It is interactive: pick a camera index, then a parameter
index, then a value index, where 0 exits. After a change it prints "Changed value. Please restart
camera." It reads and writes camera registers through `dcamdev_getdata`/`dcamdev_setdata`, not through
`dcamprop_*`.

Cooling-related parameters in its menu (which ones it shows depends on the camera model):

| dcamcfgc parameter | Values seen in the binary | Relevance |
| --- | --- | --- |
| `Cooler Type` | `Air(Standard)`, `Air(Rapid)`, `Air(Off)`, `Water` | **This is the liquid-cooling mode switch.** |
| `Water Cooling Target` | likely among `-20`, `-30`, `MAX` | Target used in water mode |
| `Target Temperature` | likely among `-20`, `-10`, `-30` | Stored default target |
| `Sensor Cooler` | `OFF`, `ON`, `MAX` | Stored default for the cooler switch |
| `Default Fan Control` / `Cooler FAN` | `ON`, `OFF` | Stored fan defaults |

The value-to-parameter mapping above comes from the string table and **must be confirmed by running
`dcamcfgc` read-only on the camera** (step 1).

**Conclusion: the switch to liquid cooling is the `dcamcfgc` "Cooler Type = Water" setting.** It is
persistent, needs a camera restart, and there is no `dcamprop` equivalent in the SDK headers. orcaCtrl
should therefore not try to switch the mode itself. `camera.liquidCooling` instead **declares** the
installed cooling method. orcaCtrl checks that the camera matches, checks that the chiller is running,
and then applies the per-session settings through `dcamprop`: fan off, `SENSORCOOLER` MAX, and the
temperature target.

### DCAM API properties (`dcamprop.h`)

- `DCAM_IDPROP_SENSORCOOLERFAN = 0x00200350` (`R/W, mode`). The generic values are
  `DCAMPROP_MODE__OFF = 1` and `DCAMPROP_MODE__ON = 2`. In water mode the fan is already off, and
  writing OFF is a safe extra step.
- `DCAM_IDPROP_SENSORCOOLER = 0x00200320` (`R/W, mode`): `__OFF = 1`, `__ON = 2`, `__MAX = 4` (`BEST =
  3` is reserved). This is the "cooling switch" from manual section 9-2-2. Per the first follow-up
  prompt, liquid cooling writes **MAX**. Air cooling writes **ON**.
- `DCAM_IDPROP_SENSORTEMPERATURETARGET = 0x00200330` (`R/W, celsius`). Declared writable in the header,
  but the Quest2 may report it read-only or restrict its range, because the manual lists fixed targets
  and `dcamcfgc` stores them. Step 1 checks `dcamprop_getattr`.
- `DCAM_IDPROP_SENSORCOOLERSTATUS = 0x00200340` (read-only): `OFF`, `READY`, `BUSY`, `ALWAYS`, `WARNING`,
  `ERROR1-4`.
- No `dcamprop` property reports the Cooler Type directly. Possible signs of water mode, all to be
  confirmed in step 1: `MAX` appearing in the valid values of `SENSORCOOLER` (from
  `dcamprop_queryvalue`), `SENSORCOOLERFAN` reading back OFF right after power-on, and
  `SENSORCOOLERSTATUS` reading OFF at power-on instead of starting to cool.

### Temperature setpoint today

It is **not written to the camera**.

- `setTempSetPt()` (~line 1100) only calls `recordCamera()` and sets `m_reconfig = true`, and it still
  has "`\todo bounds check here`".
- The `setorcaParameter( DCAM_IDPROP_SENSORTEMPERATURETARGET, m_ccdTempSetpt )` block in
  `configureAcquisition()` (~line 1370) is commented out, but the next line still logs "Set temperature
  set point: ... C".
- `powerOnDefaults()` hard-codes `m_ccdTempSetpt = -35`, and `appStartup()` hard-codes `m_minTemp = -35`
  and `m_maxTemp = 25`.
- stdCamera already has `camera.startupTemp`: after power-on, a value above -999 overrides
  `m_ccdTempSetpt`.

### Chiller status available over INDI

`apps/koolanceCtrl` publishes the INDI property `<device>.status` with the elements `liquid_temp` (°C),
`flow_rate` (L/min), `pump_rpm` and `fan_rpm`. MagAO-X apps subscribe to another device's property with
`REG_INDI_SETPROP( prop, device, name )` plus a `setCallBack`; `closedLoopIndi` is an example. That
gives orcaCtrl a live flow-rate reading to check against the manual's 0.45 L/min minimum.

### stdCamera cooling implementation to stay consistent with

- stdCamera fan control (`c_stdCamera_fanSpeed = true`, already set in orcaCtrl) has config keys in the
  `camera` section: `camera.fanSpeedControl` (bool, default `true`) and `camera.defaultFanSpeed`.
  Behavior is set by config at load time, and the user matches config to the hardware
  (`agents/plans/2026-03/pvcam-stdcamera-fan-ctrl.md`).
- Temperature control (`c_stdCamera_tempControl = true`) routes INDI setpoint requests to
  `setTempSetPt()` and applies `camera.startupTemp` after power-on.
- `telem_stdcam` already records the fan state and the temperature, setpoint and status.

So `camera.liquidCooling` is a boolean in the `camera` section, loaded next to the stdCamera fan and
temperature keys, and it drives the existing stdCamera fan and temperature state. The chiller settings
go in a new `chiller` config section, following the MagAO-X habit of one section per subsystem.

### Existing orcaCtrl fan code, which the option depends on

1. `configureAcquisition()` (~line 1315) writes `c_enableCoolingFan = 0` and `c_disableCoolingFan = 1`
   to `DCAM_IDPROP_SENSORCOOLERFAN`, which is PICam's convention. For DCAM, `0` is invalid and `1` is
   OFF.
2. `getFanSpeed()` (~line 1002) reads `DCAM_IDPROP_SENSORCOOLER` instead of `DCAM_IDPROP_SENSORCOOLERFAN`.
   Once the cooler is at `MAX` (4), it takes the "Unknown cooling-fan status" path, returns -1 on every
   `appLogic()` pass, and puts the app in `ERROR`.
3. `m_fanStatusSupported` is never set to `true`, so `getFanSpeed()` never runs. A temporary
   working-tree change forced both fan flags to `true` in `powerOnDefaults()`. The user reverted it to
   `false` on 2026-09-30. Any change that turns on `getFanSpeed()` needs item 2 fixed first, or in the
   same commit.
4. `connect()` (~line 881) logs "Cooling fan control is enabled in config but not supported" even when
   `camera.fanSpeedControl` is false.

## Implementation Plan

1. **Check the camera's settings, read-only, with orcaCtrl stopped.**
   - Run `dcamcfgc`. Select the Quest2, then for each of `Cooler Type`, `Water Cooling Target`, `Target
     Temperature`, `Sensor Cooler`, `Default Fan Control` and `Cooler FAN`, open the value list, note the
     current value (marked `(*)`) and the options, and press **0 to exit without changing anything**.
     Record the results in Execution Notes.
   - Write a short standalone `dcamprop` probe in the scratchpad (not committed). For
     `SENSORCOOLERFAN`, `SENSORCOOLER`, `SENSORTEMPERATURETARGET` and `SENSORCOOLERSTATUS`, it prints
     `dcamprop_getattr` (writable and readable flags, min, max, step), every value from
     `dcamprop_queryvalue` with `DCAMPROP_OPTION_NEXT`, and `dcamprop_getvaluetext` for each value. It
     also lists any other cooler properties with `dcamprop_getnextid`.
   - If the camera can be switched safely, with the chiller connected and running: set `Cooler Type =
     Water` with `dcamcfgc`, restart the camera, and repeat the probe. Comparing air and water readouts
     tells us which `dcamprop` signal identifies water mode (the MAX option, fan readback, or cooler
     status). The check in step 4 depends on this.

2. **Fix the DCAM fan mapping.** This is a functional commit.
   - Use `DCAMPROP_MODE__ON`/`__OFF`, or whatever values step 1 finds, instead of 0 and 1.
   - `getFanSpeed()` reads `DCAM_IDPROP_SENSORCOOLERFAN`.
   - `connect()` sets `m_fanControlSupported` and `m_fanStatusSupported` from the property's
     writable/readable attributes, and logs "unsupported" only when `m_fanSpeedControlEnabled` is true.
     `powerOnDefaults()` keeps resetting both flags to `false`.

3. **Write the temperature setpoint to `SENSORTEMPERATURETARGET`.** This is a functional commit, and it
   applies to both cooling modes.
   - In `connect()`, `dcamprop_getattr( DCAM_IDPROP_SENSORTEMPERATURETARGET )` sets a new documented
     member, `bool m_tempTargetWritable{ false }; ///< True when SENSORTEMPERATURETARGET is writable on
     this camera.` If the property is writable, also set `m_minTemp`, `m_maxTemp` and `m_stepTemp` from
     `valuemin`, `valuemax` and `valuestep`, replacing the hard-coded -35/25 in `appStartup()`, and update
     the INDI `temp_ccd` limits.
   - `setTempSetPt()` does the bounds check from its `\todo`. It clamps `m_ccdTempSetpt` to
     `[m_minTemp, m_maxTemp]` and logs if it clamps. If the target is writable, it writes the value right
     away with `setorcaParameterOnline( DCAM_IDPROP_SENSORTEMPERATURETARGET, m_ccdTempSetpt )`, so the
     change doesn't need an acquisition restart, and it stops setting `m_reconfig`. If step 1 shows the
     property can't be written during capture, it keeps the `m_reconfig` path instead.
   - `configureAcquisition()`: uncomment the `SENSORTEMPERATURETARGET` block and write
     `m_ccdTempSetpt` whenever `m_tempTargetWritable` is true. Read the value back with
     `getorcaParameter()` into `m_ccdTempSetpt` and log what was actually applied. Remove the log line
     that currently reports a setpoint that was never written.
   - If the target is read-only: read it into `m_ccdTempSetpt` in `connect()`, log once that the target is
     fixed by the camera's `dcamcfgc` "Target Temperature" / "Water Cooling Target" setting, and have
     `setTempSetPt()` reject INDI changes with a `LOG_WARNING`, putting the INDI `target` back to the
     actual value.
   - Default setpoint: `powerOnDefaults()` uses -35 °C when `m_liquidCooling` is true and -20 °C when it
     is false, matching the manual's specifications. `camera.startupTemp` still overrides this, as
     stdCamera already does.

4. **Add `camera.liquidCooling` and the chiller check.** This is a functional commit.
   - New config, documented members, and `setupConfig()` entries:
     - `camera.liquidCooling` (bool, default `false`). Help text: "Set true when the camera's Cooler
       Type has been set to Water with dcamcfgc and a chiller is connected. orcaCtrl then holds the fan
       off, sets the sensor cooler to MAX and requires chiller flow."
       Member: `bool m_liquidCooling{ false }; ///< True when the camera is configured and plumbed for liquid cooling.`
     - `chiller.device` (string, default `""`). The INDI device publishing chiller status, for example the
       `koolanceCtrl` instance. Required when `camera.liquidCooling` is true: `loadConfigImpl()` returns
       -1 if it is empty.
     - `chiller.property` (default `"status"`) and `chiller.flowElement` (default `"flow_rate"`), so a
       chiller app other than koolanceCtrl can be used.
     - `chiller.minFlowRate` (float, L/min, default `0.45`, the manual's minimum).
     - `chiller.timeout` (float, seconds, default `10`). A reading older than this counts as "not running".
   - `loadConfigImpl()`: read these after `STDCAMERA_LOAD_CONFIG`. When `m_liquidCooling` is true, set
     `m_defaultFanSpeed = "off"` and `m_fanSpeedControlEnabled = false`, so the INDI `fan_speed` switch is
     hidden. Log a warning if `camera.defaultFanSpeed = on` was also set.
   - Chiller subscription, only when `m_liquidCooling` is true: in `appStartup()`,
     `REG_INDI_SETPROP( m_indiP_chillerStatus, m_chillerDevice, m_chillerProperty )`. The `setCallBack`
     stores `m_chillerFlowRate` and the time of the last update, under `m_indiMutex`. A new helper
     `bool chillerRunning()` returns true only if a reading has arrived, it is newer than
     `chiller.timeout`, and flow ≥ `chiller.minFlowRate`. Document the members and helper per rules 3
     and 4.
   - Publish a read-only INDI property `cooling` with the elements `mode` (`"liquid"`/`"air"`),
     `chiller_ok` (bool) and `chiller_flow` (L/min), so operators and the GUI can see why cooling is or
     isn't active.
   - Check the camera matches, in `connect()`: use the water-mode signal found in step 1. If
     `m_liquidCooling` is true but the camera reports air mode, or the reverse, log `LOG_CRITICAL` ("Cooler
     Type mismatch: run dcamcfgc and set Cooler Type = Water, then restart the camera") and go to
     `stateCodes::ERROR`. orcaCtrl never switches Cooler Type itself.
   - Start cooling only when the chiller is running, in `configureAcquisition()`, after the fan block:
     - Liquid cooling: if `chillerRunning()` is false, write `SENSORCOOLER = OFF`, set
       `m_chillerFault`, log `LOG_WARNING` ("waiting for chiller flow"), and return without starting
       acquisition. This follows the manual: "confirm the water is flowing before starting the camera
       cooling." If it is true, write `SENSORCOOLERFAN = OFF`, then `SENSORCOOLER = MAX`, then the
       temperature target (step 3).
     - Air cooling: write `SENSORCOOLER = ON`, so a stale MAX left over from liquid use is cleared.
     - If MAX isn't in the valid values, write `ON` with a `LOG_WARNING`. MAX gives better cooling but
       isn't needed for safety.
     - Read the cooler mode back into `int32 m_sensorCoolerMode{ 0 }; ///< Sensor cooler mode last
       applied (DCAMPROP_SENSORCOOLER__*).` and log a `LOG_NOTICE` when it changes.
   - Keep checking while the camera runs, in `appLogic()` READY/OPERATING, next to `getTemps()`. If
     `m_liquidCooling` is true and `chillerRunning()` becomes false, **turn cooling off and stop
     acquisition**:
     - Turn the Peltier off right away with `setorcaParameterOnline( DCAM_IDPROP_SENSORCOOLER, OFF )`.
     - Set a new documented member, `bool m_chillerFault{ false }; ///< True while chiller flow is lost;
       blocks acquisition and cooling until flow is restored.`, and set `m_reconfig = true`. The
       framegrabber then leaves its loop and calls `reconfig()`, which runs `dcamcap_stop` and releases
       the buffers.
     - While `m_chillerFault` is set, `configureAcquisition()` returns before allocating buffers or
       calling `dcamcap_start`, and leaves `SENSORCOOLER = OFF`. `getAcquisitionState()` doesn't log
       "acquisition stopped. restarting" or set `m_reconfig` while the fault is set, so it doesn't fight
       the stop.
     - Log `LOG_ALERT` ("chiller flow lost: sensor cooler off, acquisition stopped"), set
       `m_tempControlStatusStr = "NO CHILLER"`, update the INDI `cooling` property, and call
       `recordCamera()`. The app state stays `READY`, not `ERROR`, so it keeps polling the chiller.
     - When `chillerRunning()` is true again, clear `m_chillerFault`, log `LOG_NOTICE` ("chiller flow
       restored, restarting cooling and acquisition"), and set `m_reconfig = true`.
       `configureAcquisition()` then restarts cooling (fan off, MAX, target) and acquisition.
     - The same gate applies at startup: with liquid cooling on and no chiller flow yet,
       `configureAcquisition()` sets `m_chillerFault` and doesn't start acquisition.
   - Log the cooling mode once at startup: "cooling mode: liquid (fan off, sensor cooler MAX, chiller
     <device> flow X L/min)" or "cooling mode: air (sensor cooler ON)".

5. **Config and documentation.**
   - Config templates: `liquidCooling=false`, plus a commented `[chiller]` section. The HAMSCI deployment
     config sets `liquidCooling=true` and `chiller.device=<chiller app>`.
   - Class docs (rule 15): describe `camera.liquidCooling`, the `chiller.*` keys, how they interact with
     `camera.fanSpeedControl`, `camera.defaultFanSpeed` and `camera.startupTemp`, and the one-time
     `dcamcfgc` "Cooler Type = Water" step with its camera restart.
   - Add a short operator note to the app's handbook page. That page is linked from the `\defgroup`, at
     `../handbook/operating/software/apps/orcaCtrl.html`.

6. **Tests** (rule 20).
   - Add `apps/orcaCtrl/tests/orcaCtrl_test.cpp` under `libXWCTest::orcaCtrlTest`, with a
     `\defgroup orcaCtrl_unit_test` that `\ingroup application_unit_test` in `tests/groups.dox`.
   - Config tests through `loadConfigImpl()`: `liquidCooling=true` forces fan off and hides fan control;
     `liquidCooling=true` with an empty `chiller.device` fails; the default leaves the stdCamera fan
     config alone; the default setpoint is -35 for liquid and -20 for air, and `camera.startupTemp`
     overrides it.
   - `chillerRunning()` tests that set the stored flow and timestamp directly: no data, stale data, low
     flow, and good flow.
   - `setTempSetPt()` clamping, when the target isn't writable, needs no hardware.
   - DCAM writes can't be tested without hardware or a mock, so those are left to step 7. Add `\ref`
     links per rule 21 where the harness hides the real symbols.

7. **Verify and commit.**
   - Format with the cpptools clang-format and build with `gcc-toolset-14`.
   - Hardware checks:
     - Liquid, chiller running: fan reads OFF, `SENSORCOOLER` reads MAX (4), `SENSORTEMPERATURETARGET`
       reads back the setpoint, `SENSORCOOLERSTATUS` reaches READY/LOCKED near -35 °C, and the app never
       goes to `ERROR` from `getFanSpeed()`.
     - Liquid, stop the chiller or its app: within `chiller.timeout`, cooling turns off, acquisition
       stops (the image stream's `cnt0` stops advancing), and a `LOG_ALERT` is logged. Restart it and both
       cooling and acquisition resume.
     - Liquid, start orcaCtrl with the chiller off: acquisition doesn't start until flow is reported.
     - Liquid config with the camera in Air mode: the mismatch sends the app to `ERROR` without cooling
       off the fan.
     - Air: `SENSORCOOLER` reads ON (2), the fan is on, and the target reads -20 or the configured value.
     - Change the setpoint over INDI and confirm the camera reads back the clamped value.
   - Commit order (rule 19): step 2, step 3, step 4, then tests, then the docs pass, then formatting if
     needed. Update Execution Notes in each commit.

## Open Questions / Assumptions

- **The mode switch is `dcamcfgc`, not code.** The plan treats Cooler Type as a one-time,
  per-installation setting that orcaCtrl checks but never changes. Scripting `dcamcfgc`, or copying its
  register writes through `dcamdev_setdata`, would mean relying on undocumented registers and a camera
  restart, so it's not planned.
- **How to detect water mode** isn't known until step 1. If no `dcamprop` signal tells the modes apart,
  the check falls back to config only, and the plan should say so in the logs.
- **Setpoint writes while capturing:** the plan uses `dcamprop_setvalue` during capture and falls back to
  a reconfigure if the camera refuses. Step 1 decides which.
- **Default setpoints** of -35 °C for liquid and -20 °C for air come from the manual's typical figures,
  -35 °C needing 20 °C water. With 25 °C water, the Quest2 is specified at -20 °C even in water mode.
- **Behavior changes:** orcaCtrl will now write `SENSORCOOLER` and `SENSORTEMPERATURETARGET`, which it
  never did before, and the default air-mode setpoint drops from -35 °C to -20 °C.

## Q and A

- **Q:** Is there a DCAM Configurator option for switching to liquid cooling mode?
  **A:** Yes. `dcamcfgc`'s "Cooler Type = Water" setting is stored in the camera and needs a camera
  restart. There is no `dcamprop` equivalent, so `camera.liquidCooling` declares the mode and orcaCtrl
  checks it, rather than switching it.
- **Q:** Should `liquidCooling=true` check that the chiller is running?
  **A:** Yes (second follow-up). Cooling starts only while chiller flow is recent and at least
  `chiller.minFlowRate` (0.45 L/min).
- **Q:** Is the temperature setpoint written to `SENSORTEMPERATURETARGET`?
  **A:** Not today. Step 3 adds the write, with bounds from the camera and a read-back.
- **Q:** When chiller flow is lost, should orcaCtrl only turn cooling off, or also stop acquisition?
  **A:** Stop acquisition as well (third follow-up). Step 4 turns the sensor cooler off, stops capture,
  and holds it stopped until flow is restored.
- **Q:** Which app publishes the chiller status?
  **A:** A koolanceCtrl-style INDI app (third follow-up). The defaults `chiller.property = status` and
  `chiller.flowElement = flow_rate` match koolanceCtrl, and `chiller.device` names the instance.
- **Q:** Why does `temp_ccd` rise to 29 °C in liquid mode with glycol flowing, and why can't
  `temp_controller` be toggled?
  **A:** orcaCtrl never turns on the sensor cooler (`DCAM_IDPROP_SENSORCOOLER`). In Water mode the camera
  doesn't start cooling by itself, so the Peltier stays off. `setTempControl()` doesn't send anything to
  the camera, so `getTemps()` reads the cooler as OFF and turns the toggle back off. See "Investigation:
  `temp_ccd` rises to 29 °C in liquid mode" below.

## Execution Notes

- Step 4 was implemented on 2026-11-17 (see "Step 4 implementation" below). Steps 2, 3 and 5-7 are
  not done. Step 1 has been run in **air mode only** (results below). The water-mode re-probe is still
  to do.
- **Deferred (2026-09-30):** the user won't be testing liquid cooling for now. orcaCtrl only gets the
  optional `camera.liquidCooling` flag: a bool, default `false`, with a documented `m_liquidCooling`
  member and config help text, read in `loadConfigImpl()`. It is logged in `connect()`: "cooling mode:
  air", or a `LOG_WARNING` that liquid cooling is declared but its settings aren't applied yet. The flag
  makes no DCAM writes and doesn't change fan, cooler or setpoint behavior. The rest of steps 1-7 stays
  here for when liquid cooling is tested.
- Found the DCAM fan property and confirmed that no liquid-cooling `dcamprop` property exists by
  searching `dcamprop.h` and `dcamapi4.h` in `/opt/hamamatsu_new/hamamatsu_sdk/dcamsdk4/inc/`.
- Checked the stdCamera fan and temperature config (`camera.fanSpeedControl`, `camera.defaultFanSpeed`,
  `camera.startupTemp`) and the earlier plans `agents/plans/2026-03/pvcam-stdcamera-fan-ctrl.md` and
  `agents/plans/2026-04/picam-cooling-fan-control.md`.
- First follow-up: liquid cooling sets `DCAM_IDPROP_SENSORCOOLER` to `MAX` (4) and air sets `ON` (2).
- Second follow-up:
  - From `ham_doc.md`: the cooling method is changed with DCAM Configurator and stored in the camera;
    water mode needs the chiller running before cooling starts, with at least 0.45 L/min flow; and
    -35 °C is available only in water "Max cooling".
  - Found the Linux DCAM Configurator, `/opt/hamamatsu_new/hamamatsu_api/tools/x86_64/dcamcfgc` (ver
    24.12.6898, lists `C15550-22UP`). Its strings show a `Cooler Type` parameter with
    `Air(Standard)/Air(Rapid)/Air(Off)/Water`, plus `Water Cooling Target`, `Target Temperature`,
    `Sensor Cooler`, `Default Fan Control` and `Cooler FAN`. It was only inspected with `strings`, not
    run.
  - Confirmed the setpoint is not written to `SENSORTEMPERATURETARGET` today: the write is commented out
    in `configureAcquisition()`, and `setTempSetPt()` only sets `m_reconfig`.
  - Found that `koolanceCtrl` publishes `status.flow_rate` (L/min) and `status.liquid_temp`, which makes a
    workable chiller signal.
- Third follow-up: losing chiller flow now also stops acquisition through `m_chillerFault` and
  `m_reconfig`, and it restarts when flow returns. The chiller source is a koolanceCtrl-style INDI app.
  Moved both answered questions out of Open Questions and into the new Q and A section.

### Step 1 results: air mode (2026-09-30)

Setup: orcaCtrl was not running. Camera `HAMAMATSU C15550-22UP`, DCAM camera ID `PHX2`, firmware 3.00,
driver 8.26.160.0000, on CoaXPress (the `aslcxp` driver is loaded).

**`dcamcfgc` (ver 24.12.6898), read-only.** This model shows only five parameters:

| # | Parameter | Current | Options |
| --- | --- | --- | --- |
| 1 | Cooler Type | **Air** | `Air`, `Water` |
| 2 | Back Panel LED | ON | not opened |
| 3 | Fusion(C14440) Emulation Mode | OFF | not opened |
| 4 | Quest(C15550-20UP) Emulation Mode | OFF | not opened |
| 5 | Raw data output mode (switch with PNR mode) | Disable | not opened |

- The Quest2 has **no** `Water Cooling Target`, `Target Temperature`, `Sensor Cooler`, `Default Fan
  Control` or `Cooler FAN` parameters. Those strings belong to other camera models, and so do the extra
  `Air(Rapid)`/`Air(Off)`/`Air(Standard)` Cooler Type values.
- Procedure note: with a single camera, `dcamcfgc` selects it automatically and never shows the "Select
  Camera Index" prompt. The first input therefore goes to "Select Parameter Index". In the first run, a
  leading `1` opened the Cooler Type value menu, and the next `0` exited it without a change ("Changed
  value" was not printed). A second run with input `0` confirmed Cooler Type is still `Air`.

**DCAM properties (read-only probe, `dcamCoolProbe.cpp`, kept in the session scratchpad and not
committed).** The camera enumerated 76 supported properties in air mode.

| Property | Result in air mode |
| --- | --- |
| `SENSORCOOLERFAN` (0x00200350) | `DCAMERR_INVALIDPROPERTYID` (0x80000825), not present |
| `SENSORCOOLER` (0x00200320) | `DCAMERR_INVALIDPROPERTYID`, not present |
| `SENSORTEMPERATURETARGET` (0x00200330) | `DCAMERR_INVALIDPROPERTYID`, not present |
| `SENSORCOOLERSTATUS` (0x00200340) | Read-only mode. Values `1=OFF`, `2=READY`, `3=BUSY`. Current `READY`. |
| `SENSORTEMPERATURE` (0x00200310) | Read-only, -50 to 100 °C, step 1. Current **-20 °C**. |

The only supported properties with COOL, TEMP, FAN or WATER in their names are those last two.

**What this means for the plan:**

- In air mode the camera offers **no** fan, cooler-switch or temperature-target control through
  `dcamprop`. Cooling is fixed at -20 °C, as the manual specifies. `connect()`'s existing
  `SENSORCOOLERFAN` `getattr` check already returns "not supported" correctly in this mode.
- Forcing `m_fanControlSupported` and `m_fanStatusSupported` to `true` in `powerOnDefaults()` breaks air
  mode. `getFanSpeed()` would read the missing `SENSORCOOLER` property, fail on every `appLogic()` pass,
  and put the app in `ERROR`. The fan flags have to come from `getattr`, as step 2 says. The temporary
  change that forced them to `true` has been reverted to `false`.
- Step 3 (the setpoint write) must check `getattr` first. In air mode `SENSORTEMPERATURETARGET` doesn't
  exist, so the setpoint is read-only at -20 °C, the INDI target should reflect that, and the
  hard-coded -35 °C default is wrong for air mode.
- `SENSORCOOLERSTATUS` in air mode offers no `WARNING` value, so `getTemps()`'s `WARNING` branch can
  only fire, if at all, in water mode.
- Candidate water-mode signal for step 4: whether `SENSORCOOLER`, `SENSORCOOLERFAN` or
  `SENSORTEMPERATURETARGET` exist at all. They are absent in air mode. The water-mode re-probe has to
  confirm this.
- The camera ID string is `PHX2`, but `connect()` compares against a hard-coded `"000044"`, so
  `camera.serialNumber` never selects the camera. This is a separate follow-up.

**Still to do in step 1 (the user will run this by hand):** with the chiller connected and circulating, set `Cooler Type = Water` with
`dcamcfgc`, restart the camera, re-run the probe, and record which properties appear and their values
(especially whether `SENSORCOOLER` offers `MAX`). Then either leave the camera in Water mode or set it
back to Air.

### Investigation: `temp_ccd` rises to 29 °C in liquid mode (2026-11-17)

**Reported:** the camera is set to liquid cooling and glycol is flowing, but after power-on `camscc.temp_ccd`
rises to about 29 °C instead of falling toward the -35 °C target, and `camscc.temp_controller` can't be
turned on. This was a code-only investigation done on a workstation, without the camera. Nothing was run
against the hardware.

**Root cause: orcaCtrl never turns on the sensor cooler.**

- In Water mode the camera doesn't start cooling at power-on. The manual (section 9-2-2) says the
  application has to "turn on the cooling switch of the camera", which is `DCAM_IDPROP_SENSORCOOLER`
  (0x00200320, `OFF = 1`, `ON = 2`, `MAX = 4`).
- `grep` finds no write of `DCAM_IDPROP_SENSORCOOLER` anywhere in `apps/orcaCtrl/`. It only appears as the
  `DCAMPROP_SENSORCOOLER__OFF/__ON` constants in `getFanSpeed()`, and `getFanSpeed()` never runs because
  `m_fanStatusSupported` is never set to `true`.
- So in Water mode the Peltier stays off. The internal fan is also off (Water mode never runs it), so the
  only heat path is the cold plate to the glycol. With no thermoelectric cooling, the sensor settles a few
  degrees above the coolant because the camera electronics keep heating it. Rising to 29 °C fits this.
  Step 4 of this plan, which would write `SENSORCOOLER = MAX`, has not been implemented. The
  `camera.liquidCooling` flag only forces the default fan setting to `off` (`loadConfigImpl()`), and in
  Water mode the fan is already off.
- In Air mode the camera cools to -20 °C by itself (Step 1 results), so this problem only appears after
  Cooler Type is switched to Water.

**Why `temp_controller` can't be toggled:**

1. stdCamera's `newCallBack_temp_controller()` sets `m_tempControlStatusSet` and calls
   `orcaCtrl::setTempControl()`.
2. `setTempControl()` (`orcaCtrl.hpp` ~line 1681) sends nothing to the camera. It sets
   `m_tempControlStatus = true` and pushes the switch On. It also ignores the requested state, so an Off
   request is treated as On.
3. On the next `appLogic()` pass, `getTemps()` (~line 1525) reads `SENSORCOOLERSTATUS`. With the cooler
   off, the camera reports `OFF` (1). There is no `OFF` branch, so it falls into the default `else`:
   `m_tempControlStatus = false` and the status string becomes `"UNKNOWN"`.
4. stdCamera's `updateINDI()` sees `m_tempControlStatus == false` and turns the `temp_controller` toggle
   back Off.

The toggle therefore can't stay on until orcaCtrl actually writes `SENSORCOOLER`.

**Why the -35 °C target has no effect:**

- The `SENSORTEMPERATURETARGET` write in `configureAcquisition()` (~line 1987) is still commented out.
  Every reconfigure logs "Set temperature set point: -35.000000 C", but nothing is written to the camera.
  The -35 °C target is only the value hard-coded in `powerOnDefaults()`.
- Even with the cooler on, the manual gives -35 °C only for Water mode with `SENSORCOOLER = MAX` and 20 °C
  water. With `ON` the typical value is -20 °C.

**Other things found along the way:**

- `m_liquidCooling` now defaults to `true` in the member initializer (`orcaCtrl.hpp:231`, commit
  `d90709a5`). The `config.add()` help text still says "Default is false" and "Currently only logged". The
  default should either go back to `false`, with `liquidCooling=true` set in `camscc.conf`, or the help
  text should be fixed.
- The untracked `camscc_20261115123443932659433.binlog` (2026-11-15 12:34) is from a build older than
  `d90709a5`. It logs `camera.liquidCooling` as `default` (false), "cooling mode: air", and "Cooling fan
  control is enabled in config but not supported". Temperatures are recorded in the `.bintel` telemetry,
  not in this binlog, so it doesn't show the 29 °C rise. It does show that `SENSORCOOLERFAN` was missing
  from that run.
- `dcamprop.h` also defines `DCAM_IDPROP_SENSORTEMPERATURE_STATUS` (0x00200390, `NORMAL`, `WARNING`,
  `PROTECTION`) and `DCAM_IDPROP_SENSORTEMPERATURE_PROTECT` (0x00200400). They weren't in the air-mode
  probe's list of COOL, TEMP and FAN properties. Add them to the water-mode re-probe, since a sensor that
  is warming may trip protection.

**What the user should check on the instrument** (agents don't touch the hardware):

1. With orcaCtrl stopped, run `dcamcfgc` and confirm Cooler Type reads **Water**. (Pressing `0` exits
   without changing anything.)
2. With the chiller circulating and the camera power-cycled after the Cooler Type change, run the
   scratchpad `dcamCoolProbe` again. This is the outstanding Step 1 water-mode re-probe. Record whether
   `SENSORCOOLER` exists, whether it is writable, its valid values (looking for `MAX`), and its current
   value (expected `OFF`). Also record `SENSORCOOLERSTATUS` (expected `OFF`), `SENSORCOOLERFAN`,
   `SENSORTEMPERATURETARGET` (attributes and range), and the two `SENSORTEMPERATURE_*` properties above.
3. While orcaCtrl is running, `camscc.temp_control.status` should read
   `UNKNOWN`. That confirms `SENSORCOOLERSTATUS` is outside BUSY, READY and WARNING, which is consistent
   with OFF.
4. Optional, with orcaCtrl stopped and flow confirmed: have the probe write `SENSORCOOLER = ON` (then
   `MAX`) and watch `SENSORTEMPERATURE` fall. This confirms the fix before any orcaCtrl code changes.

**Minimum code fix, a subset of steps 3 and 4, to do after the re-probe confirms `SENSORCOOLER` is
writable in Water mode:**

- In `connect()`, run `dcamprop_getattr( DCAM_IDPROP_SENSORCOOLER )` and set a documented
  `m_coolerControlSupported` from the writable attribute.
- `setTempControl()`: when supported, write `SENSORCOOLER` from `m_tempControlStatusSet`. On means `MAX`
  when `m_liquidCooling` is true (falling back to `ON` if `MAX` isn't a valid value), or `ON` in air mode.
  Off means `OFF`. Then read it back. Don't force the status to `true`.
- In `configureAcquisition()` (or right after `connect()`), when `m_liquidCooling` is true, apply the
  cooler state so cooling starts at power-on without an INDI toggle. Until the chiller gate from step 4
  exists, this relies on the operator starting flow before powering the camera, as the manual requires.
  Write the chiller gate before leaving this unattended.
- `getTemps()`: add a `DCAMPROP_SENSORCOOLERSTATUS__OFF` branch that sets `m_tempControlStatusStr = "OFF"`
  (instead of "UNKNOWN") and keeps `m_tempControlStatus` false.
- Step 3: uncomment the `SENSORTEMPERATURETARGET` write, guarded by `getattr`, and remove the misleading
  log line.

### Step 4 implementation (2026-11-17)

**Scope change from the user:** coolant flow is started before the camera is powered on, and there is no
way to monitor the chiller. So the chiller part of step 4 was **dropped**: no `chiller.*` config, no INDI
subscription, no flow gate at startup, and no stop when flow is lost. This replaces the earlier Q and A
answers about checking the chiller and stopping on flow loss. Running the chiller before power-on and
keeping it running is the operator's job, as the manual requires.

What was built, in `apps/orcaCtrl/orcaCtrl.hpp`:

- **Config.** In liquid mode, `loadConfigImpl()` sets `m_fanSpeedControlEnabled = false` and
  `m_defaultFanSpeed = "off"`, which hides the INDI `fan_speed` switch. This replaces the old warning,
  which fired every time because the constructor's default fan speed is `on`.
  `camera.liquidCooling` still defaults to `true` (commit `d90709a5`), and its help text now says so.
- **Water-mode check, in `connect()`.** `dcamprop_getattr( SENSORCOOLER )` sets `m_coolerControlSupported`
  (writable) and `m_coolerMaxSupported` (`valuemax >= MAX`). If `m_coolerControlSupported` doesn't match
  `m_liquidCooling`, in either direction, it logs `LOG_CRITICAL` "Cooler Type mismatch" (once, through
  `stateLogged()`), closes the camera and sets `ERROR`. On success it logs "cooling mode: liquid (fan off,
  sensor cooler MAX)" or "cooling mode: air", plus a warning if MAX isn't available.
- **`configureAcquisition()`.** In liquid mode it writes `SENSORCOOLERFAN = OFF` (if supported), then
  `SENSORCOOLER = sensorCoolerTarget()`: MAX, or ON when MAX isn't available, or OFF when temperature
  control is off. `setSensorCooler()` reads the mode back into `m_sensorCoolerMode` and logs a notice when
  it changes. This is the fix for the 29 °C problem found in the investigation.
- **`temp_controller`.** `powerOnDefaults()` sets `m_tempControlStatusSet = true`, so cooling starts at
  power-on. With a writable cooler, `setTempControl()` writes `sensorCoolerTarget()` right away, falling
  back to `m_reconfig` if the camera refuses during capture. Air mode keeps the old always-on behavior.
  `getTemps()` now maps `SENSORCOOLERSTATUS__OFF` to `"OFF"` instead of `"UNKNOWN"`.

Where this differs from the plan text above:

- **No chiller monitoring**, as described above.
- **Air mode doesn't write `SENSORCOOLER = ON`.** The property doesn't exist in Air mode (step 1), and a
  camera that has it while `liquidCooling=false` is now a mismatch error.
- **Water-mode detection** depends on `SENSORCOOLER` being writable in Water mode. That is consistent with
  the manual's "turn on the cooling switch from application software", but it is unconfirmed until the
  water-mode re-probe.
- **Step 3 isn't done.** `SENSORTEMPERATURETARGET` still isn't written, so the camera cools to its own
  MAX/ON target, and the "Set temperature set point" log line is still misleading.

Tests, in `apps/orcaCtrl/tests/orcaCtrl_test.cpp`:

- The fake `dcamprop_getattr` now reports `SENSORCOOLER` and `SENSORCOOLERFAN` as missing, as in Air mode,
  unless a test clears `m_missing`, and `m_coolerMax` sets the cooler's `valuemax`. `tests/dcam/dcamprop.h`
  gained `DCAMPROP_SENSORCOOLER__MAX` and `DCAMPROP_SENSORCOOLERSTATUS__OFF`.
- The existing tests run in air mode (`m_liquidCooling = false` in the fixture).
- New cases cover:
  - config loading;
  - mismatch in both directions;
  - liquid apply, with and without MAX;
  - the `OFF` status string;
  - the `temp_controller` toggle in liquid and air mode.
- These tests were **not built or run**, because the workstation used has no mxlib (`gengithead.sh` is
  missing, too). Build and run them with `make -f Makefile.one t=../apps/orcaCtrl/tests/orcaCtrl_test` in
  `tests/`.
