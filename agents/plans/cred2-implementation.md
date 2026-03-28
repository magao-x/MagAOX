Task: review AGENTS.md, then consider: we need to create a MagAO-X app to control a C-RED 2 camera.  We will use an EDT framegrabber.  Examples of similar apps are the ocam2KCtrl and andorCtrl.  Some points:
  - This is a First Light Imaging camera, and so is very similar to the ocam2KCtrl case.  We expect to use serial-over-cameralink in almost the same way
  - The manual for the C-RED 2 is here /home/jrmales/Documents/MyPapers/Projects/MagAOX/Electronics/Cameras/C-RED_2/C-RED2_UserManual_20180625-2.pdf.  c.f. Section 9.1 and 9.2 for details of serial commands.
  - An important difference in the C-RED 2 compared to OCAM-2K is that C-RED 2 supports arbitrary ROIs.  This is implemented for EDT configuration in andorCtrl which writes tmp config files for loading.

For this first attempt we want to implement the same functionality that is in ocam2KCtrl, to include:
  - monitoring of temperatures
  - temperature setpoint control and status
  - setting of FPS and status

Differences from ocam2KCtrl:
  - no EM gain
  - no shutter
  - use of arbitary ROIs (see andorCtrl)

An example cameralink config for this camera is here: /home/jrmales/Documents/MyPapers/Projects/MagAOX/Electronics/Cameras/C-RED_2/edt.cfg

Please develop a plan and upate this document with it below:

Plan

1. Create a new `apps/cred2Ctrl` app using the same overall structure as `andorCtrl` and `ocam2KCtrl`.
   - Add:
     - `apps/cred2Ctrl/cred2Ctrl.hpp`
     - `apps/cred2Ctrl/cred2Ctrl.cpp`
     - `apps/cred2Ctrl/Makefile`
   - Register the new app in the top-level [Makefile](/home/jrmales/Source/MagAOX/Makefile), most likely under `apps_rtc`.
   - Base classes should be:
     - `MagAOXApp<>`
     - `dev::stdCamera<cred2Ctrl>`
     - `dev::edtCamera<cred2Ctrl>`
     - `dev::frameGrabber<cred2Ctrl>`
     - `dev::telemeter<cred2Ctrl>`
   - Do not include `dev::dssShutter` or EM-gain-specific logic.

2. Configure the new app around the C-RED 2 feature set rather than copying `ocam2KCtrl` blindly.
   - Recommended `stdCamera` compile-time settings for the first pass:
     - `c_stdCamera_tempControl = true`
     - `c_stdCamera_temp = true`
     - `c_stdCamera_readoutSpeed = false`
     - `c_stdCamera_vShiftSpeed = false`
     - `c_stdCamera_emGain = false`
     - `c_stdCamera_exptimeCtrl = false`
     - `c_stdCamera_fpsCtrl = true`
     - `c_stdCamera_fps = true`
     - `c_stdCamera_fan = true`
     - `c_stdCamera_analogGain = true`
     - `c_stdCamera_led = true`
     - `c_stdCamera_synchro = false` for the first pass
     - `c_stdCamera_usesModes = false`
     - `c_stdCamera_usesROI = true`
     - `c_stdCamera_cropMode = false` for the first pass
     - `c_stdCamera_hasShutter = false`
   - Follow the `andorCtrl` pattern of using one synthetic EDT mode backed by a temporary config file, rather than fixed named camera modes.

3. Reuse the EDT serial-over-Camera-Link path from `ocam2KCtrl`, but make the C-RED 2 response handling explicit.
   - The manual says the CLI uses ASCII commands terminated by line feed (`\n`).
   - The example EDT config already shows the needed serial settings:
     - `serial_baud: 115200`
     - `serial_term: <0A>`
     - `serial_waitc: 0D`
   - Hardware testing with EDT `serial_cmd` indicates the practical response path is a plain line such as `10.000000`, without a visible `fli-cli>` prompt.
   - Treat any prompt suffix described in the manual as optional rather than required.
   - Reassert the configured baud on the live PDV handle after `edtCamera::appStartup()` and after `pdvReconfig()`, since opening/reopening the PDV device can otherwise leave serial at the default rate before the first real command.
   - Keep that runtime baud-reset path compatible with EDT installs whose headers do not declare the baud helpers, for example by resolving those symbols dynamically and falling back to cfg-only behavior when unavailable.
   - If needed, add a small helper layer in `cred2Ctrl` or `cred2Utils.hpp` that:
     - sends a command
     - truncates the response at the first `\r`
     - ignores any trailing prompt when present
     - returns the clean payload for parsing

4. Prefer the camera’s `raw` CLI responses wherever possible.
   - This should simplify parsing compared to the verbose strings described in the manual.
   - Core commands for the initial implementation:
     - `temperatures snake raw`
     - `temperatures snake setpoint raw`
     - `temperatures motherboard raw`
     - `temperatures frontend raw`
     - `temperatures powerboard raw`
     - `temperatures peltier raw`
     - `temperatures heatsink raw`
     - `fps raw`
     - `minfps raw`
     - `maxfps raw`
     - `fan mode raw`
     - `fan speed raw`
     - `sensibility`
     - `led raw`
     - `cropping raw`
     - `cropping columns raw`
     - `cropping rows raw`
     - `set temperatures snake <value>`
     - `set fps <value>`
     - `set fan mode automatic`
     - `set fan mode manual`
     - `set fan speed <value>`
     - `set sensibility low|medium|high`
     - `set led on|off`
     - `set cropping on|off`
     - `set cropping columns <value>`
     - `set cropping rows <value>`
   - Do not issue `save` from the controller. Runtime control should remain volatile unless an operator explicitly persists settings out-of-band.

5. Implement a small C-RED utility/parser layer modeled after [ocamUtils.hpp](/home/jrmales/Source/MagAOX/apps/ocam2KCtrl/ocamUtils.hpp).
   - Add a `cred2Utils.hpp` with:
     - a struct holding the relevant temperatures
     - parsing helpers for raw numeric responses
     - ROI conversion helpers from MagAO-X center/size form to C-RED start/end rows/columns
   - Add unit tests for parser behavior, especially:
     - numeric raw responses
     - responses with the `fli-cli>` prompt suffix
     - invalid/malformed responses

6. Implement temperature monitoring and temperature setpoint/status handling first.
   - Mirror the `ocam2KCtrl` flow for:
     - a live multi-value INDI `temps` property
     - updating `m_ccdTemp`
     - updating `m_ccdTempSetpt`
     - logging and telemetry refresh when values change
   - Suggested `temps` elements:
     - `motherboard`
     - `frontend`
     - `powerboard`
     - `snake`
     - `setpoint`
     - `peltier`
     - `heatsink`
   - Recommended initial status logic:
     - `m_ccdTemp` comes from `temperatures snake`
     - `m_ccdTempSetpt` comes from `temperatures snake setpoint`
     - `m_tempControlOnTarget` is derived from `fabs( m_ccdTemp - m_ccdTempSetpt )`
     - `m_tempControlStatusStr` reports `ON TARGET`, `OFF TARGET`, or `UNKNOWN`
   - Important design note:
     - the manual documents a setpoint, but not a true cooler on/off command
     - for this camera, temperature control should be treated as setpoint-only control
     - keep `stdCamera` temperature-control enabled, but define INDI “off” as returning the setpoint to the default warm value of `20 C`, not disabling cooling hardware outright

7. Implement FPS query/set and keep FPS limits dynamic.
   - Use `fps raw` for current rate.
   - Use `set fps <value>` to apply the requested rate.
   - Query `minfps raw` and `maxfps raw` after startup and after ROI changes rather than hardcoding 400/600 fps limits.
   - Update `m_fps`, `m_minFPS`, and `m_maxFPS` from camera-reported values.

8. Implement arbitrary ROI support by combining `stdCamera` ROI handling with an `andorCtrl`-style temporary EDT config.
   - The manual says:
     - columns have granularity 32
     - rows have granularity 4
     - column start range is `0-639`
     - row start range is `0-511`
   - Set the full-frame ROI to:
     - center `x = 319.5`
     - center `y = 255.5`
     - width `640`
     - height `512`
   - `checkNextROI()` should:
     - clamp to sensor bounds
     - round start coordinates to the required granularity
     - round width/height to the required granularity
     - restore the last valid ROI on impossible requests
   - `setNextROI()` should:
     - update the pending target ROI
     - mark `m_reconfig = true`
     - leave the actual hardware change to the reconfigure path

9. Generate the EDT config file dynamically, following the [andorCtrl.hpp](/home/jrmales/Source/MagAOX/apps/andorCtrl/andorCtrl.hpp) `writeConfig()` approach.
   - Write to a temp file such as `/tmp/cred2_<configName>.cfg`.
   - Use `c_edtCamera_relativeConfigPath = false` so the generated config can live outside the MagAO-X config tree.
   - Seed the generated file from the provided example `edt.cfg`, preserving at minimum:
     - 4-tap layout
     - `CL_DATA_PATH_NORM`
     - `CL_CFG_NORM`
     - `CL_CFG2_NORM`
     - `htaps`
     - `method_framesync`
     - serial termination/wait-char settings
   - Rewrite only the ROI-dependent parts:
     - `width`
     - `height`
     - any active-region fields needed by EDT if padding/cropping directives are used

10. Apply ROI changes in the camera and the framegrabber as one reconfiguration step.
   - On reconfigure:
     - stop/abort acquisition
     - send the camera’s cropping commands for the target ROI
     - write the matching EDT config
     - call `edtCamera::pdvReconfig()`
     - update `m_currentROI`, `m_width`, `m_height`, and INDI current/target fields
   - For the first pass, treat ROI state as:
     - full-frame ROI => `set cropping off`
     - subframe ROI => `set cropping on`, then push column/row settings
   - This keeps the public interface simpler than exposing a separate crop-mode toggle immediately.

11. Keep acquisition and image handling minimal unless live testing shows additional work is needed.
   - Start with direct frame copies through `frameGrabber::loadImageIntoStreamCopy(...)`.
   - Set:
     - `m_dataType = _DATATYPE_INT16`
     - `m_width` and `m_height` from the active ROI
   - Proceed on the assumption that the supplied EDT config already delivers correctly ordered 4-tap images.
   - Do not add a descrambling or deinterleaving path in the first implementation.
   - Verify image ordering with hardware once the camera is connected.

12. Use a dedicated camera mutex around serial, reconfigure, and grab paths.
   - Mirror the `m_cameraMutex` approach from `ocam2KCtrl`.
   - Guard:
     - serial command/response traffic
     - ROI reconfiguration
     - any framegrabber restart path that can race with control commands

13. Add telemetry in two layers.
   - Always record `telem_stdcam` through `dev::telemeter`.
   - Expose the full temperature set through INDI in the first functional pass.
   - Record the full C-RED 2 temperature set in a dedicated `cred2_temps` logger patterned after `ocam_temps`.

14. Verify in stages.
   - Build-only verification:
     - app target compiles cleanly
     - parser/unit tests pass
   - Controller verification without live acquisition changes:
     - connect
     - query temperatures
     - query fps
     - set a new temperature setpoint
     - set a new fps
   - ROI/reconfigure verification:
     - full-frame startup matches 640x512
     - representative subframe ROI loads with the expected dimensions
     - returning to full frame restores the original size
   - Runtime safety checks:
     - no serial timeouts caused by the C-RED prompt suffix
     - no stale ROI metadata after reconfigure
     - no EDT/camera geometry mismatch after ROI changes

15. Defer non-core features until the first pass is stable.
   - Explicit external sync control
   - Tint/exposure-time control
   - Bias/flat/bad-pixel toggles
   - Persistence commands such as `save`
   - Any camera-mode abstraction beyond the one synthetic dynamic EDT mode

Resolved Decisions

- Temperature control semantics:
  - colleagues familiar with the camera confirmed that C-RED 2 is effectively controlled by target temperature setpoint, not by a separate cooler on/off command
  - the first implementation should therefore treat INDI “off” as “go to `20 C`”

- ROI public API:
  - keep `c_stdCamera_cropMode = false` for the first pass
  - derive camera cropping behavior from the requested ROI rather than exposing a separate crop-mode property

- Telemetry scope:
  - implement `telem_stdcam`, the live INDI `temps` property, and a dedicated `cred2_temps` logger for the full temperature set

- Image ordering:
  - proceed with the assumption that the supplied EDT config delivers correctly ordered images
  - do not implement descrambling in the first pass

Implementation Status

- Initial `cred2Ctrl` first-pass scaffolding is now in the tree, including:
  - `dev::stdCamera`, `dev::edtCamera`, `dev::frameGrabber`, and `dev::telemeter` integration
  - C-RED 2 serial helpers and ROI/config generation helpers
  - temperature/FPS/ROI control paths
  - focused helper tests for response parsing and ROI formatting
  - dedicated `cred2_temps` telemetry for the full detailed camera temperature set

- Local verification completed so far:
  - `cred2Utils_test` passes
  - `cred2Ctrl.o` syntax-checks successfully when the EDT headers are stubbed locally for compile validation

- Hardware validation completed so far on the EDT host:
  - serial command/response handling is now working reliably with the generated config
  - live image streaming works
  - temperature setpoint control works
  - FPS query/set works
  - framegrabber flip control works
  - fan control works
  - analog gain control works
  - LED control works
  - `telem_stdcam` now carries fan speed, analog gain, and LED state
  - ROI reconfiguration is the next hardware validation target

- Remaining environment limitation on this host:
  - full `cred2Ctrl` app builds still require a real EDT SDK install and headers on the machine
  - the app `Makefile` now sets `EDT=true` explicitly so the controller is built in EDT-enabled mode when that dependency is present

Follow-Up Items / Edge Cases

- Keep the EDT serial-response handling changes that preserve partial reads and drain trailing bytes between commands, since those were required for reliable hardware communication with the C-RED 2.

- Confirm with hardware that the sample EDT 4-tap configuration produces correctly ordered images, while proceeding under the assumption that no descrambling is needed in the first pass.

- Confirm whether EDT needs explicit active-region directives in addition to `width` and `height` for subframe ROIs, or whether the camera-side cropping commands alone are sufficient once the temporary config is rewritten.
  - Current assumption: EDT does not normally need them, so the first implementation should try `width`/`height` only.

- Validate that ROI rounding in `checkNextROI()` preserves the requested science target location as closely as possible when enforcing the C-RED 2 column/row granularities.

- Treat the first implementation as volatile runtime control only.
  - Do not issue `save` automatically from the controller unless operations explicitly ask for persisted camera settings.
