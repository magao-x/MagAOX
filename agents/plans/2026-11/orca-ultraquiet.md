# Task Description
The Hamamatsu Orca Quest2 does not let you switch into the ultraquiet readout mode.

## Problem Statement
- The Orca Quest2 supports an "ultraquiet" readout mode but this has not yet been implemented
## Discussion
- Add the "ultraquiet" mode to the orcaCtrl.hpp
## Requirements
- Camera should be able to seamlessly switch between standard and ultraquiet readout modes by selecting the appropriate readout from the stdCam dropdown menu
## Tests and Metrics
- Add test to verify that camera can seamlessly switch between readout modes
## Scope and Caveats

# Instructions to Agent
<!-- Specific instructions for the agent.  Below is our standard, but you can modify it as needed. -->

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan
<!-- This section will be filled out by the agent -->

## Task Summary
<!-- The agent should summarize the task as they understand it -->

The ORCA-Quest2 has two scan speeds (manual section 11-1-3):

| Scan speed | Read noise | Full-frame rate (CoaXPress) | 1H line time | Exposure range |
| --- | --- | --- | --- | --- |
| Standard scan | 0.43 e⁻ rms | 120 fps | 7.2 µs | 7.2 µs to 1800 s |
| Ultra quiet scan | 0.30 e⁻ rms | 25.4 fps | 33.9 µs | 33.9 µs to 1800 s |

orcaCtrl already lists both in the stdCamera `readout_speed` INDI switch (`{ "Standard", "Ultra-quiet" }`
in the constructor), so the dropdown is there. Selecting a speed doesn't change anything on the camera,
though:

- `setReadoutSpeed()` only calls `recordCamera()` and sets `m_reconfig`.
- The "Readout Speed" section of `configureAcquisition()` *reads* `DCAM_IDPROP_READOUTSPEED` into an unused
  `cmode` and never writes `m_readoutSpeedNameSet`. It has a `\todo` saying so.
- `m_readoutSpeedName`, which stdCamera publishes as the current speed and records in `telem_stdcam`, is
  hard-coded to `"Standard"` in `powerOnDefaults()` and never updated. INDI and telemetry show "Standard"
  whatever the camera is really doing.

The task is to write the requested speed to the camera when acquisition is reconfigured, read back the
speed actually applied, and report it, so the dropdown really switches between Standard and Ultra quiet
scan.

## Key Assumptions
<!-- The agent should list any assumptions they have made -->

1. **DCAM value mapping.** `dcamprop.h` defines only `DCAMPROP_READOUTSPEED__SLOWEST = 1` and
   `DCAMPROP_READOUTSPEED__FASTEST = 0x7FFFFFFF` ("w/o", i.e. write-only: DCAM replaces it with the
   fastest valid value). The manual doesn't give the DCAM values for each scan speed. Assumed: Ultra quiet
   = slowest (1) and Standard = fastest (2 on the Quest2). To avoid hard-coding 2, the plan writes
   `__SLOWEST` / `__FASTEST` and then identifies the applied speed from the read-back value, compared with
   `dcamprop_getattr()`'s `valuemin` / `valuemax`. The user confirms this on the camera (Q1).
2. **Capture has to stop to change the speed.** It changes the line time, frame rate and exposure limits.
   The plan keeps the existing reconfigure path (`setReadoutSpeed()` sets `m_reconfig`, `reconfig()`
   stops capture, `configureAcquisition()` applies the change and restarts). So "seamless" here means one
   automatic acquisition restart with no operator action, not a switch without a break in frames (Q2).
3. **Exposure time and frame rate.** `configureAcquisition()` already reads the readout time, the
   exposure-time limits from `dcamprop_getattr( EXPOSURETIME )`, and the frame rate *after* the readout
   speed section. Writing the speed first makes those values follow automatically, and `capExpTime()` then
   raises an exposure below 33.9 µs to the new minimum. The ring buffer is resized from the new frame rate
   by `allocateBuffers()`.
4. **Area readout mode only.** Photon number resolving and Lightsheet modes, which use different sensor
   modes, are out of scope.
5. Coolant handling and the liquid-cooling code (`agents/plans/2026-09/orcaCtrl-liquidCooling.md`) aren't
   affected. The readout speed section runs after the liquid-cooling section in `configureAcquisition()`.

## Requirements
<!-- The agent should list the requirements to which they are planning -->

- R1: Selecting a speed in `readout_speed` writes it to the camera on the next reconfigure, which
  `setReadoutSpeed()` already requests.
- R2: `m_readoutSpeedName` (INDI `readout_speed` current value, and `telem_stdcam`) reports the speed read
  back from the camera, not a hard-coded string.
- R3: The power-on and connect speed comes from `camera.defaultReadoutSpeed` (stdCamera config, default
  Standard). `powerOnDefaults()` stops hard-coding `"Standard"`.
- R4: After a switch, the readout time, exposure-time limits, exposure time (capped to the new minimum),
  frame rate and DCAM buffer size are updated, which the existing `configureAcquisition()` code does once
  the speed is written first.
- R5: An unknown requested name is logged and reset to the current speed, and acquisition continues; it
  doesn't put the app in ERROR.
- R6: Unit test(s) show switching both ways through the reconfigure path.

## Questions and Points of Clarification
<!-- The agent should list any open issues requiring user clarification -->

1. **DCAM values (confirm on the camera).** Please run a read-only check with orcaCtrl stopped. I can
   extend the earlier probe to print, for `DCAM_IDPROP_READOUTSPEED`: `dcamprop_getattr` (min, max, step,
   writable), each valid value with `dcamprop_queryvalue`, and its text with `dcamprop_getvaluetext`. I
   expect min 1 (Ultra quiet) and max 2 (Standard). If the camera reports something else, the mapping in
   step 2 changes.
2. **"Seamless".** Is one automatic acquisition restart (frames pause for about the length of a reconfigure)
   acceptable? DCAM normally doesn't allow `READOUTSPEED` to change during capture. I could try
   `dcamprop_setvalue` during capture first and fall back to a restart, but that adds a path that's hard to
   test without the camera, so I don't plan to unless you want it.
3. **Switch names.** Current element names are `Standard` and `Ultra-quiet`. Other MagAO-X cameras use
   lower-case snake_case names with separate labels (e.g. pvcamCtrl `sub_electron` / "Sub-Electron"), and
   a hyphen in an INDI element name is unusual for this repo. Should I rename them to `standard` and
   `ultra_quiet` (labels unchanged)? That changes the allowed values of `camera.defaultReadoutSpeed` and the
   names in GUIs and scripts. `camscc.conf` doesn't set it today (the 2026-11-15 binlog shows it at
   `default`). The plan assumes yes; say no to keep the current names.
4. **Default speed.** Keep Standard as the default (`camera.defaultReadoutSpeed`), or should HAMSCI start in
   Ultra quiet?

### Answers (2026-11-21)

1. Run the probe. The agent can't run it (AGENTS.md rule 0; the camera isn't on this workstation), so it
   wrote a read-only probe, `dcamReadoutProbe.cpp`, for the user to run. The implementation doesn't depend
   on the exact values, since it writes `__SLOWEST` / `__FASTEST` and identifies the speed from the
   read-back value, so it went ahead without waiting for the probe.
2. A brief pause is acceptable, so the speed changes through the existing reconfigure (acquisition
   restart).
3. Rename the entries to `standard` / `ultra_quiet`, following MagAO-X convention.
4. The default stays Standard (`standard`).

## Tests
<!-- The agent should list and describe the test it plans to implement.  It should be specific about the purpose and goal of the test. -->

In `apps/orcaCtrl/tests/orcaCtrl_test.cpp`, using the existing fake DCAM library. Fake changes:

- `dcamprop_getattr( READOUTSPEED )` reports min 1, max 2, step 1, writable.
- `dcamprop_setgetvalue( READOUTSPEED )` clamps the written value to [1, 2] and returns it, as DCAM does
  with `__FASTEST`. Every other property keeps its current behavior.
- `dcamprop_getattr( EXPOSURETIME )` reports a minimum of 7.2 µs at speed 2 and 33.9 µs at speed 1, so the
  test can check that exposure limits follow the speed.

Test cases:

1. **"orcaCtrl switches between Standard and Ultra quiet readout"** (R1, R2, R4, R6). Start the app on the
   fake camera, run `configureAcquisition()`, and check the camera is at speed 2 and
   `m_readoutSpeedName == "standard"`. Then set `m_readoutSpeedNameSet = "ultra_quiet"`, call
   `setReadoutSpeed()` (expect `m_reconfig`), run `reconfig()` and `configureAcquisition()`, and check:
   speed 1, `m_readoutSpeedName == "ultra_quiet"`, `m_minExpTime` = 33.9 µs, an exposure of 10 µs raised to
   33.9 µs, and capture restarted (`dcamcap_start` count 2). Switch back to Standard and check speed 2 and
   the 7.2 µs minimum. No error logs.
2. **"orcaCtrl starts in the configured default readout speed"** (R3). Load config with
   `camera.defaultReadoutSpeed=ultra_quiet`, power on, connect, configure, and check speed 1 and the
   reported name.
3. **"orcaCtrl rejects an unknown readout speed"** (R5). Set `m_readoutSpeedNameSet = "bogus"`, configure,
   and check: a warning is logged, the camera speed is unchanged, `m_readoutSpeedNameSet` is reset to the
   current speed, and `configureAcquisition()` still succeeds.

Each `TEST_CASE` gets a Doxygen block, `\ingroup orcaCtrl_unit_test`, and an `ORCACTRL_TEST_DOXYGEN_REF`
block that references `setReadoutSpeed()` and `configureAcquisition()` (AGENTS.md rules 19 and 20).

Hardware check, run by the user: switch to Ultra quiet in the GUI and confirm `readout_speed`,
`readout_time` and `fps` change (full frame: about 25.4 fps maximum, against 120 fps for Standard), and the
image noise drops. Then switch back.

## Implementation Plan
<!-- Here the agent documents its plan for the implementation -->

1. **Confirm the DCAM values** (Q1). Extend the scratchpad probe and have the user run it read-only.
   Record the result here.
2. **Write and read back the speed, in `configureAcquisition()`'s "Readout Speed" section** (replacing the
   unused `cmode` read and its `\todo`):
   - Map `m_readoutSpeedNameSet` to `DCAMPROP_READOUTSPEED__FASTEST` (`standard`) or `__SLOWEST`
     (`ultra_quiet`). An unknown name logs a `LOG_WARNING`, resets `m_readoutSpeedNameSet` to
     `m_readoutSpeedName`, and keeps the current camera setting (R5).
   - Write it with `setorcaParameter( DCAM_IDPROP_READOUTSPEED, ... )` while capture is stopped, which
     `reconfig()` guarantees. A DCAM failure follows the section's existing error pattern.
   - Read it back with `getorcaParameter()`, and set `m_readoutSpeedName` from the read-back value, using
     `dcamprop_getattr()`'s `valuemin` / `valuemax` to tell slowest from fastest. Log a `LOG_NOTICE` when the
     speed changes ("readout speed changed from standard to ultra_quiet").
   - Keep the section before the ROI, readout-time, exposure and frame-rate sections, so they all read
     values for the new speed (R4).
   - Put the name-to-value mapping in a small documented helper, either `orcaCtrl` member functions or
     functions in `orcaUtils.hpp` so they can be tested on their own, rather than inline in
     `configureAcquisition()`.
3. **Names and defaults** (Q3, Q4). Set `m_readoutSpeedNames = { "standard", "ultra_quiet" }` with labels
   `{ "Standard", "Ultra Quiet" }`, and `m_defaultReadoutSpeed = "standard"`. In `powerOnDefaults()`, set
   `m_readoutSpeedName` to `m_defaultReadoutSpeed` instead of the literal `"Standard"`. `connect()` already
   sets `m_readoutSpeedNameSet = m_defaultReadoutSpeed`.
4. **`setReadoutSpeed()`** stays as it is: telemetry plus `m_reconfig`. Update its declaration docs to say
   the change is applied by the next `configureAcquisition()` and restarts acquisition.
5. **Tests**, as described under Tests, with the fake DCAM changes in `orcaCtrl_test.cpp`. No change to
   `tests/dcam/dcamprop.h` is needed (`DCAM_IDPROP_READOUTSPEED` is already there); add
   `DCAMPROP_READOUTSPEED__SLOWEST` and `__FASTEST` to it.
6. **Docs.** Update the class Doxygen block to describe the `readout_speed` choices and
   `camera.defaultReadoutSpeed`, and give the operator note in the handbook page the two speeds and their
   frame-rate and noise trade-off.
7. **Verify and commit.** clang-format, build and run `orcaCtrl_test` on the build machine (this
   workstation has no mxlib), then the user's hardware check. Commits per AGENTS.md rule 22: code, then
   tests, then docs, then this plan file with execution notes.

## Follow-up and Edge Cases
<!-- The agent should list any planned follow up and any edge cases that are not addressed -->

- **Exposure time silently raised.** Going to Ultra quiet with an exposure below 33.9 µs raises it to the
  new minimum; `capExpTime()` logs that. Going back to Standard doesn't restore the shorter exposure.
- **Frame rate drops to about 25 fps full frame.** Anything downstream that expects 120 fps (latency
  statistics, cbuffs) has to cope. Only the DCAM buffer size is adjusted automatically.
- **Readout time and exposure steps.** The exposure step becomes 33.9 µs. The INDI `exptime` step is
  already updated from `dcamprop_getattr()` in `configureAcquisition()`.
- **Photon number resolving mode** uses Ultra quiet but is a separate sensor mode. Not in scope.
- **Mid-capture switching** (Q2) isn't planned. If wanted, it's a follow-up once we know whether the
  camera accepts `READOUTSPEED` during capture.
- **Telemetry history.** Old `telem_stdcam` records show `Standard`. After the rename (Q3) new ones show
  `standard` / `ultra_quiet`, which matters to any analysis script that matches the old string.
- **Unrelated, noticed while reading:** the misleading "Set temperature set point" log line from the
  liquid-cooling plan is still in `configureAcquisition()`.

## Execution Notes

**2026-11-21: implemented (steps 2-6), not yet built or run.**

- `orcaCtrl.hpp`:
  - The constructor uses `m_readoutSpeedNames = { "standard", "ultra_quiet" }`, labels
    `{ "Standard", "Ultra Quiet" }`, and default `standard`.
  - `powerOnDefaults()` sets `m_readoutSpeedName = m_defaultReadoutSpeed`.
  - The "Readout Speed" section of `configureAcquisition()`:
    - It replaces an unknown `m_readoutSpeedNameSet` (with a warning) by the current speed.
    - It writes `DCAMPROP_READOUTSPEED__SLOWEST` for `ultra_quiet` or `__FASTEST` for `standard`, then
      reads back the value and `dcamprop_getattr()`.
    - It sets `m_readoutSpeedName` to `ultra_quiet` when the read-back value is at `valuemin`, and to
      `standard` otherwise, logging a notice on change.
    - It runs before the ROI, readout-time, exposure-limit and frame-rate sections, so they pick up the
      new speed.
  - `setReadoutSpeed()` is unchanged: it records telemetry and sets `m_reconfig`.
  - Class and declaration docs were updated.
- I kept the name-to-value mapping inline in `configureAcquisition()` rather than adding a helper (step 2's
  last bullet). With only two speeds it's a single conditional, and the tests exercise it through
  `configureAcquisition()`.
- `tests/dcam/dcamprop.h` gained `DCAMPROP_READOUTSPEED__SLOWEST` / `__FASTEST`.
- `orcaCtrl_test.cpp`:
  - In the fake DCAM, `READOUTSPEED` has range [1, 2], and writes are clamped to that range, as DCAM does
    with `__FASTEST`.
  - The exposure minimum is 33.9 µs at speed 1 and 7.2 µs otherwise.
  - New tests:
    - "orcaCtrl switches between Standard and Ultra quiet readout"
    - "orcaCtrl starts in the configured default readout speed"
    - "orcaCtrl rejects an unknown readout speed"
- The handbook operator note (step 6) isn't written yet. It isn't in this repo.
- Still to do:
  - Build and run `orcaCtrl_test` on the build machine. This workstation has no mxlib.
  - The user runs the read-only probe `dcamReadoutProbe.cpp` with orcaCtrl stopped, to confirm the DCAM
    values (Q1) and whether `READOUTSPEED` can be set while capturing.
  - The hardware check described under Tests.
