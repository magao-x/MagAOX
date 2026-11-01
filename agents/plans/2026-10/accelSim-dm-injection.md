# Task Description

## Problem Statement

Extend the MagAO-X `accelSim` application to optionally generate a deformable-mirror (DM) tip/tilt disturbance corresponding to the same simulated physical vibrations used to generate the `acc000` accelerometer stream.

The purpose is to create a physically correlated accelerometer and optical disturbance for future sensor-fusion and DDSPC testing. The simulated vibration should appear in `acc000` as an accelerometer signal and, when DM injection is enabled, as a real optical disturbance measured by the MagAO-X wavefront sensor (WFS).

The existing accelerometer simulation, `acc000` output, timing, synchronization, ADC conversion, and ImageStreamIO behavior should remain unchanged.

## Discussion

`accelSim` currently simulates telescope vibrations and produces the corresponding accelerometer measurements. These measurements are converted to simulated ADC counts and published through the `acc000` shared-memory stream.

The next step is to use the same simulated physical vibration state to generate a corresponding tip/tilt disturbance on the MagAO-X DM.

The accelerometer signal and DM disturbance should therefore originate from the same underlying simulated vibration. The DM disturbance should not be reconstructed by numerically integrating the final noisy accelerometer output.

### Physical Model

For a sinusoidal vibration mode `i`, acceleration and angular pointing are related by

a_i(t) = -(2*pi*f_i)^2 * L * theta_i(t)

where

a_i(t)     = acceleration [m/s^2]
f_i        = vibration frequency [Hz]
L          = effective lever arm [m]
theta_i(t) = angular pointing disturbance [rad]

The lever arm is
 
L = 10.0 m

so

theta_i(t) = -a_i(t) / ((2*pi*f_i)^2 * L)

The simulator already has access to the underlying state of each vibration mode. The implementation should use that physical state directly to produce the corresponding pointing disturbance rather than attempting to recover pointing from the noisy `acc000` signal.

For a small angular pointing disturbance, the corresponding optical path difference (OPD) across the entrance pupil is

OPD(x,y,t) = theta_x(t)*x + theta_y(t)*y

where `x` and `y` represent coordinates across the telescope entrance pupil.

Use

D = 6.5 m

for the Magellan entrance-pupil diameter when a physical pupil diameter is required for the conversion.

The 6.5 m dimension describes the telescope entrance pupil, which is reimaged onto the DM. It is not the physical diameter of the DM.

For a reflective DM,

OPD = 2 * h_DM

so the equivalent DM surface required to produce the pointing disturbance is

h_DM(x,y,t) = 0.5 * [theta_x(t)*x + theta_y(t)*y]

These equations describe the desired physical disturbance. They do not specify how that disturbance must be represented within the MagAO-X DM-control software.

### Offline Python Reference

The validated offline simulation is available at

scratch/accel_ts_sim_ddspc_cleaned.ipynb

Review this notebook before planning the implementation.

Use it as the reference for the simulated vibration physics, including:

- vibration modes and amplitudes,
- acceleration generation,
- acceleration-to-pointing relationship,
- tip/tilt time series,
- pointing-to-OPD conversion,
- reflective-DM factor of two,
- and the resulting physical DM disturbance.

The nominal model currently uses

lever arm = 10.0 m
vibration scale = 1.0
entrance pupil diameter = 6.5 m

and produces approximately

RMS DM surface P2V = 0.046 micron
RMS pointing       = 2.9 mas

The notebook's 50 x 50 DM representation is used to validate the physical conversion and visualize the disturbance. Do not assume that this is the correct representation for commanding the actual MagAO-X DM.

The goal is to reproduce the physical behavior of the notebook using the appropriate existing MagAO-X DM infrastructure.

If the notebook and this plan appear to disagree about the physical model, document the discrepancy before implementation.

### MagAO-X DM Command Reference

It is recommended to review `dmPokeWFS.hpp` as a reference for how DM commands are sent in MagAO-X.

Begin the software investigation with `dmPokeWFS.hpp`.

Trace the relevant code sufficiently to determine:

- which DM stream or interface is used,
- how the stream is opened or connected,
- the dimensions and datatype of the command,
- the units of the command,
- how commands are written,
- how ImageStreamIO metadata and semaphores are handled,
- how DM geometry is represented,
- how commands are combined with other DM commands,
- and how the application interacts with the normal AO control path.

Also inspect other MagAO-X code as needed to determine whether there is an existing mechanism for injecting tip/tilt or arbitrary disturbances.

In particular, determine whether the appropriate implementation should use:

- an existing DM command stream,
- an additive DM channel,
- a modal command,
- calibrated tip/tilt modes,
- an actuator-space command,
- or another existing MagAO-X mechanism.

Do not choose the operational DM representation until the existing architecture has been investigated.

### WFS Measurement

Do not create a synthetic WFS signal or introduce an artificial WFS coupling factor.

The purpose of applying the disturbance to the DM is for the real MagAO-X WFS to measure the resulting optical tip/tilt.

The implementation should therefore produce the physical DM disturbance and allow the existing optical/WFS system to determine the measured response.

### Physical Sanity Check

Use the following relationship as an independent check of the pointing-to-DM conversion.

For a pure wavefront tilt, one wave peak-to-valley OPD across the entrance pupil corresponds to approximately one `lambda/D` of image motion.

At lambda = 900 nm, a reflective DM requires

450 nm DM surface P2V to 900 nm OPD P2V for 1 lambda/D

For

D = 6.5 m

this corresponds to approximately

lambda/D = 28.6 mas

at 900 nm.

The current nominal simulated disturbance of approximately

0.046 micron RMS DM surface P2V

corresponds to approximately

2.9 mas RMS pointing.

This provides a useful scale check but should not replace the actual MagAO-X DM calibration discovered during the software investigation.

## Requirements

1. Add optional DM tip/tilt disturbance generation to `accelSim`.

2. The DM disturbance and simulated accelerometer measurement must originate from the same underlying simulated vibration state.

3. Do not derive pointing by numerically integrating or double-integrating the final noisy `acc000` signal.

4. Preserve the physical relationship between modal acceleration and pointing using the nominal L = 10.0 m lever arm.

5. Use a nominal vibration scale of 1.0

6. Initially support only tip/tilt disturbance injection.

7. Use `scratch/accel_ts_sim_ddspc_cleaned.ipynb` as the reference for the validated offline physical model.

8. Use `dmPokeWFS.hpp` as the starting reference for investigating how DM commands should be handled within MagAO-X.

9. Follow existing MagAO-X DM command and calibration conventions rather than creating an independent DM-control mechanism when an appropriate mechanism already exists.

10. Determine the actual DM command units, dimensions, datatype, stream/interface, coordinate convention, sign convention, and calibration before implementation.

11. Do not assume that the Python notebook's 50 x 50 surface representation is the correct operational DM representation.

12. Determine whether an existing calibrated tip/tilt basis, modal representation, additive DM channel, or other existing mechanism should be used.

13. Do not generate a synthetic WFS signal.

14. Do not introduce an artificial WFS coupling coefficient.

15. The real WFS should measure the optical disturbance produced by the DM.

16. DM injection must be optional.

17. DM injection must be disabled by default.

18. When DM injection is disabled, existing `accelSim` behavior should remain unchanged.

19. Existing `acc000` timing and synchronization behavior must remain unchanged.

20. The accelerometer must remain capable of operating at its configured sampling rate, including the current 8 kHz operation with an approximately 2 kHz WFS synchronization reference.

21. Determine the appropriate DM update cadence from the existing MagAO-X DM architecture. Do not assume that the DM should update at the 8 kHz accelerometer sampling rate.

22. Determine how the DM disturbance should remain synchronized with the underlying simulated vibration state if the DM and accelerometer operate at different rates.

23. Do not improperly take ownership of or overwrite a DM stream controlled by another process.

24. Determine the correct method for combining the simulated disturbance with the normal DM/AO command path.

25. The simulated DM disturbance must be capable of being safely enabled, disabled, and zeroed.

26. Add only the configuration parameters required to control the DM simulation and its physical parameters.

27. Existing accelerometer simulation, ADC conversion, ImageStreamIO output, synchronization, and INDI behavior should not be modified unless required for this feature.

28. Clearly document all conversions between simulated pointing and the final MagAO-X DM command representation.

## Tests and Metrics

The implementation plan should include tests for the following.

### Physical Model

Verify the relationship

a_i(t) = -(2*pi*f_i)^2 * L * theta_i(t)

for the simulated vibration modes before sensor noise and ADC conversion.

Verify that the nominal physical configuration uses

L = 10.0 m
vibration scale = 1.0

and agrees with the offline Python reference.

### Python Comparison

Where practical, use deterministic seeds and equivalent vibration parameters to compare the physical vibration output from `accelSim` with

scratch/accel_ts_sim_ddspc_cleaned.ipynb

The goal is to verify the physical model rather than require identical Python and C++ implementations.

### Pointing-to-DM Conversion

Verify the conversion between pointing angle, OPD, and equivalent reflective-DM surface displacement.

### Nominal Disturbance Scale

Verify that the nominal physical model produces a disturbance consistent with the offline result of approximately

0.046 micron RMS DM surface P2V
2.9 mas RMS pointing

before any additional calibration required by the operational DM interface.

### DM-Disabled Regression

With DM injection disabled, verify that existing `accelSim` behavior is unchanged.

### Accelerometer Timing Regression

Verify that adding DM functionality does not change the configured `acc000` sampling rate or synchronization behavior.

In particular, the existing approximately

8 kHz accelerometer
2 kHz WFS synchronization reference

configuration should continue to operate correctly.

### DM Cadence

Verify that DM commands are updated at the cadence required by the existing MagAO-X DM architecture without forcing the accelerometer stream to use the same cadence.

### Zero Disturbance

With zero simulated pointing disturbance, the DM contribution should be zero.

Disabling or stopping DM injection should leave no persistent simulated disturbance.

### Tip/Tilt Sign and Orientation

Test positive and negative X and Y disturbances independently.

Verify the mapping between simulated tip/tilt and the actual DM/WFS coordinate convention.

Do not assume the signs or axis orientation before investigating the existing MagAO-X implementation.

### Stream Safety

Verify that the selected implementation does not overwrite or improperly take ownership of a DM stream controlled by another process.

### Long-Duration Operation

Verify that DM injection does not introduce:

- timing drift,
- command accumulation,
- memory growth,
- synchronization instability,
- or degradation of the existing accelerometer stream.

## Scope and Caveats

This task is limited to simulated tip/tilt disturbance injection.

Higher-order vibration modes or aberrations are out of scope.

Closed-loop DDSPC control is out of scope. This task is intended to provide the correlated accelerometer and optical disturbance required for later DDSPC testing.

Do not implement a synthetic WFS.

Do not numerically integrate the noisy accelerometer output to obtain pointing.

Do not introduce a synthetic WFS coupling factor.

Do not assume that the Python notebook's 50 x 50 DM surface should be transferred directly into MagAO-X.

The 6.5 m coordinate system used in the physical model represents the Magellan entrance pupil, not the physical dimensions of the DM.

`scratch/accel_ts_sim_ddspc_cleaned.ipynb` is the reference for the validated physical simulation.

`dmPokeWFS.hpp` is the initial reference for investigating the operational MagAO-X DM command mechanism.

Additional MagAO-X applications, libraries, configuration files, calibration files, and shared-memory streams should be investigated as necessary.

The Python notebook should not be copied mechanically into C++. Preserve its physical model while using the correct MagAO-X real-time architecture.

Do not send commands to physical DM hardware during automated testing.

Do not assume that writing directly to a final DM command stream is appropriate. Determine the intended disturbance/addition mechanism first.

# Instructions to Agent

Analyze the task above and create a plan to implement a solution.

Review `AGENTS.md`.

Review

    scratch/accel_ts_sim_ddspc_cleaned.ipynb

to understand the validated offline vibration and DM physics.

Then investigate `dmPokeWFS.hpp' and trace the relevant MagAO-X DM command architecture.

Inspect additional source files, applications, libraries, configuration files, calibration files, and shared-memory interfaces as necessary.

Before proposing implementation, determine:

1. How `dmPokeWFS` sends DM commands.
2. Which DM stream or interface should be used for this task.
3. What the command values physically represent and their units.
4. Whether tip/tilt should be represented in modal or actuator space.
5. What calibration or basis is required to convert the simulated pointing disturbance into a DM command.
6. How the simulated disturbance should be combined with the normal AO DM command.
7. What DM update cadence should be used.
8. How the DM updates should remain correlated with the underlying vibration state while `acc000` continues at its independently configured rate.
9. How DM injection should be enabled, disabled, and zeroed safely.
10. What new configuration parameters are required.
11. How the implementation can be tested without commanding physical DM hardware.

Treat the Python notebook as the reference for the physical model and the existing MagAO-X code as the reference for the operational implementation.

Document any unresolved unit, calibration, coordinate, sign, timing, stream-ownership, or safety questions.

Document your findings below under "Agent Findings and Plan".

Keep this document up to date as you work.

Do not alter any text above the "Agent Findings and Plan" section.

Do not begin implementation until the user has reviewed the completed plan and answered any questions.

# Agent Findings and Plan

## Task Summary

Investigation completed on 2026-11-01. **The user subsequently authorized independent local numerical implementation while deployed-system access is unavailable. Operational DM publication remains blocked; the acquisition application and its configuration will remain unchanged in this stage.** The current user instruction authorizes implementation without a separate plan-approval round when the investigation resolves the important questions, superseding the earlier review gate above. The pause at the operational writer boundary is specifically required by the user's instruction to ask about unresolved DM stream, calibration, ownership, timing, and safety questions.

Extend `accelSim` with an optional disturbance derived from its existing noise-free physical pointing state. Preserve its oscillator evolution, random draws, ADC conversion, `acc000` publication, timing, and phase synchronization. Use an exclusively assigned additive DM channel and the selected DM's calibrated actuator basis. Do not write the combined DM output, invent a WFS signal, or reproduce the notebook grid as an operational calibration.

### Sources inspected and command architecture

| Source | Finding and implication |
| --- | --- |
| `libMagAOX/app/dev/dmPokeWFS.hpp`, `allocate`, `basicTimedPoke`, `basicRunSensor` | Opens configured `pokecen.dmChannel` through `milkImage<float>`, sets `passive(true)`, sizes a local image from that stream, writes signed actuator pokes by image assignment, and writes zeros when the sequence ends or is stopped. This is a channel writer, not a DM driver or angular calibration. Its settling sleeps and WFS averaging are measurement procedures, not the required simulator cadence. |
| Local dependency `/Users/parkerjohnson/source/mxlib/include/improc/milkImage.hpp`, `open`, `operator=`, `post` | Opening requires the existing stream and matching float datatype. Assignment sets the write flag, copies the image, then posts. Active posting delegates to `ImageStreamIO_UpdateIm`. Passive posting updates `writetime`/`atime`, clears `write`, and **posts all semaphores but does not increment `cnt0`**. The member documentation says this avoids triggering cacao dmcomb until something else triggers it. A passive write is not a promise of immediate physical application. The installed instrument dependency version and dmcomb configuration remain unverified. |
| `apps/dmMode/dmMode.hpp` | Existing modal mechanism: load a FITS mode cube, form a weighted sum of actuator-space planes, and write an existing `dm.channelName`. It checks float datatype and mode/stream dimensions, then sets `write`, copies, increments `cnt0`, clears `write`, and posts semaphores. Mode coefficients inherit the supplied basis normalization; the app provides no universal conversion from sky angle to coefficient. |
| `apps/dmSpeckle/dmSpeckle.hpp`, `apps/sparkleClock/sparkleClock.hpp` | Existing disturbance generators use additive image channels and separate worker threads, with timer or external semaphore triggering. `dmSpeckle` also supports arbitrary FITS cubes. Its counter updates depend on trigger/delay mode, and stopping writes zeros. These are useful writer/lifecycle references, but neither solves the shared accelSim state or supplies calibrated tip/tilt. The 28-degree science-camera angle offset in these apps is not evidence for a WFS tip/tilt calibration. |
| `libMagAOX/app/dev/dm.hpp` | The driver monitors the combined `dm.shmimName` and calls the derived `commandDM`. Flat defaults to suffix `00`. The test-channel help says `01`, but `loadConfig` defaults to `02`; retrieve the deployed effective setting rather than assuming either reservation. It discovers the numbered channel images and supports `dm.deltaChannels` for computing the desaturated correction relative to non-delta channels. Assignment of the simulator to that accounting must be confirmed; it affects downstream correction interpretation. Its global `zeroAll` is inappropriate for stopping one simulator. |
| `apps/bmcCtrl/bmcCtrl.hpp`, `apps/alpaoCtrl/alpaoCtrl.hpp` | Continuous-face-sheet commands are float actuator displacement images in micron-based calibrated command units, not sky angles, raw voltage, or OPD. The drivers map image pixels to hardware actuator order and apply stroke/gain and influence-volume calibration. BMC applies `volume_factor/act_gain`, clamps, and square-roots for fractional voltage; ALPAO applies `volume_factor/max_stroke`, removes mean, and clamps. The command-to-surface calibration and active pupil cannot be replaced by an arbitrary planar pixel grid. Do not apply driver calibration a second time in accelSim. |
| `scripts/dmdispbridge` and local handbook `operating/software/utils/shmimTCP.rst` | Instrument conventions identify `dm00dispYY` as woofer (11 by 11) and `dm01dispYY` as tweeter (50 by 50), with bridge support. These are layout conventions, not evidence that a particular channel is free, locally driven, or correctly calibrated. The tweeter's numerical agreement with the notebook's 50 by 50 size does not establish equivalent pupil coordinates. |
| Local handbook `testbed/index.rst` | CACTI examples use channel 11 for turbulence, 03 for AO, and 00 for flat; its BMC DM is 32 by 32 and IrisAO uses piston/tip/tilt with mixed micron/milliradian units. Those testbed examples must not be adopted as MagAO-X channel reservations or calibration. |
| `apps/accelSim/accelSimModel.hpp`, `accelSim.hpp`, README, example config, and tests | `model::step` already sums physical pointing in arcsec separately from sensor noise and ADC counts. `m_pointing` is the correct source. `m_wfsPointing` retains legacy per-mode attenuation and must not be used for DM generation. The framegrabber owns `acc000` metadata/semaphores; its post-publication hook is available for a bounded state handoff. The phase clock continues independent sampling through missing references. |

The supported architecture is therefore: existing vibration state -> calibrated two-mode actuator image -> exclusively assigned additive `dmXXdispYY` -> existing cacao dmcomb -> combined stream -> existing DM controller -> optical WFS. The normal flat and AO channels remain independently owned. A modal basis is useful internally, while the channel receives the complete instantaneous actuator contribution, never an accumulated increment.

No deployed configuration/calibration was found in this checkout or the inspected sibling mxlib/handbook trees. `/opt/MagAOX`, `/opt/MagAOX/config`, and `/opt/MagAOX/calib` do not exist on this workstation. The actual cacao dmcomb implementation/configuration and ImageStreamIO C implementation were not available in those trees. Stream names, gains, trigger selection, active mode files, masks, and writer assignments consequently cannot be verified from this source checkout. No instrument connection or shared-memory access was attempted.

### Physical reference and differences

- The cleaned notebook contains the same 14 prominent acceleration modes and a 200-mode population spanning 100–900 Hz. Defaults are lever arm 10 m, vibration scale 1, and pupil diameter 6.5 m. Independent X/Y AR(2) pointing states produce modal acceleration through `a_i = -(2*pi*f_i)^2 * L * theta_i_rad`; sensor noise is added afterward.
- The cleaned notebook has no legacy per-mode optical attenuation. Its example `XY_TO_TT` is a positive 30-degree rotation. No inspected source establishes this rotation as the actual DM/WFS orientation. Retain all existing C++ model outputs and random evolution, but consume only `m_pointing` for the new feature.
- Convert arcsec to radians once. For physically calibrated pupil axes, `OPD = theta_x*x + theta_y*y` and reflective surface `h_um = 0.5e6 * OPD_m`. Across a circular pupil, surface P2V is `0.5 * D * hypot(theta_x_rad, theta_y_rad) * 1e6`. This is a physical validation formula, not a prescription to interpolate across an unknown actuator footprint.
- Prefer existing calibrated tip/tilt actuator planes. After verifying their units and normalization, define their conversion so each plane corresponds to a known angular displacement. Then synthesize `command = theta_x * basis_x + theta_y * basis_y`, with the verified coordinate transform either incorporated in the planes or applied once before synthesis. Do not assume Zernike index/order, unit RMS, unit P2V, an illuminated radius, or an extra reflection factor.
- Saved notebook outputs: radial pointing RMS **2.9076 mas**, surface P2V RMS **0.045814 micron**, maximum **9.5257 mas / 0.150091 micron**. The ideal-plane reconstruction reports errors near `5e-19` arcsec; that is a numerical geometry check, not a hardware calibration or a WFS signal to implement.
- Independent scale check gives `900 nm / 6.5 m = 28.5597424 mas`; 0.450 micron reflective surface P2V produces 0.900 micron OPD P2V.
- Existing C++ uses analytically scaled innovations and stationary initial histories; Python uses initially zero filter history and finite-record normalization after a burn interval. Their RNGs and weak-mode populations also differ. Preserve the C++ behavior as requested. The C++ stationary population predicts **2.939379 mas / 0.0463142 micron**. A fresh seed-42, 60-second C++ record gives **3.255637 mas / 0.0512973 micron**. Narrow, high-radius modes retain realization-to-realization amplitude variation. Exact finite-record equality to 2.9 mas is not an appropriate regression requirement and must not be forced by changing accelerometer amplitudes.

## Key Assumptions

- Use the current continuous C++ physical model unchanged, including its configured `sim.leverArm` (default 10 m) and unit vibration scale. No second oscillator bank and no integration of accelerometer samples.
- Implementation remains header-only with out-of-class definitions, matching this app and its tests. Use the existing framegrabber/telemeter interface macros. No style ambiguity needs resolution for this proposed structure.
- The operator must allocate the channel exclusively. ImageStreamIO's `write` flag, zero data, creator PID, or a name that looks like a disturbance channel does not establish exclusive writer ownership; network bridge receivers may also write it.
- The proposed basis and worker design below are conditional on the answers. No target DM, channel number, sky-axis transform, safe amplitude, or numeric DM rate is assumed.
- All automated tests use numerical models, mocks, or isolated workstation/Linux-development shared memory without hardware drivers or bridges. Agents do not execute on instrument hosts, issue INDI hardware commands, commit, or push.

## Requirements

1. Disabled by default; disabled operation must not open, zero, or otherwise access a DM stream. Preserve existing RNG sequence, physical/ADC outputs, INDI properties, and `acc000` creation and publication.
2. Preserve `waitUntil`, `waitSync`, `phaseClock`, deadlines, source-counter handling, catch-up, nominal rate changes, and framegrabber publication semantics. Keep 8 kHz acceleration sampling independent of a roughly 2 kHz reference.
3. Reuse the completed sample's physical pointing with a sequence identifier and acquisition/model-time information. The DM worker must never advance, restart, reseed, or read mutable oscillator state concurrently.
4. Synthesize only calibrated tip/tilt into a preallocated actuator image with verified type, dimensions, mask, normalization, signs, and axis orientation. Preserve the normal flat and AO contributions.
5. Perform stream connection, validation, calibration loading, image synthesis, waits, and semaphore publication outside the acquisition worker. The enabled acquisition path may only offer a bounded, nonblocking state copy; disabled operation bypasses that handoff.
6. Establish the DM trigger contract and accepted sample-to-application latency before implementation. Dropped snapshots must not queue stale commands or create replay bursts. A 2 kHz command rate is plausible for modes up to 900 Hz, but is not inferred from the WFS reference alone; sample-and-hold and latency affect the spectrum.
7. Enable only after validation and fresh state; disable/stop/fault must remove only the simulator contribution using the approved trigger/zero mechanism. A zero channel buffer does not prove that the mirror applied zero. Uncatchable process death cannot be handled solely by a destructor.
8. Document conversions, operational ownership, verified calibration provenance, limitations, and validation evidence. Keep changes in the working tree and update this plan.

## Questions and Points of Clarification

**Blocking operational questions sent to the user on 2026-11-01:**

1. **Target, ownership, and calibration:** Which physical DM and numbered additive channel should accelSim exclusively own? Supply the location of a local copy of the relevant controller/dmcomb configuration and tip/tilt calibration. Needed facts include dimensions/type, channel gains and existing writers (including bridges/disturbance apps), active actuator/pupil geometry, basis plane order and normalization, command units, simulated-X/Y to optical-axis sign/orientation, and `deltaChannels` classification. Existing `dmMode` basis files may be suitable if their angular normalization can be established. The notebook's 30-degree transform and the testbed's channel 11 are not substitutes for these facts.
2. **Trigger, cadence, and zeroing:** Should writes be passive and sampled by the existing AO trigger, or should accelSim actively trigger a specified DM rate? Identify the trigger source/semantics, intended update phase relative to WFS exposure/AO output, and behavior when that trigger stops. Confirm how a final zero is applied during AO inactivity and what supervisor/operator mechanism removes a held disturbance after an uncatchable simulator failure. Existing driver clipping limits total commands but does not by itself remove a stale simulator channel.

These are missing implementation inputs, not a request to reapprove already authorized work. Do not implement the DM writer or select operational defaults until they are resolved. Any needed live configuration inspection or hardware checks must be performed by the user; only local copies are inspected by the agent.

## Tests

### Investigation validation performed

| Check | Result |
| --- | --- |
| Build current numerical tests with `c++ -std=c++20 -O3 -fno-fast-math -Wall -Wextra -DCATCH_CONFIG_MAIN -DACCELSIM_MODEL_ONLY apps/accelSim/tests/accelSim_test.cpp -o /tmp/accelSim_dm_review_model_test` | Passed without diagnostic output. |
| `/tmp/accelSim_dm_review_model_test '[model]'` | **16 cases / 154423 assertions passed.** Covers physical conversion, ADC, deterministic sequences, stationary scaling, pacing, and phase-reference regression including 8000/2000 Hz, noninteger ratios, outages, and reacquisition. This is existing numerical coverage, not a runtime DM/timing test. |
| Existing hidden export test with `ACCELSIM_EXPORT_DIR=/tmp/accelSim-dm-review-export` and filter `'[accelSim-export]'` | **1 case / 6 assertions passed**, exporting 360000 samples, 214 modes, and metadata for 60 simulated seconds at 6000 Hz. No streams or hardware used. |
| Python standard-library calculation from exported modes and samples | Ensemble and realization pointing/P2V values recorded above; independent 900-nm scale agrees with the notebook. No optical calibration inferred. |
| Notebook inspection | Read all code cells and relevant saved numerical outputs. Did not execute the notebook; the default Python lacks NumPy/SciPy. Did not generate its movie or synthetic WFS outputs. |

No application build, live stream test, DM injection test, or instrument test is claimed. At the investigation-only stage, only the plan was changed; clang-format did not apply to that Markdown-only edit. `git diff --check` passed, and a comparison against HEAD confirmed that all text above Agent Findings and Plan is unchanged.

### Tests to add/run after the blockers are resolved

1. **Shared physics and disabled regression:** At fixed seeds, compare every physical/ADC sample and counters against the unchanged model with injection disabled and enabled. For a noise-free single mode verify the signed frequency-squared acceleration/pointing relation. Test X and Y independently, zero pointing, and rejection of nonfinite inputs. The disturbance must not include `m_wfsPointing` attenuation or ADC noise.
2. **Physical conversion and basis:** Use synthetic calibrated basis fixtures to verify arcsec/radian conversion, factor of two, pupil scale, positive/negative X/Y mapping, basis normalization, and zero output. Check dimensions, type, plane count, masks, finite values, and bad calibration errors. Confirm the real basis with operator-supplied calibration evidence; a unit test cannot establish optical orientation.
3. **Statistical comparison:** Compare ensemble predictions and spectra for equivalent mode parameters; use multiple seeds/statistical tolerances for narrow stationary modes. Check the roughly 0.046-micron / 2.9-mas nominal ensemble scale without renormalizing existing acceleration. Exact Python/C++ RNG identity is not required.
4. **Handoff and cadence:** Test coherent sequence/timestamp snapshots under concurrency, stale/missing samples, noninteger rates, missed DM deadlines, and no stale replay. Retain all existing phase-clock regressions unchanged. Verify DM outages or backpressure cannot wait in the acquisition path.
5. **Local ImageStreamIO fixture:** Use an isolated shared-memory directory, unique test names, no bridge, no hardware driver, and a fake combiner honoring the approved trigger contract. Verify datatype/shape checks, timestamps, write flag, `cnt0`, semaphore behavior, stream replacement, missing streams, and expected active/passive behavior. Verify only the assigned additive contribution changes and other channels/combined-stream ownership are preserved.
6. **Lifecycle/safety:** Verify default-disabled opens nothing; zero/disable/shutdown remove the contribution, repeated zero is harmless, stream replacement requires revalidation/rearming, and stale state/inactive triggers follow the agreed policy. Verify command limits if required by the supplied calibration. Test the chosen external crash-cleanup mechanism separately rather than claiming in-process cleanup covers SIGKILL.
7. **Runtime/load:** On an isolated non-instrument Linux development host, run the full app tests and sustained 8 kHz acceleration with a synthetic 2 kHz reference, with injection both off and on. Measure actual publication cadence, skipped samples, jitter, bounded handoff latency, resource growth, and drift. Run DM cadence independently at the approved rate. Numerical clock tests alone do not establish real-time performance.

Follow the existing `application_unit_test` group and namespace. Add TEST_CASE documentation and explicit real-API Doxygen references for harness indirection; hide harness-only helpers. Run clang-format on all touched C++ files after a full-file documentation pass.

## Implementation Plan

1. **Resolve operational inputs first.** Record the selected DM/channel, owner, calibration files/normalization, signs, pupil geometry, combiner gain/trigger policy, delta-channel role, rate/phase, and clearing/fault contract here. Inspect local copies of those files and the matching dependency/combiner version. No DM writer implementation before this step. The independent numerical work authorized below does not depend on it.
2. **Reuse physical state without modifying the model.** Read `m_model.m_sample.m_pointing` at a completed acquisition/publication boundary. Preserve all existing model functions and legacy fields. Offer a small coherent snapshot to a dedicated DM worker, with sequence and time markers, using preallocated storage and a bounded nonblocking producer operation. Do not add waits, image construction, calibration I/O, or DM stream operations to the accelerometer worker.
3. **Implement calibrated synthesis.** Use existing mxlib FITS/eigen facilities, as `dmMode` does, to load/validate the approved two-plane basis before enabling. Form a full contribution from each selected physical sample. Prefer a small app-local header for nontrivial writer/synthesis state if that keeps `accelSim.hpp` manageable and testable; do not introduce a new generic base class without a reusable need. Do not derive from the hardware-owning `dev::dm` or the WFS-poke measurement helper.
4. **Implement the agreed worker cadence and lifecycle.** Reuse app thread startup conventions and ImageStreamIO/milkImage mechanisms consistent with the selected trigger contract. Connect only to the existing assigned channel, validate it against the calibration, and detect replacement. Use the freshest eligible timestamped state according to the confirmed phase policy; quantify sample-hold delay. Require fresh state on enable and confirmed zero publication on disable/stop, with the approved trigger path and stale/failed-connection policy. Never invoke global zeroAll or alter the AO stream.
5. **Add minimal controls and documentation.** The following are proposed, not added or finalized:

   | Configuration/control | Proposed meaning |
   | --- | --- |
   | `dm.enabled=false` | Startup opt-in; also expose an INDI enable toggle and a zero request with explicit status. Disabled startup performs no DM access. |
   | `dm.channelName` (empty default) | Explicit exclusive additive channel; required to enable. No operational stream-name default. |
   | `dm.modeCube` (empty default) | Approved tip/tilt calibration in the existing FITS convention, with documented angular conversion. Required to enable. Add plane indices or a transform only if needed by the actual supplied basis. |
   | Cadence/trigger setting, name pending | Add only the option(s) needed by the selected contract: either a specified active update rate or the approved trigger stream/phase. No inferred 8 kHz or 2 kHz default. |

   Reuse existing `sim.leverArm=10`; do not add a second lever arm. Keep nominal vibration scale fixed at 1 initially; an independent DM gain must not be represented as changing the common vibration physics. The physical 6.5-m pupil scale belongs in verified calibration/conversion, not an arbitrary new actuator-grid generator. Add amplitude/staleness bounds only as required by the agreed operating contract.
6. **Validate and report.** Extend `apps/accelSim/tests/accelSim_test.cpp`, `apps/accelSim/config/example.conf`, and `apps/accelSim/README.md` along with `accelSim.hpp` and any justified app-local helper. Avoid changes to `accelSimModel.hpp`, shared framegrabber/synchronization code, DM drivers, or global startup configurations unless newly discovered evidence requires them. Run the tests above, keep this plan current, and leave all changes uncommitted.

## Follow-up and Edge Cases

- **Working-tree scope:** the independent numerical implementation and its tests/documentation described below, plus this plan below the Agent Findings and Plan heading. Application behavior, `acc000` timing, and configuration remain unchanged. No commits or pushes.
- Local code proves neither current stream ownership nor an instrument calibration. Multiple writers and bridged channels need operational exclusion; a local cooperative lock cannot exclude unrelated producers that ignore it.
- Writing zero passively may leave the last combined command applied if the normal trigger stops. Even an active zero depends on the deployed combiner trigger mode. Resolve this before claiming safe disable/stop behavior.
- `SIGKILL`, process/host failure, a dead combiner/controller, and disappearing streams require an external recovery policy. Hardware clipping is not stale-disturbance cleanup. Decide whether changes in controller state must inhibit injection to avoid repopulating a channel after an operator global zero.
- A lower DM rate introduces hold response, latency, and possible aliasing near the upper 900-Hz modes. Preserve the accelerometer physics; document any approved DM filtering separately rather than changing the shared state or silently dropping high-frequency modes.
- The AO loop may reject the injected optical disturbance, so a closed-loop WFS residual need not equal the open-loop command amplitude. Driver saturation and influence-function/pupil fitting also limit correspondence. Do not compensate using an invented WFS coupling factor.
- Live tests are user-operated: low-amplitude positive/negative axes, WFS/DM orientation and physical scale, additive operation with AO, actual command timing and latency, saturation margin, disable/zero during normal operation and trigger loss, and sustained operation. Agents remain on non-instrument workstations.


## Independent numerical implementation — authorized continuation

The user cannot retrieve deployed configuration yet and explicitly requested all safe local implementation and testing. This supersedes the earlier all-implementation pause, while preserving the operational questions. This stage implemented:

1. An app-local, header-only `accelSimDisturbance.hpp` numerical helper consuming the existing `sample::m_pointing` by const reference. It expresses physical pointing in **simulated X/Y coordinates**, provide OPD and reflective surface evaluation at supplied entrance-pupil coordinates, and circular-pupil surface P2V (nominal diameter 6.5 m). It does not change the model, RNG, sensor noise, ADC, or legacy coupled output.
2. A separate numerical two-plane actuator synthesis helper. Callers must supply dimensions and both calibration planes explicitly, in **native command units per radian of simulated X/Y pointing**, in native stream pixel order. This defines an input contract, not an instrument calibration. The planes must already incorporate measured orientation/sign, pupil fitting, reflection physics, and unit normalization. No default basis, geometric grid, factor-of-two reapplication, FITS convention, channel, or trigger will be assumed. Synthesis replaces the entire local contribution, does not accumulate, and has no stream access.
3. A separate application-unit-test file registered with the existing accelSim group and build system. Test physical sign/scale and the notebook's optical formulas, model immutability and RNG/ADC equivalence, nominal population scale, explicit synthetic orientation and non-square geometry, numerical validation, output replacement/zeroing, and storage reuse. Re-run existing model/phase-clock tests unchanged.
4. README documentation and a reproducible standalone test command. Use only local numerical fixtures; no physical hardware, shared-memory streams, synthetic WFS, worker threads, or publication.

**Exact remaining boundary:** connecting the helper to acquisition, selecting sample/trigger phase and DM cadence, inter-thread snapshot delivery, loading a deployed FITS calibration, app configuration/INDI enable controls, channel ownership checks, DM stream writes, applied zero confirmation, and fault/supervisor behavior remain unimplemented. These depend on the unavailable operational contract. There is no functional `dm.enabled` option in this stage; injection remains unavailable and consequently off. The eventual channel and calibration will be explicit configuration inputs, never hard-coded defaults. No fake option that appears to enable a DM is added.

### Results and review boundary

Implemented `pointingDisturbance` and `tipTiltBasis` in `apps/accelSim/accelSimDisturbance.hpp`. Both are independent of the MagAO-X runtime and consume caller-provided values only. No production app header includes the helper yet, and there are no new configuration keys. `pointingDisturbance` copies the physical state before any optical-axis interpretation; `tipTiltBasis` applies only explicitly supplied planes. Initial/local zero and synthesis-overflow cleanup affect memory only. Returned vector references are owned by the basis object, so a future publisher must obey the documented lifetime/thread-ownership contract.

The numeric input contract allows arbitrary supplied calibration geometry and simulated-X/Y orientation; it does not claim that any provided calibration has been optically validated. Empty/mismatched/overflowed dimensions and nonfinite planes are rejected. Synthesis uses preallocated storage, replaces the entire image, and clears all output before throwing on nonfinite or float-overflowed commands. Zero/dependent calibration planes are not interpreted as physically valid or invalid by this numerical layer; optical validity is a deployment responsibility.

Changed files:

- `apps/accelSim/accelSimDisturbance.hpp`: physical quantities and explicit-basis synthesis.
- `apps/accelSim/tests/accelSimDisturbance_test.cpp`: eight hardware-free cases under the existing `accelSim_unit_test` / `application_unit_test` group.
- `tests/Makefile.one`: numerical floating-point flags and header dependencies for the new suite.
- `tests/tests.list`: register the new suite.
- `apps/accelSim/README.md`: helper input/output contracts, standalone test command, and unavailable runtime controls.
- This plan, below Agent Findings and Plan only.

| Validation | Result |
| --- | --- |
| Optimized standalone new suite, C++20/O3/`-fno-fast-math`/`-Wall -Wextra` | **8 cases / 8244 assertions passed.** |
| AddressSanitizer + UndefinedBehaviorSanitizer, O1/debug/`-fno-fast-math`/`-fno-omit-frame-pointer` | **8 cases / 8244 assertions passed**, no sanitizer findings. |
| Existing unchanged numerical model/phase-clock suite | **16 cases / 154423 assertions passed**, including 8 kHz acceleration / 2 kHz phase-reference tests. |
| Fresh combined numerical build using `tests/testMain.cpp` | **24 cases / 162667 assertions passed.** Includes deterministic model/ADC/counter equality with repeated disturbance reads and synthesis, skipped ticks, stationary recovery, and 6000-to-8000 Hz rate changes. |
| Direct cleaned-notebook comparison | Generated 1000 existing-model samples through the new C++ helper. Executed the notebook's actual cell-11 surface function with those same simulated-axis angles and compared coordinate `[32,7]` of its 50-by-50 validation grid plus the cell-12 circular-pupil P2V formula. Passed `rtol=1e-12`, `atol=1e-16`; maximum absolute surface error **3.46945e-18 micron**, P2V error **2.77556e-17 micron**. No example rotation was assumed to be instrument calibration. |
| Formatting/documentation/scope | clang-format 23.1.3 applied and `--dry-run --Werror` passed on both new C++ files. Full-file declaration/member/test documentation checked. `git diff --check` passed. Byte comparisons against HEAD confirmed the specification prefix and existing app/model/config/test files unchanged. |

Reproducible combined numerical command (run from repository root):

```sh
c++ -std=c++20 -O3 -fno-fast-math -Wall -Wextra -DACCELSIM_MODEL_ONLY \
  tests/testMain.cpp apps/accelSim/tests/accelSim_test.cpp \
  apps/accelSim/tests/accelSimDisturbance_test.cpp -o /tmp/accelSim_combined_test
/tmp/accelSim_combined_test '[accelSim]'
```

The initial combined-build invocation incorrectly defined `CATCH_CONFIG_MAIN` for both test translation units and failed with duplicate Catch symbols. Using the repository's single `tests/testMain.cpp` corrected the command; no source workaround was needed. New-suite sanitizer binary: `/tmp/accelSim_disturbance_sanitize`. Direct notebook comparison source/executable/data: `/tmp/accelSim-disturbance-compare.cpp`, `/tmp/accelSim-disturbance-compare`, `/tmp/accelSim-disturbance-compare.csv`. The comparison used available `/opt/anaconda3/bin/python` (NumPy 2.3.5); it did not execute the notebook's full simulator/movie/WFS cells. clang-format was installed with network approval in `/tmp/accelsim-dm-tools`, without project dependency changes.

No full MagAO-X runtime build, shared-memory publication, instrument operation, wall-clock performance test, or live zeroing result is claimed. The application source is unchanged, so disabled-path preservation is structural as well as numerically tested; this is not evidence of the future enabled path's real-time performance. All deployment questions and the exact runtime boundary above remain open. No commits or pushes were made.
