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

Latest implementation update: 2026-11-02. **The user authorized runtime implementation and supplied the deployed combiner contract: changed summed additive `cnt0` causes recombination, AveMode=0, DClevel=0, process trigger mode 4 with 10-us delay. Active same-sample 8-kHz injection is now implemented for an explicitly configured existing channel, intended `dm01disp10`.** Startup enable is prohibited and scale defaults to zero. Preparation and final zero use the native active ImageStreamIO operation; no extra worker, model advance, or random draw is added. The original numerical projection remains unchanged. Only workstation numerical/mock tests have run; the full app/native-stream build is blocked by pre-existing macOS dependencies. See the final runtime implementation section for validation and commissioning limits. Preserve the earlier plan edits; no staging, commits, pushes, instrument access, deployed process changes, or physical stream writes.

The "Direct physical DM-pupil calibration" section records the accepted physical mapping. The "Channel 10 and same-sample 8-kHz publication" section records the preceding design; its implementation gate was subsequently resolved by the user's deployed-system evidence and explicit authorization below. Earlier requirements for a separate worker, passive/lower-rate publication, or no runtime implementation are superseded. The original task specification above Agent Findings and Plan is retained verbatim; newer user instructions govern cadence, ownership, and implementation scope. No external Az/El, sky, or WFS registration is claimed, and the telescope-secondary AEG chain is excluded.

Jared Males subsequently confirmed that existing MagAO-X/DM command protection is
sufficient for this injection. The instrument-owner update at the end of this
record removes the provisional actuator/P2V limiter. `dm.scale` remains an
experiment amplitude control, not a hardware safety mechanism.

Extend `accelSim` with an optional disturbance derived from its existing noise-free physical pointing state. Preserve its oscillator evolution, random draws, ADC conversion, `acc000` publication, timing, and phase synchronization. Use an exclusively assigned additive DM channel and the selected DM's calibrated actuator basis. Do not write the combined DM output, invent a WFS signal, or reproduce the notebook grid as an operational calibration.

### Sources inspected and command architecture

| Source | Finding and implication |
| --- | --- |
| `libMagAOX/app/dev/dmPokeWFS.hpp`, `allocate`, `basicTimedPoke`, `basicRunSensor` | Opens configured `pokecen.dmChannel` through `milkImage<float>`, sets `passive(true)`, sizes a local image from that stream, writes signed actuator pokes by image assignment, and writes zeros when the sequence ends or is stopped. This is a channel writer, not a DM driver or angular calibration. Its settling sleeps and WFS averaging are measurement procedures, not the required simulator cadence. |
| Local dependency `/Users/parkerjohnson/source/mxlib/include/improc/milkImage.hpp`, `open`, `operator=`, `post` | Opening requires the existing stream and matching float datatype. Assignment sets the write flag, copies the image, then posts. Active posting delegates to `ImageStreamIO_UpdateIm`. Passive posting updates `writetime`/`atime`, clears `write`, and **posts all semaphores but does not increment `cnt0`**. The member documentation says this avoids triggering cacao dmcomb until something else triggers it. A passive write is not a promise of immediate physical application. The installed instrument dependency version and dmcomb configuration remain unverified. |
| `apps/dmMode/dmMode.hpp` | Existing modal mechanism: load a FITS mode cube, form a weighted sum of actuator-space planes, and write an existing `dm.channelName`. It checks float datatype and mode/stream dimensions, then sets `write`, copies, increments `cnt0`, clears `write`, and posts semaphores. Mode coefficients inherit the supplied basis normalization; the app provides no universal conversion from sky angle to coefficient. |
| `apps/dmSpeckle/dmSpeckle.hpp`, `apps/sparkleClock/sparkleClock.hpp` | Existing disturbance generators use additive image channels and separate worker threads, with timer or external semaphore triggering. `dmSpeckle` also supports arbitrary FITS cubes. Its counter updates depend on trigger/delay mode, and stopping writes zeros. These are useful writer/lifecycle references, but neither solves the shared accelSim state or supplies calibrated tip/tilt. The 28-degree science-camera angle offset in these apps is not evidence for a WFS tip/tilt calibration. |
| `libMagAOX/app/dev/dm.hpp` | The driver monitors the combined `dm.shmimName` and calls the derived `commandDM`. Flat defaults to suffix `00`. The test-channel help says `01`, but `loadConfig` defaults to `02`; retrieve the deployed effective setting rather than assuming either reservation. It discovers the numbered channel images and supports `dm.deltaChannels` for computing the desaturated correction relative to non-delta channels. Assignment of the simulator to that accounting must be confirmed; it affects downstream correction interpretation. Its global `zeroAll` is inappropriate for stopping one simulator. |
| `apps/bmcCtrl/bmcCtrl.hpp`, `apps/alpaoCtrl/alpaoCtrl.hpp` | For the tweeter, the user confirms float actuator images in **microns of physical DM surface**, not sky angles, raw voltage, or OPD. The drivers map image pixels to hardware actuator order and apply stroke/gain and influence-volume calibration. BMC applies `volume_factor/act_gain`, clamps, and square-roots for fractional voltage; ALPAO applies `volume_factor/max_stroke`, removes mean, and clamps. The confirmed TT cube supplies surface-micron modes; its pupil registration still cannot be replaced by an arbitrary planar pixel grid. Do not apply driver calibration a second time in accelSim. |
| `scripts/dmdispbridge` and local handbook `operating/software/utils/shmimTCP.rst` | Instrument conventions identify `dm00dispYY` as woofer (11 by 11) and `dm01dispYY` as tweeter (50 by 50), with bridge support. These are layout conventions, not evidence that a particular channel is free, locally driven, or correctly calibrated. The tweeter's numerical agreement with the notebook's 50 by 50 size does not establish equivalent pupil coordinates. |
| Local handbook `testbed/index.rst` | CACTI examples use channel 11 for turbulence, 03 for AO, and 00 for flat; its BMC DM is 32 by 32 and IrisAO uses piston/tip/tilt with mixed micron/milliradian units. Those testbed examples must not be adopted as MagAO-X channel reservations or calibration. |
| `apps/accelSim/accelSimModel.hpp`, `accelSim.hpp`, README, example config, and tests | `model::step` already sums physical pointing in arcsec separately from sensor noise and ADC counts. `m_pointing` is the correct source. `m_wfsPointing` retains legacy per-mode attenuation and must not be used for DM generation. The framegrabber owns `acc000` metadata/semaphores; its post-publication hook is available for a bounded state handoff. The phase clock continues independent sampling through missing references. |

The supported architecture is therefore: existing vibration state -> calibrated two-mode actuator image -> exclusively assigned additive `dmXXdispYY` -> existing cacao dmcomb -> combined stream -> existing DM controller -> optical WFS. The normal flat and AO channels remain independently owned. A modal basis is useful internally, while the channel receives the complete instantaneous actuator contribution, never an accumulated increment.

During the initial investigation, no deployed configuration/calibration was found in this checkout or the inspected sibling mxlib/handbook trees. `/opt/MagAOX`, `/opt/MagAOX/config`, and `/opt/MagAOX/calib` do not exist on this workstation. The user subsequently supplied the calibration files and physical geometry, and then assigned channel 10 at 8 kHz. The actual deployed cacao dmcomb implementation/configuration, ImageStreamIO C implementation, and tweeterModulator executable/configuration mapping are still unavailable locally as of 2026-11-02. Gains, trigger behavior, and exclusive handover from existing writers remain to be verified. No instrument connection or shared-memory access was attempted.

### Physical reference and differences

- The cleaned notebook contains the same 14 prominent acceleration modes and a 200-mode population spanning 100–900 Hz. Defaults are lever arm 10 m, vibration scale 1, and pupil diameter 6.5 m. Independent X/Y AR(2) pointing states produce modal acceleration through `a_i = -(2*pi*f_i)^2 * L * theta_i_rad`; sensor noise is added afterward.
- The cleaned notebook has no legacy per-mode optical attenuation. Its example `XY_TO_TT` is a positive 30-degree rotation. No inspected source establishes this rotation as the actual DM/WFS orientation. Retain all existing C++ model outputs and random evolution, but consume only `m_pointing` for the new feature.
- Convert arcsec to radians once. For physically calibrated pupil axes, `OPD = theta_x*x + theta_y*y` and reflective surface `h_um = 0.5e6 * OPD_m`. Across a circular pupil, surface P2V is `0.5 * D * hypot(theta_x_rad, theta_y_rad) * 1e6`. This is a physical validation formula, not a prescription to interpolate across an unknown actuator footprint.
- Prefer existing calibrated tip/tilt actuator planes. Use the confirmed surface-micron/RMS-one normalization and derive the angular response by projecting physical pupil-coordinate planes onto the supplied modes, as implemented below. Then synthesize `command = theta_x * basis_x + theta_y * basis_y`, with the verified coordinate transform either incorporated in the planes or applied once before synthesis. The supplied CMmodesDM.00 pair has unit RMS surface over dmmask and measured masked P2V of approximately 3.8492 and 3.8128 microns. P2V/RMS = 4 is the nominal ideal continuous-pupil pure-tilt relation, not an exact sampled-grid requirement; do not assume its X/Y order/sign, registered illuminated radius in pixels, or an extra reflection factor.
- Saved notebook outputs: radial pointing RMS **2.9076 mas**, surface P2V RMS **0.045814 micron**, maximum **9.5257 mas / 0.150091 micron**. The ideal-plane reconstruction reports errors near `5e-19` arcsec; that is a numerical geometry check, not a hardware calibration or a WFS signal to implement.
- Independent scale check gives `900 nm / 6.5 m = 28.5597424 mas`; 0.450 micron reflective surface P2V produces 0.900 micron OPD P2V.
- Existing C++ uses analytically scaled innovations and stationary initial histories; Python uses initially zero filter history and finite-record normalization after a burn interval. Their RNGs and weak-mode populations also differ. Preserve the C++ behavior as requested. The C++ stationary population predicts **2.939379 mas / 0.0463142 micron**. A fresh seed-42, 60-second C++ record gives **3.255637 mas / 0.0512973 micron**. Narrow, high-radius modes retain realization-to-realization amplitude variation. Exact finite-record equality to 2.9 mas is not an appropriate regression requirement and must not be forced by changing accelerometer amplitudes.

## Key Assumptions

- Use the current continuous C++ physical model unchanged, including its configured `sim.leverArm` (default 10 m) and unit vibration scale. No second oscillator bank and no integration of accelerometer samples.
- Implementation remains header-only with out-of-class definitions, matching this app and its tests. Use the existing framegrabber/telemeter interface macros. No style ambiguity needs resolution for this proposed structure.
- Channel 10 is operationally assigned for this experiment. The operator must complete exclusive handover from tweeterModulator/turbulence and any bridge writers before accelSim even zeros it. ImageStreamIO's `write` flag, zero data, or creator PID is not an ownership lock.
- Both outputs target 8 kHz on RTC/exao2. The approximately 2-kHz WFS stream remains the existing phase reference, not the DM publication clock. The publication mechanism, actual combiner/driver throughput, and direct-path timing remain conditional on evidence.
- All automated tests use numerical models, mocks, or isolated workstation/Linux-development shared memory without hardware drivers or bridges. Agents do not execute on instrument hosts, issue INDI hardware commands, commit, or push.

## Requirements

1. Injection disabled and scale zero by default. With no configured/owned channel, perform no DM access. Once channel 10 is explicitly configured and exclusively handed over, validate it and publish startup zero even while disabled. Disabled does not mean ownership has been returned to tweeterModulator. Preserve existing RNG sequence, physical/ADC outputs, and `acc000` creation and publication.
2. Preserve `waitUntil`, `waitSync`, `phaseClock`, deadlines, source-counter handling, catch-up, nominal rate changes, and framegrabber publication semantics. Keep 8 kHz acceleration sampling independent of a roughly 2 kHz reference.
3. Reuse each successfully published accelerometer sample's physical `m_pointing`, with sequence and acquisition/model-time information. No additional model advancement, random draws, integration of ADC data, or use of `m_wfsPointing`. If a worker is necessary, pass an immutable sample snapshot rather than read mutable model state concurrently.
4. Synthesize only calibrated tip/tilt into a preallocated actuator image with verified type, dimensions, mask, normalization, signs, and axis orientation. Preserve the normal flat and AO contributions.
5. Load and validate calibration, solve the projection, connect streams, and preallocate/prefault buffers before enabling. Prefer bounded synthesis and publication in `frameGrabberPostPublish` after the corresponding ADC frame. No hot-path allocations, blocking locks, waits, filesystem I/O, calibration solves, or synchronous logging/INDI. Verify the installed publication helper before treating it as nonblocking. Use a preallocated nonblocking handoff only if the direct path cannot meet the timing budget.
6. Publish one complete contribution per successful 8-kHz accelerometer sample while enabled/healthy, not one per WFS frame. Do not publish catch-up-only model steps or replay skipped frames. Verify whether active channel-10 writes cause each required combination; do not confuse an 8-kHz producer counter with measured combiner or hardware throughput. A failure to keep up must fault/inhibit injection without delaying `acc000`, not silently degrade to a lower-rate experiment.
7. Enable only after validation and fresh state; disable/stop/fault must remove only the simulator contribution using the approved trigger/zero mechanism. A zero channel buffer does not prove that the mirror applied zero. Uncatchable process death cannot be handled solely by a destructor.
8. Document conversions, operational ownership, verified calibration provenance, limitations, and validation evidence. Keep changes in the working tree and update this plan.

## Questions and Points of Clarification

**Remaining operational inputs (updated after Jared's channel/rate guidance):**

1. **Deployed writer identity and handover:** provide the RTC process-list entry, resolved tweeterModulator configuration/executable/version, channel-10 bridge/other writers, combiner channel gain, and `deltaChannels` classification. Channel `dm01disp10`, RTC host, and requested 8-kHz publication are already decided; do not ask to select them again.
2. **Publication and zeroing semantics:** provide the deployed DMch2disp source/version/effective FPS settings and matching ImageStreamIO implementation. Establish counter/semaphore/trigger behavior, rereading/caching of channels, concurrent-copy handling, output posting, rate limiting, and zero application without an AO trigger. Determine how concurrent AO updates affect total output rate. Existing driver clipping does not remove a stale simulator channel.
3. **Acceptance and timing:** finalize runtime calibration identity/residual tolerance. Measure direct publication latency and `acc000` deadline effects on an isolated suitable development host. Establish the external crash-cleanup procedure; no destructor can guarantee cleanup after process/host failure.

These are missing implementation inputs, not a request to reapprove the channel or cadence. Do not implement the DM writer until publication semantics are understood. Any needed live configuration inspection or hardware checks must be performed by the user; only local copies are inspected by the agent.

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
4. **Same-sample cadence:** Match each successful enabled ADC frame to its DM source sequence/timestamp at 8 kHz. Test missing frames, catch-up-only model steps, rate changes, stale state, and no replay. If a handoff is needed, test coherence, overflow, and explicit faulting rather than silent loss. Retain existing phase-clock regressions. Verify DM failures cannot wait in or terminate accelerometer acquisition.
5. **Local ImageStreamIO fixture:** Use an isolated shared-memory directory, unique test names, no bridge, no hardware driver, and a fake combiner honoring the approved trigger contract. Verify datatype/shape checks, timestamps, write flag, `cnt0`, semaphore behavior, stream replacement, missing streams, and expected active/passive behavior. Verify only the assigned additive contribution changes and other channels/combined-stream ownership are preserved.
6. **Lifecycle/safety:** Verify unset/unowned startup accesses no DM; configured/owned startup explicitly zeros while disabled. Test enable, scale zero, disable/normal shutdown, repeated zero, concurrent control requests, stream replacement, calibration invalidation, and no later nonzero publication after disable acknowledgement. Test external crash cleanup independently of in-process cleanup.
7. **Runtime/load:** On an isolated non-instrument Linux development host, run full app tests and sustained 8-kHz ADC plus 8-kHz DM publication with a synthetic 2-kHz reference. Compare disabled/enabled timing, skips, jitter tails, sample-to-DM latency, CPU/RSS, and allocations. Exercise concurrent synthetic AO updates and the matching combiner with no hardware driver. Measure distinct producer, combiner, and dummy-driver counts; numerical clock tests alone do not establish throughput or unique-frame delivery.

Follow the existing `application_unit_test` group and namespace. Add TEST_CASE documentation and explicit real-API Doxygen references for harness indirection; hide harness-only helpers. Run clang-format on all touched C++ files after a full-file documentation pass.

## Implementation Plan

1. **Resolve remaining operational inputs first.** Channel 10 at 8 kHz is assigned. Inspect matching local copies of the deployed modulator, combiner, ImageStreamIO, effective configuration, and clearing/fault contract. No DM writer implementation before these publication semantics are understood.
2. **Reuse physical state without modifying the model.** Prefer `frameGrabberPostPublish` on the existing acquisition thread: after ADC publication, consume that same `m_model.m_sample` and publish the matching image. Preserve all model/timing functions. This hook still consumes the next deadline's budget, so code inspection and measured worst-case timing must justify direct publication. If necessary, use a preallocated bounded SPSC handoff carrying the same sample identity; overflow/staleness faults the injection instead of blocking acquisition or silently decimating it.
3. **Implement calibrated synthesis.** Use existing mxlib FITS/eigen facilities, as `dmMode` does, to load/validate the approved two-plane basis before enabling. Form a full contribution from each selected physical sample. Prefer a small app-local header for nontrivial writer/synthesis state if that keeps `accelSim.hpp` manageable and testable; do not introduce a new generic base class without a reusable need. Do not derive from the hardware-owning `dev::dm` or the WFS-poke measurement helper.
4. **Implement the agreed publication and lifecycle.** Reuse native ImageStreamIO/milkImage mechanisms consistent with the verified channel-10 contract. Keep one writer, fixed buffers, sample identity, and separate acquisition/write timestamps where supported. Prefer no additional thread; if timing requires one, it consumes each snapshot promptly at the producer's 8-kHz cadence, not the WFS cadence. Connect only to existing channel 10 after exclusive handover; validate and zero before enabling. Serialize startup/disable/fault/shutdown zeros with nonzero writes. Never invoke global zeroAll or alter the AO stream.
5. **Add minimal controls and documentation.** The following are proposed, not added or finalized:

   | Configuration/control | Proposed meaning |
   | --- | --- |
   | `dm.enabled=false` | Injection off by default; runtime INDI toggle/status. Explicitly configured and owned startup publishes zero before any enable, including startup opt-in. |
   | `dm.channelName` (empty default) | Experiment configuration explicitly supplies `dm01disp10`; empty means no DM access. This is not a request to reselect the channel. |
   | `dm.scale=0` | Runtime current/target in [0,1], applied only to the DM contribution. Never changes model/RNG/ADC amplitudes. |
   | `dm.modeCube` (empty default) | Approved tip/tilt calibration in the existing FITS convention, with documented angular conversion. Required to enable. Add plane indices or a transform only if needed by the actual supplied basis. |
   | Cadence | No independent WFS-paced DM rate. One publication per successful 8-kHz ADC sample. Initially require the configured/active ADC rate to be 8000 Hz for injection; a rate change inhibits injection without overriding existing ADC rate behavior. |

   Reuse existing `sim.leverArm=10`; do not add a second lever arm. Keep nominal vibration scale fixed at 1 initially; an independent DM gain must not be represented as changing the common vibration physics. The physical 6.5-m pupil scale belongs in verified calibration/conversion, not an arbitrary new actuator-grid generator. Preserve the agreed synchronization health checks.
6. **Validate and report.** Extend `apps/accelSim/tests/accelSim_test.cpp`, `apps/accelSim/config/example.conf`, and `apps/accelSim/README.md` along with `accelSim.hpp` and any justified app-local helper. Avoid changes to `accelSimModel.hpp`, shared framegrabber/synchronization code, DM drivers, or global startup configurations unless newly discovered evidence requires them. Run the tests above, keep this plan current, and leave all changes uncommitted.

## Follow-up and Edge Cases

- **Current working-tree scope:** this plan below the Agent Findings and Plan heading only. The numerical implementation is already committed/pushed per the user. Application behavior, `acc000` timing, and configuration remain unchanged. No new commits or pushes.
- Local code proves neither current stream ownership nor an instrument calibration. Multiple writers and bridged channels need operational exclusion; a local cooperative lock cannot exclude unrelated producers that ignore it.
- Writing zero passively may leave the last combined command applied if the normal trigger stops. Even an active zero depends on the deployed combiner trigger mode. Resolve this before claiming safe disable/stop behavior.
- `SIGKILL`, process/host failure, a dead combiner/controller, and disappearing streams require an external recovery policy. Hardware clipping is not stale-disturbance cleanup. Decide whether changes in controller state must inhibit injection to avoid repopulating a channel after an operator global zero.
- Publishing at 8 kHz does not make the approximately 2-kHz WFS measure 8,000 independent exposures per second. Preserve the physical state and characterize DM hold, latency, WFS integration, and AO rejection without introducing synthetic WFS attenuation or silently changing DM cadence.
- The AO loop may reject the injected optical disturbance, so a closed-loop WFS residual need not equal the open-loop command amplitude. Driver saturation and influence-function/pupil fitting also limit correspondence. Do not compensate using an invented WFS coupling factor.
- Live tests are user-operated: low-amplitude positive/negative axes, WFS/DM orientation and physical scale, additive operation with AO, actual command timing and latency, saturation margin, disable/zero during normal operation and trigger loss, and sustained operation. Agents remain on non-instrument workstations.


## Independent numerical implementation — authorized continuation

At the independent numerical stage, the user could not retrieve deployed configuration yet and explicitly requested all safe local implementation and testing. This supersedes the earlier all-implementation pause, while preserving the operational questions. This stage implemented:

1. An app-local, header-only `accelSimDisturbance.hpp` numerical helper consuming the existing `sample::m_pointing` by const reference. It expresses physical pointing in **simulated X/Y coordinates**, provide OPD and reflective surface evaluation at supplied entrance-pupil coordinates, and circular-pupil surface P2V (nominal diameter 6.5 m). It does not change the model, RNG, sensor noise, ADC, or legacy coupled output.
2. A separate numerical two-plane actuator synthesis helper. Callers must supply dimensions and both calibration planes explicitly, in **surface microns per radian of simulated X/Y pointing** (units subsequently confirmed by the user), in native stream pixel order. This defines an input contract, not an instrument calibration. At this stage the planes had to incorporate measured orientation/sign, pupil fitting, reflection physics, and unit normalization. The subsequent `tipTiltProjection` layer below now derives them from the confirmed surface-micron modes and supplied registered pupil coordinates. No default basis, geometric grid, factor-of-two reapplication, FITS convention, channel, or trigger will be assumed. Synthesis replaces the entire local contribution, does not accumulate, and has no stream access.
3. A separate application-unit-test file registered with the existing accelSim group and build system. Test physical sign/scale and the notebook's optical formulas, model immutability and RNG/ADC equivalence, nominal population scale, explicit synthetic orientation and non-square geometry, numerical validation, output replacement/zeroing, and storage reuse. Re-run existing model/phase-clock tests unchanged.
4. README documentation and a reproducible standalone test command. Use only local numerical fixtures; no physical hardware, shared-memory streams, synthetic WFS, worker threads, or publication.

**Exact remaining boundary:** connecting the helper to acquisition, selecting sample/trigger phase and DM cadence, inter-thread snapshot delivery, loading a deployed FITS calibration, app configuration/INDI enable controls, channel ownership checks, DM stream writes, applied zero confirmation, and fault/supervisor behavior remain unimplemented. Stream publication depends on the unavailable operational contract; offline physical projection and normalization do not. There is no functional `dm.enabled` option in this stage; injection remains unavailable and consequently off. The eventual channel and calibration will be explicit configuration inputs, never hard-coded defaults. No fake option that appears to enable a DM is added.

### Results and review boundary

Implemented `pointingDisturbance` and `tipTiltBasis` in `apps/accelSim/accelSimDisturbance.hpp`. Both are independent of the MagAO-X runtime and consume caller-provided values only. No production app header includes the helper yet, and there are no new configuration keys. `pointingDisturbance` copies the physical state before any optical-axis interpretation; `tipTiltBasis` applies only explicitly supplied planes. Initial/local zero and synthesis-overflow cleanup affect memory only. Returned vector references are owned by the basis object, so a future publisher must obey the documented lifetime/thread-ownership contract.

The numeric input contract allows arbitrary supplied calibration geometry and simulated-X/Y orientation; it does not claim that any provided calibration has been optically validated. Empty/mismatched/overflowed dimensions and nonfinite planes are rejected. Synthesis uses preallocated storage, replaces the entire image, and clears all output before throwing on nonfinite or float-overflowed commands. The low-level angular-response synthesizer does not interpret zero/dependent response planes. The subsequent projection layer rejects rank-deficient input modes or coordinate geometry and checks the physical fit residual; optical validity remains a deployment responsibility.

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

Reproducible combined numerical command (run from repository root; Eigen include flags added for the projection layer):

```sh
c++ -std=c++20 -O3 -fno-fast-math -Wall -Wextra -DACCELSIM_MODEL_ONLY \
  $(pkg-config --cflags eigen3) \
  tests/testMain.cpp apps/accelSim/tests/accelSim_test.cpp \
  apps/accelSim/tests/accelSimDisturbance_test.cpp -o /tmp/accelSim_combined_test
/tmp/accelSim_combined_test '[accelSim]'
```

The initial combined-build invocation incorrectly defined `CATCH_CONFIG_MAIN` for both test translation units and failed with duplicate Catch symbols. Using the repository's single `tests/testMain.cpp` corrected the command; no source workaround was needed. New-suite sanitizer binary: `/tmp/accelSim_disturbance_sanitize`. Direct notebook comparison source/executable/data: `/tmp/accelSim-disturbance-compare.cpp`, `/tmp/accelSim-disturbance-compare`, `/tmp/accelSim-disturbance-compare.csv`. The comparison used available `/opt/anaconda3/bin/python` (NumPy 2.3.5); it did not execute the notebook's full simulator/movie/WFS cells. clang-format was installed with network approval in `/tmp/accelsim-dm-tools`, without project dependency changes.

No full MagAO-X runtime build, shared-memory publication, instrument operation, wall-clock performance test, or live zeroing result is claimed. The application source is unchanged, so disabled-path preservation is structural as well as numerically tested; this is not evidence of the future enabled path's real-time performance. The exact runtime boundary remains open; the confirmed units/normalization and subsequent projection work below supersede those earlier unknowns. No commits or pushes were made.


## Confirmed tweeter calibration and projection implementation

The user supplied the following deployed calibration facts after the independent numerical stage. They supersede earlier unknown-unit/normalization statements: all `dm01dispXX` values are **microns of physical DM surface displacement**. `/opt/MagAOX/cacao/tweeter-vispyr/tweeter-vispyr-calibs/default/CMmodesDM/CMmodesDM.00.fits` is a `(2,50,50)` pair of approximately orthogonal rotated TT modes, each RMS-normalized to **1 micron of DM surface over the deployed dmmask**. **P2V = 4 * RMS** is the nominal pure-tilt relation for a uniformly weighted continuous circular pupil, not an exact requirement on a sampled actuator grid. Local validation measured masked P2V of approximately **3.8492 and 3.8128 microns**. Neither mode is assumed to equal simulation X/Y. No channel is assigned, including `dm01disp02`.

Implementation decision before editing code: add a calibration projection layer that takes the actual mode samples B, explicit entrance-pupil X/Y coordinate maps in simulation axes, and nonnegative pupil weights. Construct H = 0.5e6 * [x,y] (surface microns/radian), and solve min ||sqrt(W) (B K - H)|| for the two-by-two map K. At runtime c = K theta_rad and h = B c. Reuse Eigen's column-pivoted Householder QR (already a project dependency), not an explicit inverse of the normal equations. Normalize columns internally for numerical conditioning only, then restore their scales; never renormalize the supplied calibration modes or a stochastic realization. Preserve existing `tipTiltBasis` synthesis by supplying projected angular-response planes B K.

Require both mode and coordinate fits to have rank two; report per-axis relative fit residuals and reject a fit above an explicitly supplied tolerance. A zero-weight pixel is excluded from fitting, but output follows the complete supplied mode planes (no invented outside-pupil mask). This performs no piston subtraction or optical geometry inference. Registered physical pupil coordinates, channel ownership, and combiner trigger/zero/fault contracts remain publication blockers. For ideal RMS-one orthogonal TT on a continuous circular pupil, ||c|| = D ||theta|| * 1e6 / 8 is only an ideal-basis sanity check: 2.9 mas gives about 0.04569 micron P2V and 0.01142 coefficient magnitude. These are checks, not calibration constants or gain corrections.

Added tests: arbitrary rotated/sign-swapped and nonorthogonal pairs, exact physical surface reconstruction, unit RMS/four-micron P2V synthetic normalization, mask handling, rank deficiency and residual rejection, nominal ensemble scale in modal coefficients, and unchanged RNG/ADC/time regressions. Runtime remains entirely disabled and no stream/trigger behavior is added.


### Projection execution results

The branch is `parkertjohnson1/accelSim`. Changes in this continuation are limited to `accelSimDisturbance.hpp`, its test file, README, and this plan. The deployed FITS cube was not available during the initial projection implementation; subsequent local validation is recorded below. No instrument or shared-memory connection was made, and no additive channel was chosen.

`tipTiltProjection` now computes the two-by-two coefficient map and full-image angular responses once at construction. `coefficients()` applies the map to the existing noise-free sample, and `response()` supplies B K to the unchanged preallocated `tipTiltBasis` synthesis. Construction uses scale-normalized column-pivoted QR with relative pivot threshold 1e-10 for both mode and pupil matrices, explicit caller-selected residual tolerance, finite-input checks, and overflow rejection. Stored outputs retain the original physical calibration scale. Zero-weight coordinates can be undefined; every mode pixel must be finite because the original complete mode shapes define the synthesized surface. Fit errors are reported rather than hidden by renormalizing modes or changing oscillator amplitudes.

The synthetic fixture uses a non-square 5-by-4 storage layout with explicit circular-pupil coordinates: two radial Gauss quadrature nodes and eight azimuths give exact RMS-one tilts, with four zero-weight boundary samples verifying the four-micron P2V normalization. Tests cover several arbitrary rotations, positive/negative X/Y, mixed and zero pointing, 6.5-m and 8-m pupils, swapped/reversed/rescaled/nonorthogonal modes, masks, residual rejection, rank/conditioning checks, overflow, and surface reconstruction. This fixture establishes numerical behavior only, not deployed geometry or orientation.

| Check | Result |
| --- | --- |
| Fresh combined C++20/O3 build, `-fno-fast-math -Wall -Wextra`, Eigen include path, existing `ACCELSIM_MODEL_ONLY` path | **29 cases / 165928 assertions passed** in `/tmp/accelSim_TT_combined_test '[accelSim]'`: 16 existing model/timing cases plus 13 disturbance/projection cases. |
| Fresh O1/debug AddressSanitizer + UndefinedBehaviorSanitizer build, `-fno-omit-frame-pointer` | **13 cases / 11505 assertions passed**, no sanitizer findings, in `/tmp/accelSim_TT_sanitize '[disturbance]'`. |
| Existing Python PSD regressions | **4 tests passed** using `/opt/anaconda3/bin/python -m unittest discover -s apps/accelSim/tests -p 'test_validate_psd.py' -v`. |
| Formatting and diff | clang-format 23.1.3 applied/checked on both touched C++ files; `git diff --check` passed. |
| Disabled-path preservation | Byte comparisons against HEAD confirm no changes to `accelSim.hpp`, `accelSimModel.hpp`, `config/example.conf`, or existing `accelSim_test.cpp`. The deterministic sequence regression now also invokes the new projected calibration; ADC, physical state, RNG progression, counters, recovery, and rate changes remain equal. |

Eigen is an existing project dependency (`Make/common.mk` already supplies `pkg-config --cflags eigen3`). Standalone instructions now include that flag. No application/runtime timing benchmark or live publication test is claimed. No empirical realization gain, additional reflection factor, calibration-mode rescaling, synthetic WFS, trigger logic, or DM enable control was added. The protected specification prefix is unchanged. No commits or pushes were made.

### Read-only deployed-calibration validation

The user supplied `scratch/accelSim_dm_calib/{CMmodesDM.00,dmmask,dmmap}.fits`
from the active `default` calibration
`fklcombo_z10_2400_smoothed_co0.27_pb0.5_mod3_20241113_181929_1563modes`.
Only local files were read; no instrument access, shared-memory writes, mode
renormalization, or physical-axis registration was performed. All three FITS
SHA-256 hashes were unchanged after analysis:

| File | SHA-256 |
| --- | --- |
| `CMmodesDM.00.fits` | `a5bb41b535fe46cb225c17f1086dbf9ac190cf3bd451379e9551ffb77383daf9` |
| `dmmask.fits` | `0970dab118bcdb73236ae045da93c550288a4d632b70b2748903272f2aa166f8` |
| `dmmap.fits` | `c4d7f4fffad70d122b138507093feba9421713c5e201ffbf09b01ff0f42a5cd9` |

`dmmask` is a binary 50-by-50 image with 1715 included pixels, approximately
annular with additional exclusions. Equal weights on that support reproduce the
specified unit RMS and are appropriate fitting support for this calibration.
`dmmap` is a nonnegative 50-by-50 scalar image with 2039 positive pixels and 461
zeros, spanning 0 to 7.08180e-6 in undocumented units. Its appearance suggests a
response/sensitivity map, but its provenance is not established. It must not be
used as a coordinate map or fitting weights based on current evidence; direct
weighting by dmmap gives RMS 0.951004 and 0.988255 rather than the specified unity.

Statistics below use W = dmmask / 1715, without piston removal or mode rescaling:

| Measurement | Mode 0 | Mode 1 |
| --- | --- | --- |
| Masked RMS [surface micron] | 0.9999999269 | 0.9999999916 |
| Masked mean [surface micron] | 0.0009943123 | 0.0000768712 |
| Masked sampled P2V [surface micron] | 3.849202037 | 3.812793255 |

The weighted dot product is -0.0004894652373 micron squared; its normalized
value is -0.0004894652772 (mean-subtracted correlation -0.0004895419546).
The weighted two-mode basis has rank two, singular values 1.000244662 and
0.999755197, and condition number 1.000489585. Normalized pivoted QR agrees
with an independent SVD least-squares pixel fit to 5.33e-15 in the coefficients.
This supports the current least-squares implementation for the real basis.

Pixel X/Y ramps reconstruct with relative residuals 8.46879e-8 and 8.63474e-8,
respectively. They used column/row offsets from (24.5,24.5); the modes' fitted common
zero is (24.4999999655,24.4999999874). Using the slightly asymmetric mask centroid
instead introduces piston and increases relative residuals to 8.94846e-4 and
4.39726e-4. The helper deliberately does not remove piston. There are 244 nonzero
pixels outside dmmask in each mode, consistent with preserving full supplied
mode shapes rather than imposing the fitting mask on output.

The measured P2V/RMS ratios are 3.849202318 and 3.812793287, below the nominal
continuous-pupil ratio by 3.76994% and 4.68017%. No empirical gain or normalization
correction is justified: use the supplied modes and registered physical targets.
The ideal-basis D*theta*1e6/8 check is not a deployed conversion constant.

The FITS headers contain no physical pupil scale or simulation-axis registration.
The pixel geometry and modal rotation do not establish entrance-pupil metres,
rotation/reflection/sign relative to accelSim, or a final physical theta mapping.
No measured pixel mapping is hard-coded. Runtime projection must still derive
its mapping from supplied modes, weights, and registered physical pupil x/y.
The local analysis validates the numerical basis, not optical registration or
runtime publication; unit tests remain independent of the scratch calibration.

### Exact inputs still required before a writer

1. **Calibration acceptance:** establish the acceptable surface-fit residual and validate the intended calibration identity at load time. The nominal physical coordinate convention is now established by the user: x=(column-24.5)*6.5/48 and y=(row-24.5)*6.5/48 metres. This resolves scale and registration for the chosen DM-pupil simulation axes; it does not assert external instrument registration. The local cube, native array layout, dmmask fitting support, surface-micron units, and measured normalization remain established. Runtime FITS loading remains unimplemented.
2. **Exclusive handover and deployed settings:** Jared has assigned `dm01disp10` on RTC/exao2 for this experiment. Identify and exclude existing/occasional writers including tweeterModulator and network bridges, then validate channel type/shape, combiner channel gain, and controller `dm.deltaChannels` classification. No other channel is proposed.
3. **Live combiner timing contract:** Jared specified 8-kHz publication from the same samples as `acc000`. The matching deployed `DMch2disp-01`/controller implementation and effective configuration must still establish active-write behavior, triggering stream/channel/semaphore and counter semantics, concurrent channel handling, actual output throughput, and zero application. Existing `acc000` synchronization must remain untouched; the approximately 2-kHz WFS reference does not decimate the DM path.
4. **Applied-zero/failure contract:** how only the simulator contribution is removed on disable/exit, how zero is applied if AO triggering is stopped, how faults/stream replacement/controller restart/operator zero-all inhibit or rearm injection, and which supervisor/operator procedure clears a held contribution after an uncatchable process failure. Writing zeros into local memory is not confirmation of an applied optical zero.

### Finalization validation (2026-11-01)

Updated the README, helper comments, and engineering record to distinguish
unit-RMS normalization over dmmask from the ideal continuous-pupil P2V relation,
record the measured sampled P2V, identify the validated fitting support, and
retain the requirement for registered physical coordinates. No pixel mapping,
physical scale, orientation, channel, or trigger was added as a runtime default.

Fresh builds and validation on the local macOS workstation:

| Check | Result |
| --- | --- |
| Standalone disturbance suite, C++20/O3, `-fno-fast-math -Wall -Wextra` | **13 cases / 11505 assertions passed**. |
| Combined disturbance and existing model/timing suite, same flags with `ACCELSIM_MODEL_ONLY` | **29 cases / 165928 assertions passed** (13 disturbance plus 16 existing model/timing cases). |
| Disturbance suite, O1/debug with AddressSanitizer and UndefinedBehaviorSanitizer, `-fno-omit-frame-pointer`, both sanitizers configured to halt on errors | **13 cases / 11505 assertions passed**, no sanitizer findings. |
| Existing Python PSD regressions | **4 tests passed**. |
| clang-format 23.1.3 | Applied to both changed C++ files; `--dry-run --Werror` passed. |
| Whitespace and scope | `git diff --check` passed; final diff reviewed for unrelated or accidental changes. |
| Preservation checks | App, model/timing helper, example config, existing model/timing and Python test sources, and plan specification prefix are byte-identical to HEAD. All three calibration FITS hashes and the Git index hash are unchanged. |

Reproduction commands (Eigen is the existing project dependency):

```sh
c++ -std=c++20 -O3 -fno-fast-math -Wall -Wextra -DCATCH_CONFIG_MAIN \
  $(pkg-config --cflags eigen3) apps/accelSim/tests/accelSimDisturbance_test.cpp \
  -o /tmp/accelSim_finalize_disturbance
/tmp/accelSim_finalize_disturbance '[disturbance]'
c++ -std=c++20 -O3 -fno-fast-math -Wall -Wextra -DACCELSIM_MODEL_ONLY \
  $(pkg-config --cflags eigen3) tests/testMain.cpp \
  apps/accelSim/tests/accelSim_test.cpp apps/accelSim/tests/accelSimDisturbance_test.cpp \
  -o /tmp/accelSim_finalize_combined
/tmp/accelSim_finalize_combined '[accelSim]'
c++ -std=c++20 -O1 -g -fno-fast-math -Wall -Wextra -fno-omit-frame-pointer \
  -fsanitize=address,undefined -DCATCH_CONFIG_MAIN $(pkg-config --cflags eigen3) \
  apps/accelSim/tests/accelSimDisturbance_test.cpp -o /tmp/accelSim_finalize_sanitize
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  /tmp/accelSim_finalize_sanitize '[disturbance]'
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -m unittest discover \
  -s apps/accelSim/tests -p 'test_validate_psd.py' -v
/tmp/accelsim-dm-tools/bin/clang-format --dry-run --Werror \
  apps/accelSim/accelSimDisturbance.hpp apps/accelSim/tests/accelSimDisturbance_test.cpp
git diff --check
```

Only the four intended files remain modified: this plan, `apps/accelSim/README.md`,
`apps/accelSim/accelSimDisturbance.hpp`, and its disturbance test source. Nothing
is staged or committed. No full runtime/INDI/shared-memory build or live DM test
is claimed. Publication, FITS runtime loading, acquisition handoff, enable controls,
and trigger behavior remain unimplemented; the exact writer blockers above remain.

### Direct physical DM-pupil calibration (2026-11-01)

The user rejected the tweeter -> woofer -> telescope-secondary offload path as
the disturbance amplitude calibration. AEG command units are now confirmed as
arcsec, but neither those commands nor their matrix belongs in this conversion.
The deployed ten-mode offloader has significant TT overlap with higher modes;
its static result differed from direct actuator offloading by 25.64% in relative
Frobenius norm. That analysis remains an ignored scratch artifact, not an input
to accelSim. Both 0.05 controller gains are excluded.

The user explicitly defines the current simulation axes as **+X = increasing DM
image column, +Y = increasing DM image row**. These are not telescope Az/El, sky,
or WFS axes. No notebook rotation is used. With zero-based indices and native
linear index `column + 50*row`, the expert-supplied nominal geometry is:

```text
center_column = center_row = 24.5
pupil_diameter_pitches = 48
entrance_pupil_diameter_m = 6.5
metres_per_pitch = 6.5 / 48 = 0.1354166666666667
x = (column - 24.5) * metres_per_pitch
y = (row    - 24.5) * metres_per_pitch
H = 0.5e6 * [x,y]                       # surface micron per radian
K = argmin ||sqrt(W) * (B*K - H)||
```

`scratch/accelSim_dm_calib/dmmask.fits`, with 1715 active pixels, remains the
fitting support. It reproduces the confirmed unit RMS of the supplied TT pair.
The 2039-pixel `bmc_2k_actuator_mask_F32.fits` is the offloader normalization
support and is not substituted. `dmmap.fits` is not used. The first two planes
of `scratch/accelSim_offload_calib/CMmodesDM.fits` are bit-for-bit identical to
the supplied `CMmodesDM.00.fits`; no modes are renormalized or forced to P2V=4.

The existing `tipTiltProjection` already accepts these coordinates and weights.
No coordinate-helper API or numerical implementation change is necessary.
The production header changes are documentation only. The new synthetic unit
test exercises center 24.5, 48 pitches, D=6.5 m, native column/row order, both
signs, rotated/swapped/sign-flipped bases, reflective and unit factors, and the
47/48 sampled-to-ideal P2V ratio. It does not require any scratch FITS file or
offloader component. Existing tests continue to verify unchanged RNG, ADC,
catch-up, rate changes, and phase-clock behavior.

Local analysis `scratch/direct_tt_analysis/project.cpp` calls the unchanged
production projector and float synthesizer. It uses 1e-6 as an explicit analysis
fit tolerance, not an adopted runtime acceptance policy. Independent NumPy SVD
and physical-slope calculations in `analyze.py` cross-check the result. All input
calibration files are read-only; outputs remain ignored under that directory.

```text
K [coefficient/radian] = [[-798911.7511681296, -315942.5949319362],
                         [ 318826.6904934726, -792420.6490882029]]
K [coefficient/mas]    = [[-0.003873233469655, -0.001531732924683],
                         [ 0.001545715414541, -0.003841763718717]]
K^-1 [radian/coefficient] = [[-1.079878906139e-6,  4.305538279578e-7],
                            [-4.344841570877e-7, -1.088724718299e-6]]
det(K) = 7.338051003529e11
singular values = [860183.102069, 853080.115836]
condition number = 1.008326282726
```

Mode gradient directions, measured from increasing column toward increasing row,
are -158.082845 and -68.422878 degrees. Their separation is 89.659968 degrees.
This mixing is derived from the actual planes in the chosen DM-pupil axes, not
an external registration or an imposed pure rotation.

| Input | c0 | c1 | Sampled P2V [micron] | Ideal P2V [micron] |
| --- | --- | --- | --- | --- |
| +1 mas X | -0.003873233470 | 0.001545715415 | 0.01542818566 | 0.01575644464 |
| +1 mas Y | -0.001531732925 | -0.003841763719 | 0.01542818690 | 0.01575644464 |
| +2.9 mas X | -0.011232377062 | 0.004482574702 | 0.04474173842 | 0.04569368944 |
| +2.9 mas Y | -0.004442025482 | -0.011141114784 | 0.04474174201 | 0.04569368944 |

The axis-aligned mask extrema are at indices 1 and 48, separated by 47 pitches.
Thus the sampled target P2V is 47/48 of the nominal 48-pitch optical P2V, to
calibration precision. The full supplied-mode synthesis has the same extrema
for these four cases. The 0.04569368944-micron ideal result at 2.9 mas is
consistent with the physical D*theta/2 relationship; no amplitude correction is
justified by the smaller sampled value.

| Input | RMS residual [micron] | Maximum residual [micron] | Relative residual |
| --- | --- | --- | --- |
| +1 mas X | 3.53231e-10 | 1.62500e-9 | 8.46879e-8 |
| +1 mas Y | 3.57061e-10 | 1.47503e-9 | 8.63474e-8 |
| +2.9 mas X | 1.02437e-9 | 4.71251e-9 | 8.46879e-8 |
| +2.9 mas Y | 1.03548e-9 | 4.27759e-9 | 8.63474e-8 |

These residuals compare double-precision B*K*theta with the physical target over
dmmask. After float synthesis, the +2.9 mas X/Y RMS residuals are 1.06887e-9 and
1.06550e-9 micron, with relative errors 8.83668e-8 and 8.88508e-8. The fitted
center intercepts are 3.28525e-11 and 1.19539e-11 micron. The target's masked
means (-1.08239e-5 and -5.27320e-6 micron) reflect mask asymmetry; they are not
removed. No piston correction is applied. The supplied two-mode basis is
sufficiently planar for this numerical purpose. All 244 nonzero out-of-mask
pixels per mode remain in synthesized commands.

If S holds the mode gradients in surface microns per actuator pitch, then
`theta = 2e-6 / metres_per_pitch * S * c`. Equivalently, with G in microns per
entrance-pupil metre, `theta = 2e-6 * G * c`, so `K = inverse(2e-6 * G)`.
This slope calculation agrees with the projection to 2.05e-12 relative error.
Its pixel-to-metre conversion uses precisely the same D/48 scale. Telescope D
appears because the physical angle is an OPD gradient across entrance-pupil
metres; a measured empirical angular calibration could absorb that geometry.
This work uses the explicit geometry and surface modes instead.

Validation completed on the local workstation:

| Check | Result |
| --- | --- |
| Production QR vs independent SVD | Max K difference 1.63e-9 coefficient/radian (about 2e-15 relative scale). |
| Standalone disturbance suite, O3, no fast-math | 14 cases / 12023 assertions passed. |
| Combined disturbance and existing model/timing suite, `ACCELSIM_MODEL_ONLY` | 30 cases / 166446 assertions passed. |
| AddressSanitizer + UndefinedBehaviorSanitizer disturbance suite | 14 cases / 12023 assertions passed, no findings. |
| Existing Python PSD regressions | 4 tests passed. |
| Formatting and whitespace | clang-format applied to touched C++ files; dry-run and git diff --check passed. |
| Preservation | App/model/config/existing model and PSD test files, all nine calibration FITS files, Git index, and plan specification prefix unchanged. Production helper code tokens unchanged; comments only. |

Build flags and commands follow the preceding finalization section, with binary
names `/tmp/accelSim_direct_disturbance`, `/tmp/accelSim_direct_combined`, and
`/tmp/accelSim_direct_sanitize`. No full runtime/INDI build or live hardware test
is claimed. The remaining publication blockers are calibration acceptance,
exclusive channel ownership, combiner gain and delta-channel
classification, cadence/trigger/phase semantics, and applied-zero/failure policy.
External Az/El registration and AEG calibration are not blockers for the chosen
DM-pupil simulation convention.

## Channel 10 and same-sample 8-kHz publication (2026-11-02)

### Authoritative decisions and scope

The user relayed Jared Males's operational guidance:

- "for vibration simulation use 10 for now, that's what we use for turbulence simulation"
- "So the idea is to publish commands to that dm at 8kHz"

These settle the experiment's additive channel (`dm01disp10`) and requested
publication cadence (8 kHz). They supersede the earlier approximately 2-kHz,
WFS-paced passive design and the requirement for a separate DM worker. They do
not establish the installed executable behind the tweeterModulator instance,
its current write/trigger mode, or per-frame combiner/driver delivery.

Only this plan is updated. The physical mapping and tests are already complete,
validated, committed, and pushed according to the user. No runtime code,
configuration, shared memory, process state, or hardware is changed here. The
runtime implementation gate remains closed until publication semantics are
understood from matching local deployment copies.

### Publication evidence and unresolved deployed behavior

Read-only inspection of this checkout, scratch, and sibling mxlib/handbook trees
still finds no tweeterModulator configuration/process-list entry, cacao
DMch2disp implementation, effective FPS settings, or ImageStreamIO C source.
The bridge's channel-10 entry establishes 50-by-50 layout, not writer identity.
`apps/ttmModulator` controls a different tip/tilt modulator; its name does not
establish a tweeter additive-channel writer. The following are exact local code
facts, with deployment applicability explicitly conditional:

| Source/function | Established behavior |
| --- | --- |
| `apps/dmSpeckle/dmSpeckle.hpp:558`, `appLogic` | Opens an existing configured `dm.channelName`; checks float datatype at line 597. No channel creation at this point. |
| Same file, `generateSpeckles`, line 637 | `arbcube` mode loads a FITS cube, validates dimensions, multiplies its planes by configured amplitude, and cycles them in its worker. This is a plausible turbulence-player implementation, not proof it is the deployed tweeterModulator executable. |
| Same file, `modThreadExec`, lines 864-877 | Sets `write=1`, copies a complete plane, sets `atime`/`writetime`, increments `cnt0` when untriggered OR trigger delay is positive, clears `write`, and posts all image semaphores with index -1. A triggered zero-delay write leaves `cnt0` unchanged. This 2-D write does not update `cnt1`. |
| Same file, lines 900-913 | Ordinary stop/loop exit writes a zero image, updates timestamps, increments `cnt0` only when untriggered, clears `write`, and posts all semaphores. The restart branch at line 895 skips this zero. |
| Same file, `m_indiP_zero` callback, line 1250 | Refuses zero while modulating; an explicit accepted zero increments `cnt0` and posts all semaphores. A disabled modulation toggle alone is not an ownership transfer or exclusive lock. |
| Same file, `appShutdown`, line 619 | Joins the modulation thread. This is normal lifecycle behavior, not a crash cleanup guarantee. No channel-ownership lease/release is demonstrated by these paths. |
| `apps/dmMode/dmMode.hpp:351`, `sendCommand` | Native active-write precedent: write flag, full copy, `cnt0++`, clear flag, post all semaphores. Its wait on an existing write flag must not be copied into accelSim's acquisition hot path. |
| Sibling `mxlib/include/improc/milkImage.hpp:790`, `post` | Active mode delegates to unavailable `ImageStreamIO_UpdateIm`; passive mode timestamps/clears write/posts all semaphores without incrementing `cnt0`. The documented passive contract intentionally avoids triggering dmcomb. Active mode's detailed blocking/error behavior still needs the deployed C implementation. |
| `scripts/cacao/RTC/cacao-startup:19` | Starts `DMch2disp-01`, but supplies no evidence of its effective trigger settings. |
| `libMagAOX/app/dev/shmimMonitor.hpp:743` | Driver-side monitor waits on its selected combined-stream semaphore, then processes the current image without a `cnt0` comparison in this loop. It does not establish combiner behavior on an additive channel. |
| `libMagAOX/app/dev/dm.hpp:1897`, `apps/bmcCtrl/bmcCtrl.hpp:509` | The combined-stream handler dispatches `commandDM`, which submits the converted command through `BMCSetArray`. This proves the software path, not achieved frequency or hardware settling. |

There are no grounds yet to report exact answers for how deployed DMcomb reacts
to channel 10, whether each input write produces a unique combined frame, or
whether each reaches the physical driver. A single image buffer is not a queue
of immutable commands. Inspect semaphore accumulation/coalescing, counter tests,
channel rereading/caching, write-flag handling, output posting, and rate limits.
Concurrent AO writes could add output events or be coalesced; total driver rate
is not necessarily exactly 8 kHz even with an 8-kHz channel-10 producer. Multiple
semaphore wakes can also process the same latest buffer if earlier frames were
overwritten. Do not claim per-frame delivery from mean counters alone.

### Preferred minimal architecture

Recommend option C: accelSim owns channel 10 for this experiment and reuses the
verified native publication pattern. This means direct publication by accelSim,
not sharing the channel with tweeterModulator. No existing local API is shown
that accepts live, individually identified accelSim samples into the cube player.
Routing through an independent player would add another clock, buffering, and
ownership boundary without established benefit. Revisit only if deployed source
reveals a suitable interface.

Prefer no new thread. Use `accelSim::frameGrabberPostPublish` on the existing
acquisition thread, after `acc000` metadata and semaphore publication and before
any next `m_model.advance`. Sources: `frameGrabber.hpp:1014-1020`,
`accelSim.hpp:753`, `accelSim.hpp:791-821`. This creates a deterministic software
order, ADC then DM, from the same state. Two distinct streams cannot be published
atomically; retain source sequence/timestamps and measure the publication skew.

At startup, outside the hot path:

1. Load the approved TT pair and dmmask; check provenance, dimensions, finite
   samples, rank, and accepted projection residual. Do not substitute the
   offloader mask. Reject stale/invalid calibration; any reload requires inhibit,
   zero, and full validation before rearming.
2. Compute K and full angular-response planes B*K using the validated geometry
   and `tipTiltProjection`. Allocate/prefault the complete float output and any
   metadata/control storage. Do not hardcode arrays or solve at sample rate.
3. After operator handover, open existing `dm01disp10`, validate exactly 2-D
   50-by-50 float32 and the matching publication contract, and explicitly zero it.
   Never create/replace the channel or use the combined `dm01disp` as output.

For every successful ADC publication while enabled/healthy:

1. Use that exact `m_model.m_sample`, retaining its model tick, ADC sequence,
   acquisition time, and calibration/control generation. Never take a second
   model step or use catch-up-only intermediate states as extra DM frames.
2. Construct the existing `pointingDisturbance` from `m_pointing`. The stored
   angles are arcsec; the helper converts to radians once. Do not use legacy
   `m_wfsPointing`, integrate ADC counts, or draw any additional random values.
3. Use precomputed responses to fill the reusable image, mathematically
   `c=K*theta_rad` and `h=B*c` in surface microns. Geometry remains +X increasing
   column and +Y increasing row, center 24.5, nominal 48 pitches, D=6.5 m. No
   Az/El/AEG/K-mirror/offload transform, controller gain, or P2V rescaling.
4. Apply the requested experiment amplitude scale [0,1], preserving TT
   direction/planarity across the full image, including outside dmmask.
   Nonfinite/overflow or invalid state requests fault/zero; do not rely on allocating
   exceptions as the planned hot-path error mechanism.
5. Publish the complete 10,000-byte image through the verified active native
   contract that drives combination at the required cadence. Expected metadata
   pattern is write flag/copy/timestamps/`cnt0++`/clear flag/semaphore posts, but
   actual order/helper selection awaits the deployed dependency. Leave `cnt1`
   under its documented 2-D convention, not an invented ADC-sequence field.
   Record the ADC source identity separately. Use acquisition and write times
   distinctly if the supported metadata API allows it.
6. Record compact counters/timing only; defer INDI/logging and filesystem checks
   to ordinary app supervision. DM-only errors must inhibit/zero/report without
   returning a fatal post-publish error that stops frameGrabber/acc000.

The hook runs after the current ADC frame but still spends the next frame's
125-microsecond budget. The source of `ImageStreamIO_UpdateIm`/semaphore posting
must be inspected for waits, locks, allocation, retries, and failure behavior.
The current `milkImage` wrapper has throwing error paths, so it must not be
described as an allocation-free/nonblocking contract without further inspection.
Use native utilities where their verified contract fits; do not blindly copy
the modulation app's wait/sleep or timestamp code. Normal synthesis is already
preallocated in `tipTiltBasis::synthesize`.

If direct publication cannot meet the existing ADC timing requirements under
load, use a preallocated bounded SPSC handoff of immutable same-sample state to
one DM worker. No producer waits, allocation, second model, independent DM clock,
or 2-kHz decimation. The healthy worker must sustain all 8-kHz samples. An overflow,
stale backlog, or missed delivery invalidates the one-to-one experiment: inhibit
and zero, count the fault, and continue ADC acquisition. Do not silently overwrite
queued samples or burst stale commands to catch up. A worker mitigates blocking
in the producer; it does not establish successful DM delivery by itself.

### Enable, zero, shutdown, and ownership sequence

Defaults: injection disabled, scale 0. Keep `dm.channelName` empty in generic
configuration; the reviewed experiment explicitly sets `dm01disp10`. Require the
active ADC rate to be 8 kHz before arming. An existing ADC-rate change remains
effective and inhibits DM injection rather than silently overriding ADC control.

1. **Handover/startup:** the operator disables/stops every channel-10 producer,
   including automatic reactivation and bridge writes, waits for the old writer's
   final zero to complete, and assigns exclusive ownership to accelSim. A late
   tweeterModulator stop-zero could overwrite the first accelSim command. Only
   after handover may accelSim open/validate and actively publish startup zero.
   Do this even with injection disabled, provided the channel is configured and
   owned. An unset or unowned configuration must perform no DM access.
2. **Enable:** validate readiness, fresh same-sample state, 8-kHz active rate,
   calibration/control generations, stream identity, and publication contract.
   Accept enable at a sample boundary. Scale remains zero until explicitly raised.
3. **Disable/fault:** request a control-generation change. The sole publisher
   finishes any in-flight write, cancels queued nonzero work, writes zero using
   the verified active mechanism, then acknowledges disabled. No later nonzero
   write from the older generation is allowed. INDI callbacks never race-write
   the image. Continue zero contribution while disabled without needless 8-kHz
   zero traffic. Monitor state outside the sample path; invalidate/rearm on
   stream replacement, lost calibration validity, restart, or stale/invalid sync
   according to the accepted health policy. ADC holdover behavior stays intact.
4. **Normal shutdown:** inhibit, quiesce/join acquisition (and the DM worker if
   present), then publish the final zero while the validated channel remains
   mapped. A designated shutdown owner may take over only after the hot-path
   writer is quiescent. Bound/report failures; close the mapping afterward.
   Do not use `zeroAll` or clear another app's channel. Existing accelSim shutdown
   already joins frameGrabber before closing synchronization (`accelSim.hpp:425`).
5. **Return ownership:** only after the final zero and no further accelSim writes
   may an operator re-enable tweeterModulator. Disabling injection alone does not
   grant permission for two owners to write, including their zero callbacks.

Zero publication must cause recombination even without the usual AO stream;
that remains a deployment verification gate. A channel-buffer zero is not proof
of optical zero. Normal cleanup cannot cover SIGKILL/crash/host failure. The
last complete image, a partial image, or `write=1` may remain. No channel-specific
expiry watchdog was found locally; an external fenced cleanup/recovery policy
is still required, and an automatic process restart is not a bounded crash-zero
guarantee.

### Required local deployment evidence and validation gate

Obtain local copies of `/opt/MagAOX/config/proclist_RTC.txt`, the resolved
tweeterModulator config (normally `/opt/MagAOX/config/tweeterModulator.conf`,
confirm via the process entry), dmtweeter config, and the effective DMch2disp-01
FPS settings/startup configuration. Also obtain the deployed modulator/combiner
source and commit/build identifiers plus matching ImageStreamIO and mxlib source.
The process list must establish instance-to-executable identity; do not assume
that a configuration filename proves which binary is running. User-operated
read-only commands from the preceding investigation can locate these files.
No agent connects to instrument hosts or executes their control commands.

Before implementation, trace precisely: input channel gains/enables, trigger
channel/mode/semaphore, `cnt0` tests, write-flag consistency checks, output image
counter/timestamps/semaphores, queue/coalescing policy, rate caps, and clearing
behavior. The operational assignment and 8-kHz intent are already authoritative;
these checks establish the mechanism and measurable delivery contract.

After semantics are established, implement only the app-local integration,
minimal enable/scale/status controls, and associated isolated tests. Validate:

- Fixed-seed sample-by-sample ADC/model/RNG equality with injection off/on and
  one DM source identity per successful ADC frame; no extra model advances.
- Existing physical projection tests unchanged; nonfinite, invalid/stale
  calibration, shape/type errors, amplitude scaling, and allocation-free error handling.
- Private ImageStreamIO fixture: unique names, no bridge/hardware driver;
  correct flag/counter/timestamp/semaphore order, verified combiner behavior,
  concurrent synthetic AO updates, no torn reads, and final-zero application.
- Enable/disable/shutdown races, generation cancellation, stream replacement,
  source loss/holdover, ADC rate changes, no stale replay, and external crash
  cleanup tested independently of normal cleanup.
- Sustained 8-kHz ADC and DM with a synthetic 2-kHz phase reference on an isolated
  suitable Linux development host. Measure actual source identities, distinct
  combined/dummy-driver frames, timing/skew distributions, missed deadlines,
  CPU/RSS, hot-path allocations, and syscall/blocking behavior. A mean rate alone
  does not prove one-to-one delivery or preserve ADC jitter. Set explicit timing
  acceptance criteria from the existing baseline before accepting the direct path.
- Existing model/synchronization/full-app tests, sanitizer coverage, Python PSD
  regressions, clang-format on eventual C++ changes, and git diff --check.

Current validation is plan-only: diff/whitespace review, unchanged original task
specification prefix, and unchanged production/test/configuration files. No
runtime or physical performance result is claimed, no process was restarted,
and nothing is staged, committed, or pushed.

## Runtime implementation following deployed-system authorization (2026-11-02)

### Resolved publication contract

The user inspected the deployed `AOloopControl_DM_comb.c`: each compute iteration
sums additive-channel `cnt0` values, and a changed sum causes recomputation from
the current contents of all twelve channels and publication of `dm01disp`.
Authoritative deployed settings are AveMode=0, DClevel=0, process trigger mode 4,
triggerdelay=0.00001 seconds. `tweeterModulator` is confirmed configured for channel
10. Together with the explicit instruction to implement, this resolves the prior
publication gate. These facts are user-supplied deployment evidence; the combiner
source is not independently available in this checkout.

The app now uses **active** publication, never passive. Existing `dmMode`,
`dmSpeckle`, `t2wOffloader`, and `dmPokeWFS` patterns were rechecked. The output
adapter calls `ImageStreamIO_UpdateIm`, the primitive called by mxlib's active
`milkImage::post`, after flagging/copying the complete image. It does not invent
counter increments or semaphore logic. Raw existing-only attachment is used to
validate `naxis`, exact dimensions, datatype, and metadata without allocating or
throwing in the normal sample path. There is no image creation or consumer wait.

### Implemented data flow and lifecycle

- `accelSimDM.hpp`: independent, testable owner-thread state using the unchanged
  `tipTiltProjection`, `pointingDisturbance`, and `tipTiltBasis`. At preparation,
  construct the accepted center-24.5/D-6.5/48-pitch maps, fit B*K, and allocate
  command storage. Runtime performs only synthesis, finite/overflow preflight,
  requested amplitude scaling, and publisher invocation across the full image.
- `accelSimDMStream.hpp`: attach only to an existing 2-D 50x50 float32 stream;
  reject a busy write flag without waiting. Publish using the native active
  primitive and retain ADC acquisition time separately from native write time.
  Supervision checks backing-file inode outside acquisition and only invalidates;
  it never automatically reconnects, replaces, or zeroes a newly recreated stream.
- `accelSim.hpp`: cold FITS loading/validation, configured dmmask support, and
  startup zero before frameGrabber starts. `frameGrabberPostPublish` consumes
  exactly the already published ADC frame's `m_model.m_sample`; no second model
  advance, RNG draw, integration, or legacy WFS attenuation. No worker or queue.
  The model, acquisition deadlines, phase-clock algorithms, ADC conversion, and
  shared frameGrabber code remain unchanged.
- Controls use atomic, monotonically revised enable requests. All runtime image
  writes and zero acknowledgements happen on the acquisition thread. An old
  revision cannot rearm a completed disable. Scale changes do not change physical
  model amplitudes. Faults latch until an explicit new request. Missing/invalid
  calibration, changed files, invalid stream, nonfinite samples, source loss,
  reconfiguration, and rate changes inhibit injection while ADC behavior continues.
  Source loss uses the existing synchronization validity/timeout, not a new clock.
- Clean shutdown joins acquisition before final zero/close, serialized against
  callbacks. No global zero function, other-channel writes, process control, or
  automatic tweeterModulator interaction is added. Failure to clear is reported;
  process death cannot guarantee zero. Stream/calibration metadata supervision
  is not instantaneous or a cryptographic calibration-provenance guarantee.
- Configuration: `dm.channelName` empty by default, intended experiment value
  `dm01disp10`; explicit `dm.modeCube` and `dm.mask`; `dm.enabled=false` (true is
  rejected to prohibit startup enable); `dm.scale=0`;
  `dm.maxRelativeResidual=1e-6`. Unset channel performs no access.
  Configuring a channel transfers startup zeroing responsibility, so exclusive
  handover must precede launching that configuration, not just nonzero enable.
- INDI: `dm_enable.toggle`, `dm_scale.current/target`, numeric read-only
  `dm_status.ready/active/fault`. Busy transitions acknowledge at sample boundaries;
  no 8-kHz INDI traffic. Fault codes: 2 calibration, 3 stream, 4 numerical,
  5 rate/synchronization. Active ADC rate must be 8000 Hz to arm; an ADC rate
  change remains effective and inhibits the DM path rather than overriding it.

There is no actual per-frame delivery guarantee from the combiner counter test:
several writes can coalesce before one compute iteration. Concurrent AO updates
can also add output events. Preserve sample identity and measure distinct
combined/driver delivery and timing during commissioning; neither producer rate
nor a count of semaphore receipts proves unique hardware commands.

### Validation performed locally before the instrument-owner update below

| Check | Result |
| --- | --- |
| New standalone runtime suite | 4 cases / 60,148 assertions passed. Includes disabled/scale-zero, same-sample model/ADC/noise equality for 8,000 samples, nonfinite faults, zero/disable/rearm, failed-publisher retry, and stale control rejection. |
| Combined existing model/synchronization, physical projection, and runtime suite, O3/no fast-math | 34 cases / 226,594 assertions passed. |
| Same combined suite, AddressSanitizer + UndefinedBehaviorSanitizer, O1/debug, halt-on-error | 34 cases / 226,594 assertions passed; no findings. |
| Existing Python PSD regressions | 4 tests passed. |
| Real local deployed TT/dmmask comparison | Runtime pixels equal the unchanged projector/synthesizer bit-for-bit at 1 and 2.9 mas for both axes; maximum difference zero. Full-image 2.9-mas P2V approximately 0.04474174 micron, preserving the accepted sampled-vs-ideal distinction. |
| Allocation-instrumented real-calibration numerical/mock path | Zero C++ heap allocations over 80,000 healthy frames plus numerical-fault/disable zero paths. Startup allocation is outside measurement. This does not instrument native ImageStreamIO. |
| Numerical synthesis plus preallocated mock publication timing | About 3.6 us/frame mean; final O3 suite observed 20.7 us maximum on this run. Instrumented 80,000-frame mean 3.59 us. OS-dependent measurements, not native stream, full-app scheduling, or hardware bounds. |
| Full app and new full-runtime test build attempts | Both blocked in existing local macOS dependencies: `timegm` undeclared and `mx/math/floatUtils.hpp` missing. No full-app or native-stream build/pass claimed. |
| Formatting, whitespace, and preservation | clang-format and dry-run passed on all touched C++ files, with comment-prefix minimum 0 to retain AGENTS.md's exact `//mutex scope` spelling. git diff --check passed. Original task prefix, numerical/model sources, original tests, and shared acquisition/synchronization sources remain unchanged. |

New full-runtime tests are provided for a configured non-instrument Linux host:
`[dm-controls]` tests INDI defaults/scale/rejected enable; hidden
`[accelSim-dm-streams]` tests a synthetic existing channel, native counter/semaphore
publication, timestamps, incorrect dimensions/datatype, busy flag, missing-image
rejection, and the app's exact ADC-to-DM hook plus disable zero. The latter first
checks that ImageStreamIO's resolved path belongs to a fresh private `/tmp`
namespace; run it in its own process because namespace selection can be cached.
These tests have not run on this macOS host. They never name a physical channel.
`tests/Makefile.one` disables fast-math for the new suite and tracks header changes.

The ignored `scratch/direct_tt_analysis/runtime_check.cpp` uses read-only local
FITS data and a mock sink; it does not run the app or open any stream. Fresh
standalone builds use the commands from the README, adding all three test files
and `tests/testMain.cpp` for combined coverage. No instrument host was accessed,
no deployed process/configuration was changed, and no physical stream was written.

### Remaining development and commissioning checks

1. On a suitable isolated Linux development host, build the app and full tests,
   run the private native tests, and perform sustained 8-kHz ADC/DM testing with
   a synthetic approximately 2-kHz phase reference. Measure native-update duration,
   jitter/skips/skew, allocation/syscall behavior, and distinct combiner/dummy-driver
   delivery under concurrent AO load. Numerical timing suggests no need for a worker,
   but native timing is not established on this workstation. Keep injection disabled
   until this acceptance is complete; if native latency requires a worker, implement
   a separately reviewed bounded handoff with explicit overflow/fault handling.
2. Operator handover must exclude tweeterModulator and bridge/automatic writers,
   including their late zero callbacks. No software ownership lease is claimed.
   Confirm the deployed channel gain and delta-channel accounting for the experiment.
3. Review runtime calibration paths/identity and 1e-6 residual tolerance.
   Commission at low scale with positive and negative axes, verifying WFS
   response, combined command margin, actual timing,
   stop/zero application during AO inactivity, and source-loss behavior.
4. Establish external recovery for SIGKILL/crash/host/combiner/driver failure.
   Neither close/destructor, normal final zero, nor automatic restart provides
   a bounded crash-zero guarantee. No automatic supervisor is added in this scope.

Final whitespace/formatting and preservation checks are recorded in the user
report. All prior work is retained, including the original task specification
prefix. Nothing is staged, committed, or pushed.

## Instrument-owner command-protection guidance (2026-11-02)

The user relayed Jared Males's confirmation that the existing MagAO-X/DM system
already provides sufficient command protection for this vibration injection.
Removed the provisional P2V/absolute-pixel bounds, their configuration keys and
validation, the common reduction calculation/accessor, and their enforcement
tests/documentation. There were no limit-specific INDI/status fields to remove;
the readiness/activity/fault fields remain. `dm.scale` retains its existing [0,1]
range and zero default solely to adjust experiment and commissioning amplitude.
Each output pixel is now the unchanged synthesized float command multiplied by
the requested scale, with no additional actuator or P2V limiting.

Software-integrity protections are preserved: disabled startup, explicit
operator enable, nonfinite/overflow detection, latched fault inhibition,
controlled-disable zero, clean-shutdown zero after acquisition joins, stream
validation, and exclusive channel ownership. The physical projector/basis,
calibration acceptance, sample timing, synchronization, native active publisher,
and existing same-sample model/RNG behavior are unchanged. The overflow preflight
remains a numerical representability check, not an experiment amplitude bound.

The mapping regression now compares the full calibrated surface at three scales
for axis-aligned, mixed-sign, and large finite pointing samples, including commands
above both former bounds. Lifecycle, failed-publication retry, nonfinite faults,
zero/rearm, stale-control rejection, and 8,000-sample model/ADC/noise equality tests
remain. The full native fixture's preparation calls use the revised helper API.

Validation rerun on the local workstation:

| Check | Result |
| --- | --- |
| Combined model/synchronization, physical projection, and DM runtime tests; C++20, O3, no fast-math | 34 cases / 236,598 assertions passed. |
| Same combined suite; O1/debug, AddressSanitizer + UndefinedBehaviorSanitizer, halt-on-error | 34 cases / 236,598 assertions passed; no findings. |
| Existing Python PSD regressions | 4 tests passed. |
| App and full DM runtime test build attempts | Both still blocked by existing macOS dependencies: undeclared `timegm` and missing `mx/math/floatUtils.hpp`. Native-stream/INDI tests remain unexecuted here. |
| Formatting and whitespace | clang-format run and checked on the three touched C++ files, retaining `//mutex scope`; `git diff --check` passed. |

The combined builds use all three accelSim test source files plus
`tests/testMain.cpp`, `-DACCELSIM_MODEL_ONLY`, and the existing Eigen include path;
both binaries run the `[accelSim]` filter. Python tests use
`python -m unittest discover -s apps/accelSim/tests -p test_validate_psd.py`.
Full native/load commissioning checks above remain outstanding on an isolated
non-instrument Linux development host. No instrument access, physical stream
writes, staging, commits, or pushes occurred.
