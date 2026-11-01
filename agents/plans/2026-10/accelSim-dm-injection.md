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

<!-- This section will be filled out by the agent -->

## Task Summary

<!-- Summarize the task as understood after reviewing the references. -->

## Key Assumptions

<!-- List assumptions made while developing the plan. -->

## Requirements

<!-- List the requirements to which the implementation will be designed. -->

## Questions and Points of Clarification

<!-- List unresolved questions that require clarification before implementation. -->

## Tests

<!-- Describe the specific tests planned, their purpose, and expected result. -->

## Implementation Plan

<!-- Document the proposed implementation, including relevant files/classes/streams and the planned sequence of changes. -->

## Follow-up and Edge Cases

<!-- Document follow-up work, limitations, and edge cases not addressed by the initial implementation. -->