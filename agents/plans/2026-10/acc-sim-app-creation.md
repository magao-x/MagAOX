# Task Description
<!-- The first section is filled out by the user 
     
     This is intended as a guide for how to prompt a coding agent to solve a problem.
     
     Make a copy of this template with a new name under agents/plans/<YYYY-MM>. Then
     Fill out each subsection below as needed. Feel free to add additional
     subsections, etc.  

     When complete, commit it on your feature branch and then prompt the agent to review this file.
-->

## Problem Statement
Develop a new MagAO-X application, accelSim, that generates realistic simulated accelerometer data for testing accelerometer-based DDSPC control without requiring the physical accelerometer system.

The simulator should generate continuous X/Y structural vibration signals, convert the simulated physical acceleration into raw ADC counts, and publish those counts through an ImageStreamIO stream named accsim000. The stream should mimic the interface of the real acc000 accelerometer stream so that DDSPC can consume simulated and real accelerometer data through the same basic interface.

The initial implementation is limited to generating and verifying accsim000. A later stage will use the same underlying simulated pointing disturbances to inject correlated tip/tilt disturbances onto a MagAO-X DM so that the real WFS measures a disturbance correlated with accsim000.

## Discussion
A Python prototype of the vibration simulator currently exists at:

scratch/accel_ts_sim_ddspc.ipynb

This notebook should be used as the reference for the physical vibration model. The simulator represents structural vibration modes using AR(2) oscillators. For each mode and each X/Y axis,
[
x[n] = \phi_1 x[n-1] + \phi_2 x[n-2] + w[n],
]
with
[
\phi_1 = 2r\cos(2\pi f/f_s), \qquad \phi_2 = -r^2.
]
The Python prototype generates independent X and Y realizations for each vibration mode. The C++ implementation should preserve this behavior.

Unlike the finite Python simulation, accelSim must operate continuously in real time. Therefore, the streaming implementation should use analytically scaled Gaussian innovations and initialize each AR(2) oscillator from its stationary joint distribution. It should not depend on finite-record RMS normalization, precomputing a complete simulation, or future samples.

The existing Python model specifies mode strength using acceleration RMS. Pointing RMS is derived using the configured effective lever arm and mode frequency. The reference lever arm is 10 m.

For each vibration mode, pointing is converted to acceleration using
[
a = -(2\pi f)^2 L\theta_{\rm rad},
]
where
[
\theta_{\rm arcsec} \frac{\pi}{180\times3600}.
]
Mode accelerations are summed, then simulated accelerometer measurement noise is added.

The default one-sided acceleration noise PSD is
[
10^{-8};({\rm m/s^2})^2/{\rm Hz}.
]

The real mcp3208Ctrl publishes raw 12-bit ADC values stored as UINT16. DDSPC is intended to consume these raw ADC values directly. Therefore, accelSim should convert its simulated acceleration back into raw ADC counts before publication.

The existing offline acceleration calibration is
[
a = \frac{(ADC-ADC_0)V_{\rm per,bit}}{S},
]
so the simulator should use its inverse:
[
ADC = ADC_0 + a\frac{S}{V_{\rm per,bit}}.
]

Initial calibration values are:

ADC_zero = 2048
sensitivity = 1.02
V_PER_BIT = 0.0083799

These values must be configurable rather than permanently hard-coded. The calculated ADC value should be rounded, clamped to the valid 12-bit range [0,4095], and stored as uint16_t.

Relevant existing MagAO-X code should be used as implementation examples, especially:

apps/mcp3208Ctrl for the existing accelerometer stream interface and acquisition behavior.

libMagAOX/app/dev/frameGrabber.hpp for ImageStreamIO stream creation, publication, timestamps, counters, circular-buffer behavior, and semaphores.

apps/cameraSim as an example of simulated acquisition using the frameGrabber framework.

A later phase will use the same simulated physical pointing state to calculate a WFS-coupled disturbance. The Python prototype contains separate X/Y WFS coupling coefficients and an X/Y-to-tip/tilt coordinate transformation. These quantities should remain conceptually separate from accelerometer sensor noise.

DM injection is explicitly outside the initial implementation.

## Requirements
1. Create a new MagAO-X application named accelSim following the repository conventions and AGENTS.md.
2. Follow the standard MagAO-X feature-branch, testing, and pull-request workflow.
3. Use dev::frameGrabber<accelSim> and existing MagAO-X infrastructure wherever practical rather than implementing duplicate ImageStreamIO publication machinery.
4. Generate vibration signals continuously and sample-by-sample. The production simulator must not require precomputing a finite record.
5. Default simulation/publication rate should be 6000 Hz and must be configurable.
6. Implement independent X and Y AR(2) oscillators for every vibration mode.
7. Configure AR(2) damping using r directly, matching the Python prototype.
8. Scale innovations analytically so each oscillator has the requested stationary variance.
9. Initialize the two AR(2) history samples from the correct stationary joint Gaussian distribution rather than initializing them to zero.
10. Use double precision for internal simulation calculations.
11. Preserve the active Python reference vibration model as closely as practical, including the 14 prominent modes and the additional high-frequency vibration population.
12. Use a default effective lever arm of 10 m.
13. Maintain internally, at minimum: physical pointing X/Y in arcsec, WFS-coupled pointing X/Y in arcsec, acceleration X/Y in m/s^2, raw ADC X/Y in counts.
14. Convert each vibration mode from pointing to acceleration separately before summing the acceleration contributions.
15. Add independent X/Y accelerometer measurement noise after summing the vibration-mode accelerations.
16. Use a default acceleration noise PSD of 1e-8 (m/s^2)^2/Hz.
17. Convert final acceleration to raw ADC counts using configurable ADC_zero, sensitivity, and V_PER_BIT.
18. Round ADC values before conversion to integer representation and clamp values to [0,4095] before casting to uint16_t.
19. Track ADC clipping events.
20. Publish an ImageStreamIO stream named accsim000 by default.
21. The initial accsim000 stream contract should match the real MCP3208 accelerometer stream: width = 2, height = 1, datatype = _DATATYPE_UINT16, element 0 = X raw ADC count, element 1 = Y raw ADC count.
22. Each published sample must have a meaningful acquisition timestamp.
23. Use frameGrabber infrastructure for stream creation, timestamps, counters, circular-buffer handling, and semaphore publication where supported by the existing framework.
24. Use deterministic random seeding where practical for reproducible testing. Exact sample-by-sample equality with NumPy random-number generation is not required.
25. Do not implement transient/ring-down disturbances in the initial version.
26. Do not open, modify, or publish to a DM stream in the initial version.
27. Do not add instrument startup configuration in the initial version.
28. Avoid unrelated changes to the repository.

## Tests and Metrics
The implementation should include focused tests for the numerical model and ADC conversion.

Tests should verify:
1. AR(2) coefficient calculation for known f, r, and sample rate.
2. Analytical innovation variance/scaling for a requested stationary RMS.
3. Stationary initialization produces the expected variance and lag-one covariance.
4. X and Y use independent oscillator realizations.
5. Pointing-to-acceleration conversion agrees with the analytical relation
[
a = -(2\pi f)^2L\theta_{\rm rad}.
]
6. Zero acceleration maps to the configured ADC midpoint, initially 2048.
7. Known positive and negative accelerations produce the expected ADC counts.
8. ADC rounding behavior is correct.
9. Values below the ADC range clamp to 0.
10. Values above the ADC range clamp to 4095.
11. Output values are stored as uint16_t.
12. Fixed seeds provide deterministic behavior within the C++ implementation where appropriate.
13. accsim000 has the expected dimensions and datatype.
14. X and Y appear in the intended stream elements.
15. Stream counters, timestamps, and semaphore publication behave consistently with the existing frameGrabber infrastructure.
16. Sustained operation should be tested at the target 6 kHz rate and achieved rate / missed deadlines should be measured.
17. PSD and RMS validation should be performed offline using Python rather than adding PSD calculations to the real-time C++ loop.

For very high-r modes, individual short records are not expected to reproduce the requested RMS exactly. Validation of stationary variance should account for the long correlation time and use appropriate long-duration or ensemble tests.

## Scope and Caveats
The first implementation is intentionally limited to the simulated accelerometer stream.

It does not include: DM injection, physical WFS disturbance injection, closed-loop DDSPC testing, transient/ring-down disturbances, detailed simulation of the individual analog electronics, exact reproduction of NumPy random sequences, instrument startup/deployment configuration.

The Python notebook uses finite-record normalization after generating a complete realization. The continuous C++ implementation will intentionally differ by using stationary AR(2) initialization and analytical innovation scaling. This difference should be documented.

The ADC calibration constants are initial values based on the existing offline analysis and should remain configurable. The initial implementation does not attempt to reconstruct the complete voltage-divider, clamping, signal-conditioner, or ADC analog circuit.

WFS coupling should remain separate from accelerometer sensor noise. WFS-coupled pointing may be maintained internally for future DM work, but no DM output should occur during this task.

Future work will add:
1. WFS-coupled pointing generation from the same underlying physical disturbances.
2. Configurable X/Y-to-tip/tilt coordinate transformation.
3. Calibrated conversion from angular tip/tilt to DM commands.
4. Publication to a dedicated DM disturbance channel.
5. Measurement of the resulting disturbance with the real WFS.
6. End-to-end testing of accelerometer + WFS DDSPC.

# Instructions to Agent
Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan
<!-- This section will be filled out by the agent -->

## Task Summary
<!-- The agent should summarize the task as they understand it -->

## Key Assumptions
<!-- The agent should list any assumptions they have made -->

## Requirements
<!-- The agent should list the requirements to which they are planning -->

## Questions and Points of Clarification
<!-- The agent should list any open issues requiring user clarification -->

## Tests
<!-- The agent should list and describe the test it plans to implement.  It should be specific about the purpose and goal of the test. -->

## Implementation Plan
<!-- Here the agent documents its plan for the implementation -->

## Follow-up and Edge Cases
<!-- The agent should list any planned follow up and any edge cases that are not addressed -->
