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

The simulator should generate continuous X/Y structural vibration signals, convert the simulated physical acceleration into raw ADC counts, and publish those counts through the same ImageStreamIO interface used by the real accelerometer system.

The simulator is intended to act as a drop-in replacement for the existing accelerometer acquisition system during simulation and DDSPC testing. By default, it should publish the stream `acc000`, not a simulation-specific stream name. Downstream applications, including DDSPC, should not need to distinguish between simulated and real accelerometer data based on the stream interface.

In addition to matching the ImageStreamIO data format, `accelSim` should reproduce the relevant INDI interface of the existing accelerometer application so that it looks and behaves like the existing accelerometer system from the perspective of MagAO-X control software.

The initial implementation is limited to generating and verifying this simulated accelerometer interface. A later stage will use the same underlying simulated pointing disturbances to inject correlated tip/tilt disturbances onto a MagAO-X DM so that the real WFS measures a disturbance correlated with the simulated `acc000` data.

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
20. Publish the simulated accelerometer ImageStreamIO stream as `acc000` by default so that it can act as a drop-in replacement for the real accelerometer stream. The stream name may remain configurable if this follows existing MagAO-X conventions, but the normal simulator configuration should use `acc000`.
21. Match the real accelerometer stream contract, including width = 2, height = 1, datatype = `_DATATYPE_UINT16`, element 0 = X raw ADC count, and element 1 = Y raw ADC count.
22. Match the relevant INDI interface of the existing accelerometer application. Before implementation, inspect `mcp3208Ctrl` and its configuration to identify the INDI properties exposed by the real accelerometer system and determine which properties `accelSim` must reproduce for drop-in compatibility. Reuse existing MagAO-X interfaces and naming conventions rather than creating simulation-specific equivalents where possible.
23. Each published sample must have a meaningful acquisition timestamp. Stream counters, timestamps, circular-buffer behavior, and semaphore publication should match the existing accelerometer interface as closely as practical.
24. Use existing `frameGrabber` infrastructure for stream creation, timestamps, counters, circular-buffer handling, semaphore publication, and the applicable INDI controls rather than duplicating that functionality.
25. Use deterministic random seeding where practical for reproducible testing. Exact sample-by-sample equality with NumPy random-number generation is not required.
26. Do not implement transient/ring-down disturbances in the initial version.
27. Do not open, modify, or publish to a DM stream in the initial version.
28. Do not add instrument startup configuration in the initial version.
29. Avoid unrelated changes to the repository.

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
13. Verify that the simulator publishes `acc000` with the same datatype, dimensions, channel ordering, circular-buffer behavior, counters, timestamps, and semaphore behavior expected from the real accelerometer application.
14. Compare the INDI properties exposed by `accelSim` against the relevant INDI properties exposed by the existing accelerometer application. Properties required by downstream MagAO-X software should have compatible names, types, meanings, and behavior.
15. Verify that software which normally connects to the real accelerometer interface can connect to `accelSim` without requiring a simulation-specific stream name or interface change.
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

Planning review: 2026-10-26. Implementation was subsequently authorized by the user after answering Q1–Q6 below. The ANSWER entries are preserved as the authoritative decisions. Requirements above this heading take precedence over earlier discussion of an `accsim000` stream. Implementation and available workstation validation are recorded under Execution Record; full Linux integration/performance acceptance remains outstanding.

## Task Summary

Implement an accelerometer-only replacement producer for `acc000`, using the notebook's physical vibration model with stationary, continuously streaming AR(2) oscillators. Preserve the existing accelerometer's raw ADC layout and relevant INDI behavior so consumers can retain their normal stream and device configuration. Generate physical pointing, WFS-coupled pointing, acceleration, and ADC counts internally; publish only the two ADC counts. No DM stream access, transient model, or instrument deployment changes belong in this implementation.

Repository findings:

| Source | Finding and use |
| --- | --- |
| `apps/mcp3208Ctrl/mcp3208Ctrl.hpp` | Primary structural and interface template: `MagAOXApp<true>`, `frameGrabber`, `telemeter`, custom accelerometer INDI properties, timer/semaphore acquisition, channel-ordered UINT16 copy. Relevant methods are `setupConfig`, `loadConfigImpl`, `appStartup`, `configureAcquisition`, `startAcquisition`, both acquisition methods, and INDI callbacks. |
| `apps/mcp3208Ctrl/dependencies/MCP3208.{h,cpp}` | SPI implementation returns the 12-bit result without physical-unit calibration. Acquisition reads channel 0, then channel 1. No ADC voltage, accelerometer sensitivity, or divider simulation is needed in accelSim. Do not import this dependency or its `lgpio` link flags. |
| `apps/mcp3208Ctrl/tests/mcp3208Ctrl_test.cpp` | Existing hardware-free stubs and tests for channel order, synchronization, timing, and reconfiguration provide comparison cases. |
| `libMagAOX/app/dev/frameGrabber.hpp` | Owns the acquisition thread, ImageStreamIO allocation/publication, timing statistics, circular buffers, semaphore posts, and `fg_*` properties. This is the publication implementation to reuse. |
| `libMagAOX/app/dev/telemeter.hpp` | Needed by the current frameGrabber contract: `recordFGTimings` calls the derived app's `telem<telem_fgtimings>`. Include this mixin and its existing timing telemetry, without introducing a new telemetry schema. |
| `apps/cameraSim/cameraSim.hpp` | Secondary example for synthetic acquisition, setting `m_currImageTimestamp`, and frameGrabber lifecycle. Its `stdCamera` controls, image synthesis, and Pylon-related Makefile settings do not fit this app. |
| `apps/timeSeriesSimulator/timeSeriesSimulator.hpp` | Publishes slow INDI values through the main app loop, without frameGrabber. It is not the appropriate 6 kHz stream template. |
| `scratch/accel_ts_sim_ddspc.ipynb`, cells 11–16 | Physical-model reference, resolved prominent-mode parameters, high-frequency population construction, seeds, and offline PSD convention. |
| `libMagAOX/app/MagAOXApp.hpp`, `apps/template`, `Make/magAOXApp.mk`, `scripts/xctrl` | Standard main entrypoint, configuration loading, INDI device identity, build, and launch conventions. |
| `tests/Makefile.one`, `tests/tests.list`, `tests/groups.dox` | Focused Catch2 build/registration and application-test Doxygen grouping. |
| `utils/shmimLat` | Existing stream timing consumer, including an `acc000` invocation example. It records stream write timestamps and arrival times, so it alone does not verify acquisition timestamps or sample values. |
| `apps/loPredCtrl/loPredCtrl.hpp` | The checked-out consumer processes floating-point modal input; no raw-ADC `acc000` DDSPC integration was found here. An actual downstream consumer/configuration is needed for final compatibility acceptance. |

Applications normally use an app-local Makefile including `../../Make/magAOXApp.mk`, an entrypoint `.cpp`, and a class/implementation `.hpp`. MagAOXApp loads global and named-instance configuration under its configured config directory. `-n <instance>` sets the configuration/INDI device name, independently of the executable name; `xctrl` selects an executable from `proclist_<role>.txt` and launches it with that instance name. Thus changing the executable to accelSim need not change the INDI device name. A standalone launch with the default name `accelSim` would not by itself preserve the real device identity.

No deployed accelerometer `.conf` or process mapping was found in this repository. The top-level ACC build list still names `mcp3008Ctrl`, which cannot establish how the reviewed `mcp3208Ctrl` is deployed. Do not change that list or infer the live instance name from `acc000`.

## Key Assumptions

- Only one producer owns `acc000` in a given ImageStreamIO namespace. The operator stops the real producer before substituting the simulator and vice versa. This assumption also applies to the shared INDI instance name; matching a stream name alone is insufficient.
- Compatibility is with the reviewed `mcp3208Ctrl` interface and a two-channel acquisition configuration. The physical wiring's X/Y interpretation is supplied by the specification; the code establishes channel order, not sensor orientation.
- Default simulator rate is 6000 Hz even though this checkout of `mcp3208Ctrl` initializes its target to 2000 Hz. A synchronization source, if enabled, determines actual cadence; these are not guaranteed real-time rates merely because a target is configured.
- Configured mode strengths are acceleration RMS per axis. Derive pointing RMS from frequency and lever arm. Changing the lever arm changes pointing RMS inversely while retaining the configured acceleration RMS.
- The C++ implementation uses double precision and independent random draws across modes, axes, initialization, and sensor noise. Same seed/configuration/build gives reproducible numerical steps, not reproducible OS scheduling or cross-standard-library Gaussian sequences.
- Use two deterministic seeds as in the notebook: simulation seed 42 and mode-parameter seed 123. The 200 mode parameters can be generated once during initialization and retained; they are not regenerated on every acquisition restart. Exact NumPy sequences are not required. Record the resolved parameters in an offline test export for comparisons.
- Prefer the app header-only convention in AGENTS.md rule 21: declarations followed by out-of-class inline definitions, with only `main` in the `.cpp`. This is the specific app convention qualifying the more general source-placement rule 7.
- All agent-executed tests remain on a workstation or non-instrument development system, with isolated shared-memory and INDI paths. AGENTS.md prohibits executing agents or instrument/tunneled INDI commands on the instrument. Any eventual instrument-side checks must be performed by the user.

## Requirements

### Existing INDI contract and proposed compatibility

Use the same device/instance name when substituting the producer. Keep existing property and element names, types, access permissions, units, and meanings. The following inventory is from the reviewed source, not a live instrument enumeration.

| Owner | Property / elements | Existing behavior; simulator proposal |
| --- | --- | --- |
| MagAOXApp | `fsm.state` (read-only text), `fsm_clear_alert.request`, `logs_rotate.request`, `logs_maxtime.current/target` | Reuse the base class. No hardware power dependency is introduced. |
| MagAOXApp thread helper, invoked by frameGrabber | `th-framegrabber.pid/prio` (read-only number) | Reuse worker startup and priority/cpuset handling. |
| frameGrabber | `fg_shmimName.name` (read-only text) | Report `acc000` by default. |
| frameGrabber | `fg_frameSize.width/height/depth` (read-only number) | Report 2, 1, and configured circular depth. |
| frameGrabber | `fg_timing` (read-only number) | Reuse all elements: `acq_fps`, `acq_min`, `acq_max`, `acq_jitter`, `write_fps`, `write_min`, `write_max`, `write_jitter`, `delta_aw`, `delta_aw_jitter`. Rates are Hz; timing intervals are seconds. |
| telemeter | `telem_rotate.request`, `telem_maxtime.current/target` | Reuse telemetry controls and `telem_fgtimings`. |
| mcp3208Ctrl | `fps.current/target` (read-write number, range 0–10000, step 1, `%d`) | Current is the accepted nominal target, not a measured rate; measured rate comes from `fg_timing`. Preserve this distinction. Start at `sim.fps=6000`; validate updates against oscillator/Nyquist limits. See Q3. |
| mcp3208Ctrl | `numChannels.current/target` (read-write number, range 0–8, step 1, `%d`) | Hardware allows variable channels; simulator specification fixes X/Y. Proposed compatibility: preserve the property, accept 2, reject other values without resizing. See Q2. |
| mcp3208Ctrl | `alpha.current/target` (read-write number, range 0–1, step 0.00001, `%.5f`) | EMA coefficient for acquisition timing diagnostics; default 0.01. Preserve clamping to [0,1]. This is not AR damping `r`. |
| mcp3208Ctrl | `synchroDelay.current/target` (read-write number, range -1000000–1000000, step 1, `%d`, microseconds) | Signed phase offset relative to the synchronized timing model, not the measured or applied delay. Preserve if synchronization is required; see Q1. |
| mcp3208Ctrl | `timingDiag` (read-only numbers) | Preserve the elements below. Report simulator acquisition/service timing honestly; do not synthesize SPI measurements or claim synchronization lock without a source. |

`timingDiag` has these 23 elements:

```text
avg_read_latency_us          synchro_delay_us
synchro_delay_target_us      delay_applied_us
delay_model_us              delay_phase_error_us
delay_lock                  delay_budget_us
avg_non_delay_service_us     delay_capped
read_latency_error_us        avg_semaphore_period_us
channel_readout_us           trigger_interval_us
trigger_time_us              local_frame_seq
sync_frames_received         sync_frames_written
sync_frames_dropped          sync_frame_id_gap_count
sync_producer_frame_id       sync_producer_frame_delta
mode_code
```

All `_us` elements are microseconds. `mode_code` is 0 for timer acquisition and 1 for synchronization; lock/capped are numeric flags. The frame fields count local copies, semaphore arrivals, copies in synchronized mode, and gaps in the source producer's `cnt0`. `trigger_time_us` is a relative predicted trigger offset, not an epoch timestamp. `channel_readout_us` should measure the simulator's work to produce both channels. In timer mode, synchronization-only measurements/counters should follow the existing zero/not-locked conventions; retain the configured delay target as the hardware app does. If synchronized, measure source-to-acquisition latency and actual delay/service budgets. Reuse names, not fabricated hardware readings. There is no exposed SPI-connect, voltage, sensor-calibration, or ADC-range property to emulate.

The hardware app also **subscribes** to configurable external properties through `fps.device/property/element/tol` and `numChannels.device/property/element/tol`. These are configuration keys, not extra published `fpsSource` or `numChannelsSource` properties. Defaults select `fps.current` and `numChannels.current` on the configured devices. In this checkout the callbacks apply received values directly; the tolerance fields are loaded but not used to filter updates. Preserve external subscriptions that the actual consumer setup needs, subject to Q2/Q3 validation.

### Acquisition, synchronization, and reconfiguration findings

- `mcp3208Ctrl::appStartup` starts frameGrabber/telemeter, registers controls, connects SPI, and sets `OPERATING`. There is no dedicated accelerometer INDI acquire/start/stop switch. frameGrabber waits for `READY` or `OPERATING` and positive power state, calls `configureAcquisition`, creates the stream, calls `startAcquisition`, then repeatedly acquires and copies samples. Do not import camera acquisition controls from `stdCamera`.
- With empty `synchro.shmimName`, hardware acquisition uses elapsed `high_resolution_clock` time and a feedback-adjusted trigger interval, sleeping 1 microsecond while waiting. Local/external FPS callbacks update the nominal rate and trigger period directly. They do not request full stream reconfiguration. frameGrabber notices FPS changes and resizes its latency-history buffers independently of output circular depth.
- With a synchronization name, it opens the source stream, checks semaphore availability, claims a wait index (initial preference 5), flushes it at acquisition start, and waits with a one-second timeout. It tracks source `cnt0` gaps and producer timestamps. A missing/replaced source, detected through the backing-file inode, requests reconfiguration. `reconfig` closes the source and releases its `semReadPID` slot; configuration retries when a source is unavailable.
- The synchronized delay model uses the EMA of semaphore arrival intervals, `T`. Its model phase is `wrap(0.5*T - (dtTransfer + wfsProcess + dtF + wfsRead), T)`. Add the signed `synchro.postDelay` in microseconds and wrap again. Defaults for the four terms are 3000, 51500, 10000, and 276100 ns. Applied delay is capped by `max(0, T - average_non_delay_service - cadenceGuard)`, with default guard 20000 ns. Lock thresholds default to 50000 ns and 0.1 of a period. Preserve observable behavior if this mode is in scope; avoid claiming that an ideal sleep reproduces hardware latency.
- The synchronized path timestamps immediately before channel reads. The timer path currently does **not** assign `m_currImageTimestamp`. The simulator must timestamp every valid sample in both modes, as required by the specification, rather than copying this omission. No fix to mcp3208Ctrl is proposed.
- Hardware channel-count callbacks only change the member; they do not request the necessary resize/reconfiguration. Also, `accel.numChannels` is loaded but not registered in this app's `setupConfig`. These are existing inconsistencies to document, not reproduce or repair in this task.

### Exact output stream contract

| Item | Contract to retain |
| --- | --- |
| Name | Default `acc000`, configurable through `framegrabber.shmimName`. Set the app's default before frameGrabber configuration loading, so the executable's default instance name cannot accidentally select `accelSim` as the stream name. |
| Shape | Frame width 2, height 1; newly created frameGrabber streams have **three axes**, `[2, 1, D]`, including `D=1`. `D=framegrabber.circBuffLength`, default 1. Match the deployed value once supplied. |
| Storage | `_DATATYPE_UINT16`, two contiguous `uint16_t` values per circular plane: index 0 = X/channel 0, index 1 = Y/channel 1. Values remain in [0,4095]. No floating-point acceleration stream or extra channels. |
| Creation | frameGrabber uses `ImageStreamIO_createIm_gpu` with CPU location -1, shared memory enabled, `IMAGE_NB_SEMAPHORE` semaphores, zero keywords, and `CIRCULAR_BUFFER | ZAXIS_TEMPORAL`. Keep the installed ImageStreamIO constant rather than hardcoding a semaphore count. |
| Write transaction | Framework sets `write=1`, calls `loadImageIntoStream` to copy the pair, sets real-time `writetime`, assigns `atime=m_currImageTimestamp`, sets `cnt1` to this plane, increments `cnt0`, and updates per-plane `writetimearray`, `atimearray`, and `cntarray`. It then clears `write` and calls `ImageStreamIO_sempost(image,-1)` to post all semaphores. |
| Ring progression | Creation initializes `cnt1=D-1`; first publication uses plane 0, then wraps modulo D. Consumers use `cnt1`/per-plane counters to select the latest plane. `cnt0` counts published frames, not simulated ticks skipped after an overrun. |
| Timestamp meaning | One acquisition timestamp for the X/Y pair, in `CLOCK_REALTIME` timespec format; write timestamp is later publication time. Internal pacing uses a monotonic clock. No guarantee of epoch continuity across wall-clock corrections. |
| Lifecycle | Framework owns stream destruction on shutdown and reconstruction when needed. Counters need not persist across producer replacement. Semaphore wakes are notifications, not an unlimited queue of samples; consumers must detect gaps by counters. |

No application-level ImageStreamIO output creation, manual counter increment, or semaphore publication is needed. A synchronization input stream, if approved under Q1, is separate from this output ownership.

### Notebook model to preserve

Each row specifies per-axis acceleration RMS; X/Y have identical requested amplitudes but independent realizations.

| Frequency (Hz) | Acceleration RMS (m/s²) | r | WFS coupling X / Y |
| --- | --- | --- | --- |
| 11, 14, 17, 20, 25 (five modes) | 0.00025 each | 0.9999999 | 0.05 / 0.05 |
| 28 | 0.0020 | 0.99999 | 1 / 1 |
| 54 | 0.0020 | 0.99995 | 1 / 1 |
| 56 | 0.0010 | 0.99995 | 1 / 1 |
| 58 | 0.0020 | 0.99995 | 1 / 1 |
| 75, 85 (two modes) | 0.0008 each | 0.9998 | 1 / 1 |
| 120 | 0.0080 | 0.9980 | 1 / 1 |
| 220 | 0.0150 | 0.9960 | 1 / 1 |
| 300 | 0.0180 | 0.9950 | 1 / 1 |

Add 200 modes with frequencies uniform on [100,900), parameter seed 123. Draw the frequency population first, then amplitudes, following the notebook's parameter-generation order. For each frequency:

```text
r = 0.998 - (f - 100)/800 * 0.006
envelope = exp(-0.5*((f - 250)/250)^2)
accelRMS = uniform(0.0005, 0.0020) * (0.5 + 2*envelope)
couplingX = couplingY = 0.01
pointingRMS_arcsec = accelRMS / ((2*pi*f)^2 * leverArm * pi/(180*3600))
```

Sort the resulting 214 modes by frequency. Keep distinct X/Y coupling fields even when their defaults agree. The active notebook's angular X/Y-to-tip/tilt transform is a 30-degree rotation (class default identity); defer the transform and all DM work. Transients are supported by notebook code but the active example is commented out. Welch PSD calculation remains offline: Hann window, constant detrending, density scaling, 10-second segments, and SciPy's default 50% overlap/one-sided output.

### Streaming normalization and initialization

For desired pointing variance `v=s²`, let `gamma1=E[x[n]x[n-1]]`. The stationary Yule–Walker equations give:

```text
gamma1 = phi1*v/(1-phi2)
gamma2 = phi1*gamma1 + phi2*v
q = Var(w) = v - phi1*gamma1 - phi2*gamma2
  = v*(1+phi2)*((1-phi2)^2-phi1^2)/(1-phi2)
```

Use the following equivalent factorization to avoid subtracting nearly equal squared coefficients for high-r modes:

```text
omega = 2*pi*f/fs
phi1 = 2*r*cos(omega); phi2 = -r*r
d = (1-r)*(1+r)
B = d*d + 4*r*r*sin(omega)^2
sigmaInnovation = s*sqrt(d*B/(1+r*r))
rho = phi1/(1+r*r)
eta = sqrt(B)/(1+r*r)           # sqrt(1-rho*rho), evaluated stably
xPrevious1 = s*z1
xPrevious2 = s*(rho*z1 + eta*z2)
```

Here `z1,z2` are independent standard normals, independent of all future innovations. Both histories have variance `s²` and covariance `rho*s²`. Draw a separate pair for each axis of each mode. Store coefficients, innovation standard deviation, acceleration conversion factor, and four history values per mode. Coefficients and conversion factors are computed outside the sample loop.

Validate finite inputs, `0 <= r < 1`, positive lever arm, nonnegative RMS/noise PSD, and `0 < f < fs/2`. Zero-RMS modes produce zero state/innovation. Calibration requires finite zero point and positive volts-per-bit/sensitivity; reject nonfinite generated values before integer conversion.

The notebook starts `lfilter` with zero histories, subtracts the full-record mean, and scales by the measured standard deviation after `min(2*fs,N/10)` samples. It retains those initial samples in the returned record. The streaming model intentionally does none of these future-dependent operations. At `r=0.9999999`, `fs=6000`, the amplitude-envelope time constant `-1/(fs*log(r))` is about 1667 s. Sixty seconds spans only 0.036 time constants. Stationary initialization supplies the correct ensemble variance immediately, but cannot force every 60-second realization to have exactly the configured RMS. Test ensembles rather than imposing the notebook's per-record normalization.

Sensor noise standard deviation is `sqrt(sensor.noisePSD*fs/2)`, about 0.00547723 m/s² at defaults. Add independent noise to the two summed mode accelerations, then compute `adc.zero + acceleration*adc.sensitivity/adc.voltsPerBit`. Proposed tie rule is `std::round` (half away from zero), then floating-point clamp to [0,4095], then `uint16_t`. Count actual clamp operations separately for X/Y and lower/upper rail; values rounding naturally onto a valid rail are not clipping events. The physical acceleration state retains its unclipped value.

### Requirement coverage

The test identifiers below are defined in the Tests section. Every numbered requirement above is accounted for:

| Specification requirements | Planned implementation / verification |
| --- | --- |
| 1–3 | Standard app layout, existing feature-branch workflow, frameGrabber/telemeter macros; T1, T7, T8. |
| 4–10 | Sample-wise double AR(2), direct r, analytic innovations and stationary histories; T2, T3, T9. |
| 11–12 | Reference 14+200 modes and lever arm 10 m; T2, T4, T10. |
| 13–16 | Physical and WFS-coupled pointing, modewise acceleration, independent sensor noise; T4, T10. |
| 17–19 | Configurable calibration, round/clamp/UINT16, clipping counters; T1, T5. |
| 20–24 | `acc000`, exact framework publication, meaningful acquisition times, INDI identity and behavior; T6–T9. |
| 25 | Deterministic initialization/stepping for a fixed implementation; T2, T3. |
| 26–29 | No transients, DM access, startup configuration, or unrelated changes; T1 review and final diff audit. |

## Questions and Points of Clarification

**Resolved:** Q1–Q6 were answered and implementation was authorized. Each ANSWER below supersedes the preceding proposed choice where applicable. Deployed configuration and consumer inputs remain unavailable by acknowledged limitation, not an unanswered implementation question.

1. **Which deployed configuration and synchronization behavior define drop-in acceptance?** Please supply the real INDI instance name, accelerometer configuration (including circular depth, external FPS/channel subscriptions, and `synchro.*` settings), and the raw-ADC DDSPC consumer/configuration to test. Neither the live mapping nor that consumer is present here. Recommend implementing both timer and optional semaphore-synchronized acquisition for full mcp3208Ctrl compatibility. If only timer mode is required initially, explicitly approve that narrower compatibility claim; do not expose a working-looking synchronization control that is silently ignored.

ANSWER: Implement both timer-based and optional semaphore-synchronized acquisition so that accelSim preserves the relevant mcp3208Ctrl behavior. Use mcp3208Ctrl as the authoritative reference for the synchronization interface and configuration names. I do not currently have the deployed accelerometer configuration or raw-ADC DDSPC consumer configuration available in this repository. Do not invent those values. Keep the applicable settings configurable so they can be matched to the deployed system later. The normal simulator output stream remains acc000. The exact INDI instance name will be supplied at launch using the normal MagAO-X instance/configuration mechanism rather than hard-coded into accelSim.

2. **Fixed two channels versus writable `numChannels`:** Requirement 21 fixes the stream at X/Y, while hardware exposes values 0–8 and external channel updates. Recommend retaining the property/elements and accepting only 2, returning an INDI error/Alert for other requests while leaving the stream unchanged. Confirm this intentional restriction and whether the external channel subscription is used. Do not reproduce the hardware callback's missing resize behavior.

ANSWER: Yes. Preserve the numChannels INDI property for interface compatibility, but accelSim is fixed to two channels (X/Y). Accept 2 and reject other requested values without changing the stream shape. Do not reproduce the existing mcp3208Ctrl resize inconsistency. Preserve the external channel subscription interface if that is part of the existing mcp3208Ctrl interface, but values other than 2 should likewise be rejected.

3. **FPS changes and invalid rates:** The hardware property advertises 0–10000 Hz; the reference population needs a rate greater than twice its highest frequency. Recommend retaining the compatible property metadata, rejecting zero/nonfinite/aliased rates, and retaining the last valid setting. For an accepted rate change, apply it at an acquisition boundary, recompute oscillator/noise coefficients, draw stationary histories for the new covariance, and reset pacing; document a model discontinuity rather than falsely preserving stationarity with old histories. Confirm that this restart behavior is acceptable. In synchronized mode, recommend using the configured/subscribed nominal FPS for the discrete model and checking source cadence against it, rather than varying AR coefficients with every jittering semaphore interval.

ANSWER: Approved. Preserve the existing fps INDI property metadata for compatibility, but reject invalid, nonfinite, zero, or aliased sample rates and retain the last valid setting. For an accepted FPS change, apply the change at an acquisition/sample boundary, recompute the AR(2) coefficients, innovation scaling, and sensor noise scaling, and reinitialize the oscillator states from their stationary distribution at the new sample rate. A discontinuity at an FPS change is acceptable and should be documented. In synchronized mode, use the configured/subscribed nominal FPS for the simulation model. Do not continuously change AR(2) coefficients based on individual semaphore timing jitter.

4. **Missed periods and long stalls:** The specification asks to measure missed deadlines but does not define recovery. Recommend advancing the model over missed nominal ticks, publishing only the current pair, and counting skipped publication opportunities. Bound catch-up work; for a long stall, report a discontinuity and restart in stationarity instead of publishing a burst or entering an unlimited catch-up loop. Approve this policy and the acceptable recovery bound before implementation. It affects reproducibility relative to wall time, not deterministic step-by-step numerical tests.

ANSWER: Approved in principle. For ordinary missed deadlines, advance the simulation state through the missed nominal ticks so that the simulated physical process remains consistent with elapsed simulation time, but publish only the newest sample. Count and report skipped publication opportunities. Do not publish a burst of old samples to catch up. Bound the amount of catch-up work. If the required catch-up exceeds a reasonable implementation-defined bound, record/report the timing discontinuity and restart the oscillator states in stationarity rather than entering an unbounded catch-up loop. Choose and document a conservative catch-up bound during implementation and include it in the tests. If selecting that bound has significant architectural consequences, bring it back for review before proceeding.

5. **WFS coupling scope:** Requirement 13 requires internal WFS-coupled pointing, while Scope and Caveats also lists its generation as future work. Recommend computing the two weighted sums now from the same mode samples, without a transform, output stream, or DM access. This satisfies requirement 13 with negligible extra work. Confirm that interpretation.

ANSWER: Yes. Compute and retain the WFS-coupled X/Y pointing internally now using the per-mode X/Y coupling coefficients from the Python model. Do not apply the X/Y-to-tip/tilt transform yet, and do not publish this quantity to another stream or write anything to a DM. The purpose is to retain the physical state needed for the later DM-injection phase while keeping the current implementation limited to acc000 output.

6. **6 kHz acceptance threshold:** A sustained 6000 Hz target is specified without a benchmark duration, allowed jitter/miss rate, or development-host class. Recommend a 10-minute run on a non-instrument Linux development host with the normal framework thread settings: mean write rate within 1% of 6000 Hz, no invalid samples or unexplained counter gaps, bounded memory, and all deadline/consumer misses reported. This is a proposed acceptance threshold, not an assertion that general-purpose workstation scheduling guarantees every 166.667-microsecond interval. Set the required miss/jitter limits before declaring timing acceptance.

ANSWER: Use the proposed 10-minute sustained test on a non-instrument Linux development host as the initial performance test.For this stage, require: mean publication rate within 1% of 6000 Hz, no invalid ADC samples, no unexplained stream counter gaps, bounded/stable memory use, all missed deadlines/skipped publications explicitly counted and reported.
Record the interval distribution, jitter, lateness, missed periods, and maximum observed delay, but do not establish a hard pass/fail jitter or miss rate yet. The purpose of this test is to determine whether the implementation can reasonably sustain the target 6 kHz cadence and characterize its timing performance. Final real-time jitter/miss acceptance criteria should be set after testing on the intended MagAO-X Linux/real-time environment.

## Tests

The original planning review performed no runtime/build tests. Implementation verification uses the existing Catch2 harness plus isolated process/stream checks and offline Python analysis; actual results and unrun checks are listed under Execution Record. Statistical tests state sample size and tolerances and avoid testing high-r stationarity from one short correlated record.

| ID | Planned checks and acceptance evidence | Tests and Metrics numbers covered |
| --- | --- | --- |
| T1: configuration/build/scope | Build the app and focused tests through normal Makefiles. Parse the example; check every default and override, invalid/nonfinite rates, r, lever arm, noise, and calibration; check failure before publication. Audit include/link dependencies and the diff for no SPI, DM, startup files, or unrelated changes. Check Doxygen, formatting, and test registration. | Supports all tests and requirements 1–3, 26–29. |
| T2: coefficients/model/seed | Verify known coefficients (including r=0 and a quarter-sample-rate mode), all 14 prominent rows, exactly 200 additional modes, bounds/envelope/r/coupling formulas, sorted 214-mode list, and per-axis pointing RMS conversion. Same seed/configuration yields identical initialization and subsequent numerical samples within one C++ build; different seeds produce different sequences. Reinitialization clears any Gaussian-distribution cached state. | 1, 12. |
| T3: stationary normalization/independence | Check q analytically using the stationary covariance equation `P=A*P*A^T+Q`; include r=0.9999999 and low frequencies to exercise stable factorization. Across independent initialized histories verify variance/covariance with tolerances derived from sampling error (e.g. six standard errors), plus the small conditional residual variance. Repeat after one/multiple steps to establish invariant covariance. Check cross-axis and cross-mode covariance in ensembles, not just inequality of samples. Longer moderate-r runs provide a separate RMS check. | 2, 3, 4, 12. |
| T4: physical model/noise | Known-angle/frequency/lever-arm conversion checks sign and arcsec-to-radian scale; two different-frequency modes catch conversion-after-summing errors. Check weighted X/Y WFS sums use their own coefficients and contain no sensor noise. Check zero-noise mode, noise-only variance `PSD*fs/2`, independent XY draws, and noise scaling after an FPS change. | 5; physical/noise requirements 11–16. |
| T5: ADC | Zero acceleration = 2048; calibrated positive/negative inputs; configurable zero/sensitivity/volts-per-bit; rounding just below/at/above half-counts; exact rails; under/overflow rounded counts; float clamp before cast; type assertion for `uint16_t`; all outputs within 12 bits. Check per-axis/rail clipping totals, no count for a naturally valid rail, and rejection of NaN/infinity. | 6–11. |
| T6: actual output contract | Start an isolated producer with default `acc000` name in a private ImageStreamIO namespace; test D=1 and D>1. Open with a real ImageStreamIO reader, check three axes, UINT16, X/Y order with an asymmetric numerical fixture, flags and installed semaphore count. Follow many wraps: `cnt1`, `cnt0`, per-plane timestamps/counters, valid `write` at notification, and reader wakeups after complete publication. Check acquisition stamps are populated and ordinarily precede write time. No extra counter increments/posts on invalid acquisition or shutdown. Restart and reconnect reader; test configurable output name. | 11, 13. |
| T7: INDI/lifecycle | Compare serialized property definitions and callback behavior against the inventory and hardware-free mcp3208Ctrl fixture: device/name/type/permission/elements/ranges/units, FPS nominal versus measured, alpha, channel restriction, delay, all diagnostics, base/telemeter controls. Verify local and external changes, invalid-update behavior, OPERATING acquisition, orderly shutdown, and application of accepted rate changes without callback/worker data races. | 14. |
| T8: downstream substitution/synchronization | Use the unchanged supplied consumer configuration in an isolated local server/session, substituting only the producer executable while preserving INDI instance and `acc000`. Verify raw UINT16 ingestion without a physical-unit conversion requirement. For approved sync scope, use a synthetic local trigger stream: one pair per accepted trigger, phase offset/model/wrap/cap, diagnostics, counter gaps, timeout/EINTR, stale source replacement, semaphore release, and reconnection. Never connect the test to a DM or instrument tunnel. | 13–15. |
| T9: timing/recovery | Fake-clock tests cover deadlines, fractional-nanosecond period accumulation, interruption/shutdown, rate changes, missed-period accounting, and approved bounded recovery. Then run the sustained 6 kHz benchmark with all 428 oscillators and real framework publication. Measure acquisition/write rate, interval distribution, lateness percentiles/max, whole-period misses, skipped publications, reader counter gaps, and CPU/memory. Separate producer misses from a slow reader's misses; independently record `atime` and `writetime`. | 16. |
| T10: offline PSD/RMS | Export numerical test samples and the resolved mode table without adding an RT PSD path. Use Python Welch settings from the notebook and compare ensemble modal RMS and theoretical AR spectra, prominent frequencies, broad envelope, and noise floor. Analyze raw stream data after applying the inverse calibration only for validation; account for quantization/clipping. Do not demand that each 60-second high-r trace has the target RMS. | 17 and statistical interpretation of 2–5. |

Use `libXWCTest::accelSimTest`, per-TEST_CASE Doxygen blocks, and an `accelSim_unit_test` group under `application_unit_test`. Where test access wrappers prevent links, use `tests/testXWC.hpp` and the AGENTS.md Doxygen-reference convention. The existing mcp3208Ctrl test stubs permit local interface comparison without opening SPI hardware. Integration tests requiring unavailable ImageStreamIO/INDI dependencies must be reported as unrun, not counted as passing through mocks.

For numerical checks independent of the notebook's record normalization, the two-sided per-Hz AR spectrum is `q/fs / |1-phi1*exp(-i*w)-phi2*exp(-2*i*w)|²`; double interior positive-frequency bins for the one-sided density. Compare acceleration spectra after each mode's conversion-factor squared. ADC quantization alone has an approximate uniform-error PSD of `LSB_accel²/(6*fs)` when the usual quantization-noise approximation applies; with defaults `LSB_accel=0.0083799/1.02`, this is about 1.87e-9 (m/s²)²/Hz. Thus raw-count spectra need not reproduce the pre-ADC 1e-8 noise floor exactly, and pre-ADC numerical tests remain necessary.

## Implementation Plan

Implementation was authorized after approval of this plan and all six answers. The approved sequence below is retained as the design record; completion and validation status follow it.

1. **Resolve acceptance configuration and freeze the interface.** Record answers to Q1–Q6 here. Retain the existing namespaced feature branch (`parkertjohnson1/accelSim` at review time), following AGENTS.md for any later commits/PR. Capture the agreed INDI inventory and stream depth as test expectations. Do not edit real producer code to make a comparison pass.

2. **Add the small numerical core and its focused tests.** Keep it independent of ImageStreamIO/INDI so numerical stepping can be tested without a live app. Suggested app-local types are a mode specification, an AR(2) state, and a simulation model; no new shared base class. Each resolved mode holds frequency, r, acceleration RMS/derived pointing RMS, separate coupling X/Y, coefficients/innovation scale, acceleration-per-arcsec factor, and independent X/Y histories. The model owns RNG/distribution state, mode storage, current physical/WFS pointing pairs, acceleration pair, raw ADC pair, sample index, and clipping totals. Allocate/initialize once; no growing histories or 60-second precomputation.

3. **Add the app and example configuration.** Derive from `MagAOXApp<true>`, `dev::frameGrabber<accelSim>`, and `dev::telemeter<accelSim>`. Use friendship/type aliases and the `FRAMEGRABBER_*`/`TELEMETER_*` interface macros. Set `c_frameGrabber_flippable=false`. Keep mcp3208Ctrl's lifecycle and controls, replacing hardware acquisition with the numerical core. `appLogic` updates INDI/telemetry at the ordinary app cadence; it does not execute the 6 kHz simulation. Use a short, coherent handoff of pending control settings to the acquisition thread; the worker exclusively owns oscillator/RNG state. Never hold the INDI mutex while waiting, stepping all modes, or copying to the stream.

   Proposed minimal configuration:

   | Key | Default / role |
   | --- | --- |
   | `sim.fps` | 6000; initial nominal rate, subsequently controlled by compatible `fps` INDI interface. |
   | `sim.seed` | 42; oscillator initialization, innovations, and sensor-noise reproducibility. |
   | `sim.modeSeed` | 123; independent deterministic generation of the high-frequency population. |
   | `sim.leverArm` | 10 m. |
   | `adc.zero` | 2048. |
   | `adc.voltsPerBit` | 0.0083799, used as the supplied empirical calibration constant. |
   | `adc.sensitivity` | 1.02. |
   | `sensor.noisePSD` | 1e-8 (m/s²)²/Hz, one-sided. |
   | `framegrabber.shmimName` | `acc000` in code and example. |
   | `framegrabber.circBuffLength` | 1 unless matching a supplied different deployment value. |
   | `framegrabber.threadPrio`, `cpuset`, `latencyTime`, `latencySize` | Existing framework settings; defaults 2, empty, 5 s, and 100000 samples respectively. |
   | `fps.device/property/element/tol`, `numChannels.device/property/element/tol` | Existing external-subscription keys as needed for approved compatibility. |
   | `synchro.shmimName`, `postDelay`, `alpha`, timing-model and lock/guard keys | Preserve the hardware configuration names/defaults if Q1 includes synchronized acquisition; empty source selects timer mode. |
   | `accel.numChannels` | Explicitly register as fixed-2 compatibility configuration if retained under Q2. |

   All keys after the slash in the grouped rows retain their section prefix. Keep the reference mode list as defaults in the numerical core; do not add a general mode-file format, new runtime mode editor, or transform configuration in this stage. Keep r as a direct mode parameter. Clipping/lateness totals can be exposed through one additional read-only simulator diagnostic property and a shutdown summary, without renaming or changing the existing `timingDiag` contract. Do not log per sample.

4. **Implement acquisition and publication.** `configureAcquisition` sets width 2, height 1, and UINT16; it prepares any approved synchronization input. `startAcquisition` establishes pacing and timing counters. Numerical histories are initialized on initial model setup or an approved model restart, not whenever an ordinary INDI diagnostic refresh runs. `fps()` returns the accepted model sample rate. `loadImageIntoStream` copies exactly the two prepared ADC values and updates local accounting; frameGrabber performs the output transaction. `reconfig` releases/reopens any synchronization input and applies pending changes under the approved policy. Shutdown returns a nonzero acquisition result if no fresh sample was generated, preventing publication of an old pair.

   Each normal simulation tick performs these operations in order:

   1. Check shutdown/reconfiguration and apply an accepted pending rate change at a sample boundary.
   2. Wait for the next monotonic timer deadline or an approved synchronization trigger/phase delay. Account for missed periods according to Q4. No successful acquisition return occurs merely because a wait was interrupted.
   3. Capture `CLOCK_REALTIME` into `m_currImageTimestamp` at the acquisition event, immediately before generating this pair. Track actual monotonic acquisition time separately for pacing/lateness.
   4. Clear current physical-pointing, WFS-pointing, and modal-acceleration accumulators.
   5. For each mode, draw independent X/Y innovations, advance each two-sample history, accumulate physical pointing, accumulate each axis's coupled pointing, and add that mode's negative frequency-squared acceleration contribution.
   6. Draw and add independent X/Y accelerometer noise using the current nominal sample rate. Retain the resulting physical acceleration pair.
   7. Apply calibration, round, clamp in floating point, update clipping counts, and store the two `uint16_t` values.
   8. Update local timing/sample accounting and return a valid acquisition. frameGrabber copies the pair, fills metadata, and posts semaphores. Advance the nominal deadline based on the schedule, not on the end of this publication.

5. **Schedule and verify 6 kHz.** Use the existing frameGrabber thread/priority/cpuset facilities, not a second publication thread. Timer deadlines should be absolute on `CLOCK_MONOTONIC` (or the repository's equivalent monotonic facility), with time derived from an epoch and tick index/fractional period so 1e9/6000 is not truncated cumulatively. Sleep to the deadline with interruption checks; measure whether a bounded final wait is needed rather than introducing an unconditional busy loop. This changes the hardware app's feedback pacing algorithm, not the INDI/stream contract. Keep waiting bounded for shutdown/reconfiguration. Precompute all trigonometric factors, avoid allocations and INDI operations per tick, and benchmark the full path, not just the oscillator arithmetic. Apply Q4 recovery explicitly and distinguish simulation ticks from published-frame counters.

6. **Verify compatibility before declaring completion.** Run T1–T10 in stages: numerical tests, app/callback tests, real local stream publication, unchanged consumer and optional trigger tests, sustained rate, then offline PSD. Maintain normal telemetry during performance tests. Check the effective build flags: `Make/common.mk` defaults to `-ffast-math`, which can invalidate finite-value checks. Use an app/test-local build setting that preserves those checks and stable near-unit-root arithmetic; do not change global flags or dependencies. Record any environment build limitations, without patching unrelated library/platform code.

7. **Document and review.** Update this engineering record with results, resolved questions, and limitations. Document local build/launch/inspection commands, calibration, seeds, timing recovery, and the intentional normalization differences. Review the final diff, run clang-format on new C++ files, and follow the repository's commit separation and AI-attribution rules when preparing commits/PR. No startup/process-list deployment changes are part of that PR.

Approved file set (created/modified during the implementation described below):

| Action | File | Purpose |
| --- | --- | --- |
| Create | `apps/accelSim/accelSim.hpp` | App lifecycle, compatible INDI controls, acquisition scheduling, and frameGrabber integration. |
| Create | `apps/accelSim/accelSim.cpp` | Main entrypoint only. |
| Create | `apps/accelSim/accelSimModel.hpp` | Small app-local numerical model and reference mode construction; inline out-of-class definitions. |
| Create | `apps/accelSim/Makefile` | Standard app build with the model header dependency and appropriate local numerical flags. No SPI/camera SDK dependencies. |
| Create | `apps/accelSim/config/example.conf` | Minimal standalone example, default `acc000`; no instrument startup installation. |
| Create | `apps/accelSim/tests/accelSim_test.cpp` | Numerical, configuration, callback, scheduling, and stream checks using the existing harness. |
| Create | `apps/accelSim/tests/validate_psd.py` | Small offline analyzer for exported numerical/stream samples; no production Python dependency. |
| Create | `apps/accelSim/README.md` | Interface, model, local validation instructions, and limitations. |
| Modify | `tests/tests.list`, `tests/groups.dox` | Register the app tests and application-unit-test documentation group. |
| Modify, narrowly scoped | `tests/Makefile.one` | Add only an accelSim-test conditional for numerical compiler flags and dependencies on the app/model headers, so normal test-suite builds exercise the same floating-point assumptions as the app. |
| Update | This planning document, below the agent heading only | Decisions and execution/verification evidence. |

No top-level app build-list change is necessary for an app-local build. No modification to mcp3208Ctrl, cameraSim, frameGrabber, the reference notebook, DM code, or startup configuration is planned. If synchronization compatibility exposes a need for shared-code extraction, bring that back for review rather than expanding this feature into a refactor.

Expected build/test commands after implementation, from the repository root on a configured development host:

```sh
make -C apps/accelSim
make -C tests -f Makefile.one t=../apps/accelSim/tests/accelSim_test.cpp
./apps/accelSim/tests/accelSim_test
```

Local launch documentation will use `./apps/accelSim/accelSim -n <accelerometer-instance> -c apps/accelSim/config/example.conf` within isolated MagAO-X configuration/log/FIFO and ImageStreamIO paths. The actual instance and isolation setup must be supplied/verified before treating this as a runnable compatibility command. An unchanged ImageStreamIO reader and `shmimLat -n acc000` can then inspect publication; use the dedicated reader checks for values, acquisition stamps, and ring metadata. No launch or INDI command has been executed as part of this plan.

## Follow-up and Edge Cases

- Q1–Q6 are resolved and implementation is authorized. Missing deployment configuration and the actual raw-ADC DDSPC consumer still prevent claiming complete deployed-system compatibility; Q1 explicitly accepts this limitation while preserving configuration interfaces.
- A matching producer name/data format does not coordinate two simultaneous producers, migrate existing mappings across recreation, or preserve counters across restart. Retain the one-producer operating assumption and test consumer reconnects; do not introduce a producer-arbitration service.
- Very high-r modes have slowly changing random amplitudes. Analytic normalization preserves ensemble RMS, not exact RMS in each short record. A restart or rate change may visibly change those amplitudes. No per-window renormalization should be added to hide this behavior.
- A synchronized trigger slower than nominal FPS changes physical frequencies if only one fixed-dt step is taken per trigger. Q3 must establish a consistent nominal source rate, and Q4 must cover missing triggers/backlog without misrepresenting timing. Do not silently retune coefficients to every measured jitter interval.
- Full-record mean subtraction, NumPy RNG sequences, and the notebook's arbitrary 60-second duration are intentionally not reproduced. Use the C++ resolved mode table for numerical spectral comparisons rather than expecting identical random high-frequency peak locations from NumPy.
- Validate nonfinite/overflowing configuration and generated state before ADC casting. Handle zero noise/RMS, r near one, rates near Nyquist, shutdown during waits, trigger recreation/counter reset, clipping counters over long runs, and clock corrections. Finite-precision near-unit-root behavior should be exercised under the actual optimized compiler settings.
- Timestamp accuracy and attainable 6 kHz jitter depend on OS scheduling and load. Passing a macOS numerical build is not evidence of Linux ImageStreamIO/INDI compatibility or real-time cadence. Record missing dependencies or unrun tests explicitly; instrument-side follow-up remains user-run under AGENTS.md.
- The hardware timer timestamp omission, channel reconfiguration inconsistency, and unregistered channel configuration key remain outside this feature's blast radius. Simulator behavior that intentionally differs is listed above for review.
- Future work only: transient/ring-down disturbances, configurable angular X/Y-to-tip/tilt transform, calibrated angular-to-DM conversion, a dedicated DM disturbance channel, real WFS measurement, and end-to-end DDSPC. Retained physical/WFS mode state supports this later work without implementing any DM output now.

### Execution Record

Implementation date: 2026-10-26. No commits, pushes, instrument-side commands, or instrument/tunneled INDI testing were performed. All edits are within the approved file set. The user-written specification above the agent heading and all ANSWER entries are preserved.

**Implemented:**

- App-local header-only numerical model and app, main entrypoint, standard Makefile, example configuration, README, Catch2 tests, and offline Python analyzer. Test registration and local numerical compile flags are added only for accelSim.
- `MagAOXApp<true>`, `frameGrabber<accelSim>`, and its required `telemeter<accelSim>` integration with existing helper macros. No output ImageStreamIO allocation/counter/semaphore code is duplicated in the app. The only direct mapping lifecycle in the app is for the optional synchronization input.
- Reference 14+200 modes, direct r, independent stationary X/Y states, stable analytic innovation scaling, double-precision physical quantities, per-mode acceleration, independent sensor noise, separate internal WFS sums, and calibrated round/clamp/UINT16 conversion. Parameter and simulation seeds are separate; the population is resolved once per configuration load.
- Default `acc000`, fixed 2×1 UINT16 frames, framework-managed circular depth/metadata/publication, and acquisition timestamps for both timer and synchronized paths. All 23 hardware timingDiag names and their order were checked against the existing source.
- Compatible FPS, channel, alpha, synchronization-delay, and external-subscription interfaces. Only two channels are accepted. Invalid FPS requests preserve the last accepted rate; accepted changes update the model at an acquisition boundary and redraw stationary histories. No `stdCamera`, SPI, DM, coordinate transform, transient, or startup/deployment code was added.
- Timer pacing uses monotonic epoch/tick deadlines with rounded accumulated intervals, not rounded individual periods. Synchronized pacing uses the hardware phase model and configuration names, follows producer counter gaps, suppresses repeated notifications for the same counter, and reconnects after stale-source timeouts. Arrival cadence is reported relative to nominal model FPS; no coefficient retuning based on jitter occurs.
- **Q4 implementation choice:** at most eight skipped ticks are numerically advanced before one current sample is published. More than eight skipped opportunities causes a counted stationary restart. At 6 kHz the catch-up window is 1.333 ms; no history allocation or publication burst is introduced. The exact bound and both branches are tested. A long synchronization pause also invokes bounded recovery even if the source counter did not advance.
- `sim_diag` reports skipped opportunities, discontinuities, positive deadline lateness count/max, source cadence ratio, and X/Y lower/upper clamp counts. A shutdown summary records timing and clipping totals. Clipping counts include generated catch-up samples; a rail value that did not require clamping is not a clipping event. Diagnostics use a nonblocking worker-to-app snapshot, while runtime control requests use atomics and a short property mutex.
- Configuration and callback tests, real framework-stream and synthetic-trigger tests, and a separate ten-minute live-reader benchmark are provided. Runtime test groups are explicit opt-in commands in the README. The live-reader benchmark observes a separately launched normal app with telemetry active; RSS and producer diagnostics must also be collected as documented, rather than confusing reader counter gaps with producer skips.

**Available validation and results:**

| Check | Result / evidence |
| --- | --- |
| Normal app build: `make -C apps/accelSim` | Attempted; blocked in existing `libMagAOX` compilation before the new app is compiled. Errors: undeclared `timegm` in `libMagAOX/file/stdFileName.hpp` and missing installed `mx/math/floatUtils.hpp` included by `telem_pupilfit.hpp`. No unrelated dependency/platform changes made. |
| Normal focused test build: `make -C tests -f Makefile.one t=../apps/accelSim/tests/accelSim_test.cpp` | Attempted; blocked by the same existing header/platform errors. The compile command includes the scoped `-fno-fast-math` flag. The full app/INDI/stream test binary could not be built here. |
| Standalone numerical test build | Passed using `c++ -std=c++20 -O3 -fno-fast-math -Wall -Wextra -DCATCH_CONFIG_MAIN -DACCELSIM_MODEL_ONLY apps/accelSim/tests/accelSim_test.cpp -o /tmp/accelSim_model_test`. This intentionally excludes MagAO-X-dependent tests; it is not a substitute for a full app build. |
| `/tmp/accelSim_model_test '[model]'` | Passed: **8 cases, 8583 assertions**. Covers coefficients, innovation variance, stationary covariance and propagated variance, high-r conditional residual variance, X/Y independence, all prominent parameters and population envelope, deterministic initialization/steps, physical conversion and WFS coupling, noise PSD/rate scaling, ADC rounding/type/clipping, invalid inputs, absolute pacing, and bounded recovery. Stationary/noise ensembles use 100000 samples and six-standard-error tolerances. |
| Numerical address/undefined-behavior sanitizers | Passed the same 8 cases and 8583 assertions with `-O1 -g -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer`; no sanitizer findings. This applies only to the model/scheduler tests, not the unavailable runtime build. |
| Numerical export: `ACCELSIM_EXPORT_DIR=/tmp/accelSim-validation /tmp/accelSim_model_test '[accelSim-export]'` | Passed: 1 case, 4 assertions; exported 360000 samples and 214 resolved modes outside the repository. Production acquisition remains streaming and does not use this export. |
| Offline Python validation | Passed finite-sample, time-grid, and raw-count checks; wrote `/tmp/accelSim-validation/validation.json` and `psd.csv`. Acceleration RMS X/Y = 0.03998387 / 0.03964176 m/s²; stationary expectation = 0.03979352 m/s². Zero rail samples. Measured/theoretical band-power ratios: 100–200 Hz = 1.0187/1.0086; 200–400 Hz = 1.0136/0.9836; 400–900 Hz = 0.9922/1.0030; 1200–2800 Hz = 0.9981/1.0016. No short-record high-r RMS pass/fail criterion was imposed. |
| Python syntax | Passed with bytecode redirected to `/tmp/accelsim-pycache`. |
| Source interface comparison | Passed: all 23 `timingDiag` names and ordering match mcp3208Ctrl. This is a source comparison, not a live INDI enumeration. |
| Formatting and diff | clang-format 23.1.3 applied to all four new C++ files; `git diff --check` passed. The specification prefix SHA256 remains `b9b164ac19afe92e1dc7236b0c5529e8c5e86763a3b0f791875fad2f08db7a8c`. |

Formatting/analysis tools were installed, with network approval, only in `/tmp/accelsim-validation-env` (clang-format 23.1.3, NumPy 2.0.2, SciPy 1.13.1). No project dependency files or installed instrument software were changed.

**Coverage still requiring a configured non-instrument Linux host:** T1 full application build; T6 real stream creation, metadata, rings, and notifications; T7 app/callback tests and runtime INDI comparison; T8 actual trigger lifecycle and unchanged DDSPC consumer substitution; T9 ten-minute rate, memory, and timing acceptance. The included stream tests cover two circular depths, fresh acquisition timestamps, counters and per-plane metadata, channel order at the app copy boundary, producer restart, source counter gaps, duplicate notifications, source timeout/recreation, and semaphore release. Their presence is not evidence that they passed on this workstation. Full INDI wire serialization comparison against a hardware-free mcp3208Ctrl process, exhaustive wait-interruption/fault injection, and deployed consumer substitution remain follow-up validation, not completed checks.

**Plan deviations/limitations:** No physics, interface, or scope changes beyond the approved answers. Added the `ACCELSIM_MODEL_ONLY` compile path within the planned test file to obtain numerical evidence despite unavailable runtime dependencies. The Linux runtime/performance test work could not be executed; the app and framework-dependent tests therefore remain unverified by compilation in this environment. No new user decision is currently required to complete a Linux build/test attempt. Runtime compatibility cannot be declared accepted until those checks and the operator-supplied deployment/consumer configuration are exercised.

Commands for normal builds, standalone model tests, isolated local launch, stream checks, the ten-minute benchmark, RSS observation, shutdown summaries, and offline PSD validation are maintained in `apps/accelSim/README.md`. Do not run any of those runtime checks through an instrument tunnel or on instrument computers as an agent.

### Post-implementation Review — 2026-10-26

This review covers the complete tracked diff and all eight untracked files under `apps/accelSim`, against AGENTS.md, this specification and its ANSWER entries, the active notebook cells, mcp3208Ctrl, and frameGrabber. Only this planning document was edited during the review. No application behavior, dependencies, shared infrastructure, or deployment files were changed; no commit, push, or instrument testing occurred. The findings below qualify the preceding execution record: the implementation is **not yet accepted**, independently of the workstation build blockers.

**Findings at review time, in severity order:** All four were subsequently corrected in the Synchronization Fix Record below. Full Linux integration acceptance remains outstanding.

1. **P1 — Ordinary synchronization pauses can omit elapsed model time.** In `apps/accelSim/accelSim.hpp`, `waitSync()` derives skipped ticks from the producer counter, but consults elapsed time only when `elapsedTicks > maxCatchUp + 1`. At 6000 Hz, consider successive source counters 10 and 11 separated by four nominal periods (about 666.7 us). The counter-derived skip count is zero, the elapsed-time branch is bypassed, and `advance(0)` generates only one numerical tick. Three elapsed nominal ticks and their skipped publication opportunities are omitted. This violates Q4's elapsed-time recovery policy, even with the nominal source rate configured correctly. Correction needs a persistent nominal-time accounting scheme that covers ordinary pauses as well as long stalls without double-counting counter gaps or delayed/queued notifications. Add deterministic tests for a short pause with consecutive counters, matching counter/time gaps, and jitter/backlog recovery. The existing source-gap test covers a counter jump, not this case.
2. **P2 — Interrupted phase waits lose skipped-publication accounting.** `waitSync()` commits `m_lastProducer` and updates source-gap diagnostics before `waitUntil()` finishes. If an FPS update interrupts that phase wait, `acquireAndCheckValid()` returns without accumulating its local `skipped` count. The next acquisition sees the already-consumed counter, so the skipped frames and the canceled current publication never reach `sim_diag.skipped_publications`. For example, accepting counter 14 after 10, then changing FPS during the phase wait, loses three older opportunities plus the canceled current one from that total. The approved stationary restart on FPS change is correct; the accounting around the aborted acquisition is incomplete. Test a deterministic control interruption after counter acceptance, including a preceding counter gap, and ensure each omitted opportunity is counted once without publishing stale data.
3. **P2 — Synchronization status can remain locked after the source disappears.** Worker diagnostics are copied to the app-loop snapshot only in `frameGrabberPostPublish()`. Timeout/stale-source handling and `reconfig()` neither invalidate the published lock state nor transfer their updated diagnostics. After a previously locked source vanishes, the app loop can continue publishing the old `delay_lock=1` and cadence indefinitely while configuration retries. This contradicts the requirement to report synchronization honestly. Clear invalid synchronization status and transfer an outage/reconfiguration snapshot independently of successful frame publication, preserving worker ownership and bounded locking. The existing vanished-source test checks reconfiguration but not the externally visible snapshot; extend it to check loss of lock and recovery.
4. **P3 — Exact mutex-scope comment style.** Three lifetime-only blocks in `accelSim.hpp` use `{ // mutex scope` rather than AGENTS.md's required `{ //mutex scope`. The files pass the current clang-format check, which does not by itself establish compliance with that explicit documentation rule. No formatting-only source changes were made in this review.

These are source-review findings, not claims of successful Linux runtime reproduction. The numerical tests do not exercise the app-level paths above. No new physics, interface, or recovery policy was substituted during this review; Q3/Q4 and the existing diagnostic requirements remain authoritative for corrections.

**Other reviewed areas:**

- The AR(2) implementation is consistent with the approved stationary model. With `d=1-r^2` and `B=d^2+4*r^2*sin(omega)^2`, it uses innovation variance `s^2*d*B/(1+r^2)`, correlation `rho=phi1/(1+r^2)`, and conditional scale `eta=sqrt(B)/(1+r^2)`. Drawing `(s*z1, s*(rho*z1+eta*z2))` initializes adjacent histories with variance `s^2` and covariance `s^2*rho`. The factorized calculation avoids subtracting nearly equal unit-sized terms for high-r modes. Independent axis histories/draws, the 14 prominent parameters, 200-mode parameter distributions/envelope, lever-arm conversion, one-sided noise scaling, and separate WFS sums agree with the approved notebook adaptation. Ensemble stationarity deliberately replaces finite-record normalization; NumPy sequence equality is not promised.
- ADC conversion applies noise first, then calibration, `std::round`, floating-point clipping, and UINT16 conversion. Nonfinite values are rejected before casting. Clip counters count actual clamps after rounding, including generated catch-up ticks. No conversion is required in the production consumer.
- The timer uses an epoch and accumulated fractional periods, publishes only the current pair, bounds catch-up at eight skipped ticks, and reports stationary restarts for longer gaps. Both acquisition paths assign a realtime acquisition stamp immediately before numerical generation; pacing uses monotonic time. Real-time scheduling performance and wall-clock adjustment behavior still require runtime validation.
- frameGrabber retains exclusive output ownership: width 2, height 1, UINT16, X then Y, three axes `[2,1,depth]`, circular planes, acquisition/write timestamps, counters, and semaphore posts. The app only maps the optional input source directly. No DM stream or additional output stream is present.
- The app-owned numerical/RNG/source state stays on the acquisition thread after setup; control requests use atomics; snapshots and app-owned callback/property updates have separate mutexes. INDI property/element access also uses the existing library's internal read/write locks. No additional app-owned data race was identified by inspection, but the model-only sanitizer run does not validate framework concurrency. Existing framework lifecycle flags and timing-buffer concurrency were not refactored.
- Relevant INDI names/ranges/elements, all 23 timingDiag elements, external subscriptions, fixed two-channel rejection, FPS validation/boundary restart, synchronization configuration names, phase/budget equations, and source semaphore release/reconnect paths were inspected. Full wire compatibility, shutdown during waits, and reconfiguration behavior remain unverified at runtime. No accidental feature/deployment scope expansion was found.

**Build blocker 1: exact origin of `timegm`.**

The normal build fails with `libMagAOX/file/stdFileName.hpp:366:18: error: use of undeclared identifier 'timegm'`. The include chain is:

```text
libMagAOX/app/MagAOXApp.cpp
  -> app/MagAOXApp.hpp
  -> logger/logManager.hpp
  -> logger/generated/logTypes.hpp
  -> logger/types/git_state.hpp
  -> logger/types/flatbuffer_log.hpp
  -> logger/logMeta.hpp
  -> logger/logMap.hpp
  -> file/stdFileName.hpp:366 (timegm)
```

`Make/common.mk` imports `pkg-config --cflags mxlib`. This workstation's `/opt/homebrew/lib/pkgconfig/mxlib.pc` supplies `-D_XOPEN_SOURCE=700`. Apple's SDK `sys/cdefs.h:853–855` consequently defines `_POSIX_C_SOURCE=200809L`; `_time.h:133–142` hides `timegm` unless `_DARWIN_C_SOURCE` is also enabled. A standalone file including only `<time.h>` and calling `timegm` compiles with ordinary flags, fails with `-D_XOPEN_SOURCE=700`, and compiles with both `-D_XOPEN_SOURCE=700 -D_DARWIN_C_SOURCE`. This demonstrates a macOS feature-test-macro/environment incompatibility in existing shared code, independent of accelSim and independent of whether a time header is explicitly included.

Diagnostic command `make -C apps/accelSim CPPFLAGS=-D_DARWIN_C_SOURCE` removes this error from the failing shared object compilation, but still fails at the missing mxlib header. This is an invocation-only diagnostic, not a verified complete macOS build remedy: the shared precompiled-header rule does not consume CPPFLAGS. A durable macOS fix must apply the platform feature policy consistently to the shared library/PCH and dependencies, not add a declaration/shim in accelSim. No such shared policy change was made. Recommended acceptance environment remains the configured non-instrument Linux development host; Linux still needs the matching dependency described next.

**Build blocker 2: exact origin of `mx/math/floatUtils.hpp`.**

The fatal error is `libMagAOX/logger/types/telem_pupilfit.hpp:8:10: fatal error: 'mx/math/floatUtils.hpp' file not found`. Its include chain is:

```text
libMagAOX/app/MagAOXApp.cpp
  -> app/MagAOXApp.hpp
  -> logger/logManager.hpp
  -> logger/generated/logTypes.hpp:47
  -> logger/types/telem_pupilfit.hpp:8
  -> mx/math/floatUtils.hpp (absent)
```

The header provides the `mx::math::isFinite` API used by existing pupil-fit telemetry validation at lines 100–101. The shared telemetry file and this dependency were introduced by existing commit `bee71c67edc450ad5dd4796dc80e3a6d5663296c` (subsequently formatted in `4bed3178`), not by the accelSim diff. `pkg-config --modversion mxlib` reports **0.2.0**. Installed `mxlib_uncomp_version.h` identifies revision **3cd28b6e53765079859407b0d47ef5a3d7a41430**, branch dev, built from `/Users/parkerjohnson/source/mxlib`. Neither the installed include tree nor that clean local checkout contains `floatUtils.hpp` or `mx::math::isFinite`; the locally available history has no entry for `include/math/floatUtils.hpp`.

This is a pre-existing MagAO-X/mxlib compatibility mismatch exposed by the installed dependency, not an accelSim include or a generic macOS-only missing header. A Linux host with the same mxlib revision would also lack the header. The required environment must provide a compatible mxlib header/API and matching build configuration, or the existing shared telemetry dependency must be corrected separately by its owners. Local evidence does not identify a release/revision known to provide that API, so an unverified “upgrade to version X” is not prescribed. Do not stub the header, suppress common log types, remove telemeter, or change shared telemetry to make accelSim appear to compile. None would be an app-local repair of the dependency.

The full test translation unit reaches both errors through `accelSim_test.cpp -> accelSim.hpp -> libMagAOX.hpp -> MagAOXApp.hpp`, then the same shared include chains. The normal app build fails in `app/MagAOXApp.o` before compiling accelSim itself. Neither blocker admits an accelSim C++ source fix; no permanent build workaround was added to the app Makefile.

**Checks rerun during this review:**

| Check | Result |
| --- | --- |
| `make -C apps/accelSim` | Failed, exit 2, both shared errors above. Full log: `/tmp/accelSim-review-app.log`. |
| `make -C tests -f Makefile.one t=../apps/accelSim/tests/accelSim_test.cpp` | Failed, exit 2, same shared errors. Full log: `/tmp/accelSim-review-tests.log`. |
| `make -C apps/accelSim CPPFLAGS=-D_DARWIN_C_SOURCE` | Diagnostic only: `timegm` error removed from shared object compilation; missing `floatUtils.hpp` remains, exit 2. Log: `/tmp/accelSim-review-darwin-probe.log`. |
| Independent `<time.h>` compiler probes | Default: pass; `_XOPEN_SOURCE=700`: fail; adding `_DARWIN_C_SOURCE`: pass. Source: `/tmp/accelSim-timegm-probe.cpp`. |
| Fresh standalone model build, C++20/O3/`-fno-fast-math`/warnings | Passed, then **8 cases / 8583 assertions** passed in `/tmp/accelSim_review_model_test '[model]'`. |
| Fresh standalone ASan/UBSan build, O1/debug/`-fno-fast-math` | Passed, then **8 cases / 8583 assertions**, no sanitizer reports, in `/tmp/accelSim_review_sanitize '[model]'`. |
| Fresh numerical export and offline PSD analysis | Export: **1 case / 4 assertions**, 360000 samples and 214 modes. Python finite/time-grid/count checks passed; measured RMS and band ratios reproduce the earlier execution record. Outputs in `/tmp/accelSim-review-validation`. Spectral ratios are reported measurements, not an additional automated tolerance test. |
| Python bytecode syntax check | Passed with cache outside the repository. |
| clang-format 23.1.3 `--dry-run --Werror` on all four C++ files | Passed; exact mutex-comment issue above remains a separate AGENTS.md check. |
| `git diff --check`, expected-file audit, protected text | Passed. All changed/untracked paths match the approved file table. Specification-prefix SHA256 and all ANSWER entries remain unchanged. |

**Git scope:** the four tracked modified files are this plan, `tests/Makefile.one`, `tests/groups.dox`, and `tests/tests.list`. The eight untracked app files are `Makefile`, `README.md`, `accelSim.cpp`, `accelSim.hpp`, `accelSimModel.hpp`, `config/example.conf`, `tests/accelSim_test.cpp`, and `tests/validate_psd.py`, all under `apps/accelSim/`. Normal `git diff --stat` excludes those untracked files; they were inspected separately. No unexpected modified or untracked file was introduced by review/testing.

**Remaining Linux validation:** first resolve the shared dependency mismatch and build the app and full tests with the normal commands below. Correct and regression-test the synchronization findings before compatibility acceptance. Run app/configuration/INDI callbacks, isolated stream/ring/timestamp/semaphore tests, source counter reset/gap/duplicate/outage/recreation cases, wait interruption and FPS changes during phase delay, shutdown/reconfiguration, and the ten-minute 6000 Hz benchmark with producer diagnostics and RSS. Compare the real INDI wire interface and run the unchanged raw-ADC consumer when its operator-supplied configuration is available. No Linux integration, race-detection, INDI runtime, or sustained-performance pass is claimed here.

```sh
pkg-config --modversion mxlib
pkg-config --cflags mxlib
make -C apps/accelSim
make -C tests -f Makefile.one t=../apps/accelSim/tests/accelSim_test.cpp
./apps/accelSim/tests/accelSim_test '[accelSim]'
./apps/accelSim/tests/accelSim_test '[accelSim-streams]'
```

Use the README's isolated local launch and `[accelSim-live]` commands for performance testing on that non-instrument host. The source-reviewed bugs were open at the end of that review; their subsequent fixes and validation follow below. Runtime compilation remains blocked, and the earlier statement that no new implementation decision was required did not constitute approval of these defects.

### Synchronization Fix Record — 2026-10-26

The user authorized correction of the three functional findings and the three mutex-scope comments. **All four findings are corrected in source.** Changes in this turn are limited to `apps/accelSim/accelSim.hpp`, `accelSimModel.hpp`, `tests/accelSim_test.cpp`, `README.md`, and this plan below the agent heading. No shared MagAO-X code, dependency, build workaround, instrument configuration, or new feature was added. The earlier test-registration changes remain untouched. No commits, pushes, or instrument-side testing occurred.

**1. Elapsed nominal ticks and source progress:**

- An app-local `syncClock` in the existing numerical/timing header retains a monotonic epoch, cumulative source-counter progress, the next unaccounted publication opportunity, and the next ungenerated numerical tick. These last two positions are separate because cancellation omits a publication without advancing the numerical model.
- For each unique notification, `prepare()` computes the latest opportunity as the maximum of elapsed nominal time and cumulative source progress, rather than adding their gaps or consulting elapsed time only for long stalls. Rounding follows the existing fractional-period timer convention. With counters 10 then 11 separated by four nominal periods, the result is three skipped publications and four model steps before one current pair is published.
- Keeping source progress independent of elapsed-time recovery prevents queued source notifications from counting a previously covered interval again. A repeated counter, or a new counter whose nominal opportunity was already accounted for, produces no duplicate sample. A later new opportunity resumes ordinary stepping. No per-trigger coefficient retuning, new output stream, or catch-up publication burst is introduced.
- The existing eight-skipped-tick numerical catch-up bound remains unchanged. Larger model advances restart in stationarity. Rate changes re-anchor the nominal clock at the next new source event while retaining consumed-counter history and cumulative diagnostic totals; counter gaps across that boundary still count. Producer counter resets establish a new epoch and retain the existing stationary-restart behavior.

**2. Phase-wait cancellation and accounting:**

- Preparing a trigger does not commit its source counter or publication ledger. The app retains a pending notification through the phase wait and model acquisition.
- Successful acquisition finalizes the pending notification once. Cancellation instead finalizes it as omitted: older missed opportunities **plus the canceled current opportunity** are added to `skipped_publications`. Source gap/counter diagnostics are committed at this same finalization point. A pending flag prevents cleanup or reconfiguration from finalizing the same notification again.
- Semaphore consumption is now accounted for even when a control change arrives during that wait. FPS/reconfiguration/shutdown checks after the phase wait also use cancellation cleanup. No canceled sample is marked fresh, copied, or published. The next FPS change keeps the approved stationary restart; recovery without a rate change advances canceled-but-ungenerated model ticks without counting those publications twice.
- A failed phase sleep requests the existing reconfiguration lifecycle after cancellation. Output stream ownership and publication remain entirely with frameGrabber.

**3. Source loss and diagnostic delivery:**

- The existing app-local diagnostic data and timing index enum were moved into `accelSimModel.hpp` so the production invalidation/accounting logic can be tested without unavailable framework dependencies. No INDI names or ordering changed.
- `invalidateSync()` finalizes any canceled notification, clears lock, measured cadence, delay/model/budget/service/latency/trigger measurements and the instantaneous producer delta, resets timing history, and invalidates the fresh-frame flag. It retains the configured delay target, source/local frame accounting, skipped/discontinuity totals, and clipping totals.
- Timeout, including a still-mapped but stalled source, stale/missing input, phase cancellation, and reconfiguration transfer an invalidated snapshot independently of frame publication. The normal app loop can therefore send unlocked status while the source is unavailable. Reconnection cannot establish lock from an old period measurement.
- Ordinary frame snapshots still use `try_lock`. Lifecycle/outage snapshots require delivery and wait only for the existing app-loop snapshot copy. That mutex never covers numerical stepping, sleep, INDI work, or logging. Startup also transfers its reset status. Source-loss detection retains the existing one-second semaphore timeout; invalidation is delivered when that timeout/reconfiguration is detected, not instantaneously on remote producer failure.

**4. AGENTS.md comments:** all three lifetime-only scopes now use `{ //mutex scope`. clang-format was run with the repository style plus the command-line override `CommentPragmas: "^mutex scope$"` to preserve the explicitly required spelling. The shared `.clang-format` file was not changed.

**Focused regression coverage:**

| Regression | Coverage and execution status |
| --- | --- |
| `accelSim synchronization accounts elapsed nominal ticks` | Executed: consecutive counters after a four-period pause produce the same physical sample as uninterrupted numerical stepping; overlapping counter/time gaps count once; queued/repeated notifications do not replay; 6000 alternating-jitter arrivals do not accumulate extra misses; a long pause still uses bounded stationary recovery. |
| `accelSim synchronization cancellation retains accounting` | Executed: preparation leaves committed state unchanged; canceling counter 14 after 10 counts four omissions and three source drops; duplicates do not recount; canceled numerical ticks are advanced on recovery; FPS changes preserve consumed counters/totals; repeated cancellations and producer counter resets retain consistent accounting. |
| `accelSim synchronization diagnostics invalidate stale lock` | Executed: every stale synchronization measurement is cleared, configured delay retained, counters/clipping preserved, and repeated invalidation is idempotent. |
| `accelSim outage and reconfiguration deliver unlocked diagnostics` | Added but blocked by framework build dependencies: checks the actual app snapshot without another published frame, pending-cancellation finalization only once, reconfiguration delivery, and fresh timing measurements before relocking. |
| Extended `accelSim real local streams` | Added but blocked by framework build dependencies: a consumed synthetic source semaphore followed by cancellation at phase-wait entry retains the exact source gap and four omitted opportunities; duplicate notifications do not recount; stale ADC copy is rejected. Destroying a previously locked source clears the actual snapshot after timeout and through reconfiguration/reconnection. The cancellation fixture uses a fresh clock epoch to avoid dependence on earlier OS scheduling delays. |

The three dependency-free regressions exercise the same production helper methods called by the app. They do not substitute for compiling/executing the app integration paths. Full-path wait interruption, actual semaphore behavior, INDI serialization, and runtime concurrency still require the non-instrument Linux development environment.

**Validation rerun after the fixes:**

| Check | Result |
| --- | --- |
| Fresh optimized standalone build, C++20/O3/`-fno-fast-math`/`-Wall -Wextra` | Passed. `/tmp/accelSim_fix_model_test '[model]'`: **11 cases, 20721 assertions passed**. Includes all eight prior numerical cases and the three new regressions. |
| Explicit new-regression selection | `/tmp/accelSim_fix_model_test '[sync-regression]'`: **3 cases, 12138 assertions passed**. |
| Fresh ASan/UBSan standalone build, O1/debug/`-fno-fast-math` | `/tmp/accelSim_fix_sanitize '[model]'`: **11 cases, 20721 assertions passed**, no sanitizer findings. Applies to numerical/timing/status helpers, not the blocked full app. |
| Numerical export | **1 case, 4 assertions passed**; 360000 samples and 214 modes in `/tmp/accelSim-fix-validation`. Both CSV files are byte-for-byte identical to the preceding review export, confirming unchanged vibration/ADC numerical behavior. |
| Offline Python analysis and syntax check | Passed finite/time-grid/count checks and syntax compilation. RMS and spectral measurements reproduce the earlier record; spectral ratios remain reported measurements, not new tolerance assertions. |
| Normal app build | `make -C apps/accelSim` failed, exit 2, with the unchanged shared `timegm` and missing `mx/math/floatUtils.hpp` errors. Log: `/tmp/accelSim-fix-app.log`. No workaround applied. |
| Normal full test build | `make -C tests -f Makefile.one t=../apps/accelSim/tests/accelSim_test.cpp` failed, exit 2, with the same shared errors. Log: `/tmp/accelSim-fix-tests.log`. |
| Formatting / diff / scope | clang-format applied and `--dry-run --Werror` passed using the comment-preservation override; `git diff --check` passed; all three exact mutex comments verified. Only the five authorized files above changed in this turn. Specification-prefix SHA256 and all ANSWER entries remain unchanged. |

Standalone reproduction commands, from the repository root:

```sh
c++ -std=c++20 -O3 -fno-fast-math -Wall -Wextra \
  -DCATCH_CONFIG_MAIN -DACCELSIM_MODEL_ONLY \
  apps/accelSim/tests/accelSim_test.cpp -o /tmp/accelSim_fix_model_test
/tmp/accelSim_fix_model_test '[model]'
/tmp/accelSim_fix_model_test '[sync-regression]'
c++ -std=c++20 -O1 -g -fno-fast-math -fsanitize=address,undefined \
  -fno-omit-frame-pointer -DCATCH_CONFIG_MAIN -DACCELSIM_MODEL_ONLY \
  apps/accelSim/tests/accelSim_test.cpp -o /tmp/accelSim_fix_sanitize
/tmp/accelSim_fix_sanitize '[model]'
```

**Remaining status:** no known implementation defect from the post-implementation review remains open, and no additional defect was found in this fix pass. This is not a claim of full runtime acceptance: the app and framework-dependent regressions remain uncompiled here, and Linux stream/INDI/lifecycle/concurrency checks, unchanged-consumer substitution, and the ten-minute performance test are still required. Both existing environment/dependency blockers remain documented above and unchanged. There was no scope expansion beyond the requested fixes and their tests/documentation.

### Offline Visual Validation — 2026-10-26

After committing the initial implementation (`eb3c2c91`), the user requested an offline visual inspection workflow using the same C++ numerical model, without changing production physics. This addition touches only `apps/accelSim/tests/accelSim_test.cpp`, `tests/validate_psd.py`, the new `tests/test_validate_psd.py`, the app README, and this planning record. Production `accelSim.hpp`, `accelSimModel.hpp`, Makefiles, shared infrastructure, and the reference notebook are unchanged. No commit, push, instrument access, or real-time plotting was performed.

**Export and analysis:**

- The existing `[accelSim-export]` test still calls production `model::step()` for a finite diagnostic record: fs=6000 Hz, duration=60 s, seed=42, modeSeed=123. It writes 360000 simultaneous X/Y pairs containing time, physical and WFS-coupled pointing in arcsec, acceleration in m/s², and raw ADC counts, plus the 214-mode parameter table. The duration is explicit in the export test; no runtime playback dependency or production model setting changed.
- Added `metadata.json` with sample rate, duration, both seeds, lever arm, empirical ADC calibration, sensor noise PSD, and actual lower/upper model clipping counters for X/Y. Files are flushed and checked. Rail sample counts remain distinct from clamp events.
- The Python analyzer reads the export metadata, preserving analysis-only override flags and legacy-default support. It prints and writes full-record physical pointing/WFS pointing RMS in mas, physical/ADC-derived acceleration RMS in m/s², exact C++ clamp counts, and separate rail counts. RMS is `sqrt(mean(x²))`, not mean-subtracted standard deviation. The inverse ADC calibration is analysis-only.
- Five saved figures: `pointing_timeseries.png`, `pointing_psd.png`, `wfs_psd.png`, `accel_psd.png`, and `adc_psd.png`. Each has separate X/Y panels; every PSD marks all 14 prominent frequencies. The ADC figure includes the same pre-ADC realization for direct comparison. Defaults show the first second in the time trace; `--time-start` and `--time-seconds` change only that view. `--show` displays the figures after saving and printing results; default plotting uses Agg without a GUI.
- Welch matches the notebook: SciPy `welch`, Hann window, 10*fs samples per segment (capped by record length), 50% overlap, constant detrending, and density scaling. PSDs use mas²/Hz or (m/s²)²/Hz as labeled, and are also exported to `psd.csv`.

**Theory overlay:** the analytic stationary AR(2) lag covariance is evaluated from each resolved mode's frequency, radius, and RMS. The lag covariance uses `r^k * [cos(k*omega) + (1-r²)/(1+r²) * cot(omega) * sin(k*omega)]` times the relevant variance. Pointing conversion, squared WFS couplings, and independent acceleration-noise covariance are applied separately. Window autocorrelation and the exact segment-mean projection produce the ensemble expectation of the same finite Hann/detrended estimator. This avoids showing an infinite-record narrow peak against a finite-resolution Welch estimate. The high-r realization amplitudes still need not match the ensemble expectation in a short record. ADC overlays are explicitly pre-ADC expectations, without pretending that linear AR theory predicts nonlinear clipping/quantization.

**Validation performed locally on macOS:**

| Check | Result |
| --- | --- |
| Fresh model-only C++20/O3/`-fno-fast-math` build | Passed, independently of the known MagAO-X framework build blockers. |
| Existing numerical/timing suite | **11 cases, 20721 assertions passed.** |
| Updated diagnostic export | **1 case, 6 assertions passed**; default record and metadata generated in `/tmp/accelSim-visual-validation`. |
| New Python analysis tests | **4 tests passed**, also with warnings treated as errors: known white-noise density/DC/Nyquist/mean-removal limits; independent dense covariance quadratic-form comparison for odd/even windows and r=0.9/0.9999999; physical units, lag-one covariance, distinct X/Y couplings and sensor noise; RMS retaining the record mean. |
| Full analyzer | Passed finite/time-grid/raw-count checks and produced all five PNGs, `psd.csv`, and `validation.json`; all five figures were visually inspected. Interactive `show()` dispatch was verified with a mocked GUI under Agg; actual interactive desktop windows were not opened. |
| Formatting / syntax / scope | clang-format applied and checked on the touched C++ test file; Python syntax and `git diff --check` passed. Production app/model files have no diff. |

Recorded full-record RMS X/Y: physical pointing **2.168665 / 2.428181 mas**; WFS-coupled pointing **1.372940 / 2.049623 mas**; acceleration **0.039983882 / 0.039641765 m/s²**; ADC-derived acceleration **0.040061091 / 0.039713301 m/s²**. Both lower and upper clipping counts are **[0,0]**, and there are no rail samples. Acceleration band-power ratios remain within about 2% of ensemble theory in the recorded broad bands; no high-r finite-record amplitude acceptance threshold was imposed.

Matplotlib 3.9.4 and its dependencies were installed with network approval only into the temporary `/tmp/accelsim-validation-env`, alongside NumPy 2.0.2 and SciPy 1.13.1. No repository dependency or instrument software was changed. The README contains complete macOS build/export/environment/plot commands and plot interpretation. All generated data and figures stay outside the repository. Full MagAO-X integration and sustained real-time validation remain separate pending tasks, unaffected by this offline workflow.

### ImageStreamIO / INDI Compatibility Review — 2026-10-26

Reviewed committed implementation `1df0135f` on local macOS, after reading AGENTS.md and the approved specification/answers. This is a source/interface review with local dependency-free checks, **not runtime integration acceptance**. The working tree was clean at the start. Only this planning record is changed; no application, numerical model, test, dependency, shared build file, deployment configuration, or instrument was changed. No commit or push was performed. The findings below supersede the earlier statement of no known defects: four additional INDI diagnostic/control compatibility findings remain open.

**Stream comparison:**

| Item | mcp3208Ctrl | accelSim / assessment |
| --- | --- | --- |
| Output name | Inherits frameGrabber's empty default, resolved to the instance/config name unless `framegrabber.shmimName` is supplied. The deployed configuration is absent here. | Constructor and example config explicitly default to `acc000`; intentional requirement. Hardware source code alone does not establish that its default executable invocation publishes `acc000`. Match the deployed instance/config separately. |
| Datatype / sample | `_DATATYPE_UINT16`; two channels by default; ADC channels read in ascending order. | `_DATATYPE_UINT16`, exactly two contiguous raw counts, X then Y, each in [0,4095]. Software establishes hardware channel order; physical X/Y wiring is the user-provided contract. |
| Dimensions / depth | Width `m_numChannels`, height 1; frameGrabber creates `[width,1,D]`, three axes even at D=1. | `[2,1,D]`; `framegrabber.circBuffLength` default 1, same minimum-depth handling. Fixed two-channel validation is explicitly approved in Q2. Match deployed D when available. |
| Allocation / semaphores | Shared frameGrabber implementation, CPU location -1, shared memory, zero keywords, `IMAGE_NB_SEMAPHORE`, `CIRCULAR_BUFFER | ZAXIS_TEMPORAL`. | Identical implementation and flags; no app-specific output creation or semaphore posting. |
| Publication transaction | `write=1`; copy sample; real-time `writetime`; acquisition `atime`; `cnt1=plane`; increment `cnt0`; write per-plane metadata; `write=0`; `ImageStreamIO_sempost(image,-1)`. | Same frameGrabber code, including ordering. Post-publication hook only transfers diagnostics. Failed/invalid acquisition does not enter this transaction. |
| Ring / counters | New stream starts `cnt1=D-1`; first write uses plane 0, then wraps. `cnt0` counts publications. Per-plane `cntarray`, `atimearray`, and `writetimearray` follow each write. | Identical. Missed simulated ticks do not artificially increment stream counters; they appear in `sim_diag`. Counters can reset on stream recreation/producer replacement. |
| Acquisition timestamp | Sync mode stamps immediately before ADC reads; timer mode omits assignment to `m_currImageTimestamp`. | Both modes stamp immediately before model acquisition, including catch-up computation, using `CLOCK_REALTIME`. This deliberate, approved difference fixes the simulator contract rather than reproducing zero timer acquisition timestamps. |
| Write timestamp / lifecycle | Framework captures publication time and owns stream destruction/recreation. | Identical. Pacing uses monotonic time; epoch timestamps can change with wall-clock corrections. |

The shared write path is `libMagAOX/app/dev/frameGrabber.hpp:945–1050`; creation precedes it in the same acquisition thread. No incompatible app-specific counter, plane, layout, timestamp-copy, or semaphore transaction was found. This conclusion is source-based: real stream attachment and metadata checks could not run here.

**INDI inventory and differences:**

Both applications use `MagAOXApp<true>`, frameGrabber, and telemeter, so FSM/logging/thread properties, `fg_shmimName.name`, `fg_frameSize.width/height/depth`, all ten `fg_timing` elements, `telem_rotate.request`, `telem_maxtime.current/target`, and `telem_fgtimings` come from the same implementations. Neither adds a camera acquisition switch. Stream acquisition starts through the normal OPERATING/frameGrabber lifecycle. The INDI identity remains the launch instance (`-n`); the executable names alone are not interchangeable identities.

| Property / interface | Comparison | Classification |
| --- | --- | --- |
| `fps.current/target` | Same RW Number metadata: 0–10000, step 1, `%d`. Hardware initial nominal rate is 2000 Hz; simulator initial nominal rate is 6000 Hz. Simulator rejects invalid/aliased rates and applies accepted changes at an acquisition boundary with stationary restart. | Defaults and validation/boundary application are approved. Premature `Ok` acknowledgment is an open finding below. |
| `numChannels.current/target` | Same RW Number metadata: 0–8, step 1, `%d`. Simulator accepts only 2, sets Alert otherwise, retains shape and accepted values. | Approved Q2 difference; do not copy hardware's unresized channel-count mutation. |
| `alpha.current/target` | Same RW Number metadata: 0–1, step 0.00001, `%.5f`; default 0.01; both clamp finite requests. Simulator additionally rejects nonfinite requests. | Compatible timing-EMA meaning; stricter validation intentional. Different numeric-literal spelling in source has no schema effect. |
| `synchroDelay.current/target` | Same signed integer-microsecond RW Number metadata: ±1000000, step 1, `%d`, label `us`. Simulator validates finite/range and updates at worker consumption; hardware mutates timing state in the callback. | Same requested phase-offset meaning; safe handoff/validation intentional. Diagnostic boundary differences remain below. |
| Valid local control acknowledgments | Hardware uses `indiTargetUpdate(...,true)`, sends Busy, then appLogic settles current/target to Ok. Simulator's callbacks send Ok directly for accepted values. | Immediate synchronous alpha/channel/offset acknowledgments differ in transition sequence. For pending FPS this also falsely implies completion; see finding 2. |
| `timingDiag` | All 23 names, order, RO Number access, units, mode codes, label/group match by source comparison. | Schema matches; atomic delivery, half-period sign, and startup prediction differences below are not approved differences. |
| Diagnostic measurements | Simulation computation replaces SPI duration. Monotonic arrival timing replaces hardware's real-time arrival arithmetic; simulator timer-mode synchronization-only measurements are zero. Hardware timer `read_latency_error_us` can equal minus a configured nonzero offset; simulator keeps it zero. | Honest simulated measurements and timer-mode clearing follow the approved plan. Do not fabricate hardware timing. |
| External subscriptions | Same `fps.device/property/element/tol` and `numChannels.device/property/element/tol`, empty device disables, default elements `current`. Both load but do not use tolerance. Simulator validates incoming values with the same local restrictions. | Subscription names/config are compatible; invalid/missing element now produces Alert/error instead of hardware's log/ignore or unchecked assignment. No separate published `fpsSource` property is expected. |
| Additional diagnostics | Simulator adds nine RO `sim_diag` elements for timing, skips, cadence, discontinuities, clipping. | Approved additive interface. No extra physical-units stream or DM control. |

`synchro.*` configuration names, timing constants, semaphore preference (5), one-second wait timeout, EMA, cadence guard, and absolute/fractional lock criteria match the hardware reference. Hardware-only SPI setup has no public SPI/voltage/calibration INDI property to reproduce. Float-to-double internal control precision is not an outward type change. Simulator-specific configuration and the fixed channel guard do not make an arbitrary unreviewed deployment configuration automatically compatible.

**Open findings, ordered by severity:**

1. **P2 — `timingDiag` is transmitted as intermediate mixed snapshots.** `accelSim::appLogic` (`accelSim.hpp:368`) calls `indi::updateIfChanged` once per field. Each changed field sends the entire property immediately. `mcp3208Ctrl::updateTimingDiagnosticsIndi` uses one `updatesIfChanged` for all fields. On source loss, accelSim can therefore send cleared latency with the preceding `delay_lock=1`, before a later message clears the lock; up to 23 messages can be sent per changed snapshot. The acquisition snapshot itself is coherent and the final message is unlocked, so this is not a persistent failure to invalidate worker state. It is an observable wire-level inconsistency and avoidable INDI traffic. Recommended app-local correction: update/send all 23 values together and assert one coherent message with a recording driver.
2. **P2 — accepted FPS requests report completion before application.** `requestRate` (`accelSim.hpp:795`) updates requested/target and calls `propertyState(...,true)`, which sends Ok. The worker applies that target later; during a source wait, `fps.current` can still contain the prior rate while the property is Ok. Hardware signals Busy on local requests before settling. Q3 permits pending current/target values, not a premature completion indication. Recommended app-local correction: Busy while a validated rate is pending, Ok when active equals requested, Alert on rejection; test the complete message/state sequence including external updates and cancellation. Other immediate controls also skip the hardware Busy transition; preserve or explicitly resolve that transition convention when correcting the control protocol.
3. **P3 — phase-error sign differs at exactly half a source period.** `syncTiming` (`accelSim.hpp:540`) uses `std::remainder`, while hardware maps phase error to `(-T/2,T/2]`. With T=500 us, default timing terms summing to 340.6 us, and postDelay=-250 us, the model delay is 409.4 us and the uncapped applied delay is 159.4 us. Hardware reports +250 us; simulator reports -250 us. Lock uses magnitude, so this does not change the lock verdict or ADC data, but the signed diagnostic convention differs. Recommended app-local correction: identical wrap rule and a boundary test.
4. **P3 — startup trigger predictions become valid one event too early.** Hardware's `updateTriggerTiming` returns before establishing a prediction when no period has been measured, and reports zero interval for the first valid prediction. Simulator `syncTiming` stores a prediction on the first event regardless of period and can report a nonzero interval on the second. With a positive offset it also reports a nonzero `trigger_time_us` on the first event, where hardware reports zero. Recommended app-local correction: establish trigger history only after a positive measured period, and test first/second/third events plus reconnection. This is a diagnostic compatibility difference, not a change to the physical model.

These findings are documented, not patched during this focused review. No production behavior was changed. They do not require a shared-code workaround or a numerical-physics decision. Exact interface acceptance is withheld pending their resolution and runtime tests.

**Synchronization / recovery review:**

- Timer mode uses a monotonic epoch and fractional-period deadlines. `schedule::consume` selects the latest due nominal tick. `model::advance` advances all skipped ticks up to the eight-skipped-tick bound and returns one newest sample; larger gaps restart stationarity and increment discontinuities. There is no loop publishing historical frames. Counter gaps observed by a slow reader are distinct from these producer skips.
- Semaphore mode reconciles elapsed nominal time and cumulative source-counter progress using their maximum, preventing double-counted overlapping gaps. Unique notifications already covered by recovery and repeated counters do not replay samples. This intentionally differs from hardware, which can perform a new ADC read for every queued semaphore notification.
- `finishSync` commits source-counter/drop accounting and canceled-publication accounting exactly once; canceled but ungenerated numerical ticks remain available for recovery. Existing regressions cover short pauses, overlapping gaps, duplicates, cancellation, FPS changes, and producer-counter resets. Nominal physics coefficients are not retuned from arrival jitter.
- Timeout/stale source/reconfiguration invalidates delay lock, cadence, timing history, pending/fresh acquisition state, and forces delivery of a snapshot while preserving totals. The one-second timeout means failure detection is not instantaneous. The earlier persistent stale-lock defect is fixed; finding 1 concerns intermediate INDI messages after detection.
- Source inode replacement is checked on timeout and reconnects through frameGrabber. Hardware additionally estimates a producer period from source metadata, but that estimate is not used by its phase model or published diagnostics in this checkout. Both actual phase estimates use arrival cadence. Actual semaphore wakeups, reconnects, rate changes, and concurrent shutdown remain runtime tests, not validated by source review alone.

**Build blockers reverified without changing shared code or the installed environment:**

1. Normal `make -C apps/accelSim` fails in shared `app/MagAOXApp.o` before compiling accelSim. Error: `libMagAOX/file/stdFileName.hpp:366:18: use of undeclared identifier 'timegm'`. Include chain: `app/MagAOXApp.cpp -> MagAOXApp.hpp -> logger/logManager.hpp -> logger/generated/logTypes.hpp:4 -> types/git_state.hpp -> flatbuffer_log.hpp -> logMeta.hpp -> logMap.hpp -> file/stdFileName.hpp`. `Make/common.mk` imports `pkg-config --cflags mxlib`, which includes `-D_XOPEN_SOURCE=700`. The macOS SDK's `sys/cdefs.h:853–855` consequently selects `_POSIX_C_SOURCE=200809L`; `_time.h:133–142` hides `timegm` unless Darwin extensions are enabled. A fresh standalone `<time.h>` probe passes without that flag, fails with it, and passes with both `_XOPEN_SOURCE=700` and `_DARWIN_C_SOURCE`. Those flags were used only on the temporary probe, not on an application build. This is an existing shared macOS feature-policy/environment conflict, not an accelSim error.
2. Same build also fails at `libMagAOX/logger/types/telem_pupilfit.hpp:8:10: fatal error: 'mx/math/floatUtils.hpp' file not found`. Chain: `MagAOXApp.cpp -> MagAOXApp.hpp -> logger/logManager.hpp -> logger/generated/logTypes.hpp:47 -> types/telem_pupilfit.hpp`. Existing telemetry uses `mx::math::isFinite`; that dependency entered in `bee71c67`, not accelSim. Installed mxlib is version 0.2.0, SHA `3cd28b6e53765079859407b0d47ef5a3d7a41430`, matching the sibling checkout. Neither installed math headers nor that checkout provide the header/API. This is a MagAO-X/mxlib revision/API mismatch, not a simulator problem or necessarily macOS-specific. Linux with the same missing header also fails. A known-compatible dependency revision must be identified by the maintainers; the evidence does not justify prescribing an arbitrary upgrade.
3. The normal full test build reaches the same two errors through `accelSim_test.cpp -> accelSim.hpp -> libMagAOX.hpp -> MagAOXApp.hpp`. An additional isolated INDI-helper probe compiling existing `INDI/libcommon/IndiProperty.cpp` directly in C++20 exposed existing libc++ errors for removed `std::not1`/`std::ptr_fun` at line 194. Compiling only this temporary probe in C++14 succeeded, with existing warnings, and demonstrated finding 1. This is an additional portability issue for rebuilding that shared library with this toolchain, not an observed third error in the normal app build (which stops earlier). No shared standard flags, library sources, or dependencies were changed.

Recommendation: resolve the matching mxlib API/build prerequisites separately with the project lead, and perform integration on a configured **non-instrument Linux development host**. If macOS full-app support is desired, the shared feature-test-macro/PCH policy and old INDI/libc++ compatibility need a separately scoped review. Do not add a simulator shim, fake header, remove telemetry, or suppress shared log types. The full app may encounter additional prerequisites after these first blockers are resolved; this review does not claim a complete macOS repair recipe.

**Checks performed in this review:**

| Check | Result |
| --- | --- |
| Normal app build | Failed, exit 2, the two shared errors above. `/tmp/accelSim-compatibility-app.log`. |
| Normal full test build | Failed, exit 2, same errors. `/tmp/accelSim-compatibility-tests.log`. |
| Fresh optimized standalone C++20 build, warnings, `-fno-fast-math` | Passed; `/tmp/accelSim_compat_model_test '[model]'`: **11 cases / 20721 assertions passed**. |
| Explicit synchronization regressions | **3 cases / 12138 assertions passed**. |
| Fresh ASan/UBSan standalone build/run | **11 cases / 20721 assertions passed**, no sanitizer reports. This covers numerical/timing helpers, not full-app concurrency or RSS. |
| Current-model diagnostic export | **1 case / 6 assertions passed**, 360000 pairs / 214 modes under `/tmp/accelSim-compatibility-validation`. |
| Python PSD-analysis unit tests, warnings as errors | **4 tests passed**. |
| Full offline PSD analyzer | Passed finite/time-grid/count validation and generated all five plots, `psd.csv`, and `validation.json` under `/tmp/accelSim-compatibility-validation`. Zero ADC clipping; numerical results reproduce the preceding offline record. Matplotlib used a temporary cache because the default user cache was not writable. |
| Local recording-driver probe | Existing INDI helper/library code with a fake driver, no socket/server: per-element updates send `(latency=0,lock=1)` then `(0,0)`; batch helper sends only `(0,0)`. Temporary source/binary `/tmp/accelSim-compatibility-indi.cpp` and `/tmp/accelSim-compatibility-indi`. This is helper-level evidence, not a full app test. |
| Schema / edge-case comparison | All 23 timing names/order match; four control schemas have equivalent metadata. Half-period sign difference reproduced numerically; acknowledgment/startup findings confirmed in source. |

No running accelSim executable or `acc000` was produced by the failed builds. No live metadata, frame-counter/plane/semaphore read, achieved 6000-Hz rate, jitter distribution, skipped-publication total, long-stall count, maximum delay, or app memory-stability result is available from this Mac. `shmimLat`/milk command-line tools are not on the local PATH. Offline samples and sanitizer results are not substitutes for those measurements.

**Commands and remaining integration checks:**

From the repository root on the configured non-instrument development host, after resolving shared build prerequisites:

```sh
make -C apps/accelSim
make -C tests -f Makefile.one t=../apps/accelSim/tests/accelSim_test.cpp
./apps/accelSim/tests/accelSim_test '[accelSim]'
./apps/accelSim/tests/accelSim_test '[accelSim-streams]'
```

The stream test creates its own private namespace and checks real frameGrabber metadata/publication, circular planes, timestamps, counters, semaphore notifications, source cancellation/loss/reconnection. It must run in a separate process from the live benchmark. For a separately launched producer, use the private local environment from the README:

```sh
accel_sim_root=$(mktemp -d /tmp/accelSim-local.XXXXXX)
export MAGAOX_PATH="$accel_sim_root"
export MILK_SHM_DIR="$accel_sim_root/shm"
mkdir -p "$MILK_SHM_DIR" "$MAGAOX_PATH/config" "$MAGAOX_PATH/logs" \
  "$MAGAOX_PATH/telem" "$MAGAOX_PATH/sys" "$MAGAOX_PATH/calib" \
  "$MAGAOX_PATH/drivers/fifos"
./apps/accelSim/accelSim -n accelSim-local \
  -c apps/accelSim/config/example.conf --framegrabber.threadPrio=0 \
  >"$accel_sim_root/app.log" 2>&1 &
accel_sim_pid=$!
ACCELSIM_LOCAL_BENCHMARK=1 ./apps/accelSim/tests/accelSim_test '[accelSim-live]'
ps -p "$accel_sim_pid" -o pid,etime,rss,pcpu
```

The benchmark opens `acc000`, checks count range/counter progression, and reports observed intervals and mean publication rate for ten minutes. Type/shape/plane/semaphore assertions are in `[accelSim-streams]`; the live benchmark assumes the running stream has that contract and does not independently reassert its datatype/dimensions. Use the development host's appropriate thread priority/cpuset for performance acceptance; priority 0 is the unprivileged smoke-test setting. Sample RSS repeatedly during the test, not only once afterward. With the existing `shmimLat` utility installed, this additional reader records stream write/arrival times to `times` in the temporary directory:

```sh
(cd "$accel_sim_root" && shmimLat -n acc000)
kill -TERM "$accel_sim_pid"
wait "$accel_sim_pid"
cat "$accel_sim_root/app.log"
```

Remaining acceptance work: compare actual INDI property definitions and valid/invalid request transitions through a private local server; test external subscriptions; verify one producer and operator-supplied deployment identity/depth/config; repeat timer and synthetic semaphore tests including short/long pauses, cancellation, duplicates, replacement, reconnect, FPS change, and shutdown; measure 6000-Hz performance for ten minutes with producer `fg_timing`/`sim_diag`, reader gaps, max lateness/delay, discontinuities, and repeated RSS. The shutdown log supplies producer skip/lateness/discontinuity/clipping totals; live lock/delay/cadence requires the isolated INDI session. Final unchanged raw-ADC consumer substitution remains pending the absent deployment/consumer configuration. No instrument-side or tunneled INDI command was run or requested for agent execution.

Final scope verification: `git diff --check` passes; only this plan is modified. No C++ formatting pass is needed because no code was touched. The protected specification-prefix SHA256 remains `b9b164ac19afe92e1dc7236b0c5529e8c5e86763a3b0f791875fad2f08db7a8c`; all ANSWER entries are unchanged.

### INDI Compatibility Fixes — 2026-10-27

The user authorized correction of the four app-local findings from the preceding review. **All four are now corrected in source.** This supersedes the preceding review's open-finding status, but does not claim full application/runtime acceptance. The prior review's uncommitted planning notes were present at the start of this turn and are retained. The specification above the agent heading and all ANSWER entries remain unchanged. No shared MagAO-X/mxlib files, build settings, dependencies, instrument configuration, simulation physics, stream contract, or DM functionality were changed; no commit or push was performed.

**Production changes (exact files):** `apps/accelSim/accelSim.hpp` and `apps/accelSim/accelSimModel.hpp`. The latter changes only app-local timing diagnostic methods; oscillator coefficients/initialization, mode generation, RNG/noise, physical conversions, ADC conversion, and simulation stepping are unchanged. Other edits are limited to `apps/accelSim/tests/accelSim_test.cpp`, `apps/accelSim/README.md`, and this plan.

1. **One coherent timingDiag update.** `appLogic` still copies the worker snapshot under the snapshot mutex, then calls the extracted app-local `updateINDI` method. That method keeps the existing control mutex around app-owned property updates and calls the existing `indi::updatesIfChanged<double>` once with all 23 timing names/values. There is no per-element send loop. An outage therefore sends the cleared latency/cadence and unlocked state in the same property. Unchanged snapshots send nothing. Extraction permits tests of actual app property publication without starting frameGrabber or telemeter threads.
2. **FPS Busy until worker acknowledgment.** Valid local and subscribed FPS requests increment a request revision and set the target/Busy using the existing INDI helper. The worker captures a consistent requested value/revision under the control mutex only when a request is pending, releases that mutex, applies the already-approved rate change/reinitialization and timing reset, then acknowledges the captured revision. `updateINDI` reports Ok only when the applied and requested revisions match; rejected requests retain Alert until a valid request. Revisions prevent an older in-flight request from acknowledging a newer request, including a request back to the previously active rate. Same-rate requests still pass through worker acknowledgment but do not redraw states or increment discontinuities. The worker performs no INDI calls; numerical work remains outside locks. This adds a short control-state copy at FPS-update boundaries, not a mutex on every normal simulation tick. At those boundaries it can wait for an existing control/INDI critical section; no new real-time guarantee is asserted for reconfiguration.
3. **Positive half-period phase boundary.** `diagnostics::updatePhaseError` applies the hardware's separate positive wraps of applied/model delay, then maps their difference into `(-T/2,T/2]` using `<= -T/2` and `> T/2`. `syncTiming` uses this value for the unchanged absolute/fractional lock criteria. Zero/unmeasured periods produce zero phase error. The 500-us / -250-us-offset case now reports +250 us, matching mcp3208Ctrl.
4. **Trigger history requires measured timing.** `diagnostics::updateTrigger` leaves prediction history uninitialized and trigger time/interval zero while the source period is unmeasured/nonpositive. The first positive-period prediction reports its offset but a zero interval; the next valid prediction can establish an interval. `syncTiming` delegates to this helper. Existing startup, outage, counter-reset, rate-change, and reconfiguration paths continue to clear prediction history, so reconnection cannot use an old prediction. Neither the model phase target nor acquisition pacing was changed to implement this diagnostic correction.

**Focused regression tests added:**

| Test | Specific coverage | Execution status |
| --- | --- | --- |
| `accelSim timing diagnostics serialize one snapshot` (`[indi-regression]`) | Captures actual outgoing INDI XML. An invalidated locked snapshot sends exactly one 23-element timingDiag message with latency/cadence/lock cleared together and retained lifetime frame count. An identical snapshot emits no message. | Added; requires the blocked full application dependencies. |
| `accelSim FPS acknowledgment follows worker application` (`[indi-regression]`) | Actual local callback sends Busy with old current; app-loop update remains Busy; real acquisition boundary applies rate and permits Ok/current update. Covers superseding revisions, a request back to the active rate without a physical restart, persistent Alert after rejection, and a valid external subscription recovering through Busy to Ok. | Added; requires the blocked full application dependencies. |
| `accelSim phase error positive boundary` (`[model][sync-regression]`) | Both exact half-period signs, multiple-period equivalents, values immediately on both sides, zero difference, unmeasured period, and the review's concrete 409.4/159.4-us case. | Executed successfully. |
| `accelSim trigger history requires measured timing` (`[model][sync-regression]`) | Repeated no-period arrivals with positive configured offset; first/second valid predictions; invalidation/reconnection with cleared history. | Executed successfully. |

The existing full-app `accelSim synchronized delay model` test was also extended to check the first, second, and third actual `syncTiming` calls and the positive half-period case. The existing outage/reconfiguration tests continue to cover application lifecycle invalidation. New full-app tests use a recording subclass of the existing INDI driver, overriding only XML transport and parsing the real serialized messages. It has a private temporary input file, unused `/dev/null` endpoints, no activated input thread/server/socket, and explicit input cleanup following the repository's Linux `/proc/self/fd` fixture pattern. This fixture is for the Linux full test suite, not a macOS build workaround. Doxygen references identify the real app APIs under test.

**Validation after the changes:**

| Check | Result |
| --- | --- |
| Fresh standalone C++20/O3/`-fno-fast-math`/`-Wall -Wextra` build | Passed. |
| Complete standalone numerical/timing suite | **13 cases / 20764 assertions passed** in `/tmp/accelSim_indi_fix_model_test '[model]'`. |
| Synchronization regression selection | **5 cases / 12181 assertions passed**, including the two newly executable timing-diagnostic regressions. |
| Fresh ASan/UBSan O1/debug build and complete standalone run | **13 cases / 20764 assertions passed**, no sanitizer findings. Covers model/timing helpers, not full app/INDI concurrency. |
| Python PSD tests, warnings treated as errors | **4 tests passed**. |
| Fresh diagnostic export | **1 case / 6 assertions passed**; 360000 paired samples, 214 modes, 6000 Hz / 60 s / seeds 42 and 123. |
| Full offline PSD validation | Passed finite/time-grid/count checks and regenerated all five plots, `psd.csv`, and `validation.json` in `/tmp/accelSim-indi-fix-validation`; log `/tmp/accelSim-indi-fix-psd.log`. No ADC clipping. |
| Physics preservation check | `samples.csv`, `modes.csv`, and `metadata.json` are byte-for-byte identical to the preceding `/tmp/accelSim-compatibility-validation` export. |
| Formatting / diff / documentation | clang-format applied and `--dry-run --Werror` passed on all three touched C++ files, preserving exact `//mutex scope` spelling via the previously documented style override. `git diff --check` passed. No authors added; declarations, members, parameters, test groups, and explicit wrapper-test Doxygen references checked. |
| Normal application / full INDI and stream tests | Not rerun or bypassed: the user explicitly retained the existing shared build blockers. The new/extended full-app regressions are not claimed to have compiled or passed. |

Commands used for the available C++ checks, from the repository root:

```sh
c++ -std=c++20 -O3 -fno-fast-math -Wall -Wextra \
  -DCATCH_CONFIG_MAIN -DACCELSIM_MODEL_ONLY \
  apps/accelSim/tests/accelSim_test.cpp -o /tmp/accelSim_indi_fix_model_test
/tmp/accelSim_indi_fix_model_test '[model]'
/tmp/accelSim_indi_fix_model_test '[sync-regression]'
c++ -std=c++20 -O1 -g -fno-fast-math -fsanitize=address,undefined \
  -fno-omit-frame-pointer -DCATCH_CONFIG_MAIN -DACCELSIM_MODEL_ONLY \
  apps/accelSim/tests/accelSim_test.cpp -o /tmp/accelSim_indi_fix_sanitize
/tmp/accelSim_indi_fix_sanitize '[model]'
/tmp/accelsim-validation-env/bin/python -W error -m unittest discover \
  -s apps/accelSim/tests -p 'test_validate_psd.py' -v
mkdir -p /tmp/accelSim-indi-fix-validation
ACCELSIM_EXPORT_DIR=/tmp/accelSim-indi-fix-validation \
  /tmp/accelSim_indi_fix_model_test '[accelSim-export]'
MPLCONFIGDIR=/tmp/accelSim-indi-fix-matplotlib \
  /tmp/accelsim-validation-env/bin/python apps/accelSim/tests/validate_psd.py \
  /tmp/accelSim-indi-fix-validation
```

The Python environment already existed; no dependencies were installed or changed. The per-process Matplotlib cache is confined to `/tmp`.

**Remaining compatibility differences and limits:** no known defect among the four requested fixes remains in source. Approved differences remain (6000-Hz default, fixed two channels/stricter validation, meaningful timer acquisition timestamps, bounded simulated catch-up, honest computation timing, additive sim_diag). Valid immediate `alpha`, `numChannels`, and `synchroDelay` controls still acknowledge Ok without the hardware's brief Busy transition; this previously recorded non-FPS transition difference was not one of the four requested corrections and was not silently changed. Exact live protocol equivalence and downstream substitution remain unverified. The user-supplied deployment identity/depth/config and downstream consumer are still unavailable here.

Both retained build blockers are unchanged: macOS `timegm` visibility under mxlib's `_XOPEN_SOURCE=700`, and the missing installed `mx/math/floatUtils.hpp` API required by shared pupil-fit telemetry. No shared/dependency workaround was attempted. On a configured non-instrument Linux development host, build the normal app/tests and run `[accelSim]`, `[indi-regression]`, and `[accelSim-streams]` in their appropriate separate processes, followed by the approved ten-minute live test and actual INDI/subscription checks. Instrument-side or tunneled INDI execution remains excluded.

### Destructor Exception-Specification Fix — 2026-10-28

The user reported that the Linux/RTC build now reaches accelSim and rejects its implicitly declared `~accelSim() noexcept(false)` as a looser override of `MagAOXApp::~MagAOXApp() noexcept`. This is an **accelSim declaration omission**, separate from the previously documented workstation dependency blockers. Investigation and testing were performed only on the local workstation; no instrument connection or execution occurred. The starting tree was clean at commit `8e148b87`.

**Cause and existing patterns:** `pcf::ReadWriteLock::~ReadWriteLock` in `INDI/libcommon/ReadWriteLock.hpp:84` is explicitly `noexcept(false)` and throws if `pthread_rwlock_destroy` fails. `pcf::IndiProperty` owns that lock as `m_rwData`; its destructor has no explicit exception specification and is consequently potentially throwing. accelSim owns INDI properties directly, and its frameGrabber base also owns INDI properties and has an implicit destructor. Either path makes accelSim's implicit destructor potentially throwing, violating its virtual MagAOXApp base contract. The current telemeter destructor is already explicitly `noexcept`, so changing telemeter alone would not address these remaining paths. This is unrelated to AR state, RNG, ADC conversion, or acquisition timing.

`mcp3208Ctrl` uses precisely the same three bases and explicitly declares an empty `noexcept` destructor. `cameraSim` and `ocam2KCtrl` also include MagAOXApp, frameGrabber, and telemeter (plus other helpers) and explicitly declare/define `noexcept` destructors. accelSim now follows this convention with a documented public declaration `~accelSim() noexcept override;` and an out-of-class inline defaulted definition. Defaulting preserves automatic member/base cleanup; no duplicate shutdown, catch/suppression logic, resource handling, or shared-class change was introduced. As with the existing apps' explicit `noexcept` contract, an escaping member-destructor exception would terminate rather than propagate.

**Exact changed files:**

- `apps/accelSim/accelSim.hpp`: four added lines for the documented declaration and inline defaulted definition; the only production change.
- `apps/accelSim/tests/accelSim_test.cpp`: a full-app-only `static_assert(std::is_nothrow_destructible_v<MagAOX::app::accelSim>)` and explanatory comment, using the already included `<type_traits>`.
- This planning record, below the agent heading. No model, configuration, README, shared MagAO-X, mxlib, or build-file changes.

**Verification:**

| Check | Result |
| --- | --- |
| Independent compile diagnosis using the real INDI headers | `static_assert` confirms ReadWriteLock and IndiProperty are potentially throwing. A minimal virtual-noexcept-base class owning an IndiProperty fails with an implicit destructor and passes with the same explicit/defaulted noexcept fix. Both probes use C++20 syntax-only compilation, no stubs or dependency workarounds. Temporary sources: `/tmp/accelSim-destructor-implicit.cpp` and `/tmp/accelSim-destructor-fixed.cpp`. An existing unrelated IndiProperty warning remains. |
| Fresh optimized standalone build and complete suite | **13 cases / 20764 assertions passed** in `/tmp/accelSim_destructor_model_test '[model]'`. |
| Synchronization regressions | **5 cases / 12181 assertions passed**. |
| Fresh ASan/UBSan build and complete standalone suite | **13 cases / 20764 assertions passed**, no sanitizer findings. |
| Python PSD unit tests with warnings treated as errors | **4 tests passed**. |
| Fresh 60-second diagnostic export | **1 case / 6 assertions passed**, 360000 pairs / 214 modes. Samples, modes, and metadata are byte-for-byte identical to the preceding INDI-fix export. |
| Full offline PSD validation | Passed; all five plots, `psd.csv`, and `validation.json` in `/tmp/accelSim-destructor-validation`. Log: `/tmp/accelSim-destructor-psd.log`. |
| Normal workstation app build | `make -C apps/accelSim` failed, exit 2, in shared compilation with the unchanged `stdFileName.hpp:366` timegm and `telem_pupilfit.hpp:8` missing mx/math/floatUtils.hpp errors. Log: `/tmp/accelSim-destructor-build.log`. No workaround applied. |
| Formatting and scope | clang-format applied and checked on both touched C++ files with the existing mutex-comment preservation override; `git diff --check` passed. Specification and ANSWER entries preserved. |

The added assertion is deliberately excluded from model-only builds: those do not instantiate the app and cannot validate its virtual override. The independent real-header probe confirms the exception-specification mechanism and fix, but is not a full app build. Rebuild the app and normal tests on the configured non-instrument Linux development host to exercise the actual app assertion and identify any subsequent compiler errors. No claim is made that the entire application now builds or that instrument integration is complete. No commit or push was performed.

### Continuous WFS-Referenced Sampling — 2026-10-30

The user now explicitly requires the WFS semaphore to be a phase reference, not an output trigger. This supersedes the earlier one-ADC-publication-per-source-notification behavior and its corresponding source-to-sample accounting assumptions. The user reports free-running `acc000` at **6000.013 Hz on exao2**; that is user-provided evidence, not a measurement made by this agent. All work here was local. No instrument access, shared infrastructure/dependency edits, commit, or push occurred.

Before editing, inspection confirmed that every synchronized acquisition called `waitSync`, blocked on one semaphore, waited the model phase, and returned just one frame. Its recovery helper advanced missing numerical ticks but could not publish intervening samples. The revised implementation separates WFS reference accounting from an independent accelerometer clock. The physical model, ADC calibration, RNG sequences, timestamps/stream publication ownership, and free-running timer remain unchanged.

**Implementation and scheduling decisions:**

- Added app-local `phaseClock` in `accelSimModel.hpp`, replacing the obsolete `syncClock`/`syncTick` one-trigger accounting helper. The original `schedule` implementation remains unchanged and is still used by free-running operation.
- Synchronized acquisition uses one worker/event loop, waiting for either a WFS semaphore or the next absolute accelerometer deadline. References never change the already pending deadline and never publish a frame. At an already-late deadline the loop consumes at most one queued reference, preventing source-accounting starvation under load, then proceeds with acquisition.
- Let `Ta=1e9/fps`, `E` be the accelerometer epoch, `c` the source counter, and `Tr` the nominal WFS reference period. After measuring a first positive WFS period, retain its source counter `c0`. For each reference, compute the requested WFS phase `d` from the measured/smoothed period and the existing four latency terms plus signed postDelay. The desired accelerometer phase is `wrap_half_acc((arrival-E) + d - (c-c0)*Tr)`. Subtracting fractional WFS cycles is essential: snapping each individual WFS event to a sample would distort non-integer rate ratios.
- The target and applied phase offsets stay within half an accelerometer period of the nominal epoch. Each completed acquisition applies a bounded correction only to the **following** interval. For on-time operation, deadlines have the form `E+n*Ta+bounded_phase`; the bounded phase cannot accumulate a long-term frequency change. Constant 2-kHz references therefore support both 8-kHz and 6-kHz sampling, and fractional ratios such as 5.5/2 kHz. A sample cannot coincide with every WFS phase reference for a non-integer ratio while retaining the requested rate; the relative phase advances fractionally instead.
- Acquisition begins at the requested rate even before a reference is available. Phase acquisition takes several samples while bounded corrections settle; it is not an immediate phase jump or a withheld batch of samples. At a stable 2-kHz cadence, the existing latency sum 340.6 us yields **409.4 us**, calculated rather than hard-coded.
- WFS cadence measurement divides the elapsed unique-arrival interval by source-counter progress, so missed notifications do not masquerade as a slower WFS. The first interval seeds the period estimate directly; subsequent measurements use the existing alpha EMA. This deliberately avoids the former zero-initialized EMA producing an artificially short startup period.
- Source counters are committed immediately on receipt, independently of later wait cancellation or publication. Duplicate notifications do not create extra frames. `sync_frames_dropped` and gap counts now describe missing WFS references only. `skipped_publications` describes missed accelerometer deadlines only; there is no conversion of WFS gaps into accelerometer skips.
- On a late acquisition, elapsed accelerometer ticks are accounted once; `model::advance` still advances up to eight skipped ticks and publishes only the newest sample. Larger gaps retain the existing counted stationary restart. The next synchronized publication deadline is at least half an accelerometer period beyond the acquisition, preventing an immediate catch-up burst. Normal corrections shorten an interval by at most the configured slew fraction.
- Missing/stalled/replaced sources clear lock/cadence/delay measurements and enter holdover at the last applied phase. Sampling continues, including when the source is absent at startup. The input mapping is closed/retried within the worker, without requesting frameGrabber output reconfiguration, resetting output counters, or restarting the numerical model. A source counter reset also clears timing history without restarting physics. Explicit application reconfiguration and FPS changes retain their existing stationary-restart semantics.
- FPS controls/subscriptions still specify the **accelerometer** nominal rate. No INDI property names or permissions changed. Leave `fps.device` empty unless its source advertises the requested accelerometer FPS; subscribing it to WFS `fps.current=2000` still requests 2000-Hz accelerometer operation. The new WFS cadence setting is separate.

**New configuration (all app-local):**

| Key | Default | Meaning |
| --- | --- | --- |
| `synchro.referenceFps` | 0 | Nominal WFS cadence used only to unwrap fractional WFS cycles. Zero learns the first measured interval after timing acquisition/reset. A known nominal cadence such as 2000 is preferable when available because the first arrival interval can contain wake-up jitter. The phase-delay formula itself still uses measured/smoothed cadence. |
| `synchro.maxSlew` | 0.05 | Maximum phase adjustment per accelerometer interval as a fraction of that period; valid `(0,0.25]`. Measured computation time and existing `cadenceGuard_ns` can reduce the available headroom. |
| `synchro.timeout` | 0.01 s | Silence timeout and source-file probe/retry interval; valid positive values through 3600 seconds. Set longer than the expected WFS period for slow references. Sampling does not stop on timeout. |

The reference period is a nominal phase-unwrapping baseline, fixed until timing reacquisition. A changed nominal WFS rate requires matching `referenceFps`/reacquiring the reference; an incorrect or noisy learned baseline can produce repeated correction transients. Bounded phase prevents those transients from changing the long-term requested accelerometer rate, but this is not a claim of stable phase lock with a wrong cadence baseline. This limitation is documented in the README rather than hiding a rate change in the simulation model. Configuration rejects nonfinite/negative/overflowing reference periods and invalid slew/timeout settings.

The shipped example retains `sim.fps=6000`, empty source/subscription names, `acc000`, UINT16, and default `[2,1,1]`. Existing optional circular depth behavior is unchanged. For the requested synchronized example, set `sim.fps=8000`, `synchro.shmimName=camwfs_sync`, `synchro.referenceFps=2000`, and leave `fps.device` empty. No instrument startup file was added.

**Diagnostic meanings updated where necessary:**

- `delay_model_us`, target phase, and trigger prediction remain WFS-phase quantities. The first valid prediction still has no interval until a second prediction exists.
- Applied delay represents the requested WFS phase plus the remaining scheduled-clock phase error, wrapped over the WFS period. Phase-error wrapping keeps the hardware's positive half-period convention. Lock requires a fresh reference, the existing absolute/fractional thresholds, and acceptable acquisition lateness; it is not an independently measured hardware-latency guarantee.
- Delay budget/capped now describe available per-accelerometer-interval correction and remaining correction beyond that budget, rather than a single semaphore sleep.
- Source-to-sample latency is measured for retained WFS predictions at the first sample at/after their phase. An intervening WFS event does not erase the pending latency measurement. Not every reference must contribute a measurement; latency-error uses that retained prediction, not a newer event's target. Service/readout reports actual model work.
- Received/gap/ID fields describe WFS notifications; written counts all accelerometer publications in synchronized mode, including holdover. Written can exceed received. `source_period_ratio` is WFS period / accelerometer period, so 4 for 8/2 kHz and 3 for 6/2 kHz are expected.
- Outage invalidation still sends a coherent snapshot and preserves lifetime accounting. Existing batched INDI publication and FPS Busy/Ok/Alert acknowledgment are unchanged.

**Regression coverage and results:**

The two old one-trigger accounting tests were replaced with tests of their relevant invariants under the new continuous-clock semantics; keeping their old expected one-to-one omissions would enforce the behavior the user asked to remove. All numerical tests and the existing diagnostic invalidation, half-period boundary, and startup prediction tests remain.

| Check | Result |
| --- | --- |
| Requested-rate event-loop regression | Passed for WFS/accel pairs 2/8, 2/6, 2/5.5, 2.3/6, and 9/6 kHz, with one-second counts within one boundary sample of the requested rate. Reference receipt leaves the pending deadline unchanged; normal intervals remain at least 95% of the nominal accelerometer period. These are deterministic virtual-time results, not measured live publication rates. |
| Unchanged free-running timer | Existing 6000-step test and added one-second 6000-Hz schedule check pass. Original timer schedule definitions, waitUntil, free-running acquisition branch, and stream-copy function are unchanged. |
| Phase calculation/correction | Passed 409.4-us calculation at 2 kHz, exact settled phase grid, positive/negative corrections, duplicate-reference handling, and no immediate reference-triggered bursts. |
| Missing references / holdover | Passed counter-gap normalization, continuous 6000-sample holdover, timeout and reacquisition, duplicate and reset source counters without extra samples. |
| Jitter / persistent phase shifts / rate stability | Passed a two-second 8-kHz virtual run with alternating reference jitter, positive/negative phase steps, and a 50-ms reference outage. Output count remains within one sample of 16000; intervals respect the correction bound; holdover/recovery both occur. |
| Missed sample / cancellation invariants | Passed immediate source accounting independent of canceled waits, preserved source history at FPS reset, exact numerical advance through three skipped accelerometer ticks, and bounded stationary recovery after a long stall. |
| Optimized C++20 standalone build and complete suite | **16 cases / 154423 assertions passed**. Binary `/tmp/accelSim_phase_model_test`. |
| Synchronization selection | **8 cases / 145840 assertions passed** with `[sync-regression]`. |
| Fresh ASan/UBSan build and standalone suite | **16 cases / 154423 assertions passed**, no sanitizer findings. Binary `/tmp/accelSim_phase_sanitize`. |
| Python PSD tests, warnings as errors | **4 tests passed**. |
| Diagnostic export and full offline PSD analysis | Export **1 case / 6 assertions passed**; complete analyzer passed and wrote all five plots plus spectra/JSON in `/tmp/accelSim-phase-validation`. Exported samples, modes, and metadata are byte-identical to the preceding destructor-fix export, confirming unchanged physics/noise/ADC behavior. |
| Normal app and full test builds | Both failed at the unchanged shared `timegm` and missing `mx/math/floatUtils.hpp` errors, before full-app validation. Logs `/tmp/accelSim-phase-app.log` and `/tmp/accelSim-phase-app-tests.log`. No dependency or shared build workaround applied. |
| Formatting / scope / documentation | clang-format applied/checked with the existing mutex-comment spelling override; `git diff --check` passed. Only the app headers, app tests, example config, README, and this plan changed. |

Full-app tests were updated to check phase/budget/lock and coherent holdover diagnostics, continued frameGrabber publication with a missing startup source, more accelerometer samples than source notifications, duplicates, disappearance/reconnection without output reconfiguration/model restart, and invalid new configuration values. These tests, existing INDI serialization tests, and real stream/timestamp/semaphore checks remain **uncompiled/unexecuted on this Mac** because of the shared build blockers. They must run on the configured non-instrument Linux development host. In particular, no actual 6/8-kHz synchronized rate, jitter distribution, semaphore wake latency, or sustained memory result is claimed here.

**Files changed:** `apps/accelSim/accelSim.hpp`, `apps/accelSim/accelSimModel.hpp` (timing helpers only), `apps/accelSim/tests/accelSim_test.cpp`, `apps/accelSim/config/example.conf`, `apps/accelSim/README.md`, and this planning document below the agent heading. `MagAOXApp`, frameGrabber, telemeter, mcp3208Ctrl, ImageStreamIO, mxlib, and the reference notebook are untouched. The new synchronized behavior intentionally differs from mcp3208Ctrl's one ADC read per semaphore; free-running acquisition and the raw ADC stream contract remain as before.

The existing read-only ten-minute benchmark now accepts the test-only environment variable `ACCELSIM_EXPECTED_FPS` (default 6000), so the same reader can validate an independently configured 8000-Hz producer without a hard-coded 6000-Hz assertion. For example, on the isolated non-instrument development host with the producer and private shared-memory namespace already configured:

```sh
ACCELSIM_LOCAL_BENCHMARK=1 ACCELSIM_EXPECTED_FPS=8000 \
  ./apps/accelSim/tests/accelSim_test '[accelSim-live]'
```

This is validation tooling only and does not send an FPS command or change application configuration. It was not run here because the full test binary remains blocked.
