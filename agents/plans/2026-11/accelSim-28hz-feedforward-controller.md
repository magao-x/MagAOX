# Task Description
<!-- The first section is filled out by the user 
     
     This is intended as a guide for how to prompt a coding agent to solve a problem.
     
     Make a copy of this template with a new name under agents/plans/<YYYY-MM>. Then
     Fill out each subsection below as needed. Feel free to add additional
     subsections, etc.  

     When complete, commit it on your feature branch and then prompt the agent to review this file.
-->

## Problem Statement
Develop a simple real-time accelerometer-based feedforward controller that converts raw accelerometer measurements into calibrated deformable mirror (DM) commands to suppress the 28–29 Hz telescope vibration observed on MagAO-X.

The controller will initially be tested using the existing accelSim application, which generates synchronized synthetic accelerometer measurements and injects corresponding physical pointing disturbances onto the tweeter DM. The eventual goal is to use the same controller architecture with real on-sky accelerometer data and extend it to data-driven subspace predictive control (DDSPC).

## Discussion
The existing accelSim application generates simulated accelerometer measurements at approximately 8 kHz through the acc000 shared-memory stream. It also generates a corresponding physical pointing disturbance and publishes the calibrated DM surface command to dm01disp10.

The proposed controller should independently read the raw acc000 measurements, estimate the corresponding telescope pointing disturbance, and apply an opposing correction through a separate tweeter DM channel.

For a harmonic vibration with frequency (f), acceleration and displacement are related by

[
a(t)=-(2\pi f)^2x(t).
]

Assuming transverse acceleration at an effective lever arm (L), the corresponding pointing disturbance is

[
\theta(t)=-\frac{a(t)}{(2\pi f)^2L}.
]

This relationship applies to an isolated vibration frequency. The controller must therefore estimate the 28–29 Hz component of the raw accelerometer signal rather than directly converting broadband acceleration into displacement.

A possible initial implementation is a narrowband harmonic estimator:

[
a(t)\approx A\cos(2\pi f_0t)+B\sin(2\pi f_0t)+n(t),
]

where (A) and (B) describe the amplitude and phase of the targeted vibration, (f_0) is the target frequency, and (n(t)) represents other vibrations and measurement noise.

The estimator should predict the disturbance at the expected DM actuation time to compensate for acquisition, processing, and actuation latency.

The estimated pointing disturbance must then be transformed into the DM coordinate system using an adjustable rotation and converted into a calibrated tweeter surface command.

The existing accelSim implementation provides a reference for the pointing-to-DM conversion using the first two calibrated tip/tilt modes from CMmodesDM.00.fits and the associated dmmask.fits.

The correction can be expressed as

[
-gBK R(\alpha)\hat{\boldsymbol{\theta}}(t+\tau),
]

where:

(g) is the adjustable controller gain.

(B) contains the calibrated tweeter tip/tilt modes.

(K) converts physical pointing angles into modal coefficients.

(R(\alpha)) transforms accelerometer coordinates into DM coordinates.

(\tau) is the prediction horizon accounting for system latency.

The controller must use only the accelerometer measurements available through the shared-memory stream. It must not access the simulator's internal physical pointing state.

The initial controller is intended as a simple physics-based benchmark for eventual DDSPC implementation, which will learn the relationship between accelerometer history and wavefront correction directly.

Relevant existing software to investigate includes:

apps/accelSim/

apps/loPredCtrlAcc/

Existing accelerometer shared-memory consumers

Existing tweeter DM command applications

MagAO-X ImageStreamIO and INDI control interfaces



## Requirements
1. Read the latest raw accelerometer measurements from acc000 without modifying the existing simulator or acquisition behavior.

2. Determine and correctly implement the conversion from the published accelerometer values into physical acceleration units, including ADC scaling, offsets, and channel conventions.

3. Implement a real-time narrowband estimator targeting an adjustable frequency initially centered near 28–29 Hz.

4. Estimate the corresponding physical pointing disturbance using an effective lever arm and the harmonic acceleration-to-displacement relationship.

5. Account for estimator phase response, measurement timestamps, processing latency, and expected DM actuation latency. Provide an adjustable prediction horizon where appropriate.

6. Support an adjustable accelerometer-to-DM coordinate rotation.

7. Reuse the existing calibrated pointing-to-DM conversion from accelSim where practical, including the deployed tip/tilt modes and pupil mask.

8. Generate an opposing DM correction using an adjustable controller gain.

9. Publish corrections to a separate, operator-approved tweeter additive channel. Do not write to dm01disp10, which is used by accelSim to inject the simulated disturbance.

10. Provide INDI controls for controller enable/disable, gain, target frequency, rotation angle, effective lever arm, prediction horizon, and controller status.

11. Default to correction disabled and zero gain. Implement appropriate initialization, fault handling, and controlled zero-command behavior.

12. Ensure that the controller operates independently of accelSim's internal model state so that the same implementation can eventually consume real accelerometer data.

13. Avoid blocking operations, dynamic allocations, file loading, and unnecessary logging in the real-time processing loop.

14. Design the controller interface so that the narrowband estimator can later be replaced or supplemented by a DDSPC predictor.

15. Preserve the existing accelSim accelerometer generation, DM disturbance injection, and synchronization behavior.

## Tests and Metrics
1. Accelerometer decoding: Verify correct ADC-to-acceleration conversion, channel ordering, physical units, and handling of invalid measurements.

2. Single-frequency reconstruction: Generate synthetic 28–29 Hz acceleration signals with known amplitude and phase. Verify the reconstructed pointing amplitude and phase against the analytical solution.

3. Noise rejection: Add broadband sensor noise and additional vibration frequencies. Verify that the estimator isolates the target vibration without excessive amplification of unrelated frequencies.

4. Frequency sensitivity: Evaluate estimator performance when the actual vibration frequency differs slightly from the configured target frequency.

5. Latency compensation: Introduce known delays and verify that the predicted disturbance has the correct phase at the intended DM actuation time.

6. Coordinate transformation: Verify the rotation convention and sign of the resulting DM correction.

7. DM calibration: Compare generated DM commands against the existing accelSim pointing-to-DM calibration for known pointing disturbances.

8. Real-time performance: Measure processing latency, timing jitter, and computational cost relative to the selected control-loop period.

9. Controller state transitions: Verify that enabling, disabling, zero gain, stream interruptions, and fault conditions produce the expected behavior without stale corrections.

10. Simulation validation: Use accelSim to generate synchronized acceleration and physical DM disturbances. Compare the controller's estimated pointing against simulation ground truth in offline validation.

11. Physical WFS validation: Measure WFS tip/tilt with correction disabled and enabled. Quantify suppression of the 28–29 Hz PSD peak, residual tip/tilt variance, and any amplification at other frequencies.

12. Gain response: Verify that controller gain changes produce the expected correction amplitude and identify the gain and phase settings that minimize residual vibration.

## Scope and Caveats
1. The initial implementation targets one narrowband vibration near 28–29 Hz, not general broadband vibration correction.

2. The effective lever arm of 10 m is an initial simulation parameter, not a verified universal conversion for real telescope accelerometers.

3. Real telescope vibrations may involve multiple structural modes, non-rigid-body motion, and frequency-dependent accelerometer-to-WFS transfer functions.

4. A single instantaneous accelerometer measurement is insufficient to determine displacement without additional assumptions or temporal estimation.

5. Accelerometer and DM coordinate axes may differ. The rotation and sign convention must be verified experimentally.

6. The controller must account for the difference between accelerometer acquisition rate, WFS sampling rate, and actual DM command application rate.

7. The initial implementation should not assume that every 8-kHz published DM command is physically applied as a distinct actuator update.

8. No physical DM correction should be activated during development or automated testing.

9. The controller must not access accelSim's internal m_pointing except through separate offline validation tools.

10. The choice of controller application, DM output channel, and control-loop cadence must be investigated before implementation.

11. Do not introduce arbitrary hardware displacement limits without instrument-specific justification. Preserve existing MagAO-X protections and software-integrity checks.

12. Do not commit or push implementation changes automatically.

# Instructions to Agent
<!-- Specific instructions for the agent.  Below is our standard, but you can modify it as needed. -->

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan

Investigation date: 2026-11-05. Reviewed `AGENTS.md` and local source on
`parkertjohnson1/accelSim`, HEAD `c786efca525bccbf29e5be6d628e202d81242ce3`
(`Add accelSim tweeter disturbance injection`). This update is a proposal for
review; no implementation, build, stream access, instrument access, commit, or
push was performed. All text preceding this heading is preserved byte-for-byte.

## Task Summary

Build an independent, two-axis accelerometer feedforward benchmark that reads
raw `acc000`, estimates one adjustable vibration near 28–29 Hz, predicts its
pointing contribution at DM application time, and writes the opposing calibrated
surface to a separately assigned tweeter additive channel. Establish performance
in numerical replay and private Linux stream tests before any user-operated
physical correction. Keep the simulator's acquisition, model, RNG, timing,
synchronization, and channel-10 injection behavior intact.

Recommended first implementation:

- A new application, provisionally `apps/accelFFCtrl/`, rather than adding a
  controller to accelSim or changing the existing WFS DDSPC loop.
- Timestamped raw-ADC ingestion at the source rate, initially 8000 Hz; a bounded
  three-parameter harmonic fit per axis with shared small-matrix work; and an
  initially 2000-Hz correction publisher. Both rates are explicit and measured.
- Weighted least squares of offset, cosine, and sine at an adjustable frequency,
  using a two-stage exponential window for stronger off-band rejection.
  Evaluate the fitted harmonic at the intended application
  time; apply the harmonic acceleration-to-angle conversion, axis registration,
  controller gain, and existing calibrated TT synthesis exactly once.
- Observation/shadow operation by default, correction disabled, gain zero,
  no default DM output channel, and one owner for all correction publications
  and zeros. Reuse small numerical and stream helpers through neutral interfaces
  rather than importing simulator model state or the DDSPC learning machinery.

The following distinguishes checked source facts, proposed choices, and missing
deployment evidence. In particular, 2 kHz is a proposed controller output rate;
it does not change Jared's existing 8-kHz requirement for accelSim injection.

### Source inventory and reusable interfaces

Paths in this table are relative to the repository root unless marked otherwise.
Function names are more durable references than line numbers during development.

| Source | Finding and intended use |
| --- | --- |
| `apps/accelSim/accelSimModel.hpp`: `parameters`, `adcCounts`, `accelerationFactor`, `referenceModes`, `model::step` | Authoritative simulation units, ADC forward conversion, negative frequency-squared pointing/acceleration relation, and target-mode parameters. Read for the contract and use only in separate offline validation fixtures, not in the controller. |
| `apps/accelSim/accelSim.hpp`: `configureAcquisition`, `acquireAndCheckValid`, `loadImageIntoStream`, `frameGrabberPostPublish`, `prepareDM` | UINT16 X/Y publication, acquisition timestamp, nominal model pacing, same-sample channel-10 injection, FITS loading and identity checks. Preserve behavior; reuse calibration preparation principles and owner-thread control transitions. |
| `apps/accelSim/config/example.conf`, `apps/accelSim/README.md` | Defaults are 6000 Hz, circular depth 1, DM disabled, scale 0, and empty DM channel. The experiment explicitly selects 8000 Hz and `dm01disp10`; neither follows just from the app name. ADC constants and calibration conventions are documented here. |
| `apps/accelSim/accelSimDisturbance.hpp`: `tipTiltProjection`, `response`, `coefficients`, `tipTiltBasis::synthesize`, `zero` | Existing validated QR projection and full-image angular response B*K. Reuse these algorithms and their float synthesis order. Current public calls take simulator-specific `pointingDisturbance`; introduce a neutral radians interface while preserving its adapters. |
| `apps/accelSim/accelSimDMStream.hpp`: `dmStream::open`, `valid`, `probe`, `acquisitionTime`, `operator()`, `close` | Existing-only 2-D 50x50 float attachment and active `ImageStreamIO_UpdateIm` publication, no consumer wait, busy-write rejection, inode invalidation, and explicit close. Reuse through a common helper with the same publication behavior. Mapping ownership is not an exclusive-writer lease. |
| `apps/accelSim/accelSimDM.hpp`: `dmInjection::service`, `zero`, `dmState` | Reference for disabled startup, revised enable requests, stale-request rejection, fault latching, overflow/nonfinite checks, and pending-zero retry. Its sample type and 8-kHz injection health policy belong to the simulator; do not reuse the entire injection state machine for correction. |
| `apps/mcp3208Ctrl/mcp3208Ctrl.hpp`: `configureAcquisition`, `acquireTimerAndCheckValid`, `acquireSynchroAndCheckValid`, `loadImageIntoStream`, `readChannelValue`; `dependencies/MCP3208.cpp`: `MCP3208::read` | Real acquisition publishes ADC codes, reading hardware channel 0 then 1. The SPI reader extracts 12 bits; it does not convert to volts, g, or m/s². Real sensor calibration and physical axis labels cannot be recovered from this driver. |
| `libMagAOX/app/dev/frameGrabber.hpp`: `fgThreadExec` | Source layout and publication contract: normally a 3-D temporal cube, `cnt1` identifies the last completed plane, `cnt0` counts publications, and per-plane counter/acquisition/write-time arrays accompany the data. Semaphores are posted before accelSim's DM hook. |
| `libMagAOX/app/dev/shmimMonitor.hpp`: `allocate`, `processImage` CRTP interface, monitor thread, `SHMIMMONITOR_*` macros | Reusable input attachment, thread management, geometry/type discovery, semaphore claiming, and reconnect machinery. It passes a pointer to the latest plane; it does not guarantee unique, ordered, coherent delivery of every ADC pair or provide a short application timeout callback. |
| `libMagAOX/app/MagAOXApp.hpp`: `threadStart`, standard INDI property helpers, callback registration, application loop | Use the normal application, INDI, scheduling-priority/cpuset conventions. The default main-loop pause is one second; it is unsuitable as the correction stale-input watchdog. |
| `apps/streamWriter/streamWriter.hpp`: acquisition loop using `cntarray`, `atimearray`, `writetimearray` | Reference for distinguishing depth-one streams, per-plane metadata, duplicate semaphore posts, and skipped frames. Its latest-frame drain is a recorder policy, not ordered sample recovery; do not copy that policy into the estimator. |
| `utils/shmimLat/shmimLat.hpp`, `run_with_cset.txt` | A checked-out consumer explicitly used with `acc000`. It records `md.writetime` and arrival time, not decoded acceleration or true acquisition-to-actuation latency. Useful timing practice, insufficient physical-unit validation. |
| `apps/loPredCtrl/loPredCtrl.hpp`, `ar_controller.{hpp,cpp}`, `recursive_least_squares.{hpp,cpp}`, `utils.hpp` | Current-branch WFS/modal DDSPC example. `realT` is float; `processImage` maps modal floats, learning uses dynamic matrices, and `send_to_shmim` / its invocation are commented out in this version. It is not a working raw-ADC-to-actuator adapter to extend blindly. |
| `apps/loPredCtrlAcc/` on local refs described below | Existing accelerometer-as-exogenous-input DDSPC work, inspected read-only with `git show`; absent from the current checkout. Useful future model interfaces, but raw type handling, sample identity, cadence, freshness, and allocation behavior need work. |
| `apps/hoPredCtrl/` | Existing GPU predictive-control alternative. Its broader modal/GPU architecture is unnecessary for a two-axis harmonic benchmark; do not add a CUDA dependency. |
| `apps/dmMode/dmMode.hpp`: `sendCommand`; `apps/dmSpeckle/dmSpeckle.hpp`: modulation/zero publication; `apps/t2wOffloader/t2wOffloader.hpp`: `processImage`, `zero` | Additive-channel attachment, full-image synthesis, counters/semaphore and zeroing precedents. Some paths wait on write flags, use trigger/passive modes, or involve other DMs and masks; reuse the contract rather than their waits, offload transforms, or application-specific limits. |
| `libMagAOX/app/dev/dm.hpp`; `apps/bmcCtrl/bmcCtrl.hpp`: `commandDM`; `apps/alpaoCtrl/alpaoCtrl.hpp`: `commandDM` | Driver-owned calibration/saturation and channel accounting. The controller must not inherit the hardware-owning `dev::dm`, call `zeroAll`, redo voltage conversion, or change flat/AO channels. Confirm correction-channel gain and `deltaChannels` assignment operationally. |
| Local mxlib `include/improc/milkImage.hpp`: `post`, `passive` | Active `post` delegates to `ImageStreamIO_UpdateIm`; passive mode avoids the active counter behavior. This wrapper can throw, and its active path sets `atime` to write time. Use the existing explicit DM adapter and record source identity separately; verify the installed native primitive's metadata semantics. |

### The `loPredCtrlAcc` branch discrepancy

`apps/loPredCtrlAcc/` does not exist at current HEAD. No checkout, merge, fetch,
or worktree change was needed: the following local Git objects were available.

| Local ref inspected | Commit | Relevant behavior |
| --- | --- | --- |
| `parkertjohnson1/loPredCtrlAcc` | `651aa21bcb336bb8edeac2ad5e4d568ac3b33939` | Dual tagged `dev::shmimMonitor` parents, `controller_mode` legacy/accel selection, latest normalized accelerometer sample, WFS-driven controller update, and legacy fallback. |
| `parkertjohnson1/accel_ddspc` | `1a311d279e3282a9629a6a6004d09f2866ba8b74` | Earlier accelerometer-history extension with normalization and missing-frame fallback. Same important raw-input and time-alignment limitations. |

Specific findings in both versions:

1. `processImage(void *, const accelShmimT &)` maps `curr_src` as
   `Eigen::Map<eigenImage<realT>>` with `realT=float`. It does not validate and
   decode `_DATATYPE_UINT16` codes from the current producer. A two-channel ADC
   pair is four bytes, whereas two floats occupy eight; reinterpretation can
   produce meaningless values and read beyond that plane. Normalization cannot
   repair the type mismatch.
2. `normalizeAccelSample` performs running z-score normalization with optional
   sigma clipping. Those are learned-feature choices; their output has no fixed
   m/s² amplitude. They must not precede the physics conversion in this controller.
3. The accelerometer callback retains `m_latestAccelSample` under `m_accelMutex`.
   The WFS/modal callback pushes it into `PredictiveController` once per WFS
   callback. This discards intermediate accelerometer samples and does not align
   an 8-kHz history to the WFS exposure. `m_haveAccelSample` is not consumed or
   timestamped at each use; a stopped source can leave an old sample accepted as
   present, defeating the intended missing-frame count.
4. The branch's `ar_controller.{hpp,cpp}` provides
   `DDSPC::PredictiveController::push_accelerometer_sample`,
   `get_accelerometer_past`, `calculate_command`, `update_system`, and
   `update_controller`. Regressors combine past measurements, commands, and
   acceleration features. This is a useful future interface, not the initial
   narrowband estimator. Updates construct/resize dynamic matrices, and
   `update_controller` forms a matrix inverse; learning must not be imported into
   an allocation-free 8-kHz path on that basis alone.

These are observations of local branch snapshots, not statements about the
deployed consumer or its configuration. The deployed version remains an input
to obtain from the user. Merging either branch is outside this proposal.

### Correct decoding and physical units

The current simulation contract is unambiguous:

```text
raw[0] = X/channel-0 uint16_t ADC code
raw[1] = Y/channel-1 uint16_t ADC code
code = uint16_t(clamp(round(zero + acceleration * sensitivity / voltsPerBit), 0, 4095))
acceleration = (double(code) - zero) * voltsPerBit / sensitivity  [m/s²]
```

Cast to a signed/floating type before subtracting the zero point. Do not treat
the codes as signed int16, floats, millimetres, or pre-calibrated acceleration.
Do not replace the empirical scale with `3.3/4096`, or add a factor of 9.81.
For the simulation, the effective sensitivity has the units needed to map m/s²
to the supplied voltage calibration. Its numerical value is not proof of a
real accelerometer's V/g specification.

| Quantity | Simulation default or derived value |
| --- | --- |
| Zero point | 2048 counts, configurable |
| Volts per count | 0.0083799, configurable empirical value |
| Sensitivity | 1.02, configurable empirical value |
| One ADC count in acceleration | 0.008215588235 m/s² |
| 28-Hz component in `referenceModes` | 0.002 m/s² RMS per axis; pole radius 0.99999 |
| Corresponding pointing at L=10 m, f=28 Hz | Approximately 1.333 mas RMS per axis |
| One-count instantaneous harmonic equivalent at 28 Hz, L=10 m | Approximately 5.475 mas; not a displacement resolution claim for a temporal estimator |
| Sensor-noise standard deviation at 8000 Hz and PSD=1e-8 | Approximately 0.006325 m/s², before quantization and other vibrations |

The target RMS is less than one ADC count. Fit many quantized samples and retain
the full precision of the calibration. Broadband vibration/noise can provide
dither, but do not assume ideal independent uniform quantization error for a
quiet, sub-count sinusoid. Test both dithered and undithered cases. Clipping loses
information irreversibly; no inverse calibration can restore it.

Proposed decoder configuration allows separate X/Y offsets, scale/sensitivity,
channel selection, and polarity, with defaults matching the simulator's common
values. Those settings are a versioned startup calibration. Monitor/report ADC
rails; values above 4095 are invalid, not silently masked to twelve bits. A code
of 0 or 4095 is legal and does not alone prove that clipping occurred. Report
rail occupancy and inhibit for sustained saturation according to a reviewed
data-quality policy, rather than inventing an actuator limit.

No physical calibration metadata is embedded in the two-code stream. The
controller must be configured to match the producer; it cannot infer offsets,
sensitivity, lever arm, sensor axes, or source INDI instance from `acc000`.

### Stream, timestamp, and sample-identity contract

1. Accept the normal 3-D `(2,1,D)` UINT16 temporal cube, including D=1. Support a
   verified 2-D `(2,1)` depth-one producer explicitly; reject other shapes/types,
   invalid plane indices, absent data, and unsupported metadata layouts.
   One frame is exactly two contiguous codes, not the entire circular cube.
2. Select the completed plane using `cnt1`. For temporal arrays, retain that
   plane's `cntarray`, `atimearray`, and `writetimearray`; for a verified depth-one
   layout use its associated global metadata. Do not combine old-plane pixels
   with the newest plane's timestamp. Capture source generation, counter, codes,
   acquisition/write timestamps, and local monotonic arrival in one fixed record.
3. `dev::shmimMonitor` already calculates the latest plane pointer, but the
   callback must copy into owned fixed storage while checking the publication
   identity/write state before and after the copy. Use a bounded retry/reject
   policy and the installed ImageStreamIO consistency contract, not an unbounded
   spin or a borrowed pointer queued to another thread. Exercise ring-wrap races
   in tests; per-plane counters alone do not prove no overwrite is in progress.
4. Deduplicate by source generation and counter. Semaphore receipts are wakeups,
   not unique samples. Duplicate wakeups must not update the estimator or refresh
   its freshness deadline. On a counter gap, recover only consecutive retained
   planes whose sequence/data/timestamps can still be validated, if that bounded
   recovery is implemented. Otherwise inhibit/reset and report the gap; never
   advance the estimator several times using the same latest code pair.
5. For the first version, unrecoverable gaps, queue overflow, counter rollback,
   stream replacement, invalid time, or rate change invalidate the fit, zero
   correction, and require explicit rearm after a fresh warmup. Input observation
   may reconnect; correction does not automatically resume. A configured source
   circular depth greater than one can improve retention without altering
   acquisition algorithms. Default depth one cannot recover missed samples.
6. Use acquisition time for harmonic phase, write time for producer latency, and
   monotonic arrival for liveness. Frame counters do not encode omitted simulator
   model ticks: accelSim can advance skipped ticks without publishing them.
   Do not infer physical sample time solely from `cnt0 / expectedFps`.

Important timestamp limitations found in source:

- accelSim stamps `m_currImageTimestamp` using `CLOCK_REALTIME` immediately before
  model advancement. Its physical model evolves at nominal sample intervals;
  scheduling jitter, phase slew, or skipped ticks can separate model time from
  publication time. Validate with the actual recorded timing; do not promise
  exact cancellation under arbitrary clock warping or alter its phase clock.
- The hardware synchronized path stamps immediately before sequential channel
  reads. X and Y therefore have a real readout skew. The timer read function in
  this checkout does not itself populate `m_currImageTimestamp`; real timer-mode
  timestamp validity must be verified before using acquisition-based prediction.
  An absent timestamp must not silently become the callback arrival time.
- Convert realtime sample times to the controller's monotonic scheduling domain
  using an explicitly measured clock offset; detect clock steps and timestamp
  discontinuities. Subtract integer seconds/nanoseconds before converting short
  intervals to double. Do not subtract an epoch realtime stamp from a monotonic
  timestamp, or assume a remote producer shares this host's clock.

The short timeout/zero mechanism below is independent of the input monitor's
one-second semaphore wait. Missing frames must not leave a correction held until
the ordinary application loop notices.

## Key Assumptions

1. Initial simulation trials use 8000-Hz `acc000` with the documented ADC
   calibration, L=10 m, and matching physical DM-pupil axes. Source rate is
   configuration plus measured timestamps/counters, not a hardcoded consequence
   of the stream name. The existing 6000-Hz numerical export remains useful and
   must be labelled with its actual rate.
2. The first estimator targets one component, configured to 28 Hz for accelSim's
   default target. An on-sky 28–29 Hz peak will require tuning. Other modes and
   sensor noise remain disturbance/noise, not input to broadband double integration.
3. A real acceleration-to-pointing transfer is not established by the simulator.
   L=10 m and a rigid-body two-axis rotation are simulation starting points only.
   Real commissioning may require a complex, frequency-dependent 2x2 transfer.
4. A separate existing tweeter additive channel will be assigned by the owner.
   The default correction channel is empty. Reject `dm01disp10`, combined
   `dm01disp`, the flat channel, input streams, and known AO-owned channels;
   do not assume `02` or `11` is available.
5. Calibration remains the approved first two TT planes and associated dmmask,
   in surface microns and native pixel order. File identity and residual tolerance
   must match the reviewed calibration; arbitrary planar commands are not a
   substitute for loading it.
6. Jared's guidance that existing MagAO-X/DM command protection is sufficient
   remains applicable. Add no arbitrary actuator/P2V bound, common-scalar limiter,
   or displacement-limit INDI controls. Gain is an experiment amplitude control.
   Numerical representability, freshness, ownership, and fault checks still apply.
7. Initial development is local numerical replay or isolated non-instrument
   Linux shared memory, without hardware drivers/bridges. Agents never execute
   on exao1/exao2/exao3/exao5 or issue hardware INDI commands through a tunnel.
   Any live inspection, calibration experiment, or correction is user-operated.

## Requirements

The original requirements above remain authoritative. The proposed implementation
adds the following concrete contracts for review.

### Narrowband estimator and prediction

Use a causal harmonic fit with a two-stage exponential window, shared in
structure across X and Y. T denotes the approximate total mean memory time;
each smoothing stage has time constant T/2:

```text
phi(t) = 2*pi*f0*(t - reference_epoch), maintained as a wrapped phase
u(t) = [1, cos(phi(t)), sin(phi(t))]^T
a_i(t) ~= d_i + A_i*cos(phi(t)) + B_i*sin(phi(t))

lambda_n = exp(-2*(t_n - t_(n-1))/T)
G1_n = lambda_n*G1_(n-1) + (1-lambda_n)*u_n*u_n^T
Q1_n = lambda_n*Q1_(n-1) + (1-lambda_n)*u_n*[a_x,n, a_y,n]
G2_n = lambda_n*G2_(n-1) + (1-lambda_n)*G1_n
Q2_n = lambda_n*Q2_(n-1) + (1-lambda_n)*Q1_n
G2_n * C_n = Q2_n                  # one 3x3 solve, two right-hand sides
C_n rows = [d_xy, A_xy, B_xy]

a_hat_i(t_apply) = A_i*cos(phi(t_apply)) + B_i*sin(phi(t_apply))
theta_hat_i(t_apply) = -a_hat_i(t_apply) / ((2*pi*f0)^2 * L)   [radians]
```

The fitted offset is a nuisance parameter, excluded from predicted vibration.
Including it separates residual ADC bias/DC from the target without assuming
the finite weighted sine/cosine window has zero mean or perfect orthogonality.
Unlike two independent lock-in averages, solving the small Gram system accounts
for finite-window coupling between offset, cosine, and sine at startup and in
irregularly timed data. At exactly f0, a constant noise-free sinusoid plus DC is
represented by the fit without an arbitrary carrier phase delay after readiness.

Implementation properties:

- The second window stage is deliberate. A single exponential lock-in/fit has
  only first-order rejection far from f0. Leakage from strong high-frequency
  acceleration, divided by omega0², can create substantial spurious displacement
  compared with that mode's true acceleration/omega_actual². Do not accept the
  first-order fit merely because it reconstructs a pure target tone. Cascading
  the moment filters gives second-order envelope rejection while preserving an
  exact matched-frequency sinusoid/DC model. For uniform spacing the positive
  window weights are proportional to `(k+1)*lambda^k`; this is not an assertion
  that the final window is the ordinary single-exponential RLS objective.
- Keep two stages of double-precision fixed-size `Eigen::Matrix3d` and
  `Eigen::Matrix<double,3,2>` state. Update weighted statistics once per accepted
  source sample; solve at
  the correction cadence using fixed-size `Eigen::LDLT` and two right-hand sides.
  This is a bounded three-parameter estimator solve, not a calibration solve or
  a history-sized optimization. Check positivity/pivots, reciprocal conditioning,
  solution residual, and finite state; `info()==Success` alone is not readiness.
- Initialize statistics to zero and inhibit output until sufficient temporal
  coverage and acceptable conditioning. Proposed initial warmup is
  `max(3/f0, 3*T)` seconds; measure settling and set the reviewed value before
  accepting metrics. Do not regularize an unobservable startup fit into a
  plausible-looking correction. Reset on a frequency/timebase change.
- Start characterization with total memory T=0.1 s, a proposed tuning value,
  not a proven optimum. Each stage then has a 0.05-s time constant; at uniform
  8 kHz lambda is approximately 0.997503122. T is measured in seconds, not a
  number of samples; preserve its meaning when
  source rate changes and reset/revalidate the configuration.
- Share the oscillator and both G matrices between axes. Use a precomputed oscillator
  step for uniform intervals, or bounded timestamp-based phase increments for
  accepted irregular timing; periodically normalize/re-anchor recurrence drift.
  There is no per-actuator trigonometry, FFT, full-window scan, or dynamic matrix
  growth. Any irregular-dt exponential/trigonometric cost must be benchmarked.
- Normal equations square the design condition number. The model is only three
  well-scaled regressors, but conditioning and long-run drift remain test cases.
  Use a fixed-size QR weighted-batch oracle offline. If the statistics/LDLT route
  cannot meet conditioning and drift tests, replace its internals with a small
  square-root recursive QR implementation, keeping the same predictor contract.
  Do not repurpose the dynamic DDSPC learner without an allocation/stability audit.

The exponential weighting concept is documented in the primary
[recursive-estimation reference](https://www.mathworks.com/help/ident/ug/algorithms-for-online-estimation.html).
Eigen provides the small symmetric factorization, `solve`, and conditioning
interfaces needed here; its documentation also cautions that LDLT is not a
rank-revealing decomposition. See the
[Eigen LDLT reference](https://libeigen.gitlab.io/eigen/docs-nightly/classEigen_1_1LDLT.html).
The equations and tuning choices above are this proposal; those references are
not evidence that this controller has been implemented or validated.

#### Phase response and detuning

Do not label a narrowband estimator as phase-free in all conditions. For a slowly
varying complex envelope, the proposed window approximately has baseband response
`1/(1 + j*2*pi*delta_f*T/2)^2`. At total T=0.1 s, an offset of +0.5 Hz gives
about 0.976 amplitude and -17.9 degrees phase; +1 Hz gives about 0.910 amplitude
and -34.9 degrees. These are approximate envelope calculations, not exact
acceptance curves for the finite Gram fit, quantized noise, or the AR model.
Measure the actual amplitude/phase response over the full target/drift range.
Detuning also biases the physical conversion by `(f_actual/f0)^2`, separately
from the fit's amplitude/phase response, because the inverse uses configured f0.

Evaluate the fitted harmonic forward in time to remove the known carrier phase
from transport/application delay. This cannot remove unknown envelope evolution
or all detuning error. Tune f0 to the measured peak and choose T from the
noise/tracking tradeoff. No automatic frequency tracker, broadband inverse,
unbounded phase lead, or extra stochastic model belongs in version one.
If drift makes a single configured line inadequate, use the measured results to
justify a later frequency tracker or oscillator/Kalman predictor behind the same
interface. An arbitrary all-band time shift is not a replacement for that work.

#### Prediction horizon and latency budget

At each command deadline, predict from the newest accepted sample to:

```text
t_apply = estimated publication time
          + reviewed publication-to-effective-DM delay
          + any explicitly accounted hold-center offset
horizon_from_sample = t_apply - newest_acquisition_time
```

Use current monotonic time and a measured allowance for remaining synthesis and
publication work when estimating publication time. Source readout/publication
and queue age are already contained in the age of the sample; do not add them
again as a fixed configured latency. Log the total effective horizon actually used.

The tunable prediction-horizon property denotes the downstream delay component
with its reference points documented, not a guessed total that silently double
counts age. At 28 Hz, 1 ms is 10.08 degrees. For a regularly held 2-kHz correction,
the hold contributes approximately 0.25 ms to the center of its action interval;
include that once if targeting the interval center. Actual combiner/driver holds
may differ. A signed phase-trim control can be added for measured residual plant
phase, with positive trim defined as advancing the reconstructed harmonic.
Keep this separate from physical latency in logs and configuration.

Native stream write time is not a hardware-actuation timestamp. Record and
measure sensor time, ADC publication, local receipt, command computation,
correction publication, combiner publication, and dummy-driver/physical response
as distinct events. The previously recorded deployed combiner contract is
user-supplied evidence: changed summed additive `cnt0` triggers recombination,
AveMode=0, DClevel=0, process trigger mode 4, delay 10 us. It does not establish
one-to-one delivery, constant delay, or the new correction channel's gain.

### Physical transform and calibrated DM synthesis

Let sensor polarity/channel selection be applied during decoding. Define rotation
in the numerical +X/+Y axes explicitly:

```text
R(alpha) = [[cos(alpha), -sin(alpha)],
            [sin(alpha),  cos(alpha)]]
theta_dm = R(alpha) * theta_hat_sensor(t_apply)
correction_angle = -gain * theta_dm
surface_um = (B*K) * correction_angle
```

Angles passed to the common synthesis core are radians. INDI rotation is degrees,
converted once per accepted setting. The simulator's DM axes are increasing image
column and increasing image row; a positive numerical rotation should not be
described as sky counterclockwise without a coordinate registration. A reflection
or channel swap cannot always be represented by rotation alone; keep input
channel ordering/polarity explicit and verify them before physical enable.

There are two negative signs in the physics/control chain: acceleration-to-angle
and opposing correction. With identity registration, the correction's angular
command is therefore `+gain * a_hat / (omega0^2*L)`. Test this against a known
simulated pointing sign; do not insert an additional sign or reflection factor
in the DM mapper to make an observed loop appear to work.

Reuse the exact current calibration procedure:

1. Before runtime, use `mx::fits::fitsFile<double>` and
   `mx::improc::eigenCube<double>` as in `accelSim::prepareDM`. Validate a 50x50
   cube with at least the two selected TT planes, one matching dmmask plane,
   finite full-mode pixels, valid nonnegative fitting weights, rank, accepted
   residual, and unchanged calibration-file identity across loading.
2. For matching simulation axes, supply the established entrance-pupil maps
   `x=(column-24.5)*6.5/48`, `y=(row-24.5)*6.5/48` metres. Those are the accepted
   registration inputs to the calibration fit, not permission to bypass B and
   publish a newly invented planar actuator grid.
3. Reuse `tipTiltProjection`'s column-scaled pivoted QR to fit
   `min ||sqrt(W)*(B*K - 0.5e6*[x,y])||`. Cache `response()` = B*K and allocate
   the complete float command before enabling. The simulator uses residual
   tolerance 1e-6; propose the same initial acceptance for the same calibration,
   with provenance and the configured value visible in status.
4. Synthesize the full image using the existing two-response summation order,
   preserving outside-mask pixels. dmmask weights the fit; it is not an output
   clipping mask. Retain the measured nonorthogonality and normalization of B.
   Do not add another factor of two, P2V normalization, diameter scale, AEG or
   K-mirror/offload transform, influence gain, or driver voltage conversion.

The approved cube path and measured K are recorded in `apps/accelSim/README.md`
and `agents/plans/2026-10/accelSim-dm-injection.md`. Load explicit paths rather
than hardcode those local calibration numbers. The same B/K is appropriate for
the simulated DM-pupil axes; the new R accounts for sensor registration and is
not the rotation of the TT basis itself.

**Reuse boundary:** propose moving the model-independent projector/basis to
`libMagAOX/utils/tipTiltCalibration.hpp`, exposing radians-based synthesis and
coefficient APIs. Keep `accelSimDetail` compatibility aliases/adapters and
`pointingDisturbance` in the original header so existing tests and simulator
callers retain their behavior. The controller must not construct a fake simulator
`sample`, read `m_pointing`, or include the simulator model as its runtime API.
Similarly move the existing stream adapter into
`libMagAOX/utils/dmChannel.hpp`, with an old-header compatibility alias. These
are small reusable utilities for two consumers, not a new controller base-class
framework. Confirm extraction parity with existing tests before adding the app.

### Architecture and real-time cadence

Use `MagAOXApp<true>` plus one tagged `dev::shmimMonitor` input. Use its interface
macros where provided. Add one controller worker using `MagAOXApp::threadStart`
and the usual configurable priority/cpuset. The monitor performs only validated
copying into a preallocated SPSC queue. The controller worker is the only owner
of decoding/estimator state, DM commands, zeroing, and active publication.

```text
acc000, approximately 8 kHz
  -> shmimMonitor: validate/copy raw pair and source metadata
  -> bounded SPSC records, preserving unique sample order
  -> controller worker: decode and update fit for all accepted samples
  -> absolute 2-kHz deadlines: solve, predict current application time,
     rotate/scale, synthesize, publish one complete correction
  -> ordinary app loop: INDI/status, file/inode supervision, deferred reporting
```

Why this arrangement:

- The raw target is weak relative to broadband acceleration. Keep the 8-kHz
  observations rather than selecting every fourth ADC pair without filtering.
  The harmonic statistics accumulate all accepted samples before the bounded
  output-rate fit/synthesis. Qualify off-band attenuation before reducing command
  rate; a narrowband fit is not automatically an ideal anti-alias filter.
- A 28-Hz period is about 35.7 ms. A 2-kHz publisher gives approximately 71
  commands per cycle and 0.5-ms command periods. Its approximate zero-order-hold
  phase is 2.52 degrees at 28 Hz and can be included in prediction. It avoids
  four times as many full actuator writes for this one line; this is an
  engineering hypothesis to benchmark, not a demonstrated hardware advantage.
- The separate owner continues to process disable, fault, and freshness
  deadlines while the input monitor is waiting for a vanished source. A direct
  `processImage`-only writer would leave no callback to clear held correction
  during source loss. Fixing that through the one-second app loop is insufficient.
- The monitor's latest-frame policy and the SPSC queue do not guarantee 8-kHz
  capture. Count loss at each boundary. Any retained-plane recovery is bounded
  and sequence-checked; otherwise a lost sample inhibits the first version.
  Do not present the existing monitor as a lossless ring reader.

The SPSC queue has one producer and one consumer, fixed startup capacity, and
release/acquire publication of complete records. mxlib's existing
`circularBufferIndex` is useful single-owner storage but is not established here
as a concurrent SPSC transport. Prefer an already available proven SPSC utility
if the dependency audit finds one; otherwise keep a small app-local helper with
explicit ownership and overflow tests. There is no heap growth or overwrite of
unconsumed records. Queue capacity and maximum processed batch are chosen from
measured worst-case scheduling delay, with a separate time-based freshness bound;
capacity does not authorize processing old data indefinitely.

At each absolute deadline, take a bounded snapshot of queued records, update in
order, and generate at most one current correction. There is no burst of old
actuator commands to catch up. Overrun, excessive age, backlog/overflow, or an
unrecoverable source gap causes zero/inhibit. Propose 2 ms as the first local
freshness-test setting, to be confirmed against the Linux baseline; distinguish
this data-quality deadline from a hardware displacement limit.

The correction rate is a startup setting, initially 2000 Hz; input rate and rate
source are independent settings, initially 8000 Hz for this experiment. A source
rate change resets/inhibits rather than changing the producer or silently using
wrong coefficients. Do not subscribe to `camwfs.fps` as if it were accelerometer
sampling rate. If desired later, a measured WFS-synchronous correction clock can
be added without changing the estimator input contract. No WFS semaphore or
second model clock is needed for the initial acceleration-only controller.

The worker does no filesystem I/O, FITS loading, dynamic Eigen resizing, ordinary
logging/INDI, long mutex wait, or calibration solve. Control changes arrive as
bounded immutable/versioned requests; disable/fault inhibition must remain
observable even if a configuration queue is full. Publish low-rate snapshots
without blocking the worker, following accelSim's deferred diagnostics pattern.
Measure actual native-update duration and allocation behavior; the absence of
explicit waits in the app does not prove the linked ImageStreamIO code cannot
block. If the two-thread arrangement or 2-kHz cadence fails the acceptance tests,
revisit the reader/scheduling design using the measurements, without changing
accelSim timing or silently dropping samples.

### Output channel, controls, and lifecycle

The default mode is `shadow`: read/estimate and expose/log predictions, with no
DM attachment or writes, including zeros. `output.mode=native` is an explicit
startup choice after channel handover. Only then attach/validate the configured
existing channel and publish startup zero before accepting enable. Never create,
replace, rename, or implicitly reserve a physical DM stream.

Proposed controls and diagnostics:

| Configuration / INDI interface | Default and behavior |
| --- | --- |
| `shmimMonitor.shmimName` / standard monitor status | `acc000`; expected UINT16 pair contract above. Source instance/rate subscription is separate configuration. |
| `adc.*` / read-only decoder calibration identity | Simulator-matching offsets/scales initially, explicit channels and polarities. Startup settings, not hidden online z-score normalization. |
| `output.mode`, `dm.channelName`, `dm.modeCube`, `dm.mask`, `dm.maxRelativeResidual` | Shadow, empty channel and explicit calibration paths; same initial residual tolerance 1e-6. Read-only output/calibration identity in INDI. Changes require disabled restart/revalidation. |
| `control.enabled` / `enable.toggle` | False; startup true rejected. Explicit operator request acknowledged only after the owner applies it. |
| `control.gain` / `gain.current,target` | 0; finite commissioning amplitude, proposed range [0,1] for the benchmark. Review any needed range extension as an experimental control, not an actuator safety rating. Gain zero actively clears the owned native contribution. |
| `estimator.frequency` / `frequency.current,target` | 28 Hz for default accelSim target; adjustable to the measured 28–29 Hz peak and wider tested characterization range. Must be positive and compatible with input/output Nyquist and the validated estimator band. Retune inhibits, resets fit, and requires rearm. |
| `estimator.memoryTime` / `memory_time.current,target` | Proposed total mean memory 0.1 s, implemented as two 0.05-s stages; tune from characterization. Changes reset readiness rather than mixing old/new windows. |
| `registration.rotation` / `rotation.current,target` | 0 degrees for matching simulator axes, defined by R above. Disabled-only configuration revision before rearm. |
| `physics.leverArm` / `lever_arm.current,target` | 10 m, explicitly labelled a simulation starting value; finite and positive. Disabled-only change before rearm. |
| `prediction.horizon` / `prediction_horizon.current,target` | 0 ms downstream delay until calibrated; includes only the documented downstream component. Native enable requires reviewed timing settings, even if their accepted value is zero in a private fixture. |
| `prediction.phaseTrim` / `phase_trim.current,target` | Optional signed degrees, initially 0; measured plant/estimator trim with documented advance sign. Distinct from time delay. Apply in a versioned disabled/rearm transition initially. |
| `control.outputFps`, `input.expectedFps`, freshness and queue settings | Startup settings: proposed output 2000 Hz, simulation input 8000 Hz. Report measured rates, actual age/horizon, queue high-water mark, source and command counters. |
| `status` / read-only numeric flags plus state/reason text | Source/calibration/output readiness, fit warmup/conditioning, enabled requested/applied revision, active/shadow, fault reason, source generation/counter, sample age, last source time, last publication, zero pending/failed. |
| `estimate` / low-rate read-only values | Harmonic acceleration amplitude/phase per axis, pointing in explicitly labelled radians or mas, fitted DC, fit residual/noise diagnostic. Readiness means a valid fit/timebase, not demonstrated WFS cancellation. |
| `reset` request | Disabled-only reset of fit/fault after readiness checks; never automatically enables or changes gain. |

Use the repository's `CREATE_REG_INDI_*` / standard helpers and
`INDI_NEWCALLBACK_DECL/DEFN` patterns. Callbacks validate finite values and ranges,
serialize app-owned property updates, and enqueue requests; they never touch
the image or mutable estimator. Changes involving several values are committed
as one revision. Avoid a non-lock-free atomic configuration struct or callback
write races. The owner reports actual applied settings, with Busy until applied
and Alert on rejection/fault; INDI updates remain at ordinary application cadence.

Required state behavior:

1. **Observe/warm up:** source observation is independent of correction. With
   shadow/default configuration, perform no DM operations. In owned native mode,
   attach and zero before processing any enable. Gain zero does not skip input
   validation or establish ownership by itself.
2. **Enable:** require a fresh coherent source, warmed/conditioned fit, matching
   calibration/control/source generations, validated output, accepted timing,
   and an explicit request. Gain may remain zero; distinguish enabled-zero from
   nonzero correction in status. No replay of a command made before enable.
3. **Disable/gain zero/fault:** the sole owner inhibits old-generation commands,
   publishes a complete active zero, and acknowledges completion. Keep observing
   if healthy, but do not hold or replay the last nonzero image. Retry a failed
   zero only on the same validated mapping, bounded by worker service/reporting;
   keep fault/zero-pending status rather than claiming success. No needless
   high-rate zero traffic after a successful clear.
4. **Source silence:** the controller worker checks monotonic age at its own
   deadlines, regardless of input callbacks. After the accepted freshness bound,
   zero/inhibit. A duplicate post or old frame cannot extend the timeout. A
   prediction oscillator must not free-run indefinitely after measurement loss.
5. **Mapping/calibration change:** app supervision can invalidate, never switch
   an active pointer underneath the owner. A changed inode or calibration requires
   zero if still possible, new validation, and explicit rearm; never automatically
   zero a newly created stream that might now have a different owner.
6. **Shutdown:** inhibit, stop/join the input monitor, finish/cancel queued work,
   and obtain the final zero from the owner before releasing the output mapping.
   A designated shutdown thread may take over only after joining the writer.
   Report bounded final-zero failures. No callback or older revision can publish
   nonzero after disable/shutdown acknowledgement.

Exclusive ownership includes startup/final zeros, bridge writers, scheduled
writers, and late callbacks from the previous owner. Disabled does not transfer
the channel back. No `zeroAll`, automatic process stop/restart, or writes to
`dm01disp10`, the combined image, flat, or an AO channel are permitted. A crash,
SIGKILL, host loss, or failed combiner/driver is outside a clean-shutdown guarantee;
an owner-approved external cleanup procedure is still needed.

### Simulator injection scale and physical plant

The ADC stream describes the model's full physical acceleration even when
accelSim injection is disabled or `dm.scale` is zero. `dm.scale` multiplies only
the injected DM contribution. It is not encoded in `acc000` and must not be read
from simulator internals by the controller.

For identical calibration and unity combiner channel gains, if injection scale
is s and the controller estimate is exact, the residual is approximately
`s*theta - g*theta_hat`. Cancellation therefore occurs at g=s, not automatically
g=1. With unequal combiner gains, account for their ratio. Set a known nonzero
injection scale in the test description and sweep controller gain relative to it.
Raw acceleration may continue while optical injection is off; blindly enabling
the correction then creates optical motion rather than cancelling an injected
disturbance. This is a commissioning state/configuration issue, not evidence for
changing the simulator or inventing an automatic hidden gain.

The current injection uses full `m_pointing`, not the model's legacy
`m_wfsPointing` attenuation. Offline comparison must use the injected physical
pointing/calibrated surface; an actual WFS measures its optical/closed-loop
response. Do not multiply this controller by the notebook's per-mode WFS coupling
or the telescope-secondary AEG/offload calibration.

## Questions and Points of Clarification

These are review inputs, not requests to run hardware commands through an agent.
The proposed local numerical/shadow implementation can proceed after plan review;
native or physical enable needs the applicable operational answers.

| Question | Proposed position / required evidence |
| --- | --- |
| Application and cadence | Approve a separate `accelFFCtrl` benchmark, full-rate input statistics, and initial 2-kHz correction output. If an 8-kHz correction or WFS-phase-synchronous output is specifically required, state that independently of the existing 8-kHz injection requirement. |
| Shared-helper extraction | Approve a minimal behavior-preserving promotion of the numerical TT mapper and stream adapter, keeping the simulator's adapters and public tests. No broad DDSPC merge/refactor is proposed. |
| Which `loPredCtrlAcc` version is authoritative? | Local refs above differ and neither is checked out here. Supply the intended deployed source revision and consumer configuration before future compatibility or DDSPC integration claims. |
| Correction channel and writer handover | Owner must assign an existing non-10 tweeter channel and identify all current/automatic writers. Confirm gains/enables, `deltaChannels` classification, and handover/return-of-ownership procedure. No channel number is chosen by this plan. |
| Real ADC calibration | Supply per-channel zero point, effective volts/count and sensitivity units, analog-chain scaling, channel labels/polarities, clipping conventions, sampling configuration, and any input-filter response. Simulator constants are not proof of the hardware chain. |
| Sensor-to-optical registration | Confirm whether a positive-determinant 2-D rotation and L suffice near 28–29 Hz, or whether gain asymmetry, reflection, structural coupling, and per-axis complex phase require a measured 2x2 transfer. |
| Frequency/tracking requirement | Is the peak stationary enough for operator-tuned f0, and what detuning range must retain specified suppression? Characterize T before choosing its default or promising rejection over all 28–29 Hz. |
| Timing contract | Obtain actual source timestamp meaning/mode, X/Y readout skew, relevant clock domains, combiner/driver timing, application/hold interval, and acceptable age/jitter/zero deadlines. Resolve hardware timer-mode missing timestamps before using them. |
| Simulation operating point | Record injection enable/scale, ADC rate, synchronization reference, correction gain range, and AO-loop state for each test. Recommended initial cancellation sweep is around g=s for unity combiner gains. |
| Acceptance target | Proposed numerical gates below include a 10-dB target-band PSD reduction in a well-matched benchmark. Agree the detuning envelope, off-band amplification allowance, and timing/freshness policy before accepting native/physical operation. |
| Fault and recovery policy | Confirm initial strict inhibit/reset on unrecoverable input gaps and explicit rearm after source/calibration changes; define saturation persistence and external held-command recovery. No extra displacement bounds are requested. |

## Tests

All runtime tests must use private names/namespaces, synthetic images, and no
hardware driver/bridge. Numerical tests should run with no MagAO-X runtime
dependencies beyond the existing Eigen/Catch setup. Disable fast-math locally
to preserve finite checks and estimator arithmetic. New application tests follow
`namespace libXWCTest { namespace accelFFCtrlTest { ... } }`, an
`accelFFCtrl_unit_test` group under `application_unit_test` in `tests/groups.dox`,
per-TEST_CASE Doxygen blocks, and explicit real-symbol Doxygen references for
test harness indirection per `tests/testXWC.hpp` and `AGENTS.md`.

### Numerical and interface tests to implement

| Test | Purpose and expected evidence |
| --- | --- |
| Decoder round trip | Generate codes with production `adcCounts` only in the test fixture. Verify exact zero/positive/negative code conversion, per-axis calibration, channel selection/polarity, rounding half-count uncertainty, no unsigned underflow, and rejection of >4095 or wrong input type/shape. Treat legal rails separately from proven clipping. |
| Quantized weak tone | Exercise the default sub-count 28-Hz amplitude with and without noise/dither, with DC drift and both axes. Compare against a floating-acceleration oracle and quantify amplitude/phase bias and uncertainty. Do not test only easy many-count sinusoids. |
| Harmonic reconstruction | Pure 28, 28.5, and 29-Hz sinusoids with positive/negative and quadrature phases, zero signal, unequal X/Y amplitudes, and known offsets. At matched f0, after warmup require proposed <1% amplitude and <1 degree phase error for noiseless adequately resolved fixtures; separate tolerances for quantized/noisy cases. Compare the recursive statistics to an independently assembled weighted QR batch fit. |
| Frequency and envelope response | Sweep actual frequency around configured f0, including ±0.1, ±0.5, ±1, and ±2 Hz; vary T and amplitude/phase envelopes. Publish gain/phase/settling curves, including inability to predict stochastic innovations or cure detuning with one latency number. |
| Broadband rejection and command aliasing | Include 11–25, 54/56/58, 75/85, 120/220/300 Hz, weak 100–900 Hz population, DC, sensor noise, and higher frequencies up to input Nyquist. Measure transfer into the 2-kHz correction spectrum. If envelope/command anti-alias attenuation is inadequate, add a characterized filter or revise the output rate before enabling; never unfiltered-decimate raw ADC pairs. |
| Prediction/phase sign | Inject known acquisition, transport, queue, computation, application, and hold delays independently. Verify positive prediction advances the harmonic, the acceleration inversion and control sign cancel correctly, and measured sample age is counted once. Test clock steps, future/stale/zero timestamps, and sequential-axis skew. |
| Calibration parity | For the same supplied modes/mask/geometry and radians, compare shared-helper responses, modal coefficients, and synthesized float images against the unchanged existing numerical API. Use positive/negative X/Y and mixed angles, nonorthogonal/swapped/sign-flipped bases, and outside-mask pixels. Require bitwise equality where summation order is unchanged; justify any boundary conversion tolerance explicitly. Reuse existing projection/rank/residual/overflow tests. |
| Rotation and gain | Check 0/90/180-degree and general rotations against independent matrix multiplication; verify radians/arcsec/mas boundaries and two-negative-sign correction. Test s=0, 0.1, 1 injection scales offline and g=0, matched, and mismatched gain without modifying simulator amplitudes. |
| Long-run estimator integrity | Many memory time constants, rate/target changes, irregular intervals, phase wrapping and oscillator drift, ill-conditioned startup, finite/overflow checks, reset, memory stability, and source-generation separation. Ensure numerical failure yields a fault/zero result without throwing allocating exceptions in the normal worker path. |
| Lifecycle and concurrency | Disabled startup/gain zero, shadow no-DM access, missing/invalid calibration, rejected startup enable, enable during warmup, stale revisions, simultaneous config/disable requests, zero retry, source-loss timeout with no callbacks, queue overflow, gain-zero while active, orderly shutdown, failed publisher, and stream recreation. Assert no older nonzero write after zero acknowledgement. |
| Raw stream recovery | Depth 1 and >1 wraps, 2-D compatibility, wrong datatype, out-of-range index, duplicate semaphores/counters, backwards counters, recoverable retained frames versus overwritten gaps, write races, timestamp/plane mismatch, source replacement, and input shutdown/reconnect. Count each unique sample exactly once or explicitly fault; never manufacture samples. |

### Validation stages and metrics

1. **Offline physics first:** add a separate replay/export fixture using the
   unchanged simulator model at specified 6000/8000-Hz rates and seeds. Give the
   controller only raw codes and timestamp/counter records. Store physical
   pointing, injected surfaces, and noise-free target information in a separate
   oracle used only for scoring. The existing hidden `[accelSim-export]` test
   and `tests/validate_psd.py` provide export/PSD conventions; its default 60-s
   export is 6000 Hz, not proof of 8-kHz timing. New fixtures may select a single
   mode without adding a production simulator mode or accessing truth in the
   controller. Also replay the complete existing 214-mode population.
2. **Shadow app:** run the controller against private replay and then private
   accelSim `acc000`, with no DM output configured. Compare frequency, amplitude,
   phase, timestamp alignment, fit warmup, CPU, and missing-frame accounting.
   A simulation oracle can be captured separately; do not publish or consume a
   hidden runtime ground-truth stream for feedback. Verify ADC/model outputs
   and source timing are unaffected by the consumer's presence within the
   measured scheduling baseline.
3. **Private native integration:** use accelSim injection and a separate
   correction channel inside a fresh private ImageStreamIO namespace, connected
   only to a dummy combiner/driver implementing the recorded contract. Use the
   same B/K for injection and correction, explicit s/g, independent clocks,
   publication delays, zero-order holds, concurrent synthetic AO updates, and
   combiner coalescing. Verify full-image active counters/semaphores, metadata,
   source identity, gain/sign, final-zero application during AO inactivity, and
   preservation of all other channels. Native tests must validate their resolved
   private backing paths before creating images; run separately when namespace
   selection is cached by ImageStreamIO.
4. **Sustained Linux development test:** measure at least ten minutes of 8-kHz
   input plus proposed 2-kHz output with correction off/shadow/on to dummy output.
   Record mean, percentiles, maximum, and histograms for read/fit/synthesis/native
   update, source age, application skew and missed deadlines; RSS, allocations,
   queue high-water mark, source/consumer/combiner/dummy-driver counts, and source
   publication jitter/skips. Proposed nominal criteria: mean rates within 1%,
   no unexplained input gaps or duplicate updates, no unreported command skips,
   bounded RSS, zero hot-path C++ allocations after warmup, and no service-time
   overrun of the 500-us output period under the qualified load. Set tail/headroom
   and stale-to-zero limits from the measured baseline before acceptance. Test
   overload intentionally and verify bounded zero/inhibit rather than catch-up.
5. **User-operated optical commissioning only after development acceptance:**
   establish exclusive correction-channel ownership, correct calibration and
   source identity, and measured plant timing/axis registration. Start with
   correction disabled/gain zero; verify the signs/rotation at low reviewed
   amplitude. Sweep gain relative to injection scale and horizon/phase near the
   measured transfer. Record WFS TT with both commands, hardware/combiner timing,
   and AO state. Repeat source loss, controlled disable, and stop-zero during AO
   inactivity. Agents do not start processes or execute INDI on instrument hosts.

For spectral scoring, use matched durations, seeds/ensembles where applicable,
sampling/hold models, Welch window/segment length, and timestamp alignment.
Propose a documented 27–30-Hz target band for the initial 28–29-Hz characterization,
with resolution sufficient to separate the 25-Hz neighbor. Report integrated
target-band power, peak PSD, residual TT RMS, off-band changes, coherence, and
command/measurement cross-phase per axis. Define suppression as
`10*log10(P_off/P_on)`; compare target-band quantities rather than demanding that
a one-line controller remove broadband pointing RMS.

Proposed benchmark gate: at matched frequency and calibrated delay/gain, achieve
at least 10 dB reduction of the selected target-band power in the full quantized
simulation, across agreed seeds after warmup, without more than the reviewed
off-band amplification allowance (initial proposal 0.5 dB in specified adjacent
bands). This is a target to validate, not a measured result. Publish detuning and
noise envelopes even where that target fails. In a noiseless single-tone unit
fixture, residual should follow the analytical gain/phase error:
`residual_amplitude / disturbance_amplitude = |1 - g*exp(j*phase_error)|`
after normalizing g to the injected amplitude/channel gains.

Actual WFS residual depends on the AO rejection transfer and optical response;
it need not equal commanded surface or the simulator's legacy WFS-coupled state.
Establish feedforward signs/timing under a documented AO condition, then measure
performance with the intended AO loop. No physical suppression or stability is
claimed by a numerical/dummy-driver pass.

Regression scope for future implementation: rerun all existing accelSim
model/synchronization/projection/DM runtime suites, PSD tests, and their sanitizer
variants; add shared-helper and new-controller suites plus private native tests
on an appropriate host. Add ThreadSanitizer where supported for the new transport
and control handoff, and instrument allocation paths separately from timing.
Existing docs record macOS runtime build blockers (`timegm` and missing
`mx/math/floatUtils.hpp`); do not claim a current full app/native pass or expand
this feature into a platform-dependency repair without a separate reason.

## Implementation Plan

Do not execute these steps until the user reviews this proposal and resolves the
applicable choices. Proposed filenames below are new work, not existing APIs.

| Step | Planned files and concrete result |
| --- | --- |
| 1. Freeze contracts and acceptance | Update this plan with review decisions: application/cadence, decoder units, estimator T/frequency band, shared extraction, liveness and gap policy, and numerical acceptance. Record separate deployment blockers; do not invent a channel or real sensor calibration. |
| 2. Promote reusable numerical/output helpers | New `libMagAOX/utils/tipTiltCalibration.hpp` and `libMagAOX/utils/dmChannel.hpp`; compatibility adapters in `apps/accelSim/accelSimDisturbance.hpp` and `accelSimDMStream.hpp`; necessary header dependency entries in `apps/accelSim/Makefile` and `tests/Makefile.one`. Preserve projector/synthesis/native-publication behavior and existing call sites. Keep simulator model adapters outside neutral utilities. Add parity coverage before controller work. |
| 3. Implement pure decoder/predictor | New `apps/accelFFCtrl/accelFFCtrlModel.hpp` and `tests/accelFFCtrlModel_test.cpp`, names finalized at implementation review. Fixed raw/physical/prediction records, decoder, harmonic statistics, readiness, timestamp-based prediction, rotation/gain, and status-returning error paths. No ImageStreamIO, INDI, WFS dependency, or simulator model state in the production numerical core. |
| 4. Add independent offline assessment | New `apps/accelFFCtrl/tests/validate_feedforward.py` and a numerical replay/export fixture in the new tests. Raw-only controller input; separately retained simulator/analytical oracle. Characterize quantization, memory/detuning response, phase/delay/hold, full-population rejection, and gain-versus-injection scale. Select reviewed defaults before native correction work. |
| 5. Add the application in shadow mode | New `apps/accelFFCtrl/accelFFCtrl.hpp`, main-only `.cpp`, `Makefile`, `README.md`, `config/example.conf`, and an app-local stream/queue helper if needed. Reuse `MagAOXApp`, `shmimMonitor` macros, normal `threadStart`, fixed bounded handoff, owner worker, source-generation checks, and coherent deferred status. Default DM access absent. No shared monitor/frameGrabber algorithm change is planned. |
| 6. Add native correction lifecycle | Connect the common mapper/output helper to the owner; add explicit shadow/native startup choice, ownership/output validation, versioned INDI requests, warmup/readiness, numerical/freshness faults, zero acknowledgements/retry, and shutdown ownership transfer. Test against synthetic existing channels before any commissioning request. |
| 7. Register and document | Register the new executable in the appropriate top-level `Makefile` app list after choosing its deployment/build grouping; tests in `tests/tests.list`, dependencies/no-fast-math in `tests/Makefile.one`, and `application_unit_test` grouping in `tests/groups.dox`. Document calibration, units, phase signs, sample/time provenance, gain/scale relation, source-loss policy, private test commands and results. No instrument startup/process configuration is added. |
| 8. Validate and review | Run numerical, existing-regression, sanitizer and private Linux/native/load stages. Record commands, dependencies, results/failures, and measured limitations here. Run clang-format on touched C++ with exact `//mutex scope` preserved, full changed-file documentation checks, and `git diff --check`. Leave all changes uncommitted/unpushed for review unless later explicitly instructed otherwise. |

Use repository include guards, `m_` naming, Doxygen file/function/parameter/member
documentation, and out-of-class inline definitions in header-only application
style. Remove `\author` from any touched C++ file. Keep protected data sections
before public accessor sections. No style ambiguity blocks this plan; the exact
new utility/API names are proposed design choices requiring review, not an
established existing naming convention to silently impose on unrelated apps.

### Path to future DDSPC

Keep sensor acquisition/decoding, temporal transport, predictor, coordinate/unit
adapter, calibrated synthesis, and output lifecycle as explicit boundaries.
The initial backend accepts timestamped physical-acceleration records and returns
a two-axis physical pointing prediction in radians for a requested application
time, plus validity, source/calibration/control generation and coverage. It does
not own the DM writer, INDI, enable state, or hardware channel. Select the backend
outside the worker's sample loop rather than constructing/deleting polymorphic
objects during operation.

Retain raw codes and decoder calibration identity in transport/recording so a
future learned model can choose raw or calibrated features deliberately. DDSPC
does not need to assume the physics-based lever arm; a learned predictor can
target calibrated pointing directly. A model trained in TT coefficients needs
an explicit calibrated adapter or a reviewed common output-space extension,
not another implicit omega-squared conversion or second gain/sign application.

Future work can reuse the local `loPredCtrlAcc` branch's accelerometer-history
interfaces and `DDSPC` algorithms after correcting raw decoding, timestamp/freshness
handling, and rate alignment. A WFS/DDSPC extension must align accelerometer
history to WFS acquisition/exposure times and command history at the chosen
control cadence. A WFS frame receiving one latest accelerometer sample is not
equivalent to the preceding four 8-kHz samples. Record produced command history
and observed application evidence separately; `cnt0` is not a hardware ACK.

Online normalization, history lengths, training/validation intervals, exploration,
regularization, and model updates are backend-specific. Learning/solves and
file load/save must be kept out of the qualified real-time writer; stage immutable
preallocated model revisions at disabled transitions. No automatic legacy
feedback fallback, exploratory actuation, or simultaneous independent controller
writers are enabled by the benchmark. A failure always uses the same zero/inhibit
contract, regardless of predictor choice.

## Follow-up and Edge Cases

- A structural resonance can drift, split, or have different X/Y transfer phase.
  One f0, one L, and one rotation may fail; publish that failure envelope rather
  than fitting a hardware safety limit or hidden empirical gain into the mapper.
- High-Q stochastic excitation cannot be predicted perfectly. Window tracking,
  measurement noise, quantization, phase jitter, and a finite hold set a residual
  floor even with a correct rigid-body sign and calibration.
- Default depth-one source buffers and latest-plane monitor callbacks can lose
  measurements during scheduling stalls. The new queue fixes neither producer
  overwrite nor missing acquisition stamps. Validate each boundary separately.
- Missing/zero or stepped timestamps, model restarts, source rate changes,
  remote clock offsets, counter wrap/reset, and channel readout skew are explicit
  readiness/fault cases. Do not hide them by synthesizing sample times from
  callback count.
- Input silence and output/combiner failure are different faults. A successful
  image zero is not proof of optical zero, and an invalid/replaced mapping may
  be impossible to clear. Keep operator-visible status and external recovery.
- Real accelerometers may respond to control-induced motion or electrical pickup.
  A nominal feedforward path can then contain feedback. Assess measured
  sensor-to-command/WFS coupling before increasing gain or adding learning.
- The existing AO loop can reject or interact with the injected/correcting
  disturbance. Simulation cancellation in additive surfaces alone is not an
  on-sky closed-loop stability or performance claim.
- No general broadband controller, automatic frequency tracking, telescope AEG/
  AzEl/K-mirror registration, new hardware limits, deployed process changes,
  DDSPC branch merge, or live correction is included in version one.

### Investigation verification and stop point

Read the source and local Git objects listed above; derived the ADC/angle and
phase-per-millisecond numerical checks with a local standard-library calculation.
Consulted primary Eigen/recursive-estimation references for the proposed solver
and weighting. No new controller, helper extraction, test, configuration, stream,
or runtime implementation was created. No existing suite was rerun for this
Markdown-only investigation; all validation stages here are proposed work.

Only this plan's Agent Findings and Plan section is changed. The 10,142-byte
protected prefix matches the pre-edit snapshot byte-for-byte (SHA256
`f03f6af78d1ed5091e4a90feed004628564c6e6b3bed4a2355633af1688890a0`).
`git diff --check` and a whitespace check of the untracked plan against that
snapshot passed. No C++ files were touched, so clang-format does not apply.
Stop here for user review; no commits, pushes, branch changes, or implementation.
