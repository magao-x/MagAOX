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

**Live observation phase:** the user has authorized the non-actuating MagAO-X
application, live raw-stream ingestion, INDI configuration/diagnostics and tests.
The phase-two record below supersedes the earlier numerical-only boundary for
that scope. No DM reader/writer, calibration helper extraction or actuation is
included. The committed numerical estimator and its existing tests stay unchanged.

**Phase-one authorization:** the user has reviewed this investigation and
authorized only the numerical raw-ADC acceleration-to-pointing estimator and
offline tests/benchmark. The numerical design review and execution record at
the end supersede the earlier implementation stop for that scope alone. No
application, ImageStreamIO reader/writer, DM-helper extraction, INDI, or physical
stream access is authorized or needed in this phase.

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

## Phase one: numerical design review and implementation record

### Decisions resolved before implementation

- Implement only `apps/accelFFCtrl/accelFFCtrlModel.hpp` and numerical validation
  support. Leave the simulator and its mapping/publication unchanged. The core
  depends on standard C++ and fixed-size Eigen, not the simulator or MagAO-X
  runtime. Separate replay tests may read the unchanged model's truth for scoring.
- Default target frequency is **28.5 Hz**, as requested for this implementation;
  the simulator's default target remains 28 Hz. Replay tests must configure the
  estimator to the scenario rather than silently changing that model.
- Preserve the planned two-stage positive window. T is total approximate mean
  memory, each stage has time constant T/2, and both stages filter **both** the
  Gram matrix and data moments. Filtering only the data or treating the window
  as ordinary single-exponential RLS would change the fitted amplitude/phase.
  Use the newly updated first-stage value for the second stage in the same tick.
  Compute `1-lambda` with `-expm1(-2*dt/T)` to avoid cancellation for short intervals.
- Fit a nuisance DC plus cosine/sine per sensor channel. Return sensor-channel
  peak acceleration amplitudes [m/s²] and phases relative to the latest sample
  time, predicted acceleration at the requested target, and rotated physical
  pointing [radians]. Exclude DC from vibration. Do not apply correction gain,
  opposing DM sign, or any actuator/P2V limit in this estimator.
- Require callers to supply ordered raw pairs, source generation/counter, sample
  timestamp and current time as nonnegative integer nanoseconds in the **same
  monotonic clock domain**. A numerical epoch of zero is valid; conversion of
  real stream epoch stamps/missing metadata belongs to a future input adapter.
  Consume each accepted unique record once; duplicates do not refresh freshness.
  Gaps, rollback, generation changes, invalid codes, invalid or excessive time
  intervals, and stale input latch a fault until reset/reconfigure. The core
  cannot recover overwritten samples and does not fabricate them.
- Use explicit `predict(now, target)` plus a configurable `predict(now)` horizon
  measured from now. The carrier advances by `target - last_sample_time`, already
  including age. Check sample freshness on prediction even when no new records
  arrive. Initially use configurable 2-ms age/interval and 50-ms maximum forward
  prediction bounds as numerical validity policies, not physical safety limits.
  A bad prediction request returns an invalid, zero-valued result; an input fault
  inhibits future valid results until reset. No clock, file, or stream is read.
- The first sample uses the configured nominal interval for initial window
  weight. Later updates use actual sample intervals; acceptable irregular timing
  is supported rather than forcing uniform spacing or estimating time from the
  counter. Reconfiguring frequency, memory, decoding, or rotation resets the fit.
- Preserve `max(3/f0, 3*T)` warmup plus finite/positive/condition/residual checks
  on a fixed 3x3 LDLT solve. This prevents startup numerical regularization from
  manufacturing a ready result. All per-sample state and prediction storage are
  fixed size; benchmark ingestion and prediction separately, including allocation
  checks and the expected 8-kHz-input/2-kHz-query pattern.
- Distinguish matched-frequency accuracy from detuned tracking. Validate the
  actual fitted response rather than demanding zero phase at the wrong f0.
  The `(f_actual/f0)^2` conversion bias is additional to window response. The
  0.1-s memory is a tunable starting point, not a blanket 28–29-Hz suppression
  guarantee. ADC sub-count accuracy likewise depends on noise/dither.
- Measure unwanted-frequency transmission and noise-only output as estimates,
  separately from hypothetical cancellation residuals. A bounded off-band
  estimate does not establish the earlier full-controller 0.5-dB adjacent-band
  correction gate; a causal estimate's phase can worsen a neighboring line if
  subtracted. Report that limitation before future publication/control work.

Implementation, test results, processing costs, and remaining limits are recorded
below as this numerical phase progresses. No commit or push is permitted.

### Validation-driven qualification of the original performance proposal

The first raw-only replay exposed a material inconsistency in treating the
earlier proposed **10-dB full-controller gate** as an unconditional estimator
correctness assertion for every full-population seed/axis. Independent analytic
and weighted-QR tests pass, but complete-model target-band SNR and the effective
acceleration-to-pointing transfer differ from the isolated harmonic assumption.
`accelerationFactor(f_m,L)` multiplies each stochastic mode by its own nominal
frequency squared; its spectral tails at f0, plus sensor noise, do not become
the physical pointing sum through multiplication by `-1/omega0²/L`.

Characterization uses 12-s recordings, scores the common final 8 s after all
tested warmups, and retains both axes/seeds 42, 123 and 701. At default T=0.1 s,
isolated 28.5-Hz stochastic-mode replay gives **13.59–23.25 dB** of 27–30-Hz
offline residual reduction; the unchanged 214-mode population, matched at 28 Hz,
gives **0.49–11.23 dB**. Sweeping total memory to 0.3 and 0.8 s improves some
full-population cases but worsens tracking in others; it does not establish a
universal 10-dB result. Keep the planned 0.1-s default and expose memory rather
than secretly selecting a different setting per seed or accessing model truth.

The numerical regression now checks calibrated analytical accuracy and exact
weighted-fit equivalence, strict input/freshness integrity, noise/off-band
attenuation, positive target-band benefit for the explicitly tested recordings,
and >10 dB for the qualified isolated/default-window replay. The earlier
full-controller gate remains **unmet and outstanding**, not redefined as passed.
At the default window, the separate off-band sweep also measures approximately
0.765 dB amplification of a 25-Hz line if this estimate were naively subtracted;
the estimator's unwanted-pointing transmission itself is below unity. This also
leaves the proposed 0.5-dB adjacent-band controller gate unresolved. No physical
writer or automatic correction is justified by the phase-one numerical pass.

### Delivered numerical interface and scope

- `apps/accelFFCtrl/accelFFCtrlModel.hpp`: fixed-size calibration/configuration,
  `adcSample`, `prediction`, fault/readiness states and `harmonicEstimator`.
  `consume(sample,now)` processes every unique ordered record; `predict(now,target)`
  and `predict(now)` return timestamped/source-identified results. Input faults
  latch until reset; invalid target requests clear their output without
  destroying a valid fit. Prediction checks age without needing another sample.
  All times are already mapped integer monotonic nanoseconds; no stream reader
  or realtime-clock mapper is implemented. Nominal sample rate initializes the
  first window weight and validates settings, while actual accepted intervals
  drive the fit. Detecting deployment-level cadence changes is future adapter work.
- `apps/accelFFCtrl/tests/accelFFCtrlModel_test.cpp`: independent analytic/raw-ADC
  fixtures, a weighted batch-QR oracle (including the exact variable-interval
  two-stage weights), frequency/noise/off-band characterization and integrity tests.
- `apps/accelFFCtrl/tests/accelFFCtrlReplay_test.cpp`: separate offline recorder
  and scorer using unchanged accelSim source. The core receives no `m_pointing`,
  acceleration truth, model object or extra RNG advance. Both default and
  nondefault ADC calibration are exercised by the analytic/replay suites.
- `apps/accelFFCtrl/tests/accelFFCtrl_benchmark.cpp`: prepared fixtures, clocked
  consumption/prediction and C++ allocation counting with mandatory Eigen
  runtime malloc checking. Healthy processing, reset, invalid-input and stale
  result paths are covered while allocation checking is active.
- `apps/accelFFCtrl/README.md`, `tests/Makefile`, and repository test registration
  in `tests/Makefile.one`, `tests/tests.list`, `tests/groups.dox`: standalone
  numerical build/run instructions and application-unit-test grouping. No app
  executable, production Makefile registration, INDI, DM utilities/refactor,
  ImageStreamIO dependency, or physical writer was added.

The implementation uses wrapped timestamp-based phase and one shared
`sin/cos/expm1` update per accepted pair, rather than an oscillator recurrence.
The measured cost leaves ample numerical headroom, and direct phase evaluation
avoids recurrence normalization/drift machinery for this stage. Query-time
trigonometry is also shared across the two axes. Configuration revalidation is
cold; ordinary sample/query errors use status returns and cleared fixed records.

### Final local validation

| Check | Result |
| --- | --- |
| Standalone numerical/replay suite, C++20/O3/no fast-math | **10 cases / 3,335,597 assertions passed.** Includes all available-record ingestion, queried at 2 kHz; matched amplitude within 0.1% and phase within 0.001 rad for fine-calibration fixtures; exact weighted-QR comparison; noise, quantization, detuning, rotation, prediction, conditioning and clock/source faults. |
| Existing unchanged accelSim model/synchronization/projection/DM numerical suites, O3/no fast-math | **34 cases / 236,598 assertions passed.** Compiled with `ACCELSIM_MODEL_ONLY`; no streams accessed. |
| Combined new and existing suites, O1/debug, AddressSanitizer + UndefinedBehaviorSanitizer, halt-on-error | **44 cases / 3,572,195 assertions passed; no findings.** |
| Existing Python PSD suite | **4 tests passed.** |
| Default calibrated weak target with sensor noise, seeds 42/123/701 | Peak gain approximately 1.0028–1.0104, phase error approximately -0.08 to -0.43 degrees; relative RMS error 11.9–12.4%. The same undithered sub-count sinusoid produces zero codes after offset removal and cannot be recovered. |
| White sensor noise, default quantizer | Estimated acceleration RMS / input sensor-noise RMS **0.03586**, approximately 28.9 dB attenuation in power ratio. No white-noise amplification was observed. |
| Pure off-target tones, 11–3500 Hz, default target 28.5 Hz | Physical-pointing RMS transmission below 0.35 across the tested tones; below 0.061 at 54 Hz and above. No off-target estimate exceeds its input's physical pointing amplitude in these fixtures. Hypothetical subtraction remains subject to the 25-Hz caveat above. |
| Full-model performance qualification | Default-window full-population 27–30-Hz offline reduction **0.49–11.23 dB**. The proposed universal 10-dB full-controller gate remains unmet. No physical suppression claim. |
| Formatting/documentation/preservation | Four new C++ files clang-formatted; header declarations/parameters/members and each TEST_CASE documented; whitespace and protected-prefix/source-preservation checks completed. |

Reproduction from the repository root:

```sh
make -C apps/accelFFCtrl/tests test
make -C apps/accelFFCtrl/tests benchmark
python -m unittest discover -s apps/accelSim/tests -p test_validate_psd.py
```

The standalone Makefile places binaries in `/tmp/accelFFCtrl-numerics`. Its
README includes a sanitizer build command. The combined sanitizer run compiled
both new test files, all three existing accelSim test files, and `tests/testMain.cpp`
with `-DACCELSIM_MODEL_ONLY -fsanitize=address,undefined -fno-omit-frame-pointer`
and the existing Eigen include path; filter `'[accelFFCtrl],[accelSim]'`.
Compiler diagnostic flags were `-Wall -Wextra`; no new build diagnostics occurred.

### Measured processing cost

Unloaded local Darwin/arm64 workstation, O3/no fast-math; 800,000 prepared source
pairs per pattern after 8,000 warmup pairs. Clock calls/resolution and loop
overhead are included. These are numerical throughput measurements, not actual
8-kHz waking/scheduling, input acquisition, stream/native publication, or DM
application latency.

| Input / query cadence | Mean consume | Mean predict | Mean complete work per source sample | Allocation result |
| --- | --- | --- | --- | --- |
| 8000 / 2000 Hz | 37.3 ns | 309.7 ns | **138.8 ns (0.139 us)** | Zero C++ new calls; Eigen malloc guard remained satisfied, including reset/fault paths. |
| 8000 / 8000 Hz | 40.0 ns | 296.2 ns | **359.7 ns (0.360 us)** | Same zero-allocation result. |

For the 8k/2k pattern, p99 consume/predict were 83/417 ns and observed maxima
were 20.875/34.209 us. Individual short measurements are quantized by the clock;
OS outliers and hardware/compiler differences prevent interpreting these as
hard worst-case bounds. No real-time process or instrument benchmark was run.

### Remaining limits and final boundary

Frequency and memory still require tuning: half-Hz detuning at the default window
has about 17.9-degree acceleration phase error plus the frequency-squared
conversion bias. Input quantization/dither, stochastic innovation and finite
target SNR bound reconstruction. The multi-mode and neighboring-line controller
acceptance gates described above remain open. Real clock mapping, coherent stream
recovery, rate supervision, software ownership/enable/zero publication, plant
calibration and actual latency are deferred to later reviewed phases.

The original task prefix above Agent Findings and Plan remains byte-for-byte
unchanged. All five accelSim implementation/model/mapping/stream headers retain
their pre-phase contents; other applications were not changed. No instrument or
physical stream was accessed. The agent did not stage, commit, or push changes.

## Phase two: live non-actuating application

### Input API findings and decisions before implementation

- Rechecked `dev::shmimMonitor::smThreadExec`, `dev::frameGrabber::fgThreadExec`,
  native attachment/semaphore patterns and the upstream ImageStreamIO API.
  The generic monitor delivers only the latest `cnt1` plane on each wake and
  has a one-second timeout. It cannot supply a lossless 8-kHz sample history.
  `ImageStreamIO_readBufferAt` provides a pointer, not a retained-frame lease;
  semaphore posts are notifications, not a queue of immutable payloads.
- **Depth one is intrinsically lossy for an asynchronous consumer.** If two
  acquisitions complete before the consumer copies the first, the overwritten
  pair is gone. A second consumer buffer, faster polling or counting semaphore
  receipts cannot recover it. Do not describe depth-one operation as lossless.
- Recommend a producer temporal ring of 256 frames (32 ms retention at 8 kHz),
  configured by the operator through existing `framegrabber.circBuffLength` at
  startup. This changes storage depth, not model evolution/synchronization.
  Per-plane `cntarray`, `atimearray`, `writetimearray` must exist for depth >1.
  The reader drains retained consecutive frames in order, verifies counters and
  publication state around a bounded local copy, counts overwritten history and
  resets/rewarms after discontinuities. Retention is finite and is not an
  unconditional delivery guarantee. A true every-frame guarantee needs a
  producer-supported acknowledged queue or equivalent transport; that is outside
  this phase and must not be bolted onto the unchanged simulator silently.
- Use one application-owned input/estimator worker, reusing MagAOXApp threadStart,
  native existing-only attachment and semaphore claiming. Do not derive from the
  latest-only monitor or change that shared helper. This supersedes the earlier
  two-worker/SPSC proposal: there is no actuator publisher requiring isolation
  now, and a single owner can drain, reset, predict, and check stale input on
  bounded waits. No second payload queue or borrowed raw pointer is needed.
- Worker performs only bounded snapshot/copy, timestamp translation, numerical
  ingestion, and periodic prediction. Attachment/reconnection is a cold path;
  backing-file identity probes and INDI/reporting stay in supervision. No stream
  payload writes, creates, destroys, or output attachment. Reject DM-prefixed
  input names before opening them.
- Source acquisition stamps must be valid CLOCK_REALTIME values as published by
  frameGrabber/accelSim. Map them using a sampled realtime/monotonic clock anchor,
  check anchor uncertainty/clock steps and timestamp ordering, and pass actual
  monotonic receipt time to the unchanged estimator. Old retained records that
  exceed numerical freshness are counted/discarded rather than assigned fake
  arrival times. Recovered recent frames are ingested once each. Prediction uses
  actual now plus the requested horizon, counting measurement age once.
- Reset/reconfigure and input restarts automatically clear/re-warm observation;
  there is no enable, gain, or actuator state in this app. Keep lifetime loss,
  rejected-frame, restart, reset and timing diagnostics even when a fit resets.
  Use versioned coherent parameter requests applied by the worker; no INDI
  callbacks mutate its live estimator. Publish snapshots at 5 Hz by default,
  with prediction initially 2 kHz and source ingestion at every available frame.

Reference: [upstream ImageStreamIO API](https://milk-org.github.io/milk-package/ImageStreamIO_8h.html).
Its pointer/semaphore helpers do not establish a deployment-specific queue or
memory-consistency guarantee. The deployed producer/library contract and native
Linux load tests remain qualification inputs; local framework build limitations
will be reported explicitly. No agent runs commands on instrument hosts.

### Implemented architecture and boundaries

Work started from branch `parkertjohnson1/accelSim`, HEAD `4e58786c`, with a clean
working tree. The standalone numerical estimator and original numerical/replay
test files remain byte-for-byte unchanged. No accelSim source, disturbance,
calibration, RNG, publication, pacing or synchronization code changed. No agent
opened an instrument stream or executed instrument/INDI commands, and no commit
or push was made.

| File | Implemented responsibility |
| --- | --- |
| `apps/accelFFCtrl/accelFFCtrl.hpp` | Normal `MagAOXApp<true>` lifecycle, `threadStart`, single input/prediction owner, nonblocking control/snapshot handoff, low-rate supervision/INDI. |
| `apps/accelFFCtrl/accelFFCtrl.cpp` | Main entrypoint only. |
| `apps/accelFFCtrl/accelFFCtrlInput.hpp` | Fixed parameter vocabulary/validation, raw owned frame records, bounded retained-ring drain, epoch translation, counter/generation/freshness faults, observation reset/recovery, throughput and timing diagnostics. No native I/O or simulator truth. |
| `apps/accelFFCtrl/accelFFCtrlStream.hpp` | Existing-only native ADC mapping, type/geometry checks, reader semaphore, stable copy adapter, bounded monotonic waits and close. No payload publication/create/destroy API. |
| `apps/accelFFCtrl/Makefile`, repository `Makefile` | Standard `magAOXApp.mk`, IEEE arithmetic, RTC/all-buildable registration. No service/autostart or existing runtime configuration change. |
| `apps/accelFFCtrl/config/example.conf` | Non-actuating startup example, nominal 8-kHz input, 2-kHz queries, 5-Hz INDI, one-ms example future horizon. |
| `apps/accelFFCtrl/tests/accelFFCtrlInput_test.cpp` | Deterministic retained transport/clock/control/recovery/loss-rate coverage without ImageStreamIO or app dependencies. |
| `apps/accelFFCtrl/tests/accelFFCtrlStream_test.cpp` | Opt-in real ImageStreamIO reader tests, synthetic raw streams only in a verified private namespace; never an instrument/DM image. |
| `apps/accelFFCtrl/tests/accelFFCtrl_test.cpp` | Offline full-framework config parsing, INDI handoff/reset, stale telemetry and backing-file identity supervision tests; no worker startup or stream access. |
| `apps/accelFFCtrl/tests/accelFFCtrlInput_benchmark.cpp` | Allocation-guarded bounded ingestion/prediction benchmark with synthetic transport and clocks. Existing numerical benchmark is unchanged. |
| `apps/accelFFCtrl/tests/Makefile`, `tests/Makefile.one`, `tests/tests.list`, `tests/groups.dox` | Standalone/input/native build targets, dependency registration and application-unit-test grouping. |
| `apps/accelFFCtrl/README.md` | Architecture, buffer limitation, units/clocks, properties/status codes, build/test commands, operator-only source transfer/deployment/INDI runbook. |

Numerical work remains single-owner. Callback requests are complete validated
parameter candidates plus revision numbers. An owner `try_lock` applies the
latest revision, resets the fit and acknowledges it through a value snapshot;
callbacks never mutate the live estimator. Supervision may block on its own
locks/INDI, while the worker only tries the handoff mutexes. Snapshot reporting is
5 Hz by default; 2-kHz prediction does not decimate ingestion. No high-rate
prediction stream was added. A future DDSPC owner can reuse the native/owned ADC
records and time/identity validation without a dependency on the simulator or a
DM sink; no speculative predictor abstraction was added to the per-sample path.

Input attaches at the newest completed record and then walks retained history
in order. A stable local batch uses actual post-copy receipt time for all
age checks, with each frame's own acquisition stamp. Initial history before
attachment is outside the observed counter interval. Batch capacity is allocated
once (default 256); no borrowed shared-memory pointer is queued. Changed/torn
copies leave the cursor unchanged. Exact overwritten gaps plus rejected frames
increment lifetime dropped counts. Invalid frames are a separate subset. Old
retained samples exceeding the default 2-ms age limit are discarded explicitly;
32-ms retention is not permission to treat old samples as fresh.

The source must supply CPU UINT16 `2 x 1 [x D]`, X/Y 12-bit codes and valid
host CLOCK_REALTIME acquisition/write stamps. The deployed hardware producer's
clock, ring arrays and frameGrabber-like publication order remain commissioning
assumptions. A post-copy realtime/monotonic bracket establishes a fixed integer
clock anchor. Excessive uncertainty/offset step inhibits and reanchors; bad,
future, stale or nonmonotonic sample times reset/re-warm. This does not establish
remote-host clock synchronization. Prediction uses `now + horizon`, and the
unchanged numerical core advances from the latest acquisition stamp, counting
age once. The forward bound still includes that age.

The worker waits for source notifications or its next prediction deadline.
glibc >=2.30 uses `sem_clockwait(CLOCK_MONOTONIC)`; fallback is nonblocking
`sem_trywait` with bounded relative sleeps, avoiding a realtime timeout that can
be prolonged by clock steps. Notification draining is bounded and never used as
a sample count. Attachment/reconnection is cold work, retried at one-second
intervals. Supervision probes source inode/device at telemetry cadence; only
the owner closes/reopens mappings. Counter rollback/replacement attachment
increments generation and clears the fit. Existing producers must safely
unlink/recreate streams; truncating an attached mapping is not supported.

Two implementation review corrections are recorded explicitly:

- Acquisition timestamp rate and accepted throughput are distinct diagnostics.
  `acquisition_fps` smooths valid consecutive timestamp intervals; `input_fps`
  counts accepted samples over 250-ms monotonic windows, including gaps. Fit
  resets do not erase throughput accounting. A deterministic depth-one test
  deliberately loses alternate samples and reports approximately 4 kHz accepted
  throughput from an 8-kHz producer, plus exact drop counts.
- Generic `pcf::IndiElement::get<double>()` does not reliably report parse
  failures. Scalar controls instead parse the entire raw text with conversion
  status, reject nonfinite/trailing/malformed values, and validate the complete
  candidate before acceptance. Startup scalar config uses the same parser.
  This prevents a malformed ADC zero from silently becoming a legal zero.

There is no actuation state, enable switch, correction gain, DM name/config,
DM mapping extraction or output publication. Input names beginning `dm` are
rejected before any native filename lookup/open. The app's purpose is continuous
observation, with cleared invalid predictions and automatic re-warmup. This
phase does not relax or implement any future actuator enable/ownership/zeroing
requirements.

### Phase-two validation and operational qualification

The standalone command `make -C apps/accelFFCtrl/tests test benchmark` passed
17 cases / **3,350,757 assertions**, preserving the original estimator/replay
expectations. Seven new input cases exercise raw decoding/batched timestamps,
one-frame loss versus retained recovery, finite/bounded overrun, busy/torn copies,
counter rollback/generation, invalid/future/backwards/stale times, clock bracket
bounds/steps, explicit reset/config changes, malformed controls and effective
throughput under loss. The original four Python PSD validation tests also passed.

The final combined AddressSanitizer/UndefinedBehaviorSanitizer executable passed
**51 cases / 3,587,355 assertions** with halt-on-error enabled, including these
17 cases and all 34 existing accelSim model/disturbance/DM numerical
cases (236,598 assertions). It was built with `-DACCELSIM_MODEL_ONLY`, so DM
regressions use local mock sinks and no native stream tests are compiled or
executed. No sanitizer findings occurred. The binary used C++20 and
`-O1 -g -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer`, the original
three accelSim numerical test sources, the model/replay/input accelFFCtrl test
sources, and the existing `tests/testMain.cpp`.

On the local Apple Silicon workstation, the final allocation-guarded input
benchmark measured mean **122.089 ns/sample**, p50 42 ns, p99 417 ns, maximum
49.5 us across 240,000 synthetic samples, with **zero C++ new calls**, Eigen's
allocation guard active, all records accepted and none dropped. Synthetic
transport/clock callbacks exclude native mapping, actual clock syscalls,
semaphore wake latency and scheduling. The unchanged numerical benchmark at
8-kHz input / 2-kHz queries measured 138.644 ns/source sample including its
measurement clocks; consume/predict p99 were 83/417 ns. These are workstation
cost observations, not hard realtime bounds. At 8 kHz, the source period is
125 us; the default query period is 500 us. Linux load testing must qualify
delivery, acquisition age, latency distribution and CPU scheduling separately.

Full `make -C apps/accelFFCtrl` and the full-framework app test build stop in
existing local framework dependencies before compiling the application:
`timegm` is unavailable under the configured macOS feature flags and installed
mxlib lacks `mx/math/floatUtils.hpp`. Upstream ImageStreamIO 2.00.00 headers
downloaded to `/tmp` pass native reader/test compilation, but its library cannot
build on this Mac (`strerror_r` GNU/POSIX mismatch, unavailable `sem_timedwait`
and process-shared semaphore support). Consequently native tests cannot link/run
here. A limited GCC 15 syntax check of the new app and app tests against the real
MagAOXApp/INDI/native headers passed using temporary local declarations for
missing platform/mxlib APIs and excluding unused framework device helpers. No
compatibility shims or dependency edits were added to the repository. This is
not a full linked build, app test pass or native runtime qualification.

The README gives exact commands for Linux standalone/native/full-app tests,
source-archive transfer without commit/push, human-operated exao2 build/install,
observation-only config, optional isolated INDI server on an unused port, and
10-minute load/reset/restart observations. It reuses existing producer ring/rate
configuration only with owner coordination, prohibits duplicate `acc000`
producers, and never changes an existing DM injection setting. No startup
management or production INDI configuration was modified. Operators must match
producer ADC/frequency/lever/clock values and independently qualify raw-stream
semantics; default 10-m geometry remains an assumption. Native/full-framework
Linux tests and actual 8-kHz delivery/timing are outstanding deployment gates.
INDI at 5 Hz is unsuitable for a 28-Hz residual PSD or correction-phase score;
future high-rate recording/correction acceptance remains separately scoped.

Earlier detuning, weak-signal quantization, neighboring-frequency rejection and
stochastic replay limitations remain in force. No new suppression claim,
physical calibration result, DDSPC implementation or DM correction is implied.

## Phase 3A: Linux build readiness and native validation review

### Scope, source identity and evidence boundary

Started from clean `parkertjohnson1/accelSim` at `fb71d5cf` (committed Phase 2).
Read AGENTS.md, the full application/helpers/tests, README and this engineering
record. Reviewed frameGrabber/shmimMonitor, MagAOXApp lifecycle/thread/INDI helpers,
the shared make rules, and the hardware mcp3208Ctrl acquisition timestamps.
This phase authorizes isolated build/correctness fixes and private validation;
it does not authorize actuation, simulator changes, instrument commands,
application restarts or configuration changes on exao2. No merge/commit/push.

The workstation is Darwin arm64, with no local Linux VM/container or installed
ImageStreamIO package. It has mxlib 0.2.0 and Eigen 5.0.1. The previously
downloaded upstream ImageStreamIO 2.00.00 sources/headers in `/tmp` are reference
material, **not evidence of the version installed on exao2**. This checkout
does not vendor/pin those headers; `setup/README.md` points to a separate setup
repository. Installed header/library identity and ABI remain pending the
human-operated Linux commands now provided in the README. Do not substitute an
upstream download for that evidence.

### Findings and isolated corrections

| Finding | Resolution / evidence |
| --- | --- |
| App/thread/native test linkage relied on indirect dependency flags. `sem_clockwait` is in pthread on older supported glibc releases. | App explicitly links `-lImageStreamIO` and compiles/links with `-pthread`; native/full app test builds also specify pthread/ImageStreamIO dependencies. `_GNU_SOURCE` already exposes the glibc 2.30+ API. Existing monotonic fallback remains unchanged. |
| Bounded ingestion compared a new counter only with the consumed cursor. After observing C=16 but consuming through 12, rollback to 15 went undetected and could mix an old batch tail into a new source generation. | Track the last stable snapshot counter separately; compare against the larger of it and the cursor, preserving the guard for a first copy that races before any stable snapshot exists. Rollback resets/rebinds at the newest publication. Count known unconsumed old-generation records as dropped on rollback/disconnection and clear the reported per-generation counter. Regression coverage includes both the bounded tail and an initial torn copy followed by restart. |
| Temporal array pointer presence alone was insufficient to identify a retained temporal cube. | Native header checks `(imagetype & 0xF000F) == (CIRCULAR_BUFFER | ZAXIS_TEMPORAL)` for depth >1, matching the upstream array-allocation convention. 2-D/depth-one global metadata remains supported. Frame copies guard missing temporal arrays. |
| A mapped read handle was implicitly copyable, and close cleared a reader slot even if its PID claim had changed. | Delete copy construction/assignment, release only this PID's slot, clear the local native descriptor after close. Native tests check ownership cleanup; no stream payload writes were added. |
| A write timestamp later than actual post-copy receipt was accepted. | Reject future write stamps as invalid time, in addition to invalid/future acquisition stamps and write-before-acquisition. Tests cover it. This also exposes mixed realtime/TAI metadata rather than silently accepting a nonsensical producer delay. |
| Native tests covered only five sequential publications and one reattachment. | Extend private native coverage to depth 256, repeated wraps, both raw channels, 2-D input, temporal flags, finite overrun, stale retained samples, reader claim cleanup and real same-name recreation. Add an explicit Linux-only paced producer test. |

Changed files are limited to `apps/accelFFCtrl/Makefile`,
`accelFFCtrlInput.hpp`, `accelFFCtrlStream.hpp`, `README.md`,
`tests/Makefile`, `tests/accelFFCtrlInput_test.cpp`,
`tests/accelFFCtrlStream_test.cpp`, repository `tests/Makefile.one`, and this plan.
`accelFFCtrl.hpp`/main, numerical model and original analytic/replay tests are
unchanged. All accelSim implementation/model/DM/mapping/stream headers are
unchanged. No shared framework source was modified.

### API, publication and timing review

The native calls match the repository's use of ImageStreamIO:
`ImageStreamIO_filename`, `openIm`, `getsemwaitindex`, `semflush`, `closeIm`;
CPU location -1, `_DATATYPE_UINT16`, axes/size, `write`, `cnt0`, `cnt1`, and
`cntarray`/`atimearray`/`writetimearray`. The reference native sizing function
allocates temporal arrays only for the checked circular/temporal type. The
consumer's four-byte pair offset equals two UINT16 codes per plane. A snapshot
checks counters/geometry/write state before and after copying owned records;
no queued pointer or semaphore count is treated as immutable sample history.

`frameGrabber::fgThreadExec` creates a `2 x 1 x D` circular temporal image,
copies data under `md.write`, sets realtime write/acquisition stamps, updates
`cnt1` then increments `cnt0`, copies corresponding per-plane metadata, clears
write and posts semaphores. Counter-relative indexing therefore handles wrap
without assuming plane index equals `cnt0 % D`. Depth one is still intrinsically
lossy; depth 256 gives finite 32-ms retention at 8 kHz, while the unchanged 2-ms
freshness policy rejects older retained records. Restart loss accounting covers
the known stable snapshot tail; unobserved records after that snapshot cannot
be counted when counters reset. A restart event remains evidence of that gap.

ImageStreamIO's header may select CLOCK_TAI for `CLOCK_ISIO`. That is not the
controller's time contract: frameGrabber explicitly stamps CLOCK_REALTIME and
the controller maps those stamps to local monotonic time. Creation time and
generic native update defaults are not acquisition timestamps. The hardware
synchronized path in this checkout sets `m_currImageTimestamp` before reading
channels; the timer path does not assign it. Actual producer/mode/timestamp
validity must be confirmed later; no hardware driver or receipt-time fallback
was added. The private fixtures deliberately publish the frameGrabber realtime
contract.

Prediction has no acquisition-age/processing-delay double counting:
`target = query_now + configured_horizon`, then the numerical core advances by
`target - newest_acquisition`. The query clock is read after input processing,
so completed readout/copy/ingestion age is already included. Producer write
delay is diagnostic only; neither it nor service time is added to the horizon.
Remaining solve time, a future publisher/actuator delay and hold behavior are
not measured actuation latency in this non-actuating phase. Five-Hz INDI
snapshots are diagnostic records with target timestamps, not commands to apply
at INDI receipt. Numerical equations/defaults remain unchanged.

INDI APIs/macros and callback signatures match MagAOXApp. Properties are defined
before framework INDI startup. Callbacks validate/coalesce settings under a
control mutex; the worker uses `try_lock`, owns the estimator/mapping, and
publishes value snapshots. Stop/run atomics and joinable checks cover early
startup failure, shutdown and destruction. The shared `threadStart` still uses
its existing plain `pid_t` handoff and checks `pthread_setschedparam` failure
with `rv < 0`, although that API returns positive error numbers. These are
inherited framework limitations, not fixed by an app-specific replacement in
this phase. Its success log is not evidence of actual RT policy/priority;
worker scheduling and a separate shared-helper correction remain follow-ups.
The private paced test does not exercise framework startup or RT scheduling.

### Tests, instructions and remaining qualification

- Local optimized standalone estimator/replay/input tests: **18 cases,
  3,350,774 assertions passed**. The original numerical expectations are intact.
- Local AddressSanitizer/UndefinedBehaviorSanitizer standalone tests: the same
  **18 cases / 3,350,774 assertions passed**, with halt-on-error enabled and no reported findings.
- Allocation-guarded input benchmark: 240,000 accepted samples, zero dropped,
  zero C++ allocations and no Eigen allocation; observed mean 134.582 ns,
  p99 500 ns, maximum 30.209 us. This excludes native I/O/syscalls/scheduling and
  is not a Linux deadline claim. The existing numerical benchmark also passed.
- A real full app make was attempted and remains blocked in local framework
  dependencies (`timegm`, missing installed mxlib `mx/math/floatUtils.hpp`),
  before the app compiles. No Linux linked build or native execution occurred.
- Native test source passes local syntax checks against reference headers.
  Its Linux-only test body was additionally type-checked in a temporary copy
  with a POSIX `clock_nanosleep` declaration; no fake implementation was linked
  or run. App/app-test GCC 15 syntax checks against real base/INDI/native headers
  passed using the previous temporary missing-platform/mxlib declarations and
  omitting unused aggregate device helpers. These limited checks are not Linux
  compilation or installed ABI validation. No shims were added to the repo.
- clang-format and whitespace checks pass; the README's operator shell block
  passes `bash -n`. Preservation hashes verify the original numerical model,
  original tests, simulator headers and plan prefix above Agent Findings.

`native-test` now contains four deterministic native cases: original
retained/type validation, 12,000-sample two-channel replay in four-sample batches,
and 2-D input/recreation. The replay crosses many 256-plane wraps and validates
the 50-us measurement age plus one-ms future horizon (1.05-ms total). It then
withholds consumption for 300 publications, verifying exactly 44 overwritten
and 240 stale-retained drops, with 16 recent samples accepted.
The native tests print the library-written metadata version and header sizes.

`native-rate-test` is a separate opt-in Linux case. It forks a private producer
for 40,000 additional samples over about five seconds at 125-us absolute
deadlines, samples real acquisition/write/receipt clocks, uses a 256-plane ring
and 2-kHz prediction, accounts for all 40,001 records including the initial bind,
and requires readiness to have been observed. It reports rates, losses, invalids,
races and resets. A pass with loss validates accounting, not lossless throughput.
The producer is reaped on normal exit or assertion unwinding; no existing app
or instrument image is used. Both native targets create a fresh verified private
namespace before starting the test process, and cleanup failure is no longer
silently ignored. These native cases are **not yet executed**.

README now distinguishes the current Phase 3A build/private tests from the
historical later install/launch/INDI/producer-reconfiguration runbook. It gives
exact human-operated commands for compiler/glibc/pkg-config identities, app
and app-test builds, standalone tests/benchmarks, both native targets, header
traces and `ldd` paths, with logs in a unique `/tmp` report directory. It points
out that standalone tests do not inherit `local/common.mk`, that installed
ImageStreamIO headers/libraries must agree across app and native builds, and
that a missing `.pc` file requires actual installed paths rather than a guessed
replacement library. No install, observer launch, INDI command, instrument
configuration change or app restart belongs to those Phase 3A commands.

Request the first failing compiler/linker command and full error context, test
totals, native ABI line, `Native 8k/2k` summary, and selected header/PCH/library
paths from the operator. Until those results arrive, Linux app linkage, installed
ABI/metadata compatibility, native restart/wrap correctness and paced delivery
remain unverified. Live `acc000` timing under RTC load, actual clock/calibration,
timer-mode timestamp validity and worker RT policy remain later qualifications.
No DM actuation, simulator disturbance change, merge, commit or push was made.

## GCC 14 / C++20 Eigen LDLT compatibility correction

### Diagnosis and selected workaround

The operator reported a GCC 14 exao2 build failure instantiating
`Eigen::LDLT<Eigen::Matrix3d>` in `harmonicEstimator::predict`, with a rejected
comparison in the installed `LDLT.h` near line 372. Work began from clean branch
`parkertjohnson1/accelSim`, HEAD `a5cae31f`. The exact installed Eigen version/full
diagnostic was requested but was not available during this local investigation.

[Eigen issue 2012](https://gitlab.com/libeigen/eigen/-/issues/2012) describes the
matching C++20 failure for Eigen 3.3.7/3.3.8, already observed with GCC 10.2.
C++20 rewritten equality candidates interact with Eigen's older scalar/array
overloads; the selected reversed comparison returns an expression instead of
the required bool. This is not a data-dependent failure of the harmonic fit or
a reason to lower the language standard for the repository.

Downloaded the unmodified Eigen 3.3.7 release into `/tmp` and reproduced the
failure locally using GCC 15, `-std=c++20 -fno-fast-math`, without permissive
compiler flags. The original model fails at `LDLT.h:373` and `:381`, naming
non-bool rewritten equality candidates. An LLT/solve/rcond probe compiles and
runs with those same headers/compiler. The local system Eigen remains 5.0.1;
no installed header or separate worktree was modified.

The targeted source fix replaces the one fixed-size LDLT factorization with
`Eigen::LLT<Eigen::Matrix3d>`. The accepted Gram matrix was already required to
be strictly positive definite: LDLT success, positive sign, positive pivots and
sufficient reciprocal conditioning were prerequisites for valid output.
[Eigen LLT](https://libeigen.gitlab.io/eigen/docs-3.3/classEigen_1_1LLT.html) supplies
that positive-definite Cholesky solve without instantiating the broken LDLT
comparison path. It solves the same two-right-hand-side `G2*C=Q2` system; it
does not change window statistics, raw decoding, units, oscillator phase,
warmup, timestamps, horizon, coordinate transform or fault/cleared-output policy.

LLT failure is checked **before** calling `rcond()`, because LLT's condition
estimate requires a successful factorization. `info()!=Success` inhibits a
non-positive-definite/rank-deficient fit. The existing finite/rcond threshold
`1e-8`, coefficient finite checks and residual threshold `1e-10` are retained.
No regularization, inverse, retuned tolerance or extra per-sample work is added.
Roundoff-level coefficient changes are expected; this is mathematical/numerical
equivalence within the tested tolerances, not a bitwise-identity guarantee.

Changed files: `apps/accelFFCtrl/accelFFCtrlModel.hpp`,
`apps/accelFFCtrl/tests/accelFFCtrlModel_test.cpp`, `apps/accelFFCtrl/README.md`,
and this plan. No make/compiler configuration changes were needed. Earlier
phase preservation statements describe their historical work; the user's new
request explicitly authorizes this local solver/test compatibility correction.
Simulator, DM, stream/INDI/application code and separate DDSPC worktree remain
unchanged. All text above Agent Findings and Plan remains unchanged.

### Regression, parity and allocation results

Extended the independent weighted-batch QR oracle to test both 0.02-s and 0.1-s
memories with uniform/irregular acquisitions. Existing amplitudes, phases,
offsets and prediction accuracy tolerances are unchanged. Extended the numeric
fault regression to cover rank-deficient and very short windows, verifying
cleared numerical-fault output without querying a failed LLT condition estimate.

- Unmodified Eigen 3.3.7 + GCC 15 / C++20 optimized standalone suite:
  **18 cases / 3,359,793 assertions passed**. Allocation-guarded numerical and
  input benchmarks also passed with zero C++/Eigen allocations.
- System Eigen 5.0.1 + Apple Clang / C++20 optimized standalone suite:
  **18 cases / 3,359,793 assertions passed**, with the same allocation results.
- Current-Eigen AddressSanitizer/UndefinedBehaviorSanitizer combined executable:
  **52 cases / 3,596,391 assertions passed**, halt-on-error enabled. This includes
  all 34 existing accelSim model/disturbance/DM numerical cases compiled with
  `ACCELSIM_MODEL_ONLY` (mock sinks; no native/physical stream access).
- Old-Eigen AddressSanitizer/UndefinedBehaviorSanitizer standalone executable:
  **18 cases / 3,359,793 assertions passed**, halt-on-error enabled. The local
  GCC installation lacks libasan, so this sanitizer run used Apple Clang. Old
  Eigen also refers to removed `std::result_of`; this Mac/libc++-only test run
  enabled `_LIBCPP_ENABLE_CXX20_REMOVED_TYPE_TRAITS` in the command's CXXFLAGS.
  That flag is not in the repository or the GNU/Linux rebuild instructions.
  The normal GCC 15/C++20 old-Eigen proof requires no such compatibility flag.
- Four existing Python PSD validation tests passed. No sanitizer findings.

A separate temporary comparison loads the committed LDLT model under a renamed
namespace beside the new LLT model, using current Eigen where both compile.
It feeds identical analytic/noisy quantized ADC records to both over 72
combinations: f=28/28.5/29 Hz, T=0.02/0.1/0.3/0.8 s, uniform/irregular intervals,
and seeds 42/123/701. Across **514,614 ready prediction comparisons**, there
were zero consume-status or estimator-state mismatches. Maximum absolute
differences were acceleration/amplitude `5.42e-20 m/s^2`, wrapped phase
`8.88e-16 rad`, pointing `2.07e-25 rad`, offset `1.82e-20 m/s^2`, and rcond
`3.33e-16`. This comparison is an offline investigation artifact, not a
second production solver or a simulator input path. It does not establish
bitwise equivalence or behavior exactly on a floating-point threshold boundary.

### Operator rebuild and remaining limit

README now gives exact human-run commands using the installed Eigen and the
same GCC 14 compiler as the failed build, forcing recompilation with
`-W accelFFCtrlModel.hpp`, then fresh-directory numerical tests, benchmarks and
ASan/UBSan tests. No install, service restart, observer launch, INDI command,
instrument configuration change or system Eigen edit is part of those commands.

A local full application make was attempted and remains blocked in pre-existing
macOS framework dependencies (`timegm`, missing installed mxlib
`mx/math/floatUtils.hpp`) before the app compiles. GCC 14/Linux application
rebuild confirmation therefore remains with the operator; report the exact
Eigen/compiler versions and full first diagnostic if another failure occurs.
No instrument-facing commands, merge, commit or push were performed.

## Offline end-to-end prediction validation (2026-11-05)

### Scope and reproducible experiment

Added `apps/accelFFCtrl/tests/accelFFCtrlValidation_test.cpp` (opt-in hidden
export case), `validate_predictions.py`, `test_validate_predictions.py`, and
standalone Makefile targets `validation` / `validation-test`. README documents
commands and results. Existing replay/model/input/native tests are retained;
they already cover analytic amplitudes/phases, calibration, noise, detuning,
timestamp faults, ring ordering, rollback/restart, stale input and configuration.
No production header, simulator step/RNG/synchronization, configuration, DM
publication, reset rule or readiness gate changed. No instrument was accessed.

The experiment generates each source recording once with the unchanged model:
8 kHz; simulation seed 42; mode seed 123; lever 10 m; sensor noise PSD `1e-8`;
ADC zero 2048, volts/count `0.0083799`, sensitivity `1.02`. It compares one
28-Hz stochastic mode (`accelRms=.002`, `r=.99999`, unit coupling) and the full
214-mode population. Both use f0=28 Hz, T=0.1 s, rotation=0, configured horizon
1 ms. Additional lossless replays sweep horizons 0/0.5/1/2/5/10/20 ms. Source
records include 162 future samples; controller replay does not advance the model
or RNG. No simulator truth enters the production ingestion or numerical API.

The full-population 28-Hz truth is extracted read-only from that model's actual
mode histories after each step. A separate single-mode realization with the
same seed is **not** the same component: changing the population changes RNG
consumption. Clean full acceleration excludes sensor noise; full physical
pointing excludes WFS coupling. Truth is used only by the offline scorer.

Production `inputProcessor::drain/predict` consumes a coherent synthetic
depth-one source: acquisition at `base+n*125000` ns, completed publication
25 us later, normal worker drain 50 us later. Predictions occur every 500 us,
125 us after an acquisition grid boundary, before that boundary's publication.
Thus the newest available sample is 125 us old. Queries use their actual receipt
clock; realtime and monotonic clocks have an exact fixed synthetic offset.
Blackouts suppress the worker, never producer steps/publications. Missed nominal
query deadlines count as unavailable; a late coalesced query is recorded
separately and does not add a scheduled slot to the denominator. A final drain
accounts for the known tail, including its overrun event in the query-only case.

Scoring uses the common 4–60 s interval (112,000 lossless 2-kHz predictions per
axis). RMSE is `sqrt(mean((prediction-truth)^2))`; NRMSE divides by
`sqrt(mean(truth^2))`, not peak or standard deviation. Bias is mean(pred-truth);
correlation is Pearson's coefficient. Invalid predictions are masked, not scored
as zero. Availability includes all scheduled slots, including blocked queries.
Recovery statistics censor episodes interrupted by another loss.

### Timing, units and amplitude conventions

`a=(code-2048)*0.0083799/1.02` m/s². One count is
`0.008215588235294117 m/s²`; maximum rounding error is half a count. Across both
recordings/axes, measured quantization RMS is 0.002370–0.002372 m/s²; no ADC rail
clipping occurs. Sensor-noise RMS is 0.006317–0.006330 m/s², consistent with
`sqrt(PSD*8000/2)=0.00632456`. Noise supplies dither for the sub-count target;
the existing undithered weak-tone test correctly shows it can disappear.

`theta=-a_hat/((2*pi*28)^2*10)` radians, then rotation (zero here).
The negative sign is the harmonic acceleration/pointing relation, not a
correction sign. Convert to mas with `180*3600*1000/pi`; do not apply gravity,
reflection or DM factors. The simulator target obeys this identity to
`3.31e-24 rad`; the exporter checks the same relation for predictions. Full
pointing uses each mode's own frequency; applying the 28-Hz inverse to the
entire acceleration population would be incorrect.

Target time is `query_now+1 ms`, and phase advance is `target-latest_acquisition`.
Here this is **1.125 ms = nine samples**. The scorer requires an exact integer
truth-grid timestamp and rejects any fractional/off-grid/out-of-range target.
Unit tests distinguish the correct index 9 from the incorrect index 8. There is
no extra processing/window latency in the target; measurement age is included
once. Publication delay is exactly 25 us in this model. Existing replay tests
use the explicit target API (sample+1 ms with a 100-us query age); that separate
experiment is valid and was not silently reinterpreted as query+1 ms.

`prediction.m_amplitude` is peak sensor acceleration; `m_phase` is the cosine
phase at the newest acquisition. Export checks reconstruct the target value
from those fields and `omega*(target-sample)`. A pure sinusoid has peak/RMS
ratio sqrt(2). A stochastic AR mode has a changing envelope/phase: finite-record
coherent projection, Welch transfer and instantaneous envelope are distinct.

### Sixty-second accuracy results

Errors below reference the actual **28-Hz component**, including for the full
population. Acceleration and pointing NRMSE/correlation coincide because that
component has the same scalar harmonic inverse as the estimator.

| Population / axis | Acceleration RMSE [m/s²] | Pointing RMSE [mas] | NRMSE | Correlation | Acceleration bias [m/s²] | Pointing bias [mas] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single / X | 0.000309011 | 0.205932 | 15.6267% | 0.987821 | +1.406e-7 | -9.368e-5 |
| Single / Y | 0.000320195 | 0.213385 | 17.7829% | 0.984220 | -1.034e-6 | +6.891e-4 |
| Full / X | 0.000930590 | 0.620165 | 43.1750% | 0.917982 | -1.929e-6 | +0.001286 |
| Full / Y | 0.000965499 | 0.643430 | 49.1985% | 0.892906 | +8.490e-7 | -0.000566 |

Two explicitly labeled 28-Hz amplitude/phase statistics are exported:

| Population / axis | Coherent peak-amplitude error | Coherent phase error | Welch gain error at 28 Hz | Welch phase at 28 Hz |
| --- | ---: | ---: | ---: | ---: |
| Single / X | +0.176% | -0.348° | -0.247% | -0.640° |
| Single / Y | -0.401% | +0.178° | -0.654% | -0.311° |
| Full / X | +3.041% | -2.915° | +1.010% | -1.691° |
| Full / Y | +0.503% | +6.120° | -1.221% | -0.941° |

Coherent values come from independent `[1,cos(omega*t_target),sin(omega*t_target)]`
least squares across the full scoring interval. Positive phase means prediction
leads truth. The full-Y coherent truth amplitude is only 0.000505 m/s² despite
0.001962 m/s² RMS: envelope cancellation makes its finite-record phase more
sensitive. Welch uses 8-s Hann segments, 50% overlap, constant detrending and
the exact 28-Hz bin (0.125-Hz resolution); transfer is `S_truth,pred/S_truth,truth`.
The 27–30-Hz component residual reductions are single X/Y 19.15/17.76 dB and
full X/Y 10.04/9.35 dB. This does not establish a universal 10-dB correction gate.

Comparing with **all modes** instead gives full pointing RMSE 1.9125/1.5503 mas,
NRMSE 83.71%/81.43%, correlation 0.5608/0.5996. Full acceleration RMSE is
0.039385/0.039146 m/s², NRMSE 99.90%/99.94%: the narrowband predictor is not a
broadband acceleration estimator. These statistics distinguish residual other
modes from errors of the intended component. Other modes' stochastic tails
near 28 Hz also contaminate the observed band and explain some additional
full-population component error; not all nearby power belongs to the 28-Hz mode.

| Horizon beyond query [ms] | Single NRMSE X/Y | Full NRMSE X/Y |
| ---: | ---: | ---: |
| 0 | 15.58% / 17.73% | 43.16% / 49.17% |
| 1 | 15.63% / 17.78% | 43.18% / 49.20% |
| 5 | 15.77% / 17.95% | 43.09% / 49.11% |
| 10 | 15.96% / 18.15% | 42.99% / 48.96% |
| 20 | 16.53% / 18.78% | 43.44% / 49.47% |

The actual sample-to-target horizon adds 0.125 ms to each row. Small
non-monotonic full-population changes reflect interference/envelope evolution;
they do not justify retuning latency to minimize this finite record's error.
Unknown future stochastic innovations cannot be predicted exactly.

### Depth-one overruns and recovery

The following counts apply to **each** population: 480,000 produced records,
120,000 nominal prediction slots. Resets include the initial attachment.
Blackouts start at each period boundary after t=0 and exclude the t=60 boundary.

| Worker condition | Overrun events / rate [1/s] | Dropped | Resets | Availability, all 60 s | Availability, 4–60 s | Recovery after detection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ideal 8-kHz ingestion | 0 / 0 | 0 | 1 | 99.4992% | 100% | Initial ready at 300.625 ms |
| 125-us block every 1 s | 0 / 0 | 0 | 1 | 99.4992% | 100% | No overrun; predictions bit-identical to ideal |
| 250-us block every 5 s (one loss) | 11 / 0.1833 | 11 | 12 | 93.9900% | 94.0973% | 300.375 ms |
| 250-us block every 1 s (one loss) | 59 / 0.9833 | 59 | 60 | 69.9500% | 69.9500% | 300.375 ms |
| 250-us block every 100 ms | 599 / 9.9833 | 599 | 600 | 0% | 0% | All episodes interrupted before warmup |
| 2-ms block every 5 s | 11 / 0.1833 | 165 | 12 | 93.9625% | 94.0679% | 300.125 ms |
| Incorrect query-only 2-kHz ingestion | 120,000 / 2000 | 359,999 | 120,001 | 0% | 0% | Never ready |

The real app ingests on source wakes independently of the query cadence; the
last row is a negative control, not the deployed architecture. Fifteen frames
are overwritten per 2-ms block. No malformed/stale records or source restarts
are injected in this study; those are covered by existing input/native cases.
All accounting checks `processed+dropped=published`. Recovery differences of
fractions of a millisecond reflect query/sample phases and coverage rounding.
Initial warmup uses `ceil(max(3/f0,3*T)*1e9)`; floating-point rounding plus the
2-kHz query grid makes first readiness 300.625 ms in this schedule.

During the first 0.5 s after recovery from a single loss every 5 s, component
NRMSE X/Y is single 14.88%/19.45% and full 40.29%/50.44%; with losses every
second it is single 15.39%/18.18% and full 42.43%/48.87%. Availability and
accuracy are reported separately; scoring only valid predictions must not hide
the 30%-per-second availability penalty of the strict reset policy.

### Warmup investigation and recommendation only

Added an independent batch-QR oracle for the exact two-stage window. For a
uniform interval, weight of lag j is `b²*(j+1)*(1-b)^j`, with
`b=-expm1(-2/(8000*T))`. For variable b, sample j at endpoint k has weight
`b_j*product(lambda_[j+1:k])*sum(b_[j:k])`. A new unit test compares batch QR
to independently accumulated recursive moments, with and without a missing
acquisition and with an unrelated 120-Hz component. It does not call or alter
the production estimator's reset/readiness logic.

Cold-start probes use the raw samples following the overwritten sample at each
of 59 one-second epochs; each age below is one prediction per epoch, not a
continuous availability result. Targets remain newest acquisition+1.125 ms.

| Fresh history [ms] | Gram reciprocal condition | Single NRMSE X/Y | Full NRMSE X/Y |
| ---: | ---: | ---: | ---: |
| 5 | 0.000116 | 1411% / 1921% | 29008% / 39870% |
| 20 | 0.02848 | 49.86% / 55.28% | 586.02% / 543.85% |
| 50 | 0.33182 | 45.54% / 30.25% | 114.43% / 111.62% |
| 100 | 0.41280 | 17.31% / 21.19% | 51.04% / 66.08% |
| 150 | 0.48768 | 13.95% / 20.37% | 44.60% / 73.97% |
| 300 | 0.48407 | 21.22% / 19.92% | 41.58% / 39.23% |

All these probes are SPD with reciprocal condition above the production 1e-8
threshold; maximum normal-equation relative residual is below 3e-13. Thus
**300 ms is not a hard floating-point stability requirement**. Short windows
nevertheless strongly admit sensor noise, ADC noise and other modes. At
100 ms the cascaded window has accumulated only about 59.4% of its steady-state
mass; at 300 ms about 98.3%. Conditioning alone does not qualify accuracy or
envelope stability, and these sparse, correlated probes do not define a safe
universal shortened gate. Do not reduce cold-start warmup based on this study.

A separate hypothetical candidate retains one second of mature history across
exactly one omitted acquisition, uses the true 250-us acquisition interval
at the next sample, and compares batch fits with an uninterrupted oracle. It
probes 0/1/10/50/100/300 ms after receipt of the post-gap sample. Gram reciprocal
condition remains >=0.48368; normal residuals remain below 4e-15. Immediately
after the gap, maximum acceleration difference from the uninterrupted fit is
1.00e-6/5.86e-7 m/s² (single X/Y) and 1.47e-6/1.77e-6 (full X/Y). Across all
probe ages the maximum changes are 5.46e-5 (single) and 6.88e-5 (full) m/s².
This favors investigating preservation of an already-qualified window for a
strictly bounded isolated gap, rather than forcing another cold start. It is
**not** production support or general proof of safe gap handling. Future work
must qualify burst/repeated loss, arbitrary gap phases/lengths, noise/detuning,
clock/identity faults, stale data, ADC faults and supported memory/frequency
settings; counter ordering/loss telemetry and invalidation for real integrity
faults must remain explicit. No missing sample may be fabricated. Current
production reset/warmup behavior remains unchanged.

### Commands, verification and limits

Run entirely on a workstation from the repository root, using an existing Python
environment with NumPy/SciPy/pandas/Matplotlib (set `accff_python` accordingly):

```bash
accff_python=python3
make -C apps/accelFFCtrl/tests PYTHON="$accff_python" validation \
  BUILD_DIR=/tmp/accelFFCtrl-validation-build \
  VALIDATION_DIR=/tmp/accelFFCtrl-validation-60s VALIDATION_SECONDS=60
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-validation-build test benchmark
"$accff_python" -m unittest discover -s apps/accelSim/tests -p test_validate_psd.py -v
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-validation-asan test \
  CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  ACCELFFCTRL_VALIDATION_DIR=/tmp/accelFFCtrl-validation-asan-smoke \
  ACCELFFCTRL_VALIDATION_SECONDS=6 \
  /tmp/accelFFCtrl-validation-asan/accelFFCtrl_test '[accelFFCtrl-validation-export]'
```

For the combined hardware-free simulator/controller sanitizer run:

```bash
c++ -std=c++20 -O1 -g -fno-fast-math -Wall -Wextra -pthread \
  -fsanitize=address,undefined -fno-omit-frame-pointer -DACCELSIM_MODEL_ONLY \
  $(pkg-config --cflags eigen3) \
  apps/accelSim/tests/accelSim_test.cpp \
  apps/accelSim/tests/accelSimDisturbance_test.cpp \
  apps/accelSim/tests/accelSimDM_test.cpp \
  apps/accelFFCtrl/tests/accelFFCtrlModel_test.cpp \
  apps/accelFFCtrl/tests/accelFFCtrlReplay_test.cpp \
  apps/accelFFCtrl/tests/accelFFCtrlInput_test.cpp \
  apps/accelFFCtrl/tests/accelFFCtrlValidation_test.cpp tests/testMain.cpp \
  -o /tmp/accelFFCtrl-validation-combined-asan
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  /tmp/accelFFCtrl-validation-combined-asan
```

Local environment: Apple Clang 21, libc++ 210106, Eigen 5.0.1; NumPy 2.3.5,
SciPy 1.16.3, pandas 2.3.3. Actual mode specs, compiler/STL identity and Python
versions are exported; C++ random distributions need not produce identical
realizations across standard libraries. Seeds alone do not establish identity
with a running instrument trajectory. Numerical results above belong to this
exported realization and supplied parameters, not measured exao2 performance.

- Optimized existing controller suite: 18 cases / 3,359,793 assertions passed;
  the same suite passed under ASan/UBSan.
- New 60-s exporter: 1 case / 29,464,003 assertions passed, including exact
  target/phase/inverse checks and analytical loss/reset accounting for each
  schedule. Four new Python tests and four existing PSD tests passed.
- Combined simulator/controller ASan/UBSan: 52 cases / 3,596,391 assertions
  passed. The six-second exporter under the same sanitizers passed 1 case /
  2,872,567 assertions. No sanitizer findings; halt-on-error enabled. Simulator
  DM numerical cases use mock sinks only and access no physical/native stream.
- Both allocation-guarded benchmarks passed with zero C++/Eigen allocations.
  Synthetic input runtime averaged 114 ns/source opportunity, p99 417 ns;
  these workstation timings exclude native mapping, semaphore/clock syscalls,
  filesystem, INDI and scheduling. They are not real-time latency guarantees.
- Existing unrelated-tone tests (f0=28.5 Hz) give acceleration RMS gains <=0.454
  across 11–3500 Hz tested, and <=0.01696 for tested frequencies >=54 Hz.
  White-noise output/input RMS ratio is 0.03586. These check rejection without
  claiming exact isolation of other modes' tails at the target frequency.
- Raw binaries and reports are under `/tmp/accelFFCtrl-validation-60s` locally;
  no large generated artifacts are added to the repository. Trace CSVs thin to
  500 Hz for viewing; PSDs use the complete uniform 2-kHz lossless output.
  Gapped predictions are never concatenated into a fake uniform PSD. Full-mode
  truth sampled at 2 kHz can alias broadband AR tails; the PSD is explicitly the
  output-cadence comparison, not a reconstructed 8-kHz sensor spectrum.

No new production correctness bug was found. Strict single-frame-loss reset is
the principal availability limitation. The new analysis initially mislabeled
ns-to-ms values as microseconds during its smoke review; it was corrected before
the reported study and now has an explicit 125-us/1125-us regression assertion.
Timing/ordering in the synthetic source is coherent by construction. This
workstation cannot execute native Linux ImageStreamIO/app tests; semaphore
ordering/races, real 8-kHz worker scheduling, independent clocks, physical
calibration and simulator scheduling deviations are not qualified by this
offline study. Existing native restart/ordering cases are retained unchanged.

The numerical path supports a longer **observation-only** run, conditional on
recording actual dropped/reset/ready/age diagnostics and interpreting unavailable
intervals explicitly. Five-Hz INDI snapshots cannot validate high-rate phase/PSD;
a later human-collected timestamped raw/prediction recording would be needed
for that comparison. Recommend retaining current behavior, reviewing full-mode
interference, and qualifying the isolated-gap candidate offline before any
recovery-policy change. This is not authorization or evidence for DM correction.
No deployment, native stream access, configuration change, commit or push was
performed. Text above Agent Findings and Plan remains unchanged.

## Offline bounded-gap investigation and 8/4-kHz comparison (2026-11-05)

### Authorization and evidence boundary

This phase permits investigation and an isolated offline prototype only.
Production `accelFFCtrlInput.hpp`, `accelFFCtrlModel.hpp`, the application/native
reader, simulator, configurations and all DM paths are unchanged. Depth remains
one. Earlier retention recommendations are historical, not an instruction to
change this phase's producer storage. No instrument host/INDI was accessed and
no deployment, commit, push or merge was performed. Production adoption requires
separate user approval after reviewing these results.

The user supplied one live observation: at 8 kHz, with accelSim on CPU30 and
accelFFCtrl on CPU29, 2,241 drops and 2,202 resets occurred over 602 s. This gives
3.7226 drops/s and 3.6578 resets/s. The drop count is 0.04653% of **nominal**
8-kHz opportunities; actual published-counter totals were not supplied. Mean
time per reset is 0.27339 s, shorter than the 0.3-s warmup, but the distribution
and overlap of reset intervals are unknown. Actual readiness cannot be inferred
by summing warmups. Counts alone do not prove that each reset was a single-frame
overrun or identify the operating-system cause. The separate live 4-kHz
experiment is running; no result was supplied or assumed.

### Exact production loss/reset trace

`apps/accelFFCtrl/accelFFCtrlInput.hpp` owns lifetime counters and the fit:

| Path / API | `dropped` effect | Reset effect |
| --- | --- | --- |
| `drain`: first retained counter exceeds cursor+1 | Add the missing publications | `invalidate(overrun)` once per observed gap, even one lost frame |
| `drain`: malformed/absent timestamp, write-before-acquisition, write-after-receipt, mapped future time, non-monotonic accepted time | Add one; increment `invalid` | Clear immediately with `invalidTime` |
| `drain`: `harmonicEstimator::consume` rejects ADC >4095, stale input, time/interval/sequence or numerical failure | Add one; increment `invalid` | Clear immediately; reason is stale or numerical |
| `attached` / `disconnected`: known stable snapshot has an unconsumed tail | Add seenCounter-cursor | Attach always resets; disconnect resets only if connected |
| `drain`: counter rollback below max(cursor,seenCounter) | Count known abandoned tail through `attached` | Increment generation/restarts and reset |
| `drain`: incompatible geometry/type/location/temporal metadata | No direct addition | Reset on entry into incompatible state |
| `drain`: invalid clock bracket, uncertainty, offset step or rollback | No direct addition; cursor is not advanced | Reset on entry into clockFault, discard anchor; later overwrites can add drops |
| `predict`: acquisition older than maxAge | No direct addition | Inhibit/latch stale immediately; reset is counted when subsequent ingestion clears the fault |
| `predict`: failed/ill-conditioned numerical fit | No direct addition | Inhibit/latch numerical fault; subsequent ingestion clears it |
| Accepted configuration revision or operator reset | No addition by itself | Reset even with no lost sample |
| Application mapping replacement/wait failure/shutdown | Known abandoned tail only | `disconnected`; successful reopen also calls `attached` |

All counted resets pass through `inputProcessor::invalidate`. A multi-frame
overrun need not produce one reset per missing frame; invalid records or other
causes can produce resets without counter-gap loss. A stale query itself does
not increment the reset counter. Repeated incompatible/clock-fault polls are
guarded against resetting on every identical fault poll. Initial attachment
starts at the latest publication and does not count unknown pre-attachment
history. Producer-side skipped nominal ticks are also different from lost
publications: `accelSim::acquireAndCheckValid` calls `model::advance(skipped)`,
but frameGrabber increments `cnt0` only for actual publications.

Busy writes increment `busy`; inconsistent counter/metadata copies increment
`races`; no new counter increments `duplicates`. These paths do not immediately
increment drops/resets. They defer reading, so a later coherent snapshot can
reveal an overrun. Long reader delays with a continuing producer usually yield
a fresh latest sample plus a counter gap; silence can instead yield staleness.
The strict reader resets before consuming a post-gap record. Consequently it
forgets previous acquisition-order history at that boundary; a retaining policy
must explicitly check the previous timestamp across the gap as this prototype does.

`accelFFCtrlStream.hpp::frame` copies the two UINT16 codes and global counter /
acquisition / write stamps at depth one. `header` brackets native write state,
geometry and counters around the copy. `wait` uses monotonic semaphore deadlines
on modern glibc and bounds additional semaphore draining at 256 posts; the
fallback uses trywait and requested sleeps up to 20 us. Posts are notifications,
not retained payloads. At depth one, two completed publications before copying
mean overwritten history regardless of semaphore count. `maxBatch` only caps
how many retained planes can be copied; `count=min(counter-first+1,maxBatch)`
is at most one here. Increasing it cannot recover overwritten frames or make
the reader wait to form a batch.

`accelFFCtrl.hpp::worker` drains on source wakes and prediction deadlines,
queries at 2 kHz, and coalesces missed prediction deadlines. It does not ingest
only at 2 kHz. Supervision handles filesystem identity and 5-Hz INDI outside
the worker; control/snapshot locks use try-lock in the worker. Affinity to
different CPUs does not guarantee sub-125-us wake/copy latency. Scheduler,
interrupt, copy-race, semaphore-backlog and producer timing effects are possible
contributors, not confirmed causes from the supplied aggregate counts. To
distinguish them later, use a human-collected offline trace of physical counters,
acquisition/write/receipt times and existing invalid/last_event/busy/races/age
diagnostics, together with simulator skipped/late/discontinuity diagnostics.
Five-Hz INDI alone cannot reconstruct per-frame scheduling.

### Offline prototype and numerical admission policy

Added test-only `tests/accelFFCtrlGap_test.hpp` and `accelFFCtrlGap_test.cpp`.
The application does not include the helper and has no runtime switch for it.
The helper's first `inputProcessor` is the unchanged production validator.
Every owned source pair and clock is validated there. A private query retrieves
the exact mapped acquisition timestamp even during the validator's warmup.
Only then does a separate unchanged `harmonicEstimator` consume original ADC
codes and that timestamp. Its consecutive sequence is an **accepted-record
ordinal**, while physical publication counters/generations and all loss counts
remain separately audited. No sample or timestamp is filled in or fabricated.

The production validator still resets internally on every gap; its reset counts
are not presented as experimental fit resets. The second numerical owner may
retain mature history only when all these conditions hold:

- Same source generation, strictly increasing physical counter and acquisition
  time; the validator accepted the current record and its only reset cause is
  one overrun (not an ADC, timestamp, clock or source fault).
- Kernel was already ready. The original `max(3/f,3*T)` cold-start coverage,
  LLT/condition/residual checks, maxInterval, maxAge and target bounds are unchanged.
- `gap1`: at most one missing frame. `gap7`: an exploratory allowance of seven,
  still bounded by acquisition elapsed time <=1 ms. Therefore gap7 can admit
  at most seven missing samples at 8 kHz, three at 4 kHz.
- For physical counter increment d, `abs(dt-d*nominalPeriod) <= 0.05*nominalPeriod`.
  This is a deliberately strict **offline** cadence/identity consistency guard.
- At least T=0.1 s since the previous retained gap. A dense second gap resets;
  repeated gaps during cold start continue to restart the original warmup.

Any other validator reset, rejected record, stale/numerical kernel state or
cadence mismatch clears the experimental fit/output. Fault tests check
inhibition immediately after ingestion, before another query. The next accepted
fresh record starts cold. Production data/clock checks are reused, not weakened.

Actual elapsed acquisition time drives the original cascades:
`beta=1-exp(-2*dt/T)`, then the original Gram/data recurrences. Carrier phase
advances by `omega*dt`; prediction advances from the newest acquisition to the
actual target. Renumbering accepted records is not a fixed-rate time surrogate.
This is essential: a missing sample changes elapsed time even though the next
accepted ordinal is consecutive. No simulator state/model/RNG enters this path.

This two-owner validation scaffold is intentionally an offline experiment, not
the recommended live implementation. A future approved production change should
integrate explicit bounded-gap admission with the existing owner/validator and
retain physical counters in public diagnostics, avoiding a second fit/query on
every input record.

### Study construction and units

Extended the existing deterministic exporter, preserving the original validation
case and all assertions. Recording/truth export and scheduler replay are shared.
There are 132 replays: two rates, two populations, three policies, 11 schedules.
Each 60-s source recording is generated **once**, with 162 future samples. At
8/4 kHz there are 480,000/240,000 publications; all runs have 120,000 nominal
2-kHz query slots. ADC/physical settings remain seed42/modeSeed123, noise PSD
1e-8, zero2048, volts/count0.0083799, sensitivity1.02, lever10m, f0=28Hz,
T=0.1s, zero rotation, query+1ms prediction. Full truth's actual 28-Hz mode is
extracted from that same recording; full pointing is a separate reference.

Mode specs including **per-sample** pole radii are identical at both rates.
Halving the rate doubles the physical pole-decay time and changes the stochastic
trajectory/realized signal RMS even with the same seed. Sensor-noise RMS is
sqrt(PSD*fs/2), so it changes from 0.0063246 to 0.0044721 m/s²; quantization
step remains 0.008215588 m/s²/count. Thus this is the literal same-parameter
comparison requested, not a controlled comparison of identical continuous
disturbance histories. Holding physical damping fixed would require a separate
pole-radius transformation, which is not applied here.

Publication and normal receipt delays remain 25/50 us. Queries use actual now,
with normal latest-sample ages 125/250 us. Effective horizons are exactly
1.125/1.250 ms (nine/five truth samples). Reader-only delays can increase sample
age up to the existing 2-ms freshness boundary; every ready target still equals
query+1ms. Integer grid checks prohibit silently flooring targets. Acceleration
and pointing remain related by the same negative harmonic inverse; peak/RMS and
coherent-phase conventions remain as in the preceding section.

`validate_gaps.py` reuses the existing scorer. It verifies identical accepted
physical sample indices **and receipt times** between policies, and bit-identical
lossless predictions. Accuracy uses nominal slots after 4 s, reports each
policy's own valid mask **and** a common production-valid mask, and compares
each result with lossless output at the same timestamps. Invalid slots are not
zeros. Coherent amplitude/phase fits use actual target times even across holes;
gapped samples are never concatenated into a uniform PSD. Availability counts
every nominal slot, including blocked worker deadlines. A late coalesced query
can mark recovery but does not add another nominal deadline.

### Accuracy and phase results

The following errors reference the actual 28-Hz component. First table is the
lossless baseline; all three policies give bit-identical predictions there.

| Rate / population | Acceleration RMSE X/Y [m/s²] | Pointing RMSE X/Y [mas] | NRMSE X/Y | Correlation X/Y |
| --- | ---: | ---: | ---: | ---: |
| 8 kHz / single | 0.0003090 / 0.0003202 | 0.20593 / 0.21338 | 15.627% / 17.783% | 0.98782 / 0.98422 |
| 8 kHz / full | 0.0009306 / 0.0009655 | 0.62017 / 0.64343 | 43.175% / 49.199% | 0.91798 / 0.89291 |
| 4 kHz / single | 0.0002931 / 0.0002951 | 0.19532 / 0.19667 | 14.271% / 16.160% | 0.98982 / 0.98702 |
| 4 kHz / full | 0.0006663 / 0.0007542 | 0.44406 / 0.50261 | 47.040% / 57.661% | 0.89920 / 0.86680 |

With one physical frame lost every second, `gap1` gives:

| Rate / population | Pointing RMSE X/Y [mas] | NRMSE X/Y | Coherent target peak-amplitude error X/Y | Coherent target phase error X/Y |
| --- | ---: | ---: | ---: | ---: |
| 8 kHz / single | 0.20602 / 0.21334 | 15.632% / 17.779% | +0.178% / -0.387% | -0.343° / +0.189° |
| 8 kHz / full | 0.62002 / 0.64363 | 43.159% / 49.212% | +3.037% / +0.516% | -2.913° / +6.133° |
| 4 kHz / single | 0.19525 / 0.19678 | 14.264% / 16.168% | -0.002% / -0.571% | -1.221° / -0.298° |
| 4 kHz / full | 0.44337 / 0.50293 | 46.961% / 57.694% | +2.835% / -5.673% | -3.418° / +16.336° |

These absolute coherent phase errors include the original stochastic envelope,
interference and finite-record cancellation; they are not gap-induced biases.
Against lossless output **on the same valid timestamps**, additional coherent
gain/phase changes from one loss/s are:

| Rate / population | Gain change X/Y | Phase change X/Y |
| --- | ---: | ---: |
| 8 kHz / single | +0.0011% / +0.0157% | +0.0039° / +0.0121° |
| 8 kHz / full | -0.0042% / +0.0497% | +0.0017° / +0.0366° |
| 4 kHz / single | +0.0151% / -0.0392% | -0.0098° / +0.0334° |
| 4 kHz / full | +0.0003% / -0.0735% | -0.0639° / +0.0544° |

On the common production-ready mask, full-population NRMSE is production
43.244%/48.820% versus gap1 43.204%/48.765% at 8 kHz, and production
46.835%/57.320% versus gap1 46.783%/57.237% at 4 kHz. Thus the availability
gain is not obtained by scoring unavailable production intervals as zeros.
CSV includes signed bias, correlation and delta-from-ideal RMSE for every case.

For the 4-Hz-loss schedule after maturity, gap1 full-component NRMSE remains
43.126%/49.220% at 8 kHz and 46.919%/57.812% at 4 kHz. Production has no ready
slots after the 4-s settling cut, so its paired accuracy is explicitly undefined.
For exploratory gap7 with a 1-ms blackout each second, full NRMSE is
43.346%/49.288% at 8 kHz and 46.970%/58.374% at 4 kHz. These limited results
do not justify deploying the larger-burst allowance.

### Scheduling, drop accounting and recovery

Results below apply equally to single/full populations. All availability values
include the initial cold start over the full 60 s; reset counts include initial
attachment. Policy B is gap1, C is exploratory gap7. Periodic blackouts begin
after one period, except the 4-Hz schedule begins at t=1 s so it tests loss after
maturity. That is a simulation assumption, not the observed live loss pattern.

| Condition | Lost frames 8/4 kHz | Production resets 8/4 kHz | Gap1 resets 8/4 kHz | Production availability 8/4 kHz | Gap1 availability 8/4 kHz | Gap7 availability 8/4 kHz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ideal | 0 / 0 | 1 / 1 | 1 / 1 | 99.499% / 99.499% | 99.499% / 99.499% | 99.499% / 99.499% |
| Block one acquisition period every second | 0 / 0 | 1 / 1 | 1 / 1 | 99.499% / 99.499% | 99.499% / 99.499% | 99.499% / 99.499% |
| Block 250 us every second | 59 / 0 | 60 / 1 | 1 / 1 | 69.950% / 99.499% | 99.450% / 99.499% | 99.450% / 99.499% |
| Block 500 us every second | 177 / 59 | 60 / 60 | 60 / 1 | 69.950% / 69.950% | 69.950% / 99.450% | 99.450% / 99.450% |
| Block 1 ms every second | 413 / 177 | 60 / 60 | 60 / 60 | 69.901% / 69.901% | 69.901% / 69.901% | 99.401% / 99.401% |
| One frame lost every second (2-period blackout) | 59 / 59 | 60 / 60 | 1 / 1 | 69.950% / 69.950% | 99.450% / 99.450% | 99.450% / 99.450% |
| One frame lost at 4 Hz, starting t=1 s | 236 / 236 | 237 / 237 | 1 / 1 | 1.166% / 1.166% | 99.303% / 99.303% | 99.303% / 99.303% |
| Three one-frame losses spaced 1 ms, every 5 s | 33 / 33 | 34 / 34 | 23 / 23 | 93.953% / 93.953% | 93.963% / 93.963% | 93.963% / 93.963% |
| Block 2 ms every 5 s | 165 / 77 | 12 / 12 | 12 / 12 | 93.963% / 93.963% | 93.963% / 93.963% | 93.963% / 93.963% |
| Delay reader 3 ms every 5 s; keep queries running | 253 / 121 | 12 / 12 | 12 / 12 | 93.981% / 93.981% | 93.981% / 93.981% | 93.981% / 93.981% |
| One-frame losses every 100 ms from startup | 599 / 599 | 600 / 600 | 600 / 600 | 0% / 0% | 0% / 0% | 0% / 0% |

Divide dropped counts by 60 for rates (e.g. one-loss/s schedule 0.9833 drops/s,
4-Hz schedule 3.9333 drops/s over the complete recording). The prototype never
hides physical loss: all policies have identical source-loss counts and accepted
record histories. Every replay asserts processed+dropped=published, zero invalid
records for these valid-data scheduling studies, and the analytically expected
loss count. Raw frame losses are unchanged by preserving a numerical fit.

Single-loss production recovery is 300.375 ms at 8 kHz and 300.250 ms at 4 kHz
from overrun detection to first ready query. Gap1 is ready at the first post-gap
query (0-ms measured recovery from receipt), while any blocked deadlines remain
unavailable. At one loss/s, post-settling availability is 69.95% production
versus 99.95% gap1; at 4-Hz loss it is 0% versus 99.8%. Dense clusters admit
their first isolated gap but reset on the second and third; the final recovery
still costs about 300 ms. Interrupted warmups are censored. The CSV's mean
recovery across both retained (0 ms) and reset events must not be mistaken for
a shorter cold-start gate.

The reader-only delay deliberately calls `predict` during input silence. All
three policies produce 22 stale queries per recording, all inhibited, zero
fault-state queries for otherwise valid records, and no ready sample older than
2 ms. Maximum ready effective horizon becomes 2.750 ms at 8 kHz / 3.000 ms at
4 kHz because sample age is included. Worker-wide blocked deadlines instead
record no produced prediction; they are unavailable, not fresh zero estimates.

### Coefficient uncertainty, residuals and safety tests

`validate_gaps.py::fit_quality` reuses the independent batch-QR oracle and
computes an investigation-only white-noise covariance proxy:
`Cov(C)=sigma² G^-1 (sum(w_i² u_i u_i^T)) G^-1`, where
`sigma²=PSD*fs/2+ADCstep²/12`. It reports coefficient-derived peak-amplitude
and phase standard errors plus weighted input-fit residual RMS. Colored mode
contamination and stochastic model error are excluded from this covariance;
it is not a calibrated confidence bound or a new admission threshold.

Probes use at most 12 actually retained events per schedule, at receipt and
50 ms later, with one second of accepted history. Probes following another
reset are excluded. Against the same uninterrupted history, gap1 increases the
amplitude standard-error proxy by at most 0.266% at 8 kHz and 0.526% at 4 kHz;
weighted residual ratios across populations lie in 0.9990–1.0024 at 8 kHz and
0.9963–1.0076 at 4 kHz. Gram reciprocal conditions stay >=0.4842 in these batch
probes and normal-equation relative residuals below 3.1e-15. Exploratory gap7
can increase the amplitude-error proxy about 7.24% (8 kHz) / 3.13% (4 kHz).
This further favors restricting an initial production proposal to single frames.

Phase uncertainty can be very large when the estimated harmonic amplitude
approaches a null (up to 328° in a full-population proxy probe). This is an
existing signal-identifiability limitation; a finite well-conditioned fit alone
does not certify useful physical phase or correction. Retention does not solve
it. The linearized phase standard error is itself unreliable near zero amplitude
and must not be interpreted as a calibrated confidence interval. The small
**incremental** gap phase changes above do not imply uniformly
small absolute phase error in the noisy full population.

New ordinary tests cover one/two/three/seven/eight consecutive missing frames
at both rates and both gap caps, actual target-time/phase reconstruction,
unmodified cold start, dense repeated gaps, ADC >4095, NaN configuration,
absent/future/backwards timestamps, bad write ordering, counter/time mismatch,
backwards time across a gap, counter rollback, explicit producer generation
restart, wrong datatype, realtime clock step, excessive bracket uncertainty,
stale receipt, input silence and wrong nominal-rate assumptions. Every injected
integrity fault inhibits and clears numerical output; no faulted prediction
remains ready. Wrong cadence
assumptions cannot accumulate a ready fit in this deliberately strict prototype.
The production estimator itself correctly uses timestamps after its initial
weight; a wrong nominal rate is not inherently a fixed-rate phase error there.

The prototype cadence check is intentionally conservative. Actual accelSim
timestamps are sampled from realtime at acquisition, and synchronization/scheduler
behavior can make valid intervals irregular or skip nominal numerical ticks
without counter gaps. The fixed 5%-of-period guard may therefore cause additional
live resets. It must be qualified against an offline recording of real timestamp
jitter before production adoption, not silently relaxed. Supported memory/frequency
ranges, long-duration/different-seed behavior, burst distributions and remote
clock/calibration assumptions also need further qualification.

### Reproduction, regressions and recommendation

All commands are workstation-only and create artifacts under `/tmp`. Select an
existing Python environment with NumPy/SciPy/pandas/Matplotlib; local runs used
Apple Clang21/libc++210106/Eigen5.0.1 and Python package versions exported into
`report/analysis_versions.json`. Actual mode specs/compiler/STL metadata accompany
the raw binaries; standard-library RNG distribution differences preclude assuming
bit-identical trajectories from seeds alone on exao2.

```bash
accff_python=python3
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-gap-build \
  PYTHON="$accff_python" VALIDATION_DIR=/tmp/accelFFCtrl-gap-60s \
  VALIDATION_SECONDS=60 gap-validation
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-gap-build test benchmark
"$accff_python" -m unittest discover -s apps/accelSim/tests -p test_validate_psd.py -v
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-gap-final-asan test \
  CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  ACCELFFCTRL_VALIDATION_DIR=/tmp/accelFFCtrl-gap-asan-smoke \
  ACCELFFCTRL_VALIDATION_SECONDS=6 \
  /tmp/accelFFCtrl-gap-final-asan/accelFFCtrl_test '[accelFFCtrl-gap-export]'
MPLCONFIGDIR=/tmp/accelFFCtrl-gap-60s/.matplotlib \
  "$accff_python" apps/accelFFCtrl/tests/validate_gaps.py /tmp/accelFFCtrl-gap-60s
```

The preceding section's combined simulator/controller sanitizer command also
applies, adding `apps/accelFFCtrl/tests/accelFFCtrlGap_test.cpp` to its sources.
No full app/INDI/native stream launch is part of these commands. Local macOS
cannot qualify native Linux scheduling or ImageStreamIO concurrency; existing
native tests remain unchanged. Original assertions were preserved.

- Ordinary controller regression: 21 cases / 3,403,380 assertions passed,
  including the original 18 cases and three new gap-policy cases; ASan/UBSan
  passed the same suite. Both allocation-guarded production benchmarks passed.
- Combined hardware-free simulator/controller ASan/UBSan: 55 cases /
  3,639,978 assertions passed, including simulator DM numerical cases with mock
  sinks only. No sanitizer findings; halt-on-error enabled.
- Full 60-s gap exporter: 1 case / 148,400,155 assertions passed. Dedicated
  ASan/UBSan six-second gap export: 1 case / 14,528,351 assertions passed.
- Seven Python scoring/oracle tests plus four existing simulator PSD tests passed.
- Original six-second exporter: 1 case / 2,872,567 assertions passed, with every
  binary truth/prediction record bit-identical to the prior-phase export after
  extracting the shared helper. The original replay/model/input assertions remain.
- Generated CSVs/plots/binaries are local `/tmp/accelFFCtrl-gap-60s` artifacts,
  not repository additions. `accuracy.csv` distinguishes own/common valid masks
  and incremental phase/gain; `availability.csv` includes rates, resets, retained
  events, stale/fault/blocked counts and recovery; `fit_quality.csv` gives the
  independent uncertainty/residual probes. Plots show before/after pointing,
  residuals, readiness during isolated/clustered loss and per-rate availability.

Recommendation: approve a **separate, reviewed observation-only production
implementation limited initially to one missing frame**, preserving physical
loss telemetry, mature-history admission, real elapsed time, freshness, full
cold-start warmup and all non-gap fault resets. The offline results justify
that next implementation; they do not justify deployment, larger gap caps,
automatic cold-start shortening or DM correction. First qualify cadence/jitter
admission using a human-provided timestamped recording and review the pending
live 4-kHz experiment once its results are supplied. Keeping the nominal rate
at 4 kHz improves modeled scheduling headroom but is not proven to improve
full-population normalized prediction accuracy or the unknown live loss pattern.
Production remains unchanged pending that explicit approval. Text above Agent
Findings and Plan remains unchanged.

Files changed in this phase: `apps/accelFFCtrl/tests/Makefile`,
`accelFFCtrlValidation_test.cpp`, `validate_predictions.py`, new
`accelFFCtrlGap_test.hpp`, `accelFFCtrlGap_test.cpp`, `validate_gaps.py`,
`test_validate_gaps.py` (all under that tests directory),
`apps/accelFFCtrl/README.md`, and this plan. No other application is modified.
The future DDSPC interface must still see physical counters, actual timestamps
and gaps. Accepted ordinals are private to this harmonic experiment; they must
not silently turn lossy measurements into a purported uniform DDSPC history.

## Optional production gap retention and timestamp qualification (2026-11-06)

### Authorization and supplied live evidence

The user now authorizes a production implementation behind a default-disabled
configuration option, plus offline qualification. This supersedes the preceding
offline-only implementation boundary for that narrowly defined scope. No
instrument access, deployment, production configuration edit, DM actuation,
commit, push or merge is authorized or performed.

The user reports the completed 4-kHz observation: 602 s, 2,407,458 processed
frames, zero drops, zero additional resets, depth one (about 3,999.1 accepted
frames/s using the rounded duration). The earlier 8-kHz observation had 2,241
drops / 2,202 resets in 602 s. These are user-supplied aggregate observations,
not agent measurements. The 4-kHz result demonstrates good observed transport
under those conditions; it does not measure prediction error, acquisition
jitter, or gap-guard acceptance, since there were no gaps to admit.

### Timestamp investigation and qualification boundary

Reviewed the current plan, numerical/replay/input/native tests, both offline
scorers, the prototype, application/configuration tests and allocation benchmarks.
Searched the repository's accelFFCtrl/agent data paths (including ignored local
CSV/binary/log candidates) and local accelFFCtrl validation artifact directories.
Available per-sample recordings there are the previously generated synthetic
datasets. No representative live timestamp recording was found or supplied.
An asynchronous request for an existing local recording path was made; no live
qualification is claimed without such data. Aggregate counts cannot replace it.

The prototype checked cadence on every accepted interval against nominal rate:
`abs(dt-d*P) <= .05*P`, where P=1e9/sampleRate and d is the counter increment.
This can reject valid ordinary irregular samples. The numerical estimator already
uses actual acquisition times, so a nominal-rate mismatch alone is not a
fixed-rate phase error. Applying this extra gate to every sample is unnecessary
for source/data validation and unsuitable for a production default.

There is a second concrete source of legitimate deviations:
`accelSim.hpp` defaults `m_maxSlew=.05`, and
`accelSimModel.hpp::phaseClock::consume` budgets up to `slew*m_period` of phase
correction per following interval, limited by headroom. Across a single missing
sample's two intervals, phase slew can change dt by 0.1P, exceeding the 0.05P
gap guard. Actual acquisition timestamps also include scheduling lateness;
producer skipped ticks do not necessarily appear as a publication-counter gap.
Neither simulator timing nor synchronization was modified.

Production therefore uses the strict cadence gate **only for the decision to
retain history across a counter gap**. Ordinary valid samples keep existing
timestamp-aware behavior. Failure of the cadence gate resets history and
consumes the valid fresh record from cold; it does not label that frame invalid
or discard it. This is an intentional correction to the prototype design, not
a relaxation of the original ADC/clock/stream/timestamp/freshness checks.

Live qualification requires an offline human-provided trace containing, for each
accepted stable-copy record: exact physical counter, source generation and
mapping identity, two UINT16 ADC codes, acquisition/write CLOCK_REALTIME integer
nanoseconds, actual receipt CLOCK_MONOTONIC integer nanoseconds, and the clock
anchor pairs plus bracket uncertainty used for translation. Include nominal
acquisition rate, estimator/ADC controls, producer skipped/late/discontinuity
and phase-slew settings/diagnostics. Cover steady operation, active slewing,
losses/clusters, restarts and load at both rates. Do not substitute floating-point
epoch seconds or 5-Hz INDI snapshots. Characterize recording overhead; creating
an instrument-side recorder is outside this phase and was not done.

Retain rejected stable-copy candidates with validation/reset reasons as well.
For exact live-prediction replay, also record each query's actual now, target,
sample time, readiness/state, physical source identity and predicted values.
Without query/reset records, accepted timestamp pairs qualify interval/guard
distributions but cannot reconstruct every live fit reset or actual target.

### Production implementation

`accelFFCtrlInput.hpp` adds a fourth constructor argument
`allowSingleFrameGap=false`. The app loads startup `input.allowSingleFrameGap`,
accepting exactly `true/false/1/0`; malformed text fails configuration through
the unit-tested `singleFrameGapSetting` parser. Missing keys preserve false.
No shipped configuration file is edited. No INDI write switch is added.

An optional gap is admitted only after a stable owned copy and validated clock,
with all of the following:

- Exactly one missing publication at depth one; the physical counter advances
  by two from the last accepted frame in the current generation.
- The estimator and latest query were ready. Cold-start coverage remains
  `ceil(max(3/f0,3*T)*1e9)`; there is no shortened startup/recovery window.
- Both previous and new acquisitions are fresh at actual receipt. Source
  acquisition/write order, monotonic translation, ADC codes, maxAge/maxInterval,
  clock mapping/bracket, and snapshot validation remain mandatory.
- `0 < dt <= min(1 ms,maxInterval)` and `abs(dt-2P) <= .05P`: ±6.25 us around
  250 us at 8 kHz, ±12.5 us around 500 us at 4 kHz.
- At least one memory time (0.1 s in these studies) since a previous admitted gap.
  A dense second gap or a multi-frame loss forces a full reset.

The same existing `harmonicEstimator` consumes original ADC codes and mapped
acquisition timestamps exactly once. In optional mode its sequence is a private
accepted-record ordinal, because the existing kernel intentionally faults on
nonconsecutive source counters. Physical ordering/gaps are validated in the
reader before that call, and physical counters are restored in public prediction
identities. Generation changes and all other integrity faults still reset.
No new numerical solver, second estimator, fabricated sample, or simulator/RNG
input is introduced. The numerical model header remains unchanged.

An admission is counted only after successful consume. The cached pre-gap
prediction is immediately cleared without resetting the fit; a subsequent
query must solve and pass conditioning/freshness/horizon checks before becoming
ready. This also prevents retaining stale state when no query ran during a
reader delay. Phase/window updates use the actual elapsed acquisition interval,
and target remains actual query-now plus horizon, with measurement age once.

Missing publications are always added to `dropped`. The enabled-only acquisition
cadence diagnostic divides a retained two-period interval by its physical
counter increment; accepted throughput still includes loss. Disabled processing
preserves the original validation/reset order, physical kernel sequences and
numerical outputs. Configuration/reset/reattach clear admission history while
lifetime counters persist.

At ordinary telemetry cadence, the app adds read-only `gap_enabled`,
`admitted_gaps`, `rejected_gaps`, `full_resets`. Full resets aliases existing
`resets`, including initial attachment, configuration and other faults. Rejected
gap events include policy-disabled, cold, clustered, excessive and unqualified
gaps; they are not a count of all reset causes. `last_event=overrun` can now also
describe an admitted event with ready output. No DM/actuator control is added.

### Deterministic results

Extended the existing shared exporter/scorer to include the actual production
option (`optional`) alongside the original `reset`, `gap1`, `gap7` cases. Four
source recordings are still generated once with the prior parameters: 8/4 kHz,
seed42/modeSeed123, single28 or full214, noise PSD1e-8, ADC2048/0.0083799/1.02,
lever10m, f0=28Hz, T=.1s, rotation0, query+1ms horizon. Per-sample pole radii
are unchanged between rates; physical decay time/realizations differ as before.
There are now 208 replays, including jitter-tag ideal/loss cases; old assertions
and historical prototype cases are preserved. Normal effective prediction
horizons remain exactly 1.125/1.250 ms at 8/4 kHz (nine/five truth samples).

With one missing sample/s, both rates count 59 lost frames over 60 s:

| Rate / population | Optional acceleration RMSE X/Y [m/s²] | Pointing RMSE X/Y [mas] | NRMSE X/Y | Correlation X/Y |
| --- | ---: | ---: | ---: | ---: |
| 8 kHz / single | 0.0003091 / 0.0003201 | 0.20602 / 0.21334 | 15.632% / 17.779% | 0.98782 / 0.98423 |
| 8 kHz / full | 0.0009304 / 0.0009658 | 0.62002 / 0.64363 | 43.159% / 49.212% | 0.91802 / 0.89286 |
| 4 kHz / single | 0.0002930 / 0.0002953 | 0.19525 / 0.19678 | 14.264% / 16.168% | 0.98983 / 0.98700 |
| 4 kHz / full | 0.0006653 / 0.0007547 | 0.44337 / 0.50293 | 46.961% / 57.694% | 0.89957 / 0.86655 |

Production-option numerical results match the offline single-frame prototype.
Additional coherent gain/phase change versus matching lossless output remains
<=0.0736% / 0.064° for one loss/s. Absolute full-population coherent target phase
errors remain about -2.91°/+6.13° at 8 kHz and -3.42°/+16.34° at 4 kHz; they
include existing stochastic envelope/interference/cancellation. Recovery does
not establish useful phase near signal nulls or physical correction performance.

| Schedule | Default availability | Optional availability | Optional admitted/rejected | Optional full resets |
| --- | ---: | ---: | ---: | ---: |
| Ideal, either rate | 99.499% | 99.499% | 0/0 | 1 |
| One loss/s, either rate | 69.950% | 99.450% | 59/0 | 1 |
| One loss at 4 Hz after t=1s | 1.166% | 99.303% | 236/0 | 1 |
| Three one-frame losses spaced 1ms every 5s | 93.953% | 93.963% | 11/22 | 23 |
| 2ms blackout every 5s | 93.963% | 93.963% | 0/11 | 12 |
| 3ms reader-only delay every 5s | 93.981% | 93.981% | 0/11 | 12 |
| One-frame losses every 100ms from startup | 0% | 0% | 0/599 | 600 |

These apply to both source populations. Initial warmup and missing worker
deadlines are included; availability after the 4-s cut is separately exported.
Retained-gap recovery is at the first post-gap query, versus default
300.375/300.250 ms. Dense/rejected gaps still require the original full warmup.
The reader-delay cases give 22 stale queries per recording/policy, all inhibited;
no ready prediction exceeds maxAge or target bounds.

True irregular-sampling analytic probes (`accelFFCtrlJitter_test.cpp`) use codes
evaluated at each actual jittered acquisition time, uniform 2-kHz query times,
and exact analytic truth at the requested target (no interpolation). At each
rate, 48 independently matured probes per profile give:

| Profile | Admitted | Rejected | Valid frames discarded |
| --- | ---: | ---: | ---: |
| Uniform | 48 | 0 | 0 |
| Three-phase jitter ±2% P | 48 | 0 | 0 |
| Three-phase jitter ±10% P | 0 | 48 | 0 |
| Bounded slew 2% P/interval | 48 | 0 | 0 |
| Bounded slew 5% P/interval | 6 | 42 | 0 |

Maximum retained analytic gain error is 0.003604%, phase error 0.001902°; fine
ADC calibration removes weak-signal quantization/noise as a confounder. This
qualifies synthetic timing/admission, not live jitter. Larger valid deviations
produce conservative resets rather than invalid-frame losses.

Separately, full/single stochastic replays perturb **timestamp tags** by ±2%/
±10% P while keeping the original discrete physical recording/ADC values and
nominal truth grid unchanged. They model timestamp measurement error, not a
new irregular-time stochastic model. ±2% admits all 59 isolated gaps and retains
99.45% availability. ±10% admits none and falls back to 69.95%; no additional
invalid-frame loss occurs. Full optional NRMSE with ±2% tags is
43.165%/49.211% at 8 kHz and 46.951%/57.689% at 4 kHz. The scorer compares to
the **same-jitter lossless baseline** and skips uniform-time covariance probes
for these tag-error cases rather than mislabeling their timing assumptions.

### Compatibility, safety and execution record

- Disabled mode matches all **92 archived binary files byte-for-byte**: default
  predictions, accepted histories and source truth for both rates/populations
  and the original 11 scheduling cases. The added diagnostics do not change the
  existing binary schema's legacy values.
- A temporary independent checker compiles the pre-change Input/Model under a
  separate namespace beside current disabled mode. It passed **66,002 exact
  legacy prediction/status comparisons** across normal inputs, counter gaps,
  bad ADC/timestamps/clocks, stale queries, rollback, restart and configuration.
  Source/artifacts are `/tmp/accelFFCtrl-optional-before/parity.cpp` and sibling
  baseline headers, derived from the starting `56fb71cd` source. Exact parity is
  a same-compiler/dependency result, not cross-Eigen/STL bit reproducibility.
- Ordinary current controller suite: **23 cases / 3,438,630 assertions passed**.
  Existing assertions were retained; both historical prototypes and production
  opt-in exercise the fault/phase tests. Added tests cover physical sequence
  identity, dense gaps, prior freshness without a stale query, cold/configuration
  resets, canonical/invalid Boolean settings, and wrong nominal rate (ordinary
  timestamp-driven fitting remains valid; gap retention rejects the mismatch).
- Combined ASan/UBSan simulator/controller: **57 cases / 3,675,228 assertions
  passed**; simulator DM cases use mock sinks only. Six-second expanded gap
  exporter under ASan/UBSan: **22,580,043 assertions passed**. No findings.
- Full 60-s expanded export: **230,903,647 assertions passed**. Seven existing
  Python scoring/oracle tests and four simulator PSD tests passed.
- Both numerical/input allocation guards passed with zero C++/Eigen allocations.
  The input benchmark also exercises actual production opt-in with 29 admitted
  gaps, 29 drops, one initial full reset and 239,971 accepted records out of
  240,000 source opportunities. Approximate workstation means were 128 ns
  default / 127 ns optional per opportunity; these noisy, differently scheduled
  fixtures are not comparative speed claims or native real-time guarantees.

The full application configuration/INDI test compile was attempted locally and
blocked before the application header by existing macOS framework dependencies:
unavailable `timegm` and missing installed `mx/math/floatUtils.hpp`. No dependency
or shared framework workaround is introduced. The source adds tests that load
the unchanged example (flag false), reject malformed opt-ins, parse true, and
verify the four read-only INDI fields, but they are **not counted as executed**.
Linux application/native build/INDI integration remains a required check.

Reproduce the main offline checks from the checkout root with an existing Python
environment containing NumPy/SciPy/pandas/Matplotlib:

```bash
accff_python=python3
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-optional-build test benchmark
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-optional-build \
  PYTHON="$accff_python" VALIDATION_DIR=/tmp/accelFFCtrl-optional-60s \
  VALIDATION_SECONDS=60 gap-validation
"$accff_python" -m unittest discover -s apps/accelSim/tests -p test_validate_psd.py -v
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-optional-asan test \
  CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  ACCELFFCTRL_VALIDATION_DIR=/tmp/accelFFCtrl-optional-asan-smoke \
  ACCELFFCTRL_VALIDATION_SECONDS=6 \
  /tmp/accelFFCtrl-optional-asan/accelFFCtrl_test '[accelFFCtrl-gap-export]'
```

The preceding combined sanitizer command also applies, adding
`apps/accelFFCtrl/tests/accelFFCtrlJitter_test.cpp` to its sources. Reports are
`/tmp/accelFFCtrl-optional-60s/{timestamp_guard.csv,report/accuracy.csv,
report/availability.csv,report/fit_quality.csv}` plus plots, compiler/STL and
Python version metadata. Raw recordings remain temporary artifacts, not repo
data. README lists exact build/application/native test commands for a configured
**non-instrument Linux development host**; none launches the live observer.

### Recommendation and remaining limits

The optional implementation is ready for review and a controlled observation-only
test **after Linux application/configuration/INDI and native tests pass**. Keep
the default disabled and depth one. The successful 4-kHz default observation
does not require enabling it. A controlled opt-in can measure admission/rejection
and availability under real scheduling, with conservative reset fallback; do
not present it as already live-qualified or broaden the cadence guard without
representative timestamp data. Per-frame recording and supported non-default
memory/frequency/rate settings still require qualification. No actuation or
hardware protection claim follows from these results.

Changed production files: `apps/accelFFCtrl/accelFFCtrlInput.hpp` and
`accelFFCtrl.hpp`. Changed tests: `tests/Makefile`, `accelFFCtrlGap_test.hpp/.cpp`,
`accelFFCtrlValidation_test.cpp`, `validate_gaps.py`, `accelFFCtrlInput_benchmark.cpp`,
`accelFFCtrl_test.cpp`, and new `accelFFCtrlJitter_test.cpp` under the app's tests
directory. Documentation: README and this plan. Numerical/source model headers,
native stream code, entrypoint, all configuration files, simulator/DM behavior
and the separate DDSPC worktree remain unchanged. Public counters/timestamps
still expose gaps for future DDSPC; private harmonic ordinals are not a uniform
DDSPC history. All text above Agent Findings and Plan remains unchanged.

The production gap and jitter cases are also registered in `tests/tests.list`;
their scoped `tests/Makefile.one` entries retain IEEE/nonfinite behavior with
`-fno-fast-math` and track the input/prototype headers. These two test-build files
are additional changes in this phase; other applications' rules are unchanged.

## Linux observation-test preparation and timestamp audit (2026-11-06)

### Scope and evidence boundary

Starting revision `dd3792fc` on `parkertjohnson1/accelSim` includes the previous
optional single-frame implementation. This phase modifies only tests, offline
tools and documentation. Production app/input/model/native-reader headers,
configuration files, all simulator files, and DM behavior remain unchanged.
No instrument host, tunnel, INDI command, operational stream, deployment,
application restart, commit, push or merge was accessed/performed.

The workstation is Darwin/arm64. No docker/podman/colima/limactl/orb/multipass/qemu
runtime executable or local Linux environment was found; `ImageStreamIO.pc` and
its native library are unavailable. Existing cached upstream ImageStreamIO
headers permit a macOS syntax-only check of the deterministic native source,
but do not supply an installed Linux ABI or compile the Linux-only paced branch.
Complete Linux app builds (both accelFFCtrl and accelSim), framework app tests,
native ImageStreamIO execution and native sanitizer results remain **pending**.
They must not be inferred from numerical/macOS tests.

Only prior synthetic accelFFCtrl validation recordings were found in the local
source/artifact paths searched. No representative instrument per-frame trace
was available. The user-reported 602-s 4-kHz run (2,407,458 processed, zero drops,
zero additional resets) remains aggregate evidence of successful default
observation, not timestamp-jitter or optional-gap qualification.

### Source-grounded recording requirements

- `accelFFCtrlStream.hpp::adcStream::{header,frame}` and
  `accelFFCtrlInput.hpp::inputProcessor::drain` define the stable owned-copy and
  metadata/clock validation boundary. At depth one the relevant timestamps and
  counter are global `md->{atime,writetime,cnt0}`, even for a 3-D depth-one cube.
  `cnt1` is zero. Keep before/after geometry/write/counter checks, original two
  UINT16 codes and rejected/busy/racing attempts. Notification counts cannot
  recover overwritten data.
- A sidecar's missing counter observations can differ from the controller's
  actual dropped records. Log controller acceptance and lifetime diagnostics
  at the input owner if exact controller reconstruction is required. A separate
  recorder process must claim its own semaphore, never drain a live consumer's
  slot. The cached ImageStreamIO implementation first searches for an existing
  same-PID slot; the new deterministic native test consequently uses one owned
  mapping for both serialized policy replays, rather than two same-PID owners.
- `libMagAOX/app/dev/frameGrabber.hpp` records CLOCK_REALTIME acquisition/write
  stamps, increments publication `cnt0`, clears write, then posts. The write
  stamp precedes final commit/post; publication visibility is bounded by copy
  validation/receipt, not precisely timestamped by that field.
- In `apps/ocam2KCtrl/ocam2KCtrl.hpp::frameGrabberPostPublish`, the sync-only
  1x1x1 UINT8 image mirrors the parent camera's counter/acquisition/write stamps,
  clears write and posts. It does not sample a separate sync-post timestamp.
  Confirm this is the actual `camwfs_sync` producer/version before interpreting
  a recording. An independent clock/TAI/default-library stamp is unsupported.
- `apps/accelSim/accelSim.hpp::{waitSync,syncTiming}` use the observed WFS counter
  and the simulator's own monotonic semaphore-receipt time as a phase reference.
  WFS acquisition timestamps alone do not reconstruct those reference arrivals.
  Exact sync replay also needs source generation/counter, actual simulator
  reference arrival, frozen deadline, phase target/applied slew and skipped
  ticks/discontinuities/configuration. Simulator acquisition is stamped before
  model advancement; skipped model ticks need not increment publication count.
  Neither acquisition wall time nor `cnt0/rate` is an exact live model-tick oracle.

README now specifies integer-only frame/query CSV schemas plus the larger
manifest/event contract: mapping device/inode/creation identity, raw timespec
parts for malformed stamps, clock anchors and uncertainty actually used, ADC
codes/calibration, nominal rates and all estimator/sync controls/revisions,
validation/reset reasons, logger overflow, query execution/state/source identity
and predicted values. Initial/final counters and process/boot boundaries are
explicit. Five-Hz snapshots cannot substitute for per-frame/query events.

No instrument recorder or production logging hook is added. A future recorder
needs preallocated bounded records and an asynchronous writer, independently
measured capture loss/overhead and development-host qualification. Per-sample
file/CSV I/O must remain outside the real-time worker. Stream-only observer data
can characterize source timing but cannot prove exact controller or simulator
internal event ordering.

### Offline timestamp-analysis implementation

New `apps/accelFFCtrl/tests/analyze_timestamps.py` reads local CSVs only. It parses
integer epoch nanoseconds/counters directly, rejects malformed/float fields and
decreasing receipt clocks, checks optional ADC/raw-timespec fields, and writes
per-frame `frames.csv`, `summary.json` and optional `timing.png`.

Cadence error is `(a_i-a_previous)/(cnt_i-cnt_previous)-P` on valid ordered pairs.
The drift slope is fitted on centered integer counter/time differences within
source generations/rollback segments; reported ppm is `(slope/P-1)*1e6`.
Distributions include RMS error, standard deviation, p50/p99, acquisition age,
write-minus-acquisition and receipt-minus-mapped-write latency. Clock-offset
changes are available per record and in plots. Missing publications are an
observed lower bound; rollback, duplicate and generation counts remain separate.

For gaps, the tool reports only necessary timing eligibility: depth one,
counter +2, monotonic valid acquisition/write/clock/ADC metadata, both previous
and new acquisition freshness, elapsed <=min(1ms,maxInterval), and the existing
`abs(dt-2P)<=.05P` guard. It deliberately does not claim actual admission without
estimator maturity/latest-query state, gap spacing, numerical conditioning and
complete reset history. Rejected candidates clear its conservative prior context.
Observed admission/rejection/reset totals come from logged lifetime counters,
not inferred timing eligibility.

Queries are checked at their actual requested target: `target=query+horizon`,
age `query-sample`, effective horizon `target-sample`. Ready rows exceeding
freshness/target bounds are counted and cause the CLI to fail. Unexecuted query
deadlines must be logged unavailable for scheduled-availability accounting.
Optional WFS input reports alignment relative to the latest valid observed
receipt/acquisition under the verified realtime contract, not an unrecorded
simulator wake/reference time. The WFS rate defaults to 2 kHz; `--wfs-rate` selects another documented reference rate.
The report includes WFS cadence/drift separately from observer-relative phase.

Seven new deterministic Python tests cover exact one-ns epoch preservation,
CSV rejection, both-rate cadence boundaries, clustered loss, restart/rollback,
stale previous samples, bad clocks/ADC/raw timestamps, drift, horizon-age
accounting and observed WFS alignment. They do not simulate instrument data.

Synthetic traced cadence-error RMS (counter-normalized per publication) is zero
for uniform acquisition; ±2% timestamp tags give 3.535 us at 8 kHz / 7.070 us at 4 kHz,
and ±10% tags give 17.677 us / 35.352 us. All five mature gaps pass necessary timing
eligibility at ±2%; none at ±10%. These are injected timestamp-tag errors, not
measured scheduler jitter. The CLI passed the valid exported trace and correctly
returned failure for a deliberately stale ready query.

### Native qualification prepared, not executed

`accelFFCtrlStream_test.cpp` retains the original deterministic native assertions
and adds a fifth case with 36 rate/geometry/fault combinations: 4/8 kHz, 2-D/3-D
depth one, single and clustered missing publications, writer busy, delayed reader,
invalid/stale acquisition, prediction during silence, counter rollback and
same-name replacement. Both policies read identical frozen native publications;
assertions verify drops, admitted/rejected gaps, full resets, generation changes,
physical prediction identity and analytic acceleration at the actual target.

The existing opt-in Linux paced test now covers eight five-second combinations:
4/8 kHz, depth one/256 and default/optional. Private producer/consumer processes
exercise native metadata/semaphores and natural scheduling/races. Each must
account for all publications as processed/dropped and observe readiness.
Summaries add admission/rejection, executed-query availability and service work
times. Runs have independent scheduler realizations and are not paired numerical
comparisons. Executed-query availability excludes unexecuted deadlines; the
deterministic replay provides scheduled availability. Catch/clock-probe overhead
and ordinary scheduling preclude a hard real-time or lossless-delivery claim.
Historical depth-256 fixtures change no operational circular-buffer setting.

New `tests/validate_linux.sh` (under the app directory) builds both full apps with
header tracing and forced app translation-unit rebuilds; runs framework config/
INDI handoff/default tests without app startup, numerical/allocation tests, both
native targets, replay/Python and numerical/native ASan+UBSan checks. It records
exact commands, platform/compiler/glibc/dependency/linked-library paths, source
revision/diff/untracked paths/hashes and outputs. It refuses non-Linux and known
instrument hostnames. No install or app launch command is present. The local
non-Linux refusal and `bash -n` checks pass. This is a reproducible pending Linux
qualification procedure, not evidence of a Linux pass.

On a configured non-instrument Linux host, with the reviewed source and actual
MagAO-X GCC 14/dependency installation:

```bash
CXX=g++ PYTHON=python3 VALIDATION_SECONDS=60 \
  bash apps/accelFFCtrl/tests/validate_linux.sh /tmp/accelFFCtrl-linux-report
```

Use a fresh report directory and return the full report, especially first failure,
all eight paced summaries, header/linked-library paths and test totals. Missing
`.pc`/headers/dependencies require investigation of the actual installation,
not replacement of shared system libraries. A source patch alone excludes
untracked files; copy any new scripts/tests with their reviewed relative paths.

### Refreshed replay, timing, regression and cost results

The existing `accelFFCtrlValidation_test.cpp` exporter/scorer still provides the
same 208 paired production/prototype replays and independent simulator truth.
No duplicate estimator/model/replay implementation was introduced. Optional
`ACCELFFCTRL_TIMING_TRACE=1` adds integer CSV fixtures for reset/optional paths,
single28, both rates, one loss/s and uniform/±2%/±10% timestamp-tag error. Binary
schemas remain unchanged. CSV I/O is offline only and excluded from cost results.

- Repeated 60-s export: **230,903,647 assertions**, all 208 scoring comparisons
  passed. All **124** archived expanded default prediction/history/truth binaries
  are byte-identical (including the prior 92-file subset and newer tag cases).
- At either rate/population with one missing frame/s: both policies count 59
  drops; default 60 full resets, 59 rejected, 69.95% availability; optional one
  full reset, 59 admitted, zero rejected, 99.45% availability. Recovery and other
  scheduling/phase tables in the preceding section are reproduced unchanged.
- Optional single28 NRMSE X/Y: 15.6315%/17.7792% at 8 kHz,
  14.2644%/16.1678% at 4 kHz. Full214 target-component NRMSE:
  43.1590%/49.2123% at 8 kHz, 46.9608%/57.6937% at 4 kHz. Accuracy is scored at
  the actual target, includes sensor/ADC/model interference, and is not a live
  hardware-correction claim. Default/optional masks differ; paired same-mask
  scores remain in the CSV alongside all-ready scores.
- Six-second traced export: **23,443,947 assertions passed**, including under
  ASan/UBSan. Analyzer audits at both rates check five lost frames each, no
  timing-fault candidates or invalid ready queries, and uniform/±2% eligible
  gaps versus ±10% ineligible gaps. Actual production admission matches that
  classification in these already-mature isolated cases. Uniform ready sample
  ages are exactly 125/250 us and effective horizons 1125/1250 us at 8/4 kHz:
  measurement age is included once. Cold start still costs 300 ms; optional
  six-second availability is 94.95%, versus 99.45% over 60 s.
- Current ordinary controller suite: **23 cases / 3,438,630 assertions passed**.
  Combined simulator/controller ASan/UBSan: **57 cases / 3,675,228 assertions**.
  New trace-export branches also pass standalone sanitizers; no findings.
  **18 Python tests** pass (14 controller scoring/timing plus four simulator PSD).
- Allocation-guarded benchmark now uses identical 30-s one-loss/s inputs for
  both policies at both rates, preserving its lossless case. Zero C++/Eigen
  allocations; 29 drops per policy, default 30 resets versus optional 1 with 29
  admitted. Workstation mean service per source opportunity: default/optional
  ~104/126 ns (8 kHz), ~158/200 ns (4 kHz), p99 <=458 ns. Default spends more time
  warming; query/source ratio doubles at 4 kHz; the existing cost fixture uses f0=28.5 Hz. These exclude native waits/mapping,
  actual clock syscalls, INDI/capture I/O and scheduling, and establish no Linux
  latency guarantee.

Reproduce the local offline work from the checkout root:

```bash
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-linuxprep-build test benchmark
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-linuxprep-build \
  PYTHON=python3 VALIDATION_DIR=/tmp/accelFFCtrl-linuxprep-60s VALIDATION_SECONDS=60 gap-validation
ACCELFFCTRL_TIMING_TRACE=1 ACCELFFCTRL_VALIDATION_DIR=/tmp/accelFFCtrl-linuxprep-timing \
  ACCELFFCTRL_VALIDATION_SECONDS=6 /tmp/accelFFCtrl-linuxprep-build/accelFFCtrl_test '[accelFFCtrl-gap-export]'
python3 apps/accelFFCtrl/tests/analyze_timestamps.py \
  /tmp/accelFFCtrl-linuxprep-timing/8000/optional/single28_loss1_1s_1000000_frames.csv \
  --queries /tmp/accelFFCtrl-linuxprep-timing/8000/optional/single28_loss1_1s_1000000_queries.csv \
  --sample-rate 8000 --output /tmp/accelFFCtrl-timing-report --plots
make -C apps/accelFFCtrl/tests PYTHON=python3 validation-test
python3 -m unittest discover -s apps/accelSim/tests -p test_validate_psd.py -v
```

Use an existing Python environment with the documented scientific packages.
Actual local executable was Anaconda `python`; no dependencies were installed.
The previous combined sanitizer command still applies with gap/jitter sources;
the Linux script contains its complete updated source list. Standalone trace
sanitizer builds use the same existing Makefile ASan/UBSan flags, then run the
above hidden exporter with halt-on-error enabled. Logs/artifacts are under
`/tmp/accelFFCtrl-linuxprep-*`; full accuracy/availability/fit CSVs and recovery
plots are in `...-60s/report`, timing audits/plots in `...-timing/timing-report`.

### Remaining deployment gates and changed files

The optional core remains suitable for a controlled observation-only comparison
conditional on Linux full-app/config/native validation. Timestamp qualification
remains synthetic. Before deployment: run/review the Linux report; verify actual
producer clocks/ABI/depth and matching startup `input.sampleRate`; qualify the capture design/overhead on a development
host; obtain human-recorded representative timing and reconcile source/controller/
logger loss and reset/readiness histories; then request explicit authorization
for human deployment and a bounded observation-only opt-in. Keep the unchanged
default false, depth one and conservative guard. The successful lossless 4-kHz
default run is not a reason to enable tolerance or broaden the guard.

Changed files: `apps/accelFFCtrl/tests/{Makefile,accelFFCtrlStream_test.cpp,
accelFFCtrlValidation_test.cpp,accelFFCtrlInput_benchmark.cpp}`; new
`analyze_timestamps.py`, `test_analyze_timestamps.py`, `validate_linux.sh` in that
directory; `apps/accelFFCtrl/README.md`; this plan. No production file or
configuration changed. Text above Agent Findings and Plan remains unchanged.

## Offline feed-forward DM cancellation (2026-11-06)

### Authorization, source review and architectural boundary

Starting revision `027949bb`, branch `parkertjohnson1/accelSim`, was clean. The
user authorized an entirely offline correction component and causal simulation,
with zero default gain and no production integration. All existing production
source/configurations remain unchanged, including simulator model/RNG behavior,
DM injection, observer numerics, timing and the default-disabled gap policy.
No Linux environment, instrument access, stream writes, DM commands, deployment,
commit, push or merge are involved. The new code is not included by the live app.
All earlier text above Agent Findings and Plan remains unchanged.

Reviewed interfaces and conventions:

- `accelFFCtrlModel.hpp::{prediction,harmonicEstimator}`: pointing is rotated
  physical disturbance in radians, not a correction. Sensor acceleration is
  `(ADC-zero)*voltsPerBit/sensitivity` [m/s²]. Harmonic inversion is
  `-R*a/(omega²*L)`. Amplitude is peak acceleration; phase is at the acquisition
  timestamp. Prediction target is explicit and includes acquisition age once.
- `accelFFCtrlInput.hpp::inputProcessor`: full production counter/clock/ADC and
  freshness validation, numerical reset and optional gap handling. Reused
  unchanged in all raw-only causal study replays; no alternative fitting kernel.
- `accelSimModel.hpp::{model,adcCounts,accelerationFactor}`: record physical
  source truth and noisy quantized ADC once, with independent future truth
  padding. Only raw pairs are extracted into the input replay's argument.
  Independent scoring/oracle code can read truth; no truth field is accessible
  through the production input processor or the correction component.
- `accelSimDisturbance.hpp::tipTiltProjection`: existing weighted column-scaled
  QR solves `B*K=0.5e6*[x,y]`, mapping physical radians into surface-micron mode
  coefficients. This applies reflection once; no extra diameter/gain/reflection
  factor belongs in command synthesis. The synthetic fixture reuses its
  `response()` and recovers K in the supplied B basis; it does not fit a new
  physical calibration or load deployed FITS.
- `apps/dmMode/dmMode.hpp::sendCommand`: complete surface image is the sum of
  modal amplitudes times calibration planes. `apps/hoPredCtrl/hoPredCtrl.hpp`
  loads mapping/calibration and writes a complete shaped contribution.
  `libMagAOX/app/dev/dm.hpp` consumes the combined stream and delegates hardware
  commands. A future correction is a complete additive contribution, not an
  accumulated increment or a replacement for AO/flat ownership.
- The existing `agents/plans/2026-10/accelSim-dm-injection.md` records user-supplied
  additive combiner behavior: changed summed channel counters, AveMode=0,
  DClevel=0, trigger mode 4/delay 10 us. This is historical supplied evidence, not
  a measured dynamic response or independently available deployed combiner code.
  Exclusive ownership, AO combination and physical channel behavior remain a
  future integration gate. `loPredCtrlAcc` remains absent in this checkout;
  the present `loPredCtrl`/`hoPredCtrl` sources do not provide a calibrated
  acceleration-to-pointing cancellation plant that could substitute here.

No common-library/projector extraction is needed for this isolated test phase.
`apps/accelFFCtrl/offline/accelFFCtrlCancellation.hpp` depends only on the
numerical prediction API and Eigen. It contains no ImageStreamIO, FITS, INDI,
clock, application lifecycle or publication interface. Existing single-band
production observation is unchanged; only the offline dual-frequency harness
instantiates two independent readers/estimators.

### Equations, explicit synthetic calibration and validity

For query time t and common target t+h:

```text
theta_sum(t+h) = sum of coherent ready predicted physical bands [rad]
u_FF(t) = -g*K*theta_sum(t+h)                       [modal coefficient]
surface(t) = B*u_FF(t)                             [surface micron]
residual(t) = theta_disturbance(t) + A_true*x_DM(t)  [rad]
```

The opposing correction sign follows the already-negative acceleration-to-angle
inversion. With identity registration, correction is therefore proportional to
positive estimated acceleration. The new tests verify the signs, coordinate
rotation, reflection scale and complete residual rather than introducing a
hardware sign convention. No telescope/WFS registration or physical DM gain is
assumed from this synthetic example.

Synthetic calibration is deliberately explicit: eight equally spaced azimuths
at pupil radius `6.5/(2*sqrt(2))` m, unit weights, and two modes
`B0=(4/D)*(cos(phi)*x+sin(phi)*y)`,
`B1=(4/D)*(-sin(phi)*x+cos(phi)*y)`, with D=6.5 m and phi=30 deg.
The discrete quadrature gives each mode one micron surface RMS at coefficient
one and zero cross product. The existing projector yields
`K=(D*1e6/8)*R(-phi)`; nominal synthetic `A=K^-1` [rad/coefficient].
These are synthetic quadrature locations, not physical actuators or approved
on-instrument constants. True-plant errors multiply A by a gain and output-axis
rotation; the wrong-sign case uses -A. No real calibration file is opened.

`correctionParameters` defaults gain and modal bounds to zero; explicit mapping
is required. It configures gain 0..1, horizon, max age/target and per-mode absolute
coefficient bounds. The baseline bound 0.1 is nonbinding; bound 0.001 exercises
clipping. A common limiting scalar preserves proposed modal direction. These
are user-requested **offline experiment controls**, not hardware protection or
reintroduction of removed accelSim actuator/P2V limits.

`feedForward::propose` is stateless and returns a fresh zero on disabled/invalid
input. All bands must be ready, finite and fresh, with matching acquisition,
source generation/counter and target exactly equal to query+h. Empty input,
nonfinite calibration/control/data, bad bounds, future/stale acquisition or
inconsistent targets/identities cannot silently retain a prior command. Full
multi-band output is inhibited if any configured band is unavailable.

Synthetic DM dynamics at dt=125 us with integer pure delay d are:

```text
a = exp(-2*pi*bandwidth*dt)
x[n] = a*x[n-1] + (1-a)*u[n-d-1]   (bandwidth > 0)
x[n] = u[n-d]                      (explicit ideal bandwidth=0 case)
```

This is exact first-order evolution under the preceding ZOH input interval.
A finite-bandwidth plant has no response at the instant of a new input before
an interval elapses. Delay must be divisible by dt; histories are bounded to
one million steps at construction. Reset clears only this simulated history.
A nonfinite plant input clears simulated state and throws. Commands are held
between 2-kHz queries. Nominal delay 1 ms/bandwidth 300 Hz are chosen simulation
parameters, not measured MagAO-X transfer functions. The new component uses
fixed-size arithmetic and a startup-allocated delay line, with no per-step I/O.

An inhibited proposal is a zero at the plant input. Previously issued commands
can still emerge through real-model delay or settle naturally; they are not
retroactively erased. This separates software inhibition from physical settling.
No inverse plant or adaptive gain is applied. A longer prediction horizon can
anticipate phase but does not automatically compensate plant attenuation.

### Causal experiments and scoring

The standalone exporter uses 8-kHz acquisition and 2-kHz queries, depth-one
owned snapshots and original production input/numerics. Query precedes current
sample consumption; the newest ready acquisition is 125 us old. Horizon1 ms means
an effective sample-to-target interval1.125 ms. Targets are on the exact 125-us
truth grid; there is no rounding, interpolation or extra sample-age addition.
The raw replay takes a separate vector of two UINT16 codes, with no truth
members. Its only inputs are those codes, known calibration and ordered times.

Recordings use existing model seed 42/modeSeed 123, lever 10 m, per-mode/axis
acceleration RMS 0.002 m/s², pole 0.99999, WFS couplings 1, ADC 2048/0.0083799/1.02 and
noise PSD 1e-8. Cases are isolated 28, isolated 56 and both together; no214-mode
claim is made in this phase. Independent coherent analytic cases use the same
acceleration/ADC conversion with volts-per-bit1e-5 and noise 0 to isolate
mathematics from coarse quantization. Existing producer defaults are never edited.

Every source is generated once (with 200 future samples for independent scoring),
then reused across gain/plant conditions. No controller call advances a model/RNG.
Raw-only predictions are cached by input fault/policy and horizon, not gain or
plant. The separately named oracle substitutes exact truth at the target in
`evaluate` only; it never enters the production readers. Oracle gain curves use
linear scaling of that plant effect below nonbinding bounds. They are an explicitly
noncausal perfect-prediction baseline, not an implementable controller.

76 runs cover 0..1 gain in0.1 steps; both single frequencies and dual bands;
perfect prediction; horizons0,1,1.75 ms; pure delay 1,5,20 ms; bandwidth 0,50,300 Hz;
wrong sign; +20% calibration gain;15-degree rotation error; nominal/tenfold/zero
sensor PSD; one omitted ADC frame/s; three-frame clusters; bad ADC 4096;10-ms
input silence; explicit reset/generation reattachment; and a tight modal bound. Input-fault
queries continue on schedule. Dropped counts and all invalid proposal zeros
are checked, and all observed source frames must be processed or accounted lost.

The 60-s study scores every 8-kHz residual sample from 4s onward, including inhibited
intervals. Availability covers all 120,000 scheduled queries including cold start.
The simulated WFS is an ideal zero-exposure linear TT observation at 8 kHz with
unity coupling. It is not a model of real2-kHz WFS integration/readout or AO
feedback. Before/after RMS retain bias; variance reduction uses mean-subtracted
variance. Peak complex phasors use the same complete scoring interval; residual
amplitude/phase and opposing correction amplitude/phase are separate quantities.
Prediction error is computed only on ready query targets against exact future
truth, with availability also reported so censored errors are not mistaken for
continuous correction. PSDs use Welch8-s Hann windows, exact 28/56-Hz bins and
an additional ±1-Hz integrated band metric. No unavailable records are compressed
into a fake uniform time series.

### Quantitative results: nominal gain 1, horizon/delay 1 ms, bandwidth 300 Hz

| Source | Before RMS X/Y [mas] | Residual RMS X/Y [mas] | Variance reduction X/Y | Carrier-bin PSD suppression X/Y [dB] |
| --- | ---: | ---: | ---: | ---: |
| 28 Hz | 1.31778 / 1.19992 | 0.28479 / 0.27031 | 95.3295% / 94.9253% | 16.370 / 16.666 at 28 |
| 56 Hz | 0.44531 / 0.28239 | 0.12955 / 0.09463 | 91.5362% / 88.7692% | 11.467 / 10.997 at 56 |
| 28+56 Hz | 1.29912 / 1.17524 | 0.29396 / 0.28209 | 94.8803% / 94.2388% | 16.695 / 16.766 at 28;11.066 / 11.073 at 56 |

Integrated ±1-Hz suppression is 15.028/14.956dB (single 28),11.165/10.497dB
(single 56),15.349/15.127dB at 28 and10.817/10.753dB at 56 for dual bands.
Finite-record AR RMS differs from ensemble RMS; no realization rescaling occurs.

| Source | Target prediction RMSE X/Y [mas] | NRMSE X/Y | Opposing phase error X/Y | Opposing amplitude error X/Y |
| --- | ---: | ---: | ---: | ---: |
| 28 Hz | 0.20593 / 0.21338 | 15.6268% / 17.7829% | -8.223 / -7.703 deg | -0.286% / -0.865% |
| 56 Hz | 0.05325 / 0.05453 | 11.9571% / 19.3089% | -15.895 / -14.948 deg | -2.008% / -2.459% |
| Dual,28 component | 0.21655 / 0.21271 combined | 16.6690% / 18.0997% combined | -8.236 / -7.804 deg | -1.248% /+2.564% |
| Dual,56 component |same combined |same combined | -16.617 / -16.173 deg | -8.309% / -5.571% |

Corresponding residual amplitude ratios are 0.1432/0.1340 (28),0.2745/0.2581 (56),
0.1433/0.1402 (dual 28),0.2889/0.2790 (dual 56). Residual phase relative to truth is
84.748/82.456 deg,77.898/77.106 deg,80.896/96.612 deg and65.160/70.513 deg,
respectively. These phasors are finite-record projections; phases near nulls
should not be interpreted as calibrated instrument timing.

Dual-band correction modal RMS is 0.005691/0.003830 coefficient, peaks
0.016424/0.012244; maximum absolute surface at the eight synthetic test points is 0.025205 um. Single28
surface peak0.022424 um; single 56 peak0.007330 um. The0.1 baseline coefficient
bounds never engage. A0.001 bound engages117,135 queries and reduces dual-band
variance by only25.91/24.66%, quantifying the imposed nonlinear clipping.

### Perfect prediction, compensation and robustness

At nominal 1-ms horizon the oracle achieves98.116/98.119% variance reduction
(single 28),92.669/92.668% (single 56),97.720/97.600% (dual). Thus nominal 56-Hz
residual is substantially plant/ZOH phase error, rather than just prediction
noise. With1.75-ms horizon, causal reductions become97.539/96.831%,
98.558/96.285%, 97.197/96.696%; oracle reductions become99.993%, 99.952%,
99.990%. Both bands share the target; no frequency-dependent inverse plant is
inserted. The analytic single-tone cases track their oracle closely. Analytic
dual-band NRMSE remains about 1.4% due to leakage between independent fits;
changing to a joint multi-harmonic fit is outside this phase.

Gains0..1 are finite/bounded and zero gain reproduces disturbance exactly.
Gain1 is best among that discrete grid for the nominal source/plant in both
axes; this is not a recommended physical setting. For dual bands:

| Perturbation | Variance reduction X/Y |
| --- | ---: |
| Reversed correction sign | -290.26% / -289.68% (amplification) |
| Actual delay 5 ms, horizon 1 ms | 21.18% / 17.61% |
| Actual delay 20 ms, horizon 1 ms | -257.94% / -250.77% |
| Bandwidth50 Hz | 67.19% / 65.96% |
| Plant gain+20% | 90.41% / 89.63% |
| Output rotation error15 deg | 90.30% / 88.45% |
| Sensor PSD 1e-7 (tenfold) | 81.39% / 79.22% |
| Noise0, unchanged coarse ADC | 76.81% / 64.79% |
| Ideal delayed DM, bandwidth 0 | 96.99% / 96.47% |
| Horizon0, nominal plant | 85.49% / 84.35% |

Removing noise can worsen sub-count quantized estimation because dither changes;
this is not an instruction to change physical noise/calibration. Real DM
resonances, hysteresis and feedback from DM into accelerometers are unmodelled.
This is feed-forward into a stable prescribed plant, not a demonstrated real
closed-loop stability margin. Wrong signs/latencies amplify variance while
remaining finite; gain bounds alone do not establish useful cancellation.

| Input event | Drops | Full resets per reader | Admitted/rejected gaps | Prediction availability | Variance reduction X/Y |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ideal | 0 | 1 | 0/0 | 99.4992% | 94.88% / 94.24% |
| One ADC loss/s, default | 59 | 60 | 0/59 | 69.9992% | 66.12% / 65.99% |
| One ADC loss/s, optional | 59 | 1 | 59/0 | 99.4992% | 94.88% / 94.24% |
| Three consecutive losses every 5s, either policy | 33 | 12 | 0/11 | 93.9992% | 89.22% / 88.61% |
| One invalid ADC frame | 1 | 2 | 0/0 | 98.9992% | 94.16% / 92.38% |
| Input silence10 ms | 80 | 2 | 0/1 | 98.9850% | 94.15% / 92.32% |
| Explicit reset | 0 | 2 | 0/0 | 98.9983% | 94.16% / 92.38% |
| Generation reattachment | 0 | 3 | 0/0 | 98.9983% | 94.16% / 92.38% |

Cold start remains 300 ms. Here only ADC ingestion is omitted while prediction
deadlines keep running: availability consequently differs from the preceding
69.95/99.45% study that also missed query deadlines. Every disabled/inhibited
proposal is explicitly zero; traces and scoring verify this even after a prior
nonzero proposal. The physical response may continue through delay/decay as
expected. The production default gap policy was not changed.

### Tests, artifacts and reproduction

- Controller suite:28 cases / 3,440,057 assertions passed, including five new
  cancellation cases. New tests cover analytic K/surface units, sign/gain,
  default zero gain, invalid controls/mapping, common limits, empty/mismatched/
  nonfinite/stale prediction inhibition, zero after a prior command, delay/step
  response and steady-state complex first-order transfer at 28/56 Hz. The
  perfect-prediction ideal-plant sinusoidal residual agrees with the analytical
  formula averaged over the four command-hold phases at gains 0,0.5,1. A causal
  two-tone raw-ADC test exercises startup and explicit reset through both real
  production readers and achieves residual variance ratio<0.02.
- Combined simulator/controller ASan+UBSan: 62 cases / 3,676,655 assertions passed;
  simulator DM tests use existing mock sinks only. The independent 76-run 8-s
  exporter also passes sanitizers and its Python numerical/PSD cross-checks.
- Python: 17 controller scoring/timing tests plus four simulator PSD tests pass.
  Three new scoring cases verify signed sinusoidal gain/phase, exact carrier and
  integrated PSD dB conventions and absence-of-mode labelling.
- The full 76-run 60-s study and all selected raw binary/RMS/phase/PSD cross-checks
  pass. Existing 208 gap replays pass again using 6-s recordings; all 124 archived
  default prediction/history/truth binaries remain byte-identical. The normal
  numerical/input allocation benchmarks pass with their original checksums,
  zero C++/Eigen allocations and accurate default/optional loss accounting.
- Root application-test registration compiles the new test object with scoped
  `-fno-fast-math`. No original assertion is weakened. Existing Linux app/native
  validation remains pending; no instrument or native-stream test is run here.

From the checkout root (use the available existing scientific Python environment;
locally this was Anaconda `python`, with Apple Clang 21/Eigen 5.0.1/libc++ 210106):

```bash
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-cancellation-build test benchmark
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-cancellation-build \
  PYTHON=python3 VALIDATION_DIR=/tmp/accelFFCtrl-cancellation-final-60s \
  VALIDATION_SECONDS=60 cancellation-validation
make -C apps/accelFFCtrl/tests PYTHON=python3 validation-test
python3 -m unittest discover -s apps/accelSim/tests -p test_validate_psd.py -v
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-cancellation-asan \
  CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' test
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-cancellation-asan \
  CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' \
  /tmp/accelFFCtrl-cancellation-asan/accelFFCtrlCancellation_export
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  /tmp/accelFFCtrl-cancellation-asan/accelFFCtrlCancellation_export /tmp/accelFFCtrl-cancellation-asan-smoke 8
python3 apps/accelFFCtrl/tests/validate_cancellation.py /tmp/accelFFCtrl-cancellation-asan-smoke --no-plots
```

The previous combined sanitizer command applies with the new cancellation source;
`tests/validate_linux.sh` under the app now includes it in its explicit combined
source list. Nothing in that script was run on Linux/instrument hardware here.
Export duration is 8..600 s (smoke 8), score starts 4 s. Artifacts include metadata,
`summary.csv`, selected uniform 8-kHz float64 trace binaries, `report/metrics.csv`,
`spectral_suppression.csv`, per-axis PSD CSVs, original/residual/command/readiness
PNGs and `gain_sweep.png`. Large generated artifacts remain under `/tmp`, not git.

### Limitations, integration gates and changed files

The offline causal controller successfully suppresses both isolated frequencies
and their combination under the explicit synthetic plant. This establishes no
physical DM calibration, on-instrument suppression, native Linux timing or AO
stability result. Full 214-mode correction, clock-jitter cancellation, WFS
integration/readout, actual driver/combiner dynamics and robust automatic gain/
band selection are not qualified. The current common-horizon two-bank fit also
has measurable cross-band leakage. No physical constants should be inferred
from the chosen 300-Hz/1-ms synthetic transfer function.

Before production integration, review this result and approve a separate design
phase. That phase needs approved full-image B/K calibration with measured
sensor/pupil/DM/WFS registration/sign/units, measured source-age/combiner/driver/
WFS transfer timing, existing AO-loop interaction analysis, exclusive additive
channel ownership, validated Linux/native operation and explicit enabled/fault/
disable/shutdown behavior. No publication path, live gain control or new
hardware limit is supplied in this phase. Default observer behavior remains
observation-only and gap tolerance remains disabled.

Changed files: new `apps/accelFFCtrl/offline/accelFFCtrlCancellation.hpp`; new
`tests/accelFFCtrlCancellation_test.hpp`, `_test.cpp`, `_export.cpp`,
`validate_cancellation.py`, `test_validate_cancellation.py` under the app;
app test `Makefile` and `validate_linux.sh`; root `tests/Makefile.one` and
`tests/tests.list`; both app READMEs; this plan. Runtime source, existing
numerical/source model headers, configuration files and the separate DDSPC
worktree remain untouched. No staging, commits, pushes or deployment occurred.

## DM integration preparation and shadow validation (2026-11-06)

### Scope and evidence

Starting revision `5e227a47` on `parkertjohnson1/accelSim` was clean. The user
approved integration **preparation**, an isolated host-memory shadow prototype,
AO interaction studies and an observation-only protocol. No live writer, instrument
connection, configuration change, application restart or deployment is authorized.
All work ran on the macOS workstation. Runtime app/source/model/configuration
files remain byte-identical; default depth one and `allowSingleFrameGap=false`
are preserved. No commit, push, merge or staging was performed.

The detailed source inventory, calibration specification and proposed human-run
validation protocol are in
[the integration report](../../../apps/accelFFCtrl/offline/INTEGRATION.md).
It distinguishes code facts, historical owner guidance, local archived basis
validation and unmeasured physical transfer functions. This phase does not alter
or supersede the earlier record of the simulator's assigned disturbance channel.

### Source-grounded decisions

- The tweeter/DM01 is the supported candidate because the existing TT basis,
  `accelSim::prepareDM`, `dmMode` and BMC driver surface command path provide an
  established numerical convention. This is not measured bandwidth/headroom
  qualification versus the woofer. No new correction channel is assigned;
  `dm01disp10` belongs to the historical simulator experiment and must not be
  assumed available for correction.
- `dmMode::sendCommand` and `hoPredCtrl::send_dm_command` replace full channel
  contributions and increment writer-local counters. The eventual FF contribution
  is `B*u_FF`, not a time increment or an overwrite of AO/flat. `dm::makeDelta`
  computes output relative to non-delta contributions and a requested/applied
  difference; `deltaChannels` classification is not incremental write semantics.
- Existing code routes the combined surface through `dm::processImage` to
  `bmcCtrl::commandDM`, where address/volume/actuator conversion and clipping
  occur. No voltage or driver gain belongs in the angular/modal proposer.
  Woofer/TCS offloading and TTM modulation have different roles/calibrations;
  their gains and masks are not pointing calibration constants.
- Prior owner evidence supplies the combiner's changed-summed-counter behavior,
  AveMode=0/DClevel=0/trigger mode 4/delay 10 us. No deployed combiner implementation,
  current settings or dynamic latency were observed here. Existing publication
  counters and time fields are not hardware delivery acknowledgements.
- `accelSimDMStream` assigns ADC acquisition time before active UpdateIm;
  `dmMode` does not assign acquisition/target time in its direct write path.
  Neither defines scheduled physical application at a prediction target. The
  installed library's final timestamp behavior and combiner/driver sequence
  correspondence require Linux/native and optical qualification.
- `wooferTweeterRecon` names `aol1_modevalWFS`, `aol1_modevalDMf_mon` and
  `aol0_modevalDMf_mon`. Its WFS `writetime-1/fps+offset` versus DM `atime+offset`
  demonstrates explicit timing assumptions, not a measured delay to reuse.
  A WFS modal coefficient is not automatically radians or pupil X/Y.
- The physical chain requires calibrated ADC response, effective mechanical
  acceleration-to-angle transfer, reported-frame-to-DM/WFS registration and sign,
  current B/mask/modal order/units, measured DM/WFS response, total timing and AO
  state/headroom. `Q(omega)` and `J(omega)` in WFS units would give
  `u_FF=-g*J^+*Q*theta_pred`; alternatively validated static C/K plus a measured
  dynamic model can be used. No missing Q/J/C, physical delay or bandwidth is
  replaced with synthetic data. Estimator rotation already applied to pointing
  must not be applied twice.

### Locally available archived calibration

The archived FITS pair and mask exist under `scratch/accelSim_dm_calib`, with
unchanged SHA-256 identities matching the earlier owner-provided record:

```text
CMmodesDM.00.fits a5bb41b535fe46cb225c17f1086dbf9ac190cf3bd451379e9551ffb77383daf9
dmmask.fits      0970dab118bcdb73236ae045da93c550288a4d632b70b2748903272f2aa166f8
```

Read-only Astropy/NumPy reanalysis uses the **previously owner-defined DM-pupil**
coordinates: center 24.5, D=6.5 m over 48 pitches, +X column/+Y row. Weighted SVD of
`B*K=.5e6*[x,y]` agrees with the archived C++ projector columns and yields:

```text
K = [[-798911.75116813, -315942.59493194],
     [ 318826.69049347, -792420.64908820]] coefficient/radian
relative fit residuals = [8.468786589e-8, 8.634744717e-8]
masked mode RMS = [0.999999926912, 0.999999991558] surface micron
```

No normalization, piston removal, output masking or offloader mask substitution
occurs. The six signed/mixed-angle examples pass a new C++ shadow fixture (32
assertions), also under sanitizers. This establishes numerical pupil mapping
only: accelerometer/telescope/WFS registration, optical sign, dynamic transfer
and current operational calibration identity remain unmeasured. The fixture is
labelled `archivedPupilOnly`; it never enters the synthetic cancellation runs or
an actual command path. Scratch files are local archived evidence, not guaranteed
current versioned instrument calibration.

### Prototype and test design

New `offline/accelFFCtrlShadow.hpp::shadowCommand` composes explicitly supplied
C (reported pointing radians to DM-pupil radians) and K (modal coefficients/rad)
and delegates to the unchanged offline `feedForward`. It accepts the existing
`prediction` interface and has no stream, file, clock, application or actuator
API. Gain and bounds default zero. A zero-bound validity checker reuses the
existing numerical validation even at zero proposal gain, so observation records
retain qualified predicted/mapped angles without emitting a command.

Every record contains exact integer query/acquisition/input-target/requested-target
times, source generation/counter, log ordinal, reset epoch, input revision and
calibration identity/evidence, along with numerical validity, faults, limiting
scalar and absolute modal coefficients. Invalid, stale or nonfinite input gives
zero; no previous proposal is reused. All configured bands must share source/time
identity. A source/query rollback or generation change latches inhibition until
reset; counter-to-acquisition inconsistencies latch too. Reset requires a newer
acquisition and advances an epoch separately from configuration revision. The
study's reset/restart keeps actual input revision 0. Source counter advances of
four between 2-kHz queries on 8-kHz input are normal, not a drop detector; the
unchanged input processor retains drop accounting and warmup authority.

The old cancellation exporter optionally writes shadow records/manifests only
when `ACCELFFCTRL_SHADOW_DIR` is supplied. Every modal/state result must match
the original proposer. Diagnostics preserve forecast horizon rather than adding
sample age twice. Blocking file output is entirely offline; it is not a proposed
real-time logger or live-app hook. Physical qualification is never inferred from
`active`, `prediction_valid` or a provenance ID.

Five new ordinary C++ cases cover coordinate rotations/reflections, actual modal
order/scale, repeated absolute commands, zero gain, limits, invalid/nonfinite/stale
input, target mismatch, query/source/sample ordering, reset/reattachment barriers
and a synthetic AO integrator's analytical DC dynamics/back-calculation. One
additional hidden case validates the locally generated archived-pupil fixture.
Doxygen-only references retain real shadow API links through the test fixtures.

### AO interaction results and limits

The new standalone `accelFFCtrlAO_export` consumes the **recorded causal** dual-band
commands from the previous cancellation study. It reuses the existing synthetic
`dmResponse` and a separate test-only AO integrator driven by the previous WFS
sample. It does not replace or model the actual CACAO kernel. All gains, one-query
WFS latency, 1/3-ms plant delays, 300-Hz bandwidth and 0.03/0.003 modal headroom
proxies are explicit synthetic choices. No physical values are inferred.

For a compatible unsaturated linear AO loop, `r=S*(d+P*u_FF)` with
`S=(I+P*C_AO)^-1`; an integrator does not inherently double-correct exact FF.
Duplicate FF writers, incorrect POL accounting, saturation, unknown latency and
modal cross coupling can instead produce adverse interaction. Real reconstruction
must include all actually applied contributions and response/exposure timing.

All 15 60-s interaction cases passed their numerical/recording audits. The
AO-only and shadow-only runs match **every full-rate AO command and residual**
by direct comparison; fingerprints and independently read 2-kHz traces agree too.
The shadow-only file contains nonzero proposed diagnostic commands but applies
none. Existing production app sources/configurations remain unchanged; actual
Linux logging/scheduling overhead is not qualified by this offline equality.

| Synthetic case | Variance reduction X/Y | Diagnostic consequence |
| --- | ---: | --- |
| AO 0.1 only / shadow only | 25.47% / 23.14% | Exactly identical trajectories. |
| FF only | 94.88% / 94.24% | Matches original recorded cancellation. |
| AO 0.1 + FF | 95.75% / 95.09% | Compatible residual feedback helps in this prescribed plant. |
| AO 0.3 only / + FF | 86.79% / 85.66%; 99.07% / 98.89% | Not measured live gains. |
| Duplicate FF | 20.17% / 17.22% | 34 saturated queries. |
| Wrong sign | -190.69% / -198.65% | Amplification. |
| Incorrect incremental FF | -1873.94% / -1993.77% | 105121 saturated queries. |
| Tight total headroom | 17.15% / 4.68% | 118092 saturated queries; individual FF bounds do not cover total. |
| Illustrative back-calculation | 62.97% / 60.66% | Candidate integrator peak 0.06025→0.00719 coefficient. |
| Additional actual delay 3 ms | -1611.13% / -2332.86% | Clipped poor trajectories. |
| AO gain 0.8 | -1606.50% / -1997.33% | Clipped output is not a stability margin. |

FF disable and AO reset halfway through the record expose settling/state changes.
Full-rate RMS scores include all intervals after 4 s; diagnostic PSDs use uniformly
sampled 2-kHz WFS outputs. There is no real WFS exposure, resonance, actuator/flat
clipping model or DM-to-accelerometer feedback. These sensitivity cases neither
establish closed-loop stability nor recommend a physical gain or anti-windup rule.

### Validation record and reproduction

- Normal controller suite: **33 cases / 3,440,635 assertions passed**.
- Combined simulator/controller ASan+UBSan: **67 cases / 3,677,233 assertions**,
  no findings; simulator DM tests use existing mock sinks. Archived fixture:
  **32 assertions** under normal and sanitizer builds.
- Updated 8-s cancellation+shadow and 15-case AO exporters pass ASan/UBSan and
  independent Python audits. No nonfinite/invalid nonzero proposal is accepted.
- Python: **21 controller tests plus four simulator PSD tests passed**. Four new
  cases check integer times/counters above 2^53, strict target/freshness/inhibition,
  common clipping, calibrated reflection/order and bad pupil geometry.
- The previous **76-run 60-s cancellation study** passed again. All **18 original
  cancellation binary/summary artifacts** remain byte-identical with shadow
  logging. **22 logs / 2,640,000 records** cover disabled gain, ordinary operation,
  isolated loss, invalid/stale input and reset/reattachment; evidence is synthetic.
- The existing **208 gap replays** pass using 6-s recordings; all **420 archived
  default/gap prediction/history/truth binaries** remain identical. Original numerical
  and input allocation benchmarks pass, with unchanged checksums and no C++/Eigen
  allocations. This does not measure the new offline file logger's RT suitability.
- Root application-test build registers/compiles the new shadow tests with scoped
  `-fno-fast-math`. No original assertion is weakened. Formatting, Python syntax,
  shell syntax and `git diff --check` pass. Main runtime sources/configurations,
  the original specification prefix and archived FITS inputs are preserved.

The integration report and app README contain complete normal replay commands.
Use the existing scientific Python environment; Astropy is optional and required
only for archived-FITS analysis. Local tooling is Apple Clang 21/Eigen 5.0.1,
libc++ 210106 and Anaconda Python. No dependencies were installed. Sanitizer
reproduction for the new standalone tools:

```bash
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-shadow-asan \
  CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' shadow-tools
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  ACCELFFCTRL_SHADOW_DIR=/tmp/accelFFCtrl-shadow-asan-final/logs \
  /tmp/accelFFCtrl-shadow-asan/accelFFCtrlCancellation_export /tmp/accelFFCtrl-shadow-asan-final/cancellation 8
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  /tmp/accelFFCtrl-shadow-asan/accelFFCtrlAO_export \
  /tmp/accelFFCtrl-shadow-asan-final/cancellation/ar_2_gain10.bin /tmp/accelFFCtrl-shadow-asan-final/ao
python3 apps/accelFFCtrl/tests/validate_shadow.py --shadow-directory /tmp/accelFFCtrl-shadow-asan-final/logs \
  --ao-directory /tmp/accelFFCtrl-shadow-asan-final/ao --output /tmp/accelFFCtrl-shadow-asan-final/report --no-plots
```

The existing combined sanitizer source list now includes `accelFFCtrlShadow_test.cpp`
(in the Linux runner too). The report gives the optional archived fixture commands;
its absence must not be silently replaced with synthetic “real” calibration.
Artifacts stay under `/tmp/accelFFCtrl-shadow-*`, including integer CSV logs,
provenance manifests, archived calibration JSON/examples, exact trajectory checks,
AO metrics and `ao/interactions.png`. Prior study plots/PSDs remain reproducible.

### Future observation-only gates and stop point

The report specifies future human authorization, matching Linux/compiler/library
builds, private native ADC validation, a coherent approved calibration bundle,
bounded capture/overhead qualification, accelerometer/WFS/AO/clock/source telemetry,
shadow logging and timestamp-aware PSD/transfer comparison. Required numerical
criteria are zero invalid/stale/disabled nonzero proposals, preserved horizons,
complete provenance/loss/reset accounting and unchanged AO output. Physical
coherence/phase/gain uncertainty, timing-error and total-headroom budgets must be
set by the owner; absent values fail physical qualification rather than receiving
invented thresholds. Current input depth one and default-disabled gap policy must
remain unchanged throughout observation.

No representative local live prediction/WFS/command timing record was available;
the earlier 602-s zero-loss 4-kHz aggregate does not qualify registration or phase.
Linux full-app/native validation remains outstanding. Current B/K is conditional
on archived DM-pupil geometry; the accelerometer/telescope/WFS chain, actual DM
response and AO transfer/headroom are blockers before real commands. A dedicated
correction channel and applied-zero/fault/crash-cleanup contract are also unassigned.
No live writer, output attachment or enabled gain was added to the application.

Changed files: new offline `accelFFCtrlShadow.hpp` and `INTEGRATION.md`; new app
`tests/accelFFCtrlShadow_test.cpp`, `accelFFCtrlAO_test.hpp`, `accelFFCtrlAO_export.cpp`,
`validate_shadow.py`, `test_validate_shadow.py`; app test `Makefile`,
`accelFFCtrlCancellation_export.cpp` (optional diagnostic logging only),
`validate_linux.sh` (test source list); root `tests/Makefile.one` and `tests/tests.list`;
both app READMEs and this plan. Existing runtime helpers/numerics, calibration
inputs, production configs and separate DDSPC worktree remain unchanged.

## Phase 12: recorded-telemetry calibration and timing qualification (2026-11-06)

### Scope, inventory and assumptions

Starting revision `89bd483e` on `parkertjohnson1/accelSim` was clean. This phase
adds workstation-only file analysis/replay; no instrument, application, stream,
configuration, deployment or git publication operation was performed. Runtime
controller/simulator sources and calibration inputs remain unchanged. The complete
report is [RECORDED_TELEMETRY.md](../../../apps/accelFFCtrl/offline/RECORDED_TELEMETRY.md).

The local repository/ignored scratch inventory contains nine FITS, twelve CSV and
two NPZ calibration/derived-operator files, but no XRIF, binary telemetry/log,
HDF5 or simultaneous measured ADC/WFS sequence. Prior `/tmp/accelFFCtrl-*` outputs
are synthetic. Earlier aggregate live 4/8-kHz drop/reset summaries cannot identify
WFS coherence, optical registration or physical phase. No remote host was searched.
This is not a claim about recordings elsewhere on the user's workstation.

Source review establishes the five-word XRIF timing record in `streamWriter`:
counter, acquisition sec/nsec, publication sec/nsec. The writer can replace absent
acquisition time with its own realtime clock and absent write time with acquisition
time. Consequently XRIF acquisition trust needs producer/recorder provenance.
`utils/xrif2fits` retains those integer fields in per-frame metadata exports; cube
images or filenames alone are insufficient. There is no locally installed Python
XRIF decoder or actual WFS image sequence here. Raw WFS images need the real
reference/reconstructor/indices/units before they can be treated as TT.

WFS frame-center, publication, `camwfs_sync` receipt, simulator phase reference
and controller query clocks are not interchangeable. The existing
`wooferTweeterRecon` write-time-minus-period and DM-acquisition-plus-offset
conventions are source facts, not measured offsets to copy. The new manifest
requires explicit ADC and WFS units/axes/AO state, trusted acquisition/exposure
center, independent clock offset/drift/anchor/uncertainty and common-domain identity.
Unknown clocks block phase inference; signals are never shifted to maximize fit.

### New isolated architecture

- `tests/qualify_telemetry.py` consumes exact-integer numeric ADC/WFS CSVs plus a
  provenance manifest. It rejects malformed/nonfinite/invalid time data, splits
  gaps/generations, checks cadence/loss, and uses antialiased rational resampling
  only for spectral scoring. Eight two-second nonoverlapping Hann blocks are
  required. PSD/CSD use `conj(X)*Y`; H and phase are masked below coherence 0.8 or
  the independent-block null threshold. Input cross-coherence/condition ratio
  expose correlated-axis identifiability limits. This is paired apparent H,
  not a coupled two-input or automatically open-loop physical calibration.
- `tests/accelFFCtrlTelemetry_replay.cpp` uses unchanged production
  `inputProcessor`, `harmonicEstimator` and Phase 11 `shadowCommand` with owned
  depth-one file snapshots. Each query sees only records received by that time.
  No resampled/WFS/future/truth measurement enters the estimator. Default gap
  tolerance is explicitly false; gain remains zero unless an offline manifest
  explicitly requests otherwise. No native or application interface is included.
- The first half fits passive complex H at 28/56 Hz; fixed H maps actual
  estimator target-time acceleration quadratures to WFS units in the held-out
  second half. This can absorb mechanical/AO transfer, and therefore does not
  validate the assumed lever arm or physical pointing frame. Raw pointing is
  logged separately. Query+horizon is preserved exactly; age is counted once.
- Scores use common held-out query coverage and actual targets, with WFS
  interpolation only between adjacent trusted centers. Missing WFS records are
  excluded. Uniform-ready chunks alone support prediction coherence; unavailable
  blocks are not compressed into a uniform record. WFS interpolation and finite
  real exposure are separate error sources. The 2-kHz interpolation amplitude
  error can approach 0.39% at 56 Hz.
- Shadow CSVs preserve acquisition/query/target/source, reset epoch, numerical
  validity/fault, raw pointing, absolute coefficients, clipping and input loss/
  reset counters. Integer origin registration permits exact epoch reconstruction.
  Invalid/stale/disabled commands are zero, with no command hold. Actual recorded
  queries can be supplied; assumed archive receipt/query schedules remain labelled
  replay assumptions, not live availability. Blocking output stays offline.
- An optional analytic-fixture counterfactual explicitly uses gain 0.5 and the
  existing synthetic `dmResponse`; it never enters a production command path.

### Synthetic quantitative results

All results below are synthetic, not measured MagAO-X calibration. The 60-s fixture
uses 8-kHz ADC/2-kHz WFS/query, 28/56-Hz analytic disturbances, memory 0.1 s,
lever 10 m, ADC 2048/0.0083799/1.02, ADC noise SD 0.002 m/s² and WFS noise SD
2e-9 rad. WFS gains 1.2/0.8, passive delay 1 ms and timestamp correction +700 us
are explicit known choices. These differ from the earlier stochastic accelSim
sensor-PSD/full-214-mode study and must not be presented as improved performance
on that study. Every generated file carries synthetic provenance.

| Band / axis | Coherence | H magnitude [rad/(m/s²)] | Phase [deg] | Referenced delay [ms] |
| --- | ---: | ---: | ---: | ---: |
| 28 / X | 0.9999991 | 3.877633e-6 | 169.87514 | 1.004450 |
| 28 / Y | 0.9999973 | 2.583473e-6 | 169.86993 | 1.004967 |
| 56 / X | 0.9999994 | 9.685313e-7 | 159.85226 | 0.999392 |
| 56 / Y | 0.9999984 | 6.453941e-7 | 159.83191 | 1.000402 |

Delay is recovered within 5 us. The reported stationary-Gaussian Welch interval
approximations exclude quantization/resampling/clock/model bias and are not formal
coverage guarantees for deterministic lines or physical delay. Referenced phase
delay remains ambiguous by integer carrier periods and is absent without a known
signed reference. Strong sensor cross-coherence cannot identify independent axes.

At 1.75-ms query horizon, 57,939 held-out scores per axis yield RMSE
4.05655e-9/3.45236e-9 rad, NRMSE **2.2218/3.5527%**, correlation
0.999801/0.999561 and bias -8.8144e-12/+1.7654e-12 rad. At 28 Hz amplitude errors
are -0.3032/-0.2881% and phase errors +0.01389/+0.04411 degree; at 56 Hz they are
-5.4577/-5.4465% and -0.21746/-0.20284 degree. Existing independent-bank leakage,
ADC quantization and WFS interpolation contribute; high coherence does not prove
gain accuracy. Whole-record readiness is **99.4992%**, zero drops and one initial
attachment reset. Sample age is 100 us, so sample-to-target horizon is **1.85 ms**.

| Horizon [ms] | X / Y NRMSE |
| --- | ---: |
| 0 | 2.2903% / 3.6701% |
| 0.5 | 2.2930% / 3.6740% |
| 1 | 2.2903% / 3.6702% |
| 1.75 | 2.2218% / 3.5527% |
| 3 | 2.2271% / 3.5812% |
| 5 | 2.1055% / 3.4069% |
| 10 | 1.9327% / 3.1401% |

The synthetic grid minimum at 10 ms is not a physical DM delay or an operating
recommendation. Harmonic-bank leakage rotates with target phase; the earlier
1.75-ms cancellation compensation includes a synthetic plant and command hold,
whereas this forecast score compares fixed target times. **No physical horizon
is qualified or recommended by Phase 12.**

Bounded ±2/±10-us acquisition-tag jitter keeps readiness near 99.5%, with no
drops/additional resets, and NRMSE 2.2219/3.5528% and 2.2224/3.5537%. Applying an
incorrect ±1-ms WFS comparison offset instead produces about **21–25% NRMSE**,
with 10.08/20.16 degrees of phase shift per millisecond at 28/56 Hz. These are
synthetic sensitivity tests, not live timestamp-jitter qualification.

The explicit counterfactual uses identity C, K=diag(1e6), gain 0.5, coefficient
bounds 1/1 and a chosen 1-ms/300-Hz response at 2 kHz with diagonal response
1.2e-6/0.8e-6 WFS rad/coefficient. Simulated variance reduction is **72.31/70.91%**;
RMS changes 1.82644e-7→9.61099e-8 rad and 9.72056e-8→5.24290e-8 rad. This is
neither measured suppression nor a substituted physical calibration. Default
gain-zero proposals remain exactly zero. Synthetic-response metadata and the
nonzero counterfactual are separate files.

### Verification, reproduction and remaining gates

Eleven new deterministic analysis/integrated-replay tests cover known spectra,
phase/delay, antialiasing/fractional registration, exact ns, malformed/unknown clocks,
low-coherence refusal through the complete pipeline, target-based horizon selection,
WFS holes, ADC units and default/clipped/drop/restart/invalid/stale shadow behavior.
The report gives workstation commands, the real-data manifest command and schema,
all assumptions and output CSV/plot names. Artifacts are under
`/tmp/accelFFCtrl-phase12-*`; no generated measurement is added as a real fixture.

Existing regression results: **33 C++ controller cases / 3,440,635 assertions**;
**67 combined simulator/controller ASan+UBSan cases / 3,677,233 assertions**;
**32 controller Python + four simulator PSD tests**. The new file replay passes
sanitizer integration cases and a 40-s full analysis. Existing allocation-guarded
benchmarks remain allocation-free with unchanged checksums. The previous **76-run
60-s cancellation study** passes again; **all 18 original binary/summary artifacts
are byte-identical**. No assertion is weakened. Formatting/syntax and preservation
checks apply to the final changed files. Native app/INDI/ImageStreamIO validation
has not been performed on macOS.

Physical boundaries remain separate: (1) passive accelerometer→WFS response,
(2) current WFS→DM calibration, (3) optical correction sign, and (4) actual DM
delay/bandwidth. Archived pupil B/K is conditional geometry only and supplies none
of the missing WFS/sensor registration/dynamics. Passive telemetry cannot measure
DM correction sign/response without separate physical response evidence. Real
paired telemetry with trusted acquisition/center/clock data and AO/POL/command
context, current optical calibration, measured response, bounded capture/overhead
and native Linux validation remain prerequisites. Observation-only defaults and
depth one are unchanged. No live writer, DM attachment, instrument access,
configuration change, staging, commit, push, merge or deployment occurred.

Changed files: new `tests/qualify_telemetry.py`, `test_qualify_telemetry.py`,
`accelFFCtrlTelemetry_replay.cpp`, `offline/RECORDED_TELEMETRY.md` under accelFFCtrl;
its test `Makefile`, `offline/INTEGRATION.md`, both application READMEs and this
plan. Production sources/configuration, existing models, archived calibration
inputs and the separate DDSPC worktree are untouched.
