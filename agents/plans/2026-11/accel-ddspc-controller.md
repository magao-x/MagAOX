# Task Description

## Problem Statement
Extend the existing MagAO-X data-driven subspace predictive control (DDSPC) implementation to incorporate synchronized accelerometer measurements directly into the mathematical prediction and control framework described by Haffert et al. (2021).

The original DDSPC framework predicts future WFS residuals using histories of WFS measurements and DM commands. This project introduces an additional accelerometer history vector and learns its contribution to future WFS residuals through recursive least-squares identification.

The extended model is

[
\mathbf y_f =
A\mathbf y_p+
B\mathbf u_p+
D\mathbf a_p+
C\mathbf u_f.
]

The goal is to learn the extended prediction matrix

[
\Theta=[A\ B\ D\ C]
]

and derive the optimal DM controller using the original DDSPC quadratic cost function, including the additional accelerometer contribution.

The implementation must preserve the existing DDSPC architecture, online learning procedure, and WFS/DM control functionality wherever practical.

The primary research objective is to determine whether synchronized accelerometer telemetry improves predictive wavefront correction compared with the original DDSPC formulation.

The initial validation will use the accelSim application to generate synchronized accelerometer measurements and physical DM disturbances. Although the known 28–29 Hz telescope vibration provides a useful initial test, the extended DDSPC framework must support multiple vibration frequencies and broadband accelerometer telemetry without requiring a predetermined vibration frequency or analytical acceleration-to-displacement conversion.

The implementation should ultimately support real on-sky accelerometer measurements and adaptive vibration correction on MagAO-X.
## Discussion
Haffert et al. (2021) developed a data-driven subspace predictive control (DDSPC) framework for adaptive optics that predicts future wavefront sensor (WFS) residuals using historical WFS measurements and deformable mirror (DM) commands.

The original predictive model is

[
\mathbf{y}_f^i =
A\mathbf{y}_p^i +
B\mathbf{u}_p^i +
C\mathbf{u}_f^i,
]

where (\mathbf{y}_p^i) and (\mathbf{u}_p^i) represent past WFS residuals and DM commands, respectively, and (\mathbf{u}_f^i) represents future DM commands. The matrices (A), (B), and (C) are identified from telemetry using recursive least squares (RLS).

The proposed extension introduces an accelerometer history vector,

[
\mathbf{a}p^i =
\begin{bmatrix}
\mathbf{a}i^T &
\mathbf{a}{i-1}^T &
\cdots &
\mathbf{a}{i-N+1}^T
\end{bmatrix}^T,
]

where (\mathbf{a}_i) contains the synchronized accelerometer measurements at time index (i), and (N) is the number of historical samples.

The extended predictive model becomes

[
\mathbf{y}_f^i =
A\mathbf{y}_p^i +
B\mathbf{u}_p^i +
D\mathbf{a}_p^i +
C\mathbf{u}_f^i.
]

The extended regression vector and prediction matrix are

[
\boldsymbol{\phi}^i =
\begin{bmatrix}
\mathbf{y}_p^i\
\mathbf{u}_p^i\
\mathbf{a}_p^i\
\mathbf{u}_f^i
\end{bmatrix},
\qquad
\Theta =
\begin{bmatrix}
A & B & D & C
\end{bmatrix}.
]

The RLS identification framework should remain mathematically unchanged, except for the additional accelerometer regressors and the corresponding increase in model dimensions.

The quadratic cost function is

[
J =
\mathbf{y}_f^T\mathbf{y}_f+
\lambda\mathbf{u}_f^T\mathbf{u}_f,
]

where (\lambda) regularizes the magnitude of the DM commands.

Minimizing this cost function gives the extended optimal controller,

[
\boxed{
\mathbf{u}_f^* =
-\left(C^TC+\lambda I\right)^{-1}
\begin{bmatrix}
C^TA & C^TB & C^TD
\end{bmatrix}
\begin{bmatrix}
\mathbf{y}_p\
\mathbf{u}_p\
\mathbf{a}_p
\end{bmatrix}.
}
]

The primary software objective is to implement this additional accelerometer contribution while retaining the original DDSPC identification, prediction, and optimal-control architecture.

## Existing MagAO-X Implementation
Existing MagAO-X Implementation

The parkertjohnson1/accel_ddspc branch contains the loPredCtrlAcc application, which should be investigated as the primary starting point.

Relevant applications and software include:

1. apps/loPredCtrlAcc/ — existing accelerometer-related predictive controller.
2. apps/loPredCtrl/ — existing predictive-control implementation.
3. Existing RLS and DDSPC utilities.
4. Existing WFS telemetry and DM command interfaces.
5. MagAO-X shared-memory monitoring and synchronization infrastructure.
6. accelSim — independently developed accelerometer and physical DM disturbance simulator.

The agent should determine how closely the existing implementation follows the DDSPC formulation, whether accelerometer measurements are already incorporated correctly, and which components can be reused.


## Training and Closed-Loop Operation
The controller must distinguish between the WFS residual, the underlying wavefront disturbance, and the influence of previously applied DM commands.

Training data must be indexed so that each regression vector corresponds to the appropriate future WFS measurements and actual applied DM commands.

The model should support online RLS updates while ensuring that training and prediction use causally available measurements.

The agent should investigate the existing implementation's treatment of closed-loop identification, command latency, regularization, numerical conditioning, and model updates.

## Runtime DDSPC Mode Selection

Implement an INDI property, ddspc_mode, that allows the operator to select between:

Standard DDSPC: Uses historical WFS residuals and DM commands without accelerometer measurements.

Accelerometer-assisted DDSPC: Uses historical WFS residuals, DM commands, and synchronized accelerometer measurements.

Maintain separate RLS models and prediction matrices for the two configurations so that standard DDSPC remains a valid baseline rather than approximating the baseline by setting the accelerometer matrix (D) to zero.

Both models should be capable of training and producing diagnostic predictions using the same telemetry, even when only one model is selected for control.

Provide INDI status properties indicating the active mode, training state, model readiness, accelerometer stream health, and prediction error for each model.

Mode switching must not unexpectedly enable DM actuation. Initially, require correction to be disabled before switching between active controllers. Reject selection of an unready model and document the behavior when accelerometer telemetry becomes stale or unavailable.

Ensure the original DDSPC mathematical formulation and controller behavior remain available for direct comparison with the accelerometer-assisted implementation.

## Scientific Validation
The primary comparison is between the original DDSPC controller and the accelerometer-augmented DDSPC controller.

Both implementations should be evaluated using identical disturbance realizations, WFS telemetry, DM response models, prediction horizons, and control parameters wherever possible.

The initial experiments can use the known 28–29 Hz simulated vibration to verify the accelerometer contribution. Subsequent tests should include multiple vibration frequencies, broadband disturbances, changing vibration conditions, and realistic sensor noise.

The goal is to determine whether accelerometer telemetry improves prediction accuracy, vibration suppression, and total residual wavefront variance beyond what is achievable using WFS and DM telemetry alone.

## Requirements
1. Review existing DDSPC implementation. Investigate loPredCtrlAcc, loPredCtrl, and related predictive-control utilities to identify the existing model, RLS algorithm, history vectors, control law, and DM output architecture.

2. Preserve the original DDSPC framework. Retain the existing WFS residual and DM command histories, online learning procedure, and optimal-control formulation wherever practical.

3. Extend the regression vector. Incorporate an accelerometer history vector (\mathbf{a}_p) alongside the existing WFS and DM histories.

4. Extend the prediction matrix. Implement the additional learned matrix (D) so that the predictor follows

[
\mathbf{y}_f =
A\mathbf{y}_p+B\mathbf{u}_p+D\mathbf{a}_p+C\mathbf{u}_f.
]

5. Maintain RLS identification. Extend the existing recursive least-squares identification to accommodate accelerometer regressors without unnecessarily replacing the learning algorithm.

6. Implement the extended optimal controller. Include the learned accelerometer contribution (C^TD) in the optimal DM command calculation, with correct matrix dimensions, signs, and regularization.

7. Support raw accelerometer telemetry. Correctly decode the approximately 8-kHz, two-channel UINT16 acc000 stream, including calibration, channel ordering, timestamps, and stream counters.

8. Synchronize accelerometer and WFS measurements. Align accelerometer measurements with WFS integrations and preserve the temporal relationships required by the DDSPC history vectors.

9. Support configurable accelerometer history. Allow the accelerometer history length and representation to be configured independently where appropriate, while maintaining consistent prediction and control timing.

10. Handle multirate measurements. Determine whether to use one synchronized accelerometer measurement per WFS frame or multiple measurements per integration, and implement a documented strategy.

11. Preserve correct training targets. Account for historical DM commands, closed-loop WFS residuals, and measurement-to-actuation delays during model identification.

12. Support online model adaptation. Allow the learned accelerometer contribution to adapt to changes in vibration amplitude, frequency, phase, and accelerometer-to-WFS coupling.

13. Maintain original DDSPC functionality. Provide a mode that disables the accelerometer contribution and reproduces the original WFS/DM-only controller for baseline comparisons.

14. Provide configurable controls. Expose appropriate INDI controls for accelerometer enable/disable, history length, RLS parameters, prediction horizon, regularization, controller gain, training state, and correction state.

15. Preserve real-time performance. Avoid unnecessary allocations, blocking operations, and excessive logging in the control loop. Evaluate the additional computational cost introduced by accelerometer regressors.

16. Maintain DM output compatibility. Reuse the existing calibrated DM command architecture and ensure that the DDSPC controller uses a separately approved additive channel when tested with accelSim.

17. Support offline validation. Allow the extended predictor and controller to be tested using recorded or simulated accelerometer, WFS, and DM telemetry without physical DM actuation.

18. Maintain compatibility with real sensors. Avoid direct dependencies on accelSim internals so that the same DDSPC implementation can eventually use real on-sky accelerometer measurements.

19. Document mathematical and software changes. Clearly identify modifications to the original DDSPC formulation, history construction, identification, and control law.


## Tests and Metrics
1. Original DDSPC regression: Verify that disabling accelerometer inputs reproduces the original DDSPC predictions and controller behavior within numerical tolerance.

2. Matrix dimensions and control law: Validate the extended regression vector, learned prediction matrices, and optimal-controller calculation against an independent numerical implementation.

3. RLS identification: Test convergence of the extended prediction matrix using synthetic systems with known WFS, DM, and accelerometer relationships.

4. Accelerometer contribution: Verify that the learned (D) matrix captures known correlations between accelerometer history and future WFS disturbances.

5. Accelerometer decoding: Confirm correct conversion of raw acc000 measurements, channel ordering, timestamp handling, and sample-counter behavior.

6. Synchronization accuracy: Measure accelerometer-to-WFS timing offsets, jitter, dropped samples, and history alignment. Verify that the regressor contains only measurements available at the corresponding prediction time.

7. Multirate integration: Compare one synchronized accelerometer measurement per WFS frame against multiple accelerometer measurements per integration, evaluating prediction accuracy and computational cost.

8. Prediction accuracy: Compare original and accelerometer-augmented DDSPC using future WFS residual prediction error, RMS error, variance explained, and frequency-dependent prediction performance.

9. Vibration suppression: Compare the residual tip/tilt PSD and integrated variance with accelerometer inputs disabled and enabled. Initially evaluate the 28–29 Hz vibration, then multiple frequencies and broadband disturbances.

10. Model adaptation: Introduce changes in vibration amplitude, frequency, phase, and coupling. Measure the time required for RLS to adapt and recover prediction performance.

11. Sensor noise robustness: Evaluate performance under different accelerometer noise levels, ADC quantization, missing measurements, and additional uncorrelated vibrations.

12. Closed-loop identification: Verify that the training algorithm correctly accounts for previously applied DM commands and does not incorrectly identify controller-induced changes as uncontrolled disturbances.

13. Control stability: Evaluate gain, regularization, latency, and model-update sensitivity. Check for unintended amplification of vibrations or growth in DM command amplitudes.

14. Real-time performance: Measure prediction latency, RLS update time, memory consumption, and timing jitter at the intended WFS control cadence.

15. End-to-end simulation: Validate the controller using accelSim accelerometer measurements and the corresponding physical DM disturbances, with WFS telemetry used to measure correction performance.

16. Scientific comparison: Quantify the incremental improvement from accelerometer telemetry using matched baseline and extended-controller experiments, including uncertainty across multiple disturbance realizations.

## Scope and Caveats
The primary objective is to extend the original DDSPC mathematical framework, not develop a separate frequency-specific feedforward controller.

The 28–29 Hz vibration is an initial validation target, not a restriction on the predictor or controller.

Begin with the existing loPredCtrlAcc implementation and preserve working functionality wherever possible.

Do not assume the current loPredCtrlAcc implementation correctly incorporates accelerometer telemetry without verifying its mathematical formulation.

Do not replace the existing RLS learning procedure unless a demonstrated numerical or architectural issue requires modification.

The accelerometer history must be constructed from causally available measurements and correctly aligned with WFS and DM histories.

The 8-kHz accelerometer stream and 2-kHz WFS stream may require explicit multirate handling; do not assume that reading the latest accelerometer sample preserves all measurements.

Accelerometer measurements may contain useful predictive information at multiple frequencies, but not all accelerometer motion necessarily corresponds to observable WFS tip/tilt.

The accelerometer contribution may be correlated with WFS history. Investigate model conditioning, regularization, and whether the additional regressors provide independent predictive information.

Adding accelerometer regressors increases the RLS model dimension and may substantially increase computational and memory requirements.

Do not assume that a lower open-loop prediction error guarantees improved closed-loop correction.

Do not train using future measurements or future DM commands that would not have been available at the prediction time.

Distinguish commanded DM values from actual applied commands where latency, saturation, or other instrument behavior may introduce differences.

Do not enable physical DM correction before completing numerical, offline, and non-actuating real-time validation.

Do not overwrite dm01disp10, which is reserved for the simulated disturbance injection.

The separate accelSim worktree is being developed concurrently. Do not modify it, merge branches, or introduce dependencies without explicit approval.

Do not commit or push implementation changes automatically.

# Instructions to Agent
<!-- Specific instructions for the agent.  Below is our standard, but you can modify it as needed. -->

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan

## Task Summary

Investigation and Phase 1 numerical implementation completed on 2026-11-05 on
`ASTR-LT-428.local`, branch `parkertjohnson1/accel_ddspc`, based on
`b9648ec958ebe3401dc20deaa240e84a60f25d41`. The user authorized Phase 1 only:
mathematical correctness and independent offline tests. No application was
started, instrument interrogated, physical command written, branch changed,
commit created, or change pushed. The separate `~/source/MagAOX` worktree was
not modified.

**Current status:** the numerical core now has an explicit opt-in
`HistoryMode::Causal` with corrected command indexing, origin-aligned
accelerometer histories, complete-record warm-up, and independently supplied
synthetic applied-command records. A separate reference helper implements the
literal-lambda optimal law without added feedback. `HistoryMode::Legacy` remains
the constructor default used by the unchanged application. Its original
operational law and known history defects remain available for compatibility.
The original `apps/loPredCtrl/` implementation is untouched.

Numerical tests reproduced both reported history defects before the corrections,
verified conventional and QR RLS against independent batch least squares, and
recovered the learned `D` block of a known synthetic two-mode plant. They also
exposed a QR structural-zero defect; an opt-in correction is enabled only for
the causal offline path. Test results and reproducible commands are recorded
below. These are numerical results, not live prediction or correction results.

Original source line references in the investigation below refer to the
pre-implementation revision above. The new Phase 1 contract and implementation
record name the current functions explicitly; formatting and documentation
changes shifted the old numerical-helper line numbers.

The objective is a general DDSPC predictor with learned accelerometer coupling
`D`, followed by the optimal controller specified above. Accelerometer decoding,
causal alignment, and identification must be independent of vibration frequency
and simulator internals. Preserve a WFS/DM-only comparison using the same timing,
plant, and numerical conventions.

**Existing functionality:** `apps/loPredCtrl/` implements a dense CPU modal
predictor, conventional and QR-based RLS, rolling histories, finite-horizon
quadratic control, exploration sequences, optional command accumulation, and a
FLOAT modal output through `dev::frameGrabber`. `apps/loPredCtrlAcc/` copies this
architecture and adds a second `dev::shmimMonitor`, accelerometer history columns,
normalization, and startup disable/fallback options. These are implemented code
paths, not evidence that the present branch operates correctly on the instrument.
The RLS algebra is reusable; the history and integration contracts need correction
and tests before treating either app as a verified reference.

**Critical findings in the unchanged app/legacy path:** the accelerometer path reinterprets UINT16 payloads as
FLOAT, loses intermediate samples, cannot detect a stream that becomes stale
after its first sample, and trains with accelerometer histories from the wrong
prediction origin. A shared command-indexing defect affects multiple controlled
modes. The quadratic controller contains accelerometer cross terms implicitly,
but its applied law also includes scaled regularization and an extra feedback
term. Training records calculated commands rather than confirmed applied
commands. There is no explicit non-actuating mode.

The mathematical reference is [Haffert et al. (2021), Section 2,
equations 3–15](https://arxiv.org/pdf/2103.07566). Its integral/incremental command
convention and receding-horizon selection need an explicit mapping to the app's
input/output conventions. The extension should retain RLS and add exogenous
accelerometer regressors; it does not require an acceleration-to-displacement
formula or a predetermined spectral line.

## Key Assumptions

1. **Prediction origin:** define decision index `i` after WFS residual `y_i` is
   available. Nominally, command `u_i`, issued at that decision, affects `y_(i+1)`.
   Real exposure and actuation delays must replace this nominal convention once
   measured. An origin's state is frozen at its decision deadline; identification
   waits until its future WFS targets and actual command sequence have occurred.
   Those later observations are training labels/actions, never future information
   supplied to the originating prediction.

2. **Dimensions:** let `m_y` be the number of controlled residuals, `m_u` the
   number of independently commanded modes, `F` the future horizon, `N_y` and
   `N_u` the past history lengths, and `q_a` the total accelerometer feature count.
   For one `c`-channel vector per WFS step, `q_a=c*N_a`; for `s` phase bins per
   step, `q_a=c*s*N_a`. Define

   ```text
   r = m_y*N_y + m_u*N_u + q_a
   p = r + m_u*F
   y_f: (m_y*F) x 1             phi: p x 1
   A:   (m_y*F) x (m_y*N_y)     B: (m_y*F) x (m_u*N_u)
   D:   (m_y*F) x q_a           C: (m_y*F) x (m_u*F)
   Theta = [A B D C]: (m_y*F) x p
   P: p x p                    K_full: (m_u*F) x r
   ```

   The current app uses `m_y=m_u=m`. Its effective legacy convention is
   `N_y=N`, `N_u=N-1`, despite `get_command_past()` returning `N` blocks: one is
   absorbed into `C`. Phase 1 preserves this effective convention without adding
   or removing a command block. Independent history lengths can follow with
   approved explicit layouts and model-version changes.

3. **Ordering:** histories currently use newest time first and mode/channel index
   within each time block. Retaining newest-first future sequences means the last
   command block is the next action. Public mathematical ordering is `[A B D C]`;
   an internal permutation is acceptable only with named, tested block offsets.

4. **Signal semantics require confirmation:** the app stores incoming `m_modeval`
   values directly and stores `new_delta` as command history. It does not
   difference WFS inputs or read applied DM values. When `is_integrating=true`,
   output accumulates those deltas; otherwise output is replaced. The future
   implementation must choose absolute commands or increments consistently and
   derive the corresponding cost, histories, and actuator mapping. Do not add WFS
   differencing or a second integrator based on notation alone.

5. **Rates and clocks are measured, not inferred from nominal ratios.** The target
   experiment is approximately 8000-Hz accelerometry and 2000-Hz WFS, but the
   reviewed simulator defaults to 6000 Hz. Timestamp meaning, exposure boundaries,
   transport delays, and clock offsets must be established for the deployed WFS
   modal stream. A shared semaphore reference does not establish simultaneous
   acquisition or physical application of a DM command.

6. **Simulator evidence is read-only and provisional.** `accelSim` is absent from
   this worktree. References below to `~/source/MagAOX/apps/accelSim/` refer to the
   separate worktree inspected at HEAD
   `4e58786c9a4ac47266cbea6d8847e3a2663aa798`; that worktree has concurrent work.
   Its files were not changed, imported, or made build dependencies. Reconfirm
   the interface before integration.

## Requirements

### Phase 1 implemented numerical contract

- `apps/loPredCtrlAcc/ar_controller.hpp` adds `HistoryMode` as an optional final
  constructor argument. Default construction and all existing app call sites
  select `Legacy`; no live mode selector or ImageStreamIO interface is changed.
  `Causal` must be selected explicitly by offline callers. The app lifecycle
  header and its physical output behavior are unchanged.

- `PredictiveController::append_sample(index, measurement, applied_command,
  acceleration)` accepts exactly one already synchronized, finite record per
  zero-based decision step. It rejects duplicate/skipped indices, invalid shapes,
  and nonfinite values before mutation. The synthetic plant contract is
  `y_(i+1)` responding to the input `u_i` held over `[i,i+1)`. This is an explicit
  discrete plant-input convention; it does not establish whether the live route
  expects absolute DM commands or increments. No WFS differencing is introduced.
  Real sensor timestamps, missing-sample reconstruction, and latency estimation
  remain outside Phase 1.

- Effective history lengths remain `N_y=N`, `N_u=N-1`, independent accelerometer
  length `N_a`, and equal WFS/command mode count `m`. Let `a=c*N_a`:
  `d=F*m`, `r=(2*N-1)*m+a`, `p=r+F*m`. The learner is `d x p`, its information
  inverse is `p x p`, and the full reference controller is `F*m x r`.
  Internal order remains `[C B D A]`, with widths
  `[F*m, (N-1)*m, a, N*m]`. `get_reference_prediction_matrix()` explicitly
  permutes the same learned columns to `[A B D C]`. No new identification
  algorithm replaces RLS.

- `get_command_future()` skips **time rows**, independently of mode count, in
  Causal mode. At latest record `t`, `get_training_example()` uses origin
  `i=t-F`, target `[y_t,...,y_(i+1)]`, future inputs
  `[u_(t-1),...,u_i]`, past inputs `[u_(i-1),...,u_(i-N+1)]`, WFS history
  `[y_i,...,y_(i-N+1)]`, and acceleration history
  `[a_i,...,a_(i-N_a+1)]`. The newly recorded `u_t` is excluded from these targets.
  Before `F+max(N,N_a)` complete records exist, no training occurs; disabled
  accelerometers do not extend that warm-up. Repeated `update_system()` calls
  on the same record do not perform repeated RLS updates.

- `get_prediction_state()` returns a value snapshot `[y_p;u_p;a_p]` at the
  current origin, excluding that record's `u_i`. Tests save this snapshot and
  prove that the state reconstructed from its later matured training example is
  identical, including ring wrap. Single-record insertion keeps these discrete
  synthetic histories aligned; it is not the timestamp/deadline-based acquisition
  adapter proposed for later phases.

- `DDSPC::reference_controller(A,B,D,C,lambda)` solves
  `(C^T*C+lambda*I)*K=-C^T*[A B D]` using Eigen LDLT. It validates dimensions,
  finite inputs/products, positive literal lambda, and successful finite solves.
  `get_reference_controller(lambda)` extracts the learned blocks and returns
  the full gain. For newest-first future ordering, the last `m` rows produce
  the nearest action. Neither function adds `gain*y_i`, exploration, model-scaled
  regularization, nor output accumulation. The operational `calculate_command()`
  and `update_controller()` retain those legacy behaviors and reject use in
  Causal mode; they cannot implicitly turn a proposal into an applied record.

- `QRDRecursiveLeastSquares(..., enforce_triangular)` adds a final optional flag
  defaulting to false. When true, `apply_givens()` sets each eliminated
  `A_(i,j)` entry exactly to zero after its rotation. Without this invariant,
  tiny remnants can cause spurious rotations in later updates near unexcited
  columns. The causal controller enables the invariant; legacy construction
  retains the old arithmetic. Conventional RLS and both original `loPredCtrl`
  RLS implementations are unchanged. Public legacy RLS field names are retained
  for source compatibility; added private state uses `m_` names.

- Corrected standard and augmented models are separate instances with separate
  learner dimensions/state. Disabling channels or setting history length zero
  removes `D` columns rather than approximating the baseline by zeroing them.
  A supplementary zero-column experiment verifies numerical equivalence within
  FLOAT tolerance, not a replacement for the dimension-reduced baseline. The
  planned runtime `ddspc_mode` property and simultaneous diagnostic model
  management are deferred to live integration, as requested for Phase 1.

- Causal save/load is explicitly rejected before file I/O: the existing legacy
  artifact format lacks sufficient layout/timing/scaling metadata. Dimensional
  and RLS-prior validation is added only to the Causal constructor. Existing
  operational gain, regularization, accumulation, serialization, and startup
  behavior are not silently changed.

### Existing histories and prediction layout

- `loPredCtrlAcc/ar_controller.cpp:9–70`, `PredictiveController()` allocates
  power-of-two rings and both RLS variants. With `a=c*N_a`, its current sizes are
  `d=F*m`, `p=(F-1+2*N)*m+a`, and `r=(2*N-1)*m+a`. The prediction matrix is
  `d x p`, the inverse information matrix is `p x p`, and the stored controller
  is only the next `m x r` command block. This is a coupled modal model, unlike
  the per-mode GPU models in `hoPredCtrl/distributed_ar_controller.cu`.

- `get_measurement_future()` and `get_measurement_past()`
  (`loPredCtrlAcc/ar_controller.cpp:127–155`) return the newest `F` WFS frames
  and then `N` frames shifted back by `F`. After `calculate_command()` at frame
  `t`, these are `[y_t,...,y_(t-F+1)]` and
  `[y_(t-F),...,y_(t-F-N+1)]`. The nominal training origin is therefore `t-F`.

- `update_system()` (`loPredCtrlAcc/ar_controller.cpp:259–274`) concatenates
  `[future_cmd_(F-1); past_cmd_N; accel_history; past_WFS_N]`, not `[y_p;u_p;a_p;u_f]`.
  `update_controller():278–294` treats the first `F*m` columns as `C`. If command
  indexing is corrected, this is a permutation `[C B D A]` with `N-1` past
  command blocks; the first of the `N` blocks returned by `get_command_past()`
  supplies the last future-command block. The unusual layout is not by itself
  proof of an incorrect control law, but must be made explicit and tested.

- **Inherited indexing defect:** `get_command_future(1)` uses
  `offset=skip_cmds*_num_modes` as a number of ring rows, even though each row
  already contains all modes (`loPredCtrlAcc/ar_controller.cpp:157–188`;
  `loPredCtrl/ar_controller.cpp:137–170`). After storing `u_t`, for `m=2,F=3`
  it returns `[u_(t-2),u_(t-3)]`; `get_command_past()` starts again at `u_(t-3)`.
  Thus `u_(t-1)` is missing and `u_(t-3)` is duplicated. Scalar tests conceal
  this defect. The row skip must be independent of mode count.

- **Accelerometer timing defect:** `get_accelerometer_past():197–213` always
  uses the latest accelerometer head. Both training and current prediction call
  it without an origin offset. The WFS callback pushes a current sample before
  predicting and learning (`loPredCtrlAcc.hpp:860–922`), so training at `t`
  includes `a_t` while its past WFS origin is `t-F`. This gives the learner
  information from its target interval that is unavailable at the intended
  prediction origin. Extra columns can represent a learned `D` algebraically,
  but they do not currently implement the required causal `D*a_p^i` model.

- **Required change:** construct a single timestamped decision record containing
  the exact WFS, applied-command, and accelerometer state available at origin
  `i`. Mature that record only when the whole target interval and command
  sequence are known. Use the same state builder for inference and training;
  reject incomplete/gapped records instead of treating zero-filled startup rings
  as measured history. Declare the WFS/DM delay and any commands already in
  flight explicitly. Do not optimize commands whose physical application is
  already fixed by the pipeline.

### RLS identification, cadence, and computational cost

- `loPredCtrlAcc/recursive_least_squares.cpp:9–34,69–81` implements conventional
  multi-output RLS with shared feature covariance. Its algebra is, for column
  regressor `phi` and target `z`:

  ```text
  e       = z - Theta*phi
  k       = P*phi / (gamma + phi^T*P*phi)
  Theta'  = Theta + e*k^T
  P'      = (P - k*phi^T*P) / gamma
  ```

  The row-gain implementation is equivalent in exact arithmetic when `P` is
  symmetric. The variable named `initial_covariance` actually initializes
  `P=I/initial_covariance`; define it as prior information strength `rho`, not
  an initial `P` amplitude. Preserve the RLS algorithm while testing numerical
  equivalence, symmetry, positive definiteness, and valid `0<gamma<=1`, `rho>0`.
  At the initial review both apps' conventional and QR RLS sources were duplicate
  copies. Conventional RLS remains unchanged; the accelerometer app's QR helper
  now has the documented opt-in structural-zero correction.

- `QRDRecursiveLeastSquares::update()`
  (`loPredCtrlAcc/qrd_rls.cpp:147–179`) maintains an information factor:
  `R <- qr([sqrt(gamma)*R; phi^T])`, solves `R^T*z=phi`, `R*g=z`, and adds
  `e*g^T` to the weights. Its prior `R=sqrt(rho)*I` matches conventional RLS.
  The equations are now verified independently for the Phase 1 test cases. The
  opt-in structural-zero correction above is required for the zero-column
  regression; legacy construction remains unchanged.
  `apply_givens():79–110` rotates adjacent rows through nested loops; it can
  require cubic work for dense regressors. Do not assume this implementation has
  the quadratic cost of an optimized appended-row QR update.

- The WFS callback calls `calculate_command()` before `update_system()` and
  `update_controller()` (`loPredCtrlAcc.hpp:884–922`), once per callback when
  `learn` is true. The newly learned controller is used on the next frame.
  Both RLS and the controller solve therefore target the WFS cadence, not 8 kHz.
  However, histories advance only when `predict` is true, while `learn` can be
  enabled independently. Learning with prediction off uses stale histories.
  The app also lacks the warm-up gate present in `testPredCtrl.cpp:174`.
  Measurement/actual-command collection must continue independently of whether
  prediction, training, or correction is enabled.

- All DDSPC matrices use FLOAT (`loPredCtrlAcc/utils.hpp:11–12`). Conventional
  RLS costs `O(p^2+d*p)` per update and `O(p^2+d*p)` storage. Current controller
  recomputation forms the entire `Theta^T*Theta`, costing `O(d*p^2)` and a
  temporary `p x p` matrix, plus an inverse of size `F*m`. Both learners are
  allocated even when only one is used. Histories and prediction matrices are
  repeatedly allocated/copied; `get_prediction_matrix()` returns by value
  (`ar_controller.hpp:58–64`).

- Example, **not a benchmark**: for `m=2,N=5,F=3`, no accelerometers gives
  `p=24,d=6,r=18`. Two channels with 20 WFS-rate history steps gives
  `p=64,r=58`: the FLOAT `P` alone grows from 2304 to 16384 bytes, a 7.11-fold
  quadratic increase. Four phase bins for each of those same 20 steps gives
  `p=184,r=178`, a 58.78-fold increase relative to the baseline. QR workspaces,
  duplicate learners, and controller temporaries add substantially more memory.
  At 2 kHz there is about 500 microseconds between WFS frames, of which this app
  receives only a portion. Measure tail latency and skipped frames, not just the
  mean printed every 4000 callbacks (`loPredCtrlAcc.hpp:925–930`).

- After correctness is established, compute only `C^T*C` and
  `C^T*[A B D]`, preallocate workspaces, and publish complete controller snapshots.
  Consider a slower identification/solve cadence only if measurements require
  it; retain WFS-rate prediction. Express forgetting in physical time:
  an update spacing `Delta_t` implies `gamma=exp(-Delta_t/tau)`. Simply retaining
  the same per-update gamma after decimating updates changes adaptation speed.

### Optimal law and command semantics

- `update_controller()` (`loPredCtrlAcc/ar_controller.cpp:278–294`, original
  `loPredCtrl/ar_controller.cpp:223–239`) calculates

  ```text
  H11 = C^T*C
  H12 = C^T*[B D A]              # under the intended internal permutation
  lambda_effective = maxCoeff(H11) * configured_regularization
  K_full = -(H11 + lambda_effective*I)^-1 * H12
  K_next = last m rows of K_full
  ```

  Thus `C^T*D` is already included implicitly if the feature partition is
  correct. Reordering columns alone is not the missing accelerometer extension.
  The time mismatch and command-indexing defect invalidate that interpretation
  for current training data. The regularization is model-scaled rather than
  the literal configured `lambda` in the requested cost. A zero/unidentified
  `C` gives zero regularization even with a positive configured value, followed
  by inversion of a singular matrix.

- `calculate_command():297–328` actually produces
  `new_delta = K_next*[u_p;a_p;y_p] + gain*y_current + exploration_noise`.
  The integrator matrix picks the current WFS value; this extra term is not in
  the stated optimum. `loPredCtrlAcc.hpp:897–907` optionally accumulates the
  result into `full_command`. These two different operations must not be
  conflated. The command clipping block is disabled by `if(false)`.

- **Required change:** expose explicit blocks and compute
  `(C^T*C+lambda*I)*K_full = -C^T*[A B D]` with an Eigen factorization rather
  than explicit inversion. Apply the nearest future block, verify sign/units,
  and make any feedback blend or exploration a separately documented operating
  mode. A literal optimal-law test uses no extra feedback/noise. Preserve scaled
  regularization only as an explicit compatibility option if approved. Require
  finite, adequately identified models and a well-conditioned positive
  regularized solve before accepting a controller. `C=0` with positive lambda
  should produce zero optimal action, not NaNs.

- The current app consumes a FLOAT modal vector and publishes a FLOAT vector of
  the same width (`loPredCtrlAcc.hpp:597–633,771–784,937–992`). It does not itself
  open a 50x50 DM stream, load a mode cube, or verify a downstream calibration.
  Modes beyond `m_num_modes` are copied from the input. Exact deployed modal
  stream names and the downstream modal-to-DM process are not specified in the
  reviewed app/config files; this review cannot establish the active route.
  `hoPredCtrl` has interaction/mapping-matrix and physical output examples
  (`hoPredCtrl.hpp:682–712,968–1000`), but its direct writer is not a calibrated
  drop-in replacement for this app's modal output contract.

- Training currently stores the locally calculated `new_delta` before the
  frameGrabber thread copies `full_command`. There is no publication
  acknowledgement, applied-command input, saturation feedback, or counter-based
  WFS/DM matching. The mutable `full_command`, `m_updated`, and timestamp are
  shared without a complete frame handoff (`loPredCtrlAcc.hpp:897–922,956–992`).
  A calculated command can differ from a held, dropped, accumulated, limited,
  combined, or physically applied command. Record each stage and use a defined
  accepted/applied modal command sequence for learning. If physical readback is
  unavailable, label the publication-plus-delay model as an assumption and test
  its limits rather than calling it measured DM response.

- `dev::dm` describes desaturated shape/delta/difference outputs
  (`libMagAOX/app/dev/dm.hpp:120–122,2877–2900`) that may help characterize
  applied commands. Whether they are enabled and preserve the required modal
  information must be established by the operator. Commands from another AO
  loop and the simulator's disturbance channel must be distinguished from the
  DDSPC-controlled input. The simulator injection is an exogenous disturbance,
  not a future action the optimizer may choose.

### Accelerometer acquisition and synchronization

- **Current receiver:** `loPredCtrlAcc.hpp:636–650` checks only that the image
  contains enough values. `processImage(accelShmimT):738–767` maps `curr_src`
  as `realT*` (FLOAT), reads two values, applies running z-score normalization,
  and saves only the latest sample. Two UINT16 channels occupy four bytes,
  whereas two FLOAT reads require eight; this is invalid decoding and may read
  beyond the frame. Normalization cannot correct it.

- **Reviewed producer contract:** in the separate worktree,
  `accelSim/accelSim.hpp:616–622,941–949` publishes a `2 x 1` UINT16 frame,
  X followed by Y. `accelSimModel.hpp:38–49,367–377` specifies a 12-bit payload
  `[0,4095]` in the UINT16 container and the encoder
  `raw=clip(round(zero+acceleration*sensitivity/voltsPerBit))`. The inverse is

  ```text
  acceleration[channel] = (double(raw[channel]) - zero[channel])
                          * voltsPerBit[channel] / sensitivity[channel]
  ```

  Reviewed defaults are `zero=2048`, `voltsPerBit=0.0083799`, and
  `sensitivity=1.02`, yielding approximately `0.00821559 m/s^2` per count.
  Subtract in floating point to avoid unsigned underflow. Validate datatype,
  geometry, channel ordering, endpoints/clipping, and calibration version;
  allow per-channel calibration for real sensors. Reject an unexpected datatype
  or use an explicitly configured conversion, never reinterpret FLOAT and UINT16.
  The local hardware interface also publishes UINT16
  (`apps/mcp3208Ctrl/mcp3208Ctrl.hpp:1091–1097,1211–1213`), but sequential hardware
  channel readout need not share the simulator's simultaneous X/Y timing.

- `accelSim.hpp:832–929` schedules using `CLOCK_MONOTONIC` and stamps actual
  acquisition using `CLOCK_REALTIME` before model advancement. The separate
  worktree's `dev::frameGrabber` publishes that as `atime`, records publication
  `writetime`, advances `cnt0`, and selects `cnt1`; the corresponding mechanism
  in this worktree is `libMagAOX/app/dev/frameGrabber.hpp:1008–1041`. A cube
  additionally has `atimearray`, `writetimearray`, and `cntarray` per slot.
  `cnt0` counts published frames, not every nominal simulator tick. Scheduling
  skips can advance the model without publishing intervening samples
  (`accelSimModel.hpp:467–475`); use timestamp gaps and public skipped/discontinuity
  diagnostics as well as counters. The two-value payload has no embedded tick
  counter or timestamp.

- The simulator's approximately 2-kHz semaphore input is a **phase reference**
  for an independently paced accelerometer clock, with latency parameters,
  phase slew, and holdover, not a trigger that emits exactly four samples per
  WFS frame (`accelSim.hpp:631–645,695–829`;
  `accelSimModel.hpp:526–565`; `config/example.conf:38–55`). Its example config
  uses `fps=6000` and `circBuffLength=1`; 8000 Hz must be selected deliberately.
  Do not copy simulator timing constants into the controller as measured WFS
  exposure boundaries.

- `dev::shmimMonitor` supplies the current `cnt1` slot on each semaphore wake
  (`libMagAOX/app/dev/shmimMonitor.hpp:638–717,737–779`). It does not drain every
  historical slot, deduplicate by `cnt0`, verify a stable snapshot, or supply a
  timestamped immutable record. A queue of semaphore posts is not a queue of
  distinct payloads. A fast dedicated callback can collect more samples than a
  WFS callback, but a depth-one producer cannot recover overwritten frames.

- **Required transport:** retain the existing monitor/base-class lifecycle and
  build a bounded, app-local acquisition adapter that copies data and matching
  metadata into a preallocated ring. Deduplicate by stream generation/counter,
  use the installed ImageStreamIO consistency protocol with bounded validation
  of counters/write state, and mark races or overwrites invalid. For producer
  cubes, support ordered recovery of retained slots using per-slot counters and
  timestamps, rather than assuming the monitor delivers them all. Size producer
  retention and the local ring for measured scheduling delays; adding a local
  queue cannot recover samples already overwritten at the producer. A producer
  `circBuffLength` configuration change requires coordination, without a code
  dependency on the simulator worktree.

- Reuse timestamped-ring and offset concepts from
  `apps/wooferTweeterRecon/wooferTweeterRecon.hpp:152–184,695–723,858–993` and
  recorder metadata conventions from `apps/streamWriter/streamWriter.hpp:1490–1526,1602–1659`.
  Its interpolator can use samples on both sides of a timestamp, so it cannot
  simply be used as a causal online interpolator. `streamWriter` handles
  repeats/gaps but does not guarantee capture of every original sample.
  `apps/streamCircBuff/streamCircBuff.hpp:279–295,310–361` converts to FLOAT,
  retains a source pointer, and replaces acquisition time with callback time;
  do not assume an existing `acc000-cbuff` preserves raw counts or original
  acquisition metadata. If multiple consumers need a general timestamped
  multirate utility, propose that small shared helper after its contract is
  tested, rather than broadly rewriting `shmimMonitor` now.

- **Required alignment:** associate each WFS modal record with exposure start/end
  (or a documented center plus integration duration), original frame identity,
  source `atime`, publication time, and local receive/deadline time. The present
  WFS callback overwrites the output acquisition timestamp with local arrival
  (`loPredCtrlAcc.hpp:774–784`) and ignores input counters and exposure metadata.
  Upstream processing may likewise restamp frames, so modal `atime` cannot be
  assumed to mean exposure time. Keep acquisition and availability times
  separate. Translate clock domains using an established offset; use monotonic
  time for age/deadline checks and integer timespec/nanosecond differences for
  cross-stream alignment. Detect clock jumps and invalidate affected windows.

- Build features only from samples acquired within the declared causal window
  **and received by that origin's decision deadline**. Include measured sensor
  latency, WFS exposure weighting, transport/reconstruction latency, filter
  group delay, and command application delay. Online inference must not wait
  indefinitely for accelerometers or interpolate using a sample received later.
  Offline training must reproduce the same cutoff, not use hindsight to repair
  origin features. Log sample count, timestamp span, maximum age, phase error,
  gaps, and overflow per WFS record.

- **Recommended first representation:** one two-channel, causally filtered
  vector per WFS interval, using the retained raw samples, not one arbitrary
  latest raw sample. A time-weighted integration is a useful exposure-matching
  feature; broadband decimation also needs a specified anti-alias filter and
  accounted group delay. A four-sample boxcar alone is not adequate general
  anti-alias protection near/above the WFS Nyquist frequency. This representation
  gives the smallest initial `D` and a clean baseline experiment.

- **Required comparison:** retain the 8-kHz records so offline experiments can
  compare that representation with approximately four causally available
  phase-bin vectors per WFS interval, or a fixed-duration raw history with a
  tested resampling rule. Multirate features retain subframe phase information
  and may help broadband/latency prediction but increase covariance cost and
  collinearity. Define bins by timestamps and handle noninteger rate ratios;
  never assume exactly four arrivals. Compare equal physical history durations:
  20 WFS-rate steps span about 10 ms, while 20 raw samples span about 2.5 ms.
  Choose the production representation using held-out prediction, closed-loop
  simulation, and runtime measurements, not the 28–29-Hz case alone.

- **Missing-data policy:** the current `m_haveAccelSample` remains true after
  consumption, so its missing-frame counter cannot detect a stopped producer
  after one received sample (`loPredCtrlAcc.hpp:763–766,865–881`). Replace this
  with advancing sequence and age/coverage checks. Reject incomplete training
  windows, reset history across discontinuities, and report validity. Do not
  silently repeat stale samples, fill gaps with zeros, compress missing WFS
  time steps, or insert interpolated data without a declared tested policy.

### Baseline, operating states, and configuration

- Startup `accel_enabled=false` sets channels/history to zero and omits the
  accelerometer monitor (`loPredCtrlAcc.hpp:368–414`), reducing the core to the
  legacy feature dimensions. This is the useful existing baseline entry point.
  The library constructor instead defaults to two channels and 20 samples
  (`ar_controller.hpp:54`); the copied `testPredCtrl.cpp` never supplies any
  accelerometer samples, so it is not a valid accelerometer regression test.

- Distinguish **legacy compatibility** from the **corrected experimental
  baseline**. Compare legacy behavior on supported scalar traces first; document
  deviations caused by fixes. Compare augmented and corrected baseline modes
  using the same corrected WFS/DM indexing, regularization, command semantics,
  and training policy. `q_a=0` must remove the columns and corresponding
  covariance state; merely zeroing inputs in an augmented model is insufficient
  for a clean baseline.

- Current fallback can delete/recreate the learner with zero accelerometer
  dimensions, losing its learned WFS/DM model, or disable the path without
  rebuilding its augmented dimensions (`loPredCtrlAcc.hpp:711–735`, called with
  different arguments at `457,524,542,878`). It also changes monitor lifecycle
  and shared state across threads. A dropout transition must occur through the
  control thread at a defined boundary. The runtime mode-selection requirement
  above calls for independent standard and augmented models; measure their cost
  and choose adaptation/solve cadence accordingly. On a fault, inhibit correction
  or deliberately select an already ready baseline under the agreed policy.
  Never silently deploy an untrained rebuilt model.

- Separate collection, RLS training, controller calculation, and physical
  correction states. In shadow mode, calculate hypothetical commands only for
  diagnostics; train with the actual command history from the operating loop,
  never those hypothetical commands. Learning `C` requires sufficiently
  informative command data. Zero or fixed commands do not identify control
  response, regardless of how well `A` or `D` predicts disturbances; use recorded
  excited data or an operator-approved later experiment.

- The current `predict=false` selects the copy/gain path, and `integrate=false`
  selects command replacement; neither prevents publication. Every WFS callback
  posts the frameGrabber semaphore. `learn_std` enables prediction, training,
  and exploration (`loPredCtrlAcc.hpp:1129–1162`). Non-actuating validation must
  use an explicit state that cannot open/write a physical correction endpoint;
  a diagnostic output must have an unconnected name verified by the operator.

- Existing configuration covers mode/history/horizon, gamma, covariance, gain,
  regularization, accelerometer enable/history/normalization, and stream names
  (`loPredCtrlAcc.hpp:337–365`). Existing INDI controls primarily cover learning,
  prediction, integration, exploration, save/load, and reset (`446–507`);
  accelerometer controls and many numerical parameters are config-only. Add
  validated configuration and appropriate INDI state/controls for calibration,
  representation/history, alignment offsets, data validity, model readiness,
  adaptation, regularization, gain, and a separate correction enable. Apply
  dimensional changes with correction inhibited and full model/history reset;
  do not resize active RLS state in a callback.

- `normalizeAccelSample()` uses an ever-changing running mean/variance
  (`loPredCtrlAcc.hpp:676–708`), so historical and new columns can have different
  feature scales. Use calibrated units first; if conditioning needs
  normalization, estimate it on training data and freeze/version it, or transform
  the model/covariance consistently when it changes. Do not normalize on the
  held-out record. Current normalization updates occur outside the mutex used
  for reset, and enable/dimension/controller state is accessed by multiple
  threads; resolve ownership before live use.

- Saved models must include feature ordering, history lengths, calibration,
  scaling, sample representation, delays, command convention, and learner type.
  Current accelerometer state is not serialized in
  `ar_controller.cpp:331–378`; dimensions/workspaces are not fully reconstructed
  on load. QR `save_state/load_state` are no-ops apart from printing
  (`qrd_rls.cpp:182–226`), and `set_regularization()` does not update the scalar
  later saved in metadata (`ar_controller.cpp:112–125,345`). Version and validate
  artifacts; reject incompatible models with correction disabled.

## Questions and Points of Clarification

Phase 1 was explicitly authorized and completed without resolving live instrument
semantics. These decisions remain before operational integration;
recommendations are not assumptions about the deployed instrument.

1. **Correction scope and command convention:** which WFS/modal input and output
   endpoints will be used, and does this replace the current low-order controller
   or add a separate correction alongside it? Confirm residual units, sign,
   mode order, incremental versus absolute output, downstream accumulation, and
   which accepted/applied command telemetry is available. Recommend preserving
   the established modal/calibration route once it is documented, with an
   explicit increment convention if the downstream route integrates.

2. **Baseline and mathematical integration:** the inherited corrections and
   literal reference law now exist as an opt-in offline path in `loPredCtrlAcc`,
   while the app and `loPredCtrl` remain historical/operational references.
   Approve how the corrected path should eventually replace/select alongside
   the legacy path in the app. Recommend an explicit
   corrected baseline with `N_y=N,N_u=N-1`, literal positive lambda, and no
   undocumented added feedback in optimal-law mode. Decide whether legacy scaled
   regularization/feedback remains as a named compatibility mode. Any later
   common-core extraction or fixes to `loPredCtrl` require an agreed scope.

3. **Acquisition/timing evidence:** approve collecting raw producer metadata and,
   if necessary, configuring a producer ring with sufficient retention? The
   operator must supply actual `acc000` geometry/datatype/depth/rate/calibration,
   WFS exposure timestamp semantics and frame correspondence, sensor clock
   relation/latency, and the command-to-WFS response delay. Recommend metadata
   and recorded traces before choosing offsets; no agent should query hardware.

4. **Multirate starting point:** approve one causally filtered two-channel vector
   per WFS step first, retaining raw data for the phase-bin comparison? Approve
   independently configurable accelerometer history measured in physical time
   and a later representation decision based on broadband offline results.
   The filter response/group delay and acceptable coverage/age thresholds need
   agreement after timing measurements.

5. **Dropout and adaptation:** the requested runtime design retains separate
   standard and augmented models. Decide whether a sensor fault inhibits
   correction or deliberately selects an already ready baseline, and how each
   model adapts while unselected. Recommend inhibition until timing and model
   validity are proven; measure the cost of training both models.
   Agree on model-readiness criteria, adaptation time, solve cadence, and maximum
   acceptable missed deadlines before commissioning.

6. **Later simulator/DM experiment:** identify the approved correction additive
   channel, calibration/mapping, ownership, amplitude limits, and other active
   AO loops. `dm01disp10` remains reserved for simulator disturbance injection;
   do not select another channel by guess. Operator-run disturbance excitation
   and correction are separate later approvals. Also confirm access to recorded
   WFS/DM/accelerometer data and the supported Linux test environment; static
   source review does not establish deployment performance.

## Tests

### Phase 1 executed tests

`apps/loPredCtrlAcc/tests/Makefile.offline` builds only the numerical sources,
the repository Catch2 entrypoint, and the two new `ddspc_*_test.cpp` suites.
No app startup or stream/hardware libraries are linked. On the inspected macOS
host, `otool -L` reports only libc++ and libSystem. Dependencies are the installed
Eigen and mxlib headers plus the repository Catch2 header. Builds use C++20,
`EIGEN_DONT_PARALLELIZE`, and `-fno-fast-math`; no downloads were required.

From the repository root:

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j3 check
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j3 check \
  BUILD_DIR=build/offline-sanitize \
  CXXFLAGS='-std=c++20 -O1 -g -Wall -Wextra -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer'
```

Both optimized and AddressSanitizer/UndefinedBehaviorSanitizer builds passed
**46,147 assertions in 12 test cases**, with no sanitizer findings, using Apple
Clang 21.0.0 and Eigen 5.0.1. `clang-format` 23.1.3 was applied to all six touched
C++ files, and its dry-run check with `--Werror` passes. `git diff --check` passes.

- Before implementation, the two direct history regression assertions both
  failed on the original numerical code (2 failed cases; 1 passed and 2 failed
  assertions). Final characterization tests reproduce the old selections through
  `Legacy` while asserting the corrected selections through `Causal`.
- `ddspc_history_test.cpp` covers the origin-5/frame-7 leakage example, two-mode
  missing/duplicated command blocks, all 54 combinations of mode count
  `{1,2,3}`, horizon `{1,2,3}`, WFS history `{1,3}`, and accelerometer history
  `{0,2,5}`, with 130 frames each to exercise wrap. It checks frozen-state
  equivalence, complete-history warm-up, zero-length command histories,
  invalid/duplicate/skipped records, finite values, resets, rejected implicit
  proposal recording, and one RLS update per matured record.
- `ddspc_math_test.cpp` checks rectangular/mixed prediction blocks, the full
  control sequence against an independently assembled augmented least-squares
  QR solve, optimal-cost stationarity and perturbations, positive lambda,
  zero/rank-deficient `C`, overflow/nonfinite rejection, and nearest-action
  block selection. A scalar example explicitly distinguishes the literal
  penalty from scaled regularization and the extra legacy feedback term.
- Every conventional and corrected-QR RLS update for a deterministic noisy
  `4 x 12` regression is compared with a newly assembled double-precision
  weighted batch fit, including the decaying `gamma^k*rho` prior. SVD provides
  the independent inverse-information oracle. Tests compare weights, gains,
  `P`, and `R^T*R`, symmetry/positive definiteness, image/matrix overloads,
  reset, and zero inputs. Gamma values are 1 and 0.97. Errors below use
  `||actual-reference||/max(1,||reference||)`, rather than dividing by a
  potentially near-zero fit.

| Gamma | Max weights error | Max `P` error | Max QR information error | Max gain error |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.94e-7 | 2.22e-7 | 4.93e-7 | 2.30e-7 |
| 0.97 | 3.20e-7 | 1.18e-6 | 9.49e-7 | 5.81e-7 |

- The zero-column test uses a separate 320-observation, 16-feature batch oracle.
  Before enforcing triangular zeros, the legacy QR information error was
  **0.101032** on that trace; the zero-column baseline comparison also exposed
  a weights discrepancy of **0.00843862** on its separate trace. After the
  opt-in correction, the independent zero-column weights error is **1.68e-7**
  and information error **2.55e-6**, with an exactly zero lower triangle. Tests
  retain a default/explicit-false legacy QR comparison and verify its dense
  short-trace agreement; no claim is made that legacy QR is numerically sound
  for all histories.
- The causal identification experiment uses a known two-mode AR plant with
  cross-channel `A,B,D,C`, deterministic independent excitation, and synthetic
  held/clipped commands. It trains through frame 1799, then freezes both
  independent standard and augmented models and evaluates 600 further
  conditional one-step predictions. The simulator and its hidden state are
  not used. Each matured target is also checked against the known equation
  using the actual synthetic applied-command history, not its proposal.

| Learner | All-block coefficient error | `D` error | Augmented held-out RMSE | Standard held-out RMSE |
| --- | ---: | ---: | ---: | ---: |
| Conventional | 5.71e-5 | 2.22e-5 | 1.729e-5 | 0.276412 |
| Corrected QR | 5.72e-5 | 2.21e-5 | 1.729e-5 | 0.276411 |

These noiseless-plant, arbitrary-unit results verify the constructed mathematical
problem. They are not vibration-suppression, on-sky, or closed-loop performance
measurements. The batch RLS oracle includes deterministic output noise separately.
Accelerometer disabling by zero channels versus zero history produces identical
dimension-reduced fits; the supplementary zero-feature-column fits/predictions/
reference commands agree within the scaled FLOAT tolerance of 3e-5. A scalar
compatibility trace checks the unchanged model-scaled operational law, exploration,
and extra WFS feedback, and matches its WFS/DM identification to the corrected
baseline once both use complete histories.

### Remaining test roadmap

The older `testRLS.cpp` and `testPredCtrl.cpp` are retained as historical demos;
they are not substitutes for the new assertion-based numerical tests. The broader
roadmap below remains applicable: its numerical history/RLS/control/baseline cases
and the numerical applied-command portions are now exercised, while decoding,
timestamped multirate transport, native app states, and physical correction remain
unimplemented and unvalidated.

1. **History/layout and delayed-label oracle.** Use uniquely numbered frames,
   modes, and channels to assert every element of `y_p,u_p,a_p,u_f,y_f` at a
   chosen origin. Cover `m=1,2,>2`, `F=1,2,3`, unequal history lengths, zero
   accelerometer features, startup, reset, and ring wrap. Reproduce the
   `m=2,F=3` duplicate-command defect. Freeze origin features, mature them `F`
   steps later, and prove that no target-period/late-arriving accelerometer
   sample enters `a_p`. Verify the nearest command block for newest-first
   ordering and explicit command-delay cases.

2. **Independent RLS numerical oracle.** On seeded, well-conditioned data, compare
   every conventional and QR update against double-precision exponentially
   weighted batch least squares solved with QR/SVD. Include the decaying prior:
   after `k` updates, minimize
   `gamma^k*rho*||Theta||_F^2 + sum_j gamma^(k-j)*||z_j-Theta*phi_j||^2`
   for the zero initial model. Check gains, predictions, and weights; compare
   conventional `P` to inverse information and `R^T*R` to information, including
   reset and gamma below one. Assert symmetry/positive definiteness and finite
   outputs. Use scaled tolerances established for FLOAT (initial target about
   `1e-4` relative for
   well-conditioned short traces); report conditioning and revise tolerances
   from error analysis, not merely to make a test pass. Stress collinearity,
   scale imbalance, long runs, and invalid parameter rejection separately.

3. **Known extended predictor.** Generate a causal MIMO plant with known `A,B,D,C`,
   independent excitation, channel mixing, and known delays. Assert prediction
   convergence and recovery of identifiable coefficients. Include `D=0`, sensor
   noise, unrelated sensor motion, and correlated WFS/accelerometer regressors.
   In rank-deficient cases, test predictive performance and conditioning rather
   than requiring a unique recovered `D`. Use multiple lines and broadband
   disturbances, not only 28–29 Hz.

4. **Optimal-law oracle.** With frozen matrices, compare the entire command
   sequence and selected next block with an independent double-precision solve
   of the requested quadratic objective. Assert the gradient
   `(C^T*C+lambda*I)*u_f + C^T*(A*y_p+B*u_p+D*a_p)` is near zero; perturb the
   optimum to check cost increase. Cover non-square blocks, mixed channels,
   permutations, rank-deficient/zero `C`, positive lambda, sign, and gain/noise
   disabled. Test any compatibility regularization/blend separately and verify
   it is not mislabeled as the same optimum.

5. **Baseline regression.** Compare legacy scalar supported behavior on captured
   traces, documenting intended corrections. Then assert that the corrected
   core with `q_a=0` matches its WFS/DM-only reference in dimensions, histories,
   RLS, predictions, and commands within numerical tolerance. Do not compare a
   bug-fixed augmented model against a differently indexed baseline. Test
   disabled accelerometer startup without opening that stream and explicit
   model transitions/dropout without applying an untrained model.

6. **ADC decoding and preprocessing.** Feed exactly four-byte X/Y UINT16 frames:
   zero point, one count above/below, endpoints, saturation, swapped channels,
   wrong datatype/shape, and malformed values. Check calibrated signed outputs
   within encoder quantization, no out-of-frame access, optional explicit FLOAT
   input handling, fixed normalization, and calibration-version changes. Use
   address/undefined-behavior sanitizers on these offline paths.

7. **Multirate timing and transport.** Use a synthetic 8-kHz producer and 2-kHz
   WFS timeline with known exposure/availability times, then noninteger rates,
   jitter, offset, drift, late callbacks, stale frames, duplicate semaphores,
   counter gaps/resets, producer tick skips, dropped WFS frames, stream
   replacement, clock jumps, ring overflow, and overwritten slots. Assert exact
   bin/history assignment, cutoff causality, correct invalidation, no fabricated
   samples, and bounded queue behavior. Compare one filtered vector and phase-bin
   features over equal physical durations. Verify filter response/alias rejection
   and included group delay. Native ImageStreamIO tests must exercise real
   metadata publication using isolated test names and mock sinks, without any
   physical DM endpoint.

8. **Command/training state tests.** In a delayed, sampled, saturating/holding mock
   plant, distinguish proposed, published, and applied commands. Verify training
   follows the accepted/applied sequence while prediction/training/correction
   toggle independently. Shadow proposals must never enter actual-command
   history. Assert no physical endpoint is opened or written in offline/shadow
   modes, including reset, startup, shutdown, stale data, and load failures.
   Check output initialization and atomic frame/model handoff; use thread
   sanitization or targeted concurrency stress where supported.

9. **Offline closed-loop scientific comparison.** Use matched disturbance seeds,
   WFS integration/noise, modal response, latency, regularization, and command
   constraints. Compare WFS/DM DDSPC against augmented DDSPC and a conventional
   feedback reference. Evaluate unseen contiguous records and multiple seeds,
   with a history/horizon gap at train/test boundaries. Measure held-out future
   WFS RMSE, explained variance, residual TT PSD, integrated/total variance,
   command RMS/peaks, and adaptation after changing frequency, amplitude, phase,
   coupling, latency, or sensor quality. Distinguish predictions conditioned on
   recorded future actions from causal closed-loop forecasts/actions. Report
   uncertainty across runs; reduced prediction error alone does not establish
   stable or improved correction.

10. **Non-actuating runtime acceptance.** Replay recorded public telemetry at
    target rates and then use operator-connected live streams with correction
    inhibited. Record collection, feature building, prediction, RLS, solve,
    handoff, and publication timings, including p99/p99.9/max, queue depth,
    sample ages, missed deadlines, and memory/allocations. Test conventional
    and QR paths at the intended dimensions; confirm any slower solve cadence
    preserves physical forgetting time. Agree on an app budget within the
    500-microsecond WFS period before enabling correction.

11. **Operator-run accelSim validation, without hidden state.** First validate
    raw `acc000` capture with simulator DM access unset. For an eventual optical
    experiment, the operator runs separately approved disturbance injection and
    records external WFS residuals, actual control-command telemetry, and raw
    accelerometer metadata. The controller and performance evaluator must not
    consume `sample::m_pointing`, `m_wfsPointing`, mode states, analytical
    acceleration conversion, or future simulator disturbance commands as
    predictor features/targets. WFS residuals are the measured targets. Public
    disturbance-channel telemetry may audit delivered injection/timing, not
    serve as privileged disturbance knowledge for either controller. Confirm
    matched disturbance realizations from recordings: identical RNG seeds alone
    do not guarantee identical delivered waveforms when scheduling skips differ.
    Compare correction disabled, corrected baseline, and augmentation using
    repeated/matched runs and broadband metrics. All hardware steps are executed
    by the user; numerical/mock tests precede them.

Use existing Eigen and the repository Catch2/test harness conventions, with
application groups under `application_unit_test`, namespace
`libXWCTest::loPredCtrlAccTest`, per-case Doxygen, and explicit real-API references
where harness indirection hides them (`tests/testXWC.hpp`, `tests/groups.dox`,
`tests/Makefile.one`). Keep numerical tests independent of app startup and streams.
Follow existing app header-only patterns for app lifecycle code and retain
out-of-class `.cpp` definitions for the established numerical helpers. Touched
numerical headers/sources now have full-file documentation and formatting passes;
the app lifecycle header remains untouched. Standalone numerical tests use the
dedicated Makefile above rather than linking the application library through
`Makefile.one`.

## Implementation Plan

Each stage has a reviewable exit condition. Phase 1 (P1 below) is implemented and
numerically tested under its explicit synthetic command/timing contract; live
instrument decisions from P0 remain unresolved. Stop here as requested. P2–P5
require subsequent authorization and are not part of this implementation.

1. **P0 — Agree on the mathematical and telemetry contract.** Resolve the
   decisions above; record signal units, increment/absolute convention, frame
   timeline, command delay, block ordering, history lengths, lambda meaning,
   and baseline definition. Define model readiness and explicit offline/shadow
   states. Preserve legacy scalar traces for comparison. Establish an independent
   numerical oracle and failing history/control-law tests before changing
   production logic. Exit: accepted equations and reproducible failures for the
   identified indexing, causality, and zero-model issues.

2. **P1 — Complete: correct and verify the DDSPC numerical core offline.** Work
   is isolated to `apps/loPredCtrlAcc/` and its offline tests. Origin-based
   records, command row indexing, delayed accelerometer features, independent
   actual synthetic input collection, verified RLS, opt-in QR structural zeros,
   explicit block extraction, positive regularization, nearest-action selection,
   complete-history gates, and the `q_a=0` corrected baseline are implemented.
   Extra feedback/integration remain explicit legacy conventions. Exit achieved
   for the stated synthetic discrete-time contract: the numerical tests above
   pass without application startup or actuation. Live applied-command semantics,
   native app readiness, timestamped acquisition, and closed-loop stability are
   not part of that completion claim.

3. **P2 — Implement the input contract and offline multirate replay.** Add typed
   UINT16 decoding/calibration, bounded timestamped acquisition records, gap and
   generation tracking, exposure/deadline alignment, and the one-vector feature
   builder. Retain raw samples; compare phase-bin features over equal durations.
   Reuse base-class interfaces/macros, tested recorder metadata patterns, and
   existing numerical utilities. Add versioned preprocessing/model metadata and
   reject incompatible restores. Exit: tests 6–8 and matched offline predictor
   and closed-loop comparisons pass; representation and history duration are
   selected from broadband results.

4. **P3 — Integrate live collection and shadow operation with no correction
   connection.** Add the requested `ddspc_mode` INDI selector, independent
   standard/augmented model training and diagnostic predictions, per-model
   readiness/error status, and separate collection/training/prediction/correction
   controls. Require correction disabled during selection, reject unready models,
   and report accelerometer health. Add robust frame/model handoff and the agreed
   fallback. Keep file I/O, allocation, lifecycle changes, and routine logging
   out of WFS processing. Publish diagnostic predictions/proposals only to an
   unconnected stream. Benchmark RLS/solve cost; compute needed block products
   directly and allocate only the selected learner if validated. Move expensive
   updates to a bounded worker or reduce solve cadence only if measurements
   justify the architectural change. Exit: native transport/state tests and
   test 10 meet agreed timing/coverage limits; actual commands remain the
   training inputs even when proposals differ.

5. **P4 — Validate against accelSim through public streams.** Begin with a
   local/isolated simulator configuration that has no DM channel. The user then
   supplies recorded or operator-run optical WFS/DM traces from independently
   approved disturbance injection. Freeze appropriate training/evaluation
   intervals and compare baseline/augmented prediction in shadow mode, including
   multiple frequencies and broadband cases. Do not import simulator internals
   or the concurrently developed frequency-specific controller. Exit: test 11
   demonstrates valid synchronization and predictive improvement on held-out
   measured WFS data, with documented model conditioning and latency.

6. **P5 — Add the approved correction route and commission through the operator.**
   Only after numerical, offline closed-loop, and live non-actuating gates,
   connect the agreed calibrated modal-to-DM route to a separately approved
   additive correction channel. Validate mapping, sign, gain, publication and
   application delay, command limits, inhibition/zero/ownership behavior, and
   interaction with the main AO loop using mock sinks first. The user performs
   all real disturbance/correction actions and reports results. Repeat matched
   baseline versus augmented suppression/stability/adaptation measurements.
   Exit: measured correction improves agreed metrics without deadline failures
   or unintended amplification; no agent controls instrument hardware.

Keep the original `loPredCtrl` app intact until scope for shared fixes is approved.
After correctness, consider a small common DDSPC numerical core to remove the
duplicated RLS/controller implementations, with unchanged adapters and regression
coverage; do not combine that cleanup with initial mathematical fixes. For future
authorized commits, use clean functional increments with this engineering record,
then documentation-only changes and separate formatting-only cleanup as required
by `AGENTS.md`. This work does not switch branches, commit, or push.

## Follow-up and Edge Cases

- **Phase 1 boundary:** corrected mathematics are available only through the
  explicitly selected offline Causal path. The app still defaults to Legacy,
  including its known accelerometer decoding, timing, and applied-command
  limitations. The tests do not establish real exposure/actuation delay,
  increment-versus-absolute DM semantics, simultaneous sensor acquisition,
  on-sky improvement, or closed-loop stability. The synthetic held/clipped input
  history is deliberately labeled and is not inferred instrument readback.

- **Numerical/build limits:** the independent tests use `-fno-fast-math`.
  `Make/common.mk` defaults to `-ffast-math`; review those flags before claiming
  the same nonfinite-input checks in a native app build. Native Linux application
  linking, real-time scheduling, and instrument interfaces were not exercised.
  The tests establish FLOAT accuracy for the specified well-conditioned and
  zero-column examples, not arbitrary long, ill-conditioned, adaptive telemetry.
  QR persistence, artifact migration, independent WFS/DM history lengths beyond
  the existing `N`/`N-1` convention, and measured multi-frame command delays are
  deferred. The new reference gain may be evaluated for arbitrary rectangular
  blocks, while the existing controller class still uses equal residual/input
  mode counts.

- **Deployment evidence remains outstanding:** exact live stream endpoints,
  datatype/retention, WFS exposure correspondence, command acceptance/readback,
  physical latency, mode calibration, and available real-time budget. The operator
  must provide this evidence; source inspection cannot verify it.

- **Identifiability and causality:** accelerometer histories may add no independent
  information beyond WFS history, and unobservable sensor modes should not drive
  DM commands. Collinear regressors can give nonunique `D`, and closed-loop
  measurement noise/action correlations can bias least-squares fits. Test
  excitation and conditioning before considering more identification machinery.
  Validate expected delay/causal structure of `C`; a dense fitted horizon matrix
  can capture spurious correlations with actions taken after a measurement.
  Scalar stability assumptions do not automatically extend to coupled adaptive
  MIMO control with extra regressors.

- **Control state initialization:** `full_command` is resized but not zeroed
  in `allocate()` (`loPredCtrlAcc.hpp:607–615`), then read on the first integrating
  update (`899–903`). Model reset does not establish a known applied output.
  Initialize from the agreed current-command/zero contract, account for any
  downstream integrator, and test transitions without discontinuities. This is
  a prerequisite for later output work, not a request to reset a real DM.

- **Configuration/persistence defects to include when their paths are touched:**
  `copy_gain` is declared but loaded from `parameters.gain`
  (`loPredCtrlAcc.hpp:347,382`); model files omit accelerometer layout and QR state;
  dimensional changes and load/reset must rebuild dependent workspaces. Validate
  mode count against input width, single-column FLOAT WFS geometry, positive
  horizons/history, finite numerical parameters, and exploration sequences.
  Existing malformed-input and concurrency behavior is not a sound baseline to
  preserve silently.

- **Stream failure modes:** initial existing-but-stale images, duplicate posts,
  missing per-slot metadata, producer restart/counter reset, actual counter wrap,
  overwritten rings, clock adjustments, WFS or ADC rate changes, late transport,
  and sequential hardware channel skew all require explicit validity handling.
  Rewarm after discontinuities rather than treating them as stationary data.
  Reconnection or reconfiguration must not resume physical correction implicitly.

- **Preprocessing and duration:** changing calibration, normalization, filter,
  history duration, representation, rate, or timing offsets changes the meaning
  of learned columns. Invalidate/retrain or use a proven state transformation;
  do not reuse saved weights solely because dimensions happen to match. A
  28–29-Hz validation does not establish broadband alias/noise robustness.

- **Other commands and delivered disturbance:** shared DM channels, a combiner,
  clipping, hold behavior, and another feedback loop can make calculated and
  physically delivered commands differ. The reviewed simulator publishes its
  DM disturbance after its ADC publication using the same completed sample
  (`~/source/MagAOX/apps/accelSim/accelSim.hpp:967–979`); the two publications have
  skew and the driver may coalesce updates. Its physical injection uses
  `m_pointing`, not its separately attenuated `m_wfsPointing`
  (`accelSim/README.md:57–73,103–105`). Learn external sensor-to-WFS coupling from
  observations rather than assume its internal attenuation is the optical plant.

- **Validation boundaries:** raw accelerometer-only tests cannot identify `C`
  or demonstrate WFS suppression; shadow proposals cannot demonstrate closed-loop
  stability. Offline success does not authorize correction. A shared-channel
  zero is itself a hardware command and must remain an operator action on the
  actual instrument. `dm01disp10` must never be written by this controller.

- **Changed files for Phase 1:**
  `apps/loPredCtrlAcc/ar_controller.hpp` and `.cpp` (documented indexing modes,
  corrected offline records, explicit prediction blocks and reference law);
  `apps/loPredCtrlAcc/qrd_rls.hpp` and `.cpp` (documented opt-in triangular
  invariant); `apps/loPredCtrlAcc/tests/ddspc_history_test.cpp` and
  `ddspc_math_test.cpp` (independent numerical assertions);
  `apps/loPredCtrlAcc/tests/Makefile.offline` (standalone reproducible build);
  `tests/groups.dox` (application test group); and this planning record.
  All original app/stream interfaces and `apps/loPredCtrl/` are unchanged.

- **Scope verification:** preserve the existing task text above this heading,
  including the user's runtime mode-selection requirement. Phase 1 does not
  implement that future INDI feature, live `acc000` ingestion, ImageStreamIO
  changes, or physical DM correction. The separate `~/source/MagAOX` worktree
  and Git history were not modified. Build products remain in ignored
  `apps/loPredCtrlAcc/build/` directories. Formatting, optimized tests,
  sanitizer tests, and `git diff --check` pass; no hardware actions were taken.
