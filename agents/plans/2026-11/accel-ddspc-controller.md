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

**Phase 5 offline build/validation work (2026-11-05): comparisons and benchmarks
are implemented; native Linux compilation remains unverified and blocks the
phase exit.** The working tree already contained the uncommitted Phase 4 review
and tests; these were retained. HEAD/branch remain unchanged. This session runs on
`ASTR-LT-428.local`, macOS 26.3.2 arm64, with Apple Clang 21 and Homebrew GCC
15.2.0. No Linux VM/container/runtime or approved development host was available.
No production application, dependency, original `loPredCtrl`, interface, service,
DM path, or separate worktree was modified or executed.

New deterministic comparisons fit three formulations on identical synthetic
applied histories and then freeze their parameters. Informative broadband and
multitone sensors improve one-step held-out RMSE from **0.286633 to 0.00596977**;
zero-coupling and unrelated-sensor controls show no fabricated benefit. At F=3,
independent future disturbance innovations remain unpredictable. Separate frozen-fit
numerical feedback comparisons explicitly retain different control costs and
synthetic clipping/holds; these are not on-sky suppression estimates.

The benchmark establishes a scaling problem in synchronous shadow processing:
with N=5,F=3,c=2,Ha=20, the synchronized builder plus model pair has median
**46.6/54.7 us for two modes**, but **553.7/619.2 us for ten modes**
(classic/QR). All 2,000 ten-mode records exceed the whole-frame 500-us budget in
this run. Hot-path allocations remain. This is an instrumented laptop benchmark,
not a target Linux deadline guarantee or permission to relax timing gates.

**Compatibility qualification:** Clang strict/sanitized suites pass
229,669 assertions / 43 cases; fast legacy suites pass 179,695 / 6. GCC builds the
stream-free application harness and all numerical sources, but two historical
bit-exact QR regressions fail. A separate diagnostic measures maximum clone
command drift **0.000216633** and coefficient drift **0.0323329** over the matched
48-configuration replay; original versus restored remains exact. Classic RLS is
exact. `-ffp-contract=off` does not restore all exact checks. The assertions and
operational QR implementation were not weakened or changed. Phase 4's exact
results below refer to its Apple Clang environment, not a universal compiler
compatibility claim.

`tests/native_linux_build.py` prepares two independent fresh source builds of the
real headers, logger generation, PCH, libraries and binary, preserving default
Legacy flags and separately rebuilding the strict shadow profile. It rejects
non-Linux and instrument hosts, never runs/installs the app, and records manifests
and logs. Its Linux build path is pending an approved environment. A local real-header
syntax attempt still stops at missing generated `logger/generated/logTypes.hpp`;
the test doubles are not a native compilation substitute. Stop before any live
integration/testing; remaining decisions and exact reproduction commands follow.

**Phase 4 compatibility review (2026-11-05): the original `apps/loPredCtrl/`
implementation is preserved byte-for-byte.** Starting HEAD is
`078a3932aa4a339cf873fff0eeb149e40cf08f34`, on the same feature branch and
`ASTR-LT-428.local`. Git identifies the unaugmented working lineage at
`f59d68be7311fb2d3138ef2a4ce269d091ee49e7`, including the earlier explicit
on-sky update. Its complete 15-file app tree equals both restored `loPredCtrl`
at HEAD and the working files. The new tests compile that Git snapshot separately
from the restored app and current accelerometer-disabled Legacy clone, rather
than substituting a reimplementation of the historical algorithm.

With matched configuration and defined initial state, 9,216 numerical frames
and 384 actual-app-header callbacks match **bit-for-bit** in calculated commands,
mock publications, histories and learned matrices, under strict and legacy
fast-math builds. The original path and both production controller sources are
untouched in Phase 4. New shadow selection cannot publish or change legacy
commands; rejected selection/training requests are included in the replay.

**The clone is not an unconditional replacement for the original app.** Existing
pre-Phase-1 changes include different app QR/copy defaults, larger history rings
(which change extreme-mode modulo aliasing), and feedback reconstruction on reset
after loading a different gain. Tests characterize these differences explicitly;
they are behavior differences, not floating-point drift. Thus the earlier broad
“unchanged legacy” wording applies to the preserved original and matched ordinary
replays, not every clone configuration/state transition. No defaults, timing gate,
control law, synchronization algorithm or runtime behavior is changed in this
review. Live accelerometer work remained deferred when Phase 4 ended.

**Completed Phase 3 status (2026-11-05): independently testable causal input integration is
implemented; live training remains gated.** Work began from clean committed
Phase 2 HEAD `fa0e51c5f097a289278c6274ebdd0aef70cd32ce` on
`parkertjohnson1/accel_ddspc`, host `ASTR-LT-428.local` (not an instrument host).
The new `CausalInputBuilder` accepts explicit WFS exposures and confirmed held
applied-input intervals, checks identity/clock/availability/coverage, and emits
one shared chronological record to the two existing independent learners.
Application integration is through protected, serialized timing APIs, with INDI
input-reason and provenance diagnostics. All live source, application and
exposure assertions remain unset: no endpoint or physical timing is guessed.

The repository identifies useful **candidate** telemetry, not the active
instrument route: WFS input is configurable, camera DMA timestamps are not
specified exposure boundaries, `_shape` is posted before SDK submission, and
`dmRecon` replaces source timing when projecting `_delta` into modes. Thus no
new live monitor/reader is attached. A native WFS-only callback disarms any timing
session and keeps applied/timing validity false. This satisfies the user's
fallback scope when verified telemetry is unavailable; it is not a claim that
live applied-DM history or 2-kHz operation has been achieved.

No `acc000` acquisition, ImageStreamIO interface, simulator worktree, physical
output, instrument command or Git history was changed. Legacy execution,
independent models, selection and the Phase 1 law remain intact. Stop after
Phase 3; requirements for a verified live adapter remain listed below.

Investigation and Phase 1 numerical implementation were completed on 2026-11-05 on
`ASTR-LT-428.local`, branch `parkertjohnson1/accel_ddspc`, based on
`b9648ec958ebe3401dc20deaa240e84a60f25d41`. That implementation was authorized
for Phase 1 only:
mathematical correctness and independent offline tests. No application was
started, instrument interrogated, physical command written, branch changed,
commit created, or change pushed. The separate `~/source/MagAOX` worktree was
not modified.

**Completed Phase 2 application integration (historical record):** the user's subsequent
instruction authorizes application/model management and INDI selection only,
starting from committed Phase 1 HEAD `e184a2a48d5de02cfa59723387bfdc7d110d30c4`.
`loPredCtrlAcc` now has an opt-in startup `parameters.shadow=true` path with
independently owned standard and assisted causal RLS models, readiness-gated
`ddspc_mode`, independent training requests, and prediction/validity diagnostics.
The legacy operational controller remains the startup default and retains its
historical law, histories, output dispatch, and limitations. Neither causal
model has an output publisher. No native app, driver, or stream was started;
no branch/history operation or separate-worktree modification was performed.

The application input seam accepts explicitly verified or synthetic synchronized
records. **It is deliberately not connected to live training:** no applied-DM
readback or WFS exposure/command correspondence is established by the current
app. The WFS-only shadow callback invalidates training rather than substituting
its local command candidates or zeros. Live `acc000` ingestion remains deferred.
Shadow execution skips accelerometer/output/telemetry helper startup and rejects
all output acquisition/copy entrypoints. Mode selection cannot attach them.

The existing `HistoryMode::Causal`, origin alignment, corrected QR invariant,
and literal-lambda reference law are reused without numerical-core changes.
`HistoryMode::Legacy` remains the constructor default. The original
`apps/loPredCtrl/` implementation is untouched. Phase 2 details and reproducible
validation appear below, followed by Phase 3 results; these are offline integration results, not instrument
or real-time results.

Numerical tests reproduced both reported history defects before the corrections,
verified conventional and QR RLS against independent batch least squares, and
recovered the learned `D` block of a known synthetic two-mode plant. They also
exposed a QR structural-zero defect; an opt-in correction is enabled only for
the causal offline path. Test results and reproducible commands are recorded
below. These are numerical results, not live prediction or correction results.

Original source line references in the investigation below refer to the
pre-implementation revision above. The new Phase 1 contract and implementation
record name the current functions explicitly; formatting and documentation
changes shifted the old numerical-helper line numbers. Phase 2 app references
use function names; the required full-file documentation/formatting pass also
shifted application line numbers.

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

**Critical findings retained in the legacy path:** the accelerometer path reinterprets UINT16 payloads as
FLOAT, loses intermediate samples, cannot detect a stream that becomes stale
after its first sample, and trains with accelerometer histories from the wrong
prediction origin. A shared command-indexing defect affects multiple controlled
modes. The quadratic controller contains accelerometer cross terms implicitly,
but its applied law also includes scaled regularization and an extra feedback
term. Training records calculated commands rather than confirmed applied
commands. The historical path had no explicit non-actuating mode; the new
startup-only shadow path below supplies one without changing that path.

The mathematical reference is [Haffert et al. (2021), Section 2,
equations 3–15](https://arxiv.org/pdf/2103.07566). Its integral/incremental command
convention and receding-horizon selection need an explicit mapping to the app's
input/output conventions. The extension should retain RLS and add exogenous
accelerometer regressors; it does not require an acceleration-to-displacement
formula or a predetermined spectral line.

## Key Assumptions

**Phase 5 assumptions and scope:** synthetic `u_i` is an absolute modal input
actually held for the transition to `y_(i+1)` by the numerical plant. No stream
counter, publication, callback, requested increment, SDK return or nominal rate
establishes that convention physically. Evaluation sensor samples are available
at the decision origin by construction, not by a measured instrument clock
mapping. Approximately 8 kHz / 2 kHz implies nominal 125-us / 500-us periods;
phase, jitter, per-channel skew, timestamp meaning and usable decision budget are
unknown until independently measured. All Phase 5 source additions are offline
test/build tools. Original operational defaults, reference law, INDI gates,
independent learners and non-actuation protections remain unchanged.

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

7. **Phase 2 execution and input boundary:** `parameters.shadow` is a startup-only
   configuration, default false, with no runtime actuation-enable or execution
   transition. Both causal models use `u_i` held on `[i,i+1)` and the Phase 1
   newest-first ordering; the next hypothetical command is the last horizon
   block. `ShadowRecord` validity assertions are an explicit caller contract,
   not inferred hardware readback. Current application records in tests are
   labeled synthetic; live applied-command/timing validity remains false.

8. **Readiness means usable shadow diagnostics only.** It is not an acceptance
   criterion for correction, stability, suppression, or real-time performance.
   Selection does not change either learner's training request, coefficients,
   histories, or output policy. An unselected learner can train independently.
   Accelerometer loss stops the assisted history without rebuilding the standard
   model; no automatic switch or physical correction is introduced.

9. **Phase 3 exposure/input convention:** `y_i` is the residual from physical
   half-open exposure `[s_i,e_i)`, captured and available by its immutable decision
   deadline `d_i`. Let `L` be an independently established applied-input to optical
   response lag. `u_i` is the calibrated absolute modal input proven constant
   over `[s_(i+1)-L,e_(i+1)-L)`, expanded for WFS/clock/lag uncertainty. Both
   learners still use `u_i` for the next residual, newest-first histories and the
   same `F`-frame prediction horizon. No WFS differencing or additional integrator
   is introduced. Actual live units/sign/order/channel inclusion and latency
   remain unknown; the implementation does not set them from stream names.

10. **Assembly is delayed shadow/replay work:** to confirm `u_i`, the builder
    waits for the following exposure and a closed held-input certificate. It
    subsequently emits `(y_i,u_i,a_i)` in origin order, never appending the
    already observed `y_(i+1)` early to the manager. Phase 1 excludes current
    `u_i` from the origin prediction state, and its target maturation uses only
    previously emitted records, preserving virtual causal identification despite
    this additional processing delay. Origin features are frozen at `d_i`; later
    features cannot repair them. This API is not a next-frame correction path.

11. **Provenance and clock trust are explicit:** all times are signed integer
    nanoseconds in one externally established monotonic clock epoch. Producer
    `atime`/`writetime`, mapped availability, local receipt, exposure boundaries,
    request time and physical effective interval are distinct quantities. The
    builder does not infer offsets, application/settling delays or exposure time.
    Boundary uncertainty must conservatively include clock mapping, exposure and
    response-lag calibration error. An estimated latency without an established
    bound is insufficient for `m_timingVerified=true`. Every test trace is
    synthetic, including tests that simulate an externally verified certificate.
    A provenance enum/flag is a caller assertion, not physical authentication.

12. **Original versus corrected baseline:** `apps/loPredCtrl/` is the preserved
    historical standard DDSPC application. `loPredCtrlAcc` with shadow off and
    acceleration off uses the copied Legacy law; `ddspc_mode=standard` in shadow
    execution selects the separate corrected Causal standard learner. These are
    distinct comparisons. “Standard shadow” is not a renamed/replacement on-sky
    controller, and its literal-lambda law must not overwrite the historical law.

13. **Evidence scope:** Git proves source identity and documents on-sky intent,
    while the user supplies the premise that the original worked on sky. It does
    not identify an actual deployed binary/configuration/session. The replay uses
    the same local compiler/Eigen/flags for all variants, mocked native bases,
    known initial output zero and finite informative traces. It proves data-path
    equality under those conditions, not historical binary identity, thread
    scheduling, physical application or closed-loop behavior under changed
    libraries/hardware. Preserve the original as the operational reference.

## Requirements

### Phase 1 implemented numerical contract

- At the Phase 1 commit, `apps/loPredCtrlAcc/ar_controller.hpp` added `HistoryMode` as an optional final
  constructor argument. Default construction and all existing app call sites
  select `Legacy`; no live mode selector or ImageStreamIO interface is changed.
  `Causal` must be selected explicitly by offline callers. The app lifecycle
  header and its physical output behavior were unchanged by Phase 1. Phase 2
  subsequently connected Causal only to the explicit shadow path.

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
  runtime `ddspc_mode` property and simultaneous diagnostic model management
  were deferred in Phase 1 and are now implemented under Phase 2 authorization
  through the independent shadow manager described next.

- Causal save/load is explicitly rejected before file I/O: the existing legacy
  artifact format lacks sufficient layout/timing/scaling metadata. Dimensional
  and RLS-prior validation is added only to the Causal constructor. Existing
  operational gain, regularization, accumulation, serialization, and startup
  behavior are not silently changed.

### Phase 2 implemented application/model contract

- **Independent models:** new `shadow_controller.hpp/.cpp`,
  `DDSPC::ShadowController::initialize()`, owns two separate
  `PredictiveController(HistoryMode::Causal)` instances. Each owns independent
  conventional and corrected-QR numerical state, histories, update counters,
  excitation evidence, proposals, and frozen-origin forecast snapshots. The
  configured `parameters.qrd` selects each model's update algorithm. Standard
  has zero accelerometer dimensions, not zeroed assisted coefficients:

  ```text
  d = F*m
  p_standard = (F+2*N-1)*m
  p_assisted = p_standard+c*N_a
  Theta_standard: d x p_standard; P_standard: p_standard x p_standard
  Theta_assisted: d x p_assisted; P_assisted: p_assisted x p_assisted
  internal training permutation [C B D A]; public coefficients [A B D C]
  ```

  Shared incoming observations do not imply shared fitted parameters or
  covariance. Reset/pause of one manager model leaves the other unchanged.
  `reset_model` in the app deliberately resets both, retaining their independent
  training requests and current diagnostic selection.

- **Input contract and failures:** `ShadowController::process()` checks explicit
  WFS/applied/timing flags plus column shapes and finite values before updates.
  Assisted additionally requires a finite, synchronized `c x 1` feature vector.
  Missing acceleration breaks only assisted histories/forecasts. Unknown common
  inputs break both; coefficients and requested training are retained, readiness
  and segment evidence are cleared. Forward index gaps start fresh contiguous
  local histories; duplicate/backward indices are rejected without a fabricated
  sample. Restart/generation handling is not inferred from counters: a future
  adapter must explicitly invalidate/reset at a producer generation change.
  Rewarming and at least `max(p,shadow_min_updates)` new updates per segment are
  required. No proposal, repeated accelerometer sample, or zero input fills gaps.
  Numeric faults latch until explicit reset. `invalidate()` retains validity flags
  only, not sample matrices.

- **Readiness and training:** `ShadowController::step()` appends every valid record
  independently of training selection. Both training flags start off. Adaptation
  occurs once per matured complete example only while that model is enabled;
  gamma is per actual update. Readiness requires complete causal histories,
  sufficient segment updates, full empirical regressor rank, finite coefficients,
  and a finite reference-law proposal. The rank test uses the last `p` actual
  regressors, normalizes nonzero columns, and uses Eigen column-pivoted QR with
  relative threshold `1e-5`; the RLS prior cannot supply excitation evidence.
  Zero/fixed/unidentifiable command data cannot become ready just by waiting.
  Pausing retains a ready fit while valid records and predictions continue.

- **Prediction diagnostics:** at a ready origin, save canonical coefficients and
  `[y_p;u_p;a_p]` before any future target is available. On horizon maturity,
  `step()` scores that frozen model with the actual subsequent `u_f` from the
  supplied record stream, before updating with the target. Per-model RMSE is
  cumulative within the uninterrupted segment, over all `F*m` residual values;
  `prediction_valid` indicates scored forecasts exist and distinguishes initial
  zero RMSE from a measured error. This is a **conditional future-input prediction
  score**, not an open-loop forecast with future inputs known in advance and not
  an in-sample error recomputed with today's fit. It does not use later `a` values.

- **Reference control law:** diagnostic proposals use the existing pure
  `get_reference_controller(lambda)` with literal positive
  `parameters.regularization`, then the last `m` rows times the causal state.
  No scaled regularization, WFS feedback addition, exploration, or accumulator
  is added. Proposals remain in the manager and are never appended as applied
  commands or copied into the application's legacy command buffer. Legacy
  `calculate_command()/update_controller()`, feedback, scaled lambda, and
  optional integration retain the compatibility behavior documented in Phase 1.

- **App integration:** `loPredCtrlAcc::configureShadow()` constructs the pair;
  protected `processShadowRecord()` is the synchronized-record integration seam,
  serialized by `m_shadowMutex`. It has no public live transport adapter yet.
  `allocate(shmimT)` validates FLOAT, single-column WFS geometry and resets both
  shadow histories/models without allocating legacy output buffers.
  `processImage(shmimT)` can report finite WFS payload validity but always leaves
  applied-command and timing validity false, invalidating both models. It never
  trains from `m_newCommand` or `m_fullCommand`. These locally computed commands
  still do not establish physical application, clipping, downstream integration,
  ownership, or response delay. No operational behavior is altered to guess them.

- **Output inhibition:** with `parameters.shadow=true`, config/startup/logic/
  shutdown omit `frameGrabber` and legacy telemetry helper dispatch; startup
  omits output semaphore initialization, and WFS processing returns before
  clock rewriting, legacy command generation, signaling, or publication.
  `configureAcquisition()/startAcquisition()/loadImageIntoStream()` reject;
  `acquireAndCheckValid()` returns no frame. The legacy accelerator monitor is
  never loaded/started/processed by shadow execution, regardless of
  `accel_enabled`. `learn`, `learn_std`, `predict`, `integrate`, exploration,
  legacy save, and legacy load requests reject in shadow execution. There is no
  correction-enable property, diagnostic stream, or raw `acc000` adapter.

- **INDI schema:** `setupShadowINDI()/updateShadowINDI()` register the properties
  below. Selection publishes a complete one-of-many vector atomically through
  the existing INDI utility. Callback validation rejects wrong device/property,
  wrong type, unknown elements/states, no selection, and multiple selections.
  A changed selection requires a ready, currently valid target model; an
  idempotent request is allowed while warming. Rejected requests retain current
  selection/training and publish Alert when the identity is valid. A selected
  model becoming invalid stays selected but unready; selection is not an output
  policy and no automatic fallback is guessed.

  | Property | Elements/meaning |
  | --- | --- |
  | `ddspc_mode` | OneOfMany `standard`, `accel_assisted`; default standard; read-only during legacy execution |
  | `ddspc_train` | AnyOfMany independent adaptation `standard`, `accel_assisted`; both initially off; read-only during legacy execution |
  | `ddspc_shadow` | Read-only `toggle`, startup execution choice; cannot enable correction |
  | `ddspc_standard`, `ddspc_accel_assisted` | Read-only numbers: `ready`, `training_enabled`, `training` (updated on latest record), `data_valid`, `prediction_valid`, `rmse`, lifetime `updates`, segment `predictions`/`samples`, `feature_rank`, `features` |
  | `ddspc_data` | Read-only `wfs_valid`, `applied_valid`, `timing_valid`, `accel_valid` |
  | `ddspc_status` | Read-only text `execution`, `output`, `selected`, `standard_reason`, `accel_assisted_reason` |

  Legacy execution explicitly reports `execution=legacy`,
  `output=legacy_compatibility`, and inactive causal-model reasons: a displayed
  standard diagnostic selection does not claim that corrected standard DDSPC is
  operating the instrument. Shadow reports `execution=shadow, output=inhibited`.
  INDI state publication takes `m_indiMutex` then snapshots `m_shadowMutex`;
  record processing takes only the model mutex. Reset releases its model lock
  before taking the INDI lock. The manager itself requires caller serialization.

- **Build and compatibility:** app `Makefile` links `shadow_controller.o` and
  offers `DDSPC_SHADOW_BUILD=yes`, exporting `OPTIMIZE=-fno-fast-math` into the
  recursive PCH build. A clean rebuild of app/numerical objects/PCH is required;
  `configureShadow()` rejects a fast/finite-only compiled application. The normal
  legacy build flags stay unchanged. Strict-build numerical objects are required
  for all finite checks. Full app-header documentation, member naming, out-of-class
  inline definitions, and formatting follow `AGENTS.md`; existing config/INDI
  names and legacy calculation expressions are retained. No common base class,
  ImageStreamIO interface, original `loPredCtrl`, or simulator source is changed.

### Phase 3 telemetry investigation: established facts and missing evidence

| Source/path | What the inspected code establishes | What it does not establish |
| --- | --- | --- |
| `loPredCtrlAcc::loadConfigImpl()`, `loPredCtrl::loadConfig()` and `dev::shmimMonitor::setupConfig()/loadConfig()` (`shmimMonitor.hpp:373–386`) | Input comes from `shmimMonitor.shmimName`; its default is the application config name. `parameters.fpsSource=camwfs` is only a nominal rate subscription. | No deployed DDSPC config, active stream name, WFS coordinate/calibration contract, or downstream output ownership is present in this checkout. `camwfs` must not be assumed to be the modal input. |
| `wooferTweeterRecon::loadConfigImpl()` (`apps/wooferTweeterRecon/wooferTweeterRecon.hpp:435–445`) | Defaults include `aol1_modevalWFS`, `aol0_modevalDMf_mon` and `aol1_modevalDMf_mon`. These identify plausible modal products. | These are defaults of another app, not proof of the DDSPC input route, deployed loop number, exposure identity or units. |
| `dev::shmimMonitor::smThreadExec()` (`shmimMonitor.hpp:638–779,805–827`) | Exposes `m_imageStream`, `m_inode`, shape/type, latest `cnt1` slot and pointer callback. Reconnect detects inode/file changes on timeout and calls allocation again. | It does not produce an immutable counter/timestamp/payload snapshot, deduplicate `cnt0`, drain all slots, map clocks, or determine exposures. Semaphore posts can repeat a latest frame. Inode alone is not a persistent generation token. |
| `dev::frameGrabber::fgThreadExec()` (`frameGrabber.hpp:1008–1047`) | Copies producer-selected `m_currImageTimestamp` into `atime`, records `CLOCK_REALTIME` publication time, increments publication `cnt0`, selects slot `cnt1`, and writes per-slot arrays for cubes. | `cnt1` is a slot, `cnt0` is publication identity, and `atime` is producer-defined. Neither timestamp nor counter implies exposure boundaries or DM application. A future adapter must copy/verify metadata and payload consistently and recover retained frames where possible. |
| `edtCamera::pdvAcquire()` (`edtCamera.hpp:549–560`), `ocam2KCtrl::acquireAndCheckValid()/loadImageIntoStream()/frameGrabberPostPublish()` (`ocam2KCtrl.hpp:1592–1655,1658,1716–1743`) | Camera acquisition uses `pdv_wait_last_image_timed()` DMA timestamps. Raw camera frame numbers detect missed/corrupt images. The optional `framegrabber.syncShmimName` (default output name + `_sync`) copies raw publication timestamps/counters to a 1x1 UINT8 sync product. | The code does not identify DMA timestamp as exposure start/end/center or give measured delay to modal reconstruction. Camera hardware frame number is not automatically the modal stream's `cnt0`; sync is publication metadata, not exposure or application proof. |
| `pwfsSlopeCalc::acquireAndCheckValid()` (`pwfsSlopeCalc.hpp:556–575`), `nnReconstructor::processImage()` (`nnReconstructor.hpp:697–705`), both legacy DDSPC WFS callbacks | These processing paths stamp output acquisition time with local `CLOCK_REALTIME` processing/arrival time. | Upstream exposure timing cannot be recovered simply by treating modal `atime` as camera exposure time. The active upstream pipeline is unconfirmed. |
| `alpaoCtrl::commandDM()` (`alpaoCtrl.hpp:417–505`) and `bmcCtrl::commandDM()` (`bmcCtrl.hpp:452–530`) | Apply actuator mapping/scaling and clipping (ALPAO also removes mean); write/post `m_outputShape` **before** `asdkSend()`/`BMCSetArray()`. BMC errors can occur after shape publication. | `_shape` is a prepared/desaturated drive shape, not a post-success application acknowledgment, command-ID-linked hold interval, or measured surface/settling time. A shape publication alone cannot set applied validity. |
| `dev::dm::loadConfig()/processImage()/makeDelta()` (`dm.hpp:1343–1353,1888–1923,2877–2905`) | Defaults `_shape`, `_delta`, `_diff`; after successful `commandDM`, `_delta=outputShape-totalFlat`, `_diff=totalDelta-outputDelta`, with channel reads at calculation time. | No original command counter, hardware effective timestamp, exposure correspondence, atomic combiner/channel snapshot or physical readback certificate is supplied. Other-loop channels, flats, saturation and disturbance inclusion require an explicit projection contract. |
| `dmRecon::loadConfigImpl()/processImage(dmCommandShmimT)/loadImageIntoStream()` (`dmRecon.hpp:432–460,1008–1082,1258–1264`) | Defaults `dmNNdisp_delta`, `dmNNdisp_actmask`, `aolN_CMmodesDM`, and output `aolN_modevalDMf`/`_mon`. Uses a masked modal pseudoinverse; sets output acquisition timestamp to **processing time**. | A plausible desaturated modal-command candidate, not a preserved WFS/driver application timeline. No stable cross-stream snapshot or physical hold proof is added by the projection. |
| `pi335Ctrl::getPos()/appLogic()/recordPI335()` (`pi335Ctrl.hpp:541–621,1342–1362,1613–1634`) | `POS?` queries exist, with sequential axis polling/sleeps in READY and position/setpoint telemetry. | No WFS-rate per-command application certificate. Polling/status timestamps and setpoint-tolerant INDI display cannot be treated as synchronized applied-modal history during OPERATING. |

The installed mxlib `improc/milkImage.hpp::post()` also stamps acquisition time
from publication time rather than preserving upstream exposure or SDK identity.
This dependency observation is local to this host, not a guarantee of the deployed
mxlib version. No driver, producer, shared-memory stream, or instrument command
was run during the investigation. No active telemetry endpoint was queried. The conventional local
`/opt/MagAOX/config` directory is absent on this workstation, so it supplies no
additional deployment evidence.

**Physical-application limit:** a successful SDK call would at most establish
that a particular drive request was accepted according to that SDK's contract.
It does not itself measure physical surface application or optical settling.
Current `_shape` publication occurs even earlier. Obtaining verified input needs
an independently justified source/recorder for the post-limit combined drive,
original identity, success/hold/drop behavior and effective-time bounds, plus a
validated modal projection. A publication-plus-estimated-delay alternative must
remain explicitly labeled and separately approved; the implemented builder
rejects Requested, DriverPrepared, DriverAccepted and Unverified evidence.

### Phase 3 implemented causal record builder and application seam

- **Types and ownership:** new `causal_input.hpp/.cpp` implements
  `CausalInputBuilder`, `InputConfig`, `InputStamp`, `TimedWfsRecord`,
  `AppliedInputRecord`, `PairedInputRecord`, and `InputBuildResult`. It has no
  ImageStreamIO, SDK, INDI driver, clock read, stream name, output sink or hardware
  access. Caller serialization is required. Two bounded owning deques deep-copy
  immutable matrices and metadata; there is no pointer to a mutable producer
  frame. `m_capacity` bounds each queue; overflow clears pending evidence and
  reports a common discontinuity rather than overwriting old samples.

- **Evidence contract:** constructor validates numerical timing/dimensions.
  Every operation checks established timing and modal mapping, nonzero source
  generation/clock/layout IDs, consistent snapshot flags, shape and finite data,
  and explicit `SyntheticApplied` or `VerifiedApplied` session provenance.
  Command records must match that provenance exactly; switching evidence classes
  requires a newly established session. The positive verified branch is tested
  with synthetic certificates simulating an external assertion, not hardware.
  Nothing in native configuration or INDI sets these verification flags or
  fabricates such certificates. `input_evidence()` labels synthetic data plainly.

- **Physical chronology:** `push_wfs()` requires nonempty exposures of the
  configured duration, monotonic deadlines and availability, receipt no later
  than the origin deadline, and finalization no earlier than that deadline.
  Publication/receipt cannot precede the latest possible exposure end. Consecutive
  WFS counters must correspond to expected physical spacing within configured
  jitter. This detects a skipped physical tick even if published `cnt0` merely
  advances by one. `push_command()` requires a nonempty closed held-input interval,
  nonnegative request time at/before effective start, and confirmation available
  only after the latest possible interval end. Requested time is retained only
  for validation/diagnostics; it never determines matching or shifts a request
  forward by an assumed delay.

- **Alignment and uncertainty:** `drain()` matches the next exposure's expanded
  response-aligned interval
  `[s_(i+1)-L-epsilon_w, e_(i+1)-L+epsilon_w)` against the certified command's
  shrunken interval `[start+epsilon_u,end-epsilon_u)`. A **single** confirmed
  constant value must cover the entire exposure. It never interpolates or
  averages two commands, chooses the nearest timestamp, pairs equal counters
  across streams, or assumes equal stream rates. Uncertain boundaries, transitions
  within exposure, or missing coverage reject the pair and break both histories.
  Multiple exposures may use the same long, explicitly confirmed hold interval.
  This conservative implementation also breaks on gaps between command
  certificates, even if they fall outside a WFS exposure; it does not infer holds.

- **Latency eligibility:** before pairing, the earliest response-aligned input
  window must occur no earlier than `d_i+m_minCommandDelayNs`. This is a necessary
  lead-time condition only, not permission for correction or a measured live
  delay budget. Longer pipelines that leave the next `u_i` already committed
  report `UnsupportedLatency` (`latency_requires_committed_input_prefix`), rather
  than silently reindex commands or change the verified control law. Supporting
  a committed input prefix, exposure-weighted multiple inputs or fractional
  latency models requires separately derived/tested prediction/control changes.
  Neither live `L` nor this lead time is known; tests use declared synthetic
  50-us response lag and 100-us minimum lead time with 150-us request-to-effective
  latency at 2-kHz cadence. There is no frequency-specific assumption.

- **Record and audit:** after confirmation, emit `(y_i,u_i,a_i)` with original WFS
  counter as external index, flags for the verified common input contract, and
  origin/next-target/command metadata, source provenance, frozen deadline and
  response-aligned interval. The original WFS and command counters remain
  independent. `ShadowController::process()` receives that same record once and
  owns the separate standard/assisted histories and fits. The new tests compare
  both fits against manually assembled histories rather than taking the builder
  as their oracle. No calculated controller proposal appears in input history.
  Optional accelerometer features are already-built API inputs only; they must
  have finite correct shape and acquisition/availability before the frozen
  origin deadline. Late/absent features disable assistance for that record while
  preserving the identical WFS/applied input for standard. No raw ingestion or
  resampling is implemented.

- **Discontinuities and silence:** out-of-order/duplicate source counters,
  overlapping intervals, backward timestamps, ambiguous snapshots, identity
  changes, malformed payloads, missing coverage and overflows reject and clear
  pending records. Counter/physical timing gaps clear histories but can accept a
  fresh anchor without compressing omitted time steps. Per-source high-water
  metadata persists through `invalidate()` to reject replay. A different stream
  generation/clock/layout requires a newly constructed session; clock reversal
  latches until then. `advance(nowNs)` checks pending and latest source age during
  silence; an adapter must schedule ticks in the established clock. Fresh events
  can rewarm after a stale-data break; no missing sample is synthesized. Partial
  records from a drain operation that subsequently faults are discarded together,
  not returned across the fault boundary.

- **Application integration:** `loPredCtrlAcc::configureShadowInputs()` establishes
  the explicit source/timing session only in shadow execution, checking mode and
  optional-feature dimensions. It resets both independent fits for any new
  contract (units/timing/layout may have changed), retaining their training flags
  and selection, then invalidates histories. `processShadowWfs()`,
  `processShadowApplied()` and `expireShadowInputs()` serialize through the
  existing model mutex. `consumeShadowInputs()` invalidates both models before
  consuming any returned records when a common discontinuity is reported.
  `processShadowRecord()` remains the Phase 2 pre-synchronized test seam only
  while no timing builder exists, preventing mixed seams. Native WFS allocation
  disarms the timing session even on invalid geometry; the WFS-only raw callback
  also disarms/inhibits, preserving false applied/timing flags. Operator reset
  clears pending records while retaining source high-water protection; shutdown
  releases the builder. The actual raw callback still cannot supply a valid
  record, and no applied-input monitor is added.

- **Diagnostics and compatibility:** `ddspc_status` adds read-only `input_reason`
  and `input_provenance`, including `unverified_input_contract`, waiting,
  paired-input, stale, ambiguity and restart reasons, plus `synthetic_applied`
  versus `externally_verified_applied`. These describe the caller's input contract,
  not a physical validation performed by this app. Default native execution has
  no configured timing builder. Existing independent training/mode selection,
  output inhibition, legacy default, legacy calculation and stream interfaces
  remain unchanged. A new source contract resets shadow fits explicitly; ordinary
  data gaps retain coefficients while clearing readiness/segment histories.
  New builder objects link through the app and standalone test Makefiles.

### Phase 4 historical identity and compatibility findings

**Pinned source and provenance.** `f59d68be7311fb2d3138ef2a4ce269d091ee49e7`
(by syhaffert, author date 2026-05-22, rebased commit date 2026-05-24) is the
latest unaugmented `add_lopredctrl` source restored by
`54b6fa0bec701d09fdc61c212deeda683400fe33` (2026-05-25, “Revert loPredCtrl back
to original version from add_lopredctrl”). Earlier ancestry includes
`e5686991780f713867941c0f5029df57ee84d0e3` (author date 2026-04-10, “small
updates for on-sky usage”) and `9cccca4c26439c81de07b8af0a3a2985ee618dac`
(author date 2026-04-20, QR RLS addition). Rebased commit dates are not evidence
that acceleration was present in the original tree: inspect source and ancestry,
not date ordering alone. The first acceleration edits and copied-app history
are separate from this unaugmented reference.

The app directory tree object is exactly
`d37d0a7e416204b4ab844fb8caf1ba83e0ebff4a` at `f59d68be` and current HEAD.
`git diff f59d68be HEAD -- apps/loPredCtrl` is empty. New
`tests/prepare_legacy.py` verifies both tree IDs and every one of the 15 working
files against the pinned Git blobs on each check, without fetching, switching
branches or editing source. Snapshot compilation occurs only in ignored build
fixtures, with namespace/class renaming and guarded offline base/version includes.
No historical arithmetic, history construction, or app callback body is rewritten.
The exact deployed on-sky binary/configuration remains unknown, rather than being
inferred from an “on-sky” commit message.

| Area | Original `loPredCtrl` and restored current app | `loPredCtrlAcc` Legacy with acceleration disabled | Corrected shadow models |
| --- | --- | --- | --- |
| WFS processing | `loPredCtrl.hpp::processImage(shmimT)` (`499–639`) reads first `m` FLOAT modes directly; no differencing or timestamp-based history alignment. `calculate_command()` alone appends measurements. | Same operational sequence and residual values on the matched trace; shadow/input-builder branches return before legacy work only when startup shadow is explicitly enabled. | Explicit Causal records, contiguous data/warm-up/validity requirements; both models receive the same common record. |
| Command history | `ar_controller.cpp::calculate_command()` (`243–279`) appends the calculated **delta/candidate**, before app accumulation and publication; no applied-command reader. | Same candidate history. Larger copied ring capacity is a genuine compatibility limit at extreme mode counts. No applied telemetry is silently substituted. | Explicit supplied input, Phase 3 builder presently specifies absolute held modal input; this is not automatically the legacy incremental command convention. |
| Indexing | Newest-first rows; original `get_command_future(1)` skips `m` rows, retaining the multimode indexing defect (`137–156`). | Legacy keeps the row skip, including its defect; matched ordinary histories are exact. | Causal skips one frame independent of mode count. This intentional mathematical correction is not applied to the on-sky implementation. |
| Identification | Conventional and legacy QR RLS update once when `learn` is set, after generating/publishing the request; zero-filled startup histories have no app readiness gate. | Conventional algebra unchanged; QR structural-zero correction defaults **off** for Legacy. Exact coefficients through resets/regularization changes in the replay. | Independent standard/assisted fits; corrected QR invariant, complete-history/readiness gates and one update per matured record. |
| Operational gain | `update_controller()` (`223–240`) forms `H=Theta^T Theta`, uses `lambda_effective=maxCoeff(C^T C)*configured_regularization`, explicit inverse, and takes the last `m` rows of the horizon gain. | Same scaled inverse-based law, not the new literal-lambda helper. | Separate LDLT reference law with literal positive lambda. |
| Feedback/integration | `calculate_command()` uses `(controller+integrator)*[u_p;y_p]+noise`; the matrix named `integrator` adds fixed current-WFS feedback. App `is_integrating` separately accumulates the output buffer. Clipping remains disabled by `if(false)`. | Same feedback/accumulation expressions. Reset after loading a different feedback gain is a preexisting clone difference, described below. | No added feedback, exploration or accumulation in reference proposals; no proposal becomes a training command or physical output. |
| Generation/publication | `processImage()` sets `m_updated`, posts the output semaphore, then updates learner/gain. `loadImageIntoStream()` (`690–697`) copies the shared full buffer and clears `m_updated`. | Same behavior when shadow is off, with extra guards that inhibit output only in shadow execution. Test publication sinks are host matrices, not streams. | Startup skips the output helper/semaphore; direct output methods reject. Mode changes cannot start a publisher. |
| Defaults/config/INDI | App gain=0, lambda=100, gamma=1, information prior=1, modes=1, N=5, F=3; prediction/learning/integration off, own output true, **app QR true**. Native names include learn/learn_std/predict/integrate/reset/save/load/exploration/filename/fps. | Same main dimensions/gains, but app QR=false and copy gain=1 if `parameters.gain` is absent. `copy_gain` is declared but reads `parameters.gain` instead. Acceleration and shadow default off. New selector/training properties are read-only/rejected in legacy execution. | Startup-only shadow opt-in; diagnostic selector defaults standard, independent training initially off; no correction-enable property. |

**Confirmed clone differences (not introduced by Phases 1–3):**

1. `539f2bb52e7ea167255bebe6b66dc8dc82254027` (2026-05-26) changed the clone's
   app QR default true -> false, introduced `m_copygainCtrl=1`, used it in the
   nonpredictive output path, and loaded it from `parameters.gain` even though
   `parameters.copy_gain` was declared. With gain and qrd explicitly set to the
   same values the replay is exact. Without those keys the member-default
   characterization gives original output `0` versus clone output `0.25` for a
   `0.25` residual; this test does not certify a missing-required-key configuration
   as deployable. Both numerical classes still default `use_qrd=false`; distinguish
   library default from the original app's true default.
2. The old ring is `next_power(2*(N+F))`; the copied ring is
   `next_power(2*(N+F+N_a+1))`, even when acceleration is disabled. This is already
   present before Phase 1. Ordinary retained windows match exactly. With
   `m=17,N=5,F=2`, old/new capacities 16/32 change how the inherited `skip=m`
   alias wraps: after commands `100..147`, original/restored future history reads
   `146` but clone reads `130`. No tolerance can equate these histories. This
   case is characterized without training or actuation, not silently fixed.
3. The clone already rebuilt its feedback matrix in `reset()` by the copied-app
   creation (`4a14b17d`; the preceding acceleration/rebase edits supply its
   lineage), while original reset leaves the existing feedback matrix intact.
   This behavior exists at pre-Phase-1 `b9648ec9`. A same-dimension artifact with
   stored gain `-0.5` loaded into models constructed at `+0.125` initially produces
   `+0.125*y` in all variants. After reset, original/restored still produce
   `+0.125*y`, clone produces `-0.5*y`. A temporary-artifact regression proves it.
   No persistence/reset behavior is modified in this review. Other dimensional
   artifact migration and legacy persistence defects are outside the equivalence
   claim; ordinary resets with unchanged gain match bit-for-bit.

Original library/app files, `loPredCtrlAcc` production sources and helpers, all
Phase 1–3 code, base interfaces and build flags are unchanged in Phase 4. This
review adds tests/build support and updates this plan only. Source identity of
`loPredCtrl` is stronger than saying the copied legacy mode is universally equal.

### Phase 4 original synchronization and timing reuse

Let `k` count calls to **`calculate_command()`**, not hardware or WFS publication
counters. At original `processImage()` entry a local `CLOCK_REALTIME` arrival
stamp is assigned to the output timestamp. There is no input `cnt0/cnt1` matching,
no exposure interval, no input timestamp copied into histories, no DM readback,
no command application acknowledgment, and no measured/configured latency in the
numerical class. The generic monitor hands over the latest payload on semaphore
wake. Dropped frames can compress history time and repeated payloads can advance
it again; the numerical model cannot detect either from the values alone.

The exact original ordering is:

1. Handle deferred reset/load/save and exploration-stage changes; read `y_k`.
2. If prediction is enabled, append `y_k`, form current `[v_(k-1)..; y_k..]`,
   calculate candidate `v_k` with the previously learned gain plus feedback/noise,
   and append that candidate to the model's command ring.
3. If prediction is disabled, generate copy/gain output; **neither model history
   advances**. The general app frame counter still advances at the end.
4. Replace or accumulate `v_k` into full output and copy uncontrolled input modes
   through. Set `m_updated` and post the frameGrabber semaphore unconditionally.
5. If learning is enabled, build the legacy regressor from its current histories,
   update RLS, and recompute the operational gain for the next command call.
   Learning with prediction disabled can repeatedly update from frozen histories.
6. The independent frameGrabber thread may copy the latest full buffer before or
   after learning; there is no per-command immutable handoff. Multiple callbacks
   can overwrite the buffer before a copy. The regression's skipped mock copies
   demonstrate that the ring records candidates even when publication holds an
   older value. That is a software-order observation, not a hardware drop test.

For F future steps and N WFS history steps after command call k:

```text
Y_future = [y_k, ..., y_(k-F+1)]
Y_past   = [y_(k-F), ..., y_(k-F-N+1)]
V_future(skip=1) = [v_(k-m), ..., v_(k-m-(F-2))]    # F-1 blocks, old multimode skip
V_past   = [v_(k-F), ..., v_(k-F-N+1)]            # N blocks
phi      = [V_future; V_past; Y_past]
```

For `m=1`, the first F command blocks become `[v_(k-1),...,v_(k-F)]`: the intended
one-step convention is that `v_i` influences `y_(i+1)`, with nominal origin
`i=k-F`. The current `v_k` is excluded from that target. For `m>1,F>1`, the old
skip can omit and duplicate frames; for example `m=2,F=3` selects
`[v_(k-2),v_(k-3)]` then starts past commands at `v_(k-3)`. For F=1 the skipped
future vector is empty and that particular defect does not arise. These facts
are preserved independently of whether a specific on-sky configuration suffered
from them. In integrating operation `v_i` is an increment, while the published
full command is accumulated; it is not interchangeable with a measured absolute
DM shape or the absolute-input Phase 3 certificate.

**Recommendation: reuse the software clock and capture point, not an unproven
physical guarantee.** There is no mathematical need to invent a second WFS
scheduler. A future shared, read-only capture adapter can freeze the original
WFS callback identity/order and the corresponding calculated delta, accumulated
request and eventual publication/acceptance identity, without changing its
control law. Both shadow learners should consume the same immutable records and
origin deadlines. For requested-command compatibility studies, a clearly labeled
legacy software-order replay is appropriate; the new regressions supply that
reference. It must not be described as measured applied-command history.

If later evidence establishes one publication/accepted command per WFS step,
fixed latency, command units and no unknown clipping/hold/drop, that ordinal
convention can drive a shared shadow adapter with a validated delay map; an
independently paced WFS/DM timestamp matcher would not be necessary just for
those two signals. Accelerometer features still need causal availability at the
same origin. If commands are coalesced or latency varies, actual publication/
acceptance metadata and a richer timing model remain necessary. Physical surface
readback is distinct from a validated delivered-drive convention; decide the
scientific input contract explicitly. Phase 3's full held-interval certificate is
one conservative supported contract, **not a requirement of the original DDSPC
algorithm**. No such gate is relaxed or alternate live adapter implemented here.

The corrected Causal command indexing and literal-lambda/no-feedback law differ
intentionally from the original even if callback timing is reused. Preserve
both named comparisons: original legacy working controller, and corrected
standard versus assisted shadows on identical declared inputs. Do not call the
latter a bit-identical replacement for the former.

### Phase 4 interference and execution review

| State/resource | Ownership and possible interference |
| --- | --- |
| Original `apps/loPredCtrl` | No new model/selector/data-builder references at all; directory unchanged. It is the safest preserved source reference for operator-run baseline work. |
| Legacy `m_controller` versus `m_shadowController`/`m_shadowInputs` | Separate owned objects and buffers. Default `shadow=false, accel_enabled=false` allocates only the legacy model; timing/model processing APIs reject in that execution. No coefficient, history or output-buffer alias with the new models. |
| New INDI mode/training requests | Read-only and rejected in legacy execution, including forced callback requests in replay. Shadow execution never starts or uses the output helper. Selection/training do not change legacy gain, integration, exploration, model or published buffer. |
| WFS producer versus frameGrabber | Shared legacy full-command buffer, `m_updated` and arrival timestamp have no complete immutable/mutex handoff; coalescing/races predate the new models. This review does not fix them or claim physical publication equivalence under concurrent scheduling. |
| Existing INDI flags/reset/save/load | Legacy flags are written under the INDI mutex and read by WFS processing without a complete synchronization protocol; deferred state/file work remains in the WFS callback. These are inherited risks, not new model buffers. |
| `m_indiMutex`/`m_shadowMutex` | New diagnostics share the existing INDI lock, then take the model lock for a snapshot. Model record processing uses the model lock only. Reset releases its model lock before taking INDI. Legacy WFS does not take the shadow lock. New diagnostics can still add CPU/INDI contention; source separation does not prove identical native latency. |
| Accelerometer monitor | No live monitor is started for disabled acceleration or for shadow execution. Enabling the historical legacy accelerometer path is outside standard compatibility; its fallback can rebuild state across threads and its decoding/timing limits remain. New selection cannot enable that config-only path. |
| External endpoints/dependencies | Running two operational apps against an aliased configured output could interfere independently of model isolation. No native app/stream is opened in tests. Current generic base/OS/dependency builds are not historical binaries; deployment/configuration still determines ownership and timing. |

`git diff b9648ec9 HEAD -- apps/loPredCtrl libMagAOX Make/common.mk` is empty:
Phases 1–3 did not change the original application, generic bases or common flags.
Compared with May `f59d68be`, the repository's unrelated dev merges include
frameGrabber atomic reconfiguration handling, MagAOXApp updates and a DEBUG flag
addition; shmimMonitor changes here remove old comments only. Thus a byte-identical
app tree cannot certify an identical historical executable or real-time behavior.
The stream-free tests deliberately do not start these native worker threads.
Native Linux/PCH build and deployment traces remain required for that evidence.

### Phase 5 build portability, timing and computational findings

**Build evidence and isolation.** `tests/Makefile.offline` now force-includes
`tests/legacy_platform.hpp`, which supplies only the historical BSD `uint` alias
on Darwin. GCC otherwise fails compiling the unmodified historical header; Linux
is unaffected by this alias. The original 15-file tree remains verified against
`f59d68be7311fb2d3138ef2a4ce269d091ee49e7`. All current numerical sources and the
actual `loPredCtrlAcc.hpp` body compile with GCC 15.2 in the stream-free harness.
This validates compiler syntax for that harness, not native base interfaces,
PCH ABI, SDK dependencies or Linux linking. GCC warns about existing
initializer-list pairs in shadow property setup (possible dangling-reference
warning); Clang/ASan/UBSan do not report a lifetime violation. No production
rewrite was made on the strength of that warning.

`tests/native_linux_build.py::main()` stages current versioned/nonignored source
bytes and records SHA-256 manifests. Each profile starts without ignored object,
PCH, logger or local override artifacts. It builds only prerequisite libraries
and `loPredCtrlAcc`, never `loPredCtrl` or a service, and never executes the
linked app. The real Git-header generator reads original metadata with
`GIT_OPTIONAL_LOCKS=0`; generated files stay in the build stage. Shadow's
`OPTIMIZE=-fno-fast-math` is passed to prerequisite builds as well as the app,
preventing reuse of a fast-math PCH. Linux host guards and Python syntax pass
locally; the actual Linux build commands remain unexecuted. Existing development
container examples are `.github/workflows/build-coverage.yml` and
`buildability-test.yml`; those workflows also install/deploy elsewhere and were
not invoked. Do not use their service/install steps for this task. No Linux
compiler compatibility fix to production code can be claimed without real
build output. The initialized VLA in both historical `testPredCtrl` demos is a
separate existing demonstration portability failure, left unchanged.

**Physical timeline, not callback order.** Preserve distinct identities/times:

| Event | Required interpretation/evidence | Inspected source and current limit |
| --- | --- | --- |
| WFS exposure `[s_i,e_i]` | Residual is an exposure-integrated optical measurement, with exposure duration/rolling or global readout semantics and uncertainty. | `edtCamera::pdvAcquire()` and `ocam2KCtrl` provide DMA/camera counters, not established exposure boundaries. Camera sync copies publication metadata. |
| Modal residual available `w_i`, copied locally `r_i`, decision frozen `d_i` | Require `w_i <= r_i <= d_i` in a verified common clock, original camera/exposure lineage and calibration. | `frameGrabber::fgThreadExec()` writes producer-defined `atime`, local REALTIME `writetime`, publication `cnt0`, slot `cnt1`; reconstruction and DDSPC callbacks can restamp arrival time. `loPredCtrlAcc.hpp::processImage(shmimT)` does so at lines 1641–1642. |
| Calculated/requested `v_i`, published output `p_i` | Requested incremental/absolute semantics and original request identity must be explicit; publication can be skipped/overwritten. | Legacy `calculate_command()` records a candidate before app accumulation/publication and before asynchronous frameGrabber delivery; it is not a delivered-input history. Phase 4 replay demonstrates this distinction. |
| Prepared drive, SDK accepted request, effective physical input | Track combined channels, mapping, clipping, hold/drop/rejection, SDK success and bounded physical onset/settling separately. | `alpaoCtrl::commandDM()` posts shape at 473 before `asdkSend` at 480; `bmcCtrl` posts at 502 before `BMCSetArray` at 509. `dev::dm::makeDelta()` and `dmRecon` lose request/exposure lineage and do not measure physical application. |
| Sensor acquisition `t_k`, publication `b_k`, receipt `r_k` | Both acquisition and availability must precede the frozen feature cutoff, with clock epoch, skew, calibration and drop/restart evidence. | The old float-map/latest-sample clone callback is not a typed UINT16 or causal adapter. No live sensor path was added. `mcp3008Ctrl::acquireAndCheckValid()` (334–365) intends sequential channel reads after a software clock trigger; its nominal rate property does not lock conversion phase to WFS. It supplies no per-channel exposure/ADC timestamp here. |

The sequential MCP3008 code is an additional source-level warning against
assuming simultaneous channels or equating `fpsSource` subscription with physical
synchronization. It was inspected, not compiled, run or modified, and it is not
proof of the `accelSim` producer's clock. Previous simulator encoding findings
below remain a separate producer contract; no simulator internals or worktree
were used for the new comparisons.

For the Phase 3 held-input contract, require the confirmed constant input
interval to cover the **whole** response-shifted next exposure
`[s_(i+1)-tau-u_y, e_(i+1)-tau+u_y]`, while shrinking its certified bounds by input
timing uncertainty. Features use only acquisition/availability at or before
`d_i`. Actual input may be learned later; it must never be replaced by a
calculated proposal. Closed hold certificates intentionally delay shadow
assembly past the origin; this builder is not an online correction scheduler.
The deterministic fixture's 50-us response lag, 100-us minimum delivery delay,
200-us exposure and 500-us period are **synthetic assumptions only**. If actual
latency leaves part of the next exposure already fixed, future control needs a
known delayed-input prefix or another validated discretization. If a command
changes within an exposure or DM response is not a pure delay, a piecewise or
response-weighted input model is required; callback reordering, a widened
tolerance, or selecting the nearest telemetry timestamp cannot repair that.

At nominal 8 kHz, four conversions per WFS period do not establish alignment.
A future downsampling/feature study must compare a causally available sample,
causal antialias filtering/averaging with measured group delay, and multiple
ordered samples. Include only the samples available by `d_i`; preserve drop and
channel-skew evidence. Estimate sensor-to-optical lead/response with independent
WFS observations and recorded driver/input events over broadband/multiple
frequencies, without using simulator pointing state. None of those transport,
filtering, calibration or physical-latency changes is implemented in Phase 5.

**Comparisons and allocation/latency measurement.**
`tests/comparison.cpp::compare_models()` generates portable seeded innovations
plus 37/83/211-Hz components, a stable coupled two-mode ARX plant, clipped/held
actual inputs, independent process noise, and separate sensor negative controls.
It uses N=2, Ha=3, F=1/3, identical 1,800 training records and mature update counts,
then freezes fits for 600 evaluation samples. The accelerometer-disabled clone Legacy path is used (Clang-pinned replay above).
Legacy gain=0 and an unsolved gain
matrix let its exploration interface record the common applied sequence exactly;
a precondition checks this. Corrected models append the very same actual input.
Conditional scoring uses an independent newest-first history oracle at each
origin and actual future inputs only as an explicitly offline conditioning
variable. It does not update on held-out labels. Frozen numerical feedback uses
the same exogenous evaluation realization, resets internal histories, and applies
clipping/holds before corrected history construction. Legacy still internally
records proposals during feedback; its known delivery mismatch is not repaired.
App accumulation is outside this numerical comparison. Legacy's scaled penalty
and inverse are retained; corrected/assisted use literal lambda=0.3 and the
nearest block of the Phase 1 reference law. Those distinct costs are disclosed,
not tuned to equal command effort or presented as operational equivalence.

`tests/benchmark.cpp::measure()` warms 512 records and measures 2,000, reporting
p50/p95/p99/p99.9/max, whole-frame 500-us overruns, Eigen allocation permission
checks and separate ordinary C++ allocation counts/bytes. `allocation_probe.hpp`
is force-included only in `Makefile.benchmark`; assertions still execute and a
known allocation self-check must pass. Counts include Eigen's direct-malloc
request path, not just C++ new, but are not system allocator calls/bytes or peak
live heap. Timing includes probe and fixture work; end-to-end builder runs use
pre-generated immutable records. Both models must reach readiness. No external
streams, INDI or native app are linked. At N=5,F=3,c=2,Ha=20, dimensions are
`(p_standard,p_assisted,d)=(24,64,6)` for m=2 and `(120,160,30)` for m=10.

The managed pair recomputes normalized empirical feature-window rank with a
p-by-p pivoted QR, solves the reference law and snapshots forecast matrices
inside `ShadowController::step()` (`shadow_controller.cpp:169–234`) for each
mature valid record. This explains why pair cost greatly exceeds two bare RLS
updates. Per-record allocations and cubic rank work preclude a claim of bounded
real-time execution. Consider a bounded worker with immutable model snapshots,
preallocated workspaces and separately specified rank/solve/diagnostic cadences,
then remeasure on the intended Linux target under load. These scheduling and
readiness-policy changes require a separate implementation/review; none was
made in Phase 5. Do not weaken provenance/readiness or enable actuation to meet
a benchmark.

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

- In the legacy path, startup `accel_enabled=false` sets channels/history to zero and omits the
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

- The legacy fallback can delete/recreate the learner with zero accelerometer
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

- In legacy execution, `predict=false` selects the copy/gain path, and `integrate=false`
  selects command replacement; neither prevents publication. Every WFS callback
  in that legacy path posts the frameGrabber semaphore. `learn_std` enables prediction, training,
  and exploration (`loPredCtrlAcc.hpp:1129–1162`). Non-actuating validation must
  use an explicit state that cannot open/write a physical correction endpoint;
  any later diagnostic stream would need an unconnected name verified by the
  operator. Phase 2 creates no such stream.

- Existing configuration covers mode/history/horizon, gamma, covariance, gain,
  regularization, accelerometer enable/history/normalization, and stream names
  (`loPredCtrlAcc.hpp:337–365`). Legacy INDI controls primarily cover learning,
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

**Phase 5 blockers/decisions before live testing:**

- Supply an approved non-instrument Linux environment with real MagAO-X/mxlib,
  logger, ImageStreamIO, INDI and library dependencies. Run/review both fresh
  native build profiles; no Linux compilation success is claimed from this Mac.
- Decide whether clone QR compatibility must be bit-exact on the supported GNU
  compiler. Investigate the observed drift under that exact compiler/Eigen/flags
  before any clone replacement. Retain the original implementation as the
  operational baseline; do not silently insert the corrected QR algorithm or
  accept the measured coefficient drift as an approved tolerance.
- Provide the actual stream/configuration lineage, exposure/ADC timing meaning,
  delivered combined-drive identity, absolute/increment projection, clock mapping,
  latency bounds and operator-recorded traces. Decide whether a delayed or
  exposure-weighted plant/input representation is needed; no physical values
  were measured here.
- Agree target mode/history dimensions, latency budget, tail/deadline acceptance
  and a bounded adaptation/diagnostic execution design. The ten-mode synchronous
  pair already exceeds the whole 2-kHz frame budget on this laptop.
- Approve any future sensor representation/calibration and live non-actuating
  adapter only after those gates. New actuation, production changes and deployment
  remain outside this work. No permission to use an instrument host was inferred.

**Phase 4 recommendations/decisions before further implementation:**

- Keep `apps/loPredCtrl` and its historical law intact. Use the original as the
  operational source reference; identify the deployed binary/configuration if
  stronger on-sky equivalence is required. Do not silently repair its indexing,
  lambda, startup, accumulation or threading behavior during accelerometer work.
- If `loPredCtrlAcc` must be interchangeable rather than a matched experimental
  clone, separately decide its QR/copy defaults, ignored `copy_gain`, ring-alias
  edge cases and loaded-gain reset behavior. This review exposes those differences
  without choosing a new operational behavior. Explicit matched configuration
  avoids the default differences; it does not erase the other edge cases.
- Agree whether shadow training uses original requested increments under a
  validated software-order convention or verified delivered/absolute commands.
  A common WFS capture point is preferable to duplicate scheduling, but the two
  semantics must not be conflated. Retain existing input safety gates until that
  scientific/timing contract is established.

**Phase 3 missing requirements before any live adapter can set validity:**

- Identify the deployed DDSPC WFS/modal stream, upstream producer chain, residual
  calibration/order, hardware-frame to publication-counter correspondence,
  exposure start/end weighting, and timing of the origin deadline. Supply
  configuration/recorded metadata; candidate names above are not selections.
- Identify accepted/applied input telemetry with original command identity,
  post-limit/combiner values, drop/hold/failure behavior, effective/settling time
  bounds, included channels and calibrated projection into the WFS modal basis.
  SDK preparation/publication/success alone is not physical application proof.
  Decide separately whether a labeled accepted-command-plus-delay approximation
  is scientifically acceptable if direct verification remains unavailable.
- Establish camera/driver/receiver clock relationships and clock-jump handling,
  latency/uncertainty and jitter bounds, verified generations, exposure duration,
  replay retention and freshness thresholds. Decide whether the conservative
  single-held-input exposure model applies or requires a richer sampled plant.
- If the next input is committed by the pipeline, derive the fixed prefix and
  optimization horizon before using a reference proposal for correction. No
  command-offset guess or nominal 2-kHz ratio resolves this automatically.
- Provide a supported Linux build/PCH environment and independent recorded traces
  for native snapshot/retention, callback/race, throughput and timeout validation.
  A future adapter must schedule `expireShadowInputs()` ticks; this phase does
  not equate replay time with the current workstation clock or add a live timer.

These are integration/scientific decisions, not a request for the agent to query
or manipulate the instrument. All necessary independent components above were
implemented without waiting for unavailable live evidence.

Phases 1–3 offline/shadow component work was explicitly authorized without
resolving live instrument semantics. Startup-only shadow execution, separate
learners, independent adaptation, and diagnostic-only INDI selection are now
implemented under that authorization. These decisions remain before live or
operational integration;
recommendations are not assumptions about the deployed instrument.

1. **Correction scope and command convention:** which WFS/modal input and output
   endpoints will be used, and does this replace the current low-order controller
   or add a separate correction alongside it? Confirm residual units, sign,
   mode order, incremental versus absolute output, downstream accumulation, and
   which accepted/applied command telemetry is available. Recommend preserving
   the established modal/calibration route once it is documented, with an
   explicit increment convention if the downstream route integrates.

2. **Baseline and mathematical integration:** the inherited corrections and
   literal reference law now exist in the opt-in shadow application path of
   `loPredCtrlAcc`; legacy execution and `loPredCtrl` remain operational references.
   Approve any future corrected actuation path separately from this diagnostic
   selection. Recommend an explicit
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
   model adapts while unselected for later live use. Phase 2 exposes separate
   training flags and conservatively keeps selection on an invalid model with
   readiness false; no actuation or automatic fallback occurs. Recommend
   inhibition until timing and model validity are proven; measure the cost of
   training both models. Review the Phase 2 empirical-rank/segment-update gates
   before using them operationally. Agree on adaptation time, solve cadence, and maximum
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

At the Phase 1 commit, `apps/loPredCtrlAcc/tests/Makefile.offline` built only the numerical sources,
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

### Phase 2 executed integration tests

The same standalone target now also compiles `shadow_controller.cpp`,
`ddspc_shadow_test.cpp`, `ddspc_app_test.cpp`, and the repository's real
`IndiProperty.cpp`, `IndiElement.cpp`, and `TimeStamp.cpp`. No native MagAOX
application base, INDI driver, ImageStreamIO, or hardware library is linked.
`tests/app_test_support.hpp` substitutes only app/base lifecycle, logger,
config/registration helpers, and semaphore operations; it includes the **real
application header, callback macros, INDI objects, and INDI update utilities**.
Lifecycle/semaphore counters and unchanged destination buffers verify inhibition.
It cannot certify native thread scheduling or native helper integration.
Bundled `IndiProperty.cpp` uses removed `std::ptr_fun/std::not1`, so that unchanged
source alone compiles in C++14; the test/application/numerical sources use C++20.
The existing INDI dangling `what()` pointer and deprecated adaptor warnings are
reported, not modified as part of DDSPC work.

Reproduce from the repository root:

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j3 check
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j3 check \
  BUILD_DIR=build/offline-phase2-sanitize \
  CXXFLAGS='-std=c++20 -O1 -g -Wall -Wextra -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer'
```

Optimized suite: **47,451 assertions in 22 test cases passed**, including all 12
unchanged Phase 1 cases. AddressSanitizer/UndefinedBehaviorSanitizer suite:
**47,451 assertions in 22 test cases passed**, with no sanitizer findings.
New coverage:

- Seven manager cases: conventional and corrected QR independent training,
  pause/reset and dimension independence; readiness-gated/idempotent/invalid-enum
  selection without model mutation; exact standard-model equality with a separate
  corrected `q_a=0` controller at every update; reference proposal equality;
  acceleration-only loss, unknown applied inputs/timing, duplicates and gaps;
  minimum updates and empirical excitation gates; frozen-origin errors for
  `F=2,N=2` against separately reconstructed chronological states/actual future
  inputs (absolute RMSE tolerance `1e-7`); NaN/Inf/shape rejection and invalid lambda.
- Three actual-app-header cases: startup-only shadow configuration and skipped
  helper lifecycle; all direct output methods guarded and destination unchanged;
  no semaphore call or legacy command buffer/model allocation; no training on
  WFS-only data; rejection of legacy actuation/training/exploration/persistence;
  real INDI mode/train callbacks including malformed requests, independent partial
  pause, both models ready with two-mode synthetic records, diagnostics, and reset;
  absent `parameters.shadow` retains legacy lifecycle/config dispatch and rejects
  causal selection/training. Lifecycle is doubled: these tests never start the
  real legacy output path or open a native stream.
- Known-plant `D` learning, independent batch RLS, causal history indexing, and
  pure optimal-law tests remain unchanged and pass as Phase 1 regressions. The
  new manager does not replace those independent mathematical oracles.

A native application `-fsyntax-only` attempt, using installed Eigen/mxlib
includes and the repo `flatlogs/include`, stops at missing generated
`libMagAOX/logger/generated/logTypes.hpp` before validating the native app.
The generated logger/PCH prerequisites and native Linux compile/link/helper
lifecycle remain outstanding; no dependencies were downloaded or generated to
pretend this was a deployment build. `otool -L` on the offline executable lists
only libc++ and libSystem. No native application binary was run.

All six touched/new C++ files receive clang-format 23.1.3 and its dry-run check.
The sole style override, `SpacesInLineCommentPrefix.Minimum=0`, preserves the
literal `{ //mutex scope` annotation required by `AGENTS.md`, where formatter
23 defaults would insert a space. `git diff --check` and the protected-plan-prefix
comparison pass. The initial HEAD/branch remain unchanged; build/log artifacts
stay ignored. These static/offline checks do not establish stream ownership,
actual command application, optical timing, or a real-time budget.

### Phase 3 executed causal integration tests

`tests/ddspc_input_test.cpp` and `tests/causal_input_test_data.hpp` provide
explicitly synthetic 2-kHz exposures, separate WFS/command counters, delayed
closed hold certificates, and clipped actual inputs distinct from requests.
The same standalone target additionally builds `causal_input.cpp`. Eleven new
builder cases and two additional real-app-header cases exercise:

- Exact `u_i -> y_(i+1)` physical exposure mapping with nonzero response/delivery
  delays, original source metadata/provenance and every multichannel
  `F=2,N=3,N_a=2` training-history block across 28 records. A separately assembled
  chronological oracle checks the entire regressor/target, not just a selected
  coefficient or a one-mode example.
- Rejection of Requested, DriverPrepared, DriverAccepted, mismatched provenance,
  unverified timing/mapping, wrong effective interval, wrong generation/clock/
  layout, unstable snapshot flag, malformed/nonfinite data, duplicate/overlapping
  certificates, frame/certificate gaps, publication-count-continuous physical
  tick skips, impossible/future/late timestamps, stale/missing input, clock
  reversal latch and explicit new-generation reconstruction.
- Within-exposure transitions, uncertainty at a boundary, insufficient lead time
  and multi-frame response lag; positive bounded jitter/uncertainty with complete
  hold coverage; independent queue overflow and multi-record drain from an
  explicit long hold; reversed cross-source processing order; exact integer
  nanoseconds above `2^53`, and signed overflow boundary validation.
- Both conventional and corrected QR managers match separately fed, manually
  aligned standard and assisted fits exactly at every step for 60 records. A
  feature arriving after origin 30's deadline affects only assisted validity,
  without changing common WFS/DM histories or enabling hindsight repair.
- Actual app header/callback seam: 80 paired synthetic records match both manual
  fits at every update, become ready with adaptation enabled through real INDI
  requests, allow existing readiness-gated selection, and expire both histories
  during silence while retaining coefficients/training/selection. A changed
  layout/clock contract explicitly resets both fits while preserving training
  and selection. Synthetic provenance is visible in INDI. No output helper or
  semaphore is invoked and no destination/legacy command buffer is written.
  Unconfigured/unverified sessions and driver-prepared commands reject; a raw
  WFS allocation/callback cannot reuse an established synthetic timing session.
  Legacy execution rejects the new timing interface and retains its lifecycle.

Reproduce from the repository root:

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j3 check
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j3 check \
  BUILD_DIR=build/offline-phase3-sanitize \
  CXXFLAGS='-std=c++20 -O1 -g -Wall -Wextra -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer'
```

Optimized and AddressSanitizer/UndefinedBehaviorSanitizer builds each passed
**49,836 assertions in 35 test cases**, including all Phase 1/2 regression cases,
with no sanitizer findings. Build logs remain in ignored app `build/` directories.
These traces exercise the declared input contract; they are not records from the
instrument and do not establish actual latency, physical command acceptance,
exposure timing, prediction improvement, or a native 2-kHz budget.

The six touched/new C++ files receive a full documentation pass and clang-format
23.1.3 using the previously documented comment-prefix override for exact mutex
annotations; its dry-run/`--Werror` and `git diff --check` pass. A native
`-fsyntax-only` attempt using installed Eigen/mxlib and repo flatlogs include paths
still stops at missing `libMagAOX/logger/generated/logTypes.hpp`, before validating
the native app. Native Linux/PCH compile/link, source snapshots and transport are
unverified; the actual-app-header tests use stream-free lifecycle doubles.
`otool -L` on the offline binary lists only libc++ and libSystem. No native app
was started, dependency installed, logger header generated or physical stream
opened. The protected plan prefix, original branch and initial HEAD are unchanged.

### Phase 4 historical regression tests

New `tests/prepare_legacy.py`, `legacy_replay.hpp`,
`legacy_{original,restored,current}_adapter.cpp` and `ddspc_legacy_test.cpp`
execute pinned historical numerical/app source, verified working-tree restored
source, and current Legacy source with zero accelerometer dimensions. They use
separate C++ namespaces/classes and the existing stream-free app/base/semaphore
fixture. The historical source bytes are not formatted or algorithmically
rewritten; only test compilation names and environment include guards differ.
No private-member macro rewriting or arithmetic oracle substitutes for running
that source. Doxygen references preserve links through the harness.

Six regression cases cover:

1. **9,216 numerical frames:** 48 combinations of `m={1,2}`, `N={1,5}`,
   `F={1,2,3}`, conventional/legacy QR RLS, gamma `{1,0.97}`, 192 records each,
   repeated ring wrap, regularization switching, full reset and buffer-only reset.
   During the zero-gain/unsolved-controller phase, the exploration interface
   inserts the exact prescribed synthetic DM trace. Later predictive operation
   is compared as well. Every output, all seven public history/coefficient
   snapshots and weights are checked by float-bit comparison (`memcmp`), not an
   adjustable tolerance. **Maximum observed matched-trace difference: zero.**
2. Explicit old/new high-mode ring alias characterization (`m=17,N=5,F=2`),
   original/restored equality and clone history difference retained as a tested
   compatibility limit rather than hidden by a tolerance.
3. **384 real-app-header callbacks:** matched gain, lambda, prior, history, qrd,
   disabled acceleration and default legacy execution, two controlled plus two
   passthrough modes. Prediction, exploration/learning, integration, deferred
   reset, pause/resume and skipped mock publications use actual callbacks/copy
   methods. Rejected standard/assisted selection and training requests leave all
   outputs unchanged. Candidate, full output, last mock publication, histories
   and fits are bit-identical throughout. The test initializes the otherwise
   undefined output buffer to the same known zero and warms histories before
   solving; these are declared test preconditions, not production behavior fixes.
4. Missing-key member-default characterization and `copy_gain` loading behavior.
   The test fixture does not impose native required-key validation, so it does
   not call a missing-parameter deployment configuration supported.
5. History freezes when prediction is disabled, despite continuing copy/gain
   publication. A skipped-copy schedule also proves candidates can enter command
   history without a matching publication. Neither result claims a physical DM
   observation.
6. Same-dimension legacy persistence to an owned temporary directory, followed by
   a changed-gain reset: original/restored `+0.125` versus clone `-0.5`, with all
   implementations left unchanged. This is not a general serialization/migration
   certification.

Reproduce from repository root:

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j3 check
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j3 check \
  BUILD_DIR=build/offline-phase4-sanitize \
  CXXFLAGS='-std=c++20 -O1 -g -Wall -Wextra -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer'
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 check-legacy-fast
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 check-legacy-fast \
  BUILD_DIR=build/offline-phase4-sanitize \
  LEGACY_FAST_FLAGS='-O1 -g -std=c++20 -Wall -Wextra -ffast-math -fsanitize=address,undefined -fno-omit-frame-pointer'
```

The complete strict offline suite passes **229,531 assertions in 41 test cases**,
including all 35 Phase 1–3 cases. The fast-math target deliberately runs only the
legacy cases, since the causal finite-value gates require strict math: it passes
**179,695 assertions in 6 cases** under `-O3 -ffast-math`, matching native legacy
optimization policy. The full strict ASan/UBSan build also passes **229,531
assertions in 41 cases**; the legacy fast-math ASan/UBSan build passes **179,695
assertions in 6 cases**, with no sanitizer findings in either run.
All variants are built with the same installed Apple Clang/Eigen for each run;
bitwise equality is scoped to those builds and traces, not different compilers,
Eigen releases, unsupported configurations or uninitialized/racing executions.

The existing numerical **demo** programs are also attempted, separately from
assertion tests. `check-demos DEMO_NAMES=testRLS` runs original and current RLS
demos normally and under ASan/UBSan; all four executions exit zero. Full
`check-demos` attempts expose an existing portability failure in both original
and current `testPredCtrl.cpp`: `realT x[num_steps]={0.0}` initializes a variable
length array, rejected by this host's C++20 Clang. No source/fixture arithmetic
is patched to force it to run. These demos have no assertions, and the current
accelerometer-default demo is not a standard-DDSPC equivalence test anyway.
Logs and any generated demo outputs remain under ignored build directories.

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j3 check-demos DEMO_NAMES=testRLS
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j3 check-demos DEMO_NAMES=testRLS \
  BUILD_DIR=build/offline-phase4-sanitize \
  CXXFLAGS='-std=c++20 -O1 -g -Wall -Wextra -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer'
# Full demonstration build attempt (known initialized-VLA portability failure on this host):
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j3 check-demos
```

Native application/link/runtime validation remains blocked by the previously
recorded generated logger/PCH prerequisites and unsupported host environment;
no instrument action is attempted to fill that gap. Existing bundled INDI
warnings are not modified. All new C++ files receive documentation and
clang-format using the established exact-comment-spacing override. Git source
identity, protected plan prefix, unchanged HEAD/branch and `git diff --check`
are checked at completion. No production C++ or configuration file is modified.

### Phase 5 executed offline tests and measurements

Environment: macOS 26.3.2 arm64, Apple Clang 21.0.0, Homebrew GCC 15.2.0_1;
Eigen pkg-config 5.0.1 (header version macros print 3.5.0), mxlib 0.2.0.
No CPU affinity, real-time scheduler, target Linux measurements, native app run,
stream access or instrument communication was involved.

| Check | Result |
| --- | --- |
| Clang full strict suite | **229,669 assertions / 43 cases pass**, including 12 scenario/horizon/RLS comparisons and repeatability. |
| Clang full ASan+UBSan suite | Same **229,669 / 43 pass**, no sanitizer findings. |
| Clang Legacy `-O3 -ffast-math` | **179,695 / 6 pass**; production flags/law remain unchanged. |
| Clang Legacy fast-math with ASan+UBSan (`-O1`) | Same **179,695 / 6 pass**, no findings. |
| Clang hidden compiler-drift diagnostic | **55,298 / 1 pass**; maximum original/restored/clone output and fit difference zero for both RLS methods. |
| GCC full strict suite | **41/43 cases pass; two exact legacy QR regression cases fail** (91,239/91,241 assertions pass). No production behavior change or assertion tolerance was introduced. |
| GCC non-legacy selection `~[legacy]` | **105,272 / 38 pass**; this Catch selection also includes the hidden drift diagnostic. Corrected mathematics, causal records, new comparisons, shadow and app isolation pass. |
| GCC drift diagnostic | **55,298 / 1 pass** as a diagnostic: classic exact; QR clone maximum absolute command difference 0.000216633 (1,451 differing records), fit coefficient difference 0.0323329 (3,573 fits); original/restored exact. This does not pass/replace the exact clone-compatibility acceptance tests. |
| GCC `-ffp-contract=off` full suite | The same two exact QR cases still fail (41/43 cases pass); disabling contraction alone is not a demonstrated fix. |
| Original/current `testRLS` demos | Four smoke runs exit 0: normal and ASan+UBSan; no assertions in those demonstrations. |
| Full `check-demos` | Still fails compiling original initialized VLA in `testPredCtrl.cpp`; both original/current versions contain the issue. Left unchanged; new deterministic comparison is independent of this demo. |
| Benchmark | 24 workload rows, 2,000 timed records each, all pairs ready; CSV and environment captured. Separate ASan+UBSan run with 100 timed records per workload completes without findings. |
| Real-header macOS syntax attempt | Stops at missing generated `logger/generated/logTypes.hpp`; not a successful native app compile. |
| Native build helper | Python syntax and both host guards pass (actual Darwin and simulated instrument hostname rejected before any build). Linux compile/link path pending environment. |

The historical bit-check failures arise only in comparing the clone under this
GCC experiment; the original/restored pair is exact. The optional diagnostic
runs the full 48 matched configurations without stopping on clone drift, while
requiring finite models and exact original/restored results. Its source is
`ddspc_legacy_test.cpp` and it leaves the original exact tests intact. A specific
root cause, Linux reproduction and acceptable operational tolerance have not
been established. Do not generalize Clang compatibility evidence to GNU/Linux.

Classic RLS representative numerical results (QR differs in the last shown
figures, with identical qualitative outcomes):

| F / scenario | Held-out prediction RMSE Legacy / standard / assisted | Frozen numerical feedback residual RMS Legacy / standard / assisted |
| --- | --- | --- |
| 1, informative sensor | 0.286633 / 0.286633 / **0.00596977** | 0.288287 / 0.289549 / **0.144193** |
| 1, zero coupling | 0.00596657 / 0.00596657 / 0.00597431 | 0.00603483 / 0.00605907 / 0.00605627 |
| 1, unrelated sensor | 0.286633 / 0.286633 / 0.287295 | 0.288287 / 0.289549 / 0.289469 |
| 3, informative sensor | 0.405571 / 0.293841 / **0.238611** | 0.289812 / 0.289636 / **0.143372** |
| 3, zero coupling | 0.284608 / 0.00612032 / 0.00612886 | 0.0060612 / 0.00606178 / 0.00606018 |
| 3, unrelated sensor | 0.405571 / 0.293841 / 0.293885 | 0.289812 / 0.289636 / 0.289596 |

In F=1 informative feedback, synthetic applied-command RMS is respectively
0.098314 / 0.0727824 / 0.262306. Greater assisted effort and different penalties
matter; residual improvement is not a matched-effort or physical performance
claim. D-block norm is 0.66894 with useful coupling, 0.00119007 in the D=0 control
and 0.0632672 with unrelated sensors. A finite D estimate alone is not evidence
of usefulness; frozen negative-control prediction is essential. Legacy's F=3
multichannel regression defect explains its incorrect conditional input indexing;
future random sensor innovations explain the remaining assisted F=3 error.

Representative **instrumented** latency, microseconds per record (the whole
frame budget is 500 us, not an allowance for each component):

| Workload / modes | p50 | p99 | max | >500 us / 2000 | Eigen requests + C++ new per record |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy classic core / 2 | 3.375 | 3.625 | 6.125 | 0 | 34 + 0 |
| Standard classic core / 2 | 2.667 | 2.875 | 8.000 | 0 | 24 + 0 |
| Assisted classic core / 2 | 6.042 | 7.458 | 19.250 | 0 | 27 + 0 |
| Builder + pair, classic / 2 | 46.625 | 55.875 | 198.875 | 0 | 111.975 + 1 |
| Builder + pair, QR / 2 | 54.708 | 61.917 | 189.208 | 0 | 103.975 + 1 |
| Legacy classic core / 10 | 43.750 | 54.125 | 81.791 | 0 | 34 + 0 |
| Standard classic core / 10 | 38.834 | 50.125 | 191.667 | 0 | 24 + 0 |
| Assisted classic core / 10 | 57.667 | 76.792 | 249.959 | 0 | 27 + 0 |
| Builder + pair, classic / 10 | **553.667** | 605.042 | 736.292 | **2000** | 112 + 1 |
| Builder + pair, QR / 10 | **619.167** | 671.750 | 811.792 | **2000** | 104 + 1 |

Builder-only medians are 0.167–0.208 us, with six Eigen requests and one C++ new
request (312 C++ bytes) per record. These byte counts omit Eigen matrix storage;
this is not a total-heap/peak-memory report. Managed-pair medians without builder
are 47.542/56.375 us (m=2) and 555.083/616.458 us (m=10), classic/QR. An earlier
run of the same small pair had occasional >500-us scheduler tails (up to
1,541.83 us for the classic pair), so the final run's zero small-case overruns
does not certify worst-case timing. The hot path performs many allocations and
remains unsuitable for a bounded real-time claim. Benchmark execution under
sanitizers checks correctness only; its times are not used for performance.

Reproduction is in `apps/loPredCtrlAcc/tests/README.md`, including strict,
sanitizer, fast, compiler-drift and native build commands. Logs and CSV files
are ignored artifacts under `apps/loPredCtrlAcc/build/phase5-*`; the numerical
and benchmark source is retained in the engineering change. Main benchmark:

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.benchmark -j3
apps/loPredCtrlAcc/build/benchmark/ddspc_benchmark 2000 \
  > apps/loPredCtrlAcc/build/phase5-benchmark.csv \
  2> apps/loPredCtrlAcc/build/phase5-benchmark-environment.log
```

### Remaining test roadmap

The older `testRLS.cpp` and `testPredCtrl.cpp` are retained as historical demos;
they are not substitutes for the new assertion-based numerical tests. The broader
roadmap below remains applicable: its numerical history/RLS/control/baseline cases
and the numerical applied-command portions are now exercised, while decoding,
native timestamped/multirate transport and physical correction remain
unimplemented/unvalidated. Phase 3's explicit timing components and app-header
states are tested independently of native transport/helper lifecycle.

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
the app lifecycle header now receives the Phase 2 documentation/formatting pass.
Standalone tests use the
dedicated Makefile above rather than linking the application library through
`Makefile.one`.

## Implementation Plan

Each stage has a reviewable exit condition. Phases 1–3 are committed by the
user; Phase 4 review changes were already present and uncommitted. The current
**Phase 5 is offline build portability, timing analysis, comparison and
benchmarking**, superseding the old planned acquisition scope. Stop after this
Phase 5 work; native Linux build and GNU QR compatibility remain blockers to its
exit. Live acquisition, operational changes, deployment and new actuation remain
outside this authorization.

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

3. **P2 — Complete within the explicit input contract: application integration,
   independent models, and INDI shadow selection.** The causal numerical API is
   integrated through `ShadowController` and `processShadowRecord()`, with
   independently owned dimension-reduced standard and assisted learners,
   independent training/error/readiness state, strict validation, frozen-origin
   conditional scoring, reference proposals, and tested selection/reset behavior.
   Startup-only shadow execution inhibits the full output helper lifecycle and
   rejects legacy controls. Exit achieved for the offline synthetic input seam
   and actual-app-header INDI/lifecycle tests. **Live training remains blocked by
   absent applied-command/timing evidence**, deliberately leaving readiness false.
   Native Linux compile/link and real-time behavior are not covered by this exit.

4. **P3 — Complete for independently testable components; live integration
   gated.** Source review identifies configured WFS input and candidate DM paths
   but cannot prove deployed endpoint, exposure correspondence or physical
   application. Implemented bounded immutable exposure/held-input pairing,
   explicit clock/generation/calibration/provenance, latency/uncertainty/coverage
   gates, original-counter audit, stale/gap/restart handling, shared records into
   both independent models, protected application integration and INDI input
   diagnostics. Deterministic chronology/manual-history tests and non-actuating
   application tests pass. The native WFS-only callback stays unverified; no
   command monitor or guessed source is connected. Exit achieved only for the
   synthetic/external-assertion component contract. Live evidence, producer
   snapshot/retention adapter and real-time behavior remain unresolved.

5. **P4 — Complete: preserve and audit the original, then compare the clone.**
   Pin and verify the full original source tree from Git; execute its numerical
   and application code independently alongside restored original and current
   Legacy paths. Confirm exact matched-trace outputs and selector isolation;
   characterize preexisting defaults/ring/reset differences, software-order
   synchronization and inherited shared-state risks. Reuse original timing only
   under an explicit requested/delivered input convention; do not weaken Phase 3
   evidence gates. Exit: original source unchanged, deterministic regressions
   and sanitizer checks complete, differences and limitations recorded. No
   production behavior or physical output is changed.

6. **P5 — Offline components implemented; native build/compatibility exit pending.**
   Add matched deterministic comparisons, negative controls and repeatability;
   benchmark real numerical/builder/manager code and hot-path allocation requests.
   Preserve the original app and all production behavior. Add fresh isolated
   Linux build profiles and run all available offline/sanitizer checks. Completed
   locally: comparisons, timing/source review, allocation/latency measurements,
   Clang suites and GCC numerical/harness compilation. Pending: execute real
   Linux compile/link with supported dependencies, investigate the two exact GNU
   clone-QR failures on the supported target, and agree bounded execution and
   physical timing evidence. Do not mark complete or proceed to live training
   solely because deterministic mathematics tests pass.

7. **P6 — Future verified transport, multirate acquisition/replay, and accelSim
   validation through public streams.** First supply the missing WFS/applied-input
   evidence and supported native build, then implement the stable snapshot,
   retained-frame and established-clock adapter. Verify it against independent
   recordings before asserting flags. Under separate approval add typed raw
   UINT16 accelerometer ingestion/calibration, bounded retention, causal feature
   construction and broadband representation comparison. Benchmark the builder,
   model/QR/solve work, lock duration and native transport at target dimensions;
   introduce bounded workers/cadence changes only with evidence and tests.
   No physical correction connection is part of those steps. Begin simulator
   validation with a
   local/isolated simulator configuration that has no DM channel. The user then
   supplies recorded or operator-run optical WFS/DM traces from independently
   approved disturbance injection. Freeze appropriate training/evaluation
   intervals and compare baseline/augmented prediction in shadow mode, including
   multiple frequencies and broadband cases. Do not import simulator internals
   or the concurrently developed frequency-specific controller. Exit: test 11
   demonstrates valid synchronization and predictive improvement on held-out
   measured WFS data, with documented model conditioning and latency.

8. **P7 — Add the approved correction route and commission through the operator.**
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

- **Phase 5 changed files:** new `apps/loPredCtrlAcc/tests/comparison.hpp`,
  `comparison.cpp`, `ddspc_comparison_test.cpp`, `allocation_probe.hpp`,
  `benchmark.cpp`, `Makefile.benchmark`, `legacy_platform.hpp`,
  `native_linux_build.py` and `README.md`; updates to the already-dirty
  `tests/Makefile.offline`, `ddspc_legacy_test.cpp` and this plan. Other Phase 4
  replay adapters/fixtures already in the working tree are retained. All C++
  touched in this phase was clang-formatted using the existing comment-prefix
  override; Python syntax, native host guards, original-source pin verification
  and `git diff --check` are checked. Nothing above Agent Findings and Plan is
  changed.
- **Phase 5 exit remains open:** no Linux environment was supplied, native app
  compile/link is not demonstrated, and two GNU exact QR replay tests fail.
  Preserve these failures as compatibility evidence; do not label the entire
  validation green or infer production compatibility from stream-free doubles.
  A full exact-compatibility decision needs supported Linux compiler/flags/Eigen
  and independent deployment traces. No live testing, deployment, new actuation,
  physical stream access, instrument command, branch change, commit, merge or
  push occurred. Both new models remain strictly non-actuating and Legacy is
  still the startup default.

- **Phase 1 boundary:** corrected mathematics are available only through the
  explicitly selected Causal path, now also used by the Phase 2 shadow manager.
  The app still defaults to Legacy,
  including its known accelerometer decoding, timing, and applied-command
  limitations. The tests do not establish real exposure/actuation delay,
  increment-versus-absolute DM semantics, simultaneous sensor acquisition,
  on-sky improvement, or closed-loop stability. The synthetic held/clipped input
  history is deliberately labeled and is not inferred instrument readback.

- **Numerical/build limits:** the independent tests use `-fno-fast-math`.
  `Make/common.mk` defaults to `-ffast-math`; the new opt-in
  `DDSPC_SHADOW_BUILD=yes` exports strict flags and a startup guard rejects
  fast/finite-only application builds. Rebuild every numerical object and PCH;
  do not reuse stale objects from a legacy build. Native Linux application
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
  These Phase 1 files are unchanged by Phase 2 except the expanded offline
  Makefile; `apps/loPredCtrl/` and native stream interfaces remain untouched.

- **Changed files for Phase 2:** `apps/loPredCtrlAcc/shadow_controller.hpp` and
  `.cpp` (independent model manager, input contract, readiness and frozen-origin
  diagnostics); `apps/loPredCtrlAcc/loPredCtrlAcc.hpp` (startup shadow integration,
  lifecycle inhibition, INDI callbacks/diagnostics and full-file documentation/
  formatting); app `Makefile` (link manager, strict-build option);
  `tests/ddspc_shadow_test.cpp`, `tests/ddspc_app_test.cpp`,
  `tests/app_test_support.hpp`, `tests/Makefile.offline`; and this plan.

- **Remaining architectural limits:** the seam is serialized, synchronous, and
  intentionally disconnected from unverified live data. Two learner states plus
  two empirical regressor windows require quadratic storage; ready-record rank
  QR costs cubic work in each `p`, in addition to RLS, horizon solves and bounded
  `F+1` coefficient/state snapshots. Allocation and work are not real-time safe
  by assertion. Phase 3 adds explicit established-clock
  `advance()/expireShadowInputs()` timeout APIs; no live clock adapter or timer is
  armed. A future adapter must schedule ticks; latest-record validity alone is
  not continuing sensor health after arrivals stop. Native INDI lifecycle,
  supported Linux build/PCH, empirical-rank policy under real noisy/collinear data, saturation,
  application delay and command ownership need validation before deployment.
  Persistence, calibration changes, dynamic dimensions, hot execution-mode
  transitions, output handoff and automatic fallback are intentionally absent.

- **Phase 3 changed files:** `apps/loPredCtrlAcc/causal_input.hpp` and `.cpp`
  (source/timing contract, bounded pairing, audit/provenance and rejection);
  `apps/loPredCtrlAcc/loPredCtrlAcc.hpp` (protected timing session APIs, common
  model invalidation, INDI input diagnostics and native-source gates);
  app `Makefile` and `tests/Makefile.offline` (link/build input builder);
  `tests/causal_input_test_data.hpp` and `tests/ddspc_input_test.cpp` (independent
  synthetic timing/history tests); `tests/ddspc_app_test.cpp` (real-header timing,
  lifecycle/INDI and non-actuation tests); and this plan.

- **Phase 3 limitations and next evidence:** no trustworthy active WFS/applied-DM
  timing source was established, so only the independent component seam is
  connected. An actual reader needs immutable ImageStreamIO snapshot validation,
  original source frame correspondence, retained-frame recovery and explicit
  generation/clock/layout management. Counters and per-interval certificates
  need completeness proof from the recorder, not a boolean assertion guessed
  from `_shape` or `_modevalDMf`. Physical settling/readback remains a distinct
  uncertainty beyond SDK acceptance. New sessions must be independently
  established and must not reuse identities to hide a reset or replay. The
  constructor checks supplied IDs; it does not authenticate external sources.
  Audit records are returned by the builder for replay/recording; the app consumes
  them but adds no production audit logger/file transport in this phase.

- **Supported timing boundary:** the current one-vector-per-frame model accepts
  only one constant held-input certificate covering a full response-aligned
  exposure, with admissible lead time. It rejects multiple commands in an
  exposure and horizons requiring a fixed pipeline prefix. Do not replace this
  with an undocumented average, nearest frame, interpolation, offset, or proposed
  command. Derive any richer model and its control law separately. Memory is
  bounded by record count but deques/matrices/results allocate, and models/solves
  remain synchronous under the model lock; no real-time claim is made.

- **Phase 4 changed files:** only `apps/loPredCtrlAcc/tests/Makefile.offline`,
  new `tests/prepare_legacy.py`, `tests/legacy_replay.hpp`,
  `tests/legacy_original_adapter.cpp`, `tests/legacy_restored_adapter.cpp`,
  `tests/legacy_current_adapter.cpp`, `tests/ddspc_legacy_test.cpp`, and this
  engineering plan. Original/current production controller, app, RLS, timing,
  generic bases and operational configuration are untouched.

- **Compatibility conclusion:** original `loPredCtrl` is preserved exactly in
  source. The clone preserves the operational algorithm and exact outputs for
  the matched tested traces, but is not universally source/default/state-machine
  equivalent. Do not silently fix historical quirks or use the new reference law
  as the working legacy law. Keep known output initialization, startup zero-model,
  prediction/learning coupling, serialization and publication races visible as
  inherited limitations. No tests require an undefined initial value or a data
  race to be reproducible.

- **Timing recommendation:** a shared original WFS callback/order capture is a
  sensible future foundation for both shadows; actual input semantics and
  applied/publication latency remain a separate contract. Requested increments,
  accumulated requests, accepted drives and physical surface histories are
  different observations. Do not force a separate clock solely because the
  shadow learners are independent, and do not label ordinal candidate history
  physically verified just because the original controller worked on sky.
  Native callback/worker contention and operator deployment traces are needed
  before certifying unchanged real-time publication.

- **Scope verification:** the task text above `# Agent Findings and Plan` is
  preserved byte-for-byte. Phase 4 changes tests/build support and this plan only;
  it does not change either controller law, original timing, legacy defaults,
  input safety gates, live `acc000` ingestion, ImageStreamIO interfaces, simulator
  worktrees, native streams or physical correction. No instrument-facing command,
  branch switch, commit, merge or push was performed during that Phase 4 review.
