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
