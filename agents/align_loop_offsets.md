# Task Description
<!-- The first section is filled out by the user 
     
     This is intended as a guide for how to prompt a coding agent to solve a problem.
     
     Make a copy of this template with a new name under agents/plans/<YYYY-MM>. Then
     Fill out each subsection below as needed. Feel free to add additional
     subsections, etc.  

     When complete, commit it on your feature branch and then prompt the agent to review this file.
-->

## Problem Statement
<!-- A brief statement of the problem -->
Need to add offsets to align loop.

## Discussion
<!-- Put more expository text here.  Fully describe the expected behavior and rationale. Include references to other parts of the software that serve as examples. Provide a link to an API/SDK manual.   -->

When we switch to unmodulated operation the pupilFit algorithm has a bias due to the ellipticity of the pupils.  We need to add offsets so the alignLoop drives to a different position.

The offsets should be INDI properties with target & current, one per axis.  The reported deltas should include the offsets, i.e. 0 should mean that we are on target.

If we when modulating we have (0,0) from pupilFit, and we switch to unmodulated and see (0.5,-0.2), we should enter (0.5,-0.2) in the offsets and see deltas of (0,0) in alignLoop with no stage motion.

## Requirements
<!-- List the requirements that must be met.  A numbered list works best -->

## Tests and Metrics
<!-- List any useful metrics that should be targeted by tests -->

## Scope and Caveats
<!-- Add any needed caveats and scope restrictions -->

# Instructions to Agent
<!-- Specific instructions for the agent.  Below is our standard, but you can modify it as needed. -->

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan
<!-- This section will be filled out by the agent -->

## Task Summary
<!-- The agent should summarize the task as they understand it -->

Give `alignLoop` a two-axis measurement setpoint for compensating the unmodulated pupil-fit bias. Compute
`residual = measurement - offset` on each axis. Feed that same residual into the existing interaction matrix
and publish it as `deltas.delta0` and `deltas.delta1`. A measurement and offset of `(+0.5, -0.2)` produce
`(0, 0)` and no stage command from that frame.

## Key Assumptions
<!-- The agent should list any assumptions they have made -->

- Axis 0 is the first stream sample (`dx` for `pupilFit`) and axis 1 is the second (`dy`). The offsets use
  measurement units, which are pixels for the pupil loop.
  [pupilFit stream writer](../apps/pupilFit/pupilFit.hpp#L1764)
- Proposed INDI names are `offset0` and `offset1`, matching the existing delta indices. The neighboring
  `closedLoopIndi` app exposes an analogous setpoint as `reference0` and `reference1`.
  [closedLoopIndi property setup](../apps/closedLoopIndi/closedLoopIndi.hpp#L479)
- Offsets are runtime settings. They start at zero on process launch and hold their values across loop
  open/close and input stream reconnects. An accepted write updates that property's `target` and `current`
  together, and the monitor uses the accepted value on the next frame it samples.

## Requirements
<!-- The agent should list the requirements to which they are planning -->

1. Register two read/write INDI number properties with `current` and `target`, supporting signed fractional
   values and initializing both elements to zero.
2. Subtract the offsets before the interaction matrix and publish both corrected residuals under the existing
   `deltas` property, including while the loop is open or waiting for controller readback.
3. Keep the existing gain, matrix and command sign for nonzero residuals. Suppress stage commands when both
   residuals are exactly zero.
4. Reject invalid offset requests before changing the active offset or either INDI element.
5. Accept offset writes in either loop state, including while the alignment loop is actively closed.

## Questions and Points of Clarification
<!-- The agent should list any open issues requiring user clarification -->

- Resolved. The user confirmed that offsets must be editable with the alignment loop closed.
  Each write applies to one axis. A frame between writes can produce a correction from the partially
  updated offset pair.

## Tests
<!-- The agent should list and describe the test it plans to implement.  It should be specific about the purpose and goal of the test. -->

- Extend the existing Catch2 suite with synthetic two-float frames and a workstation-local INDI fixture.
  Exercise the real offset callbacks and frame processing. Capture outgoing property updates and stage
  commands through the fixture.
- Check zero defaults and the original calculation for a nonzero frame with zero offsets. Verify that a signed
  fractional target updates only its axis and reports matching `current` and `target`. Check that malformed,
  non-finite and missing values leave the previous offset intact.
- Set the example offsets while the loop is closed. Verify that the callbacks send no stage targets and the
  next matching frame publishes two zeros without a stage target.
  With offsets `(+0.5, -0.2)`, feed `(+0.75, -0.30)` and check that the published residual is approximately
  `(+0.25, -0.10)` and that the captured stage targets use that residual with the existing matrix and gains.
  Check that the open loop publishes corrected deltas without controller readback or stage output.
- Build the app and run only the `alignLoop` Catch2 suite via `tests/Makefile.one`.

## Implementation Plan
<!-- Here the agent documents its plan for the implementation -->

1. Add zero-initialized offset state, standard INDI number properties, and the usual callback wrappers in
   `apps/alignLoop/alignLoop.hpp`. Share validation and application logic between axes. Use the existing
   numeric property helpers and validate the complete input value before publishing it. Reuse
   `mx::math::isFinite` for the optimized build.
2. Use the app's INDI mutex to serialize loop state, accepted offsets, gain and controller readbacks with
   frame processing. Subtract the offsets from the app's private measurement copy and use it for both
   reported deltas and the existing matrix multiplication.
3. Publish the deltas before the controller readiness check and gate stage output on whether the loop is
   closed and the corrected residual is nonzero. Serialize the stage send with opening the loop so any
   send already in progress precedes the open acknowledgement.
4. Extend `apps/alignLoop/tests/alignLoop_test.cpp` and run the targeted checks. Follow the local header-only
   app pattern, update full-file Doxygen coverage, and run `clang-format` on the touched C++ files. The INDI
   declaration macros hide callback parameters, so document those parameters alongside each macro invocation.

## Follow-up and Edge Cases
<!-- The agent should list any planned follow up and any edge cases that are not addressed -->

- Each axis takes effect independently on the next processed frame. A closed loop can respond to the new
  measurement bias during the modulation switch or to the intermediate offset pair. Restore both offsets to
  zero for the original modulated setpoint.
- Exact zero suppresses a stage command. Residuals from measurement noise still receive the existing
  gain-weighted correction while closed. A matching frame leaves any previously issued stage target in effect.

## Implementation and Verification

- `apps/alignLoop/alignLoop.hpp` registers the independent live setpoints. The numeric range follows the
  neighboring `closedLoopIndi` references at ±10¹⁵ measurement units.
- `apps/alignLoop/tests/alignLoop_test.cpp` uses two synthetic measurements and a local command capture
  driver. The app build and targeted test suite passed with 75 assertions in two test cases.
- The software checks ran offline. The modulation transition and stage response remain operator checks on
  the instrument.
