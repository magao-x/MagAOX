# Task Description
<!-- The first section is filled out by the user 
     
     This is intended as a guide for how to prompt a coding agent to solve a problem.
     
     Make a copy of this template with a new name under agents/plans/<YYYY-MM>. Then
     Fill out each subsection below as needed. Feel free to add additional
     subsections, etc.  

     When complete, commit it on your feature branch and then prompt the agent to review this file.
-->

## Problem Statement
We want to automate the "J-test", which is the alignment of the pupil on `dmncpc`

## Discussion
<!-- Put more expository text here.  Fully describe the expected behavior and rationale. Include references to other parts of the software that serve as examples. Provide a link to an API/SDK manual.   -->
The manual J-test puts the a pattern on dmncpc which is then manually aligned using ttmperi to center it on an image of a pupil.

To automate this, what we can do is register `dmncpc` actuators to `dmtweeter` actuators after the tweeter-wfs auto-alignment procedure has completed.    Here we can poke the ncpc actuators in a pattern, measure their location on a camera (a slightly out-of-pupil image), and use that as the fiducial .  We then poke tweeter actuators and measure their location.  We then reference to a calibrated offset between those two positions and adjust ttmperi to move the tweeter actuators w.r.t. the ncpc actuators.  This will be a closed-loop process.

This will be done on camflowfs to enable doing this with bump mask in and minimizing coronagraph disturbances, but we also want to be able to do it on camsci1.  We'll want calibrations for both.  Here we will start with camflowfs.

Additional details:
- when we do this we are moving the image of the tweeter on the lower bench.  It will also move on camwfs.  We will need to follow this with stagecamlensx/y, though not necessarily automatically.  We do NOT move ttmpupil as part of this procedure.
- stagelosel needs to be in pupil
- stageflowfs will need to be at a calibration-specific point.  Start with 32.0
- there is no analog of the response matrix used to calculate a reference in dmPokeXCorr.  We will need to record a calibration image for each DM to use cross-correlation
- we will need to deal with the presence of the pupil.  Probably by subtracting an un-poked image as a dark. I think that dmPokeCenter has this in it.
- dmPokeXCorr is much more up to date than dmPokeCenter, so we should reference its conventions.
- the actual position of the pupil should not need to be fit (no circle fitting), we're just measuring the two actuator grids and moving the tweeter on the ncpc to a pre-recorded calibration offset
- So the sequence is:
  -- take unpoked pupil image
  -- take ncpc poked image
  -- subtract unpoked from ncpc
  -- cross-corr ncpc with its reference
  -- take tweeter poked image
  -- subtract unpoked from tweeter
  -- cross-corr tweeter with its reference
  -- difference the two
  -- publish different
  -- closedLoopIndi process sends correction to ttmperi

Poke patterns: we will need to create new poke patterns to optimize for use with the bump mask in, etc.  We should build in a reference taking mode that saves a reference image for a pattern when we have the system aligned.  Write to calib following other similar practices, enabling auto-recall, etc.  

Calibration: includes the poke patterns, relevant stage positions, references, and aligned offsets.

## Requirements
<!-- List the requirements that must be met.  A numbered list works best -->
- coordinate taking unpoked, ncpc-poke, and tweeter-poked images on a configurable camera
- process the images w.r.t. a reference for each and produce a delta signal
- check stagelosel and stageflowfs or fwscind/stagebs/stagesci1 for positions to match the calibration
- load the most recent calibration
- verify that calibration and setup match
- automatically select the poke patterns and references for a calibration
- reject measurements that are too large or are low SNR
- references and current camera ROIs should not need to match in size

## Tests and Metrics
<!-- List any useful metrics that should be targeted by tests -->
Should have tests of all calibration management (recording, recalling)

Should test for image processing edge cases (different sizes, pokes not completely in pupil such that only N out of M sensed)


## Scope and Caveats
<!-- Add any needed caveats and scope restrictions -->
For now creating of the patterns is not in scope, just the handling of them.

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
