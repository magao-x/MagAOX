# Task Description
<!-- The first section is filled out by the user 
     
     This is intended as a guide for how to prompt a coding agent to solve a problem.
     
     Make a copy of this template with a new name under agents/plans/<YYYY-MM>. Then
     Fill out each subsection below as needed. Feel free to add additional
     subsections, etc.  

     When complete, commit it on your feature branch and then prompt the agent to review this file.
-->

Update GUI interfaces to camera ROI settings

## Problem Statement
<!-- A brief statement of the problem -->

ROI settings could be more efficient if there was a way to set them relative to the current ROI, and if the rtimv display could be used to update the target ROI.

## Discussion
<!-- Put more expository text here.  Fully describe the expected behavior and rationale. Include references to other parts of the software that serve as examples. Provide a link to an API/SDK manual.   -->

### Relative ROI Setting
Currently ROI target settings must always be entered w.r.t. the full frame, i.e. in absolute coordinates.  But if the current ROI is 64x64, the rtimv GUI shows the center as 31.5,31.5, though the absolute center might be 511.5,511.5.  To move the ROI by 1 pixel, the user must type in 510.5, not 30.5.  This becomes unwieldy for large cameras with arbitrarily centered ROIs. See https://github.com/magao-x/MagAOX/issues/150.  This is not as easy as it sounds.  We need the `current` ROI coords to always reference the full frame.  We also need the ability to send target ROI coords relative to the full frame.  A possible plan:
 - add a "relative target" ROI setting to stdCamera, which sets the target in absolute terms
 - add a checkbox to roiGUI to change from one to the other.  It should default to checked for relative.

### rtimv ROI Setting

We could also enable setting/updating of ROIs from rtimv using the existing plugins.  Currently whenever the target ROI is different from current, a purple box appears on the rtimv display to show the new ROI.  This box is not interactive. A potential dialog is:
 - make box interactive, so user changing size or moving it updates target
 - have a shortcut, maybe `R`, that sets the target ROI using the same rules as a new user box
 - a desired feature is to have the box show a center cross when it is selected, with ROI parameters as is done for user box
 - something like `alt-r` sets the new ROI

To have this work we need to send commands via rtimv.  See https://github.com/magao-x/MagAOX/issues/119, which references https://github.com/jaredmales/rtimv/issues/40.  It might make sense to add a generalized outgoing command interface to rtimv, but it might be the wrong direction.  We need to decide what is the best way to provide a connection from the cameraStatus rtimv plugin to the INDI rtimv plugin.  


## Requirements
<!-- List the requirements that must be met.  A numbered list works best -->

Existing interfaces must be preserved.  We can not change the meaning of target in setting ROI.

## Tests and Metrics
<!-- List any useful metrics that should be targeted by tests -->

Ensure that relative ROI works with even and odd-sized arrays with arbitrarily sized and positioned ROIs.  

## Scope and Caveats
changes to rtimv are in scope, but may not be required.

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
