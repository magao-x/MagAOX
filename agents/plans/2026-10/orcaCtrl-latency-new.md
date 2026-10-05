# Task Description
Create a plan to investigate why the Hamamatsu image stream is experiencing high latency. 

## Problem Statement
I should be able to stream images at 2kHz in a 128x128 ROI but it's instead capped at ~780 Hz despite the exposure time being set to its minimum value (7.2e-06s).

## Discussion
- The 780 Hz FPS that I am referencing corresponds to `milk-shmimmon camham` which is the rate at which images are being streamed out from the camera to the computer. 
- The issue might be appearing within the `DCAMWAIT functions` contained in `/opt/hamamatsu_new/hamamatsu_sdk/dcamsdk4/inc/dcamapi4.h`. It's possible that the camera is waiting until the first frame has been written out before reading in the second frame which could be causing the delay. 
- Consult the camera manual `ham_doc.md` as needed.Pages 72 and 73 of the manual contain relevant information on the camera's timing settings.
- The Orca Quest 2 supports a `frame bundling` mode which, in theory, allows the camera to run up to `19841 Hz` with 4 vertical lines in the camera's `Standard Scan` mode. Consider this the stretch goal.

## Requirements
1. When I set ROI to 256x256 with the shortest exposure time, the FPS reported in `milk-shmimmon camham` should report as approx. 1030 +/- 10 Hz.
2. When I set ROI to 128x128 with the shortest exposure time, the FPS reported in `milk-shmimmon camham` should report as approx. 2000 +/- 10 Hz.
3. The FPS reported by the `camham.fps` INDI prop should closely match that which is reported by `milk-shmimmon camham`


## Tests and Metrics
- Test that I can seamlessly switch between different ROIs
- Test that the FPS reported by `milk-shmimmon camham` updates when I switch to a new ROI.
- Test that the value reported by `camham.fps` matches that which is reported by `milk-shmimmon camham` to within 10 Hz.

## Scope and Caveats
None.
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
<!-- The agent should list any planned follow up and any edge cases that are not addressed >
