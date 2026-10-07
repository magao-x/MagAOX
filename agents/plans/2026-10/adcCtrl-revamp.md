# Task Description
The Python app adcCtrl needs to be updated to work more robustly and with a different algorithm. 

## Problem Statement
The adcCtrl app needs a major upgrade. A new algorithm using the method-of-moments method of finding satellite spot pointing angle needs to be implemented. Additionally, the app often crashes and needs to be revised to work more robustly.

## Discussion
The adcCtrl app is a purepyindi app that actively sends commands to stagadc1 and stageadc2 in order to eliminate residual dispersion left over after the primary compensation by the adctrack C++ app. Dispersion amount is estimated using the direction of pointing of satellite spots that are either generated actively by the tweeterSpeck app or by passive actuator print-through on the DM. Read at least the first three sections in the paper at https://arxiv.org/abs/2608.10307. The satellite spot pointing concept will still be used (the offset between the pairs of spots will still be proportional to the dispersion), but a new method for measuring spot angle needs to be implemented. This method, utilizing the method of moments, is already written in the adcCtrl app.py file. The exact syntax of the algorithm can be changed if it has to be, but the algorithm as-written right now has been tested and verified in simulations, so it should not be altered on a fundamental level. 

The current format of the app has the option to set it into one of four states: closed-loop ADC correction, one-shot ADC correction, measure-only (dispersion measurements and potential ADC commands are logged but no correction is sent), and idle. This structure should remain.

Right now, the calibration is performed external to the app. The ADC response is calculated by counter-rotating the prisms by a known amount and calculating the resulting change in satellite spot offset angle, as described in the arXiv paper above. The inverse of the slopes of these lines provides the control matrix. The result is input by the operator using cursesINDI. Suggestions for improving this are welcome.

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
<!-- The agent should summarize the task as it understands it -->

## Key Assumptions
<!-- The agent should list any assumptions it has made -->

## Requirements
<!-- The agent should list the requirements to which it is planning -->

## Questions and Points of Clarification
<!-- The agent should list any open issues requiring user clarification -->

## Tests
<!-- The agent should list and describe the test it plans to implement.  It should be specific about the purpose and goal of the test. -->

## Implementation Plan
<!-- Here the agent documents its plan for the implementation -->

## Follow-up and Edge Cases
<!-- The agent should list any planned follow up and any edge cases that are not addressed >