# Task Description
The orcaCtrl app is throwing critical errors during startup

## Problem Statement
When the Hamamatsu camera is powered on, the logs throw critical errors during startup, despite the camera powering on without issue.

## Discussion
- the app throwing the errors is ``/home/jliberman/source/MagAOX-ham/apps/orcaCtrl``
- the logs report critical errors for about 3-seconds at startup before the camera config properly loads
- the issue may be occurring within ``orcaCtrl.hpp``'s ``appStartup()`` or within ``connect()``

## Requirements
- OrcaCtrl app startup should haved a longer wait time after power on before polling the camera status/config. Otherwise, the log falsely reports crit errors. 
- only the powerOn logic of the app should be modified

## Tests and Metrics
- write test to check that camera powers on cleanly without throwing crit errors 
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
