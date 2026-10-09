# Task Description

Kinetix Hotplug

## Problem Statement
The Teledyne Kinetix cameras (camllowfs and camflowfs) require PCIe hotplug operations after power on.

## Discussion
The Kinetix cameras are read by PCIe expansion cards, and as such are part of the PCIe bus.  This means they need to be on when the computer is booted to be recognized.  After that boot, the cameras can be power cycled but if so they need to be "hotplugged" to be reinitialized.  The vendor provides this script: /opt/pvcam/drivers/in-kernel/pcie/hotplug_pcie.sh.

This is not part of normal MagAO-X power control operations.  See apps/pvcamCtrl for the relevant controller.  To be able to power down the cameras when not in use, we want to have this hotplugging occur as part of the power on logic in pvcamCtrl.

## Requirements
1. pvcamCtrl should handle execution of the needed commands to reinitialized the PCIe device

2. it must do this for one camera without interfering with the other camera. 

3. pvcamCtrl must gracefully handle the case where the camera is not found, which will occur on a reboot if the camera was not powered

4. pvcamCtrl must be at 100% line coverage.  This does not include libMagAOX includes, this is handled elsewhere.

## Tests and Metrics
Add tests following usual MagAO-X practices for apps.  Recently standardized apps include zaberLowLevel and virtualPDU.

Key logic to test is power-off/power-on with and without cameras.  

## Scope and Caveats
Changes to libMagAOX, e.g. stdCamera are in scope.  Test coverage of anything outside apps/pvcamCtrl is out of scope.

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
<!-- The agent should list any planned follow up and any edge cases that are not addressed >
