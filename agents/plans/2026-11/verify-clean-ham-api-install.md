# Task Description
Confirm that Hamamatsu API and SDK install on a new machine will not affect the existing libraries.

## Problem Statement
Please examine ``/opt/hamamatsu_new/install_firebird_el9.sh`` and verify that an install on a new machine (using a copy of the ``/opt/hamamatsu_new/`` directory) will not modify any existing libraries that are unrelated to the Hamamatsu API and SDK.
## Discussion
- the purpose of this script is to install the Hamamatsu API and firebird drivers on a RHEL9/Rocky 9 machine that contains the MagAO-X (``/opt/MagAOX``) source code
- this script is meant to replace an existing, vendor-provided script (``install.sh``) that performs an in-place upgrade of gcc and other lower-level libraries
- this script should NOT be performing any in-place upgrades of the lower-level libraries on the machine
- the hamamatsu api and sdk are located within ``/opt/hamamatsu_new/``
- api and sdk documentation can be found in the corresponding ``docs`` file within that directory
## Requirements
1. Executing the install_firebird_el9 script on a new machine running RHEL9 should not affect the pre-existing lower-level libraries that are unrelated to either the Hirebird driver, the Hamamatsu (ie. gcc) on said machine
2. The ``install_firebird_el9.sh`` script should successfully run on a new RHEL9 machine after copying the ``/opt/hamamatsu_new/`` directory over from this machine
## Tests and Metrics
1. Test that an install on a clean RHEL9 machine with a copy of ``/opt/hamamatsu_new/`` runs without errors
2. Test that running the script on a clean RHEL9 machine does not damage the libraries/apps in the existing (``/opt/MagAOX``) source code

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
