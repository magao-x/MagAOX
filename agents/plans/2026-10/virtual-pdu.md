# Task Description
<!-- The first section is filled out by the user 
     
     This is intended as a guide for how to prompt a coding agent to solve a problem.
     
     Make a copy of this template with a new name under agents/plans/<YYYY-MM>. Then
     Fill out each subsection below as needed. Feel free to add additional
     subsections, etc.  

     When complete, commit it on your feature branch and then prompt the agent to review this file.
-->

## Problem Statement
Create a virtual power distribution unit (PDU) that combines channels from other PDUs into a single channel.  

## Discussion
Several devices require two power switches, usually one on a `trippLitePDU` or an `xt1121DCDU` and one on an `acronameUsbHub`.  One has to turn both on or both off to manage the device.  This results in the same device, e.g. `fwtelsim`, showing up in two disparate places on the power controller and otherwise causes operational headaches since the operator has to know and remember.

We will crate a virtual PDU (vPDU), that serves to combine such device power switches into a single switch that operates both.

This is in principle quite simple, as dev::outletController already has multi-outlet semantics, with ordering, timing, etc.  We should be able to derive from that and expose the necessary configuration front end, then manage the INDI client connections to the actual PDUs.

pwrGUI shouldn't need a change, as it's just a configuration change.

To clear a pending issue/to-do item, we will also take this opportunity to make dev::outletController a telemeter, and also add telemetry to 

## Requirements

### Reqs for vPDU:

1. Define and implement a standard MagAO-X configuration language for combining outlets from dev::outletController devices.

2. Implement a standard outletController INDI interface 

3. Must work with MagAO-X FSM power control logic

### Reqs for Telemetry:

1. Define a telemetry schema for outletController
  - handle arbitrary number of outlets
  - use a small integer to represent the states 

2. Define the telemetry type class with full support for messages and FITS headers, etc.

3. Update all existing outletController derived classes to support telemetry.

4. Define telemetry for derived classes as needed.  E.g. for voltage/current/frequency on trippLitePDUs.

5. Define the type class for any such base classes with full support for messages and FITS headers, etc.

## Tests and Metrics

Bring all outletController derived classes up to 100% line coverage.  Do not bother with outletController itself as that is done as separate effort waiting to be merged.

Testing should verify that the vPDU single outlet control works by at least verifying creation of proper INDI outgoing traffic.  It is not required that the behavior of the controlled/slaved devices be tested.

## Scope and Caveats

N/A.

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
