# Task Description

## Problem Statement

We need a system that enables dm updates within the loop time (i.e. at 8 kHz when the loop is at 2 kHz).  The challenge is that we have to space dm writes due to the time it takes to process a DM command.

## Discussion

The DM is normally triggered by the AO loop, which is itself triggered by the WFS readout semaphore.  The WFS readout triggers the reconstruction, which triggers the filter, which triggers the command calculation.  At the end of this the command is written to shared memory channel dmC (symlink for dm01disp03 in MagAO-X).  That causes dmcomb to trigger a sum to dm01disp, and that sempahore then triggers the DM controller app to actually address the hardware.    

We make use of "passive" writes, where we write to other channels, e.g. dm01disp05, without triggering.  Such commands are then summed with the loop command when the loop triggers.

We can also trigger the DM by such writes.  However, there is a finite time after the loop command during which the DM can not accept new commands.  Likewise any such command must be completed before the next loop trigger.  This limits arbitrary DM writes.  This is currently about 115 us for the 2k tweeter.

Finally, we want to send commands from a separate higher speed loop (i.e. accelerometers) at a uniform cadence.  For instance, an 8 kHz loop on top of the main 2 kHz loop, but synchronized with it.

We need a system that allows this higher speed update at uniform cadence but syncronized with the loop.  For a 2 kHz main loop and an 8 kHz aux loop:

step | 0 |   |   |   | 1 |   |   |   | 2 |   |   |   |
main | x |   |   |   | x |   |   |   | x |   |   |   |
aux  | x | x | x | x | x | x | x | x | x | x | x | x |

where the step number is in terms of the 2kHz loop (i.e. 500 us), and the x denotes DM writes.  The aux loop writes at 0, 0.25, 0.5, 0.75, and then 1, 1.25, 1.5, 1.75, etc.

This likely means a parallel set of "dmcomb" shared memory arrays, for some configurable number of the main dmcomb structure.  A command written to these arrays will be loaded onto the main dmcomb and then it will triggered at the set higher cadence.  We will need to assume that the provider writes early enough, it will be a requirement on the provider to do so but we will need a way to provide it with enough information to know how to do this.

The loop must set the 0 of the cycle, so we need to track this.  For this new app "step 0" will not trigger, rather it will be a passive write which waits for the loop.  

The other (in this case 3) writes are triggered.  To optimize this process we should use integrator controllers on the sleep time as is done in milkzmq.

There are a few dm poking apps under libMagAOX/app/dev.  

See /home/jrmales/Source/milk-xwcl/plugins/cacao-src/AOloopControl_DM/AOloopControl_DM_comb.c for the structure of the DM comb system.

## Requirements

The new syncopation app must provide:

1. configurable DM hardware keepout time

2. automatic calculation of the maximum number of subframes given the keepout time and integer intervals

3. a configurable maximum aux loop rate

4. determination of the actual loop rate from the above, and publication of it on INDI

5. a parallel set of dmcomb channels, configurable in number, with a mapping to the main dmcomb.  E.g. dm01disp06_aux_00, dm01disp06_aux_01 might correspond to the aux loop step 0 and 1 for chanel 6 of dm 1. 

6. a passive write to the DM channels for step 0, which will complete just ahead of the loop trigger. 

7. A method to determine or configure the dealine for the above write.

8. Requirement 6 implies a way to track the loop trigger time and trigger the passive writes from the parallel dmcomb.

9. writes to the DM channels for all other steps (at subframe intervals of the main loop) which are individually passive but the DM is triggered at the end

10. Integrator feedback self-tuning maintenance of sleep times to minimize timing jitter on the sub-frame triggers

11. a method for communicating deadlines to provider applications for when their writes to the parallel channels must be complete by.  

12. detection of and recovery from loss of synchro, missed deadlines, etc.

## Tests and Metrics

1. Test all configuration 

2. Test all INDI interactions

3. Test the ability to passively write by a deadline before the loop update

4. Test that self-tuning integrator control of sleep times works for the multiple distinct updates

5. loss of synchro or missed deadlines is detected and reported and recovered from

## Scope and Caveats

Provider apps are not in scope.  Make no assumptions about their behavior other than they respond to timing information and signals provided by this app to synchronize with it.

# Instructions to Agent

Analyze the above task and create a plan to implement a solution as a MagAOXApp.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

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
