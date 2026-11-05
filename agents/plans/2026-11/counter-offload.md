# Task Description

## Problem Statement
In the current version of t2wOffloader, the tweeter has no knowledge of the offloading command being sent to the woofer. This results in a double-correction of the low-order modes that are being offloaded and results in increased rms wavefront error whenever an offload command is sent. We want to stop this from happening by taking advantage of the tweeter's speed. While the woofer offloads, the tweeter will make the exact opposite shape at the same time, thus eliminating the perturbation. 

## Discussion
Read through the t2wOffloader app. You will see this is the way the offloading currently works: every N frames, the average shape of a certain number M of low-order tweeter modes (in practice, usually 8-20) is projected onto the woofer basis and applied. The tweeter control algorithm does not explicitly take this into effect; instead, it sees the perturbation on the wavefront sensor and corrects it accordingly. This causes a spike in the RMS wavefront error in the low-order modes, which affects system performance - especially coronagraphs. The current t2wOffloader maximum offload speed is 1 kHz, which is 2 frames at the standard 2 kHz camwfs operating speed.

The woofer moves much slower than the tweeter. The disparity between the two has been explored using the app dmTemporalResponse (located on branch ktwitchell/dmTemporalResponse). The results of this exploration, while they are not complete or comprehensive, demonstrate that the tweeter moves with a 10-90 rise time of < 100 us (vendor documentation measured 50 us for a 100nm poke), while the woofer has a 10-90 rise time of around 300 us. You should look through these results in the directory /Users/Katie/Documents/MagAO-X/dm_response_data. We will exploit this to create a "counter-offload" command, where tweeter commands are written to step the tweeter shape through the opposite of the woofer shape at the same time. 

The execution will be in two distinct parts:
    1. addition of code into the t2wOffloader that calculates and publishes the offload command as back-projected onto the tweeter; and 
    2. a new C++ app entitled "counterOffload" that takes care of the writing of the tweeter commands.
For now, we will focus entirely on the first part. When it is time for the second part, a new prompt will be supplied.

Most of the framework for part 1 already exists within the t2wOffloader app, so we do *not* want to overcomplicate things. The solution should be as simple as possible while still fulfilling requirements. To understand how I want this solution implemented, read the dummy file in the path '/Users/Katie/Documents/MagAO-X/simulations/agent_workshop/t2wOffloader.hpp'. Check the diff between this file and the .hpp file in the current version of the t2wOffloader app in MagAOX/apps. There are a few lines that have been added in the dummy version that should explain the approach that should be taken. These lines are non-functional; they exist only to show you the general approach that should be taken. Again, follow the outline provided by this dummy file and implement the simplest solution. Once the tweeter values have been calculated, they should be published to a new shmim of our creating. 

## Requirements
1. t2wOffloader should be modifified to publish a new shmim with the offload pattern as it would appear on the tweeter
2. t2wOffloader should still function exactly as it already does; the innate function of the app should be completely unchanged. Only additions are welcome.
3. the method for doing this shall follow the general outline provided by the dummy .hpp file
4. the new shmim shall publish these values at a rate faster than the 8 kHz rate at which we desire to send the commands to the tweeter.

## Tests and Metrics

## Scope and Caveats
I will emphasize:  t2wOffloader should still function exactly as it already does; the innate function of the app should be completely unchanged. Only additions are welcome.

Code may only be added within the t2wOffloader app (anything in t2wOffloader sub-folder is fair game). Do not plan to or implement any code outside of that directory.

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
<!-- The agent should list any planned follow up and any edge cases that are not addressed -->
