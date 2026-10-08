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

Planning status: updated on 2026-10-08 with the requested pre-implementation user guide and the user's 5 us timing answer. The user has requested a documentation checkpoint commit and explicitly directed that implementation remain pending. Plan/interface review remains open. The original questions and answers are preserved below as the decision record.

## Task Summary

Create the `dmSyncopation` MagAOXApp, initially configured as instance `dmtweeter_synco`, to copy auxiliary contributions into existing dmcomb input channels at an integer subdivision of the main AO-loop period. The main loop's publication of `dmC` defines cycle zero. Step 0 completes its destination copies before the next genuine `dmC` publication and does not trigger dmcomb. For each later step, copy all mapped destinations without changing their counters, then trigger exactly one combination by incrementing `cnt0` on one designated mapped destination at the scheduled time. This app must never publish, increment counters, or post trigger semaphores on `dmC`.

Use one auxiliary stream per mapped channel, initially `dm01disp07_aux -> dm01disp07` and `dm01disp08_aux -> dm01disp08`, as selected in the follow-up answers. Read the latest complete contribution at every step, including unchanged data. No per-step pixel streams, fresh-command requirement, provider generation commits, or consumption acknowledgements are required. Providers remain responsible for explicitly writing zero when they want to remove their contribution. Faults, disable, and shutdown hold the last complete destination contents; resumption requires synchronization with the main loop.

Accepted follow-up decision (2026-10-08): always copy every mapped contribution at every active step, including step 0 and unchanged commands. Do not compare pixels for equality or skip destination copies based on equality or unchanged source counters. The optional `INTENTIONAL HOLD` branch discussed in chat is excluded. Reusing a last-complete source cache still means copying it; retaining destinations during disable/recovery is separate from active step processing.

At a 2 kHz main rate and the initial 6 kHz cap, select three steps with trigger offsets approximately `0`, `166.667 us`, and `333.333 us`, followed by the next genuine main event at `500 us`. Only the latter two steps increment an auxiliary destination counter. The nominal spacing margin over the 115 us keepout is approximately 51.667 us; this is a timing budget, not measured worst-case headroom.

### Evaluation of the answers and corrected findings

| Decision/evidence | Revision and consequence |
| --- | --- |
| Answers 1 and 3: `dmC` is observation-only; trigger auxiliary updates by one other channel's `cnt0` | Remove the proposed `dmC` consumer-semaphore route entirely. **Correction to the first plan:** under the intended counter-sum combiner behavior, a passive destination copy must leave `cnt0` unchanged. Incrementing every destination during preparation would trigger prematurely. |
| [`AOloopControl_DM_comb.c`](/home/jrmales/Source/milk-xwcl/plugins/cacao-src/AOloopControl_DM/AOloopControl_DM_comb.c), `compute_function()` | The inner loop tests the sum of input-channel `cnt0` values and sums current pixels after a change. This supports the requested single-counter trigger even when pixels are unchanged. Its summation does not check each input's `write` flag or provide an atomic multi-channel snapshot. |
| [`processtools_trigger.c`](/home/jrmales/Source/milk-xwcl/src/CommandLineInterface/processtools_trigger.c) and [`CLIcore_utils.h`](/home/jrmales/Source/milk-xwcl/src/CommandLineInterface/CLIcore/CLIcore_utils.h) | The processinfo wrapper can run immediately or wait on a configured stream. The first plan treated its single-stream wait too broadly. Accept the user's counter-scanning deployment contract; the actual combiner must revisit the counter-sum check promptly after an auxiliary counter change. A wrapper that waits only for main `dmC` events would not support this route. Qualification must check this without changing or triggering `dmC`. |
| Answer 2 and [`dev::dm::processImage()`](../../../libMagAOX/app/dev/dm.hpp) | Interpret keepout as controller receipt/wake through processing completion. The inspected `m_piTimes` measures `m_tf - m_t0`: entry into `processImage()` through `commandDM()` and optional `makeDelta()`, before the separately measured saturation-semaphore work. It is relevant evidence but not a timestamp of the initial semaphore wake or a live completion handshake. The stated 115 us is a low estimate; retain a configurable extra guard. |
| [`bmcCtrl::commandDM()`](../../../apps/bmcCtrl/bmcCtrl.hpp) | `m_outputShape` is published before `BMCSetArray()`, so it still cannot acknowledge completed processing. The approximately 102 us hardware transfer and possible controller pipelining are future scope; the first app budgets the complete current controller path. |
| Answer 3 and [`MVMextractModes.c`](/home/jrmales/Source/milk-xwcl/plugins/milk-extra-src/linalgebra/MVMextractModes.c) | Both the CPU output path and initialized GPU output path use `processinfo_update_output_stream()` after writing the MVM result. Observe that primary publication for phase and predict the next one for step-0 preparation. Do not move cycle zero to combined output or to controller receipt. |
| Answer 4: persistent contributions, one channel per provider mapping | Replace the strict fresh-per-step provider protocol with ordinary ImageStreamIO publication plus scheduler timing information. Sample each source independently, then freeze a complete destination batch. Different providers need not commit together. An old counter is normal, not a missed-provider fault. |
| Answers 5 and 6 | Hold commands until explicitly changed/zeroed by providers, reacquire main synchronization before resuming, one DM per instance, total-rate cap includes step 0, `N=1` supported, and settings editable only while disabled. A maximum step count now bounds scheduling/feedback storage, not the number of auxiliary image slots. |
| Answer 7 and follow-up answers | Instance `dmtweeter_synco`, 6 kHz cap, three-step capacity, and real-time priority 30. The follow-up selects destinations 07/08 and `spare61`, superseding the initial 06/07 mapping and static `spare43` recommendation. Use `CLOCK_ISIO` for the build's ImageStreamIO timestamp domain. |
| Updated [`AGENTS.md`](../../../AGENTS.md) | Agents never run on instrument machines or issue instrument-connected INDI commands; all live qualification is user-run. Favor reuse and concise code; remove `\author` from touched code. Follow the app header-only convention in rule 21 and the revised commit/PR attribution rules. |

The implementation will reuse [`MagAOXApp::threadStart()`](../../../libMagAOX/app/MagAOXApp.hpp), [`dev::shmimMonitor`](../../../libMagAOX/app/dev/shmimMonitor.hpp), [`dev::telemeter`](../../../libMagAOX/app/dev/telemeter.hpp), and existing ImageStreamIO/mxlib stream operations where their behavior matches this contract. [`dmSpeckle`](../../../apps/dmSpeckle/dmSpeckle.hpp), [`sparkleClock`](../../../apps/sparkleClock/sparkleClock.hpp), and [`milkzmqServer.hpp`](/home/jrmales/Source/milkzmq/milkzmqServer.hpp) remain references for worker configuration and timing feedback. `dev::dmPokeWFS` and `dev::dm` introduce workflows/hardware-controller responsibilities that this app does not need.

[`mx::improc::milkImage`](../../../../mxlib/include/improc/milkImage.hpp) already defines passive writes as not incrementing `cnt0`. However, its passive `post()` still posts semaphores; `ImageStreamIO_UpdateIm()` increments the counter and posts all semaphores. Neither is the correct finalization call for this app's counter-free, semaphore-free preparation. Reuse ownership/access conveniences where helpful, with a small explicit passive-copy/final-counter-publication helper rather than modifying dependency behavior.

### Local cpuset and configuration analysis

The original 2026-10-06 analysis read local files only; no setup script was run and no instrument process/task state was queried. Follow-up user answers below supersede its initial deployment recommendation.

| Static source | Finding |
| --- | --- |
| [`rtSetup/RTC/make_cpusets`](../../../rtSetup/RTC/make_cpusets) | CPUs 32-63 are configured as single-thread RT CPUs; sibling hardware threads 96-127 are disabled. `spare43` assigns CPU 43 and the selected `spare61` assigns CPU 61, each with exclusive cpuset membership. |
| Active references in local `/opt/MagAOX/config`, excluding commented text | Referenced spare sets: 35, 36, 42, 44, 50, 51, 52, 53, 59, 60, 61, 62, 63. `spare43` is the only defined spare set with no active local configuration reference. This global config scan was the basis of the earlier recommendation; the user's machine-specific follow-up selects `spare61` because its `acc001-cbuff` assignment is on another computer. |
| [`fpssetup.setval.tweeter-vispyr.conf`](/opt/MagAOX/config/cacao/tweeter-vispyr/fpssetup.setval.tweeter-vispyr.conf) and [`dmtweeter.conf`](/opt/MagAOX/config/dmtweeter.conf) | Combiner: `dm01comb`, CPU 39, priority 49. Controller: `tweeter`, CPU 40, priority 55, 50x50 geometry. Avoid sharing those critical cores with the new worker. The local FPS setup does not establish the combiner's live wrapper trigger mode. |
| [`twAlign-camwfs-wfs.conf`](/opt/MagAOX/config/twAlign-camwfs-wfs.conf), `twRotSpeck.conf` (present at the original inspection, now absent from the local tree), and [`fdpr2_dmtweeter_camsci2.conf`](/opt/MagAOX/config/fdpr2_dmtweeter_camsci2.conf) | Channel 06 appears in alignment configuration and is no longer selected. Channel 07 appeared in rotating-speckle configuration and is also an FDPR control channel; the user confirms `twRotSpeck` is not a current procList process. The selected 07/08 destinations still require exclusive ownership while this app is enabled; configuration presence alone does not establish a live conflict. |

Use the user's selected worker `cpuset=spare61` (CPU 61), priority 30, with no cpuset-setup changes for the first deployment. Dedicated high-rate observers can use the same configured set initially; keep their work bounded and exclude unbounded worker spinning. Slower app/INDI/logging work stays outside the timing core. The user performs any live availability verification before deployment; the agent will not perform that step.

### Clock-domain explanation and proposal

The clock domain identifies which clock produced a timestamp. `CLOCK_REALTIME` is wall time and can be adjusted; `CLOCK_MONOTONIC` measures elapsed host time; `CLOCK_TAI` uses a different epoch offset from realtime. They cannot be mixed without conversion. This is independent of the displayed timezone.

[`processinfo_update_output_stream.c`](/home/jrmales/Source/milk-xwcl/src/CommandLineInterface/processinfo/processinfo_update_output_stream.c) calls `ImageStreamIO_UpdateIm()`, which stamps the primary `writetime` using `CLOCK_ISIO`. The inspected [`ImageStreamIO.h`](/home/jrmales/Source/milk-xwcl/src/ImageStreamIO/ImageStreamIO.h) chooses `CLOCK_TAI` when defined, otherwise realtime, and [`timeutils.h`](/home/jrmales/Source/milk-xwcl/src/CommandLineInterface/timeutils.h) defines `CLOCK_MILK` as `CLOCK_ISIO`. Thus realtime is plausible for a different producer/build, but cannot be assumed for this MVM path.

Use `CLOCK_ISIO`, as selected in the follow-up answer, to match the ImageStreamIO timestamp clock chosen by the build. Convert paired `CLOCK_ISIO`/monotonic reads into bounded offsets and schedule internal waits on monotonic time. Publish deadlines with an explicit clock identifier and optionally equivalent realtime seconds/nanoseconds for providers; detect conversion discontinuities and reacquire lock. The producer and consumer builds must agree on the timestamp domain; the user performs any instrument-side verification. No independent realtime/TAI override is required for the initial interface.

## Key Assumptions

1. One instance owns writes to an explicit list of existing 2-D float32 destination channels of one DM. `dmC` and combined output are observation-only; reject duplicate aliases/inodes and mappings to those streams.
2. The combiner checks the sum of main input-channel counters independently and promptly enough for auxiliary counter increments to cause combinations. No writes or trigger posts on `dmC` are permitted as a workaround for a different wrapper configuration.
3. `N` includes passive step 0; the total cadence is `N * measuredMainRate`. `maxSubframes=3` initially bounds active steps and feedback storage; there remains only one auxiliary pixel stream per mapping. `N=1` is passive-only. A cap below the main rate is infeasible for auxiliary operation.
4. Sources use ordinary ImageStreamIO write/counter publication. Unchanged source contents remain valid indefinitely. Providers can use published deadlines/signals but are not required to send cycle-tagged ready records or wait for acknowledgements.
5. Snapshot consistency is required; freshness is not. On an in-progress/racing source write, bounded sampling uses that mapping's last stable cached command and still copies it into the destination. The other mappings and scheduled combination can still proceed. Age/reuse diagnostics do not disarm the schedule; unchanged data never selects a no-copy path.
6. A valid initial cache is necessary, but a fresh provider command is not. Attach compatible existing auxiliary streams without resetting them. Newly created auxiliary streams should be seeded from stable destination contents to avoid silently zeroing an existing contribution. Missing/incompatible streams or inability to obtain an initial valid baseline prevent arming.
7. Source/provider data and mapped destination contents are retained on disable, fault, and restart. Do not zero, restore an earlier step, replay missed triggers, or invalidate otherwise valid pixel data when a timing generation changes.
8. Cycle zero is primary `dmC` publication; step 0 for cycle `k+1` is prepared during cycle `k`. Its deadline is predicted because observing the next primary publication would be too late. Initially no upstream WFS timing interface is required.
9. Controller keepout is receipt-to-completion, while this app controls combination submission. Guards must cover differential combiner/controller delivery latency and early main arrival. A standalone observer cannot prevent an arbitrarily early future main command from overlapping an already submitted auxiliary command; detection/recovery is included, controller pipelining/gating is future scope.
10. Development and simulated tests run locally. Any live instrument setup, command, inspection, or hardware measurement is performed by the user and reported back, per AGENTS.md rule 0. Local source/config reads are not live qualification.

## Requirements

| Original requirement | Revised behavior |
| --- | --- |
| 1. Configurable keepout | Positive `keepoutUs`, initially 115 us as a low controller-processing estimate; separate configurable guard and path-latency diagnostics. |
| 2. Integer subdivisions | Calculate the largest feasible `N` from measured period, guarded keepout, total-rate cap, and `maxSubframes`; check actual event windows and all gaps including wraparound. Publish the keepout limit and active count. |
| 3. Maximum rate | Positive `maxAuxRateHz`, initially 6000; counts every step including step 0. |
| 4. Rate on INDI | Publish measured main rate, selected cadence, `N`, interval, and confirmed combined-output rate. Distinguish commanded cadence from delivered output and invalidate lock-dependent values during reacquisition. |
| 5. Parallel channels | **Revised by answer 4 and follow-up:** one persistent stream per mapped destination, initially `dm01disp07_aux` and `dm01disp08_aux`; no `_00`, `_01` per-step payload arrays. |
| 6. Passive step 0 | Copy every mapping by the predicted next-primary deadline minus lead. Leave all destination counters unchanged and post no DM/combiner semaphores. |
| 7. Deadline | Configure completion lead/copy budget, publish the prediction and uncertainty, and measure completion slack against the subsequent genuine primary event. |
| 8. Phase tracking | Observe primary `cnt0`/commit time, estimate main period/phase, and detect gaps/reset/replacement. Own auxiliary counter increments cannot count as primary events. |
| 9. Later steps | Snapshot current or cached source contents, copy the entire batch passively, then increment one designated mapped destination's `cnt0` once at the step target, even if pixels are unchanged. |
| 10. Wait tuning | Independent bounded integrators for step-0 completion and later counter increments; clamp/reset/freeze appropriately without moving cycle zero. |
| 11. Provider timing | A coherent shared-memory schedule with step targets, source snapshot cutoffs, completion deadlines, validity/generation, and independent consumer semaphore notifications. No provider readiness/acknowledgement channel. |
| 12. Recovery | Detect scheduler/main timing faults, uncertain copies, missing/extra combinations, and stream replacement; suppress extra counter triggers, retain commands, reset timing feedback, and synchronize before resuming. Unchanged provider data is normal. |

### Configuration and INDI surface

| Group | Interface and policy |
| --- | --- |
| Streams | Primary/combined names, source-to-destination mapping, designated trigger destination (default first mapped channel, initially 07), and `maxSubframes`. Read-only resolved mapping/geometry. Mapping/topology changes require disabled restart. |
| Timing | Keepout, rate cap, guard, step-0 lead, bounded copy/snapshot budgets, and read-only resolved `CLOCK_ISIO` source timestamp domain. Validate finite values and combined feasibility; writable settings only while disabled. No configurable `dmC` write or consumer-trigger route. |
| Tracking/feedback | Expected acquisition-rate range, phase/period bounds, main timeout, good-frame count, integrator gain/clamps, optional bounded final spin. Publish lock, measured rates, count, uncertainty, per-step residuals/corrections. |
| Operation | Start disabled; enable/disable and reset/reacquire requests. Commands are always retained. Re-enable reacquires phase using existing valid source data. |
| Worker | Initial priority 30 and selected `spare61`; expose thread configuration/health using existing MagAOX facilities. |
| Diagnostics | Late snapshot/copy, retained-snapshot use, missed-step0, suppressed/late counter trigger, missing/unexpected output, clock discontinuity, main timing fault, and recovery counters. Source age is informational. Aggregate telemetry and bounded raw timing samples. |

### Initial example configuration decisions

Executable/class `dmSyncopation`; instance `dmtweeter_synco`; primary `dmC` (resolved to `dm01disp03`); combined output `dm01disp`; mapping 07/08 as above; counter-trigger destination 07; maximum cadence 6000 Hz; maximum steps 3; keepout 115 us; priority 30; cpuset `spare61`; source clock `CLOCK_ISIO` from the matching ImageStreamIO build. Geometry is discovered/validated, with the local controller config providing a 50x50 expectation.

For review, propose an initial 20 us extra guard and 20 us step-0 completion lead. These are configurable starting estimates, not measurements or hard guarantees. The copy/snapshot budgets and earliest-main bound must be measured or explicitly configured before operational qualification. At an ideal 500 us period, `K+G=135 us` permits three steps; the rate/count limits also select three. A small rise above 2 kHz can make three steps exceed a strict 6 kHz cap and reduce `N`; do not hide that boundary with a tolerance that violates the cap.

The follow-up answer supplies 5 us as an initial timing value, motivated by typical semaphore-driven jitter below 2 us RMS. The user guide currently interprets this as `scheduler.maxTimingErrorUs` for primary phase/later-trigger residuals, with the separate 20 us guard and step-0 lead still proposals. This interpretation is explicitly open for review; the 5 us tolerance does not extend a step-0 completion deadline. No numerical miss-rate allowance or qualified worst-case site budgets have been supplied.

## Questions and Points of Clarification

1. **Combiner deployment and permitted trigger route.** Is the deployed dmcomb the inspected processinfo implementation, and which `.procinfo` trigger mode/stream/consumer semaphore does it actually use? May this app post only that consumer's semaphore on `dmC` after an auxiliary batch, without changing `dmC` pixels or metadata? This is the proposed minimal route for a single-stream combiner. If that is not acceptable, a dedicated merged-trigger interface will need cooperation outside the new app. Posting an arbitrary mapped channel is sufficient only if the deployed combiner actually listens to it.

Answer: this app should not write to or trigger dmC.  We can trigger dmcomb on one of the other channels by updating its `cnt0`.  So for step `j > 0` we load all of the commands for that step, and then at the correct time we trigger by updating `cnt0` on just one of the channels.

2. **Keepout and timing guarantee.** What events bound the stated 115 us: main-command submission, combined-output publication, controller receipt, or hardware call start/end? What bounds can be assumed for main early arrival and differential main/auxiliary command-path latency? Is conservative scheduling with detection/recovery sufficient, or is a hard no-overlap guarantee required even when the main loop arrives unexpectedly early? The latter cannot be provided by this standalone app without controlling/coordinating the main trigger path.

Answer: It is the time from when the semaphore wakes the dm controller (bmcCtrl) to when the all processing is complete.  It's what is tracked by `m_piTimes` in dev::dm and bmcCtrl.  It is a low estimate of when the DM controller will be ready to be triggered again.  The actual actuator command is only 102 us limited by the hardware, so under future scope we could consider pipelining the auxiliary processing in the dm controllers.

3. **Phase/deadline source and acceptance metrics.** Should cycle zero refer to `dmC` submission or combined-output/controller timing? Is a bounded upstream WFS/loop signal available for the next step-0 deadline, or should the app predict the next `dmC` commit? What step-0 lead, worst-case timing guard, and acceptable mean/p99/worst-case jitter and missed-deadline rate should be used? At 2 kHz/8 kHz, all spacing uncertainty beyond the 115 us keepout has only a 10 us budget.

Answer: Cycle zero should refer to the time when dmC is triggered by the loop (it is normally triggered by the process defined by /home/jrmales/Source/milk-xwcl/plugins/milk-extra-src/linalgebra/MVMextractModes.c after it completes the matrix-vector-multiply to calculate the DM command).  This is the time by which this new app must have finished passively loading commands such that they are picked up by the subsequent summation in dmcomb.

4. **Provider interface and freshness.** Is the proposed shared-memory schedule + semaphore notification + explicit generation/cycle commit/acknowledgement interface acceptable? Should every mapped channel supply a fresh command at every active step, or may a provider explicitly commit a reuse of its last command? Recommendation: require an explicit commit for each required mapping/step, including explicit reuse; require complete batches and skip a step when any required contribution misses its cutoff. Do not infer freshness from unchanged pixels.

Answer: the current semantics of dmcomb are that anything in one of the commands is summed and added, regardless of freshness.  If a provider/writer wants to zero it, it is responsible for doing so.  We should apply the same practice.  The only choice to make is whether we re-write unchanged data or not -- unchanged data has not actual impace on what gets put on the DM. It's probably cleanest to always write what is on the aux channels regardless.

Note: this also leads to a revision to my plan above.  There's no 0th order reason to have dm01dispXX_aux_00 and dm01dispXX_aux_01 -- we're going to write what is on each no matter what.  Deeper justification might involve deconfliction or something.  But it will be simpler for the provider to have a single channel to write to, and the semantics of the write doesn't change from how it currently is.

5. **Fault, disable, and shutdown command policy.** Should mapped destinations retain their last committed values, or be passively restored to the current cycle's step-0 values or zeroed at a safe main boundary? Recommendation: suppress auxiliary triggers immediately, retain the last complete batch initially, and resume only after fresh step-0 staging and lock. Holding a nonzero accelerometer correction may be scientifically undesirable; clearing it is also a command change, so the intended policy needs approval. Decide whether provider misses merely skip a step or disarm the schedule after a configured threshold.

Answer: I think this is addressed under 4 above.  we should hold the last command until explicitly zeroed.  We should restart only after synchronizing with the main loop.

6. **Rate/count semantics and scope.** Confirm that the maximum auxiliary rate includes passive step 0, that `N=1` is a supported passive-only mode, and that a stable configurable slot capacity may bound active `N`. Is one DM per app instance sufficient? Are keepout/rate/deadline settings editable only while disabled acceptable for the first implementation? Recommendation: use these semantics, publish both the unrestricted keepout limit and the capacity-limited active count, and require a disabled restart to change mapping/capacity.

Answer: I accept your recommendation.

7. **Initial deployment parameters.** Which destination channels, allocated capacity, stream prefix/app instance name, source clock domain, real-time priority, and cpuset should the initial example configuration use? Recommendation: document a non-operational example with channels such as `dm01disp06` and `dm01disp07`, four allocated steps, 8 kHz cap, and 115 us keepout; do not label it ready for hardware until the timing margins and trigger route are established.

Answer: 06 and 07 are fine.  Let's go with 6 kHz, so 3 steps, to start with to give some margin.  Let's callthis `dmtweeter_synco`.  For cpusets please analyze rtSetup/RTC/make_cpusets against /opt/MagAOX/config to identify which cpusets are available.  R/T priority 30 to start.  Not sure what clock domain means, but I suspect the answer is CLOCK_REALTIME.

### Evaluation outcome and remaining review items

The seven answers are accepted and incorporated above. The four follow-up review items and subsequent answers are preserved below as the decision record. Items 2 and 3 are resolved by the user's answers (`CLOCK_ISIO`, channels 07/08, `spare61`); item 1 supplies an initial 5 us timing value whose exact application is described for review in the user guide. Worst-case site bounds, numerical miss-rate criteria, and snapshot fallback in item 4 remain open. Always-copy behavior is settled by the 2026-10-08 chat instruction. Earlier questions about provider freshness, `dmC` triggering, and fault zeroing do not need to be answered again:

1. Review the proposed guard/step-0 lead and set acceptable timing-error/miss thresholds. The keepout definition is resolved, but actual worst-case processing/path/early-arrival bounds are not established by the nominal 115 us value.

Answer: We are typically able to maintain less than 2 us rms jitter on semaphore-driven events.  So I think using 5 us as an initial value for timing  is conservative.

2. Review the `ImageStreamIO` source-clock default and converted monotonic scheduling. Realtime is supported when it matches the actual producer; it must not be selected solely because other MagAOX streams use it.

Answer: this is a bit of a mess because ImageStreamIO uses TAI, but on the O/S we run on this is not defined so it falls back to REALTIME.  We can use "CLOCK_ISIO" to be compliant with whatever is going on in the build in question.

3. Confirm the static recommendation `spare43` and exclusive ownership of destinations 06/07. Live availability, combiner counter-scanning responsiveness, and producer clock/build are user-run deployment checks, not agent actions.

Answer: We can move to 7 and 8.  "twRotSpeck" is not a current process in procList. We should use spare61. It is in use on a different computer for acc001-cbuff, but not on the relevant machine.

4. Review last-stable-cache fallback for a source being written at its snapshot cutoff. It preserves persistent-command semantics while reporting an incomplete new write; it does not impose a fresh-per-step handshake.

The user has requested a Markdown user guide before C++ implementation. The guide is created at [apps/dmSyncopation/doc/dmSyncopation.md](../../../apps/dmSyncopation/doc/dmSyncopation.md), with stream/cadence/deadline/decision/recovery diagrams, configuration tables and template, and a proposed INDI interface. Its exact names and unselected defaults are marked as review proposals. The user's plan/interface review remains the gate for C++ implementation.

## Tests

Use the existing Catch2 setup with fake clocks/waits and a recording stream adapter, plus isolated real ImageStreamIO streams and a mock combiner that checks the sum of all main channel counters. Provider simulators are test-only. All INDI tests use local mocks or a disconnected local test server, never an instrument tunnel.

| Test set | Concrete purpose/assertions |
| --- | --- |
| Configuration | Exercise every key/default/required value, mapping parse, aliases/roles, designated counter channel, geometry/type, invalid timing bounds, nonfinite/range errors, timing feasibility, `maxSubframes`, resource bounds, and disabled-only updates. Invalid configuration performs no destination publication and never posts to `dmC`. |
| Count/rate selection | At `T=500 us`, `K=115 us`, `G=20 us`, 6 kHz cap, capacity 3: `N=3`, approximately 166.667 us, two extra triggers. Also test the 8 kHz/four-step case with adequate bounds, 130 us keepout, equality/rounding/wraparound, capacity limiting, passive-only `N=1`, cap below main rate, and strict cap crossing when the main rate rises above 2 kHz. |
| Primary phase/clock | Stable/jittered primary events, delayed observer wakes, extra aux events, counter gaps/resets, offset conversion, clock steps, and source replacement. Only primary publications establish cycle zero; converted commit times drive phase, with bounded acquisition/recovery. |
| INDI | Every property/type/permission/element/unit/default and writable callback/request; wrong device/property/element, finite/range checks, registration failures, current/target updates, disabled-only enforcement, enable/reset/disable state, rates/counters/feedback. Connection loss in the local test environment must not block the timing worker. |
| One-stream source lifecycle | Exactly one auxiliary image per mapping regardless of `N`; compatible existing values retained; new sources seeded from stable destinations; missing/incompatible/replaced streams handled; restart does not zero commands or destroy main streams. |
| Persistent-command sampling | No source changes for many cycles must still produce a destination copy for every mapping at every active step, including step 0, and every scheduled auxiliary combination. An explicit zero is propagated. Independent provider changes are sampled without an all-provider freshness barrier. In-progress/changed-during-copy sources use bounded retry/cache fallback, copy the cached commands, and raise the appropriate diagnostic without interpreting old `cnt0` as a fault. A snapshot completed after the cutoff must not replace that step's frozen command. |
| Provider schedule | Coherent version/clock/generation/validity/step targets, provider completion deadlines and snapshot cutoffs, future step-0 notification, consumer semaphore allocation, update/cancel ordering, and notification backlog handling. Assert that no ready record or consumption acknowledgement is necessary to deliver persistent commands. |
| Passive step 0 | Copy multiple known contributions before the predicted main deadline, including when provider data is unchanged. All mapped destination counters remain unchanged and no DM/combiner semaphore is posted. Polling the mock combiner during preparation must not produce output; the next genuine main counter increment must combine the prepared batch. Distinguish a copy that could not start safely from a copy completed late; both are timing failures, with no intentional-HOLD classification or catch-up triggering. |
| Single-counter trigger | For `j>0`, complete every destination copy before incrementing only designated channel 07 once. No non-trigger counter or `dmC` pixels/counters/timestamps/semaphores change. Repeating identical source data still causes destination copies and one output per step. A wrapper waiting only on primary `dmC` must fail route qualification rather than cause a workaround on `dmC`. |
| Consumption/keepout | Controllable combiner polling and controller delivery/processing delays. Confirm preceding consumption before destination replacement; verify adjacent/wraparound bounds including differential latency and early-main uncertainty. Missing/coalesced/extra outputs lose lock. A combined publication is not a hardware completion acknowledgement. |
| Independent wait integrators | Different fixed biases for step-0 completion and each later counter increment; correct update sign/convergence, zero gain, clamps/anti-windup, early wake gating, interrupted waits, jitter, resets after misses/rate changes. Feedback does not shift primary phase or couple per-step corrections. |
| Fault/disable/recovery | Main timeout/gaps/early events, copy overrun, missing/extra combinations, observer overflow, clock discontinuity, and deleted/replaced streams cancel pending counter increments, retain destination contents, and reacquire phase. Resume with unchanged cached/source commands is allowed; no fresh-provider gate or catch-up bursts. |
| Shutdown/concurrency | Interrupt all waits, disable with staged work, fail an observer/worker, and reject live topology mutation. Prompt worker joins; no further aux counter increments after disable acknowledgement; normal reader-registration cleanup; commands/streams retained according to ownership. |
| Local timing qualification | Synthetic 2 kHz main, two independent persistent sources, three steps/6 kHz, and controlled load. Measure copy/wait/output latency, spacing, misses, and CPU usage. Optional four-step stress testing remains a separate capability test. Development-host timing results do not establish instrument worst-case bounds. |

Metrics: selected/delivered cadence; primary phase error; per-step mean/RMS/p50/p95/p99/max counter-increment residual; source snapshot/copy time; cached-snapshot use; step-0 completion slack against the genuine next primary event; combination latency/minimum output spacing; suppressed/missed/coalesced/extra events; recovery duration and CPU load. Hardware receipt/processing completion metrics use user-provided controller timing evidence, including `m_piTimes`, without relabeling software submission/output times as hardware completion.

Acceptance requires deterministic tests to pass and user-run target-system measurements to support the chosen guard/count. Numerical instrument jitter/miss limits remain a review item. The agent prepares a procedure and analyzes returned results; the user runs every instrument step.

Add deterministic coverage of the proposed 5 us residual limit: later-trigger/primary-phase errors on both sides of the limit, suppression before an overdue trigger, and recovery after an executed out-of-limit trigger. A step-0 completion 1 us after its deadline remains a miss; unchanged provider data still follows the normal copy path. Review and freeze the guide's exact configuration keys, INDI permissions/elements, and default-versus-required-site distinctions before writing the corresponding C++ tests.

## Implementation Plan

### 0. Review user documentation before C++

- Completed the requested draft at `apps/dmSyncopation/doc/dmSyncopation.md`. It describes persistent one-stream contributions, always-copy steps, the single-counter trigger route, provider versus app deadlines, 5 us timing-error interpretation, retention/recovery, proposed configuration keys, and INDI controls/status.
- The guide uses ASCII diagrams readable in both plain text and rendered Markdown. Its configuration template contains selected deployment values and labels unfilled site budgets; it is not an operational configuration yet.
- Resolve the 5 us scope and remaining timing/fallback choices, approve interface names/defaults, and keep the guide and this plan consistent during implementation. No C++ app, tests, build registration, or logger implementation is part of this documentation step.

### 1. Finalize the reviewed contract and minimize new machinery

- Record the remaining timing/fallback decisions from the review items above. Apply the accepted `CLOCK_ISIO`, 07/08, `spare61`, and always-copy choices. Keep the counter-sum route and prohibition on `dmC` publishing explicit in code/test boundaries; no alternate main-stream trigger mode.
- Use existing stream access/ownership, thread lifecycle, circular-buffer/statistics, time utilities, INDI, and telemetry helpers. Keep pure count/phase/feedback methods in the app or one small app-local timing helper; do not introduce a general provider protocol framework or new base classes without a demonstrated reusable need.
- Use `dev::shmimMonitor` for primary and combined observation through distinct existing tag/config sections and `SHMIMMONITOR_*` macros when practical. Use `TELEMETER_*` for telemetry. Only write an app-local observer if existing monitor timing/metadata access proves inadequate; record why before changing common helpers.

### 2. Add the app/configuration and stable mapping

- Add `apps/dmSyncopation/dmSyncopation.hpp`, entrypoint-only `.cpp`, `Makefile`, and a non-operational `dmtweeter_synco` example. Derive from `MagAOXApp<true>` and the applicable helpers. Follow AGENTS.md rule 21: definitions outside class declarations, inline below them in app headers.
- Validate configuration in `loadConfigImpl()` and real stream geometry/roles after attachment. Start disabled, open main channels without publishing to them, and create/attach one auxiliary stream per destination while retaining compatible data.
- Seed new sources from stable destination values rather than implicitly clearing them. Build per-mapping latest-complete caches/staging buffers and per-step feedback storage before arming; `maxSubframes` changes only this schedule capacity. Determine trigger destination from the configured mapped list, default 07.
- Use existing priority/cpuset setup: priority 30, selected `spare61`. Do not edit or execute `make_cpusets`, alter local `/opt/MagAOX/config`, install to an instrument, or query remote processes/INDI as part of agent work.

### 3. Implement phase, count selection, and deadlines

Define `P[k]` as the primary publication time for cycle `k`; `T` as the estimated period; `Tfast` as the bounded shortest allowed period; `K` as controller keepout; `G` as extra differential-path/timing margin; `Fmax` as total cadence cap; and `C` as maximum scheduling steps. Initial bounds are:

```text
N_keepout = floor(Tfast / (K + G))
N_rate    = floor(Fmax * T)       # T in seconds
N_active  = min(N_keepout, N_rate, C)
F_selected = N_active / T
A[k,j] = P[k] + floor(j*T/N_active), j > 0
D[k+1,0] = predicted P[k+1] - step0Lead
```

- Use checked integer nanosecond arithmetic, conservative shortest-gap rounding, and full copy/consumption feasibility tests in addition to these bounds. Reject zero/infeasible counts instead of clamping to one. Check the last-to-next-main gap and step-0 copy window.
- Main `writetime` is a close publication timestamp produced just before `cnt0` increment/semaphore post, not an exact receipt time. Convert its established clock to monotonic time and include timestamp/observer/path uncertainty. Use genuine primary counter progression for lock; auxiliary events cannot reset `P[k]`.
- Preserve the approved phase: later counter triggers target `P[k] + j*T/N`, rather than moving cycle zero or targeting a different combined-publication phase. Measure actual combined output latency separately and include latency variation in the guard.
- Controller receipts occur at `A[k,j] + L[k,j]` for a path delay `L`. Safe receipt spacing depends on changes in `L`, not merely spacing between counter increments. Validate latency bounds/earliest-next-main predictions; observing combined output confirms consumption but does not expose completed controller processing.
- Publish future-cycle step-0 deadlines during acquisition; prepare step 0 for a fresh future main cycle before arming its later steps. Changes in rate/count/bounds invalidate the timing schedule and reacquire phase at a main boundary, while all pixel values remain valid.
- A strict cap takes precedence over hysteresis. At a main rate slightly above 2 kHz, three steps exceed 6 kHz; publish the limiting reason if selection drops to two.

### 4. Provide timing information without freshness handshakes

- Create one app-named timing/control ImageStreamIO stream with a documented typed integer layout: version, session/schedule generation, valid/lock state, primary cycle identifier/reference, clock identifier, period/count, and each active step's provider completion deadline, source snapshot cutoff, destination completion deadline, and combination target. Place the provider completion deadline before the snapshot cutoff by the bounded sampling budget. Use typed counters/times rather than float pixel encodings of 64-bit nanoseconds.
- Publish coherent schedules and notify independent provider consumer semaphores. Notifications indicate that timing information is available; authoritative cycle/generation/state fields handle missed/coalesced notifications. Provide next-cycle step-0 information before its cutoff. INDI carries low-rate summaries, not high-rate provider synchronization.
- Providers write their single source with normal ImageStreamIO semantics and may keep values unchanged across arbitrarily many steps. No separate ready/reject/consumed streams, explicit reuse commit, cycle tag on payloads, or acknowledgement wait.
- In the bounded preparation window ending at the published snapshot cutoff, obtain each source's latest stable snapshot using normal publication ordering, `write`/counter consistency checks, and bounded retry. Account for memory visibility in the deployed ABI; these checks do not make several providers one atomic transaction. If a source is currently being rewritten, use its last stable cache and report retained sampling. Freeze the batch by the cutoff; a snapshot finishing after it cannot replace that step's selected command. Provider publication after that cutoff naturally targets a later sample. Snapshot-consistency counter checks do not authorize skipping destination copies.
- Freeze each full staged destination batch and never let provider activity block the final deadline. A new source value is not a prerequisite for a scheduled combination; an inaccessible/invalid stream is a resource fault, while an unchanged valid stream is ordinary operation.

### 5. Copy passively and publish one destination counter

- Always copy the complete selected batch into every mapped destination at each active step. Do not add a pixel-equality pass or an `INTENTIONAL HOLD` branch, and do not skip destination copies when source counters are unchanged. Selected cached commands take the same copy path as new commands.
- Confirm the preceding combined output has consumed the current destination contents before replacing them. Then copy all mappings in a bounded window close to the next target; do not expose later-step values far in advance of their scheduled trigger.
- For every passive destination copy, set/clear `write` and update appropriate write metadata, but **leave `cnt0` unchanged and post no DM/combiner semaphore**. The helper must not call `milkImage::post()` or `ImageStreamIO_UpdateIm()` for this operation.
- Step 0 ends after its complete batch copy by `D[k+1,0]`. The genuine main publisher alone causes the following combination. If preparation is missed, report the miss and cancel later steps in that cycle; do not generate a substitute main trigger.
- Record step-0 outcomes as `COPIED_ON_TIME`, `NOT_STARTED` when timing/resources prevent starting safely, or `COPIED_LATE` when copying completes after its deadline. The latter two are separate failure diagnostics and cancel that cycle's additional triggers; neither is an intentional no-copy success. No new provider write is normal: copy the previous complete command and evaluate the app's copy timing normally.
- For `j>0`, complete the entire copy, recheck phase/window and deadline, then increment only the designated mapped channel's `cnt0` once at `A[k,j]`. Finalize metadata/order before exposing this trigger counter. Do this even when all source counters/pixels are unchanged; that repeated combination is the required cadence. No DM input semaphore posts are needed for the accepted counter-scanning route.
- Verify expected combined-output progression with bounded latency and at most one pending scheduled combination before destination reuse. Missing/coalesced/extra outputs invalidate lock. A combiner that only wakes for main events fails qualification; no workaround may touch `dmC`.
- Monitor/recheck primary progression before copying/counter publication. An unbounded unexpected main event can still race a multi-channel copy or follow an auxiliary trigger too early; the existing combiner supplies no transaction/gate. Report this limit and recover rather than promise elimination of that race.

### 6. Tune waits and recover while retaining commands

- Keep primary/output observation responsive during worker waits through existing monitor threads and bounded event transfer. Snapshot/source caching and slower log/INDI/health work must not stall final destination copying/counter publication. Preallocate hot-path storage; reject overflow/lost timing observations by disarming.
- For each event, maintain advance `a[j]`; successful residual `e[j] = actual - target` updates `a[j] = clamp(a[j] + gain*e[j])`. Step 0 uses batch completion residual; later steps use final counter increment residual. Wait on absolute monotonic targets minus bounded work estimate/advance, with an optional bounded final wait/spin. An early wake never authorizes an early counter increment.
- Reset/freeze feedback on misses, lock/rate/count changes, or discontinuities; apply anti-windup. Fixed bias can be tuned out, arbitrary scheduler jitter cannot. Do not move primary phase or let one step's correction change another's state.
- Use `DISABLED -> ACQUIRING -> LOCKED`, with recovery on timing loss and resource/configuration fault reporting. Separate requested enable from actually armed state. On loss, invalidate the timing schedule, cancel additional counter increments, reset feedback, and hold every complete destination value.
- Reacquire a configured run of genuine main events, prepare a future step 0 using existing source/cache data, then resume. No provider freshness gate. Disable/shutdown cancels and joins work while retaining contributions; an explicit provider zero is the way to clear them during normal operation. If disabled, the app does not copy later provider zero writes until re-enabled; document that operational consequence.
- Publish count/rate/lock, retained-sampling and timing/resource faults on INDI. Add aggregate `telem_dmsyncopation` through existing logger schema/code generation, with bounded raw diagnostics for qualification rather than a log entry every subframe.

### 7. Verify and document locally; hand off instrument steps

- Build the new app and focused test suite using repository makefiles; exercise the deterministic and isolated-stream tests above and affected logger/build checks. Local simulated counters/semaphores/INDI must use unique stream names and cannot be attached to instrument data/tunnels.
- Prepare an operator procedure for the user to verify live cpuset occupancy, destinations' exclusive ownership, counter-sum responsiveness without `dmC` intervention, source timestamp clock, controller processing/path timing, and step-0/auxiliary jitter on the actual RTC. The user executes all instrument commands/steps and returns results; the agent analyzes them locally. Future controller pipelining is a separate task.
- Document persistent source semantics, cache fallback, counter-free preparation, one-counter triggers, stream ownership, deadlines/clock conversion, rate-count limits, retention/disabled-zero behavior, and recovery. Keep this plan current with accepted bounds and returned measurements.
- Apply the updated AGENTS.md full-file documentation pass to changed code: `\file`/`\brief`, **no `\author` and remove it from touched files**, declaration/inline parameter summaries, member roles, named Data sections, out-of-class definitions, and mutex-scope annotations. Run clang-format on touched C++ files; review Markdown as Markdown.
- Application tests use `libXWCTest::dmSyncopationTest`, `application_unit_test`, `tests/testXWC.hpp`, documented cases, protected Doxygen references, and hidden harness helpers as required. Use `SHMIMMONITOR_*`/`TELEMETER_*` macros where their helpers are integrated.
- Stay on `jrmales/dm-syncopation`. Separate functional/engineering-record, documentation-only, and formatting-only changes; use brief commit purposes plus the required `Co-authored by  <model-name>` line. The user has requested a documentation-only checkpoint containing this plan and the user guide, with no implementation. If a PR is later requested, provide its actual-model attribution with this planning-file path, a copyable Markdown description, and a separate copyable title.

### Expected affected files after implementation approval

- Already created for pre-implementation review: `apps/dmSyncopation/doc/dmSyncopation.md`. Keep this user guide aligned with the subsequently implemented configuration and INDI surface.
- New app: `apps/dmSyncopation/dmSyncopation.hpp`, `dmSyncopation.cpp`, `Makefile`, any justified small timing helper, a `dmtweeter_synco` example, and provider/operator documentation.
- Tests/build: `apps/dmSyncopation/tests/dmSyncopation_test.cpp`, optional isolated-stream suite, `tests/tests.list`, `tests/groups.dox`, and root `Makefile` RTC/buildable-app registration.
- Telemetry: `libMagAOX/logger/types/telem_dmsyncopation.hpp`, schema, `logger/logCodes.dat`, and `libMagAOX/Makefile`; generated files through existing rules.
- Engineering record: this plan. No external milk/provider/controller/cpuset changes or local `/opt/MagAOX/config` edits are currently planned. Actual deployment config registration is a separately prepared user-applied handoff.

## Follow-up and Edge Cases

- **Counter-scanning versus wrapper wait:** accept the requested all-channel counter behavior, but verify the actual process revisits that check promptly. Static local FPS configuration is not running-state evidence. The first plan's main-stream semaphore route and counter-changing passive-copy convention are superseded.
- **Repeated values are valid:** no provider update, old source timestamp, or unchanged pixels cause a freshness fault. Always sample/rewrite valid cached/current values and issue scheduled later-step counter increments. Loss of a source resource is different from its provider being idle.
- **Always-copy policy:** unchanged/current/cached source data is copied at every active step. No pixel comparison or intentional no-copy optimization is planned. Timing/resource suppression remains a failure outcome; destination retention during disable/recovery remains the approved command policy.
- **Single-stream sampling tradeoff:** values published after a snapshot cutoff affect the next sample; a racing publication may cause a previous complete cached sample to be copied. Publish/report this explicitly. Per-step arrays or double-buffering would be future scope only if measurements show a need for advance commands or stronger read/write isolation.
- **Retention:** timing generations invalidate schedules, not pixel data. Never automatically zero nonzero contributions. While disabled, later provider writes (including zero) stay on the aux stream until copying resumes; clearing an already applied destination while disabled needs an explicit user operation or a separately reviewed feature.
- **Controller timing:** 115 us is a low full-processing estimate; `m_piTimes` and combined-output times have different endpoints. Budget the remaining wake/path/saturation overhead and variability; 102 us controller pipelining is future scope.
- **6 kHz margin and strict cap:** three steps at 2 kHz provide approximately 51.667 us nominal spacing margin. Guards/copy consumption and main-rate drift can still reduce the active count. Reacquire on schedule change without discarding commands.
- **Unexpected main/other writers:** multi-channel destination copies are not atomic to dmcomb. Early/unrelated triggers can observe partial preparation or violate keepout after an aux submission. Require exclusive destination ownership, bounded normal timing, detection/recovery, and user-run qualification; hard gating would require a separate scope change.
- **Static cpuset evidence:** the global config scan identified unreferenced `spare43`, but the user's machine-specific answer selects `spare61`; its referenced `acc001-cbuff` runs on another computer. Do not infer same-machine occupancy from the global config tree. Some local configs also reference `spare15`/`spare34`, which this RTC setup does not define; those pre-existing mismatches are outside this task.
- **Source clocks and restarts:** convert the actual primary publication clock, handle discontinuities/counter resets/replaced streams, and announce fresh schedule session/generation identifiers. Compatible retained source data remains usable after restart.
- **Current change and review gate:** added `apps/dmSyncopation/doc/dmSyncopation.md` and updated this document's Agent Findings and Plan; original task text and user answers remain intact. No C++, AGENTS.md, external source, deployment configuration, instrument command, or hardware step has been changed/executed. Review the plan and proposed user interface before implementation.
