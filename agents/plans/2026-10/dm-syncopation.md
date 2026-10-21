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

Implementation status: implemented and verified locally; operator-run instrument timing qualification remains pending. The [debugging and optimization record](dm-syncopation-debugging.md) tracks instrument trials, subsequent fixes and verification. This section records the settled design and implementation notes. The [user guide](../../../apps/dmSyncopation/doc/dmSyncopation.md) describes the implemented configuration, timing record and INDI interface.

## Task Summary

### Current contract

- Build `dmSyncopation` as a MagAOXApp, initially instance `dmtweeter_synco`, one DM per instance, RT priority 30 on the selected RTC's `spare61` (CPU 61). Main destinations remain `dm01disp07` and `dm01disp08`; corresponding persistent sources are `dm01disp07_aux` and `dm01disp08_aux`.
- The real main-loop publication of `dmC` defines cycle zero. The app observes that stream but never publishes pixels/counters or posts trigger semaphores on it. It copies the next cycle's step-0 contribution before the main publication. Each later step copies all mapped contributions and publishes exactly one designated destination counter, initially channel 07.
- Always copy every selected contribution at every active step, including unchanged/cached values. There is no pixel-equality scan or successful no-copy/HOLD state. An idle provider's complete command remains valid; providers explicitly zero their contributions.
- **Requested count is fixed by `maxSubframes`, initially 3**, as directed in the review response. At a nominal 2 kHz main rate this gives approximately 6 kHz total cadence, including step 0; `Nreq=1` remains supported as passive-only operation. Automatically calculate/report the maximum feasible count and validate the request; do not toggle 2/3 steps because a noisy measured period straddles 500 us. An infeasible request is reported/unarmed rather than silently changed. Start disabled; timing settings change only while disabled, and mapping/topology changes require a disabled restart.
- Prefer the measured acquisition-rate report `camwfs.fg_timing.acq_fps`, not the requested `fps.current`, as the period reference while it is fresh. Configure its device/property/element independently. If unavailable/stale, use a qualified multi-interval primary window, then the configured nominal rate. Real primary DM-counter progression still anchors phase; camera statistics are not timestamps or instantaneous early-main bounds. This assumes a nominal 1:1 main/camera cadence; disable the camera reference for an intentional divider/non-1:1 cadence. A valid camera update is evidence of property receipt, not continuing acquisition.
- Controller keepout starts at controller receipt and includes command processing and downstream channel rereads. Retain the 115 us estimate and separate initial guard/step-0 lead of 20 us each. Destination replacement must wait for **both dmcomb consumption and the previous controller reader-lifetime bound**. The full window must fit; the initial numbers are not a demonstrated timing guarantee.
- The user's reported less-than-2 us RMS jitter is **at dmcomb output**. The requested 10 us combiner sleep does not establish a 10 us delivered-jitter floor. **5 us is a guideline for analysis**, not a hard scheduling or alarm threshold. A late step may execute when its full window is safe; otherwise skip it. Reacquire after three consecutive original unsafe/missed steps; genuine synchronization faults disarm immediately.
- Retain contributions on ordinary disable/fault. Explicit controller zero/release clears discovered main and auxiliary channels. **The controller is not responsible for writes occurring after that clear**; no provider-pause, clear-epoch or shared quiescence protocol is required. Adapt dmSpeckle as the first local auxiliary provider, generating a 200 Hz alternating tip/tilt command measurable by the WFS. Programmatic dmMode integration and `deltaChannels` policy changes are deferred.

### Review disposition

The six 2026-10-18 answers below settle the review. They approve the cadence/soft-failure policy, select monotonic timing and the first test provider, review defaults/interface simplification, and narrow controller clearing responsibility. Earlier proposals for coordinated provider clearing are superseded.

| Finding | Disposition in this revision |
| --- | --- |
| 1. Rate/count instability | Apply fixed requested count and camera-FPS assistance. Remove `floor(6000*Tmeasured)` as a live count selector. Approved nominal 6 kHz sanity rating uses 1% tolerance and 32-frame persistence for ordinary rate changes; hard safety bounds remain immediate. |
| 2. 5 us reset thrashing | Approved safe-window execution/individual-step suppression and recovery after three consecutive original failures. 5 us is only an analysis guideline. Genuine synchronization faults still disarm immediately; preserve the dmcomb-output jitter endpoint. |
| 3a. Controller rereads | Respect the user's exclusion of delta-policy changes. Strengthen copy-start gating through controller processing, including `makeDelta()`; changing delta-channel membership would not eliminate all rereads anyway. |
| 3b. Zero/release | Discover and zero matching `_aux` sources as well as main channels. The user explicitly excludes responsibility for later writes; do not add provider quiescence, clear acknowledgements or cache-generation protocols. |
| 3c. FDPR | Keep 07/08. The user confirms tweeter FDPR is historical and, if used, runs with control loops off. Require ordinary exclusive destination ownership; do not reopen channel selection. |
| 3d. Other writers | Add the 200 Hz alternating tip/tilt dmSpeckle provider. Defer programmatic dmMode integration; it is rarely used for lab alignment on this DM. Document disable-before-independent-triggering for maintenance. An unexpected combination remains a timing anomaly. |
| 4. Combiner mode | Correct the static finding: cacao configures DELAY mode 4 with a requested 10 us sleep. Counter-only triggering works in that setup; live verification remains operator-run. Qualify latency, coalescing, astrogrid exclusion, and ZPO interactions. |
| 5. Threading | Use one primary `dev::shmimMonitor` RT callback that executes the cycle, with direct combined-counter checks. No separate worker/output-observer threads or event queues. |
| 6. Clock anchoring | Use the selected `CLOCK_MONOTONIC` observation timing for scheduling; image timestamps are metadata/diagnostics. Budget publication-to-observation lag explicitly; feedback cannot discover a constant lag absent from its measurements. |
| 7. Passive posting | Reuse passive `milkImage::post()` for preparation and a normal final post on the trigger destination, conditional on the counter-polling route. Passive destination semaphore notifications are allowed; `dmC` publication remains prohibited. |
| 8. Provider interface | Implement a small versioned integer timing stream with future cutoffs and normal semaphore notifications. Preserve restart identity/validity and coherent records; no ready/consumed handshake. First provider is local dmSpeckle; remote provider transport remains future scope. |
| 9. Defaults/fallback | Implement the reviewed default budgets, one-pass consistency checks and last-complete fallback with a reduced interface. Running maxima are evidence, not proof of worst-case bounds; detailed timing remains diagnostic. |

### Source findings supporting the revision

Internal links are repository-relative. External checkouts use `repository:path` text so the plan does not depend on one developer's absolute filesystem layout.

- `milk-xwcl:plugins/cacao-src/AOloopControl_DM/scripts/cacao-fpslistadd-DMCH2DISP:34-35` sets DELAY mode 4 and `triggerdelay=0.00001`. `milk-xwcl:src/CommandLineInterface/processtools_trigger.c:224-235` implements that delay using `nanosleep`. This establishes the requested polling setup, not an observed controller-latency bound.
- `milk-xwcl:plugins/cacao-src/AOloopControl_DM/AOloopControl_DM_comb.c:1285-1337` checks the input-counter sum; its summation reads input pixels without waiting on their write flags. Astrogrid mode can exclude its channel from the trigger sum, and selected channels can also drive ZPO work.
- [`dev::dm::processImage()`](../../../libMagAOX/app/dev/dm.hpp) calls `commandDM()` and then `makeDelta()`. `makeDelta()` rereads both delta and non-delta inputs. `m_piTimes` spans entry through that work; `m_triggerDelta` measures successive controller callback-entry spacing. Both timing buffers require `XWC_DMTIMINGS`. Combined output confirms dmcomb's read, not the later controller read or hardware completion.
- `mxlib:include/improc/milkImage.hpp:790-816` already supports passive posts that leave `cnt0` unchanged while stamping metadata and posting semaphores. Normal publication increments a counter and posts. Use this existing idiom instead of a new semaphore-free write abstraction.
- [`shmimMonitor`](../../../libMagAOX/app/dev/shmimMonitor.hpp) calls `processImage()` synchronously in its wait thread. It requests semaphore 5 by default, but the allocation can fall back; dmSpeckle's legacy semaphore 9 is not registered. Its existing wait timeout is one second, and no callback occurs on timeout. Account for these limitations explicitly.
- `milk-xwcl:src/ImageStreamIO/ImageStreamIO.h:13-17` selects `CLOCK_ISIO` according to whether `CLOCK_TAI` is defined when the header is parsed. App/library include order can therefore disagree. The review's workstation finding is evidence of this hazard, not a measurement of the RTC's clock offset. Metadata use of `CLOCK_ISIO` does not by itself establish a scheduling-clock contract.
- At review, [`zeroAll()` and release handling](../../../libMagAOX/app/dev/dm.hpp) cleared only numbered main channels; `zeroAll(true)` still incremented their counters. The implementation now separately discovers and clears matching compatible `_aux` images without adding them to exact numbered-channel discovery or changing the existing main notification policy.
- [`frameGrabber::updateINDI()`](../../../libMagAOX/app/dev/frameGrabber.hpp) publishes `fg_timing.acq_fps` as the reciprocal of mean acquisition interval, and `acq_min`/`acq_max`/`acq_jitter` as interval statistics in seconds. The inspected local camwfs configuration uses a roughly two-second timing window. These local acquisition observations do not bound downstream AO calculation, DM publication or callback-latency tails.

## Key Assumptions

1. All agent activity remains on the workstation. The operator runs instrument setup, INDI queries/actions, and timing qualification, then supplies results. No agent is installed or executed on instrument computers.
2. v1 sources follow `<destination>_aux` naming so controller clear/discovery can find every contribution. Arbitrary source aliases require an explicit controller-visible mapping before being supported. Main mappings must resolve uniquely and never target `dmC`, combined output, or timing/control records.
3. Exactly one designated provider writes each auxiliary image. Valid cached data may be arbitrarily old; no freshness requirement or implicit zeroing is introduced. A one-pass inconsistent/in-progress snapshot selects the previous complete cache and still copies it.
4. Synco normally owns mapped destination writes; controller zero/release may explicitly clear main/auxiliary streams. Writes after clearing are the writer's responsibility. Adapted dmSpeckle writes its auxiliary source; independent maintenance triggering requires disabling synco. No programmatic dmMode arbitration is included.
5. The requested count is stable configuration. Measured/camera rates affect scheduling, sanity checks and feasibility, not per-frame integer rounding of the count. A camera-reported average bound cannot replace earliest-main or controller-reader bounds.
6. Conceptual cycle zero remains the real primary publication. Selected monotonic callback observation is an estimate with qualified lag/uncertainty, not an identical timestamp. A queued/stale observation cannot be used to arm a cycle.
7. A nominal 115 us processing figure, a configured budget, and a running maximum are different quantities. Qualify bounds for every reader before overwriting shared destinations. Physical completion is not inferred solely from dmcomb output.
8. Ordinary disable/fault retains contents. Controller zero/release explicitly clears current main/auxiliary contents; writes after that operation may reapply contributions and are outside the controller's clearing responsibility. Do not infer a provider acknowledgement, pause or persistent inhibition contract.

## Requirements

| Original requirement | Current implementation target |
| --- | --- |
| 1. Keepout | Configurable controller-processing keepout plus uncertainty margin; include downstream destination readers. |
| 2. Integer count | Automatically calculate/report feasible maximum; validate fixed requested `maxSubframes`, initially 3. Do not silently switch counts on jitter. |
| 3. Maximum rate | Configurable nominal total-cadence sanity bound, including step 0, with approved 1% tolerance/32-frame persistence. It is not a strict cap on every measured period. |
| 4. Published rate | INDI selected-reference reciprocal period (`mainHz`), measured camera/qualified primary-window rates, reference source/age, acquisition jitter/extrema, requested/active count and planned/delivered cadence. |
| 5. Parallel channels | One persistent auxiliary source per configurable main destination; initial 07/08. No per-step payload arrays. |
| 6. Passive step 0 | Copy every mapping before the next primary deadline; leave counters unchanged. Passive observer semaphore notifications are permitted. |
| 7. Deadline | Distinguish writer cutoff, app preparation target, hard copy deadline and earliest-main bound. Publish the cutoffs providers need. |
| 8. Main reference | Anchor each cycle to genuine primary-counter progression observed in `CLOCK_MONOTONIC`; select a stable period from fresh measured acquisition timing, a qualified primary window or configured nominal rate, with explicit uncertainty bounds. |
| 9. Later steps | Always copy all selected contributions, then publish one designated destination counter at a safe scheduled time. |
| 10. Feedback | Step 0 learns bounded nonnegative copy-attempt wake advance; later steps use counter-publication residuals. Valid wake observations may survive safe command rejection; ordinary miss/budget reacquisition retains learning, while true faults/reconfiguration reset it. Feedback does not identify primary observation bias. |
| 11. Provider timing | Small coherent versioned integer stream, future cutoffs, normal semaphore notifications, restart invalidation and documented host-clock scope. |
| 12. Recovery | Separate analysis residuals, individually skipped unsafe steps, three consecutive original misses and genuine loss of sync; retain commands except explicit controller clears. |

The implementation scope also includes controller `_aux` clearing and dmSpeckle integration, as directed in the review responses. Delta-channel policy, controller hardware pipelining, a general direct-writer arbiter, and remote provider clock/transport implementation remain outside the initial app unless separately agreed.

## Questions and Points of Clarification

### Reviewed decisions and user answers

1. **Cadence sanity policy.** `maxSubframes=3` and camera-FPS assistance are settled. Proposed v1: use the nominal/camera cadence for a 6 kHz sanity rating with an explicit 1% tolerance and 32-frame persistence before declaring a nominal-rate configuration change. Hard safety windows remain immediate. Approve these values or choose a ceiling with margin instead; neither choice may reintroduce per-frame 2/3 switching.

ANSWER: approved

2. **5 us and soft failures.** Proposed v1: 5 us is a quality warning/qualification target only; a safe late step may execute. Skip unsafe individual steps, and reacquire after three consecutive unsafe/missed steps. Primary timeout/reset/early-arrival violations, ambiguous combinations and invalid resources disarm immediately. Confirm the interpretation/thresholds; the user has corrected the jitter endpoint but has not assigned 5 us a specific guard/lead/alarm role.

ANSWER: Proposed v1 approved. 5 us is just a guideline for analysis.

3. **Clock/architecture.** Approve one primary-monitor RT callback with monotonic observation timing and a qualified publication-to-observation allowance. If publication-phase accuracy cannot be met that way, use a separately verified producer timestamp contract; do not silently claim observation time is publication time. A fallback source-time conversion would be a documented scope/design revision.

ANSWER: we should use CLOCK_MONOTONIC

4. **Clearing/direct-writer operation.** Controller discovery/zeroing and dmSpeckle adaptation are settled. Propose coordinated inhibit/quiesce/clear-complete handling for controller zero/release and a disable-before-direct-write rule for dmMode/maintenance tools. Confirm automatic-release coordination and rearm policy; source zeroing alone cannot clear an already staged command.

ANSWER: dmCtrl is only responsible for clearing the main channels and aux. If a writer writes after that is not dmCtrl's problem.  We can defer a programmatic way for dmMode to interface -- it is not used for this dm except rarely in lab alignment tasks

5. **First provider and host.** Propose adapted dmSpeckle on the RTC as the first local consumer of the small timing record. Are accelerometer correction writers also on the RTC, or is an ICC/ACC001 bridge needed? A remote host cannot interpret RTC monotonic nanoseconds as its own clock; queued data/clear handling would also need a transport contract.

ANSWER: a dmSpeckle instance that writes an alternating tip/tilt command to simulate a high-frequency vibration would be perfect.  Make the actual frequency of vibration be 200 Hz so it's measureable by the WFS.

6. **Defaults/interface alignment.** Review the proposed single-pass fallback, default budgets and slim control surface below. The original repository-local Markdown guide remains the requested documentation location; handbook publication is a later documentation task. Update the guide's interfaces and diagrams after these choices, before C++ work.

ANSWER: reviewed.

Channels 07/08, `spare61`, always-copy, observation-only `dmC`, one counter trigger, and retained-command semantics do not need to be answered again.

### Preserved planning discussion

The following records the earlier questions and user answers verbatim. Recommendations inside it may be superseded by the current contract and review disposition above.

<details>
<summary>Original questions and user responses</summary>

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

</details>

<details>
<summary>Follow-up timing, clock, cpuset and snapshot discussion</summary>

1. Review the proposed guard/step-0 lead and set acceptable timing-error/miss thresholds. The keepout definition is resolved, but actual worst-case processing/path/early-arrival bounds are not established by the nominal 115 us value.

Answer: We are typically able to maintain less than 2 us rms jitter on semaphore-driven events.  So I think using 5 us as an initial value for timing  is conservative.

2. Review the `ImageStreamIO` source-clock default and converted monotonic scheduling. Realtime is supported when it matches the actual producer; it must not be selected solely because other MagAOX streams use it.

Answer: this is a bit of a mess because ImageStreamIO uses TAI, but on the O/S we run on this is not defined so it falls back to REALTIME.  We can use "CLOCK_ISIO" to be compliant with whatever is going on in the build in question.

3. Confirm the static recommendation `spare43` and exclusive ownership of destinations 06/07. Live availability, combiner counter-scanning responsiveness, and producer clock/build are user-run deployment checks, not agent actions.

Answer: We can move to 7 and 8.  "twRotSpeck" is not a current process in procList. We should use spare61. It is in use on a different computer for acc001-cbuff, but not on the relevant machine.

4. Review last-stable-cache fallback for a source being written at its snapshot cutoff. It preserves persistent-command semantics while reporting an incomplete new write; it does not impose a fresh-per-step handshake.

</details>

The [independent review](dm-syncopation-plan-review.md) remains the separate review record. All original questions/answers above are preserved verbatim. The user subsequently authorized implementation; that authorization supersedes earlier implementation gates without changing the recorded answers.

## Tests

Register deterministic correctness tests in the existing Catch2 suite. Use fake monotonic clocks, controlled counters/metadata, and bounded copy/wait adapters. Real ImageStreamIO integration tests use unique workstation-local streams and no instrument connection. Keep load/jitter/host-timing qualification outside `tests.list` so CI does not assert workstation scheduler performance.

| Test set | Required evidence |
| --- | --- |
| Count and rate policy | `T=499.98`, `500.00`, and `500.02 us` retain requested `N=3`. Test fresh measured acquisition reference, integer-nanosecond rounding, qualified sequential primary-window fallback, nominal fallback, stale/unchanged camera reports, normal Set updates and base reconnect handling, reference changes, sustained rate changes and infeasible windows. No callback EMA/single-outlier period adoption or silent count reduction. |
| Schedule arithmetic | Conservative nanosecond rounding, adjacent/wraparound spacing, writer cutoffs and step-0 deadline/earliest-main relation, preparation reserve, downstream-reader copy-start bound, and a request that fits counter spacing but cannot fit final preparation. Equal 20 us lead/early-main defaults do not themselves provide a positive early-main margin; enforce the earlier effective cutoff. |
| Always-copy snapshots | Every active mapping is copied at every active step, even if pixels/counters are unchanged. One-pass source checks use the prior complete cache on write/counter mismatch; a stable explicitly zeroed source refreshes that cache normally. |
| Passive/normal publication | Passive posts preserve counters and notify observers; one normal final post increments only the designated destination counter. Main `dmC` publication metadata/pixels/trigger posts remain untouched. |
| Soft versus hard failures | An out-of-5-us residual inside a safe window produces diagnostics without unconditional recovery; an unsafe late step is skipped. Exercise the approved consecutive-failure threshold, successful-step reset, genuine main anomalies and ambiguous/extra output. Do not equate provider silence with a missed app copy. |
| Step-0 classification | Distinguish copied on time, not started, and copied late against the effective hard cutoff, recording nominal lead slack separately. Aim preparation before that cutoff; count/report even a small overrun, cancel affected-cycle extra steps, and recover according to the consecutive-miss policy unless actual main association/read consistency is uncertain. No no-copy success state. |
| Single-thread execution | Primary counter change during waits, before/after copies and before final publication must be noticed within bounded checkpoints. Ignore unchanged-counter queued callbacks; adopt a jump/reset once and reacquire instead of repeatedly processing notifications of the same value. Test silence after callback return, cancellation, source/output replacement, and no post-disable counter publication. Do not assert that checks can undo a main event during memcpy. |
| Clock/observation model | Fake constant observation lag, variable lag and observed-period noise. Verify uncertainty/last-gap protection and demonstrate that wait feedback cannot learn an unobserved publication offset. Source metadata clock changes must not shift a monotonic schedule under the selected model. |
| Same-fault interval diagnostics | Correlate the latched primary counter, callback/publication intervals, signed observation difference, previous callback duration/gap and entry-to-metadata delay. Exercise stable consecutive publication samples, write/counter races, malformed/nonincreasing timestamps, counter gaps and ignored duplicate callbacks. Recovery retains the latch; another real fault clears it. Diagnostics do not alter phase, bounds, gates, provider timing layout or aggregate telemetry schema. |
| Output/reader lifetime | A mock combiner confirms quickly while a mock controller rereads destinations late. No replacement before the full reader-lifetime bound. Test timeout/coalescing/astrogrid-excluded trigger channels and ZPO side effects. |
| Feedback | Independent wait convergence, clamps/anti-windup and zero gain. Step-0 advance stays nonnegative, begins conservatively at its configured maximum and learns copy-attempt start latency; faster-than-budget copying cannot push it later. Valid safely rejected wake attempts may teach latency; faults/cancellation/reader-clamped attempts do not. Ordinary missed-step/budget reacquisition retains learned waits, true faults/reconfiguration reset them. No phase shift or trigger replay. |
| Controller clear/release | Discover matching `_aux` counterparts, handle absent/incompatible sources and cover `zeroAll()`/release. Verify compatible main and auxiliary contents are cleared; malformed optional auxiliary streams report failure without preventing main clearing/cleanup. A later source/destination write is permitted and does not make the clear fail; no provider quiesce/timeout protocol is tested or required. |
| dmSpeckle and other writers | Adapted speckle writes only its configured auxiliary source and follows future cutoffs. Its alternating tip/tilt waveform has a 200 Hz physical vibration period rather than toggling every auxiliary sample. No independent delayed main trigger or source-owner conflict. dmMode integration remains deferred. |
| Configuration/INDI/lifecycle | Validate the reduced guide-defined interface, defaults/overrides, fixed requested count, configurable measured-acquisition subscription, finite/nonnegative optional statistics, FPS-only packets clearing absent statistics, malformed/nonfinite numeric and min-max packet rejection, normal update/reconnect handling and strict expiry, 1:1 cadence assumptions, semaphore assignment/reserved-slot rejection, resource/stream roles, enable/disable/reacquire and shutdown. Ordinary source age is informational. An ordinary disable completes an already-started bounded passive batch without an extra trigger, then acknowledges quiescence; cancellation is not a missed deadline. |

Operator-run qualification must use a controller built with `XWC_DMTIMINGS` to assess `m_piTimes` and `m_triggerDelta` alongside app trigger residuals and dmcomb output. Collect mean/RMS/percentiles/max, minimum controller receipt spacing, last-reader/copy-window evidence, coalesced/skipped events, delivered cadence and recovery duty cycle. The user's existing output-jitter observation is retained, but auxiliary phases and tail behavior need their own measurements. The agent analyzes supplied results locally; it runs no instrument command.

## Implementation Plan

### 0. Apply reviewed specification and track implementation

- The six answers above settle this review stage. Align the repository-local user guide's diagrams, defaults and INDI interface with the implementation; keep this plan focused on decisions, scope and verification rather than duplicate full tables.
- Implement app/timing support, focused tests, dmSpeckle integration and shared controller clearing on the feature branch. Record actual validation results below. Deployment and all instrument actions remain operator-run.

### 1. Use one main-monitor timing thread

- Use `MagAOXApp<true>` plus one primary `dev::shmimMonitor`, using the helper interface macros. Keep the usual app `.cpp` entrypoint and out-of-class inline definitions in `.hpp`; use existing aggregate diagnostics/telemetry facilities as appropriate. Do not inherit a hardware-controller or WFS-poke workflow.
- On enable/reacquisition, first collect clean main progression without extra triggers, prepare a future cycle's step 0 in a valid window, then confirm that cycle's genuine main event and combined output before arming its later steps. A callback cannot start with `j>0` merely because the period estimate became valid.
- Once armed, the primary monitor's synchronous `processImage()` validates a genuine new main counter, establishes the cycle observation, runs later-step waits/copies/triggers, prepares the next step 0, and returns before the next expected main event. Open/inspect the combined output directly; no output-monitor thread, separate timing worker or observer event queue.
- Timestamp monotonic callback entry initially; call it `Pobs`, since it follows semaphore return and monitor bookkeeping. If qualification requires a timestamp at semaphore return, add the smallest justified monitor hook, not a duplicate wait implementation. Keep `writetime` diagnostic unless a producer clock contract is explicitly verified.
- Latch interval-anomaly diagnostics with `faultPrimaryCounter`: compare the callback interval with the primary `writetime` interval in its own clock, and expose signed `faultObserveExtraUs` only with `faultPublishValid`. Sample metadata once, bracketed by write/counter checks; require stable consecutive counters and positive, normalized, representable, increasing timestamps, with no retries. Also retain entry-to-sample delay and the immediately preceding callback body's duration/completion gap, including ignored duplicate invocations. Body duration includes mutex wait and ends under the cycle lock before final unlock/monitor bookkeeping. Ordinary recovery retains the same-fault fields; another real fault clears them. A 500 us publication interval with a 555 us callback interval indicates changing observation lag; a 555 us publication interval indicates a producer-timestamp interval excursion subject to its clock contract. No absolute cross-clock subtraction, phase/bound/gate change or constant pipeline-delay/camera-age inference is permitted; forward wall-clock steps can affect this diagnostic. No configuration, provider timing-record or aggregate telemetry-schema change is required.
- During bounded absolute monotonic waits, recheck enable/cancellation, primary `cnt0`, pending output and deadline state before every destination mutation and final counter publication. Observe an early main during the callback, not only after returning to the semaphore wait. A queued/backlogged observation cannot arm a fresh cycle.
- Keep measured-camera input independent of full-cycle ownership: validate/store a coherent report under a short mailbox mutex, and let the cycle/status path adopt it only with a nonblocking read. Retain the previous coherent report on contention and preserve its original receipt timestamp. The sole INDI input thread must not wait for the full timed cycle just to accept a camera report; the RT cycle must not wait for the camera mailbox.
- Prefer primary semaphore 5, configure it in the app, and verify the actual assigned index. Allocation can silently fall back; reject reserved/unexpected assignments, particularly legacy dmSpeckle's unregistered 9, and release only this reader's registration. Default monitor configuration does not expose this preference, so app-level validation or a narrowly scoped helper extension is required.
- Use ordinary appLogic heartbeat/state publication for silence/resource health without another RT thread. Its update interval bounds status invalidation; providers must also expire old schedules locally. The base's one-second realtime timeout does not provide a subframe callback. Disable/shutdown cancellation must use explicit state/in-flight checks and existing interruption mechanisms, never a synthetic `dmC` post.

### 2. Stabilize count/rate selection

- Let `Nreq=maxSubframes` select the requested integer count, initially 3. Compute `Nsafe` from controller/reader windows and report it. Arm only if the request is feasible. Changing `Nreq` is an explicit disabled configuration action, not the result of a noisy period sample.
- Subscribe to configurable camera device/property/element, default `camwfs.fg_timing.acq_fps`, through normal base subscription/reconnect handling and camera Set updates. Do not issue an app-specific periodic `sendGetPropertySetList(true)` refresh: repeated definition replies reset `getINDI` query timeouts and can keep queries running. Prefer measured acquisition timing only while valid-reply age is below five seconds and its rate is within the configured main-rate range; unchanged values are not an assumed heartbeat. Stale/unavailable reports fall back to the qualified primary window, then configured nominal rate. Reference-source changes or selected-period changes greater than `tracking.earlyMainUs` reset phase/output/feedback acquisition without adding a synchronization fault; `fps.current` is not the default rate input.
- Rate sanity is named `maxCadenceHz` (replacing draft `maxAuxRateHz`). Use the approved nominal 6 kHz rating, 1% tolerance and 32-frame persistence for ordinary rate changes. A transient measured 6000.24 Hz must not switch the requested count. A sustained nominal-rate incompatibility reports/disarms the configuration; it does not silently choose two steps. Safety-bound violations are handled immediately, independent of persistence.
- Select the period independently of single callback residuals: fresh measured camera acquisition first; otherwise a qualified primary window of `tracking.rateFrames` sequential intervals with broad-valid rates and interval range no greater than twice `tracking.earlyMainUs`; otherwise configured nominal rate. The primary-window average uses total elapsed time divided by interval count. Remove the short callback EMA and single-anomaly period adoption. Reject out-of-bound phase observations without turning a delayed callback/counter gap into a longer safe period; real rate/reference changes require reacquisition.

### 3. Specify phase, copy windows and wait feedback

Use distinct symbols: `Nreq` is requested steps; `T` is planned main period; `Kread` is the conservative controller processing/last-reader lifetime; `Lpath` bounds submission-to-controller delivery; `guard` is additional timing reserve; `Bcopy` is whole-batch copy budget; and `lead` is step-0 completion lead. `Kread` must cover `makeDelta()` even though delta-channel policy changes are out of scope. The guard does not replace a qualified delivery bound.

```text
Conceptual cycle reference: genuine main publication Ppub[k]
Observed cycle reference:  Pobs[k] = Ppub[k] + observation lag
Monotonic estimate:        Phat[k] = Pobs[k]  # v1 uses the qualified lag allowance
Later-step target:          A[k,j] = Phat[k] + floor(j*T/Nreq),  j > 0
Nominal step-0 deadline:   D0[k+1] = predicted Phat[k+1] - lead
Effective hard cutoff:     Dcopy = min(D0, earliest plausible main - visibility margin)
Preparation target:        complete before Dcopy with a positive scheduling reserve

Before replacing previous batch:
  combined output must confirm that dmcomb consumed it
  and now >= previous submission + Lpath + Kread + guard

Before a new copy:
  copy start + Bcopy <= applicable hard deadline
  with the entire copy inside the guarded earliest-main window

Before a later-step trigger:
  candidate publication + Lpath + Kread + guard <= earliest plausible next main
```

- v1 uses `Pobs` directly with the configured `tracking.observationLagUs` allowance; it does not subtract a calibrated bias or convert image timestamps. Do not identify `Pobs` with publication or claim per-step feedback discovers an offset absent from its measurements. If the lag/variation budget cannot support the intended phase/keepout window, report infeasibility rather than assume zero uncertainty.
- The reader-hold base time is the actual recorded auxiliary counter publication, not scheduled `A[k,j]`. For the preceding main command, use a conservative upper bound on publication, such as `Pobs` or `Phat` plus its qualified uncertainty. A late submission cannot shorten its controller-reader hold by using the earlier target time.
- Verify the pre-trigger bound again using the recorded actual publication afterward. An overrun beyond the safe window is a synchronization/safety fault, not merely a quality warning, even if the candidate publication passed its precheck.
- Treat dmcomb consumption and controller completion as separate conditions. A first observed combined update can support a conservative remaining-controller hold, but it is not a completion acknowledgement. Include polling delay/overshoot, controller wake and processing tails in the timing evidence without inferring a jitter floor from the requested sleep interval.
- Verify the last-to-main gap and next-step-0 copy window separately from integer spacing. At 2 kHz with three steps, the interval is about 166.667 us. With `Kread=115 us`, `Bcopy=10 us` and `lead=20 us`, only about 21.667 us remains for delivery, guard and an early-completion reserve. Applying the starting 20 us guard leaves about 1.667 us before delivery, visibility margin and preparation reserve are included; the three-step default may therefore be infeasible. The guard before replacement is a conservative extra reserve beyond reader completion and needs qualification. All quantities are illustrative, not qualified bounds. A request can satisfy spacing but fail preparation; report that instead of moving the deadline silently.
- Enforce the effective cutoff `Dcopy` as well as the nominal lead deadline `D0`. The visibility margin is part of the qualified early-main/observation uncertainty budget and conservative timestamp rounding; the preparation target adds a positive scheduling reserve. With the starting 20 us early-main allowance and 20 us lead, `D0` alone coincides with the earliest-main estimate and supplies no positive margin: the actual safe cutoff/target must be earlier, and feasibility must be rechecked for the smaller window. Do not declare the default bundle proven usable at three steps from nominal lead alone, or train a symmetric-error controller to finish exactly at a hard deadline.
- Keep separate bounded wait controllers. Step 0 learns from actual copy-attempt start minus desired start (`preparation target - current copy budget`), with a nonnegative advance seeded to `feedback.maxAdvanceUs` on a real feedback reset. A valid unclamped wake observation can teach the wait even when the copy is safely rejected; it does not become a successful command sample. Completion residual remains diagnostic and does not move the desired start later because actual copying used less than its budget. Later steps keep counter-publication residual feedback. Preserve learned waits on ordinary missed-step/budget reacquisition; reset on true synchronization/resource faults, explicit reacquisition and configuration changes. Use absolute monotonic waits, reader bounds and bounded cancellation checkpoints; keep allocation, logging, INDI and filesystem work off the final copy/publication path.
- The 5 us analysis guideline does not itself produce a scheduling fault or mandatory alarm. Execute a late step if its full copy/trigger window remains safe; otherwise skip that step. A step-0 miss of `Dcopy` cancels that cycle's extra triggers, but does not by itself demand a full reacquisition if main association/read consistency remains known. Approved recovery threshold is three consecutive unsafe/missed steps, reset by a successful step. Count the original timing failures, not additional steps cancelled only as a consequence of that failure; one bad step-0 preparation must not count as three failures just because `Nreq=3`. Genuine main anomalies, uncertainty about which data were combined, unexpected output, or invalid resources disarm immediately. Never replay a skipped trigger.

### 4. Reuse passive publication and keep the provider interface small

- Attach/create one `<destination>_aux` image per mapping, seed new sources from stable destination contents, and retain compatible existing images. Preallocate one-pass snapshot/cache/staging buffers before arming. Inspect write flag/counter, copy once, inspect again; a mismatch selects the previous complete cache. No retry loop or freshness gate.
- Always copy every mapping, use passive `milkImage` publication for preparation, and then one normal post on the designated trigger destination for `j>0`. Passive semaphore notifications keep viewers current without changing the combiner's counter sum. Step 0 does not increment a destination counter. Never publish/post on `dmC`.
- Implement a small `int64` timing stream containing layout version, restart-unique epoch/generation, state, `Nreq`, period, monotonic reference and future per-step snapshot cutoffs. Include enough identification to associate cutoffs with the upcoming cycle. Use one coherent ImageStreamIO record and ordinary semaphore fan-out, not per-provider ready/consumed state.
- Post after each snapshot with the **next usable cutoffs**, including next-cycle step 0. A notification of the just-expired cutoff cannot help a writer meet it. Publish/announce future timing during acquisition before arming; state/generation changes invalidate predictions, not persistent pixel values. Define schedule expiry from main reference/period/timeout so silence cannot leave an apparently valid indefinite schedule.
- A restart-unique epoch may be encoded in a generation identity; a plain counter reset to zero is insufficient. Specify fixed same-host monotonic semantics and stable record reads. Freeze the compact layout together with the first adapted dmSpeckle provider; do not defer restart/clock/validity semantics to guesswork.
- The first provider is a local dmSpeckle instance producing alternating tip/tilt at 200 Hz for WFS measurement. An ICC/ACC001 provider or bridge remains future scope and needs a separate deadline clock mapping/network budget. Do not send monotonic nanoseconds across hosts as though their epochs agree.

### 5. Integrate clearing and the first provider

- Extend `dev::dm` to discover matching `<main-channel>_aux` streams and zero compatible counterparts along with the existing main channels. Cover normal `zeroAll()` and `baseReleaseDM()` without changing delta-channel policy or numbered main-channel discovery.
- Follow the user's latest clearing boundary: dmCtrl clears current main and auxiliary contents; writers are responsible for any later write. Do not implement shared inhibit/clear epochs, provider pause/acknowledgements, bridge draining or cache-generation handshakes. A later synco/provider write is outside the controller's clear operation.
- Adapt dmSpeckle to the small local timing record and a configured auxiliary source. In managed mode it publishes that source without issuing its legacy delayed main-channel trigger. Preserve existing unmanaged modes.
- Add an alternating tip/tilt vibration mode with a physical frequency of 200 Hz, independent of sample cadence, so the WFS can measure it. Configure one provider-owned mapped auxiliary source; initial destinations 07/08 remain the default. Provider and scheduler settings/examples must agree on source ownership and timing-stream name.
- Defer programmatic dmMode integration. For rare independent lab-alignment triggering, document disabling synco first. Unexpected combinations remain timing anomalies; no general direct-writer arbitration layer is included.

### 6. Supply defaults and reduce the interface

These are **reviewed starting defaults**, not measured instrument bounds. They replace seven empty mandatory numeric fields in the old example for local bring-up/qualification; they do not establish an operationally feasible three-step bundle. Validate the effective copy cutoff, reader lifetime and all actual windows, and report an infeasible request without silently lowering the count.

| Setting family | Proposed starting point |
| --- | --- |
| Existing selections | Three requested steps, nominal 6 kHz, 07/08, priority 30, `spare61`, 115 us processing estimate. |
| Guard and step-0 lead | Retain separate 20 us starting values; 5 us is an analysis guideline and does not replace these settings. Qualify the actual copy window. |
| Snapshot/copy budgets | 10 us each; record running maxima plus margin and increase the planning budget if observed work exceeds it. Never shrink bounds solely from a short quiet sample. |
| Main acquisition range | 500-4000 Hz as a broad validation default; camera FPS/nominal fallback narrows the operating reference. |
| Early-main allowance | 20 us starting value; qualify together with observation lag and step-0 lead. Equal lead/early allowance needs an earlier effective copy cutoff and may reduce available preparation time. It is not established by a 2 us RMS statistic. |
| Combination timeout | 50 us initial failure watchdog; it is distinct from the controller delivery/reader bound used to allow copying. |
| Feedback advance | 20 us initial maximum; gain 0.1. Step 0 starts at this nonnegative maximum on a true reset and learns copy-attempt wake latency. Final spin remains off; no unbounded busy loops. |
| Recovery/rate persistence | Approved three consecutive original unsafe/missed steps; 1% nominal cadence tolerance and 32-frame rate-change persistence. |

- Measure source-snapshot, whole-batch copy, response and per-step timing while operating. Exceeding a snapshot/copy budget enlarges that runtime budget to observed duration plus 1 us, invalidates the schedule and requires reacquisition/feasibility validation. A running maximum still does not prove a worst-case bound. Controller delivery/reader lifetime needs explicit operator qualification; defaults do not authorize an unsafe window.
- Keep essential INDI controls: enable/reacquire, requested steps, cadence rating, keepout/guard/lead, plus a compact timing/health summary and counters. Camera-FPS subscription, source/destination identities and active/requested/feasible counts must be visible. Proposed detailed per-step histories/corrections, sample-window lengths and spin controls can be configuration/telemetry-only rather than dozens of INDI properties.
- The aligned guide records the implemented `settings`, `timing`, `health`, `enable` and `reacquire` properties, configuration keys and same-host timing layout. Use **steps** for step 0 plus later steps, `maxSubframes` for requested count and `maxCadenceHz` for the nominal sanity rating. Record capacity separately as the timing layout's fixed 16-step storage bound.

### 7. Verify and package after approval

- Implement the deterministic tests above and a small local-stream integration harness; keep target/host timing qualification in an explicit operator procedure outside normal CI. Use existing logger generation and aggregate telemetry; avoid high-rate per-event log formatting.
- The operator verifies the deployed mode/delay, that the trigger destination is not excluded as an astrogrid channel, that auxiliary inputs are not unintentionally driving ZPO at the auxiliary rate, usable monitor semaphore allocation, controller `XWC_DMTIMINGS`, and receipt/processing/copy-window margins. No live command is executed by an agent.
- Keep the plan's decisions and the guide's interfaces synchronized. Full-file documentation/style checks follow the current AGENTS.md: no `\author`, documented declarations/inline parameters/member roles, out-of-class app definitions, helper macros, mutex-scope annotations, and application test grouping/Doxygen links. Run clang-format on touched C++ files when implementation occurs; Markdown is reviewed as Markdown.
- Stay on the namespaced feature branch. Separate app/integration functional changes with their engineering record, documentation-only updates and formatting-only cleanup when commits are requested. Implementation is authorized; committing/deployment still follow the user's requested scope.

### Affected implementation files and dependencies

- New app and deterministic tests under `apps/dmSyncopation/`, its Markdown user guide, root app-build registration, `tests/tests.list`/application test grouping, and aggregate telemetry/schema registration.
- Shared controller `_aux` clear/discovery changes and tests in [`libMagAOX/app/dev/dm.hpp`](../../../libMagAOX/app/dev/dm.hpp) and its existing test suite. No provider-pause, shared clear epoch or acknowledgement protocol is included.
- dmSpeckle auxiliary-provider adaptation and tests under `apps/dmSpeckle/`. Other direct-writer integrations or remote bridges are separate unless explicitly agreed.
- A minimal monitor hook only if verified necessary for timestamp/cancellation/semaphore requirements; prefer existing synchronous callbacks and macros. No external milk change or cpuset setup change is currently required by this plan.

## Execution record

### 2026-10-18

- Recorded the six reviewed answers and implementation authorization; preserved the original task prefix, every user answer and the independent review record.
- Narrowed controller clearing to current main/auxiliary contents, with no provider pause/acknowledgement protocol; subsequent writes remain the writer's responsibility. Deferred programmatic dmMode integration.
- `dmSyncopation` build passed; its focused suite passed **267 assertions in 10 test cases**, including shared-memory/fake-clock lifecycle, passive-copy outcomes, late-main anomalies, INDI callbacks and telemetry serialization/verification.
- Controller `_aux` discovery/clearing implementation passed a forced fresh build and `dm_test`: **274 assertions in 5 test cases**. Coverage includes absent sources, compatible sources, incompatible dimensions/type/cube, main clearing despite an optional-source failure and release semantics. Scoped formatting/diff checks passed.
- Forced sequential dmSpeckle app build and fresh focused tests passed: **59 assertions in 5 test cases**. Coverage includes the true 200 Hz quadrature waveform, managed source publication/timing, trigger rejection and retention versus explicit zero. Formatting/whitespace checks passed. Sequential rebuilding avoids the existing shared-PCH race observed during parallel app compilation.
- Total focused verification: **600 assertions in 20 test cases**. These are workstation correctness tests, not instrument timing qualification. No instrument actions or commits were performed.
- Documented bounded passive-batch completion on disable, symmetric early/late primary-interval rejection, runtime snapshot/copy-budget enlargement plus reacquisition, strict CPU stream types/shapes/pixel bound and one-second unarmed resource retries.
- Aligned the guide's diagrams, full configuration/INDI tables, v1 monotonic timing layout and managed 200 Hz quadrature tip/tilt provider. Created `apps/dmSyncopation/dmtweeter_synco.conf` with the actual initial bounds; it explicitly reports that three steps are infeasible at 2 kHz until qualified bounds support that request. Preservation, local links and scoped whitespace checks passed.

### 2026-10-19 test checkpoint

- User requested a commit for testing. Package the implementation, tests, example configurations and associated plan/review record together, followed by a documentation-only user-guide commit per AGENTS.md.
- Retain the verified local build/test results above; instrument timing qualification remains operator-run. The conservative example still reports three steps at 2 kHz infeasible until qualified timing bounds support that request.

### Debugging and optimization record

The [debugging and optimization record](dm-syncopation-debugging.md) contains the first operator-run trial, subsequent diagnosis, fixes and local verification. Continue recording future debugging and optimization there; retain settled design decisions in this plan. Operator retesting and instrument timing qualification remain pending.

### Affected file groups

- **Scheduler and shared timing:** `apps/dmSyncopation/{dmSyncopation.hpp,dmSyncopation.cpp,syncoSchedule.hpp,Makefile,dmtweeter_synco.conf,tests/dmSyncopation_test.cpp}` and `libMagAOX/app/dmSyncopationTiming.hpp`.
- **First provider:** `apps/dmSpeckle/{dmSpeckle.hpp,Makefile,dmtweeter_vibration.conf,tests/dmSpeckle_test.cpp}`.
- **Controller clearing:** `libMagAOX/app/dev/dm.hpp` and its `tests/dm_test.hpp`/`dm_test.cpp` harness/cases.
- **Telemetry and registration:** `libMagAOX/logger/types/telem_dmsynco.hpp`, `types/schemas/telem_dmsynco.fbs`, logger code/metadata registration, `libMagAOX/Makefile`, root `Makefile`, `tests/tests.list`, `tests/groups.dox` and generated-artifact ignore handling. The app tests include FlatBuffers telemetry serialization/verification.
- **Documentation:** this plan, the [debugging and optimization record](dm-syncopation-debugging.md) and `apps/dmSyncopation/doc/dmSyncopation.md`. The independent review and its user responses remain the review evidence.

### 2026-10-21 independent primary publication/receipt diagnostics

The operator requested adding primary-stream cadence statistics to this app after
the DM controller's new combined-publication measurement closely matched its
receipt timing. Reuse `dev::dmTiming` for the primary observer instead of adding a
second accumulator: the existing monitor supplies real receipt/attachment events,
and `processImage` copies best-effort `cnt0`/`writetime` metadata before the cycle
mutex and scheduling/resource early returns. Timing collection runs while synco
is disabled, enabled, acquiring or faulting. Only monitor attachment changes its
session; enable/reacquire and auxiliary-resource failures do not reset it.

Preallocate the bounded handoff before starting the monitor. New startup-only
`diagnostics.windowSec=2` and `diagnostics.maxEvents=20000` select the elapsed
window and queue/retained-event cap; either zero disables collection. The monitor
captures markers/metadata and hands off fixed events without allocation,
statistics, INDI or telemetry. App logic drains/calculates outside the cycle mutex.
External publication differences remain in the producer clock; local receipts
(entry if unavailable) define coverage/window membership. No cross-clock latency
or verified frame identity is inferred. Invalid/missing timestamps, discontinuous
counters, timing-event loss and monitor attachment changes break publication pairs.

Read-only `primary_timing` exposes `publication_` and `receipt_` prefixes with
`fps`, `mean`, `jitter`, `min`, `max`, `count`, `span` (14 elements). Durations use
seconds; unavailable duration/FPS is -1 and empty counts are zero. Separate
`primary_timing_quality` reports configuration, validity, coverage, epoch/sequence,
queue occupancy and cumulative observation/drop counts (17 elements). Primary
validity requires both interval populations' sample variances, independently of
SDK-oriented compatible timing availability or synco schedule state.

Append an optional primary table and default-true `schedule_valid` to the existing
`telem_dmsynco` schema; retain existing fields and eleven-argument construction.
With diagnostics enabled, record on each ordinary app pass, normally about 1 Hz.
When the cycle mutex is busy, fresh primary statistics still record with
`schedule_valid=false` and explicitly unavailable scheduler placeholders; readers
must ignore those schedule fields. Schema/log formatting and ordinary recording
remain outside the real-time path. Older buffers without primary data remain
readable. Older readers remain structurally compatible but cannot honor the new
schedule-validity flag; rebuild readers before interpreting new busy-cycle
records, whose scheduler placeholders are explicitly unavailable.

This is a diagnostic addition only. Period selection, counter-driven scheduling,
timing bounds, fault/recovery decisions and provider schedule layout stay intact.
The existing fault-latched primary timing context retains its own sampling and
meaning. Tests cover disabled/resource-failure collection, exact integer timestamp
differences, invalid metadata/discontinuities/loss, staleness/zero configuration,
INDI/telemetry independence and a real isolated-stream monitor path. The user
guide documents the new configuration, properties, record meanings and an
operator-run primary-versus-combined comparison.

Scoped workstation verification passes:

- App tests: **2,487 assertions / 30 cases**, retaining the existing scheduler,
  resource, camera, callback and fault regressions, plus five primary-cadence
  cases. A real private-stream monitor exercises hook discovery, actual wait
  notifications, callbacks and the collector. Its finite five-post sequence uses
  a generous test-only retention window and does not assume hard 2 kHz host
  scheduling. Deterministic cases cover the two-second default and staleness.
- Dedicated logger tests: **138 assertions / 4 cases**, including binary/JSON
  roundtrip, physically absent legacy primary/validity fields, disabled/default
  construction, independent scheduler validity and malformed metric rejection.
  Total scoped checks: **2,625 assertions / 34 cases**.
- Normal library/schema generation, dmSyncopation and logdump builds pass. App,
  reader and focused test builds are warning-free; schema generation retains
  existing unrelated field-name style warnings.
- All four touched C++ files pass clang-format and whitespace checks. Full-file
  formatting applies the standing explicit-brace preference to legacy app/test
  control blocks as well as the new code.

Review corrected a temporary-pair compiler warning and made the busy-lock test
bounded, releasing the owner before checking the result so a future blocking
regression fails instead of hanging. The real-monitor fixture joins before
clearing its already-closed imported handle, and uses an initial known external
timestamp to test the first publication pair deterministically. New INDI coverage
checks registration, permissions, dimensions, initial unavailable values and a
safe no-driver update; it does not exercise numeric XML output through a driver.
No new performance benchmark, ThreadSanitizer, or instrument qualification is
claimed. Existing timing bounds/control behavior are unchanged.

Affected files:

- `apps/dmSyncopation/dmSyncopation.hpp`, `Makefile`,
  `tests/dmSyncopation_test.cpp`, `doc/dmSyncopation.md`.
- `libMagAOX/logger/types/telem_dmsynco.hpp`,
  `types/schemas/telem_dmsynco.fbs`, `tests/dmSyncoTelemetry_test.cpp`.
- `tests/Makefile.one`, `tests/tests.list`.
- This plan, `agents/plans/2026-10/dm-syncopation-debugging.md` and
  `agents/plans/2026-10/dm-timing-telemetry.md`.

The user authorized committing and pushing this update to
`origin/jrmales/dm-syncopation`. Package the implementation and this engineering
record first, followed by documentation and separate formatting cleanup. No
deployment or agent instrument action is included. The next operator run can keep
synco disabled and compare its primary publication population with the
controller's combined-publication population over overlapping windows, using a
rebuilt app and reader.

## Follow-up and Edge Cases

- **Hard guarantee limit:** an unbounded early main event can race a multi-channel copy or overlap an already submitted command. Polling counters before writes cannot undo that race. Bounds and operational coordination are required; hard gating would be a separate cooperating-system change.
- **Reader lifetime versus cadence:** dmcomb output is only one reader's completion. The late `makeDelta()` reread can leave a very narrow step-0 window; changing delta membership would not remove every read. Do not declare three steps feasible from `T/N >= K+G` alone.
- **Quality versus safety:** retain the user's actual dmcomb-output jitter evidence and qualify auxiliary phases. A 5 us warning need not become a 16 ms reacquisition outage; a safe-looking RMS must not override minimum spacing or hard deadlines.
- **Retention versus clearing:** deliberate controller zero/release explicitly clears current main/auxiliary contents. Later writes can reapply contributions and are outside the controller's responsibility, as directed by the user. Operator control of active writers is separate from this clearing implementation.
- **Other writers and coalescing:** counter changes before a poll can merge into one output. Unknown combinations can already have violated keepout before detection. Until integrated, direct-trigger tools require quiescence; do not silently permit them because they are infrequent.
- **Clock and provider scope:** scheduling can avoid image timestamp conversions only by accepting a qualified observation model. Remote providers need explicit host-clock/transport semantics, not merely more fields in a local timing record.
- **Documentation status:** the plan records completed local implementation/verification and the settled review answers; instrument qualification is pending. The guide is aligned with the actual API and includes the 200 Hz tip/tilt test provider. All task text above Agent Findings and Plan and all user answers are preserved. No instrument action is included.
