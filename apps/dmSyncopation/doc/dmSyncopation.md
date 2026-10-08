# dmSyncopation user guide

This guide specifies the user interface before C++ implementation. Configuration keys and INDI property names below are proposed for review. The example is a configuration template; measured timing budgets still need to be supplied before operational use. The [engineering plan](../../../agents/plans/2026-10/dm-syncopation.md) records decisions and implementation work.

## Purpose

`dmSyncopation` schedules auxiliary deformable-mirror contributions between updates from the main AO loop. One app instance serves one DM. The initial instance is `dmtweeter_synco`, using main dmcomb channels 07 and 08, a maximum command cadence of 6 kHz, and up to three steps per main-loop cycle.

The main loop remains the timing reference. Its publication of `dmC` defines step 0. The app prepares auxiliary contributions before that publication, then adds uniformly spaced combinations during the rest of the cycle. A 2 kHz main loop with three steps therefore produces a nominal 6 kHz command cadence: one main-triggered combination and two auxiliary-triggered combinations per cycle.

The app always copies every mapped contribution at every active step. Unchanged provider values remain valid and are copied again. There is no pixel-comparison pass or intentional no-copy branch.

## Streams and ownership

```text
                            MAIN COMMAND PATH

   WFS -> reconstruction/filter -> DM command calculation
                                          |
                                          v
                                dmC / dm01disp03
                                          |
                              genuine main publication
                                          |
                                          v
 AUXILIARY COMMAND PATH                 dmcomb ------> dm01disp ------> bmcCtrl
                                          ^                            hardware
 provider A -> dm01disp07_aux --+          |
                               |          |
                               v          |
                         dmSyncopation     |
                               |          |
 provider B -> dm01disp08_aux --+          |
                               |          |
                    copy all mappings     |
                               |          |
                               v          |
                     dm01disp07 / 08 ------+

 Step 0: copy only; main dmC publication causes the combination.
 Step j > 0: copy all mappings, then increment dm01disp07.cnt0 once.

 dmSyncopation observes dmC and dm01disp for timing and output confirmation.
 It does not publish or post trigger semaphores on dmC.
```

| Stream | Writer | Role |
| --- | --- | --- |
| `dmC`, normally an alias of `dm01disp03` | Main AO command process | Primary publication timestamp/counter and authoritative main command. |
| `dm01disp07_aux` | Provider for mapping 07 | Persistent auxiliary contribution, a 2-D float32 image. |
| `dm01disp08_aux` | Provider for mapping 08 | Persistent auxiliary contribution, a 2-D float32 image. |
| `dm01disp07`, `dm01disp08` | `dmSyncopation` while enabled | Destination inputs read by dmcomb. The app replaces each mapped input with its selected auxiliary contribution. |
| `dm01disp` | dmcomb | Sum of all main DM input channels; observed to confirm consumption of a prepared batch. |
| `dmtweeter_synco_timing` | `dmSyncopation` | Proposed timing stream for provider deadlines and synchronization notifications. |

Source and destination dimensions must match the DM; the current tweeter configuration is 50x50. Only one provider owns each auxiliary source, and this app must have exclusive ownership of its destination channels. Contributions on other dmcomb channels continue to participate in the sum.

There is one auxiliary image per mapping, shared by all steps. Increasing the number of subframes does not create `_00`, `_01`, or other per-step image arrays. Compatible existing auxiliary images retain their contents. Newly created auxiliary images are initialized from stable destination contents so attaching the app does not implicitly remove an existing contribution.

The combiner must check changes to the sum of its input counters promptly. A configuration that waits only for primary `dmC` events cannot supply the additional combinations through this counter-only route. The operator must qualify the route on the target system.

## Cadence and step 0

For measured main period `T` and subdivision count `N`, step spacing is `T/N`, the selected command rate is `N/T`, and the app generates `N-1` extra combinations per cycle. Step 0 is included in the configured maximum auxiliary rate.

```text
Example: T = 500 us, N = 3, selected cadence = 6000 Hz

Time (us)         0            166.667         333.333       480       500
                  |               |               |          |         |
Main dmC          ^                                                    ^
Aux counter                       ^               ^
Combined step     0               1               2                    0
                                                                     next cycle

Step 0 at t=0 was copied during the previous cycle.
Step 0 for t=500 must finish by t=480 with a 20 us completion lead.
No destination counter changes at t=480; the main event at t=500 triggers.

The last auxiliary batch must be consumed before next-cycle step-0 copying.
```

The diagram shows intended combination submissions. Combined-output publication and controller receipt occur later, after processing/delivery latency. The hardware keepout applies at the controller, so spacing between submissions must also allow for variation in that latency.

The app calculates the largest feasible integer `N`. Useful upper bounds are:

```text
keepout limit = floor(shortest allowed main period / (keepout + guard))
rate limit    = floor(maximum auxiliary rate * estimated main period)
active steps  <= min(keepout limit, rate limit, configured maximum steps)
```

The app additionally checks copying, output confirmation, early-main uncertainty, and the final gap to the next main event. A rate cap or maximum-step setting is a limit, not a request to run at an independent frequency. `N=1` provides passive step-0 updates only. If no count is feasible, auxiliary operation stays unarmed.

At 2 kHz, a 115 us keepout plus the proposed 20 us guard permits three subdivisions. Each 166.667 us spacing has 51.667 us nominal margin over keepout, of which 20 us is reserved by the guard. These figures do not establish worst-case timing on an instrument. With a strict 6 kHz cap, a measured main rate slightly above 2 kHz can reduce the count to two; the app reports the limiting reason rather than exceeding the cap.

## Timing and deadlines

There are separate deadlines for the provider and the app:

| Event | Meaning |
| --- | --- |
| Provider completion deadline | Finish publishing the desired source value before the bounded sampling window. This is timing information for the writer; the app cannot force the writer to finish. |
| Source snapshot cutoff | The app freezes the selected complete batch by this time. Later-completing snapshots cannot replace values for that step. |
| Destination completion deadline | The app must finish copying every selected contribution into the main input channels. For step 0 this precedes the predicted primary publication by `step0LeadUs`. |
| Combination target | For step 0, the genuine primary publication. For later steps, the target time for the app's single destination counter increment. |

Let `Pnext` be the predicted next primary publication, `S` the bounded snapshot budget, and `C` the bounded destination-copy budget. For step 0:

```text
Destination completion deadline D0 = Pnext - step0LeadUs
Source snapshot cutoff         F0 <= D0 - C
Provider completion deadline   W0 <= F0 - S

                 sampling window      copying window        main lead
                <------ S ------>    <------ C ------>    <----------->
Time -----------W0---------------F0-------------------D0---------------Pnext-->
                |                |                    |                |
Writer complete |                |                    |                |
Batch frozen --------------------+                    |                |
All destinations copied ------------------------------+                |
Genuine main publication ----------------------------------------------+
```

Output confirmation and timing uncertainty may require earlier preparation than these simple budget inequalities indicate. The app publishes actual schedule deadlines; providers should read them rather than reconstructing the deadline from INDI rates.

### Keepout, guard, lead, and timing-error limit

These settings serve different purposes:

| Setting | Initial value/status | Application |
| --- | --- | --- |
| `scheduler.keepoutUs` | 115 us, selected low estimate | Time from controller receipt/wake through processing completion; used to bound subdivisions and trigger spacing. |
| `scheduler.guardUs` | 20 us, proposal | Additional spacing margin for path variation and timing uncertainty. |
| `scheduler.step0LeadUs` | 20 us, proposal | How early the entire step-0 destination copy must finish before predicted `dmC` publication. |
| `scheduler.maxTimingErrorUs` | 5 us, initial interpretation of the user's timing answer | Absolute timing-error limit for primary phase residuals and later-step counter triggers; also a qualification target for those events. |

The 5 us timing-error limit is kept separate from the 20 us guard and completion-lead proposals. The user reports that semaphore-driven events typically achieve less than 2 us RMS jitter; that observation motivates the initial tolerance but is not a worst-case bound for this app.

A step-0 copy finished after `D0` is a deadline miss even if it is only 1 us late, within the 5 us event tolerance, and still before the predicted primary publication. The event tolerance does not extend that deadline. Finishing earlier than `D0` is acceptable when the preceding batch has been consumed and the copy window is otherwise safe; early completion is measured for wait tuning.

For a later step, an already-overdue target beyond the timing limit suppresses the auxiliary counter increment. If an out-of-limit trigger error is detected after publication, the app records the violation and cancels remaining auxiliary triggers while recovering. Keepout and the earliest-next-main window are checked independently and take precedence over a tolerable timing residual. No overdue trigger is replayed as a catch-up burst.

The initial recovery policy stops the remaining auxiliary work on a scheduler timing failure and reacquires synchronization. A numerical acceptable miss rate over time has not been specified. Miss counts and qualification statistics remain visible; no miss-rate allowance permits unsafe spacing.

### Persistent values and late provider writes

Providers publish through ordinary ImageStreamIO write/counter semantics. They may leave a source unchanged for any number of cycles. Each step still copies its complete contribution. When a source is being rewritten or races the bounded snapshot, the proposed fallback selects that mapping's last complete cached value, reports cached-snapshot use, and copies it normally. A new value completed after a step's cutoff becomes eligible for a later snapshot.

For example, provider A may publish nothing new for step 0 and then publish a new command before the step-1 writer deadline. The app copies A's previous value for step 0 and its new value for step 1. This is normal operation, not a step-0 miss. An app copy that could not start safely or finished too late is a separate scheduler failure.

Providers explicitly publish zero to remove their contribution. Source age or an unchanged counter never implies zero and never causes a freshness fault. The proposed last-complete-cache fallback remains a review item; there is no provider ready/acknowledgement handshake.

### Step decision tree

```text
        Locked; previous destination batch consumed; next step planned
                                  |
             Select each source's complete value by the cutoff
             (use last complete cache for an in-progress write)
                                  |
                    Valid complete batch available?
                       NO /                 \ YES
                 Resource fault             |
                 retain values      Copy fits guarded window?
                 and disarm           NO /           \ YES
                               NOT_STARTED           |
                               report/cancel    ALWAYS COPY
                               and recover      EVERY MAPPING
                                                    |
                                              Which step?
                                      STEP 0 /           \ STEP > 0
                           Finished by D0?               Copy complete and
                             NO /     \ YES              trigger window valid?
                     COPIED_LATE       COPIED_ON_TIME       NO /       \ YES
                     report/cancel          |          suppress        |
                     and recover       No counter       trigger   Wait to target;
                                       increment;      and recover recheck; increment
                                       await main                ONE destination cnt0

No provider update is a normal data case. No destination-copy HOLD branch exists.
```

`NOT_STARTED` and `COPIED_LATE` are distinct diagnostics. Both cancel the affected cycle's additional triggers and initiate recovery. `COPIED_ON_TIME` permits later steps only after the genuine main event and its combination have been confirmed, with phase/keepout still valid. The main loop continues independently of auxiliary recovery.

## Provider timing stream

`dmtweeter_synco_timing` is the proposed high-rate interface for providers. It publishes a coherent schedule, then posts independent consumer semaphores so each provider can reread it. A notification is a prompt to read current timing; it is not a lossless cycle counter.

The schedule describes its version, session and schedule generation, validity, primary cycle identifier, active step count/period, primary reference, and each step's writer deadline, snapshot cutoff, copy deadline, and combination target. Times and identifiers use typed integer fields with an explicit clock identifier. The binary field layout will be frozen with implementation; providers must not guess offsets from this guide.

The schedule includes step 0 of a future cycle early enough for provider work. A provider seeing a new generation or invalid schedule should discard obsolete timing predictions. Existing pixel values remain valid when timing generations change. Providers do not send per-cycle commits, explicit reuse records, or consumption acknowledgements.

Source metadata uses the build's `CLOCK_ISIO`, which may resolve to TAI or realtime. Internal app waits and the provider schedule use host monotonic time; the app converts the primary publication timestamp into that domain. Providers must honor the schedule's clock identifier. A same-host provider can use the same monotonic clock; a remote provider needs an explicit clock/transport arrangement. INDI is not the transport for subframe deadlines.

## Configuration

The proposed configuration file is `dmtweeter_synco.conf`, using normal MagAOX section/key syntax. In the tables, a **selected** value comes from the reviewed plan; a **proposal** is an initial interface/default choice for review; a **required site value** needs a measured or explicitly approved bound. All durations ending in `Us` are microseconds; rates ending in `Hz` are hertz.

### Streams and threads

| Key | Initial value/status | Meaning and constraints |
| --- | --- | --- |
| `dm.channels` | `dm01disp07,dm01disp08`, selected | Ordered comma-separated destination list; nonempty, existing, same-DM 2-D float32 geometry. Reject aliases that duplicate a mapping or name the primary/combined/timing stream. |
| `dm.auxChannels` | Optional explicit list | Source names paired with `dm.channels` in the same order; equal list lengths and unique source ownership. When omitted, derive sources using `dm.auxSuffix`. |
| `dm.auxSuffix` | `_aux`, proposal | Suffix for generated source names. With the selected destinations this produces `dm01disp07_aux` and `dm01disp08_aux`; ignored when an explicit source list is supplied. |
| `dm.triggerChannel` | `dm01disp07`, selected | Exactly one member of the destination list whose counter is incremented for later steps. If omitted, use the first destination. |
| `primary.shmimName` | `dmC`, selected | Main-command stream observed for phase; may resolve to `dm01disp03`. |
| `combined.shmimName` | `dm01disp`, selected | Combined output observed for confirmation. |
| `primary.getExistingFirst`, `combined.getExistingFirst` | `false`, proposal | Wait for new observations; existing images must not be counted as a new main event or a pending combination's confirmation. |
| `scheduler.threadPrio` | `30`, selected | Timing-worker RT priority; validate against the platform scheduler range. Failure to obtain required RT scheduling prevents arming. |
| `scheduler.cpuset` | `spare61`, selected | Worker cpuset on the relevant RTC, CPU 61 in the current setup. The user's machine-specific assignment supersedes the earlier global-config recommendation. |
| `primary.threadPrio`, `combined.threadPrio` | `30`, proposal | Primary/output observer priorities; bounded callbacks must remain responsive. |
| `primary.cpuset`, `combined.cpuset` | `spare61`, proposal | Initial observer placement. Keep slower app, INDI, and logging work outside the timing core. |

Changing mapping, names, maximum-step capacity, or thread placement requires disabling and restarting the app. Compatible auxiliary streams are retained across restart. Main DM streams are never destroyed by this app.

### Cadence and timing

| Key | Initial value/status | Meaning and constraints |
| --- | --- | --- |
| `scheduler.keepoutUs` | `115`, selected estimate | Positive controller processing keepout. Increasing it can reduce the selected count. |
| `scheduler.guardUs` | `20`, proposal | Nonnegative additional spacing margin. It must cover the approved path/timing uncertainty; a 5 us event tolerance does not automatically make a 5 us guard sufficient. |
| `scheduler.step0LeadUs` | `20`, proposal | Positive lead for the complete step-0 destination batch. |
| `scheduler.maxAuxRateHz` | `6000`, selected | Positive strict cap on total step cadence, including step 0. |
| `scheduler.maxSubframes` | `3`, selected | Positive maximum step count; bounds scheduling/feedback storage. It does not change the number of auxiliary images. |
| `scheduler.maxTimingErrorUs` | `5`, initial interpretation | Positive primary phase/later-trigger timing-error limit. It never extends the step-0 deadline or weakens keepout. |
| `scheduler.snapshotBudgetUs` | Required site value | Positive worst-case budget for sampling/validating all mapped sources with bounded retry. Used to put writer deadlines before snapshot cutoffs. |
| `scheduler.copyBudgetUs` | Required site value | Positive worst-case budget for writing the entire destination batch and its metadata. Averages are insufficient for deciding whether a copy fits. |

The clock follows `CLOCK_ISIO` for source timestamp interpretation and metadata publication. It is reported on INDI rather than independently selected by configuration. Host monotonic time is used for waits and provider schedule timestamps. Producer and consumer builds must agree on the source timestamp domain.

### Tracking, feedback, and diagnostics

| Key | Initial value/status | Meaning and constraints |
| --- | --- | --- |
| `tracking.minMainRateHz`, `tracking.maxMainRateHz` | Required site values | Positive acquisition-rate bounds with minimum no greater than maximum. They constrain acceptable observations; they do not prescribe the actual main rate. |
| `tracking.earlyMainUs` | Required site value | Nonnegative bound on early primary arrival used to construct the guarded copy/trigger windows. An error threshold is not proof of this bound. |
| `tracking.periodWindow` | `32`, proposal | Positive number of main observations used for period estimation. A longer window smooths noise but adapts more slowly to rate changes. |
| `tracking.lockFrames` | `32`, proposal | Consecutive valid main frames required before resuming; actual acquisition also needs a usable period estimate and prepared future step 0. |
| `tracking.mainTimeoutPeriods` | `3`, proposal | Positive timeout multiplier of the estimated main period; acquisition uses the configured rate bounds until an estimate exists. Main silence disarms auxiliary triggers. |
| `tracking.outputTimeoutUs` | Required site value | Positive bound for a corresponding combined output. It must fit the selected schedule's consumption/copy windows. Missing/coalesced output cannot be treated as confirmation. |
| `feedback.gain` | `0.1`, proposal | Integrator gain in `[0,1]`; zero disables adaptation. Each step has its own correction. |
| `feedback.maxAdvanceUs` | Required site value | Nonnegative limit on how much wait advancement may be learned. Clamping prevents windup and excessive early wakes. |
| `feedback.spinUs` | `0`, proposal | Nonnegative bound on optional final spinning before a target. Zero disables spinning; any nonzero choice needs timing/CPU qualification. |
| `diagnostics.windowSamples` | `2000`, proposal | Positive bounded sample window per step for displayed statistics and timing diagnostics. |
| `telemeter.maxInterval` | `10` seconds, existing helper default | Maximum interval between aggregate telemetry records. Telemetry is separate from the provider schedule. |

The worker learns repeatable wake/work bias by advancing its waits after late successful events and reducing the advance after early completion. It does not move cycle zero or permit an early auxiliary counter increment. Corrections reset on lock/count/timing changes, freeze after failed events, and stay within `feedback.maxAdvanceUs`. Jitter from unpredictable scheduling still needs margin.

### Example configuration template

This template contains the selected deployment values and proposed review defaults. The commented required site values must be filled before using it as an operational configuration. The app always starts disabled; enabling is an explicit INDI action after validation and qualification.

```ini
# Proposed dmtweeter_synco.conf; complete measured site values before use.
[dm]
channels=dm01disp07,dm01disp08
auxSuffix=_aux
triggerChannel=dm01disp07

[primary]
shmimName=dmC
getExistingFirst=false
threadPrio=30
cpuset=spare61

[combined]
shmimName=dm01disp
getExistingFirst=false
threadPrio=30
cpuset=spare61

[scheduler]
threadPrio=30
cpuset=spare61
keepoutUs=115
guardUs=20
step0LeadUs=20
maxAuxRateHz=6000
maxSubframes=3
maxTimingErrorUs=5
# snapshotBudgetUs=<measured worst-case complete-source sampling budget>
# copyBudgetUs=<measured worst-case complete-destination copy budget>

[tracking]
# minMainRateHz=<approved acquisition range minimum>
# maxMainRateHz=<approved acquisition range maximum>
# earlyMainUs=<approved early-primary arrival bound>
periodWindow=32
lockFrames=32
mainTimeoutPeriods=3
# outputTimeoutUs=<approved combination response bound>

[feedback]
gain=0.1
# maxAdvanceUs=<qualified bound on wait advancement>
spinUs=0

[diagnostics]
windowSamples=2000

[telemeter]
maxInterval=10
```

All numeric inputs must be finite and in range. Required budgets and the complete schedule must be feasible together. Stream role/type/geometry conflicts, unavailable resources, and invalid combinations prevent arming. The app does not silently clamp an infeasible subdivision count to one.

## INDI interface

All properties below belong to device `dmtweeter_synco`. They specify the proposed interface, following existing MagAOX `toggle`, `request`, and numeric `current`/`target` conventions. Standard MagAOX FSM, monitor, thread-health, logging, and telemetry properties will also be present.

### Controls

| Property | Type/access | Elements and use |
| --- | --- | --- |
| `enabled` | Switch, read/write | `toggle`: request On to acquire/enable, Off to cancel auxiliary work. On expresses intent, not proof of lock; consult `sync_status`. Off is acknowledged after pending work is cancelled so no further auxiliary counter increment can occur. |
| `reacquire` | Switch, read/write | `request`: momentary request to invalidate old timing, reset wait feedback, and acquire a fresh main reference. Requires enable intent; it does not zero contributions. Resets to Off after acceptance. |
| `keepout` | Number, read/write while disabled | `current`, `target`, us; maps to `scheduler.keepoutUs`. |
| `guard` | Number, read/write while disabled | `current`, `target`, us; maps to `scheduler.guardUs`. |
| `step0_lead` | Number, read/write while disabled | `current`, `target`, us; maps to `scheduler.step0LeadUs`. |
| `max_aux_rate` | Number, read/write while disabled | `current`, `target`, Hz; maps to `scheduler.maxAuxRateHz`. |
| `timing_error_limit` | Number, read/write while disabled | `current`, `target`, us; maps to `scheduler.maxTimingErrorUs`. |
| `snapshot_budget`, `copy_budget` | Number, read/write while disabled | Each has `current`, `target`, us; maps to the corresponding scheduler budget. |
| `feedback_gain` | Number, read/write while disabled | `current`, `target`, dimensionless; maps to `feedback.gain`. |
| `feedback_advance_limit` | Number, read/write while disabled | `current`, `target`, us; maps to `feedback.maxAdvanceUs`. |

Write a numeric property's `target`, then verify readback. Valid values become `current` while disabled. A request made while enabled or an invalid/nonfinite/infeasible value is rejected; the active setting remains unchanged and the property reports Alert. Mapping/capacity/thread settings and configuration-only tracking/diagnostic settings require restart. An INDI update does not write the source images or directly trigger the DM.

### Status and measurements

| Property | Type/access | Elements and interpretation |
| --- | --- | --- |
| `sync_status` | Text, read-only | `state`, `reason`, `limit`, `session`, `generation`: scheduler state, most recent fault/transition, subdivision limiting reason, and timing identity. |
| `rates` | Number, read-only | `main`, `selected`, `delivered`, Hz: measured primary rate, armed nominal total cadence, and measured confirmed combination rate. Selected/delivered are zero when no auxiliary schedule is armed; this does not mean the main loop has stopped. |
| `subframes` | Number, read-only | `active`, `extra`, `keepout_limit`, `capacity`: active count `N`, extra trigger count `N-1`, guarded keepout upper bound, and configured maximum steps. Active/extra are zero while unarmed. |
| `interval` | Number, read-only | `current`, us: armed step interval; zero while unarmed. |
| `streams` | Text, read-only | `primary`, `combined`, `trigger`, `timing`: resolved stream identities and designated counter-trigger destination. |
| `mapping_0`, `mapping_1`, ... | Text, read-only | `source`, `destination`: one property per configured mapping in list order. |
| `clock` | Text, read-only | `primary`, `schedule`: resolved `CLOCK_ISIO` domain and host monotonic schedule domain. |
| `step0` | Text, read-only | `outcome`, `cycle`: last planned step-0 result (`COPIED_ON_TIME`, `NOT_STARTED`, `COPIED_LATE`; `NONE` before a result) and its cycle identifier. |
| `timing_step0`, `timing_step1`, ... | Number, read-only | `lastErrorUs`, `meanErrorUs`, `jitterRmsUs`, `maxAbsErrorUs`, `advanceUs`: event residuals and learned wait advance. Properties exist up to capacity; inactive steps are Idle. |
| `counters` | Number, read-only | `step0NotStarted`, `step0Late`, `timingErrors`, `suppressedTriggers`, `missingOutputs`, `unexpectedOutputs`, `cacheReuse`, `mainFaults`, `clockFaults`, `recoveries`: cumulative diagnostics for the app run. |

Step-0 residual is actual full-batch completion minus `D0`; positive means late. Later-step residual is actual counter increment minus its target. `jitterRmsUs` describes variation about the mean residual; `meanErrorUs` shows systematic bias and `maxAbsErrorUs` preserves excursions. Timing statistics reset for a new schedule generation and use the bounded diagnostic window; fault counters persist until app restart. Skipped triggers have no execution residual and are represented by counters. Failed executed events remain visible in diagnostics even when excluded from feedback learning.

`cacheReuse` counts fallback caused by an incomplete/racing source snapshot, not ordinary unchanged provider data. None of these freshness-related observations cause automatic zeroing. Confirmed combined-output rate is a software measurement, not proof of hardware completion or receipt spacing. INDI publishes summaries at the app's status-update cadence; providers use the timing stream for subframe scheduling.

## Operating sequence and recovery

```text
                       enable
             DISABLED ---------> ACQUIRING
                  ^                  |
                  |        valid main frames + period estimate
                  |        + complete future step-0 preparation
                  |                  |
               disable               v
                  +--------------- LOCKED
                  |                  |
                  |            timing/phase failure
                  |                  |
                  |                  v
                  +------------ RECOVERING
                                     |
                          good frames + new prepared step 0
                                     |
                                     +---------------------> LOCKED

Invalid/missing resources -> FAULT; retain commands and disarm.
After the resource is corrected, request reacquisition with enable intent On.
Disable from any state -> DISABLED.
```

| Scheduler state | Operation |
| --- | --- |
| `DISABLED` | No destination copying or extra counter triggers. Existing contributions persist; mutable timing controls may be edited. |
| `ACQUIRING` | Observe/estimate genuine main timing and publish future deadlines. Preparation for an upcoming step 0 can occur; no extra auxiliary trigger is armed yet. |
| `LOCKED` | Copy every active step and execute guarded later-step counter triggers. Enable property is Ok. |
| `RECOVERING` | Cancel additional triggers, retain destination contents, reset feedback, and reacquire valid timing. Enable intent can remain On while the schedule is unarmed/Busy. |
| `FAULT` | Resource/configuration failure, Alert status, no armed auxiliary schedule. Correct the cause before reacquisition. |

For operational use after implementation, the operator should:

1. Establish exclusive ownership of destinations 07/08, a counter-responsive combiner, compatible streams, and the selected machine's `spare61` assignment.
2. Qualify source clock agreement, controller keepout/path variability, complete snapshot/copy budgets, early-main bounds, and expected main-rate range. Fill required site values in the configuration.
3. Start `dmtweeter_synco` disabled and verify stream/mapping/clock readback and resource health.
4. Allow providers to publish initial values or use retained compatible source contents. Request `enabled.toggle=On`; wait for `sync_status.state=LOCKED` and verify count/rate/interval.
5. Watch per-step jitter/worst-case residuals, `step0` outcomes, delivered rate, and fault/suppression counters under the relevant load. Use those measurements to qualify the 5 us target and chosen margins.
6. Disable before changing writable timing settings. For a mapping/capacity/thread change, disable and restart. Re-enable always reacquires synchronization.

A valid previous command is enough to resume; fresh provider writes are not required. Recovery does not restore an earlier step, zero a contribution, or replay overdue triggers. A main timeout, counter reset, stream replacement, clock discontinuity, missing/extra combination, or invalid timing window can disarm operation and change the timing generation.

Disabling or stopping the app preserves destination values. A zero written to an auxiliary source while disabled remains there until copying resumes; it does not immediately clear the corresponding main input channel. To clear through the normal path, publish zero and confirm it has been copied/applied while the schedule is operating before disabling. Any direct maintenance clearing while disabled is an explicit operator action outside this app's normal scheduling path.

The existing combiner does not atomically snapshot multiple input channels or block an unexpectedly early primary event during copying. An arbitrarily early main command can also arrive inside the keepout of an already submitted auxiliary command. This app detects/reports these cases and recovers; a hard gating or pipelining solution would require a separate change to the cooperating loop/controller path. All live setup, INDI, and qualification steps on the instrument are performed by the operator.

## Items to finalize before implementation

- Confirm that the selected 5 us timing value applies to the event error limit; the separate 20 us guard and step-0 lead remain proposals in this draft.
- Approve the proposed configuration/INDI names and unselected defaults (observer placement, acquisition window, feedback, and diagnostics).
- Approve the bounded last-complete-cache fallback; always-copy behavior itself is already settled.
- Supply qualified site budgets/bounds and any numerical acceptable miss-rate criteria. The initial recovery policy cancels on timing failure without a miss-rate allowance.
- Freeze the provider timing stream's binary layout/version with implementation, before external providers depend on it.
