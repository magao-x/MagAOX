# dmSyncopation user guide

`dmSyncopation` copies auxiliary deformable-mirror contributions into existing dmcomb input channels at a fixed integer subdivision of the main AO-loop period. One instance serves one DM. The initial instance, `dmtweeter_synco`, requests three steps per main cycle: approximately 6 kHz total cadence when the main loop runs at 2 kHz.

The [engineering plan](../../../agents/plans/2026-10/dm-syncopation.md) records settled design decisions; the [debugging record](../../../agents/plans/2026-10/dm-syncopation-debugging.md) records operator trials and local fixes. The trials have not yet established sustained requested cadence, and the measured-acquisition reference changes described here need operator retesting. Initial timing values are starting budgets for qualification, not measured instrument bounds. All live instrument setup, INDI actions and timing measurements are performed by the operator.

## Streams and command flow

```text
                 MAIN AO COMMAND PATH
 WFS -> reconstruction/filter -> command calculation
                                       |
                                       v
                             dmC / dm01disp03
                                       |
                        genuine main publication
                                       |
                                       v
                                    dmcomb ------> dm01disp ------> bmcCtrl
                                       ^                         DM hardware
                                       |
 provider A -> dm01disp07_aux --+       |
                              |       |
                        dmSyncopation  |
                              |       |
 provider B -> dm01disp08_aux --+       |
                              |       |
                      copy all mappings
                              |       |
                              v       |
                      dm01disp07 / 08 --+

 Step 0: copy all mappings passively; the main publication triggers dmcomb.
 Step >0: copy all mappings passively, then publish channel 07's counter once.
 dmC and dm01disp are observed; dmSyncopation never publishes on dmC.
```

| Stream | Writer | Purpose |
| --- | --- | --- |
| `dmC`, normally `dm01disp03` | Main AO command calculation | Genuine primary commands; their publication defines cycle zero. |
| `dm01disp07_aux`, `dm01disp08_aux` | One provider per source | Persistent 2-D float32 contributions, one image per mapping. |
| `dm01disp07`, `dm01disp08` | dmSyncopation during normal operation | Existing dmcomb input channels receiving complete copies. Controller zero/release may also clear them. |
| `dm01disp` | dmcomb | Combined output; its counter confirms that dmcomb consumed the inputs. |
| `dmtweeter_synco_timing` | dmSyncopation | Monotonic timing information and semaphore notifications for local providers. |

The first operator deployment uses INDI device `dmtweeter-synco` and shared-memory timing image `dmtweeter_synco_timing`; the hyphen and underscore names are distinct. Its primary filesystem image is `dm01disp03`/`aol1_dmC`. The diagrams use `dmC` as a logical command label; a bare `dmC` name is not an automatic filesystem alias. Set `shmimMonitor.shmimName` to an image that actually exists and set `streams.timing` explicitly to match the provider.

Primary, combined, source and destination pixel streams must be CPU-backed 2-D float32 images with matching geometry and at most 1,048,576 pixels. The current tweeter uses 50x50 images. Sources follow `<destination>_aux` naming so controller clearing can discover them. A provider writes only its own auxiliary source. A source cannot be shared silently by two writers.

There is one auxiliary image per mapping, regardless of the requested step count. Compatible existing source values are retained; newly created sources are seeded from stable destination contents. No per-step `_00`/`_01` payload arrays or provider readiness acknowledgements are needed.

Every active step copies every mapping, including unchanged values and cached complete values. There is no pixel-comparison pass or successful no-copy/HOLD branch. An unchanged source remains valid indefinitely. Providers explicitly publish zero to remove a contribution.

## Cadence and synchronization

```text
Example main period: 500 us; requested steps: 3

Time (us)       0              166.667           333.333              500
                |                  |                 |                  |
Main dmC        ^                                                       ^
Aux counter                        ^                 ^
Combined step   0                  1                 2                  0
                                                                  next cycle

Step 0 at t=0 was copied during the previous main cycle.
After step 2's readers finish, prepare step 0 for t=500 before its safe cutoff.
Only the real main publisher triggers the combination at t=500.
```

`maxSubframes` sets the requested count, initially three; step 0 counts toward it. The app calculates a feasible maximum from keepout, reader lifetime and preparation windows, and validates the request. It reports an infeasible request rather than silently changing three steps to two. `N=1` is passive-only operation.

The planned interval is the selected main period divided by the requested count. Small period changes around 500 us do not switch the integer count. The nominal total-cadence sanity rating is 6000 Hz with 1% tolerance and 32-frame persistence for ordinary rate changes. Hard safety violations take effect immediately.

The period reference is selected in this order:

1. A fresh measured camera acquisition rate within the configured main-rate range, default `camwfs.fg_timing.acq_fps`.
2. A qualified primary window: `tracking.rateFrames` sequential, broadly valid intervals whose range is at most twice `tracking.earlyMainUs`, averaged as total elapsed time divided by interval count.
3. The configured nominal rate when neither measured source is usable.

A short callback EMA or single anomalous interval no longer determines the period. Changing reference source, or changing the selected period by more than `tracking.earlyMainUs`, resets phase/output/feedback acquisition without incrementing a synchronization-fault counter. The camera period reference assumes one nominal main DM cycle per camera frame. For an intentionally divided/non-1:1 main cadence, disable the camera reference with an empty `tracking.cameraDevice` and use the qualified primary-window/nominal path. Unexpected counter gaps remain faults. The genuine DM publication/counter still anchors each cycle; the camera report supplies a period, not a replacement phase. Primary intervals outside the selected prediction's configured bounds still cause a timing fault. A stable acquisition average does not establish that every downstream DM publication/callback lies within those bounds.

Only primary counter progression establishes a new physical main event. A callback whose counter is unchanged is ignored and counted as `duplicatePosts`; a queued semaphore notification does not supply a new cycle or timestamp. Counter jumps/resets are adopted once and cause reacquisition, so repeated notifications of that same value do not repeatedly reacquire.

Scheduling uses `CLOCK_MONOTONIC` on the RTC. The primary callback observes publication after semaphore delivery and monitor bookkeeping; that observation has lag and uncertainty. Wait feedback tunes later scheduling bias but cannot discover an unobserved publication-to-callback offset. Image metadata follows ImageStreamIO's clock conventions and is not interpreted as a monotonic deadline.

The selected counter-polling dmcomb configuration uses DELAY mode 4 and a requested 10 us sleep. Passive destination posts can notify viewers while leaving counters unchanged. Only the final designated counter publication triggers an auxiliary combination. The requested poll interval does not establish a delivered-jitter floor: the user's existing less-than-2 us RMS measurement is at dmcomb output.

## Deadlines and reader lifetime

Provider timing and app timing are distinct:

| Event | Meaning |
| --- | --- |
| Source snapshot cutoff | The desired source publication must be complete before the app samples it for that step. A later value becomes eligible at a later sample. |
| Preparation target | The app aims to finish copying before the hard cutoff, leaving a positive reserve for scheduling variability. |
| Step-0 hard cutoff | Latest safe completion of the entire destination batch; constrained by both lead and the earliest plausible next main event. |
| Later-step target | Time for the single final destination-counter publication. |

```text
Previous submission      last controller reader finishes        next main
        |                              |                             |
        +---- delivery + processing ---+                             |
                                       |                             |
                                  sample/copy                        |
                                       +-----> target ----> cutoff --+
                                                  reserve       lead

Copying starts only after dmcomb consumption AND the controller-reader hold.
All mapped copies must complete before the applicable hard cutoff.
```

For a predicted next main event `Pnext`, the nominal step-0 deadline is `D0 = Pnext - step0LeadUs`. The effective cutoff is the earlier of `D0` and the earliest plausible main publication minus the required visibility margin. The preparation target is earlier again. A 20 us lead and 20 us early-main allowance alone provide no positive margin; validate the smaller effective window.

Before replacing a batch, the app requires dmcomb output confirmation and the previous submission's conservative delivery/processing/reader hold. `makeDelta()` rereads destination channels after `commandDM()`, so dmcomb consumption alone is insufficient. Combined output is not proof that controller processing or hardware transfer completed. Changing `deltaChannels` policy is outside this work.

The nominal three-step interval is about 166.667 us. A 115 us processing estimate, 20 us guard, 10 us copy budget and 20 us lead leave very little final preparation time once delivery and uncertainty are included. These defaults can be infeasible together; qualify the complete window rather than relying only on `interval >= keepout + guard`.

Step-0 wait feedback controls the copy-attempt start, separately from completion diagnostics. Its desired start is `preparation target - current copy budget`. After a true feedback reset the advance starts at `feedback.maxAdvanceUs` (20 us by default); the advance stays between zero and that limit. A valid wake sample compares the actual attempted start with the desired start. A safely rejected late attempt can still teach the wait controller; it does not become a successful copy.

A copy finishing sooner than its budget no longer pushes the next planned start later. Completion residuals remain diagnostics, and the reader hold still limits how early copying may start. If the reader-safe planned start plus the current copy budget already exceeds the cutoff, the step is suppressed before waiting into that known-impossible window. Ordinary missed-step reacquisition and snapshot/copy-budget enlargement preserve learned waits; true synchronization/resource faults, explicit reacquisition and configuration changes reset them. These changes do not extend a hard cutoff or relax keepout.

The 5 us value is a guideline for timing analysis. It is not a hard execution limit, a grace period extending a copy deadline, or a mandatory reacquisition threshold.

## Persistent values and decisions

Each source uses normal ImageStreamIO publication. The app checks the write flag and counter around one snapshot copy. If publication is in progress or changed during the copy, it uses that mapping's last complete cache. This is consistency fallback, not a freshness fault. The selected current/cached value is always copied to its destination.

```text
           Synchronized; step planned; previous readers finished
                                  |
                 Select complete values from all sources
                 (last complete cache if a snapshot races)
                                  |
                       Valid complete batch?
                      NO /              \ YES
                 Resource fault            |
                 disarm/retain      Copy fits safe window?
                                    NO /          \ YES
                              skip/report           |
                                             ALWAYS COPY
                                             EVERY MAPPING
                                                    |
                                                Which step?
                                       STEP 0 /            \ STEP >0
                              Complete by cutoff?       Safe to trigger?
                              NO /          \ YES        NO /      \ YES
                       cancel cycle's      await         skip      publish ONE
                       later triggers      main          step      counter once
```

A provider can leave its command unchanged for step 0, then publish a new value for step 1. Both steps copy a valid value; neither provider silence nor source age is a scheduler miss.

An unsafe individual auxiliary step is skipped. A step-0 preparation miss cancels that cycle's later triggers. Three consecutive original unsafe/missed steps cause reacquisition; a successful step resets that sequence. Steps cancelled as consequences of one failure are not additional original failures. Main timeout/reset, unexpected or ambiguous combined output, invalid resources and genuine early-main/safety violations disarm immediately. No skipped trigger is replayed.

Ordinary disable, fault and shutdown retain contributions. On an ordinary disable, an already-started bounded passive batch finishes across all mappings, then no extra trigger is issued; this avoids intentionally leaving a partially updated batch solely because of cancellation. Disable is acknowledged after quiescence, and cancellation is not counted as a deadline miss. Re-enable reacquires the main reference and prepares a future step 0 before later triggers resume. Disabled operation does not copy later provider writes, including zero, into destinations.

## Provider timing stream

The timing record supplies future snapshot cutoffs, requested count, estimated period, monotonic reference and validity/restart identity. An ImageStreamIO notification tells a provider to reread the coherent current record; it is not a lossless event count. State/generation changes invalidate old timing predictions, not persistent pixel values.

Notifications after sampling describe the next usable cutoffs, including a future step 0. Providers must honor validity and expiry when main timing stops. The first provider is local dmSpeckle on the RTC. A remote ICC/ACC001 provider needs a separately specified clock mapping and transport budget; RTC monotonic nanoseconds cannot be used as another host's clock.

The shared declaration is [dmSyncopationTiming.hpp](../../../libMagAOX/app/dmSyncopationTiming.hpp). Layout version 1 is a CPU-backed one-dimensional signed-int64 image with 41 words and room for 16 steps:

| Word offset | Field | Meaning |
| --- | --- | --- |
| 0 | `version` | Layout version, currently 1. |
| 1 | `epoch` | Positive restart identity. |
| 2 | `generation` | Positive schedule generation. |
| 3 | `state` | 0 invalid, 1 acquiring, 2 armed. Acquiring records allow preparation of the first step 0. |
| 4 | `steps` | Active requested count, 1–16. |
| 5 | `period` | Estimated main period in nanoseconds. |
| 6 | `reference` | Monotonic main observation reference in nanoseconds. |
| 7 | `expires` | Monotonic expiry in nanoseconds. |
| 8 | `cycle` | Main cycle identifier. |
| 9–24 | `cutoffs[j]` | Per-step future snapshot cutoff; zero means unavailable. |
| 25–40 | `targets[j]` | Corresponding command target in monotonic nanoseconds. |

Read the whole record with ImageStreamIO write/counter consistency checks, then use `validRecord()` and `nextStep()` from the shared header. A valid record has a current reference, an unexpired bounded lifetime of at most three periods, a supported state/count and cutoffs preceding their targets. Choose the earliest usable future cutoff rather than relying on semaphore count or array order. Reject unsupported layout versions and obsolete epoch/generation predictions.

## Configuration

The complete [dmtweeter_synco.conf example](../dmtweeter_synco.conf) contains the selected mapping and initial bounds. All `Us` values are microseconds, `Hz` values are hertz, and count settings are integers. Configuration-only changes require restart. The five `settings` elements below can also be changed through INDI while disabled.

| Key | Default | Meaning |
| --- | --- | --- |
| `shmimMonitor.shmimName` | `dmC` | Code default; configure the actual primary filesystem image (`dm01disp03`/`aol1_dmC` in the first deployment). Existing/unchanged-counter notifications are not new main events. |
| `shmimMonitor.threadPrio` | 30 | Primary monitor/cycle thread priority. |
| `shmimMonitor.cpuset` | `spare61` | Primary monitor/cycle thread cpuset. |
| `streams.destinations` | `dm01disp07,dm01disp08` | Ordered numbered main-channel mapping. Sources append `_aux`; the first destination supplies every later-step trigger. |
| `streams.combined` | `dm01disp` | Existing combined-output stream. |
| `streams.timing` | `<instance>_timing` | Signed-int64 provider schedule; `dmtweeter_synco_timing` for this instance. |
| `scheduler.maxSubframes` | 3 | Fixed requested count, 1–16, including passive step 0. |
| `scheduler.maxCadenceHz` | 6000 | Nominal total-cadence sanity rating, including step 0. |
| `scheduler.keepoutUs` | 115 | Controller processing/last-reader duration estimate. |
| `scheduler.guardUs` | 20 | Reserve beyond explicit delivery/reader bounds. |
| `scheduler.step0LeadUs` | 20 | Nominal step-0 completion lead. |
| `scheduler.deliveryUs` | 20 | Submission-to-controller delivery allowance. |
| `scheduler.snapshotBudgetUs` | 10 | Whole-batch one-pass snapshot budget. |
| `scheduler.copyBudgetUs` | 10 | Whole-batch destination copy/publication budget. |
| `scheduler.visibilityUs` | 1 | Positive margin before the earliest plausible main publication. |
| `scheduler.preparationReserveUs` | 2 | Preparation target's positive reserve before the hard cutoff. |
| `scheduler.outputTimeoutUs` | 50 | Combined-output watchdog, separate from the delivery/reader hold. |
| `tracking.cameraDevice` | `camwfs` | Device publishing measured acquisition timing; empty disables subscription. |
| `tracking.cameraProperty` | `fg_timing` | Subscribed acquisition-timing property. |
| `tracking.cameraElement` | `acq_fps` | Measured rate element used for the preferred period reference. |
| `tracking.nominalMainRateHz` | 2000 | Fallback period when camera and qualified primary-window references are unavailable. |
| `tracking.minMainRateHz`, `tracking.maxMainRateHz` | 500, 4000 | Accepted primary-rate range. |
| `tracking.earlyMainUs` | 20 | Early-main safety allowance and accepted early/late callback-interval deviation from the selected period; also constrains primary-window qualification. |
| `tracking.observationLagUs` | 5 | Maximum publication-to-callback lag allowance. |
| `tracking.lockFrames` | 32 | Clean main observations before preparing an active cycle. |
| `tracking.rateTolerance` | 0.01 | Relative nominal-cadence tolerance: 1%. |
| `tracking.rateFrames` | 32 | Primary-window interval count and consecutive ordinary cadence-violation persistence. |
| `tracking.primarySemaphore` | 5 | Required primary reader index; unexpected allocation/fallback is rejected, and legacy reserved index 9 is prohibited. |
| `recovery.consecutiveMisses` | 3 | Consecutive original unsafe/missed events before reacquisition. |
| `feedback.gain` | 0.1 | Independent per-event wait-integrator gain, 0–1. |
| `feedback.maxAdvanceUs` | 20 | Maximum wait correction; step-0 advance is nonnegative and starts at this value after a feedback reset. |
| `diagnostics.windowSec` | 2 | Elapsed-time window for independent primary publication/receipt statistics; zero disables collection. |
| `diagnostics.maxEvents` | 20000 | Maximum queued and retained primary timing events; zero disables collection. Storage is allocated before the monitor starts. |
| `telemeter.maxInterval` | 10 seconds | Existing helper's aggregate telemetry interval. |

The example starts disabled. Its conservative full timing bundle is **infeasible for three steps at 2 kHz**: the shortest 166.667 us interval cannot cover the reader hold, copy and effective early-main cutoff. The app reports this without silently lowering the requested count. Qualified bounds must support the requested schedule before instrument enablement.

Measured camera reports are eligible while their accepted-reply age is less than five seconds; at five seconds they expire. The app uses normal base subscription/reconnect handling and camera Set updates, without a periodic camera refresh. Unchanged values are not a heartbeat: stale/unavailable reports fall back to the qualified primary window, then the configured nominal rate. A valid INDI update can still contain an older camera statistic; receipt age alone does not prove acquisition is continuing. Primary counter progression and silence checks remain necessary. Camera reports are adopted coherently without making the timed cycle wait for input handling; a busy handoff retains the previous report, and copying it later does not reset its receipt age. Requested `fps.current` is no longer the default reference. `health.periodSource` identifies camera acquisition, qualified primary window or nominal fallback; `mainHz` is the reciprocal of the selected integer-nanosecond period, so it can differ slightly from the original floating-point camera report.

`fg_timing.acq_jitter`, `acq_min` and `acq_max` are acquisition-interval statistics in seconds. The inspected camwfs configuration uses a roughly two-second window; INDI readbacks convert these values to microseconds. They describe local camera acquisition, not downstream AO processing, DM publication jitter, callback wake latency or qualified worst-case limits. A valid FPS-only packet is accepted and missing optional statistics become -1. Malformed numeric data, trailing junk, a present optional statistic that is nonfinite/negative/out of range, or a supplied minimum greater than its supplied maximum rejects the whole packet without refreshing the last valid camera report; old statistics are not paired with a new FPS. This reference change does not relax immediate keepout/early-main checks or turn an observation gap into permission to stretch the cycle.

Numeric values must be finite and within the accepted range. Main destinations must be unique numbered channels of the expected geometry/type; source, output and timing roles cannot alias. Invalid configuration or incompatible resources prevent operation. Unavailable secondary resources are retried once per second while unarmed. If a complete snapshot or destination-copy batch exceeds its runtime budget, the budget grows to the observed duration plus 1 us, the schedule is invalidated, and acquisition/feasibility validation repeat while preserving learned wait corrections. This enlarges the planning window conservatively; a quiet running maximum still does not prove a bound.

## INDI interface

Application properties belong to the configured INDI device: `dmtweeter_synco` in the earlier example, `dmtweeter-synco` in the first operator deployment. The ordinary MagAOX FSM and monitor/telemeter properties are also present. The FSM can remain `READY` while synchronization is acquiring or unarmed; use `health.reason` and the timing summary to determine scheduling state.

| Property | Type/access | Elements and behavior |
| --- | --- | --- |
| `enable` | Switch, read/write | `toggle`: On requests acquisition/operation; Off cancels work and acknowledges after the current callback is quiescent. Source/destination values are retained. |
| `reacquire` | Switch, read/write | `request`: On invalidates the schedule and resets acquisition/feedback; the switch returns Off. It does not zero contributions or enable a disabled app. |
| `settings` | Number, read/write while disabled | `steps`, `maxCadenceHz`, `keepoutUs`, `guardUs`, `step0LeadUs`. Supply one or more named elements; the update is validated and applied together. |
| `timing` | Number, read-only | Main/camera/planned rate, requested/active/feasible counts, delivery/miss/fault/cache counters and maximum snapshot/copy durations, detailed below. |
| `primary_timing` | Number, read-only | Independent publication and receipt interval statistics, detailed below. Durations are seconds, not the microseconds used by the existing `timing` diagnostic elements. |
| `primary_timing_quality` | Number, read-only | Collection/window/queue validity and cumulative observation-quality counts, detailed below. |
| `health` | Text, read-only | `reason`, `lastFault`, `periodSource`, `primary`, `combined`, `timingStream`, `mapping`. Current phase/reason, latched fault, selected period reference and resolved stream identities. |

`settings` uses its named elements directly; there are no numeric `current`/`target` pairs. Updates while enabled, unknown elements, nonfinite values and invalid counts/ranges are rejected without changing the active configuration. Changing `steps` never changes source-image count. After an accepted disabled update, enable again to acquire a new schedule.

The `timing` summary includes:

| Element | Meaning |
| --- | --- |
| `mainHz` | Reciprocal of the selected integer-nanosecond period; it is not a short callback EMA. |
| `cameraHz`, `cameraAgeSec` | Most recently coherently adopted camera rate and its original accepted-reply age; 0 Hz/-1 seconds before adoption of a valid report. A fresh reply does not itself prove ongoing acquisition. |
| `primaryMeasuredHz` | Most recent qualified primary-window rate, separate from the selected period reference; zero before a window is qualified or after a counter gap invalidates it. |
| `cameraJitterUs`, `cameraMinUs`, `cameraMaxUs` | Acquisition-interval standard deviation/minimum/maximum in microseconds; -1 when absent from an accepted packet or not yet available. Present invalid statistics reject the packet. These are diagnostics, not safety bounds. |
| `plannedHz` | Requested count divided by selected period; a planned rate does not prove the schedule is armed. |
| `deliveredHz` | Confirmed main-plus-auxiliary combinations per second over the latest status interval. This is software observation while synco tracks the schedule; zero while disabled does not mean the main AO loop stopped. |
| `delivered` | Cumulative confirmed later-step combinations. |
| `requested`, `active`, `feasible` | Requested count, currently primed count (zero when unprimed), and maximum count fitting the complete timing bounds. |
| `missed`, `faults` | Original timing failures and synchronization/resource invalidations. |
| `cached` | Fallbacks due to an incomplete/racing snapshot; unchanged sources are not counted as misses. |
| `notStarted`, `copiedLate` | Step-0 preparation suppressed before copying or finished after the effective cutoff. |
| `snapshotMaxUs`, `copyMaxUs` | Observed maxima for complete snapshot and destination-copy batches. |
| `copyBudgetUs`, `snapshotBudgetUs` | Current runtime planning budgets, including enlargement after an overrun; distinct from observed maxima. |
| `primaryFaults`, `resourceFaults`, `outputFaults`, `triggerFaults` | Cumulative synchronization/resource faults split by cause; ordinary missed-step/budget reacquisition does not increment these fault totals. |
| `duplicatePosts` | Unchanged-primary-counter callbacks ignored while draining queued notifications. |
| `step0WakeUs`, `step0WakeMaxUs` | Latest/maximum nonnegative copy-attempt lateness relative to the requested copy wake, including checks before the attempt. Only valid, unclamped step-0 wait samples update these values. |
| `step0AdvanceUs` | Current nonnegative advance before the desired step-0 copy start, bounded by `feedback.maxAdvanceUs`. |
| `lastPrimaryDelta` | Counter delta at the latest distinct primary observation; a jump/reset can differ from one. |
| `lastIntervalUs` | Observed interval at the latest distinct primary event, independent of schedule reacquisition; zero before a previous distinct observation exists. |
| `lastOutputDelta` | Combined-output counter minus last confirmed counter at the latest output fault; it is not a continuously refreshed output delta. |
| `lastFaultAgeSec` | Age of the latched synchronization/resource fault, or -1 if none has been recorded. |
| `faultIntervalUs`, `faultPeriodUs`, `faultResidualUs` | Latched faulting primary interval, selected prediction and their difference for an interval anomaly. Later healthy observations do not overwrite them. A subsequent fault other than an interval anomaly clears these interval fields to zero; inspect them together with `health.lastFault`. |
| `faultPrimaryCounter` | Primary counter associated with the latched interval anomaly; zero when absent. |
| `faultPublishUs`, `faultPublishValid` | Same-fault interval between coherent consecutive primary `writetime` values in their own clock, or -1 when unavailable; validity is 0/1. |
| `faultObserveExtraUs` | Same-fault callback interval minus publication interval. This is signed; use `faultPublishValid` to distinguish an unavailable -1 from a valid -1 us difference. |
| `faultPrevCallbackUs` | Immediately preceding callback body's duration, including cycle-mutex wait; -1 when unavailable. |
| `faultCallbackGapUs` | Faulting callback entry minus the preceding callback body's completion; -1 when unavailable. |
| `faultEntryToCounterUs` | Faulting callback entry to its primary metadata sample; -1 when unavailable. |

`health.reason` follows the current phase, for example `acquiring main progression` or `armed`; `health.lastFault` remains latched when that phase changes. A later acquiring status therefore does not erase the cause of a primary/resource/output/trigger fault. Inspect its matching counter and `lastFaultAgeSec` when diagnosing repeated recovery. Missed-step and budget reacquisition can update the current reason without changing that fault latch. For the operator retest, compare snapshots over a known interval: retain the latched fault and age, inspect the per-cause fault increments, and compare step-0 wake lateness/advance with current budgets and copy outcomes. A growing observed maximum alone is not the current budget, and occasional auxiliary `delivered` increments do not demonstrate sustained requested cadence.

The `faultPrimaryCounter` and associated publication/callback fields describe only the latest primary interval anomaly. Ordinary recovery retains them; another synchronization/resource fault clears them. Publication metadata is sampled once with write/counter checks on both sides, without retries. A publication interval requires two stable consecutive counters with positive, normalized, representable and increasing timestamps. Invalid metadata or a counter gap makes that interval unavailable. The immediately preceding callback includes an ignored duplicate notification; its body ends under the cycle lock just before unlock, excluding the final unlock and monitor bookkeeping. Its duration and gap help distinguish callback overrun from delay between callbacks.

For a 555 us callback interval, `faultPublishUs` near 500 us and `faultObserveExtraUs` near +55 us indicate changing observation lag; publication near 555 us indicates an excursion in the producer timestamp interval, subject to its clock contract. These compare intervals within each clock, without subtracting absolute timestamps or changing scheduling phase, bounds or gates. Forward stream-clock steps can affect the publication diagnostic. It cannot measure constant EDT/pipeline delay or camera capture age. These fault-latched fields retain their existing configuration and provider timing layout; the independent cadence aggregates below have separate configuration and telemetry.

Repeated property-definition replies reset `getINDI` query timeouts and can keep a query running even while the app responds. dmSyncopation therefore does not send once-per-second property refresh requests.

The `timing` property's state is Ok when the provider schedule is armed and Idle otherwise. Counts and software output timing do not prove completed hardware transfer. INDI is a low-rate status/control path; providers use the timing image and its notifications.

Aggregate `telem_dmsynco` records include schedule state/count/period, delivery/miss/fault/cache totals, maximum snapshot/copy durations, per-step residuals and learned wait advances. Residuals use actual completion/publication minus the event target; hard-cutoff classification is separate. No 5 us fault-limit configuration exists.

## Independent primary cadence measurements

Primary measurements run while `enable.toggle=Off` as well as while synco is
enabled, acquiring or faulting. They continue when auxiliary/combined resources
are unavailable, provided the primary monitor is receiving commands. They do not
select the scheduling period, change timing bounds, trigger faults, or publish DM
commands. A scheduler disable/reacquire does not reset this independent history.
The monitor's primary attachment changes its timing session.

```text
primary AO command (dm01disp03 / aol1_dmC)
  U: writer's writetime + cnt0
  A: first timestamp after synco's successful semaphore wait
  P: callback entry, sample primary metadata
       |
       +--> existing synco scheduling, if enabled
       |
       +--> fixed timing event --> ordinary app logic --> INDI + dmsynco

publication interval = U[i] - U[i-1], in the stream's own clock
receipt interval     = A[i] - A[i-1], using CLOCK_MONOTONIC
```

Publication intervals require adjacent retained observations with stable metadata
reads, valid consecutive counters and positive, increasing `writetime`. Invalid
timestamps, counter gaps/resets/duplicates, timing-event loss and new attachments
break the pair. Receipt intervals count actual wait returns, including subsequent
rejections and duplicate notifications; their reciprocal mean is a wake-up rate,
not proof of distinct command delivery. An initial existing-image callback has no
fabricated receipt, but can contribute a publication observation.

`primary_timing` has these suffixes for each prefix, `publication_` and `receipt_`:

| Suffix | Meaning |
| --- | --- |
| `fps` | Reciprocal mean interval, Hz, once sample variance is defined. |
| `mean`, `jitter`, `min`, `max` | Mean, sample standard deviation and extrema, **seconds**. |
| `count` | Number of contributing intervals. |
| `span` | Coverage of contributing local observations, seconds. |

For regular 2 kHz publication, means are approximately `0.0005`. Unavailable
durations/FPS are `-1`; an empty count is zero. One interval provides mean/extrema/
span but insufficient evidence for jitter/FPS; variance needs two intervals.
Both observations must be inside the trailing time window and retained population.
The cap can shorten coverage; a stopped input ages out. Each metric has its own
population, so unrelated window standard deviations cannot form a jitter budget.

`primary_timing_quality` has `enabled`, `valid`, `window`, `span`, `age`,
`capacity`, `queued`, `session`, `sequence`, `events`, `receipts`, `synthetic`,
`rejected`, `duplicates`, `counterGaps`, `invalid`, `dropped`. Durations are seconds;
`age=-1` means no retained endpoint. Counters accumulate since configuration.
`dropped` counts a full timing-event queue, not missing upstream frames. `valid=1`
and property state Ok mean both interval populations have sample variance,
independently of scheduler/SDK status. They do not qualify nominal cadence, frame
provenance or safety bounds. Zero gaps alone does not prove full counter coverage.

`telem_dmsynco` appends optional `primary_timing` containing these populations,
coverage, monotonic evaluation timestamp, session/sequence and quality counts.
Human-readable `[dmsynco]` records label `primaryPublicationInterval` and
`primaryReceiptInterval`, using seconds. When collection is enabled, aggregates
are emitted on each ordinary app-logic pass (normally about once per second),
rather than waiting for the ten-second forced-telemetry interval. Statistics and
serialization remain outside the monitor thread and cycle mutex.

The record also carries `schedule_valid`. If the cycle mutex is busy, fresh
primary measurements are still recorded with `schedule_valid=0`; ignore the
scheduler state/counts/feedback placeholders in that record. This flag does not
invalidate primary statistics. Older records without the primary table report it
unavailable; their scheduler fields retain their meaning. Appended optional fields
preserve the binary layout for older readers, but rebuild readers to see the
primary data and honor `schedule_valid=0`; older readers can display placeholders
as scheduler values. With collection disabled, recording follows the ordinary
forced-telemetry schedule.

Metadata consistency checks are best-effort, not proof of token/counter/timestamp/
pixel association. The stream clock can be TAI or REALTIME. Publication intervals
use differences in that clock, with local receipts (entry if unavailable) for
window membership and coverage. External timestamps are never subtracted from
monotonic timestamps. Neither publication-to-wake nor camera-to-DM latency is
established here.

For an operator comparison, keep synco disabled and compare its primary publication
statistics with the DM controller's combined-stream `publicationInterval` over
overlapping windows, inspecting counts/quality alongside jitter/extrema. Variation
already at the primary input suggests investigating upstream; variation appearing
between primary and combined publication suggests that intervening path. Extra
triggers change the combined population, so enabled comparisons also need their
schedule context.

## 200 Hz tip/tilt test provider

The first dmSpeckle provider generates tip/tilt with a physical vibration frequency of 200 Hz, so the WFS can measure it. Its normalized x/y ramps are driven in quadrature:

```text
command(x,y,t) = amp * [x*cos(2*pi*frequency*t) + y*sin(2*pi*frequency*t)]
```

Both axes complete a cycle at 200 Hz; this is not a 200 Hz sample-to-sample toggle producing a 100 Hz waveform. `t` is the target time in monotonic seconds, and `amp` is each axis's edge amplitude. Sampling cadence and vibration frequency are separate settings.

The [dmtweeter_vibration.conf example](../../dmSpeckle/dmtweeter_vibration.conf) selects managed operation:

```ini
[dm]
channelName=dm01disp07_aux
frequency=200
amp=0.01

[modulator]
opMode=tiptilt

[syncopation]
timingStream=dmtweeter_synco_timing
writeLeadUs=50
```

A nonempty `syncopation.timingStream` enables managed mode. The DM channel name must end in `_aux` and match a synco source mapping. The provider chooses the first future cutoff, evaluates the command for its corresponding target and publishes 50 us before the cutoff, revalidating the schedule identity before publication. Invalid/expired schedules pause writes. The producer publishes its source normally; it neither attaches a legacy main trigger nor changes a main destination counter.

Existing dmSpeckle INDI `frequency`, `amp`, `modulating` and `zero` controls apply. Read-only `dm.timing` reports the managed timing stream alongside the existing DM name/channel readback. The legacy trigger is forced off, enabling it is rejected, and its delay setting has no managed effect. Stopping modulation retains the last auxiliary value; use the explicit zero control to remove the contribution.

## Controller clearing and maintenance

Controller zero/release checks each existing numbered main channel for a matching `_aux` counterpart. Missing auxiliary streams are normal. Compatible CPU 2-D streams matching the configured geometry and scalar type are zeroed alongside the main channels. An incompatible or unopenable optional source is left untouched and reported as an error after the main channels and normal cleanup are cleared; it does not block main-channel clearing. Auxiliary clears update publication metadata/counters without semaphore notifications, preserving the controller's existing main-notification behavior.

The controller is responsible for those clear operations; a writer that writes afterward can reapply a contribution. There is no shared provider pause, clear acknowledgement, bridge drain or persistent inhibition protocol. To keep a contribution zero, the operator must manage its writer separately.

Programmatic dmMode integration is deferred. Disable dmSyncopation before independent lab-alignment or other direct-trigger maintenance. Source/destination ownership and the combiner's other trigger-producing inputs remain operational constraints; an unexpected combination is not treated as harmless.

## Operator qualification

1. Verify exclusive destination/source ownership, `spare61` availability, the deployed dmcomb DELAY/counter route and primary monitor semaphore assignment. Confirm the trigger channel is not excluded by astrogrid mode and auxiliary inputs do not unintentionally drive ZPO work.
2. Use the runtime [DM controller timing telemetry](../../../libMagAOX/app/dev/doc/dmTiming.md): `dm_timing`, `dm_timing_stages` and `[dmtimes]` records. Compare the documented processing/reader endpoints and receipt spacing with app residuals and dmcomb output. Use synco's independent primary statistics for the input-to-combined cadence comparison; the legacy `XWC_DMTIMINGS` buffers are no longer needed.
3. Qualify delivery tails, observation lag, early-main uncertainty, snapshot/copy budgets and every adjacent/wraparound/copy window under the relevant load. Keep host-timing qualification separate from deterministic CI tests.
4. Start disabled, verify configuration/status and the selected period source/camera reply age, start the local 200 Hz provider, then enable and wait for lock. Check requested/active/feasible counts, selected/measured rates and delivered cadence.
5. Measure residual mean/RMS/percentiles/max, minimum receipt spacing, skipped/coalesced events and recovery duty cycle. Use 5 us as an analysis guideline. WFS measurements should recover the configured 200 Hz vibration.
6. Disable before changing timing controls or performing independent maintenance. Mapping/topology changes require restart. Controller clearing does not stop active writers.

An arbitrarily early main publication can race a multi-channel memcpy or follow an auxiliary submission too soon. This observer-based app can detect and recover from such a violation but cannot block the genuine main publisher. A hard gating/pipelining guarantee requires a separate change to the cooperating loop/controller path.
