# dmSyncopation user guide

`dmSyncopation` copies auxiliary deformable-mirror contributions into existing dmcomb input channels at a fixed integer subdivision of the main AO-loop period. One instance serves one DM. The initial instance, `dmtweeter_synco`, requests three steps per main cycle: approximately 6 kHz total cadence when the main loop runs at 2 kHz.

The [engineering plan](../../../agents/plans/2026-10/dm-syncopation.md) records settled design decisions; the [debugging record](../../../agents/plans/2026-10/dm-syncopation-debugging.md) records operator trials and local fixes. The trials have not yet established sustained requested cadence, and the measured-acquisition reference changes described here need operator retesting. Initial timing values are starting budgets for qualification, not measured instrument bounds. All live instrument setup, INDI actions and timing measurements are performed by the operator.

## Read-only schedule simulation

`simulation.enabled=true` selects a startup-only observer of the proposed
long-baseline cadence, counter-labelled scheduler. It runs automatically while
`enable.toggle` is Off and rejects requests to turn scheduling On. It opens
the existing primary and combined streams for observation and records
conditional **would-run/skip** decisions. It creates no auxiliary source or
provider timing stream, copies no destination pixels, changes no command
counters, and sends no DM triggers. Normal monitor semaphore registration and
notification consumption still occur.

```text
primary counter + monotonic time ---> 10-second cadence (refresh 1 Hz)
                                                     |
primary observation history -------> median phase + frozen future grid
                                                     |
existing combined trace -----------> actual main/output association
                                                     |
configured timing bounds ----------> evaluate each fixed step independently
                                                     |
                                     simulation INDI + [dmsynco] telemetry

                    No path from this model to DM command streams
```

This is distinct from `diagnostics.only`, which observes an arbitrary primary
without a combined DM stream. Simulation needs the real primary/combined
pair, and its combined observer remains enabled. The ordinary scheduler
behavior described in the following sections is unchanged; the simulation
does not enable the proposed grid policy for real commands.

The non-default starting configuration is available in
[dmtweeter_synco_sim.conf](../dmtweeter_synco_sim.conf):

```ini
[shmimMonitor]
shmimName=dm01disp03

[scheduler]
maxSubframes=2

[simulation]
enabled=true
```

Configure the actual primary filesystem image; `aol1_dmC` may be its alias.
The INDI device can retain the historical name `dmtweeter-synco`; any shared
memory names continue to use underscores. Simulation creates no
`dmtweeter_synco_timing` stream for a provider. An existing provider timing
stream from an earlier run is not refreshed by this mode.

The model estimates period from primary counter/monotonic-observation
endpoints approximately ten seconds apart, refreshing at most once per second.
It divides elapsed time by the actual counter advance, including forward
gaps. A fixed 64-anchor ring holds approximately one-second samples. At each
refresh it selects the newest retained anchor at least the requested window
old; while growing, it uses the oldest available anchor. Individual producer
publication intervals remain independent diagnostics and do not steer cadence.
Phase uses the existing 32 coherent counter/observation pairs, gain 0.1 and
one-microsecond correction limit. A
main keeps its genuine counter-labelled cycle even when it crosses a grid
boundary. Each phase residual uses a prediction frozen before that main was
observed. Phase-window warmup, missing or ambiguous combined traces, gaps,
and reader holds can leave steps unavailable. A skipped passive preparation
does not itself suppress a later active slot; that slot still requires its
actual main/output association and reader-safe preparation window. A late
main does not slide an active target.

Nominal model feasibility also respects `scheduler.maxCadenceHz` with the
existing `tracking.rateTolerance`, as well as the complete reader/copy timing
window. Simulation does not bypass the requested total-cadence cap.

The cadence starts with `tracking.nominalMainRateHz`, obtains its first
measured rate after one second, and grows toward the full ten-second baseline.
A grid can become valid after phase warmup while cadence is still nominal.
The read-only `simulation_cadence.state` distinguishes nominal (0), growing
(1), and full (2). Primary silence beyond twice the period at
`tracking.minMainRateHz` (4 ms with the default minimum) clears cadence and
phase anchors once. A one-second cadence refresh is not mistaken for stale
primary data. Duplicate notifications refresh neither the anchors nor primary
freshness. Backward primary counters, regressing primary observation time,
attachment replacement, and configuration changes start new epochs. A fresh
camera FPS report differing from the accepted rate by more than
`tracking.rateTolerance` (1% by default) also starts a new epoch and supplies
its nominal seed; ordinary measurement noise does not reset the estimate.

Cadence and frozen plans retain fractional nanoseconds as integer Q32 values
(`period_q32 / 4294967296` ns). Each plan preserves the fractional remainder
when extrapolating to the next counter and rounds only the final target to
nanoseconds. Integer `period_ns` remains a rounded diagnostic; feasibility
uses the conservative floor of the exact period. Previously committed targets
remain frozen across cadence refreshes and phase corrections.

Would-run counts assume complete source data, the configured snapshot/copy
budgets, and synthetic auxiliary submissions consumed under the configured
delivery and reader-hold allowances. No actual auxiliary transaction tests
those assumptions. Combined traces must be coherent and linked to the
configured primary; an unlinked/stale trace cannot establish a reader-safe
main-output association. Conservative holds may therefore produce skips even
when a physical delivery could have succeeded.

The physical early-main reserve remains undecided. Simulation uses the
configured `tracking.earlyMainUs` value, **20 us by default**, as an explicit
assumption; would-run counts do not qualify it. Observation phase includes
monitor wake lag, cadence/pipeline callback bookkeeping before the metadata
sample, and any cycle-mutex wait. It is not an exact primary publication phase. Compare the
independent publication/receipt timing diagnostics and qualify clock/lag
uncertainty before interpreting negative phase tails as physical earliness.
An early main can cancel future modeled work but cannot protect an auxiliary
command already issued by a future operational implementation.

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
| `shmimMonitor.cpuset` | `dmtweeter-synco` | Primary monitor/cycle thread cpuset; CPU41 in the RTC layout. |
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
| `simulation.enabled` | false | Startup-only read-only grid simulation. Automatically observes with scheduler enable Off and rejects enable On; creates no auxiliary/provider streams or command writes. |
| `simulation.window` | 32 | Counter/observation pairs in the median phase filter, 4–256. Also bounds the independent qualified producer-interval diagnostic count. |
| `simulation.cadenceWindowSec` | 10 | Requested counter/time cadence baseline. At least the refresh interval, at most 1,000 seconds and at most 63 refresh intervals. Actual selected endpoint span is reported. |
| `simulation.cadenceUpdateSec` | 1 | Minimum time between cadence refreshes, 0.01–60 seconds. The first measured estimate requires this much primary history. |
| `simulation.phaseGain` | 0.1 | Positive simulation phase correction gain, at most 1; affects only a future uncommitted cycle. |
| `simulation.phaseSlewUs` | 1 | Maximum simulation phase correction per future cycle. Nonnegative and at most one quarter of the shortest accepted primary interval (62.5 us with the default maximum primary rate). Does not change any physical timing reserve. |
| `diagnostics.windowSec` | 2 | Elapsed-time window for all independent observers; zero disables collection. |
| `diagnostics.maxEvents` | 20000 | Queue and retained-event cap for each observer; zero disables collection. Storage is allocated before the monitors start. |
| `diagnostics.excursionUs` | 20 | Camera acquisition/publication and primary excursion threshold around each diagnostic window median, in microseconds; zero disables pair/conditional classification. Ordinary trace timing remains enabled. Independent of scheduler timing bounds. |
| `diagnostics.combiner` | true | Start an independent combined-output observer for cadence and process-trace timing. Scheduling can remain disabled. |
| `diagnostics.camera` | true | Start an independent metadata-only camera observer for separate acquisition and publication timestamp intervals. |
| `diagnostics.pipeline` | true | Capture up to ten existing upstream process-trace rows at primary observation. |
| `diagnostics.only` | false | Observe the configured primary without attaching scheduler DM resources or creating a timing stream. Forces combined monitoring off, selects an available primary semaphore automatically, and rejects scheduler enable requests. |
| `cameraMonitor.shmimName` | `camwfs` | Camera image whose `atime` and `writetime` intervals are analyzed separately. Independent of the `tracking.cameraDevice` INDI subscription. Semaphore selection is automatic by default. |
| `cameraMonitor.threadPrio` | 0 | Camera metadata observer scheduling priority; zero uses ordinary scheduling. |
| `cameraMonitor.cpuset` | empty | Optional camera metadata observer CPU placement. |
| `combinerMonitor.threadPrio` | 0 | Combined observer scheduling priority; zero uses ordinary scheduling. |
| `combinerMonitor.cpuset` | empty | Optional observer cpuset. Choose an available CPU placement on the instrument. |
| `combinerMonitor.shmimName` | `streams.combined` | Forced to the scheduler's configured combined output; a separate value is ignored. Semaphore selection is automatic. |
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
| `primary_interval_pairs` | Number, read-only | Ordered primary publication-interval excursions, conditional successors and two-interval sums; durations are seconds. |
| `combiner_timing` | Number, read-only | Independent combined cadence and producer-trace duration populations, in seconds. |
| `combiner_timing_quality` | Number, read-only | Combined-observer coverage, writer/trigger identity and trace rejection/loss counters. |
| `camera_timing` | Number, read-only | Independent `acquisition_` and `receipt_` interval populations, in seconds. |
| `camera_timing_quality` | Number, read-only | Camera coverage, queue state, gaps, rejections and losses. |
| `camera_interval_pairs` | Number, read-only | Independent acquisition excursion ordering around its own window median. |
| `camera_write_timing` | Number, read-only | Independent camera `publication_` and observer `receipt_` interval statistics, in seconds. |
| `camera_write_timing_quality` | Number, read-only | Camera publication coverage, gaps, rejected metadata and collection losses. |
| `camera_write_interval_pairs` | Number, read-only | Camera publication excursion ordering around its own retained-window median. |
| `pipeline_timing_quality` | Number, read-only | Primary excursion classification, trace stage count, coverage and qualification counters. |
| `pipeline_stage0` ... `pipeline_stage9` | Number, read-only | Trace-row identity, ordinary processing/handoff durations and excursion-conditioned changes. Trace rows are unrelated to synco substep numbers. |
| `pipeline_boundary_timing` | Number, read-only | Oldest represented stage-start intervals and signed changes in the primary publication tail. |
| `simulation` | Number, read-only | Simulation estimator validity, period, frozen grid/phase residuals, configured early reserve and observation-quality counts. Durations use the units in their element names. |
| `simulation_steps` | Number, read-only | Per-step conditional considered/would-run/skip/overlap counts. These are modeled outcomes, not DM deliveries. |
| `simulation_status` | Text, read-only | Current simulation state and explicit assumptions/limitations. |
| `simulation_trace` | Number, read-only | Bounded overlap-event FIFO coverage and losses, detailed below. |
| `simulation_estimator` | Number, read-only | Triggered estimator pre/post history, export coverage and losses. |
| `simulation_cadence` | Number, read-only | Long-baseline cadence qualification, exact-period display, endpoint coverage and independent primary/refresh ages. |
| `health` | Text, read-only | `reason`, `lastFault`, `periodSource`, `primary`, `combined`, `timingStream`, `mapping`, `mode`. Current phase/reason, latched fault, selected period reference and resolved stream identities. Mode also identifies simulation when selected. |

`settings` uses its named elements directly; there are no numeric `current`/`target` pairs. Updates while enabled, unknown elements, nonfinite values and invalid counts/ranges are rejected without changing the active configuration. Changing `steps` never changes source-image count. After an accepted disabled update, enable again to acquire a new schedule.

Simulation is selected in the startup configuration, not through an INDI toggle.
Accepted disabled `settings` changes restart its model; observations then resume
automatically with operational enable Off. Its numeric `simulation` property
contains:

| Elements | Meaning |
| --- | --- |
| `enabled`, `valid`, `feasible` | Simulation selected, fresh predicted grid available, and nominal complete-cycle feasibility. A valid grid can still have unavailable individual steps. |
| `session`, `sequence` | Simulation reset/identity epoch and distinct accepted primary-observation count. |
| `periodUs`, `periodSamples`, `phaseSamples` | Fractional cadence period in microseconds, broadly qualified producer-interval diagnostic count, and retained phase observation count. Producer intervals no longer set the period. |
| `gridNs`, `nextGridNs`, `nextActiveNs` | Frozen/current and next predicted targets in same-host monotonic nanoseconds; zero means unavailable. These are diagnostic targets, not provider deadlines. |
| `phaseUs`, `phaseMinUs`, `phaseMaxUs`, `phaseCount` | Latest, session minimum/maximum and count of observation residuals against predictions committed before those observations. Negative means observed earlier than predicted. |
| `earlyReserveUs` | Configured early-main assumption, explicitly unqualified by simulation. |
| `duplicates`, `gaps`, `resets`, `invalid`, `excluded` | Unchanged-counter notifications, forward gaps, resets, rejected observations, and producer intervals excluded from diagnostics. An excluded interval does not itself become a synchronization fault or exclude genuine counter advances from cadence. |
| `outputGaps`, `associations` | Missing/ambiguous combined progression and accepted coherent actual main/output associations. |

For each configured index `j`, `simulation_steps` exposes
`step{j}Considered`, `step{j}WouldRun`, `step{j}Skipped`, and `step{j}Overlap`.
Detailed skip-reason counts are retained in the appended `[dmsynco]`
simulation table. The text `simulation_status.state` describes warmup or the
current conditional state, and `policy` identifies the modeling assumptions.
Use simulation fields for modeled outcomes; operational `timing.delivered`
does not count hypothetical submissions. The requested count remains in
`settings.steps`; the full assumed timing bounds and estimator settings are
recorded in the appended telemetry table.

The `simulation_cadence` property exposes:

| Elements | Meaning |
| --- | --- |
| `state`, `epoch` | Nominal (0), measured growing baseline (1), or full baseline (2), and process-lifetime cadence epoch. Primary silence can change this epoch within one model session. |
| `periodNs` | Fractional nanoseconds per cycle, displayed from Q32. |
| `spanSec`, `cycles`, `anchors` | Actual measured endpoint span, genuine counter advances in that span, and currently retained slow anchors (at most 64). Span/cycles are zero before the first accepted measurement. |
| `updates`, `rejected` | Accepted/rejected endpoint measurements in this cadence epoch. Rejected measurements leave the prior estimate unchanged. |
| `windowSec`, `updateSec` | Configured baseline and minimum refresh interval. |
| `observationAgeSec`, `refreshAgeSec` | Ages of the latest coherent distinct primary and accepted cadence update. A value of −1 means unavailable. A healthy refresh age approaching one second does not imply stale primary data. |

Each new `[dmsynco]` simulation summary includes `simulation_cadence={...}` with
the same state/coverage, exact `period_q32` and `period_scale`, plus monotonic
`observation_ns` and `updated_ns`. Older records omit this optional table.

An overlap count is retrospective main-observation evidence inside a
previously modeled copy or active reader-hold interval. It does not undo the
would-run count. The combined observer can supply evidence before the primary
callback, and that later callback can count the same main again. These counts
are therefore not deduplicated physical collision rates. Retain their
before/after differences even when most increments precede the first saved
telemetry row; observation lag and the assumed timing bounds still need
qualification.

Simulation automatically retains timestamped overlap evidence in a fixed
256-event FIFO. The ordinary application thread drains at most 64 events per
`[dmsynco]` record, after releasing the model lock; callbacks only copy fixed
records. Each `sim_overlap={...}` entry identifies:

| Fields | Meaning |
| --- | --- |
| `id`, `session` | Process-lifetime evidence sequence and model session owning the event. Pending events survive model resets and can belong to an earlier session than the current summary. |
| `planEstimatorId`, `lastEstimatorId` | Estimator update that created the modeled copy's frozen plan, and latest completed estimator update before the overlap evidence. Zero means unavailable in an older record. |
| `origin`, `window` | Observer (`primary` or `combined`) and containing modeled interval (`copy`, active `readerHold`, or `backend` in release-readiness shadow mode). Boundaries include the start and exclude the end. Passive step 0 has no reader hold or its own backend operation. |
| `mainCounter`, `combinedCounter`, `planCounter`, `step` | Actual main supplying evidence, observed combined counter, and cycle/step of the last modeled copy. On the primary path, the combined counter is the most recently observed one, possibly zero. |
| `mainGrid_ns`, `planGrid_ns`, `period_ns` | Committed grid or extrapolation from the last frozen cycle for the evidence main, frozen main target for the modeled copy's cycle, and that cycle's period. Zero main-grid means unavailable. |
| `period_q32`, `planFraction_q32` | Exact frozen period and signed subnanosecond remainder of `planGrid_ns`, scaled by 2^32. New records retain these alongside rounded period diagnostics. |
| `observed_ns`, `copyBegin_ns`, `copyEnd_ns`, `readerEnd_ns` | Evidence time and modeled copy/reader boundaries on the same-host monotonic clock. Passive `readerEnd_ns` is zero. |
| `backendEnd_ns` | Optional modeled backend retirement boundary for an active step in release-readiness mode. A backend conflict keeps that active command's provenance even after a later passive preparation copy. Zero/absent in legacy events. |
| `lastMainCounter`, `lastMain_ns`, `lastOutput_ns` | Previously accepted primary identity/time and latest associated combined observation. |
| `streamInode`, `triggerInode` | Observed stream identity and combined trace's named primary identity when available. |
| `receipt_ns`, `callback_ns`, `lockRequest_ns`, `lockAcquired_ns` | Monotonic receipt, callback entry and model-lock endpoints. Zero means unavailable. Their differences separate callback bookkeeping and lock wait from model observation. |
| `publication_ns`, `primaryPublication_ns`, `start_ns`, `update_ns` | Raw producer-clock stream/primary publication and combined trace endpoints. Primary publication is supplied only when its counter matches the coherent primary metadata already observed. |

Within one app process, group events by `(session, planCounter, step, mainCounter)`
to identify repeated evidence for the same modeled step and actual main,
while retaining origin and event IDs. Keep process restarts separate. The
primary and combined observers can both report one main.
Release-readiness mode also reports a main arriving after modeled readers clear
but before the active backend retires. The [three-step shadow procedure](readiness-shadow-test.md)
explains its independent reader and backend assumptions.
The log labels model/observer times `model_clock:monotonic` and producer times
`producer_clock:unconverted`. Do not subtract absolute producer timestamps
from the monotonic grid. The telemetry record's wall-clock timestamp is its
emission time, not the individual evidence time.

The read-only `simulation_trace` elements and telemetry `trace_quality` expose
`enabled`, `detected`, `drained`, `dropped`, `queued`, `capacity`, and
`logFailed`. Counts persist for the app process, across model resets:
`detected = drained + dropped + queued`. A full FIFO drops new evidence
without changing model decisions or aggregate overlap counts. `drained`
means removed for a telemetry attempt; `logFailed` counts those records lost
to a detected encoding/submission failure. It does not attest to disk
durability or detect filtering by the general logger. A busy model lock
defers draining until a later record. `trace_quality` is absent from an older
record or from a record where no queue snapshot could be taken. Inspect
losses and pending events before interpreting a capture as complete.

An overlap also selects estimator history for the same `[dmsynco]` telemetry:
up to 256 preceding completed primary updates, followed by 64 updates after
the latest overlap. At 2 kHz these cover up to 128 ms before and 32 ms after
an isolated trigger. Primary-origin evidence is detected before its own
estimator update completes, so that update is the first post-trigger sample.
Nearby triggers extend the post window, with each selected sample exported
once by process-lifetime ID. Model resets clear the rolling history/post
window but preserve queued samples and their original sessions.

Each `sim_estimator={...}` record contains:

| Fields | Meaning |
| --- | --- |
| `id`, `session`, `sequence`, `counter` | Process sample ID, owning model session, accepted-primary sequence and genuine main counter. |
| `observed_ns`, `publicationInterval_ns`, `sequential`, `intervalAccepted` | Monotonic observation, raw producer-clock elapsed interval, consecutive-counter qualification and whether the interval entered the period ring. Zero interval means unavailable; an excluded raw interval can be signed. |
| `periodSamples`, `phaseSamples`, `periodBefore_ns`, `period_ns` | Producer diagnostic/phase sample counts and rounded period before/after this update. |
| `predicted_ns` | Current main's frozen prediction, made before observing that main; zero while unavailable. |
| `nextCounter`, `sourcePlanCounter`, `sourceEstimatorId`, `sourceGrid_ns` | Attempted future commitment and the frozen plan from which its base was extrapolated. No attempt is represented by zero future fields. |
| `baseGrid_ns`, `candidate_ns`, `phaseGain`, `phaseSlew_ns` | Uncorrected future target, median phase candidate, gain and correction limit used in the existing calculation. |
| `rawCorrection_ns`, `appliedCorrection_ns`, `committedGrid_ns`, `committed` | Requested correction truncated toward zero to integer nanoseconds, applied bounded correction and successful new future commitment. A zero correction is valid. |
| `periodBefore_q32`, `sourceFraction_q32`, `baseFraction_q32`, `committedFraction_q32` | Exact previous period and signed source/base/committed grid remainders. Each remainder is in [−2^31, 2^31) scaled nanoseconds. |

Each new sample is followed by `estimator_cadence={...}`, carrying its own
cadence epoch, endpoint span/count and exact selected period. The logger checks
the source-to-base extrapolation with signed 128-bit Q32 arithmetic. Replaying
only rounded `period_ns` would discard fractional carry and can falsely suggest
an arithmetic mismatch. Older samples retain their original integer arithmetic
contract. Queued samples can belong to earlier epochs than the current summary.

Join an overlap's `planEstimatorId` to the estimator sample's `id`, within
the same app process. That sample's committed future counter/grid identifies
the frozen copy plan. `lastEstimatorId` identifies the last completed update
at evidence time; it can differ from the update that created the plan.
Independent drain limits mean related records can appear in different
telemetry emissions. Preserve all records and their IDs across the capture.
These diagnostics record values from the original calculations; they do not
recalculate medians or change targets. All absolute grid/observation times
are monotonic. The producer interval is an elapsed duration, labeled
`interval_clock:producer_elapsed`, rather than a converted absolute timestamp.

Read-only `simulation_estimator` and telemetry `estimator_quality` expose
`enabled`, `observed`, `triggers`, `selected`, `drained`, `dropped`, `queued`,
`capacity`, `logFailed`, `history`, `historyCapacity`, and `postRemaining`.
`observed` counts every completed accepted-primary update; `selected` counts
unique samples chosen by triggered windows, including export losses.
`selected = drained + dropped + queued` holds for the app process. The fixed
export FIFO holds 4,096 samples and ordinary telemetry drains at most 512
per record. A full FIFO drops new diagnostic samples, leaving model
decisions unchanged. `logFailed` has the same submission-only meaning as
the overlap trace's counter. An omitted optional quality snapshot is not a
zero-event report.

The bounded history can begin after the disturbance that originally shifted
the phase. It is shorter than the ten-second cadence baseline; retained sample
history alone cannot reconstruct all slow endpoints. Retain the recorded exact
period, cadence span/count, candidate and correction fields when diagnosing
such cases.

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

## Ordered primary interval excursions

`primary_interval_pairs` tests whether a long publication interval is immediately
followed by a short one, or the reverse. It uses existing primary timestamps in
the ordinary statistics pass, independently of scheduling and its cycle mutex.
No extra monitor-thread timestamp reads, pixel work or primary publications are
added. It collects with synco scheduling disabled.

The reference is the median of eligible publication intervals in the retained
diagnostic window; for an even population, it is the average of the two middle
values. This diagnostic median does not change the scheduler's period selection.
An interval is **late** only when it exceeds `median + threshold`, and **early**
only when it is below `median - threshold`. Equality is ordinary. The default
threshold is 20 us, set by `diagnostics.excursionUs`; this is separate from
`tracking.earlyMainUs` and any proposed early-arrival reserve. Classification
describes interval length, not absolute phase relative to a camera exposure.

The property has `enabled`, `median`, `threshold`, `intervals` and `pairs`, plus
`mean`, `jitter`, `min`, `max`, `count`, `span` suffixes for four prefixes:

| Prefix | Population |
| --- | --- |
| `afterLate_` | Next interval after each eligible late interval, including ordinary successors. |
| `afterEarly_` | Next interval after each eligible early interval, including ordinary successors. |
| `lateEarlySum_` | Sum of two consecutive intervals classified late then early. |
| `earlyLateSum_` | Sum of two consecutive intervals classified early then late. |

All durations, including `median` and `threshold`, are **seconds**. Counts are
zero when empty; unavailable durations are `-1`, and jitter needs two samples.
Property state Ok means at least one eligible interval pair exists, not that an
excursion was seen or that its variance is available. `intervals` counts all
qualified intervals and `pairs` all qualified adjacent interval pairs, including
ordinary ones. Each pair requires three real, ordered observations with adjacent
event sequences/counters and valid increasing publication timestamps wholly
inside the retained window. Gaps, resets, duplicates, invalid/rejected/synthetic
observations and collection loss break adjacency. Coherent metadata at a break
can establish a fresh baseline; no interval is divided across a counter gap.

The conditional late-to-early fraction is
`lateEarlySum_count / afterLate_count`; the reverse is
`earlyLateSum_count / afterEarly_count`. A zero denominator means no eligible
follow-up evidence. An interval at the end of a window without an observed
successor is not included in either denominator. For a 500 us median, sums near
0.001 seconds support two-interval compensation. For example:

```text
primary_pairs={enabled:1,intervals:3999,pairs:3998,median_s:0.0005,threshold_s:0.00002,lateEarly:12/12,earlyLate:0/12}
```

This would mean every one of twelve eligible late intervals had an immediate
early successor in that window; it is not proof that this always happens.
`[dmsynco]` also reports full `primaryAfterLateInterval`,
`primaryAfterEarlyInterval`, `primaryLateEarlySum`, `primaryEarlyLateSum`
populations. These are overlapping trailing-window measurements, not cumulative
counters; do not sum counts across successive reports. Opposite-order pairs can
overlap in a late–early–late sequence.

The pair table is an appended optional part of primary timing. Rebuild the app
and `logdump` to record/display it. Older records report
`primary_pairs=unavailable`; disabled analysis reports `enabled:0`. Setting
`diagnostics.excursionUs=0` leaves existing primary/combiner statistics enabled.
No scheduler recovery, trigger route or timing limit is changed by this check.

## Independent combiner trace timing

A second `shmimMonitor` attaches to `streams.combined` and reads only metadata
and the first two `STREAM_PROC_TRACE` entries. It runs independently of the
primary scheduling thread, cycle mutex and auxiliary resources. It performs no
pixel copies, DM writes, producer reconfiguration, statistics or logging in its
callback. Both collectors hand fixed observations to ordinary app logic through
separate bounded queues. The combined observer starts at ordinary scheduling
priority; its `combinerMonitor` controls allow operator-selected CPU placement.
Its ordinary monitor properties use the `comb_` prefix.

The inspected MILK producer already supplies these external-clock endpoints:

```text
U  primary's trace-update timestamp, copied into combined trace entry 1
             |
             | primaryStart
             v
S  dmcomb iteration execution start (after its wait)
             |
             | execution: counter checks, optional work, summation, trace setup
             v
T  dmcomb trace-update timestamp in trace entry 0
             |
             | publication: remaining trace setup through ImageStreamIO timestamp
             v
W  combined stream writetime (before counter/notification publication)

primaryPublication spans U -> W. All external durations stay in that clock.
The observer's own monotonic wait return supplies coverage and receipt intervals.
```

These endpoints measure the existing producer trace; `execution` is not an
isolated summation benchmark, `publication` excludes subsequent semaphore posts,
and `primaryStart` is not the requested poll sleep. `U` is the primary producer's
trace-update timestamp, rather than its later `writetime`. Comparisons require the
producer trace and ImageStreamIO timestamps to use the same clock contract. A
TAI/REALTIME mismatch, clock step or broken ordering cannot be converted into
latency by combining it with local monotonic timestamps.

`combiner_timing` has six prefixes, each with `mean`, `jitter`, `min`, `max`,
`count`, `span` suffixes separated by an underscore:

| Prefix | Population |
| --- | --- |
| `publicationInterval` | Consecutive observed combined `writetime` differences. |
| `receiptInterval` | Actual combined observer semaphore-receipt intervals. |
| `execution` | `S` to `T` for a distinct output with an advancing process-start timestamp. |
| `publication` | `T` to `W` for that qualified trace observation. |
| `primaryStart` | Linked primary trace update `U` to `S`. |
| `primaryPublication` | Linked primary trace update `U` to `W`. |

Durations/coverage are **seconds**, empty counts are zero and unavailable durations
are `-1`. Each population has independent count/variance availability. Whole
property state Ok means at least one population has sample variance; it does not
assert that every trace population is available. `[dmsynco]` appends the optional
`combiner_timing` table with these populations, named `combinerPublicationInterval`,
`combinerReceiptInterval`, `combinerExecution`, `combinerPublication`,
`combinerPrimaryStart`, `combinerPrimaryPublication`, and each population's `valid`
flag. Rebuild the app and reader; legacy records without this table report it
unavailable. Combined statistics also record when the scheduler snapshot is busy.

`combiner_timing_quality` reports `enabled`, `window`, `span`, `age`, `capacity`,
`queued`, `session`, `sequence`, `writerPid`, `triggerInode`, `primaryCounter`,
and cumulative `events`, `receipts`, `synthetic`, `rejected`, `duplicates`,
`counterGaps`, `invalidMetadata`, `missingTrace`, `invalidTrace`, `repeatedStart`,
`unlinkedPrimary`, `invalidPrimary`, `repeatedPrimary`, `dropped`. Missing or
unusable traces do not suppress otherwise valid cadence observations. Epochs,
losses and discontinuities break interval/trace freshness. Repeated process-start
timestamps can indicate disabled MILK timing or multiple outputs from one
iteration; they are excluded rather than treated as new execution measurements.
Freshness requires a baseline for each writer/attachment and linked primary.
Both the linked counter and its upstream trace-update timestamp must advance;
a changed upstream writer starts a new baseline. `primaryCounter` displays the
trace's named input counter and identifies a primary only when `triggerInode`
matches that stream.

Metadata and trace checks remain best-effort. The inspected dmcomb writes output
and trace without bracketing the entire operation with `write=1`. Stable repeated
reads reduce detectable mixing but cannot establish atomic frame/pixel identity.
The producer samples `trace[0].cnt0` at output update; it does not record the exact
primary counter consumed at iteration start. In a linked trace this counter is
also different from `trace[1].cnt0`, which identifies the primary writer's own
upstream trigger. The linked duration populations are conditioned on timestamp
ordering and advancing identities, not a proven per-command latency series.
Do not subtract their marginal variances from other populations.

### Configuration experiments

The inspected dmcomb supports these FPS controls; the operator changes them on
the instrument and records their actual values with each comparison:

| Control | Effect and measurement consequence |
| --- | --- |
| `.procinfo.triggersname` | Names the stream used for trace linkage. Pointing it to the real primary (`dm01disp03` / `aol1_dmC`) supplies its inode/trace even while retaining DELAY mode. This is a startup/configuration setting and requires a dmcomb restart. |
| `.procinfo.triggermode` | Mode 4 retains sleep plus all-channel counter scanning; mode 3 waits on one stream's semaphore. A switch to one primary semaphore would not wake for every auxiliary-only counter update and needs a compatible trigger design. |
| `.procinfo.triggerdelay` | Requested DELAY-mode sleep, in timespec units. Actual wake behavior includes scheduler effects; the configured value is not a measured delay or jitter floor. |
| `.procinfo.RTprio`, `.procinfo.cset`, `.procinfo.taskset` | Producer scheduling and CPU placement. Record actual thread policy/affinity as well as requested configuration. |
| `.procinfo.semindexrequested`, `.procinfo.triggertimeout` | Semaphore-mode reader allocation and timeout controls; they do not replace DELAY-mode all-channel scans. |

MILK's processinfo timing measurement must be enabled for `ts_procstart` to
advance. `MeasureTiming` is a processinfo/CLI control in the inspected checkout,
not an FPS key named `timing.Measure`. Also, that checkout does not populate the
trace `triggermode` field correctly; the observer does not infer current trigger
mode from it. Obtain actual mode/delay from FPS configuration.

Start with scheduling disabled and the current dmcomb settings. Verify combined
cadence plus `execution`/`publication` counts; `unlinkedPrimary` is expected when
the trigger stream is unnamed or different. If upstream trace linkage is needed,
configure the primary trigger-stream name while retaining DELAY mode, restart
dmcomb, then verify inode linkage, freshness and linked-population counts. Compare
one configuration change at a time over the same workload, inspecting counts,
tails, rejections and minimum spacing alongside RMS. No configuration changes are
made automatically, and these measurements do not relax synco's timing bounds.
Naming the trigger also adds the producer's upstream trace-copy work; include that
configuration change in the comparison record.

## Independent camera acquisition and publication excursions

A third metadata-only monitor reads `cameraMonitor.shmimName`, normally `camwfs`.
Its default scheduling priority is ordinary. It never copies image pixels,
changes camera settings or writes/posts to the stream. It runs while synco
scheduling is disabled and while auxiliary resources are unavailable. Its helper
properties use the `cam_` prefix.

The acquisition interval is **`atime[i] - atime[i-1]`**. A separate collector
measures **`writetime[i] - writetime[i-1]`** from the same coherent metadata
snapshot. Each has its own median, qualification history and population;
neither is the observer's wake interval. `camera_timing` uses the same suffixes as
`primary_timing`, replacing `publication_` with `acquisition_`; `receipt_` remains
the observer's independent monotonic wake population. `camera_write_timing`
uses `publication_` and `receipt_`; its quality and pair properties use the same
elements as their primary counterparts. The two receipt populations describe the
same observer notifications and must not be counted as independent measurements.
No acquisition timestamp is subtracted from a publication or DM timestamp.

Inspect `acquisition_min` and `acquisition_max` to detect occasional long/short
intervals even when RMS is small. The appended `[dmsynco]` camera table records
these extrema, count, coverage and quality, unlike the existing camera
`[fgtimes]` record, which contains only means and jitters. Text populations are
`cameraAcquisitionInterval`, `cameraReceiptInterval`, `cameraAfterLateInterval`,
`cameraAfterEarlyInterval`, `cameraLateEarlySum` and `cameraEarlyLateSum`, with
`camera_pairs` reporting the directed fractions.

The appended optional publication table reports `cameraWriteInterval`,
`cameraWriteReceiptInterval` and `cameraWrite_pairs`, with conditional successor
and sum populations prefixed `cameraWrite`. It records publication extrema,
counts, coverage and quality separately from acquisition. A malformed or
nonprogressing publication stamp rejects that collector's sample without
discarding an otherwise valid acquisition stamp, and vice versa. Both histories
reset on camera reattachment. Rebuild the app and reader to see this table;
older data reports `cameraWrite_timing=unavailable`.

The ordinary-priority observer can miss notifications. Counter gaps, duplicates,
queue drops, rejected/unstable metadata and missing interval coverage prevent a
claim that acquisition had no excursions during the entire test. Positive
nonincreasing acquisition timestamps for consecutive observed counters are
rejected, as are malformed stamps; they do not become valid long/short samples.
An excursion at the end of a window still affects interval extrema even before
its successor is available for the pair statistics. Attachments and broken
observation chains restart associations.

## Locate excursions within the upstream AO trace

The primary callback now copies at most ten populated existing process-trace
rows into a bounded queue before acquiring the cycle mutex. It takes no extra
clock readings and performs no statistics, logging or stream writes for this
capture. Ordinary app logic computes the timing populations. This is separate
from the combined observer, which continues to inspect its first two rows.

The trace is ordered backwards from the primary writer:

```text
camera acquisition (independent atime check)
    ... camera publication / first-stage wake may precede represented trace ...
oldest represented stage start -> work -> upstream update
                                           |
                                        handoff
                                           v
                                  next stage start -> work -> update
                                                               |
                                                          ... handoffs ...
                                                               v
                                      primary writer start -> update -> writetime
```

`pipeline_stage0` identifies the primary writer; higher row numbers identify
earlier upstream writers. For the previously observed AO chain, these correspond
to DM-command conversion, filtering, reconstruction and WFS preparation,
respectively. `writerPid` and `inputInode` identify the current row; the inode is
its **trigger input**, not its output. Stream names and PIDs may change after a
restart. The stage number is not synco's passive/active step number.

Each row has `present`, `row`, `writerPid`, `inputInode` and these prefixes, with
`mean`, `jitter`, `min`, `max`, `count`, `span` suffixes:

| Prefix | Population |
| --- | --- |
| `execution_` | Stage start to its trace-update timestamp, including any waiting within that work. |
| `handoff_` | Next upstream row's update to this stage's start; unavailable for the oldest row. |
| `lateExecution_`, `lateHandoff_` | Those durations when the current primary publication interval exceeds its median plus threshold. |
| `earlyExecution_`, `earlyHandoff_` | Those durations when the current primary interval is below its median minus threshold. |
| `lateExecutionChange_`, `lateHandoffChange_` | Signed changes from the preceding associated observation during late primary intervals. |
| `earlyExecutionChange_`, `earlyHandoffChange_` | Signed changes from the preceding associated observation during early primary intervals. |

All values are **seconds**. Changes may be negative. Counts distinguish an
unavailable value from a valid negative change; jitter needs two samples.
Conditioned means remain useful with one excursion. Ordinary timing remains
available when `diagnostics.excursionUs=0`, with conditional populations empty.
Each primary interval uses the retained primary-publication median, including
ordinary tails, even when its trace cannot be qualified.

For example, a 580 us primary interval, 500 us oldest-stage start interval and
an 80 us `lateExecutionChange_mean` in one row point to added work in that stage.
A positive `lateHandoffChange_mean` instead points to added delay between stages.
If all represented changes are small but `lateSourceInterval_mean` is about
580 us, the variation already exists at the oldest represented start. A steady
camera acquisition series would still leave camera publication and first-stage
wake as possible contributors. Several stages can contribute to one excursion;
do not force a single-cause interpretation or add marginal standard deviations.

`pipeline_boundary_timing` exposes `sourceInterval_`, `lateSourceInterval_`,
`earlySourceInterval_`, `latePublicationChange_` and `earlyPublicationChange_`.
The source interval measures consecutive oldest represented starts, not camera
acquisition. Publication changes measure the change in primary `writetime`
minus row-zero trace update. Together with row changes, these distinguish
variation entering the trace from variation added within it or after it.

`pipeline_timing_quality` reports `enabled`, `window`, `span`, `age`, `capacity`,
`queued`, `session`, `sequence`, `median`, `threshold`, `intervals`, `late`,
`early`, `stages`, plus cumulative `events`, `receipts`, `synthetic`, `rejected`,
`duplicates`, `counterGaps`, `invalidMetadata`, `missingTrace`, `invalidTrace`,
`topologyChanges`, `truncated`, `dropped`. Class counts describe publication
intervals; trace-conditioned population counts can be smaller. Statistics include
only the latest observed topology; identity changes break adjacent-observation
associations. A coherent first chain seeds
the baseline; subsequent real consecutive fresh observations establish timings.
Nonprogressing starts, input counters, malformed or unreceived rows, gaps and
losses break associations. Oversized chains are counted as truncated and cannot
establish full-chain attribution.

Existing MILK process timing must be enabled for the represented stages' starts
to advance. The app does not change `MeasureTiming`, trigger modes, priorities
or producer configuration. Trace input counters are sampled at output update,
and repeated metadata reads cannot prove exact consumed-frame identity; the
statistics are **best-effort trace associations**. All trace differences stay
within the external trace clock, independently of camera acquisition and local
monotonic coverage clocks. They do not establish physical DM response or full
camera-to-DM latency.

Both new observers append optional `[dmsynco]` tables. Rebuild/install
`dmSyncopation` and `logdump`; old records report new tables unavailable. For the
first comparison, leave scheduling disabled and compare `cameraAcquisitionInterval`
extrema with `pipeline_stage` conditioned changes and `pipelineSourceInterval`.
Ordinary app-logic passes emit these diagnostics even if the schedule snapshot
is busy; an absent trace does not suppress camera or primary interval statistics.
These diagnostics do not change operational scheduling,
skip-recovery or targeted extra-step wake behavior. The separate read-only
simulation evaluates the proposed grid policy without issuing commands.

### Isolate camera publication and the first AO stage

Use the [isolated test procedure](camera-acquWFS-test.md) and
[camera-acquWFS.conf](../camera-acquWFS.conf) to compare `camwfs` and
`acquWFS-1` without the downstream reconstruction/filter/DM-command RUN stages.
The separate observer instance selects `aol1_imWFS2` as its primary. Its one-row
`pipelineSourceInterval` measures `acquWFS-1` iteration starts after the trigger
wait; this is a processing-start proxy for wake cadence, not an exact semaphore
return timestamp. Compare those intervals with independent `cameraWriteInterval`
and `cameraAcquisitionInterval` populations. No absolute cross-clock latency is
claimed by comparing their interval distributions.

`diagnostics.only=true` bypasses scheduler resource attachment and processing,
creates no auxiliary/timing streams, disables combined monitoring and rejects
enable requests. `health.mode` identifies the mode. Its records have
`schedule_valid=0`; scheduler fields are placeholders. Primary/camera/pipeline
statistics remain usable without a DM combiner or DM-controller stream. The
profile explicitly selects ordinary primary monitor scheduling and clears its
usual cpuset. Normal scheduler mode retains its existing defaults and behavior.

### Capture a simulation run

The operator performs these steps on the instrument. Rebuild/install both
`dmSyncopation` and `logdump` so the writer and reader understand the appended
simulation table. The instrument's `teldump` alias is
`logdump --dir=/opt/MagAOX/telem --ext=.bintel`; no separate `teldump` build is
needed.

1. For a repeat capture with `simulation.enabled=true` already configured,
   keep the configuration and CPU placement unchanged. For the first
   simulation setup, ensure synco's `enable.toggle=Off`, stop that instance,
   then add the simulation settings above to its active configuration.
   A separately named test instance needs its own
   `/opt/MagAOX/config/<instance>.conf` installed before launch; `-c` alone
   does not replace that required instance file.
2. After installing the rebuilt app, restart the existing instance with
   `xctrl restart dmtweeter-synco`. Keep the camera and main AO loop running
   for this full-loop comparison. Keep any auxiliary
   providers stopped. Verify `enable.toggle=Off`, simulation mode in
   `health.mode`, and the existing primary/combined stream names. An On
   request is rejected in this mode; no enable operation is needed to collect.
3. Wait at least ten seconds after restart or a rate/attachment change, then
   confirm `simulation_cadence.state=2`, `spanSec` approximately 10–11 seconds,
   `updates` increasing about once per second, and fresh `observationAgeSec`.
   A valid grid alone does not indicate a full cadence window. Inspect
   `simulation`, `simulation_cadence`, `simulation_steps`,
   `simulation_status`, `simulation_trace`, `simulation_estimator`, `primary_timing_quality`, and
   `combiner_timing_quality`. A grid and a coherent output association are
   separate requirements. Before a long policy capture, confirm that
   `simulation.associations` increases and the combiner quality report has a
   nonzero matching `triggerInode` and an advancing `primaryCounter`. A valid
   grid with zero associations cannot evaluate successful auxiliary copies.
   If the trace reports zero inode/counter, inspect dmcomb's
   `.procinfo.triggersname`, including whether it resolved at initialization,
   and retain the actual trigger mode/delay in the record. A named primary
   can provide linkage in mode 4; changing to mode 3 is not required for this
   diagnostic. If FPS values are already correct, use `milk-procCTRL` F4
   (Triggering), match the combiner writer PID, and inspect the live trigger
   name/inode/mode. The inspected MILK implementation resolves the name at
   RUN initialization; live mode/delay updates do not re-resolve that name.
   Correct FPS readback alone therefore does not establish active linkage.
   Record absent associations, gaps and rejected observations along with
   would-run counts.
4. Verify `sysMonRTC.set_latency.toggle=On`, which is required for the RTC
   performance baseline, and retain it throughout the capture. Save its
   before/after readback. Keep camera/loop settings, dmcomb trigger mode,
   CPU/IRQ placement and workload fixed. Record those settings and capture
   at least 60 seconds of synco telemetry, ideally alongside `dmtweeter` and
   `camwfs` telemetry over the same period. For example, run
   `teldump -f dmtweeter-synco` and save its
   output, then stop it with Ctrl-C after the requested interval. Save the
   initial INDI snapshot as well as the final snapshot so cumulative outcome
   differences can be interpreted.
5. Confirm new `[dmsynco]` records contain `simulation_cadence` with `state:full`,
   a baseline near ten seconds, and exact fractional period fields, as well as `trace_quality` and
   `estimator_quality`, even if no overlap events have occurred. Review frozen phase residual tails,
   collection/association quality, nominal feasibility, per-step skip reasons
   and every `sim_overlap` entry. Group repeated main evidence using the key
   above, distinguish copy from reader hold, and inspect callback/lock lag.
   Use `sim_estimator` history to inspect the preceding period/phase inputs,
   frozen-plan provenance and correction/recovery. Check losses and whether
   selected sample IDs cover each interval under review. For complete
   post-trigger history, the final captured `estimator_quality` should have
   `queued=0` and `postRemaining=0`; if either is nonzero, extend the telemetry
   capture until the history drains or retain the incomplete boundary explicitly.
   Retain FIFO losses and the initial snapshot interval. Do not infer physical
   delivered cadence or an acceptable early reserve from would-run totals. Restart
   with `simulation.enabled=false` to return to normal scheduling mode; keep
   real scheduling disabled until timing bounds and wake routing are qualified.

For a 120-second simultaneous capture, run the following Bash block from the
instrument's MagAOX checkout after the checks above. It starts the collectors
before taking snapshots and waits only for those collectors. Snapshot queries
have an external eight-second limit: wildcard `getINDI` queries normally wait
for two seconds without matching property definitions, and repeated definitions
can keep that wait open. Ordinary property updates do not reset it. A snapshot
status of 124 means the external timeout expired; retain its partial output
and investigate the repeated definitions separately. Bounding the query does
not correct their cause.

The block takes about two minutes plus the final bounded snapshots and any
collector termination grace. The first snapshots overlap the start of the
telemetry capture. Record the dmcomb trigger mode and AO gain with the result.

```bash
(
    synco_capture_dir="$HOME/tmp/synco-sim-$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$synco_capture_dir" || exit 1
    git rev-parse HEAD > "$synco_capture_dir/git.txt"
    synco_capture_pids=()

    for synco_device in dmtweeter-synco dmtweeter camwfs
    do
        timeout -k 5s -s INT 120s logdump \
            --dir=/opt/MagAOX/telem --ext=.bintel -f "$synco_device" \
            > "$synco_capture_dir/$synco_device.txt" 2>&1 &
        synco_capture_pids+=( "$!" )
    done

    for synco_device in dmtweeter-synco dmtweeter holoop camwfs sysMonRTC.set_latency.toggle
    do
        if timeout -k 2s 8s getINDI "$synco_device" \
            > "$synco_capture_dir/$synco_device-indi-before.txt" \
            2> "$synco_capture_dir/$synco_device-indi-before.err"
        then
            synco_status=0
        else
            synco_status=$?
        fi

        printf '%s before status=%s\n' "$synco_device" "$synco_status" \
            >> "$synco_capture_dir/status.txt"
    done

    for synco_pid in "${synco_capture_pids[@]}"
    do
        if wait "$synco_pid"
        then
            synco_status=0
        else
            synco_status=$?
        fi

        printf 'collector pid=%s status=%s\n' "$synco_pid" "$synco_status" \
            >> "$synco_capture_dir/status.txt"
    done

    for synco_device in dmtweeter-synco dmtweeter sysMonRTC.set_latency.toggle
    do
        if timeout -k 2s 8s getINDI "$synco_device" \
            > "$synco_capture_dir/$synco_device-indi-after.txt" \
            2> "$synco_capture_dir/$synco_device-indi-after.err"
        then
            synco_status=0
        else
            synco_status=$?
        fi

        printf '%s after status=%s\n' "$synco_device" "$synco_status" \
            >> "$synco_capture_dir/status.txt"
    done

    printf '%s\n' "$synco_capture_dir"
)
```

Collector status 124 is expected when `timeout` ends follow mode after 120
seconds; status 137 indicates that its five-second kill grace was needed.
Other nonzero statuses and short/empty telemetry files need review before
interpreting the capture.

## Measure the remaining DM reader lifetime

The optional [reader-protection shadow test](reader-protection-test.md) matches
the controller's dedicated completion stream to these same primary/combined
observations. It reports signed time remaining, positive required hold,
notification delay and slack against the current modeled budget, with exact
lifetime worst-case evidence and matching/loss quality. It requires the updated
controller publisher, synco and `logdump`, and is enabled only in simulation
mode by `completionMonitor.shmimName`. Existing scheduling and shadow decisions
remain unchanged.

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

1. Verify exclusive destination/source ownership, `dmtweeter-synco` availability, the deployed dmcomb DELAY/counter route and primary monitor semaphore assignment. Confirm the trigger channel is not excluded by astrogrid mode and auxiliary inputs do not unintentionally drive ZPO work. Verify `sysMonRTC.set_latency.toggle=On` for the required RTC performance baseline, and save its readback before and after qualification captures.
2. Use the runtime [DM controller timing telemetry](../../../libMagAOX/app/dev/doc/dmTiming.md): `dm_timing`, `dm_timing_stages` and `[dmtimes]` records. Compare the documented processing/reader endpoints and receipt spacing with app residuals and dmcomb output. Use synco's independent primary statistics for the input-to-combined cadence comparison; the legacy `XWC_DMTIMINGS` buffers are no longer needed.
3. Qualify delivery tails, observation lag, early-main uncertainty, snapshot/copy budgets and every adjacent/wraparound/copy window under the relevant load. Keep host-timing qualification separate from deterministic CI tests.
4. Start disabled, verify configuration/status and the selected period source/camera reply age, start the local 200 Hz provider, then enable and wait for lock. Check requested/active/feasible counts, selected/measured rates and delivered cadence.
5. Measure residual mean/RMS/percentiles/max, minimum receipt spacing, skipped/coalesced events and recovery duty cycle. Use 5 us as an analysis guideline. WFS measurements should recover the configured 200 Hz vibration.
6. Disable before changing timing controls or performing independent maintenance. Mapping/topology changes require restart. Controller clearing does not stop active writers.

An arbitrarily early main publication can race a multi-channel memcpy or follow an auxiliary submission too soon. This observer-based app can detect and recover from such a violation but cannot block the genuine main publisher. A hard gating/pipelining guarantee requires a separate change to the cooperating loop/controller path.
