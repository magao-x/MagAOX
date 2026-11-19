# dmSyncopation production providers

## Objective and settled decisions

Promote the qualified counter-publication scheduler to continuous production
operation and support same-host producers through a reusable
`MagAOX::app::dev::dmSyncopationProducer<derivedT>` CRTP helper. Migrate
dmSpeckle as the first producer while preserving its legacy operation.

The [counter-publication record](dm-syncopation-counter-publication-targets.md)
contains the 8 kHz qualification and the decisions preceding this integration.
The final 120-second constant-zero run delivered and acknowledged all 719,997
post-startup auxiliary slots with a 6 us final admission allowance. These
measurements establish the transport/timing path, not nonzero application
correctness. No agent may operate instrument hardware.

Production providers are independent and same-host only. A missing, incomplete,
invalid or late input retains only that producer's last complete destination
contribution; other producers proceed. Always copy every selected mapping.
Explicit zero removes a contribution. A missed input does not move targets or
invalidate a valid later-slot input. Expired slot commands cannot be replayed
implicitly at a later slot.

Step 0 uses the same ready-copy operation within the reader-safe before-main
window, preserving destination counters. Genuine main commands are never gated
by producer readiness. Physical consumption is separate telemetry, not a
per-producer acknowledgement required for passive copying. A completed passive
copy remains visible if a later auxiliary trigger is suppressed. Previous
read-done evidence does not block a new, unexpectedly early main during copying;
copy-window qualification remains necessary.

## Protocol and ownership

Reuse the 41-word provider schedule layout with production version 2. Keep the
older version-1 validation APIs for legacy behavior. A production request
identifies epoch, generation, cycle/step, monotonic target and cutoff.

Retain one 2-D float32 `<destination>_aux` source per producer. Add a
producer-owned signed-int64 `_ready` record and scheduler-owned `_ack` record.
The ready record commits the request identity, exact source inode/counter,
completion timestamp and producer session/sequence/PID. The acknowledgement
echoes the exact command and states copied/discarded. The helper freezes the
single outstanding source payload until that acknowledgement or a validated
schedule reset. This is transport ownership: the acknowledgement occurs after
a private snapshot/discard, without waiting for a physical DM completion.

Source, ready and acknowledgement coherence must be checked across snapshots.
Reject nonfinite pixels, stale/mismatched identities and malformed transports.
A rejected source must not replace its retained contribution or block eligible
independent producers. Revalidate shared reader/clock/primary/output identity
and timing conditions before changing destinations and publishing a trigger.

## Implementation

1. Share the tested finite-zero grid/readiness/wait mechanism with opt-in
   continuous production mode. Preserve finite trial behavior and the older
   scheduler when neither new mode is selected.
2. Publish production schedule targets/cutoffs from the frozen grid. Add
   independent tagged source admission, retained values and complete passive
   copies. Include passive step 0 and continuous disable/reacquire behavior.
3. Put coherent schedule reads, ready/ack transport, deadline checks and
   immutable-buffer handling in the reusable CRTP helper. Provide normal
   base-class integration macros and documented application responsibilities.
4. Replace dmSpeckle's duplicated provider loop/transport with the helper while
   retaining target-time sparkle, arbitrary-cube and tip/tilt generation.
5. Verify workstation-local private streams: independent input misses,
   on-time/late command identity, passive nonzero copies, retention, malformed
   input, restart/replacement, cancellation and timing safety. Re-run existing
   finite zero and legacy application regressions.
6. Publish the [production guide](../../../apps/dmSyncopation/doc/production-providers.md)
   and operator configuration. Instrument qualification remains operator-only.

Initial production preparation settings are `readyLeadUs=50` and
`step0LeadUs=20`; final publication admission remains 6 us with 10 us early
validation wake and zero unmeasured commit lead. Input preparation deadlines
remain separate from publication admission. Changed geometry/mapping counts
require measured copy-budget qualification.

## Verification record

The initial shared producer helper suite passes **10 cases / 269 assertions** under the normal
optimized build flags. Its private-stream cases cover exact acknowledgement,
single-buffer ownership, cooperative writer exclusion, invalid transports,
epoch reset, stale identities, nonfinite pixels, publication after a missed
cutoff, changed source/record extents and predecessor ownership after a producer
restart. A replacement helper adopts an intact unacknowledged commitment before
it may reuse the source; later publications use its new session. The build and run logs are `/tmp/dmSyncopationProducer-build.log` and
`/tmp/dmSyncopationProducer-run.log`.

The scheduler suite passes **152 cases / 6,514,030 assertions**, including the
existing finite-zero and legacy regressions and the new production cases.
Production coverage includes independent source misses in both directions,
exact transport acknowledgements, stale/late identities, nonfinite/busy
sources, malformed record metadata and future commands, valid/invalid startup
configurations, and nonzero common-engine operation at both passive-only
`N=1` and four-step cadence. Continuous mode ignores finite-trial expiry in
both the timed callback and ordinary status path. Build and run logs are
`/tmp/dmSyncopation-production-final-build.log` and
`/tmp/dmSyncopation-production-final-run.log`.

The final dmSpeckle migration suite passes **6 cases / 85 assertions** against
the final helper, including predecessor ownership and geometry hardening.
Both applications compile and link successfully. Logs are
`/tmp/dmSyncopation-production-app-build.log`,
`/tmp/dmSpeckle-production-final-test-build.log`,
`/tmp/dmSpeckle-production-final-test-run.log`, and
`/tmp/dmSpeckle-production-final-app-build.log`. clang-format verification and
`git diff --check` pass for all touched C++ files.

Shared record utilities validate mapped pointers and exact element/byte extents.
The helper freezes source geometry and identity and revalidates them before
access, including after publication waits. Pixel checks use mxlib's bitwise
finite-value classifier under the normal fast-math flags. The independent
consumer rejects incompatible input metadata before reading payloads. Source
or sidecar replacement retains that producer's contribution until explicit
resource reacquisition and any necessary producer reconnection.

Standard production timing/health reflects the active grid and confirmed
publications. The mode exposes only steps/cadence in the disabled-only settings
property; static reader-hold knobs remain specific to the legacy scheduler.
Main silence invalidates the provider schedule and starts reacquisition. Rate
reporting resets its baseline on rearm and guards against unsigned differences
when session totals restart. dmSpeckle backs off unavailable transport retries
by 100 ms while retaining short waits for an attached, idle protocol.

Two initial offline failures were fixture issues: imported ImageStreamIO
handles leave `used=0`, so the fixture must explicitly claim its private
files before destroying them; otherwise old sidecars survive between cases.
NaN fault injection now writes IEEE bits directly so finite-math optimization
cannot remove the intended malformed payload. Neither issue identified a
runtime admission failure. The subsequent full scheduler suite passes.

No hardware actions or changes to instrument configuration are part of this
work. Changing nonzero commands and the selected mapping/producer setup remain
operator qualification work.

## 2026-11-04: correct production INDI startup registration

The first operator startup failed while inserting `dmtweeter-synco.production`.
`setupProductionINDI()` used `REG_INDI_NEWPROP_NOCB`, which already registers
the property, and then called `registerIndiPropertyReadOnly()` on the same
property. The second insertion failed and aborted app startup. This was a
production integration error; the prior offline tests did not exercise this
property's startup registration.

Create the read-only numeric property without registering it, populate all
elements with initial numeric values, then register it once. The new regression
runs the production property setup with INDI enabled but no driver or monitor
threads. It verifies one registry entry, the property pointer and null callback,
read-only/idle metadata, initialized elements and serializable definition XML.
It also verifies that disabled production mode registers no property. No
instrument or external INDI connection is involved.

The normal app build also lacked a dependency on the new production header.
Add that header and the shared producer helper to dmSyncopation's declared
dependencies, and the producer helper to dmSpeckle's dependencies, so ordinary
incremental builds pick up their changes. The corrected scheduler regression
suite passes **153 cases / 6,514,054 assertions**; its log is
`/tmp/dmSyncopation-production-indi-run.log`.
The normal incremental dmSyncopation build recompiles and links successfully
with the corrected dependency. Dry-run dependency checks also select a fresh
compile for changes to the production header and the shared producer helper.
Build and dependency logs are `/tmp/dmSyncopation-production-indi-app-build.log`,
`/tmp/dmSyncopation-production-dependency-check.log`, and
`/tmp/dmSpeckle-producer-dependency-check.log`. Formatting and diff checks pass.

## 2026-11-04: first changing-command run uses two steps and exposes a passive-copy budget mismatch

The operator reports both applications running, with tweeterSpeck on channel
09's auxiliary source. The INDI readback requests two steps: a 2 kHz main
therefore produces the observed approximately 4 kHz total cadence. This is the
requested configuration, not automatic reduction from four steps.

Passive-copy totals are zero completed / 123,119 skipped. The operator confirms
`readyLeadUs=50` and `step0LeadUs=20`. This leaves at most 30 us between the input
cutoff and passive completion cutoff, before snapshot time. The learned copy
budget is 31.297 us (maximum observed copy 30.297 us); the reported snapshot
maximum is 4.989 us. The pre-copy fit check therefore cannot admit step 0 with
these settings. A running maximum does not establish typical copy cost.

The next proposed operator configuration uses four steps with an 8 kHz cap and
`readyLeadUs=70`, retaining `step0LeadUs=20`. This provides 50 us for snapshot and
copy without decreasing the learned budget or moving the passive completion
cutoff. It is provisional preparation headroom requiring a new observation,
not a qualified timing bound. At four steps the first input cutoff becomes
55 us after the fitted main, reducing available producer computation time;
check both passive-copy progress and producer misses after restarting.

The channel-09 producer has accepted inputs at mapping index 2. Missing inputs
for other inactive mappings are expected under independent retained-value
semantics. The separately queried counters do not establish a common interval
or acceptance fraction. No configuration, application code or instrument state
was changed by an agent for this analysis.

The operator subsequently confirms four mapped channels (07/08/09/10) and
changes the requested count to four steps. The reported dmtweeter acquisition
interval is `1.250000e-4` s, consistent with an 8 kHz mean controller wakeup
cadence. Mapping count and subdivision count are independent. This confirms
the cadence setting, but a fresh passive-copy readback is still needed: the
confirmed 50/20 us preparation settings and earlier learned 31.297 us budget
cannot fit passive step 0, independently of the requested step count.

The next readback confirms four requested/active steps and approximately
8,000.004 Hz total delivery. Passive totals are 4,037 completed / 422,016
skipped (0.9475% completed over those cumulative counts). The current learned
copy budget is 27.761 us with a 26.761 us observed maximum; snapshot maximum is
8.015 us. The 30 us default preparation window therefore leaves only 2.239 us
for snapshot, wake delay and any remaining reader wait before the copy fit
check. This is still inadequate preparation headroom. These separate maxima
are not paired observations and do not classify every passive skip.

Mapping 2's 1,528,063 accepted + 176,148 missing + 1 rejected inputs reconcile
to 1,704,212 opportunities, also four times the passive total. Accepted inputs
are private snapshots, so the subsequent passive-copy veto can prevent their
application. Cumulative totals include startup and producer enable history;
use two readbacks separated by about ten seconds after the proposed 70/20 us
restart to evaluate current passive success and producer misses.

## 2026-11-04: wider-window comparison restores passive copies and complete producer admission

The next attachment, `fdf5acdd-4827-4bbc-9f82-25c9c88b6961/Pasted text.txt`,
contains two production readbacks following the proposed 70/20 us comparison.
It does not independently include effective preparation-lead settings.

| Counter | First | Second | Increase |
| --- | ---: | ---: | ---: |
| Mapping 2 accepted | 248,140 | 467,060 | 218,920 |
| Mapping 2 missing | 150,423 | 150,423 | 0 |
| Mapping 2 rejected | 1 | 1 | 0 |
| Passive copied | 99,641 | 154,371 | 54,730 |
| Passive skipped | 0 | 0 | 0 |

The accepted-input increase is exactly four times the passive-copy increase.
Every input opportunity in this interval was accepted from tweeterSpeck, and
every passive batch completed. Existing source-missing/rejected totals are
historical events that did not increase in the comparison. Missing inputs on
the other three mappings grow by the expected 218,920 while their retained
contributions continue to be copied.

The timing readbacks show approximately 8,000.02 and 8,000.03 Hz total delivery,
four requested/active steps, zero faults, zero passive not-started and copied-late
counts, and an unchanged auxiliary-miss total of three. The current copy budget
is 28.221 us (27.221 us observed maximum), with a 6.172 us snapshot maximum.
These fit nominally within a 50 us preparation window without qualifying a
worst-case bound. The production snapshots represent 54,730 cycles, nominally
27.365 s at 2 kHz; no wall-clock capture timestamps are supplied. The timing and
production queries cover different intervals, so their count deltas must not
be directly compared.

Retain the comparison settings for a longer production timing capture. These
counters establish input admission and passive-copy progress; per-slot timing
errors and actual waveform correctness still require the corresponding timing
or pixel evidence. No application code or instrument state was changed by an
agent during this analysis.

## 2026-11-04: first two-minute production capture with tweeterSpeck

Capture: `synco-production-20261105T051504Z`, extracted on the workstation at
`/home/jrmales/tmp/home/jrmales/tmp/synco-production-20261105T051504Z`.
The recorded revision is `8972958b8043f5b17f5c7a2d6f161b5f343b26bd`.
The operator confirms a stable AO loop, modal gains zero, and unchanged settings
and workload throughout. All analysis was local on `mx11`; no agent accessed
the instrument.

### Setup and evidence coverage

The copied configuration confirms production mode, four steps, an 8 kHz cap,
destinations 07/08/09/10, and preparation leads of 70/20 us. Simulation and the
finite zero test are disabled. The release observer uses spare57 at requested
priority 30. Only tweeterSpeck is supplying commands, through `dm01disp09_aux`;
its direct trigger is Off. Modulation remains On, with sparkle amplitude 0.01,
frequency 2000, dwell 1, separation 15, angle 0 and cross On. All twelve captured
producer telemetry records agree with the before/after snapshots.

Camera state remains bin2, 2 kHz, synchronization On, EM gain 1, temperature
target +20 C and shutter SHUT. The loop is On at scalar gain 1, and RTC latency
is On. These saved states are unchanged; the modal-gain and workload statements
come from the operator.

The collection bracket is 05:15:14.195080594--05:17:14.205102751 UTC,
**120.010022157 s**. All twelve bounded INDI queries succeed with empty stderr;
all four collectors end with the expected timeout status 124. Before/after
queries enclose a longer interval: the synco snapshots span **130.805139804 s**.
Do not assign those counter deltas to precisely 120 seconds.

Synco text ends at 05:17:09.896520358 UTC, 4.31 s before collection end. Its first
and last captured full-rate summary snapshots span **114.567121279 s**. No
event-log failures, event drops or rising queue high-water explain the missing
text tail. The enclosing INDI counters cover that interval independently, but
the decoded file does not provide individual source events for the full two
minutes. Controller and camera statistics below use rolling two-second windows
fully contained within the collection bracket; those windows overlap.

### Production admission and delivery

| Counter | Increase between INDI snapshots |
| --- | ---: |
| Main cycles / passive copies completed | 261,610 |
| Channel-09 inputs accepted | 1,046,440 |
| Channel-09 inputs missing / rejected | 0 / 0 |
| Passive copies skipped | 0 |
| Auxiliary opportunities / posted / acknowledged | 784,830 / 784,830 / 784,830 |
| Auxiliary skips / clock-excluded cycles | 0 / 0 |
| Scheduler faults / event drops | 0 / 0 |

Accepted input is exactly four times the passive-copy count; auxiliary output
is exactly three times that count. Thus every source opportunity in this
snapshot interval was accepted, every passive batch completed, and every
auxiliary opportunity was posted and acknowledged. The absent producers at
07/08/10 each accumulate the expected 1,046,440 missing inputs while their
retained contributions remain in use. The existing channel-09 missing/rejected
totals and six auxiliary skips are historical and do not increase.

The first/last captured telemetry summaries independently show 229,134 main
cycles and **687,402 auxiliary opportunities, posts and acknowledgements**, with
no new skips, clock exclusions, source faults or drops. Standard timing reports
approximately 8 kHz. Its separately sampled lifetime delivery counters should
not be subtracted against production counters as though they share one capture
instant.

### Source publication timing

Full-rate cumulative-moment differences across the captured summary interval
give 229,134 observations per step. Values derived from printed summary
moments are approximate:

| Step | Mean source publication error | RMS variation about the mean |
| --- | ---: | ---: |
| Main 0 | +0.135686 us | 1.859774 us |
| Auxiliary 1 | +0.243177 us | 0.026918 us |
| Auxiliary 2 | +0.260698 us | 0.229391 us |
| Auxiliary 3 | +0.239080 us | 0.040320 us |

The endpoint is the source writetime recorded immediately before counter
publication, a proxy for the counter store rather than a timestamp of that
store or the controller's SDK call. The main is not scheduled by synco.

Auxiliary 2 has a brief timing burst in summaries logged approximately
05:16:32--05:16:38 UTC. Its cumulative maximum increases from 5.736 to
**5.951 us**, locating that new maximum within this capture. The strongest
summary-delta bin contains 2,007 publications with mean error 1.221255 us and
RMS variation 1.365806 us. Its mean final admission-check error is 0.864058 us,
with another 0.357198 us between that check and source writetime. Both waiting
and publication overhead contribute. No slot is skipped during the burst.
Unchanged lifetime extrema on the other steps cannot be reported as maxima
measured in this capture.

Healthy individual events are sampled with `primary % 4 == 0`: 57,675 retained
events per step after filtering their source UTC. This misses the disturbed
parity. Their small tails therefore do not describe the full-rate tail or
establish how many final admission checks exceeded a smaller allowance. The
full-rate moment deltas include the burst. The configured final admission
allowance remains 6 us; source writetime errors do not directly measure that
check's remaining margin.

### Controller and camera timing

| Measurement | Observed range across contained two-second windows |
| --- | ---: |
| Controller wakeup interval mean | 124.9996--125.0005 us |
| Controller wakeup interval RMS jitter | 3.199--3.642 us |
| Controller wakeup interval minimum / maximum | 92.264 / 160.553 us |
| Native SDK duration maximum | 110.117 us |
| SDK worker service maximum | 115.108 us |
| Controller preparation maximum | 25.228 us |
| Controller callback entry to last-read endpoint maximum | 33.644 us |
| Camera acquisition interval RMS jitter | 0.252--0.816 us |
| Camera publication interval RMS jitter | 0.835--1.210 us |

Controller first/last telemetry counters add 949,144 receipts and submissions,
with no new failure, duplicate, counter-gap, invalid or dropped events. The
broader INDI snapshots likewise show no new SDK/pipeline rejection, coalescing,
overload, cancellation, timeout, preparation, post-processing or reader faults.
Controller clock mapping excludes 93 wide brackets; these exclude timing
measurements, not DM commands.

Main-impact matching adds 229,132 exact SDK/reader matches in the captured
summary interval. Interval-derived means are approximately 31.28 us from the
primary observer to SDK entry, 16.19 us from reader start to SDK entry, and
104.31 us inside the native SDK. Two additional main impacts remain
unclassified. Candidate timing exclusions increase by 7,489 in those summaries
(8,508 across the longer INDI bracket), without source resets, timing eviction
or new SDK gaps/failures. Explicit SDK-wait timing remains unavailable. These
records do not prove zero added main SDK wait or a hard 125 us service bound.

Camera counters add 236,508 published frames between first/last telemetry, with
no new frame/counter gaps, failures, invalid events or drops. Ten retained
complete excursion traces span -21.180 to +25.708 us and all overlap serial
activity. No new trace loss occurs. Camera acquisition-to-write means are
47.799--48.058 us, with a 57.222 us maximum. Relative to the preceding constant
zero test, controller jitter is similar and camera publication jitter is
lower; the changed production workload prevents attributing that difference
to one cause.

### Interpretation and follow-up

This is a successful continuous production transport/timing observation with
one active producer and four mapped destinations. Retain 70/20 us preparation
leads for this setup. It does not establish a worst-case bound, simultaneous
operation of several active producers, or pixel-level end-to-end waveform
correctness.

The configured sparkle frequency of 2000 with dwell 1 advances the four-plane
pattern at 2 kHz, typically repeating a plane over four 8 kHz protocol slots.
The full pattern repeats at approximately 500 Hz. Scheduler cadence and
waveform progression are independent: accepted commands at every slot need
not contain different pixels. A test intended to advance one sparkle plane
per 125 us slot would select an 8000 frame-rate setting; no such change was
made here.

Both snapshots expose stale health text, `production publication clock
unqualified`, despite valid clocks, active production and no new exclusions.
Code review finds that clock rejection sets this text, while the active text
is only set on the first trial start, not on later qualified recovery. Correct
that status update separately; the recorded historical rejection is not a new
failure in this run. Also retain the successful-event sampling limitation when
interpreting future clustered-tail studies.

Analysis artifacts are `/tmp/production0515-source-timing.json`,
`/tmp/production0515-camera-dm-analysis.json`, and
`/tmp/production0515-camera-dm-analysis.out`. Only this engineering record was
updated for the capture; no runtime code, settings or instrument state changed.

## 2026-11-04: production comparison with sparkle frame rate at 8 kHz

Capture: `synco-production-20261105T052815Z`, supplied as
`/home/jrmales/tmp/synco-production-20261105T052815Z.tgz` and extracted for
analysis under `/tmp/synco-production-052815-analysis/home/jrmales/tmp/`.
The operator reports changing only the modulator frequency to 8000. Revision
remains `8972958b8043f5b17f5c7a2d6f161b5f343b26bd`; this is the same continuous
production trial, not a fresh counter session.

### Setup, coverage and command admission

Before/after producer snapshots are identical with modulation On, frequency
8000, dwell 1, amplitude 0.01 and direct trigger Off. Relative to the preceding
capture, frequency current/target are the only changed producer fields. All
twelve captured producer telemetry records agree. Scheduler configuration is
unchanged: four steps, 8 kHz cap, four mappings and 70/20 us preparation leads.
The copied producer file has no frequency entry; the saved live properties and
telemetry establish the effective 8000 setting.

Camera settings remain bin2, 2 kHz, synchronization On, EM gain 1, target +20 C
and shutter SHUT. The loop is On at scalar gain 1, and RTC latency is On. The
operator's unchanged-setup statement carries forward the earlier modal gains
of zero. No additional instrument commands were requested or executed.

Collection spans 05:28:25.170141871--05:30:25.181668242 UTC,
**120.011526371 s**. All twelve INDI queries succeed, stderr files are empty,
and all four collectors finish with expected timeout status 124. The enclosing
synco INDI snapshots span **130.824600727 s**:

| Counter | Increase between INDI snapshots |
| --- | ---: |
| Main cycles / passive copies completed | 261,649 |
| Active producer inputs accepted | 1,046,596 |
| Active producer inputs missing / rejected | 0 / 0 |
| Passive copies skipped | 0 |
| Auxiliary opportunities / posted / acknowledged | 784,947 / 784,947 / 784,947 |
| Auxiliary skips / clock exclusions | 0 / 0 |

Every input opportunity is accepted, every passive copy completes, and every
auxiliary opportunity is posted and acknowledged in this interval. The three
absent producers each accumulate the expected 1,046,596 missing inputs. Healthy
continuous operation retains their preceding contributions.

Decoded synco text ends at approximately 05:30:22.926 UTC, 2.26 s before the
collector end. Its captured full-rate summary snapshots span 116.590591786 s
and independently account for 233,181 mains and 699,543 auxiliary opportunities,
posts and acknowledgements, again with no new skips, clock exclusions or drops.
As before, do not assign the broader INDI deltas to exactly two minutes or
represent the decoded source events as complete coverage of the collection.

### Timing comparison

Across the longer before/after INDI brackets, pooled auxiliary source-target
error is almost unchanged:

| Measurement | Earlier 2 kHz modulator | New 8 kHz modulator |
| --- | ---: | ---: |
| Auxiliary publication error mean | +0.246720 us | +0.247030 us |
| Auxiliary publication error RMS variation | 0.127674 us | 0.129370 us |
| Controller wakeup interval RMS jitter | 3.199--3.642 us | 3.198--3.634 us |
| Controller wakeup interval maximum | 160.553 us | 159.812 us |
| Native SDK duration maximum | 110.117 us | 121.609 us |
| SDK worker service maximum | 115.108 us | 123.944 us |
| Controller preparation maximum | 25.228 us | 26.580 us |
| Camera publication interval RMS jitter | 0.835--1.210 us | 0.840--1.188 us |

Source moments are derived from cumulative counts, means and variances;
controller/camera ranges use overlapping two-second windows fully contained
within each collection. These are observed distributions, not timing bounds.
The source endpoint remains writetime immediately before counter publication.

For the shorter captured source-summary interval, approximate mean/RMS
variation in us is +0.150262/1.918685 for main 0, +0.265953/0.228203 for auxiliary
1, +0.239828/0.037539 for auxiliary 2, and +0.237751/0.043115 for auxiliary 3.
A brief disturbance now affects auxiliary 1. The retained sample contains 693
source errors above 2 us between source times approximately 05:30:05.963 and
05:30:10.889 UTC, with a 4.766 us maximum and a 4.571 us final-check maximum.
They cluster near 15.34 us modulo 1 ms. These are sampled counts, not all-slot
tail counts: healthy events still retain only `primary % 4 == 0`. No lifetime
source maximum increases, so the full-rate capture maximum cannot be recovered
from cumulative extrema. No slot is skipped during the disturbance.

Controller wakeup means are 124.9997--125.0006 us. First/last controller
telemetry adds 949,067 receipts/submissions without new failures, gaps,
duplicates, invalid events or drops. Native SDK means remain
104.337--104.642 us. The new 121.609 us tail includes a main command: the
matched-main SDK maximum increases from 110.148 us during this capture.
The 123.944 us worker-service maximum is close to one 125 us slot, without
establishing a hard service bound or the amount of SDK queue wait.

Main matching adds 233,180 exact SDK/reader matches and one unclassified impact
in the captured summaries. Derived means are 31.23 us from primary observer to
SDK entry, 16.18 us from reader start to SDK entry, and 104.40 us inside the
native SDK. Candidate exclusions increase by 8,216 in those summaries and
9,177 across the INDI bracket, without new SDK gaps, failures, source resets
or evictions. Explicit SDK-wait telemetry remains unavailable. Controller
delivery clock mapping excludes 142 wide brackets, not commands.

Camera first/last telemetry adds 236,512 published frames with no new
frame/counter gaps, failures or drops. Acquisition RMS jitter is
0.095--0.842 us; acquisition-to-write means are 47.825--48.075 us, with a
59.259 us maximum. Seven complete qualified excursion traces span -21.441 to
+26.149 us and all overlap serial activity. There is no new trace loss.

### Separate the captured result from events between captures

This captured interval is clean, but the continuous session is not entirely
skip-free. Several counters increased after the previous capture and before
the first snapshot of this one:

| Lifetime counter | Previous after | New before | Increase during new capture |
| --- | ---: | ---: | ---: |
| Passive skipped | 0 | 2 | 0 |
| Auxiliary skipped | 6 | 13 | 0 |
| Clock-excluded cycles | 186 | 294 | 0 |
| SDK worker coalesced | 0 | 2 | 0 |
| SDK worker rejected | 1 | 3 | 0 |
| Pipeline overloaded | 0 | 2 | 0 |

The saved files do not locate these between-capture events or identify whether
an affected controller command was main or auxiliary. Do not attribute them
to the frequency change or count them as new failures during this two-minute
test. A longer observation including application error/event logs would help
locate a recurrence. The stale health text identified above also remains a
separate status follow-up.

The 8000/dwell-1 setting selects one sparkle plane per nominal 125 us target;
the four-plane pattern cycles at approximately 2 kHz. This run therefore
exercises command admission with the intended faster waveform progression,
while pixel-level end-to-end waveform correctness remains unrecorded. Retain
the current timing/preparation settings: no steady-capture regression or new
skip is demonstrated by this comparison.

Analysis artifacts are `/tmp/production0528-source-timing.json`,
`/tmp/production0528-source-summary.json`, and
`/tmp/production0528-camera-dm-analysis.json`. Only this engineering record was
updated; runtime code and instrument settings were not changed by an agent.

## 2026-11-04: prepare the branch for review against dev

Fetch and merge current `origin/dev` (`5a45f1c7`) before the pull request. Its
removal of obsolete self-monitor comments conflicts with this branch's adjacent
out-of-class timing-hook definitions. Preserve the definitions and remove the
obsolete comments; the resolved merge is `7fc90c43`.

Correct the stale health text found in both production captures. Every
qualified active cycle now refreshes the active reason, including recovery
within an existing trial. The trial start time and lifecycle event remain
unchanged. The new private-stream regression covers both finite and production
modes, injects a rejected publication-clock capture, and verifies active health
on recovery. Its immediate fake dmcomb requires one conservative baseline
recovery cycle before auxiliary posting resumes; the test preserves and checks
that existing behavior rather than weakening clock/output admission.

The first regression run incorrectly expected posts during that baseline
recovery cycle. All existing cases and the new health assertions passed, but
both mode variants failed the new posting-count assertion. After correcting
the fixture expectation and checking resumed posts on the following cycle,
the rebuilt full scheduler suite passes **154 cases / 6,634,353 assertions**.
Logs are `/tmp/dm-syncopation-pr-scheduler-final-build.log` and
`/tmp/dm-syncopation-pr-scheduler-final-run.log`. clang-format verification and
the working-tree whitespace check pass. The change affects status reporting;
the qualified scheduling policy is retained.

The standard test list omitted six implemented suites: pipeline timing,
pipeline integration, saturation accumulation, the producer helper, pipeline
telemetry and the pipeline simulator. Register them in their existing library,
logger and application groups. All 26 newly added C++ test files on this branch
are now included in the standard test list.

Fresh optimized builds/runs of the producer, dmSpeckle, mock-SDK BMC controller,
monitor timing, pipeline timing/integration, saturation, pipeline telemetry,
pipeline simulation, synco telemetry and logdump suites pass **140 cases /
11,987 assertions** in total. Together with the scheduler this readiness run
covers **294 cases / 6,646,340 assertions**. The C++ build/run manifest is
`/tmp/dm-syncopation-pr-results.json`; scheduler logs are listed above. All
builds are serialized, and the tests use workstation-local private streams or
software SDK stubs without instrument access.

Python helper suites pass **56 finite-trial capture tests**, **31 camera-capture
tests** and **22 EDT-probe tests**. Logs are
`/tmp/dm-syncopation-pr-capture-tests.log`,
`/tmp/pr-scope-camera-capture-test.log` and
`/tmp/pr-scope-edt-probes-test.log`. Normal builds of dmSyncopation, dmSpeckle
and logdump also pass; their logs are in the C++ manifest.

For deployment, update the production preset and guide to the measured four
destinations and 70/20 us leads. Document required rebuilds, BMC worker/post
placement, dmcomb behavior, an 8 kHz sparkle setup, and the distinction between
zero transport submission and actual destination clearing. CPU assignments and
rates remain setup-specific. Retain the capture limitations and inter-capture
overload observations above in the review description.

## 2026-11-05: restore circular-buffer header compatibility after merge

The first reported full build of merged dev (`45fc8a42`) fails in baslerCtrl:
`mx::sigproc::circularBufferIndex` is undeclared. The framegrabber timing
refactor removed `<mx/sigproc/circularBuffer.hpp>` after replacing its internal
buffers, but baslerCtrl, psfFit and refRMS still receive those declarations
through the shared app headers. modalFilter and modalPSDs include it directly.

Restore the existing include in frameGrabber rather than change the three
applications or their temperature/statistics behavior. The functional fix is
one include line. clang-format verification and `git diff --check` pass.

Reproduce the same missing-namespace failure with an ordinary refRMS build
before the change, then successfully compile/link refRMS and psfFit afterward.
Logs are `/tmp/framegrabber-include-refRMS-before.log`,
`/tmp/framegrabber-include-refRMS-after.log` and
`/tmp/framegrabber-include-psfFit-build.log`. The workstation lacks the Basler
Pylon SDK, so a complete baslerCtrl build remains an operator-side check. The
reported Pylon deprecated-copy warnings are distinct from the missing type.
All validation is local to `mx11`, without instrument access.

## 2026-11-05: post-merge timing issue and verified 5 us dmcomb delay

After merging, rebuilding and restarting, the operator reports a dmtweeter
`fgtimes` acquisition interval of approximately 252 us with roughly 150 us
jitter while synco is enabled. With synco disabled, the corrected baseline is
500 us with 20 us jitter. The reported enabled mean is about 3.97 kHz; the
disabled baseline remains 2 kHz but has elevated jitter.

The supplied readback confirms RTC latency On, four requested steps and an
8 kHz cadence cap. Synco is inactive in that snapshot (`active=0`,
`deliveredHz=0`); its cumulative missed/not-started counts do not establish the
failure cause or a current skip fraction. The readback does not support the
initial hypotheses of latency mode Off or a two-step configuration.

The operator initially attributes the issue to dmcomb's trigger delay,
reporting an improvement after changing a setting believed to be 5 us to
10 us. Further operator testing identifies the problematic setting as
**50 us**, ten times the intended 5 us. The initial delay value was not
captured in the supplied INDI snapshot.
DELAY mode 4 uses a requested `nanosleep` between counter checks; the requested
delay does not establish the actual polling period or a monotonic relationship
with output jitter.

With the delay restored to 10 us and synco enabled, the operator reports
`acq = 1.25e-4 +/- 6.2e-6` s: 125 us mean with 6.2 us RMS interval jitter.
This confirms a restored 8 kHz mean controller receipt cadence and substantially
reduced jitter.

The operator then confirms dmcomb at 5 us and reports
`acq = 1.25e-4 +/- 3.7e-6` s with synco enabled: 125 us mean and 3.7 us RMS
interval jitter, close to the prior roughly 3.2--3.6 us qualification range.
The subsequent identification of the 50 us setting resolves the apparent
day-to-day regression as an unintended polling-delay setting. The verified
5 us measurement is consistent with the preceding baseline.

The current working setting is again 5 us (`0.000005` s); the problematic
50 us setting is `0.000050` s. Mean cadence and reported jitter are back near
the preceding baseline. No application code or instrument configuration was
changed by the agent; the operator performed the instrument checks and delay
changes.

## 2026-11-05: timing-overhead audit before the fgtimes-only change

The operator asks how much measurement remains in the loop, particularly in
ocam2KCtrl and bmcCtrl. Live readbacks confirm:

| Property | Reported value |
| --- | ---: |
| `camwfs.fg_timing_quality.enabled` | 1 |
| `camwfs.fg_timing_quality.stagesEnabled` | 1 |
| `camwfs.fg_timing_quality.traceEnabled` | 1 |
| `camwfs.fg_timing_quality.traceCapacity` | 16 |
| `dmtweeter.dm_timing_quality.enabled` | 1 |
| `dmtweeter.dm_timing_delivery.enabled` | 1 |

All six values also match the preceding successful capture
`synco-production-20261105T052815Z`. These diagnostics are still active, although
their enable state is not a newly observed difference from that capture.

Source inspection identifies the following application capture work:

- The normal successful OCAM path performs 13 additional monotonic clock reads
  per frame with stage timing enabled: one at event begin, six EDT markers,
  and six framegrabber markers. Event begin and EDT both capture wait entry.
  Existing SDK acquisition timestamps and the stream's realtime write timestamp
  are separate. At 2 kHz this is about 26,000 diagnostic clock reads per second.
- With configured reader release, a successful full BMC pipeline command has
  17 collector clock reads across its monitor and post stages, plus two for
  enabled delivery-clock bracketing. At 8 kHz those 19 reads amount to about
  152,000 per second across the two threads. This excludes SDK worker,
  timeout, input-identity and reader/completion protocol timestamps.
- Capture includes fixed event initialization/copies, marker ordering checks
  and a bounded queue handoff. History, statistics, excursion classification,
  trace construction, INDI formatting and telemetry emission run on ordinary
  app paths. Trace capacity 16 bounds retained contexts; it does not limit
  camera stage timestamp collection to 16 frames.
- These operation counts do not measure current instrument latency or CPU
  cost. The older workstation capture benchmark predates the complete current
  pipeline and delivery path; it does not establish their overhead.

At the time of this audit, a first operator-applied reduction that retains basic
`fgtimes` is:

```ini
# camwfs.conf
[framegrabber]
stageTiming=false
stageTraceMax=0
```

```ini
# dmtweeter.conf
[dm]
deliveryClock=none
```

These settings require application restarts. Both camera settings must change
together: a positive trace capacity requires stage timing. Basic camera paired
timestamps and event handoff remain. BMC delivery bracketing loses its two
extra reads, while the common DM collector still records processing stages.
At this point there is no independent DM stage-timing switch; `dm.latencyTime=0`
disables the whole collector, including compatible DM `fgtimes`. A separate
cadence-only mode would allow further reduction while retaining rate monitoring.

Keep SDK/pipeline behavior and reader-release signals unchanged in an overhead
comparison: they serve command ownership, timeout handling and synco readiness,
independently of timing summaries. No application code or instrument setting
was changed during this audit. Only this engineering record was updated.

## 2026-11-05: follow-up fgtimes-only normal operation

Following the audit, the operator requests that normal camera and DM operation
collect and report only compatible `fgtimes`. That change is implemented on the
separate `jrmales/fgtimes-only` branch:

- [6eb3ec0b](https://github.com/magao-x/MagAOX/commit/6eb3ec0b68e1a72c3d4a5b0cc80dc2d5d9ad7b68)
  adds the normal mode, regression tests and its execution record.
- [baf736ad](https://github.com/magao-x/MagAOX/commit/baf736ade019cdb4cf67ea6196a2da1062d68fbe)
  documents normal operation and the detailed diagnostic opt-in.

With that implementation, use camera `[framegrabber] stageTiming=false` together
with `stageTraceMax=0`, and DM `[dm] stageTiming=false` (the new DM default).
Detailed stage, trace and delivery measurements become opt-in. Required SDK,
reader-release and completion clocks/signals remain independent of profiling.
All 149 software test cases pass (4239 assertions).

Applying the new mode requires rebuilding/installing and operator-controlled
application restarts. The instrument measurements above precede that deployment;
no post-change instrument latency or jitter measurement has been supplied.
