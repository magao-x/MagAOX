# Coordinated zero-auxiliary empirical trial

## Current endpoint and timing-interpretation clarification

The user now explicitly places the desired 125 us cadence at dmcomb's combined
semaphore trigger to the controller, with sufficiently earlier auxiliary
publication/provider deadlines. The default finite trial schedules auxiliary
publication on its primary-observer phase grid. The opt-in 2026-11-03 test at
the end of this record advances that grid by 5 us and measures the result. The
[combined-trigger deadline plan](../2026-11/dm-syncopation-combined-trigger-deadlines.md)
records the revision, existing-capture phase analysis and experimental change.

**Exact-main prefix correction:** throughout the capture history below,
`zero_test.controllerToSdk` actually measures native SDK entry minus the
reader record's callback/read-begin timestamp, which follows input-identity
capture. It does not start at the first semaphore-wait receipt. Aggregate
`dmtimes.receiptSdk` does start at that earlier receipt. Baseline/active
subtractions of +0.724, +0.730, +0.570 and +0.496 us using those different
origins are not measurements of added main latency; that interpretation is
withdrawn. Raw values, command joins, primary-observer-to-SDK and native SDK
service remain valid. A matched true-receipt export is needed for the intended
main-only wake-to-SDK comparison. The new plan gives the source evidence.

## Objective and decision

Keep the genuine main loop at 2 kHz and insert zero-valued auxiliary commands on
the established cadence/phase grid. Two total steps target 4 kHz effective
writes: main step 0 and one auxiliary at +250 us. Three target 6 kHz: main step 0
and auxiliaries at +166.667/+333.333 us in a nominal 500 us cycle. There is no
extra step-0 post, independent 4/6 kHz auxiliary timer, or paused main writer.

The operator chose a **finite empirical attempt that measures possible main
impact**. The trial does not retain the shadow's conservative prediction of zero
added future main SDK wait as an admission veto. Actual reader clearance,
identity/lifecycle and finite-deadline checks remain. Requested cadence is not
qualification of undisturbed main timing.

This follows `synco-sim-20261031T173634Z`: measured SDK-service-through-readiness
maximum 112.212 us and receipt-through-callback-finish maximum 150.133 us, with no
new overload/rejection/coalescing increments at genuine 2 kHz. Overlap permits
SDK service to limit throughput even when whole-job latency exceeds one SDK
service interval; actual interleaved behavior still requires measurement. See
[the delivery/preparation record](dm-syncopation-delivery-preparation-measurements.md#fifo-post-worker-comparison-synco-sim-20261031t173634z).

All instrument configuration, restarts and controls are operator actions. Agents
edit/build software and test private workstation fixtures only.

## Execution contract

The explicit `zeroTest.enabled=true` startup mode is distinct from the old
arrival-relative scheduler and pure shadow. All three simulation flags must be
false: `enabled`, `readerReleaseReady`, `pipelineBackend`. Existing shadow
no-write behavior remains a separate contract; turning simulation Off alone
does not select the trial.

The merge overlay
[`dmtweeter_synco_zero_test.conf`](../../../apps/dmSyncopation/dmtweeter_synco_zero_test.conf)
sets zero-test channel `dm01disp07`, active `durationSec=120`, `lateUs=5`, two
steps initially, and reader-release observation on FIFO30/spare57. Merge its
sections into the working instance config; do not replace primary/routing/CPU
settings. The app independently bounds arm/warmup at 30 seconds and active
duration at at most 120 seconds. Off cancels; a new On begins a
fresh finite trial after the established ten-second cadence qualification,
which can warm while disabled. Change two/three steps only while Off.

Each cycle's fixed main-labelled grid retains targets through genuine late/early
main observations. Two steps provide one auxiliary opportunity, three provide
two. Each actual slot copies zeros and publishes through the selected auxiliary's
normal counter/timestamp/semaphore path. Step 0 belongs to the genuine main;
the trial never writes or posts the primary. Expired/conflicting/ambiguous slots
are skipped, never replayed in a catch-up burst. Exact reader-release evidence
protects shared arrays; it does not prove native SDK idleness.

The existing selected auxiliary must already contain zeros and be exclusively
available. No unknown nonzero contribution is silently erased. Stop that
channel's other provider while keeping the normal main writer running. The zero
contribution remains retained on cancellation or completion.

## Trigger and controller path

Retain dmcomb mode 4 / DELAY / 10 us. The inspected `AOloopControl_DM_comb.c`
recombines when the sum of participating channel counters changes; an active
auxiliary publication requests a combination without a dirty primary semaphore.
Mode 3 waiting only on the primary would not implement the requested auxiliary
slots. Exclude astrogrid/ZPO/reference-update roles for the selected channel.

Keep the existing camera/main/reconstruction and closed-loop workload at 2 kHz,
camera synchronization On and RTC latency On. Retain synco's CPU41 monitor,
reader-release observer FIFO30/spare57, SDK worker FIFO55/spare56 and ordered
post worker FIFO30/spare31. Preserve three-credit pipeline and matching delivery
clock diagnostics. Synco stays running as the coordinated publisher. The SDK
result stream defaults to `dmtweeter_sdk_done` (`zeroTest.sdkStream`).

## Operator capture

The concrete procedure is
[zero-aux-rate-test.md](../../../apps/dmSyncopation/doc/zero-aux-rate-test.md).
Build/install the updated app, merge the overlay and restart synco
only. Keep main-loop/settings unchanged. Capture the 4 kHz target first and
preserve its result before attempting 6 kHz.

```text
apps/dmSyncopation/scripts/capture-zero-aux-test.sh RATE [SECONDS]
```

`RATE=4000|6000` is total effective cadence mapped to two/three steps against
running 2 kHz main. Active seconds default to 120, range 1–120. There is no
independent/additive rate mode. The wrapper starts three telemetry readers,
saves bounded before snapshots, requires test mode and Off, checks camera
2 kHz/synchronization and RTC latency, sets disabled steps and verifies them,
then requests On. Warmup is bounded at 30 seconds. It retains zero-test history,
requests Off at completion or requested shorter duration, takes after snapshots
and stops only its own readers.

The wrapper ignores a stale previous finished readback until new progress is
observed. A finished attempt without an observed active phase, or one ending
before the requested active duration, is incomplete and returns nonzero.

A pre-existing On run is refused without cancelling it. Cleanup requests Off
only if this wrapper attempted On, including an uncertain enable timeout. The
app's configured finite duration remains an independent upper bound. No main,
camera, controller, dmcomb, CPU or persistent config setting is changed by the
wrapper. Development tests substitute fake services; no real INDI is called.

Results under `$HOME/tmp/zero-aux-<total-rate>Hz-<UTC>` include simultaneous
synco/controller/camera telemetry; before/configured/after INDI; revision and
requested settings; control statuses and zero-test history.

## Measurement and limits

Actual posted/acknowledged/skipped outcomes are distinct from shadow `wouldRun`.
Retain per-slot timing/residuals and skip reasons, resource faults, readiness
invalid/drop coverage, exact main/combined/release matches and event losses.
Compare actual combined/controller cadence against 4/6 kHz. Zero pixels do not
make a missing publication equivalent to delivery.

Inspect capture-local BMC overload/rejection/coalescing, superseded/cancelled
jobs, SDK failures/timeouts and queue losses as well as timing-event quality.
Admission failures can precede timing handoff. Compare publication/receipt/send
cadence, delivery/preparation, SDK service and post tails with simultaneous
camera/primary excursion measurements.

The exact main-impact join supplies primary-to-SDK and controller-to-SDK timing
and SDK service where classified. Those prefixes include delivery/preparation
and are not an isolated SDK-wait measurement. Aggregate `sdkWait` mixes main and
auxiliary jobs. Retain matched/unmatched/unclassified/drop populations with each
conclusion; never substitute a nearby command for unavailable exact linkage.

Mode-4 dmcomb can have a blank named-primary process trace. In that case the
exact main-impact prefixes are unavailable and `mainImpactUnclassified` grows,
while actual coordinated slot attempts still proceed through exact combined
counter acknowledgements and reader-release checks. This is an explicit
coverage limitation, not evidence that aggregate SDK wait is main-only.

This finite mode measures the selected grid, zero contribution, deployed path,
placement and workload. It does not automatically qualify nonzero providers,
future production bounds, arbitrary main excursions or zero added main latency.
Leave the trial Off and retain each capture before retuning.

## Verification

Trial records use the existing text-log wire format; installed logdump needs no
format/code update. Only dmSyncopation must be rebuilt for these records.

The capture wrapper passes `bash -n` and thirteen offline fake-service tests
for two/three-step selection, pre-existing On ownership, invalid camera/test
mode, collector startup failure, uncertain enable cleanup, polling failure,
operator interruption, shorter active duration, a stale prior finished readback,
warmup-only timeout, premature active finish and invalid inputs. No test
invokes real INDI or instrument controls.

Workstation C++ verification passes 95 application cases (972,838 assertions),
58 cadence/shadow cases (54,363 assertions), and ten exact-identity timing cases
(69 assertions). The application cases include nonzero primary-pixel
preservation, both requested total rates, an 80 us late main without retiming,
independent slot skipping, delayed main combination/rebaselining, lost release
evidence, source replacement, ownership intrusion, startup validation, cancellation,
wall-clock expiry without primary callbacks and compatibility with the existing
text telemetry reader. All stream fixtures use private workstation directories.
The production dmSyncopation build, `clang-format` checks and whitespace checks
also pass.

Only the explicit zero-test path performs real zero auxiliary publications;
ordinary scheduler and shadow defaults retain their existing behavior. Hardware
delivery, disturbance and timing-tail qualification remain the next operator test.

## First setup attempts: October 31, 2026

`zero-aux-4000Hz-20261031T234136Z` and
`zero-aux-4000Hz-20261031T234240Z` both stopped during baseline setup. Neither
contains a control log or enable request; neither is a zero-write performance
measurement. Both record checkout `b97d424f`; this does not identify the running
application binary.

Both baseline synco queries returned status 1 with
`No dmtweeter-synco.*.* from localhost:7624`. The other four baseline INDI
queries succeeded. Camera readbacks were 2000 Hz with synchronization On in both
attempts. RTC latency was Off in the first and On in the second, leaving the
missing synco response as the second attempt's blocker. Collector status 143
reflects the script's requested cleanup after refusal.

Each synco telemetry file contains only an old shadow-pipeline record at
23:33:42.824064301 UTC, preceding both attempts; there are no zero-test records.
Camera and controller telemetry are fresh at the capture times. Startup/error
logs and process status are needed to distinguish an exited app from an
initialization or communications problem; stale telemetry is not evidence of
the newly configured mode running.

The capture helper now reports every failed baseline command, collector and
property gate with actual versus expected values, while preserving the same
no-control-on-failure behavior. Seventeen offline fake-service cases pass;
no instrument commands are executed by those tests.

The subsequent operator startup log identifies the launch failure. The installed
app reported `2d8c104a` at 23:33:48.612972790 UTC, while the capture checkout was
`b97d424f`. It rejected all four `zeroTest` config targets as unrecognized and
exited on invalid configuration; the process query found no dmSyncopation
threads. That older validator also rejects configured completion/read-release
observers when simulation is Off. The new trial configuration therefore cannot
run under that binary. A clean rebuild, install and synco-only restart are
required before retrying; the observations do not identify whether the earlier
build, install or restart step was missed. Current build dependencies include
the app header and both new implementation/timing headers.

## First complete 4 kHz trial: `zero-aux-4000Hz-20261101T040937Z`

The capture ran in the explicit coordinated zero-test mode, with two total
steps, and completed **120.000099036 s** active time. Its estimated active wall
interval was November 1, 04:09:55.706–04:11:55.706 UTC (October 31,
21:09:55.706–21:11:55.706 Phoenix time). Deferred event-log headers are drain
times; the estimate uses summary timestamps minus monotonic elapsed time.

All baseline/configured/after queries and 44 polls succeeded. Step selection,
On and Off requests succeeded; collector status 143 is requested cleanup.
Runtime properties show the new mode, and checkout is `b97d424f`. No startup log
independently records the installed binary hash. Synco was Off before and after.

The camera remained bin2, 2000 Hz, synchronization On, EM gain 1, shutter SHUT,
and target −45 C. Holoop snapshots show closed loop, scalar gain 1, unchanged
multiplier; modal gains are not captured. RTC latency is On. BMC readbacks
confirm FIFO55/single CPU for SDK, FIFO30/single CPU for ordered post, and three
credits. Actual CPU-number lists and external GPU workload are not archived.
The operator confirmed that the AO loop remained stable, modal gains stayed
at zero, and settings/workload were unchanged during the capture. Stability at
zero effective modal gain does not by itself exclude a small added main latency;
the unchanged workload supports the within-capture timing comparison.

### Actual delivery

| Active-trial population | Result |
| --- | ---: |
| Genuine main observations | 240,000, all sequential |
| Auxiliary opportunities | 240,000 |
| Auxiliary posts / acknowledgements | 229,831 / 229,831 |
| Auxiliary skips | 10,169 (4.2371%) |
| Reader missing / invalid / dropped | 0 / 0 / 0 |
| Resource faults | 0 |
| Main output acknowledgements | 239,999; one initial unmatched main |
| Exact main SDK / reader joins | 239,999 / 239,999 |
| Main timing invalid captures | 2 |
| SDK gaps / failures; event drops | 0 / 0; 0 |

The union of main and acknowledged auxiliary identities covers **469,831
contiguous combined counters**, `301000419` through `301470249`. Their producer
timestamps measure **3915.254778 Hz** actual combined cadence. Every posted
auxiliary produced an acknowledged combination; the deficit comes from skipped
auxiliary opportunities, with no observed dmcomb coalescing in this population.

Combined intervals have mean 255.411220 us, standard deviation 37.307615 us and
range 210.738–536.873 us. Exactly 10,169 intervals exceed 350 us, matching the
skips; none are below 200 us or above 600 us. Genuine-main combined intervals
remain mean 500.000443 us, standard deviation 6.499861 us, range
458.606–538.116 us. Main-to-aux intervals average 254.632673 us; aux-to-next-main
intervals average 245.305756 us.

### Skip classification needs correction

The counters label 10,168 skips `primaryChanged` and one `unclassified` from the
initial unmatched main. **This is not evidence of 10,168 early main arrivals.**
The final compound guard in `processZeroTestPrimary` also assigns late deadline,
resource and write-flag failures to `primaryChanged`.

All 10,168 outcome-4 events occur **5.031–6.746 us after target**, median
5.3135 us, against `lateUs=5`. The next main callback is another
234.342–279.668 us later. This strongly points to the final lateness check,
although these POD events do not preserve which individual guard failed and
cannot exclude transient write-flag or resource conditions.

Successful post-completion residuals have median 3.731 us, mean 3.87057 us,
p99 5.725 us, maximum 21.112 us. That stamp is after the publication call returns;
the admission deadline check occurs before the copy/publication. Thus successful
completion residuals above 5 us do not contradict the configured admission rule.
Separate the final guard's skip reasons before attributing another run's losses
to main timing; no late-limit or production bound is changed by this analysis.

### Backend and main timing

Controller INDI endpoints show **no new overloads, coalescing, rejections,
SDK failures/timeouts, preparation/post failures, supersession, cancellation,
vector rejection, reader-unqualified outcomes or timing queue losses/gaps**.
Historical rejected=2 and reader-unqualified=1 do not grow. Endpoint admitted,
completed and retired each increase by 516,763 across the whole capture,
including main-only baseline and drain time. Timing events/submitted increase
by 516,764; different sampling endpoints explain the one-command difference.

For complete steady-active rolling windows, native SDK call mean has median
103.711 us and maximum 109.286 us. SDK service through worker readiness has
median mean 106.068 us and maximum **111.751 us**. Full receipt-to-native-entry
prefix has median mean 16.275 us and maximum **25.789 us**; this exceeds the
provisional shadow `prepareUs=25` by 0.789 us. Pure preparation's 21.241 us
maximum excludes handoff and cannot substitute for the complete prefix.
The 120 us SDK assumption is not exceeded.

Reader release reaches 36.699 us; publication work 21.390 us; whole
receipt-to-callback-finish 168.428 us. Whole-job latency slightly exceeding one
6 kHz period does not by itself violate pipelined throughput with three credits.
The observed minimum receipt interval is 207.612 us and minimum send interval
205.919 us, well above the measured SDK-service maximum. No backend capacity
failure is demonstrated at this attempted 4 kHz cadence.

Exact main-only joins, available in this capture, measure:

| Duration | Mean, us | Maximum, us |
| --- | ---: | ---: |
| Primary observer receipt → native SDK entry | 30.158630 | 43.442 |
| Controller receipt → native SDK entry | 15.500755 | 23.054 |
| Main native SDK entry → return | 103.696589 | 107.973 |

There is no same-definition paired main-only baseline for these exact joins:
they reset on On and classify active mains. Prefix durations include delivery
or preparation and are not isolated added SDK queue wait. `sdkWait[n=0]` is the
monitor's SDK-result wait population, absent in the detached pipeline; it does
not establish zero main queue wait.

### Camera and measurement coverage

There are 152 consecutive valid controller summaries and 143 consecutive valid
camera stage summaries, with no malformed or backward-timestamp rows. Camera
endpoints add 299,966 published frames, with no new frame/counter gaps,
rejected/failed/invalid/dropped frames. There are 25 excluded wide controller
delivery-clock brackets and 59 wide camera write brackets; retain those coverage
exclusions when describing maxima.

Median rolling camera acquisition period stays 500.0005 us; acquisition RMS is
0.2890 us in the main-only baseline and 0.2957 us in the steady active segment.
Publication RMS medians are 0.9168 and 0.8384 us, while the active worst rolling
RMS is 2.9433 us. Acquisition extrema are 480–525 us before and 479–526 us
active. Mean acquisition-to-write has rolling median 46.8709 us before,
47.2197 us active and 47.0361 us after. These are overlapping rolling-window
comparisons, not pooled independent-sample statistics.

Temperature housekeeping remains on its 10 s cadence; gain/FPS query counts do
not change. Ten qualified camera trace contexts include roughly 489–510 us of
serial overlap, two before and eight active. No frame-scale timing excursion or
frame-behind condition is demonstrated by this capture.

The first complete trial therefore establishes successful coordinated delivery
of all attempted zero commands, approximately 3.915 kHz achieved cadence, and
backend headroom at this rate. It does not qualify 6 kHz or prove unchanged main
latency. The immediate measurement issue is the compound skip label and its
5 us deadline-edge population.

## Explicit per-slot outcome labels

The label cleanup follows the first complete 4 kHz trial above. It changes
attribution and event coverage, not the scheduling policy: `lateUs=5`, frozen
targets, actual reader clearance and the finite-trial rules are unchanged.
The earlier 10,168 compound `primaryChanged` events remain ambiguous records;
they are not retroactively relabelled as proven late or early-main events.

Subsequent summaries/events identify `labelVersion=2`; events retain numeric
`outcome` and add named `action`, `reason` and `stage`. A pending slot that is
deliberately not actively published is `action=skipped`, even if its zero copy
already occurred. The stage identifies
main acknowledgement, wait/deadline, combined/readiness checks, `final_guard`,
`post_copy_guard` or post-publication acknowledgement. Successful completion
residuals remain after-call timestamps, separate from the 5 us admission check.

Skip reasons are disjoint: `late`, `readerMissing`, `primaryChanged`,
`primaryWriting`, `resourceUnavailable`, `waitError`, `combinedUnclassified`,
`cancelled`, `durationExpired` and `invalidTarget`. All pre-publication deadline
guards count `late`; `primaryChanged` now requires an actual primary counter
change, and an asserted primary write flag has its own counter. Each scheduled
auxiliary slot gets one outcome event, including each dependent slot skipped
when the genuine main combination could not be matched. Two/three total steps
still mean one/two auxiliary opportunities per main cycle.

An active publication attempt without a confirmed expected combined update is
`action=unacknowledged` and increments the separate `unacknowledged` counter.
It is part of `posted`, not `skipped`. Thus `posted` separates into acknowledged
and unacknowledged attempts, while `skipped` is the sum of its reason counters.
The former compound `unclassified` population is separated into pre-publication
`combinedUnclassified` and post-publication acknowledgement failure. Reader
invalid/drop coverage and resource faults outside a scheduled slot must not be
added as additional skips.

Post-publication reasons distinguish `ack_timeout`, `ack_ambiguous` and
`publication_failed`, as well as any captured cancellation, primary, resource
or wait failure. `wait_error` retains the actual numeric `waitResult`. The stage
and action retain the difference between an unpublished skip and a posted
attempt even when they share a rejecting condition's name.

The operator-facing counter table is in
[the trial procedure](../../../apps/dmSyncopation/doc/zero-aux-rate-test.md#read-actual-slot-outcomes).
The implementation's focused regression checks accompany the functional change;
no historical capture, controller setting, timing bound or hardware state is
changed by this documentation update.

Workstation validation for the label cleanup passes: the complete app suite has
97 cases / 1,393,294 assertions, including 14 coordinated-trial cases. Fault
injection separately exercises the initial, final pre-copy and post-copy late
checks, primary counter changes versus an active write flag, competing auxiliary
ownership, cancellation, finite expiry and wait errors. Tests also verify one
event per denied slot, the disjoint skip-reason sum, the posted
acknowledged/unacknowledged partition, and a successful later slot following an
earlier deadline skip. Final/post-copy late events retain the exact rejecting
clock sample. Production `dmSyncopation` builds successfully; all 17 offline
capture-helper tests, shell syntax, clang-format and whitespace checks pass.
Instrument qualification awaits rebuilding/restarting synco and another capture;
the text-log format remains readable by the existing reader.

## Label-qualified 4 kHz trial: `zero-aux-4000Hz-20261101T050156Z`

The workstation copy is under
`/home/jrmales/tmp/home/jrmales/tmp/zero-aux-4000Hz-20261101T050156Z`, with the
matching archive in `/home/jrmales/tmp`. The recorded checkout is `5baea984`;
the observed `labelVersion=2` records establish that the new labels were running.
The capture does not archive a startup binary hash. All snapshot/control
commands and 44 polls succeed; collector status 143 follows requested shutdown.
The twelve error files are empty.

The active interval lasts 120.000102807 s, approximately 05:02:14.1086 through
05:04:14.1087 UTC on November 1 (October 31 in Phoenix). Use event observation
times and summary elapsed time for this interval, rather than the delayed
telemetry formatting timestamps. The operator confirms the AO loop was stable,
modal gains remained zero, and settings/workload were unchanged apart from
rebuilding/restarting synco. The snapshots retain 2 kHz camera acquisition,
synchronization On, bin2, EM gain 1, shutter SHUT, target -45 C, scalar loop gain
1 and RTC latency On. The effective modal gain remains zero.

### Actual auxiliary outcomes

| Population | Count |
| --- | ---: |
| Sequential genuine main observations | 240,000 |
| Auxiliary opportunities | 240,000 |
| Posted / acknowledged | 230,571 / 230,571 |
| Posted but unacknowledged | 0 |
| Skipped, total | 9,429 (3.92875%) |
| Late at post-copy guard | 9,426 |
| Late at initial deadline check | 2 |
| Initial unmatched main combination | 1 |
| Primary changed / writing, reader missing, resource or wait failures | 0 |
| Reader invalid / drops, SDK gaps / failures, event drops | 0 |

There are exactly 480,002 event rows: one main and one auxiliary outcome for
each main cycle, plus start/finish. All post-copy late events passed both the
initial and final pre-copy deadline checks, copied zeros, then rejected the
active publication at the unchanged 5 us limit. The channel counter/semaphore
was not published for those occurrences.

Post-copy rejection samples are 5.001-6.752 us after the frozen slot target:
median 5.252 us, p95 5.561 us and p99 5.76025 us. The two initial-deadline
rejections are 5.266 and 6.577 us late. This is total target-relative lateness,
not the duration of the zero copy. Without a retained successful pre-copy clock
sample, the actual copy/check duration cannot be isolated. The next genuine
main callback is still 231.016-281.562 us later (median 244.201 us) for these
late points. They do not represent early main arrivals or lost reader clearance.

The exact event identities cover 470,571 consecutive combined counters,
307507049 through 307977619, with no gaps or conflicting publication stamps.
Achieved combined cadence is 3,921.421341 Hz, versus 3,915.254778 Hz previously.
There are 740 fewer skips, a 0.308333 percentage-point difference; this variation
does not establish a performance improvement from the label-only change.
Combined intervals span 209.325-549.027 us. Exactly 9,429 intervals exceed
350 us, matching the skipped opportunities; none are below 200 or above 600 us.
Main-only combined intervals have mean 500.000464 us, RMS 6.256431 us and range
456.462-549.027 us.

The ordinary combiner observer separately adds two counter discontinuity
observations and three duplicates, with no queue drops. These imperfect observer
counts do not erase the complete counter coverage in the exact trial events.
Exact main SDK and reader matches both total 239,999; there are no unmatched
classified SDK joins or unclassified main-impact identities. Retain the ten
invalid timing-evidence observations as a coverage diagnostic rather than
describing every timing observation as valid.

### Controller and camera checks

There are 152 consecutive valid controller timing summaries. Across the INDI
endpoints, admitted/completed/retired and timed/submitted events each increase
by 517,521, with no new overload, coalescing, rejection, failure, timeout,
preparation/post failure, supersession, cancellation, unqualified reader event
or timing loss. The broader first/last telemetry endpoints cover 533,561
submissions. Delivery-clock measurements exclude 79 new wide brackets.

For 118 complete steady-active controller windows, native SDK call maximum is
106.912 us and SDK-through-worker-readiness maximum is 109.347 us. Full
controller-receipt-to-SDK-entry maximum is 22.242 us, preparation 19.306 us,
reader release 27.933 us, publication 15.308 us and receipt-to-callback-finish
153.730 us. Whole-capture SDK readiness reaches 111.651 us. No capacity failure
is demonstrated. Exact main-only primary-observer-receipt-to-SDK timing is
30.373852 us mean / 41.699 us maximum; controller receipt to SDK is 15.552648 /
21.501 us, and native SDK service is 103.698884 / 106.912 us. These prefixes do
not isolate added main SDK wait, and the prior-run comparison is not a paired
main-only baseline using the same exact joins.

Camera telemetry has 144 consecutive valid stage summaries and 301,973
additional published frames over its broader first/last endpoints, with no new
gaps, failures, invalid/dropped samples or wide clock brackets. Median rolling
acquisition RMS is 0.2923 us before, 0.2935 us steady-active and 0.2860 us after;
publication RMS is 0.9004, 0.8291 and 0.8708 us. Active acquisition intervals span
479-525 us; the worst active rolling publication RMS is 1.1404 us. Temperature
polls and serial transactions each increase by 13, while FPS/gain query counts
stay fixed. No measured frame-scale excursion or camera timing regression is
demonstrated. These are comparisons of overlapping two-second windows.

The dominant limitation demonstrated by this run is the auxiliary deadline
cutoff after the zero copy. At analysis time, a same-setup finite 4 kHz trial
with `zeroTest.lateUs=10` was proposed because every recorded late rejection was
earlier than that limit. **That proposal is superseded by the user's command-ready
deadline contract below.** The wider-lateness experiment was not performed as
part of this capture; the recorded 5 us results above are unchanged. No timing
setting or scheduler code was changed by the capture analysis itself.

## Ideal zero publisher and command-ready deadline

The user chose to separate command preparation from the timed publication,
instead of widening publication lateness to compensate for copying after the
target. This finite trial has an ideal constant-zero source: the command is
already available. It is not a simulation of an external producer, notification
transport or full publisher protocol.

For each immutable auxiliary target `T`, obtain exact reader clearance and stage
the complete zero frame into the auxiliary destination early. Clear its shared
write flag when the passive copy finishes; do not advance its counter, writetime
or semaphores. A copy finishing **at or before `T`** meets the ready deadline.
Then wait for `T`, revalidate the frozen primary/combined identities, auxiliary
ownership and reader clearance, and issue the ordinary active publication.
Preparation is not repeated at the publication wake.

If exact reader clearance is unavailable by `T`, skip with `readerMissing`.
A copy that starts or finishes after `T` is an exclusive `preparationLate` skip
(`reason=preparation_late`, `stage=preparation`). Scheduler publication wake and
admission lateness remain separate `late` outcomes against the existing
`T + lateUs` limit, with `lateUs=5` unchanged. `deadline` and `publication_guard`
identify those publication checks; the former ideal-mode `post_copy_guard` path
is removed. Cancellation and identity/resource/readiness changes after staging
still revoke the pending publication. There is no catch-up burst or movement of
subsequent committed targets.

The implementation identifies this behavior as `labelVersion=3` and
`idealPublisher=1` in the trial readbacks/summary. Only dmSyncopation needs a
rebuild; the existing text-log wire format remains readable by installed
logdump. The capture helper checks the ideal-publisher readback so an older
copy-after-target binary cannot silently satisfy the next trial's setup.

`staged` counts completed zero copies, including an overrun or a command later
cancelled before publication. It overlaps posted/skipped populations and is not
another skip-reason counter. Each outcome event carries `stagedNs`: the actual
copy-completion time, including an overrun, or zero when no copy completed.
For positive `stagedNs`, `targetNs - stagedNs` supplies measured staging lead;
a negative result documents preparation that missed its ready deadline.
`preparationLate` is included once in the disjoint skip-reason sum. The existing
`posted = acknowledged + unacknowledged` and one-outcome-per-auxiliary-slot
accounting are retained.

This ideal-source trial establishes the consumer/scheduler's behavior when the
command can always be supplied. A future production provider contract must
instead require a coherent command by its advertised ready deadline and skip
if missing. No such external-provider interface is implemented by this change.
The 4 kHz capture above remains evidence of the previous label-version-2
copy-after-target policy; its post-copy residuals cannot retrospectively supply
the new `stagedNs` measurements. Preserve its measurements when comparing the
next finite trial.

Workstation verification passes: the full app suite has 100 cases / 1,935,176
assertions, including the ideal-staging boundary and revocation cases. Tests
verify that a 20 us staging delay still succeeds when completed early, exact
completion at `T-1 ns` and `T` qualifies while `T+1 ns` skips only that slot,
reader evidence observed after the first deadline can qualify the second slot,
and revocation after staging prevents publication. Wait checkpoints verify a
clear write flag and unchanged auxiliary counter/writetime/semaphore before the
first target. Existing main identity, cancellation, expiry, resource, publication
acknowledgement and independent-slot accounting tests remain in the full suite.
The capture helper passes all 18 private fake-service tests, including refusal
of a binary lacking the ideal-publisher capability. Production dmSyncopation
build, shell syntax, clang-format and whitespace checks also pass. No instrument
commands were run. The next operator test remains 4 kHz with `lateUs=5` after
rebuilding/restarting synco; logdump and the DM controller need no rebuild for
this change.

## Ideal-publisher 4 kHz capture: `zero-aux-4000Hz-20261101T145916Z`

The local directory and archive are under `/home/jrmales/tmp`. Recorded checkout
is `fda6e4e2`; the running trial exposes `labelVersion=3` and `idealPublisher=1`.
All snapshot/control commands and 44 polls succeed; the twelve error files are
empty and collector status 143 follows requested termination. The operator
confirms unchanged settings/workload, zero modal gains and stable closed-loop
operation. Camera acquisition remains 2 kHz with synchronization On, bin2,
EM gain 1, shutter SHUT and target -45 C; scalar loop gain is 1 and RTC latency
is On. Camera/controller sessions differ from the earlier run, so comparisons
between captures are observational rather than one uninterrupted process run.

The exact active monotonic duration is 120.000087453 s. Paired camera
writer-clock observations map the interval approximately to 14:59:34.6095 through
15:01:34.6096 UTC on November 1 (07:59 through 08:01 Phoenix). Telemetry emission
times are later than the events they describe; phase comparisons exclude
transition windows.

### Early staging and actual delivery

| Population | Count |
| --- | ---: |
| Sequential genuine main observations / auxiliary opportunities | 240,000 / 240,000 |
| Completed passive zero copies | 239,996 |
| Posted / acknowledged auxiliary commands | 239,995 / 239,995 |
| Posted but unacknowledged | 0 |
| Skips, total | 5 (0.00208333%) |
| Publication lateness, initial deadline check | 1 |
| Unmatched main combinations | 4 |
| Preparation late / reader missing / primary changed or writing | 0 |
| Reader invalid / drops, resources/waits, SDK gaps/failures, event drops | 0 |

All observed passive copies finish before their immutable publication targets.
Staging lead is 160.067-221.005 us, median 200.065 us, p1 190.155 us and
p99 209.698 us. The single late occurrence was already staged 206.084 us early;
its publication check is 5.095 us after target, 0.095 us beyond `lateUs=5`.
Successful post-return residuals have median 2.405 us and range 1.024-22.651 us.
Those are after-call timestamps, not exact publication instants or proof that
the before-call deadline check exceeded its limit.

The recurring 9,426 post-copy deadline skips from the preceding capture are
absent. This supports the intended ideal-source test: early preparation is
complete before the deadline, while publication occurs on the retained grid.
The 9,429 total skips previously have fallen to five without widening `lateUs`.

Combined counter span 314148627 through 314628621 represents 479,995 updates
and an effective cadence of approximately 3,999.955 Hz. The direct event-key
union contains 479,994 distinct identities and has one hole, counter 314586418,
in the unmatched-main cluster below. Report this exact-event coverage limitation
separately from the complete auxiliary acknowledgement population and the
controller's zero timing-counter-gap increment. Do not describe this capture's
direct event identities as fully contiguous.

### Remaining unmatched-main cluster

One unmatched main occurs at activation. The other three are consecutive main
identities 283198025-283198027, around 15:01:24.057 UTC. The first rejection is
259.690 us after its predicted main grid point and has no coherent combined key;
the following observations reacquire a combined identity and then encounter a
two-counter advance. The current main-association logic discards the empty
anchor and subsequently rebaselines, explaining the additional recovery skips.
These labels do not mean that three genuine main commands were deliberately
suppressed by synco.

The encompassing primary producer-timestamp window contains 758.833 us and
242.548 us intervals, with no primary counter gap. Its late-then-early paired
sum reaches 1001.381 us. The upstream trace's third row (row 2, writer PID
2523020, input inode 4719) has execution maximum 313.732 us, with late-conditioned
execution change +258.327 us and early-conditioned change -257.245 us; source
intervals for the late cases are 500.034-501.035 us. This supports an upstream
processing excursion in the same window, followed by association recovery.
These are window-conditioned trace populations, not an exact per-frame join
that proves the row's duration for each of the three named mains. The current
process name is not archived with that row's PID/inode.

Camera intervals in the qualified window containing this cluster are
498-502 us acquisition and 494.624-505.504 us publication, with no serial overlap
or excursion count. The camera trace emitted near 15:01:24 actually describes
exposures near 15:01:22.513, so it must not be aligned by its later log timestamp.
Controller SDK/readiness maxima in the cluster window are 104.127/106.752 us,
receipt-to-SDK 19.327 us and receipt-to-callback completion 141.267 us, with no
error increments. Neither the camera nor BMC processing evidence shows a
corresponding stall. Combiner execution is at most 6.622 us in the relevant
primary-trace window.

### Backend and camera comparison

Controller endpoints add 526,930 admissions/completions and 526,929
retirements/timed submissions. An in-flight job at the after endpoint accounts
for the one-command difference. Historical rejection/unqualified-reader counts
do not increase. There are no new overloads, coalescing, rejections, SDK failures
or timeouts, preparation/post failures, supersessions/cancellations, invalid
timing events, counter gaps or dropped events. Delivery measurements exclude
174 additional wide clock brackets. There are 151 consecutive valid controller
summaries; the broader telemetry endpoints cover 540,963 submissions.

For 118 complete steady-active controller windows, median publication-interval
mean is 250.000230 us and median RMS is 6.235154 us, versus 254.973 us and
35.819 us in the preceding copy-after-target run. Active publication intervals
span 212.231-505.845 us. Native SDK maximum is 111.150 us and service through
worker readiness 113.524 us, below the provisional 120 us service assumption.
Receipt-to-SDK maximum is 24.356 us, preparation 20.088 us, reader release
31.800 us, publication 21.280 us and receipt-to-callback completion 152.738 us.
The earlier 25.789 us receipt-to-SDK observation remains relevant; this run
does not turn 25 us into a globally qualified bound.

All 239,996 classified mains match exact SDK and reader evidence; twelve invalid
timing-capture observations remain a coverage diagnostic. Primary-observer
receipt to SDK entry is 30.817490 us mean / 42.170 us maximum; controller receipt
to SDK is 15.773277 / 23.585 us, and native SDK service is 103.818653 / 111.150 us.
These prefixes do not isolate added main SDK queue wait.

Camera telemetry has 144 consecutive valid stage summaries, with 303,004
additional published frames over its broader endpoints and no new gaps,
failures, invalid/dropped observations or wide brackets. Median rolling
acquisition RMS is 0.3147 us before, 0.3115 us active and 0.3001 us after;
publication RMS is 0.8521, 0.7926 and 0.8604 us. The worst active rolling
publication RMS is 1.1622 us. Temperature polling and serial transactions each
increase by 13; FPS/gain query counts are unchanged. These overlapping-window
comparisons show no material camera timing regression.

This run demonstrates near-complete coordinated 4 kHz delivery with the ideal
publisher and the original 5 us publication guard. It supports proceeding to
the finite three-step / 6 kHz trial with the same readiness/identity guards and
simultaneous main/controller/camera measurements. It does not itself qualify
6 kHz or prove zero added main latency. No code or instrument setting is changed
by this analysis.

## Ideal-publisher 6 kHz capture: `zero-aux-6000Hz-20261101T224640Z`

The workstation directory is
`/home/jrmales/tmp/home/jrmales/tmp/zero-aux-6000Hz-20261101T224640Z`, with its
archive under `/home/jrmales/tmp`. Recorded checkout remains `fda6e4e2`.
This is trial **2**: baseline summaries retain trial-1 results, which must not
be counted as new outcomes. The operator confirms stable AO operation, zero
modal gains and no other settings/workload changes from the 4 kHz run. The
helper changes disabled steps from two to three; camera/main cadence remains
2 kHz, synchronization and RTC latency On, bin2, EM gain 1, shutter SHUT, target
-45 C and scalar loop gain 1. The camera process/session/inode and worker CPU
policies remain unchanged from the preceding comparison.

All snapshots, step/enable controls and 43 polls succeed; all twelve error files
are empty and collector status 143 follows requested shutdown. Trial 2 lasts
120.000094182 s. Paired camera writer clocks map it approximately to
22:46:58.830011-22:48:58.830105 UTC on November 1 (15:46-15:48 Phoenix).
Use event/data times instead of delayed log-formatting timestamps for phase
comparisons.

### Actual six-command-per-millisecond behavior

| Population | Count |
| --- | ---: |
| Sequential genuine main observations | 240,000 |
| Step-1 / step-2 auxiliary opportunities | 240,000 / 240,000 |
| Step-1 staged / posted / acknowledged | 239,999 / 239,999 / 239,999 |
| Step-2 staged / posted / acknowledged | 239,999 / 239,999 / 239,999 |
| Total posted / acknowledged auxiliaries | 479,998 / 479,998 |
| Skipped auxiliaries | 2, both from the initial unmatched main |
| Runtime preparation / publication deadline skips | 0 / 0 |
| Missing reader clearance, changed/writing primary, resource/wait faults | 0 |
| Unacknowledged posts, reader invalid/drops, SDK gaps/failures, event drops | 0 |

There are 720,002 schema-3 event rows: one main and two auxiliary outcomes for
each of 240,000 main cycles, plus start/finish. After the first unmatched main,
all 239,999 complete cycles deliver both intermediate commands. Main input
counters are sequential, without gaps.

Every passive copy finishes before its immutable target. Step-1 staging lead
is 80.194-138.057 us, median 116.810 us; step 2 is 94.959-122.383 us, median
113.294 us. Targets remain +166.666-166.668 us and +333.333-333.335 us relative
to the frozen main grid. Successful post-return residual medians are 2.439 us
and 2.473 us. Their maxima, 11.648 and 5.702 us, are after-call observations;
they do not show that the earlier 5 us publication-admission guard failed.

All 719,998 combined event identities are contiguous, counters 370222119 through
370942116, with no missing or conflicting publication stamps. The measured
counter-span cadence is **5,999.978721 Hz**. Full-span combined intervals have
mean 166.667258 us and RMS 6.420762 us. The 494.323 us maximum is the initial
main-only interval from the two startup skips. Complete steady-active windows
instead span 139.433-201.761 us, with no later 500 us skipped-slot interval.
Excluding that initial main-only interval, the event-derived steady cadence is
5,999.995104 Hz.
Ordinary combiner observer gaps/duplicates also do not increase; its repeated
primary count increases by exactly 479,998, matching the auxiliary publications.

Main-to-step-1, step-1-to-step-2 and step-2-to-next-main mean combined intervals
are 170.958381, 166.431078 and 162.610932 us. These distinguish actual main
arrival/combination phase from the evenly spaced frozen auxiliary grid. Main
combined periods have mean 500.000359 us, RMS 6.251817 us and range
467.743-540.380 us.

### Controller capacity and main prefixes

Controller INDI endpoints add 761,207 admissions, completions and retirements.
Timed/submitted events add 761,206, with one timing event queued at the after
endpoint. There are no new overloads, coalescing, rejections, SDK failures or
timeouts, preparation/post failures, supersessions/cancellations, unqualified
reader events or timing-counter losses. Historical overload/coalescing counts
of four, worker rejected count eight and reader-unqualified count two are
already present before this capture and do not increase. SDK/post workers
remain FIFO55/FIFO30 on one allowed CPU each, with three pipeline credits.

There are 148 consecutive valid controller summaries, sequence 90334-90481.
For 118 complete steady-active two-second windows:

| Controller duration | Median rolling mean, us | Maximum, us |
| --- | ---: | ---: |
| Receipt to native SDK entry | 16.340833 | 21.841 |
| Preparation | 13.354538 | 18.575 |
| Native SDK call | 103.761253 | 109.367 |
| SDK service through worker readiness | 106.123684 | 111.841 |
| Reader release | 22.103405 | 29.416 |
| Publication | 7.765697 | 21.611 |
| Receipt to callback finish | 139.316857 | 167.065 |

The whole-callback maximum is slightly longer than the nominal 166.667 us slot,
yet the separated SDK/post stages and three credits sustain the requested
throughput without a capacity failure. SDK service through readiness stays
below both the provisional 120 us service bound and the nominal slot period.
The shortest observed controller receipt/send intervals are 139.423/138.141 us,
also above the measured SDK-service maximum. These independent extrema must
not be added into a hypothetical single command path.

All 239,999 classified mains match exact SDK and reader evidence, with one
invalid timing-evidence observation and no unmatched classified joins, evictions
or source resets. Primary-observer receipt to SDK is 30.731120 us mean /
41.638 us maximum; controller receipt to SDK is 15.569901 / 19.497 us, and native
main SDK service is 103.746610 / 107.172 us. The corresponding 4 kHz prefixes
were 30.817490 / 42.170 us and 15.773277 / 23.585 us. There is no measured
prefix growth here, but these measurements do not isolate added main SDK queue
wait. Qualified delivery measurements exclude 38 additional wide brackets
between the INDI endpoints; the broader telemetry endpoints exclude 50.

### Camera comparison and conclusion

Camera telemetry contains 142 consecutive valid stage summaries and 297,911
additional published frames across its broader endpoints. There are no new
failures, rejections, gaps, invalid samples, wide brackets or dropped samples.
For baseline / steady-active / after windows, median acquisition RMS is
0.2767 / 0.2992 / 0.2864 us and publication RMS is
0.8572 / 0.8331 / 0.8508 us. The worst active rolling publication RMS is
1.1098 us. Acquisition-to-write mean medians are 46.1652 / 46.2405 / 46.2859 us.
Compared with the 4 kHz active acquisition/publication RMS of 0.3115/0.7926 us,
these remain comparable. Temperature queries and serial transactions each
increase by 14 while FPS/gain query counts remain fixed.

The finite trial demonstrates sustained coordinated **6 kHz total** delivery
with an ideal constant-zero publisher, a genuine 2 kHz main writer and the
unchanged 5 us publication guard. All post-start auxiliary slots are delivered;
the backend shows no new capacity/loss failures and measured main prefixes and
camera cadence show no material regression. Rolling-window comparisons overlap
and apply to this zero-command, zero-modal-gain workload. A real external
publisher's command-ready/deadline protocol and nonzero command behavior remain
separate production integration work; the finite trial does not implement that
interface or establish a hard zero-added-main-wait guarantee. No code or
instrument setting is changed by this analysis.

## Finite ideal-zero extension to 8 kHz

After the successful 6 kHz trial above, the user authorized extending the app and
capture helper to a four-step empirical trial. The genuine main remains at
2 kHz. Each immutable 500 us grid cycle has its main at step 0 and three
auxiliary targets at +125, +250 and +375 us, for **8 kHz total**, not an added
8 kHz auxiliary timer.

Only the explicit finite ideal-zero trial gains this option. It retains
`lateUs=5`, exact reader/identity and ownership checks, early staging by each
target, three controller credits, existing CPU placement and the 120 s active
upper bound. No ordinary live/shadow maximum-cadence default, controller bound
or external-provider protocol is changed.

No `scheduler.maxCadenceHz=8000` configuration edit is required. Source review
confirms that `configureZeroTest()` selects `clockOnly=true` without setting a
cadence-rating limit from `m_maxCadence`; the zero trial qualifies the genuine
main near 2 kHz and selects its frozen subdivision by the explicit step count.
The ordinary shadow's `configureSimulation()` and old live `processPrimary()`
retain their separate cadence-rating behavior. A nominal 6000 readback in those
ordinary settings therefore does not select or cap the four-step finite trial.

Rebuild/install dmSyncopation and use the updated capture script; neither the
controller nor logdump needs rebuilding for this extension. Keep the working
instance configuration and instrument settings unchanged. The operator starts
with ten active seconds:

```bash
bash apps/dmSyncopation/scripts/capture-zero-aux-test.sh 8000 10
```

After a clean short result, extend to the full active interval:

```bash
bash apps/dmSyncopation/scripts/capture-zero-aux-test.sh 8000 120
```

The helper maps 8000 to four total steps while Off and otherwise retains its
bounded setup/enable/capture/cleanup procedure. This is an operator-run test;
agents do not execute instrument commands. The app's arm/warmup and active
timeouts remain independent of the helper.

The 6 kHz run measured SDK service through worker readiness up to
111.841 us. Against 125 us nominal 8 kHz spacing, that leaves **13.159 us of
nominal headroom**. This is a comparison with an observed maximum, not a hard
service bound or proof of zero added main wait. Actual entry spacing also
depends on delivery, preparation, genuine-main excursions and pipeline resource
ownership. Three overlapping credits still matter because whole-job retirement
latency can exceed one nominal slot.

Compare the short trial with the recorded 6 kHz main-prefix distributions and
exact-match coverage: primary-observer receipt to SDK was 30.731120 us mean /
41.638 us maximum, and controller receipt to SDK was 15.569901 / 19.497 us.
These prefixes are not isolated SDK queue wait. Retain matched/unmatched and
invalid/drop diagnostics, actual per-slot staged/posted/acknowledged/skip counts,
controller admissions/completions/retirements, overload/rejection/coalescing,
SDK failures/timeouts, queue losses and post-service tails. Preserve the first
result before changing the setup or moving to the longer trial.

The preceding 4 kHz and 6 kHz analyses remain measurements of their original
rates and settings. The 8 kHz extension makes no retroactive qualification claim
for them and does not qualify nonzero provider behavior or production operation.

Workstation validation passes: 101 app cases / 2,175,886 assertions, including
four-step publications at +125/+250/+375 us, three auxiliary posts and no extra
main post, startup acceptance of four steps/rejection of five, disabled INDI
selection and rejection of enabled settings changes. The startup fixture now
writes each INI key once: concatenated duplicate defaults had produced `24`
instead of the intended override `4`. Invalid-duration/channel/lateness cases
also use unique keys so they exercise the intended validation values. Ordinary
maximum cadence remains 6000 in the four-step test.

All 20 offline capture-helper tests pass, including `8000 10` selecting four
steps with main rate 2000, and refusal to enable an old app that silently keeps
its previous step count. Unsupported 10000 Hz input is rejected. Production
dmSyncopation build, shell syntax, clang-format and whitespace checks pass.
No instrument commands or an actual 8 kHz trial were run by the agent.

## Short 8 kHz attempt: `zero-aux-8000Hz-20261101T232252Z`

The local directory is
`/home/jrmales/tmp/home/jrmales/tmp/zero-aux-8000Hz-20261101T232252Z`, with its
archive under `/home/jrmales/tmp`. Recorded checkout is `1e06f6b3`; the helper
requests four steps and ten active seconds. The operator confirms unchanged
settings/workload, zero modal gains and stable AO operation.

This is **not a completed ten-second helper-controlled capture**. All eleven
INDI polls retain `active=0`, `posted=0`, `elapsedSec=0`, first idle and then
warming, despite actual publication. The helper eventually reports that the
trial failed to become active within its 30 s warmup guard and requests Off.
Actual activity lasts approximately **25.2229 s**, from
23:23:10.659159 through 23:23:35.882041 UTC on November 1 (16:23 Phoenix).
The configured independent app limit remains 120 s. All command/snapshot
statuses are zero and error files are empty, but the helper explicitly marks
the attempt incomplete; successful INDI requests do not mean their returned
progress values were fresh.

### Actual command and staging outcomes

Final post-cancellation telemetry, rather than the stale INDI snapshots, reports:

| Population | Count |
| --- | ---: |
| Sequential genuine main cycles | 50,446 |
| Auxiliary opportunities | 151,338 |
| Completed passive copies | 151,335 |
| Posted / acknowledged auxiliaries | 151,334 / 151,334 |
| Initial unmatched-main dependent skips | 3 |
| Final step-3 operator cancellation | 1 |
| Late / preparation late / reader missing / resource failures | 0 |
| Unacknowledged auxiliary attempts | 0 |

Step 1 and step 2 each acknowledge 50,445 commands; step 3 acknowledges 50,444.
The final cancelled step was already staged. The four skips account for all
auxiliary opportunities, with no catch-up burst or replay. The combined counter
span represents 201,780 updates and approximately 7,999.876 Hz, including the
startup interval. Retained consecutive-counter publications have mean
125.000052 us and range 91.914-152.357 us. Missing event-key spans must not be
treated as long physical command intervals or converted into a falsely low
delivered cadence.

For retained copies, step-1 staging lead is 42.507-91.640 us (median 75.178 us),
step 2 is 55.053-81.935 us (median 71.334 us), and step 3 is 54.533-81.405 us
(median 71.478 us). These detailed distributions cover retained records only.

### Status starvation and measurement loss

There are 150,093 retained event rows and **51,692 dropped event records**.
Their sum is exactly the 201,785 expected records: 50,446 main observations,
151,334 posts, four skips and one start event. The drops omit 12,923 main
observations and the corresponding auxiliary outcomes. The exact combined-key
union has 150,088 observations over a 201,780-counter span, with four holes
accounting for the same 51,692 records. Full per-event continuity is therefore
unavailable, even though final outcome counters remain internally consistent.
SDK-result observation gaps reach 46 and invalid sampled timing observations
2,331; these are diagnostic coverage failures, not BMC admission/delivery-loss
counters. All 50,445 classified main SDK and reader joins eventually match.

Source review finds a direct progress-reporting weakness: `processImage()` holds
`m_cycleMutex` for the complete active main/auxiliary cycle, while
`updateZeroTest()` refreshes INDI and summary telemetry only after a successful
`try_lock` of that mutex. Event draining occurs outside that snapshot condition.
The last warming summary is at 23:23:10.250689, no active summary appears, and
the first fresh summary follows Off at 23:23:37.813432. Event bursts continue
while status is stale. This is strongly consistent with snapshot starvation;
failed lock attempts and scheduling delays were not instrumented, so their
exact timing is not directly measured.

Ordinary-loop event-drain gaps reach **5.462 s**, exceeding the 20,000-event
queue's roughly 2.5 s capacity at 8 kHz. Overflow is demonstrated by the exact
record accounting. Formatting a full 20,000-record burst takes only about
24-28 ms in emission timestamps, so formatting alone is not established as the
cause of the multi-second drain gaps. Ordinary diagnostic work, logging handoff
and thread scheduling need separate timing measurements.

### Controller pressure and camera comparison

BMC counters show no new overloads, coalescing, rejections, SDK failures or
timeouts, preparation/post failures, supersession/cancellation, unqualified
reader events or timing losses. Historical overload/coalescing/rejection/
unqualified-reader values remain 4/4/8/2. Admissions increase by 237,703 and
completions/retirements by 237,704, consistent with one pre-existing in-flight
job draining. Independently sampled timing properties increase by 237,701;
the broader first/last telemetry endpoints cover 255,747 submissions with no
gaps, invalid/dropped timing events or delivery-quality exclusions. There are
53 consecutive valid controller summaries.

For 23 complete steady-active two-second windows, native SDK call median
rolling mean/maximum is 103.797/105.168 us and service through worker readiness
is 106.230/111.781 us, essentially unchanged from 6 kHz. Receipt-to-SDK is
17.610/31.029 us, preparation 14.188/22.553 us, SDK dispatch 3.423/16.582 us,
reader release 23.164/37.882 us and receipt-to-callback finish
140.634/176.683 us. Front-end/handoff latency grows while active and returns
toward baseline afterward. The shortest incoming controller interval is
91.853 us, below the measured SDK-service minimum of 105.549 us; send intervals
bottom at 105.970 us. This enters a serialization/queue-pressure regime absent
from the 6 kHz run, without an observed capacity loss in this attempt.

Exact main controller-receipt-to-SDK mean/maximum rises from 15.569901/19.497 us
at 6 kHz to 17.0587/25.508 us. Primary-observer-receipt-to-SDK rises from
30.731120/41.638 us to 32.2419/46.347 us: approximately **1.5 us added mean
prefix duration**. Native main SDK service remains approximately 103.8 us.
These prefixes and aggregate dispatch/preparation durations do not isolate
added main-only SDK queue wait.

Camera telemetry contains 49 consecutive valid stage summaries and 101,685
additional published frames over its broader endpoints, with no new gaps,
errors, invalid/dropped observations or wide brackets. Same camera process,
session and settings as 6 kHz. Median acquisition RMS before/active/after is
0.2880/0.2922/0.3017 us; publication RMS is 0.8194/0.8190/0.8461 us. Median
acquisition-to-write mean increases from 46.2770 to 46.8503 us during activity
and returns to 46.3036 us afterward. Cadence stays steady; the short set of
22 overlapping active windows is limited tail evidence.

### Next correction before a longer 8 kHz attempt

The backend sustained approximately 8 kHz over the actual 25 s interval, but
progress control and detailed measurement coverage failed. Correct these before
the planned 120 s run. Publish an ownership-safe, coherent fixed-size latest
progress/timing snapshot at a bounded rate, independent of ordinary-loop
contention for the whole-cycle mutex; include sequence/time/age so the helper
can distinguish stale status from genuine warmup. Independently profile and
bound ordinary diagnostic/event work, or aggregate healthy event logging while
preserving exact outcome totals, all rejection evidence and explicit coverage.
Merely enlarging the event queue postpones overflow. Do not make the real-time
command path wait for reporting or relax timing/readiness/admission guards.
No code or instrument settings are changed by this capture analysis.

## Progress and event reporting correction after the short 8 kHz attempt

The user authorized fixing the monitoring failures above before repeating the
short trial. This changes status transport and ordinary-thread diagnostics;
immutable slot targets, the 5 us publication guard, exact reader clearance,
identity/ownership checks and controller admission remain unchanged.

### Cycle progress without ordinary-thread mutex acquisition

The cycle/control owner publishes a fixed-size `ZeroSnapshot` at completed
callback boundaries, at most every 100 ms during an unchanged phase. Enable,
disable, settings, reacquire and phase transitions publish immediately under
the existing cycle ownership. Startup configures it before workers start.
The snapshot contains counters, timing summary, trial generation, requested
state, steps/rate/duration, queue coverage and a monotonic capture timestamp
and sequence. The publisher uses the callback's existing completed timestamp.

`MagAOX::latestSnapshot<T>` provides three fixed inline slots with exclusive
producer, consumer and atomic-handoff ownership. Unread publications coalesce;
neither owner waits for the other or overwrites a payload still being copied.
All producer calls are serialized by existing cycle ownership; only the
ordinary app loop consumes. There is no allocation, retry loop or new mutex
operation in publication. Live ownership is never reset while either owner
can access it. The helper is reusable independently of this application.

The ordinary loop consumes the latest state without needing the cycle mutex.
A cold or at least 200 ms old snapshot permits a nonblocking mutex attempt for
idle reader/SDK observation and stopped-main expiry servicing. Failure to
acquire it leaves the last coherent state readable. `elapsedSec` advances only
to that state's captured time (or recorded stop), while `snapshotAgeSec` exposes
its age; stale progress is not extrapolated. Schema 4 adds `requested`,
`snapshotSequence`, `snapshotTimeNs`, `snapshotAgeSec`, `snapshotValid`,
`eventsQueued`, `eventStride` and `eventsSampledOut`. Snapshot validity means
coherent publication with a valid local timestamp, not a freshness guarantee.

### Bounded event service and explicit sampling

`zeroTest.eventStride` defaults to 4, accepts 1 through 1024 at startup, and
selects every Nth healthy main cycle for detailed main/auxiliary event rows.
All lifecycle, skipped, unacknowledged and error outcomes are retained.
`eventStride=1` restores all healthy event rows. Command execution, exact
outcome counters and main timing joins remain full rate. The intentional row
omissions are counted separately in per-trial `eventsSampledOut`; queue
capacity loss remains cumulative `eventsDropped`.

At steady 8 kHz, stride 4 reduces healthy text events from about 8,000 to 2,000
per second, giving the unchanged 20,000-row queue approximately ten seconds of
healthy-event headroom instead of 2.5 seconds. This covers the previously
observed 5.462 s gap in capacity terms, but is not proof against an arbitrary
stall or an exceptional-event flood. Detailed event data is explicitly sampled;
it must not be presented as a full per-slot trace. A future full-trace
qualification can select stride 1 once drain delays are understood.

The ordinary loop drains at most 4,096 rows per chunk at entry, between coarse
diagnostic groups and at exit. Each chunk freezes its available population so
it cannot chase the producer indefinitely. It formats and logs without the
cycle mutex. `logic_timing.eventsDrained` counts popped rows, not guaranteed
log persistence; a negative logger enqueue result increments the separate
`eventsLogFailed` counter. Existing text-log wire compatibility is preserved.

### Locate remaining ordinary-loop delays

The `logic_timing` INDI property and `[synco_logic]` text summary report ordinary
loop sequence, start-to-start gap and maximum, queue depth/high-water,
drained/current-loop-drained rows, queue loss and logger enqueue failures.
Coarse `primary`, `combiner`, `camera`, `pipeline`, `readers`, `legacy`, `status`,
`events` and `telemetry` durations include both computation and associated
INDI/logging I/O. Multiple visits to a group add into its current loop total;
maxima retain the largest loop total since startup. These measurements are on
the ordinary thread and do not add per-step instrumentation to the command
path. `reportPublicationUs` and `reportPublicationMaxUs` report the previous
completed profile publication's cost, including formatting, INDI and telemetry
enqueue, so its own I/O does not become an unmeasured group.

The capture helper requires schema at least 4, a coherent baseline no older
than two seconds, and a positive sequence. Polls reject invalid/missing/older
snapshots, sequence regression (including the first poll versus baseline),
five local seconds without sequence advancement, or new event queue loss.
The local sequence watchdog also catches cached INDI ages that themselves stop
updating. Final readback must preserve both queue-drop and logger-failure
baselines; missing quality fields invalidate the capture. Cancellation still
uses only the helper-owned enable request. No camera, main-loop, dmcomb, CPU or
controller setting is changed by the helper.

### Qualification and next instrument action

Rebuild/install dmSyncopation and use the matching capture helper. Neither
logdump nor the DM controller needs rebuilding for these existing text records.
Keep all working instrument settings unchanged and repeat:

```bash
bash apps/dmSyncopation/scripts/capture-zero-aux-test.sh 8000 10
```

Verify fresh active progress, an approximately ten-second active interval,
exact outcome totals, documented sampling and zero new queue/logger losses.
Use the ordinary-loop profile to identify any remaining long gaps. Only after
a clean short capture should the 120 s 8 kHz attempt proceed. Hardware
qualification is pending; workstation tests cannot establish instrument drain
throughput or backend timing.

Workstation verification for this correction:

- `latestSnapshot_test`: 3 cases / 139 assertions pass, including a slow
  concurrent consumer overtaken by 100,000 multiword publications.
- `dmSyncopation_test`: 105 cases / 2,236,431 assertions pass. New regressions
  cover status delivery with the cycle mutex held by another thread, frozen
  elapsed versus snapshot age, publication coalescing/cadence/phase transitions,
  successful-cycle sampling with retained exceptions and exact command counts,
  bounded event drains and profiling-property mappings. Existing timing,
  ownership, wall-clock cancellation and no-write cases continue to pass.
- `capture_zero_aux_test.py`: 26 offline cases pass, including cached/frozen
  status, stale/missing snapshots, first-poll restart detection, queue loss and
  final logger enqueue failure. Its services are private mocks, not instrument
  connections.
- Production dmSyncopation compiles and links successfully. Touched C++ files
  were formatted with clang-format; shell syntax and `git diff --check` pass.

No instrument command was executed by the agent. The unchanged-slot timing and
new reporting throughput still require the operator's short rerun above.

## Reporting-corrected 8 kHz captures: `20261102T003833Z` and `20261102T003954Z`

The short and full captures use checkout `14881210a7b05dd96cba2571d76d9dd2273c0a96`,
schema 4, four steps and `eventStride=4`. Local copies were found under
`/home/jrmales/tmp/home/jrmales/tmp/zero-aux-8000Hz-<UTC>`, with the corresponding
`.tgz` archives in `/home/jrmales/tmp`. No instrument commands were run by the
agent. This analysis changes only this engineering record.

**Changed baseline:** the operator reports that a compilation-settings fix made
`acquWFS-1` about 70 us faster. The loop otherwise remained as before and stable;
modal gains remain zero under the previously established test setup. This is
an upstream latency/phase change, so comparisons with older 6/8 kHz captures
cannot attribute differences solely to syncopation. The comparisons below use
each new capture's own disabled baseline. The saved camera/main settings remain
2 kHz, synchronization On, scalar loop gain 1 and RTC latency On.

### Short trial: coverage restored, requested duration still overshoots

`zero-aux-8000Hz-20261102T003833Z` requested ten active seconds. It actually ran
**16.246623416 s** in the final duration statistic preserved in the next
capture's baseline. The start-to-last-cancel-event span is 16.246618567 s;
the 4.849 us difference is the subsequent stop bookkeeping. Approximate UTC
activity is 00:38:51.33838 through 00:39:07.58498 on November 2.

- 32,494 genuine-main observations and 97,482 auxiliary opportunities.
- 97,435 auxiliary publications, all acknowledged; zero unacknowledged attempts.
- 47 skips: three initial unclassified combinations, 41 deadline misses and
  three final operator-cancelled slots.
- All 41 deadline misses were step 2, 5.023-6.907 us late. Their commands had
  already been staged 66.421-80.507 us before target. They began 13.4508 s into
  the actual interval, after the requested ten seconds, and were publication
  wake/deadline failures rather than publisher preparation failures.
- Excluding startup/shutdown cycles, auxiliary delivery is 97,435 / 97,476 =
  99.95794%. No preparation, readiness, identity, resource or wait-error skips
  occurred.

Coverage closes exactly: **32,523 retained events + 97,454 intentionally
sampled-out = 129,977 expected rows**, comprising 32,494 main observations,
97,482 auxiliary opportunities and one start event. Every exceptional event is
retained. Queue drops and logger enqueue failures stay zero, and the final event
queue is empty. All 32,493 classified mains have matching SDK and reader
records; separate result-observer coverage counters are 1,424 invalid samples
and 35 skipped SDK counters, with no failed SDK outcomes.

All eight helper polls and control commands returned success. Unlike the
previous run, progress advances: active elapsed readbacks are 1.2034, 4.4129,
5.4159, 9.7274 and 13.4364 s, with reported snapshot ages 7.95-82.32 ms. The
post-Off INDI snapshot still reports active progress from 15.8429 s even though
the enable switch reads Off; later telemetry correctly reports the stop.

The remaining overshoot combines coarse/cached ordinary status with the
helper's polling behavior. `INDI/INDI/getINDI.c` defaults to a two-second wait
for a wildcard whole-property query; the script also sleeps one second and
starts its local duration deadline when it first observes active status.
`setINDI` exit zero proves request transmission, not callback completion.
Off's callback clears the atomic enable flag before acquiring the cycle mutex,
so continued posts must not be attributed to cycle-mutex blocking after that
callback starts. The app's independent duration remains configured at 120 s,
not the helper's shorter ten-second request. App-enforced requested duration
would avoid relying on polling for the precise short stop.

The short capture has 43 consecutive valid controller summaries. No new
controller overload, coalescing, rejection, SDK failure/timeout, preparation/
post failure or timing loss occurs. Delivery-clock qualification excludes 136
new wide brackets. Its 32,493 exact main joins show reader-begin-to-SDK mean/max
17.071066/24.295 us; a 39,998-sample nonoverlapping disabled baseline has true
receipt-to-SDK mean 16.347216 us. The arithmetic difference of 0.724 us has
mismatched origins and does not measure added main latency. Camera records remain
qualified and continuous, with no new frame/counter gaps, errors or timing
loss. The short interval is supporting evidence, not the two-minute result.

### Full 120-second trial: sustained delivery with small deadline skips

`zero-aux-8000Hz-20261102T003954Z` finishes under the app's independent duration
limit at **120.000107659 s**, measured directly from lifecycle monotonic times
2447850961447200 through 2447970961554859. Approximate UTC activity is
00:40:12.37146 through 00:42:12.37157, with 55 us spread in the local-to-UTC
mapping. All 44 helper polls/control requests succeed; final readback is Off,
finished, with elapsed 120.000107659 and no queued events.

| Actual population | Count |
| --- | ---: |
| Genuine-main observations | 240,000 |
| Auxiliary opportunities | 720,000 |
| Auxiliary publications, all acknowledged | 719,745 |
| Initial unclassified skips | 3 |
| Deadline-late skips | 252 |
| All other skip/fault classes | 0 |

Excluding the three startup slots, auxiliary delivery is
719,745 / 719,997 = **99.96500%**. Counting main observations plus issued
auxiliary publications gives approximately **7,997.9 commands/s** over the
active interval, with the genuine main remaining at 2 kHz. This is near the
requested total 8 kHz with explicitly skipped occurrences, not an assertion
that every nominal 125 us slot was delivered.

Actual posts by step are 239,750 / 239,996 / 239,999. Late skips are 249 at
step 1 and three at step 2, unlike the short trial's exclusively step-2 pattern.
Each affects a distinct main cycle; 244 occur between 70 and 80 seconds of
activity. Of the 252, 237 fail the `deadline` check 5.003-7.336 us late and
15 fail `publication_guard` 5.004-5.846 us late. Every late command had already
been staged 57.267-83.673 us before its ready deadline. All 719,997 non-startup
slots were staged; preparation/readiness did not cause the skips. Keep the
existing 5 us guard when interpreting this result; no guard was relaxed.
The changed step distribution and clustering do not establish a permanent
step-specific backend bottleneck or identify the cause of wake lateness.

Coverage again closes exactly: **240,122 retained + 719,880 sampled-out =
960,002 expected events**, comprising 240,000 main observations, 720,000
auxiliary opportunities and start/finish. Zero-event queue drops and logger
enqueue failures remain zero, and the queue drains completely. Lifetime event
drain count increases from 32,523 to 272,645, exactly 240,122. Reported snapshot
ages are 0.339-99.657 ms; two repeated sequences subsequently advance without
triggering the local watchdog.

There are 239,999 matched main combinations and one unmatched startup main.
One additional main has unclassified impact, leaving **239,998 exact SDK and
reader joins**, all matched. Result-observer diagnostics separately report
11,371 invalid observations and 256 skipped SDK counters; these are coverage
exclusions, not evidence of that many lost DM commands. No timing-source
resets, evictions, failed SDK results or real-time reader-readiness queue loss
occur.

### Full-run controller timing and camera behavior

There are 152 consecutive valid controller summaries with no sequence gaps.
Their broader endpoints cover 1,023,207 additional submissions; INDI admissions
increase by 1,005,165 and retirements/completions by 1,005,164, with one job in
flight afterward. There are no new overloads, coalescing, rejections, SDK
failures/timeouts, preparation/post failures, supersession/cancellation,
unqualified-reader/vector rejection or timing drops/gaps/invalids. Delivery
qualification excludes 301 new wide clock brackets; other delivery-error
counters do not increase.

The table uses median rolling means and the greatest observed sample maximum,
in microseconds. It covers 20 disabled baseline, 117 complete active two-second
and 11 post-trial windows; adjacent windows overlap.

| Measurement | Before mean / maximum | Active mean / maximum | After mean / maximum |
| --- | ---: | ---: | ---: |
| Native SDK | 103.860 / 104.527 | 103.811 / 109.527 | 103.897 / 105.259 |
| SDK service through readiness | 106.167 / 107.153 | 106.233 / 116.530 | 106.209 / 107.643 |
| Receipt to SDK, mixed active population | 16.288 / 19.477 | 17.590 / 42.941 | 16.370 / 19.547 |
| Preparation | 13.299 / 16.652 | 14.184 / 23.625 | 13.302 / 16.351 |
| SDK dispatch | 2.988 / 3.767 | 3.413 / 28.895 | 3.063 / 3.647 |
| Reader release | 21.894 / 26.029 | 23.188 / 36.639 | 21.931 / 25.929 |

The exact main reader-begin-to-SDK population is **17.019824 us mean /
24.907 us maximum**; primary-observer receipt to SDK is 31.975720 / 45.456 us.
A nonoverlapping 39,999-sample disabled baseline has true receipt-to-SDK mean
16.289952 us. Their 0.730 us arithmetic difference mixes start endpoints and
cannot establish added main latency or SDK wait. Exact native main SDK service
is 103.914297 / 109.527 us.
The 116.530 us mixed-population readiness maximum leaves only 8.470 us nominal
headroom in a 125 us slot; a finite observed maximum is not a hard bound.

Camera telemetry has 144 consecutive valid stage summaries and 301,960
additional published frames, without new frame/counter gaps, errors, rejected/
invalid/wide observations or dropped records. Fourteen serial-control
transactions occur without control failures/loss. Across 112 complete active
windows, median acquisition RMS before/active/after is
0.285772/0.489764/0.293593 us; publication RMS is
0.830161/0.785300/0.828159 us. Median acquisition-to-write mean rises from
46.715216 to 47.413302 us and returns to 46.837170 us afterward.

Preserve a separate pretrial event: camera windows emitted at 00:40:03.063 and
00:40:04.066, before activation, contain publication intervals
390.498-610.002 us and acquisition-to-write up to 157.721 us. PDV acquisition
intervals remain 498-502 us, while wait-return intervals already span
390.607-610.132 us. This excursion was present by SDK return and preceded
syncopation activation; it does not identify hardware versus driver versus
scheduling. During active operation, publication intervals span
479.205-525.763 us. Unbroken frame numbers alone still cannot exclude the
previously discussed frame-behind condition.

### Remaining ordinary-reader diagnostic cost

The zero-event queue fix works in both captures, but ordinary reader diagnostics
still have multi-second service times. The full capture has 82 consecutive
profile rows, sequence 89 through 170. New observed maxima are:

- Reader diagnostics: **3.837887970 s**, versus 3.271835604 s in the short run.
- Start-to-start ordinary-loop gap: **4.860362745 s**, including the normal
  one-second sleep, versus 4.291238914 s in the short run.
- Event formatting/enqueue per loop: 12.166681 ms; camera/pipeline/primary/
  combiner groups remain below 3.843/3.317/2.731/2.189 ms.
- Status work: 1.341660 ms; previous report's own publication: 121.529 us.
- Event queue high-water: **7,676 / 20,000**, versus 6,543 in the short run.

The largest event-emission gap is 3.837892919 s, within about 5 us of the
reader-group maximum. Thus the profile locates the gap in that group; event
formatting and status publication do not account for its seconds of cost.

Both legacy reader collectors have **primary=combined=matched=0** throughout
these zero trials, while retaining completion observations. Their primary and
combined feeds are confined to simulation branches in `dmSyncopation.hpp`.
`dev::dmReaderTiming::retain()` scans its entire pending map with
`std::min_element` on each capacity eviction. `update()` drains the captured
completion batch before joining and expiring records. At 8 kHz, two seconds of
retention (~16,000 records) plus a one-second incoming batch (~8,000) crosses
the 20,000 cap, potentially causing thousands of full-map scans in each of two
collectors. This is a strong source-backed explanation for the measured reader
cost, although individual collector calculation and INDI I/O were not separately
timed. It is not evidence of an isolated-core scheduling problem.

Before/after full-capture changes in these legacy diagnostic populations are:

| Counter | Completion collector | Release collector |
| --- | ---: | ---: |
| Consumed completion records | +794,607 | +794,721 |
| Dropped completion records | +211,608 | +211,494 |
| Capacity evictions | +332,967 | +326,499 |
| Expirations | +461,638 | +468,222 |
| Missing combined identities | +794,605 | +794,721 |

Those are **diagnostic coverage losses**, separate from clean zero-event
transport, exact main-impact joins and the dedicated real-time readiness queue.
The short capture already shows +23,085/+22,086 legacy completion-queue drops.
Do not present the legacy reader diagnostics as qualified timing evidence for
these trials.

The next reporting cleanup should stop retaining unjoinable legacy completions
in zero-test mode while preserving the independent release observer and actual
readiness evidence. The general collector should also expire before capacity
pressure and avoid a full-map scan for each overflow. Separately, move the
requested short duration into an app-enforced bound, or use exact-element
bounded polling when only prompt status is required. No code, guard, controller
setting or hardware configuration is changed by this analysis.

The full result supports sustained finite ideal-zero **8 kHz total** operation
with about 0.035% deadline skips and no observed controller admission loss on
this workload. It does not qualify nonzero external publishers, prove zero
added main SDK wait, or make the lossy legacy diagnostic collectors valid.

## Implemented cleanup: zero-test reader work and app-enforced short durations

Following the two reporting-corrected captures, the user authorized the next
cleanup. Slot grids, 5 us publication admission, early staging, exact readiness
and controller admission are unchanged.

### Separate readiness observation from unused legacy joins

Zero-test startup now configures both legacy `dev::dmReaderTiming` collectors
disabled. Their INDI quality properties remain visible with `enabled=0`; no
legacy reader-protection population is qualified in this mode. Zero-mode
callbacks also avoid enqueuing legacy completion observations, and ordinary
logic skips their update/statistics/INDI work. The unused full-completion
observer thread is not started for this mode.

The independent reader-release monitor has an explicit lifecycle predicate
separate from its legacy diagnostic collector. It still starts, captures the
coherent release record and receipt, forwards it to the real-time readiness
queue, observes attachment epochs and invalid/discarded records, and shuts down
normally. Disabling the unused join therefore does not disable
`dmtweeter_read_done`, its queue/drop guards or exact `ZeroTiming` main-impact
joins. Ordinary and shadow collector behavior remains selected as before.

### Remove repeated full-map eviction work in the shared collector

Consumer-owned pending evidence uses ordered `std::map` lookup in place of
unordered maps with a full `min_element` scan for every eviction. Existing
progression rules admit one current inode per source/generation, increasing
counters and nondecreasing local observation times; source resets clear pending
state. The first identity is consequently the oldest pending observation.
Eviction removes that entry directly, and expiration walks only the expired
prefix. Replacing an already retained key does not evict an unrelated entry at
capacity. Same-attempt result enrichment still preserves its original timing.

The queues still receive their bounded captured-population drains, followed by
joining and then expiration. Expiring first could discard old pending evidence
whose matching peer is already in another captured queue. Ordered eviction
removes the excessive scan cost without changing that lifetime-join behavior.
All joining/container work remains on the ordinary consumer; capture methods
retain their fixed-record, nonblocking SPSC handoff.

### Schema 5 selected duration and trial identity

`zeroTest.durationSec` remains the startup ceiling, positive and no greater than
120 seconds. It initializes the selected duration. While zero-test enable is
Off, `settings.durationSec` can select any positive finite duration up to that
ceiling. Other modes, active changes, nonfinite/nonpositive values and values
above the ceiling are rejected. Multi-element settings changes validate and
commit atomically; failure preserves the previous selected duration and steps.
A duration-only change preserves the warmed cadence instead of reconfiguring it.

Schema 5 exposes the existing trial generation as `zero_test.trial`, the
selected app-enforced `durationSec`, and immutable startup `durationLimitSec`.
The selected bound is used by the existing command-path wait, cycle expiry and
stopped-main ordinary fallback; it does not depend on a helper polling at the
right instant. Snapshots publish selected duration, ceiling and trial identity
coherently. Settings duration readback is refreshed outside cycle ownership.
The 30-second independent arm/warmup limit is unchanged.

The capture helper now requires schema at least 5 and checks the startup ceiling
against the requested interval. While Off, it sets steps and selected duration,
then verifies disabled status, effective duration and trial/sequence readback
before requesting On. Polls request exact `zero_test` elements together, so
getINDI can return as soon as that vector arrives rather than waiting for its
whole-property wildcard timeout.

The configured trial generation is the baseline: old-generation finished or
active readbacks cannot complete the new capture. The helper requires exactly
the next generation and rejects subsequent identity changes. Success requires
that trial to report finished, requested/active/warming all false, positive main
observations and at least the requested elapsed interval. A short trial can
finish before any active poll and still qualify from that evidence. Reaching
elapsed time while still active is no longer success. Bounded reporting grace
allows status to arrive after physical stopping; failure to confirm completion
or a finish beyond that grace marks the capture incomplete and requests Off.
Existing ownership, interruption, freshness and event-loss checks remain active.

### Operator follow-up

Rebuild/install dmSyncopation and use the matching updated helper. No controller
or logdump rebuild, working configuration replacement or instrument retuning is
required. A startup ceiling of 120 permits both requested intervals. Repeat the
short test first:

```bash
bash apps/dmSyncopation/scripts/capture-zero-aux-test.sh 8000 10
```

Confirm app-enforced elapsed near ten seconds, finished/Off at the matching new
trial identity, disabled legacy diagnostic collectors with active readiness,
no new event/readiness/controller loss and a much smaller reader diagnostic
service cost. Then repeat the 120-second capture under the same faster-acquWFS
baseline. Workstation verification cannot establish the instrument's new
ordinary-loop cost or qualify a hard worst-case backend bound.

Workstation verification for this implementation:

- `dmReaderTiming_test`: **28 cases / 603 assertions pass**. New pressure tests
  process 60,000 unjoined completions with a 20,000-entry cap, exercise full-cap
  replacement without spurious eviction, backward-observation rejection,
  source-reset cleanup and delayed queued-peer joins before expiration. Existing
  enrichment coverage also runs at capacity one.
- `dmSyncopation_test`: **109 cases / 2,542,975 assertions pass**. New coverage
  includes disabled legacy populations with a 1,024-callback readiness burst,
  readiness invalidation on discard, normal collector selection, selected
  duration/ceiling/readback validation and rollback, independent 1/10-second
  command-path expiry and new trial identity. Existing slot timing, no-write,
  ownership and cancellation cases still pass.
- `capture_zero_aux_test.py`: **35 private-mock cases pass**. They cover schema
  rejection, ignored duration writes, a longer selection after a short one,
  exact-element polling, stale or wrong trial identity, first-poll completion,
  active/premature/late finish failures, control ownership, interruption,
  freshness and event-loss checks. No instrument connection is used.
- Production dmSyncopation compiles and links. Touched C++ files pass
  clang-format verification, the helper passes Bash syntax checking, and
  `git diff --check` passes.

Only the operator can qualify the new stop timing and diagnostic service cost
on the instrument. No hardware command, deployment, commit or push was performed
as part of this implementation step.

## Schema-5 short qualification: `zero-aux-8000Hz-20261102T025610Z`

The operator provided the extracted directory, located locally under
`/home/jrmales/tmp/home/jrmales/tmp/zero-aux-8000Hz-20261102T025610Z`.
The checkout is `97f04b6a3eddbb9a29e5d4c64b9bb29bb7577e9c`, reporting schema 5,
four steps and `eventStride=4`. The user confirms stable loop operation, modal
gains zero and no settings/workload changes apart from synco/helper updates.
Saved before/after settings are 2 kHz, synchronization On, scalar loop gain 1,
RTC latency On, camera EM gain 1, target +20 C and shutter open. The latter
thermal/shutter settings differ from the older -45 C/shut capture; timing
comparisons below use this capture's own disabled baseline. No instrument
command was executed by the agent and no application code changes accompany
this analysis.

### Selected ten-second bound and actual delivery

The helper successfully selects `settings.durationSec=10` while Off, with a
startup `durationLimitSec=120`. Lifecycle monotonic times
2456028634694279 through 2456038634700321 give **10.000006042 s**, approximately
02:56:30.044704 through 02:56:40.044710 UTC on November 2. The local-to-UTC
mapping spread is 18 us. The selected app bound now works without a delayed
helper stop extending the actual active interval.

| Actual population | Count |
| --- | ---: |
| Genuine-main observations | 20,000 |
| Auxiliary opportunities | 60,000 |
| Posted, all acknowledged | 59,994 |
| Startup-unclassified skips | 3 |
| Deadline-late skips | 3 |
| All other skip/fault classes | 0 |

All three late skips are step 3: two `deadline` observations at 5.009 and
5.083 us, and one `publication_guard` observation at 5.018 us. They occur
0.054881, 1.315381 and 5.396884 s into activity. Their commands had already
been staged 68.510, 80.343 and 68.146 us early. This is publication-admission
lateness at the unchanged 5 us guard, not a publisher preparation miss.
Actual posts by step are 19,999 / 19,999 / 19,996. Excluding the three startup
slots, auxiliary delivery is 59,994 / 59,997 = **99.9950%**.

Coverage closes exactly: **20,009 retained events + 59,993 sampled-out = 80,002
expected**, comprising 20,000 main observations, 60,000 auxiliary opportunities
and start/finish. Queue drops and logger enqueue failures stay zero; the queue
is empty at completion. All 19,999 classified mains match both SDK and reader
evidence, with no unclassified impact. Separate result-observer sampling
exclusions are 767 invalid observations and 14 skipped SDK counters, without
failed SDK outcomes, resets or evictions.

All 19 exact-element helper polls and control requests succeed. An initial old
trial-0 response is ignored, followed by trial-1 warming, active progress and
confirmed finish. Sequences advance strictly from 270 to 453; reported snapshot
ages are 1.282-70.118 ms. Final readback is Off, finished, trial 1, elapsed
10.000006042 s, selected duration 10 and ceiling 120.

### Reader-diagnostic stalls removed, readiness retained

Forty consecutive ordinary profile records cover sequence 15 through 54.
Observed reader-group median/maximum is **4.0475 / 5.630 us**. Its 6.002 us
lifetime maximum was already present before the capture. This replaces the
previous long run's 3.837888 s reader-group maximum. The removal of unused
legacy work is validated here; this is not a hardware benchmark of ordered-map
performance while those collectors are enabled.

The ordinary start-to-start gap is 1.005995 s median and **1.009874656 s
maximum**, including the configured one-second wait. The largest event-emission
gap is 1.000889282 s; there is no extra multi-second computation gap. Event
formatting/enqueue per-loop maximum is 2.912509 ms, status 714.629 us and profile
publication 100.299 us. Event queue high-water is **2,004 / 20,000**; drained
count increases from zero to 20,009, exactly matching the event text rows.

Both legacy reader collectors explicitly report `enabled=0`. Their completion,
queue, pending, drop, eviction and expiration counts remain zero before and
after. The independent reader-release observer remains present at requested
priority 30, and real readiness retains 19,999 main-reader matches with zero
`readerDrops` or `readerInvalid`. These snapshots do not independently verify
actual CPU affinity/FIFO policy, but there is no need to change placement for
this qualification.

### Controller and camera comparison

There are 40 consecutive valid controller summaries, no sequence gaps and
138,237 additional submissions over their broader endpoints. All 138,237 are
delivery-clock qualified, with zero new wide brackets or delivery errors.
No new overload, coalescing, rejection, SDK failure/timeout, preparation/post
failure, supersession/cancellation, unqualified-reader/vector rejection or
timing gap/invalid/drop occurs. INDI admissions/retirements increase by 122,200;
worker completions increase by 122,201 across separately sampled endpoints.

Eight complete active two-second controller windows show native SDK median
rolling mean/max 103.796/105.529 us and service through readiness
106.196/110.900 us. Mixed main-plus-aux receipt-to-SDK is 17.555/22.703 us,
preparation 14.240/18.925 us, dispatch 3.316/8.656 us and reader release
22.893/27.943 us. These maxima are finite observations, not hard bounds.

Exact main reader-begin-to-SDK mean/max is **16.890124/21.371 us**. The
43,997-sample nonoverlapping disabled baseline has true receipt-to-SDK mean
16.320308 us. The 0.570 us arithmetic difference uses different starts and
cannot be described as added main latency. Primary-observer receipt to SDK is
32.148816/41.618 us and native main SDK is 103.787392/105.529 us; these exact
measurements retain their stated origins.

Camera telemetry has 38 consecutive valid stage records and 78,526 additional
published frames, with no new frame/counter gaps, errors, invalid/wide
observations or dropped records. Four serial controls occur over the broader
telemetry endpoints without failures/loss; before/after INDI has three additional
temperature queries with fixed FPS/gain query counts. Eight complete active
windows give median acquisition RMS before/active/after
0.272315/0.458731/0.228472 us, publication RMS
0.880850/0.751582/0.883563 us, and acquisition-to-write mean
47.596842/47.372584/47.606406 us. Active acquisition intervals span 481-514 us,
publication 479.515-513.008 us. No previous roughly 110 us publication excursion
appears in this short capture; short duration limits tail evidence.

Both requested reporting corrections are qualified by this short run: the
selected app duration stops on time, and the unused reader-diagnostic stalls
are absent while readiness and event coverage remain intact. The next action
is a same-settings 120-second capture using the updated helper. Retain the
unchanged publication guard, controller settings and faster-acquWFS baseline;
this short result does not replace long-duration tail qualification.

## Schema-5 full qualification: `zero-aux-8000Hz-20261102T025808Z`

This is the same-checkout (`97f04b6a3eddbb9a29e5d4c64b9bb29bb7577e9c`),
same-settings full follow-up to the short qualification above. The extracted
capture is under
`/home/jrmales/tmp/home/jrmales/tmp/zero-aux-8000Hz-20261102T025808Z`.
Saved camera/main/latency and worker settings match the short run: 2 kHz,
synchronization On, EM gain 1, camera target +20 C/shutter open, scalar loop gain
1 and RTC latency On. Modal gains and stability were confirmed for the short
run; this follows the requested unchanged-condition comparison, with no further
changes reported. The analysis uses the full run's own disabled baseline.
No instrument command or application code change accompanies this record.

### App bound, command populations and coverage

The helper restores selected `durationSec=120` within startup
`durationLimitSec=120` while Off. Trial 2 starts at monotonic
2456145445295833 and finishes at 2456265445395684: **120.000099851 s**.
Approximate UTC activity is 02:58:26.855311-03:00:26.855411 on November 2,
with 34 us local-to-UTC mapping spread. Prior trial-1 summaries and counters
remain in the baseline; trial-2 event populations are counted independently,
not subtracted from the previous short trial's resettable totals.

| Actual population | Count |
| --- | ---: |
| Genuine-main observations | 240,000 |
| Auxiliary opportunities | 720,000 |
| Posted, all acknowledged | 719,927 |
| Startup-unclassified skips | 3 |
| Deadline-late skips | 70 |
| Other skip/fault classes | 0 |

Excluding startup, auxiliary delivery is 719,927 / 719,997 = **99.99028%**;
the steady deadline-skip fraction is **0.009722%**. Main observations plus
auxiliary publications give approximately **7,999.4 commands/s** over the
active interval. All 719,997 non-startup slots were staged. Actual posts by
step are 239,999 / 239,929 / 239,999.

All 70 late slots are step 2, each in a distinct main cycle, between
105.769845 and 112.866850 s into activity. There are 64 `deadline` failures at
5.008-6.751 us late and six `publication_guard` failures at 5.006-5.034 us.
Their commands had already been staged **64.988-80.357 us early**. Preparation,
reader clearance, resource/identity checks and wait errors do not cause skips.
The 5 us admission guard is unchanged; these are controlled publication skips,
not commands submitted late or missing publisher data. The late-step pattern
varies between trials, so this cluster does not establish a permanent
step-specific bottleneck or identify the cause of wake lateness.

Event coverage closes exactly: **240,041 retained + 719,961 sampled-out =
960,002 expected events** (240,000 main + 720,000 auxiliary + start/finish).
All event rows belong to trial 2. Lifetime drained count rises from 20,009 to
260,050, exactly 240,041. Queue drops and logger enqueue failures stay zero.
The immediate after-INDI `zero_test.eventsQueued=381` is a depth captured with
that particular snapshot, not evidence of residual data loss: current
`logic_timing.eventsQueued=0` and subsequent telemetry show queue zero, and the
complete accounting proves the tail was retained.

All **239,999 classified mains** match both SDK and reader records, with no
unclassified impact; one startup main is unmatched. Separate result-sampling
coverage records 8,159 invalid observations and 324 skipped SDK counters, with
zero failed SDK results, source resets or timing evictions. These sampling
exclusions are not lost DM commands.

All 125 helper polls and control requests succeed. The old trial-1 finished
response is ignored before trial 2; sequences advance strictly from 1439 to
2720, and reported snapshot ages are 1.412-100.118 ms. Final state is trial 2,
Off and finished at the selected interval. The capture therefore validates
both the short-to-long duration selection and full-run completion protocol.

### Ordinary diagnostics remain bounded in observed service time

The capture contains 148 consecutive ordinary profile reports, sequences
132-279. Reader-group median/maximum is **3.9625 / 19.477 us**, with no return
of the previous 3.84-second stalls. The ordinary loop gap is 1.009810 s median,
**1.011542363 s maximum**, including its normal one-second wait. Maximum
event-emission gap is 1.001068217 s, consistent with that wait.

Event queue high-water is **2,012 / 20,000**, slightly above the preceding
2,004. Observed per-loop event service peaks at 3.100473 ms; camera, pipeline,
combiner and primary groups remain below 3.540, 2.239, 1.419 and 1.237 ms.
Status peaks at 805.650 us and profile publication at 99.388 us. Inherited
lifetime maxima are distinguished from these current-capture observations.

Both legacy reader collectors remain disabled, with all completion, queue,
pending, drop, eviction and expiration populations zero before and after.
The dedicated observer keeps the same TID 2838414 and requested priority 30;
actual affinity/FIFO policy is not independently measured here. Real readiness
continues to provide 239,999 main-reader matches with zero `readerDrops` or
`readerInvalid`. These data validate the unused-work removal and its separation
from active readiness, rather than measuring enabled-collector map performance.

### Controller capacity and main prefix

There are 148 consecutive valid controller summaries with no sequence gaps.
Their broader endpoints contain **1,015,351 new submissions**, without new
admission failure, timing loss/gap/invalid or failed outcomes. Delivery-clock
qualification excludes 126 wide brackets, with other delivery-error counters
unchanged. No new overload, coalescing, rejection, SDK failure/timeout,
preparation/post failure, supersession/cancellation, unqualified-reader or
vector rejection occurs. INDI admissions/retirements/completions each increase
by 999,316; independently sampled timing admissions increase by 999,315.

Timing below is median rolling mean / greatest observed sample maximum in us,
from 20 disabled baseline, 117 complete active and seven post-trial two-second
windows. Adjacent windows overlap.

| Measurement | Before | Active | After |
| --- | ---: | ---: | ---: |
| Native SDK | 103.858 / 105.149 | 103.793 / 109.828 | 103.864 / 104.647 |
| SDK service through readiness | 106.191 / 107.643 | 106.200 / 112.773 | 106.199 / 107.112 |
| Receipt to SDK, mixed active population | 16.334 / 19.747 | 17.545 / 26.611 | 16.369 / 19.718 |
| Preparation | 13.366 / 16.531 | 14.204 / 22.593 | 13.429 / 16.752 |
| SDK dispatch | 2.966 / 6.251 | 3.343 / 11.401 | 2.936 / 3.747 |
| Reader release | 21.648 / 25.689 | 22.907 / 31.720 | 21.825 / 25.718 |

Exact main reader-begin-to-SDK mean/max is **16.823482 / 23.744 us**. A
nonoverlapping pretrial baseline contains 39,998 true receipt-to-SDK samples
with mean 16.327502 us. The 0.496 us arithmetic difference mixes origins and
is not a measurement of added main latency. Primary-observer receipt to SDK is
31.995927 / 49.043 us; native main SDK service is 103.786113 / 106.902 us.
A common true-receipt origin is required before assessing added main latency. The mixed-population
readiness maximum of 112.773 us leaves 12.227 us nominal room against a 125 us
slot, without proving a hard bound.

### Camera coverage and qualification outcome

Camera telemetry has 140 consecutive valid stage records and 297,088 additional
published frames, with no new frame/counter gaps, rejected/failed/invalid
observations or dropped records. Ten wide brackets occur before active output,
in records emitted at 02:58:08.375 and 02:58:09.378. Four baseline windows have
nonzero ambiguity; all **110 complete active camera windows have ambiguity
zero**. Seventeen serial controls occur across the broader endpoints without
control failure/loss.

Median acquisition RMS before/active/after is 0.278457/0.500492/0.273015 us;
publication RMS is 2.636073/0.797061/0.868297 us. Median acquisition-to-write
mean is 48.819635/47.387026/47.539646 us and conversion mean is
28.032058/26.422811/26.673565 us. The disabled baseline is noisier than the short
run; the improvement during activity is not proof that auxiliary output caused
better camera timing. Active publication intervals span 479.305-523.739 us,
acquisition 480-525 us, with no earlier roughly 110 us excursion in the active
interval.

The full run confirms the reporting fixes and sustained finite ideal-zero
8 kHz total operation on this workload: app-controlled 120-second stopping,
complete sampled-event coverage, correct readiness, small controlled deadline
skips and no observed controller admission loss. Further identical zero runs
are not required to establish that these fixes work. Production external
publisher integration, nonzero command qualification and a strict zero-added-
main-SDK-wait contract remain separate work; this result does not extend the
ordinary live/shadow cadence limit or change scheduling guards.


## Revised downstream target following the full schema-5 capture

The user confirms the goal is dmcomb's trigger to the controller every nominal
125 us, and directs earlier deadlines for the auxiliary actions. The existing
full capture already supports relative output-phase analysis: 59,965 complete
selected cycles have mean main-to-auxiliary combined-publication offsets
129.314 / 254.161 / 379.905 us. These are about 4-5 us later than nominal,
even though aggregate controller receipt intervals average close to 125 us.
They compare same-clock publication durations and are not literal semaphore
post timestamps or full auxiliary path delays. Sampling does not retain the
immediate last-to-next-main pair, which needs explicit future coverage.

The [new endpoint/deadline plan](../2026-11/dm-syncopation-combined-trigger-deadlines.md)
separates combined trigger targets, earlier auxiliary publication/admission,
validation/wake and provider-ready cutoffs. It preserves the long-baseline grid,
immutable advertised cycles, exact readiness and per-step skips. Lead selection
will account for the main combined phase as well as the auxiliary path, avoiding
a blind subtraction from the current primary-observer phase. It also specifies
clock-safe paired measurements and true controller-receipt export. No schedule,
configuration or application source is changed by this analysis/planning step.

## 2026-11-03: compensated-grid experiment prepared

The user requests a test aiming to put combined outputs at n*125 us. The first
experimental correction is a fixed **5 us differential publication advance**,
based on the previous mean relative-output offsets, with **2 us admission
grace**. The final check therefore occurs at least 3 us before the old grid
point. This is not a qualified transport bound or an absolute selected-post
deadline. Ordinary zero mode defaults to advance zero; the experiment is opt-in.

Version 6 distinguishes the reference grid, earlier publication target, final
admission deadline, final-guard-to-update-return bracket and raw auxiliary
publication identity. The new fixed-storage `ZeroPhase` collector uses all
acknowledged command identities, independently of sampled event rows. Its
same-clock interval populations cover main-to-1, 1-to-2, 2-to-3 and
3-to-next-main; gaps and omitted labels are excluded explicitly. Per-step
same-main phase error and auxiliary-to-combined transport populations report
count/mean/spread/min/max through coherent snapshots, INDI and text telemetry.
Phase statistics reset on trial enable and unused step fields clear after
reconfiguration. Reader readiness and finite duration remain enforced.

The dedicated overlay and operator procedure are in
[`dmtweeter_synco_zero_phase_test.conf`](../../../apps/dmSyncopation/dmtweeter_synco_zero_phase_test.conf)
and the [trial guide](../../../apps/dmSyncopation/doc/zero-aux-rate-test.md).
After rebuilding/restarting synco and merging the overlay, the first command is:

```bash
bash apps/dmSyncopation/scripts/capture-zero-aux-test.sh 8000 10 5
```

The third argument verifies the startup advance before enabling; the helper
requires version 6 and a nonnegative grace smaller than the advance. It records
that selection and preserves existing finite-duration/snapshot/loss checks.
After reviewing this short run, a 120-second capture uses `8000 120 5`.
No logdump, controller, dmcomb, camera or CPU-setting change is part of this test.

Workstation verification passed:

- Production dmSyncopation clean build.
- Full application suite: **117 cases / 3,024,414 assertions**. Coverage includes
  4/6/8 kHz advanced targets, earlier preparation/admission skips, immutable later
  slots, exact auxiliary metadata and API brackets, full-rate phase collection
  despite sampling, all four adjacent transitions, identity/gap exclusions,
  normal-enable trial reset and clearing unused step fields.
- **41 private fake-service capture-helper tests**, including wrong/missing
  preset/schema rejection before enabling and legacy zero-advance compatibility.
- Bash syntax, clang-format verification and `git diff --check`.

Changes are scoped to dmSyncopation trial code, the new `syncoZeroPhase.hpp`
collector, the capture helper and its tests, preset and documentation. No DM
controller endpoint, dmcomb implementation or hardware state was changed. The
operator capture is pending; no agent operated the instrument.
