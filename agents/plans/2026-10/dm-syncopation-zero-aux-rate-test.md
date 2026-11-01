# Coordinated zero-auxiliary empirical trial

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
