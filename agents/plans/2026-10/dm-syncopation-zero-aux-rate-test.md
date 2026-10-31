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
