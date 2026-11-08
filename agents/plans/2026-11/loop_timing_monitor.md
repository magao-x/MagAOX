# Task Description
<!-- The first section is filled out by the user 
     
     This is intended as a guide for how to prompt a coding agent to solve a problem.
     
     Make a copy of this template with a new name under agents/plans/<YYYY-MM>. Then
     Fill out each subsection below as needed. Feel free to add additional
     subsections, etc.  

     When complete, commit it on your feature branch and then prompt the agent to review this file.
-->

## Problem Statement
We need a minimally invasive way to track loop computationaly latency

## Discussion
<!-- Put more expository text here.  Fully describe the expected behavior and rationale. Include references to other parts of the software that serve as examples. Provide a link to an API/SDK manual.   -->

We currently don't have a way to track the stop_time - start_time latency of the loop.  Here this means dmtweeter.sdk_start_time - camwfs.atime.

I think we can do this in a minimally invasive way by busy waiting and checking for an update on camwfs.atime and a the SDK start time in dmtweeter.  No semaphores.  And we some simple logic will line up the dmtweeter time that comes after a camwfs time.

Catch: dmtweeter doesn't really publish sdk start time.  So we should add am empty shmim that does so as part of post processing.

Then we the updates are detected, we calculate the difference, and publish a mean and stddev.

## Requirements
<!-- List the requirements that must be met.  A numbered list works best -->

Do not use semaphores

Any changes to dmtweeter must only have effect in post processing

The difference is large enough that I don't think this needs to be on an isolated cpu, but we should build in the option. I.e. waking up 20 usec late shouldn't be a problem.

The sleep period is adjustable.

## Tests and Metrics
<!-- List any useful metrics that should be targeted by tests -->

test detection of skipped frames, out of orders, etc.

## Scope and Caveats
<!-- Add any needed caveats and scope restrictions -->


# Instructions to Agent
<!-- Specific instructions for the agent.  Below is our standard, but you can modify it as needed. -->

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan

Planning review: 2026-11-06. Status: **implemented and verified offline**.
Feature branch: `jrmales/loop-timing-monitor`. Host verification returned `mx11`. This review used source and
existing engineering records only; no instrument commands were run.

## Task Summary

Add an independent `loopTimingMonitor` application that polls camera and DM
timestamp metadata, associates observations, and publishes the mean and sample
standard deviation of `SDK start - camera atime`. Add an optional, metadata-only
DM timestamp stream whose per-command publication occurs in post-processing.
The monitor will use no semaphore notifications and will support adjustable
polling sleep and optional CPU affinity.

This measures two software endpoints. It does not measure exposure-to-mirror
settling, and timestamp association alone does not prove which camera frame
produced a command.

### Findings in the current code

| Finding | Evidence and consequence |
| --- | --- |
| SDK entry is already timestamped in worker mode. | `apps/bmcCtrl/bmcCtrl.hpp`, `performSdk()`, records `SdkResult::startNs` using `CLOCK_MONOTONIC` before the native call. Reuse that value; do not add timestamp capture or publication to the SDK worker. |
| The pipeline has a suitable post-processing owner. | `apps/bmcCtrl/bmcPipeline.hpp`, `pipelineThreadExec()`, has the completed job's SDK timestamps and submission outcome. Publish there, after existing finalization and before retirement. |
| Other controller modes need separate consideration. | Joined worker mode has the timestamp in `finalizeCommand()`, but publication there adds work to the monitor callback. Direct synchronous mode does not retain `sdkStart` with default timing: `dev::dmTiming::captures()` retains it only with detailed timing enabled. Initial pipeline-only support best preserves the requested separation. |
| An optional SDK-result stream already exists. | `bmcPipeline.hpp` publishes `m_pipelineResult` from the SDK worker. It carries richer completion data and uses notification semantics. Enabling or extending that path is not the proposed post-processing-only solution. |
| Camera and SDK timestamps use different clocks. | `edtCamera::pdvAcquire()` copies the EDT timestamp into `m_currImageTimestamp`; `frameGrabber` publishes that as `atime`. The [camera investigation](../2026-10/ocam2k-acquisition-publication-timing.md), section “Ordinary timestamp placement is now identifiable,” identifies the inspected ordinary EDT mode as REALTIME, truncated to 1 us. SDK entry uses MONOTONIC. Direct subtraction is invalid. |
| Clock mapping and statistics already exist. | `apps/dmSyncopation/syncoPublicationClock.hpp` qualifies bracketed external-clock/MONOTONIC mappings; `libMagAOX/common/timingStats.hpp` supplies Welford accumulation and sample deviation. `libMagAOX/common/spscQueue.hpp` supplies bounded producer/consumer handoff. |
| The standard image monitor uses semaphores. | `libMagAOX/app/dev/shmimMonitor.hpp` claims, flushes and waits on a semaphore. Use direct ImageStreamIO attachment in a small polling helper rather than changing this widely used base. |
| An “empty” ImageStreamIO image still needs storage. | The locally inspected ImageStreamIO implementation rejects zero-sized axes. Use a 1 x 1 `uint8` dummy payload with zero reader semaphores. Its update helpers call `sempost`, including the logging semaphore, so publish metadata directly instead. |

The [previous full-loop baseline](../2026-10/full-loop-timing-20261025.md)
measures cadence and local stages, and explicitly does not establish absolute
camera-to-DM latency. It therefore cannot supply the latency bound needed for
unambiguous temporal association.

## Key Assumptions

- The camera producer, BMC controller and monitor share a host and clock
  domains. Relayed streams from another host are outside the initial scope.
- The first implementation targets BMC pipeline operation with
  `sdk.workerEnabled=true` and `sdk.pipelineEnabled=true`. No controller mode
  will be enabled automatically; the user approved this scope.
- The camera clock is explicitly configured. The proposed configuration for
  the previously inspected ordinary OCAM/EDT mode is `realtime`; metadata alone
  cannot identify the clock or establish that every future camera mode uses it.
- Initial temporal association assumes one relevant SDK submission per camera
  frame and a known positive latency shorter than one camera period. This is
  an operating assumption confirmed by the user, not something the monitor
  can prove from the two streams.
- “No semaphores” means no new semaphore synchronization, claims, waits, flushes
  or posts in this feature. Existing camera/controller synchronization remains
  unchanged. ImageStreamIO reserves logging-semaphore storage even for a
  zero-reader-semaphore image; the feature will not use it.
- Reporting includes read-only INDI and persistent `telem_fgtimings`, with a
  two-second trailing window and approximately one-second publication cadence.
- `monitor.maxFPS` is configurable, default 3600. Its reciprocal bounds latency
  (about 277.778 us at the default); actual adjacent camera timestamps also bound
  each association. This is a declared ceiling, not an instantaneous cadence test.
- Collection requires configurable camera and DM devices to report
  `fsm.state=OPERATING`, and a configurable loop device to report
  `loop_state.toggle=On`. Defaults are `camwfs`, `dmtweeter`, and `holoop`.
  Use passive INDI subscriptions: each source is initially unknown, and the
  last received state remains valid until an update changes it. State has no
  receipt-age expiry; camera/SDK timestamps and retained samples still do.

## Requirements

1. Reuse the original SDK-start timestamp. Never substitute SDK return,
   post-processing time, or poll-observation time for either requested endpoint.
2. Restrict new per-command DM work to post-processing. Add no work to actuator
   preparation, SDK dispatch, or the native-call path. Make publication opt-in
   and independent of detailed timing collection.
3. Use direct polling of `write`, `cnt0`, timestamps and stream identity with
   bounded consistency checks. Do not read image pixels or claim semaphore slots.
4. Convert clocks in the independent monitor, with bounded capture uncertainty,
   age limits and discontinuity detection. Preserve integer nanoseconds until
   after subtraction; report duration statistics in seconds, matching existing
   timing properties.
5. Make polling sleep configurable, including zero for explicit busy polling.
   Reuse `MagAOXApp::threadStart()` for optional cpuset and priority. Default to
   ordinary scheduling and no affinity; CPU isolation is an operator choice.
6. Report accepted sample count, freshness and observation quality alongside
   mean/deviation. Never hide skipped publications or ambiguous associations
   behind an apparently valid statistic.
7. Bound queues, retries, retained samples and shutdown latency. Recover from
   absent/recreated streams without controlling their producers.
8. Keep diagnostic publication errors from becoming DM command failures. Report
   unavailable diagnostics without retry loops, allocation or logging per frame
   in post-processing.

## Questions and Points of Clarification

The user approved implementation and resolved the planning questions:

1. Initial support requires the existing BMC pipeline.
2. One relevant submission per camera frame and sub-period latency are confirmed.
   The fastest FPS must be configurable, default 3600. Use its reciprocal rather
   than asking for a separate maximum-latency value.
3. Persistent telemetry is required, just like `fgtimes`. Reuse the existing
   `telem_fgtimings` six-double schema: camera acquisition cadence, SDK-start
   cadence, and paired latency, each with mean and sample deviation in seconds.
   Existing DM timing already reuses this record for documented endpoint pairs.
4. Camera and DM must both be OPERATING and the loop must be on. Device names
   are independently configurable. Unknown/malformed/negative state closes the
   gate. The user subsequently specified pure INDI publish/subscribe: assume
   the last received state until it changes, without periodic property requests
   or receipt-age expiry.

**Superseded design:** the initial implementation used periodic INDI
get-properties requests and a configurable receipt-age timeout because the
wrapper does not forward source deletion/disconnection to application callbacks.
The user explicitly replaced that policy with passive subscriptions. Normal
base-class subscription establishment/recovery remains; the monitor adds no
periodic refresh. The obsolete `state.pollSec` and `state.timeoutSec` configuration
keys are removed at the user's request; there is only one deployment. Remove its
old `[state]` configuration section when updating. INDI silence alone does not
close the gate.
A gate-generation change clears queued/pending samples and retained statistics;
reopening requires fresh stream baselines. Camera/SDK timestamp freshness,
clock-mapping age limits and statistics-window expiry remain unchanged.

No outstanding clarification blocks implementation. The app follows the existing
header-only pattern with out-of-class definitions.

## Tests

All tests will be offline, with injected clocks and SDK mocks or private
ImageStreamIO fixtures under a temporary `MILK_SHM_DIR`.

| Test area | Specific checks |
| --- | --- |
| Endpoint publication | Disabled configuration does not create/update a stream. An enabled pipeline publishes the original `startNs`, even when post-processing is delayed. No-send/cancelled-before-send jobs do not masquerade as SDK entries. Define and test failed submissions as excluded from the successful-command population. |
| Post-processing placement | Extend existing BMC pipeline mocks and `BMCCTRL_TEST_PIPELINE_POST` hook: no new publication before the post worker runs; SDK calls, arguments and existing timestamps are unchanged. Test with detailed timing disabled. Publisher errors do not fault command execution. |
| Semaphore independence | Zero reader semaphores; no claim/wait/flush/post/update-helper calls from the new publisher or monitor. Account separately for unrelated pre-existing controller synchronization. |
| Coherent metadata | Active writer, timestamp/counter changes during read, repeated unchanged polls, malformed timestamps and bounded retry exhaustion. Concurrent synthetic publication must never admit a torn new DM timestamp. |
| Ordinary association | Known fixed and varying latencies; exactly the expected accepted count and no reuse of either endpoint. Exercise both read orders when both streams update between polls. |
| Delayed visibility | SDK publication arriving after the next camera observation still pairs by recorded event time. Delayed observation within coverage limits changes neither measured mean nor deviation. |
| Coverage loss | Camera and DM counter jumps separately; exact count of missed stream publications where inferable; counter regressions/wrap, handoff/pending-queue overflow, and startup baselines. Reject associations crossing the loss and resume only after fresh continuity evidence. |
| Ordering and ambiguity | Timestamp regressions with advancing counters, zero/multiple SDK events per camera interval, missing commands, boundary equality and uncertainty overlapping a boundary. Extra indistinguishable commands and latency aliasing are documented limitations, not falsely claimed detectable errors. |
| Clock mapping | Known offset, wide/failed brackets, stale mapping, clock step, wrong declared domain, second rollover and integer range checks. Include the known camera timestamp quantization in uncertainty. Regress existing synco clock tests if the helper is generalized. |
| Statistics and state | Known sample mean and n-1 deviation; empty/singleton populations, expiry, stopped sources, restart epochs, and capacity limits. Mean may exist for one sample; standard deviation must remain unavailable. |
| Operating gate | Configurable device subscriptions, exact FSM/switch semantics, initially unknown/malformed/negative state, each condition independently false, affirmative state retained without age expiry, no periodic property requests, recovery, rapid Off/On, and rejection of queued samples from prior gate generations. |
| Persistent telemetry | Exact six-field fgtimes mapping and units, sample deviations, normal/forced recording, no stale or gate-disabled emission, and no records before all populations are valid. |
| Lifecycle/configuration | Missing stream at startup, incompatible stream, same-name recreation/new inode, collision with another publisher, bounded shutdown during retry/sleep, invalid sleep/window/capacity/latency settings and optional scheduling errors. |

Run the new monitor suites, affected BMC mock suites, shared clock/statistics
regressions, and app builds. Use the repository's `tests/Makefile.one` and
application Makefiles. Document any unavailable build dependencies rather than
substituting instrument runs. A later operator-run comparison can measure
observation coverage and controller/post-worker load with monitoring off/on;
software tests alone cannot establish production timing impact.

## Implementation Plan

### 1. Confirm the contract and reuse boundaries

- User decisions are recorded above; implementation uses the feature branch
  `jrmales/loop-timing-monitor`.
- Generalize the clock-neutral calculation in `syncoPublicationClock.hpp` into
  a small common helper, retaining a compatibility alias/wrapper for synco and
  its existing behavior/tests. Its callers declare which external clock they
  sample; do not import an application namespace into another app.
- Reuse `timing::Accumulator`, `timing::Stats`, `spscQueue` and `threadStart`.
  Keep stream polling app-local initially; a broad new device base class is not
  justified by this one new consumer.

### 2. Publish the SDK endpoint in BMC post-processing

- Add a startup-only optional stream name, proposed `sdk.startTimeStream`, empty
  by default; example value `dmtweeter_sdk_start`. Validate supported controller
  mode and stream ownership before starting the pipeline.
- Create a 1 x 1 `uint8` stream with zero reader semaphores. The pixel is unused.
  Document `atime = original CLOCK_MONOTONIC SDK start`,
  `writetime = CLOCK_MONOTONIC diagnostic publication time`, and `cnt0` as the
  diagnostic publication sequence, **not** a camera frame number. Initial
  timestamps/counter do not constitute a sample.
- Use a single post-worker writer and explicit ordered metadata publication:
  mark write-in-progress, store timestamp fields, advance the publication
  counter, then clear write-in-progress. Use the project's atomic access
  conventions for this new producer and its reader. Do not call ImageStreamIO
  update or notification helpers.
- Publish successful native set submissions once, retaining their original
  start even if later diagnostic/product work fails. Exclude administrative
  SDK calls, no-send jobs and failed native submissions; document this population
  and use existing controller outcome/status properties to report those
  exclusions. The new publication counter alone cannot identify failed attempts.
  If the agreed requirement instead includes all native attempts, revise this
  contract before coding rather than silently mixing populations.
- Put the hook in `pipelineThreadExec()`, after existing finalization and before
  job retirement. Create/close outside the per-command path and after the writer
  is stopped. Adapt safe name/collision/lifetime patterns from `dmCompletion`;
  do not reuse its semaphore-based publisher unchanged or replace arbitrary
  existing streams. Give each publisher incarnation a distinct stream inode so
  readers can detect restart even if they miss the initial zero counter.

### 3. Implement the independent polling app

- Add `apps/loopTimingMonitor/` with a header-only class, entrypoint `.cpp`,
  Makefile and documented example configuration. Suggested settings are camera
  and SDK stream names, camera clock, `pollSleepUs`, thread priority/cpuset,
  `windowSec`, bounded capacity, `maxFPS` (default 3600), clock-bracket/age
  limits and device names. Register passive INDI subscriptions for the operating
  gate; do not add periodic state refresh or receipt-age expiry. Remove
  the obsolete `state.pollSec` and `state.timeoutSec` keys entirely.
- A dedicated worker attaches directly with ImageStreamIO and reads metadata
  only. Baseline existing counters on attachment; do not count an old image as
  a new event. Check file/inode replacement periodically, with slower retry
  pacing while disconnected even if active polling sleep is zero.
- Use bounded write/counter/timestamp consistency checks modeled on
  `dmSyncopation::processImage(cameraTimingT)`. The existing camera producer's
  metadata protocol remains unchanged; describe its snapshot qualification as
  best-effort rather than claiming atomic frame provenance.
- For new camera observations, sample MONOTONIC/external/MONOTONIC brackets in
  the monitor and map the stored camera timestamp into the SDK clock. Reject
  invalid/wide/stale mappings, retain uncertainty and reset association on clock
  discontinuity. No new clock conversion is done in the DM critical path.
- Hand bounded observation records and quality information to ordinary app
  logic using the existing SPSC queue. Distinguish queue loss from producer
  counter gaps. Keep INDI, aggregation, allocation and routine logging outside
  the fast polling loop. Sleep after scanning both streams; zero selects
  continuous scanning. Neither sleeping nor delayed publication is subtracted
  from the latency measurement.

### 4. Associate timestamps conservatively

- Maintain bounded camera and SDK event histories in timestamp order per source;
  arrival order across the two streams is irrelevant. Never combine only the
  latest values or blindly zip counters from unrelated producers.
- Under the agreed sub-period contract, consecutive camera timestamps define
  an interval. Require exactly one published SDK start in the full interval,
  then require its positive latency to be below both `1 / maxFPS` and the
  actual camera interval; a second start invalidates the interval even if outside
  that bound. Do not reject an entire interval simply because ordinary cadence
  jitter puts it below `1 / maxFPS`. Reject boundary overlap due to clock uncertainty.
- Finalize only after the next camera timestamp is known and contiguous SDK
  observations have advanced past the interval. This small reporting delay
  allows late post-processing publication and detects multiple candidates.
  Expire incomplete intervals if the source stops; do not invent a missing
  timestamp or reuse the previous SDK event.
- A counter gap/regression, stream replacement, timestamp regression, clock
  epoch change, unqualifiable new update or queue overflow breaks continuity
  and clears pending matches.
  Establish fresh baselines before resuming. Repeated unchanged polls are
  ordinary, not duplicates or errors. Count missing stream publications
  separately from rejected associations and physical frame loss.

### 5. Publish statistics and quality

- Retain accepted integer-nanosecond differences for the bounded trailing time
  window; aggregate with the existing common accumulator at ordinary app cadence.
  Base age/expiry on event time in MONOTONIC, not when post-processing becomes
  visible. Expire old values even when streams stop advancing. Clear retained
  statistics on source/clock epoch changes and return to warming/unavailable.
  The pure matcher can retain accepted history across ordinary gaps, but the
  application conservatively clears its full window on any capture-continuity
  failure; pending associations never cross a gap.
- Publish read-only INDI latency mean, sample deviation, count, window/span and
  age, plus clear disconnected/warming/valid/stale/unavailable state. Publish
  missed-camera/DM observations, invalid/ambiguous pairs, queue drops, clock
  failures and attachment epochs. Expose the endpoint/clock contract.
- Reuse `telem_fgtimings` and `dev::telemeter` interface macros. Map `atime`
  to camera interval, `wtime` to SDK-start interval and `wmatime` to paired
  camera-to-SDK latency; the corresponding jitter fields hold sample deviations.
  Emit normal and forced records only when all populations are valid, fresh and
  gated on. Keep gate/coverage diagnostics in INDI; the existing telemetry schema
  remains unchanged.

### 6. Integrate, verify and document

- Add the app to the root Makefile's RTC and workstation build lists; register
  tests in `tests/tests.list` and an `application_unit_test` group in
  `tests/groups.dox`.
- Expected changes: new app/helper/tests/config/docs; BMC controller header,
  pipeline header, mock tests and SDK-worker documentation; common clock helper
  and synco compatibility/tests; root build/test registries; this plan. Verify existing telemetry serialization through application tests;
  no new logger schema is required. No camera
  acquisition or SDK-call-path changes are planned.
- Apply full-file documentation review to touched source files, inline parameter
  documentation, explicit Allman control-flow braces, Doxygen test references,
  and `clang-format`; restore `{ //mutex scope` where required. Keep functional,
  documentation-only and formatting-only commits separate as applicable, with
  this plan accompanying the engineering changes it records.

## Execution Notes

- 2026-11-06: user approved the plan and requested implementation. Added the
  configurable 3600 FPS ceiling, persistent fgtimes telemetry and configurable
  camera/DM/loop operating-state gate. Host rechecked as `mx11`.

### Completed implementation and verification

- Added `loopTimingMonitor` with configurable source devices/streams, the 3600 Hz
  default maximum FPS, bounded semaphore-free polling, common-clock mapping,
  conservative pairing, INDI quality/state and persistent six-field fgtimes.
- Added optional `sdk.startTimeStream` publication exclusively in the BMC ordered
  post worker. Existing SDK capture/dispatch and command outcomes are unchanged;
  optional diagnostic failures are nonfatal. The stream is disabled by default.
- The initial source-state design used one-second refreshes and a five-second
  timeout. **Superseded by the passive INDI subscription update below.** Gate
  transitions and capture discontinuities reject queued old-generation samples
  and clear the window.
- Initial review fixed coherent gate/INDI publication, expiry before same-state
  refresh, actual requested scheduling-policy verification, failed ImageStreamIO
  attach cleanup, and polling-side counter-regression quality reporting.
- Extracted the existing clock mapper into a shared header with a compatibility
  alias for dmSyncopation. Added build dependencies and test registration.
- Ran clang-format on touched C++ sources and restored mutex-scope annotations.
  Source and agent-section whitespace checks pass. The original task section
  above Agent Findings and Plan is byte-for-byte unchanged, including its four
  pre-existing trailing-whitespace lines; those intentionally remain in the
  newly tracked plan.

| Verification | Result |
| --- | --- |
| `make -C apps/loopTimingMonitor -j2` | Builds and links successfully. |
| `loopTiming_test` | 142 assertions / 14 cases pass. |
| `loopTimingGate_test` | 27 assertions / 3 cases pass. |
| `loopTimingStream_test` | 61 assertions / 5 cases pass. |
| `loopTimingMonitor_test` | 267 assertions / 4 cases pass, including normal/forced fgtimes serialization and gate suppression. |
| `loopTimingMonitor_integration_test` | 50 assertions / 2 cases pass using the actual worker and private synthetic streams; known 100 us cross-clock latency, counter reset, queued-sample rejection, gate recovery and disconnected shutdown. |
| BMC SDK-mocked suite | 1582 assertions / 37 cases pass; includes post-only publication, disabled detailed timing, concurrent snapshots, failed submission and nonfatal diagnostic errors. |
| Existing dmSyncopation `[publicationClock]` cases | 63 assertions / 4 cases pass after rebuilding against the shared clock header. |

Tests use `make -C tests -f Makefile.one t=<test-source>`; affected header consumers
were explicitly rebuilt. No instrument commands were issued. The BMC vendor SDK
is absent from this workstation, so production BMC linking and operator timing
qualification remain follow-up work. No test failure remains unresolved.

### Passive INDI subscription update (2026-11-06)

- Source audit confirmed that the monitor's forced `sendGetPropertySetList(true)`
  calls sent real requests once per second. Indiserver deduplicates its snooping
  list, but forwards every request and distributes the source's definition replies
  to interested clients/drivers. Repeated matching definitions can keep wildcard
  `getINDI` queries alive by resetting their response timeout.
- The user requested pure publish/subscribe and last-received-state semantics.
  Remove application-level refreshes and gate receipt-age expiry. Initially
  unknown, malformed or negative source states close the gate; affirmative states
  remain valid until subsequent updates change them. Rapid Off/On transitions
  still invalidate queued observations and establish a fresh measurement epoch.
- The user confirmed that there is only one deployment and requested removal
  of `state.pollSec` and `state.timeoutSec` entirely, including the example's
  `[state]` section. Remove these keys from the deployed configuration on update.
  Stream polling, timestamp freshness, clock mapping and trailing-window expiry
  are unchanged.
- Verification for this update: the app builds and links; gate tests pass
  51 assertions / 5 cases; app tests pass 283 assertions / 5 cases; worker
  integration tests pass 50 assertions / 2 cases; the new private-driver
  subscription regression passes 30 assertions / 1 case. Total: 414 assertions
  across 13 cases. The subscription test observes three initial requests and
  no further requests after resolution across repeated real appLogic calls;
  an Off callback still closes collection. The synthetic-hour test confirms
  that source states persist while timing samples expire and later samples
  can be collected without another state update.
- Ran clang-format on changed C++ files and restored mutex-scope comments.
  Whitespace and independent source review pass. Tests use only workstation-local
  private resources; no instrument commands were issued. The verification table
  above records the original implementation. The Syncopation extension below
  remains a separate, unimplemented design.
- Files for this update: `loopTimingMonitor.hpp`, `loopTimingGate.hpp`,
  `loopTimingMonitor.conf`, `README.md`, gate/app/worker tests, new
  `loopTimingMonitor_subscription_test.cpp`, `tests/tests.list`, and this plan.

### Affected files

- `.gitignore`, root `Makefile`, `tests/tests.list`, `tests/groups.dox`.
- `apps/bmcCtrl/Makefile`, `bmcCtrl.hpp`, `bmcPipeline.hpp`,
  `tests/bmcCtrl_test.cpp`, `doc/sdk-worker.md`.
- `apps/dmSyncopation/Makefile`, `syncoPublicationClock.hpp`.
- `libMagAOX/app/dev/dmSdkTimingStream.hpp` and
  `libMagAOX/common/publicationClock.hpp`.
- New `apps/loopTimingMonitor/Makefile`, `loopTimingMonitor.hpp`,
  `loopTimingMonitor.cpp`, `loopTiming.hpp`, `loopTimingGate.hpp`,
  `loopTimingStream.hpp`, `loopTimingMonitor.conf`, `README.md`.
- New monitor tests: `loopTiming_test.cpp`, `loopTimingGate_test.cpp`,
  `loopTimingStream_test.cpp`, `loopTimingMonitor_test.cpp`,
  `loopTimingMonitor_integration_test.cpp`.
- This planning/execution record.

## Follow-up and Edge Cases

- **A timestamp match is an inference.** For a steady 500 us frame period,
  true latency of 600 us can appear identical to 100 us when no frame identity
  is propagated. The monitor cannot detect every violation of the sub-period
  assumption. A configurable bound is not proof of that assumption.
- Extra commands can also be indistinguishable from the desired loop command.
  Reliable support for auxiliary/syncopation operation or multi-frame latency
  needs additional source identity; that is a separate, more invasive design.
- Polling latest metadata cannot recover overwritten updates. Waking 20 us
  late does not bias an otherwise valid timestamp difference, but enough delay
  can reduce coverage. Report the loss rather than claiming complete capture.
- Post-processing-only publication can still consume cache/memory bandwidth and
  delay pipeline credit retirement. The design avoids new pre-SDK work but
  cannot promise zero indirect timing impact. Benchmark on the workstation and
  have the operator qualify production impact.
- The ordinary EDT timestamp includes driver completion processing and 1 us
  truncation. Clock-conversion uncertainty does not remove that quantization or
  make the endpoint a physical acquisition timestamp.
- Stream restart, a frame-rate change that violates the association contract,
  realtime clock steps, and observable gaps/staleness from long SDK calls or
  post-processing stalls must yield a fresh association epoch or unavailable/
  stale statistics. Undetectable whole-period aliasing remains the limitation
  described above.
- Deployment configuration, CPU isolation changes and real-instrument
  validation remain operator actions. No hardware commands are part of agent
  implementation or tests.


## Syncopation step-0 extension (2026-11-06)

Status: **implemented and verified offline** (2026-11-07). The user
reports the deployed instance as `holoop-time`, at commit `56cf3fc0`, and requests
that only synco step 0 contribute to loop timing; auxiliary DM writes are ignored.
The user confirmed **production providers** as the intended four-step mode.
On 2026-11-07 the user additionally required automatic switching between ordinary
and step-0-only timing when synco's enable state changes, without restarting or
reconfiguring the monitor. The disabled-unavailable policy originally proposed
for this extension is superseded by the both-mode classification design below.
The supplied snapshot has a 500.001 us camera period, 166.921 us latency mean,
3.004 us sample deviation and 3998 accepted samples over roughly two seconds.
Its quality counters are cumulative, not necessarily losses in that last window.

### Behavior before the extension with four steps

The standalone matcher requires exactly one SDK start per adjacent-camera interval.
Four-step operation normally presents multiple candidates, increments ambiguous
counts and stops admitting new latency samples. Previously accepted samples can
remain valid until they expire from the two-second window; source gaps can clear
that window earlier. Persistent fgtimes stops once the populations no longer
qualify. This affects the monitor only, not controller/scheduler operation.
This is not guaranteed rejection of all auxiliary traffic: an interval containing
only one auxiliary candidate can still pass the current temporal test if the
main is absent and that candidate lies within the configured latency bound.
The deployed monitor therefore cannot guarantee step-0-only statistics during
syncopation.

A position-based filter is unsafe. At a 500 us camera period, four evenly spaced
commands would be 125 us apart. Illustratively extending the observed 166.921 us
step-0 offset gives 166.921, 291.921, 416.921 and 541.921 us. The preceding cycle's
step 3 would then be only 41.921 us after the next camera timestamp. These are
association examples, not predictions of measured synco SDK timing. Neither
“first SDK after camera” nor an SDK counter modulo four identifies step 0.

### Existing evidence to reuse

- `bmcCtrl::PipelineJob::identity` already freezes the combined-stream inode,
  publication counter and raw writetime seconds/nanoseconds. It is validated
  around vector conversion and remains available with `result.startNs` in the
  ordered post worker. No new capture is needed on the SDK critical path.
- The shared production/zero-test engine identifies a genuine main combined
  acknowledgement and applies its additional primary-trace qualification before
  calling `m_zeroTiming.main(...)` in `processZeroTestPrimary()`.
- This positive main qualification requires valid named-primary combiner trace
  linkage and advancing process timestamps (`MeasureTiming`). The ordinary loop
  monitor did not require either. Missing qualification must have an explicit
  unavailable reason; the extension must not alter dmcomb configuration itself.
- `synco::ZeroTiming` already joins this exact combined key to native SDK-result
  records. Reuse its key/qualification principles rather than infer identity
  from independent counters or arrival order.
- The advertised `_timing` stream contains future schedules, and provider
  `_ready`/`_ack` records describe payload preparation/copy disposition. They do
  not establish that an SDK submission occurred for that step.
- Existing ZeroEvent log transport is sampled (`eventStride` defaults to four)
  and cannot be used as a complete online label source.
- The combiner's `trigger_inode` is a configured trace link in DELAY mode;
  auxiliary outputs can carry that same named-primary link. Live trace matching
  alone is insufficient, and reading it later in DM post-processing can observe
  a different command.

### Implemented design

1. Preserve the current v1 SDK timestamp stream and standalone monitor behavior.
   Add an optional versioned diagnostic stream carrying the original SDK-start
   time and frozen combined identity, published only from BMC post-processing.
   Include publisher incarnation and an explicit record sequence.
2. Export positively qualified main identities in both enabled and disabled
   production-provider operation. Each record includes scheduler incarnation,
   mode generation, effective enabled state and primary cycle identity. Reuse
   the existing active main/step-0 qualification and extract its read-only part
   for passive observation while disabled. Capture a fixed-size record into a
   bounded queue before any sampled logging. Publish from ordinary diagnostic
   processing; do not add work in the scheduler's quiet final-publication window.
3. Retain a bounded sequenced history in both diagnostic transports. At the
   reported 2 kHz camera rate, four steps nominally produce 8 kHz SDK events,
   but post-processing may publish several in a burst. A latest-only word or a
   requested 10 us poll sleep cannot guarantee their recovery. Reuse one small
   shared record-ring utility for these two transports, with coherent slots,
   explicit overruns, and no semaphore operations.
4. The monitor joins SDK records and qualified-main labels by all four combined
   identity fields, independent of arrival order. In automatically managed mode,
   retain this exact join both when synco is enabled and when disabled. Only
   positive main matches enter camera pairing. Disabled operation still reports
   ordinary one-main-per-camera timing, but an unlabelled SDK write is never
   admitted merely because the last enable notification was Off. Unmatched or
   expired records are excluded as unclassified; absence of a label is not proof
   that a command was auxiliary.
5. Preserve raw SDK transport loss accounting before filtering. Feed matched
   mains using their primary-cycle continuity, not the original all-SDK counters
   (e.g. 1, 5, 9), which the existing matcher would incorrectly flag as gaps.
   Restore event order before advancing the filtered population.
6. Continue reporting the existing fgtimes schema: camera cadence, **step-0 SDK
   cadence**, and camera-to-step-0 latency. Auxiliary writes contribute to none
   of those latency/SDK-cadence populations. Keep `monitor.maxFPS=3600` as a
   camera-rate limit; do not multiply it by four. Size full-rate capture separately.
7. Add configurable synco device/diagnostic stream names and switch modes
   automatically: disabled selects ordinary main timing; enabled selects
   step-0-only timing. Subscribe to INDI passively without periodic property
   requests or receipt-age expiry. Ordered diagnostic mode generations and
   per-main effective-mode labels establish which measurement epoch each command
   belongs to; notification arrival time must not classify a command. Clear
   queued associations and statistics across actual enable/disable, mode,
   step-count, clock and publisher changes. Repeated unchanged notifications do
   not restart collection. Use a brief warming/transition state while new-mode
   main identities and camera/SDK baselines are acquired, then resume without
   operator action. Unknown or conflicting mode evidence, missing labels, record
   loss or unsupported modes suppress qualification until coherent evidence
   returns; never fall back to admitting arbitrary SDK writes.
8. Keep legacy non-synco instances/configurations working unchanged. Implement
   the new filter for the user-confirmed production-providers mode. The shared
   main-classification hook also covers coordinated zero tests, but legacy
   scheduling lacks this hook and is explicitly unsupported by the new filter.

### Automatic enable/disable switching

The production `enable` callback sets `m_enabled=false` before taking
`m_cycleMutex`, then acknowledges Off after the current scheduler callback
quiesces. This prevents subsequent scheduler publications but does not drain
already published auxiliary triggers through dmcomb or BMC. In particular, the
auxiliary-acknowledgement wait aborts on disable, so a pending auxiliary can
produce a later combined counter than the one visible at Off. Neither the INDI
Off arrival time nor a snapshot of the combined counter at that instant is a
safe cutoff. Similarly, On notification delivery can lag the first auxiliary.

Continue exporting positive main classifications in both modes instead of
trying to admit all SDK writes after a timed grace period. A delayed auxiliary
SDK result cannot match a main identity regardless of when enable updates,
main labels or SDK records arrive. Epoch changes discard old-mode matches and
avoid mixed statistics; measurement can briefly warm after an edge, but returns
automatically when sufficient valid new-mode samples exist.

Use a consistent diagnostic mode generation for both edges and internal disables;
the current schedule generation alone is insufficient because Off paths do not
all advance it. Freeze the mode generation with the observed primary identity
and reject qualification that crosses a transition. Never assign a late SDK
record the mode observed at its publication or receipt time.

The existing disabled branch returns before main qualification. Extract the
primary/combined progression checks and `impactClassified` validation
into reusable read-only helpers, then use them from both paths. Do not simply
remove that early return: active waits currently require `m_enabled`, and the
active path also starts trials and prepares/publishes auxiliary commands. The
passive classifier must:

- Observe only primary and combined stream identities/trace using nonwaiting
  snapshots. Retain a bounded prior-primary history so a later callback can
  qualify the preceding main. Skip uncertain samples; introduce no new waits,
  sleeps, clocks, allocations, logging or stream publications in the callback.
  A mode-generation change invalidates pending passive classification.
- Require no provider-ready records, auxiliary copy/read-release permissions,
  valid multi-step schedule or SDK-result observer to classify an ordinary main.
- Perform no destination writes, auxiliary publications or trial-state updates.
- Retain conservative first-frame rebaselining, counter-gap rejection and trace
  qualification; a named-primary trace alone still does not establish a main.
- Cover startup while disabled, explicit toggles and internally triggered
  disables/restarts; mode records and main records share ordered generations.

Without synco configured, the existing standalone monitor remains available.
With automatic synco integration configured, both modes require the synco main
classification source. A stopped or unqualified classification source makes
timing unavailable until it recovers, even if the last INDI enable value is Off.
This changes no instrument state and requires no additional BMC changes beyond
the already planned post-processing command-identity publisher.

### Extension verification

- Four-step synthetic traffic with known step-0 latency and step 3 crossing the
  next camera timestamp: only step 0 contributes, with camera-rate SDK cadence.
- Skipped/cancelled auxiliary steps do not shift classification. A missing or
  failed main is omitted and never replaced by a nearby auxiliary SDK event.
- Labels arriving before/after SDK records, burst publication, clock uncertainty,
  duplicates, source restarts and wrong/partially matching identities.
- Ring overwrite, producer queue loss, missing main labels and bounded join
  expiry: count the loss and reject affected associations without inventing data.
- Transitions from one to four steps and back, rapid enable/disable, malformed
  state and unsupported modes clear the measurement epoch and suppress forced
  telemetry. Unchanged affirmative INDI state does not expire with age; diagnostic
  records still obey their continuity, freshness and join-expiry checks.
- Startup with synco Off automatically produces ordinary timing; On switches
  to step-0-only timing and Off returns to ordinary timing without restarting.
- Delayed/reordered INDI notifications, labels and SDK records cannot relabel
  pre-transition commands. Include disable after auxiliary publication but before
  combiner acknowledgement, auxiliary SDK return after Off, On arriving after
  auxiliary SDK events, and rapid Off/On with the same final boolean value.
- Passive main classification works without ready providers or an armed
  multi-step schedule, never performs auxiliary writes and retains rejection of
  uncertain main identities. Repeated unchanged state produces no resets or
  getProperties traffic. Loss/restart of classification recovers automatically.
- `eventStride` and text logging configuration do not affect label completeness.
- No new SDK-path clock reads or semaphore synchronization; disabled diagnostic
  configuration preserves existing behavior. Operator qualification remains
  necessary for indirect cache/load and scheduler effects.

Extension touch points: shared record transport/key helpers; the BMC post
publisher and tests; synco qualified-main export and tests; monitor joining,
configuration, state handling, tests and docs. This design retains the existing
camera-to-main temporal association assumption; it does not introduce a true
camera-frame ID through the complete AO pipeline.

### Extension execution notes (2026-11-07)

- Implemented the approved automatic both-mode classification. The enabled
  callback captures a fixed-size record with a bounded nonblocking queue push at
  the already-qualified main hook. Disabled observation uses nonwaiting metadata
  and trace snapshots. Neither path adds clocks, allocation, waits, logging,
  semaphores or shared-stream publication. Unconfigured diagnostic branches
  return before additional capture work. Ordinary app logic manages diagnostic
  attachment, publishes history records and reports queue/publication health.
- Added `libMagAOX/common/dmTimingEvents.hpp` and
  `libMagAOX/app/dev/dmTimingEventStream.hpp`: one validated record contract and
  reusable, versioned, zero-semaphore history for both publishers. Coherent
  atomic slots have contiguous transport sequences; separate producer sequences
  reveal records lost before transport. Recreated publishers get fresh inodes.
- BMC emits successful native SDK starts with the existing frozen four-field
  combined identity from the ordered post worker. Existing v1 timestamp output
  remains unchanged. Publisher failures are counted without faulting the DM.
- Added `apps/dmSyncopation/syncoMainTiming.hpp` for mode generations, capture
  queuing and passive main observation. Classification does not depend on sampled
  event logging. An independent read-only combined attachment supports disabled
  operation without ready providers or active auxiliary scheduling. Filesystem
  attach/check/retire operations run outside the cycle mutex.
- Added `apps/loopTimingMonitor/loopTimingSynco.hpp` for bounded exact joining
  with ordered SDK watermarks, delayed-label handling, duplicate/conflict checks,
  expiry and explicit loss accounting. Only qualified mains enter camera pairing
  in either mode; their primary counters establish SDK cadence continuity.
- The monitor passively subscribes to synco enable once. Actual INDI and diagnostic
  generation changes reset statistics; unchanged notifications do not. Telemetry
  requires the current received enable state and diagnostic mode to agree.
  Status includes mode, mode generation and join quality, with `waitingMode`,
  `modeMismatch` and `waitingMain` availability reasons. Stopped classification
  expires normally and never enables raw-SDK fallback.
- Updated application Makefiles, test registration, example configurations and
  deployment documentation. All three applications must be rebuilt; automatic
  mode requires both publishers plus all three monitor `[synco]` settings.
- Final review checked disabled transitions against the actual counter-driven
  dmcomb implementation: a late auxiliary computation can acquire a newer primary
  trace counter at publication. Its process-start must therefore be at or after
  that primary publication, in addition to exact one-output progression from a
  pre-primary baseline. An auxiliary-only computation begun earlier is rejected.
  A first response that coalesces the new main and last auxiliary is a main
  response under this all-current-input combiner contract. Trace linkage alone
  remains insufficient. Regression coverage explicitly distinguishes both cases.

Offline validation on workstation `mx11` (no instrument actions):

- Shared event transport: 225 assertions / 11 cases.
- Full BMC mock SDK suite: 1857 assertions / 43 cases. The proprietary SDK is
  absent on this workstation, so a production BMC binary was not linked.
- Full dmSyncopation suite: 6,694,622 assertions / 156 cases; dedicated main
  timing helper: 139 assertions / 7 cases after the final review regressions.
- Pure monitor join: 136 assertions / 4 cases, including the four-step example
  with step 3 crossing into the next camera interval.
- Automatic monitor worker using private shared streams: 1108 assertions / 2
  cases, including Off/On/Off, delayed auxiliary SDK events, source recreation,
  history overwrite and recovery. Five consecutive runs passed.
- Existing monitor application tests: 294 assertions / 5 cases; standalone worker
  integration: 50 assertions / 2 cases. Passive INDI subscription test: 73
  assertions / 1 generated case covering standalone and automatic configuration.
- Both `loopTimingMonitor` and `dmSyncopation` applications build and link.
  Changed C++ files pass clang-format (with required mutex-comment restoration),
  file-header checks and `git diff --check`. Original task text is preserved.

Remaining deployment qualification: operator updates/restarts the three instances
and checks coverage, source trace/MeasureTiming configuration, history capacity
and timing impact on the RTC. Small added CPU/cache work is not zero impact; no
hardware benchmark was run. The existing camera-to-main temporal association
still lacks an end-to-end camera frame ID and retains its whole-period aliasing
limitation. A non-counter-driven combiner or additional external channel writers
requires separate classification qualification.

### Extension affected files

- `agents/plans/2026-11/loop_timing_monitor.md`
- `apps/bmcCtrl/Makefile`
- `apps/bmcCtrl/bmcCtrl.hpp`
- `apps/bmcCtrl/bmcPipeline.hpp`
- `apps/bmcCtrl/doc/sdk-worker.md`
- `apps/bmcCtrl/tests/bmcCtrl_test.cpp`
- `apps/dmSyncopation/Makefile`
- `apps/dmSyncopation/dmSyncopation.hpp`
- `apps/dmSyncopation/dmSyncopationZeroTest.hpp`
- `apps/dmSyncopation/dmtweeter_synco_production.conf`
- `apps/dmSyncopation/doc/dmSyncopation.md`
- `apps/dmSyncopation/syncoMainTiming.hpp`
- `apps/dmSyncopation/tests/dmSyncopation_test.cpp`
- `apps/dmSyncopation/tests/syncoMainTiming_test.cpp`
- `apps/loopTimingMonitor/Makefile`
- `apps/loopTimingMonitor/README.md`
- `apps/loopTimingMonitor/loopTimingMonitor.conf`
- `apps/loopTimingMonitor/loopTimingMonitor.hpp`
- `apps/loopTimingMonitor/loopTimingSynco.hpp`
- `apps/loopTimingMonitor/tests/loopTimingMonitor_subscription_test.cpp`
- `apps/loopTimingMonitor/tests/loopTimingMonitor_synco_test.cpp`
- `apps/loopTimingMonitor/tests/loopTimingMonitor_test.cpp`
- `apps/loopTimingMonitor/tests/loopTimingSynco_test.cpp`
- `libMagAOX/app/dev/dmTimingEventStream.hpp`
- `libMagAOX/app/dev/tests/dmTimingEventStream_test.cpp`
- `libMagAOX/common/dmTimingEvents.hpp`
- `tests/tests.list`
