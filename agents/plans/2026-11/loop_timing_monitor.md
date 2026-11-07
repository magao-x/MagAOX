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
   are independently configurable. Unknown/malformed/stale state closes the gate.

Use a periodic INDI get-properties refresh and configurable receipt-age timeout
because the existing wrapper does not forward source deletion/disconnection to
application callbacks. A gate-generation change clears queued/pending samples
and retained statistics; reopening requires fresh stream baselines. This avoids
mixing pre-close observations into a later operating interval.

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
| Operating gate | Configurable device subscriptions, exact FSM/switch semantics, unknown/malformed/stale state, each condition independently false, recovery, rapid Off/On, and rejection of queued samples from prior gate generations. |
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
  limits, device names, and state refresh/timeout settings.
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
- Source-state refresh defaults to one second with a five-second timeout. Unknown,
  stale or non-operating sources close the gate. Gate transitions and capture
  discontinuities reject queued old-generation samples and clear the window.
- Review fixed coherent gate/INDI publication, expiry before same-state refresh,
  actual requested scheduling-policy verification, failed ImageStreamIO attach
  cleanup, and polling-side counter-regression quality reporting.
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
