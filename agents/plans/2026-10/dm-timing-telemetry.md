# DM controller timing telemetry

Date: 2026-10-21. Status: controller telemetry implemented; local verification complete below, instrument qualification pending.

## Request and purpose

Add timing telemetry to `dev::dm`, following the integration pattern used by
`dev::frameGrabber` and `ocam2KCtrl`. Reuse `telem_fgtimings` for controller
semaphore receipt and SDK-return timing, and introduce a DM-specific type for the
other processing measurements. Obtain reliable controller statistics that can be
correlated with camera/pipeline timing to measure total loop latency.

The operator reports substantially lower jitter in analyses of recorded
`dm01disp` timestamps than in MILK viewer summaries. Preserve that distinction:
viewer refresh windows, stream publication, controller receipt and SDK return
measure different endpoints. Establish direct controller measurements before
making further dmSyncopation timing-control changes. This plan does not relax
synco bounds or change DM command handling.

Source-audit starting points: MagAOX checkout HEAD `0df1cb48`, especially
`libMagAOX/app/dev/{dm,shmimMonitor,frameGrabber,telemeter}.hpp`,
`apps/ocam2KCtrl/ocam2KCtrl.hpp` and the four controller headers below. The
ImageStreamIO/process-trace observations use the local `milk-xwcl` checkout
(HEAD `336160d0`): `src/ImageStreamIO/ImageStreamIO.{h,c}`,
`src/CommandLineInterface/processinfo/processinfo_update_output_stream.c`,
`plugins/milk-extra-src/linalgebra/MVMextractModes.c` and
`plugins/cacao-src/AOloopControl_DM/AOloopControl_DM_comb.c`. These local sources
do not establish the deployed versions or clock settings.

## 1. Define the timestamps before adding telemetry

```text
camera acquisition C                [requires verified frame/clock provenance]
       |
       v
AO processing -> dmcomb publishes dmXXdisp
       |
       v
A: first timestamp after successful semaphore wait returns
       | monitor metadata/state checks
P: dev::dm::processImage entry
       | actuator conversion, clipping, output-shape publication
S: immediately before SDK send
       | SDK call
W: immediately after SDK send returns
       | synchronous saturation-map work / commandDM epilogue
       | delta-channel calculation
D: end of delta calculation                [old piTimes endpoint]
       | saturation-worker semaphore post
F: end of measured callback work
       | fixed-size timing-event handoff and callback return
       v
next semaphore wait

physical DM response / settling: not measured by SDK-return timestamps
```

The actuator preparation/SDK/saturation ordering above describes ALPAO/BMC;
backend-specific differences are listed below. `A` is a user-space wait-return
timestamp, not the kernel scheduling instant or producer's `sem_post`. A queued
token can return immediately without blocking. Existing `m_t0` is `P`, not `A`.
Existing generic `commandDM()` return is not necessarily `W`.

Use `CLOCK_MONOTONIC` integer nanoseconds for all local markers. Subtract integer
timestamps before conversion to seconds; do not subtract floating-point epoch
times. Store external metadata timestamps separately with their clock identity.
Timestamp failure makes that measurement unavailable and must not change command
processing or its return value.

Add a minimal optional hook in `shmimMonitor`, immediately after a successful
`sem_timedwait`, following the existing optional-hook dispatch pattern in
`frameGrabber`. Apps without the hook incur no additional timestamp reads. The
hook and its lifecycle handling must distinguish:

- A real successful wait from timeouts, interrupts and discarded notifications.
- The initial `getExistingFirst` callback, which has no corresponding wait.
- Restart/reattachment epochs, so an old wake cannot pair with a new command.
- Duplicate notifications and observed input-counter jumps. These are recorded
  as quality information, not silently removed from actual callback/send cadence.

Successful waits rejected by metadata/state checks produce receipt-only events
with the rejection reason, so receipt statistics do not silently become
forwarded-callback statistics. Finalize each receipt exactly once, as either a
receipt-only event or part of its callback event, then clear the pending marker.
Apply this on metadata rejection, command failure and epoch transitions. Capture
input counter/stream identity with explicit validity; an ordinary metadata read
is not proof of atomic image/frame identity.

## 2. Reuse the existing six-field timing record

Keep the `telem_fgtimings` wire schema and camera behavior unchanged. Its fields
contain aggregate durations in **seconds**, not absolute `atime`/`wtime` values:

| Existing field | Meaning in a DM controller |
| --- | --- |
| `atime` | Mean interval between successive valid semaphore receipts, `A[i]-A[i-1]` |
| `atime_jitter` | Sample standard deviation of those intervals |
| `wtime` | Mean interval between successive successful command-submission returns, `W[i]-W[i-1]` |
| `wtime_jitter` | Sample standard deviation of those intervals |
| `wmatime` | Mean paired controller receipt-to-submission-return latency, `W[i]-A[i]` |
| `wmatime_jitter` | Sample standard deviation of that paired latency |

Use the actual `telem_fgtimings` spelling. Existing ACQ/WRT log labels remain
compatible; DM documentation and status explicitly identify their endpoint
meanings. Device identity distinguishes these records from camera records.

Receipt cadence includes rejected receipts and callbacks whose subsequent
submission fails. Successful send cadence includes a longer interval across a known failed/no-send attempt,
with the failure/gap counts reported. This describes actual successful-submission
spacing. Telemetry loss, unknown endpoint validity or a new epoch breaks interval
pairing; do not silently bridge those gaps. Receipt-to-send latency includes only
same-command, valid, successful pairs. A later delta/post failure does not erase a
successful SDK submission or its valid earlier measurements.

## 3. Add detailed DM statistics

Propose `telem_dmtimings`, using the existing FlatBuffers/logger infrastructure.
Represent each metric with a stable identifier, count, mean, sample standard
deviation, minimum and maximum. Keep all durations in seconds; formatters may
display microseconds. Include:

| Metric | Endpoints / interpretation |
| --- | --- |
| Receipt interval, send interval, receipt-to-send | Extrema/counts complement the six-field record |
| Observed publication interval | Consecutive input `writetime` differences in their own stream clock; local observation anchors define window/coverage |
| Monitor dispatch | `P-A` |
| Actuator preparation | Existing `tact1-tact0`; includes output-shape publication |
| SDK call | `W-S`, distinct from receipt-to-send latency |
| Synchronous saturation update | Existing `tact4-tact3` instantaneous saturation-map work |
| Delta update | Existing `tdeltaf-tdelta0`, only when delta work occurs |
| Saturation semaphore post | Existing `tsatf-tsat0`; excludes asynchronous worker execution |
| Processing through delta | `D-P`, preserving the exact old `piTimes` endpoints |
| Measured callback work | `F-P`, also including saturation semaphore posting |
| Receipt through callback work | `F-A`, the controller-side elapsed processing path |

`F` precedes the timing-event handoff and return; label it accordingly rather than
claiming to measure its own instrumentation or all monitor-loop overhead.
Unsupported/inapplicable phases are unavailable, not zero-duration observations.
Do not combine asynchronous saturation-thread work into a per-command span without
a separate association contract.

Record session/window sequence, actual coverage span, age of newest event,
per-metric validity/counts and cumulative plus since-last-summary quality counters.
Counters distinguish submitted, failed, partial, no-send, synthetic initial read,
duplicate/counter-gap observations, invalid timestamps and dropped timing events.
The implementation preserves the original twelve metric IDs and appends observed
publication cadence as ID 12. It records cumulative and since-summary quality
counters; individual metadata/state rejection causes
are currently combined in `rejected`.

Use a trailing **2-second elapsed-time window**, capped at **20,000 events**, as
the proposed defaults. These match the inspected camwfs deployment settings;
frameGrabber's library defaults are 5 seconds/100,000. Report the actual span if
the cap shortens coverage. Both endpoints of an interval or duration must lie
within the current time window and retained event population; no outside-window
predecessor contributes an interval. Report each metric's actual endpoint span
and count. Propose `dm.latencyTime` and `dm.latencySize`, analogous
to the frame-grabber keys; zero disables collection. Changes take effect only
through a safe reset outside command processing.

Calculate sample standard deviation with `N-1`. Interval jitter requires two
intervals, hence at least three timestamps; each phase has its own valid sample
count. Do not filter large valid durations as outliers. Explicitly distinguish
warming up, valid, stale, disabled and unavailable results. Emit the compatible
six-field record only when its populations are valid; the detailed record carries
availability during warm-up/inactivity. Forced recording must report current
availability and emit unchanged valid statistics when requested, rather than
relabeling an old window as fresh. Evict against the current monotonic time, not
the last event time, so a stopped command stream ages out of the valid window.
Maintain this consumer-side update/record/publication path in `whilePowerOff` as
well, because power management bypasses ordinary `appLogic` while power is off.

## 4. Keep collection bounded and outside the RT computation path

The command thread captures one fixed-size event containing markers, validity,
status and sequence information. Preallocate storage before monitoring starts.
No allocation, statistics, serialization, INDI calls or extra logging occurs in
the new RT capture/handoff path.

Use an ownership-safe bounded single-producer/single-consumer queue: the producer
never overwrites unread slots; a full queue drops/counts the timing event and
never delays a DM command. `appLogic` drains up to a captured producer position
with a bounded event budget, owns the elapsed-time window and computes a coherent
cached summary for INDI/telemetry. Queue capacity uses the configured event cap and has separate occupancy/drop accounting from window
retention. Stream/session resets travel with events; resizing requires quiescence.
Force-record callbacks consume the cached summary, not live producer storage.

Reuse a dependency queue if its payload ownership guarantees are suitable;
otherwise use a small focused timing collector with a tested fixed-slot handoff.
Do not copy the current frameGrabber/DM unsynchronized circular-buffer reads and
resizing. The installed atomic-index circular buffer also permits payload
overwrite, so atomic indices alone are insufficient. Keep state per instance,
including last-publication caches. A general frameGrabber collection refactor is
outside this change; a focused collector can be reused there later.

Replace the legacy `XWC_DMTIMINGS` stderr/ring implementation with this one
collector, rather than measuring twice. New timing is runtime-configured and
available without that compile flag. Preserve/document the old phase meanings.

## 5. Integrate the actual controller endpoints and telemeter

| App | Submission-return marker | Background telemetry integration |
| --- | --- | --- |
| `bmcCtrl` | Immediately after `BMCSetArray` | Add telemeter; retain later saturation work outside `W` |
| `alpaoCtrl` | Immediately after `asdkSend` | Add telemeter; retain later saturation work outside `W` |
| `irisaoCtrl` | Immediately after final `MirrorCommand(...MirrorSendSettings)` | Add telemeter; distinguish preparation/staging from final submission and qualify SDK status semantics |
| `pi335Ctrl` | Return from the last serial write belonging to this DM command | Extend existing telemeter; explicitly tag command-origin writes and partial/no-send/failure outcomes |

There are four production `dev::dm` users. Keep PI's existing `telem_pi335` and
other behavior intact. PI has no single mirror SDK call: non-OPERATING callbacks
send nothing, several axis writes make one command, and existing move helpers
can log a write failure while returning success. Capture transport status at the
write itself without changing existing control/error behavior. Manual INDI moves
must not enter semaphore-driven statistics. Serial-write completion is the
documented PI transport endpoint, not acknowledgement or motion completion.

Reset optional markers for every event. Do not substitute `commandDM()` return
for an unavailable SDK endpoint, or infer actual send success solely from a
generic callback return. Preserve valid partial measurements on all error paths.
Until a backend endpoint/status is qualified, report it unavailable explicitly.

Follow `ocam2KCtrl`'s integration pattern: use `TELEMETER_*` lifecycle/config
macros and extend each app's `checkRecordTimes`/`recordTelem` interface. Common
`dev::dm` helpers emit a detailed record and, when valid, the compatible
`telem_fgtimings` from the same cached summary. Schedule the detailed type for
forced recording and emit the compatible record alongside it conditionally;
do not independently schedule an invalid six-field record that cannot be emitted
and would be retried continuously as overdue. Emit detailed availability before
the compatible record. Preserve existing app telemetry types and their schedules.
Do not impose a mandatory new interface on unrelated monitor users.

Expose read-only `dm_timing` INDI statistics using receipt/send names: FPS,
interval min/max/jitter and receipt-to-send mean/jitter; expose detailed stage
statistics and validity/window/drop information separately. Publish only from
the ordinary app logic, following the frame-grabber update pattern. Status must
make unavailable results distinguishable from measured zeros.

## 6. Establish the camera-to-DM measurement separately and explicitly

The total loop endpoint proposed here is **the camera acquisition associated with
an AO command through that command's successful SDK return**, `W-C`. Controller
receipt-to-send and complete callback work remain separately named measurements.
SDK return does not establish physical DM response/settling.

The inspected local sources do not supply a ready-made `C` at the combined image:

- ocam2KCtrl obtains the camera acquisition timestamp from EDT; frameGrabber
  separately timestamps camera writes with `CLOCK_REALTIME`.
- The inspected MILK ImageStreamIO uses `CLOCK_ISIO` (TAI when available).
  `ImageStreamIO_UpdateIm` sets `atime` to publication time when no explicit
  acquisition timestamp is supplied.
- Both the AO producer and dmcomb publish via that helper. Thus combined-stream
  `md->atime` must not be assumed to retain camera acquisition time.
- Propagated process traces offer upstream stream/counter identities, but their
  sampled current counters/timestamps need validation against the pixels actually
  used. Auxiliary combinations may reuse an older main AO command.

Before claiming full-loop statistics:

1. Verify deployed camera timestamp clock/meaning and upstream counter/trace
   propagation, using source review and operator-supplied records.
2. Establish an unambiguous camera frame identity/acquisition time for each main
   AO command reaching the controller. Identify auxiliary-only commands that
   reuse it; report those separately rather than pairing by nearest timestamp or
   nominal FPS. Preserve discontinuities, coalescing and missing provenance.
3. Define a verified common-clock endpoint or bounded clock conversion for `C`
   and `W`. Continue local stage timing in monotonic ns; never directly subtract
   REALTIME/TAI metadata from monotonic values. Include clock/epoch validity.
4. Compute each qualified `W-C` sample before aggregating, retaining frame and
   command identity in a bounded diagnostic capture for offline cross-checks.
   Record loop-latency statistics in the detailed type only with valid provenance.

The first implementation milestone is controller telemetry. The next is a
matched-record proof and qualified full-loop statistics; the end-to-end objective
is not complete merely because controller telemetry is available. If existing
traces cannot prove frame association, bring back a concrete upstream provenance
extension for review rather than silently broadening this into AO/MILK changes.
Do not add camera and DM standard deviations or infer full-loop jitter from
unmatched summaries. No raw per-command logging is enabled by default.

## 7. Implementation sequence and acceptance

1. Finalize this endpoint/statistics contract and detailed schema. Implement the
   optional monitor hook and bounded collector with a software DM harness.
2. Add backend endpoint markers and telemeter integration for the four apps.
   Replace legacy DM timing collection. Add schema/type/accessors, event code,
   library include dependency, verification dispatch and telemetry recording.
3. Add the read-only INDI interface and configuration. Update this engineering
   record with actual decisions, then add user documentation separately.
4. Validate on the workstation with deterministic clock/SDK stubs and isolated
   local streams. Build the four controllers where their SDKs are available;
   report unavailable SDK/build coverage explicitly. Build logger consumers.
5. Have the operator compare telemetry with recorded `dmXXdisp` timestamps and
   independently captured command endpoints, first without auxiliary commands,
   then at intended rates. Compare timing enabled/disabled to quantify overhead.
   Keep instrument acceptance thresholds explicit before qualification; do not
   turn observed means/maxima into hard synco safety bounds automatically.
6. Use qualified controller measurements for their specific contribution to
   synco fault analysis. Complete the matched-frame/clock proof above and review
   any required upstream extension before implementing or claiming full-loop
   latency; that proof need not block analysis of the controller endpoints.

Focused tests must cover exact sample statistics/units/extrema, startup and
minimum counts, elapsed-window boundaries and capacity, independent instances,
forced unchanged records, staleness, optional phases, failed/no-send/partial SDK
paths, successful send followed by delta/post failure, initial no-wait callbacks,
timeouts/restarts, counter discontinuities, queue overflow and producer/consumer
ownership. Exercise the real monitor hook with a local-stream fixture and apps
without the optional hook. Verify FlatBuffer validation/round trips, metadata
accessors and generated logger registration. Measure capture overhead separately
from instrument qualification; workstation results cannot qualify instrument
timing. Frame/clock correlation tests must reject ambiguous and stale provenance.

Use the existing logger generation rule; do not hand-edit or commit generated
FlatBuffer/dispatch products. Keep the existing frameGrabber schema/readers and
PI telemetry compatible. Existing generic telemeter/frameGrabber defects found
during review are separate follow-ups, not an unreviewed base-class rewrite.

Functional changes and their engineering record belong in clean feature-branch
commits, followed by documentation-only changes and separate formatting cleanup
where needed. Follow the full-file documentation and test-grouping rules in
`AGENTS.md`. No instrument commands are to be executed by an agent.

## Affected files

- `AGENTS.md` and `.clang-format`: explicit control-flow braces, Allman layout,
  multiline lambdas and blank lines between logical code groups.
- `libMagAOX/app/dev/shmimMonitor.hpp`: optional receipt hook/lifecycle support.
- `libMagAOX/app/dev/dm.hpp`, `libMagAOX/app/dev/dmTiming.hpp` and
  `libMagAOX/common/dmTiming.hpp`: shared timing types, safe handoff, statistics,
  INDI and recording helpers.
- The four `apps/{bmcCtrl,alpaoCtrl,irisaoCtrl,pi335Ctrl}/*Ctrl.hpp` files and
  relevant Makefiles: endpoint markers, telemeter integration and legacy flag.
- New `libMagAOX/logger/types/telem_dmtimings.hpp` and matching `.fbs`;
  `logger/logCodes.dat`, `logger/logMeta.cpp`, `libMagAOX/Makefile`, and a single
  `lastRecord` definition if not inline: registration/verification/dependencies.
- Three new generic DM/monitor/collector tests, four controller tests and three
  test-only SDK declaration headers, plus the detailed logger tests.
- `tests/Makefile.one`, `tests/tests.list` and `tests/groups.dox`: central build,
  SDK/FlatBuffers dependencies and test registration/documentation.
- `libMagAOX/app/dev/doc/dmTiming.md`, this plan and the separate
  [synco debugging record](dm-syncopation-debugging.md).

## Implementation and verification record

The user approved implementation after clarifying that adding `dev::telemeter`
means background telemetry integration; the command thread only captures
markers/status/counters and hands off a fixed-size event. The implementation is
runtime-configured independently of the removed legacy `XWC_DMTIMINGS` buffers,
wall-clock doubles and stderr reports.

Implemented the optional monitor observation hook, ownership-safe preallocated
SPSC collector and consumer-owned elapsed-time statistics, all four controller
adapters, read-only INDI and compatible/detailed telemetry. New event code
`TELEM_DMTIMINGS=20892` initially recorded twelve metrics and collection quality;
the publication-cadence follow-up below appends a thirteenth metric.
`telem_fgtimings` and camera readers/schema remain unchanged. Detailed records
precede compatible records; only the detailed type is scheduled independently for
forced reports. Software owners without a telemetry helper retain a no-op
recording fallback.

Backend qualifications are deliberate: BMC/ALPAO use actual SDK result codes;
PI uses actual results of command-origin serial writes while preserving historical
returns, and its SDK metric spans the grouped transaction. IrisAO's final SDK call
is timed but successful submission is unqualified until vendor status semantics
are verified. Its successful-return cadence and paired latency remain unavailable.
Generic callback errors preserve already-qualified partial outcomes, and a
successful submission survives subsequent delta/post failure.

Review caught and corrected power-off staleness: ordinary `whilePowerOff` now
maintains collection/statistics/recording/INDI even when the app main loop skips
`appLogic`. A software regression exercises that route. A fake INDI driver checks
actual warm-up, valid, stale and disabled values without FIFOs or networking.

Local checks completed:

- Collector: **131 assertions / 7 cases**, including 100,000 concurrent events;
  ThreadSanitizer passes without a race report.
- Actual monitor-thread/local-stream integration: **42 assertions / 4 cases**,
  including initial read, real wait, discarded receipt, restart, timeout and
  genuinely absent optional hook.
- DM callback/telemetry/configuration/INDI/power-off integration: **96 assertions /
  6 cases**; hot-path logging is absent, forced records are coherent and successful
  SDK timing survives a later notification failure.
- Four controller suites: **127 assertions / 7 cases** (PI 67/4, BMC 18/1,
  ALPAO 18/1, IrisAO 24/1). Test SDK/transport stubs perform no device access.
- Detailed telemetry binary/JSON verification, generated dispatch and scalar
  accessors: **807 assertions / 4 cases**. Tests include stable metric IDs,
  nonfinite/corrupt packets, full-width counters and invalid availability.
- Existing generic DM regressions: **274 assertions / 5 cases**; dmSyncopation
  regressions: **2349 assertions / 25 cases**.
- Normal library, PI controller, dmSyncopation and logdump builds pass. BMC/ALPAO/IrisAO vendor
  SDKs are unavailable on this workstation; their app headers are compiled through
  software SDK stubs, which do not qualify vendor ABI or instrument behavior.

A workstation-only capture benchmark (i9-12900HK, g++15.2, `-O3
-ffast-math -march=native`, ordinary scheduling with no affinity changes) used
101 batches of 4096 complete events after five warm-up batches. Each enabled
event included all thirteen endpoint clock reads and the fixed-size handoff;
consumer draining/statistics were outside the timed interval. Enabled capture
averaged **193.120 ns/event** (median batch 193.474 ns/event); disabled capture
averaged **1.205 ns/event** at the same non-inlined call boundary. There were
434,176 enabled events, **zero drops** and **zero `new`/`new[]` allocations**
inside either capture loop. Each event occupies 144 bytes. Source/binary were
kept under `/tmp/dmTiming_benchmark{.cpp,}`. These are local batch averages,
not an instrument timing bound or tail-latency qualification.

The [user guide](../../../libMagAOX/app/dev/doc/dmTiming.md) documents exact
configuration, endpoint diagram, INDI names/units, statistics and backend limits.
All new app tests use the central test build and application Doxygen groups.
Full-file documentation checks and clang-format pass for all 21 touched C++
files; Markdown links/fences and `git diff --check` pass.
The focused feature suites total **1203 assertions / 28 cases**; including generic
DM and synco regressions, **3826 assertions / 58 cases** pass.
Generated logger/FlatBuffers outputs remain uncommitted build products.

Instrument acceptance/overhead comparison and verified frame/clock association
remain operator-run follow-ups. No camera-to-DM latency metric or raw timestamp
stream has been fabricated from combined-image metadata. Any upstream provenance
extension remains a separate concrete review item, as agreed in section 6.
No instrument commands, deployment or instrument configuration edits were
performed during implementation. Subsequent local commits are recorded below.


### Formatting review follow-up

The user requested explicit control-flow braces and blank lines between logical
code groups, using `dm::processImage` as an example. Applied that style across
the timing work, recorded the standing preference in `AGENTS.md` and enabled
`InsertBraces` plus multiline lambda bodies in `.clang-format`. Validation
against saved pre-cleanup copies confirms all non-brace C++ tokens and
preprocessor directives are unchanged across all 21 C++ files. The formatter and
whitespace checks pass. This follow-up changes style only; the implementation
validation results above remain the existing test/build checkpoint. No tests
were added or rerun. The cleanup is kept in a separate final formatting commit.


### 2026-10-21 local commit checkpoint

The user authorized local commits before operator-run hardware testing. Functional
implementation and this engineering record are committed as `09028bb2`
(`Add DM controller timing telemetry`). User documentation, accumulated debugging
evidence, test groups and the standing style preference are kept in a separate
`Document DM timing telemetry and validation` commit. The reviewed braces/spacing
and formatter configuration are kept in the final `Apply explicit braces and
logical spacing` commit. Each commit carries the requested GPT-6 attribution.

The validated implementation is unchanged by commit preparation. The formatting
pass retains all non-brace C++ tokens and preprocessor directives; the existing
3826-assertion/58-case test checkpoint and qualified build/SDK limitations above
remain applicable. No new tests or builds were needed for committing. Hardware
qualification is pending operator testing. No push, deployment or agent instrument
action is part of this commit request.

### 2026-10-21 observed input-publication cadence

The operator's first installed run reports irregular receipt/SDK-return intervals
despite a continuously running 2 kHz AO loop. The pasted detailed record has
2,741 receipt intervals across 1.99590386 seconds, mean 0.0007281663116 seconds,
with extrema 0.000124295 and 0.02520309 seconds. Its receipt-to-SDK latency is
approximately 108.448 us and SDK duration approximately 101.269 us. The compatible
record from the preceding second represents a different rolling window; its
approximately 1.384 ms receipt mean does not establish a field-mapping error.
The operator requested recording the input `dmXXdisp` publication timestamp.

The callback now copies `writetime` alongside its existing `cnt0` observation,
validates timespec components/overflow and retains it in the fixed-size event.
No additional clock calls, allocation, locking or logging are added to the
producer. `publicationInterval` is appended as stable metric ID 12 and reports
count, mean, sample jitter, extrema and coverage through detailed telemetry,
LogMeta and six new `dm_timing_stages.publicationInterval_*` elements. The
compatible `fgtimings` record and its validity criteria remain unchanged.

Only adjacent retained events with consecutive valid counters and positive,
strictly increasing publication timestamps contribute a pair. Missing metadata,
duplicate/skipped/reset counters, queue loss and attachment changes break pairing.
External timestamp differences produce the values; local receipt observations
(entry when a receipt timestamp is unavailable) determine window membership and coverage. No
stream-clock timestamp is subtracted from a monotonic marker. Malformed external
timestamps do not increment the counter for invalid local clock captures.

Source inspection of the local MILK checkout establishes the clock and limits:
`src/ImageStreamIO/ImageStreamIO.h` selects `CLOCK_TAI` where available, otherwise
`CLOCK_REALTIME`; `ImageStreamIO_UpdateIm_atime` in
`src/ImageStreamIO/ImageStreamIO.c` writes the timestamp, increments `cnt0`, clears
`write`, then posts. The dmcomb code in
`plugins/cacao-src/AOloopControl_DM/AOloopControl_DM_comb.c` does not raise `write`
before updating the combined stream, and delegates publication through
`src/CommandLineInterface/processinfo/processinfo_update_output_stream.c`.
Consequently the existing write/counter bracket is best-effort observation, not
proof of token/pixel/timestamp association: a new timestamp with the prior
counter can be visible briefly. Positive stream-clock steps can also resemble
real stalls. No MILK source or publication behavior is changed here.

The rebuilt detailed reader accepts complete original twelve-row records and
new thirteen-row records, preserving old metric IDs. Missing publication data in
old records projects as unavailable. Older readers that enforce exactly twelve
rows need rebuilding for the new records. This is diagnostic publication cadence,
not publication-to-receipt latency or verified total-loop timing.

Scoped workstation checks pass:

- Collector: **158 assertions / 10 cases**, including the existing 100,000-event
  concurrent ownership/loss test and the new publication precision, association,
  retention and reset cases.
- DM callback/configuration/telemetry/INDI/power-off integration: **153 assertions
  / 8 cases**, including real metadata-copy/validation, publication versus receipt
  cadence, new property elements and unchanged compatible-field meanings.
- Detailed logger: **893 assertions / 4 cases**, including binary/JSON roundtrip,
  all 91 metric accessors, malformed metric sets and complete old twelve-row
  record compatibility. Total scoped checks: **1,204 assertions / 22 cases**.
- Normal library/schema generation, PI-335 and logdump builds pass. PI reports
  nine existing `convertFromString` deprecation warnings; schema generation
  reports existing field-name style warnings. Focused test builds are warning-free.
- All seven touched C++ files pass clang-format and whitespace checks. This
  follow-up does not rerun the earlier vendor SDK-stub, monitor or synco suites,
  ThreadSanitizer or capture benchmark. The original 144-byte/0.193 us capture
  benchmark predates the added metadata read and publication event field and
  does not qualify this version's overhead.

No commit, push, deployment, instrument command or agent hardware action was
performed. Comparing the new publication metric with receipt/send cadence on the
instrument remains an operator follow-up.

### 2026-10-21 operator workload comparison

In the Python-workload pause comparison, 19 consecutive `dmtweeter` compatible
records recover receipt/send means of approximately 500 us, with interval jitter
approximately 7–8 us and receipt-to-SDK latency approximately 108.53 us with
0.43–0.47 us jitter. The [debugging record](dm-syncopation-debugging.md) preserves
the source, timestamp range, exact window-statistic ranges and interpretation.
This strongly supports workload-associated timing disruption; it does not yet
localize the interference or independently qualify full-loop timing. The
publication-metric comparison remains pending. No implementation or validation
checkpoint changes accompany this operator evidence.

### 2026-10-21 operator camera IRQ-pinning comparison

After camera IRQ pinning, the first 15 compatible records show receipt jitter of
5.87–6.80 us, but five detailed windows approximately two minutes later return to
7.31–7.82 us. A sustained improvement is not established. Detailed receipt
intervals span 454.428–572.382 us while receipt-to-SDK and SDK durations remain
consistent. Event counts support the expected 2 kHz cadence and the reported
outcome/drop counters are zero in the new summary deltas. The
[debugging record](dm-syncopation-debugging.md) preserves exact values and count,
window and metadata qualifications. These records still have only twelve metrics;
the new `publicationInterval` build has not been demonstrated by this sample.
No implementation or validation checkpoint changes accompany this evidence.

### 2026-10-21 operator publication-metric validation sample

The operator's new detailed DM records include `publicationInterval`. Over nine
windows, publication/receipt populations match at 3,998–3,999 intervals, with
publication jitter 5.735–6.695 us and receipt jitter 5.758–6.714 us and closely
matching extrema. This supports timing variation already present at combined
publication. Camera acquisition/write jitter is lower in the supplied earlier
windows, but the camera and DM populations are not simultaneous. The
[debugging record](dm-syncopation-debugging.md) preserves exact sources, ranges,
quality counts and qualifications. These aggregates cannot establish
publication-to-receipt delay jitter or total camera-to-DM latency; external-clock
and metadata association limits remain in force. The useful next comparison is
contemporaneous primary (`dm01disp03` / `aol1_dmC`) and combined (`dm01disp`)
publication timing. No implementation, test/build checkpoint or timing-control
setting changes accompany this operator evidence.

### 2026-10-21 primary observer implemented in dmSyncopation

The requested upstream comparison is now supported locally by reusing
`dev::dmTiming` in dmSyncopation for primary publication and semaphore receipts.
The [synco engineering record](dm-syncopation.md) and
[guide](../../../apps/dmSyncopation/doc/dmSyncopation.md) define the independent
configuration, INDI and optional primary table in existing synco telemetry. It
collects while scheduling is disabled and through resource/control failures;
primary validity does not depend on unavailable SDK populations. Ordinary-thread
recording also continues when the scheduling mutex is busy, with explicit
schedule-snapshot validity. Scoped checks pass **2,625 assertions / 34 cases**
and normal library/app/logdump builds. The DM controller collector/schema and
its existing statistics are unchanged. An operator comparison over overlapping
primary/combined windows remains pending.
