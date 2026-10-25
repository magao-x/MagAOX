# Frame-grabber timing diagnostics

`dev::frameGrabber` transfers paired camera timestamps and optional stage
observations through fixed-capacity queues. The acquisition worker copies one
fixed-size event; ordinary app logic owns the history, statistics, INDI updates
and telemetry. A full queue drops a diagnostic event and records the loss. The
worker does not wait for the statistics reader or allocate timing storage.

The existing `fg_timing` property and `[fgtimes]` telemetry retain their endpoint
meanings. Published-frame intervals include elapsed time across rejected
attempts or skipped camera frame numbers. Stricter continuity checks apply to
stage attribution, where a gap must break a frame-to-frame comparison.

## Configuration

All settings belong to `[framegrabber]` and take effect after an app restart.

| Setting | Default | Meaning |
| --- | --- | --- |
| `latencyTime` | `5` | Trailing statistics window, seconds; zero disables timing collection. |
| `latencySize` | `100000` | Maximum events in each queue and retained history; zero disables collection. |
| `stageTiming` | `false` | Enable additional local stage markers and completed serial-operation observations. |
| `stageExcursionUs` | `20` | Distance from the retained monotonic publication-interval median used to classify excursions, microseconds; zero disables classification. |
| `stageBracketUs` | `0` | Maximum accepted write-clock bracket for conditional attribution, microseconds. Zero leaves that bound undecided and all stage intervals ambiguous. |
| `stageTraceMax` | `0` | Maximum ready plus pending individual excursion contexts, 0–256. Zero disables individual records; requires stage timing, an excursion threshold of at least 0.001 microseconds and at least four timing slots. |

For the existing two-second, 20000-event configuration, the only added setting
needed to collect raw stages is:

```ini
[framegrabber]
stageTiming=true
```

A positive `stageBracketUs` must be selected through timing qualification before
conditional attribution is enabled. `stageExcursionUs` classifies measurements;
it is independent of scheduler safety margins and does not alter acquisition
or publication timing.

Capacity is the full configured `latencySize`, independently of FPS. Acquisition
and serialized-control queues each have that capacity; each retained history is
also bounded by elapsed time and capacity. At 2 kHz a two-second window normally
contains about 4000 frames. Capacity pressure or a stalled consumer can shorten
coverage, so inspect counts, span, age and loss counters together. Storage is
allocated before acquisition starts; FPS changes advance the timing session
without reallocating timing queues in the worker.

With `stageTiming=false`, paired collection can remain enabled. Additional
stage-marker clock reads and the control-event handoff are disabled.

## Endpoints and clocks

For OCAM with multiple queued DMA buffers, rearm runs at the end of the
existing post-publication hook. The image and sync stream are both posted
before that SDK call. Startup queues the configured number of acquisitions;
rearm replenishes one request after each consumed frame. The current
64-buffer configuration therefore has further captures queued while the
completed frame is processed. This changes the placement of queue replenishment,
with timeout and frame validation still required before publication.

The optional stage markers describe that order:

```text
waitEnter       enter acquisition attempt / SDK wait
    |
waitReturn      SDK wait returns
    |
timeoutEnter    enter SDK timeout-counter query
    |
timeoutReturn   timeout-counter query returns
    |
validated       camera result/frame validation completes
    |
loadDone        descramble and optional binning complete
    |           existing CLOCK_REALTIME writetime call
writeStampDone  immediately after that call
    |           metadata completion and main-stream semaphore broadcast
mainPostDone
    |           OCAM hook posts the sync stream
rearmEnter      enter SDK queue replenishment
    |
rearmReturn     next EDT acquisition has been queued
    |
syncPostDone    post-publication hook returns
    |           bookkeeping, pointer preparation and FPS observation
loopTailDone
    |           event handoff and next iteration
next waitEnter
```

Other EDT controllers retain their existing ordering. OCAM also retains
immediate rearm for a single DMA buffer or a mode that forces single-frame or
serial-triggered acquisition:

```text
waitReturn -> rearmEnter -> rearmReturn -> timeoutEnter -> timeoutReturn -> validated
```

Deferred rearm remains on the acquisition thread. Failed/rejected frames
request recovery without leaving a queue-replenishment obligation for a later
unrelated frame. The finite acquisition queue must still be replenished during
normal operation. No additional configuration setting is required.

For controller implementations, `pdvAcquire(timestamp)` retains immediate
rearm. Passing `true` as its second argument opts into eligible deferral and
requires `pdvFinishAcquisition(true)` after publication or
`pdvFinishAcquisition(false)` when abandoning the attempt. The finish call is
harmless when no replenishment is pending. OCAM currently supplies this
completion handling; existing EDT controllers need no change.

Local markers use monotonic integer timestamps. The SDK `atime` and existing
realtime `writetime` are retained separately and unchanged. Their interval
populations retain their respective clocks. `acquisitionToWrite` preserves the
legacy `writetime - atime` context, including signed values; this does not verify
the SDK timestamp's event or clock provenance or establish absolute camera
latency. The camera app compiles and links against the copied EDT PDV
**5.6.8.0** SDK/library; identifying the instrument's loaded library and driver
remains necessary. The copied SDK decoder
specifies seconds/nanoseconds for its default `EDT_TM_SEC_NSEC` mode, and this
app does not change that mode. Some SDK comments still say microseconds; the
actual default decoder and header are used for this contract. DMA-completion
system time includes driver/interrupt latency; it is not an exposure timestamp.

`loadDone` and `writeStampDone` bracket the existing realtime timestamp call.
The `writeBracket` width measures its uncertainty relative to the monotonic
proxy endpoint. Both neighboring brackets must fit a configured positive limit
for conditional attribution. Wide or incomplete brackets remain in raw timing
populations and increment the ambiguous interval population.

The current frame's main/sync posts, deferred rearm and later bookkeeping occur
after its write timestamp. They can delay the next SDK wait, but cannot delay
that already sampled timestamp. `previousTail` measures the preceding write
endpoint to the current wait entry, including work between iterations.
The timeout-counter query remains before publication because partial transfers
must be rejected. Its independent duration shows whether driver contention
also affects that call after rearm has moved.

## Readbacks and telemetry

Durations, means, standard deviations, extrema, coverage and limits in INDI and
telemetry are **seconds**. FPS values are hertz. Configuration names ending in
`Us` are the microsecond inputs described above.

| Property | Contents |
| --- | --- |
| `fg_timing` | Existing acquisition/write FPS, interval extrema/jitter, and `delta_aw` statistics. |
| `fg_stage_timing` | Raw metrics, `lateChange_`/`earlyChange_` metrics, completed `serial` durations and the six matched interval/tail populations below. Each has `_count`, `_valid`, `_mean`, `_stddev`, `_min`, `_max`. |
| `fg_timing_quality` | Enable/availability state, session and sequence, window/span/age, queue occupancy/capacity, median/threshold/bracket limit, excursion populations, frame/mode provenance and quality counters. |

A population's `_count` determines whether it has observations; `_valid` means
sample variance is available, requiring at least two samples. Unpopulated INDI
numeric fields can be zero, so consult count and validity. The quality `valid`
field refers to the three compatible frame-grabber populations, not every
optional stage. Availability codes are `0=disabled`, `1=warmingUp`, `2=valid`,
`3=stale`, `4=unavailable`.

Raw metric names are:

| Metric | Endpoints |
| --- | --- |
| `acquisitionInterval`, `publicationInterval` | Successive published SDK `atime` or realtime `writetime` values. |
| `acquisitionToWrite` | Same-frame realtime write stamp minus SDK acquisition stamp; external-clock context. |
| `waitReturnInterval`, `writeEndpointInterval` | Successive continuous SDK-return or monotonic write-endpoint markers. |
| `wait` | Wait entry to SDK return; a normal frame-period wait is expected. |
| `rearm` | Explicit rearm entry to return, at either supported location. Older events without an entry marker use SDK wait return to rearm return. |
| `timeoutQuery` | Explicit timeout-counter query entry to return; unavailable in older events. |
| `validation` | Timeout query return to validated result, or rearm return to validated result in older events. |
| `conversion` | Validated result to loaded image. |
| `writeBracket` | Loaded image to completion of the realtime write-clock call. |
| `mainPost` | Write endpoint to main-stream post completion. |
| `syncPost` | Main post to rearm entry when rearm is deferred inside the publication hook; otherwise main post to hook completion. This excludes deferred rearm. |
| `loopTail` | Publication-hook completion to loop-tail marker. |
| `waitToWrite` | SDK return to monotonic write endpoint. |
| `previousTail` | Previous frame's write endpoint to this frame's wait entry. |
| `serialOverlap` | Duration of completed observed serial transactions intersecting this frame's wait-entry to write-endpoint span. |

`telem_fgstage` is event **20906**, printed as `[fgstage]`. OCAM records it from
ordinary app logic while stage collection is enabled; the telemeter can also
force a record, including a disabled or unavailable snapshot. Rebuild/install
both the camera application and `logdump` to collect and read the new type.
The old `[fgtimes]` schema is unchanged.

The updated reader accepts existing 16-metric stage records and nine-marker
chronological records. New records contain 17 metrics and 12 endpoint slots,
appending `timeoutQuery` and the three explicit rearm/timeout endpoints while
preserving existing numeric identifiers. Raw
`syncPostDone` is the publication-hook return, so it follows deferred rearm;
the `syncPost` duration ends at `rearmEnter` in that case. Rebuild the camera
and `logdump` to produce and interpret the extended records.
If a new endpoint capture fails, its affected duration remains unavailable;
the reader retains the unqualified raw neighborhood and its invalid-marker
count. It does not reinterpret missing modern endpoints as a legacy segment.

Binary/JSON records retain each population's count, validity and local coverage
span. Metadata projections use the same raw metric names and
`lateChange_<metric>_<field>` / `earlyChange_<metric>_<field>`. The completed
serial-duration label in text and logMeta is `serialDuration`, the binary/JSON
field is `serial_duration`, and INDI uses `serial`. Missing duration projections
are NaN; a single sample's mean
is available while its standard deviation is unavailable.

## Reading excursion attribution

Let `R_i` be SDK return and `W_i` the monotonic write endpoint. For the same
adjacent frame pair:

```text
W_i - W_(i-1) = R_i - R_(i-1)
                 + [(W_i - R_i) - (W_(i-1) - R_(i-1))]
```

The interval median includes ordinary tails. Late/early classes lie strictly
above/below that median plus/minus the configured excursion threshold. Only
continuous successful stage observations with acceptable neighboring clock
brackets enter conditional attribution. `intervals = qualified + ambiguous`;
`late` and `early` are subsets of `qualified`.

Use `lateWriteEndpointInterval`, `lateWaitReturnInterval` and
`lateChange_waitToWrite` for the late cohort, and their `early` equivalents for
the early cohort. Changes are signed current-minus-previous durations. Complete
common marker coverage and matching cohort counts are required before applying
the identity to their means. Missing markers can make individual populations
smaller; subtracting independently measured RMS values cannot locate a delay.
`latePreviousTail` and `earlyPreviousTail` preserve preceding-tail context for
those same output excursion classes.

Inspect `total_` and `delta_` quality counters for acquisition outcomes,
duplicates, output/frame gaps, invalid captures, wide brackets, dropped events
and completed/failed/invalid/dropped control observations. Totals cover the
configuration lifetime; deltas cover the preceding consumer update. Rolling
windows overlap, so do not sum their population counts as independent samples.

Counter names are `events`, `published`, `rejected`, `failed`, `duplicates`,
`counterGaps`, `frameGaps`, `invalid`, `bracketWide`, `dropped`, `controls`,
`controlFailed`, `controlInvalid` and `controlDropped`, each exposed with the
`total_` and `delta_` prefixes. `published` counts main-stream outputs even if
a later sync hook fails, so it can overlap `failed`. Gaps count nonunit observed
transitions, rather than an inferred number of missing frames. Dropped diagnostic
events limit outcome accounting; they do not themselves drop camera frames.

Serial records use a separate producer channel, serialized by the existing
control path. They introduce no acquisition-worker mutex. Only completed
observed transactions contribute; missing or dropped control events limit
coverage. Temporal overlap identifies an association to investigate, not proof
that a serial transaction blocked acquisition or caused an excursion.

## Individual excursion records

To retain chronological examples for the current two-second/20000-slot setup:

```ini
[framegrabber]
stageTiming=true
stageBracketUs=1
stageTraceMax=16
```

These settings require an application restart. The one-microsecond bracket is
a provisional diagnostic accuracy filter, not a scheduler reserve or an
instrument overhead limit. Leave the camera mode, cadence, affinity and polling
unchanged for the comparison.

The collector uses the existing fixed producer events. Individual selection,
history, median calculation, formatting and logging run on the ordinary
application thread. Enabling individual records adds no acquisition-thread
clock reads, locks, allocations or extra event handoff. It adds ordinary-thread
work and telemetry traffic whose instrument effect still needs comparison.

Each `[fgtrace]` record has five fixed chronological positions: two preceding
attempts, the trigger, and two following attempts. A presence mask and completion
reason distinguish complete neighborhoods from missing neighbors at a stream,
session, queue-gap or stopped-acquisition boundary. Overlapping neighborhoods
can repeat the same frame; deduplicate by session, stream identity and attempt
sequence before counting events. A unique excursion ID identifies each trigger.

Selection uses a rolling median of at least three strictly preceding continuous
intervals, with its exact integer-nanosecond median and population stored in
the record. This causal baseline differs from a `[fgstage]` summary median that
includes its whole retained window. Warmup and continuity resets are counted.
Raw early/late excursions are retained even with a wide or undecided clock
bracket, with their attribution qualification reported separately. Selection
examines newly consumed events before summary-window expiry, including events
between the periodically emitted summary windows. Handoff drops, trace-capacity
losses and incomplete contexts remain explicit limits on coverage.

`telem_fgtrace` is event **20907**. Its binary/JSON payload and text retain exact
integer `CLOCK_MONOTONIC` nanoseconds for all available stage markers, marker
validity, attempt/output/raw-frame identities, mode, outcomes and the original
SDK/realtime timestamp components. They also include the process PID,
acquisition TID, consumer evaluation and batch-emission times, and completed-control
overlap known at finalization. Completed-control coverage does not include
in-flight transactions and cannot prove that zero overlap means no concurrent
serial activity.

Ready plus pending records are bounded by `stageTraceMax`. On saturation, the
newest trigger is dropped and counted; acquisition continues. Following frames
can arrive during the next ordinary update. Complete or partial records are
emitted once when ready, without a periodic forced replay. Ordinary telemetry
emission clears the ready queue. The `fg_timing_quality` property adds:

- `traceEnabled`, `traceCapacity`, `traceReady`, `tracePending`.
- Cumulative `traceDetected`, `traceRecorded`, `traceDropped`, `traceIncomplete`,
  `traceUnclassified` and `traceDiscontinuities`.
- Cumulative source losses `traceEventDropped`, `traceControlDropped`, and
  `traceLogFailed` for an unavailable emission clock or errors reported by the
  telemetry wrapper. These counters
  are not a durable logger-write acknowledgement.

Rebuild/install the camera app and `logdump`. For the operator-run monotonic
scheduler/IRQ capture and the files to retain, follow
[camera-timing-trace.md](../../../../apps/ocam2KCtrl/doc/camera-timing-trace.md).
The capture runs in a separate tracefs instance and does not change camera
configuration, worker affinity or scheduling policy. Only the operator runs
it on the instrument.

## Qualification

Workstation tests exercise ownership, continuity, classification, serialization
and failure paths. Instrument timing overhead, the accepted bracket width and
installed EDT timestamp semantics still require operator qualification. Compare
paired-only and stage-enabled runs with the same camera mode, workload,
affinity and IRQ placement, checking coverage and losses alongside timing.

The failed-result guard is active in both modes and adds one EDT timeout-counter
query per non-null wait (plus a start baseline). With stage timing enabled,
`timeoutQuery` measures its cost independently before validation and before
deferred rearm. Earlier records included this query in `validation` after
immediate rearm. Compare optional-clock overhead at fixed rearm placement;
compare binaries separately to measure the effect of changed SDK-call ordering.

The copied 5.6.8.0 SDK also documents keeping DMA/open/close operations in one
thread. The existing initial setup, acquisition and final close cross threads.
Moving the complete EDT resource lifecycle to its owner worker is a separate
follow-up requiring its own controlled comparison; instrumentation does not
qualify that existing topology.
