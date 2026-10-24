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

The OCAM/EDT path records these boundaries without moving the existing realtime
write timestamp or stream notifications:

```text
waitEnter       enter acquisition attempt / SDK wait
    |
waitReturn      SDK wait returns
    |
rearmReturn     next EDT acquisition has been queued
    |
validated       camera result/frame validation completes
    |
loadDone        descramble and optional binning complete
    |           existing CLOCK_REALTIME writetime call
writeStampDone  immediately after that call
    |           metadata completion and main-stream semaphore broadcast
mainPostDone
    |           post-publication hook, including the OCAM sync stream
syncPostDone
    |           bookkeeping, pointer preparation and FPS observation
loopTailDone
    |           event handoff and next iteration
next waitEnter
```

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

The current frame's main/sync posts and later bookkeeping occur after its write
timestamp. They can delay the next SDK wait, but cannot delay that already
sampled timestamp. `previousTail` measures the preceding write endpoint to the
current wait entry, including work between iterations.

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
| `rearm`, `validation`, `conversion` | SDK return to rearm, rearm to validated result, and validation to loaded image. |
| `writeBracket` | Loaded image to completion of the realtime write-clock call. |
| `mainPost`, `syncPost`, `loopTail` | Write endpoint to main post, main post to sync-hook completion, and sync completion to loop-tail marker. |
| `waitToWrite` | SDK return to monotonic write endpoint. |
| `previousTail` | Previous frame's write endpoint to this frame's wait entry. |
| `serialOverlap` | Duration of completed observed serial transactions intersecting this frame's wait-entry to write-endpoint span. |

`telem_fgstage` is event **20906**, printed as `[fgstage]`. OCAM records it from
ordinary app logic while stage collection is enabled; the telemeter can also
force a record, including a disabled or unavailable snapshot. Rebuild/install
both the camera application and `logdump` to collect and read the new type.
The old `[fgtimes]` schema is unchanged.

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

## Qualification

Workstation tests exercise ownership, continuity, classification, serialization
and failure paths. Instrument timing overhead, the accepted bracket width and
installed EDT timestamp semantics still require operator qualification. Compare
paired-only and stage-enabled runs with the same camera mode, workload,
affinity and IRQ placement, checking coverage and losses alongside timing.

The failed-result guard is active in both modes and adds one EDT timeout-counter
query per non-null wait (plus a start baseline). Its cost is included after
`rearmReturn`, in `validation`. Off/on comparison isolates the optional clocks
and stage collection; comparison with the previous binary is also needed to
measure the result-guard and repaired paired-handoff baseline cost.

The copied 5.6.8.0 SDK also documents keeping DMA/open/close operations in one
thread. The existing initial setup, acquisition and final close cross threads.
Moving the complete EDT resource lifecycle to its owner worker is a separate
follow-up requiring its own controlled comparison; instrumentation does not
qualify that existing topology.
