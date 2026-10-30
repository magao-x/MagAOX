# DM controller timing telemetry

`dev::dm` reports observed input-publication cadence, controller receipt cadence,
successful submission-return cadence and local processing times for BMC, ALPAO,
IrisAO and PI-335 controllers. All local
markers use integer `CLOCK_MONOTONIC` nanoseconds; reported durations and jitter
use **seconds**, and FPS uses **Hz**. SDK or serial-write return is a software
endpoint; physical DM response and settling require separate measurements.
Publication intervals use integer differences of input-stream `writetime` values
in the stream's own clock, which can be TAI or REALTIME. No external timestamp is
subtracted from a local monotonic timestamp.

## Endpoints and metrics

```text
U  dmcomb writes input stream writetime and publishes combined command
   -> successful controller semaphore wait
  A  first timestamp after wait return
     monitor metadata/state checks
  P  dev::dm::processImage entry
     observe input cnt0 and writetime with metadata consistency checks
     actuator preparation [actuatorStart -> actuatorEnd]
  S  backend submission begins
     SDK call / grouped PI serial writes
  W  backend submission returns
     synchronous saturation-map work [satStart -> satEnd]
     delta work [deltaStart -> D], when configured
  D  processing-through-delta endpoint
     saturation-worker semaphore post [postStart -> postEnd]
  F  measured callback work ends
     timing-event handoff, callback return, then next wait
```

This ordering describes synchronous BMC/ALPAO. `A` follows the user-space wait return; queued
tokens can return immediately. `F` precedes timing-event handoff and final return.
The saturation worker runs asynchronously and has no per-command completion
measurement here. An initial existing-image callback has no `A`.

An optional [reader-completion publisher](dmCompletion.md) signals after the
timing-event handoff, when the callback has finished all command-channel reads.
It has its own stream and semaphores and remains independent of timing-event
loss or disabled collection. It does not change these telemetry endpoints.

With the optional [BMC SDK worker](../../../../apps/bmcCtrl/doc/sdk-worker.md),
SDK execution overlaps private saturation/delta preparation. The monitor
publishes a distinct last-read endpoint, waits for the matching SDK result,
then publishes successful products and full completion. One monitor-owned
event imports the worker's actual timestamps using causal ordering checks;
the SDK worker never mutates the timing collector.

| Stable metric name | Measurement |
| --- | --- |
| `receiptInterval` | Successive valid receipts, `A[i] - A[i-1]` |
| `sendInterval` | Successive qualified successful submission returns, `W[i] - W[i-1]` |
| `receiptSend` | Same successful command's `W - A` |
| `dispatch` | Monitor dispatch, `P - A` |
| `actuator` | Actuator preparation; includes BMC/ALPAO output-shape publication |
| `sdk` | `W - S`; PI uses its grouped serial-write span |
| `saturation` | Instantaneous saturation classification; synchronous or deferred by the controller |
| `delta` | Configured delta-channel calculation |
| `post` | Saturation-worker semaphore posting |
| `throughDelta` | `D - P`, preserving the old `piTimes` endpoints |
| `callback` | `F - P`, including semaphore posting |
| `receiptCallback` | `F - A` |
| `publicationInterval` | Observed input publication interval, `U[i] - U[i-1]`, for consecutive counters |

Valid earlier stages survive a later error. `throughDelta` has an endpoint even
when delta work is absent; the `delta` metric is then unavailable. Unsupported
phases remain unavailable.

`publicationInterval` helps distinguish irregular input publication from irregular
controller receipts. For a continuously published 2 kHz command stream, its
expected mean is 0.0005 seconds. It contributes only when adjacent retained
events have valid, strictly consecutive input counters and positive, increasing
`writetime` values. Missing/malformed timestamps, duplicate/skipped/reset counters,
timing-event loss and attachment changes break the pair; intervals are not divided
by a skipped-frame count. A failed/no-send callback can still observe a publication.
Its count and validity are independent of the compatible three-population summary.

Metadata is sampled at callback entry. The checks reject an active writer or a
counter change across the read, but do not guarantee association with the semaphore
token or pixels actually sent. In particular, the inspected dmcomb implementation
does not raise `write` before updating the combined stream, and publishes
`writetime` before incrementing `cnt0`. A read can therefore briefly observe a new
timestamp with the previous counter. These are **observed publication intervals**,
not a verified frame trace. Stream-clock jumps can also resemble publication stalls.
The metric does not establish publication-to-receipt latency or camera-to-DM latency.

## Detached pipeline events

Controllers with preparation and finalization on separate threads can transfer
an event with `detachEvent()`. This deactivates the monitor's current event and
returns its complete value without publishing it. The monitor can then capture
the next command; the finalizer exclusively owns the detached value and adds
endpoints with `markEvent()`, `importSdkEvent()` and `outcomeEvent()`. These use
the same timestamp validation and successful-submission preservation as the
legacy member methods. `submitEvent()` finally copies the immutable event into
the existing bounded telemetry queue.

The finalizer must be the **only queue producer** in this mode. The monitor must
detach every active event before another `beginEvent()` or `newSession()`, and
must not call `finishEvent()` concurrently with finalizer submissions. Those
legacy methods can themselves publish an active event. Disabled or empty events
do not publish. Configuration still requires every producer and consumer to be
stopped. External job ownership and ordered transfer are the controller's
responsibility; the helper does not create a worker or retain job pointers.

The SDK worker passes immutable result timestamps through the job; it does not
mutate another thread's current event. Actual SDK execution and independent
private preparation can overlap. The finalizer must join their evidence before
submitting the event. Queue loss remains diagnostic and cannot control command
ownership or invalidate an independent reader-release signal.

## Configuration and collection

Add these keys to the controller configuration and restart to apply changes:

```ini
[dm]
latencyTime=2
latencySize=20000
```

`dm.latencyTime` selects the trailing elapsed-time window in seconds.
`dm.latencySize` caps both queued events and retained window events. These are the
defaults; either value set to zero disables collection. There is no runtime INDI
setting for them. Timing is available without `XWC_DMTIMINGS`.

The command path captures fixed-size events and attempts a bounded handoff. A
full queue drops/counts the timing event while command processing continues.
Ordinary application logic drains a bounded queue population, owns the statistics
window and publishes INDI/telemetry. Serialization, statistics, allocation and
telemetry logging occur outside the real-time capture path. The telemeter's
background writer handles file output.

Both endpoints must lie within the current time window and retained event
population. The event cap can shorten actual coverage. The window ages against
current monotonic time, so stopped input becomes stale. Timing maintenance and
publication also run in the ordinary power-off loop, where `appLogic` is skipped. Large valid durations
remain in the statistics. Jitter is sample standard deviation using `N-1`:
two intervals require at least three timestamps; each metric has its own count.
For publication intervals, window membership and reported coverage use the two
local receipt observations (entry when no receipt timestamp is available), while the interval
value uses only the external timestamps. Different clock epochs cannot affect
window eviction or coverage.

Receipt cadence includes rejected receipts and callbacks that fail later.
Successful-return cadence includes longer intervals across known failed/no-send
attempts. Queue loss, unknown submission status and attachment/session changes
break matching instead of inventing an interval across missing evidence.
Duplicate/counter-gap observations are counted. Initial callbacks can contribute
processing/send measurements, with no fabricated receipt or paired `W-A`.

## Read-only INDI properties

Properties belong to the controller's configured device. Unavailable durations
and FPS are **-1**; counts are zero when empty. Mean/extrema/span can be available
from one observation while jitter remains -1. Property state is Ok only when all
three compatible populations are valid; individual stage measurements can still
be useful while state is Idle.

| Property | Elements and units |
| --- | --- |
| `dm_timing` | `receipt_fps`, `receipt_min`, `receipt_max`, `receipt_jitter`, `receipt_count`; `send_fps`, `send_min`, `send_max`, `send_jitter`, `send_count`; `receipt_send`, `receipt_send_jitter`, `receipt_send_min`, `receipt_send_max`, `receipt_send_count`. FPS is reciprocal mean interval; durations are seconds. |
| `dm_timing_stages` | Names `dispatch`, `actuator`, `sdk`, `saturation`, `delta`, `post`, `throughDelta`, `callback`, `receiptCallback`, `publicationInterval`, `sdkDispatch`, `readerRelease`, `staging`, `publication`, `sdkWait`, `sdkTail`, `sdkService`, `conversion`, `sourceCheck`, `shapePublication`, `receiptSdk`, each suffixed `_mean`, `_jitter`, `_min`, `_max`, `_count`, `_span`. Durations/coverage use seconds. |
| `dm_timing_quality` | `window`, `span`, `age` in seconds; `capacity`, `queued`, `session`, `sequence`; `enabled`, `valid` as 0/1; cumulative counters listed below: 21 elements. `age=-1` means no retained endpoint. |
| `dm_timing_state` | Text elements `state`, `clock`, `receipt`, `send`. States are `disabled`, `warmingUp`, `valid`, `stale`, `unavailable`; endpoint descriptions identify the clock and software return semantics. |

`dm_timing_quality.valid` means receipt interval, successful-return interval and
paired receipt-to-return populations all have defined sample variance. `window`
is configured duration; `span` is actual retained endpoint coverage; `age` is time
since the newest retained endpoint. `sequence` identifies a consumer summary and
`session` changes on monitor attachment/reset.

Quality counters are `events`, `receipts`, `submitted`, `failed`, `partial`,
`noSend`, `synthetic`, `rejected`, `duplicates`, `counterGaps`, `invalid`, `dropped`.
They accumulate since collection configuration. `invalid` counts failed/backward
timestamp captures; `counterGaps` counts nonconsecutive observations, not an
inferred number of missing commands. A later callback failure can coexist with a
successful submission. Dropped events prevent complete outcome accounting.

## Recorded telemetry and backend interpretation

`telem_dmtimings` carries every metric's count, mean, sample standard deviation,
minimum, maximum, span and validity, plus session/sequence, window/age/queue state
and cumulative/since-previous-summary quality counters. Those counter deltas
describe summary intervals, rather than exactly the trailing statistics window.
Forced recording emits current availability and unchanged valid statistics.

`telem_fgtimings` is emitted alongside it only when all three populations are
valid. Its existing six fields retain their wire format: `atime`/`atime_jitter`
are receipt-interval mean/jitter; `wtime`/`wtime_jitter` are successful-return
interval mean/jitter; `wmatime`/`wmatime_jitter` are paired `W-A` mean/jitter.
These are seconds, **not raw timestamps**. Existing ACQ/WRT labels remain; use
device identity and endpoint definitions when comparing camera and DM records.
An old compatible record does not establish current validity: inspect the
detailed record first. LogMeta names use `<metric>_mean`, `_stddev`, `_min`,
`_max`, `_count`, `_span`, `_valid`; unavailable duration projections are NaN.
The detailed record appends `publicationInterval` as metric ID 12; the original
twelve IDs and compatible `fgtimings` fields are unchanged. Worker metrics append
IDs 13–17, SDK-worker readiness appends IDs 18–19, and preparation detail appends
IDs 20–23. Rebuilt readers accept complete historical twelve/thirteen/eighteen/
twenty-row records and the current twenty-four-row records. Historical records
project the absent new durations as NaN.
Rebuild `logdump` to read new records and access the new names.

The appended metrics are:

| Metric | Measurement |
| --- | --- |
| `sdkDispatch` | Worker SDK entry minus request dispatch. |
| `readerRelease` | Last-read endpoint minus callback entry; independent of SDK outcome. |
| `staging` | Private delta channel-read/calculation interval. |
| `publication` | Deferred delta/difference publication after SDK success. |
| `sdkWait` | Monitor's matching-result wait, including handoff overhead. |
| `sdkTail` | SDK return to worker readiness after result publication, bookkeeping and notifications. |
| `sdkService` | SDK entry to that same worker-ready endpoint, excluding idle waiting for another request. |
| `conversion` | Pipeline actuator-start through completed private shape initialization and numeric conversion. |
| `sourceCheck` | Completed conversion through coherent source-identity revalidation. |
| `shapePublication` | Completed source check through shared output-shape copy/publication and actuator-end. |
| `receiptSdk` | Successful command's actual SDK entry minus genuine semaphore receipt, using existing timestamps. |

The BMC pipeline records the three preparation segments inside `actuator`.
Classification/counting of its private saturation mask runs in the ordered post
worker from immutable actuator values, after monitor staging and SDK completion;
its separate `saturation` duration is outside preparation and reader release.
Other paths without preparation-segment markers leave those populations empty.
`receiptSdk` excludes SDK execution but includes monitor dispatch and any wait
for the SDK owner; it is a latency measurement, not isolated preparation service.
Its directly accumulated extrema avoid subtracting independent population
maxima. It remains unavailable for synthetic callbacks and unsuccessful sends.

The BMC pipeline captures readiness on its next worker-loop mutex acquisition,
after publishing the prior native result and committing its handoff, immediately
before waiting for or selecting another request. The prior job remains leased
until the stamp is stored, so post processing cannot recycle the result early.
`sdkService` and `sdkTail` are paired within each command; their maxima need not
belong to the command with the maximum `sdk` duration. Readiness can precede
monitor staging/reader release when those stages overlap. These metrics remain
unavailable on paths that do not supply a worker-ready endpoint.

Worker readiness excludes subsequent request selection and native-entry glue.
It also does not establish a free pipeline credit, successful post retirement or
open admission after a fault. The SDK-result image still marks actual native
return, which precedes this readiness endpoint. An isolated command contributes
both new timing populations without requiring a following command or faster DM
updates.

Worker-mode `delta` spans staging, SDK-result wait and publication. It is a wall
interval rather than approximately the old delta computation cost; `staging`
and `publication` separate that work. Overlapping stage durations do not sum
to callback duration. Synchronous controller stage definitions remain intact;
metrics for unexecuted worker stages remain unavailable. Successful SDK-return
cadence and `fgtimes` receipt-to-return latency keep their existing meanings.

BMC qualifies success from `BMCSetArray`; ALPAO uses `asdkSend`. IrisAO measures
the final `MirrorCommand(...MirrorSendSettings)` span, but its success indication
is unqualified: SDK duration is available while successful-return cadence and
paired latency remain unavailable. Its preparation includes segment staging and
saturation queries. PI measures the group from its first command-origin serial
write through the final attempted write; success requires all configured axes.
Manual moves are excluded, partial/failing groups are counted, and historical
controller return/error behavior is preserved. PI return does not acknowledge
motion completion.

Measuring camera-to-DM `W-C` requires a verified camera frame association and
clock relationship for each command. Combined-image `atime`, nominal FPS and
nearest timestamps do not establish that association; auxiliary commands may
reuse an earlier AO command. Keep main/auxiliary provenance distinct and qualify
clock conversion before aggregation. Camera and controller jitter summaries
cannot be added to obtain full-loop jitter. No raw per-command logging is enabled
by default, and observed statistics do not replace qualified synco safety bounds.

Sources: [DM integration](../dm.hpp), [collector](../dmTiming.hpp),
[shared metric definitions](../../../common/dmTiming.hpp),
[detailed telemetry](../../../logger/types/telem_dmtimings.hpp) and
[schema](../../../logger/types/schemas/telem_dmtimings.fbs).
