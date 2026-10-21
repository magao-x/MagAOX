# DM controller timing telemetry

`dev::dm` reports controller receipt cadence, successful submission-return cadence
and local processing times for BMC, ALPAO, IrisAO and PI-335 controllers. All local
markers use integer `CLOCK_MONOTONIC` nanoseconds; reported durations and jitter
use **seconds**, and FPS uses **Hz**. SDK or serial-write return is a software
endpoint; physical DM response and settling require separate measurements.

## Endpoints and metrics

```text
dmcomb publishes combined command -> successful controller semaphore wait
  A  first timestamp after wait return
     monitor metadata/state checks
  P  dev::dm::processImage entry
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

This ordering describes BMC/ALPAO. `A` follows the user-space wait return; queued
tokens can return immediately. `F` precedes timing-event handoff and final return.
The saturation worker runs asynchronously and has no per-command completion
measurement here. An initial existing-image callback has no `A`.

| Stable metric name | Measurement |
| --- | --- |
| `receiptInterval` | Successive valid receipts, `A[i] - A[i-1]` |
| `sendInterval` | Successive qualified successful submission returns, `W[i] - W[i-1]` |
| `receiptSend` | Same successful command's `W - A` |
| `dispatch` | Monitor dispatch, `P - A` |
| `actuator` | Actuator preparation; includes BMC/ALPAO output-shape publication |
| `sdk` | `W - S`; PI uses its grouped serial-write span |
| `saturation` | Synchronous instantaneous saturation-map update |
| `delta` | Configured delta-channel calculation |
| `post` | Saturation-worker semaphore posting |
| `throughDelta` | `D - P`, preserving the old `piTimes` endpoints |
| `callback` | `F - P`, including semaphore posting |
| `receiptCallback` | `F - A` |

Valid earlier stages survive a later error. `throughDelta` has an endpoint even
when delta work is absent; the `delta` metric is then unavailable. Unsupported
phases remain unavailable.

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
| `dm_timing_stages` | Nine names `dispatch`, `actuator`, `sdk`, `saturation`, `delta`, `post`, `throughDelta`, `callback`, `receiptCallback`, each suffixed `_mean`, `_jitter`, `_min`, `_max`, `_count`, `_span`: 54 elements. Durations/coverage use seconds. |
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
