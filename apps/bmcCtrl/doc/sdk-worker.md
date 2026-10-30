# Overlapped BMC submission and channel-reader release

The optional BMC SDK worker lets the command monitor finish reading dmcomb's
channels while `BMCSetArray()` is running. It can shorten the time those inputs
need protection. SDK return and physical mirror response are separate endpoints;
moving the call to a worker does not make the SDK itself faster.
The first sections describe the joined worker; the final section describes
the independently enabled full pipeline.

```text
monitor: prepare command -> dispatch -> stage saturation and delta -> reader release
                                                                   -> wait SDK result
                                                                   -> publish products
                                                                   -> full completion

worker:                     wake -> BMCSetArray ---------------------> result
```

There is one immutable SDK command in flight. The monitor waits for its actual
result before admitting another command. Saturation and delta/difference
products are committed only on success. Channel changes after reader release
cannot change the captured vector or private delta products.

## Configuration

Worker mode is disabled by default and requires a restart to change. Add these
settings to the **DM controller's** configuration after selecting an available
dedicated worker core:

```ini
[sdk]
workerEnabled=true
threadPrio=55
cpuset=spare56

[dm]
completionStream=dmtweeter_done
readerReleaseStream=dmtweeter_read_done
```

`spare56` is an example, not an assignment. Confirm its availability and use a
different physical core from the existing command monitor (`tweeter`, CPU40
in the current RTC layout). The requested priority above matches the monitor's
FIFO55 on a separate core. `sdk.threadPrio` defaults to 0 (ordinary scheduling)
and `sdk.cpuset` defaults to empty (inherited affinity). Verify the actual
scheduling policy, priority and affinity before comparing timing.

`sdk.timeoutSec` defaults to **1 second** and must be finite and positive.
It is a diagnostic deadline for normal commands, not an SDK cancellation or
physical settling limit. Ordinary app logic checks outstanding command age;
finalization also checks elapsed time if the call returns between polls.
Initialization/control operations are exempt from this command deadline.

The two `[dm]` publishers are independently optional and default to empty names.
`completionStream` keeps its existing successful full-callback semantics.
`readerReleaseStream` has a distinct protocol: it reports the last channel read
even if the pending SDK call subsequently fails. Use distinct names for these
streams and all command/channel/output images.

## Readbacks and timing

The standard `th-sdk` property identifies the worker thread and requested
priority. `sdk_worker` exposes whether a callback is active, admission is closed,
or a fault is latched, plus completed, failed, rejected and coalesced counts.
It also reports verified numeric `policy`, `priority` and `allowedCpus` readbacks.
`activeAgeSec` reports outstanding normal-command age and `timeouts` counts
deadline violations once per job.
Rejected receipts include duplicate counters and receipts while admission is
closed. Coalesced counts report counter advances that were not individually
submitted; stale semaphore posts never cause a burst of repeated commands.

`dm_reader_release` and `dm_reader_release_stream` appear when the reader
publisher is configured. The release remains associated with its original
input identity when the SDK result enriches the record. A release-qualified
record permits no inference about SDK success, readiness for another command,
or physical mirror settling. Large integer INDI values are diagnostic doubles;
exact matching uses the shared record.

The common [`dmtimes` telemetry](../../../libMagAOX/app/dev/doc/dmTiming.md)
retains actual SDK-return cadence and receipt-to-return latency. New metrics
measure SDK dispatch, early reader release, private staging, final publication
and result wait. With overlap, stage durations do not sum to callback duration.
The `delta` wall interval includes staging, the SDK-result wait and final
publication; use `staging` and `publication` to separate those costs.

## Shadow comparison

Rebuild the controller, `dmSyncopation` and `logdump`. Follow the
[reader-protection test](../../dmSyncopation/doc/reader-protection-test.md), adding
this optional observer to the synco configuration:

```ini
[readerReleaseMonitor]
shmimName=dmtweeter_read_done
```

Keep the existing `completionMonitor.shmimName=dmtweeter_done` for an independent
full-callback comparison. Both observers require `simulation.enabled=true`.
The updated controller emits eighteen timing-metric rows even with worker mode
disabled, so update `logdump` before recording the synchronous baseline too.
The release population does not require a qualified full-completion peer: an
input advance after release can legitimately make that later record unqualified.

Compare synchronous and worker runs with both publishers/observers present,
camera and loop settings, dmcomb mode/delay, CPU/IRQ placement and GPU workload
unchanged, and RTC latency tuning enabled. Retain the existing 155 us shadow
budget and operational synco disabled. Compare quality counters and camera/DM
cadence as well as reader-release timing before considering a smaller budget.

## Errors and lifecycle

Worker-mode handle operations, including zero, map loading, clear and close,
are serialized. Control requests stop admission and wait for the whole callback
to retire; SDK return alone does not release its storage. Power transitions
and orderly shutdown drain in-flight work before closing or freeing resources.

An SDK, processing or diagnostic-timeout failure latches the worker fault and closes command
admission. Recover through release and successful reinitialization; there is
no automatic retry of an uncertain submission. An SDK success arriving after
the deadline remains recorded as a successful submission, while its callback
fails and success-dependent products remain unpublished. A stuck SDK call can prevent
drain and shutdown. Its buffer and handle remain owned until it returns;
thread cancellation or concurrent close is not a supported recovery mechanism.

The [engineering plan](../../../agents/plans/2026-10/bmc-sdk-worker-reader-release.md)
records implementation decisions, tests and remaining hardware qualification.

## Full preparation / SDK / post pipeline

A second startup-only option removes the monitor's SDK-result join. The SDK
owner still makes every native handle call in order, while a separate post
worker publishes successful private products. Both options default off;
`sdk.pipelineEnabled=true` requires `sdk.workerEnabled=true`.

```text
monitor: A prepare + stage -> read_done   B prepare + stage -> read_done
SDK:          A BMCSetArray -----------------> B BMCSetArray ---------->
post:                                         A publish -> done
                                                                    B publish -> done
```

There are exactly three private job credits. Each owns its clipped shape,
SDK vector, delta/difference, saturation mask, frozen input identity and timing
record through ordered post retirement. The SDK may accept B while A's post
work is running. Preparing B cannot change A's vector or publication products.
The monitor remains the sole writer of the prepared `*_shape` image, preserving
its preparation-time publication behavior. Delta/difference and saturation
are published by the sole post owner after successful submission.

The queue coalesces to the latest **observed** coherent input. Every genuine
monitor receipt announces its pointer-free identity, including receipts
rejected when all credits are busy. Before native entry, the SDK owner cancels
an unsent job displaced by newer evidence. It does not reread a potentially
closed monitor mapping. Source disconnect/reattachment also invalidates queued
identities; a call already in the SDK drains with its original identity.
Consequently an SDK stall does not cause an old FIFO of commands to replay.

Add these settings to the worker configuration, selecting an available post
core and a distinct SDK-result image name:

```ini
[sdk]
workerEnabled=true
pipelineEnabled=true
pipelineThreadPrio=0
pipelineCpuset=
pipelineResultStream=dmtweeter_sdk_done
```

The post priority defaults to 0 and its cpuset to empty, meaning ordinary
scheduling and inherited affinity. They are independently verified, using the
same rules as the SDK owner. The SDK-result image is optional and defaults to
empty. Its name must not match any command/channel/output image or use the
configured DM stream prefix. The post worker appears as `th-pipeline`;
`sdk_pipeline` reports verified `policy`, `priority`, `allowedCpus`, plus
`capacity`, `inFlight`, `queued`, `sdkActive`, `postActive`, `admitted`,
`retired`, `overloaded`, `cancelled`, `superseded`, `postFailures`,
`vectorRejected`, `readerUnqualified`, and `preparationFailures`.
`cancelled` includes `superseded`; supersession is a nonfault `noSend`
outcome. `sdk_worker.coalesced` retains its original meaning: skipped counter
advances between admitted inputs. Overloaded receipts and admitted superseded
jobs are distinct populations.

In this mode `readerReleaseStream` is written only by the monitor. It publishes
a busy observation followed by reader release, with SDK outcome still unknown;
later SDK results never rewrite that record. At an actual 2 kHz command rate,
its metadata counter therefore advances at approximately 4 kHz, with one
semaphore post per command (approximately 2 kHz). The separate
SDK-result stream publishes one actual return per native call, and the full
completion stream publishes one ordered retirement per admitted command
(approximately 2 kHz each without skips). These are software notifications,
not additional physical DM commands. SDK-result kind qualifies only native
return; it does not qualify channel clearance or successful post work. If
the combined input changes while its pixels are being converted, the private
vector is discarded before SDK entry or shape publication (`vectorRejected`).
This ordinary supersession is a nonfault `noSend`. A validated private vector
can qualify its actual SDK result independently of later channel staging.
If the combined input advances after that vector validation, the reader and
full completion records lose exact source qualification (`readerUnqualified`),
but successful native work, post products, timing and later admissions continue.
Initial metadata-capture instability is also a `vectorRejected` skip.
Qualification loss is distinct from actual conversion/channel-staging
errors or unexpected preparation exceptions (`preparationFailures`), which still fault admission. The qualification
counter can overlap vector rejection and preparation failures. Both diagnostic
properties have complete numeric definitions and an Idle state at startup,
before the first ordinary app update. A latched backend/preparation/post fault
changes their ordinary readback state to Alert.

Detached timing keeps each command's genuine receipt and counter, even when
three stages overlap. SDK intervals use the actual native owner timestamps;
receipt-to-return and receipt-to-post-timing-finish measure latency, not the
pipeline's admission spacing. The callback finish marker precedes full-record
wire publication, timing-queue submission and credit retirement; those final
tails are outside that duration. The full completion endpoint is also stamped
before publishing its own wire record. Saturation decisions are made in the
actuator preparation loop, so the separate `saturation` timing population is empty in
pipeline mode and its cost is included in `actuator`. Its `sdkWait`
population is also empty because the monitor does not join the result. The
producer copies only fixed job payloads; the ordinary saturation consumer drains accumulated
mask counts without losing intermediate jobs when notifications coalesce.

A native error, actual preparation error, diagnostic timeout or post error closes
admission and cancels unsent jobs. Native success arriving after a timeout
remains an actual successful SDK result, but its success-dependent post
products are suppressed. A post failure can be detected after B has already
entered the SDK; B drains truthfully and later unsent work is cancelled.
There is no rollback of a completed mirror submission. Controls, reallocation,
release, reinitialization and shutdown wait for all preparation, native and
post ownership to drain before changing or freeing shared resources. A stuck
native call or stuck post worker can therefore prevent shutdown.

This pipeline is opt-in and needs hardware qualification. Measured typical
SDK work around 104 microseconds suggests more throughput headroom than the
joined callback, but observed maxima already exceed 110 microseconds. This
implementation does not establish a 6 kHz physical-delivery guarantee or
change dmSyncopation's operational policy.
