# Overlapped BMC submission and channel-reader release

The optional BMC SDK worker lets the command monitor finish reading dmcomb's
channels while `BMCSetArray()` is running. It can shorten the time those inputs
need protection. SDK return and physical mirror response are separate endpoints;
moving the call to a worker does not make the SDK itself faster.

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
