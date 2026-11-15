# DM controller reader completion

`dev::dm` can publish a dedicated shared-memory record and post its semaphores
after the command callback finishes all command-channel reads. The endpoint is
after backend submission, saturation-map work, `makeDelta()`, saturation-worker
notification and timing-event handoff. It is software completion; physical
mirror motion and settling are not measured.

```text
combined command -> controller callback -> SDK/serial return
  -> saturation map -> delta/channel reads -> worker post -> timing handoff
  -> completion record + dedicated semaphore posts -> callback return
```

## Enable publication

Add this to the **DM controller's** configuration, for example dmtweeter's
`bmcCtrl` configuration, and restart the controller:

```ini
[dm]
completionStream=dmtweeter_done
```

The default is an empty name, which disables the publisher and its extra INDI
properties. The name must identify a separate stream and contain only letters,
digits, underscores, periods or hyphens. It cannot name the monitored combined
command stream. Timing telemetry's `dm.latencyTime`/`dm.latencySize` settings do
not control this signal.

The controller creates or reuses a compatible one-dimensional, 16-element
`uint64` CPU ImageStreamIO stream. Its own standard semaphores notify readers;
the combined-command and saturation-worker semaphores keep their existing roles.
Only one completion publisher may own a name. Conflicting publishers and foreign
or incompatible streams fail setup. Compatible stream files remain in place
for attached readers across publisher restarts, with a new session each restart.

This publisher does not change dmSyncopation's scheduler or shadow reader holds.
The publisher itself has no telemetry schema or `logdump` build requirement.
Rebuild the controller to use the feature. The separate
[synco reader-protection shadow test](../../../../apps/dmSyncopation/doc/reader-protection-test.md)
consumes this stream and adds matched `dmsynco` telemetry; that test requires an
updated synco and reader. Existing
[DM timing telemetry](dmTiming.md) remains the source of duration distributions.

## Read-only INDI

Properties belong to the controller device. They appear only when publication
is configured.

| Property | Elements |
| --- | --- |
| `dm_completion` | `state`, `session`, `epoch`, `sequence`, `inputInode`, `inputCounter`, `processingUs`, `ageSec`, `outcome`, `qualified` |
| `dm_completion_stream` | `name`, `meaning` |

`processingUs` is the latest attempt's callback entry to completion-record
timestamp, before publication/notification; it excludes semaphore wake and
monitor dispatch. `ageSec` is the age of that timestamp. Either is `-1` when
unavailable. These are latest samples, not rolling statistics or observer
delivery latency. Property state is Ok for qualified completion and Idle
otherwise. An ordinary INDI update may sample busy during a healthy callback.
Large integer IDs are displayed through INDI doubles and may lose precision;
consumers must use the full-width shared record for exact matching.

| `state` | Name | Meaning |
| ---: | --- | --- |
| 0 | unavailable | No current completion; startup, detach or invalidation |
| 1 | busy | A genuine new stable input is being processed |
| 2 | complete | Successful callback and backend, with stable input identity |
| 3 | error | Callback failure, failed/partial submission or rejected outcome |
| 4 | unqualified | Initial/duplicate/incoherent attempt, unknown backend or no send |

`outcome` uses the shared DM timing enum: unknown 0, success 1, failed 2,
partial 3, noSend 4, rejected 5. A completion is qualified only for state 2,
outcome 1, callback result zero and the valid identity/timestamp fields below.
Backends without explicit successful-submission reporting remain unqualified.

## Shared record and notifications

Use `MagAOX::app::dev::dmCompletion::read()` for a bounded coherent read and
`MagAOX::dmCompletion::qualified()` to check protocol-level success. Definitions
are in `common/dmCompletion.hpp`; the wire is explicitly serialized, not a raw
copy of a C++ struct.

| Word | Version-1 field |
| ---: | --- |
| 0 | Magic `0x4d414758444d444e` |
| 1 | Protocol version, currently 1 |
| 2–4 | Publisher session, attachment/progression epoch, callback sequence |
| 5 | State |
| 6–7 | Combined-input metadata inode and `cnt0` |
| 8–9 | Raw combined-input `writetime` seconds and nanoseconds |
| 10–11 | Callback start and completion, integer `CLOCK_MONOTONIC` nanoseconds |
| 12–13 | Backend outcome and signed callback result |
| 14 | Publisher PID, diagnostic only |
| 15 | Reserved, zero |

Signed fields use their 64-bit word representation. External `writetime` fields
are identity only: do not subtract them from monotonic timestamps. The completion
record's timestamps are authoritative for this protocol; image metadata timestamps
are not a completion-time interface.

At callback start the controller overwrites previous completion with busy or
unqualified, without posting. At callback finish it publishes the result and
posts dedicated stream semaphores. Attachment, rejected receipt, disconnection
and orderly shutdown publish unavailable and notify. Input inode change or
counter rollback starts a new epoch; restart starts a new session. Synthetic
initial callbacks never qualify, even if the backend sends a command.

Both start and finish increment the completion stream's `cnt0`. A 2 kHz
controller therefore normally shows a 4 kHz stream counter rate in monitoring
tools, while finish semaphore notifications remain at 2 kHz. Lifecycle
invalidation adds occasional notifications and counter updates.

A semaphore wake means **read the latest record**. Notifications can be stale or
coalesced; the semaphore count is not a command identity or a completion count.
A reader that wants to act on completion must:

1. Read a coherent recognized record; reject unavailable/busy/error/unqualified.
2. Match its complete 64-bit session, epoch, sequence and input identity to the
   command being protected and the current combined stream. An old completion
   cannot authorize writes after a newer command starts.
3. Enforce freshness, publisher liveness and expected progression. A process
   crash can leave its last record in shared memory until restart.
4. Retain other consumers' completion checks and the scheduler's copy,
   early-main, keepout and configuration/FSM protections.

Input identity must be coherent, new and unchanged across the callback to qualify.
These checks do not establish an atomic relationship between a semaphore token,
combined pixels and every input channel. Concurrent producer updates, including
dmcomb paths that do not raise `write`, can defeat such an association. The signal
reports the controller's observed software endpoint; use it with the cooperating
system's ordering and protection rules.

## Separate early reader release

The optional last-read publisher uses a distinct protocol and stream. For the
[BMC SDK worker](../../../../apps/bmcCtrl/doc/sdk-worker.md), it can release
channel readers while `BMCSetArray()` remains in flight. Add to the controller:

```ini
[dm]
readerReleaseStream=dmtweeter_read_done
```

This name defaults to empty and must differ from completion, command, channel
and output streams. The early endpoint also works with a synchronous controller;
there it follows the existing SDK and delta processing. It does not change the
version-1 `completionStream` contract or any scheduling hold.

The 16-word early record has magic **`0x4d414758444d5244`** and version 1.
Common identity words retain their definitions above; differences are:

| Word | Reader-release meaning |
| --- | --- |
| 5 | `released` (5) means qualified channel reads finished, independently of SDK outcome. Other applicable states remain unavailable, busy, error and unqualified. |
| 11 | Frozen last-read endpoint in local monotonic nanoseconds. |
| 12 | Actual backend outcome, initially unknown; updated after submission finalization. |
| 13 | Reader/processing result at the last-read endpoint. A later SDK failure does not change it. |
| 15 | Raw backend result, or `INT64_MIN` when unavailable. |

Use `MagAOX::dmCompletion::readersQualified()` for early records; the existing
`qualified()` predicate accepts only full-completion records. The helper's
coherent `read()` decodes both distinct protocols. A released record can carry
a failed SDK outcome and remain a truthful release of that attempt's input
reads. It cannot authorize another hardware submission while the backend is busy.

Begin updates the stream without posting. Release timestamps and posts the
last-read result. Actual SDK outcome/code then enrich the same immutable job
identity and endpoint and post another notification. A normal BMC command
therefore produces three counter increments and two notifications: at 2 kHz,
monitoring can show 6 kHz counter progression and 4 kHz semaphore posts.
Notifications may coalesce or lead to repeated observations of the same enriched
record. They do not represent multiple DM submissions.

The later combined input may advance without changing the frozen release record.
That advance can make the independent full-completion record unqualified;
measure qualified release coverage independently of a full-completion peer.
Consumers must still match the exact current command, reject stale/restarted
publishers and protect against a newer callback already reading channels.

When enabled, read-only `dm_reader_release` reports state, session, epoch,
sequence, input inode/counter, processing microseconds, endpoint age seconds,
backend outcome/result and release qualification. `dm_reader_release_stream`
reports `name` and `meaning`. These properties appear independently of the
full-completion properties; exact integer matching belongs to the shared record.

## Detached jobs and SDK-only results

`captureRecord(input, genuine)` freezes the combined stream's best-effort
metadata identity and current monotonic entry time into a pointer-free value.
It works when publication is disabled. Its busy/unqualified state records the
capture's qualification; session, epoch and sequence are assigned later by a
publisher. The caller must validate that this identity belongs to the consumed
channel snapshot. Capturing metadata alone does not lock pixels or make a
multi-stream snapshot atomic.

After exclusive job ownership transfers to a result thread,
`publishCompleted(record, callbackResult, outcome, finishedNs)` publishes its
frozen result. It never rereads the combined image: that image may already
contain a newer command or have been detached. Each publisher assigns its own
session, epoch, sequence and kind, rejects duplicate input progression, and
starts a new epoch on input inode change or counter rollback. An explicit
`finishedNs` preserves the actual stage endpoint; its default `-1` captures the
current monotonic time. Invalid captured identity, non-genuine input and
backward endpoints remain unqualified. A nonzero stage result is an error.

Each stream has exactly one publication owner. Do not mix detached publication
with a concurrently active legacy `begin()`/`finish()` owner. An early-release
stream belongs to the monitor; publishing an old SDK enrichment into it after
a newer callback begins would overwrite current readiness. A pipeline therefore
uses a separate SDK result publisher, opened with `Kind::sdk` (2), and leaves
the early stream's current command untouched.

The SDK-only version-1 wire uses magic **`0x4d414758444d5344`**, the same 16 words,
`complete` for qualified successful SDK return, and word 15 for the native SDK
result (or `INT64_MIN` when unavailable). The producer supplies the SDK-stage
start/end timestamps in its record; the input identity remains the original
job's. `sdkQualified()` accepts only this endpoint. `qualified()` and
`readersQualified()` reject it, so an SDK success cannot substitute for early
reader clearance or full output publication. Failure remains an error with the
actual backend outcome/code. Notifications are still latest-record hints and
can coalesce; this stream is not a lossless job-result queue.

Full completion in detached mode is published by the ordered finalizer after
that job's required output products and notifications. It qualifies the frozen,
caller-validated job identity, even after the live input advances. This is
separate from legacy `finish()`, which retains its original live-input stability
check. A result stream never grants ownership of a later command or promises
physical mirror settling.
