# DM controller reader-completion signal

## Purpose and scope

On 2026-10-27 the user requested a semaphore indicating when the DM controller
is done, following discussion of the two reader-hold skips in
`synco-sim-20261027T185754Z`. The current shadow model protects command-channel
readers until the later primary/combined observation plus 20 us delivery,
115 us keepout and 20 us guard. A late observer can therefore lengthen the
modeled hold even when the controller has already finished.

The controller can report its actual software endpoint. The latest capture's
BMC SDK duration is about 101 us; roughly 15–17 us of processing follows SDK
return. Completion includes that later work, especially `makeDelta()`, which
rereads command channels. The new signal measures completion of those reads;
it does not shorten SDK execution or measure physical mirror settling.

This change adds an optional publisher to `dev::dm`. It does **not** change
dmSyncopation's live scheduler, shadow decisions, keepout or reader-hold bounds.
A separate [shadow consumer comparison](dm-reader-protection-measurement.md)
now observes this endpoint. Instrument qualification and any replacement of a
scheduling hold remain subsequent work.

## Endpoint and transport

```text
combined command semaphore receipt
  -> monitor validation -> processImage: publish busy
  -> commandDM: actuator preparation -> SDK/serial submission
  -> synchronous saturation-map work
  -> makeDelta: final command-channel reads
  -> saturation-worker semaphore post
  -> timing-event handoff
  -> publish completion record -> post dedicated stream semaphores
  -> callback returns -> next wait
```

The asynchronous saturation worker reads its own instantaneous map, not the
command channels. Its existing process-local semaphore is retained for that
worker. Completion uses a separate optional ImageStreamIO stream and its
standard semaphores; input counters and semaphores are not modified.

`dm.completionStream` is empty by default. An enabled publisher opens its
resources at startup and closes only after the monitor has joined. The hot path
uses fixed-size metadata checks, integer clock reads, coherent atomic record
stores and non-waiting semaphore posts. There is no allocation, filesystem
access, statistics, logging, INDI traffic or additional worker in that path.
Publication overhead still requires measurement on the instrument.

The version-1 CPU stream contains 16 uint64 words. The shared protocol in
`libMagAOX/common/dmCompletion.hpp` serializes fields explicitly, avoiding struct
padding. The corresponding publisher/reader is
`libMagAOX/app/dev/dmCompletion.hpp`. See the
[interface guide](../../../libMagAOX/app/dev/doc/dmCompletion.md) for the layout,
configuration and readbacks.

## Qualification and lifecycle

The record identifies publisher session, attachment/progression epoch, callback
sequence, combined-stream inode/counter and raw producer publication timestamp.
Start/finish use local `CLOCK_MONOTONIC`; external timestamps are identity only.

A qualified complete record requires a genuine semaphore receipt, a new stable
input counter, unchanged input inode/counter/publication timestamp throughout
the callback, callback success and an explicitly successful backend outcome.
Timing collection can be disabled or lose events without suppressing completion.

Initial-image callbacks, duplicate receipts, incoherent metadata, unknown backend
outcomes and no-send callbacks remain unqualified. Failures, partial submission
and rejected outcomes publish error. Begin overwrites old completion with busy
or unqualified without posting; finish posts every result. Connect, discarded
receipt, disconnect and orderly shutdown invalidate and notify. A new optional
`shmimMonitorEvent::disconnected` is appended without renumbering existing
events, and occurs before established attachments close, including allocation
failure.

The publisher holds an exclusive nonblocking lifetime lock. A foreign or
incompatible stream is rejected. A new image is first created under a private
name, then linked without replacing an existing destination: ImageStreamIO's
normal create operation can replace an existing filename. Compatible completion
streams remain mapped at the same inode across orderly or crashed publisher
restarts; restart publishes unavailable under a new session. A crash-left
write flag can be recovered only under that exclusive lock. Stream files and
the lock file remain for reuse; they are not removed while readers may be
attached.

Semaphore counts can coalesce or contain stale posts. They mean “inspect the
latest record,” never “this particular command is complete.” A future synco
consumer must read coherently and validate identity, session/epoch, current input
counter, freshness and progression. INDI numeric fields are diagnostic doubles;
the 64-bit wire values must be used for exact matching. A publisher crash can
leave a previously complete record, so freshness and liveness remain necessary.

This is a software observation, not an atomic provenance transaction across
dmcomb and its input channels. In particular, the inspected dmcomb path does
not consistently raise the combined stream's write flag. Repeated identity
checks reject detected races but cannot prove which semaphore token or exact
multi-channel pixel set the backend consumed. A future consumer must retain
dmcomb-consumption checks, early-main and copy protection, and configuration/FSM
gates. Completion cannot grant permission for writes that race a new main.

## Local implementation and verification

Implemented the optional publisher, generic controller integration, lifecycle
invalidation, configuration and read-only INDI properties. No application-specific
backend behavior, telemetry schema or synco scheduling rule is changed.

- Protocol/publisher suite: **404 assertions / 11 cases**, including actual
  final-name ImageStreamIO attachment and semaphore delivery, no begin post,
  errors/unqualified states, stale/coalesced notices, exclusive ownership,
  malformed/foreign streams and restart recovery.
- Generic DM integration suite: **162 assertions / 5 cases**, exercising real
  one-pixel delta reads, endpoint ordering, timing enabled/disabled, backend
  qualification, counter/stamp/inode/write changes, lifecycle and INDI setup.
- Monitor hook suite: **63 assertions / 5 cases**, covering restart, state
  rejection, shutdown, allocation failure and owners without an optional hook.
- Existing generic DM, timing integration and BMC/ALPAO/IrisAO/PI335 controller
  regressions: **554 assertions / 20 cases**. All focused/regression suites
  total **1,183 assertions / 41 cases**.
- Normal library/PCH and dmSyncopation builds pass. Full touched C++ files pass
  clang-format checks; whitespace checks pass. SDK backend regression tests use
  local software stubs; this is not a hardware controller build/qualification.

Both new completion suites are registered in `tests/tests.list` for the standard
build/run workflow.

All work and tests run on the workstation with private test streams and backend
stubs. No agent runs commands on, deploys to or controls instrument hardware.

## Operator follow-up

1. Rebuild/install the DM controller (`bmcCtrl` for dmtweeter). There is no new
   telemetry record and no `logdump` rebuild requirement for this change.
2. Add `completionStream=dmtweeter_done` under `[dm]` in the **dmtweeter
   controller configuration**, then restart the controller through the normal
   operator procedure. The stream name uses underscores; the INDI device name
   remains unchanged. This is not a dmtweeter-synco configuration setting.
3. Confirm the completion stream/read-only properties exist and compare existing
   DM timing and quality populations before/after enabling publication. Disabled
   timing must not disable the signal. Treat an INDI sample of busy as expected
   when it lands during a callback; it is not an aggregate fault rate.
4. Run the new bounded
   [reader-protection shadow measurement](dm-reader-protection-measurement.md)
   before considering replacement of any fixed reader hold. Its optional synco
   observer and telemetry consumer do not alter scheduling guards; instrument
   qualification remains pending.
