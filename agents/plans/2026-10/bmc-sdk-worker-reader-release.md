# BMC SDK worker and early channel-reader release

## Objective and evaluation

Requested on 2026-10-27, following the
[first reader-protection measurement](dm-reader-protection-measurement.md#2026-10-27-first-operator-capture-044834z):
move `BMCSetArray()` to a worker, signaled at the present call site in
`bmcCtrl::commandDM()`, while the shmimMonitor thread finishes the remaining
DM processing. Prevent another command from reusing the worker's data before
the SDK returns. Record the design in a separate plan before implementation.

**The approach is workable.** The SDK consumes an actuator vector already
copied out of the combined command. The later `makeDelta()` channel reads can
run concurrently with the SDK call. A dedicated reader-release endpoint can
therefore precede SDK return, while a separate submission result retains the
actual hardware-interface outcome.

Recommendation: start with **one immutable command in flight and a wait at
the tail of the same monitor callback**. Publish reader release before that
wait. The callback returns only after collecting the matching SDK result and
finishing success-dependent publication. This achieves earlier input clearance
without allowing overlapping command preparation or introducing a command FIFO.

Status: implementation authorized on 2026-10-28 and completed on the workstation,
with local verification recorded below. Worker mode remains disabled by default.
The live scheduler, its protection budget, instrument configuration and CPU
placement remain unchanged. Instrument actions and qualification remain operator
tasks.

## 1. Current path and expected benefit

Relevant source locations at the time of this review:

- [`bmcCtrl::commandDM()`](../../../apps/bmcCtrl/bmcCtrl.hpp): actuator mapping,
  conversion/clipping and output-shape publication; `BMCSetArray()` at line
  587; then instantaneous saturation processing.
- [`dev::dm::processImage()` and `makeDelta()`](../../../libMagAOX/app/dev/dm.hpp):
  completion/timing bookkeeping, controller call, additional reads of the
  individual dmcomb channels, output-delta/difference publication and
  saturation-worker notification.
- [`dmCompletion`](../../../libMagAOX/app/dev/doc/dmCompletion.md): the current
  `dmtweeter_done` endpoint follows SDK return and all of that callback work.

The latest capture reports a typical reader requirement near **125 us** and
a retained maximum of **133.372 us**, including approximately **101 us** in
the SDK. Approximate stage means provide this comparison:

```text
Current, all in the monitor callback:

  prepare ~7 us -> SDK ~101 us -> saturation ~5 us -> delta ~10 us -> finish
  |<----------------------- channel reader hold ~125 us ----------------->|

Proposed, with separate physical cores:

  monitor: prepare -> dispatch -> stage saturation/channel reads -> RELEASE
                                                                    |
                                  wait for matching SDK result <-----+
                                    -> publish staged outputs -> finish

  worker:              wake -> BMCSetArray ---------------------> result

  RELEASE: input/channel reads finished; SDK may still be running
  result:  actual SDK return; needed before the next command can be admitted
```

This suggests reader clearance on the order of **20–30 us from callback
entry**, plus the new buffer-copy, handoff and scheduling costs. It is an
estimate, not a measured bound. The shadow test must also retain its matched
primary/combined-observation basis for a comparable reader-hold measurement.
SDK completion would still be around 108 us from callback entry plus the new
copy/dispatch overhead; physical DM response is not measured by either
endpoint. Output publication after the join adds its own tail work.

The existing **155 us** shadow protection budget remains unchanged until the
new endpoint has been measured and qualified. Separating readers from SDK
submission does not remove the independent physical cadence/keepout limits.

## 2. SDK evidence and its limits

A local vendor source archive is available at:

```text
/home/jrmales/Documents/MyPapers/Projects/MagAOX/Electronics/DMs/BMC/2K/Software/dm-sdk-3.5.3
```

In that **3.5.3 archive**:

- `include/BMC/BMCApi.h:260` declares a const input array. A null mapping
  argument selects the previously loaded LUT.
- `BMC/BMC.cpp:211` converts that input synchronously into mutable per-handle
  storage before dispatching to the PCIe/USB backend.
- `BMC_PCIeAPI/BMC_PCIeApi.cpp:401` uses further mutable per-handle frame
  buffers. A driver-side lock does not protect these userspace buffers from
  concurrent SDK calls.
- `BMCVersionString()` is declared at `include/BMC/BMCApi.h:592`.

This supports immutable input ownership through SDK return and serialization
of **all operations on a handle**. It supplies no general reentrancy,
cross-thread affinity, cancellation or physical-settling guarantee. The SDK's
internal work and allocations are not eliminated by moving it to a worker.

The installed instrument library/version and active transport branch have not
been established by this review. Log the actual SDK version and relevant
device/transport identification during qualification; verify its input-lifetime
contract before sharing an input array between threads. Do not infer those
contracts from the repository's SDK test stub or timings in another SDK branch.

## 3. Command ownership and sequencing

The proposed callback sequence is:

1. Admit the attempt only when no earlier command or control operation owns
   the slot. Capture its session/epoch/sequence and combined inode, counter and
   raw publication timestamp, with the existing qualification rules.
2. Map/convert/clamp into preallocated owned storage. Publish `m_outputShape`
   at its existing point; it already precedes SDK success today.
3. Freeze the worker request and signal it at the current `BMCSetArray()`
   location. A request contains the command identity, buffer and configuration
   generation. Acceptance of a request is not successful submission.
4. On the monitor thread, stage instantaneous saturation results and snapshot
   the individual channels needed by `makeDelta()`. Once the final external
   pixel read is complete, publish the qualified reader-release record.
5. Wait for the matching worker result. An already completed result requires
   no blocking. Stale, duplicate or interrupted notifications do not satisfy
   the sequence/generation predicate.
6. Import actual SDK start/end timestamps and outcome. On SDK success, commit
   the staged saturation results and publish the delta/difference products,
   notify the saturation worker, and finish the callback. On failure, discard
   staged success-dependent products, report failure and latch a fault that
   inhibits new admissions pending an explicit recovery transition.

The worker performs SDK operations and fixed timestamp/result handoff. It does
not read command streams, alter saturation/delta state, call the monitor's
timing collector, or perform ordinary INDI/logging work. Error codes return to
the controller thread for reporting. No application allocation is introduced
in the per-command handoff or staging path.

| Resource | Owner and lifetime |
| --- | --- |
| Combined image pixels | Monitor reads only during preparation; never passed to the worker |
| Individual dmcomb channel pixels | Monitor reads during staging; no access after reader release |
| Prepared actuator vector | Monitor prepares it; immutable until both SDK and monitor processing finish |
| SDK request buffer | Worker owns its use through actual SDK return; never reused by a later attempt |
| Saturation/channel-sum staging | Monitor-owned preallocated storage, committed on SDK success |
| SDK result/timestamps | Worker publishes once for its job; monitor consumes by exact identity |
| SDK handle, mapping and configuration | Serialized SDK owner; no close, reallocation or reconfiguration during an operation |

Initially use a **separate preallocated SDK command buffer** so the worker's
input and the monitor's saturation input have independent storage. Measure
the extra copy. Sharing the immutable prepared vector is a later optimization
if the installed SDK contract supports it. Neither approach requires a queue
of future commands or a second active monitor callback. Request/result
visibility must use an explicit release/acquire handoff; semaphore counts alone
do not establish the matching job or authorize buffer reuse.

## 4. Preserve existing products and errors

Today an SDK failure returns before saturation updates, `makeDelta()` and its
output publications. Simply running the existing `makeDelta()` early would
publish products for a command that may subsequently fail.

Split channel snapshot/sum staging from delta/difference publication:

- Read the configured non-delta and delta channels into the existing owned
  sums, `m_totalFlat` and `m_totalDelta`, before releasing readers.
- After SDK success, calculate/publish the current formulas using those sums
  and the prepared output shape: output delta is shape minus total flat;
  output difference is total delta minus output delta. Do not reread channels.
- Stage saturation flags and their count privately, and commit/post only on
  success. A worker result alone must not mutate those monitor-owned products.
- Keep a synchronous wrapper preserving the current order and failure behavior
  for other DM controllers and for BMC with the worker disabled.

The generic `dev::dm` integration needs an explicit opt-in preparation,
staging/release and submission-finalization path. Its intermediate success
means preparation succeeded; only finalization qualifies SDK submission.
Other controllers retain their existing synchronous interface. Reuse optional
helper interfaces and existing base-class patterns rather than introducing a
general task scheduler.

### 2026-10-28 refinement: optional callback-tail interface

The user's proposed `prePostCleanup()` interface is a suitable location for
the normal-path SDK join, immediately ahead of final output publication and
`m_satSemaphore` notification. A name such as `finalizeCommand()` would make
its submission/result role clearer; the final name remains an implementation
choice. Keep the default implementation a no-op for synchronous controllers.

For worker-enabled BMC, use this order:

```text
commandDM: prepare and dispatch SDK work; stage saturation
  -> makeDelta deferred variant: read channels and calculate private results
  -> publish reader release: no further command-channel reads
  -> prePostCleanup/finalizeCommand: wait for matching SDK result, import outcome
  -> on success: commit saturation and publish deferred delta/difference
  -> post m_satSemaphore
  -> finish timing and existing full completion
```

Both current assignments to `m_outputDelta` and `m_outputDiff` publish shared
products. The deferred path must calculate into preallocated private storage;
merely withholding a semaphore post after modifying shared pixels/counters
would not preserve failure behavior. Common `dev::dm` code owns the deferred
delta/difference publication after the hook succeeds; BMC need not publish
those generic products. Keep the current synchronous `makeDelta()` wrapper
and ordering for controllers without deferred work. No-delta commands still
need reader release and finalization.

Calculate `stagedDelta = preparedShape - totalFlat` and
`stagedDiff = totalDelta - stagedDelta`. The deferred calculation must not read
the previous public `m_outputDelta()` as though the new delta were published.

The hook is also needed on failure paths: a staging error, exception or stop
after dispatch cannot bypass draining the worker. Use unified finalization or
a scoped drain guarantee, rather than appending a normal-path-only call before
the existing semaphore post. Preserve actual SDK success independently if
subsequent software processing fails.

Any software error after dispatch must still collect/drain the job before
returning or releasing its storage. If the SDK succeeded but later publication
failed, retain successful submission independently of the failed callback.

## 5. Reader release and submission are different interfaces

Keep `dm.completionStream=dmtweeter_done` and its **version-1 meaning** intact.
A qualified complete record currently requires SDK success, callback success
and stable combined identity through callback completion. It cannot be
published early with a fabricated successful outcome.

Add a separate optional reader-release stream; proposed configuration/name:

```ini
[dm]
readerReleaseStream=dmtweeter_read_done
```

Its protocol must distinguish unavailable, busy, released and unqualified
attempts, and carry the frozen job identity, callback entry and last-read
monotonic endpoint. Reuse the existing completion publisher's coherent fixed
records, exclusive ownership and lifecycle conventions where possible, with
an explicitly distinct protocol identifier/meaning. Release does not require
an SDK success that has not yet occurred. Synthetic/duplicate/incoherent
attempts cannot grant qualified permission.

The result still belongs to that frozen job if the combined stream legitimately
advances after reader release. Do not relabel a new live image as the completed
job. Leave the old version-1 live-stability predicate conservative: such an
advance may make its completion unqualified, which is **not** an SDK failure.
Actual submission success/failure remains recorded against the immutable job
in timing/result state. A future phase-aware completion protocol can expose
both endpoints together if needed; it must not silently change version 1.

A late SDK error does not undo the fact that external reads already finished.
It instead invalidates further command admissions. Reader-release availability
alone must never be interpreted as backend readiness or successful DM action.

For synco, a matching release could eventually shorten protection of actual
destination channels such as `dm01disp07` and `dm01disp08`. Producer-owned
`*_aux` streams are already independent. Synco must also retain dmcomb's own
consumption checks, protection against a newer main command, freshness/liveness,
copy budgets and physical trigger spacing. A release for an earlier command
cannot authorize writes while a newer command is reading. Semaphore posts
remain hints to inspect the latest exact record, not per-command tokens.

First extend the **shadow measurement** to measure qualified read releases
independently, joining full completion opportunistically for the same job.
Report release-only, qualified-version-1 and immutable-SDK-result coverage
separately. Requiring a qualified version-1 peer would censor valid releases
followed by a permitted input advance. Expose the frozen identity with the
new SDK-result diagnostics so actual submission can be associated even when
version 1 is unqualified. Do not change the live scheduler as part of the
initial worker implementation.

## 6. Timing with overlapping stages

[`dev::dmTiming`](../../../libMagAOX/app/dev/dmTiming.hpp) has one active event
owned by the monitor. Its current `markAt()` rejects timestamps earlier than
any already recorded marker. Importing an SDK-start timestamp after recording
the monitor's later channel-read markers would therefore be rejected even
when the new overlap is correct.

Return fixed SDK timestamps and outcome through the worker result. Add a
deliberate single-owner import/finalization mechanism that validates causal
ordering, including:

```text
receipt <= entry <= preparation end <= SDK start <= SDK end <= callback finish
                         |
                         +-> staged reads -> reader release <= callback finish

SDK and staged reads may overlap; SDK end need not precede saturation start.
```

Preserve existing sequential validation for unchanged controllers. Existing
`fgtimes` meanings remain receipt interval, actual successful SDK-return
interval, and receipt-to-SDK-return latency. Keep stage durations independent;
their sum is no longer callback duration. In worker mode, the delta operation's
start-to-end wall interval spans staging, the SDK-result wait and final delta
publication; document that changed composition explicitly. Add separate
channel-staging and delta-publication durations plus the last-read endpoint,
so the existing delta span is not mistaken for CPU time spent reading channels.
Synchronous delta timing retains its current meaning.

Add measurements for dispatch-to-SDK-entry, callback-entry-to-reader-release,
release-notification delay, SDK-result wait and total callback duration.
Record busy/overrun/coalesced attempts, SDK failures, invalid identities,
stale results and dropped observations. Statistics and serialization remain
outside the realtime threads. Extend telemetry compatibly and document which
durations include output publication or wait time.

## 7. SDK lifecycle, waits and overload

In worker-enabled mode, make the **worker the sole caller of handle operations**,
including open, map loading, normal set, zero, clear and close. Control requests
are synchronous from the caller's perspective but use the same serialized
owner after stopping command admission. Return needed handle metadata through
owned results rather than inspecting mutable SDK internals concurrently.
Command requests and app/control requests have different calling threads:
serialize their access to the single slot explicitly. Do not treat those
multiple callers as a single-producer queue. Keep the admission lock out of
the SDK call and do not hold it while waiting for a result needed by another
thread to finish the current callback.

Admission closure for control operations also matters while no SDK command is
running. Reject receipts while admission is closed, or wait before input
selection and identity capture and select fresh input afterward. Waiting only
inside `commandDM()` would retain an obsolete pixel pointer/identity even with
the recommended tail join.

Shutdown, power transitions, reinitialization and calibration/mapping changes
must stop admissions, drain the in-flight SDK operation **and retire the whole
monitor callback**, including output publication and timing/completion finalization,
then perform the required zero/clear/close sequence. The worker can remain
idle across DM power cycles, with a new handle generation on reopen; stop/join
it before application teardown or worker reconfiguration and before freeing
its resources. `releaseDM()` currently signals the monitor and
sleeps one second; a sleep is not proof of quiescence and must not serve as the
new ownership barrier.

SDK idle and command-slot free are different states. SDK return makes the
backend operation idle; only callback retirement releases the command slot
and permits control operations or reallocation. Early reader release does
neither. Keep those state transitions explicit in the handoff and drain tests.

Use an explicit stop flag, blocking wake and join when idle. Do not send a
generic thread-stop signal into an active SDK call with unknown interruption
semantics. Startup failure after thread creation must also wake/join that
thread before releasing its resources. A worker must never require an app or
INDI mutex held by a caller waiting for its result.

A timeout reports a fault and stops new admissions; it does **not** authorize
buffer reuse, concurrent zero/close, thread cancellation or freeing a handle
still used by a blocked SDK call. Record operator recovery requirements if
the SDK cannot be drained. SDK exceptions must become a failed result while
still respecting resource ownership.

The current monitor logs callback errors and can continue; that is not the
new fault latch. Add an explicit worker-path fault/admission state and select
the reinitialization/release transition that clears it only after successful
drain and recovery. Do not automatically retry an uncertain submission.

There is one in-flight command and no SDK backlog. The tail wait prevents
reentry into preparation. Explicitly handle accumulated semaphore posts and
counter advances: after readiness, select a coherent current input, count
skipped/coalesced advances and reject duplicate counters; never replay old
posts as a burst of identical SDK commands. Scope this admission behavior to
the opted-in path and retain the existing generic monitor behavior elsewhere.

Waiting only at the beginning of the next `commandDM()` is insufficient:
`dev::dm::processImage()` has already captured attempt identity and the monitor
has selected its pixel pointer by then. A fully asynchronous callback return
would need a gate before those operations and fresh input selection after
waiting. Defer that more disruptive design unless the tail join proves inadequate.

## 8. Realtime placement and configuration

Use existing `MagAOXApp::threadStart`/thread readback conventions for priority
and cpuset setup. Provisional startup-only settings are `sdk.workerEnabled`
(default false), `sdk.threadPrio` and `sdk.cpuset`; final names can follow the
local controller convention during implementation. Changing execution mode
requires a restart/drain, not an online toggle.

The workstation config assigns the command monitor to `tweeter`, CPU40,
FIFO55. Synco uses CPU41. The
[RTC cpuset plan](rtc-cpuset-restructuring.md) retains `spare30`, `spare31`,
`spare56` and `spare57` as development reserves. Confirm current ownership and
topology before choosing one for the SDK worker; no core is assigned here.

Use a **different physical core** from the monitor for the intended overlap.
On one core a higher-priority worker can hold off the monitor during the SDK
call, while a lower/equal-priority FIFO worker can delay SDK entry until the
monitor blocks. Neither establishes the intended timing. Measure placement,
wakeup and SDK-entry costs and retain RTC latency tuning. A requested priority
or cpuset must not be mistaken for verified runtime placement: `threadStart()`
can log a scheduling-policy failure without returning failure. Verify actual
policy, priority and affinity before accepting a worker-enabled setup.

## 9. Implementation and qualification sequence

1. **Ownership and baseline tests.** Establish SDK version/transport reporting,
   reusable single-slot handoff and lifecycle drain tests. Keep worker mode
   disabled by default and preserve the synchronous baseline.
2. **Staged DM processing.** Separate channel snapshots from output publication
   with synchronous regression coverage. Add opt-in callback finalization so
   every post-dispatch error path drains the matching job.
3. **Worker and endpoints.** Dispatch at the current SDK call site, stage reads
   concurrently, publish the distinct release endpoint, join at callback tail
   and merge actual submission timing/outcome. Add lifecycle/control operations
   to the same SDK owner and preserve version-1 completion behavior.
4. **Shadow observation and documentation.** Measure release independently and
   match available completion/result endpoints by job identity in the
   reader-protection test, retaining separate coverage counts. Document
   configuration, diagnostics, expected counter/semaphore behavior and
   failure/recovery semantics. Update this plan
   with implemented decisions and verification results.
5. **Operator hardware comparison.** Run paired synchronous/worker captures
   with identical camera, loop, dmcomb, CPU/IRQ, latency and GPU workload settings.
   Compare primary/combined publication, camera timing, DM send cadence and all
   quality counters as well as earlier reader clearance. Retain the current
   shadow budget and do not enable extra real triggers in this qualification.

Deterministic workstation tests use a stub SDK with entry/return barriers,
not sleeps as ordering evidence:

- Hold `BMCSetArray()` in flight and prove the last channel reads and release
  occur before permitting SDK return; prove the callback cannot admit another
  command or overwrite its buffers meanwhile.
- Change shared pixels after release and verify the worker receives the frozen
  vector and staged delta products use the earlier channel sums.
- Exercise SDK return before/after staging, failure, exception and delayed
  return; preserve clipping/mapping/output-shape and success-dependent products.
- Assert at most one SDK operation, including zero/open/map/clear/close, is
  active. Exercise stop, power, reconfiguration and startup failures at each
  handoff phase without releasing live buffers or deadlocking.
- Exercise stale/duplicate/coalesced notifications, EINTR, source replacement,
  counter rollback, generation changes, backlog and older results after reset.
- Validate overlapping timing, actual submission independent of callback error,
  release qualification and version-1 compatibility. Test other controller
  regressions and default-off BMC behavior.

Hardware acceptance requires earlier qualified reader clearance **and** no
unexplained new SDK errors, stale/duplicate sends, counter gaps or camera/loop
timing degradation. Retain missing/dropped/unqualified coverage alongside
observed extrema. A useful short capture is followed by a longer unchanged
run; neither observed maximum is a hard realtime guarantee.

## 10. Alternatives and remaining review choices

A no-worker alternative is to snapshot the channels before the synchronous
SDK call and publish their release then. It avoids the extra thread but moves
roughly 10–16 us of work in front of SDK submission. Moving the existing
`makeDelta()` wholesale before `commandDM()` would also use the previous output
shape, so preparation and publication would still need separation. The worker
approach better preserves early SDK launch if handoff latency is small.

Before enabling the worker on hardware, resolve:

- Installed SDK version/transport and supported input/handle lifecycle contract.
- Dedicated worker core and priority, supported by measured dispatch latency.
- Final release-stream/configuration names and visibility of actual submission
  status when the old full-completion record is conservatively unqualified.
- Explicit overload/fault recovery behavior and acceptable measured overhead.

Selecting a smaller reader budget, an early-main reserve or a revised live
syncopation policy is subsequent work, based on the separated measurements.

## 11. 2026-10-28 implementation record

### Interfaces, ownership and lifecycle

Implemented the generic opt-in callback-tail path using inherited default
interfaces `admitCommand()`, `deferredCommand()`, `finalizeCommand(result)` and
`retireCommand()`. Synchronous owners admit normally and use a no-op finalizer.
Deferred owners stage delta/difference in preallocated private arrays, release
readers, finalize the actual submission, then publish products and notify the
saturation worker. A callback scope guard drains an unexpectedly unfinished
submission before retiring its storage. Ordinary processing exceptions follow
the same finalization path. No-delta callbacks also finalize and release readers.

The BMC implementation uses a single fixed request/result slot with mutex and
condition-variable predicates. All handle operations run on the sole worker
when enabled, including initialization, map loading, zero, clear and close.
An immutable SDK vector is copied separately from monitor saturation input.
Control requests close admission and wait for whole-callback retirement before
using the SDK slot or changing buffers. The worker initialization barrier is
mutex-protected, and startup checks actual policy/priority and requested cpuset
affinity before accepting work; the existing generic startup boolean is not
read concurrently by this worker.

Worker-mode admission rejects stale/duplicate counters and coalesces advances
without replaying semaphore backlog. The first same-inode counter rollback is
rejected and clears the progression baseline so a subsequent genuine receipt
can establish a new epoch. SDK, processing and late publication/notification
failures close admission; successful submission remains independent of a later
failed callback. Only release followed by successful reinitialization clears
the worker fault. Saturation reporting now uses an atomic exchange to retain
increments across the ordinary reporting thread.

The implementation waits for actual SDK return while enforcing a diagnostic
command deadline, `sdk.timeoutSec` (default **1 second**, finite and positive).
Ordinary app logic observes pending normal-command age; finalization also
checks actual elapsed time if the call returned between observations. A
violation is counted once per job and closes admission without releasing the
in-flight buffer/handle. Control/initialization operations are exempt. An SDK
success after the deadline retains its actual outcome/code but fails the
callback and suppresses success-dependent products. A hung call can still
block drain: there is no cancellation, concurrent close or fabricated
completion. This watchdog is not the scheduler's microsecond timing reserve.

### Release protocol, diagnostics and compatibility

`dm.readerReleaseStream` is independent of `dm.completionStream`. Both use
16-word records with distinct magic values. The new reader kind adds a
`released` state and uses word 15 for raw backend result (`INT64_MIN` when no
raw return code is available, including SDK exceptions). Read qualification
is independent of backend outcome. A later result update preserves the exact
read identity and last-read timestamp. Full version-1 encoding/qualification
remains unchanged. If the two publishers observe different admitted input
identities, the early record is conservatively unqualified.

The worker never mutates `dev::dmTiming`. Its result returns actual SDK
timestamps/code to the monitor for causal import. New metric IDs 13–17 are
`sdkDispatch`, `readerRelease`, `staging`, `publication` and `sdkWait`;
historical IDs and `fgtimes` meanings remain intact. The detailed logger accepts
complete 12-, 13- and 18-row metric sets. Updated controllers emit 18 rows even
in synchronous mode, so `logdump` must also be rebuilt for the baseline run.
Dispatch-object dependencies include the changed timing and reader-result
headers, preventing stale verifiers/accessors after a header-only change.

Synco's optional ordinary-priority `readerReleaseMonitor` feeds an independent
protocol-aware reader collector. Its `reader_release_timing` INDI/telemetry
population uses the same frozen budget and source observations as the existing
full-completion measurement. Valid release evidence requires no full-completion
peer. SDK enrichment is counted once per observed attempt and never creates
a second reader sample; an explicit outcome can be counted even when an
exception supplies no raw code. Missing/coalesced observations remain visible
in the independent quality counts. Scheduler decisions and the 155 us budget
are unchanged.

The [BMC worker guide](../../../apps/bmcCtrl/doc/sdk-worker.md), common
[completion protocol guide](../../../libMagAOX/app/dev/doc/dmCompletion.md),
[timing guide](../../../libMagAOX/app/dev/doc/dmTiming.md) and expanded
[shadow procedure](../../../apps/dmSyncopation/doc/reader-protection-test.md)
describe configuration and interpretation. No worker core or instrument config
has been selected by this implementation.

### Local verification

Deterministic SDK barriers establish early release while `BMCSetArray()` is
held, immutable command/delta capture after shared input changes, no SDK overlap,
and control-operation exclusion through callback retirement. Fault tests cover
SDK errors/exceptions, processing exceptions, counter resets, stale notifications,
startup rollback and a real saturation `sem_post` overflow after SDK success.
The last case must close admission while preserving the successful immutable
SDK result.

Final workstation verification:

| Suite | Assertions | Cases |
| --- | ---: | ---: |
| BMC controller/worker | 583 | 15 |
| Generic DM | 274 | 5 |
| Callback/finalization/completion integration | 293 | 9 |
| Timing integration | 153 | 8 |
| Completion/release protocol and publisher | 539 | 15 |
| Timing collector | 380 | 18 |
| Reader-protection collector | 347 | 18 |
| Synco application integration | 8,383 | 65 |
| DM timing telemetry | 1,218 | 7 |
| Synco telemetry | 4,283 | 25 |
| Existing ALPAO/IrisAO/PI controller regressions | 109 | 6 |
| **Total** | **16,562** | **191** |

The library/PCH, normal `dmSyncopation` and normal `logdump` builds pass with
fresh dispatch and binary schemas. Final BMC source also passes compilation
against the real archived **3.5.3 SDK headers**, independently of the offline
test declarations. A normal BMC vendor-library link was not performed: that
runtime installation is absent on the workstation. No SDK or controller was
run against hardware. Full changed C/C++ files pass `clang-format`, whitespace
checks pass, and new/updated guide paths and Markdown anchors resolve.

Affected implementation files are the BMC app/test SDK adapter, generic DM
base, shared completion/timing/reader collectors, synco observer, detailed
telemetry/schema types, and their existing test suites. The library Makefile
adds dispatch dependencies for timing headers. The existing test registry and
Doxygen group structure cover the expanded suites without new test binaries.

Hardware timing, installed SDK/transport and dedicated-core qualification
remain pending. Rebuild/install `bmcCtrl`, `dmSyncopation` and `logdump` together
before a synchronous-baseline/worker comparison. Keep operational syncopation
disabled and the existing shadow protection budget unchanged for that test.
