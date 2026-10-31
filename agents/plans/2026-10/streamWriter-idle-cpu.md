# streamWriter idle CPU investigation

Prompt: "apps/streamWriter has very high usage when idle. I think it's after it
has been toggle to write, then toggle off, possibly when nothing was written in
the mean time. Please analyze and document any findings under agents/plans"

Reviewed: 2026-10-22, local `dev` at
`cd37915d2b9bfc4f5261b5226882ff905ee9fb92`.

Status: The user approved implementation on 2026-10-22. The changes below are
implemented and validated on `jrmales/streamWriter-idle-cpu`.
All inspection, builds, and tests ran on the local workstation. No instrument
access or INDI traffic was used.

Follow-up, 2026-10-26: the user confirmed idle CPU usage stayed down, then
reported XRIF `-100` alerts around producer restarts. The separate
[datatype compatibility investigation](streamWriter-xrif-datatypes.md) records
that reproduction and correction.

## Implementation record (2026-10-22)

The original investigation and regression assessment below describe the reviewed
base revision, not the corrected code. The implementation covers all six items
in the approved plan.

- **Empty recordings:** `m_chunkFrames` counts accepted, unsaved recording frames
  independently of idle ingestion and the START state. On followed by Off with
  no accepted frame settles immediately to idle without publishing a save. If a
  previous time flush is pending, Off marks that existing save as the final work
  and lets it finish exactly once. The next frame still completes an ordinary
  nonempty stop, or the configured stop timeout flushes the existing remainder.
- **Writer ownership:** `m_writeMutex` serializes callbacks, chunk bookkeeping,
  and descriptor publication. A pending descriptor and its circular-buffer
  region remain owned until compression, output, and timestamp use are finished.
  The producer waits on a condition variable before replacing the descriptor or
  wrapping into its owned region. Compression and disk I/O run without this
  mutex. A delayed consumer therefore causes backpressure rather than additional
  semaphore posts carrying mutable work. The existing five-second completion
  timeout now also bounds these producer waits; expiration requests shutdown
  while retaining buffers until the writer is joined.
- **Completion and failure:** A stop is published once, and stale semaphore wakes
  cannot encode completed data again. Failed encoding releases ownership and
  prevents the producer from publishing further work to the failed worker.
  Off during a reconnect flush cancels the pending automatic recording resume.
  Immediate stops retain the stop log and state telemetry, including when the
  last full chunk already finished; repeated Off commands do not duplicate them.
- **Encoder guards:** Before copying, `doEncode()` rejects reversed or
  out-of-buffer indices, excess frame counts, zero or overflowing frame sizes,
  missing buffers, and insufficient image/timing encoder capacity. Both XRIF
  sizing calls must succeed before either destination is copied. Valid empty
  ranges and ranges ending exactly at circular-buffer capacity remain supported.
  Unrelated legacy XRIF warning and short-write behavior is unchanged.
- **Reconnect and shutdown:** Normal source metadata/path/inode checks remain.
  StreamWriter no longer installs SIGSEGV/SIGBUS handlers or keeps a handler
  instance pointer. Source readiness is published only after allocation,
  semaphore flushing, and frame-counter baselining. Reconnect drains pending
  and unsaved data before freeing buffers. During shutdown the writer remains
  available until the producer has published its final chunk and stopped.
- **Configuration and statistics:** Zero or out-of-range nanosecond semaphore
  waits are rejected; the default remains 500 ms. INDI statistics use a snapshot
  from completed encoding instead of reading a live encoder. Saving-state
  telemetry takes a synchronized snapshot and keeps deduplication per instance.

The existing app header-only layout and test fixtures are retained. Modified
C++ files received a documentation pass and clang-format; lock-lifetime blocks
use the repository's required `//mutex scope` annotation.

### Regression coverage added

| Behavior | Permanent regression |
| --- | --- |
| Empty On/Off after idle ingestion | Fault tests drive real callbacks/producer code after 0, 1, 3, 5, or 7 idle frames, selecting timeout and next-frame paths; no work or archive is allowed. |
| Empty session after recording and buffer wrap | Decode the previous valid archive and require no additional data from the empty session. |
| Off after time-flush rearming | Hold the queued save through repeated timeout events and require one publication and one decoded copy of the recorded frame. |
| Delayed or failed consumer | Preserve the original descriptor on timeout; allow a controlled delayed consumer to release it and decode both chunks; reject further publication after encoder failure. |
| Buffer reuse while writing | Attempt circular-buffer wrap while the writer owns the destination range; require bounded failure before overwrite and decode the original frames. |
| Invalid ranges and sizing failures | Check indices, allocated capacities, missing buffers, zero geometry, and byte-count overflow; require failure before copying, encoding, or output. |
| Normal signal disposition | Bounded subprocesses run actual startup, inspect SIGSEGV/SIGBUS dispositions, and require genuine invalid/truncated mapping accesses to terminate with the corresponding signal. |
| Sustained idle after empty toggles | Both real workers run against a temporary quiet stream after 0 or 3 idle frames; three empty sessions produce no archive and consume less than 0.15 CPU seconds per worker during a 0.6-second observation. |
| Source replacement | Both real workers handle same-geometry deletion/recreation while idle and active; decoded counters preserve the old pending frames and replacement data. |
| Shutdown flush | After active replacement, shutdown without Off must preserve the final unsaved replacement frame. Another test holds the real writer at its normal readiness gate, ingests one queued chunk plus a partial next chunk, and requires both archives to survive shutdown. |

### Validation results

The application and all five streamWriter test executables were freshly built
with `COVERAGE=1`. Source rebuilds were forced with `-W` because the single-test
makefile does not track every included app-header change. All normal runs passed:

| Executable | Test cases | Assertions |
| --- | ---: | ---: |
| `streamWriter_test` | 6 | 124 |
| `streamWriter_fault_test` | 18 | 1102 |
| `streamWriter_lifecycle_test` | 8 | 195 |
| `streamWriterSizing_test` | 1 | 34 |
| `streamWriter_entrypoint_test` | 1 | 11 |

Lifecycle counts above are parent-process assertions; subprocess checks also
passed, including CPU limits, decoded archives, fatal signals, and shutdown
deadlines. The entrypoint build also rebuilds the real application and checks its
help/configuration interface. Build and invocation pattern, from `tests/`:

```sh
make -f Makefile.one t=../apps/streamWriter/tests/streamWriter_fault_test.cpp -W ../apps/streamWriter/tests/streamWriter_fault_test.cpp COVERAGE=1
GCOV_PREFIX=/tmp/streamWriter-validation-gcov timeout 120s ../apps/streamWriter/tests/streamWriter_fault_test
```

The complete fault suite also passed **18 cases / 1102 assertions under
AddressSanitizer and UndefinedBehaviorSanitizer**, with no sanitizer reports.
A temporary translation unit included the actual repository fault test source;
`CXX='g++ -fsanitize=address,undefined -fno-omit-frame-pointer'` was supplied to
`Makefile.one` with `COVERAGE=1`. Execution used
`ASAN_OPTIONS=detect_leaks=0:abort_on_error=1` and
`UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1`. LeakSanitizer cannot run under
this environment's ptrace sandbox, so leak detection was disabled. Existing
prebuilt test support and dependency libraries were not rebuilt with sanitizers;
the test translation unit, including the production streamWriter header, was
instrumented. No thread-sanitizer claim is made.

The ordinary unit suite passed its 124 assertions under sanitizers but failed in
existing logger teardown: ASan reports sized-delete mismatch for the variable-size
`shared_ptr<char>` allocation in `flatlogs/include/flatlogs/logHeader.hpp`; suppressing only that
diagnostic exposed a UBSan misaligned FlatBuffers read in the logger thread. The
full unit sanitizer run is therefore **not clean**. These independent logger
findings are deferred, rather than weakening more checks or changing logger code
in this fix.

Final logs are disposable artifacts under `/tmp`:
`streamWriter-implementation-{fault,unit}-final2.log`,
`streamWriter-lifecycle-pending-shutdown.log`,
`streamWriter-final-{sizing,entrypoint}.log`, and
`streamWriter-sanitizers-20261022/{fault-run-final,unit-run,unit-run-limited}.log`.
All touched C++ files were clang-formatted and `git diff --check` passed. A
separate documentation change places the streamWriter test group definition in
`tests/groups.dox`, under `application_unit_test`.

### Remaining limits

- The reported instrument process was not sampled. The stale ranges and signal
  loop mechanism were reproduced separately; attributing the original incident
  still requires user-run diagnostics if it recurs.
- Off continues to ingest and copy incoming source frames. High source rates can
  therefore consume CPU while recording is idle, as before.
- Backpressure can skip source frames when compression/storage cannot keep up;
  existing source-counter backlog reporting remains in place. The application
  does not gain an unbounded work queue.
- The producer's ownership waits are bounded, but joining a writer blocked inside
  filesystem I/O is not. Buffers cannot safely be freed while that writer may
  still access them.
- Same-inode truncation and unrelated invalid accesses remain real memory faults;
  they now follow normal process signal handling. Inode checking is not a
  recovery mechanism for arbitrary mapping faults.

## Main finding and confidence

The reported sequence can produce an invalid save window. Turning writing on
and off before the next frame arrives skips initialization of the new chunk's
start position. The subsequent stop uses old indices, which can include frames
captured while idle, exceed the encoder capacity, or underflow the frame count
after circular-buffer wrap. These save-window defects were reproduced using the
current production framegrabber loop and callbacks through the existing
deterministic fault-test harness.

A separate local diagnostic confirmed a mechanism for sustained CPU consumption:
the installed SIGSEGV/SIGBUS handler only sets a restart flag and returns. It
does not repair or escape a persistent synchronous memory fault. An invalid copy
can therefore lead to repeatedly executing the faulting instruction and signal
handler. The diagnostic consumed approximately one CPU core until its one-second
alarm terminated it.

This is the strongest explanation found for CPU usage triggered by an apparently
empty recording session, but the actual reported process has not been profiled.
The complete chain from the user's toggle sequence to a faulting writer thread
has not been reproduced in a running application. The confirmed save-window
defects warrant correction independently of that remaining attribution.

## Findings

All line numbers below refer to the reviewed source. Unless otherwise specified,
the file is [streamWriter.hpp](../../../apps/streamWriter/streamWriter.hpp).

### 1. An empty recording session is not necessarily an empty save window

- Every new source frame is copied into the circular buffer before the writing
  state is checked (1573–1618). The destination index advances even in
  `NOT_WRITING` (1737–1740).
- The On callback changes only the writing state and reconnect/deadline fields
  (2256–2260). Chunk start is initialized by the framegrabber's `START_WRITING`
  case, after a new frame arrives (1628–1644).
- Off accepts `START_WRITING` and changes it to `STOP_WRITING` (2248–2253).
  Thus On followed by Off without an intervening frame skips chunk initialization.
- Both stop paths use `m_currChunkStart` without checking whether it belongs to
  the current recording session: the next-frame path at 1710–1723 and the timeout
  path at 1779–1794.
- Successful ordinary stop completion returns to `NOT_WRITING`, but leaves the
  old chunk start in place (2217–2233).

Examples with a four-frame circular buffer and a two-frame write chunk:

| Precondition | On/Off with no intervening frame, then timeout | Consequence |
| --- | --- | --- |
| No frames acquired since connection | Save `[0,0)` | Existing empty-write cleanup succeeds. |
| One frame acquired while idle | Save `[0,1)` | Writes a frame acquired before recording was requested. |
| Three frames acquired while idle | Save `[0,3)` | Exceeds the two-frame encoder capacity. |
| Previous stop saved `[2,4)` and destination wrapped to zero | Save `[2,0)` | Unsigned frame count becomes `18446744073709551614`. |

All four cases were reproduced locally. The one-frame case created an archive
through `doEncode()`, confirming that this is observable behavior rather than
only stale bookkeeping. The invalid three-frame and wrapped windows were
inspected before calling the encoder; no unsafe buffer copy was executed.

"Nothing written" therefore differs from "no frames have been acquired since
connection." A slow source or long exposure makes it easy to toggle between
frames even after the application has already ingested idle frames.

### 2. The encoder trusts the invalid range before copying

`allocate_xrif()` sizes image and timing encoders for `m_writeChunkLength`
(1194–1206, 1219–1231). `doEncode()` instead calculates
`m_currSaveStop - saveStart` using unsigned indices without checking ordering,
circular-buffer bounds, or allocated capacity (2012–2016).

The zero-frame guard at 2022–2047 handles equal indices only. XRIF size errors
are logged but execution continues (2050–2055, 2067–2072). The subsequent copies
use the unchecked frame count (2064, 2088). A library error check cannot protect
an application-side `memcpy` that still executes afterwards.

In the local three-frame diagnostic, the timing buffer had 80 allocated bytes
and the queued range required 120 bytes. This establishes an actual capacity
violation, regardless of any spare capacity in the compressed image buffer.
Depending on the size and placement of allocations, an invalid range can cause
memory corruption or a synchronous fault; a small overrun need not fault
immediately.

### 3. Returning from a persistent memory fault can consume a CPU core

`setSigSegvHandler()` installs process-wide SIGSEGV and SIGBUS handlers
(1002–1034). The trampoline calls `handlerSigSegv()`, which only sets
`m_restart = true` and returns (1039–1052). It does not distinguish a source
mapping fault from an invalid writer-buffer access.

For a persistent fault, ordinary execution cannot reach the framegrabber's
restart check: returning retries the faulting access. The writer loop itself
does not check `m_restart` either. This explains how a memory error can appear as
a process that remains alive and uses high CPU instead of exiting visibly.

The bounded local signal diagnostic and its measurements are recorded under
Validation below. It tests the handler mechanism separately from invalid save
windows; it is not an end-to-end reproduction of the reported incident.

#### Follow-up: remove the obsolete stream-restart signal handler

On 2026-10-22 the user noted that inode-change detection should make this handler
unnecessary. Inspection supports removing it for normal stream deletion and
recreation. The framegrabber detects a cleaned-up source (1808–1810), a missing
path or failed `stat` (1827–1843), and a changed inode (1846–1851), then follows
the existing flush/close/reopen path (1856–1918). Successful-frame processing
also checks dimensions and datatype (1515–1529).

The inspected local ImageStreamIO source,
`/home/jrmales/Source/milk-xwcl/src/ImageStreamIO/ImageStreamIO.c` at
`21a8ae160f16a934a287ba82d14810043c19f47c`, creates a new file exclusively and
unlinks an existing name before retrying (1426–1444). Its destroy path closes and
unmaps the producer's mapping and removes the pathname (1623–1674); it does not
truncate the consumer's existing mapped file. Normal unlink/recreate therefore
leaves the old consumer mapping alive until that consumer closes it. This is
consistent with the existing reconnect design and does not require a signal
handler. This source inspection does not establish the deployed library version.

Inode detection is a reconnect mechanism, not protection against all memory
faults: it runs on the failed-wait path, and same-inode truncation can still
invalidate mapped pages. The current returning handler cannot recover that
case either. The recommended change is to remove streamWriter's SIGSEGV/SIGBUS
installation, handler functions, and handler-only static instance pointer while
retaining `m_restart` and normal reconnect logic. Leave genuine memory faults to
the normal process signal disposition. The save-window and ownership fixes
remain necessary; handler removal alone would turn a fault loop into process
failure without preventing the invalid access.

### 4. Stop requests can overwrite work that is already pending

Both stop paths set `m_writePending` and post the writer semaphore without first
checking whether a descriptor is already owned by the writer (1710–1729,
1779–1801). They leave the state at `STOP_WRITING` and clear the stop deadline to
zero. Until `doEncode()` changes the state, subsequent frames or timeouts can
post again and overwrite `m_currSaveStart`, `m_currSaveStop`, and the frame number.

The writer consumes a counting semaphore but reads a single mutable descriptor.
`m_writePending` is an atomic boolean, not a queue or protection for these fields.
Copying the fields into locals at 2012–2015 does not make the multi-field snapshot
atomic. `m_writing`, the indices, and the stop deadline are otherwise shared
between callbacks/workers without a common synchronization protocol.

This is an additional correctness issue, particularly with a slow encoder or
disk. It can produce extra work or inconsistent ranges, but is not by itself
proof of a permanent idle CPU loop at the default wait interval.

### 5. Other idle work and alternative CPU explanations

- Writing Off does not stop ingestion: image copies and timing updates continue
  for every new frame. High-rate or large images can consume substantial CPU
  without any disk writing, before or after a toggle.
- The default semaphore wait is 500 ms (92–96). Both worker loops rebuild the
  deadline each iteration (1487–1490, 1964–1974), so an expired deadline reused
  forever is not evident here. The common helper in
  [semUtils.hpp](../../../libMagAOX/app/semUtils.hpp), lines 20–27, also takes a
  fresh timestamp. A true initial `[0,0)` stop returns to idle correctly.
- `framegrabber.semWait` is loaded without validation (614). Zero removes the
  intended blocking delay from both workers; very small values increase wake
  frequency. The framegrabber also opens/closes/stats the source path after each
  timeout (1827–1846). This explanation depends on configuration and would not
  normally require a prior toggle.
- Missing-source retries sleep one second (1335, 1348, 1375), and restart
  completion polling sleeps 50 ms with a deadline (1255–1283). Neither is an
  obvious tight loop at defaults.
- Recent telemetry changes do not introduce an obvious toggle-specific spin.
  `c56f05f9` enabled telemetry scheduling in READY; the default scheduler interval
  is 10 seconds in [telemeter.hpp](../../../libMagAOX/app/dev/telemeter.hpp).
  `07b1210e` and `32d899c3` change telemetry snapshots/deduplication, not semaphore
  waiting. The empty-window encoder guard dates to `7295ec8a` (2025-03-16);
  the stop deadline and reconnect bookkeeping were added by `4396c450`
  (2026-05-22).

## Validation and test gaps

1. Ran the existing `streamWriter_test` executable filtered to
   `streamWriter writing toggle transitions and stopped writes`: **27 assertions
   passed**. This was an existing local binary, not a fresh rebuild. Its empty
   case explicitly supplies equal save indices before invoking `doEncode()`
   ([streamWriter_test.cpp](../../../apps/streamWriter/tests/streamWriter_test.cpp),
   690–712), so it does not exercise how the framegrabber constructs the window.
2. Built a temporary probe against the current source, reusing the synchronous
   `Fixture` and dependency wrappers from
   [streamWriter_fault_test.cpp](../../../apps/streamWriter/tests/streamWriter_fault_test.cpp).
   The production callbacks and `fgThreadExec()` generated the four ranges in
   the table above: **54 assertions passed in two diagnostic cases**. These
   assertions verify the defects as observed; they are not regression tests
   expecting corrected behavior. Only valid windows were passed to `doEncode()`.
3. Existing lifecycle scheduling tests exercise stop after active capture and
   inspect semaphore work, but use a harness that does not launch the writer
   ([streamWriter_lifecycle_test.cpp](../../../apps/streamWriter/tests/streamWriter_lifecycle_test.cpp),
   225–240, 916–993). They do not establish that a no-frame session with stale
   indices is safe or that a slow consumer cannot accumulate repeated stop work.
4. Built a separate bounded subprocess probe using the actual production signal
   trampoline installed through real `sigaction`. One store into a `PROT_NONE`
   page consumed **1.0026 CPU seconds in 1.00591 elapsed seconds** (about 99.7% of
   one core). The child never advanced past the store and terminated on its
   one-second SIGALRM as designed: **9 assertions passed**. This verifies the
   signal-loop mechanism without running an invalid encoder copy. Source and
   results are `signal_probe.cpp` and `signal_probe.log` in the same temporary
   directory below.
5. Following the handler-removal discussion, ran the existing lifecycle binary
   in a fresh process with only the `replaced shmims flush the pending chunk and
   reconnect ready to keep writing` section selected. Its direct framegrabber
   harness never installs streamWriter's signal handlers. The first run passed
   reconnection checks but failed waiting for the first replacement frame at
   line 1117; one retry passed all **21 assertions**. A likely test readiness
   race is that the test publishes once dimensions match (1109–1115), while the
   worker sets dimensions before its post-allocation semaphore flush and frame
   counter baseline (1414–1478). This explanation is inferred, not separately
   reproduced. Make acquisition readiness explicit when extending this test.
   The passing run supports ordinary replacement without the handler, but does
   not distinguish inode detection from the earlier metadata cleanup check;
   writer completion is simulated by this harness.
6. Ran the existing fault-test binary filtered to `every timed wait error
   releases the image before reopening`: **60 assertions passed**, including
   synthetic missing-path, failed-stat, cleanup, and changed-inode events. This
   separately exercises the inode branch without installing real handlers.

Temporary source and results are at
`/tmp/streamWriter-idle-analysis-20261022/probe.cpp` and `results.log`.
The probe used synthetic stream dependencies, direct in-process callbacks, and
temporary output files. Its stop timeout is zero to select the timeout path
deterministically, not to measure normal idle CPU. Compile and invocation:

```sh
cd /home/jrmales/Source/MagAOX-working/tests
make -f Makefile.one t=/tmp/streamWriter-idle-analysis-20261022/probe.cpp COVERAGE=1 /tmp/streamWriter-idle-analysis-20261022/probe
GCOV_PREFIX=/tmp/streamWriter-idle-analysis-20261022/gcov-recheck timeout 20s /tmp/streamWriter-idle-analysis-20261022/probe '[idle-diagnostic]'
```

These `/tmp` artifacts are disposable; the event sequences and observations
above are the durable record. Temporary C++ was clang-formatted. At this analysis
stage the repository change was Markdown only; implementation followed approval.

## Regression coverage assessment (2026-10-22)

The existing suite is a useful starting point, but is not yet sufficient to
close either the handler-removal change or the full idle-CPU defect. These have
different minimum regression requirements:

| Scope / behavior | Current coverage | Required addition or correction |
| --- | --- | --- |
| Handler removal: missing stream and changed inode cause reopening | Deterministic fault tests cover these branches; real replacement test succeeds without the streamWriter handler. | Preserve these tests and fix the real test's readiness synchronization so passing does not depend on retrying. |
| Handler removal: startup preserves process fault handling | Signal tests currently expect handler installation or call the handler directly. The executable smoke test only runs `--help`, before worker startup. | In a bounded subprocess, exercise actual app startup and verify SIGSEGV/SIGBUS dispositions remain unchanged. Replace obsolete handler-specific expectations. |
| Handler removal: deletion/recreation while idle or actively saving | Real replacement test exercises active capture but manually acknowledges the absent writer. | Exercise idle replacement and replacement with pending data using both real workers. Include unchanged geometry to distinguish replacement from a size change; verify replacement ingestion and decoded pending output without manually changing ownership/state. |
| Idle-CPU fix: On/Off before a new frame, including after idle ingestion or wrap | Empty-stop unit tests supply equal indices; temporary diagnostic probes reproduce the defects but are not repository regressions. | Add permanent tests driving the callbacks and framegrabber through both timeout and next-frame stop paths. Require no archive for an empty session, bounded return to idle, and valid indices after wrap. |
| Idle-CPU fix: stop while a time-flushed chunk is queued | Time-flush scheduling and encoding have separate tests. | Exercise START re-arming with a pending write, then Off; require queued frames to be preserved exactly once. |
| Idle-CPU fix: invalid range cannot reach a buffer copy | No guard regression covers the demonstrated oversized or reversed window. | Test reversed indices, out-of-buffer indices, and counts beyond encoder capacity; require rejection before copy/encode/output. Preserve valid empty windows and exclusive end-at-capacity cases. |
| Idle-CPU fix: slow writer cannot receive overwritten or repeated stop work | Scheduling tests inspect notifications or simulate completion. | Hold the consumer at a controlled boundary while supplying more frames/timeouts; verify descriptor ownership, bounded notifications, and exact final output. |
| Idle-CPU fix: settled idle does not spin | No sustained per-thread idle regression was identified. | Run both workers against a quiet temporary source through On/Off, then observe bounded CPU/wake activity and shutdown in a subprocess with a hard timeout. |

For handler removal alone, complete the first three rows and rerun the relevant
startup/reconnect suites. The remaining rows are the acceptance tests for the
save-window and idle-CPU fixes; they should not be confused with prerequisites
for the smaller handler-only change. Reuse the existing fixtures rather than
creating another broad harness. Convert the diagnostic event sequences into
tests asserting corrected behavior; do not keep assertions that merely confirm
the buggy ranges. Use explicit worker acknowledgements for scheduling tests,
and keep the CPU check separate with a generous threshold to avoid fragile
wall-clock timing assertions.

Some existing tests explicitly expect encoding to continue after a reported
XRIF sizing error (`streamWriter_test.cpp`, 908–924, and the generated sizing
cases in `streamWriter_fault_test.cpp`, 1565–1599). If failed sizing becomes a
hard prerequisite failure as proposed, update those expectations and verify
that neither image nor timing sizing failure reaches copying, encoding, or
writing. Keep unrelated warning behavior tests scoped to their existing
contracts. The fault harness's `swPost()` currently discards notifications;
the delayed-consumer regression will need notification counting or the real
semaphore fixture to detect duplicate publication.

This pre-implementation assessment is based on source review and the executions
recorded above. The implementation record at the top of this document describes
the subsequent regression additions and validation.

## Proposed implementation and regression plan

1. **Make save-window validation unconditional.** Before subtraction, XRIF size
   changes, pointer arithmetic, or copying, enforce
   `start <= stop <= m_circBuffLength`, a frame count within the allocated write
   capacity, and checked byte counts for both encoders. Reject invalid work and
   report a controlled error. Preserve valid zero-length stop completion and
   the exclusive end index equal to the circular-buffer length. Do not continue
   to copy after a prerequisite sizing error.
2. **Represent empty sessions explicitly.** Track whether the current session
   owns any unsaved frames independently of `START_WRITING`. An On/Off sequence
   without an accepted frame must not write old idle data. Do not simply change
   every `START_WRITING` stop immediately to `NOT_WRITING`: START also means
   re-armed after a time flush (1705, 1775), and `doEncode()` drops work when it
   sees NOT_WRITING (2004–2007). Preserve already queued valid data.
3. **Make stop publication and completion happen once.** Define ownership for a
   stable save descriptor and its source-buffer region; prevent descriptor
   replacement or buffer reuse while the consumer owns them. Serialize state
   transitions and callbacks with that ownership protocol. Do not hold locks
   across compression or disk I/O. Define behavior when another chunk becomes
   ready before the writer finishes, rather than using additional semaphore
   posts with one mutable descriptor.
4. **Remove the obsolete SIGSEGV/SIGBUS restart handler.** Remove its startup
   registration, declarations/definitions, `m_selfWriter` storage/initialization,
   obsolete comments, and handler-specific tests. Retain `m_restart` and the
   existing metadata/path/inode reconnect logic. Verify normal deletion and
   recreation in idle and active states without installing the app's handlers.
   Genuine memory faults should follow the normal process signal disposition;
   no replacement recovery handler or jumps across C++ frames are needed.
5. **Validate timeout configuration.** Reject a zero semaphore wait, or document
   and explicitly opt into polling if there is a real supported use case.
   Keep the existing default. Treat skipping idle image copies as a separate
   performance decision because it changes the current ingestion behavior.
6. **Add focused regressions using the existing fixtures.** Cover no frames ever
   seen; idle frames before the first On/Off; stop before the first new frame;
   stop after wrap; stop after a time flush has re-armed START; a frame arriving
   concurrently with stop; delayed writer completion; repeated toggles; source
   reconnect; and shutdown. Require bounded stop completion, valid ranges,
   preservation of queued real data, and no archive for an empty session.
   Add a bounded subprocess check that startup leaves SIGSEGV/SIGBUS handling
   unchanged, and an isolated real-worker idle check measuring per-thread
   CPU/wakeup counts after stop. Make the replacement-stream test wait for actual
   acquisition readiness rather than dimensions alone. Use sanitizer validation
   for the save-window and buffer-ownership paths.

Reuse `waitForWriteCompletion()` and the existing test fixtures where their
contracts fit. The mutex/condition-variable worker acknowledgement in
[modalPSDs.hpp](../../../apps/modalPSDs/modalPSDs.hpp), lines 644–646 and
903–919, is a local ownership precedent, not a drop-in save queue. No new shared
base class is needed merely to fix this app's state machine.

## Follow-up needed to attribute the observed CPU usage

The remaining evidence is the deployed revision/configuration and a sample of
the affected process. If it recurs, the user should record the hot thread,
whether source frames are still arriving, actual versus requested writing state,
save activity, and configured `framegrabber.semWait`/`writer.stopTimeout`.
A short user-run stack or signal trace can distinguish repeated SIGSEGV/SIGBUS
from image copying, encoder work, or immediate semaphore timeouts. Logger and
main-loop pause settings are secondary configuration checks if those threads
are hot.

Any steps on instrument computers, including INDI queries through a tunnel,
must be performed by the user under `AGENTS.md` rule 0. No live diagnosis or
deployment is part of this analysis.
