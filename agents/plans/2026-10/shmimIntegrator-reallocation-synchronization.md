# shmimIntegrator reallocation synchronization

Issue: [MagAOX #220](https://github.com/magao-x/MagAOX/issues/220)

Prompt: Have a look at issue #220 and document findings and a plan under
`agents/plans`.

Reviewed: 2026-10-18, local `dev` at
`96db7e5664187cbfa58321035f84e011c6058cbc`.

Status: Plan approved on 2026-10-18. The user accepted all proposals and
recommendations: "I agree with all of your proposals/recommendations".
Implementation completed on 2026-10-18 on
`jrmales/shmimIntegrator-reallocation`. See the implementation record below.
The issue was open and had no comments when reviewed. Findings below come from static inspection; the reported race has not
been reproduced at runtime.

## Problem and intended result

The issue reports that `configureAcquisition()` and the three `allocate()`
callbacks use different mutexes, and requests coordinated restart and
synchronization whenever allocation occurs. The mismatch is still present, but
using the same mutex in those four functions would not by itself fix the problem.
Processing, publication, configuration updates, and loading saved images also
participate in the lifetime of the affected buffers.

The intended result is that every allocation starts a coordinated configuration
transition. No callback may use storage being resized, no output copy may combine
dimensions from different configurations, and processing must resume with a
consistent configuration after the output worker acknowledges it. Keep ordinary
averaging and the existing optional-dark behavior intact.

## Findings

Line numbers refer to the reviewed commit. Paths below are relative to this plan.

### 1. Allocation and output publication have no shared lifetime protection

In [shmimIntegrator.hpp](../../../apps/shmimIntegrator/shmimIntegrator.hpp):

- Source `allocate()` takes `m_darkMutex`, resizes `m_accumImages` and `m_avgImage`,
  resets accumulation counters, and finally sets `m_reconfig` (lines 690–769).
- Both dark allocations take `m_darkMutex` (946–975 and 994–1023).
- `configureAcquisition()` instead takes `m_indiMutex` and copies the source
  monitor's dimensions into the frame-grabber dimensions (1128–1147).
- `loadImageIntoStream()` takes neither lock. It reads `m_avgImage.data()` and
  computes the copy length from the *current source monitor* dimensions, rather
  than the dimensions used to allocate its destination (1184–1188).

Consequently, a source resize can invalidate the pointer being copied, or a larger
source can cause a copy beyond the old destination frame. A smaller source can
leave old pixels in a larger output. These are possible interleavings inferred
from the code, not observed crashes from this investigation.

### 2. Base-class metadata changes precede the app's allocation lock

[shmimMonitor.hpp](../../../libMagAOX/app/dev/shmimMonitor.hpp), lines 648–675,
sets each monitor's datatype, width, and height before invoking the derived
`allocate()`. An app mutex acquired inside that callback cannot protect those
earlier assignments from another thread reading the base members.

`configureAcquisition()` already has a TODO acknowledging this issue. Cross-thread
users need an app-owned, committed descriptor, copied by the relevant monitor's
own callback. The output worker and the other monitors must not infer buffer
shape from another monitor's live base members. Likewise, the app must stop
writing source-monitor dimensions from its main thread.

Each monitor calls its own `allocate()` and `processImage()` serially. Thus a dark
callback does not race with its *own* allocation within that worker, but it does
race with the source worker consuming its pixels and validity flags.

### 3. Restart flags are requests, not acknowledgements

[frameGrabber.hpp](../../../libMagAOX/app/dev/frameGrabber.hpp) exposes several
important boundaries:

- `configureAcquisition()` returns before output stream creation/recreation
  (856–940). The base clears `m_reconfig` afterwards, at line 943, then calls
  `startAcquisition()`.
- The inner loop checks `m_reconfig` before calling `acquireAndCheckValid()`
  (965–970). It does not recheck between a successful acquisition and
  `loadImageIntoStream()` (983–986).
- `reconfig()` runs after leaving that loop, conditional on `m_reconfig`
  (1066–1069). The integrator's implementation currently does nothing.
- Both `frameGrabber::m_reconfig` and `shmimMonitor::m_restart` are plain `bool`
  members (149 and 224 respectively), although the integrator writes them from
  other threads.

For example, the output thread can pass its loop check and wait on the semaphore;
the source can then resize and request reconfiguration; the output thread can
wake and copy into its old stream without acknowledging the request. A request
arriving during output setup can also be overwritten by the base's clear at line
943. Merely making the flags atomic would not fix either sequencing problem.

### 4. The producer/consumer handoff is incomplete

The source sets `m_updated` before posting `m_smSemaphore`; the output thread reads
the flag and clears it after copying. The source also reads it to avoid overwriting
an unread result (781, 907). These reverse-direction accesses are not protected by
a common lock or atomic ownership protocol. Posting a semaphore does not alone
establish safe ownership for the entire round trip.

Source allocation resets counters but does not clear `m_updated` or discard old
semaphore notifications. A queued notification can therefore outlive the image
configuration that produced it. Both block and moving-average paths need the same
defined ownership and restart rules.

### 5. Dark images have races and independent validity defects

The dark `processImage()` overloads write image pixels and set `m_darkSet` or
`m_dark2Set` without `m_darkMutex` (977–1039). Source processing tests the flags
before acquiring the mutex for subtraction (797–810, 925–928). The current lock
does not prevent a partially updated dark from being subtracted.

All four dimension checks use width equality **or** height equality (749, 758,
965, 1013). Both dimensions must match. A dark with only one matching axis is
currently marked valid and can reach incompatible Eigen arithmetic. Successful
dark reallocation also fails to clear its previous `Set` flag, although the image
has just been resized and zeroed. Clear readiness until a complete frame has been
copied for that allocation.

The simple-average branch subtracts either or both darks; the moving-average
branch only subtracts the first dark. This is a separate behavioral discrepancy.
Preserve that behavior in the synchronization change unless explicitly expanded.

### 6. Disk loading and INDI callbacks must join the same protocol

`appLogic()` holds `m_indiMutex` while invoking `findMatchingDark()` (608–643).
That function reads a FITS image directly into `m_avgImage`, checks dimensions
only afterwards, and can write source-monitor dimensions after a one-second
sleep (1042–1125). It can replace the output buffer while the output worker copies
it. A mismatched file can already have resized the live buffer before it is
rejected. This path also supports publishing a saved image before the input
stream has connected; preserve that capability without fabricating monitor state.

The `nAverage`, `avgTime`, and `nUpdate` callbacks change active configuration and
then request a source restart (1196–1285). An active moving-average callback can
observe a new count with an old cube allocation. The FPS callback writes both
base restart flags (1327–1353). Start/stop and state-string callbacks share flags
and strings with processing and file loading, with inconsistent locking. A
configuration snapshot must include the values that determine buffer sizing,
processing mode, and saved-image validity.

### 7. There are useful local precedents, but no drop-in coordinator

- [modalPSDs.hpp](../../../apps/modalPSDs/modalPSDs.hpp), lines 625–647,
  792–797, and 903–920, uses a restart request, mutex, condition variable, and
  explicit worker acknowledgement before replacing resources. Reuse that
  ownership principle. Its single-consumer protocol is not directly sufficient
  for three monitors plus an output worker.
- [psfAcq.hpp](../../../apps/psfAcq/psfAcq.hpp), lines 594–605, 630–647, and
  919–953, uses one image mutex across source/dark allocation and processing.
  This is a useful local locking pattern, but does not solve the integrator's
  separate output-stream lifecycle.
- The existing optional `frameGrabberPostPublish()` hook runs after output
  semaphore posting (`frameGrabber.hpp`, 1018). It is available if an explicit
  publication-complete notification is needed; do not invent a duplicate hook.
- [shmimIntegrator_test.cpp](../../../apps/shmimIntegrator/tests/shmimIntegrator_test.cpp)
  currently only constructs the app. There are no allocation, numerical,
  publication, or concurrency regressions in that file.
- [streamWriter_lifecycle_test.cpp](../../../apps/streamWriter/tests/streamWriter_lifecycle_test.cpp)
  provides a reusable example of temporary local ImageStreamIO resources and
  worker cleanup. Its `fgThreadExec()` is streamWriter's own worker, not a test of
  this `dev::frameGrabber` contract.

## Accepted design and requirements

### One app-owned configuration and image-lifetime protocol

Introduce a clearly documented image-state mutex and monotonically increasing
configuration generation. All three allocation callbacks, all three processing
callbacks, saved-image installation, and the output handoff participate. Separate
requested configuration from the configuration currently used by processing.

Store app-owned descriptors for the source, each dark, and the output generation:
dimensions, datatype/converter, readiness, and applicable averaging settings.
Only the corresponding monitor callback imports that monitor's base metadata.
The output worker alone sets its frame-grabber dimensions from a committed
descriptor. Validate nonzero geometry, supported types, valid averaging counts,
and checked pixel/byte counts before admitting a configuration.

A short shared lock can establish that app callbacks have stopped using shared
storage before replacement. Do not hold it across semaphore waits, file I/O,
thread joins, INDI traffic, or waits for another worker. If a processing operation
releases the lock while computing, it must retain an independently owned snapshot
or explicit lifetime protection; raw pointers into resizable storage are not
sufficient.

### Coordinated restart for every allocation

Use the following logical phases; these describe behavior, not required enum
names:

1. **Invalidate.** On any allocation or sizing/mode change, record a new request
   generation under the image-state lock and gate ordinary source processing.
   Serialize simultaneous requests. Discard the partial accumulation and any
   completed but unclaimed output; reset counters and readiness. Invalidate the
   affected dark's `Set` flag and recompute compatibility using both dimensions.
2. **Prepare.** Replace shared working buffers only after other app callbacks have
   released them. Install validated descriptors and buffers together. A failed
   allocation must leave the requested generation unavailable, with no partially
   installed state or null converter reachable by processing.
3. **Reconfigure output.** Wake the output worker through app-owned synchronized
   state. At a safe callback boundary it abandons unclaimed old work and enters
   its existing reconfiguration cycle. It must not accept an old semaphore token
   as evidence of a new-generation frame.
4. **Acknowledge.** `configureAcquisition()` selects a prepared generation;
   `startAcquisition()` verifies that the generation still matches after the base
   has created/configured the stream. A request arriving between those callbacks
   forces another cycle. Only a matching acknowledgement releases new-generation
   source processing. Same-size and dark-only allocations still pass through the
   handshake, even when the output mapping can be reused.

Treat “everybody restarted” as a coordinated reset of processing state and output
configuration. Do not unconditionally set every monitor's reconnect flag inside
every `allocate()`: those reconnects call `allocate()` again and can create an
endless restart cascade. The monitor callbacks synchronize at the shared state
boundary; idle or absent optional monitors must not be required to produce a
frame or acknowledge a barrier. Dark callbacks may install a complete new dark
while source accumulation is gated.

Prefer keeping the fix local to this app. Replace its cross-thread writes to
base restart flags with synchronized requests consumed by the owning worker.
Parameter changes can rebuild the integrator's working state from its committed
source descriptor without physically reopening an unchanged source stream.
Keep `frameGrabber::m_reconfig` writes for this protocol on the frame-grabber
thread. Verify all paths, including READY and saved-image startup, before claiming
that the app no longer relies on unsafe cross-thread base flags.

If that cannot be achieved with the existing callbacks, add the smallest
backward-compatible request/acknowledgement facility to the appropriate base,
with focused base tests. Do not silently broaden this work into a repository-wide
conversion of shared fields to atomics. The common need in `modalPSDs` and this
app makes a reusable `dev` restart helper worth considering, but extract one only
after the callback/ownership contract is demonstrated here.

### Stable output ownership through publication

Have `acquireAndCheckValid()` claim a complete frame and its descriptor under the
image-state lock. Keep that frame independently owned and immutable until the
frame-grabber has copied it. A preallocated staging image or explicit buffer swap
is preferable to holding a mutex across two frame-grabber callbacks. Preserve the
existing single-pending-result/backpressure behavior; do not introduce a queue.

`loadImageIntoStream()` uses the claimed frame's checked byte count and verifies
its descriptor matches the configured output. It must never use live source
dimensions. Notification counts are wakeups only; a locked ready/generation
predicate determines whether data exists. Clear or drain stale notifications
while publication state is protected, without dropping a new valid notification.

An already claimed old frame may finish publication into its matching old output
stream before the worker acknowledges the new generation. All unclaimed old
frames are discarded. This defines a safe cutover without aborting a copy midway.
Do not acknowledge cutover at the end of `loadImageIntoStream()` if publication is
still pending in the base; the next worker boundary, or the existing post-publish
hook where needed, must establish completion.

Avoid using a negative `loadImageIntoStream()` result as the normal restart path:
the base has already set the stream's `write` flag and breaks without clearing it
on that return (984–989). Normal restarts should be caught before claiming a frame;
in-flight frames remain safe through their independent ownership. Any new error
path here needs explicit write-flag cleanup and a regression test.

### Control updates, saved images, and shutdown

Stage INDI requests under a consistent ownership policy and apply them atomically
at the same transition boundary. Audit `fps()`, start/stop, validity flags, and
state-string readers as well as the three numeric callbacks. Avoid nested INDI
and image-state locks where possible: snapshot or enqueue, release, then perform
property updates. Document a single order for any unavoidable nesting.

Read a candidate FITS file into a temporary image outside both locks. Recheck the
state-string/configuration generation and running state before installing it.
Discard a stale or mismatched candidate without changing live storage. When no
source is connected, use a separate saved-image output descriptor. Save completed
averages from stable snapshots so disk writes cannot race with restart or reload.

Shutdown and partial startup failure must wake any new waits, prevent further
publication, and let all workers finish before destroying synchronization objects
or buffers. Waiting for another thread while holding `m_indiMutex`, waiting for an
unstarted output worker, and requiring a callback from an absent dark stream are
all prohibited by the accepted lifecycle contract.

## Implementation sequence

1. Extend the existing app test harness to exercise real allocation/processing and
   frame-grabber interface methods using local arrays, private INDI state, and
   deterministic synchronization points. Establish block/moving-average and
   saved-image baseline behavior before changing ownership.
2. Implement app-owned descriptors, protected image state, and the generation
   transition. Route all three allocations through one common transition helper;
   correct the dark dimension checks and readiness reset in the same change.
3. Implement output claiming and acknowledgement across `configureAcquisition()`,
   `startAcquisition()`, acquisition, copy, and reconfiguration. Cover requests
   arriving during output setup and between acquisition and copy.
4. Route configuration callbacks and FITS installation through the protocol;
   remove direct cross-thread mutation of base geometry and restart flags from
   the app where ownership has replaced it. Finish shutdown/error-path handling.
5. Run the focused tests, production build, and isolated stream stress tests below.
   Record actual results and any remaining shared-base limitations in this plan.
6. Complete the full changed-file Doxygen pass, preserving header-only app
   implementation with out-of-class definitions. Use existing `FRAMEGRABBER_*`,
   `SHMIMMONITORT_*`, and `TELEMETER_*` interfaces where applicable. Document
   parameters inline and annotate lifetime-only lock scopes as `{ //mutex scope`.
   Put the app test group in `tests/groups.dox` under `application_unit_test`,
   retain explicit Doxygen test references, and remove the touched test's author
   tag. Run `clang-format` on changed C++/Doxygen files.

For implementation commits, create a `jrmales/<feature-name>` branch first; do not
commit on `dev`. Keep functional changes/tests and this engineering record
together, followed by documentation-only changes and separate formatting cleanup
when needed, following `AGENTS.md`.

### Expected files

| File | Planned change |
| --- | --- |
| `apps/shmimIntegrator/shmimIntegrator.hpp` | Ownership, restart coordination, callbacks, FITS handoff, full-file documentation. |
| `apps/shmimIntegrator/tests/shmimIntegrator_test.cpp` | Numerical, lifecycle, and deterministic concurrency regressions. |
| `tests/groups.dox` | App test group declaration under `application_unit_test`. |
| This plan | Decisions, implementation progress, and validation evidence. |
| `libMagAOX/app/dev/frameGrabber.hpp` and/or `shmimMonitor.hpp`, plus corresponding tests | Conditional only if a required safe lifecycle boundary cannot be provided within the app. |
| `tests/tests.list` / test build files | Only if a separate lifecycle suite or necessary instrumentation is added; the current app suite is already listed. |

## Validation plan

Use barriers/condition variables or controlled test hooks to force interleavings;
do not rely on sleeps to make a race likely. Every blocking test needs a bounded
deadline and cleanup that releases blocked workers even after an assertion fails.

| Case | Required evidence |
| --- | --- |
| Source grows, shrinks, changes datatype, or reconnects at the same size | Counters/history reset; committed descriptors match buffers; output copies stay inside guarded destinations. |
| Resize while output is waiting, after acquire, and during configure/start | No old/new shape mixture; an in-flight old snapshot is safe; an obsolete setup is rejected; latest request eventually wins. |
| Both dark allocations overlap source processing and output setup | No restart loop/deadlock; coherent complete dark pixels; each allocation participates in the generation transition. |
| Width-only or height-only dark match, both darks, invalid type, and reallocation before first dark frame | Incompatible/uninitialized darks are never subtracted; each `Set` flag reflects a complete frame for its allocation. |
| Old semaphore tokens, slow output consumer, repeated requests | No stale/uninitialized or duplicate publication from tokens alone; no lost new-generation ready result. |
| `nAverage`, `nUpdate`, `avgTime`, and FPS changes during moving-average processing | No old-cube/new-count access; a coherent configuration is adopted; numerical results match the selected mode. |
| Block and moving averages, start/stop, continuous and one-shot operation | Expected pixel values, update cadence, existing backpressure, and run-state behavior remain correct outside restart intervals. |
| FITS candidate mismatch, state change during read, reload during output, saved-image startup before source | Candidate rejection leaves live data intact; valid saved output works without mutating monitor dimensions; stale state is not marked valid. |
| Missing source/darks, READY state, failed allocation/setup, partial startup, shutdown during transition | No waits for nonexistent participants; no premature readiness; bounded shutdown with no waiter using destroyed state. |

Build/run the focused suite on the workstation using the existing test makefile:

```sh
make -C tests -f Makefile.one t=../apps/shmimIntegrator/tests/shmimIntegrator_test.cpp
./apps/shmimIntegrator/tests/shmimIntegrator_test
make -C apps/shmimIntegrator
git diff --check
```

Match `COVERAGE` to the local shared-library build if needed. Then use isolated,
uniquely named local ImageStreamIO test streams to repeatedly resize/recreate the
source and darks while publishing and changing averaging settings. Exercise the
actual base worker sequence as well as direct callback tests. Run ASan/UBSan for
buffer/type faults and a separately instrumented TSan build for thread ownership
where dependencies permit. Distinguish app failures from pre-existing base or
dependency reports; do not claim a race-free full stack from a limited harness.
Measure processing latency and frame loss against the baseline so the added
locking/copy does not introduce an unnoticed throughput regression.

No agent commands may run on instrument computers or through an instrument INDI
tunnel. Any eventual instrument verification must be performed by a human
operator. No instrument access is needed for this plan or its offline tests.

## Accepted decisions and follow-up

The user's approval resolves the design choices below; no further confirmation
of these choices is needed for implementation.

- Reset and synchronize all participating processing state without forcing
  unaffected input streams to close/reopen. Physical reconnect of every monitor
  is not required.
- Discard unclaimed old frames, but let one already claimed frame finish before
  acknowledging cutover. The new generation becomes active only after that
  publication completes and output setup is acknowledged.
- Keep the existing first-dark-only behavior in moving-average mode for this fix;
  decide separately whether it should match the two-dark simple-average behavior.
- Ordinary dark pixel updates should be atomic with respect to subtraction, but
  should not restart accumulation on every dark frame. Allocations/reconnections
  do restart it.
- Loss of an optional dark stream currently has no implemented derived cleanup
  callback in `shmimMonitor` (only comments at 834–850). Choosing whether to retain
  the last complete dark on disconnect is separate from making allocations safe;
  do not silently change that policy.
- Shared-base status/metadata publication and shutdown fields have their own
  cross-thread access patterns. Keep any remaining limitations explicit; this
  app fix must not be presented as a complete base-class concurrency audit.

## Investigation verification

Read issue #220 and its empty comment list, inspected all integrator methods and
its current test, traced the relevant monitor/frame-grabber loops, and compared
the local synchronization/test precedents above. Only this Markdown plan was
added. No C++ tests or application processes were run for this documentation-only
investigation; `clang-format` is not applicable to the Markdown artifact.


## Implementation record — 2026-10-18

Implemented within `shmimIntegrator`; neither shared device base needed a change.

- A shared image-state mutex and configuration generation now cover source/dark
  allocation, processing, runtime requests, and saved-image installation. Monitor
  callbacks import only their own geometry/converter. Processing stays gated until
  `startAcquisition()` acknowledges the generation selected by
  `configureAcquisition()`, including requests arriving during output setup.
- The output notification is now a condition variable with a protected predicate,
  replacing the counting semaphore. The output worker claims an independently
  owned, preallocated image and copies only its checked configured shape. A
  previously claimed frame can finish; all unclaimed old work is discarded.
  Single-result backpressure remains in effect until the copy completes.
- All three allocations use the same invalidation path. Darks require both axes
  to match and a complete frame after allocation. Invalid source/dark allocations
  keep publication unavailable until the affected allocation succeeds. Missing,
  unconfigured dark streams do not block processing. History allocation failure
  leaves the generation unprepared and permits a later retry.
- Configuration changes are staged without cross-thread writes to either base's
  restart flag. Only the output worker sets `frameGrabber::m_reconfig`. The app no
  longer changes monitor dimensions from disk-loading code. Runtime input uses
  the strict stream-parsing pattern already present in `kcubeCtrl`, together with
  `mx::math::isFinite`, which works with this repository's fast-math builds.
- Geometry/history checks account for `eigenCube`'s signed-int dimension products.
  Moving-average burn-in and update counters saturate instead of wrapping.
  Pixel summation uses contiguous traversal with the same summation order per
  pixel. Simple averages retain the original combined two-dark subtraction;
  moving averages retain first-dark-only subtraction.
- FITS reads use temporary images, then check configuration and state revisions
  before installing them. This rejects a start/stop cycle, a state change away and
  back, and validity-only changes during a read. Saved-image startup still works
  before the source connects. FITS writes use a completed-image snapshot outside
  both app locks. State validity is required before a saved image is admitted.
- Shutdown wakes the condition-variable waiter before joining the workers. No new
  wait requires an optional monitor, a future input frame, or an unstarted thread.
  Unexpected copy errors explicitly release the output stream's write flag.

The changed application and test retain header-only production implementation and
out-of-class non-trivial definitions. The full changed-file documentation pass
covers the new ownership rules and all declarations. Application tests use the
`application_unit_test` group and explicit Doxygen references to the real API.

### Verification results

All execution was on the local workstation with synthetic arrays, private `/tmp`
FITS data, and isolated local ImageStreamIO streams. No instrument connection,
application installation, or instrument INDI command was used.

- Established the original numerical baseline: 3 cases / 48 assertions covering
  block averaging, moving-average cadence, dark subtraction, and one-shot output.
- Final focused suite: **20 cases / 2,003 assertions passed**. This includes
  concurrent dark writes and allocations, old claimed frames during resize,
  stale/unclaimed work, setup-generation changes, strict numeric validation,
  injected history-allocation failure, state/FITS races, shutdown, and output
  write-flag cleanup.
- The suite's real-worker test starts the production three monitor loops and the
  production frame-grabber loop. It recreates local source/dark streams through
  4x3, 7x5, 2x2, same-size 2x2, and 4x3 transitions while changing averaging counts.
  It checks output dimensions, pixel values, publication, and shutdown. The test
  waits for all three new allocations as well as the configuration request, so a
  same-size replacement cannot pass merely because the old dimensions match.
- `make -C apps/shmimIntegrator COVERAGE=1` compiled and linked successfully.
  `COVERAGE=1` matches the workstation's existing shared-library build.
- ASan/UBSan: **20 cases / 2,003 assertions passed**, with no address/undefined
  behavior reports. Leak detection was disabled after LeakSanitizer reported that
  it cannot operate under the sandbox's tracing environment; leak checking is not
  claimed. The test translation unit was built with `-O1 -g1
  -fsanitize=address,undefined -fno-omit-frame-pointer`; shared dependency binaries
  were not rebuilt with sanitizers.
- TSan: **19 cases / 1,938 assertions passed**, with no race reports. This run
  excludes `[streams]` to isolate app-owned synchronization from the known plain
  shared fields in the base worker lifecycle. It includes the deterministic and
  concurrent app-level tests; it does not establish a race-free full framework.
  The test translation unit used `-O1 -g1 -fsanitize=thread`.
- `clang-format` and `git diff --check` passed.

Normal test commands (run from the repository root):

```sh
make -C tests -f Makefile.one t=../apps/shmimIntegrator/tests/shmimIntegrator_test.cpp COVERAGE=1
./apps/shmimIntegrator/tests/shmimIntegrator_test
make -C apps/shmimIntegrator COVERAGE=1
```

### Throughput probe and limits

Compared the original header at the reviewed commit with the implemented header
using a temporary synchronous producer/consumer probe: float input, window of 10,
20 warm-up frames, median of three measured runs, `-O3 -ffast-math`, and one
OpenMP/OpenBLAS thread. Measurements include processing, claiming, and copying
output; construction and configuration are outside the timer. Both probes link
against the workstation's existing dependencies, including the coverage-built
shared library. These are local comparative measurements, not instrument latency
or real-time scheduling guarantees.

| Image / mode | Original microseconds per input | Implemented microseconds per input | Frames / outputs per measured run |
| --- | ---: | ---: | ---: |
| 128x128 block average | 142.2 | 151.7 | 100 / 10 |
| 128x128 moving average, update every frame | 4,085.2 | 1,256.7 | 100 / 100 |
| 512x512 block average | 2,728.3 | 2,406.3 | 40 / 4 |
| 512x512 moving average, update every frame | 73,920.6 | 20,590.0 | 40 / 40 |

Both implementations produced the expected output counts and constant pixel
values. The smaller block-average case showed about 7% overhead; the extra output
snapshot and synchronization have a cost. Contiguous moving-average traversal
substantially improved the two moving-average cases. Actual frame loss under
instrument scheduling/load still requires operator validation.

### Remaining follow-up

- A first sanitizer run exposed a pre-existing logger deallocation mismatch:
  `flatlogs::logHeader::createLog<git_state>` allocates a multi-byte buffer that is
  released through `std::shared_ptr<char>`'s scalar deleter. ASan reports this at
  logger teardown, after the suite completes. No logger code was changed here.
  The offline fixture now suppresses startup logging before base construction so
  the integrator sanitizer run can finish without that unrelated failure.
- Shared-base status/telemetry metadata and lifecycle flags remain outside this
  app-scoped concurrency correction, as agreed. The real-worker test verifies
  behavior, while the app-only TSan run deliberately does not certify those fields.
- Retaining the last complete dark after disconnect and first-dark-only correction
  in moving mode remain unchanged policies. Hardware verification and any policy
  changes remain separate follow-up work.

### Commit record

- `ad2dbee4`: functional implementation, regression suite, and this engineering
  record. The shared `dev` bases remain unchanged.
- The following documentation-only change registers the application test group
  in `tests/groups.dox`. Declaration-spacing cleanup is kept separate from the
  functional change.
