# ocam2KCtrl acquisition and publication timing review

Created: 2026-10-23; updated 2026-10-24 (America/Phoenix).
Status: the approved deferred OCAM rearm and independent timeout-query
measurements are implemented locally. Workstation verification includes
compilation/linkage against the copied PDV 5.6.8.0 SDK. Operator qualification
of the new ordering, instrumentation overhead, loaded driver and existing
DMA-owner-thread topology remains pending. Agents perform no instrument actions.

Paired chronological and kernel captures identify EDT interrupts as the source
of recurring stage extensions. The operator moved IRQ 405 from acquisition
CPU 33 to CPU 61 as the only experimental change. Retained publication-interval
deviation decreased from 1.864 to 0.995 us, and large conversion/notification
extensions disappeared from the selected neighborhoods. Remaining long rearm
calls track the remote handler, strongly supporting driver/device serialization
without identifying a specific lock. These captures establish no scheduler
safety margin or continuous full-population timing bound.

The latest implementation replenishes eligible OCAM DMA queues at the end of
the existing post-publication hook, after both stream notifications. Timeout
and frame validation stay before publication; the independent `timeoutQuery`
measurement will show whether contention also affects that guard. The source
review, earlier implementations and each operator capture remain below as the
engineering history.

## Request and purpose

> ok, well let's take a hard look at ocam2KCtrl.  I think this calls for a new plan document to record findings on this.

Locate where camera timing excursions enter between the EDT timestamp attached
to a frame and publication of that frame to `camwfs`. Record code findings,
measurement limitations and proposed fixes here. The
[synco debugging record](dm-syncopation-debugging.md) retains the preceding
operator trials; the [synco plan](dm-syncopation.md) retains scheduler work.
This investigation does not choose an early-arrival reserve or change the
median-grid/skip policy.

All agent inspection and development run on the workstation. The operator
performs every instrument action, including read-only process/INDI checks and
timing qualification. The approved design below has now been implemented locally;
this document does not deploy changes or alter instrument camera settings. The
implementation record describes current behavior; the source review and approved
work sequence remain as the historical basis for those changes.

## Evidence and current configuration

### Operator timing evidence

The isolated `wfs-acqu-timing.txt` excerpt has 13 records at October 23
20:55:55–20:56:07 Phoenix time (October 24 03:55:55–03:56:07 UTC). It shows
diagnostics-only operation, a disabled combiner observer and one stable
camera-triggered acquWFS trace row. These are ranges across overlapping
two-second windows, not one independently sampled population per printed row.

| Interval | Window standard deviation, us | Observed extrema, us |
| --- | --- | --- |
| Camera SDK `atime` | 0.646–0.906 | 480–520 |
| Camera `writetime` | 2.810–3.510 | 437.045–579.825 |
| acquWFS processing start | 2.164–3.288 | 430.132–577.219 |
| acquWFS output publication | 2.585–3.585 | 430.012–577.320 |

Camera observation quality is clean in this excerpt. The primary/pipeline
observer has two nonunit-counter observations and ten duplicate observations,
with approximately 3 ms local receipt gaps followed by catch-up. Qualified
source/publication statistics exclude those discontinuities. They do not prove
corresponding producer stalls, but they limit coverage of rare excursions.

Matched conditional decomposition within the acquWFS trace places almost all
of its large output interval excursions at or before processing start. The
independent camera publication series already has comparable tails. This
narrows the next investigation to the camera path and first-stage wake region;
it does not associate individual camera and acquWFS extrema or measure their
absolute handoff latency. Source identity, exact populations and cohort sums
are recorded in the debugging record.

### Operator-confirmed settings

The operator supplied these active settings and confirmed current mode `bin2`:

```ini
[framegrabber]
numBuffs=64
threadPrio=90
cpuset=camwfs
shmimName=camwfs
circBuffLength=1
latencyTime=2
latencySize=20000

[camera]
startupMode=bin2
startupTemp=-45
ocamDescrambleFile=ocam2k/fli/ocam2_descrambling.txt
```

The operator initially reported EDT PDV **5.6.2.0**. During implementation the
copied installation log identified **5.6.8.0**, and the operator clarified that
a manual install overwrote the default; see the current evidence below. Match
that installed tree when checking the SDK contract. The 6.3.0 web reference and
historical 4.2.4.3 local source are not substitutes for its headers,
implementation and documentation.

At approximately 2 kHz, distinguish three different buffers:

- EDT DMA ring: 64 frames, approximately 32 ms of nominal storage capacity.
  Capacity is not a measurement of current frame age or queued latency.
- Output shared-memory image: depth one, overwritten once per published frame.
- Existing timing rings: nominal capacity 8000 timestamps each, calculated as
  `min(2 * latencyTime * fps, latencySize)`. Ordinary app logic analyzes roughly
  4000 timestamp entries per update with these settings.

Priority 90 and cpuset `camwfs` are requested settings. Actual scheduling policy,
priority, allowed CPUs and camera IRQ placement remain runtime facts to verify.
The active `[bin2]` mode definition and digital factors were initially unknown;
they are now confirmed from the current local configuration below. Synchro state
and separately loaded driver/build details remain runtime evidence to obtain.
The name `bin2` alone would not prove every software mode field.

### Current local configuration and manual SDK installation

During implementation the operator confirmed that the local `/opt/MagAOX/config`
copy is current. Read-only inspection established:

```ini
[bin2]
configFile=ocam2k/ocam-mode3.cfg
serialCommand=mode 3
maxFPS=3620
```

Neither digital-binning key is present. `stdCamera.cpp` initializes both digital
factors to one, so the active bin2 path uses the SDK's 120x120 binning descrambler
without the deferred extra software-binning path. Its EDT file selects raw
1056x62, 8-bit input, eight horizontal taps, `fv_once: 1` and
`method_framesync: EMULATE_TIMEOUT`, with serial baud 115200. File SHA-256:

- `camwfs.conf`: `20826b8ea57d2a3cf3964e3a761c4cd1988fc5f9092ebe69be99cbd1d5515cbd`.
- `ocam2k/ocam-mode3.cfg`: `94c518f21e278fbdd32d6d3761ee11a7db63ff139f924740dbbb142632341c6c`.

EDT documents that EMULATE_TIMEOUT exposes frame-synchronization errors through
the timeout counter; it is not a timeout on each healthy frame. Thus the new
counter check also observes configured footer/synchronization errors, rather
than only waits that returned a null pointer. The copied implementation now
confirms this behavior; the detailed source findings are recorded below.
[EDT 5.x acquisition reference](https://edt.com/api-2018/group__acquisition.html),
[EDT frame-sync release note](https://edt.com/edt-pdv-driver-version-5-4-1-1-packages-released/).

The earlier reported installed version was 5.6.2.0. The SDK copy being transferred
to `/home/jrmales/Source/EDTpdv` contains `install_pdv.log`: line 1724 records
**5.6.8.0**, installed November 12, 2025. The operator clarified that a manual
installation overwrote the default installation. Target the manually installed
SDK tree for the remaining contract/build checks; confirm its version file and
headers as they become available. The separately loaded driver/binary version
is still a runtime qualification fact, not proven by the installation log.

### Reviewed source

MagAOX HEAD: `55f38bcab5fe620f2567059fea755dbf0fb04010`.
Local mxlib HEAD: `32c6a0b3d8691bce6c84b36a9690d25d07eaf106`.
These are inspection revisions, not newly verified instrument binary hashes.

The frame-path description and findings below describe the pre-implementation
source at these inspection revisions. They are preserved as historical evidence;
resolved findings are identified in the implementation record rather than
rewriting the original observations.

Principal files, with line numbers below referring to these revisions:

- [ocam2KCtrl.hpp](../../../apps/ocam2KCtrl/ocam2KCtrl.hpp): application,
  controls, frame checks, descrambling and synchronization output.
- [edtCamera.hpp](../../../libMagAOX/app/dev/edtCamera.hpp): device setup,
  serial transactions, DMA wait/rearm and timestamp propagation.
- [frameGrabber.hpp](../../../libMagAOX/app/dev/frameGrabber.hpp): acquisition
  loop, publication order and existing statistics.
- [ocam2_sdk.c](../../../apps/ocam2KCtrl/fli/ocam2_sdk.c): the OCAM descrambler
  actually compiled by the app Makefile; this is distinct from EDT's PDV SDK.
- [MagAOXApp.hpp](../../../libMagAOX/app/MagAOXApp.hpp): thread scheduling setup.
- mxlib `include/sigproc/circularBuffer.hpp`: both legacy and fixed-size ring
  implementations; dependency revision matters to the concurrency findings.

## The actual frame path

```text
frameGrabber acquisition worker
    |
    +-- acquireAndCheckValid()
    |      +-- pdv_wait_last_image_timed() -> image pointer + SDK timestamp
    |      +-- pdv_start_image()           -> requeue one acquisition
    |      +-- copy SDK timestamp to m_currImageTimestamp
    |      +-- validate raw OCAM frame number
    |
    +-- output write flag = 1
    +-- loadImageIntoStream()
    |      +-- OCAM SDK descramble
    |      +-- optional software binning
    |
    +-- sample CLOCK_REALTIME into existing output writetime
    +-- copy atime; update counters and per-plane timestamp arrays
    +-- output write flag = 0
    +-- broadcast main-stream semaphores
    +-- OCAM post-publish hook: mirror metadata and broadcast sync stream
    +-- append existing timing samples
    +-- prepare/touch next output slot
    +-- compare FPS; potentially reset timing rings
    |
    +-- next iteration / next SDK wait
```

The `writetime` sample precedes reader notification. Main/sync semaphore fan-out
and later bookkeeping can delay the following iteration's SDK wait. They cannot
directly delay the current frame's timestamp after it has already been sampled.
Measure both sides of that boundary and retain previous-frame context.

## Findings relevant to timing attribution

### 1. SDK timestamp provenance needs an installed-version contract check

`edtCamera::pdvAcquire()` (549–560) copies two SDK values to seconds/nanoseconds;
OCAM passes that value through and `frameGrabber` assigns it unchanged to
`md->atime` (999). That verifies the application assignment, not the precise
driver event or clock implementation.

The current EDT 6.3.0 acquisition documentation describes a DMA-completion
timestamp based on host system time, with interrupt/timestamping latency. Its
latest-image calls can skip completed frames. Thus an SDK-provided timestamp
is not automatically an exposure or hardware-only timestamp. These are vendor
reference semantics, not verification of the manually installed SDK/driver.
[EDT acquisition documentation](https://edt.com/api-pdv/group__acquisition.html)

The current API reference specifies seconds/nanoseconds for `edt_get_timestamp`,
but uses `edt_timespec_t`; this app uses the older two-integer interface. Inspect
the matching installed header/library and driver details before asserting
binary compatibility or an absolute acquisition-to-write clock relationship.
[EDT timestamp reference](https://edt.com/api-pdv/libedt_8h.html)

Historical local EDT 4.2.4.3 source is useful corroborating material, not the
deployed implementation. The initial 5.6.2.0 report was superseded by evidence
of the manual 5.6.8.0 install; the loaded library and driver must still match the
reviewed source before asserting runtime provenance.

### 2. The SDK return and rearm costs are currently combined

`pdvAcquire()` performs the blocking wait, `pdv_start_image()` and timestamp
copy before returning. The app then validates the camera frame number. Existing
timing cannot separate SDK wake/return, requeue, validation and descramble.
At least the wait return and requeue return need distinct local markers.

Continuity of OCAM frame numbers cannot alone establish low frame age. A
64-buffer ring can absorb delayed consumption; whether this call selects the
latest completed frame and how it reports skips must be checked against the
installed SDK and active EDT configuration. Preserve the existing queue policy
until its behavior and recovery contract are understood.

### 3. Housekeeping shares the device, but the normal grab path takes no app lock

`ocam2KCtrl::appLogic()` (769–813) polls temperature, FPS when not synchronized,
and gain. Serial helpers take `m_cameraMutex` and use the same `PdvDev` handle;
`pdvSerialWriteRead()` includes a 10 ms trailing-byte wait (319–331).

Normal `acquireAndCheckValid()`, `pdvAcquire()` and `loadImageIntoStream()` take
neither `m_cameraMutex` nor `m_indiMutex`. Consequently the explicit serial mutex
is not an established acquisition-thread wait. Driver/device/shared-handle
contention and CPU scheduling remain hypotheses to measure. Adding that mutex
around the blocking grab would introduce a new multi-millisecond blocking path.

Record serial transaction intervals on their ordinary control paths and
correlate them with the camera segment outliers. Configuration owns both locks
and performs serial I/O, SDK initialization and allocation outside the normal
frame loop; mark those epochs explicitly.

### 4. Exact FPS changes can put timing-buffer reset work into the worker

`frameGrabber` compares `m_cbFPS != derived().fps()` every frame (1056–1062).
`configCircBuffs()` clears/reserves both timestamp rings (780–808). A reported
FPS change can therefore trigger storage work and a diagnostic reset in the
acquisition worker. Stable reported FPS avoids this path.

`getFPS()` writes the reported float (1136 onward), and the external sync
callback (1756–1775) can also update FPS and request full reconfiguration.
First count actual resets/rate changes and their durations; do not infer they
occur merely from small fluctuations in independently measured acquisition FPS.

### 5. Existing timing statistics have a cross-thread ownership defect

The worker appends to `m_atimes`/`m_wtimes` (1024–1027) while ordinary app logic
reads `size()`, `latest()` and payloads (602–648). The selected default mxlib
`circularBufferIndex<timespec, cbIndexT>` specialization has ordinary vector
state and indices. There is no common lock or immutable handoff. Ring reset can
also overlap the consumer's scan. This is a source-level concurrency concern;
it is not evidence of a statistics mutex stalling camera publication.

The newer `fixed_size=true` specialization preallocates and supplies atomic
indices/snapshot APIs, but payload overwrite and live resizing are not protected
by reader ownership. Retrying after detecting a changed generation does not
undo an overlapping ordinary payload read/write. Simply changing the template
argument is not a complete fix. It remains useful for consumer-owned history.

Also, the existing statistics convert epoch timestamps to `double` before
differencing (629–652). Integer differences would avoid sub-microsecond
rounding. That rounding does not explain the independent approximately 80 us
publication tails already measured by the new synco observer.

### 6. Requested scheduling can be reported as successful after failure

`MagAOXApp::threadStart()` calls `pthread_setschedparam()` and checks `rv < 0`
(2587–2604). That API reports errors as positive return codes. The check can
therefore log success when the policy change failed. This is a concrete shared
helper defect to fix with focused tests and a separate commit. It does not
establish the actual scheduling policy in this run. Verify the worker's policy,
priority and affinity before interpreting configured priority 90 as applied.

### 7. Shipped 2x2 descrambling is bounded; active mode fields need verification

The shipped 2x2 hardware-mode descrambler is a bounded table lookup/copy loop
(`ocam2_sdk.c`, 260–269). Its dispatch path performs no normal-path allocation,
logging or locking. Measure it before proposing a replacement. Software binning
is a separate optional path selected by the mode's digital factors.

The OCAM sync hook (1688–1713) takes no application mutex on the normal path
and posts after the main stream. Its fan-out cost, timing-ring writes and
next-slot preparation belong to the post-publication tail. With output depth one, the output stays
hot. Reserved-but-lazily-populated timing pages may matter at startup/reset;
page-fault effects are not established as the recurring steady-state cause.

## Additional correctness findings

These warrant isolated tests/fixes and should not be presented as established
explanations of the current steady-state jitter.

| Finding | Source evidence | Proposed treatment |
| --- | --- | --- |
| Failed/partial acquisition is not validated | `pdvAcquire()` always returns zero, leaves the timestamp array uninitialized before the SDK call, and checks no null result. OCAM ignores its return and dereferences `m_image_p` at 1580. Old comments explicitly remove timeout/overrun checks. | Define a version-correct acquisition result contract; reject null/invalid/partial frames before reading their header or publishing. Count the failure/recovery path. Preserve queue/recovery behavior until tested. |
| SDK handle zero is valid but treated as empty | Vendor `ocam2_init()` returns its first ID as zero (431–461), which is valid (60–65). The app releases an old ID only when `> 0` (1468). Shutdown/destructor do not release the final ID. Tests default to ID one. | Track initialization separately from numeric ID; test zero, reconfigure, failed reinitialization and final cleanup after the worker stops. |
| Wrap check accepts more than exact wrap | The predicate at 1594 bypasses error handling for any previous-to-zero transition or any `UINT_MAX`-to-current transition. | Test exact wrap, ordinary gaps, duplicates and backward resets; make any recovery-policy correction a separate change. |
| Software binning policy/implementation is unclear | Enable tests only X (1437); the inner loop repeatedly assigns a sample (1659–1663); Y changes dimensions without scaling the source column. Existing all-seven pixels cannot distinguish selection from summation. | Confirm active digital factors and intended sum/mean/decimation policy. Use asymmetric-pixel tests before changing science pixels. |
| Control-to-worker reconfiguration state lacks a coherent handoff | Sync callback writes plain FPS, request string and reconfigure flag; the worker reads them without a common ownership protocol. | Define immutable requested configuration plus worker acknowledgement. An atomic flag alone does not protect the associated string/storage or prevent a request being overwritten. |
| EDT device lifetime crosses threads | Initial setup/open/DMA allocation runs in main-thread `appStartup`; wait/rearm and subsequent reconfiguration run in the worker; final close is in `edtCamera` destruction. | Verify the installed SDK's requirements. Current vendor documentation requires same-thread DMA resource lifecycle; assess an owner-thread design if applicable rather than blindly applying a different SDK version's contract. |

Current vendor lifecycle reference:
[EDT multithreaded programming](https://edt.com/api-pdv/group__dma.html).
The app's `m_cameraMutex` comment claims broader acquisition protection than
the actual hot path supplies; correct the documentation when that file is
changed during implementation.

## Approved measurement design

### Endpoint contract

Use monotonic integer nanoseconds for new local durations. Preserve the SDK
timestamp and existing `CLOCK_REALTIME` writetime unchanged and in their current
positions. Store their provenance separately; do not subtract monotonic markers
from either external timestamp. A matched `writetime - atime` diagnostic is
qualified only after validating the SDK's epoch, units and event semantics for
the installed version. Existing `delta_aw` is useful context, not that proof.

| Marker | Position and purpose |
| --- | --- |
| `wait_enter` | Immediately before the SDK wait; includes the previous frame's completed tail in successive-loop accounting. |
| `wait_return` | First local marker after the SDK returns, before `pdv_start_image()`. |
| `rearm_return` | After queuing the next acquisition. |
| `validated` | After raw frame-number/result validation; retain rejected outcomes too. |
| `load_done` | After descramble/binning, directly before the existing realtime writetime call. This is the call's monotonic lower bracket. Split conversion further only if it warrants it. |
| `write_stamp_done` | Immediately after the existing realtime writetime call; the existing stamp itself is retained. |
| `main_post_done` | After metadata completion and main-stream semaphore broadcast. |
| `sync_post_done` | After the OCAM post-publication hook returns. |
| `loop_tail_done` | After ordinary per-frame bookkeeping, pointer preparation and any diagnostic/FPS reset work. |

Record one fixed-size event per acquisition attempt, with session/generation,
event sequence, validated OCAM frame number when available, published stream
counter/identity, mode/digital factors, original timestamps, marker validity and
outcome. Invalid waits, discarded frames, reconfiguration and clock failures
must remain visible; statistics of published frames alone can hide interruptions.
Store supported SDK progress/skip/timeout information only after its exact
version/API cost and safety are established.

Do not move current notification points to simplify instrumentation. Marker
calls have their own overhead; measure that overhead and preserve `errno`.
Disabling optional stage timing adds no stage-marker clock calls or additional
stage-event handoff. Existing paired timestamp collection may remain active
according to `latencyTime`/`latencySize`, through its repaired safe handoff.

### Ownership, aggregation and reuse

- Use a preallocated, pre-touched bounded producer-to-consumer handoff. The
  acquisition worker never waits for the reader, formats telemetry, allocates,
  sorts, or computes distributions. A full queue drops a diagnostic event and
  increments an explicit count, without changing frame acquisition/publication.
- Carry paired `atime`/`writetime` and all segment markers in one event. The
  ordinary consumer owns retained history and summary storage. Stop/restart
  generations are explicit; never clear/resize shared live payload storage.
- Reuse the ownership-safe SPSC mechanism and statistics/qualification patterns
  already demonstrated by `dev::dmTiming`. Prefer extracting the minimal generic
  fixed-event queue into a common helper, with unchanged DM tests, to copying
  the complete DM-specific event/marker model. A consumer-owned circular buffer
  remains suitable for retained history. Decide this small extraction during
  review; a broad controller rewrite is not required.
- Migrate the existing paired framegrabber timing samples to the same safe
  handoff, as its own correctness step, while retaining `fg_timing` and
  `telem_fgtimings` endpoint meanings. New stage timing is explicitly optional;
  its absence/disabled/stale/lost states must be distinguishable.
- Reuse the existing ordinary-thread `appLogic` and `dev::telemeter` integration.
  Add stage/quality readbacks and a compatible new telemetry type rather than
  changing meanings in the existing `fgtimes` schema. Use helper interface
  macros when available. Keep telemetry construction outside the worker.
- Serial transaction markers use a distinct control-event channel, serialized
  by existing control-path ownership. Multiple threads must not write the frame
  worker's SPSC producer state. Record polling overlap; an ordinary-thread
  telemetry record is not a new lock in the acquisition worker.

Reuse `framegrabber.latencyTime`/`latencySize` for window/capacity only where the
new queue and history semantics are explicit. Proposed additional configuration
is a default-off stage-timing enable. Exact public names and record fields
should be fixed in the implementation review and documented with the tests.

### Analysis and attribution

For a qualified same-clock frame sequence, let `R_i` be SDK wait return, `W_i`
the monotonic write-stamp endpoint, and `D_i = W_i - R_i`. Then:

```text
W_i - W_(i-1) = (R_i - R_(i-1)) + (D_i - D_(i-1))
```

Split `D` into rearm, validation, conversion and write-stamp segments. This
matched decomposition locates publication-interval changes without subtracting
unrelated RMS values. Analyze the main/sync post tail against the following
frame's wait-entry/return interval, not the already sampled current writetime.

`write_stamp_done` is a nearby monotonic endpoint, not the exact instant of the
existing realtime metadata sample. Preemption inside the `load_done` to
`write_stamp_done` bracket can separate them. Report that bracket's width and
qualify attribution to actual `camwfs.writetime` excursions by its uncertainty.
Agreement with realtime intervals also depends on realtime clock behavior; the
identity is exact for the monotonic proxy endpoints only.
Wide brackets keep their original timestamp outliers in the raw population,
with attribution marked ambiguous; do not silently censor them. Set the
allowable bracket width during overhead/accuracy qualification.

Report count, median, standard deviation, extrema, coverage/age, errors and
conditional segment changes for late/early intervals. Retain frame/counter
continuity and session qualifications, plus the diagnostic observer's own
quality when comparing its measurements. Include discarded/recovered attempts
and do not classify an SDK wait of roughly one normal frame period as a stall
merely because it blocks.

## Approved work sequence and planned verification (historical)

1. **Freeze provenance and define tests.** Inspect the matching installed SDK
   headers/source/documentation (initially reported as 5.6.2.0; the manual
   installation log subsequently identified 5.6.8.0). Obtain the
   active bin2/EDT mode definitions, loaded driver/build details, runtime
   worker/IRQ placement and matching binary hashes through the operator. Review
   existing error/recovery semantics and confirm the timestamp/event contract.
2. **Guard failed acquisitions before instrumented trials.** Make the minimal
   version-correct failed-result/timestamp propagation and null-header guard a
   separate tested prerequisite. Instrumentation alone cannot safely observe
   failed SDK returns while the current unconditional dereference remains.
   Keep requeue/recovery policy unchanged except where the validated failure
   contract requires it. Review other `edtCamera::pdvAcquire()` callers,
   including `cred2Ctrl` and `andorCtrl`, and test their error propagation if
   changing the shared helper. Baseline comparisons use the same result guard.
3. **Repair observation ownership.** Implement the smallest generic handoff and
   safe existing framegrabber timing collection. Keep behavior and timestamp
   locations stable; remove allocation/reset work from steady live collection.
   Test disabled collection, wrap, consumer stalls, capacity loss, reconnect and
   configuration epochs before using new statistics for attribution.
4. **Add optional stage measurements.** Add the marker/result hooks at the
   actual EDT, OCAM and framegrabber boundaries, plus ordinary control-event
   timestamps. Follow the existing optional-hook pattern; other applications
   should retain their existing behavior without implementing new hooks.
5. **Operator comparison.** Keep the current 2 kHz/bin2 configuration and the
   isolated camera/acquWFS producer chain. Compare baseline, safe paired timing,
   and stage timing with consistent workload/IRQ/affinity. Warm buffers first;
   retain startup/reset separately. Compare at least two minutes with counter,
   SDK failure/recovery, queue-loss and coverage evidence. Check the observer
   receipt gaps too. Quantify added clock/event cost on the instrument before
   claiming acceptable overhead; no camera-specific overhead budget is settled.
6. **Apply isolated fixes supported by tests/evidence.** Following the failed
   acquisition guard above, address SDK-zero-ID lifecycle, scheduling error
   reporting and coherent configuration requests. Confirm software binning
   semantics before a separate pixel-result fix. Queue-size/policy, polling
   cadence, SDK owner-thread changes and pixel optimizations require their own
   comparisons; do not change them together with baseline instrumentation.
7. **Record results and maintain scope.** Keep source findings and measurements
   in this document, link significant conclusions back to the synco record, and
   update configuration/user docs with implemented behavior. Commit functional
   changes with their engineering record first, user docs next, and standalone
   formatting cleanup last as required by AGENTS.md.

Targeted workstation tests should extend the existing OCAM SDK/EDT harnesses
and use isolated shared-memory fixtures:

- Null/failed/partial wait, invalid timestamp and independent wait/rearm/convert
  delays; no invalid header read or successful publication on a failed attempt.
- SDK timestamp preservation and exact main metadata/post/sync-post ordering;
  post-publication delay attributed to the next iteration. Stage disablement
  must preserve results and avoid diagnostic clock calls.
- Queue full/consumer stall, warm-up/wrap, session transitions and concurrent
  configuration requests; meaningful concurrency checks and sanitizer coverage
  where supported by the SDK-free harness.
- Existing `telem_fgtimings` compatibility plus new stage/quality text, binary,
  validity and forced-record behavior, without producer-thread serialization.
- Zero-valued SDK ID, reinitialization failure, cleanup after worker shutdown;
  exact frame-number wrap/duplicate/backward/gap cases; asymmetric binning data.
- Inject a positive pthread scheduling error and verify honest reporting.
  Verify all users if extracting shared queue or changing framegrabber helpers.

Build the affected application with matching EDT headers/libraries in an
appropriate workstation environment, and run SDK-free OCAM lifecycle/control
tests, relevant framegrabber/queue and telemetry tests. Run clang-format and the
full changed-file documentation pass on touched C++ files during implementation.
Local functional tests do not qualify hardware jitter or buffer recovery.

## Local implementation record

### Shared ownership and population reuse

The implementation extracts [common/spscQueue.hpp](../../../libMagAOX/common/spscQueue.hpp)
from the ownership mechanism already used by `dev::dmTiming`. It uses fixed,
initialized storage and release/acquire positions: the producer cannot overwrite
an unread slot, never waits for the consumer, and counts full-queue losses.
Configuration/resizing is allowed only with both owners stopped.
`dev::dmTiming` now uses this helper with its existing public API and quality
semantics. [common/timingStats.hpp](../../../libMagAOX/common/timingStats.hpp)
provides the neutral population/accumulator; `dmTiming::Stats` remains an alias
with the same seven fields and units.

[common/frameGrabberTiming.hpp](../../../libMagAOX/common/frameGrabberTiming.hpp)
and [dev/frameGrabberTiming.hpp](../../../libMagAOX/app/dev/frameGrabberTiming.hpp)
provide one fixed event for original paired timestamps, optional local stage
markers, frame/output identity, numeric mode and digital factors, outcome and
acquisition generation. The consumer alone owns retained histories and computes
statistics. Controls have a separate queue whose producers are serialized by
existing camera control ownership; they do not write the acquisition producer's
state.

The old unsynchronized atime/writetime rings and exact-FPS-driven live storage
reset have been replaced. Existing `latencyTime` remains a trailing time window;
`latencySize` now directly caps each queue/history independently of FPS. With the
operator's values, that is a two-second window and a 20000-event capacity, not
the old FPS-derived 8000-timestamp allocation. Neither the EDT 64-frame DMA ring
nor output depth one is changed. FPS/reconfiguration transitions create explicit
epochs without reallocating storage during acquisition.

Paired-only collection adds no stage-marker clocks. Original realtime writetime
is retained unchanged in its publication position; ordinary-thread realtime and
monotonic anchors supply retention/age when stage markers are disabled. Records
published after an update's captured evaluation time remain retained for a later
update, rather than being discarded or flagged as bad clocks. The same rule
applies to completed control records. Zero/invalid external stamps are excluded
from their affected populations and counted; an unsuccessful attempt that
intentionally has no timestamp does not invent a clock failure.

### Compatible populations and optional attribution

Legacy acquisition/publication intervals use a separate preceding published
output with the same generation/stream identity and adjacent output counter.
Rejected acquisition attempts and raw camera-frame gaps do not censor an actual
long delivery interval. Diagnostic queue losses/output-counter discontinuities
break that interval chain and remain visible in quality. Strict consecutive
attempt/frame qualification remains separate for optional stage attribution.
A main output remains counted as published if the later sync hook fails; the
failure is counted independently. `fg_timing` and `telem_fgtimings` retain their
acquisition interval, publication interval and paired external write-minus-
acquisition meanings. The latter remains signed external-clock context, not
new proof of an absolute SDK latency.

Optional stage metrics cover the approved wait-entry/return, rearm, validation,
conversion, write-clock bracket, main post, sync post and loop-tail endpoints.
They preserve the separation between the existing metadata writetime and the
nearby monotonic endpoint. Raw outlier populations retain wide/unknown brackets;
conditional attribution requires both adjacent write brackets to fit a positive
configured bound. Signed per-frame segment changes and matched late/early
SDK-return/write-endpoint interval populations support the decomposition above.
Previous-publication-to-next-wait-entry populations retain the previous frame's
post-publication context. No stage clock or extra stage-event handoff occurs
when optional stages are disabled.

Completed serial operations provide their own duration, success, invalid/lost
counts and frame-overlap populations. Overlap is an association, not causation.
An operation still in progress is not yet in that channel: its overlap can appear
retrospectively while affected frames remain retained, and a transaction or
consumer delay beyond the retention/capacity limits can leave incomplete
coverage. These data do not assert continuous observation of active polling or
prove that a concurrent transaction delayed the camera.

### Public configuration, readbacks and telemetry

The implemented `[framegrabber]` options are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `stageTiming` | `false` | Enable optional local stage clocks and completed serial observations. |
| `stageExcursionUs` | `20` | Diagnostic classification distance from the retained monotonic write-endpoint interval median; zero disables classification. |
| `stageBracketUs` | `0` | Positive maximum acceptable write-clock bracket for conditional attribution; zero leaves attribution ambiguous. |

The excursion threshold is a measurement classifier, not a scheduler safety
margin. The bracket default deliberately leaves the accuracy criterion
undecided; raw stage timing can be inspected without selecting one. No new
syncopation margin or camera instrumentation overhead limit has been chosen.

`fg_stage_timing` publishes raw populations, signed conditional changes, matched
late/early interval cohorts, previous-tail context and completed serial
statistics, in seconds. `fg_timing_quality` publishes availability/state,
window/age/coverage, mode/identity, generations, queue populations, classification
counts and cumulative/delta quality counts. A forced cached record distinguishes
disabled, warming, stale and unavailable collection.

New `telem_fgstage` (event code 20906, `[fgstage]`) carries the typed summary and
quality without changing the existing `fgtimes` schema. Its construction and
logger serialization run on ordinary app logic/telemeter paths. The reader must
be rebuilt with the new schema. The public interface is described in
[frameGrabberTiming.md](../../../libMagAOX/app/dev/doc/frameGrabberTiming.md).

### Acquisition and lifecycle guards

The EDT acquisition wrapper initializes result timestamps, guards null returns,
and rejects zero/non-normalized SDK timestamps before any camera-header read.
It compares `pdv_timeouts()` against the baseline for the current acquisition
configuration and rejects a newly reported timeout/partial acquisition. Optional
wait/rearm/serial hooks are available without imposing stage clocks on callers
that do not enable them. Shared EDT callers propagate acquisition failures;
OCAM requests recovery instead of dereferencing an invalid image pointer.

OCAM frame continuity now uses unsigned subtraction to permit exactly the
`UINT_MAX -> 0` wrap, while duplicate, backward and nonunit transitions retain
explicit rejection/recovery. A separate SDK-initialized flag treats handle zero
as valid. Reconfiguration releases a prior initialization, failed setup cleans
up, and final SDK cleanup follows worker shutdown.

Control requests and their mode strings are protected by the INDI/control lock;
worker recovery preserves newer requested modes. Completion acknowledges only
consumed requests. An atomic FPS value supplies the worker without reading
control-owned float storage. `setSynchro()` propagates the reconfiguration choice
through `setFPS()`, so acquisition configuration does not generate another
request while an actual control update still requests its restart.

`MagAOXApp::threadStart()` now checks `pthread_setschedparam()` with `rv != 0`
and reports the returned error code, without trusting errno after privilege
restoration. It retains the helper's existing continuation/return behavior;
requested INDI priority still does not substitute for verifying actual runtime
policy and CPU placement.

### Scope deliberately deferred

- Software-binning pixel policy and the EDT same-thread resource-lifecycle
  topology remain deferred. This implementation does not change science-pixel
  binning, DMA queue size/latest-frame policy, serial polling cadence or IRQ
  placement.
- The copied SDK header and library identify 5.6.8.0, correcting the earlier
  5.6.2.0 report. Core source inspection and matching OCAM compilation/linkage
  are complete; the findings are recorded below. The bulk SDK copy need not
  finish for those checks. Loaded driver/binary provenance, kernel initialization
  and actual query latency remain runtime qualification facts.
- Hardware overhead and jitter comparisons remain pending, including the
  configured priority 90/cpuset and loaded driver/IRQ verification. SDK-free
  workstation tests cannot qualify those results or prove buffer-recovery
  behavior on the instrument.
- `stageBracketUs` and acceptable instrumentation overhead must be selected
  through accuracy/performance qualification. No unresolved early-arrival
  reserve or other syncopation scheduling decision is resolved by this work.

The failed-result guard is active even with stages disabled: it queries the EDT
cumulative timeout counter once per non-null wait and once at acquisition start.
The per-frame query follows `rearmReturn`, so its measured cost belongs to
`validation`. Stage-off/stage-on comparison isolates optional instrumentation;
a pre-change-binary comparison is also required to qualify the guard and repaired
paired-handoff baseline cost.

### Verification record

All **16 focused suites passed: 5,737 assertions in 109 cases**. Tests use
SDK-free camera stubs and isolated local shared-memory fixtures; no instrument
or tunneled INDI commands were run.

| Suite | Assertions | Cases |
| --- | ---: | ---: |
| `MagAOXApp_threadStart_test` | 56 | 1 |
| `cred2Ctrl_lifecycle_test` | 13 | 1 |
| `cred2Ctrl_test` | 405 | 11 |
| `dmTimingTelemetry_test` | 1000 | 7 |
| `dmTiming_integration_test` | 153 | 8 |
| `dmTiming_test` | 317 | 15 |
| `frameGrabberTelemetry_test` | 2236 | 5 |
| `frameGrabberTiming_test` | 92 | 9 |
| `frameGrabber_integration_test` | 54 | 4 |
| `ocam2KCtrl_lifecycle_test` | 7 | 1 |
| `ocam2KCtrl_temperature_test` | 69 | 3 |
| `ocam2KCtrl_test` | 607 | 30 |
| `shmimMonitor_timing_test` | 42 | 4 |
| `spscQueue_test` | 40 | 3 |
| `stdCamera_roi_test` | 529 | 3 |
| `stdCamera_test` | 117 | 4 |

The library/schema generation, `dmSyncopation`, `logdump` and the normal
`ocam2KCtrl` application build/link all pass. The camera build uses the copied
5.6.8.0 headers/library, with no test include directory or SDK stubs:

```bash
make -C apps/ocam2KCtrl EDT=true \
    'EDT_INCLUDES=-I/home/jrmales/Source/EDTpdv -I/home/jrmales/Source' \
    'EDT_LIBS=-L/home/jrmales/Source -lpdv -lpthread -lm -ldl' -j2
```

That binary was not launched. The copied library's embedded version strings
also identify 5.6.8.0. The normal camera build retains the existing warning from
passing `-std=c++20` to the vendor OCAM C source; generated git-status pragmas
report the intentionally modified/untracked worktree. Schema generation has
pre-existing naming warnings in other schemas. Focused C++ test builds have no
new compiler warnings.

Standalone ASan+UBSan and TSan stress checks pass with 300000 coherent multiword
queue transfers and 200000 mixed frame attempts plus 40000 controls over
concurrent snapshots/epochs, including explicit loss accounting and expiry.
LeakSanitizer could not run under the environment's ptrace setup; the ASan/UBSan
run used `detect_leaks=0`. Source/log artifacts are `/tmp/fgstage-sanitizer*`.
A permanent concurrent collector regression is included in the normal suite
(10000 attempts, 3000 controls, bounded queues, epochs and stopped reconfigure).

All 20 touched C++ files pass full-file clang-format verification, and
`git diff --check` passes. The formatter now preserves the mandated
`{ //mutex scope` annotation. Relative documentation links were checked.

Review corrected diagnostic semantics before the passing run: paired delivery
gaps remain visible independently of stricter stage continuity, a main output
still counts as published after a later sync-hook failure, and events captured
after a consumer's evaluation timestamp remain retained for the next update.
The CRED2 fixture was corrected to exercise its actual ROI-lock handoff and
explicit offline telemetry dependency, with every worker joined before
assertions. This required no production CRED2 locking change.


The added/extended suites cover generic handoff ownership and concurrent payloads,
unchanged DM collection, paired-only behavior, rejected-attempt delivery gaps,
late-hook failure, stage decomposition/ambiguous brackets, control overlap,
future-at-evaluation retention, forced telemetry/validation, SDK null/timeout/
timestamp handling, SDK ID zero, exact raw wrap, request acknowledgement and
positive scheduling errors. SDK-free OCAM/EDT and framegrabber integration tests
exercise the new interfaces without contacting a camera.

No agent instrument command or deployment was performed. The operator approved
committing and pushing this verified implementation after reviewing the result.

## Remaining review and qualification items

- Matching copied 5.6.8.0 compilation/linkage and documented timestamp units
  are verified. Confirm the instrument's loaded SDK/driver builds and kernel
  timestamp initialization/default through operator evidence.
- Active bin2/digital factors are now confirmed from the current local config.
  Confirm actual worker/IRQ scheduling, then
  perform the controlled baseline/safe-paired/stage-enabled comparison with the
  same 2 kHz isolated camera/acquWFS producer chain.
- Establish acceptable clock bracket and instrumentation overhead criteria;
  retain raw excursions and incomplete coverage rather than censoring them.
- Review any later software-binning, SDK owner-thread, polling or DMA-policy
  change as a separate scoped task with its own tests and operator comparison.

## Copied PDV 5.6.8.0 contract inspection and matched build

The operator prioritized the core SDK files after the bulk copy had begun.
The audited copies are `/home/jrmales/Source/libpdv.c`, `libedt.c`, `libpdv.h`,
`libedt.h` and `libpdv.so`; other copied headers remain under
`/home/jrmales/Source/EDTpdv`. These locations matter because the core files
arrived in the parent directory rather than inside `EDTpdv`.

The copied `EDTpdv/edtdef.h:13` identifies version **5.6.8.0**; read-only string
inspection of the copied `libpdv.so` also contains **5.6.8.0**. The installation
log records that version at `EDTpdv/install_pdv.log:1724` and a November 12, 2025
installation date at line 1584. This supersedes the earlier operator recollection
of 5.6.2.0 for identifying the copied SDK. It does not establish the kernel
module or shared library currently loaded on the instrument.

The normal OCAM application has now compiled and linked against these actual
5.6.8.0 headers and shared library, using the two source locations above. The
coordinator's build log is `/tmp/ocam-sdk-app-build.log`; no application, installer
or device access was executed. This resolves matching-SDK build availability,
while runtime provenance and performance qualification remain operator work.

Source snapshot identification (SHA-256):

```text
libpdv.c   6e8dc7697dc7ecc54356e78d40c148e454d55d7c8cc31ddc26dd60cceb836fe5
libedt.c   b4a8520b61e63a80b7328117449627f0c0c6c2052a2ff47c1263d72c8a301d71
libpdv.so  05fdcfb69ab7bd139fb1e4e6cc60a80ebe793f460a01b3e8273f3f6e7c00456c
```

### Timestamp units, event and default-mode limitation

The copied API's actual default format is seconds plus **nanoseconds**:

- `libedt.h:2005` defines `EDT_TM_SEC_NSEC` as zero.
- `libpdv.c:5272–5279` explicitly describes that default and implements
  `timestamp[0] * 1e9 + timestamp[1]` before conversion to seconds.
- `libedt.c:2515–2518` documents a seconds/nanoseconds DMA-completion stamp;
  lines 2558–2565 retrieve its two fields through `EDTG_TMSTAMP`.

Some PDV Doxygen paragraphs (`libpdv.c:4968` and 5221) still say microseconds.
They conflict with the actual decoder and lower-level API contract and are
stale documentation, not a reason to multiply the application's `tv_nsec` by
1000. The existing direct field assignment is correct for the documented
`EDT_TM_SEC_NSEC` mode.

`libpdv.c:4971–4976` describes a host system-clock/gettimeofday timestamp around
DMA completion, including interrupt/response latency. It is not an exposure
start or an independent hardware-only clock. `libpdv.c:5213` obtains the stamp
for the selected completed buffer (`donecount - 1`), rather than sampling the
user-thread return time.

The available library source contains only the mode setter
(`libedt.c:6708–6710`, `EDTS_TIMETYPE`); no MagAOX app sets that mode. The decoder
labels alternative modes experimental/unsupported. Kernel initialization/reset
of the field has not yet been traced, because the kernel-module sources were
not part of the prioritized core files. Thus the documented default and its
numeric definition are verified, while initialization of the instrument's
loaded driver is not. Retain that distinction when qualifying absolute
SDK-atime-to-realtime-writetime latency; never subtract either external stamp
from monotonic stage endpoints.

### Latest-buffer selection and skips

`pdv_wait_last_image_timed()` delegates to its `_raw(..., FALSE)` variant
(`libpdv.c:5238–5240`). The implementation at 5197–5213:

1. Queries the driver's completed-buffer count.
2. Subtracts the user handle's previously consumed count.
3. Uses one when no new completed frame was ready at that query.
4. Waits for that delta and returns its last buffer, with that buffer's stamp.

It can therefore skip older completed buffers and select the newest one seen
at entry. It does not continuously resample the newest buffer until the user
function returns. If none was ready, the call waits for one new completion.
Its returned pointer and timestamp describe that selected image. Raw OCAM
frame-number continuity remains necessary to expose skipped images, but does
not by itself prove small DMA-completion-to-user-publication age.

The path also includes library work around the actual DMA wait: completed-count
and timeout queries, wait-status retrieval, optional frame-sync/deinterleave
work and timestamp retrieval. Consequently the local `waitReturn` marker is an
SDK-function return boundary, not the exact kernel wakeup instant.

### Timeout/frame-sync guard and query cost

`pdv_timeouts()` (`libpdv.c:1090–1097`) calls `edt_timeouts()`;
`libedt.c:4329–4336` performs one `EDTG_TIMEOUTS` ioctl and returns the cumulative
count. It performs no additional DMA wait. It is nevertheless an additional
kernel query per checked frame, with SDK debug-dispatch overhead, rather than
a free cached counter read. Its actual duration and possible kernel/device
contention must be included in the operator overhead comparison.

The existing timed-image path already queries the count in
`pdv_wait_images()` (`libpdv.c:5687`) and its nested raw implementation (5580).
The initial application result guard added one more query after `rearmReturn`,
placing its time in the rearm-to-validation segment of those captures. The
2026-10-24 deferred-rearm implementation below measures it independently as
`timeoutQuery` before publication and validation.

The active mode's `fv_once=1` and `method_framesync: EMULATE_TIMEOUT` make this
check relevant. `libpdv.c:1054–1064` explains that continuous/fv-once operation
can otherwise stay out of sync without a normal timeout. The raw wait checks
footer magic (`libpdv.c:6107–6119`); on failure, with no existing new timeout,
it invokes `edt_do_timeout()` (`libpdv.c:5615–5618`). That helper sends
`EDTS_DOTIMEOUT` (`libedt.c:2332–2337`); its documented behavior includes FIFO
flush/DMA abort (2315–2322). This is distinct from the separate
`edt_inc_timeout()` helper, which only increments a count. The application's
new-count rejection therefore also covers the configured frame-sync error
signal; it should not be described as detecting only an ordinary expired wait.

The SDK can return a buffer pointer even with a timeout status
(`libedt.c:2481–2510`), supporting the need for more than a null-pointer guard.
Its legacy wrappers do not propagate every ioctl failure cleanly:
`edt_timeouts()` does not initialize its local result or check the ioctl return,
and `edt_get_timestamp()` does not propagate its ioctl result before copying
fields. Normalized/nonzero-stamp and timeout-count checks improve the app's
result contract without claiming complete diagnosis of arbitrary SDK/driver
I/O failures. Recovery remains explicit rather than treating those results as
successful science frames.

### Thread/lifetime contract and retained scope

The matching 5.6.8.0 source itself includes the same-thread restriction:
`libpdv.c:120–122` places start/wait/abort operations in one thread, and
`libedt.h:198–204` documents open/DMA/close ownership in one thread. The old
TGID change noted in `CHANGELOG_LINUX.txt:392–394` is historical implementation
context; it does not override this contract or prove that the present cross-
thread resource lifetime is supported.

The implementation does not change the EDT owner-thread topology, DMA/latest-
frame policy, pixel binning or polling cadence on this evidence alone. Those
remain separate review and operator-qualification tasks. No scheduling margin,
write-bracket limit or acceptable instrumentation overhead has been selected
by inspecting the SDK.

## 2026-10-23 first operator camera-stage capture

The operator supplied
`/home/jrmales/.codex/attachments/7332fbdb-2b8b-4413-9bf2-67a0c97ee839/Pasted text.txt`.
It contains three `[fgstage]` records at October 24 05:27:54.969–05:27:59.081
UTC (October 23 22:27:54.969–22:27:59.081 Phoenix time), session 5, sequences
34–36. All show `state=valid`, `enabled=1`, `stages_enabled=1`,
`fg_valid=1`, a two-second window and approximately 4000 published frames /
3999 adjacent intervals. Mode is SDK mode 3, with digital factors 1x1; the
ordinary camera telemetry identifies bin2, sync On and 2000 FPS.

Ranges below are across the three independently reported windows. Extrema
from different rows or metrics are not matched frame observations.

| Metric | Mean, us | Window standard deviation, us | Overall observed minimum–maximum, us |
| --- | ---: | ---: | ---: |
| SDK acquisition interval | 500.000250–500.000500 | 0.318670–0.346496 | 488.000–512.000 |
| Realtime publication interval | 500.000256–500.000582 | 1.088327–2.017938 | 424.882–572.732 |
| SDK wait-return interval | 500.000464–500.000635 | 0.870343–0.916738 | 463.284–537.705 |
| SDK acquisition to realtime write | 52.055944–52.078971 | 1.077642–1.478071 | 50.056–127.623 |
| Rearm | 4.751741–4.752506 | 0.036111–0.044226 | 4.699–6.572 |
| Validation | 0.363483–0.365375 | 0.074861–0.079971 | 0.310–0.922 |
| Conversion/load | 26.368864–26.392050 | 0.800376–0.991539 | 24.646–64.261 |
| Write-clock bracket | 0.049020–0.049169 | 0.004565–0.004807 | 0.040–0.100 |
| Metadata/main notification | 3.755463–3.759527 | 0.604713–0.608384 | 3.186–41.088 |
| SDK return to write endpoint | 31.533744–31.558336 | 0.809346–1.012921 | 29.777–69.872 |
| Previous write endpoint to current wait entry | 4.856048–4.864455 | 0.617117–0.619596 | 4.118–42.230 |

The compatible `[fgtimes]` statistics agree with the new paired populations.
Monotonic write-endpoint intervals agree closely with realtime publication
intervals. At the last snapshot, all 153981 reported attempts are published,
with cumulative rejected, failed, duplicate, counter-gap, raw-frame-gap,
invalid and dropped counts zero. Completed-control failure, invalid and loss
counts are also zero. One queued frame is a small reader backlog, not a
reported loss. These counters establish the coverage reported by this collector,
not an independent proof of physical camera completeness or negligible overhead.

The SDK timestamp intervals are much steadier than the output timestamps in
these windows. Return intervals already have excursions of about +/-37 us;
conversion/load and the post-write main-notification segment also have large
individual duration tails. This narrows candidates but does not assign any
particular output excursion to a segment. Do not subtract the independent
RMS values or pair independent maxima. The main-notification segment follows
the current write timestamp, so its effect must be evaluated against the next
frame, not attributed to delaying that already sampled timestamp.

All records have `bracket_limit_s=0`, `qualified=0`, `ambiguous=3999`, and
zero late/early cohort counts. This is the intended undecided-bracket behavior,
not evidence that the visible approximately -75/+73 us publication excursions
are absent. Raw populations are retained; the matched conditional tables are
unavailable until a positive bracket criterion is configured.

For the next operator comparison, a **provisional diagnostic**
`stageBracketUs=1` is proposed: the observed brackets are only 0.040–0.100 us.
Retain `stageTiming=true`, `stageExcursionUs=20`, and the same camera mode,
workload and placement, then restart the camera app for this restart-only
setting. Check qualified/ambiguous counts and all losses with the new late/early
cohorts before assigning a segment. This proposed one-microsecond bracket is
an accuracy filter around the existing write-clock sample; it is neither an
early-arrival reserve nor an approved instrumentation-overhead limit. No local
configuration or instrument setting was changed for this analysis.

Two completed serial transactions are present per window, each lasting about
0.520–0.534 seconds; 1998 of 4000 frame observations overlap completed serial
transactions. This measures concurrency, not half-second acquisition stalls.
No causal relation to the outliers is established. Completed transactions alone
provide retrospective overlap coverage; retain that limitation when inspecting
the future conditional serial-change populations.

Only engineering documentation changed for this analysis. Numerical fields
were parsed from the attachment and compared with collector/guide semantics;
whitespace validation passed. No C++ changes, test/build reruns, commit, push or
agent instrument actions were performed.

## 2026-10-23 qualified operator camera-stage capture

The next attachment is
`/home/jrmales/.codex/attachments/61217b8a-d747-43cb-97f0-765ab06ab792/Pasted text.txt`.
Its three records are at October 24 05:34:55.320–05:34:59.433 UTC (October 23
22:34:55.320–22:34:59.433 Phoenix time), session 5, sequences 40–42, stream
inode 3322. Each has approximately 4000 outputs, 3999 intervals, and
`bracket_limit_s=1e-06` with `qualified=3999`, `ambiguous=0`. The 20 us
excursion threshold and bin2/mode-3/digital-1x1/2000-FPS settings remain visible
in the records. The stream identity differs from the preceding raw capture,
consistent with a restart/recreation; these are separate runs.

The windows span approximately 1.9995 seconds and their capture endpoints are
over 2.056 seconds apart. Their retained populations therefore do not overlap.
Together they contain 11997 qualified intervals, four late and six early
intervals. Cumulative attempts/published reach 200723 with rejected, failed,
duplicates, counter gaps, frame gaps, invalid and dropped counts zero. Control
failure/invalid/drop counts are zero. `bracketWide=4` is unchanged cumulatively,
with zero new wide brackets in each reported delta and none excluding the
current windows; retain those earlier observations rather than interpreting
this sample as a lifetime bound on bracket width.

The matched populations now support the identity, in microseconds:

```text
write endpoint interval = SDK return interval + change in return-to-write duration
```

| Sequence/cohort | Count | Write interval | Return interval | Change in return-to-write | Change in conversion/load | Previous post-write tail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 late | 1 | 545.249 | 508.139 | +37.110 | +36.790 | 4.609 |
| 40 early | 2 | 465.2485 | 502.369 | -37.1205 | -37.806 | 5.0895 |
| 41 late | 2 | 534.639 | 514.036 | +20.603 | +20.384 | 23.3485 |
| 41 early | 2 | 466.952 | 504.4675 | -37.5155 | -38.047 | 4.824 |
| 42 late | 1 | 577.591 | 540.140 | +37.451 | +36.780 | 43.743 |
| 42 early | 2 | 458.9415 | 495.1545 | -36.213 | -36.2485 | 4.7395 |

All entries are cohort means with matching counts, not independently paired
extrema. All six identities agree to the printed precision; rearm, validation
and write-bracket signed changes also sum with conversion to the reported
return-to-write change. For a one-event population, `valid=0` means that sample
variance is unavailable, not that the event's duration or mean is unusable.

The largest late event is particularly informative because its cohort has one
event. Relative to 500 us, the publication interval is +77.591 us: +40.140 us
is already present between SDK returns, and +37.451 us is added through a
longer current return-to-write duration. Conversion/load contributes +36.780 us,
rearm +0.661 us and validation +0.010 us, with unchanged write bracket.
The same event has a 43.743 us previous post-write tail, versus this window's
ordinary mean of 4.865 us. Its current main-notification duration is 39.074 us
shorter than its preceding frame's, consistent with a preceding notification
tail. These facts support contributions in both the preceding post-write path
and current conversion region; they do not prove which operating-system or
device activity caused either elapsed-time increase.

The early cohorts have conversion/load contractions of approximately 36–38 us
and SDK-return intervals near 500 us on average. They strongly associate the
short output intervals with a shorter current conversion/load duration than
the preceding frame. This is not evidence that every early interval directly
follows a classified late interval: the summary does not retain that ordering,
and the previous interval could lie below the 20 us classification threshold.
Four late versus six early intervals must not be read as universal pairing.

The current active path has no extra digital binning: the conversion/load
bracket surrounds the write-flag setup and `ocam2_descramble()` path in
`loadImageIntoStream()`. It measures wall time, including preemption, interrupts,
memory/cache effects and instrumentation between its endpoints. It does not
show that the descrambler itself performs approximately 37 us of extra
instructions. Distinguishing additional on-CPU work from interruption during
that bracket is the next causal question. Retain this configuration for a
longer sample before changing pixel code, polling, DMA policy or scheduling.

Serial durations and approximately half-window overlap persist. Excursion-
conditioned serial-overlap populations are unavailable in these records, so no
serial/outlier correlation or causal conclusion follows from the overlap
count. Existing stage measurements locate elapsed-time changes, not a root
cause or overhead qualification. No scheduler reserve is selected.

The attachment was parsed offline and all six matched-mean identities and
segment sums were checked numerically. Only engineering documentation changed;
whitespace checks passed. No functional change, new test/build run, commit,
push or agent instrument action was performed for this analysis.

## 2026-10-23 longer qualified camera-stage capture

The operator supplied `/home/jrmales/camwfs_fgstage.txt`, SHA-256
`59a29b768f137f4a7585aa2efc72ec171467bd7ee0e06ca1beedfcf314081241`.
ANSI terminal escapes were removed for offline parsing. The file contains 35
stage records, sequences 199–233, at October 24 05:40:22.096–05:41:31.985 UTC
(October 23 22:40:22.096–22:41:31.985 Phoenix time). Session 5, stream inode
3322, SDK mode 3, digital factors 1x1, two-second windows, 20 us excursion
threshold and provisional 1 us clock bracket remain consistent.

The record endpoints span 69.889 seconds. Each retained window contains 4000
published frames and 3999 adjacent intervals. Endpoint spacing of
2.049–2.057 seconds makes these populations disjoint, with approximately
50–57 ms gaps between windows; the summaries do not provide continuous
per-event coverage. The sample comprises 140000 retained frame observations
and 139965 intervals, all qualified with zero ambiguous intervals. There are
69 late and 84 early intervals, respectively 0.0493% and 0.0600% of the retained
interval population at the configured median +/-20 us classifier. The median
ranges from 499.984 to 500.024 us.

All cumulative reject, failure, duplicate, output-counter gap, raw-frame gap,
invalid and diagnostic-drop counters remain zero. Control failure/invalid/drop
counters also remain zero. First-to-last output and raw frame counters advance
139778 over 69.889 seconds, consistent with the nominal 2 kHz stream. Cumulative
wide brackets increase from 71 to 73, and the reported deltas also sum to two
new wide brackets. However, all retained brackets are at most 0.241 us and
every retained interval qualifies: the two new wide observations are outside
the retained windows. They remain visible in lifetime accounting and must not
be silently treated as never occurring.

The following pooled standard deviations combine each disjoint population's
count, mean and within-window sample variance; they are not averages of the
printed RMS values. All durations below are microseconds.

| Interval | Pooled mean | Pooled standard deviation | Observed minimum–maximum |
| --- | ---: | ---: | ---: |
| SDK acquisition | 500.000429 | 0.359410 | 480.000–525.000 |
| SDK return | 500.000440 | 0.868726 | 462.604–538.066 |
| Realtime publication | 500.000428 | 1.603656 | 424.031–575.917 |

Acquisition is much steadier statistically, but is not excursion-free: the
525 us acquisition interval occurs in sequence 232. Monotonic write-endpoint
statistics agree closely with the realtime publication statistics. SDK
acquisition-to-write context averages 52.815 us, spanning 37.344–129.017 us;
its external-clock interpretation remains subject to the earlier provenance
qualifications.

Matched late means, weighted by the 69 cohort observations, decompose as:

```text
543.265362 = 517.381391 + 25.883971 us
write interval = return interval + change in return-to-write duration
```

Conversion contributes +22.537377 us of the mean +25.883971 us post-return
change; rearm contributes +2.837391 us, validation +0.508217 us and the clock
bracket +0.000986 us. Across the 84 early observations, the identity is:

```text
464.690226 = 490.576964 - 25.886738 us
```

Conversion contributes -25.265119 us of that mean contraction. Thus conversion
remains a frequent contributor, but these pooled means cannot establish that
each outlier has the same cause or location.

The longer capture supplies decisive counterexamples to conversion-only
attribution. Each example below has exactly one late event in its window:

| Sequence | Write interval | Return interval | Change after return | Relevant same-cohort evidence |
| --- | ---: | ---: | ---: | --- |
| 200 | 536.773 | 500.876 | +35.897 | Rearm increases 36.749 us; conversion decreases 1.321 us. |
| 208 | 537.755 | 537.905 | -0.150 | Almost all interval extension is already present at SDK return. |
| 209 | 575.477 | 537.344 | +38.133 | Conversion increases 36.349 us; previous post-write tail is 41.499 us. |
| 214 | 575.917 | 537.385 | +38.532 | Conversion increases 37.440 us; previous post-write tail is normal at 4.719 us. |

Large individual elapsed durations also occur in several different regions:

| Region | Pooled mean, us | Observed maximum, us |
| --- | ---: | ---: |
| Rearm | 4.754 | 41.759 |
| Validation | 0.362 | 37.090 |
| Conversion/load | 27.091 | 66.536 |
| Metadata/main notification | 3.767 | 41.508 |
| Sync notification | 0.790 | 37.992 |
| Loop tail | 0.153 | 29.205 |

The recurring approximately 36–39 us additions in multiple regions support
an intermittent interruption or shared-resource delay as a working hypothesis,
rather than a unique conversion algorithm path. They do not identify a
specific interrupt, scheduling event, firmware activity, SDK operation or
memory-system cause. Regional maxima are not matched to each other; the
single-event examples above are the stronger per-event evidence. Wall-time
markers include interruption as well as actual execution, and post-write
notification work must still be related to the following frame.

Seventy completed serial transactions average 526.561 ms (518.938–534.682 ms
extrema), with 1998 overlapping frame observations per retained window, or
49.95% of those observations. These are successful completed control envelopes,
not camera stalls. The capture still lacks excursion-conditioned serial
overlap and ordered neighboring event records, so it cannot establish control
causation or that every early interval follows a late one.

This sample is sufficient to establish the recurring multi-region pattern.
The next useful diagnostic is a bounded chronological record of excursions
and their neighboring frames, with marker timestamps, identities and clock
references suitable for an operator-controlled camera-CPU scheduler/IRQ
comparison. That would preserve event order and help distinguish interruption
from extra execution. The causal experiment and its overhead require their
own review; no new instrumentation, pixel change, polling change, scheduling
change or DMA-policy change is implemented here. The early-arrival reserve
remains undecided.

Both an independent offline parse and the coordinator's parse agree on counts,
pooled means, variances, extrema and matched examples. All 70 matched cohort
identities and corresponding segment sums were checked to the printed
precision. Offline analysis files are `/tmp/analyze_ocam_fgstage.py`,
`/tmp/ocam-long-fgstage.json` and `/tmp/ocam-long-summary.json`. Only engineering
documentation changed; whitespace checks passed. No functional changes,
test/build reruns, commit, push or agent instrument actions were performed.

## 2026-10-23 chronological excursion and scheduler/IRQ setup

The operator approved setting up the causal measurements. The implementation
adds optional `[fgtrace]` records and the
[operator capture procedure](../../../apps/ocam2KCtrl/doc/camera-timing-trace.md),
without changing pixel processing, SDK/DMA policy, cadence, affinity or
scheduling policy. No instrument or tunneled INDI command was run by an agent.

### Consumer-only chronological capture

`framegrabber.stageTraceMax` defaults to zero. A positive value, at most 256,
bounds ready plus pending excursion contexts; 16 is the proposed initial
operator setting. It requires stage timing, a positive timing window, at least
four timing slots and `stageExcursionUs >= 0.001` (one nanosecond). Existing
`stageTiming=true`, `stageExcursionUs=20` and the provisional
`stageBracketUs=1` remain the comparison settings. All are restart-only.

The existing producer event layout and acquisition methods are unchanged.
Selection, context retention and logging run on the ordinary consumer.
Each selected trigger keeps two preceding and two following continuous frames
in five fixed positions, including original SDK/realtime stamps, every available
integer monotonic endpoint, identities, mode/outcomes and marker validity.
The presence mask and `complete`, `boundary` or `expired` reason expose missing
neighbors. Pending following frames survive ordinary updates; ready records are
emitted once and cleared, without a periodic forced replay. At full capacity,
the newest trigger is dropped and counted.

The causal baseline is the median of at least three strictly preceding
continuous intervals within the configured time/capacity history. Its exact
integer value and population are stored per trigger. It need not equal the
whole-window median in `[fgstage]`. A preallocated sorted workspace and fixed
interval ring maintain that baseline on the consumer; insertion/removal is
linear in retained history size. Warmup and continuity breaks remain counted.
Selection examines consumed events before summary expiry, preserving events
that fall between the approximately 2.055-second summary updates. Wide or
undecided write brackets still produce raw excursion records with independent
qualification rather than censoring those outliers.

New typed telemetry `telem_fgtrace` is event **20907**. Binary, JSON, text and
metadata preserve the five contexts, exact monotonic ns, PID/acquisition TID,
selection baseline, consumer evaluation and ordinary batch-emission times, and
bounded/source loss status. Available completed-control overlap is attached at
finalization; it is not an assertion of in-flight control coverage. Queue losses,
trace saturation and incomplete contexts are reported, with additional trace
readbacks in `fg_timing_quality`. The ordinary wrapper also counts discarded
records if its emission clock fails or telemetry reports an error; it does not
claim durable logging acknowledgement. Source frames repeated in overlapping
contexts must be deduplicated by session/stream/attempt identity when counting.

### Operator-controlled kernel trace

`apps/ocam2KCtrl/scripts/capture-camera-timing.py` creates only its own uniquely
named tracefs instance. It requires the camera TID, captures all tasks on one
to four selected CPUs including every CPU allowed for that TID, selects the
matching `mono`/`CLOCK_MONOTONIC` clock, and takes a bounded 1–30 second capture
(10 seconds initially). It enables scheduler and IRQ/softirq events, plus
available x86 IRQ-vector events; timer/workqueue events are an optional separate
comparison. It changes no worker/IRQ affinity or scheduler policy, leaves global
and unrelated tracing untouched, and does not modify potentially shared trace
options. CPU tracing can miss a wakeup emitted on an unselected remote CPU;
that coverage limit remains explicit.

The helper saves the stopped trace, per-CPU loss counters, event formats,
clock/options/filters, boot/kernel identity, thread and IRQ placement, CPU
topology, before/after interrupt counts and clock anchors. Initial metadata
records the private instance path. Ordinary interruption stops/saves/cleans up;
save or stop/disable errors preserve that instance and report its exact path.
The procedure specifies baseline tracing-off/on telemetry, checks loss and CPU
coverage, and describes recovery of only the owned instance. Missing kernel
events, trace losses, capture boundaries and invisible NMI/SMI/shared-resource
stalls limit causal and negative conclusions. Hardware overhead remains to be
measured by the operator.

### Workstation verification

Seven C++ suites pass **4013 assertions in 72 cases**:

| Suite | Assertions | Cases |
| --- | ---: | ---: |
| `frameGrabberTiming_test` | 193 | 15 |
| `frameGrabberTraceTelemetry_test` | 461 | 4 |
| `frameGrabber_integration_test` | 98 | 6 |
| `ocam2KCtrl_test` | 613 | 30 |
| `ocam2KCtrl_lifecycle_test` | 7 | 1 |
| `cred2Ctrl_test` | 405 | 11 |
| Existing `frameGrabberTelemetry_test` | 2236 | 5 |

Tests cover ordered late/early neighborhoods, causal baseline warmup, events
outside retained summary windows, pending following frames, all continuity
boundaries, expiry, raw wide-bracket retention, source and capacity losses,
concurrent handoff, and complete/partial production records through the actual
logger verifier. Camera integration compares producer clock counts with
individual capture off/on and confirms no additional acquisition clock reads.
Each emitted context is delivered only once, including wrapper-error injection.
The initial integration expectation of a compensating early interval was
corrected: that fixture advances its SDK return relative to loop execution,
so a single conversion delay produces one late interval. Separate fixed-grid
collector tests exercise late-then-early order. A review found sub-nanosecond
threshold truncation; configuration rejects it and the standalone collector
defensively disables tracing for that unusable range, with regression coverage.

The capture helper passes **seven fake-filesystem tests**, including setup,
save and stop/disable failures and preservation of unrelated traces. No test
accesses a live tracefs instance. The normal library/schema and `logdump` builds
pass; the normal camera app compiles and links against the copied PDV **5.6.8.0**
headers/library. No camera binary was launched. Build and suite logs are under
`/tmp/fgtrace-`.

ASan+UBSan and TSan stress runs pass with chronological capture enabled:
300000 coherent queue payloads, 200000 mixed acquisition attempts and 40000
completed controls across concurrent updates/epochs. They check ready+pending
bounds and exact trigger-to-neighbor interval identities alongside existing
loss and summary invariants. ASan uses `detect_leaks=0` because of the previously
recorded environment restriction. Stress artifacts are `/tmp/fgtrace-sanitizer.cpp`
and `/tmp/fgtrace-asan*` / `/tmp/fgtrace-tsan*`.

Full-file clang-format, whitespace and relative documentation-link checks pass.
The normal SDK build reports existing git-status pragmas; no new focused-test
compiler warnings were found. No deployment or agent instrument action was
performed.

### 2026-10-24 commit/push checkpoint

The operator approved committing and pushing this verified follow-up to the
existing `jrmales/dm-syncopation` feature branch at `origin` (`magao-x/MagAOX`).
Commit the measurements, helper, tests and engineering records first, followed
by the user guide and operator procedure as documentation-only changes. No
standalone formatting cleanup is needed. Hardware correlation and overhead
qualification remain operator work.

## 2026-10-24 first paired camera/kernel capture

The operator supplied the local directory
`/home/jrmales/tmp/camwfs-trace-20261024T215606Z`. Analysis was performed on
those copied files only. The two telemetry files each contain 29 disjoint
two-second `[fgstage]` windows and 464 chronological `[fgtrace]` records.
The off/on labels refer to kernel tracing; stage and neighborhood telemetry
remain enabled in both files.
Both retain session 5, stream inode 3333, mode 3/bin 1x1, PID 2874366 and
acquisition TID 2874370. Camera telemetry reports synchronized 2000 FPS in bin2;
the 1x1 values describe additional software binning. No new rejected/failed
publications, frame/counter gaps or frame/control queue losses are reported.

Input SHA-256 values:

| File, relative to the supplied directory | SHA-256 |
| --- | --- |
| `camwfs-trace-off.txt` | `6f83e87de88ccc4412c9505ccab001f2421bebae1c09f64ad4129058abb86db3` |
| `camwfs-trace-on.txt` | `5733f2a854c0b17999a10c921fdfda95b6626c58d896ad32a5a0b43f7b611fcf` |
| `camwfs-kernel-20261024T215606Z/trace.txt` | `140946ffaf119b77be88dd30b4aab5a6c53512119b896eb531fdb79bc185cfbb` |
| `camwfs-kernel-20261024T215606Z/metadata.json` | `c092bcf491b3f813fe72d617f4747f45ea1ea0c27746e469468a4d0b7d88c3c0` |

### Coverage and diagnostic selection

The requested kernel capture covers monotonic 1746805.693291 through
1746815.693429, approximately **21:56:07.103166–21:56:17.103304 UTC**.
The CPU 33 ring overwrote **103364 of 313487 events**. Its retained tail has
210123 events from **1746809.003536 to 1746815.693541**, or **6.690005 seconds**.
`dropped events=0` and `commit overrun=0` do not cancel the reported overwrite.
The retained suffix supports positive event correlations; the missing first
3.31 seconds cannot support negative conclusions. Kernel text timestamps have
one-microsecond precision, despite the camera's integer-nanosecond endpoints.

The actual acquisition thread is **SCHED_FIFO 90 on CPU 33**, and the EDT
device's **IRQ 405 is also on CPU 33**. Placement is unchanged before/after;
CPU 33 has no SMT sibling. Only CPU 33 is traced, so execution of the main
control thread on CPUs 78/79 is outside coverage. Absolute monotonic endpoints,
not telemetry emission times, select 80 excursion centers in the requested
interval, of which 64 centers and 120 unique neighboring frames have their
complete camera contexts inside the retained kernel suffix.

Every telemetry batch reaches `stageTraceMax=16`. Between first/last batch
counters, the off file detects 624 new excursions and drops 176 contexts;
the on file detects 628 and drops 180. These are bounded diagnostic selection
losses, not camera frame losses. Repeated neighbors deduplicate without
conflicting values; all 928 records have five present neighbors and complete
termination. One off-file record has an unqualified bracket; all on-file
records qualify.

Ordinary updates occur about every **2.056 seconds**, while summary retention
is two seconds. The resulting approximately **56 ms gaps are systematic**:
excluding each file's first batch, 447/448 off-file and 440/448 on-file centers
occur within 20 ms after the preceding emission. Consequently the summary
windows omit most of these short disturbance bursts, and the chronological
records preferentially retain their first 16 excursions. Neither population
provides an unbiased, continuous count of all excursions.

### EDT interrupts explain the recurring stage extensions

All 13381 recorded acquisition-thread switch-outs are voluntary sleeps; no
other task displaces a runnable camera thread in the retained trace. From
13380 matched wakes, wake-to-schedule-in latency has median 1 us, p99 2 us
and maximum 3 us at the text trace's resolution. Local timer/RCU events are
short (maxima 4/2 us).

In contrast, **127 EDT IRQ handlers interrupt acquisition execution**, lasting
32–37 us, mean **35.874 us**. Of these, **122 wake only main/control TID
2874366**, five wake no task, and none wake acquisition TID 2874370. The
remaining ordinary acquisition completions mostly occur in idle context.
Across all retained EDT handlers, 13348 wake acquisition only (mean 30.203 us),
528 wake main only (36.460 us), 32 wake both (50.937 us), and 35 wake neither.

Among the 120 selected neighboring frames with full kernel coverage, every
large stage extension in the following cohorts contains an EDT IRQ:

| Camera stage | Long-stage criterion | Long stages | Containing EDT IRQ |
| --- | --- | ---: | ---: |
| Conversion/load | At least 50 us | 39 | 39 |
| Main notification | At least 20 us | 31 | 31 |
| SDK rearm | At least 20 us | 5 | 5 |
| Validation | At least 20 us | 1 | 1 |
| Sync notification | At least 20 us | 1 | 1 |

One additional EDT interruption overlaps a wide write-timestamp bracket,
bringing the matched post-return stage interruptions to 78. Of these, 76
wake the main/control thread. This directly identifies an interruption during
the stage as a source of its elapsed-time extension. It does not establish
that every excursion outside the selected, covered frames has the same cause.

For example, sequence **2293237** has a **539.208 us** publication interval
and **65.093 us** conversion. IRQ 405 runs inside that conversion at monotonic
**1746811.550014–1746811.550050**, waking the main/control thread. The following
sequence **2293238** has a **471.320 us** publication interval and normal
**26.821 us** conversion. A separate 37 us EDT interrupt then extends its
post-write main notification to **42.530 us**. This separates the cause of the
late write from work after the following write's timestamp.

### Serial housekeeping is the strongly supported source of these IRQ bursts

The main-thread wakes form 125-interrupt bursts over approximately 10.95 ms,
repeating every 2.054–2.056 seconds. Shorter 20-interrupt bursts lasting
1.708 ms follow approximately 0.529 seconds later. Inter-interrupt spacing is
consistent with serial byte reception at the configured 115200 baud. All
retained excursion centers overlap a completed serial transaction, though
the saturated selection prevents using that association alone as proof.

Local source independently connects the paths:

- `ocam2KCtrl::appLogic()` calls `getTemps()` and `getEMGain()` on its ordinary
  thread. Synchronized `getFPS()` does not issue a serial query.
- `edtCamera::pdvConfig()` enables serial reception on the same `m_pdv` device
  used by acquisition. `pdvSerialWriteRead()` waits for serial data using
  `pdv_serial_wait(..., 1)`.
- Copied PDV 5.6.8.0 `libpdv.c`, at `SERIAL_ENABLED_FLAGS` and
  `pdv_serial_read_enable()`, enables receive/device interrupts and the remote
  interrupt path. `pdv_serial_wait()` uses the `EDTS_SERIALWAIT` ioctl;
  `libedt.c::edt_set_remote_intr()` enables remote-interface/PCI interrupts.

Together with the captured control-only wakes, this strongly identifies
serial-response handling on the shared EDT interrupt as the source of the
bursts. The current SDK copy does not contain the running kernel ISR source,
and the trace does not record a serial-specific hardware status bit. The
exact driver operations consuming those approximately 36 us remain unverified.

Serial transaction envelopes average approximately **525.7 ms**, mostly
blocking wall time. The local active mode file has no `serial_waitchar`, and
the SDK disables wait-character completion when it is unspecified. The
controller's 1000 ms read timeout gives the response loop a 500 ms trailing
wait, followed by its 10 ms drain. Two transactions plus the ordinary
one-second application sleep explain the 2.056-second update cadence.
Shortening completion waits alone would not establish fewer serial RX
interrupts or shorter handlers. A response-terminator change would require
separate validation of full OCAM response and prompt framing.

### Statistics, ordering and next comparison

The two telemetry files each retain 115971 qualified intervals. Their pooled
SDK acquisition deviations are 0.431 us off and 0.398 us on, versus publication
deviations 1.864/1.967 us. These summary values omit the systematic gaps above.
Only ten seconds of the on file have kernel tracing enabled, so whole-file
off/on differences do not isolate tracing overhead. Four two-second summaries
are wholly inside the requested trace interval; comparison with 24 wholly
outside windows in the same file finds mean main-notification duration
4.484 versus 3.679 us, and sync notification 0.884 versus 0.806 us. This is
consistent with measurable tracepoint cost in the notification path; it is
far smaller than the approximately 36 us interrupt extensions and does not
quantify the untraced ISR duration.

Late-then-early ordering is not universal under the +/-20 us classification.
Among 560 qualified retained early centers, 217 follow late intervals, 233
follow normal intervals and 110 follow early intervals. Of 367 late centers,
235 are followed by early intervals, 131 by normal intervals and one by late.
These are selected-sample counts. For example, sequence 2285022 has successive
publication intervals 511.356, **471.971**, 517.497 and 506.417 us: its early
interval follows a normal classified interval. Do not turn the earlier
late-then-early hypothesis into a scheduler invariant.

The next useful operator-controlled A/B comparison is to separate EDT IRQ
execution from the acquisition CPU while preserving the camera worker's
CPU 33/FIFO 90 and the serial polling workload. Select an actually spare CPU
after checking instrument placement, and capture both CPUs. This tests the
specific identified interference while exposing any new cross-CPU completion
latency. It is a proposal, not a configuration change performed here.

For a subsequent capture, five seconds fits CPU 33's observed event rate
in the helper's 8 MiB per-CPU ring; check overwrite counters again, including
any newly selected CPU with different traffic. Increasing
`stageTraceMax` provisionally to 64 and checking that its drop counter stays
unchanged would reduce the first-16 selection loss. This requires the normal
operator restart. Separately, continuous interval statistics or summary
retention covering the actual consumer cadence should be considered so that
routine serial bursts cannot repeatedly fall between summary windows.
Reducing housekeeping frequency would need to preserve temperature and EM
protection monitoring; no polling or serial-framing changes are made here.

Independent local parsers checked clock matching, exact trigger/neighbor
interval identities, deduplication, IRQ entry/exit and wake associations,
stage containment, counter deltas and pooled statistics. Working artifacts
are `/tmp/analyze_fg_capture.py`, `/tmp/review_ocam_kernel.py`,
`/tmp/parsed_fg_capture.json` and `/tmp/ocam-all-kernel-correlations.csv`.
Only this engineering record and the syncopation debugging record change.
No C++ changes, build/test reruns, commit, push or instrument actions accompany
the analysis.

## 2026-10-24 second capture with EDT IRQ on CPU 61

The operator supplied
`/home/jrmales/tmp/camwfs-trace-20261024T222140Z`. Before/after metadata
confirms **IRQ 405 on CPU 61**, with acquisition TID 2874370 still **FIFO 90
on CPU 33**. Neither CPU has an SMT sibling. PID 2874366, session 5, inode
3333, mode 3 and software bin 1x1 are unchanged. The diagnostics still use
two-second windows, a 20 us excursion threshold, a 1 us bracket limit and
`stageTraceMax=16`; the helper captures ten seconds on both CPUs. These are
observed settings, rather than an assumption that all earlier proposed capture
changes were applied. The operator subsequently confirmed that moving IRQ 405
from CPU 33 to CPU 61 was the only change; camera settings, other CPU placement,
loop settings and workload were unchanged. This establishes the intended IRQ
placement A/B comparison, subject to the sampling limits below.

Input SHA-256 values, relative to the supplied directory:

| File | SHA-256 |
| --- | --- |
| `camwfs-trace-off.txt` | `8b44b57e4b715f5586bd0f4947481da03095b1533211b5c8e9554d94ff5ef1e7` |
| `camwfs-trace-on.txt` | `04febab448f86314c20e0c59bf4bcdf2bbd3e9111525cc786e4c55215b9f08d0` |
| `camwfs-kernel-20261024T222140Z/trace.txt` | `a00e239395dca12a4242553732c65360d43b589133ff08472f92e72eb3e38867` |
| `camwfs-kernel-20261024T222140Z/metadata.json` | `644faaae00e91ead23c5a9d0fb4d2f54a455dc46ff487d45ef48d9880beded5a` |

### Coverage and quality

The requested kernel interval is approximately
**22:21:40.802146–22:21:50.802284 UTC**. CPU 61 retains 111140 events over
9.999537 seconds without overwrite. CPU 33 overwrites 50936 events and retains
200058 events over **7.969006 seconds**, monotonic
**1748341.423232–1748349.392238**. Joint camera/IRQ coverage ends at CPU 61's
last event, 1748349.392196, with one-microsecond text resolution.
All 20611 EDT IRQ handlers run
on CPU 61 in idle context; none run on the acquisition CPU. The camera's
nonvoluntary context-switch count is unchanged across the metadata snapshots.

Both telemetry files contain 29 disjoint two-second stage windows and
115971 qualified intervals, with no ambiguous intervals, new bracket-wide
observations, publication failures, rejected frames, frame/counter gaps or
source queue losses. There are 416 off-file and 403 on-file excursion records,
all qualified, complete and consistent with their neighbor interval identities.
The on file begins with only one record from an older 14-record batch; that
file boundary is not a newly reported diagnostic drop.

Between first/last counters, diagnostic selection drops are **4 of 407** new
detections off and **4 of 406** on, approximately 1% versus approximately 28%
in the first capture. Selection remains bounded, and the approximately 56 ms
gaps between stage-summary windows persist. The requested kernel interval
contains 61 centers and 136 unique neighboring frames; full joint retained
coverage contains **44 centers and 98 unique neighboring frames**. Counts and
stage ranges below use the stated population, rather than mixing these sets.

### Publication jitter improves and recorded stage interruptions disappear

The off files provide a comparison with kernel tracing absent in both runs;
stage/neighborhood telemetry remains enabled in both. Pooled statistics use
the 29 disjoint retained windows per file:

| Metric | IRQ on CPU 33, first off file | IRQ on CPU 61, second off file |
| --- | ---: | ---: |
| SDK acquisition-interval deviation | 0.431 us | 0.431 us |
| SDK-return-interval deviation | 1.039 us | 0.467 us |
| Publication-interval deviation | 1.864 us | **0.995 us** |
| Mean acquisition-to-write latency | 52.744 us | 53.600 us |
| Acquisition-to-write deviation | 1.440 us | 0.824 us |
| Conversion maximum in retained summaries | 84.760 us | 30.036 us |
| Main-notification maximum in retained summaries | 41.308 us | 8.255 us |
| Mean rearm duration | 4.765 us | 5.192 us |

Thus publication RMS decreases about **47%**, while mean acquisition-to-write
latency increases approximately **0.86 us**. This is an improvement in timing
spread with a small mean-latency tradeoff. Summary gaps still prevent claiming
continuous full-population jitter. The direct chronological evidence agrees:
the retained neighborhoods no longer have the approximately 37 us additions
in conversion, validation or main/sync notification. In the 98 frames with
full joint kernel coverage, conversion is 24.927–27.332 us and main
notification is 4.188–5.300 us.

Cross-CPU wake latency remains small. Using `sched_waking` to camera
schedule-in consistently in both traces, the first capture has mean 1.474 us,
median 1 us, p99 2 us and maximum 3 us; the new capture has mean 1.661 us,
median 2 us, p99 2 us and maximum 3 us. Do not mix these with shorter
`sched_wakeup`-to-run intervals. These observations do not show a large new
scheduler delay after moving the IRQ.

Only ten seconds of the on file have kernel tracing enabled. Its whole-file
publication deviation is 1.010 us, versus 1.967 us in the first on file.
Three new two-second summaries fit wholly inside the requested kernel
interval, 24 wholly outside, and two cross its boundaries. Inside/outside mean
main-notification duration is 4.481/3.673 us, repeating the approximately
0.8 us tracing cost observed in the first capture. Whole-file off/on values
therefore remain unsuitable for isolating kernel instrumentation overhead.

### Remaining rearm delays track the remote EDT handler

Rearm is now the recurring long post-return segment. Among the 98 frames
with full joint coverage, 72 ordinary rearm calls have median **5.300 us**
(5.159–6.031 us); **26 calls** take **31.219–39.645 us**, mean 38.803 us.
Every long call overlaps IRQ 405 on CPU 61: 25 handlers wake only the ordinary
control thread, and one wakes no task. None of these 26 rearm intervals
contains a CPU 33 schedule switch or local IRQ/vector entry.

The remote handler begins 1.931–3.767 us before the SDK wait-return marker.
Rearm finishes 4.987–5.977 us after handler exit, mean **5.462 us**. The extra
rearm time above the 5.300 us ordinary median averages **33.503 us**, while
overlap with the remaining remote IRQ averages **33.341 us**. The close
agreement strongly supports serialization with driver/device work on CPU 61.

For sequence **5356674**, the monotonic endpoints are:

| Event | Monotonic seconds |
| --- | ---: |
| EDT IRQ entry on CPU 61 | 1748343.269693000 |
| SDK wait returns on CPU 33 | 1748343.269694989 |
| EDT IRQ exit on CPU 61 | 1748343.269729000 |
| SDK rearm returns on CPU 33 | 1748343.269734283 |

The rearm interval is **39.294 us**: **34.011 us** overlap with the remote
handler, followed by **5.283 us**, essentially the ordinary rearm duration.
The same call remains running on CPU 33 throughout the measured interval.
This establishes the timing relation; it does not identify a particular
driver lock or distinguish spinning from slow shared-device access.

Local SDK source narrows the call path:

```text
edtCamera::pdvAcquire()
  pdv_start_image(m_pdv)
    pdv_start_images(m_pdv, 1)
      edt_start_buffers(m_pdv, 1)
        edt_ioctl(m_pdv, EDTS_STARTBUF, &count)
          ioctl(fd, EDT_NT_IOCTL, &request)
```

The ordinary initialized path has no visible application or user-library
mutex. `EDTS_STARTBUF` is action 39 in the copied SDK. Historical EDT wrappers
use `spin_lock_irqsave()` for driver critical sections, but the running
kernel core is unavailable locally. The copied 5.6.8 SDK does not by itself
establish the running kernel module's version. A current shared driver lock is
therefore a well-supported hypothesis, not a source-verified mechanism.

### Ordering and next measurement

The new retained sample is more often a simple late/early pair: 325 of 348
late centers are followed by early, while 23 are followed by normal. Of 471
early centers, 323 follow late and 148 follow normal. Universal classified
late-then-early ordering is still not established at the 20 us threshold.

The next focused measurement is to bracket acquisition-thread ioctl entry
and exit, correlated with existing rearm markers and EDT interrupts on both
CPUs. This would establish whether the approximately 34 us excess is inside
the kernel call. If so, inspect the running driver's available symbols/source
and selectively measure its ioctl and critical-section acquisition; remote
IRQ overlap alone cannot prove a specific lock. Preserve rearm/DMA semantics
until that mechanism is understood. A five-second kernel capture and a
larger bounded neighborhood capacity remain useful coverage improvements,
with loss counters checked rather than assumed zero.

Independent local parsers checked per-CPU coverage, IRQ entry/exit and wake
associations, exact neighborhood identities, event/diagnostic counter deltas,
stage pooling and all 26 long rearm correlations. Artifacts include
`/tmp/parsed_fg_capture_222140.json`, `/tmp/review_ocam_kernel_222140.py` and
`/tmp/ocam-rearm-irq-correlations-222140.csv`. Only the OCAM engineering plan
and syncopation debugging record are updated. No C++ changes, build/test
reruns, commit, push or agent instrument actions accompany this analysis.

## 2026-10-24 defer OCAM rearm until after publication

The operator approved moving queue replenishment out of the completed frame's
publication path and separately measuring the timeout-counter query. The
implementation uses the existing `ocam2KCtrl::frameGrabberPostPublish()` hook,
as the operator suggested: it posts the sync stream after the main stream,
then replenishes the consumed EDT request on the same acquisition worker.
No new framegrabber completion hook is introduced.

### Acquisition and recovery contract

Startup still queues `m_numBuffs` acquisitions. The SDK documents that this
count releases requests to the driver; it is more than allocating ring memory.
With the current 64-buffer pipeline, further captures remain queued while one
completed image is validated, descrambled and published. Deferral is eligible
only when both the requested and actual allocated buffer counts exceed one,
`force_single` is clear, and no serial acquisition trigger is configured.
Eligibility is cached at acquisition startup. Other EDT controllers opt out
by default, and ineligible OCAM modes preserve immediate rearm.

`edtCamera::pdvAcquire(timestamp, deferRearm)` retains latest-image selection,
the original SDK acquisition timestamp, timeout/partial-transfer checks and
timestamp validation. Eligible OCAM acquisition records one pending queue
obligation instead of starting another request before those checks.
`pdvFinishAcquisition(continuing)` clears that obligation once and either
replenishes it or discards it before recovery/exit. A second acquire cannot
silently consume another frame while an earlier obligation is unfinished.
Configuration/start reset this state. Acquisition errors, invalid frame
numbers and recovery requests discard it; post-hook failures, shutdown,
reconfiguration or power-off do not queue a later unrelated request.

The normal ordering is now:

```text
SDK wait -> timeout/partial check -> OCAM frame validation -> descramble
  -> main image publication -> sync publication -> EDT rearm -> next wait
```

The timeout query remains before publication so an SDK partial transfer is
never published as valid. It is another driver ioctl and may still encounter
the same contention. Moving rearm is not a claim that all publication jitter
has been eliminated. Hardware comparison should examine both query duration
and output timing with the IRQ on CPU 61 and other settings unchanged.

### Timing and compatibility

Existing marker IDs 0–8 remain stable. Appended endpoints are
`rearmEnter` (9), `timeoutEnter` (10) and `timeoutReturn` (11).
`timeoutQuery` is appended as metric 16. The producer checks actual capture
chronology independently of numeric marker IDs, and the reader validates
both immediate and deferred semantic orders. Retention uses the greatest
valid endpoint rather than the last numeric marker slot.

`rearm` measures its explicit entry-to-return span. `timeoutQuery` measures
the query independently, and `validation` starts after query return. Existing
controllers/events without the explicit endpoints retain legacy definitions.
`syncPostDone` remains the full publication-hook return, after deferred rearm;
the `syncPost` duration ends at deferred `rearmEnter` so notification and rearm
populations remain separate. `loopTail` starts at hook return. `waitToWrite`
retains its SDK-return-to-publication meaning, excluding deferred rearm.
The existing per-metric INDI and matched-change loops expose `timeoutQuery`
without adding another property or configuration setting.

New stage/trace writers emit 17 metric populations and 12 endpoint slots.
Updated binary, text, JSON and metadata readers also accept existing 16-metric
and nine-marker records, with appended observations unavailable. This preserves
the captures used in both IRQ comparisons. Rebuild/install the camera app and
`logdump` for the extended records. The operator capture procedure uses five
seconds to reduce ring overwrite at the observed event rate; loss counters
and two-CPU coverage must still be checked.

Independent review found two failure-path details before completion: a failed
new entry clock must not widen an SDK duration through the legacy fallback,
and a reported missing deferred entry must not censor the otherwise usable
raw trace. Affected durations remain unavailable, while explicitly unqualified
new records retain their raw endpoints and invalid counts. Legacy nine-marker
records keep their original ordering checks.

### Workstation verification

Eight suites pass **4642 assertions in 86 cases**, relinked against the final
library build:

| Suite | Assertions | Cases |
| --- | ---: | ---: |
| `frameGrabberTiming_test` | 323 | 20 |
| `ocam2KCtrl_test` | 713 | 33 |
| `frameGrabberTelemetry_test` | 2415 | 6 |
| `frameGrabberTraceTelemetry_test` | 648 | 7 |
| `frameGrabber_integration_test` | 118 | 7 |
| `ocam2KCtrl_lifecycle_test` | 7 | 1 |
| `cred2Ctrl_test` | 405 | 11 |
| `cred2Ctrl_lifecycle_test` | 13 | 1 |

Coverage includes deferred ordering, separate timeout/rearm durations,
ordinary/rejected/error attempts, requested/allocated ring eligibility,
single-frame and serial-trigger fallbacks, exact-once replenishment,
current-versus-next publication effects, failed optional clocks and old/new
binary, text, JSON and metadata compatibility. Actual-loop tests establish
that deferred rearm does not enter the current publication timestamp or
notification duration, while timeout-query delay remains before publication.
Existing CRED2 behavior and both controller lifecycles remain covered.

The library/schema build, normal `logdump` build and OCAM compile/link against
the copied PDV **5.6.8.0** headers/library pass. No compiler warnings or errors
are reported beyond existing git-state pragma messages. Full changed-C++
clang-format checks, whitespace checks and relative documentation-link checks
pass. Build/test logs are `/tmp/ocam-deferred-*` and
`/tmp/ocam-rearm-frameGrabber*`; test-created binary logs are retained under
`/tmp/ocam-deferred-test-binlogs` rather than in the working tree. No camera
binary is launched or instrument setting changed by the agent. The next step
is an operator rebuild and timing comparison with IRQ 405 kept on CPU 61;
existing configuration requires no new setting.

### 2026-10-24 commit/push checkpoint

The operator approved committing and pushing the verified changes to the
existing `jrmales/dm-syncopation` feature branch at `origin` (`magao-x/MagAOX`).
The functional commit includes the associated IRQ findings and engineering
records, followed by a documentation-only commit for the timing guide and
operator procedure. No standalone formatting commit is needed. Hardware
qualification remains operator work.

## Affected files

- `.clang-format`
- `agents/plans/2026-10/dm-syncopation-debugging.md`
- `agents/plans/2026-10/dm-syncopation.md`
- `agents/plans/2026-10/ocam2k-acquisition-publication-timing.md`
- `apps/cred2Ctrl/tests/cred2Ctrl_test.cpp`
- `apps/dmSyncopation/doc/camera-acquWFS-test.md`
- `apps/ocam2KCtrl/doc/camera-timing-trace.md`
- `apps/ocam2KCtrl/ocam2KCtrl.hpp`
- `apps/ocam2KCtrl/scripts/capture-camera-timing.py`
- `apps/ocam2KCtrl/scripts/tests/capture_camera_timing_test.py`
- `apps/ocam2KCtrl/tests/ocam2KCtrl_test.cpp`
- `libMagAOX/Makefile`
- `libMagAOX/app/MagAOXApp.hpp`
- `libMagAOX/app/dev/dmTiming.hpp`
- `libMagAOX/app/dev/doc/frameGrabberTiming.md`
- `libMagAOX/app/dev/edtCamera.hpp`
- `libMagAOX/app/dev/frameGrabber.hpp`
- `libMagAOX/app/dev/frameGrabberTiming.hpp`
- `libMagAOX/app/dev/tests/frameGrabberTiming_test.cpp`
- `libMagAOX/app/dev/tests/frameGrabber_integration_test.cpp`
- `libMagAOX/app/tests/MagAOXApp_threadStart_test.cpp`
- `libMagAOX/common/dmTiming.hpp`
- `libMagAOX/common/frameGrabberTiming.hpp`
- `libMagAOX/common/spscQueue.hpp`
- `libMagAOX/common/tests/spscQueue_test.cpp`
- `libMagAOX/common/timingStats.hpp`
- `libMagAOX/logger/logCodes.dat`
- `libMagAOX/logger/logMeta.cpp`
- `libMagAOX/logger/tests/frameGrabberTelemetry_test.cpp`
- `libMagAOX/logger/tests/frameGrabberTraceTelemetry_test.cpp`
- `libMagAOX/logger/types/schemas/telem_fgstage.fbs`
- `libMagAOX/logger/types/schemas/telem_fgtrace.fbs`
- `libMagAOX/logger/types/telem_fgstage.hpp`
- `libMagAOX/logger/types/telem_fgtrace.hpp`
- `tests/Makefile.one`
- `tests/edtinc.h`
- `tests/groups.dox`
- `tests/tests.list`
