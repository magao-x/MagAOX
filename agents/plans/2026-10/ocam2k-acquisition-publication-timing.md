# ocam2KCtrl acquisition and publication timing review

Date: 2026-10-23 (America/Phoenix).
Status: approved implementation and workstation verification complete, including
compilation/linkage against copied PDV 5.6.8.0. Operator overhead/bracket,
loaded-driver and DMA-owner-thread qualification remain pending. No instrument
changes have been performed. The operator approved committing and pushing the
verified implementation.

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
The new application result guard adds one more query. In the current wrapper
it occurs after `rearmReturn`, so its time falls in the rearm-to-validation
segment.

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

## Affected files

- `.clang-format`
- `agents/plans/2026-10/dm-syncopation-debugging.md`
- `agents/plans/2026-10/dm-syncopation.md`
- `agents/plans/2026-10/ocam2k-acquisition-publication-timing.md`
- `apps/cred2Ctrl/tests/cred2Ctrl_test.cpp`
- `apps/dmSyncopation/doc/camera-acquWFS-test.md`
- `apps/ocam2KCtrl/ocam2KCtrl.hpp`
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
- `libMagAOX/logger/types/schemas/telem_fgstage.fbs`
- `libMagAOX/logger/types/telem_fgstage.hpp`
- `tests/Makefile.one`
- `tests/edtinc.h`
- `tests/groups.dox`
- `tests/tests.list`
