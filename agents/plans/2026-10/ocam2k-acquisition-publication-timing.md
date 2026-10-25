# ocam2KCtrl acquisition and publication timing review

Created: 2026-10-23; updated 2026-10-25 (America/Phoenix).
Status: the first normal / paused / normal housekeeping comparison is complete
with the AO loop closed and all settings/workload unchanged, as confirmed by
the operator. Serial attempts remain at 981 through the paused capture, then
resume. Retained driver `atime` intervals narrow from 481–525 us before and
480–515 us afterward to **499–501 us** while paused. The normal files' qualified
raw >20 us excursions disappear from the paused file. Together with the prior
shared-lock/handler traces, this strongly supports serial/device handling as a
source of the recurring large completion-stamp excursions.

Publication timing has another source of variation. A late change during the
paused capture raises conversion spread while SDK-return timing stays steady;
pooled publication deviation is **0.761 / 1.428 / 0.747 us** before/paused/after.
Thus the pause removes the large stamp tails without improving overall RMS in
this particular run. The late paused conversion change remains unexplained.
The paused telemetry updates twice as often, so comparisons use disjoint
summary windows rather than pooling overlapping samples.

The accepted wrapper capture previously established the same lock in the frame
thread and IRQ 405, 34 us acquisition brackets overlapping remote holds, and
roughly 20 us of extra handling when device/control and frame completion share
an invocation. This identifies delayed software completion processing and
wakeup; physical interrupt assertion and exposure timing remain unmeasured.
The 120-second diagnostic pause is implemented and operator-qualified in this
capture, with automatic expiry serviced about 0.426 s after its deadline.
Production housekeeping scheduling and syncopation scheduling are unchanged.
Agents perform no instrument actions. Capture evidence, limits and implementation
history follow below.

The first two paired chronological and kernel captures identify EDT interrupts
as the source of recurring stage extensions. The operator moved IRQ 405 from
acquisition CPU 33 to CPU 61 as the only experimental change. Retained publication-interval
deviation decreased from 1.864 to 0.995 us, and large conversion/notification
extensions disappeared from the selected neighborhoods. Remaining long rearm
calls track the remote handler, strongly supporting driver/device serialization
without identifying a specific lock. These captures establish no scheduler
safety margin or continuous full-population timing bound.

The first deferred-rearm capture reduces mean acquisition-to-write time from 53.600 to
50.167 us in the retained kernel-off summaries, and selected excursion contexts
fall from 416 to 19. This is not a uniform jitter improvement: pooled publication
deviation in that file rises from 0.995 to 2.224 us during an unexplained change
in conversion/publication spread. Most windows in the second file are near
0.8 us, but an uncovered burst reaches 799.700 and 924.626 us. The operator
confirmed no other settings or workload changes. The source review, earlier
implementations and each operator capture remain below as engineering history.
The subsequent capture separates metadata/setup/settling from the enabled
interval. Its large setup burst is absent from the retained steady trace;
remaining covered 525/479 us excursions already occur at SDK return. This
does not prove unchanged frame age after the setup burst or eliminate the
previous capture's unexplained spread.
The following five-second repeat resolves the ring overwrite and supplies
complete covered examples of the remaining rearm serialization. Setup does
not cause a large recorded excursion on every invocation.

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

## 2026-10-24 operator capture after deferred rearm

The operator supplied
`/home/jrmales/tmp/camwfs-trace-20261025T034126Z` after rebuilding/restarting
the camera and reader. The directory timestamp is October 25 UTC, corresponding
to the evening of **October 24 in America/Phoenix**. The operator confirmed
that no other camera, loop, workload or CPU settings changed, including during
the first file's changing timing spread. All analysis uses local copied files.

### Provenance, populations and coverage

The restarted camera is PID **3077681**, acquisition TID **3077683**, FIFO **90**
on CPU **33**. IRQ **405** remains on CPU **61** in both metadata snapshots;
neither CPU has an SMT sibling. Boot identity remains
`b63fa319-8340-4848-8f36-dfadd059a064`. The stream has inode **3335**, timing
session **5**, camera mode **3**, software binning **1x1**, synchronized camera
`bin2` at **2000 FPS**. The new process/inode distinguish this session from
the earlier session with the same numeric session value. Twelve-endpoint traces
with validity mask **4095** establish the new timing format; this capture does
not include a runtime git-hash readback.

Input checksums (SHA-256):

| File relative to the capture directory | SHA-256 |
| --- | --- |
| `camwfs-trace-off.txt` | `f8e6c3da7669e5255e39cb217a910b55fee716fc5cc9891e685c38d9f3b2a804` |
| `camwfs-trace-on.txt` | `b4759ff6fa8009363f06d09de6f9a981e6f41bc30e1a207cd241ecb83dd7c042` |
| `camwfs-kernel-20261025T034126Z/trace.txt` | `a1b5ec612cde6683083a093fceb98b0a463a3799816686de7738917e7ba6349a` |
| `camwfs-kernel-20261025T034126Z/metadata.json` | `a4d635d3617525b5058c67e0011619a9369837390c2bff5d1a0438b19e2b6e52` |

Here `off` and `on` describe the kernel trace, not camera or AO-loop enablement.
Each telemetry file contains **29 nonoverlapping two-second summaries**. The
first has **115971 raw intervals**, of which **115835** are qualified and
**136** are ambiguous; the second has **115970**, all qualified. Seventy new
wide write brackets occur in three first-file windows, with maximum width
**4.038 us**, above the configured **1 us** qualification limit. Pooled raw
interval statistics below retain those measurements; the ambiguity counts
must accompany interpretation of the first file. No new frame/counter gaps,
rejections, acquisition/publication failures, invalid-marker events or
source/control queue losses are reported.

The bounded chronological diagnostic retains **19 off-file** and **9 on-file**
complete qualified excursion contexts, compared with **416/403** in the previous
IRQ-on-61 capture. No new diagnostic selection drops occur; cumulative
`dropped=2891` predates these excerpts. All five neighbors are present in each
context, with no conflicting repeated frames. The maximum ready batch is three,
well below capacity sixteen. Selection counts cover similar one-minute files
but do not establish a continuous full-population timing bound. Summary updates
still leave approximately **56 ms gaps** between the retained two-second
windows; the chronological diagnostic can capture events in those gaps.

The helper requests ten seconds, **03:41:27.476171–03:41:37.476315 UTC**.
CPU 61 retains **9.999538 s**, with no overwritten events. CPU 33 overwrites
**50350 events** and retains **7.983017 s**. Exact joint retained coverage is
monotonic **1767528.083240–1767536.066212**, or **7.982972 s**. Kernel timestamps
have microsecond printed resolution. All four selected contexts inside the
requested ten seconds survive in that joint suffix, comprising **20 unique
neighboring frames**. The other five on-file centers are outside the requested
window: three before it and two after it. Thus their absence from correlation
is not a diagnostic drop or loss of a selected context to ring overwrite.

### The intended ordering is observed

In the twenty jointly covered frames, the independent timeout query lasts
**0.230–0.481 us**. Five queries overlap remote control-waking EDT handlers
and still finish in **0.240–0.381 us**. The retained summaries' query maxima
are **4.278 us off** and **0.772 us on**. The earlier tens-of-microseconds
rearm wait has not appeared in this guard in these samples.

Six deferred rearms last **23.113–24.306 us**, averaging **23.601 us**. Every
one overlaps a **36 us** EDT handler on CPU 61 that wakes the camera's main
control thread, with no local CPU 33 interrupt/vector entry or thread switch
during the call. Remaining remote-handler overlap averages **18.432 us**;
rearm completes on average **5.169 us after IRQ exit**, close to its ordinary
duration. This continues to support driver/device serialization without
identifying a particular lock.

These calls start **5.180–5.821 us after `writeStampDone`** and
**0.811–0.961 us after `mainPostDone`**. Sync notification ends before
`rearmEnter` under the implemented endpoint contract. Consequently the recorded
rearm stall follows the current image timestamp and both notifications.
It no longer extends that frame's publication path, though it still occupies
the acquisition worker before the next wait.

For example, sequence **675844** has a **0.241 us** timeout query. Its rearm
bracket is **1767529.410251508–1767529.410275814**, overlapping the remote
IRQ at **1767529.410234–1767529.410270**. The **24.306 us** call starts
**5.681 us** after the write endpoint and **0.932 us** after the main-post
endpoint, then returns **5.814 us** after IRQ exit. The absolute endpoints
above are monotonic seconds.

Conversion in these twenty frames is **26.199–27.703 us**, main notification
**4.369–5.199 us**, and sync notification excluding rearm **0.811–1.092 us**.
The four selected publication intervals are **524.611, 523.910, 479.125 and
479.305 us**. Their SDK-return intervals are already **525.472, 523.539,
479.325 and 479.375 us**. Their deviations therefore enter before the
post-return query/conversion/publication work. Both early cases follow combined
control/acquisition IRQs of about 50 us with acquisition-only handlers of
30–31 us. The late cases also have combined handlers, but their preceding
handlers are combined too and IRQ-entry spacing is **526 us**: changing handler
duration alone does not explain every late interval.

All **20709** EDT IRQs run on CPU 61, predominantly waking only acquisition.
Camera switchouts are voluntary sleeps. Acquisition waking-to-run latency is
mean **1.585 us**, median **2 us**, p99 **2 us**, maximum **3 us** in retained
kernel coverage. No local EDT IRQ or scheduler displacement explains the
covered selected post-return stages.

### Latency improves; jitter changes within the capture

The following pooled values use disjoint summaries and population counts, not
averages of the displayed standard deviations. Units are microseconds.

| Kernel-off summaries | Previous IRQ-on-61 capture | New deferred-rearm capture |
| --- | ---: | ---: |
| Mean SDK acquisition to write | 53.600 | 50.167 |
| Mean SDK return to write | 31.710 | 28.287 |
| Mean conversion | 26.120 | 27.782 |
| Publication-interval standard deviation | 0.995 | 2.224 |
| SDK-acquisition-interval standard deviation | 0.431 | 0.382 |
| Mean timeout query | not separately measured | 0.246 |

Mean acquisition-to-write latency improves **3.434 us**, and return-to-write
improves **3.423 us**, despite conversion being **1.662 us** slower on average.
This is consistent with removing the ordinary roughly 5.2 us rearm cost from
the critical path. It does not establish a uniform jitter improvement.

The first six off-file windows have publication deviation **0.748–0.861 us**.
It rises through **0.915, 1.542 and 2.617 us**, remains approximately
**2.566–2.842 us**, then falls to **0.930 us** by the final window. Conversion
mean rises from about **27.0 to 28.3 us** during the wider-spread period; near
its end, conversion returns to about 27 us while SDK-return spread briefly
increases. The operator confirmed no other changes. There is no simultaneous
kernel trace for this period, so its cause remains unresolved; CPU frequency,
memory behavior or an unmeasured workload must not be assigned as explanations
from these records alone. Seventy wide write brackets also occur near the
start of this transition, as qualified above.

The on-file's other **28 windows** have publication deviation
**0.757–0.879 us**. One burst-containing window reaches **9.072 us**, making
the whole-file pooled value **1.864 us**. Partitioning complete summaries by
the requested kernel interval gives:

| New on-file population | Windows | Intervals | Publication deviation (us) | Mean acquisition to write (us) |
| --- | ---: | ---: | ---: | ---: |
| Wholly inside requested kernel interval | 4 | 15996 | 0.799 | 49.855 |
| Wholly outside requested kernel interval | 23 | 91977 | 0.815 | 49.442 |
| Crossing either boundary | 2 | 7997 | 6.438 | 49.542 |

The previous on-file's wholly outside summaries had deviation **1.015 us**
and mean acquisition-to-write **53.635 us**. This restricted comparison
improves both values, but excludes the crossing windows and must not replace
the whole-file result. New mean main-notification duration is **4.623 us**
inside versus **3.831 us** outside, a **0.791 us** difference consistent with
the trace instrumentation cost observed previously. Serial transaction
envelopes remain about **525.5 ms** of wall time; that is not CPU busy time.

### Larger uncovered events and next decision

The off-file chronological diagnostic preserves sequences **499040/499041**,
published at **03:40:02.417660337/03:40:02.418045855 UTC**, with intervals
**614.931/385.518 us**. The SDK-return intervals are already
**613.930/385.157 us**, versus SDK acquisition intervals **510/486 us**.
Timeout queries are **0.270/0.231 us**, conversion **26.720/27.171 us**;
the first frame's **27.351 us** rearm is after publication. This pair lies
in a summary-window gap and has no kernel coverage. Summary maxima alone
would miss it.

The on-file burst publishes sequences **668956–668958** at
**03:41:27.375907750, 03:41:27.376832376 and 03:41:27.377096154 UTC**.
Publication intervals are approximately **799.700/924.626/263.8 us**;
SDK-return intervals are already **799.139/925.839/275.410 us**. SDK acquisition
intervals are **517/1205/278 us**. Queries remain **0.230–0.251 us** and rearms
**5.400–5.490 us**. There are no frame-identity gaps. This is not the newly
deferred rearm stalling publication.

This burst occurs **99–100 ms before** the helper's `capture_begin` anchor,
after its before-metadata anchor at **03:41:27.199536 UTC**. The helper first
collects process/device metadata and writes JSON, then configures the trace
instance/events, records `capture_begin` and enables tracing. There are no
intermediate phase anchors or kernel events for the burst. Its placement
within that metadata/setup envelope is an association, not proof that tracing
setup caused it. The much earlier off-file pair also remains unattributed.

The current publication-path objective is supported by matched endpoint/kernel
evidence. Remaining larger delays are already present at SDK return and need
coverage there before assigning a cause. SDK acquisition timestamps identify
host DMA completion under the reviewed contract, not physical exposure;
counter continuity alone does not establish constant frame age. A possible
next controlled measurement would separate helper setup/settling from the timed
capture and use the documented five-second duration or enough ring storage
to avoid overwrite. No such helper, camera or driver change is made here.
No scheduler reserve or continuous worst-case bound is inferred.

Reproducible analysis scripts and detailed results are retained locally as
`/tmp/analyze_fg_capture_034126.py`, `/tmp/fg_capture_followup_034126.py`,
`/tmp/parsed_fg_capture_034126.json`, `/tmp/review_ocam_kernel_034126.py`,
`/tmp/ocam-kernel-analysis-034126.json`,
`/tmp/ocam-frame-kernel-correlations-034126.csv`,
`/tmp/ocam-deferred-rearm-correlations-034126.csv` and
`/tmp/ocam-centered-irq-correlations-034126.csv`. Earlier capture artifacts
are preserved. Only this engineering record and the synco debugging record
change; no code, configuration, commit, push or agent instrument action
accompanies this analysis.

## 2026-10-24 separate capture setup, settling and measurement

The operator approved preparing the next controlled measurement. The helper
now defaults to **five seconds of settling with tracing off**, followed by
**five seconds of capture**. This addresses both the uncovered burst within
the previous metadata/setup envelope and the ten-second camera-CPU ring
overwrite. It does not assign that burst's cause or establish that every setup
effect disappears after five seconds.

Before-metadata collection and private trace setup complete before settling.
The helper retains explicit monotonic/realtime phase anchors in memory for
metadata, setup, settling and both trace-control writes. The enable/disable
brackets distinguish uncertain switch-boundary ordering from the interior of
the enabled interval. The legacy capture anchors remain available for existing
analysis. Ordinary saving, metadata queries and console output occur outside
the interval between completed enable and the beginning of disable. Existing
private-instance ownership and stop/save/remove recovery behavior remain.
The `capture_started` flag confirms the enable write returned;
`capture_completed` requires the full requested capture wait and a confirmed
stop. These flags do not certify later save/cleanup success. Preparation
interrupted before enable has no legacy capture boundaries; a failed enable
attempt retains its bracket begin without a confirming end.

`--settle-seconds` defaults to **5**, accepts finite values **0–60**, and
does not consume ring history because tracing remains off. `--seconds` now
defaults to **5**, retaining the **1–30** range. Metadata exposes incomplete
or interrupted phases rather than labeling setup/settling as a completed
capture. Only the operator runs the helper on the instrument.

The [operator procedure](../../../apps/ocam2KCtrl/doc/camera-timing-trace.md)
keeps the sixty-second telemetry baseline, then starts overlapping telemetry
about ten seconds before invoking the helper. That second file retains
metadata/setup/settling, the five-second kernel interval and the post-capture
tail. The invocation explicitly requests `--settle-seconds 5 --seconds 5`
and includes both currently observed CPUs **33,61**; the operator must use the
current acquisition TID and adjust CPUs if their placement changes. Events
outside kernel coverage remain useful chronological evidence and must be kept.

This is a helper/procedure change; it needs no new camera/reader rebuild,
camera restart, configuration setting, affinity change or scheduling change.
Local verification uses fake tracefs controls and mocked waits only. Instrument
qualification of the new capture sequence remains pending; no agent instrument
action, commit or push accompanies this setup work.

Verification passes **17 offline Python tests** using ordinary temporary fake
tracefs controls and mocked waits. Coverage includes phase ordering, tracing-off
settling, interruption during metadata/settling/capture, failed setup/start/stop,
preservation of unrelated instances, bounded finite arguments, and strict absence
of diagnostic filesystem/console/metadata I/O inside the timed interval. A timed
wait error is stopped before its diagnostic is printed. Python syntax/compile,
CLI help, whitespace and **46 relative documentation links** pass; independent
code/procedure review reports no material findings. No C++ files are changed.

### 2026-10-24 commit/push checkpoint

The operator approved committing and pushing the settled capture setup on
`jrmales/dm-syncopation` to `origin` (`magao-x/MagAOX`). The first commit
contains the helper/tests and associated engineering records, including the
latest capture findings; the second contains the operator procedure. Validation
remains the **17 passing offline tests**, Python syntax/compile, CLI help,
whitespace and documentation-link checks recorded above. No functional changes
were made after that verification. The next hardware comparison remains an
operator action; the helper update needs no camera/reader rebuild or restart.

## 2026-10-24 operator capture with explicit setup and settling phases

The operator supplied
`/home/jrmales/tmp/camwfs-trace-20261025T040828Z` and confirmed no camera/loop,
workload or CPU/IRQ changes from the previous run. Times below are **October 25
UTC**, corresponding to **October 24 in America/Phoenix**. Camera identity is
unchanged: PID **3077681**, acquisition TID **3077683**, FIFO **90**, CPU **33**,
IRQ **405** on CPU **61**, inode **3335**, session **5**, mode **3**, software
bin **1x1**. Neither CPU has an SMT sibling. There was no camera restart.
Boot ID remains `b63fa319-8340-4848-8f36-dfadd059a064`.

The new helper ran successfully with complete phase anchors and no capture or
cleanup errors. Metadata confirms **five-second settling followed by an explicitly
requested ten-second capture**, rather than the new five-second capture default.
The operator subsequently agreed to repeat using `--settle-seconds 5 --seconds 5`.

Input SHA-256 values, relative to the supplied directory:

| File | SHA-256 |
| --- | --- |
| `camwfs-trace-off.txt` | `106a3eb742982531e9519435b9e3a7d488e1b094f88cbc00bf668add6f253f5e` |
| `camwfs-trace-on.txt` | `21766da0e3e4d921c519aa4013595e220db275b687ba8e42ffd67ef684f243f9` |
| `camwfs-kernel-20261025T040828Z/trace.txt` | `90a42f8acf52a890dd7125c63487dc719976a72620ef7a5f1658c7c4bc2c3d8c` |
| `camwfs-kernel-20261025T040828Z/metadata.json` | `67e590601014e3f2a58e60de8ee9e5519160cef5f1a0719528b15c9251fd09d6` |

### The larger excursion falls inside trace setup

Phase brackets, rounded from their realtime anchors, are:

| Phase | UTC start | UTC finish |
| --- | --- | --- |
| Before metadata | 04:08:28.281983 | 04:08:28.354042 |
| Private trace setup | 04:08:28.356368 | 04:08:28.567230 |
| Settling, tracing off | 04:08:28.579131 | 04:08:33.579205 |
| Enable control write | 04:08:33.579211 | 04:08:33.579360 |
| Disable control write | 04:08:43.579451 | 04:08:43.579583 |
| After metadata | 04:08:44.610794 | 04:08:44.674383 |

The two large chronological centers, IDs **3240/3241**, publish at
**04:08:28.453519/04:08:28.453879 UTC**, within the explicit setup phase,
about five seconds before tracing is enabled. Their publication intervals are
**1138.810/359.769 us**, SDK-return intervals **1140.714/359.148 us**, and SDK
acquisition intervals **1138/362 us**. Timeout queries are **0.220/0.240 us**,
conversion **26.220/26.830 us**, and deferred rearm **5.390/5.451 us**.
Thus the burst is already upstream of the query/conversion/publication work.

This narrows the previous capture's metadata/setup-envelope association to
the private trace setup itself. It supports setup-related perturbation as an
explanation, but no kernel events or internal setup-operation brackets exist
at that time. It does not identify buffer allocation, clock/event setup or
another particular operation as the cause. There are no selected centers in
before-metadata collection, settling, either control-write bracket or
after-metadata collection. Two occur during enabled capture; five occur after
after-metadata collection finishes.

The short interval only partly compensates for the long one. The two SDK
acquisition intervals sum **1500 us**, versus **1000 us** nominal; publication
intervals sum **1498.579 us**. Header counters remain consecutive, but this
does not prove unchanged phase or frame age. SDK timestamps describe host DMA
completion under the reviewed contract, not physical exposure. Normal cadence
after settling must not be described as proof that the camera has no backlog.

### Quality and pooled timing

Both files contain **29 disjoint two-second summaries**: **115971 intervals
off**, **115970 on**, all qualified. There are no new wide write brackets,
invalid markers, failed/rejected acquisitions, publication failures,
frame/counter gaps or source/control queue drops. All **12 off** and **9 on**
chronological contexts are qualified, complete and consistent with their
neighbor interval identities; no new diagnostic selection drops occur.
Cumulative diagnostic drops remain the historical **2891**. Maximum ready
batch is two, below capacity sixteen.

Pooled values use population counts and disjoint summaries. Units are
microseconds; deviations are interval standard deviations.

| Population | Windows | Intervals | Publication deviation | Mean acquisition to write |
| --- | ---: | ---: | ---: | ---: |
| Entire kernel-off file | 29 | 115971 | 0.782 | 49.289 |
| Entire kernel-on file | 29 | 115970 | 2.084 | 49.461 |
| On file excluding the one setup-containing summary | 28 | 111972 | 0.804 | 49.464 |
| On summaries wholly before helper metadata | 5 | 19995 | 0.803 | 49.271 |
| On summaries wholly inside settling | 2 | 7998 | 0.757 | 49.450 |
| On summaries wholly inside requested enabled interval | 4 | 15996 | 0.804 | 49.809 |
| On summaries wholly after helper metadata | 15 | 59985 | 0.809 | 49.427 |

The phase-contained rows omit boundary-crossing summaries and are not additive
to the whole-file rows. Latency populations have one additional frame per window
compared with the listed interval populations. The setup-containing summary has publication deviation
**10.386 us** and dominates the whole on-file result. Its explicit exclusion
is useful for understanding steady timing, but must accompany the whole-file
result. The twenty-three summaries outside the requested kernel interval
include setup and therefore are not a clean trace-off comparison.

The earlier first-file broad spread does not repeat: every current off-file
window has publication deviation **0.721–0.873 us**, with conversion mean
**26.946–27.016 us**. This baseline precedes the helper invocation, so its
improvement over the previous **2.224 us** pooled value cannot be attributed
to the new settling interval. The camera implementation/process is unchanged.
Current off-file SDK-acquisition and SDK-return deviations are **0.381/0.407 us**;
mean return-to-write is **27.453 us**. Timeout-query mean is about **0.238 us**
in both files, with summary maxima **0.561 us off / 0.672 us on**. Mean main
notification is **4.623 us** in requested-inside summaries versus **3.840 us**
outside, consistent with the previously observed tracing cost; the outside
group includes setup and is explicitly qualified above.

### Kernel coverage and the remaining selected pair

CPU 33 again overwrites events: **50203**, retaining **7.992017 s**.
CPU 61 retains **10.000009 s** with zero overwrite. Exact joint retained
coverage is monotonic **1769154.177512–1769162.169486**, or **7.991974 s**.
Both commit-overrun and dropped-event counters are zero. The four summaries
inside the requested interval are not all wholly inside retained joint kernel
coverage; part of the first such window is lost on CPU 33.

Both selected contexts during enabled capture survive in full, comprising
**six unique neighboring frames**. Their publication intervals are
**524.841/479.184 us**, already **525.152/479.355 us** at SDK return and
**525/480 us** in SDK acquisition timestamps. The late interval has a **50 us**
combined control/acquisition IRQ after a **52 us** combined IRQ, with **526 us**
between entries. The early interval then has a **30 us** acquisition-only IRQ
after the **50 us** combined handler, with **500 us** between entries. Handler
duration explains part of the pattern, but not all of the late IRQ-entry spacing.

In those six frames, query duration is **0.231–0.662 us**, conversion
**26.410–27.171 us**, rearm **5.290–9.829 us**, main notification
**4.508–4.849 us** and sync notification excluding rearm **0.832–0.982 us**.
There is no long post-return segment in these selected neighborhoods. The pair
falls in the approximately **56 ms summary gap** after the 04:08:43.274 update;
all four requested-inside summaries consequently classify zero excursions.
Summary counts alone would miss this pair.

All **20706 complete EDT IRQ handlers** run in idle context on CPU 61;
one additional initial handler lacks its entry because enable occurs mid-handler.
All **15985** observed acquisition switchouts are voluntary sleeps. Matched
waking-to-run latency is mean **1.582 us**, median/p99 **2 us**, maximum **3 us**.
The full CPU 61 trace contains **20001 acquisition-waking events** with
adjacent spacing **480–525 us**, median **500 us**; no interval exceeds
550 us or falls below 450 us during the retained enabled trace. That constrains
these host wake events, not exposure timing or absolute frame age.

Across the retained camera-CPU trace, **54** scheduled runs last **60–79 us**.
All overlap a remote EDT handler, with no local vector entry; 51 handlers wake
only the control thread and three wake no task. Without corresponding raw
frame endpoints, these unselected intervals cannot be assigned to a particular
SDK stage or treated as long publication intervals. Earlier matched rearm
serialization evidence remains valid, but this capture adds no selected
tens-of-microseconds query/rearm stall within joint kernel coverage. Other
selected neighborhoods retain **sixteen off-file** and **eight on-file** rearms
above 20 us, with maxima **37.361/26.190 us** respectively. These calls remain
after publication but lack joint kernel coverage for IRQ attribution.

The next operator comparison retains five-second settling and explicitly uses
`--seconds 5`, with the current TID and both CPUs, to avoid the observed
ten-second overwrite. The individual setup perturbation and physical frame-age
effects remain unresolved. No camera code/configuration, scheduler margin,
helper implementation, commit or push changes accompany this analysis.

Analysis artifacts are preserved under `/tmp` with suffix `040828`, including
`parsed_fg_capture_040828.json`, `fg_capture_results_040828.txt`,
`fg_capture_phase_results_040828.txt`, `fg_capture_windowphase_results_040828.txt`,
`review_ocam_kernel_040828.py`, `ocam-kernel-analysis-040828.json`,
`ocam-frame-kernel-correlations-040828.csv`,
`ocam-centered-irq-correlations-040828.csv` and
`ocam-global-kernel-outliers-040828.csv`. Previous artifacts remain intact.
Only this engineering record and the synco debugging record change. All agent
work uses local copied files; the operator performs every instrument action.

## 2026-10-24 loss-free five-second capture

The operator supplied
`/home/jrmales/tmp/camwfs-trace-20261025T041847Z` after selecting the explicit
five-second duration. Metadata confirms **five seconds of settling and five
seconds of capture**, successful start/completion, no capture/cleanup errors,
and unchanged observed camera/IRQ placement. PID **3077681**, acquisition
TID **3077683**, FIFO **90** on CPU **33**, IRQ **405** on CPU **61**, inode
**3335**, session **5**, mode **3** and software bin **1x1** match the preceding
run. Boot ID is `b63fa319-8340-4848-8f36-dfadd059a064`; neither CPU has an SMT
sibling. All work uses local copied files. Dates/times below are **October 25
UTC**, the evening of **October 24 in America/Phoenix**.

Input SHA-256 values, relative to the supplied directory:

| File | SHA-256 |
| --- | --- |
| `camwfs-trace-off.txt` | `24d8b20f6c201f26a913fa968aa365d67d3e9245cad45184505a065e7786155e` |
| `camwfs-trace-on.txt` | `5d7ec4beaaf672acbaee75d3e26d712b28a7bc1337ebd96835352b0705f75b95` |
| `camwfs-kernel-20261025T041847Z/trace.txt` | `c671254cacf13faf527aeb103304a5890e0cf5a0feecf020622408c2e697c794` |
| `camwfs-kernel-20261025T041847Z/metadata.json` | `50e97aa41e013b4629db725de8887e1c11b1ee9f1a5652954b6b9e42d477fd3c` |

### Phase boundaries and complete retained coverage

| Phase | UTC start | UTC finish |
| --- | --- | --- |
| Before metadata | 04:18:47.380825 | 04:18:47.450392 |
| Private trace setup | 04:18:47.452575 | 04:18:47.651154 |
| Settling, tracing off | 04:18:47.656060 | 04:18:52.656128 |
| Enable control write | 04:18:52.656132 | 04:18:52.656256 |
| Disable control write | 04:18:57.656346 | 04:18:57.656480 |
| After metadata | 04:18:58.292966 | 04:18:58.359371 |

Both CPUs report **zero overrun, commit overrun and dropped events**. CPU 33
retains **124699 events** over **5.000045 s** and CPU 61 **54965 events** over
**5.000033 s**. Exact joint event coverage is monotonic
**1769771.246472–1769776.246473**, or **5.000001 s**. Event endpoints are not
the exact tracing-switch instants; the metadata brackets the switches separately.
The shorter duration removes the observed ring overwrite and fully covers both
selected capture contexts, comprising **ten distinct neighboring frames**.

No chronological center falls in metadata collection, setup, either control-write
bracket or after-metadata collection. Two on-file centers precede the helper,
one falls during settling, two during capture, and five after the helper.
The setup-containing summary has publication deviation **0.920 us**, SDK
acquisition intervals **487–513 us** and no classified excursions. The previous
millisecond setup burst does not repeat. Thus setup-associated perturbation is
intermittent; the five-second duration fixes ring retention but cannot be
credited with removing the earlier burst, which happened before capture began.

### Stable whole-file populations and remaining qualifications

Both files contain **29 disjoint two-second summaries**, each with **115971 raw
intervals**. Every off-file interval qualifies. The on file has **115969
qualified** and **two ambiguous** intervals, caused by one **1.102 us** write
bracket exceeding the configured **1 us** limit in the 04:18:58.018678 summary.
Pooled raw statistics below retain those intervals. There are no new acquisition
or publication failures, rejections, invalid markers, frame/counter gaps,
source/control queue drops or diagnostic selection drops.

All **four off-file** and **ten on-file** chronological contexts are qualified,
complete and consistent with repeated frames/neighbor interval identities.
Maximum ready batch is two, below capacity sixteen; cumulative diagnostic
drops remain the historical **2891**. Off-file contexts are all early;
on-file contexts are four late and six early.

Pooled values use disjoint summaries and their population counts. Units are
microseconds; deviations are interval standard deviations.

| Whole-file metric | Kernel off | Kernel on |
| --- | ---: | ---: |
| SDK-acquisition-interval deviation | 0.380 | 0.378 |
| SDK-return-interval deviation | 0.411 | 0.403 |
| Publication-interval deviation | 0.791 | 0.800 |
| Mean acquisition to write | 49.256 | 49.393 |
| Mean return to write | 27.442 | 27.524 |
| Mean / maximum timeout query | 0.237 / 0.641 | 0.237 / 1.323 |

The whole on-file value is close to the preceding capture's **0.804 us** after
excluding its setup-containing window, without needing such an exclusion in
this run. There is no millisecond event or repeat of the earlier broad-spread
period in these excerpts. This does not establish a continuous worst-case bound.

Phase-contained aggregate subsets, excluding boundary-crossing windows, are:

| On-file population | Windows | Intervals | Publication deviation | Mean acquisition to write |
| --- | ---: | ---: | ---: | ---: |
| Before helper metadata | 5 | 19995 | 0.846 | 49.253 |
| Settling | 2 | 7998 | 0.707 | 49.393 |
| Enabled capture | 1 | 3999 | 0.764 | 49.767 |
| After helper metadata | 17 | 67983 | 0.792 | 49.405 |

These rows are subsets of the whole file, not additional populations. The
latency populations contain one more frame per window than the interval
populations. Only **one complete two-second summary** fits wholly inside the
five-second trace; **0.764 us** is its deviation, not a full-five-second RMS.
Mean main-notification duration is **4.618 us** inside versus **3.840 us**
outside the requested interval, repeating approximately **0.779 us** of tracing
cost. Both captured centers and all four off-file centers fall in the persistent
approximately **56 ms summary gaps**; the sole complete inside summary and
all off summaries classify zero excursions despite those chronological records.

### Covered excursions and deferred rearm serialization

ID **3329** publishes at **04:18:53.911396 UTC** with interval **524.781 us**,
SDK return **524.811 us**, and SDK acquisition **525 us**. ID **3330** publishes
at **04:18:55.968379 UTC** with interval **479.806 us**, SDK return **480.016 us**,
and SDK acquisition **481 us**. They are approximately **2.057 seconds apart**,
not an adjacent late/early pair. Most of the timing deviation is already present
at SDK return; subsequent stage changes can still affect strict classification
at the 20 us threshold.

The late center follows combined control/acquisition EDT handlers of **51 then
50 us**, with **525 us** between IRQ entries. The early center follows a
**50 us** combined handler with a **30 us** acquisition-only handler at
**500 us** entry spacing. Its previous publication interval is **509.823 us**,
classified normal, so this is another example against assuming that every early
interval follows a classified late one.

Across the ten covered neighboring frames, timeout queries last
**0.230–0.370 us**, conversion **26.500–27.192 us**, main notification
**4.388–4.809 us**, and sync notification excluding rearm **0.812–1.103 us**.
Two queries overlap remote **36 us** control handlers but remain **0.370/0.231 us**.

Three deferred rearms last **23.264/24.396/23.525 us**. Each overlaps a remote
EDT IRQ waking only the control thread, lasting **35/36/36 us**, with no local
interrupt/vector entry or thread switch during the rearm. Rearm mean
**23.728 us** comprises mean remaining IRQ overlap **18.440 us** plus
**5.288 us** after IRQ exit. All three begin **5.461–5.741 us after the write
endpoint** and **0.932–0.942 us after main notification**, with sync notification
already complete. This supplies further direct evidence of the same
postpublication driver/device serialization. It does not identify a driver lock.
Other retained raw neighborhoods also contain long rearms, reaching
**25.539 us off / 26.841 us on**, outside this jointly covered selected set.

All **10293 EDT IRQ handlers** are complete and execute on CPU 61 in idle context.
All **10001 acquisition switchouts** are voluntary sleeps. Matched waking-to-run
latency is mean **1.596 us**, median **2 us**, p99 **3 us**, maximum **4 us**.
There is no observed runnable-camera displacement. The full CPU 61 trace has
**10001 acquisition wakes**, yielding **10000 intervals** spanning **481–524 us**,
median **500 us**. No large wake excursion occurs during this trace.

Globally, **29** camera execution runs last **61–79 us**, each overlapping a
remote EDT handler, with no local vector entry. Twenty-seven handlers wake only
control and two wake no task. Without matching per-frame endpoints their stage
cannot be assigned. Do not infer a publication delay from the full run duration.

The five-second capture procedure now provides the intended complete kernel
history for this event rate. The remaining demonstrated long rearms are after
publication, while occasional approximately 20–25 us interval excursions still
enter by SDK return. Prior setup-associated millisecond transients and absolute
frame age remain unresolved; this capture contains no independent exposure or
frame-age reference. No timing margin or universal late-then-early policy is
selected.

Reproducible scripts/results are retained under `/tmp` with suffix `041847`,
including `parsed_fg_capture_041847.json`, `fg_capture_results_041847.txt`,
`fg_capture_phase_results_041847.txt`, `fg_capture_windowphase_results_041847.txt`,
`review_ocam_kernel_041847.py`, `ocam-kernel-analysis-041847.json`,
`ocam-frame-kernel-correlations-041847.csv`,
`ocam-centered-irq-correlations-041847.csv` and
`ocam-deferred-rearm-correlations-041847.csv`. Earlier artifacts remain intact.
Only this engineering record and the synco debugging record change; no camera
or helper implementation, configuration, commit, push or agent instrument action
accompanies this analysis.

## 2026-10-24 proposed SDK/driver attribution

The operator asked whether the remaining SDK-return excursions can be localized
to camera/card, kernel driver, SDK work or scheduling. Existing matched frames
already provide this decomposition, using the printed microsecond IRQ times:

| Covered center | IRQ-entry interval (us) | Handler-duration change (us) | IRQ-exit-to-SDK-return change (us) | SDK-return interval (us) |
| --- | ---: | ---: | ---: | ---: |
| Late ID 3329 | 525 | -1 | +0.811 | 524.811 |
| Early ID 3330 | 500 | -20 | +0.016 | 480.016 |

The late example is largely present by software IRQ-handler entry; the early
example largely reflects handler-duration variation. For these examples,
selective driver observations are more informative than another generic app
timestamp. First establish the actual loaded module/library build and available
driver tracing symbols, then observe completion processing, timestamp storage,
serial handling and acquisition wakeup if supported. The proposal below also
describes request tracing to resolve SDK/driver gaps in other cases. No new
measurement helper or driver probe is implemented here.

The late example's preceding control-only IRQ ends at monotonic
**1769772.501390**. Nominal next acquisition entry, based on the preceding
acquisition entry plus 500 us, is **1769772.501415**; actual combined entry is
**1769772.501440**. There are no retained CPU 61 events in the intervening
50 us gap. The preceding control handler therefore does not overlap that
nominal arrival. Silence in these events does not establish physical DMA or
interrupt assertion time, nor prove continuous interrupt enablement. The expanded
timeline is retained as `/tmp/ocam-late3329-cpu61-timeline-041847.txt`.

The copied PDV 5.6.8.0 source confirms that the app's existing
`waitEnter`/`waitReturn` bracket encloses more than the blocking frame wait:

```text
pdv_wait_last_image_timed()
  buffer-done query / newest-buffer selection
  timeout queries
  blocking WAITBUF request
  WAIT_STATUS request
  optional frame-sync / software processing
  TMSTAMP request to retrieve the stored driver timestamp
  SDK return -> existing app marker
```

Relevant source is `/home/jrmales/Source/libpdv.c` at 5190, 5558 and 5671,
and `/home/jrmales/Source/libedt.c` at 2392, 2539, 4175 and 8599. Linux EDT
requests all use the same outer `EDT_NT_IOCTL` syscall command; their actual
action is `edt_ioctl_struct.controlCode`. Important actions are **36 BUFDONE**,
**40 WAITBUF**, **47 TIMEOUTS**, **161 WAIT_STATUS**, **87 TMSTAMP** and
**39 STARTBUF**. A generic syscall-command trace alone cannot identify them.
An attribution capture must decode the nested action or also record the
library `edt_ioctl()` action argument with a targeted probe. Confirm the actual
loaded library/build and available trace features before specifying probe offsets.

Proposed request comparison: add request/action entry and exit events scoped to
the acquisition TID, preserving unfiltered selected-CPU IRQ/scheduler events,
five-second settling/capture, counter/frame matching and loss accounting.
Separate pre-wait queries, blocking WAITBUF, post-wake status/timestamp requests
and user-space gaps. A long blocking wait alone is not evidence of a slow driver:
waiting for the next camera frame is expected. Compare IRQ entry, acquisition
wakeup, actual schedule-in and request return to localize its excess. Kernel
[event filters](https://www.kernel.org/doc/html/v5.14/trace/events.html),
[kernel entry/return probes](https://www.kernel.org/doc/html/v5.14/trace/kprobetrace.html)
and [user-library probes](https://www.kernel.org/doc/html/v5.14/trace/uprobetracer.html)
are possible mechanisms, conditional on the running kernel's configuration and
available symbols. Measure their overhead as a separate controlled comparison.

For the covered handler variation, selectively observe buffer completion,
timestamp storage, wakeup and serial handling, using existing vendor trace
support or verified driver function probes. The copied SDK has a decoder for
`ISR_START/END`, `DONE`, `TMSTAMP`, `SERIAL`, `WAKEUP`, `WAIT_START/END` and
`LOCK/UNLOCK` records, and an `EDTG_TRACEBUF` getter (action **48**). Its small
ring, timestamp semantics, enabled instrumentation and overhead in the running
driver remain unverified. Repeated retrieval adds device ioctls and may itself
contend; it is not yet an approved continuous measurement. The supplied current
SDK tree lacks the running driver core; the historical `Old_mx7` wrapper does
not establish the current implementation or precise timestamp placement.

IRQ-handler entry is a software observation, not the physical DMA-completion
or interrupt-assertion instant. If delay precedes IRQ entry, these software
measurements cannot uniquely separate camera/readout, card/DMA/PCIe and delayed
host interrupt servicing. That final split needs an independent camera
exposure/frame-valid or supported hardware timestamp reference. The ordinary
SDK `atime` is itself subject to interrupt response delay; its timing does not
prove exposure jitter. No new code, probes, instrument actions, commit or push
accompanies this proposal.

## 2026-10-24 complete EDT copy supplies current driver information

The operator replaced the partial EDT copy with a complete archive. The
extracted tree is **`/home/jrmales/Source/opt/EDTpdv`**, with archive
`/home/jrmales/Source/EDTpdv.tgz`; the earlier
`/home/jrmales/Source/EDTpdv` path no longer exists. This materially improves
the preceding proposal: the tree now includes the current Linux wrapper source,
proprietary driver core object and linked kernel module. No vendor program,
module or probe was executed; inspection uses source, ELF metadata, symbols
and static disassembly on the workstation.

`version.txt` and `edt.ko` report **5.6.8.0**. Module metadata gives
`vermagic=5.14.0-362.8.1.el9_3.x86_64 SMP preempt mod_unload modversions`,
matching the kernel release in the captures; `srcversion` is
`A26E0B4FD875E93168B83BC`, and ELF build ID is
`97857cff4fc9d75717d3b58e9c5bc3359a7204f0`. The files
`module/edt.ko` and `5.14.0-362.8.1.el9_3.x86_64/edt.ko` are byte-identical.
These initially identified only the supplied build. The operator subsequently
confirmed the loaded module's matching identity, as recorded below.

| File relative to the new tree | SHA-256 |
| --- | --- |
| `module/edt.ko` | `e49afdb6e8261459c86cf9cc41ea671defb24bc0e29503774891a505b46bf522` |
| `module/edtdrvx86_64_11.o` | `e8fef3ad4732e00d2c0c575e0df19e6a4b71881b873b605632ddb8c2870745ae` |
| `module/edt_lnx_kernel.c` | `a1f9fd89e691f5742d84558b3b32e8c295d01a7548bc73af9a21af08974e7485` |

The core remains a **compiled proprietary object**, rather than supplied C
source. However, the object/module retain symbols and DWARF information, so
the timing-relevant call paths and candidate function boundaries are available.

### Ordinary timestamp placement is now identifiable

Static inspection of `edt_increment_done` finds a call to
`EdtTraceTimeStamp` with the current ring buffer's timestamp address before
completion-event broadcast. In ordinary timestamp mode its clock path is:

```text
edt_increment_done
  -> EdtTraceTimeStamp
     -> edt_get_hres_time
        -> edt_do_gettimeofday
           -> ktime_get_real_ts64
```

The wrapper at `module/edt_lnx_kernel.c:1110` divides `tv_nsec` by 1000 to
form microseconds; the core then multiplies microseconds by 1000 for the
nanosecond timestamp field. Thus this ordinary `atime` is **CLOCK_REALTIME
sampled during driver completion processing, quantized to 1 us**. It is not
a hardware-latched physical DMA-completion or exposure timestamp. Nondefault
timestamp modes have additional conditions and are not generalized here.

The normal PDV interrupt path is recoverable as
`edt_intr_func -> edt_intr -> do_intr -> edt_test_intr -> edt_DpcForIsr`.
When the conditional PDV device/serial callback is taken, its processing precedes
the DMA-completion path through `edt_increment_done`; completion processing precedes acquisition
wakeup. This gives a concrete mechanism by which variable device/serial
handling can shift the software stamp even with steady physical delivery.
It strengthens the interpretation of the observed 50-versus-30 us handlers,
but does not establish which branches executed in each captured handler.

### Concrete lock and tracing targets

Current `module/edt_lnx_preempt.c:15` routes critical sections through channel
zero's `crit_lock_ptr`. The wrapper uses `spin_lock_irqsave` and restores
interrupt flags at release. The static IRQ path enters with source **32**;
rearm's `edt_start_buffers` enters with source **7**. These are concrete shared
lock paths consistent with the observed rearm/IRQ serialization; actual lock
wait and hold times remain unmeasured.

Useful candidate observations are `edt_test_intr`, `edt_DpcForIsr`,
`edt_increment_done`, `EdtTraceTimeStamp`, serial-processing functions and
entry/return of `edt_start_intr_critical`, with the lock pointer, CPU/TID and
source context. Retain acquisition `sched_waking` and the generic EDT wake
wrappers: the normal IRQ path calls `edt_wake_up[_interruptible]` from `do_intr`
instead of relying on the separately named `edt_wakeup_dma` helper. Probing
that helper alone would miss the normal wake path.

The module also implements trace-size/get/clear functions. Vendor trace
retrieval adds ioctls and has debug/printing paths; the legacy ring is small.
The complete copy permits inspecting these facilities but does not qualify
continuous vendor tracing as a low-overhead measurement. Prefer a small set of
qualified probes for the initial driver comparison.

Named core functions do not automatically support function-graph tracing:
the inspected core prologues lack `fentry`/`mcount`, while wrapper sections
are instrumented. The loaded build is now matched below; kernel-probe
eligibility and symbol exposure still require operator checks. No recompilation is
inherently required to observe supported function boundaries, but no probe
has yet been installed or validated.

The remaining physical boundary is unchanged: software completion processing
and IRQ entry do not reveal the exact camera/card/DMA or interrupt-assertion
instant. An independent hardware reference is still required to separate
delays already upstream of those software observations. Only engineering
records change; no vendor/app implementation, configuration, commit, push or
agent instrument action accompanies this inspection.

Reproduction commands and static evidence are retained in
`/tmp/collect_edt_current_static_evidence.py`,
`/tmp/edt-current-static-evidence.txt`,
`/tmp/edt-current-module-static-analysis.json` and
`/tmp/edt-current-module-static-review.md`.

## 2026-10-24 loaded EDT module identity verified

The operator supplied these readbacks from **exao2**:

| Readback | Value | Comparison with the copied module |
| --- | --- | --- |
| `uname -r` | `5.14.0-362.8.1.el9_3.x86_64` | Matches its kernel release |
| `/sys/module/edt/version` | `5.6.8.0` | Matches |
| `/sys/module/edt/srcversion` | `A26E0B4FD875E93168B83BC` | Matches |
| `/sys/module/edt/notes/.note.gnu.build-id` | `97857cff4fc9d75717d3b58e9c5bc3359a7204f0` | Matches |

The note starts with the 16-byte ELF note header/name
`04 00 00 00 14 00 00 00 03 00 00 00 47 4e 55 00`; its remaining 20 bytes
are the build ID above. A repeat local `readelf -n module/edt.ko` confirms the
same ID in the complete copied tree. This resolves loaded-module identity for
the proposed driver investigation and ties the static function-boundary
analysis to the running build. It does not establish runtime probe support
or identify the loaded user-space library.

The next operator checks are read-only: inspect the running kernel's Kprobes
configuration, the tracefs `kprobe_events` interface, the selected EDT names
in `/proc/kallsyms`, and any reported kernel lockdown state. Missing interfaces
can reflect configuration or mounts; symbol presence alone does not establish
that a probe can be registered. Actual registration and measurement overhead
remain later qualification steps. No probes, module changes or tracing changes
accompany this verification; only the two engineering records change.

## 2026-10-24 operator confirms kernel probe prerequisites

The operator supplied the requested read-only checks from exao2:

- `CONFIG_KALLSYMS=y`, `CONFIG_KALLSYMS_ALL=y`, `CONFIG_KPROBES=y` and
  `CONFIG_KPROBE_EVENTS=y` in the running kernel's configuration.
- `/sys/kernel/tracing/kprobe_events` exists with root ownership and mode
  `0640` (`-rw-r-----`).
- `/proc/kallsyms` exposes all ten selected names in module `[edt]`:
  `edt_test_intr`, `edt_DpcForIsr`, `edt_increment_done`, `EdtTraceTimeStamp`,
  `pdv_DeviceIntPdv`, `edt_start_intr_critical`, `edt_end_intr_critical`,
  `edt_start_buffers`, `edt_wake_up` and `edt_wake_up_interruptible`.
- `/sys/kernel/security/lockdown` reports `[none] integrity confidentiality`;
  the active mode is **none**.

These satisfy the interface/configuration/symbol prerequisites. They do not
yet establish that the kernel accepts each entry/return probe. The next check
uses the new operator-run
[`check-edt-probes.py`](../../../apps/ocam2KCtrl/scripts/check-edt-probes.py)
to register the ten entry and ten return definitions, verify they remain
disabled, and remove only its own definitions. It checks loaded module identity
before and after the registration attempts; no event is enabled and no capture is taken.
The registration result and cleanup outcome must be reported before selecting
the first active probe set.

The helper follows the Linux 5.14
[Kprobe event interface](https://www.kernel.org/doc/html/v5.14/trace/kprobetrace.html):
module-qualified function boundaries, a unique event group, append-only
definitions and selective deletion. It uses no offsets or fetched arguments.
This isolates registration eligibility from later questions about argument
layout, hit/miss coverage, active instrumentation overhead and retained trace
coverage. The original scheduler/IRQ capture helper remains unchanged.

Return definitions use explicit `r1:` so their readback is deterministic:
Linux fills in a platform-dependent capacity when the request leaves it at
zero/default, and includes that capacity in `kprobe_events` output. The single
reserved slot has no live measurement meaning while disabled; an active
capture must choose return capacity and qualify missed returns separately.
Cleanup checks exact definitions and qualifies each deletion with its EDT
target. Signals request a stop between operations, and output errors cannot
interrupt cleanup.

Validation: **16 offline checks pass** using ordinary temporary files and
mocked command writes. They exercise all twenty registrations, refusal of
pre-existing groups/events, root/mount/build prerequisites, registration
rejection, short/failed writes after application, changed definitions,
cleanup failure, external enablement, interruption, failed result output and
post-registration module changes. CLI help, `git diff --check` and all **54**
local Markdown links across the two records and operator guide also pass.
Independent review found no remaining blockers for the registration-only
operator check. No instrument registration result is claimed yet.

## 2026-10-24 disabled registration isolates the kernel notrace restriction

The operator ran `check-edt-probes.py` on exao2. Loaded EDT identity matched
before and after registration. The invocation used group
`ocam_edt_check_3111568_ad0e407e18cb4dc581e933a4be53ec21` and finished with
**20 attempted, 12 errors**:

| Entry and return result | Functions | Definitions |
| --- | --- | --- |
| Registered disabled, then removed | `edt_wake_up`, `edt_wake_up_interruptible`, `edt_start_intr_critical`, `edt_end_intr_critical` | 8 |
| Rejected with `EINVAL` | `edt_DpcForIsr`, `edt_increment_done`, `edt_test_intr`, `EdtTraceTimeStamp`, `pdv_DeviceIntPdv`, `edt_start_buffers` | 12 |

All eight reported registrations have corresponding successful removal lines;
there are no reported cleanup errors. The twelve errors are registration
rejections. No event was enabled and no timing capture was performed.

Local inspection finds the same split in the copied module. Accepted functions
are Linux wrappers with `__fentry__` sites. Rejected functions come from
`edtdrvx86_64_11.o` without those sites. All ten are global `STT_FUNC` symbols
in ordinary `.text`; the rejected functions begin with ordinary x86 function
prologues, and the module has no `__kprobe_blacklist` section. These facts alone
do not make generic kprobes unavailable, but they explain the following
confirmed tracefs policy.

The operator's follow-up configuration reads:

```text
CONFIG_KPROBES_ON_FTRACE=y
CONFIG_DYNAMIC_FTRACE=y
# CONFIG_KPROBE_EVENTS_ON_NOTRACE is not set
```

Kernel messages at monotonic **1774038.748405–1774039.304395 s** explicitly
report `Could not probe notrace function edt:...` for each rejected function,
twice each for entry and return attempts. This confirms the cause. Linux 5.14's
[`within_notrace_func` check](https://github.com/torvalds/linux/blob/v5.14/kernel/trace/trace_kprobe.c#L410-L484)
rejects functions without an ftrace site when dynamic ftrace is enabled and
`CONFIG_KPROBE_EVENTS_ON_NOTRACE` is unset. It returns `EINVAL` before either
kind of probe is registered. The restriction tests the whole function range;
instruction offsets or numeric address spelling do not resolve it.

### Revised first active measurement

Use the accepted lock and wakeup wrappers with the existing native IRQ and
scheduler events. A bounded capture can then distinguish the lock-acquisition
wrapper's elapsed duration, the interior held-lock interval before unlock,
and acquisition wakeup/dispatch relative to EDT IRQ entry/exit. Record the
lock/wait-queue pointer and retain CPU, task and interrupt context so unrelated
locks or nested calls are not paired. The acquisition bracket includes wrapper
and probe overhead; return-to-unlock-entry covers only the interior of the
held interval. Neither should be labeled an exact unperturbed spin-wait/hold
duration.

Qualify active overhead, complete entry/return pairing, return-probe misses,
and per-CPU ring loss against the existing camera telemetry before assigning
causality. A lock owner outside the selected CPUs can remain unseen, even when
the observed acquisition bracket is long. Keep setup, settling and active
capture phases separate as in the existing helper.

Direct serial/completion/timestamp-core probes are unavailable through this
tracefs configuration. The instrumented `edt_do_gettimeofday` wrapper at
`module/edt_lnx_kernel.c:1110` is a candidate substitute along the ordinary
completion-clock path, but its registration and call attribution remain
untested. It can also serve other driver callers, so a clock call alone does
not identify a particular frame's completion. This remains a software clock
observation; the physical camera/card/IRQ-assertion boundary is unchanged.

The next capture uses the available wrappers; no kernel configuration change
is needed for that approach. Active probe capture is **not yet implemented**.
This checkpoint changes only the operator guide and two engineering records;
the registration helper and camera implementation remain unchanged.

## 2026-10-24 accepted-wrapper capture implemented

The operator requested implementation of the revised capture. The existing
`capture-camera-timing.py` now accepts **`--edt-wrappers`**. Native mode retains
its previous scheduler/IRQ events and limits. The optional mode requests only
the four accepted wrappers, with entry and return definitions:

| Function | Entry fetch | Return reservation |
| --- | --- | --- |
| `edt_start_intr_critical` | `lock=%di:x64` | `r64` |
| `edt_end_intr_critical` | `lock=%di:x64` | `r64` |
| `edt_wake_up` | `queue=%di:x64` | `r64` |
| `edt_wake_up_interruptible` | `queue=%di:x64` | `r64` |

The x86-64 ABI/source and copied module disassembly agree on the first argument
in `%di`; no pointed-to kernel memory is dereferenced. Typed registration and
actual argument values still need operator qualification. No rejected core
function or untested clock wrapper is included.

### Ownership and timing window

The capture lazily loads the existing registration helper and reuses its
exact-build/mount preflight and `ProbeCheck` ownership. Reusable extensions add
literal entry fetch arguments, explicit return reservations, returned full
event names and per-invocation event prefixes; the registration-only CLI keeps
its existing ten-target/default-`r1` behavior. Names fit the kernel event-name
limit. Unique event names also disambiguate `kprobe_profile` implementations
whose first column omits the group.

Preflight runs before output/private-trace mutations. Definitions are first
verified disabled everywhere visible, then enabled only in the new private
instance. Existing `context-info` and `irq-info` options are required, with no
trace-option writes. Neither native nor probe events use a camera-PID filter.
Probes execute on calls from unrecorded CPUs too, so the recorded CPU mask does
not bound active hook cost or identify every possible lock owner.

Wrapper mode accepts **1–5 seconds**, allocating **32 MiB per selected CPU**
for at most four CPUs. Native mode retains 1–30 seconds and 8 MiB per selected
CPU. The first operator procedure uses one second, five seconds of settling,
and separate native/probe captures with overlapping full camera telemetry.
The changed allocation is part of this mode comparison, so it is not a pure
measurement of hook cost alone.

Probe hooks activate during setup and remain active during settling even while
`tracing_on=0`. Metadata brackets activation and deactivation separately from
the ring capture. Before/after raw and invocation-row `kprobe_profile` snapshots
cover the broader activation lifetime, including settling. `edt_wrappers` in
`metadata.json` also retains definitions, formats, verified build identity,
return capacity, safety errors and remaining definitions. All file/diagnostic
I/O stays outside the successful trace-enable/stop interval.

### Cleanup and interpretation

After stop, cleanup attempts every exact owned private disable before optional
diagnostic format reads, including partially registered/enabled setups. A
successful save permits selective definition removal and instance removal.
A save failure retains both the stopped buffer and disabled definitions;
unconfirmed stop/disable also retains them and reports exact paths/names.
Removing dynamic definitions before recovery could discard their formatter or
allow event-ID reuse, so the earlier idea of deleting definitions after a failed
save was rejected. No global trace is cleared and no unrelated enable is changed.

The capture saves raw observations for later context-aware pairing. Do not pair
by PID alone: idle IRQ contexts share PID zero. Check CPU, function nesting,
native IRQ/scheduler context and entry pointer identity; exclude incomplete
boundary calls. Acquisition-wrapper elapsed time includes probe overhead, and
return-to-unlock-entry covers only the interior held interval.

Profile miss semantics are deliberately left uncertified
(`return_pool_misses_verified=false`). The
[initial Linux 5.14 profile implementation](https://github.com/torvalds/linux/blob/v5.14/kernel/trace/trace_kprobe.c#L1116-L1130)
prints the underlying kprobe miss counter; a
[later kernel correction](https://github.com/torvalds/linux/commit/dfea08a2116fe327f79d8f4d4b2cf6e0c88be11f)
adds return-instance exhaustion. A release number does not establish which
backport this installed kernel contains. Retain counters and complete call
records; zero reported misses cannot alone prove return-pool completeness.

### Validation and operator follow-up

**53 offline tests pass**: 31 capture-lifecycle tests and 22 registration-owner
tests. Coverage includes default native behavior, typed accepted probes,
partial registration/enable, ordinary interruption, preflight/build changes,
missing context flags, modified definitions, save/stop/disable/removal failure,
decoder preservation, unrelated trace state and absence of diagnostic I/O
during recording. Independent source/lifecycle review passes. Python CLI help
and whitespace/link checks are run locally; no live tracing or instrument
commands are performed by agents.

The operator guide gives the paired native/one-second wrapper procedure. Use
both updated Python scripts; no camera, reader or kernel rebuild/restart is
needed for this helper change. Hardware qualification remains pending.

## 2026-10-25 first accepted-wrapper capture: direct shared-lock evidence

### Inputs, operating conditions and retention

The operator supplied `/home/jrmales/tmp/camwfs-step5-20261025T155627Z`.
The chronological order is **native trace, wrapper trace, then helper-free
telemetry**. Each trace requests one second after five seconds of settling.
The camera was off overnight, then powered on before these measurements. The
operator confirms that the software was not reconfigured: it followed its
normal power-cycle FSM states. CCD temperature falls from approximately
−30.5 to −36.8 C during the native file, −38.9 to −45.0 C during the wrapper
file, and remains near −45 C during the final helper-free file. This is a
changing condition, not evidence that cooling causes the timing variation.

Both traces retain the same camera PID **3077681**, acquisition TID
**3077683**, FIFO priority **90** on CPU **33**, and EDT IRQ **405** on CPU
**61**. The boot ID, stream inode **3335**, timing session **12**, bin2 mode,
2000 Hz setting, closed shutter and enabled synchronization agree. PID/boot
continuity with the previous capture does not imply uninterrupted camera
operation across the intervening power cycle.

| Input relative to supplied directory | SHA-256 |
| --- | --- |
| `camwfs-edt-native.txt` | `8b7e48e8a1ee1ce18da9ed489bb86f439300a54da942eacadb69d9ba83c03330` |
| `camwfs-edt-wrappers.txt` | `af8c1825029b3b33b69e0ed94808e0619c2d746f2dcd66eb318915f006df172a` |
| `camwfs-trace-off.txt` | `94cc314696459af91f870cacaf10c7091ce448c1436aba90a3dfd771df8de6e2` |
| `camwfs-native-20261025T155627Z/metadata.json` | `d13aa67907bc35c8e3205c2a248a792b35a0fba9c15544ef7e56abc23bb94ffb` |
| `camwfs-native-20261025T155627Z/trace.txt` | `6be550515f1201cb4f2c0a23f857660a46daaa697a51e2fe53fc1402f00d2526` |
| `camwfs-edt-20261025T155751Z/metadata.json` | `9a734ca82d9a891a68c59224d48da812745276fcdd828560bc92c2362c080117` |
| `camwfs-edt-20261025T155751Z/trace.txt` | `1110412f67bdef1794233bc5b3bff000684b1a2ac2e6aa7b9681a7c5e7913999` |

The joint retained native interval is approximately **15:56:33.382952–
15:56:34.382950 UTC**, and the wrapper interval is **15:57:57.483481–
15:57:58.483017 UTC**, both on October 25. Metadata separately brackets trace
enable/disable. Native CPU event counts are **24,072 / 11,442** on CPUs 33/61;
wrapper counts are **40,878 / 23,878**. Both CPUs report zero overruns,
commit overruns and dropped events in both runs. Capture and cleanup complete
without errors; no owned wrapper definitions remain, and post-stop module
identity still matches the verified EDT build.

Wrapper hooks are active from approximately **15:57:52.430–52.439** until
**15:57:58.483–58.592 UTC**, including settling with the ring disabled. Their
profile counters cover this broader lifetime and calls on unrecorded CPUs.
Reported misses are zero, but return-pool miss semantics remain unverified.
Actual retained call pairing, rather than whole-lifetime profile count equality,
is the completeness check used below.

### The same lock serializes acquisition and interrupt handling

The wrapper file contains **28,878 probe events**, yielding **14,439 complete
entry/return pairs** and **6,147 complete acquire/release cycles**. Pairing uses
CPU, TID, function, full IRQ context and lock identity. There are no unmatched
entries/returns, nested entries, orphan releases, open holds or unmatched IRQ
boundaries. Pointer formats are eight-byte unsigned values and actual arguments
are nonzero and consistent.

All **4,002** acquisition-thread lock cycles and **2,145** IRQ-side cycles use
**`0xffff94a582288ab0`**. This turns the earlier serialization inference into
direct evidence of a shared lock:

- **25** camera-side acquisition brackets exceed 2 us, with a maximum of
  **34 us**. Every one overlaps a held interval on the same lock on CPU 61.
  Twenty-four overlap an IRQ that wakes the main/control thread; one overlaps
  an IRQ with no observed wake.
- **7** IRQ-side acquisitions exceed 2 us, with a maximum of **5 us**. Every
  one overlaps a camera-thread hold of that same lock on CPU 33.

These are elapsed instrumented wrapper brackets on a 1 us printed trace grid,
including probe cost; they are not exact uninstrumented spin-wait durations.
Zero-duration printed calls are below the trace's displayed resolution.

The acquisition wake queue is `0xffff94a61b4f0640`: all **2,000** calls return
to `do_intr+0x175` and wake the acquisition thread. The other queue is
`0xffff94a61b4f0a00`: all **145** ordinary wake calls return to
`pdv_DeviceIntPdv+0x5f8` and wake the camera main thread. This directly associates
the latter events with the PDV device/control callback. It does not identify
the particular serial command responsible for each call.

| IRQ class in wrapper capture | Count | Mean IRQ duration, us | Mean held interior, us |
| --- | ---: | ---: | ---: |
| Acquisition wake only | 1,991 | 32.183 | 29.120 |
| Acquisition and main/control wakes | 9 | 53.000 | 49.444 |
| Main/control wake only | 136 | 37.963 | 34.846 |
| No observed wake | 9 | 29.889 | 26.889 |

The combined handlers' extra time lies inside the critical section, rather than
in a long attempt to acquire it. This is consistent with the inspected device
callback preceding ordinary DMA completion processing. The probes do not split
every internal core operation or prove that all of the additional time is
exclusively serial processing.

### A captured short interval follows a change in handler work

Qualified raw contexts **6203/6204**, frames **1349872/1349873**, overlap the
wrapper ring. The following values are intervals between successive endpoints,
except for the final current-frame rearm column:

| Frame | IRQ-entry interval, us | `atime` interval, us | SDK-return interval, us | Write interval, us | Rearm, us |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1349872, late | 526 | 526 | 524.701 | 524.841 | 7.525 |
| 1349873, early | 500 | 479 | 478.974 | 477.862 | 10.510 |

For the early interval, the handler changes from **53 to 32 us**, and its
software-stamp offset from IRQ entry changes from approximately **33 to 12 us**.
That 21 us reduction explains a 479 us `atime` interval even though the IRQ
entries are 500 us apart. Each stamp is projected into MONOTONIC using that
frame's tightly bracketed publication clock offset, with the additional
**1 us truncation** of the driver's CLOCK_REALTIME stamp retained as uncertainty.

The preceding late interval has **526 us between IRQ entries** and almost no
change in handler duration: its preceding handler was also combined, at 54 us.
Consequently, the late interval itself is not explained solely by extra work in
its current handler. The four-frame neighborhood illustrates the distinction
(times are MONOTONIC seconds relative to `1811716`):

| IRQ entry | Handler, us | Acquisition wake | Interval between acquisition wakes, us |
| ---: | ---: | ---: | ---: |
| .512106 | 32 | .512137 | — |
| .512580 | 54 | .512632 | 495 |
| .513106 | 53 | .513158 | 526 |
| .513606 | 32 | .513637 | 479 |

The shared IRQ can enter for device/control work before frame completion is
processed. Its entry is therefore not a dedicated physical frame-arrival
marker. This capture identifies software contributions to stamp/SDK timing;
it does not locate exposure, card completion or interrupt assertion in hardware,
nor establish that every excursion must be late then early.

In the same six-frame raw neighborhood, frame **1349871** has a **40.957 us**
rearm, including a **34 us** acquisition bracket overlapping the remote
main/control IRQ's same-lock hold. Frame **1349874** has **24.275 us** rearm
with an **18 us** acquisition bracket. All six current-frame rearms begin
after publication, so they cannot delay that frame's already completed write.
The six prepublication timeout queries take only **0.230–0.521 us**.

### Native comparison, probe cost and telemetry coverage

Native tracing also sees longer combined interrupts: seven average **50.714 us**,
versus **30.494 us** for acquisition-only handlers. Thus the association between
device/control work and longer completion handling exists without wrapper
probes. All retained camera switch-outs in both runs are sleeping (`S`), with
no scheduled preemption explaining the long camera runs.

| Retained kernel measurement | Native | Wrappers |
| --- | ---: | ---: |
| Acquisition-only IRQ mean, us | 30.494 | 32.183 |
| Main/control-only IRQ mean, us | 36.399 | 37.963 |
| Combined IRQ mean, us | 50.714 | 53.000 |
| Acquisition `sched_waking` to on-CPU mean / max, us | 1.618 / 3 | 2.522 / 7 |
| Camera scheduled-run mean / max, us | 42.574 / 80 | 44.115 / 81 |
| Acquisition-wake interval deviation, us | 0.901 | 1.268 |

The differences are consistent with measurable probe overhead. Runs are
sequential, roughly 84 seconds apart, with different buffer allocation and
changing camera/conversion conditions; these numbers do not isolate an exact
causal overhead budget. Selected same-event lock overlap remains direct evidence
despite those comparison limits.

Each camera telemetry file contains **29 disjoint two-second summary windows**,
with **115,971** interval samples. Pooling their counts/means/variances gives:

| Retained summary population | Native file | Wrapper file | Helper-free file, last |
| --- | ---: | ---: | ---: |
| `atime` interval deviation, us | 0.384 | 0.435 | 0.529 |
| SDK-return interval deviation, us | 0.616 | 1.035 | 0.543 |
| Publication interval deviation, us | 0.914 | 1.821 | 0.829 |
| Acquisition-to-write mean, us | 48.980 | 49.549 | 48.974 |
| Conversion mean / deviation, us | 26.672 / 0.523 | 26.977 / 1.024 | 26.666 / 0.524 |

These are pooled retained populations, not continuous full-file statistics or
one-second trace-only statistics. The approximately 2.05-second update cadence
leaves gaps between summary windows, and no two-second window fits wholly inside
a one-second ring capture. In particular, **both raw contexts 6203/6204 are in
summary gaps**. This explains why the raw 526 us `atime` interval and 40.957 us
rearm can exceed the wrapper summary maxima of 520 us and 25.909 us.

The wrapper file already has **2.758 us** publication deviation across seven
complete windows **before helper/probe activation**, alongside **1.413 us**
conversion deviation. Eighteen complete post-helper windows fall to
**0.936 us** publication and **0.509 us** conversion deviation. It is incorrect
to attribute the whole wrapper-file increase to probes or to cooling from these
data alone. The power-cycle/cooling history is relevant context for a repeat at
settled temperature.

There are no new event/control drops, failures, duplicate frames, frame/counter
gaps or invalid records between first and last snapshots of any file. The raw
context selection-drop counter remains at its historical **4,752**; this is not
new trace loss. Native telemetry adds 21 wide clock brackets, producing 42
ambiguous adjacent intervals. Wrapper/helper-free files add none; an additional
53 wide brackets arise in the unrecorded gap before the wrapper file, so they
cannot be assigned to its active probe interval.

### Remaining unexplained case and next focused experiment

Before the native helper starts, raw contexts **6180/6181** at approximately
**15:56:19.555 UTC** show **612.756 / 386.941 us** publication intervals and
**612.747 / 387.351 us** SDK-return intervals, while `atime` intervals remain
**500 / 501 us**. This approximately 113 us excursion occurs after the driver
stamp and before SDK return. There is no kernel trace for it, so this capture
cannot assign it to lock contention, IRQ tail work, thread scheduling or a
particular SDK request. The captured 21 us mechanism does not explain every
remaining excursion.

The next proposed comparison is routine serial status polling, after temperature
has settled. The current [application logic](../../../apps/ocam2KCtrl/ocam2KCtrl.hpp)
calls `getTemps()`, `getFPS()` and `getEMGain()` during READY/OPERATING passes;
`getTemps()` and `getEMGain()` issue `temp` and `gain` serial transactions.
`getFPS()` queries `fps` only with synchronization off; these captures have
synchronization on, so it uses the synchronization rate without a serial query.
A controlled change in status-query cadence, with query identities/times retained, can test
whether main/control wake bursts, combined handlers and rearm waits change
together. Existing observations do not identify which of these commands causes
each burst or establish a safe replacement polling policy. Cooling/protection
requirements must inform any such implementation. No serial polling, camera
settings, kernel, or syncopation scheduling behavior is changed by this analysis.

Analysis artifacts remain in `/tmp`: `camwfs-step5-155627-audit.json`,
`analyze_step5_camera.py`, `step5-camera-parsed.json`,
`step5-camera-summary.json`, `analyze_step5_kernel_compare.py`,
`step5-kernel-compare-summary.json`, `analyze_step5_edt_pairs.py`,
`correlate_step5_frames.py`, and `step5-edt-pairing/` with paired cycles,
IRQ classes, correlated raw frames and selected timelines. The source hashes
above identify the durable input evidence. All analysis is local; no instrument
commands or probes are executed by agents.

Independent review of input identity, pairing, source paths and interpretation
passes. Local Markdown link targets and `git diff --check` also pass. This
checkpoint changes only the two engineering records.

## 2026-10-25 proposed housekeeping pause experiment

The operator asks whether to add a testing mode that pauses all housekeeping
serial commands, and whether housekeeping ultimately needs scheduling to avoid
acquisition. A finite pause of automatic status polling is the recommended
first experiment. This section is a proposal, not an implemented or qualified
operating mode.

### Proposed diagnostic behavior

- Provide an ordinary-thread, runtime INDI request with a bounded duration,
  defaulting to normal polling after startup. Accept it only in a stable
  READY/OPERATING state. Do not persist a paused state across restart.
- Suppress automatic `temp` and `gain` transactions, plus `fps` polling when
  synchronization is off. Preserve synchronization-rate updates that require
  no serial I/O. Acquisition, frame timing, telemetry, property servicing and
  FSM checks continue normally.
- Separate requested/pending from active. Begin the quiet window only after
  any in-flight serial transaction and its trailing-byte drain finish. Expose
  actual activation, monotonic deadline/remaining duration, skipped-poll
  counters and ages of the last successful temperature/gain measurements.
- Keep the last readbacks but explicitly mark them stale. In particular,
  `getEMGain()` recognizes an `HV` response and reports an EM protection trip
  before entering ERROR. Suppressing it delays that software recognition;
  the mode must report that protection polling is paused. This source does
  not establish the independent hardware protection behavior.
- Automatically resume on expiry or explicit resume, without accumulating
  missed polls into a catch-up burst. Power/reconnect/reconfiguration/error
  transitions clear the diagnostic pause. Explicit operator commands remain
  available and end the recorded quiet window before issuing serial I/O.
  Keep protection-reset confirmation expiry and other nonserial logic active.

Use normal / paused / normal captures at settled temperature with the same
camera, IRQ/CPU placement and workload. Retain actual pause boundaries and the
existing camera/IRQ/lock measurements. A command already in flight can produce
late effects, so distinguish the activation edge from the interior quiet
interval. Verify that automatic serial transaction counts stop there and
resume afterward. If suppressing all polling helps, separate temperature and
gain queries in a subsequent experiment to locate the responsible traffic.

The existing control timing channel records transaction start/end and success,
but no command identity. Add a compact query category on the ordinary control
path if individual attribution is needed; reuse existing timing collection
rather than introducing another frame-thread polling mechanism.

### Why command phase alone is insufficient

Issuing a command just after publication can control when the request begins.
It does not control the camera's response-byte timing or the resulting shared
EDT interrupts. `pdvSerialWriteRead()` performs command, response waits/reads
and a trailing-byte drain; the whole exchange can span many 500 us frame
periods. Its elapsed wall time is not a continuous driver-lock hold. Moving
that blocking exchange into `frameGrabberPostPublish()` would put waits back
on the acquisition worker and is not the proposed approach.

First establish the effect of removing routine serial traffic. Then measure
per-query reply/interrupt timing to evaluate lower polling frequency, staggered
queries, or phase-aware request dispatch on an ordinary control thread. A
guaranteed quiet acquisition interval requires a measured bound or control over
the interrupt-producing reply path; merely choosing a request phase does not
supply that guarantee. Any production policy must retain timely temperature
and protection status. No code or instrument configuration changes are made
for this proposal.

## 2026-10-25 bounded housekeeping pause implemented

The operator approved implementation with a **120-second timeout** to allow
collection of a 60-second telemetry set. The operator also reports that EM
protection trips are usually silent and rarely recognized by current polling;
this is an operational observation, not a newly verified hardware guarantee.
The diagnostic explicitly reports software protection polling as paused.

The implementation remains local to `ocam2KCtrl`; the generic EDT helper,
logger schemas, normal housekeeping cadence and acquisition/publication hot
paths are unchanged. No configuration setting enables the pause at startup.

| INDI property | Implemented behavior |
| --- | --- |
| `housekeeping_pause.toggle` | On activates a finite pause in powered READY/OPERATING with no pending reconfiguration; Off resumes. Repeating On while active retains the original deadline. |
| `housekeeping` | Read-only `timeoutSec=120`, `remainingSec`, `tempAgeSec`, `gainAgeSec`, `fpsAgeSec`, cumulative `skippedTemp`, `skippedGain`, `skippedFps`, `serialTransactions`, and last `pauseStartMonoSec`/`pauseEndMonoSec`. |
| `housekeeping_status` | Read-only state/reason, temperature/gain/FPS validity, and `protectionPolling`; state is `paused`, `polling` or `inactive`. Readbacks are `current`, `stale` or `unavailable`; synchronized FPS is `external sync`. |

The callback follows existing **INDI mutex then recursive camera mutex** order.
It acquires serial ownership before sampling the monotonic start time, so an
in-flight command and trailing-byte drain finish before the 120-second window
begins. Its synchronous acknowledgment represents activation rather than a
queued request. Start/end logs include monotonic nanosecond boundaries; INDI
retains those boundaries in seconds for comparison with existing timing records.
Expiry is serviced by the ordinary app loop, and each poll decision checks it.
No dedicated thread or timer is added to the frame path.

Routine polls are grouped under the existing control locks. During a pause,
`temp` and `gain` are skipped, and serial `fps` is skipped when synchronization
is off. With synchronization on, the existing nonserial FPS update still runs.
Protection-reset confirmation expiry, INDI servicing, frame timing and camera
telemetry continue. Counters count suppressed app-loop poll opportunities;
resuming does not replay them as a burst. All existing OCAM serial call sites
now route through an app wrapper that ends an active quiet interval before
sending explicit commands, and counts serial attempts including failures.
Power-off, reconfiguration, acquisition recovery, sync-frequency reconfiguration,
shutdown and invalid FSM states cancel the mode. READY still returns promptly
when another thread owns the INDI mutex.

Successful parse times determine readback ages. No prior valid reply, a failed
or malformed reply, and power-off report age **−1**. Cached pre-pause readings
remain stale after resume until fresh replies arrive; a synchronous FPS source
is identified separately. Read-only updates are batched per property to avoid
sending one message for every field. The original temperature/gain values are
retained during quiet operation, with the dedicated validity fields exposing
that they are cached.

The existing cooling-Off guard depends on measured temperature. If its cached
temperature predates the pause, the control now queries a fresh temperature
before applying that guard. A failed/invalid refresh rejects the request;
a newly cold reply cannot be overridden by a cached warm value. The refresh
itself is an explicit command and ends the diagnostic quiet interval.

The [operator guide](../../../apps/ocam2KCtrl/doc/camera-timing-trace.md)
section **7** gives a helper-free normal / paused / normal comparison after
cooling settles, each with 60 seconds of telemetry. It includes readback/counter
checks, a settling interval after activation, explicit resume and optional
matched short kernel captures. Rebuild/install the camera app and restart it
using the operator's normal procedure; no `logdump` rebuild or new camera
configuration is needed. Hardware qualification is still outstanding.

Local verification passes, totaling **1,026 assertions in 49 test cases**:

| Suite | Assertions | Test cases |
| --- | ---: | ---: |
| `ocam2KCtrl_housekeeping_test` | 237 | 12 |
| `ocam2KCtrl_test` | 713 | 33 |
| `ocam2KCtrl_lifecycle_test` | 7 | 1 |
| `ocam2KCtrl_temperature_test` | 69 | 3 |

The focused suite covers synchronized/unsynchronized suppression, original
query order on resume, exact deadline/nonextension, callback serialization
behind serial ownership, real app-loop telemetry during a quiet interval,
invalid states, explicit command cancellation, readback freshness, cooling-Off
validation after active/resumed/expired pauses, and ordinary protection errors.
Tests use existing SDK stubs; no actual camera is opened. Independent source
review verifies lock order, routing of all existing OCAM serial calls and the
unchanged frame hot paths. `clang-format --dry-run --Werror`, local Markdown
link checks and `git diff --check` pass.

The normal application also builds and links against the complete copied
**EDT 5.6.8.0 SDK**, with no test includes/stubs:

```bash
make -C apps/ocam2KCtrl EDT=true \
    EDT_INCLUDES=-I/home/jrmales/Source/opt/EDTpdv \
    'EDT_LIBS=-L/home/jrmales/Source/opt/EDTpdv -lpdv -lpthread -lm -ldl' -j2
```

The SDK-linked binary is never launched by an agent. Build diagnostics contain
only the existing modified/untracked git-header notices. Logs are retained in
`/tmp/ocam-housekeeping-{base-build,base-test,sdk-build}.log` and
`/tmp/ocam-{housekeeping,lifecycle,temperature}-{build,tests}.txt`. Instrument
pause/expiry and normal / paused / normal timing qualification remain operator
follow-ups.

## 2026-10-25 closed-loop normal / paused / normal operator comparison

### Conditions and pause qualification

The operator supplied `/home/jrmales/tmp/camwfs-housekeeping` and confirms
that the loop was **closed throughout all three captures**, with the same
loop gain, camera settings, CPU/IRQ placement and workload. Treat this as its
own comparison, rather than pooling it with earlier open-loop captures.

All files report timing session **5**, stream inode **3383**, bin2 / SDK mode
**3**, software binning **1x1**, **2000 Hz**, synchronization **ON**, EM gain
**595**, and shutter **OPEN**. Normal raw records identify PID **3316955** and
framegrabber TID **3316957**. The paused file has no raw excursion records from
which to read PID/TID, but there is no session/counter reset. Normal temperature
readbacks are near −45.1/−45.2 C and on target. The paused −45.4 C value is
explicitly stale; it is not a fresh temperature measurement throughout the pause.

| Input | SHA-256 |
| --- | --- |
| `camwfs-housekeeping-normal-before.txt` | `de05326db3e4237468e1bfefdfba9dc1143e5f358d974f44151673d0c1eefd35` |
| `camwfs-housekeeping-paused.txt` | `f8dee9cefbf08af30e261003f434fdc876c52ccff461f82a041d6898555a3919` |
| `camwfs-housekeeping-normal-after.txt` | `48855bd5962c96edb433da4b4f5d6f62efb49bdcaafdd4705cccc34b41436475` |
| `camwfs-housekeeping-paused-indi.txt` | `9b85060dfb098ccff9a22472ef7f40d54b8cf366598a19b5512fa9d6975e0f0f` |
| `camwfs-housekeeping-normal-after-indi.txt` | `f0cc76d466242be0da49c72ca92960292bfeb1a4e0bc9353d86b5bd6d8adf9dd` |

The two paused INDI snapshots retain **serialTransactions=981**, with remaining
time **102.939 / 25.666 s**. Both report `paused`, stale temperature/gain,
paused protection polling and external-sync FPS. Skipped temperature/gain
counts advance **256 to 333**; those totals already include earlier suppressed
polls, rather than measuring this pause from zero. All paused timing summaries
also retain **controls=981**, zero serial overlap and no serial-duration samples.

The actual monotonic pause start is **1817966.45489468 s**, and automatic resume
is **1818086.88048839 s**, with reason `120 second timeout`. It lasts
**120.42559371 s**: the ordinary loop services expiry approximately **0.426 s**
after the deadline. Previous `pauseEndMonoSec=1817946.90099174` is historical,
not the active deadline. Subsequent INDI snapshots show normal polling, fresh
readbacks, serial attempts **1053 to 1123**, and unchanged skipped-poll totals.

A common CLOCK_REALTIME-minus-MONOTONIC offset interval from all **77**
deduplicated normal raw frames maps activation to approximately
**17:42:07.864770 UTC** and resume to **17:44:08.290363 UTC**, October 25
(**10:42:07.864770 / 10:44:08.290363 America/Phoenix**). All 60 paused summary
windows lie safely inside that quiet interval: the first window begins roughly
30.1 s after activation and the last ends roughly 29.1 s before resume. These
windows therefore do not contain the transition edges or timeout recovery.

### Populations and measured changes

Normal telemetry updates approximately every **2.056 s**, while paused updates
arrive approximately every **1.004 s**, because serial waits are absent from
the ordinary loop. Each record still summarizes a **two-second** population.
The 60 paused records overlap; pooling them all would count frames repeatedly.
Use all **29 disjoint windows** from each normal file and alternating paused
records (**30 disjoint windows**). Their sample counts are **115,971 / 119,970 /
115,971 intervals**, or **116,000 / 120,000 / 116,000 per-frame durations**.
The complementary alternating paused selection gives essentially the same result.

| Metric, us unless stated | Normal before | Paused | Normal after |
| --- | ---: | ---: | ---: |
| `atime` interval deviation | 0.424 | 0.262 | 0.409 |
| `atime` interval range, all retained summaries | 481–525 | **499–501** | 480–515 |
| SDK-return interval deviation | 0.464 | 0.291 | 0.434 |
| SDK-return interval range, all retained summaries | 480.467–525.092 | 495.806–504.263 | 480.177–514.812 |
| Publication interval deviation | 0.761 | **1.428** | 0.747 |
| Publication interval range, all retained summaries | 480.157–525.112 | 487.621–512.087 | 479.306–515.634 |
| Acquisition-to-write mean / deviation | 48.625 / 0.579 | 48.835 / 1.004 | 48.615 / 0.588 |
| Conversion mean / deviation | 26.267 / 0.456 | 26.507 / 0.941 | 26.255 / 0.477 |
| Rearm maximum, all retained summaries | 27.151 | **9.809** | 31.019 |
| Serial-duration samples in selected windows, count | 58 | **0** | 58 |

Deviations are pooled with sample counts and between-window mean changes.
These remain retained-summary statistics, not a continuous full-run bound.
In particular, the normal files have small gaps between two-second windows.
The raw acquisition/publication/SDK distributions are accumulated independently
of clock-bracket qualification, so the table does not remove the paused file's
wide-bracket frames from its aggregate timing distributions.

The normal-before file emits **12 qualified complete raw contexts** (6 late,
6 early); normal-after emits **5** (2 late, 3 early). Every center overlaps a
completed serial transaction. Driver `atime` and SDK-return intervals track
within approximately 1 us; changes in conversion duration are too small to
explain these >20 us excursions. Each retained raw late event is followed by
a publication interval approximately **479.275–480.858 us**.

No raw contexts are emitted while paused, and all 60 paused summaries report
zero late/early threshold crossings and serial overlap. Normal summary windows
contain only **2 of the 12** before-context centers and **1 of the 5**
after-context centers; the remaining **10 / 4** lie in summary gaps. Do not
substitute summary crossing counts for raw/full-stream excursion counts.
The historical raw-selection drop count remains **2881** in both normal files,
with no new incomplete contexts, unclassified records or discontinuities.

### Separate changes within the paused capture

The first **21 disjoint paused windows**, sequences **758–798**, have
publication deviation **0.656 us**, conversion **26.221 +/- 0.446 us**, and
SDK-return interval deviation **0.299 us**. Later the capture changes despite
unchanged operator settings and zero serial traffic:

- Rearm spread first increases in summary rows around **17:43:10–20 UTC**,
  reaching 9.809 us maximum, then recovers.
- Main-notification spread rises around **17:43:18–26 UTC**, reaching 9.177 us
  maximum. There are **61 new wide write-clock brackets** around
  **17:43:22–26 UTC**, with bracket maximum **3.897 us**.
- Conversion spread then increases and stays elevated through the end of the
  paused file. The final **six disjoint windows**, sequences **806–816**, have
  conversion **27.474 +/- 1.385 us** and publication interval deviation
  **2.700 us**, while `atime` and SDK-return interval deviations remain
  **0.246 / 0.270 us**. Main-post and rearm spread have already recovered.

These are post hoc segments describing nonstationarity, not separate controlled
experiments. Two-second rolling summaries blur exact change boundaries. The
extra late publication spread is localized after SDK return, predominantly to
conversion; this telemetry does not identify whether scheduling, memory/cache
behavior or another implementation detail causes it. It is present without
routine serial traffic and must be investigated separately. Consequently, this
run supports removal of the large serial-associated tails, **not an overall
publication-RMS improvement** from pausing.

The 61 wide brackets yield **122 ambiguous adjacent intervals** across the
selected disjoint windows. Qualification affects direction attribution/raw
selection, not the raw distributions above. There are no new failed/rejected
frames, duplicates, frame/counter gaps, invalid events, event/control drops or
control failures between first and last summaries in any file. Control totals
increase by **56** in each normal file and **zero** in the paused file.

### Interpretation of the shared interrupt

The operator asks whether serial IRQs delay the frame-ready IRQ. The precise
conclusion combines the preceding accepted-wrapper trace with this comparison:
serial/device events and frame completion share **EDT IRQ 405**. Within a
combined invocation, the device/serial callback precedes ordinary DMA completion
processing, including the software completion stamp and frame-thread wakeup:

```text
EDT IRQ 405
    acquire shared driver lock
    process device/serial work, when present
    process DMA completion and sample atime
    wake the acquisition thread
    release driver lock and return
```

The preceding trace measures approximately **20–21 us** extra work in combined
handlers, and the short following interval reflects returning to the ordinary
stamp offset. Thus serial activity can delay **servicing/reporting frame
completion**, as well as contend with later rearm. These measurements do not
locate physical frame arrival or hardware interrupt assertion. They should not
be described as proof that the camera itself exposes late or that two separate
hardware IRQ lines were timed against one another.

The unchanged closed-loop normal / paused / normal comparison strongly supports
routine housekeeping traffic as a source of the recurring large driver-stamp
excursions. It does not identify `temp` versus `gain` individually or establish
a phase that keeps asynchronous reply interrupts clear of completion. Preserve
that distinction when designing a production housekeeping schedule. The
remaining post-SDK conversion spread can be investigated with a short matched
kernel trace during paused operation, retaining the same closed-loop workload.
No new implementation or instrument configuration changes accompany this analysis.

Reproducible local artifacts are `/tmp/housekeeping-capture-audit.json`,
`/tmp/housekeeping_capture_audit.py`, `/tmp/analyze_housekeeping_root.py`,
`/tmp/housekeeping-root-{data,summary}.json`, and
`/tmp/camwfs-housekeeping-metrics/` with the independent summary and paused
segments. `/tmp/plot_housekeeping_compare.py` generates
`/tmp/camwfs-housekeeping-comparison.png` and `.svg` from the parsed records.
The plot shows every rolling summary without treating its points as independent
samples. All analysis is local; agents run no instrument commands.

## Production housekeeping policy implementation — 2026-10-25

The operator approved and the local implementation now follows the separate
[housekeeping policy](ocam2k-housekeeping.md): confirm gain/FPS once per request,
use adaptive configurable temperature polling (10 s idle / 1 s changing), and
service at most one automatically scheduled serial query per ordinary loop.
Fresh cooling-Off checks and the bounded diagnostic pause remain available.
The policy record documents response/inference limitations, exact configuration
and INDI behavior, and passing local verification (2,037 assertions / 67 cases
plus an EDT SDK build). Hardware comparison is pending; this does not add a
new measured claim to the captures above or resolve the separate conversion
variation.

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
- `apps/ocam2KCtrl/scripts/check-edt-probes.py`
- `apps/ocam2KCtrl/scripts/tests/capture_camera_timing_test.py`
- `apps/ocam2KCtrl/scripts/tests/check_edt_probes_test.py`
- `apps/ocam2KCtrl/tests/ocam2KCtrl_test.cpp`
- `apps/ocam2KCtrl/tests/ocam2KCtrl_housekeeping_test.cpp`
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
