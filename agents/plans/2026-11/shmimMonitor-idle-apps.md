# Idle NN, PO4AO and collection lifecycle

Date: 2026-11-08. Status: implementation and workstation verification; isolated Linux/GPU and operator acceptance remain required. No deployment or instrument performance claim.

## Baseline and authorization

Implementation branch: `rlandman/po4ao-idle-lifecycle`, created directly from verified `rlandman/nn-po4ao-gui` commit `a84232b938f359fdebb2973b6a1e367e0343002a`, in `/Users/ricolandman/.codex/worktrees/po4ao-idle-lifecycle/MagAOX`. The combined checkout and its existing GUI, saved policy pairs, NN/PO4AO controls and collector improvements are preserved. Only this planning document was copied from the review checkout; no NN implementation was copied, merged or cherry-picked from `nnreconstructor-speedups`.

All implementation, builds and tests run on the workstation. The earlier implementation used separately authorized read-only exao2 source/readback inspection. That authorization does not apply to this review: no instrument SSH, INDI connection/tunnel, control utility, deployment or hardware operation is permitted. This review recovered the earlier source evidence from a local session record only. Instrument acceptance below is operator-run.

The final scope decision supersedes the original gain-restoration proposal: “complete routing, do not do anything with gains/leakage, also no note”. There is no gain/leakage snapshot, restoration, verification wait or GUI reminder. Existing utility side effects are unchanged. No hogainctrl interlock or changes outside the user's apps for instrument controls were added.

## Implemented lifecycle

Use `shmimMonitor`'s existing READY/OPERATING gate, following `shmimIntegrator`. There is no independent monitor pause flag, condition variable or subscription framework.

- Integration decision (2026-11-08): the user explicitly approved “Yes—use upstream READY behavior” for NN Disable, PO4AO whole-app Stop and collector Stop/completion. `shmimMonitor.hpp`, `MagAOXApp.hpp` and the shared FSM tests now match `dev` exactly. The previous strict mapping-closed acknowledgement, attachment cancellation changes and atomic FSM changes are removed. App workers communicate intent to appLogic, which requests READY/OPERATING on the app thread.
- NN disabled startup and completed Disable settle in READY. Enable recreates GPU resources from retained validated engine bytes, requests attachment, validates it and admits fresh frames. Disable closes inference admission, drains in-flight work, acknowledges its final zero, and then destroys owned TensorRT context/engine/runtime and CUDA buffers outside the callback. No `cudaDeviceReset`. CPU output buffers remain available for final publication. A cancelled/late loader cannot reinstall resources.
- `model_status.loaded` identifies the host-prepared model, `resident` reports GPU residency, and `active` still requires published inference. Compound NN/CACAO activation prepares GPU resources before stopping a usable CACAO writer. Failure does not silently substitute a writer.
- Collector startup is READY. Start queues a bounded five-second attachment/observing-readback stage before the existing session preflight; Stop cancels it. Terminal/draining session phases park the input monitor. An unexpected reconnect during an active session still aborts the partial run. Existing bank, metadata and output-ownership checks remain in place.

READY is asynchronous. A stopped producer can delay cleanup until the existing semaphore timeout, and the idle outer loop can delay reattachment. No fixed sleep is treated as acknowledgement. Completion acknowledges drained app work and a READY request, not mapping closure. Rapid READY/OPERATING transitions can be coalesced by upstream; attachment cancellation and missing-source cleanup retain upstream behavior. NN revalidates a retained attachment on its monitor thread and still requires a fresh frame. Collector rapid restart can time out waiting for a new attachment; retry after upstream cleanup. Do not infer detachment from an Off/Stopped readback.

## Whole-app PO4AO routing

Native `po4ao.run_enable.toggle` is separate from `control_enable.toggle` (policy Enable/Disable). `run_status` publishes phase, observed route, error, operation owner and verified status. The GUI uses explicit Start/Stop targets and blocks competing runtime operations during a transition. Policy selection is retained; learning never resumes automatically. The optional trainer pauses an existing training worker on entry to routing and rejects new learning/recollection while stopped. Saved policy selection and saving remain available while stopped.

A single non-real-time C++ control worker owns an operation with a 90-second deadline. Startup retains the prior OPERATING behavior with an unverified route: the app must not assume downstream has stopped using its output after an app restart. Output channel must be `po4ao_output` for this installed routing adapter. NN reservation subscriptions use device `nnReconstructor`.

Start sequence:

1. Read current route/calibration and require an already operator-open HOWFS loop.
2. Attach and observe fresh PO4AO output before changing the downstream input.
3. Reserve NN/CACAO writer ownership through NN's native INDI reservation.
4. With NN reserved, capture the current verified WFS writer process identity, then run `howfs_mfilt_switch po4ao_output` in the verified HOWFS directory.
5. Run `cacao-calib-apply default`, check its fresh completion marker and resolved calibration identity, and await a verified running WFS writer with a different PID/Linux-start-tick identity captured after reservation and before the first utility. A stopped writer, the old process, or changed FPS metadata alone cannot complete this stage.
6. Restore the actual pre-request NN/CACAO writer using its existing handover and first-result checks. The reservation does not alter the operator's modulation preference or request modulation changes.
7. Observe fresh output again after restarts and advancing mfilt consumption on that exact stream inode. Acknowledge Running.

Stop follows the same reservation/switch/calibration/writer-verification sequence with `aol1_modevalWFS`. Only after downstream consumption is verified does it close callback admission, drain active callback work, request READY through appLogic and acknowledge Stopped. It interrupts an unfinished warmup collection after callback quiescence. Already completed saved policies are retained.

The app does not open or close the loop. Calibration's existing procedure opens it, hence the preflight requirement that the operator already has it open. Loop closure remains an operator action.

NN exposes `routing_hold` text elements `owner` and `action` (`hold`/`release`) with `routing_status` owner/source/phase/instance/sequence/released_owner readbacks. It rejects competing native Enable/model/handover/modulation controls while held. Hold snapshots the verified actual writer; it acknowledges held only after NN is fully disabled when needed. Calibration restarts CACAO even when NN was active, so release uses the standard handover to stop CACAO before enabling NN. No external gain-control interlock is implied.

Failure behavior:

- Failed Stop never parks the monitor while the downstream route may depend on PO4AO. Report actual or unknown route, failed stage, error and unverified status.
- Commands use fixed executable/argument pairs and the required working directory. Timeout/cancellation kills and reaps the utility process group. FIFO requests already queued can still complete; do not automatically roll back or release an unresolved writer reservation.
- Preserve the operation token, captured route/calibration/writer and reserved NN instance after failure. New Start/Stop requests receive Alert without replacing this evidence until a fresh `released_owner` equal to that exact token arrives from the same NN instance, with phase `released` and no current owner. A stale release, a different token, a restarted NN or restoration still in progress cannot unblock retry.
- Operator recovery: keep downstream supplied and stop competing commands; independently settle all outstanding utility/FIFO work and check actual routing/writer state. If NN holds the matching token, submit `routing_hold` with that retained `owner` and `action=release`, then verify completed restoration. If hold delivery was uncertain and NN has no owner, first settle pending delivery, acquire `hold` using the *same retained token*, wait for `held`, then release it after the same external checks. Do not issue a new token to bypass recovery. A different NN instance/owner requires independent reconciliation; automatic retry remains blocked. Do not blindly restart either app to hide a reservation.
- Every rejected run request publishes Alert, including repeated identical rejection. Accepted requests publish Busy before launching the worker. Rejection diagnostics are separate from the retained operation error. Visible reset/reload conflicts gate the GUI; raced conflicts are acknowledged by the app. The existing unknown-on-timeout latch is unchanged.
- A repeated verified target checks route and consumption without reapplying either utility. A changed or stale route fails verification.
- The reservation serializes controls in these apps; it cannot prevent a separate manual instrument command. Operators must not issue competing routing/writer commands during the procedure.

## Installed source evidence

Read-only inspection on 2026-11-08 established:

- `/usr/local/bin/howfs_mfilt_switch`, SHA256 `bf4c315b992a2b085b014957419c4dcbb814320fed3be1f51a211374e6608524`, queues RUN/CONF stops for mvalC2dm, mfilt, wfs2cmodeval and acquWFS to `/milk/shm/tweeter-vispyr_fpsCTRL.fifo`, changes the mfilt alias and restarts acquWFS. Exit is not consumption acknowledgement.
- `/usr/local/milk/bin/cacao-calib-apply` resolves to milk-1.03.00, SHA256 `7dd743af16c8c103b37f001d95e4891c1f676cf0ec0da73999df18e2fca88845`. Required cwd: `/opt/MagAOX/cacao/tweeter-vispyr/tweeter-vispyr-rootdir`. It sources the milk configuration/argument helpers, opens the loop, stops/restarts the four processes, runs pre/post hooks and writes calibration source/time records. Missing calibration can return zero; the adapter therefore verifies readbacks.
- Post-hook SHA256 `1a91404f4f39ce996bdd76dc64d50fb592518637315901670c1eee1edbb0996b`. The calibration-source file contains a full directory path; the loaded file contains a timestamp, not the word `default`. The default directory is resolved through `../tweeter-vispyr-calibs/default`.
- Installed FPS headers provide `runpid`, `mfilt-1.inmval` and `mfilt-1.loopON`. Process-info provides PID, ACTIVE loop status, trigger stream inode/ID/counter and loop counter. Completion requires advancing counters on the requested inode under the same process. Existing `cacaoWriter` verifies Linux PID/start-time/cmdline identity; no guessed PID from a process name is used.
- Milk ABI includes are isolated in separate compilation units to avoid the existing EDT BOOL collision. The local fake ABI does not establish installed binary layout compatibility.

## Typed telemetry

PO4AO now uses `dev::telemeter` and the TELEMETER lifecycle conventions. The previous periodic INFO latency summary is removed; INDI `runtime_stats` is preserved.

Event `TELEM_PO4AO` (20908) records callback p50/p95/p99/max in microseconds, valid sample count, sequence and monotonic sample time, freshness and mixed-controller-window flags, exact uint64 cumulative missed-frame/deadline/inference/replay counters, deadline, model ID and backend (0 fallback, 1 CPU, 2 TensorRT). It is callback timing, not end-to-end optical/control latency. Empty windows carry zero samples; sub-microsecond samples are valid zeros; idle windows are marked stale; transition windows are marked mixed and identify the newest included sample. The fixed atomic sample ring is filled by the callback; allocation, sorting and serialization run outside it.

## Workstation evidence and reproduction

Historical pre-integration evidence (the current integration evidence below supersedes shared-monitor/FSM claims):

Baseline tests were run before edits: NN 545 assertions/33 cases and CACAO writer 25/2; PO4AO 259/27. The final local checks include:

- NN production-method SDK-double lifecycle/ownership/handover suite (610 assertions/37 cases), CACAO process-identity tests (25/2) and Makefile header-isolation regression (1 test). Address/undefined-behavior sanitizers pass. These validate ownership logic, not CUDA/TensorRT ABI or measured VRAM release.
- PO4AO production callback/control/worker tests plus routing sequence failure injection and telemetry window tests (367 assertions/31 cases); address/undefined-behavior sanitizers pass.
- Native adapter (17 assertions/4 cases) compiled against an explicitly fake ABI and private fixed utility paths: actual child execution/cwd, failed/zero-exit calibration, timeout, closed-loop rejection, truncated metadata, wrong inode and stalled/advancing consumption.
- Generated production FlatBuffers schema round trips (76 assertions/1 case) preserve exact large counters/model IDs, empty/idle/mixed windows and reject corrupt buffers. Normal logger code generation succeeds; generated files remain build products.
- The actual production monitor loop (40 assertions/2 cases) with a synthetic producer, real named POSIX semaphore and ImageStreamIO mapping double confirms no additional semaphore waits or frame callbacks while READY, reader release, restart and early-exit cleanup. Darwin uses bounded semaphore polling inside the wait wrapper; this is not a Linux timing/performance test.
- Existing collector session/observation suite: 216 assertions/16 cases. Collector app-level INDI/real-stream acceptance is still required.
- Qt6 widget suite: 426 assertions/27 cases, offscreen and no socket; existing policy/NN controls remain covered.
- Trainer suite: 134 tests and 17 subtests pass. One restricted run failed because macOS denied child-process enumeration; rerunning with local process-inspection access passed. Existing plotting/deprecation warnings remain.
- Shared FSM concurrent refresh check compiles the actual method bodies with isolated logging/INDI boundaries and address/undefined-behavior sanitizers. The new regression in the normal MagAOXApp suite requires the production dependencies and has not run here.

Typical commands from the feature worktree:

```sh
make -C apps/nnReconstructor/tests test
make -C apps/nnReconstructor/tests asan
make -C apps/po4ao/tests test routing
make -C apps/po4ao/tests telemetry FLATBUFFERS_CFLAGS=-I/opt/homebrew/include
make -C apps/po4ao/tests asan
make -C apps/po4ao/tests monitor
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=apps/po4aoTrainer:magaox-python /tmp/po4ao-dev/bin/python -m pytest -p no:cacheprovider -q apps/po4aoTrainer/tests
```

Schema tests require the normal logger `flatlogcodes`/`flatc` generation step first. The workstation Qt build uses `learningCtrl_test.pro`, Qt6 and locally compiled INDI libcommon/liblilxml archives. The existing collector test uses its real CPU session and observation sources with local INDI/OpenSSL dependencies. Detailed current assertion counts are in the completion report; tests do not establish instrument deployment success.

## Independent review follow-up (2026-11-08)

Starting checkout was clean on `rlandman/po4ao-idle-lifecycle` at reviewed HEAD `d9d37b2c`, with no subsequent changes. Comparison baseline remains `a84232b938f359fdebb2973b6a1e367e0343002a`. No implementation from `nnreconstructor-speedups` was accessed or used.

1. **Accept: silent Start/Stop rejection.** The production callback returned -1 during reset/reload without an INDI acknowledgement. The GUI allowed those clicks. `MagAOXApp::handleNewProperty` discards callback return values, and the GUI waits for Busy then Ok or Alert. Those generic conventions already existed at the baseline; this new run-control integration failed to honor them. Before the fix, the native regression failed all three conflict sections and the Qt regression failed both visible-conflict sections. Reusing explicit Busy/Alert publication and existing GUI acknowledgement is smaller than changing generic timeout semantics. Repeated Alert publication is deliberate: change-only publication could silently discard the next identical rejection.
2. **Accept: lost recovery identity.** The new run callback replaced the token/NN instance on every retry, even after a held reservation and partial utility failure. The prior plan promised retained identity but code did not enforce it. The pre-fix regression demonstrably accepted this retry. The corrected guard retains token/snapshot/error and uses one additive NN readback (`released_owner`) to associate completed writer restoration with the exact token. This uses the existing reservation/handover; it adds no second recovery controller or automatic release. Tests cover unresolved hold, wrong/stale release, NN restart, restoration in progress, successful matching recovery and missing calibration acknowledgement.
3. **Accept the missing boundary check; no claim of observed hardware dual-writing.** The independent NN probe demonstrates a possible boundary sequence, not installed behavior. The [locally recovered utility snapshots](shmimMonitor-idle-apps-evidence/README.md) have SHA-256 hashes matching the earlier inspected copies. The calibration script queues WFS runstart, sleeps, and writes the marker without awaiting execution. The input-switch script does not start this writer. Therefore a marker does not establish that the queued WFS start has settled. The old adapter regression returned successfully with a new marker and no restart. The fix awaits a different verified PID/start-tick identity before NN release; FPS identity/timestamp changes alone cannot satisfy it. The baseline identity is refreshed after NN reservation, immediately before the first utility, so an intervening native handover cannot be mistaken for the calibration restart. A new regression reproduced this edge before that adjustment. Existing NN handover then stops that writer before resuming NN. A timeout retains the reservation and leaves PO4AO available. The completion argument assumes the inspected single WFS restart and no competing external writer commands; installed dependencies/hooks still require operator verification.

The review probes were inspected as fault demonstrations, not copied as acceptance tests. New tests assert corrected behavior. An initial too-short adapter deadline produced an uninformative timeout pass; the retained regression uses a two-second deadline and exact restart-error assertion, and failed on the pre-fix adapter because no exception was thrown. Test harness initialization was also corrected to include the actual run property elements before testing explicit publications.

No gain/leakage behavior, policy selection, learning resume behavior, trainer control framework or shared GUI timeout semantics changed. Existing READY/OPERATING, GPU release and telemetry implementation is preserved.

### Follow-up validation

Final post-format runs on the workstation:

| Suite actually run for this follow-up | Result | Boundary |
| --- | --- | --- |
| PO4AO production-method/unit tests | 430 assertions / 35 cases passed | ASan + UBSan; SDK/INDI/monitor doubles |
| Production routing adapter | 24 assertions / 6 cases passed | ASan + UBSan; private fake executables, FPS and Linux-process records |
| NN production-method/unit tests | 614 assertions / 37 cases passed | ASan + UBSan; SDK/INDI/monitor doubles |
| Production CACAO helper | 29 assertions / 3 cases passed | ASan + UBSan; private FPS/proc/FIFO fixtures |
| CACAO build isolation regression | 1 unittest passed | Conflicting local header fixtures |
| Qt learningCtrl | 440 assertions / 28 cases passed | Offscreen Qt6; socket-free publisher |

Both app suites and adapter/helper were built with `-fsanitize=address,undefined -fno-omit-frame-pointer`. All touched C++ files passed the repository clang-format configuration and `git diff --check`. Qt emitted the existing font/offscreen and local macOS archive-version/override warnings, without test failures. The final run commands were `make -C apps/po4ao/tests` with `test routing` and sanitizer CXXFLAGS, `make -C apps/nnReconstructor/tests` with `test` and sanitizer CXXFLAGS, and the existing `/tmp/magaox-idle-gui` build followed by `QT_QPA_PLATFORM=offscreen ./learningCtrl_test`.

The earlier trainer, collector, monitor and telemetry results above are historical and were not rerun for these routing-boundary changes. Tests use private paths and explicitly fake utilities/SDK/ABI boundaries. No new real ImageStreamIO, GPU or instrument validation was performed. The result is ready for isolated Linux/GPU validation, not operator acceptance or production deployment.

Follow-up operator acceptance must additionally verify rejection acknowledgement during a reset/reload race; retained token and snapshot after failure; stale/wrong-token and restarted-NN recovery refusal; matching-token restoration; and delayed CACAO restart before NN release. First inject those failures in an isolated replica. Confirm no dual writer and no premature PO4AO detachment using actual readbacks; the workstation tests do not establish either on hardware.

### Files changed by this review follow-up

- `agents/plans/2026-11/shmimMonitor-idle-apps.md`
- `apps/nnReconstructor/README.md`
- `apps/nnReconstructor/cacaoWriter.cpp`
- `apps/nnReconstructor/cacaoWriter.hpp`
- `apps/nnReconstructor/nnReconstructor.hpp`
- `apps/nnReconstructor/reconstructorControls.hpp`
- `apps/nnReconstructor/tests/cacaoWriter_test.cpp`
- `apps/nnReconstructor/tests/nnReconstructor_test.cpp`
- `apps/nnReconstructor/tests/nnTestSupport.hpp`
- `apps/po4ao/README.md`
- `apps/po4ao/po4ao.hpp`
- `apps/po4ao/po4aoRouteIO.cpp`
- `apps/po4ao/po4aoRouteIO.hpp`
- `apps/po4ao/po4ao_run.hpp`
- `apps/po4ao/tests/Makefile`
- `apps/po4ao/tests/po4aoRouteTestAbi.hpp`
- `apps/po4ao/tests/po4ao_route_io_test.cpp`
- `apps/po4ao/tests/po4ao_run_test.cpp`
- `gui/widgets/learningCtrl/README.md`
- `gui/widgets/learningCtrl/learningCtrl.hpp`
- `gui/widgets/learningCtrl/tests/learningCtrl_test.cpp`
- `agents/plans/2026-11/shmimMonitor-idle-apps-evidence/README.md`
- `agents/plans/2026-11/shmimMonitor-idle-apps-evidence/cacao-calib-apply.source.txt`
- `agents/plans/2026-11/shmimMonitor-idle-apps-evidence/howfs_mfilt_switch.source.txt`

## Remaining acceptance gates

See the [operator acceptance and PR integration plan](shmimMonitor-idle-acceptance.md) for the ordered procedure. Integration against `dev` at `16138ab7b5b90bae028fb09aeb2c73e3f391844a` resolves seven conflict paths; upstream owns event 20906 (`telem_fgstage`) and 20907 (`telem_fgtrace`), so PO4AO uses 20908. The workstation tests do not establish instrument readiness. PR #418 is already included by ancestry and can be updated after integration/qualification.

1. On an isolated non-instrument Linux host, build libMagAOX and all affected C++ apps against the installed-version milk/ImageStreamIO/CUDA/TensorRT dependencies. Run the normal MagAOXApp regression suite. Verify telemetry logger registration/accessors and decode a real emitted TELEM_PO4AO record, including idle and model transitions.
2. With a real ImageStreamIO synthetic producer, exercise missing-source startup, cancellation during attachment/allocation, producer stalls, rapid Stop/Start and stream replacement. Check reader slots and actual wait/wakeup counts after READY while the producer continues. Validate collector pending Start/Stop and observe that idle sampling stops. The workstation mapping double alone does not close this gate.
3. On an isolated GPU host, measure enabled, completed-disabled and re-enabled NN resource usage across repeated cycles. Require owned buffers/objects to be gone only after final-zero acknowledgement and in-flight completion; retained CUDA context overhead is allowed. Inject allocation/load/cleanup failures and verify no late residency or duplicate NN/CACAO writer.
4. Operator-run instrument acceptance: verify the inspected utility/header versions and working directory, actual device/channel names and readback semantics. Begin with the HOWFS loop already open; independently verify writer ownership. Test both NN and CACAO starting owners, Stop to WFS, Start to fresh PO4AO, downstream inode/counter advancement, READY detachment, retained policy and no automatic learning restart. Close the loop only through the normal operator procedure after reviewing readbacks.
5. Operator-run failure acceptance: withhold a required readback or simulate a utility failure in an isolated replica first. Confirm failed Stop keeps output available, no false completion, owner token/partial route are visible and recovery does not race queued commands. No automatic recovery from external manual routing or unresolved FIFO work is claimed.

## Affected files

The following is the implementation inventory; generated artifacts and local build dependencies are not committed:

- `agents/plans/2026-11/shmimMonitor-idle-apps.md`
- `apps/dlDataCollection/README.md`
- `apps/dlDataCollection/dlDataCollection.hpp`
- `apps/nnReconstructor/README.md`
- `apps/nnReconstructor/nnReconstructor.hpp`
- `apps/nnReconstructor/reconstructorControls.hpp`
- `apps/nnReconstructor/tests/Makefile`
- `apps/nnReconstructor/tests/nnReconstructor_test.cpp`
- `apps/nnReconstructor/tests/nnTestSupport.hpp`
- `apps/po4ao/Makefile`
- `apps/po4ao/README.md`
- `apps/po4ao/po4ao.hpp`
- `apps/po4ao/po4aoRouteIO.cpp`
- `apps/po4ao/po4aoRouteIO.hpp`
- `apps/po4ao/po4aoRouting.hpp`
- `apps/po4ao/po4aoStatistics.hpp`
- `apps/po4ao/po4ao_control.hpp`
- `apps/po4ao/po4ao_impl.hpp`
- `apps/po4ao/po4ao_run.hpp`
- `apps/po4ao/tests/Makefile`
- `apps/po4ao/tests/po4aoRouteTestAbi.hpp`
- `apps/po4ao/tests/po4ao_control_test.cpp`
- `apps/po4ao/tests/po4ao_lifecycle_test.cpp`
- `apps/po4ao/tests/po4ao_route_io_test.cpp`
- `apps/po4ao/tests/po4ao_run_test.cpp`
- `apps/po4ao/tests/po4ao_telemetry_test.cpp`
- `apps/po4aoTrainer/tests/test_review_regressions.py`
- `apps/po4aoTrainer/xapp/po4aoTrainer/supervisor.py`
- `apps/po4aoTrainer/xapp/po4aoTrainer/switching.py`
- `gui/widgets/learningCtrl/README.md`
- `gui/widgets/learningCtrl/learningCtrl.hpp`
- `gui/widgets/learningCtrl/learningCtrl.ui`
- `gui/widgets/learningCtrl/tests/learningCtrl_test.cpp`
- `libMagAOX/Makefile`
- `libMagAOX/logger/logCodes.dat`
- `libMagAOX/logger/logMeta.cpp`
- `libMagAOX/logger/types/schemas/telem_po4ao.fbs`
- `libMagAOX/logger/types/telem.cpp`
- `libMagAOX/logger/types/telem_po4ao.hpp`
- `tests/groups.dox`

## Current dev integration decision (2026-11-08)

The prior review changes were committed as `ef8fbfeb` (functional fixes and evidence) and `24f8c265` (operator documentation). The merge uses current fetched `origin/dev` at `16138ab7b5b90bae028fb09aeb2c73e3f391844a`. PR #418 head `4ff2634124a5a6fcf428e22ce2807dc31b21b823` is already an ancestor; its implementation is preserved without replaying commits.

Conflict resolutions:

- `gui/lib/multiIndiPublisher.hpp` and `libMagAOX/app/dev/shmimMonitor.hpp`: exact upstream. The automatically merged `MagAOXApp.hpp` and shared FSM tests were also restored to exact upstream.
- Logger Makefile, code registry and metadata switch: retain all upstream entries and register PO4AO at unused 20908. These are necessary to build and decode the explicitly requested typed telemetry, not changes to shared monitor behavior.
- `tests/Makefile.one`: retain upstream rules and the collector-only crypto linkage / finite-value test flags. `tests/groups.dox`: retain upstream groups and app/GUI test groups, remove the custom shared lifecycle group.

NN uses its existing result lock and in-flight flags for GPU ownership. PO4AO uses one app-local callback mutex to serialize allocation, callback work, Stop draining and stopped-model preparation. Stop waits boundedly for that app work; it does not wait for mapping closure. Idle model work uses a try-lock, and telemetry aggregation runs after releasing the callback lock. No common lifecycle framework is introduced.

The remaining changes outside the owned apps are the requested GUI; essential telemetry schema/type/registration; build/test registration and documentation; and pre-existing PO4AO notebooks. `Makefile` adds only trainer/GUI build registration. No shared runtime lifecycle or generic GUI publisher behavior differs from dev. The normal-monitor regression is moved under `apps/po4ao/tests` and compiles the unchanged upstream monitor with a synthetic producer, real POSIX semaphore and mapping double. Its assertions cover ordinary READY detachment and absence of subsequent waits, not guaranteed attachment cancellation or rapid transition acknowledgements.

No push, PR update, instrument access or deployment is part of this integration. Remaining gates are the full target Linux/dependency build, real ImageStreamIO tests, real GPU ownership/release tests and operator-run internal-source acceptance before on-sky qualification.

Post-integration workstation evidence:

- NN: 624 assertions / 38 cases, including retained-attachment revalidation and fresh-frame gating; CACAO writer 29 / 3 plus the header-isolation build regression. NN built with ASan/UBSan.
- PO4AO: 434 / 36 with ASan/UBSan, including a blocked active callback that Stop must drain. Native fake-ABI routing adapter: 24 / 6. Normal upstream monitor with synthetic producer: 19 / 1, including three READY/restart cycles and no subsequent waits while parked. This fixture is not real ImageStreamIO ABI validation or a performance measurement.
- Qt GUI: 440 assertions / 28 cases, offscreen with no socket connection. Local INDI support archives were compiled as C++14 (their existing deprecated function adapters do not compile as C++20); no INDI sources changed.
- Collector CPU session/observation suite: 216 / 16 with ASan/UBSan. The app adapter still needs its real target build and streams.
- Logger outputs regenerated using the repository flatlogcodes tool and local flatc; schema round trip 76 / 1 with ASan/UBSan. Registry check confirms all upstream entries are preserved and codes are unique.
- Shared monitor, MagAOXApp, shared FSM tests and multiIndiPublisher compare byte-for-byte with origin/dev. No instrumentation or asynchronous FSM changes are merged into those files.
- The full native Linux/milk/CUDA build and real GPU/stream tests remain unrun on this macOS workstation. Trainer/Python suites were not rerun during this integration; their files and magaox-python have no new integration changes. Prior results above are historical evidence only.

Formatting uses current dev's clang-format configuration. App-only control-flow/format cleanup is kept in a separate final commit; no shared upstream formatting cleanup is included.
