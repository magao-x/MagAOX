# Idle NN, PO4AO and collection lifecycle

Date: 2026-11-08. Status: implementation and workstation verification; isolated Linux/GPU and operator acceptance remain required. No deployment or instrument performance claim.

## Baseline and authorization

Implementation branch: `rlandman/po4ao-idle-lifecycle`, created directly from verified `rlandman/nn-po4ao-gui` commit `a84232b938f359fdebb2973b6a1e367e0343002a`, in `/Users/ricolandman/.codex/worktrees/po4ao-idle-lifecycle/MagAOX`. The combined checkout and its existing GUI, saved policy pairs, NN/PO4AO controls and collector improvements are preserved. Only this planning document was copied from the review checkout; no NN implementation was copied, merged or cherry-picked from `nnreconstructor-speedups`.

All implementation, builds and tests run on the workstation. The user separately authorized read-only exao2 source/readback inspection; no instrument control utility, INDI command, tunnel, installation, build or agent runtime was executed there. Instrument acceptance below is operator-run.

The final scope decision supersedes the original gain-restoration proposal: “complete routing, do not do anything with gains/leakage, also no note”. There is no gain/leakage snapshot, restoration, verification wait or GUI reminder. Existing utility side effects are unchanged. No hogainctrl interlock or changes outside the user's apps for instrument controls were added.

## Implemented lifecycle

Use `shmimMonitor`'s existing READY/OPERATING gate, following `shmimIntegrator`. There is no independent monitor pause flag, condition variable or subscription framework.

- Monitor admission and FSM requests share a short mutex. The idle acknowledgement becomes true only after mapping/semaphore ownership and callbacks are released. RAII cleanup covers cancellation after opening, failed allocation, missing semaphore, stat failures and normal detachment. Frame-loop gating reads the atomic application FSM without adding a per-frame lock.
- A supporting `MagAOXApp` correction makes FSM state/alert/log-count access atomic. Periodic INDI publication and asynchronous log alerts now refresh status without reissuing a potentially stale state transition. Otherwise `state(state())` could undo a concurrent READY request. This is the additional common lifecycle change needed by the worker transitions, not a new pause mechanism.
- NN disabled startup and completed Disable settle in READY. Enable recreates GPU resources from retained validated engine bytes, requests attachment, validates it and admits fresh frames. Disable closes inference admission, drains in-flight work, acknowledges its final zero, waits for monitor detachment, and then destroys owned TensorRT context/engine/runtime and CUDA buffers outside the callback. No `cudaDeviceReset`. CPU output buffers remain available for final publication. A cancelled/late loader cannot reinstall resources.
- `model_status.loaded` identifies the host-prepared model, `resident` reports GPU residency, and `active` still requires published inference. Compound NN/CACAO activation prepares GPU resources before stopping a usable CACAO writer. Failure does not silently substitute a writer.
- Collector startup is READY. Start queues a bounded five-second attachment/observing-readback stage before the existing session preflight; Stop cancels it. Terminal/draining session phases park the input monitor. An unexpected reconnect during an active session still aborts the partial run. Existing bank, metadata and output-ownership checks remain in place.

READY is asynchronous. A stopped producer can delay cleanup until the existing semaphore timeout, and the idle outer loop can delay reattachment. No fixed sleep is treated as acknowledgement. Rapid reactivation waits for actual detach completion.

## Whole-app PO4AO routing

Native `po4ao.run_enable.toggle` is separate from `control_enable.toggle` (policy Enable/Disable). `run_status` publishes phase, observed route, error, operation owner and verified status. The GUI uses explicit Start/Stop targets and blocks competing runtime operations during a transition. Policy selection is retained; learning never resumes automatically. The optional trainer pauses an existing training worker on entry to routing and rejects new learning/recollection while stopped. Saved policy selection and saving remain available while stopped.

A single non-real-time C++ control worker owns an operation with a 90-second deadline. Startup retains the prior OPERATING behavior with an unverified route: the app must not assume downstream has stopped using its output after an app restart. Output channel must be `po4ao_output` for this installed routing adapter. NN reservation subscriptions use device `nnReconstructor`.

Start sequence:

1. Read current route/calibration and require an already operator-open HOWFS loop.
2. Attach and observe fresh PO4AO output before changing the downstream input.
3. Reserve NN/CACAO writer ownership through NN's native INDI reservation.
4. Run `howfs_mfilt_switch po4ao_output` in the verified HOWFS directory.
5. Run `cacao-calib-apply default`, check its fresh completion marker and resolved calibration identity.
6. Restore the actual pre-request NN/CACAO writer using its existing handover and first-result checks. The reservation does not alter the operator's modulation preference or request modulation changes.
7. Observe fresh output again after restarts and advancing mfilt consumption on that exact stream inode. Acknowledge Running.

Stop follows the same reservation/switch/calibration/writer-verification sequence with `aol1_modevalWFS`. Only after downstream consumption is verified does it close callback admission, request READY, wait for monitor detachment and acknowledge Stopped. It interrupts an unfinished warmup collection after callback quiescence. Already completed saved policies are retained.

The app does not open or close the loop. Calibration's existing procedure opens it, hence the preflight requirement that the operator already has it open. Loop closure remains an operator action.

NN exposes `routing_hold` text elements `owner` and `action` (`hold`/`release`) with `routing_status` owner/source/phase/instance/sequence readbacks. It rejects competing native Enable/model/handover/modulation controls while held. Hold snapshots the verified actual writer; it acknowledges held only after NN is fully disabled when needed. Calibration restarts CACAO even when NN was active, so release uses the standard handover to stop CACAO before enabling NN. No external gain-control interlock is implied.

Failure behavior:

- Failed Stop never parks the monitor while the downstream route may depend on PO4AO. Report actual or unknown route, failed stage, error and unverified status.
- Commands use fixed executable/argument pairs and the required working directory. Timeout/cancellation kills and reaps the utility process group. FIFO requests already queued can still complete; do not automatically roll back or release an unresolved writer reservation.
- Preserve the operation token and NN reservation for operator recovery. After independently settling outstanding utility/FIFO actions and verifying ownership, the operator can release the matching token through NN and retry the explicit target. Do not clear another operation's reservation or blindly restart an app to conceal it.
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

Event `TELEM_PO4AO` (20906) records callback p50/p95/p99/max in microseconds, valid sample count, sequence and monotonic sample time, freshness and mixed-controller-window flags, exact uint64 cumulative missed-frame/deadline/inference/replay counters, deadline, model ID and backend (0 fallback, 1 CPU, 2 TensorRT). It is callback timing, not end-to-end optical/control latency. Empty windows carry zero samples; sub-microsecond samples are valid zeros; idle windows are marked stale; transition windows are marked mixed and identify the newest included sample. The fixed atomic sample ring is filled by the callback; allocation, sorting and serialization run outside it.

## Workstation evidence and reproduction

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
make -C libMagAOX/app/dev/tests -f Makefile.lifecycle test
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=apps/po4aoTrainer:magaox-python /tmp/po4ao-dev/bin/python -m pytest -p no:cacheprovider -q apps/po4aoTrainer/tests
```

Schema tests require the normal logger `flatlogcodes`/`flatc` generation step first. The workstation Qt build uses `learningCtrl_test.pro`, Qt6 and locally compiled INDI libcommon/liblilxml archives. The existing collector test uses its real CPU session and observation sources with local INDI/OpenSSL dependencies. Detailed current assertion counts are in the completion report; tests do not establish instrument deployment success.

## Remaining acceptance gates

1. On an isolated non-instrument Linux host, build libMagAOX and all affected C++ apps against the installed-version milk/ImageStreamIO/CUDA/TensorRT dependencies. Run the normal MagAOXApp regression suite. Verify telemetry logger registration/accessors and decode a real emitted TELEM_PO4AO record, including idle and model transitions.
2. With a real ImageStreamIO synthetic producer, exercise missing-source startup, cancellation during attachment/allocation, producer stalls, rapid Stop/Start and stream replacement. Check reader slots and actual wait/wakeup counts after READY while the producer continues. Validate collector pending Start/Stop and observe that idle sampling stops. The workstation mapping double alone does not close this gate.
3. On an isolated GPU host, measure enabled, completed-disabled and re-enabled NN resource usage across repeated cycles. Require owned buffers/objects to be gone only after final-zero acknowledgement and in-flight completion; retained CUDA context overhead is allowed. Inject allocation/load/cleanup failures and verify no late residency or duplicate NN/CACAO writer.
4. Operator-run instrument acceptance: verify the inspected utility/header versions and working directory, actual device/channel names and readback semantics. Begin with the HOWFS loop already open; independently verify writer ownership. Test both NN and CACAO starting owners, Stop to WFS, Start to fresh PO4AO, downstream inode/counter advancement, READY detachment, retained policy and no automatic learning restart. Close the loop only through the normal operator procedure after reviewing readbacks.
5. Operator-run failure acceptance: withhold a required readback or simulate a utility failure in an isolated replica first. Confirm failed Stop keeps output available, no false completion, owner token/partial route are visible and recovery does not race queued commands. No automatic recovery from external manual routing or unresolved FIFO work is claimed.

## Affected files

The following is the implementation inventory; generated artifacts and local test/build dependencies are not committed:

- `apps/dlDataCollection/dlDataCollection.hpp`
- `apps/nnReconstructor/nnReconstructor.hpp`
- `apps/nnReconstructor/reconstructorControls.hpp`
- `apps/nnReconstructor/tests/Makefile`
- `apps/nnReconstructor/tests/nnReconstructor_test.cpp`
- `apps/nnReconstructor/tests/nnTestSupport.hpp`
- `apps/po4ao/Makefile`
- `apps/po4ao/po4ao.hpp`
- `apps/po4ao/po4ao_control.hpp`
- `apps/po4ao/po4ao_impl.hpp`
- `apps/po4ao/tests/Makefile`
- `apps/po4ao/tests/po4ao_control_test.cpp`
- `apps/po4aoTrainer/tests/test_review_regressions.py`
- `apps/po4aoTrainer/xapp/po4aoTrainer/supervisor.py`
- `apps/po4aoTrainer/xapp/po4aoTrainer/switching.py`
- `gui/widgets/learningCtrl/learningCtrl.hpp`
- `gui/widgets/learningCtrl/learningCtrl.ui`
- `gui/widgets/learningCtrl/tests/learningCtrl_test.cpp`
- `libMagAOX/Makefile`
- `libMagAOX/app/MagAOXApp.hpp`
- `libMagAOX/app/dev/shmimMonitor.hpp`
- `libMagAOX/app/tests/MagAOXApp_test.cpp`
- `libMagAOX/logger/logCodes.dat`
- `libMagAOX/logger/logMeta.cpp`
- `libMagAOX/logger/types/telem.cpp`
- `tests/groups.dox`
- `agents/plans/2026-11/shmimMonitor-idle-apps.md`
- `apps/po4ao/po4aoRouteIO.cpp`
- `apps/po4ao/po4aoRouteIO.hpp`
- `apps/po4ao/po4aoRouting.hpp`
- `apps/po4ao/po4aoStatistics.hpp`
- `apps/po4ao/po4ao_run.hpp`
- `apps/po4ao/tests/po4aoRouteTestAbi.hpp`
- `apps/po4ao/tests/po4ao_lifecycle_test.cpp`
- `apps/po4ao/tests/po4ao_route_io_test.cpp`
- `apps/po4ao/tests/po4ao_run_test.cpp`
- `apps/po4ao/tests/po4ao_telemetry_test.cpp`
- `libMagAOX/app/dev/tests/.gitignore`
- `libMagAOX/app/dev/tests/Makefile.lifecycle`
- `libMagAOX/app/dev/tests/shmimMonitor_lifecycle_test.cpp`
- `libMagAOX/app/dev/tests/stubs/`
- `libMagAOX/logger/types/schemas/telem_po4ao.fbs`
- `libMagAOX/logger/types/telem_po4ao.hpp`
