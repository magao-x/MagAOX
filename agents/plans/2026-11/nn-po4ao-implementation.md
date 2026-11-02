# NN / PO4AO implementation record

Started 2026-11-02 from the user's approved [requirements](../2026-10/nn-po4ao-gui-requirements.md). This is an execution record, not a replacement for the agreed behavior.

## Current status, 2026-11-02

The agreed local implementation is complete as a deployment candidate: native GUI,
C++ current-controller recollection/readbacks, managed XDevice supervisor/worker,
hot configuration, switching, staged export, notebook example and installation
entries. The latest section below supersedes the earlier incremental status notes.
Full target SDK compilation, actual GPU export, live INDI/FPS acceptance and timing
remain operator-run requirements in `apps/po4aoTrainer/DEPLOYMENT.md`; this record does
not claim instrument deployment or physical acceptance. No source or configuration
was changed on an instrument host in this completion increment.

## Branches and baseline

- Fixed baseline branch: `rlandman/nn-po4ao-baseline`, commit `88271a3e6ec3c3add36a4983d806f05501108f20`.
- Child implementation branch: `rlandman/nn-po4ao-gui`, created directly from that baseline.
- Isolated managed checkout: `/Users/ricolandman/.codex/worktrees/nn-po4ao-gui/MagAOX`.
- The original dirty checkout remains intact. No instrument deployment, push or PR was performed. The separately authorized historical read-only inspection is recorded below; the completion increment used local evidence only.

The [baseline record](nn-po4ao-baseline.md) identifies exact source revisions and verification. NN comes from the newer correctness/control implementation, without the speedups. Collection comes from the latest committed donor subtree, with its test registrations merged into the PO4AO base. The accepted attachment is preserved byte-for-byte in the baseline commit; this child updates its implementation-status statements.

## First functional increment: shared numerical session

Implemented in `apps/po4aoTrainer`:

- Installable `xapp.po4aoTrainer` numerical package, matching the existing Python app namespace/distribution conventions. No personal project imports or mandatory deployed configuration on import.
- Pinned numerical helpers from local `PO4AO_2026` revision `b9076dd7aaf00ff7cd4f7f644e8f40dc1ff10a4f`; individual source hashes recorded in `source_manifest.json`.
- Explicit `TrainingSession` initialization and one-iteration method. Models, Adam state, fixed normalization, replay/history, counters and publication-reference state persist between calls.
- Caller-supplied complete replay snapshots and explicit active-policy identity. Unchanged or incomplete data do not consume completed iterations. Malformed/nonfinite data fail the session without leaving an older candidate eligible.
- Cooperative callback before every gradient update and at iteration boundaries. A blocking pause resumes the same stack and optimizer state; terminal interruption invalidates the session. This is the numerical hook, not the supervisor pause/reset protocol.
- Recollection invalidates historical replay and anchors, then waits for complete new warmup data. It preserves learned weights, Adam, reference state and established normalization. The future runtime adapter must signal recollection start promptly; new stream counters alone do not identify its start.
- Notebook diagnostic calculations and publication gates are retained. Explicit publication bookkeeping awaits the active-ID acknowledgement. No automatic export/reload or controller writes occur in the session.
- Existing plots import rendering dependencies only on request. Bounded, atomically replaced diagnostic snapshots include session/config/iteration/replay identity, fixed normalization, current recent replay and metric history. Observing notebooks render the same plots from these files. They are not resumable checkpoints.

Reuse is intentionally source-specific: the nonlinear networks, update losses, replay mixing, export helpers and diagnostic functions are ported from the pinned source. Only the needed modal layers are extracted from the old configuration-coupled model module; legacy linear models used by regression tests stay in test fixtures. Numerical expressions from the notebook's first iteration are checked against a fixture produced from the original source, rather than against another call to the new session wrapper.

The package has no console entry point, app Makefile or aggregate install entry yet. Adding a fake app entry point would make normal application installation advertise an unfinished supervisor. These entries will be added with the real XDevice implementation. This does not change the approved production packaging design.

## Local verification

Baseline, before new implementation:

- NN app-local CPU tests: 461 assertions, 27 cases.
- PO4AO helper CPU tests: 79 assertions, 14 cases.
- Collection CPU tests: 216 assertions, 16 cases. Apple libc++ required compiling the legacy INDI dependencies as C++14 and linking them with the C++20 test/OpenSSL; no source modification was needed.

New numerical implementation:

- 60 Python tests and 15 subtests passed with Python 3.11, PyTorch 2.13.0 and NumPy 2.4.6 on CPU.
- Original-notebook fixture comparison covers all model state, all Adam state, selected losses/validation metrics, publication status and reasons (`rtol=1e-5`, `atol=1e-7`). Fixture source hash and seeds are recorded beside the NPZ.
- Cooperative pause between dynamics updates resumes with bit-identical model outputs to uninterrupted CPU work. Incomplete/duplicate data, recollection invalidation, persistent optimization, pending publication acknowledgement and interrupted-session eligibility are covered.
- Direct and recorded diagnostic figures rendered with identical pixels. Existing optional SciencePlots deprecation and STIX superscript-minus font warnings remain; numerical tests and render comparisons passed.
- A normal wheel was built and installed into a temporary local virtual environment. From `/tmp`, Python isolated mode imported the installed package with no source-project paths or personal `PYTHONPATH`, performed a synthetic CPU iteration and saved/read diagnostics. Model initialization and CUDA initialization were prohibited during the import check. No plotting module or legacy deployed config was imported by the headless core.
- Declaration docstrings checked throughout changed Python files. Python was formatted with Black. No new C++ code was edited; imported app subtrees remain byte-identical to their pinned donors, so they were not reformatted.

These checks establish CPU extraction/packaging behavior only. No full training campaign, CUDA/TensorRT build, installed INDI contract, physical routing, timing or closed-loop behavior was validated.

## Remaining work in the approved order

1. Finish phase A contracts: import the coherent upstream GUI publisher/subscriber/test dependencies; pin the current upstream revision at integration. Reconcile the operator's intended notebook snapshot, runtime/device mappings and installed milk writer-observation interface. The local notebook source is pinned but equality with the instrument copy is unverified.
2. Complete phase B: typed TOML/xconf schema and atomic config updates at declared boundaries; preserved-Adam learning-rate updates; managed-worker protocol; bounded build cancellation, artifact staging and reset-generation eligibility; engine metadata/config revision integration. Adapt the operator's verified development notebook to the packaged core. Existing synchronous `publish_tensorrt_engine` still performs separate engine/sidecar replacements and must not yet be wired to a managed worker.
3. Phase C: C++ current-controller warmup exploration/recollection, control-Off collection and completion/readback semantics, with seeded helper tests and the required coding/documentation/format pass.
4. Phase D: XDevice supervisor, owned worker lifecycle, narrow switching adapter and completion-aware INDI operations. Do not invent CACAO RUN evidence; an actual installed observation adapter is still needed for handover acceptance. No notebook locks or ownership enforcement.
5. Phase E: compact existing-style Qt widget and launcher, shared upstream offline test harness, build/install entries and separate instrument-config changes.
6. Phase F: operator-run deployment and acceptance. Agents do not run on or connect commands to instrument hosts. Loop state and notebook/app responsibility remain operator-managed as agreed.

No product decision was changed by this increment. The package is ready for continued implementation and local notebook experiments with explicit inputs; the complete GUI/trainer system is not ready for instrument deployment.

## Native GUI increment, 2026-11-02

The user requested continuing the GUI implementation and explicitly authorized read-only exao2 inspection. The earlier instruction-file change was reverted in `60b0d8a6`; `AGENTS.md` is now identical to the baseline and is not part of this GUI change.

The current upstream revision was refreshed and remains `5a45f1c7b73faeb7dbe63dda18209e15306a2be8`. Import only `gui/lib/multiIndiPublisher.hpp` and its three offline test/harness files. The subscriber, manager, xWidget, common qmake settings and INDI dependencies required by that publisher already match this base. No pupilGuide implementation or unrelated upstream app change is needed. The upstream publisher's six offline cases passed 95 assertions before integration.

The new `learningCtrl` widget and `learningCtrlGUI` launcher use the shared stylesheet, xWidget/statusLabel controls, Designer layouts, dynamic model definitions and queued publisher callbacks. Dropdown changes stay local until Enable. Reconstruction commands are explicit targets, independent of learning/collection state. Control labels reflect observed status; pending requests disable conflicting actions until Busy followed by Ok or Alert. The ten-second initial-acknowledgement timeout leaves the request visibly unknown and never retries it. Reset stays available during ordinary training/collection transitions. Property deletion, sparse updates, device mapping changes, two clients, disconnected state and long labels are covered by the socket-free widget tests.

A request property must acknowledge Busy before sending its terminal state; requested values alone are not completion. This is the concrete GUI/companion contract documented in `gui/widgets/learningCtrl/README.md`. The supervisor is still required to implement it, including serialized operations and actual writer/worker completion. The GUI does not bypass the companion or invoke commands through a local shell. Its controls remain disabled when that companion is absent.

Local verification: the real widget/launcher built with Qt 5.15.15, and 115 assertions in eleven widget cases passed. The optional preview case passed four assertions and rendered the actual widget using the normal resource stylesheet. No socket is constructed in these tests. The local launcher build omits the unused mxlib pkg-config linkage via a command-line qmake override; the production project retains the common build rules. Qt reports that the installed macOS SDK is newer than its tested SDK, and the existing shared publisher emits missing-override warnings. Native Linux/instrument deployment remains untested.

Read-only exao2 evidence:

- All six numerical helper hashes match `source_manifest.json` exactly.
- Instrument notebook SHA256: `cb8b68310b2b95e1506526b8bdfb87064f484b874c8fbdeab7bf0884528f3da6`. Its source differs only by commenting the initial random identity probe, adding a direct warmup plot, and whitespace; the training cell's numerical expressions are unchanged. The packaged core already leaves that optional probe to notebook code. The local parity fixture still intentionally includes the pinned notebook's probe and records its seed.
- Installed milk source revision: `336160d054df5773a49d603ba987bea74692fcda`. `CacaoProcessTools.fps` exposes `RUNrunning`, `md().runpid`, `runpidstarttime` and status. `RUNrunning` checks PID liveness and distinguishes an unclean exit; clearing `CMDRUN` is only a stop request. PID identity/reuse still needs attention in the eventual adapter.
- `FPSCONNECT_SIMPLE` writes bookkeeping metadata and connection logs, so no FPS object was instantiated during this read-only inspection. Only existing files, hashes, source definitions and directory/revision information were read. No INDI requests, control commands, process changes, installs or remote file writes occurred.

Remaining acceptance is unchanged: managed configuration/worker/publication, C++ current-policy recollection, the companion's real operations and deployment are not supplied by a rendered frontend. This increment completes the native panel and its offline request/readback behavior; it does not claim working instrument control.

## Current-controller recollection increment, 2026-11-02

The PO4AO runtime now keeps collection state separate from policy history/selection.
The production frame path computes the selected learned or integrator command,
adds the existing exploration schedule once, applies existing limits, emits it,
and records that emitted command in history/replay. Explicit collection also
works with control Off and a retained model. Initial exploration still waits for
an enabled frame. No loop readback, loop command or NN inference change was added.

A generation-tagged lifecycle distinguishes collecting, awaiting snapshot
publication, ready and interrupted. Source gaps, controller/model/reset changes,
inference fallback, invalid output and publication backlog interrupt unfinished
collection. Old publisher callbacks cannot complete a newer generation. Complete
warmup publication requires contiguous episodes and the full target sample count;
metadata captures the effective policy at the final episode frame rather than
sampling the current model later. A reader must pair the existing stream counters
with the requested collection's ready readback, rechecked after its copy.

The C++ control/reset acknowledgements now follow successful frame publication.
The collection request identity remains available after admission so appLogic
cannot report completion in the interval before the frame starts collection.
Reallocation invalidates old snapshots and preserves requests made before stream
attachment. Observed effective controller/model, request/applied identities and
collection progress are published every appLogic cycle before acknowledgements;
latency statistics retain their configured period. Load errors remain observable
until retry. A bounded publisher wait avoids an indefinite lost-notification stall
without adding a real-time mutex.

`po4ao_control.hpp` contains the actual frame/episode/publication definitions, still
included by the header-only app. The CPU boundary harness includes those same
methods, substituting only SDK/output/model-loading boundaries. This keeps runtime
algorithms under test without importing a GPU SDK or copying the control logic.
The new tests are registered with the existing application test group/list.

Verification: 259 assertions in 27 cases passed, including AddressSanitizer and
UndefinedBehaviorSanitizer. These checks cover helpers and production frame/replay
methods; they do not compile the complete CUDA/TensorRT app or exercise real INDI
callbacks, shared-memory transport, or instrument timing. Full SDK compilation and
operator-run acceptance remain required. The GUI tests/build from the preceding
increment remain valid; no GUI behavior changed during this runtime increment.

Remaining implementation: typed hot configuration, managed worker/build lifecycle,
atomic eligible-engine publication, the XDevice supervisor, actual NN/CACAO and
modulation transactions, notebook adaptation, and deployment registration. The
frontend and C++ runtime are committed development increments; the complete shared
control system is not yet operational or ready for instrument installation.

Final source review also found that the inherited PO4AO callbacks updated shared
INDI properties without the app mutex used by appLogic. The four runtime request
callbacks now use that existing mutex, matching NN callback practice. This
serializes property/request publication without adding a lock to the frame path.

## Managed operations and deployment increment, 2026-11-02

The companion now uses the existing XDevice/BaseConfig/xconf and PurePyINDI2
interfaces, not a second GUI transport or service manager. Reconstruction and
training each have one serialized operation queue. Callbacks acknowledge Busy
before scheduling; observed status precedes terminal Ok/Alert. GUI disconnects do
not own process lifetime. Multiple clients share the same backend state.

Implemented scope:

- Typed TOML configuration, explicit hot-field map and whole-revision reset
  staging. Runtime dimensions/history/gains/limits come from the C++ configuration.
  Invalid edits retain the last good revision; hot learning-rate updates preserve
  Adam. The current numerical contract requires equal recent/warmup capacities:
  `warmup_episodes * iterations_per_ep == replay_buffer_size`.
- One owned worker with JSON-line control, cooperative checkpoints, persistent
  models/Adam/normalization, immutable config snapshots, metrics and bounded
  diagnostics. EOF/heartbeat loss stops the worker. Worker errors remain observable
  through the companion and do not block independent reconstruction or Disable.
- An owned, cancellable export child wraps the unchanged numerical export helper
  in a private candidate directory. Pause stops that child while retaining the
  worker. Reset first invalidates the old generation, then stops owned processes,
  selects the integrator and waits for runtime reset acknowledgement. No notebook
  discovery, ownership locks or notebook process control were introduced.
- Complete candidate verification ties the engine hash, metadata, config revision,
  snapshot and generation together. A single atomic directory symlink exposes the
  matching bundle to C++. Eligible retained engines survive Disable; Reset makes
  old files ineligible. Reloading even the same ID requires a fresh Busy/terminal
  callback. Cancelled/failed publication always releases the worker's candidate
  wait. Recollection pauses/resumes only its managed session, preserving the actual
  controller and requiring the newer full snapshot.
- The narrow switching adapter uses the installed milk FPS fields observed in the
  preceding source inspection, PID creation identity/command-line matching and
  nonblocking FIFO delivery. Shared FPS observation is serialized across polling
  and operation threads. Commands never query/change loop state or gate on PO4AO.
  Linked modulation is optional and uses 2000 Hz/radius 3 for CACAO. An already
  loaded selected NN model does not issue a redundant reload; other selections
  disable, load and enable through the pinned donor controls.
- Standard console entry point, app Makefile and RTC Python-app registration;
  dedicated `INSTRUMENT_CONDA_ENV=po4aoTrainer` selection without modifying shared
  Python build infrastructure. The wheel includes config examples and the numerical
  source manifest. Shared paths and external instrument-config entries remain
  explicit deployment steps. The configured installed `source_revision` is logged
  and recorded in launches, metrics and engine metadata for experiment provenance.
- `notebooks/po4ao/development.ipynb` uses the shared core for an executable synthetic
  CPU session, repeated iterations, direct plots and recorded diagnostics. It stays
  outside `apps/` and does not connect to the instrument. The original notebook is
  not overwritten. Its existing operator acquisition cells can supply explicit
  snapshots to this same core; notebook exports are not automatically adopted by
  the managed supervisor.

Review fixes included stale same-engine reload acknowledgements, release of a
cancelled publication wait, duplicate Reset cancellation, worker failure during
Disable, failed protocol handling, and independent loaded-versus-effective model
identity (learning can acknowledge a published model while the integrator is
selected). These are operation correctness fixes, not new loop interlocks.

Local validation:

- 82 Python tests and 15 subtests passed. This includes the original numerical/plot
  parity checks, actual XDevice property definitions without sockets, hot config,
  two operation domains, generation eligibility, writer/modulation order, real
  owned worker pause/resume/recollection/EOF, export cancellation and cleanup that
  preserves an unrelated process. GPU/stream/export numerical boundaries are
  synthetic in process tests. Process metadata checks needed normal local process
  visibility; no instrument process was inspected or controlled.
- PO4AO CPU production-method/helper suite: 259 assertions in 27 cases, using the
  existing sanitizer build. Native GUI: 115 assertions in 11 cases. Upstream
  publisher: 95 assertions in 6 cases. Earlier launcher build remains applicable;
  no native GUI/C++ implementation changed in this completion increment.
- The notebook's code cells executed on synthetic CPU inputs and rendered both
  direct and recorded diagnostic figures. Existing optional SciencePlots/Matplotlib
  deprecation and STIX font warnings remain; they do not fail plot comparisons.
- Normal wheels were built/installed in a temporary local Python 3.11 environment.
  From `/tmp`, isolated mode imports the installed supervisor without importing
  torch/SHM/FPS bindings; console/manual-adapter/deployment-check help works.
  Config examples and source manifest are installed under `share/po4aoTrainer`.
  No personal source-project path is a runtime import dependency.
- The offline framework checks used actual xwcl xconf
  `d6f4de40bd290510bfe191b777d85d36b467d0ce` and purepyindi2
  `c26a8276c95b95aef145c50439e291b782eab190`. No unrelated PyPI xconf package is
  substituted. Target deployments must use compatible installed framework builds.
- Python formatted with Black; declaration docstrings checked. New app tests are
  grouped under `application_unit_test` in `tests/groups.dox`. No new C++ formatting
  was necessary. `git diff --check` passes. NN and dlDataCollection remain exactly
  equal to their pinned donor subtrees; `AGENTS.md` remains equal to the baseline.

Affected files in this completion increment:

- `Makefile`, `tests/groups.dox`.
- `apps/po4aoTrainer/Makefile`, `pyproject.toml`, `README.md`, `DEPLOYMENT.md`.
- `apps/po4aoTrainer/configs/po4aoTrainer.conf`, `po4ao_training.conf`.
- `apps/po4aoTrainer/xapp/po4aoTrainer/{app,configuration,supervisor,process,worker,switching,artifacts,check_deployment}.py`.
- `apps/po4aoTrainer/xapp/po4aoTrainer/__init__.py` (description only).
- `apps/po4aoTrainer/tests/test_supervisor.py`, `test_worker_process.py`.
- `notebooks/po4ao/development.ipynb`, `gui/widgets/learningCtrl/README.md`.
- This execution record and the requirements' current-instruction/status notes.

No further product decision is needed for the operator deployment trial. Before
scientific or on-sky use, the target acceptance in DEPLOYMENT.md and the handbook's
committed/pushed source-and-config provenance requirements still apply. This branch
has not been pushed or deployed by the agent.
