# NN / PO4AO implementation record

Started 2026-11-02 from the user's approved [requirements](../2026-10/nn-po4ao-gui-requirements.md). This is an execution record, not a replacement for the agreed behavior.

## Current status, 2026-11-02

The independent review of `4790c3c8` found control-coordination bugs, so the earlier
deployment-trial readiness statement was premature. All eight findings and the
additional artifact/diagnostic/mapping/progress gaps are addressed in the
[review response](nn-po4ao-review-response.md), with regression evidence and limits.
The corrected branch is ready for independent re-review and operator-run target
validation, not established ready for control-capable instrument use. Full target
SDK compilation, actual GPU export, live INDI/FPS acceptance and timing remain
requirements in `apps/po4aoTrainer/DEPLOYMENT.md`. Historical increment notes below
describe their respective revisions; the review response supersedes their readiness
claims and Reset/publication sequencing. This review used local resources only.

## Notebook independence and one launcher, 2026-11-02

The user rejected the mandatory companion and a second runtime-only GUI. Replaced
`learningCtrlGUI` with one `po4aoGUI [po4ao-device [trainer-device]]`, keeping the
existing widget, Designer layout, stylesheet, publisher and build rules. Default
names are `po4ao` and `po4aoTrainer`. Both devices are independently subscribed.
Without the trainer, native Enable/Disable, Reset and Recollect remain available;
Start/Pause is disabled. With a fresh trainer mapped to this runtime, the same
buttons use existing managed transactions. The NN section remains unchanged and
needs the companion. No C++ or Python backend changes are needed for this correction.

Native Reset sends control Off only when observed enabled, then requests runtime
reset. Enable selects the resident model; notebook export/reload remains notebook
code. No Python kernel, models or optimizer objects are manipulated. The old notebook
can remain in its existing environment; the shared-core example is optional.

Requests and acknowledgements are scoped by device/property. Independent heartbeats
and runtime process identity reject stale capabilities. Trainer arrival/loss sends
no commands, and unresolved local requests cannot be retried through the other
backend. GUI actions do not open/close the loop or enforce notebook/app exclusivity.

Local verification: Qt widget suite passes 240 assertions in 19 cases, including
native-only use, Reset ordering, optional trainer arrival/departure, independent
freshness, native permissions, restart invalidation and cross-device acknowledgement.
The shared publisher suite also passes 95 assertions in 6 cases. The native-only
500-pixel preview was rendered and inspected. The renamed launcher builds with
Qt 5.15.15 on macOS. Local qmake overrides disable
unavailable mxlib pkg-config linkage; production build files keep standard linkage.
Existing shared-INDI override and SDK/deployment-target warnings remain unchanged.
No remote access, instrument build, deployment, GPU or live INDI test was performed.
The consolidated `apps/po4aoTrainer/DEPLOYMENT.md` replaces the notebook runbook and
contains both installation paths without requiring another guide.

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

At that revision, no further product decision was identified for an operator deployment trial. Before
scientific or on-sky use, the target acceptance in DEPLOYMENT.md and the handbook's
committed/pushed source-and-config provenance requirements still apply. This branch
has not been pushed or deployed by the agent.

## Independent-review corrections, 2026-11-02

The [finding-by-finding response](nn-po4ao-review-response.md) records the implemented
corrections and revised acceptance boundary. Runtime reset/process identity and
advancing status now support external-state reconciliation. Pending Reset/load
state survives a failed wait; pause preserves the exact issued model reference.
Received definitions and heartbeat expiry replace reliance on retained client cache.
NN fault recovery, same-model no-op, NN-only replacement, busy-state Disable,
configuration validation, artifact identity, final diagnostic outcome, configurable
modulation mappings and visible collection progress are covered by new tests.

Current local results: 107 Python tests plus 15 subtests; 151 Qt assertions in 14
cases; existing PO4AO CPU suite 259 assertions in 27 cases. Launcher and wheel build,
and isolated installed-package imports pass. These do not compile/test the complete
target C++ status path or establish GPU/hardware behavior. The requirements were not
relaxed; no NN/dlDataCollection/AGENTS.md changes or instrument access were made.

## Follow-up review: exact publication identity, 2026-11-02

The review of `91ec5e2841bd63b73ef94eeeea8221d425185ff0` identified a second
publication interleaving: the status thread could capture candidate A's completion
and apply it after candidate B became pending. Each pending publication now carries
its exact runtime reload ticket in memory. Completion checks candidate object,
ticket, model and generation under the existing supervisor lock before changing
eligibility or notifying the worker. A stale completion becomes a no-op.

The worker's candidate reference is registered after ticket creation and before
reload delivery. A failure before ticket issuance rejects the new candidate; an
uncertain send retains only the new ticket and reference until completion or Reset.
A second candidate cannot replace the current link/pending record while a previous
load is unresolved. Restart recovery continues to require Reset when the ticket
and numerical session were lost; no new persistence or orchestration layer was added.

A deterministic two-thread regression holds A's status result while B is issued,
then verifies persisted eligibility and worker notifications before and after B
actually loads. Additional tests cover both sides of issuance failure, immediate
delivery and admission while a load remains unresolved. The stale-completion and
pre-ticket-failure regressions fail against the reviewed revision in an isolated
temporary source tree.

Validation: the complete local source suite passed **112 Python tests and 15
subtests** with the actual xwcl xconf/PurePyINDI2 framework. Normal local process
visibility was needed for the existing owned-export-child test; no instrument
processes or sockets were used. The wheel rebuilt and all 18 packaged Python
modules match this checkout byte-for-byte. Existing plot/font warnings remain.
Black and `git diff --check` pass. This does not validate target GPU export/reload.

## Follow-up review: GUI route changes and native Reset, 2026-11-02

The same review's two GUI findings are accepted. Route changes now retain observed
Busy operations, including other clients' requests and managed publication phases,
in addition to this window's local acknowledgement tracking. Trainer expiry cannot
bypass managed work through native controls, and trainer arrival cannot bypass
native work through managed controls. Completion or property/device invalidation
releases the old route. Returning to its original route restores normal same-route
coordination. No cross-route superseding action is introduced; existing managed
Reset/Disable and native Disable-during-recollection rules remain available.

Native Reset can supersede Enable while input frames are absent. It always sends
control Off before reset_model, even when the applied toggle remains Off and its
requested target is On. Reset owns the superseded selection's completion: advancing
reset_request acknowledges admission; matching reset/control applied generations,
target Off and actual control Off/Ok confirm completion. A coalesced Busy update or
an idempotent Off cannot strand a separate local request latch. Runtime invalidation
discards the old generation ticket without replaying a command. Button handlers
use the same route snapshot as their availability check.

The new GUI tests reproduced all three failing cases on the reviewed header: both
route-transition directions and Reset during pending Enable. The corrected suite
passes **313 assertions in 22 cases**, including other-client Busy and internal
publication state, completion/invalidation, stale/coalesced acknowledgements and
eventual Reset completion. Existing publisher tests pass **95 assertions in 6
cases**. All use the production queued publisher with a socket-free transport.
The renamed po4aoGUI launcher builds with local Qt 5.15.15; the existing local qmake
override omits unavailable mxlib pkg-config linkage without changing production
build rules. The executable was not launched against an endpoint.

The unchanged PO4AO CPU production-method/helper suite also passes **259 assertions
in 27 cases**. Full target CUDA/TensorRT compilation and real INDI/FPS behavior are
not established by these checks. clang-format and `git diff --check` pass; existing
Qt SDK/INDI/publisher warnings remain. NN, dlDataCollection and AGENTS.md still match
baseline `88271a3e6ec3c3add36a4983d806f05501108f20`; notebook code/environment/plots
and NN/CACAO writer switching were not changed. No remote access or deployment occurred.

The requirements now record these acknowledgement rules and the user's manual-only
downstream routing decision. The deployment guide includes initial warmup before
managed learning and verification/restoration of the recorded downstream route and
calibration. The updated review response assesses notebook and managed trials
separately; routine instrument readiness remains unverified. One GUI, optional
trainer and operator-managed loop state remain the agreed design.

## Home-checkout update and shorter deployment guide, 2026-11-03

The user explicitly authorized updating the home checkout on exao2 and reading
deployment files there. `/home/rlandman/MagAOX` was on
`rlandman/nnreconstructor-correctness-controls` at `4353737e3`, with six modified
dlDataCollection files and nine untracked files. The modified code and four new
source files already matched this GUI branch; its README only lacked three lines
present here. The complete old working state was saved in Git stash
`7a5d3f59f05a13ef000595fb0f424327f2ed9034`, named
`Before nn-po4ao-gui branch update 2026-11-03`. After switching to
`rlandman/nn-po4ao-gui`, the five remaining untracked helpers/binaries were restored
and compared byte-for-byte with that stash. Tracked files match the new branch.
Transfer used a Git bundle, without publishing to GitHub.

File inspection confirmed existing PO4AO process/INDI registration, notebook kernel
`/opt/conda/envs/torchpy/bin/python3.11`, the notebook engine path, current
10-by-1000 warmup/10000 replay configuration and installed TensorRT builder.
Neither trainer nor GUI was installed, and qmake was not on PATH. The downstream
symlink pointed to `aol1_modevalWFS`; its calibration-source record named `default`.
Those files do not establish actual stream consumption or current control state.
The installed routing scripts were read, not executed. They restart CACAO processes
and the calibration script opens the loop, so the guide retains manual setup and
verification in the recorded HOWFS loop directory.

The deployment guide is shortened from 1663 to about 1200 words. It uses actual RTC
paths and the notebook environment, removes already-completed registration steps,
and includes the two INDI archives needed by a fresh GUI checkout. Inspection also
found no Python 3.11 CacaoProcessTools module: installed milk copies are for 3.10
and 3.13. The guide uses milk's existing standalone Python-module CMake target,
installed headers/libraries and the trainer's python3-config to build just this
extension into the separate environment; it does not reinstall milk. The build
command was checked against source/build files, not executed. All five shell
blocks pass `bash -n`; no deployment block was executed. No application code changed,
so functional suites were not rerun. No builds, installs, process/loop changes, INDI
commands or notebook execution occurred on the instrument. The installed PO4AO
binary and config SHA-256 values remained unchanged:
`048af3ba45a1c2353e2712cd29da080330225307378142610310adbef0a28553` and
`51b3badd85fe660a928b7f50b4bcb43aada802437e286c3e2def589a7145bc38` respectively.


## Independent NN control ownership and section labels, 2026-11-03

The user rejected the trainer dependency for NN operation. The earlier companion-based
NN contract was therefore incorrect and is superseded. One GUI remains; `po4aoTrainer`
is optional for notebook use and has no NN role. The user also requested explicit
section titles and destination/action labels.

Implemented:

- Configure the NN device independently in the launcher (optional third device argument,
  default `nnReconstructor`); discover its catalog, command definitions and advancing
  `reconstruction_status` heartbeat directly. Trainer absence/deletion/remapping cannot
  remove NN capabilities. NN restart invalidates only NN capabilities.
- Move `nn_activate`, expected-source `reconstructor`, and `link_modulation` into the
  native NN app. Reuse its existing enable/quiescence/model loader, with serialized
  compound phases in `appLogic`. No new real-time thread, numerical kernel, speedup,
  loop check, or automatic downstream routing was added. Explicit low-level Off cancels
  pending NN activation; low-level On/model commands cannot bypass an ongoing handover.
- Port the FPS evidence/FIFO boundary into a small native transport. It reads the
  installed milk metadata ABI and Linux process identity, verifies completion separately
  from bounded FIFO delivery, and handles a disappearing reader without SIGPIPE killing
  inference. No writable FPS mapping or additional milk library linkage is used.
  A timed-out delivered CACAO start stays unresolved until observed RUN completion;
  late completion never causes a follow-on action. Target headers must match its milk.
- Remove the competing Python NN handover implementation and trainer discovery/config
  fields. The trainer now observes only PO4AO and no longer needs CacaoProcessTools.
  Existing managed publication ticket, Reset, worker and native/managed-route guarantees
  remain. Old trainer configs need the removed NN/writer/modulator keys removed.
- Add nnReconstructor / PO4AO headings. Use CNN Reconstructor, Set reconstructor,
  Switch to CACAO and Switch to CNN. Keep the standard Qt widgets and resource style.
- Update the single deployment guide for the existing IOC container, normal-user
  `make install`, NN plus GUI upgrade, and notebook-first/optional-trainer testing.

Validation:

- Native NN production methods with CPU SDK doubles: 545 assertions / 33 cases.
- Production FPS/FIFO transport with fixture metadata and process files: 19 assertions /
  2 cases. The fixture deliberately substitutes the milk ABI; it does not validate the
  target's installed layout. Real local FIFO and PID operations are exercised.
- Qt widget using the real socket-free publisher: 344 assertions / 23 cases; independent
  NN startup, trainer mapping/deletion, NN heartbeat expiry/restart, labels, handover
  requests, existing PO4AO route barriers and Reset semantics pass.
- Trainer suite: 99 passed plus 15 subtests, 9 existing plotting/deprecation warnings.
  NN regressions moved to the native owner's tests; obsolete Python NN fixtures were
  removed. Worker tests require ordinary local process visibility for owned-child cleanup.
- Launcher builds with the existing local Qt setup; offscreen preview inspected at
  500 pixels. The native header also syntax-checks against real libcommon INDI types
  with SDK doubles, rather than relying solely on the simplified INDI test boundary.
- Trainer wheel builds; all 18 packaged Python modules and the installed supervisor
  configuration match the checkout byte-for-byte.
- clang-format, Black and `git diff --check` pass. Full Linux/CUDA/TensorRT app build,
  installed milk ABI, real FPS/INDI handover, modulator and latency acceptance remain
  operator work. No instrument access, installation, restart or control occurred in
  this increment. AGENTS.md, dlDataCollection and PO4AO numerical code are untouched.

The milk RUNrunning semantics were checked against the prior read-only source capture
and public milk source (local temporary checkout `add4f667c501392447161f35be8b9c056ed40153`).
Production compilation uses installed headers, not that checkout. Definitions are in
[function_parameters.h](https://github.com/milk-org/milk/blob/add4f667c501392447161f35be8b9c056ed40153/src/CommandLineInterface/function_parameters.h).

Affected files for this increment:

- `apps/nnReconstructor/nnReconstructor.hpp`, `reconstructorControls.hpp`,
  `cacaoWriter.hpp`, `cacaoWriter.cpp`, `Makefile`, `README.md`.
- `apps/nnReconstructor/tests/nnReconstructor_test.cpp`, `nnTestSupport.hpp`,
  `cacaoWriter_test.hpp`, `cacaoWriter_test.cpp`, `cacaoWriterTestAbi.hpp`, `Makefile`;
  `tests/groups.dox`.
- `gui/apps/po4aoGUI/po4aoGUI_main.cpp`; `gui/widgets/learningCtrl/learningCtrl.hpp`,
  `learningCtrl.ui`, `tests/learningCtrl_test.cpp`, `README.md`.
- `apps/po4aoTrainer/xapp/po4aoTrainer/app.py`, `supervisor.py`, `switching.py`,
  `check_deployment.py`; `configs/po4aoTrainer.conf`, `pyproject.toml`,
  `tests/test_supervisor.py`, `tests/test_review_regressions.py`, `README.md`, `DEPLOYMENT.md`.
- This record and `agents/plans/2026-10/nn-po4ao-gui-requirements.md`.

## EDT/milk build conflict, 2026-11-03

The operator's exao2 build reached `cacaoWriter.cpp` and failed because the common
Makefile force-includes `libMagAOX.hpp`, which brings in EDT's `typedef int BOOL`;
milk's `CLIcore.h` declares a different `BOOL`. The compile log also showed nested
milk headers coming from a second installation supplied by mxlib's include flags.
This is a build integration defect, unrelated to sudo permissions or GPU inference.

The app-local `cacaoWriter.o` rule now removes only that forced include, puts its
selected milk include directories first, and depends on the Makefile rather than
the unused MagAO-X precompiled header. Other app objects retain their normal flags
and forced header. FPS file-size validation uses signed sizes and division/remainder
to avoid the reported signedness warnings and multiplication overflow.

Validation: 545 assertions / 33 native NN cases and 25 assertions / 2 FPS/FIFO cases
pass. A new local build regression compiles the real helper through the actual app
Makefile using conflicting EDT/milk header fixtures; it also confirms the main app
object still receives its forced header. Reintroducing either the forced-include
collision or mixed milk include order makes that regression fail for the intended
reason. The fixtures do not validate installed milk ABI or a full Linux/GPU build.
clang-format, Black and `git diff --check` pass. Installation/restart and the actual
instrument build remain with the operator.

Affected files: `apps/nnReconstructor/Makefile`, `cacaoWriter.cpp`,
`tests/Makefile`, `tests/cacaoWriter_test.cpp`, `tests/test_cacao_writer_build.py`,
`README.md`, and this record. No shared build rules or AGENTS.md changes.

## Trainer startup dependency and diagnostics, 2026-11-03

The captured startup traceback fails in purepyindi2's FIFO constructor before
trainer setup. Revision `c26a8276c95b95aef145c50439e291b782eab190` resolves a default
`ctrl_path` but calls `os.open(self.indiserver_ctrl_path)`, which is still `None`.
The failure appears only when the server control FIFO exists, so imports and the
previous socket-free property tests did not catch it. The subsequent `_telem_file`
destructor exception follows partial XDevice construction.

Use purepyindi2 `6595f5770d84fcf010223e317e07c970a8ba3f17` in the dedicated trainer
environment. The entire upstream diff from that revision to the previous pin is
the faulty automatic-registration addition. This restores the existing
configured-driver handshake without modifying XDevice or patching site-packages.
It supersedes the earlier dependency pin recorded above. The user performs the
package replacement and startup; no instrument changes were made in this increment.

The new FIFO startup regression constructs the real dependency transport in a
temporary directory with and without an `indiserver.ctrl` FIFO. Only the fixture's
allowed username and handle cleanup are substituted; no server or worker starts.
It reproduces `os.open(None)` with the former pin and passes with the selected
revision. Full trainer validation with that revision: **101 tests and 15 subtests
passed**, with 9 existing plotting warnings. GPU and live INDI startup remain
operator validation; this is transport construction and CPU validation only.

The operator also reported `NameError: name '_brief_status' is not defined`.
Local `scripts/xctrl` confirms that `peek` calls an undefined helper; this is
independent of the trainer failure. The deployment guide now uses tmux capture
for startup tracebacks and `status` for process liveness. It uses `startup` for
the failed trainer because this xctrl's targeted `restart` also selects isRTC.
No shared xctrl code was changed. The guide also carries forward the previously
agreed isolated Python installation, explicit notebook package versions, shared
directory permissions, and actual Enable PO4AO button label.

Affected files: `apps/po4aoTrainer/tests/test_fifo_startup.py`,
`apps/po4aoTrainer/DEPLOYMENT.md`, and this record.

## Trainer output prevents driver startup flush, 2026-11-03

The operator supplied process and log evidence: trainer PID 3527998 completed
setup at 13:37:44 UTC; isRTC PID 3530456 started at 13:42:37 UTC with the correct
trainer driver, PID 3530704. Its log completes the trainer `.in` FIFO flush and
then stops at `Starting flush ... po4aoTrainer.out`. No heartbeat is served.
This confirms that the driver is registered but cannot complete its output flush.

`INDI/xindidriver/xindidriver.cpp::flushFIFO` drains until a one-second quiet gap.
The supplied trainer config used a 0.2-second main-loop interval; each iteration
unconditionally publishes status, configuration and progress, so the output FIFO
never becomes quiet. A temporary local C++ harness compiled the production
`flushFIFO` function unchanged and used a real temporary FIFO: 5 Hz writes kept
the flush active for the entire four-second probe; stopping writes let it finish
one second after the last write. A two-second update interval permits completion.
Only local temporary files/processes were used; the instrument evidence was
provided by the operator.

Change the supplied `sleep_interval_sec` to 2.0 and document the same one-line
change for existing installations. This is the smallest configuration correction;
no shared driver, framework, GUI or numerical implementation changes. Supervisor
polling and periodic status/publication handling now occur every two seconds,
still within the GUI's five-second heartbeat window and the default 30-second
worker heartbeat timeout. Command callbacks run separately; real-time C++
inference and the numerical worker's training loop retain their own cadence.

The operator should restart only po4aoTrainer with shutdown/startup, keeping isRTC
running. The updated interval is parsed from the distributed TOML config locally.
Recovery and heartbeat advancement on the instrument remain to be verified.

Affected files: `apps/po4aoTrainer/configs/po4aoTrainer.conf`,
`apps/po4aoTrainer/DEPLOYMENT.md`, and this record.

## Operator penalty settings, 2026-11-03

The operator identified the intended notebook values as `ACTION_PENALTY_WEIGHT =
0.01` and `POLICY_UPDATE_PENALTY_WEIGHT = 0.5`. These already map to the managed
schema's `[RL] loss_function_penalty` and `[policy_update] weight`, respectively.
The former weights normalized action magnitude in `train_policy`; the latter
weights normalized action changes from the acknowledged reference policy in
`PolicyUpdatePenalty.penalty`. Both are in `configuration.HOT` and are applied
between training iterations without resetting models or replacing Adam.

The distributed example had values 0.2 and 1.0 and did not identify the notebook
variable names. Correct the example to the operator's values, label both mappings
in comments, and document them in the single deployment guide. Keep schema keys
and numerical implementations unchanged. Historical CPU parity fixtures and the
explicitly synthetic example notebook retain their recorded reference settings;
this correction is to the managed deployment configuration.

Validation: the full supplied training TOML parses through `TrainingConfig`; a
local CPU session accepts both changes through `needs_reset`/`apply_hot`, updates
the action coefficient and regularizer weight to 0.01/0.5, and retains the same
optimizer object. No instrument file was changed and no live training was run.

Affected files: `apps/po4aoTrainer/configs/po4ao_training.conf`,
`apps/po4aoTrainer/DEPLOYMENT.md`, and this record.

## Notebook settings comparison and public penalty name, 2026-11-03

Compared the operator's pasted nonlinear settings source (attachment
`0f9e54a5-6a91-4767-a201-8a5d434c7863/Pasted text.txt`, SHA-256
`976d2c417f8196ebffef3cf993055894e764729cce614796c221c4ccba66113a`) using
literal AST assignments without executing the attachment. All 33 explicit
constants were mapped to the managed configuration. The pasted source assigns
both penalty constants twice: the later FAINT values 0.1 and 1.0 are effective.
The operator explicitly requested keeping the earlier 0.01 and 0.5 values now.

Six other defaults differed and are aligned in the distributed example:

| Managed setting | Previous | Notebook setting |
| --- | --- | --- |
| `NN_models.training_batch` | 256 | 128 |
| `nonlinear.channels` | 8 | 32 |
| `nonlinear.shared_layers` | 2 | 3 |
| `training.model_publish_interval` | 100 | 50 |
| `training.first_model_publish_iteration` | 250 | 150 |
| `training.max_candidate_action_rms_ratio` | 1.5 | 5.0 |

The other explicit settings match. Both networks are constructed from the same
`nonlinear` block by `build_nonlinear_models`, so the notebook's duplicate
`NN_models.dynamics_architecture` and `policy_hidden_channels` assignments do not
require additional managed keys. Inherited `po4ao_config` contents were not
provided: no claim is made about parity of stream paths/dimensions, gains,
command limits, compute devices or the diagnostic sample rate. The runtime
contract remains supplied by the installed C++ configuration.

Rename the public TOML field to `[policy_update] penalty`. At the existing
configuration boundary, accept the former `weight` spelling for deployed files
and reject both spellings together. Resolve either spelling to the numerical
notebook core's original `weight` key. This keeps existing notebooks and recorded
resolved snapshots compatible, preserves configuration hashes for spelling-only
edits, and reuses the established hot-update path without changing numerical
code or adding another migration layer. The updated app must be installed before
an operator changes an existing file to the new spelling; older apps reject it.
Changing network channels/layers requires Reset, even though the penalties and
other changed operational settings are hot fields.

Regression coverage exercises scalar and per-mode penalties, equal/conflicting
duplicate names, invalid signs/counts/non-finite values, stable revisions, and a
real CPU regularizer whose hot-updated cost halves without replacing the policy,
optimizer or normalization. Per-mode tests exposed the pinned xconf parser's lack
of PEP 604 union handling; use its supported `typing.Union` spelling as in the
framework, without changing dependencies.

Validation: 111 tests and 15 subtests passed, including the existing notebook
numerical parity suite; 9 existing plotting warnings remain. The full distributed
TOML also passed the production loader with an explicit synthetic runtime fixture.
AST comparison now finds only the two intentional FAINT-penalty differences.
Black and `git diff --check` pass. No instrument operation or installed-file change
was performed; full GPU execution with the larger notebook network is unverified.

Future request recorded: expose action and policy-update penalty editing in the
GUI. This increment changes the config and defaults only; it adds no GUI controls.

Affected files: `apps/po4aoTrainer/xapp/po4aoTrainer/configuration.py`,
`apps/po4aoTrainer/configs/po4ao_training.conf`,
`apps/po4aoTrainer/tests/test_configuration.py`, `apps/po4aoTrainer/README.md`,
`apps/po4aoTrainer/DEPLOYMENT.md`, and this record.

## 2026-11-03: restore notebook replay reuse and add a progress notebook

The operator reported slow managed training and requested the notebook's behavior
and an observing notebook. Local comparison with the previously captured instrument
notebook (`cb8b68310b2b95e1506526b8bdfb87064f484b874c8fbdeab7bf0884528f3da6`)
found that the notebook keeps training on cached replay between publications. The
managed session instead rejected repeated replay identities, and the worker copied
both buffers every iteration. No instrument command was run for this review/change.

- Restore repeated optimization on complete cached replay. Copy/validate stream
  arrays only when a publication changes, and convert each snapshot to chronological
  controlled-mode arrays once. Preserve the existing stable-copy checks, incomplete
  data handling, collection-generation cancellation, and newer-warmup requirement
  after invalidation. Historical episodes retain their original age; repeated
  optimization does not create additional experience or repeat dynamics pretraining.
- Bound Torch CPU threads by the process's allowed CPU count after the existing
  system-cpuset intersection, retaining any smaller thread limit. Log the resolved
  GPU/CPU resources. This addresses thread oversubscription without moving work onto
  real-time cores or changing GPU assignments; actual latency remains unmeasured.
- Export only the policy/dynamics models, without constructing another session,
  extra policy copy or Adam objects. Keep cancellable child export and the existing
  candidate/activation handshake; the notebook also exports synchronously. No
  asynchronous publication redesign or NN inference change is included.
- Compress each completed diagnostic snapshot once, after publication outcome (or
  immediately for an iteration that does not publish). Keep the append-only JSONL
  issuance/outcome records, so publication status can be newer than plot data while
  building. Cancelled exports retain a diagnostic snapshot with cancellation status.
- Add sampled replay, diagnostics, export and activation timings. Training time now
  includes chronological-array conversion. Snapshot writes are not separately timed;
  the elapsed interval between diagnostic measurements includes writes and pauses.
- Add `notebooks/po4ao/watch_training.ipynb`: file-only observation of the latest or
  explicitly selected run, existing training/debug plots, measurement age, model IDs,
  publication result, measured/predicted costs and timings. It tolerates an incomplete
  final JSONL line and prefers the last record for each iteration. A new empty run
  never falls back to a previous run's plots. Refreshing unchanged snapshots reuses
  figures. Automatic refresh is opt-in; interrupting it has no effect on training.

Validation: **115 tests and 17 subtests passed**, including the pinned first-iteration
numerical/Adam fixture, repeated cached updates, unchanged history count, coherent
cache refresh, recollection/pause/resume, one snapshot per publication, export-only
model restoration, and execution of the observing notebook cells on saved CPU data.
Nine existing plotting warnings remain. The local suite uses the deployment-pinned
purepyindi2 `6595f577`; the older local environment reproduced the already documented
FIFO-constructor bug. Real child-process tests required local macOS process-list
permission. No instrument access was involved. Black, notebook JSON/Python syntax
checks and `git diff --check` pass; no C++ files require clang-format in this change.

Delivery: preserve the current training configuration and update only the Python
trainer. Its restart loses in-memory Python models/Adam; the operator starts a fresh
managed session. The deployment guide provides the notebook copy/kernel commands.
The agent prepares the notebook and Git bundle locally because `AGENTS.md` prohibits
agent commands on exao2. GPU training speed, TensorRT export, physical performance
and NN latency still require operator measurements; no speedup factor is claimed.

Affected files: `training.py`, `worker.py`, `po4ao_runtime.py` under
`apps/po4aoTrainer/xapp/po4aoTrainer/`; their three regression test files;
`notebooks/po4ao/watch_training.ipynb`; this execution record. The subsequent
documentation commit updates `apps/po4aoTrainer/README.md` and `DEPLOYMENT.md`.

## 2026-11-03: live parameters and paired saved-model library

Implemented the approved parameter/library plan from attachment
`7ffd48f8-a057-4714-a5c5-fe730cf748e5`, based on `881ce8cb`, with the operator's
correction: the delta-policy penalty follows the currently acknowledged effective
policy, never a permanently frozen saved initialization. Work stays on
`rlandman/nn-po4ao-gui`; AGENTS.md, native PO4AO, NN and instrument configuration
are unchanged. No instrument commands, access, installation or restart were used.

Behavior and implementation decisions:

- Three `statusLineEdit` fields share one horizontal row in the existing widget:
  command regularization, delta policy regularization, and policy learning rate.
  Defaults/readback come from the trainer, not GUI constants. Enter sends a numeric
  `target`; `current` remains acknowledged state. Busy and requested values are
  visible in status/tooltip. Nonuniform modal delta penalties remain file-controlled.
  Standard font/resource styles are preserved; default width is 540 px and wrapped
  headings reserve two lines to avoid clipping. Native/NN controls remain independent.
- Session overrides merge into the latest validated configuration. Same-field valid
  file edits supersede only that override. Reset, loading, pause and disable retain
  overrides; app restart restores file values. Configuration tickets accompany hashes
  so A → B → A while paused cannot mistake an earlier A acknowledgement for completion.
  The worker acknowledges only at an iteration boundary, retaining models, Adam and
  replay. No system configuration writes or dependency additions are introduced.
- `library.py` adds only the immutable bundle/catalog functionality. The configurable
  library defaults to `<artifact_root>/saved_engines`. IDs are sanitized label + UTC
  timestamp + UUID suffix; display names include original iteration. Manifest version 1
  records provenance and SHA-256 hashes. Complete engine/metadata/NPZ files are copied
  from one pinned acknowledged publication, verified, then atomically renamed into
  view. Save uses its own operation queue, permitting concurrent learning/publication.
  Reset and restart never delete the library. Missing/malformed entries are skipped.
- Set policy validates compatibility and hashes, pauses learning and settles exact
  outstanding tickets, then prepares a separate CPU/GPU training session using the
  existing worker/exporter. Strict tensor loading restores both networks and their
  normalization buffers, plus policy-update scales. Current hyperparameters prevail;
  optimizers, replay, anchors, diagnostics and iteration start fresh. The old session
  remains paused until exact new activation succeeds. This can temporarily use more
  training GPU memory; an allocation/export failure leaves the previous session paused.
- Set policy preserves controller enable state, does not invoke Reset or alter loop,
  reconstruction, modulation or CACAO routing, and does not start learning or noise.
  Loaded dynamics bypass cold pretraining and the cold first-publication delay.
  Source bundle/session/iteration remain separate from the new session/generation.
  A newer ready warmup generation is required before learning resumes. Explicit
  Enable (if needed) → Recollect warmup → Resume uses noise on the loaded policy.
- The worker can follow active-ID readback at every gradient checkpoint, refreshing
  anchor reference actions only when the acknowledged policy changes. Publication
  outcomes carry fresh effective-policy readback; heartbeat reads are ordered with
  them under the supervisor lock. Temporary fallback retains the known network and
  pending candidate so return to PO4AO does not discard the reference. Unknown external
  networks are not guessed. The original numerical equations/normalization remain.
- Failed preparation/reload does not admit the new worker. Reset invalidates staged
  generations and stops the owned preparation process. An interrupted issued load
  can still complete remotely; when that would leave old training networks against a
  different engine, a persistent initialization-required flag blocks Start until an
  explicit successful Set policy or Reset. No unrelated ticket can clear this state.
- Calibration identity is recorded if available and otherwise explicitly unknown.
  Matching architecture/mode/history dimensions is not a calibration-compatibility
  claim. The operator continues to own notebook/app use and instrument setup.

Validation covers exact CPU policy/dynamics outputs and buffers after save/restore,
retained normalization, fresh optimizers/replay, warm-start cadence, same-iteration
reference changes, scalar/file/per-mode parameter precedence and acknowledgements,
concurrent Save/publication, duplicate labels, incomplete/corrupt/incompatible
bundles, preparation/reload failures, Reset and post-activation cancellation,
restart persistence, and real owned worker preparation/pause/config protocols.
Qt tests use the actual widget and socket-free publisher, including staged stable
catalog IDs across deletion/redefinition, numeric target-only sends, a real name
input dialog, pending/rejected/disconnected states, warmup/initialization gating,
and unchanged native/NN independence. The real widget preview was inspected.

Affected functional files:
- `apps/po4aoTrainer/xapp/po4aoTrainer/{app,configuration,library,po4ao_policy_update,process,supervisor,training,worker}.py`
- `apps/po4aoTrainer/configs/po4aoTrainer.conf`
- `apps/po4aoTrainer/tests/{test_library,test_review_regressions,test_supervisor,test_worker_process}.py`
- `gui/widgets/learningCtrl/{learningCtrl.hpp,learningCtrl.ui}` and its two test sources/project files
- `gui/apps/po4aoGUI/po4aoGUI.pro` and this execution record.

The following documentation commit updates the trainer README, widget README and
consolidated deployment guide. Deployment updates only the trainer in its existing
environment and the GUI in its normal IOC container, and provisions the library
with existing xsup/magaox permissions. The bundle contains feature commits since
baseline `88271a3e` for existing installations of this branch. On-sky release still
requires publishing/recording tested source/config revisions under the handbook's
version-control policy; local CPU/Qt tests do not establish CUDA/TensorRT or live
instrument performance.

Final local results: **133 Python tests + 17 subtests passed**, including the pinned
notebook numerical/Adam fixture and real child-process tests. Nine existing plotting
warnings remain. **403 Qt assertions in 26 cases passed**; the separate actual-widget
preview passes five assertions. The production launcher builds with the existing
local Qt 5.15.15 setup and a command-line override for unavailable mxlib pkg-config
linkage; production project settings retain the standard linkage. Existing SDK,
shared-header override and deployment-target warnings remain. Black, clang-format
and `git diff --check` pass. CUDA/TensorRT execution, actual GPU memory headroom,
live INDI deployment and closed-loop improvement remain operator acceptance items.
