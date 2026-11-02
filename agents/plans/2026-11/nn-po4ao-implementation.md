# NN / PO4AO implementation record

Started 2026-11-02 from the user's approved [requirements](../2026-10/nn-po4ao-gui-requirements.md). This is an execution record, not a replacement for the agreed behavior.

## Branches and baseline

- Fixed baseline branch: `rlandman/nn-po4ao-baseline`, commit `88271a3e6ec3c3add36a4983d806f05501108f20`.
- Child implementation branch: `rlandman/nn-po4ao-gui`, created directly from that baseline.
- Isolated managed checkout: `/Users/ricolandman/.codex/worktrees/nn-po4ao-gui/MagAOX`.
- The original dirty checkout remains intact. No instrument access, deployment, push or PR was performed.

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

Local verification: the real widget/launcher built with Qt 5.15.15, and 102 assertions in ten widget cases passed. The optional preview case passed four assertions and rendered the actual widget using the normal resource stylesheet. No socket is constructed in these tests. The local launcher build omits the unused mxlib pkg-config linkage via a command-line qmake override; the production project retains the common build rules. Qt reports that the installed macOS SDK is newer than its tested SDK, and the existing shared publisher emits missing-override warnings. Native Linux/instrument deployment remains untested.

Read-only exao2 evidence:

- All six numerical helper hashes match `source_manifest.json` exactly.
- Instrument notebook SHA256: `cb8b68310b2b95e1506526b8bdfb87064f484b874c8fbdeab7bf0884528f3da6`. Its source differs only by commenting the initial random identity probe, adding a direct warmup plot, and whitespace; the training cell's numerical expressions are unchanged. The packaged core already leaves that optional probe to notebook code. The local parity fixture still intentionally includes the pinned notebook's probe and records its seed.
- Installed milk source revision: `336160d054df5773a49d603ba987bea74692fcda`. `CacaoProcessTools.fps` exposes `RUNrunning`, `md().runpid`, `runpidstarttime` and status. `RUNrunning` checks PID liveness and distinguishes an unclean exit; clearing `CMDRUN` is only a stop request. PID identity/reuse still needs attention in the eventual adapter.
- `FPSCONNECT_SIMPLE` writes bookkeeping metadata and connection logs, so no FPS object was instantiated during this read-only inspection. Only existing files, hashes, source definitions and directory/revision information were read. No INDI requests, control commands, process changes, installs or remote file writes occurred.

Remaining acceptance is unchanged: managed configuration/worker/publication, C++ current-policy recollection, the companion's real operations and deployment are not supplied by a rendered frontend. This increment completes the native panel and its offline request/readback behavior; it does not claim working instrument control.
