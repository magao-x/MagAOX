# Shared NN / PO4AO GUI: requirements and implementation plan

Created: 2026-10-28. Revised and reviewed: 2026-11-02. Status: implementation started on 2026-11-02; baseline committed, the first shared numerical-core increment implemented, the native Qt frontend built and tested, and C++ current-policy recollection implemented with CPU boundary tests. Remaining phases and acceptance checks are tracked in `../2026-11/nn-po4ao-implementation.md`.

Prepared by GPT-6 from the user's shared-GUI request and subsequent decisions in this chat. This document is the implementation handoff. Sections 1–6 record the agreed behavior and explicit engineering defaults; sections 7–11 describe implementation, evidence, and acceptance work; section 12 records the final review. It supersedes conflicting proposals in `agents/plans/2026-09/nnReconstructor-gui.md` and the earlier interactive preview, particularly their loop-open and PO4AO-reset requirements.

Packaging revision, 2026-10-28: use a first-class installed `po4aoTrainer` Python application alongside the C++ `po4ao` app. This replaces the earlier `learningCtrl` backend and its home-directory worker/helper proposal. The GUI keeps its proposed name `learningCtrlGUI`. Production execution must not depend on a personal checkout, editable install, or home-directory interpreter.

Final scope decisions, 2026-11-02: start from `rlandman/po4ao-pr`, import the newer NN correctness/control implementation from `rlandman/nnreconstructor-correctness-controls` at `4353737e331671cd6a9b6aad5d92039c27f26715`, and import the newest committed `dlDataCollection` separately. Exclude the NN speedup changes, not the newer control functionality. Preserve notebook training and plots using the same numerical core as the installed app. The operator manages notebook/app training responsibilities; do not add notebook detection, session locks, leases, or ownership enforcement. Reuse current upstream GUI infrastructure. Include C++ recollection under the current active policy with exploration noise.

## 1. Scope and execution boundary

Build a compact, shared MagAO-X Qt GUI for NN engine selection, CACAO/NN switching, PO4AO control, warmup recollection, and nonlinear training. Reuse the existing GUI and INDI frameworks. This is an operator tool, not an autonomous controller.

The user explicitly decided:

- Do not check or change AO loop open/closed state before NN engine changes or CACAO/NN switching. The operator manages the loop.
- Do not require PO4AO disabled, learning paused, or a PO4AO Reset before reconstruction changes. Do not perform those actions automatically during reconstruction changes. The operator manages compatibility and comparisons.
- Keep actual operation-completion checks: completed NN disable, successful model load, acknowledged activation, valid requests, connected devices, and serialized conflicting operations. These are not loop-state interlocks.
- Do not implement NN hot-swapping or bring in NN speedups. Reuse the newer correctness/control branch's existing disable, load, enable interface (section 7.6); do not rebuild it on the old NN app in `po4ao-pr`.
- Disable PO4AO pauses learning but preserves the Python process and models. Reset terminates the training process and clears the policy session.
- Warmup recollection runs under the current controller with exploration noise and leaves that controller selected afterward.
- Omit modal gains, tuning panels, plots from the compact GUI, and automatic loop management. Preserve the training/debug plots in the notebook.
- The operator chooses and manages notebook versus app training. GUI learning controls affect only the app's own worker; they do not pause, reset, terminate, or adopt a notebook kernel.

The user subsequently authorized read-only exao2 inspection on 2026-11-02; that limited exception is recorded in the implementation record. No remote control, process changes or file writes are authorized.

All agent development and tests run on a local workstation or an isolated non-instrument environment. Under `AGENTS.md` rule 0, the agent must not execute on exao1/exao2/exao3/exao5, SSH there for inspection, or issue INDI commands through a tunnel to the instrument. An operator performs instrument installation, process changes, routing checks, and acceptance tests. The conventional PO4AO/training software is not an LLM agent; its operator-run deployment is the intended eventual product.

Implementation was separately authorized on 2026-11-02, including creating a baseline branch and a child implementation branch. Instrument installation, launch and operation remain operator-run under the boundary above.

## 2. Sources inspected and revision boundaries

These are local source observations and GitHub checks on 2026-11-02, not claims about installed binaries or running devices.

| Source | Snapshot / finding |
| --- | --- |
| Review checkout, not implementation base | Branch `rlandman/nnreconstructor-speedups`, HEAD `41f21c44d95b4acc2c6870c7901db6f01f140068`. Preserve unrelated modified/untracked files. Do not merge this branch into the implementation. |
| Required implementation base | `rlandman/po4ao-pr`, commit `4ff2634124a5a6fcf428e22ce2807dc31b21b823`; GitHub branch confirmed identical. Replace its old NN subtree with the pinned newer NN implementation below. |
| Required NN implementation | Local `rlandman/nnreconstructor-correctness-controls`, `4353737e331671cd6a9b6aad5d92039c27f26715`: engine discovery, model selection, acknowledged enable/disable, model status and correctness fixes. Synchronous `executeV2`, with no pinned transfers, CPU performance profiles or CUDA graphs. `rlandman/nnreconstructor-speedup-baseline` at `5c7e91da` has identical NN files. GitHub branch search currently exposes no branch with these NN names; retain the local donor or publish it as part of implementation handoff. |
| Newest committed local collection implementation | Take only `apps/dlDataCollection/` from `41f21c44d95b4acc2c6870c7901db6f01f140068`, plus the collection-specific test registration/build/docs changes. This includes metadata changes through `5c7e91da` and later Start-preflight changes `fa31b095`, `04658ec2`, `41f21c44`; the separate collection branches stop earlier. |
| Upstream framework reference | GitHub `dev` at `5a45f1c7b73faeb7dbe63dda18209e15306a2be8`, including current GUI publisher/discovery and offline tests. Its collection header is older than the committed local collection implementation. Import needed GUI changes with dependencies, not a blanket branch merge. |
| Local training repository | Sibling `../PO4AO_2026`, branch `rlandman/po4ao-linear-comparison`, HEAD `b9076dd7aaf00ff7cd4f7f644e8f40dc1ff10a4f`; nonlinear notebook and helpers inspected statically. No Git remote is configured; a newer instrument copy has not been verified. |
| Operator's intended training source | `exao2:/home/rlandman/PO4AO_2026/po4ao_notebook_nonlinear.ipynb`, supplied by the user. Equality with the local copy is unverified. |
| Instrument configuration | Local sibling `../config` only. It contains `dlReconstructor` in `isRTC.conf`, commented NN process entries in `proclist_RTC.txt`, and PO4AO configuration. Do not assume this is the installed configuration or hard-code the NN device identity. |

Before implementation, switch to the `po4ao-pr` baseline in a clean checkout and create `<active-username>/nn-po4ao-gui` from it. If the current checkout remains dirty, use an isolated checkout rather than stashing, overwriting, or carrying unrelated edits into the new branch. This planning revision does not itself switch branches or integrate application files.

Import the collection subtree from the exact committed revision above, not the working tree. Merge only its entries in `tests/Makefile.one`, `tests/tests.list`, and `tests/groups.dox`, preserving PO4AO test registration. Retain its engineering documentation (`dlDataCollection-improvements.md` and `dldata-observation-metadata.md`) as appropriate to explain the imported code. In particular, copying the donor `tests/groups.dox` wholesale would remove the PO4AO group. Do not import NN source, configs, export scripts, CUDA-graph work, or untracked training-progress tools from the donor branch. Existing standalone collection helpers are baseline code, not a precedent for adding new ad hoc scripts under `apps/`.

Import `apps/nnReconstructor/` from the pinned correctness/control donor, including its tests and relevant config/export documentation, separately from the collection import. This donor already includes the FP32-only, single-input simplification; do not restore obsolete FP16 or explicit-TT paths from `po4ao-pr`. Preserve its app-local tests and post-publication acknowledgements. The reviewed `frameGrabber.hpp` is identical between this donor and `po4ao-pr`; no new framegrabber hook is required for this import.

Verify that the imported collection and NN subtrees match their respective donors before GUI work, and that collection/PO4AO registrations and NN app-local tests survive. Refresh upstream references if implementation starts later, recording any changed source decision rather than silently broadening this import.

## 3. Compact interface

Use approximately a 500-pixel-wide compact window; final height follows the existing Qt font and layout metrics. Use layout managers rather than absolute positioning. Long filenames expand in the existing combo popup and remain available in tooltips.

```text
NN / PO4AO
Connection: Ready

Engine: [ selected-engine.trt                  v ] [Enable]
[Switch reconstructor]          Active: NN
[ ] Link modulation            Modulation: Off
----------------------------------------------------------
[Enable/Disable PO4AO] [Reset PO4AO]    Integrator / PO4AO
[Recollect warmup]                    Ready / Collecting
[Start/Pause/Resume learning]          Training state
Policy: <active or retained ID>        Iteration: <n>
Status: <operation, config update, or error>
```

The slash-separated labels above denote one state-dependent button, not literal text or multiple buttons. Changing action labels are an established pattern in `pupilGuide.hpp` (`set test` / `zero test`, and movement-target labels). Derive labels from acknowledged backend state. During a transition, disable the conflicting control and show progress; do not optimistically label it successful.

- Model dropdown selection is local pending choice only. It must not send a load request until Enable is pressed.
- Keep requested, loaded, active, and retained models distinct. A selected dropdown value is not an active-model report.
- Show the effective controller separately from policy identity: a retained nonzero model ID does not establish that the policy is controlling the output.
- Show `Unknown`, `Transition`, `Disconnected`, or `Conflict` when those are the actual states; never infer CACAO active merely because NN is disabled.
- No loop control or loop-state dependency is required. An optional read-only loop label must never gate actions; omit it from the minimum design.
- Use the existing `magaox.qss`, font sizing, status styles, standard buttons, and icons/resources. Do not introduce a new visual theme, web UI, or new widget framework.
- The preview previously shown in the chat is illustrative and obsolete where it blocks reconstruction changes or omits recollection noise. This text governs implementation.

## 4. Button requirements

### R1. NN engine dropdown and Enable

Populate the dropdown from the selected NN branch's existing `model` switch-vector definitions and labels (section 7.6). Reuse its startup discovery of compatible engine candidates in configured `engineDirs`, including configured `engineName`. Use element identifiers from INDI, not filenames constructed into shell commands. Live directory rescan is out of scope. Reconnect/property-redefinition handling must work.

Enable performs one serialized transaction:

1. Validate the selected catalog entry and capture the requested model identity.
2. If NN is the current writer, request `enable.toggle=Off` and await completed disable, including its acknowledged final zero publication. If CACAO is the writer, the selected NN engine may be prepared while NN remains disabled and CACAO continues; stop CACAO before enabling NN.
3. Send the selected `model` request. Await the requested filename in `model_status.loaded`, completion of loading, and absence of a runtime/load error. Do not equate the selected switch with loaded status.
4. If handing over from CACAO, verify its writer has stopped; apply linked modulation if requested.
5. Request NN enable and wait for `model_status.active` and an active/fresh-result acknowledgement before reporting success.

If the same engine is already active on NN, return success without interrupting output. If it is already loaded but disabled, skip redundant loading. Do not alter PO4AO, training, gains, routing, or loop state.

Preserve the selected correctness/control branch's disable behavior: quiesce inference, publish one zero modal vector, acknowledge that publication, then stop publishing. This behavior predates the speedups and must not be removed as part of their exclusion. The loop may remain logically closed during the loading gap, as requested. This is not continuous correction or hot-swapping. This supersedes the intermediate proposal to add a no-zero disable to the old `po4ao-pr` NN app.

A load/enable failure must report the resulting state, leave output disabled, and stop the sequence. Do not enable a different engine silently or perform an unrequested rollback. Correctly handle failed deserialization, allocation, incompatible tensors, and partial cleanup in the new sequential loader.

### R2. Switch reconstructor and optional modulation

Toggle between CACAO `wfs2cmodeval-1` and the already loaded NN model. Do not apply an uncommitted dropdown selection. Explicit target requests, computed from confirmed state, prevent double-clicks or two GUI clients from toggling twice unintentionally. Reject a stale target-state assumption rather than guessing.

- NN to CACAO: complete NN disable; if linked, set modulation frequency to 2000 Hz, radius to 3, and modulation state to the configured modulated state; await readback; start and observe CACAO.
- CACAO to NN: stop and observe CACAO; if linked, request unmodulated SET; await readback; enable the loaded NN and await active output.
- The existing script uses `modState.target=3` for unmodulated SET and `4` for modulation. Keep these as explicit configured mappings checked against the intended device, not generic state-number assumptions.
- `Link modulation` defaults Off. Toggling the checkbox itself does not change modulation. The option applies to both Switch reconstructor and an Enable action that hands over from CACAO to NN; an NN-to-NN engine replacement leaves modulation unchanged.
- With linking Off, issue no modulation writes. Missing modulation readback must not block an otherwise valid unlinked switch.
- No loop-state checks, PO4AO-state checks, automatic learning pauses/resets, or loop open/close commands belong in this transaction.

Use finite timeouts for process/FIFO/INDI operations and prevent simultaneous NN and CACAO writers. Observe actual state after commands; FIFO delivery or a fixed sleep is not evidence that the outgoing writer stopped. Do not confuse this writer-ownership requirement with the explicitly excluded operator-state checks.

### R3. Enable / Disable PO4AO

Enable selects the latest successfully published, compatible engine known to the managed app's current non-reset policy session. If that exact policy is already resident, reactivate it; otherwise reload it and await activation. A pending/partial/failed export is not the latest published policy. Verify model identity rather than relying on modification time alone. The app does not automatically adopt an external notebook's file or optimizer session; notebook publishing remains an operator-managed workflow.

If no policy is eligible, enable the existing PO4AO startup/fallback path, permit initial exploratory collection, and show fallback/waiting-for-policy. Do not resurrect an older engine file after Reset. Enabling never starts/resumes learning automatically.

Disable returns to integrator fallback and pauses learning, retaining the policy, dynamics, optimizer state, and Python process. Prevent new reloads immediately, settle/cancel outstanding publication consistently, and acknowledge completion only when the fallback is confirmed and learning is paused. Keep the selected policy eligible for later Enable.

The normal fallback is PO4AO's configured proportional modal output followed by downstream CACAO integration. Preserve configured gain and limits. Do not claim that control Off is an unconditional bypass or that it changes downstream routing.

### R4. Start / Pause / Resume learning

One button follows these states:

| Acknowledged state | Label | Action |
| --- | --- | --- |
| No training session after startup/Reset | Start learning | Launch the configured Python worker and initialize the nonlinear models once. |
| Running or waiting for data | Pause learning | Cooperatively pause at a bounded safe point. |
| Paused, worker alive | Resume learning | Continue the same model/dynamics/optimizer session. |
| Starting, pausing, resetting, or config transition | Disabled | Display the transition. |
| Worker unexpectedly exited | Disabled/error until explicit recovery | Do not silently claim the prior in-memory session can resume. |

Pause preserves policy/dynamics weights, Adam state, normalization, counters, publication reference/pending bookkeeping, and eligible replay state. It stops gradient updates and new policy publication, not real-time inference. An already-issued reload may finish before Pause is acknowledged; after acknowledgement no late policy change from that trainer is allowed. Do not implement pause using SIGSTOP or by terminating/restarting Python.

Poll cooperative commands during long pretraining/gradient loops and data waits, not only after an entire training run. External TensorRT builds need bounded process handling. Waiting for data does not consume the requested number of completed training iterations. Reaching the configured iteration limit leaves a resumable paused/completed session; increasing the limit and Resume continues it.

Start/Resume does not change loop state, reconstruction, or PO4AO enable state. If fresh/valid data are unavailable, report Waiting rather than training indefinitely on an unchanged snapshot. Paused workers do not append duplicate replay episodes.

### R5. Reset PO4AO

Reset is the explicit destructive action for the in-memory learning session, not file deletion:

1. Block new publications from the managed worker and invalidate the supervisor's old-session eligibility. This does not block an operator-run external notebook.
2. Request integrator fallback promptly, without waiting for a long training iteration or model build.
3. Terminate the worker and its owned build subprocesses through bounded cooperative shutdown, then owned-process termination if necessary. Never use a broad process-name kill.
4. Clear the runtime learned policy/history through the existing reset generation mechanism, and ensure every pre-reset pending load is rejected or drained.
5. Acknowledge Reset only after worker exit and runtime reset completion. Reset may remain pending if the input frames needed by the C++ acknowledgement are absent.

The next Start initializes new models; the button reads Start learning. Retain logs and saved artifacts. Preserve existing warmup data unless Recollect is requested; compatibility checks still apply. Old files cannot become eligible merely because they remain at the configured engine pathname. Record/reset the eligible publication identity and session generation atomically.

### R6. Recollect warmup under the current controller, with noise

1. Pause the managed worker's learning/publication if it was running; remember whether it should resume. The operator handles any notebook trainer independently.
2. Capture the current effective controller and policy identity. Preserve it and its temporal history.
3. Invalidate the previous warmup snapshot and start a new collection generation. Reset collection counters, partial training episodes, and exploration schedule, not controller history or selected policy.
4. For every collection frame, compute the current controller output, add the existing configured exploration sequence/schedule on the controlled modal prefix, then apply existing finite-value and absolute-command limits. Add noise only once. Higher modes remain pass-through.
5. Store the emitted, post-noise/post-limit command in controller history and replay so the training input represents what PO4AO actually sent downstream. Do not claim this is the physically applied DM command after downstream integration/clipping.
6. Publish a complete new snapshot using existing paired stream/metadata counter conventions. Report progress and completion; the request switch returning Off is not sufficient evidence that collection finished.
7. Remove exploration at completion, continue the same selected controller, refresh training snapshot/anchor caches, and resume learning only if it was running before collection.

Collection must also work under integrator fallback while `control_enable` is Off; the current early-return path otherwise prevents this. Do not temporarily enable a retained learned policy to collect integrator data.

Do not recompute an established model's normalization or reset trained weights/optimizer state during recollection. For a newly initialized session, preserve the existing normalization-from-warmup behavior. If an external client changes controller/policy, a stream discontinuity breaks collection, or inference failure forces fallback from the captured learned policy, preserve normal runtime fault handling but mark the collection interrupted/incomplete rather than combining incompatible samples into a supposedly fixed-policy dataset. End recollection noise on interruption as well as normal completion. Do not block the user's reconstruction buttons based on learning/PO4AO state; report interrupted collection if those changes disrupt it. Do not silently resume learning after an incomplete collection.

## 5. Training configuration without application restart

Use a separate `/opt/MagAOX/config/po4ao_training.conf`, parsed as TOML with the existing Python `xconf` convention despite the `.conf` suffix. The companion app has its normal `po4aoTrainer.conf` for device names, command timeouts, and data locations. Launch its packaged worker using the same installed interpreter and module namespace, not a configurable script in a personal checkout. Keep these roles separate from the C++ `po4ao.conf` runtime/tensor/stream contract.

The companion polls the training-config file at a modest interval and parses an immutable candidate configuration when it changes. It also checks on Start/Resume. Use the existing typed loader/validation facilities; do not add a filesystem-watcher dependency or use Python module re-importing as a reload mechanism.

Parse and validate the whole candidate before applying anything. An incomplete save, unknown key, or invalid value leaves the last good config active and reports an error. Compare content revisions, not only timestamps. Prefer atomic replacement when editing. Keep desired, applied, and pending-reset revisions distinct and expose them through status/logs. A worker acknowledges the revision it actually applied.

| Settings | Application boundary |
| --- | --- |
| Policy/dynamics learning rates; gradient clipping; gradient steps; batch size; action/update penalty weights; publication and diagnostics cadence; publication thresholds; replay sampling fractions; total training-iteration target | Apply as one snapshot at the next safe training-iteration boundary, or while paused. Update optimizer parameter groups without recreating optimizers. Update cached scalar settings and helper objects as well as the parsed config. |
| Architecture, widths/layers, history length, normalization policy, delay/horizon, initialization/pretraining settings, training/build GPU assignments, replay retention/capacity and validation partition shape | Stage for the next explicit Reset then Start. Report the pending change; do not rebuild/reset models silently. The GUI, supervisor, and real-time C++ app remain running. |
| Runtime stream width, controlled modes, shared-memory geometry, inference GPU, C++ command limits, C++ exploration settings | Owned by the existing runtime configuration. Do not claim a Python config edit can hot-reconfigure the C++ app. Reject conflicting overrides and report the source of truth. |

For a candidate containing any reset-only changes during a session, stage the entire revision; do not partially apply its hot fields under a misleading common revision. The operator can make a separate hot-only edit if needed. Fresh sessions load the latest fully valid configuration. Keep the existing numerical defaults unless the user edits them.

A sample config must enumerate supported keys and their hot/reset-only behavior. Persist the resolved configuration and revision with training diagnostics and engine metadata. Seed/rate changes must not silently discard model or optimizer state. No extra configuration editor is required in the compact GUI; its status line reports reload success, errors, or Reset required.

## 6. Shared use, failures, and ownership

- The GUI is an INDI client. Closing/disconnecting it does not stop training, disable PO4AO, reset models, or replay old requests when it reconnects.
- One backend owns compound operations and one training worker per configured session. Multiple GUI clients observe the same state. Serialize conflicting requests centrally; Reset/Disable can cancel or supersede relevant learning operations. Keep reconstruction transactions independent of the training/collection operation state: a long collection or training run must not lock out NN changes or require training to pause.
- Keep callbacks and the GUI event loop responsive. Commands queue work; callbacks do not synchronously load models, wait on FIFOs, or execute a training loop.
- Use exact child handles/process groups only for processes started by this app. Do not detect notebook processes, implement notebook/app session locks or leases, refuse Start because a notebook exists, or kill/adopt a notebook kernel. The operator manages which trainer publishes to the runtime. Retain the app's own operation serialization and reset-generation checks so its own pending export cannot undo Reset; this does not coordinate external notebooks.
- Use normal INDI readback for state, not button-click history. Listen for external changes. Do not turn a missing or stale state into a positive claim.
- Runtime failure keeps the GUI connected where possible and reports a bounded, actionable error. Worker crash does not reset the currently active C++ policy; the in-memory optimizer session is lost and requires explicit recovery/Reset, not fake Resume.
- Supervisor loss must stop the owned worker from further publication (control-pipe EOF/heartbeat), while leaving real-time inference and the selected policy alone. A supervisor restart must reconcile reset/eligible-model records before issuing anything; no automatic training restart or policy reload.
- All persistent paths must be configured, accessible to the runtime account, and independent of the operator's working directory. Do not silently change ownership or broad permissions.

### 6.1 Notebook development and diagnostics

The notebook is a supported development interface, not a discarded prototype. Implement one numerical training session with explicit initialization and one-iteration methods, used both directly by notebook cells and by the managed worker. Keep policy/dynamics models, optimizers, normalization, replay caches, counters, publication bookkeeping, and metric history accessible to notebook code. Separate initialization from iteration so rerunning an iteration cell does not reset this state. Core imports must not connect INDI/streams, read a mandatory deployed configuration, allocate a GPU model, or start training; pass configuration and data explicitly at initialization. Reuse the existing numerical helpers and preserve defaults.

Initial algorithm tests use notebook execution with synthetic/recorded inputs; instrument notebook runs remain operator-run. The standalone notebook need not start `po4aoTrainer` or its INDI supervisor. Keep development notebooks outside `apps/`; import the packaged numerical core. Development may use an editable install on the developer workstation; production remains a normal installed package independent of personal paths.

Preserve the nonlinear notebook's `plot_training_diagnostics` and `plot_model_debug_diagnostics`, including loss history, measured/predicted dynamics traces, policy/fallback diagnostics, modal spectra/time series, replay health and publication feedback where currently produced. Do not add linear coefficient maps to nonlinear models. Keep Matplotlib/IPython rendering optional and outside the headless training loop; no GUI plotting panel or plotting server is required.

When the app trains, a notebook can observe completed JSONL metric records and matching bounded diagnostic snapshots. Reuse `append_metrics`, existing plot functions, and model-snapshot helpers. Include the observation/action data needed by those plots, resolved config, session/iteration/replay identity, and candidate/published/active IDs so data from different iterations are not silently combined. Save compact diagnostic inputs at the configured cadence, with bounded retention, independently of successful policy export; do not record all replay buffers by default. Label snapshot age and iteration. Existing `.training.npz` artifacts contain analysis model state, not a complete resumable optimizer checkpoint.

No live attachment to worker Python variables, automatic notebook/app optimizer handover, or resumable disk checkpoint system is required. Notebook debugging accesses its own in-memory training session; app diagnostics inspect saved data. Add an acceptance example that renders the existing plots in both workflows without changing numerical training behavior.

## 7. Framework reuse and proposed implementation

### 7.1 GUI: one widget and one launcher

Add `gui/widgets/learningCtrl/learningCtrl.hpp` and `learningCtrl.ui`, plus `gui/apps/learningCtrlGUI/{learningCtrlGUI_main.cpp,learningCtrlGUI.pro,Makefile}`.

Reuse these references directly:

- `gui/apps/loopCtrlGUI/loopCtrlGUI_main.cpp`: QApplication, shared stylesheet resources, `multiIndiManager`, and a small standalone launcher.
- `gui/apps/dmCtrlGUI/dmCtrlGUI.pro`, `gui/apps/magaoxQtApp.pri`, `Make/magAOXGUI.mk`: qmake settings, generated UI paths, existing INDI libraries, and installation.
- `gui/widgets/dmCtrl/dmCtrl.hpp`: dynamic combo definitions, pending selections, deletion/redefinition, timeouts, mutex-protected cached state, and queued Qt-thread updates. Adapt the needed pieces, not the entire DM widget.
- `gui/widgets/xWidgets/xWidget.hpp`: shared font and automatic combo-popup sizing; `statusLabel` / `fsmDisplay` for established status presentation as appropriate.
- `gui/widgets/pupilGuide/pupilGuide.hpp`: state-dependent action labels and normal disabled-button styling.
- Upstream `gui/lib/multiIndiPublisher.hpp` and its subscriber dependencies: current discovery, property deletion, queued Qt delivery, and reconnect behavior. Bring in the coherent needed changes from the pinned upstream revision; do not implement another callback dispatcher inside this widget.
- Upstream `gui/lib/tests/offlineIndiPublisher.hpp`, its publisher tests, and `gui/widgets/pupilGuide/tests/`: socket-free tests, command capture, property lifecycle and stale-readback cases. Reuse the test harness and qmake patterns rather than building a second fake transport.

Reuse stock QPushButton/QComboBox/QCheckBox and existing status helpers. Keep widget updates on the Qt thread. Apart from importing existing upstream fixes and dependencies, no new shared widget infrastructure is anticipated. Do not use a generic immediate-send selection widget for the NN dropdown if it would load on selection instead of Enable.

Launch as `learningCtrlGUI po4aoTrainer`, using the existing INDI transport/aggregation. Device-name mappings come from the companion's configuration/status, not duplicated hard-coded GUI names. Use the current localhost:7624 launcher convention initially; no new SSH tunnel manager or browser server.

### 7.2 One Python INDI companion, not a service per button

Add a standard installed Python app at `apps/po4aoTrainer/`, following `apps/pythonIndiExample`, `apps/visxCtrl`, and `apps/aoSim`. It replaces the earlier proposed `learningCtrl` app; do not add both:

```text
apps/po4aoTrainer/
  Makefile                 # TARGET=po4aoTrainer; include ../../Make/pythonApp.mk
  pyproject.toml           # entry point po4aoTrainer
  xapp/po4aoTrainer/
    __init__.py
    app.py                 # XDevice control/status and bounded orchestration
    worker.py              # installed numerical child-process entry point
    training.py            # shared notebook/worker training session and iteration
    switching.py           # packaged switching-helper adapter
    resources/             # adapted existing switching script, if retained as shell
    ...                    # required model/replay/export/config helpers
  tests/
```

Use `magaox.indi.device.XDevice`, `BaseConfig`, `xconf`, PurePyINDI2 properties, standard logging/telemetry, and the existing client access pattern shown in `pupilCorAlign`. This app owns NN compound commands, helper-script execution, PO4AO/trainer coordination, config polling, and status. It remains alive when Reset terminates the numerical worker, so the GUI can start a new session.

`magaox/state_manager.py::XStateMachine` is an available precedent but directly couples a selection with FSM state; it does not supply completion acknowledgements or child-process supervision. Reuse it only if it fits without changing shared behavior. Otherwise use a small app-local operation/state record with explicit pending and acknowledged state. Do not create a new general orchestration framework.

Install the package and its declared dependencies into a shared, administrator-managed Python environment compatible with the existing instrument libraries and deployed CUDA/TensorRT stack. Prefer a dedicated environment such as `/opt/conda/envs/po4aoTrainer` if the trainer dependencies would disturb other apps; this is a proposed location, not an environment verified to exist. `Make/python.mk` already supports selecting `INSTRUMENT_CONDA_ENV`; no shared build-rule modification is needed. Verify the interpreter actually selected when installing, because `local/python.mk` may also set it.

Both supervisor and child use that installed environment. Keep heavy numerical/GPU imports in worker modules so the supervisor starts promptly without allocating a training model. Launch the child with `[sys.executable, "-m", "xapp.po4aoTrainer.worker", ...]`. Use `subprocess.Popen` with fixed argument arrays, bounded nonblocking stdout/stderr handling, and owned-process cleanup. Use a small private line-delimited command/status protocol over child pipes for pause/resume/config/stop/publication events; no additional network service or generic RPC layer. Logs must not block or corrupt the status channel.

The managed worker builds/stages an artifact; the companion owns its final publication/reload and current session identity. Adapt the existing publication helper rather than rewriting export checks. A Reset arriving during its export must make that candidate ineligible. Verify the engine hash/ID and matching metadata before publishing a final eligible-session record; the current helper replaces the engine and sidecars separately, so the existence of the engine file alone is not a complete publication acknowledgement. The already-existing C++ model-reset generation still protects in-flight runtime loads. These guarantees cover the app's own worker; no external notebook coordination is added.

### 7.3 Installed training code and writable data

The user requested a proper Python trainer app to avoid dependence on home-directory permissions. Migrate the required nonlinear training implementation and its transitive numerical helpers from the verified `PO4AO_2026` snapshot into `xapp.po4aoTrainer`, preserving numerical behavior and provenance. Do not merely copy the notebook or a thin script that still imports helpers from `/home/rlandman`. Keep one canonical implementation: development notebooks should import the packaged implementation, with optional plotting staying outside the headless worker. Port the relevant tests and use package-relative imports.

This follows the existing Python app pattern: `pyproject.toml` declares a console entry point and installable `xapp.*` modules; `Make/pythonApp.mk` performs a normal pip install into the selected interpreter, links the target into `/opt/MagAOX/bin`, checks imports from outside the source directory, and adds read/traverse permissions to the installed app package. Runtime imports come from installed site-packages, not the source checkout. Match the Makefile target, console script, package namespace, and device name consistently. The local `efcControl` example has a target/entry-point spelling mismatch (`efcControl` versus `efcCtrl`), so use the consistent `pythonIndiExample`/`aoSim` pattern rather than copying that mismatch.

`AGENTS.md` rule 26 still applies to standalone/ad hoc scripts. These modules are part of an installed, tested first-class Python application, as explicitly requested, and do not require a blanket change to that rule. Keep the C++ `apps/po4ao/` directory focused on the real-time app.

Use separate configured locations for installed code, editable config, and runtime output:

- Code and dependencies: administrator-managed Python environment, non-editable installation for production. Do not point imports, symlinks, Python interpreters, resource paths, or `PYTHONPATH` into a personal home directory.
- Control/training configuration: `/opt/MagAOX/config/po4aoTrainer.conf` and `po4ao_training.conf`; runtime needs read access, while authorized operators need the established config-edit permissions.
- Logs/telemetry: standard XDevice directories `/opt/MagAOX/logs/po4aoTrainer` and `/opt/MagAOX/telem/po4aoTrainer`.
- Engine artifacts, training runs/checkpoints, and session records: explicit shared paths on the appropriate instrument data/calibration filesystem, provisioned for the runtime account. The exact storage root is a deployment choice; do not invent a guaranteed existing directory. Set the trainer publication path and C++ `engineDirs`/`engineName` to the same location, with atomic-rename support and the required read/write/traverse access.

Normal xctrl operation uses `xsup`; provision only the required output directories with the established ownership/group/ACL policy. Installed code does not need to be writable by the trainer. Normal installation addresses code access, but does not automatically provision writable data/config/runtime directories or fix inaccessible parent directories and external dependencies. Never solve this with recursive world-write permissions or by running training as root.

Packaging acceptance must run imports and worker startup under the intended runtime account from an unrelated directory, with no personal `PYTHONPATH` and no access to the original source checkout. Confirm all transitive helpers/resources are installed, the launcher uses the intended interpreter, outputs are writable, and the C++ process can read the published artifacts. Runtime tests on the instrument remain operator-run.

### 7.4 CACAO switching helper

The untracked `apps/nnReconstructor/scripts/reconstructor_switch` in the review checkout is reference material only, not part of the `po4ao-pr` baseline. Reuse its bounded FIFO/INDI command approach where applicable, but do not copy it unchanged or import the speedups branch to obtain it. It uses the selected correctness/control branch's NN properties, but hard-requires an open loop, always changes modulation, hard-codes device/host/account names, rejects on any NN error before trying CACAO recovery, and explicitly leaves CACAO RUN state unverified. Use the existing NN controls in section 7.6 and serialize requests inside the managed app; notebook/app training locks remain excluded.

Ship the revised helper inside the installed `po4aoTrainer` package, so switching also has no home-directory dependency. To minimize changes, retain the adapted shell helper as declared package data and resolve it through standard package-resource facilities, invoking it with an explicit shell; alternatively expose the same implementation as a small package module if that is simpler. Provide manual access through the packaged adapter, e.g. `python -m xapp.po4aoTrainer.switching` using the app environment. Share one implementation between manual use and the backend, preserve bounded/typed command behavior, and test resource inclusion in the built package. Do not maintain a separate divergent home-directory copy.

Give the helper explicit target and modulation-policy arguments; supply trusted device/FIFO/preset settings from configuration. Remove loop-open checks and associated loop-opening advice. Do not add PO4AO/training checks. Avoid GUI text shell interpolation and direct GUI QProcess/SSH execution: the companion invokes it on the host owning the CACAO FIFO. GUI installation therefore does not require local CACAO access or an RTC shell account.

Use an installed, supported milk FPS/process observation interface to verify the specific `wfs2cmodeval-1` writer's RUN completion and identity. The local checkout has no verified adapter for the deployed milk version. `cacaoInterface::checkLoopProcesses()` still has a TODO for real process evidence and is not a substitute. Do not parse guessed binary offsets, accept a command flag as process exit, or infer writer identity from a shared output counter alone.

This observation adapter is an explicit implementation/acceptance dependency: obtain its version/help and sample readback from the operator, implement a small bounded adapter, then test it against fixtures. Until resolved, show Unknown/unverified and do not claim the full shared switching feature is accepted. Keep handover writer verification even though operator loop/PO4AO checks are excluded. Prefer existing installed milk tools/bindings; add a small local helper only if they cannot expose the required observation.

### 7.5 Targeted PO4AO changes

Work on `po4ao-pr`'s `po4ao.hpp`, `po4ao_impl.hpp`, and existing test/helper files. These C++ changes are required implementation scope, not a later optional enhancement.

- Separate collection state from controller-selection/history state. Current `beginWarmupCollection()` clears histories and disables learned inference; that contradicts R6. During recollection, keep the selected learned engine and history, and remove the current `!warmup` condition that would force fallback solely because collection restarted. Preserve startup/history-priming fallback when no usable policy/history exists.
- Separate exploration addition/finalization from the integrator-only path. Reuse the existing noise sequence, interpolation, controlled prefix, finite checks, and clipping; support adding it to either selected controller output without double clipping/noise or losing pass-through modes.
- Permit explicit recollection while control is Off, without enabling a retained policy. Keep startup warmup behavior distinct from recollection under an existing controller.
- Reuse current episode slots and background replay publisher; do not move replay-sized copies or disk work into the real-time callback.
- Add effective-controller/collection progress/generation/completion readback. Publish command acknowledgements/model state promptly, independently of the existing statistics interval; the inspected local config uses 10 seconds for statistics. A request counter consumed at callback entry is not completed output/reset/collection: acknowledge the relevant frame or replay publication only after that operation actually finishes.
- Preserve and expose useful reload errors rather than relying only on a transient status that is later cleared. Avoid issuing repeated reloads while one is pending.
- Reconcile externally requested reset/reload/disable changes with the companion and worker. CursesINDI remains a usable interface; GUI-origin commands are not the only possible writes.

Keep downstream routing fixed during ordinary GUI actions. Local `cacao-startup` links `aol1_mfiltINPUT` directly to `aol1_modevalWFS`; that does not establish that PO4AO output feeds mfilt on the instrument. Operator verification/setup of the intended routing is a deployment prerequisite. Report active PO4AO output separately from unverified downstream routing; do not claim Enable rewires CACAO.

### 7.6 Reuse the newer NN controls without speedups

Use `apps/nnReconstructor/` from `rlandman/nnreconstructor-correctness-controls` at `4353737e331671cd6a9b6aad5d92039c27f26715`. Do not implement new controls on the obsolete NN app originally in `po4ao-pr`. The same NN subtree exists at `rlandman/nnreconstructor-speedup-baseline` (`5c7e91da`); prefer the explicitly pinned correctness/control donor to avoid ambiguity from branch names.

Retain:

- Correctness/resource-lifetime fixes, protected inference/publication handoff and post-publication acknowledgement.
- Existing `model` catalog, `enable.toggle`, `model_status` (`requested`, `loaded`, `active`, `status`, `error`), startup discovery and configurable `parameters.startEnabled` behavior.
- Sequential disabled loading, fresh-result activation, and the single acknowledged zero publication on disable. Reuse existing failure/recovery behavior and app-local tests.
- The donor's FP32-only, single-input contract and synchronous `executeV2` execution. Preserve its numerical/preprocessing behavior; the older FP16/explicit-TT paths are already removed in this branch.

Exclude the separate pinned-input experiment (`4637d84b`, branch `rlandman/nnreconstructor-pinned-input`) and later performance work: CPU performance profiles (`b0030bfe`), pinned asynchronous execution/warm-up (`05be7c55`), CUDA graph replay (`2605506e`, `16190ca3`), and subsequent performance-profile documentation/tests/formatting that carry those changes. Do not merge `nnreconstructor-speedups` and merely turn options off; import the pre-speedup source itself.

No NN control/inference implementation changes are planned beyond importing this selected baseline. Run its existing app-local tests and test the GUI/supervisor against its actual property/readback semantics. Preserve the existing `frameGrabberPostPublish` path; the donor's shared framegrabber dependency already matches `po4ao-pr`. If integration reveals a concrete issue, document a narrow fix rather than reintroducing the excluded performance implementation.

## 8. Proposed narrow INDI contract

Reuse NN's existing `model`, `enable`, and `model_status` from the selected correctness/control baseline (section 7.6). Reuse PO4AO's existing `control_enable`, `reset_model`, `recollect_warmup`, `reload_engine`, and `model_status`, adding the missing runtime readbacks in section 7.5. Exact new names below are proposed; finalize them once against project naming conventions and keep code/tests/docs synchronized.

| Companion property | Meaning |
| --- | --- |
| `devices` (read-only text) | NN, PO4AO, modulation device identities for subscriptions. |
| `nn_activate` (text request) | Validated NN catalog element ID; one compound load/enable operation. |
| `reconstructor` (selection request) | Explicit `nn` or `cacao` target; confirmed source reported separately. |
| `link_modulation.toggle` | Shared runtime option, defaults Off on companion startup; changing it does not move modulation. |
| `po4ao_enable.toggle` | Coordinated Enable/Disable, including pause-on-disable. |
| `reset.request`, `recollect.request` | Coordinated session reset or recollection. |
| `learning` (selection request) | Run or pause; Run starts if no session exists and resumes if one is paused. |
| `status` (read-only text) | Operation/phase/error, actual reconstructor, effective controller, training state, active/eligible/pending policy identity. |
| `progress` (read-only numbers) | Completed training iterations and collection progress. |
| `training_config` (read-only text) | Path, desired/applied revision, pending-reset fields, validation error. |

Read-only status reflects observed completion, not writable request values. Busy/Ok/Alert accompany asynchronous operations. A connection heartbeat and explicit acknowledgements establish freshness; properties with unchanged values need not be re-sent solely to refresh a display timestamp. Do not use the lack of value changes as proof of disconnection.

## 9. Minimal installation and launch impact

The following are future operator-side deployment steps, not commands for the agent to execute on the instrument.

1. Build/install only `gui/apps/learningCtrlGUI` using its existing Makefile pattern. `Make/magAOXGUI.mk` runs qmake/make and installs the binary into `/usr/local/bin`. No change to the shared Makefile fragment is needed. Add one entry to top-level `all_guis` for normal aggregate builds; RTC roles already exclude GUI builds.
2. Install `apps/po4aoTrainer` using `Make/pythonApp.mk` into the selected shared environment. Match the Makefile target, package import, and console entry point to `po4aoTrainer`; the launcher is linked into `/opt/MagAOX/bin`. Add one entry to `pythonapps_rtc` for standard role-based installation, ensuring that this app selects its configured environment without changing other apps' environments. Do not add heavy training dependencies to the global instrument environment.
3. In the separate instrument-config repository, add `po4aoTrainer.conf`, `po4ao_training.conf`, one `po4aoTrainer po4aoTrainer` process entry to `proclist_RTC.txt`, and a local driver entry in `isRTC.conf`. Include the normal per-device runtime directories/FIFO driver setup used by the deployment environment; `pythonApp.mk` alone does not establish all of this.
4. Expose the new device through the existing INDI aggregation path used by the operator workstation/AOC. Add only required remote driver entries where the aggregator uses explicit allowlists. Do not create a new tunnel service or assume every RTC property is automatically forwarded. Verify underlying NN/PO4AO/modulation names through the same path.
5. Install the worker, numerical dependencies, and switching resources with the trainer package. Provision shared writable artifact/diagnostics/session directories and configure their exact paths and permissions for the runtime account. Verify operation without access to the developer's home directory, and training/build GPU separation from inference using the existing helper/configuration. Install the selected pre-speedup NN baseline and PO4AO recollection/status changes; deploy the imported collection app separately if requested by the operator.
6. Start the companion using the standard named `xctrl startup po4aoTrainer` workflow after operator registration. This starts the supervisor only; training starts from the GUI. Reset stops its numerical worker and owned build children, not the supervisor or C++ PO4AO app.
7. Launch `learningCtrlGUI po4aoTrainer` manually, like other GUI executables. Adding it to `gui/scripts/magaox_guis.sh` is optional and deferred by default, so routine observing startup does not gain another window unexpectedly.

No new service manager, HTTP server, Qt framework, C++ base class, core XDevice modification, or broad instrument rebuild is proposed. Existing INDI aggregation may need an operator-managed bridge reload/restart for its new device; do not hide that deployment requirement or restart unrelated processes.

## 10. Implementation sequence and acceptance criteria

### Phase A: baseline and contracts

Start from `rlandman/po4ao-pr` and prepare `<active-username>/nn-po4ao-gui` as specified in section 2. Import the pinned NN correctness/control subtree and the latest committed collection subtree from their separate donors, with their supporting tests/build/docs; import the needed current upstream GUI infrastructure with its dependencies. Do not merge `nnreconstructor-speedups` or transplant its NN implementation. Verify both collection and PO4AO test registrations, run the collection regression suite in a local configured environment, and record the reviewed baseline. Keep imported baseline changes separate from new GUI/training functionality.

Reconcile the operator-provided trainer snapshot with the inspected local copy before extraction. Record exact revisions and identify device names, routing, runtime account/environment, and supported milk writer observation. Preserve unrelated changes; do not commit on dev/main.

Deliver reviewed INDI/worker/config contracts and source-specific fixtures before relying on hardware behavior. Keep this document updated with concrete choices rather than silently widening scope.

### Phase B: preserve notebook workflow and extract the shared training core

Implement the shared session and notebook/observer contract in section 6.1 first. Extract numerical initialization and one completed training iteration into `xapp.po4aoTrainer`, including required numerical helpers; keep notebook execution using this same implementation. Preserve the existing plots and direct model inspection. Separate rendering from numerical diagnostics, and preserve export gates and numerical defaults. Add cooperative pause/resume/stop and configuration snapshots. Add a managed-worker publication staging boundary and reset-generation check; do not add notebook ownership enforcement.

Acceptance: with fixed synthetic inputs/seeds, extraction preserves model initialization, updates, validation/publication decisions, and emitted metadata to documented tolerances. Pause/resume retains parameters, Adam state, normalizations, reference identity, and counters. Config updates apply only at declared boundaries; invalid edits retain the old config; learning-rate changes do not recreate Adam. Long pretraining and export cannot indefinitely prevent pause/reset. Never run a full training job or touch instrument streams in these tests.

Notebook acceptance: render the existing training/debug plots from a direct training session and from the matching recorded diagnostics. Repeating an iteration does not recreate models/optimizers. Importing the core without deployed config/streams performs no runtime action. Initial algorithm tests can proceed in the notebook before the supervisor or GUI is built. Document that the operator manages notebook/app responsibilities; GUI Pause/Reset affects only its own worker.

Packaging acceptance: install into an isolated local test environment and import/run the headless fake-data path from outside both source checkouts, with no personal `PYTHONPATH` or accessible home-project dependency. Verify that switching resources and all transitive helpers are installed. Operator acceptance repeats the relevant permission checks using the deployed runtime account and shared data paths.

### Phase C: C++ controls, recollection and readbacks

Verify the imported NN controls using their existing app-local tests and the compound-control fixtures (section 7.6); do not reimplement them. Implement R6 using existing PO4AO real-time/replay machinery, with progress and prompt effective-controller/model readbacks. Neither task imports NN speedups.

Acceptance: seeded tests demonstrate exploration added once to learned-policy output and once to integrator output; schedule resets only at collection start; command limits and higher-mode pass-through remain correct; history/replay contains the final emitted commands; controller history is not reset; completion removes noise; invalidated/partial snapshots are not trained on. Test control Off with a retained model, source gaps, duplicate counters, publication backlog, model/reset races, and interrupted collection. Preserve existing ownership/reset-generation tests.

### Phase D: companion and switching helper

Implement the app using XDevice and existing PurePyINDI2 conventions; supervise one worker and one narrow switching helper. Centralize completion, timeout, cancellation, and publication eligibility. Adapt the switching helper/tests for explicit modulation policy, configured identities, observed writer completion, and the user's removed checks.

Acceptance with fake INDI/FIFO/process adapters:

- NN disable/load/enable uses the existing correctness/control implementation, waits for true completion including its single zero frame, never hot-swaps, and retains the selected pre-speedup inference computation. No old result follows disable acknowledgement.
- Tests run with loop Closed, unavailable loop readback, PO4AO active, and training running: NN/reconstructor commands are not blocked and issue no loop/PO4AO/training writes.
- Linked switching uses 2000 Hz/radius 3 and acknowledged modulation state; unlinked switching issues zero modulation writes and requires no modulation readback.
- No new writer starts until the old writer is observed stopped. Failed FIFO delivery, missing process observation, NN fault, missing model, and modulation timeout produce accurate failure states.
- Two GUI clients cannot make the one supervisor spawn duplicate workers or overlap conflicting handovers. Requests return target/operation acknowledgement rather than executing a toggle twice. No test or implementation requirement enforces exclusivity with external notebook training.
- Disable pauses but does not terminate the worker; Enable reactivates the latest eligible policy without resuming learning. Reset during build/reload prevents all old-session reactivation and leaves no owned build child running.
- Worker crash/control-channel loss does not silently reset the C++ policy; supervisor restart does not automatically start training or reload an old file.
- Recollect pauses/resumes only the previous running state and does not perform operator-state checks on reconstruction requests.

### Phase E: GUI and scoped install integration

Build the widget/launcher from the named examples and current upstream shared publisher. Reuse the existing offline test harness for dropdown staging and labels, long filenames, property delete/redefine, two clients, reconnect, delayed acknowledgements, changing action labels, config-error display, and disabled transitions. Check normal and narrow window layouts with the shared stylesheet. Link the new build/install entries without changing common build rules.

### Phase F: operator acceptance

The operator supplies installed-version and writer-observation evidence, verifies actual runtime/configuration/routing, and executes fake-stream/lab checks. Confirm the single acknowledged NN zero publication followed by no new NN publications during the loading gap, actual controller return/re-enable behavior, exploration under the selected policy, and unchanged effective policy after Pause. Exercise the operator's chosen loop state without the GUI imposing a loop-open check. Verify measured latency and compute contention under real loading/training conditions; local static or CPU tests do not establish those results.

No feature is called fully accepted based on a screenshot, mock GUI, successful command send, or source inspection alone. Record which checks passed locally and which still require operator evidence.

## 11. Coding discipline and planned changed files

Follow the full current `AGENTS.md`: minimal scoped edits, existing include guards/order/member naming, Doxygen file blocks without authors, brief declarations and inline parameter documentation, non-trivial definitions outside class declarations, and full changed-file documentation pass. Preserve the app header-only pattern with out-of-class definitions where established. Use base-class interface macros when available. Mark lock-only blocks `{ //mutex scope` and run clang-format on touched C++ files. Do not run clang-format on Markdown/Python.

For application Catch2 tests, register under `application_unit_test`, use the established `libXWCTest::<appName>Test` namespace and TEST_CASE documentation, and preserve explicit Doxygen-only API references when harness indirection breaks linking. Python companion/worker tests follow local Python conventions; do not force them into a C++ harness.

Functional commits include the relevant updated plan; follow with documentation-only and then formatting-only commits if needed. Use short commit messages and the required actual-model attribution. PR descriptions require the agent/prompt-or-plan attribution and separately copyable title/description. The original drafting request did not include commits or a PR. The subsequent baseline/implementation request authorizes local feature-branch commits; no push or PR has been requested.

| Planned files / location | Purpose |
| --- | --- |
| `gui/widgets/learningCtrl/`, `gui/apps/learningCtrlGUI/` | New compact Qt widget and launcher/build files. |
| `apps/po4aoTrainer/` | Installed Python INDI supervisor, numerical worker/training helpers, switching helper/resources, packaging, and tests. |
| `apps/nnReconstructor/` from pinned correctness/control donor | Import newer controls/correctness fixes, tests and relevant configuration/export documentation; exclude all later speedups. |
| `po4ao-pr`'s `apps/po4ao/po4ao.hpp`, `po4ao_impl.hpp`, helpers/tests/README as needed | Current-policy exploration/recollection and precise operational status. |
| `apps/dlDataCollection/` and its existing engineering plans | Import the latest committed collection baseline without NN donor code or unrelated uncommitted work. |
| Top-level `Makefile`, `tests/Makefile.one`, `tests/tests.list`, `tests/groups.dox` only as required | Narrow build/install and collection/new app test registration; preserve PO4AO entries. |
| Existing upstream GUI publisher/subscriber/test files and required dependencies | Import current reusable GUI fixes and offline harness; do not duplicate framework code. |
| Existing `PO4AO_2026` project | Verified migration source; adapt development notebooks to import the canonical installed trainer. This checkout is not a production runtime dependency. |
| Separate instrument-config repository | Two companion/training configs and necessary process/INDI registration. |
| This document | Requirements, concrete decisions, and execution/acceptance record. |

No modal-gain controls, newly invented GUI framework, NN speedups, unrelated calibration scripts, or automatic startup-window script changes are planned. The selected NN baseline import and reuse of existing upstream GUI fixes are explicit integration work; no new NN control/lifecycle implementation is required. If implementation discovers another necessary exception, document the concrete reason before expanding the change.

## 12. Final handoff review, 2026-11-02

Ready to share for implementation with the following scope and dependencies made explicit:

- The repository base is `po4ao-pr`, with the newer NN implementation imported from `nnreconstructor-correctness-controls` at `4353737e`. Controls, discovery and correctness fixes are retained; pinned-input, CPU profiles, asynchronous performance work and CUDA graphs are excluded. This corrects the intermediate interpretation that the user wanted the old NN app from `po4ao-pr`.
- The latest committed local collection files are imported by path with their narrow build/test dependencies, preserving PO4AO registration. Untracked tools and unrelated donor changes are excluded.
- Notebook training/plots remain supported through one numerical core, with an app-independent development path and file-based diagnostics while the app trains. The operator manages notebook/app responsibilities; no cross-trainer security/ownership layer is required.
- Managed-worker pause/reset/publication correctness remains required. It is not a promise to control external notebooks or prevent them from publishing independently.
- Warmup recollection retains the current policy/history, adds exploration once, records emitted commands, and continues that policy afterward. The C++ inference gate, control-Off collection path and completion readbacks are included in the work.
- Current upstream GUI helpers/tests are reused, with the existing MagAO-X style, build, Doxygen and branch conventions. Follow the applicable `AGENTS.md` when implementing; do not copy documentation/style defects merely because a reference file contains them.

No further product decision is required to begin. Phase A must still resolve technical integration details: verify the supplied trainer snapshot, obtain the operator's installed milk writer-observation interface, and pin runtime paths/device mappings. These are explicit implementation/deployment tasks, not evidence that runtime acceptance has already passed. A missing milk observation adapter blocks acceptance of automatic reconstructor handover, not notebook/core/GUI development with fixtures.

Review performed: local and GitHub source comparison, comparison of NN branch histories/subtrees and correctness/control publication semantics, collection revision/dependency review, requirements consistency pass and Markdown checks. No application changes, branch integration, builds, training runs, installation, instrument access or live acceptance tests were performed in this planning turn.
