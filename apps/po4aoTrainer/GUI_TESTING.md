# GUI test runbook: notebook first, managed trainer second

Prepared 2026-11-02 for testing on **2026-11-03**. Software reference:
`rlandman/nn-po4ao-gui` through `061885b1` (review fixes in `16390deb`), based on
`rlandman/nn-po4ao-baseline` at `88271a3e`. Record the exact installed commit if later
documentation or code commits are included.

This guide tests the **existing combined `learningCtrlGUI`**. Separate NN and PO4AO
windows are a proposed follow-up, not executables supplied by this revision.
Use [DEPLOYMENT.md](DEPLOYMENT.md) for the full configuration and recovery contract.
All instrument-side commands and notebook cells below are for the human operator.
No target compilation, installation, INDI/FPS operation or GPU execution was performed
while preparing this guide. The operator manages loop open/close throughout.

## What can be tested in each stage

| Component/action | Notebook-led test | Managed-app test |
| --- | --- | --- |
| `po4ao` C++ runtime | Required for live replay/inference. | Required. |
| `po4aoTrainer` companion | Running to serve the GUI, with **no managed worker**. | Running and owns its numerical worker. |
| NN selection, CACAO/NN switch, linked modulation | Available. | Available. |
| PO4AO Enable/Disable | Select the resident notebook policy/integrator, provided managed eligibility was cleared first. | Reload/reactivate the eligible managed engine or select the integrator. |
| Recollect warmup | Collects with exploration; the operator pauses notebook iteration and invalidates its replay. | Pauses/resumes its worker as needed and invalidates replay automatically. |
| Start/Pause/Resume learning | **Do not use. These start/control a separate worker, not the notebook.** | Use to control the managed session. |
| Reset PO4AO | Resets runtime and managed eligibility, **not notebook Python objects**. Stop notebook iteration first and explicitly recreate its session afterward. | Stops owned processes, clears eligibility and resets runtime. |
| Training/debug plots | Direct access to the notebook's session/models. | Notebook reads saved diagnostics; no live worker-variable attachment. |

The GUI can show Learning: Idle while a notebook is training. That label describes
the managed worker only. Disable cannot pause notebook code, and closing the GUI
does not stop either workflow.

## 1. Prepare software, accounts and paths

Use the normal deployment account for installation and the normal runtime account
for applications. The notebook kernel needs access to the replay shared memory,
training GPU and shared artifact directory; typically that means an operator-run
kernel on RTC, viewed from the workstation. A laptop-only kernel cannot read RTC
shared memory merely because INDI is tunneled.

Record these site-specific values before installing; examples are not verified
installed settings:

| Item | Example / required check |
| --- | --- |
| Checkout on each build host | `/opt/MagAOX/source/MagAOX`, maintained by `xdev`; or the approved deployment checkout. |
| Python environment | `/opt/conda/envs/po4aoTrainer`, Python 3.11+, based on the known working training stack. |
| Device names | NN may be `nnReconstructor` or `dlReconstructor`; confirm PO4AO, modulation and companion names. |
| Config files | `/opt/MagAOX/config/po4aoTrainer.conf`, `po4ao_training.conf`, `po4ao.conf`. |
| Shared storage | Example `/data/po4aoTrainer`; notebook and managed accounts need write/traverse access, C++ needs read/traverse. |
| NN/CACAO writer | Confirm FPS name, FIFO and matching process identity from deployed milk. |
| GPUs | Training/build share a GPU distinct from inference; the existing helper requires matching inference/training GPU model and compute capability. |

Retain the previous deployed source/config revisions and Python environment details
for rollback. Make this branch available on the build hosts through the normal
repository workflow; it has not been pushed by this agent. Do not merge the excluded
NN speedups or change unrelated applications for this test.

Use the normal MagAO-X build environment. On RTC, after setting `MAGAOX_SRC` to the
verified checkout and confirming its branch/commit:

```sh
git -C "$MAGAOX_SRC" status --short --branch
git -C "$MAGAOX_SRC" rev-parse HEAD
make -C "$MAGAOX_SRC/apps/po4ao"
make -C "$MAGAOX_SRC/apps/nnReconstructor"
```

Both Makefiles reference TensorRT under `/opt/MagAOX/vendor/TensorRT-10.0.0.6` and
CUDA under `/usr/local/cuda`. Stop here if the target SDK build fails; the macOS CPU
tests do not compile the full app. Once builds succeed and the operator has arranged
the runtime transition, install only these app directories:

```sh
make -C "$MAGAOX_SRC/apps/po4ao" install
make -C "$MAGAOX_SRC/apps/nnReconstructor" install
```

If the exact selected NN donor is already installed, its reinstall is unnecessary.
`dlDataCollection` has no GUI dependency and need not be redeployed for this trial.
Do not run a top-level installation.

Provision the dedicated Python environment with the working CUDA/torch/TensorRT,
ImageStreamIOWrap and CacaoProcessTools versions, `magaox-python`, and the real xwcl
`xconf`/`purepyindi2` packages. Do not substitute an unrelated PyPI `xconf`. Then:

```sh
make -C "$MAGAOX_SRC/apps/po4aoTrainer" INSTRUMENT_CONDA_ENV=po4aoTrainer -n install
make -C "$MAGAOX_SRC/apps/po4aoTrainer" INSTRUMENT_CONDA_ENV=po4aoTrainer install
```

Inspect the interpreter printed by the dry run. The app Makefile installs through
the existing Python rules and links `/opt/MagAOX/bin/po4aoTrainer`. The notebook
must use this same installed numerical package, not helpers imported from a personal
source directory. Install plotting/Jupyter-kernel dependencies into that environment
if absent, and select/register that interpreter in your existing Jupyter setup.

On the GUI host, build/install from the same source revision using normal Qt/mxlib
dependencies:

```sh
make -C "$MAGAOX_SRC/gui/apps/learningCtrlGUI"
make -C "$MAGAOX_SRC/gui/apps/learningCtrlGUI" install
```

The resulting launcher is `/usr/local/bin/learningCtrlGUI`. Do not copy the local
macOS test build or its mxlib-linkage override to the instrument.

## 2. Configure and start the control services

Follow [DEPLOYMENT.md](DEPLOYMENT.md) to provision shared storage, copy the two trainer
config examples into the instrument-config repository, and add the companion's
normal process/INDI-driver entries. Set `source_revision` to the installed software
branch and full commit. Confirm device names, FPS identity, modulation mappings,
GPU assignments and stream geometry. Preserve existing runtime gains/limits.

The companion and notebook will use the same runtime engine path:

```ini
[parameters]
engineDirs=/data/po4aoTrainer/current
engineName=policy.trt
```

Replace the example root consistently in both configurations. `current` must be a
directory symlink, not a real directory. Notebook candidates and managed candidates
live in separate immutable directories beneath the shared root. The companion does
not automatically reload a file merely because that symlink exists.

Confirm `warmup_episodes * iterations_per_ep == replay_buffer_size` and that the
configured four replay/metadata stream names exist when the runtime is operating.
Use the updated C++ runtime and companion together: the latter requires the new
`model_status.instance`, `sequence` and `control_target` fields.

Run as the intended runtime account, from an unrelated directory without a personal
`PYTHONPATH`:

```sh
/opt/conda/envs/po4aoTrainer/bin/python -m xapp.po4aoTrainer.check_deployment
/opt/MagAOX/bin/po4aoTrainer -h
```

The checker validates imports, config, the engine-path contract, executable builder
and a temporary atomic-link operation. It does not connect INDI or prove GPU export.
Confirm separately that the notebook account can access the same package, streams
and storage.

Start/restart the affected C++ devices using their actual configured process names
after the operator has arranged the transition. Start the registered companion:

```sh
xctrl startup po4aoTrainer
xctrl status po4aoTrainer
xctrl peek po4aoTrainer
```

On the GUI host, with the normal local INDI aggregation available:

```sh
learningCtrlGUI po4aoTrainer
```

This launcher connects to `127.0.0.1:7624`; it has no hostname/port option. Confirm
that route supplies the companion and configured NN definitions. The companion's
own INDI client must see NN, PO4AO and modulation on RTC.

**Pass:** heartbeat advances, GUI becomes Ready, actual reconstruction/controller
state is known, NN catalog is populated, and opening the GUI has not started learning
or switched either controller. Allow two companion heartbeat observations after
discovery. If stale/unknown, inspect definitions, mappings and logs before proceeding.

## 3. Notebook stage: preserve interactive algorithm development

### 3a. Run the supplied notebook unchanged as a package/plot smoke test

Open [development.ipynb](../../notebooks/po4ao/development.ipynb) in the selected kernel.
It deliberately uses small synthetic CPU inputs and does **not** connect to the
instrument or publish an engine. Run initialization, repeat the iteration cell and
render both direct and saved-snapshot plots. Confirm session identity persists and
iterations advance. This verifies the shared core and plotting installation only.

**Live notebook integration is not supplied as a finished Run All workflow.** The
new notebook is an example; before live training, make a test copy and connect the
existing operator acquisition cells to its shared-core calls as described next.
Do not run its synthetic-data cells as if they were instrument training. If that
adaptation is not ready tomorrow, complete the smoke test and GUI/runtime controls,
and leave live notebook training untested rather than substituting artificial data.

### 3b. Establish the notebook-owned session

1. Finish/stop any existing notebook iteration or export. In the GUI, Reset PO4AO
   and wait for completion. Confirm no retained managed policy and no managed
   worker. Do not delete `session.json` to achieve this. If Reset is pending, provide
   the expected frames/readback or retry it before continuing.
2. Leave the companion running for the GUI. Do not press Start learning in this stage.
3. In the live notebook copy, replace synthetic initialization with the shared
   resolved config and an explicitly selected training device:

```python
from xapp.po4aoTrainer.app import TrainerConfig
from xapp.po4aoTrainer.configuration import load_training
from xapp.po4aoTrainer.po4ao_runtime import configure_compute_resources
from xapp.po4aoTrainer.training import TrainingSession

app_config = TrainerConfig.from_config(
    config_path_or_paths="/opt/MagAOX/config/po4aoTrainer.conf"
)
config = load_training(app_config.training_config, app_config.runtime_config)
compute = config["compute"]
configure_compute_resources(
    compute["inference_device"], compute["training_device"], compute["build_device"]
)
session = TrainingSession(config, device=f"cuda:{compute['training_device']}")
```

Run initialization once. The existing acquisition cells must supply all of these:

- ImageStreamIO handles for the configured `buffer_channel`,
  `buffer_metadata_channel`, `warmup_buffer_channel`, `warmup_metadata_channel`.
- Fresh runtime `model_status` and `collection_status` observations. Invalidate
  prior-connection observations; a retained INDI cache is not fresh readback.
- `stable_replay_snapshot(image, metadata_image)` copies for recent and warmup data.
  Require collection `ready` with `generation == requested_generation` before and
  after copying, the same requested generation/runtime instance/reset identity,
  and complete valid lengths. Discard a copy if these change. Do not invent counters.
- The actual **loaded `model_status.active_id`** for numerical publication feedback.
  `effective_id` may be zero while the integrator is selected with a policy retained.

Recollect from the GUI while notebook iteration is stopped. Call
`session.invalidate_replay()` when recollection is requested; wait for the newer
complete snapshot before the next step. This retains models/Adam/normalization.
Then replace the synthetic iteration cell with one explicit live iteration:

```python
result = session.step(
    recent_snapshot, warmup_snapshot,
    active_model_id=observed_loaded_model_id,
)
print(session.session_id, session.iteration, session.state)
```

Here the three input variables come from the acquisition contract above, not the
example's random arrays or hard-coded zero model ID. Repeat only after genuinely new
complete replay is available. Keep the direct plot cells; inspect session models and
optimizers normally. There is no automatic training-file watcher in notebook mode:
load a valid revision and apply hot changes at iteration boundaries with `apply_hot`;
reset-only changes require explicit session reinitialization.

### 3c. Publish one notebook policy and compare controllers

Use the existing notebook export cells adapted to the packaged
`publish_tensorrt_engine` helper. Before any live publication, complete one export
and load against the target stack in the approved fake-stream/test setup.

For the live test, execute one iteration/export at a time:

1. Require the current diagnostic record's `publication_status == "ready"`.
   Export **`session.policy_copy`**, matching dynamics and resolved config, not an
   independently initialized model. The helper accepts an example input shaped
   `[1, 2*n_history-1, 1, n_modes_controlled]` on the training device.
2. Export to a new notebook-owned directory beneath `artifact_root`, for example
   `notebook-<session-id>/candidate-<unique-id>/policy.trt`. Pass the configured
   builder/build GPU and write build logs there. Metadata must include session ID,
   iteration, resolved config/revision, software revision and a notebook-owned
   generation string. Do not reuse a managed generation or overwrite a managed bundle.
3. Confirm engine, JSON sidecar and `.training.npz` are complete and matching. Reuse
   `verify_candidate(candidate_directory, artifact_root, notebook_generation)` if
   using the same metadata contract; it returns the verified directory and model ID.
4. Only after export completes, replace the runtime's `current` directory symlink
   atomically using a new temporary symlink in the same parent and `os.replace`.
   Do not edit the companion's eligibility/session record. It will not adopt this
   notebook engine as a managed publication.
5. Call `session.stage_publication(model_id)` **before issuing** the runtime reload,
   so the exact exported reference is known even if the reload wait is interrupted.
   Request `po4ao.reload_engine.toggle=On` through your existing operator INDI cells
   or cursesINDI. Wait for actual matching `model_status.active_id`, idle reload
   state and completion acknowledgement. Do not interpret a sent command as success.
   Do not start another iteration/export while that load is unresolved; retain its
   model identity through timeout. On failure, inspect and resolve it explicitly.
6. With no managed eligible engine, GUI Enable PO4AO selects the resident notebook
   policy. Disable selects the integrator and retains the policy; Enable selects it
   again. Neither button pauses/resumes the notebook. Keep notebook iteration stopped
   during these initial comparisons.

No reload is automatically requested merely because a notebook engine is on disk.
After Reset, its resident copy is invalidated and GUI Enable cannot recover it from
managed eligibility. End the notebook session or explicitly initialize a new one;
do not silently continue old reference bookkeeping after runtime Reset.

### 3d. Notebook-stage button checks

| Action | Expected observation |
| --- | --- |
| Choose another NN in the dropdown | No change until Enable. |
| Enable chosen NN | Disable/load/enable acknowledgement; actual filename matches. |
| Enable already active NN | No output interruption or modulation writes. |
| Replace one active NN with another | NN-only replacement; modulation unchanged even when linked. |
| Switch reconstructor, linking Off | Actual writer handover, no modulation change. |
| Switch reconstructor, linking On | NN unmodulated; CACAO at 2000 Hz/radius 3 with configured state mapping. |
| Disable/Enable PO4AO after notebook publication | Integrator/notebook policy comparison; loaded ID retained through Disable. |
| Recollect on notebook policy, then on integrator | Exploration applied to the selected controller, frames/target visible, newer complete warmup; notebook resumes only manually after replay invalidation. |
| Close/reopen GUI | Runtime selection and notebook variables persist; discovery refreshes state. |

Save session/config IDs, model hashes, relevant plots and observations. A passing
CPU notebook smoke test is not a passing live notebook stage.

## 4. Handoff: notebook to managed Python app

Stop notebook iteration and wait for any export/reload to resolve. Save the desired
plots/data. Close its acquisition/control handles or stop the kernel so it cannot
keep publishing during the next stage. This is operator coordination, not a lock.

Click Reset PO4AO and wait for acknowledged completion. Confirm no retained managed
engine, Learning: Idle, and the intended integrator selection. Recollect fresh
warmup. The managed app will start a **new** numerical session; it does not import
the notebook's weights, Adam state or reference bookkeeping. `.training.npz` files
are analysis snapshots, not complete resumable checkpoints.

## 5. Managed-app stage

| Step | Expected result / record |
| --- | --- |
| Start learning | One owned worker starts with a new session; iterations advance from complete replay. No manual notebook training. |
| First eligible publication | Complete bundle, matching loaded model ID, retained policy displayed and publication outcome saved. Build/loading alone is not activation. |
| Enable PO4AO | Latest eligible managed engine reloads/reactivates; effective learned control is reported. |
| Pause, then Resume learning | Same worker PID, session, models, Adam and normalization; pause settles any already-issued load. Current policy stays active. |
| Disable, then Enable PO4AO | Disable pauses learning and selects integrator; Enable restores eligible policy without resuming learning. Resume separately. |
| Recollect with learning running | Same controller plus exploration, full newer snapshot, then learning resumes with the same session. Repeat while learning paused and confirm it stays paused. |
| Change a hot setting | Valid revision applied together at an iteration boundary; inspect desired/applied revisions and diagnostics. |
| Make an invalid edit | Last valid config retained and visible error. Restore a valid file. |
| Change a reset-only setting | Whole revision staged, not partly applied; takes effect in a new session after Reset. |
| Reset during build/load | Owned process group stops; old artifacts stay on disk but become ineligible; old delayed load cannot restore eligibility. |
| Start again after Reset | New worker/session, with appropriate fresh replay and selected config. |

Observe the managed run with the notebook's final section: select one explicit
`run-.../diagnostics` directory, load a complete NPZ with `load_diagnostics`, and render
its plots. Record snapshot age and iteration. Metrics JSONL can contain multiple
publication revisions for one session/iteration; use the **last** record for that pair.
Do not compare arrays from different snapshots or claim access to live worker variables.

## 6. Failure/reconnect acceptance and decision to continue

Run controlled failure cases in the fake-stream/lab setup before relying on the GUI:

- Reset with input frames absent still sends runtime invalidation and stops its
  worker, stays pending through supervisor restart, and blocks Enable until completion.
- External runtime Reset invalidates managed eligibility; external Disable pauses
  managed learning, including during an Enable wait.
- Delay engine activation beyond the operation deadline: Pause/Disable must not
  report Ok while a previously issued load remains unresolved. After it settles,
  Resume must retain the exact numerical policy reference.
- A quiescent NN fault allows replacement or Use CACAO recovery. Unknown writer
  state must not be reported as stopped.
- Disconnect/reconnect clients without replaying commands. A silent companion becomes
  stale within five seconds; frozen C++ app-logic status also expires. Fresh discovery
  and advancing heartbeat restore availability.
- Measure inference latency and training/build contention on the real GPUs. The
  current local test evidence cannot establish timing or closed-loop performance.

Record pass/fail and evidence for each stage, plus exact software/config revisions,
runtime instance, session ID, selected NN filename, loaded/effective PO4AO IDs and
GPU assignments. Stop the trial at the first unexplained partial state; inspect
actual readbacks and logs before retrying. Do not clear persistent eligibility/reset
records by hand to make a button available.

For rollback, first select the intended real-time controller through the normal
operator procedure, then stop only the companion with `xctrl shutdown po4aoTrainer`.
Stopping it does **not** disable C++ PO4AO inference. Restore/reinstall the prior
affected app/config versions and restart only the named processes as appropriate.

## Proposed GUI split after this test

- **NN window:** engine dropdown/Enable, CACAO/NN switch, active model/writer and
  optional modulation link/status.
- **PO4AO window:** Enable/Disable, Reset, Recollect/progress, Start/Pause/Resume,
  controller/policy/training/config status.

The current implementation already serializes reconstruction and training separately.
Moving the existing controls into two native Qt widgets/launchers and adapting their
tests is a contained frontend refactor. Reuse xWidget, statusLabel, stylesheet,
publisher and the existing backend commands; keep orchestration out of Qt.

Two windows alone still depend on `po4aoTrainer`. If independence means NN must run
without the trainer service/environment/configuration, extract its existing switching
operations into a small independent backend as a separate change. The present
supervisor's startup reads training/runtime config even for NN-only use. Preserve one
owner for compound switching operations rather than running separate copies per GUI.

Recommendation: test the corrected current implementation first; then split the
frontends without changing numerical/runtime behavior. Decide separately whether
removing NN's dependency on the trainer service is also required.

References: [MagAO-X development/install policy](https://magao-x.org/docs/handbook/software/hacking.html),
[named xctrl process operations](https://magao-x.org/docs/handbook/operating/software/utils/xctrl.html),
[review response](../../agents/plans/2026-11/nn-po4ao-review-response.md).
