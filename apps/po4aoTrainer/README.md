# PO4AO trainer and controls

`po4aoTrainer` is the standard `XDevice` companion for the compact native
`po4aoGUI`. It owns one numerical worker and exposes the shared INDI
PO4AO request/status contract. NN selection, writer handover and modulation linking
belong to `nnReconstructor`, independently of this trainer. The C++ `po4ao` app retains real-time control. The same
`TrainingSession` is available to notebooks without starting the companion. The same
GUI also controls native C++ inference with the trainer stopped; Start/Pause is then
disabled. An existing notebook can keep its own training implementation/environment.
It need not migrate to the packaged core or use the optional synthetic example.

The [independent-review corrections](../../agents/plans/2026-11/nn-po4ao-review-response.md)
pass local regression checks. Target CUDA/TensorRT, FPS, INDI and timing acceptance
remains operator-run; control-capable instrument readiness is not established.
Follow [DEPLOYMENT.md](DEPLOYMENT.md)
for the dedicated environment, shared paths, registration and acceptance steps.
No personal source checkout is a production dependency. See the
[requirements](../../agents/plans/2026-10/nn-po4ao-gui-requirements.md) and
[execution record](../../agents/plans/2026-11/nn-po4ao-implementation.md) for provenance
and the checks actually performed.

## Controls and ownership

The coordinated behaviors below describe the optional managed trainer. In notebook
use, the GUI sends native C++ controls and the notebook owns all Python training state.

- Enable PO4AO reloads the latest eligible engine and selects learned control.
  Disable pauses learning and selects the integrator; Enable does not resume it.
- Start/Resume and Pause retain the same process, models, Adam state and fixed
  normalization. Pause stops an outstanding export child and waits for any issued
  reload to settle, preserving its exact numerical reference. Reset invalidates
  the old session, sends fallback/runtime invalidation before blocking cleanup,
  then waits for runtime completion. A timed-out Reset stays pending across restart.
- Recollect pauses managed learning, collects with exploration on the current
  controller, and resumes only a previously running session after a complete
  newer snapshot. Invalid or interrupted snapshots are never trained on.
- Closing the GUI leaves inference and training alone. Loss of the supervisor's
  control pipe/heartbeat stops its worker; restarting the supervisor does not
  automatically start learning or change the real-time controller. A lost
  in-memory optimizer session requires Reset.
- External runtime Reset/restart invalidates managed eligibility and terminates
  the owned worker; an external Disable transition pauses it. An unknown external
  policy pauses learning instead of being adopted as its numerical reference.
- Reconnected/stale properties cannot authorize operations. Fresh definitions and
  an advancing runtime heartbeat are required; the GUI expires a silent companion.

The operator manages loop state and notebook/app responsibility. There are no loop
interlocks, notebook detection, leases, locks against other trainers, or notebook
process control. Only Reset discards the managed in-memory session; files remain
available for analysis. Failed operations report their actual partial state,
without inventing rollback or successful completion.

## Configuration and artifacts

`configs/po4aoTrainer.conf` contains PO4AO device mapping, freshness
and operation deadlines, and the shared
artifact root. `configs/po4ao_training.conf` is typed TOML, watched by the supervisor.
Runtime geometry/history capacity, gains and limits come from the C++ config.
The parser rejects unknown keys and invalid combinations, retaining the last valid
revision after an incomplete edit.

`configuration.HOT` is the explicit hot-field list. Valid hot revisions apply as a
whole at iteration boundaries; learning-rate changes update existing Adam parameter
groups. A revision containing any reset-only field is staged in its entirety until
Reset. The INDI `training_config` vector reports desired/applied revisions, validation
errors and pending reset. Pending hot changes are applied when learning resumes.

Each run contains its resolved launch/config snapshots, worker log, completed
metrics JSONL and bounded diagnostic snapshots. Publication issuance/outcome append
revisions to the same session/iteration record; **the last record for that pair is
authoritative**. The matching diagnostic snapshot is rewritten with the outcome.
Exports run in private candidate directories. The supervisor verifies engine hash,
metadata/config revision, reset generation, matching NPZ identity and finite model
state, then replaces one `current` directory symlink. C++ reload completion
requires a fresh Busy/terminal acknowledgement and the intended model identity.
Completion is bound to the exact candidate and reload ticket under the supervisor
lock; an older status-thread result cannot promote a newer candidate. An unresolved
load cannot be replaced, and a send failure never borrows the preceding ticket.
The retained engine is eligible until Reset, including after Disable. Eligibility
is persisted in `session.json`, separately from unresolved publication and Reset.
Old files alone never restore eligibility. A lost load ticket across a supervisor
restart requires Reset; timeout/cancellation alone never promotes a candidate.

Set the app's `source_revision` to the installed branch@commit when deploying.
It is recorded in the app log, launch record, metrics and engine metadata. Diagnostic
snapshots also carry session/iteration/replay identity and resolved configuration.

## Notebook development and plots

[notebooks/po4ao/development.ipynb](../../notebooks/po4ao/development.ipynb) runs a small
synthetic CPU example, repeated iterations, the existing direct training/debug
plots, and observation of recorded diagnostics. It makes no instrument connection.
It also describes replacing acquisition inputs with the operator's existing cells.
The original personal notebook is not overwritten.

Importing the package or numerical core does not access deployed configuration,
streams, INDI or a GPU. Initialize once, then repeat only the iteration call:

```python
from xapp.po4aoTrainer.training import TrainingSession

session = TrainingSession(resolved_config, device="cpu")  # explicitly choose device
result = session.step(
    recent_snapshot, warmup_snapshot,
    active_model_id=observed_loaded_model_id,
    telemetry=runtime_telemetry,  # optional
)
```

Inputs are `ReplaySnapshot` objects with `[stream modes, samples, 2]` data in the
producer's newest-first order; the last dimension is observations and commands.
`None` means waiting for complete fresh data or reaching the iteration target.
Repeated calls preserve models, optimizers, fixed normalization and reference state.
`session.config` is a copy. Use `configuration.apply_hot` only at an iteration
boundary; reset-only changes require explicit reinitialization.

`session.invalidate_replay()` invalidates old collection data/history/anchors while
preserving weights, Adam and normalization. It waits for a complete newer warmup
counter; it does not perform recollection. The managed worker pairs copied streams
with C++ generation/ready readbacks and unwinds old-data work on generation change.
Completed gradient updates are not rolled back. Other iteration exceptions fail the
session and prevent publication.

Direct plots remain available through `plot_training_diagnostics` and
`plot_model_debug_diagnostics`, including optional direct model inspection. While
the app trains, observe a completed snapshot without accessing its worker:

```python
from xapp.po4aoTrainer.observer import load_diagnostics

snapshot = load_diagnostics(explicit_snapshot_path)
print(snapshot.metadata["session_id"], snapshot.metadata["iteration"], snapshot.age_seconds)
training_figure, debug_figure = snapshot.plot()
```

Snapshots contain matched diagnostic inputs, bounded history and the recent replay
required by those plots. They are not resumable optimizer checkpoints. Matplotlib
is optional and is not imported by the headless worker. The original synchronous
export helper remains available for explicit operator notebook use; the managed
worker calls it only inside a cancellable child/private candidate directory.
Notebook exports do not acquire managed-session eligibility automatically.

## Verification and source

Install test dependencies into a local environment containing the repository's
`magaox-python`, actual xwcl `xconf` and `purepyindi2` packages, then run:

```sh
python -m pip install './apps/po4aoTrainer[test,plots]'
MPLBACKEND=Agg python -m pytest apps/po4aoTrainer/tests
```

Tests use synthetic CPU data, real INDI property classes without sockets, and owned
local subprocesses. They cover numerical parity, persistent optimization, config
boundaries, worker/export cancellation, reset eligibility, request acknowledgements
and native/managed PO4AO operation ordering. Native NN handover tests live in
`apps/nnReconstructor/tests`. The notebook example's cells are executable locally.
GPU execution and live instrument behavior are separate acceptance checks.

`source_manifest.json` pins numerical helpers from `PO4AO_2026` revision
`b9076dd7aaf00ff7cd4f7f644e8f40dc1ff10a4f`. Previously authorized read-only inspection
matched all six helper hashes and confirmed unchanged numerical expressions in the
instrument notebook; details and notebook hash are in the execution record.
`tests/fixtures/notebook_iteration.npz` retains source hash, seeds and library versions
in its adjacent JSON. The CPU comparison uses `rtol=1e-5`, `atol=1e-7`; pause/resume
also compares exact CPU outputs, and direct/recorded plots compare rendered pixels.
