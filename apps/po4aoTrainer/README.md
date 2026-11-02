# Shared NN / PO4AO trainer and controls

`po4aoTrainer` is the standard `XDevice` companion for the compact native
`learningCtrlGUI`. It owns one numerical worker and exposes the shared INDI
request/status contract. The C++ `po4ao` app retains real-time control. The same
`TrainingSession` is available to notebooks without starting the companion.

Implementation and local verification are complete; target CUDA/TensorRT, FPS,
INDI and timing acceptance remains operator-run. Follow [DEPLOYMENT.md](DEPLOYMENT.md)
for the dedicated environment, shared paths, registration and acceptance steps.
No personal source checkout is a production dependency. See the
[requirements](../../agents/plans/2026-10/nn-po4ao-gui-requirements.md) and
[execution record](../../agents/plans/2026-11/nn-po4ao-implementation.md) for provenance
and the checks actually performed.

## Controls and ownership

- NN Enable stops the CACAO writer, disables the NN output, loads the selected
  catalog model if needed, then enables NN. Switch reconstructor uses an explicit
  destination and the observed source. Writer exit/start is verified through the
  installed milk FPS interface. Optional linked modulation uses unmodulated NN
  versus CACAO at 2000 Hz/radius 3. Reconstruction is independent of learning.
- Enable PO4AO reloads the latest eligible engine and selects learned control.
  Disable pauses learning and selects the integrator; Enable does not resume it.
- Start/Resume and Pause retain the same process, models, Adam state and fixed
  normalization. Pause stops an outstanding export child. Reset invalidates the
  old session before stopping owned processes, then requests the runtime reset.
- Recollect pauses managed learning, collects with exploration on the current
  controller, and resumes only a previously running session after a complete
  newer snapshot. Invalid or interrupted snapshots are never trained on.
- Closing the GUI leaves inference and training alone. Loss of the supervisor's
  control pipe/heartbeat stops its worker; restarting the supervisor does not
  automatically start learning or change the real-time controller. A lost
  in-memory optimizer session requires Reset.

The operator manages loop state and notebook/app responsibility. There are no loop
interlocks, notebook detection, leases, locks against other trainers, or notebook
process control. Only Reset discards the managed in-memory session; files remain
available for analysis. Failed operations report their actual partial state,
without inventing rollback or successful completion.

## Configuration and artifacts

`configs/po4aoTrainer.conf` contains app/device mappings, deadlines and the shared
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
metrics JSONL and bounded diagnostic snapshots. Exports run in private candidate
directories. The supervisor verifies engine hash, metadata/config revision and reset
generation, then replaces one `current` directory symlink. C++ reload completion
requires a fresh Busy/terminal acknowledgement and the intended model identity.
The retained engine is eligible until Reset, including after Disable. Eligibility
is persisted in `session.json`; old files alone never restore eligibility.

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
and writer/modulation ordering. The notebook example's cells are executable locally.
GPU execution and live instrument behavior are separate acceptance checks.

`source_manifest.json` pins numerical helpers from `PO4AO_2026` revision
`b9076dd7aaf00ff7cd4f7f644e8f40dc1ff10a4f`. Previously authorized read-only inspection
matched all six helper hashes and confirmed unchanged numerical expressions in the
instrument notebook; details and notebook hash are in the execution record.
`tests/fixtures/notebook_iteration.npz` retains source hash, seeds and library versions
in its adjacent JSON. The CPU comparison uses `rtol=1e-5`, `atol=1e-7`; pause/resume
also compares exact CPU outputs, and direct/recorded plots compare rendered pixels.
