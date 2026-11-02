# PO4AO trainer: numerical core

This is the first implementation increment of the [approved shared GUI plan](../../agents/plans/2026-10/nn-po4ao-gui-requirements.md). It packages the nonlinear notebook's numerical implementation and provides a persistent `TrainingSession` for notebook development and the future managed worker.

**This increment is not a runnable INDI application or GUI.** The supervisor, managed worker, hot TOML configuration, bounded/cancellable TensorRT export, switching adapter, C++ recollection changes, GUI and deployment registration remain to be implemented. There is deliberately no console entry point, `pythonApp.mk` target or aggregate install entry until the supervisor exists. Do not launch this package with `xctrl` yet.

## Source and numerical behavior

`source_manifest.json` pins the local `PO4AO_2026` source at `b9076dd7aaf00ff7cd4f7f644e8f40dc1ff10a4f` with individual file hashes. The instrument notebook has not been inspected; the operator still needs to establish that it matches the intended migration source. Development here uses the available pinned local source, not a claim about the installed instrument copy.

The nonlinear models, fixed normalization, replay mixing, Adam updates, policy-update reference tracking, validation and publication gates retain that source's numerical behavior. Only the needed modal layers are extracted from the legacy model module, avoiding its import-time configuration dependency. Linear reference models are test fixtures, not a second production training path. Plotting dependencies are imported on demand. Missing runtime telemetry is labeled unavailable rather than fabricated.

`TrainingSession` adds explicit initialization and one-iteration calls, fresh/complete replay checks, cooperative checkpoints before each gradient update, configuration snapshots, and recollection invalidation. It neither opens streams nor performs INDI writes or automatic export. The existing export helper remains available for explicit notebook use; its synchronous build and separate file replacements are **not yet suitable for the managed worker**. Session-generation eligibility and cancellation must be implemented before connecting it to GUI controls.

## Notebook use

Install normally into a suitable local development environment:

```sh
python -m pip install './apps/po4aoTrainer[plots]'
```

Use a resolved configuration dictionary and `ReplaySnapshot` inputs supplied explicitly by notebook code. A snapshot has `[stream modes, samples, 2]` data in the existing newest-first producer order; the final dimension contains observations and commands. `stable_replay_snapshot` remains available to operator-run notebook adapters. Importing the package does not access deployed config or connect to the instrument.

Initialize once:

```python
from xapp.po4aoTrainer.training import TrainingSession

session = TrainingSession(resolved_config, device="cpu")  # choose the training device explicitly
```

Run the following repeatedly, without rerunning initialization:

```python
result = session.step(
    recent_snapshot,
    warmup_snapshot,
    active_model_id=observed_active_id,
    telemetry=runtime_telemetry,  # optional runtime_* fields
)
```

`None` means waiting for fresh complete data or the configured completed-iteration target has been reached; inspect `session.state`. `result["record"]` is populated at the diagnostic cadence. Model/optimizer objects, normalization, history and publication bookkeeping remain accessible on `session`. The constructor copies the configuration, and `session.config` returns a copy; hot updates are not implemented in this increment.

A future worker passes `checkpoint=...`. The callback may block until Resume; returning continues the same Python stack and optimizer state. Raising terminates the numerical session and invalidates publication eligibility. Completed gradient steps are not rolled back, and a failed session must be discarded. The process-level Pause/Reset protocol and bounded export cancellation remain separate work.

At recollection start, call `session.invalidate_replay()` before any replacement warmup snapshot is available. It clears historical replay and anchor caches, preserving weights, Adam, reference identity and established normalization. Training waits for a complete newer warmup counter. This call does not recollect data itself; the runtime changes and acknowledgements in phase C are still required.

An explicit notebook export can use the unchanged publication gate, then `session.stage_publication(exported_id)` to register the exported current candidate. This records a pending reference; a matching `active_model_id` on a later iteration acknowledges it. It does not issue a reload command or establish managed-session eligibility.

## Existing plots and recorded diagnostics

For direct inspection, call `plot_training_diagnostics` and `plot_model_debug_diagnostics` from `xapp.po4aoTrainer.po4ao_diagnostics`, using `session.metrics_history`, `session.obs_replay` and `session.act_replay`. Model attributes remain available for notebook experiments.

At diagnostic cadence, record matched inputs into an explicit per-session run directory:

```python
from xapp.po4aoTrainer.observer import save_diagnostics, load_diagnostics

path = save_diagnostics(session, run_directory, keep=3)
snapshot = load_diagnostics(path)
print(snapshot.metadata["session_id"], snapshot.metadata["iteration"], snapshot.age_seconds)
training_figure, debug_figure = snapshot.plot()
```

Snapshots contain session/iteration/replay identities, active/published policy IDs, resolved config/revision, fixed normalization, bounded metric history and the current fixed-capacity recent replay needed by the plots. They exclude warmup/historical buffers and optimizer/model checkpoints. Atomic replacement prevents readers seeing partial files; retention only removes this session's older diagnostic snapshots. These files are for observation, not resuming training. Existing `append_metrics` and `save_training_snapshot` remain explicit utilities; the supervisor will wire their cadence and paths.

The operator manages notebook/app responsibility. No notebook detection, leases, ownership locks or automatic notebook control are introduced.

## Local verification

```sh
cd apps/po4aoTrainer
MPLBACKEND=Agg python -m pytest
```

Tests use synthetic data on CPU. `tests/fixtures/notebook_iteration.npz` was generated from the pinned original notebook's initialization and first iteration, using the synthetic settings in `session_config.json`. Its adjacent JSON records source hash, library versions and seeds. It contains all model/Adam state plus selected validation/publication results. Tests compare with `rtol=1e-5`, `atol=1e-7`; pause/resume additionally compares exact CPU outputs. The plot test compares rendered pixels from direct state and the matching recorded snapshot.

No CUDA/TensorRT execution, live streams, instrument deployment or closed-loop performance is established by these tests. Production installation will use the existing Python app framework and an operator-provisioned shared environment; this package has no import dependency on a personal PO4AO source checkout.
