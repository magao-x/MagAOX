# PO4AO trainer

`po4aoTrainer` is an optional Python `XDevice` companion to the native `po4ao`
application. A lightweight supervisor owns configuration, INDI requests and one
numerical worker. Training and engine builds run in the worker and its export
child; the real-time loop stays in C++. `po4aoGUI` also supports native inference
without the trainer, and its NN controls belong independently to `nnReconstructor`.

## Installation and configuration

The app uses the standard `Make/pythonApp.mk` installation rules. Its Makefile
selects the `po4aoTrainer` environment unless `INSTRUMENT_CONDA_ENV` is supplied.
Use an administrator-managed environment containing the repository's
`magaox-python`, the xwcl `xconf` and `purepyindi2` packages, and the dependencies
in `pyproject.toml`. GPU operation also requires compatible CUDA/TensorRT,
`trtexec` and ImageStreamIO Python bindings.

- `configs/po4aoTrainer.conf`: runtime device, shared artifact paths and operation
  deadlines. Provision an absolute writable artifact root accessible to the
  trainer and native PO4AO app. Match the native engine path to its `current`
  publication directory. `saved_engines` defaults beneath this root.
- `configs/po4ao_training.conf`: model, optimization, replay and diagnostic
  settings. Native geometry, gains and command limits come from the C++ config.
  Assign the training/build GPU separately from the inference GPU.

Configuration is validated as a whole. `configuration.HOT` lists settings that
apply at iteration boundaries; other changes wait for Reset. Hot learning-rate
changes retain Adam state. Invalid edits retain the last valid configuration.
The public delta-policy penalty is `[policy_update] penalty`; the legacy `weight`
spelling remains accepted, but specifying both is rejected.

The GUI's command regularization, delta-policy regularization and learning-rate
controls submit session overrides without writing configuration files. Overrides
survive Reset and Set policy, but restart restores file defaults. Pending settings
remain Busy until the worker acknowledges their request. Set `source_revision`
to the installed branch/commit so records identify the running software.

## Session and control behavior

- Start/Resume and Pause retain the same worker, models, Adam state, normalization
  and policy references. Pause settles issued publications before acknowledging.
- Enable selects the latest eligible learned policy. Disable pauses learning and
  selects fallback; Enable does not automatically resume learning.
- Reset invalidates the managed session and runtime policy, stops its worker and
  waits for completion. A timed-out Reset remains pending across restart.
- Recollect pauses learning, obtains fresh exploration data and resumes only a
  previously running session after collection completes. Partial data is rejected.
- Save policy copies one acknowledged engine and matching policy/dynamics snapshot
  into an immutable, checksummed library bundle.
- Set policy validates and restores both networks in a new worker. Normalization
  is retained, current hyperparameters apply, and Adam/replay/history/iteration
  start fresh. Learning stays paused. Enable if needed and recollect fresh warmup
  before resuming. An interrupted activation requires Set policy again or Reset.

A saved pair is initialization, not an optimizer checkpoint. `session.json` stores
eligibility and pending operations, not a resumable training session. Supervisor
restart never automatically starts learning; losing the worker requires Reset or
explicit Set policy. Old files alone do not establish eligibility.

Runtime restart/reset, stale readbacks and unknown external policies invalidate or
pause managed operations. Publication completion requires the matching runtime
acknowledgement. Temporary fallback retains the known numerical policy reference.
Closing the GUI leaves inference/training alone; losing supervisor supervision
stops its worker. Whole-app routing does not automatically resume learning.

The operator owns loop state and the choice of notebook or managed training. The
trainer does not control notebook processes or adopt their optimizer state. Reset
and Set policy replace the managed session, while saved library files remain.

## Python API and diagnostics

The numerical core can be used directly with synthetic data or operator-supplied
acquisition. Importing it does not open streams, connect to INDI or initialize a
GPU. Create one session and repeatedly call `step`:

```python
from xapp.po4aoTrainer.training import TrainingSession
from xapp.po4aoTrainer.po4ao_runtime import ReplaySnapshot

session = TrainingSession(resolved_config, device="cpu")
result = session.step(recent_snapshot, warmup_snapshot,
                      active_model_id=observed_loaded_model_id)
```

Snapshots contain float32 `[stream modes, samples, 2]` data in newest-first order,
with observation/command planes and coherent counter, valid-length, episode and
model identities. `session.invalidate_replay()` marks a recollection boundary.
The synchronous `po4ao_runtime.publish_tensorrt_engine` helper remains available
for explicitly operator-managed export; it does not confer managed eligibility.

`observer.save_diagnostics` and `load_diagnostics` write/read bounded portable
snapshots; loaded snapshots expose `plot()`. Direct plotting functions remain in
`po4ao_diagnostics`. Matplotlib is optional. Metrics JSONL can contain revised
publication outcomes: the last record for each session/iteration is authoritative.

## Local tests

With the actual framework dependencies installed in a workstation environment:

```sh
python -m pip install './apps/po4aoTrainer[test,plots]'
MPLBACKEND=Agg python -m pytest apps/po4aoTrainer/tests
```

Tests use synthetic CPU inputs, offline INDI boundaries and owned local child
processes. The independent `tests/fixtures/notebook_iteration.npz` reference uses
`rtol=1e-5`, `atol=1e-7`; pause/resume also checks exact CPU results. Its adjacent
JSON records the pinned original notebook/helper hashes, seeds and library
versions. Those hashes describe original provenance, not refactored module bytes.
CPU/fake-backend tests do not establish CUDA/TensorRT ABI, GPU release, installed
utility behavior or live instrument readiness; these require target validation.
