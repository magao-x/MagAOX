# Operator deployment of the shared NN / PO4AO controls

This branch supplies the numerical core, managed Python worker/supervisor, native GUI,
and C++ recollection/readbacks. Local tests do not establish CUDA/TensorRT compatibility,
installed FPS identity, INDI routing or real-time performance. Complete the target-side
checks below before using the controls on the instrument. Agents do not run these steps
on instrument hosts. `AGENTS.md` is unchanged.

## Install only the affected components

1. Use `rlandman/nn-po4ao-gui`, based on the recorded `rlandman/nn-po4ao-baseline`.
   Build the selected `nnReconstructor` and changed `po4ao` against the target SDKs.
   The NN subtree retains the newer controls without the speedups. Keep the latest
   `dlDataCollection` baseline; it has no new dependencies on this GUI.
2. Provision a shared Python 3.11+ environment, normally `/opt/conda/envs/po4aoTrainer`.
   Clone or reproduce the **known working training environment**, retaining compatible
   torch, CUDA, TensorRT, ImageStreamIOWrap and CacaoProcessTools builds. Install the
   repository's `magaox-python` plus the actual xwcl `xconf` and `purepyindi2` packages;
   an unrelated PyPI package named xconf is not a substitute. `purepyindi2` is not on PyPI.
   Do not upgrade the instrument's general Python environment to satisfy this app.
3. Build/install `apps/po4aoTrainer` through `Make/pythonApp.mk`, with
   `make INSTRUMENT_CONDA_ENV=po4aoTrainer install`. The app-local Makefile selects that
   dedicated environment even if `local/python.mk` specifies another default.
   Inspect the printed interpreter first. The console script links into `/opt/MagAOX/bin`.
   Both worker and exporter use that same interpreter. The installed package has no
   personal checkout dependency. Sample configuration is also included under the
   environment's `share/po4aoTrainer` directory.
4. Build/install `gui/apps/learningCtrlGUI` using its normal Makefile on the operator
   workstation/AOC. No additional startup window or service manager is installed.
   Launch as `learningCtrlGUI po4aoTrainer` after the normal INDI aggregation is ready.

## Configure the separate instrument-config repository

Copy `configs/po4aoTrainer.conf` and `configs/po4ao_training.conf` into the existing
configuration repository/deployed config directory. Confirm all device names against
actual definitions, particularly the NN name (some configurations use `dlReconstructor`).
No agent has edited or installed the instrument configuration.

Provision an explicitly chosen shared artifact root with the normal runtime account's
write/traverse permissions and the C++ runtime account's read/traverse permissions.
`/data/po4aoTrainer` in the example is a **provisioning choice**, not an observed path.
Do not use a personal home directory, run training as root, or grant world-write access.

Set these two C++ `po4ao.conf` parameters to the same root:

```ini
[parameters]
engineDirs=/data/po4aoTrainer/current
engineName=policy.trt
```

`current` is an atomically replaced directory symlink. Do not provision it as a real
directory. Each immutable candidate contains the engine, matching metadata and training
snapshot; the supervisor verifies their identity before replacing the link. C++ starts
in fallback without automatically loading an existing file. Preserve old artifacts for
rollback/analysis; the app does not garbage-collect training runs.

The current shared numerical core requires `warmup_episodes * iterations_per_ep ==
replay_buffer_size`. Other geometry/history/limits remain owned by `po4ao.conf`; the
training file cannot override them. Restart C++ normally if changing its configuration.

Add the usual `po4aoTrainer po4aoTrainer` entry to `proclist_RTC.txt`, and a local
`po4aoTrainer` driver entry to `isRTC.conf` following that repository's existing Python
app entries. Provision the normal `/opt/MagAOX/sys/po4aoTrainer`, FIFO driver, log and
telemetry directories through the existing deployment tooling. If the aggregation uses
explicit remote-driver lists, expose the companion plus NN/PO4AO/modulation definitions
through the existing aggregation route. Installing the Python wheel alone does not do this.
Set `source_revision` in the app config to the installed branch@commit; retain this
config under version control. The app records it in logs, worker launches, metrics
and candidate metadata. Before scientific/on-sky use, commit and push the software
and instrument-config revisions as required by the [development handbook](https://magao-x.org/docs/handbook/software/hacking.html). Use the normal xdev deployment account and install only the affected app directories.

Start the supervisor with the standard named `xctrl startup po4aoTrainer` workflow.
Training starts only on the operator's Start learning request.

## Target-side checks before live use

Run as the intended runtime account from an unrelated working directory, with personal
`PYTHONPATH` removed:

```sh
/opt/conda/envs/po4aoTrainer/bin/python -m xapp.po4aoTrainer.check_deployment
/opt/MagAOX/bin/po4aoTrainer -h
```

The check verifies imports, resolved config, writable shared storage and atomic symlink
replacement. It does not launch a worker, connect INDI, or validate GPU execution.

Then perform the approved fake-stream/lab acceptance:

- Compile the full C++ app against deployed CUDA/TensorRT and run its existing tests.
  Export one candidate using the deployed torch/ONNX/TensorRT stack, confirm matching
  input dimensions and model hash, and exercise reload through actual INDI readbacks.
- Verify installed milk matches the supported FPS interface: `RUNrunning`, `md().runpid`,
  `runpidstarttime.tv_sec/tv_nsec`. The implementation follows milk source revision
  `336160d054df5773a49d603ba987bea74692fcda` inspected previously. Confirm
  `writer_process_match` uniquely identifies this FPS writer's command line. Missing,
  unclean or mismatched writer evidence stays Unknown and blocks handover completion.
  FIFO delivery alone never proves RUN exit. The binding's connection updates milk
  bookkeeping; this is an operator-run application, not a strictly passive file reader.
- Confirm disabled NN acknowledges its single zero publication, then remains quiet
  through loading. Confirm CACAO exits before NN starts, and NN disables before CACAO
  starts. With Link modulation Off, confirm zero modulation writes. With it On, confirm
  unmodulated NN versus CACAO at 2000 Hz/radius 3. Loop state is entirely operator-managed.
- Exercise Start/Pause/Resume, Disable/Enable and Reset during a candidate build/load.
  Pause must preserve PID/models/Adam and current runtime policy. Disable pauses learning.
  Reset must stop owned children and prevent reactivation of pre-reset artifacts.
  No notebook kernel is discovered, adopted, paused or terminated.
- Recollect under learned control and under integrator with a retained policy. Verify
  exploration, emitted-command history, full snapshot identity, interruption on gaps,
  unchanged normalization, and resume only when learning was previously running.
- Disconnect/reconnect both GUI clients; check shared status and no replay of commands.
  Kill only the test supervisor/worker during a controlled trial: current C++ inference
  stays selected, orphaned managed publication stops, and lost optimizer sessions require Reset.
- Measure actual inference latency and training/export contention. CPU tests cannot
  establish real-time timing or physical loop stability.

## Configuration and recovery

Edits to training TOML are validated as a whole. Invalid edits retain the last valid
revision and report an error. Hot fields are learning rates, gradient steps/clips, batch
size, action/update penalties, anchor sampling, replay fractions, publication/diagnostic
cadence and thresholds, iteration target and diagnostic sample rate. They apply together
at an iteration boundary, preserving Adam and normalization. Architecture, history,
pretraining, GPU assignments, normalization and historical retention require Reset;
a revision containing any such change is staged in its entirety.

A failed operation reports Alert and does not invent a rollback or open/close the loop.
Inspect actual state before retrying. A failed worker or supervisor restart loses the
in-memory optimizer session; Reset clears this condition. Reset invalidates eligibility
before it stops the worker; if reset completion is pending without input frames, old
files remain ineligible. Enable can reactivate the last eligible managed engine but
never resumes learning automatically. The operator manages notebook versus managed
training. Use Reset when handing a managed session over to notebook experiments;
the app does not import a notebook's optimizer or eligibility record.

For manual operator switching, the packaged implementation is shared with the app:
`python -m xapp.po4aoTrainer.switching nn` or `cacao`, optionally `--linked` and
`--model <catalog-element-id>`. Coordinate manual requests with GUI use yourself.

Rollback: stop the named supervisor, restore the prior deployed app/config versions,
and select the intended real-time controller through normal operator procedures.
No rollback command is automatically sent to the instrument by this package.
