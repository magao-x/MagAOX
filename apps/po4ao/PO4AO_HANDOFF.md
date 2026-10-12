# PO4AO project handoff

Last checked: 2026-09-30

This is a historical experiment/deployment snapshot retained for provenance.
Its recorded branch tips, configuration values, installed-binary state, and
future-work items must not be treated as current. Use [README.md](README.md)
for the supported runtime interface and [PR_PREPARATION.md](PR_PREPARATION.md)
for the current PR preparation and operator test status. Current upstream
AGENTS.md requires agents to leave instrument commands to the operator.

This document is the practical handoff for the modal PO4AO implementation on MagAO-X. It describes the current software and instrument state, the development/deployment workflow, the normal fake-stream and lab workflows, known problems, and sensible next steps.

The physical instrument always takes priority over this document. Follow the [MagAO-X handbook](https://magao-x.org/docs/handbook/) for instrument startup, environmental limits, loop operation, and hardware safety. Open the relevant AO loop before changing CACAO routing or calibration.

## Scope and operating rules

- The current work is modal control in a 1563-mode KL basis. `Nmodes_controlled` selects a leading modal prefix for PO4AO; higher modes pass through unchanged.
- Make PO4AO changes locally, commit them on a namespaced feature branch, and only update exao2 when Rico asks to run a test.
- Do not create additional timestamped or commit-named checkouts. The two permanent exao2 working directories are `/home/rlandman/MagAOX` and `/home/rlandman/PO4AO_2026`.
- Do not modify non-PO4AO files unless Rico explicitly asks. In particular, do not interfere with `WFS2modeval` or change shared CACAO defaults as part of an unrelated PO4AO task.
- Never overwrite a dirty exao2 checkout, deployed config, model artifact, or live shmim merely to make it match local state. Inspect and reconcile first.
- `po4ao.control_enable` selects learned control versus the proportional fallback. It does **not** open or close the CACAO AO loop.
- A successful process start is not sufficient validation. Check the INDI properties, source/output counters, stream contents, CACAO routing, loop state, callback telemetry, and physical response separately.

## Working directories and repositories

### Local development machine

The common local parent directory is:

```text
/Users/ricolandman/Library/CloudStorage/GoogleDrive-ricolandman1@gmail.com/Other computers/My MacBook Pro/Research/MagAOX_software
```

| Component | Directory | Repository/branch | State on 2026-09-30 |
|---|---|---|---|
| C++ real-time app | `MagAOX/apps/po4ao` | `MagAOX`, `rlandman/po4ao-controlled-modes` | `4851cc01`; PO4AO files clean |
| Python training and analysis | `PO4AO_2026` beside `MagAOX` | standalone Git repo, `rlandman/po4ao-controlled-modes` | `97e5e15`; clean |
| Runtime config source | `config/po4ao.conf` beside `MagAOX` | `config`, `rlandman/po4ao-controlled-modes` | `6f269c4`; `po4ao.conf` clean |

The local `MagAOX` checkout also contains unrelated untracked files, and the local `config` checkout contains unrelated changes to `fpmAlign.conf` and `loPredCtrl.conf`. Leave them untouched.

### exao2/RTC

| Purpose | Path | State on 2026-09-30 |
|---|---|---|
| C++ checkout | `/home/rlandman/MagAOX` | branch `rlandman/po4ao-controlled-modes`, `dce61f081`; PO4AO source files are byte-identical to local, but the final three commit IDs differ |
| Python checkout | `/home/rlandman/PO4AO_2026` | branch `rlandman/po4ao-controlled-modes`, base `5cf4431`; dirty experiment state |
| Installed app | `/opt/MagAOX/bin/po4ao` | installed 2026-09-24 15:01; predates the 2026-09-25 `reset_model` commits |
| Deployed config | `/opt/MagAOX/config/po4ao.conf` | instrument/test configuration, not identical to local `config/po4ao.conf` |
| Models and noise | `/srv/aoc/home/rlandman/data/PO4AO` | live engine, linear sidecar, metadata, noise arrays, timing cache, and many stale ONNX external-data files |
| CACAO root | `/opt/MagAOX/cacao/tweeter-vispyr/tweeter-vispyr-rootdir` | run `cacao-calib-apply` here so the loop variables are loaded |
| Shared-memory images | `/milk/shm` | includes the PO4AO input/output/replay streams |

The dirty exao2 Python changes are scientifically relevant and must be preserved before any update:

- `MODEL_HISTORY = 10`
- `KNOWN_CONTROL_DELAY_FRAMES = 1`
- first publication at iteration 200
- 20 policy gradient steps
- warm-up replay fraction 0.2
- substantial edits to `po4ao_notebook_modal.ipynb` and `replay_buffer_health.ipynb`
- untracked Strehl-monitoring files and `magaox_interface.py`

The local Python repo is ahead with ridge-regression and experiment-metrology commits, but those files have not been cleanly deployed. Reconcile the exao2 experiment changes into the local repository before replacing anything.

The deployed config currently uses 1563 total modes, 10 controlled modes, `max_Nhist = 10`, uniform exploration noise, a `0.03 -> 0.002` warm-up noise schedule, `integrator_gain = 1.0`, and `linear_inference_backend = auto`. The local config instead uses 2 controlled modes, `max_Nhist = 8`, and constant `0.005` noise. Treat the deployed values as the record of the most recent lab experiment, not as changes to discard.

At the last inspection, `aol1_mfiltINPUT` pointed directly to `aol1_modevalWFS`, so the live loop bypassed PO4AO. All six PO4AO streams were present. Recent PO4AO logs showed the CPU backend at approximately 4 microseconds with no missed frames, deadline misses, inference failures, or replay drops. This is a time-stamped observation, not a guarantee of current state.

## Software layout and current behavior

### C++ app

The real-time app is in `apps/po4ao`:

- `po4ao.hpp`: class declaration and state
- `po4ao_impl.hpp`: header-only implementation, following the local MagAO-X app convention
- `po4ao_utils.hpp`: allocation-free history, feature ordering, limiting, and other testable helpers
- `po4ao.cpp`: main entry point
- `tests/po4ao_test.cpp`: unit tests
- `PO4AO_IMPLEMENTATION_NOTES.md`: development history; useful, but some early status/config statements are now stale

The callback is triggered by the configured input shmim. It publishes a full-sized differential modal command to `po4ao_output`. Downstream `mfilt` owns leakage and integration.

The important runtime behavior is:

- `control_enable = Off`: proportional fallback `integrator_gain * measurement`, without exploration noise; replay collection is paused.
- `control_enable = On`, during warm-up or without an accepted model: proportional fallback plus scheduled exploration noise during warm-up.
- `control_enable = On`, after warm-up and model activation: learned policy command.
- modes above `Nmodes_controlled`: copied from the input without learned modification.
- source-counter discontinuity: clear histories and the partial episode, publish a safe one-frame fallback/pass-through command, and do not train across the gap.
- startup: ignore stale persisted models and begin on the proportional fallback.
- `reset_model`: return to fallback, clear history/partial episode, and require a newly published/reloaded model. The source checkout has this feature; the currently installed binary probably does not.
- `recollect_warmup`: invalidate old replay snapshots, clear histories, restart the noise schedule, disable the current policy, and collect a new applied-command warm-up after control is enabled.

Replay publication and TensorRT loading happen off the real-time callback. Candidate TensorRT engines are deserialized, validated, and warmed by the loader thread; activation is a frame-boundary pointer swap. A CPU-linear sidecar is supported and is the preferred backend for the current `linear_local` policy. TensorRT remains available for nonlinear policies.

The callback reports p50/p95/p99/max latency, source-counter missed frames, deadline misses, inference failures, replay drops, model ID, and active backend through INDI and the app log.

### Replay protocol

The app owns and creates/recreates these transient streams:

- `po4ao_buffer`
- `po4ao_buffer_meta`
- `po4ao_warmup_buffer`
- `po4ao_warmup_buffer_meta`

The normal replay shape is `(Nmodes, replay_buffer_size, 2)`, with observation in the last-axis element 0 and applied command in element 1. Do not manually create these streams during normal startup.

The four-element metadata stream publishes:

1. valid length;
2. completed episode ID;
3. low 32 bits of model ID;
4. high 32 bits of model ID.

Python copies a snapshot only when replay and metadata are not being written, their counters match, and neither counter changes across the copy.

### Python training

The current notebook is `/home/rlandman/PO4AO_2026/po4ao_notebook_modal.ipynb`. Restart its kernel after updating helper modules or configuration; Jupyter otherwise retains old imports.

Important modules are:

- `po4ao_config.py`: Python-only model/training settings plus the small runtime subset loaded from `/opt/MagAOX/config/po4ao.conf`
- `po4ao_models.py`: `linear_local`, `shared_local_output`, and dynamics models
- `po4ao_runtime.py`: stable replay snapshots, compute placement, atomic model publication, and the CPU-linear/TensorRT artifacts
- `po4ao_diagnostics.py`: online training and replay diagnostics
- `po4ao_noise.py`: uniform or mode-scaled exploration-noise generation
- `po4ao_record_experiment.py`, `po4ao_metrology.py`, and `po4ao_experiment_analysis.ipynb`: event-driven post-experiment recording and analysis; currently local but not deployed
- `replay_buffer_health.ipynb`: replay repetition, correlations, delay diagnostics, and scale checks

The current reference policy is fully linear and mode-local. Each controlled mode reads only its own temporal history. The feature order is current observation, older observations from oldest to newest, then older applied commands from oldest to newest. The dynamics model is also linear and mode-local and is fitted with ridge regression. Policy optimization still uses Adam.

`KNOWN_CONTROL_DELAY_FRAMES` is an assumed model parameter. The diagnostic delay estimate is for display only and must not silently replace it. The local committed default is 0; the dirty exao2 experiment uses 1.

Python training/building uses GPU 1. C++ TensorRT is configured for GPU 0, but `auto` selects the CPU sidecar for a published linear model. Before changing GPU placement, confirm where `WFS2modeval` runs; it is the heavy real-time GPU workload and must not be disturbed.

The C++ `policy_blend` setting still exists, but both local and deployed configs set it to `1.0`, which applies the complete learned policy and performs no blending. Do not use it as an implicit training-stability mechanism.

## Version-control and deployment workflow

### Normal workflow

1. Inspect all three local repositories and the relevant exao2 checkout before editing. Preserve unrelated and uncommitted work.
2. Work locally on a `rlandman/<feature-name>` branch. Do not commit to `dev`, `main`, or `master`.
3. Keep functional, documentation, and formatting-only changes in separate commits where appropriate. Run the relevant tests and `clang-format` on touched C++ files.
4. Record the local branch and commit IDs. Deploy only after Rico asks for a test.
5. Update the existing exao2 checkout to the same commits. Do not make a new checkout or copy individual tracked source files over a different Git history.
6. Build and install only the PO4AO app, install the reviewed config, restart only PO4AO through `xctrl`, and verify the log and streams.
7. Record the exact C++, Python, and config commits plus the deployed config in the experiment notes/metrology output.

For `MagAOX`, publish the namespaced branch to the normal Git remote and fast-forward the existing exao2 checkout. The current local and exao2 branch tips have equivalent PO4AO files but different final commit objects, so the next agent must reconcile this once; do not use `git reset --hard` to hide the divergence.

`PO4AO_2026` has no network remote on the local machine. exao2 currently fetches from `/home/rlandman/PO4AO_2026.bundle`. Preserve this commit-based workflow: create a new bundle locally, copy it to a temporary name on exao2, verify it, replace the bundle, then fetch and fast-forward the existing checkout. Never update the checkout while its current experiment edits are dirty. First bring those edits back into the local repo and commit the intended changes.

The config is a separate Git repository locally, but no third permanent exao2 checkout is wanted. Copy the reviewed `po4ao.conf` through a temporary file and install it as:

```sh
sudo install --owner=xdev --group=magaox-dev --mode=0664 \
    /tmp/po4ao.conf /opt/MagAOX/config/po4ao.conf
```

Compare the installed file and record its config commit before restarting the app.

### Build and install only PO4AO

Use the `rlandman` account for Git/build/install work. Run normal instrument process control as `xsup`, following the MagAO-X operating conventions. To build and install on exao2:

```sh
cd /home/rlandman/MagAOX/apps/po4ao
make
sudo make install
```

Do not add `.` to `make install`. A serial app build is sufficient and avoids exposing repository-wide dependency-order races. This does not build or install every MagAO-X application.

Then restart and inspect:

```sh
xctrl restart po4ao
xctrl status po4ao
logdump -f po4ao
```

`xctrl restart po4ao` may also restart `isRTC` because of the configured dependency. Use `xctrl inspect po4ao` for its tmux console and detach with `Ctrl-b d`. Prefer `logdump` for persistent diagnostics.

For Python-only changes, run the local unit suite before deployment:

```sh
cd /home/rlandman/PO4AO_2026
python -m unittest discover -v
```

Run the C++ PO4AO unit test through the MagAO-X application-test build used by the checkout, then build/link the real app against the installed CUDA and TensorRT libraries. A successful unit test alone does not validate the real stream or GPU path.

## Working fake-stream flow

1. Commit the desired fake-stream configuration, including `Nmodes`, `Nmodes_controlled`, `max_Nhist`, source, destination, noise file, and safety limit.
2. Update the existing exao2 checkouts/config, build/install PO4AO, and restart it.
3. Confirm that the simulated source and destination streams have the expected dimensions and are advancing.
4. Start `po4ao_notebook_modal.ipynb` with a fresh kernel. Confirm the printed runtime-config path, mode count, history, assumed delay, architecture, and compute devices.
5. Enable PO4AO collection/control and watch the log, replay metadata, output scale, model IDs, and diagnostics.

For a fake-stream test, use the simulator streams directly in the config, for example `aoSim_wfs` as input and `aoSim_dm` as output when those are the active simulator names. Do not reroute the physical CACAO loop for a software-only test.

## Working lab flow

Keep this short and verify every transition. Instrument startup/alignment and closing the AO loop remain normal MagAO-X operations, not PO4AO responsibilities.

1. Start with the relevant CACAO loop open. Record the current routing before changing it:

   ```sh
   readlink -f /milk/shm/aol1_mfiltINPUT.im.shm
   ```

2. Check the required streams and their dimensions/counters:

   ```sh
   milk-shmimmon aol1_modevalWFS
   milk-shmimmon po4ao_output
   milk-shmimmon po4ao_buffer
   milk-shmimmon po4ao_buffer_meta
   ```

3. Start PO4AO on the proportional fallback and confirm model ID zero, finite output, correct sign/scale, and healthy callback telemetry. Start the notebook with a fresh kernel, but do not accept a learned policy before the fallback path is physically stable.

4. With the loop still open, switch the modal-filter input:

   ```sh
   howfs_mfilt_switch po4ao_output
   cd /opt/MagAOX/cacao/tweeter-vispyr/tweeter-vispyr-rootdir
   cacao-calib-apply default
   ```

   The switch script changes `/milk/shm/aol1_mfiltINPUT.im.shm`, stops several CACAO processes, restarts only `acquWFS-1`, and does not validate dimensions or close the loop. Verify the CACAO processes and calibration before proceeding.

5. Close the CACAO loop manually on the proportional fallback. Then request fresh warm-up data and enable PO4AO collection:

   ```sh
   setINDI -w po4ao.recollect_warmup.toggle=On
   setINDI -w po4ao.control_enable.toggle=On
   ```

   `control_enable` does not close the loop. Warm-up data are useful only when the published command is actually being applied downstream.

6. During training, monitor:

   ```sh
   getINDI -t 3 'po4ao.*.*'
   logdump -f po4ao
   ```

   Check source missed frames, deadline misses, inference failures, replay drops, active model/backend, action clipping, residual RMS/PSD, and the physical DM/PSF response. Match the model ID published by Python to the ID accepted by C++.

7. If the learned controller behaves badly, turn `control_enable` Off to return immediately to the proportional fallback. Open the CACAO loop if stability is in doubt. To bypass PO4AO completely, restore the recorded original input and reapply calibration, normally:

   ```sh
   howfs_mfilt_switch aol1_modevalWFS
   cd /opt/MagAOX/cacao/tweeter-vispyr/tweeter-vispyr-rootdir
   cacao-calib-apply default
   ```

Once the rebuilt binary includes it, `setINDI -w po4ao.reset_model.toggle=On` can discard the active model without restarting the app or recollecting warm-up. Verify the property exists before relying on it.

## Known issues and open questions

### CACAO trigger timing

The tweeter-vispyr setup currently configures:

```text
mfilt-1.inmval = aol1_mfiltINPUT
mfilt-1.procinfo.triggersname = aol1_modevalWFS
```

The source is `config/cacao/tweeter-vispyr/fpssetup.setval.tweeter-vispyr.conf`. `howfs_mfilt_switch` changes the `aol1_mfiltINPUT` symlink but does not change the trigger. Therefore `mfilt` can wake on the upstream WFS frame before PO4AO has published the corresponding output, and may read the previous PO4AO frame. This is a strong explanation for the lower stable gain through PO4AO, but it has not yet been proven with aligned counters/timestamps.

Measure this directly before changing the persistent CACAO default. A controlled comparison should record the `aol1_modevalWFS`, `po4ao_output`, and downstream modal-command counters/timestamps for both direct and PO4AO routing. If confirmed, test triggering `mfilt` from `po4ao_output` with the loop open and a clear rollback path. Do not edit the startup config merely on inference.

### Local, checkout, and installed-state drift

- The local and exao2 C++ PO4AO files match, but the branch tips do not share the final commit objects.
- The exao2 Python checkout contains uncommitted experiment settings absent from local.
- Local ridge/metrology work is not fully deployed.
- The deployed config represents the latest 10-mode experiment and differs intentionally from local.
- The installed binary predates `reset_model`; a source checkout at the new commit does not prove that the installed executable contains it.

Resolve and record these before the next code deployment.

### Model artifact cleanup

`/srv/aoc/home/rlandman/data/PO4AO` contains many orphaned hidden `.onnx.data` files, including zero-byte files, from previous engine publications. First fix/verify cleanup of ONNX external-data candidates in the Python atomic-publication path. Only then remove explicitly identified stale artifacts while no build is active. Never broadly clear the data directory or `/milk/shm`.

### Episode boundaries

Frame discontinuities reset histories and partial episodes, but a model can currently activate in the middle of a replay episode. That can mix two controllers in one otherwise valid episode. Consider either activating only at an episode boundary or recording an activation boundary and quarantining the mixed episode in Python.

## Recommended next steps

### 1. Reconcile and reproduce the latest stable 10-mode run

This is the immediate priority. Preserve the dirty exao2 Python notebooks/settings, reconcile them into the local feature branch, deploy the local ridge/metrology commits, rebuild the C++ app so `reset_model` is actually installed, and record all three commit IDs plus the deployed config. Repeat the 10-mode internal-source test before expanding the controlled prefix.

### 2. Prove or reject the `mfilt` trigger-race hypothesis

Measure end-to-end frame identity and latency, not only PO4AO callback duration. The existing callback timer covers work inside `processImage`; it does not include time waiting for the source semaphore or downstream `mfilt` scheduling. Compare:

- direct `aol1_modevalWFS -> mfilt`;
- `aol1_modevalWFS -> PO4AO proportional fallback -> mfilt`;
- if safe, the same path with `mfilt` triggered by `po4ao_output`.

Keep gains, calibration, input atmosphere, and loop state fixed. This must be solved before interpreting learned-policy stability.

### 3. Deploy and use the metrology recorder

The local recorder is event-driven: it stores each newly completed controlled-mode episode, one-second health records, and compact linear-model artifacts. Use `po4ao_experiment_analysis.ipynb` to determine whether the policy is merely changing instantaneous gain or using delayed observation/action terms for prediction. Useful outputs are:

- fitted instantaneous gain and unexplained command residual;
- contribution of current measurement, older measurements, and older commands;
- policy/dynamics coefficients versus lag and mode;
- held-out one- and multi-step dynamics error;
- residual PSD and variance before/after each model ID;
- closed-loop pole/spectral-radius estimates for the linear model;
- model changes, loop state, clipping, DM saturation, and frame discontinuities on one timeline.

### 4. Add episode quality/quarantine in Python

Do not discard every poor episode: poor but valid closed-loop behavior is needed to learn that aggressive actions are bad. Quarantine only data with broken semantics, such as:

- CACAO loop open or paused;
- source-counter discontinuity;
- non-finite values;
- physical DM saturation;
- command not actually applied;
- an episode spanning a model change.

Record the reason rather than silently dropping data. Discover the deployed CACAO `loop_state` INDI property instead of hard-coding a device name. Read actual DM saturation telemetry from the deployed tweeter configuration; a configured limit is not itself a saturation event.

### 5. Add a behavioral trust region for model publication

Keep policy training unconstrained, then compare a candidate with the active reference policy on held-out replay states. Backtrack between old and candidate weights until the normalized command change, predicted cost, clipping, and model-validity tests pass. This is a Python-side publication safeguard, not a C++ policy blend. It should allow faster learning than permanently lowering the optimizer step size while preventing a destructive first jump.

Do not simply reject every candidate whose modeled cost is worse under an uncertain early dynamics model. Combine the behavioral step limit with held-out model error/uncertainty and the observed command scale.

### 6. Test dynamics uncertainty without committing to a large ensemble

The code exposes `EnsembleDynamics`, but simply increasing its member count is not currently useful: ridge fitting produces one solution and copies identical weights into every member. For a diagnostic experiment, fit 3-5 members on episode-level bootstrap samples and check whether coefficient/rollout disagreement predicts bad policy candidates. For the linear model, analytic ridge leverage/uncertainty may be simpler and cheaper.

### 7. Improve warm-up information carefully

Keep a small, known persistent excitation during identification. A reproducible PRBS or multisine sequence can cover temporal frequencies more evenly than white noise. Mode-dependent amplitudes can be encoded directly in the columns of `control_noise.csv`; the C++ sigma schedule remains a global multiplier. Varying integrator gain during warm-up is deliberately deferred until the simpler excitation and trust-region changes have been evaluated.

### 8. Deferred larger changes

- C++ probationary activation with automatic rollback on degraded residuals or DM safety events. This is useful but more invasive and is intentionally deferred.
- a genuinely nonlinear mode-local dynamics model if linear held-out rollout errors justify it;
- the `shared_local_output` policy for shared temporal features plus a locally connected final layer;
- GUI-adjustable `Nmodes_controlled`, which requires training a model that covers more modes than are currently enabled;
- direct DM-space control after the modal pipeline is stable and understood.

## References

- [PO4AO (2022)](https://arxiv.org/pdf/2205.07554)
- [PO4AO follow-up (2024)](https://arxiv.org/pdf/2401.00242)
- [2026 implementation](https://arxiv.org/pdf/2606.10771)
- [Pseudo-closed-loop bootstrapping for reinforcement-learning-based AO control](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/14150/1415058/Pseudo-closed-loop-bootstrapping-for-reinforcement-learning-based-AO-control/10.1117/12.3103615.full)
- [MagAO-X handbook](https://magao-x.org/docs/handbook/)
