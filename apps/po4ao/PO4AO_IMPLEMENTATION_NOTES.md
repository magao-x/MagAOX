# PO4AO implementation notes

This file retains the engineering record from the original lab implementation.
For current runtime behavior and test commands, use [README.md](README.md).
The dated deployment observations below are historical, not a live instrument
audit. PR preparation and new regression results are in
[PR_PREPARATION.md](PR_PREPARATION.md).

## Version-control checkpoints

The original controlled-mode work is preserved on `rlandman/po4ao-controlled-modes`.
The PR branch is `rlandman/po4ao-pr`; its source baseline is `b6d5ab65` and its
upstream base is `02363bcf`.

| Repository | Validated simulation baseline | Hardening commit |
|---|---|---|
| `MagAOX` | `b323e0b4d01991b2ca374ec784a1b9e1c213d782` | `8943a70f`, `008d7a8a`, `364a229c`, `177555df` |
| `config` | `7f82cdeff2616a2bf4c3a5476297593a17497f55` | `31c6086`, `91cec87`, `adae100`, `46ebd0f`, `e348f5e`, `b18cb08` |
| `PO4AO_2026` | `f1352b6` | `f529caa`, `ee20ae8`, `f562d7e`, `289a7d6`, `3efed84`, `32e8a86`, `5c30b0b`, `d0cf805`, `ea036fa`, `c13f476` |

The Python directory was not previously version controlled. Its baseline commit captures the known training state before these changes. Unrelated local changes in `fpmAlign.conf`, `loPredCtrl.conf`, and `agents/plans/2026-09/` were left untouched.

The canonical exao2 Python checkout is `/home/rlandman/PO4AO_2026`. Earlier
fake-stream checkpoints were `32b9937` (linear single-model dynamics), `e3aa81a`
(known control delay), and `8ca29b3` (fake-stream notebook). These are historical
checkpoints, not the current Python branch tip.

To inspect a known-working state without moving its saved branch, use
`git switch --detach <baseline-hash>` in the corresponding repository. This
changes the checkout, so preserve uncommitted work first. Use `git revert` on
explicitly selected commits when preserving later history is required.

## What changed

### Runtime and Python configuration

`config/po4ao.conf` contains only values needed by the real-time interface: stream names and dimensions, replay publication, C++ warm-up control, inference-device selection, safety limits, and telemetry cadence. `config_version = 2` identifies this runtime-configuration schema; it is not an episode, model, or shmim version. C++ and Python require version 2 because Python must agree with C++ about stream dimensions, the controlled modal prefix, and engine paths. Python resolves this file from `PO4AO_CONFIG`, then `/opt/MagAOX/config/po4ao.conf`, then the sibling development repository.

Model and training choices remain near the top of `PO4AO_2026/po4ao_config.py`: `MODEL_HISTORY`, `POLICY_ARCHITECTURE`, `TRAINING_CUDA_DEVICE`, `ENGINE_BUILD_CUDA_DEVICE`, and `SAMPLE_RATE_HZ`. `max_Nhist = 32` is only the maximum history capacity preallocated by C++; `MODEL_HISTORY = 8` is the shorter history used by the current Python model and exported engine. C++ infers that active history from the engine's input tensor, so changing `MODEL_HISTORY` does not require restarting C++ when it remains within `max_Nhist`.

The C++ exploration-noise names state their timing explicitly. `exploration_sigma_start = 0.5` is the multiplier on the noise file at the first warm-up frame, and `exploration_sigma_end = 0.0` is its multiplier at the final warm-up frame; C++ linearly interpolates between them. These are not neural-network output standard deviations or Python training hyperparameters.

### Controlled modal prefix

`Nmodes` remains the full input, output, history, and replay-stream dimension. `Nmodes_controlled` selects the leading modal prefix actively controlled by PO4AO. With control enabled, modes below `Nmodes_controlled` receive the proportional or neural-network command, while every higher mode is copied exactly from the current input observation. Exploration noise and `max_command_abs` apply only to the controlled prefix. With control disabled, the controlled modes use the proportional fallback without exploration, higher modes pass through, and replay collection pauses. Shutdown still publishes zeros.

Python copies stable full-sized replay snapshots and then trains, validates, and exports only the first `Nmodes_controlled` modes. A TensorRT engine records its trained modal width in its tensor shape. C++ accepts an engine only when `Nmodes_controlled <= trained modes <= Nmodes`; this permits a later run-time control-count setting to use an engine trained on a larger prefix without rebuilding it. The current setting is static and requires an app restart when the configuration changes. A GUI-adjustable control count remains future work.

### Verified policy feature order

The exported modal policy consumes `2 * MODEL_HISTORY - 1` planes in this order:

1. current observation;
2. older observations, oldest to newest;
3. older actions, oldest to newest.

The former C++ order placed the current observation last and selected the wrong action ages. The shared-memory replay layout was not the cause. `assemblePolicyInput` now matches the notebook exactly, and a unit test fixes the contract with explicit values.

### Real-time path

- The callback uses preallocated history and episode slots; it performs no replay-sized copy, clear, allocation, engine deserialization, or device-wide synchronization.
- The app opens or creates the differential-output stream and refuses to replace an incompatible existing output. It owns the replay and metadata streams, creates missing streams from `Nmodes` and `replay_buffer_size`, and recreates those four transient streams when their shape or type is stale. Each recreation is logged.
- Full episodes pass through a fixed single-producer/single-consumer queue. A background publisher updates the rolling replay shmim and the frozen warm-up shmim. If it falls behind, the callback drops an episode and increments `replay_drops` rather than waiting.
- TensorRT uses a dedicated non-blocking CUDA stream and stream-local synchronization. A global `cudaDeviceSynchronize()` is no longer used.
- Startup deliberately ignores any persisted engine and begins in proportional fallback mode with model ID zero. This lets a changed `Nmodes_controlled` setting start cleanly even when an artifact from the previous run is still on disk. Only an explicit reload request may activate a model. The background loader deserializes, validates, and warms each TensorRT candidate on a separate low-priority CUDA stream before exposing it to the callback, which performs a pointer swap at a frame boundary; the old engine is destroyed by the loader thread. CPU-linear candidates follow the same explicit reload path. An incompatible or failed candidate leaves the app running on its current model, or on fallback when no model has yet been accepted. `po4ao.reset_model` deactivates the accepted policy at a frame boundary, clears its histories and partial replay episode, and reports model ID zero while proportional fallback continues. Reloads requested before the reset are generation-stamped and cannot reactivate afterward; only a reload requested in the new generation can select another model.
- Host and device input buffers retain `max_Nhist` and `Nmodes` capacity, but each active engine assembles and copies only its own inferred history and trained modal width. Compatible engines can therefore hot-reload without allocation, restart, or copying unused history planes or modes to the GPU.
- Input, output, and replay stream dimensions/types are validated. Controlled commands must be finite and are limited by the configured per-mode bound; passthrough modes must be finite and are copied without limiting. Failed inference falls back to the proportional controller. Turning `control_enable` Off selects the pure proportional controller without exploration noise and pauses warm-up/replay collection. Policy history is reset while Off so re-enabling the learned controller first collects fresh history. Application shutdown still publishes zeros.
- Command finiteness is checked from the IEEE-754 representation because the RTC release flags include `-ffast-math`, which can make `std::isfinite()` incorrectly compile as always true. The regression test covers both NaN and infinity under the deployed compiler flags.
- The output remains a differential modal command. Leakage and integration belong downstream in `mfilt`.

### Replay and model publication protocol

Each replay buffer has a paired four-element uint32 metadata shmim. For each completed publication, C++ writes:

- metadata element 0: valid time-axis length;
- metadata element 1: completed episode ID;
- metadata elements 2/3: low/high words of the active model ID;
- matching replay and metadata `cnt0` publication counters;
- both `write` flags while either data structure is changing.

Python continues training on the current data even when `cnt0` has not advanced, but it copies only when both `write` flags are clear, both counters match, and neither counter changes across the copy. It rejects invalid lengths and waits for both rolling and warm-up buffers to contain the configured capacity. Startup publishes a zero valid length so stale data from an earlier run is not trained accidentally. Replay `cnt1` remains available for its normal ImageStreamIO cube-slice meaning.

`po4ao.control_enable` selects between the learned controller (On) and the pure proportional fallback (Off); it intentionally does not operate the independently managed CACAO loop. After manually closing `mfilt`, request a fresh applied-command warm-up with `setINDI -w po4ao.recollect_warmup.toggle=On`. This immediately invalidates the old replay snapshots, restarts the integrator/noise schedule at the next frame, discards queued episodes from the previous collection, and keeps the neural policy disabled until a newly built engine is reloaded.

Replay stores the commands emitted by PO4AO upstream of `mfilt`, not measured
post-integration or post-clipping actuator commands. A model ID identifies the
selected model at snapshot publication; an episode can contain a model change.
Physical applied-command fidelity and episode quality must therefore be assessed
with downstream telemetry.

To drop only the active policy while leaving the loop and ongoing replay collection enabled, use `setINDI -w po4ao.reset_model.toggle=On`. The real-time callback returns to the configured proportional fallback on its next input frame. This switch does not reinitialize Python model weights; it only prevents the C++ app from using the old published artifact until Python publishes and explicitly reloads a new one.

The notebook now builds ONNX/TensorRT candidates beside the live engine, checks the conversion result, and atomically renames a successful engine into place. Only then does it request an INDI reload. A model-metadata JSON file records the config version, episode, full stream width, trained modal width, tensor dimensions, publish time, and the same FNV-1a model ID reported by C++.

The policy loop now uses `policy_grad_steps`; warm-up code in `po4ao_magaox.py` uses `policy_grad_steps_warmup`. The noise cell creates a valid `(time, Nmodes)` float32 array and no longer references `total_nmodes` or calls `.numpy()` on a NumPy array.

### Runtime telemetry

`po4ao.runtime_stats` reports callback p50/p95/p99/max latency, source-counter missed frames, deadline misses, inference failures, and replay drops. `po4ao.model_status` reports the active model ID and reload state. The same summary is written periodically to the app log. These counters are intended to expose the training/inference GPU-contention failure mode rather than hide it.

### Historical mode-local architecture and training validation

The examples in this section describe the earlier modal notebook. Current
nonlinear architectures, normalization, historical replay, and publication
settings belong to the separate Python training repository.

The available hybrid policy uses shared temporal feature extraction followed by a locally connected output: `(2 * MODEL_HISTORY - 1) -> 16 -> 8` is shared across modes and the final `8 -> 1` weights and bias are independent for each controlled mode. All operations have modal field of view one, so a mode never reads a neighboring modal index. With `MODEL_HISTORY = 8` and `Nmodes_controlled = 10`, it has 482 trainable parameters: 392 shared parameters plus 9 per controlled mode. The dynamics model is likewise mode-local. The hybrid preserves the sample efficiency of shared temporal features while allowing each controlled KL mode to learn its own final response.

A strictly linear locally connected reference is also available and is the current Python selection. It maps the `2 * MODEL_HISTORY - 1` history values directly to one command per controlled mode with independent weights and no bias. Its parameter count is `Nmodes_controlled * (2 * MODEL_HISTORY - 1)`, or 150 parameters for the initial 10-mode, `MODEL_HISTORY = 8` test. Select either `shared_local_output` or `linear_local` with `POLICY_ARCHITECTURE` in `po4ao_config.py`. Changing architecture, `MODEL_HISTORY`, or the trained modal width requires rebuilding a matching TensorRT engine, but it does not require restarting C++ while the engine stays within `max_Nhist` and covers the configured controlled prefix. Direct-DM-space control remains future work.

GPU ownership is split by responsibility: `cuda_device = 0` in `po4ao.conf` reserves GPU 0 for C++ TensorRT inference, while the Python constants select GPU 1 for PyTorch training and `trtexec` engine building. On exao2 both devices are RTX 4090 cards with compute capability 8.9 under the same CUDA/TensorRT runtime. The existing plan deserialized on either device, and a synthetic `63 x 1 x 1563` policy was exported, built on GPU 1, and deserialized successfully with TensorRT 10.0.0.

The modal notebook now caches stable replay snapshots and continues taking gradient steps from the cached data until the publication counter advances. This preserves continuous training without repeatedly copying the full replay and warm-up shmims. The newest complete episode is held out from gradient sampling. Every ten training iterations the notebook records:

- dynamics training loss and one- through multi-step held-out rollout loss;
- held-out modeled policy cost;
- residual and action RMS, maximum action, saturation fraction, and non-finite count;
- approximate action-to-observation delay;
- per-mode residual variance, representative residual time series, and temporal PSD;
- C++ callback latency percentiles and cumulative missed/deadline/failure/drop counters;
- replay episode/counter and active/published model IDs.

Records are appended to `savedir/<timestamp>/training_metrics.jsonl`, and the same data drives a compact dashboard. A candidate is published only when its held-out modeled policy cost improves, no replay values are non-finite, and no more than 1% of its validation commands are within 1% of the configured per-mode limit. Engine publication remains atomic and the build-device ID is stored in the engine metadata.

### 1563-mode, 2 kHz delayed fake-stream test

The PO4AO runtime configuration uses 1563 modes and a 500 microsecond latency deadline. Python uses `MODEL_HISTORY = 8`, a three-frame planning horizon, and a known one-frame control delay. The known delay masks the newest action from the one-step dynamics prediction, so the policy rollout cannot receive an immediate response from a command which has not yet reached the plant.

The delay definitions differ by one index intentionally. `aoSim lag=1` means that one intervening WFS frame is unaffected by a new command. Because replay index `t` pairs observation `y[t]` with the command issued after that observation, the command correlates with observation `y[t+2]`. The diagnostics therefore search correlation offsets zero through three and should report approximately two for this test.

Exploration noise already supports mode-dependent amplitude without a C++ change: each column of `control_noise.csv` belongs to one mode, and the global `exploration_sigma_start` to `exploration_sigma_end` schedule multiplies those pre-scaled columns. A raw mode-index-to-the-minus-two amplitude law is generally too steep for KL modes. If KL index grows approximately with spatial frequency squared, amplitude proportional to spatial frequency to the minus two corresponds approximately to mode index to the minus one. KL eigenvalues or measured modal response are preferable when available.

`PO4AO_2026/po4ao_noise.py` generates this approximation reproducibly. It defines the relative spatial-frequency proxy as `sqrt(1 + mode_index / knee_mode)` and raises it to the configurable spatial-frequency power. The initial power is -2 with knee mode 20. The resulting modal RMS scale decreases from 1 at mode 0 to 0.5 at mode 20 and approximately 0.0126 at mode 1562. The generated runtime table is `control_noise_powerlaw_m2.csv`; the C++ sigma schedule remains the overall amplitude control.

## Recommended first lab procedure

Follow the normal MagAO-X deployment process and perform the test as `xsup`. The handbook notes that application configs live in `/opt/MagAOX/config`, apps normally run as `xsup`, and `xctrl`, `logdump`, and the app tmux session are the standard control/diagnostic interfaces: [application overview](https://magao-x.org/docs/handbook/operating/software/apps/overview.html), [startup](https://magao-x.org/docs/handbook/operating/startup.html), and [troubleshooting](https://magao-x.org/docs/handbook/troubleshooting.html).

1. Keep the physical loop open and PO4AO control disabled. Confirm the correct CACAO calibration and loop number before changing any `mfilt` input; the handbook explicitly requires an open loop when applying/changing calibration: [CACAO setup](https://magao-x.org/docs/handbook/operating/cacao.html).
2. Check available `/milk/shm` capacity, then start `po4ao`; it creates missing replay and metadata streams using the runtime-config dimensions and automatically replaces stale PO4AO-owned replay streams. The command destination must already exist with the expected shape and type and is never replaced by PO4AO. The former manual `ImageStreamIOWrap.Image.create(...)` and stale-stream removal commands are no longer part of normal startup. Do not clear shared memory as a routine step; the handbook reserves that for a controlled shutdown/troubleshooting sequence.
3. Configure the selected loop's `mfilt-LOOPNUMBER.inmval` to read `po4ao_output`, while retaining the WFS stream as the process trigger. Confirm the exact loop number and mapping before starting; do not point PO4AO at `dm01disp09`.
4. Build/install the feature branch, install the matching runtime config as `/opt/MagAOX/config/po4ao.conf`, then run `xctrl restart po4ao`, `xctrl status`, and `logdump -f po4ao`. Startup must confirm all stream shapes and report fallback mode with model ID zero. The engine tensor contract is checked later when the notebook explicitly requests a model reload.
5. With `po4ao.control_enable` Off, verify `po4ao_output` contains the expected proportional baseline, paired replay/metadata `cnt0` values advance together after the learned controller is enabled, `replay_drops` and `inference_failures` remain zero, and p99 latency is comfortably below the configured deadline.
6. Start with downstream `mfilt` integration effectively disabled (loop open/zero multiplier). Verify the proportional command sign and scale on the dedicated stream before turning `po4ao.control_enable` On. Then raise the downstream gain/multiplier conservatively while watching the DM path and telemetry.
7. Run the modal notebook with `PO4AO_CONFIG=/opt/MagAOX/config/po4ao.conf`. Confirm its printed runtime-config path and dimensions before training. A successful conversion publishes a new model ID; confirm the same ID appears in `po4ao.model_status` after reload.
8. Abort on non-finite or unexpectedly saturated commands, rising inference failures, replay drops, missed frames, or p99/max latency near the frame period. Turn `po4ao.control_enable` Off to restore the proportional baseline, then open/zero `mfilt`, stop training, and inspect `logdump` and GPU utilization before retrying.

## Verification and remaining gates

- Python stable-snapshot tests pass; the configuration loader was checked against `config/po4ao.conf`; Python files compile; notebook JSON validates.
- Both 1563-mode policy variants were exported and built with the installed exao2 PyTorch/TensorRT stack on GPU 1. The hybrid produced a `(1, 1563)` output with 15,227 parameters; the `MODEL_HISTORY = 8` linear reference produced the same output shape with 23,445 parameters. All 12 Python unit tests passed on exao2.
- The history-capacity update compiled and linked on exao2 against the installed CUDA and TensorRT libraries. The PO4AO unit test passed 21 assertions in four cases, including selection of the newest short history from a larger retained buffer. An isolated 25-mode `MODEL_HISTORY = 8` linear engine built and deserialized on GPU 1 with a 15-plane input while `max_Nhist = 32` remained the C++ capacity.
- The renamed configuration was rechecked in isolation on exao2: all 12 Python tests passed, the 25-mode linear policy contained 375 parameters, TensorRT built it on GPU 1, and TensorRT 10.0.0 deserialized it successfully on GPU 1. No installed file or running process was changed.
- The committed Python/notebook files were copied to `/home/rlandman/PO4AO_2026` for the fake-stream test. The four replaced files are recoverable from `/home/rlandman/PO4AO_2026_backup_before_local_output_20260921`. No application was restarted and no deployed engine or configuration was changed during this copy.
- The allocation-free C++ history/order and limiter code compiles with `-Wall -Wextra -Werror` and its smoke test passes. The added Catch2 test translation unit also compiles locally.
- Candidate-engine pre-warming compiled and linked on exao2 against CUDA 13 and TensorRT 10. The existing PO4AO unit test still passed 21 assertions in four cases. The resulting binary was not installed and the running application was not restarted.
- All touched C++ files were formatted with the available project-compatible `clang-format`.
- Commit `109c17aa` was compiled and linked in the isolated RTC home checkout `/home/rlandman/MagAOX-po4ao-49fa3eb0` with the installed GCC 14, CUDA, TensorRT 10, ImageStreamIO, mxlib, and EDT environment. All dynamic libraries resolved, and the PO4AO unit test passed 17 assertions in three cases under the RTC `-ffast-math` flags. The binary was not installed or executed.
- The app Makefile now tracks `po4ao_impl.hpp` and `po4ao_utils.hpp`, so changes to either header force recompilation. Build the standard `flatlogs_all` and `basic` prerequisites first in a new checkout, and build the app serially; the parallel direct app build exposed an existing dependency-order race.
- The per-mode numerical limit (`max_command_abs = 1.0`) preserves the simulation scale but is not asserted to be a safe physical modal limit. Establish an instrument-specific value with the loop open before enabling control.

## RTC registration

The configuration repository already contains both standard registrations: `po4ao po4ao` in `proclist_RTC.txt` for `xctrl`, and `po4ao` in the local driver list in `isRTC.conf` for INDI. No extra startup script or separate shmim-creator process is required. The notebook is intentionally not an `xctrl` process; start it only after the real-time app is healthy and stable.

The live RTC inspection on 2026-09-18 found that `aol1_mfiltINPUT` is a link to the 1563-element `aol1_modevalWFS` stream, while the validated simulation model, noise file, and runtime config use 25 modes. The 25-element `po4ao_output` must therefore not be connected directly to loop-1 `mfilt`. Physical integration requires either a 1563-mode model or an explicit, verified 25-to-1563 modal selection/embedding step; that choice is intentionally not guessed here.

## Missing reset_model property: 2026-10-06

The local and `/home/rlandman/MagAOX` source checkouts already register
`po4ao.reset_model.toggle` in `appStartup()` and contain its callback and
frame-boundary reset handling. Their `po4ao.hpp` and `po4ao_impl.hpp` hashes
match. No new property registration or INDI-server configuration is needed.

The installed `/opt/MagAOX/bin/po4ao` is dated 2026-09-24 15:01 UTC and does
not contain the `reset_model` property name. The already-built home-checkout
binary is dated 2026-09-25 and does contain it. A read-only INDI query finds
`po4ao.control_enable.toggle=On` but no `po4ao.reset_model.*`. The missing
property is therefore a deployment mismatch, not a missing source implementation.

This checkout synchronization does not build, install, restart, or modify the
live configuration. The property will remain absent until the updated app is
installed and restarted in a separately authorized operation. Then verify it
with `getINDI -t 3 'po4ao.reset_model.*'` before issuing a reset. The reset's
live behavior has not been exercised during this source-only update.
