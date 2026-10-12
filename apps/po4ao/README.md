# Modal PO4AO real-time application

PO4AO receives reconstructed WFS modal values and publishes a differential
modal input for downstream `mfilt`. Leakage, modal integration, actuator-space
conversion, and physical DM limits remain downstream. The Python trainer and
instrument configuration live in separate repositories.

## Runtime behavior

- `Nmodes` is the full float32 stream width. `Nmodes_controlled` is the leading
  modal prefix controlled by the policy; all higher modes pass through.
- Startup uses proportional fallback and model ID zero. Persisted models are
  ignored until an explicit `reload_engine` request.
- With `control_enable` Off, controlled modes use
  `integrator_gain * observation`, without exploration, and replay collection
  pauses. This is transparent pass-through when the gain is one and commands
  remain within the configured limit.
- With control On, warm-up uses the same fallback plus scheduled noise. A
  compatible published model replaces it once warm-up/history are ready.
- `reset_model` deactivates the selected policy and clears history/partial
  replay at the next input frame. Only a reload from the new reset generation
  may activate afterward; model destruction stays on the background thread.
- `recollect_warmup` invalidates replay snapshots and restarts warm-up/history.
- Repeated semaphore notifications for the same source counter are ignored,
  without republishing output or changing history/replay state.
- Source-counter gaps reset history and the partial replay episode. The gap
  frame uses pass-through with control On, or proportional fallback with it Off.
- Normal controlled outputs reject non-finite values and use the configured
  per-mode absolute limit. This modal limit does not represent actuator clipping.
- Shutdown stops the monitor and workers, publishes zeros, and releases resources.

Neither INDI control switch closes or opens the independently managed CACAO loop.
Follow normal instrument procedures for routing, calibration, and loop operation.

## Models and history

`max_Nhist` preallocates the maximum retained history. Each model supplies its
actual history length and trained modal width. Compatible models satisfy
`Nmodes_controlled <= trained_modes <= Nmodes` and do not exceed the history
capacity.

The fixed float32 input is batch-one NCHW with `2 * history - 1` planes:
current observation, older observations from oldest to newest, then older
commands from oldest to newest. The output contains one value per trained mode.

`linear_inference_backend` selects `auto`, `cpu`, or `tensorrt`. CPU inference
uses the matching `.linear` sidecar emitted by the trainer; it evaluates the
same per-mode temporal weights without CUDA synchronization. TensorRT candidates
are loaded, validated, warmed, and captured into a CUDA graph by the loader.
The real-time callback launches the graph on the dedicated inference stream.
The existing TensorRT/CUDA build dependencies remain required for both backends.

Reloaded models are checked again at activation. If reset occurs after the
loader's check, a stale candidate is rejected and returned for background
destruction. An occupied retirement slot defers activation rather than replacing
an owned pointer. The loader publishes ready status before releasing the pending
pointer so it cannot overwrite the idle status set by activation.

## Replay interface and telemetry

The app creates its configured output, rolling replay, warm-up replay, and
metadata streams. It refuses to replace an incompatible output stream and may
recreate stale transient replay/metadata streams. Reopen Python stream handles
after a restart that changes dimensions.

Replay streams have logical shape `(Nmodes, replay_buffer_size, 2)` and float32
values: observation in plane 0 and the emitted modal command in plane 1.
Snapshot samples are stored newest first. Metadata has four uint32 values:
valid sample count, episode ID, and the low/high words of the publication-time
model ID. Replay and metadata publication counters match.

A reader must copy only while both write flags are clear, counters match, and
neither counter changes during the copy. A valid count of zero invalidates the
snapshot. The background publisher performs replay-sized copying and clearing;
the real-time callback fills preallocated episode slots and drops an episode if
publication cannot keep up.

Replay records PO4AO's emitted commands before downstream integration/clipping.
It does not measure the actual applied DM command, and a model can change within
an episode. Those distinctions matter for training-data quality checks.

`runtime_stats` reports callback p50/p95/p99/max latency and cumulative missed
source frames, deadline misses, inference failures, and replay drops.
`model_status` identifies the selected policy, inference backend, and reload
state. The latency ring contains the latest 1024 callback samples; its maximum
is not the maximum over the entire logging interval. These timings exclude
upstream notification and downstream `mfilt`/DM scheduling.

## Build and tests

Build/install only this app using the standard app Makefile on a configured
instrument development environment. CUDA and TensorRT include/library paths
are inherited from its Makefile; helper headers are explicit build dependencies.
Operator-side example:

```sh
cd /home/rlandman/MagAOX/apps/po4ao
make
sudo make install
```

The registered Catch2 suite tests CPU handoff ownership/generation, source-counter
continuity and duplicates, feature ordering, CPU linear inference, controlled-prefix
pass-through, command limiting, and replay reset behavior. Its helpers do not
require camera or DM access.
On a configured MagAO-X development environment:

```sh
cd /home/rlandman/MagAOX/tests
make -f Makefile.one t=../apps/po4ao/tests/po4ao_test.cpp
../apps/po4ao/tests/po4ao_test '[po4ao]'
```

The suite does not substitute for compiling the full app against CUDA/TensorRT
and running the operator's fake-stream/lab acceptance test. Publication to GitHub
does not install a binary, replace the live config, or restart the application.
