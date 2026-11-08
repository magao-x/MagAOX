# Modal PO4AO real-time application

PO4AO receives reconstructed WFS modal values and publishes a differential
modal input for downstream `mfilt`. Leakage, modal integration, actuator-space
conversion, and physical DM limits remain downstream. The shared Python training core lives in `apps/po4aoTrainer`; instrument
configuration is maintained separately.

## Runtime behavior

- `Nmodes` is the full float32 stream width. `Nmodes_controlled` is the leading
  modal prefix controlled by the policy; all higher modes pass through.
- Startup uses proportional fallback and model ID zero. Persisted models are
  ignored until an explicit `reload_engine` request.
- With `control_enable` Off, controlled modes use `integrator_gain * observation`.
  A retained model is not activated. Explicit recollection adds exploration to
  this integrator output; it does not enable the retained policy.
- Initial exploration starts on the first enabled input frame. A loaded policy
  can run as soon as its history is ready; collection never forces integrator
  control. Outside initial or explicitly requested collection, no noise is added.
- `reset_model` deactivates the selected policy and clears history/partial
  replay at the next input frame. Only a reload from the new reset generation
  may activate afterward; model destruction stays on the background thread.
- `recollect_warmup` invalidates replay snapshots and starts a new generation,
  resetting collection counters and partial episodes while preserving the
  controller's history and policy. It adds the configured noise once, after
  the base controller and before command limits. History/replay records the
  command only after output publication succeeds.
- Collection stops exploration after the configured number of emitted frames.
  It becomes ready only when the background publisher has written every
  required episode into the frozen snapshot. Control changes, reset, source
  gaps, inference fallback, invalid commands and publication backlog interrupt
  an unfinished collection. An interrupted generation cannot become ready.
- History follows emitted commands even with control Off. Rolling replay
  continues after explicit integrator recollection; control-Off startup alone
  does not start exploration or collection.
- Repeated semaphore notifications for the same source counter are ignored,
  without republishing output or changing history/replay state.
- Source-counter gaps reset history and the partial replay episode. The gap
  frame uses pass-through with control On, or proportional fallback with it Off.
- Normal controlled outputs reject non-finite values and use the configured
  per-mode absolute limit. This modal limit does not represent actuator clipping.
- Shutdown stops the monitor and workers, publishes zeros, and releases resources.

Policy Enable/Disable and Reset do not control the independently managed CACAO loop.
Whole-app routing below requires the loop already open; closure remains operator-managed.

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
valid sample count, episode ID, and the low/high words of the effective
model ID at the final emitted frame of that episode (zero for integrator). Replay and metadata publication counters match.

A reader must copy only while both write flags are clear, counters match, and
neither counter changes during the copy. A valid count of zero invalidates the
snapshot. The background publisher performs replay-sized copying and clearing;
the real-time callback fills preallocated episode slots and drops an episode if
publication cannot keep up.

Replay records PO4AO's emitted commands before downstream integration/clipping.
It does not measure the actual applied DM command. Ordinary rolling replay can
span policy changes; the frozen warmup accepts only one effective controller
and policy for its entire collection. Those distinctions matter for training-data quality checks.

`runtime_stats` reports callback p50/p95/p99/max latency and cumulative missed
source frames, deadline misses, inference failures, and replay drops.
`model_status` distinguishes the retained model (`active_id`, `active_backend`)
from the last emitted command (`effective_controller`, `effective_id`). Controllers
are `unknown`, `integrator`, `po4ao` or the existing source-gap `pass-through`.
`control_request`/`control_applied` and `reset_request`/`reset_applied` are decimal
sequence/generation strings. A control/reset request remains Busy until its
selected state has produced an output frame while running. When fully stopped and detached,
non-real-time app logic applies model/reset/control changes without requiring frames; this
acknowledges preparation, not active output. Duplicate input notifications do not acknowledge requests. Reload stays Busy through pending activation; a load
failure stays Alert with `reload_error` until another explicit reload.

`collection_status` is read-only text with `state`, `requested_generation`,
`generation`, `frames` and `target_frames`. States are `idle`, `pending`,
`collecting`, `publishing`, `ready` and `interrupted`. A supervisor must require
`ready` for its requested generation **and** a valid paired snapshot; the request
switch returning Off is not completion. Recheck this generation/state after
copying a snapshot, since a new request or interruption can invalidate it.
The generation may also advance when the monitored stream is reallocated.
Controller/model/collection readbacks and acknowledgements update in every app
logic cycle, independently of the slower statistics period. Readbacks precede
terminal request acknowledgements.

The latency ring contains the latest 1024 callback samples; its maximum is not
the maximum over the entire logging interval. These timings exclude upstream
notification and downstream `mfilt`/DM scheduling.

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

The registered Catch2 suites cover CPU model handoff, history/feature ordering,
command limiting, exploration scheduling and collection generations. The app-local
suite also includes the production frame/episode/publication methods against
CPU output/inference boundaries: policy and integrator recollection, duplicate
frames, interrupted data, queue backlog, publication races, and emitted-command
history/replay. The harness does not replace these algorithms with copies.
It does not exercise CUDA, the full MagAOXApp lifecycle, INDI serialization or
physical downstream behavior.

Local CPU tests (including the production-method boundary tests):

```sh
make -C apps/po4ao/tests
make -C apps/po4ao/tests asan
```

On a configured MagAO-X development environment:

```sh
cd /home/rlandman/MagAOX/tests
make -f Makefile.one t=../apps/po4ao/tests/po4ao_test.cpp
../apps/po4ao/tests/po4ao_test '[po4ao]'
```

The suite does not substitute for compiling the full app against CUDA/TensorRT
and running the operator's fake-stream/lab acceptance test. Publication to GitHub
does not install a binary, replace the live config, or restart the application.

## Whole-app Start/Stop and idle telemetry

`run_enable.toggle` controls the whole app independently of policy `control_enable`.
Stop reserves the actual NN/CACAO writer, switches downstream to `aol1_modevalWFS`,
applies `default`, restores that writer and verifies advancing mfilt consumption before
requesting READY and waiting for detachment. Start first establishes fresh output,
then performs the corresponding switch to `po4ao_output` and verifies consumption after
calibration restarts. The policy remains selected. Managed learning stays paused until
an explicit learning request. The optional trainer is not needed for native routing.

`run_status` exposes `phase`, actual/unknown `route`, `error`, `owner` and `verified`.
Startup remains OPERATING with an unverified route because downstream may already use
this output. A routing request requires an already-open HOWFS loop, the inspected milk
utilities, output channel `po4ao_output` and the updated `nnReconstructor` device. A
90-second worker deadline bounds the procedure. Utility exit alone never establishes
completion. Failed Stop keeps PO4AO available if downstream may depend on it. Repeating
a verified target rechecks consumption without repeating calibration. Do not issue
competing manual routing/writer commands during an operation.

The periodic INFO latency line is now `TELEM_PO4AO` typed telemetry via `dev::telemeter`.
INDI `runtime_stats` remains available. Records contain callback percentiles, sample count,
sequence and monotonic sample time, fresh/mixed-window flags, exact uint64 counters,
model identity, backend and deadline. Empty windows have zero samples; idle windows are
marked stale. Mixed windows identify their newest included sample. Sorting and serialization
run outside the frame callback. These are callback timings, not end-to-end latency.

The [engineering plan](../../agents/plans/2026-11/shmimMonitor-idle-apps.md) records utility
source hashes, failure recovery, local tests and required isolated-host/operator acceptance.
The local fake-ABI routing tests do not establish installed milk ABI or deployment readiness.
