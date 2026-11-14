# Compact native input diagnostics

The native application defaults to 11 application properties and 59 elements.
Standard MagAO-X framework properties, including its FSM/log controls, remain.
Only `ddspc_mode` is writable; it records a request and cannot qualify a model or
enable training, prediction or output. Individual 8-kHz ADC records, 2-kHz WFS
records and complete observed DM command images are retained without averaging.

| Property | Elements |
| --- | --- |
| `app_health` (text) | `state`, `uptime_s`, `inputs`, `shadow`, `error_events` |
| `input_health` (text) | `adc`, `modal`, `camera`, `dm` |
| `input_rates` (number) | `adc_capture_hz`, `adc_forward_hz`, `modal_hz`, `camera_hz`, `dm_hz` |
| `adc_values` (number) | `raw0`, `raw1`, `acceleration0`, `acceleration1`, `invalid`, `rail_rejected` |
| `input_losses` (number) | `adc_dropped`, `adc_forward_missing`, `modal_gaps`, `camera_gaps`, `dm_gaps`, `counter_discontinuities`, `reader_races`, `adc_breaks`, `timing_breaks` |
| `input_buffers` (text) | `history_occupancy`, `history_evictions`, `adc_forward_pending`, `recorder_backlog`, `recorder_overflow` |
| `timing` (text) | `phase_offset_us`, `phase_jitter_us`, `clock`, `exposure`, `modal_association`, `slots_per_wfs` |
| `ddspc_mode` (switch) | `standard`, `accel_assisted` |
| `ddspc_control` (text) | `requested_mode`, `effective_mode`, `training`, `prediction`, `standard_ready`, `assisted_ready` |
| `ddspc_performance` (number) | `standard_updates`, `standard_predictions`, `standard_rmse`, `assisted_updates`, `assisted_predictions`, `assisted_rmse`, `compute_us` |
| `safety` (text) | `dm_write_enabled`, `output_reason`, `timing_interlock`, `model_interlock` |

`input_health` values report transport state plus the source restart count. A
valid static DM request is `idle_valid;restarts=0`, with zero DM rate. A mapping
with no first publication is `idle_no_command_yet`; absence is `disconnected`.
Neither case prevents healthy ADC/WFS observation. Overall `app_health.inputs`
requires healthy ADC forwarding, modal and camera inputs, independently of DM.
`state` describes the observation worker; the framework FSM remains separate.

`error_events` sums reported loss/rejection/recorder error categories. Categories
can overlap; it is not a unique count of lost physical samples. Expected history
evictions, duplicate polls, unknown clocks and unavailable modal associations are
excluded. Use the separate loss counters for diagnosis.

Updates are admitted at `[diagnostics] period_ms=250` by default (50–10000 ms).
Values are sent only when changed. Uptime changes once per second. Acknowledging
an explicit mode request is immediate; acquisition runs independently of INDI.
Native model update/prediction counters come from the existing independent model
states. RMSE and computation latency are -1 when unavailable, never invented zeros.

## ADC forwarding investigation

`adc_capture_hz` measures accepted acquisition intervals. `adc_forward_hz`
measures samples actually copied by the native observer against local observer
time. They are independent: a healthy 8-kHz reader does not establish forwarding
health. In particular, `reader_healthy_forwarding_stale` remains a degraded state
and reports zero forwarding rate; the application does not substitute capture rate.

The previous observer captured `now = CLOCK_MONOTONIC` before entering the ADC
reader mutex and reused it across bounded copies. `AdcCapture::copy_after()`
correctly excludes samples with decoder receipt later than that frozen cutoff.
If lock/scheduling delay advances acquisition beyond the cutoff, retained records
can all be withheld while acquisition remains valid. Ring eviction can then add
forwarding loss independently of capture loss. A deterministic regression reproduces
healthy 8-kHz capture with no forwarded samples under an obsolete cutoff.

The native observation path now calls `AdcReader::copy_ready()`. It measures a
fresh cutoff **after acquiring the reader lock**, then calls the same bounded
`AdcCapture::copy_after()` processing. The original frozen-cutoff API is unchanged
for replay/control decisions: observation availability is not a past WFS deadline.
No cursor reset hides loss, no generation is invented, and no timestamp is changed.
The observer still limits batches and preserves chronological source/segment IDs.

This fixes a demonstrated cutoff/starvation hazard. The reported exao2 values
alone do **not** prove that lock delay caused that particular sustained stall.
There was no contemporaneous cursor/cutoff/receipt trace. Its exact trigger remains
to be confirmed in the next operator-run test, rather than declared solved from
the acquisition rate alone. A single `adc_breaks` increment cannot distinguish an
overwrite, rejected sample, source transition, timing gap or prior stale segment.
The observer's 10-ms silence threshold and the ADC reader's default 20-ms age
threshold also differ: a brief delay can show the reported valid/stale combination
transiently. Neither threshold qualifies a 125-us control deadline. Repeated
snapshots/counters are needed to distinguish that transient from a sustained stall.

Snapshots now refresh receipt-based silence at readout, so a delayed observation
worker cannot leave cached health looking current. No clock-host/exposure mapping
is inferred. New pending/cutoff diagnostics distinguish backlog from samples beyond
the copy cutoff. Health transitions are logged as `native_input_health` key/value
records with source, counter, generation, segment and ADC forwarding position.

## History, loss and association semantics

- `history_occupancy` lists retained native records per source (`used/capacity`).
  It is **not** pending work. Full histories are expected. `history_evictions`
  count expected oldest-record removal. The independent ADC reader has its own
  retention/overwrite counters in debug mode.
- `adc_forward_pending` is unforwarded ADC data still in reader retention at the
  last copy. `recorder_backlog` and `recorder_overflow` refer only to the optional
  metadata/command disk queues, separately labelled. With recording off both are zero.
- `adc_dropped` is the reader's unavailable/rejected source-frame count.
  `adc_forward_missing` is missing counters after an established forwarding cursor;
  these populations can overlap and must not be added as unique losses.
- Modal/camera/DM `*_gaps` count actual source counter advances not copied by this
  latest-frame observer. They are acquisition coverage loss, **not** retention
  eviction and not necessarily producer or camera hardware loss.
- `counter_discontinuities` counts observed resets/reattachments across inputs.
  `reader_races` counts rejected unstable snapshots; a retry can succeed, so this
  count is not itself a lost-frame count. `timing_breaks` combines rejected ADC
  timestamp frames, ADC interval gaps and non-ADC timestamp regressions.
- Debug `sequence_gaps` belongs only to the optional completion/SDK wire protocol.
  A zero value on modal/camera says nothing about their `counter_gaps`.
- `NativeObservationCapture::accept()` deliberately flags every modal record as
  missing association: there is no verified camera-origin/exposure envelope or
  clock mapping. `timing.modal_association=camera_origin_and_exposure_contract_missing`
  describes that missing adapter/evidence. Equal counters or nearby publications
  cannot replace it. These expected counters never enable training.

The nominal architecture remains four individual 125-us ADC/auxiliary intervals
per 500-us WFS frame. The recording/observer does not establish their physical
phase lock. Phase/jitter, exposure boundaries, source-host clock mapping, modal/DM
coordinates and physically applied DM intervals remain unqualified. Future output
must use synco slot/input-ready contracts; nominal 15-us write slack is not a
DDSPC computation allowance. No output path was added.

## Detailed opt-in interface and migration

The former default `adc`, `adc_source`, `native_*`, `native_*_identity`,
`native_shadow`, detailed model/input/safety properties and obsolete Legacy
learn/predict/integrate/save/load/FPS controls are not registered in compact native
mode. They remain available only with explicit startup debug configuration:

```ini
[diagnostics]
period_ms = 250
debug_indi = true
```

This selects the detailed schema instead of the compact one; changing schemas
requires an app restart. It grants no additional control authority. Debug
`input_queue_occupancy/capacity` were renamed `history_retained/capacity`.
The new `native_adc_forwarding` text property preserves exact reader/cursor
counters, generations, segments, next receipt, actual copy cutoff, pending count
and a reason. Existing finite observation journals retain the CSV format and all
per-record metadata; full DM recording remains separately opt-in. Nothing is
written to production streams.

## Build and operator verification

Local tests (repository root, no services or native streams):

```bash
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 \
  BUILD_DIR=build/native-app-integration check-native-observation
```

Only the operator runs the following on exao2 during an approved app-only update:

```bash
cd ~/source/MagAOX-DDSPC/apps/loPredCtrlAcc
make -j4
xctrl stop loPredCtrlAcc
sudo make install
sudoedit /opt/MagAOX/config/loPredCtrlAcc.conf
```

Preserve the ADC calibration and input selections. In the existing configuration
set `[diagnostics] period_ms=250` and `debug_indi=false`. Installation does not
overwrite existing configuration. Keep shadow on, pipeline/training/Legacy off,
and leave accelSim injection disabled. Validate and start only this app:

```bash
/opt/MagAOX/bin/loPredCtrlAcc -n loPredCtrlAcc --config.validate
xctrl start loPredCtrlAcc
xctrl status loPredCtrlAcc
cursesINDI
```

The existing broker registration is retained. Avoid `xctrl restart`: this xctrl
version also restarts the role INDI server. See [NATIVE_APPLICATION.md](NATIVE_APPLICATION.md)
for initial registration and rollback. No registration change is needed for
renamed properties on the same app. Do not restart producers or the broker here.
The standard framework sends a device-wide INDI deletion on graceful shutdown,
then redefines the smaller schema after startup; no broker restart is needed.

After warmup, observe **at least 120 seconds**, extending beyond the previously
reported ~75-second forwarding count. Require capture and forwarding rates near
8 kHz (within 2%), modal/camera near 2 kHz, advancing forwarding history and no
new capture/forward losses, invalid/rail samples, timing breaks or restarts.
Expected bounded-history evictions may grow; recorder backlog must remain zero
with recording off. DM may stay idle; safety must stay false for DM writes and
both native models must remain unqualified with no updates/predictions. Inspect
the ordinary app log for any health transitions and verify rate admission.

If capture stays healthy but forwarding becomes stale, retain the app logs. At
an authorized app-only restart use `debug_indi=true` and capture the entire
`native_adc_forwarding`, `adc_source`, `adc`, `native_adc`, and `native_shadow`
properties together. That evidence is required to identify the original stall's
precise trigger. Do not enable training or treat stale data as valid to proceed.

## Local validation and changed files

The native suite passes 56 C++ cases, including actual reader/cursor APIs, a
960,000-sample virtual acquisition, stale/recovery, bounded retention, restarts,
counter/race reporting, missing/idle DM, INDI schema/permissions/rate admission and
shadow-only guards. The same suite passes ASan/UBSan. Its Python tools/configuration/
deployment checks pass 44 tests. The broader existing regression passes 133 cases;
confirmed-input replay passes 6 cases. These use local doubles/private ordinary
files, not native instrument streams. No Linux/ImageStreamIO ABI link or actual
broker/cursesINDI interaction was performed on this Mac.

The first broad run aborted with cached objects built against older header layouts:
its dependency files mixed relative and absolute build paths. The offline Makefile
now canonicalizes `BUILD_DIR` and rebuilds objects when that build contract changes.
A fresh build passes; no DDSPC equations were changed to resolve this failure.

Changed application files: `loPredCtrlAcc.hpp`, `loPredCtrlAcc.conf`,
`adc_input.hpp/.cpp`, `adc_reader.hpp/.cpp`, `native_observation.hpp/.cpp`,
`native_observer.cpp`, and `examples/loPredCtrlAcc-native-observe.conf.example`.
Changed test files: `tests/Makefile.offline`, `tests/app_test_support.hpp`,
`tests/ddspc_adc_test.cpp`, `tests/ddspc_native_observation_test.cpp`,
`tests/native_application_test.cpp`, `tests/native_observation_app_test.cpp`,
`tests/native_deployment_test.py`, and `tests/README.md`.
Documentation: this file, `NATIVE_APPLICATION.md`, and the engineering record
`agents/plans/2026-11/accel-ddspc-controller.md`.

`apps/loPredCtrl`, calibration, numerical controllers, operational configuration,
producers and services are unchanged. No install, commit, push or instrument access
was performed.
