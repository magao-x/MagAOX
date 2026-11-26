# loPredCtrlAccGUI

Concise operator panel for `loPredCtrlAcc` (DDSPC tip/tilt control): the core predictive-control controls and essential health monitoring. Use `cursesINDI` for everything else (exploration sequence, detailed timing and transport diagnostics). It replaces the common `cursesINDI` interactions and talks only to INDI; it never touches shared-memory streams, the AO loop or cacao.

Build with `make -C gui/apps/loPredCtrlAccGUI` (it is also in the top-level `all_guis` list). Run `loPredCtrlAccGUI`, or `loPredCtrlAccGUI <device>` for an INDI device name other than `loPredCtrlAcc`. The panel uses the standard workstation INDI connection at `127.0.0.1:7624`.

## Safety behavior

- **Opening or closing the panel sends nothing.** It never turns on learning, prediction or DM output by itself, and it does not change the modal-filter input or run cacao commands.
- **Every control shows the controller's INDI readback.** Toggles move only when `loPredCtrlAcc` reports the new state, and return to the reported state after 5 s if it does not.
- **Controls are disabled when the readback is missing**, when the controller is not `OPERATING`, after a lost connection, and when the device leaves the INDI server.
- **Controller mode and Reset models** are offered only when prediction is off and `safety.zero_confirmed=true`, the state in which `loPredCtrlAcc` accepts them. The controller still validates every request.
- **Reset models asks for confirmation.**
- **DM recovery is always offered while the controller is `OPERATING`,** including during prediction, because both requests only turn prediction off. Neither changes the main loop's own integrator; a tip/tilt command it held while DDSPC owned tip/tilt stays until the main loop corrects it.
- **Who corrects tip/tilt.** With prediction on, `loPredCtrlAcc` sends zero for modes 0 and 1 to the main AO loop, so DDSPC alone corrects tip/tilt. With prediction off, the main loop receives unchanged frames and corrects tip/tilt itself, but only after DDSPC's own-channel zero is confirmed; until then tip/tilt is uncorrected. In Main Loop mode, modes 0 and 1 carry DDSPC's tip/tilt delta instead (`predicted`). The prediction banner and the Main-loop tip/tilt row show which (`ddspc_control.main_loop_tt`).

## Controls

| Control | INDI property and elements |
| --- | --- |
| Standard / ACC Assisted / Main Loop / mfilt Copy 8 kHz / mfilt Copy 2 kHz / mfilt Clone 8 kHz / mfilt Clone 2 kHz | `ddspc_mode.standard`, `ddspc_mode.accel_assisted`, `ddspc_mode.main_loop`, `ddspc_mode.mfilt_copy_8k`, `ddspc_mode.mfilt_copy_2k`, `ddspc_mode.mfilt_clone_8k`, `ddspc_mode.mfilt_clone_2k` (all the controller publishes are sent, exactly one `On`; the copies are on a second row, the clones on a third). Main Loop is standard DDSPC sent through the main loop's mfilt, as loPredCtrl: no DDSPC DM channel, and tip/tilt is never zeroed. The mfilt copies run no DDSPC: with Predict on, the main loop's tip/tilt integrator, copied with its own parameters once per WFS frame, drives the DDSPC DM channel while main-loop tip/tilt is zero; 8 kHz writes it on every DM slot, 2 kHz once per frame. The mfilt clones are the copies started from mfilt's output, `aol1_modevalDM`, so the switch picks up where mfilt left off and the DM total stays mfilt's state; 2 kHz writes only step 0, with the main loop's write. Their banner and the "Clone step 0 (in sync / missed)" status row show `mfilt_clone.step0_in_sync` and `step0_missed`. Each is disabled with a controller that lacks its element. |
| Standard sequence | `learn_std.toggle`. loPredCtrl's three exploration stages, then 1000 frames of learning without probes. Turning it on also turns on learning and prediction, so probes and feedback reach the DM. When it ends, `loPredCtrlAcc` turns learning and this switch off; prediction continues. |
| Learning | `learn.toggle` |
| Reset models | `reset_model.request` |
| Prediction | `predict.toggle` |
| Gain | `gain.current`/`target` |
| Extra learning [s] | `learn_std_extra.current`/`target`: seconds the Standard sequence keeps learning after its last stage before learning turns off (0: at its end). Applies at once, including during a running sequence. Disabled with a controller that does not publish it. |
| Zero DDSPC DM | `zero_dm.request`. Turns prediction off and writes a fresh zero to the DDSPC auxiliary channel (`dm.channelName`), even if the last zero was already confirmed. |
| Return tip/tilt to main loop | `release_tt.request`. Turns prediction off and passes main-loop tip/tilt through from the next WFS frame without waiting for `safety.zero_confirmed`; the zero is still retried. |

## Status

| Display | INDI source |
| --- | --- |
| Prediction banner | `ddspc_control.prediction`, `main_loop_tt`; after a fault, `PREDICTION STOPPED:` and its reason. With prediction on but not writing: `ACC-assisted not writing` when `dm_slots` differs from the DM scheduler's steps per cycle, or `holding the last DM command` with the input condition being held through |
| Controller | `fsm.state` |
| Standard / ACC-assisted model | `ddspc_control.standard_ready`/`standard_state`, `assisted_ready`/`assisted_state` |
| ACC rate, ACC sync | `adc_values.rate_hz`/`valid`, `ddspc_control.accel_sync`. The sync is information only: the ACC may run free (acc000 `synchro.timingSource = integrator`) and the upstream lock gates nothing |
| DM slots / ACC ratio | `rates.dm_slots`/`acc_ratio` in use, with the measured `scheduler_steps` and `acc_per_wfs`; `detecting:` while the property is BUSY (not yet confirmed, or a detected change waiting for prediction off) |
| Aux DM output | `safety.output_state`; a failed or discarded DM write is explained (it never stops prediction) |
| DM write failures | `ddspc_performance.dm_write_failures`: writes that failed (scheduler restart or unavailable, DM channel unavailable, no receipt within `dm.receiptTimeoutMs`); the DM keeps its last command and the next slot is written |
| Commands discarded | `ddspc_performance.transport_discarded`: commands the DM scheduler discarded; the DM keeps the previous command |
| Last stop | Why prediction last stopped: the decoded fault from `safety.output_state`, a stop by request, or `running` |
| Main-loop tip/tilt | `ddspc_control.main_loop_tt`: `zeroed` (DDSPC only) or `passthrough` (main loop) |
| Learning overruns | `ddspc_performance.learn_overruns` (assisted-mode WFS-thread work that delays the next main-loop frame) |
| Warnings | A controller that is not `OPERATING`; an output fault with its reason; DM write failures, discarded commands or `ddspc_performance.step_mismatch` rising (shown for 10 s after each rise); DM slots / ACC ratio not yet confirmed; a refused prediction request with the condition that refused it (`integrator.ready`, not needed in Main Loop mode and always needed in the mfilt copies, then DM slots / ACC ratio still being detected, then the selected model's `ddspc_control.*_ready`/`*_state` with learning off, then `safety.output_state`) |

## Why prediction stopped

`loPredCtrlAcc` reports each fault in `safety.output_state` as the fault code, or `fault_zero_acknowledged` once its zero is copied, followed by `(<fault>: <condition> [<ACC window reason>])`. The panel turns that into a sentence and keeps the raw code in brackets, which matches the controller's `DDSPC output fault, prediction stopped` log entry:

| Code | Shown as |
| --- | --- |
| `input_or_model_qualification_lost: wfs_invalid` | the WFS measurement was invalid (non-finite) |
| `…: integrator_parameters` | the main loop's integrator parameters were unavailable |
| `…: command_invalid` | the selected model's command was not usable (a numerical fault, or no ready model) |
| `…: wfs_stale` | no WFS frame arrived within `input_timeout` |
| `…: acc_window <reason>` | the ACC window was unusable, with the reason (`stale_sample`: no ACC sample within `adc.max_age_us`; `history_filling`, `adc_stream_invalid`, `nonfinite_sample`, `adc_disabled`) |
| `nonfinite_or_command_limit` | a DDSPC command was non-finite or beyond its limit |
| `candidate_exception_zero_pending`, `copy_callback_exception_zero_pending` | an internal error while preparing or recording a DM command |
| `auxiliary_geometry` | the DDSPC DM channel has the wrong geometry |

A lost upstream lock and failed or discarded DM writes no longer stop prediction, so they are not listed: the lock only pauses learning, and write problems are counted and warned about instead.

The reason stays shown until prediction is requested again.

## Tests

The socket-free regression tests run with an offscreen Qt platform:

```bash
cd gui/widgets/loPredCtrlAcc/tests
qmake -makefile loPredCtrlAccPanel_test.pro
make -f makefile.loPredCtrlAccPanel_test
QT_QPA_PLATFORM=offscreen ./loPredCtrlAccPanel_test
```
