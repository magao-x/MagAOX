# loPredCtrlAccGUI

Operator panel for `loPredCtrlAcc` (DDSPC tip/tilt control). It replaces the common `cursesINDI` interactions and talks only to INDI; it never touches shared-memory streams, the AO loop or cacao.

Build with `make -C gui/apps/loPredCtrlAccGUI` (it is also in the top-level `all_guis` list). Run `loPredCtrlAccGUI`, or `loPredCtrlAccGUI <device>` for an INDI device name other than `loPredCtrlAcc`. The panel uses the standard workstation INDI connection at `127.0.0.1:7624`.

## Safety behavior

- **Opening or closing the panel sends nothing.** It never turns on learning, prediction or DM output by itself, and it does not change the modal-filter input or run cacao commands.
- **Every control shows the controller's INDI readback.** Toggles move only when `loPredCtrlAcc` reports the new state, and return to the reported state after 5 s if it does not.
- **Controls are disabled when the readback is missing**, when the controller is not `OPERATING`, after a lost connection, and when the device leaves the INDI server.
- **Controller mode and Reset models** are offered only when prediction is off and `safety.zero_confirmed=true`, the state in which `loPredCtrlAcc` accepts them. The controller still validates every request.
- **Reset models asks for confirmation.**
- **Prediction off leaves tip/tilt uncorrected.** `loPredCtrlAcc` always publishes zero for modes 0 and 1 to the main AO loop, so turning prediction off does not restore main-loop tip/tilt correction. The panel states this permanently.
- **Telemetry age.** The status panel shows the age of the latest update. After 10 s without one it marks values as possibly stale. `loPredCtrlAcc` publishes only on change, so a quiet idle controller can trigger this.

## Controls

| Control | INDI property and elements |
| --- | --- |
| Standard / ACC Assisted | `ddspc_mode.standard`, `ddspc_mode.accel_assisted` (both sent, exactly one `On`) |
| Standard sequence | `learn_std.toggle`. Turning it on also turns on learning and prediction, so probes and feedback reach the DM. |
| Learning | `learn.toggle` |
| Reset models | `reset_model.request` |
| Prediction | `predict.toggle` |
| Gain (entry and slider) | `gain.current`/`target`. The slider range and step come from the property's own min/max/step (−10 to 10, step 0.001). |
| Advanced > Exploration | `exploration_sequence.current`/`target` (count,sigma,regularization triples) |

## Status

| Display | INDI source |
| --- | --- |
| Prediction banner, selected-model readiness | `ddspc_control.prediction`, `requested_mode`, `standard_ready`, `assisted_ready` |
| Controller, mode, learning, prediction | `fsm.state`, `ddspc_control.*` |
| Standard / ACC-assisted model | `ddspc_control.standard_ready`/`standard_state`, `assisted_ready`/`assisted_state` |
| WFS camera rate | `camwfs.fg_timing.acq_fps` (loPredCtrlAcc does not publish its own WFS rate) |
| ACC rate, ACC sync, blockers | `adc_values.rate_hz`/`valid`, `ddspc_control.accel_sync`, `assisted_qualification.blockers` |
| Aux DM output, zero confirmed | `safety.output_state`, `dm_write_enabled`, `zero_confirmed` |
| Aux publish rate, transport, slots | `ddspc_performance.publish_rate_hz`, `transport_*`, `slot_shifts`, `step_mismatch` |
| Learning time and overruns | `ddspc_performance.learn_us_mean`/`p99`/`max`, `learn_overruns` |
| Main-loop modal output | `fg_shmimName.name`, `fg_frameSize.width` |
| Warnings | Counters that rose in the latest update (`transport_skipped`, `slot_shifts`, `step_mismatch`, `fast_holds`, `acc_anomalies`, `learn_overruns`), an alert state on `safety` or `predict`, and a controller that is not `OPERATING` |
| Advanced > Diagnostics | The remaining `assisted_qualification` elements and the per-slot `ddspc_performance` rates, counters, margins and RMSEs |

## Tests

The socket-free regression tests run with an offscreen Qt platform:

```bash
cd gui/widgets/loPredCtrlAcc/tests
qmake -makefile loPredCtrlAccPanel_test.pro
make -f makefile.loPredCtrlAccPanel_test
QT_QPA_PLATFORM=offscreen ./loPredCtrlAccPanel_test
```
