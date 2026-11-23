# loPredCtrlAccGUI

Concise operator panel for `loPredCtrlAcc` (DDSPC tip/tilt control): the core predictive-control controls and essential health monitoring. Use `cursesINDI` for everything else (exploration sequence, detailed timing and transport diagnostics). It replaces the common `cursesINDI` interactions and talks only to INDI; it never touches shared-memory streams, the AO loop or cacao.

Build with `make -C gui/apps/loPredCtrlAccGUI` (it is also in the top-level `all_guis` list). Run `loPredCtrlAccGUI`, or `loPredCtrlAccGUI <device>` for an INDI device name other than `loPredCtrlAcc`. The panel uses the standard workstation INDI connection at `127.0.0.1:7624`.

## Safety behavior

- **Opening or closing the panel sends nothing.** It never turns on learning, prediction or DM output by itself, and it does not change the modal-filter input or run cacao commands.
- **Every control shows the controller's INDI readback.** Toggles move only when `loPredCtrlAcc` reports the new state, and return to the reported state after 5 s if it does not.
- **Controls are disabled when the readback is missing**, when the controller is not `OPERATING`, after a lost connection, and when the device leaves the INDI server.
- **Controller mode and Reset models** are offered only when prediction is off and `safety.zero_confirmed=true`, the state in which `loPredCtrlAcc` accepts them. The controller still validates every request.
- **Reset models asks for confirmation.**
- **Who corrects tip/tilt.** With prediction on, `loPredCtrlAcc` sends zero for modes 0 and 1 to the main AO loop, so DDSPC alone corrects tip/tilt. With prediction off, the main loop receives unchanged frames and corrects tip/tilt itself, but only after DDSPC's own-channel zero is confirmed; until then tip/tilt is uncorrected. The prediction banner and the Main-loop tip/tilt row show which (`ddspc_control.main_loop_tt`).

## Controls

| Control | INDI property and elements |
| --- | --- |
| Standard / ACC Assisted | `ddspc_mode.standard`, `ddspc_mode.accel_assisted` (both sent, exactly one `On`) |
| Standard sequence | `learn_std.toggle`. Turning it on also turns on learning and prediction, so probes and feedback reach the DM. |
| Learning | `learn.toggle` |
| Reset models | `reset_model.request` |
| Prediction | `predict.toggle` |
| Gain | `gain.current`/`target` |

## Status

| Display | INDI source |
| --- | --- |
| Prediction banner | `ddspc_control.prediction`, `main_loop_tt` |
| Controller | `fsm.state` |
| Standard / ACC-assisted model | `ddspc_control.standard_ready`/`standard_state`, `assisted_ready`/`assisted_state` |
| ACC rate, ACC sync | `adc_values.rate_hz`/`valid`, `ddspc_control.accel_sync` |
| Aux DM output | `safety.output_state` |
| Main-loop tip/tilt | `ddspc_control.main_loop_tt`: `zeroed` (DDSPC only) or `passthrough` (main loop) |
| Learning overruns | `ddspc_performance.learn_overruns` (assisted-mode WFS-thread work that delays the next main-loop frame) |
| Warnings | A controller that is not `OPERATING`, and an alert state on `safety` or `predict` |

## Tests

The socket-free regression tests run with an offscreen Qt platform:

```bash
cd gui/widgets/loPredCtrlAcc/tests
qmake -makefile loPredCtrlAccPanel_test.pro
make -f makefile.loPredCtrlAccPanel_test
QT_QPA_PLATFORM=offscreen ./loPredCtrlAccPanel_test
```
