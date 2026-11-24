# DDSPC standard-mode integrator (mfilt-equivalent output)

2026-11-24. Branch `parkertjohnson1/accel_ddspc`. Workstation `ASTR-LT-428.local`; no instrument access.

## Why

Sebastiaan's original `apps/loPredCtrl` worked with tweeter-to-woofer offloading. Its command
`new_delta = (K + gain on newest y)·state + noise` replaced modes 0 and 1 of the main loop's modal input, so cacao
`mfilt` integrated it: `DM[k] = mc·DM[k-1] − g_m·new_delta[k]`. The DDSPC model was trained on those deltas, and the
plant it identified included mfilt's integrator, sign, leak and limits. The main loop kept integral action on the WFS
residual, so t2wOffloader moved tip/tilt to the woofer and the integrator bled it off the tweeter.

`loPredCtrlAcc_operational` zeroes modes 0 and 1 in the main-loop input and writes `R·(gain·y + K·state)` to its own
auxiliary channel as an absolute position each slot:

- no integrator, so no integral action on the residual; t2wOffloader, which integrates the tweeter shape, is the only
  integrator left in the tip/tilt path;
- `gain·y` is a proportional term whose sign comes from the mode cube, not mfilt's `−g_m`;
- the optimizer solves for the whole command, then `gain·y` is added outside it (as in the original, where it was an
  increment into mfilt and therefore harmless integral feedback);
- no amplitude limit (`m_producerDefaults=true`), where the original inherited mfilt's.

Running the original loPredCtrl on the RTC (Jared's request) tests the mfilt route itself. The mfilt route cannot carry
8-kHz assisted commands, so the integrator is added inside loPredCtrlAcc.

## Design

Configuration `[integrator]`, default off so existing behaviour and the T1 standard trace are unchanged:

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Integrate standard-mode commands. |
| `gain` | `1` | `g_m`; with 1, `parameters.gain` is the loop gain on `y`. |
| `mult` | `1` | `mc`, in (0, 1]; below 1 the state leaks. |
| `limit` | none | Per-mode absolute limit on the integrator state, modal units; required (> 0) when enabled. |

Standard mode, per WFS frame k (WFS thread, under `m_modelMutex`):

1. `delta_k = gain·y_k + K-part + probe` exactly as now: the first plan column.
2. `s_k = clamp(mc·s_{k−1} − g_m·delta_k, ±limit)`.
3. All plan columns hold `s_k`. A late WFS frame holds the state, as mfilt does, instead of playing the open-loop plan.
4. While prediction is off, or on a numerical fault, `s = 0` and the plan is zero, so prediction always starts from the
   confirmed zero.

Training: the model input is the delta, recovered from the copied commands so it reflects what was actually applied
(missed slots, clipping): `delta_k = −(p_k − mc·p_{k−1}) / g_m`, where `p_k` is the mean of the slots copied since the
previous decision (the held `s` when all were copied). The first frame after a reset or an unknown interval has no
previous position; its applied command is marked unknown. The frozen standard fit's histories, kept continuous while
assisted is selected, use the same conversion.

`gain·y` thus becomes integral feedback again, the model's plant includes the integrator as it did in the original,
and the limit bounds the output.

## Stages

1. **Standard mode** (this change). Validate against the original loPredCtrl run under the same disturbance and offload
   settings.
2. **Assisted mode** (later). Integrating per slot changes the lifted slot structure, the executed-prefix re-solve and
   the `ripple` penalty. Options: integrate the per-frame mean like standard and add the zero-mean within-frame part
   without integration, or a per-slot integrator with a reformulated lift. Decide after stage 1. Until then assisted
   output is unchanged; enabling the integrator only affects standard output and the standard fit's inputs.

## Open items (operator)

- Deployed mfilt tip/tilt gain, mult and limit; initial `[integrator]` values should copy them.
- Deployed `loPredCtrl.conf` (`is_integrating`, `gain`, `history`, `future`): if `is_integrating=true`, the original
  integrated twice and the target behaviour differs.
- Sign check: with the integrator, the feedback sign matches mfilt only if `dm.modeCube` planes 0 and 1 use mfilt's DM
  mode sign.
- A fault zero still removes the whole integrator state in one slot; a ramp-down can be added later.

## Execution notes

- Implementation in `loPredCtrlAcc_operational.hpp`: `[integrator]` configuration, `standardInput()` (mean copied
  position, or the recovered delta), and the state update in `step()`. `standardInput()` runs on every decision so the
  previous position stays current across WFS gaps and while assisted is selected; `stepAssisted()` uses the same input
  for the frozen standard fit.
- Config in `loPredCtrlAcc.conf` (and its byte-identical `examples/operational/` copy): section present, `enabled=false`.
- Tests (`tests/rtc_controller_test.cpp`, tag `[integrator]`):
  - "Standard integrator follows the mfilt law": state update, hold on all four slots, clipping, zero while prediction
    is off, restart from zero, and delta recovery (unknown without a previous position).
  - "Standard integrator corrects and removes a static offset": covariance and QR RLS, plant with a constant offset;
    residual ratio 0.0028, mean residual about 3e-5 after freezing.
- Results on `ASTR-LT-428.local`: `check-rtc-controller` 25/25, `check-operational-controls` 16/16 plus the config
  unit tests, `check-operational-output` 5/5, and T1 `standard_equivalence.py --base HEAD`: 192000 rendered values
  bit-for-bit identical with the integrator off.
- Formatting: the repository's clang-format (Apple 21, `InsertBraces`) would reformat hundreds of existing lines in
  both files; only the changed line ranges were formatted. Whole-file formatting is left for a separate commit.
