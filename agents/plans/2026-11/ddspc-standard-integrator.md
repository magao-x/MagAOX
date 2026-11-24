# DDSPC integrator (mfilt-equivalent output, standard and assisted)

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
| `enabled` | `false` | Integrate both modes' commands; train both fits on the deltas. |
| `gain` | `1` | `g_m` per WFS frame; with 1, `parameters.gain` is the loop gain on `y`. |
| `mult` | `1` | `mc` per WFS frame, in (0, 1]; below 1 the state leaks. |
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

## Assisted mode (per-slot integrator)

User direction (2026-11-24): output stays on syncopation, and both modes integrate.

- The DM worker integrates every slot: `s = mult_q·s − gain_q·delta`, clipped, with `gain_q = gain/q` and
  `mult_q = mult^(1/q)`. Four slots of a constant delta then equal one mfilt frame (exactly when `mult = 1`), so the
  configured `gain` and `mult` are per WFS frame in both modes and copy the deployed mfilt values. The per-slot delta is
  the current assisted command (`gain·y` + phase prediction + per-slot probe).
- The assisted fit is trained on per-slot deltas: `slotInputs()` converts the copied slot positions of each interval,
  `−(p_n − mult_q·p_(n−1))/gain_q`, using the newest slot of the previous interval. `step()` passes the deltas to
  `stepAssisted()` in place of the positions, so the multirate learner, lift and gains are unchanged in form.
- The executed-slot prefix the worker passes to `MultirateDDSPC::command` uses the same conversion on the copied ring.
- A slot without a current frame holds `s`. The state is zero while prediction is off, on re-enable, and on a
  non-finite command.
- With delta inputs, `ripple` penalizes delta variation within a frame rather than position variation. Re-tune it if
  needed.

## Open items (operator)

- Deployed mfilt tip/tilt gain, mult and limit; initial `[integrator]` values should copy them.
- Deployed `loPredCtrl.conf` (`is_integrating`, `gain`, `history`, `future`): if `is_integrating=true`, the original
  integrated twice and the target behaviour differs.
- Sign check: with the integrator, the feedback sign matches mfilt only if `dm.modeCube` planes 0 and 1 use mfilt's DM
  mode sign.
- A fault zero still removes the whole integrator state in one slot; a ramp-down can be added later.

## Execution notes

- `loPredCtrlAcc_operational.hpp`: `[integrator]` configuration; `standardInput()` (mean copied position, or the
  recovered frame delta); `slotInputs()` and `slotIntegrator()` for assisted; state updates in `step()` (standard) and
  `assistedCandidate()` (assisted). Both input helpers run on every decision, so the previous positions stay current
  across WFS gaps and mode changes; `stepAssisted()` uses the frame input for the frozen standard fit.
- Config in `loPredCtrlAcc.conf` (and its byte-identical `examples/operational/` copy): section present, `enabled=false`.
- Tests (`tests/rtc_controller_test.cpp`, tag `[integrator]`):
  - Standard law: state update, hold on all four slots, clipping, zero while prediction is off, restart from zero,
    frame-delta recovery.
  - Standard closed loop with a constant offset (covariance and QR RLS): residual ratio 0.0028, mean residual ~3e-5.
  - Assisted law: per-slot update with `gain/4` and `mult^(1/4)`, clipping, four slots equal one frame at `mult = 1`,
    per-slot delta recovery.
  - Assisted closed loop (test plant, ACC, offset 0.5/−0.3): mean residual −0.0067/−0.0002, residual/open power
    0.27/0.017 after freezing.
- Results on `ASTR-LT-428.local`: `check-rtc-controller` 27/27, `check-operational-controls` 16/16 plus the config
  unit tests, `check-operational-output` 5/5, and T1 `standard_equivalence.py --base HEAD`: rendered standard values
  bit-for-bit identical with the integrator off.
- Formatting: the repository's clang-format (Apple 21, `InsertBraces`) would reformat hundreds of existing lines in
  both files; only the changed line ranges were formatted. Whole-file formatting is left for a separate commit.
