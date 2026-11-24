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

`[integrator] enabled` (default `true`, user direction 2026-11-24; `false` keeps absolute positions) turns on the
mfilt law in both modes. DDSPC does not set the filter's parameters (user direction, 2026-11-24): they are always the
main loop's, followed as they change, and DDSPC learns its optimal command for whatever they are. For DDSPC mode `i`
(WFS index `modal_index0/1`), as cacao mfilt combines them:

| Parameter | Source |
| --- | --- |
| `gain_i` | `<loop.device>.loop_gain.current × aol<N>_mgainfact[i]` |
| `mult_i` | `<loop.device>.loop_multcoeff.current × aol<N>_mmultfact[i]` |
| `limit_i` | `<loop.device>.loop_max_limit.current × aol<N>_mlimitfact[i]` |

- `[loop] device` (default `holoop`, the main loop's cacaoInterface) and `number` (default `1`).
- The scalars are INDI set-property subscriptions registered in `setupINDI()`; the factor streams use three tagged
  `dev::shmimMonitor` bases with `m_getExistingFirst`, as modalFilter and userGainCtrl do. Each update is combined under
  `m_loopMutex` and published to the control threads under `m_commandMutex` (`Integrator`); the DM worker copies it
  inside the lock it already takes per slot.
- Per AGENTS.md rule 27 each source keeps its last value until updated; invalid scalar updates are ignored. Until all six
  have been received, with positive gain and limit for both DDSPC modes, `integrator.ready` is false, `outputValid()`
  reports `integrator_parameters`, and `predict`/`learn_std` requests are refused.
- Published as the read-only `integrator` property: `enabled`, `ready`, `gain0/1`, `mult0/1`, `limit0/1`.

Standard mode, per WFS frame k (WFS thread, under `m_modelMutex`):

1. `delta_k = gain·y_k + K-part + probe` exactly as now: the first plan column.
2. `s_k = clamp(mult_i·s_{k−1} − gain_i·delta_k, ±limit_i)`.
3. All plan columns hold `s_k`. A late WFS frame holds the state, as mfilt does, instead of playing the open-loop plan.
4. While prediction is off, or on a numerical fault, `s = 0` and the plan is zero, so prediction always starts from the
   confirmed zero.

Training: the model input is the delta, recovered from the copied commands so it reflects what was actually applied
(missed slots, clipping): `delta_k = −(p_k − mult_i·p_{k−1}) / gain_i`, where `p_k` is the mean of the slots copied since
the previous decision (the held `s` when all were copied). The first frame after a reset or an unknown interval has no
previous position; its applied command is marked unknown. The frozen standard fit's histories, kept continuous while
assisted is selected, use the same conversion.

`gain·y` thus becomes integral feedback again, the model's plant includes the integrator as it did in the original,
and the main loop's limit bounds the output. A change of the main loop's gain or multiplier changes that plant: keep
learning on, or retrain, after one.

## Assisted mode (per-slot integrator)

User direction (2026-11-24): output stays on syncopation, both modes integrate, and assisted divides the gain by q while
leaving the multiplier and limit unchanged.

- The DM worker integrates every slot: `s = mult_i·s − (gain_i/q)·delta`, clipped to `±limit_i`. The per-slot delta is
  the current assisted command (`gain·y` + phase prediction + per-slot probe).
- The assisted fit is trained on per-slot deltas: `slotInputs()` converts the copied slot positions of each interval,
  `−(p_n − mult_i·p_(n−1))/(gain_i/q)`, using the newest slot of the previous interval. `step()` passes the deltas to
  `stepAssisted()` in place of the positions, so the multirate learner, lift and gains are unchanged in form.
- The executed-slot prefix the worker passes to `MultirateDDSPC::command` uses the same conversion on the copied ring.
- A slot without a current frame holds `s`. The state is zero while prediction is off, on re-enable, and on a
  non-finite command.
- With delta inputs, `ripple` penalizes delta variation within a frame rather than position variation. Re-tune it if
  needed.

## Open items (operator)

- Confirm that `holoop` is the cacaoInterface of the loop whose mfilt drives tip/tilt, and that `aol1_m*fact` are that
  loop's factors.
- Confirm cacao mfilt's limit semantics (`|DM| ≤ loop_max_limit × mlimitfact`), which this implementation copies.
- The main loop's tip/tilt gain is always nonzero on holoop (user, 2026-11-24); with zero gain the integrator would stay
  not ready and prediction refused.
- Deployed `loPredCtrl.conf` (`is_integrating`, `gain`, `history`, `future`): if `is_integrating=true`, the original
  integrated twice and the target behaviour differs.
- Sign check: with the integrator, the feedback sign matches mfilt only if `dm.modeCube` planes 0 and 1 use mfilt's DM
  mode sign.
- A fault zero still removes the whole integrator state in one slot; a ramp-down can be added later.

## Execution notes

- `loPredCtrlAcc_operational.hpp`: `[integrator] enabled` and `[loop]` configuration; three tagged factor-stream
  monitors and three cacaoInterface subscriptions; `loopParameter()`, `loopFactors()` and `combineIntegrator()`;
  `standardInput()` (mean copied position, or the recovered frame delta); `slotInputs()` for assisted; state updates in
  `step()` (standard) and `assistedCandidate()` (assisted). Both input helpers run on every decision, so the previous
  positions stay current across WFS gaps and mode changes.
- The configured `gain`, `mult` and `limit` keys of the first version were removed: the main loop supplies them.
- Config in `loPredCtrlAcc.conf` (and its byte-identical `examples/operational/` copy): `[integrator] enabled=false`,
  `[loop] device=holoop`, `number=1`.
- Tests (`tests/rtc_controller_test.cpp`, tag `[integrator]`), with the parameters delivered through the real
  set-property callbacks and factor `processImage()` overloads:
  - Standard law: no parameters → zero output and `integrator_parameters`; per-mode combination of scalars and factors
    (a factor at a non-DDSPC index is ignored); state update, hold, clipping; invalid scalar update retained; a new
    factor takes effect at once; zero while prediction is off, restart from zero; frame-delta recovery.
  - Standard closed loop with a constant offset (covariance and QR RLS): residual ratio 0.0028, mean residual ~3e-5.
  - Assisted law: per-slot update with `gain/4`, unchanged multiplier and limit, clipping, per-slot delta recovery.
  - Assisted closed loop (test plant, ACC, offset 0.5/−0.3): mean residual −0.0067/−0.0002, residual/open power
    0.27/0.017 after freezing.
  - `predict` and `learn_std` refused until the parameters arrive, then accepted; `integrator` property values and
    state.
- The added `processImage`/`allocate` overloads made the existing tests' `allocate( {} )` calls ambiguous; they now name
  `dev::shmimT{}`. `tests/app_test_support.hpp` gained the tagged `SHMIMMONITORT_*` macros.
- Results on `ASTR-LT-428.local`: `check-rtc-controller` 28/28, `check-operational-controls` 16/16 plus the config
  unit tests, `check-operational-output` 5/5, and T1 `standard_equivalence.py --base HEAD`: rendered standard values
  bit-for-bit identical with the integrator off. `make check` stops at `verify-legacy` (pinned `apps/loPredCtrl` tree
  hash) and one `native_linux_build_test` shadow-only guard; both fail identically on `0b756556`, before this work.
- Default on (`m_integrate{ true }`, `enabled = true` in both configs). The test harness `Controller` records the default
  (asserted on) and turns the integrator off for the absolute-position tests; the integrator tests turn it back on.
  `standard_trace_test.cpp` turns it off through a `requires` check, so T1 still compiles and passes against
  `1ff631bb` (no integrator) and against HEAD: 192000 values bit-for-bit identical in both.
- Formatting: the repository's clang-format (Apple 21, `InsertBraces`) would reformat hundreds of existing lines in
  both files; only the changed line ranges were formatted. Whole-file formatting is left for a separate commit.
