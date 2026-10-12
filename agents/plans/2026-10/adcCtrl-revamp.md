# Task Description
The Python app adcCtrl needs to be updated to work more robustly and with a different algorithm. 

## Problem Statement
The adcCtrl app needs a major upgrade. A new algorithm using the method-of-moments method of finding satellite spot pointing angle needs to be implemented. Additionally, the app often crashes and needs to be revised to work more robustly.

## Discussion
The adcCtrl app is a purepyindi app that actively sends commands to stagadc1 and stageadc2 in order to eliminate residual dispersion left over after the primary compensation by the adctrack C++ app. Dispersion amount is estimated using the direction of pointing of satellite spots that are either generated actively by the tweeterSpeck app or by passive actuator print-through on the DM. Read at least the first three sections in the paper at https://arxiv.org/abs/2608.10307. The satellite spot pointing concept will still be used (the offset between the pairs of spots will still be proportional to the dispersion), but a new method for measuring spot angle needs to be implemented. This method, utilizing the method of moments, is already written in the adcCtrl app.py file. The exact syntax of the algorithm can be changed if it has to be, but the algorithm as-written right now has been tested and verified in simulations, so it should not be altered on a fundamental level. 

The current format of the app has the option to set it into one of four states: closed-loop ADC correction, one-shot ADC correction, measure-only (dispersion measurements and potential ADC commands are logged but no correction is sent), and idle. This structure should remain.

Right now, the calibration is performed external to the app. The ADC response is calculated by counter-rotating the prisms by a known amount and calculating the resulting change in satellite spot offset angle, as described in the arXiv paper above. The inverse of the slopes of these lines provides the control matrix. The result is input by the operator using cursesINDI. Suggestions for improving this are welcome.

## Requirements
<!-- List the requirements that must be met.  A numbered list works best -->

## Tests and Metrics
<!-- List any useful metrics that should be targeted by tests -->

## Scope and Caveats
<!-- Add any needed caveats and scope restrictions -->

## Answers to Agent Questions
1. The frozen part is only the moment_angle function itself. The radial-profile subtraction and the core mask at 0.7xseparation increased simulation performance by quite a bit with onsky images containing real noise, which is why they are included. However, they can be changed if the new approach demands it. Everyhing in the utils.py script should be disregarded. That script is *not* to be used. Everything should be within app.py. The simulations are available at https://github.com/kmtwitchell/adc_sims/blob/master/algo_26B/adc_ctrl_onsky_testing.ipynb, with the algorithm itself being stored in the same folder under adc_ctrl.py

2. The wrapping aroung +- 90° is a significant problem, and that was an imperfect attempt to solve it. the wrapping problem should be addressed in the new version, but it does not have to take that exact approach.

3. wavelength scaling should occur in order to select the spots. the normalized_wavelength method seemed to work pretty well. The grid should remain at its current dimensions/pixel size.

4. The filters that are specified are correct. The current state of the filter should be taken into account when calculating the normalized wavelength. However, either camsci1 or camsci2 can be used, and thus fwsci1 or fwsci2 might be the important filter to read. The camera and filter-wheel mappings should default to fwsci1 and camsci1, but there should be an INDI property to be able to switch to fwsci2 and camsci2. 

5. Both of those criteria for rejecting outliers are fine to implement for now.

6. keep "reject" behavior. For now, no absolute limit will be set.

7. for now, calibration will continue to be done by hand and entered into INDI. however, adding a calibration request toggle might be a good idea in the future once we are in the hardware testing stage.

8. Commands will not cause any stage movement when adcTracker.tracking is off, so there is no need for an interlock. There should be a debug level warning in the log that arises if a command is written to dadc1 or dadc2 when tracking is off. adcTracker only updates current after the stages move.

9. lab mode needs to stick around, because the sign of the gain (in order to avoid a positive feedback loop) can be opposite between the in-lab and on-sky operating states. This depends on which platform the instrument has been installed on, so sometimes this property will be needed and sometimes it won't. perhaps a better name for it than labmode is necessary.

10. no specific style type is necessary, just make it readable

11. utils.py should be ignored. it will be deleted. nothing should be written in its place, it should all appear in app.py

12. dark subtraction is useful, so it should be used when a dark frame is present. if a dark frame is not present, the app should continue to function and just use the median subtract.

# Instructions to Agent
<!-- Specific instructions for the agent.  Below is our standard, but you can modify it as needed. -->

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan
<!-- This section will be filled out by the agent -->

_Plan by Claude Opus 5.5 on branch `ktwitchell/adcCtrl-v2` (base `dev`). Rev 1: 2026-10-07, initial draft. Rev 2: 2026-10-07, updated for the user's "Answers to Agent Questions". Rev 3: 2026-10-07, N1/N2 accepted. Rev 4: 2026-10-07, functional implementation done (see Implementation Log). Status: **functional, documentation and formatting commits done. Build-machine verification pending.**_

## Task Summary
<!-- The agent should summarize the task as it understands it -->

`adcCtrl` (`apps/adcCtrl/xapp/adcCtrl/app.py`) is a purepyindi2 `XDevice`. It measures residual atmospheric dispersion from the pointing angles of satellite spots in a science-camera image, then closes a slow (~0.1 Hz) integrator loop by writing counter-rotation offsets to `adctrack.deltaADC1/2`. `adcTracker` applies those offsets on top of its model-based ADC tracking.

The physics (arXiv:2608.10307 §1–3): a 2D sinusoidal pupil grating makes four satellite spots. With residual dispersion, each broadband spot rotates away from pointing radially at the PSF core. The **pair offset angle** (a spot's angle minus the angle of the spot opposite it) is linear in dispersion and needs no core centroid. Active spots ("sparkles", from `tweeterSpeck`) sit at ~15 λ/D, and passive DM print-through spots sit at ~47 λ/D @ 28°. The paper averages 10–20 frames per measurement and ~5 measurements per command, rejects outliers, and uses a scalar gain (0.5).

The upgrade must:
1. Use the **method-of-moments** estimator `moment_angle()` (frozen; identical to `algo_26B/adc_ctrl.py` in `kmtwitchell/adc_sims`) to measure spot angles.
2. Properly fix the **±90° wrap** of spot angles.
3. Make the app **robust**. It currently cannot even import.
4. Keep the four states: idle, closed-loop, one-shot and measure-only.
5. Implement everything in `app.py`. `utils.py` is ignored and will be deleted by the user.

### Findings: current state of the code

The current `app.py` **cannot be imported**, so the crashes are total, not intermittent.

**Syntax and import errors (fatal at import time)**
- `calculate command` (lines 618, 689, 762) is missing its `#`, which is a `SyntaxError`.
- Inconsistent indentation in the `loop()` branches is an `IndentationError`.
- `ndimage`, `radial_profile`, `make_pupil_grid`, `Field`, `make_circular_aperture`, `make_rectangular_aperture` and `make_rotated_aperture` are used without the `hp.` prefix or an import.

**Runtime errors (would fire once the above is fixed)**
- `AdcFitter` methods lack `self` and are called as free functions.
- `set_control_mtx`/`calculate_command`/`speckle_pairs` exist only in `utils.py`, and `pairs` is never computed.
- `speckle_cutout(cropped, …, search_extent)`: `cropped` is undefined, and the call passes 6 args to a 5-parameter function. The `app.py` copy also still uses the old `extent=1E-6` search box, which selects essentially no pixels, so `argmax` lands on pixel 0. The notebook's version with `search_extent` is the working one.
- The inner `for i in range(4)` shadows the outer measurement index.
- `setup()` reads `tweeterSpeck.*` without `get_properties('tweeterSpeck')`, and reads `fwsci1.*`/`adctrack.*` without waiting for them. Each is a `KeyError` at startup if the property hasn't arrived.
- `handle_ctrl_mtx`: the `m01` `float()` cast is on the wrong side.
- `handle_state`/`handle_spots`: `current_state` is unbound if no element is ON.
- `XCam.grab_stack(subtract_dark=True)` (the default) raises `RuntimeError` when no `<shmim>_dark` exists.

**Hang and robustness risks**
- `send_command()` busy-waits with no timeout for `deltaADCn.current`, which `adcTracker` only updates **after the stages move**. If tracking is off, `adctrack` is down, or a stage is stuck, it never returns. It is also called from `setup()` and from INDI callbacks (`handle_offset`, `handle_reset`), which blocks all INDI handling.
- No exception handling around a measurement, so one bad frame escapes `loop()`.
- A state change to idle in the middle of a batch is ignored until the batch finishes.
- `nanmean` of all-NaN gives NaN, which leads to a misleading "exceeds threshold" log.
- The `app.py` `window_field`/`crop_image` slice past image edges and divide by zero on blank frames. The notebook (`adc_ctrl_onsky_testing.ipynb`, cell 3) already has hardened versions: a padded, clamped `window_field` and a `crop_image` with zero guards and a vectorized floor.

**Structure**
- The pipeline is copy-pasted three times (once per state).
- Magic numbers are inline: 6/21, 47/28, window sizes, 0.7 mask factor, pad 50, the 0.7° step limit, and the filter table (duplicated).
- `_normalized_wavelength` is computed but unused. `_crop_extent`, `_mask_diam` and `_command` are unused, and `_lab` is set but has no effect.
- `pyproject.toml` lacks numpy, scipy and hcipy.

**Wrap problem (Q2).** `moment_angle` returns the axis orientation in (−90°, 90°]. Spots 0 and 2 (top and bottom) are elongated near ±90° (the notebook plots `90 − angle`), so a small physical rotation flips the result between about +89° and −89°. The current `np.abs()` hides the flip but throws away the sign.

## Key Assumptions
<!-- The agent should list any assumptions it has made -->

1. Only `moment_angle()` is frozen. Its body is copied verbatim from `algo_26B/adc_ctrl.py`, which matches the current `app.py` version. The preprocessing (radial-profile subtraction, 0.7×separation core mask, rotated-box spot search) is kept because it helps on-sky. I adopt the notebook's hardened `window_field`/`crop_image` and its `speckle_cutout(…, search_extent)` signature.
2. All code (helpers, controller and device) lives in `apps/adcCtrl/xapp/adcCtrl/app.py`. Helpers become module-level functions or a small analysis class in that file, and no new modules are added. Unit tests live separately in `apps/adcCtrl/test/test_adcCtrl.py` and import from `app.py`. *(Tests are not app code. Say so if you want them elsewhere.)*
3. `utils.py` is not read, imported or modified. The user will delete it.
4. Control stays **counter-rotation only** (`δ2 = 0`, scalar output from a 1×2 `ctrl_mtx`). Calibration stays manual through INDI (Q7).
5. The image grid stays at its current dimensions and pixel scale (6/21 λ/D per pixel at the 656 nm reference). Spot search positions scale by `normalized_wavelength = λ_filter / 656 nm` (Q3).
6. Existing INDI property names stay unchanged, except `labmode`, which is renamed to `loop_sign` (N1, accepted). New properties are additive. Switching camera is only allowed while idle.
7. The Mac clone can't run the runtime. Tests run on the remote MagAO-X build machine. Branch `ktwitchell/adcCtrl-v2`. Commits per AGENTS.md #19: functional, then docs, then formatting.
8. Docstrings: plain, readable docstrings on every function and class, and comments on non-trivial attributes (Q10). No formatter is imposed.

## Requirements
<!-- The agent should list the requirements to which it is planning -->

**Measurement**
- R1. Each image is processed through one pipeline in `app.py`:
  1. Dark subtract (if a dark exists) and pad 50.
  2. Build an hcipy Field on the λ/D grid and median subtract.
  3. `crop_image` (no mask), then radial-profile subtraction (bin 5).
  4. `crop_image` with the core mask at 0.7 × the scaled separation, then median subtract and clip to 0.
  5. `speckle_cutout` for each of the four spots, then `moment_angle` on each.
- R2. **Wrap fix.** Convert each raw angle θₖ to a signed deviation from that spot's expected elongation axis: `dₖ = wrap90(θₖ − θ̂ₖ)`, where `wrap90(x) = ((x + 90) mod 180) − 90` and θ̂ₖ is the nominal axis implied by the grating angle and the spot's position. `moment_angle` itself is untouched. Pair offsets are `[d0 − d2, d1 − d3]`. Because each deviation is small and centered on 0, it is continuous across the old ±90° seam and keeps its sign.
- R3. Spot geometry:
  - Sparkles: separation and angle are read live from `tweeterSpeck.separation.current` and `tweeterSpeck.angle.current`.
  - DM spots: from config (default 47 λ/D @ 28°).
  - In both cases, the search position is separation × `normalized_wavelength`. Window size and search extent come from config, with current defaults (sparkles 20/20, DM 50/30).
- R4. **Filter and wavelength:** `normalized_wavelength = λ_filter / 656 nm`, using the active wheel's selected filter. The table is i = 762, z = 908, r = 615 nm, with 656 nm as the default. The filter is re-read on every transition out of idle and at the start of each cycle.
- R5. **Camera selection:** a new INDI switch `camera` with elements `camsci1` (default) and `camsci2`. It maps to filter wheel `fwsci1` or `fwsci2` respectively. The mapping comes from config. Switching re-creates `XCam` and is refused, with a warning, unless the state is idle.
- R6. **Dark frame:** subtract when `<shmim>_dark` exists (`XCam` detects it). Otherwise continue with median subtraction only and warn once. Darks are re-checked when the camera reconnects.

**Control**
- R7. A command uses `n_avg` frames per image and `no_measurements` images. Per-measurement NaN or failed results are dropped. Outliers beyond median ± k·MAD (k = 3, configurable) are rejected. The batch is discarded if fewer than half (configurable) of the measurements are valid.
- R8. `error = M · pairs` (with the same sign chain as today). The step is `gain_sign × gain × error`. If |step| ≥ the step limit (0.7°, configurable), the step is **rejected** (not clipped) and logged. There is no absolute limit.
- R9. Gain sign: the `labmode` switch is renamed to `loop_sign` (elements `positive` (default) and `negative`; N1). It multiplies the step by ±1, covering platform-dependent feedback polarity.
- R10. State semantics:
  - **idle**: no camera reads, no commands.
  - **closed-loop**: measure and command each cycle.
  - **one-shot**: one measure-and-command cycle, then automatically back to idle.
  - **measure-only**: measure and publish the would-be command, never write to `adctrack`.
- R11. Preserve `offset`, `reset_deltaADCs`, `gain`, `n_avg`, `no_measurements`, `ctrl_mtx` and `satellite_spots`, with their handler bugs fixed.

**Robustness**
- R12. The module imports cleanly, and `setup()` never raises because an external device is missing. A safe lookup helper returns defaults and warns once. A camera that fails to open sets `fsm` to an error state, and `loop()` retries.
- R13. INDI callbacks never block. They set parameters or flags, and `loop()` performs the `adctrack` writes (offset, reset, startup zeroing).
- R14. `send_command()` writes the targets and then waits for `current` (stages moving) up to a configurable timeout (default 30 s). It returns success or failure and never hangs. If `adctrack.tracking` is off when writing, it logs a **debug-level** warning (no interlock; Q8) and does not wait for `current`, since the stages will not move.
- R15. Each measurement is wrapped in try/except: errors are logged and the measurement is skipped. A whole failed cycle sends no command. After N consecutive failed cycles (configurable, default 5) in closed-loop, the app drops to idle with an error.
- R16. A state change is checked between measurements, and switching to idle aborts the batch without commanding.

**Observability**
- R17. New read-only INDI properties:
  - `measurement`: four raw angles, four deviations, two pair offsets, dispersion error, and n valid/total.
  - `command`: last step, total δ1, and sent/rejected/skipped.
  - `status`: last error text.

  Also log a `self.telem()` record per cycle.

**Code quality**
- R18. A single shared `measure_dispersion()` path. Magic numbers move to `AdcCtrlConfig` with today's values as defaults. Readable docstrings throughout `app.py`. `pyproject.toml` lists numpy, scipy and hcipy.

## Questions and Points of Clarification
<!-- The agent should list any open issues requiring user clarification -->

**Resolved (from "Answers to Agent Questions")**
- Q1: Only `moment_angle` is frozen. The preprocessing is kept but may change. Everything goes in `app.py`, and `utils.py` is not used. The reference is `adc_sims/algo_26B`. → Assumptions 1–3, R1.
- Q2: The wrap must be fixed properly. → R2.
- Q3: Scale the spot search by `normalized_wavelength` and keep the grid. → R3, Assumption 5.
- Q4: The filter table is correct. Add an INDI switch for camsci1/fwsci1 (default) vs camsci2/fwsci2. → R4, R5.
- Q5: Use both the k·MAD and min-valid-fraction criteria. → R7.
- Q6: Keep "reject" behavior, no absolute limit. → R8.
- Q7: Calibration stays manual for now. A calibrate request is a future item. → Follow-up.
- Q8: No interlock. Log a debug warning when writing while tracking is off. `current` updates only after the stages move. → R14.
- Q9: Keep lab mode as a gain-sign control, possibly renamed. → R9, N1.
- Q10: Readable docstrings, no specific style. → Assumption 8.
- Q11: Ignore `utils.py`. Everything goes in `app.py`. → Assumption 2.
- Q12: Dark subtract when present, otherwise median only. → R6.

**Resolved follow-ups (Rev 3)**
- N1. ✅ Accepted: rename to `loop_sign` (`positive`/`negative`).

  *Original question:* **Rename `labmode`.** I propose `loop_sign`, a one-of-many switch with elements `positive` (default) and `negative`, labelled "Feedback sign (platform dependent)". This makes it clear the switch only flips the command polarity. The cost: any operator scripts or GUI configs that set `adcCtrl.labmode.toggle` must change. I found no references in `gui/` or `apps/`. OK, or would you prefer a different name, or to keep `labmode` for compatibility?
- N2. ✅ Accepted: the default `ctrl_mtx` is provisional (a new matrix will be calibrated anyway), and `ctrl_mtx` defaults are settable from the config file.

  *Original question:* **Recalibration after the wrap fix.** Pair offsets will now be signed deviations, not differences of `|θ|`, so their sign and scale can differ from the convention used to derive the current default `ctrl_mtx` (m00 = 0.2118, m01 = 0.1928). I'll keep those defaults (moved to config). Please treat them as provisional, and re-derive `ctrl_mtx` by hand with the new app in measure-only mode before closing the loop. Also: should `ctrl_mtx` defaults be settable from the config file, so a recalibrated matrix survives restarts? I plan yes, a low-effort part of R18.

## Tests
<!-- The agent should list and describe the test it plans to implement.  It should be specific about the purpose and goal of the test. -->

pytest in `apps/adcCtrl/test/test_adcCtrl.py`, following the `apps/aoSim/test/test_aoSim.py` pattern: the device is built with `object.__new__`, with a fake INDI client (a dict-like object recording writes) and a fake camera returning synthetic frames. Synthetic images use numpy/hcipy, and the suite runs on the build machine.

**Algorithm**
- T1 `moment_angle` accuracy: noiseless elongated Gaussians at known angles (−85°…+85°). The recovered angle is within 0.1°.
- T2 `moment_angle` SNR tiers: noise at SNR ≈ 8, 30 and 100, with 50 seeds each. Report bias and σ, and assert bias < 0.5°.
- T3 `moment_angle` golden regression: fixed-seed crops with outputs pre-recorded from `adc_sims/algo_26B/adc_ctrl.py` must match the `app.py` version to 1e-10. *Goal:* prove the frozen estimator is unaltered.
- T4 wrap fix: for spots oriented at nominal ±90° (spots 0/2), rotate by ±0.5°, ±2° and ±5°. The deviation `dₖ` is continuous and signed (no 180° jump), and the pair offset has the correct sign. *Goal:* directly test Q2.
- T5 `window_field`/`crop_image` edge cases: a crop centered near or over an edge returns the requested shape (padded), and all-zero frames don't raise.
- T6 spot localization: a synthetic PSF with four spots for sparkles (15 λ/D at several angles) and DM spots (47 λ/D @ 28°), at each filter's `normalized_wavelength`. Each cutout peak is within 1 px of the truth. Also check that without wavelength scaling the z-band case is detectably wrong (proves scaling is applied).
- T7 end-to-end dispersion: an hcipy polychromatic PSF with grating spots, shifted per wavelength by a known dispersion, at 5 levels. Pair offsets are linear (R² > 0.99), and the sign of `error` tracks the sign of the dispersion.

**Control**
- T8 command math: `ctrl_mtx`, gain and `loop_sign` give the expected step, and |step| ≥ the limit is rejected, not clipped.
- T9 outliers: injected NaNs and an outlier give the inlier mean. Fewer than half valid means no command.

**Device behavior**
- T10 state machine: all four transitions update `_state`, the `fsm` text and the switch elements. A message with no ON element doesn't raise.
- T11 one-shot: exactly one `adctrack` write, then back to idle.
- T12 measure-only: N cycles give zero `adctrack` writes, and `measurement` is updated.
- T13 idle: `loop()` touches neither the camera nor the client.
- T14 `send_command` timeout: if `current` never changes, it returns False within timeout + ε.
- T15 tracking off: a write while `adctrack.tracking` is off emits a debug warning, returns without waiting, and does not raise.
- T16 camera failure: the grab raises or times out, the cycle is skipped, and after N failures the state goes to idle.
- T17 missing externals: `setup()`/`check_indi_props()` with no `tweeterSpeck`/`fwsci*`/`adctrack` entries do not raise, and use defaults.
- T18 dark handling: with a dark the frame is dark-subtracted. Without one, no exception occurs and a single warning is logged.
- T19 camera switch: idle → camsci2 re-creates the camera, and the filter is then read from `fwsci2`. A switch attempted while closed-loop is refused.
- T20 `ctrl_mtx` handler: `m00`/`m01` update the correct elements as floats.
- T21 abort mid-batch: switching to idle during a 5-measurement batch sends no command.
- T22 non-blocking callbacks: `handle_offset`/`handle_reset` return immediately, and the next `loop()` applies them.
- T23 import smoke test: `import xapp.adcCtrl` succeeds.

**Metrics:**
- `moment_angle` bias and σ vs SNR.
- Pipeline linearity (R², slope).
- Wall time per measurement, targeting < 1 s for a 512×512 frame so that 5 × 10 frames fit in about a 10 s cycle.
- Pass and fail counts from the build machine.

**Manual hardware steps (operator):**
1. Measure-only on the bench, then re-derive `ctrl_mtx` (N2).
2. One-shot.
3. Closed-loop at gain 0.5, checking `loop_sign`.

## Implementation Plan
<!-- Here the agent documents its plan for the implementation -->

All code changes are in `apps/adcCtrl/xapp/adcCtrl/app.py` unless noted. Commit order per AGENTS.md #19.

**Step 0: Golden capture.** Run `moment_angle` from `adc_sims/algo_26B/adc_ctrl.py` on deterministic synthetic crops and embed the outputs in the test file as `GOLDEN_MOMENT_ANGLES` (input for T3). *(Done. The values are inline rather than in an `.npz`, so no binary file is needed.)*

**Step 1: Analysis helpers (top of `app.py`).** Replace the broken `AdcFitter` with module-level functions:
- `window_field` and `crop_image`: the notebook's hardened versions.
- `subtract_radial_profile(img, bin_size)`.
- `speckle_cutout(img, n, angle, separation, window_size, search_extent)`: the notebook version.
- `moment_angle`: verbatim.
- `expected_axis(n, grating_angle)` and `wrap90(x)`: the R2 wrap fix.
- `measure_spot_angles(frame, geometry, cfg) -> (raw[4], dev[4])` and `pair_offsets(dev) -> [2]`.
- Control helpers: `reject_outliers(values, k, min_frac)` and `compute_step(pairs, ctrl_mtx, gain, sign, limit) -> (step, accepted, reason)`.

All with explicit `hp.`/`ndimage` imports.

**Step 2: Configuration.**
- `CameraConfig`: `shmim` and `filter_wheel` (keep `dark_shmim` as optional/unused, or drop it, since XCam uses `<shmim>_dark`).
- `AdcCtrlConfig`:
  - Cameras: `cameras: dict[str, CameraConfig]`, defaulting to camsci1→fwsci1 and camsci2→fwsci2, and `default_camera = "camsci1"`.
  - Geometry and wavelengths: `pixel_scale_lod = 6/21`, `reference_wavelength = 656e-9`, `filter_wavelengths` (i/z/r), `dm_spot_separation = 47`, `dm_spot_angle = 28`, window and search sizes, `mask_factor = 0.7`, `pad = 50`.
  - Control: `step_limit_deg = 0.7`, `send_timeout_sec = 30`, `send_tolerance_deg = 0.05`, `max_consecutive_failures = 5`, `outlier_k = 3`, `min_valid_fraction = 0.5`, `ctrl_mtx = [0.21178766, 0.19275196]`.

  All defaults equal today's values.

**Step 3: Device rewrite (`class adcCtrl(XDevice)`).**
- `setup()`:
  - Create the existing properties (`labmode` renamed to `loop_sign`), plus the new `camera`, `measurement`, `command` and `status` properties.
  - Subscribe to `adctrack`, `tweeterSpeck` and both filter wheels.
  - Open the camera inside try/except.
  - Queue the zero-offset reset as a pending flag instead of the blocking send.
  - Set `fsm` to READY.
- `_ext(key, default)`: safe client read that warns once per missing key.
- `check_indi_props()`: active filter → `normalized_wavelength`, plus sparkle geometry.
- Handlers: fix `current_state`/`ctrl_mtx`. `handle_offset`/`handle_reset` only set `_pending_send`/`_pending_reset`. `handle_camera` is refused unless idle. `handle_loop_sign`.
- `loop()`: apply pending writes, then dispatch on state to `run_cycle(send=…)`, then one-shot → idle. A top-level try/except updates `status` and the failure counter.
- `measure_dispersion()`: per-measurement loop with an abort check (R16), grab (dark-aware), `measure_spot_angles`, `pair_offsets`, and `M·pairs`. Then outlier rejection and publishing of `measurement` plus telemetry.
- `send_command()`: write targets. If tracking is off, log a debug warning and return. Otherwise poll `current` with a `time.monotonic()` deadline and return a bool.
- `transition_to_idle()` also sets `fsm`. Remove unused members.

**Step 4: Packaging.** In `pyproject.toml`, add numpy, scipy and hcipy and bump `version`. `utils.py` is left untouched (the user deletes it).

**Step 5: Tests.** Add `apps/adcCtrl/test/test_adcCtrl.py` (T1–T23) and the golden data. Run `python -m pytest apps/adcCtrl/test -v` on the build machine and record results and metrics under a **Verification** heading in this file.

**Step 6: Documentation-only commit.** Module docstring (purpose, author, operator procedure, and an INDI property table), plus docstrings on every function, class and config field.

**Step 7: Formatting-only commit** (only if needed).

**Planned affected files:**
- `apps/adcCtrl/xapp/adcCtrl/app.py`
- `apps/adcCtrl/pyproject.toml`
- `apps/adcCtrl/test/test_adcCtrl.py` (new)
- this plan file

### Implementation Log (Rev 4)

**Done (functional commit):** `app.py` rewritten per Steps 1–3, `pyproject.toml` updated (Step 4), and `test/test_adcCtrl.py` added (Step 5).

**Done (documentation commit, Step 6):** the module docstring now covers the method, states, INDI property table and operator procedure. Every function and class has a docstring, members are commented in `init_state()`, and the test module docstring explains how to run the suite. A check confirmed this commit changes no code: the AST is identical with docstrings removed.

**Done (formatting commit, Step 7):** string quotes converted to double quotes per `.editorconfig` (`quote_type = double` for `*.py`). The repo has no Python formatter config (black/ruff), so no formatter was run.

**Decisions made during implementation:**
- **Core mask diameter** is now `mask_factor × separation` in λ/D for both spot sources. The old code computed the sparkle mask as `separation / (6/21) × 0.7` (a pixel conversion applied to a λ/D grid). For sparkles at 15 λ/D that gives a 36.75 λ/D diameter mask, whose radius (18.4 λ/D) is larger than the spot separation, so it would have masked the sparkles themselves. The DM-spot mask was `47 × (6/21) × 0.7 ≈ 9.4` λ/D. Both now follow the user's stated "0.7 × separation" rule.
- **Radial-profile bin size** is `radial_bin = 5 × 6/21 ≈ 1.43` λ/D, which equals the 5-pixel bin validated in the on-sky notebook. The old app passed `5` on a λ/D grid, which is 17.5 pixels.
- **DM spot separation** (47) is interpreted as λ/D at the observing wavelength and scaled by `normalized_wavelength`, as for sparkles (this was the old `utils.py` `find_speckle` behavior). *Confirmed by the user, 2026-10-07.*
- **Expected spot axes** for the wrap fix: spots 0 and 2 at `90° − angle`, spots 1 and 3 at `−angle`. This matches the notebook's on-sky result of 54–74° for the DM spots at 28°. The synthetic tests show the old `abs()` failure directly: with sparkles at angle 0, spots 0 and 2 read +89.5° and −89.5°, so |θ0| − |θ2| = 0, while the new signed deviations correctly give −0.5° and +0.5°.
- **Gain default** is 0.1 (config `gain`), per the user (2026-10-07). The old INDI `gain` property displayed 0.10 while the code actually used `_gain = 0.5`. Now the property and the gain in use agree at 0.1. This was briefly 0.5 in commit c188bbe9 and corrected in a follow-up commit.
- **Startup zeroing** of `deltaADC1/2` is queued and applied by `loop()` once `adctrack` is visible. It now always writes 0, where before it only wrote if `deltaADC1.current != 0`. Writing 0 when it is already 0 is harmless.
- **No δ1 resync from `adctrack`.** `current` only updates after the stages move, so resyncing from it would discard commands while tracking is off. The app keeps its own integrator, as before.
- **Repeated failures:** after `max_consecutive_failures` failed closed-loop cycles the app goes idle, and `fsm` returns to READY. The reason is shown in `status.last_error` and the log.
- **Config schema change:** `camera.shmim` / `camera.dark_shmim` are replaced by a `cameras` table (`shmim`, `filter_wheel` per entry; defaults camsci1/fwsci1 and camsci2/fwsci2). Darks come from XCam's automatic `<shmim>_dark` detection. **Any deployed `/opt/MagAOX/config/adcCtrl.conf` with a `[camera]` section must be updated.**
- The INDI property `labmode` is now `loop_sign` (`positive`/`negative`). New read-only properties: `measurement`, `command`, `status`. New switch: `camera`.

**Test coverage vs. plan:** T1–T23 are implemented as 28 test functions in `apps/adcCtrl/test/test_adcCtrl.py`, including `loop_sign` and default-gain tests. T7 uses a numpy broadband model (core and spots scaled with λ, core shifted linearly across the band) rather than a full hcipy optical propagation.

### Verification

- **Local (this Mac):** all 28 tests pass using real numpy 2.4.3, scipy 1.17.1 and hcipy 0.7.0, with *stub* `xconf`, `purepyindi2` and `magaox` modules (pip could not reach PyPI from the sandbox). This validates the algorithm and control logic and the device logic against a minimal property model. It does **not** validate the real purepyindi2 property, message or `Device` APIs.
- **Golden test:** `moment_angle` in `app.py` reproduces `adc_sims/algo_26B/adc_ctrl.py` to 1e-10 on 8 inputs.
- **Pipeline linearity (sparkles, synthetic):** about 25° of `pair02` per λ/D of band-integrated dispersion, with R² > 0.99.
- **Timing:** about 0.09 s per 512×512 frame for both spot sources (local Mac), well within the ~10 s loop budget.
- **Pending on the build machine:** `cd apps/adcCtrl && python -m pytest test -v` against the real purepyindi2, xconf and magaox. Then `make install`, start the app with the updated config, and run measure-only on the bench.

## Follow-up and Edge Cases
<!-- The agent should list any planned follow up and any edge cases that are not addressed -->

- **In-app calibration (deferred per Q7):** a `calibrate` request switch that steps δ1 over configured offsets, fits the pair-offset slopes, and proposes `ctrl_mtx` for the operator to accept. Revisit at hardware testing.
- **`ctrl_mtx` recalibration:** the user will calibrate a new matrix by hand (measure-only mode) after deployment, because the wrap fix changes the pair-offset sign convention (N2).
- **Co-rotation / 2-DOF control** (using δ2) is not addressed.
- **Saturated spot cores** bias the second moments. There is no saturation detection.
- **Spots off-frame** (small ROI, or DM spots in z-band at 47 × 1.38 λ/D): handled as failed measurements via the padded `window_field`, with no ROI adaptation.
- **Sparkles selected but `tweeterSpeck` not modulating:** this gives noise-dominated measurements. A `tweeterSpeck.modulating` check would be a cheap addition.
- **Abort mid-batch (R16)** only takes effect if purepyindi2 delivers INDI callbacks on a different thread from `loop()`. If both share a thread, a state change is seen at the next cycle instead. To verify on the build machine.
- **Deployed config file** must be migrated from `[camera]` to `[cameras.*]` (see Implementation Log).
- **Concurrent writers to `deltaADC1/2`:** the app does not resync δ1 from `adctrack` (see Implementation Log), so external edits to `deltaADC1/2` are overwritten on the next command. A full ownership convention is out of scope.
- **Loop period:** with large `n_avg × no_measurements`, plus up to `send_timeout_sec` waiting for stages, a cycle can exceed `sleep_interval_sec`. That interval is a minimum, not a period.
- **purepyindi2 behavior when `loop()` raises:** unverified, because no local purepyindi2 source was found (the search timed out). The plan catches everything inside `loop()` regardless.
- **Paper reading:** only web-extracted text of §1–3 was available, because the PDF could not be parsed locally.
- **Plan template fix (Rev 1):** the guidance comment under this heading was missing its closing `-->`. I closed it in place. This was the only edit outside agent-filled content.

## DRAFT: Proposed Calibration Sequence (not adopted, pending user verification)

> **Status: DRAFT. Not part of the current plan or implementation.** This section describes how the agent *would* write a calibration sequence. It does not change Q7 above (calibration stays manual for now).
>
> **Sequencing (user direction, 2026-10-12):**
> 1. Answer the remaining follow-up questions, then save this draft.
> 2. The user installs the current app (without calibration) on the MagAO-X machine.
> 3. Verify the current app there: run the tests and check it end to end.
> 4. Only then, start the calibration procedure as an update to the working app.
>
> The calibration adds a lot of code and operator-facing output. It should not be added while the rest of the app is untested.
>
> **Decisions so far (user, 2026-10-12):**
> - Use a triangle sweep of the counter-rotation δ1 only; δ2 is never poked.
> - Push-pull is kept below for the record but is not expected to be implemented.
> - Use the internal source: its response differs insignificantly from on-sky.
> - Backlash is ignored entirely.
> - Control on the noise-weighted combination of both pairs. Report the co-rotation component, but do not correct it.
> - Single-pair fallback is allowed everywhere except during the calibration sweep.
> - Store each calibration as one FITS file in `/home/xsup/adc_calibs/`.
>
> **Decisions from round 3 (user, 2026-10-12):**
> - Each sweep position uses the usual `no_measurements` batch. Leave at least 3 s between the stages stopping and the next move, for stage safety.
> - The FITS file holds per-position mean images plus a per-batch table, with the naming as proposed.
> - There is one active calibration file at a time. A mismatch with the current setup is logged at INFO only.
> - The `ctrl_mtx` INDI property stays writable. With no calibration file, the app falls back to the config `ctrl_mtx`.
> - Use a simple straight-line fit that warns and never rejects.
> - The zero-point understanding (R6) is confirmed.
> - Implement as a `calibrate` element of the `state` switch, in `app.py`.

### What is being calibrated

The app controls one degree of freedom, the counter-rotation δ1 (`deltaADC1 = δ1 + offset`, `deltaADC2 = δ1 + offset`, with δ2 = 0). `deltaADC1/2` are always zero unless adcCtrl sets them; all other ADC motion comes from adctrack's tracking look-up table. So δ1 values, including the sweep range, are **absolute**.

Each cycle measures two pair offsets `p = [pair02, pair13]` (degrees) and reduces them to a scalar error `e = ctrl_mtx · p`. Near the δ1 where the pair offsets vanish, the response is assumed linear:

```
p(δ1) ≈ c + s · δ1          s = [s02, s13]  (deg of pair offset per deg of δ1)
```

Calibration measures the sensitivity vector `s`, and from it builds `ctrl_mtx` in units of δ1 degrees (accepted by the user).

**Control target.** The loop drives the pair offsets to zero, on sky exactly as on the internal source. AO limitations mean zero is never reached exactly, but the target is the same. What does *not* carry over between the internal source and sky is the δ1 at which the offsets cross zero, because the atmospheric dispersion is different. That is why the null δ1 is a diagnostic only and is never used as a setpoint. *(Confirmed by the user, round 3 R6.)*

### Sequence

1. **Preconditions.** Refuse to start (status message, back to idle) if any of these fail:
   - `adctrack.tracking` is on. adcTracker only moves the stages while tracking.
   - Sparkles are modulating (if the spot source is sparkles).
   - The spots are unsaturated.
   - The sweep range lies inside the safe limit `calib_max_deg` = ±5° (absolute δ1).

   Record the current δ1 so it can be restored afterwards.
2. **Sweep.** Use a triangle in absolute δ1:

   ```
   up:   −A, −A + h, …, +A          (A = calib_range_deg = 3, calib_n_steps = 21, so h = 0.3°)
   down: the same positions in reverse (the +A point is not repeated)
   ```

   - The first move goes straight to −A: no overshoot, and backlash is ignored.
   - Each move goes through `send_command()`, which waits for the stages to report arrival. Settling is taken to be zero once they have arrived.
   - At each position, take one `measure_dispersion` batch (`n_avg × no_measurements`, as in normal operation).
   - **Stage safety:** the next move is not sent until at least `calib_dwell_sec` = 3 s after the stages reported arrival. If the batch takes longer than that, the move waits for the batch.
   - Frames with invalid spots are recorded as failed points, not as a failed calibration. Single-pair fallback is **off** during the sweep, so every slope comes from one known pair.
3. **Fit (kept simple).** For each pair element, fit one straight line `p_j = c_j + s_j·δ1` over all positions of both legs. The triangle is symmetric in time about its midpoint, so a linear drift during the sweep does not bias the slope, and no drift term is needed.

   **Noise.** The per-pair noise `σ02`, `σ13` comes from the scatter of the measurements within each position's batch.

   **Reported:**
   - `s_j` with its uncertainty, and the intercept `c_j`;
   - the null `δ1*_j = −c_j / s_j` (diagnostic only);
   - R² and the residual RMS.

   **No automatic rejection.** If R² is below `calib_min_r2`, the app logs a warning. The file is still written, and the operator decides whether to use it.
4. **Build `ctrl_mtx`.** Use the noise-weighted pseudo-inverse of `s`, negated:

   ```
   W = diag(1/σ02², 1/σ13²)
   ctrl_mtx = −(sᵀ W s)⁻¹ sᵀ W          (1×2, so m00, m01)
   ```

   Then `e = ctrl_mtx · p` is the δ1 correction in degrees, and the noisier pair is downweighted. The single-pair fallback coefficients `−1/s_j` come from the same fit.
5. **Restore and save.** Return δ1 to its recorded value, whether the sweep succeeded or not; an abort, state change or failure also restores. The state then returns to idle. Then:
   - Create `/home/xsup/adc_calibs/` if it does not exist.
   - Write one FITS file, `adccal_<camera>_<filter>_<source>_<UTC YYYYmmddTHHMMSS>.fits`, containing:
     - **primary HDU:** a cube with one mean image per sweep position (41 × H × W);
     - **primary header:** the results and the parameters they were measured under (see "Spot source selection");
     - **binary-table extension:** one row per measurement (commanded and reported δ1, time, spot angles, deviations, pair offsets, valid flags).
   - Log the file path and a summary (slopes, R², proposed `ctrl_mtx`) at INFO level, and add a telemetry record.
   - The result is **not** applied automatically. To use it, the operator points the new config field `calibration_file` at it and restarts or reloads the app.
6. **Validation (optional, recommended).** After pointing the config at the new file, offset δ1 by a known amount `b` (within the step limit), switch to measure-only, and check that the measured error is about −b. Then close the loop and confirm that the pair offsets are driven back toward zero.

### Alternative: push-pull sequence (kept for the record, not expected to be implemented)

Steps 2–3 would be replaced by the following; the other steps are unchanged.

- **Pokes:** `0, +a, 0, −a, 0, +2a, 0, −2a, 0`. The returns to 0 track drift, and the two amplitudes test linearity.
- **Fit:** `p_j(t) = c_j + s_j·δ1(t) + r_j·t`. The zero-returns make the drift separable. Linearity check: the ±a vs ±2a slope ratio should be about 1.

| | Triangle sweep (primary) | Push-pull with zero-returns |
|---|---|---|
| **Main output** | Full response curve: slope, null, linearity | Slope near the operating point |
| **Drift** | Separable (symmetric in time) | Rejected by design |
| **Nonlinearity** | Visible across the full range | Only checked with two amplitudes |
| **Finds δ1\* (null)** | Directly, from the zero crossing | Only by extrapolating from the intercept |
| **Time near the operating point** | Much of the time at large offsets | Half the time at 0 |
| **Move sizes** | Small equal steps after the first move | Up to 2a per move |
| **Positions carrying information** | All | About 4 of 9 |
| **Risk at the ends** | Spots can leave the windows at ±A, so failed points must be handled | Bounded by 2a |

Push-pull's advantages (drift rejection, staying near the operating point) matter mostly on sky during science. Calibration is done on the internal source, so those advantages don't apply. (Backlash was a comparison point earlier; it is ignored per the user.)

### How the calibration drives the ADC response

With `ctrl_mtx` calibrated as above, `e` is the δ1 correction in degrees. The existing control law becomes a plain integrator with a dimensionless gain:

```
δ1[k+1] = δ1[k] + loop_sign · gain · e[k]  ≈  δ1[k] − gain · (δ1[k] − δ1*)      (loop_sign = positive)
```

Here δ1* is wherever the pair offsets are zero under the current conditions. The residual decays by `(1 − gain)` per cycle. Consequences:

- **Gain meaning.** With gain = 0.1 the loop removes 10% of the residual per cycle: about 22 cycles to 90%, about 7 to 50%. The gain becomes independent of filter, spot source and camera, because those differences are absorbed into `s`.
- **`loop_sign`.** The measured `s` already carries the platform's sign, so a calibration done on the platform it runs on needs `loop_sign = positive`. `loop_sign` then only matters for an uncalibrated or inherited matrix.
- **Step limit.** `step_limit_deg` (0.7°) rejects rather than clips. With gain 0.1, a residual above about 7° is never corrected. Because δ1 is absolute and bounded at ±5°, a calibrated `e` larger than the safe range is itself a sign of a bad measurement. A natural replacement is to reject when `|δ1 + step|` would leave ±5°, and to keep the per-step limit only as a sanity check.
- **Co-rotation diagnostic.** Counter-rotation always disperses vertically on the science camera; only co-rotation turns the dispersion direction, because the ADC is downstream of the pupil derotator. The pair-space component orthogonal to `s` therefore measures co-rotation error. It is **reported** (in the measurement property and telemetry) but **not corrected**.
- **Noise floor.** At steady state, the δ1 jitter is about `σ_e · sqrt(gain / (2 − gain))`, where `σ_e` is the error noise propagated from `σ` through `ctrl_mtx`.
- **Two degrees of freedom.** Out of scope: δ2 is not poked or controlled, per the user.

### Using both pair offsets vs. a single pair

**What each pair measures.** Dispersion smears each spot along the dispersion direction, which tilts the radial streak. Opposite spots tilt in opposite senses, so a pair offset responds only to the dispersion component **perpendicular to that pair's axis**. The 0/2 and 1/3 axes are orthogonal (`expected_axes`: `90° − θ` and `−θ`), so `[pair02, pair13]` measures the dispersion as a 2D vector. Counter-rotation errors are always vertical on the science camera, so the most sensitive pair is the one whose axis is closest to horizontal:
- with sparkles at angle 0, only pair13 responds;
- with DM spots at 28°, pair13 should have about twice pair02's slope.

**Strengths of using both pairs:**
1. **Sensitivity at any spot angle.** A single pair's slope scales with sin(angle between the dispersion and the pair axis). Two orthogonal pairs always include one with good sensitivity, whatever the sparkle `angle`.
2. **Co-rotation is separable.** The component orthogonal to `s` is the co-rotation diagnostic. A single pair cannot separate it from counter-rotation.
3. **Lower noise.** For equal sensitivity and uncorrelated noise, combining gives up to √2 (less in practice; see weakness 3).
4. Rigid rotation of the spot pattern cancels within each pair regardless.

**Weaknesses of using both pairs:**
1. **More ways to fail.** All four spots must be valid. As currently implemented, any one spot failing discards the whole frame (`measure_dispersion` in `app.py`). Spots can fail when clipped by a small ROI, when they sit on a static speckle or bad pixel, or when a sparkle is weak. DM spots are visible with the internal source at every wavelength (user).
2. **A poorly weighted combination can be worse than one pair.** Equal weights (like the near-equal default `[0.21, 0.19]`) add noise from the weaker pair. Step 4's weighting fixes this, but only with a valid calibration.
3. **The two pairs share error sources** (centering, radial-profile residuals, uneven halo), so those errors do not average down.
4. **More to calibrate and debug.**

**Decided approach (user, 2026-10-12):**
- Control on the noise-weighted combination of both pairs (step 4).
- **Single-pair fallback** when a spot is invalid, using `−1/s_j` for the surviving pair. It is allowed in every state **except the calibration sweep**. Each fallback is logged at **INFO** level so the operator is told. This needs per-spot validity flags in `measure_spot_angles` instead of failing the whole frame. The published measurement records which pairs were used.
- **Report** the co-rotation (orthogonal) component, but do not correct it.

### Spot source selection

**User direction (2026-10-12):**
- The operator chooses the spot source, mainly by spot SNR. Some observations do not allow sparkles.
- DM spots are visible with the internal source at every wavelength.
- Sparkles do not bias the DM-spot measurement.
- Calibrating both sources from one sweep is **not** wanted.

**How the calibration handles it:**
- **No automatic choice.** Calibration runs on whichever source `satellite_spots` selects at the start. Changing `satellite_spots` during a sweep aborts it, and δ1 is restored.
- **One source per sweep.** The other source needs its own sweep, after the operator switches sources.
- **Saved in the FITS header:**
  - camera, filter (and its `normalized_wavelength`), spot source;
  - for sparkles, the `tweeterSpeck` separation and angle;
  - for DM spots, the configured separation (47 λ/D) and angle (28°);
  - the sweep settings (range, steps, dwell, `n_avg`, `no_measurements`);
  - the fit results (`s_j` and their uncertainties, `c_j`, the nulls, R², residual RMS, drift, `σ_j`), `ctrl_mtx`, the fallback coefficients, and the UTC timestamp.
- **Using a calibration.** There is one active file at a time, set by the config field `calibration_file`. The operator re-points it after changing camera, filter, spot source or sparkle geometry.
  - **At startup** the app reads `ctrl_mtx`, the fallback coefficients and the slope direction (for the co-rotation diagnostic) from the header. The values read from the file initialize the INDI `ctrl_mtx` property.
  - **With no file configured,** the app uses the config `ctrl_mtx`, as it does now.
  - **The `ctrl_mtx` INDI property stays writable**, so the operator can change it on the fly.
  - **Mismatches:** before each cycle, the header's camera, filter, spot source and (for sparkles) `tweeterSpeck` separation and angle are compared with the current setup. On a mismatch, the app logs at **INFO** level and carries on: it neither refuses closed loop nor changes anything. A mismatch with the file does not necessarily mean the active `ctrl_mtx` is wrong, because the operator may have set `ctrl_mtx` by hand. The message is logged once per change of setup, not every cycle.
- **Helping the operator choose (suggestion, optional).** Measure-only could publish a spot SNR for the selected source.

### Implementation outline (deferred until the current app is verified on the MagAO-X machine)

All of this goes in `app.py` (R7a).

- **States.** Add `calibrate` to `STATE_ELEMENTS` and the `state` switch. The sweep runs inside `loop()`, one position per call, so a state change aborts it between positions, as with `measure_dispersion`. On completion or abort, δ1 is restored and the state returns to `idle`.
- **New config fields:**
  - `calibration_file` (path, default empty);
  - `calib_range_deg` = 3;
  - `calib_n_steps` = 21;
  - `calib_dwell_sec` = 3;
  - `calib_max_deg` = 5;
  - `calib_min_r2`;
  - `calib_dir` = `/home/xsup/adc_calibs`.
- **New pure functions** (unit-testable like the rest):
  - `sweep_positions(range, n_steps)`;
  - `fit_sweep(delta, pairs)`, which returns slopes, intercepts, uncertainties, nulls and R²;
  - `build_ctrl_mtx(s, sigma)`;
  - `write_calibration(path, ...)` and `read_calibration(path)`.
- **Measurement changes:**
  - per-spot validity flags in `measure_spot_angles`;
  - single-pair fallback in `measure_dispersion` (off in `calibrate`), logged at INFO;
  - the co-rotation component in the `measurement` property and telemetry.
- **Dependency:** `astropy` (FITS I/O) added to `pyproject.toml`.
- **Tests:**
  - sweep positions and the ±5° bound;
  - fit recovery of a known slope on synthetic pair data, including a linear drift on a triangle;
  - `ctrl_mtx` weighting;
  - a FITS round trip;
  - a mismatch logs at INFO and does not block;
  - δ1 is restored on abort;
  - at least 3 s between arrival and the next move;
  - fallback is disabled during `calibrate`.

### Open points for the user to verify

- Is δ1 (counter-rotation) the right axis to poke, or should calibration step the individual ADCs (`deltaADC1` and `deltaADC2` separately)?
- ~~Push-pull vs. monotonic sweep?~~ **Resolved (2026-10-12):** monotonic sweep is primary; push-pull is kept as a non-implemented alternative.
- Triangle (up and down) or one-way with baseline brackets? What range `A`, step count and dwell per step does the known sweep use?
- What sweep range is safe for each spot source and filter (spots inside the search windows, and stage limits)?
- ~~On-sky or bench?~~ **Resolved (2026-10-12):** the response differs insignificantly between on-sky and the internal source, so the internal source is acceptable. Still open: does the null (zero point) also carry over, or only the slope?
- Should `ctrl_mtx` map to δ1 degrees (as proposed) or keep the original paper's units?
- Is one pair ever preferred in practice (e.g. sparkles oriented to the dispersion direction), and should single-pair fallback on spot failure be allowed in closed loop or only in measure-only?
- Is the dispersion direction on the detector fixed relative to the spot pattern in normal operation (it depends on where the ADC and spots sit relative to the derotator)? This sets how much the orientation argument for two pairs matters.
- ~~How is the spot source chosen?~~ **Resolved (2026-10-12):** the operator chooses, mainly by spot SNR, with some observations not allowing sparkles. One source per sweep; the parameters are saved with the result. DM spots are visible with the internal source at every wavelength, and sparkles do not bias them.
- Where should saved calibration records live (config file, a calibration directory, or similar), and in what format? *(See round 2, Q7.)*

#### Responses to Open Points
- yes, the counter-rotation axis is the only one that should be poked.
- Triangle is the approach we will take. Range should default to (-3,3), default step count is 21 and step dwell is 3 seconds.
- sweep range is safe as long as it is within (-5,5) for any of the filters or sparkle positions.
- the null point does *not* carry over.
- the proposed units are acceptable. 
- one pair is preferred in practice, as will be explained in the next bullet. Single-pair fallback should be allowed in on-sky and measure-only, but a log entry with INFO priority should appear to notify the user that this has happened.
- When the ADCs are counter-rotated, the dispersion direction is always along the vertical axis on the science camera. Only by co-rotating the ADCs does the dispersion direction change, because the ADC is downstream of the pupil de-rotator.

#### Remaining Questions (Round 2)
*Questions raised from the responses above. Write answers under each one.*

**Sweep definition**

- **Q1. What the range is relative to.** Are the default sweep (−3, 3) and the safety limit (−5, 5) offsets from the starting δ1, or absolute values of δ1 (`deltaADC1/2`)? If they are relative, a start of +3 would put the top of the sweep at +6, outside the safe range. Should the app refuse to run, or clip the sweep to ±5?
  - Answer:
    `deltaADC1/2` are /always/ zero, unless this specific app tells them to be otherwise. All other ADC movement is handled by the `tracking` look-up table in adctrack. This means the sweep values are absolute.

- **Q2. What counts as the 3 s dwell.** Does the 3 s include the settling time after the stages report they have arrived? Should a step's measurement be all frames taken during the dwell, or the usual `n_avg × no_measurements` batch with 3 s as a minimum?
  - Answer:
  just the 3s as a minimum. Assume the settling time is zero once the stages report that they have arrived, so the 3s includes it.

- **Q3. Down leg and backlash.** Should the down leg of the triangle use the same 21 positions in reverse? Is a small overshoot below −3 acceptable before the up leg starts (it stays inside ±5), or do the stages have no backlash worth handling?
  - Answer:
    the backlash is really not a concern at all. In fact, it should be ignored entirely in order to simplify the code and operation of this calibration procedure.

**Choosing the pair**

- **Q4. Single best pair or weighted combination.** Counter-rotation always disperses vertically on the science camera, so the most sensitive pair is the one whose axis is closest to horizontal. With sparkles at angle 0, only pair13 responds (pair02 lies along the vertical). With DM spots at 28°, pair13 should have about twice pair02's slope. Should the app control on the single most sensitive pair, with the other pair used only as the fallback? Or should it keep the noise-weighted combination of both pairs, which also favors the stronger pair? If a single pair is used, should the preferred pair be fixed in config or chosen from the calibrated slopes?
  - Answer:
    The approach from earlier is stil acceptable. we should control both, with the noise-weighted combination of both. The co-rotation (orthogonal component) should be reported (but not corrected for), as discussed earlier.

- **Q5. Where fallback applies.** Does "on-sky" mean the closed-loop states (`adcLoop` and `oneshot`)? Should fallback be **forbidden during the calibration sweep itself** (recommended, so that each slope comes from a single, known pair)?
  - Answer:
    fallback is permitted anywhere except the calibration sweep.

**Zero point**

- **Q6. On-sky zero point.** The null does not carry over from the internal source. Where should the on-sky loop's zero point come from?
  - (a) Assume the pair offset is zero on sky, and drive it to zero (what the loop does now).
  - (b) Measure an on-sky reference `p_ref` with a separate procedure, for example a short sweep on sky to find the null.
  - (c) Another procedure already in use.

  Also, should the internal-source nulls be saved purely as diagnostics?
  - Answer:
    Let me be more clear about this. The zero-point on-sky is essentially the same as it is on the internal source - the pair offsets should be driven as close to zero as they can get. there will always be AO limitations that will cause it to be impossible to achieve this number in reality, but the theory is the same. If you do not understand this, please ask a clarifying question.

**Records and status**

- **Q7. Calibration records.** Where should saved calibration records live: the config file (`ctrl_mtx` entries per camera, filter and spot source), a separate calibration directory with one file per sweep, or both? Is there a preferred format (TOML like xconf, JSON or FITS)?
  - Answer:
    calibration records should be stored in /home/xsup/adc_calibs/ , which, if it does not yet exist, should be created. One .fits file should exist per calibration that captures the images of the sweep, with the important calibration results within the header of this file. To change the calibration, the configuration file will be changed to point at a different .fits file in this directory. 

- **Q8. Status of this draft.** Once these are answered, should this section be promoted from DRAFT into the plan (requirements, implementation steps and tests) for implementation next, or kept as a draft for now?
  - Answer:
  keep this as a draft.

#### Remaining Questions (Round 3)
*Raised by the round 2 answers. Write answers under each one. This section is still a draft; none of it will be implemented until the current app has been verified on the MagAO-X machine.*

- **R1. Filling the 3 s dwell.** The proposal: after the stages report arrival, run `measure_dispersion` batches back to back until at least 3 s have passed, always completing at least one batch, and use every batch from that position. The alternative is to run exactly one batch and then wait out the rest of the 3 s. Which do you want? Or should the dwell instead set how many frames are taken, ignoring `no_measurements`?
  - Answer:
    The number of measurements should always follow `no_measurements`. The dwell here is not very important, 3s was just an arbitrary number I picked. There should just be at least 3s between when the ADC stops moving and when it is next told to move, to ensure the safety of the stage.

- **R2. FITS contents and naming.** Proposal:
  - primary HDU: a cube with one mean image per sweep position (41 positions × H × W), not every frame (every frame could reach GB);
  - a binary-table extension with one row per batch (commanded and reported δ1, time, spot angles, deviations, pair offsets, valid flags);
  - the results and parameters in the primary header;
  - filename `adccal_<camera>_<filter>_<source>_<UTC YYYYmmddTHHMMSS>.fits`.

  Do you want every raw frame instead of (or as well as) per-position means? Is a different naming scheme preferred?
  - Answer:
    per-position means are preferred. the rest of this proposal is acceptable.

- **R3. Scope of the config pointer.** Is there **one** active calibration file at a time (the operator re-points it after changing camera, filter or spot source), or one pointer per camera, filter and spot source? If the active file's header does not match the current camera, filter, source or sparkle geometry, should the app refuse closed loop (measure-only still allowed), or only warn at INFO or WARNING level?
  - Answer:
    There is only one active calibration file at a time. the operator will re-point after changing anything. the app should send a warning at the INFO level, but note that the operator can manually change the ctrl_matrix, so a mismatch between the calibration pointer and the current state does not necessarily imply that the ctrl_matrix is mismatched with the current state.

- **R4. Relationship to the existing `ctrl_mtx`.** When a calibration file is configured, should it **replace** the `ctrl_mtx` config field and INDI property (the property becomes read-only, showing the file's values)? Or should the INDI property stay writable as a temporary override? With no calibration file configured, should the app fall back to the config `ctrl_mtx` as it does now?
  - Answer:
  the INDI property should stay writable for sure. There are many reasons why ctrl_mtx might need to be changed on the fly. It should fall back to the config ctrl_mtx like it does now if there is no calibration file configured. 

- **R5. How much fitting logic.** To keep the calibration simple: fit one straight line over the full ±3° sweep, report R² and the residuals, and only **warn** (no automatic rejection) when R² is below a threshold. The operator decides whether to use the file anyway, since nothing is applied automatically. Is that enough, or do you want the linear-range search and automatic rejection described earlier?
  - Answer: yes, that is enough. keep the procedure simple.

- **R6. Confirming the zero point.** The agent's understanding of Q6: the loop always drives the pair offsets to zero, on sky as on the internal source. The δ1 *value* at which that happens differs between the two (because the atmospheric dispersion differs), so the internal-source null is diagnostic only and no reference offset `p_ref` is subtracted. Is that correct?
  - Answer:
    yes, this understanding is correct.

- **R7. Where the calibration code lives.** Earlier, all code was to go in `app.py`. Given the concern about verbosity, should the calibration be:
  - (a) a new `calibrate` element in the existing `state` switch, in `app.py`;
  - (b) a separate module in the same package (e.g. `xapp/adcCtrl/calibration.py`), triggered from the app;
  - (c) a standalone script, run separately, that imports the measurement functions from `app.py` and writes the FITS file, leaving the app itself almost unchanged?
  - Answer: (a)

#### Remaining Questions (Round 4)
*Raised by the round 3 answers.*

- **S1. Fallback and the co-rotation diagnostic without a matching calibration.** Single-pair fallback needs each pair's own slope (`−1/s_j`), and the co-rotation diagnostic needs the direction of `s`. Neither can be derived from `ctrl_mtx` alone. The proposal:
  - Both come only from the active calibration file. With no file configured, fallback is disabled (a frame with an invalid spot is discarded, as today) and the co-rotation component is not reported. Each of these is logged once at INFO.
  - If the operator changes `ctrl_mtx` by hand, fallback and the diagnostic keep using the file's values.

  Is that acceptable?
  - Answer:
    first bullet is acceptable. second bullet iss not: if `ctrl_mtx` is changed by hand, fallback should also be disabled. 