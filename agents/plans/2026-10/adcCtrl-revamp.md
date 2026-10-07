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

_Plan by Claude Opus 5.5 on branch `ktwitchell/adcCtrl-v2` (base `dev`). Rev 1: 2026-10-07, initial draft. Rev 2: 2026-10-07, updated for the user's "Answers to Agent Questions". Rev 3: 2026-10-07, N1/N2 accepted. Rev 4: 2026-10-07, functional implementation done (see Implementation Log). Status: **functional changes committed. Build-machine verification pending.**_

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

**Decisions made during implementation:**
- **Core mask diameter** is now `mask_factor × separation` in λ/D for both spot sources. The old code computed the sparkle mask as `separation / (6/21) × 0.7` (a pixel conversion applied to a λ/D grid). For sparkles at 15 λ/D that gives a 36.75 λ/D diameter mask, whose radius (18.4 λ/D) is larger than the spot separation, so it would have masked the sparkles themselves. The DM-spot mask was `47 × (6/21) × 0.7 ≈ 9.4` λ/D. Both now follow the user's stated "0.7 × separation" rule.
- **Radial-profile bin size** is `radial_bin = 5 × 6/21 ≈ 1.43` λ/D, which equals the 5-pixel bin validated in the on-sky notebook. The old app passed `5` on a λ/D grid, which is 17.5 pixels.
- **DM spot separation** (47) is interpreted as λ/D at the observing wavelength and scaled by `normalized_wavelength`, as for sparkles (this was the old `utils.py` `find_speckle` behavior).
- **Expected spot axes** for the wrap fix: spots 0 and 2 at `90° − angle`, spots 1 and 3 at `−angle`. This matches the notebook's on-sky result of 54–74° for the DM spots at 28°. The synthetic tests show the old `abs()` failure directly: with sparkles at angle 0, spots 0 and 2 read +89.5° and −89.5°, so |θ0| − |θ2| = 0, while the new signed deviations correctly give −0.5° and +0.5°.
- **Gain default** is 0.5 (config `gain`). The old INDI `gain` property displayed 0.10 while the code actually used `_gain = 0.5`. The property now shows the value in use.
- **Startup zeroing** of `deltaADC1/2` is queued and applied by `loop()` once `adctrack` is visible. It now always writes 0, where before it only wrote if `deltaADC1.current != 0`. Writing 0 when it is already 0 is harmless.
- **No δ1 resync from `adctrack`.** `current` only updates after the stages move, so resyncing from it would discard commands while tracking is off. The app keeps its own integrator, as before.
- **Repeated failures:** after `max_consecutive_failures` failed closed-loop cycles the app goes idle, and `fsm` returns to READY. The reason is shown in `status.last_error` and the log.
- **Config schema change:** `camera.shmim` / `camera.dark_shmim` are replaced by a `cameras` table (`shmim`, `filter_wheel` per entry; defaults camsci1/fwsci1 and camsci2/fwsci2). Darks come from XCam's automatic `<shmim>_dark` detection. **Any deployed `/opt/MagAOX/config/adcCtrl.conf` with a `[camera]` section must be updated.**
- The INDI property `labmode` is now `loop_sign` (`positive`/`negative`). New read-only properties: `measurement`, `command`, `status`. New switch: `camera`.

**Test coverage vs. plan:** T1–T23 are implemented as 27 test functions in `apps/adcCtrl/test/test_adcCtrl.py`, plus a `loop_sign` test. T7 uses a numpy broadband model (core and spots scaled with λ, core shifted linearly across the band) rather than a full hcipy optical propagation.

### Verification

- **Local (this Mac):** all 27 tests pass using real numpy 2.4.3, scipy 1.17.1 and hcipy 0.7.0, with *stub* `xconf`, `purepyindi2` and `magaox` modules (pip could not reach PyPI from the sandbox). This validates the algorithm and control logic and the device logic against a minimal property model. It does **not** validate the real purepyindi2 property, message or `Device` APIs.
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
