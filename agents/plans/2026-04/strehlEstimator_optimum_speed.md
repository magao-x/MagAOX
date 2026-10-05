We need to add a tool to strehlEstimator that shows the optimum loop speed given star brightness.  This should be calculated at 100 Hz intervals from 100 Hz to 3000 Hz.  To support using this for planning, we need a new INDI element to the `star_mag` property called 'estimated', add a property to report `seeing` with `current` and `estimated`, and a `wind_speed` property with `current` and `estimated`.   We need a toggle switch for `use_estimates`.  When on, the estimated values are used. When off, the current values are used.  For `star_mag` and `seeing` the current values should be kept up to date with the on-line values.  For `wind_speed` `current` should just mirror `estimated`.

Please review AGENTS.md, then make a plan to implement this new tool.  Fill in this document with the plan below, but do not edit this prompt.  Do not make begin implementing until I have review the plan.

Plan:
1. Review the current `strehlEstimator` implementation and keep the existing live behavior intact while adding a planning path on top of it.
   The current app is a header-only MagAO-X app that already:
   - derives live star magnitude from WFS counts in `calcMag()`
   - subscribes to live `fps`, `emgain`, beamsplitter stage, TCS seeing, and telescope elevation
   - computes predicted Strehl and WFE in `appLogic()` using `m_aosys`
   The implementation should preserve that existing live path, stage-dependent photometric calibration, and current INDI output names unless a new planning surface specifically requires a new property.

2. Split the atmospheric/planning inputs into explicit `current`, `estimated`, and selected values so the `use_estimates` toggle has a single well-defined effect.
   The planned runtime state is:
   - `star_mag.current`: still driven by WFS counts and camera settings
   - `star_mag.estimated`: operator-entered planning value
   - `seeing.current`: still driven by the live `tcsi.seeing.dimm_fwhm_corr` feed
   - `seeing.estimated`: operator-entered planning value
   - `wind_speed.estimated`: operator-entered planning value
   - `wind_speed.current`: mirrored from `estimated`, per the prompt, with no new external wind subscription in this change
   - `use_estimates`: boolean switch that selects whether predicted performance calculations use the live/current or estimated inputs
   To keep the logic easy to reason about, the implementation should add helper accessors or a small update helper that returns the selected star magnitude, selected seeing, selected wind speed, and the derived `r0` used by `m_aosys`.

3. Expand the INDI interface so the planning inputs can actually be edited online.
   The current `star_mag` property is read-only and only exposes `current`, so it cannot support the requested `estimated` input without being reworked.
   The plan is to:
   - convert `star_mag` from a read-only number property to a writable/custom number property that still publishes `current` but accepts `estimated`
   - add a writable `seeing` number property with `current` and `estimated`
   - add a writable `wind_speed` number property with `current` and `estimated`
   - add a writable `use_estimates` toggle switch
   - keep the properties in the existing Error Budget/operator-facing area unless a better local group name is already in use nearby
   Because the prompt explicitly asks for `current` and `estimated` element names rather than the usual `current` and `target`, this likely needs explicit property construction and `registerIndiPropertyNew(...)` callbacks rather than the standard `CREATE_REG_INDI_NEW_NUMBER*` helpers that assume `target`.

4. Reuse the existing live properties where that keeps the change smaller, and keep the new callback behavior narrow and predictable.
   The current plan is:
   - continue using `m_indiP_mag` for `star_mag`
   - repurpose the already-declared `m_indiP_seeing_magaox` member for the new local `seeing` property rather than introducing a redundant second seeing member
   - add one new `pcf::IndiProperty` for `wind_speed`
   - add one new `pcf::IndiProperty` for `use_estimates`
   Callback rules should be:
   - operator writes update only the `estimated` element, never `current`
   - writes to `current` are ignored or overwritten on the next update cycle
   - changing `wind_speed.estimated` updates both `estimated` and `current`
   - changing `use_estimates` immediately changes which inputs are fed into the predicted-performance calculations
   - invalid operator payloads such as non-finite or non-positive seeing/wind values should be rejected without disturbing the last valid estimate

5. Centralize the AO-model setup so both the current-speed prediction and the new optimum-speed tool use the same selected inputs and the same physical assumptions.
   The implementation should add a helper that configures an `aoSystem` instance from:
   - selected star magnitude
   - selected seeing converted to `r0`
   - selected wind speed applied through `m_aosys.atm.v_wind(...)`
   - current elevation, stage-dependent wavelength/flux terms, EM gain, and WFS pixel count
   - a requested loop speed expressed as `tauWFS = 1 / fps`
   This keeps the current live prediction and the new scan consistent and avoids duplicating the AO-system setup across `appLogic()` and several callbacks.

6. Implement the optimum loop-speed scan as a discrete sweep from 100 Hz through 3000 Hz in 100 Hz steps, and keep it separate from the live/current-speed prediction.
   The planned evaluation path is:
   - keep the existing predicted Strehl/WFE outputs for the actual loop speed, using the selected current-vs-estimated inputs based on `use_estimates`
   - add a second calculation path that evaluates the same AO model at each candidate speed in the fixed 100 Hz grid
   - explicitly set `optTau(false)` for the scan path and set both `minTauWFS` and `tauWFS` to the sampled `1 / fps` value so the scan honors the requested discrete rates instead of letting the analytic model optimize continuously
   - use a local `aoSystem` copy or a temporary model object for the scan so the sweep does not perturb the live `m_aosys` state used for the current-speed outputs
   - choose the best sampled speed by maximum Strehl; on exact ties, keep the first maximum encountered so the lower FPS wins ties deterministically

7. Publish the scan result through a new read-only summary property rather than exploding the INDI surface with thirty per-speed elements in the first pass.
   The initial plan is to add one new read-only property, tentatively named something like `loop_speed_optimum`, containing at least:
   - `fps`
   - `strehl`
   - `wfe_total`
   - optionally `wfe_measurement`, `wfe_time_delay`, and `wfe_fitting` if the extra breakdown is useful to operators and mirrors the existing `wfe_predicted` property
   This keeps the first implementation focused on the requested operator question, "what speed is best?", while still computing the full 100-3000 Hz sweep internally.
   If later review says operators also need the full sampled curve, the scan helper from this change can be extended to populate an additional per-FPS property without redesigning the core logic.

8. Make sure the update triggers are broad enough that both the live/current-speed prediction and the optimum-speed summary stay fresh as conditions change.
   After this change, recalculation should occur when any of the following change:
   - WFS counts or mask-derived `npix`
   - `fps`
   - `emgain`
   - beamsplitter preset / photometric calibration branch
   - TCS seeing
   - telescope elevation
   - any operator-entered `estimated` value
   - `use_estimates`
   This can likely be implemented by keeping the existing callback updates and having them call a common "refresh predictions" helper or by continuing to recalculate in `appLogic()` while only updating the underlying state in callbacks.

9. Add unit-test coverage for the new input-selection and optimum-speed behavior instead of leaving the file as a placeholder harness.
   The intended test coverage is:
   - default construction still succeeds
   - the new writable INDI properties are created with the expected elements
   - `star_mag.current` continues to track `calcMag()` while `star_mag.estimated` remains operator-controlled
   - live TCS seeing updates only `seeing.current`
   - wind-speed writes mirror `estimated` into `current`
   - `use_estimates` flips the selected inputs used by the prediction path
   - invalid estimated seeing/wind payloads are rejected cleanly
   - the optimum-speed helper scans only the requested discrete 100 Hz grid
   - the optimum-speed helper returns the best sampled FPS with deterministic tie handling
   - stage and camera-setting changes still propagate into the predictions
   The test file should also be brought up to the current documentation style across the full file, including per-`TEST_CASE` Doxygen blocks and any needed Doxygen-only symbol references if the test harness hides the real API under test.

10. Run the normal cleanup and verification steps after implementation.
   The planned verification pass is:
   - run `clang-format` on touched source/header/test files
   - build the `strehlEstimator` unit-test target
   - run the relevant `strehlEstimator` tests
   - if an interactive INDI smoke test is practical, confirm that `star_mag`, `seeing`, `wind_speed`, `use_estimates`, and the new optimum-speed summary property all update as expected while toggling between live and estimated inputs

Assumptions captured in this plan:
- `seeing.current` should continue to come from `tcsi.seeing.dimm_fwhm_corr` in this change.  The existing `mag1`/`mag2` seeing fields can remain unused until there is a separate request to choose among them.
- `wind_speed.current` intentionally mirrors `wind_speed.estimated`; no new TCS or telemetry wind feed is planned here.
- The first implementation should publish the best sampled operating point, not the entire 30-point curve, unless review says the extra surface is needed immediately.

Implementation note:
- The delivered follow-up UI changed `wind_speed` from a numeric `current`/`estimated` property to a one-of-many switch with `slow`, `normal`, `fast`, and `very-fast` options mapped to `9.4`, `18.7`, `23.4`, and `30.0` m/s.
