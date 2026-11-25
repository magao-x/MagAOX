# Task Description

## Problem Statement

Replace the fixed weak-mode derating in the MagAO-X high-order modal gain optimizer with a detector-noise model that tracks the guide-star flux and camera settings. Use camera-derived noise or the detector calibration in disturbance recovery and gain optimization. Continue estimating the disturbance spectrum from the measured modal telemetry and extrapolating its high-frequency tail.

## Discussion

The local `hogopt.conf` uses `power_law_only` with `auto_smoothed_crossing` and estimates a flat noise floor from the minimum of the closed-loop WFS PSD below 100 Hz. The floor influences noise subtraction, where the extrapolated disturbance tail takes over, and the optimizer's measurement-noise cost. Changing the optimizer input alone leaves the original noise estimate in the disturbance model.

The high-order optics and DM can limit usable loop gain even when detector noise is low. Use an independently configurable ceiling on effective gain so the noise objective can still choose a lower gain. The initial ceiling for loop 1 is 0.30 for zero-based modes 840 through the end of the active mode vector. Treat 0.30 as the starting value for on-sky evaluation.

The two matched plane-5 self-response measurements give a repeatable estimate of the on-mode sensitivity. In the 2 kHz, magnitude-8, EMG 595 comparison, the 118 modes selected by the installed `aol1_mgaincalfact` have a median modeled noise PSD 3.46 times the median of the unflagged modes in their respective SVD blocks. This comparison is of noise PSDs. The optimum gains also depend on the disturbance spectrum and loop dynamics.

The offline analysis lives in the `loop_psds` workspace. Its relevant products are `outputs/psd2/reconstructor_noise/modal_noise_templates.fits`, `pn_vs_mode_2khz.png`, and `pn_vs_mode_2khz.txt` in the same directory. The magnitude comparison uses the Strehl estimator zero points for the Hα/IR and 65/35 beamsplitters with the WFS filter open. The online calculation uses measured WFS flux directly. [WFS bandpasses and zero points](https://magao-x.org/docs/handbook/observers/filters.html#wfs)

## Requirements

1. Predict a one-sided photon-plus-read noise PSD per mode from the active reconstructor calibration, masked dark-subtracted flux, camera EM gain, and measured frame rate.
2. Use the selected noise PSD to recover the empirical disturbance spectrum and determine the extrapolation crossover. Use that same noise PSD in the optimizer's input units for the SI and LP objectives.
3. Distinguish the transfer of detector noise into the measured WFS residual from its transfer into the physical residual. Use the current controller for disturbance recovery and the candidate controller for each gain objective.
4. Make the new noise workflow opt-in. Default to camera-derived noise in that workflow and allow switching to detector calibration while the app is running. Report the requested and active noise sources. Retain the current modal-PSD empirical estimator as a fallback.
5. Evaluate the calibrated workflow with `mgaincalfact` set to unity. Continue using `mgaincal` for effective-to-commanded gain conversion.
6. Support configurable effective-gain ceilings by mode range. Start at 0.30 for mode 840 and above, inclusive. Modes below 840 use the existing stability bound.
7. Publish a live two-column noise comparison shared-memory image. For each mode, column 0 is the median open-loop-equivalent measured WFS PSD from 0.8 times Nyquist to Nyquist. Column 1 is the predicted open-loop-equivalent noise PSD from the active camera or detector calibration source.

## Tests and Metrics

- Check the predicted PSD normalization and its flux dependence against direct propagation through the WFS normalization and reconstructor.
- Recover a known disturbance spectrum from equivalent synthetic open-loop and closed-loop telemetry, including a case where detector noise dominates the high-frequency bins.
- Replay the same on-sky recordings with the current workflow, with calibrated noise only in the objective, and with calibrated noise in disturbance recovery and the objective. Compare predicted versus observed PSDs, extrapolation crossover, modal gains, and integrated residual variance. Report the previously derated modes separately in the comparison.
- Report the modes that hit the configured ceiling and the associated change in modeled residual variance. Compare capped and uncapped optima from the same recording.
- Compare the two published columns in the November 22 on-sky replay. In the existing 800–1000 Hz check at 2 kHz, the median measured-to-camera ratio was 1.011 across 1,563 modes. The 5th to 95th percentile range was 0.938–1.119.

## Scope and Caveats

Start with `hogopt` / loop 1 and diagonal per-mode detector-noise power for the matched WFS calibration. The full modal covariance and a coupled optimizer are follow-up work. Use the November constant-delay result (546.65 µs at about 2 kHz) where it is the applicable calibration. Use the actual gain, leak, delay, and controller for each replay. Instrument acquisition and commissioning are performed by the operator.

# Instructions to Agent

Analyze the above task and document the implementation plan under “Agent Findings and Plan”. Keep the plan current as the work progresses.

Review [AGENTS.md](../../../AGENTS.md). Preserve the task description and begin implementation after the user reviews the plan and resolves any open questions required for implementation.

# Agent Findings and Plan

## Task Summary

[The gain loop](../../../apps/modalGainOpt/modalGainOpt.hpp) currently sends the measured modal PSD and current transfer functions to [`modalPsdProcessor::analyzePsd`](../../../apps/modalGainOpt/modalPsdProcessor.hpp). The processor estimates noise, uses it to recover and extrapolate the disturbance, and returns both PSDs. `modalGainOpt` then passes them to `clGainOpt` and scales the published gain by `mgaincalfact`. The calibration belongs at the PSD processor input and in the optimizer's noise vector.

## Key Assumptions

- The supplied `gain` is mean conversion gain in input electrons per ADU. `ron` is input-referred RMS read noise per pixel in electrons.

  | Camera EMG | Mean gain (e⁻/ADU) | Read noise (e⁻ RMS/pixel) |
  | ---: | ---: | ---: |
  | 10 | 1.1535 | 14.51 |
  | 100 | 0.144 | 2.0375 |
  | 595 | 0.0282 | 0.40675 |

- The prototype uses the large-register EM excess variance approximation `F² = 2 − 1/EMG`. Multiply the photon variance by `F²`. The read RMS comes from the supplied camera calibration. [EMCCD noise model](https://doi.org/10.1109/TED.2003.813462)
- The active matched WFS reference supplies the pixel flux fractions `p`. The reconstructor `R` includes the WFS mask and runtime `MODENORM`. The plane-5 selfRM amplitudes describe the static response from DM command mode to WFS modeval at the calibration conditions.
- The normalization floor in the current tweeter configuration is 1 ADU per full-frame pixel (14,400 ADU for a 120 × 120 WFS frame). Derive it from the active frame size and normalization configuration.
- The detector-noise model is white at the sampled WFS output. Low signal, saturation, excess dark or clock-induced charge, and time-dependent or modal optical gain can affect its agreement with telemetry.

## Requirements

The implementation must preserve PSD coordinate and transfer-function conventions across subtraction, extrapolation, and gain optimization.

For dark-subtracted masked signal `S` in ADU/frame, let `K` be the normalization floor in ADU, `N_e = S × gain`, and `β = S/(S+K)`. The pixel normalization Jacobian followed by modal reconstruction is proportional to `T_β = R − β(Rp)1ᵀ`. Use the diagonal contractions `q_ph,i = [T_β diag(p) T_βᵀ]_ii` and `q_rd,i = [T_β T_βᵀ]_ii`. At interior frequency bins the raw WFS modeval noise PSD is

```
P_n,WFS,i = (2 / f_s) × [F² × N_e × q_ph,i + ron² × q_rd,i] / [(S + K) × gain]²
```

For the measured residual PSD `P_y,i` in raw WFS modeval units, the current closed-loop power transfer for sampled detector noise is `E_current = |ETF_current|²`. Infer the open-loop disturbance in the WFS domain using `max(P_y,i / E_current − P_n,WFS,i, 0)` at valid bins. Set `E_current=1` for open-loop data. Compare the recovered disturbance with `P_n,WFS,i` in this same domain to locate the extrapolation crossover.

For the physical gain objective, refer the calibrated noise to the DM input using the live modal response squared. Use a consistent static response for the disturbance and account for the WFS exposure response when converting the WFS disturbance spectrum to a pre-exposure physical spectrum. Apply the WFS exposure response only to the disturbance conversion. The detector PSD already describes sampled noise. Audit the existing `m_opticalGain²` division so both spectra get the same static response conversion once. For the noise objective, `clGainOpt` already applies the candidate `|NTF|²`.

For the camera-derived source, estimate the high-frequency pixel variances in raw dark-subtracted WFS ADU using contiguous camera samples. Propagate the diagonal pixel noise estimate through the active mask, flux-normalization Jacobian, and runtime reconstructor. The replay used the high-frequency pixel spectrum at the observed flux to capture both photon and read noise. The detector-calibration source evaluates the photon and read terms above from the same live flux and camera settings. Use the selected source for both disturbance extraction and optimization.

For the comparison shmim, define `f_N = f_s / 2`. Compute column 0 from the valid bins of `P_y / |ETF_current|²` in the nominal band `[0.8 f_N, f_N]`, with unity ETF when the loop is open. Use the median in log-PSD space and convert back to linear units. Exclude the standalone Nyquist bin when the one-sided PSD producer gives it a different weight. Column 1 is the selected detector-noise prediction before feedback. Both columns use one-sided WFS modeval²/Hz, the same mode order, and the same PSD update. Use dimensions `x=2` and `y=nModes` and publish the pair together. Form the measured column before any noise subtraction or disturbance extrapolation. Flag unavailable values and report the prediction source. If the modal-PSD fallback is active, flag the predicted column as unavailable so the display can identify when an independent prediction is missing.

Apply the configured effective-gain ceiling to the search interval after computing the existing stability limit and its margin. Use the smaller upper bound for each mode. Apply this before conversion through `mgaincal` and optical gain. A candidate at the ceiling is a valid optimum. Bound integrated or cached gain targets when the configured ceiling is lowered, so published recommendations honor the new limit.

## Questions and Points of Clarification

- The 2026-11-22 on-sky bundle supplies simultaneous camera and modeval telemetry plus applied modal gains for the first replay. Matched open/closed pairs with recorded camera settings, masked flux, loop gains, leaks, and timing will extend validation across flux levels.
- Check the selfRM acquisition normalization and optical conditions so the live response can be referenced to the calibration conditions. For flux-conserving pokes, the floor correction is `β_live / β_cal`. Validate the optical response scaling with the on-sky replay before enabling the calibrated path operationally.
- The three supplied EM settings cover the initial calibration. Decide how to extend the lookup after checking measured calibration coverage at other settings. If an EM setting lacks a detector calibration, use the camera-derived estimate when available and the empirical fallback otherwise.

## Tests

- Use the existing [`modalGainOpt` tests](../../../apps/modalGainOpt/tests/modalGainOpt_test.cpp). Compare the per-mode noise calculation against a direct pixel Jacobian calculation with zero and nonzero normalization floors. Check shot-noise and read-noise limits, the one-sided PSD integral, and optical response scaling.
- Generate known disturbance and detector-noise spectra for open loop and several closed-loop gains. Forward-model the WFS PSD, recover the disturbance, and verify the crossover and gain objective. Include frequencies where `|ETF|²` and `|NTF|²` differ substantially.
- Exercise the extrapolation and no-extrapolation paths, near-zero ETF bins, and invalid or mismatched calibration, flux, or camera settings. Check that the empirical selection reproduces the legacy results and that fallback uses consistent noise for processing and optimization.
- Compare the three replay configurations using the same telemetry and transfer model. Use unity `mgaincalfact` in both calibrated configurations and the installed factor in the legacy baseline. Record the empirical and calibrated floors, residuals in the measured WFS domain, crossover frequencies, gain curves, and variance by mode. Include modes above and below the original derating threshold.
- Exercise modes 839 and 840 explicitly and a high-order mode at the end of the active basis. Verify that an optimum below 0.30 passes through, a larger optimum lands at the ceiling, and a tighter stability bound takes precedence. Verify SI and enabled LP gain paths, including regularization and cached gain updates. Check command-space publication with non-unity `mgaincal` and optical gain.
- Check the comparison stream dimensions, column order, units, band endpoints, and loop-open and loop-closed ETF handling. Verify that predicted values change with the selected source, the measured column uses the same telemetry in either case, and invalid inputs appear as unavailable. Check source changes and empirical fallback.

## Implementation Plan

1. Define a compact calibration tied to the active reconstructor and mode order. Store the per-mode photon/read diagonals, the two row contractions needed to update them for `β`, and the matched selfRM response. Load and validate it once per calibration change.
2. Compute one noise scalar per mode per PSD update. Use `acquWFS-1.out.fluxtotal` or an existing averaged masked WFS count after verifying the same dark subtraction, pixel scaling, and mask. Match the flux and camera setting to the PSD interval. For the camera-derived source, use buffered contiguous frames and compute pixel variances and modal projections away from the loop-critical processing path. Reuse existing streams where possible and measure any added acquisition or publication overhead. Provide a configuration default and a runtime source selector using the existing INDI patterns. If the selected detector model is unavailable, use a valid alternative detector estimate when available, then the empirical modal-PSD fallback. Report which source was used.
3. Extend `analyzePsd` to accept the selected detector-noise PSD in WFS modeval units. In that path, apply the current ETF power to obtain the expected detector component in the observed PSD. Normalize the recovered disturbance and the noise used by the crossover to the same domain. Apply the same subtraction when extrapolation is disabled. Route the closed-loop significance check through the matching detector component when enabled.
4. Populate the optimizer's disturbance and noise inputs in a consistent coordinate system. Feed the selected, input-referred noise to both SI and LP. Publish the selected noise source, its PSD, and an empirical comparison so a replay can trace a change in gain back to noise subtraction or to the objective.
5. Add optional ceilings for zero-based mode ranges, with an open-ended range extending to the end of the active basis. Configure loop 1 with 0.30 from mode 840 onward. Apply the ceiling for either noise source, including empirical fallback, in the SI search and in the LP regularization and gain updates when enabled. Keep each controller's stability bound and expose the applied bound and modes that reach it.
6. Publish the two-column comparison as a dedicated per-loop shmim at the modal PSD update cadence. Assemble both columns in a private buffer and publish them as one update. Reuse the existing PSD stream lifecycle and expose the active source for the live display.
7. Validate the opt-in path with software tests and on-sky replay. Compare with the current output before using calibrated noise with a unity derating vector on the instrument.

## Follow-up and Edge Cases

- Use the 2026-11-22 on-sky replay to see where the 0.30 ceiling binds. Use bright-star sweeps with pupil tracking active to refine the configured ceiling by SVD block as needed.
- Evaluate excess detector terms with dark or faint-star data if the predicted detector component disagrees with the measured WFS PSD.
- Check mismatched or changing calibrations, camera settings that change within a PSD window, and modes with insufficient signal to anchor a tail. Treat low-significance bins as unreliable extrapolation anchors.
- Revisit full modal noise covariance or optical gain by mode if repeatable structure remains in the on-sky residual.

## Implementation and commissioning

- Loop 1 defaults to an effective-gain ceiling of `840-:0.30`. Other loops default to `none`. Set `[loop] effectiveGainCeilings` or change the `effective_gain_ceilings.target` INDI text element. Ranges are inclusive (`first-last:gain`) and comma-separated. An empty upper mode extends through the active basis. Overlapping ranges use the tightest limit. Use `none` to recover the original uncapped search for replay.
- The independent-noise path defaults to disabled. In the operator's `hogopt.conf`, set `[noise] enabled=true` after commissioning. By default `hogopt` follows `$MILK_SHM_DIR/aol1_calib_source.txt` and `aol1_calib_loaded.txt` written by `cacao-calib-apply` (`/milk/shm` by default). It reads the applied archive's `CMmodesWFS/CMmodesWFS.fits`, `wfsref.fits`, and `wfsmask.fits`. Each archive supplies a `selfRM_cal.fits` symlink to its matched cube. The optional `calibrationDir` and `selfRM` overrides also support standalone calibration directories and alternate matched cubes for offline runs. Set `selfRMBeta` to the measured flux fraction during selfRM acquisition. If the selected archive changes, verify the configured beta still describes its selfRM measurement. The app reloads on changes to the CACAO apply records or the FITS files and verifies mode count, image dimensions, and the live binary mask. It uses fallback while a fresh PSD window accumulates after reapplication. The full-frame floor is frame width × height × `floorPerPixel` (default 1 ADU). Set `modeNorm` to match the active reconstructor (default true).
- The masked flux is the mean of dark-subtracted `camwfs_avg` samples published within the PSD window. The existing `[wfsavgShmim] shmimName` override can supply an equivalent averaged stream. The installed `camwfs-avg` configuration integrates `camwfs` for 2 s with `camwfs_dark` subtraction. Verify the active camera average uses the same dark and pixel scaling as WFS acquisition. The updated `modalPSDs` tags the first contributing sample in `PSDT0` and the last in `atime`. For older PSD publishers the app estimates the time span using the subscribed PSD durations and publication time.
- Set `cameraBuffer` to a timestamped uint16 history of the ordinary raw camera output (default `camwfs_noise_cbuff`). It needs to preserve the source frame counters and retain the PSD averaging interval plus producer and consumer processing time. The installed raw camera stream is one frame deep. The buffered source requires an operator-configured producer and an instrument-side timing check of any new acquisition reader. The optimizer locates four 128-sample windows by acquisition time and uses the top 20% of the band below Nyquist. It fills an isolated all-zero frame or one skipped frame counter with the pixelwise mean of its real temporal neighbors. The white-noise normalization accounts for the interpolation. A block with consecutive missing frames or non-increasing counters triggers fallback. The static dark contributes only to the pixel offset. The periodogram removes the offset, and the flux normalization uses dark-subtracted ADU.
- The default `source=camera` can be changed with `noise_source.target` to `camera`, `detector`, or `empirical`. The app reports the applied source in `noise_source.current`. When the requested independent source is unavailable it tries the other source, then the empirical estimator. FPS or EM changes that overlap a PSD interval invalidate the independent prediction for it. Detector lookups use the three measured EM settings. The EMG 200 replay supports leaving interpolation disabled. The prior interpolation predicted about twice the observed high-frequency modal floor at that setting.
- Both independent sources use the current ETF in raw WFS units for disturbance recovery and the tail crossover. The SI and LP objectives use a shared static-response conversion for both PSDs and an exposure correction for the disturbance. All sources use the loaded `mgaincalfact`, `mgaincal`, and optical gain for command publication. Load a unity `mgaincalfact` vector explicitly for the unity-derating experiment. The maximum loop gain is still determined for each controller before applying the configured ceiling.
- `aol1_noiseComparison` has dimensions 2 × modes and is published with a `SOURCE` keyword. Columns 0 and 1 are the pre-subtraction log median and the independent prediction, in WFS modeval²/Hz. An unavailable value is IEEE NaN. Set `halfWeightNyquist=true` for PSD producers that half weight the standalone Nyquist bin. `modalPSDs` currently scales all bins together. `aol1_gainLimits` has dimensions 4 × modes, with columns for the effective SI and LP search bounds and the SI and LP ceiling-hit flags. An unavailable LP result is NaN.

## November 22 replay results

The replay used the 09:09:28–09:09:50 UTC bundle, the archived applied gain and leak vectors, 1,999.998 Hz sampling, optical gain 0.8, and a 546.65 µs delay. It used the archived modal Welch PSD and ETF grid in `outputs/psd2/onsky_20261122/replay_inputs.npz`. The decoded raw frames from five contiguous camera captures were passed through the new camera estimator and the matched FITS reconstructor. The damaged 09:09:30.985 UTC camera chunk was unavailable. The replay used the matched plane-5 selfRM and assumed selfRM beta near unity for the relative response correction because the selfRM flux was unavailable. It ran the app's C++ PSD processor and SI gain objective offline, with the archived derating vector for the legacy baseline and unity for the calibrated comparisons.

Across 1,563 modes the measured-to-camera ratio from the implemented estimator was **1.006**, with 5th–95th percentiles **0.932–1.112** for the 800–999 Hz band. The same comparison code gave **1.011** and **0.938–1.119** using the previous offline camera floor. Processing the five captured camera windows took roughly 26–45 ms per update on the workstation, including modal projection. Instrument-side acquisition and publication costs still need measurement.

| Population | Modes | Recorded effective gain | Legacy | Camera noise in objective | Camera noise in recovery and objective | Recovery and objective, capped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 1,563 | 0.160 | 0.167 | 0.417 | 0.517 | 0.505 |
| Historically derated | 118 | 0.052 | 0.074 | 0.265 | 0.330 | 0.300 |
| Other modes | 1,445 | 0.169 | 0.176 | 0.426 | 0.530 | 0.523 |

Values are medians. The median crossover moved from 83 to 200 Hz overall and from 66 to 195 Hz for historically derated modes. The empirical floor was 11.13 times the camera estimate at the median mode. Evaluated against the same camera-based physical PSDs, the uncapped joint workflow reduced median modeled variance by 41.7% from legacy and by 3.8% from objective-only. For the historically derated modes those reductions were 42.9% and 2.7%, respectively.

The 0.30 ceiling bound **708 of the 723 eligible modes**, including **78 historically derated modes**. It increased the sum of modeled per-mode residual variances by **2.92%** overall and **0.70%** for the historically derated population. The median increase among ceiling-limited modes was 2.66%. The bounded mode indices were 840–1356, 1358–1366, 1368, 1370–1372, 1375, 1378–1379, 1382, 1384–1534, 1536–1542, 1544–1551, 1553–1558, and 1561–1562. These results describe modeled variance for the recorded single-integrator controller. The predictor path is covered by software tests.

The matched selfRM acquisition flux and live optical response scale still need operator validation. Verify the camera buffer's acquisition cost and the initial ceiling on the instrument before operational use.

Both applications built locally. All eight `modalPSDs` tests and all seven new `modalGainOpt` tests passed. The full optimizer suite passed 47 of 49 tests. The existing automatic-crossover boundary and initial predictor mode-count assertions failed on the original `dev` revision too (40 of 42 passed).

The CACAO archive lookup was verified offline with two synthetic applied archives, each containing a `selfRM_cal.fits` symlink. The test covers a pending apply record, reapplying the same archive, and a new PSD window after each completed apply. Repointing an archive alias while the apply records are unchanged keeps the previously resolved archive. The updated optimizer suite passed 48 of 50 tests with the same two baseline failures.


## Command conversion follow-up

A live commissioning comparison used detector noise, global gain 1, and optical gain 0.74. The optimizer had been substituting unity for the loaded `mgaincalfact` when an independent source was active. The original 118 weak-mode entries equaled the old `mgaincal` values and canceled inverse gain calibration on those modes. The November fit lowered several weak-mode values further and hit its 0.30 lower bound on ten modes. For example, an effective target of 0.30 at mode 1078 produced a command of 1.351 with unity derating. The converted command is 0.405 with its matched 0.30 factor.

The command conversion and SI state sync now honor the loaded vector for every source. CACAO applies the global gain to the live `mgainfact`. The current effective gain uses that applied value times `mgaincal` and optical gain. The extra factor belongs to recommendation publication because the applied vector can already contain it. Replacing either gain calibration vector rebases the SI integrator from the applied commands. A candidate commissioning vector preserves the historical 118-mode selection and takes the suppressed entries from the November `mgaincal` so that the same weak-mode compensation policy applies to the new pair. It was evaluated offline with the archived camera-noise optima and the reported optical gain of 0.74. The flagged-mode median and maximum raw commands went from 0.856 and 2.074 with unity, to 0.541 and 1.472 with the old factor vector, to 0.405 and 0.844 with the matched factor vector. The remaining gain structure comes from the computed effective gains.

The fitted November conversion already includes the response during the fit. Using the configured optical gain as a multiplier assumes it is relative to that reference. The fit products give the fitted effective-to-commanded ratio and do not record a separate optical gain reference. The command conversion follow-up preserves this existing reference convention.

The command publication regression covers all three noise sources and an explicit switch to unity factors. The application built and the full suite passed 48 of 50 tests, with the two existing baseline failures.


## Isolated camera frame gaps

Acquisition time places the blocks within the modal PSD interval. Frame counters determine sample spacing within each block. A jump by two identifies one missing sample between the measured frames. Driver timestamp jitter within the PSD interval does not affect that decision. A frame omitted before the source publishes its next counter can leave consecutive counters and is undetectable from those counters. An all-zero frame is treated as a placeholder. A zero in an otherwise valid frame is retained as data.

In the offline November camera replay, all five archived camera intervals yielded estimates with their original frame sequences and after deliberately removing one frame from each of the four sampled blocks. The complete-frame run reproduced the original camera estimates to the output's displayed precision. The repaired-to-complete modal PSD ratio had a median of 1.00043 and 5th–95th percentiles of 0.99935–1.00144 across 1,563 modes. The synthetic test also covers zero-filled frames, gaps at block edges, and timestamp jitter with both consecutive and skipped counters.

The frame-counter regression passed all 22 assertions. The full optimizer suite passed 49 of 51 tests, with the same crossover-boundary and initial predictor mode-count failures recorded above.

## Gain continuity through noise fallback

Noise-source transitions use the next PSD and selected noise prediction as inputs to the existing gain update. The SI integrator continues from its previous recommendation. The predictor keeps its regularization scale and per-mode schedule during fallback and recovery. With calibrated noise enabled, the predictor checks the stability bound for the current coefficients even when it reuses the regularization scale. The raw SI output shows the result from each PSD before integration.

The focused source selection, SI integration, and predictor bounds tests passed. The full suite passed 49 of 51 tests with the same two baseline failures.

## Reconstructor latency during testing

Set `[loop] computeTauReference` in `hogopt.conf` to the compute latency included in the loaded looptau calibration, in seconds. The default is `0.000139`. Enter the new total in `hogopt.computeTau.target` using the `loop_timing.mean` from the same timing setup. For example, entering `0.000185` with the default reference adds 46 µs to every loaded per-mode delay. The optimizer uses the adjusted delay for the active controller, SI, and LP. It refreshes the delay-dependent stability and predictor regularization calculations for the new setting.

`computeTau.current` reports the selected total compute latency. Startup and `computeTauReset.request=On` restore the configured reference and remove the correction. INDI changes to `computeTau` are runtime-only. Reloading looptau changes the baseline per-mode delays and keeps the selected compute-time correction.

The configuration and delay regressions passed 86 assertions, including positive and negative offsets in all three controller models and reset to a nondefault reference. The full optimizer suite passed 50 of 52 tests with the same two baseline failures.
