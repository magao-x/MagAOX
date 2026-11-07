# QR calibration duration, ADC history and initial information

**Follow-up:** [eight independent vibration realizations](independent_realizations.md)
fixes the four-second protocol and compares standard/H5/H1 without retuning.
Matched-input GCC replays separate compiler rounding from realization variation.
The selected-history instability generalizes: neither assisted model establishes
a total-RMS advantage. Favorable secondary spectral/noise effects are reported.
This earlier duration/selection experiment remains historical evidence.

2026-11-06, personal arm64 Mac `ASTR-LT-428.local`; Apple Clang21.0.0 and
Homebrew GCC15.2.0, strict floating point. All results are simulated. No instrument,
native application, real stream, production configuration, controller law or
other-worktree changes; no commit/push.

## Result

**Longer calibration improves assisted generalization, but these changes do not
establish a consistent accelerometer advantage over standard DDSPC.** Shorter
history improves conditioning; the validation-selected one-sample history fails
to beat standard on the final high-SNR interval. The modest QR-prior change has
negligible effect. All unfavorable configurations and final results are retained.

At100× source amplitude, increasing calibration from1 to4 seconds reduces assisted
final prediction RMS **27.20% Clang /31.64% GCC**, and closed-loop RMS
**25.18% /33.72%**. However, the four-second/five-history model beats standard's
total residual RMS by3.29% on the Clang realization and **loses3.51%** on the GCC
realization. Its 28/56-Hz line suppression is worse than standard in both.
Lower overall RMS must not be presented as better suppression of those peaks.

One command reproduces the preregistered candidates, selection, final test,
conditioning, matrices and plots:

```sh
python apps/loPredCtrlAcc/tests/calibration_generalization.py
```

Existing dependencies: Python NumPy/SciPy/Matplotlib, Eigen/mxlib and a C++20
compiler. This advances a virtual clock, not a real-time qualification.

## Protocol and reserved evaluation

- Three durations with **the same calibration end**: windows3–4s, 2–4s, 0–4s.
  Keeping the end fixed avoids confounding duration with recency. All cases see
  identical independent .3-RMS synthetic correction excitation throughout0–4s;
  only their declared windows update coefficients. Complete training examples
  stay within that window: first update is window-start+8 frames, last is7999.
- Validation is4.5–6.5s. Choices are locked to `selection.json` before **any final
  samples are generated**. Final evaluation is the later12–16s interval. The
  half-second after calibration and the validation/final gap are not scored.
- Six assisted candidates, no Cartesian sweep:1/2/4s with existing five-sample
  ADC history and priorρ=.001; at4s only, histories3 and1, and H5 withρ=.01.
  Three independently trained standard baselines use the same1/2/4s windows.
- WFS history6, applied-command history5, future horizon3, forgettingγ=.9995,
  control penaltyλ=.02, rank threshold1e−5, source seed42, calibration seed100,
  WFS-noise seed303, optical sign, two-frame delay and command bounds unchanged.
- Both actual noisy/quantized ADC channels0/1 are decoded and four past raw pairs
  averaged per WFS frame, exactly as before. H5/H3/H1 select5/3/1 of these
  **synchronized averaged pairs**, not5/3/1 raw ADC samples. WFS history and
  sample/availability times do not move. No ideal acceleration enters DDSPC.
- Select by pooled validation two-frame forecast MSE across reference and100×
  amplitudes in their declared normalized units. Standard and existing-H5
  baselines select their duration by the same rule; the assisted candidate can
  select any of the six settings. Lexical ID is the predefined tie break.
  Both compilers select **standard4s, existing assisted4s/H5, selected4s/H1**.
- All candidates are evaluated on the same final interval **after selection**;
  selection is never revised from these outcomes. Replay prefixes, coefficients
  and gains must match exactly between validation-only and final-length runs.
  Native update counts are1992/3992/7992, confirming the window boundaries.

Forecasters are actual ready native QR snapshots. Held-out prediction uses the
same no-control WFS records and known past consumed corrections. Training RMS
below is the same zero-future-command counterfactual score (future *actual*
correction effects removed from targets only); conditional training error with
actual future inputs is also saved. Future labels/commands never enter forecast
state. Closed-loop RMS includes every final frame, including invalid-prediction
holds. No rank or age gate is weakened.

## Duration and feature results

Clang100×, per-component two-axis normalized WFS RMS. `Train` uses only that fit's
calibration window; `Val` and `Final` are disjoint common intervals above.

| Calibration / model | ADC history | ρ | Train k+2 | Val k+2 | Final k+2 | Final WFS RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1s standard | 0 | .001 | .056554 | .065897 | .062382 | .061670 |
| 1s assisted | 5 | .001 | .047698 | .089266 | .075138 | .075215 |
| 2s standard | 0 | .001 | .059240 | .057976 | .056370 | .058314 |
| 2s assisted | 5 | .001 | .055419 | .062658 | .054842 | .056446 |
| 4s standard | 0 | .001 | .058743 | .057809 | .056488 | .058191 |
| 4s existing assisted | 5 | .001 | .055533 | .061219 | .054701 | .056278 |
| 4s assisted | 3 | .001 | .055535 | .060620 | .054602 | .056119 |
| **4s validation-selected** | **1** | **.001** | .057870 | .059956 | .056347 | .058702 |
| 4s assisted, prior adjustment | 5 | .01 | .055534 | .061220 | .054702 | .056284 |

The one-second assisted fit achieves the lowest training error and the worst
generalization. More calibration sacrifices that in-sample advantage and improves
held-out forecasting. Five→three ADC history samples slightly improves final RMS
over H5 (Clang .28%, GCC1.92% at100×), but **it was not the validation winner**.
It remains an exploratory final observation, not a reselected preferred model.
One sample removes much of D's useful dynamic contribution and does not generalize
as well as the validation result suggested.

Final locked comparison at100×; suppression is relative to the same no-control
realization. No-control final RMS is .663809 Clang / .772698 GCC.

| Realization / locked role | Final k+2 | Final WFS RMS | RMS reduction vs standard | 28 /56 Hz suppression dB |
| --- | ---: | ---: | ---: | ---: |
| Clang standard | .056488 | .058191 | — | 32.79 /17.64 |
| Clang existing assisted H5 | .054701 | .056278 | +3.29% | 31.57 /14.29 |
| Clang selected H1 | .056347 | .058702 | **−.88%** | 32.33 /16.43 |
| GCC standard | .057658 | .060530 | — | 31.03 /17.20 |
| GCC existing assisted H5 | .060271 | .062653 | **−3.51%** | 29.29 /10.35 |
| GCC selected H1 | .058106 | .062107 | **−2.61%** | 30.08 /14.90 |

Reference-amplitude final residuals (standard/H5/selectedH1) are
.058193/.058469/.058221 Clang and .060556/.060812/.060450 GCC. Selected-model RMS
reductions are −.05% /+.17%; no consistent reference-noise advantage. Full
training/validation/final prediction, residual, PSD and coefficient measurements
for **every** configuration and amplitude are in the linked CSVs. Compiler
standard libraries produce different normal samples despite the same seed;
these are two library-dependent deterministic realizations, not evidence of a
GCC numerical regression. Within each run all candidates use identical source,
ADC, WFS noise and actually consumed calibration commands.

## Rank, conditioning, magnitudes and regularization

All calibration models have full native readiness rank and full diagnostic design
rank. Native rank uses the unchanged column-normalized last-p-row QR criterion;
the full weighted design's SVD rank/condition are diagnostics, not substituted
readiness tests. Reported condition is the **design** condition, not its square
from normal equations.

Clang100× at4s:

| Model | Rank / features | Weighted scaled condition | ‖Θ‖F | Maximum coefficient | ‖D‖F |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | 28/28 | 129.8 | 5.891 | 1.545 | 0 |
| Assisted H5 | 38/38 | 324.9 | 5.412 | 1.415 | .646 |
| Assisted H3 | 34/34 | 291.7 | 5.447 | 1.417 | .806 |
| Selected H1 | 30/30 | 139.9 | 5.927 | 1.559 | .092 |
| H5, ρ=.01 | 38/38 | 324.9 | 5.411 | 1.415 | .646 |

At1/2/4s the standard conditions are128.8/124.6/129.8 versus assisted
295.8/314.9/324.9. ADC/WFS weighted column cosines reach .940 Clang / .952 GCC
at100×. Added histories are correlated and increase sensitivity of fitted
parameters, but calibration ends at full rank with bounded coefficients and no
evidence of QR arithmetic failure. Subsequent readiness protections remain active.
Better conditioning of H1 does not guarantee better
forecast/control performance. A/B/C norms, maximum coefficient and all durations
are retained in `metrics.csv`.

`QRDRecursiveLeastSquares` already initializes `R=sqrt(ρ) I` and maintains
`RᵀR=γⁿρI+Σγⁿ⁻ʲφjφjᵀ`. Increasingρ from.001 to.01 tests the existing initial
information regularization, **not** the control penaltyλ. Both models keep the
same QR update/triangular invariant. After7992 updates the initial information
terms are only1.84e−5 /1.84e−4. Final residual changes are ~6e−6 Clang /2e−6 GCC,
slightly unfavorable here. This does not establish that other regularizers would
fail; none was introduced or swept.

Withγ=.9995 at2kHz, the e-fold memory is about1s. Effective weighted record counts
approach1963 for the4s fit, not8000 equally weighted observations. Longer fitting
adds phase coverage, but forgetting still limits retained information. The
remaining limitation is validation-to-later-phase generalization of the existing
correlated multi-input predictor; simply shortening D or altering its weak initial
prior does not solve it. No timing change or production recommendation follows.

## Evidence and verification

- [Protocol](calibration_generalization/protocol.json),
  [locked choices/scores](calibration_generalization/selection.json),
  [all Clang metrics](calibration_generalization/metrics.csv),
  [manifest](calibration_generalization/manifest.json),
  [verification record](calibration_generalization/validation.json).
- [Final RMS versus time](calibration_generalization/final_rms.png)
  ([PDF](calibration_generalization/final_rms.pdf));
  [28/56-Hz PSDs](calibration_generalization/final_psd.png)
  ([PDF](calibration_generalization/final_psd.pdf)).
- [GCC metrics](calibration_generalization/gcc_metrics.csv),
  [selection](calibration_generalization/gcc_selection.json),
  [manifest](calibration_generalization/gcc_manifest.json),
  [RMS](calibration_generalization/gcc_final_rms.png),
  [PSD](calibration_generalization/gcc_final_psd.png).
- [Default fixed-fit preservation](calibration_generalization/default_preservation.json):
  all28 stored columns of the original100× standard/assisted10400-frame traces
  reproduce at their saved12-significant-digit precision under both compilers.
- **21 focused tests pass per compiler**: prior18 plus shorter-history native
  dimensions/reference-command agreement, bit-exact standard coefficients and
  complete traces across H1/H3/H5 builds, and pooled validation-only selection.
- **Both unchanged66-case strict suites pass**, preserving historical exact
  Legacy/causal tests and tolerances. No sanitizer/fast-math investigation was
  added; earlier findings remain documented. Native Linux/real-time behavior is
  not qualified by these Mac runs.
- All validation prefixes and model coefficients/gains exactly match their
  final-length replay. The selection bytes remain unchanged. Input consumption
  equals requests, optical correction is delayed exactly two frames, and final
  clipped fraction is zero in all cases. Maximum scored command is3.082 Clang /
  2.985 GCC versus±4; prediction availability remains≈99%, with holds included.

GNU repeat and tests (after building the generalization libraries):

```sh
python apps/loPredCtrlAcc/tests/calibration_generalization.py --cxx g++-15 \
  --output apps/loPredCtrlAcc/build/calibration-endaligned-gcc
python -m unittest discover -s apps/loPredCtrlAcc/tests -p calibration_generalization_test.py -v
GENERALIZATION_CXX=g++-15 python -m unittest discover \
  -s apps/loPredCtrlAcc/tests -p calibration_generalization_test.py -v
```

The prior18-case/default builds and66-case commands remain in the tests README.
New experiment build names encode settings and the end-aligned protocol to avoid
reusing objects compiled with different macros. `TRAINING_START/END` only gates
the existing offline manager's training control; `ACCEL_HISTORY/QR_PRIOR` change
only the simulation bridge's configuration. Default parameters stay unchanged.

Affected files: test `Makefile.sim`, `sim_bridge.cpp`, `async_shadow.cpp`,
`simulation_demo.py`, `prediction_information.py`; new
`calibration_generalization.py` / `calibration_generalization_test.py`; README,
prior prediction-report follow-up link, engineering plan below Agent Findings,
this report and artifacts. No QR/RLS algorithm, production controller, original
`loPredCtrl`, Legacy default or other-repository source is modified.

The next useful experiment is independent phase/seed validation of this **fixed
small candidate set**, with a fresh held-out interval. Do not promote H3 after
seeing this final table, increase the sweep, or proceed to instrument integration
on the strength of one favorable total-RMS number.
