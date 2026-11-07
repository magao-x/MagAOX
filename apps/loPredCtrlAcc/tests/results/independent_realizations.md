# Eight independent vibration realizations: fixed QR DDSPC

2026-11-06; personal arm64 Mac `ASTR-LT-428.local`; Apple Clang21.0.0 and
Homebrew GCC15.2.0, strict floating point. Simulation only: no native app,
shared-memory image, instrument, real DM command or production configuration.

## Recommendation

**The current H5 and H1 assisted configurations provide no reproducible total
closed-loop WFS-RMS advantage over standard DDSPC across these realizations.**
At high ADC signal amplitude each wins only1/8 trials and increases mean residual
RMS. At reference amplitude both win4/8; their mean-improvement intervals include
zero. Neither is recommended over standard for the next simulation integration.

There are reproducible **secondary tradeoffs**: reference H5 improves the56-Hz
line slightly in8/8 trials, and both high-signal assisted models reduce broadband
residual noise in8/8. Those gains do not establish better overall vibration
suppression: dominant28-Hz rejection and total RMS generally deteriorate at high
signal. This is category3 for the primary total-RMS objective, with limited
category2 benefits in individual spectral components.

The smallest scientifically justified next test is **balanced multi-realization
calibration using the existing QR estimator**, withholding entire new realizations
for evaluation. Interleave complete causal training examples from independent
calibration episodes soγ=.9995 does not retain only the last episode. This is a
calibration-data change, not a new controller or fitting law. Keep28/56-Hz
suppression as explicit acceptance metrics so broadband-noise reductions cannot
mask poorer vibration cancellation. This follow-up is recommended, not implemented.

## Fixed protocol and reproducibility

```sh
python apps/loPredCtrlAcc/tests/independent_realizations.py
```

This builds/reuses the existing two four-second QR simulation libraries. It
generates a reusable source bank under `apps/loPredCtrlAcc/build/independent-realization-source`
and writes reports/plots under `tests/results/independent_realizations/`.
Dependencies are unchanged: Python NumPy/SciPy/Matplotlib and Eigen/mxlib/C++20.

- Eight predefined seeds: **101,211,307,401,503,601,701,809**, distinct from the
  earlier seed42 experiments. Actual `accelSimDetail::model::restart()` draws
  seeded initial AR(2) states; the same generator subsequently supplies modal
  innovations and sensor noise. Frequencies28/56Hz, damping,8-kHz sampling and
  UINT16 ADC encoding stay fixed. No new waveform generator or phase rotation.
- Initial modal phases differ: the28-Hz X estimates from the first250ms are
  106.01°, −19.44°, −170.02°, 96.56°, −91.28°, 49.75°, 15.83°, −63.96°.
  All four modal/axis phase estimates and raw ADC/pointing hashes are recorded in
  the [source manifest](independent_realizations/source_manifest.json). Phase
  estimation uses plant truth **only for audit**, never as a controller input.
- Four paired trajectories per seed: no predictive control, independent standard,
  assisted H5, assisted H1. Only generated ADC0/1 are used, decoded/averaged exactly
  as before; H1/H5 mean1/5 synchronized four-raw-sample averaged pairs. No ideal
  acceleration substitutes for ADC values.
- Reference and100× source amplitudes repeat the previous two sensor-SNR cases.
  ADC calibration/noise stay unchanged. Normalized WFS noise stays .03; probe RMS
  .3 during calibration and the existing .005 evaluation probe stay unchanged.
  These are synthetic scale cases, not instrument calibrations or separate sets
  of16 independent vibrations. No per-realization amplitude normalization.
- All models train only0–4s, with7992 updates and fixed QR,ρ=.001,γ=.9995,
  λ=.02, WFS history6, past command history5, future horizon3, two-frame DM delay,
  rank threshold1e−5 and±4 normalized command bounds. Coefficients then freeze;
  normal readiness/validity/age checks continue. Evaluation is **12–16s** in every
  trial. No validation selection, per-seed tuning, new recovery or parameter sweep.
- Calibration/probe seed=`10000+vibration_seed`; WFS-noise seed=`20000+vibration_seed`.
  Every controller within a trial receives identical exogenous source/noise/probe
  draws. Initial four-second WFS records match exactly. Disturbance-channel traces
  match across the entire run; corrective inputs remain separate and actually
  consumed commands train the models. WFS includes their delayed optical response.

Held-out prediction is actual frozen-model k+2 forecasting on the identical
no-control WFS records, with no future correction in the input. No-control has no
predictor and its prediction field is empty. Closed-loop RMS includes invalid
prediction holds and any saturation. Reported RMS is per-component two-axis
`sqrt(mean(tip²,tilt²))`; separate tip/tilt RMS is saved for every case.

## Paired statistical comparison

Improvement is computed **within each realization** as
`100 × (1 − assisted_RMS / standard_RMS)`. Positive means assistance helps.
Means/medians below summarize these eight percentages, not pooled WFS frames or
a ratio of pooled RMS. SD is the sample SD (percentage points).

| Source | Model | Mean improvement | Median | SD | Wins | 95% interval for mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Reference | H5 | +.25% | −.05% | 1.38 | 4/8 | [−.51,+1.22]% |
| Reference | H1 | +.05% | −.05% | .54 | 4/8 | [−.28,+.42]% |
| 100× | H5 | **−13.68%** | −14.82% | 13.73 | 1/8 | [−21.93,−4.36]% |
| 100× | H1 | **−5.98%** | −6.07% | 3.02 | 1/8 | [−7.68,−3.83]% |

Intervals are descriptive95% percentile-bootstrap intervals for the mean, from
20,000 resamples of **whole paired realizations**, seed20261106. Correlated frames,
axes, amplitude repeats and compiler replays are not additional independent
samples. With only8 seeds these intervals are exploratory, conditional on this
synthetic model/protocol, and not adjusted for multiple metrics/configurations.
They are not on-sky confidence or a guarantee for other disturbances.

High-signal paired final WFS RMS and percentage improvements, Clang:

| Seed | No control | Standard | H5 | H1 | H5 improvement | H1 improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | .604839 | .076210 | .084594 | .082604 | −11.00% | −8.39% |
| 211 | .627805 | .060884 | .061021 | .064656 | −.23% | −6.20% |
| 307 | 1.685629 | .126225 | .157673 | .135798 | −24.91% | −7.58% |
| 401 | 1.155920 | .075062 | .093375 | .079518 | −24.40% | −5.94% |
| 503 | 1.473194 | .134917 | .120057 | .134053 | +11.01% | +.64% |
| 601 | .473594 | .072024 | .083579 | .075926 | −16.04% | −5.42% |
| 701 | 1.073510 | .093967 | .106744 | .102710 | −13.60% | −9.30% |
| 809 | .931190 | .087459 | .113932 | .092424 | −30.27% | −5.68% |

Reference paired improvements (H5/H1):101 −.46/−.27%,211 +.29/+.10%,307 +.32/−.69%,
401 +3.24/+1.08%,503 +.90/−.13%,601 −.67/−.22%,701 −.38/+.02%,809 −1.23/+.50%.
[Per-realization metrics](independent_realizations/per_realization.csv) contain
every no-control/standard/H5/H1 prediction, tip/tilt/combined RMS, PSD, broadband,
command and saturation result; [paired effects](independent_realizations/paired.csv)
and [full statistics](independent_realizations/summary.json) preserve all cases.

## Prediction, spectral and temporal tradeoffs

Broadband RMS is the square root of integrated Welch PSD over **5–1000Hz excluding
±2Hz around28/56Hz**. It includes WFS noise and stochastic mode wings, not a new
injected broadband disturbance. No signal is filtered in the controller.
Line suppression is integrated±2Hz power relative to each trial's own no-control
case; positive *advantage* below means more suppression than standard.

| Source/model | Mean held-out prediction RMS improvement | Mean broadband RMS improvement | Mean28-Hz advantage / wins | Mean56-Hz advantage / wins |
| --- | ---: | ---: | ---: | ---: |
| Reference H5 | +.52% | +.22% | +.05dB /2 | +.29dB /8 |
| Reference H1 | +.12% | −.18% | +.02dB /2 | +.05dB /6 |
| 100× H5 | **−15.66%** | **+9.80%** | **−2.28dB /1** | −.19dB /3 |
| 100× H1 | **−5.54%** | **+1.72%** | −.63dB /2 | **−2.57dB /0** |

Wins are out of8. High-signal broadband improves8/8 for both histories. Reference
H5's small56-Hz advantage is consistent but secondary; it does not yield a total
RMS advantage. Mean high-signal absolute suppression (standard/H5/H1) is
23.19/20.92/22.56dB at28Hz and15.96/15.77/13.39dB at56Hz.

The high-signal H5 training prediction gain averages4.95%, while held-out prediction
loses15.66%. Calibration ranks are full; H5 scaled design conditions span258–481
versus H1's95–211. Together with prior phase diagnostics this supports a
generalization/coverage limitation of the correlated multi-input fit. High ADC
coherence or a better-conditioned H1 model does not reliably predict a closed-loop
win. The isolated seed503 gain is not used to invent a favorable subgroup or
change parameters. No reliable calibration-based rule identifying winners is
established by eight trials.

- [Paired improvements and mean intervals](independent_realizations/paired_improvement.png)
  ([PDF](independent_realizations/paired_improvement.pdf)).
- [Reference RMS versus time](independent_realizations/rms_scale_1.png),
  [high-signal RMS versus time](independent_realizations/rms_scale_100.png): all8
  seeds, all4 controllers, full4–16s post-calibration period; evaluation shaded.
  [Time-bin data](independent_realizations/time_rms.csv) preserve all50-ms bins.
- [Residual PSD comparison](independent_realizations/residual_psd.png)
  ([PDF](independent_realizations/residual_psd.pdf)): median/IQR across seeds,
  not confidence bands. Individual spectra are in `spectra.npz`; individual
  integrated line scores, not integrals of median spectra, feed the statistics.
- [Calibration diagnostics](independent_realizations/conditioning.csv) preserve
  rank, coefficients, effective samples and measured ADC/WFS coherence per trial.

## Compiler replay and safety checks

GCC consumes the **same recorded Clang-produced ADC/pointing/unit arrays**, with
the same Python WFS/probe draws. Source-bank SHA256 is identical. Thus its result
is a numerical replay, not8 more independent vibrations:

| Source/model | Clang mean RMS improvement | GCC mean | Wins in both |
| --- | ---: | ---: | ---: |
| Reference H5 | +.251% | +.248% | 4/8 |
| Reference H1 | +.051% | +.050% | 4/8 |
| 100× H5 | −13.679% | −13.722% | 1/8 |
| 100× H1 | −5.983% | −5.999% | 1/8 |

Maximum calibration coefficient/gain differences are1.92e−4/1.86e−4. Common-input
held-out forecast-RMS differences are≤2.10e−6. Maximum closed-loop RMS difference
is2.80e−4 (seed503/high-signal/H5); availability counts differ by at most11/8000
evaluation frames. Feedback histories/readiness decisions can therefore differ;
cross-compiler closed-loop traces are not claimed bit-exact. Maximum difference
in a paired improvement is .226 percentage points, much smaller than the13.73-pp
between-realization SD for high-signal H5. Conclusions/win counts do not change.
The unchanged strict historical exact tests pass separately under each compiler;
no tolerance or production flag was changed.

- **26 focused cases per compiler pass**: prior21 plus seeded source/phase replay,
  unchanged default seed behavior, independent noise/probe pairing, analytic
  broadband power, per-trial bootstrap/SD and non-pooled paired comparisons.
- **66 strict cases per compiler pass**, including historical exact Legacy/causal
  regressions. No new sanitizer/fast-math investigation; earlier results remain
  documented. No Linux or real-time readiness claim from these Mac simulations.
- No ADC rails or command saturation in any trial. Maximum command is3.701
  normalized units versus±4, including the entire run. Prediction availability
  spans97.66–99.38%; invalid predictions hold the prior command and all such
  frames stay in scores. Requested/consumed commands agree exactly and corrective
  optical states match the two-frame delay. No histories use calculated commands
  as a substitute for the simulator's actual consumption.

Matched-input GNU reproduction, after the first command creates the shared bank:

```sh
python apps/loPredCtrlAcc/tests/independent_realizations.py --cxx g++-15 \
  --bank apps/loPredCtrlAcc/build/independent-realization-source \
  --output apps/loPredCtrlAcc/build/independent-realizations-gcc
PYTHONPATH=apps/loPredCtrlAcc/tests python -m unittest \
  simulation_demo_test sim_shmim_test prediction_information_test \
  calibration_generalization_test independent_realizations_test -v
```

Set `GENERALIZATION_CXX=g++-15`, `DDSPC_SIM_LIB` and `DDSPC_SIM_FIXED_LIB` to the
existing GNU library paths for the GNU focused-test repeat, as in the earlier
README. [GCC metrics](independent_realizations/gcc_per_realization.csv),
[statistics](independent_realizations/gcc_summary.json),
[matched compiler differences](independent_realizations/compiler_comparison.json),
[protocol](independent_realizations/protocol.json),
[manifest](independent_realizations/manifest.json) and
[verification](independent_realizations/validation.json) retain the evidence.

Changed files: `tests/simulation_demo.py` adds optional source/WFS/probe seeds
with unchanged defaults; new `independent_realizations.py` and its tests; tests
README, previous report follow-up link, plan below Agent Findings, this report
and artifacts. QR/RLS/controller algorithms, operational Legacy/default, original
`loPredCtrl`, production apps/configurations, and the separate worktree remain
unchanged. No instrument access, commit, push, merge or deployment.
