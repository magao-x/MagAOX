# Operational-parameter correspondence audit

2026-11-06; personal arm64 Mac `ASTR-LT-428.local`; offline only. The supplied
configuration excerpt is recorded in `../operational_parameters.ini`; neither
`/opt/MagAOX/config/loPredCtrl.conf` nor an operational binary was accessed.

## What this experiment can establish

**The earlier headless comparison did not reproduce the operational controller.**
Its short history, forgetting, information prior, literal control penalty and
absence of Legacy's added feedback differ. Copying the operational numbers into
the causal controller would still use a different control law.

This experiment implements a **conditional operational-algebra baseline**:
history15, future3, two modes, QR, gamma1, initial information1e-5, Legacy's scaled
regularization100 and additive current-WFS feedback .3. Both predictors retain
the verified corrected causal histories and actual consumed simulated commands.
The underlying QR learner, rank/readiness protections and production applications
are unchanged. It is **not an exact replay of the on-sky application**: historical
indexing, calculated-command histories, signed modal calibration and runtime
integration/learning states cannot be inferred from the excerpt.

```sh
python apps/loPredCtrlAcc/tests/operational_baseline.py
```

This reuses the immutable independent-realization ADC/pointing bank if present,
or creates it with the actual `accelSim` numerical source. The source arrays are
checked against the previously published SHA256 hashes before comparison. The
other repository is read-only. No native application, stream, device or service
is opened. Dependencies are the existing C++20/Eigen/mxlib and Python
NumPy/SciPy/Matplotlib environment; use `python` with those packages installed.

## Parameter-by-parameter correspondence

| Parameter | Previous headless comparison | Checked-out operational meaning | New offline profile / limit |
| --- | --- | --- | --- |
| `history=15` | Hardcoded6 WFS samples and5 previous commands | Number of WFS samples; historical identification has different command indexing |15 WFS samples,14 actual past consumed commands; corrected causal convention |
| `future=3` | Hardcoded3 | Three target frames and three planned command blocks; last newest-first command block is issued | Same3; latency remains a separate verified two frames |
| `num_modes=2` | Hardcoded2 | First two FLOAT modal input values, with no unit/sign calibration in `processImage()` | Two explicitly synthetic tip/tilt coordinates; physical mode identity is not verified |
| `qrd=true` | QR enabled | Selects original QR-RLS identification | Corrected causal QR path retained; original Legacy QR remains unchanged and separately tested |
| `gamma=1` | .9995 | Forgetting factor; QR scales its factor by sqrt(gamma) per update |1, no forgetting |
| `covariance=1e-5` | .001 | Despite its name, initial information rho; QR R0=sqrt(rho)I, conventional RLS P0=I/rho |rho=1e-5, corresponding initial P0=100000I |
| `regularization=100` | Literal lambda=.02 in reference law | Dimensionless multiplier of maxCoeff(CᵀC); not a fitting ridge or literal lambda | Uses Legacy scaled-inverse algebra; no substitution with literal lambda100 |
| `gain=.3` | No separate feedback; calibration probe RMS .3 is unrelated | Adds +.3 times current WFS inside predictive output; does not multiply the predictive term | Explicit added +.3 current-WFS feedback; plant sign is not changed to favor suppression |
| `copy_gain=1` | No predictive-bypass experiment | Original `loPredCtrl` has no such setting; clone declares it but loads it from `parameters.gain` | Inactive with prediction enabled. Existing clone parsing defect characterized, not repaired |
| `fpsSource=camwfs` | Virtual fixed2kHz | Subscription to `camwfs.fps.current`; does not select the measurement stream or establish exposure timing | Fixed2kHz simulation only; no camera/exposure clock equivalence claim |
| `own_shmim=true` | Private in-memory images | Framegrabber output/stream ownership setting | No shared-memory ownership/publication represented |

Sources: original [`ar_controller.cpp`](../../../loPredCtrl/ar_controller.cpp)
constructor, `update_system()`, `update_controller()` (223–240),
`calculate_command()` (242–272); original
[`loPredCtrl.hpp`](../../../loPredCtrl/loPredCtrl.hpp) `setupConfig()`/`loadConfig()`,
`processImage()` (499–639), fps callback (1031 onward); clone
[`loPredCtrlAcc.hpp`](../../loPredCtrlAcc.hpp) `loadConfigImpl()` (766–802),
`configureShadow()` (861–878); [`qrd_rls.cpp`](../../qrd_rls.cpp) constructor and
`update_preallocated()`; original
[`recursive_least_squares.cpp`](../../../loPredCtrl/recursive_least_squares.cpp)
constructor. The separate MagAOX checkout has another older `loPredCtrl` variant;
neither checkout identifies the deployed revision from an INI excerpt alone.

### Control-law and fitting distinctions

In canonical `[A B D C]` coordinates, the offline law adapter reproduces:

```
alpha = regularization * maxCoeff(CᵀC)
K = -(CᵀC + alpha I)^(-1) Cᵀ [A B D]
u(k) = last_two_rows(K) [y_p; u_p; a_p] + gain*y(k)
```

The implementation preserves the original **full Gram matrix and explicit
inverse evaluation order**, reordering to original `[C B D A]` coordinates before
the operation. It does not replace the production inverse with the corrected
reference LDLT solve. With no accelerometer features it uses `[C B A]`.
`ddspc_operational_test.cpp` compares this re-expression against the pinned
original controller's actual outputs on its own histories, including the supplied
parameters. This establishes algebraic correspondence, not equivalent estimators.

There is an important qualification to the label C: the original solve treats
the **leading6 fitted columns** as its control block. At callback n, after saving
the calculated command, original `get_command_future(1)` uses
`offset=skip_cmds*num_modes=2` rows and returns `[u(n-2),u(n-3)]`;
`get_command_past()` returns `[u(n-3)..u(n-17)]`. The first past command is
duplicated. At prediction origin k=n-3 the fitted control prefix is therefore
`[u(k+1),u(k),u(k)]`, whereas the corrected causal model uses
`[u(k+2),u(k+1),u(k)]`. Both standard matrices have shape6×64, but their columns
do **not** have identical physical meanings. This inherited behavior is preserved
in Legacy and its exact tests; it is not reintroduced into causal training to
make the parameter names look equivalent. The new adapter applies Legacy's
algebra to the corrected causal model's physical C.

The causal reference law instead uses **literal** lambda and no added feedback.
The native shadow app currently maps `m_regularizationCtrl` directly to that
lambda and does not use Legacy gain/copy gain. Thus its INI value100 would mean
`CᵀC+100I`, not the Legacy expression above. The distinction is covered by an
analytic scalar regression; test tolerances were not relaxed.

QR's information factor obeys `RᵀR <- gamma*RᵀR + phi*phiᵀ`. Initial rho and
forgetting affect identification; control regularization affects the subsequent
command solve. In this experiment gamma1 retains all7983 calibration updates
and rho stays1e-5. The old .9995 forgets with an approximately1-second memory at
2kHz and exponentially reduces its initial .001 prior. The single-precision
implementations are used directly; no new fitter or tuning is introduced.

### Horizons, latency and matrix dimensions

At decision k, causal state uses `y(k)..y(k-14)`, consumed
`u(k-1)..u(k-14)` and, for assistance, five ADC0/1 feature pairs available by k.
The latest pair is the average of four actual8-kHz UINT16 samples; channels are
decoded with the source's ADC zero, volts-per-bit and sensitivity, not replaced
with ideal acceleration or pointing truth. Source/channel mapping remains that
documented in [channel validation](channel01_validation.md).

Targets are newest-first `[y(k+3), y(k+2), y(k+1)]`, at1.5,1.0,.5ms.
Future controls are `[u(k+2),u(k+1),u(k)]`. The verified plant response is
`y(k)=d(k)+u(k-2)+noise(k)`: a command consumed after decision k first appears
in WFS frame k+2. **Future3 covers this two-frame delay; it does not prescribe
a three-frame delay.** `u(k+2)` has no optical effect within these three targets;
the true synthetic C has rank4/6, and the positive control penalty makes the
command solve well-defined. No physical exposure/settling latency is inferred.

| Quantity | Old standard / assisted H5 | New standard / assisted H5 |
| --- | --- | --- |
| WFS history length |6 /6 |15 /15 |
| Past command length |5 /5 |14 /14 |
| Acceleration length, two channels |0 /5 |0 /5 |
| Theta shape |6×28 /6×38 |6×64 /6×74 |
| State length |22 /32 |58 /68 |
| Full gain shape |6×22 /6×32 |6×58 /6×68 |
| R factor shape |28×28 /38×38 |64×64 /74×74 |

The WFS lookback becomes7ms instead of2.5ms. More features also enlarge the
unchanged square moving rank-check window. A calibration-end unready fit is
retained for **diagnostic scoring only**; the opt-in diagnostic export does not
authorize a correction. All actual decisions continue through native
readiness/data-validity/staleness gates, even with frozen coefficients.

## Fixed paired experiment

Eight existing seeds101,211,307,401,503,601,701,809 and both existing amplitude
cases (reference and100×) reuse identical recorded ADC0/1 and pointing samples.
No-control, standard and assisted H5 receive the same exogenous WFS noise and
probe draws. The no-control final RMS must exactly match the archived short-
history experiment. Calibration0–4s, scoring12–16s, .03 normalized WFS noise,
.3 calibration probe, .005 valid-prediction evaluation probe, ±4 absolute command
bounds and two-frame optical delay are preserved. No fitting or selection uses
the final interval. The associated ADC history is fixed at5 because the
operational excerpt contains no accelerometer setting.

Prediction RMS scores the frozen model's **k+2** forecast on common no-control
WFS records. It may be scored diagnostically when a model is unready; this does
not imply that its correction was available. Closed-loop scores include all
invalid holds and saturation. Combined RMS is per-component two-axis
sqrt(mean(tip²,tilt²)); per-axis RMS is retained in the CSV. Line suppression is
integrated±2Hz Welch power relative to each seed's no-control case. Broadband RMS
integrates5–1000Hz excluding those line bands.

## Results

Apple Clang21.0.0, strict `-O3 -fno-fast-math`; entries below are arithmetic
means of the eight per-realization scores in normalized synthetic WFS units.
Negative suppression means amplification, not successful rejection.

| Source | Controller | k+2 prediction RMS | WFS RMS | Broadband RMS |28Hz suppression |56Hz suppression |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Reference | No control | — |1.003210 |.078704 |0dB |0dB |
| Reference | Standard |.058531 |1.404849 |.106787 |−2.931dB |−2.681dB |
| Reference | Assisted H5 |.056703 |1.404807 |.106858 |−2.931dB |−2.679dB |
|100× | No control | — |1.003210 |.078704 |0dB |0dB |
|100× | Standard |.058531 |1.404857 |.106770 |−2.931dB |−2.682dB |
|100× | Assisted H5 |.037369 |1.404581 |.106816 |−2.930dB |−2.688dB |

At high ADC signal, the paired prediction-RMS improvement averages **35.62%**
(reference3.04%). Yet total closed-loop RMS is approximately40% **higher** than
no control and variance reduction averages approximately **−96%**. Strong control
penalty and the added positive feedback dominate the closed-loop result; this
is not an estimator divergence. The effective scaled penalty is99.71–100.59 in
this unit-response synthetic plant, so scaled versus literal100 happens to be
close here; it need not be close in calibrated instrument units.

| Source | Assisted RMS improvement vs standard | Median | Sample SD | Wins |95% bootstrap interval for mean |
| --- | ---: | ---: | ---: | ---: | --- |
| Reference |+.00143% |−.00138% |.00797pp |2/8 |[−.00318,+.00701]% |
|100× |+.01878% |+.02078% |.01182pp |7/8 |[+.01061,+.02582]% |

These are paired percentages, not ratios of mean RMS. Intervals use the existing
20,000-resample whole-realization bootstrap, seed20261106; eight trials provide
only a small exploratory ensemble. The tiny high-signal advantage does **not**
demonstrate useful suppression: both controllers amplify both lines. No setting
was selected or retuned from these results.

| Seed | No-control RMS | Standard reference | Assisted reference | Standard100× | Assisted100× |100× improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|101 |.604839 |.846525 |.846551 |.846531 |.846285 |+.02909% |
|211 |.627805 |.877804 |.877818 |.877800 |.877811 |−.00128% |
|307 |1.685629 |2.361977 |2.361691 |2.361959 |2.361248 |+.03010% |
|401 |1.155920 |1.619015 |1.618764 |1.619027 |1.618716 |+.01917% |
|503 |1.473194 |2.063536 |2.063560 |2.063577 |2.063115 |+.02238% |
|601 |.473594 |.662196 |.662196 |.662208 |.662019 |+.02843% |
|701 |1.073510 |1.503373 |1.503422 |1.503377 |1.503325 |+.00344% |
|809 |.931190 |1.304362 |1.304455 |1.304377 |1.304129 |+.01896% |

No-control normalized results differ between source scales only at float-rounding
level. [Every per-axis/prediction/spectral/command score](operational_baseline/per_realization.csv),
[paired statistics](operational_baseline/summary.json),
[fit diagnostics](operational_baseline/diagnostics.csv) and
[50-ms time bins,4–16s](operational_baseline/time_rms.csv) retain all cases.
[RMS versus evaluation time](operational_baseline/operational_rms.png)
([PDF](operational_baseline/operational_rms.pdf)) and
[28/56-Hz PSD comparisons](operational_baseline/operational_psd.png)
([PDF](operational_baseline/operational_psd.pdf)) show the median/IQR over seeds;
time-series samples are not treated as independent trials.

All traces are finite; maximum consumed correction over the **entire** experiment
is1.528781 normalized units, below±4, with zero saturation. Evaluation correction
availability is97.66–99.41%. The frozen standard seed307 fit is unready at the
calibration boundary (63/64 in the square moving rank window); it later regains
readiness as the same frozen model is checked against new records. Both source
scales retain this outcome. Diagnostic forecast export does not enable a new
command while unready. The full calibration design ranks are64/64 and74/74;
their normalized conditions span131–307 standard,136–337 reference-assisted,
282–623 high-signal-assisted. Unconstrained always-ready homogeneous pole radii
are .578–.610. This is a bounded simulation, not a physical stability certificate
or proof about gated/saturated operation.

### Comparison with the previous short-history experiment

| Source/controller | Previous prediction RMS | New prediction RMS | Previous closed-loop RMS | New closed-loop RMS |
| --- | ---: | ---: | ---: | ---: |
| Reference standard |.082855 |.058531 |.090847 |1.404849 |
| Reference assisted H5 |.082461 |.056703 |.090601 |1.404807 |
|100× standard |.082855 |.058531 |.090844 |1.404857 |
|100× assisted H5 |.095857 |.037369 |.102622 |1.404581 |

The prior standard/assisted high-signal28-Hz suppression was23.19/20.92dB;
56-Hz15.96/15.77dB. Those results used literal lambda .02 and **no added Legacy
feedback**. The new prediction gain reflects the joint changes to history,
forgetting and information prior, not an isolated history-duration effect.
The closed-loop change additionally includes a much stronger penalty and the
explicit Legacy feedback term. Multiple operational differences were changed
together by design; this is not evidence attributing the outcome to one
parameter or disproving accelerometer information. The unchanged plant sign
and missing integration/calibration evidence prevent an on-sky interpretation.

### Compiler and regression validation

Homebrew GCC15.2.0 replays the **same recorded ADC/pointing arrays** and paired
Python noise/probe draws; it is not a second independent disturbance ensemble.
Mean paired RMS effects are+.00052% reference (4/8 wins) and+.02085% high-signal
(8/8). The sign of effects this small can depend on roundoff/readiness histories;
neither compiler shows useful vibration suppression with this conditional law.

Maximum Clang/GCC coefficient difference is1.60e-4, gain difference1.57e-6,
common-input prediction-RMS difference1.36e-6, and closed-loop RMS difference
2.11e-4. Evaluation availability differs by at most11/8000 frames. Closed-loop
trajectories are not claimed bit-exact across compilers. Existing exact
historical tests remain unchanged and pass independently with each compiler.
[Matched replay differences](operational_baseline/compiler_comparison.json),
[GCC scores](operational_baseline/gcc/per_realization.csv),
[GCC summary](operational_baseline/gcc/summary.json),
[profile](operational_baseline/profile.json) and
[source/compiler/binary manifest](operational_baseline/manifest.json) preserve
the reproducibility record.

```sh
python apps/loPredCtrlAcc/tests/operational_baseline.py --cxx g++-15 \
  --output apps/loPredCtrlAcc/tests/results/operational_baseline/gcc
make -C apps/loPredCtrlAcc -f tests/Makefile.offline check
make -C apps/loPredCtrlAcc -f tests/Makefile.offline CXX=g++-15 \
  BUILD_DIR=build/offline-phase9-gcc check
PYTHONPATH=apps/loPredCtrlAcc/tests python -m unittest \
  simulation_demo_test sim_shmim_test prediction_information_test \
  calibration_generalization_test independent_realizations_test operational_baseline_test -v
```

Both strict suites pass **69 cases**, including the existing66 unchanged
regressions; both compiler simulation suites pass **30 cases** (previous26 plus
four new operational tests). The lifecycle double uses numeric1 for true boolean
fixtures; the INI itself retains the supplied `true` values. New tests check
actual pinned Legacy commands, scaled versus literal penalty, feedback placement,
the real clone's existing copy-gain load,15-sample native histories/exports,
two-frame dynamics and explicit unready diagnostic rejection. Original failing
physical-performance trials are retained; no threshold or numerical tolerance
is weakened. The historical GNU fast-math issue remains separate; production
flags are unchanged. Linux/native real-time equivalence is not established by
this macOS offline work.

Clang **ASan/UBSan also passes all69 cases** without a reported sanitizer error.
[Validation record](operational_baseline/validation.json) retains suite counts,
array/command checks and protected-file checks. The recorded experiment manifests
retain the exact tested source/binary hashes and separately identify a subsequent
three-comment documentation clarification; executable statements did not change.

Affected files for this audit:

- New `tests/operational_law.hpp/.cpp`, `operational_parameters.ini`,
  `operational_baseline.py`, `operational_baseline_test.py`,
  `ddspc_operational_test.cpp`, this report and `results/operational_baseline/`
  metrics, fitted matrices, manifests and plots.
- `tests/Makefile.sim` adds isolated profile flags and law linkage;
  `tests/Makefile.offline` links the tested helper. `sim_bridge.cpp` adds profile
  selection and read-only dimension/fit diagnostics. `simulation_demo.py`,
  `prediction_information.py` and `calibration_generalization.py` remove six-frame
  assumptions from diagnostic allocation/state/scoring while preserving defaults.
- `tests/independent_realizations.py` treats the original bridge binary hash as
  producer provenance, rather than requiring the current controller shim to be
  byte-identical to read existing source recordings. Source header/protocol and
  all recorded array hashes remain checked; this runner additionally compares
  against the archived prior-experiment ADC/pointing hashes. No source recording
  is regenerated or changed in these matched comparisons.
- Tests README and the engineering plan below `Agent Findings and Plan` record
  the audit. Pre-existing local changes and earlier reports remain preserved.

## Interpretation and unresolved operational correspondence

The absolute-command plant intentionally retains its **synthetic positive unit
response**. Legacy's added +.3*y is positive feedback in these coordinates;
this can amplify vibration while remaining bounded. The real signed
WFS-modal-to-DM mapping is unknown. Flipping the plant/feedback sign without a
calibration would fabricate operational equivalence. A poor simulation result
with this law does not diagnose a failure of the working on-sky controller.

The excerpt also omits runtime `predict`, `learn` and `integrate` states. Repository
defaults are all false. This experiment explicitly enables prediction after
calibration, freezes identification afterward, and assumes **integration off /
absolute commands**. The queried operational integration state was not provided;
the default is a labeled assumption, not a measurement. If integration is on,
the app accumulates the calculated delta into `full_command`; the current
synthetic absolute-input model does not faithfully represent that output/history
relationship. A different applied-input convention must be established before
claiming equivalence, rather than silently accumulating this causal law.

Original training saves calculated commands inside `calculate_command()` before
optional app accumulation/publication; it has no actual applied-command readback.
Its multichannel future-command skip/indexing differs from the corrected causal
model. Exact Legacy history quirks remain untouched and separately regressed.
Runtime exploration/learning sequences can also replace configured regularization
with staged values (including100,3,.1); no state/telemetry proving those sequences
is available. The fixed4-second fit here is a controlled comparison, not a
reproduction of a deployed training procedure.

Before further accelerometer optimization, obtain the deployed source/build
revision, saved runtime prediction/learning/integration and regularization states,
actual modal units/sign/map, original measurement stream/exposure timestamps and
command-to-WFS delay. Offline operator-provided recordings are sufficient for
the next audit; no agent instrument access is required. Physical parameter values
must not be invented. No production controller, Legacy defaults, runtime,
calibration, configuration or separate repository was modified.
