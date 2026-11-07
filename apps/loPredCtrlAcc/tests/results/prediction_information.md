# Fixed-fit ADC0/1 prediction and correction audit

**Follow-up:** [small duration/history/prior experiment](calibration_generalization.md)
uses end-aligned calibration windows, validation-only selection and a later
reserved final interval. Longer calibration improves assisted generalization;
the validation-selected shorter history still fails to beat standard consistently.
The original audit and unfavorable cases below remain historical evidence.

2026-11-06; personal arm64 Mac `ASTR-LT-428.local`; Apple Clang21.0.0 and
Homebrew GCC15.2.0, strict floating point. Offline only; no native applications,
shared memory, hardware, services, production configurations, or other-worktree
edits. Prior experiments and unfavorable cases are preserved.

## Finding

**The stabilized assisted model principally underperforms because its held-out
disturbance forecast is worse before any correction is applied.** High ADC/WFS
peak coherence is present; there is no detected timing/indexing bug. At 100×,
adding ADC histories reduces calibration error but increases held-out error in
both compiler realizations. The error is concentrated at the dominant 28-Hz
line. Thus this is a calibration/generalization limitation, rather than the
control solve turning a clearly better forecast into a worse correction.

The diagnostic localizes a mechanism: a short calibration of almost monochromatic
two-axis motion permits cross-axis phase/amplitude relationships to be fitted
that do not persist as the stochastic realization evolves. The larger assisted
fit improves its in-sample 56-Hz fit while worsening the dominant held-out 28-Hz
response. Finite calibration coverage / correlated features is the supported
explanation, not a proof that this is the only statistical bias present. Source
parameters stay fixed; the realization's envelopes and relative phases do not.
At reference amplitude quantization/noise further limit incremental information.

**No supported timing correction or controller improvement was found.** No
alternative timing, new controller, tuning sweep, recovery policy or production
behavior change was introduced. Longer/more diverse calibration would be a
separate experiment, not an undocumented fix in this comparison.

Reproduce all three trajectories, actual native fits, prediction/coherence
diagnostics and plots:

```sh
python apps/loPredCtrlAcc/tests/prediction_information.py
```

The existing headless source, plant and fixed-calibration QR controller are
reused. `sim_bridge.cpp::ddspc_sim_copy_model()` copies an actual coherent ready
snapshot after frame2399. An optional output dictionary in `simulate()` receives
that copy; it never replaces any coefficient. Normal control operations and
default adaptation remain unchanged.

## Forecast and closed-loop results

No-control WFS RMS is **1.242141** at either explicitly normalized amplitude.
Prediction is scored on the identical no-control records with all future
corrective commands set to zero, at the **two-frame / 1-ms actuation horizon**.
Models receive measured WFS, actual consumed past corrections, and actual decoded
ADC0/1 only. No generator pointing or ideal acceleration enters prediction.
Values are per-component two-axis RMS in the existing normalized WFS units.

| Source | Fixed model | Common open-loop forecast RMS, k+2 | Closed-loop residual RMS | 28 / 56-Hz suppression dB |
| --- | --- | ---: | ---: | ---: |
| Reference | Standard | .100793 | .108602 | 22.30 / 16.00 |
| Reference | Assisted ADC0/1 | .101428 | .109891 | 22.17 / 16.74 |
| 100× | Standard | .100793 | .108533 | 22.31 / 16.03 |
| 100× | Assisted ADC0/1 | .111269 | .117588 | 21.06 / 19.51 |

Assistance increases closed-loop RMS **1.19% / 8.34%**. At 100× its calibration
k+2 error is .046127 versus standard .052765 (**12.58% lower**), but held-out
error is **10.39% higher**. Held-out k+2 forecast-error power in the ±2-Hz bands:
28 Hz .006755 standard → .009302 assisted; 56 Hz .000276 → .000247. The much
larger 28-Hz loss dominates the smaller 56-Hz benefit.

GCC repeats: reference open-loop .064593/.064561, closed-loop .075689/.075881;
100× open-loop .064593/.069106, closed-loop .075602/.078439 (standard/assisted).
The tiny reference forecast advantage (**.05%**) does not survive closed-loop
correction; the main high-SNR forecast loss occurs with both compilers. C++
standard-normal realizations differ across standard libraries; comparisons within
each run use exactly identical disturbances, ADC and noise, not cross-compiler
RMS equality.

Clang/GCC independent double discounted batch fits reproduce the poor native
forecasts (k+2 RMS discrepancy below 1e−6). At Clang100×, coefficient maximum
differences are 7.1e−6 / 4.23e−5; column-scaled calibration design condition is
120 / 304 (standard/assisted). This excludes float QR roundoff as an explanation
for the ~.0105 forecast-RMS gap. No rank criterion or numerical tolerance changes.

## Coherence, information and timing

Same-axis magnitude-squared coherence and WFS-minus-ADC phase at the nearest
Welch bins (28.3203 / 55.6641 Hz; 2048-point Hann, 50% overlap, whole no-control
scored interval):

| Source / ADC axis | 28-Hz coherence / phase | 56-Hz coherence / phase |
| --- | ---: | ---: |
| Reference / 0→X | .99624 / 170.48° | .99250 / 163.42° |
| Reference / 1→Y | .99914 / 171.05° | .96417 / 157.23° |
| 100× / 0→X | .99999 / 171.79° | .99983 / 163.71° |
| 100× / 1→Y | .99999 / 171.79° | .99799 / 163.56° |

These are **decoded generated ADC words**, including sensor noise/quantization,
not ideal acceleration. Full cross-axis coherence/phase is also saved. High
narrowband coherence can coexist with low broadband sensor SNR: most reference
ADC noise is outside these peaks. Peak coherence alone does not establish
incremental information beyond an already predictive WFS history.

The timing audit is consistent with the observed phase:

- State at k: `[y_k … y_(k−5), u_(k−1) … u_(k−5), a_k … a_(k−4)]`;
  assisted training has 38 features, standard28. Targets are newest-first
  `[y_(k+3), y_(k+2), y_(k+1)]`; future commands `[u_(k+2), u_(k+1), u_k]`.
- ADC frame k averages four samples at decision−500/−375/−250/−125 μs,
  available1μs later. All precede the decision; no sensor shift from the future.
  The latest same-source pointing is consumed by the disturbance branch and
  becomes WFS disturbance at k+2, just as the corrective decision at k does.
- Averaging adds187.5μs relative to the latest source sample. With the negative
  acceleration/position sign, the expected WFS-minus-ADC line phase is
  `180°−360° f(.001−.0001875)`, or **171.81° / 163.62°**. The high-SNR measurements
  agree; an integer-frame offset correction would introduce an error.
- F=3 covers the two-frame response. `gain.bottomRows(2)` selects u_k. The
  canonical C has identity blocks (rows0:2, cols2:4) and (rows2:4, cols4:6).
  Historical commands are the actual values read by aoSim after the decision.
  Existing impulse/cancellation and consumed-input tests pass; no sign/index fix.
- On closed-loop trajectories, forecast scoring removes **actual future command
  effects from the scoring target only**, leaving known pending u_(k−1) at k+1.
  Future commands never enter a zero-future forecast. The assisted high-SNR k+2
  error remains worse on either controller's common state sequence. A known-C
  solve diagnostic changes commands by only ~.0024 standard/.0049 assisted RMS
  on their own states; its imperfect C estimate is secondary to forecast error.

### Does ADC contain additional information?

At reference amplitude there is no consistent held-out information gain. At100×,
there is measurable potential: after conditioning on WFS/consumed-command history,
descriptive partial R² is .239/.078 on the two axes (Clang). This same-window
statistic is optimistic and is **not** a validated controller improvement.

A single **statistical information probe**, separate from the controllers, fits
nested linear forecasts on the earlier half of the common open-loop holdout and
scores the later half. Fit targets end at6896; test origins start6898. With identical
causal states, adding only the ten ADC history features changes RMS
.049131 → .046906 (**4.53% lower**, Clang) and .049736 → .047987 (**3.52% lower**,
GCC). Reference RMS reductions are −.31% / +.15%, respectively. This uses measured WFS
targets and ADC, never generator truth. It is a post-hoc information test on a
different split, **not an actuated model, a readiness bypass, or an improvement
to the fixed native controller**. It establishes that high-SNR sensor information
exists but the original calibration fit does not reliably exploit it.

For the fixed100× model, measured 28-Hz Y/X amplitude ratio changes from .419 in
calibration to1.357 in the last second; relative phase changes −85.30°→−53.16°.
The learned normalized off-axis forecast response is .106 standard/.115 assisted,
although the declared synthetic optical/sensor mapping is axis-diagonal. Applying
the fitted frequency responses to the measured phasors predicts late relative
28-Hz errors8.55% /10.01%, versus .47% /.56% in calibration. These are post-hoc
frequency-response diagnostics using measured ADC/WFS ratios, not new control
inputs. They support calibration-specific cross-axis coupling as a mechanism
for the generalization loss. There is no instrument calibration inferred here.

## Artifacts and validation

- [Forecast error and residual RMS versus time](prediction_information/prediction_and_residual.png)
  ([PDF](prediction_information/prediction_and_residual.pdf)).
- [Residual PSD zooms at28/56Hz](prediction_information/peak_psd.png)
  ([PDF](prediction_information/peak_psd.pdf)).
- [Reference full time series](prediction_information/scale_1/full_time_series.png),
  [100× full time series](prediction_information/scale_100/full_time_series.png).
- [Complete metrics](prediction_information/metrics.json), including horizons1/2/3,
  crossed state-sequence scoring, partial-information probe, frequency responses
  and source phase evolution; [source/binary/compiler manifest](prediction_information/manifest.json).
  [GCC metrics](prediction_information/gcc_metrics.json) and
  [manifest](prediction_information/gcc_manifest.json) preserve the second realization.
  [Validation record](prediction_information/validation.json) includes test-log
  hashes and exact preservation of all previous fixed-fit residual metrics.

Four new tests verify read-only ready-model export/buffer sizes, native command
and reference-solve agreement, the two-frame counterfactual target and causal
history, and coherence/phase sign against independently quantized sinusoids.
Together with the existing14 connection/transport tests, **18 pass on each
compiler**. Unchanged strict suites pass **66 cases**, Clang1,766,593 assertions,
GCC1,764,508 in this run (concurrency assertion totals vary with scheduling).
Original exact-regression tolerances remain unchanged. No sanitizer
or GNU fast-math investigation is added in this focused iteration; their earlier
results/failures remain documented. No native Linux/timing qualification claimed.

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.sim BUILD_DIR=build/simulation-c++
PYTHONPATH=apps/loPredCtrlAcc/tests python -m unittest \
  simulation_demo_test sim_shmim_test prediction_information_test -v
python apps/loPredCtrlAcc/tests/prediction_information.py --cxx g++-15 \
  --output apps/loPredCtrlAcc/build/prediction-information-gcc
make -C apps/loPredCtrlAcc -f tests/Makefile.sim \
  BUILD_DIR=build/simulation-g++-15 CXX=g++-15
DDSPC_SIM_LIB="$PWD/apps/loPredCtrlAcc/build/simulation-g++-15/libddspc_sim.so" \
DDSPC_SIM_FIXED_LIB="$PWD/apps/loPredCtrlAcc/build/simulation-g++-15-train2400/libddspc_sim.so" \
PYTHONPATH=apps/loPredCtrlAcc/tests python -m unittest \
  simulation_demo_test sim_shmim_test prediction_information_test -v
```

Changed files: test-only `sim_bridge.cpp` and `simulation_demo.py`; new
`prediction_information.py`, `prediction_information_test.py`, this report and
its result directory; tests README, channel01 report follow-up link, and the
engineering plan below Agent Findings and Plan. Legacy, QR learning, reference
law, live applications, other worktree and production behavior stay unchanged.
No commits or pushes. The next useful scoped experiment would improve calibration
coverage and validate on independent records while retaining both unfavorable
fixed-fit cases; there is no supported alignment change to deploy from this audit.
