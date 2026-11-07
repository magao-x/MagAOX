# Phase 13: genuine numerical feedback validation

**Result: the existing paired phase calibration is not consistently qualified.**
The numerical feedback connection is causal and works, and nominal feedback
gains have stable linearized poles. Feedforward improves 28-Hz tip suppression,
but worsens 56-Hz tip in every run, raises tilt RMS in three of six runs, and
adds substantial outside-band power relative to WFS-only feedback. No gains,
source correlations or calibration topology were adjusted to make it pass.

These are synthetic numerical results, not telescope telemetry or physical DM
suppression. The earlier [recorded-response report](RECORDED_DM_RESPONSE.md)
remains an additive counterfactual; this study actually feeds corrected WFS
measurements back into subsequent commands.

## Components and exact causal loop

The file-only exporter reuses:

- `accelSimModel.hpp::model::step()` for disturbance and actual UINT16 ADC
  generation, including noise, rounding and quantization.
- `accelFFCtrlModel.hpp::harmonicEstimator` for every ordered raw ADC pair,
  with the simulator's actual ADC decoding settings and unchanged warmup,
  nonfinite/source/time checks and prediction timing.
- The existing per-frequency quadrature calibration, shared with the previous
  replay through `tests/accelFFCtrlPhase_test.hpp`.
- `offline/accelFFCtrlShadow.hpp::shadowCommand` for opposing gain, validity,
  source identity, exact target and common-scalar normalized bounds.
- The existing `tests/accelFFCtrlAO_test.hpp::syntheticIntegrator` for WFS
  feedback. The harmonic branch itself still has no runtime WFS input; WFS
  enters that existing parallel AO path and earlier calibration only.
- `offline/accelFFCtrlCancellation.hpp::dmResponse`, in ideal-delay mode,
  matching the previously verified `aoSim.update_dm/update_wfs` conventions.

For training normalization `S=diag(sigma_tip,sigma_tilt)`, the numerical loop is

```
u_AO[k] = u_AO[k-1] - G * inv(S) * y[k-1]
u_FF[k] = clip(-g * inv(S) * calibrated_harmonic_prediction[k], +/-3)
          when valid/enabled, otherwise zero
u[k]    = common_scalar_clip(u_AO[k] + u_FF[k], +/-3)
y[k]    = d[k] + S*u[k-2]
```

The existing shadow proposer supplies the validity gates and first clipping
step. `d` is the simulator's WFS-coupled disturbance in radians;
`m_wfsPointing` applies its configured static per-mode coupling once. That
source contains no feedback controller. The original physical pointing still
generates the accelerometer readings. The DM does not mechanically drive the
accelerometers in this model.

Commands are absolute replacement contributions. The AO integrator has its
existing zero leak and no new anti-windup. `aoSim` lag=1 consumes a two-entry
history before enqueuing new commands, hence the **two-control-frame DM delay**.
The AO path additionally uses the previous WFS sample: its measurement-to-effect
latency is three frames, while feedforward predicts the two-frame actuation
target. No algebraic current-WFS/current-command loop is introduced.

Acquisition runs at **8 kHz**, control/WFS at **2 kHz**. Every ADC sample is
consumed. At a query, the newest ADC is **125 us old**, because the current ADC
arrival is processed after that query. The estimator predicts exactly
**query+1 ms**, hence 1.125 ms beyond its newest acquisition, including sample
age once. Integer simulation timestamps and counters are exact. The current
corrected WFS becomes available only for the next feedback update.

The source is generated once per population/seed, then replayed identically
through every controller case. The native local binary fixture holds WFS
radians and exact integer ADC codes in double columns; codes are validated and
cast back to UINT16 before the actual decoder. No quantization is bypassed and
no future disturbance value enters evaluation-time prediction. The file is not
an instrument shared-memory format. Future-forcing mutation tests leave all
preceding commands unchanged.

## Frozen simulation calibration and fixed experiments

Real coefficients cannot transfer from recorded WFS/count coordinates into this
known-SI simulator. Each simulation fits its own eight coefficients with the
existing least-squares routine, using the same paired topology:
**ADC0 at the lower carrier, ADC1 at 56 Hz**, each contributing cosine/sine
weights to both outputs. Basis is `-A/(omega²*lever)` in simulated radians;
weights map to simulated WFS radians. A distinct simulation file version rejects
the recorded count-normalized calibration format.

This topology is deliberately retained. `accelSim` generates independent X/Y
histories at both carriers. One sensor per frequency therefore lacks guaranteed
observability of the other independent axis. Finite-interval incidental phase
correlation can fit well initially and change later. No artificial axis
correlation or additional coefficients are introduced to improve the outcome.

| Setting | Value |
| --- | --- |
| Duration / seeds | 60 s each; 42, 123, 701 |
| Populations | Two modes, 28.5/56 Hz; unchanged 214-mode `referenceModes(123)` |
| Full-population carriers | 28/56 Hz, fixed from source definitions |
| Lever / ADC | 10 m; zero 2048, volts/count 0.0083799, sensitivity 1.02 |
| ADC noise PSD | 1e-8, unchanged simulator convention |
| Estimator memory | 0.1 s; original cold-start warmup |
| Fit interval | First 20 s; only ready rows and labels strictly before 20 s |
| Control enable | 20 s, after calibration completes; no estimator reset |
| Held-out scoring | 30–60 s, 60,000 WFS samples per case |
| Nominal AO/FF gain | Existing AO G=0.1, FF g=1; no held-out tuning |
| Normalized bound | +/-3 training WFS SD, both FF and combined contribution |
| Disable experiment | FF off at 45 s; AO state continues unchanged |

Source ADC parameters are the real simulator settings, not claimed telescope
sensor calibration. Calibration coefficients are frozen after the training
interval; the harmonic fit itself continues its intended causal ADC updates.
No detector noise, exposure averaging, physical DM bandwidth or mechanical
DM-to-accelerometer coupling is added. The virtual DM mapping is training SD
per coefficient, not actuator/surface/voltage calibration.

Fixed comparisons include no correction, WFS-only, harmonic-only (an open-loop
diagnostic), WFS+harmonic, disabled-shadow parity, controlled FF disable, wrong
FF sign, AO gain 0.3 with/without FF, and high AO gain 0.8 stress. This WFS-only
baseline is the existing integrator, not the previous recorded-data AR model.

## Held-out RMS and vibration suppression

RMS is in **simulated mas**. Spectra use four-second Hann blocks, 0.25-Hz bins.
Region suppression is `10*log10(reference PSD / residual PSD)` integrated over
27–30 and 54–58 Hz. Positive values reduce power; negative values amplify it.

| Source / seed | No correction tip / tilt RMS | WFS-only tip / tilt RMS | WFS+harmonic tip / tilt RMS |
| --- | ---: | ---: | ---: |
| Dual / 42 | 1.0949 / 1.4995 | 0.8818 / 1.1756 | 0.5072 / 0.9966 |
| Dual / 123 | 1.0686 / 1.2177 | 0.8409 / 0.9441 | 0.5248 / **1.0275** |
| Dual / 701 | 0.7812 / 0.8769 | 0.6049 / 0.6929 | 0.3245 / 0.6539 |
| Full / 42 | 1.8987 / 1.7691 | 1.5213 / 1.4362 | 0.9416 / **1.4886** |
| Full / 123 | 1.5515 / 1.1736 | 1.2581 / 1.0234 | 0.8050 / **1.1453** |
| Full / 701 | 1.3126 / 1.6082 | 1.0958 / 1.3062 | 0.8158 / 1.2060 |

Increment from adding FF to the **same G=0.1 feedback loop**:

| Source / seed | 28-region dB tip / tilt | 56-region dB tip / tilt | Outside-region power change tip / tilt |
| --- | ---: | ---: | ---: |
| Dual / 42 | 16.29 / 0.85 | -0.73 / 18.51 | +148.72% / +53.45% |
| Dual / 123 | 16.79 / -1.46 | -3.75 / 13.09 | +168.10% / +48.15% |
| Dual / 701 | 14.02 / -0.62 | -3.67 / 15.64 | +161.40% / +37.04% |
| Full / 42 | 7.66 / -0.52 | -0.50 / 1.18 | +18.49% / +2.05% |
| Full / 123 | 10.48 / -1.89 | -0.46 / 1.01 | +28.77% / +2.15% |
| Full / 701 | 6.57 / 0.75 | -0.63 / 0.89 | +19.91% / +1.17% |

Tip RMS improves 25.6–46.4% relative to WFS-only. Tilt RMS worsens **8.83%,
3.65% and 11.91%** in the three highlighted runs. The existing paired map does
not consistently improve both independent axes. The full source includes
strong 54/58-Hz modes inside the 56-Hz region; at the exact, predeclared 56-Hz
bin, full-population tilt improves **4.82–6.78 dB** rather than the region's
0.89–1.18 dB. Both diagnostics are saved; the region metric is not replaced by
a more favorable bin metric.

Outside-region growth is not just a large percentage of zero: for seed 42,
outside RMS changes from **0.0689/0.0707 to 0.1087/0.0876 mas** in the dual
case, and **0.5357/0.5103 to 0.5831/0.5155 mas** in the full case. Neighboring
20–27-Hz tip power rises **140% / 444%** (dual/full seed 42), and 30–54-Hz tip
power rises **173% / 42%**. The finite estimator window and carrier-specific
calibration do not ensure neighboring-frequency neutrality. These are total
WFS spectral powers, not separately measured detector noise.

Nominal dual/123 and full/123 show failures with **zero held-out combined
clipping**. Raising a bound therefore cannot resolve the calibration/coverage
problem by itself. Using the previous study's numerical +1% RMS/+5% outside-power
screen, the paired policy does not qualify consistently. The conclusion also
follows directly from the tilt RMS and cross-axis target amplification without
adopting any particular numerical threshold.

## Commands, saturation, stability and disable

Example command RMS values in training WFS SD; saturation is the fraction of
held-out queries limited by the combined bound:

| Seed 42 case | Total-command RMS tip / tilt | FF-command RMS tip / tilt | Combined saturation |
| --- | ---: | ---: | ---: |
| Dual, WFS-only | 0.813 / 1.070 | 0 / 0 | 0.11% |
| Dual, WFS+FF | 0.930 / 1.094 | 0.931 / 0.553 | 0.40% |
| Full, WFS-only | 1.540 / 1.403 | 0 / 0 | 6.72% |
| Full, WFS+FF | 1.638 / 1.555 | 1.221 / 0.436 | 10.72% |

Across nominal runs, FF peaks are **2.45–3.00 SD**, combined saturation ranges
from **0 to 10.72%**, and applied total commands obey +/-3 within floating-point
tolerance. The integrator state is not artificially clipped or reset; its
internal excursion is reported separately. No new anti-windup is added.

For the unsaturated zero-leak feedback law, the characteristic polynomial is
`z³-z²+G`. Maximum pole magnitudes are **0.86695 at G=0.1**, **0.81271 at G=0.3**,
and **1.07805 at G=0.8**. Thus the nominal linearized loops are stable, while the
predeclared high-gain case is unstable. The latter saturates **55–72%** of held-out
queries and reaches **8–31 SD** internally. Hard output bounds do not establish
stability. The exogenous FF path does not change these homogeneous linear poles;
this does not cover mechanical coupling or nonlinear instrument behavior.

The fixed G=0.3 sensitivity does not cure the problem: compared with its own
G=0.3 WFS-only baseline, tilt RMS still increases 3.74% in dual/123 and 3.20% in
full/123. It is not selected as a replacement after seeing those results.

Wrong FF sign worsens RMS relative to WFS-only in all twelve axis/run comparisons
and amplifies the controlled 28-Hz tip and 56-Hz tilt regions. Turning
FF off at 45 s produces **zero new FF commands immediately**; its direct delayed
contribution flushes in **two frames / 1 ms**. The continuing AO state then
converges naturally to the WFS-only trajectory within **61.5–69 ms** at a
1e-8-training-SD tolerance; all six trajectories match at floating precision
after one second. No AO or estimator reset is used. This distinction between
pipeline flushing and feedback-state recovery is essential in a real loop.

## Reproduction and verification

```bash
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-closed-loop \
  closed-loop-tools telemetry-tools response-tools
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/accelFFCtrl-closed-loop-mpl \
  python apps/accelFFCtrl/tests/validate_closed_loop.py \
  --executable /tmp/accelFFCtrl-closed-loop/accelFFCtrlClosedLoop_export \
  --output /tmp/accelFFCtrl-closed-loop-study --seconds 60
OPENBLAS_NUM_THREADS=1 \
  ACCELFFCTRL_CLOSED_LOOP=/tmp/accelFFCtrl-closed-loop/accelFFCtrlClosedLoop_export \
  ACCELFFCTRL_TELEMETRY_REPLAY=/tmp/accelFFCtrl-closed-loop/accelFFCtrlTelemetry_replay \
  ACCELFFCTRL_RESPONSE_REPLAY=/tmp/accelFFCtrl-closed-loop/accelFFCtrlResponse_replay \
  make -C apps/accelFFCtrl/tests validation-test PYTHON=python
python -m unittest discover -s apps/accelSim/tests -p test_validate_psd.py -v
```

All **61 controller Python + four simulator PSD tests pass**. The full 61-test
controller suite also passes against ASan/UBSan-built file tools; the existing
numerical/mock C++ sanitizer suite passes **67 cases / 3,677,233 assertions**.
Sanitizer builds use the same three targets with
`CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'`, separate
build/output directories, and `ASAN_OPTIONS=halt_on_error=1` /
`UBSAN_OPTIONS=halt_on_error=1`.

Six new tests verify actual feedback reaction and delay, simulator ADC/counter
identity and training freeze, future-forcing exclusion, FF disable and exact
zero-gain parity, saturation versus linear instability, and rejection of the
incompatible real-telemetry calibration. Every trajectory independently verifies
the integrator recurrence using its **own corrected WFS**, the two-frame DM
effect, absolute command bounds and measurement ordering. Shadow-disabled and
WFS-only trajectories match exactly. Moving phase arithmetic into the shared
test helper also preserves **all columns of 4000 phase-enabled replay rows
exactly** against the retained pre-extraction executable.

Output subdirectories contain source hashes, exact forcing/ADC fixtures,
training rows, frozen coefficients, all ten controller traces, complete metrics,
one-second RMS histories and feedback/disable audits. Seed 42 plots show time
series, normalized commands, target PSDs and RMS evolution, including the stress
and disable cases; all seeds remain in the master tables. Reproduction is
deterministic within the same C++ standard-library RNG implementation; no
cross-platform byte identity is assumed.

Changed files: new shared `tests/accelFFCtrlPhase_test.hpp`,
`tests/accelFFCtrlClosedLoop_export.cpp`, `tests/validate_closed_loop.py`,
`tests/test_validate_closed_loop.py` and this report; phase-helper extraction
from `tests/accelFFCtrlTelemetry_replay.cpp`; Makefile targets/registration;
README, prior-report follow-up and engineering plan. Production headers,
applications, configurations and default gain/gap behavior are unchanged.

**Recommendation: no-go for general two-axis correction with the current paired
map.** The smallest next offline calibration investigation is to use the
already-estimated quadratures of both sensor axes at each frequency, or establish
that the proposed physical plant really has the correlated axis structure on
which the paired mapping relies. That would be a calibration-topology change,
not a new harmonic estimator, and was not implemented to rescue this test.
Neighboring-band behavior, headroom and measured plant/clock/exposure/WFS-to-DM
calibration also remain unresolved. No instrument access, real DM command,
shared-memory publication, production change, commit or push occurred.
