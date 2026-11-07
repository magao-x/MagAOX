# Phase 13: frequency-dependent phase calibration

2026-11-06, local macOS, `parkertjohnson1/accelSim`.

**The minimal calibration succeeds on held-out recorded-clock data.** Eight
earlier-fitted coefficients applied to the existing harmonic quadratures reduce
total RMS prediction error by **5.46% tip / 5.29% tilt** relative to the previous
real-valued map. Target-band residual-power reductions are **8.20/6.79 dB** near
28 Hz and **1.86/4.55 dB** near 56 Hz. Improvement persists across later portions
and three forward-only calibration/evaluation splits. This qualifies advancing
to normalized, synthetic-response correction testing; it does not qualify a
physical DM calibration or on-instrument correction.

The [original harmonic report](REAL_HARMONIC_REPLAY.md) remains the historical
comparison. Its phase-calibration diagnostic is now implemented as an explicit
optional **file-only** C++ output transformation with the existing shadow
safeguards. No production estimator, live controller or configuration changed.

## Minimal numerical change

`harmonicEstimator::predict()` already returns peak harmonic amplitude `A`,
acceleration phase `phi` at the newest acquisition, and exact acquisition and
target timestamps. If the internal sine/cosine fit is `a*cos(p)+b*sin(p)`, its
reported phase is `phi = p - atan2(b,a)`. Thus at the existing requested target,

```
psi_f = phi_f + 2*pi*f*(target_ns - sample_ns)*1e-9
z_f   = -A_f/(2*pi*f)^2 * [cos(psi_f), sin(psi_f)]
WFS_pred(target) = B_28*z_28 + B_56*z_56
```

Each real 2×2 `B_f` independently specifies cosine/sine weights for WFS tip and
tilt: eight coefficients total. ADC0 supplies the lower band, ADC1 the upper
band, as justified by the earlier recorded coherence. Both input channels are
not assumed to be orthogonal physical pointing axes. Both pass through the
existing estimator; only the specified channel is used in each calibrated
band. No extra filter, dynamics model, future measurement or estimator advance
is introduced.

For a coefficient pair `b_cos,b_sin`, this is equivalent to the complex gain
`H = b_cos - i*b_sin` acting on `z_f`. The sine sign is explicit; it is not a
second time shift or a correction sign. The separate correction contribution
has the existing outer **negative** feed-forward sign.

Calibration is linear least squares with no intercept, rejected for nonfinite
input, rank loss or singular-value ratio below 1e-8. Its basis is count-normalized
`-harmonic_counts/omega^2`, not physical radians. As before, numerical ADC
normalizers are zero=0, scale/sensitivity/lever=1, rotation=0; these do not claim
any real volts/count, acceleration sensitivity, lever arm or optical scale.
The fitted DC remains excluded. Outputs are recorded WFS modal units.

`tests/accelFFCtrlTelemetry_replay.cpp` accepts an optional final calibration
file argument. Without it, legacy numerical output is exactly unchanged in the
deterministic parity fixture. With it, the tool copies each existing prediction
and changes only its numerical output coordinates before the existing
`shadowCommand`. State, acquisition/target time, counter, generation and fit
diagnostics remain intact. The exported raw pointing/quadrature columns remain
unchanged. All sample consumption and prediction calls are unchanged.

The calibration file is versioned `accelFFCtrl-count-phase-v1`, followed by the
exact horizon and, per band, frequency, channel index and four coefficients in
tip/tilt row order. The tool rejects mismatched frequency, horizon, channel,
nonfinite/truncated/extra fields, and incompatible physical input normalizers.
Nothing loads this format into the live application.

## Training, timestamps and generalization protocol

Read-only dataset:
`~/data/phase13_20250415_083000_083500/`, real April 15 telescope telemetry.
Only **ADC0/1** enter every calculation in this phase, including the regression
comparator. The historical ADC0/1/2 regression is not rerun as a comparator here.
All 331 original files are byte-identical before/after.

Main calibration and ADC-only carrier selection use **08:30:00–08:32:00 UTC**.
Carriers are 28.6510714 and 55.8186193 Hz, from that initial ADC interval only.
Memory remains 0.1 s. There is no carrier, phase, window, mapping or horizon
selection using later labels. Coefficients are frozen after this initial fit;
even the following 30-second check is not included in a refit. The last label's
right interpolation bracket is **1744705919999736817 ns**, before the training
boundary **1744705920000000000 ns**.

The main held-out interval is **08:32:30–08:35:00**, final 10 ms excluded:
**299,980 common queries**. To give all comparators the same total calibration
data, WFS-only/ADC-assisted ridge baselines use the initial 90 s for fitting and
the next 30 s for regularization selection, then refit on the same initial 120 s.
They reuse existing Phase 13 64-frame features, with only ADC0/1 in the assisted
case. They intentionally use rolling current/past WFS features on the test
interval; the harmonic feed-forward model uses no held-out WFS value as input.
The original shared real map is fitted on the same 120 s, with the previously
chosen direct registration and no phase terms. Baseline numbers therefore differ
slightly from the historical 150-second-refit and three-channel comparisons.

Additional predeclared tests, all reported rather than selected:

- Freeze the main fit and score 08:32:00–08:32:30, 08:32:30–08:33:30,
  08:33:30–08:34:30 and 08:34:30–08:35:00 separately.
- Refit on 0–120, 60–180 and 120–240 s from 08:30:00, and evaluate only the next
  60 s. Carrier settings stay fixed from the initial 120 s, which precedes all
  test intervals. Coefficients use only the respective earlier fit windows.
  These are separate forward tests; no later fit is substituted into the main
  held-out result.

Acquisition times, counters and frame order are preserved. Integer nanoseconds
are retained until subtracting common epoch **1744705796737200135 ns**. Query
times are original WFS acquisition fields; targets are exactly query + **1 ms**.
Only already-available ADC samples enter prediction. Future WFS interpolation
is used only for earlier-fit labels and later scoring, with partition-safe
right brackets. Sample age is included once in the existing target phase.

**Assumptions remain unverified:** ADC receipt equals write/acquisition, WFS
availability equals acquisition, WFS exposure-center offset is zero, and the
clock fields are comparable. Write equals acquisition in every source row and
does not measure receipt. Held-out ADC age is median 81.995 us, 99th percentile
149.454 us, maximum 613.958 us; target-minus-latest-acquisition is consequently
median **1.081995 ms**, maximum **1.613958 ms**. These are recorded-field
separations, not measured physical propagation/readout/actuation delays. The fit
can absorb unknown clock/exposure/loop phase; it does not establish absolute
optical timing or the physical DM correction sign.

The fitted complex gains relative to the isolated harmonic **count** phasor
(including the negative inverse-omega-squared factor) are:

| Source | Gain magnitude, tip / tilt [recorded WFS units/count] | Apparent phase, tip / tilt [deg] |
| --- | ---: | ---: |
| ADC0, 28.6511 Hz | 0.018995 / 0.014292 | +94.36 / +97.42 |
| ADC1, 55.8186 Hz | 0.006582 / 0.018535 | -59.44 / +102.74 |

These are coefficients of the finite-window prediction calibration, not an
independently measured physical transfer or clock delay. Exact eight real
coefficients and their normalized-basis phases are stored in `report.json`.

## Held-out results

RMS is in recorded WFS modal units. Band dB is
`10*log10(observed_band_power / prediction_error_band_power)`, with four-second,
nonoverlapping Hann Welch segments, 0.25-Hz bins. Bands are **27–30 / 54–58 Hz**.
The same acquisition-field target labels are used for every model. Interpolation
to a 2-kHz grid is confined to PSD scoring, never predictor input.

| Method | RMS tip / tilt | 27–30 Hz dB tip / tilt | 54–58 Hz dB tip / tilt |
| --- | ---: | ---: | ---: |
| WFS-only AR | 0.009902 / 0.011903 | 10.35 / 9.24 | 11.04 / 14.45 |
| WFS + ADC0/1 ARX | **0.009774 / 0.011583** | 10.14 / 8.87 | 10.30 / 14.15 |
| Original harmonic, real shared map | 0.013934 / 0.018265 | 0.01 / -0.02 | -0.00 / 0.02 |
| **Phase-calibrated harmonic** | **0.013174 / 0.017298** | **8.20 / 6.79** | **1.86 / 4.55** |

The new harmonic model's total NRMSE is **94.54%/94.67%** and correlations are
**0.326/0.322**, versus about 100% NRMSE and 0.007/0.036 previously. The full
RMS improvement is modest because these are narrowband predictions of a
broadband WFS record. WFS-history baselines remain substantially better on
total error and on much of the spectral error; the calibration does not make
the harmonic model a broadband replacement. Nor does fitting the ARX broadband
loss guarantee its smallest error in each vibration band.

Reusing the Phase 13 paired five-second block bootstrap (30 blocks, 4000 draws,
fixed seed), the RMS improvements over the original harmonic map have conditional
95% intervals **4.68–6.29% tip / 4.38–6.16% tilt**. These exclude uncertainty from
retraining, clock/exposure assumptions and other observing conditions. They
support within-record improvement, not broader physical qualification.

The normalized ideal-subtraction residual removes approximately **84.9%/79.1%**
of observed 28-Hz band power and **34.9%/64.9%** of 56-Hz band power. These are
offline error-power reductions, not measured DM or feedback-loop suppression.
The 56-Hz tip improvement is real within these tests but remains limited.

### Chronological qualification

With the original first-two-minute calibration frozen:

| Later interval, seconds from 08:30 | 28-Hz dB tip / tilt | 56-Hz dB tip / tilt |
| --- | ---: | ---: |
| 120–150 | 8.84 / 6.86 | 2.12 / 4.18 |
| 150–210 | 7.17 / 5.88 | 1.83 / 4.61 |
| 210–270 | 8.18 / 7.25 | 1.50 / 3.90 |
| 270–300 | 9.93 / 8.15 | 3.10 / 5.96 |

Separate forward-only recalibrations:

| Fit → test interval [s] | 28-Hz dB tip / tilt | 56-Hz dB tip / tilt |
| --- | ---: | ---: |
| 0–120 → 120–180 | 8.66 / 7.25 | 1.85 / 3.83 |
| 60–180 → 180–240 | 9.27 / 7.48 | 0.98 / 3.77 |
| 120–240 → 240–300 | 6.78 / 4.85 | 1.21 / 3.55 |

Every split improves on the corresponding original real map. These overlapping
training windows and nearby time blocks are not independent telescope datasets.
They establish within-record chronological generalization, not night-to-night,
AO-state, pointing or frequency-drift robustness. Coefficient drift and modest
56-Hz performance warrant retaining those limits in future simulation claims.

## Normalized correction proposals

The actual calibrated C++ predictions feed the existing `shadowCommand` with
an explicitly virtual identity response. Its formula remains

`u = -gain * WFS_pred / sigma_train`,

followed by the existing common-scalar clipping. Training WFS SD is
**0.01434487/0.01884375**. Bounds are **±3 training SD**, not hardware protection
or physical modal/DM units. Gain defaults to **zero**; a separate gain-one
manifest describes the counterfactual proposals.

Maximum absolute gain-one contributions are **2.4542/2.7389 training SD**,
with zero clipping in this recording. All disabled and invalid outputs are zero.
The replay logs one initial attachment/reset-barrier inhibition, no subsequent
resets, zero dropped frames and **100% held-out availability**. It processes
606,506 original frames through the last query, including pre-window history.
It never holds or accumulates a previous proposal across invalid input.

The Python audit independently reconstructs the expected calibrated forecast,
negative gain, common clipping, bounds and exact 1-ms target and compares them
to C++ output. Existing 2-ms freshness and 50-ms prediction-span bounds remain.
Synthetic tests force clipping, stale queries, invalid ADC, counter rollback,
producer restart and warmup, and verify zero inhibited contributions. Raw C++
harmonic columns and the legacy no-calibration replay remain exactly unchanged
in the parity fixture.

## Reproduction, tests and products

```bash
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-phase13-phase telemetry-tools
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/accelFFCtrl-phase13-phase-mpl \
  python apps/accelFFCtrl/tests/validate_phase_calibration.py \
  "$HOME/data/phase13_20250415_083000_083500" \
  --replay /tmp/accelFFCtrl-phase13-phase/accelFFCtrlTelemetry_replay \
  --output /tmp/accelFFCtrl-phase13-phase-study
OPENBLAS_NUM_THREADS=1 \
  ACCELFFCTRL_TELEMETRY_REPLAY=/tmp/accelFFCtrl-phase13-phase/accelFFCtrlTelemetry_replay \
  make -C apps/accelFFCtrl/tests validation-test PYTHON=python
python -m unittest discover -s apps/accelSim/tests -p test_validate_psd.py -v
```

The four new tests pass, covering independent band phase/gain and earlier-only
fitting; actual-C++ calibration/shadow arithmetic and future exclusion; resets,
invalid/stale data and warmup inhibition; and incompatible calibration rejection.
Full checks pass **49 controller Python + four simulator PSD tests**. The full
49-test controller Python suite also passes with the ASan/UBSan replay. The
unchanged combined numerical/mock sanitizer suite passes **67 cases /
3,677,233 assertions**. C++ formatting and Python formatting/syntax checks pass.

```bash
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-phase13-phase-asan \
  CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' telemetry-tools
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 OPENBLAS_NUM_THREADS=1 \
  ACCELFFCTRL_TELEMETRY_REPLAY=/tmp/accelFFCtrl-phase13-phase-asan/accelFFCtrlTelemetry_replay \
  make -C apps/accelFFCtrl/tests validation-test PYTHON=python
```

The [earlier report](REAL_HARMONIC_REPLAY.md) supplies the combined C++ sanitizer
rebuild command. Optional environment variable
`ACCELFFCTRL_TELEMETRY_REPLAY_LEGACY` points to the retained earlier executable
for exact old/new no-calibration comparison; it was exercised here.

Products in `/tmp/accelFFCtrl-phase13-phase-study/`: prediction and residual
time-series plots, residual PSDs, normalized shadow plots/CSV, all-model metrics,
frozen chronological and forward-fold metrics, conditional block-bootstrap RMS
improvement, calibration weights/gains/phases, exact absolute timestamps,
disabled and separate counterfactual manifests, and input hashes. None is a
physical DM command. Temporary raw replay files are removed automatically.

Files changed in this continuation (paths relative to `apps/accelFFCtrl/` unless
otherwise specified):

- `tests/validate_phase_calibration.py`: real-data fit, forward tests and audits.
- `tests/test_validate_phase_calibration.py`: four deterministic cases.
- `tests/accelFFCtrlTelemetry_replay.cpp`: optional file-only calibrated output.
- `tests/qualify_telemetry.py`: serialize the optional calibration argument.
- `tests/validate_real_harmonic.py`: reuse two-channel baselines/plotting.
- `tests/analyze_timed_telemetry.py`: allow the spectral grid to use two channels.
- `tests/Makefile`: register the new tests.
- `offline/PHASE_CALIBRATED_REPLAY.md`: this report.
- `offline/REAL_HARMONIC_REPLAY.md`: historical-result follow-up link.
- `README.md`: commands, results and limits.
- Repository `agents/plans/2026-11/accelSim-28hz-feedforward-controller.md`:
  engineering record.

Earlier uncommitted work and all production sources/configurations are preserved.

## Minimum production support, not implemented

This result supports **normalized offline DM-response simulation** using the
calibrated predictions, including bandwidth, latency mismatch and AO interaction.
Physical correction remains blocked by clock/exposure/availability semantics,
sensor and WFS units, WFS-to-DM registration/sign and measured response.

The minimum eventual live-controller change would be an optional, versioned
per-band calibration applied **after** unchanged harmonic estimation and before
diagnostic/shadow output. It needs eight coefficients, exact basis/channel/
frequency/provenance identifiers, explicit configuration reset/invalid handling,
and telemetry identifying the active calibration. For simultaneous 28/56 Hz,
the current single-carrier app also needs two existing estimator instances with
identical source/target validity; the offline replay already demonstrates this
composition. Application would require only two sine/cosine evaluations and
eight coefficient multiplies per prediction (one sine/cosine pair per band),
with no new per-sample allocation or predictor architecture. No DM writer is
required for observation-only support.

The count-basis coefficients must not be loaded unchanged into a physically
scaled live estimator. Its ADC scale, polarity, lever/coordinate convention and
calibration basis must match, or the coefficients must be explicitly converted
or refitted. Passive coupling may also include existing AO/DM-driven motion;
it does not establish an exogenous disturbance or closed-loop stability.

Any such production integration remains a separate change. Keep the option
disabled, preserve default reset/gap policy, and independently qualify Linux
build/native ingestion, physical timing and calibration before interpreting
proposed output as an instrument correction. No commit, push, deployment,
instrument access or production shared-memory write occurred.

## Follow-up: delayed normalized response

The [recorded DM-response study](RECORDED_DM_RESPONSE.md) now applies these
forecasts through the existing shadow and synthetic response model with verified
`aoSim` delay/addition conventions. It reports training-only gain selection,
held-out suppression, comparator results, timing sensitivity and localized
outside-band amplification. It is an additive counterfactual, not feedback
simulation; the original phase-calibration results above are preserved.
