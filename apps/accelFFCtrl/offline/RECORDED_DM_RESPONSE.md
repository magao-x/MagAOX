# Phase 13: real-data predictions through a simulated DM response

2026-11-06, local macOS, `parkertjohnson1/accelSim`.

**Useful simulated vibration reduction is demonstrated, but the harmonic
correction does not outperform the WFS-history alternatives.** With a normalized
two-step DM delay and training-selected gain 1, harmonic correction reduces
held-out broadband RMS by **5.45% tip / 5.32% tilt**, and vibration-band power by
**8.20/6.79 dB near 28 Hz** and **1.86/4.55 dB near 56 Hz**. Total outside-band
WFS power increases **0.80%/1.04%**; some neighboring bands amplify substantially
in percentage terms. The aggregate training criteria pass, but this is not
noise-neutral qualification or evidence of physical closed-loop suppression.

## Reused numerical interfaces and meaning of the simulation

The [earlier calibration](PHASE_CALIBRATED_REPLAY.md) is replayed from the original
April 15, 2025 acquisitions, using **ADC0/1 exclusively**. The harmonic estimator,
two existing carrier instances, eight phase/amplitude coefficients, source
selection, memory and prediction horizon are unchanged. Existing Phase 13 AR/ARX
features provide the WFS-only and WFS+ADC0/1 comparators. No new controller is
introduced.

The source conventions are verified:

- `apps/aoSim/xapp/aoSim/__init__.py::aoSim.update_dm()` takes the history head
  **before** enqueuing the newest absolute command. History length is `lag+1`,
  so `lag=1` delays a command by **two iterations**, not one.
- `aoSim.update_wfs()` computes **disturbance + DM state + sensor noise**. The
  opposing contribution must therefore have an outer negative sign. Commands
  are replacement values, not accumulated increments.
- The application's internal default `_dt=0.01 s` would make two numerical
  steps 20 ms. This study explicitly uses the recording's approximately 2-kHz
  query cadence, making two steps approximately 1 ms. `aoSim`'s `frequency`
  parameter controls its sinusoidal disturbance, not this recorded-data cadence.
- `offline/accelFFCtrlCancellation.hpp::dmResponse` already implements exact
  integer-step delay and an optional causal first-order response. Its ideal
  zero-bandwidth mode has `effect[n]=P*u[n-d]`, matching `aoSim` with
  `d=lag+1`. Its finite-bandwidth mode is
  `x[n]=a*x[n-1]+(1-a)*u[n-d-1]`; the extra previous interval is retained.
- `accelSimModel.hpp::model::step()` distinguishes physical pointing,
  mode-coupled WFS pointing, noisy acceleration and ADC counts. None supplies
  the missing calibration for the real recording. No simulator truth or RNG
  is used in this replay, and no synthetic physical constants are substituted.

The new file-only C++ response shim wraps the **existing** `shadowCommand` and
`dmResponse`. It consumes qualified forecast snapshots, preserving timestamps,
source identity and validity. Comparator snapshots are explicitly numerical
fixtures; their `rcond=1` marker is not a measured harmonic fit condition number.
The harmonic validity flag and acquisition identity come from its actual C++
replay. No shim attaches to shared memory or publishes a DM command.

For training WFS standard deviations `S=diag(sigma_tip,sigma_tilt)`, the virtual
calibration is

```
u[n] = common_scalar_clip(-gain * inv(S) * forecast[n], bounds = +/-3)
effect[n] = S * u[n-2]                  # nominal aoSim lag=1
residual[n] = recorded_WFS[n] + effect[n]
```

Units are **training WFS standard deviations** for commands and recorded WFS
modal units for the reference/residual. This is not a physical actuator, surface,
voltage or calibrated DM modal mapping. No extra sensor noise is added: the
recorded WFS already contains its original noise and AO response.

**This is an additive counterfactual response replay, not feedback simulation.**
The original AO loop does not react to these added corrections; neither WFS
predictor features nor recorded accelerometers change in response to them. The
recorded WFS is an immutable reference, not a reconstructed open-loop
disturbance or a live plant. WFS-history comparisons are therefore optimistic
offline alternatives, not demonstrated live closed-loop performance.

A separate `ideal_target_addition` case adds the bounded command exactly at
its requested target with no response dynamics. It reproduces the previous
ideal subtraction result. The nominal case instead evolves the delayed DM
model and evaluates the original WFS acquisition rows at their actual times.
Their nearly identical harmonic band results identify prediction limitations,
rather than nominal delay, as the dominant residual in this particular model.

## Training, gain selection and causal timing

Calibration/carrier fitting uses **08:30–08:32 UTC** only. The frequencies remain
28.6510714/55.8186193 Hz and memory remains 0.1 s. AR/ARX regularization uses an
internal 90/30-second split and refits on those same 120 seconds. No held-out
labels select frequencies, coefficients or gain.

Three gains **0, 0.5, 1** are checked on that initial training interval through
the nominal delayed response. Select minimum mean per-axis squared NRMSE subject
to **at most 1% RMS increase per axis** and **at most 5% aggregate outside-band
power increase per axis**. These are declared numerical acceptance criteria,
not instrument limits. Gain fitting is in-sample on the calibration interval,
not an independent inner test. Gain **1** is selected for all three methods;
all timing/sign/bandwidth sensitivities keep it fixed. Runtime gain defaults are
unchanged and remain zero; these are explicit offline experiment gains.

Held-out evaluation is the same **299,980** queries at **08:32:30–08:35**, final
10 ms excluded. Original ADC/WFS order, counters and integer nanoseconds are
preserved. Prediction targets are exactly **query + 1,000,000 ns**. Sample age
is included once in the existing harmonic phase. No future input measurement
enters a held-out forecast; the earlier calibration/gain interval is offline
training, not an online trial before those parameters were available. Future labels enter fitting/scoring only. Training labels'
right interpolation brackets precede the training boundary.

The pure delay counts actual recorded query iterations. It does not replace
their irregular timestamps with a uniform synthetic clock. Across the full
trace, actual DM response time minus the issuing forecast's requested target
has median **-4 ns**, 1st–99th percentile **-9.502 to +9.123 us**, extrema
**-92.658 to +134.721 us**. Nominal finite-bandwidth integration uses a separately
declared 0.5-ms step and does not model this small timing variation exactly.

Comparable source clocks, availability at recorded acquisition/write fields,
and zero WFS exposure-center offset remain assumptions. Write equals acquisition
and does not independently measure receipt. Physical exposure/readout/DM timing
is not established. Software freshness/horizon checks cannot establish an
unknown physical actuation delay.

## Held-out nominal response results

RMS uses actual recorded WFS rows; PSDs use a common recorded-time 2-kHz analysis
grid, four-second Hann Welch blocks, 0.25-Hz bins. Suppression is
`10*log10(reference_power/residual_power)` in **27–30 / 54–58 Hz** bands.
Positive numbers mean simulated residual-power reduction.

| Correction | RMS tip / tilt | 28-Hz dB tip / tilt | 56-Hz dB tip / tilt | Outside-band power change tip / tilt |
| --- | ---: | ---: | ---: | ---: |
| None | 0.013950 / 0.018291 | 0 / 0 | 0 / 0 | 0 / 0 |
| WFS-only | 0.010016 / 0.012135 | **10.34 / 9.15** | **11.00 / 13.68** | -45.08% / -53.10% |
| Phase-calibrated harmonic | 0.013190 / 0.017318 | **8.20 / 6.79** | **1.86 / 4.55** | **+0.80% / +1.04%** |
| WFS+ADC0/1 regression | **0.009886 / 0.011816** | 10.13 / 8.79 | 10.27 / 13.44 | -45.91% / -55.12% |

The harmonic model reduces broadband RMS and passes the declared aggregate
criteria. It loses to WFS-only in both target bands and in total RMS. It is a
narrowband accelerometer-only forecast, while the alternatives exploit much
broader current/past WFS information under the optimistic availability
assumption. Its remaining target-band prediction error is present even with
the ideal target response; finite memory, incomplete/variable coupling and
unrepresented WFS components are not removed by output phase calibration.
No tuning of those limitations is performed here.

### Outside-band amplification must remain visible

Outside-band WFS power is not separately identified detector/sensor noise. Some
local increases are substantial despite the small aggregate increase:

| Frequency range | Tip RMS before → after | Tilt RMS before → after | Power change tip / tilt |
| --- | ---: | ---: | ---: |
| 0–20 Hz | 0.001509 → 0.001529 | 0.001737 → 0.001744 | +2.57% / +0.81% |
| 20–27 Hz | **0.000496 → 0.000878** | 0.001495 → 0.001603 | **+213.33% / +14.91%** |
| 30–54 Hz | 0.003736 → 0.003841 | 0.004444 → 0.004653 | **+5.69% / +9.63%** |
| 58–200 Hz | 0.010007 → 0.009997 | 0.013598 → 0.013611 | -0.21% / +0.19% |
| 200–1000 Hz | 0.005449 → 0.005449 | 0.006429 → 0.006429 | +0.001% / +0.009% |

These are Welch-integrated RMS values in recorded modal units, using half-open
ranges and excluding all inclusive 27–30/54–58-Hz target-band bins. The 20–27-Hz tip
reference is weak, but the relative amplification is real in this model. The
carrier calibration does not impose phase/response constraints on neighboring
frequencies passed by the estimator's finite window. Aggregate power guards
alone do not ensure local spectral neutrality. The selected gain is **not
changed after inspecting this held-out result**. Local acceptance criteria and
further simulation remain necessary before any live correction proposal.

## Sensitivity with frozen calibration and gain

| Harmonic response case | 28-Hz dB tip / tilt | 56-Hz dB tip / tilt |
| --- | ---: | ---: |
| One-step ideal delay, about 0.5 ms | 8.21 / 6.67 | 1.80 / 4.24 |
| Two-step ideal delay, about 1 ms | 8.20 / 6.79 | 1.86 / 4.55 |
| Three-step ideal delay, about 1.5 ms | 7.84 / 6.65 | 1.78 / 4.42 |
| Synthetic 300-Hz pole, 0.5-ms pure delay | 8.00 / 6.72 | 1.83 / 4.49 |
| Incorrect response sign | **-5.25 / -5.18** | **-2.97 / -4.28** |

The pole case uses the existing model's previous-interval convention, not a
new zero-delay filter or measured bandwidth. Sign reversal amplifies the
vibration and raises RMS to 0.016015/0.021066, as expected.

An independent frozen-input reference-label mismatch of ±0.5/±1 ms retains
harmonic reductions of **7.21–8.21 / 6.28–6.67 dB** at 28 Hz and
**1.58–1.80 / 3.59–4.42 dB** at 56 Hz. Positive offset means using later recorded
WFS labels for scoring; neither forecasts nor commands are retimed. These
illustrative sensitivity offsets are not measured exposure/clock errors or an
optimization of the physical prediction horizon. All comparator sensitivities
are retained in the CSV, without selecting an optimum on held-out data.

## Sign, bounds, integrity and verification

The existing outer negative sign and common-scalar bounds are used. Commands
replace a contribution rather than accumulate. Maxima are **2.4542/2.7389
training SD** for harmonic correction, with **zero clipping**. WFS-only and
regression commands reach the ±3-SD bounds within floating-point tolerance;
5,524 and 5,054 queries are clipped, respectively, over the full five-minute
trace. Harmonic input has one initial inhibited snapshot and 100% held-out
availability, zero lost frames and no additional estimator resets.

Disabled, invalid, stale, future-source, mismatched-target and source-fault
snapshots cannot issue a nonzero new command. A delayed plant can still contain
previously issued valid commands; this is physical model state, not reusing a
stale predictor output. Every reported target remains query+1 ms. Independent
audits verify the exact two-step effect and residual sum for every nominal row.

Six new deterministic tests pass: actual `aoSim.update_dm/update_wfs` parity,
analytic sinusoidal cancellation/sign reversal/future exclusion, common clipping
and integrity inhibition, the existing finite-bandwidth step response, training
gain guards, and analytic outside-band power amplification. Mutating reference
labels leaves every command/effect unchanged, explicitly confirming that this
is not feedback. The two actual `aoSim` methods are extracted from source and
executed with host-memory float32 fixtures; imports/setup/stream APIs never run.

Full checks pass **55 controller Python + four simulator PSD tests**. All 55
controller tests also pass against ASan/UBSan-built replay/response tools. The
unchanged numerical/mock C++ suite passes **67 cases / 3,677,233 assertions**
under ASan/UBSan. Full `aoSim` application/native tests are not claimed: its
ImageStreamIO/Python application dependencies are unavailable in this environment.
No native Linux or instrument validation was performed.

## Reproduction and artifacts

Run from the repository root using the existing scientific Python environment:

```bash
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-recorded-response \
  telemetry-tools response-tools
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/accelFFCtrl-recorded-response-mpl \
  python apps/accelFFCtrl/tests/validate_recorded_response.py \
  "$HOME/data/phase13_20250415_083000_083500" \
  --replay /tmp/accelFFCtrl-recorded-response/accelFFCtrlTelemetry_replay \
  --response /tmp/accelFFCtrl-recorded-response/accelFFCtrlResponse_replay \
  --output /tmp/accelFFCtrl-recorded-response-study
OPENBLAS_NUM_THREADS=1 \
  ACCELFFCTRL_TELEMETRY_REPLAY=/tmp/accelFFCtrl-recorded-response/accelFFCtrlTelemetry_replay \
  ACCELFFCTRL_RESPONSE_REPLAY=/tmp/accelFFCtrl-recorded-response/accelFFCtrlResponse_replay \
  make -C apps/accelFFCtrl/tests validation-test PYTHON=python
python -m unittest discover -s apps/accelSim/tests -p test_validate_psd.py -v
```

Sanitizer builds use the same targets with
`CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'` and a
separate build directory, then the same tests with `ASAN_OPTIONS=halt_on_error=1`
and `UBSAN_OPTIONS=halt_on_error=1`. The earlier calibration report links the
combined C++ numerical/mock sanitizer rebuild command.

The output directory contains `report.json`, `training_gains.csv`, all scenarios
in `metrics.csv`, three selected command/response/residual traces, residual and
command time series, full and target-band PSD figures. `predictor/` contains
the reproduced calibration/metrics, input hashes and exact causal response
inputs. Original inputs remain unchanged; transient model files are removed.

Changed files: new `tests/accelFFCtrlResponse_replay.cpp`,
`tests/validate_recorded_response.py`, `tests/test_validate_recorded_response.py`
and this report; an optional export in `tests/validate_phase_calibration.py`;
test Makefile targets/registration; README, calibration-report follow-up and
engineering-plan appendices. Previous uncommitted work is preserved.

**Recommendation:** continue normalized offline response/feedback-interaction
testing. Do not infer live superiority, noise neutrality or stability from this
additive recording. Before real integration, resolve local spectral acceptance,
independent recordings/AO conditions, physical clock/exposure/availability,
sensor/WFS units, WFS-to-DM registration/sign and measured response. Genuine
feedback qualification requires a plant/disturbance model whose WFS and sensor
signals respond to applied commands; the recorded closed-loop WFS cannot serve
that role unchanged. Production behavior, configurations, default gain/gap
policy and all original data remain unchanged. No instrument access, real DM
command, shared-memory write, commit or push occurred.

## Follow-up: actual numerical feedback

The [closed-loop study](CLOSED_LOOP_VALIDATION.md) now uses actual accelSim ADC
generation and disturbance, existing WFS feedback and delayed DM response. It
finds partial target suppression but failures in cross-axis and outside-band
performance with the retained paired calibration. The additive recorded-data
results above remain a separate experiment and are not reinterpreted as feedback.
