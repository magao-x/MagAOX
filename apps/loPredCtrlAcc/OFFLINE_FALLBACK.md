# Offline ADC-assisted DDSPC fallback review

This is a **test-harness design for review**, implemented on
`parkertjohnson1/accel_ddspc`. The original simulation stage preserved production sources. The subsequent
shared extraction and opt-in application replay integration are documented in
[SHADOW_PIPELINE.md](SHADOW_PIPELINE.md). Mathematical algorithms and default Legacy
behavior remain unchanged; the shadow-only output guard is strengthened.
No native application, accelSim, INDI server, service, hardware stream, DM output,
installation, commit, merge or push is involved. Run these commands on a personal
workstation. These results are synthetic, not telescope validation.

## Reproduce

From the repository root:

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.adc-ddspc -j2 check

# All 24 comparisons, each with open/standard/assisted-requested private plants.
# REPORT_DIR must not exist; every scenario exports its complete parameters.
make -C apps/loPredCtrlAcc -f tests/Makefile.adc-ddspc comparison \
  REPORT_DIR=build/offline-fallback-review

# A single 25-ms loss, beginning after calibration, with explicit review parameters.
apps/loPredCtrlAcc/build/adc-ddspc-offline/adc_ddspc_sim \
  --drop-start 9600 --drop-count 200 --recovery-frames 10 --max-step .2 \
  --output apps/loPredCtrlAcc/build/fallback-single-review

# ASan/UBSan: actual ADC processing, real learners, switching/windows and CLI tests.
make -C apps/loPredCtrlAcc -f tests/Makefile.adc-ddspc -j2 \
  BUILD_DIR=build/adc-ddspc-fallback-asan \
  CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' \
  check-adc check-simulation check-cli

# TSan: actual ADC worker with its existing offline ImageStreamIO API double.
make -C apps/loPredCtrlAcc -f tests/Makefile.offline \
  BUILD_DIR=build/adc-ddspc-fallback-tsan \
  CXXFLAGS='-std=c++20 -O1 -g -fno-fast-math -fsanitize=thread' check-adc
```

`REPORT_DIR` is interpreted relative to the application directory by the Makefile;
use an absolute path when selecting another location. Existing output directories
are refused. The Python runner invokes only `adc_ddspc_sim`, uses no third-party
Python packages, and produces `comparison.csv`. The C++ executable links the
actual `AdcCapture`, `ShadowController`, causal `PredictiveController`, QR/RLS and
control solve, with Eigen/mxlib headers; no native app/INDI/stream driver is linked.

Each scenario has `metrics.csv`, `residuals.csv`, `psd.csv`, `learning.csv`,
`decisions.csv` and `transitions.csv`. Decision rows contain requested/effective
mode, reason, qualified streak, source/model validity, effective/latest/available
ADC timestamps, feature sample count, acceleration, consumed command and step.
Transition rows contain every authority change plus the initial evaluation state.
All command constraints include exploration before recording the consumed input.

## Selection and bounded commands

`tests/offline_control.*` now provides compatibility names for shared
`adc_features.*` and `shadow_policy.*`, independent of any native runtime. The requested mode is fixed per comparison trajectory. Effective
mode is one of `assisted`, `standard` or `safe`; this does not change the production
manager's diagnostic selected-mode label or any INDI property.

At each frozen WFS decision:

1. Require current valid WFS/applied-input timing. Unknown common evidence makes
   both models ineligible immediately, regardless of cached readiness.
2. A standard proposal must succeed through the actual `predict_cached(Standard)`
   call with the current standard history and a maximum fit age of one WFS frame.
3. Assisted eligibility additionally requires the current ADC window and complete
   consecutive feature history, plus successful `predict_cached(AccelAssisted)`.
4. Count consecutive **fully eligible assisted decisions**, resetting on any failed
   requirement, including model rank/readiness. After `recovery_frames` decisions,
   use assisted. Otherwise use the corresponding ready standard model. Initial
   assisted entry uses the same qualification as recovery.
5. If neither qualifies, use a bounded ramp toward zero, without exploration.

There is no delay before fallback. Hysteresis defaults to 10 frames (5 ms) and is
configurable with `--recovery-frames`; setting 1 removes the additional dwell.
The production model's own history/update/rank qualification still applies.

For each axis, every consumed command obeys

`u_i = clamp(v_i, max(-L,u_(i-1)-S), min(L,u_(i-1)+S))`,

where `v_i` is the selected proposal plus common evaluation probing, or zero in
safe mode. `L=3` and `S=.2` are synthetic defaults (`--limit`, `--max-step`). The
same constraints apply to both controllers and calibration. Inward float rounding
preserves the double-valued bounds. Limiting all frames prevents a second-frame
jump after a limited transition. Both learners receive exactly this consumed
absolute command after limits/slew/probing, including fallback and safe output.

The standard model is neither reset nor disabled by ADC loss. Its rolling rank
check can nevertheless reject predictions when recent inputs become unexcited.
The offline policy respects that rejection and enters safe mode; it does not
silently reuse an unqualified gain or enable Legacy. Independent validated-gain
retention would be a separate design requiring review.

## Timestamp-defined ADC features

Nominal ADC cadence is 8 kHz; WFS decisions stay 2 kHz. The ADC cadence has independent
configurable oscillator error (`--adc-drift-ppm`) and acquisition jitter
(`--jitter-ns`). Acquisition timestamps are expressed in the known common clock.
Variable receipt latency (`--latency-ns`, `--latency-jitter-ns`) is independent of
waveform noise. The offline transport preserves publication order with FIFO
head-of-line delay, avoiding false counter rollback/restart artifacts.

For deadline `t`, the half-open feature window is
`[t-lag-500 us, t-lag)`. Only samples acquired/published/received by the deadline
can contribute. At each point use the latest already available acquisition,
with causal zero-order hold; no future interpolation or late repair is allowed.
Require a left-boundary sample, contiguous counters/source generation/segment,
finite values, and no hold interval or sample gap greater than 175 us. Missing
boundary coverage or late delivery therefore invalidates a window even while
transport health remains true. Source restarts/gaps are not bridged.

With contributing sample acquisitions `s_j` and held durations `w_j`, retain

`a_bar = sum(w_j*a_j)/500 us`, `t_effective = sum(w_j*s_j)/500 us`.

The latest acquisition and latest contributing receipt are retained separately.
The default effective-age limit is 500 us, **not** the transport's 20 ms threshold.
Nominal samples at 0, 125, 250, 375 us give effective acquisition 187.5 us into the
window, 312.5 us before its ending decision. Jitter and oscillator drift change
weights and sample counts, not arbitrary groups of four indices. A deliberate
window lag requires a separately reviewed freshness bound; the matrix includes
lag 375 us / max-age 750 us for delayed data. All durations here are test settings.

This drift model changes ADC sampling cadence relative to WFS, with a known
common timestamp mapping. It does not establish how to calibrate unknown source
clock offsets/drift, nor does it emulate WFS exposure integration or native
scheduler timing. The aggregation filter is not a qualified production anti-alias
filter. Native timing qualification remains separate.

## Causality and fair comparison

One immutable schedule supplies physical acceleration, sensor payloads/delivery,
WFS noise and independent input probes to all three private plants. Runtime ADC
processing consumes only events received by the current deadline. Latent physical
acceleration drives only the plant, integrated by acquisition time even during
sensor loss. The predictor receives decoded noisy/quantized counts only.

The unchanged synthetic plant convention is

`d_(i+1)=rho*d_i+k*M*a_bar_i`, `y_i=d_i+g*u_(i-1)`,

with `M=[[1,.15],[-.1,1]]`. Current ADC therefore has an explicit one-frame lead;
DM response is a signed static gain with one-frame delay. This is a causal
correlated-input benchmark, not a measured mechanical acceleration-to-tilt model.
Prediction horizon 3 is 1.5 ms; input delay is 500 us. A previous validated fit is
applied to current history, adding no extra measurement frame delay.

Both feedback policies use identical training excitation (sigma .3), evaluation
excitation (sigma .02), noise, physical delay, absolute limits and slew limits.
Safe output has no excitation by definition. Open-loop evaluation ramps to a
neutral input and has no evaluation excitation, so open/closed comparisons must
retain that qualification. Both controlled plants remain independent feedback
trajectories. The uncorrelated scenario substitutes independent white ADC data
while preserving the physical forcing, rather than removing the disturbance.

Summary prediction RMSE is the production frozen-origin **conditional** forecast
score accumulated across valid segments, including calibration. Later actual
commands are used only to score matured forecasts; they are unavailable to the
control decision. It is the requested learner's score, not a mixed-policy online
free-running forecast. Residual variance, PSD and command RMS cover evaluation
only, including initial transition and losses.

## Measured offline results

Local Apple Clang 21 / Eigen 5 strict-floating-point run, 2026-11-12, seed 721
unless specified. Each scenario uses 2000 calibration and 4000 evaluation WFS
records (1 s and 2 s). No numerical-fault or invalid-assisted-gate frames
occurred across the 24 scenarios. Full summaries for all 72 trajectories:
[comparison CSV](tests/results/offline_fallback_2026-11-12.csv).
The reproducible ignored per-scenario artifact directory is
`build/offline-fallback-review-20261112/` relative to the app.

Variance is mean per-axis synthetic WFS²; command RMS is per-axis synthetic
absolute-input RMS. P28 and P56 are integrated Hann PSD densities over 28–29
and 54–58 Hz respectively, in WFS². Full PSD CSVs retain WFS²/Hz densities.
Fallback is total time in standard mode with assisted requested. Safe time
is separate; longest unavailable includes both modes. Maximum switch is the
largest per-axis actual command change at any effective-mode transition.

| Policy/scenario | Variance | P28 | P56 | Command RMS | Conditional RMSE | Fallback / safe (ms) | Longest unavailable (ms) | Max switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard (all five loss scenarios) | 0.002628305 | 0.001657354 | 0.0002823321 | 0.4551924 | 0.02324554 | 0 / 6.5 | 0 | 0.2 |
| Assisted requested: normal | 0.002403437 | 0.001662426 | 0.0003154995 | 0.454272 | 0.01472709 | 88.5 / 0 | 9.5 | 0.1444723 |
| Assisted requested: drop_1ms | 0.002411375 | 0.001663335 | 0.0003157552 | 0.4542177 | 0.0147245 | 110 / 0.5 | 21.5 | 0.2 |
| Assisted requested: drop_5ms | 0.002410454 | 0.001662043 | 0.0003162334 | 0.4542314 | 0.01472876 | 97.5 / 0.5 | 29 | 0.2 |
| Assisted requested: drop_25ms | 0.002407528 | 0.001665859 | 0.000315186 | 0.4542335 | 0.01474313 | 114 / 0 | 45.5 | 0.1241717 |
| Assisted requested: drop_100ms | 0.00242828 | 0.001669203 | 0.000316361 | 0.4541435 | 0.01476853 | 198.5 / 0.5 | 120.5 | 0.2 |

The standard trajectory is bit-identical across normal and ADC-dropout cases.
Default assisted-requested variance improves 8.56%, but P56 increases 11.75%.
With 25-ms loss, variance is .00240753 versus standard .00262831; the old
hold-policy result .01515 is historical and used different command constraints.
The new comparison therefore isolates policies within each row/scenario,
rather than attributing the entire historical-to-current change to one edit.

| Scenario | Standard variance | Assisted-requested variance | Variance change | P56 change | Fallback / safe (ms) | Longest unavailable (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normal | 0.002628305 | 0.002403437 | -8.56% | +11.75% | 88.5 / 0 | 9.5 |
| drop_1ms | 0.002628305 | 0.002411375 | -8.25% | +11.84% | 110 / 0.5 | 21.5 |
| drop_5ms | 0.002628305 | 0.002410454 | -8.29% | +12.01% | 97.5 / 0.5 | 29 |
| drop_25ms | 0.002628305 | 0.002407528 | -8.40% | +11.64% | 114 / 0 | 45.5 |
| drop_100ms | 0.002628305 | 0.00242828 | -7.61% | +12.05% | 198.5 / 0.5 | 120.5 |
| intermittent | 0.002628305 | 0.002575059 | -2.03% | +7.51% | 833 / 2.5 | 26 |
| uncorrelated | 0.002628305 | 0.002457829 | -6.49% | -5.76% | 59.5 / 0 | 5 |
| coupling_0 | 0.000396118 | 0.0003960577 | -0.02% | -3.17% | 13 / 0 | 8 |
| coupling_0.3 | 0.0006430105 | 0.0005928126 | -7.81% | -10.39% | 38 / 0 | 8 |
| coupling_3 | 0.169427 | 0.1523695 | -10.07% | +14.50% | 260.5 / 11.5 | 9.5 |
| lambda_0.01 | 0.0008387335 | 0.0004656245 | -44.48% | -56.53% | 96 / 0.5 | 7 |
| lambda_1 | 0.06093951 | 0.06186237 | +1.51% | +3.14% | 64.5 / 0 | 5 |
| horizon_1 | 0.002832505 | 0.00249403 | -11.95% | +3.36% | 54.5 / 0 | 5 |
| horizon_6 | 0.002361439 | 0.002211004 | -6.37% | +19.46% | 143.5 / 1 | 9 |
| jitter | 0.002625724 | 0.002404996 | -8.41% | +16.91% | 87.5 / 0 | 10 |
| latency | 0.002628305 | 0.002628305 | +0.00% | +0.00% | 1993.5 / 6.5 | 2000 |
| late_with_lag | 0.002628305 | 0.002431193 | -7.50% | -1.80% | 78.5 / 0 | 7.5 |
| stale_delivery | 0.002628305 | 0.002628305 | +0.00% | +0.00% | 1993.5 / 6.5 | 2000 |
| drift_minus200 | 0.002636828 | 0.00241545 | -8.40% | +9.26% | 96.5 / 0 | 6.5 |
| drift_plus 200 | 0.002622799 | 0.002411832 | -8.04% | +8.05% | 62.5 / 1 | 9 |
| combined_timing | 0.002617818 | 0.002617818 | +0.00% | +0.00% | 1995 / 5 | 2000 |
| wfs_loss | 0.2239289 | 0.2239166 | -0.01% | +0.00% | 14.5 / 1800 | 1800 |
| seed722 | 0.002571261 | 0.002326805 | -9.51% | +7.24% | 61.5 / 0 | 7 |
| seed723 | 0.002659814 | 0.002429564 | -8.66% | +10.68% | 99.5 / 0.5 | 9.5 |

The default .2 slew limit is satisfied at every decision and switch. Coupling 3
is strongly slew-limited, so its absolute variance is much larger. Lambda 1
makes assistance worse; lambda .01 improves both variance and 56-Hz power in
this synthetic realization. These are not proposed instrument settings.
Normal noise seeds 721–723 show similar qualitative behavior. Uncorrelated
ADC results must be interpreted as identification/policy effects, not
demonstrated feed-forward disturbance information.

With 100±80-µs delivery latency, recurring incomplete windows prevent assisted
qualification; the policy follows standard with the same residual trajectory.
The explicit 375-µs lag / 750-µs age profile permits delayed features in its
separate test. Two-ms-old samples remain transport-healthy but are rejected
by the feature gate. WFS loss intentionally exposes conservative safe lockout.

Validation:11 ADC/synthetic cases (447999 assertions),25 model/input/window/
policy cases (38610 assertions),10 guarded safety cases (2152 assertions),
and 2 CLI plus 2 configuration Python checks passed. ASan/UBSan passed the
11+25 cases and CLI checks; TSan passed 13 ADC/worker cases (448030 assertions).
No outstanding test failure or sanitizer finding remains. Clang-format 23
and whitespace validation passed. These checks do not qualify native ABI,
Linux real-time scheduling or hardware.

## Remaining limitations and production review items

- Default assisted control still raises residual power around 56 Hz. The fallback
  change addresses sensor availability; it does not impose per-frequency control
  performance. Tuning/generalization must be evaluated separately.
- Rolling excitation/rank checks can cause normal-operation fallback and repeated
  qualification delays. The reported total fallback includes these events and
  initial dwell, not only the injected dropout. CSV reasons preserve that evidence.
- With prolonged common WFS/input invalidity, safe ramping removes input excitation.
  Both learners can remain unready after data return. The WFS-loss case exposes
  this conservative lockout; no automatic re-excitation or rank bypass is added.
  A reviewed calibration/requalification path or separately validated frozen
  standard fallback would be needed before relying on unattended recovery.
- Uncorrelated sensor regressors can change identification, rank and regularization
  behavior. A performance difference in that case is not evidence that those
  measurements carry physical disturbance information.
- Acquisition/delivery drift tests assume known common timestamps and an owned
  virtual FIFO. They do not validate ImageStreamIO temporal-cube retention, actual
  producer concurrency, source clock calibration, WFS exposure timing, applied-DM
  provenance, real-time deadlines, or calibrated acceleration-to-tip/tilt dynamics.
- The opt-in application can use these shared helpers through explicit replay/verified
  input seams; no native source/provenance adapter is installed. The native shadow WFS callback
  still invalidates its incomplete timing/applied-command contract. Production
  transfer requires explicit review of authority, stale-data gates, validated-fit
  lifetime, hysteresis, units/signs, command continuity and failure policy. Legacy
  and physical DM output remain outside this offline implementation.

## Files changed in this task

All application-relative paths below are test infrastructure or documentation:

- `tests/offline_control.hpp/.cpp`: compatibility names for shared `adc_features.*` and `shadow_policy.*`.
- `tests/offline_adc.hpp/.cpp`: oscillator error, delivery variation and repeated losses.
- `tests/offline_ddspc.hpp/.cpp`: shared event schedule, real-model fallback, consumed-command learning and metrics.
- `tests/offline_ddspc_main.cpp`: review parameters and decision/transition CSV export.
- `tests/ddspc_offline_control_test.cpp`: new analytical window and policy unit tests.
- `tests/ddspc_adc_ddspc_test.cpp`: integration, loss/recovery, timing, freshness and causality regression tests.
- `tests/offline_cli_test.py`: argument, bound, timing and artifact checks.
- `tests/offline_fallback_matrix.py`: new reproducible comparison runner.
- `tests/Makefile.adc-ddspc`: helper/test sources and comparison target.
- `tests/results/offline_fallback_2026-11-12.csv`: new 72-trajectory summary; historical files preserved.
- `OFFLINE_FALLBACK.md`, `OFFLINE_ADC_DDSPC.md`, `tests/README.md`: current behavior, commands and limitations.
- Repository-relative `agents/plans/2026-11/accel-ddspc-offline-adc-tests.md`: engineering execution record.
