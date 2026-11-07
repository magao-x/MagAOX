# Phase 13 continuation: original timing and held-out WFS prediction

2026-11-06, macOS workstation, `parkertjohnson1/accelSim`. The earlier uncommitted
inspection work is preserved. These are **real April 15, 2025 telescope data**.
The recovered timing arrays permit acquisition-field alignment and a chronological
statistical prediction test. **They do not yet establish physical exposure-center
timing, host-clock registration, DM phase/sign or deployment readiness.**

Result: channels 0/2 have strong recorded-clock coherence with WFS tip/tilt near
28.75 Hz; channel 1 is strongest near 55.75 Hz. Adding all three to WFS history
reduces later held-out RMS forecast error by **2.15% tip / 3.94% tilt** at two
WFS frames ahead (approximately 1 ms). This is an offline ARX result, not a test
of the production harmonic pointing estimator or measured DM suppression.

## Data association and exact recorded-time coverage

The two recovered arrays are UINT64 `(614400,5)` and `(602902,5)`, with columns
counter, acquisition sec/nsec and write sec/nsec. Data/manifest counts, shapes,
dtypes, file ordering, finite values and all nanosecond components pass checks.
Counters advance by exactly one everywhere; acquisition stamps strictly increase.
All **75 ADC plus 251 WFS archive starts match their timing row exactly**, including
all nine nanosecond digits. Thus the exported naming convention is verified as
the first row's acquisition field for these archives. Counts and all 326 anchors
support the provided concatenation order. Original XRIF payloads/reducer provenance
are still needed to independently recheck every within-file image/time association.

| Stream | First acquisition field, UTC | Last acquisition field, UTC |
| --- | --- | --- |
| ADC | 08:29:56.737200135 | 08:35:03.936733903 |
| WFS | 08:29:58.895666422 | 08:35:00.346345018 |

The exact recorded-field overlap on 2025-04-15 is
**08:29:58.895666422–08:35:00.346345018 UTC**, spanning
**301,450,678,596 ns**. Each stream has 602,902 rows inside this interval;
that coincidence is not a one-to-one simultaneity assertion. Independent source
counters are never aligned by value or by row index.

The analysis uses the requested **08:30:00–08:35:00 UTC** window, keeping measured
pre-window history for causal features. Integer nanoseconds are retained for
association/searches and until subtracting the common epoch
`1744705798895666422`. Only relative seconds enter interpolation/FFT. The original
metadata's “filename timestamps only” description is now stale; it was preserved
without editing. All **331 input files** remain byte-identical before/after analysis.

## Cadence and the 0.491948-ms median

| Acquisition-field statistic | ADC | WFS |
| --- | ---: | ---: |
| Mean interval [ms] | 0.500000055 | 0.500000296 |
| Median interval [ms] | 0.491948 | 0.499983 |
| Interval SD [us] | 16.665 | 4.402 |
| Minimum / maximum interval [ms] | 0.383935 / 0.661718 | 0.406477 / 0.636260 |
| Average rate [Hz] | 1999.999780 | 1999.998815 |
| Lag-one interval correlation | -0.4412 | -0.4234 |
| SD of mean interval over 100 steps [us] | 0.2061 | 0.0669 |

About **67.0%** of ADC intervals are below 500 us, averaging **489.772 us**;
the remainder average **520.770 us**. Short intervals are compensated by longer
ones, consistent with bounded scheduling/timestamp variation around a 2-kHz pace.
The skewed distribution explains the short median; `1/median(dt)` is not the
acquisition rate. No interval exceeds 1 ms and no publication-counter gap exists.
This does not identify the specific producer/driver mechanism or uncounted
hardware failures.

Fitting acquisition time versus nominal counter index gives approximately
0.053 ppm (ADC) and 0.593 ppm (WFS) period deviations. Their difference can reflect
source cadence, scheduling and/or clock behavior; it is **not an independent host
clock drift measurement**. No clock correction was inferred or applied from the
vibration signals. Independent synchronization/bracketing evidence is absent.

**Write equals acquisition on every row in both streams.** These metadata do not
independently measure publication, readout, transport or consumption latency.
Producer/recorder fallback/copy semantics remain relevant. At idealized held-out
WFS query stamps, the most recent prior ADC acquisition is typically 81.995 us
old (1st–99th percentile 14.378–149.454 us; maximum 613.958 us). This is a
recorded-time separation, not measured application latency or clock offset.

## Spectral alignment, coherence and phase

For spectral analysis only, interpolate each measured acquisition sequence onto
a 2-kHz grid in the common recorded-time coordinate. No frame-counter gaps are
filled, and acquisition intervals above 1 ms cause rejection rather than hidden
interpolation. Four-second nonoverlapping Hann Welch blocks give **0.25-Hz bins**.
Raw units remain **ADC counts** and **recorded WFS modal units**. No ADC zero,
sensitivity, lever arm, rotation or WFS optical/angular calibration is invented.

The earlier 120-s interval selects each WFS peak within fixed 27–30 and 54–58-Hz
bands, giving 28.75 and 55.75 Hz. Those bins are retained for later reporting;
the test spectrum does not choose model parameters. The later 150-s interval has
37 complete Welch blocks. Its magnitude-squared coherence is:

| ADC channel | 28.75 Hz: tip / tilt | 55.75 Hz: tip / tilt |
| --- | ---: | ---: |
| 0 | **0.9665 / 0.9520** | 0.8225 / 0.7412 |
| 1 | 0.3183 / 0.3102 | **0.8700 / 0.9418** |
| 2 | **0.9498 / 0.9541** | 0.7027 / 0.5309 |

Channels **0 and 2** are the informative 28-Hz candidates; the small tilt ranking
difference does not establish a unique winner. **Channel 1** is the strongest
56-Hz candidate, particularly for tilt. Pairwise coherence does not prove these
are independent contributions or identify physical sensor axes. Channel 1's
28-Hz coherence falls from about 0.62 in training to 0.31–0.32 in testing.

Using `Paw=mean(conj(A)*W)` and phase `arg(Paw/Paa)`, selected held-out apparent
WFS-relative-to-ADC phases are:

| Pair | Phase [deg] | Conditional approximate 95% half-width [deg] |
| --- | ---: | ---: |
| ADC 0 → tip, 28.75 Hz | +95.96 | 2.43 |
| ADC 0 → tilt, 28.75 Hz | +97.37 | 2.93 |
| ADC 2 → tip, 28.75 Hz | +48.89 | 3.00 |
| ADC 2 → tilt, 28.75 Hz | +50.69 | 2.86 |
| ADC 1 → tip, 55.75 Hz | -58.51 | 5.05 |
| ADC 1 → tilt, 55.75 Hz | +87.28 | 3.25 |

These are **recorded-field phases**, including unknown source/exposure/loop
dynamics. They are not absolute optical phase, propagation delay, telescope
pointing calibration or DM correction sign. A +1-ms shift of WFS clock labels
alone changes apparent phase by -10.35/-20.07 degrees at these frequencies.
The approximate Welch intervals exclude such systematic offsets and make
stationarity/block-independence assumptions. Phase plots mask coherence below
the inherited 0.8 qualification threshold. Raw CSD/coherence remain in CSV.

## Simple causal predictor and chronological evaluation

The prototype is a standardized ridge **AR/ARX**, with 64 current/past WFS frames
from both modes (about 32 ms), optionally augmented by three ADC channels sampled
at 64 current/past 0.5-ms deadlines. ADC samples are selected by integer
`searchsorted(..., side="right")-1`; their recorded acquisitions must be no later
than each deadline. No spectral interpolation, zero-phase filter, future ADC/WFS
value or simulator truth enters a feature. Persistence and ADC-only FIR fits are
also reported. This is direct rolling prediction, not recursive open-loop rollout.

- Initial fit: **08:30–08:32**. Validation/regularization selection:
  **08:32–08:32:30**. Five ridge strengths, 1e-4 through 1, are checked per output.
- Refit with selected strengths using only pre-test data through **08:32:30**.
  Fit/validation/refit use every fourth frame; scaling uses only the respective
  earlier fit data. Target frames crossing a partition boundary are excluded.
- Held-out interval: **08:32:30–08:35**. The common cohort contains **299,980**
  full-rate queries for every model/horizon. No held-out tuning occurs.
- Forecasts target the next 1/2/4/10/20 WFS **frames**, not an invented fixed
  physical actuation deadline. Future target times are used only to score/log.
  At two frames, the actual acquisition-field horizon has median **0.999986 ms**,
  with 1st–99th percentile **0.990729–1.008983 ms**.

All features are causal **under the assumption that these recorded clocks are
comparable and measurements are available at their recorded fields**. Actual
receipt/readout timing is missing. In particular WFS acquisition is not yet
confirmed as exposure center, and a modal stream may have processing delay.
Those assumptions prevent a claim of verified physical causal lead or live timing.

Two-frame held-out RMS error, in recorded modal units:

| Predictor | Tip | Tilt |
| --- | ---: | ---: |
| Persistence | 0.012257 | 0.015097 |
| WFS-only AR | 0.009948 | 0.011938 |
| ADC-only FIR | 0.012561 | 0.015302 |
| WFS + ADC ARX | **0.009734** | **0.011467** |
| ARX RMS improvement over WFS-only | **2.152%** | **3.943%** |

A paired bootstrap of thirty five-second test blocks (4000 resamples, fixed seed)
gives conditional 95% improvement intervals **2.033–2.277% / 3.695–4.199%**.
These concern the fixed models and this time interval; they exclude retraining,
model-selection, unknown timing/systematic and broader nonstationarity uncertainty.
ADC-only results show that the sensors do not replace WFS history for broadband
prediction. This WFS-only baseline is not a claim of an optimal AO predictor.

| Expected frame horizon [ms] | Tip / tilt RMS improvement over WFS-only |
| --- | ---: |
| 0.5 | 1.04% / 2.18% |
| 1 | 2.15% / 3.94% |
| 2 | 3.46% / 6.56% |
| 5 | 4.34% / 7.07% |
| 10 | 2.94% / 3.73% |

These are different prediction tasks, not a physical DM-latency optimization.
Withholding the newest 0.5/1/2 ms of ADC data and refitting on earlier data leaves
two-frame improvements of **2.12/3.79%, 2.06/3.71%, 2.01/3.66%**. This shows some
tolerance to information delay but cannot certify unknown exposure/clock offsets.

Post-prediction band-error spectra give ARX improvements over WFS-only of
**5.06/8.04%** at 27–30 Hz and **1.30/12.58%** at 54–58 Hz (tip/tilt).
However, at the short two-frame horizon the 28-Hz persistence errors are only
0.000825/0.000622, versus ARX 0.001389/0.001129. The broadband-loss fit trades band
accuracy against other frequencies. High coherence is not proof of an optimal
narrowband predictor, complete disturbance observability or safe feed-forward.

## Reproduction, tests and remaining work

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/accelFFCtrl-phase13-timed-mpl \
  python apps/accelFFCtrl/tests/analyze_timed_telemetry.py \
  "$HOME/data/phase13_20250415_083000_083500" --output /tmp/accelFFCtrl-phase13-timed
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s apps/accelFFCtrl/tests -p test_analyze_timed_telemetry.py -v
OPENBLAS_NUM_THREADS=1 ACCELFFCTRL_TELEMETRY_REPLAY=/tmp/accelFFCtrl-phase13-build/accelFFCtrlTelemetry_replay \
  make -C apps/accelFFCtrl/tests validation-test PYTHON=python
python -m unittest discover -s apps/accelSim/tests -p test_validate_psd.py -v
```

Use the existing scientific Python environment (`python` is Anaconda here).
Build the optional Phase 12 replay fixture with `make -C apps/accelFFCtrl/tests
BUILD_DIR=/tmp/accelFFCtrl-phase13-build telemetry-tools` if it is absent.
Outputs include exact boundary/held-out timestamp CSVs, full/train/test coherence
and phase, PSD/CSD arrays, raw/band prediction errors, bootstrap intervals, chosen
ridge strengths, input hashes and five figures. No command stream is accessed.

Five new deterministic cases pass: exact original association and invalid-row
rejection; skewed cadence; future-sample exclusion/epoch translation; known causal
predictive innovation with late-target mutation; and irregular-timestamp phase
recovery/clock-shift sensitivity. Full regressions pass **41 controller Python
plus four simulator PSD tests**, and the existing unchanged numerical/mock
ASan/UBSan suite passes **67 cases / 3,677,233 assertions**. No production C++
source, configuration, observation-only/default-gap behavior or original input
byte was changed. Earlier uncommitted work remains intact; no commit/push/deploy.

Next measurements: producer/recorder timestamp semantics, source-host clock
registration and WFS exposure/readout/modal latency, physical ADC axes/sensitivity
and WFS normalization/AO-state context. The observed coupling may include feedback
or DM-driven mechanical response; passive telemetry does not prove an exogenous
disturbance or closed-loop stability. A later offline study should compare a
band-specific predictor against persistence and the current harmonic estimator
after calibration. Native Linux/ImageStreamIO qualification and any physical
DM integration remain separate, unperformed and unauthorized here.

## Follow-up: actual harmonic controller

The [existing-controller replay](REAL_HARMONIC_REPLAY.md) now compares the actual
C++ harmonic estimator and real coordinate maps against these baselines at exact
1-ms targets. It reports weak cancellation with the current map, a separate
phase-calibration diagnostic, normalized file-only proposals and unchanged
production behavior. The historical two-frame regression results above remain
distinct from that fixed-target task.
