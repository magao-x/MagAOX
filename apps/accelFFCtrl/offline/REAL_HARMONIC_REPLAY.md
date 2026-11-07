# Phase 13: existing harmonic controller on real telemetry

2026-11-06, local macOS, branch `parkertjohnson1/accelSim`.

**Result: no-go for expecting useful vibration cancellation from the current
real-valued correction mapping on these recordings.** The unchanged numerical
controller runs successfully and its normalized shadow outputs pass timing,
sign, inhibition and amplitude checks. Its fitted real two-axis map, however,
removes almost none of the measured 28–29/56-Hz vibration. Existing causal
quadratures retain predictive information: a separately labeled phase-calibration
diagnostic improves the residual spectra. The first demonstrated blocker is the
frequency-dependent, phase-aware accelerometer-to-WFS calibration, rather than
ingestion failure. No production algorithm was redesigned to make this result
positive.

This is recorded-clock offline prediction, not measured DM suppression. The
[earlier timed analysis](PHASE13_TIMED_ANALYSIS.md) documents original archive
association, coherence, clock limitations and the separate regression study.

## Inputs, controller reuse and chronological protocol

Inputs are the real April 15, 2025 dataset at
`~/data/phase13_20250415_083000_083500/`. The 614,400 original eight-channel ADC
frames, 602,902 WFS frames, manifests and recovered five-column timing arrays
are loaded with the existing Phase 13 readers. All **331 original files are
hashed before and after** and remain unchanged. ADC channels 0/1 alone enter
the harmonic controller. They are justified signal candidates: ADC0 carries the
strong 28-Hz coupling and ADC1 carries the strongest 56-Hz coupling. They are
not established orthogonal physical pointing axes. Channel 2 is retained only
in the earlier three-channel regression comparator.

`tests/validate_real_harmonic.py` calls the existing Phase 12
`accelFFCtrlTelemetry_replay.cpp` through `qualify_telemetry.py::replay`. That
executable directly uses the unchanged production `inputProcessor` and
`harmonicEstimator`, and the existing host-memory `shadowCommand`. There is no
ImageStreamIO, application launch or DM stream API in this executable. Original
ADC frame order, counters and acquisition stamps are preserved; every frame
received by a query is consumed, with no interpolation or decimation into the
estimator. The last query has processed **606,506** frames, including pre-window
history; later source frames are not consumed. A constant generation of 1 is
an analysis segment identifier, not recovered producer-generation metadata.

Memory time stays **0.1 s**, nominal acquisition rate is the measured **2 kHz**,
and default reset-on-loss/freshness settings remain unchanged. Two small,
predeclared frequency choices are compared: configured 28.5/56 Hz, or ADC-only
training Welch peaks refined by local log-parabolic interpolation,
**28.6510714/55.8186193 Hz**. No held-out peak, phase, time-offset or memory-time
search is performed. A single-frequency result uses one existing estimator;
the simultaneous result uses the already-existing offline two-bank composition
and one shared real mapping. This does not add dual-frequency support to the
live single-frequency application.

For each frequency choice and single/dual-bank variant, compare only:

- Direct: ADC0-like output maps to tip, ADC1-like output to tilt.
- Swapped: the opposite axis assignment.
- Full: a real, origin-preserving 2×2 coordinate/modal map with cross-axis terms.

These are numerical candidate registrations, not measured sensor geometry. Fit
maps by least squares to earlier WFS data, select by validation error normalized
by **initial-training** WFS standard deviations, then refit the selected map on
all pre-test data. Zero model input maps to zero; no intercept or time shift is
added to a correction map. The full candidate permits gains and cross coupling
beyond the controller's rotation alone; even this concession does not supply
the missing band-dependent temporal phase.

| Stage | Recorded UTC on 2025-04-15 |
| --- | --- |
| Initial fit | 08:30:00–08:32:00 |
| Model/mapping selection | 08:32:00–08:32:30 |
| Final calibration refit | Before 08:32:30 only |
| Held-out scoring | 08:32:30–08:35:00, final 10 ms excluded |

There are **299,980 identical held-out queries** per method. A target's right
interpolation bracket must also precede a fit/validation boundary. No held-out
WFS values enter harmonic estimation, mapping selection or its final refit.
The WFS-only and ARX comparison methods intentionally use current/past held-out
WFS as rolling features, as in Phase 13; future WFS is never a feature. They are
comparison baselines, not accelerometer-only feed-forward implementations.

## Exact timing and causal assumptions

The original exact recorded-field overlap is
08:29:58.895666422–08:35:00.346345018 UTC. Queries use original WFS acquisition
fields within the requested five-minute window. Source times and searches stay
integer nanoseconds until subtracting epoch **1744705796737200135 ns**, the
first ADC acquisition. This epoch retains all measured ADC prehistory. The
Phase 12 executable receives a common epoch translation, which is undone in
the output CSV.

Every target is exactly

`target_ns = query_ns + 1,000,000`.

The estimator advances from the latest acquisition to that target, including
sample age exactly once. It does not predict 1 ms beyond the latest ADC sample
and then add a second latency shift. Future observed WFS values are interpolated
only for labels, calibration in earlier intervals, and scoring. Prediction and
truth are evaluated at the same recorded-field target, rather than rounded to
the next two WFS frames. Consequently the refitted baseline numbers differ
slightly from the historical two-frame results; both are preserved.

Held-out latest-sample age is median **81.995 us**, 99th percentile **149.454 us**,
maximum **613.958 us**. Thus the median latest-sample-to-target span is
**1.081995 ms**, with maximum **1.613958 ms**. The query-to-target horizon itself
remains exactly 1 ms. These are recorded-field separations, not measured physical
sensor/readout/DM latency.

Unverified assumptions are explicit:

- ADC sample availability equals its recorded write/acquisition field: **0 ms
  assumed additional latency**. Every write field equals acquisition in this
  dataset, so this is not a measured consumer receipt time.
- WFS values are available at their acquisition field to the baseline predictors:
  **0 ms assumed WFS processing latency**.
- WFS exposure center equals its acquisition field: **0 ms assumed offset**.
- Both acquisition fields share a comparable clock. No independent clock or
  exposure/readout calibration has been recovered.

Frozen-model additional ADC availability delays of **0.5 and 1 ms** are also
replayed. Held-out validity remains 100%; raw RMS stays approximately
0.013934/0.01826. These checks do not validate unknown physical timing. The
original jitter (ADC median 0.491948 ms, mean 0.500000055 ms) is retained, not
replaced by an assumed uniform cadence. There are **zero dropped frames** and
**one initial attachment reset**, with no subsequent resets. The first shadow
query is inhibited by its reset barrier; held-out prediction availability is
**100%**. Pre-window ADC history has already supplied cold-start warmup.

## Units, equations and numerical calibration

The production model normally decodes

`a = polarity * (count - zero) * voltsPerBit / sensitivity`,

fits DC/cosine/sine coefficients with its two-stage timestamp-weighted window,
and evaluates the isolated acceleration at the requested target. Its physical
pointing convention is

`theta = -R * a_harmonic / (leverArm * (2*pi*f)^2)`.

Real ADC sensitivity/axes, effective lever arm and WFS angular normalization are
not supplied for this telescope recording. Therefore this experiment uses
**arithmetic normalizers** zero=0, voltsPerBit=1, sensitivity=1, leverArm=1,
rotation=0. These do **not** assert 1 V/count, 1 V/(m/s²), 1 m, or radians.
The fitted DC absorbs the raw count offset; outputs are explicitly the numerical
coordinate `x_f = -harmonic_counts/(2*pi*f)^2`. This preserves the existing
estimator mathematics without asserting false physical acceleration. A
deterministic C++ regression verifies equivalence under linear rescaling to the
simulator's separate calibration values; those values are not assigned to the
real data.

The earlier-fit real map gives `WFS_pred = M * sum_f(x_f)` in **recorded WFS
modal units**. Selected mappings are full at configured 28.5 Hz, full at
training-peak 55.8186 Hz, and diagonal for the shared two-band case. Exact maps,
all 18 mapping/frequency/variant selection losses and provenance are saved in
`report.json` and `mapping_validation.csv`.

## Held-out prediction error

RMS values are in recorded WFS modal units. The positive dB numbers below mean
`10*log10(observed_band_power / prediction_error_band_power)`, using 4-second
Hann Welch blocks, 0.25-Hz bins, bands **27–30 and 54–58 Hz**. The observed and
error sequences are interpolated onto the same 2-kHz grid for PSD calculation
only. This is the error of an ideal subtraction at the target, not an observed
DM response or a closed-loop stability result. Negative dB indicates a larger
band error than the uncorrected observed signal.

| Method | RMS tip / tilt | 27–30 Hz dB tip / tilt | 54–58 Hz dB tip / tilt |
| --- | ---: | ---: | ---: |
| WFS-only Phase 13 AR | 0.009927 / 0.011913 | 9.94 / 9.00 | 11.05 / 14.38 |
| Phase 13 ARX, ADC0/1/2 | **0.009713 / 0.011441** | 10.40 / 9.73 | 11.16 / 15.55 |
| Same ARX, ADC0/1 only | 0.009741 / 0.011546 | 10.31 / 9.46 | 11.16 / 15.29 |
| WFS persistence | 0.012241 / 0.015079 | 14.90 / 14.89 | 9.10 / 9.08 |
| Actual harmonic, 28-Hz bank | 0.013913 / 0.018265 | **0.17 / 0.17** | -0.01 / 0.00 |
| Actual harmonic, 56-Hz bank | 0.013893 / 0.017921 | 0.00 / 0.03 | **0.20 / 1.27** |
| Actual harmonic, two banks + shared map | **0.013934 / 0.018263** | **0.01 / -0.03** | **-0.00 / 0.03** |
| Phase-calibration diagnostic ONLY | 0.013174 / 0.017299 | **8.22 / 6.79** | **1.86 / 4.55** |

The three-channel ARX reduces total RMS relative to WFS-only by about
**2.16%/3.96%**. Persistence is a useful extra narrowband comparator: a
broadband-optimized regression need not minimize 28-Hz error. The observed band
RMS is 0.004588/0.003457 at 28 Hz and 0.003114/0.006720 at 56 Hz. The shared-map
harmonic result has total NRMSE **99.998%/99.950%** and correlations
**0.0071/0.0357**. It is nearly a zero forecast in a broadband RMS sense; 100%
numerical readiness does not mean predictive usefulness.

### Phase-calibration diagnostic, separate from the current correction path

The Phase 12 exporter already exposes real/imaginary harmonic acceleration at
the target. A small diagnostic fits a real four-column map of those existing
quadratures (ADC0 at the lower band, ADC1 at the upper band) to earlier WFS
labels. This is equivalent to allowing different complex gains by band. It
uses the same earlier/validation/refit boundaries and unchanged causal C++
estimator, with no held-out fitting or phase/horizon sweep. **It is not the
current real-static pointing/modal map and supplies no shadow commands.**

Its improvement shows that the estimator has retained useful information that
the present static calibration path cannot exploit well. The prior measured
ADC/WFS phases near a quarter cycle make an instantaneous acceleration-to-angle
conversion alone insufficient. This does not establish a physical delay or
prove that phase calibration alone solves everything: finite estimator bandwidth,
multimode contamination, time variation and existing AO response remain possible
limitations, particularly at 56 Hz. No new prediction filter or production
recovery mechanism is introduced.

## File-only shadow correction and checks

The existing shadow component receives the selected shared-map two-bank
predictions. Its numerical convention is an **absolute contribution**, replaced
each query rather than accumulated. With a virtual identity response in WFS
coordinates,

`normalized_proposal = -gain * M * sum_f(x_f) / sigma_train`,

followed by the component's existing common-scalar bound. `sigma_train` is
0.01433603/0.01892305 recorded modal units, obtained entirely before testing.
The explicit numerical bounds are ±3 training WFS standard deviations per
axis. They are comparison bounds, not physical DM safety limits.

- Saved disabled manifest: **gain=0**, every proposal exactly zero.
- Separate counterfactual manifest: **gain=1**, proposed contribution opposes
  the mapped forecast in this virtual basis.
- Maximum absolute proposed values: **0.02149/0.24188** training WFS standard
  deviations; **zero clipped commands**.
- All invalid predictions produce zero, with one initial inhibited query and
  no invalid held-out queries. No previous command is reused on invalid input.
- Every target-minus-query is **exactly 1,000,000 ns**. Existing 2-ms freshness
  and 50-ms prediction-span checks remain active.

These tests establish arithmetic sign, gain, ordering, bounds and timing under
the stated identity-response assumptions. They do **not** establish optical DM
correction sign, DM modal units, a physical response or actual WFS suppression.
No calibrated DM command is claimed or emitted.

## Reproduction and products

Use the existing scientific Python environment (NumPy/SciPy/Pandas/Matplotlib;
`python` is Anaconda on this workstation) and Eigen from `pkg-config`:

```bash
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-harmonic-real telemetry-tools
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/accelFFCtrl-harmonic-real-mpl \
  python apps/accelFFCtrl/tests/validate_real_harmonic.py \
  "$HOME/data/phase13_20250415_083000_083500" \
  --replay /tmp/accelFFCtrl-harmonic-real/accelFFCtrlTelemetry_replay \
  --output /tmp/accelFFCtrl-harmonic-real-study
OPENBLAS_NUM_THREADS=1 \
  ACCELFFCTRL_TELEMETRY_REPLAY=/tmp/accelFFCtrl-harmonic-real/accelFFCtrlTelemetry_replay \
  make -C apps/accelFFCtrl/tests validation-test PYTHON=python
python -m unittest discover -s apps/accelSim/tests -p test_validate_psd.py -v
```

Products under `/tmp/accelFFCtrl-harmonic-real-study/`:

- `prediction_comparison.png`: observed target-time WFS versus all three main
  predictors, with the separate calibration diagnostic labeled.
- `residual_psds.png`: measured WFS and held-out error spectra at both bands.
- `normalized_shadow.png`: file-only virtual contributions, explicitly not DM
  commands; `normalized_shadow.csv` contains absolute acquisition/query/target
  stamps, source counter, validity/state, clipping scale, predicted WFS and
  normalized contributions for every query.
- `metrics.csv`, `mapping_validation.csv`, `availability_sensitivity.csv`,
  `report.json`, `input_sha256.json`, exact query times, gain-zero and separate
  gain-one manifests. Transient full C++ input/output CSVs are cleaned up.

All **45 controller Python tests plus four simulator PSD tests pass**. Four new
cases cover direct/swapped/full mapping, partition-safe integer target labels,
the real C++ kernel's normalized-unit/sign/bound/target/future-data behavior,
and the distinction between static maps and phase quadratures. The existing
C++ numerical/mock suite passes **67 cases / 3,677,233 assertions** with
ASan/UBSan. The replay executable also builds with ASan/UBSan and passes the
four new Python-driven cases with no sanitizer findings:

```bash
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-harmonic-real-asan \
  CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' telemetry-tools
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 OPENBLAS_NUM_THREADS=1 \
  ACCELFFCTRL_TELEMETRY_REPLAY=/tmp/accelFFCtrl-harmonic-real-asan/accelFFCtrlTelemetry_replay \
  python -m unittest discover -s apps/accelFFCtrl/tests -p test_validate_real_harmonic.py -v
```

The combined numerical/mock sanitizer suite can be rebuilt without application
or native-stream dependencies as follows. `ACCELSIM_MODEL_ONLY` excludes actual
application/native publication; DM fixture tests use preallocated host memory.

```bash
task_eigen_flags="$(pkg-config --cflags eigen3)"
c++ $task_eigen_flags -D_GNU_SOURCE -DACCELSIM_MODEL_ONLY \
  -O1 -g -std=c++20 -fno-fast-math -Wall -Wextra -pthread \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  apps/accelFFCtrl/tests/accelFFCtrlModel_test.cpp \
  apps/accelFFCtrl/tests/accelFFCtrlReplay_test.cpp \
  apps/accelFFCtrl/tests/accelFFCtrlInput_test.cpp \
  apps/accelFFCtrl/tests/accelFFCtrlValidation_test.cpp \
  apps/accelFFCtrl/tests/accelFFCtrlGap_test.cpp \
  apps/accelFFCtrl/tests/accelFFCtrlJitter_test.cpp \
  apps/accelFFCtrl/tests/accelFFCtrlCancellation_test.cpp \
  apps/accelFFCtrl/tests/accelFFCtrlShadow_test.cpp \
  apps/accelSim/tests/accelSim_test.cpp \
  apps/accelSim/tests/accelSimDisturbance_test.cpp \
  apps/accelSim/tests/accelSimDM_test.cpp tests/testMain.cpp \
  -o /tmp/accelFFCtrl-harmonic-real-combined-asan
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  /tmp/accelFFCtrl-harmonic-real-combined-asan
```

No native Linux/ImageStreamIO/full application qualification was performed.
Production controller, simulator, configurations and default-disabled gap policy
are unchanged. No instrument access, shared-memory write, commit or push occurred.

## Go/no-go and smallest next step

**Go:** normalized file-only simulation can exercise sign/timing/inhibition and
demonstrate this failure to cancel; its numerical plumbing is qualified offline.
**No-go:** describe the existing real-static-map algorithm as ready to provide
useful simultaneous 28/56-Hz cancellation on these data, or move these numbers
into physically calibrated DM commands.

The smallest supported next experiment is an explicitly approved, offline
**frequency-dependent sensor-to-WFS phase calibration using the already-available
causal harmonic quadratures**, followed by the same frozen held-out test and
synthetic response. It needs no generic prediction framework or controller
redesign. The diagnostic supports investigating that calibration; it does not
authorize integrating it into the live controller. Physical exposure/clock
semantics, ADC axis/scale calibration, WFS normalization, WFS-to-DM mapping,
optical correction sign, measured response and AO-loop interaction remain
separate gates before any physical DM interpretation or deployment.

## Follow-up: explicit offline phase calibration

The [phase-calibrated replay](PHASE_CALIBRATED_REPLAY.md) now implements the
proposed eight-coefficient output calibration using the unchanged estimator.
Earlier-only fits improve held-out spectra and generalize across later time
portions. This supersedes the calibration blocker for normalized offline
simulation, while retaining the physical timing/DM limitations. The numerical
results above remain the historical real-static-map/diagnostic comparison.
