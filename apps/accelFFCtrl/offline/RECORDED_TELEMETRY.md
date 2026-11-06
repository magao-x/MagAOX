# Phase 12: recorded-telemetry calibration and timing qualification

2026-11-06; macOS workstation; branch `parkertjohnson1/accelSim`, starting revision
`89bd483e`. **No simultaneous recorded accelerometer/WFS dataset was available.**
All numerical results below are explicitly synthetic. No physical prediction
accuracy, optical correction sign, DM response or physical optimum horizon has
been measured. No instrument connection, application launch, stream attachment,
configuration change or deployment was performed.

## Local inventory and evidence

The repository, including ignored scratch files, contains zero XRIF, `.bintel`,
`.binlog`, HDF5 or simultaneous ADC/WFS recording files. The inventory found nine
FITS, twelve CSV and two NPZ files: archived calibration/derived operators, not
time-resolved instrument measurements. Prior `/tmp/accelFFCtrl-*` studies contain
generated truth, input snapshots, timing logs and shadow proposals. They are
synthetic, including the 22 Phase 11 shadow logs with 2,640,000 records. This search
does not establish that no recordings exist elsewhere on the user's workstation.
No instrument host or remote dataset was searched.

| Available local evidence | Contents, units and provenance | Simultaneous measured ADC/WFS? |
| --- | --- | --- |
| `scratch/accelSim_dm_calib/{CMmodesDM.00,dmmask,dmmap}.fits` | Archived owner-provided tweeter modes/support/layout; conditional DM-pupil surface-micron mapping. Identities and geometry are in [INTEGRATION.md](INTEGRATION.md). | No; static calibration only. |
| `scratch/accelSim_offload_calib/*.fits` | Six archived offloader/basis/mask files, with distinct normalization and support. | No. |
| `scratch/{direct_tt_analysis,offload_analysis}` CSV/NPZ | Previously derived pupil projection and offload operators. | No; not temporal command telemetry. |
| `/tmp/accelFFCtrl-shadow-final-60s`, `linuxprep-timing`, prior validation outputs | Analytic/accelSim models, synthetic ADC/query/WFS/DM histories and timing faults. | Synthetic only. |
| Earlier live 602-s 4/8-kHz summaries supplied by the user | Aggregate drop/reset results, without per-frame ADC/WFS/clock data. | Insufficient for coherence, registration or phase calibration. |
| New `phase12-study/fixture` | Analytic two-tone ADC/WFS CSV and manifest, generated locally with seed 12342. | Explicit synthetic pair, described below. |

The tool hashes its manifest and input CSVs, copies provenance to each output,
and preserves integer timestamps before subtracting the epoch. These generated
large artifacts remain under `/tmp`, outside version control. No Python XRIF
decoder is installed locally; no dependency was installed in this phase.

## Recorder and timing source facts

- `apps/streamWriter/streamWriter.hpp`, frame copy around lines 1578–1615: XRIF
  timing has five UINT64 words per frame: `cnt0`, acquisition seconds/nanoseconds,
  write seconds/nanoseconds. Temporal arrays are used when available; otherwise
  global metadata are copied. Source-counter gaps can include recorder losses.
- That writer **substitutes its current CLOCK_REALTIME if acquisition seconds are
  zero**, and copies acquisition time into missing write time. The archive alone
  may therefore not prove genuine acquisition timing. Confirm the producer and
  recorder/version/log history; otherwise do not mark timing verified.
- `utils/xrif2fits/xrif2fits.hpp`, decoded timing and per-frame `meta_data.txt`
  export, retains counter and integer seconds/nanoseconds. Use these fields when
  preparing CSV, not floating epoch seconds or filename-derived uniform times.
  Native XRIF decoding/export remains an external dependency; this pipeline does
  not decode XRIF bytes or extract TT from camera pixels automatically.
- WFS TT extraction requires the actual reference, mask/reconstruction, modal
  indices, optical gain, units and loop state. Two arbitrary pixels or the first
  two unknown modes are not automatically physical tip/tilt. No local WFS image
  cube or calibrated TT sequence was supplied.
- `wooferTweeterRecon` uses WFS write time minus `1/fps` plus a configured offset,
  and DM acquisition time plus a different offset. Those source conventions are
  not a measured clock/optical delay to copy. Camera acquisition may represent an
  exposure endpoint; center conversion requires a verified camera convention and
  measured exposure, not an assumed half-frame offset.
- `camwfs_sync` arrival, simulator phase slewing, acquisition, publication and
  consumer receipt are distinct events. A semaphore receipt is not a WFS exposure
  center. Raw monotonic and realtime clocks require independently measured
  registration/bracket uncertainty before combining them.

## File interface and analysis

New `tests/qualify_telemetry.py` accepts one JSON manifest and two numeric CSVs:

```text
accel.csv: acquired_ns,written_ns,receipt_ns,counter,generation,adc_x,adc_y
wfs.csv:   acquired_ns,written_ns,counter,generation,tip,tilt
```

Clock/identity fields must be exact decimal integers, with valid positive times
and identities, ordered acquisition/write/receipt times and monotonic counters
within a generation. Duplicate/rollback or clock-step archives must be split into
valid runs before optical calibration. Generations and missing counter intervals
split spectral segments; they are never concatenated or filled as measurements.
ADC values are UINT16 codes in the existing twelve-bit range. The format retains
the controller's default raw channel order/signs; report their actual sensor
identity in the manifest. Nonfinite WFS/calibration data are rejected.

All times in one CSV must initially share that file's declared clock, including
receipt. Raw `CLOCK_MONOTONIC` receipts must first be registered to that clock
using recorded realtime/monotonic anchors; preserve those raw anchors separately
for `tests/analyze_timestamps.py`. The manifest requires independently supplied
offset, drift, integer anchor, uncertainty, common clock ID, registration provenance,
verified acquisition reference and WFS exposure-center reference. Registration is
`common = source + offset + round((source-anchor)*drift_ppm*1e-6)`.
The tool does not slide signals to maximize coherence or hide a clock offset in
a calibrated delay. Unknown timing blocks phase/horizon qualification.

The generated fixture manifest is a complete schema example. For a real dataset,
replace its evidence/provenance, files, channel identities, actual rates and clock
metadata, ADC calibration/provenance, WFS units and measurement kind/loop state,
estimator settings and explicit shadow C/K/provenance. Supported WFS kinds are
closed-loop residual, pseudo-open-loop, open-loop and synthetic disturbance.
Sensitivity units are explicitly `V/(m/s^2)`, not V/g. No factor 9.81 is inserted.
Raw images need a separately validated TT export. Unknown fields must not be copied
from the synthetic example and relabelled measured.

For replay, actual queries can be supplied as `query.file` with `query_ns` in
the registered common clock. Otherwise a declared rate/start-offset schedule is
generated and labelled **assumed replay cadence**, not measured live availability.
XRIF usually lacks actual controller receipt/query times: any constructed receipt
schedule must likewise be labelled assumed in `query.provenance`. Record the
uncertainty; without actual receipt/query captures the tool cannot qualify native
scheduling, transport latency or deployed prediction availability.

Equations and conventions:

```text
a = (ADC-zero)*volts_per_bit/sensitivity                 [m/s^2]
theta = -R*a_harmonic/(omega^2*L_eff)                    [rad]
Pxy = mean(conj(FFT(x))*FFT(y))                          [acceleration*WFS unit/Hz]
H = Pxy/Pxx                                            [WFS unit/(m/s^2)]
gamma_squared = |Pxy|^2/(Pxx*Pyy)
```

Spectral scoring first interpolates bounded acquisition jitter on each native
analysis grid, antialiases ADC with `scipy.signal.resample_poly`, then registers to
the WFS exposure-center grid. It trims 50 ms at each contiguous overlap edge.
Cadence deviations above 10% split analysis runs. This guard is an analysis
assumption, not a live jitter qualification. Default WFS rate must not exceed ADC
rate; rational conversion is supported. Two-second nonoverlapping Hann Welch
blocks give 0.5-Hz resolution, with at least eight blocks required. Transfer bins
are rejected below coherence 0.8 or the 5% independent-block null threshold,
whichever is stricter. Unqualified transfer bins are masked in files/plots.

Transfers are **paired-channel apparent responses**, not an identified coupled
two-input matrix. The tool also reports input cross-coherence and the normalized
two-input spectral condition ratio. Strongly coherent accelerometer axes can give
excellent paired coherence while making an independent 2x2 calibration impossible.
Closed-loop residual H includes the current AO sensitivity; it is not an open-loop
disturbance calibration. Do not reuse it after changing that loop without checking.

The reported Welch PSD/phase/magnitude interval formulas are conditional stationary
Gaussian approximations, not coverage guarantees for deterministic spectral lines.
They exclude quantization, interpolation, clock, mechanical-model and calibration
systematics. The known fixture delay biases below illustrate that distinction.
Phase-equivalent delay is reported only with a supplied signed reference and
retains integer-cycle ambiguity `1/f` (35.714/17.857 ms here). Passive phase alone
does not identify a unique propagation delay or a DM correction sign.

## Causal replay and held-out prediction

New `tests/accelFFCtrlTelemetry_replay.cpp` uses production `inputProcessor`,
`harmonicEstimator` and Phase 11 `shadowCommand`, entirely with owned depth-one
snapshots and files. It has no ImageStreamIO, application, INDI or live DM API.
It consumes each available record at its receipt time before each query; all
existing validation, cold-start, default reset-on-loss and freshness limits remain.
No resampled, future, WFS or disturbance-truth sample enters the estimator.

The first half of the recording fits paired complex H at 28/56 Hz. It is then fixed
for second-half scoring. Acceleration quadratures from the actual estimator at
the requested future target are mapped through this H and summed to forecast WFS
in its declared units. This empirical calibration can absorb sensor/mechanical/AO
response; it **does not validate the assumed 10-m lever arm or physical telescope
pointing frame**. Raw model pointing radians are also retained separately in CSV.

The tested horizons are 0, 0.5, 1, 1.75, 3, 5 and 10 ms beyond actual query time.
Scoring uses a common held-out query interval (start at midpoint+1 s, end trimmed
for the largest horizon), actual target times and adjacent valid WFS records.
It excludes WFS holes rather than interpolating across them. Whole-signal RMSE,
NRMSE, bias/correlation and coherent per-band amplitude/phase/component errors are
reported. Prediction coherence is unavailable for irregular queries or insufficient
continuous ready blocks; missing periods are never compressed into a uniform PSD.
WFS values between trusted exposure centers are linearly interpolated for scoring
only; these are not additional measured frames. At 2 kHz this introduces up to
about 0.39% sinusoidal amplitude error at 56 Hz, separate from predictor error.
Finite real WFS exposure would require its own measured timing/response model.

Replay CSV retains source acquisition, query, target, source counter/controller
generation, shadow reset epoch, validity/fault, pointing, absolute modal coefficients,
clip scalar, input processed/dropped/reset counts and optional synthetic response.
Times are normalized by integer subtraction to `1e9` ns; `report.json` retains both
origins for exact reconstruction. Shadow gain remains zero by default; disabled,
invalid and stale proposals are zero. Warmup at zero gain has state disabled but
`valid=0`; use that flag rather than treating command state as estimator readiness.
Loss counters cover the queried interval; trailing acquisitions after the final
query are outside that interval. These are archive replay statistics.

## Quantitative synthetic results (60 s)

Fixture: 8-kHz ADC / 2-kHz WFS/query, depth one, frequencies 28/56 Hz, lever 10 m,
existing ADC zero 2048, volts/count 0.0083799, sensitivity 1.02, memory 0.1 s.
Analytic X peak angles are 2e-7 / 8e-8 rad, Y 1.4e-7 / 1e-7 rad; phases are
explicit in the generator. ADC acceleration noise SD is 0.002 m/s² before rounding,
WFS noise SD 2e-9 rad; these are synthetic choices, not the previous simulator PSD
or measured sensor noise. WFS is delayed 1 ms with gains 1.2/0.8. Its independently
known timestamp correction is +700 us. WFS represents an instantaneous center,
not an integrated physical camera exposure. No AO feedback is applied.

| Frequency / axis | Coherence | Apparent magnitude [rad/(m/s²)] | Phase [deg] | Signed-reference delay [ms] |
| --- | ---: | ---: | ---: | ---: |
| 28 / X | 0.9999991 | 3.877633e-6 | 169.87514 | 1.004450 |
| 28 / Y | 0.9999973 | 2.583473e-6 | 169.86993 | 1.004967 |
| 56 / X | 0.9999994 | 9.685313e-7 | 159.85226 | 0.999392 |
| 56 / Y | 0.9999984 | 6.453941e-7 | 159.83191 | 1.000402 |

The 29 full-record Welch blocks recover the imposed 1-ms delay within 5 us.
Conditional phase interval half-widths are 0.0112–0.0241 degree. Their equivalent
delay widths of 0.55–2.39 us exclude the known quantization/systematic bias and
must not be advertised as physical delay confidence. Correlated two-tone sensor
axes do not establish an independent cross-axis matrix.
Input cross-coherence is 0.9999964 at 28 Hz and 0.9999996 at 56 Hz; normalized
two-input condition ratios are only 8.94e-7 and 9.00e-8. Thus this fixture cannot
identify an independent two-axis physical calibration despite its excellent
paired-channel held-out scores.

At 1.75-ms horizon, 57,939 held-out queries per axis give:

| Metric | X | Y |
| --- | ---: | ---: |
| Calibrated WFS RMSE [rad] | 4.05655e-9 | 3.45236e-9 |
| NRMSE | 2.2218% | 3.5527% |
| Bias [rad] | -8.8144e-12 | 1.7654e-12 |
| Correlation | 0.999801 | 0.999561 |
| 28-Hz amplitude / phase error | -0.3032% / +0.01389 deg | -0.2881% / +0.04411 deg |
| 56-Hz amplitude / phase error | -5.4577% / -0.21746 deg | -5.4465% / -0.20284 deg |
| Prediction/WFS coherence at either band | >0.999998 | >0.999996 |

The two independent harmonic banks retain their existing cross-band leakage,
especially visible in 56-Hz amplitude. High coherence does not guarantee correct
gain. These results concern a stationary, high-SNR fixture with a fitted passive
transfer, not the earlier stochastic/full-214-mode prediction study.

| Horizon [ms] | X / Y NRMSE |
| --- | ---: |
| 0 | 2.2903% / 3.6701% |
| 0.5 | 2.2930% / 3.6740% |
| 1 | 2.2903% / 3.6702% |
| 1.75 | 2.2218% / 3.5527% |
| 3 | 2.2271% / 3.5812% |
| 5 | 2.1055% / 3.4069% |
| 10 | 1.9327% / 3.1401% |

The empirical grid minimum is 10 ms. Cross-band leakage and its phase rotation
contribute to this stationary-fixture result; it is **not a measured physical
optimum** or justification to change the existing horizon. No physical horizon is
recommended. In particular, the earlier synthetic 1.75-ms actuation compensation
combined a chosen plant/delay/hold response; a WFS forecast minimum measures a
different quantity and cannot identify that physical command-path delay.

All nominal horizons have 99.4992% whole-record readiness, zero observed drops and
one initial attachment reset. At the 1.75-ms horizon, sample age is 100 us and
effective sample-to-target extrapolation is **1.85 ms**, not 1.95 or 1.75 ms.
Holding the training transfer fixed, bounded ±2-us and ±10-us acquisition-tag
errors give NRMSE 2.2219/3.5528% and 2.2224/3.5537%, zero drops, one initial reset
and about 99.5% readiness. These remain synthetic jitter tests.

An intentionally incorrect +1-ms WFS comparison offset increases NRMSE to
20.79/24.75%; -1 ms gives 20.66/24.51%. Phase errors change by approximately
10.08 degrees at 28 Hz and 20.16 degrees at 56 Hz per millisecond. The clock offset
must be registered independently rather than optimized away with the same data.

The optional `--synthetic-counterfactual` uses the Phase 11 proposal with explicit
gain 0.5, identity C, K=diag(1e6) coefficient/rad and bounds 1/1, plus the existing
`dmResponse` with an explicitly **synthetic** 1-ms/300-Hz response at 2-kHz query
cadence and diag(1.2e-6,0.8e-6) WFS rad/coefficient. This separate host-only run
reduces simulated variance by 72.31/70.91%; RMS changes 1.82644e-7→9.61099e-8 rad
and 9.72056e-8→5.24290e-8 rad. It uses no measured DM response and is not measured
suppression. The default run's proposals remain identically zero. A real manifest
cannot request this built-in analytic-fixture counterfactual.

## Calibration boundaries and remaining gates

| Quantity | Established here | Still required |
| --- | --- | --- |
| Accelerometer→WFS disturbance transfer | File/clock/schema and known synthetic paired H recovery; held-out scoring. | Simultaneous genuine ADC/WFS data with clock/exposure uncertainty, ADC response, axes, AO/POL context and sufficient excitation/conditioning. |
| WFS→DM modal map | Prior archived DM-pupil B/K algebra is preserved. | Current WFS interaction/reconstruction calibration, reference/units, registration and cross-axis response; archived pupil coordinates do not supply this. |
| Physical correction sign | None. Synthetic `-g*K*C*theta` and phase algebra are tested. | Separately authorized signed optical-response measurements/validated physical transfer. Passive ADC/WFS correlation alone is insufficient. |
| DM delay/bandwidth and AO interaction | Prior and current explicitly synthetic response models only. | Measured total source/combiner/driver/optical timing, response, AO sensitivity/headroom/offloader state and exclusive channel assignment. |

Matching Linux application/configuration/INDI builds and private native 4/8-kHz
ImageStreamIO restart/busy/counter/timestamp/depth-one checks remain outstanding;
macOS numerical tests do not satisfy them. The bounded high-rate capture and its
loss/overhead accounting remain separate implementation/review work. The next
useful input is an authorized human-provided local simultaneous recording with
raw counters, trusted acquisition/write/receipt/query timestamps and clock anchors,
WFS exposure/TT calibration, applied AO/DM histories and saturation/loop-state
context. Keep depth one, observation-only behavior, zero gain and
`allowSingleFrameGap=false`. No live command writer is authorized or implemented.

## Reproduction and verification

Run on the workstation, using an existing scientific environment (this workstation
uses `/opt/anaconda3/bin/python`; substitute the equivalent local environment):
Python 3.13.9, NumPy 2.3.5, SciPy 1.16.3, pandas 2.3.3 and Matplotlib 3.10.6
were used locally. The C++ numerical tooling uses Apple Clang and Eigen 5.0.1.

```bash
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-phase12 telemetry-test PYTHON=/opt/anaconda3/bin/python
MPLCONFIGDIR=/tmp/accelFFCtrl-phase12-mpl /opt/anaconda3/bin/python \
  apps/accelFFCtrl/tests/qualify_telemetry.py --synthetic 60 \
  --replay /tmp/accelFFCtrl-phase12/accelFFCtrlTelemetry_replay \
  --synthetic-counterfactual --output /tmp/accelFFCtrl-phase12-study

# Later: registered local CSVs and a reviewed manifest; no host/stream arguments.
/opt/anaconda3/bin/python apps/accelFFCtrl/tests/qualify_telemetry.py \
  --manifest /local/recording/manifest.json \
  --replay /tmp/accelFFCtrl-phase12/accelFFCtrlTelemetry_replay --output /tmp/accelFFCtrl-recorded-report

ACCELFFCTRL_TELEMETRY_REPLAY=/tmp/accelFFCtrl-phase12/accelFFCtrlTelemetry_replay \
  make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-phase12-regression test benchmark validation-test PYTHON=/opt/anaconda3/bin/python
/opt/anaconda3/bin/python -m unittest discover -s apps/accelSim/tests -p test_validate_psd.py -v
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-phase12-asan \
  CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' telemetry-tools
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  ACCELFFCTRL_TELEMETRY_REPLAY=/tmp/accelFFCtrl-phase12-asan/accelFFCtrlTelemetry_replay \
  /opt/anaconda3/bin/python -m unittest discover -s apps/accelFFCtrl/tests -p test_qualify_telemetry.py -v
```

Outputs include spectra/CSD/coherence/masked transfer CSVs, band intervals,
held-out prediction metrics, timestamp-offset and bounded-jitter sensitivities,
per-horizon shadow CSVs/provenance and seven figures. `spectra.png` uses separate
ADC/WFS PSD units; `prediction.png` compares at target timestamps; `horizons.png`,
`timing.png`, `offset_jitter.png`, `shadow.png` and optional counterfactual
`commands.png` retain explicit synthetic/host-only labels.

Eleven new deterministic Python/integrated C++ replay cases pass, including exact
epoch ns, PSD/CSD phase/delay, known sinusoidal transfer, antialiasing/fractional
registration, low-coherence rejection, target-based horizon selection, WFS holes,
ADC units, zero gain, clipping, drop/restart/invalid/stale inhibition and explicit
synthetic response. Existing suites pass: 33 controller cases/3,440,635 assertions;
67 combined simulator/controller ASan+UBSan cases/3,677,233 assertions; 32 controller
Python plus four simulator PSD tests. Allocation benchmarks remain allocation-free
with unchanged checksums. The new replay also passes ASan/UBSan unit cases and a
40-s full pipeline. The previous 76-run 60-s cancellation study passes, with all
18 original binary/summary artifacts byte-identical. No existing assertion or
production source/configuration was changed.
