# Offline native-journal input and EOF prediction validation

This experiment reads the completed 60-second observation as ordinary CSV/JSON
files. It never opens ImageStreamIO, imports a native transport, invokes a
MagAO-X executable, uses INDI, starts services or publishes commands. Results
must go to a new private directory outside the input tree; existing output
directories are refused. No installation or application rebuild is needed.

**Experimental context:** accelSim's corresponding disturbances were not applied
to the DM during this recording. ADC and WFS therefore have no demonstrated
physical coupling. This is an input/forecast software validation, not a control
performance experiment. The actual recording is not present on the development
workstation and has not been processed there. All local numerical results below
are explicitly synthetic.

## Scope and reused components

`tests/offline_eof.py` reuses `Series`, cadence/distribution helpers and the
`adc-wfs` ordinary-file verifier from the existing timing tools. It consumes
native `value0/value1`, raw ADC codes, counters, source identities, flags,
geometry and raw/receipt/availability timestamp columns. These are the real
ADC reader's recorded decoded values, not a new ADC conversion implementation.
UINT16 ADC pairs, FLOAT modal extent, camera layout, finite values and the two
selected modal indices are checked. Unknown clock, missing-origin, absent raw
timestamp and source-clock mismatch flags remain explicit timing limitations;
they do not erase a finite measurement in this observer-time experiment. Invalid
payload/layout/counter or observer receipt bounds invalidate its row. Regressing
observer availability aborts instead of sorting/relabeling data. Original
transport-verifier failures remain reported even when a subset fit can execute.

The CSV contains only the two selected modal entries, not the complete modal
vector. The summary must identify their indices. Tip/tilt labels follow the
experiment request for recorded columns 0/1; the report retains actual indices.
This does not independently qualify the calibrated basis, physical WFS units or
modal-to-actuator mapping. ADC calibration/profile provenance remains that of
the recorded native reader. No calibration constants are changed.

This is a standalone **EOF linear forecasting experiment**. It does not run
`ShadowController`, RLS/QR DDSPC learning, `CausalInputBuilder`, control
optimization, applied-command inference or a DM plant. The observation lacks
the exposure/applied-input certificates needed to exercise those production
seams truthfully. No shadow proposal is substituted for an applied command.

## Provisional association and timing

Only observer MONOTONIC availability is used for association. Raw acquisition
and publication timestamps are preserved verbatim and never subtracted across
unverified clock domains. No provenance manifest is silently enabled. The
separate existing timing audit remains the tool for verified clock hypotheses.

For each ADC/modal source-incarnation pair, the first ADC publication available
at a modal observer receipt establishes counter anchor `(n0,k0)`. At later modal
counter `k`, the provisional group ends at

    n(k) = n0 + 4 * (k - k0 - L),       L = 1 WFS frame by default.

The mean of original ADC counters `n(k)-3 ... n(k)` supplies two acceleration
features at that WFS row. All four must exist, be valid in the same incarnation,
and have frontend availability no later than the modal observer receipt. The
latest group's receipt age must be <=1500 us by default. That budget includes
the intentional one-frame lag; it is not a sensor freshness bound for a 500-us
control cycle. No timestamp, missing sample or acceleration value is invented.

This imposes a 4:1 counter rule, not proof of phase lock. Initial counter residue
comes from an observer-time anchor, not a camera exposure or hardware trigger.
Counter holes remain holes and do not rephase later groups. Source restarts get
new anchors and force dependent histories to warm up again. The source-clock
mean acquisition timestamp is retained as an integer centroid; it is not a
qualified WFS effective timestamp.

Diagnostics separately count ADC records actually delivered in each interval
between modal observer receipts. Counts can be 0/8 during batching even when
four counter-labelled records exist. `associations.csv` includes raw acquisition,
publication, decoder receipt and frontend availability for group endpoints,
modal records and a preceding-camera receipt **candidate**. That candidate
does not qualify modal-camera ancestry and does not define an exposure window.
Statuses distinguish warmup, absent ADC, missing counters, invalid samples,
delivery after the analysis cutoff and stale receipt.

An early acquisition time never makes a late-delivered sample usable. Conversely,
availability by this observer cutoff establishes no claim about WFS processing
deadlines, historical production-controller availability or synco input cutoffs.
Real AD7689 one-in-four locking requires its own reference/event/clock evidence.
Future tweeter output remains through dmtweeter-synco, independently qualified.

## EOF equations and chronological evaluation

With `H=100` and newest-first histories:

    xW(k) = [y(k), ..., y(k-H+1)]               (200 columns)
    xA(k) = [abar(k), ..., abar(k-H+1)]         (200 columns)
    xWA(k) = [xW(k), xA(k)]                    (400 columns)
    target(k) = [y(k+1), y(k+2), y(k+4), y(k+8)] (8 columns)

Each `y` and `abar` has two components. The accelerometer history is a 2-kHz
sequence of four-sample means, not an 8-kHz output predictor. H spans a nominal
50 ms. The outputs forecast the recorded WFS residual under whatever control
was operating during the recording; no unknown future command is invented.

Raw modal rows define fixed 60% training, 20% validation and 20% test partitions.
Entire feature/target footprints must remain inside their own partition. The
maximum horizon is purged for every model. The oldest delayed-control history
and one extra timing-control interval are also purged, by default 2100 rows at
each partition start. Invalid inputs, gaps, source changes and modal receipt
gaps >1 ms break dependent histories. Future ADC validity never selects current
origins. All models, controls and horizons use exactly the same surviving rows.
Thus the WFS-only result is explicitly a **common-coverage baseline**, not a
prediction of what its independent full-data operational performance would be.

Training-only means and standard deviations define standardized X and centered
target Y. In bounded batches, form

    G = X' X / N,       C = X' Y / N,       G = V Lambda V'.
    B(alpha) = Vr (Lambda_r + alpha * lambda_max * I)^-1 Vr' C.

Retain eigenvalues above `lambda_max * max(1e-12, eps * number_of_columns)`.
This is covariance EOF decomposition with ridge; it squares singular-value
conditioning, so negative covariance eigenvalues, retained rank/condition and
finite predictions are checked/reported. No unregularized normal-equation
inverse is taken. Constant columns have unit fallback scaling and contribute
zero centered signal. The fixed ridge-fraction grid is `1e-6,1e-4,0.01,1`.
Validation MSE selects alpha separately per horizon/axis, then coefficients and
scalers freeze. Test labels never influence fits, normalization, association
parameters or hyperparameter selection. The tool does not refit on validation.

Metrics give MSE, RMSE, error variance/bias and forecast-error power in 27-30 Hz
and 54-58 Hz bands for tip/tilt separately. Persistence is an additional baseline.
These residuals are `prediction - held-out WFS`, **not closed-loop residuals
after applying corrections**. Welch PSDs use nominal 2-kHz modal cadence, Hann
windows, 50% overlap and per-window constant detrending. Missing-data gaps are
never compressed: only complete windows inside contiguous test spans contribute,
weighted by their number of Welch windows. Unknown exposure timing remains a
limitation of the spectral frequency/time interpretation.
Bands with no resolved frequency bins are unavailable, not reported as zero
power. Weight norms for WFS and extra feature blocks show whether a fitted
model uses its extra inputs; nonzero weights alone do not establish usefulness.

## Negative controls and apparent improvements

Alongside WFS-only, ADC-only and combined fits, the default experiment includes:

- WFS plus ADC delayed by 1000 and 2000 WFS frames. Delays use only older samples
  inside the same partition, with no circular wrap or future-value shuffle.
  Stationary sinusoids can remain informative after delay; these are diagnostic
  controls, not guaranteed null realizations.
- Five WFS-plus-independent-sensor fits, seeded from training-only ADC mean,
  variance and lag-one correlation. AR(1) colored Gaussian controls are not exact
  surrogates for the simulator's narrow spectral lines. No future measured ADC
  is used to synthesize them.
- WFS plus timing histories (modal receipt-interval deviation and paired ADC
  receipt age), to expose possible information associated with delivery timing.

`chronological_blocks.csv` reports four ordered test-block gains. Blocks from one
recording are not independent disturbance realizations. The independent-control
exceedance fraction in JSON is explicitly descriptive, **not a p-value**: these
surrogates are not guaranteed exchangeable with the measured acceleration.
Eight horizon/axis comparisons and model selection increase chance findings.

For this uncoupled observation, small gains, a preferred ADC phase, or apparent
28/56-Hz prediction do not demonstrate physical accelerometer information.
Inspect gains across blocks, delayed/independent/timing controls, ridge choices,
counter/receipt artifacts and source quality. A large gain should trigger a
leakage/association/chance-correlation investigation, not a recommendation to
enable assisted control. Do not tune phase, lag, horizon or regularization on
this test split and present its revised score as an untouched held-out result.
A later repeat recording is needed to assess reproducibility.

## Commands and artifacts

Use an existing scientific Python environment with NumPy, SciPy, Matplotlib and
threadpoolctl; minimum versions are in `tests/requirements-eof.txt`. No dependency
is installed automatically. If necessary, create a separate user environment on
the development machine; do not change system or operational Python packages.

Local tests (the second command needs only the existing standard-library tools):

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.offline check-offline-eof \
  OFFLINE_EOF_PYTHON=/opt/anaconda3/bin/python
PYTHONDONTWRITEBYTECODE=1 make -C apps/loPredCtrlAcc -f tests/Makefile.offline check-native-timing
```

Operator-only analysis of the existing exao2 recording; no native app or producer
needs to run. Use the reviewed tools in a user checkout, never an installed app:

```sh
export PYTHONDONTWRITEBYTECODE=1
DDSPC_PYTHON=python3  # Set to an existing interpreter containing the dependencies.
DDSPC_REPO="$HOME/source/MagAOX-DDSPC"
DDSPC_OBS="/home/parkertjohnson1/ddspc-native-observe.9UlnxD/observations"
DDSPC_PARENT="$(mktemp -d "$HOME/ddspc-native-eof.XXXXXX")"
"$DDSPC_PYTHON" "$DDSPC_REPO/apps/loPredCtrlAcc/tests/offline_eof.py" \
  "$DDSPC_OBS" --output "$DDSPC_PARENT/analysis" --data-kind native-uncoupled \
  --history 100 --horizons 1 2 4 8
```

The output child must not exist beforehand. Default seed, ridge grid, controls,
lag, nominal rate and thread count are fixed and recorded. One BLAS thread and
bounded feature batches are used by default. Load, fit, validation selection,
test prediction and total wall times are reported separately. These are unpaced
offline timings, not 2-kHz/8-kHz deadline or synco scheduling qualifications.

| Artifact | Contents |
| --- | --- |
| `input_quality.json` | Scoped transport verification, dimensions/value/clock declarations, original rates and invalid counts. |
| `association_quality.json`, `associations.csv` | Provisional anchor/coverage/status reports and original timestamp/counter evidence. |
| `report.json` | Splits, EOF ranks/conditioning/ridge choices, controls, timings, software/BLAS versions, source/input hashes and permanently false native permissions. |
| `metrics.csv`, `chronological_blocks.csv` | Per-axis/horizon common-row forecast scores and ordered test-block comparisons. |
| `psd.csv`, `forecast_error_psd.png/.pdf` | Measured WFS and held-out forecast-error spectra on common contiguous spans. |
| `prediction_rmse.png`, `negative_controls.png`, `input_diagnostics.png` | Prediction comparisons, control gains and input/receipt diagnostics. |
| `predictions.npz`, `models.npz` | Frozen predictions/labels, exact uint64 origin/target counters, observer times, weights and training scalers; load with `allow_pickle=False`. |
| `failure.json` | Explicit failure if coverage or fitting fails after diagnostic files have been written. No successful prediction report is fabricated. |

Input CSV/summary SHA256 hashes are checked again before successful reporting.
The output includes a private Matplotlib cache. Existing results are never
overwritten. An input/parse error exits before creating results; insufficient
common coverage leaves diagnostics and a failing exit status. Inspect validity
fractions and every transport failure even when a subset fit executes.

## Qualification boundaries

| Check | Meaning of success | Still unqualified |
| --- | --- | --- |
| Loader/continuity | Recorded values and exact integer identities/timestamps survive extraction; corrupt/missing rows are diagnosed. | Producer host/build/calibration provenance absent from the journal. |
| Provisional association | No admitted group is delivered after its modal observer cutoff; missing groups are excluded. | Hardware 4:1 locking, modal-camera ancestry, exposure timing and actual processing availability. |
| EOF fitting/prediction | Purged held-out forecasts are finite, with fixed training transformations and validation-only tuning. | Native DDSPC/RLS training, readiness/fallback, applied-command evidence and control optimization. |
| Spectral/negative-control analysis | Forecast errors and comparison artifacts are reproducible from the same inputs/settings. | Physical coupling, independent-realization improvement, on-sky benefit and closed-loop suppression. |
| Offline timing | Measured local file/numerical throughput. | Real-time inference/training latency, input-ready deadlines and DM/synco feasibility. |

No successful analysis authorizes native training, physical synchronization,
applied-input qualification or DM output. The existing ADC-only/native
observation executables and their compile-time protections are unchanged.

## Local synthetic validation

The dedicated target passes 24 deterministic tests, including direct thin-SVD
comparison, perturbation of frozen ADC features, held-out-label/scaler invariance,
independent-sensor controls, missing raw acquisition metadata, counter gaps,
restart warmup, jitter/batched receipt, late/stale groups, rank-deficient fits,
gapped PSDs and read-only CLI artifacts/failure paths. The unchanged native
timing target passes 39 tests. No C++ rebuild or sanitizer qualification is
claimed for Python-only analysis tools.

Two separate 60-second synthetic journals each contain 480,018 ADC and 120,000
modal/camera samples. H100, horizons1/2/4/8, default controls, 60/20/20 splits and
one BLAS thread give 69,892/21,892/21,892 common train/validation/test origins.
There is one expected startup group without history, giving 99.99917% provisional
group coverage. Both synthetic transport checks pass and input hashes remain
unchanged. Each primary fit has rank 200/200/400 for WFS/ADC/combined.

On this development Mac (Python3.13, NumPy2.3.5, SciPy1.16.3, Matplotlib3.10.6),
the reviewed independent/correlated runs took 34.63/34.73 s inside the script,
including plots and file I/O. Association, all eleven fits and evaluation took
8.50/8.47 s; loading plus transport verification took 8.68/8.70 s. These are
unpaced offline wall times, not real-time frame/slot deadline measurements.

The positive fixture uses a causal ten-WFS-frame delay from sensor innovations
to WFS measurements, with independent measurement noise. It intentionally
contains predictive information at all four horizons. The negative fixture uses
independent WFS innovations. These results concern the fixtures, not accelSim's
optical response or the actual observation:

| Horizon | Axis | Independent WFS-only RMSE | Independent combined RMSE | Independent MSE gain | Coupled WFS-only RMSE | Coupled combined RMSE | Coupled MSE gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Tip | 0.99719 | 0.99864 | -0.2907% | 0.93160 | 0.15070 | 97.3834% |
| 1 | Tilt | 0.99727 | 0.99934 | -0.4148% | 0.72670 | 0.15107 | 95.6783% |
| 2 | Tip | 1.21855 | 1.22055 | -0.3277% | 1.11337 | 0.16238 | 97.8729% |
| 2 | Tilt | 1.22055 | 1.22239 | -0.3015% | 0.86877 | 0.16301 | 96.4794% |
| 4 | Tip | 1.35023 | 1.35230 | -0.3066% | 1.16761 | 0.16434 | 98.0189% |
| 4 | Tilt | 1.36350 | 1.36507 | -0.2302% | 0.91107 | 0.16498 | 96.7210% |
| 8 | Tip | 1.39229 | 1.39243 | -0.0214% | 1.17098 | 0.16431 | 98.0311% |
| 8 | Tilt | 1.40403 | 1.40408 | -0.0074% | 0.91330 | 0.16507 | 96.7334% |

RMSE units are arbitrary synthetic modal units. Negative gain means worse.
Full tables also retain ADC-only, persistence and every negative control's
per-axis errors and band powers. Artifacts reside in the private development
directory `/private/tmp/ddspc-offline-eof-2j0a8z05/`; the `reviewed-independent`
and `reviewed-correlated` children bind to the final analysis source hashes.
They are not tracked numerical results or the instrument recording.

To reproduce these ordinary-file fixtures and experiments with the same
scientific interpreter, from a personal development machine:

```sh
export PYTHONDONTWRITEBYTECODE=1
export DDSPC_REPO="$HOME/source/MagAOX-DDSPC"
export DDSPC_SYNTH_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ddspc-eof-synthetic.XXXXXX")"
DDSPC_PYTHON=/opt/anaconda3/bin/python  # Or another existing scientific interpreter.
"$DDSPC_PYTHON" - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(os.environ['DDSPC_REPO'])/'apps/loPredCtrlAcc/tests'))
from offline_eof_test import synthetic_records, write_measurements
root = Path(os.environ['DDSPC_SYNTH_ROOT'])
for name, coupled, seed in [('independent', False, 917), ('correlated', True, 813)]:
    folder = root/('input-'+name)
    folder.mkdir()
    rows = synthetic_records(frames=120000, coupled=coupled, seed=seed)
    write_measurements(folder, rows)
    del rows
PY
for DDSPC_CASE in independent correlated; do
  "$DDSPC_PYTHON" "$DDSPC_REPO/apps/loPredCtrlAcc/tests/offline_eof.py" \
    "$DDSPC_SYNTH_ROOT/input-$DDSPC_CASE" \
    --output "$DDSPC_SYNTH_ROOT/result-$DDSPC_CASE" \
    --data-kind "synthetic-$DDSPC_CASE"
done
```

The actual native observation's scores, association coverage and information
content remain pending its operator-run analysis. No local synthetic score is
an assertion about that recording.
