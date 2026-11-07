# Channels 0/1, late suppression loss, and simulation transport

**Subsequent fixed-fit prediction audit:** [prediction, coherence/phase and conditional information](prediction_information.md)
finds the dominant loss before correction, at held-out28Hz. ADC has useful
high-SNR information but the short calibration fit does not generalize; no
timing/indexing correction is supported. This original adaptive/calibrated report
and its manifests remain the historical record of the earlier iteration.

2026-11-06, personal Mac `ASTR-LT-428.local`, arm64 macOS; Apple Clang 21.0.0,
Homebrew GCC 15.2.0, Python 3.13.9. **Simulation only. No native applications,
instrument connections, actual shared-memory images, or physical DM commands ran.**
`~/source/MagAOX` was inspected read-only; unrelated `accelFFCtrl` work is untouched.

## Result

Late suppression loss is reproducible and is principally associated with continued
closed-loop identification of the DM response. No command-history, sign, delay,
or QR arithmetic bug was found in this focused investigation. The existing
adaptive experiment is preserved. A separately labeled **calibrated-fit** experiment
uses the existing training pause after the common 1.2-s calibration and retains
all fresh-record validity, rank, finite, and snapshot-age checks. It substantially
reduces the late loss. This is an offline experiment for this fixed synthetic
plant, not a production controller fix or a general adaptation policy.

**Accelerometer assistance does not outperform standard DDSPC in the calibrated-fit
comparison.** The earlier adaptive 2.67% / 12.31% RMS benefits remain reproducible,
but both adaptive controllers deteriorate. Do not present their aggregate benefit
as sustained superior suppression, or the earlier idealized 21.17% result as an
ADC/aoSim result.

One command from `~/source/MagAOX-DDSPC` reproduces both policies, both amplitudes,
all three trajectories, plots and the post-hoc diagnostic:

```sh
python apps/loPredCtrlAcc/tests/simulation_diagnose.py --run
```

NumPy/SciPy/Matplotlib and the existing C++ Eigen/mxlib dependencies are required;
no runtime installation or instrument service is used. Clang results are in
[comparison.csv](channel01_validation/comparison.csv),
[diagnosis.json](channel01_validation/diagnosis.json) and source/binary manifests
under [adaptive](channel01_validation/adaptive/manifest.json) and
[calibrated](channel01_validation/calibrated/manifest.json).
[Validation totals, source preservation and exact adaptive-sequence comparison](channel01_validation/validation.json)
record the unchanged outputs for all 31,200 rows per amplitude/compiler.
The virtual experiment is 2-kHz WFS / 8-kHz ADC, not wall-clock pacing.

## Three-case comparison

All values below are two-axis RMS in the existing source-derived normalized WFS
units. Each comparison shares exact source/ADC arrays, calibration, WFS noise,
plant, timestamps and scoring frames. Score includes held/invalid-prediction frames:
3.5 s from 0.5 s after the common calibration; plots also expose the control start.
The 100× case multiplies producer mode amplitudes only; ADC noise/calibration stay
fixed. Each amplitude has its own explicit normalization; it is not a physical
instrument calibration or an absolute-RMS comparison with the earlier harness.

| Training policy | Source | No DDSPC | Standard | Assisted, ADC 0/1 | Assisted RMS reduction vs standard |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adaptive (unchanged) | 1× | 1.242141 | .660365 | .642741 | +2.67% |
| Adaptive (unchanged) | 100× | 1.242141 | .660552 | .579243 | +12.31% |
| Calibrated fit | 1× | 1.242141 | .108602 | .109891 | **−1.19%** |
| Calibrated fit | 100× | 1.242141 | .108533 | .117588 | **−8.34%** |

Variance and integrated-line PSD reductions relative to the same no-control case:

| Policy / source | Standard variance reduction | Assisted variance reduction | Standard 28 / 56 Hz dB | Assisted 28 / 56 Hz dB |
| --- | ---: | ---: | ---: | ---: |
| Adaptive / 1× | 71.74% | 73.22% | 8.69 / 11.32 | 9.09 / 11.37 |
| Adaptive / 100× | 71.72% | 78.25% | 8.72 / 11.46 | 10.94 / 12.71 |
| Calibrated / 1× | 99.24% | 99.22% | 22.30 / 16.00 | 22.17 / 16.74 |
| Calibrated / 100× | 99.24% | 99.10% | 22.31 / 16.03 | 21.06 / 19.51 |

Full time series and PSD comparisons, including held frames:

- [Residual RMS versus time, both policies/amplitudes](channel01_validation/rms_vs_time.png)
  ([PDF](channel01_validation/rms_vs_time.pdf)).
- [Adaptive reference time series](channel01_validation/adaptive/scale_1/full_time_series.png),
  [PSD](channel01_validation/adaptive/scale_1/residual_psd.png).
- [Calibrated reference time series](channel01_validation/calibrated/scale_1/full_time_series.png),
  [PSD](channel01_validation/calibrated/scale_1/residual_psd.png).
- [Adaptive 100× time series](channel01_validation/adaptive/scale_100/full_time_series.png),
  [PSD](channel01_validation/adaptive/scale_100/residual_psd.png).
- [Calibrated 100× time series](channel01_validation/calibrated/scale_100/full_time_series.png),
  [PSD](channel01_validation/calibrated/scale_100/residual_psd.png).

GCC repeats give adaptive +2.66% / +12.77% and calibrated **−.25% / −3.75%**
assistance reductions. Calibrated reference standard/assisted RMS is .075689/.075881;
100× is .075602/.078439. The C++ standard-normal generator produces different
realizations across standard libraries despite the same seed. Each compiler's
three cases share identical ADC/disturbance bytes; cross-compiler aggregate RMS
is not an exact-regression test. [GNU comparison](channel01_validation/gcc_comparison.csv)
and [diagnosis](channel01_validation/gcc_diagnosis.json) retain that repeat.

## Why suppression weakens

`sim_bridge.cpp::Controller` retains QR forgetting factor `float(.9995)` and the
established reference law. Common calibration has independent input RMS .3;
afterward the existing independent probe is .005, 60 times smaller. After 8,000
updates only **.018294** of calibration information remains. Future-command
regressors are now closed-loop corrections correlated with disturbed/noisy WFS
targets. Full rank and small conditional prediction error do not establish an
unbiased causal DM transfer function suitable for counterfactual control.

`simulation_diagnose.py::batch_reference()` reconstructs the exact newest-first
regressors from recorded **consumed** commands (6 WFS, 5 command, optional 5 ADC
history samples, 3 future commands), with 28/38 features and six target values.
An independent double-precision discounted least-squares solve includes the same
initial .001 information and forgetting factor. For the known two-frame response,
`C` is a 6×6 matrix with I blocks at `(rows 0:2, cols 2:4)` and
`(rows 2:4, cols 4:6)`; all remaining entries are zero. This target is a diagnostic
only, never used to fit or choose corrections.

| Clang 100×, adaptive | End calibration | Mid-run, frame 5400 | End, frame 10399 |
| --- | ---: | ---: | ---: |
| Standard `||C−Cplant||F` | .018107 | .305461 | .878470 |
| Assisted `||C−Cplant||F` | .018425 | .245258 | .848443 |
| Standard conditional fit RMSE | .055865 | .055206 | .053547 |
| Assisted conditional fit RMSE | .048392 | .049327 | .049078 |

Independent double LS agrees with those C-error values within 2.3e−6. Column-scaled
design condition estimates are 126–195 standard and 312–471 assisted; rank remains
28/38. This is evidence for **closed-loop identification bias / weak independent
excitation**, not loss of rank or unstable QR arithmetic. The evidence does not
prove which bias mechanism dominates. A single diagnostic contrast continuing the
existing .3 independent probe ends at C errors .099991/.084533, supporting that
interpretation. This contrast is simulation-only and is not an instrument-excitation
recommendation or a proposed controller framework.

At 100×, adaptive standard/assisted RMS rises from .18557/.15365 in the first
scored second to **1.09975/.98930** in the last. Calibrated fit gives
.08188/.08611 → **.12100/.12927**. Source configuration is unchanged throughout;
its realized no-control RMS actually falls 1.49287 → 1.21567. Stationary stochastic
excitation still has finite-record amplitude changes; it cannot explain the much
larger adaptive residual increase. No reconfiguration or sample-rate change occurs.

`simulation_diagnose.py::summarize()` finds exactly zero request/consumed mismatch
and zero error in `dm_state[k+2] == consumed[k]` in all reported trajectories.
Correction and disturbance use separate calls to the actual
`aoSim.update_dm()` and sum positively in `aoSim.update_wfs()`. The optimal negative
correction sign is independently checked by cancellation/impulse tests. Adaptive
command peaks are about 3.14; calibrated peaks are at most 3.37, below the ±4
normalized limit; clipped fraction is zero. WFS is measured before the new
decision; training accepts the actual value read by the simulator, afterward.
Calculated outputs are never substituted for consumed telemetry.

`--training-end 2400` is an explicit, optional test-only build flag. It uses
`ShadowController::training(false)` through four guarded lines in the **offline**
worker; normal builds do not include the guard's body. Coefficient version stays
2392 while evidence sequence advances to 10398. Snapshot age here means age of
**current validation evidence**, not time since coefficients last changed. Fresh
inputs, current rank, prediction error, finite and age gates still run; unavailable
predictions hold the last consumed input. No validity threshold is relaxed, no
snapshot is made valid merely by stamping a new timestamp. Default adaptive and
all production/Legacy paths are unchanged. No justified algebra/indexing correction
was identified; the new option is not a hidden behavior change.

## Intended ADC pair: channels 0 and 1

Source references in `~/source/MagAOX`:

- `apps/accelSim/accelSimModel.hpp::model::step()` (lines 442–461): axis loop 0/1
  sums the same stochastic modal displacement into `m_pointing[axis]` and adds
  `−(2πf)^2 × leverArm × arcsecToRad × pointing` to `m_acceleration[axis]` before
  independent sensor noise and ADC encoding. Channel 0 is **simulated X acceleration**;
  channel 1 is **simulated Y acceleration**. They are not angular tip/tilt samples.
- `adcCounts()` (line 367) encodes `round(2048 + a × 1.02/.0083799)`, clipped to
  0–4095, stored as UINT16. `accelSim::configureAcquisition()` and
  `loadImageIntoStream()` (lines 616, 941) publish width2/height1 and memcpy that
  same ordered pair. The default stream is `acc000`; the experiment explicitly
  selects 8 kHz (code default is 6 kHz).
- `apps/mcp3208Ctrl/mcp3208Ctrl.hpp` (lines 1400, 1604, 1654, 1213): hardware
  values are read via channel index `i`, then memcpy'd in order. Thus selecting
  words0/1 selects ADC inputs0/1. Real reads are sequential; the simulator pair is
  simultaneous. Physical sensor mounting, WFS registration, skew and calibration
  remain unverified; no on-sky tip/tilt interpretation is inferred.

The bridge receives **only four actual UINT16 pairs** per WFS frame. It decodes
`a=(word−2048)×.0083799/1.02`, normalizes by the declared source scale, and averages
the four past pairs. The first is 500 μs before the WFS decision and the last
125 μs before; explicit availability is sample time+1 μs. The injected source
pointing from these same steps enters only the plant with its stated delay;
it never enters controller inputs. The existing general D matrix may mix both
sensor channels into both WFS modes. A 56-Hz displacement produces four times the
acceleration of a 28-Hz displacement with equal amplitude; one scalar conversion
from this two-frequency ADC mixture to pointing would be incorrect.

Reference acceleration RMS is .002/.001 m/s² in the two selected mode populations.
One ADC step is **.00821559 m/s²** and 8-kHz sensor-noise sigma is **.00632456 m/s²**.
The four-sample average has approximate signal/noise ratio −3.6 dB at reference
amplitude and 36.4 dB at 100×, treating independent sensor/quantization noise in
quadrature; quantization with dither is not exactly independent. No rails were hit.
Low signal level explains limited reference sensor usefulness, but does not explain
the large standard-model late loss or prove benefit at high amplitude. The longer
fit/gain error and short causal histories still matter. No ideal acceleration
replaces the ADC: a separate audit-only API returns the last **noisy** acceleration
solely to test rounding, order and the half-LSB bound. The simulation/controller
never calls that API.

## Minimum shared-memory step

`sim_shmim.py` is a small optional transport for this existing simulation, using
`ImageStreamIOWrap.Image` owners and distinct `magaox.shmim.Image` readers/writers.
It does not redesign the source, controller or plant. Fresh names have the form
`ddspc_sim_<pid>_<random>_<role>`; there is no arbitrary stream selector or attachment
to existing streams, and only owned unchanged-inode images are destroyed on exit.
Geometry, exact dtype, finite payloads, busy/counter checks and packet certificates
fail closed on mismatch/replacement. This deterministic adapter uses no blocking
consumer waits. It is not a cross-process lock or real-time transport guarantee.

| Role suffix | Width × height / type | Meaning |
| --- | --- | --- |
| `adc01` | 2×1 UINT16 | Generated raw channel0/1, never an ideal substitute |
| `pointing` | 2×1 FLOAT32 | Same-sample source displacement; plant only |
| `source_tag` | 8×1 UINT64 | epoch, sample ID, sample ns, available ns, WFS frame, ADC cnt0, pointing cnt0, first ADC channel(0) |
| `disturbance` | 2×1 FLOAT32 | Separate source injection into aoSim delay method |
| `correction` | 2×1 FLOAT32 | DDSPC correction input to another aoSim delay method |
| `wfs` | 2×1 FLOAT32 | Their resulting residual plus sensor noise |
| `consumed` | 2×1 FLOAT32 | Actual corrective input read by aoSim, fed back to training |
| `state` | 2×1 FLOAT32 | Combined delayed disturbance+corrective state at this WFS frame |
| `frame_tag` | 8×1 UINT64 | epoch, frame, time ns, consumed cnt0, state cnt0, WFS cnt0, last source ID, decision delay(2 frames) |

Tags are committed after their payloads and verified against counters. No command
stream from the instrument is opened. The **native transport has not run on this
Mac**: `ImageStreamIOWrap`, `magaox`, `purepyindi2`, `xconf` are absent. Fake-runtime
contract tests cover native-style shapes/counters, ordered ADC, packet timing,
busy state, replacement/ownership and full trace equivalence. These tests do not
establish real ImageStreamIO ABI, semaphore, Linux or scheduling compatibility.

On an **isolated non-instrument MagAO-X development host with its normal runtime
already available**, from the DDSPC root, these are the exact transport-check
commands (do not install/start the native apps):

```sh
hostname  # must be a development workstation, never exao*
python -c 'import ImageStreamIOWrap; from magaox.shmim import Image'
python apps/loPredCtrlAcc/tests/simulation_demo.py \
  --magaox-repo ~/source/MagAOX --accel-channels 0 1 \
  --training-end 2400 --transport shmim \
  --output apps/loPredCtrlAcc/build/channel01-shmim
```

Then repeat without `--training-end 2400` to retain the adaptive contrast. Compare
the generated manifest, residual metrics and impulse tests to memory transport.
This runs the numerical components through **real shmim boundaries in one process**;
it does not claim to launch native accelSim/aoSim/loPredCtrlAcc applications. The
namespace is printed in the manifest and cleaned up at exit. No production
configuration or existing stream is needed or changed.

### Remaining native application wiring

1. `accelSim::frameGrabberPostPublish()` currently has an optional physical-surface
   injector, not this two-modal packet interface. Add an explicit simulation-only
   same-sample ADC/pointing/tag publisher to private streams. Do not point its
   existing 50×50 surface-micron writer at a modal stream or physical DM channel.
2. Native `aoSim.setup()/loop()` currently hardcodes `aoSim_dm/wfs/disturbance` and
   its own sine. Make simulation names local/configurable, use same-source
   disturbance and separate correction channels, skip that unrelated sine, and
   publish consumed input, delayed optical state and WFS exposure/frame tags.
3. `loPredCtrlAcc` still rejects UINT16 on its old float sensor seam and invalidates
   raw shadow WFS callbacks lacking timing/provenance. Supply a read-only raw0/1
   decoder, causal window and validated WFS/consumed-input adapters to
   `processShadowWfs()` / `processShadowApplied()` (lines 941/951) and the existing
   shadow input builder. Preserve both
   models' identical WFS/DM histories. Add only an explicitly local simulation
   proposal sink/owner; do not enable its inhibited physical/Legacy output paths.
4. Run producer/consumer restart, layout, impulse/sign and same-three-trajectory
   tests on that runtime. Establish exposure integration/clock/latency explicitly;
   callback order and transport arrival alone do not establish physical timing.

Those native adapters are **not implemented**; there is no truthful command yet
to launch that full three-application chain. The exact command above is the
smallest implemented transport step, subject to native verification. Native apps,
Legacy/default controller, production configuration and the other worktree remain
unchanged. No new recovery framework, integration deployment, commit or push.

## Validation

- Connection/adapter tests: **14/14 pass with Clang and GCC libraries**, including
  exact memory/mock-shmim full trace equality, generated noisy ADC encoding/order,
  feature isolation for raw channel0, matched common calibration, causal input
  windows, two-frame impulse/cancellation, actual consumed commands, independent
  standard model, invalid prediction hold, and paused-fit/fresh-evidence behavior.
- Unchanged strict offline suites: **66 cases / 1,766,161 assertions Clang**;
  **66 / 1,764,361 GCC**. Historical exact Legacy/causal regressions retain their
  tolerances. The test-only training macro is absent from these normal builds.
- Default adaptive output sequences and aggregate metrics reproduce the previous
  connection results; new diagnostics do not affect corrections. `loPredCtrl`,
  `loPredCtrlAcc` production sources and other-worktree simulator sources unchanged.
- Sanitizers and GNU fast-math were not rerun in this focused iteration. Earlier
  results, including historical GNU fast-math failures, remain documented. No flag
  or tolerance was changed to conceal them. Native ImageStreamIO/Linux timing
  remains unverified, and no real-time/on-sky readiness is claimed.

Re-run tests after both adaptive and calibrated builds have been generated:

```sh
python -m unittest discover -s apps/loPredCtrlAcc/tests -p '*sim*test.py' -v
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j3 check
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j3 \
  BUILD_DIR=build/offline-phase9-gcc CXX=g++-15 check
python apps/loPredCtrlAcc/tests/simulation_diagnose.py --run --cxx g++-15 \
  --output apps/loPredCtrlAcc/build/channel01-gcc
DDSPC_SIM_LIB="$PWD/apps/loPredCtrlAcc/build/simulation-g++-15/libddspc_sim.so" \
DDSPC_SIM_FIXED_LIB="$PWD/apps/loPredCtrlAcc/build/simulation-g++-15-train2400/libddspc_sim.so" \
  python -m unittest discover -s apps/loPredCtrlAcc/tests -p '*sim*test.py' -v
```

Use the host's actual GNU compiler name on Linux. No native Linux execution was
performed here. Next useful step is the private-shmim transport qualification on
a development host, followed by the three small native adapters above, with a
calibrated-fit contrast and the unfavorable adaptive case retained. Demonstrating
sustained *additional* ADC-assisted benefit remains open; do not tune away or hide
the negative calibrated-fit result.

## Affected files in this follow-up

All implementation changes are under `apps/loPredCtrlAcc/tests/`:

- Updated `sim_bridge.cpp`, `Makefile.sim`, `simulation_demo.py`,
  `simulation_demo_test.py`, and the four guarded lines in `async_shadow.cpp`.
- Added `simulation_diagnose.py`, `sim_shmim.py`, `sim_shmim_test.py`.
- Updated `README.md`, `results/simulation_connection.md`; added this report and
  `results/channel01_validation/` metrics, diagnostics, manifests and PNG/PDF plots.
- Updated only the Agent Findings and Plan portion of
  `agents/plans/2026-11/accel-ddspc-controller.md`.

Earlier pending minimal-demo/initial-connection artifacts are retained. Nothing is
committed or pushed.
