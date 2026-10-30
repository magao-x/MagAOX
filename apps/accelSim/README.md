# accelSim

Continuous accelerometer simulation with the `mcp3208Ctrl` raw-count interface.
The default output is **`acc000`**, with frames containing X then Y as two
`uint16_t` values in [0,4095]. Only one producer may own that stream and its INDI
instance at a time. The executable name does not determine the replacement INDI
identity: supply the real instance with `-n` when its configuration is available.

The approved specification and engineering record are in
[`acc-sim-app-creation.md`](../../agents/plans/2026-10/acc-sim-app-creation.md).
There is no DM access, angular coordinate transform, transient generator, or
instrument startup configuration in this implementation.

## Model and timing

The model preserves the notebook's 14 prominent modes and 200-mode population
over 100–900 Hz, acceleration RMS envelope, direct pole radius `r`, and separate
X/Y WFS couplings. Each mode has independent X/Y histories, stationary Gaussian
initialization, and analytically scaled innovations. Pointing is in arcsec and
each mode contributes `-(2*pi*f)^2 * leverArm * theta_rad` to acceleration.
Independent accelerometer noise is added after summing the modes. WFS-coupled
pointing is retained internally without sensor noise or a separate output stream.

The ADC calculation is `zero + acceleration*sensitivity/voltsPerBit`, followed
by `std::round`, floating-point clamping, and conversion to `uint16_t`. Defaults
are 2048, 1.02, and 0.0083799 respectively. The three values and the sensor PSD
are configurable in [config/example.conf](config/example.conf).

Default sampling is 6000 Hz. The physics uses the configured nominal rate even
when a synchronization source has scheduling jitter. Set the requested accelerometer
rate through `sim.fps`, `fps.target`, or the external FPS subscription. Invalid,
nonfinite, nonpositive, above-10000, and aliased rates are rejected. An accepted
rate change takes effect at an acquisition boundary, recalculates the model,
and redraws stationary histories without reseeding. This introduces an explicit
counted discontinuity. The high-frequency parameter list is retained.

Timer acquisition uses an absolute monotonic schedule with fractional-period
accumulation and interruptible sleeps. Up to **eight skipped ticks** are advanced
before generating the current pair; only the newest pair is published. A larger
gap redraws stationary histories instead of performing unbounded catch-up.
Eight ticks represent 1.333 ms at 6 kHz. OS wake-up lateness is counted separately
from skipped publication opportunities. There is no catch-up publication burst.

Setting `synchro.shmimName` enables a **WFS phase reference for a continuous
accelerometer clock**. The worker waits for either the next accelerometer deadline
or a WFS semaphore. A semaphore updates the phase reference; it never publishes a
sample or moves the already pending deadline. Sampling starts even if the reference
stream is absent, and continues through silence, missing notifications, or source
replacement. These events clear lock/cadence diagnostics and enter holdover at the
last applied phase; they do not recreate acc000 or restart the physical model.

The measured WFS period is elapsed time divided by source-counter progress,
initialized from the first measured interval and then smoothed by `alpha`. The
phase remains `wrap(0.5*T_wfs - (dtTransfer + wfsProcess + dtF + wfsRead), T_wfs)`,
with the signed `postDelay` added and wrapped. The default latency sum is 340.6 us,
so a measured 500-us period gives 409.4 us without a hard-coded phase constant.

The clock subtracts elapsed **fractional WFS cycles** before projecting the phase
reference onto the accelerometer time grid. Thus a 2-kHz reference works with
8 kHz, 6 kHz, 5.5 kHz, and other valid sampling rates; no integer samples-per-WFS
counter or per-trigger burst is used. With a non-integer ratio, not every WFS
reference can coincide with an accelerometer sample while preserving the requested
rate. The relative phase advances continuously across WFS cycles instead.

Three synchronization options control this behavior:

- `synchro.referenceFps=0`: learn the nominal WFS phase-tracking cadence from the
  first measured interval after acquisition/reconnection. Set the known nominal
  WFS rate (for example 2000) when available to avoid basing the phase grid on a
  noisy first interval. This affects fractional-cycle accounting, not accelerometer
  FPS; the latency formula still uses measured/smoothed cadence. The learned value
  is held until timing is invalidated. A changed WFS nominal rate requires matching
  this setting/reacquiring the reference. A wrong nominal reference can produce
  repeated correction transients; it cannot force the output rate to the WFS rate.
- `synchro.maxSlew=0.05`: limit the phase adjustment of each following interval to
  5% of an accelerometer period (valid range greater than zero through 0.25).
  `cadenceGuard_ns` and measured computation time further limit available correction
  headroom. Phase offset stays within half an accelerometer period of the nominal
  epoch, so corrections cannot accumulate a long-term rate change. Lock acquisition
  takes several samples. After an overrun, the next publication is at least half
  an accelerometer period later, preventing an immediate recovery burst.
- `synchro.timeout=0.01`: reference silence timeout and source-file check/retry
  interval in seconds. Increase this above the expected source period for slow
  references. Timeout clears lock while sampling continues; a replacement source
  needs fresh timing before relocking.

Source-counter gaps count missed **WFS references** only. Actual missed
accelerometer deadlines count `skipped_publications` and advance numerical ticks
using the existing eight-skipped-tick bound; larger gaps restart stationarity once.
Counter receipt is accounted immediately, even if a subsequent wait is canceled.
Duplicates cannot create extra samples. The code never retunes AR coefficients
from WFS jitter. Keep `fps.device` empty unless that subscription supplies the
requested **accelerometer** FPS; subscribing it to a 2-kHz WFS FPS still explicitly
requests a 2-kHz accelerometer rate.

For the 8-kHz / 2-kHz example, set `sim.fps=8000`,
`synchro.shmimName=camwfs_sync`, `synchro.referenceFps=2000`, and leave
`fps.device` empty. The free-running 6000-Hz defaults and timer path are unchanged.

Acquisition timestamps use `CLOCK_REALTIME` immediately before sample generation,
including recovery computation. They represent the current acquisition event,
not reconstructed timestamps for missed frames. `frameGrabber` owns write time,
`cnt0`, `cnt1`, per-plane metadata, circular buffering, and all output semaphore
posts. The image has three axes `[2,1,depth]` even at depth one. Wall-clock
corrections can change epoch timestamps; pacing remains monotonic.

## Controls and diagnostics

The app uses `MagAOXApp<true>`, `frameGrabber`, and the required `telemeter` mixin.
It exposes the normal `fg_*`, thread, FSM, logging, and telemetry properties, plus
the hardware application's `fps`, `numChannels`, `alpha`, `synchroDelay`, and all
23 `timingDiag` elements. `fps.current` is the active nominal rate; `target` may
briefly reflect a pending update. FPS requests report Busy until the acquisition
worker acknowledges that request, then Ok; invalid requests retain Alert until a
valid request. Superseding requests cannot inherit an earlier acknowledgment.
Measured publication rate is in `fg_timing`.
`numChannels` accepts only 2, including on its external subscription; rejected
updates leave the shape and accepted values intact and set Alert. There is no
additional camera-style start/stop control.

`channel_readout_us` measures simulation acquisition computation, not fictitious
SPI hardware timing. In timer mode synchronization-only measurements are zero.
`alpha` controls timing smoothing, never AR damping.

Each `timingDiag` update sends one coherent property, including source-loss
invalidation. Names and units are retained, but the synchronized meanings now
reflect a phase-referenced sampling clock rather than one ADC read per semaphore:

- `delay_model_us` / `synchro_delay_target_us`: measured-cadence WFS model and
  offset-adjusted reference phase. `trigger_time_us` / `trigger_interval_us` describe
  those WFS predictions, not accelerometer intervals.
- `delay_applied_us` / `synchro_delay_us`: reference phase including the remaining
  offset error of the scheduled accelerometer clock, wrapped over a WFS period.
  `delay_phase_error_us` retains the positive half-period boundary. `delay_lock`
  requires a fresh reference and the existing absolute/fractional phase thresholds;
  excessive acquisition lateness also clears it. Lock is not an independent
  measurement of hardware/WFS latency.
- `delay_budget_us` / `delay_capped`: available per-accelerometer-interval phase
  correction and whether correction remains beyond that budget.
- `avg_read_latency_us`: smoothed arrival-to-sample latency for selected WFS
  references, measured at the first sample at/after their predicted phase.
  `read_latency_error_us` is that sample's lateness relative to its retained
  reference prediction. A pending measurement survives an intervening WFS event;
  not every reference necessarily contributes a measurement.
- `sync_frames_received` counts semaphore receipts; source ID/gap/drop fields
  describe WFS counters. `sync_frames_written` counts accelerometer publications
  in synchronized mode, including holdover, and can exceed received references.
- `sim_diag.source_period_ratio` is measured WFS period / accelerometer period:
  approximately 4 for 8/2 kHz and 3 for 6/2 kHz, not an error requiring equal rates.

The additional read-only `sim_diag` contains `skipped_publications`,
`discontinuities`, `late_samples`, `max_lateness_us`, `source_period_ratio`, and
`clip_x_low`, `clip_x_high`, `clip_y_low`, `clip_y_high`. Clipping counts actual
clamps after rounding, including numerical samples advanced during catch-up;
values rounding naturally onto a rail are not clamp events. Counters are retained
over FPS changes. A shutdown log summarizes timing and clipping totals. A
nonblocking snapshot transfers normal acquisition diagnostics to the ordinary app
loop, without INDI calls, logging, or allocations in normal numerical stepping.
Lifecycle/outage transfers require delivery and may wait only for the app loop's
short snapshot copy; that mutex is released before any INDI work.

Seeds default to 42 (samples) and 123 (mode parameters). Repeatability is within a
fixed C++ standard library/build and fixed numerical stepping schedule, not across
NumPy RNGs or different OS timing. Stationary RMS is an ensemble property: a
60-second record of an `r=0.9999999` mode is not forced to the configured RMS.
PSD calculation stays in Python.

## Build and focused tests

From the repository root, on a configured MagAO-X development host:

```sh
make -C apps/accelSim
make -C tests -f Makefile.one t=../apps/accelSim/tests/accelSim_test.cpp
./apps/accelSim/tests/accelSim_test '[accelSim]'
./apps/accelSim/tests/accelSim_test '[indi-regression]'
./apps/accelSim/tests/accelSim_test '[accelSim-streams]'
```

The last test creates a private temporary ImageStreamIO namespace and exercises
real frameGrabber publication and a synthetic semaphore source. It does not
connect to a live INDI server or instrument. Run it separately from the live
benchmark because ImageStreamIO can cache its shared-memory directory.

The numerical portion can run without MagAO-X dependencies (including on macOS):

```sh
c++ -std=c++20 -O3 -fno-fast-math -Wall -Wextra \
  -DCATCH_CONFIG_MAIN -DACCELSIM_MODEL_ONLY \
  apps/accelSim/tests/accelSim_test.cpp -o /tmp/accelSim_model_test
/tmp/accelSim_model_test '[model]'
/tmp/accelSim_model_test '[sync-regression]'
```

The standalone tests include the production synchronization accounting and
status-invalidation logic, integer/fractional rate ratios, correction bounds,
jitter/phase shifts, holdover, half-period phase boundaries, and trigger startup.
The INDI regressions capture real XML serialization without a server, checking
coherent timing snapshots and FPS Busy/Ok/Alert transitions. Their local driver
fixture uses Linux `/proc/self/fd` to release its unactivated input descriptor.
Full app tests additionally check snapshot delivery,
continued sampling with a missing or duplicate synthetic reference, stale-frame
rejection, and source disappearance/reconnection; those require the normal dependencies.

The app and normal test Makefiles disable fast-math locally, since nonfinite
validation and near-unit-root calculations require ordinary floating-point
semantics. Do not override this with fast-math when validating.

## Local launch and inspection

Use a configured **non-instrument Linux development host**. The following creates
private config/log/FIFO/shared-memory paths and uses a local test instance, not
an invented deployment identity. The example must be used alongside the host's
normal MagAO-X runtime prerequisites and user/group permissions. No INDI tunnel
to an instrument should be used for these checks.

```sh
accel_sim_root=$(mktemp -d /tmp/accelSim-local.XXXXXX)
export MAGAOX_PATH="$accel_sim_root"
export MILK_SHM_DIR="$accel_sim_root/shm"
mkdir -p "$MILK_SHM_DIR" "$MAGAOX_PATH/config" "$MAGAOX_PATH/logs" \
  "$MAGAOX_PATH/telem" "$MAGAOX_PATH/sys" "$MAGAOX_PATH/calib" \
  "$MAGAOX_PATH/drivers/fifos"
./apps/accelSim/accelSim -n accelSim-local \
  -c apps/accelSim/config/example.conf --framegrabber.threadPrio=0 \
  >"$accel_sim_root/app.log" 2>&1 &
accel_sim_pid=$!
```

Priority zero is for an unprivileged smoke test. The approved performance test
uses the normal framework priority/cpuset settings on a suitable development
host. Supply those settings when evaluating performance. A local INDI server,
if used, must attach to this private FIFO directory. The exact deployment
instance, source stream, circular depth, and downstream configuration remain
operator-supplied settings.

In the same environment, an existing reader can inspect the unchanged stream:

```sh
shmimLat -n acc000
```

For a ten-minute read-only benchmark of the running producer:

```sh
ACCELSIM_LOCAL_BENCHMARK=1 ./apps/accelSim/tests/accelSim_test '[accelSim-live]'
```

For a producer configured at 8 kHz, add `ACCELSIM_EXPECTED_FPS=8000` to the
benchmark command. This changes only the reader's expected-rate assertion; it
does not configure the producer. The default remains 6000 Hz.


It checks ADC ranges, counter progression, and mean publication rate within 1%
of 6000 Hz, and reports reader counter gaps and write-interval mean, standard
deviation, maximum, and histogram. Histogram upper bounds are 100, 150, 175,
200, 250, 500, and 1000 microseconds, followed by an overflow bin. Intervals are
computed only for consecutive observed frames; a slow reader cannot measure
every producer interval. Reader gaps must not be confused with producer skips.
Collect the producer's `sim_diag` values and acquisition/write timing properties
through the isolated local INDI session as well. Record RSS during the run, for
example from a second local shell with the captured PID:

```sh
ps -p "$accel_sim_pid" -o pid,etime,rss,pcpu
```

Take repeated RSS measurements after startup to assess stable memory. There is
no approved hard jitter/miss threshold yet; report the distribution and every
counted miss. Stop this local producer and inspect its summary:

```sh
kill -TERM "$accel_sim_pid"
wait "$accel_sim_pid"
cat "$accel_sim_root/app.log"
```

## Offline visual validation on macOS

This workflow calls the same C++ `model::step()` as the application, with no
MagAO-X runtime dependencies, shared-memory streams, INDI, or wall-clock pacing.
The finite record is a diagnostic export only; production remains continuous.
The export uses the normal model defaults: **6000 Hz, 60 seconds, seed 42,
modeSeed 123**, producing 360000 X/Y pairs from 214 resolved modes.

From the repository root, build and export:

```sh
c++ -std=c++20 -O3 -fno-fast-math -Wall -Wextra \
  -DCATCH_CONFIG_MAIN -DACCELSIM_MODEL_ONLY \
  apps/accelSim/tests/accelSim_test.cpp -o /tmp/accelSim_model_test
mkdir -p /tmp/accelSim-visual-validation
ACCELSIM_EXPORT_DIR=/tmp/accelSim-visual-validation \
  /tmp/accelSim_model_test '[accelSim-export]'
```

The normal full test binary also supports `[accelSim-export]`. Exported files:

- `samples.csv`: nominal time [s]; physical and WFS pointing X/Y [arcsec];
  acceleration X/Y including sensor noise [m/s²]; raw ADC X/Y [counts].
- `modes.csv`: resolved frequencies, per-axis acceleration RMS, pole radii,
  and X/Y WFS couplings used by this C++ realization.
- `metadata.json`: sample rate, duration, both seeds, lever arm, ADC calibration,
  noise PSD, and the model's actual lower/upper clipping counts for each axis.

Prepare an offline Python environment once, then generate the figures:

```sh
python3 -m venv /tmp/accelsim-validation-env
/tmp/accelsim-validation-env/bin/python -m pip install numpy scipy matplotlib
/tmp/accelsim-validation-env/bin/python apps/accelSim/tests/validate_psd.py \
  /tmp/accelSim-visual-validation
```

Add `--show` to also open interactive Matplotlib windows (the plot files are
saved first). `--time-start 5 --time-seconds 0.5` selects a different short time
trace; PSDs and RMS always use the entire exported record. Defaults show the
first second. Without `--show`, plotting uses the headless Agg backend.

Results are printed and saved in `validation.json`; all measured and predicted
spectra are saved in `psd.csv`. Figures are saved under the export directory:

| File in `plots/` | Meaning |
| --- | --- |
| `pointing_timeseries.png` | Physical pointing X/Y in mas, first second by default. |
| `pointing_psd.png` | Physical pointing PSD in mas²/Hz; narrow low-frequency modes tend to dominate the pointing variance. |
| `wfs_psd.png` | WFS-coupled pointing PSD in mas²/Hz; each mode's power is multiplied by its axis coupling squared, without accelerometer sensor noise. |
| `accel_psd.png` | Acceleration PSD in (m/s²)²/Hz, including the 100–900 Hz population and sensor noise. |
| `adc_psd.png` | Inverse-calibrated ADC acceleration PSD, overlaid with the same pre-ADC realization to expose quantization/clipping effects. |

All PSD figures mark the 14 prominent frequencies. Welch uses SciPy with
`window="hann"`, `nperseg=10*fs` (limited to record length for shorter inputs),
50% overlap, `detrend="constant"`, and `scaling="density"`.

Dashed curves are stationary ensemble AR(2) predictions, including the **same
finite Hann window and constant detrending** as the estimator. They are computed
from the analytic lag covariance, not fitted to the record. This avoids comparing
an unresolved high-r resonance's infinite-record peak height directly with a
10-second-window estimate. Physical/WFS theory uses the lever-arm conversion and
squared axis couplings; acceleration theory adds the configured white sensor noise.
The ADC plot's theory is explicitly **pre-ADC**: it does not model quantization
or clipping. The noise-dominated tail should approach 1e-8 (m/s²)²/Hz before
quantization; default ADC quantization adds roughly 1.87e-9 (m/s²)²/Hz under the
usual white quantization-error approximation.

Reported RMS is `sqrt(mean(x²))`, without subtracting the record mean. X/Y
realizations are independent. Very high-r modes can have different realized
amplitudes across axes/seeds and differ from the ensemble RMS in a 60-second
record; this is expected, not a failed normalization check. Clipping totals count
actual model clamps, separately from valid samples that happen to equal 0 or 4095.
Older exports without metadata use documented default calibration and reconstruct
clamp counts from their acceleration values; regenerate data to obtain exact C++
counters. Analysis flags such as `--fps`, `--lever-arm`, and `--volts-per-bit`
override metadata for analysis only; they do not regenerate or alter the simulation.

Optional offline checks:

```sh
/tmp/accelSim_model_test '[model]'
/tmp/accelsim-validation-env/bin/python -m unittest discover \
  -s apps/accelSim/tests -p 'test_validate_psd.py' -v
```

## Current verification limits

The macOS numerical tests (also with address/undefined-behavior sanitizers) and
offline export/PSD comparison have passed. The
normal app and full test builds are blocked before app compilation by existing
mxlib/platform dependencies (`mx/math/floatUtils.hpp` missing and `timegm`
undeclared). No Linux stream, INDI runtime, or ten-minute performance result is
claimed. The real deployment configuration and raw-ADC DDSPC consumer are absent
from this checkout, so final downstream substitution still requires those inputs.
See the planning document for exact results and remaining coverage.
