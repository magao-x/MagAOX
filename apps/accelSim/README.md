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
when a synchronization source has scheduling jitter. Set the correct source
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

Setting `synchro.shmimName` enables semaphore synchronization. Configuration names,
phase wrapping, EMA, cadence budget, and lock criteria follow `mcp3208Ctrl`.
Elapsed nominal time and cumulative source-counter progress are tracked independently;
the later position determines the current opportunity. Short pauses advance all
missed ticks, and overlapping counter/time gaps are counted once. Notifications
for a repeated counter or a nominal opportunity already covered by recovery do
not replay samples. Long pauses retain the same bounded recovery policy.
A consumed trigger is finalized after acquisition or explicit cancellation;
canceled current opportunities and older gaps remain counted even across FPS
changes. Source replacement is detected on timeout and reconnects through the
normal frameGrabber lifecycle. A timeout (including a mapped but stalled source)
or reconfiguration clears lock/cadence/delay measurements and transfers that
status to the app loop without waiting for another published frame. An input source must not be the output stream.
`sim_diag.source_period_ratio` compares smoothed arrival cadence with the nominal
model period; its startup settling follows the hardware EMA. A persistent value
different from one requires matching the nominal FPS to the source. The code
does not retune AR coefficients based on jitter.

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
briefly reflect a pending update. Measured publication rate is in `fg_timing`.
`numChannels` accepts only 2, including on its external subscription; rejected
updates leave the shape and accepted values intact and set Alert. There is no
additional camera-style start/stop control.

`channel_readout_us` measures simulation acquisition computation, not fictitious
SPI hardware timing. In timer mode synchronization-only measurements are zero.
`alpha` controls timing smoothing, never AR damping.

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
status-invalidation logic. Full app tests additionally check snapshot delivery,
phase-wait cancellation through a synthetic semaphore, stale-frame rejection,
and source disappearance/reconnection; those require the normal dependencies.

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

## Offline PSD validation

The hidden export test uses the production model, without OS pacing, to write a
60-second numerical sample CSV and its resolved mode table. It does not add a
playback dependency to the app. With NumPy/SciPy installed:

```sh
mkdir -p /tmp/accelSim-validation
ACCELSIM_EXPORT_DIR=/tmp/accelSim-validation \
  /tmp/accelSim_model_test '[accelSim-export]'
python3 apps/accelSim/tests/validate_psd.py /tmp/accelSim-validation
```

The normal test binary also supports `[accelSim-export]`. Outputs include
`validation.json` and `psd.csv`. Welch settings match the notebook. Physical and
inverse-calibrated raw-ADC spectra are reported separately to expose quantization;
the inverse calibration here is only for analysis, not a DDSPC requirement.

## Current verification limits

The macOS numerical tests (also with address/undefined-behavior sanitizers) and
offline export/PSD comparison have passed. The
normal app and full test builds are blocked before app compilation by existing
mxlib/platform dependencies (`mx/math/floatUtils.hpp` missing and `timegm`
undeclared). No Linux stream, INDI runtime, or ten-minute performance result is
claimed. The real deployment configuration and raw-ADC DDSPC consumer are absent
from this checkout, so final downstream substitution still requires those inputs.
See the planning document for exact results and remaining coverage.
