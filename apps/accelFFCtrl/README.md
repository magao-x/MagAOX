# accelFFCtrl non-actuating accelerometer observer

`accelFFCtrl` continuously reads raw accelerometer pairs, runs the existing
`accelFFCtrlDetail::harmonicEstimator`, and exposes predicted physical pointing
through diagnostic INDI properties. It has no DM publisher, correction enable,
gain, or actuator stream attachment. Input names beginning with `dm` are rejected
before a native open. Starting or stopping this application does not change a DM.

The engineering record is
[accelSim-28hz-feedforward-controller.md](../../agents/plans/2026-11/accelSim-28hz-feedforward-controller.md).
The existing simulator, its model/RNG/timing, and its DM injection are unchanged.

## Live architecture and stream retention

`accelFFCtrl.hpp` follows the normal `MagAOXApp<true>` lifecycle and `threadStart`
conventions; the `.cpp` contains only `main`. A single worker owns
`accelFFCtrlStream.hpp`'s existing-only mapping and
`accelFFCtrlInput.hpp`'s preallocated batch, cursor, clock mapping and estimator.
INDI callbacks enqueue coherent parameter revisions. The worker applies them
using `try_lock`, resets the fit, and publishes fixed-size value snapshots with
another `try_lock`. The ordinary app loop handles filesystem identity probes,
INDI formatting and communications. There is no simulator model in this path.

The existing `dev::shmimMonitor` consumes only the latest `cnt1` plane per wake;
it cannot recover an 8-kHz sample history. This app therefore uses native
ImageStreamIO attachment and semaphore claiming without changing that helper.
A semaphore wake is a notification, not an immutable sample. The worker uses
`cnt0` to enumerate available publications relative to the latest `cnt1`, and
checks each retained plane's `cntarray`, `atimearray` and `writetimearray`. It
copies owned pairs/metadata into bounded local storage, checking global write
state, geometry and counters around the copy. Inconsistent copies are retried
without advancing the cursor.

**A one-frame circular buffer cannot guarantee every-frame consumption.** Once
a pair is overwritten, semaphore counts or a consumer-side queue cannot restore
it. Depth-one input is supported for diagnosis; gaps are counted and reset the
fit. It is never silently treated as lossless.

For a frameGrabber producer, the operator can use its existing startup setting
`framegrabber.circBuffLength=256`: 32 ms retention at 8 kHz with per-plane
timestamps/counters. No producer code or synchronization change is needed.
The reader examines retained records in acquisition order in bounded batches.
Finite retention still permits overrun. A strict every-frame guarantee needs a
producer-supported acknowledged queue or equivalent transport in a later phase.
The [ImageStreamIO pointer/semaphore APIs](https://milk-org.github.io/milk-package/ImageStreamIO_8h.html)
provide neither a frame lease nor that guarantee.

Freshness is separate: with the default 2-ms maximum age, old retained records
are counted/discarded and clear the fit, never assigned fabricated receipt times.
Retention of 32 ms does not authorize 32-ms-old data. Initial attachment starts
at the newest completed pair; lifetime counters cover observed history from
that point. Every later coherent, fresh, consecutive pair is ingested once.

## Live clocks, cadence and recovery

Accepted input is CPU-resident UINT16 `(2,1)` or `(2,1,D)`, X followed by Y.
Depth >1 requires temporal per-plane timestamps/counters and the existing
frameGrabber publication contract. Codes above 4095 are invalid. The live app
fixes channel order to X/Y and positive sensor polarity; rotation is configurable.
ADC calibration is not carried in stream metadata.

Acquisition stamps must be positive, valid **CLOCK_REALTIME** values on the same
synchronized host clock as the controller. Zero, malformed, backwards, future,
stale, or write-before-acquisition stamps inhibit the fit. A post-copy clock
sample brackets realtime with two monotonic reads. Its integer epoch translation
is fixed until reattachment or a detected clock step. Excessive bracket
uncertainty/offset change clears the fit and reanchors on the next observation.
This does not synchronize an independent sensor clock or an unknown remote-host
clock offset. Every record retains its own acquisition time; a batch uses actual
post-copy receipt time for age validation.

Prediction is queried at 2 kHz by default with
`target = monotonic_now + prediction_horizon`. Phase advances by
`target - latest_acquisition`, including measurement age **once**. No extra
fixed window delay is added. INDI is a 5-Hz diagnostic snapshot, not a high-rate
prediction transport. The worker wakes on stream posts or a prediction deadline
and checks staleness even during input silence. glibc 2.30+ uses
`sem_clockwait(CLOCK_MONOTONIC)`; older platforms use `sem_trywait` and bounded
relative sleeps (up to 20 us requested). Neither relies on a realtime timeout;
OS scheduling can exceed requested delays.

Failed attachment retries once per second. The ordinary app loop checks backing
inode/device at telemetry cadence; the owner closes/reopens replacement mappings,
increments generation and resets. Counter rollback increments generation
immediately. Gaps, bad samples/times, stale data, configuration and explicit
resets clear/re-warm observation automatically. No old prediction is held ready;
supervision also checks a snapshot's age before sending it to INDI.

Steady processing uses fixed numerical state and startup-allocated batch storage,
bounded copying/semaphore draining, no blocking mutex acquisition, and no INDI,
filesystem probes or routine logging. Attachment/reconnection can allocate/do
I/O. Native library/producer publication ordering, semaphore behavior and safe
unlink/recreate semantics need Linux qualification; concurrent truncation of an
attached mapping is not a supported restart protocol.

## Configuration and INDI

[config/example.conf](config/example.conf) is observation-only. `[parameters]`
keys have same-named INDI properties with `current` (owner-applied) and `target`
(requested) elements:

| Property | Units | Default / example |
| --- | --- | --- |
| `frequency` | Hz | 28.5; use 28 for accelSim's unchanged default mode |
| `memory_time` | seconds, total mean memory | 0.1 |
| `lever_arm` | metres | 10, requiring physical calibration |
| `rotation` | degrees, sensor to pointing axes | 0 |
| `prediction_horizon` | ms beyond query time | 0 without config; example 1 |
| `adc_zero_x`, `adc_zero_y` | ADC counts | 2048 |
| `adc_volts_per_bit_x`, `adc_volts_per_bit_y` | volts/count | 0.0083799 |
| `adc_sensitivity_x`, `adc_sensitivity_y` | effective volts/(m/s²) | 1.02 |

Updates are finite/cross-validated as a complete candidate. Invalid edits preserve
accepted settings and mark that property Alert. Valid edits are Busy until
applied; each applied revision resets warmup. Requests can coalesce before an
owner iteration. `estimator_reset.request=On` requests a reset without parameter
changes and returns Off on acknowledgement. Observation runs continuously;
there is no actuator enable/disable control.

Startup-only `[input]` settings: `shmimName=acc000`, `threadPrio=2`, empty `cpuset`,
`maxBatch=256`, nominal `sampleRate=8000`, `predictionFps=2000`, `telemetryFps=5`,
`maxAge_us=2000`, `maxInterval_us=2000`, `maxPrediction_ms=50`, and
`clockTolerance_us=1000`. The last four are data/model-validity bounds, not
hardware limits. The prediction bound includes sample age. Clock tolerance is
an allowed mapping-error bound, not a phase-accuracy claim; use measured clock
performance to choose a tighter commissioning value when appropriate.

Read-only `input` text reports stream name, exact decimal counter/generation,
`mode=observation only`, and the last valid prediction's monotonic
`sample_time_ns` / `target_time_ns` (zero when invalid). `diagnostics` reports:

- `tip_x_mas`, `tilt_y_mas`: predicted rotated physical disturbance in mas at
  target, without opposing sign or DM calibration.
- `amplitude_x_ms2`, `amplitude_y_ms2`: peak acceleration in sensor axes;
  `phase_x_deg`, `phase_y_deg`: sensor harmonic phase at `sample_time_ns`.
- `ready`, `fit_state`, `stream_state`, `last_event`: validity/last reset cause.
  Invalid estimates are zero; zero alone does not imply readiness.
- `input_fps`: accepted throughput over 250-ms monotonic windows, including loss;
  `acquisition_fps`: smoothed rate within consecutive valid acquisition times.
- Lifetime `processed`, `dropped` (overwritten plus rejected), `invalid_frames`
  (subset of dropped), `restarts`, `resets`, `counter_rollbacks`; `empty_polls`,
  `busy_polls`, `copy_races`; and `retention_frames`.
- `sample_age_us`, `effective_horizon_us` (target minus sample),
  `producer_delay_us` (write minus acquisition), `service_us` (last prediction
  iteration's work excluding wait), and `applied_revision`.

`stream_state` / `last_event`: 0 disconnected, 1 waiting, 2 streaming,
3 incompatible, 4 overrun, 5 invalid time, 6 clock fault, 7 stale, 8 numerical.
`fit_state`: 0 warming, 1 ready, 2 invalid sample, 3 timing fault, 4 sequence
fault, 5 stale, 6 numerical fault, 7 invalid request. Automatic reset can clear
a fit fault before the next snapshot; counters and `last_event` retain evidence.
Default warmup requires 0.3 s of valid consecutive acquisition coverage.

## Build and a non-actuating exao2 test

First use a configured MagAO-X Linux development checkout with the normal
compiler, mxlib/Eigen, libMagAOX, INDI and ImageStreamIO dependencies:

```sh
make -C apps/accelFFCtrl
make -C apps/accelFFCtrl/tests test benchmark native-test
make -C tests -f Makefile.one t=../apps/accelFFCtrl/tests/accelFFCtrl_test
./apps/accelFFCtrl/tests/accelFFCtrl_test '[accelFFCtrl][app]'
```

`native-test` launches a fresh process in an empty
`/tmp/accelFFCtrl-native-XXXXXX` namespace, creating only synthetic
`accff-private-*` streams. Its hidden tag must be selected explicitly; never
point it at the instrument namespace. Full app tests do not start a worker or
open streams.

These deployment commands are for a **human operator**. Per
[AGENTS.md](../../AGENTS.md), an agent must not execute on exao2 or through an
instrument INDI tunnel. Pass the Linux build/native tests and review the source
diff first. To transfer the reviewed working tree without a commit/push, on the
workstation from the checkout root:

```sh
git ls-files -z --cached --others --exclude-standard -- Makefile apps/accelFFCtrl \
  tests/Makefile.one tests/groups.dox tests/tests.list \
  agents/plans/2026-11/accelSim-28hz-feedforward-controller.md \
  | tar --null -czf /tmp/accelFFCtrl-live-source.tgz -T -
scp /tmp/accelFFCtrl-live-source.tgz exao2:/tmp/
```

The file list excludes ignored build products, so workstation binaries/objects
are not transferred. On exao2, verify the checkout/path and preserve local changes before extracting.
These commands assume `/opt/MagAOX/source/MagAOX`:

```sh
cd /opt/MagAOX/source/MagAOX
git branch --show-current
git status --short
tar -tzf /tmp/accelFFCtrl-live-source.tgz
tar -xzf /tmp/accelFFCtrl-live-source.tgz
make -C apps/accelFFCtrl
make -C apps/accelFFCtrl install
sudo install -m 0644 apps/accelFFCtrl/config/example.conf \
  /opt/MagAOX/config/accelFFCtrl-observe.conf
```

Confirm the **one existing** `acc000` producer publishes UINT16 X/Y pairs, valid
host realtime acquisition stamps, the intended rate, and matching calibration.
Set observer ADC/lever/frequency values accordingly. For an 8-kHz simulator test,
the operator can use existing `sim.fps=8000` and
`framegrabber.circBuffLength=256` settings on that producer, and
`parameters.frequency=28` on the observer. Do not start a second `acc000`
producer. Changing retention requires an owner-coordinated producer restart;
the observer never requests it. With depth one, record loss/warmup failures as
the expected transport limitation. This app does not arm or change the
simulator's separate DM injection.

In a dedicated operator terminal:

```sh
/opt/MagAOX/bin/accelFFCtrl -n accelFFCtrl-observe \
  -c /opt/MagAOX/config/accelFFCtrl-observe.conf
```

For an isolated INDI connection, choose an unused port (7625 here) and a new
symlink. Do not attach a second xindidriver to an already managed FIFO or restart
the production INDI server. In a second operator terminal:

```sh
ln -s /opt/MagAOX/bin/xindidriver /tmp/accelFFCtrl-observe
indiserver -p 7625 /tmp/accelFFCtrl-observe
```

In a third operator terminal:

```sh
getINDI -h localhost -p 7625 'accelFFCtrl-observe.input.*'
getINDI -h localhost -p 7625 'accelFFCtrl-observe.diagnostics.*'
setINDI -h localhost -p 7625 'accelFFCtrl-observe.estimator_reset.request=On'
```

Observe at least ten minutes under representative load. Record throughput,
acquisition rate, drops/invalids, races, resets, age, producer delay and service
time. A qualified retained-ring run should approach 8-kHz accepted throughput,
become ready after warmup, and remain fresh without gaps. Depth one cannot
establish every-frame consumption. Change frequency/memory through `.target`
and confirm owner-applied acknowledgement, cleared estimates, and warmup. Have
the producer owner exercise a controlled pause/restart only when appropriate;
confirm stale status, new generation and warmup. Stop the observer with Ctrl-C,
stop its isolated indiserver, and remove the temporary symlink. Existing
applications retain ownership of the ADC producer and all actuator streams.

## Numerical input, units, and timing

Pass **every available sample**, in acquisition order, to `consume(record, now)`.
The record contains exactly two `uint16_t` codes, the acquisition timestamp,
source sequence, and source generation. Querying at a lower cadence does not
decimate ingestion; the tests use 8000-Hz ingestion and 2000-Hz prediction.

For each selected sensor axis:

```text
a = polarity * (double(raw[channel]) - zero) * voltsPerBit / sensitivity  [m/s²]
```

Defaults match accelSim: zero 2048, volts/count 0.0083799, sensitivity 1.02.
Set both axes from the producer's actual configured values; they are not encoded
in `acc000`. Channel selection, polarity, offsets, and sensitivity can differ
between axes. Values above 4095 are invalid. Legal rail codes are accepted; this
numerical phase has no sustained-saturation policy or hardware displacement bound.

Acquisition time and `now` must be **nonnegative integer nanoseconds in one
monotonic domain**. A numerical epoch of zero is valid. The live adapter performs
realtime-to-monotonic mapping and stream-plane coherence checks; standalone
callers must supply them. The core reads no clocks or metadata itself. Subtract
integer timestamps before converting intervals to seconds, preserving precision
even at large epochs.

The first sample binds the source and uses `1/sampleRate` as its window interval.
Subsequent samples use their actual acquisition intervals. True duplicate records
return false without advancing state/count or refreshing freshness. Missing,
backwards, changed-generation or conflicting same-counter records latch a fault.
Invalid/backwards times, excessive intervals, stale input, nonfinite arithmetic,
or an ill-conditioned warmed fit also inhibit results until `reset()` or a valid
`configure()`. The caller should distinguish a duplicate from a fault using
`state()` and `samples()`.

Default maximum acquisition age and sample interval are each 2 ms; maximum target
distance from the newest sample is 50 ms. These configurable bounds express
numerical data/model validity. `predict()` checks freshness even without new
input, and invalid results contain zero estimates. A bad target request does not
destroy an otherwise valid fit. Changing configuration resets the fit; invalid
configuration throws before modifying the prior settings. No exceptions or
allocations are used for normal per-sample or prediction error handling.

## Harmonic fit and physical conversion

The initial frequency is 28.5 Hz and total mean memory T is 0.1 s. The simulator's
existing default target remains 28 Hz; set `m_frequency=28` when matching it.
The model is a nuisance offset plus one harmonic in each decoded sensor axis:

```text
u = [1, cos(phi), sin(phi)]^T
a_i(t) ~= d_i + A_i*cos(phi(t)) + B_i*sin(phi(t))
phi_n = wrap(phi_(n-1) + 2*pi*f0*dt)
beta = -expm1(-2*dt/T); lambda = 1-beta
G1 = lambda*G1 + beta*u*u^T
Q1 = lambda*Q1 + beta*u*[a_x,a_y]
G2 = lambda*G2 + beta*G1              # newly updated first stages
Q2 = lambda*Q2 + beta*Q1
G2*C = Q2                           # fixed 3x3 LDLT, two right-hand sides
```

Each stage has time constant T/2. For uniform intervals, weights are proportional
to `(k+1)*lambda^k`; this is a two-stage window, not ordinary single-exponential
RLS. Both Gram and data moments receive exactly the same window. An independent
weighted batch-QR test verifies the result for uniform and irregular timestamps.
The second stage improves rejection of strong high-frequency acceleration before
division by the target frequency squared.

Warmup requires `max(3/f0,3*T)` seconds of acquisition coverage. Prediction also
requires a positive, finite, sufficiently conditioned Gram matrix and a small
solve residual. The offset is excluded from vibration. The result reports peak
acceleration amplitude `hypot(A,B)` and phase at its `m_sampleTimeNs`, meaning
`amplitude*cos(2*pi*f0*dt + phase)`. Phase is undefined for zero amplitude; zero
is reported for that case. These fields remain in decoded sensor axes.

```text
a_hat_i(t_target) = A_i*cos(phi_target) + B_i*sin(phi_target)
theta_sensor = -a_hat / ((2*pi*f0)^2 * leverArm)             [rad]
theta_output = [[cos(alpha),-sin(alpha)],
                [sin(alpha), cos(alpha)]] * theta_sensor
```

The default lever arm is 10 m and rotation is 0 degrees. Rotation changes the
returned pointing axes only. Output is the physical disturbance: no correction
gain or opposing command sign is applied. The inverse includes no extra factor
of 9.81, reflection factor, or DM calibration.

Use `predict(now, target)` for an explicit future target, or `predict(now)` for
`now + m_predictionNs`. The phase advance always uses `target-last_sample`, so
sample/queue age is already included once. No fixed estimator delay is added to
the matched carrier. Detuned or changing envelopes retain window lag; prediction
does not eliminate that lag or infer unknown future stochastic innovations.

```cpp
using namespace MagAOX::app::accelFFCtrlDetail;
parameters settings;
settings.m_zero.fill(producerZero);
settings.m_voltsPerBit.fill(producerVoltsPerCount);
settings.m_sensitivity.fill(producerSensitivity);
settings.m_frequency = 28.5;
settings.m_predictionNs = 1000000;
harmonicEstimator estimator(settings);
// Pass each owned timestamped ADC record to estimator.consume(record, nowNs).
// Query estimator.predict(nowNs); use estimates only when m_state is ready.
```

## Validation and benchmark

From the repository root, with Eigen available through pkg-config:

```sh
make -C apps/accelFFCtrl/tests test
make -C apps/accelFFCtrl/tests benchmark
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  make -C apps/accelFFCtrl/tests test BUILD_DIR=/tmp/accelFFCtrl-asan \
  CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'
```

Build outputs go to `/tmp/accelFFCtrl-numerics` by default. The numerical/replay,
input, native, and full application suites are registered in `tests/tests.list`.
Fast-math is disabled. No full MagAO-X runtime or ImageStreamIO dependency is
needed for the standalone numerical tests.

The analytic suite checks calibrated ADC decoding, amplitude/phase accuracy,
coordinate rotation, latency prediction, weighted-QR equivalence, batching,
duplicate/gap/source/clock faults, weak-signal quantization, noise, and unwanted
frequency transmission. The separate replay suite generates an offline recording
using the unchanged accelSim model, then feeds **only** codes and timestamp/source
identity to the estimator. Model pointing is retained solely for later scoring.
No simulator symbol is included by the production estimator header.

The benchmark prepares all fixtures before timing and counts C++ `new` calls
while Eigen's runtime allocation guard is active. It reports ingestion and
prediction mean, p50, p99 and maximum, at 8-kHz input with both 2-kHz and 8-kHz
queries. Timing includes benchmark clocks and their resolution; it is numerical
workstation cost, not stream latency or a real-time scheduling guarantee.
The separate input benchmark also guards allocations while exercising decoding,
clock translation, discontinuity checks and 2-kHz prediction with a synthetic
source/clock. It excludes native memory mapping, clock syscalls and semaphore
wake/scheduler costs.

### Results on the implementation workstation (2026-11-05)

- Analytic/replay and live-input suites: 17 cases, 3,350,757 assertions passed.
  Seven input cases cover ordered retained batches, depth-one/finite overrun,
  torn/busy copies, timestamps/clock brackets, counter rollback/generation,
  configuration/reset, stale input and throughput accounting across loss.
- The existing numerical model and its two test files are unchanged. Combined
  AddressSanitizer/UndefinedBehaviorSanitizer validation passed 51 cases /
  3,587,355 assertions, including 34 existing accelSim model/disturbance/DM
  numerical regressions (mock sinks only). The four Python PSD tests also passed.
- Input benchmark: mean 122 ns/sample, p99 417 ns, observed maximum 49.5 us;
  zero C++ allocations and no Eigen allocations across 240,000 prepared samples.
  These measurements use synthetic transport/clocks. The 8-kHz acquisition
  period is 125 us and the default prediction period is 500 us; these are timing
  requirements to measure on Linux, not workstation worst-case guarantees.
- Full application and app-INDI tests cannot build on this macOS checkout:
  existing framework dependencies fail first (`timegm`, missing installed mxlib
  `mx/math/floatUtils.hpp`). Downloaded upstream ImageStreamIO 2.00.00 headers
  compile the native reader/test, but its library cannot build here due to GNU
  `strerror_r` and unavailable `sem_timedwait`/process-shared semaphore APIs.
  Native integration tests were added but **not executed**. Limited GCC syntax
  checks against the real base/INDI/native headers passed using temporary local
  compatibility declarations and omitting unused framework device helpers;
  this is not a full application build or runtime test. No dependency workaround
  was added to the repository. The Linux build/native/app tests above remain
  prerequisites for deployment.

## Limitations

- A sub-count sinusoid can vanish without noise/dither. Temporal estimation
  cannot recover information absent from the codes. The tests explicitly cover
  both a vanished undithered tone and the same tone with sensor noise.
- At T=0.1 s, 0.5-Hz detuning produces approximately 0.976 acceleration gain and
  17.9 degrees of phase error. The pointing conversion also has the separate
  `(f_actual/f0)^2` scale bias. Tune f0 to the signal rather than assuming accurate
  phase throughout 28–29 Hz at one fixed setting.
- Rejection of a pure unwanted tone does not guarantee that subtracting the
  estimate would improve every spectral line. The tests report hypothetical
  residuals without publishing them; at 25 Hz with default settings, subtraction
  can increase power by about 0.76 dB. A future correction stage still needs its
  own band/gain/phase acceptance, separate from numerical estimation validity.
- The complete stochastic simulator contains other modes, sensor noise and
  quantization; finite-record target SNR varies by seed and axis. One inverse at
  f0 is not the broadband inverse of all those modes. Longer memory trades
  tracking against rejection; replay results must qualify any suppression claim.
  In the tested 12-s recordings (common last 8 s scored), default-window
  target-band residual reductions range from **0.49 to 11.23 dB** for the complete
  population. They do **not** meet the earlier proposed 10-dB full-controller
  gate across all seeds/axes. Isolated 28.5-Hz stochastic-mode cases give
  **13.59–23.25 dB** with the same default window. The replay suite retains this
  distinction, plus 0.3/0.8-s memory sweeps; it does not claim physical correction.
  `accelSimDetail::accelerationFactor` weights each stochastic mode by its own nominal
  frequency squared, so tails of other modes near f0 and sensor noise are not
  exactly inverted by a single `1/omega0²` factor.
- Real accelerometer calibration, lever arm, physical registration, and eventual
  application latency remain external commissioning inputs. No physical DM
  correction or native live-stream timing has been tested on this workstation.
