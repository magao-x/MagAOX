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
increments generation and resets. Counter rollback is checked against the last
stable snapshot, including any unconsumed batch tail, and increments generation
immediately. Known unconsumed records are counted as dropped when a generation
ends; records never observed before a restart cannot be counted. Gaps, bad samples/times, stale data, configuration and explicit
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
- Lifetime `processed`, `dropped` (overwritten, rejected, or known records abandoned on restart), `invalid_frames`
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

## GCC 14 / C++20 compatibility with older Eigen

Older Eigen releases have a [C++20 array-comparison bug in LDLT](https://gitlab.com/libeigen/eigen/-/issues/2012).
It was reproduced locally with unmodified Eigen 3.3.7 and GCC 15: instantiating
the previous LDLT solve fails because a rewritten equality candidate does not
return `bool`. The exact installed exao2 Eigen version is still to be reported;
this is the failure family matching the reported LDLT comparison diagnostic.

The estimator now uses fixed-size `Eigen::LLT<Matrix3d>`. Valid output already
requires a positive-definite Gram matrix, which is the
[LLT contract](https://libeigen.gitlab.io/eigen/docs-3.3/classEigen_1_1LLT.html).
It solves the same `G2*C=Q2` equation with the same window, two right-hand sides,
conditioning/residual thresholds, units and prediction times. Failed
factorization is checked before `rcond()`, as required by LLT. Roundoff can
differ; bit-identical coefficients are not promised. No system Eigen headers,
repository-wide compiler settings or simulator/DM code are changed.

After applying the reviewed fix to the matching source checkout, a human
operator can run these commands from the checkout root on exao2. Set the compiler
to the same GCC 14 executable used for the original build (`g++` here):

```bash
set -euo pipefail
accff_cxx=g++
"$accff_cxx" --version
pkg-config --modversion eigen3
make -C apps/accelFFCtrl CXX="$accff_cxx" -W accelFFCtrlModel.hpp
accff_testbuild=$(mktemp -d /tmp/accelFFCtrl-eigen-fix-XXXXXX)
make -C apps/accelFFCtrl/tests CXX="$accff_cxx" \
  BUILD_DIR="$accff_testbuild" test benchmark
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  make -C apps/accelFFCtrl/tests CXX="$accff_cxx" \
  BUILD_DIR="${accff_testbuild}-asan" test \
  CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'
```

`-W` forces the app object to rebuild even if copied header timestamps were
preserved. These commands build/test only; they do not install, launch an app,
access INDI or open native streams. Preserve the full diagnostic if rebuilding
fails, and report the Eigen/compiler versions and test totals.

Local compatibility validation passed 18 offline cases / 3,359,793 assertions
with both Eigen 3.3.7/GCC 15 and Eigen 5.0.1/Clang, plus their sanitizer checks.
The combined current-Eigen sanitizer suite passed 52 cases / 3,596,391 assertions,
including existing simulator numerical regressions with mock sinks. Both
allocation-guarded benchmarks passed without allocations. See the engineering
plan for compiler-specific test details and numerical parity results; this does
not substitute for the GCC 14 Linux application rebuild above.

## Phase 3A: Linux build and private native validation

This phase builds/tests only. It does not install binaries, launch the observer,
attach to `acc000`, contact INDI, restart applications, or change exao2 settings.
The commands below are for a human operator; agents must not execute on exao2.
They can also be run on a configured non-instrument Linux development machine.

Use the same GCC >=14 compiler and dependencies as the working MagAO-X checkout.
The standalone test Makefile does not read `local/common.mk`; pass the same
compiler explicitly. The example below uses `g++`; change `accff_cxx` to the
actual configured compiler if necessary. Run from the checkout root, with the
reviewed Phase 3A changes present. Stop at the first failing step and preserve
its full output.

For an uncommitted review, create a source patch on the workstation:
`git diff --binary > /tmp/accelFFCtrl-phase3a.patch`. Have the operator copy it
to exao2. In a matching Phase-2 source checkout on exao2, first run
`git apply --check /tmp/accelFFCtrl-phase3a.patch`, then
`git apply /tmp/accelFFCtrl-phase3a.patch`. Stop on conflicts; no merge or forced
overwrite is needed. This applies source/test/documentation changes only, not
operational configuration. The agent does not perform the remote copy/apply.

```bash
set -euo pipefail
accff_cxx=g++
accff_report=$(mktemp -d /tmp/accelFFCtrl-3a-report-XXXXXX)
printf 'Report directory: %s\n' "$accff_report"
uname -a | tee "$accff_report/platform.txt"
git branch --show-current | tee "$accff_report/branch.txt"
git rev-parse HEAD | tee "$accff_report/commit.txt"
git status --short | tee "$accff_report/worktree.txt"
"$accff_cxx" --version | tee "$accff_report/compiler.txt"
getconf GNU_LIBC_VERSION | tee "$accff_report/glibc.txt"
pkg-config --modversion mxlib eigen3 ImageStreamIO | tee "$accff_report/versions.txt"
pkg-config --cflags --libs ImageStreamIO | tee "$accff_report/isio-flags.txt"
pkg-config --variable=prefix ImageStreamIO | tee "$accff_report/isio-prefix.txt"

make -C apps/accelFFCtrl CXX="$accff_cxx -H" 2>&1 | tee "$accff_report/app-build.txt"
make -C tests -f Makefile.one CXX="$accff_cxx" \
  t=../apps/accelFFCtrl/tests/accelFFCtrl_test 2>&1 | tee "$accff_report/app-test-build.txt"
./apps/accelFFCtrl/tests/accelFFCtrl_test '[accelFFCtrl][app]' \
  2>&1 | tee "$accff_report/app-test.txt"
make -C apps/accelFFCtrl/tests CXX="$accff_cxx" \
  BUILD_DIR="$accff_report/bin" test benchmark \
  2>&1 | tee "$accff_report/numerics.txt"
make -C apps/accelFFCtrl/tests CXX="$accff_cxx -H" \
  BUILD_DIR="$accff_report/bin" native-test 2>&1 | tee "$accff_report/native.txt"
make -C apps/accelFFCtrl/tests CXX="$accff_cxx" \
  BUILD_DIR="$accff_report/bin" native-rate-test 2>&1 | tee "$accff_report/native-rate.txt"
ldd ./apps/accelFFCtrl/accelFFCtrl | tee "$accff_report/app-libraries.txt"
ldd "$accff_report/bin/accelFFCtrlStream_test" | tee "$accff_report/native-libraries.txt"
```

If `pkg-config` cannot locate ImageStreamIO, report that failure and the known
installed header/library locations. Do not install an upstream replacement or
assume a version from the source branch. The app uses the checkout's normal
include/library search paths and explicitly links ImageStreamIO/pthreads;
standalone native tests use `ImageStreamIO.pc`. Those must resolve to compatible
headers and the same installed library. `ISIO_CFLAGS` / `ISIO_LIBS` can be supplied
to the standalone Makefile when the actual installation lacks a `.pc` file, after
its paths are established. Do not change shared make configuration for this test.

`native-test` runs four deterministic cases in a newly created, verified private
namespace, using only synthetic `accff-private-*` names. It checks 2-D and 3-D
UINT16 pairs, depth one versus retention, 12,000 timestamped 8-kHz samples in
four-frame batches across many wraps, both channels' amplitude/prediction,
finite overrun and stale retained frames, bad metadata, reader-slot cleanup,
and same-name unlink/recreation. It prints the library-written stream version
and header structure sizes. The native APIs and fields match the repository's
frameGrabber contract; installed ABI compatibility remains unverified until this
build and runtime test pass on the target installation.

`native-rate-test` is Linux-only and opt-in. It forks its own private synthetic
producer for 40,000 additional samples over approximately five seconds, using
absolute 125-us monotonic deadlines, actual realtime acquisition/write stamps,
256-plane retention and 2-kHz queries. The parent must account for every record
as processed or dropped and observe estimator readiness. It reports duration,
loss, invalids, races, resets and rates. Catch assertions and ordinary scheduler
latency are included: this is a transport/recovery test, not an RT priority or
hard deadline benchmark. A passing case with drops establishes accounting;
it does **not** establish lossless 8-kHz delivery. Save its full summary even
when it passes. Producer restarts here affect synthetic images only.

Report the directory contents above, especially the first compiler/linker error
and preceding command on failure, all test totals, the native ABI line, the
`Native 8k/2k` summary, and the ImageStreamIO paths from both `ldd` outputs.
The `-H` builds record the actual selected headers; include their ImageStreamIO
header paths and any precompiled-header indication in the report as well.
Do not report a Linux pass based on the existing macOS syntax checks. Physical
producer timestamp validity, real-host load and framework worker scheduling
remain separate qualifications. In particular, `mcp3208Ctrl` timer-mode
acquisition does not assign `m_currImageTimestamp` in this checkout; synchronized
acquisition does. Confirm the actual producer/mode before future observation.
ImageStreamIO's default `CLOCK_ISIO` may be TAI; the frameGrabber path explicitly
publishes realtime acquisition/write stamps. The controller requires that
contract and never substitutes creation time, TAI or receipt time for acquisition.

## Later operator deployment (outside Phase 3A)

The historical deployment procedure below is for a later approved observation
run after Phase 3A passes. Its install, launch, INDI and producer-configuration
steps are **not** part of the current build/private-validation task.

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
G2*C = Q2                           # fixed 3x3 Cholesky LLT, two right-hand sides
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

## Offline end-to-end validation (2026-11-05)

The opt-in study uses the unchanged simulator and production `inputProcessor`
with a synthetic depth-one source. It opens no shared-memory stream or INDI
connection. Truth is recorded first; the observer receives only UINT16 pairs,
publication counters and acquisition/write/receipt timestamps. The full-population
28-Hz truth comes from that recording's actual mode state, not a separately
seeded single-mode simulation. Existing analytic/replay/input tests are retained.

From the checkout root, select an existing Python environment containing NumPy,
SciPy, pandas and Matplotlib, then run:

```bash
accff_python=python3
make -C apps/accelFFCtrl/tests PYTHON="$accff_python" validation \
  VALIDATION_SECONDS=60 VALIDATION_DIR=/tmp/accelFFCtrl-validation-60s
make -C apps/accelFFCtrl/tests test benchmark
"$accff_python" -m unittest discover -s apps/accelSim/tests -p test_validate_psd.py -v
```

`validation` first runs four independent Python scoring/window-oracle tests,
then the hidden Catch export case and analysis. A six-second smoke study can
use `VALIDATION_SECONDS=6` and a different directory. Output includes the actual
mode specifications, compiler/STL metadata, float64 truth/prediction recordings,
and `report/accuracy.csv`, `availability.csv`, `calibration.csv`,
`recovery_study.csv`, 500-Hz trace CSVs, full-resolution 2-kHz Welch PSD CSVs and
PNG prediction/residual/PSD/horizon/availability plots. Binary column definitions
are in `tests/validate_predictions.py`. To repeat just the analysis:

```bash
MPLCONFIGDIR=/tmp/accelFFCtrl-validation-60s/.matplotlib \
  "$accff_python" apps/accelFFCtrl/tests/validate_predictions.py \
  /tmp/accelFFCtrl-validation-60s
```

The 60-second study uses 8 kHz, seeds 42/123, 10 m, noise PSD `1e-8`, ADC
zero 2048, `0.0083799` V/count, sensitivity 1.02, estimator 28 Hz, 0.1-s memory,
zero rotation and a 1-ms query horizon. The common scoring interval is 4–60 s.
The query phase makes sample age exactly 125 us: the effective horizon is
**1.125 ms**, and truth is selected nine input samples ahead. Targets must lie
exactly on the truth grid; no interpolation, rounding or extra latency is added.

| Recording / axis | Target acceleration RMSE [m/s²] | Target pointing RMSE [mas] | NRMSE | Correlation |
| --- | ---: | ---: | ---: | ---: |
| Single 28 Hz / X | 0.0003090 | 0.2059 | 15.63% | 0.98782 |
| Single 28 Hz / Y | 0.0003202 | 0.2134 | 17.78% | 0.98422 |
| Full 214 modes / X | 0.0009306 | 0.6202 | 43.18% | 0.91798 |
| Full 214 modes / Y | 0.0009655 | 0.6434 | 49.20% | 0.89291 |

NRMSE is RMSE/RMS of the reference, retaining its mean. At the exact 28-Hz Welch
bin, gain error / phase error is -0.247% / -0.640° and -0.654% / -0.311° for
single-mode X/Y, and +1.010% / -1.691° and -1.221% / -0.941° for full-population
X/Y. The CSV also reports finite-record least-squares peak amplitude/phase;
these are different statistics for a stochastic envelope. Full-population
27–30-Hz target-component residual reductions are 10.04/9.35 dB. Comparing
against **all** physical pointing gives 1.913/1.550 mas RMSE: this observer
estimates a narrowband component, and the other modes remain in that reference.

Across the entire 60 s, nominal prediction-slot availability is 99.499% with
lossless ingestion, 93.990% when one sample is overwritten every 5 s, 69.950%
with one loss per second, and 0% with one loss every 100 ms. A 2-ms worker block
every 5 s loses 15 samples per event and gives 93.963% availability. A 125-us
block every second loses no samples and produces bit-identical predictions to
the ideal case. Incorrectly reading only at 2 kHz loses about 75% of frames and
never becomes ready; the app instead ingests on source wakes independently of
query cadence. One missed sample costs **300.375 ms** from overrun detection to
the first ready query. Resets/warmup remain unchanged.

The separate batch-QR recovery investigation finds that a 20-ms history already
has reciprocal condition number 0.0285, but full-population prediction NRMSE is
586%/544% at the sampled recovery epochs. Shorter windows can be numerically
solvable while rejecting interference poorly. Retaining a mature window across
one omitted record was well conditioned in these limited offline probes;
qualifying such a bounded-gap policy is a future task, not an implemented fix.
Keep the current cold-start gate. See the engineering plan's end-to-end section
for complete scheduling counts, biases, horizon sweeps, recovery results,
sanitizer commands and limitations.

This supports longer **observation-only** evaluation with attention to actual
dropped/reset/ready counters. It does not qualify native Linux scheduling,
sensor calibration, general gap recovery or physical correction. These offline
commands make no deployment or configuration changes.

## Offline frame-loss and rate comparison (2026-11-05)

This phase preserves the production reader/reset behavior and depth-one stream.
`tests/accelFFCtrlGap_test.hpp` is an **offline-only** prototype; the application
does not include it or expose a switch to activate it. See the engineering plan's
bounded-gap investigation for the complete source trace, scheduling tables,
phase/uncertainty results, safety cases and recommendation.

In production, `inputProcessor::drain` counts overwritten publications,
rejected frames and known abandoned records as `dropped`. A counter gap resets
the fit once per observed gap, regardless of its length. Invalid timestamps/data,
clock/source changes, explicit resets and configuration changes also clear it.
Staleness detected by `predict` inhibits immediately; reset is counted when
ingestion next clears that fault. Busy writes, torn snapshots and empty polls
do not immediately add drops/resets. They can delay copying until depth-one data
has been overwritten. Semaphore posts hold no old payloads; `maxBatch` cannot
recover history from a depth-one source. Worker ingestion runs on stream wakes,
independently of its 2-kHz prediction cadence.

The user reports 2,241 drops and 2,202 resets over 602 s at 8 kHz on exao2.
That is 3.723 drops/s and 3.658 resets/s, but these aggregate counts do not
identify each reset cause or determine readiness availability. The separate
live 4-kHz run has no supplied result. All results below are deterministic
workstation simulations, not measurements or predictions of that live run.

Select an existing Python environment with NumPy, SciPy, pandas and Matplotlib:

```bash
accff_python=python3
make -C apps/accelFFCtrl/tests PYTHON="$accff_python" gap-validation \
  BUILD_DIR=/tmp/accelFFCtrl-gap-build \
  VALIDATION_DIR=/tmp/accelFFCtrl-gap-60s VALIDATION_SECONDS=60
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-gap-build test benchmark
```

The same exporter/scorer utilities serve the original validation and this study.
Each of four source recordings (8/4 kHz, single/full population) is generated
once and replayed identically to production, `gap1` and `gap7` observers across
11 schedules. Model/ADC parameters match the previous phase. Poles retain the
same **per-sample** radii at both rates, so physical stochastic decay times and
the realized trajectories differ; this is not a matched-trajectory claim.

The prototype uses the unchanged production reader as a timestamp/data validation
gate, then passes original codes and mapped acquisition times to a separate,
unchanged harmonic estimator. Physical counters remain in diagnostics; an
accepted-record ordinal supplies the numerical kernel's consecutive sequence.
No missing records are filled in. A mature fit may retain one missing frame
(`gap1`) only with monotonic source identity/counters, acquisition interval
consistent within 5% of one nominal period, elapsed time <=1 ms, and at least
one memory time (0.1 s here) since another retained gap. `gap7` explores larger
bursts with the same elapsed-time and density bounds. Cold start, numerical
checks and the 2-ms freshness bound are unchanged. Every other fault resets.

| Rate / recording | Lossless NRMSE X/Y | `gap1`, one loss/s, NRMSE X/Y | `gap1` pointing RMSE X/Y [mas] |
| --- | ---: | ---: | ---: |
| 8 kHz / single | 15.63% / 17.78% | 15.63% / 17.78% | 0.2060 / 0.2133 |
| 8 kHz / full | 43.18% / 49.20% | 43.16% / 49.21% | 0.6200 / 0.6436 |
| 4 kHz / single | 14.27% / 16.16% | 14.26% / 16.17% | 0.1953 / 0.1968 |
| 4 kHz / full | 47.04% / 57.66% | 46.96% / 57.69% | 0.4434 / 0.5029 |

Both rates use 2-kHz queries and query+1 ms targets. The normal sample age is
125 us at 8 kHz and 250 us at 4 kHz, giving effective horizons 1.125/1.250 ms
and exact truth offsets of nine/five samples. Comparisons include actual sample
age once. Accuracy CSVs include both each policy's valid slots and a common
production-valid mask; invalid intervals are never scored as zero or hidden.

With one lost frame per second, both rates count 59 lost frames over 60 s.
Production makes 60 resets (including attachment) and is ready 69.95% of nominal
slots; `gap1` makes one initial reset and is ready 99.45%. Recovery from receipt
of the post-gap sample falls from 300.375 ms (8 kHz) / 300.250 ms (4 kHz) to
the first query at that receipt. Queries missed during the blackout remain
unavailable. Three closely spaced one-frame losses still trigger cold restart;
the prototype does not preserve history through unrestricted repeated loss.

Fixed 250-us blackouts lose one frame at 8 kHz and none at 4 kHz under the
specified phase. Fixed 500-us blackouts lose three/one, respectively. Thus
4 kHz improves timing headroom for these simulated blackouts, but the table
does not establish better full-population normalized prediction accuracy.
Freshness is independently checked: 3-ms reader-only delays produce 22 stale
queries per recording/policy, all inhibited. No stale prediction reports ready.

Reports under `VALIDATION_DIR/report` include `accuracy.csv`, `availability.csv`,
`fit_quality.csv`, version metadata and before/after pointing, residual and
availability plots. Raw data live under `<rate>/<policy>`; truth is shared per
rate. Gapped valid outputs are not concatenated into a fictitious uniform PSD.
The original `validation` target remains available for lossless PSD plots.

Regression checks passed: 21 controller cases, 55 combined simulator/controller
sanitizer cases, the 60-s export's 148,400,155 assertions, and eleven Python
tests including existing PSD checks. The original six-second export is
bit-identical to the previous phase. No production/configuration change or
instrument access occurred.

Recommendation: a separately reviewed implementation limited to isolated
single-frame gaps is justified for observation. Keep full resets for startup,
identity/clock/data faults, excessive/dense gaps and stale state. Qualify the
cadence guard against recorded acquisition jitter before deployment: the 5%
prototype bound can conservatively reject real irregular/phase-slewed input.
The prototype is not a production architecture or an actuation qualification;
its two-estimator validation scaffolding should not be copied into the live loop.

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
