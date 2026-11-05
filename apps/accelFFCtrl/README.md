# accelFFCtrl numerical estimator

Phase one provides `accelFFCtrlDetail::harmonicEstimator`, a standalone,
single-owner numerical acceleration-to-pointing estimator. It consumes raw ADC
pairs and returns physical disturbance estimates. This directory has no running
MagAO-X application, stream reader/writer, INDI controls, or actuator publication.

The engineering record is
[accelSim-28hz-feedforward-controller.md](../../agents/plans/2026-11/accelSim-28hz-feedforward-controller.md).
The existing simulator, its model/RNG/timing, and its DM injection are unchanged.

## Input, units, and timing

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
monotonic domain**. A numerical epoch of zero is valid. The caller must eventually
perform any realtime-to-monotonic mapping and stream-plane coherence checks;
this core reads no clocks or metadata itself. Subtract integer timestamps before
converting intervals to seconds, preserving precision even at large epochs.

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

Build outputs go to `/tmp/accelFFCtrl-numerics` by default. The same two suites
are registered in `tests/tests.list` for configured repository test builds.
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
  correction or live-stream timing has been tested in this phase.
