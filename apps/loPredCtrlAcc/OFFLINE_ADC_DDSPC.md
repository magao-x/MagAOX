# Standalone offline ADC–DDSPC validation

Everything below runs on a personal development machine with accelSim stopped:
no producer/worktree dependency, INDI server, native ImageStreamIO stream,
MagAO-X application/service, `/opt/MagAOX` write, installation or physical DM.
DDSPC mathematics and default Legacy behavior are unchanged. Shared components
and an opt-in application replay path are now described in
[SHADOW_PIPELINE.md](SHADOW_PIPELINE.md); native input adapters remain unimplemented.
The original simulation fallback policy, timing contract, comparison results and
remaining limitations are documented in [OFFLINE_FALLBACK.md](OFFLINE_FALLBACK.md).

## Audit and reuse

| Actual implementation | Offline use | Remaining dependency/boundary |
| --- | --- | --- |
| `adc_input.hpp/.cpp`: `AdcCapture`, `AdcSource` | Actual calibration, metadata checks, integer timestamp mapping, retention, gaps/recovery | Standard library only; no native API or clocks |
| `adc_reader.cpp`: `AdcReader` | Existing worker recovery tests use the explicit ImageStreamIO API double | These tests are separate from core conversion tests; native ABI/scheduling is not emulated |
| `causal_input.hpp/.cpp`: `CausalInputBuilder` | Existing synthetic exposure/applied-input provenance tests run in the numeric target; the opt-in pipeline freezes shared ADC features before pairing | Native source/clock/applied-input adapters remain required |
| `shadow_controller.*`, `ar_controller.*`, RLS/QR | Actual independent models, rank/readiness, prediction, learning and reference control solve | Existing Eigen/mxlib headers required; no MagAO-X/INDI/ImageStreamIO library linkage |
| `loPredCtrlAcc.hpp` | Real header/configuration/callback/output guards compiled with `DDSPC_SHADOW_ONLY` and lifecycle doubles | Uses real INDI value objects and a capture-only driver; not the native MagAOXApp/framework runtime |

The ADC unit file previously included transport fixtures unnecessarily. Its core
cases now stand alone; the two worker cases moved to
`tests/ddspc_adc_transport_test.cpp`. Existing complete regression targets still
run both. The learner headers retain some unnecessary mxlib image includes to
avoid an unrelated numerical-core refactor. Older simulation targets with
external accelSim headers or historical reference snapshots remain available but
are not prerequisites of the new numeric tests. The dedicated safety build uses
only its explicit sources, bypassing unused historical replay oracles.

## Exact build and execution

From this repository's root, on `parkertjohnson1/accel_ddspc`:

```sh
# Complete new offline checks: ADC, synthetic source, real models/input pairing,
# guarded app safety, CLI artifact ownership and example configuration.
make -C apps/loPredCtrlAcc -f tests/Makefile.adc-ddspc -j2 check

# ADC-only: C++20/compiler/standard library and bundled Catch; no pkg-config packages required.
make -C apps/loPredCtrlAcc -f tests/Makefile.adc-ddspc check-adc

# Real numerical models: existing Eigen/mxlib development headers via pkg-config.
make -C apps/loPredCtrlAcc -f tests/Makefile.adc-ddspc check-simulation

# Run the standalone simulation and print parameterized CSV metrics to stdout.
make -C apps/loPredCtrlAcc -f tests/Makefile.adc-ddspc simulation
apps/loPredCtrlAcc/build/adc-ddspc-offline/adc_ddspc_sim --help
```

The complete target additionally needs Python3's standard library for CLI/config
tests. It compiles the repository's INDI value classes for application tests;
it does not link a driver or connect to a server. `tests/catch2` is bundled.
No dependency installation is part of these commands. Default build outputs stay
under ignored `apps/loPredCtrlAcc/build/adc-ddspc-offline/`.

To export a fresh run, choose an output directory that does **not** exist:

```sh
apps/loPredCtrlAcc/build/adc-ddspc-offline/adc_ddspc_sim \
  --output apps/loPredCtrlAcc/build/adc-ddspc-offline/run-default

apps/loPredCtrlAcc/build/adc-ddspc-offline/adc_ddspc_sim \
  --jitter-ns 10000 --sensor-noise .015 --drop-start 9600 --drop-count 200 \
  --output apps/loPredCtrlAcc/build/adc-ddspc-offline/run-faults

apps/loPredCtrlAcc/build/adc-ddspc-offline/adc_ddspc_sim \
  --coupling 0 --output apps/loPredCtrlAcc/build/adc-ddspc-offline/run-uncoupled
```

Each writes `metrics.csv`, `residuals.csv`, `psd.csv`, `learning.csv`,
`decisions.csv` and `transitions.csv` only in the explicitly
requested new directory. A second invocation refuses existing results. Use a new
directory name for repeats; no historical report is overwritten automatically.
Metrics include parameters, mean per-axis true residual variance, conditional
prediction RMSE, actual update/readiness/proposal counts, coefficient norms,
bounded-command maxima, command RMS, integrated28–29/54–58-Hz residual power,
standard-fallback/safe/longest-unavailable durations, maximum switching step, and
ADC rate/loss accounting. The latter timing/authority evidence is exported at
every evaluation decision, with transition-only rows in `transitions.csv`. `NA` means no scored
predictions, including open loop. `learning.csv` records actual update/rank/
readiness/RMSE/coefficient-norm evidence every 100 WFS steps, at first readiness,
and at the final step. Residual traces contain both axes. PSD is the
mean of the two axes' Hann-window periodograms, in synthetic WFS²/Hz,0–200 Hz,
with `df=2000/evaluate`. Evaluation includes the initial calibration-to-feedback
transition. PSD/variance are not measured telescope data.

Options include training/evaluation lengths, history/horizon/ADC-history,
QR versus RLS, forgetting factor, information prior, literal control penalty,
coupling, plant pole, signed DM gain, command bound, source seed, sensor/WFS/force
noise, ADC acquisition jitter/drift, FIFO delivery latency/variation, periodic dropped
reads, single/repeated bursts, recovery hysteresis and slew bounds. ADC nominal
rate is8 kHz with a configurable oscillator error; WFS stays2 kHz. An independent
white-ADC option changes sensor information while preserving the physical plant.

Sanitizer checks use separate build directories:

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.adc-ddspc -j2 \
  BUILD_DIR=build/adc-ddspc-sanitize \
  CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' \
  check-adc check-simulation

# Actual reader thread with its offline API double, plus core/synthetic ADC cases:
make -C apps/loPredCtrlAcc -f tests/Makefile.offline \
  BUILD_DIR=build/adc-stage1-tsan \
  CXXFLAGS='-std=c++20 -O1 -g -Wall -Wextra -fno-fast-math -fsanitize=thread' check-adc

# Broader existing regression suite; local Git snapshots are required only by these historical comparisons.
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 BUILD_DIR=build/adc-stage1 check
```

## Signals, units and timing

`SyntheticAdcSource` owns one raw pair in memory. Every virtual 125000 ns it
generates simultaneous X/Y acceleration from 28.5-Hz and 56-Hz tones plus broadband
Gaussian force, adds independent Gaussian sensor noise, and encodes UINT16 counts.
The default numerical calibration is `accelSim12`: zero 2048, .0083799 volts/count,
1.02 volts/(m/s²), polarity+1, range 0–4095. This is a simulator calibration, not
an AD7689/on-sky calibration. Each axis can be calibrated independently through
the actual `AdcConfig` interface and unit tests.

Actual `AdcCapture::drain()` performs every decode and validation. There is no
test replacement for its conversion. Unit tests cover negative/positive counts
around zero, independent offsets/gains/sensitivities/polarities, rails/invalid
calibration, wrong dimensions/type/CPU flags before payload access, stale/future/
reversed times, duplicates, races, ring order/capacity/evictions, producer overruns,
clock faults, counter rollback and recovery. Frequency/amplitude checks fit
sinusoids using the actual accepted acquisition timestamps, including jitter and
missing samples. Quantization/noise tolerances are explicit in the tests.
The synthetic generator rejects out-of-range encoded waveforms rather than
silently clipping an experiment; rail-policy cases inject endpoint codes directly.

Jitter changes acquisition time and waveform phase together. Acquisition/write/
receipt times remain distinct. Dropped reads emulate a depth-one source
overwriting an unread frame; the next stable counter proves the loss. They are
not silently resampled or filled. Native worker bookkeeping is tested separately
against fake APIs and process-private ordinary files, never the real stream
namespace. The core/simulation executables do not link that API at all.

At each2-kHz WFS decision, test-only `acceleration_window()` integrates a physical
half-open window of decoded measurements using causal zero-order hold. Samples
must be received by the deadline; full temporal coverage, contiguous counters/
source segments, a175-µs maximum hold/gap and a500-µs effective-age limit are
required by default. The weighted acquisition timestamp is retained. Acquisition
jitter and independent ADC cadence drift change weights/counts rather than moving
index groups across exposure boundaries. Window lag is explicit and configurable.
The virtual delivery queue preserves publication order. This tests the actual
capture processor, not native ImageStreamIO retention or scheduling.

The declared synthetic plant is

`d_(i+1)=rho*d_i+k*M*a_i`, `y_i=d_i+g*u_(i-1)`,

where `M=[[1,.15],[-.1,1]]`, `a_i` is latent mean physical acceleration, `k`
is configurable coupling, `g` the signed simulated DM gain, and `u_i` a componentwise bounded
absolute input. True plant acceleration never enters prediction features.
WFS observations add independent measurement noise. WFS/input units are synthetic;
the model is not a telescope, DM hardware model, or physical acceleration-to-tilt
calibration. All three trajectories regenerate identical exogenous realizations.

The actual `ShadowController::predict_cached()` uses the previous validated fit
and current causal `[y_p;u_p;a_p]` state. Only after the synthetic plant consumes
the bounded input is the current record supplied to `ShadowController::process()`.
Both independent learners receive that trajectory's identical WFS/applied-input
records. Readiness, rank, RLS/QR updates and the regularized control solve are
production implementations. Default history 4/horizon 3/ADC history 3, QR,
gamma .9995, information prior .001 and literal lambda .1 are test parameters.

Training uses common Gaussian input excitation with sigma .3. Controlled evaluation
retains sigma .02 excitation; open-loop evaluation is neutral. A common absolute
bound3 and per-frame slew limit.2 apply to every command, including calibration.
The assisted-requested trajectory falls back to the corresponding ready standard
model, or ramps toward zero without exploration when neither qualifies. Return
to assisted requires10 consecutive fully qualified ADC/WFS/model decisions.
Both learners receive each actual constrained consumed input. No production
selection property or Legacy controller participates in this policy.

Prediction RMSE is the real frozen-origin **conditional** score, using actual
later inputs only for scoring. The summary accumulates score across recovered
segments, including calibration; it is not a free-running forecast or a signal
made available to the controller. Existing `CausalInputBuilder` exposure/provenance
tests also run, separately; this simulation directly supplies synthetic records.

## Initial hold-policy results (historical)

These numbers predate the bounded fallback/timestamp-window changes. Current
results are in [OFFLINE_FALLBACK.md](OFFLINE_FALLBACK.md); the old hold policy is
not the default simulation behavior.

Local strict-Clang run, seed 721, 2000 training WFS frames and 4000 evaluation frames:

| Scenario | Open variance | Standard variance | Assisted variance | Standard/assisted conditional RMSE |
| --- | ---: | ---: | ---: | ---: |
| Default coupled source | .24845735 | .00263336 | .00247875 | .0238283 / .0148509 |
| ±10µs jitter, sensor sigma .015, 200 lost ADC reads | .24845721 | .00263776 | .01514936 | .0238171 / .0169195 |
| Zero physical ADC-to-WFS coupling | 8.52e-7 | .000396708 | .000396649 | .00501212 / .00502756 |

Default assistance reduces variance by about 5.87% relative to standard DDSPC.
It is not uniformly better by frequency. Integrated default PSD band powers are:

| Band | Open | Standard | Assisted |
| --- | ---: | ---: | ---: |
| 27–30Hz | .20826343 | .001750763 | .001710064 |
| 54–58Hz | .03918487 | .000294169 | .000328965 |

The assisted 56-Hz band is about 11.8% worse than standard despite the overall
variance improvement. The 25-ms dropout causes a held-input transient: assisted
variance is about 5.74 times standard. Both recover readiness, report exactly 200
lost ADC frames and have zero numerical-fault/gate-violation frames. The zero-
coupling comparison shows negligible assistance; evaluation probing itself
disturbs the otherwise quiet plant. The tiny open variance includes the initial
held calibration input. These outcomes are reported without tuning away failures.

Changing compilers/standard-library Gaussian samplers or parameters can change
these results. They validate this synthetic scenario, not universal suppression
or performance on exao2.

## Shadow safeguards and automatic-test limits

Reviewed production guard locations include `loPredCtrlAcc.hpp:833` (startup-only
configuration),`:1416` (startup),`:1496` (output-helper skip),`:1650`/`:1880`
(WFS shadow path) and`:2103`–`:2164` (direct acquisition/publication rejection).
ADC transport opens existing images and writes only reader PID/semaphore
bookkeeping; it does not create/destroy/update image payloads. The offline core
does not link this transport. The broader API-double tests do not use native
shared memory, and the fake API offers no create/destroy/update primitive.

The guarded real-header tests check rejection of false/missing shadow mode and
Legacy controls, untouched sentinel output memory, zero output-helper lifecycle
calls, no Legacy controller construction, read-only ADC diagnostics, and captured
INDI updates confined to the fixture's own device identity. There is no server
connection or acquisition hardware in these executables.

**The macro is output inhibition, not namespace/security isolation.** A native
app still advertises its own diagnostics and creates app-owned log/PID/FIFO files.
It does not reserve a unique INDI name or prevent attaching the wrong broker.
Reusing a production name can therefore collide with production properties.
Unique native instance identity/private broker and installed native ABI/framework/
destructor/scheduler/producer-concurrency behavior cannot be certified by these
doubles. Existing Legacy defaults remain in an ordinary unguarded production
build; use the compile-time guard for any separately authorized commissioning.

No native app, producer, service, broker, installation, merge or push was run here.
Live applied-input provenance, WFS exposure timing, native ADC feature alignment,
fallback policy and real-time throughput remain separate work. Passing this suite
does not enable either native model or provide control authority.

## Local validation record

Current fallback checks are detailed in [OFFLINE_FALLBACK.md](OFFLINE_FALLBACK.md).
The broader94-case/build-orchestration/Legacy comparisons below refer to the
previous infrastructure pass; they were not rerun for the test-only policy change.

| Check | Result |
| --- | --- |
| Dedicated ADC/core/synthetic executable | 11 cases passed |
| Dedicated actual-model simulation, causal input, windows and policy | 25 cases passed |
| Guarded application safety executable | 10 cases passed |
| CLI ownership/argument tests and example configuration tests | 2+2 Python tests passed |
| Broader existing/new offline regression executable | 94 cases passed |
| Existing native-build orchestration tests | 8 passed; fake build tools, not a native build |
| ASan/UBSan ADC and model/input executables | 11+25 cases passed; no findings |
| ThreadSanitizer ADC/core/synthetic/actual-worker API-double executable | 13 cases passed; no findings |

No failing tests remain. The numeric/core binaries were also inspected for
unresolved ImageStreamIO, DM-command, socket/connect and INDI-command symbols;
none were present. This supplements the explicit source/link lists and is not a
security sandbox claim. The application safety build compiles only its declared
sources, without historical replay object dependencies. The full legacy source
verifier still confirms the pinned 15-file original controller snapshot. Existing
bundled INDI temporary-reference compiler warnings are outside this change.

The original test-infrastructure stage added/changed files under `apps/loPredCtrlAcc/tests/`, this guide, the tests
README, and `agents/plans/2026-11/accel-ddspc-offline-adc-tests.md`. No production
source/configuration or tracked historical result was changed in that stage.
The later shared-component and opt-in application integration, including its
current validation record, is documented in [SHADOW_PIPELINE.md](SHADOW_PIPELINE.md).
