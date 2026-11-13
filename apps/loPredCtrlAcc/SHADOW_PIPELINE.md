# Shared components and application shadow replay

This stage adds an opt-in **non-actuating** integration to the actual
`loPredCtrlAcc` header. Ordinary Legacy/default execution and DDSPC mathematical
algorithms are preserved. No native application, accelSim, INDI server or
MagAO-X service is started by these tests. No binaries/configuration are installed.
The accelSim operational fix reported by the user permits a later operator-run
commissioning phase; this stage does not run that producer or access exao2.

## Configuration and authority

The new startup flag is `parameters.shadow_pipeline`, default `false`. It is
accepted only with `parameters.shadow=true`. It does not create an output stream,
change the existing diagnostic `ddspc_mode` selection semantics, or start training.
The supervisory request is a separate startup setting, `shadow_pipeline.requested`.
New `ddspc_pipeline` numeric and `ddspc_supervisor` text properties are read-only
and belong to this app instance. `candidate_valid` means an eligible diagnostic
proposal, never permission to actuate. Effective `unavailable` produces no valid
application candidate; any bounded ramp retained internally is diagnostic only.

A complete **synthetic replay** profile is shown below. These numerical limits
are arbitrary synthetic units and are not on-sky recommendations:

```ini
[parameters]
shadow = true
shadow_pipeline = true
num_modes = 2
history = 4
future = 3
accel_channels = 2
accel_history = 3
accel_enabled = false
qrd = true
gamma = 0.9995
covariance = 0.001
regularization = 0.1

[shadow_pipeline]
requested = accel_assisted
window_ns = 500000
lag_ns = 0
max_gap_ns = 175000
max_age_ns = 500000
recovery_frames = 10
command_limit = 3
max_step = 0.2
adc_calibration = 1
adc_capacity = 128

[adc]
enabled = false

[shmimMonitor]
shmimName =
getExistingFirst = false
```

A config file alone supplies no validated timing/applied-input contract. The
C++ adapter must explicitly call `configureShadowInputs()` with an established
source/layout/clock contract. The replay supplies `SyntheticApplied` provenance,
not hardware provenance. A future native adapter must establish `VerifiedApplied`.
Native raw WFS callbacks still discard such sessions rather than assert exposure
or application timing. The Stage1 native ADC reader remains independent; this
stage does not automatically transfer its samples into live model training.

## Shared path and timing

- `adc_features.*` implements timestamp-window integration for both simulation
  and application. It accepts oldest-first spans, excludes future/late data,
  checks temporal coverage, counters and source segments, and records weighted
  acquisition time separately from availability. The in-place overload allocates
  no storage. It preserves the simulation's float arithmetic/default results.
- `shadow_policy.*` implements requested/effective selection, recovery qualification
  and common candidate limits. It accepts model evidence without owning learners
  or hardware. The in-place interface handles configured control dimensions;
  the ADC format remains the existing two-channel pair.
- `shadow_pipeline.*` combines those helpers with the actual `CausalInputBuilder`
  and borrowed `ShadowController`. It has no publisher, hardware handle or clock
  read. Callers serialize entry points; the app uses its existing shadow mutex.
- `processShadowAdc()` consumes actual decoder output plus explicit common-clock,
  uncertainty and calibration evidence. Frontend availability is retained even
  when later than the decoder's receipt. Duplicate/old source observations do not
  become new samples. Restarts/gaps break assisted history independently of standard.
- `processShadowWfs()` freezes its current ADC window and overwrites any supplied
  prebuilt accelerometer features. It submits only completed historical pairs for
  learning, forms current states from current WFS and confirmed historical inputs,
  then calls actual `predict_cached()` and the shared supervisor. Late origin
  calls do not reconstruct a prediction retroactively.
- `processShadowApplied()` is the only source of learning commands. It accepts an
  independently established closed interval through the existing conservative
  builder. There is no route from the shadow candidate to a learning record.
- `expireShadowInputs()` invalidates common history/candidates during silence.
  A native frontend must schedule this explicitly, including when sources stop.

The replay timeline uses 200-us WFS exposures spaced 500 us apart, a 50-us optical
response lag, and decisions 50 us after exposure end. Independently generated
constant input intervals cover each lag-aligned exposure and close early enough
that certificates arrive before the following decision. Only completed intervals
are admitted. A different native latency/exposure contract must be verified or
rejected; extending the forecast horizon does not establish an input delay.

ADC windows default to 500 us, maximum hold/gap 175 us and effective age 500 us.
These are separate from the reader's 20-ms transport-health threshold. Source
identity is frozen with features and checked again when delayed confirmation
completes the record, preventing a restart exactly between full windows from
reusing old assisted training history. WFS source/clock changes invalidate common
history and require an explicitly re-established contract; no source identity is
silently trusted because its counter restarted.

The supervisor uses standard if assistance is unavailable, and unavailable if
neither model qualifies. Return to assisted requires consecutive fully qualified
ADC/history/model decisions. The app's previous **diagnostic candidate** provides
slew continuity; it is never interpreted as an actual applied input. Physical
safe-point/command handoff behavior is deliberately not defined by this stage.

## Reproduce locally

Run from the repository root on a personal workstation:

```sh
# Full numerical, historical/Legacy, application, source/timing and safety regressions.
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 \
  BUILD_DIR=build/shared-integration check

# Dedicated ADC/simulation and guarded actual-header application tests plus Python checks.
make -C apps/loPredCtrlAcc -f tests/Makefile.adc-ddspc -j2 check

# Actual app-header replay workload at 2-kHz WFS/ 8-kHz ADC, including candidate generation/learning.
make -C apps/loPredCtrlAcc -f tests/Makefile.replay -j2 check

# Longer replay and the existing core/allocation benchmarks; use consistent build-directory spelling.
apps/loPredCtrlAcc/build/shadow-replay/ddspc_shadow_replay_benchmark 10000
make -C apps/loPredCtrlAcc -f tests/Makefile.benchmark -j2 \
  BUILD_DIR=build/shared-integration-core-perf check check-allocations check-probe

# Full ASan/UBSan regression; no native instrument libraries or streams.
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 \
  BUILD_DIR=build/shared-integration-asan \
  CXXFLAGS='-O1 -g -std=c++20 -Wall -Wextra -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer' check

# Guarded application replay/concurrent diagnostics under TSan.
make -C apps/loPredCtrlAcc -f tests/Makefile.adc-ddspc -j2 \
  BUILD_DIR=build/shared-integration-tsan CXXFLAGS='-O1 -g -fsanitize=thread' check-safety

# Actual ADC worker using its explicit offline API double under TSan.
make -C apps/loPredCtrlAcc -f tests/Makefile.offline \
  BUILD_DIR=build/shared-integration-tsan-worker \
  CXXFLAGS='-O1 -g -std=c++20 -Wall -Wextra -fno-fast-math -fsanitize=thread' check-adc
```

`tests/shadow_replay.*` is an input-only adapter. It generates immutable
measurements, ADC observations and external holds before application callbacks;
there is no candidate getter on the replay sink interface. Actual ADC conversion
uses `AdcCapture`. Actual application configuration, training callbacks, input
seams, model processing, prediction and output guards are compiled from the real
header with the existing stream-free framework/ImageStreamIO doubles. This does
not substitute another controller or call the native application's main entrypoint.

## Validation record

Local validation on 2026-11-12, `ASTR-LT-428.local` (Apple LLVM 21.0.0),
strict floating point throughout:

| Check | Result |
| --- | --- |
| Full ordinary-header numerical/historical/Legacy/application suite | 114 cases passed |
| Full suite under ASan/UBSan | 114 cases passed; no sanitizer findings |
| Guarded `DDSPC_SHADOW_ONLY` application replay under TSan | 21 cases passed; includes concurrent diagnostic reads |
| Actual ADC reader/worker under TSan using the explicit offline transport double | 13 cases passed |
| Dedicated ADC, numerical/simulation, CLI and example-config checks | Passed; 11 ADC cases, 25 numerical cases, 2 CLI and 2 config checks |
| Build-only tooling checks / pinned Legacy snapshot verifier | 8 Python checks passed / original 15 files verified |
| Repeated offline fallback comparison | All 24 scenarios / 72 trajectories byte-identical to the archived comparison CSV |
| Existing core benchmark, allocation contracts and allocation-probe self-test | Passed from a fresh build directory |
| Extracted in-place window/policy allocation checks | Zero Eigen/C++ heap requests at 2, 3 and 10 control dimensions |

The new application cases exercise initial unready state, explicit independent
training requests, finite conditional forecasts, 1/5/25/100-ms ADC loss, source
restart at a window boundary, common WFS/applied-evidence loss and recovery,
clock/source changes, reversed time, missed decisions, stale ADC, jitter and
independent cadence drift. Unconfirmed driver-prepared intervals never train
either learner. Different supervisory requests/slew limits produce different
candidates and **bit-identical learned matrices** from the same confirmed records.
Caller-supplied ADC matrices are ignored; forged feature fields cannot qualify
assistance. Shutdown and the unconditional guarded acquisition/publication
entry-point refusals are exercised without invoking hardware or native streams.

The longer application replay uses 10,000 WFS frames (5 s virtual time), nominal
2-kHz WFS/8-kHz ADC, QR learning and two control modes. Timings exclude recording
generation and omit the first 200 WFS intervals; `frame_total` sums actual ADC,
confirmed-input and WFS callback work between consecutive WFS origins:

| Model profile | Frame p50 / p95 / p99 (µs) | Maximum (µs) | Intervals over 500 µs | Valid / assisted candidates |
| --- | --- | --- | --- | --- |
| History 4, ADC history 3, horizon 3; prior .001, gamma .9995, lambda .1 | 14.958 / 15.792 / 17.166 | 72.624 | 0 / 9,800 | 9,974 / 9,730 |
| Example history 15, ADC history 5, horizon 3; prior .00001, gamma 1, lambda 100 | 102.834 / 105.042 / 110.125 | 442.709 | 0 / 9,800 | 9,917 / 9,507 |

Full component measurements and final conditional prediction diagnostics are in
[the application replay CSV](tests/results/shared_shadow_replay_2026-11-12.csv).
These prediction scores use the existing conditional forecast metric and are
neither free-running prediction error nor a closed-loop performance comparison.
Both application profiles recorded 59,995 Eigen allocation permission checks and
10,007 C++ allocation requests across the full recording. The extracted in-place
helpers allocate no storage, but existing causal pairing still allocates and
learning remains synchronous under the model lock. The complete path is not
certified allocation-free or hard real time.

[The existing core benchmark CSV](tests/results/shared_shadow_core_2026-11-12.csv)
also retains larger ten-mode evidence: with its separate history 5 / ADC history
20 profile, builder-plus-pair training p99 was 519.084 µs for RLS and 588.541 µs
for QR; 345/2,000 and 2,000/2,000 measured operations exceeded 500 µs, respectively.
The benchmark passing checks establishes execution/allocation contracts, not a
timing pass for those dimensions. Native worker qualification remains necessary.

An initial existing-core run mixed old relative-path cached objects with new
absolute-path dependencies after the input-record layout changed and aborted.
A fresh build completed all checks. Keep `BUILD_DIR` spelling consistent and
use fresh directories after record ABI/compiler/flag changes. Native ImageStreamIO
ABI, framework scheduling, INDI-server interaction and instrument behavior are
not validated by these doubles or virtual-cadence workload timings.

## Requirements before live exao2 tests

1. Build/link the guarded native application on a non-instrument Linux development
   system with the real MagAO-X/ImageStreamIO dependencies. This Mac replay build
   verifies the real app header using lifecycle doubles, not native ABI/runtime.
2. Implement/qualify native WFS payload/metadata snapshots, camera-frame/exposure
   association, common-clock mapping and uncertainty. Raw pointer callbacks do not
   provide those certificates.
3. Establish the actual applied-input source, all combiner/clipping/calibration
   semantics, modal projection and effective optical hold intervals. A request,
   SDK success alone, or the currently inspected BMC `_shape` post is insufficient.
4. Supply validated native ADC sample envelopes/calibration and source identity
   to the shared pipeline. Real AD7689 conversion/channel timestamps and its
   calibration profile need their own contract.
5. Schedule a silence watchdog, measure native processing and model age, and
   qualify worker/snapshot promotion or bounded storage if synchronous pairing
   and learning cannot satisfy the actual deadline. Do not weaken readiness to
   hide overload.
6. Review requested/effective behavior, source restart policy, recoverability and
   calibrated candidate constraints on independently recorded data. Repeat the
   separate 56-Hz identification/regularization/physical-plant validation.
7. Only then prepare a separately authorized, uniquely named, output-inhibited
   operator-run accelSim/ADC/WFS commissioning session. No DM ownership or output
   enable follows from this replay work.

## Files and compatibility

New shared files are `adc_features.*`, `shadow_policy.*`, and `shadow_pipeline.*`.
Application integration changes `loPredCtrlAcc.hpp` and its Makefile. Causal input
records add ADC incarnation fields; pairing rules and DDSPC equations are unchanged.
The old simulation uses aliases in `tests/offline_control.*`. New replay/test/
benchmark files are `tests/shadow_replay.*`, `tests/shadow_app_test_support.hpp`,
`tests/ddspc_shadow_pipeline_test.cpp`, `tests/shadow_replay_benchmark.cpp` and
`tests/Makefile.replay`. Offline Makefiles and documentation are updated. Original
`loPredCtrl`, accelSim, tracked historical results, calibration and installed
production configuration are untouched.
