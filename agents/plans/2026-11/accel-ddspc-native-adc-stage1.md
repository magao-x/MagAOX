# Native ADC input: Stage 1 implementation record

2026-11-12, `parkertjohnson1/accel_ddspc`, base `91d3b20e`.
Implementation was explicitly requested after the earlier read-only integration
audit. Work and safe local tests ran on personal Mac `ASTR-LT-428.local`.
No native application, producer, instrument command, remote connection, deployment
or DM output was run. No commit/push was requested or performed.

## Implemented scope

- `adc_input.hpp/.cpp`: transport-independent calibration and bounded retention.
  Input is an ordered unsigned two-channel pair with publication counter and
  original acquisition/write realtime timestamps. Defaults match accelSim:
  counts 0–4095, zero2048, .0083799 volts/count and 1.02 volts/(m/s²).
  Both channels have independent zero, scale, sensitivity and polarity. Encoding
  bounds can extend through65535; rails are rejected by default. The profile name
  records identity, while explicit numeric configuration defines decoding. No
  AD7689 calibration, signed encoding or acceleration-to-displacement mapping is
  assumed.
- `adc_reader.hpp/.cpp`: sole-owner existing-only ImageStreamIO input worker,
  default `acc000`, adapted from the simulator worktree's
  `accelFFCtrlStream.hpp` transport. Supports CPU UINT16 2×1×D temporal images
  and a depth-one pair; checks layout/allocation bounds and brackets batches
  with write/counter/geometry checks. Uses per-plane metadata when D>1.
  There is no dependency on that worktree or its harmonic estimator. Only
  reader PID-slot bookkeeping and its own semaphore consumption write shared
  memory; producer pixels, timestamps, counters and configuration are untouched.
- App configuration `[adc]` is disabled by default. Enabling it requires
  `parameters.shadow=true`, checked before helper loading and again at startup.
  Strict `DDSPC_SHADOW_ONLY=yes` remains the commissioning build. The separate
  Legacy FLOAT accelerometer monitor stays disabled in shadow mode. ADC numbers
  and exact provenance strings are exposed as read-only `adc`/`adc_source`
  properties, independently of `ddspc_data` validity.
- New [ADC_INPUT.md](../../../apps/loPredCtrlAcc/ADC_INPUT.md) documents numerical
  units, configuration, diagnostics, recovery and qualification limits.
  A separate ADC shadow example preserves the original WFS-only example.

## Timing, gaps and recovery decisions

Attachment starts at the newest committed sample, avoiding imported unknown
history. Thereafter retained producer planes are consumed in counter order,
with at most256 copies per batch. The consumer ring is preallocated and bounded
(default4096, maximum65536); callers obtain owned oldest-first copies outside
acquisition. No stream pointers escape. The reader lock is independent of model
locks and no model work runs on the acquisition worker.

A monotonic/realtime/monotonic local clock bracket establishes one mapping per
source generation. Integer timestamp differences preserve nanoseconds at epoch
magnitudes; floating-point epoch subtraction is avoided. Future/inverted/stale
times and non-increasing acquisition times are rejected. Acquisition high-water
persists across segment invalidation. Clock reversal, excessive bracket span or
offset change latches a fault until a new source session. A clock reset never
silently certifies alignment.

Publication losses, invalid samples, stale data or excessive acquisition gaps
clear the retained segment. Busy or raced snapshots are retried without advancing
the consumer cursor. Publication losses are counted once a stable batch proves
them. Missing publication counters and rejected samples count as `dropped`;
consumer retention evictions count separately as `overwrites`. Acquisition gaps
can occur without missing counters and count separately as `timing_gaps`.
Measured rate uses accepted acquisition intervals, not configured FPS or semaphore
posts. Raw rejected pairs remain visible with cleared acceleration and valid0.

Backing-file removal/replacement and source counter rollback invalidate prior
history and start a new generation on attachment. Shutdown joins the worker,
invalidates retention and releases its mapping/reader slot. Polls and notification
draining are bounded; the ordinary worker's scheduling is not realtime-qualified.

## Safe local validation

The actual reader translation unit is compiled against an explicit offline
ImageStreamIO API double. Immutable threaded fixtures use only private ordinary
files in a process-specific temporary directory and fake semaphore functions.
The production Makefile has no fake include path. No native stream was accessed.

| Check | Result |
| --- | --- |
| Full strict offline C++ suite | 81 cases passed |
| Guarded shadow-only application suite | 9 cases passed |
| Focused ADC capture/worker suite with ASan/UBSan | 6 cases passed; no sanitizer findings |
| Focused ADC capture/worker suite with ThreadSanitizer | 6 cases passed; no sanitizer findings |
| Native-build orchestration unit tests | 8 passed; these use fake build tools |
| Original and new shadow example configuration tests | 2 passed |

Coverage includes unsigned/default/custom decoding, invalid calibration/rails,
bounded retention and producer overrun, duplicates, unstable snapshots, invalid
geometry/type/counters/timestamps, silence expiry, timing gaps, counter rollback,
clock faults, large-epoch nanosecond precision, actual worker disconnect/reconnect,
per-plane timestamps in a multi-plane cube, joined cleanup and real app lifecycle
output/training inhibition. The historical source verifier checked the pinned
15-file Legacy snapshot. Builds are ignored; tracked numerical reports were not
regenerated. Existing bundled INDI temporary-reference compiler warnings are
outside this change.

Reproducible commands are in ADC_INPUT.md. Full and guarded suites use
`BUILD_DIR=build/adc-stage1`; sanitizer targets use separate
`build/adc-stage1-sanitize` and `build/adc-stage1-tsan` directories. Python config
tests run with `python3 -m unittest discover -s apps/loPredCtrlAcc/tests -p
handoff_config_test.py`. No producer or original `apps/loPredCtrl` changes are
part of this implementation.

## Remaining work

1. Qualify a fresh strict guarded native Linux build against the installed
   ImageStreamIO ABI. This Mac has no native headers/libraries or Linux runtime;
   API-double tests do not establish native linking or live scheduling.
2. Validate an isolated private simulator input with all output helpers absent:
   type/shape, raw codes, channel order, numerical calibration, acquisition versus
   write timestamps, actual retention depth, rate, age, gap accounting and restart
   behavior. Depth1 can lose overwritten samples; the reader cannot recover them.
   Producer retention changes require separate review and are outside Stage1.
3. Establish WFS exposure/source identity and common-clock correspondence, plus
   actual applied-command ingestion and calibrated modal/sign mapping. ADC capture
   alone is not verified synchronized WFS information.
4. Build causal ADC features aligned to each WFS interval, including filtering/
   resampling, feature units/schema, provenance and silence expiry, then feed
   `CausalInputBuilder` and the independent models in shadow mode.
5. Compare training/predictions and resource usage under the intended 2-kHz WFS
   workload before considering later control work. Raw native WFS callbacks still
   invalidate both models; ADC-valid does not enable training or predictions.

Any later instrument-host observation must be performed by the operator.

## Stage 1 review and correction (2026-11-12)

The follow-up review began with a clean working tree: the Stage1 implementation
had already been committed as `fdad8ac8`. The user explicitly authorized fixes,
safe local retesting, commits on the feature branch and operator-only exao2
instructions. No application or live stream was opened during this review.

One restart-detection defect was found. While draining a retained producer cube
in bounded batches, the consumed cursor may trail the latest stable publication.
Comparing rollback only with that cursor could miss a producer reset to a counter
between those two values. Capture now retains the latest stable publication count
and detects rollback against both it and the cursor. New source generations
clear both values; raced snapshots cannot advance the stable publication count.
The regression observes12, consumes through6, then resets to7 and requires a new
generation with only the fresh sample retained.

Additional worker tests cover atomic backing-file replacement without an observed
missing-file interval and shutdown while the source is absent. Fixtures remain
private ordinary files with immutable payload storage and fake semaphore APIs.
Review of calibration, integer timestamp mapping, acquisition ordering/freshness,
bounded storage, capture locking, worker ownership and joined shutdown found no
other required functional changes. App lifecycle and INDI guards continue to
skip every output helper and reject Legacy enabling while shadow is active.

Retesting passed: 83 full offline C++ cases; 9 guarded shadow application cases;
8 focused ADC cases under ASan/UBSan and under ThreadSanitizer; 8 native-build
orchestration unit tests. The latter are fake-tool tests, not native compilation.
The historical15-file original-controller verifier also passed.

The environment is still Darwin/arm64. There are no actual ImageStreamIO headers
under the repository/development include paths, no ImageStreamIO pkg-config entry,
and no Lima Linux instance. A compile-only attempt with no stub include path,
`c++ -std=c++20 -fno-fast-math -pthread -fsyntax-only
apps/loPredCtrlAcc/adc_reader.cpp`, stops at the missing
`ImageStreamIO/ImageStreamIO.h`. No installed native ABI, native linking, process
publication semantics, 6/8-kHz scheduling or live restart behavior is certified.
These remain explicit operator qualification steps before Stage2.

The example's accelSim calibration matches the simulator source at `07e09306`:
unsigned clipped0–4095 codes, channelsX/Y, zero2048, .0083799 volts/count,
sensitivity1.02. It is valid only if the running simulator retains those settings;
its default rate is6kHz, not an assumed8kHz. The WFS placeholder is intentionally
not a usable stream name. The ADC-only operator procedure blanks that field in a
private config, retaining shadow mode and the compile-time guard. It supplies no
DM output section or Legacy enable command.

The user subsequently took responsibility for `git add` and commits after the
sandbox refused writing the worktree index outside its writable roots. No agent
commit was made. [ADC_EXAO2_TESTING.md](../../../apps/loPredCtrlAcc/ADC_EXAO2_TESTING.md)
now supplies the operator-only fresh strict build, ADC-only private configuration,
foreground reader/private FIFO broker, raw/physical INDI inspection and shutdown
procedure. It leaves the simulator and the original controller untouched. The
full example remains unchanged; only the operator's private copy blanks WFS.

## Files affected

Paths below are relative to `apps/loPredCtrlAcc/`, except the plan entries.

| Area | Files |
| --- | --- |
| Capture and native transport | `adc_input.hpp`, `adc_input.cpp`, `adc_reader.hpp`, `adc_reader.cpp` |
| Application and production build | `loPredCtrlAcc.hpp`, `Makefile` |
| Offline test build and cases | `tests/Makefile.offline`, `tests/ddspc_adc_test.cpp`, `tests/ddspc_app_test.cpp`, `tests/handoff_config_test.py` |
| Offline native API fixture | `tests/adc_transport_fixture.hpp`, `tests/adc_transport_fixture.cpp`, `tests/native_stubs/ImageStreamIO/ImageStreamIO.h` |
| Handoff documentation and example | `ADC_INPUT.md`, `ADC_EXAO2_TESTING.md`, `EXAO2_TESTING.md`, `tests/README.md`, `examples/loPredCtrlAcc-adc-shadow.conf.example` |
| Engineering records | `agents/plans/2026-11/accel-ddspc-controller.md`, this Stage1 record |
