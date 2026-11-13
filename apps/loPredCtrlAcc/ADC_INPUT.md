# Stage 1: native raw ADC capture

This opt-in reader observes an existing ImageStreamIO input in shadow mode. It
never creates, destroys, updates or zeroes an image, and has no DM or DDSPC-model
interface. ImageStreamIO reader-slot/PID bookkeeping and consumption of its own
notification semaphore are the only shared-memory writes. No native app, producer,
service or instrument was launched during implementation.

Use a freshly built `DDSPC_SHADOW_ONLY=yes` application and the
[ADC shadow example](examples/loPredCtrlAcc-adc-shadow.conf.example). The normal
Legacy defaults are preserved; `adc.enabled=true` with `shadow=false` is rejected
before helpers load, and again at startup. The separate `parameters.accel_enabled`
Legacy FLOAT reader stays off. No output section/channel is needed.
The ADC example now leaves `shmimMonitor.shmimName` empty and explicitly sets
`parameters.shadow_pipeline=false`: it is ready to copy for ADC-only observation,
without filling in WFS/application timing or enabling either model's training.

For operator-run native build, ADC-only configuration and private INDI commands,
use [ADC_EXAO2_TESTING.md](ADC_EXAO2_TESTING.md). The review correction detects
producer rollback against the latest stable publication even while bounded
capture is still draining older planes.

## Configuration

All keys are startup-only in `[adc]`. `enabled` defaults false. Changing the
profile/calibration requires a fresh reader session; `profile` is a recorded
identity, not a selector that silently changes calibration values.

| Key | Default | Meaning |
| --- | --- | --- |
| `shmimName` | `acc000` | Existing CPU UINT16 input, width2/height1, temporal depth1 or greater; a 2-D depth-one pair is also accepted |
| `profile` | `accelSim12` | Calibration label; the actual numerical keys below define decoding |
| `min_code`, `max_code` | 0,4095 | Encoding range within UINT16; a custom range can extend through65535 |
| `reject_rails` | true | Reject samples at either configured endpoint |
| `zero0`, `zero1` | 2048 | Per-channel zero offsets in counts |
| `volts_per_count0/1` | .0083799 | Positive volts/count |
| `sensitivity0/1` | 1.02 | Positive volts per m/s^2 |
| `polarity0/1` | +1 | Axis sign, exactly +1 or -1 |
| `capacity` | 4096 | Preallocated consumer ring, limited to2–65536 samples |
| `max_age_us` | 20000 | Maximum mapped acquisition age, including during silence |
| `max_interval_us` | 1000 | A larger acquisition interval breaks the contiguous segment even with consecutive publication counters |
| `clock_tolerance_us` | 1000 | Maximum clock-bracket uncertainty or realtime/monotonic offset change |

Decoding, independently on channels0/1, is
`acceleration=(double(raw)-zero)*polarity*volts_per_count/sensitivity`, in m/s^2.
The default profile matches the current simulator's 12-bit codes in a UINT16
container. For example2049/2047 decode to approximately +/−.008215588 m/s^2.
No unsigned subtraction or FLOAT reinterpretation occurs. There is no online
z-score normalization, displacement conversion or frequency-specific estimator.
A custom AD7689 profile requires that acquisition system's actual code range,
channel order, zero, sensitivity, polarity and timestamp contract; this work does
not assert those calibrations or implement a signed-code reinterpretation.

## Capture, retention and recovery

`adc_reader.cpp` adapts the existing-only attachment, reader-slot management and
metadata-bracketed copying pattern from `accelFFCtrlStream.hpp` on the simulator
branch. It has no dependency on that worktree or its harmonic estimator. Native
ImageStreamIO remains a production build dependency, as it already is for the app.
A dedicated ordinary worker polls at bounded intervals; model work never holds
its lock. This is not a qualified realtime scheduling policy.
Application startup/shutdown serialize the lifecycle calls; concurrent diagnostic
and retained-sample copies use the reader's own mutex.

The reader checks CPU location, type, geometry, temporal flags/arrays and original
allocation bounds before copying. Each bounded batch (at most256 frames) is
bracketed by write/counter/layout metadata checks. It recovers retained planes in
counter order, deduplicates repeats and records raw codes, the original realtime
acquisition/write timestamps, publication counter, backing device/inode, source
generation, segment, mapped monotonic times and local receipt time. Local clock
observations occur after the stable copy. Publication time and receipt are not
assumed to be acquisition time.

Attachment begins with the newest committed sample. A backing-file replacement,
disconnection or counter rollback starts a new source generation; ring contents
never span generations. Observed publication gaps, invalid codes/times, stale
data and excessive acquisition intervals clear the retained segment. Busy/raced
copies are retried on a later bounded poll without inventing samples. Counter
loss is counted when the subsequent stable batch establishes it. Clock reversal
or an excessive realtime/monotonic offset change latches a clock fault until a new
source session (reconnect/restart or app restart); new samples do not silently
establish a replacement clock mapping. Shutdown joins the worker, invalidates
retention and releases only its reader slot/mapping.

The consumer ring overwrites its oldest entry when full and counts these retention
evictions separately. `AdcReader::samples()` returns an owned oldest-first copy
for a future alignment adapter, outside acquisition. Producer depth1 cannot retain
an overwritten raw sample: increase producer retention only through a separately
reviewed producer configuration. No producer configuration is changed here.
`dropped` counts observable lost publications plus rejected frames. It cannot infer
how many unpublished simulator ticks were skipped; `timing_gaps` reports excessive
acquisition intervals separately. `duplicates` includes empty/repeated polls.
Measured rate uses accepted acquisition intervals within the contiguous segment,
not a requested FPS or the number of semaphore posts.

## Diagnostics

Shadow instances publish read-only `adc` numbers: enabled/valid, raw0/1,
acceleration0/1, rate_hz, age_us, samples, dropped, invalid, duplicates, races,
busy, restarts, overwrites, timing_gaps, retained, source counter/generation/segment
and image dimensions. `adc_source` carries health, stream/profile/type, backing
identity and exact integer timestamp/counter strings (`sample_counter` identifies
the observed pair; `counter` is the latest source-header count). Invalid raw pairs remain
visible with cleared acceleration and valid0; other invalid states may retain
last observed values, which must be ignored when valid0. Age−1 means unavailable.
These are ordinary application-cadence snapshots, not an 8-kHz logger.

**`adc.valid=1` is ADC health only.** Raw WFS callbacks still invalidate both
models. `ddspc_data.applied_valid`, `timing_valid` and `accel_valid` remain false;
no native model update or prediction becomes valid from this reader. Independent
training switches cannot bypass those gates.

## Safe local checks

The offline suite compiles the actual reader against the explicit
`tests/native_stubs` ImageStreamIO API double. Its threaded recovery fixtures use
only private ordinary files under the process-specific temporary directory;
there is no native shared memory, hardware, service or INDI connection.

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.offline BUILD_DIR=build/adc-stage1 check
make -C apps/loPredCtrlAcc -f tests/Makefile.offline BUILD_DIR=build/adc-stage1 check-shadow-only
make -C apps/loPredCtrlAcc -f tests/Makefile.offline BUILD_DIR=build/adc-stage1-sanitize \
  CXXFLAGS='-std=c++20 -O1 -g -Wall -Wextra -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer' check-adc
make -C apps/loPredCtrlAcc -f tests/Makefile.offline BUILD_DIR=build/adc-stage1-tsan \
  CXXFLAGS='-std=c++20 -O1 -g -Wall -Wextra -fno-fast-math -fsanitize=thread' check-adc
```

Native ImageStreamIO headers/libraries and a Linux runtime are unavailable on this
Mac. API-double tests do not prove the installed native ABI, stream retention or
host scheduling. Next qualify the guarded native build and a private, no-DM
simulator input on an isolated provisioned development host. Inspect actual type,
2x1xD geometry, timestamp progress, raw values, rate, age and gap/recovery behavior;
confirm output helpers remain absent. Only the operator performs any later
instrument-host observation.

Before live DDSPC training, implement verified WFS exposure/source correspondence,
actual applied-input ingestion and calibrated modal mapping, common-clock timing,
causal feature construction and silence expiry into `CausalInputBuilder`.
Preserve the current refusal to fabricate applied commands or certify raw ADC as
synchronized features. No training, prediction, actuation or producer change is
part of Stage1.
