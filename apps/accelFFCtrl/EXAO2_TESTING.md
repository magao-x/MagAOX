# Initial exao2 shadow test — human operator only

This is a **transport, prediction and diagnostic test**, not a vibration-correction
qualification. The native app has **no DM writer, DM attachment or actuator
enable**. A configuration cannot activate a missing writer. The separate
`accelSim` DM injector and existing AO applications are outside this procedure.
No production configuration, producer rate, buffer depth or AO state is changed.

## 1. Prerequisites

- Use the reviewed `parkertjohnson1/accelSim` checkout, including the new shadow
  header/tests/configuration. Preserve local changes; do not transfer macOS build
  products. These commands are for you, never an agent on an instrument host.
- Use the site's working MagAO-X Linux compiler (GCC 14/C++20), mxlib, Eigen,
  libMagAOX, flatlogs/FlatBuffers, INDI and matching ImageStreamIO headers/library.
  Use existing site `local/common.mk` and environment; do not replace dependencies.
- Confirm the one existing `acc000` producer, CPU-resident UINT16 geometry
  `(2,1)` or `(2,1,1)`, depth **one**, valid realtime acquisition/write stamps,
  actual rate, ADC calibration and available reader semaphore. This app reads
  indices **0=X, 1=Y**, positive polarity, codes 0–4095. Historical eight-channel
  telescope `acc000` is incompatible: **stop if that is the current geometry**.
- Identify an existing WFS **timing** stream and confirm its original metadata
  semantics, clock and cadence. `camwfs_sync` is a repository example, not a
  verified current stream name. No live WFS tip/tilt modal source is configured
  by this test. Confirm timestamps use the same host CLOCK_REALTIME contract;
  ImageStreamIO defaults can differ. Exposure-center offset remains unmeasured.
- Obtain permission/access to the usual framework PID, log and INDI FIFO
  directories. A unique test app identity must have no existing process/driver.
  Existing producer and AO services remain running under their current owners.

## 2. Build and private verification

Run from your verified source checkout root. Use the actual GCC 14 executable
in place of `g++` if needed. Stop on any failure and retain full output.

```bash
set -euo pipefail
accff_cxx=g++
accff_testdir=$(mktemp -d /tmp/accelFFCtrl-shadow-test-XXXXXX)
"$accff_cxx" --version
pkg-config --modversion eigen3 ImageStreamIO
make -C apps/accelFFCtrl CXX="$accff_cxx" -W accelFFCtrl.hpp
make -C tests -f Makefile.one CXX="$accff_cxx" \
  t=../apps/accelFFCtrl/tests/accelFFCtrl_test.cpp \
  -W ../apps/accelFFCtrl/tests/accelFFCtrl_test.cpp
apps/accelFFCtrl/tests/accelFFCtrl_test '[accelFFCtrl][app]'
make -C apps/accelFFCtrl/tests CXX="$accff_cxx" \
  BUILD_DIR="$accff_testdir/bin" test benchmark native-test native-rate-test
ldd apps/accelFFCtrl/accelFFCtrl
ldd "$accff_testdir/bin/accelFFCtrlStream_test"
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  make -C apps/accelFFCtrl/tests CXX="$accff_cxx" \
  BUILD_DIR="$accff_testdir/asan" test native-test native-rate-test \
  CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'
```

No installation or simulator build/restart is needed. Native targets create only
private synthetic streams in fresh `/tmp/accelFFCtrl-native-*` directories in
child processes; do not run their executable directly in the production namespace.
They cover depth-one/ring overruns, busy writers, invalid times, restart,
default/optional policies and paced 4/8-kHz producers. The new metadata test also
checks float images and unchanged WFS semaphore ownership/counts. Rate tests
are scheduling observations, not a guarantee of lossless operation. Use verified
site `ISIO_CFLAGS`/`ISIO_LIBS` if pkg-config is unavailable; do not mix library ABIs.
`validate_linux.sh` intentionally refuses instrument hostnames; the manual
commands above do not change that restriction. Check `ldd` for missing libraries.

## 3. Private configuration checks

```bash
cp apps/accelFFCtrl/config/exao2-shadow-example.conf "$accff_testdir/test.conf"
```

Edit **only this private copy**. Verify `input.shmimName`, actual `sampleRate`,
source frequency and all ADC/model settings against the current producer. The
example's ADC zero 2048, volts/count 0.0083799, sensitivity 1.02 and lever 10 m
are existing **simulator values**, not telescope calibration. Native sensor
rotation is degrees; prediction horizon is 1 ms beyond the query, with sample
age included once. WFS is sampled for metadata at the 5-Hz reporting cadence;
it has no assumed pixel/mode calibration or configured acquisition-rate model.

Set `shadow.wfsShmimName` to the verified timing stream; set `shadow.logFile`
to a **new absolute** path such as the value of `$accff_testdir/shadow.csv`
(expand it yourself; configuration strings do not expand shell variables).
Then set `shadow.enabled=true`. Keep **`shadow.gain=0`**,
**`input.allowSingleFrameGap=false`**, producer depth one, `threadPrio=0` and
empty cpuset for this first unprivileged test. Priority/CPU tuning is a later
owner decision. Keep freshness, interval and horizon checks; stop to investigate
if the verified WFS cadence is incompatible with `maxWfsAge_us=2000`.

```bash
apps/accelFFCtrl/accelFFCtrl -n accelFFCtrl-shadow-test \
  -c "$accff_testdir/test.conf" --config.validate
```

Require the explicit **“Configuration is valid.”** message and no errors/unknown
keys, not merely exit status. This exits before startup/stream access and does
not verify stream presence or calibration. Enabled mode rejects absent names,
DM-prefixed names, ADC-as-WFS, relative/non-CSV paths and gain outside [0,1].
File existence/permissions are checked at startup. Existing files/symlinks are
rejected; use a new filename for every run. Do not use `make install` or copy
anything over `/opt/MagAOX/config`.

## 4. Shadow startup

In your dedicated terminal, after all checks pass:

```bash
apps/accelFFCtrl/accelFFCtrl -n accelFFCtrl-shadow-test \
  -c "$accff_testdir/test.conf"
```

The binary is run from the checkout, not a possibly older installed copy. It
opens only existing configured input streams, and creates its private CSV and
normal framework logs/PID/FIFOs. ADC semaphore consumer bookkeeping is expected;
the WFS attachment neither claims nor flushes a reader semaphore. No WFS pixels,
AO settings or DM streams are written. Do not start another ADC producer.

## 5. Telemetry verification

Use a dedicated INDI driver for this **unique test identity**, never a second
driver on an already managed FIFO. If `/opt/MagAOX/bin/xindidriver` is the
verified site installation and port **7625 is unused**, in a second terminal:

```bash
ln -s /opt/MagAOX/bin/xindidriver "$accff_testdir/accelFFCtrl-shadow-test"
indiserver -p 7625 "$accff_testdir/accelFFCtrl-shadow-test"
```

Carry the actual private directory value into each terminal. In a third terminal,
you may inspect only this app's properties:

```bash
getINDI -h localhost -p 7625 'accelFFCtrl-shadow-test.output.*'
getINDI -h localhost -p 7625 'accelFFCtrl-shadow-test.shadow.*'
getINDI -h localhost -p 7625 'accelFFCtrl-shadow-test.input.*'
getINDI -h localhost -p 7625 'accelFFCtrl-shadow-test.diagnostics.*'
```

Require `output.mode=shadow file only`, `output.dm_output=unavailable`,
`shadow.dm_output_enabled=0`, `shadow.gain=0`, and proposal values zero.
Record readiness, exact counters/generation, acquisition FPS, sample/producer
age, effective horizon, drops, resets, WFS validity/age and service time. WFS
readiness is separate from ADC readiness. Existing INDI numeric estimator
controls still work; avoid edits during the initial steady observation.

## 6. Expected output

Warmup is about 300 ms with the example memory and consecutive valid samples.
One lost ADC frame resets warmup with the default policy. Depth one cannot
recover overwritten frames, regardless of `maxBatch`; semaphore posts are
notifications. Report actual loss/availability rather than assuming previous
4-kHz success predicts the current run.

The private mode-0600 CSV flushes at `telemetryFps`, normally 5 Hz. It contains
exact integer ADC/WFS identities and timestamps, applied parameters/revision,
model predictions and opposing `unit_gain_*_model_rad` diagnostics. Gain-scaled
proposals stay zero in this test. Every row has `dm_output_enabled=0`. Invalid
or expired predictions produce a new zero proposal, never the last value. A
snapshot whose future target already elapsed is inhibited even if its fit was
ready. Missing/stale WFS inhibits proposals without resetting the ADC estimator.
The CSV is **not** high-rate science telemetry, an exposure-aligned WFS log or
an actuator transport. Validity refers to its recorded clock sample, not the
later time at which a file reader receives the row.

## 7. Shutdown

Ctrl-C **this observer**; allow its worker to join. Confirm the final CSV row
has `shutdown=1`, validity false and proposed values zero. Stop only the
dedicated test indiserver with Ctrl-C. Keep logs/config/source revision/test
output. Never delete, zero or restart operational ADC/WFS/DM streams. Abrupt
termination may omit the final file row; no actuator cleanup is needed because
this app has no actuator path.

## 8. Passing the initial test

- Linux application/framework/native tests pass with matching headers/library.
- About ten minutes of observation yields fresh, advancing ADC/WFS metadata and
  interpretable rate, age, loss and readiness reports, with no worker/file faults.
  Record any losses/resets; repeated warmup is a transport limitation to resolve,
  not a reason to enable gap tolerance or change producer settings during this run.
- No actuation-capable component is added; RO output is unavailable, all
  gain-zero proposals and DM enable fields remain zero, and no AO setting changes.
- Missing/invalid inputs inhibit rather than retaining proposals. Test those
  faults in private native fixtures, **not by interrupting instrument producers**.
- Return compiler/dependency versions, full test output, private config, source
  revision/diff, CSV and application logs for review before any further phase.

## 9. Known limitations and blockers

Local verification is macOS numerical/file-only. Full app build currently stops
in framework `timegm`/missing mxlib `floatUtils.hpp`; Linux application, framework
and native-stream execution remain **unverified here**. Perform section 2 first.
Current stream geometry, WFS name, clock/exposure semantics, physical ADC/lever/
rotation calibration and site runtime permissions require operator confirmation.

Native output is one-carrier **uncalibrated model pointing**, not real-telemetry
phase-calibrated WFS or DM coefficients. There is no physical modal mapping,
optical correction sign, DM delay/bandwidth calibration or hardware amplitude
limit in this path. Gain [0,1] is only a diagnostic scale; integrity time bounds
are not hardware protection. Physical pointing labels require valid sensor
calibration. The latest [closed-loop study](offline/CLOSED_LOOP_VALIDATION.md)
failed consistent two-axis suppression/neighboring-band qualification. It
supports a **shadow transport test only**, not real correction. Do not connect
a DM writer until separately authorized and physically qualified.
