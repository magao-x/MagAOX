# Manual exao2 handoff: shadow commissioning only

The normal build is now always guarded and uses continuous native acquisition.
Use [NATIVE_APPLICATION.md](NATIVE_APPLICATION.md) for installation and xctrl/INDI
registration. Legacy/default-build descriptions below are historical handoff notes,
not an instruction to enable an unguarded installed controller.

**Stage 1 update:** [ADC_INPUT.md](ADC_INPUT.md) describes the new opt-in,
read-only native UINT16 capture and its separate shadow configuration. It remains
unqualified against the native Linux/ImageStreamIO runtime. The original
attachment-only example and earlier procedure below remain unchanged. ADC capture
does not establish WFS/applied-input alignment or enable live model training.
Use [ADC_EXAO2_TESTING.md](ADC_EXAO2_TESTING.md) for the current ADC-only native
build/configuration and operator-run diagnostic commands.

2026-11-06. Prepared and tested on personal Mac `ASTR-LT-428.local`.
**No access to exao2, application launch, real DM command or deployment was performed.**

**Status: not ready for live accelerometer-assisted learning/prediction.** The
native application can attach to a modal WFS stream with output inhibited, but
its shadow WFS callback deliberately invalidates the models. Verified
applied-command ingestion and exposure/clock alignment are missing. Native ADC
decoding is now implemented separately in Stage 1. Configuration cannot supply
those remaining implementations.
The procedure below is for build qualification and WFS attachment/output-inhibition
commissioning. A working offline harness is not evidence that the native input
chain exists. Do not enable Legacy mode to work around an unready shadow model.

## 1. Required dependencies

- A provisioned, isolated Linux MagAO-X development stack, preferably qualified
  before any manual exao2 session. Native Linux compilation remains unverified
  here: this Mac has no available Linux/container environment.
- GCC14+ or a compatible Clang toolchain with C++20 and its standard library,
  Eigen, mxlib and its pkg-config metadata, ImageStreamIO, xrif, libudev and the
  normal MagAO-X numerical/logger dependencies. Normal dependencies include
  CFITSIO, FFTW, GSL, BLAS/LAPACK and FlatBuffers tooling as required by the local
  mxlib/flatlogs installation. Repository INDI/common/lilxml and libtelnet are
  built below. No camera SDK or CUDA is required for this app (`EDT=false`).
- Existing input-stream permissions and an available ImageStreamIO reader
  semaphore. Do not change ownership/permissions or restart a producer to obtain
  access during this handoff.
- For optional private INDI inspection, the existing `indiserver`, `getINDI` and
  repository `xindidriver`. Python offline tests need NumPy/SciPy/Matplotlib;
  build orchestration needs Python3.9+.

Use the exact reviewed working source snapshot. **`git archive HEAD` would omit
the current uncommitted handoff changes.** Record branch, HEAD, working changes,
source hashes, compiler version and complete build logs. No commit/push is part
of this task. Do not install dependencies or replace production libraries as a
side effect of this procedure.

## 2. Build commands

First run offline checks on a workstation/development host from the repo root:

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 check
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 check-shadow-only
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 \
  BUILD_DIR=build/offline-sanitize \
  CXXFLAGS='-std=c++20 -O1 -g -Wall -Wextra -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer' check
python3 -m unittest discover -s apps/loPredCtrlAcc/tests -p native_linux_build_test.py -v
python3 -m unittest discover -s apps/loPredCtrlAcc/tests -p handoff_config_test.py -v
```

For a **non-instrument Linux development host**, use a new output directory:

```sh
python3 apps/loPredCtrlAcc/tests/native_linux_build.py \
  --build-root /tmp/ddspc-native-handoff --cxx g++ --jobs 2
```

The helper intentionally refuses all `exao*` hosts and macOS. Do not bypass its
host guard. It freezes current nonignored source bytes, builds Legacy and strict
shadow profiles separately, and records commands/hashes in `manifest.json`.
The shadow profile now also passes `DDSPC_SHADOW_ONLY=yes`. Neither binary is
executed or installed. Review both link results; fake build-orchestration tests
are not proof of a native link. Ignored `local/` overrides and external SDKs are
not staged automatically.

If the operator must build locally on exao2, the operator must use a **fresh,
isolated source copy with the provisioned development dependencies**. The build
requirements, not executed by this agent, are:

```sh
# Set DDSPC_SRC to that fresh source copy, never a production service tree.
make -C "$DDSPC_SRC/INDI/liblilxml" -j2 CXX=g++ CXXSTD=-std=c++20 EDT=false OPTIMIZE=-fno-fast-math all
make -C "$DDSPC_SRC/INDI/libcommon" -j2 CXX=g++ CXXSTD=-std=c++20 EDT=false OPTIMIZE=-fno-fast-math all
make -C "$DDSPC_SRC" -j2 CXX=g++ CXXSTD=-std=c++20 EDT=false NEED_CUDA=no OPTIMIZE=-fno-fast-math libs_all
make -C "$DDSPC_SRC/apps/loPredCtrlAcc" -j1 CXX=g++ CXXSTD=-std=c++20 EDT=false NEED_CUDA=no \
  OPTIMIZE=-fno-fast-math DDSPC_SHADOW_BUILD=yes DDSPC_SHADOW_ONLY=yes all
```

Do not use `make install`, setuid, service scripts or `/opt/MagAOX/bin/loPredCtrl*`.
The app build is serialized to avoid the existing generated-version-header race.
Every object and PCH must be rebuilt strict; changing make flags does not
invalidate existing objects. Review the app compile for `-DDDSPC_SHADOW_ONLY`
and all numerical/PCH commands for final `-fno-fast-math`. The historical GNU
fast-math QR failure is unresolved; production flags are not changed here.

A shadow-only binary rejects missing/false `parameters.shadow` during config
loading and again at startup, before an input/output helper starts. Normal
builds retain Legacy as default. **Use only the guarded binary for this handoff.**

## 3. Configuration setup

The supplied [example](examples/loPredCtrlAcc-shadow.conf.example) records the
requested numerical values and explicit safety controls. It is intentionally
not a working live-training configuration.

| Setting | Handoff meaning |
| --- | --- |
| `shadow=true` | Startup-only output inhibition; cannot be changed through mode selection |
| `gain=.3`, `copy_gain=1` | Loaded Legacy values, unused by the shadow reference law |
| `legacy_copy_gain_compat=false` | Explicitly honor `copy_gain`; default true preserves historical Legacy loading from `gain` |
| `regularization=100` | **Literal lambda100** in native shadow, unlike Legacy's `100*maxCoeff(CᵀC)` |
| `gamma=1`, `covariance=1e-5` | No forgetting; information rho1e-5, P0=1e5I |
| `history=15`, `future=3`, `num_modes=2`, `qrd=true` | Independent standard/assisted causal QR models; no new operational control law |
| `is_integrating=false` | Explicitly disabled and forced off in shadow |
| `own_shmim=false` | Additional configuration precaution; actual protection is skipping every output helper |
| `accel_enabled=false` | Keep the legacy FLOAT reader off; it is incompatible with raw UINT16 ADC |
| `accel_channels=2`, `accel_history=5` | Intended feature dimensions only; does not attach or decode ADC0/1 |
| `fpsSource=camwfs` | Nominal INDI FPS subscription, **not** the WFS stream or an exposure-time contract |
| `shmimMonitor.shmimName` | Must be replaced with independently verified modal WFS stream name |

Choose a unique instance name `loPredCtrlAcc_shadow`, never `loPredCtrl` or the
name of a running controller. Prepare private files as the normal operator user:

```sh
export DDSPC_RUN_ROOT="$HOME/ddspc-shadow-manual"
mkdir "$DDSPC_RUN_ROOT"                 # Must be a new directory; choose another name if it exists.
mkdir -p "$DDSPC_RUN_ROOT/config" "$DDSPC_RUN_ROOT/logs" "$DDSPC_RUN_ROOT/sys" \
  "$DDSPC_RUN_ROOT/drivers/fifos" "$DDSPC_RUN_ROOT/calib" "$DDSPC_RUN_ROOT/secrets"
touch "$DDSPC_RUN_ROOT/config/magaox.conf"  # Empty private global config, not a copy of production config.
cp "$DDSPC_SRC/apps/loPredCtrlAcc/examples/loPredCtrlAcc-shadow.conf.example" \
  "$DDSPC_RUN_ROOT/config/loPredCtrlAcc_shadow.conf"
```

Edit only that private file to fill the modal WFS placeholder. Keep normal
priority0 and no cpuset; no real-time reservation is needed for this attachment
test. Keep the output and accelerometer controls as supplied. There is no
`framegrabber` section or DM output name. **`own_shmim=false` alone would not
disable Legacy writes**; never remove shadow mode or its compiled guard.

## 4. Identify and verify input streams

Operator-only, using existing records/approved read-only inspectors:

1. Identify the actual WFS **modal residual** stream from the existing pipeline
   configuration. A raw camera image is not accepted. Native shadow requires
   FLOAT, height1, width>=2, and reads the first two values. Verify their modal
   identities, ordering, units and sign; first two values do not by themselves
   prove tip/tilt. Record source PID/generation, frame counter, producer clock,
   timestamp meanings, exposure start/end and integration duration.
2. `acc000` is a candidate name, not a verified host setting. Verify its actual
   source, layout, UINT16 encoding, rate, channel definitions and timestamp clock.
   **Only ADC channels0/1 are authorized**. `accelSim` encodes simultaneous X/Y
   accelerations in those slots; that does not identify on-sky tip/tilt axes.
   Its defaults (zero2048, volts/count .0083799, sensitivity1.02,12-bit rail limits)
   are simulation parameters, **not instrument calibrations to copy**. Its source
   defaults to6kHz; prior offline runs explicitly selected8kHz. Measure the host
   rate rather than assume it. A future native decoder must consume actual ADC
   counts, verify channel mapping/calibration, detect rails/gaps/restarts, and
   construct features using acquisition and availability times.
3. Identify actual applied-command provenance after combination, limits and
   mapping. Requested output, calculated Legacy delta, accumulated request,
   combined surface and successful driver/SDK call are different observations.
   `dev::dm::processImage()` calls `commandDM()` and produces shape/delta/saturation
   diagnostics; these do not establish physical application time/readback.
   No verified applied-input stream or native reader has been identified here.
4. Establish WFS/ADC/command clocks and uncertainty independently. ImageStreamIO
   counters indicate publication ordering; `atime`/write timestamps and callback
   arrival do not automatically establish exposure or settling time. REALTIME
   cannot be used as MONOTONIC without a verified mapping. Measure command-to-WFS
   response from approved existing recordings; do not inject a test command.

Existing `shmimInfo` can measure dimensions/FPS on the verified names:

```sh
shmimInfo -n "$DDSPC_WFS_STREAM" -N 1000 -t 1
shmimInfo -n "$DDSPC_ADC_STREAM" -N 1000 -t 1
```

This utility does not establish datatype, timestamps, ADC calibration or applied
command semantics. Use the host's existing inspector/recorder for those fields.
Both it and `shmimMonitor` claim a reader semaphore and update reader bookkeeping;
they are not zero-impact subscriptions. Confirm a free reader slot and that the
operational reader retains its own slot. Native shadow never writes WFS/DM image
payloads. Do not reset semaphores, delete streams or restart sources if attachment
fails. No device-facing command is needed for this audit.

## 5. Shadow-mode startup commands

**Operator-only after native build review and stream verification.** The agent
has not run these commands. Use the absolute guarded binary, private config and
private log/PID/FIFO tree:

```sh
export MAGAOX_PATH="$DDSPC_RUN_ROOT"
export MAGAOX_CONFIG_RPATH=config MAGAOX_LOG_RPATH=logs MAGAOX_SYS_RPATH=sys
export MAGAOX_CALIB_RPATH=calib MAGAOX_SECRETS_RPATH=secrets
export DDSPC_BIN="$DDSPC_SRC/apps/loPredCtrlAcc/loPredCtrlAcc"
"$DDSPC_BIN" -n loPredCtrlAcc_shadow \
  -c "$DDSPC_RUN_ROOT/config/loPredCtrlAcc_shadow.conf" --config.validate
"$DDSPC_BIN" -n loPredCtrlAcc_shadow \
  -c "$DDSPC_RUN_ROOT/config/loPredCtrlAcc_shadow.conf"
```

Config validation exits before the application loop; review messages as well as
exit status. Startup logs must explicitly state **output inhibited**, native
WFS observation only and missing ADC/applied-command/timing adapters. Do not
launch through `xctrl`, systemd or a production controller configuration. The
application user must own the private log directory and have input access;
do not install setuid or run as root to work around access errors.

## 6. Inspect predictions and timing

Use an existing **separate, operator-approved INDI test broker**, connected only
to this instance's private FIFOs. Do not change or restart the production broker.
If none exists, an optional foreground private broker can reuse existing tools
on Linux, with a port independently confirmed unused:

```sh
# In the isolated source copy; compile only a private FIFO bridge, never install it.
make -C "$DDSPC_SRC/INDI/xindidriver" CXX=g++ CXXSTD=-std=c++20 EDT=false \
  OPTIMIZE=-fno-fast-math FIFO_PATH="$DDSPC_RUN_ROOT/drivers/fifos" all
ln -s "$DDSPC_SRC/INDI/xindidriver/xindidriver" "$DDSPC_RUN_ROOT/loPredCtrlAcc_shadow"
# Set DDSPC_INDI_PORT to a confirmed free test port; do not use the production port.
indiserver -p "$DDSPC_INDI_PORT" "$DDSPC_RUN_ROOT/loPredCtrlAcc_shadow"
```

In another operator terminal, read only this instance:

```sh
getINDI -h 127.0.0.1 -p "$DDSPC_INDI_PORT" 'loPredCtrlAcc_shadow.ddspc_status.*'
getINDI -h 127.0.0.1 -p "$DDSPC_INDI_PORT" 'loPredCtrlAcc_shadow.ddspc_data.*'
getINDI -h 127.0.0.1 -p "$DDSPC_INDI_PORT" 'loPredCtrlAcc_shadow.ddspc_standard.*'
getINDI -h 127.0.0.1 -p "$DDSPC_INDI_PORT" 'loPredCtrlAcc_shadow.ddspc_accel_assisted.*'
```

These are manual instructions, not agent instrument actions. If the private
broker does not forward `camwfs.fps`, `fps.current` may remain0; no production
snoop/routing change is required for the WFS attachment test. No per-frame
prediction vector, timestamp trace, applied-input logger or snapshot-age INDI
property is implemented in the native app. The offline asynchronous benchmark
has richer timing diagnostics; it is not connected to this application.

Leave Legacy `predict`, `learn`, `learn_std` and `integrate` **off**. Their
callbacks reject enabling requests in shadow mode. Shadow adaptation uses
`ddspc_train.standard` and `.accel_assisted`, independently; selecting
`ddspc_mode` does not enable prediction, adaptation or output. These train
switches may request learning, but **no native valid training record exists**:
enabling them will not establish timing or increase update counts. Do not claim
functional live training from a switch showing On. No activation commands are
provided for blocked live learning.

## 7. Expected diagnostics and software verification

| Property | Expected in the current native WFS attachment test |
| --- | --- |
| `ddspc_shadow.toggle` | On, read-only |
| `ddspc_status.execution / output` | `shadow / inhibited` |
| `ddspc_status.selected` | `standard`; assisted selection rejected while unready |
| `ddspc_status.input_reason / input_provenance` | `unverified_input_contract / unverified` |
| `ddspc_status.*_reason` | `awaiting_verified_record` initially, `invalid_or_missing_data` after raw WFS input |
| `ddspc_data.wfs_valid` | Finite/shape check for last callback only, not timestamp/freshness proof |
| `ddspc_data.applied_valid / timing_valid / accel_valid` |0 /0 /0 |
| Both model `ready / prediction_valid / training / updates / predictions` |0 /0 /0 /0 /0 |
| Both model `training_enabled` |0 initially; separate from actual updates |
| `ddspc_standard.features / ddspc_accel_assisted.features` |64 /74 with the supplied profile |
| Legacy `predict / learn / learn_std / integrate` | Off |
| Output `fg_*` properties/thread, command image | No output frameGrabber startup/property registration/image creation |

An unscored RMSE of0 is **not** a perfect prediction. WFS payload validity may
remain the last observed value during silence; without a native timing adapter
it is not a heartbeat. Both models remain invalid, so no stale live prediction
is possible. An output-related thread/stream, a valid native prediction or any
learned update with this attachment-only build is unexpected: stop this instance
and review the binary/configuration instead of bypassing the gate.

Current local evidence: strict Clang/GCC numerical, causal-input, historical
exact Legacy and application suites; ASan/UBSan; separate compiled shadow-only
lifecycle tests; build-orchestration and example-config tests. The profile test
uses **explicitly synthetic** two-feature records, trains independent64/74-column
QR models, verifies finite proposals and the reference-law norm bound, and
verifies no output helper is used. Native raw UINT16 input is rejected/ignored,
not decoded or certified. Numerical readiness is not a physical amplitude limit;
there is no new global native proposal clip, and no actuation is authorized.
Final counts/log hashes are in [handoff validation](tests/results/exao2_handoff.json).

## 8. Shutdown procedure

Press Ctrl-C in the **private application terminal** and wait for normal exit.
SIGINT/SIGTERM invokes normal shutdown, joins the WFS monitor and releases its
reader semaphore. No frameGrabber or DM cleanup path runs in shadow mode.
Then stop only the optional private broker in its own terminal. Preserve private
logs, source/build hashes and the operator's stream observations. Never use a
broad process-name kill, remove an input shmim or stop the existing controller.

## 9. Known limitations and operational discrepancies

- **ADC ingestion:** the native shadow monitor is not started; its allocator
  refuses attachment. The old optional Legacy reader maps FLOAT, has no ADC
  inverse calibration/counter/timestamp pairing, and can reuse a cached sample
  without a source-age check. Keep it disabled for `acc000`. Setting
  `accel_channels=2` does not fix decoding or prove ADC0/1 mapping.
- **Timing/applied inputs:** raw WFS callbacks have no exposure metadata or
  verified applied-command association and reset/invalidate any prior input
  session. No INDI/config option authenticates `InputConfig` or creates
  `AppliedInputRecord` certificates. The existing causal builder requires common
  clock epochs, source generations, calibrated modal layout and complete held
  input coverage; driver acceptance is insufficient. It rejects gaps, stale or
  ambiguous records and unsupported latency. No WFS/applied-input adapter/timer
  is armed natively; the optional Stage 1 ADC worker expires only ADC evidence.
- **Runtime state:** repository defaults are prediction/learning/integration off;
  deployed switch states are unknown. Legacy optionally accumulates calculated
  deltas, whereas causal training requires actual historical applied inputs.
  Exploration stages can override configured regularization. Do not infer these
  states from the provided numeric excerpt.
- **Control laws:** Legacy has historical indexing and calculated-command
  histories, scaled inverse regularization and added current-WFS feedback.
  Shadow uses corrected causal histories, literal positive lambda and the
  reference solve, without that added feedback or output accumulation. Those
  equations are not interchangeable. See the
  [operational audit](tests/results/operational_baseline.md).
- **copy_gain:** the historical duplicate read of `parameters.gain` is preserved
  by default for existing Legacy configurations. Set
  `legacy_copy_gain_compat=false` to honor an explicit `copy_gain`; if absent,
  the old gain fallback remains. The example opts out explicitly, but shadow
  has no bypass output. No original `apps/loPredCtrl` code is changed.
- **Calibration/delay:** actual signed WFS-to-DM projection, absolute/incremental
  semantics, included DM channels, exposure timing and physical command response
  are unverified. F3 targets .5/1/1.5ms only at an established2kHz cadence; the
  simulator's two-frame delay is not an instrument measurement. Do not copy its
  positive unit plant, ADC calibration or ideal pointing into a live controller.
- **Execution:** native shadow processing is still synchronous under its model
  mutex. The asynchronous worker/snapshot prototype remains offline. No native
  Linux link, sustained host timing, scheduling or noninterference measurement
  has been completed here. Passing macOS tests does not qualify a2kHz instrument
  loop, even when outputs are inhibited.

## 10. Criteria for proceeding

Before **prediction-enabled shadow testing**: reviewed native strict guarded
build/link; verified WFS/ADC0/1 layouts and calibrated decoding; a native stable
snapshot adapter retaining counters, generations, acquisition/availability times
and dropped-sample evidence; independently established exposure/clock/latency and
signed modal/applied-command contracts; actual source adapters into the existing
causal builder, including silence expiry; and tests on operator-provided replay
data. Live readiness must follow those records, not manually asserted booleans.

Before **any later actuation consideration**: successful observation-only
validation of both independent models, finite and physically appropriate proposal
limits, measured prediction/command timing, demonstrated stale/drop/reset
rejection, acceptable load/reader isolation and an explicit reviewed output
ownership/calibration/sign/limit policy. A separate user-approved phase is
required; this handoff contains **no DM correction route or enable procedure**.

The immediate useful operator result is a native build manifest plus verified
stream/timing/runtime-state observations. If those cannot be obtained safely,
stop at the build/attachment gate and return the missing information. Do not
start an accelerometer-assisted correction experiment with the current native
input paths.
