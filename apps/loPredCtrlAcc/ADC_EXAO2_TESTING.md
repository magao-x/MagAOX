# Stage 1 ADC reader: operator-only exao2 test

These commands are for the operator to run manually. The agent has not executed
them, launched an application, opened a native stream or connected to INDI.
This tests ADC observation only. Both DDSPC models remain unready and DM output
remains inhibited. Stage2 WFS/applied-input alignment is still absent.

## Review and build qualification

The reviewed implementation was already committed as `fdad8ac8`. The follow-up
review fixes rollback detection when the consumed cursor trails a newer stable
publication. The local retest passed83 full offline C++ cases,9 guarded app cases,
8 focused ADC cases under each of ASan/UBSan and ThreadSanitizer,8 build-helper
unit tests and2 configuration tests. Transport tests use the explicit API double,
not native shared memory. The user handles staging/committing the review changes.

This Mac is Darwin/arm64; ImageStreamIO headers/libraries and a Linux instance
are unavailable. Compiling `adc_reader.cpp` without the stub include path fails
at `ImageStreamIO/ImageStreamIO.h`. Actual Linux compilation/linking, installed
ABI, semaphore behavior, producer concurrency, scheduling and live recovery are
outstanding. Complete the native build below before attempting attachment.

Prefer initial qualification on a provisioned non-instrument Linux development
host. The existing `tests/native_linux_build.py` is build-only and intentionally
refuses `exao*` hosts; do not bypass that guard. The manual build below is for the
operator if building on exao2 is necessary. Use existing dependencies: C++20
compiler/standard library, Eigen/mxlib, ImageStreamIO, xrif, libudev and the normal
MagAO-X numerical/logger dependencies described in EXAO2_TESTING.md. No SDK,
library installation, service restart or production binary replacement is part
of this procedure.

## 1. Prepare a fresh source worktree and strict guarded binary

After committing the reviewed changes and making that revision available in the
operator's local repository, run in Bash. Set `DDSPC_ORIGIN` to that repository;
it need not be the running simulator's checkout. Choose new build/run paths if
these already exist. Worktree creation leaves existing checkouts unchanged.

```sh
set -euo pipefail
hostname
export DDSPC_ORIGIN="$HOME/source/MagAOX-DDSPC"
export DDSPC_REV=parkertjohnson1/accel_ddspc
export DDSPC_SRC="$HOME/ddspc-adc-stage1-build"
export DDSPC_RUN_ROOT="$HOME/ddspc-adc-stage1-run"
export DDSPC_INSTANCE=loPredCtrlAcc_adc_stage1
export DDSPC_INDI_PORT=17625

git -C "$DDSPC_ORIGIN" rev-parse "$DDSPC_REV"
git -C "$DDSPC_ORIGIN" worktree add --detach "$DDSPC_SRC" "$DDSPC_REV"
mkdir "$DDSPC_RUN_ROOT"
mkdir -p "$DDSPC_RUN_ROOT/config" "$DDSPC_RUN_ROOT/logs" "$DDSPC_RUN_ROOT/sys" \
  "$DDSPC_RUN_ROOT/drivers/fifos" "$DDSPC_RUN_ROOT/calib" "$DDSPC_RUN_ROOT/secrets"

g++ --version > "$DDSPC_RUN_ROOT/compiler.txt"
git -C "$DDSPC_SRC" rev-parse HEAD > "$DDSPC_RUN_ROOT/revision.txt"
(
  make -C "$DDSPC_SRC/INDI/liblilxml" -j2 CXX=g++ CXXSTD=-std=c++20 EDT=false OPTIMIZE=-fno-fast-math all
  make -C "$DDSPC_SRC/INDI/libcommon" -j2 CXX=g++ CXXSTD=-std=c++20 EDT=false OPTIMIZE=-fno-fast-math all
  make -C "$DDSPC_SRC" -j2 CXX=g++ CXXSTD=-std=c++20 EDT=false NEED_CUDA=no OPTIMIZE=-fno-fast-math libs_all
  make -C "$DDSPC_SRC/apps/loPredCtrlAcc" -j1 CXX=g++ CXXSTD=-std=c++20 EDT=false NEED_CUDA=no \
    OPTIMIZE=-fno-fast-math DDSPC_SHADOW_BUILD=yes DDSPC_SHADOW_ONLY=yes all
) 2>&1 | tee "$DDSPC_RUN_ROOT/native-build.log"
export DDSPC_BIN="$DDSPC_SRC/apps/loPredCtrlAcc/loPredCtrlAcc"
test -x "$DDSPC_BIN"
sha256sum "$DDSPC_BIN" > "$DDSPC_RUN_ROOT/binary.sha256"
```

Stop on any build failure. Review the compile/link log: the app must have
`-DDDSPC_SHADOW_ONLY`; all app/numerical/PCH objects must use strict
`-fno-fast-math`, and no compile may include `tests/native_stubs`. The fresh
worktree prevents stale objects from silently retaining old flags. The app build
is serialized because of existing generated-version-header dependencies.
Do not run `make install`, copy to `/opt/MagAOX/bin`, use a setuid binary, run as
root or launch a service. The binary above is a private build.

## 2. Prepare the ADC-only private configuration

The distributed [example](examples/loPredCtrlAcc-adc-shadow.conf.example) has the
correct default simulator calibration but deliberately contains a WFS placeholder.
For Stage1, clear only that placeholder in the private copy. The native WFS
monitor waits without opening a source when its configured name is empty
(`dev/shmimMonitor.hpp`); the independent ADC worker does not require WFS data.

```sh
touch "$DDSPC_RUN_ROOT/config/magaox.conf"
sed 's/^shmimName = REPLACE_WITH_VERIFIED_MODAL_WFS_STREAM$/shmimName =/' \
  "$DDSPC_SRC/apps/loPredCtrlAcc/examples/loPredCtrlAcc-adc-shadow.conf.example" \
  > "$DDSPC_RUN_ROOT/config/$DDSPC_INSTANCE.conf"
cat "$DDSPC_RUN_ROOT/config/$DDSPC_INSTANCE.conf"
```

The relevant configuration must be:

```ini
[parameters]
shadow = true
qrd = true
is_integrating = false
own_shmim = false
accel_enabled = false

[shmimMonitor]
shmimName =
threadPrio = 0
cpuset =
getExistingFirst = false

[adc]
enabled = true
shmimName = acc000
profile = accelSim12
min_code = 0
max_code = 4095
reject_rails = true
zero0 = 2048
zero1 = 2048
volts_per_count0 = 0.0083799
volts_per_count1 = 0.0083799
sensitivity0 = 1.02
sensitivity1 = 1.02
polarity0 = 1
polarity1 = 1
capacity = 4096
max_age_us = 20000
max_interval_us = 1000
clock_tolerance_us = 1000
```

This excerpt supplements the full example's existing model dimensions and
numerical settings. Keep the full copied example. No `framegrabber` section or DM
output name is supplied. `own_shmim=false` alone would not disable Legacy writes;
the protections are shadow execution plus the compile-time guard. The separate
Legacy FLOAT accelerometer reader stays off.

Before attaching, confirm from the already running producer's configuration/logs
that this `acc000` belongs to accelSim, not the real AD7689 acquisition, and that
its ADC calibration matches these values. The reviewed simulator source uses
channels0/1 as X/Y, UINT16 2×1×D, clipped12-bit codes. Its defaults are zero2048,
.0083799 volts/count and sensitivity1.02, with nominal6kHz sampling. A running
instance can override calibration/rate; do not assume8kHz or use this profile
with the real acquisition system. Keep any simulator DM injection disabled.
Do not start, reconfigure or restart accelSim as part of this reader test.

Use the same ImageStreamIO namespace/environment as the producer. In particular,
if it uses `MILK_SHM_DIR`, export that exact existing directory in the test shell
before the first native ImageStreamIO call; do not create or redirect a producer
namespace. `MAGAOX_PATH` below isolates application files and does not select the
ImageStreamIO namespace. Confirm existing input access and a free reader slot.
Do not alter permissions or reset any semaphores to obtain access.

An optional operator-only dimension/arrival-rate check is:

```sh
shmimInfo -n acc000 -N 1000 -t 1
```

This claims another reader slot and consumes its notifications. It does not
verify ADC values, datatype or acquisition semantics, and its notification-based
rate can be affected by scheduling. Use the reader's source/raw diagnostics below
as well. Do not delete the input stream if an inspector fails.

## 3. Validate configuration and run only the reader in shadow mode

In the prepared shell:

```sh
export MAGAOX_PATH="$DDSPC_RUN_ROOT"
export MAGAOX_CONFIG_RPATH=config MAGAOX_LOG_RPATH=logs MAGAOX_SYS_RPATH=sys
export MAGAOX_CALIB_RPATH=calib MAGAOX_SECRETS_RPATH=secrets
"$DDSPC_BIN" -n "$DDSPC_INSTANCE" \
  -c "$DDSPC_RUN_ROOT/config/$DDSPC_INSTANCE.conf" --config.validate
"$DDSPC_BIN" -n "$DDSPC_INSTANCE" \
  -c "$DDSPC_RUN_ROOT/config/$DDSPC_INSTANCE.conf" \
  2>&1 | tee "$DDSPC_RUN_ROOT/reader-console.log"
```

Config validation exits before startup; inspect its messages as well as exit
status. Run the second command only after a clean native build/config review.
Startup must say **Shadow output inhibited** and that live models remain invalid.
Keep the foreground reader terminal available for Ctrl-C. The example's
`fpsSource=camwfs` may be unresolved on the private broker; `fps.current=0` is
expected and is unrelated to `adc.rate_hz`. Do not connect the broker to production
to resolve that subscription.

## 4. View raw values and INDI diagnostics

Use a separate foreground private broker that forwards only this instance's
private FIFOs. In a second Bash terminal, export the same `DDSPC_SRC`,
`DDSPC_RUN_ROOT`, `DDSPC_INSTANCE` and `DDSPC_INDI_PORT` values from section1.
Confirm port17625 is unused (or choose another nonproduction test port):

```sh
ss -ltn 'sport = :17625'
make -C "$DDSPC_SRC/INDI/xindidriver" -B CXX=g++ CXXSTD=-std=c++20 EDT=false \
  OPTIMIZE=-fno-fast-math FIFO_PATH="$DDSPC_RUN_ROOT/drivers/fifos" all
ln -s "$DDSPC_SRC/INDI/xindidriver/xindidriver" "$DDSPC_RUN_ROOT/$DDSPC_INSTANCE"
indiserver -p "$DDSPC_INDI_PORT" "$DDSPC_RUN_ROOT/$DDSPC_INSTANCE"
```

If the port is listed as listening, select a different port before launching the
private broker. Do not attach this bridge to the production broker or restart an
existing broker. The bridge is compiled with the private FIFO path and derives
the instance name from its symlink basename (`INDI/xindidriver/xindidriver.cpp`).

In a third operator terminal, with the same instance/port variables:

```sh
getINDI -h 127.0.0.1 -p "$DDSPC_INDI_PORT" "$DDSPC_INSTANCE.adc.*"
getINDI -h 127.0.0.1 -p "$DDSPC_INDI_PORT" "$DDSPC_INSTANCE.adc_source.*"
getINDI -h 127.0.0.1 -p "$DDSPC_INDI_PORT" "$DDSPC_INSTANCE.ddspc_status.*"
getINDI -h 127.0.0.1 -p "$DDSPC_INDI_PORT" "$DDSPC_INSTANCE.ddspc_data.*"
getINDI -h 127.0.0.1 -p "$DDSPC_INDI_PORT" "$DDSPC_INSTANCE.ddspc_standard.*"
getINDI -h 127.0.0.1 -p "$DDSPC_INDI_PORT" "$DDSPC_INSTANCE.ddspc_accel_assisted.*"
getINDI -h 127.0.0.1 -p "$DDSPC_INDI_PORT" -m -t 0 \
  "$DDSPC_INSTANCE.adc.*" "$DDSPC_INSTANCE.adc_source.*"
```

The last command monitors updates until Ctrl-C; optionally capture its output
with `tee` to a private log. These are ordinary application-cadence snapshots
(normally about1Hz), not an ADC-rate raw recorder. No `setINDI` command is needed.

| Observation | Required interpretation |
| --- | --- |
| `ddspc_status.execution / output` | `shadow / inhibited` throughout |
| `adc.enabled / valid` | 1/1 while fresh valid input is accepted |
| `adc_source.stream / datatype / profile` | `acc000 / UINT16 / accelSim12` |
| `adc.width / height / depth` | 2/1/actual producer depth |
| `adc.raw0 / raw1` | Unsigned ADC counts, never FLOAT reinterpretation; endpoint codes are rejected by this profile |
| `adc.acceleration0 / acceleration1` | `(raw-2048)*.0083799/1.02` m/s²; codes2049/2047 give ±.008215588 m/s² |
| `adc.samples`, source/sample counters and acquisition/write times | Advance; exact integer provenance is in `adc_source` |
| `adc.rate_hz` | Measured acquisition rate within the current contiguous segment; nominal6kHz only if the producer uses that rate and capture keeps up |
| `adc.age_us` | Fresh mapped acquisition age below20000µs; −1 means unavailable |
| `dropped / invalid / timing_gaps` | Explain every growth; repeated loss is a scheduling/retention qualification failure for future training |
| `duplicates / races / busy` | Poll/retry counts, not necessarily lost samples; duplicates include empty polls |
| `overwrites` | Expected bounded consumer-ring evictions after capacity is reached; distinct from source loss |
| `generation / restarts` | Stable for one source session; increase after reconnection/observed rollback |
| Both models' `updates / predictions / ready / prediction_valid` | Remain0; valid ADC data alone never trains either model |
| `ddspc_data.applied_valid / timing_valid / accel_valid` | Remain0 |

`adc.valid=0` makes the last displayed physical values unusable. Rejected raw
pairs remain visible with cleared acceleration. A depth-one producer cannot
retain overwritten samples; the4096-entry consumer ring cannot recover them.
If losses repeatedly break segments, rate may stay0 even while raw values arrive.
Do not change producer retention during this test; record that limitation for
separate review. This ordinary20µs-poll worker is not realtime-qualified.

## 5. Shutdown and acceptance record

Stop the monitor with Ctrl-C, then the foreground reader with Ctrl-C. Its shutdown
joins the ADC worker, invalidates retention and closes its mapping/reader slot.
Stop only the private broker you launched. Do not kill or restart a producer,
operational controller or production broker, and do not unlink any input image.
Leave the private logs/config/build artifacts available for review.

Record source revision, binary hash, native compile/link log, exact configuration,
producer identity/calibration/rate/retention, raw/physical values, original
timestamps/counters, ADC age/rate/loss diagnostics, resource usage and clean exit.
Do not deliberately disconnect or restart a live producer to test recovery.
Qualify those failure paths with a separately approved private Linux fixture;
offline API-double coverage is not evidence of live recovery.

Acceptance here means input observation with output inhibited. It does not
validate WFS synchronization, applied-DM provenance, live DDSPC training or
prediction quality. Those remain Stage2 work.
