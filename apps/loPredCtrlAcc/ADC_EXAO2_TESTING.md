# Minimum ADC-only shadow test on exao2

This is an **operator-run** observation of the existing repaired accelSim's
`acc000`. No WFS source, applied-command adapter or DDSPC training is required.
The agent ran only local offline tests, not these native commands. No application,
producer, service, INDI connection, installation or exao2 access was performed.

Use a **fresh private build with `DDSPC_SHADOW_ONLY=yes`**, a unique instance name,
a private `MAGAOX_PATH`, and a separate INDI broker. These isolate the binary,
configuration, PID files, logs, FIFOs and INDI identity from the operational
controller. Input payloads/counters/timestamps are never written. ImageStreamIO's
normal reader PID/notification-slot bookkeeping is the only input-side shared
write; the reader does not unlink or create any image.

## Build status and limits

The app Makefile now defaults to C++20 and exports that standard to the nested
libMagAOX/PCH build. GCC 14 otherwise defaults to C++17, which cannot compile the
ADC record's defaulted equality or the shared timestamp-window `std::span` API.
All existing numerical and transport implementations are retained. The example
now directly selects ADC-only shadow mode, with an empty WFS name and
`parameters.shadow_pipeline=false`.

Local validation uses Apple Clang, installed Eigen and the actual application
header/ADC reader with explicit framework/ImageStreamIO doubles. This personal
Mac has no available GCC 14/Linux or native ImageStreamIO environment. **The
GCC 14 / Eigen 3.4 native compile/link has not been demonstrated here.** Build
errors, installed ABI compatibility and live reader behavior must be checked
with the operator-run build below. No dependency upgrades are requested.

## 1. Prepare a private source/build and run directory

Run these commands manually in Bash on exao2, as your normal operator account.
Use the reviewed `parkertjohnson1/accel_ddspc` checkout containing these changes.
The working-tree patch below includes uncommitted tracked application changes;
no agent commit or push is required. Stop if there are untracked application
sources: make those reviewed files available in the private source copy first.
No ignored objects/PCHs or production configuration are copied.

```sh
set -euo pipefail
hostname
export DDSPC_ORIGIN="$HOME/source/MagAOX-DDSPC"
test "$(git -C "$DDSPC_ORIGIN" branch --show-current)" = parkertjohnson1/accel_ddspc
test -z "$(git -C "$DDSPC_ORIGIN" ls-files --others --exclude-standard -- apps/loPredCtrlAcc)"
export DDSPC_RUN_ROOT="$(mktemp -d "$HOME/ddspc-adc-mvp.XXXXXX")"
export DDSPC_SRC="$DDSPC_RUN_ROOT/source"
export DDSPC_INSTANCE="loPredCtrlAcc_adc_$(id -un)_$(date +%s)"
export DDSPC_INDI_PORT=17625
export DDSPC_CXX=g++                     # Use g++-14 here if that is its installed name.

git -C "$DDSPC_ORIGIN" rev-parse HEAD > "$DDSPC_RUN_ROOT/revision.txt"
git -C "$DDSPC_ORIGIN" diff --binary HEAD -- apps/loPredCtrlAcc > "$DDSPC_RUN_ROOT/working.patch"
git -C "$DDSPC_ORIGIN" worktree add --detach "$DDSPC_SRC" HEAD
if test -s "$DDSPC_RUN_ROOT/working.patch"; then
  git -C "$DDSPC_SRC" apply "$DDSPC_RUN_ROOT/working.patch"
fi
mkdir -p "$DDSPC_RUN_ROOT/config" "$DDSPC_RUN_ROOT/logs" "$DDSPC_RUN_ROOT/sys" \
  "$DDSPC_RUN_ROOT/drivers/fifos" "$DDSPC_RUN_ROOT/calib" "$DDSPC_RUN_ROOT/secrets"

"$DDSPC_CXX" --version | tee "$DDSPC_RUN_ROOT/compiler.txt"
test "$("$DDSPC_CXX" -dumpversion | cut -d. -f1)" = 14
pkg-config --modversion eigen3 mxlib | tee "$DDSPC_RUN_ROOT/dependencies.txt"
# For the requested target, eigen3 must report 3.4.x. Stop on missing/incompatible dependencies.
case "$(pkg-config --modversion eigen3)" in 3.4.*) ;; *) exit 1 ;; esac
```

This creates only a detached private worktree; it does not switch or clean the
operational checkout. Keep the recorded patch and revision with the test logs.
The separate `tests/native_linux_build.py` remains a development-host build
helper and intentionally refuses `exao*`; these manual commands do not invoke it.

## 2. Build only private binaries with the installed native dependencies

```sh
(
  set -e
  make -C "$DDSPC_SRC/INDI/liblilxml" -j2 CXX="$DDSPC_CXX" CXXSTD=-std=c++20 EDT=false OPTIMIZE=-fno-fast-math all
  make -C "$DDSPC_SRC/INDI/libcommon" -j2 CXX="$DDSPC_CXX" CXXSTD=-std=c++20 EDT=false OPTIMIZE=-fno-fast-math all
  make -C "$DDSPC_SRC" -j2 CXX="$DDSPC_CXX" CXXSTD=-std=c++20 EDT=false NEED_CUDA=no OPTIMIZE=-fno-fast-math libs_all
  make -C "$DDSPC_SRC/apps/loPredCtrlAcc" -j1 CXX="$DDSPC_CXX" CXXSTD=-std=c++20 EDT=false NEED_CUDA=no \
    OPTIMIZE=-fno-fast-math DDSPC_SHADOW_BUILD=yes DDSPC_SHADOW_ONLY=yes all
  make -C "$DDSPC_SRC/INDI/xindidriver" CXX="$DDSPC_CXX" CXXSTD=-std=c++20 EDT=false \
    OPTIMIZE=-fno-fast-math FIFO_PATH="$DDSPC_RUN_ROOT/drivers/fifos" all
) 2>&1 | tee "$DDSPC_RUN_ROOT/native-build.log"
export DDSPC_BIN="$DDSPC_SRC/apps/loPredCtrlAcc/loPredCtrlAcc"
test -x "$DDSPC_BIN"
test ! -u "$DDSPC_BIN"
grep -F -- '-DDDSPC_SHADOW_ONLY' "$DDSPC_RUN_ROOT/native-build.log"
sha256sum "$DDSPC_BIN" > "$DDSPC_RUN_ROOT/binary.sha256"
ln -s "$DDSPC_SRC/INDI/xindidriver/xindidriver" "$DDSPC_RUN_ROOT/$DDSPC_INSTANCE"
```

Stop on any compilation/link failure. The app compile must include
`-DDDSPC_SHADOW_ONLY`, all numerical/PCH builds must use strict floating point,
and none may include `tests/native_stubs`. The fresh worktree avoids old objects
with different flags/record layouts. The app build is serialized for the existing
generated-version-header dependency. Both binaries above remain private and
non-setuid; no `make install`, sudo, SDK/library installation or service command
is used. Do not substitute `/opt/MagAOX/bin/loPredCtrlAcc`.

## 3. Copy the ADC-only configuration and isolate application paths

```sh
cp "$DDSPC_SRC/apps/loPredCtrlAcc/examples/loPredCtrlAcc-adc-shadow.conf.example" \
  "$DDSPC_RUN_ROOT/config/$DDSPC_INSTANCE.conf"
touch "$DDSPC_RUN_ROOT/config/magaox.conf"
export MAGAOX_PATH="$DDSPC_RUN_ROOT"
export MAGAOX_CONFIG_RPATH=config MAGAOX_LOG_RPATH=logs MAGAOX_SYS_RPATH=sys
export MAGAOX_CALIB_RPATH=calib MAGAOX_SECRETS_RPATH=secrets
cat "$DDSPC_RUN_ROOT/config/$DDSPC_INSTANCE.conf"

declare -p DDSPC_RUN_ROOT DDSPC_SRC DDSPC_INSTANCE DDSPC_INDI_PORT DDSPC_BIN \
  MAGAOX_PATH MAGAOX_CONFIG_RPATH MAGAOX_LOG_RPATH MAGAOX_SYS_RPATH \
  MAGAOX_CALIB_RPATH MAGAOX_SECRETS_RPATH > "$DDSPC_RUN_ROOT/session.bash"
printf 'Other terminals: source %q\n' "$DDSPC_RUN_ROOT/session.bash"
```

The complete copied example already includes valid model dimensions and:

```ini
[parameters]
shadow = true
shadow_pipeline = false
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

Use the **full copied example**, not only this excerpt. No framegrabber/DM output
section is present. Empty WFS name leaves the existing monitor waiting before
attachment (`libMagAOX/app/dev/shmimMonitor.hpp`); the ADC worker runs independently.
Training remains disabled and the live models remain unqualified. Keep the new
replay integration off. `own_shmim=false` alone is not output inhibition: shadow
mode and the compile-time guard provide that protection.

Confirm the already running `acc000` belongs to repaired accelSim and uses the
calibration above. The real AD7689 system needs a different verified profile.
The current simulator's defaults are 6 kHz, but its configured rate may differ;
compare `adc.rate_hz` with the actual running producer's rate, not the offline
8-kHz simulation setting. Keep simulator DM injection disabled; this reader test
must not start/reconfigure the producer or change any operational control.

Use the producer's existing ImageStreamIO namespace. If it exports `MILK_SHM_DIR`,
set that exact value in the reader shell before launch. Otherwise leave the
normal namespace unchanged. `MAGAOX_PATH` isolates application files and does not
select the input-stream directory. Do not change stream permissions, reset
semaphores, or delete/recreate images to obtain access.

## 4. Validate and launch only the private reader

In the first terminal:

```sh
"$DDSPC_BIN" -n "$DDSPC_INSTANCE" \
  -c "$DDSPC_RUN_ROOT/config/$DDSPC_INSTANCE.conf" --config.validate
"$DDSPC_BIN" -n "$DDSPC_INSTANCE" \
  -c "$DDSPC_RUN_ROOT/config/$DDSPC_INSTANCE.conf" \
  2>&1 | tee "$DDSPC_RUN_ROOT/reader-console.log"
```

Review validation messages before launch; validation exits before input startup.
Startup must say **Shadow output inhibited**. Keep this foreground terminal for
Ctrl-C. The example retains an unresolved `fpsSource=camwfs` subscription on the
private broker: `fps.current=0` is expected and unrelated to `adc.rate_hz`. Do not
connect this broker to production to resolve that subscription.

## 5. Inspect only private INDI diagnostics

In a second terminal, run the exact `source .../session.bash` line printed in
step 3, then:

```sh
# Select another nonproduction port first if this check fails.
test -z "$(ss -H -ltn "sport = :$DDSPC_INDI_PORT")"
indiserver -p "$DDSPC_INDI_PORT" "$DDSPC_RUN_ROOT/$DDSPC_INSTANCE"
```

This bridge is compiled for the private FIFO directory and derives the unique
application identity from its symlink basename. Start it after the reader has
created its private FIFOs. It does not attach to or restart the production broker.

In a third terminal, source the same session file and run:

```sh
getINDI -h 127.0.0.1 -p "$DDSPC_INDI_PORT" "$DDSPC_INSTANCE.adc.*"
getINDI -h 127.0.0.1 -p "$DDSPC_INDI_PORT" "$DDSPC_INSTANCE.adc_source.*"
getINDI -h 127.0.0.1 -p "$DDSPC_INDI_PORT" "$DDSPC_INSTANCE.ddspc_status.*" \
  "$DDSPC_INSTANCE.ddspc_data.*" "$DDSPC_INSTANCE.ddspc_standard.*" \
  "$DDSPC_INSTANCE.ddspc_accel_assisted.*"
getINDI -h 127.0.0.1 -p "$DDSPC_INDI_PORT" -m -t 0 \
  "$DDSPC_INSTANCE.adc.*" "$DDSPC_INSTANCE.adc_source.*"
```

No `setINDI` is needed. The last command monitors until Ctrl-C. Updates are
application-cadence snapshots (normally about 1 Hz), not an ADC-rate raw recorder.

| Observation | Expected result |
| --- | --- |
| `ddspc_status.execution / output` | `shadow / inhibited` throughout |
| `adc.enabled / valid` | 1 / 1 when fresh, valid samples are accepted |
| `adc_source.stream / datatype / profile` | `acc000 / UINT16 / accelSim12` |
| `adc.width / height / depth` | 2 / 1 / producer temporal depth |
| `adc.raw0 / raw1` | Unsigned X/Y ADC codes; zero acceleration is 2048 / 2048 |
| `adc.acceleration0 / acceleration1` | `(raw-2048)*0.0083799/1.02` m/s²; 2049 / 2047 gives approximately +0.008215588 / -0.008215588 |
| `adc_source.acquired_real_ns / written_real_ns / received_mono_ns` | Exact integer strings advancing with observations; realtime and monotonic epochs differ |
| `adc.samples`, `adc_source.sample_counter / counter` | Advance; sample counter identifies the displayed raw pair, source counter is the header snapshot |
| `adc.rate_hz / age_us` | Measured acquisition rate; fresh mapped age below 20000 µs; -1 age means unavailable |
| `dropped / invalid / timing_gaps` | Record any growth; gaps may invalidate/reset the contiguous segment |
| `duplicates / races / busy` | Poll/retry diagnostics; repeated polls are not necessarily lost samples |
| `overwrites` | Consumer-ring evictions after capacity is reached; distinct from source loss |
| Both models' `updates / predictions / ready / prediction_valid` | Stay 0; ADC health does not qualify DDSPC |
| `ddspc_data.wfs_valid / applied_valid / timing_valid / accel_valid` | Stay 0 in this ADC-only test |

Ignore displayed acceleration whenever `adc.valid=0`. A stale/invalid stream may
leave the last raw pair visible. An entirely valid zero-output simulator produces
constant 2048 / 2048 while its counters/timestamps advance. The 20-ms ADC age limit
is transport health, not a sufficient 500-µs control-alignment requirement.

A depth-one producer can overwrite samples before an ordinary poll runs; a larger
consumer ring cannot recover those samples. Frequent gaps may leave rate at 0
while values arrive. Record this scheduling/retention limitation rather than
changing the producer or weakening validity during the test. The 20-µs polling
worker is not real-time-qualified and consumes some CPU/a reader slot.

## 6. Stop without affecting production

Use Ctrl-C in the monitoring terminal, then Ctrl-C in the reader's foreground
terminal. Reader shutdown joins its ADC worker, invalidates retained data and
releases its mapping/reader slot. Stop the foreground **private broker** with
Ctrl-C in its own terminal. Do not use `pkill`, service stop/restart commands or
unlink any stream. Keep the private source, configuration, patch and logs for
review; no worktree removal is required for shutdown.

Acceptance is progressing ADC samples/timestamps with correct decoding,
`adc.valid=1`, `shadow / inhibited`, both models unqualified, and a clean private
exit. This does not qualify Linux real-time scheduling, producer restart behavior,
WFS alignment, applied-DM provenance or prediction/control quality. Do not disturb
a live producer to test recovery; those failure paths are exercised offline.

## Local validation record

Commands used on the personal workstation:

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.adc-ddspc -j2 check
make -C apps/loPredCtrlAcc -f tests/Makefile.offline BUILD_DIR=build/adc-mvp-asan \
  CXXFLAGS='-O1 -g -std=c++20 -Wall -Wextra -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer' check-adc
python3 -m unittest discover -s apps/loPredCtrlAcc/tests -p native_linux_build_test.py
```

Results on 2026-11-12 (`ASTR-LT-428.local`): 21 guarded application cases, 11 ADC
cases, 25 numerical cases, 2 configuration checks, 2 CLI checks and 8 build-helper
checks passed. All 13 ADC/transport/worker cases also passed under ASan/UBSan.
The app ADC case explicitly verifies successful observation/diagnostics before
any WFS/applied record, with both models unready, untrained and without a valid
prediction. Output guards and reader shutdown/slot release remain covered.
The Makefile check verifies the default C++20 setting reaches the child environment
and the shadow-only macro survives explicit compiler-flag overrides.

Native GCC 14/Eigen 3.4 build and live observations remain operator checks;
offline API doubles are not native ABI or telescope evidence.
