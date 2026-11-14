# Live ADC/WFS/DM input milestone

The normal installed application now supports continuous acquisition without a
private journal. See [NATIVE_APPLICATION.md](NATIVE_APPLICATION.md) for the standard
build/install/xctrl/cursesINDI workflow. This page remains the optional finite
recording procedure.

This increment prepares the existing private `loPredCtrlAcc_native_observe`
executable for the first live-input milestone. It preserves individual 8 kHz ADC
samples, reads the two configured WFS modal entries, and optionally records every
element of each stable observed DM command image. It does not average ADC samples.
The existing accelSim injection is reused unchanged; no producer or operational
configuration is modified.

**The second milestone is not implemented by this profile.** Native training and
prediction remain inhibited. No independently qualified exposure/availability
contract, physical applied-input intervals, or synco slot source has been
established in the available native adapter. Full command images, software
completion and SDK success must not be promoted to those certificates. The
current equal-rate controller workspace is also not an 8 kHz multirate model.
Do not select Legacy execution, forge confirmations, enable the averaged replay
pipeline, or interpret a successful transport test as model qualification.

The `ddspc_mode` callback now updates an already configured shadow pipeline's
actual requested mode, without rebuilding either model. Model readiness checks,
recovery hysteresis and candidate slew limits still apply. In this native-input
profile both models are unqualified, so an assisted selection is rejected. This
fix does not make the causal shadow standard law equivalent to the operational
Legacy law. The operational `loPredCtrl` sources and Acc Legacy calculations are
unchanged.

## Operator-run build and startup on exao2

These commands are instructions for the operator; the agent has not connected to
exao2 or run a native application. First review/commit/copy the changes yourself
so the local source branch on exao2 contains this increment. Use a fresh private
clone to avoid reusing objects compiled without the shadow guard.

```bash
set -euo pipefail
export DDSPC_ORIGIN="$HOME/source/MagAOX-DDSPC"
export DDSPC_RUN_ROOT="$(mktemp -d "$HOME/ddspc-live-inputs.XXXXXX")"
export DDSPC_SRC="$DDSPC_RUN_ROOT/source"
export DDSPC_INSTANCE="loPredCtrlAcc_inputs_$(id -un)_$(date +%s)"
export DDSPC_CXX=g++-14                 # Or the existing GCC 14 executable name.
export DDSPC_INDI_PORT=17626            # Must be unused; private loopback broker only.
git clone --no-hardlinks -b parkertjohnson1/accel_ddspc "$DDSPC_ORIGIN" "$DDSPC_SRC"
git -C "$DDSPC_SRC" rev-parse HEAD > "$DDSPC_RUN_ROOT/revision.txt"
hostname > "$DDSPC_RUN_ROOT/observer-host.txt"
mkdir -p "$DDSPC_RUN_ROOT/config" "$DDSPC_RUN_ROOT/logs" "$DDSPC_RUN_ROOT/sys" \
  "$DDSPC_RUN_ROOT/drivers/fifos" "$DDSPC_RUN_ROOT/calib" "$DDSPC_RUN_ROOT/secrets" \
  "$DDSPC_RUN_ROOT/observations"
(
  set -e
  make -C "$DDSPC_SRC/INDI/liblilxml" -j4 CXX="$DDSPC_CXX" CXXSTD=-std=c++20 EDT=false OPTIMIZE=-fno-fast-math all
  make -C "$DDSPC_SRC/INDI/libcommon" -j4 CXX="$DDSPC_CXX" CXXSTD=-std=c++20 EDT=false OPTIMIZE=-fno-fast-math all
  make -C "$DDSPC_SRC" -j4 CXX="$DDSPC_CXX" CXXSTD=-std=c++20 EDT=false NEED_CUDA=no OPTIMIZE=-fno-fast-math libs_all
  make -C "$DDSPC_SRC/apps/loPredCtrlAcc" -j4 CXX="$DDSPC_CXX" CXXSTD=-std=c++20 EDT=false NEED_CUDA=no \
    OPTIMIZE=-fno-fast-math DDSPC_SHADOW_ONLY=yes native-observe
) 2>&1 | tee "$DDSPC_RUN_ROOT/build.log"
export DDSPC_BIN="$DDSPC_SRC/apps/loPredCtrlAcc/loPredCtrlAcc_native_observe"
test -x "$DDSPC_BIN"
test ! -u "$DDSPC_BIN"
grep -F -- '-DDDSPC_SHADOW_ONLY' "$DDSPC_RUN_ROOT/build.log"
pkg-config --modversion eigen3 mxlib > "$DDSPC_RUN_ROOT/dependencies.txt"
export MAGAOX_PATH="$DDSPC_RUN_ROOT"
export MAGAOX_CONFIG_RPATH=config MAGAOX_LOG_RPATH=logs MAGAOX_SYS_RPATH=sys
export MAGAOX_CALIB_RPATH=calib MAGAOX_SECRETS_RPATH=secrets
touch "$DDSPC_RUN_ROOT/config/magaox.conf"
python3 - "$DDSPC_SRC" "$DDSPC_RUN_ROOT" "$DDSPC_INSTANCE" <<'PY'
from pathlib import Path
import sys
source, run, identity = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
text = (source / 'apps/loPredCtrlAcc/examples/loPredCtrlAcc-live-inputs.conf.example').read_text()
(run / 'config' / (identity + '.conf')).write_text(
    text.replace('@OBSERVATION_DIRECTORY@', str(run / 'observations')))
PY
cat "$DDSPC_RUN_ROOT/config/$DDSPC_INSTANCE.conf"
```

Review the private source names before launching: `acc000`, `aol1_modevalWFS`,
`camwfs`, and `dm01disp`. Change only `native_shadow.dm_stream` in the private
configuration if another existing command source is intended. `dm01disp10` is
the injector channel, not automatically the total combined input. Selected modal
indices 0/1 require confirmation of the calibrated basis. Leave raw source clock
declarations unknown unless their provenance has been established. Use the
existing ImageStreamIO namespace (`MILK_SHM_DIR`, if applicable).

The ADC calibration matches the previously validated profile. Native source
depth 128 must already be provided by the producer; this executable does not
configure it. Keep `shadow=true`, `shadow_pipeline=false`, `own_shmim=false`,
`accel_enabled=false`, `adc.enabled=true`, and `native_shadow.enabled=true`.
There is no framegrabber/output section.

Full DM recording defaults off in existing configurations. This private profile
enables `native_shadow.record_dm_commands=true`, with 64 queued command snapshots
and up to 4096 FLOAT values per image. Oversized images remain visible in metadata
but are **not truncated into a purported complete command**. The corresponding
oversize counter fails verification. Increase the explicit bounded capacity only
after inspecting the actual stream geometry. For 2500 values at 8 kHz, the binary
payload alone is about 1.2 GB per minute; reserve adequate private disk space.

```bash
"$DDSPC_BIN" -n "$DDSPC_INSTANCE" -c "$DDSPC_RUN_ROOT/config/$DDSPC_INSTANCE.conf" --config.validate
"$DDSPC_BIN" -n "$DDSPC_INSTANCE" -c "$DDSPC_RUN_ROOT/config/$DDSPC_INSTANCE.conf" \
  > "$DDSPC_RUN_ROOT/console.log" 2>&1 &
export DDSPC_PID=$!
printf '%s\n' "$DDSPC_PID" > "$DDSPC_RUN_ROOT/private-reader.pid"
declare -p DDSPC_RUN_ROOT DDSPC_SRC DDSPC_INSTANCE DDSPC_BIN DDSPC_PID DDSPC_CXX DDSPC_INDI_PORT \
  MAGAOX_PATH MAGAOX_CONFIG_RPATH MAGAOX_LOG_RPATH MAGAOX_SYS_RPATH MAGAOX_CALIB_RPATH MAGAOX_SECRETS_RPATH \
  > "$DDSPC_RUN_ROOT/session.bash"
```

The observer stops after 60 seconds and drains both bounded journals. To stop
early, send SIGINT only to the saved private PID; do not use `pkill` or service
commands. Early stopping fails the full-duration verification.

```bash
# Only if stopping the private reader early is necessary:
kill -INT "$DDSPC_PID"
wait "$DDSPC_PID"
```

Stopping this PID is also the rollback: no installed executable, service, stream,
or production configuration was replaced. Stop a private INDI broker in its own
terminal if one was started by the operator; preserve the private journals.

## Verify all live connections

Use the private FIFO bridge/loopback broker instructions in
[NATIVE_SHADOW_TESTING.md](NATIVE_SHADOW_TESTING.md#optional-private-indi-diagnostics).
Do not query a production or tunneled broker. With the session sourced:

```bash
getINDI -h 127.0.0.1 -p "$DDSPC_INDI_PORT" -m -t 0 \
  "$DDSPC_INSTANCE.native_adc.*" "$DDSPC_INSTANCE.native_modal.*" \
  "$DDSPC_INSTANCE.native_dm.*" "$DDSPC_INSTANCE.native_shadow.*" \
  "$DDSPC_INSTANCE.adc.*" "$DDSPC_INSTANCE.ddspc_mode.*" \
  "$DDSPC_INSTANCE.ddspc_standard.*" "$DDSPC_INSTANCE.ddspc_accel_assisted.*"
```

Expected: increasing ADC/modal/DM counters, `connected=1`, `stale=0`, ADC receipt
rate near 8000 Hz, modal receipt rate near 2000 Hz, and DM cadence appropriate to
the actual selected producer. `received_rate_hz` measures accepted unique
snapshots using observer monotonic time, resetting after reconnection/restart;
it does not establish a source clock or physical sample time. DM changes must
remain active throughout this test. A static stream does not qualify DM
transport. INDI exposes only the selected values; the binary journal preserves
the complete original command image.

Expected model state: training inhibited, zero model updates/predictions, both
models unqualified, physical output inhibited. No candidate stream is created.
These are deliberate qualification results, not a completed prediction milestone.

After the full run, in the original launch shell:

```bash
wait "$DDSPC_PID"
python3 "$DDSPC_SRC/apps/loPredCtrlAcc/tests/verify_native_observation.py" \
  "$DDSPC_RUN_ROOT/observations" --profile live
python3 "$DDSPC_SRC/apps/loPredCtrlAcc/tests/inspect_native_commands.py" \
  "$DDSPC_RUN_ROOT/observations/dm_commands.bin" --values 8
```

Acceptance: at least 59.5 seconds, ADC 8 kHz and modal/camera 2 kHz within 5%,
continuous active DM observations, no unexplained source-counter gaps, capture
invalids/restarts/timing gaps, metadata queue loss, full-command queue loss,
oversize commands, or I/O errors. Full-command count must equal DM metadata
sample count. Unknown clocks and missing exposure/application associations remain
explicit limitations; a transport pass still sets `training_allowed=false`.

`dm_commands.bin` starts with eight bytes `DDDM0001`. Each record contains 17
little-endian 64-bit words followed by `elements` little-endian IEEE FLOAT32
values. Header order: elements, device, filesystem inode, image inode, generation,
counter, acquisition raw ns, publication raw ns, receipt monotonic ns, observer
availability monotonic ns, width, height, depth, datatype, flags, payload hash,
cnt1. The four timestamp words are signed; all others are unsigned. Values retain
producer ordering and units. They are commanded values, not modal projections or
physical-application certificates. The reader tool tolerates an unfinished tail
only with explicit `--allow-partial` during a running observation.

## Local checks performed without native streams

```bash
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j4 \
  BUILD_DIR=build/live-input-smoke check-native-observation
```

This compiles the real application header, ADC processing, observer workers and
output guards with explicit framework/ImageStreamIO doubles. It is an essential
integration smoke check, not an additional standalone DDSPC performance study.
Native Linux compile/link still requires the operator build above: ImageStreamIO
development dependencies are unavailable on the local Mac. No application,
service, installation, producer, DM output, or git commit is run by the agent.
