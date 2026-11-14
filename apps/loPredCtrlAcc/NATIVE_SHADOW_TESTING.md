# Native observation-only handoff

This stage observes existing streams for 60 seconds. It never constructs qualified
WFS/applied-input records, trains a model, generates control candidates, or sends
DM commands. `native_shadow.enabled` defaults **false** in `loPredCtrlAcc`; the
existing ADC-only example and calibration are unchanged. The separate
`loPredCtrlAcc_native_observe` entrypoint refuses startup without a validated
native-observation configuration and a `DDSPC_SHADOW_ONLY` build.

No commands below have been run on exao2 by the agent. They are operator-run
instructions after review. Do not start, restart or reconfigure producers to make
this test pass. Observe the existing repaired accelSim and the already validated
`acc000` source depth 128. Do not enable DDSPC output or copy a production output
configuration.

## ADC/WFS-only result and timing follow-up

The operator reported a successful 60-second native run on exao2 with Eigen 5:
480,018 ADC samples at 8 kHz and 120,000 modal/camera samples each at 2 kHz,
no reported capture/journal losses, normal exit. One inactive DM observation
left the original four-source verifier failing its DM activity requirement.
Use the explicit `--profile adc-wfs` analysis for this acquisition stage; DM
qualification is deferred, not silently passed. These are operator-provided
results, not an independent analysis of the CSV on this workstation.

See [NATIVE_TIMING_AUDIT.md](NATIVE_TIMING_AUDIT.md) for read-only analysis of the
existing recording, counter-labelled 4:1 phase, clock provenance, causal sample
availability and future synco constraints. No new native run or rebuild is needed
to run those Python tools. All training/output guards remain unchanged.

## What is recorded and what remains unknown

| Role | Default existing input | Evidence retained | Interpretation |
| --- | --- | --- | --- |
| ADC | `acc000` | Every available decoded pair via `AdcReader::copy_after`, raw UINT16 pair, original acquisition/publication times, mapped monotonic times, original decoder receipt, observer availability, device/inode/generation/segment/counter | Same `accelSim12` calibration; source loss, cursor loss and journal loss counted separately |
| Modal | `aol1_modevalWFS` | FLOAT vector extent, selected entries 0/1, min/max/RMS/digest, metadata timestamps/counter and backing/image identity | Operator must verify this is the calibrated basis of interest; selected entries are not automatically tip/tilt |
| Camera | `camwfs` | UINT8/UINT16 layout and metadata counter/atime/writetime, receipt/availability and identity; no camera pixel copy | `atime` is retained as metadata, **not an exposure boundary** |
| DM | `dm01disp` | Combined FLOAT request extent, first two values, min/max/RMS/digest, metadata and identity | No assertion of post-limit actuator values, driver consumption, optical delay or physical application |
| Completion / SDK | Empty, optional | Exact existing UINT64 16-word software endpoint record | Only if already published; never enable a publisher for this test |

Native snapshot copies bracket the payload with original metadata and reject
busy/changed/unsupported layouts. They are best-effort latest-frame observers,
not lossless multi-reader transactions; modal/camera/DM counter gaps are recorded.
For cubes these observers inspect the latest plane and its per-plane metadata;
only the existing ADC reader drains retained source planes chronologically.
No semaphore wait slot is registered by the new observers. Existing ADC reader
notification-slot bookkeeping remains unchanged. No input pixels, publication
counters or timestamps are written, and no image is created/destroyed/unlinked.

The portable completion definitions are reused from accelSim worktree commit
`9d4c5a99` (`libMagAOX/common/dmCompletion.hpp`, `dmTiming.hpp`, `timingStats.hpp`).
No producer, driver, scheduler or hardware implementation is imported. A complete
software endpoint is checked using its actual protocol predicate and can match
an observed combined-input **ImageStreamIO metadata inode**, counter and raw
publication timestamp. Filesystem inode is recorded separately and never used
as a substitute. A bounded 512-command identity history is cleared on source
replacement. Missing snapshots and late arrivals can prevent a join. Publisher
session/epoch/sequence and stream publication counters are independent: a busy
and finish publication can share one command sequence. SDK success and reader
completion do not certify mirror motion. Nonmatching protocol kinds, incomplete
records, sequence regressions and publisher changes remain explicit.

Native `cnt1` is retained without interpretation, including on 2-D inputs.
`has_counter1=false` on ADC records because the existing cursor API does not
retain that header field; no plane index is inferred from the ADC sample counter.
WFS/camera origin association is unverified for every modal frame: equal counters
or nearby timestamps do not establish exposure identity. No exposure start/end,
clock mapping, applied-command interval or physical-application evidence is
invented. All five application shadow record-admission seams and training INDI
requests reject input in native-observation mode. Both models remain unqualified
with zero updates/predictions. Original WFS-monitor startup and operational FPS
subscriptions are skipped in this mode; Legacy and framegrabber startup/output
remain inhibited.

Each source declares acquisition/publication clocks independently. ADC retains
its existing REALTIME-to-MONOTONIC capture mapping. Other clocks default to
`unknown`: raw timestamps survive, publication age is unavailable, and no
cross-clock subtraction occurs. If the exact producer build has been verified,
set the private `<role>_acquisition_clock` / `<role>_publication_clock` to
`realtime`, `tai` or `monotonic`. No fixed TAI offset is assumed. A declared clock
uses a local monotonic bracket for age diagnostics only; it does not map exposure
or endpoint clocks or establish uncertainty sufficiently for control. Future
publication outside the bracket is a clock mismatch. Producer-local MONOTONIC
completion words remain raw; do not equate them across hosts without evidence.

## Local development verification

From the repository root, these targets use explicit API/framework doubles,
private ordinary files and real ADC/model/application code. They never connect
native ImageStreamIO streams or run a native application:

```sh
hostname                         # Must be a personal workstation, never exao*.
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j4 check-native-observation
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j4 check
make -C apps/loPredCtrlAcc -f tests/Makefile.offline check-adc
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j4 check-confirmed-replay
```

The confirmed replay optionally reads only `../MagAOX/apps/accelSim/accelSimModel.hpp`.
It starts no producer. Pure accounting tests cover counter discontinuities,
clock mismatches, missing exposure associations, endpoint protocol/error/restart
cases, and bounded queue overflow/recovery. Actual worker tests cover absent
streams, ADC forwarding/calibration, invalid layouts, replacement/reconnection,
exclusive files and bounded shutdown. Actual private app-header tests cover
training refusal and physical-output guards.

On an already provisioned **non-instrument Linux development host**:

```sh
python3 apps/loPredCtrlAcc/tests/native_linux_build.py --observer-only \
  --cxx g++-14 --jobs 2 --build-root /tmp/ddspc-native-build-UNIQUE
```

This builds fresh private sources/libraries and links the separate executable;
it never runs/installs it. The helper deliberately refuses `exao*`. Native Linux
ImageStreamIO compile/link is not demonstrated on this Mac; the offline doubles
cannot establish native ABI correctness or real-time loss/CPU behavior. The
operator build below is a required gate, not an assumed pass.

## Transfer the reviewed commit without installation or push

On the development workstation, after the separate native-observation commit:

```sh
export DDSPC_BUNDLE="$HOME/loPredCtrlAcc-native-observe.bundle"
git bundle create "$DDSPC_BUNDLE" HEAD ^1346b311
# Operator-run transfer only; the agent does not connect to exao2.
scp "$DDSPC_BUNDLE" exao2:loPredCtrlAcc-native-observe.bundle
```

The bundle contains the changes after the committed replay checkpoint. The
existing source clone on exao2 must contain `1346b311` to supply its prerequisite.
Do not fetch into/switch/build the operational controller checkout. Use a new
private clone, as follows in Bash on exao2:

```sh
set -euo pipefail
hostname
export DDSPC_ORIGIN="$HOME/source/MagAOX-DDSPC"
export DDSPC_RUN_ROOT="$(mktemp -d "$HOME/ddspc-native-observe.XXXXXX")"
export DDSPC_SRC="$DDSPC_RUN_ROOT/source"
export DDSPC_INSTANCE="loPredCtrlAcc_obs_$(id -un)_$(date +%s)"
export DDSPC_CXX=g++-14               # Use g++ if that is the installed GCC 14 executable.
export DDSPC_INDI_PORT=17626           # Must be unused and distinct from production.

git clone --no-hardlinks "$DDSPC_ORIGIN" "$DDSPC_SRC"
git -C "$DDSPC_SRC" bundle verify "$HOME/loPredCtrlAcc-native-observe.bundle"
git -C "$DDSPC_SRC" fetch "$HOME/loPredCtrlAcc-native-observe.bundle" HEAD
git -C "$DDSPC_SRC" checkout --detach FETCH_HEAD
git -C "$DDSPC_SRC" rev-parse HEAD > "$DDSPC_RUN_ROOT/revision.txt"
mkdir -p "$DDSPC_RUN_ROOT/config" "$DDSPC_RUN_ROOT/logs" "$DDSPC_RUN_ROOT/sys" \
  "$DDSPC_RUN_ROOT/drivers/fifos" "$DDSPC_RUN_ROOT/calib" "$DDSPC_RUN_ROOT/secrets" \
  "$DDSPC_RUN_ROOT/observations"
"$DDSPC_CXX" --version | tee "$DDSPC_RUN_ROOT/compiler.txt"
test "$("$DDSPC_CXX" -dumpversion | cut -d. -f1)" = 14
pkg-config --modversion eigen3 mxlib | tee "$DDSPC_RUN_ROOT/dependencies.txt"
# Record the installed Eigen version. The operator validated this native observer with Eigen 5; do not downgrade it.
```

If the committed source already exists locally on exao2, replace the bundle
fetch with an exact reviewed commit in the private clone. Keep a fresh clone:
Make does not track compiler-flag changes in old objects/PCHs.

## Build a separate private executable

```sh
(
  set -e
  make -C "$DDSPC_SRC/INDI/liblilxml" -j2 CXX="$DDSPC_CXX" CXXSTD=-std=c++20 EDT=false OPTIMIZE=-fno-fast-math all
  make -C "$DDSPC_SRC/INDI/libcommon" -j2 CXX="$DDSPC_CXX" CXXSTD=-std=c++20 EDT=false OPTIMIZE=-fno-fast-math all
  make -C "$DDSPC_SRC" -j2 CXX="$DDSPC_CXX" CXXSTD=-std=c++20 EDT=false NEED_CUDA=no OPTIMIZE=-fno-fast-math libs_all
  make -C "$DDSPC_SRC/apps/loPredCtrlAcc" -j1 CXX="$DDSPC_CXX" CXXSTD=-std=c++20 EDT=false NEED_CUDA=no \
    OPTIMIZE=-fno-fast-math DDSPC_SHADOW_ONLY=yes native-observe
) 2>&1 | tee "$DDSPC_RUN_ROOT/native-build.log"
export DDSPC_BIN="$DDSPC_SRC/apps/loPredCtrlAcc/loPredCtrlAcc_native_observe"
test -x "$DDSPC_BIN"
test ! -u "$DDSPC_BIN"
grep -F -- '-DDDSPC_SHADOW_ONLY' "$DDSPC_RUN_ROOT/native-build.log"
sha256sum "$DDSPC_BIN" > "$DDSPC_RUN_ROOT/binary.sha256"
```

Stop on any build error. Inspect compile commands for C++20, strict floating
point and the guard; there must be no `tests/native_stubs` includes. No install,
sudo, setuid, dependency upgrade, service action or installed binary is used.

## Private configuration and launch

```sh
export MAGAOX_PATH="$DDSPC_RUN_ROOT"
export MAGAOX_CONFIG_RPATH=config MAGAOX_LOG_RPATH=logs MAGAOX_SYS_RPATH=sys
export MAGAOX_CALIB_RPATH=calib MAGAOX_SECRETS_RPATH=secrets
touch "$DDSPC_RUN_ROOT/config/magaox.conf"
python3 - "$DDSPC_SRC" "$DDSPC_RUN_ROOT" "$DDSPC_INSTANCE" <<'PY'
from pathlib import Path
import sys
source, run, identity = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
text = (source / 'apps/loPredCtrlAcc/examples/loPredCtrlAcc-native-observe.conf.example').read_text()
(run / 'config' / (identity + '.conf')).write_text(text.replace('@OBSERVATION_DIRECTORY@', str(run / 'observations')))
PY
cat "$DDSPC_RUN_ROOT/config/$DDSPC_INSTANCE.conf"
```

Verify the **existing** source names and modal basis with the operator. Keep
`completion_stream` and `sdk_stream` empty unless those compatible streams are
already published and their provenance is known. Missing configured sources are
recorded, never created. An absent optional endpoint is a timing-evidence gap.
Keep `shadow=true`, `shadow_pipeline=false`, `own_shmim=false`,
`accel_enabled=false`, `adc.enabled=true`. No DM output name is configured.

Use the producer's already established ImageStreamIO namespace. If necessary,
export its verified `MILK_SHM_DIR`; `MAGAOX_PATH` does not select that namespace.
Do not change permissions, producer rates, stream depth or semaphores here.
Reserve at least 0.5 GB of private local disk for the ASCII journal; observe
available CPU/disk resources before launch. This poller changes no affinity,
priority or operational thread settings.

```sh
"$DDSPC_BIN" -n "$DDSPC_INSTANCE" -c "$DDSPC_RUN_ROOT/config/$DDSPC_INSTANCE.conf" --config.validate
"$DDSPC_BIN" -n "$DDSPC_INSTANCE" -c "$DDSPC_RUN_ROOT/config/$DDSPC_INSTANCE.conf" \
  > "$DDSPC_RUN_ROOT/console.log" 2>&1 &
export DDSPC_PID=$!
printf '%s\n' "$DDSPC_PID" > "$DDSPC_RUN_ROOT/private-reader.pid"
declare -p DDSPC_RUN_ROOT DDSPC_SRC DDSPC_INSTANCE DDSPC_BIN DDSPC_PID DDSPC_CXX DDSPC_INDI_PORT \
  MAGAOX_PATH MAGAOX_CONFIG_RPATH MAGAOX_LOG_RPATH MAGAOX_SYS_RPATH MAGAOX_CALIB_RPATH MAGAOX_SECRETS_RPATH \
  > "$DDSPC_RUN_ROOT/session.bash"
printf 'Other terminals: source %q\n' "$DDSPC_RUN_ROOT/session.bash"
tail -f "$DDSPC_RUN_ROOT/console.log"     # Ctrl-C stops only tail, not the reader.
```

The worker stops input after 60 monotonic seconds, drains the queue, closes the
journal/summary, and requests normal app shutdown at the next appLogic tick.
No INDI broker is required to record. **Do not interrupt unless needed**: the
verification script requires a full-duration run. The same output directory
cannot be reused; exclusive files prevent overwriting a previous observation.

## Optional private INDI diagnostics

Run only against this private loopback broker, never a tunneled instrument broker.
Build the FIFO bridge privately (build before launching if desired):

```sh
source /absolute/path/printed/above/session.bash
make -C "$DDSPC_SRC/INDI/xindidriver" CXX="$DDSPC_CXX" CXXSTD=-std=c++20 EDT=false \
  OPTIMIZE=-fno-fast-math FIFO_PATH="$DDSPC_RUN_ROOT/drivers/fifos" all
ln -s "$DDSPC_SRC/INDI/xindidriver/xindidriver" "$DDSPC_RUN_ROOT/$DDSPC_INSTANCE"
test -z "$(ss -H -ltn "sport = :$DDSPC_INDI_PORT")"
indiserver -p "$DDSPC_INDI_PORT" "$DDSPC_RUN_ROOT/$DDSPC_INSTANCE"
```

In another terminal with the session sourced:

```sh
getINDI -h 127.0.0.1 -p "$DDSPC_INDI_PORT" -m -t 0 \
  "$DDSPC_INSTANCE.native_shadow.*" "$DDSPC_INSTANCE.native_adc.*" \
  "$DDSPC_INSTANCE.native_modal.*" "$DDSPC_INSTANCE.native_camera.*" \
  "$DDSPC_INSTANCE.native_dm.*" "$DDSPC_INSTANCE.native_completion.*" "$DDSPC_INSTANCE.native_sdk.*" \
  "$DDSPC_INSTANCE.adc.*" "$DDSPC_INSTANCE.adc_source.*" \
  "$DDSPC_INSTANCE.ddspc_standard.*" "$DDSPC_INSTANCE.ddspc_accel_assisted.*"
```

No `setINDI` is needed. Expect `training_inhibited=1`, all qualification fields 0,
model updates/predictions/ready 0, growing source counts and elapsed time. ADC
values/rate use existing diagnostics. `native_*` counters/numbers are convenient
INDI doubles; exact 64-bit IDs/timestamps are in CSV. `stale` means observer
silence >10 ms, **not** a 2-kHz control freshness gate. Publication age describes
the last snapshot in a declared clock, and -1 means unknown. Busy/duplicate polls
are not lost frames. Optional endpoints remain disconnected when unconfigured.

## Verify the completed 60-second run

In the launch shell:

```sh
wait "$DDSPC_PID"
python3 "$DDSPC_SRC/apps/loPredCtrlAcc/tests/verify_native_observation.py" \
  "$DDSPC_RUN_ROOT/observations" | tee "$DDSPC_RUN_ROOT/verification.json"
python3 -m json.tool "$DDSPC_RUN_ROOT/observations/summary.json"
head -n 5 "$DDSPC_RUN_ROOT/observations/observations.csv"
```

Transport acceptance: >=59.5 s, required ADC/modal/camera/DM sample coverage;
ADC accepted rate 8 kHz +/-5% with >=59 s timestamp coverage; zero capture drops,
invalids, timing gaps and restarts; zero cursor misses and journal overflows;
zero unsupported layouts, clock mismatches, invalid endpoint records or
unexpected source/publisher restarts. The CSV row count must equal `written`,
and per-role samples must agree. Ring `overwrites` are normal retention eviction,
not source loss; `adc_missing` overlaps capture gaps and must not be added to them.
A startup `adc_breaks` event can simply establish a source/segment baseline.

Report **all** modal/camera/DM counter gaps, incomplete endpoints, sequence gaps
and failed associations. These observers may miss latest-only updates; they
must be measured rather than hidden or treated as consumed commands. Unknown
clock and missing modal-camera association are expected initial limitations,
not transport qualification. The verifier reports them separately and always
returns `training_allowed=false`. A transport pass is not a learning-readiness
pass. Optional absent telemetry prevents proving application timing even if
all ADC samples arrive.

Retain revision/hash/compiler/dependencies/config, console, CSV, summary and
verification report. Review raw camera/modal timestamps, identities, modal basis,
clock domains, endpoint coverage/association and optical-delay evidence before
any separate training proposal. This stage retains vector summaries/selected
values, not full actuator payloads, and cannot reconstruct applied command
histories for a future learner. No controller mathematics or calibration changes.

## Stop / rollback only this private test

Normally the finite run exits itself. To abort in the launch shell:

```sh
if kill -0 "$DDSPC_PID" 2>/dev/null; then
  test "$(readlink "/proc/$DDSPC_PID/exe")" = "$DDSPC_BIN"
  ps -p "$DDSPC_PID" -o pid=,args=         # Confirm the unique private identity.
  kill -TERM "$DDSPC_PID"
  wait "$DDSPC_PID" || true
fi
```

Ctrl-C only the optional private INDI monitor and broker in their own terminals.
Keep the private directory for review. Nothing was installed, so rollback needs
no service restart, production configuration edit, DM zeroing, image deletion,
`pkill` or replacement binary. A later test uses a new private run directory.

## Recorded local verification (2026-11-13)

Personal Mac `ASTR-LT-428.local`, Apple Clang, installed Eigen 5.0.1/mxlib 0.2.0;
framework/ImageStreamIO doubles, no native applications or live streams:

| Check | Result |
| --- | --- |
| Full offline regression | 123 cases passed |
| Guarded observation + application tests | 30 cases, 10,438 assertions passed |
| ADC unit/synthetic/transport | 15 cases, 448,098 assertions passed |
| Independently confirmed-input replay | 6 cases, 40,049 assertions passed |
| Focused suite under ASan + UBSan | 30 cases passed, no findings |
| Focused suite under ThreadSanitizer | 30 cases passed, no findings |
| Build orchestration / private config + artifact tools | 9 / 2 Python tests passed |
| Existing unpaced application replay, nominal 2-kHz WFS / 8-kHz ADC, H15/A5 | Total-frame p50 102.916 us; p99 132.792 us; max 302.292 us; 0/1800 >500 us |
| Native Linux executable compile/link | Not verified; required operator build gate |
| exao2 observation / exposure / physical application | Not performed / unqualified / unqualified |

The benchmark tests the existing replay path with native observation **off**;
it makes no claim about the new journal's native timing or on-instrument CPU.
The concurrent queue regression checks 100,000 owned publications with exact
accepted+overflow accounting. No mathematical algorithm or ADC calibration was
changed. All original controller/producer code and tracked numerical results
are unchanged.

Reproduce the focused sanitizer builds with separate directories and explicit
C++20 flags (Make does not track changed command-line flags):

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 \
  BUILD_DIR=build/native-observation-asan \
  CXXFLAGS='-std=c++20 -Wall -Wextra -fno-fast-math -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' \
  check-native-observation
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 \
  BUILD_DIR=build/native-observation-tsan \
  CXXFLAGS='-std=c++20 -Wall -Wextra -fno-fast-math -O1 -g -fsanitize=thread' \
  check-native-observation
make -C apps/loPredCtrlAcc -f tests/Makefile.replay -j2 check
```
