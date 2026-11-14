# Native loPredCtrlAcc commissioning

The normal `loPredCtrlAcc` executable now uses the standard MagAO-X application
lifecycle, configuration lookup, logger, PID directory, FIFOs and INDI device
identity. The installed entrypoint defaults to continuous native input acquisition,
no disk recording, and compile-time physical-output inhibition. It is no longer
necessary to run the separate finite observation executable or use a private
runtime directory for normal operation.

No installation, registration, application launch, broker restart or instrument
connection was performed by the agent. All instrument commands below are for the
operator after review. `apps/loPredCtrl` and global deployment files are unchanged.

## Build and install

On an appropriately provisioned exao2 checkout containing the reviewed changes:

Before installation, check whether this app's name/binary/configuration are already
in use. If so, arrange an authorized app-only stop and preserve that app's previous
binary/configuration for rollback. The installer preserves an existing config,
but the standard binary install replaces `/opt/MagAOX/bin/loPredCtrlAcc`.

```bash
cd ~/source/MagAOX-DDSPC/apps/loPredCtrlAcc
make -j4
sudo make install
```

The app uses the existing MagAO-X Make include and installed dependencies. GCC 14
or newer and C++20 are required; Eigen comes from the existing `eigen3` pkg-config
entry. If the intended compiler is not the default, use `CXX=g++-14` on both Make
commands. Do not upgrade system dependencies as part of this commissioning.

Every normal/default build forcibly defines `DDSPC_SHADOW_ONLY` and places
`-fno-fast-math` after caller flags. `DDSPC_SHADOW_ONLY=no` cannot enable output.
A build-profile stamp forces stale app objects to rebuild once, and numerical
objects now depend on the app headers. Clean/rebuild when changing compilers.
The historical Legacy controller remains available only to the existing numerical
test harness; the installed entrypoint cannot start it.

The inherited installation convention installs `/opt/MagAOX/bin/loPredCtrlAcc` as
root-owned mode 4755. The app-specific install hook installs `loPredCtrlAcc.conf`
as mode 0644 in `/opt/MagAOX/config` **only if no file or symlink already exists**.
An existing configuration is preserved and must be compared/merged by the operator.
The compiled guards fail closed if it requests Legacy/output behavior. There is
no implicit process registration, driver registration, service restart or launch.

```bash
stat -c '%U %a %n' /opt/MagAOX/bin/loPredCtrlAcc /opt/MagAOX/config/loPredCtrlAcc.conf
/opt/MagAOX/bin/loPredCtrlAcc -n loPredCtrlAcc --config.validate
```

Use a personal Linux development host for a private native link check, without
installation or execution (run from the repository root):

```bash
python3 apps/loPredCtrlAcc/tests/native_linux_build.py \
  --cxx g++-14 --jobs 4 --build-root /tmp/ddspc-native-build-UNIQUE
```

This helper refuses instrument hostnames and builds both guarded entrypoints from
a frozen source snapshot. The local Mac has no native ImageStreamIO development
headers; local API-double builds are not a Linux ABI/link qualification.

## xctrl and INDI registration

The actual discovery implementation is [scripts/xctrl](../../scripts/xctrl).
It reads `/opt/MagAOX/config/proclist_${MAGAOX_ROLE}.txt`, not the older
`processes_*.txt` filename mentioned in its help text. Each entry has two words:
instance name and executable name. It starts `/opt/MagAOX/bin/<executable>` with
`-n <instance>` in a tmux session owned by `xsup`. The framework then loads
`/opt/MagAOX/config/<instance>.conf`, uses `<instance>` as the INDI device, and
maintains `/opt/MagAOX/sys/<instance>/pid` and its standard FIFOs.

During an approved deployment step, merge the following single entry into the
existing role process list; do not replace the list:

```text
loPredCtrlAcc loPredCtrlAcc
```

The supplied [proclist fragment](deployment/proclist.fragment) contains that
entry. Confirm `$MAGAOX_ROLE`; the hostname `exao2` does not determine the role's
filename. An already existing entry must be reviewed, not duplicated.

For `cursesINDI`, append `loPredCtrlAcc` to the existing role server configuration's
`[local] drivers` vector. The conventional server instance is
`is${MAGAOX_ROLE^^}`; its file is `/opt/MagAOX/config/<server-instance>.conf`.
Preserve every existing driver. [indi-local.fragment](deployment/indi-local.fragment)
is a merge example, not a replacement configuration.

[xindiserver](../xindiserver/xindiserver.hpp) loads that vector and creates each
`/opt/MagAOX/drivers/<instance>` symlink to `xindidriver`, which connects the
application's FIFOs to the broker. Installing an executable/configuration alone
does not make the device appear in a running broker. The driver-list change needs
an operator-approved broker reload/restart; this app does not perform it.

Operator commands to identify and back up the existing registration files before
editing, during that approved deployment window:

```bash
test -n "$MAGAOX_ROLE" || exit 1
DDSPC_PROCESS_LIST="/opt/MagAOX/config/proclist_${MAGAOX_ROLE}.txt"
DDSPC_SERVER="is${MAGAOX_ROLE^^}"
DDSPC_SERVER_CONFIG="/opt/MagAOX/config/${DDSPC_SERVER}.conf"
DDSPC_BACKUP_DIR=$(mktemp -d "$HOME/ddspc-registration.XXXXXX")
cp -L "$DDSPC_PROCESS_LIST" "$DDSPC_BACKUP_DIR/proclist.txt"
cp -L "$DDSPC_SERVER_CONFIG" "$DDSPC_BACKUP_DIR/indiserver.conf"
cp -L /opt/MagAOX/config/loPredCtrlAcc.conf "$DDSPC_BACKUP_DIR/loPredCtrlAcc.conf"
sudoedit "$DDSPC_PROCESS_LIST" "$DDSPC_SERVER_CONFIG"
```

Add the process mapping and merge the driver into the existing vector as described
above. If installation preserved a previous app configuration, compare it with the
supplied `loPredCtrlAcc.conf` and review/merge it using `sudoedit` before launch.
If an instance with this name already runs, first arrange an authorized app-only
stop and preserve its executable/configuration; do not replace a running instance
implicitly. These steps do not authorize changing another application.

**Important restart scope:** this repository's `xctrl restart loPredCtrlAcc`
also adds the role's INDI server to the restart targets. It can therefore interrupt
all clients of that broker. Do not run it during observations without approval.
Once registration and broker setup have been completed at an authorized time,
the requested workflow is:

```bash
xctrl restart loPredCtrlAcc
xctrl status loPredCtrlAcc
cursesINDI
```

For app-only cycling with an already configured/running broker, use:

```bash
xctrl stop loPredCtrlAcc
xctrl start loPredCtrlAcc
```

This avoids the restart command's server-target expansion. The unmodified start
command may start the role broker if it is absent. Verify the deployed xctrl
version's behavior; no global xctrl patch is part of this change.

## Native acquisition and diagnostics

Installed defaults read existing `acc000`, `aol1_modevalWFS`, `camwfs` and
`dm01disp` streams. No stream is created, destroyed, unlinked or published.
Individual calibrated ADC samples retain raw counts, counter, acquisition,
publication, receipt, generation and segment information. Each source has a
bounded chronological metadata/sample history; the complete DM FLOAT payload is
also retained independently of recording. Neither ADC averaging nor synthetic
WFS upsampling is used. The ADC calibration is unchanged.

Complete DM retention uses `native_shadow.command_queue_capacity` (default 64)
and `command_max_elements` (default 4096); these bound both the number and size of
original FLOAT images. Oversize images are reported as unretained rather than
silently truncated. No command data are interpreted as physically applied.

The existing-only observer checks ImageStreamIO write/counter/layout brackets.
The operational `loPredCtrl` has a command publisher, not a DM telemetry reader;
this app reuses the established native read-only snapshot transport for the same
FLOAT command-image protocol. Command observation, reader completion and SDK
success never imply physical DM application. Modal indices 0/1 and the modal/DM
coordinate mapping still need the appropriate calibration evidence.

Useful `cursesINDI` properties under device `loPredCtrlAcc`:

| Property | Meaning |
| --- | --- |
| `app_health`, `input_health` | Observation health, uptime, independent stream states/restarts and idle DM |
| `input_rates`, `adc_values` | Capture/forwarding rates, modal/camera/DM rates, raw/calibrated ADC values and rejection |
| `input_losses`, `input_buffers` | Source/forwarding loss, races, timing breaks, retained history, pending ADC and recorder queues |
| `timing` | Explicit unavailable clock/exposure/association/phase contracts and four nominal slots |
| `ddspc_mode` | Standard or assisted request; a request never grants model readiness or output authority |
| `ddspc_control`, `ddspc_performance`, `safety` | Independent model state/counts, unavailable metrics and mandatory output interlocks |

The default is 59 app-specific elements; framework properties remain. See
[NATIVE_DIAGNOSTICS.md](NATIVE_DIAGNOSTICS.md) for the exact list, ADC forwarding
investigation, loss semantics, debug migration and the extended verification test.
The earlier detailed metadata properties are available only through explicit
startup `[diagnostics] debug_indi=true`.

Periodic updates are capped by `[diagnostics] period_ms=250`, independently of
the input workers. The main application loop is 100 ms; effective reporting may
be slower than the 4 Hz maximum. Acknowledging an explicit mode request is immediate.
In debug mode, `new_sample` means a new source publication was observed since the preceding
diagnostic update. `received_rate_hz` counts copied samples; `publication_rate_hz`
uses source counter advances over observer time. Both use bounded rate windows
and report zero during source silence. They do not qualify producer clocks.

For DM in debug mode: `connected=1, valid=1, idle=1, stale=0, publication_rate_hz=0` is a healthy
static request. An attached stream with no first publication can be idle but not
yet valid. Missing/replaced mappings, malformed payloads and a persistently busy
producer remain distinct from idle. Source absence or idle does not stop ADC/WFS
acquisition or force an application error. A stale mapping left by a dead producer
cannot by itself establish producer liveness or physical command consumption.

`valid` means transport/payload health, not a qualified DDSPC input. Timestamp
metadata validity, unknown source clock domains, and model readiness are separate.
History evictions are expected bounded retention, not dropped producer frames.
The metadata journal and full-command recorder have separate queue-loss counters.
Source counter gaps remain explicit; the latest-frame WFS/DM snapshots are not
claimed lossless on an unscheduled Linux host.

## Initial 60-second acceptance

Only the operator performs these steps on exao2. Do not restart producers or
change the already validated accelSim injection/calibration for this acceptance.

1. Review registration and broker restart scope above, install the reviewed
   binary/configuration, and validate it. Keep all shadow/output protections.
2. Start the app with xctrl. In `cursesINDI`, select device `loPredCtrlAcc`.
3. After a 2-second warmup, observe for 60 seconds. Require ADC received rate within
   2% of 8000 Hz and modal/camera within 2% of 2000 Hz, advancing counters, sensible
   ADC conversion, valid `input_health` states and stable connections. These are acquisition
   acceptance bounds, not phase or real-time deadline qualification.
4. Confirm no additional ADC capture loss, invalids, timestamp rejection or
   unexpected restart. Require no unexpected WFS/camera counter gaps or timestamp
   regressions; investigate any increase before accepting continuity.
   Continuous runs must not finish at 60 seconds or create unbounded journals.
5. Leave DM static if appropriate: its idle state and zero frequency must not
   produce an application error or make healthy ADC/WFS streams invalid. If DM
   updates are present, confirm the source counter and command history advance.
6. Confirm `app_health.shadow=true`, `safety.dm_write_enabled=false`, and native
   model updates/readiness/predictions remain zero. Change `ddspc_mode`: requests
   may change, but no interlock, fit or physical output is enabled. Legacy controls
   and training are not exposed in the compact native path.
7. Stop only this app with `xctrl stop loPredCtrlAcc`, check its PID status, and
   start it again. Confirm reconnection, safe defaults, and continued diagnostics.

`cursesINDI` in this repository connects to loopback port 7624. Registration into
the appropriate existing broker is required; it has no command-line host/port
options in this version. Do not create a second broker on an occupied port.
No physical-output activity is inferred from DM counters: accelSim and existing
AO processes may legitimately publish while this app observes. The no-output
guarantee comes from the compiled guards and inhibited output-helper lifecycle.

Rollback stops only `loPredCtrlAcc`; the operator may restore its previous binary
and configuration during authorized maintenance. This installation does not
replace `loPredCtrl`, global process lists, broker configurations or services.

```bash
xctrl stop loPredCtrlAcc
xctrl status loPredCtrlAcc
```

Leave an otherwise healthy role broker running. Removing the registration later
requires merging out only this app's entries and an approved broker reload/restart;
do not overwrite global files with the backup if other operators have changed them.

## Qualification limits and next stage

The native input histories and complete-command copy API prepare causal multirate
association. They do not yet supply qualified WFS exposure boundaries, verified
cross-source clocks, applied-command intervals, or modal command projections.
Phase offset/jitter remain `unqualified` in compact mode (debug: `phase_valid=0`, values -1) until an actual
synchronization reference and source-clock mapping are supplied. Camera publication
time is not substituted for exposure time. The 20 ms ADC observation-health
threshold is not an 8 kHz control freshness threshold.

The RLS/QR/DDSPC algorithms are unchanged, and no EOF predictor is introduced.
Both native models remain unqualified. Raw-ADC multirate fitting and slot prediction
are future integration work; the current averaged offline pipeline is not enabled
by this installed profile. Synco targets and input-ready cutoffs must govern any
future slot candidates, with computation completed before the cutoff. The nominal
15 us difference between a 110 us write and 125 us spacing is not a compute budget.
This initial installed version has no active-output option or automatic transition.

Local checks use the actual application headers, ADC processing and observer with
explicit transport/framework doubles, never native instrument streams. Run from
the repository root:

```bash
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j4 \
  BUILD_DIR=build/native-app-integration check-native-observation
apps/loPredCtrlAcc/build/native-app-integration/native-observation/ddspc_offline_test '[loPredCtrlAcc][adc]'
```

The same focused suite passes with ASan and UBSan:

```bash
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 \
  BUILD_DIR=build/native-app-asan \
  CXXFLAGS='-O1 -g -std=c++20 -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer' \
  check-native-observation
```

Previous integration checkpoint: 51 C++ cases / 11,007 assertions and 43 Python checks passed in the
optimized and sanitizer runs. The existing confirmed-input replay also passes
6 cases / 40,049 assertions. The 15 historical `loPredCtrl` source files match the
restored reference byte-for-byte. One existing Clang warning in
`INDI/libcommon/IndiProperty.hpp` about a temporary string remains outside this
change. No native Linux link, xctrl launch, broker registration, signal-handling
acceptance or live rate measurement was performed here.

## Changed-file manifest

All application paths below are relative to `apps/loPredCtrlAcc`:

| Files | Change |
| --- | --- |
| `Makefile`, `loPredCtrlAcc.cpp`, new `loPredCtrlAcc_native.hpp`, new `loPredCtrlAcc.conf` | Normal guarded executable, continuous safe defaults, non-overwriting default-config install |
| `loPredCtrlAcc.hpp` | Configuration, bounded-rate INDI diagnostics, pending mode requests, native read-only control permissions |
| `adc_input.hpp`, `adc_input.cpp` | Diagnostic rejection counters only; calibration and conversion unchanged |
| `native_observation.hpp`, `native_observation.cpp` | Source health, rates, bounded individual histories and full-command retention |
| `native_observer.hpp`, `native_observer.cpp` | Optional journal, continuous worker, full-command copy API and idle-safe acquisition |
| `tests/Makefile.offline`, `tests/ddspc_adc_test.cpp`, `tests/ddspc_native_observation_test.cpp`, new `tests/native_application_test.cpp` | Actual-code local transport, lifecycle, INDI and safety regression tests |
| `tests/native_linux_build.py`, `tests/native_linux_build_test.py`, new `tests/native_deployment_test.py` | Guarded Linux build profiles and non-operational install/registration checks |
| New `deployment/proclist.fragment`, new `deployment/indi-local.fragment` | Review-only registration fragments |
| New `NATIVE_APPLICATION.md`, `NATIVE_SHADOW_TESTING.md`, `EXAO2_TESTING.md`, `LIVE_INPUT_TESTING.md`, `tests/README.md` | Current deployment procedure and historical-profile clarification |

The associated engineering record is
`agents/plans/2026-11/accel-ddspc-controller.md`. No files in `apps/loPredCtrl`,
shared deployment machinery, producers or operational configuration were changed.
