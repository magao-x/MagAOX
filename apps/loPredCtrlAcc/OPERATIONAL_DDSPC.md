> **Current RTC procedure:** Use [RTC_TESTING.md](RTC_TESTING.md) for the operational controller, configuration, install commands and INDI sequence. The report below describes the earlier commissioning build.

# Operational-capable DDSPC experiment

This milestone adds operator enable/freeze/reset and an opt-in real syncopation
producer. It supersedes the output prohibitions in historical shadow-only reports
**only for an explicitly operational build**. Default `make` still builds strict
`DDSPC_SHADOW_ONLY`. No installation, instrument access or native Linux link was
performed during development. The supplied configurations deliberately inhibit
actuation until the real route, calibration, synchronization association and
approved DDSPC bounds are supplied. They are not commissioned instrument settings.

## Implemented path

- `learn.toggle` starts OFF and controls both independent standard/assisted phase
  model banks. OFF freezes coefficients, fitted excitation qualification and cached
  gains; prediction can continue. A numerical job already in progress finishes
  with its captured learning intent. The INDI switch stays Busy until its status
  snapshot reflects that intent. New data must still satisfy continuity/freshness.
- `predict.toggle` starts OFF. It requests only the explicitly selected qualified
  model. Physical output does **not** silently switch the experiment to a different
  model on input loss: it latches OFF, requests own-channel zero and needs explicit
  recovery. Diagnostic shadow fallback remains available.
- `reset_model.request` resets both model banks, snapshots, excitation counts and
  origin-only score accumulators. An incarnation barrier immediately revokes old
  readiness and discards old queued scores/jobs. No old model can re-enable output.
- `ddspc_mode.standard` / `accel_assisted` require predictive/probe output OFF and
  its own zero acknowledged (or no output owned this session). Switching retains
  both independently trained models. Reset has the same output restriction.
- ADC acquisition/inference stays at individual 8-kHz samples; actual 2-kHz WFS
  targets drive numerical learning. No WFS duplication, ADC averaging, EOF,
  consumer clock synchronization or new DM scheduling algorithm was introduced.
  [PHASE_DDSPC.md](PHASE_DDSPC.md) defines the existing phase-aware regression,
  rectangular lifted command block, shared control objective and baseline limits.
- `SyncoShadowProducer::command_for()` retrieves an immutable prepared image for
  the selected model and the **actual** existing producer Request. It cannot
  substitute a diagnostic fallback of a different mode. Absolute target/cutoff,
  epoch/generation/cycle/step, input-loss barriers and readiness still apply.

Both apps use `dev::dmSyncopationProducer`, as `dmSpeckle` does. The unchanged base
selects slots, waits to its configured lead, verifies the request, updates the
existing auxiliary image and commits the payload identity in `_ready`. Its
matching `_ack` permits reuse of the one source buffer. Copy/discard is **transport
receipt**, not evidence of a physical DM write or optical application.

The shared [syncoOutput.hpp](../../libMagAOX/app/syncoOutput.hpp) is a thin
bounded enable/limits/removal wrapper around that base, not a scheduler. Each app
uses the base's configuration macros and a contained CRTP transport instance with
its own shutdown lifetime: MagAOXApp raises its shutdown flag before appShutdown,
while the final tagged own-channel zero still needs to run. The contained base is
the only publisher. Shadow compilation excludes its mapping/poll/publish thread
entirely, including sidecar and semaphore access.

An explicit ON first attaches the **existing** source and obtains the existing
cooperative `.producer` ownership lock. A tagged initial zero must be copied
before active candidates. An authorized learning request can initialize only
that own-channel zero to establish an observed initial request; it cannot enable
nonzero output. No source image is created or directly zeroed. Before this session owns a contribution,
`receipts=0` and `zero_ack=0`: startup is not presented as a zero receipt or proof
that an old process left the mirror at zero. Direct or
non-cooperating writers must be excluded by commissioning; the base's lock
protects cooperating producers, not arbitrary writers.

DDSPC absolute/per-update pixel limits and receipt timeouts remain mandatory.
Both providers enforce qualification, finite values, exact slots and discard/restart
faults. accelSim follows the base-only acknowledgement contract described below. An invalid/late candidate
inhibits rather than retagging it for a later slot. Disable/fault/shutdown submits
zero for this owner only. Removal zero bypasses the active slew limit: approval
must cover removal of the bounded contribution. Requested OFF does not falsely
report the last copied nonzero contribution removed before its zero receipt.
Shutdown waits at most max(100 ms, twice the approved receipt timeout). A missing
consumer/ack cannot guarantee removal; the app reports unconfirmed zero and
requires operator cleanup through the established DM infrastructure. A crash,
SIGKILL or stopped scheduler likewise cannot be made into proof of removal.
Never stop the scheduler first. A receipt is not a mirror-state certificate.

## Simulator limits and acknowledgement convention

accelSim uses an application-selected producer-default policy: finite values,
matching copy/discard acknowledgement and the existing ownership/deadline/zero
protocol, without mandatory `max_abs_um`, `max_step_um`, or `ack_timeout_us`.
Legacy copies of those keys are accepted and ignored. No independent steady-state
acknowledgement timer is added: an unacknowledged payload remains consumer-owned
and cannot be overwritten. Disable queues own zero after that ownership is released;
shutdown retains the existing bounded wait and explicit unconfirmed-removal result.

The shared helper defaults to the strict DDSPC policy. Only accelSim selects the
producer convention in C++; no configuration/INDI switch can select it in the
controller. loPredCtrlAcc's three requirements and finite/bound/timeout checks
remain unchanged.

Source review:

- `apps/bmcCtrl/bmcCtrl.hpp:452-509`: calibrated combined displacement is converted
  to normalized squared voltage, clipped to [0,1], square-rooted and sent to the
  SDK. This is final summed-command saturation, not a signed per-auxiliary limit
  or a comparison with the prior command. Calibration is loaded at lines619-650.
- `libMagAOX/app/dev/dm.hpp:2877-2901`: delta telemetry is relative to the flat,
  not a per-update slew limiter. Saturation monitoring at lines3072-3101 is an
  interval trip mechanism, not an instantaneous source limit.
- Existing `dmSpeckle` amplitude scaling (`apps/dmSpeckle/dmSpeckle.hpp:653-691`)
  and original `loPredCtrl` modal output (`apps/loPredCtrl/loPredCtrl.hpp:584-608`,
  `690-696`) do not implement equivalent pixel amplitude/step caps.
- The locally available syncopation branch, commit
  `bb5a5d89c594bc7949af2949f457a1520b7a3845`, uses finite checks and base poll/publish
  in `apps/dmSpeckle/dmSpeckle.hpp:519-642`, with no such three mandatory settings.
  Its `apps/bmcCtrl/bmcPipeline.hpp:199-220` also clips normalized voltage, and
  `apps/dmSyncopation/dmSyncopationProduction.hpp:173-198,218-242` validates/copies
  finite requests without per-source amplitude or step limiting.
- No local executable/configuration mapping for the name `tweeterMod` was found
  in either checked-out tree or the inspected syncopation branch. It is not
  assumed to be `ttmModulator`, which drives the separate WFS modulation function
  generator (`apps/ttmModulator/ttmModulator.hpp:304-310`). Its deployed identity
  and the two actual auxiliary routes require instrument-side confirmation.

This follows the requested simulator convention without claiming equivalent
hardware slew protection or weakening DDSPC. The physical model, ADC calibration,
upstream synchronization and scale range are unchanged.

## Two independent contributions and identification

| Provider | Required auxiliary source | Naming-convention destination |
| --- | --- | --- |
| accelSim | `dm01disp10_aux` | `dm01disp10` |
| loPredCtrlAcc | `dm10disp10_aux` | `dm10disp10` |

These names are preserved exactly. The locally inspected upstream synco example
maps `dm01disp07` through `dm01disp10`; it does **not** establish a deployed route
from `dm10disp10` to the same physical tweeter sum. Before setting either
`routing_verified`, the operator must identify the actual synco instance(s),
`[dm]` destination mapping, corresponding combiner channel and final BMC input.
Confirm the destinations independently add, neither aliases the other's source
or destination, and both are surface-command channels for the intended DM. Do not
substitute names or assume both prefixes denote the same physical path. No
operational routing was inspected or changed here.

The operational example reads **its own corrective requested image** from
`dm10disp10_aux` with `command_domain=corrective_requested`, excluding accelSim's
input. Calibrated `u=P*image` and output `image=R*u` must use the same WFS modal
basis, signs, units and ordering; `P*R` is checked algebraically. The complete
source image is retained and causally sampled at each actual opportunity.
`dm01disp` remains a valid read-only *total-request diagnostic* selection in the
unchanged acquisition defaults. It cannot authorize physical prediction in this
profile. A total DM request or the disturbance must not be relabelled as the
controller's corrective input. Neither request publication nor `_ack` is promoted
to a physically applied certificate. The declared request-domain identification
includes downstream scheduling/DM dynamics; its response delay and suitability
must be validated on the controlled experiment before relying on a fitted gain.

**Cold plant identification needs independent corrective excitation.** Zero
own-channel history cannot identify the future corrective-input block, even when
accelSim excites the disturbance. Both models remain unqualified if that block
lacks rank. The existing Gaussian `exploration_sequence.target` count,strength
convention is reused as an explicitly authorized probe, with a separate startup
`parameters.exploration_authorized` and approved modal `exploration_limit`.
Counts are actual advertised synco opportunities. Strength is in the reviewed
modal coefficient units, not automatically surface microns; each Gaussian draw
is clipped to the approved modal limit, rendered with R and additionally checked
against pixel absolute/slew bounds. No amplitude is supplied or enabled by
learning. Probing requires learning ON, prediction OFF, qualified upstream/input
history and an explicitly verified observed own-channel input. It bypasses only
*model* readiness to obtain excitation; it does not bypass telemetry/route/limits.
Learning OFF or predict OFF cancels the probe and removes its own contribution.
Normal correction cannot start until probing ends and zero is acknowledged.
This is the existing excitation mechanism, not a new identification algorithm.

No experiment must reinterpret an excitation-only shadow fit as a demonstrated
closed-loop controller. With normal AO active, its dynamics and offloads are part
of the measured disturbance/response. Compare matched origin/target/horizon/
phase/channel records and origin-only RMSE, not conditional forecasts supplied
with later command history. Existing replay/report tools remain applicable.

## Configuration files and required instrument values

The pair is in [examples/operational/](examples/operational/):
[acc000.conf](examples/operational/acc000.conf) and
[loPredCtrlAcc.conf](examples/operational/loPredCtrlAcc.conf).
The simulator copy is also [apps/accelSim/acc000.conf](../accelSim/acc000.conf).

They preserve the existing seeds, physical model, raw ADC calibration, 8000-Hz
2-channel stream and depth 128. They start disturbance OFF/scale 0 and learning,
prediction and exploration OFF. Recording is optional (`native_shadow.record`,
`online_shadow.prediction_file`), never a condition for normal acquisition.
Retain `parameters.shadow=true`, `own_shmim=false`, `accel_enabled=false`: these
select the native pipeline and inhibit obsolete Legacy/FrameGrabber output.
**Do not set shadow=false to turn on corrections.** Actual actuation is the
separately authorized existing producer path, with requested/copied/zero state
in `safety.output_reason` and predict Busy/OK/Alert. `app_health.shadow` reflects
whether a correction contribution has been transport-acknowledged, not whether
a physical mirror position has been measured.

Before editing the private experiment copies, obtain/review all of:

1. Existing `[synchro].shmimName` for accelSim's 2000-Hz reference and actual
   `syncopation.timingStream`. Timer mode still supplies ADC, but cannot qualify
   assisted output. Preserve the producer's existing delay/lock logic.
2. Actual synchronized source incarnation, WFS and first-of-four ADC counter
   anchors, actual sync reference identity and WFS-to-synco cycle relation. Use
   `[native_inputs]` fields documented in SYNCO_NATIVE/PHASE_DDSPC; zeros/empty
   values inhibit. INDI snapshots alone do not establish counter anchors.
3. Reviewed modal calibration path in the existing `DDSPC_MODAL_V1` P/R format;
   the selected `aol1_modevalWFS` modal indices/units; camera reference `camwfs`;
   actual DM geometry. No instrument calibration was invented. Default two
   indices 0/1 need confirmation, not an assumption that they are tip/tilt.
4. Approved accelSim TT cube revision and **actual matching dmmask path**. The
   supplied existing documented cube path is not a certificate of its deployed
   contents. `dm.scale` is dimensionless [0,1]; approval must relate it to the
   resulting optical motion. DDSPC pixel bounds are separately approved. The physical coupling/model is
   unchanged; this milestone adds no new WFS calibration.
5. Approved controller `[output]` `max_abs_um`, `max_step_um`, and transport
   `ack_timeout_us`, plus exclusive ownership/routing checks for both apps.
   These three settings are required only for loPredCtrlAcc; zero inhibits DDSPC.
   accelSim no longer requires or uses them. No generic safe amplitude is implied.
6. Explicit `[output] authorized=true`, `routing_verified=true`,
   `correction_input_verified=true` for the controller, and `[dm]
   output_authorized=true`, `routing_verified=true` for accelSim, **only after**
   those checks. Optional probe authorization/limits need separate approval.

A source replacement or invalid upstream contract must not silently inherit old
anchors. Numerical budget faults remain latched until worker restart; a model
reset does not bypass them. The minimum cadence and timing requirements already
implemented in PhaseShadow remain checked; no exposure reconstruction or
cross-host mapping was added as a prerequisite for a verified upstream contract.

## Build, installation and registration (operator only)

The following are commands for a provisioned Linux MagAO-X development/deployment
window. None were executed on the instrument. First run all local tests below.
Default strict shadow builds:

```bash
cd ~/source/MagAOX-DDSPC/apps/accelSim
make -j4
cd ../loPredCtrlAcc
make -j4
```

One-time explicitly operational-capable builds, still disabled by configuration
and startup INDI intent:

```bash
cd ~/source/MagAOX-DDSPC/apps/accelSim
make -j4 operational
cd ../loPredCtrlAcc
make -j4 operational
```

Use the same profile on install; plain `make install` selects strict shadow and
may rebuild it. Back up existing binaries/configs under the site's normal
maintenance procedure first. No install here is authorized automatically:

```bash
cd ~/source/MagAOX-DDSPC/apps/accelSim
sudo make DDSPC_OPERATIONAL_BUILD=yes install
cd ../loPredCtrlAcc
sudo make DDSPC_OPERATIONAL_BUILD=yes install
```

The standard MagAO-X installer uses root-owned mode 4755 binaries. Both config
hooks preserve existing configs. Review the completed experiment copies, then
install them deliberately as `/opt/MagAOX/config/acc000.conf` and
`/opt/MagAOX/config/loPredCtrlAcc.conf` during that authorized window; do not
replace production calibration files. For explicit configuration installation after completing the two reviewed copies:

```bash
# These source paths must contain the operator-reviewed values listed above.
sudo install -m 0644 ~/source/MagAOX-DDSPC/apps/loPredCtrlAcc/examples/operational/acc000.conf /opt/MagAOX/config/acc000.conf
sudo install -m 0644 ~/source/MagAOX-DDSPC/apps/loPredCtrlAcc/examples/operational/loPredCtrlAcc.conf /opt/MagAOX/config/loPredCtrlAcc.conf
```

`make install` does **not** register xctrl
or reload INDI. Merge only these entries into the existing
`/opt/MagAOX/config/proclist_${MAGAOX_ROLE}.txt` if absent:

```text
acc000 accelSim
loPredCtrlAcc loPredCtrlAcc
```

Add only these instance names to the existing role broker's `[local] drivers`
list if absent. See [NATIVE_APPLICATION.md](NATIVE_APPLICATION.md#xctrl-and-indi-registration)
for exact broker/driver discovery. A broker change needs its own approved
maintenance window. **This xctrl implementation's `restart` also restarts the
role broker.** With registration already present, use app-only stop/start in the
approved window rather than surprising other observers with a broker restart:

```bash
xctrl start acc000
xctrl start loPredCtrlAcc
xctrl status acc000
xctrl status loPredCtrlAcc
cursesINDI
```

Do not start a second `acc000` producer or a second owner of either aux source.
For direct private startup, the existing executable interface is
`./accelSim -n acc000 -c /absolute/reviewed/acc000.conf` and
`./loPredCtrlAcc -n loPredCtrlAcc -c /absolute/reviewed/loPredCtrlAcc.conf`.
Private directories/identities must follow the existing native observation
procedure; the operational names here are **not** an isolated test namespace.

## Authorized internal-source experiment commands

Run these **only by the operator after the internal source, routing, limits,
upstream lock and readiness checks**. Host/port below mean the already approved
local INDI endpoint; no tunnel or connection was opened during development.
Using the supplied simulator instance name `acc000`:

```bash
getINDI 'acc000.dm_status.*' 'acc000.timingDiag.*'
getINDI 'loPredCtrlAcc.input_health.*' 'loPredCtrlAcc.input_rates.*'
getINDI 'loPredCtrlAcc.ddspc_control.*' 'loPredCtrlAcc.safety.*'
```

Require ADC ~8000 Hz, WFS/camera ~2000 Hz, no new acquisition loss, valid upstream
lock, independently owned/routed aux streams and zero correction intent. Then:

```bash
setINDI -s 'acc000.dm_enable.toggle=On'
# Set this shell variable to the separately approved dimensionless amplitude; no value is supplied here.
setINDI -n "acc000.dm_scale.target=${APPROVED_DISTURBANCE_SCALE:?approved scale required}"
```

Verify `dm_status.requested=1`, `active=1`, no transport fault, and verify the
optical WFS peaks using the already validated disturbance procedure. Record a
60-s disturbance-only baseline with the same instrument/loop state. Next:

```bash
setINDI -s 'loPredCtrlAcc.learn.toggle=On'
```

Wait for both models' rank/readiness and meaningful matched forecast counts.
If own corrective history is unexcited, **stop here** for the separately approved
probe; do not bypass rank. Optional existing probe command, only with its
explicit config approval and approved count,strength pairs supplied:

```bash
setINDI -x "loPredCtrlAcc.exploration_sequence.target=${APPROVED_EXPLORATION_SEQUENCE:?approved count,strength sequence required}"
```

For a cold probe, watch both models while the approved excitation is still
active. Once both qualify, turn learning OFF: this freezes the fit and cancels
and removes the probe. Wait for its own zero receipt before correction. Do not
wait indefinitely while learning from a subsequently unexcited input: the
existing sliding excitation check can revoke readiness. For a comparison with
fixed coefficients, wait for the learning switch to leave Busy, then use the
same disturbance settings and evaluation duration for both modes:

```bash
setINDI -s 'loPredCtrlAcc.learn.toggle=Off'
setINDI -s 'loPredCtrlAcc.ddspc_mode.standard=On;accel_assisted=Off'
setINDI -s 'loPredCtrlAcc.predict.toggle=On'
getINDI 'loPredCtrlAcc.safety.*' 'loPredCtrlAcc.ddspc_control.*'
# Record standard case; then remove only the correction.
setINDI -s 'loPredCtrlAcc.predict.toggle=Off'
getINDI 'loPredCtrlAcc.safety.*'
# Wait for requested=0, transport_ack=0, zero_ack=1, without a transport fault.
setINDI -s 'loPredCtrlAcc.ddspc_mode.standard=Off;accel_assisted=On'
setINDI -s 'loPredCtrlAcc.predict.toggle=On'
# Record assisted case with identical disturbance and targets/horizons.
setINDI -s 'loPredCtrlAcc.predict.toggle=Off'
setINDI -s 'loPredCtrlAcc.learn.toggle=Off'
# Wait for controller zero receipt. Remove only the disturbance:
setINDI -n 'acc000.dm_scale.target=0'
setINDI -s 'acc000.dm_enable.toggle=Off'
getINDI 'acc000.dm_status.*' 'loPredCtrlAcc.safety.*'
```

Both shadow models score identical contemporaneous WFS targets. Sequential
physical correction intervals do not contain identical stochastic disturbance
samples: accelSim keeps running. Match settings, durations and steady loop state,
repeat/reorder runs to assess uncertainty, and exclude probe/warmup intervals.
No recompile/restart/config change is needed between model selections. If a
fault occurs, leave OFF, diagnose telemetry/slots, verify own zero removal, then
explicitly reset/retrain the controller before a new prediction request. Model
reset is `setINDI -s 'loPredCtrlAcc.reset_model.request=On'` only while OFF/removed.
Do not use it to bypass missing evidence. After both zeros are acknowledged,
stop only the experiment apps with `xctrl stop loPredCtrlAcc` and, if this is the
experiment's simulator, `xctrl stop acc000`; keep the scheduler available until
removal is confirmed. Rollback restores the saved binaries/configs only in the
approved maintenance window, with both contributions verified removed first.

Record ACC, selected calibrated WFS modes, camera identity, own requested input,
disturbance request, actual combined DM stream and synco timing/commit/ack metadata
with existing recorder tools when quantitative comparison is wanted. Optional
origin-only prediction CSV must have a private absolute path and capacity/row
budget. Compare tip/tilt separately: residual variance/PSD at the actual 28–29 Hz
and 56 Hz peaks, matched prediction RMSE/error PSD, command RMS, qualified coverage,
learning counts, sample loss, worker/fast-path timing, missed/expired slots and
transport receipt failures. Do not equate private callback timing to real DM
latency or interpret prediction-error PSD as closed-loop optical suppression.

## Offline verification

Run from the repository root; these launch test executables with API doubles,
not MagAO-X applications, shmims, INDI servers or hardware:

```bash
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 BUILD_DIR=build/operational-check check
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 BUILD_DIR=build/operational-check check-multirate check-native-observation check-confirmed-replay check-shadow-only check-native-timing
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 BUILD_DIR=build/operational-check check-operational-output check-operational-controls
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 BUILD_DIR=build/operational-asan CXXFLAGS='-O1 -g -std=c++20 -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer' check
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 BUILD_DIR=build/operational-asan CXXFLAGS='-O1 -g -std=c++20 -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer' check-operational-output check-operational-controls
```

`check-operational-output` compiles the **unchanged real producer base**, actual
RLS/QR and simulator numerical code, with private ordinary-file ImageStreamIO API
doubles and an exact tagged mock consumer. It covers both selected models,
independent contributions, startup/removal zero, invalid/late/throwing/discarded
candidates and qualification loss. This proves protocol integration, not Linux
ImageStreamIO ABI, a deployed combiner route, scheduler jitter or physical DM
response. The installed controller header/INDI callbacks are tested separately
in both build profiles with stream-free lifecycle doubles. Native accelSim
application integration still needs real Linux compilation/lifecycle testing.

Results and observed development-host timings are recorded in the associated
[engineering record](../../agents/plans/2026-11/ddspc-operational-output.md).
Physical output remains **uncommissioned** until a Linux build, actual two-route
inspection, approved limits/calibration/anchors, producer callback timing and
controlled zero/enable/fault/removal tests succeed on the authorized setup.
