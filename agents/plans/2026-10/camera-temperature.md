# Camera temperature cautions and EM-gain protection

## Request

Address [MagAOX issue #270](https://github.com/magao-x/MagAOX/issues/270): add
labrules cautions when camwfs, camsci1, or camsci2 are not on target / locked, and
add a stdCamera flag preventing EM-gain requests above 1 while temperature is not
on target. The user confirmed that reductions above 1 must also be rejected.
The user clarified that a separate configurable setpoint must be the required
setpoint for EM-gain operation, so lock at an arbitrary temperature is insufficient.

## Decisions and implementation

- Add `camera.emGainRequiresTemp`, default `false`, exposed only for cameras with
  EM-gain and temperature controls. Opt-in preserves other cameras' behavior,
  including cameraSim, which does not maintain temperature-lock state.
- Enable the flag for camwfs, camsci1, and camsci2 in the separate configuration
  repository. Use `true`, as required by the configuration boolean parser.
- Add `camera.emGainTemp`, required when the guard is enabled and independent of
  `camera.startupTemp`. Configure camwfs at -45 C, camsci1 at -50 C, and camsci2
  at -55 C, initially matching their existing startup temperatures.
- Reject missing, malformed, or nonfinite required setpoints, and reject values
  outside the effective temperature limits when that interface is supported.
- For guarded requests above 1, require the requested temperature setpoint to
  equal `camera.emGainTemp`, active cooling, temperature lock, and a READY or
  OPERATING camera. Setpoint comparison is exact; measured-temperature tolerance
  remains the camera's lock decision. Gains 0 and 1 remain subject to each derived
  camera's existing limits: OCAM permits 1; PICam permits 0 and 1.
- Reject malformed and nonfinite gain requests. A rejection leaves the accepted
  gain target unchanged, marks its INDI property Alert, logs a warning, and does
  not call the derived setter.
- Invalidate previous lock on successful temperature/controller changes and
  power transitions. Account for PICam's deferred temperature reconfiguration so
  polling the old setpoint cannot unlock the gain guard.
- Add power-gated caution rules requiring both OCAM's `ON TARGET` / PICam's
  `LOCKED` status and the required temperature target. Warm-but-locked states
  raise caution. Reuse the existing science-camera power rules. The labrules
  configuration already loads its directory, so no include-list change is needed.
- The rule configuration repeats the three required temperatures using `tol=0`.
  Keep them synchronized with each camera's `emGainTemp`; this is documented in
  the configuration rather than adding a new INDI property solely for these rules.

## Repositories and affected files

- MagAOX branch: `jrmales/camera-emgain-temperature`.
  - `libMagAOX/app/dev/stdCamera.hpp`
  - `libMagAOX/app/dev/tests/stdCamera_test.cpp`
  - `apps/picamCtrl/picamCtrl.hpp`
  - `apps/picamCtrl/tests/picamCtrl_test.cpp`
  - This record and `camera-temperature-files/config.patch`.
- Local `/opt/MagAOX/config` branch: `jrmales/camera-temperature-cautions`.
  - `camwfs.conf`, `camsci1.conf`, `camsci2.conf`
  - New `labrules/cameratemp.conf`
  - The patch records these configuration changes for review alongside the code.
    The pre-existing `dmtweeter-synco.conf` edit is unrelated and preserved.

## Offline verification

- `stdCamera_test`: 846 assertions in 11 cases, including configuration gating,
  cooling off, lost lock, unavailable camera, requests above 1 including
  reductions, safe gains, invalid numbers, INDI element precedence, unchanged
  targets on rejection, stale-lock invalidation, required-setpoint validation,
  rejection when locked at a different temperature, and independence from the
  startup temperature.
- `stdCamera_roi_test`: 529 assertions in 3 cases.
- `ocam2KCtrl_temperature_test`: 69 assertions in 3 cases.
- `picamCtrl_test`: 8762 assertions in 6 cases, including deferred targets,
  fresh-lock requirements, and request consumption before the requested
  temperature is committed to the SDK.
- Total: 10206 assertions across 23 cases.
- Production labrules parser loaded all 126 rules. Passed 264 configuration,
  power/status/setpoint, and caution-priority checks, including wrong-but-locked
  and near-target mismatches.
- Built with normal `-O3 -ffast-math`; supplied `TESTLIBS=-lgcov` because the
  existing shared test objects/libraries use coverage instrumentation.
- Ran clang-format on touched C++ files and checked both repositories for
  whitespace errors. The prescribed `//mutex scope` spelling is retained after
  formatting where clang-format would otherwise insert a space.

## Operational boundaries and follow-up

All work and tests ran on workstation `mx11`. No instrument commands, service
restarts, or hardware tests were performed. Deploy the code and corresponding
configuration together through the usual user-controlled process.

These changes guard new gain commands and report loss of temperature lock or an
incorrect setpoint. They do not automatically reduce existing gain. The startup
behavior described in the issue is not changed here; an incorrect startup
setpoint now raises caution and blocks requests for gain above 1.

The associated frameGrabber fix now atomically consumes requests at cleanup
instead of clearing them after configuration, retaining requests arriving during
worker transitions. PICam's independent `m_tempSetPtPending` remains necessary:
consuming a request is earlier than successfully committing the SDK temperature.
The implemented protocol, regressions, and remaining recovery boundaries are in
[frameGrabber-reconfiguration-race.md](frameGrabber-reconfiguration-race.md).

The subsequent optional maximum-temperature mode, including fresh-reading
tracking and a camsci1 ceiling of -40 C at the unchanged -50 C setpoint, is
recorded in [camera-max-temp.md](camera-max-temp.md). Lock-based enforcement
remains the default. Its separate companion-config patch supersedes camsci1's
original locked-status caution with the shared EM-gain temperature-readiness
result.

On 2026-10-30 the user reported that the behavior was working and requested a
push and PR description. The code branch is published as
`jrmales/camera-emgain-temperature`. The companion configuration is already
merged into `magao-x/config` branch `onsky` through `cebce35`; remote `onsky` at
`da14b03` also includes subsequent camera-display cautions and configuration
organization. The saved config patch records the original companion change.

The final warning wording is `EM gain above 1 requires on-target temperature
control at -45.000000 C`, with the temperature taken from `camera.emGainTemp`.
This wording change leaves the guard conditions unchanged. The configured
setpoint must match exactly; the camera's own status determines temperature
lock. The user's testing report does not change the offline-only scope of the
agent's validation. No GitHub issue comments or PR were created by the agent.

## Suggested PR text

Title:

```text
Guard camera EM gain by temperature and preserve reconfiguration requests
```

Description:

```md
This work was performed by GPT-6 (Codex) in response to the prompt (condensed): "Address camera temperature setpoints (#270), require a configurable setpoint for EM-gain operation, fix lost framegrabber reconfiguration requests, and add a camera-specific EM-gain-off request and GUI button (#105)."

When `camera.emGainRequiresTemp=true`, reject EM-gain requests above 1 unless the setpoint equals `camera.emGainTemp`, cooling is active, the camera is READY or OPERATING, and the selected temperature criterion is satisfied. The default `locked` mode requires on-target cooling; optional `maxTemp` mode requires a valid measured temperature at or below `camera.emGainMaxTemp` and ignores lock status. Gain 0/1 remains available. PICam requires a fresh poll after deferred temperature targets are committed.

- Add `emgain_off.request` and a shared camera GUI button. Each camera selects its safe value: OCAM, PICam, and cameraSim use gain 1; Andor maps logical gain 1 to SDK gain 0.
- Make framegrabber reconfiguration requests atomic and consume them before cleanup, retaining requests that arrive during cleanup/configuration and requeuing failed cleanup. Adjust OCAM's internal FPS/synchro application and mode locking for repeated passes.
- Companion configuration enables protection and power-gated temperature cautions for camwfs (-45 C), camsci1 (-50 C), and camsci2 (-55 C). It is already merged into `magao-x/config` branch `onsky`; this PR includes a reference patch.
- The maximum-temperature follow-up configures camsci1's ceiling to -40 C with its -50 C required setpoint. Its caution follows the new `emgain_temp_ready.toggle`, driven by the same guard predicate. This follow-up config is on `jrmales/camsci1-emgain-max-temp`; the additional patch and validation are recorded in `camera-max-temp.md`.

Validation: 14,270 assertions across 57 camera/GUI cases passed, plus worker/consumer regressions and 264 configuration/rule checks. Deterministic lost-request tests fail against the original header. The user reports the behavior working. Andor's SDK was unavailable; MCP3208's six failures and a CRED2 test abort reproduce unchanged against the original header. No instrument hardware was operated by the agent.

The temperature guard blocks new requests; it does not automatically lower existing gain. The off control is a momentary command. Documentation and formatting cleanup are separated from functional commits.

Implementation records and affected files: `agents/plans/2026-10/camera-temperature.md`, `camera-emgain-off.md`, `frameGrabber-reconfiguration-race.md`, and `camera-max-temp.md`.

Addresses #270 and #105.
```
