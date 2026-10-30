# Optional maximum-temperature EM-gain enforcement

## Request and selected configuration

The user reported that camsci1's unstable temperature lock makes lock-based
EM-gain permission impractical in some operating conditions, and approved an
optional maximum-temperature mode that ignores lock status. The user selected
**-40 C** as the maximum; the required camsci1 setpoint remains **-50 C**.

```ini
[camera]
emGainRequiresTemp=true
emGainTemp=-50
emGainTempMode=maxTemp
emGainMaxTemp=-40
```

The existing code feature branch is `jrmales/camera-emgain-temperature`.
Companion config is on `jrmales/camsci1-emgain-max-temp` in `/opt/MagAOX/config`,
based on `onsky` at `da14b03`, with the configuration recorded in commit
`f3ac89f`. On 2026-10-30 the user requested committing and publishing both
feature branches. No hardware or service actions were performed.

## Guard semantics

- `camera.emGainRequiresTemp` remains the master enable, default false.
- `camera.emGainTempMode` defaults to `locked`, preserving existing behavior.
- In `maxTemp` mode, permission requires a valid measured temperature less than
  or equal to `camera.emGainMaxTemp`; the lock flag is ignored.
- Both enabled modes require the setpoint to equal `camera.emGainTemp`, active
  cooling, and READY/OPERATING state. Faulted/unknown/off cooling remains blocked.
- Mode names are validated. Explicit numeric settings must parse completely and
  be finite; enabled maximum mode requires a ceiling no colder than the required
  setpoint. The required setpoint still obeys the effective camera target limits.
- Gain 0/1 and the camera-specific off request bypass the temperature requirement.
- The guard still governs new requests only. It does not automatically lower
  existing gain when conditions change.

## Reading validity and publication

`m_ccdTempValid` is independent of the camera's lock status. The initial `-999`
sentinel is invalid. `updateTempReading()` accepts finite values at or above
absolute zero after a complete successful camera poll; the guard repeats these
numeric checks rather than trusting the validity bit alone.

Power, reconnect, target/controller changes, configuration, and failed/incomplete
polling invalidate the cached reading. PICam additionally requires a pending
target to have committed successfully before a subsequent poll can restore
validity. Generic framegrabber request consumption is not proof of target
application, so `m_tempSetPtPending` remains independent of `m_reconfig`.

The four EM-gain controllers maintain validity under their INDI mutex. Shared
power-on defaults/temperature initialization now hold that mutex for these
controllers. Poll-entry invalidation is internal; readiness is published after
complete polling or durable invalidation, avoiding an Off/On notification on
every successful poll.

The read-only `emgain_temp_ready.toggle` uses the same predicate as the guard.
It reports permission from the temperature interlock, including permission when
the interlock is disabled; it is not a promise that other camera-specific gain
restrictions will accept the command. Power-off clears the published switch.
Raw `LOCKED`/`UNLOCKED` reporting is unchanged.

Validity means a complete poll since the last invalidating transition. This
change does not introduce a reading-age timeout or an INDI transport heartbeat.

## Companion caution

camsci1's existing `camsci1-on-temp-not-ready` caution retains its name so existing
rtimv configuration continues to reference it. It now evaluates camera power
AND readiness not On. This follows pending-target/validity state as well as the
temperature and setpoint criteria, without duplicating them in separate rules.
Missing readiness data follows the rule engine's invalid-rule handling; it is
not treated as a healthy value. camwfs and camsci2 retain their lock-based rules.

The exact configuration patch is in
[camera-max-temp-files/config.patch](camera-max-temp-files/config.patch).
It was checked and applied to the local config feature branch after offline
validation; no service reload was performed. The saved patch uses zero context
(`git apply --unidiff-zero`) to avoid whitespace-only context lines in the record.

## Affected files

- `libMagAOX/app/dev/stdCamera.hpp` and `tests/stdCamera_test.cpp` beneath it.
- `apps/{ocam2KCtrl,picamCtrl,andorCtrl,cameraSim}/<app>.hpp` and focused camera tests.
- This record and `camera-max-temp-files/config.patch`.
- Companion repository: `camsci1.conf` and `labrules/cameratemp.conf`.

## Offline validation

- Shared stdCamera: 5283 assertions across 22 cases passed, including inclusive
  thresholds, mode/config validation, invalid/stale/nonfinite readings, lifecycle
  transitions, safe requests, readiness-property agreement with enforcement,
  and nonblocking readiness refresh when the INDI mutex is busy.
- stdCamera ROI without EM-gain support: 529 assertions across 3 cases passed.
- PICam: 13868 assertions across 8 cases passed, including target-commit/readback
  sequencing, read failures, unlocked permission, and readiness notifications
  without transient Off/On flicker on successful polls.
- OCAM temperature: 165 assertions across 4 cases; existing OCAM regression:
  474 assertions across 22 cases. Both passed.
- cameraSim EM gain: 102 assertions across 2 cases; existing simulator regression:
  107 assertions across 4 cases. Both passed.
- Total: 20528 assertions across 65 cases. Builds use the normal
  `-O3 -ffast-math` settings with `TESTLIBS=-lgcov` for existing instrumented
  dependencies. All runs were offline on workstation `mx11`.
- Companion configuration: all 124 rules loaded; 215 configuration and rule
  checks passed, including boundary/readiness routing and unchanged lock-based
  cameras. Actual config files match the validated shadow files byte for byte.
- Independent review found and resolved an Andor startup setter lock gap; the
  added INDI lock begins after its existing startup wait. No remaining scoped
  blockers were found. Formatting, changed-file documentation, and whitespace
  checks passed.
- Andor's SDK remains unavailable; its controller changes require source review.

## Suggested PR follow-up text

Title:

```text
Add optional maximum-temperature enforcement for camera EM gain
```

Description:

```md
This work was performed by GPT-6 (Codex) in response to the prompt: "camsci1's temp control is unstable ... we need to have an option to make it accept a range, i.e. a max temp value, and when in this mode ignore locked status", followed by approval to implement and "use -40".

Add camera.emGainTempMode=maxTemp as an alternative to the default locked mode. Require a valid measured temperature at or below camera.emGainMaxTemp while retaining the required setpoint, active cooling, and READY/OPERATING checks. Invalidate cached readings across lifecycle/configuration changes, and keep PICam blocked until its requested target is committed and freshly sampled.

Publish the same permission through emgain_temp_ready.toggle so camsci1's caution agrees with enforcement. Companion configuration selects a -40 C ceiling and preserves the -50 C required setpoint. Gain 0/1 and the off control remain available.

Details, affected files, and offline validation are recorded in agents/plans/2026-10/camera-max-temp.md. No instrument hardware was operated by the agent.
```
