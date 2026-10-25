# ocam2KCtrl housekeeping policy

Created: 2026-10-25 (America/Phoenix).
Status: implemented locally; hardware qualification pending. The operator approved
this plan on 2026-10-25. No instrument configuration has been changed by an agent.

## Request and intended behavior

The operator requests confirmation-driven gain/FPS handling and slower,
configurable temperature polling, with faster polling while a change is in
progress. Interpret the requested relaxed temperature rate as **one query every
10 seconds (0.1 Hz)**.

The resulting steady-state behavior should be:

| Quantity | After startup or a relevant change | Once confirmed/settled |
| --- | --- | --- |
| EM gain | Accept a verified setter acknowledgement, or query until confirmed within a bounded interval. | Retain the confirmed value; no automatic gain queries. |
| Internal FPS, synchronization off | Confirm the final effective mode/rate using a verified acknowledgement or bounded readback. | Retain the confirmed value; no automatic FPS queries. |
| External FPS, synchronization on | Use the existing sync-device subscription to confirm its requested frequency. | Continue receiving external frequency updates; no serial FPS queries. |
| Temperature/cooling | Query promptly, then at the configured changing interval while command state or thermal convergence is pending. | Query at the configured idle interval, default 10 s. |

Gain/FPS age continues to increase after confirmation. That is intentional:
the value is a cached confirmation, rather than a continuously observed device
state. The policy must make this distinction visible.

## Evidence and current code

The [acquisition/publication investigation](ocam2k-acquisition-publication-timing.md)
and [debugging record](dm-syncopation-debugging.md) contain the measured basis.
The unchanged closed-loop normal / paused / normal comparison retained `atime`
ranges of **481–525 / 499–501 / 480–515 us**. Prior driver traces identify
device/serial work before frame completion stamping/wakeup in the shared EDT
handler, and the same driver lock in the interrupt and acquisition paths.
The paused capture also has separate post-SDK conversion variation; reducing
serial traffic does not resolve every source of publication jitter.

Relevant implementation paths are:

- [ocam2KCtrl.hpp](../../../apps/ocam2KCtrl/ocam2KCtrl.hpp): `pollHousekeeping()`
  currently calls temperature, gain and applicable FPS getters each ordinary
  loop pass. The 120-second diagnostic pause, monotonic clock, serial-attempt
  counter, readback ages and control mutex already exist here.
- `setEMGain()`, `setFPS()`, `setSynchro()` and temperature setters currently
  accept transport success while leaving response verification as TODOs.
  A successful serial exchange alone is not value confirmation.
- [ocamUtils.hpp](../../../apps/ocam2KCtrl/ocamUtils.hpp) provides `parseTemps()`,
  `parseEMGain()` and `parseFPS()`. Temperature/FPS getters can return zero
  after a malformed reply; successful parsing/validity must be checked explicitly.
- [stdCamera.hpp](../../../libMagAOX/app/dev/stdCamera.hpp) owns target/current
  properties and common callbacks. Its ordinary `updateINDI()` publishes cached
  gain/FPS states as IDLE; OCAM confirmation status must survive that update.
- [edtCamera.hpp](../../../libMagAOX/app/dev/edtCamera.hpp) already records
  completed serial transaction timing. Reuse that path and its existing
  timeout/response-drain behavior.

Keep the implementation app-local and header-only, matching this application.
Use existing base classes, parsers and timing facilities. Similar deadline-based
polling exists elsewhere, but this change does not justify a new common base
class. Shared camera-base behavior should remain stable for other applications.

## Confirmation rules

Keep three concepts separate: command transport, confirmed device state, and
physical temperature settling.

```text
accepted new request -> send setting once
                           |
                 matching parsed acknowledgement?
                    yes /             \ no usable value
                 confirmed        pending readback
                                      |
                         matching reply -> confirmed
                         deadline expires -> unconfirmed/error

confirmed gain/FPS -> no further query until a new request or invalidation
confirmed temperature setting -> fast reads until thermal settling -> idle reads
```

1. Validate the requested command before marking a change pending. Preserve the
   existing gain limits and protection-reset gate. Match gain against the actual
   validated integer dispatched to the camera, including the existing handling
   of fractional targets; avoid an incidental change to that API.
2. Snapshot the expected value and a request/configuration generation. A later
   request supersedes an earlier one. Replies from an older generation cannot
   confirm a newer request or a setting subsequently overwritten by mode setup.
   Keep the last confirmed current value separate from the requested target.
3. Parse and validate the setter reply. If it explicitly confirms the dispatched
   value, update the confirmed value and timestamp immediately, without adding
   a redundant query. Otherwise schedule a readback on the ordinary control
   path. Reuse existing parsers where their response contract applies.
4. Do not infer acknowledgement formats from arbitrary test-stub strings.
   Establish the actual gain/FPS setter response formats and FPS rounding rules
   from available protocol material or operator-supplied replies. A bounded
   getter fallback permits correct operation when a setter reply carries no
   usable value. Reject malformed, nonfinite, out-of-range and mismatched data.
5. Pending readbacks use a fixed monotonic confirmation deadline. Failed or
   mismatched replies do not reset it. Readback retries never resend a setter.
   At expiry, publish an explicit unconfirmed/error state and stop autonomous
   retries for that request. A new explicit request or lifecycle recovery can
   start a new confirmation. Preserve existing transport-error recovery; do not
   add a new reconnect/reconfiguration loop just for a value mismatch.
6. Once confirmed, gain and internal FPS have no periodic refresh deadline.
   One successful startup/recovery read can establish an initial current value
   without a requested target; do not treat a default target as an instruction
   to overwrite the device.

FPS requires two additional rules:

- **FPS zero is the existing automatic/maximum-rate command**, not a measured
  zero-Hz target. Record confirmed automatic mode with a finite positive actual
  rate from the response/readback. Confirm only the final effective rate after
  mode and synchronization setup, including its intermediate `fps 0` writes.
  For explicit numeric rates, compare using established firmware precision;
  choosing that precision is part of the response-contract work below.
- With synchronization on, preserve the existing external-frequency callback,
  `m_fps`/`m_workerFPS` updates and required acquisition reconfiguration. A fresh
  source report corresponding to the latest request confirms the external
  setting; a previously cached matching frequency alone does not acknowledge a
  new command. Finish confirmation in the final mode/synchronization generation.
  This confirms the external timing source; camera acquisition statistics remain
  the independent measurement of delivered frame rate.

INDI reports carry no request ID here. A matching report received after dispatch
is observation-based confirmation, not proof that the report was generated in
response to that specific request. Qualify same-value external FPS requests:
the source may publish only on change. Reuse an already confirmed unchanged
setting only while its source/configuration validity still holds; otherwise
obtain a one-shot fresh source report instead of waiting solely for a changed
value. Do not introduce periodic resubscriptions to obtain confirmation.

When an `HV` trip response is actually received, retain its existing recognition
and error handling. Removing background gain queries means software will no
longer discover a silent change through routine gain polling. The operator has
requested this policy and reports current trip detection is usually silent.
Expose **confirmation-only** protection polling rather than claiming continuous
monitoring. Preserve reset controls, their existing two-request confirmation
workflow, request expiry and gain-setting interlocks.

## Adaptive temperature polling

Start with these proposed configuration settings, all in seconds except the
sample count:

```ini
[housekeeping]
tempIntervalSec=10
tempChangingIntervalSec=1
confirmationIntervalSec=1
confirmationTimeoutSec=10
tempSettleSamples=2
```

| Setting | Purpose and validation |
| --- | --- |
| `tempIntervalSec` | Idle temperature query interval; finite and positive. |
| `tempChangingIntervalSec` | Faster query interval during change/convergence; finite, positive and no greater than the idle interval. |
| `confirmationIntervalSec` | Interval between pending gain/FPS/temperature-state readbacks; finite and positive. |
| `confirmationTimeoutSec` | Deadline for acknowledging a dispatched setting or establishing a startup value; finite, positive and at least the confirmation interval. |
| `tempSettleSamples` | Number of consecutive fresh settled readings before returning to idle polling; positive integer, proposed default two. |

These are nominal due intervals, serviced by the ordinary app loop. Its current
default sleep is one second, and serial transactions add time. Do not promise
hard one-second sampling or shorten the global loop sleep as part of this change.
The confirmation deadline is checked before and after an attempt; it cannot
preempt an SDK call already waiting under the existing transport timeout.

Temperature state transitions should be explicit:

- On connection, unknown temperature, an accepted setpoint/cooling command,
  or a valid idle read showing enabled cooling off target, enter fast polling
  and make the next temperature read due promptly.
- Confirm the **reported setpoint/control state** separately from the CCD
  temperature reaching that setpoint. Preserve the newest requested target
  separately: `getTemps()` currently copies returned SET into `m_ccdTempSetpt`,
  which must not erase a pending request or make it self-confirming.
- Reuse the existing `abs(CCD - SET) < 1 C` thermal criterion. Return to idle
  after the configured number of consecutive fresh, valid on-target readings,
  with the requested setting confirmed. A malformed, failed or off-target
  read resets the settling count.
- Cooling can legitimately take minutes. Expiry of the setting-acknowledgement
  deadline does not impose a ten-second physical cooling deadline. Once the
  setting is confirmed, continue fast convergence monitoring while off target.
- For cooling disabled, confirm the disabled state and then return to idle;
  do not remain in fast polling merely because CCD differs from an inactive
  stored SET. Use only state evidence actually available in OCAM replies.
  Validate the existing cooling-power/state inference during implementation.
- A failed temperature transaction retains existing transport/error handling.
  A malformed nonfatal reply leaves temperature unavailable and schedules the
  next bounded fast attempt; do not mark it current or increment settling.

If temperature state remains unknown or a setting acknowledgement times out,
end that request's fast confirmation attempts, expose the error and retain
slow temperature monitoring at the idle interval. A later valid measurement
can restore observation validity, but cannot silently acknowledge an expired
request. A fresh explicit request or lifecycle recovery can begin confirmation
again. This differs from an already confirmed setpoint whose CCD is still
converging: that physical transition continues fast monitoring without an
artificial ten-second thermal deadline.

With ten-second caching, the existing cooling-Off interlock needs a broader
freshness rule: **always obtain a fresh valid temperature before applying its
existing CCD > 19 C test**, independent of whether the diagnostic pause was used.
Keep query, decision and command serialized under the camera mutex, preserve
the operator's requested Off state across the read, and reject an invalid
refresh. If an unexpected reported SET/control change is discovered, expose it
and reevaluate thermal monitoring without automatically reissuing an old command.

## Scheduling, lifecycle and the diagnostic pause

Use `housekeepingNow()` and monotonic per-operation due times in the ordinary
control path, with existing lock order **INDI mutex then camera mutex**.
Keep frame acquisition, descrambling, notification and rearm paths unchanged.

Issue at most one automatically scheduled serial query per app-loop pass.
Choose the oldest due operation, with confirmation before temperature on an
initial tie. Reschedule from actual service/completion rather than replaying
missed periods. Check pending deadlines even when another operation is selected;
test fairness when gain, FPS and temperature are all due. Existing startup or
explicit setting sequences may still require their existing ordered commands.
Due operations share these service opportunities, so each pending quantity's
effective interval can exceed its configured interval. Allow for that service
time when choosing the confirmation timeout; one-second configuration does not
promise simultaneous one-Hz polling of all quantities.

Power loss, reconnect, acquisition recovery and mode/synchronization changes
invalidate affected observations and pending generations. After final successful
setup, establish unknown gain/internal FPS once, then become quiet. Never
confirm an intermediate value that subsequent setup overwrites, and never
reapply gain automatically after a reset or trip. Continue nonserial external
frequency updates and telemetry/property servicing on every ordinary pass.

Retain the 120-second housekeeping pause for comparisons. Refuse a new pause
while a setting confirmation or thermal change is pending; this avoids hiding
unfinished work or expiring its confirmation while deliberately suppressing it.
An explicit setting command still ends an active pause before serial I/O.
On resume/expiry, schedule one temperature refresh; confirmed gain/FPS stay
quiet, and no missed polls accumulate. Preserve existing lifecycle cancellation
and expose the actual quiet interval boundaries.

Reducing traffic decreases exposure to the measured interference. It does not
guarantee deconfliction with frame completion: camera replies and EDT device
interrupts arrive asynchronously after the request. Phase scheduling, changing
EDT serial timeouts/draining, and investigation of the separate conversion
variation are separate work.

## INDI and telemetry behavior

Preserve existing standard property names, target/current values, timing
telemetry and pause controls. Extend the app's housekeeping diagnostics to show:

- Gain/internal FPS states such as `unavailable`, `confirming`, `confirmed`
  and `unconfirmed/error`, with ages defined as time since the last verified
  reply. An intentionally cached confirmation does not become stale solely
  because no background poll is scheduled.
- Temperature phase `idle`, `changing`, `paused` or `inactive`, its effective
  interval/next-due time, measured age and validity. Between scheduled successful
  reads the value remains the latest sample; an overdue/failed read becomes
  stale/unavailable using a documented grace of one ordinary loop opportunity
  plus the existing transaction timeout.
- Pending confirmation deadline/result, query counters by type, and explicit
  `protectionPolling=confirmation only` outside a diagnostic pause. Preserve
  `paused`/`inactive` where appropriate.

Publish these app-specific statuses after `stdCamera::updateINDI()` so its
cached property updates cannot erase pending/error state. Prefer this local
integration to changing shared base behavior. Reuse existing serial timing and
camera telemetry; no new binary log schema is required for the initial change.
Telemetry emission must continue when no serial operation is due, and recording
a cached value must not renew its measurement/confirmation timestamp.

## Implementation sequence and affected files

1. **Establish reply contracts.** Review real/protocol-supported setter replies;
   add focused strict parser tests for gain, FPS, setpoint/control confirmation
   and error replies. Resolve explicit-FPS rounding and automatic-rate semantics.
   Reuse the existing query fallback wherever acknowledgement lacks a value.
2. **Add policy state/configuration.** Introduce monotonic due times and small
   per-setting pending/confirmed state in `ocam2KCtrl.hpp`. Validate settings at
   load time. Reuse one small app-local confirmation representation for gain
   and FPS if that reduces duplication; avoid a general command framework.
3. **Integrate commands and lifecycle.** Parse acknowledgements, retain request
   generations, establish startup observations once, and handle final mode/sync
   state. Preserve failed-request targets/current values according to existing
   callback contracts; report unconfirmed values without pretending success.
4. **Replace continuous housekeeping.** Select due work, implement temperature
   settling and fresh cooling-Off validation, and adapt the diagnostic pause,
   readback validity and confirmation-only protection status.
5. **Verify locally.** Extend the current SDK-free OCAM/parser/housekeeping,
   temperature and lifecycle suites; build against the copied EDT SDK without
   launching the camera binary. Apply the repository's formatting and full-file
   documentation requirements to touched C++ files.
6. **Document and qualify.** Update the operator guide/configuration reference,
   keep this plan and the engineering records current, then have the operator
   compare identical closed-loop settings before/after and exercise controls.

Expected implementation files:

- [ocam2KCtrl.hpp](../../../apps/ocam2KCtrl/ocam2KCtrl.hpp).
- [ocamUtils.hpp](../../../apps/ocam2KCtrl/ocamUtils.hpp) only where reply parsing
  needs extension, with [ocamUtils_test.cpp](../../../apps/ocam2KCtrl/tests/ocamUtils_test.cpp).
- Existing [OCAM tests](../../../apps/ocam2KCtrl/tests/ocam2KCtrl_test.cpp),
  [housekeeping tests](../../../apps/ocam2KCtrl/tests/ocam2KCtrl_housekeeping_test.cpp),
  [temperature tests](../../../apps/ocam2KCtrl/tests/ocam2KCtrl_temperature_test.cpp)
  and [lifecycle tests](../../../apps/ocam2KCtrl/tests/ocam2KCtrl_lifecycle_test.cpp).
- [camera-timing-trace.md](../../../apps/ocam2KCtrl/doc/camera-timing-trace.md),
  this plan, and the linked engineering records. Add test registration only if
  another test binary is actually needed; prefer extending existing suites.

## Acceptance checks

- After initial confirmation and thermal settling, a simulated minute issues
  **zero gain/FPS queries** and temperature reads at the configured ten-second
  cadence, within ordinary-loop/transport service granularity.
- A matching valid setter reply confirms without an extra query. A missing
  value, mismatch or invalid reply starts/retains bounded readback confirmation;
  expiry stops retries and exposes failure. No automatic repeated setting writes.
- Newer requests, mode resets, synchronization changes and reconnects cannot
  consume obsolete confirmations. Cover zero/max FPS and actual integer gain.
- Synchronized FPS follows fresh external-source reports without serial queries;
  existing `m_workerFPS` publication and acquisition reconfiguration still work.
- Temperature enters fast polling on changes/off-target state, requires fresh
  settling samples, returns to idle, and handles disabled cooling without
  permanent fast polling. Reported SET cannot overwrite a newer pending target.
- Cooling-Off validation uses a fresh reading even without a diagnostic pause;
  malformed/failed reads cannot bypass the existing temperature limit.
- No-due-work, mismatch, failure and pause paths retain INDI/telemetry progress,
  reset-confirmation expiry, correct ages and no catch-up query bursts. Use a
  controllable monotonic time in tests rather than real ten-second sleeps.
- Operator qualification verifies steady serial traffic is temperature-only,
  command changes confirm and then stop gain/FPS querying, and temperature
  transitions increase polling. Compare timing tails and RMS separately; ten-
  second temperature reads can still cause occasional serial-associated tails.

## Proposed defaults and remaining qualification

The defaults above are reviewable proposals: changing temperature interval
**1 s**, idle interval **10 s**, two settled samples and a **10 s** confirmation
deadline. The operator explicitly requested configurable slow temperature
polling and no steady gain/FPS queries; this plan does not add a periodic
gain/FPS watchdog.

Implementation must establish the actual setter acknowledgement/rounding and
cooling-state evidence. Until then, strict query-based confirmation is the
fallback; transport success alone is never sufficient. All agent development
and SDK-free tests remain on the workstation. The operator performs every
instrument control and hardware qualification step.

## Implementation record — 2026-10-25

Implemented the app-local policy in `ocam2KCtrl.hpp`, with strict response parsing
in `ocamUtils.hpp`. The configuration defaults above are now the defaults in
code. Setting writes are sent once. Gain/internal FPS use a valid matching
numeric report in the existing query grammar when one is present in a setter
reply; other replies schedule bounded getter confirmation. The local material
does not establish additional setter-specific acknowledgement grammar, so no
such grammar is assumed. Internal zero/max FPS accepts a finite positive actual
rate. Explicit FPS comparison permits only four float epsilons times the rate
(with a one-Hz floor), to accommodate numeric representation, rather than an
unverified firmware rounding rule. Rounded readbacks outside that tolerance
remain unconfirmed for operator/protocol qualification.

The camera mutex now covers each complete getter exchange and its observation
update, so an older serial read cannot interleave with a superseding setter.
Each new request/invalidation advances its confirmation generation. The final
mode/synchronization write owns FPS confirmation; intermediate maximum-rate
writes cannot establish the final rate. Existing configuration handling still
logs synchronization-setup errors and continues, but the FPS confirmation
remains failed rather than silently accepting that intermediate rate.

External frequency confirmation runs even on an unchanged reported frequency.
A valid, already confirmed unchanged source setting can be reused; otherwise
one `getProperties` request asks for a fresh source observation for that request
generation. This is not a periodic resubscription. Dispatch failure is reported
as failed confirmation. As with the plan, INDI supplies no request ID, so fresh
reports provide observation-based confirmation.

Ordinary polling selects at most one serial query per pass, oldest due first,
with gain/FPS confirmation winning initial ties. Completion time determines the
next opportunity. Deadlines are checked before and after service; SDK waits
remain non-preemptible under the unchanged transport timeouts. A virtual
ordinary-thread clock permits deterministic deadline tests without waiting ten
seconds. There are no new normal frame-acquisition, descrambling, publication
or rearm operations. Acquisition recovery invalidates observations; regular
mode reload invalidates the affected FPS generation without reapplying gain.

Temperature acknowledgement is separate from physical settling. Requested SET
is retained until its readback confirms, including across mismatches and
malformed replies. Confirmed cooling can continue fast polling for an arbitrary
physical settling time. Two consecutive fresh readings within the existing
one-degree criterion return to idle. Malformed/unknown readings after a valid
observation start bounded fast observation checks. Expired requests stay failed
while idle temperature monitoring continues; a later measurement cannot revive
their acknowledgement.

Cooling-state evidence is deliberately conservative. The existing reply has no
explicit state bit: power >= 5 indicates On, while power < 5 with CCD - SET >
2.99 C indicates Off. Low power near SET is Unknown instead of inheriting an
optimistic setter state. Unknown state times out into slow monitoring. This can
leave cooling-Off unconfirmed near a warm setpoint despite a successful command;
actual reply/state semantics need hardware qualification. Every cooling-Off
attempt now queries fresh temperature under the same mutex before the existing
CCD > 19 C interlock.

The diagnostic pause refuses unavailable/pending confirmations and active
thermal convergence, but permits terminal failed requests once idle. Resume
makes one temperature read due; confirmed gain/FPS remain quiet. The read-only
housekeeping status is published after shared camera updates and distinguishes
cached confirmation, latest observation validity, pending deadlines and thermal
phase. Existing telemetry continues on no-due-work and query-error passes.
`protectionPolling=confirmation only` explicitly identifies the absence of
background gain-trip discovery. The operator guide now contains configuration,
exact INDI elements, pause eligibility and the comparison procedure.

Configuration validates finite positive, representable intervals, interval
ordering and a positive integral sample count. Both configuration and parsed
numeric data use `mx::math::isFinite`: the production `-ffast-math` build can
optimize ordinary `std::isfinite` checks away. Tests cover NaN, negative/fractional
sample counts and malformed complete numeric tokens under those build flags.

An unexpected external frequency change immediately invalidates a previously
confirmed target while still updating the observed/worker rate and requesting
the existing reconfiguration. Initial observation attempts are deferred while
mode configuration is pending, so they cannot confirm an intermediate rate.

Local verification completed:

| Suite | Test cases | Assertions |
| --- | ---: | ---: |
| OCAM controller | 33 | 841 |
| Housekeeping policy/pause | 26 | 640 |
| Response parsers | 4 | 480 |
| Temperature configuration/control | 3 | 69 |
| Acquisition lifecycle | 1 | 7 |
| Total | 67 | 2,037 |

All suites pass. The deterministic policy tests include a simulated settled
minute with six temperature queries and zero gain/FPS queries. Failure cases
also verify telemetry progress, fixed confirmation deadlines, no repeated
setting writes, obsolete-target handling, fresh cooling-Off checks and bounded
unknown-state recovery. The local SDK 5.6.8.0-linked camera build passes; its
binary was not launched. `clang-format --dry-run --Werror` passes for all five
touched C++ files, and `git diff --check` passes. Build/test logs are under
`/tmp/ocam-hk-final-*`, `/tmp/ocam-housekeeping-policy-*`,
`/tmp/ocam-housekeeping-parsers-*` and `/tmp/ocam-hk-{temperature,lifecycle}-*`.

All hardware actions and comparative captures remain operator tasks. Rebuild
and install `ocam2KCtrl`; this policy introduces no telemetry schema and needs
no `logdump` rebuild.
