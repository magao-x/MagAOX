# pupilGuideGUI device and property availability

## Task description

Date: 2026-10-13. Branch: `jrmales/pupilguide-device-recovery`, created from
`dev` at `8c652ec5b3ef43db0cbdbc3833c5c8238ebf6495`.

Status: user answers incorporated and producer contracts checked on 2026-10-14;
implementation has not started.

The user reported:

> gui/apps/pupilGuideGUI: restarting twAlign-camwfs-wfs on RTC takes the whole
> GUI to not-enabled. It also never brings back the "move woofer" button.
> Neither of these should be true, please analyze.

The user then asked how missing devices and properties are handled at startup,
and requested a feature branch and an implementation plan under `agents/plans`,
using [task_template.md](../../task_template.md) as guidance.

## Problem statement and expected behavior

Restarting one device must invalidate only that device's state and controls that
depend on it. Unrelated devices must remain usable. A device that returns must
recover through normal INDI definitions/updates without restarting the GUI or
reconnecting the shared INDI client.

Startup absence, property deletion, device restart, and loss of the shared INDI
connection need consistent availability rules. The GUI must distinguish a known
zero/Off value from a value that has never arrived or was invalidated.

The "move woofer" button is `button_ttmtel`, the selector that cycles between
woofer, telescope, and TTM motion. It does not itself move hardware. Keep this
selector usable while connected, including when the selected actuator is absent;
gate the motion buttons on the selected actuator's actual dependencies.

## Scope and working constraints

- Initial implementation is in
  [pupilGuide.hpp](../../../gui/widgets/pupilGuide/pupilGuide.hpp), with offline
  GUI tests and test documentation. Preserve the current layout, coordinate
  transforms, step sizes, target-selection order, and lab-mode command behavior.
- Follow [AGENTS.md](../../../AGENTS.md), including the full changed-file
  documentation pass, out-of-class definitions, Doxygen test links, formatting,
  feature-branch discipline, and separate functional/documentation/formatting
  commits where applicable. Preserve this widget's existing header-based layout.
- Work only on a workstation. Do not run agents, INDI commands, or test clients
  against RTC or other instrument computers, including through a tunnel. Any
  eventual instrument validation must be performed by the user.
- No controller/protocol changes, GUI redesign, general subscription-retry
  implementation, or suite-wide widget refactor are planned.
- This task currently authorizes the branch and planning document. Record future
  implementation and validation below when implementation is requested.

# Agent findings and plan

## Confirmed findings

### Device deletion incorrectly takes the full-disconnect path

`pupilGuide::handleDelProperty()` checks a list of 17 device names, including
`twAlign-camwfs-wfs`, and unconditionally calls `onDisconnect()` for any property
deletion from those devices. It does not inspect the property name. Even deletion
of an unrelated property from a listed device triggers the same path.

`onDisconnect()` calls `resetConnectionState()`, which clears all cached FSMs,
including `dmwoofer` and `wooferModes`. It also disables general controls and
sets the title to `Alignment (disconnected)`. The 250 ms `updateGUI()` timer then
disables controls whose previously healthy device states were erased.

The only code that re-enables `button_ttmtel` is `onConnect()`. Device definitions
and updates do not call it. The camera selector, alignment Start/Stop buttons,
some labels, and window title have the same recovery problem. Unrelated devices
may not immediately resend unchanged state, so their controls can also remain
disabled after their caches have been cleared.

The shared connection layer already distinguishes the events:

- [multiIndiPublisher.hpp](../../../gui/lib/multiIndiPublisher.hpp) dispatches
  property deletion without tearing down the connection or subscriptions.
- [multiIndiManager.hpp](../../../gui/lib/multiIndiManager.hpp) issues connection
  callbacks when the shared client connects/disconnects, not when an individual
  device disappears or returns.

The broad delete handler was introduced in `3147eee1` ("Harden GUI INDI reconnect
and delProperty handling"). `042dfa07` subsequently expanded the cache reset in
the disconnect path.

### Startup absence is local, but property completeness is not checked

The constructor calls `onDisconnect()` and starts the refresh timer. Once the
shared connection opens, `onConnect()` enables general controls without waiting
for device definitions. Most device controls subsequently depend on FSM state:
an absent device/FSM leaves an empty state and keeps its controls disabled.

For an absent `twAlign-camwfs-wfs`, this leaves the sensor controls disabled while
woofer controls work after their own FSMs arrive. The selector remains enabled.
Alignment Start/Stop nevertheless remain enabled even with the sensor absent.

Individual properties are not consistently tracked as available:

- READY/OPERATING from the alignment sensor enables all its controls even if
  `continuous`, `nPokeImages`, `nPokeAverage`, or `poke_amp` never arrived.
- `dmwoofer=OPERATING` and `wooferModes=READY` enable woofer motion even if
  `current_amps` never arrived. Tip, tilt, and focus start at zero, and outgoing
  absolute targets are calculated from those cached values.
- Other relative-motion implementations also calculate absolute targets from
  cached positions. Inventory their dependencies rather than fixing only the
  reported sensor and leaving the same missing-value problem elsewhere in this
  panel. `move_picoscix()` already has a position-sentinel guard worth preserving.

The GUI subscription layer requests properties when registering subscriptions;
it has no missing-property timeout or periodic retry. Later definitions/updates
are accepted. This is separate from the app-side subscription backoff discussed
in `doc/incidents/2026-03-11-indi-optional-subscriptions-plan.md`.

### Offline evidence

On workstation `mx11`, a temporary Qt harness included the actual widget and
called its connection/property callbacks directly. No INDI manager, socket,
instrument connection, or outgoing hardware command was used. The harness built
with qmake/make and ran with `QT_QPA_PLATFORM=offscreen`.

| Simulated event | Woofer selector | Woofer arrows | Sensor controls | Alignment Start |
| --- | --- | --- | --- | --- |
| Healthy FSMs received | Enabled | Enabled | Enabled | Enabled |
| Sensor device deleted | Disabled | Disabled | Disabled | Disabled |
| Sensor READY returns | Disabled | Disabled | Enabled | Disabled |
| Unrelated woofer FSMs resent | Disabled | Enabled | Enabled | Disabled |
| Full connection callback | Enabled | Enabled | Enabled | Enabled |
| Startup connected, no definitions | Enabled | Disabled | Disabled | Enabled |
| Startup woofer FSMs only | Enabled | Enabled | Disabled | Enabled |
| Sensor FSM arrives without other sensor properties | Enabled | Enabled | Enabled | Enabled |

Whole-device deletion (empty property name), `fsm`, `nPokeImages`, and an unrelated
property name all reproduced the global-reset defect. The startup checks also
confirmed enablement without `current_amps` or sensor parameter definitions.

The temporary harness was at `/tmp/pupilguide-restart-analysis`; it is not a
durable test artifact or an acceptance test suite. Add repository tests during
implementation. This evidence establishes widget behavior; it does not capture
the exact INDI message sequence from the reported RTC restart.

## Proposed requirements

1. Reserve full-state invalidation and the disconnected title for actual shared
   connection loss. Initialization uses the same unavailable state.
2. Handle whole-device deletion, FSM deletion, and individual-property deletion
   explicitly. Whole-device deletion invalidates all dependencies for that
   device; individual deletion invalidates only dependent controls/state. Treat
   FSM deletion as loss of device readiness and invalidate command baselines
   that must be refreshed before motion resumes.
3. Preserve unrelated device state, local selections, and step sizes across
   another device's deletion. Ignore unrelated device/property deletions.
4. Recover affected controls from fresh valid definitions/updates, in either
   arrival order, without invoking global `onConnect()` for a returning device.
5. Track availability separately from cached values for command dependencies.
   Do not interpret default zero/false or a retained display value as observed
   state. Check required elements and numeric validity before accepting values.
6. Keep the movement-target selector enabled while the shared connection is up.
   Enable movement only when its own FSM and required current values are valid.
   Use the same eligibility checks in command handlers to prevent commands
   between a delete callback and the next periodic GUI refresh.
7. Refresh availability immediately after connection/deletion/recovery events.
   A timer tick must not re-enable controls using stale or missing dependencies.
8. Preserve subscriptions during device loss. No reconnect, command replay, or
   automatic alignment restart is required to recover the UI.
9. Distinguish alignment Start from Stop. Per the user's answer, Start requires
   all participants used by its existing command sequence; if any required
   participant is unavailable, Start is disabled and sends no partial sequence.
   Stop is enabled if any applicable stop target is available and sends only to
   available targets, without requiring all participants to be READY. Preserve
   the existing lab-mode condition on stopping `camwfs-align`.
10. Per the user's answer, unavailable TCS means lab mode. Initialize/fall back to
    lab mode when TCS or its mode information is unavailable, including after a
    disconnect or relevant deletion. Do not retain a stale non-lab setting. A
    returning, connected TCS with fresh mode information supplies the mode again;
    independent woofer/TTM controls do not acquire a TCS dependency.

## Design and reuse

Use explicit connection state plus small, documented per-device/property
validity fields and shared eligibility helpers within `pupilGuide`. Keep the
dependency logic in one place so GUI updates and command handlers agree. Avoid
replacing the shared INDI connection manager or introducing a generic dependency
framework for this one panel.

Initial dependency inventory:

| Control | Dependencies to represent |
| --- | --- |
| Movement selector and camera selector | Shared connection; existing local selection rules |
| Woofer tip/tilt | `dmwoofer.fsm`, `wooferModes.fsm`, valid `current_amps.0000` and `.0001`; existing target command contract |
| Woofer focus | Same FSMs plus valid `current_amps.0002` |
| TTM/telescope motion | Existing FSM rules and observed command inputs; unavailable TCS means lab mode and excludes telescope motion |
| Pupil/periscope/lens/picomotor relative moves | Respective FSM plus current values for the axes actually used |
| Sensor slider and parameter entries | Sensor readiness plus each control's own property/required elements |
| Alignment loop sliders, gains, and deltas | Corresponding device readiness and individual property availability |
| Alignment Start | All dependencies of the current three-device start sequence; unavailable participant blocks the entire action |
| Alignment Stop | Any available target of the current stop sequence; send only to available applicable targets, without requiring all devices READY |

Finalize the complete input/output property inventory before coding. Where
command capability must be observed, subscribe to the relevant property rather
than inferring it solely from FSM state. Specify how Def versus Set establishes
capability and valid values; do not accidentally require a full vector on every
partial update. After invalidation, only fresh observations can restore validity.

Reuse the offline QApplication/Catch2 and captured-command patterns in
[roi_test.cpp](../../../gui/widgets/roi/tests/roi_test.cpp), and the targeted
invalidation examples in `roi.hpp` and `pwrDevice.hpp`.

Child widgets are registered directly with the shared publisher. Audit their
interaction with parent availability: `statusEntry` and `toggleSlider` currently
match an exact property name in `handleDelProperty()`, so a device-wide empty-name
delete does not clear their own cache/pending state. Ensure affected children
clear stale displays, unsent edits, and pending toggle state while unrelated
children remain intact. Prefer a scoped parent solution; any necessary shared
widget change needs its own regression coverage for existing consumers.

`coronAlign` contains a similar broad delete-to-disconnect pattern. Record it as
a follow-up. If implementation reveals an identical reusable operation, propose
a small common helper; do not expand this task into a suite-wide change.

### Producer contracts verified after user review

[dmPokeXCorr.hpp](../../../apps/dmPokeXCorr/dmPokeXCorr.hpp) inherits
[dev::dmPokeWFS](../../../libMagAOX/app/dev/dmPokeWFS.hpp), whose `appStartup()`
registers `continuous.toggle`, `nPokeImages`, `nPokeAverage`, and `poke_amp`.
The numeric properties expose `current` and `target`. The `continuous` callback
accepts On to start continuous measurements and Off to request stopping an
active measurement. The separate `stop.request` capability is not needed to
preserve the GUI's current start/stop command contract.

[closedLoopIndi.hpp](../../../apps/closedLoopIndi/closedLoopIndi.hpp) registers
`loop_state.toggle`; its callback calls `toggleLoop(true/false)` for On/Off.
Thus the controller's `loop_state` subscription is correct, while the sensor's
`loop_state` subscription is wrong.

The parent's `m_twAlignSensorState` is currently assigned/reset but never read.
Do not retain it just to duplicate `ui.actAlignSensor_slider` state. The parent
does need sensor command availability for the combined Start/Stop actions:
subscribe to `continuous` for that purpose, validate its `toggle` element, and
track availability separately from whether the sensor is currently On or Off.
Remove the unused sensor-state cache/handler when replacing the bad subscription.
If future parent behavior needs running state, obtain it from this same verified
property rather than reintroducing a `loop_state` dependency on the sensor.

[tcsInterface.hpp](../../../apps/tcsInterface/tcsInterface.hpp) exposes
`labMode.toggle` and uses CONNECTED for an available telescope connection. For
the planned GUI behavior, loss of that availability forces effective lab mode;
a missing/deleted lab-mode value also retains the lab fallback until refreshed.
Handle mode/FSM arrival in either order. If telescope was selected when TCS is
lost, disable its movement and keep the selector usable so the user can choose
woofer/TTM; do not silently redirect a pending telescope move to another actuator.

## Implementation sequence

### 1. Add an offline regression harness and isolate device loss

- Add `gui/widgets/pupilGuide/tests/pupilGuide_test.cpp` and a qmake project,
  using the actual widget and generated UI. Capture outgoing properties in
  memory; never construct a network-connected manager for these tests.
- Reproduce the reported restart, startup absence, and stuck-selector behavior.
- Add explicit shared connection state. Replace the global delete handler with
  device/property-scoped invalidation and refresh helpers; keep true connection
  loss global. Reconcile child caches without disconnecting unrelated children.
- Verify event-driven recovery, labels/title, selection persistence, and repeated
  deletes/restarts. Commit this functional change, tests, and updated plan.

### 2. Enforce property availability and command eligibility

- Complete the dependency inventory, including missing property/element cases,
  partial updates, and existing FSM conditions. Inspect property producers when
  needed to confirm command contracts and subscription names.
- Add value-validity/capability tracking, scoped invalidation, and subscriptions
  needed by the chosen controls. Derive enabled state from those dependencies.
- Guard command handlers with the same predicates and preserve current command
  values/order when all dependencies are present. Reset validity on genuine
  disconnect and relevant deletion; do not make preserved numeric caches usable
  merely because an FSM returns READY.
- Implement the agreed Start/Stop policies and verify missing sensor parameters
  do not unnecessarily disable independent sensor controls.
- Replace the sensor's incorrect `loop_state` subscription with `continuous`
  availability tracking and remove its unused running-state cache. Keep the
  controller subscriptions on `loop_state`.
- Implement the lab-mode fallback on TCS loss and fresh-mode recovery, including
  selection/command behavior when telescope was selected at the time of loss.
- Extend the regression suite and commit this functional change with the plan.

### 3. Complete documentation and verification

- Put the test group in `tests/groups.dox` under `application_unit_test`, using
  `libXWCTest::pupilGuideGUITest` and a `pupilGuideGUI_unit_test` group. Document
  each test case, hide harness-only helpers, and use `tests/testXWC.hpp` plus
  explicit Doxygen references where harness indirection hides the real API.
- Complete the full changed-file documentation pass. Keep documentation-only
  changes separate from functional commits, followed by formatting-only cleanup
  if needed. Run repository clang-format on touched C++ files and verify output.
- Build `pupilGuideGUI`, run the focused offline suite, check Doxygen test links,
  and run `git diff --check`. If shared widgets change, run their affected tests
  and representative consumer builds as well.
- Update this record with exact commands/results, any limitations, and commit
  identifiers. No deployment or instrument validation is part of agent work.

## Acceptance tests

| Scenario | Required result |
| --- | --- |
| Connected startup with sensor absent | Sensor-dependent actions unavailable; healthy woofer and unrelated controls usable; selector enabled |
| Sensor device-wide delete followed by Def/Set recovery | Only its dependents invalidate; title stays connected; normal recovery without `onConnect()` |
| Named `fsm` deletion and later READY | Affected readiness/baselines invalidated; unrelated controls unchanged; missing baselines still block commands |
| Named sensor-parameter deletion | Only controls requiring that property disable; other sensor controls remain usable |
| Unknown device/property deletion | No availability, selection, title, or cached-value changes |
| Parameterized restart of each supported device | Correct dependents invalidate and recover; no cross-device cache erasure |
| Woofer FSMs arrive before amplitudes | Tip/tilt/focus remain unavailable until their required elements arrive; no targets calculated from default zero |
| Data arrives before FSM; partial vectors arrive separately | Availability converges when all required evidence exists; unrelated valid elements are retained appropriately |
| Missing/malformed/nonfinite required values | No dependent command; valid zero remains usable; later valid data restores availability |
| Only one axis or mode value is unavailable | Disable only actions that require it; preserve independent motion and focus eligibility |
| Real connection loss, then reconnect | All remote-dependent actions invalidate; stale caches do not re-enable on timer ticks or connection alone |
| Each subset of alignment participants available | Start enabled only when all required participants are available; Stop enabled when any applicable target is available and sends only to those targets; preserve lab-mode routing |
| Stop capability present while a participant is not READY | Stop remains usable; current Off state does not mean the capability is unavailable |
| TCS absent at startup or lost after reporting non-lab mode | Effective lab mode becomes true; independent woofer/TTM controls work; telescope motion is unavailable and selection can be changed |
| TCS FSM/mode return in either order | Fresh connected state and mode restore the reported mode; a stale non-lab cache cannot override the fallback |
| Only the sensor's real properties are published | `continuous.toggle` is sufficient for sensor command availability; no dependency on sensor `loop_state`; controller still uses `loop_state` |
| Command slot called immediately after invalidation | No command relying on missing data, even before the next timer tick |
| Recovery after an unsent edit/pending toggle | No stale edit submission, automatic toggle, motion, or command replay |
| Repeated deletion/redefinition | Idempotent state; selection and scale preserved; no duplicate commands/subscriptions |

Assert actual widget enabled states and captured command devices, properties,
elements, and values. Exercise Qt event processing where queued child updates
matter. Avoid timing-dependent sleeps and arbitrary coverage-percentage targets.

## User review and resolved decisions (2026-10-14)

The original questions and user answers are retained below as the engineering
record. The resolutions supersede the earlier proposals where they differ.

- **Alignment Stop:** proposed behavior keeps available participants stoppable
  when one disappears, with no READY requirement for an otherwise valid stop
  capability. Confirm this operator behavior before changing command routing.

  ANSWER: Yes, if any are available then stop should be available.  Conversely, if any are not available then start should not be available.

  Resolution: Start uses all-required availability; Stop uses any-applicable
  availability. Apply these predicates both to the buttons and their command
  handlers. Stop does not depend on an all-READY test or on receiving an On state.

- **Unknown lab mode:** `m_labMode` currently defaults to true and survives the
  current connection reset. Define how unavailable `tcsi.labMode` affects
  telescope selection and alignment Stop without accidentally adding a TCS
  dependency to independent woofer/TTM operation. Conservative proposal: treat
  telescope eligibility as unknown/unavailable until lab mode is observed.

  ANSWER: If TCS is not available then we are in lab mode almost by definition. So let's just go to lab mode if TCS is gone.

  Resolution: use lab mode as the effective fallback on TCS loss, replacing the
  proposal to leave mode unknown. Clear stale non-lab knowledge and recover from
  fresh TCS state/mode without blocking unrelated motion.

- **Sensor property mapping:** the parent subscribes to `loop_state`, while the
  sensor slider and Start/Stop send `continuous`. Verify the producer's actual
  contract before making either property an availability requirement. Do not
  introduce an impossible prerequisite from this existing mismatch.

  ANSWER: loop_state comes from the controller, not the wfs. At line 722 this is a mistake, we should subscribe to continuous -- if we actually use the sensor state independent
  ly of ui.actAlignSensor_slider.  See the apps/dmPokeXCorr/dmPokeXCorr.hpp (and base classes) for the contract for -wfs.  For -ctrl see closedLoopIndi.

  Resolution: producer review confirms the mapping. Track `continuous.toggle`
  availability for the parent Start/Stop dependency, keep slider state handling
  in the child, and remove the currently unused parent sensor-state cache.

All three review questions are resolved. No further answer is needed on these
points to implement the corresponding planned behavior.

## Follow-up and limits

- Missing-property retry/backoff is separate work. Recovery here is driven by
  definitions/updates actually received; if a late-starting producer remains
  silent, investigate bounded rediscovery separately from UI state handling.
- The protocol provides no local generation identifier for distinguishing an
  old queued message from a new one with identical device/property names. Test
  normal callback ordering and avoid claiming stale-message rejection beyond
  the evidence available to this GUI.
- A full disconnect must also invalidate numeric baselines and mode knowledge,
  not just FSM strings. Retaining a display value is permissible only if it is
  clearly unavailable and cannot be used as command input.
- Plan review should keep the restart fix independently reviewable from the
  broader property-validity changes. Apply no instrument actions during tests.

## Execution record

- 2026-10-13: reviewed source paths, shared connection dispatch, existing Qt test
  patterns, and `agents/task_template.md`; reproduced restart and startup states
  offline; created the feature branch and this plan. No production code or
  repository test files have been changed.
- 2026-10-14: read and preserved the user's three answers; reconciled requirements,
  dependency inventory, implementation steps, and acceptance tests. Verified
  `dmPokeXCorr`/`dmPokeWFS`, `closedLoopIndi`, and `tcsInterface` contracts by local
  source inspection and confirmed the parent sensor-state cache is unused.
  This update changes only the plan; no production code or tests were modified.
