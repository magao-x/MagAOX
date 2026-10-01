# xInstGraph: parked motion stages in POWEROFF

- Date: 2026-10-01
- Status: implemented and verified in software; hardware verification pending
- Source baseline: MagAOX `4b57330e` on `jrmales/instgraph-updates`

## Objective

When a stage reports `fsm.state=POWEROFF` and `parked.current` is true,
`xInstGraph` should apply its published position to the graph puts. The FSM
and the graph's `fsmstate` extra must continue to say `POWEROFF`.

This follows the distinction already used by `stageGUI`: a retained physical
position can remain meaningful while motion commands are unavailable. In the
graph, the existing position-to-put mapping uses `presetName` or `filterName`
switch selections; it does not convert arbitrary numeric positions to routes.

## Findings

### 1. The controller contract already supports retained parked positions

[`dev::stdMotionStage`](../../../libMagAOX/app/dev/stdMotionStage.hpp) supplies
the preset/filter interface. Its `onPowerOff()` sets `m_moving=-2` without
clearing the position or selected-name state, and `updateINDI()` publishes the
name resolved by `activePresetNameIndex()`.

[`zaberCtrl`](../../../apps/zaberCtrl/zaberCtrl.hpp) supplies the parking
contract:

- `appStartup()` publishes a read-only Number property `<device>.parked` with
  element `current`, and subscribes to the low-level parked state.
- Its stage-state callback preserves `POWEROFF` and calls
  `syncPowerOffStageTelemetry()` followed by `stdMotionStage::updateINDI()`.
- While parked, that helper uses `presetNumber()` to associate the retained
  measured position with a preset. While unparked it clears numeric preset
  telemetry to zero. The parked callback also resynchronizes these values if
  the controller is already powered off.
- `appLogic()` continues to publish the motion interface in `POWEROFF`.

[`zaberStage::onPowerOff()`](../../../apps/zaberLowLevel/zaberStage.hpp) retains
the raw position. [`zaberLowLevel::onPowerOff()`](../../../apps/zaberLowLevel/zaberLowLevel.hpp)
publishes the retained parked/position snapshot before publishing `POWEROFF`.
No new INDI property or instGraph library API is needed for the graph consumer.
Parking remains a derived-controller capability; it is not guaranteed for every
app using `stdMotionStage`.

### 2. The GUI already separates FSM state from position display

`stageGUI` uses [`gui/widgets/stage/stage.hpp`](../../../gui/widgets/stage/stage.hpp).
That widget subscribes to `parked`, reads its numeric `current` element, and
recognizes exactly `POWEROFF && parked` in `updateGUI()`. It keeps the position
visible and the actual FSM text available, with all motion controls disabled.

[`statusCombo::shouldShowValue()`](../../../gui/widgets/xWidgets/statusCombo.hpp)
also accepts `POWEROFF && parked`, covering the shared preset display used by
camera stage widgets. The stage widget clears parking on disconnect or property
deletion.

The [`rtimv cameraStatus plugin`](../../../gui/rtimv/plugins/cameraStatus/cameraStatus.cpp)
has related logic, but accepts parked positions in several other non-ready
states and can fall back to numeric position. Those broader display rules are
not an appropriate default for graph routing. This change should add only the
requested `POWEROFF` exception.

### 3. The graph consumer lacks parking and has two independent READY gates

[`stdMotionNode`](../../../apps/xInstGraph/xigNodes/stdMotionNode.hpp), the
`type=stdMotion` handler, subscribes to FSM, preset names, and optional tracking
properties. It has no parked key or cached parked flag.

- `handleSetProperty()` turns normal preset puts off unless the FSM is `READY`.
- `togglePutsOn()` independently requires `READY` before applying a preset.
- Updating only the first check would therefore leave parked puts off.
- The tracking-request branch accepts active tracking only in `READY` or
  `OPERATING`; it needs a distinct parked-position fallback.
- Preset changes do not increment `m_changes` while `m_tracking` is true. A
  retained tracker flag would therefore suppress subsequent parked-position
  updates even after the FSM changes to `POWEROFF`.
- `togglePutsOff()` can label a powered-off node `tracking` from a retained
  tracker flag, despite clearing its puts.

[`fsmNode::handleSetProperty()`](../../../apps/xInstGraph/xigNodes/fsmNode.hpp)
already records the received FSM and writes the `fsmstate` extra. Preserve that
behavior and calculate position usability separately.

### 4. A producer alias bug can publish the commanded name while powered off

`stdMotionStage::activePresetNameIndex()` currently prefers the requested alias
whenever `m_moving != 0 && m_movingState == 1`. Negative motion sentinels satisfy
that condition too.

For example, with measured position at preset A, a named request for B, and
`m_movingState=1`, entering power-off sets `m_moving=-2`. The helper still returns
B even when B's configured position differs from A. This is a failure inferred
from the code path, not a hardware observation. A parked graph must not treat
that commanded name as the retained position.

The existing [zaberCtrl tests](../../../apps/zaberCtrl/tests/zaberCtrl_test.cpp)
cover parked numeric telemetry and alias preservation at a shared position,
but do not cover a requested alias that differs from the position at power-off.
One existing test is titled as clearing aliases on power-off while actually
asserting their preservation; correct that description when extending it.

### 5. Separate property messages are not one coherent device snapshot

[`xInstGraph::appStartup()`](../../../apps/xInstGraph/xInstGraph.hpp) registers
all keys requested by each handler, so adding `<device>.parked` needs no special
app subscription path. Initial DefProperty messages are forwarded through the
normal SetProperty handling by `MagAOXApp`.

The app publishes an atomic graph snapshot after each callback. FSM, parked,
and preset messages still arrive independently. The handler must recompute on
changes to any of them and converge regardless of their initial delivery order;
it cannot assume an INDI transaction across all three properties.

Unlike the GUI, the graph handlers have no property-deletion or disconnect
invalidation path. `indiDriver` forwards Def/Set messages to the app but does not
provide an app DelProperty override. Cached telemetry can therefore survive a
publisher disappearing. This is an existing app-wide freshness limitation,
which adding a parked subscription alone will not solve.

## Proposed behavior

| Reported state | Parking / position | Graph behavior |
| --- | --- | --- |
| `READY` | Any parking value | Preserve existing preset and tracking behavior. |
| `OPERATING` | Any parking value | Preserve existing tracking behavior; do not add normal preset routing. |
| `POWEROFF` | Parked, usable named preset | Apply normal preset routing and labels; keep `fsmstate=POWEROFF`. |
| `POWEROFF` | Not parked, parking unknown, or no usable preset | All puts off, including `alwaysOn`. |
| Any other FSM state | Even if parked | Preserve current inactive behavior. |

For the powered-off parked case:

- Require an affirmative `<device>.parked.current`; initialize parking false.
  A device with no parked property retains its existing behavior.
- Use the controller-published named selection from the configured
  `presetPrefix`. Never use a target or infer a route from a numeric position.
- Require one nonempty, non-`none` selected name. Ambiguous selections are not
  usable. For multiple selected-side puts, the name must match a configured put;
  an unmatched name must not activate only the common put or `alwaysOn` paths.
- Apply the existing single-put labels, input/output selection, `alwaysOn`, and
  `noAutoOn` behavior once a usable route exists. Clear all puts when it does not.
- Give the parked preset path priority over retained tracking-request/status
  flags. The position remains meaningful with motors off; active tracking does
  not. Return to existing tracking rules when power/FSM state changes.
- Keep the true FSM and show the preset as the position label. With no usable
  parked position, use the normal inactive position label rather than a stale
  `tracking` label.

Numeric-only parked positions with no named preset remain inactive in this
proposal. Supporting them would require a separate position-to-route mapping
contract, especially for multi-put nodes.

## Implementation plan

### 1. Correct name resolution for negative motion states

In `libMagAOX/app/dev/stdMotionStage.hpp`, restrict the in-progress named-command
shortcut in `activePresetNameIndex()` to actual motion (`m_moving > 0`). When
powered off or not homed, resolve from `derived().presetNumber()` and retain a
requested alias only when its configured position matches that resolved preset.
Do not indiscriminately clear aliases on power-off: aliases at the actual parked
position must remain stable.

Extend the existing zaber controller harness tests for this condition before
enabling parked graph routing. Check the resolved switch name and telemetry
name, as well as numeric current values. Keep the existing named-motion and
shared-position alias tests passing. No controller parking or FSM redesign is
needed.

### 2. Subscribe to parking in stdMotionNode

Add documented `m_parked` state, initially false, and a `<device>.parked` key
registered when `device()` is set. Consume the Number property's `current`
element, matching the GUI/controller contract. Changes to that flag must trigger
recomputation. If a received parked property lacks a usable `current` value,
clear the cached parked flag. Handle malformed data locally without throwing
out of the callback. Keep this automatic and optional; no new config setting is needed.

### 3. Separate usable position from operational FSM and tracking

Factor the routing decision so `handleSetProperty()` and `togglePutsOn()` use
consistent conditions. Prefer a small predicate for the parked-poweroff case
and a shared route-selection path, with out-of-class inline definitions matching
the current app structure.

Evaluate the parked case before tracking mode, and ensure preset changes dirty
the node in that case even when a tracker flag is retained. Validate the parked
selection without changing established powered-on routing behavior. Make both
on/off label paths respect this distinction. Leave `fsmNode` and the received
FSM values unchanged.

### 4. Verify graph publication through the existing callback

Extend `apps/xInstGraph/tests/xInstGraph_test.cpp` with a configured motion node
and parked property. Check registration, callback dispatch, published XML puts,
position labels, and the unchanged `POWEROFF` FSM extra. Use the existing checked
snapshot path; do not add library writes or an alternate publication mechanism.

### 5. Document and finish

Update `apps/xInstGraph/README.md` with the optional parking property, the exact
`POWEROFF` condition, tracking fallback, and the no-preset behavior. Record
implementation and verification results here.

Follow `AGENTS.md`: full documentation pass on changed C++ files, inline
parameter docs, header-only app conventions, `clang-format`, and documented
Catch2 cases with real API links. Keep the producer correction and the graph
change in separate functional commits, with relevant execution notes; follow
with documentation-only cleanup if needed. No separate instGraph repository
change is expected.

## Acceptance and verification

### Producer tests

- Parked power-off at a configured preset preserves numeric current position
  and the correct selected name.
- A duplicate-position alias is preserved at that position after power-off.
- A requested alias at a different position is not reported as current when
  `m_moving=-2`, including interruption of a named move.
- No matching position yields `none`, not the outstanding named target.
- Positive named motion continues to report the selected target as before.

### Node tests

Extend `apps/xInstGraph/xigNodes/tests/stdMotionNode_test.cpp`:

- `READY -> POWEROFF` with parked true retains the selected route and its label;
  the FSM extra remains `POWEROFF`.
- Initially powered-off parked nodes converge for every ordering of FSM,
  parked, and preset messages, including initial definitions.
- Changing the selected preset while parked/off updates the route immediately.
- Clearing parking while off clears every put, including `alwaysOn`; setting
  parking true again reapplies the current selection.
- Missing/false/invalid parking, all-Off selection, `none`, multiple On names,
  and an unmatched multi-put name do not enable a parked route.
- Cover single-put and multi-put input/output configurations, `presetName` and
  `filterName`, plus existing `alwaysOn`/`noAutoOn` transitions.
- Retained tracking flags and new tracking messages cannot override a parked
  position or label it active tracking. Returning to `READY` or `OPERATING`
  restores the existing tracking contract.
- Parked true does not enable positions in `HOMING`, `NOTHOMED`, `POWERON`,
  `NOTCONNECTED`, `ERROR`, or an unknown FSM state.

Run the focused node and app suites using `tests/Makefile.one`, the zaberCtrl
suite, and the stdMotionStage helper suite. Build `apps/xInstGraph` and
`apps/zaberCtrl` if the producer header changes. Check new test Doxygen links.
Then compare a physically parked powered-off stage in `stageGUI` and the graph,
including starting xInstGraph while the stage is already parked and powered off.

## Follow-up boundary and remaining risks

- Property deletion, publisher restart, and connection loss need an app-wide
  cache invalidation design. Defer that wider work explicitly; this change does
  not guarantee freshness when a device silently disappears. At app startup,
  parking defaults false until published.
- Per-property recomputation guarantees convergence, not a simultaneous
  multi-property device snapshot. A transient route based on the most recently
  received values remains possible during separate updates.
- Other controllers may preserve different position/name semantics. Enable the
  exception only when their device explicitly publishes parking, and verify
  their producer contract before relying on the graph route.
- Negative `m_moving` values also enter some existing numeric-property Busy
  checks. That property-state presentation issue is separate from name
  correctness and put routing and is not required for this change.

## Analysis verification

Findings above are based on source inspection and the existing test cases at
the recorded baseline. No code was changed and no tests or hardware checks were
run for this planning task.

## Execution notes (2026-10-01)

### Producer correction

- Restricted the named-command shortcut to positive motion. Negative motion
  states now resolve the retained position, while aliases at the same position
  remain preserved.
- Added a regression for interrupted named motion at another preset, between
  presets, and in the not-homed state, plus alias preservation at the retained
  position. Before the correction, three sections failed with the commanded
  alias reported instead of the retained position.
- Fixed the existing Zaber test harness's validation-only shortcut so
  data-bearing properties execute real callbacks. Its homing scenario had
  previously returned before processing any state transition.
- The Zaber suite passed 101 assertions in 5 cases; the stdMotionStage helper
  suite passed 12 assertions in 1 case. Hardware checks remain pending.

### Graph routing

- Producer correction committed as `aef45e0b`.
- Added the automatic optional device-local parked subscription, checked numeric
  parsing, and a shared on/off decision. Parked powered-off nodes apply valid
  named presets while retaining `fsmstate=POWEROFF`. Ambiguous selections and
  unmatched multi-put names leave every put off.
- Parked routing takes priority over retained tracking flags, including subsequent
  preset updates; normal tracking rules resume in READY/OPERATING. Powered-off
  inactive labels no longer claim tracking.
- Added isolated node fixtures covering all six initial message orders, input
  and output routing, single and multiple puts, preset/filter notation,
  alwaysOn/noAutoOn transitions, malformed parking, invalid selections, other
  FSM states, and tracking transitions. The node suite passed 1487 assertions
  in 9 cases.
- Added app publication coverage for all six initial DefProperty message orders
  through the real base-class dispatch with a /dev/null-backed driver. It checks
  subscriptions, graph colors and position labels, subsequent SetProperty
  changes, and the unchanged POWEROFF label. The app suite passed 375 assertions
  in 17 cases.

### Documentation and build checks

- Graph routing committed as `9a917e05`.
- Documented the optional parking contract, invalid-position behavior, tracking
  fallback, and existing cache freshness limitation in the app README.
- Completed the changed-file documentation pass, moved test-harness method
  bodies below their declarations, grouped the older tracking scenario under
  the app test namespace, and enabled ZABERCTRL_TEST_DOXYGEN_REF in the project
  Doxyfile so producer tests can link to the protected API under test.
- Both final app builds succeeded with `make -j1`. The build system shares
  generated version files and a precompiled header; overlapping builds caused
  cleanup and compiler races, resolved by serializing both targets and their
  build steps. No build-system changes were needed for this implementation.
- Focused Doxygen HTML generated with the project configuration contains the
  producer regression and all four new graph cases, with references from the
  real methods under test.

### Final verification

- After the documentation and formatting pass, all four focused Catch2 suites
  passed: zaberCtrl 101 assertions / 5 cases, stdMotionStage 12 / 1,
  stdMotionNode 1487 / 9, and xInstGraph 375 / 17 (1975 assertions / 32 cases total).
- Both xInstGraph and zaberCtrl built successfully from the final sources.
- `clang-format --dry-run --Werror` and `git diff --check` passed for the touched
  files. Final Doxygen HTML retained the real-API references for all five new
  regression cases.
- Rebuild and install xInstGraph and zaberCtrl for operational testing. No
  instGraph library changes were made. Confirm a physically parked powered-off
  stage has the same retained preset in stageGUI and the graph, with POWEROFF
  still shown, including an xInstGraph restart while the stage is powered off.
- Numeric-only positions without a named route and app-wide disconnect/deletion
  invalidation remain the explicit follow-up boundaries above.
