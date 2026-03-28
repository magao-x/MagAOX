Task: review AGENTS.md then consider: in zaber stage control we have a recently added feature implementing parking.  This allows a zaber stage to maintain its position through power-offs.  GUIs need to catch up so that if a stage is in POWEROFF but parked=true, the GUIs report the stage position and/or associate preset that should be reported by zaberCtrl.  This will affect the cameraGUI, stageGUI, and the rtimv cameraStaus plugin.  Please make a plan to update these guis and append it below:

Plan:

## Analysis

The current GUI stack does not have any parked-state handling:

- `gui/widgets/stage/stage.hpp` disables the stage display entirely unless the FSM is `READY`, `OPERATING`, `CONFIGURING`, `NOTHOMED`, or `HOMING`.
- `gui/widgets/xWidgets/statusCombo.hpp` and `gui/widgets/xWidgets/statusDisplay.hpp` only show the live value when the FSM is `READY` or `OPERATING`; otherwise they replace the value with the FSM text.
- `gui/widgets/xWidgets/stageStatus.hpp` depends on `statusCombo`, so `cameraGUI` inherits that same behavior for camera-associated stages.
- `gui/rtimv/plugins/cameraStatus/cameraStatus.cpp` only resolves filter/stage presets when `fsm.state` is `READY` or `OPERATING`; otherwise it prints the raw FSM state.

So today a parked Zaber stage that transitions to `POWEROFF` will be rendered as unavailable even if `zaberCtrl` still has a meaningful current position and/or preset association to report.

There is also an interface gap: neither the shared GUI widgets nor `cameraStatus` currently subscribe to any `parked` property from `zaberCtrl`, and `apps/zaberCtrl/zaberCtrl.hpp` does not currently relay a parked status from the low-level Zaber app. The GUI work therefore needs a small controller-contract update as part of the plan so the front ends can distinguish:

- `POWEROFF` and parked
- `POWEROFF` and not parked

## Recommended Behavior

For Zaber-backed stages, when:

- `fsm.state == POWEROFF`
- and `parked == true`

the GUIs should continue to show the last controller-reported logical stage value instead of replacing it with `POWEROFF`.

That means:

- `stageGUI` should continue to show the parked position or preset name, but keep all motion controls disabled.
- `cameraGUI` stage summary widgets should continue to show the parked preset/position text while still indicating the powered-off state through the FSM widget.
- `rtimv` `cameraStatus` should continue to print the associated filter/stage preset line when available, and fall back to numeric position only if no preset is selected.

If `parked != true`, the current `POWEROFF` behavior should remain unchanged.

## Scope

### 1. Add or verify the parked-state contract in `zaberCtrl`

In `apps/zaberCtrl/zaberCtrl.hpp`:

- verify that the controller preserves the last meaningful `position`/`filter` and `presetName`/`filterName` values when the low-level stage goes to `POWEROFF`
- add a relayed `parked` property from the low-level Zaber app if one is not already exposed by `zaberCtrl`
- subscribe to the low-level parked status alongside the existing stage-state/raw-position subscriptions
- keep the parked indication device-local to the stage controller so all GUIs can consume one consistent interface

This is the minimum controller-side change needed to let the GUIs recognize the parked-poweroff case without special-casing low-level app names.

### 2. Update the shared stage widgets used by `stageGUI` and `cameraGUI`

In `gui/widgets/stage/stage.hpp`:

- subscribe to the stage `parked` property
- store the parked flag in widget state
- treat `POWEROFF && parked` as a displayable state rather than a disconnected/unavailable state
- keep position text and selected preset visible in that state
- continue to disable motion commands, homing, slider movement, and setpoint submission while powered off

In `gui/widgets/xWidgets/statusCombo.hpp` and `gui/widgets/xWidgets/statusDisplay.hpp`:

- extend the show-value logic to allow value display for `POWEROFF && parked`
- preserve current behavior for all other non-ready states

In `gui/widgets/xWidgets/stageStatus.hpp`:

- subscribe to and pass through the parked state via the shared status widget path
- confirm the fallback numeric formatting still works when no preset is active but the parked position is known

Because `cameraGUI` stage rows are built from `stageStatus`, fixing the shared widgets should cover both `stageGUI` and the camera-side stage summaries with one implementation.

### 3. Update the `rtimv` `cameraStatus` plugin

In `gui/rtimv/plugins/cameraStatus/cameraStatus.cpp`:

- subscribe to each filter/stage device `parked` property in addition to `fsm.state` and preset property blobs
- treat `POWEROFF && parked` as eligible for preset lookup, just like `READY` and `OPERATING`
- if no preset switch is active, optionally show the parked numeric position if that is already available in the plugin dictionary; otherwise keep the existing state text fallback

This keeps the overlay aligned with the standalone GUIs instead of regressing to `device: POWEROFF` for parked stages.

### 4. Decide the exact fallback hierarchy once and apply it consistently

Use one display rule across all three GUI surfaces:

1. show active preset/filter name if one is selected
2. otherwise show numeric current position/filter index if available
3. otherwise show FSM state text

For parked stages in `POWEROFF`, the same hierarchy should apply, but controls must remain disabled.

## Verification Plan

### 1. Interface verification

Confirm `zaberCtrl` publishes all of the data the GUIs need in the parked-poweroff case:

- `fsm.state`
- `parked`
- current position/filter value
- preset/filter-name switch property

### 2. GUI behavior verification

Check the following cases in both `stageGUI` and `cameraGUI`:

1. `READY`, unparked: current behavior unchanged.
2. `POWEROFF`, `parked=false`: current behavior unchanged and value hidden.
3. `POWEROFF`, `parked=true`, preset selected: preset remains visible and controls are disabled.
4. `POWEROFF`, `parked=true`, no preset selected: numeric position remains visible and controls are disabled.

### 3. Overlay verification

In `rtimv` `cameraStatus`, confirm:

1. parked powered-off filter wheels/stages still render the preset line
2. unparked powered-off devices still render `device: POWEROFF`
3. no duplicate or contradictory lines are introduced when state transitions between `READY`, `OPERATING`, and `POWEROFF`

## Execution Order

1. Inspect and, if needed, extend `zaberCtrl` to publish a GUI-consumable `parked` status while preserving last meaningful position/preset values.
2. Update the shared stage/status widgets in `gui/widgets` so parked powered-off stages continue to display values but remain read-only.
3. Update `gui/rtimv/plugins/cameraStatus` to apply the same parked-poweroff display rule.
4. Build the affected GUIs/plugins and manually verify the state combinations above.

## Open Question

The one point to confirm during implementation is whether `zaberCtrl` should expose `parked` as a dedicated top-level property or fold it into an existing status property. A dedicated `parked` property is the cleaner choice because the existing GUI code already reasons about FSM state separately from displayability, and this avoids overloading the meaning of `POWEROFF`.
