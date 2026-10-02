# xInstGraph numerical position display

## Request

Display the numerical position in stdMotion when there is no valid preset,
following stageGUI and cameraStatus. Preserve the on/off logic for puts.

## Findings (2026-10-01)

- `stage::handleSetProperty()` caches `position.current` and `filter.current`.
  `stage::updateGUI()` shows positions in READY, OPERATING, HOMING, CONFIGURING,
  NOTHOMED, and affirmative parked POWEROFF. Other states show `---`.
- `stageStatus::formatValue()` and cameraStatus's overlay show four decimal places
  when the selected name is missing or `none`. cameraStatus checks `filter.current`
  before `position.current`. These are display rules, not route inference.
- stdMotionNode currently subscribes to named presets, FSM, optional parking, and
  optional tracking, with no numeric current-position subscription. Its `state`
  extra shows a selected route/tracking label or `---`. Legacy single-put nodes
  also use their selected put's label; mapped/multi-put port labels stay fixed.
- The app registers each handler key through its existing callback dispatch and
  atomically publishes the complete graph after each callback. Numeric-only
  callbacks can update labels through that path without calling route setters.

## Implementation contract

1. Subscribe to `<device>.filter` for `presetPrefix=filter`, otherwise
   `<device>.position`, using the device override. This chooses the controller's
   applicable GUI numeric property without subscribing to both mutually exclusive
   interfaces and generating unresolved-property notices. No new config is needed.
2. Cache the Number property's `current` independently of named presets, even
   while a valid preset or tracking label is visible. Parse the complete number;
   malformed/current wrong-type updates invalidate the cache. Target-only updates
   do not replace the cached current value. No value is invented before telemetry.
3. A usable name is exactly one selected nonempty Switch element other than
   `none`. Missing, `none`, wrong-type, or ambiguous selections use the numerical
   fallback when available. A valid name without a mapped route retains the
   existing route display behavior; route validity is separate from selection.
4. Use four decimal places and the stageGUI display states above, with existing
   `parkable` opt-in for POWEROFF. Preserve the actual FSM. Existing tracking and
   not-tracking labels keep priority. Numeric position never identifies a route.
5. Update `state` and legacy single-put labels only. Preserve multi-put and mapped
   port labels. Numeric callbacks must not change put states or enablement.
   Existing preset/FSM/parking/tracking callbacks retain their routing decisions.
6. Add Catch2 node regressions for label transitions, caching, telemetry validation,
   both numeric properties, message ordering, display availability, and unchanged
   routing compared with a handler that receives no numeric telemetry. Add an app
   regression through Def/Set dispatch that checks published labels and colors.
7. Complete changed-file docs, clang-format, both affected suites, a serialized
   app build, and focused Doxygen links. Commit functional work with this record,
   then update the README in a documentation commit.

## Limits and execution notes

- Numeric display uses the controller's published user units, with no unit suffix
  or position-to-preset conversion. `filter.current` may represent a filter index.
- The configured preset prefix chooses the numeric source; this change does not
  discover alternate numeric properties dynamically.
- Property deletion and connection loss currently do not invalidate handler caches;
  this display shares the existing freshness limitation. Unavailable FSM states
  still suppress the numeric label.

## Execution and verification

- Implemented the numeric subscription, independent current-position cache, and
  label-only fallback in stdMotionNode. Existing route predicates and put setters
  retain their logic. Numeric callbacks return before the routing callback path.
- Node regressions compare put states and enablement against otherwise identical
  handlers receiving no numeric telemetry, covering both directions, legacy
  single/multiple puts, explicit/default mapping, upstream changes, and parking.
  Additional checks cover cached updates behind preset/tracking labels, malformed
  values, target-only updates, wrong-device messages, and all display states.
- App regression checks real Def/Set dispatch, the device override and numeric
  property choice, all six initial FSM/preset/position message orders in READY,
  OPERATING, and parked POWEROFF, plus publication and mapped port-name retention.
- Catch2 passed: stdMotionNode 9164 assertions / 20 cases, xInstGraph 2966 / 23,
  totaling 12130 assertions / 43 cases. Existing routing and publication tests
  also remain passing.
- `make -C apps/xInstGraph -j1` completed successfully. The full changed-file
  documentation pass, clang-format, `clang-format --dry-run --Werror`, and
  `git diff --check` are complete.
- Focused Doxygen generation emitted no warnings for the changed C++ files. All
  three new test cases have verified HTML links from the real numeric-display,
  property-handler, and app dispatch APIs.
- Only xInstGraph requires rebuilding and installation. No controller, library,
  or deployed configuration changes were made for this upgrade.
- Functional implementation, regressions, and engineering record committed as
  `18ff352e`. The documentation follow-up describes numeric-source selection,
  formatting, availability, caching, and its independence from routing.
