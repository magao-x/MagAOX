# xInstGraph parked routing through startup

## Request

Treat affirmative parking in POWEROFF, POWERON, NOTCONNECTED, and CONNECTED as
usable retained-position states, so power-on sequences do not switch graph
outputs off or bounce the display while the stage remains parked and stationary.

## Findings and implementation plan (2026-10-01)

- stdMotionNode's `parkedPowerOff()` currently gates retained routing, preset
  updates while tracking is cached, tracking priority, and numerical display.
  It permits only POWEROFF, so every subsequent pre-READY FSM message removes
  an otherwise valid parked route.
- Legacy routing has a separate POWEROFF branch ahead of tracking. Extending
  only the helper would leave new states subject to the tracking/READY gate.
  Put availability must give affirmative parking priority in all four states.
- Rename the helper to `parkedState()` and require `parkable=true`, valid nonzero
  `parked.current`, and exactly the requested FSM set. Apply it consistently to
  legacy/mapped/default routes, preset refresh, tracking priority, off labels,
  and numeric-display availability. Preserve the actual published FSM label.
  Centralize the four-state set in `parkedFSMState()` so legacy routing and label
  cleanup use the same set even when affirmative parking is withdrawn.
- Keep all preset/route validation, empty-row blocking, incoming-light waiting,
  and parking opt-in rules. Numeric-only fallback remains display-only. Other
  FSM states keep their existing behavior, and false/malformed parking still
  removes the retained-position route immediately.
- Extend initial message-order tests to all four states in both directions,
  with legacy single/multi-put and mapped handlers, including parking opt-out.
  Add sequential startup regressions for routes, cached tracking, numeric
  fallback, and real published graph colors/labels. Verify upstream light
  updates still propagate while the parked FSM advances.
- Run both affected Catch2 suites, a serialized app build, formatting, and focused
  Doxygen link checks. Commit functional work and this engineering record, then
  update the README and record the implementation hash.

## Scope and limits

The change is in xInstGraph's stdMotion handler. Controller parking telemetry
remains authoritative; no position is inferred from the FSM or a target value.
Property deletion/disconnect cache invalidation remains an existing app-wide
limitation. Qt stageGUI's own FSM gating and cameraStatus's broader parked-state
list are separate display implementations and are not changed by this request.

## Execution notes

The expanded initial-message-order regression found that withdrawing parking
in POWERON left the legacy multi-put `curLabel()` cached at its last preset,
although the graph's position extra was already cleared. The shared FSM-state
check now clears that cached label consistently in all four startup states.

- Implemented the common four-state check and applied affirmed parking to legacy,
  mapped, and default routing, numeric display, preset refresh with cached tracking,
  and unavailable-label cleanup. The FSM extra continues to report its actual state.
- Extended the legacy and mapped initial-message-order tests across all four
  states, in both directions, with both preset prefixes and parking opt-out.
- Sequential node tests confirm steady route masks/labels, preset changes while
  tracking flags are cached, numeric fallback, immediate loss of routing when
  parking clears or is malformed, and upstream waiting/restore propagation.
- The app regression advances two mapped stages in series through the sequence,
  checking every published position and port label/color after each individual
  FSM update. It also verifies default routes, known empty rows, numeric-only
  display, upstream light loss/restoration, and cleared parking.
- Catch2 passed: stdMotionNode 17964 assertions / 21 cases, xInstGraph 3184 / 24,
  totaling 21148 assertions / 45 cases. Existing regressions remain passing.
- `make -C apps/xInstGraph -j1`, clang-format, the changed-file documentation pass,
  `clang-format --dry-run --Werror`, and `git diff --check` passed.
- Focused Doxygen generation emitted no warnings for changed C++ files. HTML links
  from the real renamed parked helpers, routing, numeric-display, and app dispatch
  APIs to the relevant test cases were verified.
- Rebuild and install xInstGraph to deploy. Existing `parkable=true` settings use
  the extended behavior; no new configuration setting or library change is needed.
- Functional implementation, regressions, and engineering record committed as
  `32a5008a`. The documentation follow-up updates the supported parked FSM set,
  numeric-display availability, and tracking priority in the README.

## Follow-up: NODEVICE during startup (2026-10-05)

The user identified NODEVICE as another stage startup state during which the
stage can remain parked and stationary. Add NODEVICE to `parkedFSMState()`;
all retained routing, cached-tracking priority, numeric display, and label
cleanup already share this predicate. The supported set is now POWEROFF,
POWERON, NODEVICE, NOTCONNECTED, and CONNECTED. Affirmative valid parking and
`parkable=true` remain required, and the graph must report the actual NODEVICE
FSM label.

Extend existing initial-message-order and sequential startup regressions with
NODEVICE for legacy, mapped, and default routes, numeric display, and published
XML. Also check that NODEVICE stays blocked without affirmative parking or the
parking opt-in. Verification completed:

- stdMotionNode: 20642 assertions / 21 cases; xInstGraph: 3231 / 24,
  totaling 23873 assertions / 45 cases, all passing.
- `make -C apps/xInstGraph -j1` succeeded. The changed-file documentation pass,
  clang-format, `clang-format --dry-run --Werror`, and `git diff --check` passed.
- Focused Doxygen generation emitted no warnings for changed C++ files; real-API
  HTML links were verified for all four extended startup test cases.
- The README now lists NODEVICE for parked routing and numeric display. Rebuild
  and install xInstGraph; existing `parkable=true` configuration applies.
