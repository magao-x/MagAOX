# Task Description
<!-- Drafted by the agent from the user's prompt on 2026-11-19; the user should correct anything here. -->

## Problem Statement
Finish the `qwpTracker` app with a dummy tracking law. The app reads the K-mirror angle from `stagek` and commands the
two QWP rotation stages (`stageqwprot1`, `stageqwprot2`). `qwpTracker` owns the conversion between QWP fast-axis
polarization angle and mechanical stage angle (zero point and offset), so that `stageqwprot1/2` are purely mechanical
stages and `qwptrack` deals only in polarization angles.

## Discussion
- The real tracking law (QWP angles as a function of K-mirror/derotator angle, and possibly telescope pointing) is not
  yet known. This task delivers the app plumbing with a placeholder law that can be swapped out later.
- Reference apps: `apps/kTracker` (cleaned-up tracking loop with `m_indiMutex`, `m_lastUpdate`, `sendNewProperty`),
  `apps/hwpTracker` (zero/sign mapping between polarization and stage angle), `apps/adcTracker`.
- The QWP stages are ELL21 mounts on a shared Elliptec bus, driven by `elliptecLowLevel` (`elliptecqwp`) and one
  `elliptecCtrl` per mount (see `agents/plans/2026-11/elliptec-lowlevel-split.md`).

## Requirements
1. Subscribe to the K-mirror angle from `stagek`.
2. Compute QWP1 and QWP2 fast-axis angles from a (dummy) tracking law.
3. Convert fast-axis angles to stage angles using a per-QWP zero point and sign, and command the stages.
4. Report the current fast-axis angles from the stage positions using the inverse mapping.
5. Allow manual fast-axis targets when not tracking.
6. Telemetry via `telem_qwptrack`.

## Tests and Metrics
- Angle mapping round-trips (fast axis -> stage -> fast axis) for both signs and arbitrary zeros, including wrapping.

## Scope and Caveats
- No real tracking law in this task.
- No changes to `elliptecCtrl`/`elliptecLowLevel` beyond what is needed to interoperate.

# Instructions to Agent

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan

## Task Summary
`apps/qwpTracker` exists on `ml/qwpTracker` (commits `a7862261`..`5f78f577`) with config, INDI properties, and
`telem_qwptrack`, but it cannot work yet. Finish it so that, while tracking, every update computes fast-axis targets
from the latest `stagek` angle with a dummy law, maps them to stage angles, and sends them to `stageqwprot1/2`; when not
tracking, operators set fast-axis targets directly. `qwpTracker` is the only place that knows the fast-axis zero/sign.

## Current State (review of `apps/qwpTracker/qwpTracker.hpp` at `5f78f577`)
Defects that must be fixed for the app to function:
1. **Commands never reach the stages.** `updateQwpStages()` writes the stage angle into the app's own `qwp1`/`qwp2`
   properties and calls `sendNewProperty` on them (lines 360-361, 373-374), instead of targeting the stage device.
   It also first publishes the fast-axis target to the same element, then overwrites it with the stage angle.
2. **Wrong stage property.** It subscribes to `<stage>.position`, which `elliptecCtrl` does not publish. Its absolute
   position is `absDeg` (`current`/`target`, 0-360, `apps/elliptecCtrl/elliptecCtrl.hpp:375`).
3. **Wrong device names.** Defaults are `stageqwp1rot`/`stageqwp2rot`; the deployed stages are `stageqwprot1/2`.
4. **FSM fight.** Both stage `fsm` callbacks call `state(...)` on the tracker, so the tracker's state flips between the
   two stages' states.
5. **No wrapping.** `sign*(tgt - zero)` can be negative or >= 360; `absDeg` accepts only [0, 360].
6. **Tracking on at startup** (`m_tracking{true}`) with no K-mirror angle yet received: the app would command the
   stages immediately on start.
7. **No locking.** `appLogic` reads state shared with INDI callbacks without `m_indiMutex` (cf. `kTracker::appLogic`).
8. Function-local `static` state in `appLogic` and `recordQwpTrack` (should be members, as in `kTracker`).
9. Turning tracking off re-sends the stale tracking target.
10. Test file is the template placeholder; no `groups.dox` entry.
11. Docs: missing `///` on many members/methods, class brief says "ADC Tracker", `stagek` doc mentions `tcsi`.

## Key Assumptions
- K-mirror angle is `stagek.position.current` (mechanical degrees), the property `kTracker` commands.
- QWP stage absolute position is `elliptecCtrl` `absDeg` (`current`/`target`, degrees, 0-360); `absDeg` accepts a new
  target while the stage is `OPERATING` (retarget), checked in `elliptecCtrl.hpp` `m_ipAbsDeg` callback.
- Stage FSM is the standard `<stage>.fsm.state`.
- ELL21 absolute moves do not take the shortest path across 0/360 (hence decision 6).
- A QWP is 180-degree periodic and both halves of the mount are optically equivalent.

## Requirements
1. Subscribe `stagek.position`, `stageqwprotN.absDeg`, `stageqwprotN.fsm` with `REG_INDI_SETPROP`; no polling.
2. `computeTargets(k)` holds the dummy law; pure mapping helpers (fast axis <-> stage, wrap, nearest equivalent).
3. Stage targets sent with `sendNewProperty( <stageN absDeg>, "target", deg )` from `appLogic`, outside the mutex.
4. Tracking/manual exclusive modes, event-driven tracking, pending sends, derived FSM, startup off (decisions 3-5, 8).
5. Config: `[qwp1]`/`[qwp2]` `zero`, `sign`, `devName`; `[k] devName`; telemeter config. Read-only INDI for zeros.
6. Telemetry unchanged, recorded on change of current angles or tracking.

## Questions and Points of Clarification
(Being resolved with the user; answers recorded here.)

1. **Fast-axis calibration (agreed 2026-11-19):** one `zero` and one `sign` per QWP, `stage = zero + sign * fastAxis`
   (hwpTracker convention). No separate calibration offset. `elliptecCtrl` `homeOffset` remains purely mechanical
   (post-home park position) and is not part of the fast-axis calibration (assumed; user did not object).
2. **Dummy law (agreed 2026-11-19):** hard-coded placeholder in `computeTargets(k)`: `qwp1 = 2k`, `qwp2 = -2k`
   (fast-axis degrees from K-mirror mechanical degrees). No law coefficients in config or INDI; the real law will
   replace the body of `computeTargets`.
3. **Modes (agreed 2026-11-19):** tracking and manual are exclusive. Tracking: `qwpN.target` publishes the law output;
   new `target` requests are rejected and logged. Manual: a new `target` (fast axis) is mapped and sent immediately.
   Tracking off: no move; `qwpN.target` is set to the current fast-axis angle.
4. **Update trigger (agreed 2026-11-19):** event-driven on `stagek.position.current`. The callback stores `k` and flags
   it; `appLogic` computes targets under `m_indiMutex` and sends after releasing it, so a burst of K updates sends
   only the latest. No deadband; a stage is re-sent only if its computed stage target differs from the last one sent
   (exact comparison). `tracking.updateInterval` is removed (assumed).
5. **Tracker FSM (agreed 2026-11-19):** the tracker keeps its own state from both stages' `fsm`:
   `OPERATING` if tracking or either stage is `OPERATING`; else `READY` if both stages are `READY`; else the worse of
   the stage states (`NOTCONNECTED` for unknown/`POWEROFF`/`NODEVICE`/`NOTCONNECTED`, then `NOTHOMED`/`HOMING`).
   A stage is commanded only when `READY` or `OPERATING`; otherwise its target stays pending and is sent when it
   becomes ready. Tracking sends nothing until a first `stagek` value has been received.
6. **Wrapping (agreed 2026-11-19):** use the QWP 180-degree equivalence. For a fast-axis target, the candidate stage
   angles are `s = wrap360(zero + sign*theta)` and `wrap360(s + 180)`; send the one nearest the stage's current position
   (circular distance), so moves are <= 90 degrees. Fast-axis `current` and `target` are reported folded to [0, 180).
   Caveat: assumes both halves of the mount are optically equivalent.
7. **Telemetry (agreed 2026-11-19):** keep `telem_qwptrack` unchanged (`qwp1_angle`, `qwp2_angle` current fast axis,
   `tracking`); record state moves from `recordQwpTrack` statics to members.
8. **Startup (agreed 2026-11-19):** tracking starts off (as in kTracker, hwpTracker, adcTracker); nothing is sent at
   startup; `qwpN.target` is initialized to the current fast-axis angle on the first stage position.
9. **Calibration config (agreed 2026-11-19):** `[qwp1]`/`[qwp2]` `zero`, `sign` are config-only, published read-only
   over INDI for visibility. Naming: device defaults `stageqwprot1`/`stageqwprot2`; `[imr] devName` renamed
   `[k] devName` (default `stagek`); app properties `qwp1`/`qwp2` (`current`/`target`) and `tracking` kept.

10. **External stage moves (agreed 2026-11-19):** not corrected; the next `stagek` change re-applies the law. The
    `qwpN` `current`/`target` mismatch shows it.

## Tests
`apps/qwpTracker/tests/qwpTracker_test.cpp`, `namespace libXWCTest { namespace qwpTrackerTest {...} }`,
`\defgroup qwpTracker_unit_test` `\ingroup application_unit_test` in `tests/groups.dox`, a brief Doxygen block per
`TEST_CASE`, `QWPTRACKER_TEST_DOXYGEN_REF` raw member references where wrappers hide the API (AGENTS.md 19-20).
Built with `QWPTRACKER_TEST_NOSEND`, so stage commands are recorded (device, value) instead of sent.
1. Mapping: `stage = wrap360(zero + sign*theta)` and inverse fold to [0, 180) round-trip for sign +/-1, several zeros,
   thetas near 0/180/360 and negative.
2. Nearest equivalent: chooses `s` or `s+180` by circular distance from current, including across 0/360; move <= 90.
3. `computeTargets`: `qwp1 = 2k`, `qwp2 = -2k` for a few k (pins the placeholder so a law change is deliberate).
4. Config: defaults (`stageqwprot1/2`, `stagek`, zero 0, signs) and overrides via `loadConfigImpl`.
5. Callbacks: injected `stagek.position.current` stores k and flags; injected `absDeg.current` gives folded fast axis;
   first stage position initializes `target`.
6. Tracking: with both stages READY and tracking on, a K update -> `appLogic` sends the expected stage angles once;
   a repeated identical K sends nothing; no K yet -> nothing sent.
7. Gating: stage not READY -> target pending, sent when its `fsm` becomes READY; tracking off at startup sends nothing.
8. Modes: manual target rejected while tracking; accepted and sent when not tracking; tracking off -> no send,
   `target` = current.
9. FSM: READY only when both READY; OPERATING when tracking or either stage OPERATING; NOTCONNECTED/NOTHOMED otherwise.

## Implementation Plan
Branch `ml/qwpTracker` (exists, follows `<username>/<feature>`). Header-only app; `.cpp` stays main only.
1. Members: `m_k`, `m_haveK`, `m_kUpdated`; per QWP: `zero`, `sign`, `devName`, fast-axis `current`/`target`,
   stage `current`, stage FSM state, `m_pendingN` / last sent stage angle; telemetry last-recorded values (replacing
   statics). Remove `m_updateInterval`, `m_kmirror`.
2. Helpers (static or free inline in the header): `wrap360`, `fold180`, `fastToStage`, `stageToFast`,
   `nearestEquivalent`. `computeTargets(k, t1, t2)` with the dummy law.
3. `appStartup`: `REG_INDI_SETPROP` for `stagek.position`, `absDeg`/`fsm` per stage; outbound `absDeg` properties
   built like `kTracker::m_indiP_kpos` (`setDevice`, `setName("absDeg")`, element `target`); app `tracking`,
   `qwp1`/`qwp2`, read-only `qwp1_zero`/`qwp2_zero`; telemeter. Initial state from stage states (NOTCONNECTED).
4. `appLogic`: lock `m_indiMutex` (try_lock as kTracker); if tracking and `m_kUpdated`, compute targets, publish
   `qwpN.target`, mark sends pending; for each pending stage that is READY/OPERATING, choose the nearest equivalent
   stage angle, skip if equal to last sent; release lock; send; derive and set FSM; telemetry.
5. Callbacks: `stagek` stores k; `absDeg` updates stage/fast-axis current and (once) target; `fsm` stores stage state;
   `qwpN` new-target per decision 3; `tracking` toggle per decisions 3/8. Fix the misspelled `m_indiP_stagewp1rotFsm`.
6. Remove `std::cerr` debug prints; log targets with `text_log` only on mode changes and manual sends.
7. Commits: (1) functional + tests + this plan, (2) Doxygen pass (file block, class brief, `///` on all members and
   declarations, inline param docs, `groups.dox`), (3) `clang-format` with `{ //mutex scope` restored.
8. Provide an example `qwptrack.conf` to the user (not committed here; config repo).

## Follow-up and Edge Cases
- Real tracking law replaces `computeTargets` body (and may need telescope inputs, e.g. `tcsi.teldata`).
- Handbook page `handbook/operating/software/apps/qwpTracker.html` lives outside this repo.
- If the two halves of a mount turn out not to be equivalent, switch decision 6 to a single-half wrap.
- External direct stage moves are not corrected until the next `stagek` change (decision 10).
- If `stagek` publishes very frequently during a K move, the ELL21s will be retargeted each `appLogic` loop; revisit
  only if bus traffic becomes a problem.

## Implementation Status (2026-11-19)
- `apps/qwpTracker/qwpTracker.hpp` rewritten per decisions 1-10. Per-QWP state lives in `qwpStage m_qwp[2]`
  (calibration, stage state/position, fast-axis current/target, pending flag, last command, outbound `absDeg`
  property). Static helpers `computeTargets`, `wrap360`, `fold180`, `stageToFast`, `fastToStage`, `trackerState`.
  `updateStages()` is the single send path (from `appLogic`, manual targets and tracking-on); it selects commands under
  `m_indiMutex` and sends after releasing it. Manual requests and tracking-on clear `haveSent` so they are always sent.
- Config: `[qwp1]`/`[qwp2]` `zero`, `sign` (must be +/-1, else shutdown), `devName`; `[k] devName`.
  `tracking.updateInterval` and the unused gsl includes/link were removed.
- Tests: `tests/qwpTracker_test.cpp` (mapping, nearest equivalent, law, FSM, config, callbacks, tracking, pending
  sends, stop, manual modes; built with `QWPTRACKER_TEST_NOSEND`) and `tests/qwpTracker_indi_test.cpp` (callback
  validation, separate because `testMacrosINDI.hpp` changes callback behavior for the whole translation unit).
  Both pass, and the app builds, in the `magao-x-setup:cli` container with gcc-toolset-14.
- `updateIfChanged` is a no-op without an INDI driver, so tests assert on `m_qwp[n].fastTgt`/`fastCur` rather than the
  published properties.
- Example `qwptrack.conf`:
  ```ini
  [qwp1]
  zero=0
  sign=1
  devName=stageqwprot1

  [qwp2]
  zero=0
  sign=-1
  devName=stageqwprot2

  [k]
  devName=stagek
  ```
