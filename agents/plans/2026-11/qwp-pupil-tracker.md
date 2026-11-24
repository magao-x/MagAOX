# Task Description

## Problem Statement

Rotating the dual-QWP compensator (`qwptrack`, stages `stageqwprot1` / `stageqwprot2`) shifts the MagAO-X pupil on
`camwfs`. A new app should keep the pupil aligned by reacting to the QWP angles, predicting the pupil shift relative
to a reference state, and commanding `ttmperi` to compensate.

## Discussion

- Reference state: `qwp1 = 90`, `qwp2 = 115` (degrees).
- The shift model is a harmonic series fitted in `~/research/magaox/20261114_wfs_shifts` (44 terms,
  `shift_ax = sum_j c_ax[j] * fn_j(a_j*phi1 + b_j*phi2)`), giving the shift in camwfs pixels. Reference evaluator:
  `src/project/evaluator.py`; model: `data/outputs/20261114_qwp_wfs_shift_model.json`; test vectors:
  `data/outputs/20261114_qwp_wfs_shift_model_test_vectors.csv`.
- The ttmperi -> camwfs interaction matrix and control matrix come from `~/research/magaox/20261115_ttmperi_camwfs`
  (most recent fit, `data/outputs/ttmperi_camwfs_im.json`, created 2026-11-18). `[dx,dy] = IM @ [v1,v2]`; correction
  `dv = -CM @ [dx,dy]`.
- The app should use minimal INDI properties.
- The harmonic coefficients and the CM/IM are stored as two separate calibration files in the MagAO-X calib directory.

## Requirements

1. React to the QWP angles and compute the pupil shift relative to the reference state.
2. Convert the shift to ttmperi commands with the CM.
3. Minimal INDI properties.
4. Two calib files: harmonic series; CM/IM.

## Tests and Metrics

- The C++ evaluator matches the research test vectors to ~1e-12 px.

## Scope and Caveats

- The shift model is not yet validated (predicted 0.02 px RMS vs a 0.01 px target).

# Instructions to Agent

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan

## Task Summary

A new app (working name `qwpPupilTracker`, branch `ml/qwp-pupil-tracker`) applies feed-forward to `ttmperi` from the
QWP stage angles: `dv = -CM * (shift(phi) - shift(phi_ref))`.

## Key Assumptions

- The QWPs are upstream of the ttmperi pupil plane, so ttmperi can cancel the QWP-induced shift for the whole lower
  bench (decision 1).
- ttmperi is aligned (J-test) at whatever QWP angles are current when tracking is enabled (decision 4).
- The shift model applies to the operational camwfs configuration (REPORT.md caveat) and is used as-is, unvalidated.
- The ttmperi CM is linear over the range used; hysteresis (~0.08 px) is accepted.

## Requirements

1. Subscribe to `stageqwprot1` / `stageqwprot2` `position` and `fsm`, and `ttmperi` `axis1_voltage`, `axis2_voltage`, `fsm`.
2. Evaluate `shift(phi1, phi2)` from the harmonic model in mechanical stage angles; `offset = shift(phi) -
   shift(phi_ref)`, with `phi_ref` the stage angles at which tracking was enabled (see the 2026-11-20 amendment).
3. On enable, capture `v0 = v_now + CM * offset(phi_now)`; while tracking, command `v = v0 - CM * offset(phi)` on
   every stage position update; re-base `v0` on external ttmperi target changes.
4. INDI: `tracking` toggle and read-only `offset` (`dx`, `dy`, `v1`, `v2`) only.
5. Telemetry of offset and commanded voltages via `dev::telemeter`.
6. Load the model CSV and CM FITS from the calib dir at startup; fail loudly if missing or malformed.
7. Edge behaviour per decision 10.
8. Exporter script producing both calib files from the research JSON, self-verified against the research evaluator.

## Questions and Points of Clarification

Findings from the codebase and research directories that drive the questions:

- `qwptrack.qwp1/qwp2` (branch `ml/qwpTracker`) publish **fast-axis** angles folded to [0, 180). The model takes
  **mechanical stage** angles, and q vs q+180 differ by 0.55-0.70 px in shift (REPORT.md). The fast-axis angle alone
  cannot drive the model.
- qwpTracker's convention is `stage = zero + sign * fast`; the model JSON's `stage_convention` is
  `phi = sign * (q - zero)` with qwp1 `zero=-67.42, sign=+1`, qwp2 `zero=280.8, sign=-1`. These need reconciling.
- REPORT.md says the pupil loop should *follow* the shift (offset its reference by +Δ), not cancel it. The task
  instead cancels it with ttmperi. The optical intent needs pinning down.
- No JSON library is used in the repo. Calib inputs are CSV with `#` comments (`calib/qwptrack/qwp_law_*.csv`) or
  FITS matrices (`alignLoop` reads its IM with `mx::fits`).
- `ttmperi` is a `kcubeCtrl`, with `axis1_voltage` / `axis2_voltage` (elements `current` / `target`) in volts, clamped to 0..150 V.
- ttmperi hysteresis is 0.07-0.085 px (10-13% of full scale), larger than the 0.01 px model target.

All questions were resolved in the design interview (2026-11-20).

### Resolved

1. **Optical intent: cancel with ttmperi.** The QWPs are upstream of the ttmperi pupil plane and shift the pupil for
   the whole lower bench (camwfs, dmncpc, coronagraph). ttmperi re-centres it with `dv = -CM * Δ`, so the camwfs
   pupil-tracking loop (`alignLoop`, camwfs -> ttmpupil) never sees the QWP-induced shift. REPORT.md's "follow"
   framing does not apply to this configuration.
2. **Inputs: mechanical stage angles.** Subscribe to `<stage>.position` (and `fsm`) for `stageqwprot1` and
   `stageqwprot2` (device names configurable), not `qwptrack.qwp1/qwp2`, which are fast-axis angles folded to
   [0, 180) and cannot resolve the q / q+180 branch (0.55-0.70 px apart).
3. **Reference state.** The reference is the fast-axis pair `(90, 115)` as defined by the calibration script
   (`~/dev/magaox/scripts/take_qwp_wfs_calibrations.py`), which maps polarization angle over a full 360 to stage
   with `stage = (sign * (q - zero)) mod 360`, `QWP1_ZERO = -67.42, QWP1_SIGN = +1`, `QWP2_ZERO = 280.8,
   QWP2_SIGN = -1`. That gives a unique mechanical reference `phi_ref = (157.42, 165.80)`, confirmed by the
   reference rows of `20261114_qwp_wfs.csv` (`qwp1_stage = 157.40`, `qwp2_stage = 165.79`). The model shift there is
   `(+0.359, -0.520)` px. (Amended after implementation: the app config stores the reference directly as mechanical
   stage angles, default `(157.42, 165.80)`, with no zero/sign convention keys. This avoids confusion with
   qwpTracker's differently-defined `zero` / `sign`.) Live inputs stay mechanical, so a stage sitting
   on the other 180-degree branch (qwpTracker picks the nearest equivalent) is handled by the model.
   **Amended 2026-11-20 after on-instrument testing: the reference is captured at enable, not configured.** With the
   fast axes at (90, 115), qwpTracker put QWP2 on the +180 branch, at stage (157.42, 345.80). The model shift there
   differs from (157.42, 165.80) by (-0.37, +0.53) px, so `offset` read that instead of zero right after aligning,
   contradicting the procedure (J-test, insert QWPs, realign with ttmperi, enable tracking, offsets relative to that
   position). The commanded voltages were unaffected, since the fixed reference cancels in `v0 - CM * offset`, but
   the readout was misleading. Now `phi_ref` is the stage angles at enable, `v0` is the ttmperi targets at enable,
   and while not tracking the reference follows the stages, so `offset` reads zero. The `reference.qwp1/qwp2` config
   keys are removed.
   Note: qwpTracker's own convention is `stage = zero + sign * fast`, which differs in form from the script's for
   QWP1 (equivalent zero = +67.42). The calib file states its own convention explicitly to avoid confusion.
4. **Command model: baseline captured at enable, then absolute.** When tracking is enabled, ttmperi is assumed
   aligned at the current QWP angles (the workflow is: insert QWP, J-test, enable tracking). The app backs out
   `v0 = v_now + CM * offset(phi_now)`, the voltage that would be aligned at the reference, then commands
   `v = v0 - CM * offset(phi)`, with `offset(phi) = shift(phi) - shift(phi_ref)`. Subscribe to `ttmperi.axis1_voltage` and
   `axis2_voltage` (`kcubeCtrl`, elements `current` / `target`, volts). If a `target` changes to something other than our
   own last command (beyond a small tolerance), someone moved ttmperi externally (e.g. a J-test while tracking), and
   `v0` is re-based from the new target. Using `target` rather than `current` avoids re-basing on read-back noise.
5. **Trigger: every stage position update.** Recompute and command ttmperi on each `position` update from either
   stage, using the latest value of the other, so the pupil follows during moves. No READY gating and no deadband
   (user choice). Implementation note: skip sending when the new voltage equals the last command (identical
   values), and verify that `elliptecCtrl` actually publishes `position.current` during moves. If it publishes
   only on stop, this behaves like READY gating.
6. **No extra guardrails.** No max-offset hold, validity-box checks, or branch handling in this app. Voltage limits
   are left to `kcubeCtrl`, which clamps to 0..150 V (a clamped command echoes off-target and re-bases that axis). (For the record, the nominal-branch correction is at most 15 V over the operating box
   and 10 V along the tracking law. A stage on the +180 branch can need 38-70 V.)
7. **INDI properties.** `tracking` (new toggle; enabling captures `v0`) and read-only `offset` with elements
   `dx`, `dy` (camwfs px, `shift(phi) - shift(phi_ref)`) and `v1`, `v2` (V, `-CM * offset`, i.e. the commanded
   voltage relative to `v0`). Nothing else. Telemetry via `dev::telemeter` (`TELEMETER_*` macros) recording the
   offset and the commanded voltages (proposed; user did not object).
8. **Calib files** in `calib/qwppupil/` (paths configurable relative to the calib dir):
   - `qwp_pupil_model.csv`: one row per frequency, `a, b, dx_cos, dx_sin, dy_cos, dy_sin` (22 rows from the 44-term
     model), `#` comment header with provenance, formula and units. Read with
     `mx::ioutils::readColumns<readColCommaDelim>` as in qwpTracker.
   - `qwp_pupil_cm.fits`: 2x2 CM (V/px, rows `v1, v2`, cols `dx, dy`), read with `mx::fits` as in `alignLoop`.
     Provenance (source JSON, input sha256, date) goes in FITS header cards. The IM is not needed by the app; it can
     be kept as header cards for the record. Matrix orientation through `mx::fits` / Eigen must be verified by test.
   - ~~The reference goes in the app config as mechanical stage angles.~~ Superseded: captured at enable (decision 3).
   - An exporter script writes both files from the research JSON outputs, for reproducibility.
9. **Naming and structure.** App `qwpPupilTracker` in `apps/qwpPupilTracker/`, header-only pattern, separate from
   `qwpTracker`. Base `MagAOXApp` + `dev::telemeter` (power-state template argument to match the template app /
   hwpTracker). Config device name `qwppupil`. Config keys: `stages.qwp1 = stageqwprot1`,
   `stages.qwp2 = stageqwprot2`, `ttm.device = ttmperi`, `ttm.rebaseTol = 0.01`, `calib.model = qwppupil/qwp_pupil_model.csv`,
   `calib.cm = qwppupil/qwp_pupil_cm.fits`. Exporter: `~/dev/magaox/scripts/export_qwp_pupil_calib.py`, writing
   to `~/dev/magaox/calib/qwppupil/`.
10. **Edge behaviour.**
    - Tracking off: stop commanding, leave ttmperi where it is (no restore to `v0`).
    - Startup: tracking always off, never auto-enabled. Enabling is refused (logged) unless both stage positions
      and the ttmperi `target` are valid.
    - Input invalidated while tracking (device restart, disconnect): stop commanding, keep tracking on and `v0`,
      resume when values return. Silence is not staleness (AGENTS.md rule 27). A ttmperi restart that returns a
      different `target` re-bases under decision 4.
    - ttmperi `fsm` not usable: hold, then send the latest command once usable. (Amended during implementation:
      `kcubeCtrl` reports OPERATING when both axes are enabled and accepts `axisN_voltage` commands in READY or OPERATING, so READY and
      OPERATING are both usable, for the stages too, since positions during moves are applied.)
11. **Tests: no research data copied into the repo.** Unit tests write a small synthetic model CSV and CM FITS to a
    temp dir and check against analytic values. Conformance of the real calib files to the research evaluator is
    checked by the exporter script (it re-evaluates the written CSV against `evaluator.py` on the research test
    vectors and fails if any row differs by more than 1e-9 px).
12. **Docs and config scope.** No handbook updates. The user writes the instrument config on the instrument.

## Tests

`apps/qwpPupilTracker/tests/qwpPupilTracker_test.cpp` (logic) and `qwpPupilTracker_indi_test.cpp` (callback
validation with `testMacrosINDI.hpp`), grouped under `application_unit_test` per AGENTS.md rules 19-20. No research
data is copied; tests write synthetic calib files to a temp dir.

1. Model evaluation: a synthetic few-term model gives analytic `shift` / `offset`, including angles outside [0, 360).
2. Reference conversion: default config gives `phi_ref = (157.42, 165.80)`.
3. CM FITS orientation: an asymmetric 2x2 written with `mx::fits` loads as rows `v1, v2`, cols `dx, dy`.
4. Command logic with a fake ttmperi: enable captures `v0` with no jump; a QWP move gives `v0 - CM * offset` and
   updates `offset`; our own echoed target does not re-base, an external target does; identical voltages are not
   resent; commands are held while ttmperi is not READY and sent once READY; disable stops commands; enable with
   invalid inputs is refused.
5. Config defaults; missing or malformed CSV, and non-2x2 FITS, fail at startup.
6. INDI callback validation for `tracking` and every subscribed property.

On-instrument check (user runs): enable at the reference, sweep the QWPs through the operating box, confirm the
camwfs pupil-loop deltas stay flat.

## Implementation Plan

1. Branch `ml/qwp-pupil-tracker` (current). Copy `apps/template` structure to `apps/qwpPupilTracker/` (`Makefile`,
   `.cpp` with `main` only, `.hpp` with class and out-of-class inline definitions); add to the apps build list.
2. Model: struct of `a, b, dx_cos, dx_sin, dy_cos, dy_sin` vectors, `loadModel()` via `readColumns`, static
   `shift()` in double precision.
3. CM: `loadCM()` via `mx::fits`, validated 2x2.
4. Config and reference conversion (`phi = (sign * (q - zero)) mod 360`), `shift(phi_ref)` cached at load.
5. INDI: `REG_INDI_SETPROP` for stage `position` / `fsm` and ttmperi `axis1_voltage` / `axis2_voltage` / `fsm`;
   `CREATE_REG_INDI_NEW_TOGGLESWITCH` for `tracking`; read-only `offset`.
6. Command logic: enable / re-base / compute / send (`sendNewProperty` on `axis1_voltage.target`, `axis2_voltage.target`), held
   while ttmperi not READY, skipping identical values.
7. Telemetry: `TELEMETER_*` macros; a log type for offset and commanded voltages (check whether an existing
   telem type fits before adding a new one).
8. Tests as above.
9. Exporter `~/dev/magaox/scripts/export_qwp_pupil_calib.py` (separate repo, user commits): reads the two research
   JSONs, writes `qwp_pupil_model.csv` and `qwp_pupil_cm.fits` with provenance, verifies against `evaluator.py`.
10. clang-format, restore `{ //mutex scope` comments; commits: app + tests + plan, then docs, then formatting.

Out of scope: handbook updates; the instrument config file (user writes it on the instrument).

## Follow-up and Edge Cases

- Verify `elliptecCtrl` publishes `position.current` during moves (affects decision 5).
- qwpTracker picks the nearest equivalent stage angle, so a stage can sit on the +180 branch where the correction is
  38-70 V, outside the IM's 20 V calibration and possibly the piezo range. Not handled here (decision 6).
- The model is unvalidated (0.02 px predicted vs 0.01 px target); a refit only needs the exporter rerun.
- ttmperi hysteresis (~0.08 px) exceeds the model target.
- A ttmperi move during the J-test while tracking re-bases `v0`; a J-test done while the QWPs are moving will mix
  the two.

## Implementation Notes

Implemented on `ml/qwp-pupil-tracker` (2026-11-20).

- `apps/qwpPupilTracker/`: header-only app, `main` in the `.cpp`, Makefile; added to `apps_aoc` and
  `all_buildable_apps` in the top-level Makefile, tests added to `tests/tests.list`.
- Telemetry: new `telem_qwppupil` (event code 20951: `tracking, dx, dy, v1, v2`), registered in `logCodes.dat`,
  `telem.cpp`, `logMeta.cpp` and `logTypes_Accessor_test.cpp`. The ttmperi `kcubeCtrl` has no voltage telemetry, so the
  commanded voltages are recorded only here. `ml/qwpTracker` adds `telem_qwptrack` at 20950 on the adjacent line of `logCodes.dat`, so expect a
  trivial merge conflict there.
- Echo handling: each axis keeps its last 16 commands. An `axisN_voltage.target` within `ttm.rebaseTol` (0.01 V) of any of
  them is an echo, so late echoes during fast stage updates do not re-base. Anything else re-bases that axis
  (`v0 = target - dv`). While ttmperi is not usable, target changes only set a pending re-base of both axes, applied
  once it is usable again, so a restart does not re-base on placeholder values.
- Repeat suppression compares with a 1 uV tolerance: after a re-base, `(target - dv) + dv` need not round back to
  exactly the target, and an exact comparison sent a spurious command.
- The `offset` property is also updated while not tracking. Since the 2026-11-20 amendment it reads zero then,
  because the reference follows the stages until tracking is enabled.
- Calib files: CSV model rows are checked for exactly six columns (`readColumns` silently fills a short row with
  -99) and integer `a`, `b`. The FITS CM is read in python orientation: `im(px, v) == CM[v][px]`.
- Exporter: `~/dev/magaox/scripts/export_qwp_pupil_calib.py` (PEP 723, `uv run`). Verified: 22 frequencies, max
  error 3.6e-15 px against the 93 research test vectors. The CM header carries the IM and provenance.
- Verification: unit tests pass in the `magao-x-setup:cli` container with gcc-toolset-14 (144 assertions, plus the
  INDI validation test). A one-off check (not committed) loaded the real exported files through the app's loaders:
  max error 4e-15 px against the research vectors, `phi_ref = (157.42, 165.80)`, `shift(phi_ref) = (0.359, -0.520)`
  px, and the CM loaded un-transposed (`[[63.43, -5.37], [-6.94, 60.57]]`).
- Not done (instrument side, user): commit the exported files to the calib repo, write `qwppupil.conf`, verify
  that `elliptecCtrl` publishes `position.current` during moves, and run the closed-loop check.
