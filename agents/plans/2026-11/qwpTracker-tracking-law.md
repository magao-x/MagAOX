# Task Description
<!-- Drafted by the agent from the user's prompt on 2026-11-19; the user should correct anything here. -->

## Problem Statement
Replace the placeholder law in `qwpTracker::computeTargets` (`qwp1 = 2k`, `qwp2 = -2k`) with the model-derived QWP
tracking law fit from the 2026-11-14/15 polarization calibration. The law gives the QWP1 and QWP2 angles, per science
filter, as a function of the IMR (K-mirror) angle. Decide how the law's static data is stored, vendored and loaded.

## Discussion
- Source: `~/research/magaox/20261114_pol_cal/data/outputs/20261115/tracking_{r,i,z}.npz`, written by
  `scripts/tracking_law.py` in that repo, plus the flattened `20261115_qwp_tracking_law.csv` and
  `tracking_summary.json`.
- Each `tracking_<f>.npz` holds, on a 181-point grid `imr` = -90 to 0 deg in 0.5 deg steps: the chosen law `q`
  (181x2, deg), posterior band `q_lo`/`q_hi`/`q_sd`, all solution branches `branch_q`, diagnostics (efficiency,
  rotation, cross-talk, instrumental polarization) and the empirical reference `ref_q`.
- The analysis repo evaluates the law with `project.qwp_law.QWPLaw`: unwrap each QWP track with period 180 deg, linear
  interpolation in IMR angle, error outside the grid, result folded to [0, 180).
- Precedent for loading a lookup table: `apps/adcTracker` reads a comma-delimited table from
  `m_calibDir + "/" + m_lookupFile` with `mx::ioutils::readColumns`, validates it, and interpolates.
- The app plumbing (subscriptions, modes, stage mapping, nearest-equivalent wrap, FSM) is done; see
  `agents/plans/2026-11/qwpTracker-dummy-law.md`.

## Requirements
1. `computeTargets` evaluates the fitted law for the active filter at the current IMR angle.
2. The law data is loaded at startup from a vendored, versioned file, validated, and logged.
3. Defined behavior outside the fitted IMR range and when the filter is unknown or has no law.

## Tests and Metrics
- Interpolated targets match `project.qwp_law.qwp_angles` at grid and off-grid points to < 0.01 deg.

## Scope and Caveats
- No change to the stage mapping or mode logic beyond what the law requires.

# Instructions to Agent

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan

## Task Summary
Replace the placeholder `computeTargets` with a per-filter lookup-table law: at startup load
`<calib>/qwptrack/qwp_law_<filter>.csv` (columns `imr, qwp1, qwp2`, QWP tracks unwrapped), follow the selected filter
from `fwsci1.filterName`, and while tracking linearly interpolate fast-axis targets at `stagek.position.current`.
Everything downstream of `computeTargets` (stage mapping, nearest-equivalent wrap, modes, FSM, telemetry) is unchanged.
The CSVs are produced by a new export script in the pol_cal analysis repo and committed to `calib` by the user.

## Key Assumptions
- `fwsci1.filterName` is a `stdMotionStage` preset switch: exactly one element `On` when at a preset, property state
  `INDI_BUSY` while moving (`stdMotionStage::updateINDI`). The active filter is the `On` element when the state is not
  busy; otherwise none (decision 5).
- Linear interpolation of the unwrapped tracks reproduces `project.qwp_law.QWPLaw.unwrapped`; the app's existing
  `fold180`/nearest-equivalent mapping handles the result being outside [0, 180).
- `mx::math::gslInterpolator<gsl_interp_linear<double>>`, as in `adcTracker`, is the interpolator (requires
  `LDLIBS+=-lgsl` in `apps/qwpTracker/Makefile`, removed in the dummy-law work).
- Filter names contain no characters that are invalid in a file name (true for `r`, `i`, `z`).

## Requirements
1. Config: `[filter] devName` (default `fwsci1`); `[law] dir` (default `qwptrack`, relative to `m_calibDir`).
2. Startup: load and validate every `qwp_law_*.csv` in the law directory (>= 2 rows, all finite, strictly increasing
   `imr`); a malformed file is `software_critical` (startup fails); no files is a warning (decision 9). Log each loaded
   law (filter, rows, IMR range) with `text_log`.
3. Subscribe `<filter devName>.filterName` with `REG_INDI_SETPROP`; the callback stores the active filter name under
   `m_indiMutex` and flags a recompute.
4. `computeTargets` evaluates the active law at `k`; returns false (no targets) when no law is active or `k` is
   outside the law's IMR range.
5. While tracking, a K update or a change to an available law recomputes targets; when unavailable, hold (no sends),
   log once on entry (warning) and recovery (info).
6. New read-only INDI text property `law.current`: active law filter name or `none`.
7. Manual mode, stage mapping, FSM and `telem_qwptrack` unchanged.
8. pol_cal repo: `scripts/export_qwptrack_law.py` + Snakemake rule writing `qwptrack/qwp_law_{r,i,z}.csv` from the
   chosen branch `q` of `tracking_<f>.npz`, unwrapped with `project.qwp_law._unwrap180`, with a `#` header (dataset,
   pol_cal commit, "posterior median, chosen branch").

## Questions and Points of Clarification
(Being resolved with the user; answers recorded here.)

1. **Angle frames (agreed 2026-11-19):** the law's input `imr_ang` is `stagek.position.current` as published (no
   zero/sign/offset between them; no `[k] zero/sign` config). The law's outputs `qwp1_ang`/`qwp2_ang` are fast-axis
   angles, i.e. `qwptrack.qwp{1,2}` angles, so the existing `[qwpN] zero/sign` stage mapping applies unchanged and the
   law output feeds `computeTargets` directly.
2. **Storage (agreed 2026-11-19):** the law is vendored as CSV in the `calib` repo (`/opt/MagAOX/calib`, via
   `m_calibDir`), loaded at startup like `adcTracker`'s lookup table. One file per filter,
   `qwptrack/qwp_law_<filter>.csv`, columns `imr, qwp1, qwp2` only (deg; QWP tracks unwrapped, i.e. continuous, so the
   app interpolates linearly), with a `#` provenance header (dataset, source commit, posterior median). A new fit is a
   `calib` commit plus an app restart. An export script in the pol_cal repo writes the files from `tracking_<f>.npz`.
   Unit tests use a small fixture table under `apps/qwpTracker/tests/`. No uncertainty or efficiency columns.
3. **Filter selection (agreed 2026-11-19):** automatic; qwptrack subscribes to `fwsci1.filterName` (standard
   `REG_INDI_SETPROP`, no polling) and uses the law for the selected filter. No manual filter switch.
4. **File naming (agreed 2026-11-19):** law files are named after the fwsci1 filter names exactly,
   `qwptrack/qwp_law_<filterName>.csv`; no mapping config. fwsci1 names (user, 2026-11-19): `CH4-875`, `Ha-Cont`,
   `i`, `open`, `r`, `z`, so the files are `qwp_law_r.csv`, `qwp_law_i.csv`, `qwp_law_z.csv`. At startup qwptrack
   loads every `qwp_law_*.csv` in `calib/qwptrack/`, validates each (>= 2 rows, finite, strictly increasing `imr`),
   fails startup on a bad file, and logs which filters are covered. Renaming a wheel filter silently drops its law
   (logged as "no law for filter X").
5. **No law available (agreed 2026-11-19):** if fwsci1 is on a filter without a law (`CH4-875`, `Ha-Cont`, `open`)
   or is moving/unknown, tracking stays enabled but holds: no new targets, QWPs stay put, logged once on entry
   (warning) and on recovery (info). When a law becomes available, targets are recomputed immediately from the last
   `stagek` value without waiting for K to move. A read-only INDI text property `law.current` shows the active law's
   filter name or `none`. IMR outside the table range will not happen in practice (user); defined behavior is the
   same hold plus a one-time warning (agent's choice, consistent with `project.qwp_law` raising outside its range).
6. **Branches (agreed 2026-11-19):** vendor the pipeline's chosen branch for every filter (`q` in
   `tracking_<f>.npz`), no overrides. Noted for the record: the chosen r branch (QWP1 4.7-27.3, QWP2 -3.0-11.4 deg) is
   ~90 deg from the empirical tracks, while tied-best branch 0 matches them; the chosen z branch has a 2.57 deg step
   per 0.5 deg IMR. The user may update the CSVs later; the app does not care which branch a file holds.
7. **Vendoring workflow (agreed 2026-11-19):** the export script goes in the pol_cal analysis repo
   (`scripts/export_qwptrack_law.py` plus a Snakemake rule, reusing `project.qwp_law`'s unwrap); the output CSVs live
   only in the `calib` repo under `qwptrack/` (user copies and commits them). This repo carries no copy of the real
   law: unit tests use small synthetic fixture tables (agent's choice, since the data is to live in `calib`), with
   expected values computed by hand, including a track that crosses 0/180 when folded.
8. **Telemetry (agreed 2026-11-19):** `telem_qwptrack` unchanged. Law provenance is logged once at startup with
   `text_log`; law availability changes are logged per decision 5.
9. **Startup without laws (agreed 2026-11-19):** if no `qwp_law_*.csv` is found (missing or empty directory), log a
   warning and start; manual mode works and tracking holds with `law = none`. A malformed file still fails startup.
   Laws are loaded only at startup; no INDI reload switch (a new `calib` commit needs an app restart).

## Tests
In `apps/qwpTracker/tests/qwpTracker_test.cpp` (existing structure, `QWPTRACKER_TEST_NOSEND`), with synthetic fixture
tables written to a temporary directory by the test (or small files under `apps/qwpTracker/tests/`) and `m_calibDir`
pointed at it:
1. Loading: valid files for two filters load; filter names come from file names; no files -> starts with no laws;
   one row, non-finite value, or non-increasing `imr` -> load fails.
2. Interpolation: grid points return the table values; midpoints return the linear average; a track that is unwrapped
   past 180 (e.g. 170 -> 190) interpolates continuously and the folded fast-axis target is correct (e.g. 0 at 180).
3. Range: `k` outside the table -> no targets.
4. Filter callback: `filterName` with one `On` and idle state selects that law; busy state or a filter without a law
   selects none; `law` reflects it.
5. Tracking: with stages READY and tracking on, a K update sends the interpolated targets mapped through zero/sign;
   switching to a filter without a law sends nothing (hold); switching back recomputes from the stored K without a new
   K update.
6. Remove the placeholder-law test (`qwp1 = 2k`, `qwp2 = -2k`); keep mapping/mode/FSM tests, updating those that
   relied on the placeholder to use a fixture law.
7. Doxygen per `TEST_CASE`; `QWPTRACKER_TEST_DOXYGEN_REF` references for the new members.

## Implementation Plan
Branch `ml/qwpTracker` (this app's feature branch; not yet merged to `dev`).
1. **pol_cal repo** (separate branch there): `scripts/export_qwptrack_law.py`, Snakemake rule `qwptrack_law`, output
   `data/outputs/<dataset>/qwptrack/qwp_law_{r,i,z}.csv`. Check the written files against
   `project.qwp_law.qwp_angles` at off-grid IMR angles. Hand the files to the user for `calib/qwptrack/`.
2. **qwpTracker members:** `struct qwpLaw { std::string filter; std::vector<double> imr, qwp1, qwp2;
   gslInterpolator terp1, terp2; }`; `std::map<std::string, qwpLaw> m_laws`; `std::string m_filterDevName{"fwsci1"}`,
   `m_lawDir{"qwptrack"}`, `m_activeFilter`, `m_lawAvailable` (for once-only logging); `m_indiP_filterName`,
   `m_indiP_law`.
3. **Loading:** `loadLaws()` called from `appStartup` (after config, before INDI registration): list files with
   `mx::ioutils::getFileNames(dir, "qwp_law_", "", ".csv")`, read each with
   `mx::ioutils::readColumns<mx::ioutils::readColCommaDelim>` (confirm `#` header lines are skipped, else strip them),
   validate as in `adcTracker::appStartup`, build interpolators.
4. **Law:** `computeTargets(k, qwp1, qwp2)` becomes a non-static member returning `bool`, reading the active law;
   called under `m_indiMutex` from `appLogic` where it is now (line ~574). Generalize the recompute flag
   (`m_kUpdated`) so a filter change also sets it.
5. **INDI:** `REG_INDI_SETPROP(m_indiP_filterName, m_filterDevName, "filterName")`; callback determines the active
   filter (decision 5 / key assumptions), updates `m_activeFilter`, `law.current`, flags recompute, logs transitions.
   `createROIndiText(m_indiP_law, "law", "current", ...)`.
6. **Makefile:** `LDLIBS+=-lgsl`.
7. **Docs:** class brief and file block describe the law, config, and calib file format; update
   `qwpTracker-dummy-law.md` follow-up item to point here.
8. Commits (AGENTS.md 22): (1) functional + tests + this plan, (2) Doxygen pass, (3) `clang-format` with
   `{ //mutex scope` restored. Build and test in the `magao-x-setup:cli` container as before.
9. Example `qwptrack.conf` additions for the user (config repo):
   ```ini
   [filter]
   devName=fwsci1

   [law]
   dir=qwptrack
   ```

## Follow-up and Edge Cases
- The chosen r branch differs from the empirical tracks by ~90 deg (decision 6); verify on the bench before relying on
  r-band tracking, and re-export with a different branch if needed (CSV-only change).
- The z law has a 2.57 deg step per 0.5 deg IMR; the stages are retargeted each update, so fine unless bus traffic
  becomes a problem.
- Filter renames in the fwsci1 config silently drop a law (logged as "no law for filter X").
- Laws depend only on IMR angle and filter; dependence on HWP angle, telescope pointing or temperature is out of scope.
- `calib` provenance header is informational; the app does not check dataset versions.

## Implementation Status (2026-11-19)
- **pol_cal repo:** `scripts/export_qwptrack_law.py` and Snakemake rule `qwptrack_law` added (the repo has no commits
  yet, so they are uncommitted there). The script reads the chosen branch `q` of each `tracking_<f>.npz`, unwraps it
  with `project.qwp_law._unwrap180`, checks it against `project.qwp_law.QWPLaw` on and between grid points (max
  deviation 5e-4 deg, the 6-significant-digit rounding of the law CSV), and writes
  `data/outputs/20261115/qwptrack/qwp_law_{r,i,z}.csv` (181 rows each) with a `#` provenance header (dataset, source,
  chosen branch index, npz sha256 prefix, pol_cal commit or `none`). The user copies these to `calib/qwptrack/`.
- **qwpTracker:** `loadLaws()` (called first in `appStartup`) lists `qwp_law_*.csv` with `mx::ioutils::getFileNames`,
  reads them with `readColumns<readColCommaDelim>` (its comment character is `#`, so the header is skipped), validates,
  and builds `gslInterpolator` linear interpolators. `qwpLaw` deletes copy because the interpolators keep pointers into
  its vectors; laws live in `std::map<std::string, qwpLaw> m_laws`. `computeTargets` is now a member returning `bool`.
  `m_kUpdated` was renamed `m_recompute` (set by K updates, filter changes and tracking-on). `m_lawOK` gates the
  once-only hold/resume logs and is reset on each tracking start. New `filterUpdate`/`m_indiP_filterName`
  (`fwsci1.filterName`), read-only `law.current`, config `[filter] devName`, `[law] dir`. `LDLIBS+=-lgsl`.
- **Tests:** placeholder-law test removed; the harness writes a two-row law for `r` that interpolates to exactly
  `2k`/`-2k`, so the existing tracking expectations are unchanged. New cases: table loading and validation, linear
  interpolation across 0/180 and range limits, filter-wheel following, a wrapped law tracked through 180, hold on a
  filter without a law / moving wheel / K outside the table, and re-targeting on recovery without a K update. INDI
  validation test covers `fwsci1.filterName`. All pass (11 + 1 test cases), and the app builds, in the
  `magao-x-setup:cli` container with gcc-toolset-14.
- **mxlib caveat:** `readColumns` accepts a row with a missing column and a `nan` entry without error, so those cases
  are not unit-tested; the app still checks sizes and finiteness as `adcTracker` does. The export script validates the
  tables before they reach `calib`.

## On-instrument Test Findings (2026-11-20)
- Law selection behaved as designed on the instrument: hold on `CH4-875`, `unknown` while fwsci1 moved, resume on `i`.
- QWP1 repeatedly missed its target (`qwp1.current` 90.00 vs `target` 92.41) with elliptecCtrl reporting
  `move failed (stopped away from target)`. The stage log shows a new target arriving while the stage was still moving
  (READY/OPERATING flip 0.05 ms apart) before each failure. The dummy-law plan's assumption that the ELL21 accepts a
  new target while `OPERATING` does not hold reliably: `elliptecLowLevel` sends the new `ma` and replaces its expected
  target, but the mount does not end there. Because qwptrack does not repeat an identical stage command, the missed
  QWP stays wrong until K moves enough to change its target.
- **Fix (agreed 2026-11-20):** qwptrack commands a stage only when it is `READY`; targets computed while it is
  `OPERATING` stay pending and only the latest is sent once it stops. Tested by "a moving stage is not retargeted".
- Not done (user deferred): a one-time resend when a stage settles more than 0.1 deg from the last command.
- Follow-up outside qwptrack: retargeting during a move should be handled in `elliptecLowLevel` (stop then move, or
  hold the latest target until the move completes), since `elliptecCtrl` advertises retargeting. Confirm the cause
  against the `elliptecqwp` log (move ignored vs small-move stall).
- Residual race: between sending to a READY stage and receiving its `OPERATING` state, a further target could be sent;
  the window is one INDI round trip and `appLogic` runs at 1 Hz.
- Separately observed: qwptrack fell to `NOTCONNECTED` on each tracking stop, meaning one stage was in a state other
  than READY/HOMING/NOTHOMED/OPERATING; resolved by power cycling the stages. While tracking, `trackerState` reports
  `OPERATING` regardless, which hid this (possible follow-up).
