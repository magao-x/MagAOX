# dmSpeckle runtime arbcube reload + mode switching

## Problem to solve

`apps/dmSpeckle/dmSpeckle.hpp` currently supports two operating modes — `sparkle`
and `arbcube` — selected only via `modulator.opMode` in the config file. In
`arbcube` mode it reads a FITS cube from `modulator.fileName` (an absolute path
in config) on each modulation start. Operators want to:

1. change the cube filename at runtime via INDI, with the usual
   `current` / `target` convention;
2. trigger a synchronous reload of the new cube via an INDI request switch and
   get feedback (OK / Alert) on whether the load succeeded;
3. switch between `sparkle` and `arbcube` modes at runtime via an INDI
   multi-element selection switch.

The runtime-supplied `file.target` is treated as an opaque filesystem path —
any unreadable, missing, or malformed file surfaces as a logged
`software_error` on load attempt rather than being pre-validated. Earlier
design iterations restricted the value to a leaf filename plus a configured
prefix path; that constraint was dropped per user direction (2026-06-15) in
favor of letting operators point at any path.

## Plan

### Goals
- Add a `<device>.file` text property with `current` and `target` elements
  holding the full filesystem path to the arbcube FITS file. `current`
  reflects what was last accepted by the callback (i.e. last commanded
  value), not the result of the most recent load.
- Add a `<device>.load_file` request switch (`request` element) that triggers
  a reload of the file named in `current`, with the load's success / failure
  reflected in the state of `<device>.file` (`Ok` or `Alert`).
- Add a `<device>.mode` selection switch with elements `sparkle` and
  `arbcube`, one-of-many, that swaps `m_opMode` at runtime.
- The runtime-supplied path is not validated; any I/O or dimension errors
  surface when the load is attempted.

### Files touched
- `apps/dmSpeckle/dmSpeckle.hpp` — primary implementation.
- This plan file under `## Plan`.

`dmSpeckle.cpp` is the existing 11-line main and stays unchanged.
`apps/dmSpeckle/tests/dmSpeckle_test.cpp` keeps its existing placeholder test
only; no new unit tests since the sanitizer helper was removed.

### Configuration changes
- `modulator.opMode` — unchanged, still selects initial mode at startup
  (`sparkle` default, `arbcube` opt-in).
- `modulator.fileName` — unchanged, still seeds `m_fileName` at startup;
  remains a full filesystem path. Help text reworded to reflect that the
  value is also settable at runtime via the new `file` property.

### New / changed member state
- INDI properties:
  - `pcf::IndiProperty m_indiP_file;`
  - `pcf::IndiProperty m_indiP_loadFile;`
  - `pcf::IndiProperty m_indiP_mode;`
- Flags:
  - `bool m_reloadFile{ false };` — set true by the `load_file.request`
    callback. The modulator thread checks this flag at the top of each
    outer-loop iteration, performs the load, updates INDI state of
    `m_indiP_file` to `Ok` on success / `Alert` on failure, then clears the
    flag.
- `std::mutex m_shapesMutex;` — guards `m_shapes` while a load is rewriting
  it. The modulator hot inner loop does not take this mutex; instead, the
  `load_file.request` callback sets `m_restartSp = true` so the inner loop
  exits, then `loadArbcubeFile` holds the mutex during the rewrite.

### `appStartup` changes
- Register every INDI property unconditionally, regardless of initial
  `m_opMode`. This includes the previously sparkle-only `separation`,
  `angle`, and `cross`, plus the new `mode`, `file`, and `load_file`. Users
  can adjust any of these in either mode; values take effect when the
  modulator next runs and (if applicable) generates speckles for the
  configured mode.
- Order: trigger group, sparkle group, amp, frequency, trigger, dwell,
  single, modulating, zero, then the new `mode`, `file`, `load_file`.
- New property setup:
  - `m_indiP_file` via `createStandardIndiText(... "file", ...)` and
    `registerIndiPropertyNew`. Seed `current` and `target` from
    `m_fileName`.
  - `m_indiP_loadFile` via `createStandardIndiRequestSw(... "load_file", ...)`.
  - `m_indiP_mode` via `createStandardIndiSelectionSw(... "mode",
    {"sparkle", "arbcube"}, ...)`. Set the correct element to `On` based on
    initial `m_opMode`.

### `generateSpeckles` refactor
- Extract the arbcube branch into a private helper:
  `int loadArbcubeFile()` that
  - rejects an empty `m_fileName`,
  - locks `m_shapesMutex`,
  - reads `m_fileName` directly into `m_shapes`,
  - applies the dimension check (now diagnostic, includes expected vs.
    loaded WxH and the path),
  - scales by `m_amp`,
  - returns 0 / -1. Severity downgraded from `software_critical` to
    `software_error` so a bad runtime file does not crash the app.
- `generateSpeckles()` arbcube branch becomes a call to `loadArbcubeFile()`
  whose return value is checked. After the if/else, the existing trailing
  `updateIfChanged` calls run for both modes (delay / amp / frequency /
  dwell / single).
- `generateSpeckles()` is no longer the only path that loads the cube; see
  `modThreadExec` changes below.

### `modThreadExec` changes
- Outer-loop structure becomes:
  1. If `m_reloadFile`: clear flag, call `loadArbcubeFile()` unconditionally
     (even in sparkle mode — acts as a validation check; the loaded data may
     be overwritten when sparkle next regenerates patterns). Push INDI state
     `Ok` / `Alert` to `m_indiP_file` based on the load result.
  2. If `!m_modulating`: sleep 500 ms and continue.
  3. Else: existing modulating branch. `generateSpeckles()` is still called
     once per modulating restart; in arbcube mode that re-invokes
     `loadArbcubeFile()` which is wasted work after a recent top-of-loop
     load, but the cost is negligible and avoids extra state-tracking.
- The reload request always works regardless of current mode, so the user
  can validate the file before pressing modulate, or while in sparkle mode.

### Callback semantics
- `m_indiP_file` callback:
  - `INDI_VALIDATE_CALLBACK_PROPS`.
  - If `target` is present, under `m_indiMutex` set `m_fileName = target`,
    push both `target` and `current` to `target` via `updateIfChanged`.
  - This callback does *not* trigger a load. `load_file.request` is the
    trigger. The new path is also not validated by the callback; any
    problems surface from the next `loadArbcubeFile()` attempt.
- `m_indiP_loadFile` callback:
  - `INDI_VALIDATE_CALLBACK_PROPS`.
  - If `request` element is `On`:
    - Under `m_indiMutex`, set `m_reloadFile = true` and
      `m_restartSp = true`. Push state `Busy` to `m_indiP_file`. The
      modulator thread will pick up the flag, run the load, and update
      `m_indiP_file` state to `Ok` / `Alert` accordingly.
- `m_indiP_mode` callback:
  - `INDI_VALIDATE_CALLBACK_PROPS`.
  - Determine which of `sparkle` / `arbcube` is being requested `On`.
  - Under `m_indiMutex`, update `m_opMode` accordingly, mirror the switch
    state back with `updateSelectionSwitchIfChanged`, and set
    `m_restartSp = true` so the modulator thread re-generates speckles on
    next iteration.

### Telemetry
- `recordDmSpeck` currently captures sparkle-specific parameters. Out of
  scope to extend the telemetry schema in this change. We will add a TODO
  comment noting that `m_opMode`, `m_fileName`, and the path are not yet in
  the telemetry record.

### Tests
- `tests/dmSpeckle_test.cpp` keeps its existing placeholder test only. The
  sanitizer helper was removed, so the previously planned filename-validator
  unit tests are dropped along with it. No INDI callback unit tests added;
  this app does not have an existing INDI callback test harness and adding
  one is out of scope.

### Documentation pass
- Update file-level doc block to mention runtime cube reload.
- Add `///` doc on the new members, the new helper, and the new callbacks.
- Do an `AGENTS.md` rule-15 "changed file documentation pass" sweep on the
  whole file.

### Branch + commits
- Create feature branch `jlong/dmSpeckle-runtime-arbcube-reload` from `dev`.
- Commit ordering (per `AGENTS.md` rule 19):
  1. Functional change in `dmSpeckle.hpp` and minimal test additions.
  2. Plan file under `agents/plans/2026-06/` (this file).
  3. Final `clang-format -i` pass over touched files if needed.

## Assumptions
- The modulator thread is the only writer to `m_shapes` other than
  `loadArbcubeFile()`. Both call sites take `m_shapesMutex`; the hot inner
  loop does not, and is kept consistent through the existing `m_restartSp`
  rendezvous.
- All INDI properties — including sparkle-only `separation` / `angle` /
  `cross` and arbcube-only `file` / `load_file` — are registered
  unconditionally so users can stage values for the other mode and have
  them take effect on mode switch. Per-user direction (2026-06-15).
- Runtime-supplied paths are trusted (`m_indiDriver` access is already
  authenticated upstream); validation is deferred to the `mx::fits::fitsFile`
  read itself. If hardening is ever needed, the natural place to add it is
  back in `loadArbcubeFile` rather than in the callback.

## Followups / edge cases
- Telemetry schema does not yet record `m_opMode` or filename; add fields in
  a separate change.
- In arbcube mode while modulating, a `load_file.request` causes both the
  top-of-loop eager reload AND the per-modulating-restart reload inside
  `generateSpeckles()`. Wasted I/O but functionally correct; can be
  optimized later with a cache invalidation flag if it ever shows up in
  profiling.
- The current `/tmp/specks.fits` debug write in `generateSpeckles()` is
  preserved. Could become an INDI-toggleable debug option later.
- Build is not exercised in this branch; the user will compile on the
  instrument. Reviewer should sanity-check that `calibDir()` returns a
  populated string at `loadConfigImpl` time (it does — `setDefaults()` runs
  earlier in the `mx::app::application` lifecycle, same pattern used by
  `libMagAOX/app/dev/dm.hpp`).
