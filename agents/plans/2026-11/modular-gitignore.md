# Task Description

## Problem Statement
The repository has a single top-level `.gitignore` that lists every app,
util, GUI, and plugin build product by path (e.g. `apps/adcTracker/adcTracker`).
Adding a new app requires remembering to edit this central file, and in
practice it drifts: entries are missing for some components and stale for
others. Split the component-specific ignore rules into per-directory
`.gitignore` files so each app (and the template/example apps) carries its own,
leaving only repo-wide generic rules in the top-level file.

## Discussion
Git applies `.gitignore` files hierarchically; a pattern in
`apps/foo/.gitignore` is relative to `apps/foo/`. A leading `/` anchors a
pattern to that directory, so `apps/foo/.gitignore` containing `/foo` ignores
only the `apps/foo/foo` executable and not, say, a `foo/` subdirectory.

Because new apps are created by copying `apps/template/`
(see `apps/template/readme.md`), giving the template a `.gitignore` means new
apps inherit one automatically and only need the same find/replace of
`template` that the Makefile, `.hpp`, and `.cpp` already get.

Precedent already exists: `flatlogs/.gitignore`,
`apps/ocam2KCtrl/benchmarks/.gitignore`, and
`agents/plans/2026-11/indiserver-chained-loop-files/.gitignore`.

## Requirements
1. Each app under `apps/` has its own `.gitignore`.
2. The template/example apps have a `.gitignore` that new apps inherit.
3. The top-level `.gitignore` keeps only repo-wide generic rules.
4. The set of ignored files does not change, except for deliberate fixes.

## Tests and Metrics
- `git status --ignored` lists the same build products before and after.
- `git ls-files -ci --exclude-standard` reports no newly ignored tracked files.

## Scope and Caveats
- No build-system or source changes.

# Instructions to Agent

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan

## Task Summary
Move component-specific ignore rules out of the top-level `.gitignore` into
`.gitignore` files that live beside each component, add one to
`apps/template/`, update the "add an application" docs so the new file is part
of the copy/rename step, and fix entries that have drifted.

## Key Assumptions
- Per-directory files hold only what is specific to that directory (mainly the
  executable named by `TARGET=` in its `Makefile`). Generic patterns
  (`*.o`, `*.a`, `**/tests/*_test`, `apps/*/build/`, `*.egg-info`,
  `__pycache__`, Qt `**/moc/`, `**/bin/`, etc.) stay top-level so they apply to
  every component, including ones that forget a local file.
- Every pattern in a per-directory file is anchored with a leading `/`.
- Build outputs land in the component directory: C++ apps/utils emit
  `<dir>/<TARGET>` (`Make/magAOXApp.mk`, `Make/magAOXUtil.mk`); app unit tests
  emit `<dir>/tests/<name>_test` (already covered by `**/tests/*_test`); Python
  apps (`Make/pythonApp.mk`) emit only `build/` and `*.egg-info`.

## Requirements
1. Add `apps/<app>/.gitignore` for every tracked app (101 today: 90 C++,
   11 Python via `pythonApp.mk`) and `utils/<util>/.gitignore` for all 9 utils.
2. Add `apps/template/.gitignore` containing `/template`, and update
   `apps/template/readme.md` (and `doc/templates/app_template.md` if it
   duplicates the steps) so the rename step includes `.gitignore`.
3. Remove the moved per-component lines from the top-level `.gitignore`; keep
   generic rules and anything that has no natural owner directory.
4. Fix drift found during the survey (see below) as part of the move.
5. No tracked file becomes ignored; no currently-ignored build product becomes
   untracked-visible.

## Survey Findings (drift in current top-level file)
Re-surveyed after rebasing `ml/gitignore` onto `origin/dev` (bd338e26).

- **Missing entries:** `utils/shmimLat/shmimLat` and
  `apps/qwpPupilTracker/qwpPupilTracker` are not ignored.
- **Stale entry:** `apps/doc` (doc output dir, not an app; keep with doc rules).
- **Makefile variance:** `utils/cursesINDI/Makefile` is standalone (not
  `magAOXUtil.mk`) and writes `TARGET = cursesINDI` with spaces; the PR check
  must tolerate whitespace around `=`.
- **New GUI:** `gui/apps/timingGUI` (covered by `gui/.gitignore`).
- **Tracked-but-ignored (pre-existing):** `libs/BrainStem2/libBrainStem2.a`
  (matches `*.a`, vendored on purpose) and `local/config.example.mk` (the
  `!local/...` negations are ineffective because `local/` excludes the whole
  directory; git cannot re-include files under an excluded directory).
- **Python apps** (`adcCtrl`, `aoSim`, `audibleAlerts`, `camtipSR`, `corAlign`,
  `dbIngest`, `efcControl`, `pupilCorAlign`, `pythonIndiExample`,
  `sparkleTracker`, `visxCtrl`) have no entries; their outputs are covered by
  generic rules, so they need nothing beyond generic rules today.
- **Redundant entries:** `apps/siglentSDG/tests/siglentSDG_test` and
  `apps/xindiserver/tests/*_test` are already covered by `**/tests/*_test`.
- **Case-insensitivity bug (pre-existing):** `**/old/` matches the tracked
  `apps/nnReconstructor/OLD/` on macOS (case-insensitive FS), so
  `git ls-files -ci --exclude-standard` reports those tracked files as ignored.
  On Linux it does not. Not caused by this change; noted for a decision.
- **GUI/plugins:** `gui/apps/*` build via `Make/magAOXGUI.mk` + qmake into
  `bin/`, `moc/`, `obj/`, `res/`, `ui/` (`gui/apps/magaoxQtApp.pri`); rtimv
  plugins build into their own dir (`DESTDIR = ./`). `obj/` and `ui/` are not
  covered by any rule today.

## Questions and Points of Clarification
Tracked in the grilling session; answers recorded here as they are decided.

1. **Scope (decided).** Per-directory `.gitignore` for every `apps/*`
   (including `apps/template`) and every `utils/*` (`gui/apps/*` dropped, see 4).
   Qt/GUI generic rules (`res/`, `moc/`, `bin/`, `ui*.h`, `makefile.*`, etc.)
   move from the top level into `gui/.gitignore`. INDI executables move into
   `INDI/.gitignore`. rtimv plugins rely on `gui/.gitignore` (no per-plugin
   files). `libMagAOX/logger/tests` and top-level `tests/` stay in the
   top-level file.
2. **Per-directory content (decided): minimal.** Local files hold only
   directory-specific, `/`-anchored entries (normally `/<TARGET>`). Generic
   patterns (`*.o`, `**/tests/*_test`, `apps/*/build/`, `*.egg-info`, ...)
   stay in the top-level, `gui/`, or `INDI/` file and are not duplicated.
   `doc/templates/*.md` are man-page templates and need no change.
3. **Python apps (decided).** Each Python app (including the
   `pythonIndiExample` template) gets a comment-only `.gitignore` noting that
   generic Python build outputs are ignored at the top level. This keeps the
   invariant "every `apps/*` directory has a `.gitignore`" exception-free.
   No new docs for `pythonIndiExample` (none exist describing copying it).
4. **GUI apps (decided, revises Q1).** No per-directory files under
   `gui/apps/*`. qmake places GUI binaries in `bin/` and intermediates in
   `moc/`, `obj/`, `res/`, `ui/` (`gui/apps/magaoxQtApp.pri`), so generic
   rules in `gui/.gitignore` cover every GUI and plugin.
5. **Drift check (decided).** Add a small `.github/workflows/gitignore-check.yml`
   triggered on pull requests only. It fails if any directory with a
   `Makefile` under `apps/` or `utils/` lacks a `.gitignore`, or if a C++
   app/util `.gitignore` lacks `/<TARGET>` (TARGET read from its `Makefile`).
   No make target and no change to existing workflows.
6. **`gui/.gitignore` (decided).** Contents: `bin/`, `moc/`, `obj/`, `res/`,
   `ui/`, `ui_*.h`, `makefile.*`, `.qmake.stash`. All Qt lines leave the
   top-level file (`ui*.h` narrows to `ui_*.h`; per-plugin makefile lines fold
   into `makefile.*`). Verified no tracked file sits in these dirs and no
   non-GUI build writes to `bin/`/`res/` except `flatlogs` (own rule).
7. **Remaining entries (decided).**
   - `INDI/.gitignore`: `/INDI/{evalINDI,getINDI,indiserver,setINDI}`,
     `/xindidriver/xindidriver`.
   - Existing `flatlogs/.gitignore` gains `/src/flatlogcodes`, `/src/logCodes.dat`.
   - `apps/acronameUsbHub/.gitignore` gains `/utils/configusbhub`.
   - Delete redundant `apps/siglentSDG/tests/...`, `apps/xindiserver/tests/...`
     (covered by `**/tests/*_test`) and `vm/cache/`, `vm/calib/`, `vm/config/`
     (covered by `vm/`).
   - Stay top-level: generic compiler/tool patterns, editor/OS clutter,
     Python/coverage rules, `local/`, `vm/`, `setup/`, doc outputs
     (`doc/output`, `apps/doc`, `utils/doc`, `libMagAOX/doc/sw_html`,
     `doc/www`), `libMagAOX/logger/tests/*`, and `tests/*`.
8. **`**/old/` vs tracked `apps/nnReconstructor/OLD/` (decided): out of
   scope.** Listed under follow-ups; before/after check baselines it.
9. **Commits (decided).** Keep the user's existing `update .gitignore` commit
   (now `6559263b` after rebasing onto `origin/dev`) and add on top:
   (1) per-directory split + top-level prune + this plan,
   (2) `gitignore-check.yml` PR workflow,
   (3) `apps/template/readme.md` rename-step docs.
   Branch needs `--force-with-lease` on next push (user to push).
10. **`local/` negations (decided): fix.** Change `local/` to `local/*` so the
    `!local/common.example.mk` / `!local/config.example.mk` re-includes work.
11. **Per-directory file format (decided).** Two-line header comment, then
    the anchored target, then any extras. C++ app/util:
    ```
    # Build products specific to this app. Generic patterns (*.o, tests/*_test,
    # build/, ...) are in the top-level .gitignore.
    /<TARGET>
    ```
    Python app: header only, naming `build/`, `*.egg-info`, `__pycache__`.
    Utils say "util" instead of "app". Generated from `TARGET`, then
    hand-checked (`acronameUsbHub`, `cursesINDI`).
12. **PR workflow details (decided).** `pull_request` trigger with paths
    filter `apps/**`, `utils/**`, `**/.gitignore`, and the workflow file;
    `ubuntu-latest`, no container, `contents: read`; one bash step that checks
    every `apps/*/Makefile` and `utils/*/Makefile` dir for a `.gitignore` and,
    for non-`pythonApp.mk` Makefiles, an exact `/<TARGET>` line (TARGET parsed
    with optional whitespace around `=`). Reports all failures. Does not build
    or check `gui/`/`INDI/`.

## Tests
- Before change: capture `git status --ignored --porcelain` from a tree with
  all apps, utils, GUIs, and tests built, and `git ls-files -ci --exclude-standard`.
- After change: rerun both; diff must be empty except for intended fixes.
- Spot-check with `git check-ignore -v <path>` that each executable is now
  matched by its local `.gitignore`.
- Copy `apps/template` to a scratch name, apply the readme rename steps, build,
  and confirm the new executable is ignored.

## Implementation Plan
1. Build everything locally that builds on macOS and capture the baseline
   (`git status --ignored --porcelain`, `git ls-files -ci --exclude-standard`).
2. Generate `apps/*/.gitignore` and `utils/*/.gitignore` from each
   `Makefile`'s `TARGET` using the Q11 format (includes `apps/template`,
   `apps/qwpPupilTracker`, `utils/shmimLat`, `utils/cursesINDI`).
3. Hand-add extras: `/utils/configusbhub` in `apps/acronameUsbHub`.
4. Create `gui/.gitignore` (Q6) and `INDI/.gitignore` (Q7); add the flatlogs
   lines to `flatlogs/.gitignore`.
5. Prune the top-level `.gitignore` to the Q7 "stay" list; delete redundant
   test and `vm/*` lines; change `local/` to `local/*` (Q10).
6. Rerun the checks; differences must be limited to the intended fixes
   (`shmimLat`, `qwpPupilTracker`, `local/*.example.mk`, `obj/`, `ui/`).
7. Commit 1: split + prune + this plan.
8. Add `.github/workflows/gitignore-check.yml` (Q12); verify its logic locally
   against the tree (passes) and against a deliberately broken copy (fails).
   Commit 2.
9. Update `apps/template/readme.md` rename step to include `.gitignore`.
   Commit 3.

## Implementation Notes (commit 1)
- Verification on macOS without a full build: generated ~565 expected
  build-product paths (every app/util TARGET, `foo.o`, `tests/<TARGET>_test`,
  Python `build/` and `*.egg-info`, GUI `bin/moc/obj/res/ui/makefile/.qmake.stash`,
  rtimv plugin `.so`/makefile, INDI executables, flatlogs, configusbhub, `local/*`)
  and ran `git check-ignore --no-index -v` before/after.
- Result: only differences were `apps/qwpPupilTracker/qwpPupilTracker` and
  `utils/shmimLat/shmimLat` now ignored, and `local/common.example.mk` /
  `local/config.example.mk` now not ignored. `git ls-files -ci --exclude-standard`
  lost only `local/config.example.mk`; the remaining entries are the known
  `apps/nnReconstructor/OLD/*` (case-insensitive `**/old/`) and
  `libs/BrainStem2/libBrainStem2.a`. `git status --ignored` differs only by the
  new untracked `.gitignore` files. `gui/**/obj/` and `ui/` directories are now
  ignored as directories.
- Template test: copied `apps/template`, renamed per the readme (including
  `.gitignore`), `apps/<new>/<new>` was ignored by the local file; scratch dir removed.
- `**/moc_*` was dropped from the top level (moc outputs live in `moc/`, covered
  by `gui/.gitignore`); `.qmake.stash` is also covered by `**/*.stash`.
- `apps/virtualPDU` (last line of old file) got a per-app file like the others.
- `apps/doc` and `utils/doc` stay top-level; `utils/tests` has no Makefile so
  gets no file.

## Implementation Notes (commits 2 and 3)
- `.github/workflows/gitignore-check.yml` implemented per Q12; no pyyaml on the
  Mac, so YAML was parsed with ruby and the `run` step extracted and executed
  locally: passes on the tree; with `/adcTracker` removed, `utils/shmimLat/.gitignore`
  and `apps/zaberCtrl/.gitignore` deleted it exits 1 listing all three; restored.
- `apps/template/readme.md` rename step now includes `.gitignore`.

## Follow-up and Edge Cases
- `**/old/` matches tracked `apps/nnReconstructor/OLD/` on case-insensitive
  filesystems (macOS); new files there are silently ignored. Owner to decide.
- Optional checker (script or test) that every `apps/*/Makefile` dir has a
  `.gitignore` anchoring its `TARGET`.
