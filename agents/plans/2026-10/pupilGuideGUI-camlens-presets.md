# Camera-lens presets in pupilGuideGUI

Date: 2026-10-29. Branch: `jrmales/camlens-preset-menu`.
Status: implementation and offline verification complete.

## Request and scope

Implement the GUI portion of [issue #319](https://github.com/magao-x/MagAOX/issues/319):
the Camera Lens section needs a standard stage preset dropdown listing names
shared by `stagecamlensx` and `stagecamlensy`, with selection applied to both.
Pupil fitting remains a separate desired capability. This change does not alter
controllers, stage configuration, numeric position entries, or nudge controls.

Work was performed on workstation `mx11`. All runtime checks use offline widgets
and fake properties; no instrument client or hardware command was run.

## Implementation decisions

- Add reusable `xqt::stagePairStatus`, derived from the existing `statusCombo`,
  to preserve the standard dropdown, Go button, edit timeout, and pending-command
  appearance. Expose the base editing-state enum to subclasses and make its Go
  slot virtual so the pair can check readiness before submitting a selection.
- Subscribe independently to both stages' `fsm` and `presetName`. Match exact
  switch-element names, including aliases; exclude empty names and `none`.
  Do not infer shared presets from numeric positions or element order.
- Definitions replace the corresponding stage's observations; partial Set
  updates merge them. The common list is rebuilt when names change. Malformed
  property types invalidate the affected observations, and invalid switch states
  remove the affected choices.
- Enable the selector only when both stages are READY and have common choices.
  Go sends a separate sparse `presetName` switch request to each stage, with the
  chosen name On. Selection alone sends nothing, duplicate pending submissions
  are suppressed, and callbacks/reconnection never replay commands.
- Display a name only when both stages report that unique active preset;
  otherwise display `---`. A controller can report its intended alias during
  movement, so this name acknowledges a target rather than proving completion.
- Device/FSM/preset deletion cancels staged and pending selections and clears
  only the affected observations. Recovery requires fresh properties. Numeric
  position deletion and unrelated device loss preserve this independent control.
- Keep the widget's existing header-based implementation and Qt build structure.
  Add the selector to both application and test projects and to Doxygen inputs.

## Verification

- Pupil-guide offline suite: **2,208 assertions in 20 test cases passed**,
  including seven new preset cases. Coverage includes both arrival orders,
  asymmetric lists, exact-name aliases, no-common choices, replacement and partial
  updates, paired Go commands, pending acknowledgment, multiple active switches,
  malformed types/values, per-axis restart, local deletion, and full reconnect.
- Built `pupilGuideGUI` successfully without launching its INDI client. The only
  compiler warning is the existing unchecked `QFile::open()` result in the
  untouched main entrypoint.
- The pre-existing local INDI archive contains coverage instrumentation, so these
  local qmake builds use `QMAKE_LFLAGS+=--coverage`. Test execution redirects
  coverage output under `/tmp` with `GCOV_PREFIX`; no build flags were added to
  tracked project files for this environment-specific requirement.
- Offline visual preview used the actual panel, repository stylesheet and icons,
  and mock stage properties. The new row and opened dropdown fit without overlap
  or clipping. Temporary preview artifacts are under `/tmp/camlens-preview`.
- Existing coronagraph/stage regression suite: **4,146 assertions in 9 cases
  passed**. Together the two suites pass **6,354 assertions in 29 cases**.
- Full changed-file documentation review and focused Doxygen generation passed.
  All seven new cases link to the paired selector's real APIs: 19 references
  with HTML backlinks across seven APIs. The full-file audit, including hidden
  test helpers, found briefs on all 266 declarations/members and documentation
  for all 93 named parameters. There are
  no warnings naming changed source files; remaining focused-build diagnostics
  concern pre-existing configuration tags or references outside the input subset.
  Temporary documentation output is under `/tmp/camlens-doxygen`.
- `clang-format --dry-run --Werror` on all changed C++ files and
  `git diff --check` passed.

Reproduction commands for the pupil-guide suite, from an empty temporary build
directory:

```sh
qmake /home/jrmales/Source/MagAOX-working/gui/widgets/pupilGuide/tests/pupilGuide_test.pro 'QMAKE_LFLAGS+=--coverage'
make -f makefile.pupilGuide_test -j2
QT_QPA_PLATFORM=offscreen GCOV_PREFIX=/tmp/camlens-preset-tests/gcov ./pupilGuide_test
```

## Follow-up and limits

- Instrument validation must be performed by the user: verify the configured
  shared names, select a preset, press Go, and confirm both stages' positions.
- Paired requests are sequential, not atomic. A controller or transport failure
  between them can leave only one stage moved; the GUI cannot provide a
  transaction across independent controllers. It does not automatically retry.
- This work does not implement pupil fitting or define new stage presets.
