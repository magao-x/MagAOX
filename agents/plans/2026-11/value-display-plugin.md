# Task Description
<!-- The first section is filled out by the user 
     
     This is intended as a guide for how to prompt a coding agent to solve a problem.
     
     Make a copy of this template with a new name under agents/plans/<YYYY-MM>. Then
     Fill out each subsection below as needed. Feel free to add additional
     subsections, etc.  

     When complete, commit it on your feature branch and then prompt the agent to review this file.
-->

## Problem Statement
<!-- A brief statement of the problem -->

Create an rtimv plugin to show values from INDI in the upper left corner.  

## Discussion
<!-- Put more expository text here.  Fully describe the expected behavior and rationale. Include references to other parts of the software that serve as examples. Provide a link to an API/SDK manual.   -->
The motivating use case is camflowfs-fit being displayed on the camflowfs display.  The values of interest in INDI look like:
```
camflowfs-fit.values.x_rms=0.05
camflowfs-fit.values.y_rms=0.01
```
The display would be:
```
x_rms: 0.05
y_rms= 0.01
```

Another typical one is camwfs-fit:
```
camwfs-fit.average.dx=-0.482002258300781
camwfs-fit.average.dy=-1.89847564697266
```
which would be displayed as:
```
dx: 0.482
dy:-1.898
```

The label text should be configurable, but default to the INDI element name.

Native rtimv global stats display in the upper left corner too.  We'll want to deconflict, possibly by moving one or the other if both are enabled.  But this might require trading settings with the plugin.

Simply display blank values if they are unavailable, do not print any warning.

This should not be an extension to the camera plugin b/c it might be used for other displays.

## Requirements
<!-- List the requirements that must be met.  A numbered list works best -->


## Tests and Metrics
<!-- List any useful metrics that should be targeted by tests -->

## Scope and Caveats
<!-- Add any needed caveats and scope restrictions -->
We'll limit this to 2 values for now.

# Instructions to Agent
<!-- Specific instructions for the agent.  Below is our standard, but you can modify it as needed. -->

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan

Status: prepared by GPT-6 (Codex) on 2026-11-17. The user approved the plan and answered both questions below, then requested implementation. Implementation and offline verification are complete on `jrmales/value-display-plugin`. Only this section has been changed.

Inspected MagAOX at `e78ac40b` on `dev` and the separate, clean rtimv checkout at `/home/jrmales/Source/rtimv`, `4778f30` on `dev`. Read MagAOX `AGENTS.md`. Host verification returned `mx11`; all investigation used local source files. Existing unrelated untracked files are outside this work.

## Task Summary

Add a standalone `valueDisplay` rtimv overlay plugin that displays up to two numeric INDI elements as labeled rows near the upper-left corner. Reuse the existing INDI dictionary connection, leave unavailable value portions blank, and keep native global statistics readable when both displays are enabled.

### Findings

| Area | Existing behavior and consequence |
| --- | --- |
| Plugin structure | [dmStatus](../../../gui/rtimv/plugins/dmStatus/dmStatus.hpp) and [cameraStatus](../../../gui/rtimv/plugins/cameraStatus/cameraStatus.cpp) implement `rtimvOverlayInterface`, consume configuration with `configUnused()`, and register dictionary keys in `attachOverlay()`. Follow this structure with separate `.hpp`/`.cpp` files. |
| INDI subscriptions | [indiDictionary::checkConnection()](../../../gui/rtimv/plugins/indiDictionary/indiDictionary.cpp) discovers registered `device.property.element` keys, deduplicates requests by property, and handles connection recovery. The new overlay needs no client, requests, or command interface. |
| Availability | `indiDictionary` clears blobs on disconnect and property/device deletion. Ordinary updates preserve elements omitted from a partial update. A replacement property definition currently leaves old blobs for omitted elements; correct that narrow invalidation gap so a removed element cannot remain displayed. |
| Numeric reads | `rtimvDictBlob::getBlobStr()` returns an empty string with length 1 for an empty blob. Reject empty or potentially truncated strings before strict finite-number conversion. The 10-second expiry and warning text in `dmStatus` are unsuitable for this feature. |
| Text ownership | [acquisition](../../../gui/rtimv/plugins/acquisition/acquisition.cpp) demonstrates a private `QTextEdit` using `rtimvGraphicsView::textEditSetup()`. Shared `statusText()` rows are positioned on the right and can be cleared by other plugins; use a private widget. |
| Global statistics | Local `rtimv/src/librtimv/rtimvGraphicsView.hpp` publicly exposes `globalStatsText()` and `textEditSetup()`; the installed headers do too. `rtimv/src/gui/rtimvMainWindow.cpp::updateGlobalStatsOverlay()` positions global statistics dynamically and does not know about arbitrary plugin widgets. The plugin can follow the existing widget's geometry without changing rtimv. |
| Test/build integration | [ROI plugin tests](../../../gui/rtimv/plugins/tests/roi_commands_test.cpp) provide Qt/Catch tests with fake INDI transport. [Make/rtimvPlugin.mk](../../../Make/rtimvPlugin.mk) supplies plugin build/install rules; the root [Makefile](../../../Makefile) enumerates plugins. Local qmake uses Qt 6; Qt6Widgets 6.10.2 and mxlib 0.2.0 are available. |

## Key Assumptions

- Each slot is independent: the two values may come from different devices or properties. One configured slot is valid; no configured slots unloads the plugin normally.
- Initial scope is finite numeric values. Strings, switches, units, scaling, sign inversion, and expressions are not configuration features in this version.
- Labels remain visible when the value is unavailable, e.g. `x_rms: `, so rows do not move as data becomes available.
- Preserve the published sign and use `label: value` consistently, as approved below. The positive `dx` output and `y_rms=` in the original examples are transcription differences.
- No new keyboard shortcut or independent timer is needed. Use rtimv's overlay enable/disable and update lifecycle.
- Use the existing status font/color settings and a readable translucent background. Native adaptive image-luminance styling does not cover private plugin widgets.
- No companion rtimv change is planned. Build and exercise the plugin against matching rtimv headers/library; do not install or change operational viewer configurations during implementation.

## Requirements

1. Implement a separate `valueDisplay` plugin, independent of camera identity or `cameraStatus`.
2. Support two optional slots, each with a full INDI key, optional label defaulting to the element name, and configurable decimal precision.
3. Register keys once during attachment without clearing existing values (`try_emplace`); share existing property subscriptions and retain values between publications regardless of receipt age.
4. On missing/empty, malformed, truncated, nonfinite, deleted, or disconnected data, show only the label and separator. Do not log warnings for ordinary unavailable values. Configuration errors may receive one startup diagnostic.
5. Preserve numeric signs and round for display only. Approved formatting is fixed-point with three decimal places by default and a per-slot integer precision from 0 through 15; the RMS example can select two places. Never switch to exponential notation.
6. Put values at the upper left, preserving the existing save/loop heading space. When native global statistics are visible, move the values beneath their actual displayed rectangle with a small gap. Do not change statistics settings or overwrite statistics text.
7. Recompute placement during overlay updates and relevant resize/show/hide/move events, without image-coordinate dependence. At practical window sizes both blocks must remain visible and disjoint. If a window is too small for both, clip the value widget to the remaining viewport area; do not draw over statistics or shrink fonts unpredictably.
8. Keep the private widget noninteractive, constrain its width, and preserve help-overlay stacking. Disable/unload must hide or remove only this plugin's widget.
9. Add the plugin to normal build targets, provide configuration examples, and verify its behavior offline.

## Questions and Points of Clarification

1. **Placement:** approve keeping global statistics in their native position and moving the values below them when both are shown? Recommended because the current API supports it without an rtimv change. Keeping values above statistics instead would require cooperative layout support in rtimv.

ANSWER: agree

2. **Formatting:** approve fixed decimal precision per value, defaulting to three places, with signs preserved and `: ` used consistently? With precision 2, the first example reads `x_rms: 0.05` / `y_rms: 0.01`; with precision 3, the second reads `dx: -0.482` / `dy: -1.898`. Also confirm that unavailable values retain their labels, e.g. `dx: `.

ANSWER: yes use 3 to start, fixed, don't switch to e notation. Yes labels with blank values for unavailable.

No additional repository-style ambiguity was found for the proposed files: follow the existing standalone plugin `.hpp`/`.cpp` convention and the Qt/Catch plugin-test convention. Both presentation decisions are approved.

## Tests

Use `gui/rtimv/plugins/tests/value_display_test.{cpp,pro}` with a real `rtimvGraphicsView`, an in-memory dictionary, and a fake provider where lifecycle coverage requires it. Run with `QT_QPA_PLATFORM=offscreen`; never start an INDI connection.

- **Configuration and lifecycle:** absent configuration, either single slot, both slots, different properties/devices, default/custom labels, invalid keys/precision, disable/re-enable, and destruction. Verify existing dictionary entries and another plugin's status text survive attachment and teardown.
- **Formatting:** the two supplied examples with their configured precision; positive/negative/zero values, rounding boundaries, scientific-notation inputs, and bounded behavior for very large values. Verify labels are literal text rather than interpreted HTML.
- **Unavailable values:** a configured overlay with a null dictionary/provider, initial empty dictionary, zero-length blobs, empty strings, malformed input, trailing junk, capacity-sized/truncated strings, NaN/infinity, and recovery. Assert blank value portions without warning output or accidental numeric zero.
- **Publish/subscribe semantics:** a value with an old modification timestamp remains displayed; partial updates preserve the other value. Build the old-value fixture with the documented `rtimvDictBlob` transferring constructor, which retains the payload and initializes the destination timestamp to zero; no sleeps or production API changes are needed. Property deletion, device deletion, and simulated disconnect blank affected rows, with later publications restoring them.
- **Redefinition regression:** a replacement definition omitting a formerly present element clears that element's blob; retained elements update; ordinary partial updates and unrelated properties retain their previous values.
- **Layout:** native stats hidden/shown, moved/resized, then hidden again; view resize, zoom, and pan; heading/help coexistence; no intersection with global stats at normal sizes; bounded clipping at small sizes. Verify plugin enable/disable does not alter native statistics or camera status.
- **Loading/build:** build the plugin with normal qmake rules and optional `RTIMV_INCLUDEPATH`/`RTIMV_LIBDIR` overrides, then load/unload it using Qt plugin discovery offline. Run existing `roi_commands_test` because the provider invalidation fix shares that code.

Inspect an offscreen render with both text blocks enabled. The planning phase used source inspection only; completed build/test results are recorded below. Run `clang-format` on changed C++ files, not Markdown files.

## Implementation Plan

### 1. Confirm presentation and create the feature branch

After review, use `jrmales/value-display-plugin`. Preserve unrelated working-tree content. Keep this plan current with accepted answers and implementation results.

### 2. Add the plugin and its configuration

Create `gui/rtimv/plugins/valueDisplay/valueDisplay.{hpp,cpp}`, `valueDisplay.pro`, and `Makefile`. Use `rtimvOverlayInterface` and its existing IID, with the test metadata guard used by other tested plugins. Store two small slot records; do not introduce a new base class for this limited feature.

Proposed configuration:

```ini
[valueDisplay]
key1=camflowfs-fit.values.x_rms
label1=x_rms
precision1=2
key2=camflowfs-fit.values.y_rms
label2=y_rms
precision2=2
```

`label1` and `label2` above illustrate optional overrides and can be omitted. Missing `key1` or `key2` disables only that slot; configured rows retain slot order. For camwfs:

```ini
[valueDisplay]
key1=camwfs-fit.average.dx
key2=camwfs-fit.average.dy
```

Parse with `configUnused()` following existing plugins. Validate keys and precision before registering dictionary entries. Reuse `QString::toDouble()` with its success result and `std::isfinite()`, then `QString::number()` for formatting. Treat values as plain text and avoid unchecked user-supplied printf formats.

### 3. Implement the private text widget and placement

Use one two-line `QTextEdit` parented to the graphics view, initialized with `textEditSetup()`, and protected against parent-first destruction using the existing `QPointer` pattern. Size from the configured font and content; refresh text only when it changes. Keep the widget below the help overlay in stacking order.

Read `globalStatsText()` visibility/geometry and place the values below it when needed. Otherwise use the upper-left heading clearance already present in the viewer. Observe the necessary view/global-text geometry events and use the normal overlay update as a fallback. Avoid changing the shared `statusText()` array, the global-statistics widget, or rtimv internals.

### 4. Close the provider invalidation gap

In `indiDictionary::receiveProperty()`, clear cached element blobs omitted by a replacement definition of that property. Restrict this to definitions; preserve partial-update semantics and unrelated entries. Add the focused regression test before relying on this behavior in the overlay. No subscription, reconnect, command, or transport redesign is planned.

### 5. Integrate, document, and verify

Add `valueDisplay` to `all_rtimv_plugins` in the root Makefile. Add `gui/rtimv/plugins/valueDisplay/README.md` documenting keys, precision, labels, unavailable data, placement, and both examples. Operational `rtimv_camflowfs.conf` / `rtimv_camwfs.conf` are not present in this checkout; provide snippets for the user rather than inventing their destination.

Add the focused Qt test target and a `valueDisplay_unit_test` group under `application_unit_test` in `tests/groups.dox`, following the existing plugin group. Apply file/function/member documentation rules throughout each changed source file, use explicit Allman control-flow braces, and preserve Doxygen API links through harness indirection. Run clang-format on touched C++ files and restore `{ //mutex scope` where applicable.

Run the focused tests, existing provider/ROI regression suite, plugin builds, offline load check, visual inspection, and `git diff --check`. Record results here. Keep functional changes and this engineering record together; separate documentation-only and formatting-only commits where needed. Any commits must be on the namespaced feature branch.

Expected implementation files:

- New `gui/rtimv/plugins/valueDisplay/{valueDisplay.hpp,valueDisplay.cpp,valueDisplay.pro,Makefile,README.md}`.
- New `gui/rtimv/plugins/tests/value_display_test.{cpp,pro}`; existing `roi_commands_test.cpp` if provider regression coverage is placed there.
- `gui/rtimv/plugins/indiDictionary/indiDictionary.cpp` for definition invalidation and required file-wide documentation review.
- Root `Makefile`, `tests/groups.dox`, and this plan.
- No rtimv repository changes expected with the recommended placement policy.

## Follow-up and Edge Cases

- Both presentation questions are resolved. Installation and real display/instrument verification are user-run follow-up work.
- Two independent properties have no atomic combined snapshot; each row displays its most recently received value. There is no receipt-age expiry or synchronization requirement.
- A value can briefly remain visible until the existing provider detects connection loss; the plugin follows explicit provider invalidation rather than inventing a timeout.
- Fixed decimals can round very small values to zero and produce long strings for large magnitudes. Precision is configurable; clipping bounds long labels/numbers. Scientific/general display modes can be added if needed later.
- This design resolves overlap with native global statistics. Full collision management with movable box-statistics readouts or arbitrary future overlays, adaptive background luminance, more than two values, and interactive configuration are follow-up scope.
- Long labels and very small windows can clip rows. The plugin must not obscure global statistics or intercept image interaction when space is insufficient.

Historical planning verification: original task text above this section preserved byte-for-byte; relative file links resolved; whitespace checks reported no errors. Independent source/design review found no blocking gaps. Only this plan file was edited in that phase.

## Implementation Record

- 2026-11-17: User approved the plan and both presentation decisions. Host rechecked as `mx11`; created `jrmales/value-display-plugin`. Unrelated untracked files remain untouched.
- Implemented the standalone plugin using public rtimv APIs, with one owned text widget and two independent dictionary keys. No rtimv source or interface changes were needed.
- Refined numeric reads to use `getBlob()` instead of `getBlobStr()`: checking the original final terminator also rejects unterminated and embedded-NUL payloads without displaying a plausible numeric prefix. Receipt timestamps are not used.
- Replacement INDI definitions clear omitted element blobs while keeping subscription keys. Omitting `tcsi.teldata.pa` also clears an existing derived north-angle blob. Partial updates remain unchanged.
- New `value_display_test` passed **466 assertions in 9 cases**, including strict numeric parsing, old timestamps, unavailable values and recovery, replacement definitions, geometry events, help stacking, parent-first destruction, style persistence, and loading/unloading the real shared plugin.
- Existing `roi_commands_test` passed **218 assertions in 7 cases**. Both the new `valueDisplay` and updated `indiDictionary` shared plugins built successfully against the installed rtimv headers/library.
- Inspected `/tmp/value-display-overlay.png`: the two value rows are readable below native statistics, with no overlap or clipped text at the tested 640-by-480 viewport.
- Independent implementation review found no blocking issues. All touched C++ files were clang-formatted; the provider's required full-file braces/spacing/comment-placement cleanup is separated from its functional change in commit history.
- The existing local INDI static archives contain gcov instrumentation. Tests and the provider plugin needed the command-local qmake argument `QMAKE_LFLAGS+=--coverage`; no project build setting was changed for this workstation-specific requirement.
- Source/build/test whitespace checks pass. Because the previously untracked task file is added in full, the staged whitespace check reports five existing trailing-space lines in the protected original task text (lines 2, 3, 5, 8, and 16). Those lines are deliberately preserved; the agent-written section has no trailing whitespace.
- All verification was local/offline. No installation, operational viewer configuration change, rtimv source change, or instrument operation was performed. The original task text and both user answers remain unchanged.

### Verification commands

From the repository root:

```sh
make -C gui/rtimv/plugins/valueDisplay -j4
```

In `gui/rtimv/plugins/indiDictionary`:

```sh
qmake indiDictionary.pro 'QMAKE_LFLAGS+=--coverage'
make -f makefile.indiDictionary -j2
```

In `gui/rtimv/plugins/tests`:

```sh
qmake roi_commands_test.pro 'QMAKE_LFLAGS+=--coverage'
make -f makefile.roi_commands_test -j2
QT_QPA_PLATFORM=offscreen ./roi_commands_test
```

In the isolated build directory `/tmp/value-display-tests-build`:

```sh
qmake /home/jrmales/Source/MagAOX-working/gui/rtimv/plugins/tests/value_display_test.pro 'QMAKE_LFLAGS+=--coverage'
make -f makefile.value_display_test -j4
QT_QPA_PLATFORM=offscreen \
  VALUE_DISPLAY_PLUGIN=/home/jrmales/Source/MagAOX-working/gui/rtimv/plugins/valueDisplay/librtimv_valueDisplayPlugin.so \
  VALUE_DISPLAY_RENDER_PATH=/tmp/value-display-overlay.png \
  ./value_display_test
```

No unresolved implementation issue remains. Deployment follow-up: install/rebuild both plugins on the workstation and add the relevant `[valueDisplay]` configuration snippet from the plugin README. Very small windows and long labels/numbers can clip as documented; broader movable-overlay collision handling remains outside this version.

### Appearance follow-up after instrument use (2026-11-17)

The user reports that the merged plugin works, but its font is smaller than native overlays and its background is too dark. This follow-up supersedes the initial fixed-background/status-font choice and the initial expectation that no companion rtimv change would be needed.

- Root causes: the status font defaults to 19 pixels (`14 * 1.4`, converted to integer), while native statistics use a 22-pixel font. The plugin also used a fixed background alpha of 150, which is the maximum opacity of native `fontLuminance`, even over dark image regions.
- The plugin now copies the native global-statistics font/color and follows subsequent font changes on normal overlay refresh. It initializes the background as transparent.
- The companion rtimv change exposes the existing nonblocking `mtxTry_fontLuminance(QTextEdit*, bool)` via `Q_INVOKABLE`. The plugin discovers/caches that method through the existing main-window QObject and invokes it after placing its widget, including refreshes with unchanged values. No image-sampling algorithm, new image reader, timer, overlay-access layout, or interface IID is introduced.
- Older viewers without the invokable method remain supported with a transparent background and no warning. Adaptive opacity requires the updated viewer; it samples the plugin's own footprint instead of copying the statistics widget's background.
- Qt's `setTextBackgroundColor()` changes the insertion format when no text is selected. The plugin therefore saves its cursor, selects its document during native invocation, and restores the cursor. This updates existing rendered text even when the INDI values remain unchanged; tests inspect every text fragment rather than only the current-format getter.
- MagAOX follow-up branch: `jrmales/value-display-appearance`, based on merged `origin/dev` at `26355d9f`. Companion checkout: `local/rtimv-value-display`, based on the clean sibling rtimv `dev` at `4778f30`; the original sibling checkout stays untouched.
- Standalone plugin suite: **554 assertions in 11 cases passed**, including matching native fonts and rendered fragments, runtime font changes, transparent/quiet compatibility with older viewers, host destruction, metadata dispatch after placement, and real shared-plugin loading.
- Built the plugin plus companion `rtimv`, `rtimvClient`, `rtimvServer`, and both GUI test executables. Four native/client global-statistics and viewport-bounds regression tests passed. Two additional actual-plugin integrations (local and loopback client) passed rendered-opacity transitions `0 -> 150 -> 0`, unchanged values, a bright value-display region below a dark statistics region, changed values, and resizing.
- Inspected `/tmp/value-display-native-appearance.png` and `/tmp/value-display-client-appearance.png`: font size matches native statistics and the fixed dark background is gone. C++ formatting and whitespace checks pass. Functional changes, usage documentation, and companion test formatting are committed separately.
- Client tests required sandbox escalation for loopback sockets only. No live instrument access, installation, or operational configuration change was performed. Rebuild the viewer and `valueDisplay` plugin to deploy this follow-up; no configuration changes or additional INDI requests are needed.

Companion verification, from the MagAOX workspace:

```sh
cmake -S local/rtimv-value-display -B local/rtimv-value-display/_build \
  -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=OFF
cmake --build local/rtimv-value-display/_build --parallel 6 \
  --target rtimv rtimvClient rtimvServer rtimvGlobalStatsGuiTest rtimvGlobalStatsClientGuiTest
env -u RTIMV_TEST_VALUE_DISPLAY_PLUGIN -u RTIMV_TEST_SCREENSHOT \
  RTIMV_PLUGIN_PATH=/nonexistent \
  ctest --test-dir local/rtimv-value-display/_build --output-on-failure -j 2 \
  -R '^(rtimvGlobalStats_(gui|clientGui)|rtimvViewportBounds_(gui|clientGui))$'
QT_QPA_PLATFORM=offscreen RTIMV_PLUGIN_PATH=/nonexistent \
  RTIMV_TEST_VALUE_DISPLAY_PLUGIN="$PWD/gui/rtimv/plugins/valueDisplay/librtimv_valueDisplayPlugin.so" \
  local/rtimv-value-display/_build/tests/rtimvGlobalStatsGuiTest
QT_QPA_PLATFORM=offscreen RTIMV_PLUGIN_PATH=/nonexistent \
  RTIMV_TEST_VALUE_DISPLAY_PLUGIN="$PWD/gui/rtimv/plugins/valueDisplay/librtimv_valueDisplayPlugin.so" \
  local/rtimv-value-display/_build/tests/rtimvGlobalStatsClientGuiTest \
  "$PWD/local/rtimv-value-display/_build/rtimvServer"
```

The standalone plugin test uses the same isolated-build commands recorded above with the updated source and plugin binary. The original task text and both user answers remain unchanged.
