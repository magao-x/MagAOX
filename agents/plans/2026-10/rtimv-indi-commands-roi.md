# Task Description
<!-- The first section is filled out by the user 
     
     This is intended as a guide for how to prompt a coding agent to solve a problem.
     
     Make a copy of this template with a new name under agents/plans/<YYYY-MM>. Then
     Fill out each subsection below as needed. Feel free to add additional
     subsections, etc.  

     When complete, commit it on your feature branch and then prompt the agent to review this file.
-->

Update GUI interfaces to camera ROI settings

## Problem Statement
<!-- A brief statement of the problem -->

ROI settings could be more efficient if there was a way to set them relative to the current ROI, and if the rtimv display could be used to update the target ROI.

## Discussion
<!-- Put more expository text here.  Fully describe the expected behavior and rationale. Include references to other parts of the software that serve as examples. Provide a link to an API/SDK manual.   -->

### Relative ROI Setting
Currently ROI target settings must always be entered w.r.t. the full frame, i.e. in absolute coordinates.  But if the current ROI is 64x64, the rtimv GUI shows the center as 31.5,31.5, though the absolute center might be 511.5,511.5.  To move the ROI by 1 pixel, the user must type in 510.5, not 30.5.  This becomes unwieldy for large cameras with arbitrarily centered ROIs. See https://github.com/magao-x/MagAOX/issues/150.  This is not as easy as it sounds.  We need the `current` ROI coords to always reference the full frame.  We also need the ability to send target ROI coords relative to the full frame.  A possible plan:
 - add a "relative target" ROI setting to stdCamera, which sets the target in absolute terms
 - add a checkbox to roiGUI to change from one to the other.  It should default to checked for relative.

### rtimv ROI Setting

We could also enable setting/updating of ROIs from rtimv using the existing plugins.  Currently whenever the target ROI is different from current, a purple box appears on the rtimv display to show the new ROI.  This box is not interactive. A potential dialog is:
 - make box interactive, so user changing size or moving it updates target
 - have a shortcut, maybe `R`, that sets the target ROI using the same rules as a new user box
 - a desired feature is to have the box show a center cross when it is selected, with ROI parameters as is done for user box
 - something like `alt-r` sets the new ROI

To have this work we need to send commands via rtimv.  See https://github.com/magao-x/MagAOX/issues/119, which references https://github.com/jaredmales/rtimv/issues/40.  It might make sense to add a generalized outgoing command interface to rtimv, but it might be the wrong direction.  We need to decide what is the best way to provide a connection from the cameraStatus rtimv plugin to the INDI rtimv plugin.  


## Requirements
<!-- List the requirements that must be met.  A numbered list works best -->

Existing interfaces must be preserved.  We can not change the meaning of target in setting ROI.

## Tests and Metrics
<!-- List any useful metrics that should be targeted by tests -->

Ensure that relative ROI works with even and odd-sized arrays with arbitrarily sized and positioned ROIs.  

## Scope and Caveats
changes to rtimv are in scope, but may not be required.

# Instructions to Agent
<!-- Specific instructions for the agent.  Below is our standard, but you can modify it as needed. -->

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan

Planning status: prepared by GPT-6 (Codex) on 2026-10-10; revised on 2026-10-11 after the user's answers. The relative-coordinate contract, rtimv command interface, and Qt style interpretation are approved. The user approved the revised interaction and its edge cases on 2026-10-11 and requested implementation. Implementation is in progress. Only this section of the document has been edited, and the user's answers are preserved verbatim.

Inspected MagAOX at `d2890680` on `jrmales/rtimv-indi-commands-roi` and the separate local rtimv checkout at `/home/jrmales/Source/rtimv`, `54981bb` on `dev`. Both working trees were clean at the start. Read both repositories' `AGENTS.md` files. All inspection was on workstation `mx11`; no instrument connections or commands were used.

## Task Summary

Add image-relative ROI center entry and interactive ROI selection without changing the existing absolute ROI protocol. Keep the camera's applied ROI and pending target authoritative in `stdCamera`. roiGUI selects how a user enters target centers; rtimv edits the same pending target through `indiDictionary`. Moving/resizing a box stages a target, while a separate action applies it. Starting from a full/default ROI creates a smaller box; starting from a custom ROI preserves its size. If Check adjusts the target, show the adjusted box and require another explicit apply action.

### Findings from the existing implementation

| Area | Finding | Design consequence |
| --- | --- | --- |
| [stdCamera](../../../libMagAOX/app/dev/stdCamera.hpp) | `roi_region_{x,y,w,h,bin_x,bin_y}` contain `current` and `target`. Callbacks stage `m_nextROI` under `m_indiMutex`. `roi_region_check` calls the derived validator; `roi_set` saves `m_lastROI` and calls the derived setter. | Keep all these names and meanings. New relative requests must resolve to `m_nextROI`, without applying the camera ROI. |
| [MagAOXApp](../../../libMagAOX/app/MagAOXApp.hpp) | `indiTargetUpdate()` accepts `target`, or falls back to `current`; it does not support relative coordinates. | Preserve this legacy behavior. Do not change the shared helper's semantics for unrelated devices. |
| [roi widget](../../../gui/widgets/roi/roi.hpp) and [layout](../../../gui/widgets/roi/roi.ui) | Each editor sends a separate absolute target. Reset sends current values back as absolute targets. The widget refreshes every 250 ms and has no coordinate-mode selector. | Add a local checkbox and capability detection; retain absolute current readouts and all existing action buttons. |
| [cameraStatus](../../../gui/rtimv/plugins/cameraStatus/cameraStatus.cpp) | The magenta `StretchBox` is deliberately non-stretchable and non-removable. `R` only prints a diagnostic. The overlay redraws the box from dictionary values. | Enable editing with a distinct local-drag/pending-echo state so polling cannot undo a user's drag or generate outgoing feedback loops. |
| cameraStatus geometry | Current code multiplies centers and sizes by binning, rejects centers equal to zero, and does not consider a binning-only change when deciding whether a target differs. | Use tested coordinate transforms and compare the full ROI. Zero is a valid coordinate; missing data must have a separate validity flag. |
| [indiDictionary](../../../gui/rtimv/plugins/indiDictionary/indiDictionary.cpp) | Owns the existing `pcf::IndiClient`, subscriptions, reconnect timer, and client mutex. It only exposes incoming dictionary data. `handleDelProperty()` currently does nothing, and values can survive a disconnect. | Add outgoing commands here, and track connection/property validity before allowing commands. Cached text alone is insufficient evidence that a camera is available. |
| rtimv plugin interfaces | `rtimvOverlayAccess` exposes the main-window QObject, view, and dictionary, but no command provider. Existing dictionary and overlay interface IIDs are `1.4`. | Recommend an optional command interface routed by the main window, keeping INDI types out of rtimv. |
| rtimv boxes and shortcuts | `addUserBox()` centers a box in the current view with side length one quarter of the smaller visible image dimension. Existing slots already draw a selected box's cross and coordinates. `addStretchBox()` only wires rejection/removal. Lowercase `r` re-stretches the display and can consume Alt+r before plugins receive it. | Reuse the box presentation and creation rules. Add modifier-aware handling for the apply shortcut and an opt-in interactive-box path. Preserve ordinary `r` and existing passive overlays. |
| Test infrastructure | `libMagAOX/app/dev/tests/stdCamera_test.cpp` has a threadless harness, currently with ROI disabled. `gui/widgets/pwr/tests` demonstrates an offline Qt/Catch test executable. rtimv has CMake/CTest tests. | Extend these patterns with real callback and GUI behavior tests, using fake transports rather than an instrument server. |

Additional findings for the 2026-10-11 interaction revision:

- `roi_full_region` publishes only `x/y/w/h`; there is no published default ROI or reliable full/default classification for the plugin. stdCamera already owns `m_default_*`, `m_full_*`, and optional `m_full_currbin_*`. Classify the applied ROI there and publish the result with its reference geometry instead of making the plugin guess from displayed size.
- `newCallBack_roi_check()` resets its request switch to Off **before** calling `checkNextROI()`. Some validators do not publish adjusted targets. A fresh, coherent result after validation is required to decide whether this Alt+R may apply or must wait for another press.
- rtimv uses `rtimvGraphicsView::zoomText()` for ordinary transient messages such as autoscale, centering, and JPEG quality. This is the requested status-message box, distinct from camera telemetry rows and help. Its current geometry is `250 x 30` with a two-second timeout; use a short message and test readability in that existing area.

The task's linked [rtimv issue 40](https://github.com/jaredmales/rtimv/issues/40) calls for a plugin interface for outgoing commands. The browser could not retrieve [MagAOX issue 150](https://github.com/magao-x/MagAOX/issues/150) or [issue 119](https://github.com/magao-x/MagAOX/issues/119); their intent here is taken from the supplied task text, not from independently verified issue comments.

### Coordinate conventions need explicit treatment

The camera implementations do not all use the same ROI units. This must be addressed before reusing the existing overlay formula for outgoing commands:

- `cameraSim`, `andorCtrl`, `picamCtrl`, and `cred2Ctrl` use pre-binning ROI extents and produce smaller binned images. For example, `cameraSim` uses `m_width = m_currentROI.w / m_currentROI.bin_x`.
- `baslerCtrl` writes the ROI dimensions directly to Pylon Width/Height, reads binning-dependent full dimensions, and sets `m_width = m_currentROI.w`. Its ROI coordinates are in the full image at the selected binning.
- `pvcamCtrl` and `zylaCtrl` need a focused audit of SDK offsets, reported dimensions, and readback before assigning a conversion policy. In particular, Zyla's setup adds one to its origin and its readback uses the SDK origin directly; do not silently fix legacy coordinates as part of this feature.
- `qhyCtrl` and `ocam2KCtrl` currently declare `c_stdCamera_usesROI = false`; this work should not expose editable ROI controls for them.

Recommended solution: an explicit per-camera coordinate policy in the shared camera helper, plus additive read-only metadata for clients. Preserve each camera's existing full-frame coordinate convention. Do not infer the policy solely from frame-size ratios, and do not normalize existing public `current`/`target` values to new units.

## Key Assumptions

1. "Relative" means a zero-based position in the currently displayed camera image, not an offset from its center and not a repeated incremental move. Thus `30.5` in the task's 64-pixel example means one displayed pixel left of center.
2. Only center entry changes mode in roiGUI. Width, height, and binning retain their existing meanings and units. rtimv converts a drawn box's displayed dimensions into those existing target units.
3. A relative request is converted once using the currently applied ROI, never using the previous target as its reference. Subsequent current-ROI changes update the displayed relative projection but do not silently move the stored absolute target.
4. Binning and orientation are part of the mapping. Existing camera-side flip handling remains authoritative; the graphics-scene Y inversion is a separate conversion. The editable viewer must display the configured camera's untransformed image geometry.
5. Ordinary target editing remains separate from camera reconfiguration. `checkNextROI()` remains responsible for camera-specific constraints, and `setNextROI()` remains the application path. Several existing validators are no-ops, so generic input validation cannot be delegated entirely to them. A successful Check that changes the target does not authorize Set on the same key press.
6. Existing absolute clients and cameras without the new capability remain usable. New controls fall back to absolute entry/read-only display when relative geometry or command support is unavailable.

## Requirements

1. Preserve existing absolute `current`, absolute `target`, callback fallback behavior, and check/set/full/last/default operations.
2. Add relative-center commands to `stdCamera` only when the app supports ROI, with a clear advertised capability and valid coordinate policy.
3. Default roiGUI to relative entry when supported, visibly label the target mode, and keep its current column absolute. Changing the checkbox must not send a camera command or reinterpret an unfinished edit.
4. Support even/odd, rectangular, arbitrarily positioned, and binned ROIs. Reject malformed/nonfinite values, invalid dimensions/binning, and unavailable coordinate references without corrupting the target.
5. Make the target box movable/resizable, with selected center cross and parameter readouts. Stage changes without applying the ROI. Permit a requested region to extend beyond the currently displayed subframe, subject to camera full-frame limits and validation.
6. On entering ROI editing, use new-user-box placement only when the applied ROI is full/default and no different target is pending. Otherwise reveal the existing target without resizing or moving it. Repeated Shift+R reveals the active box without resetting edits. If Check changes the target, retain the adjusted box, show a status message, and require a subsequent Alt+R before applying.
7. Reuse one INDI connection owned by `indiDictionary`. Keep protocol conversion in that plugin and make command support optional for rtimv/plugins.
8. Do not replay commands after reconnect, report a transport send as a camera acknowledgement, or issue apply after a failed/incomplete target update.
9. Keep updates and QGraphicsItem lifetime on the Qt GUI thread; respect client/dictionary locks and avoid synchronous signal reentry while holding a box lock.
10. Add focused offline tests, perform full-file documentation passes on changed C++ files, run `clang-format`, and keep functional, documentation-only, and formatting-only commits separate as required by `AGENTS.md`.

## Questions and Points of Clarification

Original questions and user answers are retained below. Questions 1 and 2 and the Qt style interpretation are resolved; the revised interaction following these answers is the subject of the requested second review.

1. **Relative-coordinate contract:** approve image-pixel center coordinates, including current binning, while roiGUI width/height remain in the existing camera units? Recommended: yes. Use separate properties `roi_region_x_relative` and `roi_region_y_relative`, each with a `target` element; all existing properties remain unchanged. Keeping relative input separate avoids ambiguity when generic clients send every element of an absolute property.

Answer: yes, agreed.

2. **rtimv command architecture:** approve a small optional command-provider interface in rtimv, implemented by `indiDictionary` and used by `cameraStatus`? Recommended: yes. A plugins-only alternative could register a shared QObject service on the existing main-window QObject, but that leaves discovery and lifetime as a MagAOX-specific convention. A second INDI connection in cameraStatus duplicates connection/subscription behavior; a command queue hidden in dictionary blobs mixes state with actions. Neither is recommended.

Answer: approved, put it in rtimv

3. **Interaction:** approve `Shift+R` (the existing uppercase `R` placeholder) to create/reset and stage one ROI box using new-user-box rules, drag/resize to stage on release, and `Alt+R` to check then apply the latest target? Recommended: yes. Keep target binning unchanged, show the camera-adjusted box after Check, ignore key auto-repeat, and require a fresh apply action after any failure/disconnect. Plain lowercase `r` keeps its current display action.

Answer: the one wrinkle is that when already in an ROI, the most common reason to have an ROI interaction is to move the center without changing its size.  So I think that applying the same conventions as user box on `Shift+R` should be done if in full or default (which can be different), but if in a smaller ROI we should only show the box but not change its size.  This will require some changes to support showing a box when target has not changed when desired.  Agree that binning does not change through this interface.  Let's also include a check if Check changed it -- continue showing it if it changed, with a message in rtimv (in the normal text box for status messages) and require `Alt+R` again in this case.

Style point to resolve before editing: `roi.hpp` uses out-of-class header definitions like neighboring Qt widgets, while the general header/source rule favors `.cpp` for non-trivial code. Proposed interpretation is to preserve this established widget pattern; new reusable non-template geometry/transport code should follow its destination directory's convention. No source has been changed on this assumption.

Answer: agree with your proposed interpretation. Qt code is different from the rest of the repo in style for a reason.

### Revised interaction for second review (2026-10-11)

The following replaces the original question 3 proposal. It incorporates the requested behavior and makes the remaining interaction choices concrete for review.

| Situation | Action | Result |
| --- | --- | --- |
| Applied ROI is full **or** configured default; target equals current; no edit session | Shift+R | Create/select the quarter-view-size box at the viewport center, using user-box rules; stage its center and dimensions. A default ROI smaller than full still takes this path. |
| Applied ROI is a custom/smaller ROI; target equals current; no edit session | Shift+R | Reveal/select the box at the current ROI's center and size. Send no target writes. Moving it later changes center only. |
| A different target already exists, or an edit session is already active | Shift+R | Reveal/select that target without changing its center, size, or binning. This takes precedence over the full/default rule, so repeated presses cannot shrink or reset work. |
| User drags or resizes the box | Mouse release | Stage only changed fields. A move sends centers only; an intentional resize also sends dimensions. Neither operation writes binning. |
| Target passes Check unchanged | Alt+R | Apply once after receiving this Check's successful completed result, or finish editing without reconfiguration if target already equals current. |
| Check changes center or dimensions | Alt+R | Redraw the adjusted target, keep it visible/selected even if it now equals current, show `ROI adjusted; Alt+R to apply`, and do **not** apply. |
| Adjusted target has not been replaced and the user presses again after seeing the result | Alt+R again | Recheck that displayed target. Apply once only if unchanged; another adjustment returns to the same review state with a new message. If already equal to current, finish without reconfiguration. |
| User leaves ROI editing | Escape, while the ROI editor owns the interaction | End explicit editing/confirmation without changing the staged camera target. Return to normal preview visibility: show when target differs, hide when equal. Other uses of Escape remain unchanged. |

Full/default classification is based on the **applied** ROI, including its center and camera coordinate/binning convention, not its area alone or the currently staged target. A same-sized ROI at a different center is custom. Recognize the actual full region at current binning where the camera provides it. Match the configured default's effective geometry and binning, including documented hardware normalization. Unknown classification preserves the existing size; it never triggers automatic shrinking. Invalid geometry disables commands.

The requested size-preservation case is translation-only by default, while deliberate edge/corner dragging still resizes as originally requested. The explicit edit-session flag keeps a box visible when target equals current; focus/deselection and periodic refresh do not clear it. Clear it after confirmed application, Escape, disconnect, or an unrelated applied-ROI change. Keep the adjusted box pinned during the additional-review state. A full-size box must remain selectable at image edges and movable beyond the current subframe; test hit tolerances and scene clipping rather than shrinking its geometry to fit.

Additional choices proposed for this review: preserve a preexisting different target, make repeated Shift+R idempotent, and use Escape to leave editing without reverting the staged target. The approved relative API and command-provider design do not need renewed approval.

## Tests

### Coordinate and stdCamera behavior

- Extend the existing threadless stdCamera harness with ROI enabled, recording derived Check/Set calls and outgoing property updates. Exercise the real callback dispatcher, not only a conversion helper or validation-only macro path.
- Check the task example: current center `(511.5, 511.5)`, size `64 x 64`, binning `1 x 1`; relative target `(30.5, 31.5)` yields absolute `(510.5, 511.5)` and does not call Set.
- Table-test sizes `1`, `2`, `63`, `64`, and `65`, mixed-parity rectangles, noncentral ROIs, centers at zero, and target regions crossing the displayed image boundary. Verify centers and rectangle edges independently to catch half/whole-pixel errors.
- Test pre-binning and post-binning coordinate policies with `1 x 1`, `2 x 3`, and different current/target binning. Verify a one-image-pixel displacement maps to the expected physical displacement and round-trips through absolute and relative coordinates.
- Repeat identical relative requests and prove there is no cumulative motion. Change pending target dimensions/binning, then change the applied ROI, and verify the documented reference/refresh behavior.
- Cover missing elements, wrong device/property/type, empty values, NaN/infinity, overflow, invalid binning/dimensions, disabled ROI capability, invalid reference data, and request failure without partial state changes.
- Verify that the original `target` and fallback `current` writes remain absolute, and that Check, Set, full/full-bin, load-last/last, default, startup, power-off, and readback refresh the relative projection consistently.
- Exercise a validator which snaps a request, one which fails, and one which does nothing. Use existing offline driver harnesses where available for camera-policy integration; no real SDK/device I/O.
- Test full/default/custom classification with default equal to full, default smaller than full, the full region at current binning, a same-sized shifted ROI, and unknown metadata. Check that classification and its reference geometry are published together; no startup defaults may masquerade as valid camera readback.
- Test the post-Check result for unchanged, adjusted, failing, and no-op validators. Require exactly one newer completed result per actual Check, containing the requested and resulting complete ROI plus the applied reference; the legacy Off switch alone is not completion.

### roiGUI

- Use an offscreen Qt test executable patterned on `gui/widgets/pwr/tests`, with a fake publisher capturing outgoing properties.
- Verify checked-by-default relative mode, unchanged absolute current readouts, correct destination properties in both modes, no commands on mode toggle, and unchanged Reset/load-last/Check/Set behavior.
- Verify unsupported/late/deleted capability, reconnect, invalid input, pending edit during a toggle, and current-frame changes during an edit. Ensure a refresh cannot overwrite focused user text or submit it in the wrong coordinate mode.

### rtimv and plugins

- Test an optional fake command provider for provider-before-overlay and overlay-before-provider loading, no provider, duplicate providers, wrong command types, exceptions/send failures, disconnect, teardown, and reconnect without replay. Verify property element grouping and order on a recording fake INDI client.
- Test a synthetic graphics scene with image dimensions known independently of ROI properties. Verify zero-based pixel/scene conversion, pen-width-independent geometry, selected cross/readout, move/resize release, and box creation at the viewport center under zoom/pan. A custom ROI equal to current must remain visible/selected without sending any command, including at image edges and across refresh/deselection.
- Exercise every row in the revised interaction table: full and separate default startup, custom ROI translation, preexisting different target, repeated Shift+R, unknown classification, deliberate resize, and Escape. Assert unchanged width/height for pure translation and zero outgoing binning writes in all cases.
- Verify passive old plugins still load; passive boxes remain passive; plain `r`, `Shift+R`, and `Alt+R` do not conflict. Include empty-text/modifier-only events, focus in text editors, and auto-repeat.
- Test that incoming target updates never issue commands, polling does not reset a drag, and release sends one ordered target update sequence. Programmatic rectangle updates must not trigger sends.
- Verify target-write failure, stale/current geometry change, partial echoes, Check failure, corrected targets, timeout, reconnect, and binning-only differences. Apply must wait for the final target/check result and occur at most once per explicit action. Ignore stale/duplicate completion, auto-repeat, and presses while checking/applying; an early second press must not preauthorize a later adjusted result.
- Count Check and Set requests: an unchanged result different from current gives one Check/one Set; an adjustment gives one Check/zero Set and the normal status message; a fresh second press with unchanged recheck gives a second Check/one Set, unless target already equals current (zero Set). Further adjustment, target replacement, binning adjustment, failed Check, or disconnect must never consume an old confirmation.
- Verify adjusted-to-current and adjusted-to-full/default cases retain the box until review is resolved. Compare complete numeric ROIs rather than rounded on-screen labels. Check that text timeout, zoom messages, ordinary deselection, and camera telemetry refresh neither apply nor reset confirmation; the short adjustment message must fit the existing status box.
- Run a workstation-only smoke test using generated images and fake transport; optionally use an isolated local cameraSim/INDI setup whose endpoint is verified unrelated to instrument tunnels. Any actual-instrument acceptance steps must be run by the user and reported back.

Expected verification entry points after implementation: the focused stdCamera test via `tests/Makefile.one`; new Qt widget/plugin test targets with `QT_QPA_PLATFORM=offscreen`; focused rtimv CTest targets; builds of roiGUI, cameraGUI (which embeds the ROI widget), the two affected plugins, and rtimv's local and available gRPC GUI variants. Record exact commands/results and unavailable dependencies here during implementation.

## Implementation Plan

### 1. Establish the coordinate contract and reusable conversion

Audit all ROI-enabled drivers listed above and record each one's units, origin, binning behavior, and flip handling. Add a small shared ROI geometry utility under `libMagAOX` usable by stdCamera and the cameraStatus plugin without pulling in the full app framework. Reuse the existing center/extent conventions, including the focused C-RED 2 helper where relevant; do not make a camera-specific helper the cross-camera abstraction.

Use an explicit policy to map each legacy axis center `q` to a common physical pixel-center coordinate `A`, and an extent to a physical span. For the verified zero-based conventions:

```text
pre-binning ROI units:  A = q;                 span = extent
post-binning ROI units: A = b * (q + 0.5) - 0.5; span = b * extent

Ac = current physical center
At = target physical center
bc = current binning
N  = current displayed size on this axis
r  = target center in current-image pixel coordinates

At = Ac + bc * (r - (N - 1) / 2)
r  = (At - Ac) / bc + (N - 1) / 2
```

Invert the target camera policy at its target binning to obtain the existing absolute target value. For unbinned `64` pixels, `(N - 1) / 2` is `31.5`; for `65`, it is `32`. Use floating-point arithmetic throughout conversion and let validated pixel-edge snapping/driver constraints determine representable rectangles. Do not truncate a center before evaluating parity.

For rtimv's image scene, box geometry represents pixel boundaries. Its user-box display subtracts `0.5` from the X center and reverses Y with another `0.5` offset. Use:

```text
image center x = scene rectangle center x - 0.5
image center y = image height - scene rectangle center y - 0.5
```

Use the mapped item rectangle, not `sceneBoundingRect()` with pen expansion, for commands. Convert width/height using the current image's physical pixel scale and the target policy. Require the displayed dimensions to agree with the declared camera geometry before enabling writes. Audit any nonzero origin or extra image transformation explicitly rather than guessing.

Publish additive read-only coordinate-policy/capability metadata from stdCamera (proposed name `roi_coordinate_system`). The property should identify the declared units and any necessary origin convention. Both relative properties must be defined and valid before the GUI enables relative entry. Do not add a camera-name switch in the plugin or a new mandatory CRTP method in every app.

### 2. Add relative requests and synchronized projections in stdCamera

- Register and dispatch `roi_region_x_relative.target` and `roi_region_y_relative.target` for supported cameras. These properties contain relative targets only; do not introduce relative `current` elements.
- Validate the request and current reference under `m_indiMutex`, convert once, and update the corresponding absolute `m_nextROI` center. Publish the resulting original absolute target and relative projection through existing update helpers.
- Keep the absolute ROI as the only stored target. Calculate relative values from that target and current geometry; do not maintain an independently authoritative relative target.
- Centralize projection refresh and call it after relevant callbacks/checks/presets and from shared periodic/power lifecycle paths so derived app readback and direct target updates cannot leave the relative display stale.
- Advertise appropriate relative limits, including negative/outside-subframe coordinates which can still lie in the full frame. Do not copy absolute minima to relative properties or clamp to the current image bounds.
- Mark relative capability unavailable when the reference is invalid. Keep legacy callbacks and camera-specific rounding/apply behavior unchanged.

Add two read-only outputs to support the revised interaction, preserving all existing writable property contracts:

- **Applied ROI classification** (proposed `roi_region_state` Number property): publish the current `x/y/w/h/bin_x/bin_y`, `is_full`/`is_default` flags, and a separate `preset_known` flag as one coherent snapshot. Property state describes geometry validity; unknown preset classification alone must not disable size-preserving editing. Compute flags from the camera's existing full/default metadata and explicit coordinate policy. Both preset flags may be true. Derive the current-binning full geometry only where supported; otherwise report an unknown classification rather than assuming it from dimensions. Treat a configured default that hardware adjusts according to a documented driver normalization consistently, without running a mutating validator just to classify it.
- **Completed Check result** (proposed `roi_check_result` Number property): after `checkNextROI()` returns, publish a monotonically increasing completion sequence, the full requested ROI, the full resulting ROI, and the applied reference ROI captured under the same lock. Use Ok/Alert state for success/failure, and publish even if no values changed or the derived validator is a no-op. This snapshot determines whether Check changed anything. Retain the old request-switch reset behavior; clients must not treat its Off value as completion. Invalidate completion/reference state on reconnect/power lifecycle changes.

The provider must deliver these as complete property snapshots to cameraStatus, preserving the message's state and connection generation. Independently polling per-element dictionary blobs cannot establish a coherent Check result. Use typed plugin notifications without introducing INDI types into rtimv. Correlate a newer completion with the captured requested target and applied reference; a mismatch invalidates the pending action. This adds reliable completion information but does not make the legacy multi-client protocol atomic.

### 3. Add roiGUI's coordinate selector

Add a checked-by-default checkbox labeled `Relative to current image` near target centers. Subscribe to the new properties and metadata. Display backend-reported relative target values while checked; send only to the relative properties. While unchecked, retain the exact existing absolute paths. Current readouts, width/height/binning, and Reset/preset actions retain their old meanings.

Maintain capability/reference validity separately from numeric values. For old cameras, show absolute mode and disable the relative option with a short explanation. Before toggling, preserve or discard an unsubmitted edit explicitly without accidentally firing `editingFinished` into the new mode. If the applied coordinate reference changes during an edit, invalidate that edit and show the refreshed value rather than reinterpret it silently.

### 4. Introduce the optional rtimv command interface

Add an optional, separately identified `rtimvCommandInterface` implemented alongside `rtimvDictionaryInterface`. A compact command contains provider/protocol identity, device/property names, a Number or Switch element vector, and an identifier for reporting send results. Support grouped property elements and ordered requests; avoid a shell-command/string-expression API. Text/BLOB transport is not required by this ROI feature.

The main window discovers providers during normal plugin loading and routes requests from overlays. Use the existing main-window QObject access for signal/slot connections and an optional command-consumer interface if necessary. Keep the existing `1.4` base interfaces and `rtimvOverlayAccess` layout unchanged if feasible; new interfaces get their own IID. If implementation requires changing the old layout/vtable, version it explicitly and rebuild all dependent plugins rather than treating that as binary-compatible.

Provider lookup must not depend on directory load order. Missing/ambiguous providers produce a visible failure, with the camera overlay still available for display. Define object ownership and disconnect connections on teardown. The command result reports local validation/transport submission only; later INDI properties report the camera's actual state. This interface stays in the GUI/plugin layer and does not add camera control RPCs to the rtimv image server.

### 5. Implement the INDI command provider

Have `indiDictionary` translate typed requests into `pcf::IndiProperty` and reuse `m_client->sendNewProperty()`/`sendNewProperties()`. Guard client access with the existing lifecycle mutex; serialize target sequences and stop on send failure. Do not hold dictionary/graphics locks while sending or emitting completion signals, and do not block the GUI waiting for property echoes.

Track a connection generation and valid property definitions. Delete-property and disconnect handling invalidate command capability and stale values; reconnect requires fresh definitions/current state. Propagate permission/type/availability information needed by ROI controls. Drop pending outgoing work on disconnect instead of resending it. Keep state blobs as incoming state; do not use them as an outgoing queue.

The shared dictionary's structural lock is currently private to the provider while overlays access the map directly. Avoid adding unsynchronized insert/erase paths. Prefer invalidating existing blobs and delivering new command/state notifications on the GUI thread; if a shared access helper is needed, scope it to the touched paths and document the remaining dictionary concurrency limitation.

### 6. Make cameraStatus's ROI box interactive

Implement the revised interaction table above, with explicit edit visibility independent of `target != current` and selection independent of that visibility. Reuse the single magenta box and rtimv's existing cross/readout and maintain-center resize behavior. Share the new-user-box placement calculation for the full/default case; do not duplicate it or reuse it for the custom-ROI case.

Track idle, editing, dragging, awaiting target echo, checking, awaiting adjusted-target confirmation, and applying. During a drag, update local presentation only. On release, compare against the captured target and stage only changed centers/dimensions in a defined order; preserve exact sizes for pure movement and never send binning writes. rtimv's `moved` signal fires during movement and release, so use an explicit edit-finished hook rather than treating every signal as release. Compute absolute centers from captured current geometry to avoid reinterpreting separate relative-axis requests against different current frames.

For each Alt+R:

1. Require valid command/current/target state and no Check/Set in flight. Finish staging any final edit and wait for fresh target acknowledgement. Capture the complete staged target, applied reference, and latest completion sequence, then request Check. Disable new edits until its result arrives; do not queue an additional apply press.
2. Accept only a newer successful complete Check result matching that request's target/reference and connection generation. Compare all six numeric ROI fields, including binning, using exact integer comparisons and only a tightly bounded serialization tolerance for centers. Never decide "unchanged" by comparing formatted labels.
3. If center or dimensions changed, render the returned target and enter adjusted-target confirmation. Keep the box visible even when equal to current. Show `ROI adjusted; Alt+R to apply` in the normal transient status box and send no Set. The message's normal timeout does not dismiss the box or authorize application.
4. A subsequent non-repeat Alt+R starts a fresh Check of the displayed adjusted target. Apply once only if that Check leaves it unchanged. If adjusted again, show the new result and require another press. No key pressed before the adjustment was displayed counts as confirmation.
5. If unchanged, send `roi_set.request=On` once and await applied readback. On confirmed success, end explicit editing and return to normal preview visibility. If target already equals current, end editing as a no-op without reconfiguring the camera. On failure, leave the target available for review and require a fresh action.

If Check changes binning, show the result and stop this apply flow with a message directing the user to roiGUI; the rtimv editor must not implicitly authorize a binning change. A pending binning change explicitly staged by another client is preserved, but an external binning/target change during this flow invalidates its confirmation.

Use the main-window QObject to expose a small reusable status-message slot wrapping `zoomText()` and the existing font-luminance adjustment, following `showQualityMessage()` and the toggle handlers. Keep the same text box, styling, and normal timeout. Do not put the adjustment notice in camera telemetry rows, open help, or create a dialog. Verify the short text fits the current box. Unrelated transient messages may replace it; the pinned adjusted box and confirmation state remain until resolved.

Invalidate confirmation if current geometry changes, the target is edited locally/replaced externally, the camera leaves an acceptable state, a required property disappears, or a timeout/send failure occurs. A local edit starts a fresh check cycle; a repeated Shift+R without edits only reveals the same box. Escape cancels unsent local drag state and pending apply authorization, keeps already staged targets, and returns to passive preview. It cannot cancel a Set already submitted; track that request until readback/failure and describe its outcome accurately.

Clear drag/selection pointers on deletion, update all graphics on the GUI thread, and prevent programmatic redraw/selection from sending commands. Route Alt+R before the main window's lowercase `r` action; use exact modifiers, guard text access, and suppress auto-repeat. Handle Escape only when ROI editing owns the interaction, and do not intercept text-field editing. Include the conditional Shift+R behavior, adjusted-target second press, and Escape in help/plugin information.

### 7. Integrate, document, and verify

Expected MagAOX changes:

- `libMagAOX/app/dev/stdCamera.hpp`, a small shared ROI geometry helper, and focused tests under `libMagAOX/app/dev/tests` (or the helper's matching test directory).
- ROI-enabled camera headers only where an explicit coordinate policy/refresh integration is needed, notably `apps/baslerCtrl/baslerCtrl.hpp`; confirm the remaining driver list during the audit.
- `gui/widgets/roi/roi.hpp`, `roi.ui`, and a new offline ROI widget test target.
- `gui/rtimv/plugins/cameraStatus/cameraStatus.{hpp,cpp}` and `gui/rtimv/plugins/indiDictionary/indiDictionary.{hpp,cpp}`, plus focused plugin test/build definitions.
- `tests/tests.list` and `tests/groups.dox` as needed, and this plan as the engineering record.

Expected rtimv changes:

- An optional command-interface header, complete-property notifications, command routing, and a reusable existing-status-box message slot in `src/gui/rtimvMainWindow.{hpp,cpp}`; change `src/librtimv/rtimvInterfaces.hpp` only if needed for interface exposure.
- An opt-in interactive-box integration/edit-finished hook in `src/librtimv/StretchBox.{hpp,cpp}` or its shared graphics-item helper, only where existing public hooks are insufficient.
- Matching focused tests, `tests/CMakeLists.txt`, and user/plugin interface documentation.

Use `jrmales/rtimv-indi-commands-roi` for MagAOX. Before any rtimv commits, create a namespaced feature branch there; it is currently on `dev`. The sibling checkout is outside this session's writable roots, so use a workspace-contained checkout for implementation if write access is not available. Do not install either build or change instrument configuration.

Keep behavioral changes with their tests and relevant plan updates in clean functional commits, followed by full-file documentation cleanup and a separate formatting-only commit where needed. Attribute commits to the actual model per `AGENTS.md`. Use `application_unit_test` for application/GUI suites and the existing `app_unit_test` lineage for stdCamera. Document each test case and preserve Doxygen links with the local test-reference pattern where indirection hides the real API.

Run the focused tests/builds described above and inspect formatting/documentation across every changed source file. Record exact results, any unsupported camera policies, dependency versions, and required rollout order here. A new plugin/core interface must be built against matching rtimv headers/library before deployment.

## Follow-up and Edge Cases

- **Concurrent clients:** the legacy separate-property protocol is not an atomic multi-client ROI transaction. Ordered sends and fresh echoes prevent known local partial-update mistakes, but another client can still change a target between validation and apply. Full compare-and-apply semantics would require a separate versioned aggregate command; do not claim this plan provides that guarantee.
- **Changing reference during entry:** stdCamera interprets a relative write against the applied ROI when the callback runs. GUI invalidation reduces stale edits but cannot eliminate network races without a reference-generation field. Document this behavior; introduce such a field if strict reference binding is required.
- **Camera quirks:** no general rewrite of existing SDK offsets, crop modes, or check algorithms is planned. Unsupported/unverified mappings must keep interactive commands disabled until audited; any required legacy behavior correction should be proposed separately.
- **Image transforms:** remote image transport should work with the same client-side plugins, but rotated/resampled/derived streams require an explicit image-to-camera mapping. Matching dimensions alone cannot prove image identity. Use the configured camera/stream association and disable writes for unsupported transformations.
- **Dictionary concurrency:** adding commands must not worsen existing unsynchronized map access. A full dictionary redesign is beyond this feature unless focused tests show it is required for the new behavior.
- **Availability and timeouts:** a send result does not prove camera acceptance. Disconnect, deletion, power transitions, stalled acknowledgement, and adjustment to an unrepresentable box must leave clear failure/read-only state and require a new explicit apply.
- **Adjusted-target review:** confirmation belongs to the exact displayed target and current reference, not to the visibility or lifetime of a status message. Non-idempotent validators may require repeated reviews; do not bypass rechecking to force application. If Set or hardware later adjusts geometry beyond Check's result, display the actual readback and record that driver limitation; guaranteeing identical Check/Set normalization requires driver-specific follow-up.
- **Full/default recognition:** an offset ROI with the same size as default is custom. Missing or unverified preset classification uses the size-preserving path. Do not query or mutate hardware to classify presets. A known but hardware-normalized default needs a deterministic driver mapping or recorded effective default before it can take the new-box path reliably.
- **Hardware acceptance:** after offline verification, the user should validate signs/flips, binning policies, and hardware snapping on representative cameras. The agent must not run on instrument computers or send commands through instrument tunnels.

Historical planning verification (before implementation approval, 2026-10-11 revision): source inspection only; no application tests/builds or hardware operations have been run because the user requested another review of the revised interaction. Verified that the original task text and all four user answers are unchanged, all requested planning sections and local links remain valid, and `git diff --check` passes. Only this Markdown file is modified; the rtimv checkout remains clean. `clang-format` does not apply to this Markdown-only change.

### Implementation record

- 2026-10-11: All review decisions are approved. Working in MagAOX on `jrmales/rtimv-indi-commands-roi` and a separate rtimv clone at `local/rtimv-roi` on the same branch name; the original sibling checkout is untouched.
- Coordinate audit: sensor units cover cameraSim, Andor, PICam, PVCAM, and C-RED 2. Basler explicitly selects binned units. Zyla remains unavailable for the new coordinate-dependent controls because its existing SDK origin/readback is inconsistent; its legacy absolute interface is preserved.

- Implemented camera relative target properties and explicit sensor/binned/unsupported policies, coherent current plus target classification, and completed Check snapshots. All original absolute callbacks and preset actions retain their meanings.
- The optional `rtimv.commandInterface/1.0` keeps the existing 1.4 interfaces and overlay-access layout unchanged. A main-window QObject child routes typed batches through exactly one provider. The INDI receive thread queues complete messages onto the GUI thread; overlays read immutable snapshot copies there. This replaces the proposed notification delivery with nonblocking snapshot polling, matching the existing overlay update loop. Definitions, permissions, generation, and latest-message completeness are retained; merging a partial update never makes it a complete Check result.
- ROI editing uses native absolute commands computed against captured geometry, so separate axis requests cannot be reinterpreted as relative to different frames. Movement writes centers only; intentional resize may also write dimensions. Binning is never written. The five-second local timeout revokes authorization without replay. Escape preserves staged targets and continues tracking an already submitted Set.
- roiGUI queues incoming updates onto its GUI thread, defaults to relative entry when writable definitions and a valid reference are present, and cancels unsubmitted center text on mode/reference changes. The preference survives temporary capability loss. Legacy cameras retain absolute mode.
- rtimv supplies shared new-box placement, optional interactive-box integration, mouse edit start/release hooks, and a normal status-message slot. Long messages fit inside the existing transient status box; short zoom messages restore the configured font. The magenta box remains visible through adjusted-target review independently of the status timeout.
- Initial offline verification: stdCamera ROI 529 assertions/3 cases, existing stdCamera 117 assertions/4 cases, roiGUI 30 assertions/2 cases, plugin integration 131 assertions/5 cases; all passed. rtimv builds both local and gRPC clients/server and passes all three CTest targets, including real mouse hooks, router lifetime, message fit, existing MTF, and asynchronous client tests. No instrument commands, network connections, installation, or deployment were used.
- Build environment: GCC 15, Qt 6.10.2, mxlib 0.2. Existing instrumented libMagAOX objects require `TESTLIBS=-lgcov` for unit tests and `EXTRA_LDFLAGS=--coverage` for cameraSim linkage. Basler/Andor SDK headers are absent on this workstation; policy assignments are covered by shared geometry tests, but those SDK applications need their normal build environment.


### Build and rollout notes

1. Build the companion rtimv branch first, then rebuild both MagAO-X plugins against its matching headers and shared library. The implemented clone is `local/rtimv-roi`; the original sibling checkout remains unchanged. No remote branch was pushed.
2. Build/deploy updated camera applications and roiGUI through the normal user-managed workflow. The new properties are additive. Older cameras retain absolute roiGUI entry and a passive overlay. Zyla advertises unsupported geometry; no legacy SDK offset corrections are included.
3. Use the configured camera's own image. Sensor and binned unit conversion is explicit; camera-native flip handling remains in the existing drivers, with graphics-scene Y inversion applied separately. Dimensions cannot identify a rotated or derived stream. No calibration is inferred from image appearance.
4. Full/default recognition compares all six native values against existing metadata. Unknown classification or a default adjusted by an unverified hardware normalization preserves the custom ROI geometry on Shift+R. This conservative fallback avoids calling a mutating validator for classification.
5. CameraSim compiles successfully on this workstation. Representative real-camera signs/flips, SDK snapping, and Check-versus-Set normalization still require user-run acceptance testing. Agents must not perform those hardware steps.

Focused offline commands, from the indicated directories:

```sh
# MagAOX/tests (existing archive in this workspace contains coverage-instrumented objects)
make -f Makefile.one t=../libMagAOX/app/dev/tests/stdCamera_roi_test TESTLIBS=-lgcov
../libMagAOX/app/dev/tests/stdCamera_roi_test
make -f Makefile.one t=../libMagAOX/app/dev/tests/stdCamera_test TESTLIBS=-lgcov
../libMagAOX/app/dev/tests/stdCamera_test

# MagAOX/gui/widgets/roi/tests
qmake roi_test.pro
make -f makefile.roi_test
QT_QPA_PLATFORM=offscreen ./roi_test

# MagAOX/gui/rtimv/plugins/tests, with matching installed rtimv headers/library
qmake roi_commands_test.pro
make -f makefile.roi_commands_test
QT_QPA_PLATFORM=offscreen ./roi_commands_test

# Companion rtimv checkout
cmake -S . -B build -DRTIMV_INSTALL_SYSTEMD_UNIT=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

The plugin and plugin-test qmake targets also accept `RTIMV_INCLUDEPATH` (directory containing `rtimv/` headers) and `RTIMV_LIBDIR` for an uninstalled workspace build. The Qt suites use their own `.pro` targets, matching existing GUI test practice; they are not added to the non-Qt `Makefile.one` test runner.

### Additive property contract

| Property | Elements and meaning |
| --- | --- |
| `roi_region_x_relative`, `roi_region_y_relative` | Writable `target` only, in zero-based currently acquired image pixels. Availability requires Ready/Operating, verified units, and valid applied/target geometry. |
| `roi_coordinate_system` | Read-only `units`: 0=sensor pixels, 1=binned full-image pixels, 2=unsupported. Existing absolute values keep their native units. |
| `roi_region_state` | Read-only `x,y,w,h,bin_x,bin_y`, corresponding `target_` fields, `units`, `is_full`, `is_default`, `preset_known`. One complete property carries the applied reference and staged target. Idle means unavailable for commands. |
| `roi_check_result` | Read-only `requested_`, `checked_`, and `current_` versions of all six fields plus monotonic `sequence`. Ok/Alert is published after validation returns, including no-op validation. Idle invalidates a previous completion. The old request-switch reset is not an acknowledgement. |

Current/reference values and target centers are stored at the camera's existing float precision. Snapshots are parsed at that same precision and all dimensions/binning are compared exactly; formatted labels do not participate in confirmation. Local command errors and later camera failures are separate outcomes. The legacy protocol still cannot atomically compare-and-apply across multiple clients, or strictly bind a relative GUI write to a remote reference generation.

- Documentation pass completed for touched declarations and member state. Existing main-window inline accessors were moved to the source file as required by the companion repository. Documentation and formatting are committed separately from behavior.

- Final offline verification after documentation and clang-format: all four MagAOX focused suites pass (529 + 117 + 30 + 131 assertions), and all three companion rtimv CTest targets pass. Both GUI client variants/server, cameraSim, roiGUI, and the loadable plugins are built locally without installation. The new rtimv test exercises the real mouse press/move/release handlers and verifies that the full adjustment message fits the existing status box. The original task text and all four user answers remain unchanged.
