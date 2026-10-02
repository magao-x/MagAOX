# Prompt
Create a Git commit to implement rule 18. From now on, always use the standards outlined in Agents.md

Follow-up prompt:

Run a documentation pass across the whole file (Rule 15 of Agents.MD). But first, create a plan for this task in orcaCtrl-Formatting.md and push your plan to GitHub via my current branch

# Plan

The rule 18 work is recorded first, below. The rule 15 documentation pass is planned in
[Rule 15 Documentation Pass](#rule-15-documentation-pass) at the end of this file.

## Analysis

`AGENTS.md` rule 18 says that when an app integrates a `dev::<base-class>` helper, it should use the
interface macros that helper provides. `apps/orcaCtrl/orcaCtrl.hpp` inherits from `dev::stdCamera`,
`dev::frameGrabber` and `dev::telemeter`, but called each base-class hook by hand
(`dev::stdCamera<orcaCtrl>::setupConfig( config )` and so on). It also had no `stdCameraT`,
`frameGrabberT` or `telemeterT` typedefs, which the macros require.

The macros available are:

- `libMagAOX/app/dev/stdCamera.hpp`: `STDCAMERA_SETUP_CONFIG`, `STDCAMERA_LOAD_CONFIG`,
  `STDCAMERA_APP_STARTUP`, `STDCAMERA_APP_LOGIC`, `STDCAMERA_UPDATE_INDI`, `STDCAMERA_APP_SHUTDOWN`.
- `libMagAOX/app/dev/frameGrabber.hpp`: `FRAMEGRABBER_SETUP_CONFIG`, `FRAMEGRABBER_LOAD_CONFIG`,
  `FRAMEGRABBER_APP_STARTUP`, `FRAMEGRABBER_APP_LOGIC`, `FRAMEGRABBER_UPDATE_INDI`,
  `FRAMEGRABBER_APP_SHUTDOWN`.
- `libMagAOX/app/dev/telemeter.hpp`: `TELEMETER_SETUP_CONFIG`, `TELEMETER_LOAD_CONFIG`,
  `TELEMETER_APP_STARTUP`, `TELEMETER_APP_LOGIC`, `TELEMETER_APP_SHUTDOWN`.

`apps/pvcamCtrl/pvcamCtrl.hpp` is the reference stdCamera app that already follows this pattern. The
`*_LOAD_CONFIG` macros `return` an error code, so they must run inside a function that returns `int`.
pvcamCtrl handles this by splitting config loading into `loadConfigImpl()`, which `loadConfig()` calls.

Issues found in the existing hooks:

- `appShutdown()` called `dev::frameGrabber<orcaCtrl>::appShutdown()` twice and never called the
  stdCamera or telemeter shutdown hooks.
- `appLogic()` logged errors from `updateINDI()` and telemeter `appLogic()` and returned 0. The macros
  return -1 instead.
- `onPowerOff()` and `whilePowerOff()` have no macros, so rule 18 doesn't apply to them.

## Implementation Plan

1. Add `stdCameraT`, `frameGrabberT` and `telemeterT` typedefs beside `MagAOXAppT`, each with a `///`
   brief (rule 4).
2. `setupConfig()`: replace the direct calls with `STDCAMERA_SETUP_CONFIG`, `FRAMEGRABBER_SETUP_CONFIG`
   and `TELEMETER_SETUP_CONFIG`.
3. Add `int loadConfigImpl( mx::app::appConfigurator &_config )`, documented with a `///` brief and an
   inline `/**< [in] ... */` parameter comment (rules 3 and 10). Move the config reads into it and use
   the `*_LOAD_CONFIG` macros. `loadConfig()` calls it, and on failure logs a critical message and sets
   `m_shutdown`.
4. `appStartup()`: use `STDCAMERA_APP_STARTUP`, `FRAMEGRABBER_APP_STARTUP` and `TELEMETER_APP_STARTUP`.
5. `appLogic()`: use `STDCAMERA_APP_LOGIC`, `FRAMEGRABBER_APP_LOGIC`, `STDCAMERA_UPDATE_INDI`,
   `FRAMEGRABBER_UPDATE_INDI` and `TELEMETER_APP_LOGIC`.
6. `appShutdown()`: call `FRAMEGRABBER_APP_SHUTDOWN` once, before releasing the DCAM handles the
   framegrabber thread uses. Then call `STDCAMERA_APP_SHUTDOWN` and `TELEMETER_APP_SHUTDOWN`.
7. Leave `onPowerOff()`, `whilePowerOff()` and all other logic unchanged (rule 7).
8. Run `clang-format` (rule 8), build with GCC 14, and commit only `apps/orcaCtrl/orcaCtrl.hpp` as a
   functional commit on the feature branch (rules 17 and 19).

## Open Questions / Assumptions

- Adopting the macros changes behavior: errors from `stdCamera`/`frameGrabber` `updateINDI()` and
  telemeter `appLogic()` now return -1 from `appLogic()` rather than being logged and ignored. This
  matches pvcamCtrl and the macro design, and is recorded in the commit message.
- Adding `STDCAMERA_APP_SHUTDOWN` and `TELEMETER_APP_SHUTDOWN` is treated as part of adopting the macro
  set, not as a separate lifecycle fix.
- `loadConfigImpl()` does not add an empty-serial-number check (pvcamCtrl has one) so config behavior
  stays the same. `camera.serialNumber` is already `argType::Required`.
- The prompt names `agents/plans/2026-09-30/`. Existing plans are grouped by month
  (`agents/plans/2026-06/`), but this plan uses the path the prompt gave.

## Execution Notes

- Before the change, found that `orcaCtrl.hpp` had moved on since the earlier review: commit `ef74fb77`
  ("Correct exp time units to be in seconds") was on the branch and the working copy of `orcaCtrl.hpp`
  was clean. Re-read the affected sections before editing.
- Implemented steps 1-7 as planned. `setupConfig()`, `loadConfigImpl()`, `appStartup()`, `appLogic()`
  and `appShutdown()` no longer call `dev::stdCamera<orcaCtrl>::`, `dev::frameGrabber<orcaCtrl>::` or
  `dev::telemeter<orcaCtrl>::` directly.
- `clang-format` is not on `PATH` on this machine. A dry run from the earlier review that piped the
  command into `grep -c` reported 0 warnings only because the command never ran. Used the binary
  bundled with the VS Code C++ extension:
  `~/.vscode-server/extensions/ms-vscode.cpptools-1.34.4-linux-x64/LLVM/bin/clang-format` (v23.1.0).
- Checked that the committed `HEAD` version of `orcaCtrl.hpp` was already clean under that
  `clang-format`, so formatting churn would not mix into the functional commit (rule 19).
- `clang-format` wrapped the `frameGrabberT` typedef awkwardly with a trailing `///<` comment. Moved all
  three typedef comments onto `///` lines above each typedef, then confirmed a clean dry run.
- Building with the system GCC 11 fails (`Make/common.mk` requires GCC >= 14). Built with
  `source /opt/rh/gcc-toolset-14/enable && make` in `apps/orcaCtrl`; the build succeeded.
- The build still warns about narrowing conversions (`DCAMERR` to `int32_t`) in the existing
  `log<software_error>( { __FILE__, __LINE__, 0, error, ... } )` calls. These existed before the change
  and were left alone.
- The IDE flags that the two `setorcaParameterOnline( ..., int32 )` overloads are declared but never
  defined. This existed before the change and was left alone.
- Committed `20bb855a` ("orcaCtrl: use dev base-class interface macros (AGENTS.md rule 18)"), with only
  `apps/orcaCtrl/orcaCtrl.hpp` staged. Unrelated working-tree changes in `apps/cred2Ctrl/cred2Ctrl.hpp`,
  `libMagAOX/Makefile` and `magaox-python/magaox/deformable_mirror.py` were left out.
- Not tested on the camera hardware.
- Recorded the standing instruction to always apply `AGENTS.md` in the agent's memory. `AGENTS.md`
  itself was not changed, because rule 16 only asks for new style rules to be added there.

## Follow-Up

- Rule 15: do a documentation pass over the whole of `orcaCtrl.hpp` as a separate docs-only commit
  (rule 19).
- Fix the lifecycle hooks that have no macros: `whilePowerOff()` calls `stdCamera::onPowerOff()` instead
  of `whilePowerOff()`, and `onPowerOff()` doesn't call `frameGrabber::onPowerOff()`.
- Fix `reconfig()` so it always releases DCAM buffers after `dcamcap_stop`.
- Add unit tests under `apps/orcaCtrl/tests/`, which `loadConfigImpl()` now makes practical (rule 20).

# Rule 15 Documentation Pass

## Analysis

AGENTS.md rule 15 says that when a file is touched, documentation quality should be brought up to
standard across the whole file, not just the changed lines. Rule 19 says documentation-only changes go
in their own commit, after the functional commits. `apps/orcaCtrl/orcaCtrl.hpp` (about 1800 lines) has
been touched by every recent commit, and its documentation is uneven.

Gaps found by reading the file (line numbers are from 2026-10-01):

- **Rule 3/10, function declarations with no `///` brief or no inline `/**< [in] ... */` parameter docs:**
  - The free helpers `dcamErrorString()` and `dcamDeviceString()` (lines 30 and 49) use plain `//`
    comments.
  - The eight `getorcaParameter`/`setorcaParameter`/`setorcaParameterOnline` overloads (lines 250-268)
    have no documentation. The `commit` argument is accepted but never used.
  - `connect()`, `getAcquisitionState()` and `getTemps()` are undocumented. `closeCamera()` (line 273) has
    only a `//` comment, and its parameter isn't documented.
  - The stdCamera hooks `powerOnDefaults()`, `setTempControl()`, `setTempSetPt()`, `setReadoutSpeed()`,
    `setExpTime()`, `capExpTime()`, `setFPS()` and `setNextROI()` are undocumented or use `//`.
    `setDcamRoi()` uses `//` for its brief and `/** pix units */` instead of `/**< [in] ... */` for its
    parameters.
  - The frameGrabber hooks `configureAcquisition()`, `fps()`, `startAcquisition()`,
    `acquireAndCheckValid()`, `loadImageIntoStream()` and `reconfig()` are undocumented.
  - The telemeter hooks `checkRecordTimes()` and `recordTelem()` are undocumented.
  - `setupConfig()` and `loadConfig()` have briefs, but `appLogic()`'s brief is copied from another app
    ("Implementation of the FSM for the Siglent SDG"), and `appShutdown()`'s brief ("Currently nothing in
    this app") is now wrong.
- **Rule 4, undocumented members:** `m_depth`, `m_frameSize`, `m_camera_timestamp`,
  `m_FrameRateCalculation`, `m_ReadOutTimeCalculation`, `m_otherCamName`, `m_cameraHandle`,
  `m_waitHandle`, `m_cameraName`, `m_cameraModel` and `m_indiP_readouttime`. The doc for `m_frameCount`
  says "circular buffer", but it is the DCAM buffer frame count.
- **Wrong or copied `app::dev` config comments:** `c_stdCamera_fps` (`true`) says "not expose", and
  `c_stdCamera_usesStateString` (`false`) says "expose" and misspells "config". There are mixed `app:dev`
  and `app::dev` spellings, and a stray `///@}` sits indented inside the comment at line 164.
- **Rule 5/9 grouping:** members and methods aren't in named `\name` groups. pvcamCtrl uses
  "stdCamera Interface", "Framegrabber Interface", "PVCAM Interface" and "Telemeter Interface" groups.
  There are duplicate `protected:` specifiers (lines 249 and 345), and declaration groups without blank
  lines between entries (lines 290-297 and 337-342).
- **Class and file docs:** the class `\todo` list (lines 89-90) is copied from another app and out of
  date. The class doc doesn't describe the app (DCAM API, CoaXPress, the `camera.*` config keys, or that
  the cooling method is set with `dcamcfgc`).
- **Comments in definitions that no longer match the code:**
  - The destructor comment talks about "clearing buffers", but the destructor now calls `closeCamera()`.
  - `// convert from msec to sec` next to `m_ReadOutTimeCalculation / 1000.0` (line 1525) is wrong now
    that DCAM times are in seconds. That is a functional bug, listed under Follow-Up; this pass only
    flags it.
  - The comment block above `setNextROI()` contains "Idlw", and its behavior description is stale.
  - `// Fall through check?` and `// print` are leftover notes.

## Implementation Plan

Documentation only. No behavior change, no renames and no deleted code (rules 7 and 19).

1. **Prerequisite:** the uncommitted `closeCamera( bool uninitAPI )` rename in the working tree is a
   functional fix; `HEAD` doesn't compile without it. It should be committed by itself first, along with
   `dcamapi_uninit;` → `dcamapi_uninit();`, so the docs commit contains only documentation. The docs pass
   is written on top of the working tree and not committed until that is done.
2. **File-level:** keep the `\file`/`\brief`/`\author` block. Document the `DEBUG`/`BREADCRUMB` macros.
   Give `dcamErrorString()` and `dcamDeviceString()` `///` briefs, inline `/**< [in] ... */` parameter
   docs and `\returns` lines.
3. **Class doc:** replace the stale `\todo`s with a short description of the app: the ORCA-Quest2 over
   DCAM-API and CoaXPress, the stdCamera, frameGrabber and telemeter interfaces, the `camera.serialNumber`
   (`DCAM_IDSTR_CAMERAID`) and `camera.liquidCooling` config keys, and that the Cooler Type is set with
   `dcamcfgc`. Keep real open items as `\todo`s, for example `setorcaParameterOnline( ..., int32 )` being
   declared but not defined.
4. **`app::dev` configuration constants:** correct the wrong comments, standardize on `app::dev`, and fix
   the misplaced `///@}`.
5. **Members (rule 4):** document every non-trivial member with its role, ownership and units (seconds
   or Hz, as DCAM reports them). Group them into `\name` sections: "Configurable Parameters",
   "DCAM State" and "Acquisition State". Keep declaration order the same within each group.
6. **Methods (rules 3, 5, 10):** add a `///` brief to every declaration, a short `/** ... */` details
   block where the behavior isn't obvious, inline `/**< [in] ... */` / `[out]` parameter docs, and
   `\returns` lines, matching pvcamCtrl's style. Group them into `\name` sections: "MagAOXApp Interface",
   "DCAM Interface", "stdCamera Interface", "Framegrabber Interface" and "Telemeter Interface". Mark
   `[stdCamera interface]` and `[framegrabber interface]` hooks as stdCamera's own docs do. Put a blank
   line between declarations. Merge the duplicate `protected:` only if no declaration's access level
   changes.
7. **Definitions:** fix comments that no longer match the code and the typos. Don't change any statement.
   Where a comment describes a bug, such as the readout-time `/1000.0`, reword it to say what the code
   actually does and add a `\todo`, rather than fixing the code.
8. **Verify:** run the cpptools clang-format (rule 8) and build with `gcc-toolset-14`. Check that
   `git diff` contains only comment, blank-line and access-specifier lines, with no statements changed.
9. **Commit (rule 19):** one docs-only commit with a short message (rule 23), after the prerequisite
   commit, then push to `joshua-liberman/hamsci-app-cpp`. Update this plan's Execution Notes in the same
   commit.

## Open Questions / Assumptions

- The member renames from the earlier review (`m_FrameRateCalculation`, `m_ReadOutTimeCalculation`,
  `m_camera_timestamp`, `getorcaParameter`) count as code changes, not documentation, so they're left
  for a separate refactor commit.
- Removing dead commented-out code (the dssShutter, focus and fxngen blocks, and the fan "forced on"
  block) is a cleanup, not documentation, so it's left in place.
- No Doxygen build is set up for apps in this checkout, so the check is clang-format plus a compile.
  Doxygen rendering isn't verified.

## Execution Notes

- Plan written and pushed before any documentation edits.

## Follow-Up (functional, found during planning)

- **Exposure time is rounded to whole seconds.** In `setExpTime()` and `configureAcquisition()`,
  `long intexptime = m_expTimeSet + 0.5; double exptime = intexptime;` (left over from the
  seconds-units commit `ef74fb77`) truncates any sub-second exposure. `capExpTime()` does the same with
  the readout time, and its log message still says "ms".
- `m_indiP_readouttime` is set to `m_ReadOutTimeCalculation / 1000.0`, but DCAM already reports
  seconds, so the INDI readout time is 1000× too small.
- `closeCamera()` calls `dcamapi_uninit;` without parentheses, so the DCAM API is never uninitialized
  (see the prerequisite in step 1).
- `setorcaParameterOnline( HDCAM, int32, int32 )` and `setorcaParameterOnline( int32, int32 )` are
  declared but never defined.
