# Prompt
Create a Git commit to implement rule 18. From now on, always use the standards outlined in Agents.md

# Plan

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
