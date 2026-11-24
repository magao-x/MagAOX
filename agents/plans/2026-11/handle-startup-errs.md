# Task Description
The orcaCtrl app is throwing critical errors during startup

## Problem Statement
When the Hamamatsu camera is powered on, the logs throw critical errors during startup, despite the camera powering on without issue.

## Discussion
- the app throwing the errors is ``/home/jliberman/source/MagAOX-ham/apps/orcaCtrl``
- the logs report critical errors for about 3-seconds at startup before the camera config properly loads
- the issue may be occurring within ``orcaCtrl.hpp``'s ``appStartup()`` or within ``connect()``

## Requirements
- OrcaCtrl app startup should haved a longer wait time after power on before polling the camera status/config. Otherwise, the log falsely reports crit errors. 
- only the powerOn logic of the app should be modified

## Tests and Metrics
- write test to check that camera powers on cleanly without throwing crit errors 
## Scope and Caveats
<!-- Add any needed caveats and scope restrictions -->

# Instructions to Agent
<!-- Specific instructions for the agent.  Below is our standard, but you can modify it as needed. -->

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan

## Task Summary
At power-on, `orcaCtrl` (camscc) logs errors for roughly 3 s before the camera finishes coming up, even though the camera
then runs normally. The goal is to stop these false errors by changing only the power-on path, most likely by waiting
longer before the app first talks to the camera, and to add a test showing a clean power-on.

## Key Assumptions
- The app is `apps/orcaCtrl` in this repo, on branch `joshua-liberman/hamsci-app-cpp`. The path in the Discussion
  (`/home/jliberman/source/MagAOX-ham/apps/orcaCtrl`) does not exist on this workstation.
- "crit errors" may mean ERR-level logs (`software_error`, `LOG_ERROR`) and not literally `LOG_CRITICAL`. No
  `LOG_CRITICAL` call in orcaCtrl, `stdCamera` or `frameGrabber` sits on the power-on/connect path (see Findings).

## Findings
How power-on works now:
1. `MagAOXApp::execute()` sees power go 0 -> 1, resets `m_powerOnCounter = 0`, and goes to `POWERON`.
2. `stdCamera::appLogic()` stays in `POWERON` until `powerOnWaitElapsed()` returns true (time = `m_powerOnWait` sec,
   counted in main-loop passes). Then it goes to `NOTCONNECTED` and calls `powerOnDefaults()`.
3. `orcaCtrl::appLogic()` then calls `connect()` every loop until it succeeds. After that it goes to `CONNECTED`, then
   `getAcquisitionState()`, `setTempSetPt()` and so on.

What this means:
- **orcaCtrl does not set `m_powerOnWait`.** It inherits the `MagAOXApp` default of **55 s**
  (`libMagAOX/app/MagAOXApp.hpp:1230`). The `power.powerOnWait` config help text says the default is 0, which is wrong.
  If camscc's config sets `power.powerOnWait` low, or to 0, the app talks to the camera almost as soon as power comes on.
  If the config does not set it, the app already waits 55 s, and a longer wait probably will not help.
- Other cameras set their own default in the constructor: `zylaCtrl` uses `m_powerOnWait = 10` and `pvcamCtrl` uses 15.
  The config file can still override this.
- Errors that `connect()` and the first `appLogic()` passes can log while the camera is still coming up:
  - `dcamapi_init` failure -> `software_error` "Failed to initialize DCAM API." and `ERROR`, retried every loop
    (`orcaCtrl.hpp:1245`).
  - `dcamdev_open` failure -> `software_error` "Could not open DCAM device N" (`orcaCtrl.hpp:1286`).
  - `dcamwait_open` / fan `dcamprop_getattr` failure -> `software_error` (`orcaCtrl.hpp:1318`, `:1338`).
  - Any of these makes `appLogic()` add its own `software_error` (`orcaCtrl.hpp:977`).
  - **Likely candidate:** in `CONNECTED`, `getAcquisitionState()` finds capture not yet running and logs
    `"acquisition stopped. restarting"` at `LOG_ERROR` (`orcaCtrl.hpp:1466`). It logs again on every `READY` pass until
    the framegrabber thread allocates buffers and starts capture. That fits "errors for ~3 s until config loads".
    This message is not tied to power-on timing, so a longer `powerOnWait` would **not** remove it.
- Leftover `std::cerr << __LINE__` debug prints remain in `appLogic()` and `connect()` (`orcaCtrl.hpp:971`, `:1255`,
  `:1260`).
- The DCAM SDK headers (`/opt/hamamatsu_new/.../dcamsdk4/inc`) are not installed on this workstation. orcaCtrl cannot be
  compiled here, and a test that includes `orcaCtrl.hpp` needs stub DCAM headers. `tests/pvcam/` does this for pvcam.

## User Answers (2026-11-15)
1. The user can only find logs from software restarts, not from a camera power-on.
2. camscc config has `power.powerOnWait = 5`.
3. Suppressing the first `"acquisition stopped. restarting"` after power-on is in scope.
4. A fake DCAM layer for tests is acceptable.
5. Removing the leftover `std::cerr << __LINE__` debug prints is approved.

Follow-up finding: on a software restart with the camera already powered, `MagAOXApp` goes straight to `POWERON`
with `m_powerOnCounter == -1` (`MagAOXApp.hpp:1986`), so `powerOnWaitElapsed()` returns true right away and no wait is
applied. Errors seen on a software restart therefore come from the connect/first-acquisition path. The camera had
already finished booting, so a longer `powerOnWait` would not remove them.

Update (2026-11-16): the startup error is
`ERR SW FILE: orcaCtrl.hpp LINE: 1248 CODE: -2147483130 [Failed to initialize DCAM API.]`.
`-2147483130` is `0x80000206`, `DCAMERR_NOCAMERA`: `dcamapi_init()` reports no camera until the camera has enumerated.
`connect()` treated that as a failure: it logged `software_error`, set `ERROR`, and retried every loop for ~3 s.
The `deviceCount == 0` branch, which handles the same condition with one `NODEVICE` notice, was never reached.

A second false error happens on every connect. In the same pass that reaches `CONNECTED`, `getAcquisitionState()`
finds capture not yet running (the framegrabber thread starts it later). It then logs
`"acquisition stopped. restarting"` at `LOG_ERROR`, and logs it again on each `READY` pass until capture starts.

## Requirements
- R1: The startup errors stop. The cause was not the wait length (see above), so the power-on wait is not changed.
- R2: Changes are limited to connecting after power-on/startup. Steady-state error reporting does not change.
- R3: Real failures are still logged: other `dcamapi_init` errors, and capture stopping while `OPERATING`.
- R4: Unit tests check that power-on and restart reach `READY`/`OPERATING` with no ERR-or-worse log entries.

## Questions and Points of Clarification
All resolved; see User Answers above and Decisions below.
1. Which log lines fire at power-on, and at what level?
2. What does camscc's config set for `power.powerOnWait`?
3. Is suppressing the first "acquisition stopped. restarting" after connect within scope?
4. Is a fake DCAM test layer like pvcam's acceptable?
5. May the leftover `std::cerr << __LINE__` debug prints be removed?

## Decisions
- Q1/Q2: The logs and the error code identified the cause. `power.powerOnWait = 5` stays as configured. No
  orcaCtrl-specific `m_powerOnWait` default is added, because a longer wait would not remove errors that also happen on
  a software restart, when no wait is applied.
- Q3: In scope. "acquisition stopped" is only logged if the app was `OPERATING`, i.e. capture had been running.
- Q4: A fake DCAM layer is used: stub headers in `tests/dcam/`, with the library functions defined in the test.
- Q5: The `std::cerr << __LINE__` debug prints in `appLogic()`/`connect()` and the "// debugging" camera-ID print in
  `connect()` are removed.

## Tests
`apps/orcaCtrl/tests/orcaCtrl_test.cpp`, in `orcaCtrl_unit_test`. It uses the real `orcaCtrl` with `outletAppTest.hpp`
log capture, a threadless framegrabber placeholder, and a simulated main loop, following `pvcamCtrl_harness.hpp`.
- **Power-on while the camera enumerates:** `powerOnWait = 1`, and `dcamapi_init` returns `DCAMERR_NOCAMERA` 3 times.
  No DCAM init happens during the wait. The app is then `NODEVICE` with exactly one "no Hamamatsu available." notice,
  then `READY`, then `OPERATING` once capture is busy. There are no ERR-or-worse logs and no "acquisition stopped" log.
- **Software restart with the camera on:** connects on the first pass and stays `READY` with no ERR-or-worse logs.
- **Real errors still reported:** capture stopping while `OPERATING` logs "acquisition stopped. restarting" and sets
  `m_reconfig`. A `dcamapi_init` error other than `DCAMERR_NOCAMERA` still logs "Failed to initialize DCAM API." and
  sets `ERROR`.

Build/run: `cd tests && make -f Makefile.one t=../apps/orcaCtrl/tests/orcaCtrl_test && ../apps/orcaCtrl/tests/orcaCtrl_test`.
The agent workstation has no MagAO-X build dependencies (mxlib, gengithead.sh, ImageStreamIO), so the tests were not
compiled there; only the DCAM stub headers were syntax-checked. They must be built and run on a machine with the
dependencies.

## Implementation
1. `connect()`: `DCAMERR_NOCAMERA` from `dcamapi_init()` sets the device count to 0. This goes through the existing
   no-device path: `dcamapi_uninit()`, `NODEVICE`, and the notice logged once per state. Other failures are unchanged.
2. `getAcquisitionState()`: "acquisition stopped. restarting" is logged only if the state was `OPERATING` before the
   check. `m_reconfig` is still set as before.
3. The debug prints are removed (Q5).
4. Commits on `joshua-liberman/hamsci-app-cpp`: functional change + this plan, tests, docs, then a formatting-only
   commit. The formatting commit is clang-format 23 on `orcaCtrl.hpp`, mostly braces inserted by `InsertBraces`.

## Follow-up and Edge Cases
- On-instrument check (user): restart camscc and power-cycle the camera, then confirm there are no ERR lines between
  power-on and `OPERATING`.
- A camera that never enumerates stays `NODEVICE` with one notice and no error. This matches the existing
  zero-device behavior.
- "acquisition stopped. restarting" is still logged when a deliberate reconfigure stops capture while `OPERATING`
  (ROI, fan, or readout-speed change). That is outside this scope; it could be suppressed while `m_reconfig` is pending.
- At startup `m_reconfig` is already set when the framegrabber first configures, so the framegrabber configures twice.
  This is existing behavior, left unchanged.
- `camera.liquidCooling` was changed (commit d90709a5) to `isRequired = true` in `config.add()`, while the help text
  still says "Default is false". Check whether making it required was intended.
- The `power.powerOnWait` help text in `MagAOXApp.hpp` says "Default is 0 sec", but `m_powerOnWait` defaults to 55. It
  affects every app.
- `whilePowerOff()` calls `stdCamera::onPowerOff()` instead of `whilePowerOff()` (existing `\todo`).
- The pvcam and orca tests each define a threadless test app base. These could be shared in `tests/`.
