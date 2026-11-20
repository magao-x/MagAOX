# Task Description
<!-- Prompt given by the user in the agent session on 2026-11-18, copied verbatim. -->

The 1 kHz offload rate is accidental - it's a growing pain from having added dmsynco capabilities. we have a GUI (MagAOX/gui/widgets/offloadCtrl) that has an input that says "No. Avg [frames]", which has a number in it (usually 8 by default). This writes to the INDI property dmtweeter-avg, which tells the offloader how many tweeter frames to average before sending the signal to offload to the woofer. When dmsynco is off, this 8 frames corresponds to 250 Hz. However, when we toggle dmynco on, the offloading signal gets sent much faster because the tweeter is being written to at a much faster rate. We don't want this to happen - we want the offloading rate to stay consistent no matter the state of dmsynco. We need to change the GUI to take, as an input, an offloading frequency in Hz. When the user inputs an offloading framerate, the dmtweeter-avg property should be updated such that t2wOffloader.fps will equal the frequency the user has input. This needs to be true for when dmsynco is on or off. The state of dmsynco might change while the offloading loop is closed.

# Instructions to Agent

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan
<!-- This section will be filled out by the agent -->

*Status (2026-11-19): implemented, installed and checked on the bench (see Implementation Log).*

## Task Summary
<!-- The agent should summarize the task as they understand it -->

The offload rate is set indirectly, as a frame count.
- **How it works now:** `dmtweeter-avg` (a `shmimIntegrator`) averages `nAverage` frames of `dm01disp_delta`, and t2wOffloader offloads once per averaged frame.
- **What goes wrong with synco:** the DM controller publishes `dm01disp_delta` once per DM command, which is 2 kHz with dmSyncopation off but 8 kHz with it on at 4 steps. So the same `nAverage` = 8 gives 250 Hz or 1 kHz offloading depending on synco. On the bench we saw `aol0_tweeterOffload` at 1 kHz while `t2wOffloader.fps` said 250.

The goal:
- the operator enters an **offload rate in Hz** in the offloadCtrl GUI;
- `dmtweeter-avg.nAverage` is set so that the real offload rate, and `t2wOffloader.fps`, equal that rate, whether synco is on or off;
- `nAverage` is kept correct when synco changes state while the offload loop is closed.

### Findings
- **GUI** (`gui/widgets/offloadCtrl/offloadCtrl.hpp:87`): `ui.avgInt` is a `statusEntry` bound directly to `dmtweeter-avg.nAverage` (INT). Its label "No. Avg [frames]" is in `offloadCtrl.ui`.
- **`t2wOffloader.fps` is wrong whenever synco is on** (`t2wOffloader.hpp`, `updateFPS()`).
  - It is calculated as `m_fps / m_navg`, where `m_fps` is `camwfs.fps.current` (2 kHz), not the actual `dm01disp_delta` rate.
  - It reported 250 Hz while the real rate was 1 kHz.
- **dmSyncopation already publishes its rate.** It has a read-only `timing` property with `mainHz`, `plannedHz`, `deliveredHz` and `active` (`dmSyncopation.hpp:4707-4715`), and `settings.steps`. With synco active, the DM command rate is `plannedHz` (main × steps). With it inactive, it is the camera rate.
- **shmimIntegrator can already keep a constant averaging time.** With `integrator.avgTime > 0` it sets `nAverage = avgTime × fps` and reconfigures whenever its `fpsSource.fps.current` changes (`shmimIntegrator.hpp:1047-1060, 1315-1335`). But no device currently publishes the true `dm01disp_delta` rate as `fps.current`, so this can't be used as-is.

## Key Assumptions
<!-- The agent should list any assumptions they have made -->
1. **DM command rate.** The rate feeding `dmtweeter-avg` is `dmtweeter-synco.timing.plannedHz` when `timing.active` = 1, and otherwise `camwfs.fps.current`. Both are already published over INDI.
2. **Achievable rates.** `nAverage` must be a whole number, so the achieved rate is `dmRate / round(dmRate / requested)`. It equals the request exactly when the request divides the DM rate (e.g. 250 Hz: 8 at 2 kHz, 32 at 8 kHz). Otherwise it is the nearest achievable rate, and `t2wOffloader.fps` shows the true achieved value.
3. **Transition lag.** When synco turns on or off, offloads run at the old `nAverage` until the INDI update arrives and the integrator reconfigures, about one INDI round trip.
4. **Gain per offload.** t2wOffloader's gain and leak apply per offload. Keeping the rate constant keeps their meaning constant, which is the point of the change.

## Requirements
<!-- The agent should list the requirements to which they are planning -->
- **R1.** The offloadCtrl GUI takes an offload rate in Hz instead of "No. Avg [frames]". Per A1, the GUI change is limited to the input box:
  - its title becomes "Offload Rate (Hz)";
  - it is rebound from `dmtweeter-avg.nAverage` to `t2wOffloader.offloadRate`, which the new title needs;
  - no achieved-rate display is added; `t2wOffloader.fps` remains available over INDI.
- **R2.** `dmtweeter-avg.nAverage` is set so that `t2wOffloader.fps` equals the requested rate (or the nearest achievable rate), with synco on or off.
- **R3.** When synco turns on or off, or its step count changes, `nAverage` is updated automatically, including while the offload loop is closed and with no GUI open.
- **R4.** `t2wOffloader.fps` reports the true offload rate, including while synco is on.
- **R5.** The existing offload behaviour is otherwise unchanged. A rate of 0 means "not managed": `nAverage` is then set by hand, as today.

## Questions and Points of Clarification
<!-- The agent should list any open issues requiring user clarification -->
- **Q1. Where should the rate-keeping logic live?** (answered, see A1)
  - **(a) t2wOffloader (recommended).**
    - t2wOffloader gets a new `offloadRate` INDI property (Hz).
    - It already watches `camwfs.fps` and `dmtweeter-avg.nAverage`. It would also watch `dmtweeter-synco.timing`.
    - It recomputes the required `nAverage` and sends it to `dmtweeter-avg` whenever the request, the camera rate or synco's state changes.
    - It also corrects its own `fps` calculation (R4).
    - The GUI field is bound to `t2wOffloader.offloadRate`.
    - This works with no GUI open, and there is one owner of `nAverage`.
  - **(b) GUI only.**
    - offloadCtrl does the same calculation and writes `nAverage`.
    - It is simpler, but synco changes are tracked only while a GUI is open, and two open GUIs could fight. That fails R3.
  - **(c) shmimIntegrator time mode.**
    - Set `dmtweeter-avg`'s `fpsSource` to a device that publishes the true DM rate as `fps.current`, and have the GUI write `avgTime = 1/Hz`.
    - This reuses existing code, but needs dmSyncopation to publish a new `fps` property (a change to another team's app). It also stops tracking if synco is not running at all.
- **Q2.** Does (a) mean changing t2wOffloader again? Part 1 was additions only. This would add a property and a synco subscription, and correct the reported `fps` when synco is on, which is a deliberate behaviour fix. Is that acceptable?
- **Q3. Rates that don't divide the DM rate.** Round to the nearest `nAverage` and report the achieved rate (recommended), or only allow rates that divide exactly?
- **Q4. Default rate.** Should the default `offloadRate` come from config, e.g. `offload.rate=250`? Or start at 0 (not managed) until the operator sets it?

- A1: The rate-keeping logic should live in t2wOffloader, and the sub-points of that plan are acceptable. This being the case, the only thing that should change in the GUI should be the title of the input box: No. Avg [frames] --> Offload Rate (Hz)
- A2: those specific changes to w2tOffloader are fine.
- A3: round and report the achieved rate.
- A4: default should come from config and should be 250 us.
  - Agent note: implemented as `offload.rate=250` **Hz**, matching today's 8 frames at 2 kHz. Please correct this if 250 µs (4 kHz) was really meant.

## Tests
<!-- The agent should list and describe the test it plans to implement.  It should be specific about the purpose and goal of the test. -->
1. **Unit tests** (t2wOffloader test file), for a pure helper that takes the requested rate, camera rate, synco active flag and synco planned rate, and returns `nAverage` and the achieved rate:
   - 250 Hz → 8 with synco off and 32 with synco on (2 kHz × 4).
   - Rounding for rates that don't divide the DM rate.
   - Clamping to ≥1.
   - Rate 0 means unmanaged.
   - Missing or invalid synco or camera data falls back safely.
   - `fps` reports DM rate / `nAverage`.
2. **GUI:** build `offloadCtrl`, check that the box is titled "Offload Rate (Hz)", and check that entering a value writes `t2wOffloader.offloadRate`.
3. **Bench (user):**
   - Set 250 Hz with synco off: `nAverage` = 8 and `aol0_tweeterOffload` updates at 250 Hz.
   - Enable synco with the offload loop closed: `nAverage` changes to 32, and the stream stays at 250 Hz.
   - Disable synco again: `nAverage` goes back to 8.
   - `t2wOffloader.fps` should read 250 throughout.

## Implementation Plan
<!-- Here the agent documents its plan for the implementation -->
Per A1 (Q1 = a):
- **`apps/t2wOffloader/t2wOffloader.hpp`:**
  - New config `integrator.syncoSource` (default `dmtweeter-synco`) and `offload.rate` (Hz, default 250, per A4).
  - New INDI number `offloadRate`, plus a SET subscription to `<syncoSource>.timing` that reads `active` and `plannedHz`.
  - A pure static helper that computes `nAverage` and the achieved rate, to be unit tested.
  - On any input change, if the rate is managed and the required `nAverage` differs from the current one, send `dmtweeter-avg.nAverage.target` with the existing INDI client send.
  - `updateFPS()` uses the DM command rate (synco `plannedHz` when active) instead of the camera rate.
- **`gui/widgets/offloadCtrl/`:**
  - Rebind `ui.avgInt` from `dmtweeter-avg.nAverage` (INT) to `t2wOffloader.offloadRate` (FLOAT), a one-line change at `offloadCtrl.hpp:87`.
  - Retitle `label_avgInt` in `offloadCtrl.ui` from "No. Avg [frames]" to "Offload Rate (Hz)".
  - Nothing else in the GUI changes (A1).
- **Tests:** add the helper test case to `apps/t2wOffloader/tests/t2wOffloader_test.cpp`.
- **Scope:** files outside `apps/t2wOffloader` and `gui/widgets/offloadCtrl` are not touched. The instance configs on the RTC (`t2wOffloader.conf`, possibly `offload.rate`) are the user's to update.

### Implementation Log

**2026-11-19: first implementation (not yet compiled).**

**`apps/t2wOffloader/t2wOffloader.hpp`.**
- **New config:**
  - `integrator.syncoSource` (default `dmtweeter-synco`). Later removed; the name is fixed in code (see below).
  - `offload.rate` (Hz, default 250; 0 leaves `nAverage` unmanaged). Later removed; see the "no config setting" entry below.
- **New INDI properties:**
  - the number `offloadRate` (0 to 10000 Hz);
  - a SET subscription to `<syncoSource>.timing`;
  - an outbound `<navgSource>.nAverage` property with a `target` element, used to send the request. This follows kTracker's `m_indiP_kpos`: the property watched with `REG_INDI_SETPROP` has no type or elements, so it cannot be used to send.
- **DM command rate.** `dmRate()` is `timing.active × timing.mainHz` while synco is active (`active` is the number of steps actually running, and 0 when synco is not primed). Otherwise it is the camera fps.
  - `plannedHz` is not used, because it reports requested steps × main rate even while synco is inactive.
- **`updateFPS()`.**
  - It now reports `fps = dmRate() / navg`, so `t2wOffloader.fps` is correct with synco on (R4).
  - When the rate is managed, it sends `nAverage.target = navgForRate(dmRate(), offloadRate)` if that differs from the current value and from the last request.
  - It runs on any change to the camera fps, `nAverage`, synco `timing` or `offloadRate`.
- **Re-sending.** `m_navgRequested` stops the request being re-sent before the averager reports back. It is cleared when `nAverage` changes or `offloadRate` is set, so a hand edit of `nAverage` is overridden on the next update.
- **`navgForRate()`** is a static helper: it rounds to the nearest whole number, is at least 1, and returns 0 if the rate is unmanaged or the DM rate is unknown.

**`gui/widgets/offloadCtrl/`** (as answered in A1):
- `ui.avgInt` is rebound to `t2wOffloader.offloadRate` (FLOAT).
- The label reads "Offload Rate (Hz)".
- The box is still enabled or disabled by `dmtweeter-avg`'s state, as before.

**`apps/t2wOffloader/tests/t2wOffloader_test.cpp`.** Two new test cases:
- `navgForRate`: 250 Hz gives 8 at 2 kHz and 32 at 8 kHz; rounding; minimum of 1; unmanaged and invalid inputs.
- `dmRate()`: synco inactive, active at 4 steps, and unknown main rate.

**2026-11-19: documentation.** Doxygen pass over `gui/widgets/offloadCtrl/offloadCtrl.hpp` (AGENTS.md §16): added a `\file`/`\brief` block, a class brief, member docs, and briefs on every declaration. There are no functional changes.

**2026-11-19: no config setting for the rate.** The user asked that switching between this branch and `dev` should need no config changes. The `offload.rate` setting was therefore removed: the 250 Hz default is now built into the INDI `offloadRate` property, and the rate can only be changed over INDI (0 = unmanaged). This supersedes the A4 implementation note above. The `integrator.syncoSource` setting was also removed: the dmSyncopation device name is fixed in code as `dmtweeter-synco`. This feature now needs no config changes at all.

**2026-11-19: NaN check under fast-math.** The first RTC unit-test run failed `navgForRate( 2000, NaN ) == 0` (it returned 1). MagAOX builds with fast-math flags, which let the compiler remove `std::isfinite` and NaN-sensitive comparisons. The fix uses mxlib's bitwise `mx::math::isFinite`, as the dmSyncopation producer helper and counterOffload already do, both in `navgForRate` and in the `offloadRate` INDI callback.

**2026-11-19: build and bench results (run by the user).**
- **Unit tests.** The first unit-test rerun after the NaN fix still failed, because `tests/Makefile.one` doesn't track app header dependencies, so the test object wasn't rebuilt. After `make -f Makefile.one t=... clean` and a rebuild, all tests passed.
- **Formatting.** `clang-format` on the touched files also reformatted `gui/widgets/offloadCtrl/offloadCtrl.hpp`.
- **First synco-on check.** Turning synco on did not change `dmtweeter-avg.nAverage` at first. The user resolved it without code changes; the cause was not recorded. Revisit if it recurs.
- **GUI build.** `offloadCtrlGUI` was built and installed from a clone on exao1. That clone first needed the INDI client libraries: `make -C INDI/liblilxml` and `make -C INDI/libcommon`.
- **GUI check.** The input box reads "Offload Rate (Hz)".
- **Bench result.** With synco on, `dmtweeter-avg.nAverage` changed from 8 to 32, and `t2wOffloader.fps` stayed at 250 Hz (8000 Hz DM rate / 32), as intended.
- **Synco on/off switching.** The offload frame rate does not change when synco is switched on or off, and `nAverage` returns to 8 when synco is turned off. The offload rate is constant across synco state, as required (R2, R3).

**Checks for the first build:**
- `sendNewProperty` with a `uint32_t` value;
- `ipRecv["active"].get<float>()` on synco's `timing` property.

## Follow-up and Edge Cases
<!-- The agent should list any planned follow up and any edge cases that are not addressed -->
- **synco disabled vs. not running.** If `dmtweeter-synco` is disabled, `timing.active`=0 and the camera rate is used. If it is not running at all, its properties disappear and the camera rate is used too.
- **Warm-up.** During synco's warm-up, `active` may lag `enable`. Offloads briefly run at the camera-based `nAverage` until `active` goes to 1.
- **Hand edits.** Changing `dmtweeter-avg.nAverage` directly while the rate is managed is overridden at the next recompute. Setting `offloadRate` to 0 releases control.
- **Moving average.** Not considered: `dmtweeter-avg` is assumed to run as a block average (`nUpdate=0`), so its output rate is input rate / `nAverage`. If `nUpdate` is used, the output rate is input rate / `nUpdate` instead.
