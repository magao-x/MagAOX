# Task Description

## Problem Statement
In the current version of t2wOffloader, the tweeter has no knowledge of the offloading command being sent to the woofer. This results in a double-correction of the low-order modes that are being offloaded and results in increased rms wavefront error whenever an offload command is sent. We want to stop this from happening by taking advantage of the tweeter's speed. While the woofer offloads, the tweeter will make the exact opposite shape at the same time, thus eliminating the perturbation. 

## Discussion
Read through the t2wOffloader app. You will see this is the way the offloading currently works: every N frames, the average shape of a certain number M of low-order tweeter modes (in practice, usually 8-20) is projected onto the woofer basis and applied. The tweeter control algorithm does not explicitly take this into effect; instead, it sees the perturbation on the wavefront sensor and corrects it accordingly. This causes a spike in the RMS wavefront error in the low-order modes, which affects system performance - especially coronagraphs. The current t2wOffloader maximum offload speed is 1 kHz, which is 2 frames at the standard 2 kHz camwfs operating speed.

The woofer moves much slower than the tweeter. The disparity between the two has been explored using the app dmTemporalResponse (located on branch ktwitchell/dmTemporalResponse). The results of this exploration, while they are not complete or comprehensive, demonstrate that the tweeter moves with a 10-90 rise time of < 100 us (vendor documentation measured 50 us for a 100nm poke), while the woofer has a 10-90 rise time of around 300 us. You should look through these results in the directory /Users/Katie/Documents/MagAO-X/dm_response_data. We will exploit this to create a "counter-offload" command, where tweeter commands are written to step the tweeter shape through the opposite of the woofer shape at the same time. 

The execution will be in two distinct parts:
    1. addition of code into the t2wOffloader that calculates and publishes the offload command as back-projected onto the tweeter; and 
    2. a new C++ app entitled "counterOffload" that takes care of the writing of the tweeter commands.
For now, we will focus entirely on the first part. When it is time for the second part, a new prompt will be supplied.

Most of the framework for part 1 already exists within the t2wOffloader app, so we do *not* want to overcomplicate things. The solution should be as simple as possible while still fulfilling requirements. To understand how I want this solution implemented, read the dummy file in the path '/Users/Katie/Documents/MagAO-X/simulations/agent_workshop/t2wOffloader.hpp'. Check the diff between this file and the .hpp file in the current version of the t2wOffloader app in MagAOX/apps. There are a few lines that have been added in the dummy version that should explain the approach that should be taken. These lines are non-functional; they exist only to show you the general approach that should be taken. Again, follow the outline provided by this dummy file and implement the simplest solution. Once the tweeter values have been calculated, they should be published to a new shmim of our creating. 

## Requirements
1. t2wOffloader should be modifified to publish a new shmim with the offload pattern as it would appear on the tweeter
2. t2wOffloader should still function exactly as it already does; the innate function of the app should be completely unchanged. Only additions are welcome.
3. the method for doing this shall follow the general outline provided by the dummy .hpp file
4. the new shmim shall publish these values at a rate faster than the 8 kHz rate at which we desire to send the commands to the tweeter.

## Tests and Metrics

## Scope and Caveats
I will emphasize:  t2wOffloader should still function exactly as it already does; the innate function of the app should be completely unchanged. Only additions are welcome.

Code may only be added within the t2wOffloader app (anything in t2wOffloader sub-folder is fair game). Do not plan to or implement any code outside of that directory.

# Instructions to Agent

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan
<!-- This section will be filled out by the agent -->

*Status (2026-11-05): functional change, unit test, and documentation pass committed on `ktwitchell/counter-offload`. Not yet built or run; that and `clang-format` have to happen on the build machine.*

## Task Summary
<!-- The agent should summarize the task as they understand it -->

This is part 1 of 2. Add code to `t2wOffloader` that works out, on each modal offload, how much the offload command *changed* this step. It back-projects that change onto the tweeter mode basis and publishes the resulting tweeter-shaped map to a new shmim. A later `counterOffload` app (part 2, out of scope) will read that shmim and step the tweeter through the opposite shape while the woofer moves.

Findings from reading the code and data:

- **Modal path (`processImage`, `m_numModes > 0`).** Each update, for each mode `p < m_numModes && p < m_maxModes`:
  `m_modevalDM(p) = gain * m_modeDeltaAmps(p) + (1 - leak) * m_modevalDM(p)`.
  The woofer command is `sum_p m_modevalDM(p) * m_wModes(p)`, clipped to `±m_actLim`.
- **Dummy diff (`agent_workshop/t2wOffloader.hpp` vs `apps/t2wOffloader/t2wOffloader.hpp`).** The dummy adds four lines:
  1. `prevCommand = m_modevalDM(p,0)` just before the modal update in the loop.
  2. `m_deltavalDM(p,0) = m_modevalDM(p,0) - prevCommand` just after it.
  3. `m_tweeter = m_deltavalDM(0,0) * m_tModes.image(0)` next to the woofer mode sum.
  4. `m_tweeter += m_deltavalDM(p,0) * m_tModes.image(p)` inside the woofer mode-sum loop.

  So the approach is: keep a per-mode delta of the modal command, then build the tweeter shape of that delta from the tweeter modes in the same way the woofer shape is built from the woofer modes.
- **Names in the dummy that don't exist in the real app.** `m_tModes` should be `m_tModesOrtho`: these are the masked, unit-normalized tweeter modes that `m_modeDeltaAmps` is projected onto, so `delta * m_tModesOrtho` is in tweeter (dmtweeter-avg) command units. `m_deltavalDM` and `prevCommand` are new.
- **`m_tweeter` already exists.** It is declared, resized in `allocate()` to the input shmim size (the tweeter size), and never used anywhere. The dummy reuses it as the work buffer, and I'll do the same.
- **Existing pattern for publishing a shmim.** `dmRecon` uses the same `mx::improc::milkImage` approach as `t2wOffloader` itself: `create(name, w, h)` in `allocate()`, then `setWrite()`, write elements, `post()`. I'll use that pattern; no new utilities are needed.
- **DM response data (`dm_response_data/figures/dmresp_metrics_woofer_vs_tweeter_5kHz*.csv`).**
  - Deconvolved 10-90 rise times: woofer about 220-285 µs; tweeter actuator about 100-190 µs.
  - The woofer's t10 comes about 200 µs after the tweeter's. That difference includes command-path latency.
  - At 8 kHz, one tweeter step is 125 µs, so the woofer transition covers roughly 2-3 tweeter commands. This matters for part 2, not here.

## Key Assumptions
<!-- The agent should list any assumptions they have made -->

1. Only the **modal** path (`m_numModes > 0`) is in scope, because that is where the dummy adds lines. Zonal offloading (`m_numModes == 0`) is left unchanged and publishes nothing (see Q1).
2. The published shape is the change in the offload during this update, in tweeter command units: `sum_p Δmodeval_p * tModesOrtho_p`. It is **not** the cumulative offload, and it is **not** negated (see Q2).
3. ~~`prevCommand` is captured where the dummy puts it, after the `m_modevalDMf` override block.~~ Per Q4, the previous command is captured for all modes *before* the `m_modevalDMf` block. The delta therefore covers both the integrator update (gain + leak) and any `aol0_modevalDMf` resync, i.e. the full change the woofer makes.
4. The tweeter shape is the ideal back-projection through the tweeter modes. It does **not** account for woofer `m_actLim` clipping, the woofer mask applied to `m_wModes`, or response-matrix fit error. These are second-order for the 8-20 low-order modes normally offloaded.
5. **Interpretation of Requirement 4 ("faster than 8 kHz").** Offloads happen at ≤ 1 kHz, so the shmim updates once per offload. I read the requirement as: the counter shape must be computed and posted well within one 8 kHz period (125 µs) of the offload, so `counterOffload` can start in step with the woofer. The added work is at most `m_maxModes` (50) × 2500 px ≈ 125k multiply-adds plus a 10 kB copy. That should take a few µs, comfortably under budget (see Q5).
6. The existing behaviour stays exactly the same. All woofer computations, the order of writes to `m_dmStream`/`m_modevalDM`, locks, telemetry, and INDI are untouched. The only additions are the per-mode delta, the tweeter sum, and the new shmim post.
7. Building, testing, and `clang-format` cannot run on this Mac clone. They have to run on the build machine (as with previous work).

## Requirements
<!-- The agent should list the requirements to which they are planning -->

- **R1.** `t2wOffloader` creates a new shmim, the same size as the tweeter input shmim, holding the offload change projected onto the tweeter.
- **R2.** Existing behaviour is unchanged: additions only, and nothing is reordered, removed, or changed in the woofer/modeval path.
- **R3.** The implementation follows the dummy outline: a per-mode change `m_deltavalDM` (previous vs. new `m_modevalDM`), back-projected as `m_tweeter = sum_p m_deltavalDM(p) * m_tModesOrtho(p)`. Placement differs from the dummy as agreed in Q4/Q6 and the unit-test answer (see Implementation Plan).
- **R4.** The new shmim is posted on every modal offload, within ≪125 µs of the woofer command.
- **R5.** Edits are limited to `apps/t2wOffloader/` and follow AGENTS.md: Doxygen on new members, `m_` naming, no `\author` in touched files, full-file doc pass, `clang-format`.

## Questions and Points of Clarification
<!-- The agent should list any open issues requiring user clarification -->

- **Q1. Zonal mode.** Should zonal offloading (`m_numModes == 0`) also publish a counter shape? It could be done as `Δwoofer` back-projected with a pseudo-inverse of `m_twRespM`, but that is new math beyond the dummy. *Recommendation:* modal only for now. In zonal mode the new shmim is simply not updated.
    - Answer: no, implement only modal offloading for now and do not update the shmim when zonal offloading is active.
- **Q2. Sign convention.** Should the shmim hold the offload as it would appear on the tweeter (`+Δ`), with `counterOffload` negating it, or should it already be negated (`-Δ`)? *Recommendation:* `+Δ`, which matches the Requirement 1 wording "offload pattern as it would appear on the tweeter".
    - Answer: It should hold the offload as it would appear on the tweeter. The counter-offloading process will be entirely handled by the code written for Step 2, including the negation.
- **Q3. Shmim name.** Options:
  - Hard-coded, e.g. `aol{m_loopNumber}_t2wTweeterDelta`, following the `aol{}_CMmodesDM` precedent.
  - A new config option, e.g. `offload.counterShmim`, with that name as the default.
    - Answer: hard code the shmim name. It does not matter that m_loopNumber is 0 because that is the only loop the woofer works with anyways.

  *Recommendation:* the config option with that default.

  Side note: `m_loopNumber` is never loaded from config, and `aol0_modevalDM` / `aol0_modevalDMf` are hard-coded. That is an existing bug and I won't fix it here, per R2.
- **Q4. External `modevalDMf` updates.** When `aol0_modevalDMf` overwrites `m_modevalDM`, the woofer jumps too, but with the dummy's placement of `prevCommand` that jump is not in the delta. Is that intended? The alternative is to capture `prevCommand` for all modes *before* the `modevalDMf` block. That is still additions only.
    - Answer: capture the previous command before the `modevalDMf` block
- **Q5. Rate / latency.** Is Requirement 4 the latency budget described in Assumption 5, or do you want the new shmim posted at ≥ 8 kHz regardless of whether an offload happened (for example, a heartbeat re-post)? The input only arrives at the offload rate, so a heartbeat would need a separate thread. I'd rather not add one.
    - Answer: the assumption is correct.
- **Q6. Post order.** Should the new shmim be posted right **after** the woofer `sempost`, or just **before** the woofer `memcpy`?
  - After: woofer timing is completely untouched.
  - Before: `counterOffload` is a few µs ahead of the woofer.
    - Answer: place it after, we do not want to affect woofer timing.

  *Recommendation:* after, to protect R2. `counterOffload` can then align on cnt0/atime and add its own delay if the woofer driver latency needs it.
- **Q7. `zero()` and offload-toggle-on.** Both step the woofer to zero, which also disturbs the tweeter. Should they publish a counter shape too (−current shape)? *Recommendation:* not now; listed as a follow-up.
    - no, they should not. 

## Tests
<!-- The agent should list and describe the test it plans to implement.  It should be specific about the purpose and goal of the test. -->

`processImage()` depends on live shmims (`m_dmStream` opened in `allocate()`, `aol0_modevalDM*`), so an end-to-end unit test would need a big harness. That goes against "keep it simple". Proposed tests:

1. **Unit test (`tests/t2wOffloader_test.cpp`): tweeter back-projection.**
   - **Purpose:** check that for a known `m_deltavalDM` and synthetic `m_tModesOrtho` (e.g. three orthonormal 4×4 modes), the computed tweeter shape equals `sum_p Δ_p * mode_p`, and that modes `≥ m_numModes` contribute nothing.
   - **Requirement:** to make this testable without shmims, the 2-line tweeter sum would need to go into a small protected method, e.g. `calcTweeterDelta()`. That is a slight structural deviation from the dummy, which puts it inline.
   - **Question:** do you want this test? If you'd rather keep the code exactly inline as in the dummy, I'll skip it and rely on test 2.
     - Answer (2026-11-05): yes, use a helper and add the unit test. Implemented as `calcTweeterOffload()`. Its sections cover: the sum over modes, the `m_numModes` limit, the `m_maxModes` limit, zero modes, and that each call replaces rather than accumulates.
   - The test follows AGENTS.md §19/§20: per-`TEST_CASE` Doxygen, `T2WOFFLOADER_TEST_DOXYGEN_REF`, and removing `\author` from the test file header.
2. **Build-machine check (you run it).** Build and `clang-format` on the build machine. Then run `t2wOffloader` in simulation or on the bench with offloading on and `numModes` about 10. Confirm:
   - the new shmim's `cnt0` increments once per woofer `cnt0` increment;
   - its shape matches the frame-to-frame difference of `aol0_modevalDM` projected onto `/tmp/tModesOrtho.fits` (offline check in Python);
   - the woofer channel output is identical to the current build for the same input. The regression check here is a diff of a short sequence of `dm00dispXX` frames with the feature on vs. the `dev` build.
3. **Latency check (you run it).** Compare `atime` of the new shmim with `atime` of the woofer channel. Expect the difference to be ≪ 125 µs.

## Implementation Plan
<!-- Here the agent documents its plan for the implementation -->

**Branch:** `ktwitchell/counter-offload` (existing; follows §14). No push.

**Decisions from the answered questions:**
- **Q1, Q7:** modal offloading only. Zonal mode, `zero()`, and offload toggle-on publish nothing.
- **Q2:** publish `+Δ`, the offload as it appears on the tweeter. `counterOffload` (part 2) does the negation.
- **Q3:** the shmim name is hard-coded as `aol{m_loopNumber}_tweeterOffload`, which is `aol0_tweeterOffload` in practice.
- **Q4:** the previous command is saved for all modes before the `modevalDMf` block.
- **Q5:** Requirement 4 means "posted well inside 125 µs of each offload".
- **Q6:** post after the woofer `sempost`.
- **Unit test:** use a helper method plus a Catch2 test.

**As implemented in `apps/t2wOffloader/t2wOffloader.hpp`** (additions only):

1. **Members.**
   - Documented the previously unused `m_tweeter` as the back-projected offload change.
   - Added `m_deltavalDM` (eigenImage, modes × 1). It holds the previous `m_modevalDM` values, then the change.
   - Added `m_tweeterOffload` (milkImage): the new stream.
2. **`calcTweeterOffload()`** (public, inline, defined after `prepareModes()`). It sets `m_tweeter` to zero, then adds `m_deltavalDM(p) * m_tModesOrtho.image(p)` for `p < m_numModes && p < m_maxModes && p < m_deltavalDM.rows()`. This is the dummy's two-line sum moved into a function so it can be tested.
3. **`allocate()`.** After the existing `modevalDM`/`modevalDMf` setup, sizes and zeros `m_deltavalDM` to `m_tModesOrtho.planes()`. Creates `aol{m_loopNumber}_tweeterOffload` at the input (tweeter) shmim size.
4. **`processImage()` modal branch.**
   - Just before the `modevalDMf` block: copy every `m_modevalDM(p)` into `m_deltavalDM(p)`. That is at most 50 floats, about tens of ns.
   - After the woofer `ImageStreamIO_sempost`, while still holding `m_shmimMutex` (so `zero()` cannot run in between):
     - set `m_deltavalDM(p) = m_modevalDM(p) - m_deltavalDM(p)` for all modes;
     - call `calcTweeterOffload()`;
     - `m_tweeterOffload.setWrite(true)`, then `m_tweeterOffload() = m_tweeter`, then `post()`.
   - **Differences from the dummy:**
     - The delta and the tweeter sum are computed *after* the woofer is written rather than inside the existing loops. So none of the existing loops are edited, and the woofer timing gains only the snapshot copy.
     - The previous command is a member snapshot instead of a `prevCommand` local (Q4).
5. **Unit test, `tests/t2wOffloader_test.cpp`.**
   - Added a `MagAOX::app::t2wOffloader_test` harness (the class the app already befriends), hidden from Doxygen with `\cond`. It has three orthonormal 4×4 modes, one of them spread over four pixels.
   - Added a `TEST_CASE` for `calcTweeterOffload()` with a Doxygen block and `T2WOFFLOADER_TEST_DOXYGEN_REF`.
   - Removed `\author` from the file header.
   - The test was already listed in `tests/tests.list`.

**Remaining:**
- **Build and test:** build, run the unit test, then the bench checks (Tests 2 and 3), all on the build machine.
- **To confirm while building:** that `m_tweeterOffload() = m_tweeter` compiles against the mxlib `milkImage` API, which isn't available on this Mac.
- **Documentation pass (§16):** done as a separate commit. It added `///` docs to all undocumented members and declarations in `t2wOffloader.hpp`, and switched the `processImage()` parameter docs to the inline `/**< */` style. `t2wOffloader.cpp` already had `\file`/`\brief` and no `\author`.
- **`clang-format`:** a separate final commit, run on the build machine.

**Commits (§22/§23)**, each a single line plus `Co-Authored-By`:
1. functional change + unit test + this plan;
2. documentation pass;
3. `clang-format`.

## Follow-up and Edge Cases
<!-- The agent should list any planned follow up and any edge cases that are not addressed -->

- **Part 2:** the `counterOffload` app reads `aol0_tweeterOffload`, negates it, and writes it to a tweeter channel in time steps at 8 kHz. Timing should be based on the ~200 µs woofer-vs-tweeter t10 offset and the ~250 µs woofer rise. It can align on the shmim's `cnt0`/`atime`.
- **Not reflected in the published shape (by design or agreement):**
  - woofer `m_actLim` clipping;
  - woofer mask and response-matrix fit error;
  - `zero()` and the woofer zeroing at offload toggle-on (Q7);
  - zonal mode (Q1).
- **Woofer write timeout.** When `processImage()` returns early on the "timed out with write==1" path, `m_modevalDM` has already been updated and posted, but the woofer and the new shmim are not written. The next offload's change is measured from the updated `m_modevalDM`, so it undercounts the woofer step by the skipped update. This is rare, and it's existing behaviour that is left alone.
- **`m_numModes` changed from INDI mid-update.** `m_numModes` is read by the woofer sum and again by `calcTweeterOffload()` without a lock. A change landing between the two could make one published shape cover a different set of modes from the woofer command. It lasts one offload only.
- **Lowering `numModes`.** Modes above the new `numModes` keep their values in `m_modevalDM` (existing behaviour). Their change is computed but not summed, because the sum is limited to `p < m_numModes`.
- **Existing issues noticed, not fixed here (R2):**
  - `m_loopNumber` is never loaded from config;
  - `m_typeSize`/`m_dataType` are used in `allocate()`/`memcpy` where `m_dmTypeSize`/`m_dmDataType` are probably meant;
  - the `"offload.loopNUmber"` typo in `setupConfig()`;
  - `zero()`, `prepareModes()`, and the telemetry functions are non-inline definitions in a header;
  - a configured `offload.numModes > offload.maxModes` is not clamped at startup (only the INDI path clamps it), so the existing zeroing loop could index past `m_modevalDM`.
