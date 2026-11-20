# Task Description
This prompt is to implement Part 2 of the task described in the counter-offload.md planning file. This file will contain the plans and tests related to the design and implementation of the C++ app that will actually send counter-offload commands to the DM without interacting with the main wavefront sensing and control loop.

## Problem Statement
We now have a back-projection of the woofer offload shape being broadcast every time an offload is triggered, thanks to the updates to t2wOffload implemented on RTC in Part 1. Now, we need to build an app that will send those commands to the tweeter in a way that properly counters the shape temporally. We will do this using the dmSyncopation app as our guide. You have already read through the agent planning documents and the app itself and come up with a general framework for this: counterOffload will be a producer on the same host as dmSynco that watches the aol0_tweeterOffload shmim and publishes deltas that temporally ramp up or decay to the offload shape. You should use the way that tweeterSpeck currently works as a guide for how to write a new producer that uses dmSynco.

We need to make sure that this runs fast enough to publish at 8 kHz so that we can properly sample the woofer's timing curve. The woofer temporal response shape has been analyzed extensively; you can find the ultimate results locally at '/Users/Katie/Documents/MagAO-X/dm_response_data/' - first, read the README.md to understand what the data mean, then look at the woofer timing summary results data in '/Users/Katie/Documents/MagAO-X/dm_response_data/figures/dmresp_metrics_poke_types_5kHz_2026-10-15.csv'. We don't want to emulate the overshoot and ringing (for now, we think that will be a negligible improvement and will be too difficult to fit), but we do want our "ramp up" or "ramp down" to emulate the smooth rise shape shown on these curves. The results of the experiment suggest the shape of the woofer command does not affect the timing by too much, so we will have one single ramp shape no matter the shape of the offload that is being countered.

## Discussion
<!-- Put more expository text here.  Fully describe the expected behavior and rationale. Include references to other parts of the software that serve as examples. Provide a link to an API/SDK manual.   -->

## Requirements
<!-- List the requirements that must be met.  A numbered list works best -->

## Tests and Metrics
<!-- List any useful metrics that should be targeted by tests -->

## Scope and Caveats
<!-- Add any needed caveats and scope restrictions -->

# Instructions to Agent
<!-- Specific instructions for the agent.  Below is our standard, but you can modify it as needed. -->

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan
<!-- This section will be filled out by the agent -->

*Status (2026-11-15): implemented on `ktwitchell/counter-offload` (see Implementation Log). Not yet built or tested; the user builds, tests and `clang-format`s on the RTC.*

## Task Summary
<!-- The agent should summarize the task as they understand it -->

Build a new RTC app, `counterOffload`, that is a same-host **dmSyncopation producer**, modelled on how tweeterSpeck (`apps/dmSpeckle`) uses `dev::dmSyncopationProducer`.

- **Input:** it watches `aol0_tweeterOffload`. Each update (≤1 kHz, published by t2wOffloader since Part 1) is one offload step Δ in tweeter command units, posted right after the woofer write.
- **Output:** for every dmSyncopation slot (up to 8 kHz, at 0/125/250/375 µs per 2 kHz main cycle), it publishes to `dm01disp08_aux` a contribution that:
  1. **ramps** to −Δ following the woofer's smooth rise, with no overshoot or ringing, using one ramp shape for every offload shape; then
  2. optionally **leaks** back to zero, under a configurable leak coefficient, so the tweeter main loop takes over without a spike and tweeter stroke is freed.

dmSyncopation copies the contribution into `dm01disp08` and triggers dmcomb. counterOffload never writes the main loop channels or triggers dmcomb itself.

### Findings

- **Woofer rise** (`dm_response_data/figures/dmresp_metrics_poke_types_5kHz_2026-10-15.csv`):
  - t10/t50/t90 are about 537–580 / 700–733 / 830–861 µs after the woofer command, for actuator, tilt and astig at amplitudes 0.1–0.3.
  - 10–90 rise time is 264–324 µs.
  - Shape dependence is small, consistent with the prompt.
  - After WFS-exposure deconvolution (`dmresp_response_woofer_vs_tweeter_5kHz_deconv.csv`): woofer t10/50/90 = 483/623/717 µs, tweeter = 266/314/391 µs.
  - A raised cosine from 400 to 825 µs fits the deconvolved woofer rise to 2.7% RMS.
- **Tweeter command ramp.** I fitted a raised-cosine *command* ramp so that (command ramp ∗ tweeter step response) ≈ woofer step response.
  - Result: start **120 µs**, end **450 µs** after the woofer command (duration 330 µs), with 4% RMS residual over 350–900 µs.
  - Both curves are measured from the DM-channel `cnt0` write. That is the same event synco's slot target represents, so `request.targetNs` is the right time at which to evaluate the ramp.
- **dmSyncopation producer contract** (`libMagAOX/app/dev/dmSyncopationProducer.hpp`, `apps/dmSyncopation/doc/production-providers.md`):
  - The app owns one `<dest>_aux` float32 image and its own thread.
  - Loop: `syncopationPoll()` then `syncopationPublish(pixels, request, source)`, one outstanding command at a time.
  - `syncopationPublish` sleeps until `cutoff − writeLeadUs` before copying (`:473-511`).
  - synco retains the last contribution until an explicit zero.
  - Production preset: `readyLeadUs=70` (cutoff = target − 70 µs), 4 steps, 8 kHz, destinations 07–10.
  - tweeterSpeck uses 09; 07 is synco's trigger channel.
- **Latency budget.**
  - If the app computes just before publishing (see the producer thread in the Implementation Plan), an offload must be received by about target − 100 µs to make that slot.
  - So the first slot lands 100–225 µs after the event, against a ramp that starts at 120 µs.
  - In the worst phase the first command already corresponds to about 24% of the rise; the rest of the ramp is tracked normally.
  - Offloads are clocked off the same loop, so their phase relative to the synco grid may be roughly fixed. Measure this on the bench.
- **Why the leak matters.** `dmtweeter-avg` averages `dm01disp_delta`, the combined tweeter shape, which includes the counter channel.
  - If the counter held −Δ forever, the main channel would keep +Δ, so tweeter stroke is never freed.
  - A slow leak lets the main loop remove Δ from its own channel. The loop holds the total tweeter shape constant while the counter leaks away, so t2wOffloader sees no change.

## Key Assumptions
<!-- The agent should list any assumptions they have made -->

1. The Δ in `aol0_tweeterOffload` has the same geometry and command units as `dm01disp08_aux` (both 50×50 tweeter). This is checked at startup and on every stream reconnect.
2. The default ramp is a raised cosine, start 120 µs, duration 330 µs after the woofer command, as fitted above. Start, duration and leak are configurable and can be set over INDI so they can be tuned on the bench.
3. **The leak works like an integrator leak** (the same convention as t2wOffloader's `leak`).
   - Completed ramps are summed into a counter accumulator `A`.
   - At every published slot, `A ← (1 − leak)·A`, with `leak` in [0, 1].
   - `leak = 0` holds the contribution indefinitely (no decay). `leak = 1` clears it at the first slot after the ramp ends.
   - The equivalent time constant is about `−Δt / ln(1 − leak)`, where Δt is the slot period. For example, at 8 kHz a leak of 0.0025 gives about 50 ms.
   - The default leak is **0** (off), per Q5. The leak is applied per published slot, so its effective time constant depends on synco's step count.
4. At most about one ramp is active at a time (offloads ≤1 kHz, ramp 330 µs). The code still sums any overlapping ramps.
5. The app starts disabled. A `gain` property (0–1) scales the output for operator qualification.
6. Work continues on the existing branch `ktwitchell/counter-offload`. No new branch without asking.
7. **Everything runs on the RTC.**
   - counterOffload runs on the RTC, the same host as dmSyncopation and t2wOffloader. The producer protocol has no cross-host transport or clock mapping, and `aol0_tweeterOffload` is local to the RTC.
   - The user builds, unit tests, `clang-format`s and bench-tests it on the RTC.
   - The agent never runs anything on the RTC or instrument hardware (AGENTS.md rule 0).

## Requirements
<!-- The agent should list the requirements to which they are planning -->

- **R1.** A new header-only app `apps/counterOffload/` (AGENTS.md §21) that is a `dev::dmSyncopationProducer` writing `dm01disp08_aux`. It uses the `DMSYNCOPATIONPRODUCER_*` and `SHMIMMONITOR_*` macros (§18).
- **R2.** On each `aol0_tweeterOffload` update while enabled, start a ramp of −Δ with the configured shape. When the ramp completes, add it to the accumulator, which is multiplied by (1 − leak) at every published slot.
- **R3.** Publish a correctly time-evaluated contribution for every synco slot (up to 8 kHz). The publish path computes only a few-µs image sum. Count missed and failed slots.
- **R4.** Never touch the main loop channels or trigger dmcomb. Fail closed on geometry mismatch or non-finite values.
- **R5.** INDI controls:
  - `enable` toggle;
  - `zero` request (clear the state and publish zeros);
  - `gain`;
  - ramp start, ramp duration, `leak` (0–1) and time offset;
  - read-only counters: events received, slots published, missed and failed slots, active ramps.
- **R6.** Disabling stops accepting new offloads.
  - With `leak > 0`, the app keeps publishing the leaking contribution until it is negligible, then publishes an explicit zero and goes idle.
  - With `leak = 0`, the contribution is held, as with dmSpeckle, until the operator sends `zero`.
- **R7.** Follow the AGENTS.md documentation and test rules.

## Questions and Points of Clarification
<!-- The agent should list any open issues requiring user clarification -->

- **Q1. What happens after the ramp?** Hold −Δ, or decay back to zero?
  - Answer: slow decay to zero.
  - Revised (2026-11-15): make it an *option* through a leak coefficient (usually 0–1) that sets how fast, if at all, the signal leaks back to zero. This is effectively a leak on the counter integrator. See Assumption 3.
- **Q2. Destination channel?**
  - Answer: `dm01disp08_aux`.
- **Q3. Edits outside `apps/counterOffload/`?** Adding the app to `apps_rtc` and `all_buildable_apps` in the top-level `Makefile`, and its test to `tests/tests.list`.
  - Answer: both allowed.
- **Q4. Time reference for the ramp.**
  - t2wOffloader posts `aol0_tweeterOffload` a few µs after writing the woofer channel, and the ramp is defined relative to that woofer command. Two options:
  - **(a) Wake time + offset (recommended).** Take `CLOCK_MONOTONIC` when counterOffload's shmim monitor thread wakes on the update, plus a configurable `timeOffsetUs` (default 0) that can be tuned on the bench.
    - Simple.
    - Adds about 10–30 µs of wake latency, with some jitter, to the reference.
  - **(b) Shmim write timestamp.** Use the shmim's own write timestamp (`CLOCK_ISIO`) and convert it to monotonic time with dmSyncopation's existing publication-clock helper (`libMagAOX/common/publicationClock.hpp`, used by synco for the same purpose).
    - More exact; free of wake jitter.
    - More code, and a clock-qualification failure path to handle.

  - Answer: We will do (a), the simpler version
- **Q5. Default leak, and how it is applied.** Two parts:
  - **Default `leak`.** The plan assumes 0 (off) until it is tuned on the bench.
    - Answer: the default should be 0
  - **Apply the leak per published slot (integrator style)?** That is the recommendation. The effective time constant then depends on synco's step count, and no leak happens during missed slots. The alternative is per unit time, i.e. `(1 − leak)` per nominal 125 µs, applied by elapsed target time; that is independent of misses and step count, but less integrator-like.
    - Answer: follow the recommendaation - only depend on synco's step count
- **Q6.** Which cpuset and priority should the producer thread use? The cpuset is set in the instance conf only, with no change to `rtSetup/RTC/make_cpusets`.
  - **Recommendation:** a spare in the RTC's L3 domain 3 (`spare26`–`spare30`). That domain is set aside for recorders and reserves, away from the loop's caches.
  - **Avoid CPUs 40–47** (L3 domain 5), which hold the tweeter monitor, synco, tweeterSpeck and the MVM. In November, moving the DM SDK worker onto CPU42 in that domain slowed the MVMs (`dm-syncopation-minimal-diagnostics.md`).
  - **Load:** counterOffload's load is small, a 50×50 image sum at up to 8 kHz. But a producer thread that is constantly polling at 8 kHz is the kind of extra activity that caused that interference.
  - **`spare31`** was used for post-worker tests; confirm it is free before using it.
  - **Priority:** moderate real-time, like tweeterSpeck's 20, well below the SDK worker's 55.
    - spare26 is fine, and emulating the priority of tweeterSpeck is a good starting point.

## Tests
<!-- The agent should list and describe the test it plans to implement.  It should be specific about the purpose and goal of the test. -->

1. **Unit tests** (`apps/counterOffload/tests/counterOffload_test.cpp`; Catch2, AGENTS.md §19/§20 grouping and Doxygen):
   - **Ramp function:** 0 before the start; 0.5 at the midpoint; 1 at and after the end; monotonic; finite for edge parameters.
   - **Single event:** at several target times relative to the event, the command equals −gain·Δ·ramp, then −gain·Δ·(1 − leak)^n after n published slots past the end of the ramp.
   - **Overlapping events:** the contributions add, and folding a finished ramp into the leaking accumulator is continuous at the hand-over.
   - **Zero request / disable:** zero clears the state and produces an all-zero command. Disabled mode ignores new events. With leak > 0 it keeps leaking, then publishes exactly zero. With leak = 0 it holds.
   - **Leak limits:** leak = 0 holds the value exactly; leak = 1 gives zero at the first slot after the ramp.
   - **Validation:** rejects a leak outside [0, 1]; rejects a channel without the `_aux` suffix, a geometry mismatch, and non-finite parameters.
   - **(Optional)** One private-stream producer case, reusing dmSpeckle's test fixture pattern if it fits cleanly.
2. **On the RTC (user):**
   - `make -C apps/counterOffload`;
   - build and run the test with `Makefile.one`;
   - `clang-format`.
3. **Bench, operator only:**
   - Start synco, then counterOffload with a small gain, then enable.
   - Confirm in synco's `production` property that mapping 1 (dm01disp08) is accepted every slot with no misses.
   - Confirm `dm01disp08` follows the expected ramp, and then the expected leak, after an offload.
   - Then compare the low-order residual spike at offloads with counterOffload on and off.
   - Tune the ramp start, duration, time offset and leak.

## Implementation Plan
<!-- Here the agent documents its plan for the implementation -->

Files (new unless noted):

- **`apps/counterOffload/counterOffload.hpp`.** Class declaration and out-of-class inline definitions.
  - `class counterOffload : public MagAOXApp<true>, public dev::shmimMonitor<counterOffload>, public dev::dmSyncopationProducer<counterOffload>`, with friend and alias declarations as in `dmSpeckle.hpp:55-63`.
  - **Config, under `[counterOffload]` unless noted:**
    - `channel` (default `dm01disp08_aux`);
    - `rampStartUs` 120, `rampDurationUs` 330, `leak` 0, `timeOffsetUs` 0;
    - `gain` 1, `computeLeadUs` 20;
    - `threadPrio` 20 and `cpuset` `spare26` in the example conf (per Q6);
    - `shmimMonitor.shmimName=aol0_tweeterOffload`;
    - `[syncopation] timingStream`, `writeLeadUs` (example 10).
  - **`allocate()` / `processImage()`** (shmimMonitor):
    - check geometry;
    - while enabled, copy Δ and t0 = monotonic receipt time + `timeOffsetUs` (Q4 option a) into a pending queue under a mutex.
  - **Pure math, to be unit tested:**
    - `rampFraction(tauUs)`, the raised cosine.
    - `buildCommand(targetNs)`: ingest pending events into the active-ramp list; fold finished ramps into accumulator `A`; apply `A ← (1 − leak)·A` once per published slot; then command = −gain·(A + Σ Δ_k·ramp(T − t0_k)), summed over the active ramps.
  - **Producer thread** (pattern of `dmSpeckle::syncopationThreadExec`, `dmSpeckle.hpp:591-643`):
    1. `syncopationPoll()`, with backoff on 0 and −1.
    2. On 1, sleep until `cutoffNs − writeLeadUs − computeLeadUs`.
    3. Lock, then `buildCommand(request.targetNs)`.
    4. `syncopationPublish()`.
    5. Update the counters.
  - **Idle when** disabled and the contribution is exactly zero.
  - **INDI** properties per R5.
  - **Shutdown:** `DMSYNCOPATIONPRODUCER_APP_SHUTDOWN` after the thread joins.
- **`apps/counterOffload/counterOffload.cpp`:** `main` only.
- **`apps/counterOffload/Makefile`:** dmSpeckle pattern, with `OTHER_HEADERS` including the producer helper.
- **`apps/counterOffload/dmtweeter_counteroffload.conf`:** example instance config, like `dmtweeter_vibration.conf`.
- **`apps/counterOffload/tests/counterOffload_test.cpp`**, plus the `tests/tests.list` entry.
- **Top-level `Makefile`** (edit): add `counterOffload` to `apps_rtc` and `all_buildable_apps`.
- **Reuse:**
  - `dev::dmSyncopationProducer`, `dev::shmimMonitor`;
  - `synco::Request` and `synco::monotonicNs()`;
  - `mx::math::isFinite`;
- **Commits** (§22/§23):
  1. functional app, tests and Makefile entries, with this plan updated;
  2. documentation;
  3. `clang-format`, run by the user on the RTC.

### Implementation Log

**2026-11-15: first implementation (not yet compiled).**

Files added or changed:
- `apps/counterOffload/counterOffload.hpp`: the class and its out-of-class inline definitions.
- `apps/counterOffload/counterOffload.cpp`: `main` only.
- `apps/counterOffload/Makefile`.
- `apps/counterOffload/dmtweeter_counteroffload.conf`: the example config.
- `apps/counterOffload/tests/counterOffload_test.cpp`.
- Top-level `Makefile`: added `counterOffload` to `apps_rtc` and `all_buildable_apps`.
- `tests/tests.list`: added the test.

Design details settled while writing the code:
- **Ramp pool.** Ramps live in a preallocated pool of 8 entries (`m_maxRamps`), so the monitor thread does not allocate memory per offload.
  - If the pool is full, the oldest ramp is folded into the accumulator straight away and counted in `overflows`.
  - The input monitor thread adds a ramp under `m_stateMutex`. The producer thread folds completed ramps when it builds a command.
- **When it publishes.** The app publishes only when something is changing:
  - a ramp is active;
  - the leak is above 0 and the accumulator is nonzero;
  - a final zero is pending after the leak reached zero;
  - a zero was requested.

  Otherwise it skips the slot, and dmSyncopation keeps the last contribution.
- **Idle.** When disabled with nothing to finish, the producer thread sleeps 1 ms between checks. Otherwise it polls the schedule every 10 µs, as dmSpeckle does.
- **When it computes.** For each slot the producer thread sleeps until `cutoff − writeLeadUs − computeLeadUs`. It then builds the command under the lock, releases the lock, and calls `syncopationPublish`.
- **Leak.** The leak is applied only after a successful publish. A leaking accumulator whose largest pixel falls below `m_zeroTol` (1e-7) is set to zero, and one explicit zero is then published.
- **Zero request.** A zero request clears all ramps and the accumulator and publishes zero. The flag is cleared only after a successful publish, as in dmSpeckle.
- **Startup state.** The app opens the auxiliary source in `appLogic` while `NOTCONNECTED`, then goes to `OPERATING`. The input monitor and the producer thread both run only in `OPERATING`.
- **INDI.**
  - `enable` toggle;
  - `zero` request;
  - numbers `gain`, `leak`, `rampStartUs`, `rampDurationUs`, `timeOffsetUs`, all set through one `setNumber()` helper;
  - read-only `stats`: `events`, `published`, `failed`, `overflows`, `active`.
- **Example config.** It uses `writeLeadUs=10`, `computeLeadUs=20`, `cpuset=spare26`, `threadPrio=20`.

**2026-11-15: documentation.** Added the operator guide `apps/counterOffload/doc/counterOffload.md`. It covers the purpose, the published command, running the app on the RTC, and the settings and `stats` reference.

Checks for the first RTC build:
- Constructing `pcf::IndiProperty` references inside the startup `numberT` table.
- `createStandardIndiNumber<double>` with `"%g"` formatting.
- `M_PI` from `<cmath>`.

**2026-11-19: bench.** counterOffload writes to the DM successfully through synco.

**2026-11-19: proposed change (not yet implemented): zero on disable.**
- Request: turning `enable` off should zero the DM stream automatically.
- Current behaviour (R6): disabling only stops accepting offloads. The contribution is kept (leak 0) or decays (leak > 0), and a zero is published only through the `zero` request.
- Change: in the `enable` NEW callback, when the toggle goes On→Off, also set `m_zeroRequested = true`.
  - The existing zero path clears the ramps and accumulator and publishes one all-zero command. `needPublish()` is already true while a zero is requested, so the producer thread does not go idle before publishing it.
  - Offloads that arrive after the disable are already ignored in `processImage()`.
  - If synco is not producing, the zero is published at the next slot after it resumes. That is the same as for the `zero` request today.
- Docs: update R6 and `doc/counterOffload.md` (disabling now zeros the output immediately; there is no decay on disable).
- Test: a unit test that disabling with a nonzero accumulator and active ramps gives an all-zero command at the next slot.

## Follow-up and Edge Cases
<!-- The agent should list any planned follow up and any edge cases that are not addressed -->

- **Missed slots:** if synco is disabled or the loop stops, no slots occur. The last contribution persists. With a per-slot leak, no leak is applied while no slots are being published; it resumes when synco does.
- **Late events:** an event that arrives after the compute time misses that slot. The next slot evaluates the ramp at the correct τ, so it starts late without replaying the missed part.
- **Ramp ripple:** the 125 µs zero-order hold produces small steps through the ramp. Bench tuning of `timeOffsetUs` (e.g. half a slot) can centre them.
- **Model limits:** overshoot and ringing of the woofer (about 3–10%) are not modelled, by design.
- **Lowering the latency further** would need a smaller synco `readyLeadUs`, which requires requalification, or delaying the woofer write in t2wOffloader. Both are out of scope.
- **Woofer actuator clipping (`actLim`)** in t2wOffloader is not reflected in Δ (Part 1 limitation).
- **Possible later additions:** telemetry and a handbook page.
