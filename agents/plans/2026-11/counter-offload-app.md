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
- **R6.** Disabling stops accepting new offloads and zeros the contribution at the next slot (changed 2026-11-19; it previously kept the contribution, or let it leak).
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

**2026-11-19: zero on disable (implemented).**
- Request: turning `enable` off should zero the DM stream automatically.
- Current behaviour (R6): disabling only stops accepting offloads. The contribution is kept (leak 0) or decays (leak > 0), and a zero is published only through the `zero` request.
- Change: in the `enable` NEW callback, when the toggle goes On→Off, also set `m_zeroRequested = true`.
  - The existing zero path clears the ramps and accumulator and publishes one all-zero command. `needPublish()` is already true while a zero is requested, so the producer thread does not go idle before publishing it.
  - Offloads that arrive after the disable are already ignored in `processImage()`.
  - If synco is not producing, the zero is published at the next slot after it resumes. That is the same as for the `zero` request today.
- Docs: update R6 and `doc/counterOffload.md` (disabling now zeros the output immediately; there is no decay on disable).
- Test: a unit test that disabling with a nonzero accumulator and active ramps gives an all-zero command at the next slot.
- Implementation: a new `setEnabled(on)` helper sets `m_enabled` and, on On→Off, `m_zeroRequested`; the `enable` callback calls it and logs "disabled, zero requested". The test section "disabling requests a zero, enabling does not" checks `setEnabled()` and that `needPublish()` stays true. The zero itself goes through the existing `clearState()` path, which "clearState removes ramps and the accumulator" already covers.

## Follow-up and Edge Cases
<!-- The agent should list any planned follow up and any edge cases that are not addressed -->

- **Missed slots:** if synco is disabled or the loop stops, no slots occur. The last contribution persists. With a per-slot leak, no leak is applied while no slots are being published; it resumes when synco does.
- **Late events:** an event that arrives after the compute time misses that slot. The next slot evaluates the ramp at the correct τ, so it starts late without replaying the missed part.
- **Ramp ripple:** the 125 µs zero-order hold produces small steps through the ramp. Bench tuning of `timeOffsetUs` (e.g. half a slot) can centre them.
- **Model limits:** overshoot and ringing of the woofer (about 3–10%) are not modelled, by design.
- **Lowering the latency further** would need a smaller synco `readyLeadUs`, which requires requalification, or delaying the woofer write in t2wOffloader. Both are out of scope.
- **Woofer actuator clipping (`actLim`)** in t2wOffloader is not reflected in Δ (Part 1 limitation).
- **Possible later additions:** telemetry and a handbook page.

## Calibrate Mode (implemented 2026-11-19, built and unit tested on the RTC; bench test pending)

### Request (user, 2026-11-19, condensed)
Tune the ramp so that the wavefront error spikes caused by offloading are mitigated. The figure of merit comes from the WFS closed-loop PSDs, published in real time in `aol1_clpsds`. The spike sits at (or very near) `t2wOffloader.offloadRate`, so integrate the PSD around that frequency and compare it across ramp parameters and with counter-offloading off. Plan a new `calibrate` mode of counterOffload that optimizes the ramp parameters to minimize this PSD spike.

### Findings
- **Producer:** `aol1_clpsds` comes from `apps/modalPSDs`. With `m_shmimBase` defaulting to `aol<loop>`, it publishes three streams:
  - `aol1_freq`: 1 × nFreq frequency axis, `n·df` with `df = 1/psdTime`. That is 1 Hz at the default `psdTime` of 1 s.
  - `aol1_raw_clpsds`: nFreq × nModes × depth circular buffer. One Hann-windowed PSD per window, published every `psdTime·(1 − overlap)` (0.5 s by default). The current plane is `cnt1`.
  - `aol1_clpsds`: nFreq × nModes, the average of the raw PSDs over `psdAvgTime` (10 s by default).
  - Each PSD is normalized so that Σ PSD·df equals the mode's variance over the window. The band integral is therefore directly a variance (WFS modal units²).
- **Response time.** The averaged stream still holds pre-change data for about `psdAvgTime + psdTime` (about 11 s) after a parameter change. A raw PSD whose `writetime` is later than change time + `psdTime` + margin covers only post-change data. So scoring from `aol1_raw_clpsds` takes about 1.5 s to settle plus N × 0.5 s per evaluation. Scoring from `aol1_clpsds` takes at least 11 s per point.
- **Spike frequency.** The real offload rate is `t2wOffloader.fps`, the read-only effective rate `dmRate/nAverage`. It can differ from the requested `offloadRate`: 300 Hz requested gives 296.3 Hz at 8 kHz. So the band should be centred on `t2wOffloader.fps`. An offload is a periodic step train, so its power can also show at harmonics 2f, 3f… below Nyquist (1 kHz at 2 kHz).
- **Parameter degeneracy.** `timeOffsetUs` and `rampStartUs` both shift the ramp in time and are interchangeable. `leak` acts at low frequency, not at the offload rate. That leaves three free parameters: **`rampStartUs`, `rampDurationUs`, `gain`**. `gain = 0` gives a clean "counter off" reference without toggling `enable`, which now zeros the output and drops state.
- **Quantization.** Commands are held for 125 µs slots, so the merit function versus `rampStartUs` will be step-like on that scale. Grid steps finer than about 25 µs are unlikely to resolve anything.
- **Repo patterns.**
  - A second input stream uses a tagged `dev::shmimMonitor<counterOffload, psdShmimT>`, as in `dmPokeCenter` / `dmPokeXCorr`.
  - Remote INDI values are read with a SET-registered property, as in t2wOffloader's `fpsSource` / `syncoSource`.

### Proposed Design
- **Mode and state.**
  - INDI `calibrate` switch (`start` / `abort`) and a read-only `calibration` property:
    - `state`, `stage`, `point`;
    - `fom`, `fomOff`, `ratio`;
    - best `rampStartUs`, `rampDurationUs`, `gain`.
  - It runs only while `enable` is on and the app is OPERATING. Otherwise it refuses with a log message.
- **Calibration thread.**
  - The calibration thread is separate and non-RT: low priority, on the same cpuset as the shmim monitor, never the producer's spare core.
  - It changes parameters only through the existing setters under `m_stateMutex`. The RT producer path is unchanged.
- **Figure of merit.**
  - `fom = Σ_{mode < numModes} Σ_{|f − f_off| ≤ halfWidth} PSD(f, mode)·df`, the fundamental only (A-C4).
  - `f_off` comes from `t2wOffloader.fps`.
  - `halfWidthHz` defaults to 10 Hz and can be set by config and INDI, to match the spike width seen on the bench (A-C4).
  - Modes: only the offloaded low-order modes `[0, numModes)` (A-C5). `numModes` is read from `t2wOffloader.numModes`, with an optional config override `modeMax`. This assumes PSD mode index k is offloaded mode k; see assumption CA1.
  - **Averaging (A-C1): a score is never taken from a single PSD.**
    - Each evaluation averages `nAverage` raw PSDs (default 6, minimum 2 enforced), all from windows that start after the parameter change, before integrating the band.
    - Consecutive raw windows overlap by 50%, so 6 windows span about 3.5 s of data.
    - Each CSV row also records the standard deviation of the per-window band integrals, as a noise estimate.
- **Baseline.**
  - `fomOff` is measured with `gain = 0` at the start, and again every K points (default 5) to track drift.
  - Each point is reported as `ratio = fom / fomOff(interpolated)`.
- **Optimizer: coordinate scans (recommended).**
  1. `rampStartUs` over a grid (default 0–400 µs, step 25), at the current duration and gain.
  2. `rampDurationUs` over a grid (default 100–600 µs, step 50), at the best start.
  3. `gain` over a grid (default 0.5–1.5, step 0.1), at the best start and duration. Gain may exceed 1 (A-C3).
  4. Repeat 1–3 once over ±2 steps around the best point at half the step size.
  5. Parabolic fit through the best point and its neighbours, for a sub-grid estimate.
  - Default run time: about 4.5 s per point (1.5 s settle + 6 × 0.5 s) for about 65 points including baselines, so about 5 minutes.
  - Coordinate scans were chosen (A-C6) because they are robust to noise and each stage gives a curve to inspect.
- **End state.**
  - On completion, apply the best parameters and log them at `LOG_INFO`, as suggested config values (A-C7). The app never writes its config file.
  - If the best ratio is not below 1 (no improvement over off), still apply them, but also log a warning.
  - Abort restores the parameters from before calibration.
- **Safety.**
  - Grid bounds are clamped to the INDI limits.
  - The `gain` INDI and config maximum is raised from 1 to 2 (A-C3). That applies to normal operation too, so a gain above 1 is an operator choice.
  - Abort and restore if:
    - any point's `fom` exceeds `abortRatio × fomOff` (default 3);
    - no new raw PSD arrives within a timeout (loop opened, or modalPSDs stopped);
    - `t2wOffloader.fps` changes during the run;
    - the PSD geometry changes.
- **Recording.**
  - Every point is logged as `text_log`: stage, parameters, fom, fomOff, ratio.
  - One CSV per run, one row per point (A-C8), written to `calibrate.csvDir`, default `/home/xsup/counterOffload/`, created if missing. The file name is `<instance>_calib_<UTC timestamp>.csv`.
  - Columns: time, stage, `rampStartUs`, `rampDurationUs`, `gain`, `f_off`, `numModes`, `halfWidthHz`, `nAverage`, fom, fom std, fomOff, ratio.
- **Config** (`[calibrate]` section): `psdStream` (`aol1_raw_clpsds`), `freqStream` (`aol1_freq`), `offloadDevice` (`t2wOffloader`), the band, mode, averaging and baseline settings above, grids, `abortRatio`, `timeoutS`, `csvDir`.

### Calibrate Key Assumptions
- **CA1 (resolved 2026-11-19).** The user notes that the modal values are in `aol1_modevalWFS` and that `aol1_clpsds` is a power spectrum as a function of temporal frequency.
  - From the code: modalPSDs reads a modal-amplitude stream, presumably `aol1_modevalWFS`, and sets `m_nModes = width × height` of that input (`modalPSDs.hpp:675`).
  - It publishes `aol1_clpsds` as nFreq × nModes (`:836-851`): one temporal PSD per input mode. Frequency runs along the first axis; `aol1_freq` gives the values.
  - So "only the offloaded low-order modes" (A-C5) means selecting PSD columns, that is modes of `aol1_modevalWFS`.
  - The open question is whether WFS mode k in `aol1_modevalWFS` is the same as t2wOffloader mode k (plane k of its tweeter modes `m_tModesOrtho`).
  - If the two bases differ, the default `modeMax = t2wOffloader.numModes` is only an approximation of "the offloaded modes". The override would then be set from the known mapping.
  - Bench check: the second dimension of `aol1_clpsds` should equal the number of elements in `aol1_modevalWFS`.
  - **Resolution:** the user believes the two mode sets differ slightly. Summing the first `t2wOffloader.numModes` WFS modes is acceptable. Keep the default `modeMax = t2wOffloader.numModes`, with the config override available.
- **CA2 (confirmed 2026-11-19).** See below.
- **CA2.** `t2wOffloader.fps` is reachable over INDI from the RTC indiserver, as `dmtweeter-synco` is for t2wOffloader. Confirmed by the user.
- **CA3.** `numModes` and `t2wOffloader.fps` are constant during a run. A change aborts the run.

### Calibrate Follow-ups
- **Harmonics** (A-C4): add an `nHarmonics` option that sums bands at k·f_off below Nyquist, for higher-precision tuning. It is a small change to the band integral.

### Questions and Points of Clarification
- **Q-C1. Raw or averaged PSDs?** You named `aol1_clpsds`. I recommend `aol1_raw_clpsds` for scoring: about 3 s per point instead of about 12 s, and it is published by the same app. `aol1_clpsds` would also work if the slower rate is acceptable, and could be an option.
- **Q-C2. Host.** Is modalPSDs for loop 1 running on the RTC, the same host as counterOffload? shmims are local, so if it runs on ICC, calibrate needs another approach. Please check with `ls /milk/shm/aol1_*psds*` on the RTC.
- **Q-C3. Parameters.**
  - Optimize `rampStartUs`, `rampDurationUs` and `gain`, leaving `timeOffsetUs` at 0 and `leak` fixed?
  - Should `gain` be allowed above 1 during calibration? Undershoot by the woofer or tweeter-response differences might want it.
- **Q-C4. Band.** Fundamental only, or include harmonics? Is a ±3 Hz half-width right, given your PSD settings?
- **Q-C5. Modes.** Sum all modes, or only the offloaded low-order modes, for example `[0, t2wOffloader.numModes)`? Fewer modes means less unrelated noise in the merit function.
- **Q-C6. Optimizer.** Coordinate scans (recommended) or Nelder–Mead?
- **Q-C7. End state.** Automatically apply the best parameters when they beat `acceptRatio`, or always restore the original parameters and only report the result?
- **Q-C8. Recording.** Is a per-run CSV plus `text_log` lines enough, or do you also want telemetry?

### Answers
- A-C1: using the raw PSDs is fine, but a single data point should never be used for scoring - there should definitely be an element of averaging, since there can be quite a bit of noise in those measurements. 
- A-C2: yes, it runs on RTC
- A-C3: yes, optimize those three while leaving the other two alone. yes, gain should be allowed to exceed 1.
- A-C4: Use the fundamental mode only for now, but log this as a potential easy upgrade we could use for higher-precision tuning. the width of the PSD integral should be wider than 3 Hz, default to 10 but keep this as a tunable parameter so we can match what the PSDs show us on the bench.
- A-C5: only the offloaded low-order modes.
- A-C6: Coordinate scans
- A-C7: apply the best parameters and record them in the log at the INFO level.
- A-C8: per-run CSV is fine. It can be saved in a directory in /home/xsup

### Tests
- **Unit tests:**
  - band-integral math on synthetic PSD/frequency arrays, covering bin edges, harmonics above Nyquist, mode range and `df` weighting;
  - raw-window selection by `writetime` relative to the change time;
  - scan logic against a synthetic quadratic objective with noise, which should converge near the true minimum;
  - parabolic refinement;
  - abort and restore paths;
  - accept or restore by ratio.
- **Bench (operator):**
  - Run calibrate at a small grid first, starting from the current defaults.
  - Confirm `fomOff` is repeatable, which sets the noise floor.
  - Then a full run. Check the CSV curves and compare the PSDs with the best parameters against gain 0.

### Calibrate Implementation Log
**2026-11-19: implemented** in `apps/counterOffload/counterOffload.hpp`, with tests, docs and the example config.

- **PSD input.**
  - A second `dev::shmimMonitor<counterOffload, psdShmimT>`, with config section `[psds]` and default `shmimName=aol1_raw_clpsds`.
  - `allocate()` records the PSD geometry, nFreq × nModes. A reallocation during a run aborts the run.
  - `processImage()` computes `bandPower()` only while a band is set and only for PSDs received after `m_psdAfterNs`.
  - The frequency resolution `df` is read from `aol1_freq` (`calibrate.freqStream`) at the start of each run, and checked against the PSD length.
- **Offloader values.** `t2wOffloader.fps` and `t2wOffloader.numModes` (`current`) come from SET subscriptions on `calibrate.offloadDevice`.
- **Thread.** A non-RT calibration thread (priority 0, no cpuset) waits for the `calibrate` toggle.
  - It sets the parameters through `setCalibParams()` under `m_stateMutex`.
  - `appLogic` now republishes `rampStartUs`, `rampDurationUs` and `gain` each cycle, because calibration changes them outside the INDI callbacks.
- **`measure()`.**
  - It applies the parameters, then collects PSDs received after one window (1/df) plus `settleS`, and averages `nAverage` of them.
  - It fails on: abort, the app being disabled, a change of more than 0.5 Hz in `fps`, a change in `numModes` (when `modeMax` is 0), or a timeout.
- **Baseline.** Measured at gain 0 at the same ramp, at the start and before every `baselineEvery`-th evaluation. Each ratio uses the latest baseline, not an interpolation between baselines.
- **`coordinateScan()`.** Static, so it can be unit tested.
  - The refine pass covers ± one step at half the step: 5 points, not the ±2 steps written in the design.
  - That is followed by one parabolic-vertex evaluation per parameter.
  - About 57 scan evaluations, plus the initial point and baselines, so about 70 measurements, roughly 5 minutes.
- **End.** On completion the best parameters are applied and logged at `LOG_INFO`, with a warning if the best ratio is ≥ 1. On abort the original parameters are restored and a warning is logged.
- **CSV.** One file per run, `<configName>_calib_<UTC>.csv` in `calibrate.csvDir` (created if missing), flushed after each row.
- **INDI.**
  - `calibrate` toggle: On starts a run and needs the app enabled and OPERATING; Off aborts.
  - `calibHalfWidthHz`, taken at the start of each run.
  - Read-only `calibration`: `stage`, `point`, `fom`, `fomOff`, `ratio`, `bestStartUs`, `bestDurationUs`, `bestGain`, `bestRatio`.
- **Gain limit.** Raised to 0–2 in config validation and over INDI (A-C3).
- **Tests.**
  - `bandPower`: edges, `df` weighting, mode limit, clipping, empty band.
  - `coordinateScan`: convergence on a synthetic quadratic, grid limits, abort.
  - PSD collection gating.
  - Validation of the gain limit and the calibrate settings.
  - The thread-level abort and restore paths are not unit tested; check them on the bench.
  - `bandPower` and `coordinateScan` were also compiled standalone on the Mac with the test's inputs, and gave the expected values (57 evaluations).
- **New edge case.** Changing `gain` scales the whole held contribution `A`. With leak 0 each gain step is a step on the tweeter, which the main loop removes during the settling time. This is documented in `doc/counterOffload.md`.
- **Follow-ups.**
  - Harmonics (A-C4).
  - An interpolated baseline.
  - Telemetry.

**2026-11-19: RTC build.** The user built it on the RTC; it compiles and all unit test cases pass. The bench run is next.

## Calibrate Change: Averaged PSDs and Full Propagation (implemented 2026-11-20, not yet built or tested)

### Request (user, 2026-11-20)
"It takes about 10 seconds for any change in counterOffload parameters to propagate through to the clpsds. We need to 1. change to using the clpsds (10-sec avg) instead of the raw psds, and we need to wait until the effect has propagated through before we evaluate the cost of any combination of parameters."

### Findings
- **What `aol1_clpsds` holds.** It is a single nFreq × nModes plane. It is the mean of the last `ceil(psdAvgTime / (psdTime·overlap))` raw PSDs (`modalPSDs.hpp:1356-1374`): 20 windows at the defaults (`psdAvgTime` 10 s, `psdTime` 1 s, overlap 0.5). It is republished with each new raw PSD, every 0.5 s.
- **Fully propagated time.** The oldest window in the average starts `(N−1)·psdTime·overlap + psdTime` = 9.5 + 1 = **10.5 s** before the publish. So an averaged PSD contains only data taken after a change if it was published at least 10.5 s after the change, plus a small margin for computation. This matches the ~10 s seen on the bench.
- **Why the current code is wrong for this stream.** It waits only one window (1/df = 1 s) plus `settleS` (0.5 s). On the averaged stream, the first PSDs it would accept still contain about 9 s of data from before the change.
- **Averaging.** One fully propagated `aol1_clpsds` sample is already an average of 20 raw PSDs (about 10 s of data), which satisfies A-C1. Successive samples share 19 of 20 windows. So averaging several consecutive samples adds little, and their spread (`fomStd`) does not measure the noise.

### Proposed Change
1. **Stream.** Change the `psds.shmimName` default from `aol1_raw_clpsds` to `aol1_clpsds`. The PSD monitor code is unchanged: it is already a 2-D float stream with the same geometry.
2. **Settling.** Replace "one window + `settleS`" with a single total wait, `settleS`, measured from the parameter change. Only PSDs published after `change + settleS` are collected.
   - Default 11 s (10.5 s + 0.5 s margin). The log and docs explain how it follows from modalPSDs `psdAvgTime` and `psdTime`.
   - Option (see Q-D1): read `psdAvgTime` and `psdTime` from modalPSDs over INDI, and compute the wait automatically.
3. **Samples per evaluation.** Default `nAverage` 1: the first fully propagated averaged PSD.
   - Relax the minimum from 2 to 1. The "never a single point" rule is met by modalPSDs' own 10 s average.
   - If `nAverage` > 1, consecutive samples are taken every 0.5 s, as now.
   - `fomStd` becomes the spread over those samples. It is 0 when `nAverage` = 1 and is documented as not a noise estimate.
4. **Timeout.** `timeoutS` still counts from the end of settling. Default 5 s.
5. **Run time.** At about 11.5 s per measurement, the current grids (about 70 measurements) take about 13–14 minutes. See Q-D3.
6. **Docs, CSV, tests.**
   - Update `doc/counterOffload.md`, the example config, and the CSV column description: `nAverage` is the number of averaged-PSD samples.
   - Update the validation test (`nAverage` 1 allowed, 0 rejected) and the PSD collection test (the settled-time gate is unchanged).

### Questions and Points of Clarification
- **Q-D1. Settling time.** Use a fixed configurable `settleS` (default 11 s), or compute it from modalPSDs' `psdAvgTime` and `psdTime` read over INDI? Computing it would need the modalPSDs device name in config, and would follow any change you make to `psdAvgTime`.
- **Q-D2. Samples per evaluation.** Is one fully propagated `aol1_clpsds` sample per measurement acceptable? Or do you want several, for example 2–3 taken 0.5 s apart, or spaced a full `psdAvgTime` apart so they are independent? Spacing them out would add about 10 s each.
- **Q-D3. Run length.** Keep the current grids (about 13–14 min), or coarsen the coarse pass? For example `rampStartUs` step 50, `rampDurationUs` step 100 and `gain` step 0.2 give about 45 measurements, about 9 min. The refine pass then works at half those steps.
- **Q-D4. Baseline cadence.** At 11.5 s per measurement, a baseline every 5 measurements is about one per minute. Keep 5?

- A-D1: use a fixed settle time of 11s
- A-D2: 3 taken 0.5 s apart
- A-D3: keep the current grid for now
- A-D4: keep 5
### Resolved Design (from A-D1–A-D4)
- **Stream:** the `psds.shmimName` default becomes `aol1_clpsds`.
- **Settling (A-D1):** a fixed `settleS`, default 11 s, counted from the parameter change. It replaces "one window + `settleS`", and the `1/df` term is dropped from the wait. Only PSDs published after `change + settleS` are collected.
- **Samples (A-D2):** `nAverage` default 3: the first three averaged PSDs published after settling, 0.5 s apart.
  - The minimum stays 2, so the A-C1 rule and validation are unchanged.
  - `fomStd` is the spread of those three correlated samples, documented as a stability check, not an independent noise estimate.
- **Grids (A-D3):** unchanged. The run is about 70 measurements at about 12 s (11 s settle + 1 s for 3 samples), so about 14 minutes.
- **Baseline (A-D4):** unchanged, every 5 measurements.
- **Timeout:** the `timeoutS` default drops from 15 s to 5 s after settling. Three samples arrive in about 1 s, so 5 s still allows for a slow modalPSDs.
- **Code changes:**
  - In `measure()`: `m_psdAfterNs = now + settleS`.
  - The defaults and help text for `psds.shmimName`, `settleS`, `nAverage` and `timeoutS`.
  - Docs, example config and the CSV description.
  - Tests: the existing gating test still applies; check the default values.

### Implementation Log
**2026-11-20: implemented.**
- **Wait time:** `measure()` now sets `m_psdAfterNs = now + settleS`, with no `1/df` term.
- **Defaults:**
  - `psds.shmimName` is `aol1_clpsds`;
  - `settleS` 11 s, `nAverage` 3, `timeoutS` 5 s;
  - validation is unchanged (`nAverage` ≥ 2, `timeoutS` ≥ 1).
- **Help text, docs and example config:** updated. The docs explain the 10.5 s figure, note that `fomStd` is a stability check, and say to change `settleS` if modalPSDs' `psdAvgTime` or `psdTime` changes.
- **Test:** a new section checks the three defaults.
- **Instance config on the RTC:** an instance config that copied the earlier example block sets the old values (`aol1_raw_clpsds`, 6, 0.5, 15). Remove those lines or update them.

### Bench: First Averaged-PSD Calibration Run (2026-11-20)
File: `counterOffload_calib_20261120T193905Z.csv`, copied to `~/Documents/MagAO-X/Data/counterOffload/`.

- **Progress.** 19 rows from 19:39:18 to 19:42:59 UTC (221 s), 12.0–12.5 s per row as designed: 4 `off`, 1 `initial`, and 14 `start` rows covering 0–325 µs.
  - The run stopped during the measurement after the 4th `off` row: `start` at 350 µs, gain 0.1. That is about 19 of about 70 measurements, so about a quarter of the run, inside the coarse `rampStartUs` scan.
- **Cause not visible in the CSV.** Rows are written only for completed measurements, and the last row is a baseline at ratio 1. So the abort happened inside `measure()`, not on `abortRatio`. Candidates:
  - (a) abort requested; this includes a reallocation of the PSD stream, for example modalPSDs restarting after a camera fps change, which sets the same flag;
  - (b) the app was disabled;
  - (c) `t2wOffloader.fps` or `numModes` changed;
  - (d) no settled PSDs arrived within `timeoutS`, for example modalPSDs stalled;
  - (e) the process exited.
  - The counterOffload log distinguishes them: a "calibration: <reason>" warning, then "calibration aborted, restored …". Requested from the user.
- **Starting gain was 0.1.** The `initial` row shows gain 0.1, so the whole `start` scan ran at a tenth of full amplitude. Even perfectly timed, gain 0.1 can only remove about 19% of the spike power ((1−0.1)² = 0.81). The scan is therefore nearly insensitive to timing.
  - Observed `start` ratios: 0.887–1.038, mean 0.944, SD 0.048.
- **Noise.** The four `off` levels are 7.92, 7.46, 8.10 and 7.30 ×10⁻⁸: 4.9% relative SD over 3.5 min.
  - The `start` values scatter by 3.6%, about the same as the drift.
  - `fomStd` is about 1% of `fom`, so it underestimates the real point-to-point noise by about 4–5×, as expected for overlapping averages.
  - With this noise, no `rampStartUs` dependence can be seen at gain 0.1.

### Bench: Instability at Gain ≥ 0.4 (2026-11-20), Analysis and Fix (implemented 2026-11-20, not yet built)
**Observation (user).** The loop goes unstable once the gain reaches about 0.4. The tweeter display shows windup, with commands up to about 2×10⁶ against a real limit of about ±1.5. "The coarse search is too extreme."

**Most likely cause: the gain scales the whole held contribution.**
- The command is `−gain·(A + Σ ramps)`. With `leak` 0, `A` is the sum of every offload since the app was enabled. It telescopes to the total change in `aol0_modevalDM` over that time, projected onto the tweeter: potentially most of the low-order shape the woofer holds, which is large, because the woofer is the high-stroke DM.
- The main loop must hold `+gain·A` on its own channel to cancel it. When `gain·|A|` exceeds the tweeter range (±1.5), the tweeter saturates.
  - The loop can no longer null the residual, so its integrator winds up. That is the 2×10⁶ on the display.
  - At 0.1, `0.1·|A|` fitted. At about 0.4 it no longer did.
- A gain step also applies `Δgain·A` to the tweeter in one slot. So during a calibration every gain change, and the gain-0 baselines, steps the tweeter by up to `|A|`.
- This also explains the earlier finding "Why the leak matters": with leak 0, counter-offloading moves woofer stroke back onto the tweeter instead of only cancelling the transient.

**Bench check that would confirm this.** Before raising the gain, look at the peak |value| of `dm01disp08_aux` at gain 0.1. If 4 × that peak exceeds about 1.5, the saturation explanation holds.

**Proposed fix:**
1. **Gain applies at ramp creation, not to the held sum.**
   - Store each ramp as `gain·Δ` when the offload arrives, and fold completed ramps into `A` already scaled. The command becomes `−(A + Σ gain_k·Δ_k·ramp)`.
   - A gain change then affects only new offloads; it never steps the held contribution.
   - The "off" measurement then means "stop countering new offloads", which is what a baseline should be.
2. **Calibration needs `A` to stay small.** A run refuses to start unless `leak` > 0, or it runs with its own `calibrate.leak`, restored at the end (see Q-E2).
   - At 8 kHz, leak 0.0025 per slot is a 50 ms time constant, so `A` holds only the last ~50 ms of offloads, a few offload steps.
3. **Output clamp and abort.**
   - New `counterOffload.maxCommand` (default 0.5): any pixel of the counter command beyond ±`maxCommand` is clipped.
   - Calibration aborts and restores if clipping happens during a run.
   - Normal operation logs a warning, rate-limited.
4. **Gentler gain scan** ("coarse search too extreme"): default `gainGrid` 0.1, 1.0, 0.1. After fixes 1–2, gain ≤ 1 cannot hand stroke to the tweeter. Above 1 over-corrects and is left to manual testing.

**Band question (user): is `halfWidthHz` an FWHM?**
- No. It is the half width of the integration window. At 10 Hz the figure of merit is the PSD power summed over 240–260 Hz.
- `ratio` compares that same 240–260 Hz power with the 240–260 Hz power at gain 0. `fom` is the raw 240–260 Hz power for each parameter set, so comparing `fom` across rows is exactly "the 240–260 Hz power compared with itself". Nothing compares 250 Hz alone with the band.
- With a ~10 Hz-wide spike, ±10 Hz contains essentially all of it plus some background. ±5 Hz (245–255) would cut background but also part of the spike wings.

**Questions:**
- **Q-E1.** Fix 1 (gain at ramp creation): OK?
- **Q-E2.** Calibration leak: refuse to start when `leak` = 0, or apply a separate `calibrate.leak` (default 0.0025) during the run and restore it afterwards?
- **Q-E3.** Clamp: default `maxCommand` 0.5? Clip and warn in normal operation, abort during calibration?
- **Q-E4.** New default gain grid 0.1–1.0 in 0.1 steps? Or a different range?
- **Q-E5.** Which stream showed the 2×10⁶: `dm01disp08_aux` (counter), the main loop channel, or the total `dm01disp`? It confirms where the windup is.

**Answers and implementation (2026-11-20).**
- **Q-E5:** the user reports the 2×10⁶ appeared on the counter channel as well as on `dm01disp`.
  - So the offload loop itself ran away: the saturated tweeter left a residual, the main loop and t2wOffloader integrated it, and every larger offload Δ was added to the counter's `A` and mirrored by it.
  - The clamp therefore limits `A` as well as the output.
- **"Go ahead and implement fix":** Q-E1–E4 are implemented with the proposed defaults.
  1. **Gain at arrival.**
     - `addRamp()` stores `gain·Δ`. `buildCommand()` publishes `−(A + Σ ramps)`.
     - A gain change never rescales `A`.
     - The gain-0 baseline now means "stop countering new offloads".
  2. **Clip.**
     - New `counterOffload.maxCommand` (default 0.5, INDI `maxCommand`).
     - `buildCommand()` clips `A` (anti-windup) and the command to ± `maxCommand`. Each slot that clips increments `stats.clipped`.
     - `appLogic` logs a warning at most once per cycle while clipping continues.
     - `measure()` aborts the calibration if `clipped` changed since the run started.
  3. **Calibration leak (Q-E2: separate leak).**
     - New `calibrate.leak` (default 0.0025, must be > 0).
     - `runCalibration()` sets `m_leak` to it at the start and restores the original at the end, on success or abort.
     - `appLogic` republishes `leak`.
  4. **Gain grid:** the `gainGrid` default is 0.1, 1.0, 0.1.
- **Tests.**
  - The expected accumulator values are now the gain-scaled ones.
  - The harness sets `maxCommand` 100 so the existing tests are not clipped.
  - New: the gain is applied at arrival (a change does not rescale `A`; new offloads use the new gain).
  - New: clipping (command only; `A` anti-windup and recovery; no count within the limit).
  - Validation: `calibrate.leak` 0 and `maxCommand` 0 are rejected; the defaults are checked.
- **Behaviour change.** With `leak` 0 in normal operation, `A` now saturates at `maxCommand` instead of growing, and the app warns. Continued operation should use `leak` > 0.

**2026-11-20 bench: gain 0.8, leak 0.001 → `stats.clipped` = 47697.**
- `clipped` counts slots since the app started in which the command or `A` was clipped, so the rate matters more than the total. 47697 slots is about 6 s of slots at 8 kHz.
- **Expected size of `A`.**
  - With leak 0.001 per slot at 8 kHz, `A` decays with τ = 1/(0.001 × 8000) = 125 ms. At 250 Hz that is about 31 offloads.
  - If successive offloads have the same sign, `A` approaches 0.8 × 31 ≈ 25 times one offload step. Any actuator whose single-step offload exceeds 0.5/25 = 0.02 then holds `A` at the 0.5 clip level.
  - If the offloads have zero mean, `A` stays near √31 × 0.8 ≈ 4.5 steps.
- **Requested from the user:**
  1. the `clipped` / `published` rate over ~10 s;
  2. the peak |`dm01disp08_aux`| (pinned at 0.5?);
  3. the peak |`aol0_tweeterOffload`| of one offload;
  4. whether `aol0_modevalDM` drifts steadily one way (persistent offloading, which would also indicate a loop interaction).

**2026-11-20 bench: windup fixed.**
- With the gain-at-arrival and clip fix installed, the tweeter no longer winds up at gain 1. Before the fix it wound up at about 0.4.
- Clipping still occurs (`stats.clipped` 47697 at gain 0.8, leak 0.001), but the clip now bounds the counter as intended.
- **Open item:** calibration aborts on any clipping during a run. Check the clip rate at the calibration leak (0.0025) before starting a run.

**2026-11-20: dev merge check (`a537e3c6`).** Nothing in t2wOffloader or counterOffload needs to change.
- **dmSyncopation:** only two docs changed (`delivery-preparation-test.md`, `zero-aux-rate-test.md`; the post worker moved from `spare31` to `dmtweeter-post`/CPU44). There are no code changes to dmSyncopation, the producer helper, timing, shmimMonitor or the dm base.
- **ocam2KCtrl:** the descramble work does not change the camwfs `fps` property or its stream names, which t2wOffloader reads.
- **RTC cpusets:** `spare31` is retired and CPU31 is now `camwfs_sw`; `camwfs_descramble` is on CPU32. `spare26` (counterOffload's producer) is unchanged. It now shares L3 domain 24–31 with `camwfs_sw`; the two share no streams, so this is only a cache-capacity neighbour, but watch it if counterOffload's `failed` rate rises.
- **`tests/Makefile.one`:** the changes add ocam-only header dependencies. Still clean before rebuilding the counterOffload test.

### Bench: Calibration Run `20261120T202219Z` (after the windup fix)
File: `~/Documents/MagAO-X/Data/counterOffload/counterOffload_calib_20261120T202219Z.csv`.

- **Progress.** 41 rows over 493 s, about 12 s each.
  - The run completed the coarse start and duration scans and the gain scan up to 0.5. It ended while measuring gain 0.6.
  - **Confirmed by the user (2026-11-20): the clip abort ended the run** ("counter command clipped at maxCommand", with maxCommand=0.5) during the gain 0.6 point. The original parameters were then restored.
- **Baseline.**
  - Band power with the counter off is about 2.1×10⁻⁵, about 270× the earlier run's 7.7×10⁻⁸. The spike now dominates the band, so the measurement is sensitive.
  - It drifts between 1.71 and 2.38×10⁻⁵ over 8 min: a relative SD of 10.7%, and ±16% about the mean as a half-range.
  - Ratios below are recomputed against a baseline linearly interpolated between neighbouring `off` rows. The difference from the logged ratio is up to 0.1.
- **Effect.** Counter-offloading clearly reduces the 240–260 Hz power: ratios 0.39–0.6 at gain 0.4–0.5.
- **Gain (start 25, duration 330).**
  - Ratio by gain: 0.1 → 0.73, 0.2 → 0.66, 0.3 → 0.61, 0.4 → 0.48, 0.5 → 0.39. It is still falling at 0.5.
  - A quadratic fit including ratio 1 at gain 0 is `0.96 − 1.73g + 1.27g²`, with its minimum about 0.37 at g ≈ 0.68. This extrapolates beyond the data.
  - The model |1 − gH|² plus an uncancelled floor fits the shape.
- **Start time (duration 330, gain 0.4).** Earlier is better. Interpolated ratio:
  - 0–75 µs: 0.48 ± 0.08;
  - 100–275 µs: 0.59 ± 0.03;
  - 300–400 µs: 0.74 ± 0.08.
  - The best values are at the bottom edge of the grid. Because the first slot lands 100–225 µs after the offload arrives, a start below about 100 µs means the ramp is already partly complete at the first slot: the counter acts as fast as the scheduling allows. The ramp fitted to the woofer response is later than optimal.
- **Duration (start 25, gain 0.4).** At most a weak trend: 0.47–0.63, mean 0.56 ± 0.05. Against the interpolated baseline, longer durations (400–600 µs) are slightly lower, by about 2σ.
- **Noise and scan method.**
  - Repeated settings disagree by more than the per-row `fomStd` suggests. Start 25 / duration 330 / gain 0.4 gave 0.42 in the start scan but 0.48 in the gain scan, and durations 300–350 at the same start gave 0.56–0.57.
  - The per-point SD of the ratio is about 0.05–0.08.
  - The coordinate scan keeps the single lowest noisy value as "best" (0.395 at start 25). Later stages must beat that lucky value, so the duration scan changed nothing ("winner's curse"). Only gain 0.5 (0.394) beat it, by 0.001.
- **Visualization.** `~/Documents/MagAO-X/simulations/DM_dynamics/offloading/counter_offload/calibration_run_analysis.ipynb` (not in this repo) plots this run and the 193905Z run.

### Clip Handling During Calibration (C2 and C3 implemented 2026-11-20, not yet built)
**Current behaviour.**
- `measure()` fails the whole run if `m_clipped` has changed since the run started.
- `runCalibration()` then restores the original parameters, so a scan that reaches a too-large gain discards everything it found.

**Options:**
- **C1 (config only, next run).** Raise `maxCommand` to about 1.0. The tweeter's real limit is about ±1.5.
- **C2 (code).**
  - Treat a clip during a point as an *infeasible point*, not a run failure: write a CSV row with stage `clipped`, and give the scan an infinite cost for it.
  - When that happens on the gain axis, skip the remaining larger gains.
  - Reset the clip reference (`m_calibClipStart`) before each point, so that one clip doesn't fail every later point.
  - Keep "abort and restore" only for operator abort, the app being disabled, offload-rate or mode changes, timeouts, and `abortRatio`.
- **C3 (code).** On any abort, apply the best parameters found so far instead of the originals, if their ratio is below 1. Log it at WARNING.

**Recommendation:** C1 for the next run, plus C2. C3 is optional.


### Calibration Settings as INDI Properties (implemented 2026-11-20, not yet built)
**Request (2026-11-20).** Make the calibration settings INDI properties and delete them from the config, so that changing them doesn't require editing the config.

**Design.**
- **INDI only, config keys removed.** Each setting gets a code default and an INDI property:
  - `halfWidthHz`, which already exists as `calibHalfWidthHz`; only its config key is dropped;
  - `modeMax`, `nAverage`, `baselineEvery`, `settleS`, `timeoutS`, `abortRatio`;
  - `leak` (calibration leak);
  - `startGridUs`, `durationGridUs`, `gainGrid`. Each grid is one number vector with elements `min`, `max` and `step`.
- **Kept in config.** These are deployment settings that rarely change:
  - `calibrate.freqStream`, `calibrate.offloadDevice`, `calibrate.csvDir`;
  - `psds.shmimName`.
- **New defaults**, from run 202219Z:
  - `startGridUs` 0,150,25;
  - `durationGridUs` 200,600,100;
  - `gainGrid` 0.3,1.0,0.1.
- **Validation.** A new value is checked as `validateConfig` checks it now: grid min ≤ max, step > 0, within limits; leak > 0; nAverage ≥ 2. An invalid value is rejected and logged, and the property is republished unchanged.
- **Changes during a run.** Rejected while a calibration is running. The calibration thread copies the settings under the mutex when the run starts.
- **Lifetime.** Values are not persistent: a restart returns them to the code defaults. Changing a default permanently requires a code change.
- **Other changes.**
  - Remove the `[calibrate]` tuning keys from `dmtweeter_counteroffload.conf`.
  - Update the doc's settings tables to list INDI properties instead.
  - Unit tests: the defaults; validation of each new setter, accepted and rejected.
- **Operator step.** Remove the same keys from the RTC instance config. The app warns about unknown keys, if the config framework does.

**Open questions.**
- **I-Q1.** The operational settings (`gain`, `rampStartUs`, `rampDurationUs`, `maxCommand`, `leak`, `timeOffsetUs`) are already INDI properties, but they also have config keys. Proposal: keep the config keys, since they are the only way the calibrated values survive a restart. Change the defaults to gain 0.5, rampStartUs 25 and maxCommand 1.0, and leave them out of the example config.
- **I-Q2.** Include the clip handling C2, with C3 optional, in the same change?

**Answers (2026-11-20).**
- **I-Q1:** keep the config keys for the operational settings.
- **I-Q2:** implement C2 and C3, each as a separate commit.

**Implementation log.**
- **Settings over INDI.** The calibration tuning settings are INDI properties only:
  - `calibHalfWidthHz`, `calibModeMax`, `calibNAverage`, `calibBaselineEvery`;
  - `calibSettleS`, `calibTimeoutS`, `calibAbortRatio`, `calibLeak`;
  - the grids `calibStartGridUs`, `calibDurationGridUs` and `calibGainGrid`, each with elements `min`, `max` and `step`.
- **How a change is applied.**
  - `applyCalibSetting()` and `applyCalibGrid()` check the range (and that integer settings are whole numbers) and refuse while `m_calibRunning`.
  - The INDI callbacks are thin wrappers around them (`setCalibNumber`, `setCalibGrid`).
  - Calibration checks moved out of `validateConfig()`. The defaults are checked in a unit test.
- **New defaults.**
  - Grids: start 0,150,25; duration 200,600,100; gain 0.3,1.0,0.1.
  - Operational: `rampStartUs` 25, `gain` 0.5, `maxCommand` 1.0. Their config keys are kept.
- **Removed config keys.** The `calibrate.*` tuning keys are gone. **An instance config that still sets them is "Unrecognized" and the app will not start.** Remove them from the RTC config before installing.

- **C2 clip handling.**
  - `measure()` returns 1 (measured), 0 (clipped) or −1 (failure).
  - Clipping is counted from `m_clipDelayS` (1 s) after each change, so the previous point's held contribution can leak out first. That takes about 50 ms at the default calibration leak.
  - `evaluate()` returns 0 for a clipped point and writes a CSV row with stage `clipped`, leaving fom, fomStd and ratio empty.
  - `calibEvalT` now returns int (1 evaluated, 0 infeasible, −1 abort). In `coordinateScan`, an infeasible point ends its sweep: larger values of that parameter are skipped, and only the evaluated points are used for the parabolic refinement.
  - A clipped baseline still aborts the run.
  - The best ratio starts at `numeric_limits<double>::max()`, so a clipped starting point never counts as the best.
- **C3 keep the best on abort.** When a run aborts for any reason, including an operator abort, `runCalibration()` applies the best parameters so far if their ratio is below 1, and logs it as a WARNING. Otherwise it restores the originals, as before. `calibration.best*` is also updated in that case. No unit test: `runCalibration` needs the threads and streams.
