# dm-syncopation plan review

Reviewed: [dm-syncopation.md](dm-syncopation.md) ("plan") and [apps/dmSyncopation/doc/dmSyncopation.md](../../../apps/dmSyncopation/doc/dmSyncopation.md) ("guide") at commit `a547e2a1` on `jrmales/dm-syncopation`, 2026-10-08. Reviewer: Claude Opus 5.5.

Evidence comes from static reads of this repo and of the local `milk-xwcl`, `mxlib`, `milkzmq` and `/opt/MagAOX/config` checkouts, plus two small local compile tests on the workstation. Nothing was run on instrument hardware. "Plan L*n*" and "guide L*n*" are line numbers in those two files at `a547e2a1`.

## Summary

The plan records the user's decisions accurately: no writes to `dmC`, a single-counter trigger, one persistent aux stream per channel, always-copy, and retain-on-fault. Its reading of dmcomb's counter-sum logic and of the `m_piTimes` endpoints is correct. The user-authored section is byte-identical to `326e65e8`.

It is not ready for implementation. Three problems would make the initial configuration fail in operation:

1. The strict 6 kHz cap is applied to a *measured* period, which makes N=3 a knife edge (finding 1).
2. The 5 us limit is used as a fault threshold. That would keep the scheduler in recovery much of the time (finding 2).
3. The plan does not account for other code that reads or writes the destination channels: `dev::dm` `makeDelta()`/`zeroAll()`, FDPR on 07, and triggered dmSpeckle (finding 3).

The plan also leaves the combiner's wait mode open, but the cacao setup scripts already settle it: dmcomb polls `cnt0` every 10 us. That fact sets a lower bound on the guard (finding 4).

Separately, the design is much larger than the requirements need. It has three RT threads on one core, a clock-conversion layer, a multi-field provider protocol, about 30 config keys and 20 INDI properties, and seven required site values with no defaults. `dmSpeckle` already implements the core trigger pattern for N=2. Findings 5–9 would simplify the implementation substantially without dropping any requirement.

## Findings that block implementation

### 1. Strict rate cap against a measured period makes N=3 a knife edge

Plan L184, L306; guide L95. `N_rate = floor(Fmax*T)`, where `T` is the *measured* main period. With `Fmax = 6000 Hz`, `N_rate = 3` only if `T >= 500.000 us`:

- If the WFS camera runs slightly fast (e.g. 2000.08 Hz, `T = 499.98 us`), the result is `floor(2.99988) = 2`, permanently.
- If the camera runs at exactly 2000 Hz, the period estimate lands on either side of 500 us from one update to the next, so N alternates between 2 and 3. Each change "invalidate[s] the timing schedule and reacquire[s] phase" (plan L305).

The rule that "a strict cap takes precedence over hysteresis" came from the planning agent, not the user. The user's answer was "Let's go with 6 kHz, so 3 steps, to start with to give some margin", so the intent is N=3.

Recommendation (user decision): let `maxSubframes` set N and treat the rate cap as a sanity bound. Either evaluate the cap with an explicit tolerance (or configure it with margin, e.g. 6500 Hz), or evaluate it against the nominal main rate rather than the running estimate. In both cases add hysteresis so N changes only on a sustained rate change. Add unit tests at `T = 499.98, 500.00, 500.02 us`.

### 2. Using the 5 us limit as a fault threshold will cause thrashing

Guide L136, L142–144; plan L186, L265. `maxTimingErrorUs = 5` applies to both primary-phase residuals and later-step trigger residuals. An executed out-of-limit trigger cancels the cycle and forces reacquisition (`lockFrames = 32`, i.e. at least 16 ms unarmed).

- **Trigger residuals.** If the user's 2 us RMS figure is taken as the trigger-residual sigma, 5 us is 2.5 sigma. About 1.2% of triggers would exceed it if the distribution were Gaussian, and real scheduling tails are heavier. At 4000 aux triggers/s that is roughly 50 recoveries/s, which leaves the scheduler unarmed about 40–50% of the time. Even with tighter residuals, a few tail events per second would cause repeated 16 ms dropouts.
- **Primary-phase residuals.** These include camera readout and GPU MVM compute jitter, not only semaphore wake jitter. Nothing shows they stay within 5 us. They also matter less than the plan implies: later steps are scheduled from the *observed* `P[k]` (plan L297). Prediction error therefore affects only the step-0 deadline and the last-gap keepout, and `tracking.earlyMainUs` already covers those.
- **Delivered timing.** dmcomb's polling (finding 4) quantizes delivered timing to about 10 us. An app-side 5 us bound therefore does not bound what reaches the controller.

The 5 us answer was also given in reply to "Review the proposed guard/step-0 lead" (plan L224–226), so the user may have meant the guard or lead rather than a new parameter. A 5 us guard would not cover dmcomb's polling.

Recommendations:
- Confirm with the user what the 5 us value is for.
- Make the residual limit a qualification statistic and INDI alarm, not a fault.
- Per event: if a late trigger still falls inside its keepout-safe window, execute it; otherwise skip that one step.
- Reacquire only after M consecutive failures, or on genuine main-timing anomalies (timeout, counter reset, main arriving earlier than `earlyMainUs`).

### 3. Other readers and writers of the destination channels are not handled

Key Assumption 1 says the app owns `dm01disp07/08` exclusively, so that only the app writes them and only dmcomb reads them. That is not the case:

**a. `dev::dm::makeDelta()` reads the channels later than dmcomb does.** [dm.hpp:2878](../../../libMagAOX/app/dev/dm.hpp#L2878) re-reads every non-delta channel in the controller, *after* `commandDM()`, to build `dm01disp_delta`. `/opt/MagAOX/config/dmtweeter.conf:23` lists only `dm01disp03` as a delta channel.

The plan's only condition for overwriting a destination is dmcomb consumption (plan L319, guide L80). The controller reads later than that. Take the last aux trigger at 333.3 us: dmcomb picks it up within about 15 us, and the controller finishes processing about 115 us later, so `makeDelta()` reads the channels at roughly 450–470 us. The next step-0 copy must finish by 480 us. A copy that starts earlier mixes step values into `dm01disp_delta`, which `dmtweeter-avg` consumes.

Two fixes are needed:
- Decide whether 07/08 should be added to `deltaChannels`.
- Change the copy-start rule to "no earlier than previous trigger + path latency + K".

**b. `zeroAll()` and `baseReleaseDM()` stop clearing aux contributions.** Both zero every `dm01dispNN` channel ([dm.hpp:2792](../../../libMagAOX/app/dev/dm.hpp#L2792), [dm.hpp:472](../../../libMagAOX/app/dev/dm.hpp#L472)).
- If the app is enabled, its next step re-copies the `_aux` contents, so the clear is undone.
- If the app is disabled, 07/08 are cleared but the `_aux` streams are not, so re-enabling restores the old contribution.

Options: the app detects writes to its destinations that it did not make (by `cnt0`/`writetime`) and disables itself; `zeroAll` also zeroes the `_aux` streams; or an operator rule is documented.

**c. Channel 07 is FDPR's control channel.** `/opt/MagAOX/config/fdpr2_dmtweeter_camsci2.conf:40` has `dmctrlchannel=dm01disp07`. Plan L127 notes this, but the question was never put to the user, whose answer covered only `twRotSpeck`.

Tweeter channel map from configs and `dev::dm` defaults:

| Channel | User |
| --- | --- |
| 00 | flat |
| 01 | `tweeterModes` |
| 02 | `dev::dm` default test channel ([dm.hpp:1331](../../../libMagAOX/app/dev/dm.hpp#L1331)) |
| 03 | `dmC` |
| 04, 05 | FDPR |
| 06 | `twAlign` |
| 07 | FDPR control |
| 09 | `tweeterSpeck`/`tweeterMod` |
| 10 | `tweeterModulator` |

`dm01disp08` and `dm01disp11` have no config references. Suggest 08/11, or an explicit interlock with FDPR.

**d. Triggered writers on other channels add mid-cycle combinations.**
- `tweeterSpeck` and `tweeterMod` are dmSpeckle instances on `dm01disp09`, triggered from `dm01disp03` (`proclist_RTC.txt:27–28`). When `triggerDelay > 0` they increment `cnt0` after the delay ([dmSpeckle.hpp:871](../../../apps/dmSpeckle/dmSpeckle.hpp#L871)). That produces a mid-cycle combination that can land inside a scheduled step's keepout.
- `tweeterModes` (dmMode on `dm01disp01`) writes whenever a user changes it.

Under plan L324, every one of these writes is an unexpected output that drops lock. Decide whether triggered dmSpeckle is forbidden while synco is enabled, and whether occasional dmMode writes should be counted rather than drop lock.

### 4. dmcomb's trigger mode is already known, and it sets a lower bound on the guard

Plan L105, L126 and L355 treat the combiner's wait mode as unknown. The cacao setup script `milk-xwcl:plugins/cacao-src/AOloopControl_DM/scripts/cacao-fpslistadd-DMCH2DISP:34–35` sets `procinfo.triggermode 4` (DELAY) and `procinfo.triggerdelay 0.00001`. `fpssetup.setval.tweeter-vispyr.conf` does not override either. In DELAY mode the loop sleeps 10 us and rechecks the `cnt0` sum (`milk-xwcl:src/CommandLineInterface/processtools_trigger.c:224–235`). The source notes that nanosleep "adds a few x10us of latency on most systems".

Consequences:

- **The route works.** Incrementing `cnt0` alone triggers a combination; no semaphore post is needed, which supports answer 1. The live check reduces to one value: `DMch2disp-01.procinfo.triggermode == 4` on the RTC, read by the user.
- **Guard lower bound.** Main and aux combinations both see a detection delay of 0 to about 10 us plus sleep overshoot. Delivered spacing at the controller jitters by about one poll period, however precise the app's trigger is. The guard must therefore be at least the poll period plus overshoot. The proposed 20 us is plausible; 5 us is not.
- **Coalescing.** Two counter changes within one poll (for example an early main event and an aux step) produce one combination. The "coalesced output" diagnostic should expect this case.
- **Interactions to rule out.** dmcomb drops `astrogridchan` from the trigger sum when astrogrid mode is on. Zero-point-offset channels drive `aol1_dmzpo` (`AOloopControl_DM_comb.c:1291–1337`). Qualification should confirm that neither the astrogrid channel nor the zpo channels include 07/08. Otherwise every aux trigger would also run the ZPO MVM at the aux rate.

## Simplification recommendations

AGENTS.md rule 1 asks for brevity and reuse. These items follow from it.

### 5. Use one RT thread instead of three on one core

Plan L129, L279, L329 and guide L210–214 put three threads on the single exclusive CPU 61, all `SCHED_FIFO` priority 30: a primary `shmimMonitor`, a combined-output `shmimMonitor`, and the worker.

Equal-priority FIFO threads on one core do not preempt each other. The observer that timestamps `dmC` cannot run while the worker copies or spins, and the hand-off between threads adds latency. It also adds failure modes, such as event-queue overflow (the "observer overflow" test case).

Recommended structure:
- Use one `dev::shmimMonitor` on `dmC` through the `SHMIMMONITOR_*` macros. It already handles reconnection, size checks, priority and cpuset.
- Its `processImage()` runs the whole cycle:
  1. Check that `cnt0` advanced, so this is a genuine main event.
  2. Run the timed copy and trigger for `j = 1..N-1`.
  3. Copy step 0 for the next cycle.
  4. Return and wait for the next post.
- Confirm consumption by reading `dm01disp`'s `cnt0` directly (non-blocking) before each copy, instead of running a second monitor thread.
- An early main event shows up on return as a semaphore that is already posted and a `cnt0` jump.

This is the existing dmSpeckle/sparkleClock trigger loop ([dmSpeckle.hpp:796–877](../../../apps/dmSpeckle/dmSpeckle.hpp#L796)) extended to N steps.

### 6. Drop the clock-conversion layer and timestamp the wake in `CLOCK_MONOTONIC`

Plan L137, L302 and guide L194, L231 convert `dmC` `writetime` from `CLOCK_ISIO` to monotonic using paired clock reads, detect discontinuities, and expose two clock domains to providers. `CLOCK_ISIO` is not a fixed clock:

- `ImageStreamIO.h:13–17` selects `CLOCK_TAI` only if `CLOCK_TAI` is already defined when the header is parsed. I compiled two test files against the header locally: including `ImageStreamIO.h` first gives `CLOCK_REALTIME` (0); including `<time.h>` first under `_GNU_SOURCE` gives `CLOCK_TAI` (11).
- `ImageStreamIO.c` defines `_GNU_SOURCE` (line 12) and includes system headers before `ImageStreamIO.h` (line 63). The library's `ImageStreamIO_UpdateIm()` therefore stamps `writetime` with `CLOCK_TAI`.
- On this workstation, TAI minus REALTIME is 0 s because the kernel TAI offset is unset. That probably explains the observation that TAI "falls back to REALTIME". If chrony or PTP sets the offset (`leapsectz`), the two clocks differ by 37 s.

So using `CLOCK_ISIO` inside the app does not guarantee it matches the library. Simpler approach:
- Timestamp the `dmC` semaphore wake with `CLOCK_MONOTONIC` in the waiting thread.
- Schedule with `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)`.
- The per-step integrators absorb the constant wake bias, and the remaining jitter is the semaphore jitter the user already quotes.
- Keep `writetime` only as a diagnostic.

### 7. Reuse the existing passive-write and trigger idiom

Plan L116 and L320 call for a new helper because passive copies must not post semaphores. That restriction is unnecessary. dmcomb polls `cnt0` and ignores semaphores, and the repo already posts semaphores on passive writes: `milkImage::post()` with `passive(true)` (`mxlib:include/improc/milkImage.hpp:797–816`), and dmSpeckle ([dmSpeckle.hpp:864–877](../../../apps/dmSpeckle/dmSpeckle.hpp#L864)). Posting also keeps viewers of `dm01disp07/08` current; `mzmqServerRTC.conf:6` streams both channels.

Use a passive `milkImage` post for preparation and a normal post (`ImageStreamIO_UpdateIm()`) on the trigger channel, unless there is a concrete documented reason for semaphore-free writes.

### 8. Reduce the provider timing interface to what a first provider needs

Plan L310–314 and guide L186–194 specify version, session, generation, validity, a cycle id, four deadlines per step, and per-provider consumer semaphores. No provider exists yet to use any of it.

A minimal v1 is a small `int64` stream holding layout version, generation, state, N, period (ns), last `P` (ns) and the per-step snapshot cutoffs (ns). It is posted once per step, just after each snapshot cutoff. Freeze the layout when the first real provider is written.

The user also needs to say where providers will run. The accelerometer chain runs on ACC001 (`mcp3208Ctrl`) and ICC (`acc001-cbuff`), not on the RTC. A provider writing `dm01disp07_aux` must either run on the RTC or bridge over the network, and a monotonic-clock schedule only helps a provider on the same host.

### 9. Give site values defaults instead of blocking arming on them

Guide L228–244 lists seven required site values with no defaults: `snapshotBudgetUs`, `copyBudgetUs`, `minMainRateHz`, `maxMainRateHz`, `earlyMainUs`, `outputTimeoutUs` and `maxAdvanceUs`. The app cannot arm until all of them are filled in.

A batch copy is two 50x50 float32 images, about 20 KB, which should take a few microseconds. The app can measure its own snapshot and copy times online as a running maximum. The config can then ship conservative defaults (e.g. 10 us copy, 10 us snapshot, main rate 500–4000 Hz, `earlyMainUs` 20 us), and qualification refines them.

The config/INDI surface should shrink as well. `tracking.periodWindow`, `diagnostics.windowSamples`, `feedback.spinUs` and the per-step `timing_stepN` properties could start out as config-only or telemetry-only.

For open review item 4 (snapshot fallback): accept it, but do it in one pass. Read `cnt0` and the write flag, copy, re-read, and use the cache on a mismatch. No retry loop is needed.

## Lower-priority notes

- **Qualification metric already exists.** With `XWC_DMTIMINGS`, `dev::dm` records `m_triggerDelta = m_t0 - m_t0Last` at `processImage()` entry ([dm.hpp:1950](../../../libMagAOX/app/dev/dm.hpp#L1950)). That is the controller's receipt spacing, which directly measures keepout. Plan L106 underrates this. Both `m_piTimes` and `m_triggerDelta` require a bmcCtrl build with `XWC_DMTIMINGS`, and the operator procedure should say so.
- **Semaphore index.** dmSpeckle uses fixed semaphore 9 on `dm01disp03` without registering `semReadPID`. `shmimMonitor` asks for index 5 by default, but a fallback could land on 9. Configure the index explicitly.
- **Naming.**
  - `C` means maximum steps in plan L290 but copy budget in guide L108.
  - `maxAuxRateHz` actually caps the total cadence, including the main step. Consider `maxCadenceHz`.
  - "Steps", "subframes", "capacity" and "slots" are used interchangeably.
- **Readability.** The Agent Findings section mixes superseded decisions (06/07, `spare43`, the `dmC` consumer-semaphore route, `INTENTIONAL HOLD`) with the current contract, in long, dense sentences. Suggest a one-page "current contract" followed by a separate decision log. Let the guide be the single interface spec instead of restating it in the plan.
- **Links.** Absolute `/home/jrmales/Source/...` links and `../../../../mxlib` links break for other users. Use `repo:path` text for external sources.
- **Tests.**
  - Keep host-timing qualification out of `tests.list`; it will be flaky in CI.
  - Center the unit suite on pure functions: N selection (including the finding 1 boundary), schedule arithmetic, step-outcome classification, and state transitions.
- **Guide location.** App user docs normally live in MagAOX-handbook as rst; sparkleClock's Doxygen block links to `../handbook/operating/software/apps/...`. `apps/pi335Ctrl/doc` is a precedent for a repo-local doc. Decide where the final version lives.
- **Verified non-issue.** `dev::dm::findDMChannels()` matches `dm01dispNN.im.shm` exactly ([dm.hpp:1716–1731](../../../libMagAOX/app/dev/dm.hpp#L1716)), so `dm01disp07_aux` does not affect channel discovery.

## Plan claims checked

| Plan claim | Status | Evidence |
| --- | --- | --- |
| dmcomb triggers on a change in the sum of channel `cnt0`, and sums pixels without checking write flags | Confirmed | `AOloopControl_DM_comb.c:1285–1328`, `1107–1150` |
| Combiner wait mode cannot be determined from local config | Superseded: cacao default is DELAY, 10 us | `cacao-fpslistadd-DMCH2DISP:34–35` |
| `m_piTimes` spans `processImage()` entry through `commandDM()` + `makeDelta()`, excluding saturation work | Confirmed; only built with `XWC_DMTIMINGS` | [dm.hpp:1888–1953](../../../libMagAOX/app/dev/dm.hpp#L1888) |
| `milkImage` passive post skips `cnt0` but posts semaphores | Confirmed | `milkImage.hpp:797–816` |
| `CLOCK_ISIO` is TAI if defined, else REALTIME | Partly: depends on include order; the library uses TAI | `ImageStreamIO.h:13–17`; `ImageStreamIO.c:12, 63` |
| `spare61` is CPU 61 on the RTC; `acc001-cbuff` uses `spare61` on another machine | Confirmed (ICC) | [make_cpusets:237–241](../../../rtSetup/RTC/make_cpusets#L237); `acc001-cbuff.conf:3,8`; `proclist_ICC.txt` |
| Channel 07 is also the FDPR control channel | Confirmed | `fdpr2_dmtweeter_camsci2.conf:40` |
| N=3 at T=500 us, K+G=135 us, 6 kHz | Arithmetic correct; boundary fragile (finding 1) | — |
| User-authored section unaltered | Confirmed | Identical in `326e65e8` and `a547e2a1` |

## Decisions needed from the user

1. **Rate cap.** Evaluate it with a tolerance or against the nominal rate, or let `maxSubframes` decide? (finding 1)
2. **The 5 us value.** What is it for, and should a limit violation skip one step or force reacquisition? (finding 2)
3. **Destination channels.** Keep 07, which conflicts with FDPR, or move to 08/11? (finding 3c)
4. **dev::dm integration.** Should 07/08 become `deltaChannels`, and how should `zeroAll`/`releaseDM` treat the `_aux` sources? (findings 3a, 3b)
5. **Other writers.** What is the policy for triggered dmSpeckle and for dmMode writes while synco is enabled? (finding 3d)
6. **Provider location.** Will providers run on the RTC or on ICC/ACC001? This decides the timing-stream clock and transport. (finding 8)
7. **Architecture.** Do you approve the single-thread `shmimMonitor` structure and monotonic wake timestamps? (findings 5–6)
