# dmTemporalResponse: Counter-Offload Mode

Branch: `ktwitchell/counter-offload` (dmTemporalResponse merged in from `ktwitchell/dm-response`, 2026-11-24).
Background, goal and answers to DT-Q1–DT-Q4: `agents/plans/2026-11/counter-offload-app.md`, section "dmTemporalResponse Counter-Offload Mode".

## Task Summary

Give dmTemporalResponse a mode in which it stands in for t2wOffloader. It pokes the woofer with an offload mode shape and posts the matching tweeter Δ to `aol0_tweeterOffload`, so counterOffload makes its real counter ramp. It measures the WFS response at sub-frame resolution, with the loop open, for three cases:
- `woofer`: the woofer poke only;
- `counter`: Δ only, giving the deployed counter's response;
- `both`: the poke and Δ, giving the residual the WFS sees.

Every case is timed from the woofer write, so the counter's curve includes all of its latency.

## Answers (user, 2026-11-23/24)

- **DT-Q1, branch:** work on `ktwitchell/counter-offload`.
- **DT-Q2, Δ pattern:** option (a), built into the app.
  - Woofer = amp · `wModes[k]`; tweeter Δ = amp · `tModesOrtho[k]`.
  - Both come from the cubes t2wOffloader writes in `prepareModes()`.
  - Copies are in `~/Documents/MagAO-X/Data/counterOffload/modeCubes/` (`SHA256SUMS` alongside).
- **DT-Q3, stream:** the app writes `aol0_tweeterOffload` itself, and refuses to start unless `t2wOffloader.offload` is off.
- **DT-Q4, parameters:** the user changes counterOffload's start, duration and gain by hand between runs.

## Findings (from the code)

- **Trial path.** Trigger, busy-wait and poke run in `processFrame()` on the shmimMonitor thread. `writeDM()` writes the woofer channel, then `m_tCmd = m_clock()` (CLOCK_REALTIME) is the time origin of every curve.
- **counterOffload's ramp reference** is its receipt time + `timeOffsetUs`, not the stamp. The stamp only feeds its latency statistics. So the delay between the woofer write and the Δ post moves the counter directly, and the stand-in must reproduce t2wOffloader's gap.
- **t2wOffloader's gap.** t2wOffloader posts Δ after the woofer sempost, with `atime` = the woofer stamp (CLOCK_ISIO). In the posted frame, `writetime − atime` is t2wOffloader's woofer-write-to-post gap.
- **Clocks.** CLOCK_ISIO is CLOCK_REALTIME or CLOCK_TAI depending on the build. The app stamps Δ with `clock_gettime( CLOCK_ISIO )` taken right after the woofer write, and keeps its CLOCK_REALTIME `m_tCmd` for its own time axis.
- **Leak and clipping.** counterOffload holds −gΔ, leaks it at `leak`, and clips at `maxCommand`. Posting −Δ to undo it is exact only with no leak and no clipping. counterOffload's `zero` request clears its state exactly.
- **Reference.** `P` comes from the reference pass of the poke. Projecting every case onto the woofer's `P` puts all three cases in woofer-step units: woofer → 1, counter → −g·κ (κ = the counter-to-woofer scale at the WFS), both → 1 − g·κ.

## Requirements

- **R1. New poke mode `t2wMode`:**
  - woofer command = amp · plane k of `t2w.wooferModesFile`;
  - counter Δ = amp · plane k of `t2w.tweeterModesFile`.
  - Checks:
    - plane sizes match the woofer channel and `aol0_tweeterOffload`;
    - k is within both cubes;
    - the woofer command is within `poke.maxCommand`.
  - Both files are re-read at each run start; their path, size and SHA-256 are logged and written to the headers.
- **R2. New setting `offloadCase`:** `woofer` (default), `counter`, or `both`. `counter` and `both` require `t2wMode`.
- **R3. Posting Δ.** In `counter` and `both`, at the poke time, with the woofer poke written first in `both`:
  - wait `counter.postDelayUs`;
  - write sign · Δ to `counter.stream` (default `aol0_tweeterOffload`) with `ImageStreamIO_UpdateIm_atime`, with `atime` = the CLOCK_ISIO time taken right after the woofer write (or at the poke time in `counter`).
- **R4. Post delay.** `counter.postDelayUs` < 0 (the default) measures the gap at run start from the stream's current frame: `writetime − atime` of t2wOffloader's last post. The run refuses if that frame has no woofer stamp (`atime == writetime`) or the gap is outside 0–1000 µs. A value ≥ 0 is used as given. The value used is logged and written to the headers (`POSTDLY`).
- **R5. End of each trial.** With the woofer zeroed, send `counterOffload.zero.request` = On, then wait `poke.settle`. Zeroing is exact whatever the leak or clipping.
- **R6. Start checks for `counter` and `both`.** The run refuses, and logs why, unless:
  - `t2wOffloader.offload` is reported off;
  - `counterOffload.enable` is reported On;
  - `counter.stream` opens as a float image matching the tweeter plane.

  The existing checks stay. For a run that changes counterOffload's output, also check that `holoop` `loop_state` is off.
- **R7. Reference pass.** It always uses the woofer poke, in all cases. That needs `poke.mode` `t2wMode`, or another woofer mode, for the reference; in `counter`, the reference still pokes the woofer only. Every case's curves are projected onto that `P`, so they share woofer-step units.
- **R8. Metrics.**
  - In `woofer`, as now.
  - In `counter`, each curve is normalized by its own settled value before `t50`/`rise`, and the settled value (−g·κ) is reported.
  - In `both`, report the settled value, the peak |r| and the RMS of r over the captured frames.
  - New `summary_metrics` columns `settled`, `peakAbs`, `residRms`, written for every case.
- **R9. Headers record the counterOffload settings.**
  - New SET subscriptions: `rampStartUs`, `rampDurationUs`, `gain`, `leak`, `enable`, and `latency` (`slotUs`, `pipelineUs`).
  - Header cards: `OFFCASE`, `T2WMODE`, `WMFILE`/`WMSHA`, `TMFILE`/`TMSHA`, `POSTDLY`, `CORAMP0`, `CORAMPD`, `COGAIN`, `COLEAK`.
  - A setting that changes during a run sets `COCHG = 1` (like `LOOPCHG`).
- **R10. Restore and safety.** On every run end (done, stop, error, shutdown), zero the woofer, then request counterOffload `zero`. The app never writes the tweeter or any synco channel.
- **R11. Documentation and tests** per AGENTS.md §16, §19 and §20.

## Questions

None. All are answered in the counter-offload plan. Decisions made here, to be confirmed at review:
- zeroing with counterOffload's `zero` request (R5), rather than posting −Δ;
- the reference pass always using the woofer (R7);
- refusing if `loop_state` is not off for `counter` and `both` (R6).

## Implementation Plan

Files:
- `apps/dmTemporalResponse/dmTemporalResponse.hpp`
- `apps/dmTemporalResponse/tests/dmTemporalResponse_test.cpp`
- `apps/dmTemporalResponse/doc/dmTemporalResponse.md`
- this plan

Steps:
1. **`dmTemporalResponseMath` additions (testable):**
   - `parseOffloadCase`;
   - `loadModePlane( plane, cube file, k, expected size )`, with a size/index check and SHA-256 via `sha256File`;
   - `stampedGapUs( atime, writetime )`, returning −1 when unstamped;
   - `normalizeBySettled` / `residualMetrics` for R8.
2. **Config and INDI:**
   - `t2w.wooferModesFile`, `t2w.tweeterModesFile`, `t2w.mode` (default 4);
   - `offloadCase`;
   - `counter.stream`, `counter.postDelayUs`, `counter.device` (counterOffload), `counter.offloadDevice` (t2wOffloader).
   - Tunables `offload_case` (a selection) and `t2w_mode`, rejected during a run like the others.
   - SET subscriptions for R6 and R9.
3. **`prepareRun()`:**
   - load the planes;
   - build `m_cmdPos`/`m_cmdNeg` (woofer) and `m_deltaPos`/`m_deltaNeg` (tweeter);
   - open `counter.stream` and resolve the post delay;
   - apply the R6 checks.
4. **`processFrame()`, after `writeDM()`:**
   - take the CLOCK_ISIO stamp;
   - busy-wait the post delay;
   - post ±Δ with that `atime` (`counter` skips `writeDM()`, and its time origin is the post time less the post delay, so all cases share the woofer-write origin);
   - record the achieved post delay per trial (`POSTMEAN`/`POSTSTD` per cube).
5. **Trial end and run end:** `zeroDM()`, then `sendNewProperty` counterOffload `zero.request` On.
6. **Analysis:** the R7 reference rule; the R8 metrics and columns; the R9 headers.
7. **Tests:** plane loading (good, wrong size, out-of-range k, over `maxCommand`); case parsing; the stamped gap; the normalized and residual metrics; that counter/both refuse when offload is on or counterOffload is disabled; the `atime` and sign of a posted Δ on a private stream (the pattern of the `t2wOffloader` atime test).
8. **Commits:**
   - functional, with this plan;
   - docs;
   - formatting by the user on exao2.

## Follow-up and Edge Cases

- **The handoff dominates.** counterOffload's ramp starts at its receipt time, so its own handoff (about 16 µs) and the post delay add directly. The app's post delay should match t2wOffloader's within a few µs; R4 measures it.
- **Woofer-write phase.** The delay scan covers every phase of the woofer write in the frame. The phase t2wOffloader actually uses is fixed by its frame processing. To read it off the per-delay results, it could be measured later from t2wOffloader's woofer-write `atime` relative to camWFS frames.
- **Actuator clipping.** t2wOffloader's woofer actuator clipping (`actLim`) is not modelled.
- **Synco.** If synco is not running, the counter produces nothing and `counter` shows no response. The settled value then flags it.
- **Leak.** Leave counterOffload's `leak` at its operational value. A leak of 0.0025 per slot decays the held counter by about 18% over 20 frames (10 ms), which shows in the settled value and the tail. Record it (R9) rather than forcing 0.

## Implementation (2026-11-24, not yet built)

User: "ok, implement". Done in `apps/dmTemporalResponse/dmTemporalResponse.hpp`, its unit test and its doc page. No other app's code was changed.

- **Helpers.**
  - `pokeMode::t2wMode`, `pokeModeName()`.
  - `offloadCase` (`woofer`, `counter`, `both`) with `parseOffloadCase()` and `offloadCaseName()`.
  - `loadCubePlane()`, `stampedGapUs()`, `curveSummary()`.
  - `buildPokeCommand()` treats `t2wMode` like `pattern`.
- **Config and INDI.**
  - Config keys: `t2w.mode`, `t2w.wooferModesFile`, `t2w.tweeterModesFile`, `counter.case`, `counter.stream`, `counter.postDelayUs`, `counter.device`, `counter.offloadDevice`.
  - INDI tunables: `offload_case` (selection) and `t2w_mode`; `poke_mode` gains `t2wMode`.
  - SET subscriptions: t2wOffloader `offload`; counterOffload `enable`, `rampStartUs`, `rampDurationUs`, `gain`, `leak`.
- **Run setup.** `prepareRun()` loads the woofer plane (`t2wMode`). For `counter`/`both`, `prepareCounter()`:
  - applies the R6 checks;
  - opens the stream with ImageStreamIO;
  - loads the tweeter plane at the stream's size;
  - builds ±Δ;
  - resolves the post delay (R4).
- **Trial path.** `processFrame()` writes the woofer unless the trial is a `counter` measurement trial, takes `m_tCmd`, then for counter trials:
  - takes a CLOCK_ISIO stamp;
  - busy-waits the post delay on the app clock;
  - `postCounter()` writes Δ and calls `ImageStreamIO_UpdateIm_atime` with the stamp.
- **Zeroing.** `runTrial()` requests counterOffload's `zero` after each trial that posted; a failed request stops the run. The run's zero guard also requests it.
- **Analysis.**
  - `curveSummary()` fills `settled`, `peakAbs` and `residRms` for every case.
  - `counter` timing metrics are taken on the curve normalized by its settled value.
  - New `summary_metrics` columns (12 in all) and cube cards.
- **Run directory.** `t2wMode` writes `woofer_poke.fits`; `counter`/`both` also writes `counter_delta.fits`.
- **Deviations from the plan.**
  - counterOffload's `latency` property is not subscribed. The cards record its settings only; the post delay is recorded directly.
  - The cube paths are config-only, not INDI.
- **Posting Δ.** It uses ImageStreamIO directly in `postCounter()`. t2wOffloader has the same operation as `atimeMilkImage::postAtime()` in its own header. A later cleanup could move that class to a common header for both apps (AGENTS.md §1); it was not moved here, to keep t2wOffloader unchanged.
- **Removed.** The `\author` tags in both touched files (AGENTS.md §2).
- **Tests.**
  - **Harness.** A counterOffload model: a posted Δ produces −gain·Δ, rising like the DM after the same latency, and a zero request clears it.
  - **New tests:**
    - the helpers;
    - the INDI tunables and set properties, including `COCHG`;
    - the counter start checks (offload on or unknown, counterOffload disabled, loop on, missing stream, unstamped and stamped post delay, the woofer case);
    - a `counter` run: only the reference writes the woofer; one post and zero per trial; the stream's `atime` is the stamp; settled ≈ −gain; t50 = post delay + latency + τ ln 2;
    - a `both` run: residual < 1%, 12 metrics columns.

## Debugging

- 2026-11-24, first bench run with `counter`: start refused with "aol0_tweeterOffload must be a 2-D float image".
  t2wOffloader creates the stream with mxlib's `milkImage::create`, which (to be confirmed on exao2) makes it 3-D
  with one slice. The start check now uses `dmTemporalResponseMath::isFloatPlane`, which accepts 2-D, or 3-D with
  `size[2] == 1`; posting already writes plane 0 only. Unit-tested on metadata directly.
- 2026-11-24, run 2026-11-24T190725 (`counter`, amp 0.5, gain 0.4, 20 delays over 1000 us, 20 trials): posts were
  made (measured gap 17.4 us, achieved ~26 us after the stamp) but `SETTLED` was within +/-0.005 at every delay,
  i.e. no counter at the WFS.  Cause: dmSyncopation's cycle zero is the main loop's command publication, so with
  the loop open synco runs no slots and counterOffload never publishes (confirmed by the user).  The plan's
  loop-open requirement was wrong.
  - Fix: `counter` and `both` now start with the loop open or with holoop on at `loop_gain` 0 and
    `loop_multcoeff` 1 (`dmTemporalResponseMath::loopQuiet`).  A running loop at gain 0 publishes its command
    every frame, so synco runs, and holds the tweeter shape.  The app subscribes to holoop `loop_gain` and
    `loop_multcoeff`, records them as `LOOPGAIN`/`LOOPMULT`, and a change during a run sets `LOOPCHG` (the run
    continues, as for `loop_state`).  `counterSetting` became `watchedSetting` with the change flag as a
    parameter.
  - t2wOffloader stays with `offload` off, not at gain 0: offloading at any gain posts to the stream the app writes.
