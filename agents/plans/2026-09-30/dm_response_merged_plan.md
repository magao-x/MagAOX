# Prompt
Review the guidelines in AGENTS.md before proceeding. The documents dmTemporalResponse.md and dm_response_app.md both contain prompts and plans to execute the same idea. Review both plans, and make a suggestion in the "plan" section below for how to implement the best parts of each approach into one cohesive software app.

# Plan
Status: **Proposal only. Not approved for execution. No code changes yet.**

Sources merged:
- **[DR]** `agents/plans/2026-09-30/dm_response_app.md` (ktwitchell), including the answers to its review questions.
- **[TR]** `agents/plans/2026-09/dmTemporalResponse.md` (tiffanytn), including its resolved decisions.

Where the two conflict, the decisions the users already made in the review answers win, and the remaining conflicts are listed under **Decisions Needed**.

## 1. Comparison of the Two Plans

| Topic | [DR] dm_response_app | [TR] dmTemporalResponse | Merged choice |
|---|---|---|---|
| Base classes | `shmimMonitor` only; own measurement thread | Inherits `dev::dmPokeWFS` (+ dark monitor, forced `telemeter`) | **[DR]**. See 1.1 |
| Where the poke happens | In the shmimMonitor RT thread, right after the frame arrives | In the WFS thread, after a second semaphore hop | **[DR]** |
| How the delay is waited | Busy-wait to an absolute deadline | `clock_nanosleep(TIMER_ABSTIME)` | **[DR]** busy-wait, using **[TR]**'s late-deadline flag |
| Delay reference | `md[0].writetime` | `md[0].atime` (acquisition time) | **[TR]** `atime`, with `writetime` also recorded |
| Delay set | K evenly spaced over a span (default one frame period); user decision | Explicit list | **[DR]** (user decision) |
| Actuators | One or more `(x, y)` | Exactly one `(x, y)` | **[DR]**: one or more (the metrics work on the combined pattern) |
| ± poke scheme | M/2 `+` trials then M/2 `−` trials (user decision) | Alternating sign, rising + falling edge | **[DR]** block ordering; **[TR]** falling edge optional (Decisions Needed) |
| Dark / baseline | None; ± difference cancels bias (user decision) | Dark shmim + `nPre` baseline | **[DR]** (user decision) |
| Response metric | None; images only | Scalar `r_k` projected on reference pattern; t50, rise, overshoot, settleErr, jitter, delayErr; best delay | **[TR]** (the biggest gain from merging) |
| Super-sampled curve | Described in principle only | Resampled average on the `t − t_cmd` axis | **[TR]** |
| Image output | One averaged FITS cube per delay in `/home/xsup/dm_response/<UTC>/` (user decision) | One multi-extension FITS in the calib dir | **[DR]** cubes and location, plus a scalar-only summary FITS from **[TR]** (Decisions Needed) |
| Live output | None | shmim streams of the curves | **[TR]**, reduced (see 6) |
| DM target | `dm<NN>disp<MM>`, default `dm00disp07` (user decision) | `pokecen.dmChannel` string | **[DR]** (user decision) |
| Telemetry | None | Forced minimal `telem_pokeloop` via `dmPokeWFS` | None; possible because `dmPokeWFS` isn't inherited |
| Testability | Injectable clock, `processFrame()`, fake-camera full run, hardware acceptance | Strong math-helper tests; sequencing not unit tested | **Both** |
| Docs | Doxygen | App doc page, `.gitignore` entry | **Both** |

### 1.1 Why not build on `dmPokeWFS`

[TR] reuses a lot of `dmPokeWFS`, but that has three costs:
- **Timing.** The base posts `m_imageSemaphore` only after copying and dark-subtracting the frame, and the poke then happens on a different thread after waking. That adds variable latency before a `clock_nanosleep` that itself jitters by tens of µs. That is the same size as the delay steps being measured.
- **Telemetry.** It forces a `telemeter` parent and the `telem_pokeloop` record, which the app doesn't need.
- **Controls.** Its `single`/`continuous`/`basicRunSensor` flow (untriggered `dmSleep` pokes) doesn't match a triggered, grid-driven run.

The merged app therefore follows [DR]'s structure and borrows `dmPokeWFS` *conventions*: the `(x, y)` pokes, the `milkImage` DM handling, the INDI property style, the `wfsFps` set-property and the thread start/stop pattern. `dmPokeWFS`, `dmPokeCenter` and `dmPokeXCorr` are not changed.

## 2. Merged App Overview

- **Name:** `dmTemporalResponse` (directory `apps/dmTemporalResponse/`). This is proposed as a compromise that keeps the `dm*` prefix (see Decisions Needed #1).
- **Pattern (AGENTS rule 22):** header-only. `dmTemporalResponse.hpp` holds the declaration plus out-of-class inline definitions (rule 13). The `.cpp` holds only `main`.
- **Parents:** `MagAOXApp<true>` and `dev::shmimMonitor<dmTemporalResponse>` (camWFS, section `wfscam`, default `camwfs`). No dark monitor and no telemeter.
- **DM-agnostic:** targets `dm<NN>disp<MM>`, default `dm00disp07` (woofer). NN = 01 is the tweeter and NN = 02 the NCPC.
- **Two products per run:**
  1. **images:** one averaged ± cube per delay, [DR];
  2. **scalars:** response curves, metrics, best delay and the super-sampled response, [TR].
- **Build:** add to `apps_rtc` and `all_buildable_apps` in the top-level `Makefile`, add the test to `tests/tests.list`, and add the binary to `.gitignore`.

## 3. Threading and Timing

Two threads, as in [DR], with [TR]'s timing refinements:

1. **shmimMonitor RT thread**: the state machine in `processFrame(src, atime, writetime, cnt0)`, called from `processImage()`:
   - `ARMED` → trigger frame. Set `t_poke = atime + d`.
   - `WAIT_POKE` → if `t_poke` falls before the next frame is expected, **busy-wait** on the injectable clock to `t_poke`. Otherwise defer to a later frame, so spans longer than one frame period work [DR].
     - If `t_poke` has already passed when the thread wakes, poke immediately and set the **late flag** [TR].
     - Write the pre-built `±amp` command to the DM, and record `t_cmd`, `t_cmd − atime_trig` and `cnt0`.
   - `CAPTURING` → copy the trigger frame and the next N frames (raw, as float) into a **per-trial buffer** `[nx, ny, N+1]` along with each frame's `atime`. Check that `cnt0` is contiguous (gap → trial invalid). Post "trial done".
   - The RT thread does only copies. All arithmetic runs in the measurement thread.
2. **Measurement thread**: orchestration and analysis:
   - A reference pass, then for each delay: M/2 `+` then M/2 `−` trials. For each trial: zero the DM, settle, arm, wait (timeout), zero the DM.
   - On trial done (valid): add the trial buffer into the `+` or `−` image accumulator. Project each frame onto the reference pattern to get the per-trial scalar curve `r_j(k)` [TR]. Store it (scalars only; cheap).
   - Invalid or late-over-threshold trials are retried up to `maxRetries` [DR]/[TR], keeping the halves balanced.
   - The DM is always zeroed on stop, error and shutdown. Files for completed delays are kept, and partial delays are never written.
- **Clock-domain check** [TR]: at startup, confirm `atime` is `CLOCK_REALTIME` by comparing a fresh frame's `atime` with the current time. If the offset is implausible (the threshold, e.g. > 1 s, is to be confirmed), refuse to start.
- **Latency scope:** end-to-end DM latency (channel combine, driver, mechanics) is included in the measurement and documented, not corrected (DR answer 6, TR edge case).

## 4. Measurement Sequence

### 4.1 Reference pattern (start of run) [TR idea, [DR] mechanics]
- Run `nRef` ± trial pairs through the same triggered path, using `d = 0` and `nFrames = N`.
- `P = (mean(+) − mean(−))/2`, averaged over the last `nSettle` frames (steady state).
- Mask `Mk = |P| > maskThresh · max|P|`, and `norm = Σ_Mk P²`.
- `P` is saved as `reference.fits` and published to the `<configName>_ref` shmim.
- This replaces [TR]'s untriggered `basicRunSensor()` reference and needs no dark: the ± difference cancels the bias.

### 4.2 Per delay `d_k` on the grid
- Delay grid `d_k = k·span/K`, `k = 0..K−1`. The default span is one frame period `1e6/fps` µs; `delaySpan` can be set explicitly [DR].
- For each trial `j` with sign `s = +1` (first M/2) or `−1` (last M/2):
  - projection `r_j(k) = s · Σ_Mk (I_k − I_trig) · P / norm`, where `I_trig` is the trial's trigger (pre-poke) frame, so a fully settled `±` poke gives `r = 1`;
  - the time axis `t_j(k) = atime_k − t_cmd` in µs [TR].
- Cube: `C_d = (Σ+ − Σ−)/M`, i.e. `(mean(+) − mean(−))/2` [DR].
- Curve statistics: `r̄_d(k)` and `σ_d(k)` over the M trials [TR].

### 4.3 Optional falling edge [TR], off by default
When `poke.captureFall = true`, the zeroing after each trial is also triggered with delay `d`. The next N frames give a second estimate `1 − r`, which doubles the statistics per trial. This is off by default so that [DR]'s agreed "M/2 + then M/2 −" scheme stays the baseline (Decisions Needed #3).

## 5. Analysis (pure functions, unit-testable) [TR]

In `namespace MagAOX::app::dmTemporalResponseMath` (same header):

| Function | Purpose |
|---|---|
| `delayGrid`, `resolveSpan`, `dmStreamName`, `parseIntList`, `validatePokes`, `runDirName`, `cubeFileName`, `achievedDelay`, `differenceCube` | from [DR] |
| `buildMask`, `projectResponse` | from [TR] |
| `crossingTime(t, r, level, &tcross)` | linear-interpolated crossing |
| `computeMetrics(...)` → `responseMetrics{ t50, rise10_90, overshoot, settleErr, jitter, delayErrMean, delayErrStd, lateFrac }` | per-delay metrics |
| `resampleAverage(curves, times, dt)` | super-sampled combined response on a `dt = T/resampleFactor` grid |
| `bestDelay(metrics, criterion)` | criterion is `jitter` (default), `rise` or `t50` |

At the end of the run: a metrics table vs delay, the best delay, and the super-sampled response. These go to the log (a `text_log` line per delay plus the best), to INDI and to the summary FITS.

## 6. Outputs

Directory `/home/xsup/dm_response/<YYYY-MM-DDTHHMMSS>/` (UTC), per the [DR] decision:
- `dmresp_delay_<DDDDD>us.fits`: one averaged ± cube `[nx, ny, N]` per delay, with the [DR] header set (requested/achieved delay stats, K, span, N, M, NINVALID, amp, POKEX/Y, DM stream, WFS shmim, fps, DATE-OBS) plus `LATEFRAC`, `T50`, `RISE`, `JITTER`.
- `reference.fits`: `P` and the mask.
- `summary.fits` (scalars only, no per-trial images): extensions for `r̄[K, N]`, `σ[K, N]`, the time axes `[K, N]`, a binary-table metrics table, and the super-sampled curve with its time axis.
- **Live shmim** [TR, reduced]: `<configName>_ref`, `<configName>_resp` (`r̄` as `N × K`) and `<configName>_respavg`, updated after each delay for rtimv/plots. The large per-delay cubes are not streamed.

## 7. Configuration

Merged table. The [DR] keys are kept, and the [TR] analysis keys are added:

| Key | Default | From |
|---|---|---|
| `wfscam.shmimName` / `wfscam.camDevName` | `camwfs` / = shmimName | both |
| `dm.index` / `dm.channel` | `0` / `7` | DR |
| `poke.x`, `poke.y` | required | both |
| `poke.amp` | `0` (must be ≠ 0) | both |
| `poke.nDelays` / `poke.delaySpan` | `10` / `0` (= one frame period) | DR |
| `poke.nFrames` (N) | `20` | DR (TR's `nPost = 20` default adopted) |
| `poke.nTrials` (M, even) | `20` | DR |
| `poke.captureFall` | `false` | TR |
| `poke.settle` / `poke.trialTimeout` / `poke.maxRetries` | `0.05 s` / `2 s` / `5` | both |
| `analysis.nRef` | `10` | merged |
| `analysis.nSettle` | `5` | TR |
| `analysis.maskThresh` | `0.1` | TR |
| `analysis.resampleFactor` | `10` | TR |
| `analysis.bestMetric` | `jitter` | TR |
| `analysis.maxLateFrac` | `0.1` | TR (threshold made explicit) |
| `output.baseDir` | `/home/xsup/dm_response` | DR |

## 8. INDI Interface

- **Tunables**, number unless noted; all rejected while running:
  - `dm_index`, `dm_channel`;
  - `poke_x`, `poke_y` (text);
  - `poke_amp`, `nDelays`, `delaySpan`, `nFrames`, `nTrials`, `settle`, `captureFall` (toggle);
  - `nSettle`, `maskThresh`, `bestMetric` (text).
- **Controls:** `start`, `stop` (request switches). [TR]'s `single`/`continuous` are dropped because a run is a finite grid.
- **Read-only:**
  - `dm_stream`, `wfsFps` (set-property on the camera);
  - `status`: state, phase (reference/measuring/analyzing), `delay_index`, `delay_us`, sign, trial, `n_invalid`, `late_frac`;
  - `delays` (text);
  - `results`: `t50_<i>`, `rise_<i>`, `jitter_<i>`, rebuilt when K changes;
  - `best`: `delay_us`, `t50`, `rise`, `jitter`;
  - `output` (the run directory).

## 9. Tests

The combined test plan keeps all three layers from [DR] and adds [TR]'s math tests.

- **A. Pure helpers:**
  - [DR] A1–A10;
  - `buildMask` / `projectResponse`: `I = B + a·P` gives `r = a`; mask rejects noise pixels;
  - `crossingTime` / `computeMetrics`: an analytic `1 − e^{−t/τ}` step plus a pure delay gives the known t50, `rise = τ ln 9` and zero overshoot; an underdamped curve gives the right overshoot; no crossing → error;
  - `resampleAverage`: interleaved phase-shifted samples recover a known curve;
  - `bestDelay`: each criterion, and ties.
- **B. App-level, no hardware:**
  - [DR] B1–B23 (config, INDI, `processFrame` state machine with fake clock, full synthetic run);
  - late-deadline flag and `maxLateFrac` abort;
  - clock-domain check;
  - `captureFall` path;
  - reference-pass correctness.
  - **Upgraded fake camera:** the DM model is a first-order response with known τ plus a pure latency, integrated over each simulated exposure. The full-run test then checks that
    - the cubes cancel bias,
    - the recovered t50 and rise match τ within tolerance,
    - the super-sampled curve matches the analytic model,
    - `summary.fits` and `reference.fits` are correct.

  This closes [TR]'s gap: "sequencing not unit tested".
- **C. Hardware acceptance** (RTC, loop open, `dm00disp07`): [DR] C1–C7, plus [TR]'s checks:
  - `delayErr` small;
  - t50 shifts ≈ `d` across the grid;
  - `jitter` vs delay inspected;
  - latency cross-checked against `shmimDelta`.
- **Standards:**
  - Catch2 with `libXWCTest::dmTemporalResponseTest`;
  - `\defgroup dmTemporalResponse_unit_test` / `\ingroup application_unit_test`;
  - Doxygen on every `TEST_CASE`;
  - `DMTEMPORALRESPONSE_TEST_DOXYGEN_REF` blocks, harness hidden with `\cond`;
  - uniquely named test streams only (never the real `camwfs`/`dmNNdispMM`);
  - 100% statement/function coverage of the app header via `make coverage`, with any `LCOV_EXCL` listed in the plan.

## 10. Implementation Phases and Commits (AGENTS rules 12, 17, 19)

Each phase is a reviewable functional commit on the feature branch, with the plan file updated alongside:

1. **Core measurement:** app skeleton, config, INDI tunables/controls, the `processFrame` state machine, the measurement thread, the ± cubes, FITS cube output and build registration, with test layers A ([DR] helpers) and B (config/INDI/state machine/basic full run).
2. **Analysis:** reference pass, projection, metrics, super-sampling, best delay, `summary.fits` / `reference.fits`, INDI `results` / `best`, with the [TR] math tests and the upgraded fake-camera test.
3. **Live outputs and options:** shmim streams, `captureFall`, late-frac handling, clock-domain check, plus their tests.
4. **Documentation commit:** Doxygen pass over all touched files (rule 15), app doc page `apps/dmTemporalResponse/doc/dmTemporalResponse.md` (usage, config, INDI, output format; `utils/shmimDelta/doc` style), example config, and `AGENTS.md` if any new standing rule emerges (rule 16).
5. **Formatting commit:** `clang-format` only.
6. **Hardware acceptance** (layer C), with results recorded in the plan.

Phase 1 is useful on its own (it produces the cubes the [DR] prompt asked for), so the analysis can be reviewed separately.

## 11. Risks and Edge Cases

- RT thread busy-wait is bounded to under one frame period per trigger, so no frames are missed. Needs `wfscam.threadPrio` / cpuset set on the RTC.
- camWFS fps changes mid-run → abort ([TR]); the delay grid depends on it.
- Multi-actuator pokes: the metrics describe the combined pattern, not the individual actuators. Documented.
- Saturation/nonlinearity: keep `poke_amp` small; ± cancels even-order terms.
- Disk: K cubes of `[nx, ny, N]` floats. For camwfs 120×120, N = 20 and K = 10, that is about 11 MB per run.
- Stop/kill always zeroes the DM channel. Verified in B22/B23 and C5.

## 12. Decisions Needed (before execution)

1. **App name and branch:** `dmTemporalResponse` (proposed), `dmResponse` [DR] or `temporalResponse` [TR]? Which branch hosts the merged work: `ktwitchell/dm-response`, `tiffanytn/dmTemporalResponse`, or a new `<username>/dm-temporal-response`? Retire the other plan file, or mark it superseded?
2. **Output content:** DR answer 4 said "only the final average cubes should be saved." Is adding the scalar-only `summary.fits` and `reference.fits` acceptable (no per-trial images are kept)?
3. **Falling edge:** keep `captureFall` as an option (default off), make it default on, or drop it?
4. **Reference pattern source:** a dedicated `nRef` reference pass (proposed), or derive `P` from the tail frames of the delay cubes after the run? The latter saves time but gives no live per-trial jitter.
5. **Delay reference:** `atime` (proposed, per [TR]) instead of [DR]'s `writetime`. Confirm the camwfs framegrabber fills `atime` with the acquisition time.
6. **Single vs multiple actuators:** [TR] required exactly one. Keep [DR]'s "one or more" (proposed)?
