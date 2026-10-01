# Prompt
Review the guidelines in AGENTS.md before proceeding. The documents dmTemporalResponse.md and dm_response_app.md both contain prompts and plans to execute the same idea. Review both plans, and make a suggestion in the "plan" section below for how to implement the best parts of each approach into one cohesive software app.

# Plan
Status: **Implemented on `ktwitchell/dm-response` (2026-09-30). Builds cleanly and all unit tests pass on exao2 (2026-10-01). Repeated test runs also pass, and clang-format has been applied. Coverage and hardware acceptance are pending (see section 13 and Debugging).**

App: `dmTemporalResponse`. Branch: `ktwitchell/dm-response` (AGENTS rules 12 and 17).

Sources merged:
- **[DR]** `agents/plans/2026-09-30/dm_response_app.md` (ktwitchell), including the answers to its review questions.
- **[TR]** `agents/plans/2026-09/dmTemporalResponse.md` (tiffanytn), including its resolved decisions.

Where the two conflict, the decisions the users already made in the review answers win. The merge decisions are recorded in section 12 (Resolved Decisions) and the Decision Responses at the end.

## 1. Comparison of the Two Plans

| Topic | [DR] dm_response_app | [TR] dmTemporalResponse | Merged choice |
|---|---|---|---|
| Base classes | `shmimMonitor` only; own measurement thread | Inherits `dev::dmPokeWFS` (+ dark monitor, forced `telemeter`) | **[DR]**. See 1.1 |
| Where the poke happens | In the shmimMonitor RT thread, right after the frame arrives | In the WFS thread, after a second semaphore hop | **[DR]** |
| How the delay is waited | Busy-wait to an absolute deadline | `clock_nanosleep(TIMER_ABSTIME)` | **[DR]** busy-wait, using **[TR]**'s late-deadline flag |
| Delay reference | `md[0].writetime` | `md[0].atime` (acquisition time) | **[TR]** `atime` (decision 5), with `writetime` also recorded for diagnostics |
| Delay set | K evenly spaced over a span (default one frame period); user decision | Explicit list | **[DR]** (user decision) |
| Actuators | One or more `(x, y)` | Exactly one `(x, y)` | **[TR]** single actuator, plus a **new pattern mode** that loads a DM command from a FITS file (decision 6; see 4.0) |
| ± poke scheme | M/2 `+` trials then M/2 `−` trials (user decision) | Alternating sign, rising + falling edge | **[DR]** block ordering; falling edge **dropped** (decision 3) |
| Dark / baseline | None; ± difference cancels bias (user decision) | Dark shmim + `nPre` baseline | **[DR]** (user decision) |
| Response metric | None; images only | Scalar `r_k` projected on reference pattern; t50, rise, overshoot, settleErr, jitter, delayErr; best delay | **[TR]** (the biggest gain from merging) |
| Super-sampled curve | Described in principle only | Resampled average on the `t − t_cmd` axis | **[TR]** |
| Image output | One averaged FITS cube per delay in `/home/xsup/dm_response/<UTC>/` (user decision) | One multi-extension FITS in the calib dir | **[DR]** cubes and location, plus `summary.fits` and `reference.fits` (decision 2) |
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

- **Name:** `dmTemporalResponse` (directory `apps/dmTemporalResponse/`), on branch `ktwitchell/dm-response` (decision 1).
- **Pattern (AGENTS rule 22):** header-only. `dmTemporalResponse.hpp` holds the declaration plus out-of-class inline definitions (rule 13). The `.cpp` holds only `main`.
- **Parents:** `MagAOXApp<true>` and `dev::shmimMonitor<dmTemporalResponse>` (camWFS, section `wfscam`, default `camwfs`). No dark monitor and no telemeter.
- **DM-agnostic:** targets `dm<NN>disp<MM>`, default `dm00disp07` (woofer). NN = 01 is the tweeter and NN = 02 the NCPC.
- **Two poke modes:** a single actuator `(x, y)` [TR], or a DM pattern loaded from a FITS file (decision 6).
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
     - Write the pre-built `±` command (single-actuator or pattern; see 4.0) to the DM, and record `t_cmd`, `t_cmd − atime_trig` and `cnt0`.
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

### 4.0 Poke command: single actuator or FITS pattern (decision 6)

- `poke.mode` selects the command shape; it is built once at run start and the RT thread only copies it.
  - `actuator` (default): **exactly one** `(x, y)` [TR]. The command is `s · amp` at `(x, y)` and 0 elsewhere. A list with zero or more than one entry is rejected, and so is `|amp| > poke.maxCommand`.
  - `pattern`: a 2-D FITS image named by `poke.patternFile`, with command `s · amp · pattern`. So `amp = 1` applies the file exactly as stored, and `amp` stays the one scale knob in both modes. The pattern is **not normalized** (decision 8).
- The pattern is **re-read at every run start**, so an edited file is picked up without restarting the app.
- Pattern validation, at start (refuse to start and log the reason on failure):
  - the file exists and is readable;
  - it is a single 2-D image (a 3-D cube is rejected);
  - its dimensions equal the DM channel's;
  - all values are finite;
  - it is not all zero;
  - `max|amp · pattern|` is at or below `poke.maxCommand` (safety limit, default `1`; decision 9).
- "Upload" means **pointing the app at a FITS file on the RTC filesystem** (decision 7) via the `patternFile` INDI text property or config. INDI does not carry binary files (see 12, item 7).
- Provenance: the pattern actually applied is copied into the run directory as `pattern.fits`, and the headers record `POKEMODE`, `PATFILE` and the pattern's SHA-256.
- The reference pass (4.1) and the analysis work unchanged in both modes, because `P` is measured from whatever command was applied.

### 4.1 Reference pattern (start of run) [TR idea, [DR] mechanics] (decision 4)
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

Falling-edge capture from [TR] is **not** included (decision 3). The zeroing between trials is untriggered and is not recorded.

## 5. Analysis (pure functions, unit-testable) [TR]

In `namespace MagAOX::app::dmTemporalResponseMath` (same header):

| Function | Purpose |
|---|---|
| `delayGrid`, `resolveSpan`, `dmStreamName`, `parseIntList`, `runDirName`, `cubeFileName`, `achievedDelay`, `differenceCube` | from [DR] |
| `validateActuator(x, y, rows, cols)` | exactly one in-bounds actuator [TR] |
| `loadPattern(path, rows, cols, &pattern)`, `validateCommand(amp, maxCommand)`, `validatePattern(pattern, amp, maxCommand)`, `buildPokeCommand(mode, …, sign, amp)` | pattern mode (4.0) |
| `buildMask`, `projectResponse` | from [TR] |
| `crossingTime(t, r, level, &tcross)` | linear-interpolated crossing |
| `computeMetrics(...)` → `responseMetrics{ t50, rise10_90, overshoot, settleErr, jitter, delayErrMean, delayErrStd, lateFrac }` | per-delay metrics |
| `resampleAverage(curves, times, dt)` | super-sampled combined response on a `dt = T/resampleFactor` grid |
| `bestDelay(metrics, criterion)` | criterion is `jitter` (default), `rise` or `t50` |

At the end of the run: a metrics table vs delay, the best delay, and the super-sampled response. These go to the log (a `text_log` line per delay plus the best), to INDI and to the summary FITS.

## 6. Outputs

Directory `/home/xsup/dm_response/<YYYY-MM-DDTHHMMSS>/` (UTC), per the [DR] decision:
- `dmresp_delay_<DDDDD>us.fits`: one averaged ± cube `[nx, ny, N]` per delay, with the [DR] header set (requested/achieved delay stats, K, span, N, M, NINVALID, amp, `POKEMODE`, POKEX/Y or `PATFILE` + SHA-256, DM stream, WFS shmim, fps, DATE-OBS) plus `LATEFRAC`, `T50`, `RISE`, `JITTER`.
- `reference.fits`: `P` and the mask (decision 2).
- `pattern.fits` (pattern mode only): a copy of the applied DM pattern, for provenance.
- `summary.fits` (decision 2; scalars only, no per-trial images): extensions for `r̄[K, N]`, `σ[K, N]`, the time axes `[K, N]`, a binary-table metrics table, and the super-sampled curve with its time axis.
- **Live shmim** [TR, reduced]: `<configName>_ref`, `<configName>_resp` (`r̄` as `N × K`) and `<configName>_respavg`, updated after each delay for rtimv/plots. The large per-delay cubes are not streamed.

## 7. Configuration

Merged table. The [DR] keys are kept, and the [TR] analysis keys are added:

| Key | Default | From |
|---|---|---|
| `wfscam.shmimName` / `wfscam.camDevName` | `camwfs` / = shmimName | both |
| `dm.index` / `dm.channel` | `0` / `7` | DR |
| `poke.mode` | `actuator` | new (decision 6): `actuator` or `pattern` |
| `poke.x`, `poke.y` | required in `actuator` mode; exactly one entry each | TR |
| `poke.patternFile` | `""` (required in `pattern` mode) | new (decision 6) |
| `poke.maxCommand` | `1` | new (decision 9): max absolute DM command allowed in either mode; configurable |
| `poke.amp` | `0` (must be ≠ 0) | both |
| `poke.nDelays` / `poke.delaySpan` | `10` / `0` (= one frame period) | DR |
| `poke.nFrames` (N) | `20` | DR (TR's `nPost = 20` default adopted) |
| `poke.nTrials` (M, even) | `20` | DR |
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
  - `poke_mode` (selection switch: `actuator` / `pattern`);
  - `poke_x`, `poke_y` (number, single value each);
  - `pattern_file` (text: path on the RTC);
  - `poke_amp`, `nDelays`, `delaySpan`, `nFrames`, `nTrials`, `settle`;
  - `nSettle`, `maskThresh`, `bestMetric` (text).
- **Controls:** `start`, `stop` (request switches). [TR]'s `single`/`continuous` are dropped because a run is a finite grid.
- **Read-only:**
  - `dm_stream`, `wfsFps` (set-property on the camera);
  - `status`: state, phase (reference/measuring/analyzing), `delay_index`, `delay_us`, sign, trial, `n_invalid`, `late_frac`;
  - `delays` (text);
  - `results`: `t50_<i>`, `rise_<i>`, `jitter_<i>`, rebuilt when K changes;
  - `best`: `delay_us`, `t50`, `rise`, `jitter`;
  - `pattern_info` (RO text: loaded file, dimensions and SHA-256, or the validation error);
  - `output` (the run directory).

## 9. Tests

The combined test plan keeps all three layers from [DR] and adds [TR]'s math tests.

- **A. Pure helpers:**
  - [DR] A1–A10;
  - `buildMask` / `projectResponse`: `I = B + a·P` gives `r = a`; mask rejects noise pixels;
  - `crossingTime` / `computeMetrics`: an analytic `1 − e^{−t/τ}` step plus a pure delay gives the known t50, `rise = τ ln 9` and zero overshoot; an underdamped curve gives the right overshoot; no crossing → error;
  - `resampleAverage`: interleaved phase-shifted samples recover a known curve;
  - `bestDelay`: each criterion, and ties;
  - `validateActuator`: exactly one in-bounds entry passes; zero entries, two or more entries, and out-of-bounds entries fail;
  - `loadPattern` / `validatePattern`: a valid 2-D file loads; missing file, 3-D cube, wrong dimensions, NaN/Inf, all-zero pattern and `amp·pattern` above `maxCommand` are each rejected;
  - `validateCommand`: `|amp| = 1` passes with the default `maxCommand = 1`; `|amp| = 1.01` fails; negative amplitudes are checked by magnitude;
  - `buildPokeCommand`: actuator mode puts `s·amp` only at `(x, y)`; pattern mode equals `s·amp·pattern` exactly, for both signs.
- **B. App-level, no hardware:**
  - [DR] B1–B23 (config, INDI, `processFrame` state machine with fake clock, full synthetic run), adapted to single-actuator validation;
  - late-deadline flag and `maxLateFrac` abort;
  - clock-domain check;
  - pattern mode: `pattern_file` / `poke_mode` INDI callbacks, rejection while running, the file re-read at start, `pattern.fits` and SHA-256 header written;
  - a full synthetic run in pattern mode (a multi-actuator FITS pattern) with the fake camera;
  - reference-pass correctness.
  - **Upgraded fake camera:** the DM model is a first-order response with known τ plus a pure latency, integrated over each simulated exposure. The full-run test then checks that
    - the cubes cancel bias,
    - the recovered t50 and rise match τ within tolerance,
    - the super-sampled curve matches the analytic model,
    - `summary.fits` and `reference.fits` are correct.

  This closes [TR]'s gap: "sequencing not unit tested".
- **C. Hardware acceptance** (RTC, loop open, `dm00disp07`): [DR] C1–C7, plus pattern mode (a small, known FITS pattern such as a low-order mode, at low `amp`: the DM channel matches `amp · pattern` during the poke and returns to zero afterwards), plus [TR]'s checks:
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

1. **Core measurement:** app skeleton, config, INDI tunables/controls, single-actuator and pattern poke modes, the `processFrame` state machine, the measurement thread, the ± cubes, FITS cube output and build registration, with test layers A ([DR] helpers) and B (config/INDI/state machine/basic full run).
2. **Analysis:** reference pass, projection, metrics, super-sampling, best delay, `summary.fits` / `reference.fits`, INDI `results` / `best`, with the [TR] math tests and the upgraded fake-camera test.
3. **Live outputs and options:** shmim streams, late-frac handling, clock-domain check, plus their tests.
4. **Documentation commit:** Doxygen pass over all touched files (rule 15), app doc page `apps/dmTemporalResponse/doc/dmTemporalResponse.md` (usage, config, INDI, output format; `utils/shmimDelta/doc` style), example config, and `AGENTS.md` if any new standing rule emerges (rule 16).
5. **Formatting commit:** `clang-format` only.
6. **Hardware acceptance** (layer C), with results recorded in the plan.

Phase 1 is useful on its own (it produces the cubes the [DR] prompt asked for), so the analysis can be reviewed separately.

## 11. Risks and Edge Cases

- RT thread busy-wait is bounded to under one frame period per trigger, so no frames are missed. Needs `wfscam.threadPrio` / cpuset set on the RTC.
- camWFS fps changes mid-run → abort ([TR]); the delay grid depends on it.
- Pattern mode: the metrics describe the combined response of the whole pattern, not individual actuators. Documented. A bad or oversized pattern could drive the DM hard, which is why `maxCommand` validation happens before any write.
- Saturation/nonlinearity: keep `poke_amp` small; ± cancels even-order terms.
- Disk: K cubes of `[nx, ny, N]` floats. For camwfs 120×120, N = 20 and K = 10, that is about 11 MB per run.
- Stop/kill always zeroes the DM channel. Verified in B22/B23 and C5.

## 12. Resolved Decisions and Remaining Items

Resolved (see Decision Responses below):
1. App `dmTemporalResponse` on `ktwitchell/dm-response`.
2. `summary.fits` and `reference.fits` are saved alongside the cubes.
3. Falling-edge capture is removed.
4. A dedicated `nRef` reference pass is used.
5. `atime` is the delay reference.
6. Single-actuator framework [TR], plus a FITS pattern mode.

Minor items (confirmed in responses 7–10):

7. Pattern "upload" means pointing the app at a FITS file already on the RTC (via `pattern_file` or config). Transferring the file is outside the app.
8. The pattern command is `±amp · pattern`, not normalized.
9. `poke.maxCommand` defaults to `1` and applies to both modes (configurable).
10. `dm_response_app.md` and `agents/plans/2026-09/dmTemporalResponse.md` are marked as superseded by this plan (done; plan-document edits only).

No open items remain. The plan is awaiting approval to execute.

## 13. Implementation Notes (2026-09-30)

Files:
- `apps/dmTemporalResponse/dmTemporalResponse.hpp`: the app and the `dmTemporalResponseMath` helpers (header-only).
- `apps/dmTemporalResponse/dmTemporalResponse.cpp`: `main` only.
- `apps/dmTemporalResponse/Makefile`.
- `apps/dmTemporalResponse/tests/dmTemporalResponse_test.cpp`.
- `apps/dmTemporalResponse/doc/dmTemporalResponse.md`.
- Build registration: top-level `Makefile` (`apps_rtc`, `all_buildable_apps`), `tests/tests.list`, `tests/Makefile.one` (`-lcfitsio`), `.gitignore`.

Verification status: the code was written on a clone without mxlib, ImageStreamIO, or clang-format, so it has **not been compiled or run**. Before hardware use, on the MagAO-X machine:
1. `make` in `apps/dmTemporalResponse`.
2. From `tests/`: `make -B -f Makefile.one t=../apps/dmTemporalResponse/tests/dmTemporalResponse_test.cpp`, then run the test.
3. `make coverage` for the 100% statement/function target (no `LCOV_EXCL` markers have been added yet; `appStartup`, `appLogic`, `appShutdown`, `allocate`, and `processImage` need the real shmim/INDI environment and are the likely candidates).
4. `clang-format -i` on the touched files (formatting-only commit).
5. Hardware acceptance (layer C).

mxlib API assumptions to check at first build:
- `fitsFile::write(name, arr, header)` works for `eigenCube` and `eigenImage`; the return value is checked generically by `writeOk()` (success must be the zero value of the return type).
- The tests use `fitsFile::read(arr, header, name)` and `fitsHeader["KEY"].value<T>()` to check headers.
- `milkImage::open()` throws on a missing stream; the code also rejects a DM with the wrong size.

Deviations from the plan text, decided during implementation:
1. **Summary output** is three primary-HDU files (`summary_curves.fits`, `summary_metrics.fits`, `summary_superres.fits`) instead of one multi-extension `summary.fits`, because `mx::fits::fitsFile` writes a single HDU. Column and plane meanings are recorded in the headers (`MCOLn`, `PLANEn`, `COLn`).
2. **Late pokes** are kept as valid measurements (the achieved delay is recorded and the analysis uses the actual command time). When `lateFrac > maxLateFrac`, a warning with the minimum achievable delay is logged and the cube gets `LATEFLAG = 1`. They are not retried and do not abort the run, which follows [TR]'s "warning is logged". Delays shorter than the frame-receive latency are always late, so retrying them could never succeed.
3. **Timeouts abort** the run (the camera is presumed stopped); only frame-counter gaps are retried.
4. **INDI results** are read-only text elements (`t50`, `rise`, `jitter`) holding comma-separated per-delay values, rather than one number element per delay. This avoids redefining the property when K changes.
5. `poke_x` / `poke_y` are single-value INDI numbers (single-actuator framework), so `parseIntList` was not needed.
6. **Testability:** `createIndiProperties()` is split out of `appStartup()`, so tests never start the shmimMonitor (which would attach to the real `camwfs` on the RTC). Semaphores are initialized in the constructor. The test DM stream is injected with `m_dmStreamOverride`.
7. **Baseline frame copy** happens after the DM write, to keep the copy off the timing-critical path.
8. **App state** is `OPERATING` whenever the app is up, because the shmimMonitor only runs in its target state. Run status is reported in the `run_state` property.
9. **Pattern SHA-256** is computed by a small in-header implementation (no new library dependency), tested against the standard vectors.
10. **Fake camera** in the tests samples the model DM response at each frame's `atime` (point sampling) rather than integrating over the exposure. This makes the t50 = latency + τ ln 2 and rise = τ ln 9 checks analytic.
11. **Commits:** phases 1–3 were implemented together in one header, so they are one functional commit rather than three.

## Decision Responses 
1. App name should be dmTemporalResponse, should be hosted on ktwitchell/dm-response
2. adding the other two files is fine.
3. eliminate the falling edge as a feature/option.
4. do a dedicated `nRef` pass.
5. `atime` is the best reference, we should use that.
6. Keep [TR]'s single actuator framework. However, we should add another feature: There should be an option to upload a .fits file to create a pattern on the DM as opposed to just poking a single actuator.
7. correct.
8. correct, it should not be normalized.
9. max command should be 1. 
10. Yes, the files should be updated as superceded.


## 14. Implementation Summary (2026-09-30)

I've implemented the merged plan as the `dmTemporalResponse` app and committed it to `ktwitchell/dm-response` in two commits (nothing pushed). **None of it has been compiled or run**, because this Mac has no mxlib, ImageStreamIO or clang-format and the Xcode compiler is locked by its licence prompt. The first build on the MagAO-X machine will probably turn up some compile errors to fix.

**What the app does:**
- **Trigger and poke:** a high-priority thread handles each camWFS frame. On a trigger frame it waits in a tight loop until the frame's `atime` plus the delay, writes the poke to the DM, records when the poke actually happened, then captures N frames. Delays longer than one frame period are held until the right frame. The baseline frame is copied after the poke so the copy doesn't delay it.
- **Measurement run:** a separate thread runs the reference pass (`nRef` +/− pairs), then works through the K evenly spaced delays with M/2 + then M/2 − trials each. A trial with a skipped frame is retried up to `maxRetries`. A timeout, stop, shutdown, or change in fps or camera aborts the run, and the DM is always left at zero.
- **Output:** everything goes in `/home/xsup/dm_response/<UTC>/`:
  - one averaged cube per delay;
  - `reference.fits`;
  - three summary files holding the response curves, the per-delay metrics, and the combined finer-sampled curve;
  - `pattern.fits` in pattern mode.
- **Pattern mode:** the pattern file is re-read and validated at every start (2-D, same size as the DM, finite, not all zero, within `maxCommand` = 1). Its SHA-256 goes in each file header.
- **Tests:** they cover the helpers, config, INDI callbacks, the per-frame logic with a fake clock, and full runs against a fake camera modelling a first-order DM with known τ. The full-run tests check that bias cancels, that t50 = latency + τ ln 2 and rise = τ ln 9, the combined curve, the +/− ordering, retries, aborts, stop/shutdown and pattern mode. They only use uniquely named test streams, never the real `camwfs` or DM channels.

**Where I departed from the plan** (all recorded in section 13 of `dm_response_merged_plan.md`):
- The summary is three single-image FITS files instead of one multi-extension `summary.fits`, because mxlib's FITS writer only writes one image per file.
- Late pokes are kept as valid measurements. If too many are late at one delay, the app logs a warning and flags the cube (`LATEFLAG`) rather than retrying. Delays shorter than the time it takes to receive the frame can never be on time, so retrying them would never succeed.
- Per-delay results are published over INDI as comma-separated text, so the property doesn't have to be rebuilt when K changes.
- INDI property creation is split into `createIndiProperties()` so the tests never start the shmim monitor. On the RTC that monitor would attach to the real `camwfs`.
- Phases 1–3 went into one functional commit because they ended up in one header.

**Files:**
- `apps/dmTemporalResponse/dmTemporalResponse.hpp`, `dmTemporalResponse.cpp` and `Makefile`
- `apps/dmTemporalResponse/tests/dmTemporalResponse_test.cpp`
- `apps/dmTemporalResponse/doc/dmTemporalResponse.md` (in the docs commit)
- Top-level `Makefile`, `tests/tests.list`, `tests/Makefile.one` (links `-lcfitsio`) and `.gitignore`
- `agents/plans/2026-09-30/dm_response_merged_plan.md`

**To do on the MagAO-X machine:**
1. Build the app, then build and run the test with `make -B -f Makefile.one t=../apps/dmTemporalResponse/tests/dmTemporalResponse_test.cpp` from `tests/`.
2. These mxlib calls are guesses, so check them if the build fails:
   - the return type of `fitsFile::write`;
   - `read(arr, header, file)` and `header["KEY"].value<T>()`, which only the tests use.
3. Run `make coverage`. The startup, shutdown and shmim functions will probably need `LCOV_EXCL` markers.
4. Run `clang-format` and commit it separately as formatting-only.
5. Do the hardware checks on the RTC with the loop open, starting with a small `poke_amp` on `dm00disp07`.

## Debugging

First build and test on the MagAO-X computer (exao2), 2026-10-01.

### Environment setup (no code changes)
- **`Detected GCC 11; GCC >= 14 is required`**: `Make/common.mk` enforces GCC >= 14. The fix was to enable the gcc-toolset-14 compiler in the build shell.
- **`No rule to make target '../flatlogs/bin/flatlogcodes'`**: the fresh checkout had never had its base pieces built. Running `make basic` from the repo root builds INDI, `flatlogcodes`, and `libMagAOX`. `libMagAOX/Makefile` has its own rule for `../../flatlogs/bin/flatlogcodes` but depends on `../flatlogs/bin/flatlogcodes` (an existing path mismatch), so building from an app directory alone can't create it.

### 1. `trialResult` enumerator clashes with the `trialTimeout()` accessor (commit `0b2c06c3`)
- **Error:** `dmTemporalResponse.hpp:1497: 'double ...::trialTimeout() const' conflicts with a previous declaration` (the enumerator at line 1011).
- **Cause:** `trialResult` was an unscoped `enum`, so its enumerator `trialTimeout` was placed in class scope, where it collided with the `trialTimeout()` accessor for `poke.trialTimeout`.
- **Fix:** renamed the enumerators to `resultStopped`, `resultValid`, `resultInvalid`, and `resultTimeout` (in `dmTemporalResponse.hpp`: the enum, `runTrial()`, and `runTrialSet()`). The accessor and config key are unchanged.
- **Result:** the app compiles and links with no warnings under `-Wall -Wextra`. The test program also compiles and links, which confirms the mxlib FITS read/write and `fitsHeader` APIs assumed in section 13.

### 2. Test stream creation fails: `ImageStreamIO_createIm_gpu returned an error` (commit `8546b12f`)
- **Symptom:** 31 test cases, 9 passed and 22 failed. The failures were unexpected exceptions from `milkImage::create` (`milkImage.hpp:547`) in the test harness.
- **Cause:** with `MILK_SHM_DIR` unset, ImageStreamIO creates streams in the system shm directory, which the `twitchell` account can't write. Every test that builds the harness failed.
- **Fix:** the harness calls `ensureMilkShmDir()` before any ImageStreamIO call, setting `MILK_SHM_DIR=/tmp/dmTemporalResponse_test/shm` (the same pattern as `dm_test.cpp` and `streamWriter_lifecycle_test.cpp`). The destructor removes the test streams from that directory. This also keeps test streams separate from real system streams.
- **Status:** pending a re-run. The 9 passing cases don't account for all 13 harness-free cases, so some failures may have another cause. Needs the full `grep -B4 -A6 FAILED` output.

The `ERR ... invalid poke.mode: both` log line during the test run is expected: it comes from the configuration test that checks a bad mode is rejected.

### 3. NaN handling under `-ffast-math`, rise-time crossing, and test thread aborts
- **Symptoms** (second test run, after fix 2): 25 test cases ran, 17 passed and 8 failed, and the run then aborted with `SIGABRT` ("terminate called without an active exception").
  - These NaN checks failed: `validateCommand(NaN)`, `validatePattern` with a NaN pixel, `resampleAverage` empty-bin NaN, `bestDelay` with all-NaN metrics, and `!isfinite(m_rise)`.
  - `resampleAverage` with NaN times threw "cannot create std::vector larger than max_size()".
  - The full run, trial-ordering, and retry tests returned -1 from `runMeasurement()`, with `FITS: bad float to formatted string conversion (fits_bad_f2c)` while writing `dmresp_delay_00000us.fits`.
- **Cause A, `-ffast-math`:** the MagAO-X build compiles with `-ffast-math`, which implies `-ffinite-math-only`. GCC may then fold `std::isfinite()`/`std::isnan()` to "finite", so every NaN check was dead code.
  - **Fix:** added `dmTemporalResponseMath::isFinite(double/float)`, which checks the IEEE 754 exponent bits directly (it can't be optimized away), and replaced all `std::isfinite`/`std::isnan` uses in the app and tests.
- **Cause B, NaN in FITS headers:** cfitsio can't format NaN in a header card (`fits_bad_f2c`), so any NaN metric made `writeCube()` fail and aborted the run.
  - **Fix:** every floating-point header card goes through `headerValue()`, which writes `headerSentinel` (`-999`) for non-finite values; the T50/RISE/JITTER comments note this. NaN values in the image data are still written as NaN (cfitsio supports that).
- **Cause C, the rise time could not be computed:** for small delays the first post-poke frame is already above 10% (with τ = 3 ms and 1 ms frames, r ≈ 0.27 at d = 0), so the curve never crossed 0.1 from below and `rise` was NaN. This would also happen on hardware.
  - **Fix:** each per-trial response curve now starts with the pre-poke baseline point, r = 0 (the trigger frame relative to itself) at t = trigger `atime` − command time (≤ 0). Curves have N + 1 points: `summary_curves.fits` is `[N+1, K, 3]` and the live `_resp` shmim is (N+1) × K. Cubes still have N frames.
- **Cause D, the SIGABRT:** in the retry test a `REQUIRE` failed while the injector `std::thread` was still joinable, so destroying it called `std::terminate`.
  - **Fix:** the tests with helper threads (retry, abort on retries, fps change, measurement thread) now store results, join their threads, and only then `REQUIRE`.
- Added the "finite checks and header sentinel" test case. Updated the doc page for the N+1 curves and the -999 sentinel.
- Expected log noise during tests: `Cannot open shm file ..._nodm_...` (the missing-DM test), `FITS: error reading ... junk.fits` (the bad-file test), and `invalid poke.mode: both`.

### 4. fps change not aborting the run; FITS string padding in tests
- **Symptoms** (third test run): 32 test cases, 30 passed and 2 failed. This was the first complete run, with no aborts.
  - Stop/shutdown, "fps change" section: `REQUIRE( rv == -1 )` got `0 == -1`, so the run finished normally after `m_fpsChanged` was set.
  - Pattern-mode run: `fh["POKEMODE"]` read back as `"pattern "`, not `"pattern"`.
- **Cause A (app bug):** `runTrial()` only checked `m_fpsChanged`/`m_camChanged` in the settle sleep and after a 100 ms semaphore-wait timeout. With a short settle time and trials that complete, neither path runs, so an fps change mid-run was never acted on. This would also happen on hardware with a small `poke.settle`.
  - **Fix:** `runTrial()` checks the stop/shutdown/camera/fps flags before arming every trial, and returns `resultStopped` if the camera or fps changed while a trial completed.
- **Cause B (test only):** FITS pads string values to at least 8 characters, so `"pattern"` is stored as `"pattern "` (`"actuator"` is exactly 8, which is why it passed).
  - **Fix:** the test trims trailing blanks with `fitsStr()` on all four string-header reads.

### 5. All tests pass (2026-10-01)
- **Result:** after the fixes in commit `6ece7483`, the app builds cleanly and the full `dmTemporalResponse_test` suite passes on exao2 ("All tests passed").
- **Remaining before hardware use:**
  1. ~~Repeat the full test run several times to check for timing-dependent failures in the fake-camera tests.~~ Done 2026-10-01: repeated runs all passed.
  2. ~~`clang-format -i` on the `apps/dmTemporalResponse` files, as a separate formatting-only commit.~~ Done 2026-10-01 in commit `63a62496` (3 files, whitespace and wrapping only, +142/−133). The app was rebuilt and all tests still pass after formatting.
     - Note: running clang-format with the three paths on one long pasted line produced `dmTemporalResponse_test.cpp: Permission denied`. The line had been broken, so bash tried to execute the test file. Running each file separately worked.
  3. `make coverage` in `tests/` for the 100% statement/function target. Add any `LCOV_EXCL` markers and list them here.
  4. Hardware acceptance (Test Plan layer C) on the RTC with the loop open, starting with a small `poke_amp` on `dm00disp07`.

### 6. Hardware setup on exao2: device missing from cursesINDI (2026-10-01)
- **Setup done:**
  - `make install`;
  - `/opt/MagAOX/config/dmTemporalResponse.conf`;
  - `dmTemporalResponse dmTemporalResponse` added to `proclist_RTC.txt`;
  - `dmTemporalResponse` added to `isRTC.conf` `local=`;
  - `xctrl restart isRTC`, then `xctrl startup dmTemporalResponse`.
- **Symptom:** after the isRTC restart the device did not appear in cursesINDI, even though the tmux session was active, the process was running, and the driver was in `isRTC.conf`. `getINDI -p 7624 "dmTemporalResponse.*.*"` on exao2 *did* return the properties, so the app and isRTC were working.
- **Cause:** cursesINDI was running on a different machine. That machine's xindiserver builds its list of remote RTC drivers from its **own local copy** of `isRTC.conf`, read at startup (`remote.servers`, in `xindiserver::addRemoteServers()`, `m_configDir + "/" + server + ".conf"`). The new driver had only been added to exao2's copy, so restarting the other server re-read the old list.
- **Workaround used:** run cursesINDI on exao2, which talks to isRTC directly. The properties appear.
- **Permanent fix (to do):** propagate the `isRTC.conf` change to the other machines through the shared config repo, then restart their INDI servers so `dmTemporalResponse` is visible instrument-wide.
- No app code changes.

### 7. First hardware runs: no WFS response detected (2026-10-01)
Analysis was done on a personal machine after copying the run directories with `scp`, using a standard-library-only FITS reader (no numpy or astropy).

| Run | Actuator | amp | N | M | nRef | Result |
|---|---|---|---|---|---|---|
| `2026-10-01T225127` | (5, 5) | 0.05 | 5 | 2 | 2 | rmean ≈ 0 (−0.09…0.07); P = noise (mask 9317/14400 px); cube max ≈ 50, flat |
| `2026-10-01T231246` | (5, 8) | 0.15 | 20 | 10 | 2 | rmean ≈ 0 (±0.1, rstd ≈ 0.2); P = noise (mask 9290/14400 px); cube max ≈ 25, flat |

- **Pipeline and timing work.**
  - Both runs completed and wrote every output file, with 0 invalid trials.
  - camWFS runs at 2 kHz (500 µs frames).
  - The poke is written **67.5 ± 2.2 µs** (run 1) and **68.7 ± 1.4 µs** (run 2) after `atime`, range about 65–70 µs. That is the minimum achievable delay, so delays below about 70 µs will always be late (`LATEFRAC = 1` at d = 0, as expected).
  - Averaging behaves correctly: the cube noise fell by about 2× with 5× the trials.
- **No DM signal reached the WFS.** No step appears in `rmean`, and P has no compact spot.
  - Run 1 at (5, 5) was also very likely behind the Magellan central obscuration, which covers roughly the central 3 actuators of the 11-across woofer. An actuator should be picked in the clear annulus.
  - Run 2 at (5, 8), in the clear pupil, at 3× the amplitude and 5× the trials, still shows nothing. That points to the command not reaching the mirror.
- **Leading hypothesis:** the woofer `dm` app finds its dmcomb channels only once, at startup (`dev::dm::findDMChannels()`, which logs `Found N chanels for dm00disp`). If it found fewer than 8 channels, or `dm00disp07` was created after it started, channel 07 is never summed.
  - **To check:**
    - the woofer log line `Found N chanels` (needs N ≥ 8);
    - the creation times of `/milk/shm/dm00disp*.im.shm` compared with the woofer start time;
    - that the woofer is `OPERATING` and camWFS shows pupils;
    - a visual check with `nFrames = 1000` (0.5 s holds), watching `dm00disp07`, `dm00disp`, and `camwfs` in rtimv.
- **Other notes:**
  - The binary in use was built from a tree with uncommitted changes (the log shows `GIT: f52dd19f… MODIFIED`). Rebuild and reinstall from the committed tree before keeping any data.
  - `logdump -f` follows the log forever. Use `logdump -n 1 <app>` to dump the latest file.
  - `nRef` stayed at 2 (config only, with no INDI property).
- **Proposed app improvement:** fail the run with a clear error when the reference pattern has no real signal (e.g. the mask covers more than half the pixels, or the peak is not well above the noise), instead of completing with meaningless curves. Consider also exposing `nRef` over INDI.

### 8. Root cause of no response: the DM stream was opened passive (2026-10-01)
- **Symptom:** in the visual check (`nFrames = 1000`, 0.5 s holds), the poke was not visible in rtimv on **either** `dm00disp07` or the summed `dm00disp`. That rules out the channel-sum hypothesis from item 7: the write was not taking effect even on the channel itself.
- **Cause:** `prepareRun()` called `m_dmStream.passive( true )`, copied from `dev::dmPokeWFS`. In mxlib (dev), `milkImage::passive` is documented as: *"If true then the cnt0 counter is not incremented on post. Usage: this will not trigger the dmcomb process in cacao, so the DM will not pick up the shape until something does trigger it."* When passive, `post()` skips `ImageStreamIO_UpdateIm()`. So every poke was written into `dm00disp07`'s memory, but `cnt0` never advanced: dmcomb never re-summed, the mirror never moved, and rtimv (which keys on `cnt0`) never refreshed. This explains hardware runs 1 and 2. The unit tests did not catch it because they read the test DM's memory directly.
- **Fix:** `prepareRun()` now calls `m_dmStream.passive( false )`, explicitly, since `m_dmStream` is reused across runs, with a comment explaining why. Added a regression check to the "valid actuator start" test: the stream is set passive beforehand, and the test requires `passive() == false` after `prepareRun()`.
- **Also note:** `dmPokeWFS` (used by `dmPokeCenter`/`dmPokeXCorr`) uses `passive( true )`, presumably on purpose for its loop. It was not changed.
- **Next:** rebuild and reinstall from the committed tree, then repeat the visual check (the poke should now blink on `dm00disp07` and `dm00disp`), then repeat run 2 at (5, 8).

### 9. Reference-pattern SNR check (2026-10-01)
- **Why:** hardware runs 1 and 2 (item 7) completed "successfully" even though there was no WFS response, so they produced flat, meaningless curves. The run should fail clearly instead.
- **Why not simpler checks:** the run 1 and 2 data rule them out.
  - The mask fraction was about 65% on pure noise, but a real extended pattern (pattern mode, all four pupils) can cover a similar fraction.
  - "Peak / RMS of unmasked pixels" was about 18 on pure noise, because the unmasked pixels are small by construction.
- **Design:** `dmTemporalResponseMath::referenceSNR()`:
  - estimates the per-frame noise from differences of consecutive frames in the settled window of the reference difference cube, where the settled signal cancels;
  - uses a median estimator, σ = median|Δ| / 0.6745 / √2, so a few pixels still settling don't bias it;
  - scales by 1/√nSettle, because P averages nSettle frames;
  - returns SNR = max|P| / σ_P.
  - For pure noise, SNR ≈ √(2 ln Npix) ≈ 4.4 for 120×120. Run 2 is consistent with this: RMS(P) ≈ 3.8 and peak ≈ 16.8.
  - Noise-free data (σ = 0 with P ≠ 0) reports 1e9. That keeps the noise-free fake-camera tests passing.
- **Behavior:**
  - New config `analysis.minRefSNR` (default 8).
  - `runReference()` writes `reference.fits` (with `REFSNR`, `REFSIGMA`, `MINRSNR` header cards) **before** the check, so a failed run can still be examined.
  - If `REFSNR < minRefSNR`, the run fails with "no WFS response to poke … (check that the DM channel is applied and the actuator is in the pupil)". Otherwise the SNR is logged.
  - `nSettle` must now be >= 2, which is needed for the noise estimate.
- **Tests:**
  - a `referenceSNR` helper test: signal well above noise (σ_P and SNR checked against expected values), pure noise below 6, noise-free, and invalid inputs;
  - config default and override for `minRefSNR`;
  - `nSettle = 1` rejected at start;
  - a new run test with a noisy camera and no DM response (gain 0, noise σ = 2): it must fail with status `error`, a low but non-zero `REFSNR`, `reference.fits` present with matching `REFSNR`, no cubes, and the DM at zero. The fake camera gained an optional Gaussian noise term (`m_noise`).
- The doc page is updated: the check, `minRefSNR`, the `nSettle >= 2` requirement, the reference header cards, and the non-passive DM note.

### 10. First measured DM response (2026-10-01, run `2026-10-01T233546`)
- **Setup:**
  - build `43215e03` (passive fix and reference SNR check);
  - actuator (5, 8), `poke_amp` 0.15;
  - `nFrames` 1000 (0.5 s holds, for the visual check), `nTrials` 2, `nRef` 2, `nSettle` 2, `nDelays` 1.
  - Only `reference.fits` and the `summary_*.fits` files were copied back; the ~60 MB cube was too large to transfer.
- **Visual and stream checks:** `shmimInfo -n dm00disp07 -N 4 -t 2` saw updates (cnt0 now advances), and the poke flashed on `dm00disp07` in rtimv. It was not seen by eye on camWFS.
- **Reference:** `REFSNR` 10.3 (passed `minRefSNR` 8), σ_P 5.7. P has real structure: the mask covers 4976 of 14400 pixels (35%, versus 65% for the pure-noise P in runs 1–2), and the peak is ±59.
- **Response (delay 0):** r = 0.000 (baseline, t = −71 µs), 0.002 (429 µs), **0.243 (929 µs)**, then **≈ 0.39** flat to 500 ms. rstd ≈ 0.03, about 10× below the signal.
  - Nothing appears in the frame ending 429 µs after the command, about 62% of the final level appears in the frame ending 929 µs, and the response is complete by about 1.4 ms (camWFS exposures are 500 µs).
- **Plateau ≈ 0.39 instead of 1:** a normalization bias from a noisy P. With `nSettle` = 2 and `nRef` = 2, the noise energy in Σ_mask P² is comparable to the signal energy, so r ≈ S / (S + N) ≈ 0.39. It is a scale error, not a measurement failure.
  - **Remedies:** a less noisy P (`nSettle` 10 and `nRef` 20 should cut σ_P by about 7×, so r → about 0.97). Or debias the normalization, `norm = Σ_mask P² − N_mask σ_P²`, using the σ_P that `referenceSNR()` already estimates (proposed, not yet implemented).
- **Next run:** `nFrames` 20, `nSettle` 10, `nRef` 20 (config), `nTrials` 20, `nDelays` 10 (0–450 µs in 50 µs steps; delays below about 70 µs are flagged late), actuator (5, 8), amp 0.15.
