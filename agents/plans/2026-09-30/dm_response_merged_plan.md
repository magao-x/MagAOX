# Prompt
Review the guidelines in AGENTS.md before proceeding. The documents dmTemporalResponse.md and dm_response_app.md both contain prompts and plans to execute the same idea. Review both plans, and make a suggestion in the "plan" section below for how to implement the best parts of each approach into one cohesive software app.

# Plan
Status: **Implemented on `ktwitchell/dm-response` (2026-09-30). Build, unit tests, coverage, clang-format, and hardware acceptance are pending on the MagAO-X machine (see section 13).**

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
