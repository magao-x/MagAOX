# Prompt

Goal: We need to measure the temporal response of single actuator on the DM using WFS images and synchronized pokes, and compare/average it with other responses after changing one parameter. Make this an app.

Currently between each reading of a WFS frame, a signal goes off that is an indicator for the DM to move. We record the response of the DM across time over the series of WFS frames.

We want to incorporate a delay [ms] between the signal going off, and the DM actuator moving. Then we record the new response (function) of a single actuator with this delay incorporated.  

We'll have a list of delays, each a different amount of [ms]. Each delay, we measure the corresponding response function.

We need a metric for the actuator's response each iteration. All the different iterations will be compared at the end and averaged.

The idea is that adding a delay will improve accuracy.

Specific ways we want it to work:
Here is a code example of how MagAO-X was written in the past that you will refer to:
/Users/tiff/Documents/GitHub/MagAOX/libMagAOX/app/dev/dmPokeWFS.hpp

Review AGENTS.md then please analyze this problem, and then formulate a plan.  Update this document below with the plan, and do not make changes until I have approved it.  Do not modify this prompt above the "Plan" header below.  I have already created and switched to the feature branch for this.

# Plan


Status: **Decisions resolved (2026-09-30). Awaiting go-ahead to implement. No code changes yet.**

Branch: `tiffanytn/dmTemporalResponse`

## Problem Restatement

Build a new MagAO-X app that measures the **temporal step response of a single DM actuator** as seen by the WFS:

1. Each WFS frame arrival is the synchronization "signal".
2. After the signal, wait a configurable delay `d` (µs), then poke the actuator.
3. Record the next `N` WFS frames and turn each one into a scalar **response value** `r_k` (0 = unpoked, 1 = fully settled).
4. Repeat for every delay in a list `{d_0, d_1, ...}`, averaging several repetitions per delay.
5. At the end, compare the per-delay responses (summary metrics, best delay) and produce an overall average.

The hypothesis to test: picking the right delay lands the DM step at a consistent point within the WFS exposure. That should make the measured response more repeatable and more accurate.

## Reference Code Reviewed

- `libMagAOX/app/dev/dmPokeWFS.hpp`: CRTP base. It provides the WFS/dark `shmimMonitor` plumbing, the dark-subtracted `m_rawImage`, `m_imageSemaphore` (posted once per WFS frame), the DM channel stream `m_dmStream`/`m_dmImage`, the WFS thread with `single`/`continuous`/`stop` INDI switches, `basicRunSensor()` (a +/- poke that averages into `m_pokeImage`), `telem_pokeloop` telemetry, and the `DMPOKEWFS_*` macros.
- `apps/dmPokeXCorr` and `apps/dmPokeCenter`: existing apps built on `dmPokeWFS`. These are the pattern to follow for the class layout, friend/typedef boilerplate, `loadConfigImpl`, Makefile and tests.
- `libMagAOX/app/dev/shmimMonitor.hpp`: `m_imageStream` is protected, so a derived class can read `md[0].cnt0` and `md[0].atime` for frame counting and timestamps.
- `libMagAOX/app/dev/dm.hpp` and `utils/shmimDelta` (PR #400): existing DM timing and trigger-delta instrumentation. Useful to cross-check the latencies we measure.

## Resolved Decisions

1. **Signal:** the semaphore posted by the `camwfs` stream (the WFS `shmimMonitor`, surfaced as `m_imageSemaphore`). No hardware trigger is involved.
2. **Delay reference:** the delay is measured from the frame's acquisition timestamp `md[0].atime`, not from when the thread wakes up. The app sleeps with `clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, atime + d)`. If that deadline has already passed, it pokes immediately and flags the sample as late.
3. **Units:** everything is in **microseconds**: delays, time axes, and timing metrics. Delays are configured as `vector<float>` in µs (e.g. `0,100,200,...`), must be `>= 0`, and may be longer than a frame period.
4. **Single actuator:** confirmed. The actuator is given as `(x, y)` via `pokecen.pokeX/pokeY`, and exactly one entry is required.
5. **App name:** `temporalResponse`.
6. **Branch:** `tiffanytn/dmTemporalResponse`. The old `tiffanytn` branch was renamed to this, and this plan file was renamed to `dmTemporalResponse.md` to match.
7. **Telemetry:** no app-specific telemetry. **FITS cubes are the output product.** Caveat: `dmPokeWFS::wfsThreadExec()` calls `recordPokeLoop()`, which calls `telem<telem_pokeloop>()`. The app must therefore still derive from `dev::telemeter` with the minimal `checkRecordTimes()` so it compiles. Nothing beyond that base-required `telem_pokeloop` record is added, and `updateMeasurement()` is left at zero. The alternative, making telemetry optional in `dmPokeWFS`, would touch `dmPokeCenter`/`dmPokeXCorr` and is out of scope.

## Proposed App Shape

`apps/temporalResponse/` follows the header-only app pattern (AGENTS rule 22):

```
apps/temporalResponse/
  Makefile
  temporalResponse.cpp        // main() only
  temporalResponse.hpp        // class decl + out-of-class inline defs
  tests/temporalResponse_test.cpp
```

```cpp
class temporalResponse : public MagAOXApp<true>,
                       public dev::dmPokeWFS<temporalResponse>,
                       public dev::shmimMonitor<temporalResponse, dev::dmPokeWFS<temporalResponse>::wfsShmimT>,
                       public dev::shmimMonitor<temporalResponse, dev::dmPokeWFS<temporalResponse>::darkShmimT>,
                       public dev::telemeter<temporalResponse>
```

`dev::telemeter` is present only because `dmPokeWFS` requires it (Resolved Decision 7).

We reuse from `dmPokeWFS`: shmim handling, dark subtraction, DM channel, WFS thread, the single/continuous/stop controls, and `basicRunSensor()` to build the steady-state reference pattern. The base class does not need to change.

To capture per-frame timing, the app overrides `processImage(void*, const wfsShmimT&)`. Under `m_wfsImageMutex` it copies `md[0].cnt0` and `md[0].atime` into `m_frameCnt0` / `m_frameATime`, then calls `dmPokeWFST::processImage()`, which posts the semaphore. This keeps the timestamp consistent with the image in `m_rawImage`. `std::mutex` is not recursive and the base locks it itself, so the override captures the metadata in its own `{ //mutex scope` block and releases the lock before calling the base.

## Measurement Algorithm

### 1. Reference pattern (the `firstRun` step of `runSensor`)
- Call `basicRunSensor()` to get the settled +/- poke difference image `P = m_pokeImage` (steady-state response).
- Build a pixel mask `M` = pixels with `|P| > maskThresh * max|P|` (config, default 0.1) to reject noise-only pixels.
- Precompute `norm = sum_M P^2`.
- Option: `ref.recompute` (default `true`) rebuilds `P` whenever amplitude or actuator changes, or at the start of each run.

### 2. Per-delay sequence (`runSensor`, for each `d` in `m_delays`, for each repeat `j < m_nRepeats`)
1. DM at zero. Record `nPre` frames and average them into the baseline `B`.
2. **Rising edge:** wait for the next frame semaphore and read its `atime` (`t_trig`) and `cnt0`. Sleep until `t_trig + d`, write `+sign*amp` to the DM channel, and record `t_cmd` (actual command time).
3. Record `nPost` frames, storing `cnt0`, `atime`, and `r_k = sum_M (I_k - B) * P / norm`.
4. **Falling edge:** repeat step 2 writing zero, then record `nPost` frames with `r_k = 1 - (...)` so both edges estimate the same rising curve.
5. Alternate `sign` (+1/-1) between repeats, as `dmPokeWFS` does, to cancel even-order nonlinearity and slow drift (the sign is folded back into `r`).
6. Detect dropped frames: if `cnt0` jumps by more than 1, discard that repeat, log a warning, and retry up to `maxRetries`.
7. Check `m_stopMeasurement` / `m_shutdown` between every frame. Always zero the DM on exit, error, or stop.

The time axis for each sample is stored two ways:
- frame index `k` (relative to the trigger frame), and
- `t_k - t_cmd` in µs (relative to the actual DM command), which lets curves from different delays be aligned.

### 3. Per-delay averaging (`analyzeSensor`)
For each delay, across `2 * nRepeats` edges, compute the mean curve `r̄_d(k)` and the standard deviation `σ_d(k)`.

### 4. Response metrics per delay (pure functions, unit-testable)
| Metric | Definition |
|---|---|
| `t50` | Time (µs, from `t_cmd`) where `r̄` crosses 0.5, linearly interpolated |
| `rise` | 10–90 % rise time (µs) |
| `overshoot` | `max(r̄) - 1` |
| `settleErr` | RMS of `r̄ - 1` over the last `nSettle` frames |
| `jitter` | `σ_d` at the transition frame (the frame closest to `t50`). This is the **repeatability / accuracy** figure the delay is expected to improve |
| `delayErr` | mean/std of `(t_cmd - t_trig) - d`. Checks that the requested delay was actually achieved |

### 5. Cross-delay comparison and overall average (end of run)
- Report a table of metrics vs delay and pick the **best delay** as the one with minimum `jitter`. The selection criterion is configurable: `jitter` | `rise` | `t50`.
- **Overall average:** average all curves on the `t - t_cmd` axis, resampled onto a common grid of `dt = frame period / resampleFactor`. Because each delay samples the response at a different phase relative to the exposure, this combination gives an effectively **super-sampled** response function. It is the main scientific product.

## Configuration (new `[temporalResponse]` section, plus the existing `wfscam`/`wfsdark`/`pokecen` sections)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `temporalResponse.delays` | `vector<float>` | `0` | Delays in µs |
| `temporalResponse.nRepeats` | int | 10 | Repeats (edge pairs) per delay |
| `temporalResponse.nPre` | int | 5 | Baseline frames before each edge |
| `temporalResponse.nPost` | int | 20 | Frames recorded after each edge |
| `temporalResponse.nSettle` | int | 5 | Trailing frames used for `settleErr` |
| `temporalResponse.maskThresh` | float | 0.1 | Fraction of `max|P|` for the pixel mask |
| `temporalResponse.maxRetries` | int | 3 | Retries per repeat on dropped frames |
| `temporalResponse.resampleFactor` | int | 10 | Super-sampling factor for the overall average |
| `temporalResponse.bestMetric` | string | `jitter` | Criterion for choosing the best delay |
| `temporalResponse.outputDir` | string | `/opt/MagAOX/calib/temporalResponse` (confirm) | Where result FITS files are written |

Existing `pokecen.dmSleep` is still used by `basicRunSensor()` for the reference pattern only. `pokecen.nPokeImages/nPokeAverage` also still apply to the reference.

## INDI Interface

Inherited: `poke_amp`, `nPokeImages`, `nPokeAverage`, `single`, `continuous`, `stop`, `measurement`.

New:
- `nRepeats`, `nPost` (NewNumber, current/target).
- `delays` (NewText, comma-separated µs). Parsed and validated in the callback, and rejected while measuring.
- `progress` (RO): `delay_index`, `delay_us`, `repeat`, `n_delays`.
- `results` (RO, rebuilt when delays change): `t50_<i>`, `rise_<i>`, `jitter_<i>` per delay.
- `best` (RO): `delay_us`, `t50`, `rise`, `jitter`.
- `last_file` (RO text): path of the last FITS output.

## Outputs

- **Shared memory** (`milkImage`), updated after each delay so the results can be watched live in rtimv/plots:
  - `<configName>_poke` (existing reference `P`)
  - `<configName>_resp`: `nFrames x nDelays` mean curves
  - `<configName>_respstd`: matching standard deviations
  - `<configName>_respavg`: overall super-sampled average curve
- **FITS cubes (primary product)** per completed run: `temporalResponse_<timestamp>.fits` with extensions for the mean curves, std, time axes, raw per-edge curves, and the overall average. Header records the delays, amplitude, actuator, fps, and nRepeats.
- **Logs:** a `text_log` summary line per delay plus the best-delay result. No app-specific telemetry (see Resolved Decision 7).

## Code Organization Within the Header

- `temporalResponse` class (declarations only; definitions out-of-class below, per AGENTS rule 13).
- A small free-function/struct block in `namespace MagAOX::app::temporalResponseMath` (in the same header):
  - `double projectResponse(const eigenImage<float>& im, const eigenImage<float>& base, const eigenImage<float>& P, const eigenImage<float>& mask, double norm)`
  - `int crossingTime(const std::vector<double>& t, const std::vector<double>& r, double level, double& tcross)`
  - `struct responseMetrics { double t50, rise, overshoot, settleErr, jitter; }` and `computeMetrics(...)`
  - `resampleAverage(...)` for the combined curve
  - `parseDelayList(const std::string&, std::vector<float>&)`

  These have no hardware dependency, so they carry most of the unit-test coverage.
- Timing helper `sleepUntil(timespec)` wrapping `clock_nanosleep` with EINTR handling.
- Full Doxygen per AGENTS rules 1, 3, 4, 9, 10: file blocks, `///` briefs, inline `/**< [in] */` parameter docs, and `... - Data` sections placed before accessor sections.

## Unit Tests (`apps/temporalResponse/tests/temporalResponse_test.cpp`)

Catch2, `namespace libXWCTest { namespace temporalResponseTest {...} }`, a `\defgroup temporalResponse_unit_test` in `\ingroup application_unit_test`, and a Doxygen block on every `TEST_CASE`. Uses the `TEMPORALRESPONSE_TEST_DOXYGEN_REF` pattern (rules 20–21).

- `loadConfigImpl`: defaults, valid delays, rejection of negative delays, rejection of more than one actuator, rejection of empty or mismatched poke lists.
- `parseDelayList`: good, empty, garbage, and negative input.
- `projectResponse`: a synthetic `P` and frames `I = B + a*P` give `r == a`; the mask excludes noise pixels.
- `crossingTime` / `computeMetrics`: an analytic first-order step `1 - exp(-t/τ)` plus a pure delay reproduces the known `t50`, rise (`τ ln 9`), and zero overshoot. An underdamped curve gives the correct overshoot. No crossing returns an error.
- `resampleAverage`: interleaved phase-shifted samples of a known curve recover it on the fine grid.
- Default construction and INDI callback validation for `delays` / `nRepeats`.

Hardware-timed sequencing (`runSensor`) is only exercised on the bench/sim (see Verification), not in unit tests.

## Build and Integration

- Add `temporalResponse` to the app lists in the top-level `Makefile`, next to `dmPokeXCorr` (both lists at lines ~78 and ~129).
- Add `../apps/temporalResponse/tests/temporalResponse_test` to `tests/tests.list`.
- Add `apps/temporalResponse/temporalResponse` to `.gitignore`.
- Add an example config `apps/temporalResponse/doc/temporalResponse.md` (usage, config, INDI, output format), following the `utils/shmimDelta/doc` style.
- Run `clang-format` on all touched files.

## Verification Plan

1. Build and run unit tests (`tests/` harness).
2. **Simulated bench:** run against `aoSim`/`cameraSim` with a DM sim channel, if available, to exercise the full single/continuous/stop flow, DM zeroing on stop, and FITS/shmim outputs.
3. **On instrument (you/operator):** at a known WFS fps, run the delay list `0 : 0.1*T_frame : T_frame` (in µs). Check that `delayErr` is small, that `t50` shifts by about `d` (sanity check), and inspect `jitter` vs delay. Cross-check the latency against `shmimDelta` numbers.

## Implementation Steps and Commit Plan (AGENTS rule 19)

1. **Functional commit:** app skeleton (Makefile, `.cpp`, `.hpp`), config, INDI, the `processImage` override, `runSensor`/`analyzeSensor`, math helpers, outputs, and build registration. This plan file gets updated with any decisions made during implementation.
2. **Tests commit:** Catch2 test file plus the `tests.list` entry.
3. **Docs commit:** app doc markdown and Doxygen polish.
4. **Formatting commit:** `clang-format` only, if needed.

## Edge Cases and Risks

- **Scheduling jitter:** the WFS thread priority (`m_wfsThreadPrio`) and cpuset may need raising for delay accuracy of a few µs to tens of µs. `delayErr` makes this visible.
- **Late deadline:** if `atime + d` is already past when the thread wakes (small `d`, or copy latency), the sample is flagged. If the late fraction exceeds a threshold, a warning is logged and the effective minimum delay is reported.
- **Clock domain:** this assumes `atime` is `CLOCK_REALTIME`, the ImageStreamIO default. It will be verified at startup by comparing it to `get_curr_time()`.
- **DM process latency:** the measured response includes the dm app's channel-combine and driver latency. That is intended (it's the end-to-end response), and is noted in the docs.
- **Saturation / nonlinearity:** keep `poke_amp` small. The +/- sign alternation reduces bias.
- **Camera fps change mid-run:** abort the run if `m_wfsFps` changes.
- **Stop/shutdown mid-sequence:** the DM is always zeroed and partial results are discarded, not written.
