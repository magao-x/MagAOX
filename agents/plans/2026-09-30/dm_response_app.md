# Prompt
review AGENTS.md, then consider the following goal: we want to measure the dynamic response of the woofer DM. We want to be able to sample the time-series of a DM poke at a higher temporal resolution than the maximum frequency of camWFS. We will do this by creating a C++ app.

The general procedure will be:
1. The test begins. The program will wait until the next semaphore is raised that indicates the camWFS frame has been read out. 
2. Once the semaphore is raised, there is a short delay that is implemented, (e.g. 50 us).
3. After the delay, the command is sent to the DM to poke the desired actuator(s).
4. N frames of camWFS data are captured after the poke is sent
5. The test is repeated M times for a given delay value.
6. The data from the M trials is averaged, such that the output is one datacube of N frames corresponding to that specific delay value.
7. This is repeated for different values of delay. The cubes are saved according to the 'outputs' section earlier in this prompt.


The input parameters will be:
- an actuator index or indices to be poked
- the delay values in microseconds (e.g. 0 us, 50 us, 100 us, 200 us) - this might not need to be input every time, we could have a set of standard delay values that are used every time the test is run
- the number N of WFS frames to be captured during each poke test
- the number M of trials to be conducted for each delay value

Inputs will be recieved from the user via cursesINDI. There should be appropriate INDI properties for each tunable parameter. 

The outputs will be:
- .fits file cubes. There will be one cube for each delay value.
    - Each cube will have N images, which are the averages of each frame across the M trials
    - The cubes will be saved in the user xsup 's home directory. In that directory, there will be a folder named "dm_response". Each time the test is executed, a sub-directory named after the date and time of the test will be created, and the cubes will be saved there.

Propose a plan to create this app that will allow us to examine the dynamics of the woofer DM with high temporal resolution, following all the guidelines in AGENTS.md

# Plan

> **Superseded (2026-09-30):** this plan has been merged into `agents/plans/2026-09-30/dm_response_merged_plan.md` (app `dmTemporalResponse`, branch `ktwitchell/dm-response`). It is kept for the record only; do not implement from it.

Status: **revised after review (answers below). Not yet approved for execution.**

Prompt (condensed): Create a C++ MagAO-X app to measure the dynamic response of the woofer DM at a temporal resolution finer than the camWFS frame period. For each delay value, trigger off the camWFS frame-ready semaphore, wait a short delay, poke the requested actuator(s), capture N camWFS frames, repeat M times, average the trials into one N-frame cube, and save one FITS cube per delay under `~xsup/dm_response/<date-time>/`. All tunables are INDI properties usable from cursesINDI.

## Measurement Principle

camWFS samples at a fixed period `T = 1/fps`. Frame `k` after a poke shows the DM state integrated over the exposure that ends roughly at `k*T - d` after the poke was issued. Stepping the poke delay `d` across a span (by default one frame period, `[0, T)`) samples the step response on a finer grid, spaced by `span/K` instead of by `T`. Put the N-frame averaged cubes from all K delays in order of `k*T - d` and you get a step-response time series that is effectively oversampled by a factor of `K*T/span`.

Each trial set uses M/2 positive pokes followed by M/2 negative pokes. Taking the half-difference `(mean(+) - mean(-))/2` removes the static WFS signal (the DM flat, the static aberrations and the camera bias/dark), so no dark or baseline frames are needed.

This only works if the poke time relative to the frame is **precise and measured**. The design below is shaped mostly by that requirement.

## App Overview

- Name: `dmResponse` (directory `apps/dmResponse/`), following the existing `dmPoke*` naming.
- Pattern (AGENTS.md rule 22): header-only app. `dmResponse.hpp` holds the class declaration plus the out-of-class inline definitions. `dmResponse.cpp` holds only `main`.
- Base classes:
  - `MagAOXApp<true>`
  - `dev::shmimMonitor<dmResponse>` on the camWFS stream (config section `wfscam`, default shmim `camwfs`). Its high-priority RT thread is where the time-critical trigger → delay → poke → capture sequence runs.
  - No dark shmimMonitor (the ± differencing makes it unnecessary).
- DM-agnostic: it targets any of the three MagAO-X DMs through `dm<NN>disp<MM>` (woofer `00`, tweeter `01`, NCPC `02`). The default is the woofer, `dm00disp07`.
- Not a telemeter. The output product is FITS files, and the run status is exposed through INDI and logged.
- Build registration: add `dmResponse` to `apps_rtc` (camWFS and the DMs are on the RTC) and to `all_buildable_apps` in the top-level `Makefile`. Add the test to `tests/tests.list`.
- The `dev::dmPokeWFS` base class is deliberately **not** reused. It pokes first, then sleeps `m_dmSleep`, then flushes and waits for whole frames. Its timing is deliberately *asynchronous* to the camera, which is the opposite of what we need here. Only its conventions are borrowed: the `(x, y)` poke specification, the DM `milkImage` handling, the INDI property style, the `wfsFps` set-property and the thread handling.

## Threading and Timing Design

Two threads, with a clear split between real-time work and bookkeeping:

1. **shmimMonitor thread** (RT priority, configurable `wfscam.threadPrio` / cpuset). Runs a small state machine inside `processImage()` on every camWFS frame:
   - `IDLE`: do nothing.
   - `ARMED`: this frame is the trigger. Store the reference time from the stream metadata (`md[0].writetime`, i.e. when the camera process posted the frame) and compute `t_poke = writetime + d`. Move to `WAIT_POKE` and fall through.
   - `WAIT_POKE`: if `t_poke` falls before the next frame is expected (`t_poke < writetime_this_frame + T`), **busy-wait** on `clock_gettime(CLOCK_REALTIME)` until `t_poke`. `microSleep` has tens of µs of scheduler jitter, which is the same size as the delays being measured, so it can't be used here. Otherwise return and re-check on the next frame. This supports delay spans longer than one frame period without ever busy-waiting more than about one frame period. When `t_poke` is reached:
     - write the pre-built `±amp` poke command to the DM channel (a single memcpy plus semaphore post via `milkImage` assignment);
     - record the achieved delay (`t_written - writetime_trigger`) and the current `cnt0`;
     - move to `CAPTURING`.
   - `CAPTURING`: add each subsequent frame into the positive or negative accumulator (the sign is set for the trial) at index `n = 0..N-1`. Check that `cnt0` is contiguous; a gap marks the trial as invalid. After N frames, post a "trial done" semaphore and move to `IDLE`.
2. **Measurement thread** (normal priority, modeled on the `dmPokeWFS` `wfsThread`). Orchestrates the run:
   - Validate the parameters, compute the delay list, open the DM channel `dm<NN>disp<MM>`, create the output directory, and build the `+amp` and `-amp` poke images from the `(x, y)` lists.
   - For each delay `d_k`:
     - Run M/2 **positive** trials, then M/2 **negative** trials. Each trial: zero the DM channel, wait the settle time, set the sign and `ARMED`, wait on the "trial done" semaphore (with timeout), then zero the DM again.
     - An invalid trial (frame gap or timeout) is retried, up to a configurable retry limit per delay, so that the + and − halves each end with exactly M/2 valid trials.
     - Compute `cube = (sum(+) - sum(-)) / M`. This is `(mean(+) - mean(-))/2` because each half has M/2 trials, i.e. the response to a poke of amplitude `amp`. Write the FITS cube. Disk I/O stays entirely off the RT thread.
   - Supports stop/shutdown between and inside trials. The DM channel is always zeroed on exit, error or stop.

Shared state between the threads (`m_state`, delay, sign, trial-valid flag) uses `std::atomic`. The accumulators are touched only by the RT thread while `CAPTURING` and only by the measurement thread otherwise, with a `{ //mutex scope` guard around the handoff (AGENTS.md rule 14).

DM command latency (channel combining in the `dm` base class, the driver write and the mechanical response) and the camera's own readout-to-post latency are **out of scope at this stage** (answer 6). The delay is referenced to the camera `writetime` stamp and recorded as measured.

## Delay Grid

- K evenly spaced delays: `d_k = k * span / K`, for `k = 0 .. K-1`. The endpoint `span` is excluded because, for the default span, `d = T` is equivalent to `d = 0` one frame later.
- Default `span` = one camWFS frame period, `T = 1e6 / fps` µs. It is computed at run start from the camera's INDI `fps` property. A start is rejected (with a logged error) if fps is unknown while using the default span.
- `span` can be set explicitly to any duration in µs. This overrides the frame-period default and may exceed `T` (handled by `WAIT_POKE` above).
- The computed delay list is published read-only over INDI and written to each FITS header.

## Configuration

Config file / command line (loaded in `loadConfig`, all mirrored to INDI where tunable):

| Key | Type | Default | Meaning |
|---|---|---|---|
| `wfscam.shmimName` (from shmimMonitor) | string | `camwfs` | camWFS stream |
| `wfscam.camDevName` | string | = shmimName | INDI device used to read `fps` |
| `dm.index` | int | `0` | DM number `NN` in `dm<NN>disp<MM>`: 00 woofer, 01 tweeter, 02 NCPC |
| `dm.channel` | int | `7` | channel number `MM` in `dm<NN>disp<MM>` |
| `poke.x` | vector<int> | *(required)* | x-coordinates of the actuators to poke (as in `dmPokeWFS` `pokeX`) |
| `poke.y` | vector<int> | *(required)* | y-coordinates of the actuators to poke (as in `dmPokeWFS` `pokeY`) |
| `poke.amp` | float | `0.0` | poke amplitude in DM command units; must be non-zero to start |
| `poke.nDelays` | int | `10` | K, number of evenly spaced delays |
| `poke.delaySpan` | float | `0` | span of the delay grid in µs; `≤ 0` means one camWFS frame period |
| `poke.nFrames` | int | `10` | N, frames captured per trial |
| `poke.nTrials` | int | `20` | M, total trials per delay (M/2 positive, M/2 negative); must be even |
| `poke.settle` | float | `0.05` s | wait after zeroing before the next trial arms |
| `poke.trialTimeout` | float | `2.0` s | max wait for a trial to complete |
| `poke.maxRetries` | int | `5` | max invalid-trial retries per delay before aborting the run |
| `output.baseDir` | string | `/home/xsup/dm_response` | root of the output tree |

The DM channel name is built as `dm` + two-digit `NN` + `disp` + two-digit `MM` (so `dm00disp07`). `NN` is limited to `0..2` and `MM` to the channels that exist, which is checked when the stream is opened.

## INDI Interface (cursesINDI)

Following `dmPokeWFS` conventions (`CREATE_REG_INDI_NEW_NUMBER*`, `current`/`target` elements, `indiTargetUpdate`). Every tunable parameter gets a property, and changes are rejected with a logged warning while a run is in progress:

- `dm_index`, `dm_channel` (number, int). Changing either one closes and reopens the DM stream at the next run start. The read-only `dm_stream` text shows the resolved name, e.g. `dm00disp07`.
- `poke_x`, `poke_y` (text, `current`/`target`): comma-separated coordinates, e.g. `5,6` / `5,5`. Text is used because INDI number vectors have a fixed length, and it is easy to edit in cursesINDI. At start the two lists must be the same length and inside the channel dimensions.
- `poke_amp` (number, float).
- `nDelays` (number, int), `delaySpan` (number, float µs; `0` = one frame period).
- `nFrames`, `nTrials` (number, int; `nTrials` must be even).
- `settle` (number, float, s).
- `start` (request switch), `stop` (request switch).
- `status` (RO): `state` (idle/running/error), `delay_index`, `delay_us`, `sign` (+1/−1), `trial`, `n_invalid`, `last_delay_mean_us`, `last_delay_std_us`.
- `delays` (RO text): the computed delay list for the current/last run.
- `output` (RO text): directory of the current/last run.
- `wfsFps` (set-property on `camDevName.fps`), used for the default span.

Validation on `start`: `poke_x`/`poke_y` are the same non-zero length and inside the channel dimensions, `amp ≠ 0`, `nFrames ≥ 1`, `nTrials` is even and `≥ 2`, `nDelays ≥ 1`, and the span is resolvable (explicit `> 0`, or fps known).

## Output

- Directory: `<output.baseDir>/<YYYY-MM-DDTHHMMSS>/` in **UTC**, created with `mkdir -p` at run start. The app runs as `xsup`, so permissions follow from that.
- **Only the final averaged cubes are saved** (answer 4). One per delay: `dmresp_delay_<DDDDD>us.fits`, float, shape `[nx, ny, N]`, holding `(mean(+) - mean(-))/2` over the M valid trials. Fractional-µs delays are rounded in the file name; the exact value is in the header.
- FITS header keywords per cube: `DELAYUS` (requested, exact), `DLYIDX`/`NDELAYS`/`DLYSPAN`, `DLYMEAN`/`DLYSTD`/`DLYMIN`/`DLYMAX` (achieved, µs), `NFRAMES`, `NTRIALS`, `NINVALID`, `POKEAMP`, `POKEX`/`POKEY`, `DMSTREAM`, `WFSSHMIM`, `WFSFPS`, `DATE-OBS`.
- Writing uses `mx::fits::fitsFile` and `fitsHeader`, as in `dmSpeckle`/`dmMode`.

## Code Structure (dmResponse.hpp)

Keep the time-critical and the pure logic separate so the pure logic can be unit tested without hardware:

- Testability hooks: `processFrame()`, the injectable clock, the overridable `m_dmStreamName` and a callable `runMeasurement()` (details in the Test Plan).
- Free/static helpers (pure, testable):
  - `parseIntList(const std::string &)`: comma list → vector, with error on bad token.
  - `dmStreamName(int nn, int mm)` → `dmNNdispMM`.
  - `validatePokes(x, y, rows, cols)`.
  - `delayGrid(nDelays, span_us)` → vector of `d_k`.
  - `resolveSpan(delaySpan, fps)` → µs, or error.
  - `runDirName(timespec)` → `YYYY-MM-DDTHHMMSS` (UTC).
  - `cubeFileName(delay)`.
  - `achievedDelay(writetime, t_poke)` → µs.
  - `differenceCube(sumPos, sumNeg, M)`.
- Class sections, each documented per AGENTS.md rules 3–5, 9–10 and 13: "Configurable Parameters - Data" (protected) before "Configurable Parameters" (public accessors); MagAOXApp interface; shmimMonitor interface (`allocate`, `processImage`); measurement thread; INDI interface with `INDI_NEWCALLBACK_DECL`/`DEFN`; all definitions out of class.
- `m_` member prefix, `///` on every non-trivial member, inline `/**< [in] ... */` parameter docs, Doxygen `\file`/`\brief`/`\author` blocks, and `\defgroup dmResponse` / `dmResponse_files`.

## Test Plan

Testing is in three layers: (A) Catch2 unit tests of the pure helpers; (B) Catch2 app-level tests of configuration, INDI, the per-frame state machine and a full synthetic run, all with no hardware; (C) a manual hardware acceptance procedure on the RTC. Layers A and B run in CI / `tests/testMagAOX.bash`. Layer C is run by an operator and its results are recorded in this plan file.

### Goals and Standards

- **Coverage target:** 100% statement and function coverage of the app-specific code in `dmResponse.hpp`, measured with the existing `make coverage` target in `tests/` (gcov). The only allowed exclusions are code that cannot run in the test build, such as RT-priority/cpuset thread setup. Each exclusion is marked `// LCOV_EXCL_LINE` / `LCOV_EXCL_START/STOP` with a one-line reason, and listed in this file.
- **File:** `apps/dmResponse/tests/dmResponse_test.cpp`, added to `tests/tests.list`. It is built with `make -B -f Makefile.one t=../apps/dmResponse/tests/dmResponse_test.cpp` from `tests/`.
- **Documentation (AGENTS.md rules 20–21):**
  - `namespace libXWCTest { namespace dmResponseTest { ... } }`;
  - a `\defgroup dmResponse_unit_test` block with `\ingroup application_unit_test` in `tests/groups.dox` style;
  - a Doxygen brief on every `TEST_CASE`;
  - `#ifdef DMRESPONSE_TEST_DOXYGEN_REF` blocks referencing the real members under test, wrapped in `// clang-format off/on`;
  - the harness class hidden with `\cond` / `\endcond`.
- **Tags:** every case is tagged `[dmResponse]`, plus `[helpers]`, `[config]`, `[indi]`, `[statemachine]` or `[run]`, so each layer can run on its own.
- **Isolation (safety):** tests never open the real `camwfs` or any real `dmNNdispMM` stream. That matters because the tests may be run on the RTC itself.
  - The harness points the app at uniquely named test streams (`dmresp_test_<pid>_cam`, `dmresp_test_<pid>_dm`), created with `mx::improc::milkImage::create` as in `libMagAOX/app/dev/tests/dm_test.cpp`, and destroyed at the end of each case.
  - Output goes to a per-test temporary `output.baseDir`, which is removed afterwards.
- **Determinism:** no test depends on wall-clock timing or scheduler behavior. Time is injected (see below), and frames are fed synchronously.

### Testability Hooks Required in the App Design

These are small additions to the design, needed so the tests above are possible. They are kept cheap on the RT path.

- `processFrame(const void *src, const timespec &writetime, uint64_t cnt0)`: the body of the state machine. `processImage()` becomes a thin wrapper that reads `writetime`/`cnt0` from the stream metadata and calls it. Tests call `processFrame` directly with synthetic buffers and timestamps.
- An injectable clock: a `timespec (*m_clock)()` member, defaulting to a `clock_gettime(CLOCK_REALTIME)` wrapper and used by the busy-wait and the achieved-delay recording. The test harness installs a fake clock that advances a fixed step per call, so busy-waits end deterministically. It also installs a stub that *records* the requested wait without spinning.
- The resolved DM stream name is stored in a protected member (`m_dmStreamName`). The harness overwrites it after `loadConfig`, pointing it at the test DM stream, while `dmStreamName()` is still tested separately.
- `runMeasurement()`: the measurement-thread body, split out as a callable member. Tests can run it on a test-owned thread while a "fake camera" thread feeds frames.
- The test harness is a subclass `dmResponse_test : public dmResponse` using `MagAOXApp<false>`-compatible construction, as in `adcTracker_test.cpp`. It exposes protected state and wraps the `newCallBack_m_indiP_*` handlers with constructed `pcf::IndiProperty` objects.

### A. Pure Helper Unit Tests (`[helpers]`)

| # | TEST_CASE | Checks |
|---|---|---|
| A1 | `parseIntList` | `"5,6,7"` → {5,6,7}; whitespace tolerated (`" 5 , 6 "`); single value; empty string → empty/error per design; bad token (`"5,a"`), trailing comma and overflow → error |
| A2 | `dmStreamName` | (0,7) → `dm00disp07`; (1,7) → `dm01disp07`; (2,3) → `dm02disp03`; NN < 0 or > 2 → error; MM < 0 or > 99 → error |
| A3 | `validatePokes` | equal-length in-bounds lists pass; length mismatch fails; empty lists fail; x or y negative / ≥ dimension fails (edges at 0 and dim−1 pass) |
| A4 | `resolveSpan` | `delaySpan > 0` → returned unchanged, regardless of fps; `delaySpan ≤ 0` with fps = 1000 → 1000 µs; `delaySpan ≤ 0` with fps unknown (≤ 0) → error |
| A5 | `delayGrid` | K = 1 → {0}; K = 4, span = 1000 → {0, 250, 500, 750} (endpoint excluded); non-integer spacing kept exact (span = 1000, K = 3); K < 1 → error |
| A6 | trial-count validation | M = 2, 20 pass; M = 0, 1, 21 fail (must be even and ≥ 2) |
| A7 | `runDirName` | fixed `timespec` → exact expected `YYYY-MM-DDTHHMMSS`; run with `TZ` set to a non-UTC zone (e.g. `America/Phoenix`) and confirm the output is still UTC; midnight/new-year rollover |
| A8 | `cubeFileName` | 0 → `dmresp_delay_00000us.fits`; 250 → `..._00250us.fits`; 333.3 rounds to `..._00333us.fits`; values ≥ 100000 µs keep all digits |
| A9 | `achievedDelay` | same-second and cross-second `timespec` pairs → correct µs; poke before reference → negative value reported (not clamped) |
| A10 | `differenceCube` | synthetic `+` frames = bias + R, `−` frames = bias − R → result equals R exactly; a large constant bias cancels; normalization by M checked for M = 2 and M = 20; mismatched cube sizes → error |

### B. App-Level Tests (`[config]`, `[indi]`, `[statemachine]`, `[run]`)

**Configuration (`[config]`)**, using `mx::app::writeConfigFile` as in `adcTracker_test.cpp`:

| # | Checks |
|---|---|
| B1 | Defaults with an empty config: `dm.index = 0`, `dm.channel = 7` → `dm00disp07`; `nDelays = 10`; `delaySpan = 0`; `nFrames = 10`; `nTrials = 20`; `settle`, `trialTimeout`, `maxRetries`; `output.baseDir = /home/xsup/dm_response` |
| B2 | Every key overridden from the file is reflected in the members (e.g. `dm.index = 1`, `dm.channel = 3` → `dm01disp03`) |
| B3 | Invalid config values (mismatched `poke.x`/`poke.y`, odd `nTrials`, `dm.index = 5`) make `loadConfig` fail, or are rejected at start, per the chosen rule and logged |

**INDI (`[indi]`)**, calling the `newCallBack_m_indiP_*` handlers with constructed properties:

| # | Checks |
|---|---|
| B4 | Each number property (`dm_index`, `dm_channel`, `poke_amp`, `nDelays`, `delaySpan`, `nFrames`, `nTrials`, `settle`): a valid `target` updates the member |
| B5 | `poke_x`/`poke_y` text: a valid list updates the member; an unparsable list is rejected and the member is unchanged |
| B6 | While the state is running, every tunable callback is rejected (member unchanged, warning logged) |
| B7 | Wrong device/property name is rejected by `INDI_VALIDATE_CALLBACK_PROPS` |
| B8 | `start` with each invalid condition (poke length mismatch, out of bounds, `amp = 0`, odd M, fps unknown with the default span, unwritable `baseDir`) → start refused, state stays idle, DM stream untouched (all zeros) |
| B9 | `stop` while idle is harmless; the `wfsFps` set-callback updates the stored fps and ignores a property without `current` |

**Per-frame state machine (`[statemachine]`)**, calling `processFrame` directly with a fake clock and a test DM stream:

| # | Checks |
|---|---|
| B10 | Same-frame poke, `+` sign: arm with `d = 250 µs`, `T = 1000 µs`, feed the trigger frame. The busy-wait target is `writetime + 250 µs`. The DM stream holds `+amp` at each `(x, y)` and 0 elsewhere. The achieved delay is recorded. The next N frames are added into the `+` accumulator at indices 0..N−1. The "trial done" semaphore is posted; the state returns to idle |
| B11 | `−` sign: same as B10, but the DM holds `−amp` and frames go only into the `−` accumulator |
| B12 | `d = 0` / target already past on wake: the poke is immediate, no wait, and the achieved delay is ≥ 0 and recorded |
| B13 | Deferred poke (`d = 1500 µs > T`): no DM write on the trigger frame; the write happens on the following frame; capture index 0 is the first frame *after* the poke |
| B14 | `cnt0` gap during capture: the trial is marked invalid, the accumulator contribution is discarded, the semaphore is still posted |
| B15 | Frames arriving while idle change nothing (accumulators and DM untouched) |
| B16 | Stop requested during `WAIT_POKE` and during `CAPTURING`: the DM is zeroed, the state is idle and no partial trial is counted |

**Full synthetic run (`[run]`)**, running `runMeasurement()` with a fake-camera thread:

The fake camera watches the test DM stream and produces frames `= bias + gain · DM(x, y)` mapped onto the camera grid, plus optional noise. It posts them on the test camera stream at a fixed simulated period with contiguous `cnt0`. This gives an exact expected answer.

| # | Checks |
|---|---|
| B17 | Nominal run, K = 2, N = 3, M = 4, known amp: <ul><li>output directory `baseDir/<UTC stamp>/` created</li><li>exactly 2 FITS files with the expected names</li><li>each cube is `[nx, ny, 3]`</li><li>pixel values equal `gain · amp` at the mapped poke location, and ≈ 0 elsewhere (bias cancelled)</li><li>header keywords all present and correct (`DELAYUS`, `NDELAYS`, `DLYSPAN`, `NFRAMES`, `NTRIALS`, `NINVALID = 0`, `POKEAMP`, `POKEX`/`POKEY`, `DMSTREAM`, `WFSSHMIM`, `WFSFPS`, `DATE-OBS`, delay stats)</li></ul> Readback uses `mx::fits::fitsFile` |
| B18 | Order: the fake camera logs the sign of each poke; confirm M/2 `+` trials then M/2 `−` trials per delay, and that the DM is zeroed between trials |
| B19 | One injected frame gap → that trial is retried, the cube is still correct, `NINVALID = 1`, and the + and − halves each have M/2 valid trials |
| B20 | Gaps exceeding `maxRetries` → the run aborts with the error state; cubes for already finished delays remain and no partial cube is written; DM zeroed |
| B21 | Fake camera stops producing frames → the trial timeout fires, the run aborts cleanly, DM zeroed, the measurement thread returns |
| B22 | Stop via INDI mid-run → returns promptly, DM zeroed, state idle; only completed cubes on disk |
| B23 | Shutdown flag set mid-run → same as B22 (exercises the `m_shutdown` paths) |

### C. Hardware Acceptance (manual, RTC, AO loop open)

Run by an operator with the app on the RTC, the woofer loop open and channel `dm00disp07` confirmed unused. Record the date, the operator, the parameters and the outcome for each step in this file.

| # | Procedure | Pass criterion |
|---|---|---|
| C1 | Start the app; check the INDI properties in cursesINDI; set `poke_x`/`poke_y` to one central actuator, a small `poke_amp`, `nDelays = 1`, N = 5, M = 2; press `start` | Run completes; one cube written in `/home/xsup/dm_response/<UTC stamp>/`; `dm00disp07` is all zeros afterwards |
| C2 | Inspect that cube | Poke signal visible at the expected WFS location; later frames plateau (steady state reached) |
| C3 | Timing: default grid (K = 10, span = one frame period), M = 20 | `NINVALID = 0` at nominal fps; for every delay, `DLYMEAN` is within a tolerance of the requested value and `DLYSTD` is below a jitter limit (both **TBD on the bench**, proposed starting point ±5 µs / 5 µs) |
| C4 | Repeat C3 immediately | Cubes agree within noise |
| C5 | Press `stop` mid-run, then kill the app mid-run (SIGTERM) | Both times `dm00disp07` returns to all zeros; only completed cubes on disk; app restarts cleanly |
| C6 | Explicit span: `delaySpan = 2·T` | Delay grid spans two frames; no invalid trials; deferred-poke cubes are consistent with the default-span ones where they overlap |
| C7 | (Optional, only with authorization for that DM) `dm_index = 1` or `2`, `dm_channel = 7` | Same checks as C1 on the tweeter / NCPC |

## Implementation Steps and Commits

Branch: `ktwitchell/dm-response` (already checked out; rules 12 and 17). **Do not start until the plan is approved.**

1. **Functional commit:** add `apps/dmResponse/{dmResponse.hpp, dmResponse.cpp, Makefile}`, register it in the top-level `Makefile`, add the tests and `tests/tests.list` entry, and update this plan file with any decisions made during implementation.
2. **Documentation commit:** finish the Doxygen pass (rule 15), add an app doc page / example config if that pattern applies, and update `AGENTS.md` if any new standing rule comes out of review (rule 16).
3. **Formatting commit:** `clang-format` on all touched files, if it changes anything beyond steps 1–2.
4. Build and run the Catch2 tests (Test Plan layers A and B), then run `make coverage` in `tests/` and confirm the 100% statement/function target for `dmResponse.hpp`. Record any `LCOV_EXCL` exclusions in this file.
5. Hardware acceptance (Test Plan layer C) on the RTC with the loop open, recording the results in this file.

## Remaining Minor Assumptions (flag if wrong)

- Default K = 10, N = 10, M = 20. These are placeholders to tune on the bench.
- Invalid trials are retried rather than just dropped, so the + and − halves stay balanced.
- The DM index (`NN`) is restricted to 0–2 to match the three MagAO-X DMs.

## Review Questions and Answers (resolved, incorporated above)

1. **Actuator index convention.** The prompt says "index or indices." `dmPokeWFS` uses `(x, y)` pairs. The proposal uses a linear index into the DM channel image, in the channel's native (column-major, `mx::improc`) order. Should it be `(x, y)` pairs instead, or ALPAO's 0–96 actuator numbering (which would need the actuator map)?
2. **Which woofer channel.** It needs a dedicated `dmNNdispMM` channel that nothing else writes. Please confirm the woofer shmim prefix and a free channel number.
3. **Poke amplitude and sign.** Amplitude isn't in the prompt; the proposal adds it as a parameter. Should each trial also do a `-amp` poke to cancel static/non-linear terms, like `dmPokeWFS` does, or is a single-sign step enough?
4. **Baseline handling.** Should cubes be saved raw (proposed, with a separate baseline file), or with the pre-poke frame subtracted? Is dark subtraction needed (would need a `wfsdark` shmimMonitor like `dmPokeWFS`)?
5. **Standard delay set.** The prompt gives 0/50/100/200 µs. A useful alternative is "K evenly spaced delays spanning one frame period," computed from the current fps. Want that as an option (e.g. `poke.nAutoDelays`)?
6. **Time reference.** The delay is measured from the camera's `writetime` stamp. If the intent was "from when the frame exposure ended at the detector," a fixed camera-latency offset would need to be characterized separately.
7. **Directory timestamp.** UTC (proposed) or local time? Format `YYYY-MM-DDTHHMMSS` OK?

### Answers:
1. the (x,y) convention in `dmPokeWFS` should be used.
2. The default configuration should be NN = 00, MM = 07 to write to woofer channel seven. However, it should be configurable so that NN can be changed to 01 for the tweeter DM, or 02 for the NCPC DM, so that the same test can be run for any of the three DMs in the system. MM should default to 7 but also be a configurable paramter.
3. amplitude should be added as a parameter, the proposal implementation is good. There should be a `-amp` poke to cancel out the bias. Rather than alternating + and - pokes, the test should be run with M/2 trials using a positive poke, and then M/2 trials using a negative poke.
4. Because we will be using the + and - pokes, so no dark subtraction should be necessary. Only the final average cubes should be saved.
5. We should switch to evenly spaced delay intervals. A default configuration should be K evenly spaced delays spanning one frame period, but there should be the option to change the configuration from "one frame period" to any desired amount of time.
6. No need to take this into account at this stage.
7. UTC time is preferred. That format is okay. 
