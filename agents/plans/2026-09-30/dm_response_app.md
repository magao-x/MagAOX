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

Prompt (condensed): Create a C++ MagAO-X app to measure the dynamic response of the woofer DM at a temporal resolution finer than the camWFS frame period. For each delay value, trigger off the camWFS frame-ready semaphore, wait a short delay, poke the requested actuator(s), capture N camWFS frames, repeat M times, average the trials into one N-frame cube, and save one FITS cube per delay under `~xsup/dm_response/<date-time>/`. All tunables are INDI properties usable from cursesINDI.

## Measurement Principle

camWFS samples at a fixed period `T = 1/fps`. Frame `k` after a poke shows the DM state integrated over the exposure that ends roughly at `k*T - delay` after the poke was issued. Stepping the poke delay `d` across `[0, T)` samples the step response on a finer grid, spaced by the delay step instead of by `T`. Put the N-frame averaged cubes from all delays in order of `k*T - d` and you get a step-response time series that is effectively oversampled.

This only works if the poke time relative to the frame is **precise and measured**. The design below is shaped mostly by that requirement.

## App Overview

- Name: `dmResponse` (directory `apps/dmResponse/`), following the existing `dmPoke*` naming.
- Pattern (AGENTS.md rule 22): header-only app. `dmResponse.hpp` holds the class declaration plus the out-of-class inline definitions. `dmResponse.cpp` holds only `main`.
- Base classes:
  - `MagAOXApp<true>`
  - `dev::shmimMonitor<dmResponse>` on the camWFS stream (config section `wfscam`, default shmim `camwfs`). Its high-priority RT thread is where the time-critical trigger → delay → poke → capture sequence runs.
- Not a telemeter. The output product is FITS files, and the run status is exposed through INDI and logged.
- Build registration: add `dmResponse` to `apps_rtc` (camWFS and the woofer are both on the RTC) and to `all_buildable_apps` in the top-level `Makefile`. Add the test to `tests/tests.list`.
- The `dev::dmPokeWFS` base class is deliberately **not** reused. It pokes first, then sleeps `m_dmSleep`, then flushes and waits for whole frames. Its timing is deliberately *asynchronous* to the camera, which is the opposite of what we need here. Only its conventions are borrowed: the DM `milkImage` handling, the INDI property style, the `wfsFps` set-property and the thread handling.

## Threading and Timing Design

Two threads, with a clear split between real-time work and bookkeeping:

1. **shmimMonitor thread** (RT priority, configurable `wfscam.threadPrio` / cpuset). Runs a small state machine inside `processImage()` on every camWFS frame:
   - `IDLE`: do nothing.
   - `ARMED`: this frame is the trigger. Take the reference time from the stream metadata (`md[0].writetime`, i.e. when the camera process posted the frame) and compute `t_poke = writetime + delay`. **Busy-wait** on `clock_gettime(CLOCK_REALTIME)` until `t_poke`. `microSleep` has tens of µs of scheduler jitter, which is the same size as the delays being measured, so it can't be used here. Then write the pre-built poke command to the DM channel (a single memcpy plus semaphore post via `milkImage` assignment). Record the actual poke time and the achieved delay (`t_written - writetime`) and the trigger frame's `cnt0`. Optionally copy the trigger (pre-poke) frame as a baseline. Move to `CAPTURING`.
   - `CAPTURING`: add each frame into the per-delay accumulator at index `n` (`n = 0..N-1`). Check that `cnt0` is contiguous; a gap marks the trial as invalid. After N frames, post a "trial done" semaphore and move to `IDLE`.
   - The busy-wait is bounded by `delay` (< one frame period), so the RT thread always gets back in time to read the next frame.
2. **Measurement thread** (normal priority, modeled on the `dmPokeWFS` `wfsThread`). Orchestrates the run:
   - Validate the parameters, create the output directory and build the poke image.
   - For each delay `d`, for each trial `m = 1..M`: zero the DM channel, wait the settle time, set `ARMED`, wait on the "trial done" semaphore (with timeout), then zero the DM again.
   - After M valid trials: divide the accumulator by M and write the FITS cube. Disk I/O stays entirely off the RT thread.
   - Supports stop/shutdown between and inside trials. The DM channel is always zeroed on exit, error or stop.

Shared state between the threads (`m_state`, delay, trial-valid flag) uses `std::atomic`. The accumulator is touched only by the RT thread while `CAPTURING` and only by the measurement thread otherwise, with a `{ //mutex scope` guard around the handoff (AGENTS.md rule 14).

**Timing caveat to document.** The measured delay is referenced to the moment the camera *posted* the frame (end of readout + transfer), which is what the prompt asks for. The DM itself adds fixed latency on top: the `dm` base class combines channels, then there is the ALPAO driver/PCIe write and the mechanical response. That offset is constant, so it shows up as a shift in the recovered time series, not as smearing. The `XWC_DMTIMINGS` instrumentation and `shmimDelta` (recent PR #400) can characterize the software part of it separately.

## Configuration

Config file / command line (loaded in `loadConfig`, all mirrored to INDI where tunable):

| Key | Type | Default | Meaning |
|---|---|---|---|
| `wfscam.shmimName` (from shmimMonitor) | string | `camwfs` | camWFS stream |
| `wfscam.camDevName` | string | = shmimName | INDI device used to read `fps` |
| `dm.channel` | string | *(required)* | woofer DM channel to poke, e.g. `dm00dispNN`. Must be a channel the AO loop does not write to |
| `poke.actuators` | vector<int> | *(required)* | actuator indices to poke |
| `poke.amp` | float | `0.0` | poke amplitude in DM command units (not in the prompt, but needed; 0 default is safe) |
| `poke.delays` | vector<float> | `0,50,100,200` µs | standard delay set; can be overridden per run |
| `poke.nFrames` | int | `10` | N, frames captured per trial |
| `poke.nTrials` | int | `20` | M, trials per delay |
| `poke.settle` | float | `0.05` s | wait after zeroing before the next trial arms |
| `poke.trialTimeout` | float | `2.0` s | max wait for a trial to complete |
| `output.baseDir` | string | `/home/xsup/dm_response` | root of the output tree |

## INDI Interface (cursesINDI)

Following `dmPokeWFS` conventions (`CREATE_REG_INDI_NEW_NUMBER*`, `current`/`target` elements, `indiTargetUpdate`). Every tunable parameter gets a property, and changes are rejected with a logged warning while a run is in progress:

- `actuators` (text, `current`/`target`): comma-separated indices, e.g. `45,46`. Text is used because INDI number vectors have a fixed length, and it is easy to edit in cursesINDI.
- `delays` (text, `current`/`target`): comma-separated µs values. Starts at the configured standard set so it rarely needs editing.
- `nFrames`, `nTrials` (number, int).
- `poke_amp` (number, float).
- `settle` (number, float, s).
- `start` (request switch), `stop` (request switch).
- `status` (RO): `state` (idle/running/error), `delay_index`, `delay_us`, `trial`, `n_invalid`, `last_delay_mean_us`, `last_delay_std_us`.
- `output` (RO text): directory of the current/last run.
- `wfsFps` (set-property on `camDevName.fps`), used for validation.

Validation on `start`: actuator indices are inside the channel dimensions, `nFrames`/`nTrials` ≥ 1, `amp` ≠ 0, and each delay satisfies `0 ≤ d < 1/fps` minus a margin. If fps is unknown, warn and continue; if a delay violates the limit, reject.

## Output

- Directory: `<output.baseDir>/<YYYY-MM-DDTHHMMSS>/`, created with `mkdir -p` at run start, using UTC for consistency with MagAO-X logs. The app runs as `xsup`, so permissions follow from that.
- One cube per delay: `dmresp_delay_<DDDD>us.fits`, float, shape `[nx, ny, N]`, the mean over the M valid trials.
- FITS header keywords per cube: `DELAYUS` (requested), `DLYMEAN`/`DLYSTD`/`DLYMIN`/`DLYMAX` (achieved, µs), `NFRAMES`, `NTRIALS`, `NVALID`, `POKEAMP`, `ACTUATRS`, `DMCHAN`, `WFSSHMIM`, `WFSFPS`, `DATE-OBS`.
- Optional (recommended): `baseline.fits`, the mean trigger (pre-poke) frame, so users can difference the cubes against the static state. Also `pokemap.fits`, the DM command that was applied.
- Writing uses `mx::fits::fitsFile` and `fitsHeader`, as in `dmSpeckle`/`dmMode`.

## Code Structure (dmResponse.hpp)

Keep the time-critical and the pure logic separate so the pure logic can be unit tested without hardware:

- Free/static helpers (pure, testable):
  - `parseIntList(const std::string &)` / `parseFloatList(const std::string &)`: comma list → vector, with error on bad token.
  - `validateActuators(indices, rows, cols)`.
  - `validateDelays(delays, fps, margin)`.
  - `runDirName(timespec)` → `YYYY-MM-DDTHHMMSS`.
  - `cubeFileName(delay)`.
  - `achievedDelay(writetime, t_poke)` → µs.
- Class sections, each documented per AGENTS.md rules 3–5, 9–10 and 13: "Configurable Parameters - Data" (protected) before "Configurable Parameters" (public accessors); MagAOXApp interface; shmimMonitor interface (`allocate`, `processImage`); measurement thread; INDI interface with `INDI_NEWCALLBACK_DECL`/`DEFN`; all definitions out of class.
- `m_` member prefix, `///` on every non-trivial member, inline `/**< [in] ... */` parameter docs, Doxygen `\file`/`\brief`/`\author` blocks, and `\defgroup dmResponse` / `dmResponse_files`.

## Tests

`apps/dmResponse/tests/dmResponse_test.cpp`, Catch2, following AGENTS.md rules 20–21:

- `namespace libXWCTest { namespace dmResponseTest { ... } }`, with `\defgroup dmResponse_unit_test` `\ingroup application_unit_test` and a Doxygen block on every `TEST_CASE`.
- Cases: list parsing (valid, whitespace, empty, bad token), actuator bounds validation, delay-vs-fps validation, directory/file name formatting, averaging arithmetic (sum of M fake frames / M), and the `processImage` state machine driven with synthetic frames through a small test subclass that stubs the DM write and the clock (hidden via `\cond`). That includes the `cnt0` gap → trial invalid path.
- `#ifdef DMRESPONSE_TEST_DOXYGEN_REF` blocks referencing the real members, wrapped in `// clang-format off/on`.

## Implementation Steps and Commits

Branch: `ktwitchell/dm-response` (already checked out; rules 12 and 17).

1. **Functional commit:** add `apps/dmResponse/{dmResponse.hpp, dmResponse.cpp, Makefile}`, register it in the top-level `Makefile`, add the tests and `tests/tests.list` entry, and update this plan file with any decisions made during implementation.
2. **Documentation commit:** finish the Doxygen pass (rule 15), add an app doc page / example config (`dmResponse.conf` example under the existing config docs if that pattern applies), and update `AGENTS.md` if any new standing rule comes out of review (rule 16).
3. **Formatting commit:** `clang-format` on all touched files, if it changes anything beyond steps 1–2.
4. Build the app and the test (`make -B -f Makefile.one t=../apps/dmResponse/tests/dmResponse_test.cpp`), and run it.
5. Test on hardware (manual, on the RTC, loop open): small amplitude, single actuator, `delay=0`. Confirm the achieved-delay stats in the FITS header, then sweep delays.

## Open Questions / Ambiguities (for review before implementation)

1. **Actuator index convention.** The prompt says "index or indices." `dmPokeWFS` uses `(x, y)` pairs. The proposal uses a linear index into the DM channel image, in the channel's native (column-major, `mx::improc`) order. Should it be `(x, y)` pairs instead, or ALPAO's 0–96 actuator numbering (which would need the actuator map)?
2. **Which woofer channel.** It needs a dedicated `dmNNdispMM` channel that nothing else writes. Please confirm the woofer shmim prefix and a free channel number.
3. **Poke amplitude and sign.** Amplitude isn't in the prompt; the proposal adds it as a parameter. Should each trial also do a `-amp` poke to cancel static/non-linear terms, like `dmPokeWFS` does, or is a single-sign step enough?
4. **Baseline handling.** Should cubes be saved raw (proposed, with a separate baseline file), or with the pre-poke frame subtracted? Is dark subtraction needed (would need a `wfsdark` shmimMonitor like `dmPokeWFS`)?
5. **Standard delay set.** The prompt gives 0/50/100/200 µs. A useful alternative is "K evenly spaced delays spanning one frame period," computed from the current fps. Want that as an option (e.g. `poke.nAutoDelays`)?
6. **Time reference.** The delay is measured from the camera's `writetime` stamp. If the intent was "from when the frame exposure ended at the detector," a fixed camera-latency offset would need to be characterized separately.
7. **Directory timestamp.** UTC (proposed) or local time? Format `YYYY-MM-DDTHHMMSS` OK?
