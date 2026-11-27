# counterOffload telemetry

## Prompt

> we want to write telemetry at a low rate, similar to other apps in the MagAOX software. It should also write
> immediately when the app is enabled or disabled rather than waiting for its next refresh.

Branch: `ktwitchell/counter-offload`.  No code until the user says "implement".

## Findings

- counterOffload (`apps/counterOffload/counterOffload.hpp`) has no telemetry today.  It writes `text_log` /
  `software_error` entries, INDI-only status (`latency`, counters, parameters) and calibrate-mode CSVs.
- The model to copy is t2wOffloader (`apps/t2wOffloader/t2wOffloader.hpp`):
  - derives from `dev::telemeter<t2wOffloader>`, uses `TELEMETER_SETUP_CONFIG`, `TELEMETER_LOAD_CONFIG`,
    `TELEMETER_APP_STARTUP`, `TELEMETER_APP_LOGIC`, `TELEMETER_APP_SHUTDOWN` (AGENTS.md rule 18);
  - `checkRecordTimes()` returns `telemeterT::checkRecordTimes( telem_offloading() )`, which forces a record when
    `telemeter.maxInterval` (default 10 s, `libMagAOX/app/dev/telemeter.hpp:83`) has passed;
  - `recordTelem( const telem_offloading * )` calls `recordOffloading( true )`;
  - `recordOffloading( bool force )` writes only when a value changed or `force` is set;
  - on a setting change it brackets: `recordOffloading( true )`, change the value, `recordOffloading()`, so the
    log shows the old value up to the change and the new one from it.
- A new record type needs (pattern of `telem_offloading` / `telem_dmspeck`):
  - `libMagAOX/logger/types/schemas/telem_counteroffload.fbs`;
  - `libMagAOX/logger/types/telem_counteroffload.hpp` (messageT, verify, msgString, member getters, getAccessor);
  - `libMagAOX/logger/logCodes.dat` entry; `flatlogcodes` then generates `logTypes.hpp`, the accessor switch and
    the binary schema (`libMagAOX/Makefile:191-194`);
  - `timespec telem_counteroffload::lastRecord = { 0, 0 };` in `libMagAOX/logger/types/telem.cpp`;
  - a `case eventCodes::TELEM_COUNTEROFFLOAD:` verify case in `libMagAOX/logger/logMeta.cpp` (~line 250);
  - the header in the `libMagAOX/Makefile` header list (~line 100);
  - a section in `libMagAOX/logger/tests/logTypes_Accessor_test.cpp`.
- Event codes: the DM block is 20890 dmspeck, 20891 dmsynco, 20892 dmtimings; 20893 is free.
- Enable/disable goes through one place, `setEnabled()` (`counterOffload.hpp:2071`), called only from the
  `enable` INDI callback (`:4318`), i.e. on the INDI thread, while `checkRecordTimes()` runs on the appLogic
  thread.  The change-detection state therefore needs a mutex.

## Record: `telem_counteroffload`

| Field | Type | Source |
|---|---|---|
| `enabled` | bool | `m_enabled` |
| `calibrating` | bool | `m_calibRunning` |
| `gain` | float | `m_gain` |
| `leak` | float | `m_leak` (per published slot) |
| `ramp_start_us` | float | `m_rampStartUs` |
| `ramp_duration_us` | float | `m_rampDurationUs` |
| `time_offset_us` | float | `m_timeOffsetUs` |
| `max_command` | float | `m_maxCommand` |
| `slot_us` | float | `m_slotNs / 1000`, -1 until measured (shows the synco rate, e.g. 125 or 167 us) |
| `events` | uint64 | `m_events` |
| `published` | uint64 | `m_published` |
| `failed` | uint64 | `m_failed` |
| `missed` | uint64 | `m_missed` |
| `clipped` | uint64 | `m_clipped` |
| `overflows` | uint64 | `m_overflows` |
| `pipeline_us` | float | latency window mean, -1 if no data |
| `pipeline_rms_us` | float | latency window RMS, -1 if no data |
| `compute_max_us` | float | latency window max compute, -1 if no data |

Counters are cumulative since startup, so rates come from differences between records.  getAccessor exposes every
field (`metaTypes::State` for settings and `enabled`, `Telem` for counters and latency), so `logdump`/lookyloo and
FITS header keywords can use them.

## Behaviour

- **Low rate:** the standard telemeter cadence, a forced record every `telemeter.maxInterval` (default 10 s, set in
  the instance conf if a different rate is wanted).  No change to the producer thread; nothing is written per slot.
- **Enable/disable immediately:** `setEnabled()` keeps its logic.  The `enable` callback brackets it the
  t2wOffloader way: `recordCounterOffload( true )` before the change and `recordCounterOffload( true )` after it,
  so the record lands at the toggle, not at the next 10 s refresh.
- **Settings (Q1):** the same bracketing on `gain`, `leak`, `rampStartUs`, `rampDurationUs`,
  `timeOffsetUs` changes, and on calibration start/end (which changes the parameters from its own thread).
- `recordCounterOffload( bool force )` takes a new `m_telemMutex`, builds the record, and writes it if `force` or
  any setting or `enabled` differs from the last record.  Counters and latency alone do not trigger a record, so
  the cadence stays low.
- `recordTelem( const telem_counteroffload * )` calls `recordCounterOffload( true )`.
- Shutdown: `TELEMETER_APP_SHUTDOWN` with the existing shutdown; a final forced record before it so the last
  counters are kept.

## Questions

- **Q1 (answered 2026-11-25: yes):** also record immediately on parameter changes (gain, leak, ramp start and
  duration, time offset) and on calibration start/end.
- **Q2 (answered 2026-11-25: 10 s is fine):** keep the telemeter default `maxInterval` of 10 s; no conf change
  needed (the example conf may show the key commented out).

## Tests

- **logger:** `logTypes_Accessor_test.cpp` section for `telem_counteroffload` (each member's accessor; build a
  message, `verify`, read back each field, `msgString` not empty).
- **app (`apps/counterOffload/tests/counterOffload_test.cpp`):** with the existing test harness, a log-capture
  check that toggling `enable` produces a `telem_counteroffload` record with the new state at once, and that a
  second unchanged `recordCounterOffload()` writes nothing; skip if the harness cannot capture log entries, and
  test the change-detection helper directly instead.
- **RTC (user):** rebuild libMagAOX (`flatlogcodes` regenerates types), `make -C apps/counterOffload`, run the
  tests (clean `Makefile.one` first), clang-format, install.  Bench: the app's telemetry file (read with `logdump`)
  shows a record every ~10 s and one at each toggle.

## Files

- New: `libMagAOX/logger/types/schemas/telem_counteroffload.fbs`, `libMagAOX/logger/types/telem_counteroffload.hpp`.
- Edit: `libMagAOX/logger/logCodes.dat`, `libMagAOX/logger/types/telem.cpp`, `libMagAOX/logger/logMeta.cpp`,
  `libMagAOX/Makefile`, `libMagAOX/logger/tests/logTypes_Accessor_test.cpp`,
  `apps/counterOffload/counterOffload.hpp`, `apps/counterOffload/tests/counterOffload_test.cpp`,
  `apps/counterOffload/dmtweeter_counteroffload.conf` (`[telemeter]` example), `apps/counterOffload/doc/*`.

## Commits

1. Functional: record type + registration + app + tests + this plan.
2. Docs: app doc page (telemetry section).
3. Format (user, on the RTC).

## Follow-up and edge cases

- Adding a log type changes the generated log headers, so every app relinks against the new libMagAOX; install
  libMagAOX together with counterOffload.
- `logdump` on another machine needs the updated libMagAOX to decode the new code.
- A crash between records loses at most `maxInterval` of counters; the toggles are always recorded.

## Implementation (2026-11-25)

- `telem_counteroffload` (code 20893): schema, header, `logCodes.dat`, `telem.cpp`, `logMeta.cpp` verify case,
  `libMagAOX/Makefile` header list, accessor-test section.  The generated serialization tests pick it up from the
  schema; the `messageT` arguments follow the schema order.
- counterOffload derives from `dev::telemeter<counterOffload>` with the `TELEMETER_*` macros.
  - `telemSettings()` snapshots the trigger fields (`telemSettingsT`, compared with a defaulted `operator==`).
  - `recordCounterOffload( force )` takes `m_telemMutex`, then `m_stateMutex` for the snapshot and counters, and
    writes if forced, if nothing has been written yet, or if the trigger fields changed.  Never called with
    `m_stateMutex` held.
  - Bracketing (forced record before, change, record after) in `setEnabled()`, `setNumber()` (gain, leak,
    rampStartUs, rampDurationUs, only when the value changes), `setCalibParams()` (only when the parameters change),
    the calibration leak swap and restore, calibration start (INDI callback) and both calibration ends.
  - A forced record at shutdown, after the threads join, keeps the final counters.
  - `timeOffsetUs` and `maxCommand` are config-only, so they change only at startup; they are in the record.
- The example conf shows `[telemeter] #maxInterval=10`.
- Unit test "counterOffload writes telemetry at once on changes" checks `telem_counteroffload::lastRecord`: first
  record, no record for unchanged settings or counters only, forced periodic record, enable/disable, each setting,
  calibration start, `setCalibParams` changed/unchanged.

## Follow-up: frame-grabber style timing (`fg_timing`, `telem_fgtimings`)

Prompt: "camwfs and a variety of other real-time apps have a telemetry (and INDI) parameter called fg_timing. Does
counterOffload measure the equivalent of that?" then "let's add this to the plan, using the existing
telem_fgtimings."  Not implemented; waits for "implement".

### Findings

- `dev::frameGrabber` publishes the read-only INDI property `fg_timing` (`libMagAOX/app/dev/frameGrabber.hpp:759`,
  updated at `:1449`): `acq_fps`, `acq_min`, `acq_max`, `acq_jitter`, `write_fps`, `write_min`, `write_max`,
  `write_jitter`, `delta_aw`, `delta_aw_jitter`.  Intervals and delays are in seconds; the fps values are 1/mean
  interval; jitter is the standard deviation.
- `telem_fgtimings` (`libMagAOX/logger/types/schemas/telem_fgtimings.fbs`) holds `atime`, `atime_jitter`, `wtime`,
  `wtime_jitter`, `wmatime`, `wmatime_jitter`: the mean and standard deviation of the acquisition interval, the
  write interval and the acquisition-to-write delay, in seconds.  frameGrabber writes it from `recordFGTimings()`
  (`:1654`) whenever the values change; the DM controllers write it too (`dev/dm.hpp:1624`).  The type is already
  registered, so no logger change is needed.
- `MagAOX::timing::Accumulator` (`libMagAOX/common/timingStats.hpp`) is the shared Welford accumulator giving
  count, mean, stddev, min and max in seconds.  Reuse it.
- counterOffload today has only part of this: `slotUs` (slot *target* spacing, not actual publication times) and
  `pipelineUs` (handoff + compute + write, which leaves out the wait for a slot).

### Mapping

| `fg_timing` / `telem_fgtimings` | counterOffload quantity | Where sampled |
|---|---|---|
| acquisition interval (`acq_*`, `atime`) | time between successive woofer post stamps (input frame `atime`, CLOCK_ISIO) | input monitor, `processImage()` |
| write interval (`write_*`, `wtime`) | time between successive successful publications to `dm01disp08_aux`, CLOCK_MONOTONIC after `syncopationPublish()` returns | producer thread, after publish |
| acquisition to write (`delta_aw`, `wmatime`) | woofer post to the publication of the first slot carrying that offload: handoff + slot wait + compute + write | `recordLatency()`, once per offload with a woofer stamp |

- The write interval counts only consecutive publications: the first publication after the producer was idle (app
  disabled and contribution zero) or after a failed publish starts a new series, so idle gaps are not counted.
- At 8 kHz the write intervals are mostly 125 us, with the cycle-to-cycle gap set by the main loop and synco; at
  6 kHz about 167 us.  The write jitter therefore shows synco and main-loop timing as seen by this producer.
- The delay includes the slot wait, so unlike `pipelineUs` it is the full woofer-to-counter delay; its jitter is
  dominated by the phase of the offload relative to the slot grid (0 to one slot).

### Design

- Three `timing::Accumulator`s (acquisition interval, write interval, delay), updated under `m_stateMutex` where
  the producer and input paths already hold it (`processImage()` ramp insertion, `afterPublish()`/
  `recordLatency()`).  Cost per slot: one subtraction and one Welford update.
- `appLogic` (about 1 Hz) swaps the accumulators out under `m_stateMutex`, keeps the resulting 1 s window
  statistics, and from them:
  - updates `fg_timing` with the frameGrabber element names and units, so existing GUIs and scripts read it the
    same way (values with no data are 0, as in frameGrabber);
  - calls `recordFGTimings()`, which, like frameGrabber's, writes
    `telem<telem_fgtimings>( { atime, atime_jitter, wtime, wtime_jitter, wmatime, wmatime_jitter } )` when the
    values differ from the last record, i.e. about every second while offloads flow (Q3).  Only while OPERATING.
- `checkRecordTimes( telem_counteroffload(), telem_fgtimings() )`; `recordTelem( const telem_fgtimings * )` forces a
  `recordFGTimings( true )` with the latest window, so a record still appears every `telemeter.maxInterval` when
  the values are unchanged (e.g. idle).
- `telem_counteroffload` keeps its 10 s cadence and immediate records; only `telem_fgtimings` posts at 1 s.
- A forced `telem_fgtimings` record at shutdown alongside the final `telem_counteroffload` record.

### Questions

- **Q3 (answered 2026-11-25):** for `telem_fgtimings` only, post about every 1 s like the camera apps (on
  change each appLogic cycle, 1 s windows).  `telem_counteroffload` stays at 10 s plus immediate records.
- **Q4 (answered 2026-11-25):** the acquisition interval uses the woofer post stamps (input frame `atime`).
  Frames without a woofer stamp (older t2wOffloader) are not sampled for the acquisition interval or the delay.

### Tests

- Unit: feed known publication times, offload stamps and delays through the sampling functions and check count,
  mean, stddev, min and max; the first publication after idle or a failure is not counted; unstamped frames are
  not sampled; the window resets when appLogic takes it; `recordFGTimings()` writes only on a change and
  `recordTelem( const telem_fgtimings * )` always writes (checked with `telem_fgtimings::lastRecord`).
- Bench (user): `fg_timing` shows `write_fps` about 8000 (or 6000), `acq_fps` about the offload rate (250), and
  `delta_aw` about the latency seen in `latency`; `logdump` shows `telem_fgtimings` about every 1 s and
  `telem_counteroffload` every 10 s.

### Files

- `apps/counterOffload/counterOffload.hpp`, `apps/counterOffload/tests/counterOffload_test.cpp`,
  `apps/counterOffload/doc/counterOffload.md`.  No libMagAOX change.

### Implementation (2026-11-25)

- `MagAOX::timing::Accumulator`s `m_fgAcq`, `m_fgWrite`, `m_fgDelay` under `m_stateMutex`:
  - `sampleOffloadPost()` from `processImage()` for frames with a valid woofer stamp (post = receipt CLOCK_ISIO −
    handoff), whether or not the app is enabled; the lock is now taken before the enabled check;
  - `samplePublication()` after each `syncopationPublish()`; `endPublicationSeries()` when the producer idles or a
    slot has nothing to publish, and on a failed publication;
  - the delay in `recordLatency()`: `handoffNs + writeEndNs − wakeNs`, stamped offloads only.
- `takeFGWindow()` in appLogic (and at shutdown) moves the window into `m_fgWindow` and resets the accumulators.
- `fg_timing` INDI property with frameGrabber's elements and conventions (`fgTimingElements()`,
  `fgTimingValues()`), updated each appLogic cycle.
- `recordFGTimings( force )` writes `telem_fgtimings` from `fgTelemValues()` when the values change (each appLogic
  cycle while OPERATING) or when forced; `recordTelem( const telem_fgtimings * )` forces it at
  `telemeter.maxInterval`; forced at shutdown.
- Unit test "counterOffload measures fg_timing".
