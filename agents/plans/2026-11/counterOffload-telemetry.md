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
