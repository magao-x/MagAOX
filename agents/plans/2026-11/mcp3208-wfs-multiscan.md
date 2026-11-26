# mcp3208Ctrl: 8 kHz scans synchronized to a 2 kHz WFS

## Purpose

With `timingSource=semaphore`, mcp3208Ctrl read the ADC once per `camwfs_sync` semaphore, so acc000 ran at the WFS rate
(2 kHz). loPredCtrlAcc's ACC-assisted mode needs 4 samples per WFS frame (`acc_ratio = 4`), still phase-locked to the
WFS. ad7689Ctrl (`parkertjohnson1/ad7689Ctrl`) already solves this; this ports its finite per-WFS schedule.

## Design

- `libMagAOX/app/wfsScanSchedule.hpp`: ad7689Ctrl's `syncSchedule`, unchanged except for namespace (`wfsScan`), moved
  to libMagAOX so both ADC apps can share it. Each WFS arrival arms `max(1, round(fps * period))` scans spaced
  `period / count` apart (4 at 125 us for 8000 / 2000 Hz). The first scan is at the requested phase modulo the scan
  interval, capped to leave the scan service plus `cadenceGuard_ns` before the next slot. Late slots are skipped, never
  published as a catch-up burst; nothing is scanned after the period without a new semaphore.
- `mcp3208Ctrl` semaphore mode:
  - `fps` (config `[fps]`, INDI `fps`) is the requested ADC rate, as in timer mode. Leave `fps.device` empty: it would
    set the ADC rate to the WFS rate (1 scan per frame).
  - `processSyncReference()` handles each semaphore: the existing producer-metadata tracking, then the existing WFS
    delay model (`updateTriggerTiming`), then arms the schedule at the controller-corrected delay. A repeated producer
    `cnt0` arms nothing (`duplicate_references`). Posts drained together count once (`coalesced_wakes`).
  - `acquireScan()` reads the due slot. The read-latency controller updates once per WFS period, at the slot nearest
    the modelled phase, from `scan start - semaphore arrival`; its correction now carries across references (an
    integrator, as in ad7689Ctrl), because the delay is now applied by waiting for the deadline.
  - `acquireSynchroAndCheckValid()` loops: a new semaphore takes precedence, else the due scan, else
    `sem_clockwait(CLOCK_MONOTONIC)` until the next deadline (polling fallback without glibc 2.30). One second without a
    semaphore keeps the old behaviour (stale check, return 1, one timer read). `EINTR` now continues the wait instead
    of falling back to a timer read.
  - Timer slack is set to 1 ns at `startAcquisition` (as accelSim), so deadline wakes are not 50 us late.
- Behaviour changes:
  - `delayBeforeRead()` (whose sleep had been commented out) and `waitOnSemaphore()` are removed: the schedule applies
    the delay. Scans now start at the modelled phase (34.4 us + k*125 us at 2 kHz with the default model) instead of
    immediately on the semaphore.
  - The semaphore-period EMA starts from the first measured interval instead of from 0, and divides an interval by the
    producer frame delta when frames were missed. Starting from 0 took hundreds of frames to converge, during which
    the scan count would have stepped 1, 2, 3, 4.
  - The first semaphore after start arms nothing (no period yet); scans start from the second.
  - `timingDiag.delay_phase_error_us` wraps at the scan interval once armed; the `delay_lock` thresholds are unchanged
    (fractions of the WFS period), so loPredCtrlAcc's lock check keeps its meaning.
  - `timingDiag.sync_frames_written` now counts 4 per WFS frame.
- New INDI `syncTiming`: `scans_per_wfs`, `scheduled_fps`, `scan_period_us`, `phase_us`, `skipped_scans`, `late_scans`,
  `max_lateness_us`, `coalesced_wakes`, `duplicate_references`, `schedule_failures`.

## Verification

- `wfsScanSchedule_test` (ad7689Ctrl's five schedule tests, ported): built and passed on the development Mac (9071
  assertions).
- `mcp3208Ctrl_test`: compiles with GCC 15 against the real ImageStreamIO headers on the Mac (syntax check, no
  warnings in the changed files). It cannot link or run there (no ImageStreamIO library, macOS has no unnamed
  semaphores), so it must be run on Linux:
  `make -C tests -f Makefile.one t=../apps/mcp3208Ctrl/tests/mcp3208Ctrl_test && apps/mcp3208Ctrl/tests/mcp3208Ctrl_test`.
  New or rewritten cases: arming at the modelled phase, deadlines and late-slot skipping, the read-latency EMA and
  controller per WFS period, the service EMA, four scans from one real semaphore post, period start and missed-frame
  normalization, reconfig reset.
- On the Pi: `syncTiming.scans_per_wfs=4`, `scheduled_fps=8000`, `skipped_scans` and `late_scans` near zero,
  `fg_timing` at 8 kHz, `timingDiag.mode_code=1` and `delay_lock=1`.

## Follow-ups

- ad7689Ctrl should include `libMagAOX/app/wfsScanSchedule.hpp` instead of its own copy when its branch is merged.
- ad7689Ctrl's DDSPC mode (continuous clock from a rolling fit of semaphore arrivals) is not ported. It tolerates
  semaphore jitter better, which matters for `camwfs_sync` delivered over TCP; port it if `delay_lock` or
  `late_scans` show the jitter.
