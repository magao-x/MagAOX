# mcp3208Ctrl: 8 kHz scans synchronized to a 2 kHz WFS

## Purpose

With `timingSource=semaphore`, mcp3208Ctrl read the ADC once per `camwfs_sync` semaphore, so acc000 ran at the WFS rate
(2 kHz). loPredCtrlAcc's ACC-assisted mode needs 4 samples per WFS frame (`acc_ratio = 4`), still phase-locked to the
WFS. ad7689Ctrl (`parkertjohnson1/ad7689Ctrl`) already solves this; this ports its finite per-WFS schedule.

## Design

- `libMagAOX/app/wfsScanSchedule.hpp`: ad7689Ctrl's `syncSchedule`, moved to libMagAOX (namespace `wfsScan`) so both
  ADC apps can share it. Each WFS arrival arms `max(1, floor(fps * period * 1.01))` scans spaced `period / count`
  apart: the highest integer multiple of the WFS rate not above `fps`. With an 8 kHz limit that is 4 scans (8 kHz) at
  a 2 kHz WFS, 16 (8 kHz) at 500 Hz, 11 (7.7 kHz) at 700 Hz, 4 (6.8 kHz) at 1.7 kHz. ad7689Ctrl rounded instead, which
  exceeds the limit (5 scans, 8.5 kHz, at 1.7 kHz). The 1% allowance (`c_ratioTolerance`) keeps 4 scans for a 2 kHz
  WFS measured slightly fast (ratio 3.9992 at 2000.4 Hz), so the rate can exceed the limit by at most 1%. A WFS faster
  than the limit still gets one scan per frame.
- The count is also lowered to the highest that the scan service fits (`ceil(period / (service + guard)) - 1`), instead
  of refusing the period as ad7689Ctrl does. On the Pi, reading 2 MCP3208 channels through lgpio averaged 106.5 us
  (208 us worst seen); with the 20 us `cadenceGuard_ns` that does not fit 125 us, and since only scans update the
  service estimate, every later period was refused (`schedule_failures` 106989, no output). It now runs 3 scans
  (6 kHz) at those numbers; `cadenceGuard_ns=10000` fits 4. If not even one scan fits, mcp3208Ctrl restarts the
  service estimate from the nominal SPI time so it is measured again. The first scan is at the requested phase modulo the scan
  interval, capped to leave the scan service plus `cadenceGuard_ns` before the next slot. Late slots are skipped, never
  published as a catch-up burst; nothing is scanned after the period without a new semaphore.
- `mcp3208Ctrl` semaphore mode:
  - `fps` is the scan rate limit: config `accel.fps` (default 8000, must be in (0, 10000], otherwise the default is
    kept with a warning) and the INDI `fps` target at runtime. In timer mode it remains the ADC rate. In semaphore
    mode an `fps.device` source only records the WFS rate (`m_wfs_fps`); it no longer overwrites the limit.
  - The WFS period estimate restarts from the latest interval after 3 consecutive intervals more than 10% from it
    (`syncTiming.cadence_resets`), so a WFS rate change re-fits the scan count within a few frames. Isolated outliers
    (jitter, a missed notification without producer metadata) only enter the EMA, as before.
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
  `max_lateness_us`, `coalesced_wakes`, `duplicate_references`, `schedule_failures`, `cadence_resets`.

## Verification

- `wfsScanSchedule_test` (ad7689Ctrl's schedule tests, ported, plus the integer-multiple limit at 2000, 500, 700,
  1700, 2000.4, 1000 and 9000 Hz): built and passed on the development Mac (6 cases).
- `mcp3208Ctrl_test`: compiles with GCC 15 against the real ImageStreamIO headers on the Mac (syntax check, no
  warnings in the changed files). It cannot link or run there (no ImageStreamIO library, macOS has no unnamed
  semaphores), so it must be run on Linux:
  `make -C tests -f Makefile.one t=../apps/mcp3208Ctrl/tests/mcp3208Ctrl_test && apps/mcp3208Ctrl/tests/mcp3208Ctrl_test`.
  New or rewritten cases: arming at the modelled phase, deadlines and late-slot skipping, the read-latency EMA and
  controller per WFS period, the service EMA, four scans from one real semaphore post, period start and missed-frame
  normalization, a 2 kHz to 700 Hz change (11 scans), `accel.fps` loading and range, the synchronized fps source,
  reconfig reset.
- loPredCtrlAcc's `acc_ratio` must equal `syncTiming.scans_per_wfs` (4 at 2 kHz); it is not adaptive.
- On the Pi: `syncTiming.scans_per_wfs=4`, `scheduled_fps=8000`, `skipped_scans` and `late_scans` near zero,
  `fg_timing` at 8 kHz, `timingDiag.mode_code=1` and `delay_lock=1`.

## Follow-ups

- The MCP3208 read (about 50 us per channel, mostly per-transfer overhead: 24 bits at 1.35 MHz is 18 us) limits the
  rate. Reading all channels in one spidev ioctl, or a faster SPI clock within the MCP3208's rating for its supply,
  would leave margin for 4 scans at 2 kHz.
- `MCP3208::read` sends the start byte `0x06` (start, single-ended, D2 = 0) and then `mode << 7 | channel << 4`, the
  MCP3008 layout, so D1 is always 1 and channels 0 and 1 both select CH2. Check whether acc000's two values are
  identical before relying on the second axis.

- ad7689Ctrl should include `libMagAOX/app/wfsScanSchedule.hpp` instead of its own copy when its branch is merged.
- ad7689Ctrl's DDSPC mode (continuous clock from a rolling fit of semaphore arrivals) is not ported. It tolerates
  semaphore jitter better, which matters for `camwfs_sync` delivered over TCP; port it if `delay_lock` or
  `late_scans` show the jitter.
