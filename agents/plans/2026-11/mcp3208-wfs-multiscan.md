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
  service estimate from the nominal SPI time so it is measured again.
- Late scans: a slot is skipped when it cannot finish before the next slot's deadline, so the implicit lateness
  tolerance was the interval minus the read-time estimate (about 20 to 35 us at 8 kHz with 90 to 106 us reads; less
  when slow reads raise the estimate). Any stall beyond that lost a sample (`skipped_scans`, about 13 per second on
  the Pi). `synchro.lateTolerance_ns` (INDI `scanLateTolerance`, us; default 50 us) now takes a slot at most that late
  even if it delays the next, which then catches up on the interval's slack (36 us per interval with 89 us reads).
  Stalls up to about 36 us plus the tolerance are absorbed as briefly uneven spacing instead of a lost sample. The
  half-interval minimum spacing now compares the actual start time (not the deadline) with the previous scan's start;
  comparing the deadline would have skipped the catching-up scans.
- Complete scans (`synchro.completeScans`, default true): the priority is an average rate of exactly the scans per
  frame times the WFS rate (8000 Hz at 2 kHz); spacing jitter is acceptable. A skipped scan is worse than a late one:
  acc000's `cnt0` is assigned per published frame, so loPredCtrlAcc (which places ACC samples by counter) sees no
  gap and its positional features shift by one sample for `accel_history` frames. In complete mode no slot is
  skipped: late slots are taken late (`lateTolerance_ns` applies only when false), and slots still unscanned at the
  next WFS reference are carried over and scanned at once, ahead of that period's, up to one period's count (the
  excess is skipped, so a persistent overload cannot build a backlog). A carried scan keeps the previous
  `delay_applied_us`, so `delay_lock` is unaffected, and does not update the read-latency controller.
  `syncTiming.carried_scans` counts them.
- Frames that arm nothing: a WFS frame whose notification was merged with the next (`coalesced_wakes`, likely when
  `camwfs_sync` arrives in bursts over TCP), missed (`sync_frames_dropped`), or arrived while the rolling clock was
  confirming a rate change, got no scans at all, so complete mode still averaged below 4 x the WFS rate. Each
  reference carries its frame ID, so the frames since the last armed reference are counted
  (`syncTiming.missed_frames`) and, in complete mode, their scans are owed into the backlog (same one-period cap).
- The 106.5 us came from sleeping to each deadline: in timer mode, which polls every microsecond, the same read takes
  about 89 us. A core woken from sleep (idle state, lower clock) reads slower. The loop now sleeps only until
  `synchro.spinLead_ns` (default 50 us) before each scan deadline and polls from there, checking the WFS semaphore
  too. At 8 kHz with 89 us reads that is polling nearly all the time, as timer mode does; the acc000 thread should
  have a core without other real-time threads. The first scan is at the requested phase modulo the scan
  interval, capped to leave the scan service plus `cadenceGuard_ns` before the next slot. Late slots are skipped, never
  published as a catch-up burst; nothing is scanned after the period without a new semaphore.
- `mcp3208Ctrl` semaphore mode:
  - `fps` is the scan rate limit: config `accel.fps` (default 8000, must be in (0, 10000], otherwise the default is
    kept with a warning) and the INDI `fps` target at runtime. In timer mode it remains the ADC rate. In semaphore
    mode an `fps.device` source only records the WFS rate (`m_wfs_fps`); it no longer overwrites the limit.
  - WFS period and scan anchor: `libMagAOX/app/wfsClock.hpp`, ad7689Ctrl's `rollingWfsClock` (namespace `wfsScan`,
    with its tests ported). It fits ten seconds of semaphore arrivals (monotonic) against the WFS frame ID (`cnt0` of
    the sync stream; the semaphore count without metadata) by centered least squares. The scans use its period and,
    as each frame's anchor, its fitted arrival for that frame ID, so semaphore jitter moves neither. An arrival outside
    the gate (`wfsCadenceTolerance` x period, at least 20 us and 6x the fitted jitter) is rejected but still scanned
    from the fit; `wfsFitMinEvents` (16) mutually consistent rejected arrivals restart the fit at the new rate. No scans
    until the fit qualifies (16 frames, 8 ms at 2 kHz) and while a rate change is being confirmed.
  - This replaced an EMA of semaphore intervals (alpha 0.01) with a reset after 3 consecutive intervals more than 10%
    off. On the Pi, `camwfs_sync` arrives over TCP; jittered or bunched arrivals moved the EMA, and three in a row reset
    it to a single jittered interval. Below a 495 us estimate the count fell to 3, so `acq_fps` dropped whenever
    `avg_semaphore_period_us` moved from 500 us. The EMA remains only for the delay-model diagnostics before the fit
    qualifies; `avg_semaphore_period_us` reports the fitted period once it does.
  - INDI `wfsClock`: `valid`, `events`, `span_s`, `fps`, `period_us`, `jitter_us` (RMS residual of arrivals),
    `arrival_error_us` (latest arrival minus fit), `drift_us`, `rejected`, `resets`.
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

## No built-in delay

At the user's request the scans no longer use the WFS delay model. Each WFS frame's scans are at the fitted
(ten-second) arrival plus 0, 125, 250 and 375 us at 2 kHz / 8 kHz, re-anchored on every semaphore; `synchroDelay`
(`synchro.postDelay`) remains an optional signed offset, default 0. The read-latency controller is removed: the read
latency (first scan start after the fitted arrival) is still reported, not fed back. The model (`dtTransfer_ns`,
`wfsProcess_ns`, `dtF_ns`, `wfsRead_ns`; WFS mid-exposure) is still computed and reported as
`timingDiag.delay_model_us`, and its config keys are still accepted, since MagAO-X stops an app on an unrecognized
config key. `delay_phase_error_us` is now the applied offset minus the requested one, and `delay_lock` also requires a
qualified WFS clock, so loPredCtrlAcc's lock check keeps a meaning. With offset 0 the last scan (375 us plus the read)
ends before the next semaphore, so the phase cap is not reached and a semaphore is not detected late behind a read.
Earlier sections describing the delay target and controller are superseded.

## Scheduling through arrival bursts

On the Pi (`dfa88de9`): `wfsClock.fps` 2000.00035 (target 8000.0014), `acq_fps` 7997.996, `jitter_us` 4.4,
`rejected` 3982 of 672233, `resets` 0, `missed_frames` 1122 with `coalesced_wakes` 0 and `sync_frames_dropped` 0,
`skipped_scans` 1607. `camwfs_sync` occasionally arrives more than the 50 us gate late, several frames in a row (TCP
bursts). From the third consecutive rejected arrival the clock is checking for a rate change (`valid()` false), and
those frames armed no scans; their owed scans overflowed the one-period backlog. The fit is unchanged unless a rate
change is confirmed, so scans now follow the fit whenever one exists (`m_ready`), and `delay_lock` likewise, so it
no longer drops during a burst.

## Verification

- `wfsClock_test` (ad7689Ctrl's clock tests, without its continuous schedule): passed on the Mac (6 cases). A
  standalone run of the clock confirmed the sequences the new mcp3208Ctrl tests expect.
- `wfsScanSchedule_test` (ad7689Ctrl's schedule tests, ported, plus the integer-multiple limit at 2000, 500, 700,
  1700, 2000.4, 1000 and 9000 Hz): built and passed on the development Mac (6 cases).
- `mcp3208Ctrl_test`: compiles with GCC 15 against the real ImageStreamIO headers on the Mac (syntax check, no
  warnings in the changed files). It cannot link or run there (no ImageStreamIO library, macOS has no unnamed
  semaphores), so it must be run on Linux:
  `make -C tests -f Makefile.one t=../apps/mcp3208Ctrl/tests/mcp3208Ctrl_test && apps/mcp3208Ctrl/tests/mcp3208Ctrl_test`.
  New or rewritten cases: arming at the modelled phase, deadlines and late-slot skipping, the read-latency EMA and
  controller per WFS period, the service EMA, four scans from one real semaphore post, period start and missed-frame
  normalization, a 2 kHz to 700 Hz change re-fitted by the clock (11 scans), anchors under +/-30 us jitter, `accel.fps` loading and range, the synchronized fps source,
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
- ad7689Ctrl should also include `libMagAOX/app/wfsClock.hpp` instead of its own copy.
- ad7689Ctrl's continuous schedule (one clock across WFS periods, so a late reference never cancels the previous
  period's pending scans) is not ported; with fitted anchors only a semaphore delayed past the last scan of its period
  loses scans.
