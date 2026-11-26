# DDSPC with a free-running accelerometer

## Purpose

The request: "default to reading the accelerometers out with the integrator in mcp3208Ctrl, and send those measurement
to rtc at that rate. loPredCtrlAcc should not require the accelerometer to be synced with the wfs and any other
restrictions you have built in. this is because we are going to simply use the most recent measurements all the time.
also loPredCtrlAcc should be adaptive to what the acc and wfs are running at".

## mcp3208Ctrl (branch parkertjohnson1/mcp3208Ctrl)

- New `synchro.timingSource`, default `integrator`: acquisition free-runs on the internal timer at `accel.fps`.
  `semaphore` (with `synchro.shmimName`) selects the WFS-synchronized scans as before; INDI `timingSource` still
  switches at runtime. A configured `synchro.shmimName` alone no longer selects semaphore timing.
- The `fps.device` subscription only records the WFS rate (`m_wfs_fps`); it no longer sets the integrator's rate to the
  WFS rate, so the accelerometers run at `accel.fps` whatever the WFS does.

## loPredCtrlAcc (branch parkertjohnson1/accel_ddspc)

- **Arrival timing** (`AdcConfig::m_strictTiming = false`, operational only; the shadow tools keep the strict default):
  every in-range sample is accepted as it arrives. Producer (Pi) timestamps are mapped for diagnostics and the rate
  only, an unmapped one is taken as the receipt time; a clock offset change re-anchors instead of latching a clock
  fault; lost publications, interval gaps and invalid samples are counted without breaking the reader's segment.
  Input age, and the stale state, are from the last receipt. The WFS-thread window's staleness check
  (`stale_sample`) uses the newest sample's receipt time: only ACC silence longer than `adc.max_age_us` holds.
- **No upstream lock**: the lock (`adc.sync_device`/`sync_property`) is shown (`accel_sync`, `assisted_qualification`)
  but gates nothing. The assisted fit learns and probes without it, the reset-required latch for fits trained without
  lock is removed, `assisted_ready` and the blockers no longer include it, and a lock report no longer invalidates the
  ACC window (with acc000 free-running, every `timingDiag` report would have). `assisted_diagnostic` is accepted and
  unused; the `assisted_qualification.diagnostic_mode` element is removed.
- **Adaptive ratio**: `acc_ratio` (detected, `agents/plans/2026-11/ddspc-detected-rates.md`) is the ACC samples per WFS
  frame rounded, at any rate. A measurement within 0.6 of the confirmed ratio keeps it (no alternation near a
  half-integer); the result is limited to 1-16 and to `128 / accel_history`. An ACC slower than the WFS gives 1.

## Limits

- An unsynchronized ACC samples at a varying phase relative to each WFS frame; the newest samples are used as they are.
- A dead ACC stream still holds the assisted output after `adc.max_age_us` and stops it after `parameters.hold_timeout`.
- mcp3208Ctrl's unit tests could only be syntax-checked on the development Mac (g++-15 with stub headers); build and
  run them on the Pi.

## Verification (offline, development Mac)

- `check-rtc-controller`: all but the known timing-flaky "mfilt clone 2k" test pass, including "Assisted DDSPC learns
  and predicts without the upstream lock" (1200 unlocked frames: the fit learns, is ready and qualifies; a no-lock
  report leaves the window valid), the updated qualification test (no lock blockers), and the ratio cases.
- `check-native-observation` (ADC capture): "ADC arrival timing accepts every in-range sample whatever the producer
  clock" (Pi clock 5 s ahead, realtime step, overwritten publications, staleness from receipt; strict unchanged).
- `check-operational-controls`, `check-operational-output`, `operational_config_test.py` pass.
