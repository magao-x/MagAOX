# counterOffload

## Purpose

When t2wOffloader moves low-order modes from the tweeter to the woofer, the tweeter does not know about the offload.
The woofer's slow rise (about 250–300 µs, 10–90%) then shows up on the wavefront sensor as a low-order disturbance,
which the main loop corrects after the fact. counterOffload cancels it.

- **Input:** t2wOffloader publishes each modal offload change, as it appears on the tweeter, to `aol0_tweeterOffload`.
- **Output:** counterOffload applies the opposite shape to the tweeter as a dmSyncopation producer. Its timing
  follows the woofer's smooth rise, at up to 8 kHz.

The design record and test results are in
[`agents/plans/2026-11/counter-offload-app.md`](../../../agents/plans/2026-11/counter-offload-app.md).
dmSyncopation's producer setup is described in
[`apps/dmSyncopation/doc/production-providers.md`](../../dmSyncopation/doc/production-providers.md).

## What it publishes

Each `aol0_tweeterOffload` update Δ, received at time t0, starts one ramp. For a dmSyncopation slot with
target time T, the command written to the auxiliary source is:

```text
command(T) = -( A + sum_k gain_k * Δ_k * ramp(T - t0_k) ),  clipped to ± maxCommand
ramp(τ)    = raised cosine, 0 before rampStartUs, 1 after rampStartUs + rampDurationUs
```

- **The gain.** `gain_k` is the gain at the moment offload k arrives. Changing `gain` affects only later offloads. It
  never rescales what is already held on the tweeter, so a gain change cannot step the tweeter.
- **The clip level.** Every pixel of the command, and of `A`, is limited to ± `maxCommand` (default 0.5). Limiting
  `A` stops the held contribution winding up: once it reaches the limit, further offloads in the same direction are
  not held, and an offload in the other direction takes effect at once. Each slot that clips is counted in
  `stats.clipped`, and the app logs a warning at most once per second while clipping continues.

- **The accumulator `A`.** A completed ramp is added into `A`. After every successfully published slot, `A` is
  multiplied by `(1 - leak)`.
  - `leak = 0` (the default) holds the contribution. Over many offloads it then approaches the woofer's total offloaded
    shape, up to the clip level, and the tweeter's main loop must hold the opposite. Use `leak > 0` for continued
    operation.
  - With `leak > 0`, the contribution leaks back to zero, so the main loop takes the correction back without a
    spike. The effective time constant depends on dmSyncopation's step count.
- **Default ramp.** 120 µs start and 330 µs duration, fitted so that the tweeter's response to the ramp matches the
  woofer's deconvolved step response.
- **The ramp reference t0.** The time counterOffload receives the update, plus `timeOffsetUs`.

The app publishes only while something is changing: an active ramp, a leaking contribution, or a zero request. In
any other slot it skips publishing and dmSyncopation keeps the last contribution.

## Running it on the RTC

1. Make sure dmSyncopation is running in production mode, with `dm01disp08` in `streams.destinations`, so that
   `dm01disp08_aux` exists. Also make sure nothing else writes `dm01disp08`.
2. Install an instance config based on [`dmtweeter_counteroffload.conf`](../dmtweeter_counteroffload.conf), then start
   the app through the normal operator procedure. It starts disabled.
3. For first qualification, set a small gain, then enable the app:

   ```bash
   setINDI -s <instance>.gain.target=0.1
   setINDI -s <instance>.enable.toggle=On
   ```

4. Check that synco's `production` property shows the `dm01disp08` mapping being accepted, and that the
   counterOffload `stats` counters `events` and `published` increase while offloading.

Turning `enable` off stops accepting new offloads and zeros the contribution at the next dmSyncopation slot: active
ramps and the accumulated contribution are dropped, with no decay. If dmSyncopation is not producing, the zero is
published when it resumes. To remove the contribution while staying enabled, send a zero request:

```bash
setINDI -s <instance>.zero.request=On
```

## Settings

| Setting | INDI | Default | Meaning |
| --- | --- | --- | --- |
| `counterOffload.channel` | — | `dm01disp08_aux` | Auxiliary source to write; must end in `_aux`. |
| `counterOffload.rampStartUs` | `rampStartUs` | 120 | Ramp start after the woofer command, µs. |
| `counterOffload.rampDurationUs` | `rampDurationUs` | 330 | Ramp duration, µs. |
| `counterOffload.leak` | `leak` | 0 | Fraction of the completed contribution removed after each published slot. |
| `counterOffload.timeOffsetUs` | `timeOffsetUs` | 0 | Added to the receipt time to give the ramp reference, µs. |
| `counterOffload.gain` | `gain` | 1 | Scale applied to each new offload when it arrives, 0 to 2. |
| `counterOffload.maxCommand` | `maxCommand` | 0.5 | Clip level for every pixel of the command and of `A`. |
| `counterOffload.computeLeadUs` | — | 20 | Time reserved to compute a command before the publish lead, µs. |
| `counterOffload.threadPrio` / `cpuset` | — | 20 / none | Producer thread placement; the example uses `spare26`. |
| `syncopation.timingStream` | — | none (required) | dmSyncopation schedule, e.g. `dmtweeter_synco_timing`. |
| `syncopation.writeLeadUs` | — | 50 | Publish lead before each slot's cutoff, µs; the example uses 10. |
| `shmimMonitor.shmimName` | — | `aol0_tweeterOffload` | The offload input stream. |

The read-only `stats` property reports these counters:

- `events`: offloads accepted.
- `published`: commands published.
- `failed`: publishes that dmSyncopation did not accept.
- `overflows`: offloads folded into `A` early because the ramp pool was full.
- `active`: ramps still running.
- `clipped`: slots in which the command or `A` was clipped to `maxCommand`.

## Calibrate mode

Calibrate mode tunes `rampStartUs`, `rampDurationUs` and `gain` to minimize the wavefront error spike that offloading
causes, measured in the closed-loop WFS PSDs.

**Figure of merit.** The power of the averaged closed-loop PSDs (`aol1_clpsds`, from modalPSDs) within `halfWidthHz` of
the offload rate `t2wOffloader.fps`, summed over the first `t2wOffloader.numModes` modes. It is reported as a ratio to a
baseline measured at gain 0, the counter-offload off.

**Averaging.** `aol1_clpsds` is the mean of the raw PSDs over the last `psdAvgTime` (10 s), republished every 0.5 s. Its
oldest data are `psdAvgTime + psdTime` (10.5 s) old. After each change, an evaluation therefore waits `settleS` (11 s),
so that only PSDs made entirely of data taken after the change are used. It then averages the next `nAverage` (3)
published PSDs, 0.5 s apart.

These samples share most of their data. Their standard deviation (`fomStd`) shows whether the value is stable; it is
not an independent noise estimate. If you change modalPSDs `psdAvgTime` or `psdTime`, change `settleS` to match. The
baseline is re-measured every `baselineEvery` evaluations to follow drift.

**Scan.**

1. Set `leak` to `calibrate.leak` (default 0.0025 per slot, about 50 ms at 8 kHz), so that the held contribution
   stays at a few offloads. Then measure the baseline and the starting parameters.
2. Scan `rampStartUs`, then `rampDurationUs`, then `gain`, each over its grid with the others at the best so far.
3. Rescan each over ± one step at half the step, then evaluate the vertex of a parabola through the minimum.
4. Apply the best parameters and log them at INFO, as values to copy into the config. Restore the original `leak`.

With the defaults a run is about 70 measurements of about 12 s each, so about 14 minutes.

**Running it.** With the loop closed, offloading, and the app enabled:

```bash
setINDI -s <instance>.calibrate.toggle=On
```

Turning `calibrate` off aborts the run. The run also aborts, restoring the parameters from before it started, if:

- an evaluation exceeds `abortRatio` times the baseline;
- no settled PSDs arrive within `timeoutS` (for example the loop opened or modalPSDs stopped);
- `t2wOffloader.fps` or `numModes` changes;
- the app is disabled, or the PSD stream is reallocated.

The run also aborts if the counter command clips at `maxCommand`.

The gain applies to offloads as they arrive, so the gain-0 baseline means "stop countering new offloads". The
contribution already held leaks away within about 50 ms at the calibration leak.

**Results.** The read-only `calibration` property shows `stage`, `point`, the latest `fom`, `fomOff` and `ratio`, and
the best parameters and ratio so far. Each run writes `<instance>_calib_<UTC time>.csv` in `csvDir`, one row per
evaluation, with the stage, parameters, offload rate, modes, band, `nAverage`, `fom`, its standard deviation, `fomOff`
and `ratio`.

| Setting | INDI | Default | Meaning |
| --- | --- | --- | --- |
| `psds.shmimName` | — | `aol1_clpsds` | Averaged closed-loop PSD stream, frequency × mode. |
| `calibrate.freqStream` | — | `aol1_freq` | Frequency axis of the PSDs. |
| `calibrate.offloadDevice` | — | `t2wOffloader` | Source of `fps` and `numModes`. |
| `calibrate.halfWidthHz` | `calibHalfWidthHz` | 10 | Band half width around the offload rate, Hz. |
| `calibrate.modeMax` | — | 0 | Modes summed; 0 uses `t2wOffloader.numModes`. |
| `calibrate.nAverage` | — | 3 | Settled PSDs averaged per evaluation, at least 2. |
| `calibrate.baselineEvery` | — | 5 | Evaluations between baselines. |
| `calibrate.settleS` | — | 11 | Wait after each change before PSDs count; must cover `psdAvgTime + psdTime`, s. |
| `calibrate.timeoutS` | — | 5 | Longest wait for an evaluation's PSDs after settling, s. |
| `calibrate.abortRatio` | — | 3 | Abort above this multiple of the baseline. |
| `calibrate.leak` | — | 0.0025 | `leak` used during a run, restored afterwards; must be above 0. |
| `calibrate.startGridUs` | — | 0,400,25 | `rampStartUs` scan: min, max, step. |
| `calibrate.durationGridUs` | — | 100,600,50 | `rampDurationUs` scan: min, max, step. |
| `calibrate.gainGrid` | — | 0.1,1.0,0.1 | `gain` scan: min, max, step. |
| `calibrate.csvDir` | — | `/home/xsup/counterOffload` | Directory for the run CSVs. |
