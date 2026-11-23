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
- **The clip level.** Every pixel of the command, and of `A`, is limited to ± `maxCommand` (default 1.0). Limiting
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
- **Default ramp and gain.** 0 µs start, 330 µs duration and gain 0.5.
  - **Start.** The start is from bench calibration runs 20261120T202219Z, 20261122T183157Z and 20261122T184435Z. In each paired comparison a start of 0 measured as good as 25 µs or better, and starts of 50 µs or more were worse by about 0.1 in ratio at gain 0.4.
  - **"As soon as possible".** The first dmSyncopation slot lands 100–225 µs after an offload arrives, so a start of 0 means "as soon as possible".
  - **Why not 120 µs.** The original 120 µs start was fitted so that the tweeter's response matches the woofer's deconvolved step response, but it is later than optimal.
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
| `counterOffload.rampStartUs` | `rampStartUs` | 0 | Ramp start after the woofer command, µs. |
| `counterOffload.rampDurationUs` | `rampDurationUs` | 330 | Ramp duration, µs. |
| `counterOffload.leak` | `leak` | 0 | Fraction of the completed contribution removed after each published slot. |
| `counterOffload.timeOffsetUs` | — | 0 | Added to the receipt time to give the ramp reference, µs. |
| `counterOffload.gain` | `gain` | 0.5 | Scale applied to each new offload when it arrives, 0 to 2. |
| `counterOffload.maxCommand` | — | 1.0 | Clip level for every pixel of the command and of `A`. |
| `counterOffload.computeLeadUs` | — | 20 | Time reserved to compute a command before the publish lead, µs. |
| `counterOffload.threadPrio` / `cpuset` | — | 20 / none | Producer thread placement; the example uses `spare26`. |
| `syncopation.timingStream` | — | none (required) | dmSyncopation schedule, e.g. `dmtweeter_synco_timing`. |
| `syncopation.writeLeadUs` | — | 50 | Publish lead before each slot's cutoff, µs; the example uses 10. |
| `shmimMonitor.shmimName` | — | `aol0_tweeterOffload` | The offload input stream. |

The operating settings with an INDI property can be changed while running, but a change lasts only until restart.
Put calibrated values in the instance config to keep them.

The read-only `stats` property reports these counters:

- `events`: offloads accepted.
- `published`: commands published.
- `failed`: publishes that dmSyncopation did not accept.
- `clipped`: slots in which the command or `A` was clipped to `maxCommand`.

If the ramp pool fills, so that offloads are folded into `A` without a ramp, the app logs a WARNING with the number
of offloads affected.

## Latency

The read-only `latency` property shows where the time goes between a woofer offload and the counter command reaching
dmSyncopation. Its purpose is to tell whether computation limits how fast the tweeter is commanded, or whether the limit
is structural: the slot rate, the fixed leads and the tweeter rise.

Each offload is measured once, at the first slot published after it arrives. The values are over the last 250 offloads
(about 1 s at 250 Hz), all in µs except `missed`. A value with no data yet is −1.

| Element | Meaning | Cause |
| --- | --- | --- |
| `pipelineUs` | Mean of `handoffUs + computeUs + writeUs`: all the computation from the woofer post to the command reaching dmSyncopation. Compare it with `slotUs`. | computation |
| `pipelineRmsUs` | RMS deviation of that sum about its mean (its jitter). | computation |
| `slotUs` | The slot period, from successive slot targets (125 at 8 kHz). | configuration |
| `handoffUs` | Woofer post to counterOffload receiving the offload: t2wOffloader's projection and post, plus the wake-up. | computation |
| `slotWaitUs` | Receipt to the start of computing the slot that carries the offload: waiting for the slot grid. | slot timing |
| `computeUs` | Time to build the command. | computation |
| `writeUs` | Time from the start of the copy into `dm01disp08_aux` to the end of the publication. | computation |
| `computeMaxUs` | The largest `computeUs` in the window. Above `computeLeadUs` (20), slots fail. | computation |
| `missed` | Offloads that waited at least one whole slot before being carried, since startup. | scheduling |

After a slot's command is computed, a fixed lead remains before its target: `computeLeadUs + writeLeadUs` plus
dmSyncopation's `readyLeadUs` (20 + 10 + 70 = 100 µs). It is logged once, at the first slot. After the target comes the
tweeter rise, about 314 µs to 50%, which the app does not measure.

How to read it:

- **Computation is the limit** if `pipelineUs` is a large part of `slotUs`, or `missed` keeps increasing.
- **The limit is structural** if `pipelineUs` is a few µs. The delay is then dominated by `slotWaitUs` (0 to one
  slot), the fixed lead and the tweeter rise.
- **The leads can be reduced** if `computeMaxUs` and `writeUs` are well below `computeLeadUs` and `writeLeadUs`.

`handoffUs` needs the woofer post time, which t2wOffloader stores in the `atime` of each `aol0_tweeterOffload` frame. With
an older t2wOffloader, which does not store it, `handoffUs`, `pipelineUs` and `pipelineRmsUs` stay at −1.

## Calibrate mode

Calibrate mode tunes `rampStartUs`, `rampDurationUs` and `gain` to minimize the wavefront error spike that offloading
causes, measured in the closed-loop WFS PSDs.

**Figure of merit.** The power of the averaged closed-loop PSDs (`aol1_clpsds`, from modalPSDs) within `calibrate.halfWidthHz` of
the offload rate `t2wOffloader.fps`, summed over the first `t2wOffloader.numModes` modes. It is reported as a ratio to a
baseline measured at gain 0, the counter-offload off. The baseline drifts (about ±16% over 8 min on the bench), so
each ratio uses the baseline linearly interpolated in time between the baselines measured before and after it.

**Averaging.** `aol1_clpsds` is the mean of the raw PSDs over the last `psdAvgTime` (10 s), republished every 0.5 s. Its
oldest data are `psdAvgTime + psdTime` (10.5 s) old. After each change, an evaluation therefore waits `calibrate.settleS` (11 s),
so that only PSDs made entirely of data taken after the change are used. It then averages the next `calibrate.nAverage` (3)
published PSDs, 0.5 s apart.

These samples share most of their data. Their standard deviation (`fomStd`) shows whether the value is stable; it is
not an independent noise estimate. If you change modalPSDs `psdAvgTime` or `psdTime`, change `calibrate.settleS` to match.

**Scan.** Repeated measurements of the same parameters scatter by about 0.05–0.08 in ratio, as much as many of the
effects being measured, so the run is built to choose values from many points, not from the single lowest one.

1. **Setup.** Set `leak` to `calibrate.leak` (default 0.0025 per slot, about 50 ms at 8 kHz), so that the held
   contribution stays at a few offloads.
2. **Noise estimate.** Measure the starting parameters `calibrate.replicates` (3) times, each between baselines. Their
   scatter is the first estimate σ of the noise of one ratio.
3. **Sweeps.** In round 1, sweep `rampStartUs`, then `gain`, then `rampDurationUs`. In later rounds (`calibrate.rounds`,
   default 2), sweep `rampStartUs` and `gain` again with the updated values, since they may interact. In each sweep:
   - the grid is measured in a scrambled order (for example 0, 100, 50, 150, 25, 125, 75), so slow drift does not
     look like a trend;
   - a baseline is measured before and after the sweep, and after every `calibrate.baselineEvery` (4) points;
   - a quadratic is fitted to all the sweep's ratios, and its minimum within the measured range is the candidate.
     With fewer than 4 points the minimum of a 3-point running mean is used. The fit residuals are pooled into σ,
     which is never taken below `calibrate.sigmaFloor`;
   - the parameter moves to the candidate only if the predicted improvement over its current value exceeds
     `calibrate.moveSigma` × σ × √(6 / n), with n the number of points (1.5 standard errors by default). Otherwise it keeps
     its value. A current value outside the swept range always moves.
4. **Confirmation.** Measure the chosen parameters `calibrate.confirmPairs` (3) times, each between baselines. If their mean
   ratio is below 1, apply them and log them, with the confirmed ratio ± its standard error, at INFO as values to copy
   into the config. Otherwise restore the parameters from before the run, with a WARNING.
5. Restore the original `leak`.

With the defaults a run is 58 measurements of about 12.5 s each, so about 12 minutes. The estimate is logged when a run
starts, as a WARNING if it is over 20 minutes, for example after widening a grid.

**Clipping.** A point is never skipped because the counter command clips at `maxCommand`: strong turbulence can
saturate the counter whatever the parameters, so every point is used in the fits, the noise estimate and the
confirmation. Clipping is recorded instead, counted from 1 s after each change (so the previous point's held
contribution can leak away first) to the end of the point. Each measurement row of the CSV has `clippedSlots` and
`clipFrac` (clipped / published slots), a point that clipped is logged at INFO, and the run logs how many measurements
clipped when it ends.

**Running it.** With the loop closed, offloading, and the app enabled:

```bash
setINDI -s <instance>.calibrate.toggle=On
```

Turning `calibrate` off aborts the run. The run also aborts if:

- a point exceeds `calibrate.abortRatio` times the latest baseline;
- no settled PSDs arrive within `calibrate.timeoutS` (for example the loop opened or modalPSDs stopped);
- `t2wOffloader.fps` or `numModes` changes;
- the app is disabled, or the PSD stream is reallocated;
- the gain-0 baseline cannot be measured.

On an abort, the parameters chosen so far are applied, with a WARNING, if their predicted ratio is below 1. Otherwise
the parameters from before the run are restored.

The gain applies to offloads as they arrive, so the gain-0 baseline means "stop countering new offloads". The
contribution already held leaks away within about 50 ms at the calibration leak.

**Results.** The read-only `calibration` property shows:

- `stage` (0 off, 1 initial, 2 start, 3 duration, 4 gain, 5 confirm, 6 no longer used, 7–9 the start, duration and gain
  decisions) and `point`, the number of points measured;
- the latest `ratio` (to the latest baseline);
- `bestStartUs`, `bestDurationUs`, `bestGain` and `bestRatio`: the parameters chosen so far and their predicted ratio;
- `confirmRatio`, the mean confirmed ratio, at the end.

The band powers, the noise estimate σ and the standard error of the confirmed ratio are in the CSV and the INFO logs.

Each run writes `<instance>_calib_<UTC time>.csv` in `csvDir`. Measurement rows have the time, stage, parameters,
offload rate, modes, band, `nAverage`, `fom`, `fomStd`, `fomOff` (the latest baseline), `ratio` (to it) and
`ratioInterp` (to the interpolated baseline, the value the run uses), then `clippedSlots` and `clipFrac`. They are written when their sweep closes. After
each sweep a `fit start`, `fit duration` or `fit gain` row records the decision: the parameters with the swept one at
the candidate, then `fitCurrent` and `fitCandidate` (the predicted ratios; `fitCurrent` is empty if the current value
was outside the sweep), `threshold`, `moved` (1 or 0) and `sigma`.

These are set in the config:

| Setting | Default | Meaning |
| --- | --- | --- |
| `psds.shmimName` | `aol1_clpsds` | Averaged closed-loop PSD stream, frequency × mode. |
| `calibrate.freqStream` | `aol1_freq` | Frequency axis of the PSDs. |
| `calibrate.offloadDevice` | `t2wOffloader` | Source of `fps` and `numModes`. |
| `calibrate.csvDir` | `/home/xsup/counterOffload` | Directory for the run CSVs. |

These numeric settings are optional config keys in the `[calibrate]` section. None needs to be set: leave them out
to use the defaults. An out-of-range or non-integral value for an integer setting is logged as an error, and the app does
not start. All of them, and the grids, are logged at DEBUG at startup and at the start of every run.

| Setting | Default | Meaning |
| --- | --- | --- |
| `calibrate.halfWidthHz` | 10 | Band half width around the offload rate, Hz. |
| `calibrate.modeMax` | 0 | Modes summed; 0 uses `t2wOffloader.numModes`. |
| `calibrate.nAverage` | 3 | Settled PSDs averaged per evaluation, 2 to 1000. |
| `calibrate.baselineEvery` | 4 | Sweep points between baselines. |
| `calibrate.settleS` | 11 | Wait after each change before PSDs count; must cover `psdAvgTime + psdTime`, s. |
| `calibrate.timeoutS` | 5 | Longest wait for an evaluation's PSDs after settling, s. |
| `calibrate.abortRatio` | 3 | Abort above this multiple of the baseline. |
| `calibrate.leak` | 0.0025 | `leak` used during a run, restored afterwards; must be above 0. |
| `calibrate.replicates` | 3 | Measurements of the starting parameters for the first noise estimate, 2 to 20. |
| `calibrate.confirmPairs` | 3 | Measurements of the chosen parameters, each after a baseline, 1 to 20. |
| `calibrate.moveSigma` | 1.5 | Standard errors of improvement needed to move a parameter. |
| `calibrate.sigmaFloor` | 0.02 | Smallest noise estimate of the ratio used. |
| `calibrate.rounds` | 2 | Rounds of sweeps; rounds after the first sweep only `rampStartUs` and `gain`, 1 to 5. |

The scan grids are INDI properties only, not config keys. A change lasts until restart, and they cannot be changed while
a run is in progress. A rejected value is logged as an error and the grid keeps its value.

| INDI property | Default | Meaning |
| --- | --- | --- |
| `calibStartGridUs` | 0, 150, 25 | `rampStartUs` scan; elements `min`, `max`, `step`. |
| `calibDurationGridUs` | 200, 600, 100 | `rampDurationUs` scan; elements `min`, `max`, `step`. |
| `calibGainGrid` | 0.3, 1.0, 0.1 | `gain` scan; elements `min`, `max`, `step`. |

For example, to scan gains from 0.5 to 1.2:

```bash
setINDI -s <instance>.calibGainGrid.min=0.5
setINDI -s <instance>.calibGainGrid.max=1.2
```

Each element is checked with the other two as they currently are, so when narrowing or moving a grid, set the elements
in an order that keeps min ≤ max.
