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
- **Default ramp and gain.** 25 µs start, 330 µs duration and gain 0.5: the best measured in bench calibration run
  20261120T202219Z. A start earlier than the first dmSyncopation slot (100–225 µs after an offload arrives) means
  "as soon as possible". The original 120 µs start was fitted so that the tweeter's response matches the woofer's
  deconvolved step response, but it is later than optimal.
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
| `counterOffload.rampStartUs` | `rampStartUs` | 25 | Ramp start after the woofer command, µs. |
| `counterOffload.rampDurationUs` | `rampDurationUs` | 330 | Ramp duration, µs. |
| `counterOffload.leak` | `leak` | 0 | Fraction of the completed contribution removed after each published slot. |
| `counterOffload.timeOffsetUs` | `timeOffsetUs` | 0 | Added to the receipt time to give the ramp reference, µs. |
| `counterOffload.gain` | `gain` | 0.5 | Scale applied to each new offload when it arrives, 0 to 2. |
| `counterOffload.maxCommand` | `maxCommand` | 1.0 | Clip level for every pixel of the command and of `A`. |
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
- `overflows`: offloads folded into `A` early because the ramp pool was full.
- `active`: ramps still running.
- `clipped`: slots in which the command or `A` was clipped to `maxCommand`.

## Calibrate mode

Calibrate mode tunes `rampStartUs`, `rampDurationUs` and `gain` to minimize the wavefront error spike that offloading
causes, measured in the closed-loop WFS PSDs.

**Figure of merit.** The power of the averaged closed-loop PSDs (`aol1_clpsds`, from modalPSDs) within `calibHalfWidthHz` of
the offload rate `t2wOffloader.fps`, summed over the first `t2wOffloader.numModes` modes. It is reported as a ratio to a
baseline measured at gain 0, the counter-offload off. The baseline drifts (about ±16% over 8 min on the bench), so
each ratio uses the baseline linearly interpolated in time between the baselines measured before and after it.

**Averaging.** `aol1_clpsds` is the mean of the raw PSDs over the last `psdAvgTime` (10 s), republished every 0.5 s. Its
oldest data are `psdAvgTime + psdTime` (10.5 s) old. After each change, an evaluation therefore waits `calibSettleS` (11 s),
so that only PSDs made entirely of data taken after the change are used. It then averages the next `calibNAverage` (3)
published PSDs, 0.5 s apart.

These samples share most of their data. Their standard deviation (`fomStd`) shows whether the value is stable; it is
not an independent noise estimate. If you change modalPSDs `psdAvgTime` or `psdTime`, change `calibSettleS` to match.

**Scan.** Repeated measurements of the same parameters scatter by about 0.05–0.08 in ratio, as much as many of the
effects being measured, so the run is built to choose values from many points, not from the single lowest one.

1. **Setup.** Set `leak` to `calibLeak` (default 0.0025 per slot, about 50 ms at 8 kHz), so that the held
   contribution stays at a few offloads.
2. **Noise estimate.** Measure the starting parameters `calibReplicates` (3) times, each between baselines. Their
   scatter is the first estimate σ of the noise of one ratio.
3. **Sweeps.** In round 1, sweep `rampStartUs`, then `gain`, then `rampDurationUs`. In later rounds (`calibRounds`,
   default 2), sweep `rampStartUs` and `gain` again with the updated values, since they may interact. In each sweep:
   - the grid is measured in a scrambled order (for example 0, 100, 50, 150, 25, 125, 75), so slow drift does not
     look like a trend;
   - a baseline is measured before and after the sweep, and after every `calibBaselineEvery` (4) points;
   - a quadratic is fitted to all the sweep's ratios, and its minimum within the measured range is the candidate.
     With fewer than 4 points the minimum of a 3-point running mean is used. The fit residuals are pooled into σ,
     which is never taken below `calibSigmaFloor`;
   - the parameter moves to the candidate only if the predicted improvement over its current value exceeds
     `calibMoveSigma` × σ × √(6 / n), with n the number of points (1.5 standard errors by default). Otherwise it keeps
     its value. A current value outside the swept range always moves.
4. **Confirmation.** Measure the chosen parameters `calibConfirmPairs` (3) times, each between baselines. If their mean
   ratio is below 1, apply them and log them, with the confirmed ratio ± its standard error, at INFO as values to copy
   into the config. Otherwise restore the parameters from before the run, with a WARNING.
5. Restore the original `leak`.

With the defaults a run is 58 measurements of about 12.5 s each, so about 12 minutes. The estimate is logged when a run
starts, as a WARNING if it is over 20 minutes, for example after widening a grid.

**Clipping.** A point at which the counter command clips at `maxCommand` is skipped, not used. Clipping is counted
from 1 s after each change, so the previous point's held contribution can leak away first. The point is written to the
CSV with stage `clipped` and empty `fom`, `fomStd` and ratios. Larger values of the same parameter in that sweep are
not measured, and those already measured are not used, so a gain sweep ends at the first gain that clips.

**Running it.** With the loop closed, offloading, and the app enabled:

```bash
setINDI -s <instance>.calibrate.toggle=On
```

Turning `calibrate` off aborts the run. The run also aborts if:

- a point exceeds `calibAbortRatio` times the latest baseline;
- no settled PSDs arrive within `calibTimeoutS` (for example the loop opened or modalPSDs stopped);
- `t2wOffloader.fps` or `numModes` changes;
- the app is disabled, or the PSD stream is reallocated;
- the gain-0 baseline cannot be measured, including when it clips.

On an abort, the parameters chosen so far are applied, with a WARNING, if their predicted ratio is below 1. Otherwise
the parameters from before the run are restored.

The gain applies to offloads as they arrive, so the gain-0 baseline means "stop countering new offloads". The
contribution already held leaks away within about 50 ms at the calibration leak.

**Results.** The read-only `calibration` property shows:

- `stage` (0 off, 1 initial, 2 start, 3 duration, 4 gain, 5 confirm, 6 clipped, 7–9 the start, duration and gain
  decisions) and `point`, the number of points measured;
- the latest `fom`, `fomOff` and `ratio` (to the latest baseline);
- `bestStartUs`, `bestDurationUs`, `bestGain` and `bestRatio`: the parameters chosen so far and their predicted ratio;
- `sigma`, the current noise estimate, and `confirmRatio` ± `confirmStd` at the end.

Each run writes `<instance>_calib_<UTC time>.csv` in `csvDir`. Measurement rows have the time, stage, parameters,
offload rate, modes, band, `nAverage`, `fom`, `fomStd`, `fomOff` (the latest baseline), `ratio` (to it) and
`ratioInterp` (to the interpolated baseline, the value the run uses). They are written when their sweep closes. After
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

These are INDI properties only, not config keys. A change lasts until restart, and they cannot be changed while a run
is in progress. A rejected value is logged as an error and the setting keeps its value.

| INDI property | Default | Meaning |
| --- | --- | --- |
| `calibHalfWidthHz` | 10 | Band half width around the offload rate, Hz. |
| `calibModeMax` | 0 | Modes summed; 0 uses `t2wOffloader.numModes`. |
| `calibNAverage` | 3 | Settled PSDs averaged per evaluation, 2 to 1000. |
| `calibBaselineEvery` | 4 | Sweep points between baselines. |
| `calibSettleS` | 11 | Wait after each change before PSDs count; must cover `psdAvgTime + psdTime`, s. |
| `calibTimeoutS` | 5 | Longest wait for an evaluation's PSDs after settling, s. |
| `calibAbortRatio` | 3 | Abort above this multiple of the baseline. |
| `calibLeak` | 0.0025 | `leak` used during a run, restored afterwards; must be above 0. |
| `calibStartGridUs` | 0, 150, 25 | `rampStartUs` scan; elements `min`, `max`, `step`. |
| `calibDurationGridUs` | 200, 600, 100 | `rampDurationUs` scan; elements `min`, `max`, `step`. |
| `calibGainGrid` | 0.3, 1.0, 0.1 | `gain` scan; elements `min`, `max`, `step`. |
| `calibReplicates` | 3 | Measurements of the starting parameters for the first noise estimate, 2 to 20. |
| `calibConfirmPairs` | 3 | Measurements of the chosen parameters, each after a baseline, 1 to 20. |
| `calibMoveSigma` | 1.5 | Standard errors of improvement needed to move a parameter. |
| `calibSigmaFloor` | 0.02 | Smallest noise estimate of the ratio used. |
| `calibRounds` | 2 | Rounds of sweeps; rounds after the first sweep only `rampStartUs` and `gain`, 1 to 5. |

For example, to scan gains from 0.5 to 1.2:

```bash
setINDI -s <instance>.calibGainGrid.min=0.5
setINDI -s <instance>.calibGainGrid.max=1.2
```

Each element is checked with the other two as they currently are, so when narrowing or moving a grid, set the elements
in an order that keeps min ≤ max.
