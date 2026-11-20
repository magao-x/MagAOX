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
command(T) = -gain * ( A + sum_k Δ_k * ramp(T - t0_k) )
ramp(τ)    = raised cosine, 0 before rampStartUs, 1 after rampStartUs + rampDurationUs
```

- **The accumulator `A`.** A completed ramp is added into `A`. After every successfully published slot, `A` is
  multiplied by `(1 - leak)`.
  - `leak = 0` (the default) holds the contribution.
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
| `counterOffload.gain` | `gain` | 1 | Scale on the published command, 0 to 2. |
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

## Calibrate mode

Calibrate mode tunes `rampStartUs`, `rampDurationUs` and `gain` to minimize the wavefront error spike that offloading
causes, measured in the closed-loop WFS PSDs.

**Figure of merit.** The power of the raw closed-loop PSDs (`aol1_raw_clpsds`, from modalPSDs) within `halfWidthHz` of
the offload rate `t2wOffloader.fps`, summed over the first `t2wOffloader.numModes` modes. It is reported as a ratio to a
baseline measured at gain 0, the counter-offload off.

**Averaging.** Each evaluation waits one PSD window plus `settleS` after the change, so that only PSDs of data taken
after the change are used, then averages `nAverage` of them (at least 2). The baseline is re-measured every
`baselineEvery` evaluations to follow drift.

**Scan.**

1. Measure the baseline and the starting parameters.
2. Scan `rampStartUs`, then `rampDurationUs`, then `gain`, each over its grid with the others at the best so far.
3. Rescan each over ± one step at half the step, then evaluate the vertex of a parabola through the minimum.
4. Apply the best parameters and log them at INFO, as values to copy into the config.

With the defaults a run takes about 5 minutes.

**Running it.** With the loop closed, offloading, and the app enabled:

```bash
setINDI -s <instance>.calibrate.toggle=On
```

Turning `calibrate` off aborts the run. The run also aborts, restoring the parameters from before it started, if:

- an evaluation exceeds `abortRatio` times the baseline;
- no settled PSDs arrive within `timeoutS` (for example the loop opened or modalPSDs stopped);
- `t2wOffloader.fps` or `numModes` changes;
- the app is disabled, or the PSD stream is reallocated.

Changing the gain scales the whole held contribution at once, so with `leak` 0 each gain step is a step on the tweeter
that the loop removes during the settling time.

**Results.** The read-only `calibration` property shows `stage`, `point`, the latest `fom`, `fomOff` and `ratio`, and
the best parameters and ratio so far. Each run writes `<instance>_calib_<UTC time>.csv` in `csvDir`, one row per
evaluation, with the stage, parameters, offload rate, modes, band, `nAverage`, `fom`, its standard deviation, `fomOff`
and `ratio`.

| Setting | INDI | Default | Meaning |
| --- | --- | --- | --- |
| `psds.shmimName` | — | `aol1_raw_clpsds` | Raw closed-loop PSD stream, frequency × mode. |
| `calibrate.freqStream` | — | `aol1_freq` | Frequency axis of the PSDs. |
| `calibrate.offloadDevice` | — | `t2wOffloader` | Source of `fps` and `numModes`. |
| `calibrate.halfWidthHz` | `calibHalfWidthHz` | 10 | Band half width around the offload rate, Hz. |
| `calibrate.modeMax` | — | 0 | Modes summed; 0 uses `t2wOffloader.numModes`. |
| `calibrate.nAverage` | — | 6 | PSDs averaged per evaluation, at least 2. |
| `calibrate.baselineEvery` | — | 5 | Evaluations between baselines. |
| `calibrate.settleS` | — | 0.5 | Settling beyond one PSD window, s. |
| `calibrate.timeoutS` | — | 15 | Longest wait for an evaluation's PSDs, s. |
| `calibrate.abortRatio` | — | 3 | Abort above this multiple of the baseline. |
| `calibrate.startGridUs` | — | 0,400,25 | `rampStartUs` scan: min, max, step. |
| `calibrate.durationGridUs` | — | 100,600,50 | `rampDurationUs` scan: min, max, step. |
| `calibrate.gainGrid` | — | 0.5,1.5,0.1 | `gain` scan: min, max, step. |
| `calibrate.csvDir` | — | `/home/xsup/counterOffload` | Directory for the run CSVs. |
