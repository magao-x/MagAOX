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
| `counterOffload.gain` | `gain` | 1 | Scale on the published command, 0 to 1. |
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
