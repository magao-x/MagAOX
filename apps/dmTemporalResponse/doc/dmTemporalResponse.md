dmTemporalResponse
==================

[TOC]

------------------------------------------------------------------------

# NAME

dmTemporalResponse - measure the temporal step response of a DM with sub-frame resolution using camWFS.

# SYNOPSIS

```
dmTemporalResponse -n dmTemporalResponse
```

Runs are started and controlled from cursesINDI (see [INDI](#indi)).

# DESCRIPTION

`dmTemporalResponse` measures how a deformable mirror responds in time to a step command, sampled faster than the
camWFS frame rate.

For every trial the app:

1. waits for the camWFS frame semaphore (the *trigger* frame);
2. busy-waits until `atime + d`, where `atime` is the trigger frame's acquisition time and `d` is the delay;
3. writes the poke command to the DM channel and records the time it was written;
4. captures the next N camWFS frames;
5. zeroes the DM and waits the settle time.

The delay `d` steps through an evenly spaced grid `d_k = k * span / K`, `k = 0..K-1`.  By default the span is one
camWFS frame period, so the K delays sample the step response at K phases within a frame.  For each delay the app
runs M/2 trials with a positive poke followed by M/2 trials with a negative poke, and saves the averaged half
difference `(mean(+) - mean(-)) / 2`.  The +/- difference removes the static WFS signal and the camera bias, so no
dark is needed.

A reference pass (`nRef` +/- pairs at `d = 0`) runs first.  Its steady-state difference image `P`, averaged over the
last `nSettle` frames, is used to reduce each frame to a scalar response `r` (0 unpoked, 1 fully settled).  From the
per-trial curves the app computes, per delay:

| Metric | Definition |
|---|---|
| `t50` | time from the DM command to `r = 0.5` [us] |
| `rise` | 10-90% rise time [us] |
| `overshoot` | `max(r) - 1` |
| `settleErr` | RMS of `r - 1` over the last `nSettle` frames |
| `jitter` | trial-to-trial std of `r` at the frame nearest `t50` |
| `delayErrMean`, `delayErrStd` | achieved minus requested delay [us] |
| `lateFrac` | fraction of trials whose poke deadline had already passed when the frame was received |

It also combines all delays on the time-since-command axis into a super-sampled response, binned at
`frame period / resampleFactor`, and reports the best delay by the chosen criterion.

## Poke modes

- `actuator` (default): exactly one actuator `(x, y)`; the command is `+/- amp` at that actuator.
- `pattern`: a 2-D FITS file on the RTC with the same size as the DM channel; the command is `+/- amp * pattern`
  (not normalized, so `amp = 1` applies the file as stored).  The file is re-read at each run start, and a copy is
  saved with the results.

In both modes the largest absolute command must not exceed `poke.maxCommand` (default 1).

## Timing notes

- The trigger, delay busy-wait, and poke run on the shmimMonitor thread.  Set `wfscam.threadPrio` and
  `wfscam.cpuset` appropriately on the RTC for accurate delays.
- Delays shorter than the time it takes the app to receive the frame are *late*: the poke is written immediately and
  the achieved delay is recorded.  If more than `maxLateFrac` of the trials at a delay are late, a warning is logged
  with the minimum achievable delay and the cube is flagged (`LATEFLAG = 1`).
- Delays longer than one frame period are supported: the poke is held until the frame before the deadline.
- The measured response includes the full DM path (dmcomb channel sum, driver, and mechanics).
- The app requires `atime` to be `CLOCK_REALTIME`, and refuses to start if the most recent frame's `atime` is more than
  1 s from the current time.
- A run aborts if camWFS changes fps or is re-allocated, if a trial times out, or if frame-counter gaps exceed
  `maxRetries` at one delay.  The DM channel is always returned to zero.

# CONFIGURATION

| Key | Type | Default | Description |
|---|---|---|---|
| `wfscam.shmimName` | string | `camwfs` | camWFS stream |
| `wfscam.camDevName` | string | = shmimName | INDI device used to read `fps` |
| `wfscam.threadPrio` | int | 2 | real-time priority of the trigger/poke thread |
| `wfscam.cpuset` | string | | cpuset of the trigger/poke thread |
| `dm.index` | int | 0 | DM number `NN` in `dm<NN>disp<MM>`: 0 woofer, 1 tweeter, 2 NCPC |
| `dm.channel` | int | 7 | dmcomb channel `MM` in `dm<NN>disp<MM>` |
| `poke.mode` | string | `actuator` | `actuator` or `pattern` |
| `poke.x`, `poke.y` | vector<int> | | the actuator, exactly one entry each |
| `poke.patternFile` | string | | FITS pattern path (pattern mode) |
| `poke.amp` | float | 0 | poke amplitude; must be non-zero |
| `poke.maxCommand` | float | 1 | largest absolute command allowed |
| `poke.nDelays` | int | 10 | K |
| `poke.delaySpan` | float | 0 | span of the delay grid [us]; `<= 0` means one camWFS frame period |
| `poke.nFrames` | int | 20 | N, frames per trial |
| `poke.nTrials` | int | 20 | M, trials per delay, must be even |
| `poke.settle` | float | 0.05 | seconds between zeroing the DM and the next trial |
| `poke.trialTimeout` | float | 2 | seconds to wait for one trial |
| `poke.maxRetries` | int | 5 | invalid-trial retries per delay |
| `analysis.nRef` | int | 10 | reference-pass +/- pairs |
| `analysis.nSettle` | int | 5 | trailing frames for `P` and `settleErr` |
| `analysis.maskThresh` | float | 0.1 | pixel mask threshold, fraction of `max|P|` |
| `analysis.resampleFactor` | int | 10 | super-sampling factor |
| `analysis.bestMetric` | string | `jitter` | `jitter`, `rise`, or `t50` |
| `analysis.maxLateFrac` | float | 0.1 | late-poke warning threshold |
| `output.baseDir` | string | `/home/xsup/dm_response` | root of the output tree |

Example `dmTemporalResponse.conf`:

```
[wfscam]
shmimName=camwfs
threadPrio=60

[dm]
index=0
channel=7

[poke]
mode=actuator
x=5
y=5
amp=0.05
nDelays=10
nFrames=20
nTrials=20
```

# INDI

Tunable properties (`current`/`target`), all rejected while a run is in progress:
`dm_index`, `dm_channel`, `poke_mode` (selection: `actuator`/`pattern`), `poke_x`, `poke_y`, `pattern_file`,
`poke_amp`, `nDelays`, `delaySpan`, `nFrames`, `nTrials`, `settle`, `nSettle`, `maskThresh`, `bestMetric`.

Controls: `start.request`, `stop.request`.

Read-only:

| Property | Elements |
|---|---|
| `dm_stream` | `name`: the resolved stream, e.g. `dm00disp07` |
| `run_state` | `status` (idle/running/done/error/stopped), `phase` (none/reference/measuring/analyzing) |
| `progress` | `delay_index`, `delay_us`, `sign`, `trial`, `n_invalid`, `late_frac` |
| `delays` | `values`: the delay grid, comma separated |
| `results` | `t50`, `rise`, `jitter`: per-delay values, comma separated |
| `best` | `delay_us`, `t50`, `rise`, `jitter` |
| `pattern_info` | `info`: loaded pattern path, size, and SHA-256, or the validation error |
| `output` | `dir`: the run directory |

# OUTPUT

Each run writes to `<output.baseDir>/<YYYY-MM-DDTHHMMSS>/` (UTC):

| File | Contents |
|---|---|
| `dmresp_delay_<DDDDD>us.fits` | one per delay: `[nx, ny, N]` averaged +/- half-difference cube |
| `reference.fits` | `[nx, ny, 2]`: plane 0 `P`, plane 1 the mask |
| `summary_curves.fits` | `[N+1, K, 3]`: mean response, trial std, and mean time from the command [us]; row 0 is the pre-poke baseline (r = 0) |
| `summary_metrics.fits` | `[K, 9]`: delay, t50, rise, overshoot, settleErr, jitter, delayErrMean, delayErrStd, lateFrac (columns named by `MCOLn`) |
| `summary_superres.fits` | `[nBins, 2]`: bin time [us] and binned response |
| `pattern.fits` | pattern mode only: copy of the applied pattern |

Every file carries the run header: `DATE-OBS`, `DMSTREAM`, `WFSSHMIM`, `WFSFPS`, `POKEMODE`, `POKEX`/`POKEY` or
`PATFILE`/`PATSHA`, `POKEAMP`, `NDELAYS`, `DLYSPAN`, `NFRAMES`, `NTRIALS`, `NREF`.  The cubes also have `DELAYUS`,
`DLYIDX`, `DLYMEAN`, `DLYSTD`, `DLYMIN`, `DLYMAX`, `NINVALID`, `LATEFRAC`, `LATEFLAG`, `T50`, `RISE`, and `JITTER`.
A header value that can not be computed (e.g. no 10%/90% crossing) is written as `-999`.

Live shared memory, updated during a run: `<name>_ref` (P), `<name>_resp` (mean response curves, (N+1) x K), and
`<name>_respavg` (super-sampled response).

# SAFETY

- Use a DM channel that no other process writes (default `dm00disp07`), with the AO loop open.
- Start with a small `poke_amp`.  `poke.maxCommand` limits the command in both modes.
- The DM channel is zeroed after every trial and whenever the run ends, including on stop, error, and shutdown.
